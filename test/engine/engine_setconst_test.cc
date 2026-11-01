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

// Tests for engine/engine_setconst.c.

#include <array>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_setconst.h"
#include "test/fixture.h"

// accesses are tracked with page protection where the kind of a faulting access
// can be read from its context, outside sanitizers, which handle memory faults
// themselves
#if defined(__has_feature)
  #if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || \
      __has_feature(thread_sanitizer)
    #define MJ_SANITIZED
  #endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  #define MJ_SANITIZED
#endif
#if ((defined(__APPLE__) && (defined(__aarch64__) || defined(__x86_64__))) || \
     (defined(__linux__) && defined(__x86_64__))) &&                          \
    !defined(MJ_SANITIZED)
  #define MJ_TRACK_ACCESS
  #include <signal.h>
  #include <sys/mman.h>
  #include <sys/ucontext.h>
  #include <unistd.h>
#endif

namespace mujoco {
namespace {

using ::std::string;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

using SetConstTest = MujocoTest;

#ifdef MJ_TRACK_ACCESS
// an mjModel array copied into pages of its own, and how a group of mj_setConst
// computations first accessed it
struct TrackedArray {
  const char* name;
  char* begin;
  size_t len;
  bool read;     // read before it was written
  bool written;  // written
};

// arrays whose pages are protected, and the previous fault handlers
std::vector<TrackedArray>* tracked = nullptr;
struct sigaction prev_segv = {}, prev_bus = {};

// whether the faulting access was a write (on x86-64, a read-modify-write
// instruction faulting on a PROT_NONE page also sets the W bit)
bool IsWrite(void* context) {
  ucontext_t* uc = static_cast<ucontext_t*>(context);
  #if defined(__APPLE__) && defined(__aarch64__)
  return uc->uc_mcontext->__es.__esr &
         (1u << 6);  // WnR bit of the data abort syndrome
  #elif defined(__APPLE__)
  return uc->uc_mcontext->__es.__err & 2;  // W bit of the page fault error code
  #else
  return uc->uc_mcontext.gregs[REG_ERR] & 2;
  #endif
}

// record the first read or write of a tracked array and let it proceed: a read
// leaves its pages readable, so that a later write is recorded too; delegate
// any other fault to the previous handler
void OnAccess(int sig, siginfo_t* info, void* context) {
  char* address = static_cast<char*>(info->si_addr);
  for (TrackedArray& a : *tracked) {
    if (address >= a.begin && address < a.begin + a.len) {
      if (IsWrite(context)) {
        a.written = true;
        mprotect(a.begin, a.len, PROT_READ | PROT_WRITE);
      } else {
        a.read = true;
        mprotect(a.begin, a.len, PROT_READ);
      }
      return;
    }
  }
  sigaction(sig, sig == SIGSEGV ? &prev_segv : &prev_bus, nullptr);
}

// how each group of mj_setConst computations accesses an array, in the order
// of the columns of MJSETCONST_ARRAYS
using Access = std::array<int, 4>;
constexpr int kGroups[] = {mjSCGROUP_FIXED, mjSCGROUP_KIN0, mjSCGROUP_INERT0,
                           mjSCGROUP_SPRING};

// how each group of mj_setConst computations, run alone, accesses each array of
// a copy of m: mjSC_IN if it reads the array before writing it, mjSC_OUT if it
// writes it; the groups run in turn on the same copy, as each rewrites what it
// writes with the values the next would find in m, and on reset data
std::map<std::string, Access> SetConstAccess(const mjModel* m, mjData* d) {
  size_t page = sysconf(_SC_PAGESIZE);
  mjModel copy = *m;
  std::vector<TrackedArray> arrays;
  #if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
  #endif
  MJMODEL_POINTERS_PREAMBLE(m)
  #define X(type, name, nr, nc)                                                \
    if (size_t bytes = sizeof(type) * m->nr * (nc)) {                          \
      size_t len = (bytes + page - 1) / page * page;                           \
      void* pages = mmap(nullptr, len, PROT_READ | PROT_WRITE,                 \
                         MAP_PRIVATE | MAP_ANON, -1, 0);                       \
      if (pages == MAP_FAILED) mju_error("mmap failed for %s", #name);         \
      std::memcpy(pages, m->name, bytes);                                      \
      copy.name = static_cast<type*>(pages);                                   \
      arrays.push_back({#name, static_cast<char*>(pages), len, false, false}); \
    }
  MJMODEL_POINTERS
  #undef X
  #if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
  #endif

  struct sigaction action = {};
  action.sa_sigaction = OnAccess;
  action.sa_flags = SA_SIGINFO;
  tracked = &arrays;
  sigaction(SIGSEGV, &action, &prev_segv);
  sigaction(SIGBUS, &action, &prev_bus);
  std::map<std::string, Access> access;
  for (int g = 0; g < 4; g++) {
    // each group runs on fresh data, as it would alone (from m, untracked)
    mj_resetData(m, d);
    for (TrackedArray& a : arrays) {
      a.read = a.written = false;
      mprotect(a.begin, a.len, PROT_NONE);
    }
    mj_setConstGroups(&copy, d, kGroups[g]);
    for (TrackedArray& a : arrays) {
      access[a.name][g] = (a.read ? mjSC_IN : 0) | (a.written ? mjSC_OUT : 0);
    }
  }
  for (TrackedArray& a : arrays) munmap(a.begin, a.len);
  sigaction(SIGSEGV, &prev_segv, nullptr);
  sigaction(SIGBUS, &prev_bus, nullptr);
  tracked = nullptr;
  return access;
}

// models of the repository on which every mark of MJSETCONST_ARRAYS appears,
// found by covering the marks seen over all of them; a change to mj_setConst
// which adds a mark may need another model here
std::vector<std::string> SetConstModels() {
  std::vector<std::string> models;
  for (const char* xml : {"flex/bag.xml", "flex/mechanisms/cantilever.xml",
                          "flex/mechanisms/dipper.xml", "flex/poncho.xml",
                          "slider_crank/slider_crank.xml"}) {
    models.push_back(GetModelPath(xml));
  }
  for (const char* xml :
       {"engine/testdata/core_constraint/dofless_contact.xml",
        "engine/testdata/derivative/dcmotor.xml",
        "engine/testdata/flex_setconst.xml",
        "engine/testdata/sleep/init_island.xml", "testdata/model.xml",
        "user/testdata/flex_line_obj.xml", "user/testdata/mesh.xml"}) {
    models.push_back(GetTestDataFilePath(xml));
  }
  return models;
}

// clear the scalars mj_setConst writes and the pointers, so that what remains
// of m is its other scalars
void MaskScalars(mjModel* m) {
  #define X(name) std::memset(&m->name, 0, sizeof(m->name));
  MJSETCONST_SCALARS
  #undef X
  #define X(type, name, nr, nc) m->name = nullptr;
  MJMODEL_POINTERS
  #undef X
  m->buffer = nullptr;
}
#endif

// MJSETCONST_ARRAYS and MJSETCONST_SCALARS are what mj_setConst does: over
// the models of SetConstModels, each group of its computations, run alone,
// reads the arrays marked mjSC_IN in its column before writing them, and writes
// those marked mjSC_OUT, and no others; mj_setConst writes the listed scalars,
// and no other scalar changes.
TEST_F(SetConstTest, AccessMatchesTable) {
#ifndef MJ_TRACK_ACCESS
  GTEST_SKIP() << "accesses are tracked with page protection, on Linux x86-64 "
                  "and macOS";
#else
  std::map<std::string, Access> declared = {
  #define X(type, name, nr, nc) {#name, Access{}},
      MJMODEL_POINTERS
  #undef X
  };
  #define X(name, fixed, kin0, inert0, spring) \
    declared[#name] = {fixed, kin0, inert0, spring};
  MJSETCONST_ARRAYS
  #undef X
  std::map<std::string, Access> observed;
  mjModel poison;
  std::memset(&poison, 0xff, sizeof(mjModel));
  for (const std::string& xml : SetConstModels()) {
    char error[1024];
    mjModel* m = mj_loadXML(xml.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(m, NotNull()) << xml << ": " << error;
    mjData* d = mj_makeData(m);
    for (const auto& [name, access] : SetConstAccess(m, d)) {
      for (int g = 0; g < 4; g++) {
        observed[name][g] |= access[g];
      }
    }

    // scalars, as compiled and with masses, sizes and gravity compensation
    // changed
    for (bool perturbed : {false, true}) {
      if (perturbed) {
        for (int i = 1; i < m->nbody; i++) {
          m->body_mass[i] *= 2;
          m->body_gravcomp[i] = 0.5 - m->body_gravcomp[i];
        }
        for (int i = 0; i < 3 * m->ngeom; i++) m->geom_size[i] *= 1.25;
      }
      mjModel* copy = mj_copyModel(nullptr, m);
      mjModel before = *copy;
      // poison and compare raw bytes: 0xff is not a valid bool
  #define X(name) std::memset(&copy->name, 0xff, sizeof(copy->name));
      MJSETCONST_SCALARS
  #undef X
      mj_setConst(copy, d);
  #define X(name)                                                            \
    EXPECT_NE(std::memcmp(&copy->name, &poison.name, sizeof(copy->name)), 0) \
        << #name << " not written in " << xml;
      MJSETCONST_SCALARS
  #undef X
      mjModel after = *copy;
      MaskScalars(&before);
      MaskScalars(&after);
      EXPECT_EQ(std::memcmp(&before, &after, sizeof(mjModel)), 0)
          << "a scalar which is not listed changed in " << xml;
      mj_deleteModel(copy);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }

  for (const auto& [name, mark] : declared) {
    EXPECT_EQ(observed[name], mark) << name;
  }
#endif
}

TEST_F(SetConstTest, AwakeActuatedJoint) {
  string xml = R"(
  <mujoco>
    <worldbody>
      <body name="B1" sleep="POLICY1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="B2" sleep="POLICY2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="J1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m;

  string sleep[] = {"auto", "never", "allowed", "init"};
  int tsp0[] = {mjSLEEP_AUTO_NEVER, mjSLEEP_NEVER, mjSLEEP_ALLOWED,
                mjSLEEP_INIT};
  int tsp1[] = {mjSLEEP_AUTO_ALLOWED, mjSLEEP_NEVER, mjSLEEP_ALLOWED,
                mjSLEEP_INIT};

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      string xml_copy = xml;
      size_t pos1 = xml_copy.find("POLICY1");
      xml_copy.replace(pos1, 7, sleep[i]);
      size_t pos2 = xml_copy.find("POLICY2");
      xml_copy.replace(pos2, 7, sleep[j]);
      m = LoadModelFromString(xml_copy.c_str(), error, sizeof(error));
      ASSERT_THAT(m.get(), NotNull()) << error;
      EXPECT_EQ(m->tree_sleep_policy[0], tsp0[i]);
      EXPECT_EQ(m->tree_sleep_policy[1], tsp1[j]);
    }
  }
}

TEST_F(SetConstTest, AwakeActuatedSite) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <general site="S1" gear="1 0 0 0 0 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_ALLOWED);
}

TEST_F(SetConstTest, AwakeActuatedMultiSite) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
        <site name="S2"/>
      </body>
    </worldbody>
    <actuator>
      <general site="S1" refsite="S2" gear="1 0 0 0 0 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_NEVER);
}

TEST_F(SetConstTest, AwakeActuatedSliderCrank) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
        <site name="S2"/>
      </body>
    </worldbody>
    <actuator>
      <general cranksite="S1" slidersite="S2" cranklength="0.5"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_NEVER);
}

TEST_F(SetConstTest, AwakeActuatedSO3Refsite) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="ball"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="ball"/>
        <geom size=".1"/>
        <site name="S2"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity site="S1" refsite="S2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_NEVER);
}

TEST_F(SetConstTest, AwakeActuatedBody) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <adhesion body="B1" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_ALLOWED);
}

TEST_F(SetConstTest, AwakeActuatedTendon) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="S1"/>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S2"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="T1">
        <site site="S1"/>
        <site site="S2"/>
      </spatial>
    </tendon>
    <actuator>
      <motor tendon="T1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_ALLOWED);
}

TEST_F(SetConstTest, AwakeStiffTendonMultiTree) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
        <site name="S2"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="T1" stiffness="1">
        <site site="S1"/>
        <site site="S2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_NEVER);
}

TEST_F(SetConstTest, SleepyTendonSingleTree) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
        <body name="B2">
          <joint name="J2" type="slide"/>
          <geom size=".1"/>
          <site name="S2"/>
        </body>
      </body>
    </worldbody>
    <tendon>
      <spatial name="T1" stiffness="1">
        <site site="S1"/>
        <site site="S2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_ALLOWED);
}

TEST_F(SetConstTest, SleepyTendonZeroStiffness) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="J1" type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
      <body name="B2">
        <joint name="J2" type="slide"/>
        <geom size=".1"/>
        <site name="S2"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="T1" stiffness="0" damping="0">
        <site site="S1"/>
        <site site="S2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_EQ(model->tree_sleep_policy[0], mjSLEEP_AUTO_ALLOWED);
  EXPECT_EQ(model->tree_sleep_policy[1], mjSLEEP_AUTO_ALLOWED);
}

TEST_F(SetConstTest, TendonTreeId) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <!-- Static body 1: world -->
      <site name="S1"/>

      <!-- Static body 2: child of world -->
      <body name="B_static">
        <site name="S2"/>
      </body>

      <!-- Tree 1 -->
      <body name="B1_1">
        <joint name="J1_1" type="slide"/>
        <geom size=".1"/>
        <site name="S3"/>
        <body name="B1_2">
          <joint name="J1_2" type="slide"/>
          <geom size=".1"/>
          <site name="S4"/>
        </body>
      </body>

      <!-- Tree 2 -->
      <body name="B2_1">
        <joint name="J2_1" type="slide"/>
        <geom size=".1"/>
        <site name="S5"/>
      </body>

      <!-- Tree 3 -->
      <body name="B3_1">
        <joint name="J3_1" type="slide"/>
        <geom size=".1"/>
        <site name="S6"/>
      </body>
    </worldbody>

    <tendon>
      <!-- Tendon 1: Between static bodies -->
      <spatial name="T_static">
        <site site="S1"/>
        <site site="S2"/>
      </spatial>

      <!-- Tendon 2: Within Tree 1 -->
      <spatial name="T_tree1">
        <site site="S3"/>
        <site site="S4"/>
      </spatial>

      <!-- Tendon 3: Between Tree 1 and Tree 2 -->
      <spatial name="T_intertree12">
        <site site="S4"/>
        <site site="S5"/>
      </spatial>

      <!-- Tendon 4: Between Tree 1, 2 and 3 -->
      <spatial name="T_intertree123">
        <site site="S4"/>
        <site site="S5"/>
        <site site="S6"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  int t_static_id = mj_name2id(model.get(), mjOBJ_TENDON, "T_static");
  int t_tree1_id = mj_name2id(model.get(), mjOBJ_TENDON, "T_tree1");
  int t_intertree12_id = mj_name2id(model.get(), mjOBJ_TENDON, "T_intertree12");
  int t_intertree123_id =
      mj_name2id(model.get(), mjOBJ_TENDON, "T_intertree123");
  int b1_1_treeid =
      model->body_treeid[mj_name2id(model.get(), mjOBJ_BODY, "B1_1")];
  int b2_1_treeid =
      model->body_treeid[mj_name2id(model.get(), mjOBJ_BODY, "B2_1")];

  // Tendon 1: Not associated with any tree
  EXPECT_EQ(model->tendon_treenum[t_static_id], 0);
  EXPECT_EQ(model->tendon_treeid[2 * t_static_id], -1);
  EXPECT_EQ(model->tendon_treeid[2 * t_static_id + 1], -1);

  // Tendon 2: Should be in Tree 1
  EXPECT_EQ(model->tendon_treenum[t_tree1_id], 1);
  EXPECT_EQ(model->tendon_treeid[2 * t_tree1_id], b1_1_treeid);
  EXPECT_EQ(model->tendon_treeid[2 * t_tree1_id + 1], -1);
  EXPECT_GE(model->tendon_treeid[2 * t_tree1_id], 0);

  // Tendon 3: Spans two trees (Tree 1 and Tree 2)
  EXPECT_EQ(model->tendon_treenum[t_intertree12_id], 2);
  EXPECT_EQ(model->tendon_treeid[2 * t_intertree12_id], b1_1_treeid);
  EXPECT_EQ(model->tendon_treeid[2 * t_intertree12_id + 1], b2_1_treeid);

  // Tendon 4: Spans three trees (Tree 1, 2 and 3)
  EXPECT_EQ(model->tendon_treenum[t_intertree123_id], 3);
  EXPECT_EQ(model->tendon_treeid[2 * t_intertree123_id], b1_1_treeid);
  EXPECT_EQ(model->tendon_treeid[2 * t_intertree123_id + 1], b2_1_treeid);
  // The third tree ID is not stored in tendon_treeid
}

TEST_F(SetConstTest, SleepingNotAllowed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>

      <!-- Tree 0 -->
      <body name="B1_1">
        <joint name="J1_1" type="slide"/>
        <geom size=".1"/>
        <site name="S3"/>
        <body name="B1_2">
          <joint name="J1_2" type="slide"/>
          <geom size=".1"/>
          <site name="S4"/>
        </body>
      </body>

      <!-- Tree 1: forbidden user sleep override -->
      <body name="B2_1" sleep="allowed">
        <joint name="J2_1" type="slide"/>
        <geom size=".1"/>
        <site name="S5"/>
      </body>

      <!-- Tree 2 -->
      <body name="B3_1">
        <joint name="J3_1" type="slide"/>
        <geom size=".1"/>
        <site name="S6"/>
      </body>
    </worldbody>

    <tendon>
      <!-- Tendon 0: Between Tree 0, 1 and 2 -->
      <spatial name="T_intertree123">
        <site site="S4"/>
        <site site="S5"/>
        <site site="S6"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(model.get(), IsNull()) << error;
  EXPECT_THAT(
      string(error),
      HasSubstr("tree 1 connected to tendon 0 which spans more than 2 trees, "
                "sleeping not allowed"));
}

TEST_F(SetConstTest, DofLength) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint name="S1" type="slide"/>
        <geom size="2"/>
      </body>
      <body name="B2">
        <joint name="H1" type="hinge"/>
        <geom size="3"/>
      </body>
      <body name="B3">
        <joint name="BA1" type="ball"/>
        <geom size="4"/>
      </body>
      <body name="B4">
        <joint name="F1" type="free"/>
        <geom size="5"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  mjtNum tol = 1e-5;

  // B1: Slider
  EXPECT_EQ(model->dof_length[0], 1);

  // B2: Hinge
  EXPECT_NEAR(model->dof_length[1], 3, tol);

  // B3: Ball
  EXPECT_NEAR(model->dof_length[2], 4, tol);
  EXPECT_NEAR(model->dof_length[3], 4, tol);
  EXPECT_NEAR(model->dof_length[4], 4, tol);

  // B4: Free
  EXPECT_EQ(model->dof_length[5], 1);
  EXPECT_EQ(model->dof_length[6], 1);
  EXPECT_EQ(model->dof_length[7], 1);
  EXPECT_NEAR(model->dof_length[8], 5, tol);
  EXPECT_NEAR(model->dof_length[9], 5, tol);
  EXPECT_NEAR(model->dof_length[10], 5, tol);
}

TEST_F(SetConstTest, BodySameframeRecomputed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1" simple="false">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  int b = mj_name2id(m.get(), mjOBJ_BODY, "B1");

  // initially sameframe should be BODY (ipos=0, iquat=identity)
  EXPECT_EQ(m->body_sameframe[b], mjSAMEFRAME_BODY);

  // perturb body_ipos, call mj_setConst
  m->body_ipos[3 * b + 0] = 1.0;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->body_sameframe[b], mjSAMEFRAME_BODYROT);

  // also perturb body_iquat
  m->body_iquat[4 * b + 0] = 0.5;
  m->body_iquat[4 * b + 1] = 0.5;
  m->body_iquat[4 * b + 2] = 0.5;
  m->body_iquat[4 * b + 3] = 0.5;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->body_sameframe[b], mjSAMEFRAME_NONE);

  // restore to identity, should go back to BODY
  m->body_ipos[3 * b + 0] = 0;
  m->body_iquat[4 * b + 0] = 1;
  m->body_iquat[4 * b + 1] = 0;
  m->body_iquat[4 * b + 2] = 0;
  m->body_iquat[4 * b + 3] = 0;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->body_sameframe[b], mjSAMEFRAME_BODY);
}

TEST_F(SetConstTest, GeomSameframeRecomputed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint type="slide"/>
        <geom name="G1" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  int g = mj_name2id(m.get(), mjOBJ_GEOM, "G1");

  // initially sameframe should be BODY
  EXPECT_EQ(m->geom_sameframe[g], mjSAMEFRAME_BODY);

  // perturb geom_pos
  m->geom_pos[3 * g + 1] = 0.5;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->geom_sameframe[g], mjSAMEFRAME_BODYROT);

  // restore, should go back to BODY
  m->geom_pos[3 * g + 1] = 0;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->geom_sameframe[g], mjSAMEFRAME_BODY);
}

TEST_F(SetConstTest, SiteSameframeRecomputed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint type="slide"/>
        <geom size=".1"/>
        <site name="S1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  int s = mj_name2id(m.get(), mjOBJ_SITE, "S1");

  // initially sameframe should be BODY
  EXPECT_EQ(m->site_sameframe[s], mjSAMEFRAME_BODY);

  // perturb site_pos
  m->site_pos[3 * s + 2] = 0.3;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->site_sameframe[s], mjSAMEFRAME_BODYROT);

  // restore
  m->site_pos[3 * s + 2] = 0;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->site_sameframe[s], mjSAMEFRAME_BODY);
}

TEST_F(SetConstTest, SameframeKinematicsCorrect) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1" pos="1 0 0" simple="false">
        <joint type="slide" axis="1 0 0"/>
        <geom name="G1" size=".1"/>
        <site name="S1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  int b = mj_name2id(m.get(), mjOBJ_BODY, "B1");
  int g = mj_name2id(m.get(), mjOBJ_GEOM, "G1");

  // perturb body inertial offset, breaking sameframe
  m->body_ipos[3 * b + 1] = 0.5;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->body_sameframe[b], mjSAMEFRAME_BODYROT);

  // run forward kinematics, check that xipos != xpos
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(d->xipos[3 * b + 1], 0.5, MjTol(1e-10, 1e-6));
  EXPECT_NEAR(d->xpos[3 * b + 1], 0.0, MjTol(1e-10, 1e-6));

  // perturb geom_pos, check geom global position
  m->geom_pos[3 * g + 2] = 0.3;
  mj_setConst(m.get(), d.get());
  EXPECT_NE(m->geom_sameframe[g], mjSAMEFRAME_BODY);
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(d->geom_xpos[3 * g + 2], 0.3, MjTol(1e-10, 1e-6));
}

TEST_F(SetConstTest, SimpleBodyLostSameframeError) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  int b = mj_name2id(m.get(), mjOBJ_BODY, "B1");

  // confirm body is compiled as simple
  EXPECT_GT(m->body_simple[b], 0);

  // perturb body_ipos, breaking sameframe; calling mj_setConst should fail
  m->body_ipos[3 * b + 0] = 1.0;

  std::string err = MjuErrorMessageFrom(mj_setConst)(m.get(), d.get());
  EXPECT_THAT(err, HasSubstr("body 1 is compiled as simple but "
                             "sameframe no longer holds"));
}

TEST_F(SetConstTest, DampRatioInertia) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="J1" type="slide"/>
        <geom size=".1" mass="1"/>
      </body>
      <body>
        <joint name="J2" type="slide"/>
        <geom size=".1" mass="1"/>
      </body>
      <body>
        <joint name="B" type="ball"/>
        <inertial pos="0 0 0" mass="1" diaginertia="2 4 4"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="T1" armature="3">
        <joint joint="J1" coef="1"/>
        <joint joint="J2" coef="1e-6"/>
      </fixed>
    </tendon>
    <actuator>
      <position name="tendon" tendon="T1" kp="9" dampratio="1"/>
      <orientation name="orient" joint="B" kp="3" dampratio="1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  // T1: mass = 1 + 3 (tendon armature) = 4, tiny J2 coef does not blow up
  // damping = 2 * sqrt(9 * 4) = 12
  EXPECT_NEAR(m->actuator_biasprm[0 * mjNBIAS + 2], -12, MjTol(1e-6, 1e-4));

  // orient: average invweight = (1/2 + 1/4 + 1/4) / 3 = 1/3, mass = 3
  // damping = 2 * sqrt(3 * 3) = 6
  EXPECT_NEAR(m->actuator_biasprm[1 * mjNBIAS + 2], -6, MjTol(1e-10, 1e-6));
}

// The constant bending factor of M + K_bend is consumed only by bending-only
// flexes: with stretching present the per-step factor replaces it, so a
// singular M + K_bend must not be an error.
TEST_F(SetConstTest, RankDeficientBendingFactor) {
  constexpr char xml[] = R"(
  <mujoco>
    <option solver="CG" integrator="discrete"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.05 0.05 0.05"
                radius=".005" dim="2" mass="0.5" dof="full">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.2" elastic2d="ELASTIC2D"
                    thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  for (const char* elastic2d : {"both", "bend"}) {
    std::string model_xml(xml);
    model_xml.replace(model_xml.find("ELASTIC2D"), 9, elastic2d);
    char error[1024];
    MjModelPtr m = LoadModelFromString(model_xml.c_str(), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    ASSERT_GT(m->nefm0dof, 0);
    MjDataPtr d(mj_makeData(m.get()));

    // vanishing vertex masses make M + K_bend singular (rigid motions)
    for (int b = 1; b < m->nbody; b++) {
      m->body_mass[b] = 1e-20;
      m->body_inertia[3 * b + 0] = 1e-20;
      m->body_inertia[3 * b + 1] = 1e-20;
      m->body_inertia[3 * b + 2] = 1e-20;
    }

    std::string err = MjuErrorMessageFrom(mj_setConst)(m.get(), d.get());
    if (std::string(elastic2d) == "both") {
      EXPECT_EQ(err, "");
    } else {
      EXPECT_THAT(err, HasSubstr("constant metric factor is rank-deficient"));
    }
  }
}

// The bounds of a geom follow its size, so that its contacts are found.
TEST_F(SetConstTest, GeomSizeBounds) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .5">
        <freejoint/>
        <geom name="ball" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  int g = mj_name2id(m.get(), mjOBJ_GEOM, "ball");

  // the grown ball reaches the plane
  m->geom_size[3 * g] = 0.75;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->geom_rbound[g], 0.75);
  EXPECT_EQ(m->geom_aabb[6 * g + 3], 0.75);
  mj_resetData(m.get(), d.get());
  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->ncon, 1);
}

// The bounds of a height field geom follow the size of the height field.
TEST_F(SetConstTest, HeightFieldSizeBounds) {
  constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="terrain" nrow="2" ncol="2" size="1 1 .5 .1"/>
    </asset>
    <worldbody>
      <geom name="terrain" type="hfield" hfield="terrain"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  // radius sqrt(2^2 + 2^2 + 1^2), box from depth -0.2 to height 1
  for (int k = 0; k < 4; k++) m->hfield_size[k] *= 2;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->geom_rbound[0], 3);
  EXPECT_THAT(AsVector(m->geom_aabb, 6),
              ElementsAre(0, 0, 0.4, 2, 2, MjNear(0.6, 1e-15, 1e-7)));
}

// The bounding volume hierarchy of a body is in its inertial frame and follows
// it, so that midphase finds the contacts it would find without it.
TEST_F(SetConstTest, InertialFrameBoundingVolumes) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom size=".1" pos="-.3 0 0"/>
        <geom size=".1" pos=".3 0 0"/>
      </body>
      <body name="moved" pos=".5 0 0">
        <freejoint/>
        <geom size=".1" pos="-.15 0 0"/>
        <geom size=".1" pos=".3 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  int b = mj_name2id(m.get(), mjOBJ_BODY, "moved");

  // move the center of mass away from the geoms
  m->body_ipos[3 * b] = 5;
  mj_setConst(m.get(), d.get());
  for (bool midphase : {true, false}) {
    m->opt.disableflags = midphase ? 0 : mjDSBL_MIDPHASE;
    mj_resetData(m.get(), d.get());
    mj_forward(m.get(), d.get());
    EXPECT_EQ(d->ncon, 1) << "midphase: " << midphase;
  }
}

// The collision masks and margin of a body accumulate those of its geoms.
TEST_F(SetConstTest, BodyMasksAndMargin) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <freejoint/>
        <geom size=".1" margin=".25"/>
        <geom size=".1" margin=".5" gap=".125" contype="2" conaffinity="4"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  int b = mj_name2id(m.get(), mjOBJ_BODY, "body");
  EXPECT_EQ(m->body_contype[b], 3);
  EXPECT_EQ(m->body_conaffinity[b], 5);
  EXPECT_EQ(m->body_margin[b], 0.625);

  m->geom_margin[0] = 1;
  m->geom_contype[0] = m->geom_conaffinity[0] = 0;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->body_contype[b], 2);
  EXPECT_EQ(m->body_conaffinity[b], 4);
  EXPECT_EQ(m->body_margin[b], 1);
}

// A geom which collides must be in the bounding volume hierarchy of its body.
TEST_F(SetConstTest, CollidingGeomOutsideBoundingVolumes) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom size=".1"/>
        <geom size=".1" contype="0" conaffinity="0"/>
      </body>
      <body>
        <freejoint/>
        <geom size=".1" contype="0" conaffinity="0"/>
        <geom size=".1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  // a body without a hierarchy collides all of its geoms
  m->geom_contype[2] = 1;
  EXPECT_EQ(MjuErrorMessageFrom(mj_setConst)(m.get(), d.get()), "");

  // a body with a hierarchy does not
  m->geom_contype[1] = 1;
  EXPECT_THAT(MjuErrorMessageFrom(mj_setConst)(m.get(), d.get()),
              HasSubstr("geom 1 collides"));
}

// The added mass and inertia of a geom in the ellipsoid fluid model follow its
// size.
TEST_F(SetConstTest, FluidAddedMass) {
  constexpr char xml[] = R"(
  <mujoco>
    <option density="1000"/>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="ellipsoid" size="SIZE" fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  std::string small(xml), large(xml);
  small.replace(small.find("SIZE"), 4, ".1 .2 .3");
  large.replace(large.find("SIZE"), 4, ".2 .4 .6");
  MjModelPtr m = LoadModelFromString(small.c_str(), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjModelPtr expected =
      LoadModelFromString(large.c_str(), error, sizeof(error));
  ASSERT_THAT(expected.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  for (int k = 0; k < 3; k++) m->geom_size[k] *= 2;
  mj_setConst(m.get(), d.get());
  EXPECT_THAT(AsVector(m->geom_fluid, mjNFLUID),
              ElementsAreArray(AsVector(expected->geom_fluid, mjNFLUID)));
}

// The automatic sleep policy of a tree is resolved anew.
TEST_F(SetConstTest, SleepPolicyResolvedAnew) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom size=".1"/>
        <site name="s1"/>
      </body>
      <body pos="1 0 0">
        <freejoint/>
        <geom size=".1"/>
        <site name="s2"/>
      </body>
    </worldbody>
    <tendon>
      <spatial stiffness="1">
        <site site="s1"/>
        <site site="s2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  EXPECT_EQ(m->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(m->tree_sleep_policy[1], mjSLEEP_AUTO_NEVER);

  m->tendon_stiffness[0] = 0;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->tree_sleep_policy[0], mjSLEEP_AUTO_ALLOWED);
  EXPECT_EQ(m->tree_sleep_policy[1], mjSLEEP_AUTO_ALLOWED);

  m->tendon_stiffness[0] = 1;
  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_EQ(m->tree_sleep_policy[1], mjSLEEP_AUTO_NEVER);
}

// Constants are computed for sleeping trees as for awake ones, and the trees,
// whose cached kinematics are overwritten, are woken.
TEST_F(SetConstTest, SleepingTrees) {
  constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <body sleep="init">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".1" fromto="0 0 0 1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  ASSERT_EQ(d->ntree_awake, 0);
  mjtNum acc0 = m->actuator_acc0[0];
  mjtNum invweight0 = m->dof_invweight0[0];
  EXPECT_GT(acc0, 0);

  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->actuator_acc0[0], acc0);
  EXPECT_EQ(m->dof_invweight0[0], invweight0);
  EXPECT_EQ(m->opt.enableflags, mjENBL_SLEEP);
  EXPECT_EQ(d->ntree_awake, m->ntree);
  EXPECT_EQ(d->nv_awake, m->nv);
}

// The constants do not depend on the contacts of mjData: the moment of an
// adhesion actuator averages them, and is zero without contacts, as compiled.
TEST_F(SetConstTest, ContactsInDataIgnored) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="box" pos="0 0 .1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
    <actuator>
      <adhesion body="box" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  mjtNum acc0 = m->actuator_acc0[0];
  mj_forward(m.get(), d.get());
  ASSERT_GT(d->ncon, 0);

  mj_setConst(m.get(), d.get());
  EXPECT_EQ(m->actuator_acc0[0], acc0);
  EXPECT_EQ(d->ncon, 0);
}

// What compilation derives once from the conventions of the spec keeps its
// value: mj_setConst neither recomputes it nor resolves the conventions.
TEST_F(SetConstTest, CompiledConstants) {
  constexpr char xml[] = R"(
  <mujoco>
    <statistic meansize="5"/>
    <worldbody>
      <body name="slider">
        <joint name="slider" type="slide" springref=".25"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed stiffness="1">
        <joint joint="slider" coef="2"/>
      </fixed>
    </tendon>
    <actuator>
      <position joint="slider" kp="4" dampratio="1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  // compiled: damping 2 * sqrt(kp * mass), length 2 * springref, the given
  // meansize
  mjtNum* biasprm = m->actuator_biasprm;
  mjtNum* lengthspring = m->tendon_lengthspring;
  EXPECT_MJTNUM_EQ(biasprm[2], -4);
  EXPECT_THAT(AsVector(lengthspring, 2), ElementsAre(0.5, 0.5));
  EXPECT_MJTNUM_EQ(m->stat.meansize, 5);
  mjStatistic stat = m->stat;

  // their inputs change: the values stay
  m->body_mass[1] = 4;
  m->qpos_spring[0] = 0.5;
  mj_setConst(m.get(), d.get());
  EXPECT_MJTNUM_EQ(m->dof_invweight0[0], 0.25);
  EXPECT_MJTNUM_EQ(biasprm[2], -4);
  EXPECT_THAT(AsVector(lengthspring, 2), ElementsAre(0.5, 0.5));
  EXPECT_EQ(std::memcmp(&m->stat, &stat, sizeof(mjStatistic)), 0);

  // the conventions are not resolved
  biasprm[2] = 1;
  lengthspring[0] = lengthspring[1] = -1;
  mj_setConst(m.get(), d.get());
  EXPECT_MJTNUM_EQ(biasprm[2], 1);
  EXPECT_THAT(AsVector(lengthspring, 2), ElementsAre(-1, -1));
}

// Compilation marks welds between bodies which have no relative pose in
// eq_data[11], and mj_setConst computes the pose of marked welds.
TEST_F(SetConstTest, WeldPose) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a" pos="0 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
      <body name="b" pos="1 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
      <body name="m" mocap="true" pos="0 1 1"/>
    </worldbody>
    <equality>
      <weld body1="a" body2="b"/>
      <weld body1="a" body2="b" relpose="0 0 0 2 0 0 0"/>
      <weld body1="m" body2="b"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));

  // compiled: the pose of b relative to a, the given pose normalized, and the
  // pose of b relative to the mocap body m
  mjtNum* computed = m->eq_data;
  mjtNum* given = m->eq_data + mjNEQDATA;
  mjtNum* mocap = m->eq_data + 2 * mjNEQDATA;
  EXPECT_THAT(AsVector(computed + 3, 7), ElementsAre(1, 0, 0, 1, 0, 0, 0));
  EXPECT_EQ(computed[11], 1);
  EXPECT_THAT(AsVector(given + 3, 7), ElementsAre(0, 0, 0, 1, 0, 0, 0));
  EXPECT_EQ(given[11], 0);
  EXPECT_THAT(AsVector(mocap + 3, 7), ElementsAre(1, -1, 0, 1, 0, 0, 0));
  EXPECT_EQ(mocap[11], 1);

  // the reference configuration and the anchor change: the marked pose follows,
  // using the mocap body's model pose rather than d->mocap_pos, which is
  // preserved
  int qposadr = m->jnt_qposadr[1];
  m->qpos0[qposadr] = 2;
  computed[2] = given[2] = 0.5;
  d->mocap_pos[0] = 10;
  mj_setConst(m.get(), d.get());
  EXPECT_THAT(AsVector(computed + 3, 7), ElementsAre(2, 0, 0.5, 1, 0, 0, 0));
  EXPECT_THAT(AsVector(given + 3, 7), ElementsAre(0, 0, 0, 1, 0, 0, 0));
  EXPECT_THAT(AsVector(mocap + 3, 7), ElementsAre(2, -1, 0, 1, 0, 0, 0));
  EXPECT_MJTNUM_EQ(d->mocap_pos[0], 10);

  // unmarked, the pose stays
  computed[11] = 0;
  m->qpos0[qposadr] = 3;
  mj_setConst(m.get(), d.get());
  EXPECT_THAT(AsVector(computed + 3, 7), ElementsAre(2, 0, 0.5, 1, 0, 0, 0));
}

// Computing reference camera and light poses in mj_setConst preserves
// cam_xpos, cam_xmat, light_xpos, and light_xdir in mjData, including for
// static and sleeping cameras and lights with targeting or tracking.
TEST_F(SetConstTest, CamLightPose0PreservesData) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <camera name="static_target" mode="targetbody" target="pedestal"
              pos="0 -1 1"/>
      <light name="static_spot" mode="targetbody" target="pedestal"
             pos="0 -1 2" dir="0 0 -1"/>
      <body name="pedestal" pos="1 0.5 0.4">
        <geom size="0.1"/>
      </body>
      <body name="sleeper" sleep="init" pos="-0.5 0 0.5">
        <joint type="hinge" axis="0 0 1"/>
        <geom size="0.1"/>
        <camera name="sleep_target" mode="targetbodycom" target="pedestal"
                pos="0 0 0.3"/>
        <light name="sleep_spot" mode="targetbodycom" target="pedestal"
               pos="0 0 0.4" dir="0 0 -1"/>
      </body>
      <body name="mover" pos="0 0 1">
        <joint name="j" type="hinge" axis="1 0 0" armature="0.05"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d(mj_makeData(m.get()));
  mj_forward(m.get(), d.get());

  // targeted orientations in mjData differ from fixed-mode reference poses
  EXPECT_NE(AsVector(d->cam_xmat, 9), AsVector(m->cam_mat0, 9));
  EXPECT_NE(AsVector(d->cam_xmat + 9, 9), AsVector(m->cam_mat0 + 9, 9));
  EXPECT_NE(AsVector(d->light_xdir, 3), AsVector(m->light_dir0, 3));
  EXPECT_NE(AsVector(d->light_xdir + 3, 3), AsVector(m->light_dir0 + 3, 3));

  std::vector<mjtNum> cam_xpos = AsVector(d->cam_xpos, 3 * m->ncam);
  std::vector<mjtNum> cam_xmat = AsVector(d->cam_xmat, 9 * m->ncam);
  std::vector<mjtNum> light_xpos = AsVector(d->light_xpos, 3 * m->nlight);
  std::vector<mjtNum> light_xdir = AsVector(d->light_xdir, 3 * m->nlight);

  // unrelated model edit + mj_setConst + mj_forward preserves camera/light
  // poses in mjData: static ones are skipped, the woken tree is unmoved
  m->dof_armature[1] = 0.2;
  mj_setConst(m.get(), d.get());
  mj_forward(m.get(), d.get());
  for (int i = 0; i < 3 * m->ncam; i++) {
    EXPECT_MJTNUM_EQ(d->cam_xpos[i], cam_xpos[i]);
  }
  for (int i = 0; i < 9 * m->ncam; i++) {
    EXPECT_MJTNUM_EQ(d->cam_xmat[i], cam_xmat[i]);
  }
  for (int i = 0; i < 3 * m->nlight; i++) {
    EXPECT_MJTNUM_EQ(d->light_xpos[i], light_xpos[i]);
    EXPECT_MJTNUM_EQ(d->light_xdir[i], light_xdir[i]);
  }

  // editing camera and light offsets updates reference constants in mjModel
  // without mutating mjData until mj_resetData
  m->cam_pos[0] = 0.25;
  m->light_pos[0] = -0.3;
  m->light_dir[0] = 3;
  m->light_dir[2] = -4;
  mj_setConst(m.get(), d.get());
  EXPECT_MJTNUM_EQ(m->cam_pos0[0], 0.25);
  EXPECT_MJTNUM_EQ(m->light_pos0[0], -0.3);
  EXPECT_MJTNUM_EQ(m->light_dir0[0], 0.6);
  EXPECT_MJTNUM_EQ(m->light_dir0[2], -0.8);
  EXPECT_MJTNUM_EQ(d->cam_xpos[0], cam_xpos[0]);
  EXPECT_MJTNUM_EQ(d->light_xpos[0], light_xpos[0]);

  mj_resetData(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->cam_xpos[0], 0.25);
  EXPECT_MJTNUM_EQ(d->light_xpos[0], -0.3);
}

TEST_F(SetConstTest, SetLengthRangeMapsTransmissionIds) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian"/>
    <worldbody>
      <body>
        <joint name="j0" type="hinge" range="-1 1" limited="true"/>
        <geom size="0.1"/>
        <body>
          <joint name="j1" type="hinge" range="-2 2" limited="true"/>
          <geom size="0.1"/>
        </body>
      </body>
    </worldbody>
    <tendon>
      <fixed name="t0" range="0.1 0.5" limited="true">
        <joint joint="j0" coef="1"/>
      </fixed>
      <fixed name="t1" range="0.2 0.8" limited="true">
        <joint joint="j1" coef="1"/>
      </fixed>
    </tendon>
    <actuator>
      <motor name="a_j0" joint="j0"/>
      <motor name="a_j1" joint="j1"/>
      <motor name="a_t0" tendon="t0"/>
      <motor name="a_t1" tendon="t1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mjLROpt opt;
  mj_defaultLROpt(&opt);
  opt.mode = mjLRMODE_ALL;
  opt.uselimit = 1;
  opt.useexisting = 0;

  for (int i = 0; i < m->nactuator; i++) {
    EXPECT_EQ(mj_setLengthRange(m.get(), d.get(), i, &opt, nullptr, 0), 1);
  }

  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[0], -1);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[1], 1);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[2], -2);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[3], 2);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[4], 0.1);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[5], 0.5);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[6], 0.2);
  EXPECT_MJTNUM_EQ(m->actuator_lengthrange[7], 0.8);
}

TEST_F(SetConstTest, SetLengthRangeMultiOutputActuator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="jnt1" type="slide" axis="1 0 0" range="-1 1"/>
        <geom size="0.1"/>
        <site name="s1"/>
        <body>
          <joint name="jnt2" type="slide" axis="0 1 0" range="-1 1"/>
          <geom size="0.1"/>
          <site name="s2"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <orientation name="orient" site="s1" refsite="s2"/>
      <motor name="motor2" joint="jnt2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // orient has outadr=0, outnum=3; motor2 has index=1, outadr=3
  ASSERT_EQ(m->actuator_outadr[1], 3);
  const std::vector<mjtNum> orient_lr(m->actuator_lengthrange,
                                      m->actuator_lengthrange + 6);

  mjLROpt opt;
  mj_defaultLROpt(&opt);
  opt.mode = mjLRMODE_ALL;
  opt.uselimit = 0;

  EXPECT_EQ(mj_setLengthRange(m.get(), d.get(), 1, &opt, error, sizeof(error)),
            1)
      << error;
  for (int i = 0; i < 6; i++) {
    EXPECT_MJTNUM_EQ(m->actuator_lengthrange[i], orient_lr[i]);
  }
  EXPECT_LT(m->actuator_lengthrange[2 * 3], m->actuator_lengthrange[2 * 3 + 1]);
}

}  // namespace
}  // namespace mujoco
