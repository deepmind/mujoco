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

// Tests for the IPC contact mode, engine/engine_ipc.c.
//
// Most tests exercise the geometry/barrier kernels directly (no stepping). The
// two behavioral tests take a single mj_step on a 3x3 cloth: one checks free
// fall, one checks the intersection-free guarantee (a fast cloth cannot tunnel
// a plane in one step).

#include "src/engine/engine_collision_continuous.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <mujoco/mujoco.h>
#include <mujoco/mjtype.h>
#include "src/engine/engine_ipc.h"
#include "src/engine/engine_memory.h"
#include "src/engine/engine_metric.h"
#include "test/fixture.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mujoco {
namespace {

// convenience shims over the MJAPI geometry kernels (pose taken from d, scratch
// dropped)
static mjtNum PtTri(const mjtNum* p, const mjtNum* a, const mjtNum* b,
                    const mjtNum* c) {
  mjtNum cp[3], w[3];
  return mjc_PtTri(p, a, b, c, cp, w);
}
static mjtNum SegSeg(const mjtNum* p1, const mjtNum* p2, const mjtNum* q1,
                     const mjtNum* q2) {
  mjtNum cp1[3], cp2[3], st[2];
  return mjc_SegSeg(p1, p2, q1, q2, cp1, cp2, st);
}
static mjtNum GeomDist(const mjModel* m, const mjData* d, int gi,
                       const mjtNum* x, mjtNum* n) {
  return mjc_GeomDist(m, gi, d->geom_xpos + 3 * gi, d->geom_xmat + 9 * gi, x, n,
                      1e30);
}
static int GeomVerts(const mjModel* m, const mjData* d, int gi, mjtNum* out) {
  return mjc_GeomVerts(m, gi, d->geom_xpos + 3 * gi, d->geom_xmat + 9 * gi,
                       out);
}
static int GeomEdges(const mjModel* m, const mjData* d, int gi, mjtNum* out) {
  return mjc_GeomEdges(m, gi, d->geom_xpos + 3 * gi, d->geom_xmat + 9 * gi,
                       out);
}

using ::testing::NotNull;
using IpcTest = MujocoTest;

static mjModel* Load(const char* xml) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(model.get(), NotNull()) << error;
  return model.release();
}

//---------------------------------- element distances
//--------------------------------------------

// point-triangle distance: interior (perpendicular), edge region, vertex region
TEST_F(IpcTest, PointTriangleDistance) {
  mjtNum a[3] = {0, 0, 0}, b[3] = {1, 0, 0}, c[3] = {0, 1, 0};

  mjtNum p_above[3] = {0.2, 0.2, 0.5};  // over the interior
  EXPECT_NEAR(PtTri(p_above, a, b, c), 0.5, 1e-12);

  mjtNum p_edge[3] = {-1, 0.5, 0};  // nearest the x=0 edge
  EXPECT_NEAR(PtTri(p_edge, a, b, c), 1.0, 1e-12);

  mjtNum p_vert[3] = {-3, -4, 0};  // nearest vertex a
  EXPECT_NEAR(PtTri(p_vert, a, b, c), 5.0, 1e-12);

  mjtNum p_on[3] = {0.25, 0.25, 0};  // on the triangle
  EXPECT_NEAR(PtTri(p_on, a, b, c), 0.0, 1e-12);
}

// segment-segment distance: perpendicular crossing, collinear gap, parallel
// offset
TEST_F(IpcTest, SegmentSegmentDistance) {
  mjtNum p1[3] = {-1, 0, 0}, p2[3] = {1, 0, 0};

  mjtNum q1[3] = {0, -1, 0.3}, q2[3] = {0, 1, 0.3};  // perpendicular, 0.3 above
  EXPECT_NEAR(SegSeg(p1, p2, q1, q2), 0.3, MjTol(1e-12, 1e-6));

  mjtNum r1[3] = {2, 0, 0}, r2[3] = {3, 0, 0};  // collinear, gap 1
  EXPECT_NEAR(SegSeg(p1, p2, r1, r2), 1.0, 1e-12);

  mjtNum s1[3] = {-1, 0, 0.5}, s2[3] = {1, 0, 0.5};  // parallel, 0.5 above
  EXPECT_NEAR(SegSeg(p1, p2, s1, s2), 0.5, 1e-12);
}

//---------------------------------- geom distance
//------------------------------------------------

constexpr char kPrimitivesXml[] = R"(
<mujoco>
  <worldbody>
    <geom name="box" type="box" size="0.1 0.2 0.3" pos="0 0 0"/>
    <geom name="sphere" type="sphere" size="0.1" pos="1 0 0"/>
    <geom name="plane" type="plane" size="0 0 1" pos="0 0 -1"/>
  </worldbody>
</mujoco>
)";

TEST_F(IpcTest, GeomDistance) {
  mjModel* m = Load(kPrimitivesXml);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);
  int box = mj_name2id(m, mjOBJ_GEOM, "box");
  int sphere = mj_name2id(m, mjOBJ_GEOM, "sphere");
  int plane = mj_name2id(m, mjOBJ_GEOM, "plane");
  mjtNum n[3];

  // box (half-extent 0.1 in x): point on +x at 0.5 -> surface distance 0.4,
  // normal +x
  mjtNum px[3] = {0.5, 0, 0};
  EXPECT_NEAR(GeomDist(m, d, box, px, n), 0.4, MjTol(1e-9, 1e-6));
  EXPECT_NEAR(n[0], 1, 1e-9);
  EXPECT_NEAR(n[1], 0, 1e-9);
  EXPECT_NEAR(n[2], 0, 1e-9);

  // interior point -> negative signed distance
  mjtNum pc[3] = {0, 0, 0};
  EXPECT_LT(GeomDist(m, d, box, pc, n), 0);

  // sphere radius 0.1 at (1,0,0): point at (1.3,0,0) -> 0.2, normal +x
  mjtNum ps[3] = {1.3, 0, 0};
  EXPECT_NEAR(GeomDist(m, d, sphere, ps, n), 0.2, MjTol(1e-9, 1e-6));
  EXPECT_NEAR(n[0], 1, 1e-9);

  // plane at z=-1: point at z=0 -> 1.0, normal +z
  mjtNum pp[3] = {0.3, -0.2, 0};
  EXPECT_NEAR(GeomDist(m, d, plane, pp, n), 1.0, 1e-9);
  EXPECT_NEAR(n[2], 1, 1e-9);

  mj_deleteData(d);
  mj_deleteModel(m);
}

//---------------------------------- geom sharp features
//------------------------------------------

// a box exposes its 8 corners (at +/-size) and 12 edges
TEST_F(IpcTest, BoxFeatures) {
  constexpr char xml[] = R"(
  <mujoco><worldbody>
    <geom type="box" size="0.1 0.2 0.3"/>
  </worldbody></mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  mjtNum verts[8 * 3], edges[12 * 6];
  int nv = GeomVerts(m, d, /*gi=*/0, verts);
  int ne = GeomEdges(m, d, /*gi=*/0, edges);
  EXPECT_EQ(nv, 8);
  EXPECT_EQ(ne, 12);
  const mjtNum tol = MjTol(1e-9, 1e-6);
  for (int i = 0; i < nv; i++) {
    EXPECT_NEAR(std::fabs(verts[3 * i + 0]), 0.1, tol);
    EXPECT_NEAR(std::fabs(verts[3 * i + 1]), 0.2, tol);
    EXPECT_NEAR(std::fabs(verts[3 * i + 2]), 0.3, tol);
  }
  // every box edge has unit length along exactly one axis (here 0.2, 0.4, or
  // 0.6)
  for (int i = 0; i < ne; i++) {
    mjtNum dx = edges[6 * i + 3] - edges[6 * i + 0];
    mjtNum dy = edges[6 * i + 4] - edges[6 * i + 1];
    mjtNum dz = edges[6 * i + 5] - edges[6 * i + 2];
    mjtNum len = std::sqrt(dx * dx + dy * dy + dz * dz);
    EXPECT_TRUE(std::fabs(len - 0.2) < tol || std::fabs(len - 0.4) < tol ||
                std::fabs(len - 0.6) < tol)
        << "edge " << i << " length " << len;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}

// a convex mesh exposes its vertices and its (deduplicated) hull edges; a
// tetrahedron has 4 and 6
TEST_F(IpcTest, MeshFeatures) {
  constexpr char xml[] = R"(
  <mujoco>
    <asset><mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
    <worldbody><geom type="mesh" mesh="tet"/></worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  mjtNum verts[64 * 3], edges[256 * 6];
  int nv = GeomVerts(m, d, /*gi=*/0, verts);
  int ne = GeomEdges(m, d, /*gi=*/0, edges);
  EXPECT_EQ(nv, 4);  // tetrahedron vertices
  EXPECT_EQ(ne, 6);  // tetrahedron edges (each shared hull edge emitted once)
  mj_deleteData(d);
  mj_deleteModel(m);
}

//---------------------------------- contact mode behavior
//------------------------------------------

// a single 2D cloth, no contact: one step is free fall (qvel gains -g*dt on
// every free vertex)
constexpr char kClothXml[] = R"(
<mujoco>
  <option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
  <worldbody>
    <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
              radius="0.005" mass="0.05" pos="0 0 0.5"/>
  </worldbody>
</mujoco>
)";

// The flag applies model-wide: every supported flex has passive contact of the
// flag's own kind, so the penalty form is replaced rather than added to.
// Running both applies each pair's force twice; that drove the sheets 1.9 m
// into each other and then to NaN.
TEST_F(IpcTest, FlagOverridesPassiveContact) {
  static constexpr char kWith[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
      <flag ipc="enable"/>
    </option>
    <worldbody>
      <flexcomp name="lower" type="grid" dim="2" count="9 9 1" spacing=".04 .04 1"
                radius=".004" mass=".3" pos="0 0 .2">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
        <pin id="0 8 72 80"/>
      </flexcomp>
      <flexcomp name="upper" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                radius=".004" mass=".1" pos="0 0 .27">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  std::string without(kWith);
  for (size_t i = without.find(" passive=\"true\""); i != std::string::npos;
       i = without.find(" passive=\"true\"")) {
    without.erase(i, std::string(" passive=\"true\"").size());
  }

  mjModel* mw = Load(kWith);
  mjModel* mo = Load(without.c_str());
  mjData* dw = mj_makeData(mw);
  mjData* do_ = mj_makeData(mo);

  for (int s = 0; s < 60; s++) {
    mj_step(mw, dw);
    mj_step(mo, do_);
    // the passive path must be off: no contact is excluded to it, and it
    // publishes no rows for the metric
    for (int i = 0; i < dw->ncon; i++) {
      ASSERT_NE(dw->contact[i].exclude, 4) << "passive exclusion at step " << s;
    }
    ASSERT_EQ(dw->nefmcon, 0) << "passive rows published at step " << s;
    ASSERT_FALSE(dw->warning[mjWARN_BADQACC].number)
        << "diverged at step " << s;
  }
  for (int i = 0; i < mw->nq; i++) {
    EXPECT_EQ(dw->qpos[i], do_->qpos[i]) << "qpos " << i << " differs";
  }

  mj_deleteData(do_);
  mj_deleteData(dw);
  mj_deleteModel(mo);
  mj_deleteModel(mw);
}

// the reported constraint force accounts for the contact pairs: at rest on a
// plane, M*qacc equals qfrc_smooth + qfrc_constraint and the contact rows carry
// the cloth's weight
TEST_F(IpcTest, ConstraintForceReported) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
      <flag ipc="enable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1" pos="0 0 0"/>
      <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1"
                radius="0.004" mass="0.1" pos="0 0 0.03">
        <edge equality="true"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  mjModel* m = Load(kXml);
  mjData* d = mj_makeData(m);
  for (int s = 0; s < 500; s++) {
    mj_step(m, d);
    ASSERT_FALSE(d->warning[mjWARN_BADQACC].number) << "diverged at step " << s;
  }
  int nv = m->nv;
  std::vector<mjtNum> Mqacc(nv);
  mj_mulM(m, d, Mqacc.data(), d->qacc);
  mjtNum scale = 0, residual = 0, load = 0;
  for (int i = 0; i < nv; i++) {
    scale = std::max(scale, std::abs(d->qfrc_smooth[i]));
    residual = std::max(residual, std::abs(Mqacc[i] - d->qfrc_smooth[i] -
                                           d->qfrc_constraint[i]));
    if (i % 3 == 2)
      load += d->qfrc_constraint[i];  // the vertices' vertical dofs
  }
  EXPECT_GT(scale, 0);
  // the identity M*qacc = qfrc_smooth + qfrc_constraint holds with the contact
  // pairs included to the solver's own tolerance: the identity is the CG's
  // stationarity condition
  EXPECT_LT(residual, MjTol(1e-4, 1e-2) * scale)
      << "residual " << residual << " scale " << scale;
  // the pairs carry the weight: the vertical constraint force sums to m*g
  mjtNum weight = 0.1 * 9.81;
  EXPECT_NEAR(load, weight, MjTol(1e-3, 1e-2) * weight);
  mj_deleteData(d);
  mj_deleteModel(m);
}

// a 5x5 cloth 3 cm above the floor, with slots for extra option flags, extra
// geoms and flex contact attributes
static constexpr char kClothOverFloorXml[] = R"(
<mujoco>
  <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
    <flag ipc="enable" %s/>
  </option>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1"/>
    %s
    <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1"
              radius="0.004" mass="0.1" pos="0 0 0.03">
      <edge equality="true"/>
      <contact selfcollide="none" %s/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

// the lowest vertex of the cloth above after 1 s
static mjtNum ClothMinHeight(const char* flags, const char* geoms,
                             const char* contact) {
  char xml[2048];
  snprintf(xml, sizeof(xml), kClothOverFloorXml, flags, geoms, contact);
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  for (int s = 0; s < 500; s++) {
    mj_step(m, d);
  }
  mjtNum zmin = 1;
  for (int v = 0; v < m->nflexvert; v++) {
    zmin = std::min(zmin, d->flexvert_xpos[3 * v + 2]);
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  return zmin;
}

// the contact pairs follow the native collision filtering: none with the
// contact flag disabled, and the contype/conaffinity rule between a flex and a
// geom
TEST_F(IpcTest, ContactFlagAndMasksFilterGeomPairs) {
  // the cloth rests on the floor
  EXPECT_GT(ClothMinHeight("", "", ""), 0);
  // contact disabled: it falls through
  EXPECT_LT(ClothMinHeight("contact=\"disable\"", "", ""), -1);
  // the flex's masks zero: filtered against everything
  EXPECT_LT(ClothMinHeight("", "", "contype=\"0\" conaffinity=\"0\""), -1);
  // the bitmask rule: type 2 misses the floor's affinity 1 and lands on a
  // lower plane whose affinity 2 meets it
  mjtNum z = ClothMinHeight(
      "",
      "<geom type=\"plane\" size=\"0 0 1\" pos=\"0 0 -0.5\" contype=\"0\" "
      "conaffinity=\"2\"/>",
      "contype=\"2\" conaffinity=\"2\"");
  EXPECT_GT(z, -0.5);
  EXPECT_LT(z, -0.45);
}

TEST_F(IpcTest, FreeFall) {
  mjModel* m = Load(kClothXml);
  mjData* d = mj_makeData(m);
  int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
  ASSERT_GE(f, 0);

  mj_step(m, d);
  EXPECT_NEAR(d->time, m->opt.timestep, 1e-12);

  // every free vertex slide-joint along z should hold v = g_z * dt after one
  // step
  mjtNum want = m->opt.gravity[2] * m->opt.timestep;
  int checked = 0;
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[f] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      if (m->jnt_type[jid] == mjJNT_SLIDE && m->jnt_axis[3 * jid + 2] > 0.5) {
        EXPECT_NEAR(d->qvel[m->jnt_dofadr[jid]], want, MjTol(1e-6, 1e-4))
            << "vertex " << i;
        checked++;
      }
    }
  }
  EXPECT_GT(checked, 0);  // the model really did expose free z slide joints
  EXPECT_FALSE(std::isnan(d->qpos[0]));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// two flexes sharing a vertex body (a seam: one flex's vertex attached to
// another flex's vertex body, here 1.5 mm above the other flex's triangle).
// Features with points on one body move rigidly together, so they make no
// contact pair: a pair would push the body against itself, bending the
// triangle. Both flexes fall freely.
TEST_F(IpcTest, SharedVertexBodyMakesNoContact) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
      <flag ipc="enable"/>
    </option>
    <default>
      <joint type="slide"/>
    </default>
    <worldbody>
      <body name="a0" pos="0 0 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
      <body name="a1" pos=".04 0 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
      <body name="a2" pos="0 .04 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
      <body name="a3" pos=".04 .04 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
      <body name="b1" pos="-.04 0 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
      <body name="b2" pos="0 -.04 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
      <body name="b3" pos="-.04 -.04 .2">
        <inertial pos="0 0 0" mass=".01" diaginertia="1e-6 1e-6 1e-6"/>
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/></body>
    </worldbody>
    <deformable>
      <flex name="a" dim="2" radius=".004" body="a0 a1 a2 a3" element="0 1 2 1 3 2">
        <contact selfcollide="none"/>
      </flex>
      <flex name="b" dim="2" radius=".004" body="a0 b1 b2 b3" element="0 1 2 1 3 2"
            vertex=".01 .01 .0015  0 0 0  0 0 0  0 0 0">
        <contact selfcollide="none"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  mjModel* m = Load(xml);
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);
  for (int s = 0; s < 50; s++) {
    mj_step(m, d);
    ASSERT_FALSE(d->warning[mjWARN_BADQACC].number) << "diverged at step " << s;
  }

  // no contact force: every vertex falls by the same amount
  mjtNum z0 = d->xpos[3 * mj_name2id(m, mjOBJ_BODY, "a0") + 2];
  EXPECT_LT(z0, 0.2 - 0.01);
  for (const char* name : {"a1", "a2", "a3", "b1", "b2", "b3"}) {
    int b = mj_name2id(m, mjOBJ_BODY, name);
    EXPECT_NEAR(d->xpos[3 * b + 2], z0, MjTol(1e-9, 1e-5)) << name;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}

// the intersection-free guarantee: a cloth driven hard at a plane cannot pass
// through in one step (a single explicit Euler step at this speed would put it
// far below the plane).
TEST_F(IpcTest, ContactBlocksTunneling) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1" pos="0 0 0"/>
      <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                radius="0.005" mass="0.05" pos="0 0 0.05"/>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
  ASSERT_GE(f, 0);

  // drive every vertex straight down at 50 m/s (0.1 m of travel per 2 ms step,
  // far past the plane)
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[f] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      if (m->jnt_type[jid] == mjJNT_SLIDE && m->jnt_axis[3 * jid + 2] > 0.5) {
        d->qvel[m->jnt_dofadr[jid]] = -50;
      }
    }
  }

  mj_step(m, d);
  mj_kinematics(m, d);
  mj_flex(m, d);

  // no vertex crossed the plane (center stays above z=0; the radius keeps the
  // surface above that)
  mjtNum minz = 1e30;
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    mjtNum z = d->flexvert_xpos[3 * (m->flex_vertadr[f] + i) + 2];
    if (z < minz) minz = z;
  }
  EXPECT_GT(minz, 0) << "cloth tunneled through the plane";
  EXPECT_FALSE(std::isnan(minz));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// an arm carrying a fully pinned 3x3 cloth as a paddle, with slots for the
// option flags, the arm's joint and a second flex
static constexpr char kPaddleXml[] = R"(
<mujoco>
  <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">%s</option>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1"/>
    <body name="arm" pos="0 0 0.2">
      <joint name="carrier" %s/>
      <geom type="capsule" fromto="0 0 0  0.1 0 0" size="0.005" mass="0.02"
            contype="0" conaffinity="0"/>
      <flexcomp name="paddle" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                radius="0.004" mass="0.05" pos="0.06 0 0">
        <contact selfcollide="none"/>
        <pin id="0 1 2 3 4 5 6 7 8"/>
      </flexcomp>
    </body>
    %s
  </worldbody>
</mujoco>
)";

// the arm on a sprung, well-damped vertical slide
static constexpr char kSlide[] =
    "type=\"slide\" axis=\"0 0 1\" stiffness=\"50\" damping=\"2\"";

// a body carrying pinned flex vertices is integrated like any articulated
// body: released 5 cm below its rest, the arm with the pinned paddle follows
// the same trajectory under the IPC mode as under the plain discrete integrator
TEST_F(IpcTest, PinnedFlexBodyIntegrates) {
  mjtNum q[2], v[2];
  const char* flags[2] = {"<flag ipc=\"enable\"/>", ""};
  for (int i = 0; i < 2; i++) {
    char xml[2048];
    snprintf(xml, sizeof(xml), kPaddleXml, flags[i], kSlide, "");
    mjModel* m = Load(xml);
    mjData* d = mj_makeData(m);
    d->qpos[0] = -0.05;
    for (int s = 0; s < 200; s++) {
      mj_step(m, d);
    }
    q[i] = d->qpos[0];
    v[i] = d->qvel[0];
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  EXPECT_GT(std::fabs(q[1] + 0.05), 0.005) << "the arm did not move";
  EXPECT_NEAR(q[0], q[1], MjTol(1e-8, 1e-4));
  EXPECT_NEAR(v[0], v[1], MjTol(1e-6, 1e-3));
}

// a free cloth dropped on the pinned paddle rests on it and loads the arm
// through the pinned corners: the slide sags further than under the paddle's
// own weight, and the slide's constraint force closes its equation of motion
TEST_F(IpcTest, PinnedPaddleCarriesCloth) {
  static constexpr char kCloth[] = R"(
      <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.03 0.03 1"
                radius="0.004" mass="0.05" pos="0.06 0 0.24">
        <edge equality="true"/>
        <contact selfcollide="none"/>
      </flexcomp>)";
  mjtNum q[2];
  const char* extra[2] = {"", kCloth};
  for (int i = 0; i < 2; i++) {
    char xml[4096];
    snprintf(xml, sizeof(xml), kPaddleXml, "<flag ipc=\"enable\"/>", kSlide,
             extra[i]);
    mjModel* m = Load(xml);
    mjData* d = mj_makeData(m);
    for (int s = 0; s < 500; s++) {
      mj_step(m, d);
      ASSERT_FALSE(d->warning[mjWARN_BADQACC].number)
          << "diverged at step " << s;
    }
    q[i] = d->qpos[0];
    if (i == 1) {
      // the cloth stays on the paddle, above the floor by far
      int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
      for (int k = 0; k < m->flex_vertnum[f]; k++) {
        EXPECT_GT(d->flexvert_xpos[3 * (m->flex_vertadr[f] + k) + 2], 0.15)
            << "cloth vertex " << k << " fell off the paddle";
      }
      // M*qacc = qfrc_smooth + qfrc_constraint on the slide: the rows of the
      // pairs with pinned corners act on the arm's dof. The mode applies the
      // joint damping implicitly, at the new velocity, where qfrc_smooth
      // carries it at the old one, so that increment is accounted for; the
      // identity holds to the outer loop's velocity tolerance on the arm's
      // dof, looser than the CG's stationarity on the cloth's
      std::vector<mjtNum> Mqacc(m->nv);
      mj_mulM(m, d, Mqacc.data(), d->qacc);
      mjtNum scale = 0;
      for (int j = 0; j < m->nv; j++)
        scale = std::max(scale, std::abs(d->qfrc_smooth[j]));
      mjtNum damping = m->dof_damping[0] * d->qacc[0] * m->opt.timestep;
      EXPECT_LT(std::abs(Mqacc[0] - d->qfrc_smooth[0] + damping -
                         d->qfrc_constraint[0]),
                MjTol(1e-3, 1e-1) * scale);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  // the arm sags under its own weight (a pinned vertex carries no mass, so the
  // paddle adds none), and further under the cloth's: 4 mm and 10 mm at this
  // stiffness; the slide keeps the paddle level, so the frictionless cloth
  // stays on it
  mjtNum dir = q[0] > 0 ? 1 : -1;
  EXPECT_GT(std::fabs(q[0]), 1e-3) << "the arm did not sag";
  EXPECT_GT((q[1] - q[0]) * dir, 3e-3)
      << "the cloth's weight did not reach the arm";
}

// a body with three slide joints is a free flex vertex only when it is the
// vertex: a cloth fully pinned to one such body shares the body's three dofs
// among its nine vertices and must take the pinned path once, so a step
// moves the carrier by h*v like the discrete integrator, not nine times that
TEST_F(IpcTest, SlideCarrierMovesOnce) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" gravity="0 0 0">
      %s
    </option>
    <worldbody>
      <body name="carrier" pos="0 0 0.5">
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
        <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.03 0.03 1"
                  radius="0.004" mass="0.05" pos="0.06 0 0">
          <pin id="0 1 2 3 4 5 6 7 8"/>
          <edge equality="true"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>)";
  const char* flags[2] = {"<flag ipc=\"enable\"/>", ""};
  mjtNum qx[2], cx[2];
  for (int i = 0; i < 2; i++) {
    char xml[2048];
    snprintf(xml, sizeof(xml), kXml, flags[i]);
    mjModel* m = Load(xml);
    mjData* d = mj_makeData(m);
    d->qvel[0] = 1;
    for (int s = 0; s < 10; s++) mj_step(m, d);
    int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
    qx[i] = d->qpos[0];
    cx[i] = 0;
    for (int k = 0; k < m->flex_vertnum[f]; k++)
      cx[i] +=
          d->flexvert_xpos[3 * (m->flex_vertadr[f] + k)] / m->flex_vertnum[f];
    EXPECT_NEAR(d->qvel[0], 1, MjTol(1e-12, 1e-6));
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  EXPECT_NEAR(qx[0], 0.02, MjTol(1e-12, 1e-6))
      << "the carrier moved by h*v per step";
  EXPECT_NEAR(qx[0], qx[1], MjTol(1e-12, 1e-6)) << "IPC vs discrete: carrier";
  EXPECT_NEAR(cx[0], cx[1], MjTol(1e-12, 1e-6)) << "IPC vs discrete: cloth";
}

// a pinned vertex rides a body whose chain translates: its path over a step is
// the straight segment the CCD sweeps. A hinge or a ball joint on the chain
// moves it on an arc and is refused, at load, since the compiler steps the
// model once; a ball joint has three dofs like a flex vertex's slides and must
// not be taken for one
TEST_F(IpcTest, PinnedCarrierMustTranslate) {
  const char* joints[2] = {"type=\"hinge\" axis=\"0 1 0\"", "type=\"ball\""};
  for (int i = 0; i < 2; i++) {
    char xml[2048], error[1024];
    snprintf(xml, sizeof(xml), kPaddleXml, "<flag ipc=\"enable\"/>", joints[i],
             "");
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    EXPECT_THAT(m.get(), testing::IsNull()) << "joint " << joints[i];
    EXPECT_THAT(error, testing::HasSubstr("slide joints only"));
  }
}

// a fully pinned paddle on a free vertical slide falls onto the floor and
// stops on it: the pinned vertices move with their body and meet the plane
// through the mode's pairs, the only contact they have under the flag
TEST_F(IpcTest, PinnedPaddleStopsOnPlane) {
  char xml[2048];
  snprintf(xml, sizeof(xml), kPaddleXml, "<flag ipc=\"enable\"/>",
           "type=\"slide\" axis=\"0 0 1\" damping=\"0.05\"", "");
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int f = mj_name2id(m, mjOBJ_FLEX, "paddle");
  mjtNum zmin = 1;
  for (int s = 0; s < 500; s++) {
    mj_step(m, d);
    ASSERT_FALSE(d->warning[mjWARN_BADQACC].number) << "diverged at step " << s;
    for (int k = 0; k < m->flex_vertnum[f]; k++)
      zmin = std::min(zmin, d->flexvert_xpos[3 * (m->flex_vertadr[f] + k) + 2]);
  }
  EXPECT_GT(zmin, 0) << "the paddle passed through the floor";
  EXPECT_LT(d->qpos[0], -0.15) << "the arm did not fall";
  EXPECT_LT(std::fabs(d->qvel[0]), 0.05) << "the arm is not at rest";
  mj_deleteData(d);
  mj_deleteModel(m);
}

// under the flag the collision pipeline does not generate the contacts the
// mode resolves: a cloth on the floor has no native contact with it, while a
// free ball resting on the cloth keeps its native contacts, the mode leaving
// flex against moving bodies to the constraint solver
TEST_F(IpcTest, NativeCollisionSkipsTheModesPairs) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">%s</option>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1"/>
      <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1"
                radius="0.004" mass="0.1" pos="0 0 0.004">
        <edge equality="true"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <body name="ball" pos="0 0 0.05">
        <freejoint/>
        <geom type="sphere" size="0.02" mass="0.05"/>
      </body>
    </worldbody>
  </mujoco>)";
  for (int flag = 0; flag < 2; flag++) {
    char xml[2048];
    snprintf(xml, sizeof(xml), kXml, flag ? "<flag ipc=\"enable\"/>" : "");
    mjModel* m = Load(xml);
    mjData* d = mj_makeData(m);
    for (int s = 0; s < 100; s++) {
      mj_step(m, d);
    }
    int floor = 0, ball = 0;
    for (int c = 0; c < d->ncon; c++) {
      const mjContact* con = d->contact + c;
      if (con->flex[0] < 0 && con->flex[1] < 0) continue;
      int g = con->geom[0] >= 0 ? con->geom[0] : con->geom[1];
      if (g < 0) continue;
      if (m->geom_bodyid[g] == 0) {
        floor++;
      } else {
        ball++;
      }
    }
    if (flag) {
      EXPECT_EQ(floor, 0) << "cloth-floor contacts generated under the flag";
    } else {
      EXPECT_GT(floor, 0) << "the cloth does not touch the floor";
    }
    EXPECT_GT(ball, 0) << "the ball's contacts with the cloth are missing";
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

// a cloth dropped on a sphere or a capsule too small to come near any of its
// vertices: the obstacle meets a triangle's interior (the sphere) or its edges
// (the capsule), and the cloth must tent over it rather than pass through
TEST_F(IpcTest, SmoothGeomsCoverTriangleInterior) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
      <flag ipc="enable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1"/>
      %s
      <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1"
                radius="0.004" mass="0.1" pos="0 0 0.05">
        <edge equality="true"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  // a 5 mm sphere under a triangle's interior, 27 mm from the nearest vertex,
  // and a 5 mm capsule along x midway between two vertex rows
  const char* obstacles[2] = {
      "<geom name=\"obstacle\" type=\"sphere\" size=\"0.005\" "
      "pos=\"0.021 0.019 0.02\"/>",
      "<geom name=\"obstacle\" type=\"capsule\" size=\"0.005 0.03\" "
      "pos=\"0 0.02 0.02\" euler=\"0 90 0\"/>"};
  for (const char* obstacle : obstacles) {
    char xml[2048];
    snprintf(xml, sizeof(xml), kXml, obstacle);
    mjModel* m = Load(xml);
    mjData* d = mj_makeData(m);
    for (int s = 0; s < 500; s++) {
      mj_step(m, d);
    }
    int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
    int gi = mj_name2id(m, mjOBJ_GEOM, "obstacle");
    mjtNum r = m->geom_size[3 * gi], half = m->geom_size[3 * gi + 1];
    const mjtNum* gp = d->geom_xpos + 3 * gi;
    const mjtNum* gR = d->geom_xmat + 9 * gi;
    // probe points on the obstacle's axis: the centre and, for the capsule,
    // both ends of the axis
    int nprobe = m->geom_type[gi] == mjGEOM_CAPSULE ? 3 : 1;
    for (int q = 0; q < nprobe; q++) {
      mjtNum along = (q == 0 ? 0 : (q == 1 ? half : -half));
      mjtNum p[3] = {gp[0] + along * gR[2], gp[1] + along * gR[5],
                     gp[2] + along * gR[8]};
      // the closest cloth triangle to the probe point
      mjtNum best = 1e30, bcp[3] = {0, 0, 0};
      const int* el = m->flex_elem + m->flex_elemdataadr[f];
      const mjtNum* xv = d->flexvert_xpos + 3 * m->flex_vertadr[f];
      for (int e = 0; e < m->flex_elemnum[f]; e++) {
        mjtNum cp[3], w[3];
        mjtNum dd = mjc_PtTri(p, xv + 3 * el[3 * e], xv + 3 * el[3 * e + 1],
                              xv + 3 * el[3 * e + 2], cp, w);
        if (dd < best) {
          best = dd;
          for (int c = 0; c < 3; c++) bcp[c] = cp[c];
        }
      }
      EXPECT_GT(bcp[2], p[2]) << obstacle << ": the cloth passed through";
      EXPECT_GT(best, 0.9 * r)
          << obstacle << ": a triangle inside the obstacle";
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

// Total linear momentum, summed over the flex vertex bodies (each a point mass
// on three slide joints). Read from qvel rather than d->cvel, which is stale
// after mj_step.
static mjtNum FlexMomentum(const mjModel* m, const mjData* d, mjtNum P[3]) {
  mjtNum mass = 0;
  mju_zero3(P);
  for (int b = 1; b < m->nbody; b++) {
    int adr = m->body_dofadr[b];
    if (adr < 0 || m->body_dofnum[b] != 3) {
      continue;
    }
    mass += m->body_mass[b];
    for (int k = 0; k < 3; k++) {
      P[k] += m->body_mass[b] * d->qvel[adr + k];
    }
  }
  return mass;
}

// CONSERVATION under self-contact. Two cloths collide in mid-air with nothing
// else in the scene: no pins, no floor, no geoms, so contact, elastic and
// damping forces are all INTERNAL and cancel in the sum. Summing the step
// update over every vertex therefore gives, exactly,
//
//     sum_i m_i * (v_i+ - v_i)  ==  h * M_total * g
//
// independent of the contact solve -- PROVIDED the committed step is the
// solution of that solve. This is what makes the test worth its cost: a step
// that commits a line-search-truncated iterate discards the motion the
// truncation dropped, and the deficit lands here as missing momentum. That
// failure looks healthy to every other test we have (no NaN, no tunneling,
// plausible timings), so without this invariant a solver that silently stops
// advancing time passes CI.
TEST_F(IpcTest, SelfContactConservesMomentum) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
    <worldbody>
      <flexcomp name="lower" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                radius="0.004" mass="0.02" pos="0 0 0.5"/>
      <flexcomp name="upper" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                radius="0.004" mass="0.02" pos="0 0 0.54"/>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int upper = mj_name2id(m, mjOBJ_FLEX, "upper");
  ASSERT_GE(upper, 0);

  // drive the upper cloth down onto the lower one: under uniform gravity alone
  // the two would fall together and never touch, so the approach velocity is
  // what creates the contact
  for (int i = 0; i < m->flex_vertnum[upper]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[upper] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      if (m->jnt_type[jid] == mjJNT_SLIDE && m->jnt_axis[3 * jid + 2] > 0.5) {
        d->qvel[m->jnt_dofadr[jid]] = -2;
      }
    }
  }

  mjtNum P0[3], P1[3];
  mjtNum mass = FlexMomentum(m, d, P0);
  ASSERT_GT(mass, 0) << "no flex vertex bodies found: the scene is not the one "
                        "this test assumes";
  mjtNum h = m->opt.timestep;
  mjtNum scale = mass * mju_norm3(m->opt.gravity) *
                 h;  // magnitude of one step's momentum change

  // 40 steps: the cloths meet at ~step 10 and stay in contact for the rest, so
  // the window covers approach, impact and sustained contact
  for (int s = 0; s < 40; s++) {
    mj_step(m, d);
    FlexMomentum(m, d, P1);
    ASSERT_FALSE(std::isnan(P1[2])) << "NaN at step " << s;
    for (int k = 0; k < 3; k++) {
      mjtNum got = P1[k] - P0[k];
      mjtNum want = h * mass * m->opt.gravity[k];
      // tolerance is a fraction of ONE step's momentum change, so it stays
      // meaningful under sustained contact; the failure this guards discards
      // most of a step and is O(1) relative
      EXPECT_NEAR(got, want, MjTol(1e-6, 1e-3) * scale)
          << "step " << s << " axis " << k
          << ": internal contact forces must cancel";
      P0[k] = P1[k];
    }
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}

// a cloth that STARTS inside a geom is pushed back out. Broad-phase must keep
// pairs whose gap is already negative: dropping them leaves the penetration
// with nothing acting on it and the cloth free-falls through.
TEST_F(IpcTest, ContactRecoversFromPenetration) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1" pos="0 0 0"/>
      <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                radius="0.005" mass="0.05" pos="0 0 -0.02"/>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
  ASSERT_GE(f, 0);

  // every vertex starts 20 mm below the plane, so every flex-geom gap is
  // negative at step 0
  for (int i = 0; i < 100; i++) {
    mj_step(m, d);
  }
  mj_kinematics(m, d);
  mj_flex(m, d);

  mjtNum minz = 1e30;
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    mjtNum z = d->flexvert_xpos[3 * (m->flex_vertadr[f] + i) + 2];
    if (z < minz) minz = z;
  }
  EXPECT_GT(minz, 0) << "cloth stayed below the plane it started inside";
  EXPECT_FALSE(std::isnan(minz));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// RUNG 0 of the all-joint rebuild: the per-tree generalized-coordinate core
// (mj_ipcTree). With NO contact the smooth forces enter explicitly through the
// predictor (qacc_smooth), so the variational step reduces to semi-implicit
// Euler EXACTLY -- the only things under test are the predictor, the
// configuration manifold (mj_integratePos/differentiatePos exp/log for
// quaternion DOFs), and the coupled per-tree mass (mj_mulM / mj_solveM). A
// free, a freely-spinning (quaternion + Coriolis), and an articulated (coupled
// M(q)) body under flag ipc must therefore match the discrete integrator
// without the flag to machine precision: the mode adds nothing without flex
// contact.
TEST_F(IpcTest, RigidNoContactMatchesDiscrete) {
  struct Scene {
    const char* name;
    const char* xml;
  };
  const Scene scenes[] = {
      {"free_ballistic", R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody><body pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.15 0.2" mass="1"/></body></worldbody>
        <keyframe><key qvel="1 0.5 -0.3 0 0 0"/></keyframe></mujoco>)"},
      {"free_spin", R"(
      <mujoco><option timestep="0.002" gravity="0 0 0"/>
        <worldbody><body pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.2 0.3" mass="1"/></body></worldbody>
        <keyframe><key qvel="0 0 0 2 1.3 0.7"/></keyframe></mujoco>)"},
      {"hinge_chain", R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody>
          <body pos="0 0 1"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
            <body pos="0.3 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
              <body pos="0.3 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body>
            </body></body>
        </worldbody></mujoco>)"},
  };
  const int N = 500;
  for (const Scene& s : scenes) {
    mjModel* mi = Load(s.xml);
    mi->opt.integrator = mjINT_DISCRETE;
    mi->opt.solver =
        mjSOL_NEWTON;  // required under the flag; the reference matches it
    mi->opt.enableflags |= mjENBL_IPC;
    mjModel* me = Load(s.xml);
    me->opt.integrator = mjINT_DISCRETE;
    me->opt.solver = mjSOL_NEWTON;
    mjData* di = mj_makeData(mi);
    mjData* de = mj_makeData(me);
    if (mi->nkey > 0) {
      mj_resetDataKeyframe(mi, di, 0);
      mj_resetDataKeyframe(me, de, 0);
    }
    for (int k = 0; k < N; k++) {
      mj_step(mi, di);
      mj_step(me, de);
    }
    // manifold qpos error (mj_differentiatePos -- never componentwise
    // quaternion subtraction) + qvel error
    mjtNum dq[64];
    mj_differentiatePos(mi, dq, 1.0, de->qpos, di->qpos);
    mjtNum eq = 0, ev = 0;
    for (int i = 0; i < mi->nv; i++) {
      mjtNum a = std::fabs(dq[i]);
      if (a > eq) eq = a;
      mjtNum b = std::fabs(di->qvel[i] - de->qvel[i]);
      if (b > ev) ev = b;
    }
    EXPECT_LT(eq, MjTol(1e-9, 1e-5))
        << s.name
        << ": qpos (manifold) IPC vs discrete should match to round-off";
    EXPECT_LT(ev, MjTol(1e-9, 1e-5))
        << s.name << ": qvel IPC vs discrete should match to round-off";
    mj_deleteData(di);
    mj_deleteModel(mi);
    mj_deleteData(de);
    mj_deleteModel(me);
  }
}

// A delayed sensor is sampled when the step advances, at the state the step
// started from, by every integrator: the IPC step commits its endpoint through
// the same advance, with the history work before the state and the effective
// acceleration published ahead of it, so a delayed position, velocity or
// acceleration sensor reads what it reads under the discrete integrator.
TEST_F(IpcTest, DelayedSensorsMatchDiscrete) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="discrete" solver="Newton"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
        <body name="tip" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
          <site name="tip" pos="0.3 0 0"/>
        </body>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="j1" delay="0.01" nsample="8"/>
      <jointvel joint="j2" delay="0.01" nsample="8"/>
      <accelerometer site="tip" delay="0.01" nsample="8"/>
      <framelinacc objtype="body" objname="tip" delay="0.01" nsample="8"/>
    </sensor>
  </mujoco>)";
  mjModel* mi = Load(xml);
  mi->opt.enableflags |= mjENBL_IPC;
  mjModel* me = Load(xml);
  mjData* di = mj_makeData(mi);
  mjData* de = mj_makeData(me);
  for (int k = 0; k < 100; k++) {
    mj_step(mi, di);
    mj_step(me, de);
  }
  for (int i = 0; i < mi->nsensordata; i++) {
    EXPECT_NEAR(di->sensordata[i], de->sensordata[i], MjTol(1e-9, 1e-5))
        << "sensordata[" << i << "] IPC vs discrete should match to round-off";
  }
  mj_deleteData(di);
  mj_deleteModel(mi);
  mj_deleteData(de);
  mj_deleteModel(me);
}

// With a 2D flex in the model the forward pass skips the constraint stage, so
// its acceleration-stage sensors see the free-flight acceleration; the step
// recomputes them from its own acceleration and constraint force before it
// commits. A sphere resting on the floor beside a pinned, non-colliding cloth
// reads gravity on its accelerometer after a step, as it does without the flag.
TEST_F(IpcTest, AccelerationSensorsSeeContact) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton"/>
    <worldbody>
      <geom type="plane" size="0 0 1"/>
      <body name="ball" pos="0 0 0.1">
        <freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="imu"/>
      </body>
      <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                radius="0.005" mass="0.05" pos="1 0 0.5">
        <contact contype="0" conaffinity="0"/>
        <pin id="0 1 2 3 4 5 6 7 8"/>
      </flexcomp>
    </worldbody>
    <sensor>
      <accelerometer site="imu"/>
      <framelinacc objtype="body" objname="ball"/>
    </sensor>
  </mujoco>)";
  mjModel* mi = Load(xml);
  mi->opt.enableflags |= mjENBL_IPC;
  mjModel* me = Load(xml);
  mjData* di = mj_makeData(mi);
  mjData* de = mj_makeData(me);
  for (int k = 0; k < 200; k++) {
    mj_step(mi, di);
    mj_step(me, de);
  }
  // resting: the accelerometer reads +g along its z axis
  EXPECT_NEAR(di->sensordata[2], 9.81, 0.1);
  // the two solves of the same contact row agree to their solver residuals
  for (int i = 0; i < mi->nsensordata; i++) {
    EXPECT_NEAR(di->sensordata[i], de->sensordata[i], MjTol(1e-4, 1e-3))
        << "sensordata[" << i << "] IPC vs discrete";
  }
  mj_deleteData(di);
  mj_deleteModel(mi);
  mj_deleteData(de);
  mj_deleteModel(me);
}

// CONTACT rung: rigid contact now reproduces MuJoCo's solref/solimp
// spring-damper (explicit aref when the timeconst >= 2h, the refsafe-safe
// regime), so a RIGID-ONLY scene through the IPC contact mode must track the
// EULER integrator of the same scene -- the fidelity property we are after. (A
// sub-2h timeconst switches to the stable implicit branch and is deliberately
// NOT Euler-faithful, since MuJoCo would have clamped it via refsafe.) The gap
// is LIVE through nonlinear FK -> no tunnel. Default solref timeconst here is
// 0.02 = 10h, well above 2h.
TEST_F(IpcTest, RigidContactMatchesEuler) {
  struct Scene {
    const char* name;
    const char* xml;
    int steps;
  };
  const Scene scenes[] = {
      {"sphere on floor", R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody><geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.1" mass="1" condim="1"/></body>
        </worldbody></mujoco>)",
       600},
      {"two spheres stack", R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody><geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.1" mass="1" condim="1"/></body>
          <body pos="0 0 0.33"><freejoint/><geom type="sphere" size="0.1" mass="1" condim="1"/></body>
        </worldbody></mujoco>)",
       800},
  };
  for (const Scene& s : scenes) {
    mjModel* mi = Load(s.xml);
    mi->opt.integrator = mjINT_DISCRETE;
    mi->opt.solver =
        mjSOL_NEWTON;  // required under the flag; the reference matches it
    mi->opt.enableflags |= mjENBL_IPC;
    mjModel* me = Load(s.xml);
    me->opt.integrator = mjINT_EULER;
    mjData* di = mj_makeData(mi);
    mjData* de = mj_makeData(me);
    for (int k = 0; k < s.steps; k++) {
      mj_step(mi, di);
      mj_step(me, de);
    }
    mjtNum dq[64];
    mj_differentiatePos(
        mi, dq, 1.0, de->qpos,
        di->qpos);  // manifold qpos diff (freejoint-quaternion safe)
    mjtNum eq = 0, ev = 0;
    for (int i = 0; i < mi->nv; i++) {
      mjtNum a = std::fabs(dq[i]);
      if (a > eq) eq = a;
      mjtNum b = std::fabs(di->qvel[i] - de->qvel[i]);
      if (b > ev) ev = b;
    }
    // bounds one decade above observed round-off accumulation (grew ~4x at the
    // port to the effective-metric base: the native contact pipeline under both
    // integrators moved). In single precision the stacked spheres are a
    // sensitive configuration: the two paths' round-off differs and the stack
    // settles differently, so the float bounds are loose (measured 0.8 mm, 7
    // mm/s)
    EXPECT_LT(eq, MjTol(1e-8, 1e-2))
        << s.name << ": IPC qpos should track Euler to round-off";
    EXPECT_LT(ev, MjTol(1e-7, 1e-1))
        << s.name << ": IPC qvel should track Euler to round-off";
    EXPECT_FALSE(std::isnan(di->qvel[2])) << s.name;
    mj_deleteData(di);
    mj_deleteModel(mi);
    mj_deleteData(de);
    mj_deleteModel(me);
  }
}

// GENERAL ARTICULATED contact: the penalty is driven by MuJoCo's own collision
// (mj_collision -- mj_step disables contact for the IPC predictor) and
// assembled through the FULL kinematic chain via mj_jac (b = J_B^T n - J_A^T
// n), with the gap LIVE through nonlinear FK. A capsule on a HINGE (its contact
// Jacobian runs through the joint, not a translation DOF) swings onto the floor
// and must NOT tunnel through it -- this is the humanoid-foot case in miniature
// (capsule geom + articulated body), which the old sphere-only / "first 3
// linear DOFs" path missed.
TEST_F(IpcTest, ArticulatedCapsuleDoesNotTunnel) {
  mjModel* m = Load(R"(
    <mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
      <worldbody><geom type="plane" size="3 3 0.1"/>
        <body pos="0 0 0.3"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.05" mass="1"/>
        </body>
      </worldbody></mujoco>)");
  mjData* d = mj_makeData(m);
  int gcap = m->body_geomadr[1];
  mjtNum worst = 1e9;  // lowest capsule SURFACE point over the run
  for (int s = 0; s < 700; s++) {
    mj_step(m, d);
    // capsule long axis = local z = xmat column 2; lowest surface z = center_z
    // - |halflen*axis_z| - radius
    mjtNum cz = d->geom_xpos[3 * gcap + 2], axz = d->geom_xmat[9 * gcap + 8];
    mjtNum lowest = cz - std::abs(m->geom_size[3 * gcap + 1] * axz) -
                    m->geom_size[3 * gcap];
    if (lowest < worst) worst = lowest;
    ASSERT_FALSE(std::isnan(d->qvel[0])) << "NaN at step " << s;
  }
  EXPECT_GT(worst, -0.03)
      << "capsule tunneled through the floor (lowest surface z = " << worst
      << ")";
  // it comes to rest propped on the floor (hinge can't fall through): the bob
  // is supported, qvel settles
  EXPECT_LT(std::abs(d->qvel[0]), 5e-2)
      << "hinge should settle against the floor";
  mj_deleteData(d);
  mj_deleteModel(m);
}

// FLEX-UNIFY 4b: articulated bodies participate in the UNIFIED flex solver
// vector with bit-exact kinetics. A hinge integrated in a flex scene (the
// unified path: its generalized DOFs are appended to the dense flex packing,
// share the one PCG + line search, with ih2*M / h2*M^-1 applied via
// mj_mulM/mj_solveM and a qdelta state) must match the SAME hinge alone (the
// mj_ipcTree path) to solver tolerance. A far, pinned cloth forces the unified
// path without any contact. This is the regression gate for the
// appended-articulated-kinetic machinery.
TEST_F(IpcTest, ArticulatedKineticInFlexMatchesTree) {
  mjModel* m1 = Load(R"(
    <mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
      <worldbody><body pos="0 0 1"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body></worldbody></mujoco>)");
  mjModel* m2 = Load(R"(
    <mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
      <worldbody><body pos="0 0 1"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body>
        <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05"
                  pos="3 0 1"><pin id="0 2 6 8"/></flexcomp>
      </worldbody></mujoco>)");
  ASSERT_GT(m2->nv, m1->nv)
      << "the flex scene must add DOFs (forces the unified path)";
  mjData* d1 = mj_makeData(m1);  // hinge alone -> mj_ipcTree
  mjData* d2 =
      mj_makeData(m2);  // hinge + far cloth -> unified flex path (na_artic==1)
  mjtNum worst = 0;
  for (int s = 0; s < 400; s++) {
    mj_step(m1, d1);
    mj_step(m2, d2);
    worst = std::max(
        worst, std::abs(d1->qpos[0] -
                        d2->qpos[0]));  // qpos[0] is the hinge angle in both
    ASSERT_FALSE(std::isnan(d2->qpos[0])) << "NaN at step " << s;
  }
  EXPECT_LT(worst, 1e-9)
      << "hinge in the unified flex path must match mj_ipcTree (max diff "
      << worst << ")";
  mj_deleteData(d1);
  mj_deleteModel(m1);
  mj_deleteData(d2);
  mj_deleteModel(m2);
}

// the flag's option fences: each incompatible setting is rejected at the start
// of the step, with the other settings valid
TEST_F(IpcTest, RejectsIncompatibleOptions) {
  mjModel* m = Load(R"(
    <mujoco><option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
      <worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>)");
  mjData* d = mj_makeData(m);
  auto forward_error = MjuErrorMessageFrom(mj_forward);
  EXPECT_EQ(forward_error(m, d), "");

  m->opt.integrator = mjINT_EULER;
  EXPECT_THAT(forward_error(m, d),
              testing::HasSubstr("flag ipc requires integrator='discrete'"));
  m->opt.integrator = mjINT_DISCRETE;

  m->opt.solver = mjSOL_CG;
  EXPECT_EQ(forward_error(m, d), "");
  m->opt.solver = mjSOL_PGS;
  EXPECT_THAT(forward_error(m, d),
              testing::HasSubstr("flag ipc requires solver='Newton' or 'CG'"));
  m->opt.solver = mjSOL_NEWTON;

  m->opt.enableflags |= mjENBL_FWDINV;
  EXPECT_THAT(forward_error(m, d),
              testing::HasSubstr("flag ipc does not support flag fwdinv"));
  m->opt.enableflags &= ~mjENBL_FWDINV;

  m->opt.enableflags |= mjENBL_SLEEP;
  EXPECT_THAT(forward_error(m, d),
              testing::HasSubstr("flag ipc does not support flag sleep"));
  m->opt.enableflags &= ~mjENBL_SLEEP;

  EXPECT_EQ(forward_error(m, d), "");
  mj_deleteData(d);
  mj_deleteModel(m);
}

// IPC inner solves run under either primal solver (Newton or CG)
TEST_F(IpcTest, NewtonAndCGAgreeOnContactStep) {
  constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" iterations="200" tolerance="1e-10">
      <flag ipc="enable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1" pos="0 0 0"/>
      <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                radius="0.005" mass="0.05" pos="0 0 0.004">
        <elasticity young="1e4" damping="0.01" thickness="0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>)";
  mjModel* mn = Load(kXml);
  mjModel* mc = Load(kXml);
  ASSERT_THAT(mn, NotNull());
  ASSERT_THAT(mc, NotNull());
  mn->opt.solver = mjSOL_NEWTON;
  mc->opt.solver = mjSOL_CG;
  mjData* dn = mj_makeData(mn);
  mjData* dc = mj_makeData(mc);
  for (int s = 0; s < 5; s++) {
    mj_step(mn, dn);
    mj_step(mc, dc);
  }
  for (int i = 0; i < mn->nq; i++) {
    EXPECT_NEAR(dn->qpos[i], dc->qpos[i], MjTol(1e-6, 1e-4));
  }
  mj_deleteData(dc);
  mj_deleteData(dn);
  mj_deleteModel(mc);
  mj_deleteModel(mn);
}

// under the flag the collision pipeline does not generate the contacts the
// IPC step handles (dim-2 flex against static geometry, dim-2 flex against
// dim-2 flex), while flex against a moving body stays: a sheet resting on a
// static box with a ball on top
TEST_F(IpcTest, NativeRowsExcludedForIpcPairs) {
  mjModel* m = Load(R"(
    <mujoco><option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
      <worldbody>
        <geom name="table" type="box" size=".3 .3 .05" pos="0 0 .05"/>
        <flexcomp name="sheet" type="grid" dim="2" count="8 8 1" spacing=".04 .04 1" radius=".004"
                  mass=".05" pos="0 0 .13">
          <edge equality="true" solref=".004 1"/>
          <contact selfcollide="auto" contype="1" conaffinity="1" solref=".004 1"/>
        </flexcomp>
        <body pos="0 0 .25"><freejoint/><geom type="sphere" size=".03" mass=".2"/></body>
      </worldbody></mujoco>)");
  mjData* d = mj_makeData(m);
  int flex_static = 0, flex_moving = 0;
  for (int s = 0; s < 400; s++) {
    mj_step(m, d);
    for (int i = 0; i < d->ncon; i++) {
      const mjContact* c = d->contact + i;
      int fs = (c->flex[0] >= 0) != (c->flex[1] >= 0);  // exactly one flex side
      if (!fs) continue;
      int g = c->flex[0] >= 0 ? c->geom[1] : c->geom[0];
      int is_static = g >= 0 && m->body_weldid[m->geom_bodyid[g]] == 0;
      if (is_static)
        flex_static++;
      else if (c->exclude == 0)
        flex_moving++;
    }
  }
  EXPECT_EQ(flex_static, 0)
      << "sheet-vs-table contacts are the IPC step's: none is generated";
  EXPECT_GT(flex_moving, 0) << "ball-vs-sheet stays on the constraint solver";
  mj_deleteData(d);
  mj_deleteModel(m);
}

// geom types the IPC step cannot detect keep their native rows, so the sheet
// still rests on them
TEST_F(IpcTest, NativeRowsKeptForUnsupportedGeoms) {
  static constexpr char kModel[] = R"(
    <mujoco><option timestep="0.002" integrator="discrete" solver="Newton"><flag ipc="enable"/></option>
      <asset><hfield name="hf" nrow="2" ncol="2" elevation="1 1 1 1" size=".2 .2 .1 .1"/></asset>
      <worldbody>
        <geom name="drum" type="%s" %s/>
        <flexcomp name="sheet" type="grid" dim="2" count="9 9 1" spacing=".03 .03 1" radius=".003"
                  mass=".05" pos="0 0 .5">
          <edge equality="true" solref=".004 1"/>
          <contact selfcollide="auto" contype="1" conaffinity="1" solref=".004 1"/>
        </flexcomp>
      </worldbody></mujoco>)";
  const char* geoms[][2] = {{"hfield", "hfield=\"hf\" pos=\"0 0 .3\""},
                            {"cylinder", "size=\".2 .1\" pos=\"0 0 .2\""},
                            {"ellipsoid", "size=\".2 .2 .1\" pos=\"0 0 .2\""},
                            {"box", "size=\".2 .2 .1\" pos=\"0 0 .2\""}};
  for (const auto& g : geoms) {
    char xml[1024];
    snprintf(xml, sizeof(xml), kModel, g[0], g[1]);
    mjModel* m = Load(xml);
    mjData* d = mj_makeData(m);
    int native = 0;
    for (int s = 0; s < 200; s++) {
      mj_step(m, d);
      native += d->ncon;
    }
    mjtNum zmin = 1;
    for (int v = 0; v < m->nflexvert; v++)
      zmin = mju_min(zmin, d->flexvert_xpos[3 * v + 2]);
    bool supported = mjc_GeomSupported(m->geom_type[0]);
    EXPECT_EQ(native > 0, !supported)
        << g[0] << ": native contacts only for unsupported types";
    EXPECT_GT(zmin, 0.25) << g[0] << ": the sheet rests on top";
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

// mj_effMulAdd's measured stack use stays within mj_effMulAddScratch on both
// matrix-free fallback paths: a bending-only 2D flex and a 3D trilinear
// interpolated flex parented to a hinged body, neither assembled into efm_K
TEST_F(IpcTest, MulAddScratchBounds) {
  static constexpr char kBendStretch[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep=".002"/>
    <worldbody>
      <flexcomp name="sheet" type="grid" dim="2" count="4 4 1"
                spacing=".05 .05 1" radius=".005" mass=".1">
        <contact selfcollide="none"/>
        <elasticity young="1e4" thickness="1e-2" elastic2d="bend"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  static constexpr char kInterp[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep=".002"/>
    <worldbody>
      <body name="root" pos="0 0 .5">
        <joint type="hinge"/>
        <geom type="sphere" size=".02" mass=".1"/>
        <flexcomp name="solid" type="grid" dim="3" dof="trilinear"
                  count="4 4 4" spacing=".05 .05 .05" radius=".005" mass=".1">
          <contact selfcollide="none"/>
          <elasticity young="1e4"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml : {kBendStretch, kInterp}) {
    mjModel* m = Load(xml);
    ASSERT_THAT(m, NotNull());
    mjData* d = mj_makeData(m);
    mj_forward(m, d);
    ASSERT_TRUE(d->efm_active);
    EXPECT_EQ(d->nefmK, 0);

    size_t budget = mj_effMulAddScratch(m, d);
    EXPECT_GT(budget, 0u);

    std::vector<mjtNum> vec(m->nv, 1.0), res(m->nv, 0.0);
    size_t parena0 = d->parena, maxarena0 = d->maxuse_arena;
    ASSERT_EQ(d->pstack, 0u);
    ASSERT_LE(parena0 + budget, d->narena);

    // cap the free stack at budget bytes: any unbudgeted allocation overflows
    d->parena = d->narena - budget;
    d->maxuse_stack = 0;
    mj_effMulAdd(m, d, res.data(), vec.data(), /*flg_contact=*/1);
    d->parena = parena0;
    d->maxuse_arena = maxarena0;

    EXPECT_GT(d->maxuse_stack, 0u);
    EXPECT_LE(d->maxuse_stack, budget);
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

//--------------------------- sparse metric factor ----------------------------

// two 2D sheets that collide (the lower one pinned), under the IPC step
static constexpr char kSheets[] = R"(
<mujoco>
  <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
    <flag ipc="enable"/>
  </option>
  <worldbody>
    <flexcomp name="lower" type="grid" dim="2" count="9 9 1" spacing=".04 .04 1"
              radius=".004" mass=".3" pos="0 0 .2">
      <contact selfcollide="auto"/>
      <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
      <pin id="0 8 72 80"/>
    </flexcomp>
    <flexcomp name="upper" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
              radius=".004" mass=".1" pos="0 0 .27">
      <contact selfcollide="auto"/>
      <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

// one pinned sheet with no contact
static constexpr char kHanging[] = R"(
<mujoco>
  <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
    <flag ipc="enable"/>
  </option>
  <worldbody>
    <flexcomp name="sheet" type="grid" dim="2" count="9 9 1" spacing=".04 .04 1"
              radius=".004" mass=".3" pos="0 0 .5">
      <contact selfcollide="none"/>
      <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
      <pin id="0 8"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

// the last step analysed, reserved and factored the sparse metric factor
static bool FactorUsed(const mjData* d) {
  mjtSize nb = d->nefmdof;
  return nb && d->nefmL > 9 * nb && d->efm_L[9 * nb] == 1;
}

// the last step used the 3x3 blocks only
static bool BlocksUsed(const mjData* d) {
  return d->nefmdof && d->nefmL == 9 * d->nefmdof;
}

// the vertices in the model's ordering
static int OrderedVertices(const mjModel* m) {
  int n = 0;
  for (int t = 0; t < m->nefmCvert; t++) {
    n += m->efmC_perm[t] >= 0;
  }
  return n;
}

// a copy of m without the ordering, as mj_setConst leaves it when it cannot
// order: steps use the blocks
static mjModel* BlocksCopy(const mjModel* m) {
  mjModel* mb = mj_copyModel(nullptr, m);
  for (int t = 0; t < mb->nefmCvert; t++) {
    mb->efmC_perm[t] = -1;
  }
  return mb;
}

static bool SameState(const mjModel* m, const mjData* a, const mjData* b) {
  return !memcmp(a->qpos, b->qpos, sizeof(mjtNum) * m->nq) &&
         !memcmp(a->qvel, b->qvel, sizeof(mjtNum) * m->nv) &&
         !memcmp(a->qacc, b->qacc, sizeof(mjtNum) * m->nv);
}

// the sparse factor and the 3x3 blocks precondition the same solves: from the
// same state, one step gives the same accelerations up to the solver tolerances
TEST_F(IpcTest, SparseFactorMatchesBlocks) {
  // converge the constraint solve rather than cap it
  std::string xml(kSheets);
  size_t pos = xml.find("iterations=\"400\"");
  xml.replace(pos, std::string("iterations=\"400\"").size(),
              "iterations=\"4000\" tolerance=\"1e-12\"");
  mjModel* m = Load(xml.c_str());
  ASSERT_THAT(m, NotNull());
  ASSERT_GT(OrderedVertices(m), 0);
  mjModel* mb = BlocksCopy(m);
  mjData* d = mj_makeData(m);
  mjData* db = mj_makeData(mb);

  bool contact = false;
  mjtNum rel = 0, rel_smooth = 0;
  for (int s = 0; s < 150; s++) {
    mj_copyData(db, mb, d);
    mj_step(m, d);
    mj_step(mb, db);
    ASSERT_TRUE(FactorUsed(d)) << "step " << s;
    ASSERT_TRUE(BlocksUsed(db)) << "step " << s;
    contact = contact || mju_norm(d->qfrc_constraint, m->nv) > 0;

    mjtNum scale = 0, diff = 0, scale_s = 0, diff_s = 0;
    for (int i = 0; i < m->nv; i++) {
      scale = mju_max(scale, mju_abs(db->qacc[i]));
      diff = mju_max(diff, mju_abs(d->qacc[i] - db->qacc[i]));
      scale_s = mju_max(scale_s, mju_abs(db->qacc_smooth[i]));
      diff_s = mju_max(diff_s, mju_abs(d->qacc_smooth[i] - db->qacc_smooth[i]));
    }
    rel = mju_max(rel, diff / scale);
    rel_smooth = mju_max(rel_smooth, diff_s / scale_s);
  }
  EXPECT_LE(rel_smooth, MjTol(2e-11, 2e-4)) << "qacc_smooth";
  EXPECT_LE(rel, MjTol(1e-4, 2e-4)) << "qacc";
  EXPECT_TRUE(contact) << "the sheets must collide";

  mj_deleteData(db);
  mj_deleteData(d);
  mj_deleteModel(mb);
  mj_deleteModel(m);
}

// for a 2D flex the factor applies whatever the IPC flag, cone and solver: it
// serves qacc_smooth, and CG preconditions with folded blocks of its own
TEST_F(IpcTest, SparseFactorFollowsIpcFlagAndSolver) {
  mjModel* m = Load(kHanging);
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));
  m->opt.enableflags &= ~mjENBL_IPC;
  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));
  m->opt.cone = mjCONE_ELLIPTIC;
  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));
  m->opt.cone = mjCONE_PYRAMIDAL;
  m->opt.solver = mjSOL_CG;
  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));
  m->opt.solver = mjSOL_NEWTON;
  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));

  mj_deleteData(d);
  mj_deleteModel(m);
}

// the ordering survives a binary save and load
TEST_F(IpcTest, SparseFactorSaveLoad) {
  mjModel* m = Load(kSheets);
  ASSERT_THAT(m, NotNull());
  ASSERT_GT(OrderedVertices(m), 0);
  std::vector<char> buffer(mj_sizeModel(m));
  mj_saveModel(m, nullptr, buffer.data(), buffer.size());
  mjModel* ml = mj_loadModelBuffer(buffer.data(), buffer.size());
  ASSERT_THAT(ml, NotNull());
  EXPECT_EQ(ml->nefmCvert, m->nefmCvert);
  EXPECT_EQ(memcmp(ml->efmC_perm, m->efmC_perm, sizeof(int) * m->nefmCvert), 0);

  mjData* d = mj_makeData(m);
  mjData* dl = mj_makeData(ml);
  for (int s = 0; s < 50; s++) {
    mj_step(m, d);
    mj_step(ml, dl);
  }
  EXPECT_TRUE(FactorUsed(dl));
  EXPECT_TRUE(SameState(m, d, dl));

  mj_deleteData(dl);
  mj_deleteData(d);
  mj_deleteModel(ml);
  mj_deleteModel(m);
}

// a model compiled without the IPC flag has the ordering too: CG without IPC
// uses the factor, and so does enabling the flag at run time
TEST_F(IpcTest, SparseFactorWithoutCompileTimeIpc) {
  std::string xml(kHanging);
  xml.erase(xml.find("<flag ipc=\"enable\"/>"),
            std::string("<flag ipc=\"enable\"/>").size());
  mjModel* m = Load(xml.c_str());
  ASSERT_THAT(m, NotNull());
  EXPECT_GT(OrderedVertices(m), 0);
  mjData* d = mj_makeData(m);
  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));
  m->opt.enableflags |= mjENBL_IPC;
  mj_step(m, d);
  EXPECT_TRUE(FactorUsed(d));
  EXPECT_FALSE(d->warning[mjWARN_BADQACC].number);
  mj_deleteData(d);
  mj_deleteModel(m);
}

// each model carries its own ordering: two models step interleaved, and a copy
// outlives its source, each matching a freshly compiled reference bitwise
TEST_F(IpcTest, SparseFactorModelCopies) {
  mjModel* a = Load(kSheets);
  mjModel* b = Load(kHanging);
  ASSERT_THAT(a, NotNull());
  ASSERT_THAT(b, NotNull());
  ASSERT_GT(OrderedVertices(a), 0);
  ASSERT_GT(OrderedVertices(b), 0);
  mjModel* c = mj_copyModel(nullptr, a);

  // the copy owns an equal ordering
  EXPECT_EQ(c->nefmCvert, a->nefmCvert);
  EXPECT_NE(c->efmC_perm, a->efmC_perm);
  EXPECT_EQ(memcmp(c->efmC_perm, a->efmC_perm, sizeof(int) * a->nefmCvert), 0);
  mj_deleteModel(a);

  mjModel* ra = Load(kSheets);
  mjModel* rb = Load(kHanging);
  mjData* dc = mj_makeData(c);
  mjData* db = mj_makeData(b);
  mjData* dra = mj_makeData(ra);
  mjData* drb = mj_makeData(rb);
  for (int s = 0; s < 100; s++) {
    mj_step(c, dc);
    mj_step(b, db);
  }
  for (int s = 0; s < 100; s++) {
    mj_step(ra, dra);
  }
  for (int s = 0; s < 100; s++) {
    mj_step(rb, drb);
  }
  EXPECT_TRUE(FactorUsed(dc));
  EXPECT_TRUE(FactorUsed(db));
  EXPECT_TRUE(SameState(c, dc, dra));
  EXPECT_TRUE(SameState(b, db, drb));

  mj_deleteData(drb);
  mj_deleteData(dra);
  mj_deleteData(db);
  mj_deleteData(dc);
  mj_deleteModel(rb);
  mj_deleteModel(ra);
  mj_deleteModel(c);
  mj_deleteModel(b);
}

// an arena too small for the factor, but large enough for the blocks: the
// blocks, with a warning, and the blocks' results
TEST_F(IpcTest, SparseFactorSmallArena) {
  // a larger sheet: the factor's arena use grows faster than the blocks'
  std::string xml(kHanging);
  xml.replace(xml.find("count=\"9 9 1\""),
              std::string("count=\"9 9 1\"").size(), "count=\"15 15 1\"");
  xml.replace(xml.find("<pin id=\"0 8\"/>"),
              std::string("<pin id=\"0 8\"/>").size(), "<pin id=\"0 14\"/>");
  mjModel* m = Load(xml.c_str());
  ASSERT_THAT(m, NotNull());
  ASSERT_GT(OrderedVertices(m), 0);
  mjModel* mb = BlocksCopy(m);

  // peak arena use of each path (without the ASAN red zones)
  auto peak = [](const mjModel* model, bool factor) {
    mjData* data = mj_makeData(model);
    for (int s = 0; s < 20; s++) {
      mj_step(model, data);
    }
    EXPECT_TRUE(factor ? FactorUsed(data) : BlocksUsed(data));
    size_t maxuse = data->maxuse_arena;
    mj_deleteData(data);
    return maxuse;
  };
  size_t blocks = peak(mb, false);
  size_t factor = peak(m, true);
  ASSERT_GT(factor, blocks);

  mock_warning_handler.ExpectWarnings("Insufficient arena memory");
  m->narena = mb->narena = blocks + (factor - blocks) / 3;
  mjData* d = mj_makeData(m);
  mjData* db = mj_makeData(mb);
  for (int s = 0; s < 20; s++) {
    mj_step(m, d);
    mj_step(mb, db);
    ASSERT_TRUE(BlocksUsed(d)) << "step " << s;
  }
  EXPECT_GT(d->warning[mjWARN_CNSTRFULL].number, 0);
  EXPECT_FALSE(db->warning[mjWARN_CNSTRFULL].number);
  EXPECT_TRUE(SameState(m, d, db));

  mj_deleteData(db);
  mj_deleteData(d);
  mj_deleteModel(mb);
  mj_deleteModel(m);
}

// threads stepping their own mjData on one model match a serial run
TEST_F(IpcTest, SparseFactorThreads) {
  mjModel* m = Load(kSheets);
  ASSERT_THAT(m, NotNull());
  constexpr int kThreads = 4;
  constexpr int kSteps = 60;

  mjData* ref = mj_makeData(m);
  for (int s = 0; s < kSteps; s++) {
    mj_step(m, ref);
  }

  std::vector<mjData*> d(kThreads);
  for (int t = 0; t < kThreads; t++) {
    d[t] = mj_makeData(m);
  }
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([m, dt = d[t]]() {
      for (int s = 0; s < kSteps; s++) {
        mj_step(m, dt);
      }
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }
  for (int t = 0; t < kThreads; t++) {
    EXPECT_TRUE(FactorUsed(d[t])) << "thread " << t;
    EXPECT_TRUE(SameState(m, d[t], ref)) << "thread " << t;
    mj_deleteData(d[t]);
  }

  mj_deleteData(ref);
  mj_deleteModel(m);
}

// a stiffness edited at run time changes the efm_K pattern: the next step
// analyses the new pattern under the model's ordering, restricted to the
// remaining vertices, silently
TEST_F(IpcTest, SparseFactorPatternChange) {
  // the upper sheet's stencils are its stretch elements only
  std::string xml(kSheets);
  size_t upper = xml.find("name=\"upper\"");
  size_t pos = xml.find("elastic2d=\"both\"", upper);
  xml.replace(pos, std::string("elastic2d=\"both\"").size(),
              "elastic2d=\"stretch\"");
  mjModel* m = Load(xml.c_str());
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);
  mj_step(m, d);
  ASSERT_TRUE(FactorUsed(d));
  int nb = d->nefmdof;

  // without its stretch stiffness, the upper sheet leaves efm_K
  m->flex_stiffness[m->flex_stiffnessadr[1]] = 0;
  for (int s = 0; s < 3; s++) {
    mj_step(m, d);
    EXPECT_TRUE(FactorUsed(d)) << "step " << s;
    EXPECT_LT(d->nefmdof, nb);
  }
  int nw = 0;
  for (int i = 0; i < mjNWARNING; i++) {
    nw += d->warning[i].number;
  }
  EXPECT_EQ(nw, 0);

  mj_deleteData(d);
  mj_deleteModel(m);
}

// the coverage rules: the compiler sizes the ordering for models with a 2D flex
// or an elastic rod, mj_setConst orders the vertices of the engine's own
// stencils, and models with a covered 3D flex use the blocks for the metric. A
// 2D flex uses the metric's sparse factor under Newton, and the 3x3 blocks
// under CG
TEST_F(IpcTest, SparseFactorCoverage) {
  static constexpr char kTemplate[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" iterations="400" %s>
      %s
    </option>
    <worldbody>
      %s
    </worldbody>
  </mujoco>
  )";
  static constexpr char kCG[] = R"(solver="CG")";
  static constexpr char kNewton[] = R"(solver="Newton")";
  static constexpr char kIpc[] = R"(<flag ipc="enable"/>)";
  static constexpr char kSheet[] = R"(
      <flexcomp name="sheet" type="grid" dim="2" count="7 6 1" spacing=".04 .04 1"
                radius=".004" mass=".3" pos="0 0 .5">
        <contact selfcollide="none"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="%s"/>
        %s
      </flexcomp>)";
  static constexpr char kSecond[] = R"(
      <flexcomp name="second" type="grid" dim="2" count="4 5 1" spacing=".04 .04 1"
                radius=".004" mass=".1" pos="1 0 .5">
        <contact selfcollide="none"/>
        <elasticity young="1e4" poisson=".3" thickness="2e-3"/>
      </flexcomp>)";
  static constexpr char kCable[] = R"(
      <flexcomp name="cable" type="grid" dim="1" count="8 1 1" spacing=".04 1 1"
                radius=".004" mass=".1" pos="0 1 .5">
        <contact selfcollide="none"/>
      </flexcomp>)";
  static constexpr char kRod[] = R"(
      <flexcomp name="rod" type="grid" dim="1" count="8 1 1" spacing=".04 1 1"
                radius=".004" mass=".1" pos="0 1 .5">
        <contact selfcollide="none"/>
        <elasticity young="1e5"/>
      </flexcomp>)";
  static constexpr char kSolid[] = R"(
      <flexcomp name="solid" type="grid" dim="3" count="3 3 3" spacing=".04 .04 .04"
                radius=".004" mass=".1" pos="0 -1 .5">
        <contact selfcollide="none"/>
        <elasticity young="1e4" poisson=".2"/>
      </flexcomp>)";
  static constexpr char kHinge[] = R"(
      <body pos="0 2 1"><joint type="hinge"/><geom size=".05"/></body>)";

  auto sheet = [&](const char* elastic2d, const char* pin) {
    char buf[1024];
    snprintf(buf, sizeof(buf), kSheet, elastic2d, pin);
    return std::string(buf);
  };
  struct Case {
    std::string name;
    std::string attrs;
    std::string option;
    std::string bodies;
    bool sized;  // the model has ordering slots
    bool used;   // the step uses the factor (efm_K is assembled)
  };
  std::vector<Case> cases = {
      {"stretch and bending", kNewton, kIpc, sheet("both", ""), true, true},
      {"pins", kNewton, kIpc, sheet("both", R"(<pin id="0 6 35 41"/>)"), true,
       true},
      {"stretch only", kNewton, kIpc, sheet("stretch", ""), true, true},
      {"bending only", kNewton, kIpc, sheet("bend", ""), true, true},
      {"two sheets", kNewton, kIpc, sheet("both", "") + kSecond, true, true},
      {"joint before the sheet", kNewton, kIpc,
       std::string(kHinge) + sheet("both", ""), true, true},
      {"sheet and cable", kNewton, kIpc, sheet("both", "") + kCable, true,
       true},
      {"sheet and elastic rod", kNewton, kIpc, sheet("both", "") + kRod, true,
       true},
      {"elastic rod, no IPC", kCG, "", kRod, true, true},
      {"sheet and elastic rod, no IPC", kCG, "", sheet("both", "") + kRod, true,
       true},
      {"sheet and solid", kNewton, kIpc, sheet("both", "") + kSolid, true,
       false},
      {"solid", kNewton, kIpc, kSolid, true, false},
      {"no IPC, CG", kCG, "", sheet("both", ""), true, true},
      {"no IPC, two sheets, CG", kCG, "", sheet("both", "") + kSecond, true,
       true},
      {"no IPC, Newton", kNewton, "", sheet("both", ""), true, true},
      {"no IPC, Newton elliptic", R"(solver="Newton" cone="elliptic")", "",
       sheet("both", ""), true, true},
      {"no IPC, elliptic", R"(solver="CG" cone="elliptic")", "",
       sheet("both", ""), true, true},
      {"no IPC, sleep", kCG, R"(<flag sleep="enable"/>)", sheet("both", ""),
       true, true},
      {"elastic rod, no IPC, Newton", kNewton, "", kRod, true, true},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    char xml[4096];
    snprintf(xml, sizeof(xml), kTemplate, c.attrs.c_str(), c.option.c_str(),
             c.bodies.c_str());
    mjModel* m = Load(xml);
    ASSERT_THAT(m, NotNull());
    EXPECT_EQ(m->nefmCvert > 0, c.sized);
    mjData* d = mj_makeData(m);
    for (int s = 0; s < 3; s++) {
      mj_step(m, d);
    }
    if (c.used) {
      EXPECT_TRUE(FactorUsed(d));
      // the model's ordering holds exactly the step's covered vertices
      EXPECT_EQ(OrderedVertices(m), d->nefmdof);
      // the factor holds at least the covered matrix's lower block triangles
      EXPECT_GE(d->nefmL, 9 * d->nefmdof + 6 * d->nefmdof);
    } else {
      EXPECT_FALSE(FactorUsed(d));
    }
    EXPECT_FALSE(d->warning[mjWARN_BADQACC].number);
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

// each budgeted frame (mj_effCholSetConst and effCholReserve/effCholEnsure)
// fits within the exact narena where its avail >= need gate first passes
TEST_F(IpcTest, SparseFactorScratchBounds) {
  mock_warning_handler.ExpectWarnings("Insufficient arena memory");
  auto min_arena = [](mjModel* m, size_t lo, auto passes) {
    size_t orig = m->narena;
    size_t hi = orig;
    while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      m->narena = mid;
      mjData* d = mj_makeData(m);
      bool ok = passes(m, d);
      EXPECT_LE(d->maxuse_arena, d->narena);
      mj_deleteData(d);
      if (ok) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    m->narena = lo;
    mjData* d = mj_makeData(m);
    EXPECT_TRUE(passes(m, d));
    EXPECT_EQ(d->warning[mjWARN_CNSTRFULL].number, 0);
    EXPECT_LE(d->maxuse_arena, d->narena);
    mj_deleteData(d);
    m->narena = orig;
  };

  for (const char* xml : {kSheets, kHanging}) {
    mjModel* m = Load(xml);
    ASSERT_THAT(m, NotNull());
    mjModel* mb = BlocksCopy(m);
    mjData* db = mj_makeData(mb);
    for (int s = 0; s < 15; s++) {
      mj_step(mb, db);
    }
    size_t blocks_peak = db->maxuse_arena;
    mj_deleteData(db);
    mj_deleteModel(mb);

    // 1. mj_effCholSetConst: exact minimum arena for ordering the stencils
    min_arena(m, 64, [](mjModel* model, mjData* d) {
      std::fill_n(model->efmC_perm, model->nefmCvert, -1);
      mj_effCholSetConst(model, d);
      return OrderedVertices(model) > 0 &&
             d->warning[mjWARN_CNSTRFULL].number == 0;
    });

    // 2. effCholReserve + effCholEnsure + effCholApply: exact minimum arena
    // for reserving, factoring, and applying the step's sparse factor
    min_arena(m, blocks_peak, [](mjModel* model, mjData* d) {
      mj_forward(model, d);
      if (!FactorUsed(d) || d->warning[mjWARN_CNSTRFULL].number) {
        return false;
      }
      std::vector<mjtNum> b(model->nv, 1), x(model->nv, 0);
      mj_effPrec(model, d, x.data(), b.data());
      return d->efm_L[9 * d->nefmdof] == 1 &&
             d->warning[mjWARN_CNSTRFULL].number == 0;
    });

    // 3. sweep narena across the full mj_step solve path from blocks_peak up to
    // the sparse-factor peak: warnings are expected, but no arena overflow
    mjData* ref = mj_makeData(m);
    for (int s = 0; s < 15; s++) {
      mj_step(m, ref);
    }
    ASSERT_TRUE(FactorUsed(ref));
    mjData* ds = mj_makeData(m);
    for (int s = 0; s < 15; s++) {
      mj_step(m, ds);
    }
    size_t overhead = mj_stackBytes(1, 1) - 1;
    size_t lo_step = blocks_peak + 128 * overhead;
    size_t hi_step = ds->maxuse_arena + 128 * overhead;
    mj_deleteData(ds);
    size_t orig = m->narena;
    for (int k = 0; k <= 32; k++) {
      m->narena =
          lo_step + (hi_step > lo_step ? ((hi_step - lo_step) * k) / 32 : 0);
      mjData* d = mj_makeData(m);
      mju_copy(d->qpos, ref->qpos, m->nq);
      mju_copy(d->qvel, ref->qvel, m->nv);
      mj_step(m, d);
      EXPECT_LE(d->maxuse_arena, d->narena);
      mj_deleteData(d);
    }
    m->narena = orig;

    mj_deleteData(ref);
    mj_deleteModel(m);
  }
}

TEST_F(IpcTest, SingleAndMultiThreadedMatch) {
  static constexpr char kXml[] = R"(
    <mujoco>
      <option timestep="0.002" integrator="discrete" solver="Newton"
              iterations="50" tolerance="1e-10">
        <flag ipc="enable"/>
      </option>
      <size memory="16M"/>
      <worldbody>
        <geom type="plane" size="2 2 0.05"/>
        <geom type="box" pos="0 0 0.04" size="0.12 0.12 0.04"/>
        <flexcomp name="s1" type="grid" dim="2" count="6 6 1"
                  spacing="0.03 0.03 0.03" pos="0 0 0.081"
                  radius="0.004" mass="0.1">
          <contact selfcollide="auto"/>
          <elasticity young="5e3" poisson="0.3" thickness="0.002"/>
        </flexcomp>
        <flexcomp name="s2" type="grid" dim="2" count="6 6 1"
                  spacing="0.03 0.03 0.03" pos="0.01 -0.01 0.083"
                  radius="0.004" mass="0.1">
          <contact selfcollide="auto"/>
          <elasticity young="5e3" poisson="0.3" thickness="0.002"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  )";
  mjModel* m = Load(kXml);
  mjModel* mt = Load(kXml);
  mjData* d = mj_makeData(m);
  mjData* dt = mj_makeData(mt);
  mju_threadpool(dt, 4);

  for (int s = 0; s < 25; s++) {
    mj_step(m, d);
    mj_step(mt, dt);
  }
  EXPECT_GT(mju_norm(d->qfrc_constraint, m->nv), 0);
  EXPECT_EQ(std::memcmp(d->qpos, dt->qpos, sizeof(mjtNum) * m->nq), 0);
  EXPECT_EQ(std::memcmp(d->qvel, dt->qvel, sizeof(mjtNum) * m->nv), 0);
  EXPECT_EQ(std::memcmp(d->qacc, dt->qacc, sizeof(mjtNum) * m->nv), 0);
  EXPECT_EQ(std::memcmp(d->qfrc_constraint, dt->qfrc_constraint,
                        sizeof(mjtNum) * m->nv),
            0);
  EXPECT_EQ(std::memcmp(d->flexvert_lambda, dt->flexvert_lambda,
                        sizeof(mjtNum) * m->nflexvert),
            0);
  EXPECT_EQ(std::memcmp(d->flexvert_conage, dt->flexvert_conage,
                        sizeof(int) * m->nflexvert),
            0);

  // Block-pool overflow in threaded mjc_candidates still merges the collected
  // blocks onto the arena instead of discarding all candidates.
  mock_warning_handler.ExpectWarnings("IPC: arena full");
  mock_warning_handler.ExpectWarnings("Too many contacts");
  int nfv = mt->nflexvert;
  std::vector<mjtNum> radii(nfv, mt->flex_radius[0]);
  std::vector<int> fidx(nfv), pt2flex(nfv);
  int flist[2] = {0, 1};
  int fxadr[2] = {0, mt->flex_vertnum[0]};
  for (int v = 0; v < nfv; v++) {
    fidx[v] = v;
    pt2flex[v] = v >= fxadr[1] ? 1 : 0;
  }
  size_t saved_parena = dt->parena;
  mjcFlexPair* cand_full = nullptr;
  int nfull = mjc_candidates(
      mt, dt, dt->flexvert_xpos, nullptr, nullptr, nullptr, nullptr, 0, 0,
      radii.data(), 0.2, 0.2, dt->flexvert_xpos, dt->flexvert_xpos, 0.05, nfv,
      nfv, fidx.data(), nullptr, flist, fxadr, 2, pt2flex.data(), &cand_full);
  dt->parena = saved_parena;
  EXPECT_GT(nfull, 128);

  mj_markStack(dt);
  size_t keep = 16384;
  size_t avail = dt->narena - dt->pstack - dt->parena;
  if (avail > keep) {
    mj_stackAllocByte(dt, avail - keep, 1);
  }
  mjcFlexPair* cand_part = nullptr;
  int npart = mjc_candidates(
      mt, dt, dt->flexvert_xpos, nullptr, nullptr, nullptr, nullptr, 0, 0,
      radii.data(), 0.2, 0.2, dt->flexvert_xpos, dt->flexvert_xpos, 0.05, nfv,
      nfv, fidx.data(), nullptr, flist, fxadr, 2, pt2flex.data(), &cand_part);
  mj_freeStack(dt);
  dt->parena = saved_parena;
  EXPECT_GT(npart, 0);
  EXPECT_LT(npart, nfull);
  EXPECT_NE(cand_part, nullptr);

  mj_deleteData(dt);
  mj_deleteModel(mt);
  mj_deleteData(d);
  mj_deleteModel(m);
}

// The active-set merge identifies a pair by role-aware canonical indices: a
// permutation within a role is the same pair, while distinct pairs sharing an
// index multiset (e.g. geom edge 1 vs flex edge (2,3) and geom edge 2 vs flex
// edge (1,3)) are kept apart.
TEST_F(IpcTest, MergeActiveSetDedupsByPairIdentity) {
  mjModel* m = Load("<mujoco/>");
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  auto count = [&](mjcFlexPair a, mjcFlexPair b) {
    mjcFlexPair pairs[2] = {a, b};
    return mj_ipcCountDistinct(d, pairs, 2, nullptr, 0);
  };

  // geom edge vs flex edge: {feature, a, b, 0}
  EXPECT_EQ(count({mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 0},
                  {mjcGEOM_EDGE_EDGE, {2, 1, 3, 0}, 0}),
            2);
  EXPECT_EQ(count({mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 0},
                  {mjcGEOM_EDGE_EDGE, {1, 3, 2, 0}, 0}),
            1);
  EXPECT_EQ(count({mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 0},
                  {mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 1}),
            2);

  // geom corner vs flex triangle: {feature, A, B, C}
  EXPECT_EQ(count({mjcGEOM_CORNER_TRI, {0, 1, 2, 3}, 0},
                  {mjcGEOM_CORNER_TRI, {1, 0, 2, 3}, 0}),
            2);
  EXPECT_EQ(count({mjcGEOM_CORNER_TRI, {0, 1, 2, 3}, 0},
                  {mjcGEOM_CORNER_TRI, {0, 3, 1, 2}, 0}),
            1);

  // flex vertex vs flex triangle: {v, A, B, C}
  EXPECT_EQ(count({mjcFLEX_VERT_TRI, {1, 2, 3, 4}, -1},
                  {mjcFLEX_VERT_TRI, {2, 1, 3, 4}, -1}),
            2);
  EXPECT_EQ(count({mjcFLEX_VERT_TRI, {1, 2, 3, 4}, -1},
                  {mjcFLEX_VERT_TRI, {1, 4, 3, 2}, -1}),
            1);

  // flex edge vs flex edge: {a1, b1, a2, b2}
  EXPECT_EQ(count({mjcFLEX_EDGE_EDGE, {1, 2, 3, 4}, -1},
                  {mjcFLEX_EDGE_EDGE, {1, 3, 2, 4}, -1}),
            2);
  EXPECT_EQ(count({mjcFLEX_EDGE_EDGE, {1, 2, 3, 4}, -1},
                  {mjcFLEX_EDGE_EDGE, {4, 3, 2, 1}, -1}),
            1);

  // flex vertex vs geom surface: {vertex, 0, 0, 0}; differing idx[1..3]
  // deduplicate
  EXPECT_EQ(count({mjcFLEX_VERT_GEOM, {1, 0, 0, 0}, 0},
                  {mjcFLEX_VERT_GEOM, {1, 2, 3, 4}, 0}),
            1);
  EXPECT_EQ(count({mjcFLEX_VERT_GEOM, {1, 0, 0, 0}, 0},
                  {mjcFLEX_VERT_GEOM, {2, 0, 0, 0}, 0}),
            2);
  EXPECT_EQ(count({mjcFLEX_VERT_GEOM, {1, 0, 0, 0}, 0},
                  {mjcFLEX_VERT_GEOM, {1, 0, 0, 0}, 1}),
            2);

  // step 1: non-empty old active set deduplication and old vs candidate merge
  auto merge_count = [&](const std::vector<mjcFlexPair>& old_set,
                         const std::vector<mjcFlexPair>& cand_set) {
    return mj_ipcCountDistinct(d, cand_set.data(), cand_set.size(),
                               old_set.data(), old_set.size());
  };

  // old set deduplication: permuted copies in old set are dropped in step 1
  EXPECT_EQ(merge_count({{mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 0},
                         {mjcGEOM_EDGE_EDGE, {1, 3, 2, 0}, 0}},
                        {}),
            1);

  // old vs candidate: permuted copy of old pair dropped from candidate set
  EXPECT_EQ(merge_count({{mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 0}},
                        {{mjcGEOM_EDGE_EDGE, {1, 3, 2, 0}, 0}}),
            1);

  // old vs candidate: distinct look-alike candidate is kept
  EXPECT_EQ(merge_count({{mjcGEOM_EDGE_EDGE, {1, 2, 3, 0}, 0}},
                        {{mjcGEOM_EDGE_EDGE, {2, 1, 3, 0}, 0}}),
            2);

  // old vs candidate: flex vertex vs geom surface (type 2) differing in
  // idx[1..3]
  EXPECT_EQ(merge_count({{mjcFLEX_VERT_GEOM, {1, 0, 0, 0}, 0}},
                        {{mjcFLEX_VERT_GEOM, {1, 5, 6, 7}, 0}}),
            1);
  EXPECT_EQ(merge_count({{mjcFLEX_VERT_GEOM, {1, 0, 0, 0}, 0}},
                        {{mjcFLEX_VERT_GEOM, {2, 0, 0, 0}, 0}}),
            2);

  mj_deleteData(d);
  mj_deleteModel(m);
}

}  // namespace
}  // namespace mujoco
