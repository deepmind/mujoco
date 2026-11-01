// Copyright 2022 DeepMind Technologies Limited
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

// Tests for ray casting.

#include "src/engine/engine_ray.h"

#include <cstring>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

static constexpr char kSingleGeomModel[] = R"(
<mujoco>
  <worldbody>
    <body pos="-2 0 0">
      <geom type="sphere" pos="1 0 0" size=".1"/>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr char kRayCastingModel[] = R"(
<mujoco>
  <worldbody>
    <geom name="static_group1" type="sphere" size=".1" pos="1 0 0" group="1"/>
    <body pos="0 0 0">
      <body pos="0 0 0">
        <joint/>
        <geom name="group0" type="sphere" size=".1" pos="3 0 0"/>
      </body>
      <geom name="group2" type="sphere" size=".1" pos="5 0 0" group="2"/>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr char kCubeletModel[] = R"(
<mujoco>
  <asset>
    <mesh name="cubelet"
    vertex="0.0085  -0.01     0.0085   -0.0085  -0.01    -0.0085    0.0085  -0.01    -0.0085
            0.01     0.0085   0.0085    0.01    -0.0085  -0.0085    0.01     0.0085  -0.0085
           -0.0085   0.0085   0.01      0.0085  -0.0085   0.01      0.0085   0.0085   0.01
           -0.01    -0.0085   0.0085   -0.01     0.0085  -0.0085   -0.01    -0.0085  -0.0085
           -0.0085   0.01     0.0085    0.0085   0.01    -0.0085   -0.0085   0.01    -0.0085
           -0.0085  -0.0085  -0.01     -0.0085  -0.01     0.0085   -0.0085  -0.0085   0.01
           -0.0085   0.0085  -0.01     -0.01     0.0085   0.0085    0.0085  -0.0085  -0.01
            0.01    -0.0085   0.0085    0.0085   0.0085  -0.01      0.0085   0.01     0.0085"/>
  </asset>

  <worldbody>
    <body pos="1 0 0">
      <geom type="mesh" mesh="cubelet"/>
    </body>
  </worldbody>
</mujoco>
)";

using ::std::string;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::NotNull;
using ::testing::Pointwise;
using RayTest = MujocoTest;

TEST_F(RayTest, NoExclusions) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(kRayCastingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  mjtNum pnt[] = {0.0, 0.0, 0.0};
  mjtNum vec[] = {1.0, 0.0, 0.0};
  mjtByte* geomgroup = nullptr;
  mjtByte flg_static = 1;  // Include static geoms
  int bodyexclude = -1;
  int geomid = -1;

  mj_kinematics(model.get(), data.get());
  mjtNum distance = mj_ray(model.get(), data.get(), pnt, vec, geomgroup,
                           flg_static, bodyexclude, &geomid, nullptr);
  EXPECT_STREQ(mj_id2name(model.get(), mjOBJ_GEOM, geomid), "static_group1");
  EXPECT_MJTNUM_EQ(distance, 0.9);
}

TEST_F(RayTest, Exclusions) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(kRayCastingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  mjtNum pnt[] = {0.0, 0.0, 0.0};
  mjtNum vec[] = {1.0, 0.0, 0.0};
  mjtByte geomgroup[] = {1, 1, 1};
  mjtByte flg_static = 1;
  int bodyexclude = -1;
  int geomid = -1;

  mj_kinematics(model.get(), data.get());
  mjtNum distance = mj_ray(model.get(), data.get(), pnt, vec, geomgroup,
                           flg_static, bodyexclude, &geomid, nullptr);
  EXPECT_STREQ(mj_id2name(model.get(), mjOBJ_GEOM, geomid), "static_group1");
  EXPECT_NEAR(distance, 0.9, MjTol(1e-12, 1e-5));

  // Exclude nearest geom
  geomgroup[1] = 0;
  distance = mj_ray(model.get(), data.get(), pnt, vec, geomgroup, flg_static,
                    bodyexclude, &geomid, nullptr);
  EXPECT_STREQ(mj_id2name(model.get(), mjOBJ_GEOM, geomid), "group0");
  EXPECT_NEAR(distance, 2.9, MjTol(1e-12, 1e-5));

  geomgroup[0] = 0;
  distance = mj_ray(model.get(), data.get(), pnt, vec, geomgroup, flg_static,
                    bodyexclude, &geomid, nullptr);
  EXPECT_STREQ(mj_id2name(model.get(), mjOBJ_GEOM, geomid), "group2");
  EXPECT_NEAR(distance, 4.9, MjTol(1e-12, 1e-5));

  geomgroup[2] = 0;
  distance = mj_ray(model.get(), data.get(), pnt, vec, geomgroup, flg_static,
                    bodyexclude, &geomid, nullptr);
  EXPECT_EQ(geomid, -1);
  EXPECT_NEAR(distance, -1, MjTol(1e-12, 1e-5));
}

TEST_F(RayTest, ExcludeStatic) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(kRayCastingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  mjtNum pnt[] = {0.0, 0.0, 0.0};
  mjtNum vec[] = {1.0, 0.0, 0.0};
  mjtByte geomgroup[] = {1, 1, 1};
  mjtByte flg_static = 0;  // Exclude static geoms
  int bodyexclude = -1;
  int geomid = -1;

  mj_kinematics(model.get(), data.get());
  mjtNum distance = mj_ray(model.get(), data.get(), pnt, vec, geomgroup,
                           flg_static, bodyexclude, &geomid, nullptr);
  EXPECT_STREQ(mj_id2name(model.get(), mjOBJ_GEOM, geomid), "group0");
  EXPECT_NEAR(distance, 2.9, MjTol(1e-12, 1e-5));
}

// ------------------------------- mj_multiRay --------------------------------

TEST_F(RayTest, MultiRayEqualsSingleRay) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kRayCastingModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());
  mj_forward(m.get(), d.get());

  // create ray array
  constexpr int N = 80;
  constexpr int M = 60;
  mjtNum vec[3 * N * M];
  mjtNum pnt[3] = {-1, 0, 0};
  mjtNum cone[4][3] = {{1, .2, -.2}, {1, .2, .2}, {1, -.2, -.2}, {1, -.2, .2}};
  memset(vec, 0, 3 * N * M * sizeof(mjtNum));

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < 3; ++k) {
        vec[3 * (i * M + j) + k] = i * cone[0][k] / (N - 1) +
                                   j * cone[1][1] / (M - 1) +
                                   (N - i - 1) * cone[2][k] / (N - 1) +
                                   (M - j - 1) * cone[3][k] / (M - 1);
      }
    }
  }

  // compute intersections with multiray functions
  mjtNum dist_multiray[N * M];
  int rgeomid_multiray[N * M];
  mj_multiRay(m.get(), d.get(), pnt, vec, nullptr, 1, -1, rgeomid_multiray,
              dist_multiray, nullptr, N * M, mjMAXVAL);

  // compare results with single ray function
  mjtNum dist;
  int rgeomid;
  int nhits = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      int idx = i * M + j;
      dist = mj_ray(m.get(), d.get(), pnt, vec + 3 * idx, nullptr, 1, -1,
                    &rgeomid, nullptr);
      EXPECT_MJTNUM_EQ(dist, dist_multiray[idx]);
      EXPECT_EQ(rgeomid, rgeomid_multiray[idx]);
      nhits += dist >= 0;
    }
  }
  EXPECT_GT(nhits, 10);
}

TEST_F(RayTest, MultiRayNormalEqualsSingleRayNormal) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kRayCastingModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());
  mj_forward(m.get(), d.get());

  // create ray array
  constexpr int N = 80;
  constexpr int M = 60;
  mjtNum vec[3 * N * M];
  mjtNum pnt[3] = {-1, 0, 0};
  mjtNum cone[4][3] = {{1, .2, -.2}, {1, .2, .2}, {1, -.2, -.2}, {1, -.2, .2}};
  memset(vec, 0, 3 * N * M * sizeof(mjtNum));

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < 3; ++k) {
        vec[3 * (i * M + j) + k] = i * cone[0][k] / (N - 1) +
                                   j * cone[1][1] / (M - 1) +
                                   (N - i - 1) * cone[2][k] / (N - 1) +
                                   (M - j - 1) * cone[3][k] / (M - 1);
      }
    }
  }

  // compute intersections with multiray normal function
  mjtNum dist_multiray[N * M];
  int rgeomid_multiray[N * M];
  mjtNum normal_multiray[3 * N * M];
  mj_multiRay(m.get(), d.get(), pnt, vec, NULL, 1, -1, rgeomid_multiray,
              dist_multiray, normal_multiray, N * M, mjMAXVAL);

  // compare results with single ray normal function
  mjtNum dist;
  int rgeomid;
  mjtNum normal[3];
  int nhits = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      int idx = i * M + j;
      dist = mj_ray(m.get(), d.get(), pnt, vec + 3 * idx, NULL, 1, -1, &rgeomid,
                    normal);
      EXPECT_MJTNUM_EQ(dist, dist_multiray[idx]);
      EXPECT_EQ(rgeomid, rgeomid_multiray[idx]);
      EXPECT_MJTNUM_EQ(normal[0], normal_multiray[3 * idx]);
      EXPECT_MJTNUM_EQ(normal[1], normal_multiray[3 * idx + 1]);
      EXPECT_MJTNUM_EQ(normal[2], normal_multiray[3 * idx + 2]);
      nhits += dist >= 0;
    }
  }
  EXPECT_GT(nhits, 10);
}

TEST_F(RayTest, EdgeCases) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kSingleGeomModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_THAT(m->nbvh, 1);
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());
  mj_forward(m.get(), d.get());

  // spherical bounding box and result arrays
  mjtNum geom_ba[4];
  mjtNum dist;
  int rgeomid;
  int flags[1] = {0};

  // pnt contained in bounding box
  mjtNum pnt1[] = {-1, 0, 0};
  mju_multiRayPrepare(m.get(), d.get(), pnt1, NULL, NULL, 1, -1, mjMAXVAL,
                      geom_ba, flags);
  EXPECT_MJTNUM_EQ(geom_ba[0], -mjPI);
  EXPECT_MJTNUM_EQ(geom_ba[1], 0);
  EXPECT_MJTNUM_EQ(geom_ba[2], mjPI);
  EXPECT_MJTNUM_EQ(geom_ba[3], mjPI);
  mjtNum vec1[] = {1, 0, 0};
  mj_multiRay(m.get(), d.get(), pnt1, vec1, NULL, 1, -1, &rgeomid, &dist,
              nullptr, 1, mjMAXVAL);
  EXPECT_FLOAT_EQ(dist, 0.1);

  // pnt at phi = Pi, -Pi
  mjtNum pnt2[] = {-.5, 0, 0};
  mju_multiRayPrepare(m.get(), d.get(), pnt2, NULL, NULL, 1, -1, mjMAXVAL,
                      geom_ba, flags);
  EXPECT_FLOAT_EQ(geom_ba[0], -mjPI);  // atan(y<0, x<0)
  EXPECT_FLOAT_EQ(geom_ba[2], mjPI);   // atan(y>0, x<0)
  mjtNum vec2[] = {-1, 0, 0};
  mj_multiRay(m.get(), d.get(), pnt2, vec2, NULL, 1, -1, &rgeomid, &dist,
              nullptr, 1, mjMAXVAL);
  EXPECT_FLOAT_EQ(dist, 0.4);

  // with cutoff
  mjtNum cutoff1 = 0.41, cutoff2 = 0.39;
  mju_multiRayPrepare(m.get(), d.get(), pnt2, NULL, NULL, 1, -1, cutoff1,
                      geom_ba, flags);
  EXPECT_EQ(flags[0], 0);
  mju_multiRayPrepare(m.get(), d.get(), pnt2, NULL, NULL, 1, -1, cutoff2,
                      geom_ba, flags);
  EXPECT_EQ(flags[0], 1);
  mj_multiRay(m.get(), d.get(), pnt2, vec2, NULL, 1, -1, &rgeomid, &dist,
              nullptr, 1, cutoff2);
  EXPECT_FLOAT_EQ(dist, -1);

  // pnt on the boundary of the box
  mjtNum pnt3[] = {.1, .1, .05};
  mju_multiRayPrepare(m.get(), d.get(), pnt3, NULL, NULL, 1, -1, mjMAXVAL,
                      geom_ba, flags);
  EXPECT_FLOAT_EQ(geom_ba[1], 0);
  EXPECT_FLOAT_EQ(geom_ba[3], mjPI);
  mjtNum vec3[] = {1, 1, 0};
  mj_multiRay(m.get(), d.get(), pnt3, vec3, NULL, 1, -1, &rgeomid, &dist,
              nullptr, 1, mjMAXVAL);
  EXPECT_FLOAT_EQ(dist, -1);

  // size 0 geom
  mjtNum pnt4[] = {-2, 0, 0};
  m->geom_aabb[0] = m->geom_aabb[1] = m->geom_aabb[2] = 0;
  m->geom_aabb[3] = m->geom_aabb[4] = m->geom_aabb[5] = 0;
  mju_multiRayPrepare(m.get(), d.get(), pnt4, NULL, NULL, 1, -1, mjMAXVAL,
                      geom_ba, flags);
  // margin = atan(max_half / dist) where max_half = max(aabb[3..5])
  // For a zero-size AABB: max_half = 0, so margin = 0
  mjtNum dist4 = mju_dist3(pnt4, d->geom_xpos);
  mjtNum max_half4 =
      mju_max(m->geom_aabb[3], mju_max(m->geom_aabb[4], m->geom_aabb[5]));
  mjtNum margin4 = mju_atan2(max_half4, dist4);
  EXPECT_NEAR(geom_ba[0], 0 - margin4, 1e-6);
  EXPECT_NEAR(geom_ba[1], mjPI / 2 - margin4, 1e-6);
  EXPECT_NEAR(geom_ba[2], 0 + margin4, 1e-6);
  EXPECT_NEAR(geom_ba[3], mjPI / 2 + margin4, 1e-6);
  mjtNum vec4[] = {1, 0, 0};
  mj_multiRay(m.get(), d.get(), pnt4, vec4, NULL, 1, -1, &rgeomid, &dist,
              nullptr, 1, mjMAXVAL);
  EXPECT_FLOAT_EQ(dist, 0.9);
}

TEST_F(RayTest, MultiRayCutoffPlane) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="10 10 .1"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  // plane origin is far from pnt, but the plane surface is within cutoff
  mjtNum pnt[] = {5, 0, 1};
  mjtNum vec[] = {0, 0, -1};
  mjtNum dist;
  int rgeomid;
  mj_multiRay(m.get(), d.get(), pnt, vec, NULL, 1, -1, &rgeomid, &dist, nullptr,
              1, 1.5);
  EXPECT_FLOAT_EQ(dist, 1);
  EXPECT_EQ(rgeomid, 0);
}

// ------------------------------- mj_rayMesh ---------------------------------

// old ray mesh intersection
mjtNum _rayMesh(const mjModel* m, const mjData* d, int geomid,
                const mjtNum* pnt, const mjtNum* vec) {
  // check geom type
  if (m->geom_type[geomid] != mjGEOM_MESH) {
    mju_error("mj_rayMesh: geom with mesh type expected");
  }

  // map to local frame
  mjtNum lpnt[3], lvec[3];
  const mjtNum* pos = d->geom_xpos + 3 * geomid;
  const mjtNum dif[3] = {pnt[0] - pos[0], pnt[1] - pos[1], pnt[2] - pos[2]};
  mju_mulMatTVec3(lpnt, d->geom_xmat + 9 * geomid, dif);
  mju_mulMatTVec3(lvec, d->geom_xmat + 9 * geomid, vec);

  // construct basis vectors of normal plane
  mjtNum b0[3] = {1, 1, 1}, b1[3];
  if (mju_abs(lvec[0]) >= mju_abs(lvec[1]) &&
      mju_abs(lvec[0]) >= mju_abs(lvec[2])) {
    b0[0] = 0;
  } else if (mju_abs(lvec[1]) >= mju_abs(lvec[2])) {
    b0[1] = 0;
  } else {
    b0[2] = 0;
  }
  mju_addScl3(b1, b0, lvec, -mju_dot3(lvec, b0) / mju_dot3(lvec, lvec));
  mju_normalize3(b1);
  mju_cross(b0, b1, lvec);
  mju_normalize3(b0);

  // init solution
  mjtNum x = -1, sol;

  // process all triangles
  int face, meshid = m->geom_dataid[geomid];
  for (face = m->mesh_faceadr[meshid];
       face < m->mesh_faceadr[meshid] + m->mesh_facenum[meshid]; face++) {
    // get float vertices
    float* vf[3];
    vf[0] =
        m->mesh_vert + 3 * (m->mesh_face[3 * face] + m->mesh_vertadr[meshid]);
    vf[1] = m->mesh_vert +
            3 * (m->mesh_face[3 * face + 1] + m->mesh_vertadr[meshid]);
    vf[2] = m->mesh_vert +
            3 * (m->mesh_face[3 * face + 2] + m->mesh_vertadr[meshid]);

    // convert to mjtNum
    mjtNum v[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        v[i][j] = (mjtNum)vf[i][j];
      }
    }

    // solve
    sol = mju_rayTriangle(v, lpnt, lvec, b0, b1, nullptr);

    // update
    if (sol >= 0 && (x < 0 || sol < x)) {
      x = sol;
    }
  }

  return x;
}

// performs a ray mesh test using a given mjModel
void _rayMeshTest(const mjModel* m) {
  mjData* d = mj_makeData(m);
  ASSERT_THAT(d, NotNull());
  mj_forward(m, d);

  // create ray array
  constexpr int N = 80;
  constexpr int M = 60;
  mjtNum vec[3 * N * M];
  mjtNum pnt[3] = {1, .2, 0};
  mjtNum cone[4][3] = {{-1, -1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, -1, -1}};
  memset(vec, 0, 3 * N * M * sizeof(mjtNum));

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < 3; ++k) {
        vec[3 * (i * M + j) + k] = i * cone[0][k] / (N - 1) +
                                   j * cone[1][1] / (M - 1) +
                                   (N - i - 1) * cone[2][k] / (N - 1) +
                                   (M - j - 1) * cone[3][k] / (M - 1);
      }
    }
  }
  // compare results with single ray function
  mjtNum dist_new, dist_old;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      int idx = i * M + j;
      dist_old = _rayMesh(m, d, /*geomid=*/0, pnt, vec + 3 * idx);
      dist_new = mj_rayMesh(m, d, /*geomid=*/0, pnt, vec + 3 * idx, nullptr);
      EXPECT_FLOAT_EQ(dist_new, dist_old);
    }
  }
  mj_deleteData(d);
}

TEST_F(RayTest, RayMeshPruning) {
  char error[1024];
  const string xml_path =
      GetTestDataFilePath("engine/testdata/ray/stanford_bunny.xml");

  MjModelPtr m(mj_loadXML(xml_path.c_str(), NULL, error, sizeof(error)));
  ASSERT_THAT(m.get(), NotNull()) << error;
  _rayMeshTest(m.get());

  m = LoadModelFromString(kCubeletModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  _rayMeshTest(m.get());
}

TEST_F(RayTest, RayAxisAlignedMesh) {
  static constexpr char kBoxMeshModel[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" scale=".025 .025 .025"
        vertex="-1 -1 -1  1 -1 -1  1  1 -1  1  1  1
                 1 -1  1 -1  1 -1 -1  1  1 -1 -1  1"/>
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="box"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kBoxMeshModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  const mjtNum pnt[3] = {0, 0, 1};
  const mjtNum vec[3] = {0, 0, -1};
  int geomid = -1;
  mjtNum dist =
      mj_ray(m.get(), d.get(), pnt, vec, nullptr, 1, -1, &geomid, nullptr);
  EXPECT_EQ(geomid, 0);
  EXPECT_THAT(dist, MjNear(0.975, 1e-6, 1e-5));
}

TEST_F(RayTest, RayHfield) {
  const char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="J" size="2 3 1 1" nrow="4" ncol="3"
              elevation="0 0 0
                         0 0 0
                         1 1 1
                         1 1 1"/>
    </asset>

    <worldbody>
      <light pos="0 0 1"/>
      <body name="dummy">
        <geom type="hfield" hfield="J" pos="0 0 .5"/>
      </body>
      <site name="1" pos="3 -2 1" zaxis="-1 0 0"/>
      <site name="2" pos="3 -2 0" zaxis="-1 0 0"/>
      <site name="3" pos="3  1 0" zaxis="-1 0 0"/>
      <body mocap="true" pos="1 0 1.5">
        <geom type="box" size=".2 .2 .2"/>
        <site name="4" zaxis="0 0 -1"/>
      </body>
    </worldbody>

    <sensor>
      <rangefinder site="1"/>
      <rangefinder site="2"/>
      <rangefinder site="3"/>
      <rangefinder site="4"/>
    </sensor>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());

  EXPECT_THAT(data->sensordata[0], MjNear(1, 1e-8, 1e-5));
  EXPECT_THAT(data->sensordata[1], MjNear(1, 1e-8, 1e-5));
  EXPECT_THAT(data->sensordata[2], MjNear(1, 1e-8, 1e-5));
  EXPECT_THAT(data->sensordata[3], MjNear(0.5, 1e-8, 1e-5));
}

static const char* const kPlaneModel = "engine/testdata/ray/plane.xml";
static const char* const kSphereModel = "engine/testdata/ray/sphere.xml";
static const char* const kCapsuleModel = "engine/testdata/ray/capsule.xml";
static const char* const kEllipsoidModel = "engine/testdata/ray/ellipsoid.xml";
static const char* const kCylinderModel = "engine/testdata/ray/cylinder.xml";
static const char* const kBoxModel = "engine/testdata/ray/box.xml";
static const char* const kMeshModel = "engine/testdata/ray/mesh.xml";
static const char* const kSdfModel = "engine/testdata/ray/sdf.xml";
static const char* const kHfieldModel = "engine/testdata/ray/hfield.xml";
static const char* const kFlexModel = "engine/testdata/ray/flex.xml";

TEST_F(RayTest, RayNormal) {
#ifdef mjUSESINGLE
  GTEST_SKIP() << "Flex face normals differ significantly in float32";
#endif
  for (const char* path : {kPlaneModel, kSphereModel, kCapsuleModel,
                           kEllipsoidModel, kCylinderModel, kBoxModel,
                           kMeshModel, kSdfModel, kHfieldModel, kFlexModel}) {
    const std::string xml_path = GetTestDataFilePath(path);
    char error[1024];
    mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
    ASSERT_THAT(m, NotNull()) << error;

    // exactly one geom or one flex, and one site in each model
    ASSERT_EQ(m->nsite, 1) << path;
    ASSERT_TRUE((m->ngeom == 1) != (m->nflex == 1)) << path;
    bool is_flex = m->nflex == 1;

    mjData* d = mj_makeData(m);

    // test parameters
    mjtNum kDuration = 2.0;  // length of rollout (seconds)
    int kCompare = 100;      // number of tests per rollout
    int compare_every = kDuration / (m->opt.timestep * kCompare);

    // roll out and compare analytic normal with fin-diff approximation
    int ntest = 0;  // tests performed
    int nstep = 0;  // steps elapsed
    while (d->time < kDuration) {
      mj_step(m, d);
      nstep++;

      // skip until this is timestep we should test on
      if (nstep % compare_every != 1) {
        continue;
      }

      // site info
      const mjtNum* pnt = d->site_xpos;
      const mjtNum vec[3] = {d->site_xmat[2], d->site_xmat[5], d->site_xmat[8]};

      // compute ray length and normal, compare with sensor
      mjtNum r, normal[3];
      if (!is_flex) {
        int geomid;
        r = mj_ray(m, d, pnt, vec, nullptr, 1, -1, &geomid, normal);

        // compare with sensor, expect geomid to be 0
        EXPECT_EQ(r, d->sensordata[0]) << path << ", time " << d->time;
        EXPECT_EQ(geomid, r >= 0 ? 0 : -1);
      } else {
        r = mj_rayFlex(m, d, /*flex_layer*/ 0, /*flg_vert*/ 1,
                       /*flg_edge*/ 1, /*flg_face*/ 1,
                       /*flg_skin*/ 1, /*flex_id*/ 0, pnt, vec, nullptr,
                       normal);
        // no sensor comparison: rangefinders only intersect with geoms
      }

      // if no intersection, skip
      if (r < 0) {
        EXPECT_THAT(normal, ElementsAre(0, 0, 0))
            << path << ", time " << d->time;
        continue;
      }

      // compute surface intersection point s
      mjtNum s[3];
      mju_addScl3(s, pnt, vec, r);

      // compute intersection points ds, nudged by eps in x,y site frame
      mjtNum eps = 1e-6;
      mjtNum ds[2][3];
      for (int i = 0; i < 2; ++i) {
        mjtNum nudge[3] = {d->site_xmat[0 + i], d->site_xmat[3 + i],
                           d->site_xmat[6 + i]};
        mjtNum dr, dpnt[3];
        mju_addScl3(dpnt, pnt, nudge, eps);
        if (!is_flex) {
          dr = mj_ray(m, d, dpnt, vec, NULL, 1, -1, nullptr, nullptr);
        } else {
          dr = mj_rayFlex(m, d, 0, 1, 1, 1, 1, 0, dpnt, vec, nullptr, nullptr);
        }
        mju_addScl3(ds[i], dpnt, vec, dr);
      }

      // compute in-plane tangents and expected normal
      mjtNum t0[3], t1[3], expected[3];
      mju_sub3(t0, ds[0], s);
      mju_sub3(t1, ds[1], s);
      mju_cross(expected, t1, t0);

      // normalize expected normal, skip if degenerate
      mjtNum norm = mju_normalize3(expected);
      if (norm < mjMINVAL) continue;

      // flipped expected normal, should match either expected or -expected
      mjtNum expected_neg[3] = {-expected[0], -expected[1], -expected[2]};

      // compare analytic with fin-diff approximation
      EXPECT_THAT(normal,
                  AnyOf(Pointwise(MjNear(100 * eps, 1e-3), expected),
                        Pointwise(MjNear(100 * eps, 1e-3), expected_neg)))
          << path << ", time " << d->time;

      // increment count
      ntest++;
    }

    // at least 10 tests should have been performed
    EXPECT_GT(ntest, 10);

    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

}  // namespace
}  // namespace mujoco
