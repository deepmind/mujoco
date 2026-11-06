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

// Benchmarks for mj_ray broadphase culling and worst-case all-hit scenes.

#include <cmath>
#include <string>
#include <string_view>

#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

constexpr double kPi = 3.14159265358979323846;

// 24-cylinder ring around a rotating 16-rangefinder sensor rig on a floor.
constexpr char kRaySceneXml[] = R"xml(
<mujoco model="raycast_scene">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="sensor_rig" pos="0 0 0.6">
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <replicate count="16" euler="0 0 22.5">
        <site name="rf" pos="0 0 0" zaxis="1 0 -0.15"/>
      </replicate>
    </body>
    <replicate count="8" euler="0 0 45">
      <body pos="1.4 0 0.45">
        <geom type="cylinder" size="0.12 0.45"/>
      </body>
      <frame euler="0 0 15">
        <body pos="1.75 0 0.45">
          <geom type="cylinder" size="0.12 0.45"/>
        </body>
      </frame>
      <frame euler="0 0 30">
        <body pos="2.1 0 0.45">
          <geom type="cylinder" size="0.12 0.45"/>
        </body>
      </frame>
    </replicate>
  </worldbody>
  <sensor>
    <rangefinder name="rf" site="rf"/>
  </sensor>
</mujoco>
)xml";

// Builds N geoms along +x in back-to-front order (x = 2*N, ..., 2) so rays
// from the origin along +x hit every geom in tightening distance order and
// never trigger bounding-sphere culling.
std::string BuildAllHitXml(std::string_view geom_type, int ngeom) {
  std::string xml = "<mujoco model=\"ray_all_hit\">\n";
  if (geom_type == "mesh") {
    xml += R"xml(  <asset>
    <mesh name="cube"
      vertex="-0.5 -0.5 -0.5  0.5 -0.5 -0.5  0.5  0.5 -0.5 -0.5  0.5 -0.5
              -0.5 -0.5  0.5  0.5 -0.5  0.5  0.5  0.5  0.5 -0.5  0.5  0.5"/>
  </asset>
)xml";
  }
  std::string pos = std::to_string(2 * ngeom) + " 0 0";
  std::string geom;
  if (geom_type == "mesh") {
    geom = "<geom type=\"mesh\" mesh=\"cube\" pos=\"" + pos + "\"/>";
  } else if (geom_type == "sphere") {
    geom = "<geom type=\"sphere\" size=\"0.5\" pos=\"" + pos + "\"/>";
  } else if (geom_type == "cylinder") {
    geom = "<geom type=\"cylinder\" size=\"0.5 0.5\" pos=\"" + pos + "\"/>";
  } else {
    geom = "<geom type=\"box\" size=\"0.5 0.5 0.5\" pos=\"" + pos + "\"/>";
  }
  xml += "  <worldbody>\n    <replicate count=\"" + std::to_string(ngeom) +
         "\" offset=\"-2 0 0\">\n      " + geom +
         "\n    </replicate>\n  </worldbody>\n</mujoco>\n";
  return xml;
}

void ABSL_ATTRIBUTE_NO_TAIL_CALL BM_RaySceneSensors(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  char error[1024] = "";
  MjModelPtr m = LoadModelFromString(kRaySceneXml, error, sizeof(error));
  ASSERT_THAT(m, ::testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);
  double yaw = 0.0;
  for (auto s : state) {
    yaw += 0.017;
    d->qpos[0] = yaw;
    mj_forward(m.get(), d.get());
    benchmark::DoNotOptimize(d->sensordata[0]);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RaySceneSensors);

void ABSL_ATTRIBUTE_NO_TAIL_CALL BM_RayScene16Rays(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  char error[1024] = "";
  MjModelPtr m = LoadModelFromString(kRaySceneXml, error, sizeof(error));
  ASSERT_THAT(m, ::testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  int rig_body = mj_name2id(m.get(), mjOBJ_BODY, "sensor_rig");
  const mjtNum origin[3] = {0, 0, 0.6};
  mjtNum base_dir[16][3];
  for (int i = 0; i < 16; ++i) {
    base_dir[i][0] = d->site_xmat[9 * i + 2];
    base_dir[i][1] = d->site_xmat[9 * i + 5];
    base_dir[i][2] = d->site_xmat[9 * i + 8];
  }

  double yaw = 0.0;
  for (auto s : state) {
    yaw += 0.017;
    mjtNum c = std::cos(yaw);
    mjtNum sn = std::sin(yaw);
    mjtNum sum = 0;
    for (int i = 0; i < 16; ++i) {
      mjtNum rvec[3] = {
          c * base_dir[i][0] - sn * base_dir[i][1],
          sn * base_dir[i][0] + c * base_dir[i][1],
          base_dir[i][2],
      };
      int geomid = -1;
      sum += mj_ray(m.get(), d.get(), origin, rvec, nullptr, 1, rig_body,
                    &geomid, nullptr);
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * 16);
}
BENCHMARK(BM_RayScene16Rays);

void RunAllHitBenchmark(benchmark::State& state, std::string_view geom_type) {
  MujocoErrorTestGuard guard;
  int ngeom = state.range(0);
  char error[1024] = "";
  MjModelPtr m = LoadModelFromString(BuildAllHitXml(geom_type, ngeom), error,
                                     sizeof(error));
  ASSERT_THAT(m, ::testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  const mjtNum pnt[3] = {0, 0, 0};
  mjtNum vecs[16][3];
  for (int q = 0; q < 16; ++q) {
    double angle = (2.0 * kPi * q) / 16.0;
    vecs[q][0] = 1.25;
    vecs[q][1] = 0.02 * std::cos(angle);
    vecs[q][2] = 0.02 * std::sin(angle);
  }

  int idx = 0;
  for (auto s : state) {
    int geomid = -1;
    mjtNum normal[3];
    mjtNum dist = mj_ray(m.get(), d.get(), pnt, vecs[idx & 15], nullptr, 1, -1,
                         &geomid, normal);
    benchmark::DoNotOptimize(dist);
    benchmark::DoNotOptimize(geomid);
    benchmark::DoNotOptimize(normal[0]);
    ++idx;
  }
  state.SetItemsProcessed(state.iterations());
}

void ABSL_ATTRIBUTE_NO_TAIL_CALL BM_RayAllHitSphere(benchmark::State& state) {
  RunAllHitBenchmark(state, "sphere");
}
BENCHMARK(BM_RayAllHitSphere)->Arg(1)->Arg(2)->Arg(4);

void ABSL_ATTRIBUTE_NO_TAIL_CALL BM_RayAllHitCylinder(benchmark::State& state) {
  RunAllHitBenchmark(state, "cylinder");
}
BENCHMARK(BM_RayAllHitCylinder)->Arg(1)->Arg(2)->Arg(4);

void ABSL_ATTRIBUTE_NO_TAIL_CALL BM_RayAllHitBox(benchmark::State& state) {
  RunAllHitBenchmark(state, "box");
}
BENCHMARK(BM_RayAllHitBox)->Arg(1)->Arg(2)->Arg(4);

void ABSL_ATTRIBUTE_NO_TAIL_CALL BM_RayAllHitMesh(benchmark::State& state) {
  RunAllHitBenchmark(state, "mesh");
}
BENCHMARK(BM_RayAllHitMesh)->Arg(1)->Arg(2)->Arg(4);

}  // namespace
}  // namespace mujoco
