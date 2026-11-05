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

// Tests for engine/engine_util_solve.c.

#include "src/engine/engine_util_misc.h"

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Pointwise;
using ::testing::StrEq;

using UtilMiscTest = MujocoTest;

TEST_F(UtilMiscTest, PrintsMemoryWarning) {
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 10)),
              HasSubstr("1K bytes"));
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 20)),
              HasSubstr("1M bytes"));
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 30)),
              HasSubstr("1G bytes"));
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 40)),
              HasSubstr("1T bytes"));
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 50)),
              HasSubstr("1P bytes"));
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 60)),
              HasSubstr("1E bytes"));
  EXPECT_THAT(mju_warningText(mjWARN_CNSTRFULL, pow(2, 30) + 1),
              HasSubstr("1073741825 bytes"));
}

TEST_F(UtilMiscTest, Sigmoid) {
  // function values
  EXPECT_EQ(mju_sigmoid(-1), 0);
  EXPECT_EQ(mju_sigmoid(0), 0);
  EXPECT_EQ(mju_sigmoid(0.5), 0.5);
  EXPECT_EQ(mju_sigmoid(1), 1);
  EXPECT_EQ(mju_sigmoid(2), 1);

  // epsilon for finite-differencing
  const mjtNum dx = MjTol(1e-7, 1e-3);
  const mjtNum fd_tol = MjTol(1e-7, 1e-3);

  // derivative at 0
  mjtNum dy_dx_0 = (mju_sigmoid(0 + dx) - mju_sigmoid(0)) / dx;
  EXPECT_NEAR(dy_dx_0, 0, fd_tol);

  // derivative at 1
  mjtNum dy_dx_1 = (mju_sigmoid(1) - mju_sigmoid(1 - dx)) / dx;
  EXPECT_NEAR(dy_dx_1, 0, fd_tol);

  // derivative at 0.5
  const mjtNum x = 0.5;
  mjtNum dy_dx_0p5 = (mju_sigmoid(x + dx) - mju_sigmoid(x - dx)) / (2 * dx);
  mjtNum expected = 30 * x * x * x * x - 60 * x * x * x + 30 * x * x;
  EXPECT_NEAR(dy_dx_0p5, expected, fd_tol);
}

TEST_F(UtilMiscTest, SpringDamperDampingRegimes) {
  constexpr mjtNum pos0 = 1;
  constexpr mjtNum vel0 = 0;
  constexpr mjtNum stiffness = 1;
  constexpr mjtNum dt = 1;
  const mjtNum tol = MjTol(1e-14, 1e-5);

  EXPECT_NEAR(mju_springDamper(pos0, vel0, stiffness, 3, dt),
              0.7866455993033682, tol);  // overdamped
  EXPECT_NEAR(mju_springDamper(pos0, vel0, stiffness, 2, dt),
              0.7357588823428847, tol);  // critically damped
  EXPECT_NEAR(mju_springDamper(pos0, vel0, stiffness, 1, dt),
              0.6597001533917017, tol);  // underdamped
}

TEST_F(UtilMiscTest, SpringDamperInvariantToTimeUnits) {
  constexpr mjtNum pos0 = 1;
  constexpr mjtNum vel0 = 0.5;
  constexpr mjtNum stiffness = 1;
  constexpr mjtNum dt = 1;
  constexpr mjtNum time_scale = 1e-8;

  for (mjtNum damping : {3, 2, 1}) {
    mjtNum expected = mju_springDamper(pos0, vel0, stiffness, damping, dt);
    mjtNum scaled = mju_springDamper(pos0, time_scale * vel0,
                                     time_scale * time_scale * stiffness,
                                     time_scale * damping, dt / time_scale);
    EXPECT_NEAR(scaled, expected, MjTol(1e-14, 1e-5)) << "damping=" << damping;
  }
}

TEST_F(UtilMiscTest, SpringDamperStrongOverdamping) {
  constexpr mjtNum damping = 1e9;
  const mjtNum tol = MjTol(1e-14, 1e-6);

  // At t = damping with k = 1, the fast mode has decayed and the slow mode
  // contributes exp(-1) to within O(1/damping^2) for either initial condition.
  EXPECT_NEAR(mju_springDamper(1, 0, 1, damping, damping), mju_exp(-1), tol);
  EXPECT_NEAR(mju_springDamper(0, damping, 1, damping, damping), mju_exp(-1),
              tol);
}

TEST_F(UtilMiscTest, SpringDamperZeroStiffness) {
  constexpr mjtNum pos0 = 1;
  constexpr mjtNum vel0 = 0.5;
  constexpr mjtNum dt = 0.5;

  for (mjtNum damping : {-3, 3}) {
    mjtNum expected = pos0 + vel0 * (1 - mju_exp(-damping * dt)) / damping;
    EXPECT_NEAR(mju_springDamper(pos0, vel0, 0, damping, dt), expected,
                MjTol(1e-14, 1e-6));
  }
}

TEST_F(UtilMiscTest, SpringDamperNonpositiveDamping) {
  // Roots 1 and 2: x(0) = x'(0) = 1 selects exp(t).
  EXPECT_NEAR(mju_springDamper(1, 1, 2, -3, 1), mju_exp(1), MjTol(1e-14, 1e-6));

  // Roots -1 and 1: the same initial conditions select exp(t).
  EXPECT_NEAR(mju_springDamper(1, 1, -1, 0, 1), mju_exp(1), MjTol(1e-14, 1e-6));
}

TEST_F(UtilMiscTest, SphereWrap) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <site size=".015" rgba="1 0 0 1"/>
    </default>

    <worldbody>
      <light pos="0 0 3"/>

      <site name="fixed" pos="0 0 1"/>
      <geom name="sphere" size=".1" pos="0 0 0.5"/>
      <site name="sidesite" pos=".52 0 .5"/>
      <body pos="0 0 .1">
        <freejoint/>
        <geom size=".05"/>
        <site name="body" pos="0 0 .05"/>
      </body>
    </worldbody>

    <tendon>
      <spatial name="tendon" range="0 0.8">
        <site site="fixed"/>
        <geom geom="sphere" sidesite="sidesite"/>
        <site site="body"/>
      </spatial>
    </tendon>

    <sensor>
      <tendonpos tendon="tendon"/>
    </sensor>

    <keyframe>
      <key qpos="-0.00653537 -0.068031 0.301253 0.982186 -0.180204 -0.0273515 0.0457068"/>
      <key qpos="-0.00653537 -0.069 0.301253 0.982186 -0.180204 -0.0273515 0.0457068"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // measure tendon length for keyframe 0
  mj_resetDataKeyframe(model.get(), data.get(), 0);
  mj_forward(model.get(), data.get());
  mjtNum ten_length0 = data->sensordata[0];

  // measure tendon length for keyframe 1
  mj_resetDataKeyframe(model.get(), data.get(), 1);
  mj_forward(model.get(), data.get());
  mjtNum ten_length1 = data->sensordata[0];

  // difference should be small
  mjtNum diff = ten_length1 - ten_length0;
  EXPECT_LT(mju_abs(diff), 1e-3);
}

TEST_F(UtilMiscTest, WrapInside) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum side[3] = {0, 0, 0};
  const mjtNum radius = 0.5;

  struct Case {
    mjtNum end[4];
    mjtNum pnt[2];
  };
  const Case cases[] = {
      {{2.0, 0.0, -0.7, 0.7}, {0.12578591002669121, 0.48391931645549874}},
      {{2.0, 0.0, -0.7, -0.7}, {0.12578591002669121, -0.48391931645549874}},
      {{1.0, 0.0, -1.4, 1.4}, {0.25140493009239173, 0.43219852050329788}},
      {{1.666667, 0.0, -2.185531, 1.213859},
       {0.1502227865088118, 0.47689948040811281}},
      {{1.0, 0.0, -1.425755, 1.402577},
       {0.25020009696447665, 0.43289711419570182}},
  };

  for (int type : {mjWRAP_SPHERE, mjWRAP_CYLINDER}) {
    for (const Case& c : cases) {
      const mjtNum x0[3] = {c.end[0], c.end[1], 0};
      const mjtNum x1[3] = {c.end[2], c.end[3], 0};
      mjtNum wpnt[6] = {0};
      mjtNum wlen = mju_wrap(wpnt, x0, x1, xpos, xmat, radius, type, side);
      EXPECT_EQ(wlen, 0);
      EXPECT_NEAR(wpnt[0], c.pnt[0], MjTol(1e-6, 1e-6));
      EXPECT_NEAR(wpnt[1], c.pnt[1], MjTol(1e-6, 1e-6));
      EXPECT_EQ(wpnt[2], 0);
      EXPECT_NEAR(wpnt[3], c.pnt[0], MjTol(1e-6, 1e-6));
      EXPECT_NEAR(wpnt[4], c.pnt[1], MjTol(1e-6, 1e-6));
      EXPECT_EQ(wpnt[5], 0);
    }
  }
}

TEST_F(UtilMiscTest, WrapCircleOnset) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  struct Case {
    mjtNum end[4];
    mjtNum radius;
    mjtNum wlen;
    mjtNum wpnt0[2];
    mjtNum wpnt1[2];
  };
  const Case cases[] = {
      // tangent segments nearly collinear (a > 1, b > 1)
      {{0.158835, -0.02149, -0.190895, 0.059719},
       0.015,
       1.1253506488708159e-6,
       {0.0033934139554204273, 0.014611117059525527},
       {0.0033923177705339506, 0.014611371603779007}},
      // vanishing wrap angle with opposite cross sign
      {{0.125, 0, MjEps(-0.099303395631717667, -0.219594762f),
        MjEps(0.027112324066336239, 0.0416523553f)},
       0.015,
       0,
       {0.0018, 0.014891608375189028},
       {0.0018, 0.014891608375189028}},
      // coincident tangent points (p2 == p4) at wrap onset
      {{16.0, 0, MjEps(-26.864008298238332, -14.2271366f),
        MjEps(5.1811203325342365, 3.65365791f)},
       1.92,
       0,
       {0.2304, 1.9061258720241956},
       {0.2304, 1.9061258720241956}},
  };

  for (int type : {mjWRAP_SPHERE, mjWRAP_CYLINDER}) {
    for (const Case& c : cases) {
      const mjtNum side_pos[3] = {0, c.radius, 0};
      for (const mjtNum* side :
           {static_cast<const mjtNum*>(nullptr), side_pos}) {
        const mjtNum x0[3] = {c.end[0], c.end[1], 0};
        const mjtNum x1[3] = {c.end[2], c.end[3], 0};
        mjtNum wpnt[6] = {0};
        mjtNum wlen = mju_wrap(wpnt, x0, x1, xpos, xmat, c.radius, type, side);
        EXPECT_NEAR(wlen, c.wlen, MjTol(1e-15, 1e-6));
        EXPECT_NEAR(wpnt[0], c.wpnt0[0], MjTol(1e-15, 1e-6));
        EXPECT_NEAR(wpnt[1], c.wpnt0[1], MjTol(1e-15, 1e-6));
        EXPECT_EQ(wpnt[2], 0);
        EXPECT_NEAR(wpnt[3], c.wpnt1[0], MjTol(1e-15, 1e-6));
        EXPECT_NEAR(wpnt[4], c.wpnt1[1], MjTol(1e-15, 1e-6));
        EXPECT_EQ(wpnt[5], 0);
      }
    }
  }
}

// wrap onset is robust to perturbing an endpoint by a few ulps
TEST_F(UtilMiscTest, WrapCircleOnsetUlp) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum radius = 1.92;
  const mjtNum end[4] = {16.0, 0, MjEps(-26.864008298238332, -14.2271366f),
                         MjEps(5.1811203325342365, 3.65365791f)};
  for (int k = -4; k <= 4; k++) {
    for (int coord = 2; coord < 4; coord++) {
      mjtNum x[4] = {end[0], end[1], end[2], end[3]};
      const mjtNum toward = k > 0 ? mjMAXVAL : -mjMAXVAL;
      for (int n = 0; n < std::abs(k); n++) {
        x[coord] = std::nextafter(x[coord], toward);
      }
      const mjtNum x0[3] = {x[0], x[1], 0};
      const mjtNum x1[3] = {x[2], x[3], 0};
      mjtNum wpnt[6] = {0};
      mjtNum wlen =
          mju_wrap(wpnt, x0, x1, xpos, xmat, radius, mjWRAP_CYLINDER, nullptr);
      EXPECT_LT(wlen, MjTol(1e-6, 1e-3)) << "k=" << k << " coord=" << coord;
    }
  }
}

// with a sidesite on the far side, a wrap close to 2*pi keeps its length
TEST_F(UtilMiscTest, WrapCircleSidesiteFullWrap) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum side[3] = {-1, 0, 0};
  for (mjtNum h : {1e-3, 0.0}) {
    for (mjtNum g : {1e-2, 1e-7}) {
      // endpoints just outside the unit circle on either side of (1, 0); the
      // wrap goes clockwise around the far side
      const mjtNum x0[3] = {1 + h, -g, 0};
      const mjtNum x1[3] = {1 + h, g, 0};
      // tangent angle acos(1/d) = atan(sqrt(d^2 - 1)), without the cancellation
      mjtNum tangent = mju_atan2(mju_sqrt(2 * h + h * h + g * g), 1);
      mjtNum expected = 2 * mjPI - 2 * mju_atan2(g, 1 + h) - 2 * tangent;
      mjtNum wpnt[6] = {0};
      mjtNum wlen =
          mju_wrap(wpnt, x0, x1, xpos, xmat, 1, mjWRAP_CYLINDER, side);
      EXPECT_NEAR(wlen, expected, MjTol(1e-9, 2e-5)) << "h=" << h << " g=" << g;
    }
  }
}

// sidesite at the arc midpoint selects that arc, including arcs longer than pi
TEST_F(UtilMiscTest, WrapCircleSidesiteSelectsArc) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum r = 0.5;

  // distance from origin to segment p-q
  auto segdist = [](const mjtNum p[2], const mjtNum q[2]) {
    mjtNum d[2] = {q[0] - p[0], q[1] - p[1]};
    mjtNum a = -(d[0] * p[0] + d[1] * p[1]) / (d[0] * d[0] + d[1] * d[1]);
    a = mju_clip(a, 0, 1);
    return mju_sqrt(mju_pow(p[0] + a * d[0], 2) + mju_pow(p[1] + a * d[1], 2));
  };

  // strict crossing of segments p1-p2 and p3-p4
  auto crossing = [](const mjtNum p1[2], const mjtNum p2[2], const mjtNum p3[2],
                     const mjtNum p4[2]) {
    auto orient = [](const mjtNum a[2], const mjtNum b[2], const mjtNum c[2]) {
      return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    };
    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 &&
           orient(p3, p4, p1) * orient(p3, p4, p2) < 0;
  };

  // construct the wrap first (start angle a, wrap angle th, direction dir),
  // then extend the tangent rays to get the endpoints; the expected tangent
  // points and length are exact
  int nmajor = 0, nminor = 0;
  for (mjtNum a : {0.1, 1.3, 2.9, 4.4, 5.8}) {
    for (mjtNum th : {0.4, 1.5, 2.8, 3.5, 4.5, 5.5}) {
      for (int dir : {1, -1}) {
        for (mjtNum len0 : {0.1, 0.6, 2.0}) {
          for (mjtNum len1 : {0.15, 0.8, 1.7}) {
            mjtNum b = a + dir * th;
            mjtNum t0[2] = {r * mju_cos(a), r * mju_sin(a)};
            mjtNum t1[2] = {r * mju_cos(b), r * mju_sin(b)};
            mjtNum p0[2] = {t0[0] + len0 * dir * mju_sin(a),
                            t0[1] - len0 * dir * mju_cos(a)};
            mjtNum p1[2] = {t1[0] - len1 * dir * mju_sin(b),
                            t1[1] + len1 * dir * mju_cos(b)};

            // keep valid wraps whose straight segment cuts the circle
            if (crossing(p0, t0, t1, p1) || segdist(p0, p1) >= r) {
              continue;
            }
            (th > mjPI ? nmajor : nminor)++;

            mjtNum m = a + dir * th / 2;
            const mjtNum side[3] = {2 * r * mju_cos(m), 2 * r * mju_sin(m), 0};
            const mjtNum x0[3] = {p0[0], p0[1], 0};
            const mjtNum x1[3] = {p1[0], p1[1], 0};
            for (int type : {mjWRAP_SPHERE, mjWRAP_CYLINDER}) {
              mjtNum wpnt[6] = {0};
              mjtNum wlen = mju_wrap(wpnt, x0, x1, xpos, xmat, r, type, side);
              EXPECT_NEAR(wlen, r * th, MjTol(1e-10, 1e-4))
                  << "a=" << a << " th=" << th;
              EXPECT_NEAR(wpnt[0], t0[0], MjTol(1e-10, 1e-4));
              EXPECT_NEAR(wpnt[1], t0[1], MjTol(1e-10, 1e-4));
              EXPECT_NEAR(wpnt[3], t1[0], MjTol(1e-10, 1e-4));
              EXPECT_NEAR(wpnt[4], t1[1], MjTol(1e-10, 1e-4));
            }
          }
        }
      }
    }
  }

  // both regimes are exercised
  EXPECT_GT(nmajor, 10);
  EXPECT_GT(nminor, 10);
}

// signed angle swept from p to q around the origin
static mjtNum SweptAngle(const mjtNum p[2], const mjtNum q[2]) {
  return mju_atan2(p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1]);
}

// distance from the origin to the segment p-q
static mjtNum SegmentDistance(const mjtNum p[2], const mjtNum q[2]) {
  mjtNum d[2] = {q[0] - p[0], q[1] - p[1]};
  mjtNum a = mju_clip(
      -(d[0] * p[0] + d[1] * p[1]) / (d[0] * d[0] + d[1] * d[1]), 0, 1);
  return mju_sqrt(mju_pow(p[0] + a * d[0], 2) + mju_pow(p[1] + a * d[1], 2));
}

// the wrap is the shortest path homotopic to the polyline p0 -> s -> p1, with s
// the sidesite projected onto the circle; checked with winding numbers around
// the center
TEST_F(UtilMiscTest, WrapCircleSidesiteHomotopy) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum r = 0.5;

  int nwrap = 0, nstraight = 0;
  for (int k0 = 0; k0 < 12; k0++) {
    for (int k1 = 0; k1 < 12; k1++) {
      for (int ks = 0; ks < 12; ks++) {
        for (mjtNum rho0 : {0.6, 1.5}) {
          for (mjtNum rho1 : {0.8, 2.0}) {
            mjtNum a0 = 2 * mjPI * k0 / 12 + 0.11,
                   a1 = 2 * mjPI * k1 / 12 + 0.37;
            mjtNum as = 2 * mjPI * ks / 12 + 0.23;
            mjtNum p0[2] = {rho0 * mju_cos(a0), rho0 * mju_sin(a0)};
            mjtNum p1[2] = {rho1 * mju_cos(a1), rho1 * mju_sin(a1)};
            mjtNum s[2] = {r * mju_cos(as), r * mju_sin(as)};

            // winding of the reference polyline and of the straight segment
            mjtNum wref = SweptAngle(p0, s) + SweptAngle(s, p1);
            mjtNum wline = SweptAngle(p0, p1);
            bool clear = SegmentDistance(p0, p1) > r;

            const mjtNum x0[3] = {p0[0], p0[1], 0};
            const mjtNum x1[3] = {p1[0], p1[1], 0};
            const mjtNum side[3] = {3 * s[0], 3 * s[1], 0};
            mjtNum wpnt[6] = {0};
            mjtNum wlen =
                mju_wrap(wpnt, x0, x1, xpos, xmat, r, mjWRAP_CYLINDER, side);

            if (wlen < 0) {
              // no wrap: the segment clears the circle and is in the reference
              // class
              nstraight++;
              EXPECT_TRUE(clear);
              EXPECT_LT(mju_abs(wline - wref), mjPI);
            } else {
              // wrap: in the reference class, and the segment is not a shorter
              // valid path
              nwrap++;
              mjtNum t0[2] = {wpnt[0], wpnt[1]}, t1[2] = {wpnt[3], wpnt[4]};
              mjtNum orient = t0[0] * (t0[1] - p0[1]) - t0[1] * (t0[0] - p0[0]);
              mjtNum warc = (orient > 0 ? 1 : -1) * wlen / r;
              mjtNum wpath = SweptAngle(p0, t0) + warc + SweptAngle(t1, p1);
              EXPECT_LT(mju_abs(wpath - wref), mjPI)
                  << "a0=" << a0 << " a1=" << a1;
              EXPECT_FALSE(clear && mju_abs(wline - wref) < mjPI);
            }
          }
        }
      }
    }
  }

  // both outcomes are exercised
  EXPECT_GT(nwrap, 100);
  EXPECT_GT(nstraight, 100);
}

// a sidesite more than 90 degrees from the segment does not force a wrap around
// the far side
TEST_F(UtilMiscTest, WrapCircleSidesiteNoSpuriousWrap) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum x0[3] = {-4.17, -0.44, 0};
  const mjtNum x1[3] = {-1.13, -1.53, 0};
  const mjtNum side[3] = {-0.79 * 1.5, 0.61 * 1.5, 0};
  mjtNum wpnt[6];
  for (int type : {mjWRAP_SPHERE, mjWRAP_CYLINDER}) {
    EXPECT_EQ(mju_wrap(wpnt, x0, x1, xpos, xmat, 1, type, side), -1);
  }
}

// a segment passing exactly through the center wraps on the side of the
// sidesite
TEST_F(UtilMiscTest, WrapCircleSidesiteThroughCenter) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum x0[3] = {0, 2, 0};
  const mjtNum x1[3] = {0, -1, 0};
  for (mjtNum sx : {1.5, -1.5}) {
    const mjtNum side[3] = {sx, 0, 0};
    for (int type : {mjWRAP_SPHERE, mjWRAP_CYLINDER}) {
      mjtNum wpnt[6];
      mjtNum wlen = mju_wrap(wpnt, x0, x1, xpos, xmat, 0.5, type, side);
      EXPECT_GT(wlen, 0);
      EXPECT_GT(wpnt[0] * sx, 0);
      EXPECT_GT(wpnt[3] * sx, 0);
    }
  }
}

// with the endpoints away from the far side of the sidesite, tendon length is
// 1-Lipschitz in the endpoint position (no jumps)
TEST_F(UtilMiscTest, WrapCircleSidesiteContinuity) {
  const mjtNum xpos[3] = {0, 0, 0};
  const mjtNum xmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const mjtNum r = 0.5;

  // total length of the path, wrapped or straight
  auto pathlen = [&](const mjtNum x0[3], const mjtNum x1[3],
                     const mjtNum side[3]) {
    mjtNum wpnt[6];
    mjtNum wlen = mju_wrap(wpnt, x0, x1, xpos, xmat, r, mjWRAP_CYLINDER, side);
    if (wlen < 0) {
      return mju_dist3(x0, x1);
    }
    return mju_dist3(x0, wpnt) + wlen + mju_dist3(wpnt + 3, x1);
  };

  const int nstep = 400;
  const mjtNum sector = 2 * mjPI / 3;
  const mjtNum rho1 = 1.2;
  for (mjtNum as : {0.0, 1.0, 2.5, 4.0}) {
    const mjtNum side[3] = {2 * r * mju_cos(as), 2 * r * mju_sin(as), 0};
    for (mjtNum da1 : {-1.8, -0.6, 0.4, 1.9}) {
      for (mjtNum rho : {0.7, 1.4}) {
        const mjtNum x1[3] = {rho1 * mju_cos(as + da1),
                              rho1 * mju_sin(as + da1), 0};
        mjtNum prev = 0, prevx[3] = {0};
        for (int k = 0; k <= nstep; k++) {
          mjtNum a0 = as - sector + 2 * sector * k / nstep;
          const mjtNum x0[3] = {rho * mju_cos(a0), rho * mju_sin(a0), 0};
          mjtNum len = pathlen(x0, x1, side);
          if (k > 0) {
            EXPECT_LE(mju_abs(len - prev),
                      mju_dist3(x0, prevx) + MjTol(1e-10, 1e-4))
                << "as=" << as << " da1=" << da1 << " k=" << k;
          }
          prev = len;
          mju_copy3(prevx, x0);
        }
      }
    }
  }
}

// compute time constant as in Millard et al. (2013)
// https://doi.org/10.1115/1.4023390
mjtNum muscleDynamicsMillard(mjtNum ctrl, mjtNum act, const mjtNum prm[2]) {
  // clamp control
  mjtNum ctrlclamp = mju_clip(ctrl, 0, 1);

  // clamp activation
  mjtNum actclamp = mju_clip(act, 0, 1);

  mjtNum tau;
  if (ctrlclamp > act) {
    tau = prm[0] * (0.5 + 1.5 * actclamp);
  } else {
    tau = prm[1] / (0.5 + 1.5 * actclamp);
  }

  // filter output
  return (ctrlclamp - act) / mjMAX(mjMINVAL, tau);
}

TEST_F(UtilMiscTest, SmoothMuscleDynamics) {
  mjtNum prm[3] = {0.01, 0.04, 0.0};

  // exact equality if tau_smooth = 0
  for (mjtNum ctrl : {-0.1, 0.0, 0.4, 0.5, 1.0, 1.1}) {
    for (mjtNum act : {-0.1, 0.0, 0.4, 0.5, 1.0, 1.1}) {
      mjtNum actdot_old = muscleDynamicsMillard(ctrl, act, prm);
      mjtNum actdot_new = mju_muscleDynamics(ctrl, act, prm);
      EXPECT_EQ(actdot_new, actdot_old);
    }
  }

  // positive tau_smooth
  mjtNum tau_smooth = 0.2;
  prm[2] = tau_smooth;
  mjtNum act = 0.5;
  mjtNum eps = 1e-6;

  mjtNum ctrl = 0.4 - eps;  // smaller than act by just over 0.5*tau_smooth
  EXPECT_EQ(muscleDynamicsMillard(ctrl, act, prm),
            mju_muscleDynamics(ctrl, act, prm));

  ctrl = 0.6 + eps;  // larger than act by just over 0.5*tau_smooth
  EXPECT_EQ(muscleDynamicsMillard(ctrl, act, prm),
            mju_muscleDynamics(ctrl, act, prm));

  // right in the middle should give average of time constants
  mjtNum tau_act = 0.2;
  mjtNum tau_deact = 0.3;
  for (mjtNum dctrl : {0.0, 0.1, 0.2, 1.0, 1.1}) {
    mjtNum lower =
        mju_muscleDynamicsTimescale(-dctrl, tau_act, tau_deact, tau_smooth);
    mjtNum upper =
        mju_muscleDynamicsTimescale(dctrl, tau_act, tau_deact, tau_smooth);
    EXPECT_EQ(0.5 * (upper + lower), 0.5 * (tau_act + tau_deact));
  }
}

TEST_F(UtilMiscTest, MuscleGainLength) {
  mjtNum lmin = 0.5;
  mjtNum lmax = 1.5;

  EXPECT_EQ(mju_muscleGainLength(0.0, lmin, lmax), 0);
  EXPECT_EQ(mju_muscleGainLength(0.5, lmin, lmax), 0);
  EXPECT_EQ(mju_muscleGainLength(0.75, lmin, lmax), 0.5);
  EXPECT_EQ(mju_muscleGainLength(1.0, lmin, lmax), 1);
  EXPECT_EQ(mju_muscleGainLength(1.25, lmin, lmax), 0.5);
  EXPECT_EQ(mju_muscleGainLength(1.5, lmin, lmax), 0);
  EXPECT_EQ(mju_muscleGainLength(2.0, lmin, lmax), 0);
}

TEST_F(UtilMiscTest, MjuSparseMap) {
  // nr = 3
  // src = [[1, 2, 0],
  //        [0, 3, 4],
  //        [5, 0, 6]]
  constexpr int nr = 3;
  const mjtNum mat_src[] = {1, 2, 3, 4, 5, 6};
  const int rownnz_src[] = {2, 2, 2};
  const int rowadr_src[] = {0, 2, 4};
  const int colind_src[] = {0, 1, 1, 2, 0, 2};

  // res = [[1, 0, 0],
  //        [0, 3, 0],
  //        [5, 0, 6]]
  constexpr int nnz_res = 4;
  const int rownnz_res[] = {1, 1, 2};
  const int rowadr_res[] = {0, 1, 2};
  const int colind_res[] = {0, 1, 0, 2};

  int map[nnz_res];
  mju_sparseMap(map, nr, rowadr_res, rownnz_res, colind_res, rowadr_src,
                rownnz_src, colind_src);

  // Expected map:
  // res[0] (1 at 0,0) -> src[0] (1 at 0,0) => map[0] = 0
  // res[1] (3 at 1,1) -> src[2] (3 at 1,1) => map[1] = 2
  // res[2] (5 at 2,0) -> src[4] (5 at 2,0) => map[2] = 4
  // res[3] (6 at 2,2) -> src[5] (6 at 2,2) => map[3] = 5
  EXPECT_THAT(map, ElementsAre(0, 2, 4, 5));

  // Verify the map by checking values
  mjtNum mat_res_gathered[nnz_res];
  mju_gather(mat_res_gathered, mat_src, map, nnz_res);
  EXPECT_THAT(AsVector(mat_res_gathered, nnz_res), ElementsAre(1, 3, 5, 6));
}

TEST_F(UtilMiscTest, MjuSparseLower2SymMap) {
  // nr = 3
  // src = [[1, 0, 0],
  //        [2, 3, 0],
  //        [4, 5, 6]]
  constexpr int nr = 3;
  const mjtNum mat_src[] = {1, 2, 3, 4, 5, 6};
  const int rownnz_src[] = {1, 2, 3};
  const int rowadr_src[] = {0, 1, 3};
  const int colind_src[] = {0, 0, 1, 0, 1, 2};

  // res = [[*, *, *],
  //        [*, *, *],
  //        [*, *, *]] (dense symmetric)
  constexpr int res_nnz = 9;
  const int rownnz_res[] = {3, 3, 3};
  const int rowadr_res[] = {0, 3, 6};
  const int colind_res[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

  int map[res_nnz];
  int cursor[nr];

  mju_lower2SymMap(map, nr, rowadr_res, rownnz_res, colind_res, rowadr_src,
                   rownnz_src, colind_src, cursor);

  // Expected map:
  // res(0,0) -> src(0,0) (k=0) => map[0] = 0
  // res(0,1) -> src(1,0) (k=1) => map[1] = 1
  // res(0,2) -> src(2,0) (k=3) => map[2] = 3
  // res(1,0) -> src(1,0) (k=1) => map[3] = 1
  // res(1,1) -> src(1,1) (k=2) => map[4] = 2
  // res(1,2) -> src(2,1) (k=4) => map[5] = 4
  // res(2,0) -> src(2,0) (k=3) => map[6] = 3
  // res(2,1) -> src(2,1) (k=4) => map[7] = 4
  // res(2,2) -> src(2,2) (k=5) => map[8] = 5
  EXPECT_THAT(map, ElementsAre(0, 1, 3, 1, 2, 4, 3, 4, 5));

  // Verify the map by checking values
  mjtNum mat_res[res_nnz];
  mju_gatherMasked(mat_res, mat_src, map, res_nnz);

  EXPECT_THAT(AsVector(mat_res, res_nnz),
              ElementsAre(1, 2, 4, 2, 3, 5, 4, 5, 6));
}

TEST_F(UtilMiscTest, MjuSparseLower2SymMapPartial) {
  // nr = 3
  // src = [[1, 0, 0],
  //        [2, 3, 0],
  //        [0, 0, 6]]
  constexpr int nr = 3;
  const mjtNum mat_src[] = {1, 2, 3, 6};
  const int rownnz_src[] = {1, 2, 1};
  const int rowadr_src[] = {0, 1, 3};
  const int colind_src[] = {0, 0, 1, 2};

  // res with a sparse symmetric pattern
  // res = [[*, *, *],
  //        [*, *, 0],
  //        [*, 0, *]]
  constexpr int res_nnz = 7;
  const int rownnz_res[] = {3, 2, 2};
  const int rowadr_res[] = {0, 3, 5};
  const int colind_res[] = {0, 1, 2, 0, 1, 0, 2};

  int map[res_nnz];
  int cursor[nr];

  mju_lower2SymMap(map, nr, rowadr_res, rownnz_res, colind_res, rowadr_src,
                   rownnz_src, colind_src, cursor);

  // Expected map for the non-zeros in res:
  // res(0,0) -> src(0,0) (k=0) => map[0] = 0
  // res(0,1) -> src(1,0) (k=1) => map[1] = 1
  // res(0,2) -> Unmapped       => map[2] = -1
  // res(1,0) -> src(1,0) (k=1) => map[3] = 1
  // res(1,1) -> src(1,1) (k=2) => map[4] = 2
  // res(2,0) -> Unmapped       => map[5] = -1
  // res(2,2) -> src(2,2) (k=3) => map[6] = 3
  EXPECT_THAT(map, ElementsAre(0, 1, -1, 1, 2, -1, 3));

  // Verify the map by checking values
  mjtNum mat_res[res_nnz];
  mju_gatherMasked(mat_res, mat_src, map, res_nnz);

  // Expected res values based on map:
  // mat_res[0] = mat_src[0] = 1
  // mat_res[1] = mat_src[1] = 2
  // mat_res[2]              = 0 (unmapped)
  // mat_res[3] = mat_src[1] = 2
  // mat_res[4] = mat_src[2] = 3
  // mat_res[5]              = 0 (unmapped)
  // mat_res[6] = mat_src[3] = 6
  EXPECT_THAT(AsVector(mat_res, res_nnz), ElementsAre(1, 2, 0, 2, 3, 0, 6));
}

TEST_F(UtilMiscTest, MjuIsZero) {
  mjtNum vec[1] = {1};
  EXPECT_EQ(mju_isZero(vec, 1), 0);
  EXPECT_EQ(mju_isZero(vec, 0), 1);
  vec[0] = 0;
  EXPECT_EQ(mju_isZero(vec, 1), 1);
  vec[0] = -0.0;
  EXPECT_EQ(mju_isZero(vec, 1), 1);
  EXPECT_EQ(mju_isZeroByte((const unsigned char*)vec, sizeof(mjtNum)), 0);
}

TEST_F(UtilMiscTest, MjuIsZeroByte) {
  // Zero length array
  EXPECT_TRUE(mju_isZeroByte(nullptr, 0));

  // zero length array with non-null pointer
  unsigned char vec0[1] = {0};
  EXPECT_TRUE(mju_isZeroByte(vec0, sizeof(vec0)));

  // one zero element array
  unsigned char vec1[1] = {0};
  EXPECT_TRUE(mju_isZeroByte(vec1, sizeof(vec1)));

  // one non-zero element array
  unsigned char vec2[2] = {1};
  EXPECT_FALSE(mju_isZeroByte(vec2, sizeof(vec2)));

  // Non-zero at start
  unsigned char vec3[3] = {1, 0, 0};
  EXPECT_FALSE(mju_isZeroByte(vec3, sizeof(vec3)));

  // Non-zero at end
  unsigned char vec4[3] = {0, 0, 1};
  EXPECT_FALSE(mju_isZeroByte(vec4, sizeof(vec4)));

  // Non-zero in middle
  unsigned char vec5[3] = {0, 1, 0};
  EXPECT_FALSE(mju_isZeroByte(vec5, sizeof(vec5)));
}

// --------------------------------- Interpolation -----------------------------

using InterpolationTest = MujocoTest;

TEST_F(InterpolationTest, mju_interpolate3D) {
  // quadratic functions should be interpolated exactly if order = 2
  auto quadratic_function_1 = [](mjtNum x, mjtNum y, mjtNum z) {
    return x * x + y * y + z * z;
  };
  auto quadratic_function_2 = [](mjtNum x, mjtNum y, mjtNum z) {
    return x * y * z + y * z * z + x * z * z;
  };
  auto quadratic_function_3 = [](mjtNum x, mjtNum y, mjtNum z) {
    return x * y * z + y * z * z + x * z * z + y * y * z + x * x * z + x + y +
           z;
  };
  static constexpr int order = 2;
  mjtNum coeff[3 * (order + 1) * (order + 1) * (order + 1)];
  int index = 0;
  for (int i = 0; i <= order; ++i) {
    for (int j = 0; j <= order; ++j) {
      for (int k = 0; k <= order; ++k) {
        coeff[3 * index + 0] = quadratic_function_1(.5 * i, .5 * j, .5 * k);
        coeff[3 * index + 1] = quadratic_function_2(.5 * i, .5 * j, .5 * k);
        coeff[3 * index + 2] = quadratic_function_3(.5 * i, .5 * j, .5 * k);
        index++;
      }
    }
  }
  static constexpr int nsample = 5;
  for (int i = 0; i < nsample; ++i) {
    mjtNum sample[3];
    mjtNum expected[3];
    mjtNum res[3] = {0};
    sample[0] = mju_Halton(i, 2);
    sample[1] = mju_Halton(i, 3);
    sample[2] = mju_Halton(i, 5);
    expected[0] = quadratic_function_1(sample[0], sample[1], sample[2]);
    expected[1] = quadratic_function_2(sample[0], sample[1], sample[2]);
    expected[2] = quadratic_function_3(sample[0], sample[1], sample[2]);
    mju_interpolate3D(res, sample, coeff, order, NULL);
    EXPECT_NEAR(res[0], expected[0], MjTol(1e-10, 1e-5));
    EXPECT_NEAR(res[1], expected[1], MjTol(1e-10, 1e-5));
    EXPECT_NEAR(res[2], expected[2], MjTol(1e-10, 1e-5));
  }
}

TEST_F(InterpolationTest, mju_cellLookup_SingleCell) {
  // single cell (1x1x1): local coords should equal global coords
  int cellnum[3] = {1, 1, 1};
  mjtNum coord[3] = {0.3, 0.7, 0.5};
  mjtNum local[3];
  int nodeindices[8];

  int npc = mju_cellLookup(coord, cellnum, 1, -1, local, nodeindices);
  EXPECT_EQ(npc, 8);
  EXPECT_NEAR(local[0], 0.3, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(local[1], 0.7, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(local[2], 0.5, MjTol(1e-12, 1e-6));

  // for trilinear 1x1x1: nodes are 0..7 in lexicographic order
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(nodeindices[i], i);
  }
}

TEST_F(InterpolationTest, mju_cellLookup_MultiCell) {
  // 2x3x4 grid, trilinear: 3x4x5 = 60 nodes
  int cellnum[3] = {2, 3, 4};
  int order = 1;
  int ny_g = 3 * 1 + 1;  // 4
  int nz_g = 4 * 1 + 1;  // 5

  // point at (0.75, 0.5, 0.125) -> cell (1, 1, 0)
  mjtNum coord[3] = {0.75, 0.5, 0.125};
  mjtNum local[3];
  int nodeindices[8];

  int npc = mju_cellLookup(coord, cellnum, order, -1, local, nodeindices);
  EXPECT_EQ(npc, 8);

  // cell (1,1,0): local = (0.75*2 - 1, 0.5*3 - 1, 0.125*4 - 0)
  EXPECT_NEAR(local[0], 0.5, 1e-12);
  EXPECT_NEAR(local[1], 0.5, 1e-12);
  EXPECT_NEAR(local[2], 0.5, 1e-12);

  // expected node indices for cell (1,1,0), trilinear:
  //   (gi, gj, gk) for li,lj,lk in {0,1}
  //   gi = 1+li, gj = 1+lj, gk = 0+lk
  //   gidx = gi*ny_g*nz_g + gj*nz_g + gk
  int expected[8];
  int ni = 0;
  for (int li = 0; li <= 1; li++) {
    for (int lj = 0; lj <= 1; lj++) {
      for (int lk = 0; lk <= 1; lk++) {
        expected[ni++] = (1 + li) * ny_g * nz_g + (1 + lj) * nz_g + lk;
      }
    }
  }
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(nodeindices[i], expected[i]);
  }
}

TEST_F(InterpolationTest, mju_cellLookup_Boundary) {
  // point exactly at coord=1.0 should clamp to last cell
  int cellnum[3] = {3, 3, 3};
  mjtNum coord[3] = {1.0, 1.0, 1.0};
  mjtNum local[3];

  mju_cellLookup(coord, cellnum, 1, -1, local, NULL);
  // cell (2,2,2), local = (1*3 - 2, 1*3 - 2, 1*3 - 2) = (1, 1, 1)
  EXPECT_NEAR(local[0], 1.0, 1e-12);
  EXPECT_NEAR(local[1], 1.0, 1e-12);
  EXPECT_NEAR(local[2], 1.0, 1e-12);

  // point at coord=0.0 should map to first cell
  mjtNum coord0[3] = {0.0, 0.0, 0.0};
  mju_cellLookup(coord0, cellnum, 1, -1, local, NULL);
  EXPECT_NEAR(local[0], 0.0, 1e-12);
  EXPECT_NEAR(local[1], 0.0, 1e-12);
  EXPECT_NEAR(local[2], 0.0, 1e-12);
}

TEST_F(InterpolationTest, mju_cellLookup_EmptyCellExtrapolation) {
  // 2x1x1 grid, trilinear: Cell 0 active (cell=0), Cell 1 empty
  int cellnum[3] = {2, 1, 1};
  int order = 1;

  // Point at x = 0.505 (1% into empty Cell 1) redirected to active Cell 0
  // with extrapolated local[0] = 0.505 * 2 - 0 = 1.01
  mjtNum coord[3] = {0.505, 0.25, 0.75};
  mjtNum local[3];
  int nodeindices[8];
  mju_cellLookup(coord, cellnum, order, 0, local, nodeindices);
  EXPECT_NEAR(local[0], 1.01, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(local[1], 0.25, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(local[2], 0.75, MjTol(1e-12, 1e-5));
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(nodeindices[i], i);
  }

  // Point far inside empty Cell 1 (x = 0.9 -> raw local = 1.8) is clamped
  // to 1.02
  mjtNum coord_far[3] = {0.9, 0.25, 0.75};
  mju_cellLookup(coord_far, cellnum, order, 0, local, nodeindices);
  EXPECT_NEAR(local[0], 1.02, MjTol(1e-12, 1e-5));
}

TEST_F(InterpolationTest, mju_defGradient) {
  int order = 1;
  mjtNum mat[9];
  mjtNum p1[3] = {.5, .5, .5};
  mjtNum p2[3] = {.25, .25, .25};
  mjtNum dof0[24] = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
                     1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};

  // identity
  mjtNum dof1[24];
  for (int i = 0; i < 24; ++i) dof1[i] = dof0[i];
  mju_defGradient(mat, p1, dof1, order);
  EXPECT_THAT(mat, ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));

  // translation
  mjtNum dof2[24];
  for (int i = 0; i < 24; ++i) dof2[i] = 2 + dof0[i];
  mju_defGradient(mat, p1, dof2, order);
  EXPECT_THAT(mat, ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
  mju_defGradient(mat, p2, dof2, order);
  EXPECT_THAT(mat, ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));

  // constant stretch
  mjtNum dof3[24];
  for (int i = 0; i < 24; ++i) dof3[i] = 2 * dof0[i];
  mju_defGradient(mat, p1, dof3, order);
  EXPECT_THAT(mat, ElementsAreArray({2, 0, 0, 0, 2, 0, 0, 0, 2}));
  mju_defGradient(mat, p2, dof3, order);
  EXPECT_THAT(mat, ElementsAreArray({2, 0, 0, 0, 2, 0, 0, 0, 2}));

  // axial stretch
  mjtNum dof4[24];
  for (int i = 0; i < 24; ++i) dof4[i] = (i % 3 == 1 ? 2 : 1) * dof0[i];
  mju_defGradient(mat, p1, dof4, order);
  EXPECT_THAT(mat, ElementsAreArray({1, 0, 0, 0, 2, 0, 0, 0, 1}));
  mju_defGradient(mat, p2, dof4, order);
  EXPECT_THAT(mat, ElementsAreArray({1, 0, 0, 0, 2, 0, 0, 0, 1}));

  // z-axis 90 degree rotation
  mjtNum dof5[24];
  for (int i = 0; i < 8; ++i) {
    mjtNum quat[4] = {0, 0, 0, 1};
    mjtNum axis[3] = {0, 0, 1};
    mju_axisAngle2Quat(quat, axis, mjPI / 2);
    mju_rotVecQuat(dof5 + 3 * i, dof0 + 3 * i, quat);
  }
  mju_defGradient(mat, p1, dof5, order);
  EXPECT_THAT(mat, Pointwise(MjNear(1e-8, 1e-6), {0, -1, 0, 1, 0, 0, 0, 0, 1}));
  mju_defGradient(mat, p2, dof5, order);
  EXPECT_THAT(mat, Pointwise(MjNear(1e-8, 1e-6), {0, -1, 0, 1, 0, 0, 0, 0, 1}));

  // z-axis 30 degree rotation
  mjtNum dof6[24];
  mjtNum rot6[9];
  for (int i = 0; i < 8; ++i) {
    mjtNum quat[4];
    mjtNum axis[3] = {0, 0, 1};
    mju_axisAngle2Quat(quat, axis, mjPI / 6);
    mju_rotVecQuat(dof6 + 3 * i, dof0 + 3 * i, quat);
    mju_quat2Mat(rot6, quat);
  }
  mju_defGradient(mat, p1, dof6, order);
  EXPECT_THAT(mat, Pointwise(MjNear(1e-8, 1e-6), rot6));
  mju_defGradient(mat, p2, dof6, order);
  EXPECT_THAT(mat, Pointwise(MjNear(1e-8, 1e-6), rot6));

  // z-axis CoM rotation
  mjtNum dof7[24];
  mjtNum rot7[9];
  for (int i = 0; i < 8; ++i) {
    mjtNum quat[4];
    mjtNum axis[3] = {0, 0, 1};
    mjtNum offset[3] = {-.5, -.5, 0};
    mju_axisAngle2Quat(quat, axis, mjPI / 6);
    mju_add3(dof7 + 3 * i, dof0 + 3 * i, offset);
    mju_rotVecQuat(dof7 + 3 * i, dof0 + 3 * i, quat);
    mju_quat2Mat(rot7, quat);
  }
  mju_defGradient(mat, p1, dof7, order);
  EXPECT_THAT(mat, Pointwise(MjNear(1e-8, 1e-6), rot7));
  mju_defGradient(mat, p2, dof7, order);
  EXPECT_THAT(mat, Pointwise(MjNear(1e-8, 1e-6), rot7));
}

TEST_F(InterpolationTest, mju_flexInterpState_MultiCell) {
  int order = 1;  // trilinear
  int cy = 2;
  int cz = 2;
  int nodenum = 27;  // 3x3x3

  std::vector<mjtNum> xpos(3 * nodenum);
  mjtNum quat[4];

  // Populate xpos directly for a grid centered at origin, rotated 90 deg around
  // Z Original grid points: {-0.1, 0.0, 0.1}^3 Rotated: (x, y, z) -> (-y, x, z)
  int idx = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mjtNum x = (i - 1) * 0.1;
        mjtNum y = (j - 1) * 0.1;
        mjtNum z = (k - 1) * 0.1;

        // Apply rotation
        xpos[3 * idx + 0] = -y;
        xpos[3 * idx + 1] = x;
        xpos[3 * idx + 2] = z;
        idx++;
      }
    }
  }

  int npc = (order + 1) * (order + 1) * (order + 1);
  std::vector<mjtNum> xpos_c(3 * npc);

  mju_flexGatherCellState(order, cy, cz, 0, 0, 0, xpos.data(), NULL, NULL,
                          xpos_c.data(), NULL, NULL, NULL, quat);

  // Expected quaternion for -90 deg around Z (global to local):
  // [sqrt(0.5), 0, 0, -sqrt(0.5)]
  mjtNum expected_val = mju_sqrt(0.5);
  EXPECT_NEAR(quat[0], expected_val, 1e-5);
  EXPECT_NEAR(quat[1], 0.0, 1e-5);
  EXPECT_NEAR(quat[2], 0.0, 1e-5);
  EXPECT_NEAR(quat[3], -expected_val, 1e-5);
}

using Base64Test = MujocoTest;

TEST_F(Base64Test, mju_encodeBase64) {
  std::array<char, 9> buffer;
  std::array<std::uint8_t, 5> arr = {15, 134, 190, 255, 240};

  std::size_t n = mju_encodeBase64(buffer.data(), arr.data(), arr.size());

  EXPECT_THAT(buffer.data(), StrEq("D4a+//A="));
  EXPECT_THAT(n, std::strlen(buffer.data()) + 1);
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_encodeBase64_align0) {
  std::array<char, 5> buffer;
  std::array<std::uint8_t, 3> arr = {'A', 'B', 'C'};

  std::size_t n = mju_encodeBase64(buffer.data(), arr.data(), arr.size());

  EXPECT_THAT(buffer.data(), StrEq("QUJD"));
  EXPECT_THAT(n, std::strlen(buffer.data()) + 1);
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_encodeBase64_align1) {
  std::array<char, 5> buffer;
  std::array<std::uint8_t, 2> arr = {'A', 'B'};

  std::size_t n = mju_encodeBase64(buffer.data(), arr.data(), arr.size());

  EXPECT_THAT(buffer.data(), StrEq("QUI="));
  EXPECT_THAT(n, std::strlen(buffer.data()) + 1);
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_encodeBase64_align2) {
  std::array<char, 5> buffer;
  std::array<std::uint8_t, 1> arr = {'A'};

  std::size_t n = mju_encodeBase64(buffer.data(), arr.data(), arr.size());

  EXPECT_THAT(buffer.data(), StrEq("QQ=="));
  EXPECT_THAT(n, std::strlen(buffer.data()) + 1);
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_encodeBase64_null) {
  std::array<char, 1> buffer;

  std::size_t n = mju_encodeBase64(buffer.data(), NULL, 0);

  EXPECT_THAT(n, 1);
  EXPECT_THAT(buffer[0], '\0');
}

TEST_F(Base64Test, mju_encodeBase64_ones) {
  std::array<char, 5> buffer;
  std::array<std::uint8_t, 3> arr = {255, 255, 255};

  std::size_t n = mju_encodeBase64(buffer.data(), arr.data(), arr.size());

  EXPECT_THAT(buffer.data(), StrEq("////"));
  EXPECT_THAT(n, std::strlen(buffer.data()) + 1);
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_isValidBase64_emptyStr) {
  std::size_t n = mju_isValidBase64("");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_invalid1) {
  std::size_t n = mju_isValidBase64("A");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_invalid2) {
  std::size_t n = mju_isValidBase64("AAA");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_invalid3) {
  std::size_t n = mju_isValidBase64("A==A");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_invalid5) {
  std::size_t n = mju_isValidBase64("A===");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_invalid6) {
  std::size_t n = mju_isValidBase64("aaaa====");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_invalid7) {
  std::size_t n = mju_isValidBase64("A#AA");

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_isValidBase64_valid1) {
  std::size_t n = mju_isValidBase64("AB+/");

  EXPECT_THAT(n, 3);
}

TEST_F(Base64Test, mju_isValidBase64_valid2) {
  std::size_t n = mju_isValidBase64("ABC=");

  EXPECT_THAT(n, 2);
}

TEST_F(Base64Test, mju_isValidBase64_valid3) {
  std::size_t n = mju_isValidBase64("AB==");

  EXPECT_THAT(n, 1);
}

TEST_F(Base64Test, mju_isValidBase64_valid4) {
  std::size_t n = mju_isValidBase64("az09AZ+/11==");

  EXPECT_THAT(n, 7);
}

TEST_F(Base64Test, mju_decodeBase64) {
  std::array<std::uint8_t, 5> buffer;
  const char* s = "D4a+//A=";

  std::size_t n = mju_decodeBase64(buffer.data(), s);

  EXPECT_THAT(buffer, ElementsAreArray({15, 134, 190, 255, 240}));
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_decodeBase6_align0) {
  std::array<std::uint8_t, 3> buffer;
  const char* s = "QUJD";

  std::size_t n = mju_decodeBase64(buffer.data(), s);

  EXPECT_THAT(buffer, ElementsAreArray({'A', 'B', 'C'}));
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_decodeBase64_align1) {
  std::array<std::uint8_t, 2> buffer;
  const char* s = "QUI=";

  std::size_t n = mju_decodeBase64(buffer.data(), s);

  EXPECT_THAT(buffer, ElementsAreArray({'A', 'B'}));
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_decodeBase64_align2) {
  std::array<std::uint8_t, 1> buffer;
  const char* s = "QQ==";

  std::size_t n = mju_decodeBase64(buffer.data(), s);

  EXPECT_THAT(buffer, ElementsAreArray({'A'}));
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, mju_decodeBase64_null) {
  const char* s = "";

  std::size_t n = mju_decodeBase64(NULL, s);

  EXPECT_THAT(n, 0);
}

TEST_F(Base64Test, mju_decodeBase64_ones) {
  std::array<std::uint8_t, 3> buffer;
  const char* s = "////";

  std::size_t n = mju_decodeBase64(buffer.data(), s);

  EXPECT_THAT(buffer, ElementsAreArray({255, 255, 255}));
  EXPECT_THAT(n, buffer.size());
}

TEST_F(Base64Test, decodeAndEncode) {
  std::array<std::uint8_t, 5> buffer1;
  std::array<char, 9> buffer2;
  const char* s = "D4a+/vA=";

  mju_decodeBase64(buffer1.data(), s);
  mju_encodeBase64(buffer2.data(), buffer1.data(), buffer1.size());

  EXPECT_THAT(buffer2.data(), StrEq(s));
}

// --------------------------------- History Buffers ---------------------------

using HistoryTest = MujocoTest;

// buffer layout: [user(1), cursor(1), times(n), values(n*dim)]
// cursor points to newest element (logical index n-1)
// after init, cursor=n-1, so physical indices equal logical indices

TEST_F(HistoryTest, Init) {
  constexpr int n = 4;
  constexpr int dim = 1;
  mjtNum buf[2 + n + n * dim];

  std::vector<mjtNum> times = {4, 6, 8, 10};
  std::vector<mjtNum> values = {99, 99, 99, 99};
  mju_historyInit(buf, n, dim, times.data(), values.data(), 0.0);

  // check header
  EXPECT_EQ(buf[0], 0.0);                         // user
  EXPECT_EQ(buf[1], static_cast<mjtNum>(n - 1));  // cursor = n-1

  // timestamps: [4, 6, 8, 10] (t=10 is newest)
  // values: [99, 99, 99, 99]
  // verify via read function (logical order)
  mjtNum res;
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 4.0, 0), 99.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 99.0);
}

TEST_F(HistoryTest, Init_Vector) {
  constexpr int n = 3;
  constexpr int dim = 2;
  mjtNum buf[2 + n + n * dim];

  std::vector<mjtNum> times = {-2, -1, 0};
  std::vector<mjtNum> values = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
  mju_historyInit(buf, n, dim, times.data(), values.data(), 0.0);

  EXPECT_EQ(buf[1], static_cast<mjtNum>(n - 1));  // cursor = n-1

  // verify via read function
  mjtNum res[dim];
  const mjtNum* ptr = mju_historyRead(buf, n, dim, res, -2.0, 0);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr[0], 1.0);
  EXPECT_EQ(ptr[1], 2.0);
}

TEST_F(HistoryTest, Append) {
  constexpr int n = 4;
  constexpr int dim = 1;
  // Initialize buffer properly, then insert
  mjtNum buf[2 + 2 * n];
  buf[0] = 0.0;
  buf[1] = n - 1;
  // timestamps: [4, 6, 8, 10]
  mjtNum times[] = {4, 6, 8, 10};
  mju_copy(buf + 2, times, n);
  // values: [0, 0, 0, 0]
  mju_zero(buf + 2 + n, n);

  // overwrite with specific values
  *mju_historyInsert(buf, n, dim, 4.0) = 1.0;
  *mju_historyInsert(buf, n, dim, 6.0) = 2.0;
  *mju_historyInsert(buf, n, dim, 8.0) = 3.0;
  *mju_historyInsert(buf, n, dim, 10.0) = 4.0;

  // now append at t=12
  *mju_historyInsert(buf, n, dim, 12.0) = 99.0;

  // verify logical order: [6, 8, 10, 12] -> [2, 3, 4, 99]
  mjtNum res;
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 4.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 12.0, 0), 99.0);

  // oldest should now be t=6
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 4.0, 0), 2.0);
}

TEST_F(HistoryTest, Append_Multiple) {
  constexpr int n = 3;
  constexpr int dim = 1;
  mjtNum buf[2 + 2 * n];
  buf[0] = 0.0;
  buf[1] = n - 1;
  mjtNum times[] = {-2, -1, 0};
  mju_copy(buf + 2, times, n);
  mju_zero(buf + 2 + n, n);

  for (int i = 1; i <= 4; i++) {
    mjtNum i_real = static_cast<mjtNum>(i);
    *mju_historyInsert(buf, n, dim, i_real) = i_real;
  }
  // Final: logical timestamps [2, 3, 4], values [2, 3, 4]
  mjtNum res;
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 2.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 3.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 4.0, 0), 4.0);
}

TEST_F(HistoryTest, ReadVector_ExactMatch) {
  constexpr int n = 3;
  constexpr int dim = 2;
  mjtNum buf[2 + n + n * dim];
  buf[0] = 0.0;
  buf[1] = n - 1;
  mjtNum times[] = {0, 1, 2};
  mju_copy(buf + 2, times, n);
  mju_zero(buf + 2 + n, n * dim);

  // set values: t=0->(1,2), t=1->(3,4), t=2->(5,6)
  mjtNum* slot0 = mju_historyInsert(buf, n, dim, 0.0);
  slot0[0] = 1.0;
  slot0[1] = 2.0;
  mjtNum* slot1 = mju_historyInsert(buf, n, dim, 1.0);
  slot1[0] = 3.0;
  slot1[1] = 4.0;
  mjtNum* slot2 = mju_historyInsert(buf, n, dim, 2.0);
  slot2[0] = 5.0;
  slot2[1] = 6.0;

  mjtNum res[dim];
  const mjtNum* ptr = mju_historyRead(buf, n, dim, res, 1.0, 0);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr[0], 3.0);
  EXPECT_EQ(ptr[1], 4.0);
}

TEST_F(HistoryTest, ReadVector_ZOH) {
  constexpr int n = 3;
  constexpr int dim = 2;
  mjtNum buf[2 + n + n * dim];
  buf[0] = 0.0;
  buf[1] = n - 1;
  mjtNum times[] = {0, 1, 2};
  mju_copy(buf + 2, times, n);
  mju_zero(buf + 2 + n, n * dim);

  mjtNum* slot0 = mju_historyInsert(buf, n, dim, 0.0);
  slot0[0] = 1.0;
  slot0[1] = 2.0;
  mjtNum* slot1 = mju_historyInsert(buf, n, dim, 1.0);
  slot1[0] = 3.0;
  slot1[1] = 4.0;
  mjtNum* slot2 = mju_historyInsert(buf, n, dim, 2.0);
  slot2[0] = 5.0;
  slot2[1] = 6.0;

  mjtNum res[dim];
  const mjtNum* ptr = mju_historyRead(buf, n, dim, res, 0.5, 0);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr[0], 1.0);
  EXPECT_EQ(ptr[1], 2.0);
}

TEST_F(HistoryTest, ReadVector_Linear) {
  constexpr int n = 3;
  constexpr int dim = 2;
  mjtNum buf[2 + n + n * dim];
  buf[0] = 0.0;
  buf[1] = n - 1;
  mjtNum times[] = {0, 1, 2};
  mju_copy(buf + 2, times, n);
  mju_zero(buf + 2 + n, n * dim);

  mjtNum* slot0 = mju_historyInsert(buf, n, dim, 0.0);
  slot0[0] = 1.0;
  slot0[1] = 2.0;
  mjtNum* slot1 = mju_historyInsert(buf, n, dim, 1.0);
  slot1[0] = 3.0;
  slot1[1] = 4.0;
  mjtNum* slot2 = mju_historyInsert(buf, n, dim, 2.0);
  slot2[0] = 5.0;
  slot2[1] = 6.0;

  mjtNum res[dim];
  const mjtNum* ptr = mju_historyRead(buf, n, dim, res, 0.5, 1);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_NEAR(res[0], 2.0, MjTol(1e-10, 1e-10));  // (1+3)/2
  EXPECT_NEAR(res[1], 3.0, MjTol(1e-10, 1e-10));  // (2+4)/2
}

TEST_F(HistoryTest, InsertOutOfOrder) {
  constexpr int n = 4;
  constexpr int dim = 1;
  mjtNum buf[2 + 2 * n];
  mjtNum res;

  auto reset = [&]() {
    buf[0] = 0.0;
    buf[1] = n - 1;
    mjtNum times[] = {4, 6, 8, 10};
    mju_copy(buf + 2, times, n);
    mju_zero(buf + 2 + n, n);

    *mju_historyInsert(buf, n, dim, 4.0) = 1.0;
    *mju_historyInsert(buf, n, dim, 6.0) = 2.0;
    *mju_historyInsert(buf, n, dim, 8.0) = 3.0;
    *mju_historyInsert(buf, n, dim, 10.0) = 4.0;
  };

  // insert in middle (between t=8 and t=10)
  reset();
  *mju_historyInsert(buf, n, dim, 9.0) = 99.0;
  // logical: [6, 8, 9, 10] -> [2, 3, 99, 4]
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 9.0, 0), 99.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 4.0);

  // insert near start (between t=4 and t=6)
  reset();
  *mju_historyInsert(buf, n, dim, 5.0) = 99.0;
  // logical: [5, 6, 8, 10] -> [99, 2, 3, 4]
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 5.0, 0), 99.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 4.0);

  // insert before oldest (t=3 < t=4): replaces oldest
  reset();
  *mju_historyInsert(buf, n, dim, 3.0) = 99.0;
  // logical: [3, 6, 8, 10] -> [99, 2, 3, 4]
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 3.0, 0), 99.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 4.0);
}

TEST_F(HistoryTest, InsertReplaceOnCollision) {
  constexpr int n = 4;
  constexpr int dim = 1;
  mjtNum buf[2 + 2 * n];
  mjtNum res;

  auto reset = [&]() {
    // timestamps: [4, 6, 8, 10], values initialized to 0
    buf[0] = 0.0;
    buf[1] = n - 1;
    mjtNum times[] = {4, 6, 8, 10};
    mju_copy(buf + 2, times, n);
    mju_zero(buf + 2 + n, n);

    *mju_historyInsert(buf, n, dim, 4.0) = 1.0;
    *mju_historyInsert(buf, n, dim, 6.0) = 2.0;
    *mju_historyInsert(buf, n, dim, 8.0) = 3.0;
    *mju_historyInsert(buf, n, dim, 10.0) = 4.0;
  };

  // collision in middle (t=8)
  reset();
  *mju_historyInsert(buf, n, dim, 8.0) = 99.0;
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 4.0, 0), 1.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 99.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 4.0);

  // collision at newest (t=10)
  reset();
  *mju_historyInsert(buf, n, dim, 10.0) = 99.0;
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 4.0, 0), 1.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 99.0);

  // collision at oldest (t=4)
  reset();
  *mju_historyInsert(buf, n, dim, 4.0) = 99.0;
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 4.0, 0), 99.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 6.0, 0), 2.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 8.0, 0), 3.0);
  EXPECT_EQ(*mju_historyRead(buf, n, dim, &res, 10.0, 0), 4.0);
}

void TriggerHistoryInitNonMonotonic() {
  mjtNum buf[10];
  mjtNum times[4] = {1, 2, 2, 4};  // not strictly increasing
  mjtNum values[4] = {0};
  mju_historyInit(buf, 4, 1, times, values, 0.0);
}

TEST_F(HistoryTest, Init_NonMonotonic) {
  EXPECT_FATAL_FAILURE(TriggerHistoryInitNonMonotonic(),
                       "mju_historyInit: times must be strictly increasing");
}

TEST_F(HistoryTest, CubicInterpolation) {
  int n = 2;
  int dim = 2;
  mjtNum buf[100];  // 2 + 2 + 2*2 = 8
  buf[0] = 0.0;
  buf[1] = n - 1;
  mjtNum times[] = {-1, 0};
  mju_copy(buf + 2, times, n);
  mju_zero(buf + 2 + n, n * dim);

  // Insert (0, 0, 1) and (1, 1, 0).
  // Dim 0: 0 -> 1. Spline: p(x) = 3x^2 - 2x^3
  // Dim 1: 1 -> 0. Spline: p(x) = 1 - 3x^2 + 2x^3
  mjtNum* slot0 = mju_historyInsert(buf, n, dim, 0.0);
  slot0[0] = 0.0;
  slot0[1] = 1.0;
  mjtNum* slot1 = mju_historyInsert(buf, n, dim, 1.0);
  slot1[0] = 1.0;
  slot1[1] = 0.0;

  mjtNum res[2];

  // Test midpoint x=0.5
  // Dim 0: 0.5
  // Dim 1: 1 - 0.5 = 0.5
  mju_historyRead(buf, n, dim, res, 0.5, 2);
  EXPECT_NEAR(res[0], 0.5, MjTol(1e-9, 1e-9));
  EXPECT_NEAR(res[1], 0.5, MjTol(1e-9, 1e-9));

  // Test x=0.25
  // Dim 0: 3*0.25^2 - 2*0.25^3
  // Dim 1: 1 - (3*0.25^2 - 2*0.25^3)
  mju_historyRead(buf, n, dim, res, 0.25, 2);
  mjtNum expected_0_25 = 3 * 0.25 * 0.25 - 2 * 0.25 * 0.25 * 0.25;
  EXPECT_NEAR(res[0], expected_0_25, MjTol(1e-9, 1e-9));
  EXPECT_NEAR(res[1], 1.0 - expected_0_25, MjTol(1e-9, 1e-9));

  // Test x=0.8
  // Dim 0: 3*0.8^2 - 2*0.8^3
  // Dim 1: 1 - (3*0.8^2 - 2*0.8^3)
  mju_historyRead(buf, n, dim, res, 0.8, 2);
  mjtNum expected_0_8 = 3 * 0.8 * 0.8 - 2 * 0.8 * 0.8 * 0.8;
  EXPECT_NEAR(res[0], expected_0_8, MjTol(1e-9, 1e-9));
  EXPECT_NEAR(res[1], 1.0 - expected_0_8, MjTol(1e-9, 1e-9));
}

// -------------------------------- Face State ---------------------------------

using FaceStateTest = MujocoTest;

// verify mju_flexGatherFaceState returns correct node indices for all 6 faces
// of a 1x1x1 trilinear grid (2x2x2 = 8 nodes, 4 nodes per face)
TEST_F(FaceStateTest, NodeIndicesSingleCell) {
  int order = 1;
  int cx = 1, cy = 1, cz = 1;
  int ny_g = cy * order + 1;  // 2
  int nz_g = cz * order + 1;  // 2

  // nelem_fe = 2*(1*1 + 1*1 + 1*1) = 6 face elements
  // face 0: x=0, face 1: x=max, face 2: y=0, face 3: y=max,
  // face 4: z=0, face 5: z=max

  // create dummy positions for 8 nodes
  std::vector<mjtNum> xpos(3 * 8, 0);
  for (int i = 0; i < 8; i++) {
    xpos[3 * i + 0] = (i / 4) * 1.0;
    xpos[3 * i + 1] = ((i / 2) % 2) * 1.0;
    xpos[3 * i + 2] = (i % 2) * 1.0;
  }

  // helper: compute expected global node index from (gx, gy, gz)
  auto gidx = [&](int gx, int gy, int gz) {
    return gx * ny_g * nz_g + gy * nz_g + gz;
  };

  // face 0: x=0 (fixed g[0]=0, varying g[1], g[2])
  // normal_axis=0, na0=1, na1=2
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 0, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 0, 0));
    EXPECT_EQ(indices[1], gidx(0, 0, 1));
    EXPECT_EQ(indices[2], gidx(0, 1, 0));
    EXPECT_EQ(indices[3], gidx(0, 1, 1));
  }

  // face 1: x=max (fixed g[0]=1, varying g[1], g[2])
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 1, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(1, 0, 0));
    EXPECT_EQ(indices[1], gidx(1, 0, 1));
    EXPECT_EQ(indices[2], gidx(1, 1, 0));
    EXPECT_EQ(indices[3], gidx(1, 1, 1));
  }

  // face 2: y=0 (fixed g[1]=0)
  // normal_axis=1, na0=2(z slow), na1=0(x fast)
  // loop order: l0→z, l1→x
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 2, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 0, 0));  // l0=0(z=0), l1=0(x=0)
    EXPECT_EQ(indices[1], gidx(1, 0, 0));  // l0=0(z=0), l1=1(x=1)
    EXPECT_EQ(indices[2], gidx(0, 0, 1));  // l0=1(z=1), l1=0(x=0)
    EXPECT_EQ(indices[3], gidx(1, 0, 1));  // l0=1(z=1), l1=1(x=1)
  }

  // face 3: y=max (fixed g[1]=1)
  // normal_axis=1, na0=2(z slow), na1=0(x fast)
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 3, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 1, 0));  // l0=0(z=0), l1=0(x=0)
    EXPECT_EQ(indices[1], gidx(1, 1, 0));  // l0=0(z=0), l1=1(x=1)
    EXPECT_EQ(indices[2], gidx(0, 1, 1));  // l0=1(z=1), l1=0(x=0)
    EXPECT_EQ(indices[3], gidx(1, 1, 1));  // l0=1(z=1), l1=1(x=1)
  }

  // face 4: z=0 (fixed g[2]=0, varying g[0], g[1])
  // normal_axis=2, na0=0, na1=1
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 4, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 0, 0));
    EXPECT_EQ(indices[1], gidx(0, 1, 0));
    EXPECT_EQ(indices[2], gidx(1, 0, 0));
    EXPECT_EQ(indices[3], gidx(1, 1, 0));
  }

  // face 5: z=max (fixed g[2]=1)
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 5, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 0, 1));
    EXPECT_EQ(indices[1], gidx(0, 1, 1));
    EXPECT_EQ(indices[2], gidx(1, 0, 1));
    EXPECT_EQ(indices[3], gidx(1, 1, 1));
  }
}

// verify node indices for a multi-cell grid (2x2x2 cells → 3x3x3 = 27 nodes)
TEST_F(FaceStateTest, NodeIndicesMultiCell) {
  int order = 1;
  int cx = 2, cy = 2, cz = 2;
  int ny_g = 3, nz_g = 3;  // (2*1+1) = 3

  // nelem_fe = 2*(2*2 + 2*2 + 2*2) = 24 face elements
  // face 0: x=0, cy*cz = 4 quads (indices 0-3)
  // face 1: x=max, 4 quads (indices 4-7)
  // face 2: y=0, cx*cz = 4 quads (indices 8-11)
  // face 3: y=max, 4 quads (indices 12-15)
  // face 4: z=0, cx*cy = 4 quads (indices 16-19)
  // face 5: z=max, 4 quads (indices 20-23)

  std::vector<mjtNum> xpos(3 * 27, 0);
  for (int i = 0; i < 27; i++) {
    int gi = i / 9;
    int gj = (i / 3) % 3;
    int gk = i % 3;
    xpos[3 * i + 0] = gi * 0.1;
    xpos[3 * i + 1] = gj * 0.1;
    xpos[3 * i + 2] = gk * 0.1;
  }

  auto gidx = [&](int gx, int gy, int gz) {
    return gx * ny_g * nz_g + gy * nz_g + gz;
  };

  // face 0 (x=0), quad 0: (q0=0, q1=0) within cy*cz face
  // c1 = face_count1[0] = cz = 2, so quad (0,0) → within_face = 0
  // na0=1, na1=2: g[0]=0, g[1]=0..1, g[2]=0..1
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 0, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 0, 0));
    EXPECT_EQ(indices[1], gidx(0, 0, 1));
    EXPECT_EQ(indices[2], gidx(0, 1, 0));
    EXPECT_EQ(indices[3], gidx(0, 1, 1));
  }

  // face 0 (x=0), quad 3: (q0=1, q1=1) → within_face = 1*2+1 = 3
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 3, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 1, 1));
    EXPECT_EQ(indices[1], gidx(0, 1, 2));
    EXPECT_EQ(indices[2], gidx(0, 2, 1));
    EXPECT_EQ(indices[3], gidx(0, 2, 2));
  }

  // face 1 (x=max), quad 0: fe_idx = 4 (after face 0's 4 quads)
  // g[0] = cx*order = 2
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 4, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(2, 0, 0));
    EXPECT_EQ(indices[1], gidx(2, 0, 1));
    EXPECT_EQ(indices[2], gidx(2, 1, 0));
    EXPECT_EQ(indices[3], gidx(2, 1, 1));
  }
}

// verify node indices for a non-cubic grid (cx != cz)
TEST_F(FaceStateTest, NodeIndicesNonCubicGrid) {
  int order = 1;
  int cx = 2, cy = 1, cz = 3;
  int ny_g = cy * order + 1;  // 2
  int nz_g = cz * order + 1;  // 4

  // create dummy positions for (2*1+1)*(1*1+1)*(3*1+1) = 3*2*4 = 24 nodes
  std::vector<mjtNum> xpos(3 * 24, 0);
  for (int i = 0; i < 24; i++) {
    int gi = i / 8;
    int gj = (i / 4) % 2;
    int gk = i % 4;
    xpos[3 * i + 0] = gi * 0.1;
    xpos[3 * i + 1] = gj * 0.1;
    xpos[3 * i + 2] = gk * 0.1;
  }

  auto gidx = [&](int gx, int gy, int gz) {
    return gx * ny_g * nz_g + gy * nz_g + gz;
  };

  // face 2 (y=0): normal_axis=1, na0=2(z slow), na1=0(x fast)
  // counts: na0 -> cz = 3, na1 -> cx = 2
  // total quads on face 2 = 6
  // we test within_face = 2 (third quad)
  // correct: c1 = cx = 2. q0 = 2/2 = 1, q1 = 2%2 = 0
  //
  // face element index calculation:
  // face 0: cy*cz = 1*3 = 3 quads (indices 0-2)
  // face 1: cy*cz = 1*3 = 3 quads (indices 3-5)
  // face 2: cx*cz = 2*3 = 6 quads. Quad 2 is index 2 within this face.
  // Total flat index = 3 + 3 + 2 = 8
  {
    int indices[4];
    mju_flexGatherFaceState(order, cx, cy, cz, 8, xpos.data(), NULL, NULL, NULL,
                            NULL, NULL, indices, NULL);
    EXPECT_EQ(indices[0], gidx(0, 0, 1));
    EXPECT_EQ(indices[1], gidx(1, 0, 1));
    EXPECT_EQ(indices[2], gidx(0, 0, 2));
    EXPECT_EQ(indices[3], gidx(1, 0, 2));
  }
}

// verify data gathering: positions, velocities, and reference positions
TEST_F(FaceStateTest, DataGathering) {
  int order = 1;
  int cx = 1, cy = 1, cz = 1;
  int npe = 4;
  int nnodes = 8;

  // create positions and velocities for 8 nodes
  std::vector<mjtNum> xpos(3 * nnodes);
  std::vector<mjtNum> vel(3 * nnodes);
  std::vector<mjtNum> xpos0(3 * nnodes);
  for (int i = 0; i < nnodes; i++) {
    for (int d = 0; d < 3; d++) {
      xpos[3 * i + d] = 10 * i + d;
      vel[3 * i + d] = 100 * i + d;
      xpos0[3 * i + d] = 1000 * i + d;
    }
  }

  // gather face 4 (z=0): nodes at (0,0,0), (0,1,0), (1,0,0), (1,1,0)
  // = global indices 0, 2, 4, 6
  std::vector<mjtNum> xpos_f(3 * npe);
  std::vector<mjtNum> vel_f(3 * npe);
  std::vector<mjtNum> xpos0_f(3 * npe);
  int indices[4];

  mju_flexGatherFaceState(order, cx, cy, cz, 4, xpos.data(), vel.data(),
                          xpos0.data(), xpos_f.data(), vel_f.data(),
                          xpos0_f.data(), indices, NULL);

  for (int n = 0; n < npe; n++) {
    int gi = indices[n];
    for (int d = 0; d < 3; d++) {
      EXPECT_EQ(xpos_f[3 * n + d], xpos[3 * gi + d]);
      EXPECT_EQ(vel_f[3 * n + d], vel[3 * gi + d]);
      EXPECT_EQ(xpos0_f[3 * n + d], xpos0[3 * gi + d]);
    }
  }
}

// verify that flexInterpRotation2D produces identity for axis-aligned faces
// (tested via mju_flexGatherFaceState with quat output)
TEST_F(FaceStateTest, IdentityRotationAxisAligned) {
  int order = 1;
  int cx = 1, cy = 1, cz = 1;
  int npe = 4;

  // create an axis-aligned unit cube: 8 nodes at {0,1}^3
  std::vector<mjtNum> xpos(3 * 8);
  int idx = 0;
  for (int i = 0; i <= 1; i++) {
    for (int j = 0; j <= 1; j++) {
      for (int k = 0; k <= 1; k++) {
        xpos[3 * idx + 0] = i;
        xpos[3 * idx + 1] = j;
        xpos[3 * idx + 2] = k;
        idx++;
      }
    }
  }

  std::vector<mjtNum> xpos_f(3 * npe);
  mjtNum quat[4];

  // test all 6 faces: each should give identity rotation (quat = [1,0,0,0])
  int nelem_fe = 6;
  for (int fe = 0; fe < nelem_fe; fe++) {
    mju_flexGatherFaceState(order, cx, cy, cz, fe, xpos.data(), NULL, NULL,
                            xpos_f.data(), NULL, NULL, NULL, quat);
    EXPECT_NEAR(mju_abs(quat[0]), 1.0, 1e-10) << "face " << fe;
    EXPECT_NEAR(quat[1], 0.0, 1e-10) << "face " << fe;
    EXPECT_NEAR(quat[2], 0.0, 1e-10) << "face " << fe;
    EXPECT_NEAR(quat[3], 0.0, 1e-10) << "face " << fe;
  }
}

// verify that flexInterpRotation2D extracts the correct rotation for a
// globally rotated cube (90° around z-axis)
TEST_F(FaceStateTest, RotatedCubeRotation) {
  int order = 1;
  int cx = 1, cy = 1, cz = 1;
  int npe = 4;

  // create an axis-aligned unit cube, then rotate 90° around z
  // rotation: (x,y,z) → (-y, x, z)
  std::vector<mjtNum> xpos(3 * 8);
  int idx = 0;
  for (int i = 0; i <= 1; i++) {
    for (int j = 0; j <= 1; j++) {
      for (int k = 0; k <= 1; k++) {
        mjtNum orig[3] = {(mjtNum)i, (mjtNum)j, (mjtNum)k};
        mjtNum axis[3] = {0, 0, 1};
        mjtNum rot_quat[4];
        mju_axisAngle2Quat(rot_quat, axis, mjPI / 2);
        mju_rotVecQuat(xpos.data() + 3 * idx, orig, rot_quat);
        idx++;
      }
    }
  }

  std::vector<mjtNum> xpos_f(3 * npe);
  mjtNum quat[4];

  // expected rotation: global→local is inverse of the 90° z rotation
  // 90° around z: quat = [cos(45°), 0, 0, sin(45°)]
  // inverse (global→local): [cos(45°), 0, 0, -sin(45°)]
  mjtNum sq2 = mju_sqrt(0.5);

  // test face 4 (z=0): normal_axis=2, in-plane axes are (0,1)
  // tangent vectors should reflect the 90° z rotation
  mju_flexGatherFaceState(order, cx, cy, cz, 4, xpos.data(), NULL, NULL,
                          xpos_f.data(), NULL, NULL, NULL, quat);

  EXPECT_NEAR(quat[0], sq2, 1e-5);
  EXPECT_NEAR(quat[1], 0.0, 1e-5);
  EXPECT_NEAR(quat[2], 0.0, 1e-5);
  EXPECT_NEAR(quat[3], -sq2, 1e-5);

  // test face 5 (z=max): should give same rotation
  mju_flexGatherFaceState(order, cx, cy, cz, 5, xpos.data(), NULL, NULL,
                          xpos_f.data(), NULL, NULL, NULL, quat);

  EXPECT_NEAR(quat[0], sq2, 1e-5);
  EXPECT_NEAR(quat[1], 0.0, 1e-5);
  EXPECT_NEAR(quat[2], 0.0, 1e-5);
  EXPECT_NEAR(quat[3], -sq2, 1e-5);
}

// verify that flexInterpRotation2D matches the 3D cell rotation for
// the same globally-rotated cube
TEST_F(FaceStateTest, RotationConsistencyWith3D) {
  int order = 1;
  int cx = 1, cy = 1, cz = 1;

  // create 90° z-rotated unit cube
  std::vector<mjtNum> xpos(3 * 8);
  int idx = 0;
  for (int i = 0; i <= 1; i++) {
    for (int j = 0; j <= 1; j++) {
      for (int k = 0; k <= 1; k++) {
        mjtNum orig[3] = {(mjtNum)i, (mjtNum)j, (mjtNum)k};
        mjtNum axis[3] = {0, 0, 1};
        mjtNum rot_quat[4];
        mju_axisAngle2Quat(rot_quat, axis, mjPI / 6);
        mju_rotVecQuat(xpos.data() + 3 * idx, orig, rot_quat);
        idx++;
      }
    }
  }

  // get 3D cell rotation
  int npc = 8;
  std::vector<mjtNum> xpos_c(3 * npc);
  mjtNum quat_3d[4];
  mju_flexGatherCellState(order, cy, cz, 0, 0, 0, xpos.data(), NULL, NULL,
                          xpos_c.data(), NULL, NULL, NULL, quat_3d);

  // get 2D face rotation for each face and verify it matches the 3D rotation
  int npe = 4;
  std::vector<mjtNum> xpos_f(3 * npe);

  int nelem_fe = 6;
  for (int fe = 0; fe < nelem_fe; fe++) {
    mjtNum quat_2d[4];
    mju_flexGatherFaceState(order, cx, cy, cz, fe, xpos.data(), NULL, NULL,
                            xpos_f.data(), NULL, NULL, NULL, quat_2d);

    // quaternions may differ by sign; compare unsigned
    mjtNum dot = quat_3d[0] * quat_2d[0] + quat_3d[1] * quat_2d[1] +
                 quat_3d[2] * quat_2d[2] + quat_3d[3] * quat_2d[3];
    EXPECT_NEAR(mju_abs(dot), 1.0, 1e-5)
        << "face " << fe << ": 2D rotation differs from 3D cell rotation";
  }
}

// ------------------------------ Shell TFI Interpolation ----------------------

using ShellTFITest = MujocoTest;

// helper: set up a regular nx*ny*nz grid with positions at grid indices
static void MakeRegularGrid(mjtNum* nodexpos, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int idx = i * ny * nz + j * nz + k;
        nodexpos[3 * idx + 0] = (mjtNum)i;
        nodexpos[3 * idx + 1] = (mjtNum)j;
        nodexpos[3 * idx + 2] = (mjtNum)k;
      }
    }
  }
}

TEST_F(ShellTFITest, IdentityGrid) {
  // 3x3x3 grid: 1 interior node at (1,1,1)
  constexpr int nx = 3, ny = 3, nz = 3;
  mjtNum nodexpos[3 * nx * ny * nz];
  MakeRegularGrid(nodexpos, nx, ny, nz);

  // save expected interior position
  mjtNum expected[3] = {1.0, 1.0, 1.0};

  // run TFI
  mju_shellTrackInterior(nodexpos, nx, ny, nz);

  // interior node at (1,1,1) should match
  int idx = 1 * ny * nz + 1 * nz + 1;
  EXPECT_NEAR(nodexpos[3 * idx + 0], expected[0], MjTol(1e-12, 1e-5));
  EXPECT_NEAR(nodexpos[3 * idx + 1], expected[1], MjTol(1e-12, 1e-5));
  EXPECT_NEAR(nodexpos[3 * idx + 2], expected[2], MjTol(1e-12, 1e-5));
}

TEST_F(ShellTFITest, UniformScaling) {
  // 3x3x3: scale all boundary nodes by 2x, interior should follow
  constexpr int nx = 3, ny = 3, nz = 3;
  mjtNum nodexpos[3 * nx * ny * nz];
  MakeRegularGrid(nodexpos, nx, ny, nz);

  // scale all nodes
  for (int i = 0; i < 3 * nx * ny * nz; i++) {
    nodexpos[i] *= 2.0;
  }

  // run TFI — interior should be reconstructed to 2*original
  mju_shellTrackInterior(nodexpos, nx, ny, nz);

  int idx = 1 * ny * nz + 1 * nz + 1;
  EXPECT_NEAR(nodexpos[3 * idx + 0], 2.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(nodexpos[3 * idx + 1], 2.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(nodexpos[3 * idx + 2], 2.0, MjTol(1e-12, 1e-5));
}

TEST_F(ShellTFITest, AffineDeformation) {
  // 4x4x4 grid with 8 interior nodes. Apply affine transform to boundary,
  // then verify TFI reproduces the same affine transform on interior nodes.
  constexpr int nx = 4, ny = 4, nz = 4;
  mjtNum nodexpos[3 * nx * ny * nz];
  MakeRegularGrid(nodexpos, nx, ny, nz);

  // affine: F(x,y,z) = A*[x,y,z]^T + b
  // A = [[2, 0.5, 0], [0.3, 1.5, 0], [0, 0, 1]], b = [10, 20, 30]
  auto affine = [](mjtNum x, mjtNum y, mjtNum z, mjtNum out[3]) {
    out[0] = 2.0 * x + 0.5 * y + 10.0;
    out[1] = 0.3 * x + 1.5 * y + 20.0;
    out[2] = z + 30.0;
  };

  // apply affine to all nodes
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int idx = i * ny * nz + j * nz + k;
        affine((mjtNum)i, (mjtNum)j, (mjtNum)k, nodexpos + 3 * idx);
      }
    }
  }

  // corrupt interior nodes to verify TFI actually reconstructs them
  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int k = 1; k < nz - 1; k++) {
        int idx = i * ny * nz + j * nz + k;
        nodexpos[3 * idx + 0] = -999;
        nodexpos[3 * idx + 1] = -999;
        nodexpos[3 * idx + 2] = -999;
      }
    }
  }

  // run TFI
  mju_shellTrackInterior(nodexpos, nx, ny, nz);

  // check all interior nodes match affine
  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int k = 1; k < nz - 1; k++) {
        int idx = i * ny * nz + j * nz + k;
        mjtNum expected[3];
        affine((mjtNum)i, (mjtNum)j, (mjtNum)k, expected);
        EXPECT_NEAR(nodexpos[3 * idx + 0], expected[0], MjTol(1e-12, 1e-4))
            << "i=" << i << " j=" << j << " k=" << k;
        EXPECT_NEAR(nodexpos[3 * idx + 1], expected[1], MjTol(1e-12, 1e-4))
            << "i=" << i << " j=" << j << " k=" << k;
        EXPECT_NEAR(nodexpos[3 * idx + 2], expected[2], MjTol(1e-12, 1e-4))
            << "i=" << i << " j=" << j << " k=" << k;
      }
    }
  }
}

TEST_F(ShellTFITest, BoundaryUnmodified) {
  // verify that boundary nodes are not modified by TFI
  constexpr int nx = 4, ny = 4, nz = 4;
  mjtNum nodexpos[3 * nx * ny * nz];
  MakeRegularGrid(nodexpos, nx, ny, nz);

  // save boundary node values
  mjtNum saved[3 * nx * ny * nz];
  mju_copy(saved, nodexpos, 3 * nx * ny * nz);

  mju_shellTrackInterior(nodexpos, nx, ny, nz);

  // check all boundary nodes unchanged
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        bool is_boundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 ||
                            k == 0 || k == nz - 1);
        if (is_boundary) {
          int idx = i * ny * nz + j * nz + k;
          EXPECT_EQ(nodexpos[3 * idx + 0], saved[3 * idx + 0]);
          EXPECT_EQ(nodexpos[3 * idx + 1], saved[3 * idx + 1]);
          EXPECT_EQ(nodexpos[3 * idx + 2], saved[3 * idx + 2]);
        }
      }
    }
  }
}

TEST_F(ShellTFITest, NoInteriorSmallGrid) {
  // 2x2x2 and 2x3x2: no interior nodes, TFI should be a no-op
  constexpr int nx = 2, ny = 3, nz = 2;
  mjtNum nodexpos[3 * nx * ny * nz];
  MakeRegularGrid(nodexpos, nx, ny, nz);

  mjtNum saved[3 * nx * ny * nz];
  mju_copy(saved, nodexpos, 3 * nx * ny * nz);

  mju_shellTrackInterior(nodexpos, nx, ny, nz);

  // all nodes unchanged
  for (int i = 0; i < 3 * nx * ny * nz; i++) {
    EXPECT_EQ(nodexpos[i], saved[i]);
  }
}

TEST_F(ShellTFITest, CoonsWeightsAndScatterMatchTrackInterior) {
  constexpr int nx = 5, ny = 4, nz = 6;
  constexpr int nnode = nx * ny * nz;
  mjtNum nodexpos[3 * nnode];
  for (int idx = 0; idx < nnode; idx++) {
    nodexpos[3 * idx + 0] = 0.7 * (idx % 5) - 0.2 * (idx % 3);
    nodexpos[3 * idx + 1] = -0.4 * (idx % 7) + 0.1 * idx;
    nodexpos[3 * idx + 2] = 0.3 * (idx % 11) - 0.5;
  }
  mju_shellTrackInterior(nodexpos, nx, ny, nz);

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int k = 1; k < nz - 1; k++) {
        int nb = 0;
        int nodes[32];
        mjtNum weights[32];
        mju_shellTFIWeights(nx, ny, nz, i, j, k, 1.0, &nb, nodes, weights,
                            nullptr, 0);
        EXPECT_EQ(nb, 26);
        mjtNum wsum = 0;
        mjtNum recon[3] = {0, 0, 0};
        for (int m = 0; m < nb; m++) {
          wsum += weights[m];
          for (int d = 0; d < 3; d++) {
            recon[d] += weights[m] * nodexpos[3 * nodes[m] + d];
          }
        }
        EXPECT_NEAR(wsum, 1.0, MjTol(1e-12, 1e-5));
        int idx = i * ny * nz + j * nz + k;
        for (int d = 0; d < 3; d++) {
          EXPECT_NEAR(nodexpos[3 * idx + d], recon[d], MjTol(1e-12, 1e-5));
        }
      }
    }
  }

  // Check exact transpose duality <f_all, T x_bnd> == <T^T f_all, x_bnd>.
  mjtNum f_all[3 * nnode], f_scatter[3 * nnode];
  for (int k = 0; k < 3 * nnode; k++) {
    f_all[k] = 0.15 * ((k % 13) - 6);
    f_scatter[k] = f_all[k];
  }
  mju_shellScatterInterior(f_scatter, nx, ny, nz);

  mjtNum dot_track = mju_dot(f_all, nodexpos, 3 * nnode);
  mjtNum dot_scatter = 0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int idx = i * ny * nz + j * nz + k;
        bool is_boundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 ||
                            k == 0 || k == nz - 1);
        if (is_boundary) {
          dot_scatter += mju_dot3(f_scatter + 3 * idx, nodexpos + 3 * idx);
        } else {
          EXPECT_EQ(f_scatter[3 * idx + 0], 0.0);
          EXPECT_EQ(f_scatter[3 * idx + 1], 0.0);
          EXPECT_EQ(f_scatter[3 * idx + 2], 0.0);
        }
      }
    }
  }
  EXPECT_NEAR(dot_scatter, dot_track, MjTol(1e-11, 1e-4));
}

TEST_F(UtilMiscTest, Round) {
  // basic rounding
  EXPECT_EQ(mju_round(2.3), 2);
  EXPECT_EQ(mju_round(2.7), 3);
  EXPECT_EQ(mju_round(-2.3), -2);
  EXPECT_EQ(mju_round(-2.7), -3);

  // exact integers
  EXPECT_EQ(mju_round(0.0), 0);
  EXPECT_EQ(mju_round(3.0), 3);
  EXPECT_EQ(mju_round(-3.0), -3);

  // ties: round() rounds away from zero
  EXPECT_EQ(mju_round(0.5), 1);
  EXPECT_EQ(mju_round(1.5), 2);
  EXPECT_EQ(mju_round(-0.5), -1);
  EXPECT_EQ(mju_round(-1.5), -2);

  // overflow clamps to INT_MAX/INT_MIN
  EXPECT_EQ(mju_round(1e18), INT_MAX);
  EXPECT_EQ(mju_round(-1e18), INT_MIN);
}

TEST_F(UtilMiscTest, EncodeDecodePyramidDim1) {
  const mjtNum force[1] = {25.0};
  const mjtNum mu[1] = {0.5};
  mjtNum pyramid[2] = {-999.0, -999.0};
  mjtNum decoded[1] = {0.0};

  mju_encodePyramid(pyramid, force, mu, /*dim=*/1);
  EXPECT_MJTNUM_EQ(pyramid[0], 25.0);

  mju_decodePyramid(decoded, pyramid, mu, /*dim=*/1);
  EXPECT_MJTNUM_EQ(decoded[0], 25.0);
}

TEST_F(UtilMiscTest, Type2StrAndStr2TypeRoundTrip) {
  for (int i = 1; i < mjNOBJECT; ++i) {
    const char* name = mju_type2Str(i);
    ASSERT_THAT(name, NotNull()) << "Object type " << i << " has null string";
    EXPECT_EQ(mju_str2Type(name), i) << "Round-trip failed for " << name;
  }
  EXPECT_STREQ(mju_type2Str(mjOBJ_FRAME), "frame");
  EXPECT_EQ(mju_str2Type("frame"), mjOBJ_FRAME);
}

}  // namespace
}  // namespace mujoco
