// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "role_transitions.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mujoco::studio {
namespace {

using ::testing::IsNull;
using ::testing::StrEq;

TEST(RoleTransitionsTest, FirstSettledRoleOnPageLoadIsNotATransition) {
  RoleTransitions t;
  EXPECT_THAT(t.Update(SessionRole::kClaiming, 0.0), IsNull());
  EXPECT_THAT(t.Update(SessionRole::kSpectating, 0.1), IsNull());
  EXPECT_EQ(t.settled_role(), SessionRole::kSpectating);
  // Staying in the role is not a transition either.
  EXPECT_THAT(t.Update(SessionRole::kSpectating, 0.2), IsNull());
}

TEST(RoleTransitionsTest, ClaimingToControllingAfterLoadToastsAGrant) {
  RoleTransitions t;
  t.Update(SessionRole::kSpectating, 0.0);
  // A grant passes through claiming on the way to controlling.
  EXPECT_THAT(t.Update(SessionRole::kClaiming, 1.0), IsNull());
  EXPECT_THAT(t.Update(SessionRole::kControlling, 1.5),
              StrEq("You now have control"));
  EXPECT_EQ(t.settled_role(), SessionRole::kControlling);
}

TEST(RoleTransitionsTest, ExternalStealToastsThatAnotherViewerTookControl) {
  RoleTransitions t;
  t.Update(SessionRole::kControlling, 0.0);
  EXPECT_THAT(t.Update(SessionRole::kSpectating, 1.0),
              StrEq("Another viewer took control"));
}

TEST(RoleTransitionsTest, OwnClickUsesItsOwnWordingOnTheNextFlip) {
  RoleTransitions t;
  t.Update(SessionRole::kSpectating, 0.0);
  t.ExpectToast("You took control", 1.0);
  EXPECT_THAT(t.Update(SessionRole::kClaiming, 1.1), IsNull());
  EXPECT_THAT(t.Update(SessionRole::kControlling, 1.2),
              StrEq("You took control"));
  // The expectation is consumed by the flip.
  EXPECT_THAT(t.Update(SessionRole::kSpectating, 2.0),
              StrEq("Another viewer took control"));
}

TEST(RoleTransitionsTest, ClearedExpectationFallsBackToGenericWording) {
  RoleTransitions t;
  t.Update(SessionRole::kSpectating, 0.0);
  t.ExpectToast("You took control", 1.0);
  t.ClearExpectedToast();
  EXPECT_THAT(t.Update(SessionRole::kControlling, 1.5),
              StrEq("You now have control"));
}

TEST(RoleTransitionsTest, IgnoredClickExpiresSoALaterGrantIsNotMislabelled) {
  RoleTransitions t;
  t.Update(SessionRole::kSpectating, 0.0);
  t.ExpectToast("You took control", 1.0);  // The server ignores the steal.
  for (double now = 1.1; now < 1.0 + RoleTransitions::kExpectSec; now += 0.5) {
    EXPECT_THAT(t.Update(SessionRole::kSpectating, now), IsNull());
  }
  // Much later, a queue grant arrives.
  EXPECT_THAT(t.Update(SessionRole::kControlling,
                       1.0 + RoleTransitions::kExpectSec + 10.0),
              StrEq("You now have control"));
}

TEST(RoleTransitionsTest, ExpectationWithinTheWindowStillApplies) {
  RoleTransitions t;
  t.Update(SessionRole::kSpectating, 0.0);
  t.ExpectToast("You took control", 1.0);
  EXPECT_THAT(t.Update(SessionRole::kControlling,
                       1.0 + RoleTransitions::kExpectSec - 0.1),
              StrEq("You took control"));
}

}  // namespace
}  // namespace mujoco::studio
