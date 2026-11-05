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

namespace mujoco::studio {

void RoleTransitions::ExpectToast(const char* text, double now) {
  expected_toast_ = text;
  expected_at_ = now;
}

void RoleTransitions::ClearExpectedToast() { expected_toast_ = nullptr; }

const char* RoleTransitions::Update(SessionRole role, double now) {
  if (expected_toast_ != nullptr && now - expected_at_ > kExpectSec) {
    expected_toast_ = nullptr;
  }
  if (role == SessionRole::kClaiming || role == settled_role_) {
    return nullptr;
  }
  const char* toast = nullptr;
  if (settled_role_ != SessionRole::kClaiming) {
    if (expected_toast_ != nullptr) {
      toast = expected_toast_;
    } else if (role == SessionRole::kControlling) {
      toast = "You now have control";
    } else {
      toast = "Another viewer took control";
    }
  }
  settled_role_ = role;
  expected_toast_ = nullptr;
  return toast;
}

}  // namespace mujoco::studio
