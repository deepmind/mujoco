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

#include "studio/sim/sim_history.h"

#include <algorithm>
#include <climits>
#include <span>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

void SimHistory::Init(int state_size, int max_history, int max_bytes) {
  // History buffer will be smaller of number of states and total memory.
  const int state_bytes = state_size * sizeof(mjtNum);
  const int history_length = std::min(INT_MAX / state_bytes, max_history);
  const int history_bytes = std::min(state_bytes * history_length, max_bytes);

  const int max_states = std::max(1, history_bytes / state_bytes);
  history_.resize(max_states);

  for (State& state : history_) {
    state.resize(state_size, 0);
  }
  cursor_ = 0;
  offset_ = 0;
  size_ = 0;
}

std::span<mjtNum> SimHistory::AddToHistory() {
  const int max_size = history_.size();
  if (offset_ != 0) {
    // offset will be a negative number between 1 - history_.size() and 0.
    size_ += offset_;
    cursor_ += offset_;
    if (cursor_ < 0) {
      cursor_ += max_size;
    }
    offset_ = 0;
  }

  std::span<mjtNum> state;
  if (max_size > 0 && cursor_ < max_size) {
    state = history_[cursor_];
    cursor_ = (cursor_ + 1) % max_size;
    size_ = std::min(size_ + 1, max_size);
  }
  return state;
}

std::span<mjtNum> SimHistory::SetIndex(int offset) {
  if (history_.empty() || size_ <= 0) {
    return {};
  }

  offset_ = std::clamp<int>(offset, 1 - size_, 0);

  const int actual_index = (cursor_ - 1 + offset_ + history_.size()) % history_.size();
  return history_[actual_index];
}

}  // namespace mujoco::studio
