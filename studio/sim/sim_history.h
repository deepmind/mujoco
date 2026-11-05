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

#ifndef MUJOCO_STUDIO_SIM_SIM_HISTORY_H_
#define MUJOCO_STUDIO_SIM_SIM_HISTORY_H_

#include <algorithm>
#include <span>
#include <vector>
#include <mujoco/mujoco.h>

namespace mujoco::studio {

// A historical buffer of simulation data.
class SimHistory {
 public:
  SimHistory() = default;

  // Simulation data is stored a an array of mjtNum; see mj_getState().
  using State = std::vector<mjtNum>;

  // Clears and initializes the history buffer to store state.
  void Init(int state_size, int max_history = 2000, int max_bytes = 1e8);

  // Adds an uninitialized state to the history and returns a reference to it
  // so that the caller can populate the data. Also resets the current index to
  // 0; see SetIndex() for details.
  std::span<mjtNum> AddToHistory();

  // Returns the history at the given index (i.e. the number of steps) in the
  // past. The `offset` will be clamped internally to the range [0, Size() - 1].
  // This function returns the valid, clamped value.
  //
  // For example, calling SetIndex(0) will return the most recently recorded
  // state. Calling SetIndex(-N) will return the state from N steps ago.
  //
  // Note that future calls to `AddToHistory` will begin recording from the
  // newly set index, effectively creating a new "branch" of the history buffer.
  // If you do not want to lose any states, you must call `SetIndex(0)` before
  // before resuming playback.
  //
  // For example, consider you have recorded 6 states:
  //   N(-5)   N(-4)   N(-3)   N(-2)   N(-1)   N(0)
  //   -----------------------------------------^
  //
  // You then call `SetIndex(-3)`:
  //   N(-5)   N(-4)   N(-3)   N(-2)   N(-1)   N(0)
  //   ------------------^
  //
  // And then call `AddToHistory()
  //   N(-5)   N(-4)   N(-3)   N(-2)   N(-1)   N(0)
  //     |       |       |       x       x       x
  //     v       v       v
  //   N'(-3)  N'(-2)  N'(-1)  N'(0)
  //   -------------------------^
  //
  // In this case, we effectively erase states 0 to -3 of the previous history,
  // "copy" the older states into the new branch, and add the most recent state
  // at the "head" of the history buffer.
  std::span<mjtNum> SetIndex(int offset);

  // Returns the currently set index.
  int GetIndex() const { return offset_; }

  // Returns the number of states in the history buffer.
  int Size() const { return size_; }

  // Sets the total number of states in the history buffer. Clamped to [0,
  // history_.size()].
  void SetSize(int size) {
    size_ = std::clamp(size, 0, static_cast<int>(history_.size()));
    offset_ = size_ > 0 ? std::clamp(offset_, 1 - size_, 0) : 0;
  }

 private:
  // The history of states.
  std::vector<State> history_;

  // The index at which the next AddToHistory() call will write.
  int cursor_ = 0;

  // The most recently requested offset from SetIndex().
  int offset_ = 0;

  // The total number of states available in the history buffer.
  int size_ = 0;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_SIM_SIM_HISTORY_H_
