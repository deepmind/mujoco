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

#ifndef MUJOCO_STUDIO_UX_WELL_TEMPERED_H_
#define MUJOCO_STUDIO_UX_WELL_TEMPERED_H_

#include <array>
#include <cmath>

namespace mujoco::studio {

// Steps `value` to the next value in a "well-tempered" logarithmic ladder, in
// the given direction (+1 up, -1 down). The ladder is spaced roughly evenly in
// log space -- like clicks of a slowdown control -- but each rung is snapped to
// a round number so that ten steps multiply by exactly ten, closing the decade
// on a power of ten:
//
//   ... 1, 1.3, 1.6, 2, 2.5, 3.2, 4, 5, 6.3, 8, 10, ...
//
// The step always lands strictly on the far side of `value`, so a value between
// two rungs snaps onto the ladder.
//
// `zero_below` gives the ladder a bottom rung of exactly zero: any value below
// it collapses to 0, so stepping down snaps to 0 and stepping up off 0 lands on
// `zero_below`. This lets quantities for which zero is meaningful (e.g. a
// solver tolerance) reach it. Quantities that never want zero leave
// `zero_below` at 0, in which case a non-positive value (or a zero direction)
// is returned unchanged.
inline double WellTemperedStep(double value, int direction, double zero_below = 0.0) {
  static constexpr std::array<double, 10> kLadder = {1.0, 1.3, 1.6, 2.0, 2.5,
                                                     3.2, 4.0, 5.0, 6.3, 8.0};
  if (direction == 0) {
    return value;
  }
  if (zero_below > 0 && value < zero_below) {
    return direction > 0 ? zero_below : 0.0;
  }
  if (!(value > 0)) {
    return value;
  }
  const double decade = std::floor(std::log10(value) + 1e-9);
  const double scale = std::pow(10.0, decade);
  const double mantissa = value / scale;  // in [1, 10)
  if (direction > 0) {
    for (double rung : kLadder) {
      if (rung > mantissa * (1.0 + 1e-6)) {
        return rung * scale;
      }
    }
    return 10.0 * scale;  // carry up: 1.0 in the next decade
  }
  double next = 0.8 * scale;  // carry down: 8.0 in the previous decade
  for (int i = kLadder.size() - 1; i >= 0; --i) {
    if (kLadder[i] < mantissa * (1.0 - 1e-6)) {
      next = kLadder[i] * scale;
      break;
    }
  }
  return (zero_below > 0 && next < zero_below) ? 0.0 : next;
}

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_WELL_TEMPERED_H_
