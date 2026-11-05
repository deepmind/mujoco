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

#ifndef MUJOCO_PLUGIN_SDF_SDF_H_
#define MUJOCO_PLUGIN_SDF_SDF_H_

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::plugin::sdf {

inline mjtNum Union(mjtNum a, mjtNum b) { return mju_min(a, b); }

inline mjtNum Intersection(mjtNum a, mjtNum b) { return mju_max(a, b); }

inline mjtNum Subtraction(mjtNum a, mjtNum b) { return mju_max(a, -b); }

inline mjtNum Fract(mjtNum x) { return x - floor(x); }

// reads numeric attributes
bool CheckAttr(const char* name, const mjModel* m, int instance);

// converts attributes to numeric or returns default if not present
template <typename T>
class SdfDefault {
 public:
  SdfDefault() {
    for (int i = 0; i < T::nattribute; i++) {
      default_[T::names[i]] = T::defaults[i];
    }
  }

  // get a single default value, NaN if the value is invalid
  mjtNum GetDefault(const char* name, const char* value) {
    // empty or whitespace, as in CheckAttr
    if (!value[std::strspn(value, " \t\n\v\f\r")]) {
      return default_[name];
    }
    char* end = nullptr;
    mjtNum num = strtod(value, &end);
    if (end == value || end[std::strspn(end, " \t\n\v\f\r")]) {
      return std::nan("");
    }
    return num;
  }

  // populate attribute array
  void GetDefaults(mjtNum* attribute, const char* names[],
                   const char* values[]) {
    for (int i = 0; i < default_.size(); i++) {
      attribute[i] = GetDefault(names[i], values[i]);
    }
  }

 private:
  std::map<std::string, mjtNum> default_;
};

// stores the history of gradient descent iterations
class SdfVisualizer {
 public:
  SdfVisualizer();
  SdfVisualizer(SdfVisualizer&&) = default;
  SdfVisualizer& operator=(SdfVisualizer&&) = default;

  void Visualize(const mjModel* m, const mjData* d, const mjvOption* opt,
                 mjvScene* scn, int instance);

  void AddPoint(const mjtNum point[3]);
  void Reset();
  void Next();  // adds a new gradient descent trajectory to be visualized

 private:
  std::unique_ptr<std::mutex> mutex_ = std::make_unique<std::mutex>();
  std::vector<mjtNum> points_;  // query points
  std::vector<int> npoints_;    // number of iterations from the starting point
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_SDF_H_
