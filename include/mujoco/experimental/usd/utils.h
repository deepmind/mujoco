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

#ifndef MUJOCO_SRC_EXPERIMENTAL_USD_UTILS_H_
#define MUJOCO_SRC_EXPERIMENTAL_USD_UTILS_H_

#include <mujoco/mujoco.h>
#include <pxr/usd/sdf/path.h>

namespace mujoco {
namespace usd {

// Sets a user value on an mjsElement with the key "usd_primpath".
void SetUsdPrimPathUserValue(mjsElement* element,
                             const pxr::SdfPath& prim_path);

// Gets the user value associated with the key "usd_primpath" from an
// mjsElement. Returns empty pxr::SdfPath() if the value is not found.
pxr::SdfPath GetUsdPrimPathUserValue(mjsElement* element);

}  // namespace usd
}  // namespace mujoco

#endif  // MUJOCO_SRC_EXPERIMENTAL_USD_UTILS_H_
