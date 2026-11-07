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

#ifndef MUJOCO_PYTHON_RESOURCE_PROVIDER_H_
#define MUJOCO_PYTHON_RESOURCE_PROVIDER_H_

#include <pybind11/pybind11.h>

namespace mujoco::python {

// Registers the Python resource provider API (mjpResourceProvider, MjResource,
// mjp_registerProvider, mjp_findProvider and mjp_defaultProvider).
void InitResourceProvider(pybind11::module_& m);

}  // namespace mujoco::python

#endif  // MUJOCO_PYTHON_RESOURCE_PROVIDER_H_
