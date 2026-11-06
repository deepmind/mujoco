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

#ifndef MUJOCO_PYTHON_USD_PYBIND_H_
#define MUJOCO_PYTHON_USD_PYBIND_H_

#include "pybind_boost_py_utils.h"
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/timeCode.h>

// Add type casts here for any types needed.
REGISTER_BOOST_PYTHON_INTEROP(pxr::UsdStagePtr, _("pxr.Usd.Stage"));
REGISTER_BOOST_PYTHON_INTEROP(pxr::UsdTimeCode, _("pxr.Usd.TimeCode"));
REGISTER_BOOST_PYTHON_INTEROP(pxr::SdfPath, _("pxr.Sdf.Path"));

#endif  // MUJOCO_PYTHON_USD_PYBIND_H_
