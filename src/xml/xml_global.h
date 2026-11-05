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


#ifndef MUJOCO_SRC_XML_XML_GLOBAL_H_
#define MUJOCO_SRC_XML_XML_GLOBAL_H_

#include <string>

#include <mujoco/mujoco.h>

extern "C" {
MJAPI void _mjPRIVATE__set_xml_spec(mjSpec* spec = nullptr);
}

inline void SetGlobalXmlSpec(mjSpec* spec = nullptr) {
  _mjPRIVATE__set_xml_spec(spec);
}

std::string GetGlobalXmlSpec(const mjModel* m, char* error, int error_sz);

#endif  // MUJOCO_SRC_XML_XML_GLOBAL_H_
