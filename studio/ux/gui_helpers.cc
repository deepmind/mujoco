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

#include "studio/ux/gui_helpers.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "engine/engine_vis_visualize.h"

namespace mujoco::studio {

void SaveText(const std::string& contents, const std::string& filename) {
  std::ofstream file(filename);
  file.write(contents.data(), contents.size());
  file.close();
}

std::string LoadText(const std::string& filename) {
  std::ifstream file(filename);
  std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();
  return contents;
}

const void* GetValue(const mjModel* model, const mjData* data, const char* field, int index) {
#define X(TYPE, NAME, NR, NC)                                        \
  if (!std::strcmp(#NAME, field) && !std::strcmp(#TYPE, "mjtNum")) { \
    if (index >= 0 && index < model->NR * NC) {                      \
      return &data->NAME[index];                                     \
    } else {                                                         \
      return nullptr;                                                \
    }                                                                \
  }
  MJDATA_POINTERS
#undef X
  return nullptr;  // Invalid field.
}

std::string CameraToString(const mjData* data, const mjvCamera* camera) {
  mjtNum pos[3], forward[3], up[3], right[3];
  mjv_cameraFrame(pos, forward, up, right, data, camera);
  char str[500];
  std::snprintf(str, sizeof(str),
                "<camera pos=\"%.3f %.3f %.3f\" xyaxes=\"%.3f %.3f %.3f %.3f "
                "%.3f %.3f\"/>\n",
                pos[0], pos[1], pos[2], right[0], right[1], right[2], up[0], up[1], up[2]);
  return str;
}

std::string KeyframeToString(const mjModel* model, const mjData* data, bool full_precision) {
  char buf[200];
  const char p_regular[] = "%g";
  const char p_full[] = "%-22.16g";
  const char* format = full_precision ? p_full : p_regular;

  std::string str = "<key\n";

  // time
  str += " time=\"";
  std::snprintf(buf, sizeof(buf), format, data->time);
  str += buf;

  // qpos
  str += "\"\n  qpos=\"";
  for (int i = 0; i < model->nq; i++) {
    std::snprintf(buf, sizeof(buf), format, data->qpos[i]);
    str += buf;
    if (i < model->nq - 1) str += " ";
  }

  // qvel
  str += "\"\n  qvel=\"";
  for (int i = 0; i < model->nv; i++) {
    std::snprintf(buf, sizeof(buf), format, data->qvel[i]);
    str += buf;
    if (i < model->nv - 1) str += " ";
  }

  // act
  if (model->na > 0) {
    str += "\"\n  act=\"";
    for (int i = 0; i < model->na; i++) {
      std::snprintf(buf, sizeof(buf), format, data->act[i]);
      str += buf;
      if (i < model->na - 1) str += " ";
    }
  }

  // ctrl
  if (model->nu > 0) {
    str += "\"\n  ctrl=\"";
    for (int i = 0; i < model->nu; i++) {
      std::snprintf(buf, sizeof(buf), format, data->ctrl[i]);
      str += buf;
      if (i < model->nu - 1) str += " ";
    }
  }

  if (model->nmocap > 0) {
    str += "\"\n  mpos=\"";
    for (int i = 0; i < 3 * model->nmocap; i++) {
      std::snprintf(buf, sizeof(buf), format, data->mocap_pos[i]);
      str += buf;
      if (i < 3 * model->nmocap - 1) str += " ";
    }

    // mocap_quat
    str += "\"\n  mquat=\"";
    for (int i = 0; i < 4 * model->nmocap; i++) {
      std::snprintf(buf, sizeof(buf), format, data->mocap_quat[i]);
      str += buf;
      if (i < 4 * model->nmocap - 1) str += " ";
    }
  }

  str += "\"\n/>";
  return str;
}

}  // namespace mujoco::studio
