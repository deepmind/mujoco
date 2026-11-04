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

#include <assert.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "studio/ux/file_dialog.h"

namespace mujoco::studio {

static DialogResult RunZenity(std::vector<std::string>& args) {
  std::string cmd = "zenity";
  for (const std::string& arg : args) {
    cmd += " " + arg;
  }

  static constexpr int kBufferSize = 1024;
  FILE* output = popen(cmd.c_str(), "r");
  if (!output) {
    return DialogResult{.status = DialogResult::kCancelled};
  }
  char buffer[kBufferSize];
  auto ret = fgets(buffer, kBufferSize, output);
  if (ret == nullptr || ferror(output)) {
    return DialogResult{.status = DialogResult::kError};
  }
  // Zenity outputs a newline at the end of the buffer which we need to remove.
  const int len = std::strlen(buffer);
  if (len > 0) {
    buffer[len - 1] = '\0';
  }
  return DialogResult{.status = DialogResult::kAccepted, .path = std::string(buffer)};
}

static void UpdateArgs(std::vector<std::string>& args, std::string_view path,
                       std::span<std::string_view> filters = {}) {
  if (!path.empty() && path.front() != 0) {
    args.push_back("--filename=" + std::string(path.data()));
  }
  for (std::string_view filter : filters) {
    std::string pattern = FormatFileFilter(filter, ' ');
    if (!pattern.empty()) {
      args.push_back("--file-filter=" + pattern);
    }
  }
}

DialogResult OpenFileDialog(std::string_view path, std::span<std::string_view> filters) {
  std::vector<std::string> args;
  args.push_back("--file-selection");
  args.push_back("--title=Open File");
  UpdateArgs(args, path, filters);
  return RunZenity(args);
}

DialogResult SaveFileDialog(std::string_view path, std::span<std::string_view> filters) {
  std::vector<std::string> args;
  args.push_back("--file-selection");
  args.push_back("--title=Save File");
  args.push_back("--save");
  UpdateArgs(args, path, filters);
  return RunZenity(args);
}

DialogResult SelectPathDialog(std::string_view path) {
  std::vector<std::string> args;
  args.push_back("--file-selection");
  args.push_back("--title=Select Folder");
  args.push_back("--directory");
  UpdateArgs(args, path);
  return RunZenity(args);
}

}  // namespace mujoco::studio
