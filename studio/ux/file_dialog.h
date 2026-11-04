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

#ifndef MUJOCO_STUDIO_UX_FILE_DIALOG_H_
#define MUJOCO_STUDIO_UX_FILE_DIALOG_H_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace mujoco::studio {

// Formats a comma-separated extension list (e.g. "xml,mjb") as wildcard
// patterns joined by `separator` (e.g. "*.xml *.mjb" or "*.xml;*.mjb").
inline std::string FormatFileFilter(std::string_view filter, char separator) {
  std::string result;
  while (!filter.empty()) {
    const size_t pos = filter.find(',');
    const std::string_view ext = filter.substr(0, pos);
    if (!ext.empty()) {
      if (!result.empty()) {
        result += separator;
      }
      result += "*.";
      result.append(ext);
    }
    if (pos == std::string_view::npos) {
      break;
    }
    filter.remove_prefix(pos + 1);
  }
  return result;
}

// Result returned by various file dialog functions.
struct DialogResult {
  // The status of the dialog.
  enum Status {
    // The dialog is still open.
    kPending,
    // A file/folder was selected.
    kAccepted,
    // The dialog was dismissed.
    kCancelled,
    // An error occurred.
    kError,
  };

  // The status of the dialog.
  Status status = kPending;

  // The path selected by the dialog if Status is kAccepted. If status is
  // kError, the path may contain an error message. Empty otherwise.
  std::string path;
};

// Opens a file dialog used for selecting a file to open.
DialogResult OpenFileDialog(std::string_view path, std::span<std::string_view> filters = {});

// Opens a file dialog used for selecting a file to save.
DialogResult SaveFileDialog(std::string_view path, std::span<std::string_view> filters = {});

// Opens a file dialog used for selecting a folder.
DialogResult SelectPathDialog(std::string_view path);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_FILE_DIALOG_H_
