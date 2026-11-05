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

#include <sstream>
#include <string>

#include <Cocoa/Cocoa.h>
#include "studio/ux/file_dialog.h"

namespace mujoco::studio {

// Points the panel at the directory of `path` with its filename pre-filled,
// falling back to the user's Documents directory when no path is given.
// NSOpenPanel inherits from NSSavePanel.
static void SetInitialPath(NSSavePanel* panel, std::string_view path) {
  NSString* str = [[NSString alloc] initWithBytes:path.data()
                                           length:path.size()
                                         encoding:NSUTF8StringEncoding];
  if (str.length > 0) {
    [panel setDirectoryURL:
        [NSURL fileURLWithPath:[str stringByDeletingLastPathComponent]
                   isDirectory:YES]];
    [panel setNameFieldStringValue:[str lastPathComponent]];
  } else {
    NSURL* userDocumentsDir =
        [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory
                                             inDomains:NSUserDomainMask].firstObject;
    [panel setDirectoryURL:userDocumentsDir];
  }
}

DialogResult OpenFileDialog(std::string_view path,
                            std::span<std::string_view> filters) {
  NSOpenPanel* panel = [NSOpenPanel openPanel];
  SetInitialPath(panel, path);

  DialogResult result;
  if ([panel runModal] == NSModalResponseOK) {
    std::ostringstream s;
    s << [panel.URL.path cStringUsingEncoding:NSUTF8StringEncoding];
    result.path = s.str();
    result.status = DialogResult::kAccepted;
  } else {
    result.status = DialogResult::kCancelled;
  }
  return result;
}

DialogResult SaveFileDialog(std::string_view path,
                            std::span<std::string_view> filters) {
  NSSavePanel* panel = [NSSavePanel savePanel];
  SetInitialPath(panel, path);

  DialogResult result;
  if ([panel runModal] == NSModalResponseOK) {
    std::ostringstream s;
    s << [panel.URL.path cStringUsingEncoding:NSUTF8StringEncoding];
    result.path = s.str();
    result.status = DialogResult::kAccepted;
  } else {
    result.status = DialogResult::kCancelled;
  }
  return result;
}

DialogResult SelectPathDialog(std::string_view path) {
  return OpenFileDialog(path);
}

}  // namespace mujoco::studio
