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

#import <AppKit/NSWindow.h>
#import <AppKit/NSView.h>

void* GetNativeWindowOsx(void* window) {
  NSWindow* ns_window = (__bridge NSWindow*)window;
  NSView* ns_view = [ns_window contentView];
  return (__bridge void*)ns_view;
}
