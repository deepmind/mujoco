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

#ifndef MUJOCO_STUDIO_UX_CONSOLE_H_
#define MUJOCO_STUDIO_UX_CONSOLE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Message counts for the Studio Console since it was last cleared. A message
// may span several lines.
struct ConsoleCounts {
  int64_t messages = 0;
  int64_t warnings = 0;
  int64_t errors = 0;
};

// The Studio Console: the log of engine and Studio messages that the Console
// window shows and the status bar summarizes. There is one per process,
// Console::Get(), like the engine's log handler. All methods are thread-safe.
class Console {
 public:
  // The console keeps only the most recent lines that fit in this much memory:
  // their text plus a per-line overhead of sizeof(Line), 32-40 bytes depending
  // on the standard library (see LineBytes).
  //
  // Assuming ~80 chars per line (~112 bytes with overhead, ~9,300 lines/MiB):
  //   64 MiB (desktop) holds ~600k lines: ~10 min at a worst-case 1 kHz
  //     (1 line/step, ~112 KiB/s), or ~1.6 hours at 100 lines/s.
  //   16 MiB (browser, 32-bit WASM heap) holds ~150k lines: ~2.5 min at 1 kHz,
  //     or ~25 min at 100 lines/s.
#ifdef __EMSCRIPTEN__
  static constexpr size_t kMaxBytes = size_t{16} << 20;
#else
  static constexpr size_t kMaxBytes = size_t{64} << 20;
#endif

  // Returns the process-wide console.
  static Console& Get();

  // Installs a log handler via mju_setLogHandler that adds every engine
  // message to the console (info and debug messages only from enabled topics)
  // and then passes it on to the handler it replaced, which prints it and
  // writes it to the log file as before. Only the first call installs the
  // handler.
  static void Init();

  // Returns the memory a line with this text counts for towards the cap.
  static size_t LineBytes(std::string_view text);

  explicit Console(size_t max_bytes = kMaxBytes) : max_bytes_(max_bytes) {}
  Console(const Console&) = delete;
  Console& operator=(const Console&) = delete;

  // Clears all messages and resets the counts.
  void Clear();

  // Adds the message at the given level (mjLOG_DEBUG, mjLOG_INFO,
  // mjLOG_WARNING, mjLOG_ERROR) to the console and, like the default log
  // handler, follows mju_getLogConfig() for the rest: prints it to the
  // terminal when logto_console is set (warnings and errors to stderr, others
  // to stdout) and appends it to logfile when logto_file is set, in the same
  // format as the default handler's output.
  void Log(mjtLogLevel level, std::string_view message);

  // Adds a message at the given level without printing it, one line per line
  // of `message` and then of `body`. Use it for messages that are printed
  // elsewhere, e.g. engine messages and Python log records.
  void Add(mjtLogLevel level, std::string_view message, std::string_view body = "");

  // Returns the number of messages, warnings, and errors.
  ConsoleCounts GetCounts() const;

  // Returns a copy of the lines currently in the console, oldest first. It
  // copies every line, up to kMaxBytes of text, so use it to read the console
  // on demand, e.g. in tests and scripts, rather than every frame.
  std::vector<std::string> GetLines() const;

  // UX for the Console window: a scrollable table of log lines with
  // alternating row backgrounds, drawn in the current ImGui window.
  void WindowGui();

  // UX for the console summary in the status bar. Displays the number of
  // messages, warnings, and errors; clicking toggles `*show_console`.
  void StatusGui(bool* show_console);

 private:
  struct Line {
    mjtLogLevel level = mjLOG_INFO;
    std::string text;
  };

  void AppendLinesLocked(mjtLogLevel level, std::string_view prefix, std::string_view text);
  void ClearLocked();

  const size_t max_bytes_ = kMaxBytes;
  mutable std::mutex mutex_;
  std::deque<Line> lines_;
  size_t total_bytes_used_ = 0;  // sum of LineBytes over lines_
  int64_t dropped_lines_ = 0;    // oldest lines dropped to stay within max_bytes_
  ConsoleCounts counts_;
  bool auto_scroll_ = true;
  bool scroll_to_bottom_ = false;
  float last_scroll_max_y_ = 0.0f;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_CONSOLE_H_
