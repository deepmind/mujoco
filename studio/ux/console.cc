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

#include "studio/ux/console.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>
#include "studio/ux/imgui_widgets.h"

namespace mujoco::studio {
namespace {

// The log handler that Console::Init replaced and passes messages on to.
std::atomic<mjfLogHandler> prev_log_handler{nullptr};

// The log handler that Console::Init installs.
void LogHandler(const mjLogMessage* msg) {
  // Like the default handler, skip info and debug messages from topics that
  // are not enabled. Everything else goes in the console, whether or not the
  // default handler prints it.
  const bool topic_disabled = (msg->level == mjLOG_INFO || msg->level == mjLOG_DEBUG) &&
                              msg->topic > 0 &&
                              !(mju_getLogConfig().topics & (1 << (msg->topic - 1)));
  if (!topic_disabled) {
    Console::Get().Add(static_cast<mjtLogLevel>(msg->level), msg->subject,
                       msg->body ? msg->body : "");
  }

  // The previous handler prints the message and writes it to the log file as
  // before. It also handles errors, which must not return.
  if (mjfLogHandler prev = prev_log_handler.load(); prev != nullptr) {
    prev(msg);
  }
}

// Returns the prefix that shows the level of a message, e.g. "ERROR: ".
std::string_view LevelPrefix(mjtLogLevel level) {
  switch (level) {
    case mjLOG_ERROR:
      return "ERROR: ";
    case mjLOG_WARNING:
      return "WARNING: ";
    case mjLOG_INFO:
      return "INFO: ";
    case mjLOG_DEBUG:
      return "DEBUG: ";
  }
  return "DEBUG: ";  // Levels out of range, which Python can pass.
}

// Returns the text color for console lines and counts at the given level.
ImVec4 ConsoleTextColor(mjtLogLevel level) {
  const bool is_dark = ImGui::GetStyle().Colors[ImGuiCol_WindowBg].x < 0.5f;
  if (level >= mjLOG_ERROR) {
    return is_dark ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(0.8f, 0.1f, 0.1f, 1.0f);
  }
  if (level == mjLOG_WARNING) {
    return is_dark ? ImVec4(1.0f, 0.8f, 0.2f, 1.0f) : ImVec4(0.7f, 0.45f, 0.0f, 1.0f);
  }
  return ImGui::GetStyleColorVec4(ImGuiCol_Text);
}

}  // namespace

Console& Console::Get() {
  static auto* console = new Console();
  return *console;
}

void Console::Init() {
  static std::once_flag once;
  std::call_once(once, [] { prev_log_handler = mju_setLogHandler(&LogHandler); });
}

void Console::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  ClearLocked();
}

void Console::Log(mjtLogLevel level, std::string_view message) {
  Add(level, message);

  // Print and write the message as the default log handler does: "ERROR: ..."
  // on the terminal and "ERROR <time>: ..." in the log file, with a blank line
  // after every message except debug ones.
  const mjLogConfig cfg = mju_getLogConfig();
  const bool log_to_file = cfg.logto_file && cfg.logfile[0] != '\0';
  if (!cfg.logto_console && !log_to_file) {
    return;
  }
  const std::string_view prefix = LevelPrefix(level);
  if (message.starts_with(prefix)) {
    message.remove_prefix(prefix.size());
  }
  const std::string_view name = prefix.substr(0, prefix.size() - 2);  // No ": "
  const char* const end = level == mjLOG_DEBUG ? "\n" : "\n\n";
  if (cfg.logto_console) {
    FILE* stream = level >= mjLOG_WARNING ? stderr : stdout;
    std::fprintf(stream, "%.*s: %.*s%s", static_cast<int>(name.size()), name.data(),
                 static_cast<int>(message.size()), message.data(), end);
    std::fflush(stream);
  }
  if (log_to_file) {
    if (FILE* fp = std::fopen(cfg.logfile, "a+t")) {
      const std::time_t now = std::time(nullptr);
      std::tm local_time{};
#ifdef _WIN32
      localtime_s(&local_time, &now);
#else
      localtime_r(&now, &local_time);
#endif
      char time_str[64];
      std::strftime(time_str, sizeof(time_str), "%c", &local_time);
      std::fprintf(fp, "%.*s %s: %.*s%s", static_cast<int>(name.size()), name.data(), time_str,
                   static_cast<int>(message.size()), message.data(), end);
      std::fclose(fp);
    }
  }
}

void Console::Add(mjtLogLevel level, std::string_view message, std::string_view body) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++counts_.messages;
  if (level == mjLOG_WARNING) {
    ++counts_.warnings;
  } else if (level >= mjLOG_ERROR) {
    ++counts_.errors;
  }
  AppendLinesLocked(level, LevelPrefix(level), message);
  if (!body.empty()) {
    AppendLinesLocked(level, "", body);
  }
  if (auto_scroll_) {
    scroll_to_bottom_ = true;
  }
}

size_t Console::LineBytes(std::string_view text) { return sizeof(Line) + text.size(); }

ConsoleCounts Console::GetCounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return counts_;
}

std::vector<std::string> Console::GetLines() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> result;
  result.reserve(lines_.size());
  for (const Line& line : lines_) {
    result.push_back(line.text);
  }
  return result;
}

void Console::AppendLinesLocked(mjtLogLevel level, std::string_view prefix, std::string_view text) {
  bool first = true;
  size_t pos = 0;
  while (pos <= text.size()) {
    const size_t next = text.find('\n', pos);
    std::string_view line =
        (next == std::string_view::npos) ? text.substr(pos) : text.substr(pos, next - pos);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (next == std::string_view::npos && line.empty() && !first) {
      break;
    }
    std::string full_line;
    if (first && !prefix.empty() && !line.starts_with(prefix)) {
      full_line.reserve(prefix.size() + line.size());
      full_line.append(prefix);
      full_line.append(line);
    } else {
      full_line.assign(line);
    }
    lines_.push_back(Line{level, std::move(full_line)});
    total_bytes_used_ += LineBytes(lines_.back().text);
    while (total_bytes_used_ > max_bytes_ && lines_.size() > 1) {
      total_bytes_used_ -= LineBytes(lines_.front().text);
      lines_.pop_front();
      ++dropped_lines_;
    }
    first = false;
    if (next == std::string_view::npos) {
      break;
    }
    pos = next + 1;
  }
}

void Console::ClearLocked() {
  lines_.clear();
  total_bytes_used_ = 0;
  dropped_lines_ = 0;
  counts_ = ConsoleCounts();
  auto_scroll_ = true;
  scroll_to_bottom_ = false;
  last_scroll_max_y_ = 0.0f;
}

void Console::WindowGui() {
  ScopedStyle style;
  style.Font(ScopedFont::kMono);

  const ImGuiTableFlags table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                      ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("##ConsoleTable", 2, table_flags)) {
    std::optional<std::string> copy_to_clipboard;
    {
      // Draw the lines in place rather than copying them every frame. The
      // clipper only visits the visible rows, so the lock is held briefly.
      std::lock_guard<std::mutex> lock(mutex_);

      const float scroll_y = ImGui::GetScrollY();
      const float scroll_max_y = ImGui::GetScrollMaxY();
      if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) &&
          ImGui::GetIO().MouseWheel > 0.0f) {
        auto_scroll_ = false;
      } else if (scroll_y < last_scroll_max_y_ - 20.0f) {
        auto_scroll_ = false;
      } else if (scroll_y >= scroll_max_y - 15.0f) {
        auto_scroll_ = true;
      }
      last_scroll_max_y_ = scroll_max_y;

      if (ImGui::BeginPopupContextWindow("##ConsoleContext")) {
        if (ImGui::MenuItem("Clear")) {
          ClearLocked();
        }
        if (ImGui::MenuItem("Copy All")) {
          std::string all_text;
          all_text.reserve(total_bytes_used_);
          for (const Line& line : lines_) {
            all_text.append(line.text);
            all_text.push_back('\n');
          }
          copy_to_clipboard = std::move(all_text);
        }
        if (ImGui::BeginMenu("Logging")) {
          mjLogConfig cfg = mju_getLogConfig();
          bool log_terminal = cfg.logto_console;
          bool log_file = cfg.logto_file;
          if (ImGui::Checkbox("Terminal", &log_terminal)) {
            cfg.logto_console = log_terminal;
            mju_setLogConfig(cfg);
          }
          if (ImGui::Checkbox("File", &log_file)) {
            cfg.logto_file = log_file;
            mju_setLogConfig(cfg);
          }
          ImGui::SeparatorText("Topics");
          for (int i = 0; i < mjNTOPIC; ++i) {
            bool enabled = (cfg.topics & (1 << i)) != 0;
            if (ImGui::Checkbox(mjTOPICSTRING[i], &enabled)) {
              if (enabled) {
                cfg.topics |= (1 << i);
              } else {
                cfg.topics &= ~(1 << i);
              }
              mju_setLogConfig(cfg);
            }
          }
          ImGui::EndMenu();
        }
        ImGui::EndPopup();
      }

      const int num_lines = static_cast<int>(lines_.size());
      const int64_t last_line_number = dropped_lines_ + num_lines;
      const int digits = last_line_number > 0
                             ? std::max(3, static_cast<int>(std::log10(last_line_number)) + 1)
                             : 3;
      const float char_width = ImGui::CalcTextSize("0").x;

      ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, (digits + 1) * char_width);
      ImGui::TableSetupColumn("Log", ImGuiTableColumnFlags_WidthStretch);

      const bool do_scroll = auto_scroll_ && scroll_to_bottom_ && num_lines > 0;
      scroll_to_bottom_ = false;
      const ImVec4 warning_color = ConsoleTextColor(mjLOG_WARNING);
      const ImVec4 error_color = ConsoleTextColor(mjLOG_ERROR);
      const ImVec4 line_num_color = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);

      ImGuiListClipper clipper;
      clipper.Begin(num_lines);
      if (do_scroll) {
        clipper.IncludeItemByIndex(num_lines - 1);
      }
      while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
          const Line& line = lines_[i];
          ImGui::TableNextRow();

          ImGui::TableNextColumn();
          {
            ScopedStyle num_style;
            num_style.Color(ImGuiCol_Text, line_num_color);
            ImGui::Text("%" PRId64, dropped_lines_ + i + 1);
          }

          ImGui::TableNextColumn();
          if (line.level >= mjLOG_ERROR) {
            ScopedStyle err_style;
            err_style.Color(ImGuiCol_Text, error_color);
            ImGui::TextUnformatted(line.text.c_str());
          } else if (line.level == mjLOG_WARNING) {
            ScopedStyle warn_style;
            warn_style.Color(ImGuiCol_Text, warning_color);
            ImGui::TextUnformatted(line.text.c_str());
          } else {
            ImGui::TextUnformatted(line.text.c_str());
          }

          if (do_scroll && i == num_lines - 1) {
            ImGui::SetScrollHereY(1.0f);
          }
        }
      }
    }
    ImGui::EndTable();
    if (copy_to_clipboard.has_value()) {
      MaybeSaveToClipboard(*copy_to_clipboard);
    }
  }
}

void Console::StatusGui(bool* show_console) {
  const ConsoleCounts counts = GetCounts();
  char label[128];
  std::snprintf(label, sizeof(label),
                "Logs: %" PRId64 " | Warnings: %" PRId64 " | Errors: %" PRId64 "###ConsoleStatus",
                counts.messages, counts.warnings, counts.errors);

  ScopedStyle style;
  const float scale = ImGui::GetWindowDpiScale();
  style.Var(ImGuiStyleVar_FramePadding, ImVec2(8.0f * scale, 2.0f * scale));

  if (counts.errors > 0) {
    style.Color(ImGuiCol_Text, ConsoleTextColor(mjLOG_ERROR));
  } else if (counts.warnings > 0) {
    style.Color(ImGuiCol_Text, ConsoleTextColor(mjLOG_WARNING));
  }

  if (show_console && *show_console) {
    style.Color(ImGuiCol_Button, ImGuiCol_ButtonActive);
  }

  if (ImGui::Button(label)) {
    if (show_console) {
      *show_console = !*show_console;
    }
  }
  ImGui::SetItemTooltip(
      "%s", (show_console && *show_console) ? "Minimize Console to Status Bar" : "Open Console");
}

}  // namespace mujoco::studio
