// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_NETIMGUI_GOOGLE_LOGGING_H_
#define THIRD_PARTY_NETIMGUI_GOOGLE_LOGGING_H_

#include <time.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#if defined(__EMSCRIPTEN__)
#include <emscripten/console.h>
#endif

namespace NetImgui {

enum class LogSeverity { kInfo, kWarning, kError };

// Set to 1 or higher to enable VLOG messages at runtime.
constexpr int kLogVerbosity = 0;

__attribute__((format(printf, 4, 5))) inline void NetImGuiLog(
    LogSeverity severity, const char* file, int line, const char* fmt, ...) {
  char buf[1024];
  const char* basename = strrchr(file, '/');
  basename = basename ? basename + 1 : file;

  auto now_tp = std::chrono::system_clock::now();
  time_t now = std::chrono::system_clock::to_time_t(now_tp);
  auto now_usec = std::chrono::duration_cast<std::chrono::microseconds>(
                      now_tp.time_since_epoch())
                      .count() %
                  1000000;
  struct tm tm_info;
  localtime_r(&now, &tm_info);

  const char severity_char = severity == LogSeverity::kError     ? 'E'
                             : severity == LogSeverity::kWarning ? 'W'
                                                                 : 'I';

  int prefix_len = snprintf(
      buf, sizeof(buf), "%c%02d%02d %02d:%02d:%02d.%06d %s:%d] ", severity_char,
      tm_info.tm_mon + 1, tm_info.tm_mday, tm_info.tm_hour, tm_info.tm_min,
      tm_info.tm_sec, static_cast<int>(now_usec), basename, line);
  if (prefix_len < 0 || prefix_len >= static_cast<int>(sizeof(buf))) {
    prefix_len = 0;  // overwrite prefix on error or truncation
  }

  va_list args;
  va_start(args, fmt);
  vsnprintf(buf + prefix_len, sizeof(buf) - prefix_len, fmt, args);
  va_end(args);

  if (severity == LogSeverity::kInfo) {
#if defined(__EMSCRIPTEN__)
    emscripten_out(buf);
#else
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
#endif
  } else {
#if defined(__EMSCRIPTEN__)
    emscripten_err(buf);
#else
    fputs(buf, stderr);
    fputc('\n', stderr);
    fflush(stderr);
#endif
  }
}

}  // namespace NetImgui

#define LOG(severity, fmt, ...)                                           \
  ::NetImgui::NetImGuiLog(::NetImgui::LogSeverity::k##severity, __FILE__, \
                          __LINE__, fmt, ##__VA_ARGS__)

#define VLOG(level, fmt, ...)                                           \
  do {                                                                  \
    if (::NetImgui::kLogVerbosity >= (level))                           \
      ::NetImgui::NetImGuiLog(::NetImgui::LogSeverity::kInfo, __FILE__, \
                              __LINE__, fmt, ##__VA_ARGS__);            \
  } while (0)

#endif  // THIRD_PARTY_NETIMGUI_GOOGLE_LOGGING_H_
