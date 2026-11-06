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

#ifndef MUJOCO_PYTHON_STUDIO_WEB_LOGGING_H_
#define MUJOCO_PYTHON_STUDIO_WEB_LOGGING_H_

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>

#if defined(__EMSCRIPTEN__)
#include <emscripten/console.h>
#endif

namespace mujoco::studio {

enum class LogSeverity { kInfo, kWarning, kError };

// Set to 1 or higher to enable VLOG messages at runtime.
inline constexpr int kLogVerbosity = 0;

#if defined(__GNUC__) || defined(__clang__)
#define MUJOCO_STUDIO_WEB_PRINTF_ATTR(fmt_idx, arg_idx) \
  [[gnu::format(printf, fmt_idx, arg_idx)]]
#else
#define MUJOCO_STUDIO_WEB_PRINTF_ATTR(fmt_idx, arg_idx)
#endif

MUJOCO_STUDIO_WEB_PRINTF_ATTR(4, 5)
inline void StudioWebLog(LogSeverity severity, const char* file, int line,
                         const char* fmt, ...) {
  char buf[1024];
  const char* slash = std::strrchr(file, '/');
  const char* backslash = std::strrchr(file, '\\');
  const char* sep = (slash && backslash)
                        ? (slash > backslash ? slash : backslash)
                        : (slash ? slash : backslash);
  const char* basename = sep ? sep + 1 : file;

  std::timespec ts = {};
  std::timespec_get(&ts, TIME_UTC);
  time_t now = ts.tv_sec;
  int now_usec = static_cast<int>(ts.tv_nsec / 1000);
  struct tm tm_info = {};
#if defined(_WIN32)
  localtime_s(&tm_info, &now);
#else
  localtime_r(&now, &tm_info);
#endif

  const char severity_char = severity == LogSeverity::kError     ? 'E'
                             : severity == LogSeverity::kWarning ? 'W'
                                                                 : 'I';

  int prefix_len = std::snprintf(
      buf, sizeof(buf), "%c%02d%02d %02d:%02d:%02d.%06d %s:%d] ", severity_char,
      tm_info.tm_mon + 1, tm_info.tm_mday, tm_info.tm_hour, tm_info.tm_min,
      tm_info.tm_sec, now_usec, basename, line);
  if (prefix_len < 0 || prefix_len >= static_cast<int>(sizeof(buf))) {
    prefix_len = 0;  // overwrite prefix on error or truncation
  }

  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf + prefix_len, sizeof(buf) - prefix_len, fmt, args);
  va_end(args);

  if (severity == LogSeverity::kInfo) {
#if defined(__EMSCRIPTEN__)
    emscripten_out(buf);
#else
    std::fputs(buf, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
#endif
  } else {
#if defined(__EMSCRIPTEN__)
    emscripten_err(buf);
#else
    std::fputs(buf, stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
#endif
  }
}

#undef MUJOCO_STUDIO_WEB_PRINTF_ATTR

}  // namespace mujoco::studio

#define LOG(severity, fmt, ...)                                         \
  ::mujoco::studio::StudioWebLog(::mujoco::studio::LogSeverity::k##severity, \
                                 __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define VLOG(level, fmt, ...)                                               \
  do {                                                                      \
    if (::mujoco::studio::kLogVerbosity >= (level)) {                       \
      ::mujoco::studio::StudioWebLog(::mujoco::studio::LogSeverity::kInfo,  \
                                     __FILE__, __LINE__, fmt,               \
                                     ##__VA_ARGS__);                        \
    }                                                                       \
  } while (0)

#endif  // MUJOCO_PYTHON_STUDIO_WEB_LOGGING_H_
