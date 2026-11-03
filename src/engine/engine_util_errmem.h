// Copyright 2021 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_ERRMEM_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_ERRMEM_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <mujoco/mjexport.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef mjPRINTFLIKE
  #if defined(__GNUC__)
    #define mjPRINTFLIKE(n, m) __attribute__((format(printf, n, m)))
  #else
    #define mjPRINTFLIKE(n, m)
  #endif  // __GNUC__
#endif  // mjPRINTFLIKE


//------------------------------ file operations ---------------------------------------------------

// open file with UTF-8 filename support (uses _wfopen on Windows)
MJAPI FILE* mju_fopen(const char* filename, const char* mode);

// stat file with UTF-8 filename support (uses _wstat on Windows)
MJAPI int mju_stat(const char* filename, struct stat* file_stat);


//------------------------------ malloc and free ---------------------------------------------------

// allocate memory; byte-align on 8; pad size to multiple of 8
MJAPI void* mju_malloc(size_t size);

// free memory with free() by default
MJAPI void mju_free(void* ptr);

// user memory handlers
MJAPI extern void* (*mju_user_malloc)(size_t);
MJAPI extern void (*mju_user_free)(void*);


//------------------------------ logging configuration and handlers --------------------------------

// set the active log handler, return the previous handler
// if handler is NULL, restore the default handler
MJAPI mjfLogHandler mju_setLogHandler(mjfLogHandler handler);

// set/get default handler configuration
MJAPI mjLogConfig mju_getLogConfig(void);
MJAPI void mju_setLogConfig(mjLogConfig config);

// clear user handlers; restore default processing
MJAPI void mju_clearHandlers(void);

// legacy error/warning handlers (deprecated: prefer mju_setLogHandler)
MJAPI extern void (*mju_user_error)(const char*);
MJAPI extern void (*mju_user_warning)(const char*);


//------------------------------ public message logging --------------------------------------------

// log a fatal error message, write to logfile and console, pause and exit
MJAPI void mju_error(const char* msg, ...) mjPRINTFLIKE(1, 2);
MJAPI void mju_error_v(const char* msg, va_list args);

// log a warning message, write to logfile and console
MJAPI void mju_warning(const char* msg, ...) mjPRINTFLIKE(1, 2);

// log an info message with optional topic filtering
MJAPI void mju_info(int topic, const char* msg, ...) mjPRINTFLIKE(2, 3);

// dispatch a structured log message to the active handler
MJAPI void mju_message(const mjLogMessage* msg);

// (deprecated) write [datetime, type: message] to MUJOCO_LOG.TXT
MJAPI void mju_writeLog(const char* type, const char* msg);


//------------------------------ internal helpers and macros ---------------------------------------

// set thread-local log handler; return previous thread-local handler
MJAPI mjfLogHandler _mjPRIVATE_setTlsLogHandler(mjfLogHandler handler);

// get the currently active global log handler (read-only, no modification)
MJAPI mjfLogHandler _mjPRIVATE_getGlobalLogHandler(void);

// check whether an info topic is enabled
MJAPI mjtBool mju_isTopicEnabled(int topic);

// strip directory from __FILE__ (cross-platform)
static inline const char* BaseName(const char* path) {
  const char* slash = strrchr(path, '/');
  const char* bslash = strrchr(path, '\\');
  if (slash && bslash) return (slash > bslash ? slash : bslash) + 1;
  if (slash) return slash + 1;
  if (bslash) return bslash + 1;
  return path;
}

// internal macro to emit a structured error with source location
#define mjERROR(...)                                           \
  {                                                            \
    mjLogMessage _msg = {.level = mjLOG_ERROR,                 \
                         .func = __func__,                     \
                         .file = __FILE__,                     \
                         .line = __LINE__};                    \
    snprintf(_msg.subject, sizeof(_msg.subject), __VA_ARGS__); \
    mju_message(&_msg);                                        \
  }

// internal macro to emit a structured debug trace with fast producer-side topic filtering
#ifndef MJ_DISABLE_DEBUG_TRACING
#define mjDEBUG(_topic, ...)                                   \
  if (mju_isTopicEnabled(_topic)) {                            \
    mjLogMessage _msg = {.level = mjLOG_DEBUG,                 \
                         .topic = _topic,                      \
                         .func = __func__};                    \
    snprintf(_msg.subject, sizeof(_msg.subject), __VA_ARGS__); \
    mju_message(&_msg);                                        \
  }
#else
#define mjDEBUG(_topic, ...) ((void)0)
#endif

#ifdef __cplusplus
}
#endif
#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_ERRMEM_H_
