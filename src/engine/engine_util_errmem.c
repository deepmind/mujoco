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

#include "engine/engine_util_errmem.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(__EMSCRIPTEN__)
#include <sched.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <unistd.h>
#endif

#include <mujoco/mjexport.h>
#include "engine/engine_crossplatform.h"  // IWYU pragma: keep

//------------------------- cross-platform aligned malloc/free -------------------------------------

// forward declaration for active log handler dispatch
static void mju_defaultLogHandler(const mjLogMessage* msg);


//------------------------- UTF-8 file operations --------------------------------------------------

#ifdef _WIN32
// convert UTF-8 string to newly allocated wide string; caller must free
static wchar_t* utf8ToWide(const char* str) {
  if (!str) {
    return NULL;
  }
  int wlen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, -1, NULL, 0);
  if (wlen == 0) {
    return NULL;
  }
  wchar_t* wstr = (wchar_t*)malloc(wlen * sizeof(wchar_t));
  if (!wstr) {
    return NULL;
  }
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, -1, wstr, wlen);
  return wstr;
}
#endif


// open file with UTF-8 filename support
FILE* mju_fopen(const char* filename, const char* mode) {
#ifdef _WIN32
  wchar_t* wfilename = utf8ToWide(filename);
  wchar_t* wmode = utf8ToWide(mode);
  FILE* fp = (wfilename && wmode) ? _wfopen(wfilename, wmode) : NULL;
  free(wfilename);
  free(wmode);
  return fp;
#else
  return fopen(filename, mode);
#endif
}


// stat file with UTF-8 filename support
int mju_stat(const char* filename, struct stat* file_stat) {
#if defined(_WIN32) && !defined(__MINGW32__)
  wchar_t* wfilename = utf8ToWide(filename);
  if (!wfilename) {
    return -1;
  }
  struct _stat* wfile_stat = (struct _stat*)file_stat;
  int result = _wstat(wfilename, wfile_stat);
  free(wfilename);
  return result;
#else
  return stat(filename, file_stat);
#endif
}


//------------------------------ malloc and free ---------------------------------------------------

// user memory handlers
MJAPI void* (*mju_user_malloc) (size_t) = 0;
MJAPI void (*mju_user_free) (void*) = 0;

// cross-platform aligned malloc
static inline void* mju_alignedMalloc(size_t size, size_t align) {
#ifdef _WIN32
  return _aligned_malloc(size, align);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  return aligned_alloc(align, size);
#endif
}

// cross-platform aligned free
static inline void mju_alignedFree(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

// allocate memory; byte-align on 64; pad size to multiple of 64
void* mju_malloc(size_t size) {
  void* ptr = 0;

  if (mju_user_malloc) {
    ptr = mju_user_malloc(size);
  } else {
    if (size > 0 && (size % 64)) {
      size += 64 - (size % 64);
    }
    if (size > 0) {
      ptr = mju_alignedMalloc(size, 64);
    }
  }

  if (!ptr && size > 0) {
    mju_error("Could not allocate memory");
  }
  return ptr;
}

// free memory
void mju_free(void* ptr) {
  if (!ptr) return;
  if (mju_user_free) {
    mju_user_free(ptr);
  } else {
    mju_alignedFree(ptr);
  }
}


//------------------------------ logging configuration and handlers --------------------------------

// global log handler
static mjfLogHandler global_log_handler = mju_defaultLogHandler;

// legacy error/warning handlers (deprecated)
MJAPI void (*mju_user_error) (const char*) = 0;
MJAPI void (*mju_user_warning) (const char*) = 0;

// default handler configuration
static mjLogConfig log_config = {.logto_console = true,
                                 .logto_file = true,
                                 .logfile = "MUJOCO_LOG.TXT",
                                 .topics = 0};
static mjtBool env_init_claimed = 0;  // claimed by first thread to enter init
static mjtBool env_init_done = 0;     // set after log_config init completes

// parse MUJOCO_LOG_TOPICS env var to seed initial topic bitmask
// example: MUJOCO_LOG_TOPICS="time_stp,sleep"
static void mju_initLogTopicsFromEnv(void) {
  const char* env = getenv("MUJOCO_LOG_TOPICS");
  if (!env) return;

  // mjTOPIC_X enum names with the mjTOPIC_ prefix stripped and lowercased; keep in sync with mjtLogTopic
  static const char* topic_names[mjNTOPIC] = {"time_stp", "time_cmp", "sleep"};
  char buf[256];
  strncpy(buf, env, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  char* token = buf;
  while (*token) {
    while (*token == ' ' || *token == ',') token++;
    if (!*token) break;

    char* end = token;
    while (*end && *end != ',') end++;
    bool trailing_comma = (*end == ',');

    char* tEnd = end - 1;
    while (tEnd > token && *tEnd == ' ') tEnd--;
    *(tEnd + 1) = '\0';

    for (int i = 0; i < mjNTOPIC; i++) {
      if (strcasecmp(token, topic_names[i]) == 0) {
        log_config.topics |= (1 << i);
        break;
      }
    }
    token = trailing_comma ? end + 1 : end;
  }
}

// private pointer getter encapsulates lazy init with zero copy overhead
static const mjLogConfig* mju_getLogConfigPtr(void) {
  if (!mj_atomic_load_bool(&env_init_done)) {
    // atomically claim right to initialize; only one thread gets old value 0
    if (!mj_atomic_exchange_bool(&env_init_claimed, 1)) {
      mju_initLogTopicsFromEnv();
      mj_atomic_store_bool(&env_init_done, 1);
    } else {
      // another thread is initializing; spin until it completes
      while (!mj_atomic_load_bool(&env_init_done)) {
#if defined(_WIN32)
        Sleep(0);
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(__EMSCRIPTEN__)
        sched_yield();
#endif
      }
    }
  }
  return &log_config;
}

// check whether an info topic is enabled
mjtBool mju_isTopicEnabled(int topic) {
  if (!topic) return 1;
  const mjLogConfig* cfg = mju_getLogConfigPtr();
  return ((cfg->topics & (1 << (topic - 1))) != 0);
}

// set the active log handler; return the previous handler
mjfLogHandler mju_setLogHandler(mjfLogHandler handler) {
  mjfLogHandler prev = global_log_handler;
  global_log_handler = handler ? handler : mju_defaultLogHandler;
  return prev;
}

// get default handler configuration
mjLogConfig mju_getLogConfig(void) {
  return *mju_getLogConfigPtr();
}

// set default handler configuration
void mju_setLogConfig(mjLogConfig config) {
  log_config = config;
  mj_atomic_store_bool(&env_init_claimed, 1);
  mj_atomic_store_bool(&env_init_done, 1);
}

// restore default processing
void mju_clearHandlers(void) {
  global_log_handler = mju_defaultLogHandler;
  log_config = (mjLogConfig){.logto_console = true,
                             .logto_file = true,
                             .logfile = "MUJOCO_LOG.TXT",
                             .topics = 0};
  mj_atomic_store_bool(&env_init_claimed, 1);
  mju_initLogTopicsFromEnv();
  mj_atomic_store_bool(&env_init_done, 1);

  mju_user_error = 0;
  mju_user_warning = 0;
  mju_user_malloc = 0;
  mju_user_free = 0;
}

// fill buffer with formatted local time string (thread-safe)
static void mju_localTimeStr(char* buf, int buf_sz) {
  time_t rawtime;
  struct tm timeinfo;
  time(&rawtime);

#if defined(_POSIX_C_SOURCE) || defined(__APPLE__) || defined(__STDC_VERSION_TIME_H__) || defined(__EMSCRIPTEN__) || defined(__linux__)
  localtime_r(&rawtime, &timeinfo);
#elif defined(_WIN32)
  localtime_s(&timeinfo, &rawtime);
#elif __STDC_LIB_EXT1__
  localtime_s(&rawtime, &timeinfo);
#else
  #error "Thread-safe version of `localtime` is not present in the standard C library"
#endif

  strftime(buf, buf_sz, "%c", &timeinfo);
}

// write formatted message to stream
static void mju_fprint_message(FILE* stream, const char* timestr,
                               const mjLogMessage* msg) {
  const char* type = msg->level == mjLOG_ERROR   ? "ERROR" :
                     msg->level == mjLOG_WARNING ? "WARNING" :
                     msg->level == mjLOG_INFO    ? "INFO" : "DEBUG";
  fprintf(stream, "%s", type);
  if (msg->func) fprintf(stream, " %s", msg->func);
  if (msg->file && msg->line) fprintf(stream, " (%s:%d)", BaseName(msg->file), msg->line);
  if (timestr[0]) fprintf(stream, " %s", timestr);
  fprintf(stream, ": %s\n", msg->subject);
  if (msg->body) fprintf(stream, "%s\n", msg->body);

  // add blank line after message except for DEBUG, for compactness
  if (msg->level != mjLOG_DEBUG) fprintf(stream, "\n");
}

// format legacy adapter string: "func: subject" or just "subject"
static const char* mju_legacy_text(const mjLogMessage* msg, char* buf, int bufsz) {
  if (msg->func) {
    snprintf(buf, bufsz, "%s: %s", msg->func, msg->subject);
    return buf;
  }
  return msg->subject;
}

// default log handler: topic filtering, console/file output, legacy compat, exit on error
static void mju_defaultLogHandler(const mjLogMessage* msg) {
  const mjLogConfig* cfg = mju_getLogConfigPtr();

  if ((msg->level == mjLOG_INFO || msg->level == mjLOG_DEBUG) && !mju_isTopicEnabled(msg->topic)) {
    return;
  }

  if (msg->level == mjLOG_ERROR && mju_user_error) {
    char buf[2048];
    mju_user_error(mju_legacy_text(msg, buf, sizeof(buf)));
    return;
  }

  if (msg->level == mjLOG_WARNING && mju_user_warning) {
    char buf[2048];
    mju_user_warning(mju_legacy_text(msg, buf, sizeof(buf)));
    return;
  }

  char timestr[64] = "";
  if (msg->timestamp || (cfg->logto_file && cfg->logfile[0])) {
    mju_localTimeStr(timestr, sizeof(timestr));
  }

  if (cfg->logto_file && cfg->logfile[0]) {
    FILE* fp = mju_fopen(cfg->logfile, "a+t");
    if (fp) {
      mju_fprint_message(fp, timestr, msg);
      fclose(fp);
    }
  }

  if (cfg->logto_console) {
    FILE* stream = (msg->level >= mjLOG_WARNING) ? stderr : stdout;
    mju_fprint_message(stream, msg->timestamp ? timestr : "", msg);
  }

  if (msg->level == mjLOG_ERROR) {
    exit(EXIT_FAILURE);
  }
}


//------------------------------ public message logging --------------------------------------------

// thread-local log handler override
static mjTHREADLOCAL mjfLogHandler _mjPRIVATE_tls_log_handler = NULL;

// recursion guard for log handler
static mjTHREADLOCAL bool in_log = false;

// dispatch to active handler (TLS > global)
static inline mjfLogHandler mju_activeHandler(void) {
  return _mjPRIVATE_tls_log_handler ? _mjPRIVATE_tls_log_handler : global_log_handler;
}

// dispatch a structured log message to the active handler
void mju_message(const mjLogMessage* msg) {
  // recursion guard: silently drop messages dispatched from within a handler
  if (in_log) return;

  // error handlers are expected to not return (longjmp or exit); we cannot set in_log
  // around the call because longjmp would leave it permanently true on this thread
  if (msg->level != mjLOG_ERROR) {
    in_log = true;
    mju_activeHandler()(msg);
    in_log = false;
  } else {
    mju_activeHandler()(msg);
  }
}

void mju_error_v(const char* msg, va_list args) {
  mjLogMessage m = {.level = mjLOG_ERROR};
  vsnprintf(m.subject, sizeof(m.subject), msg, args);
  mju_message(&m);
}

// write message to logfile and console, pause and exit
void mju_error(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  mju_error_v(msg, args);
  va_end(args);
}

// write message to logfile and console
void mju_warning(const char* msg, ...) {
  mjLogMessage m = {.level = mjLOG_WARNING};
  va_list args;
  va_start(args, msg);
  vsnprintf(m.subject, sizeof(m.subject), msg, args);
  va_end(args);
  mju_message(&m);
}

// log an info message
void mju_info(int topic, const char* msg, ...) {
  mjLogMessage m = {.level = mjLOG_INFO, .topic = topic};
  va_list args;
  va_start(args, msg);
  vsnprintf(m.subject, sizeof(m.subject), msg, args);
  va_end(args);
  mju_message(&m);
}

// (deprecated) write datetime, type: message to MUJOCO_LOG.TXT
void mju_writeLog(const char* type, const char* msg) {
  char timestr[64];
  mju_localTimeStr(timestr, sizeof(timestr));

  FILE* fp = mju_fopen("MUJOCO_LOG.TXT", "a+t");
  if (fp) {
    fprintf(fp, "%s\n%s: %s\n\n", timestr, type, msg);
    fclose(fp);
  }
}


//------------------------------ internal helpers --------------------------------------------------

// set thread-local log handler; return previous
mjfLogHandler _mjPRIVATE_setTlsLogHandler(mjfLogHandler handler) {
  mjfLogHandler prev = _mjPRIVATE_tls_log_handler;
  _mjPRIVATE_tls_log_handler = handler;
  return prev;
}


// get the currently active global log handler (read-only, no modification)
mjfLogHandler _mjPRIVATE_getGlobalLogHandler(void) {
  return global_log_handler;
}
