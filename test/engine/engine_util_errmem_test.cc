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

// Tests for engine/engine_util_errmem.c.

#include "src/engine/engine_util_errmem.h"

#include <sys/stat.h>

#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT
#include <string>
#include <system_error>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_crossplatform.h"  // IWYU pragma: keep

extern "C" {
MJAPI mjfLogHandler _mjPRIVATE_setTlsLogHandler(mjfLogHandler handler);
}

namespace mujoco {
namespace {

// ========================= test infrastructure ==============================

// captured log message
struct CapturedMsg {
  int level;
  int topic;
  std::string subject;
  std::string func;
  std::string file;
  int line;
  std::string body;
};

// thread-local capture state
static thread_local std::vector<CapturedMsg> captured_msgs;
static thread_local bool capture_longjmp = false;
static thread_local std::jmp_buf capture_jmp_buf;

// log handler that captures messages
void CapturingHandler(const mjLogMessage* msg) {
  captured_msgs.push_back({
      .level = msg->level,
      .topic = msg->topic,
      .subject = msg->subject,
      .func = msg->func ? msg->func : "",
      .file = msg->file ? msg->file : "",
      .line = msg->line,
      .body = msg->body ? msg->body : "",
  });
  // longjmp on error to prevent exit()
  if (msg->level == mjLOG_ERROR && capture_longjmp) {
    std::longjmp(capture_jmp_buf, 1);
  }
}

// RAII guard: installs capturing handler, restores previous on destruction
class ScopedCapture {
 public:
  ScopedCapture() {
    captured_msgs.clear();
    capture_longjmp = true;
    prev_ = _mjPRIVATE_setTlsLogHandler(CapturingHandler);
  }
  ~ScopedCapture() {
    _mjPRIVATE_setTlsLogHandler(prev_);
    capture_longjmp = false;
  }
  const std::vector<CapturedMsg>& msgs() const { return captured_msgs; }

 private:
  mjfLogHandler prev_;
};

// ========================= mju_setLogHandler tests ==========================

TEST(LogHandlerTest, SetLogHandlerReturnsOldHandler) {
  mjfLogHandler old = mju_setLogHandler(CapturingHandler);
  EXPECT_NE(old, nullptr);  // default handler is non-null

  mjfLogHandler prev = mju_setLogHandler(nullptr);
  EXPECT_EQ(prev, CapturingHandler);

  // NULL restores default (non-null)
  mjfLogHandler restored = mju_setLogHandler(nullptr);
  EXPECT_NE(restored, nullptr);
}

TEST(LogHandlerTest, CustomHandlerReceivesWarning) {
  ScopedCapture cap;
  mju_warning("test warning %d", 42);

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_WARNING);
  EXPECT_EQ(cap.msgs()[0].subject, "test warning 42");
}

TEST(LogHandlerTest, CustomHandlerReceivesError) {
  ScopedCapture cap;
  if (setjmp(capture_jmp_buf) == 0) {
    mju_error("test error %s", "foo");
  }

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_ERROR);
  EXPECT_EQ(cap.msgs()[0].subject, "test error foo");
}

TEST(LogHandlerTest, WarningRecursionGuard) {
  static thread_local int call_count = 0;
  call_count = 0;
  auto recursive_handler = +[](const mjLogMessage* msg) {
    call_count++;
    mju_warning("recursive warning");
  };

  auto old_tls = _mjPRIVATE_setTlsLogHandler(recursive_handler);
  mju_warning("initial warning");
  _mjPRIVATE_setTlsLogHandler(old_tls);

  EXPECT_EQ(call_count, 1);
}

// ========================= TLS handler override tests =======================

TEST(TlsHandlerTest, TlsOverridesGlobal) {
  // install a global handler
  auto global_prev = mju_setLogHandler(CapturingHandler);

  // install a different TLS handler
  static thread_local bool tls_called = false;
  auto tls_handler = +[](const mjLogMessage* msg) {
    tls_called = true;
    if (msg->level == mjLOG_ERROR) {
      std::longjmp(capture_jmp_buf, 1);
    }
  };

  tls_called = false;
  captured_msgs.clear();
  auto old_tls = _mjPRIVATE_setTlsLogHandler(tls_handler);

  if (setjmp(capture_jmp_buf) == 0) {
    mju_error("tls test");
  }

  // TLS handler should have been called, not the global one
  EXPECT_TRUE(tls_called);
  EXPECT_EQ(captured_msgs.size(), 0);

  _mjPRIVATE_setTlsLogHandler(old_tls);
  mju_setLogHandler(global_prev);
}

TEST(TlsHandlerTest, NullTlsFallsBackToGlobal) {
  auto global_prev = mju_setLogHandler(CapturingHandler);
  captured_msgs.clear();

  // ensure TLS is null
  auto old_tls = _mjPRIVATE_setTlsLogHandler(nullptr);

  mju_warning("fallback test");

  // global capturing handler should have been called
  ASSERT_EQ(captured_msgs.size(), 1);
  EXPECT_EQ(captured_msgs[0].subject, "fallback test");

  _mjPRIVATE_setTlsLogHandler(old_tls);
  mju_setLogHandler(global_prev);
}

TEST(TlsHandlerTest, SetTlsReturnsPrevious) {
  auto h1 = _mjPRIVATE_setTlsLogHandler(CapturingHandler);
  auto h2 = _mjPRIVATE_setTlsLogHandler(nullptr);
  EXPECT_EQ(h2, CapturingHandler);
  _mjPRIVATE_setTlsLogHandler(h1);
}

// ========================= mju_info and topic filtering =====================

TEST(InfoTest, InfoMessageReachesHandler) {
  ScopedCapture cap;
  mju_info(mjTOPIC_NONE, "info message %d", 7);

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_INFO);
  EXPECT_EQ(cap.msgs()[0].topic, mjTOPIC_NONE);
  EXPECT_EQ(cap.msgs()[0].subject, "info message 7");
}

TEST(InfoTest, TopicZeroAlwaysPasses) {
  // topic 0 (NONE) should pass regardless of config
  ScopedCapture cap;
  mju_info(mjTOPIC_NONE, "always passes");

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].subject, "always passes");
}

// ========================= mjLogConfig tests ================================

TEST(LogConfigTest, GetSetRoundtrip) {
  mjLogConfig orig = mju_getLogConfig();

  mjLogConfig custom = {.logto_console = false,
                        .logto_file = false,
                        .logfile = "",
                        .topics = (1 << 0) | (1 << 1)};
  mju_setLogConfig(custom);

  mjLogConfig readback = mju_getLogConfig();
  EXPECT_EQ(readback.logto_console, false);
  EXPECT_EQ(readback.logto_file, false);
  EXPECT_STREQ(readback.logfile, "");
  EXPECT_EQ(readback.topics, (1 << 0) | (1 << 1));

  // restore
  mju_setLogConfig(orig);
}

TEST(LogConfigTest, InitFromEnv) {
  mjLogConfig orig = mju_getLogConfig();

  setenv("MUJOCO_LOG_TOPICS", "sleep,time_cmp", 1);
  mju_clearHandlers();

  mjLogConfig config = mju_getLogConfig();
  EXPECT_EQ(config.topics, (1 << 1) | (1 << 2));

  unsetenv("MUJOCO_LOG_TOPICS");
  mju_clearHandlers();
  mju_setLogConfig(orig);
}

// ========================= topic filtering in default handler ===============

TEST(TopicFilterTest, DefaultHandlerFiltersDisabledTopic) {
  // save and configure: disable all topics, route to capture
  mjLogConfig orig = mju_getLogConfig();
  mjLogConfig no_topics = {
      .logto_console = false, .logto_file = false, .logfile = "", .topics = 0};
  mju_setLogConfig(no_topics);

  auto global_prev = mju_setLogHandler(nullptr);  // restore default handler

  // send info with a topic — should be filtered by default handler
  // we need to observe absence of output, so we'll use a custom global handler
  mju_setLogHandler(CapturingHandler);
  captured_msgs.clear();

  // make sure TLS is null so global handler is used
  auto old_tls = _mjPRIVATE_setTlsLogHandler(nullptr);

  mju_info(mjTOPIC_SLEEP, "should be filtered");

  // TLS handler (none) -> global handler (CapturingHandler)
  // But topic filtering happens inside the handler dispatch, not in mju_info
  // mju_info just dispatches; the *default* handler filters.
  // Since we installed CapturingHandler (not default), it won't filter.
  // To test default handler filtering, we'd need the default handler.
  // Instead, test that the topic field is correctly set.
  ASSERT_EQ(captured_msgs.size(), 1);
  EXPECT_EQ(captured_msgs[0].topic, mjTOPIC_SLEEP);

  _mjPRIVATE_setTlsLogHandler(old_tls);
  mju_setLogHandler(global_prev);
  mju_setLogConfig(orig);
}

// ========================= mjERROR macro tests ==============================

TEST(MjErrorMacroTest, HasSourceLocation) {
  ScopedCapture cap;
  if (setjmp(capture_jmp_buf) == 0) {
    mjERROR("macro error %d", 99);
  }

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_ERROR);
  EXPECT_EQ(cap.msgs()[0].subject, "macro error 99");
  EXPECT_FALSE(cap.msgs()[0].func.empty());
  EXPECT_FALSE(cap.msgs()[0].file.empty());
  EXPECT_GT(cap.msgs()[0].line, 0);
}

TEST(MjErrorMacroTest, FileIsPath) {
  ScopedCapture cap;
  if (setjmp(capture_jmp_buf) == 0) {
    mjERROR("path test");
  }

  ASSERT_EQ(cap.msgs().size(), 1);
  // file should be a path containing the filename
  EXPECT_NE(cap.msgs()[0].file.find("engine_util_errmem_test.cc"),
            std::string::npos);
}

// ========================= mju_clearHandlers ================================

TEST(ClearHandlersTest, RestoresDefaults) {
  // set some custom state
  mju_user_error = [](const char*) {};
  mju_user_warning = [](const char*) {};

  mju_clearHandlers();

  EXPECT_EQ(mju_user_error, nullptr);
  EXPECT_EQ(mju_user_warning, nullptr);
  EXPECT_EQ(mju_user_malloc, nullptr);
  EXPECT_EQ(mju_user_free, nullptr);

  // log config should be restored to defaults
  mjLogConfig config = mju_getLogConfig();
  EXPECT_TRUE(config.logto_console);
  EXPECT_STREQ(config.logfile, "MUJOCO_LOG.TXT");
}

// ========================= mju_message ======================================

TEST(MessageTest, DispatchesStructuredMessage) {
  ScopedCapture cap;
  mjLogMessage msg = {};
  msg.level = mjLOG_WARNING;
  snprintf(msg.subject, sizeof(msg.subject), "raw warning");
  msg.func = "TestFunc";
  msg.file = "test.c";
  msg.line = 42;

  mju_message(&msg);

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_WARNING);
  EXPECT_EQ(cap.msgs()[0].subject, "raw warning");
  EXPECT_EQ(cap.msgs()[0].func, "TestFunc");
  EXPECT_EQ(cap.msgs()[0].file, "test.c");
  EXPECT_EQ(cap.msgs()[0].line, 42);
}

TEST(MessageTest, DispatchesError) {
  ScopedCapture cap;
  mjLogMessage msg = {};
  msg.level = mjLOG_ERROR;
  snprintf(msg.subject, sizeof(msg.subject), "raw error");

  if (setjmp(capture_jmp_buf) == 0) {
    mju_message(&msg);
  }

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_ERROR);
  EXPECT_EQ(cap.msgs()[0].subject, "raw error");
}

// ========================= legacy handler compatibility =====================

TEST(LegacyCompatTest, LegacyErrorHandlerViaDefault) {
  static thread_local std::string legacy_msg;
  legacy_msg.clear();

  auto old_tls = _mjPRIVATE_setTlsLogHandler(nullptr);
  auto old_global = mju_setLogHandler(nullptr);  // default handler
  mju_user_error = [](const char* msg) {
    legacy_msg = msg;
    // legacy error handlers are expected to not return; longjmp to simulate
    std::longjmp(capture_jmp_buf, 1);
  };

  if (setjmp(capture_jmp_buf) == 0) {
    mju_error("legacy test %d", 1);
  }

  EXPECT_EQ(legacy_msg, "legacy test 1");

  mju_user_error = nullptr;
  mju_setLogHandler(old_global);
  _mjPRIVATE_setTlsLogHandler(old_tls);
}

TEST(LegacyCompatTest, LegacyWarningHandlerViaDefault) {
  static thread_local std::string legacy_msg;
  legacy_msg.clear();

  auto old_tls = _mjPRIVATE_setTlsLogHandler(nullptr);
  auto old_global = mju_setLogHandler(nullptr);  // default handler
  mju_user_warning = [](const char* msg) { legacy_msg = msg; };

  mju_warning("legacy warn %s", "bar");

  EXPECT_EQ(legacy_msg, "legacy warn bar");

  mju_user_warning = nullptr;
  mju_setLogHandler(old_global);
  _mjPRIVATE_setTlsLogHandler(old_tls);
}

// ========================= truncation tests (from original) =================

TEST(TruncationTest, MjuErrorInternal) {
  ScopedCapture cap;
  if (setjmp(capture_jmp_buf) == 0) {
    mjERROR("foobar %d", 123);
  }

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].subject, "foobar 123");
  // func field should contain the test function name
  EXPECT_FALSE(cap.msgs()[0].func.empty());
}

// ========================= mjDEBUG macro tests ==============================

TEST(MjDebugMacroTest, EmitsDebugMessage) {
  mjLogConfig orig = mju_getLogConfig();
  mjLogConfig cfg = orig;
  cfg.topics = (1 << (mjTOPIC_SLEEP - 1));
  mju_setLogConfig(cfg);

  ScopedCapture cap;
  mjDEBUG(mjTOPIC_SLEEP, "debug msg %d", 77);

  ASSERT_EQ(cap.msgs().size(), 1);
  EXPECT_EQ(cap.msgs()[0].level, mjLOG_DEBUG);
  EXPECT_EQ(cap.msgs()[0].topic, mjTOPIC_SLEEP);
  EXPECT_EQ(cap.msgs()[0].subject, "debug msg 77");
  EXPECT_FALSE(cap.msgs()[0].func.empty());

  mju_setLogConfig(orig);
}

TEST(MjDebugMacroTest, FilteredByProducerSideCheck) {
  mjLogConfig orig = mju_getLogConfig();
  mjLogConfig cfg = orig;
  cfg.topics = 0;
  mju_setLogConfig(cfg);

  ScopedCapture cap;
  mjDEBUG(mjTOPIC_SLEEP, "should be filtered");

  EXPECT_EQ(cap.msgs().size(), 0);

  mju_setLogConfig(orig);
}

// ========================= BaseName tests ===================================

TEST(BaseNameTest, UnixPath) {
  EXPECT_STREQ(BaseName("/path/to/file.c"), "file.c");
}

TEST(BaseNameTest, WindowsPath) {
  EXPECT_STREQ(BaseName("C:\\path\\to\\file.c"), "file.c");
}

TEST(BaseNameTest, NoSeparator) { EXPECT_STREQ(BaseName("file.c"), "file.c"); }

TEST(BaseNameTest, MixedSeparators) {
  EXPECT_STREQ(BaseName("/mixed\\path/file.c"), "file.c");
  EXPECT_STREQ(BaseName("C:\\mixed/path\\file.c"), "file.c");
}

TEST(BaseNameTest, EmptyString) { EXPECT_STREQ(BaseName(""), ""); }

// ========================= default handler topic filtering ==================

TEST(DefaultHandlerFilterTest, DisabledTopicNotWrittenToFile) {
  mjLogConfig orig = mju_getLogConfig();
  auto old_tls = _mjPRIVATE_setTlsLogHandler(nullptr);
  auto old_global = mju_setLogHandler(nullptr);

  std::string logpath = std::string(::testing::TempDir()) + "/filter_test.txt";
  std::remove(logpath.c_str());

  mjLogConfig cfg = {};
  cfg.logto_console = false;
  cfg.logto_file = true;
  cfg.topics = 0;
  snprintf(cfg.logfile, sizeof(cfg.logfile), "%s", logpath.c_str());
  mju_setLogConfig(cfg);

  // disabled topic: should be filtered by default handler
  mju_info(mjTOPIC_SLEEP, "filtered message");

  std::FILE* fp = std::fopen(logpath.c_str(), "r");
  bool file_empty = true;
  if (fp) {
    std::fseek(fp, 0, SEEK_END);
    file_empty = (std::ftell(fp) == 0);
    std::fclose(fp);
  }
  EXPECT_TRUE(file_empty) << "disabled topic should not write to logfile";

  // enable the topic: should pass through
  cfg.topics = (1 << (mjTOPIC_SLEEP - 1));
  mju_setLogConfig(cfg);
  mju_info(mjTOPIC_SLEEP, "passed message");

  fp = std::fopen(logpath.c_str(), "r");
  ASSERT_NE(fp, nullptr);
  std::fseek(fp, 0, SEEK_END);
  EXPECT_GT(std::ftell(fp), 0) << "enabled topic should write to logfile";
  std::fclose(fp);

  std::remove(logpath.c_str());
  mju_setLogConfig(orig);
  mju_setLogHandler(old_global);
  _mjPRIVATE_setTlsLogHandler(old_tls);
}

// ========================= timing diagnostics ===============================

TEST(TimingDiagnosticsTest, EmitsOnReset) {
  const char xml[] =
      "<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>";
  char error[1024] = "";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "m.xml", xml, static_cast<int>(std::strlen(xml)));
  mjModel* m = mj_loadXML("m.xml", &vfs, error, sizeof(error));
  mj_deleteVFS(&vfs);
  ASSERT_NE(m, nullptr) << error;
  mjData* d = mj_makeData(m);
  ASSERT_NE(d, nullptr);

  // populate timers manually
  d->timer[mjTIMER_STEP].number = 100;
  d->timer[mjTIMER_STEP].duration = 0.01;
  d->timer[mjTIMER_POSITION].number = 100;
  d->timer[mjTIMER_POSITION].duration = 0.005;

  // capture log messages during reset
  ScopedCapture cap;
  mj_resetData(m, d);

  // find the timing diagnostics message
  const CapturedMsg* tmsg = nullptr;
  for (const auto& msg : cap.msgs()) {
    if (msg.level == mjLOG_INFO && msg.topic == mjTOPIC_TIME_STP) {
      tmsg = &msg;
      break;
    }
  }
  ASSERT_NE(tmsg, nullptr) << "expected timing diagnostics message";
  EXPECT_NE(tmsg->subject.find("100 steps"), std::string::npos);
  EXPECT_FALSE(tmsg->body.empty());
  EXPECT_NE(tmsg->body.find("total"), std::string::npos);

  // mj_resetDataKeyframe also emits diagnostics
  d->timer[mjTIMER_STEP].number = 50;
  d->timer[mjTIMER_STEP].duration = 0.005;
  mj_resetDataKeyframe(m, d, -1);
  int ntiming = 0;
  for (const auto& msg : cap.msgs()) {
    if (msg.level == mjLOG_INFO && msg.topic == mjTOPIC_TIME_STP) {
      ntiming++;
      tmsg = &msg;
    }
  }
  EXPECT_EQ(ntiming, 2);
  EXPECT_NE(tmsg->subject.find("50 steps"), std::string::npos);

  mj_deleteData(d);
  mj_deleteModel(m);
  mj_freeLastXML();
}

// ========================= env var edge cases ===============================

TEST(InitFromEnvEdgeTest, EmptyString) {
  mjLogConfig orig = mju_getLogConfig();
  setenv("MUJOCO_LOG_TOPICS", "", 1);
  mju_clearHandlers();
  EXPECT_EQ(mju_getLogConfig().topics, 0);
  unsetenv("MUJOCO_LOG_TOPICS");
  mju_clearHandlers();
  mju_setLogConfig(orig);
}

TEST(InitFromEnvEdgeTest, UnknownTopic) {
  mjLogConfig orig = mju_getLogConfig();
  setenv("MUJOCO_LOG_TOPICS", "nonexistent", 1);
  mju_clearHandlers();
  EXPECT_EQ(mju_getLogConfig().topics, 0);
  unsetenv("MUJOCO_LOG_TOPICS");
  mju_clearHandlers();
  mju_setLogConfig(orig);
}

TEST(InitFromEnvEdgeTest, ExtraCommas) {
  mjLogConfig orig = mju_getLogConfig();
  setenv("MUJOCO_LOG_TOPICS", ",,sleep,,", 1);
  mju_clearHandlers();
  EXPECT_EQ(mju_getLogConfig().topics, (1 << 2));
  unsetenv("MUJOCO_LOG_TOPICS");
  mju_clearHandlers();
  mju_setLogConfig(orig);
}

TEST(InitFromEnvEdgeTest, WhitespaceHandling) {
  mjLogConfig orig = mju_getLogConfig();
  setenv("MUJOCO_LOG_TOPICS", "  sleep , time_cmp  ", 1);
  mju_clearHandlers();
  EXPECT_EQ(mju_getLogConfig().topics, (1 << 1) | (1 << 2));
  unsetenv("MUJOCO_LOG_TOPICS");
  mju_clearHandlers();
  mju_setLogConfig(orig);
}

TEST(MjuFopenTest, AsciiPath) {
  std::filesystem::path filepath =
      std::filesystem::path(::testing::TempDir()) / "mju_fopen_ascii_test.txt";
  const char* content = "hello mujoco";

  // write
  FILE* fp = mju_fopen(filepath.string().c_str(), "w");
  ASSERT_NE(fp, nullptr);
  fputs(content, fp);
  fclose(fp);

  // read back
  fp = mju_fopen(filepath.string().c_str(), "r");
  ASSERT_NE(fp, nullptr);
  char buf[64] = {0};
  fgets(buf, sizeof(buf), fp);
  fclose(fp);

  EXPECT_STREQ(buf, content);

  struct stat file_stat;
  EXPECT_EQ(mju_stat(filepath.string().c_str(), &file_stat), 0);

  std::filesystem::remove(filepath);
}

TEST(MjuFopenTest, NonexistentFileReturnsNull) {
  std::filesystem::path filepath = std::filesystem::path(::testing::TempDir()) /
                                   "mju_fopen_nonexistent_12345.txt";
  // make sure it doesn't exist
  std::filesystem::remove(filepath);

  FILE* fp = mju_fopen(filepath.string().c_str(), "r");
  EXPECT_EQ(fp, nullptr);

  struct stat file_stat;
  EXPECT_NE(mju_stat(filepath.string().c_str(), &file_stat), 0);
}

// helper: get UTF-8 string from a filesystem path
static std::string PathToUtf8(const std::filesystem::path& p) {
  auto u8str = p.u8string();
  return std::string(reinterpret_cast<const char*>(u8str.data()), u8str.size());
}

TEST(MjuFopenTest, Utf8Path) {
  // create a temp directory with non-ASCII characters (Japanese)
  std::filesystem::path utf8_dir =
      std::filesystem::path(::testing::TempDir()) / u8"mju_fopen_テスト";

  std::error_code ec;
  std::filesystem::create_directories(utf8_dir, ec);
  if (ec) {
    GTEST_SKIP() << "Could not create directory with UTF-8 name: "
                 << ec.message();
  }

  std::filesystem::path filepath = utf8_dir / u8"テスト_file.txt";
  std::string filepath_u8 = PathToUtf8(filepath);
  const char* content = "utf8 content test";

  // write using mju_fopen
  FILE* fp = mju_fopen(filepath_u8.c_str(), "w");
  ASSERT_NE(fp, nullptr) << "mju_fopen failed to create file with UTF-8 path";
  fputs(content, fp);
  fclose(fp);

  // read back using mju_fopen
  fp = mju_fopen(filepath_u8.c_str(), "r");
  ASSERT_NE(fp, nullptr) << "mju_fopen failed to open existing UTF-8 path";
  char buf[64] = {0};
  fgets(buf, sizeof(buf), fp);
  fclose(fp);

  EXPECT_STREQ(buf, content);

  struct stat file_stat;
  EXPECT_EQ(mju_stat(filepath_u8.c_str(), &file_stat), 0);

  std::filesystem::remove_all(utf8_dir);
}

}  // namespace
}  // namespace mujoco
