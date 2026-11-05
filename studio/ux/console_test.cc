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

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>

namespace mujoco::studio {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::MatchesRegex;
using ::testing::Not;

class ConsoleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orig_config_ = mju_getLogConfig();
    mjLogConfig cfg = orig_config_;
    cfg.logto_console = true;
    cfg.logto_file = false;
    mju_setLogConfig(cfg);
    console_.Clear();
  }

  void TearDown() override { mju_setLogConfig(orig_config_); }

  mjLogConfig orig_config_;
  Console& console_ = Console::Get();
};

TEST_F(ConsoleTest, AddsPrefixedLinesAndCountsMessages) {
  console_.Add(mjLOG_INFO, "info");
  console_.Add(mjLOG_WARNING, "two\nlines");
  console_.Add(mjLOG_ERROR, "ERROR: already prefixed");

  EXPECT_THAT(console_.GetLines(),
              ElementsAre("INFO: info", "WARNING: two", "lines", "ERROR: already prefixed"));
  const ConsoleCounts counts = console_.GetCounts();
  EXPECT_EQ(counts.messages, 3);
  EXPECT_EQ(counts.warnings, 1);
  EXPECT_EQ(counts.errors, 1);
}

TEST_F(ConsoleTest, EmptyMessageAddsPrefixOnly) {
  console_.Add(mjLOG_INFO, "");
  EXPECT_THAT(console_.GetLines(), ElementsAre("INFO: "));
}

TEST_F(ConsoleTest, AddsBodyLinesAfterMessage) {
  console_.Add(mjLOG_WARNING, "subject", "body\nmore body");

  // Only the first line gets the prefix, and it all counts as one message.
  EXPECT_THAT(console_.GetLines(), ElementsAre("WARNING: subject", "body", "more body"));
  const ConsoleCounts counts = console_.GetCounts();
  EXPECT_EQ(counts.messages, 1);
  EXPECT_EQ(counts.warnings, 1);
}

TEST_F(ConsoleTest, OnlyLogPrints) {
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  console_.Log(mjLOG_INFO, "printed");
  console_.Log(mjLOG_ERROR, "ERROR: printed");
  console_.Add(mjLOG_ERROR, "not printed");

  // Like the default handler, with a blank line after each message.
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "INFO: printed\n\n");
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "ERROR: printed\n\n");
  EXPECT_THAT(console_.GetLines(),
              ElementsAre("INFO: printed", "ERROR: printed", "ERROR: not printed"));
}

TEST_F(ConsoleTest, LogPrintsAndWritesPerLogConfig) {
  const std::string log_path = testing::TempDir() + "/console_test_log.txt";
  std::remove(log_path.c_str());

  mjLogConfig cfg = mju_getLogConfig();
  cfg.logto_console = false;
  cfg.logto_file = true;
  std::snprintf(cfg.logfile, sizeof(cfg.logfile), "%s", log_path.c_str());
  mju_setLogConfig(cfg);

  testing::internal::CaptureStderr();
  console_.Log(mjLOG_ERROR, "load failed");
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
  // The console still gets the message; only printing is off.
  EXPECT_THAT(console_.GetLines(), ElementsAre("ERROR: load failed"));

  // Log file entries carry a timestamp, like the default handler's.
  std::ifstream file(log_path);
  std::string file_contents((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_THAT(file_contents, MatchesRegex("ERROR .+: load failed\n\n"));
  std::remove(log_path.c_str());
}

TEST_F(ConsoleTest, ClearRemovesLinesAndResetsCounts) {
  console_.Add(mjLOG_WARNING, "warning");
  console_.Add(mjLOG_ERROR, "error");
  console_.Clear();

  EXPECT_THAT(console_.GetLines(), IsEmpty());
  const ConsoleCounts counts = console_.GetCounts();
  EXPECT_EQ(counts.messages, 0);
  EXPECT_EQ(counts.warnings, 0);
  EXPECT_EQ(counts.errors, 0);
}

TEST_F(ConsoleTest, KeepsOnlyTheMostRecentBytes) {
  constexpr size_t kSmallMaxBytes = 1024;
  Console console(kSmallMaxBytes);
  const std::string filler(128, 'x');
  const int num_lines = static_cast<int>(kSmallMaxBytes / filler.size()) + 5;
  for (int i = 0; i < num_lines; ++i) {
    console.Add(mjLOG_INFO, std::to_string(i) + ' ' + filler);
  }

  const std::vector<std::string> lines = console.GetLines();
  ASSERT_THAT(lines, Not(IsEmpty()));
  EXPECT_LT(lines.size(), num_lines);
  size_t bytes = 0;
  for (const std::string& line : lines) {
    bytes += Console::LineBytes(line);
  }
  EXPECT_LE(bytes, kSmallMaxBytes);
  EXPECT_GT(bytes + 2 * filler.size(), kSmallMaxBytes);  // Not too eager.
  // The oldest lines were dropped and the most recent ones kept.
  const int first_kept = num_lines - static_cast<int>(lines.size());
  EXPECT_TRUE(lines.front().starts_with("INFO: " + std::to_string(first_kept) + ' '))
      << lines.front().substr(0, 20);
  EXPECT_TRUE(lines.back().starts_with("INFO: " + std::to_string(num_lines - 1) + ' '))
      << lines.back().substr(0, 20);
  EXPECT_EQ(console.GetCounts().messages, num_lines);
}

// Console::Init installs its handler only once per process, so this is the
// only test that calls it.
TEST_F(ConsoleTest, InitAddsEngineMessagesAndPassesThemOn) {
  static auto* passed_on = new std::vector<std::string>();
  const mjfLogHandler original =
      mju_setLogHandler([](const mjLogMessage* msg) { passed_on->push_back(msg->subject); });
  Console::Init();
  Console::Init();  // Does nothing, so messages are passed on only once.

  mju_warning("engine warning");

  // Turning off terminal printing does not affect the console.
  mjLogConfig cfg = mju_getLogConfig();
  cfg.logto_console = false;
  mju_setLogConfig(cfg);
  mju_warning("not printed");

  // Info messages from topics that are not enabled are skipped, like in the
  // default handler.
  mju_info(mjTOPIC_SLEEP, "topic off");
  cfg.topics |= 1 << (mjTOPIC_SLEEP - 1);
  mju_setLogConfig(cfg);
  mju_info(mjTOPIC_SLEEP, "topic on");

  mju_setLogHandler(original);

  EXPECT_THAT(console_.GetLines(),
              ElementsAre("WARNING: engine warning", "WARNING: not printed", "INFO: topic on"));
  EXPECT_THAT(*passed_on, ElementsAre("engine warning", "not printed", "topic off", "topic on"));
}

}  // namespace
}  // namespace mujoco::studio
