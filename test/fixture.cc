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

#include "test/fixture.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "src/xml/xml_global.h"

namespace mujoco {
namespace {
using ::testing::_;
using ::testing::Not;
using ::testing::Return;
using ::testing::Truly;

// Returns true if the warning matches a known benign warning to ignore.
bool IsBenignWarning(const std::string& msg) {
  static const char* const kBenignWarnings[] = {
      "is not rigid and has no equality constraints",
      "Attach conflict",
      // MujocoTest scans MUJOCO_PLUGIN_DIR on startup. On Windows that is the
      // shared bin/ directory, where libraries that are not plugins (such as
      // the Python-linked USD DLLs) fail to load.
      "Error loading plugin library",
  };
  for (const char* warning : kBenignWarnings) {
    if (absl::StrContains(msg, warning)) {
      return true;
    }
  }
  return false;
}
}  // namespace

thread_local MockWarningHandler* MockWarningHandler::active_handler = nullptr;

// Registers this handler as the active one.
MockWarningHandler::MockWarningHandler() {
  prev_ = active_handler;
  active_handler = this;

  // By default, ignore matches in the benign warnings list
  ON_CALL(*this, Warn(Truly(IsBenignWarning)))
      .WillByDefault([](const std::string&) {});

  // Fail on all other warnings
  ON_CALL(*this, Warn(Not(Truly(IsBenignWarning))))
      .WillByDefault([](const std::string& msg) {
        ADD_FAILURE() << "mju_user_warning: " << msg;
      });
}

// Restores the previously active warning handler.
MockWarningHandler::~MockWarningHandler() { active_handler = prev_; }

// Configures the mock warning handler to ignore all warnings (if empty) or
// expect at least one warning containing substring (if non-empty).
void MockWarningHandler::ExpectWarnings(std::string_view substring) {
  if (substring.empty()) {
    EXPECT_CALL(*this, Warn(_)).WillRepeatedly(Return());
  } else {
    EXPECT_CALL(*this, Warn(::testing::HasSubstr(std::string(substring))))
        .Times(::testing::AtLeast(1))
        .WillRepeatedly(Return());
  }
}

// Returns the active warning handler.
MockWarningHandler* MockWarningHandler::GetActive() { return active_handler; }

namespace {

using ::testing::NotNull;

ABSL_CONST_INIT static absl::Mutex handlers_mutex(absl::kConstInit);
static int guard_count ABSL_GUARDED_BY(handlers_mutex) = 0;
static mjfLogHandler prev_log_handler ABSL_GUARDED_BY(handlers_mutex) = nullptr;

void default_mj_log_handler(const mjLogMessage* msg) {
  std::string subject = msg->subject;
  if (msg->func) {
    subject = absl::StrCat(msg->func, ": ", msg->subject);
  }

  if (msg->level == mjLOG_ERROR) {
    // legacy fallback: some tests still install mju_user_error to intercept
    // errors with longjmp-based capture
    if (mju_user_error) {
      mju_user_error(subject.c_str());
    } else {
      FAIL() << "mju_user_error: " << subject;
    }
  } else if (msg->level == mjLOG_WARNING) {
    std::string full_msg = subject;
    if (msg->body) {
      absl::StrAppend(&full_msg, "\n", msg->body);
    }
    if (auto* handler = MockWarningHandler::GetActive()) {
      handler->Warn(full_msg);
    } else {
      ADD_FAILURE() << "Unexpected warning: " << full_msg;
    }
  }
}
}  // namespace

MujocoErrorTestGuard::MujocoErrorTestGuard() {
  absl::MutexLock lock(handlers_mutex);
  if (++guard_count == 1) {
    prev_log_handler = mju_setLogHandler(default_mj_log_handler);
  }
}

MujocoErrorTestGuard::~MujocoErrorTestGuard() {
  absl::MutexLock lock(handlers_mutex);
  if (--guard_count == 0) {
    mju_setLogHandler(prev_log_handler);
    prev_log_handler = nullptr;
  }
}

namespace {
std::string GetRunfilesPrefix(std::string_view subpath) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir && test_srcdir[0] != '\0') {
    const char* workspace = std::getenv("TEST_WORKSPACE");
    if (workspace && workspace[0] != '\0') {
      std::string ws_path = absl::StrCat(test_srcdir, "/", workspace);
      if (std::filesystem::exists(ws_path)) {
        return absl::StrCat(ws_path, "/", subpath);
      }
    }
    if (std::filesystem::exists(test_srcdir)) {
      return absl::StrCat(test_srcdir, "/", subpath);
    }
  }
  if (std::filesystem::exists(subpath)) {
    return std::string(subpath);
  }
  return "";
}
}  // namespace

const std::string GetTestDataFilePath(std::string_view path) {  // NOLINT
  std::string prefix = GetRunfilesPrefix("test/");
  if (prefix.empty()) {
    return std::string(path);
  }
  return absl::StrCat(prefix, path);
}

const std::string GetModelPath(std::string_view path) {  // NOLINT
  std::string prefix = GetRunfilesPrefix("model/");
  if (prefix.empty()) {
    return absl::StrCat("../model/", path);
  }
  return absl::StrCat(prefix, path);
}

std::string GetMenagerieModelPath(std::string_view path) {
  std::string prefix = GetRunfilesPrefix("third_party/mujoco_menagerie/");
  if (prefix.empty()) {
    return absl::StrCat("../../mujoco_menagerie/", path);
  }
  return absl::StrCat(prefix, path);
}

MjModelPtr LoadModelFromString(std::string_view xml, char* error,
                               int error_size, mjVFS* vfs) {
  if (error) {
    error[0] = '\0';
  }

  // This duplicates the logic in mj_loadXML, but allows us to use a string
  // directly rather than having to write the contents to a file. Most
  // importantly, we "save" the spec to global storage so that subsequent calls
  // to mj_saveLastXML will be done using the parsed mjSpec.
  mjSpec* spec = mj_parseXMLString(xml.data(), vfs, error, error_size);

  mjModel* model = nullptr;
  if (spec) {
    model = mj_compile(spec, vfs);

    if (error) {
      if (!model) {
        strncpy(error, mjs_getError(spec), error_size);
        error[error_size - 1] = '\0';
      } else {
        int num_warnings = mjs_numWarnings(spec);
        if (num_warnings > 0) {
          std::string all_warnings;
          for (int i = 0; i < num_warnings; ++i) {
            if (!all_warnings.empty()) {
              all_warnings += '\n';
            }
            all_warnings += mjs_getWarning(spec, i);
          }
          strncpy(error, all_warnings.c_str(), error_size);
          error[error_size - 1] = '\0';
        } else {
          error[0] = '\0';
        }
      }
    }
  }

  SetGlobalXmlSpec(spec);
  return MjModelPtr(model);
}

MjDataPtr MakeData(const MjModelPtr& model) {
  return MjDataPtr(mj_makeData(model.get()));
}

static void AssertModelNotNull(mjModel* model,
                               const std::array<char, 1024>& error) {
  ASSERT_THAT(model, NotNull()) << "Failed to load model: " << error.data();
}

mjModel* LoadModelFromPath(const char* model_path) {
  const std::string xml_path = GetModelPath(model_path);
  std::array<char, 1024> error;
  mjModel* model =
      mj_loadXML(xml_path.c_str(), nullptr, error.data(), error.size());
  AssertModelNotNull(model, error);
  return model;
}

std::string GetFileContents(const char* path) {
  std::ifstream ifs;
  ifs.open(path, std::ifstream::in);
  EXPECT_FALSE(ifs.fail());
  std::ostringstream sstream;
  sstream << ifs.rdbuf();
  return sstream.str();
}

std::string SaveAndReadXmlImpl(const mjModel* model, const mjSpec* spec) {
  constexpr int kMaxPathLen = 1024;
  std::string path_template =
      std::filesystem::temp_directory_path().append("tmp.XXXXXX").string();
  EXPECT_LT(path_template.size(), kMaxPathLen);

  char filepath[kMaxPathLen];
  mju_strncpy(filepath, path_template.c_str(), path_template.size() + 1);

#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
  int fd = mkstemp(filepath);
  EXPECT_NE(fd, -1) << std::strerror(errno);
#elif defined(_WIN32)
  EXPECT_NE(_mktemp_s(filepath), EINVAL);
#endif

  if (spec) {
    mj_saveXML(spec, filepath, nullptr, 0);
  } else if (model) {
    mj_saveLastXML(filepath, model, nullptr, 0);
  }
  std::string contents = GetFileContents(filepath);

#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
  close(fd);
#endif
  std::remove(filepath);

  return contents;
}

std::string SaveAndReadXml(const mjModel* model) {
  EXPECT_THAT(model, testing::NotNull());
  return SaveAndReadXmlImpl(model, nullptr);
}

std::string SaveAndReadXml(const mjSpec* spec) {
  EXPECT_THAT(spec, testing::NotNull());
  return SaveAndReadXmlImpl(nullptr, spec);
}

std::vector<mjtNum> GetCtrlNoise(const mjModel* m, int nsteps,
                                 mjtNum ctrlnoise) {
  std::vector<mjtNum> ctrl;
  for (int step = 0; step < nsteps; step++) {
    for (int i = 0; i < m->nu; i++) {
      mjtNum center = 0.0;
      mjtNum radius = 1.0;
      mjtNum* range = m->actuator_ctrlrange + 2 * i;
      if (m->actuator_ctrllimited[i]) {
        center = (range[1] + range[0]) / 2;
        radius = (range[1] - range[0]) / 2;
      }
      radius *= ctrlnoise;
      ctrl.push_back(center + radius * (2 * mju_Halton(step, i + 2) - 1));
    }
  }
  return ctrl;
}

MockFilesystem::MockFilesystem(std::string unit_test_name) {
  prefix_ = absl::StrCat("MjMock.", unit_test_name);
  dir_ = "/";
  if (mjp_findProvider(prefix_.c_str()) != nullptr) {
    return;
  }

  mjpResourceProvider resourceProvider;
  mjp_defaultProvider(&resourceProvider);
  resourceProvider.prefix = prefix_.c_str();
  resourceProvider.data = (void*)this;

  resourceProvider.open = +[](mjResource* resource) {
    MockFilesystem* fs = static_cast<MockFilesystem*>(resource->provider->data);
    std::string filename = fs->StripPrefix(resource->name);
    return fs->FileExists(filename) ? 1 : 0;
  };

  resourceProvider.read = +[](mjResource* resource, const void** buffer) {
    MockFilesystem* fs = static_cast<MockFilesystem*>(resource->provider->data);
    std::string filename = fs->StripPrefix(resource->name);
    return (int)fs->GetFile(filename, (const unsigned char**)buffer);
  };

  resourceProvider.close = +[](mjResource* resource) {};
  mjp_registerProvider(&resourceProvider);
}

bool MockFilesystem::AddFile(std::string filename, const unsigned char* data,
                             std::size_t ndata) {
  std::string fullfilename = PathReduce(dir_, filename);
  auto [it, inserted] = filenames_.insert(fullfilename);
  if (inserted) {
    data_[fullfilename] = std::vector(data, data + ndata);
  }
  return inserted;
}

bool MockFilesystem::FileExists(const std::string& filename) {
  std::string fullfilename = PathReduce(dir_, filename);
  return filenames_.find(fullfilename) != filenames_.end();
}

std::size_t MockFilesystem::GetFile(const std::string& filename,
                                    const unsigned char** buffer) const {
  std::string fullfilename = PathReduce(dir_, filename);
  auto it = data_.find(fullfilename);
  if (it == data_.end()) {
    return 0;
  }
  *buffer = it->second.data();
  return it->second.size();
}

void MockFilesystem::ChangeDirectory(std::string dir) {
  if (dir.empty()) {
    return;
  }

  dir_ = PathReduce(dir_, dir);
  if (dir_.back() != '/') {
    dir_ = absl::StrCat(dir_, "/");
  }
}

std::string MockFilesystem::FullPath(const std::string& path) const {
  return absl::StrCat(prefix_, ":", PathReduce(dir_, path));
}

std::string MockFilesystem::StripPrefix(const char* path) const {
  return &path[prefix_.size() + 1];
}

std::string MockFilesystem::PathReduce(const std::string& current_dir,
                                       const std::string& path) {
  std::stringstream stream;
  if (!path.empty() && path[0] != '/') {
    stream = std::stringstream(absl::StrCat(current_dir, path));
  } else {
    stream = std::stringstream(path);
  }

  std::string temp;
  std::vector<std::string> dirs;
  while (std::getline(stream, temp, '/')) {
    if (temp == ".." && !dirs.empty()) {
      dirs.pop_back();
      continue;
    }

    if (temp != "." && !temp.empty()) {
      dirs.push_back(temp);
    }
  }
  if (dirs.empty()) {
    return "/";
  }

  return absl::StrJoin(dirs, "/");
}

}  // namespace mujoco
