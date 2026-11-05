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

#include "xml/xml_global.h"

#include <cstdlib>
#include <mutex>
#include <string>
#include <type_traits>

#include <mujoco/mujoco.h>
#include "xml/xml.h"
#include "xml/xml_util.h"

namespace {

// global user model class
class GlobalModel {
 public:
  // deletes current model and takes ownership of model
  void Set(mjSpec* spec = nullptr);

  // writes XML to string
  std::string ToXML(const mjModel* m, char* error, int error_sz);

 private:
  // using raw pointers as GlobalModel needs to be trivially destructible
  std::mutex* mutex_ = new std::mutex();
  mjSpec*     spec_  = nullptr;
};

std::string GlobalModel::ToXML(const mjModel* m, char* error, int error_sz) {
  std::lock_guard<std::mutex> lock(*mutex_);
  if (!spec_) {
    mjCopyError(error, "No XML model loaded", error_sz);
    return "";
  }
  return WriteXML(m, spec_, error, error_sz);
}

GlobalModel& GetGlobalModel();

void GlobalModel::Set(mjSpec* spec) {
  static std::once_flag atexit_once;
  std::call_once(atexit_once, []() { std::atexit([]() { GetGlobalModel().Set(nullptr); }); });

  std::lock_guard<std::mutex> lock(*mutex_);
  if (spec_ != nullptr) { mj_deleteSpec(spec_); }
  spec_ = spec;
}


// returns a single instance of the global model
GlobalModel& GetGlobalModel() {
  static GlobalModel global_model;

  // global variables must be trivially destructible
  static_assert(std::is_trivially_destructible_v<decltype(global_model)>);
  return global_model;
}

}  // namespace

void _mjPRIVATE__set_xml_spec(mjSpec* spec) {
  GetGlobalModel().Set(spec);
}

std::string GetGlobalXmlSpec(const mjModel* m, char* error, int error_sz) {
  return GetGlobalModel().ToXML(m, error, error_sz);
}
