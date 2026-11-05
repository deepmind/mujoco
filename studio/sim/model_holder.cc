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

#include "studio/sim/model_holder.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <span>  // NOLINT
#include <string>
#include <string_view>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

std::unique_ptr<ModelHolder> ModelHolder::FromSpec(mjSpec* spec, mjVFS* vfs) {
  auto mh = std::unique_ptr<ModelHolder>(new ModelHolder(vfs));
  mh->InitFromSpec(spec);
  return mh;
}

std::unique_ptr<ModelHolder> ModelHolder::FromFile(std::string_view filepath, mjVFS* vfs) {
  auto mh = std::unique_ptr<ModelHolder>(new ModelHolder(vfs));
  mh->InitFromFile(filepath);
  return mh;
}

std::unique_ptr<ModelHolder> ModelHolder::FromBuffer(std::span<const std::byte> buffer,
                                                     std::string_view content_type,
                                                     std::string_view filename, mjVFS* vfs) {
  auto mh = std::unique_ptr<ModelHolder>(new ModelHolder(vfs));
  mh->InitFromBuffer(buffer, content_type, filename);
  return mh;
}

ModelHolder::ModelHolder(mjVFS* vfs) : vfs_(vfs) {
  if (vfs_ == nullptr) {
    mj_defaultVFS(&local_vfs_);
    vfs_ = &local_vfs_;
  }
}

ModelHolder::~ModelHolder() {
  if (data_) {
    mj_deleteData(data_);
  }
  if (model_) {
    mj_deleteModel(model_);
  }
  if (spec_) {
    mj_deleteSpec(spec_);
  }
  if (vfs_ == &local_vfs_) {
    mj_deleteVFS(&local_vfs_);
  }
}

void ModelHolder::PostInit() {
  if (spec_ && !model_) {
#ifdef __EMSCRIPTEN__
    // Disable threaded compilation on WASM since pthreads are not available.
    spec_->compiler.usethread = 0;
    // Leave file-backed textures by reference so the renderer loads and decodes
    // them on demand rather than duplicating pixels in WASM memory.
    spec_->compiler.passthrough = mjPASSTHROUGH_TEXTURES;
#endif
    model_ = mj_compile(spec_, vfs_);
    if (!model_) {
      SetLoadError(mjs_getError(spec_));
      return;
    }
    int num_warnings = mjs_numWarnings(spec_);
    for (int i = 0; i < num_warnings; ++i) {
      if (!warning_.empty()) warning_ += '\n';
      warning_ += mjs_getWarning(spec_, i);
    }
  }
  data_ = mj_makeData(model_);
  if (!data_) {
    SetLoadError("Error making data for model.");
  }
}

void ModelHolder::InitFromSpec(mjSpec* spec) {
  spec_ = spec;
  PostInit();
}

void ModelHolder::InitFromFile(std::string_view filepath) {
  if (filepath.ends_with(".mjb")) {
    model_ = mj_loadModel(filepath.data(), vfs_);
  } else {
    spec_ = mj_parse(filepath.data(), nullptr, vfs_, error_, sizeof(error_));
  }
  if (error_[0] == 0) {
    PostInit();
  }
}

void ModelHolder::InitFromBuffer(std::span<const std::byte> buffer, std::string_view content_type,
                                 std::string_view filename) {
  if (content_type == "text/xml") {
    const char* ptr = reinterpret_cast<const char*>(buffer.data());
    // Pass vfs so any archive mounts created when parsing attached
    // sub-models persist in the VFS for compilation.
    spec_ = mj_parseXMLString(ptr, vfs_, error_, sizeof(error_));
  } else if (content_type == "application/mjb") {
    model_ = mj_loadModelBuffer(buffer.data(), buffer.size());
  } else if (content_type == "application/zip") {
    std::string name(filename);
    mj_addBufferVFS(vfs_, name.c_str(), buffer.data(), static_cast<int>(buffer.size()));
    spec_ = mj_parse(name.c_str(), content_type.data(), vfs_, error_, sizeof(error_));
  } else {
    SetLoadError("Unknown content type; expected text/xml or application/mjb");
  }
  if (error_[0] == 0) {
    PostInit();
  }
}

void ModelHolder::Recompile() {
  if (!spec_) {
    return;
  }

  if (mj_recompile(spec_, vfs_, model_, data_) != 0) {
    model_ = nullptr;
    data_ = nullptr;
    SetLoadError(mjs_getError(spec_));
  }
}

void ModelHolder::SetLoadError(std::string_view error) {
  strncpy(error_, error.data(), sizeof(error_) - 1);
  error_[sizeof(error_) - 1] = 0;
}

mjSpec* ModelHolder::ReleaseSpec() {
  mjSpec* spec = spec_;
  spec_ = nullptr;
  return spec;
}

mjData* ModelHolder::ReleaseData() {
  mjData* data = data_;
  data_ = nullptr;
  return data;
}

mjModel* ModelHolder::ReleaseModel() {
  mjModel* model = model_;
  model_ = nullptr;
  return model;
}

}  // namespace mujoco::studio
