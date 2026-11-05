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

#ifndef MUJOCO_STUDIO_SIM_MODEL_HOLDER_H_
#define MUJOCO_STUDIO_SIM_MODEL_HOLDER_H_

#include <memory>
#include <cstddef>
#include <span>  // NOLINT
#include <string>
#include <string_view>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Container for storing an mjModel as well as its associated mjData. Also, if
// applicable, stores the mjSpec from which the model was constructed.
class ModelHolder {
 public:
  ModelHolder(const ModelHolder&) = delete;
  ModelHolder& operator=(const ModelHolder&) = delete;
  ~ModelHolder();

  // Creates an mjModel from an existing mjSpec. This class takes ownership of
  // the spec and will `mj_deleteSpec` it on destruction. If a VFS is provided,
  // its lifetime must outlive this object.
  static std::unique_ptr<ModelHolder> FromSpec(mjSpec* spec, mjVFS* vfs = nullptr);

  // Creates an mjModel by attempting to open and parse the give file.  If a VFS
  // is provided, its lifetime must outlive this object.
  static std::unique_ptr<ModelHolder> FromFile(std::string_view filepath, mjVFS* vfs = nullptr);

  // Creates an mjModel by attempting to decode the given buffer. If a VFS is
  // provided, its lifetime must outlive this object.
  static std::unique_ptr<ModelHolder> FromBuffer(std::span<const std::byte> buffer,
                                                 std::string_view content_type,
                                                 std::string_view filename, mjVFS* vfs = nullptr);

  // Accessors to the MuJoCo structures managed by this object.
  mjSpec* spec() { return spec_; }
  mjData* data() { return data_; }
  mjModel* model() { return model_; }

  // Returns true if the holder holds a valid model (and data).
  bool ok() const { return error_[0] == 0 && model_ && data_; }

  // Returns the error message if the model failed to load.
  std::string_view error() const { return error_; }

  // Returns the warning message(s), newline-separated.
  std::string_view warning() const { return warning_; }

  // Attempts to recompile the model and data in-place from the current spec.
  // If, unsuccessful, the model and data will be reset to nullptr and an error
  // will be set.
  void Recompile();

  // Releases ownership of the underlying MuJoCo structures. Callers are now
  // responsible for calling the appropriate `mj_delete*` function on these
  // objects.
  mjSpec* ReleaseSpec();
  mjData* ReleaseData();
  mjModel* ReleaseModel();

 private:
  explicit ModelHolder(mjVFS* vfs);
  void InitFromSpec(mjSpec* spec);
  void InitFromFile(std::string_view filepath);
  void InitFromBuffer(std::span<const std::byte> buffer, std::string_view content_type,
                      std::string_view filename);
  void PostInit();
  void SetLoadError(std::string_view error);

  mjVFS* vfs_ = nullptr;
  mjVFS local_vfs_ = {nullptr};
  mjSpec* spec_ = nullptr;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  char error_[1000] = "";
  std::string warning_;
};
}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_SIM_MODEL_HOLDER_H_
