// Copyright 2022 DeepMind Technologies Limited
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

#include <Python.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <ios>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "errors.h"
#include "gil.h"
#include "private.h"
#include "raw.h"
#include "serialization.h"
#include "structs.h"
#include "vfs.h"
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace mujoco::python::_impl {

using ::mujoco::python::MutexLockIfGilDisabled;

namespace py = ::pybind11;

namespace {

#define PTRDIFF(x, y)                                                   \
  reinterpret_cast<const char*>(x) - reinterpret_cast<const char*>(y)

// Returns the shape of a NumPy array given the dimensions from an X Macro.
// If dim1 is a _literal_ constant 1, the resulting array is 1-dimensional of
// length dim0, otherwise the resulting array is 2-dimensional of shape
// (dim0, dim1).
#define X_ARRAY_SHAPE(dim0, dim1) XArrayShapeImpl(#dim1)((dim0), (dim1))

std::vector<mjtSize> XArrayShapeImpl1D(mjtSize dim0, mjtSize dim1) {
  return {dim0};
}

std::vector<mjtSize> XArrayShapeImpl2D(mjtSize dim0, mjtSize dim1) {
  return {dim0, dim1};
}

constexpr auto XArrayShapeImpl(const std::string_view dim1_str) {
  if (dim1_str == "1") {
    return XArrayShapeImpl1D;
  } else {
    return XArrayShapeImpl2D;
  }
}

inline std::size_t NConMax(const mjData* d) {
  return d->narena / sizeof(mjContact);
}
template <typename Callback = void()>
struct Cleanup final {
  Callback clean_func;
  Cleanup(Callback callback) : clean_func(std::move(callback)) {}
  ~Cleanup() { clean_func(); }
};

// `Cleanup c = /* callback */;`
//
// C++17 type deduction API for creating an instance of `Cleanup`
template <typename Callback>
Cleanup(Callback callback) -> Cleanup<Callback>;

}  // namespace

// ==================== MJOPTION ===============================================
#define X(type, var, dim)
#define XVEC(type, var, dim) \
  , var(InitPyArray(std::array{dim}, ptr_->var, owner_))
MjOptionWrapper::MjWrapper()
    : WrapperBase([]() {
        raw::MjOption* const opt = new raw::MjOption;
        mj_defaultOption(opt);
        return opt;
      }()) MJOPTION_FIELDS {}

MjOptionWrapper::MjWrapper(raw::MjOption* ptr, py::handle owner)
    : WrapperBase(ptr, owner) MJOPTION_FIELDS {}

#undef XVEC
#undef X

MjOptionWrapper::MjWrapper(const MjOptionWrapper& other) : MjOptionWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVISUAL ===============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjVisualHeadlightWrapper::MjWrapper()
    : WrapperBase(new raw::MjVisualHeadlight{}),
      X(ambient),
      X(diffuse),
      X(specular) {}

MjVisualHeadlightWrapper::MjWrapper(raw::MjVisualHeadlight* ptr,
                                    py::handle owner)
    : WrapperBase(ptr, owner), X(ambient), X(diffuse), X(specular) {}
#undef X

MjVisualHeadlightWrapper::MjWrapper(const MjVisualHeadlightWrapper& other)
    : MjVisualHeadlightWrapper() {
  *this->ptr_ = *other.ptr_;
}

#define X(var) var(InitPyArray(ptr_->var, owner_))
MjVisualRgbaWrapper::MjWrapper()
    : WrapperBase(new raw::MjVisualRgba{}),
      X(fog),
      X(haze),
      X(force),
      X(inertia),
      X(joint),
      X(actuator),
      X(actuatornegative),
      X(actuatorpositive),
      X(com),
      X(camera),
      X(light),
      X(selectpoint),
      X(connect),
      X(contactpoint),
      X(contactforce),
      X(contactfriction),
      X(contacttorque),
      X(contactgap),
      X(rangefinder),
      X(constraint),
      X(slidercrank),
      X(crankbroken),
      X(frustum) {}

MjVisualRgbaWrapper::MjWrapper(raw::MjVisualRgba* ptr, py::handle owner)
    : WrapperBase(ptr, owner),
      X(fog),
      X(haze),
      X(force),
      X(inertia),
      X(joint),
      X(actuator),
      X(actuatornegative),
      X(actuatorpositive),
      X(com),
      X(camera),
      X(light),
      X(selectpoint),
      X(connect),
      X(contactpoint),
      X(contactforce),
      X(contactfriction),
      X(contacttorque),
      X(contactgap),
      X(rangefinder),
      X(constraint),
      X(slidercrank),
      X(crankbroken),
      X(frustum) {}
#undef X

MjVisualRgbaWrapper::MjWrapper(const MjVisualRgbaWrapper& other)
    : MjVisualRgbaWrapper() {
  *this->ptr_ = *other.ptr_;
}

MjVisualWrapper::MjWrapper()
    : WrapperBase(new raw::MjVisual{}),
      headlight(&ptr_->headlight, owner_),
      rgba(&ptr_->rgba, owner_) {}

MjVisualWrapper::MjWrapper(raw::MjVisual* ptr, py::handle owner)
    : WrapperBase(ptr, owner),
      headlight(&ptr_->headlight, owner_),
      rgba(&ptr_->rgba, owner_) {}

MjVisualWrapper::MjWrapper(const MjVisualWrapper& other) : MjVisualWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJMODEL ================================================
static void MjModelCapsuleDestructor(PyObject* pyobj) {
  mj_deleteModel(
      static_cast<raw::MjModel*>(PyCapsule_GetPointer(pyobj, nullptr)));
}

static absl::flat_hash_map<raw::MjModel*, MjModelWrapper*>&
MjModelRawPointerMap() {
  static auto* hash_map =
      new absl::flat_hash_map<raw::MjModel*, MjModelWrapper*>();
  return *hash_map;
}

static std::mutex& MjModelMapMutex() {
  static auto* mtx = new std::mutex;
  return *mtx;
}

MjModelWrapper* MjModelWrapper::FromRawPointer(raw::MjModel* m) noexcept {
  try {
    auto& map = MjModelRawPointerMap();
    {
      py::gil_scoped_acquire gil;
      MutexLockIfGilDisabled lock(MjModelMapMutex());
      auto found = map.find(m);
      return found != map.end() ? found->second : nullptr;
    }
  } catch (...) {
    return nullptr;
  }
}

#undef MJ_M
#define MJ_M(x) ptr_->x
#define X(dtype, var, dim0, dim1) \
  , var(InitPyArray(X_ARRAY_SHAPE(ptr_->dim0, dim1), ptr_->var, owner_))
MjModelWrapper::MjWrapper(raw::MjModel* ptr)
    : WrapperBase(ptr, &MjModelCapsuleDestructor),
      opt(&ptr->opt, owner_),
      vis(&ptr->vis, owner_),
      stat(&ptr->stat, owner_) MJMODEL_POINTERS,
      text_data_bytes(ptr->text_data, ptr->ntextdata),
      names_bytes(ptr->names, ptr->nnames),
      paths_bytes(ptr->paths, ptr->npaths),
      indexer_(ptr, owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(MjModelMapMutex());
    is_newly_inserted = MjModelRawPointerMap().insert({ptr_, this}).second;
  }
  if (!is_newly_inserted) {
    throw UnexpectedError(
        "MjModelWrapper(mjModel*): MjModelRawPointerMap already contains this "
        "raw mjModel*");
  }
}

MjModelWrapper::MjWrapper(MjModelWrapper&& other)
    : WrapperBase(other.ptr_, other.owner_),
      opt(&ptr_->opt, owner_),
      vis(&ptr_->vis, owner_),
      stat(&ptr_->stat, owner_) MJMODEL_POINTERS,
      text_data_bytes(ptr_->text_data, ptr_->ntextdata),
      names_bytes(ptr_->names, ptr_->nnames),
      paths_bytes(ptr_->paths, ptr_->npaths),
      indexer_(ptr_, owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(MjModelMapMutex());
    is_newly_inserted =
        MjModelRawPointerMap().insert_or_assign(ptr_, this).second;
  }
  if (is_newly_inserted) {
    throw UnexpectedError(
        "MjModelRawPointerMap does not contains the moved-from mjModel*");
  }
  other.ptr_ = nullptr;
}
#undef X
#undef MJ_M
#define MJ_M(x) x

// Delegating to the MjModelWrapper::MjWrapper(raw::MjModel*) constructor,
// no need to modify MjModelRawPointerMap here.
MjModelWrapper::MjWrapper(const MjModelWrapper& other)
    : MjModelWrapper(InterceptMjErrors(mj_copyModel)(NULL, other.get())) {}

MjModelWrapper::~MjWrapper() {
  if (ptr_) {
    bool erased = false;
    {
      py::gil_scoped_acquire gil;
      MutexLockIfGilDisabled lock(MjModelMapMutex());
      erased = MjModelRawPointerMap().erase(ptr_);
    }
    if (!erased) {
      std::cerr << "MjModelRawPointerMap does not contain this raw mjModel*"
                << std::endl;
      std::terminate();
    }
  }
}

namespace {
mjVFS* GetOpenVfs(MjVfs* vfs) {
  if (vfs) {
    if (!vfs->is_open()) {
      throw std::runtime_error("VFS is closed");
    }
    return vfs->get();
  }
  return nullptr;
}
}  // namespace

// Helper function for both LoadXMLFile and LoadBinaryFile.
// Creates a temporary MJB from the assets dictionary if one is supplied.
template <typename LoadFunc>
static raw::MjModel* LoadModelFileImpl(const std::string& filename,
                                       const std::vector<VfsAsset>& assets,
                                       mjVFS* vfs,
                                       LoadFunc&& loadfunc) {
  if (!assets.empty() && vfs != nullptr) {
    throw py::value_error("Cannot specify both 'assets' and 'vfs'.");
  }

  std::optional<mjVFS> local_vfs;
  Cleanup vfs_cleanup = [&]() {
    if (local_vfs.has_value()) {
      mj_deleteVFS(vfs);
    };
  };

  if (!assets.empty()) {
    vfs = &local_vfs.emplace();
    mj_defaultVFS(vfs);

    for (const auto& asset : assets) {
      std::string buffer_name = StripPath(asset.name);
      const int vfs_error = InterceptMjErrors(mj_addBufferVFS)(
          vfs, buffer_name.c_str(), asset.content,
          asset.content_size);
      if (vfs_error) {
        if (vfs_error == 2) {
          throw py::value_error("Repeated file name in assets dict: " +
                                buffer_name);
        } else {
          throw py::value_error("Asset failed to load: " + buffer_name);
        }
      }
    }
  }

  raw::MjModel* model = loadfunc(filename.c_str(), vfs);
  if (model && !model->buffer) {
    mj_deleteModel(model);
    model = nullptr;
  }

  return model;
}

MjModelWrapper MjModelWrapper::LoadXMLFile(
    const std::string& filename,
    const std::optional<std::unordered_map<std::string, py::bytes>>& assets,
    MjVfs* vfs) {
  if (assets.has_value() && vfs != nullptr) {
    throw py::value_error("Cannot specify both 'assets' and 'vfs'.");
  }
  mjVFS* vfs_ptr = GetOpenVfs(vfs);

  const auto converted_assets = ConvertAssetsDict(assets);
  raw::MjModel* model;
  {
    py::gil_scoped_release no_gil;
    char error[1024];
    model = LoadModelFileImpl(
        filename, converted_assets, vfs_ptr,
        [&error](const char* filename, const mjVFS* vfs) {
          return InterceptMjErrors(mj_loadXML)(
              filename, vfs, error, sizeof(error));
        });
    if (!model) {
      throw py::value_error(error);
    }
  }
  return MjModelWrapper(model);
}

MjModelWrapper MjModelWrapper::LoadBinaryFile(
    const std::string& filename,
    const std::optional<std::unordered_map<std::string, py::bytes>>& assets,
    MjVfs* vfs) {
  if (assets.has_value() && vfs != nullptr) {
    throw py::value_error("Cannot specify both 'assets' and 'vfs'.");
  }
  mjVFS* vfs_ptr = GetOpenVfs(vfs);

  const auto converted_assets = ConvertAssetsDict(assets);
  raw::MjModel* model;
  {
    py::gil_scoped_release no_gil;
    model = LoadModelFileImpl(filename, converted_assets,
                              vfs_ptr,
                              InterceptMjErrors(mj_loadModel));
    if (!model) {
      throw py::value_error("mj_loadModel: failed to load from mjb");
    }
  }
  return MjModelWrapper(model);
}

MjModelWrapper MjModelWrapper::LoadXML(
    const std::string& xml,
    const std::optional<std::unordered_map<std::string, py::bytes>>& assets,
    MjVfs* vfs) {
  if (assets.has_value() && vfs != nullptr) {
    throw py::value_error("Cannot specify both 'assets' and 'vfs'.");
  }
  mjVFS* vfs_ptr = GetOpenVfs(vfs);

  auto converted_assets = ConvertAssetsDict(assets);
  raw::MjModel* model;
  {
    py::gil_scoped_release no_gil;
    std::string model_identifier = "model_.xml";
    bool file_added = false;
    Cleanup file_cleanup = [&]() {
      if (file_added) {
        mj_deleteFileVFS(vfs_ptr, model_identifier.c_str());
      }
    };
    if (vfs_ptr != nullptr) {
      while (mj_containsBufferVFS(vfs_ptr, model_identifier.c_str())) {
        model_identifier =
            model_identifier.substr(0, model_identifier.size() - 4) + "_.xml";
      }

      mj_addBufferVFS(vfs_ptr, model_identifier.c_str(), xml.c_str(),
                      xml.length());
      file_added = true;
    } else {
      while (assets.has_value() &&
             assets->find(model_identifier) != assets->end()) {
        model_identifier =
            model_identifier.substr(0, model_identifier.size() - 4) + "_.xml";
      }
      converted_assets.emplace_back(model_identifier.c_str(), xml.c_str(),
                                    xml.length());
    }
    char error[1024];
    model = LoadModelFileImpl(model_identifier, converted_assets,
                              vfs_ptr,
                              [&error](const char* filename, const mjVFS* vfs) {
                                return InterceptMjErrors(mj_loadXML)(
                                    filename, vfs, error, sizeof(error));
                              });
    if (!model) {
      throw py::value_error(error);
    }
  }
  return MjModelWrapper(model);
}

MjModelWrapper MjModelWrapper::WrapRawModel(raw::MjModel* m) {
  return MjModelWrapper(m);
}

namespace {

// A byte at the start of serialized mjModel structs, which can be incremented
// when we change the serialization logic to reject pickles from an unsupported
// future version.
constexpr static char kSerializationVersion = 1;

void CheckInput(const std::istream& input, std::string class_name) {
  if (input.fail()) {
    throw py::value_error("Invalid serialized " + class_name + ".");
  }
}

}  // namespace

void MjModelWrapper::Serialize(std::ostream& output) const {
  WriteChar(output, kSerializationVersion);

  mjtSize model_size = mj_sizeModel(get());
  WriteInt(output, model_size);
  std::string buffer(model_size, 0);
  mj_saveModel(get(), nullptr, buffer.data(), model_size);
  WriteBytes(output, buffer.data(), model_size);
}

std::unique_ptr<MjModelWrapper> MjModelWrapper::Deserialize(
    std::istream& input) {
  CheckInput(input, "mjModel");

  char serializationVersion = ReadChar(input);
  CheckInput(input, "mjModel");

  if (serializationVersion != kSerializationVersion) {
    throw py::value_error("Incompatible serialization version.");
  }

  std::size_t model_size = ReadInt(input);
  CheckInput(input, "mjModel");
  if (model_size < 0) {
    throw py::value_error("Invalid serialized mjModel.");
  }
  std::string model_bytes(model_size, 0);
  ReadBytes(input, model_bytes.data(), model_size);
  CheckInput(input, "mjModel");

  raw::MjModel* model = LoadModelFileImpl(
      "model.mjb",
      {{"model.mjb", model_bytes.data(), static_cast<std::size_t>(model_size)}},
      nullptr,
      InterceptMjErrors(mj_loadModel));
  if (!model) {
    throw py::value_error("Invalid serialized mjModel.");
  }
  return std::unique_ptr<MjModelWrapper>(new MjModelWrapper(model));
}

// ==================== MJPRECONTACT ==========================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjPreContactWrapper::MjWrapper()
    : WrapperBase(new raw::MjPreContact{}),
      X(pos),
      X(normal),
      X(tangent) {}

MjPreContactWrapper::MjWrapper(raw::MjPreContact* ptr, py::handle owner)
    : WrapperBase(ptr, owner),
      X(pos),
      X(normal),
      X(tangent) {}
#undef X

MjPreContactWrapper::MjWrapper(const MjPreContactWrapper& other)
    : MjPreContactWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJCONTACT ==============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjContactWrapper::MjWrapper()
    : WrapperBase(new raw::MjContact{}),
      X(pos),
      X(frame),
      X(friction),
      X(solref),
      X(solreffriction),
      X(solimp),
      X(surfacevel),
      X(H),
      X(geom),
      X(flex),
      X(elem),
      X(vert) {}

MjContactWrapper::MjWrapper(raw::MjContact* ptr, py::handle owner)
    : WrapperBase(ptr, owner),
      X(pos),
      X(frame),
      X(friction),
      X(solref),
      X(solreffriction),
      X(solimp),
      X(surfacevel),
      X(H),
      X(geom),
      X(flex),
      X(elem),
      X(vert) {}
#undef X

MjContactWrapper::MjWrapper(const MjContactWrapper& other)
    : MjContactWrapper() {
  *this->ptr_ = *other.ptr_;
}

MjContactList::MjStructList(raw::MjContact* ptr, int nconmax, int* ncon,
                            py::handle owner)
    : StructListBase(ptr, nconmax, owner, /* lazy = */ true), ncon_(ncon) {}

// Slicing
MjContactList::MjStructList(MjContactList& other, py::slice slice)
    : StructListBase(other, slice), ncon_(nullptr) {}

// ==================== MJDATA =================================================
static void MjDataCapsuleDestructor(PyObject* pyobj) {
  mj_deleteData(
      static_cast<raw::MjData*>(PyCapsule_GetPointer(pyobj, nullptr)));
}

absl::flat_hash_map<raw::MjData*, MjDataWrapper*>& MjDataRawPointerMap() {
  static auto* hash_map =
      new absl::flat_hash_map<raw::MjData*, MjDataWrapper*>();
  return *hash_map;
}

static std::mutex& MjDataMapMutex() {
  static auto* mtx = new std::mutex;
  return *mtx;
}

MjDataWrapper* MjDataWrapper::FromRawPointer(raw::MjData* m) noexcept {
  try {
    auto& map = MjDataRawPointerMap();
    {
      py::gil_scoped_acquire gil;
      MutexLockIfGilDisabled lock(MjDataMapMutex());
      auto found = map.find(m);
      return found != map.end() ? found->second : nullptr;
    }
  } catch (...) {
    return nullptr;
  }
}

MjDataWrapper::MjWrapper(MjModelWrapper* model)
    : WrapperBase(InterceptMjErrors(mj_makeData)(model->get()),
                  &MjDataCapsuleDestructor),
#undef MJ_M
#define MJ_M(x) model->get()->x
#define X(dtype, var, dim0, dim1) \
  var(InitPyArray(X_ARRAY_SHAPE(model->get()->dim0, dim1), ptr_->var, owner_)),
      MJDATA_POINTERS
#undef MJ_M
#define MJ_M(x) (x)
#undef X

      contact(MjContactList(ptr_->contact, NConMax(ptr_), &ptr_->ncon, owner_)),

#define X(dtype, var, dim0, dim1) var(InitPyArray(ptr_->var, owner_)),
      MJDATA_VECTOR
#undef X
      model_(model),
      model_ref_(py::cast(model_)),
      indexer_(ptr_, model_->get(), owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(MjDataMapMutex());
    is_newly_inserted = MjDataRawPointerMap().insert({ptr_, this}).second;
  }
  if (!is_newly_inserted) {
    throw UnexpectedError(
        "MjDataRawPointerMap already contains this raw mjData*");
  }
}

MjDataWrapper::MjWrapper(const MjDataWrapper& other)
    : WrapperBase(other.Copy(), &MjDataCapsuleDestructor),
#undef MJ_M
#define MJ_M(x) other.model_->get()->x
#define X(dtype, var, dim0, dim1)                                            \
  var(InitPyArray(X_ARRAY_SHAPE(other.model_->get()->dim0, dim1), ptr_->var, \
                  owner_)),
      MJDATA_POINTERS
#undef MJ_M
#define MJ_M(x) (x)
#undef X

      contact(MjContactList(ptr_->contact, NConMax(ptr_), &ptr_->ncon, owner_)),

#define X(dtype, var, dim0, dim1) var(InitPyArray(ptr_->var, owner_)),
      MJDATA_VECTOR
#undef X
      model_(other.model_),
      model_ref_(other.model_ref_),
      indexer_(ptr_, model_->get(), owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    is_newly_inserted = MjDataRawPointerMap().insert({ptr_, this}).second;
  }
  if (!is_newly_inserted) {
    throw UnexpectedError(
        "MjDataRawPointerMap already contains this raw mjData*");
  }
}

MjDataWrapper::MjWrapper(MjDataWrapper&& other)
    : WrapperBase(other.ptr_, other.owner_),
#undef MJ_M
#define MJ_M(x) other.model_->get()->x
#define X(dtype, var, dim0, dim1)                                            \
  var(InitPyArray(X_ARRAY_SHAPE(other.model_->get()->dim0, dim1), ptr_->var, \
                  owner_)),
      MJDATA_POINTERS
#undef MJ_M
#define MJ_M(x) (x)
#undef X

      contact(MjContactList(ptr_->contact, NConMax(ptr_), &ptr_->ncon, owner_)),

#define X(dtype, var, dim0, dim1) var(InitPyArray(ptr_->var, owner_)),
      MJDATA_VECTOR
#undef X
      model_(other.model_),
      model_ref_(std::move(other.model_ref_)),
      indexer_(ptr_, model_->get(), owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(MjDataMapMutex());
    is_newly_inserted =
        MjDataRawPointerMap().insert_or_assign(ptr_, this).second;
  }
  if (is_newly_inserted) {
    throw UnexpectedError(
        "MjDataRawPointerMap does not contains the moved-from mjData*");
  }
  other.ptr_ = nullptr;
}

MjDataWrapper::MjWrapper(const MjDataWrapper& other, MjModelWrapper* model)
    : WrapperBase(other.Copy(), &MjDataCapsuleDestructor),
#undef MJ_M
#define MJ_M(x) other.model_->get()->x
#define X(dtype, var, dim0, dim1)                                            \
  var(InitPyArray(X_ARRAY_SHAPE(other.model_->get()->dim0, dim1), ptr_->var, \
                  owner_)),
      MJDATA_POINTERS
#undef MJ_M
#define MJ_M(x) (x)
#undef X

      contact(MjContactList(ptr_->contact, NConMax(ptr_), &ptr_->ncon, owner_)),

#define X(dtype, var, dim0, dim1) var(InitPyArray(ptr_->var, owner_)),
      MJDATA_VECTOR
#undef X
      model_(model),
      model_ref_(py::cast(model_)),
      indexer_(ptr_, model_->get(), owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(MjDataMapMutex());
    is_newly_inserted = MjDataRawPointerMap().insert({ptr_, this}).second;
  }
  if (!is_newly_inserted) {
    throw UnexpectedError(
        "MjDataRawPointerMap already contains this raw mjData*");
  }
}

MjDataWrapper::MjWrapper(MjModelWrapper* model, raw::MjData* d)
    : WrapperBase(d, &MjDataCapsuleDestructor),
#undef MJ_M
#define MJ_M(x) model->get()->x
#define X(dtype, var, dim0, dim1) \
  var(InitPyArray(X_ARRAY_SHAPE(model->get()->dim0, dim1), ptr_->var, owner_)),
      MJDATA_POINTERS
#undef MJ_M
#define MJ_M(x) (x)
#undef X

      contact(MjContactList(ptr_->contact, NConMax(ptr_), &ptr_->ncon, owner_)),

#define X(dtype, var, dim0, dim1) var(InitPyArray(ptr_->var, owner_)),
      MJDATA_VECTOR
#undef X
      model_(model),
      model_ref_(py::cast(model_)),
      indexer_(ptr_, model_->get(), owner_) {
  bool is_newly_inserted = false;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(MjDataMapMutex());
    is_newly_inserted = MjDataRawPointerMap().insert({ptr_, this}).second;
  }
  if (!is_newly_inserted) {
    throw UnexpectedError(
        "MjDataRawPointerMap already contains this raw mjData*");
  }
}

MjDataWrapper::~MjWrapper() {
  if (ptr_) {
    bool erased = false;
    {
      py::gil_scoped_acquire gil;
      MutexLockIfGilDisabled lock(MjDataMapMutex());
      erased = MjDataRawPointerMap().erase(ptr_);
    }
    if (!erased) {
      std::cerr << "MjDataRawPointerMap does not contain this raw mjData*"
                << std::endl;
      std::terminate();
    }
  }
}

void MjDataWrapper::Serialize(std::ostream& output) const {
  // TODO: Replace this custom serialization with a protobuf
  WriteChar(output, kSerializationVersion);

  model_->Serialize(output);

  // Write struct and scalar fields
#define X(var) WriteBytes(output, &ptr_->var, sizeof(ptr_->var))
  X(maxuse_stack);
  X(maxuse_arena);
  X(maxuse_con);
  X(maxuse_efc);
  X(solver);
  X(timer);
  X(warning);
  X(ncon);
  X(ne);
  X(nf);
  X(nJ);
  X(nY);
  X(nA);
  X(nefc);
  X(nisland);
  X(nidof);
  X(time);
  X(energy);
#undef X

  // Write buffer and arena contents
  {

#define X(type, name, nr, nc)    \
  WriteBytes(output, ptr_->name, \
             sizeof(type) * (this->model_->get()->nr) * (nc));
    MJDATA_POINTERS
#undef X

#undef MJ_M
#define MJ_M(x) this->model_->get()->x
#undef MJ_D
#define MJ_D(x) this->ptr_->x
#define X(type, name, nr, nc)                                \
  if ((nr) * (nc)) {                                         \
    WriteBytes(output, ptr_->name,                           \
               ptr_->name ? sizeof(type) * (nr) * (nc) : 0); \
  }

    MJDATA_ARENA_POINTERS_CONTACT
    MJDATA_ARENA_POINTERS_SOLVER
    if (mj_isDual(this->model_->get())) {
      MJDATA_ARENA_POINTERS_DUAL
    }
    if (this->ptr_->nisland) {
      MJDATA_ARENA_POINTERS_ISLAND
    }
#undef MJ_M
#define MJ_M(x) x
#undef MJ_D
#define MJ_D(x) x
#undef X
  }
}

MjDataWrapper MjDataWrapper::Deserialize(std::istream& input) {
  char serializationVersion = ReadChar(input);
  CheckInput(input, "mjData");

  if (serializationVersion != kSerializationVersion) {
    throw py::value_error("Incompatible serialization version.");
  }

  // Read the model that was used to create the mjData.
  std::unique_ptr<MjModelWrapper> m_wrapper =
      MjModelWrapper::Deserialize(input);
  raw::MjModel& m = *m_wrapper->get();

  bool is_dual = mj_isDual(&m);

  raw::MjData* d = mj_makeData(&m);
  if (!d) {
    throw py::value_error("Failed to create mjData.");
  }

  // Read structs and scalar fields
#define X(var)                                      \
  ReadBytes(input, (void*)&d->var, sizeof(d->var)); \
  CheckInput(input, "mjData");

  X(maxuse_stack);
  X(maxuse_arena);
  X(maxuse_con);
  X(maxuse_efc);
  X(solver);
  X(timer);
  X(warning);
  X(ncon);
  X(ne);
  X(nf);
  X(nJ);
  X(nY);
  X(nA);
  X(nefc);
  X(nisland);
  X(nidof);
  X(time);
  X(energy);
#undef X

  // Read buffer and arena contents
  {

#define X(type, name, nr, nc) \
  ReadBytes(input, d->name, sizeof(type) * (m.nr) * (nc));
    MJDATA_POINTERS
#undef X

#undef MJ_M
#define MJ_M(x) m.x
#undef MJ_D
#define MJ_D(x) d->x
// arena pointers might be null, so we need to check the size before allocating.
#define X(type, name, nr, nc)                                                 \
  if ((nr) * (nc)) {                                                          \
    std::size_t actual_nbytes = ReadInt(input);                               \
    if (actual_nbytes) {                                                      \
      if (actual_nbytes != sizeof(type) * (nr) * (nc)) {                      \
        input.setstate(input.rdstate() | std::ios_base::failbit);             \
      } else {                                                                \
        d->name = static_cast<decltype(d->name)>(                             \
            mj_arenaAllocByte(d, sizeof(type) * (nr) * (nc), alignof(type))); \
        input.read(reinterpret_cast<char*>(d->name), actual_nbytes);          \
      }                                                                       \
    } else {                                                                  \
      d->name = nullptr;                                                      \
    }                                                                         \
  }

    MJDATA_ARENA_POINTERS_CONTACT
    MJDATA_ARENA_POINTERS_SOLVER
    if (is_dual) {
      MJDATA_ARENA_POINTERS_DUAL
    }
    if (d->nisland) {
      MJDATA_ARENA_POINTERS_ISLAND
    }
#undef MJ_M
#define MJ_M(x) x
#undef MJ_D
#define MJ_D(x) x
#undef X
  }
  CheckInput(input, "mjData");

  // All bytes should have been used.
  input.ignore(1);
  if (!input.eof()) {
    throw py::value_error("Invalid serialized mjData.");
  }

  // Let Python own the model, since MjDataWrapper only keeps a reference to it.
  py::object model_ref = py::cast(std::move(m_wrapper));
  return MjDataWrapper(model_ref.cast<MjModelWrapper*>(), d);
}

raw::MjData* MjDataWrapper::Copy() const {
  const raw::MjModel* m = model_->get();
  return InterceptMjErrors(mj_copyData)(NULL, m, this->ptr_);
}

// ==================== MJSTATISTIC ============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjStatisticWrapper::MjWrapper()
    : WrapperBase(new raw::MjStatistic{}), X(center) {}

MjStatisticWrapper::MjWrapper(raw::MjStatistic* ptr, py::handle owner)
    : WrapperBase(ptr, owner), X(center) {}
#undef X

MjStatisticWrapper::MjWrapper(const MjStatisticWrapper& other)
    : MjStatisticWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJWARNINGSTAT ==========================================
MjWarningStatWrapper::MjWrapper() : WrapperBase(new raw::MjWarningStat{}) {}

MjWarningStatWrapper::MjWrapper(raw::MjWarningStat* ptr, py::handle owner)
    : WrapperBase(ptr, owner) {}

MjWarningStatWrapper::MjWrapper(const MjWarningStatWrapper& other)
    : MjWarningStatWrapper() {
  *this->ptr_ = *other.ptr_;
}

#define X(type, var)                                                       \
  var(std::vector<int>{num}, std::vector<int>{sizeof(raw::MjWarningStat)}, \
      &ptr->var, owner)
MjWarningStatList::MjStructList(raw::MjWarningStat* ptr, int num,
                                py::handle owner)
    : StructListBase(ptr, num, owner), X(int, lastinfo), X(int, number) {}
#undef X

// Slicing
#define X(type, var) var(other.var[slice])
MjWarningStatList::MjStructList(MjWarningStatList& other, py::slice slice)
    : StructListBase(other, slice), X(int, lastinfo), X(int, number) {}
#undef X

// ==================== MJLOGCONFIG ============================================
MjLogConfigWrapper::MjWrapper() : WrapperBase(new raw::MjLogConfig{}) {}

MjLogConfigWrapper::MjWrapper(raw::MjLogConfig* ptr, py::handle owner)
    : WrapperBase(ptr, owner) {}

MjLogConfigWrapper::MjWrapper(const MjLogConfigWrapper& other)
    : MjLogConfigWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJLOGMESSAGE ===========================================
MjLogMessageWrapper::MjWrapper() : WrapperBase(new raw::MjLogMessage{}) {}

MjLogMessageWrapper::MjWrapper(raw::MjLogMessage* ptr, py::handle owner)
    : WrapperBase(ptr, owner) {}

MjLogMessageWrapper::MjWrapper(const MjLogMessageWrapper& other)
    : MjLogMessageWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJTIMERSTAT ============================================
MjTimerStatWrapper::MjWrapper() : WrapperBase(new raw::MjTimerStat{}) {}

MjTimerStatWrapper::MjWrapper(raw::MjTimerStat* ptr, py::handle owner)
    : WrapperBase(ptr, owner) {}

MjTimerStatWrapper::MjWrapper(const MjTimerStatWrapper& other)
    : MjTimerStatWrapper() {
  *this->ptr_ = *other.ptr_;
}

#define X(type, var)                                                     \
  var(std::vector<int>{num}, std::vector<int>{sizeof(raw::MjTimerStat)}, \
      &ptr->var, owner)
MjTimerStatList::MjStructList(raw::MjTimerStat* ptr, int num, py::handle owner)
    : StructListBase(ptr, num, owner), X(mjtNum, duration), X(int, number) {}
#undef X

// Slicing
#define X(type, var) var(other.var[slice])
MjTimerStatList::MjStructList(MjTimerStatList& other, py::slice slice)
    : StructListBase(other, slice), X(mjtNum, duration), X(int, number) {}
#undef X

// ==================== MJSOLVERSTAT ===========================================
MjSolverStatWrapper::MjWrapper() : WrapperBase(new raw::MjSolverStat{}) {}

MjSolverStatWrapper::MjWrapper(raw::MjSolverStat* ptr, py::handle owner)
    : WrapperBase(ptr, owner) {}

MjSolverStatWrapper::MjWrapper(const MjSolverStatWrapper& other)
    : MjSolverStatWrapper() {
  *this->ptr_ = *other.ptr_;
}

#define X(type, var)                                                      \
  var(std::vector<int>{num}, std::vector<int>{sizeof(raw::MjSolverStat)}, \
      &ptr->var, owner)
MjSolverStatList::MjStructList(raw::MjSolverStat* ptr, int num,
                               py::handle owner)
    : StructListBase(ptr, num, owner),
      X(mjtNum, improvement),
      X(mjtNum, gradient),
      X(mjtNum, lineslope),
      X(int, nactive),
      X(int, nchange),
      X(int, neval),
      X(int, nupdate) {}
#undef X
#undef XN

// Slicing
#define X(type, var) var(other.var[slice])
MjSolverStatList::MjStructList(MjSolverStatList& other, py::slice slice)
    : StructListBase(other, slice),
      X(mjtNum, improvement),
      X(mjtNum, gradient),
      X(mjtNum, lineslope),
      X(int, nactive),
      X(int, nchange),
      X(int, neval),
      X(int, nupdate) {}
#undef X

// ==================== MJVPERTURB =============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvPerturbWrapper::MjWrapper()
    : WrapperBase([]() {
        raw::MjvPerturb* const pert = new raw::MjvPerturb;
        mjv_defaultPerturb(pert);
        return pert;
      }()),
      X(refpos),
      X(refquat),
      X(refselpos),
      X(localpos) {}
#undef X

MjvPerturbWrapper::MjWrapper(const MjvPerturbWrapper& other)
    : MjvPerturbWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVCAMERA ==============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvCameraWrapper::MjWrapper()
    : WrapperBase([]() {
        raw::MjvCamera* const cam = new raw::MjvCamera;
        mjv_defaultCamera(cam);
        return cam;
      }()),
      X(lookat) {}
#undef X

MjvCameraWrapper::MjWrapper(const MjvCameraWrapper& other)
    : MjvCameraWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVGLCAMERA ============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvGLCameraWrapper::MjWrapper()
    : WrapperBase(new raw::MjvGLCamera{}), X(pos), X(forward), X(up) {}

MjvGLCameraWrapper::MjWrapper(raw::MjvGLCamera* ptr, py::handle owner)
    : WrapperBase(ptr, owner), X(pos), X(forward), X(up) {}
#undef X

MjvGLCameraWrapper::MjWrapper(raw::MjvGLCamera&& other) : MjvGLCameraWrapper() {
  *this->ptr_ = other;
}

MjvGLCameraWrapper::MjWrapper(const MjvGLCameraWrapper& other)
    : MjvGLCameraWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVGEOM ================================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvGeomWrapper::MjWrapper()
    : WrapperBase(new raw::MjvGeom{}),
      X(size),
      X(pos),
      mat([this]() {
        static_assert(sizeof(ptr_->mat) == sizeof(ptr_->mat[0]) * 9);
        return InitPyArray(std::array{3, 3}, ptr_->mat, owner_);
      }()),
      X(rgba),
      X(texrepeat) {
  mjv_initGeom(ptr_, mjGEOM_NONE, nullptr, nullptr, nullptr, nullptr);
}

MjvGeomWrapper::MjWrapper(raw::MjvGeom* ptr, py::handle owner)
    : WrapperBase(ptr, owner),
      X(size),
      X(pos),
      mat([this]() {
        static_assert(sizeof(ptr_->mat) == sizeof(ptr_->mat[0]) * 9);
        return InitPyArray(std::array{3, 3}, ptr_->mat, owner_);
      }()),
      X(rgba),
      X(texrepeat) {}
#undef X

MjvGeomWrapper::MjWrapper(const MjvGeomWrapper& other) : MjvGeomWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVLIGHT ===============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvLightWrapper::MjWrapper()
    : WrapperBase(new raw::MjvLight{}),
      X(pos),
      X(dir),
      X(attenuation),
      X(ambient),
      X(diffuse),
      X(specular) {}

MjvLightWrapper::MjWrapper(raw::MjvLight* ptr, py::handle owner)
    : WrapperBase(ptr, owner),
      X(pos),
      X(dir),
      X(attenuation),
      X(ambient),
      X(diffuse),
      X(specular) {}
#undef X

MjvLightWrapper::MjWrapper(const MjvLightWrapper& other) : MjvLightWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVOPTION ==============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvOptionWrapper::MjWrapper()
    : WrapperBase([]() {
        raw::MjvOption* const opt = new raw::MjvOption;
        mjv_defaultOption(opt);
        return opt;
      }()),
      X(geomgroup),
      X(sitegroup),
      X(jointgroup),
      X(tendongroup),
      X(actuatorgroup),
      X(flexgroup),
      X(skingroup),
      X(flags) {}
#undef X

MjvOptionWrapper::MjWrapper(const MjvOptionWrapper& other)
    : MjvOptionWrapper() {
  *this->ptr_ = *other.ptr_;
}

// ==================== MJVSCENE ===============================================
static void MjvSceneCapsuleDestructor(PyObject* pyobj) {
  py::gil_scoped_acquire gil;
  auto* scn = static_cast<raw::MjvScene*>(PyCapsule_GetPointer(pyobj, nullptr));
  if (scn) {
    mjv_freeScene(scn);
    delete scn;
  }
}

#define X(var) var(InitPyArray(ptr_->var, owner_))
#define XN(var, n) var(InitPyArray(std::array{n}, ptr_->var, owner_))
MjvSceneWrapper::MjWrapper()
    : WrapperBase(
          []() {
            raw::MjvScene* const scn = new raw::MjvScene;
            mjv_defaultScene(scn);
            InterceptMjErrors(mjv_makeScene)(nullptr, scn, 0);
            return scn;
          }(),
          &MjvSceneCapsuleDestructor),
      nskinvert(0),
      nflexface(0),
      nflexedge(0),
      nflexvert(0),
      XN(geoms, 0),
      XN(geomorder, 0),
      XN(flexedgeadr, 0),
      XN(flexedgenum, 0),
      XN(flexvertadr, 0),
      XN(flexvertnum, 0),
      XN(flexfaceadr, 0),
      XN(flexfacenum, 0),
      XN(flexfaceused, 0),
      XN(flexedge, 0),
      XN(flexvert, 0),
      XN(flexface, 0),
      XN(flexnormal, 0),
      XN(flextexcoord, 0),
      XN(skinfacenum, 0),
      XN(skinvertadr, 0),
      XN(skinvertnum, 0),
      XN(skinvert, 0),
      XN(skinnormal, 0),
      X(lights),
      X(camera),
      X(translate),
      X(rotate),
      X(flags),
      X(framergb) {}

#define XN(var, n) var(InitPyArray(std::array{n}, ptr_->var, owner_))
MjvSceneWrapper::MjWrapper(const MjModelWrapper& model, int maxgeom)
    : WrapperBase(
          [maxgeom](const raw::MjModel* m) {
            raw::MjvScene* const scn = new raw::MjvScene;
            mjv_defaultScene(scn);
            InterceptMjErrors(mjv_makeScene)(m, scn, maxgeom);
            return scn;
          }(model.get()),
          &MjvSceneCapsuleDestructor),
      nskinvert([](const raw::MjModel* m) {
        int nskinvert = 0;
        for (int i = 0; i < m->nskin; ++i) {
          nskinvert += m->skin_vertnum[i];
        }
        return nskinvert;
      }(model.get())),
      nflexface([](const raw::MjModel* m) {
        int nflexface = 0;
        int flexfacenum = 0;
        for (int f = 0; f < m->nflex; f++) {
          if (m->flex_dim[f] == 0) {
            // 1D : 0
            flexfacenum = 0;
          } else if (m->flex_dim[f] == 2) {
            // 2D: 2*fragments + 2*elements
            flexfacenum = 2 * m->flex_shellnum[f] + 2 * m->flex_elemnum[f];
          } else {
            // 3D: max(fragments, 4*maxlayer)
            // find number of elements in biggest layer
            int maxlayer = 0, layer = 0, nlayer = 1;
            while (nlayer) {
              nlayer = 0;
              for (int e = 0; e < m->flex_elemnum[f]; e++) {
                if (m->flex_elemlayer[m->flex_elemadr[f] + e] == layer) {
                  nlayer++;
                }
              }
              maxlayer = mjMAX(maxlayer, nlayer);
              layer++;
            }
            flexfacenum = mjMAX(m->flex_shellnum[f], 4 * maxlayer);
          }

          // accumulate over flexes
          nflexface += flexfacenum;
        }
        return nflexface;
      }(model.get())),
      nflexedge(model.get()->nflexedge),
      nflexvert(model.get()->nflexvert),
      XN(geoms, ptr_->maxgeom),
      XN(geomorder, ptr_->maxgeom),
      XN(flexedgeadr, ptr_->nflex),
      XN(flexedgenum, ptr_->nflex),
      XN(flexvertadr, ptr_->nflex),
      XN(flexvertnum, ptr_->nflex),
      XN(flexfaceadr, ptr_->nflex),
      XN(flexfacenum, ptr_->nflex),
      XN(flexfaceused, ptr_->nflex),
      XN(flexedge, 2 * nflexedge),
      XN(flexvert, 3 * nflexvert),
      XN(flexface, 9 * nflexface),
      XN(flexnormal, 9 * nflexface),
      XN(flextexcoord, 6 * nflexface),
      XN(skinfacenum, ptr_->nskin),
      XN(skinvertadr, ptr_->nskin),
      XN(skinvertnum, ptr_->nskin),
      XN(skinvert, 3 * nskinvert),
      XN(skinnormal, 3 * nskinvert),
      X(lights),
      X(camera),
      X(translate),
      X(rotate),
      X(flags),
      X(framergb) {}
#undef X
#undef XN

template <typename T>
static T* MallocAndCopy(const T* src, int count) {
  if (src && count > 0) {
    T* out = static_cast<T*>(mju_malloc(count * sizeof(T)));
    std::memcpy(out, src, count * sizeof(T));
    return out;
  } else {
    return nullptr;
  }
}

MjvSceneWrapper::MjWrapper(const MjvSceneWrapper& other) : MjvSceneWrapper() {
  mjv_freeScene(ptr_);
  *ptr_ = *other.ptr_;
  nskinvert = other.nskinvert;
  nflexface = other.nflexface;
  nflexedge = other.nflexedge;
  nflexvert = other.nflexvert;

#define XN(var, n)                               \
  ptr_->var = MallocAndCopy(other.ptr_->var, n); \
  var = InitPyArray(std::array{n}, ptr_->var, owner_);

  XN(geoms, ptr_->maxgeom);
  XN(geomorder, ptr_->maxgeom);
  XN(flexedgeadr, ptr_->nflex);
  XN(flexedgenum, ptr_->nflex);
  XN(flexvertadr, ptr_->nflex);
  XN(flexvertnum, ptr_->nflex);
  XN(flexfaceadr, ptr_->nflex);
  XN(flexfacenum, ptr_->nflex);
  XN(flexfaceused, ptr_->nflex);
  XN(flexedge, 2 * nflexedge);
  XN(flexvert, 3 * nflexvert);
  XN(flexface, 9 * nflexface);
  XN(flexnormal, 9 * nflexface);
  XN(flextexcoord, 6 * nflexface);
  XN(skinfacenum, ptr_->nskin);
  XN(skinvertadr, ptr_->nskin);
  XN(skinvertnum, ptr_->nskin);
  XN(skinvert, 3 * nskinvert);
  XN(skinnormal, 3 * nskinvert);

#undef XN
}

// ==================== MJVFIGURE ==============================================
#define X(var) var(InitPyArray(ptr_->var, owner_))
MjvFigureWrapper::MjWrapper()
    : WrapperBase([]() {
        raw::MjvFigure* const fig = new raw::MjvFigure;
        mjv_defaultFigure(fig);
        return fig;
      }()),
      X(flg_ticklabel),
      X(gridsize),
      X(gridrgb),
      X(figurergba),
      X(panergba),
      X(legendrgba),
      X(textrgb),
      X(linergb),
      X(range),
      X(highlight),
      X(linepnt),
      X(linedata),
      X(xaxispixel),
      X(yaxispixel),
      X(xaxisdata),
      X(yaxisdata),
#undef X

      linename([](raw::MjvFigure* ptr, py::handle owner) {
// Use a macro to help us static_assert that the array extents here are kept
// in sync with mjVisualize.h.
#define MAKE_STR_ARRAY(N1, N2)                                           \
  static_assert(                                                         \
      std::is_same_v<decltype(raw::MjvFigure::linename), char[N1][N2]>); \
  return py::array(py::dtype("|S" #N2), N1, ptr->linename, owner);
        MAKE_STR_ARRAY(mjMAXLINE, 100);

#undef MAKE_STR_ARRAY
      }(ptr_, owner_)) {
}

MjvFigureWrapper::MjWrapper(const MjvFigureWrapper& other)
    : MjvFigureWrapper() {
  *this->ptr_ = *other.ptr_;
}

}  // namespace mujoco::python::_impl
