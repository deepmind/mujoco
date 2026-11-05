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

#ifndef MUJOCO_PYTHON_STRUCTS_H_
#define MUJOCO_PYTHON_STRUCTS_H_

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <functional>
#include <istream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "gil.h"
#include "indexers.h"
#include "raw.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = ::pybind11;

namespace mujoco::python {

class MjVfs;

namespace _impl {

struct VfsAsset {
  VfsAsset(const char* name, const void* content, std::size_t content_size)
      : name(name), content(content), content_size(content_size) {}
  const char* name;
  const void* content;
  std::size_t content_size;
};

// strip path prefix from filename and make lowercase
inline std::string StripPath(const char* name) {
  std::string filename(name);
  size_t start = filename.find_last_of("/\\");

  // get name without path
  if (start != std::string::npos) {
    filename = filename.substr(start + 1, filename.size() - start - 1);
  }

  // make lowercase
  std::transform(filename.begin(), filename.end(), filename.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return filename;
}


// Converts a dict with py::bytes value to a vector of standard C++ types.
// This allows us to release the GIL early. Note that the vector consists only
// of pointers to existing data so no substantial data copies are being made.
inline std::vector<VfsAsset> ConvertAssetsDict(
    const std::optional<std::unordered_map<std::string, py::bytes>>& assets) {
  std::vector<VfsAsset> out;
  if (assets.has_value()) {
    for (const auto& [name, content] : *assets) {
      out.emplace_back(name.c_str(), PYBIND11_BYTES_AS_STRING(content.ptr()),
                       py::len(content));
    }
  }
  return out;
}

template <typename T>
class WrapperBase {
 public:
  WrapperBase(WrapperBase&& other) = default;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  pybind11::handle owner() const { return owner_; }

 protected:
  static void DefaultCapsuleDestructor(PyObject* pyobj) {
    T* ptr = static_cast<T*>(PyCapsule_GetPointer(pyobj, nullptr));
    if (ptr) {
      delete ptr;
    }
  }

  // Takes ownership of ptr.
  explicit WrapperBase(T* ptr,
                       void (*destructor)(PyObject*) = DefaultCapsuleDestructor)
      : ptr_(ptr),
        owner_(pybind11::capsule(ptr_, /* name = */ nullptr, destructor)) {}

  // `ptr` is owned by `owner`.
  WrapperBase(T* ptr, pybind11::handle owner)
      : ptr_(ptr),
        owner_(pybind11::reinterpret_borrow<pybind11::object>(owner)) {}

  T* ptr_;
  pybind11::object owner_;
};

template <typename T, typename = void>
struct py_array_or_tuple {
  using type = pybind11::tuple;
};

template <typename T>
struct py_array_or_tuple<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = pybind11::array_t<T>;
};

// A type that resolves to a NumPy array if the dtype is numeric, and
// a Python tuple otherwise.
template <typename T> using py_array_or_tuple_t =
    typename py_array_or_tuple<T>::type;

template <typename T>
struct enable_if_mj_struct {};

template <typename T>
class MjWrapper {};

template <typename T>
class StructListBase {
 public:
  StructListBase(T* ptr, int num, pybind11::handle owner, bool lazy = false)
      : ptr_(ptr),
        num_(num),
        owner_(pybind11::reinterpret_borrow<pybind11::object>(owner)) {
    if (!lazy) {
      PopulateUpTo(size());
    }
  }

  StructListBase(const StructListBase& other) = delete;
  StructListBase(StructListBase&& other)
      : ptr_(other.ptr_),
        num_(other.num_),
        stride_(other.stride_),
        owner_(std::move(other.owner_)),
        wrappers_(std::move(other.wrappers_)) {}
  // populate_mutex_ is default-constructed (std::mutex is not movable)

  virtual ~StructListBase() = default;

  MjWrapper<T>& operator[](int i) {
    if (i < 0 || i >= size()) {
      throw pybind11::index_error();
    }
    PopulateUpTo(i);
    return *wrappers_[i];
  }

  virtual int size() const {
    return num_;
  }

  pybind11::ssize_t stride() const { return stride_; }
  T* get() const { return ptr_; }
  pybind11::handle owner() const { return owner_; }

 protected:
  void PopulateUpTo(int n) {
    if (n < 0) {
      return;
    }
    MutexLockIfGilDisabled lock(populate_mutex_);
    wrappers_.reserve(n + 1);
    while (static_cast<int>(wrappers_.size()) <= n) {
      wrappers_.push_back(std::make_shared<MjWrapper<T>>(
          ptr_ + static_cast<pybind11::ssize_t>(wrappers_.size()) * stride_,
          owner_));
    }
  }

  // Slicing
  StructListBase(StructListBase& other, pybind11::slice slice)
      : owner_(other.owner_) {
    pybind11::ssize_t start, stop, step, slicelength;
    if (!slice.compute(other.size(), &start, &stop, &step, &slicelength)) {
      throw pybind11::error_already_set();
    }
    num_ = slicelength;
    stride_ = other.stride_ * step;
    if (slicelength > 0) {
      ptr_ = other.ptr_ + start * other.stride_;
      pybind11::ssize_t max_idx =
          step > 0 ? start + (slicelength - 1) * step : start;
      other.PopulateUpTo(static_cast<int>(max_idx));
      wrappers_.reserve(slicelength);
      for (pybind11::ssize_t i = 0, idx = start; i < slicelength;
           ++i, idx += step) {
        wrappers_.push_back(other.wrappers_[idx]);
      }
    } else {
      ptr_ = nullptr;
    }
  }

  T* ptr_;
  int num_;
  pybind11::ssize_t stride_ = 1;
  pybind11::object owner_;

  // Using shared_ptr here so that we get identical Python objects when slicing.
  std::vector<std::shared_ptr<MjWrapper<T>>> wrappers_;
  mutable std::mutex populate_mutex_;
};

template <typename T>
struct is_mj_struct_list { static constexpr bool value = false; };

template <typename T>
class MjStructList {};

// ==================== MJOPTION ===============================================
template <>
class MjWrapper<raw::MjOption> : public WrapperBase<raw::MjOption> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjOption* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(type, var, dim)
  #define XVEC(type, var, dim)                                   \
    py_array_or_tuple_t<                                         \
        std::remove_all_extents_t<decltype(raw::MjOption::var)>> \
        var;
  MJOPTION_FIELDS
  #undef XVEC
  #undef X
};

using MjOptionWrapper = MjWrapper<raw::MjOption>;

template <>
struct enable_if_mj_struct<raw::MjOption> { using type = void; };

// ==================== MJVISUAL ===============================================
template <>
class MjWrapper<raw::MjVisualHeadlight>
    : public WrapperBase<raw::MjVisualHeadlight> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper& other);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjVisualHeadlight* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(type, var, dim)
  #define XVEC(type, var, dim)                                            \
    py_array_or_tuple_t<                                                  \
        std::remove_all_extents_t<decltype(raw::MjVisualHeadlight::var)>> \
        var;
  MJVISUAL_HEADLIGHT_FIELDS
  #undef XVEC
  #undef X
};

using MjVisualHeadlightWrapper = MjWrapper<raw::MjVisualHeadlight>;

template <>
struct enable_if_mj_struct<raw::MjVisualHeadlight> { using type = void; };

template <>
class MjWrapper<raw::MjVisualRgba> : public WrapperBase<raw::MjVisualRgba> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper& other);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjVisualRgba* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

#define XVEC(type, var, dim)                                       \
  py_array_or_tuple_t<                                             \
      std::remove_all_extents_t<decltype(raw::MjVisualRgba::var)>> \
      var;
  MJVISUAL_RGBA_FIELDS
#undef XVEC
};

using MjVisualRgbaWrapper = MjWrapper<raw::MjVisualRgba>;

template <>
struct enable_if_mj_struct<raw::MjVisualRgba> { using type = void; };

template <>
class MjWrapper<raw::MjVisual> : public WrapperBase<raw::MjVisual> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper& other);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjVisual* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  MjVisualHeadlightWrapper headlight;
  MjVisualRgbaWrapper rgba;
};

using MjVisualWrapper = MjWrapper<raw::MjVisual>;

template <>
struct enable_if_mj_struct<raw::MjVisual> { using type = void; };

// ==================== MJSTATISTIC ============================================
template <>
class MjWrapper<raw::MjStatistic> : public WrapperBase<raw::MjStatistic> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjStatistic* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(var, dim)
  #define XVEC(var, dim)                                            \
    py_array_or_tuple_t<                                            \
        std::remove_all_extents_t<decltype(raw::MjStatistic::var)>> \
        var;
  MJSTATISTIC_FIELDS
  #undef XVEC
  #undef X
};

using MjStatisticWrapper = MjWrapper<raw::MjStatistic>;

template <>
struct enable_if_mj_struct<raw::MjStatistic> { using type = void; };

// ==================== MJWARNINGSTAT ==========================================
template <>
class MjWrapper<raw::MjWarningStat> : public WrapperBase<raw::MjWarningStat> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjWarningStat* ptr, pybind11::handle owner);
  ~MjWrapper() = default;
};

using MjWarningStatWrapper = MjWrapper<raw::MjWarningStat>;

template <>
struct enable_if_mj_struct<raw::MjWarningStat> { using type = void; };

template <>
class MjStructList<raw::MjWarningStat>
    : public StructListBase<raw::MjWarningStat> {
 public:
  MjStructList(raw::MjWarningStat* ptr, int num, pybind11::handle owner);
  MjStructList(MjStructList&&) = default;
  ~MjStructList() override = default;

  using StructListBase::operator[];
  using StructListBase::size;
  MjStructList Slice(pybind11::slice slice) {
    return MjStructList(*this, slice);
  }

#define X(type, var) pybind11::array_t<type> var
  X(int, lastinfo);
  X(int, number);
#undef X

 protected:
  MjStructList(MjStructList& other, pybind11::slice slice);
};

using MjWarningStatList = MjStructList<raw::MjWarningStat>;

template <>
struct py_array_or_tuple<raw::MjWarningStat> {
  using type = MjWarningStatList;
};

template <>
struct is_mj_struct_list<raw::MjWarningStat> {
  static constexpr bool value = true;
};

// ==================== MJLOGCONFIG ============================================
template <>
class MjWrapper<raw::MjLogConfig> : public WrapperBase<raw::MjLogConfig> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjLogConfig* ptr, pybind11::handle owner);
  ~MjWrapper() = default;
};

using MjLogConfigWrapper = MjWrapper<raw::MjLogConfig>;

template <>
struct enable_if_mj_struct<raw::MjLogConfig> { using type = void; };

// ==================== MJLOGMESSAGE ===========================================
template <>
class MjWrapper<raw::MjLogMessage> : public WrapperBase<raw::MjLogMessage> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjLogMessage* ptr, pybind11::handle owner);
  ~MjWrapper() = default;
};

using MjLogMessageWrapper = MjWrapper<raw::MjLogMessage>;

template <>
struct enable_if_mj_struct<raw::MjLogMessage> { using type = void; };

// ==================== MJTIMERSTAT ============================================
template <>
class MjWrapper<raw::MjTimerStat> : public WrapperBase<raw::MjTimerStat> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjTimerStat* ptr, pybind11::handle owner);
  ~MjWrapper() = default;
};

using MjTimerStatWrapper = MjWrapper<raw::MjTimerStat>;

template <>
struct enable_if_mj_struct<raw::MjTimerStat> { using type = void; };

template <>
class MjStructList<raw::MjTimerStat> : public StructListBase<raw::MjTimerStat> {
 public:
  MjStructList(raw::MjTimerStat* ptr, int num, pybind11::handle owner);
  MjStructList(MjStructList&&) = default;
  ~MjStructList() override = default;

  using StructListBase::operator[];
  using StructListBase::size;
  MjStructList Slice(pybind11::slice slice) {
    return MjStructList(*this, slice);
  }

#define X(type, var) pybind11::array_t<type> var
  X(mjtNum, duration);
  X(int, number);
#undef X

 protected:
  MjStructList(MjStructList& other, pybind11::slice slice);
};

using MjTimerStatList = MjStructList<raw::MjTimerStat>;

template <>
struct py_array_or_tuple<raw::MjTimerStat> {
  using type = MjTimerStatList;
};

template <>
struct is_mj_struct_list<raw::MjTimerStat> {
  static constexpr bool value = true;
};

// ==================== MJSOLVERSTAT ===========================================
template <>
class MjWrapper<raw::MjSolverStat> : public WrapperBase<raw::MjSolverStat> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjSolverStat* ptr, pybind11::handle owner);
  ~MjWrapper() = default;
};

using MjSolverStatWrapper = MjWrapper<raw::MjSolverStat>;

template <>
struct enable_if_mj_struct<raw::MjSolverStat> { using type = void; };

template <>
class MjStructList<raw::MjSolverStat>
    : public StructListBase<raw::MjSolverStat> {
 public:
  MjStructList(raw::MjSolverStat* ptr, int num, pybind11::handle owner);
  MjStructList(MjStructList&&) = default;
  ~MjStructList() override = default;

  using StructListBase::operator[];
  using StructListBase::size;
  MjStructList Slice(pybind11::slice slice) {
    return MjStructList(*this, slice);
  }

#define X(type, var) pybind11::array_t<type> var
  X(mjtNum, improvement);
  X(mjtNum, gradient);
  X(mjtNum, lineslope);
  X(int, nactive);
  X(int, nchange);
  X(int, neval);
  X(int, nupdate);
#undef X

 protected:
  MjStructList(MjStructList& other, pybind11::slice slice);
};

using MjSolverStatList = MjStructList<raw::MjSolverStat>;

template <>
struct py_array_or_tuple<raw::MjSolverStat> {
  using type = MjSolverStatList;
};

template <>
struct is_mj_struct_list<raw::MjSolverStat> {
  static constexpr bool value = true;
};

// ==================== MJMODEL ================================================
template <>
class MjWrapper<raw::MjModel> : public WrapperBase<raw::MjModel> {
 public:
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&);

  // Takes ownership of the raw mjModel pointer.
  explicit MjWrapper(raw::MjModel* ptr);

  ~MjWrapper();

  MjModelIndexer& indexer() { return indexer_; }

  void Serialize(std::ostream& output) const;
  static std::unique_ptr<MjWrapper<raw::MjModel>> Deserialize(
      std::istream& input);

  static MjWrapper LoadXMLFile(
      const std::string& filename,
      const std::optional<
          std::unordered_map<std::string, pybind11::bytes>>& assets,
      MjVfs* vfs = nullptr);

  static MjWrapper LoadBinaryFile(
      const std::string& filename,
      const std::optional<
          std::unordered_map<std::string, pybind11::bytes>>& assets,
      MjVfs* vfs = nullptr);

  static MjWrapper LoadXML(
      const std::string& xml,
      const std::optional<
          std::unordered_map<std::string, pybind11::bytes>>& assets,
      MjVfs* vfs = nullptr);

  static MjWrapper WrapRawModel(raw::MjModel* m);

  static constexpr char kFromRawPointer[] =
      "__MUJOCO_STRUCTS_MJMODELWRAPPER_LOOKUP";
  static MjWrapper* FromRawPointer(raw::MjModel* m) noexcept;

  MjOptionWrapper opt;
  MjVisualWrapper vis;
  MjStatisticWrapper stat;

#define X(dtype, var, dim0, dim1) py_array_or_tuple_t<dtype> var;
  MJMODEL_POINTERS
#undef X
  // TODO(nimrod): Exclude text_data and names from the MJMODEL_POINTERS macro.
  pybind11::bytes text_data_bytes;
  pybind11::bytes names_bytes;
  pybind11::bytes paths_bytes;

 protected:
  MjModelIndexer indexer_;
};

using MjModelWrapper = MjWrapper<raw::MjModel>;

template <>
struct enable_if_mj_struct<raw::MjModel> { using type = void; };

// ==================== MJCONTACT ==============================================
template <>
class MjWrapper<raw::MjPreContact> : public WrapperBase<raw::MjPreContact> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjPreContact* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(var)                                                     \
    py_array_or_tuple_t<                                             \
        std::remove_all_extents_t<decltype(raw::MjPreContact::var)>> \
        var
  X(pos);
  X(normal);
  X(tangent);
  #undef X
};

using MjPreContactWrapper = MjWrapper<raw::MjPreContact>;

template <>
class MjWrapper<raw::MjContact> : public WrapperBase<raw::MjContact> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjContact* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(var)                                                  \
    py_array_or_tuple_t<                                          \
        std::remove_all_extents_t<decltype(raw::MjContact::var)>> \
        var
  X(pos);
  X(frame);
  X(friction);
  X(solref);
  X(solreffriction);
  X(solimp);
  X(surfacevel);
  X(H);
  X(geom);
  X(flex);
  X(elem);
  X(vert);
  #undef X
};

using MjContactWrapper = MjWrapper<raw::MjContact>;

template <>
struct enable_if_mj_struct<raw::MjContact> { using type = void; };

template <>
class MjStructList<raw::MjContact> : public StructListBase<raw::MjContact> {
 public:
  MjStructList(raw::MjContact* ptr, int nconmax,
               int* ncon, pybind11::handle owner);
  MjStructList(MjStructList&&) = default;
  ~MjStructList() override = default;

  using StructListBase::operator[];

  int size() const override {
    if (ncon_) {
      return *ncon_;
    } else {
      return StructListBase::size();
    }
  }

  MjStructList Slice(pybind11::slice slice) {
    return MjStructList(*this, slice);
  }

 protected:
  MjStructList(MjStructList& other, pybind11::slice slice);
  int* ncon_ = nullptr;
};

using MjContactList = MjStructList<raw::MjContact>;

template <>
struct py_array_or_tuple<raw::MjContact> {
  using type = MjContactList;
};

template <>
struct is_mj_struct_list<raw::MjContact> {
  static constexpr bool value = true;
};

// ==================== MJDATA =================================================
template <>
class MjWrapper<raw::MjData>: public WrapperBase<raw::MjData> {
 public:
  explicit MjWrapper(MjModelWrapper* model);
  MjWrapper(const MjWrapper& other);
  MjWrapper(MjWrapper&&);

  // Used for deepcopy
  MjWrapper(const MjWrapper& other, MjModelWrapper* model);

  // Internal constructor which takes ownership of given mjData pointer.
  // Used for deserialization and recompile.
  explicit MjWrapper(MjModelWrapper* model, raw::MjData* d);

  ~MjWrapper();

  const MjModelWrapper& model() const { return *model_; }
  MjDataIndexer& indexer() { return indexer_; }

  void Serialize(std::ostream& output) const;
  static MjWrapper<raw::MjData> Deserialize(std::istream& input);

  static constexpr char kFromRawPointer[] =
      "__MUJOCO_STRUCTS_MJDATAWRAPPER_LOOKUP";
  static MjWrapper* FromRawPointer(raw::MjData* m) noexcept;


#define X(dtype, var, dim0, dim1) py_array_or_tuple_t<dtype> var;
  MJDATA_POINTERS
#undef X

  py_array_or_tuple_t<mjContact> contact;

  py_array_or_tuple_t<size_t> maxuse_threadstack;
  py_array_or_tuple_t<raw::MjSolverStat> solver;
  py_array_or_tuple_t<int> solver_niter;
  py_array_or_tuple_t<int> solver_nnz;
  py_array_or_tuple_t<mjtNum> solver_fwdinv;
  py_array_or_tuple_t<raw::MjWarningStat> warning;
  py_array_or_tuple_t<raw::MjTimerStat> timer;
  py_array_or_tuple_t<mjtNum> energy;

 protected:
  raw::MjData* Copy() const;

  // A reference to the model that was used to create this mjData.
  MjModelWrapper* model_;
  // A py::object pointing to the same model as model_, to make sure Python
  // doesn't doesn't garbage collect it until this mjData is released.
  pybind11::object model_ref_;
  MjDataIndexer indexer_;
};

using MjDataWrapper = MjWrapper<raw::MjData>;

template <>
struct enable_if_mj_struct<raw::MjData> { using type = void; };

// ==================== MJVPERTURB =============================================
template <>
class MjWrapper<raw::MjvPerturb> : public WrapperBase<raw::MjvPerturb> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  ~MjWrapper() = default;

  #define X(var)                                                   \
    py_array_or_tuple_t<                                           \
        std::remove_all_extents_t<decltype(raw::MjvPerturb::var)>> \
        var
  X(refpos);
  X(refquat);
  X(refselpos);
  X(localpos);
  #undef X
};

using MjvPerturbWrapper = MjWrapper<raw::MjvPerturb>;

template <>
struct enable_if_mj_struct<raw::MjvPerturb> { using type = void; };

// ==================== MJVCAMERA ==============================================
template <>
class MjWrapper<raw::MjvCamera> : public WrapperBase<raw::MjvCamera> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  ~MjWrapper() = default;

  #define X(var)                                                  \
    py_array_or_tuple_t<                                          \
        std::remove_all_extents_t<decltype(raw::MjvCamera::var)>> \
        var
  X(lookat);
  #undef X
};

using MjvCameraWrapper = MjWrapper<raw::MjvCamera>;

template <>
struct enable_if_mj_struct<raw::MjvCamera> { using type = void; };

// ==================== MJVGLCAMERA ============================================
template <>
class MjWrapper<raw::MjvGLCamera> : public WrapperBase<raw::MjvGLCamera> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjvGLCamera* ptr, pybind11::handle owner);
  explicit MjWrapper(raw::MjvGLCamera&& other);
  ~MjWrapper() = default;

  #define X(var)                                                    \
    py_array_or_tuple_t<                                            \
        std::remove_all_extents_t<decltype(raw::MjvGLCamera::var)>> \
        var
  X(pos);
  X(forward);
  X(up);
  #undef X
};

using MjvGLCameraWrapper = MjWrapper<raw::MjvGLCamera>;

template <>
struct enable_if_mj_struct<raw::MjvGLCamera> { using type = void; };

// ==================== MJVGEOM ================================================
template <>
class MjWrapper<raw::MjvGeom> : public WrapperBase<raw::MjvGeom> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjvGeom* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(var)                                                \
    py_array_or_tuple_t<                                        \
        std::remove_all_extents_t<decltype(raw::MjvGeom::var)>> \
        var
  X(matid);
  X(size);
  X(pos);
  X(mat);
  X(rgba);
  X(texrepeat);
  #undef X
};

using MjvGeomWrapper = MjWrapper<raw::MjvGeom>;

template <>
struct enable_if_mj_struct<raw::MjvGeom> { using type = void; };

// ==================== MJVLIGHT ===============================================
template <>
class MjWrapper<raw::MjvLight> : public WrapperBase<raw::MjvLight> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(raw::MjvLight* ptr, pybind11::handle owner);
  ~MjWrapper() = default;

  #define X(var)                                                 \
    py_array_or_tuple_t<                                         \
        std::remove_all_extents_t<decltype(raw::MjvLight::var)>> \
        var
  X(pos);
  X(dir);
  X(attenuation);
  X(ambient);
  X(diffuse);
  X(specular);
  #undef X
};

using MjvLightWrapper = MjWrapper<raw::MjvLight>;

template <>
struct enable_if_mj_struct<raw::MjvLight> { using type = void; };

// ==================== MJVOPTION ==============================================
template <>
class MjWrapper<raw::MjvOption> : public WrapperBase<raw::MjvOption> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  ~MjWrapper() = default;

  #define X(var)                                                  \
    py_array_or_tuple_t<                                          \
        std::remove_all_extents_t<decltype(raw::MjvOption::var)>> \
        var
  X(geomgroup);
  X(sitegroup);
  X(jointgroup);
  X(tendongroup);
  X(actuatorgroup);
  X(flexgroup);
  X(skingroup);
  X(flags);
  #undef X
};

using MjvOptionWrapper = MjWrapper<raw::MjvOption>;

template <>
struct enable_if_mj_struct<raw::MjvOption> { using type = void; };

// ==================== MJVSCENE ===============================================
template <>
class MjWrapper<raw::MjvScene> : public WrapperBase<raw::MjvScene> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  MjWrapper(const MjModelWrapper& model, int maxgeom);
  ~MjWrapper() = default;

  int nskinvert;
  int nflexface, nflexedge, nflexvert;

  #define X(dtype, var) py_array_or_tuple_t<dtype> var
  X(mjvGeom, geoms);
  X(int, geomorder);
  X(int, flexedgeadr);
  X(int, flexedgenum);
  X(int, flexvertadr);
  X(int, flexvertnum);
  X(int, flexfaceadr);
  X(int, flexfacenum);
  X(int, flexfaceused);
  X(int, flexedge);
  X(float, flexvert);
  X(float, flexface);
  X(float, flexnormal);
  X(float, flextexcoord);
  X(int, skinfacenum);
  X(int, skinvertadr);
  X(int, skinvertnum);
  X(float, skinvert);
  X(float, skinnormal);
  #undef X

  #define X(dtype, var) py_array_or_tuple_t<dtype> var
  X(raw::MjvLight, lights);
  X(raw::MjvGLCamera, camera);
  X(float, translate);
  X(float, rotate);
  X(mjtByte, flags);
  X(float, framergb);
  #undef X
};

using MjvSceneWrapper = MjWrapper<raw::MjvScene>;

template <>
struct enable_if_mj_struct<raw::MjvScene> { using type = void; };

// ==================== MJVFIGURE ==============================================
template <>
class MjWrapper<raw::MjvFigure> : public WrapperBase<raw::MjvFigure> {
 public:
  MjWrapper();
  MjWrapper(const MjWrapper&);
  MjWrapper(MjWrapper&&) = default;
  ~MjWrapper() = default;

  #define X(dtype, var) py_array_or_tuple_t<dtype> var
  X(int, flg_ticklabel);
  X(int, gridsize);
  X(float, gridrgb);
  X(float, figurergba);
  X(float, panergba);
  X(float, legendrgba);
  X(float, textrgb);
  X(float, linergb);
  X(float, range);
  X(int, highlight);
  X(int, linepnt);
  X(float, linedata);
  X(int, xaxispixel);
  X(int, yaxispixel);
  X(float, xaxisdata);
  X(float, yaxisdata);
  #undef X

  pybind11::array linename;
};

using MjvFigureWrapper = MjWrapper<raw::MjvFigure>;

template <>
struct enable_if_mj_struct<raw::MjvFigure> { using type = void; };

#ifdef MEMORY_SANITIZER
template <typename T>
class ScopedMsanDisabler {
 public:
  ScopedMsanDisabler(T* mj)
      : mj_(mj), shadow_(std::malloc(mj_->nbuffer)) {
    __msan_copy_shadow(shadow_, mj_->buffer, mj_->nbuffer);
    __msan_unpoison(mj_->buffer, mj_->nbuffer);
  }
  ~ScopedMsanDisabler() {
    __msan_copy_shadow(mj_->buffer, shadow_, mj_->nbuffer);
    std::free(shadow_);
  }
 private:
  T* mj_;
  void* shadow_;
};
#endif

}  // namespace _impl

template <typename T>
using MjWrapper = typename std::conditional_t<
    std::is_const_v<T>, const _impl::MjWrapper<std::remove_const_t<T>>,
    _impl::MjWrapper<T>>;

template <typename T>
using enable_if_mj_struct_t =
    typename _impl::enable_if_mj_struct<std::remove_const_t<T>>::type;

using _impl::MjOptionWrapper;
using _impl::MjVisualHeadlightWrapper;
using _impl::MjVisualRgbaWrapper;
using _impl::MjVisualWrapper;
using _impl::MjStatisticWrapper;
using _impl::MjWarningStatWrapper;
using _impl::MjLogConfigWrapper;
using _impl::MjLogMessageWrapper;
using _impl::MjTimerStatWrapper;
using _impl::MjSolverStatWrapper;
using _impl::MjModelWrapper;
using _impl::MjDataWrapper;
using _impl::MjContactWrapper;
using _impl::MjPreContactWrapper;
using _impl::MjvPerturbWrapper;
using _impl::MjvCameraWrapper;
using _impl::MjvGLCameraWrapper;
using _impl::MjvGeomWrapper;
using _impl::MjvLightWrapper;
using _impl::MjvOptionWrapper;
using _impl::MjvSceneWrapper;
using _impl::MjvFigureWrapper;

template <typename T>
using MjStructList = typename std::conditional_t<
    std::is_const_v<T>, const _impl::MjStructList<std::remove_const_t<T>>,
    _impl::MjStructList<T>>;

template <typename T>
static constexpr bool is_mj_struct_list_v =
    _impl::is_mj_struct_list<std::remove_const_t<T>>::value;

using _impl::MjContactList;
using _impl::MjWarningStatList;
using _impl::MjTimerStatList;
using _impl::MjSolverStatList;

// ==================== HELPER FUNCTIONS FOR BINDING ARRAYS ====================

// Python array initialization.
// If T is an arithmetic (i.e. numeric) type, returns a NumPy array that wraps
// around a preexisting data buffer.
template <typename T, typename Shape>
std::enable_if_t<std::is_arithmetic_v<T>, pybind11::array_t<T>>
static InitPyArray(Shape&& shape, T* buf, pybind11::handle owner) {
  mjtSize size = 1;
  for (const auto& i : shape) {
    size *= i;
  }
  if (shape.empty() || size == 0) {
    return pybind11::array_t<T>(shape);
  } else {
    return pybind11::array_t<T>(shape, buf, owner);
  }
}

// Same as above, but where we can determine array dimensions through the
// C array type directly.
template <typename T, typename Int, Int N>
std::enable_if_t<(N > 0) && std::is_arithmetic_v<T>, pybind11::array_t<T>>
static InitPyArray(T (&buf)[N], pybind11::handle owner) {
  return pybind11::array_t<T>({N}, &buf[0], owner);
}

template <typename T, typename Int, Int N1, Int N2>
std::enable_if_t<(N1*N2 > 0) && std::is_arithmetic_v<T>, pybind11::array_t<T>>
static InitPyArray(T (&buf)[N1][N2], pybind11::handle owner) {
  return pybind11::array_t<T>(
      {N1, N2}, &buf[0][0], owner);
}

template <typename T, typename Shape>
std::enable_if_t<is_mj_struct_list_v<T>, MjStructList<T>>
static InitPyArray(Shape&& shape, T* buf, pybind11::handle owner) {
  return MjStructList<T>(buf, shape[0], owner);
}

// For arrays of non-arithmetic type, we create tuple of tuples of MjWrapper<T>.
template <typename T, typename Shape>
std::enable_if_t<!std::is_arithmetic_v<T> && !is_mj_struct_list_v<T>,
                 pybind11::tuple>
static InitPyArray(Shape&& shape, T* buf, pybind11::handle owner) {
  mjtSize size = 1;
  for (const auto& i : shape) {
    size *= i;
  }
  if (shape.empty() || !size) {
    return pybind11::tuple();
  }

  pybind11::list out;
  const auto n = shape[0];
  if (shape.size() == 1) {
    for (int i = 0; i < n; ++i) {
      out.append(MjWrapper<T>(&buf[i], owner));
    }
  } else {
    auto block_shape = absl::MakeConstSpan(shape).subspan(1);
    auto block_size = std::accumulate(
        block_shape.begin(), block_shape.end(),
        1, std::multiplies<decltype(n)>());
    for (int i = 0; i < n; ++i) {
      out.append(InitPyArray(block_shape, &buf[i * block_size], owner));
    }
  }
  return std::move(out);
}

// Same as above, but where we can determine array dimensions through the
// C array type directly.
template <typename T, typename Int, Int N>
std::enable_if_t<(N > 0) && !std::is_arithmetic_v<T> &&
                 !std::is_array_v<T> && !is_mj_struct_list_v<T>,
                 pybind11::tuple>
static InitPyArray(T (&buf)[N], pybind11::handle owner) {
  return InitPyArray(std::array{N}, buf, owner);
}

template <typename T, typename Int, Int N1, Int N2>
std::enable_if_t<(N1*N2 > 0) && !std::is_arithmetic_v<T> &&
                 !std::is_array_v<T> && !is_mj_struct_list_v<T>,
                 pybind11::tuple>
static InitPyArray(T (&buf)[N1][N2], pybind11::handle owner) {
  return InitPyArray(std::array{N1, N2}, buf, owner);
}

template <typename T, typename Int, Int N>
std::enable_if_t<is_mj_struct_list_v<T>, MjStructList<T>>
static InitPyArray(T (&buf)[N], pybind11::handle owner) {
  return MjStructList<T>(buf, N, owner);
}

// Helpers for defining array/tuple properties in pybind11 classes.
//
// Defines a NumPy array property of a Python class that supports assignments.
// Specifically, we implement the setter such that `obj.arr = val` is the same
// as `obj.arr[:] = val`.
//
// Use `DefinePyArray(c, "somearray", &MjStructHolder<C>::somearray)`
// as a drop-in replacement for
// `c.def_readonly("somearray", &MjStructHolder<C>::somearray)`.
template <typename T, typename C, typename... O>
static void DefinePyArray(pybind11::class_<C, O...> c, const char* name,
                          pybind11::array_t<T> C::* arr) {
  namespace py = pybind11;
  c.def_property(
      name,
      [arr](const C& wrapper) { return wrapper.*arr; },
      [arr](const C& wrapper, py::handle rhs) -> void {
        (wrapper.*arr)[py::slice(py::none(), py::none(), py::none())] = rhs;
      }
  );
}

// For array of non-arithmetic type, we bind to tuples rather than NumPy array.
// These can't be assigned to directly so we just use def_readonly.
template <typename T, typename C, typename... O>
static void DefinePyArray(pybind11::class_<C, O...> c, const char* name,
                          T C::* arr) {
  c.def_property_readonly(
      name,
      [arr](const C& wrapper) -> auto& { return wrapper.*arr; });
}

template <typename Raw, int N, typename C, typename... O>
static void DefinePyStr(pybind11::class_<C, O...> c, const char* name,
                        char (Raw::* arr)[N]) {
  c.def_property(
      name,
      [arr](const C& c) { return pybind11::str(c.get()->*arr); },
      [name = std::string(name), arr](C& c, std::string_view rhs) {
        constexpr int kMaxLen = sizeof(c.get()->*arr);
        const int actual_len = rhs.size();
        if (actual_len >= kMaxLen) {
          std::ostringstream msg;
          msg << "len(" << name << ") cannot exceed " << kMaxLen - 1
              << ": got length " << actual_len;
          throw pybind11::value_error(msg.str());
        }
        rhs.copy(c.get()->*arr, actual_len);
        (c.get()->*arr)[actual_len] = '\0';
      });
}

// Defines a struct property of a Python class that supports assignments by
// copying the underlying C struct in place.
//
// Use `DefinePyStruct(c, "somestruct", &MjStructHolder<C>::somestruct)`
// as a drop-in replacement for
// `c.def_readonly("somestruct", &MjStructHolder<C>::somestruct)`.
template <typename T, typename C, typename... O>
static void DefinePyStruct(pybind11::class_<C, O...> c, const char* name,
                           T C::* member) {
  c.def_property(
      name,
      [member](const C& wrapper) -> const T& { return wrapper.*member; },
      [member](C& wrapper, const T& rhs) {
        *(wrapper.*member).get() = *rhs.get();
      });
}

// An equals operator that works for comparing numpy arrays too.
// Unlike other objects, the equality operator for numpy arrays returns a
// numpy array.
inline bool FieldsEqual(pybind11::handle lhs, pybind11::handle rhs,
                        pybind11::handle array_equal) {
  // np.array_equal handles non-arrays.
  return PyObject_IsTrue(array_equal(lhs, rhs).ptr());
}

// Returns an iterable object for iterating over attributes of T.
template <typename T>
pybind11::object Dir() {
  pybind11::object type = pybind11::type::of<T>();
  auto dir =
      pybind11::reinterpret_steal<pybind11::object>(PyObject_Dir(type.ptr()));
  if (PyErr_Occurred()) {
    throw pybind11::error_already_set();
  }
  return dir;
}

// Returns true if all public fields in lhs and rhs are equal.
template <typename T>
bool StructsEqual(pybind11::object lhs, pybind11::object rhs) {
  // Equivalent to the following python code:
  // if type(lhs) != type(rhs):
  //   return False
  // for field in dir(lhs):
  //   if field.startswith("_"):
  //     continue
  //   # equal() handles equality of numpy arrays
  //   if not equal(getattr(lhs, field, None), getattr(rhs, field, None)):
  //     return False
  //
  // return True

  auto np = pybind11::module::import("numpy");
  auto array_equal = np.attr("array_equal");

  const pybind11::handle lhs_t = pybind11::type::handle_of(lhs);
  const pybind11::handle rhs_t = pybind11::type::handle_of(rhs);
  if (!lhs_t.is(rhs_t)) {
    return false;
  }
  for (pybind11::handle f : Dir<T>()) {
    auto name = f.cast<std::string_view>();

    if (name.empty() || name[0] == '_') {
      continue;
    }
    pybind11::object l = pybind11::getattr(lhs, f, pybind11::none());
    pybind11::object r = pybind11::getattr(rhs, f, pybind11::none());
    if (!FieldsEqual(l, r, array_equal)) {
      return false;
    }
  }
  return true;
}

// Returns a string representation of a struct like object.
template <typename T>
void StructReprImpl(pybind11::object self, std::ostringstream& result,
                    int indent) {
  result << "<"
         << self.attr("__class__").attr("__name__").cast<std::string_view>();
  for (pybind11::handle f : Dir<T>()) {
    auto name = f.cast<std::string_view>();
    if (name.empty() || name[0] == '_') {
      continue;
    }

    result << "\n" << std::string(indent + 2, ' ') << name << ": "
           << self.attr(f).attr("__repr__")().cast<std::string_view>();
  }
  result << "\n" << std::string(indent, ' ') << ">";
}

template <typename T>
std::string StructRepr(pybind11::object self) {
  std::ostringstream result;
  StructReprImpl<T>(self, result, 0);
  return result.str();
}

template <typename T>
std::string MjModelStructRepr(pybind11::object self) {
#ifdef MEMORY_SANITIZER
  _impl::ScopedMsanDisabler<raw::MjModel> msan_disabler(
      pybind11::cast<T&>(self).ptr());
#endif
  return StructRepr<T>(self);
}

template <typename T>
std::string MjDataStructRepr(pybind11::object self) {
#ifdef MEMORY_SANITIZER
  _impl::ScopedMsanDisabler<raw::MjData> msan_disabler(
      pybind11::cast<T&>(self).ptr());
#endif
  return StructRepr<T>(self);
}

template <typename C, typename... O>
void DefineStructFunctions(pybind11::class_<C, O...> c) {
  c.def("__eq__", StructsEqual<C>);
  c.def("__repr__", StructRepr<C>);
}

}  // namespace mujoco::python

#endif  // MUJOCO_PYTHON_STRUCTS_H_
