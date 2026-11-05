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

#ifndef MUJOCO_PYTHON_RENDER_FILAMENT_H_
#define MUJOCO_PYTHON_RENDER_FILAMENT_H_

#include <Python.h>

#include <mujoco/mjrfilament.h>
#include "structs.h"

namespace mujoco::python {

namespace _impl {

template <typename T, typename... Args>
T* Create(Args&&... args) {
  if constexpr (std::is_same_v<T, mjrfContext>) {
    return mjrf_createContext(std::forward<Args>(args)...);
  } else if constexpr (std::is_same_v<T, mjrfTexture>) {
    return mjrf_createTexture(std::forward<Args>(args)...);
  } else if constexpr (std::is_same_v<T, mjrfMesh>) {
    return mjrf_createMesh(std::forward<Args>(args)...);
  } else if constexpr (std::is_same_v<T, mjrfScene>) {
    return mjrf_createScene(std::forward<Args>(args)...);
  } else if constexpr (std::is_same_v<T, mjrfLight>) {
    return mjrf_createLight(std::forward<Args>(args)...);
  } else if constexpr (std::is_same_v<T, mjrfRenderable>) {
    return mjrf_createRenderable(std::forward<Args>(args)...);
  } else if constexpr (std::is_same_v<T, mjrfRenderTarget>) {
    return mjrf_createRenderTarget(std::forward<Args>(args)...);
  } else {
    static_assert(!sizeof(T), "Unsupported type for Create");
  }
}

inline void Destroy(mjrfContext* obj)      { mjrf_destroyContext(obj); }
inline void Destroy(mjrfTexture* obj)      { mjrf_destroyTexture(obj); }
inline void Destroy(mjrfMesh* obj)         { mjrf_destroyMesh(obj); }
inline void Destroy(mjrfScene* obj)        { mjrf_destroyScene(obj); }
inline void Destroy(mjrfLight* obj)        { mjrf_destroyLight(obj); }
inline void Destroy(mjrfRenderable* obj)   { mjrf_destroyRenderable(obj); }
inline void Destroy(mjrfRenderTarget* obj) { mjrf_destroyRenderTarget(obj); }

}  // namespace _impl

// Either owns the lifetime of the wrapped object (using a PyCapsule) or acts as
// a simple wrapper around an existing instance of the object. The former is
// used when creating objects in python code.  The latter is used when returning
// objects from the C API.
#define DEFINE_WRAPPED_OBJECT(PY_TYPE_, TYPE_)                           \
  class PY_TYPE_ : public _impl::WrapperBase<TYPE_> {                    \
    static void DestroyCapsule(PyObject* pyobj) {                        \
      void* ptr = (PyCapsule_GetPointer(pyobj, nullptr));                \
      if (ptr) {                                                         \
        _impl::Destroy(static_cast<TYPE_*>(ptr));                        \
      }                                                                  \
    }                                                                    \
                                                                         \
   public:                                                               \
    template <typename... Args>                                          \
    PY_TYPE_(Args&&... args)                                             \
        : _impl::WrapperBase<TYPE_>(                                     \
              _impl::Create<TYPE_>(std::forward<Args>(args)...),         \
              &DestroyCapsule) {}                                        \
    PY_TYPE_(TYPE_* ptr) : _impl::WrapperBase<TYPE_>(ptr, nullptr) {}    \
    PY_TYPE_(const TYPE_* ptr)                                           \
        : _impl::WrapperBase<TYPE_>(const_cast<TYPE_*>(ptr), nullptr) {} \
  }

DEFINE_WRAPPED_OBJECT(PyContext, mjrfContext);
DEFINE_WRAPPED_OBJECT(PyTexture, mjrfTexture);
DEFINE_WRAPPED_OBJECT(PyMesh, mjrfMesh);
DEFINE_WRAPPED_OBJECT(PyScene, mjrfScene);
DEFINE_WRAPPED_OBJECT(PyLight, mjrfLight);
DEFINE_WRAPPED_OBJECT(PyRenderable, mjrfRenderable);
DEFINE_WRAPPED_OBJECT(PyRenderTarget, mjrfRenderTarget);
#undef DEFINE_WRAPPED_OBJECT

}  // namespace mujoco::python

#endif  // MUJOCO_PYTHON_RENDER_FILAMENT_H_
