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

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>

#include <mujoco/mujoco.h>
#include "errors.h"
#include "gil.h"
#include "raw.h"
#include "structs.h"
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace mujoco::python {
namespace {

namespace py = ::pybind11;

[[noreturn]] static void EscapeWithPythonException() {
  mju_error("Python exception raised");
  std::terminate();  // not actually reachable, mju_error doesn't return
}

template <typename T, typename U>
using enable_if_not_const_t =
    std::enable_if_t<std::is_same_v<std::remove_const_t<T>, T>, U>;

// MuJoCo passes raw mjModel* and mjData* as arguments to callbacks, but Python
// callables expect the corresponding MjWrapper objects. To avoid creating new
// wrappers each time we enter callbacks, we instead maintain a global lookup
// table that associates raw MuJoCo struct pointers back to the pointers to
// their corresponding wrappers.
template <typename Raw>
static enable_if_not_const_t<Raw, py::handle> MjWrapperLookup(Raw* ptr) {
  using LookupFnType = MjWrapper<Raw>* (Raw*);
  static LookupFnType* const lookup = []() -> LookupFnType* {
    py::gil_scoped_acquire gil;
    auto m = py::module_::import("mujoco._structs");
    pybind11::handle builtins(PyEval_GetBuiltins());
    if (!builtins.contains(MjWrapper<Raw>::kFromRawPointer)) {
      return nullptr;
    } else {
      try {
        return reinterpret_cast<LookupFnType*>(
            builtins[MjWrapper<Raw>::kFromRawPointer]
                .template cast<std::uintptr_t>());
      } catch (const py::cast_error&) {
        return nullptr;
      }
    }
  }();

  MjWrapper<Raw>* wrapper = nullptr;
  if (lookup) {
    wrapper = lookup(ptr);
  } else {
    {
      py::gil_scoped_acquire gil;
      PyErr_SetString(
          UnexpectedError::GetPyExc(),
          "_structs module did not register its raw pointer lookup functions");
    }
  }

  if (!wrapper) {
    {
      py::gil_scoped_acquire gil;
      PyErr_SetString(
          UnexpectedError::GetPyExc(),
          "cannot find the corresponding wrapper for the raw mjStruct");
    }
  }

  // Now we find the existing Python instance of our wrapper.
  // TODO(stunya): Figure out a way to do this without invoking py::detail.
  {
    py::gil_scoped_acquire gil;
    auto* type = py::detail::get_type_info(typeid(MjWrapper<Raw>));
    if (type) {
      py::object instance = py::reinterpret_steal<py::object>(
          py::detail::find_registered_python_instance(wrapper, type));
      if (!instance) {
        if (!PyErr_Occurred()) {
          PyErr_SetString(
              UnexpectedError::GetPyExc(),
              "cannot find the Python instance of the MjWrapper");
        }
      } else {
        return std::move(instance);
      }
    } else {
      if (!PyErr_Occurred()) {
        PyErr_SetString(
            UnexpectedError::GetPyExc(),
            "MjWrapper type isn't registered with pybind11");
      }
    }
  }

  EscapeWithPythonException();
}

template <typename Raw>
static const py::handle MjWrapperLookup(const Raw* ptr) {
  return MjWrapperLookup(const_cast<Raw*>(ptr));
}

// CallPyCallback takes ownership of py_callback: it will Py_XDECREF it before
// returning or escaping.  This is necessary because EscapeWithPythonException()
// calls mju_error which uses longjmp, bypassing C++ stack unwinding and any
// trailing Py_XDECREF in callers.
template <typename Return, typename... Args>
static Return
CallPyCallback(const char* name, PyObject* py_callback, Args... args) {
  {
    py::gil_scoped_acquire gil;
    if (!py_callback) {
      std::ostringstream msg;
      msg << "py_" << name << " is null";
      PyErr_SetString(UnexpectedError::GetPyExc(), msg.str().c_str());
    } else {
      py::handle callback(py_callback);
      try {
        if constexpr (std::is_void_v<Return>) {
          callback(args...);
          Py_XDECREF(py_callback);
          return;
        } else {
          auto result = callback(args...).template cast<Return>();
          Py_XDECREF(py_callback);
          return result;
        }
      } catch (py::error_already_set& e) {
        e.restore();
      } catch (const py::cast_error&) {
        std::ostringstream msg;
        msg << name << " callback did not return ";
        if constexpr (std::is_integral_v<Return>) {
          msg << "an integer";
        } else if constexpr (std::is_floating_point_v<Return>) {
          msg << "a number";
        } else {
          msg << "the correct type";
        }
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
      }
    }
    // Error path: DECREF before escaping (longjmp won't unwind the stack).
    Py_XDECREF(py_callback);
  }
  EscapeWithPythonException();
}

static PyObject* py_mju_user_warning = nullptr;
static void PyMjuUserWarning(const char* msg) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mju_user_warning;
    Py_XINCREF(cb);
  }
  // CallPyCallback takes ownership of cb (will XDECREF it).
  CallPyCallback<void>("mju_user_warning", cb, msg);
}

// We only support ctypes function pointers for these.
// The PyObject* are only here so that we can return the ctypes pointers back
// through the getters.
static PyObject* py_mju_user_malloc = nullptr;
static PyObject* py_mju_user_free = nullptr;

static PyObject* py_mjcb_passive = nullptr;
static void PyMjcbPassive(const raw::MjModel* m, raw::MjData* d) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_passive;
    Py_XINCREF(cb);
  }
  CallPyCallback<void>("mjcb_passive", cb,
                       MjWrapperLookup(m), MjWrapperLookup(d));
}

static PyObject* py_mjcb_control = nullptr;
static void PyMjcbControl(const raw::MjModel* m, raw::MjData* d) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_control;
    Py_XINCREF(cb);
  }
  CallPyCallback<void>("mjcb_control", cb,
                       MjWrapperLookup(m), MjWrapperLookup(d));
}

static PyObject* py_mjcb_contactfilter = nullptr;
static int PyMjcbContactfilter(
    const raw::MjModel* m, raw::MjData* d, int geom1, int geom2) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_contactfilter;
    Py_XINCREF(cb);
  }
  return CallPyCallback<int>("mjcb_contactfilter", cb,
                             MjWrapperLookup(m), MjWrapperLookup(d),
                             geom1, geom2);
}

static PyObject* py_mjcb_sensor = nullptr;
static void
PyMjcbSensor(const raw::MjModel* m, raw::MjData* d, int stage) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_sensor;
    Py_XINCREF(cb);
  }
  CallPyCallback<void>("mjcb_sensor", cb,
                       MjWrapperLookup(m), MjWrapperLookup(d), stage);
}

static PyObject* py_mjcb_time = nullptr;
static mjtNum PyMjcbTime() {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_time;
    Py_XINCREF(cb);
  }
  return CallPyCallback<mjtNum>("mjcb_time", cb);
}

static PyObject* py_mjcb_act_dyn = nullptr;
static mjtNum
PyMjcbActDyn(const raw::MjModel* m, const raw::MjData* d, int id) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_act_dyn;
    Py_XINCREF(cb);
  }
  return CallPyCallback<mjtNum>("mjcb_act_dyn", cb,
                                MjWrapperLookup(m),
                                MjWrapperLookup(d), id);
}

static PyObject* py_mjcb_act_gain = nullptr;
static mjtNum
PyMjcbActGain(const raw::MjModel* m, const raw::MjData* d, int id) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_act_gain;
    Py_XINCREF(cb);
  }
  return CallPyCallback<mjtNum>("mjcb_act_gain", cb,
                                MjWrapperLookup(m),
                                MjWrapperLookup(d), id);
}

static PyObject* py_mjcb_act_bias = nullptr;
static mjtNum
PyMjcbActBias(const raw::MjModel* m, const raw::MjData* d, int id) {
  PyObject* cb;
  {
    py::gil_scoped_acquire gil;
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    cb = py_mjcb_act_bias;
    Py_XINCREF(cb);
  }
  return CallPyCallback<mjtNum>("mjcb_act_bias", cb,
                                MjWrapperLookup(m),
                                MjWrapperLookup(d), id);
}

// If the Python object is a ctypes function pointer, returns the corresponding
// C function pointer. Otherwise, returns a null pointer.
template <typename FuncPtr>
static FuncPtr GetCFuncPtr(py::handle h) {
  struct CTypes { PyObject* cfuncptr; PyObject* cast; PyObject* c_void_p; };
  static const CTypes ctypes = []() -> CTypes {
    try {
      auto m = py::module_::import("ctypes");
      PyObject* cfuncptr = m.attr("_CFuncPtr").ptr();
      PyObject* cast = m.attr("cast").ptr();
      PyObject* c_void_p = m.attr("c_void_p").ptr();
      Py_XINCREF(cfuncptr);
      Py_XINCREF(cast);
      Py_XINCREF(c_void_p);
      return {cfuncptr, cast, c_void_p};
    } catch (const py::error_already_set&) {
      return {nullptr, nullptr, nullptr};
    }
  }();

  if (!ctypes.cfuncptr) {
    throw UnexpectedError("cannot find `ctypes._CFuncPtr`");
  }
  const int is_cfuncptr = PyObject_IsInstance(h.ptr(), ctypes.cfuncptr);
  if (is_cfuncptr == -1) {
    throw py::error_already_set();
  } else if (is_cfuncptr) {
    if (!ctypes.cast) {
      throw UnexpectedError("cannot find `ctypes.cast`");
    }
    if (!ctypes.c_void_p) {
      throw UnexpectedError("cannot find `ctypes.c_void_p`");
    }
    const uintptr_t func_address =
        py::handle(ctypes.cast)(h, py::handle(ctypes.c_void_p))
            .attr("value")
            .template cast<std::uintptr_t>();
    return reinterpret_cast<FuncPtr>(func_address);
  } else {
    return nullptr;
  }
}

static bool IsCallable(py::handle h) {
  static PyObject* const is_callable = []() -> PyObject* {
    try{
      PyObject* o = py::eval("callable").ptr();
      Py_XINCREF(o);
      return o;
    } catch (const py::error_already_set&) {
      return nullptr;
    }
  }();

  if (!is_callable) {
    throw UnexpectedError("cannot find `callable`");
  }

  return py::handle(is_callable)(h).cast<bool>();
}

template <typename CFuncPtr>
void SetCallback(py::handle h, CFuncPtr py_trampoline,
                 PyObject** py_callback, CFuncPtr* mj_callback) {
  CFuncPtr cfuncptr = GetCFuncPtr<CFuncPtr>(h);
  PyObject* old = nullptr;
  if (h.is_none()) {
    {
      MutexLockIfGilDisabled lock(GetCallbackMutex());
      old = *py_callback;
      *py_callback = nullptr;
      *mj_callback = nullptr;
    }
  } else if (cfuncptr) {
    Py_INCREF(h.ptr());
    {
      MutexLockIfGilDisabled lock(GetCallbackMutex());
      old = *py_callback;
      *py_callback = h.ptr();
      *mj_callback = cfuncptr;
    }
  } else if (IsCallable(h)) {
    Py_INCREF(h.ptr());
    {
      MutexLockIfGilDisabled lock(GetCallbackMutex());
      old = *py_callback;
      *py_callback = h.ptr();
      *mj_callback = py_trampoline;
    }
  } else {
    throw py::type_error("callback is not an Optional[Callable]");
  }
  // XDECREF outside mutex: __del__ may re-enter callback setters/getters.
  Py_XDECREF(old);
}

// Default timer callback (seconds).
static mjtNum DefaultTimer() {
  using Clock = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<mjtNum>;
  static const Clock::time_point tm_start = Clock::now();
  return Seconds(Clock::now() - tm_start).count();
}

static void InstallDefaultTimer() {
  // Warm up static tm_start during single-threaded module initialization to
  // avoid later thread contention on the static guard mutex.
  static_cast<void>(DefaultTimer());

  PyObject* old = nullptr;
  {
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    old = std::exchange(py_mjcb_time, nullptr);
    ::mjcb_time = DefaultTimer;
  }
  Py_XDECREF(old);
}

py::object GetCallback(PyObject** py_callback) {
  MutexLockIfGilDisabled lock(GetCallbackMutex());
  if (!*py_callback) {
    return py::none();
  }
  return py::reinterpret_borrow<py::object>(*py_callback);
}

PYBIND11_MODULE(_callbacks, pymodule, pybind11::mod_gil_not_used()) {
  InstallDefaultTimer();

  // Setters
  pymodule.def("set_mju_user_warning", [](py::handle h) {
    SetCallback(h, PyMjuUserWarning, &py_mju_user_warning,
               &::mju_user_warning);
  });
  pymodule.def("set_mju_user_malloc", [](py::handle h) {
    PyObject* old = nullptr;
    if (h.is_none()) {
      {
        MutexLockIfGilDisabled lock(GetCallbackMutex());
        old = py_mju_user_malloc;
        py_mju_user_malloc = nullptr;
        ::mju_user_malloc = nullptr;
      }
    } else {
      auto* cfuncptr = GetCFuncPtr<decltype(::mju_user_malloc)>(h);
      if (!cfuncptr) {
        throw py::type_error("mju_user_malloc must be a C function pointer");
      }
      Py_XINCREF(h.ptr());
      {
        MutexLockIfGilDisabled lock(GetCallbackMutex());
        old = py_mju_user_malloc;
        py_mju_user_malloc = h.ptr();
        ::mju_user_malloc = cfuncptr;
      }
    }
    Py_XDECREF(old);
  });
  pymodule.def("set_mju_user_free", [](py::handle h) {
    PyObject* old = nullptr;
    if (h.is_none()) {
      {
        MutexLockIfGilDisabled lock(GetCallbackMutex());
        old = py_mju_user_free;
        py_mju_user_free = nullptr;
        ::mju_user_free = nullptr;
      }
    } else {
      auto* cfuncptr = GetCFuncPtr<decltype(::mju_user_free)>(h);
      if (!cfuncptr) {
        throw py::type_error("mju_user_free must be a C function pointer");
      }
      Py_XINCREF(h.ptr());
      {
        MutexLockIfGilDisabled lock(GetCallbackMutex());
        old = py_mju_user_free;
        py_mju_user_free = h.ptr();
        ::mju_user_free = cfuncptr;
      }
    }
    Py_XDECREF(old);
  });
  pymodule.def("set_mjcb_passive", [](py::handle h) {
    SetCallback(h, PyMjcbPassive, &py_mjcb_passive, &::mjcb_passive);
  });
  pymodule.def("set_mjcb_control", [](py::handle h) {
    SetCallback(h, PyMjcbControl, &py_mjcb_control, &::mjcb_control);
  });
  pymodule.def("set_mjcb_contactfilter", [](py::handle h) {
    SetCallback(h, PyMjcbContactfilter,
                &py_mjcb_contactfilter, &::mjcb_contactfilter);
  });
  pymodule.def("set_mjcb_sensor", [](py::handle h) {
    SetCallback(h, PyMjcbSensor, &py_mjcb_sensor, &::mjcb_sensor);
  });
  pymodule.def("set_mjcb_time", [](py::handle h) {
    // Passing None restores DefaultTimer (rather than setting ::mjcb_time
    // to null). This preserves the idiomatic Python save/restore pattern
    // (get_mjcb_time() returning None -> set_mjcb_time(None)), and matches
    // historical bindings behavior where MjData construction ensured a timer
    // was present.
    if (h.is_none()) {
      InstallDefaultTimer();
    } else {
      SetCallback(h, PyMjcbTime, &py_mjcb_time, &::mjcb_time);
    }
  });
  pymodule.def("set_mjcb_act_dyn", [](py::handle h) {
    SetCallback(h, PyMjcbActDyn, &py_mjcb_act_dyn, &::mjcb_act_dyn);
  });
  pymodule.def("set_mjcb_act_gain", [](py::handle h) {
    SetCallback(h, PyMjcbActGain, &py_mjcb_act_gain, &::mjcb_act_gain);
  });
  pymodule.def("set_mjcb_act_bias", [](py::handle h) {
    SetCallback(h, PyMjcbActBias, &py_mjcb_act_bias, &::mjcb_act_bias);
  });

  // Getters
  pymodule.def("get_mju_user_warning", []() {
    return GetCallback(&py_mju_user_warning);
  });
  pymodule.def("get_mju_user_malloc", []() {
    return GetCallback(&py_mju_user_malloc);
  });
  pymodule.def("get_mju_user_free", []() {
    return GetCallback(&py_mju_user_free);
  });
  pymodule.def("get_mjcb_passive", []() {
    return GetCallback(&py_mjcb_passive);
  });
  pymodule.def("get_mjcb_control", []() {
    return GetCallback(&py_mjcb_control);
  });
  pymodule.def("get_mjcb_contactfilter", []() {
    return GetCallback(&py_mjcb_contactfilter);
  });
  pymodule.def("get_mjcb_sensor", []() {
    return GetCallback(&py_mjcb_sensor);
  });
  pymodule.def("get_mjcb_time", []() {
    return GetCallback(&py_mjcb_time);
  });
  pymodule.def("get_mjcb_act_dyn", []() {
    return GetCallback(&py_mjcb_act_dyn);
  });
  pymodule.def("get_mjcb_act_gain", []() {
    return GetCallback(&py_mjcb_act_gain);
  });
  pymodule.def("get_mjcb_act_bias", []() {
    return GetCallback(&py_mjcb_act_bias);
  });
}  // PYBIND11_MODULE

}  // namespace
}  // namespace mujoco::python
