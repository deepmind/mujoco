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

#ifndef MUJOCO_PYTHON_ERRORS_H_
#define MUJOCO_PYTHON_ERRORS_H_

#include <array>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <mujoco/mjtype.h>
#include "private.h"
#include "util/crossplatform.h"
#include "util/func_wrap.h"
#include <pybind11/pybind11.h>

// When building for Linux and statically linking against a "hermetic" libc++abi
// (i.e. where libc++/libc++abi symbols all have "hidden" visibility), exception
// types do not propagate correctly across shared library boundaries.
//
// This occurs even when PYBIND11_EXPORT_EXCEPTION is used, or when the virtual
// destructor is defined in errors.cc rather than than defined inline in the
// class definition (following the FAQ comment about "key functions" on
// https://libcxxabi.llvm.org/). For some reason, the type_info address is
// different across different shared objects (DSO), even when the symbols for
// typeinfo, type name, and vtable are left undefined in all but one DSO.
// Since libc++abi matches exception types by comparing type_info addresses,
// this breaks exception handling across DSO boundaries.
//
// Instead, we manually create and store exception types in PyEval_GetBuiltins,
// then effectively register separate pybind11 exception translators in each
// module that all translate to the same exception type (which is what's
// happening with pybind11's built-in exceptions under this setup).
//
// Effectively, we are doing almost the same thing as what a pybind11 is doing
// with its "internals" struct, but we store the exception types directly in
// the builtin context rather than in a PyCapsule.
namespace mujoco::python {
namespace _impl {
template <typename T>
class ErrorBase : public pybind11::builtin_exception {
 public:
  virtual ~ErrorBase() = default;

  static PyObject* GetPyExc() {
    static PyObject* const e = []() {
      pybind11::gil_scoped_acquire gil;
      std::string unique_identifier = "__MUJOCO_ERROR_";
      unique_identifier += T::kName;
      pybind11::str py_builtin_identifier(unique_identifier);

      // We can end up here while handling another Python exception.
      // Temporarily clear the Python error indicator since we need to interact
      // with the interpreter.
      struct PyErrCache {
        PyErrCache() { PyErr_Fetch(&type, &value, &traceback); }
        ~PyErrCache() { if (type) PyErr_Restore(type, value, traceback); }
        PyObject* type;
        PyObject* value;
        PyObject* traceback;
      };
      PyErrCache err_cache;

      pybind11::handle builtins(PyEval_GetBuiltins());
      if (!builtins.contains(py_builtin_identifier)) {
        std::string full_name = "mujoco.";
        full_name += T::kName;
        auto ret =
            PyErr_NewException(full_name.c_str(), PyExc_Exception, nullptr);
        builtins[py_builtin_identifier] = pybind11::handle(ret);
        return ret;
      } else {
        return builtins[py_builtin_identifier].ptr();
      }
    }();
    return e;
  }

  void set_error() const override { PyErr_SetString(T::GetPyExc(), what()); }

 protected:
  using builtin_exception::builtin_exception;
};

// We shouldn't throw a C++ exception from a function that's a callback
// from C. (Usually it would work, but it would be undefined behavior since
// there'd be no guarantee that C++ can unwind the stack correctly through the
// C functions. On Windows, longjmp actually uses the same mechanism as
// C++ exceptions.)
// Instead, we call setjmp before entering MuJoCo, and do a longjmp from
// mju_user_error back to C++ to throw an exception.
static thread_local std::jmp_buf* tls_error_jmp_buf = nullptr;
static thread_local std::array<char, 2048> mju_error_msg{0};

// The handler to forward non-error messages to.  Set by WrapFunc before each
// call into MuJoCo C code, pointing to either the previously installed TLS
// handler or the active global handler.
static thread_local mjfLogHandler mju_forward_handler = nullptr;

static inline void MjErrorHandler(const mjLogMessage* msg) {
  if (msg->level != mjLOG_ERROR) {
    // Forward warnings, info, and debug messages to the previous handler so
    // that legacy mju_user_warning callbacks and console output continue to
    // work.
    if (mju_forward_handler != nullptr) {
      mju_forward_handler(msg);
    }
    return;
  }
  if (msg->func != nullptr) {
    std::snprintf(mju_error_msg.data(), mju_error_msg.size(), "%s: %s",
                  msg->func, msg->subject);
  } else {
    std::snprintf(mju_error_msg.data(), mju_error_msg.size(), "%s",
                  msg->subject);
  }
  std::longjmp(*tls_error_jmp_buf, 1);
}

template <typename InterceptAsType>
struct MjErrorIntercepter {
  template <typename Return, typename... Args, typename Callable>
  MUJOCO_ALWAYS_INLINE
  static constexpr auto WrapFunc(Callable&& callable) {
#if defined(__GNUC__) && !defined(__clang__)
    // GCC can't inline functions that call setjmp
    return [callable](Args... args) mutable {
#else
    return [callable](Args... args) MUJOCO_ALWAYS_INLINE_LAMBDA_MUTABLE {
#endif
      mjfLogHandler prev_handler = _mjPRIVATE_setTlsLogHandler(&MjErrorHandler);

      // Determine the handler to forward non-error messages to.
      // If there was a previously installed TLS handler, forward to it.
      // Otherwise, probe the global handler via mju_setLogHandler.
      mjfLogHandler old_forward = mju_forward_handler;
      if (prev_handler != nullptr) {
        mju_forward_handler = prev_handler;
      } else {
        mju_forward_handler = _mjPRIVATE_getGlobalLogHandler();
      }

      // DON'T MIX RAII WITH SETJMP!
      // From https://en.cppreference.com/w/cpp/utility/program/longjmp:
      //   If replacing of std::longjmp with `throw` and setjmp with `catch`
      //   would execute a non-trivial destructor for any automatic object, the
      //   behavior of such std::longjmp is undefined.
      std::jmp_buf local_jmp_buf;
      std::jmp_buf* prev_jmp_buf = tls_error_jmp_buf;
      tls_error_jmp_buf = &local_jmp_buf;
      if (setjmp(local_jmp_buf) == 0) {
        if constexpr (std::is_void_v<decltype(callable(args...))>) {
          callable(args...);
          tls_error_jmp_buf = prev_jmp_buf;
          mju_forward_handler = old_forward;
          _mjPRIVATE_setTlsLogHandler(prev_handler);
        } else {
          auto ret = callable(args...);
          static_assert(std::is_trivially_destructible_v<decltype(ret)>);
          tls_error_jmp_buf = prev_jmp_buf;
          mju_forward_handler = old_forward;
          _mjPRIVATE_setTlsLogHandler(prev_handler);
          return ret;
        }
      } else {
        // This branch is entered via a longjmp back from our mju_error handler.
        tls_error_jmp_buf = prev_jmp_buf;
        mju_forward_handler = old_forward;
        _mjPRIVATE_setTlsLogHandler(prev_handler);
        {
          // Check if a Python callback has thrown an exception.
          // We cannot use a py::gil_scoped_acquire here: on Windows its
          // destructor isn't triggered before pybind returns control to the
          // interpreter.
          auto gil = PyGILState_Ensure();
          if (PyErr_Occurred()) {
            // We must hold the GIL when we create the py::error_already_set,
            // since its constructor calls PyErr_Fetch.
            pybind11::error_already_set err;

            // But the GIL must be released before we throw, otherwise a
            // deadlock ensues!
            PyGILState_Release(gil);
            throw err;
          }
          PyGILState_Release(gil);
        }
        throw InterceptAsType(std::string(mju_error_msg.data()));
      }
    };
  }
};
}  // namespace _impl

class FatalError : public _impl::ErrorBase<FatalError> {
 public:
  static constexpr char kName[] = "FatalError";
  using ErrorBase<FatalError>::ErrorBase;
  virtual ~FatalError() = default;
};

class UnexpectedError : public _impl::ErrorBase<UnexpectedError> {
 public:
  static constexpr char kName[] = "UnexpectedError";
  UnexpectedError(const std::string& msg)
      : ErrorBase(msg +
                  " (this error not expected to ever occur,"
                  " please report it to MuJoCo developers!)") {}
  virtual ~UnexpectedError() = default;
};

template <typename Callable>
MUJOCO_ALWAYS_INLINE
static constexpr auto InterceptMjErrors(Callable&& callable) {
  return util::_impl::WrapFunc<_impl::MjErrorIntercepter<FatalError>>(
      std::forward<Callable>(callable));
}
}  // namespace mujoco::python

#endif  // MUJOCO_PYTHON_ERRORS_H_
