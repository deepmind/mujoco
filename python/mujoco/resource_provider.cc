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

#include "resource_provider.h"

#include <Python.h>

#include <cctype>
#include <climits>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "gil.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace mujoco::python {
namespace {

namespace py = ::pybind11;

// A Python object representing an open MuJoCo resource. MuJoCo passes a raw
// mjResource* to the provider callbacks; this class exposes the parts of it
// that are meaningful to a Python provider and keeps user data and the last
// read buffer alive for as long as the resource is open.
class PyResource {
 public:
  explicit PyResource(mjResource* resource) : resource_(resource) {}

  std::string GetName() const {
    return resource_ && resource_->name ? resource_->name : "";
  }

  std::string GetArgs() const {
    return resource_ && resource_->args ? resource_->args : "";
  }

  std::string GetTimestamp() const {
    return resource_ ? resource_->timestamp : "";
  }

  void SetTimestamp(const std::string& timestamp) {
    if (!resource_) { return; }
    if (timestamp.size() >= sizeof(resource_->timestamp)) {
      throw py::value_error("timestamp is too long");
    }
    std::memcpy(resource_->timestamp, timestamp.c_str(), timestamp.size() + 1);
  }

  py::object data = py::none();

 private:
  mjResource* resource_ = nullptr;
};

// Callbacks of a provider, as registered with MuJoCo. The callbacks are kept
// as raw Python references for the lifetime of the process, mirroring the
// global provider table in the C API. Registered providers are never freed.
struct RegisteredProvider {
  std::string prefix;
  PyObject* open = nullptr;
  PyObject* read = nullptr;
  PyObject* close = nullptr;
  PyObject* mount = nullptr;
  PyObject* unmount = nullptr;
  PyObject* modified = nullptr;
  PyObject* write = nullptr;
  mjpResourceProvider provider;
};

// The Python-visible provider. Unlike RegisteredProvider, this object is owned
// by Python and can be modified freely until it is registered.
struct PyProvider {
  std::string prefix;
  py::object open = py::none();
  py::object read = py::none();
  py::object close = py::none();
  py::object mount = py::none();
  py::object unmount = py::none();
  py::object modified = py::none();
  py::object write = py::none();
};

// State kept for each open resource, keyed by the mjResource pointer.
struct ResourceHolder {
  py::object object;
  Py_buffer buffer = {};
  bool has_buffer = false;

  ~ResourceHolder() { ReleaseBuffer(); }

  void ReleaseBuffer() {
    if (has_buffer) {
      PyBuffer_Release(&buffer);
      buffer = {};
      has_buffer = false;
    }
  }
};

std::vector<RegisteredProvider*>& RegisteredProviders() {
  static auto* providers = new std::vector<RegisteredProvider*>();
  return *providers;
}

std::unordered_map<mjResource*, std::unique_ptr<ResourceHolder>>& Holders() {
  static auto* holders =
      new std::unordered_map<mjResource*, std::unique_ptr<ResourceHolder>>();
  return *holders;
}

RegisteredProvider* ProviderOf(mjResource* resource) {
  if (!resource || !resource->provider) { return nullptr; }
  return static_cast<RegisteredProvider*>(resource->provider->data);
}

// Looks up the holder of an open resource, creating a temporary one if needed.
// A resource opened by PyResourceProvider always has a holder; the fallback
// covers resources opened through other paths.
std::unique_ptr<ResourceHolder> MakeHolder(mjResource* resource,
                                           bool* is_temporary) {
  {
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    auto it = Holders().find(resource);
    if (it != Holders().end()) {
      *is_temporary = false;
      return nullptr;
    }
  }
  auto holder = std::make_unique<ResourceHolder>();
  holder->object = py::cast(new PyResource(resource),
                            py::return_value_policy::take_ownership);
  *is_temporary = true;
  return holder;
}

ResourceHolder* FindHolder(mjResource* resource) {
  MutexLockIfGilDisabled lock(GetCallbackMutex());
  auto it = Holders().find(resource);
  return it == Holders().end() ? nullptr : it->second.get();
}

// Calls a Python callback, printing and clearing any exception. Returns false
// if the callback raised or did not return a result.
bool CallPython(const char* name, PyObject* callback, PyObject** result,
                const py::object& resource) {
  *result = PyObject_CallFunctionObjArgs(callback, resource.ptr(), nullptr);
  if (*result) { return true; }
  PyErr_Print();
  std::fprintf(stderr,
               "mujoco: error in resource provider %s callback, ignoring\n",
               name);
  return false;
}

int PyOpen(mjResource* resource) {
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(resource);
  if (!provider || !provider->open) { return 0; }

  auto holder = std::make_unique<ResourceHolder>();
  holder->object = py::cast(new PyResource(resource),
                            py::return_value_policy::take_ownership);

  PyObject* result = nullptr;
  if (!CallPython("open", provider->open, &result, holder->object)) {
    return 0;
  }
  int success = PyObject_IsTrue(result);
  Py_DECREF(result);
  if (success <= 0) {
    if (success < 0) { PyErr_Print(); }
    return 0;
  }

  {
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    Holders()[resource] = std::move(holder);
  }
  return 1;
}

int PyRead(mjResource* resource, const void** buffer) {
  *buffer = nullptr;
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(resource);
  if (!provider || !provider->read) { return -1; }

  bool is_temporary = false;
  auto temporary = MakeHolder(resource, &is_temporary);
  ResourceHolder* holder =
      is_temporary ? temporary.get() : FindHolder(resource);
  if (!holder) { return -1; }
  holder->ReleaseBuffer();

  PyObject* result = nullptr;
  if (!CallPython("read", provider->read, &result, holder->object)) {
    return -1;
  }

  int nbytes = -1;
  if (PyObject_GetBuffer(result, &holder->buffer, PyBUF_SIMPLE) == 0) {
    if (holder->buffer.len <= INT_MAX) {
      holder->has_buffer = true;
      *buffer = holder->buffer.buf;
      nbytes = static_cast<int>(holder->buffer.len);
    } else {
      PyBuffer_Release(&holder->buffer);
      holder->buffer = {};
      PyErr_SetString(PyExc_OverflowError,
                      "resource is too large for a MuJoCo read callback");
      PyErr_Print();
    }
  } else {
    PyErr_Print();
  }
  Py_DECREF(result);
  return nbytes;
}

void PyClose(mjResource* resource) {
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(resource);
  if (!provider) { return; }

  bool is_temporary = false;
  auto temporary = MakeHolder(resource, &is_temporary);
  ResourceHolder* holder =
      is_temporary ? temporary.get() : FindHolder(resource);
  if (holder && provider->close) {
    PyObject* result = nullptr;
    if (CallPython("close", provider->close, &result, holder->object)) {
      Py_DECREF(result);
    }
  }

  if (!is_temporary) {
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    auto it = Holders().find(resource);
    if (it != Holders().end()) {
      it->second->ReleaseBuffer();
      Holders().erase(it);
    }
  }
}

int PyMount(mjResource* resource) {
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(resource);
  if (!provider) { return 0; }
  if (!provider->mount) { return 1; }

  py::object object = py::cast(
      new PyResource(resource), py::return_value_policy::take_ownership);
  PyObject* result = nullptr;
  if (!CallPython("mount", provider->mount, &result, object)) { return 0; }
  int success = PyObject_IsTrue(result);
  Py_DECREF(result);
  if (success < 0) { PyErr_Print(); }
  return success > 0;
}

int PyUnmount(mjResource* resource) {
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(resource);
  if (!provider) { return 0; }
  if (!provider->unmount) { return 1; }

  py::object object = py::cast(
      new PyResource(resource), py::return_value_policy::take_ownership);
  PyObject* result = nullptr;
  if (!CallPython("unmount", provider->unmount, &result, object)) {
    return 0;
  }
  int success = PyObject_IsTrue(result);
  Py_DECREF(result);
  if (success < 0) { PyErr_Print(); }
  return success > 0;
}

int PyModified(const mjResource* resource, const char* timestamp) {
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(const_cast<mjResource*>(resource));
  if (!provider || !provider->modified) { return 1; }

  py::object object = py::cast(
      new PyResource(const_cast<mjResource*>(resource)),
      py::return_value_policy::take_ownership);
  py::object stamp = timestamp ? py::str(timestamp) : py::str("");
  PyObject* result = PyObject_CallFunctionObjArgs(
      provider->modified, object.ptr(), stamp.ptr(), nullptr);
  if (!result) {
    PyErr_Print();
    return 1;
  }
  long value = PyLong_AsLong(result);
  Py_DECREF(result);
  if (value == -1 && PyErr_Occurred()) {
    PyErr_Print();
    return 1;
  }
  return value < 0 ? -1 : (value > 0 ? 1 : 0);
}

mjtSize PyWrite(mjResource* resource, const void* buffer, mjtSize nbytes) {
  py::gil_scoped_acquire gil;
  RegisteredProvider* provider = ProviderOf(resource);
  if (!provider || !provider->write) { return -1; }

  bool is_temporary = false;
  auto temporary = MakeHolder(resource, &is_temporary);
  ResourceHolder* holder =
      is_temporary ? temporary.get() : FindHolder(resource);
  if (!holder) { return -1; }

  py::bytes data(static_cast<const char*>(buffer),
                 static_cast<py::ssize_t>(nbytes));
  PyObject* result = PyObject_CallFunctionObjArgs(
      provider->write, holder->object.ptr(), data.ptr(), nullptr);
  if (!result) {
    PyErr_Print();
    return -1;
  }
  long long written = PyLong_AsLongLong(result);
  Py_DECREF(result);
  if (written == -1 && PyErr_Occurred()) {
    PyErr_Print();
    return -1;
  }
  return static_cast<mjtSize>(written);
}

bool ValidPrefix(const std::string& prefix) {
  if (prefix.empty() ||
      !std::isalpha(static_cast<unsigned char>(prefix[0]))) {
    return false;
  }
  for (char ch : prefix) {
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '+' &&
        ch != '.' && ch != '-') {
      return false;
    }
  }
  return true;
}

// Registers a provider and returns its slot. The registered callbacks are
// borrowed references, kept alive for the lifetime of the process.
int RegisterProvider(PyProvider provider) {
  if (!ValidPrefix(provider.prefix)) {
    throw py::value_error(
        "provider prefix must be a valid URI scheme, e.g. 'pyasset'");
  }
  if (provider.open.is_none() || provider.read.is_none() ||
      provider.close.is_none()) {
    throw py::value_error(
        "provider must define open, read and close callbacks");
  }
  // MuJoCo treats a repeated prefix with different callbacks as a fatal error,
  // so reject it before calling the C API.
  if (::mjp_findProvider((provider.prefix + ":").c_str())) {
    throw py::value_error(
        "a resource provider with prefix '" + provider.prefix +
        "' is already registered");
  }

  auto* registered = new RegisteredProvider();
  registered->prefix = provider.prefix;
  registered->open = provider.open.inc_ref().ptr();
  registered->read = provider.read.inc_ref().ptr();
  registered->close = provider.close.inc_ref().ptr();
  registered->mount = provider.mount.is_none()
                          ? nullptr
                          : provider.mount.inc_ref().ptr();
  registered->unmount = provider.unmount.is_none()
                            ? nullptr
                            : provider.unmount.inc_ref().ptr();
  registered->modified = provider.modified.is_none()
                             ? nullptr
                             : provider.modified.inc_ref().ptr();
  registered->write = provider.write.is_none()
                          ? nullptr
                          : provider.write.inc_ref().ptr();

  registered->provider = {};
  registered->provider.prefix = registered->prefix.c_str();
  registered->provider.open = &PyOpen;
  registered->provider.read = &PyRead;
  registered->provider.close = &PyClose;
  registered->provider.mount = registered->mount ? &PyMount : nullptr;
  registered->provider.unmount = registered->unmount ? &PyUnmount : nullptr;
  registered->provider.modified =
      registered->modified ? &PyModified : nullptr;
  registered->provider.write = registered->write ? &PyWrite : nullptr;
  registered->provider.data = registered;

  int slot = ::mjp_registerProvider(&registered->provider);
  if (slot <= 0) {
    delete registered;
    throw py::value_error(
        "could not register provider, the prefix must be a valid URI scheme "
        "and must not already be registered");
  }

  {
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    RegisteredProviders().push_back(registered);
  }
  return slot;
}

void DefaultProvider(PyProvider& provider) {
  provider.prefix.clear();
  provider.open = py::none();
  provider.read = py::none();
  provider.close = py::none();
  provider.mount = py::none();
  provider.unmount = py::none();
  provider.modified = py::none();
  provider.write = py::none();
}

py::object FindProvider(const std::string& name) {
  const mjpResourceProvider* found = ::mjp_findProvider(name.c_str());
  if (!found) { return py::none(); }

  RegisteredProvider* registered =
      static_cast<RegisteredProvider*>(found->data);
  bool known = false;
  {
    MutexLockIfGilDisabled lock(GetCallbackMutex());
    for (const RegisteredProvider* provider : RegisteredProviders()) {
      if (provider == registered) {
        known = true;
        break;
      }
    }
  }
  if (!known) { return py::none(); }

  PyProvider provider;
  provider.prefix = registered->prefix;
  provider.open = py::reinterpret_borrow<py::object>(registered->open);
  provider.read = py::reinterpret_borrow<py::object>(registered->read);
  provider.close = py::reinterpret_borrow<py::object>(registered->close);
  provider.mount = registered->mount
                       ? py::reinterpret_borrow<py::object>(registered->mount)
                       : py::none();
  provider.unmount =
      registered->unmount
          ? py::reinterpret_borrow<py::object>(registered->unmount)
          : py::none();
  provider.modified =
      registered->modified
          ? py::reinterpret_borrow<py::object>(registered->modified)
          : py::none();
  provider.write = registered->write
                       ? py::reinterpret_borrow<py::object>(registered->write)
                       : py::none();
  return py::cast(provider);
}

}  // namespace

void InitResourceProvider(py::module_& m) {
  py::class_<PyResource>(m, "MjResource",
                         "Handle to a resource opened by a resource provider.")
      .def_property_readonly("name", &PyResource::GetName,
                             "Name of the resource, including its prefix.")
      .def_property_readonly("args", &PyResource::GetArgs,
                             "Arguments of the resource, from its URI query.")
      .def_property("timestamp", &PyResource::GetTimestamp,
                    &PyResource::SetTimestamp,
                    "Timestamp of the resource, used by the modified callback.")
      .def_readwrite("data", &PyResource::data,
                     "Arbitrary Python data associated with this resource.");

  py::class_<PyProvider>(m, "mjpResourceProvider",
                         "A resource provider that serves assets to MuJoCo.")
      .def(py::init<>())
      .def_readwrite("prefix", &PyProvider::prefix,
                     "URI scheme served by this provider, without the colon.")
      .def_readwrite("open", &PyProvider::open,
                     "Called with an MjResource; return True on success.")
      .def_readwrite("read", &PyProvider::read,
                     "Called with an MjResource; return the resource bytes.")
      .def_readwrite("close", &PyProvider::close,
                     "Called with an MjResource when it is closed.")
      .def_readwrite("mount", &PyProvider::mount,
                     "Called with an MjResource when a prefix is mounted.")
      .def_readwrite("unmount", &PyProvider::unmount,
                     "Called with an MjResource when a prefix is unmounted.")
      .def_readwrite("modified", &PyProvider::modified,
                     "Called with an MjResource and timestamp; return 0 if "
                     "the resource matches the timestamp.")
      .def_readwrite("write", &PyProvider::write,
                     "Called with an MjResource and bytes; return the number "
                     "of bytes written.");

  m.def("mjp_defaultProvider", &DefaultProvider, py::arg("provider"),
        "Resets a resource provider to its default (empty) state.");
  m.def("mjp_registerProvider", &RegisterProvider, py::arg("provider"),
        "Registers a resource provider and returns its slot.");
  m.def("mjp_findProvider", &FindProvider, py::arg("resource_name"),
        "Returns the registered provider matching a resource name, or None.");
}

}  // namespace mujoco::python
