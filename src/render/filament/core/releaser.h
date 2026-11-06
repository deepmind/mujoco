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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_RELEASER_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_RELEASER_H_

#include <cstddef>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>

namespace mujoco {

// Utility for deferring the release of data with Filament's buffer descriptor.
//
// MuJoCo's mesh and texture classes provide a callback mechanism for releasing
// user-owned data. Filament's buffer descriptors have their own mechanism as
// well. This class helps bridge the two.
//
// There are three common patterns:
// - We are unable to create the Filament buffers, so we need to release the
//   user data immediately.
// - We need to defer the release of the user data until Filament has finished
//   reading the data.
// - In addition to user data, we may generate additional data (e.g. orientation
//   buffers) that also need to be released at the same time as the user data.
//
// To use this class, an instance of it should be created on the stack and
// user data should be added to it. Once we are ready to populate a Filament
// buffer descriptor, we can call `Acquire()` to take (shared) ownership of the
// registered callbacks and data pointers. The return value of `Acquire()`,
// along with the `Releaser::Release()`, can then be passed to the buffer
// descriptor. Users can continue to add additional callbacks and data
// pointers to the original Releaser instance, even after `Acquire()` has been
// called. `Acquire()` can be called as often as needed.
//
// This class works by storing the callbacks and data pointers in a shared
// reference-counted object. Each call to `Acquire()` makes a new reference to
// the shared object. Once all instances have been released (either by the
// initial object falling out of scope or by Filament calling all the buffer
// descriptor callbacks), the registered callbacks will be called and the
// shared object will be deleted.
class Releaser {
 public:
  Releaser() : callbacks_(new Callbacks) {}
  ~Releaser() { callbacks_->ReleaseRef(); }

  Releaser(const Releaser& other) = delete;
  Releaser& operator=(const Releaser& other) = delete;

  // Registers a callback and user data pointer to be called at a later time.
  void Add(mjrfCallback release, void* user_data) {
    callbacks_->Add(release, user_data);
  }

  // Allocates and returns an array of `num` elements of type `T`. The memory
  // will be automatically released along with all other data registered with
  // this Releaser.
  template <typename T = std::byte>
  T* Alloc(size_t num) {
    T* ptr = new T[num];
    mjrfCallback callback = +[](void* buffer) {
      delete[] reinterpret_cast<T*>(buffer);
    };
    Add(callback, ptr);
    return ptr;
  }

  // Takes shared ownership of the registered callbacks and data pointers.
  // Release() must be called on the returned pointer, otherwise the callbacks
  // will never be called. This function can be called multiple times. Callers
  // can continue to Add callbacks to this object even after this function has
  // been called.
  void* Acquire() {
    return callbacks_->AddRef();
  }

  // Releases ownership of the value returned by Acquire. The signature of this
  // function matches the signature of the Filament buffer descriptor callbacks
  // so it can be passed directly to the buffer descriptor.
  static void Release(void* buffer, size_t size, void* user) {
    static_cast<Callbacks*>(user)->ReleaseRef();
  }

 private:
  struct Callbacks {
    ~Callbacks() {
      for (int i = num_callbacks_ - 1; i >= 0; --i) {
        callbacks_[i](user_data_[i]);
      }
    }

    void Add(mjrfCallback release, void* user_data) {
      if (release) {
        if (num_callbacks_ >= kMaxCallbacks) {
          mju_error("Releaser::Callbacks::Add: too many callbacks");
          return;
        }
        callbacks_[num_callbacks_] = release;
        user_data_[num_callbacks_] = user_data;
        num_callbacks_++;
      }
    }

    void* AddRef() {
      ++ref_count_;
      return this;
    }

    void ReleaseRef() {
      if (--ref_count_ == 0) {
        delete this;
      }
    }

    static constexpr int kMaxCallbacks = 8;
    mjrfCallback callbacks_[kMaxCallbacks];
    void* user_data_[kMaxCallbacks];
    int num_callbacks_ = 0;
    // All calls to ReleaseRef() and AddRef() are made from the same thread so
    // a simple ref count will suffice.
    int ref_count_ = 1;
  };
  Callbacks* callbacks_;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_RELEASER_H_
