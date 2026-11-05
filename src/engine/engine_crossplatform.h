// Copyright 2021 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_CROSSPLATFORM_H_
#define MUJOCO_SRC_ENGINE_ENGINE_CROSSPLATFORM_H_

// IWYU pragma: begin_keep
#if !defined(__cplusplus)
  #include <stddef.h>
  #include <stdlib.h>
#else
  #include <cstddef>
  #include <cstdlib>
#endif
// IWYU pragma: end_keep

// Case-insensitive comparison functions.
#ifdef _WIN32
  #define strcasecmp _stricmp
  #define strncasecmp _strnicmp
#else  // assumes POSIX
  #include <strings.h>
#endif

// Environment variable handling.
#ifdef _WIN32
  #define setenv(name, value, overwrite) _putenv_s(name, value)
  #define unsetenv(name) _putenv_s(name, "")
#endif

// Switch-case fallthrough annotation.
#if defined(__cplusplus)
  #define mjFALLTHROUGH [[fallthrough]]
#elif defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 7)
  #define mjFALLTHROUGH __attribute__((fallthrough))
#else
  #define mjFALLTHROUGH ((void) 0)
#endif

// MSVC only provides max_align_t in C++.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__cplusplus)
  typedef long double mjtMaxAlign;
#else
  typedef max_align_t mjtMaxAlign;
#endif

// Branch prediction hints.
#if defined(__GNUC__)
  #define mjLIKELY(x) __builtin_expect(!!(x), 1)
  #define mjUNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define mjLIKELY(x) (x)
  #define mjUNLIKELY(x) (x)
#endif

// Force-inline function specifier.
#if defined(_MSC_VER) && !defined(__clang__)
  #define mjALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define mjALWAYS_INLINE inline __attribute__((always_inline))
#else
  #define mjALWAYS_INLINE inline
#endif

// Thread-local storage.
#if !defined(mjTHREADLOCAL)
  #ifdef _MSC_VER
    #define mjTHREADLOCAL __declspec(thread)
  #else
    #define mjTHREADLOCAL _Thread_local
  #endif
#endif

// Define ADDRESS_SANITIZER if implied by other macros.
#if !defined(ADDRESS_SANITIZER)
  #if defined(__SANITIZE_ADDRESS__)
    #define ADDRESS_SANITIZER
  #elif defined(__has_feature)
    #if __has_feature(address_sanitizer)
      #define ADDRESS_SANITIZER
    #endif
  #endif
#endif

#if defined(ADDRESS_SANITIZER) && !defined(_MSC_VER)
  #define mjUSEASAN
#endif

// Atomics helper for size_t.
#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
  #define mj_atomic_add_size_t(ptr, val) \
      (size_t)_InterlockedExchangeAdd64((__int64 volatile*)(ptr), (__int64)(val))
#else
  #define mj_atomic_add_size_t(ptr, val) \
      __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)
#endif

// Atomics helpers for mjtBool (1-byte _Bool) with acquire/release semantics.
#if defined(_MSC_VER) && !defined(__clang__)
  #define mj_atomic_load_bool(ptr) \
      (mjtBool) _InterlockedCompareExchange8((volatile char*)(ptr), 0, 0)
  #define mj_atomic_store_bool(ptr, val) \
      (void)_InterlockedExchange8((volatile char*)(ptr), (char)(val))
  #define mj_atomic_exchange_bool(ptr, val) \
      (mjtBool) _InterlockedExchange8((volatile char*)(ptr), (char)(val))
#else
  #define mj_atomic_load_bool(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
  #define mj_atomic_store_bool(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
  #define mj_atomic_exchange_bool(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef mjUSEASAN
int mj__comparePcFuncName(void* pc1, void* pc2);
const char* mj__getPcDebugInfo(void* pc);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_CROSSPLATFORM_H_
