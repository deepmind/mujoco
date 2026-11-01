// Copyright 2025 DeepMind Technologies Limited
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

#include "engine/engine_memory.h"

#include <inttypes.h>  // IWYU pragma: keep
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mjmacro.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_crossplatform.h"
#include "engine/engine_util_errmem.h"


#ifdef mjUSEASAN
  #include <sanitizer/asan_interface.h>
  #include <sanitizer/common_interface_defs.h>
#endif

#ifdef MEMORY_SANITIZER
  #include <sanitizer/msan_interface.h>
#endif

#ifdef _MSC_VER
  #pragma warning (disable: 4305)  // disable MSVC warning: truncation from 'double' to 'float'
#endif

// add red zone padding when built with asan, to detect out-of-bound accesses
#ifdef mjUSEASAN
  #define mjREDZONE 32
#else
  #define mjREDZONE 0
#endif

// compute a % b with a fast code path if the second argument is a power of 2
static inline size_t fastmod(size_t a, size_t b) {
  // (b & (b - 1)) == 0 implies that b is a power of 2
  if (mjLIKELY((b & (b - 1)) == 0)) {
    return a & (b - 1);
  }
  return a % b;
}

typedef struct {
  uintptr_t bottom;      // first memory address available to the stack
  uintptr_t top;         // current memory address used by the stack
  uintptr_t limit;       // top limit of the stack (stack grows down)
  uintptr_t stack_base;  // current stack base for mark and free stack
} mjStackInfo;


typedef struct {
  size_t pbase;   // value of d->pbase immediately before mj_markStack
  size_t pstack;  // value of d->pstack immediately before mj_markStack
  void* pc;       // program counter of the call site of mj_markStack (only set when under asan)
} mjStackFrame;

static inline mjStackInfo get_stack_info_from_data(const mjData* d) {
  mjStackInfo stack_info;
  stack_info.bottom = (uintptr_t)d->arena + (uintptr_t)d->narena;
  stack_info.top = stack_info.bottom - d->pstack;
  stack_info.limit = (uintptr_t)d->arena + (uintptr_t)d->parena;
  stack_info.stack_base = d->pbase;

  return stack_info;
}


#ifdef mjUSEASAN
// get stack usage from red-zone (under ASAN)
static size_t stack_usage_redzone(const mjStackInfo* stack_info) {
  size_t usage = 0;

  // actual stack usage (without red zone bytes) is stored in the red zone
  if (stack_info->top != stack_info->bottom) {
    char* prev_pstack_ptr = (char*)(stack_info->top);
    size_t prev_misalign = (uintptr_t)prev_pstack_ptr % _Alignof(size_t);
    size_t* prev_usage_ptr =
      (size_t*)(prev_pstack_ptr +
                (prev_misalign ? _Alignof(size_t) - prev_misalign : 0));
    ASAN_UNPOISON_MEMORY_REGION(prev_usage_ptr, sizeof(size_t));
    usage = *prev_usage_ptr;
    ASAN_POISON_MEMORY_REGION(prev_usage_ptr, sizeof(size_t));
  }

  return usage;
}
#endif

// allocate memory from the mjData arena
void* mj_arenaAllocByte(mjData* d, size_t bytes, size_t alignment) {
  uintptr_t addr = (uintptr_t)d->arena + d->parena;
  size_t misalignment = fastmod(addr, alignment);
  size_t padding = misalignment ? alignment - misalignment : 0;

  // check size
  size_t bytes_available = d->narena - d->pstack;
  if (mjUNLIKELY(d->parena + padding + bytes > bytes_available)) {
    return NULL;
  }

  size_t stack_usage = d->pstack;

  // under ASAN, get stack usage from red zone
#ifdef mjUSEASAN
  mjStackInfo stack_info = get_stack_info_from_data(d);
  stack_usage = stack_usage_redzone(&stack_info);
#endif

  // allocate, update max, return pointer to buffer
  void* result = (char*)d->arena + d->parena + padding;
  d->parena += padding + bytes;
  d->maxuse_arena = mjMAX(d->maxuse_arena, stack_usage + d->parena);

#ifdef mjUSEASAN
  ASAN_UNPOISON_MEMORY_REGION(result, bytes);
#endif

#ifdef MEMORY_SANITIZER
  __msan_allocated_memory(result, bytes);
#endif

  return result;
}


// internal: allocate size bytes on the provided stack shard
// declared inline so that modular arithmetic with specific alignments can be optimized out
static inline void* stackallocinternal(mjData* d, mjStackInfo* stack_info, size_t size,
    size_t alignment, const char* caller, int line) {
  // return NULL if empty
  if (mjUNLIKELY(!size)) {
    return NULL;
  }

  // start of the memory to be allocated to the buffer
  uintptr_t start_ptr = stack_info->top - (size + mjREDZONE);

  // align the pointer
  start_ptr -= fastmod(start_ptr, alignment);

  // new top of the stack
  uintptr_t new_top_ptr = start_ptr - mjREDZONE;

  // exclude red zone from stack usage statistics
  size_t current_alloc_usage = stack_info->top - new_top_ptr - 2 * mjREDZONE;
  size_t usage = current_alloc_usage + (stack_info->bottom - stack_info->top);

  // check size
  size_t stack_available_bytes = stack_info->top - stack_info->limit;
  size_t stack_required_bytes = stack_info->top - new_top_ptr;
  if (mjUNLIKELY(stack_required_bytes > stack_available_bytes)) {
    char info[1024];
    if (caller) {
      snprintf(info, sizeof(info), " at %s, line %d", caller, line);
    } else {
      info[0] = '\0';
    }
    mju_error(
        "mj_stackAlloc: out of memory, stack overflow%s\n"
        "  max = %" PRIuPTR ", available = %" PRIuPTR ", requested = %" PRIuPTR
        ", narena = %" PRIuPTR " (increase <size memory=\"...\"/>)\n"
        "  nefc = %d, ncon = %d",
        info, stack_info->bottom - stack_info->limit, stack_available_bytes,
        stack_required_bytes, (uintptr_t)d->narena, d->nefc, d->ncon);
  }

#ifdef mjUSEASAN
  usage = current_alloc_usage + stack_usage_redzone(stack_info);

  // store new stack usage in the red zone
  size_t misalign = new_top_ptr % _Alignof(size_t);
  size_t* usage_ptr =
    (size_t*)(new_top_ptr + (misalign ? _Alignof(size_t) - misalign : 0));
  ASAN_UNPOISON_MEMORY_REGION(usage_ptr, sizeof(size_t));
  *usage_ptr = usage;
  ASAN_POISON_MEMORY_REGION(usage_ptr, sizeof(size_t));

  // unpoison the actual usable allocation
  ASAN_UNPOISON_MEMORY_REGION((void*)start_ptr, size);
#endif

  // update max usage statistics
  stack_info->top = new_top_ptr;
  d->maxuse_stack = mjMAX(d->maxuse_stack, usage);
  d->maxuse_arena = mjMAX(d->maxuse_arena, usage + d->parena);

  return (void*)start_ptr;
}


// internal: allocate size bytes in mjData
// declared inline so that modular arithmetic with specific alignments can be optimized out
static inline void* stackalloc(mjData* d, size_t size, size_t alignment,
                               const char* caller, int line) {
  // size zero: no-op
  if (!size) {
    return NULL;
  }

  // call in mju_dispatch: atomically reserve space on the stack
  if (d->threadlock) {
    size_t alloc_size = mj_stackBytes(size, alignment);
    size_t old_pstack = mj_atomic_add_size_t(&d->pstack, alloc_size);

    // check for stack overflow
    size_t stack_available_bytes = (size_t)d->narena - d->parena;
    if (mjUNLIKELY(old_pstack + alloc_size > stack_available_bytes)) {
      char info[1024];
      if (caller) {
        snprintf(info, sizeof(info), " at %s, line %d", caller, line);
      } else {
        info[0] = '\0';
      }
      mju_error(
          "mj_stackAlloc: out of memory, stack overflow%s (threadlock)\n"
          "  max = %" PRIuPTR ", available = %" PRIuPTR ", requested = %" PRIuPTR
          ", narena = %" PRIuPTR " (increase <size memory=\"...\"/>)\n"
          "  nefc = %d, ncon = %d",
          info, (uintptr_t)stack_available_bytes,
          (uintptr_t)(stack_available_bytes - old_pstack),
          (uintptr_t)alloc_size, (uintptr_t)d->narena, d->nefc, d->ncon);
    }

    uintptr_t bottom = (uintptr_t)d->arena + (uintptr_t)d->narena;
    uintptr_t start_ptr = bottom - old_pstack - size - mjREDZONE;
    start_ptr -= fastmod(start_ptr, alignment);
    ASAN_UNPOISON_MEMORY_REGION((void*)start_ptr, size);
    return (void*)start_ptr;
  }

  mjStackInfo stack_info = get_stack_info_from_data(d);
  void* result = stackallocinternal(d, &stack_info, size, alignment, caller, line);
  d->pstack = stack_info.bottom - stack_info.top;
  return result;
}


// mjStackInfo mark stack frame, inline so ASAN errors point to correct code unit
#ifdef mjUSEASAN
__attribute__((always_inline))
#endif
static inline void markstackinternal(mjData* d, mjStackInfo* stack_info) {
  size_t top_old = stack_info->top;
  mjStackFrame* s =
    (mjStackFrame*) stackallocinternal(d, stack_info, sizeof(mjStackFrame), _Alignof(mjStackFrame), NULL, 0);
  s->pbase = stack_info->stack_base;
  s->pstack = top_old;
#ifdef mjUSEASAN
  // store the program counter to the caller so that we can compare against mj_freeStack later
  s->pc = __sanitizer_return_address();
#endif
  stack_info->stack_base = (uintptr_t) s;
}


// mjData mark stack frame
#ifndef mjUSEASAN
void mj_markStack(mjData* d)
#else
void mj__markStack(mjData* d)
#endif
{
  // no-op if called from mju_dispatch
  if (d->threadlock) {
    return;
  }

  mjStackInfo stack_info = get_stack_info_from_data(d);
  markstackinternal(d, &stack_info);
  d->pstack = stack_info.bottom - stack_info.top;
  d->pbase = stack_info.stack_base;
}


#ifdef mjUSEASAN
__attribute__((always_inline))
#endif
static inline void freestackinternal(mjStackInfo* stack_info) {
  if (mjUNLIKELY(!stack_info->stack_base)) {
    return;
  }

  mjStackFrame* s = (mjStackFrame*) stack_info->stack_base;
#ifdef mjUSEASAN
  // raise an error if caller function name doesn't match the most recent caller of mj_markStack
  if (!mj__comparePcFuncName(s->pc, __sanitizer_return_address())) {
    mjERROR("mj_markStack %s has no corresponding mj_freeStack (detected %s)",
            mj__getPcDebugInfo(s->pc),
            mj__getPcDebugInfo(__sanitizer_return_address()));
  }
  uintptr_t old_top = stack_info->top;
#endif

  // restore pbase and pstack
  stack_info->stack_base = s->pbase;
  stack_info->top = s->pstack;

  // if running under asan, poison the newly freed memory region
#ifdef mjUSEASAN
  ASAN_POISON_MEMORY_REGION((char*)old_top, stack_info->top - old_top);
#endif
}


// mjData free stack frame
#ifndef mjUSEASAN
void mj_freeStack(mjData* d)
#else
void mj__freeStack(mjData* d)
#endif
{
  if (d->threadlock) {
    return;
  }

  mjStackInfo stack_info = get_stack_info_from_data(d);
  freestackinternal(&stack_info);
  d->pstack = stack_info.bottom - stack_info.top;
  d->pbase = stack_info.stack_base;
}


// allocate bytes on the stack
void* mj_stackAllocByte(mjData* d, size_t bytes, size_t alignment) {
  return stackalloc(d, bytes, alignment, NULL, 0);
}


// allocate bytes on the stack, with caller information
void* mj_stackAllocInfo(mjData* d, size_t bytes, size_t alignment,
                        const char* caller, int line) {
  return stackalloc(d, bytes, alignment, caller, line);
}


// allocate mjtNums on the stack
mjtNum* mj_stackAllocNum(mjData* d, size_t size) {
  if (mjUNLIKELY(size >= SIZE_MAX / sizeof(mjtNum))) {
    mjERROR("requested size is too large (more than 2^64 bytes).");
  }
  return (mjtNum*) stackalloc(d, size * sizeof(mjtNum), _Alignof(mjtNum), NULL, 0);
}


// allocate ints on the stack
int* mj_stackAllocInt(mjData* d, size_t size) {
  if (mjUNLIKELY(size >= SIZE_MAX / sizeof(int))) {
    mjERROR("requested size is too large (more than 2^64 bytes).");
  }
  return (int*) stackalloc(d, size * sizeof(int), _Alignof(int), NULL, 0);
}


// free bytes between the arena and the stack
size_t mj_stackBytesAvailable(const mjData* d) {
  return d->narena - d->pstack - d->parena;
}


// stack bytes one allocation of the given size and alignment takes, at most: the alignment
// padding and the red zones on both sides, as stackalloc reserves them
size_t mj_stackBytes(size_t bytes, size_t alignment) {
  if (!bytes) {
    return 0;
  }
  return bytes + alignment - 1 + 2 * mjREDZONE;
}


// stack bytes one mj_markStack frame takes, at most
size_t mj_stackFrameBytes(void) {
  return mj_stackBytes(sizeof(mjStackFrame), _Alignof(mjStackFrame));
}
