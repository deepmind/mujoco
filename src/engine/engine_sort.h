// Copyright 2024 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_SORT_H_
#define MUJOCO_SRC_ENGINE_ENGINE_SORT_H_

// threshold size of a run to do insertion sort on
#define _mjRUNSIZE 32

// insertion sort sub-macro that runs on a sub-array [start, ..., end)
#define _mjINSERTION_SORT(type, arr, start, end, cmp, context)                                     \
{                                                                                                  \
  for (int j = start + 1; j < end; j++) {                                                          \
    type tmp = arr[j];                                                                             \
    int i = j - 1;                                                                                 \
    for (; i >= start && cmp(arr + i, &tmp, context) > 0; i--) {                                   \
      arr[i+1] = arr[i];                                                                           \
    }                                                                                              \
    arr[i+1] = tmp;                                                                                \
  }                                                                                                \
}

// sub-macro that merges two sub-sorted arrays [start, ..., mid), [mid, ..., end) together
#define _mjMERGE(type, src, dest, start, mid, end, cmp, context)                                   \
{                                                                                                  \
  int i = start, j = mid, k = start;                                                               \
  while (i < mid && j < end) {                                                                     \
    if (cmp(src + i, src + j, context) <= 0) {                                                     \
       dest[k++] = src[i++];                                                                       \
    } else {                                                                                       \
      dest[k++] = src[j++];                                                                        \
    }                                                                                              \
  }                                                                                                \
  if      (i < mid) memcpy(dest + k, src + i, (mid - i) * sizeof(type));                           \
  else if (j < end) memcpy(dest + k, src + j, (end - j) * sizeof(type));                           \
}

// defines an inline stable sorting function via tiled merge sorting (timsort)
// function is of form:
// void name(type* arr, type* buf, int n, void* context)
// where arr is the array of size n to be sorted inplace and buf is a buffer of size n.
#define mjSORT(name, type, cmp)                                                                    \
  static inline void name(type* arr, type* buf, int n, void* context) {                            \
    for (int start = 0; start < n; start += _mjRUNSIZE) {                                          \
      int end = (start + _mjRUNSIZE < n) ? start + _mjRUNSIZE : n;                                 \
      _mjINSERTION_SORT(type, arr, start, end, cmp, context);                                      \
    }                                                                                              \
    type* src = arr, *dest = buf, *tmp;                                                            \
    for (int len = _mjRUNSIZE; len < n; len *= 2) {                                                \
      for (int start = 0; start < n; start += 2*len) {                                             \
        int mid = start + len;                                                                     \
        int end = (start + 2*len < n) ? start + 2*len : n;                                         \
        if (mid < end) {                                                                           \
          _mjMERGE(type, src, dest, start, mid, end, cmp, context);                                \
        } else {                                                                                   \
          memcpy(dest + start, src + start, (end - start) * sizeof(type));                         \
        }                                                                                          \
      }                                                                                            \
      tmp = src; src = dest; dest = tmp;                                                           \
    }                                                                                              \
    if (src != arr) memcpy(arr, src, n * sizeof(type));                                            \
  } static inline void name(type* arr, type* buf, int n, void* context)


// sub-macro that sifts down a node in a max heap to its correct position
#define _mjSIFT_DOWN(type, buf, start, end, cmp, context)                                          \
{                                                                                                  \
  int root = start;                                                                                \
  while (2 * root + 1 < end) {                                                                     \
    int child = 2 * root + 1;                                                                      \
    int swap = root;                                                                               \
    if (cmp(buf + swap, buf + child, context) < 0) swap = child;                                   \
    if (child + 1 < end && cmp(buf + swap, buf + child + 1, context) < 0) swap = child + 1;        \
    if (swap == root) break;                                                                       \
    type tmp = buf[root]; buf[root] = buf[swap]; buf[swap] = tmp;                                  \
    root = swap;                                                                                   \
  }                                                                                                \
}

// defines an inline function that selects the bottom k elements using partial heap sort
// buf needs to be of size k
#define mjPARTIAL_SORT(name, type, cmp)                                                            \
  static inline void name(type* arr, type* buf, int n, int k, void* context) {                     \
    if (k <= 0 || n < k) return;                                                                   \
    /* fill initial heap */                                                                        \
    for (int i = 0; i < k; i++) buf[i] = arr[i];                                                   \
    for (int j = (k - 2) / 2; j >= 0; j--) _mjSIFT_DOWN(type, buf, j, k, cmp, context);            \
    /* scan remaining elements */                                                                  \
    for (int i = k; i < n; i++) {                                                                  \
      if (cmp(arr + i, buf, context) < 0) {                                                        \
        buf[0] = arr[i];                                                                           \
        _mjSIFT_DOWN(type, buf, 0, k, cmp, context);                                               \
      }                                                                                            \
    }                                                                                              \
    /* copy back and sort the result */                                                            \
    for (int j = 0; j < k; j++) arr[j] = buf[j];                                                   \
    _mjINSERTION_SORT(type, arr, 0, k, cmp, context);                                              \
  } static inline void name(type* arr, type* buf, int n, int k, void* context)

#endif  // MUJOCO_SRC_ENGINE_ENGINE_SORT_H_
