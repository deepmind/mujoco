// Copyright 2023 DeepMind Technologies Limited
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

#ifndef MUJOCO_MJMACRO_H_
#define MUJOCO_MJMACRO_H_

// max and min (use only for primitive types)
#define mjMAX(a, b) (((a) > (b)) ? (a) : (b))
#define mjMIN(a, b) (((a) < (b)) ? (a) : (b))

// return current value of mjOption enable/disable flags
#define mjDISABLED(x) (m->opt.disableflags & (x))
#define mjENABLED(x)  (m->opt.enableflags & (x))

// is actuator disabled
#define mjACTUATORDISABLED(i)                                           \
  (((m->actuator_group[(i)] >= 0) && (m->actuator_group[(i)] <= 30)     \
    && (m->opt.disableactuator & (1 << m->actuator_group[(i)]))) ? 1 : 0)

// annotation for functions that accept printf-like variadic arguments
#ifndef mjPRINTFLIKE
  #if defined(__GNUC__)
    #define mjPRINTFLIKE(n, m) __attribute__((format(printf, n, m)))
  #else
    #define mjPRINTFLIKE(n, m)
  #endif
#endif

#endif  // MUJOCO_MJMACRO_H_
