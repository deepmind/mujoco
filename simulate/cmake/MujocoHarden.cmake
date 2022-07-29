# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

option(MUJOCO_HARDEN "Enable build hardening for MuJoCo." OFF)
if(MUJOCO_HARDEN
   AND NOT
       CMAKE_CXX_COMPILER_ID
       MATCHES
       ".*Clang.*"
)
  message(FATAL_ERROR "MUJOCO_HARDEN is only supported when building with Clang")
endif()

if(MUJOCO_HARDEN)
  set(MUJOCO_HARDEN_COMPILE_OPTIONS -D_FORTIFY_SOURCE=2 -fstack-protector)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MUJOCO_HARDEN_LINK_OPTIONS -Wl,-bind_at_load)
  elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(MUJOCO_HARDEN_LINK_OPTIONS -Wl,-z,relro -Wl,-z,now)
  endif()
else()
  set(MUJOCO_HARDEN_COMPILE_OPTIONS "")
  set(MUJOCO_HARDEN_LINK_OPTIONS "")
endif()
