# Copyright 2021 DeepMind Technologies Limited
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

include(FindOrFetch)

if(SIMULATE_STANDALONE)
  # If standalone, by default look for MuJoCo binary version.
  set(DEFAULT_USE_SYSTEM_MUJOCO ON)
else()
  set(DEFAULT_USE_SYSTEM_MUJOCO OFF)
endif()

option(MUJOCO_SIMULATE_USE_SYSTEM_MUJOCO "Use installed MuJoCo version."
       ${DEFAULT_USE_SYSTEM_MUJOCO}
)
unset(DEFAULT_USE_SYSTEM_MUJOCO)

option(MUJOCO_SIMULATE_USE_SYSTEM_MUJOCO "Use installed MuJoCo version." OFF)
option(MUJOCO_SIMULATE_USE_SYSTEM_GLFW "Use installed GLFW version." OFF)

set(MUJOCO_DEP_VERSION_glfw
    7d5a16ce714f0b5f4efa3262de22e4d948851525 # 3.3.6
    CACHE STRING "Version of `glfw` to be fetched."
)
mark_as_advanced(MUJOCO_DEP_VERSION_glfw)

find_package(Threads REQUIRED)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_BUILD_PYTHON OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  MUJOCO_SIMULATE_USE_SYSTEM_MUJOCO
  PACKAGE_NAME
  mujoco
  LIBRARY_NAME
  mujoco
  GIT_REPO
  https://github.com/deepmind/mujoco.git
  GIT_TAG
  main
  TARGETS
  mujoco
  EXCLUDE_FROM_ALL
)

option(MUJOCO_EXTRAS_STATIC_GLFW "Link MuJoCo sample apps and simulate libraries against GLFW statically." OFF)
if(MUJOCO_EXTRAS_STATIC_GLFW)
  set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS
      OFF
      CACHE INTERNAL "Build SHARED libraries"
  )
endif()

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  MUJOCO_SAMPLES_USE_SYSTEM_GLFW
  PACKAGE_NAME
  glfw
  LIBRARY_NAME
  glfw
  GIT_REPO
  https://github.com/glfw/glfw.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_glfw}
  TARGETS
  glfw
  EXCLUDE_FROM_ALL
)

if(MUJOCO_EXTRAS_STATIC_GLFW)
  set(BUILD_SHARED_LIBS
      ${BUILD_SHARED_LIBS_OLD}
      CACHE BOOL "Build SHARED libraries" FORCE
  )
  unset(BUILD_SHARED_LIBS_OLD)
endif()

if(NOT SIMULATE_STANDALONE)
  target_compile_options(glfw PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
  target_link_options(glfw PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
endif()
