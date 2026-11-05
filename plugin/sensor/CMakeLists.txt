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

set(MUJOCO_SENSOR_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

set(MUJOCO_SENSOR_SRCS
    register.cc
    touch_grid.h
    touch_grid.cc
)

add_library(sensor SHARED)
set_target_properties(
  sensor PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mujoco_plugin"
)
target_sources(sensor PRIVATE ${MUJOCO_SENSOR_SRCS})
target_include_directories(sensor PRIVATE ${MUJOCO_SENSOR_INCLUDE})
target_link_libraries(sensor PRIVATE mujoco)
target_compile_options(
  sensor
  PRIVATE ${AVX_COMPILE_OPTIONS}
          ${MUJOCO_MACOS_COMPILE_OPTIONS}
          ${EXTRA_COMPILE_OPTIONS}
          ${MUJOCO_CXX_FLAGS}
)
target_link_options(
  sensor
  PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)
