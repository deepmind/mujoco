# Copyright 2026 DeepMind Technologies Limited
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

if(NOT TARGET mujoco::platform)
  message(WARNING "mujoco::platform target not found. studio bindings will not be built.")
  return()
endif()

set(STUDIO_COMMON_LIBS
  mujoco::platform
  structs_header
  _render_filament_header
)

mujoco_pybind11_module(_studio
  studio.cc
  native_viewer.cc
  ux.cc
  sim.cc
  window.cc
)
target_compile_definitions(_studio PRIVATE MUJOCO_STUDIO_COMBINED_MODULE)
target_link_libraries(_studio PRIVATE ${STUDIO_COMMON_LIBS})

add_subdirectory(web)

set(MUJOCO_STUDIO_TARGETS
    _studio
    PARENT_SCOPE
)

# dear_imgui bindings
mujoco_pybind11_module(dear_imgui dear_imgui/dear_imgui.cc)
target_link_libraries(dear_imgui PRIVATE
  mujoco::dep::dear_imgui
  mujoco::dep::dear_imgui_stdlib
)
target_include_directories(dear_imgui PRIVATE dear_imgui)
set_target_properties(dear_imgui PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dear_imgui"
)

# implot bindings
mujoco_pybind11_module(implot implot/implot.cc)
target_link_libraries(implot PRIVATE
  mujoco::dep::implot
  mujoco::dep::dear_imgui
)
target_include_directories(implot PRIVATE dear_imgui)
set_target_properties(implot PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/implot"
)

# Workaround for duplicate symbol error (wp_fractional_scale_manager_v1_interface)
# when linking both GLFW (via Filament) and SDL2 statically on Linux.
if(UNIX AND NOT APPLE)
  target_link_options(_studio PRIVATE "-Wl,--allow-multiple-definition")
endif()

if(APPLE)
  # A module records where to look for the libraries it needs as a list of
  # rpaths baked into the binary at link time, where @loader_path means "the
  # directory this module is in". mujoco_pybind11_module (../CMakeLists.txt)
  # gives every module it creates exactly one: @loader_path. That is correct for
  # the modules at the mujoco/ package root, which sit next to libmujoco, and
  # wrong for these, which are installed below it -- in
  # mujoco/studio/, mujoco/studio/dear_imgui/ and so on --
  # while libmujoco stays at the root. So append the hop back up to the root.
  set_property(TARGET _studio APPEND PROPERTY BUILD_RPATH "@loader_path/..")
  foreach(target dear_imgui implot)
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "@loader_path/../..")
  endforeach()
endif()
