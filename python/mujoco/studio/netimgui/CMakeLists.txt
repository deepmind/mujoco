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

# NetImgui protocol library (vendored, MIT licensed — see LICENSE).
#
# This CMakeLists is included from two different build contexts:
#  1. The main MuJoCo tree under Emscripten (for the web viewer WASM client),
#     where the `dear_imgui` target exists.
#  2. The Python bindings build (for the _studio pybind module), where Dear
#     ImGui is an imported archive named `mujoco::dep::dear_imgui`.

if(TARGET netimgui_protocol)
  return()
endif()

# Resolve the Dear ImGui target for the current build context. Check the
# imported archive first: in the Python bindings build a pybind module named
# `dear_imgui` is also defined (the Python ImGui bindings), which must not be
# confused with the C++ library.
if(TARGET mujoco::dep::dear_imgui)
  set(NETIMGUI_IMGUI_TARGET mujoco::dep::dear_imgui)
elseif(TARGET dear_imgui)
  set(NETIMGUI_IMGUI_TARGET dear_imgui)
else()
  message(WARNING "No Dear ImGui target found. netimgui_protocol will not be built.")
  return()
endif()

# NetImgui's protocol: its messages and draw frame codec, without the client
# runtime and its connection thread and sockets. The MuJoCo web viewer carries
# the messages itself, so it uses only this.
add_library(netimgui_protocol STATIC
  Code/Client/Private/NetImgui_CmdPackets_DrawFrame.cpp
)

target_include_directories(netimgui_protocol
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                      # google/logging.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Code/Client          # NetImgui_Api.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Code/Client/Private  # NetImgui_CmdPackets.h etc.
)

target_link_libraries(netimgui_protocol PUBLIC ${NETIMGUI_IMGUI_TARGET})

set_target_properties(netimgui_protocol PROPERTIES POSITION_INDEPENDENT_CODE ON)
