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

# NetImgui client library (vendored, MIT licensed — see LICENSE).
#
# This CMakeLists is included from two different build contexts:
#  1. The main MuJoCo tree under Emscripten (for the web viewer WASM client),
#     where the `dear_imgui` target exists.
#  2. The Python bindings build (for the headless_ui pybind module), where Dear
#     ImGui is an imported archive named `mujoco::dep::dear_imgui`.

if(TARGET netimgui)
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
  message(WARNING "No Dear ImGui target found. netimgui will not be built.")
  return()
endif()

add_library(netimgui STATIC
  Code/Client/Private/NetImgui_Api.cpp
  Code/Client/Private/NetImgui_Client.cpp
  Code/Client/Private/NetImgui_CmdPackets_DrawFrame.cpp
)

if(EMSCRIPTEN)
  # WebSocket-based networking for the browser (WASM) build.
  target_sources(netimgui PRIVATE google/NetImgui_NetworkWASM.cpp)
  # emscripten_websocket_* requires the JS websocket library.
  target_link_options(netimgui PUBLIC -lwebsocket.js)
elseif(WIN32)
  # Winsock2 networking for Windows.
  target_sources(netimgui PRIVATE Code/Client/Private/NetImgui_NetworkWin32.cpp)
  # The source self-links ws2_32 via `#pragma comment(lib, ...)` under MSVC;
  # link it explicitly so non-MSVC toolchains (e.g. clang-cl) resolve it too.
  target_link_libraries(netimgui PUBLIC ws2_32)
elseif(UNIX)
  # BSD-socket networking for Linux and macOS (both are UNIX).
  target_sources(netimgui PRIVATE Code/Client/Private/NetImgui_NetworkPosix.cpp)
else()
  message(WARNING "NetImgui networking is not available on this platform. "
                  "netimgui will not be built.")
  return()
endif()

target_include_directories(netimgui
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                      # google/logging.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Code/Client          # NetImgui_Api.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Code/Client/Private  # NetImgui_CmdPackets.h etc.
)

target_link_libraries(netimgui PUBLIC ${NETIMGUI_IMGUI_TARGET})

set_target_properties(netimgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
