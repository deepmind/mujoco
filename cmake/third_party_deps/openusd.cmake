# Copyright 2025 DeepMind Technologies Limited
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

# Find OpenUSD package. The user can either provide pxr_DIR to use their own
# OpenUSD installation or they can build external_dependencies.
#
# We use QUIET here to provide a better error message when pxr is not found.
find_package(pxr
  QUIET
  HINTS "${CMAKE_BINARY_DIR}/_deps/openusd-build"
)

if(NOT pxr_FOUND)
    message(FATAL_ERROR
        "-----------------------------------------------------------\n"
        "USD Configuration Error:\n${pxr_NOT_FOUND_MESSAGE}\n"
        "If you have built USD yourself, provide -Dpxr_DIR=your_pxr_install_dir' \n"
        "Otherwise you can build cmake/third_party_deps/openusd:\n"
        "cd ~/mujoco\n"
        "cmake -Bcmake/third_party_deps/openusd/build cmake/third_party_deps/openusd -DBUILD_USD=True\n"
        "cmake --build cmake/third_party_deps/openusd/build\n"
        "cmake -Bbuild -S. -DMUJOCO_WITH_USD=True\n"
        "cmake --build build\n"
        "cmake --install build"
        "-----------------------------------------------------------"
    )
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)
endif()
