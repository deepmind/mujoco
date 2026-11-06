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
cmake_minimum_required(VERSION "3.16")

project(openusd-dependency)

include(ExternalProject)
find_package(Python3 3.11...<3.14 COMPONENTS Interpreter Development QUIET)
if(NOT (Python3_Development_FOUND AND Python3_LIBRARIES))
  find_package(Python3 3.11...<3.14 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

get_filename_component(DEPS_DIR "${CMAKE_BINARY_DIR}/../../../../build/_deps" ABSOLUTE)
set(OPENUSD_TAG v25.11)
set(OPENUSD_TARBALL "${MUJOCO_CMAKE_DEP_CACHE}/openusd-${OPENUSD_TAG}.tar.gz")

set(OPENUSD_PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
if(Python3_Development_FOUND AND Python3_LIBRARIES)
  if(Python3_LIBRARY_RELEASE)
    set(OPENUSD_PYTHON_LIB "${Python3_LIBRARY_RELEASE}")
  else()
    list(GET Python3_LIBRARIES 0 OPENUSD_PYTHON_LIB)
  endif()
else()
  # Manylinux Python installations omit libpython3.XX.a on disk to save space.
  # PXR_PY_UNDEFINED_DYNAMIC_LOOKUP=ON avoids linking against it, but CMake's
  # FindPython3 inside OpenUSD still requires Python3_LIBRARY to exist as a file.
  file(MAKE_DIRECTORY "${DEPS_DIR}/openusd-build")
  set(OPENUSD_PYTHON_LIB "${DEPS_DIR}/openusd-build/libpython${OPENUSD_PYTHON_VERSION}.a")
  file(WRITE "${OPENUSD_PYTHON_LIB}" "!<arch>\n")
endif()

if(MUJOCO_CMAKE_DEP_CACHE AND EXISTS "${OPENUSD_TARBALL}")
  set(OPENUSD_SOURCE_ARGS URL "${OPENUSD_TARBALL}")
  file(MAKE_DIRECTORY "${DEPS_DIR}/openusd-build/src")
  foreach(tbb_archive v2020.3.1.zip v2020.3.zip tbb-2020.3-win.zip v2021.12.0.zip)
    if(EXISTS "${MUJOCO_CMAKE_DEP_CACHE}/${tbb_archive}")
      configure_file(
        "${MUJOCO_CMAKE_DEP_CACHE}/${tbb_archive}"
        "${DEPS_DIR}/openusd-build/src/${tbb_archive}"
        COPYONLY
      )
    endif()
  endforeach()
else()
  set(OPENUSD_SOURCE_ARGS
      GIT_REPOSITORY https://github.com/PixarAnimationStudios/USD.git
      GIT_TAG ${OPENUSD_TAG}
  )
endif()

set(OPENUSD_BUILD_ENV "")
set(OPENUSD_USD_BUILD_ARGS "USD,-DPXR_PY_UNDEFINED_DYNAMIC_LOOKUP=ON")
if(UNIX AND NOT APPLE)
  # Match PyPI's manylinux2014 usd-core wheel, which is built with the pre-C++11
  # libstdc++ string ABI (_GLIBCXX_USE_CXX11_ABI=0), and hide static
  # libstdc++_nonshared.a symbols so libusd_ms.so does not export them.
  set(
    OPENUSD_BUILD_ENV
    ${CMAKE_COMMAND} -E env
    CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
    LDFLAGS=-Wl,--exclude-libs,libstdc++_nonshared.a
  )
  set(OPENUSD_USD_BUILD_ARGS
      "USD,-DPXR_PY_UNDEFINED_DYNAMIC_LOOKUP=ON,-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
  )
endif()

ExternalProject_Add(openusd
  ${OPENUSD_SOURCE_ARGS}
  PREFIX openusd
  BUILD_IN_SOURCE 0
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  SOURCE_DIR ${DEPS_DIR}/openusd-src
  BINARY_DIR ${DEPS_DIR}/openusd-build
  INSTALL_DIR ${DEPS_DIR}/openusd-build
  BUILD_COMMAND ${OPENUSD_BUILD_ENV} ${Python3_EXECUTABLE} ${DEPS_DIR}/openusd-src/build_scripts/build_usd.py
      --build-monolithic
      --build-args ${OPENUSD_USD_BUILD_ARGS}
      --build-python-info
          ${Python3_EXECUTABLE}
          ${Python3_INCLUDE_DIRS}
          ${OPENUSD_PYTHON_LIB}
          ${OPENUSD_PYTHON_VERSION}
      --no-examples
      --no-tools
      --no-ptex
      --no-prman
      --no-openimageio
      --no-opencolorio
      --no-alembic
      --no-draco
      --no-materialx
      --no-tutorials
      --no-tests
      --no-docs
      --no-imaging
      --no-usdValidation
      <INSTALL_DIR>
)
