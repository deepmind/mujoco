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

# Build configuration for third party libraries used in MuJoCo.

set(MUJOCO_DEP_VERSION_tinyxml2
    e6caeae85799003f4ca74ff26ee16a789bc2af48
    CACHE STRING "Version of `tinyxml2` to be fetched."
)
set(MUJOCO_DEP_VERSION_tinyobjloader
    2945a967c5303b2c8c14174117c45f3302591150
    CACHE STRING "Version of `tinyobjloader` to be fetched."
)
set(MUJOCO_DEP_VERSION_MarchingCubeCpp
    f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d
    CACHE STRING "Version of `MarchingCubeCpp` to be fetched."
)
set(MUJOCO_DEP_VERSION_ccd
    7931e764a19ef6b21b443376c699bbc9c6d4fba8 # v2.1
    CACHE STRING "Version of `ccd` to be fetched."
)
set(MUJOCO_DEP_VERSION_qhull
    d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6
    CACHE STRING "Version of `qhull` to be fetched."
)
set(MUJOCO_DEP_VERSION_miniz
    77d0dce8627735138c51770d1799a1ef48f2117d
    CACHE STRING "Version of `miniz` to be fetched."
)
set(MUJOCO_DEP_VERSION_Eigen3
    1da510b8155f8fa81924616dcc517ecc4202e4d3
    CACHE STRING "Version of `Eigen3` to be fetched."
)

set(MUJOCO_DEP_VERSION_abseil
    2065f4ded0558c6f89fee67c8e5228feb4eb960e # LTS 20260817.0
    CACHE STRING "Version of `abseil` to be fetched."
)

set(MUJOCO_DEP_VERSION_gtest
    063de7e9578f82b369302001269680b4b1553359 # v1.18.0
    CACHE STRING "Version of `gtest` to be fetched."
)

set(MUJOCO_DEP_VERSION_benchmark
    a6b4a9f00b36c19af9997d68a1cf6e0b5b3d032c
    CACHE STRING "Version of `benchmark` to be fetched."
)

mark_as_advanced(MUJOCO_DEP_VERSION_MarchingCubeCpp)
mark_as_advanced(MUJOCO_DEP_VERSION_tinyxml2)
mark_as_advanced(MUJOCO_DEP_VERSION_tinyobjloader)
mark_as_advanced(MUJOCO_DEP_VERSION_ccd)
mark_as_advanced(MUJOCO_DEP_VERSION_qhull)
mark_as_advanced(MUJOCO_DEP_VERSION_Eigen3)
mark_as_advanced(MUJOCO_DEP_VERSION_abseil)
mark_as_advanced(MUJOCO_DEP_VERSION_gtest)
mark_as_advanced(MUJOCO_DEP_VERSION_benchmark)

include(FetchContent)
include(FindOrFetch)

# Override the BUILD_SHARED_LIBS setting, just for building third party libs (since we always want
# static libraries). The ccd CMakeLists.txt doesn't expose an option to build a static ccd library,
# unless BUILD_SHARED_LIBS is set.

# We force all the dependencies to be compiled as static libraries.
# TODO(fraromano) Revisit this choice when adding support for install.
set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
    OFF
    CACHE INTERNAL "Build SHARED libraries"
)

include(third_party_deps/lodepng)

if(NOT TARGET marchingcubecpp)
  find_package(Git REQUIRED)
  _mujoco_ensure_git_retry_script()
  FetchContent_Declare(
    marchingcubecpp
    # GIT_REPOSITORY https://github.com/aparis69/MarchingCubeCpp.git
    # GIT_TAG ${MUJOCO_DEP_VERSION_MarchingCubeCpp}
    DOWNLOAD_COMMAND
      ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>
      COMMAND ${GIT_EXECUTABLE} init <SOURCE_DIR>
      COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> remote add origin https://github.com/aparis69/MarchingCubeCpp.git
      COMMAND ${CMAKE_COMMAND} -P "${_MUJOCO_GIT_RETRY_SCRIPT}" -- ${GIT_EXECUTABLE} -C <SOURCE_DIR> fetch --depth 1 origin ${MUJOCO_DEP_VERSION_MarchingCubeCpp}
      COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> -c advice.detachedHead=false checkout FETCH_HEAD
    UPDATE_DISCONNECTED TRUE
  )

  FetchContent_GetProperties(marchingcubecpp)
  if(NOT marchingcubecpp_POPULATED)
    FetchContent_Populate(marchingcubecpp)
    include_directories(${marchingcubecpp_SOURCE_DIR})
  endif()
endif()

set(QHULL_ENABLE_TESTING OFF)
# Patch changes in https://github.com/qhull/qhull/pull/173.patch
set(QHULL_PATCH_COMMAND
  git --git-dir=. -c core.autocrlf=false -c core.whitespace=cr-at-eol apply --verbose --whitespace=fix --ignore-space-change ${mujoco_SOURCE_DIR}/cmake/qhull-support-emscripten.patch
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  qhull
  LIBRARY_NAME
  qhull
  GIT_REPO
  https://github.com/qhull/qhull.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_qhull}
  TARGETS
  qhull
  EXCLUDE_FROM_ALL
  PATCH_COMMAND ${QHULL_PATCH_COMMAND}
)
# MuJoCo includes a file from libqhull_r which is not exported by the qhull include directories.
# Add it to the target.
target_include_directories(
  qhullstatic_r INTERFACE $<BUILD_INTERFACE:${qhull_SOURCE_DIR}/src/libqhull_r>
)
target_compile_options(qhullstatic_r PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
target_link_options(qhullstatic_r PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})

set(tinyxml2_BUILD_TESTING OFF)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  tinyxml2
  LIBRARY_NAME
  tinyxml2
  GIT_REPO
  https://github.com/leethomason/tinyxml2.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_tinyxml2}
  TARGETS
  tinyxml2
  EXCLUDE_FROM_ALL
)
target_compile_options(tinyxml2 PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
target_link_options(tinyxml2 PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})

# update cmake_minimum_required version for compatibility with newer version of cmake
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(CMAKE_POLICY_VERSION_MINIMUM ${MUJOCO_CMAKE_MIN_REQ})
  set(CMAKE_POLICY_VERSION_MINIMUM_LOCALLY_DEFINED ON)
endif()
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  tinyobjloader
  LIBRARY_NAME
  tinyobjloader
  GIT_REPO
  https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_tinyobjloader}
  TARGETS
  tinyobjloader
  EXCLUDE_FROM_ALL
)
if(CMAKE_POLICY_VERSION_MINIMUM_LOCALLY_DEFINED)
  unset(CMAKE_POLICY_VERSION_MINIMUM)
  unset(CMAKE_POLICY_VERSION_MINIMUM_LOCALLY_DEFINED)
endif()

# build ccd in single precision when MuJoCo is built with mjUSESINGLE
if(CMAKE_C_FLAGS MATCHES "mjUSESINGLE")
  set(ENABLE_DOUBLE_PRECISION OFF)
else()
  set(ENABLE_DOUBLE_PRECISION ON)
endif()
set(CCD_HIDE_ALL_SYMBOLS ON)

# Patch changes in https://github.com/danfis/libccd/pull/83.patch
set(CCD_PATCH_COMMAND
  git --git-dir=. -c core.autocrlf=false -c core.whitespace=cr-at-eol apply --verbose --whitespace=fix --ignore-space-change ${mujoco_SOURCE_DIR}/cmake/ccd-support-emscripten.patch
)

# update cmake_minimum_required version for compatibility with newer version of cmake
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(CMAKE_POLICY_VERSION_MINIMUM ${MUJOCO_CMAKE_MIN_REQ})
  set(CMAKE_POLICY_VERSION_MINIMUM_LOCALLY_DEFINED ON)
endif()
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  ccd
  LIBRARY_NAME
  ccd
  GIT_REPO
  https://github.com/danfis/libccd.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_ccd}
  TARGETS
  ccd
  EXCLUDE_FROM_ALL
  PATCH_COMMAND ${CCD_PATCH_COMMAND}
)
if(CMAKE_POLICY_VERSION_MINIMUM_LOCALLY_DEFINED)
  unset(CMAKE_POLICY_VERSION_MINIMUM)
  unset(CMAKE_POLICY_VERSION_MINIMUM_LOCALLY_DEFINED)
endif()
target_compile_options(ccd PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
target_link_options(ccd PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})

# libCCD has an unconditional `#define _CRT_SECURE_NO_WARNINGS` on Windows.
# TODO(stunya): Remove this after https://github.com/danfis/libccd/pull/77 is merged.
if(WIN32)
  if(MSVC)
    # C4005 is the MSVC equivalent of -Wmacro-redefined.
    target_compile_options(ccd PRIVATE /wd4005)
  else()
    target_compile_options(ccd PRIVATE -Wno-macro-redefined)
  endif()
endif()

if(DEFINED BUILD_TESTS)
  set(_OLD_BUILD_TESTS "${BUILD_TESTS}")
  set(_BUILD_TESTS_WAS_DEFINED TRUE)
else()
  set(_BUILD_TESTS_WAS_DEFINED FALSE)
endif()
set(BUILD_TESTS OFF)
fetchpackage(
  PACKAGE_NAME  miniz
  GIT_REPO      https://github.com/richgel999/miniz.git
  GIT_TAG       ${MUJOCO_DEP_VERSION_miniz}
  TARGETS       miniz
)
if(_BUILD_TESTS_WAS_DEFINED)
  set(BUILD_TESTS "${_OLD_BUILD_TESTS}")
else()
  unset(BUILD_TESTS)
endif()
unset(_BUILD_TESTS_WAS_DEFINED)


if(MUJOCO_BUILD_TESTS OR MUJOCO_BUILD_STUDIO OR MUJOCO_USE_FILAMENT)
  set(ABSL_PROPAGATE_CXX_STD ON)

  # This specific version of Abseil does not have the following variable. We need to work with BUILD_TESTING
  set(BUILD_TESTING_OLD ${BUILD_TESTING})
  set(BUILD_TESTING
      OFF
      CACHE INTERNAL "Build tests."
  )

  set(ABSL_PATCH_COMMAND
    git --git-dir=. -c core.autocrlf=false -c core.whitespace=cr-at-eol apply --verbose --whitespace=fix --ignore-space-change ${mujoco_SOURCE_DIR}/cmake/abseil-cpp-source_location.patch
  )
  set(ABSL_BUILD_TESTING OFF)
  findorfetch(
    USE_SYSTEM_PACKAGE
    OFF
    PACKAGE_NAME
    absl
    LIBRARY_NAME
    abseil-cpp
    GIT_REPO
    https://github.com/abseil/abseil-cpp.git
    GIT_TAG
    ${MUJOCO_DEP_VERSION_abseil}
    TARGETS
    absl::core_headers
    EXCLUDE_FROM_ALL
    PATCH_COMMAND
    ${ABSL_PATCH_COMMAND}
  )

  set(BUILD_TESTING
      ${BUILD_TESTING_OLD}
      CACHE BOOL "Build tests." FORCE
  )
endif()

if(MUJOCO_BUILD_TESTS)

  # Avoid linking errors on Windows by dynamically linking to the C runtime.
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE
  )

  findorfetch(
    USE_SYSTEM_PACKAGE
    OFF
    PACKAGE_NAME
    GTest
    LIBRARY_NAME
    googletest
    GIT_REPO
    https://github.com/google/googletest.git
    GIT_TAG
    ${MUJOCO_DEP_VERSION_gtest}
    TARGETS
    gtest
    gmock
    gtest_main
    EXCLUDE_FROM_ALL
  )

  set(BENCHMARK_EXTRA_FETCH_ARGS "")
  if(WIN32 AND NOT MSVC)
    set(BENCHMARK_EXTRA_FETCH_ARGS
        PATCH_COMMAND
        "sed"
        "-i"
        "-e"
        "s/-Wformat=2/-Wformat/g"
        "${CMAKE_BINARY_DIR}/_deps/benchmark-src/CMakeLists.txt"
    )
  endif()

  set(BENCHMARK_ENABLE_TESTING OFF)

  findorfetch(
    USE_SYSTEM_PACKAGE
    OFF
    PACKAGE_NAME
    benchmark
    LIBRARY_NAME
    benchmark
    GIT_REPO
    https://github.com/google/benchmark.git
    GIT_TAG
    ${MUJOCO_DEP_VERSION_benchmark}
    TARGETS
    benchmark::benchmark
    benchmark::benchmark_main
    ${BENCHMARK_EXTRA_FETCH_ARGS}
    EXCLUDE_FROM_ALL
  )
endif()

if(MUJOCO_TEST_PYTHON_UTIL)
  add_compile_definitions(EIGEN_MPL2_ONLY)
  if(NOT TARGET eigen)
    # Support new IN_LIST if() operator.
    set(CMAKE_POLICY_DEFAULT_CMP0057 NEW)

    find_package(Git REQUIRED)
    _mujoco_ensure_git_retry_script()
    FetchContent_Declare(
      Eigen3
      # GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      # GIT_TAG ${MUJOCO_DEP_VERSION_Eigen3}
      DOWNLOAD_COMMAND
        ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>
        COMMAND ${GIT_EXECUTABLE} init <SOURCE_DIR>
        COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> remote add origin https://gitlab.com/libeigen/eigen.git
        COMMAND ${CMAKE_COMMAND} -P "${_MUJOCO_GIT_RETRY_SCRIPT}" -- ${GIT_EXECUTABLE} -C <SOURCE_DIR> fetch --depth 1 origin ${MUJOCO_DEP_VERSION_Eigen3}
        COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> -c advice.detachedHead=false checkout FETCH_HEAD
      UPDATE_DISCONNECTED TRUE
    )

    FetchContent_GetProperties(Eigen3)
    if(NOT Eigen3_POPULATED)
      FetchContent_Populate(Eigen3)

      # Mark the library as IMPORTED as a workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/15415
      add_library(Eigen3::Eigen INTERFACE IMPORTED)
      set_target_properties(
        Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}"
      )
    endif()
  endif()
endif()

# Reset BUILD_SHARED_LIBS to its previous value
set(BUILD_SHARED_LIBS
    ${BUILD_SHARED_LIBS_OLD}
    CACHE BOOL "Build MuJoCo as a shared library" FORCE
)
