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

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  mujoco_samples
  VERSION 2.2.1
  DESCRIPTION "MuJoCo samples binaries"
  HOMEPAGE_URL "https://mujoco.org"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

# Check if we are building as standalone project.
set(SAMPLE_STANDALONE OFF)
set(_INSTALL_SAMPLES ON)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(SAMPLE_STANDALONE ON)
  # If standalone, do not install the samples.
  set(_INSTALL_SAMPLES OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if(SAMPLE_STANDALONE)
  include(SampleOptions)
else()
  enforce_mujoco_macosx_min_version()
endif()
include(SampleDependencies)

set(MUJOCO_SAMPLE_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MUJOCO_SAMPLE_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

if(MUJOCO_HARDEN)
  if(WIN32)
    set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -Wl,/DYNAMICBASE)
  else()
    set(MUJOCO_SAMPLE_COMPILE_OPTIONS "${MUJOCO_SAMPLE_COMPILE_OPTIONS}" -fPIE)
    if(APPLE)
      set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -Wl,-pie)
    else()
      set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -pie)
    endif()
  endif()
endif()

# Utility library
add_library(uitools STATIC)
target_sources(uitools PRIVATE uitools.h uitools.c)
target_include_directories(uitools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(uitools PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(uitools PUBLIC glfw mujoco::mujoco)
target_link_options(uitools PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

# Build sample binaries
add_executable(compile compile.cc)
target_compile_options(compile PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(compile Threads::Threads)
target_link_options(compile PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(derivative derivative.cc)
target_compile_options(derivative PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(derivative Threads::Threads)
target_link_options(derivative PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(testspeed testspeed.cc)
target_compile_options(testspeed PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(testspeed Threads::Threads)
target_link_options(testspeed PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(testxml testxml.cc array_safety.h)
target_compile_options(testxml PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(testxml Threads::Threads)
target_link_options(testxml PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

target_link_libraries(compile mujoco::mujoco)
target_link_libraries(derivative mujoco::mujoco)
target_link_libraries(testspeed mujoco::mujoco)
target_link_libraries(testxml mujoco::mujoco)

# Build samples that require GLFW.

add_executable(basic basic.cc)
target_compile_options(basic PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
  basic
  mujoco::mujoco
  glfw
  Threads::Threads
)
target_link_options(basic PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(record record.cc array_safety.h)
target_compile_options(record PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
  record
  mujoco::mujoco
  glfw
  Threads::Threads
)
target_link_options(record PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

if(APPLE)
  set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.icns)
elseif(WIN32)
  set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/appicon.rc)
else()
  set(SIMULATE_RESOURCE_FILES "")
endif()

add_executable(simulate simulate.cc array_safety.h ${SIMULATE_RESOURCE_FILES})
target_compile_options(simulate PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
if(WIN32)
  add_custom_command(
    TARGET simulate
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.ico
            ${CMAKE_CURRENT_SOURCE_DIR}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_SOURCE_DIR}/mujoco.ico
  )
endif()

target_link_libraries(
  simulate
  mujoco::mujoco
  uitools
  glfw
  Threads::Threads
  portable-file-dialogs
)
target_link_options(simulate PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

if(APPLE)
  target_sources(simulate PRIVATE macos_save.mm)
  target_link_libraries(simulate "-framework Cocoa")
endif()

if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
  set_target_properties(
    simulate
    PROPERTIES INSTALL_RPATH @executable_path/../Frameworks
               BUILD_WITH_INSTALL_RPATH TRUE
               RESOURCE ${SIMULATE_RESOURCE_FILES}
               MACOSX_BUNDLE TRUE
               MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/../dist/Info.plist.simulate.in
               MACOSX_BUNDLE_BUNDLE_NAME "MuJoCo"
               MACOSX_BUNDLE_GUI_IDENTIFIER "org.mujoco.app"
               MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
               MACOSX_BUNDLE_INFO_STRING ${PROJECT_VERSION}
               MACOSX_BUNDLE_LONG_VERSION_STRING ${PROJECT_VERSION}
               MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION}
               MACOSX_BUNDLE_ICON_FILE "mujoco.icns"
               MACOSX_BUNDLE_COPYRIGHT "Copyright 2021 DeepMind Technologies Limited."
  )

  macro(embed_in_bundle target)
    add_dependencies(${target} simulate)
    set_target_properties(
      ${target}
      PROPERTIES INSTALL_RPATH @executable_path/../Frameworks
                 BUILD_WITH_INSTALL_RPATH TRUE
                 RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:simulate>
    )
  endmacro()

  embed_in_bundle(basic simulate)
  embed_in_bundle(compile simulate)
  embed_in_bundle(derivative simulate)
  embed_in_bundle(record simulate)
  embed_in_bundle(testspeed simulate)
  embed_in_bundle(testxml simulate)

  # Embed mujoco.framework inside the App bundle.
  add_custom_command(
    TARGET simulate
    POST_BUILD
    COMMAND mkdir -p $<TARGET_FILE_DIR:simulate>/../Frameworks
    COMMAND rm -rf $<TARGET_FILE_DIR:simulate>/../Frameworks/mujoco.framework
    COMMAND cp -a $<TARGET_FILE_DIR:mujoco::mujoco>/../../../mujoco.framework
            $<TARGET_FILE_DIR:simulate>/../Frameworks/
  )
endif()

# Do not install if macOS Bundles are created as RPATH is managed manually there.
if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
  set(_INSTALL_SAMPLES OFF)
endif()

if(_INSTALL_SAMPLES)

  include(TargetAddRpath)

  # Add support to RPATH for the samples.
  target_add_rpath(
    TARGETS
    basic
    compile
    derivative
    record
    testspeed
    testxml
    simulate
    INSTALL_DIRECTORY
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
    LIB_DIRS
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    DEPENDS
    MUJOCO_ENABLE_RPATH
  )

  install(
    TARGETS basic
            compile
            derivative
            record
            testspeed
            testxml
            simulate
    EXPORT ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT samples
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
    BUNDLE DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT samples
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT samples
  )

  if(NOT MUJOCO_SAMPLES_USE_SYSTEM_GLFW)
    # We downloaded GLFW. Depending if it is a static or shared LIBRARY we might
    # need to install it.
    get_target_property(MJ_GLFW_LIBRARY_TYPE glfw TYPE)
    if(MJ_GLFW_LIBRARY_TYPE STREQUAL SHARED_LIBRARY)
      install(
        TARGETS glfw
        EXPORT ${PROJECT_NAME}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT samples
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT samples
      )
    endif()
  endif()
endif()
