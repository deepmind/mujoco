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
  mujoco_simulate
  VERSION 4.0.0
  DESCRIPTION "MuJoCo simulate binaries"
  HOMEPAGE_URL "https://mujoco.org"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

option(SIMULATE_BUILD_EXECUTABLE "Build the simulate executable binary." ON)
option(SIMULATE_GLFW_DYNAMIC_SYMBOLS "Whether to resolve GLFW symbols dynamically." OFF)

# Check if we are building as standalone project.
set(SIMULATE_STANDALONE OFF)
set(_INSTALL_SIMULATE ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(SIMULATE_STANDALONE ON)
  # If standalone, do not install the samples.
  set(_INSTALL_SIMULATE OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

if(SIMULATE_STANDALONE)
  include(SimulateOptions)
else()
  enforce_mujoco_macosx_min_version()
endif()
include(SimulateDependencies)

set(MUJOCO_SIMULATE_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MUJOCO_SIMULATE_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

if(MUJOCO_HARDEN)
  if(WIN32)
    set(MUJOCO_SIMULATE_LINK_OPTIONS "${MUJOCO_SIMULATE_LINK_OPTIONS}" -Wl,/DYNAMICBASE)
  else()
    set(MUJOCO_SIMULATE_COMPILE_OPTIONS "${MUJOCO_SIMULATE_COMPILE_OPTIONS}" -fPIE -fPIC)
    if(APPLE)
      set(MUJOCO_SIMULATE_LINK_OPTIONS "${MUJOCO_SIMULATE_LINK_OPTIONS}" -Wl,-pie)
    else()
      set(MUJOCO_SIMULATE_LINK_OPTIONS "${MUJOCO_SIMULATE_LINK_OPTIONS}" -pie)
    endif()
  endif()
endif()

# Fetch lodepng dependency.
include(third_party_deps/lodepng)

# Simulate library
add_library(platform_ui_adapter OBJECT)
target_sources(
  platform_ui_adapter
  PUBLIC glfw_adapter.h glfw_dispatch.h platform_ui_adapter.h
  PRIVATE glfw_adapter.cc glfw_dispatch.cc platform_ui_adapter.cc
)
target_compile_options(platform_ui_adapter PRIVATE ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
if(APPLE)
  target_sources(platform_ui_adapter PUBLIC glfw_corevideo.h PRIVATE glfw_corevideo.mm)
  set_source_files_properties(glfw_corevideo.mm PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  target_link_libraries(platform_ui_adapter PUBLIC "-framework CoreVideo")
endif()
target_include_directories(
  platform_ui_adapter PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                             $<TARGET_PROPERTY:glfw,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(platform_ui_adapter PUBLIC mujoco::mujoco)
if(SIMULATE_GLFW_DYNAMIC_SYMBOLS)
  target_compile_definitions(platform_ui_adapter PUBLIC mjGLFW_DYNAMIC_SYMBOLS)
endif()
add_library(mujoco::platform_ui_adapter ALIAS platform_ui_adapter)

add_library(libmujoco_simulate STATIC $<TARGET_OBJECTS:platform_ui_adapter> $<TARGET_OBJECTS:lodepng>)
set_target_properties(libmujoco_simulate PROPERTIES OUTPUT_NAME simulate)
add_library(mujoco::libmujoco_simulate ALIAS libmujoco_simulate)
set_target_properties(libmujoco_simulate PROPERTIES PUBLIC_HEADER "simulate.h;platform_ui_adapter.h;glfw_adapter.h;glfw_dispatch.h;glfw_corevideo.h;array_safety.h")


target_sources(
  libmujoco_simulate
  PRIVATE simulate.cc
)
target_include_directories(libmujoco_simulate
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/simulate>
    PRIVATE
        $<TARGET_PROPERTY:lodepng,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_options(libmujoco_simulate PRIVATE ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
target_link_libraries(libmujoco_simulate PUBLIC mujoco::mujoco)
target_link_options(libmujoco_simulate PRIVATE ${MUJOCO_SIMULATE_LINK_OPTIONS})

if(APPLE)
  target_sources(libmujoco_simulate PRIVATE macos_gui.mm)
  target_link_libraries(libmujoco_simulate PUBLIC "-framework Cocoa")
endif()

# Build simulate executable
if(SIMULATE_BUILD_EXECUTABLE)
  if(APPLE)
    set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.icns)
  elseif(WIN32)
    set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/simulate.rc)
  else()
    set(SIMULATE_RESOURCE_FILES "")
  endif()

  add_executable(simulate main.cc array_safety.h ${SIMULATE_RESOURCE_FILES})
  target_compile_options(simulate PUBLIC ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
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
    libmujoco_simulate
    mujoco::mujoco
    glfw
    Threads::Threads
  )

  if(APPLE)
    target_link_libraries(simulate "-framework CoreVideo")
  endif()

  target_link_options(simulate PRIVATE ${MUJOCO_SIMULATE_LINK_OPTIONS})

  if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
    set_target_properties(
      simulate
      PROPERTIES INSTALL_RPATH @executable_path/../Frameworks
                 BUILD_WITH_INSTALL_RPATH TRUE
                 RESOURCE ${SIMULATE_RESOURCE_FILES}
                 MACOSX_BUNDLE TRUE
                 MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/../dist/Info.plist.simulate.in
                 MACOSX_BUNDLE_BUNDLE_NAME "MuJoCo"
                 MACOSX_BUNDLE_GUI_IDENTIFIER "org.mujoco.mujoco"
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

    # Embed mujoco.framework inside the App bundle ane move the icon file over too.
    add_custom_command(
      TARGET simulate
      POST_BUILD
      COMMAND mkdir -p $<TARGET_FILE_DIR:simulate>/../Frameworks
      COMMAND rm -rf $<TARGET_FILE_DIR:simulate>/../Frameworks/mujoco.framework
      COMMAND cp -a $<TARGET_FILE_DIR:mujoco::mujoco>/../../../mujoco.framework
              $<TARGET_FILE_DIR:simulate>/../Frameworks/
      # Delete the symlink and the TBD, otherwise we can't sign and notarize.
      COMMAND rm -rf $<TARGET_FILE_DIR:simulate>/../Frameworks/mujoco.framework/mujoco.tbd
      COMMAND rm -rf
              $<TARGET_FILE_DIR:simulate>/../Frameworks/mujoco.framework/Versions/A/libmujoco.dylib
    )
  endif()

  # Do not install if macOS Bundles are created as RPATH is managed manually there.
  if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
    set(_INSTALL_SIMULATE OFF)
  endif()

  if(_INSTALL_SIMULATE)

    include(TargetAddRpath)

    # Add support to RPATH for the samples.
    target_add_rpath(
      TARGETS
      simulate
      INSTALL_DIRECTORY
      "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
      LIB_DIRS
      "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      DEPENDS
      MUJOCO_ENABLE_RPATH
    )

    if(NOT MUJOCO_SIMULATE_USE_SYSTEM_GLFW)
      # We downloaded GLFW. Depending if it is a static or shared LIBRARY we might
      # need to install it.
      get_target_property(MJ_GLFW_LIBRARY_TYPE glfw TYPE)
      if(MJ_GLFW_LIBRARY_TYPE STREQUAL SHARED_LIBRARY)
        install(
          TARGETS glfw
          EXPORT ${PROJECT_NAME}
          RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT simulate
          LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT simulate
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT simulate
          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT simulate
        )
      endif()
    endif()

    install(
      TARGETS simulate
      EXPORT ${PROJECT_NAME}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT simulate
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT simulate
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT simulate
      BUNDLE DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT simulate
      PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT simulate
    )

    # Specify public headers for libmujoco_simulate

    install(
    TARGETS libmujoco_simulate
    EXPORT mujoco
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT simulate
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT simulate
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT simulate
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/simulate" COMPONENT simulate
    )
  endif()
endif()
