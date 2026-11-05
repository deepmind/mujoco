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

set(MUJOCO_CMAKE_MIN_REQ "3.16")
cmake_minimum_required(VERSION ${MUJOCO_CMAKE_MIN_REQ})

# Make CMAKE_C_VISIBILITY_PRESET work properly.
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)
# Avoid BUILD_SHARED_LIBS getting overridden by an option() in ccd.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  mujoco
  VERSION 4.0.0
  DESCRIPTION "MuJoCo Physics Simulator"
  HOMEPAGE_URL "https://mujoco.org"
)

enable_language(C)
enable_language(CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if(NOT EMSCRIPTEN)
  option(MUJOCO_BUILD_EXAMPLES "Build samples for MuJoCo" ON)
  option(MUJOCO_BUILD_SIMULATE "Build simulate library for MuJoCo" ON)
  option(MUJOCO_BUILD_STUDIO "Build studio library for MuJoCo" OFF)
  option(MUJOCO_BUILD_TESTS "Build tests for MuJoCo" ON)
  option(MUJOCO_TEST_PYTHON_UTIL "Build and test utility libraries for Python bindings" ON)
  option(MUJOCO_WITH_USD "Build with OpenUSD" OFF)
  option(MUJOCO_USE_FILAMENT "Use filament rendering" OFF)
  # Allows filament renderer to be used with mjr API; disables src/render to prevent conflicts.
  option(MUJOCO_USE_FILAMENT_MJR_COMPAT "Use filament mjr compatibility layer" OFF)
endif()

if(EMSCRIPTEN)
  option(MUJOCO_BUILD_TESTS_WASM "Build tests for WASM bindings" ON)
  option(MUJOCO_BUILD_STUDIO "Build studio for MuJoCo (WASM)" OFF)
  option(MUJOCO_USE_FILAMENT "Use filament rendering" OFF)
  option(MUJOCO_WASM_THREADS "Build with multi-threading support" ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -fexceptions")
  if(MUJOCO_WASM_THREADS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  endif()

  # Filament uses the WASM variable (not EMSCRIPTEN) to identify web builds.
  # Without this, it falls into the LINUX path and tries to compile with
  # futex, X11, Vulkan, etc.
  if(MUJOCO_USE_FILAMENT)
    set(WASM ON CACHE BOOL "Filament WASM mode" FORCE)
  endif()

  # Automatically generate host tool imports for Filament cross-compilation.
  # When WASM is true, Filament hardcodes the path to:
  #   ${FILAMENT}/${IMPORT_EXECUTABLES_DIR}/ImportExecutables-Release.cmake
  # So we write our file with that exact name and set IMPORT_EXECUTABLES_DIR
  # to point at CMAKE_BINARY_DIR (which is ../../ relative to filament-src).
  set(MUJOCO_NATIVE_BUILD_DIR "${PROJECT_SOURCE_DIR}/build-host" CACHE PATH "Path to native build directory containing host tools")
  find_program(MATC_EXE matc PATHS "${MUJOCO_NATIVE_BUILD_DIR}/bin" NO_DEFAULT_PATH)
  find_program(RESGEN_EXE resgen PATHS "${MUJOCO_NATIVE_BUILD_DIR}/bin" NO_DEFAULT_PATH)
  find_program(CMGEN_EXE cmgen PATHS "${MUJOCO_NATIVE_BUILD_DIR}/bin" NO_DEFAULT_PATH)

  if(MATC_EXE AND RESGEN_EXE AND CMGEN_EXE)
    message(STATUS "Found host tools in ${MUJOCO_NATIVE_BUILD_DIR}/bin")
    set(IMPORT_EXECUTABLES_FILE "${CMAKE_BINARY_DIR}/ImportExecutables-Release.cmake")
    file(WRITE "${IMPORT_EXECUTABLES_FILE}"
      "add_executable(matc IMPORTED)\n"
      "set_property(TARGET matc PROPERTY IMPORTED_LOCATION \"${MATC_EXE}\")\n"
      "add_executable(resgen IMPORTED)\n"
      "set_property(TARGET resgen PROPERTY IMPORTED_LOCATION \"${RESGEN_EXE}\")\n"
      "add_executable(cmgen IMPORTED)\n"
      "set_property(TARGET cmgen PROPERTY IMPORTED_LOCATION \"${CMGEN_EXE}\")\n"
    )
    # Filament's WASM path resolves: ${FILAMENT}/${IMPORT_EXECUTABLES_DIR}/ImportExecutables-Release.cmake
    # FILAMENT = _deps/filament-src, so ../../ resolves to CMAKE_BINARY_DIR.
    set(IMPORT_EXECUTABLES_DIR "../../" CACHE PATH "" FORCE)
  else()
    message(WARNING "Host tools (matc, resgen, cmgen) not found in ${MUJOCO_NATIVE_BUILD_DIR}/bin. WASM build of Studio might fail.")
  endif()
endif()

if(APPLE AND (MUJOCO_BUILD_EXAMPLES OR MUJOCO_BUILD_SIMULATE OR MUJOCO_USE_FILAMENT))
  # NOTE: OBJC* languages are proactively enabled when using Filament to
  # workaround a CMake bug triggered by Filament recently adding:
  #   set(EXTRA_LANGS OBJC OBJCXX)
  # See also: https://gitlab.kitware.com/cmake/cmake/-/work_items/24104
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

include(MujocoOptions)
include(MujocoMacOS)
include(MujocoDependencies)

set(MUJOCO_HEADERS
    include/mujoco/mjassert.h
    include/mujoco/mjbatch.h
    include/mujoco/mjdata.h
    include/mujoco/mjexport.h
    include/mujoco/mjmacro.h
    include/mujoco/mjmodel.h
    include/mujoco/mjplugin.h
    include/mujoco/mjrender.h
    include/mujoco/mjrfilament.h
    include/mujoco/mjsan.h
    include/mujoco/mjspec.h
    include/mujoco/mjspecmacro.h
    include/mujoco/mjtype.h
    include/mujoco/mjui.h
    include/mujoco/mjvisualize.h
    include/mujoco/mjxmacro.h
    include/mujoco/mujoco.h
)

# Add metadata to mujoco.dll when building on Windows.
if(WIN32)
  set(MUJOCO_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/dist/mujoco.rc)
else()
  set(MUJOCO_RESOURCE_FILES "")
endif()

# Emscripten does not support SHARED libs
if(NOT EMSCRIPTEN)
  find_package(Threads REQUIRED)
  add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})
else()
  add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})
endif()

target_include_directories(
  mujoco
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
  PRIVATE src
)

if(NOT EMSCRIPTEN)
  add_subdirectory(plugin/actuator)
  add_subdirectory(plugin/sensor)
  add_subdirectory(plugin/sdf)
endif()
add_subdirectory(plugin/gmsh_decoder)
add_subdirectory(plugin/obj_decoder)
add_subdirectory(plugin/stl_decoder)

add_subdirectory(src/engine)
add_subdirectory(src/user)
add_subdirectory(src/xml)
add_subdirectory(src/xml/mjz)
add_subdirectory(src/batch)

add_subdirectory(src/render/noop)

if(NOT EMSCRIPTEN AND NOT MUJOCO_USE_FILAMENT_MJR_COMPAT)
  add_subdirectory(src/render/classic)
  add_subdirectory(src/ui)
endif()

if(MUJOCO_USE_FILAMENT)
  add_subdirectory(src/render/filament)
  if(MUJOCO_USE_FILAMENT_MJR_COMPAT)
    add_subdirectory(src/experimental/filament)
  endif()
endif()

if(EMSCRIPTEN)
  add_subdirectory(wasm)
endif()


target_compile_definitions(mujoco PRIVATE _GNU_SOURCE CCD_STATIC_DEFINE MUJOCO_DLL_EXPORTS -DMC_IMPLEM_ENABLE)
if(MUJOCO_ENABLE_AVX_INTRINSICS)
  target_compile_definitions(mujoco PUBLIC mjUSEPLATFORMSIMD)
endif()

target_compile_options(
  mujoco
  PRIVATE ${AVX_COMPILE_OPTIONS}
          ${MUJOCO_MACOS_COMPILE_OPTIONS}
          ${EXTRA_COMPILE_OPTIONS}
          ${MUJOCO_CXX_FLAGS}
)
target_link_options(
  mujoco
  PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)

target_link_libraries(
  mujoco
  PRIVATE ccd
          lodepng
          qhullstatic_r
          tinyobjloader
          tinyxml2
)

if(NOT EMSCRIPTEN)
  target_link_libraries(mujoco PRIVATE Threads::Threads)
  if(APPLE)
    target_link_options(
      mujoco PRIVATE "-Wl,-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/mujoco.unexported_symbols"
    )
    set_property(
      TARGET mujoco
      APPEND
      PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mujoco.unexported_symbols"
    )
  elseif(UNIX)
    target_link_options(
      mujoco PRIVATE "-Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/mujoco.lds"
    )
    set_property(
      TARGET mujoco
      APPEND
      PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mujoco.lds"
    )
  endif()
endif()

set_target_properties(
  mujoco PROPERTIES VERSION "${mujoco_VERSION}" PUBLIC_HEADER "${MUJOCO_HEADERS}"
)

if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND MUJOCO_ENABLE_LTO)
  # Keep whole-program optimization (/GL + /LTCG) on the core mujoco DLL even
  # when global CMAKE_INTERPROCEDURAL_OPTIMIZATION is disabled in CI: turning
  # off /GL on mujoco exposes an MSVC /EHsc + longjmp stack-unwinding crash in
  # SetConstTest.SleepingNotAllowed.
  set_target_properties(mujoco PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# CMake's built-in FRAMEWORK option doesn't give us control over the dylib name inside the
# Framework. We instead make our own Framework here.
if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
  set(TAPI
      "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/tapi"
  )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/dist/Info.plist.framework.in
    ${CMAKE_CURRENT_SOURCE_DIR}/dist/Info.framework.plist
  )
  set_target_properties(
    mujoco
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mujoco.framework/Versions/A"
               BUILD_WITH_INSTALL_NAME_DIR TRUE
               INSTALL_NAME_DIR "@rpath/mujoco.framework/Versions/A"
  )
  add_custom_command(
    TARGET mujoco
    POST_BUILD
    COMMAND mkdir -p $<TARGET_FILE_DIR:mujoco>/Headers
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && cp ${MUJOCO_HEADERS} $<TARGET_FILE_DIR:mujoco>/Headers
    COMMAND mkdir -p $<TARGET_FILE_DIR:mujoco>/Modules
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/dist/module.modulemap $<TARGET_FILE_DIR:mujoco>/Modules
    COMMAND mkdir -p $<TARGET_FILE_DIR:mujoco>/Resources
    COMMAND mv ${CMAKE_CURRENT_SOURCE_DIR}/dist/Info.framework.plist
            $<TARGET_FILE_DIR:mujoco>/Resources/Info.plist
    COMMAND ${TAPI} stubify $<TARGET_FILE:mujoco> -o $<TARGET_FILE_DIR:mujoco>/mujoco.tbd
    COMMAND ln -fhs A $<TARGET_FILE_DIR:mujoco>/../Current
    COMMAND ln -fhs Versions/Current/mujoco.tbd $<TARGET_FILE_DIR:mujoco>/../../mujoco.tbd
    COMMAND ln -fhs Versions/Current/Headers $<TARGET_FILE_DIR:mujoco>/../../Headers
    COMMAND ln -fhs Versions/Current/Modules $<TARGET_FILE_DIR:mujoco>/../../Modules
    COMMAND ln -fhs Versions/Current/Resources $<TARGET_FILE_DIR:mujoco>/../../Resources
    COMMAND_EXPAND_LISTS
  )
endif()

# Add a namespace alias to mujoco to be used by the examples.
# This simulates the install interface when building with sources.
add_library(mujoco::mujoco ALIAS mujoco)

add_subdirectory(model)

# `simulate` defines a macro, embed_in_bundle, that's used by `sample`, so that
# subdirectory needs to be added first.
# TODO: Remove this order dependency.
if(MUJOCO_BUILD_SIMULATE)
  add_subdirectory(simulate)
endif()

if(MUJOCO_BUILD_STUDIO)
  add_subdirectory(studio)
endif()

if(MUJOCO_BUILD_EXAMPLES)
  add_subdirectory(sample)
endif()

if(MUJOCO_WITH_USD)
  include(third_party_deps/openusd)

  get_filename_component(USD_INSTALL_ROOT "${pxr_DIR}/lib" ABSOLUTE)
  install(
    DIRECTORY ${USD_INSTALL_ROOT}/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  add_subdirectory(src/experimental/usd)
  add_subdirectory(plugin/usd_decoder)
endif()

if(BUILD_TESTING AND MUJOCO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
  if(MUJOCO_TEST_PYTHON_UTIL)
    add_subdirectory(python/mujoco/util)
  endif()
endif()

if(NOT EMSCRIPTEN AND NOT (APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS))
  set(MUJOCO_TARGETS mujoco)

  # Install the libraries.
  install(
    TARGETS ${MUJOCO_TARGETS}
    EXPORT ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT dev
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mujoco" COMPONENT dev
  )

  set(CONFIG_PACKAGE_LOCATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

  # Generate and install the mujocoTargets.cmake file. This defines the targets as
  # IMPORTED libraries for downstream users.
  install(
    EXPORT ${PROJECT_NAME}
    DESTINATION ${CONFIG_PACKAGE_LOCATION}
    NAMESPACE mujoco::
    FILE "${PROJECT_NAME}Targets.cmake"
  )

  include(CMakePackageConfigHelpers)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${mujoco_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CONFIG_PACKAGE_LOCATION}
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
          DESTINATION ${CONFIG_PACKAGE_LOCATION}
  )

  # Install also models into share folder.
  install(
    DIRECTORY model
    DESTINATION "${CMAKE_INSTALL_DATADIR}/mujoco"
    PATTERN "CMakeLists.txt" EXCLUDE
  )
endif()

if(MUJOCO_REQUIRED_TARGETS)
  foreach(target IN LISTS MUJOCO_REQUIRED_TARGETS)
    if(NOT TARGET ${target})
      message(
        FATAL_ERROR
        "Required target '${target}' is not defined. "
        "This target is required by the internal packaging workflow."
      )
    endif()
  endforeach()
endif()
