# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Global Configuration ---

# Match OpenUSD's C++17 standard and avoid C++20 deprecation warnings on
# OpenUSD headers.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Plugin target names (used for library and plugInfo.json)
set(MJCF_PLUGIN_TARGET_NAME usdMjcf)
set(MJC_PHYSICS_PLUGIN_TARGET_NAME mjcPhysics)

# Installation directory for USD plugins
set(MJ_USD_INSTALL_DIR_LIB ${CMAKE_INSTALL_LIBDIR}/mujoco-usd-resources)

# Common OpenUSD libraries
set(OPENUSD_CORE_LIBS
  ${PXR_LIBRARIES}
  usd ar kind tf gf vt usdShade usdLux usdGeom usdPhysics
)

# We need to set the visibility to default until core type symbol visibility
# is resolved in OpenUSD https://github.com/PixarAnimationStudios/OpenUSD/issues/1475
set(DEFAULT_CXX_VISIBILITY_PROPS
  CXX_VISIBILITY_PRESET default
)

# Pure C++ USD targets do not embed a Python interpreter; disable inline
# Boost.Python VtValue proxies so they do not emit direct references to
# _Py_NoneStruct / _Py_Dealloc when OpenUSD is built with Python enabled.
add_compile_options(-UPXR_PYTHON_SUPPORT_ENABLED)

if(UNIX AND NOT APPLE)
  # OpenUSD (and PyPI's manylinux2014 usd-core wheel) is built against
  # libstdc++ with _GLIBCXX_USE_CXX11_ABI=0 on Linux, so strip any global
  # -stdlib=libc++ / -lc++ flags inherited from the root MuJoCo build.
  # Link -lstdc++ before OpenUSD and hide static libstdc++_nonshared.a symbols
  # so each library embeds its own hidden copy of symbols such as
  # _ZSt28__throw_bad_array_new_lengthv instead of binding to the build-time
  # libusd_ms.so.
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  list(PREPEND OPENUSD_CORE_LIBS -Wl,-lstdc++ -Wl,--exclude-libs,libstdc++_nonshared.a)
  string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  foreach(
    _flag
    "-stdlib=libc++"
    "-nostdlib++"
    "-rtlib=compiler-rt"
    "-Wl,-Bstatic"
    "-lc++"
    "-Wl,-Bdynamic"
  )
    string(REPLACE "${_flag}" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  endforeach()
endif()

# --- Helper Functions ---
function(configure_and_install_usd_plugin_info plugin_name source_subpath install_base_dir)
  # Determine shared library prefix
  if(CMAKE_SHARED_LIBRARY_PREFIX)
    set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  else()
    set(LIB_PREFIX "")
  endif()

  set(PLUG_INFO_LIBRARY_PATH "../../${LIB_PREFIX}${plugin_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/${install_base_dir}/${plugin_name}")
  set(OUTPUT_FILE "${OUTPUT_DIR}/plugInfo.json")

  # Create the output directory if it doesn't exist
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")

  # Configure the plugInfo.json file to replace @PLUG_INFO_LIBRARY_PATH@ instances.
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${source_subpath}/plugInfo.json"
    "${OUTPUT_FILE}"
    @ONLY
  )

  if(WIN32 AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(RUNTIME_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mujoco-usd-resources/${plugin_name}")
    file(MAKE_DIRECTORY "${RUNTIME_OUTPUT_DIR}")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/${source_subpath}/plugInfo.json"
      "${RUNTIME_OUTPUT_DIR}/plugInfo.json"
      @ONLY
    )
  endif()

  install(FILES "${OUTPUT_FILE}"
    DESTINATION "${install_base_dir}/${plugin_name}"
  )
endfunction()

# Function to apply Houdini-specific link directories and include paths
function(configure_houdini_target target_name link_scope include_scope)
  set(HOUDINI_LIBS_DIR "${HOUDINI_HFS_DIR}/../Libraries")
  get_filename_component(HOUDINI_LIBS_DIR "${HOUDINI_LIBS_DIR}" ABSOLUTE) # Normalize the path

  # Assume everyone using Houdini on 3.11 for now.
  set(USD_MJCF_PYTHON_LIB python3.11)
  set(USD_MJCF_PYTHON_LIB_NUMBER python311) # For boost library names
  set(PYTHON_LIB "${HOUDINI_HFS_DIR}/Frameworks/Python.framework/Versions/3.11/Python")
  set(PXR_LIB_PREFIX "pxr_")

  # Houdini-specific PXR and other libraries
  set(HOUDINI_SPECIFIC_LIBS
    ${PXR_LIB_PREFIX}usd
    ${PXR_LIB_PREFIX}ar
    ${PXR_LIB_PREFIX}kind
    ${PXR_LIB_PREFIX}tf
    ${PXR_LIB_PREFIX}gf
    ${PXR_LIB_PREFIX}vt
    ${PXR_LIB_PREFIX}sdf
    ${PXR_LIB_PREFIX}usdShade
    ${PXR_LIB_PREFIX}usdLux
    ${PXR_LIB_PREFIX}usdGeom
    ${PXR_LIB_PREFIX}usdImaging
    ${PXR_LIB_PREFIX}usdPhysics
    tbb
    hboost_${USD_MJCF_PYTHON_LIB_NUMBER}
    ${PYTHON_LIB}
  )

  target_link_directories(${target_name} ${link_scope} ${HOUDINI_LIBS_DIR})
  target_include_directories(${target_name} ${include_scope}
    "${HOUDINI_HFS_DIR}/toolkit/include"
    "${HOUDINI_HFS_DIR}/toolkit/include/python3.11"
  )
  target_link_libraries(${target_name} ${link_scope}
    ${HOUDINI_SPECIFIC_LIBS}
    mujoco
    tinyxml2
  )
endfunction()

## usdMjcf

add_library(${MJCF_PLUGIN_TARGET_NAME} SHARED)
target_sources(${MJCF_PLUGIN_TARGET_NAME} PRIVATE
  plugins/mjcf/mjcf_file_format.cc
  plugins/mjcf/mjcf_file_format.h
)
set_target_properties(${MJCF_PLUGIN_TARGET_NAME} PROPERTIES
  OUTPUT_NAME ${MJCF_PLUGIN_TARGET_NAME}
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  ${DEFAULT_CXX_VISIBILITY_PROPS}
)
target_include_directories(${MJCF_PLUGIN_TARGET_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/plugins"
)

## ----- mjcPhysics -----

add_library(${MJC_PHYSICS_PLUGIN_TARGET_NAME} SHARED)
target_sources(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PRIVATE
  mjcPhysics/actuator.cpp
  mjcPhysics/collisionAPI.cpp
  mjcPhysics/equalityAPI.cpp
  mjcPhysics/equalityConnectAPI.cpp
  mjcPhysics/equalityJointAPI.cpp
  mjcPhysics/equalityWeldAPI.cpp
  mjcPhysics/imageableAPI.cpp
  mjcPhysics/jointAPI.cpp
  mjcPhysics/keyframe.cpp
  mjcPhysics/materialAPI.cpp
  mjcPhysics/meshCollisionAPI.cpp
  mjcPhysics/sceneAPI.cpp
  mjcPhysics/siteAPI.cpp
  mjcPhysics/tendon.cpp
  mjcPhysics/tokens.cpp
)
set_target_properties(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PROPERTIES
  OUTPUT_NAME ${MJC_PHYSICS_PLUGIN_TARGET_NAME}
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  ${DEFAULT_CXX_VISIBILITY_PROPS}
)
target_compile_definitions(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PRIVATE MJCPHYSICS_EXPORTS)
target_include_directories(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

## ----- mujoco_usd -----

add_library(mujoco_usd SHARED
  layer_sink.cc
  utils.cc
  writer.cc
  plugins/mjcf/mujoco_to_usd.cc
  plugins/mjcf/mujoco_to_usd.h
  plugins/mjcf/utils.cc
  plugins/mjcf/utils.h
)
set_target_properties(mujoco_usd PROPERTIES
  OUTPUT_NAME mujoco_usd
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  ${DEFAULT_CXX_VISIBILITY_PROPS}
)
target_include_directories(mujoco_usd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)
target_include_directories(mujoco_usd PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  "${CMAKE_CURRENT_SOURCE_DIR}/plugins"
)

# pxr/usd/tf/hashset.h might include deprecated headers so disable this check
# when compiling with USD enabled.
target_compile_options(
  mujoco_usd
  PRIVATE -Wno-deprecated
)

# Configure RPATH for macOS and UNIX.
if(APPLE)
  set(_MJ_USD_APPLE_RPATH
    "@loader_path;@loader_path/..;@loader_path/../..;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs;@loader_path/../Frameworks;@loader_path/../../Frameworks;${USD_INSTALL_ROOT}"
  )
  set_target_properties(${MJCF_PLUGIN_TARGET_NAME} PROPERTIES
    BUILD_RPATH "${_MJ_USD_APPLE_RPATH}"
    INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../..;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs;@loader_path/../Frameworks;@loader_path/../../Frameworks")

  set_target_properties(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PROPERTIES
    BUILD_RPATH "${_MJ_USD_APPLE_RPATH}"
    INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../..;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs;@loader_path/../Frameworks;@loader_path/../../Frameworks")

  set_target_properties(mujoco_usd PROPERTIES
    BUILD_RPATH "${_MJ_USD_APPLE_RPATH}"
    INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../..;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs;@loader_path/../Frameworks;@loader_path/../../Frameworks")
elseif(UNIX)
  set(_MJ_USD_UNIX_RPATH
    "$ORIGIN;$ORIGIN/..;$ORIGIN/../..;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs;${USD_INSTALL_ROOT}"
  )
  set_target_properties(${MJCF_PLUGIN_TARGET_NAME} PROPERTIES
    BUILD_RPATH "${_MJ_USD_UNIX_RPATH}"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../..;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs")

  set_target_properties(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PROPERTIES
    BUILD_RPATH "${_MJ_USD_UNIX_RPATH}"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../..;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs")

  set_target_properties(mujoco_usd PROPERTIES
    BUILD_RPATH "${_MJ_USD_UNIX_RPATH}"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../..;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs")
endif()

target_link_libraries(${MJC_PHYSICS_PLUGIN_TARGET_NAME} PRIVATE
  ${OPENUSD_CORE_LIBS}
)

target_link_libraries(mujoco_usd PUBLIC
  ${OPENUSD_CORE_LIBS}
  mujoco
  ${MJC_PHYSICS_PLUGIN_TARGET_NAME}
)

target_link_libraries(${MJCF_PLUGIN_TARGET_NAME} PRIVATE
  ${OPENUSD_CORE_LIBS}
  mujoco
  mujoco_usd
  tinyxml2
  ${MJC_PHYSICS_PLUGIN_TARGET_NAME}
)

## ----- Newton USD Schemas (codeless plugin) -----

function(install_newton_usd_plugin install_base_dir)
  include(FetchContent)

  set(NEWTON_USD_SCHEMAS_TAG v0.4.0)
  set(NEWTON_TARBALL_PATH
      "${MUJOCO_CMAKE_DEP_CACHE}/newton-usd-schemas-${NEWTON_USD_SCHEMAS_TAG}.tar.gz"
  )
  if(MUJOCO_CMAKE_DEP_CACHE AND EXISTS "${NEWTON_TARBALL_PATH}")
    FetchContent_Declare(
      newton-usd-schemas
      URL "${NEWTON_TARBALL_PATH}"
    )
  else()
    FetchContent_Declare(
      newton-usd-schemas
      GIT_REPOSITORY https://github.com/newton-physics/newton-usd-schemas.git
      GIT_TAG ${NEWTON_USD_SCHEMAS_TAG}
      GIT_SHALLOW TRUE
      UPDATE_DISCONNECTED TRUE
    )
  endif()

  FetchContent_GetProperties(newton-usd-schemas)
  if(NOT newton-usd-schemas_POPULATED)
    FetchContent_Populate(newton-usd-schemas)
  endif()

  set(NEWTON_USD_DIR "${newton-usd-schemas_SOURCE_DIR}/newton_usd_schemas")
  if(NOT EXISTS "${NEWTON_USD_DIR}")
    message(FATAL_ERROR "newton_usd_schemas directory not found in fetched repository")
  endif()

  set(NEWTON_BUILD_DIR "${CMAKE_BINARY_DIR}/${install_base_dir}/newton")
  file(MAKE_DIRECTORY "${NEWTON_BUILD_DIR}")
  configure_file("${NEWTON_USD_DIR}/plugInfo.json" "${NEWTON_BUILD_DIR}/plugInfo.json" COPYONLY)
  configure_file("${NEWTON_USD_DIR}/generatedSchema.usda" "${NEWTON_BUILD_DIR}/generatedSchema.usda" COPYONLY)

  if(WIN32 AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(NEWTON_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mujoco-usd-resources/newton")
    file(MAKE_DIRECTORY "${NEWTON_RUNTIME_DIR}")
    configure_file("${NEWTON_USD_DIR}/plugInfo.json" "${NEWTON_RUNTIME_DIR}/plugInfo.json" COPYONLY)
    configure_file("${NEWTON_USD_DIR}/generatedSchema.usda" "${NEWTON_RUNTIME_DIR}/generatedSchema.usda" COPYONLY)
  endif()

  install(FILES "${NEWTON_BUILD_DIR}/plugInfo.json"
    DESTINATION "${install_base_dir}/newton"
  )
  install(FILES "${NEWTON_BUILD_DIR}/generatedSchema.usda"
    DESTINATION "${install_base_dir}/newton"
  )
endfunction()

## Installation

# Generate and install plugInfo.json for each plugin
configure_and_install_usd_plugin_info(
  ${MJCF_PLUGIN_TARGET_NAME}
  plugins/mjcf
  ${MJ_USD_INSTALL_DIR_LIB}
)
configure_and_install_usd_plugin_info(
  ${MJC_PHYSICS_PLUGIN_TARGET_NAME}
  mjcPhysics
  ${MJ_USD_INSTALL_DIR_LIB}
)
if(WIN32 AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mjcPhysics/generatedSchema.usda"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mujoco-usd-resources/mjcPhysics/generatedSchema.usda"
    COPYONLY
  )
endif()

# Install specific resources that are not plugInfo.json (like generatedSchema.usda)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/mjcPhysics/generatedSchema.usda"
  "${CMAKE_BINARY_DIR}/${MJ_USD_INSTALL_DIR_LIB}/mjcPhysics/generatedSchema.usda"
  COPYONLY
)
install(FILES
  mjcPhysics/generatedSchema.usda
  DESTINATION ${MJ_USD_INSTALL_DIR_LIB}/mjcPhysics
)

install_newton_usd_plugin(
  ${MJ_USD_INSTALL_DIR_LIB}
)

# Install shared libraries and public USD headers
install(DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/mujoco/experimental/usd"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mujoco/experimental"
)
install(TARGETS
  mujoco_usd
  ${MJCF_PLUGIN_TARGET_NAME}
  ${MJC_PHYSICS_PLUGIN_TARGET_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
