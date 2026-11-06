# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OpenUSD's pxrConfig.cmake calls find_dependency(Python3 COMPONENTS Development)
# and requires Python3_VERSION to be defined when building against a different
# Python minor version than the one used to build OpenUSD. When Python3_LIBRARY
# is unset, a directory, or missing on disk (e.g. on manylinux or macOS),
# create a placeholder archive so FindPython3(Development) succeeds.
set(Python3_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
if(NOT Python3_INCLUDE_DIR AND Python3_INCLUDE_DIRS)
  list(GET Python3_INCLUDE_DIRS 0 Python3_INCLUDE_DIR)
endif()
if(NOT Python3_LIBRARY)
  if(Python3_LIBRARY_RELEASE)
    set(Python3_LIBRARY "${Python3_LIBRARY_RELEASE}")
  elseif(Python3_LIBRARIES)
    foreach(_mj_py_lib ${Python3_LIBRARIES})
      if(NOT _mj_py_lib MATCHES "^(optimized|debug|general)$")
        set(Python3_LIBRARY "${_mj_py_lib}")
        break()
      endif()
    endforeach()
  endif()
endif()
if(UNIX AND (NOT Python3_LIBRARY OR IS_DIRECTORY "${Python3_LIBRARY}" OR NOT EXISTS "${Python3_LIBRARY}"))
  set(_dummy_py_lib "${CMAKE_CURRENT_BINARY_DIR}/libpython${Python3_VERSION}.a")
  file(WRITE "${_dummy_py_lib}" "!<arch>\n")
  set(Python3_LIBRARY "${_dummy_py_lib}" CACHE FILEPATH "" FORCE)
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)
elseif(UNIX AND NOT APPLE)
  # Match PyPI's manylinux2014 usd-core wheel, which is built with GCC against
  # the pre-C++11 libstdc++ ABI (_GLIBCXX_USE_CXX11_ABI=0 and
  # ARCH_HAS_GNU_STL_EXTENSIONS).
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0 ARCH_HAS_GNU_STL_EXTENSIONS)
endif()

find_package(
  pxr REQUIRED
  HINTS "${MUJOCO_LIBRARY_DIR}/.." "${MUJOCO_FRAMEWORK_DIR}" "${CMAKE_BINARY_DIR}/_deps/openusd-build"
)

# OpenUSD's generated pxrTargets.cmake hardcodes the Python include paths from
# the host used to build OpenUSD (e.g. C:/Python313/Include) directly on
# imported targets such as python, tf, gf, vt, ar, sdf, pcp, and usd. Strip
# those stale include paths and bind to the active wheel's Python headers/module
# target instead.
set(_mj_pxr_targets ${PXR_LIBRARIES} python boost pxr_boost_python arch)
if(TARGET usd_m)
  get_target_property(_mj_usd_m_deps usd_m INTERFACE_LINK_LIBRARIES)
  if(_mj_usd_m_deps)
    list(APPEND _mj_pxr_targets ${_mj_usd_m_deps})
  endif()
endif()
list(REMOVE_DUPLICATES _mj_pxr_targets)

set(_mj_pxr_own_inc ${PXR_INCLUDE_DIRS})
if(TARGET arch)
  get_target_property(_mj_arch_inc arch INTERFACE_INCLUDE_DIRECTORIES)
  if(_mj_arch_inc)
    list(APPEND _mj_pxr_own_inc ${_mj_arch_inc})
  endif()
endif()
set(_mj_stale_py_inc "")
if(TARGET python)
  foreach(_mj_inc_prop INTERFACE_SYSTEM_INCLUDE_DIRECTORIES INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_mj_py_inc python ${_mj_inc_prop})
    if(_mj_py_inc)
      list(APPEND _mj_stale_py_inc ${_mj_py_inc})
    endif()
  endforeach()
endif()
if(_mj_stale_py_inc)
  list(REMOVE_DUPLICATES _mj_stale_py_inc)
  if(_mj_pxr_own_inc)
    list(REMOVE_ITEM _mj_stale_py_inc ${_mj_pxr_own_inc})
  endif()
  if(Python3_INCLUDE_DIRS)
    list(REMOVE_ITEM _mj_stale_py_inc ${Python3_INCLUDE_DIRS})
  endif()
endif()

foreach(_mj_tgt ${_mj_pxr_targets})
  if(TARGET ${_mj_tgt})
    if(_mj_stale_py_inc)
      foreach(_mj_inc_prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
        get_target_property(_mj_tgt_inc ${_mj_tgt} ${_mj_inc_prop})
        if(_mj_tgt_inc)
          list(REMOVE_ITEM _mj_tgt_inc ${_mj_stale_py_inc})
          set_target_properties(${_mj_tgt} PROPERTIES ${_mj_inc_prop} "${_mj_tgt_inc}")
        endif()
      endforeach()
    endif()
    get_target_property(_mj_tgt_libs ${_mj_tgt} INTERFACE_LINK_LIBRARIES)
    if(_mj_tgt_libs)
      if(UNIX)
        list(FILTER _mj_tgt_libs EXCLUDE REGEX "^Python3::")
      elseif(TARGET Python3::Module)
        list(TRANSFORM _mj_tgt_libs REPLACE "^Python3::.*$" "Python3::Module")
      endif()
      set_target_properties(${_mj_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_mj_tgt_libs}")
    endif()
  endif()
endforeach()
if(TARGET python AND Python3_INCLUDE_DIRS)
  set_property(
    TARGET python APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Python3_INCLUDE_DIRS}
  )
  set_property(
    TARGET python APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${Python3_INCLUDE_DIRS}
  )
endif()

find_library(
  MUJOCO_USD_LIB
  NAMES mujoco_usd libmujoco_usd
  PATHS "${MUJOCO_LIBRARY_DIR}" "${MUJOCO_BIN_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(
  MJC_PHYSICS_LIB
  NAMES mjcPhysics libmjcPhysics
  PATHS "${MUJOCO_LIBRARY_DIR}" "${MUJOCO_BIN_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)

set(USD_PYTHON_COMMON_LIBS
  ${PXR_LIBRARIES}
  usd
  sdf
  tf
  gf
  vt
  arch
  usdGeom
  usdLux
  usdShade
)
if(TARGET python)
  list(APPEND USD_PYTHON_COMMON_LIBS python)
elseif(TARGET pxr_boost_python)
  list(APPEND USD_PYTHON_COMMON_LIBS pxr_boost_python)
endif()

mujoco_pybind11_module(writer writer_module.cc)
target_include_directories(
  writer
  PRIVATE ${Python3_INCLUDE_DIRS}
          ${PXR_INCLUDE_DIRS}
          ${MUJOCO_INCLUDE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(writer PRIVATE -Wno-deprecated)
target_link_libraries(
  writer
  PRIVATE mujoco
          ${MUJOCO_USD_LIB}
          ${USD_PYTHON_COMMON_LIBS}
)

mujoco_pybind11_module(layer_sink layer_sink_module.cc)
target_include_directories(
  layer_sink
  PRIVATE ${Python3_INCLUDE_DIRS}
          ${PXR_INCLUDE_DIRS}
          ${MUJOCO_INCLUDE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(layer_sink PRIVATE -Wno-deprecated)
target_link_libraries(
  layer_sink
  PRIVATE mujoco
          ${MUJOCO_USD_LIB}
          ${USD_PYTHON_COMMON_LIBS}
)

add_library(
  _mjcPhysics MODULE
  MjcPhysics/module.cpp
  MjcPhysics/moduleDeps.cc
  MjcPhysics/wrapActuator.cpp
  MjcPhysics/wrapCollisionAPI.cpp
  MjcPhysics/wrapEqualityAPI.cpp
  MjcPhysics/wrapEqualityConnectAPI.cpp
  MjcPhysics/wrapEqualityJointAPI.cpp
  MjcPhysics/wrapEqualityWeldAPI.cpp
  MjcPhysics/wrapImageableAPI.cpp
  MjcPhysics/wrapJointAPI.cpp
  MjcPhysics/wrapKeyframe.cpp
  MjcPhysics/wrapMaterialAPI.cpp
  MjcPhysics/wrapMeshCollisionAPI.cpp
  MjcPhysics/wrapSceneAPI.cpp
  MjcPhysics/wrapSiteAPI.cpp
  MjcPhysics/wrapTendon.cpp
  MjcPhysics/wrapTokens.cpp
)
set_target_properties(
  _mjcPhysics
  PROPERTIES PREFIX ""
             OUTPUT_NAME "_mjcPhysics"
             CXX_VISIBILITY_PRESET default
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/MjcPhysics"
)
if(WIN32)
  set_target_properties(_mjcPhysics PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
  set_target_properties(_mjcPhysics PROPERTIES SUFFIX ".so")
endif()
target_compile_definitions(
  _mjcPhysics
  PRIVATE MFB_PACKAGE_NAME=mjcPhysics
          MFB_ALT_PACKAGE_NAME=mjcPhysics
          MFB_PACKAGE_MODULE=MjcPhysics
)
target_include_directories(
  _mjcPhysics
  PRIVATE ${Python3_INCLUDE_DIRS}
          ${PXR_INCLUDE_DIRS}
          ${MUJOCO_INCLUDE_DIR}
)
target_compile_options(_mjcPhysics PRIVATE -Wno-deprecated)
target_link_libraries(
  _mjcPhysics
  PRIVATE mujoco
          ${MJC_PHYSICS_LIB}
          ${USD_PYTHON_COMMON_LIBS}
)
if(WIN32)
  target_link_libraries(_mjcPhysics PRIVATE ${Python3_LIBRARIES})
elseif(APPLE)
  target_link_options(_mjcPhysics PRIVATE -Wl,-undefined,dynamic_lookup)
  add_custom_command(
    TARGET _mjcPhysics
    POST_BUILD
    COMMAND
      install_name_tool -change
      $<TARGET_PROPERTY:mujoco,INSTALL_NAME_DIR>/$<TARGET_FILE_NAME:mujoco>
      @rpath/$<TARGET_FILE_NAME:mujoco> $<TARGET_FILE:_mjcPhysics>
    VERBATIM
  )
endif()

if(APPLE)
  set_target_properties(writer layer_sink _mjcPhysics PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  set_property(
    TARGET writer layer_sink
    APPEND PROPERTY BUILD_RPATH "@loader_path/..;@loader_path/../_usd;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs"
  )
  set_property(
    TARGET writer layer_sink
    APPEND PROPERTY INSTALL_RPATH "@loader_path/..;@loader_path/../_usd;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs"
  )
  set_property(
    TARGET _mjcPhysics
    APPEND PROPERTY BUILD_RPATH "@loader_path/../..;@loader_path/../../_usd;@loader_path/../../../pxr;@loader_path/../../../pxr/.dylibs"
  )
  set_property(
    TARGET _mjcPhysics
    APPEND PROPERTY INSTALL_RPATH "@loader_path/../..;@loader_path/../../_usd;@loader_path/../../../pxr;@loader_path/../../../pxr/.dylibs"
  )
elseif(UNIX)
  set_target_properties(writer layer_sink _mjcPhysics PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  set_property(
    TARGET writer layer_sink
    APPEND PROPERTY BUILD_RPATH "$ORIGIN/..;$ORIGIN/../_usd;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs"
  )
  set_property(
    TARGET writer layer_sink
    APPEND PROPERTY INSTALL_RPATH "$ORIGIN/..;$ORIGIN/../_usd;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs"
  )
  set_property(
    TARGET _mjcPhysics
    APPEND PROPERTY BUILD_RPATH "$ORIGIN/../..;$ORIGIN/../../_usd;$ORIGIN/../../../pxr;$ORIGIN/../../../usd_core.libs"
  )
  set_property(
    TARGET _mjcPhysics
    APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../..;$ORIGIN/../../_usd;$ORIGIN/../../../pxr;$ORIGIN/../../../usd_core.libs"
  )
endif()
