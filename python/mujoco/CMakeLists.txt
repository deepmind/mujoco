# Copyright 2022 DeepMind Technologies Limited
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

project(mujoco_python)
cmake_minimum_required(VERSION 3.15)

option(MUJOCO_PYTHON_BUILD_CORE "Build core MuJoCo Python extensions." ON)
option(MUJOCO_PYTHON_BUILD_STUDIO "Build Studio and Filament Python extensions." OFF)

# Support new IN_LIST if() operator.
set(CMAKE_POLICY_DEFAULT_CMP0057 NEW)
# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJCXX)
endif()

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1927)
  set(CMAKE_CXX_STANDARD 20) # For forceinline lambdas.
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# TODO(stunya) Figure out why this is required.
separate_arguments(CMDLINE_LINK_OPTIONS UNIX_COMMAND ${CMAKE_SHARED_LINKER_FLAGS})
add_link_options(${CMDLINE_LINK_OPTIONS})

if(MSVC)
  add_compile_options(/Gy /Gw /Oi)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(/MP)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fdata-sections -ffunction-sections)
endif()

include(MujocoLinkOptions)
get_mujoco_extra_link_options(EXTRA_LINK_OPTIONS)
list(FILTER EXTRA_LINK_OPTIONS EXCLUDE REGEX "STACK")
list(FILTER EXTRA_LINK_OPTIONS EXCLUDE REGEX "stack")
add_link_options(${EXTRA_LINK_OPTIONS})

include(MujocoMacOS)
enforce_mujoco_macosx_min_version()

if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

include(MujocoHarden)
add_compile_options("${MUJOCO_HARDEN_COMPILE_OPTIONS}")
add_link_options("${MUJOCO_HARDEN_LINK_OPTIONS}")

find_package(Python3 COMPONENTS Interpreter Development)

# Detect free-threaded Python and enable Py_GIL_DISABLED.
# This is needed because we have some STATIC libraries that do not automatically
# get the Py_GIL_DISABLED macro definition through pybind11.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED') or 0)"
  OUTPUT_VARIABLE PYTHON_GIL_DISABLED
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYTHON_GIL_DISABLED EQUAL "1")
  message(STATUS "Free-threaded Python detected, enabling Py_GIL_DISABLED")
  add_compile_definitions(Py_GIL_DISABLED)
endif()

include(FindOrFetch)

# ==================== MUJOCO LIBRARY ==========================================
if(NOT TARGET mujoco)
  add_library(
    mujoco
    SHARED
    IMPORTED
    GLOBAL
  )
  if(APPLE)
    # On macOS, check if we are using mujoco.framework first.
    # Framework headers are searched differently from normal headers.
    # We need to use -F instead of the usual target_include_directories.
    find_path(MUJOCO_FRAMEWORK mujoco.Framework HINTS ${MUJOCO_FRAMEWORK_DIR})
    if(MUJOCO_FRAMEWORK)
      message("MuJoCo framework is at ${MUJOCO_FRAMEWORK}/mujoco.framework")
      set(MUJOCO_LIBRARY
          ${MUJOCO_FRAMEWORK}/mujoco.framework/Versions/A/libmujoco.4.0.0.dylib
      )
      target_compile_options(mujoco INTERFACE -F${MUJOCO_FRAMEWORK})
      set(MUJOCO_INCLUDE "${MUJOCO_FRAMEWORK}/mujoco.framework/Versions/A/Headers")
    endif()
  endif()

  if(NOT MUJOCO_FRAMEWORK)
    find_library(
      MUJOCO_LIBRARY mujoco mujoco.4.0.0 HINTS ${MUJOCO_LIBRARY_DIR} REQUIRED
    )
    find_path(MUJOCO_INCLUDE mujoco/mujoco.h HINTS ${MUJOCO_INCLUDE_DIR} REQUIRED)
    message("MuJoCo is at ${MUJOCO_LIBRARY}")
    message("MuJoCo headers are at ${MUJOCO_INCLUDE}")
    target_include_directories(mujoco INTERFACE "${MUJOCO_INCLUDE}")
  endif()

  if(WIN32)
    set_target_properties(mujoco PROPERTIES IMPORTED_IMPLIB "${MUJOCO_LIBRARY}")
  else()
    set_target_properties(mujoco PROPERTIES IMPORTED_LOCATION "${MUJOCO_LIBRARY}")
  endif()

  if(APPLE)
    execute_process(
      COMMAND otool -XD ${MUJOCO_LIBRARY}
      COMMAND head -n 1
      COMMAND xargs dirname
      COMMAND xargs echo -n
      OUTPUT_VARIABLE MUJOCO_INSTALL_NAME_DIR
    )
    set_target_properties(mujoco PROPERTIES INSTALL_NAME_DIR "${MUJOCO_INSTALL_NAME_DIR}")
  elseif(UNIX)
    execute_process(
      COMMAND objdump -p ${MUJOCO_LIBRARY}
      COMMAND grep SONAME
      COMMAND grep -Po [^\\s]+$
      COMMAND xargs echo -n
      OUTPUT_VARIABLE MUJOCO_SONAME
    )
    set_target_properties(mujoco PROPERTIES IMPORTED_SONAME "${MUJOCO_SONAME}")
  endif()
  add_library(mujoco::mujoco ALIAS mujoco)
endif()

# =================== MUJOCO PLATFORM STATIC LIBRARY ==========================
if(MUJOCO_PYTHON_BUILD_STUDIO AND NOT TARGET mujoco::platform)
  set(MUJOCO_PLATFORM_HINTS ${MUJOCO_LIBRARY_DIR})
  if(MUJOCO_FRAMEWORK)
    list(APPEND MUJOCO_PLATFORM_HINTS "${MUJOCO_FRAMEWORK}/mujoco.framework/Versions/A")
  endif()
  find_library(
    MUJOCO_PLATFORM_LIBRARY mujoco_studio_platform HINTS ${MUJOCO_PLATFORM_HINTS}
  )
  if(MUJOCO_PLATFORM_LIBRARY)
    message(STATUS "Found mujoco_studio_platform library at ${MUJOCO_PLATFORM_LIBRARY}")
    add_library(mujoco_studio_platform STATIC IMPORTED GLOBAL)
    set_target_properties(mujoco_studio_platform PROPERTIES IMPORTED_LOCATION "${MUJOCO_PLATFORM_LIBRARY}")
    add_library(mujoco::platform ALIAS mujoco_studio_platform)

    set(PLATFORM_INCLUDES "${MUJOCO_INCLUDE}" "${MUJOCO_INCLUDE}/mujoco" "${MUJOCO_INCLUDE}/SDL2")
    set_target_properties(mujoco_studio_platform PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PLATFORM_INCLUDES}"
    )
    target_compile_features(mujoco_studio_platform INTERFACE cxx_std_20)

    set(PLATFORM_DEPS)
    set(dep_libs
      mujoco_filament
      dear_imgui
      dear_imgui_SDL2
      dear_imgui_OpenGL3
      dear_imgui_stdlib
      implot
      filament
      backend
      glfw3
      image
      ktxreader
      utils
      math
      bluegl
      bluevk
      filabridge
      filaflat
      filamat
      matlang
      shaders
      smol-v
      glslang
      SPIRV
      SPIRV-Tools
      SPIRV-Tools-opt
      basis_encoder
      basis_transcoder
      zstd
      webp
      sharpyuv
      z
    )

   set(platform_system_libs)
    if(WIN32)
      list(APPEND dep_libs SDL2-static)
      set(platform_system_libs winmm setupapi version imm32 opengl32)
    else()
      list(APPEND dep_libs SDL2)
      if(UNIX AND NOT APPLE)
        set(platform_system_libs pthread dl rt)
      elseif(APPLE)
        set(platform_system_libs
          "-framework Cocoa"
          "-framework IOKit"
          "-framework Carbon"
          "-framework ForceFeedback"
          "-framework CoreVideo"
          "-framework CoreAudio"
          "-framework AudioToolbox"
          "-framework Metal"
          "-framework GameController"
          "-framework CoreHaptics"
          "-framework AVFoundation"
          "-framework QuartzCore"
        )
      endif()
    endif()

    foreach(dep_lib IN LISTS dep_libs)
      find_library(DEP_PATH_${dep_lib} ${dep_lib} HINTS ${MUJOCO_PLATFORM_HINTS})
      if(DEP_PATH_${dep_lib})
        add_library(mujoco::dep::${dep_lib} STATIC IMPORTED GLOBAL)
        set_target_properties(mujoco::dep::${dep_lib} PROPERTIES
          IMPORTED_LOCATION "${DEP_PATH_${dep_lib}}"
          INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE}"
        )
        list(APPEND PLATFORM_DEPS mujoco::dep::${dep_lib})
      endif()
    endforeach()

    list(APPEND PLATFORM_DEPS ${platform_system_libs})

    set_target_properties(mujoco_studio_platform PROPERTIES INTERFACE_LINK_LIBRARIES "${PLATFORM_DEPS}")
  endif()
endif()

# ==================== ABSEIL ==================================================
set(MUJOCO_PYTHON_ABSL_TARGETS absl::core_headers absl::flat_hash_map absl::span)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  absl
  LIBRARY_NAME
  abseil-cpp
  GIT_REPO
  https://github.com/abseil/abseil-cpp
  GIT_TAG
  2065f4ded0558c6f89fee67c8e5228feb4eb960e # LTS 20260817.0
  TARGETS
  ${MUJOCO_PYTHON_ABSL_TARGETS}
  EXCLUDE_FROM_ALL
)
foreach(absl_target IN ITEMS ${MUJOCO_PYTHON_ABSL_TARGETS})
  get_target_property(absl_target_aliased ${absl_target} ALIASED_TARGET)
  if(absl_target_aliased)
    set(absl_target ${absl_target_aliased})
  endif()
  get_target_property(absl_target_type ${absl_target} TYPE)
  if(NOT
     ${absl_target_type}
     STREQUAL
     "INTERFACE_LIBRARY"
  )
    target_compile_options(${absl_target} PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(${absl_target} PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
  endif()
endforeach()

# ==================== EIGEN ===================================================
add_compile_definitions(EIGEN_MPL2_ONLY)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  Eigen3
  LIBRARY_NAME
  eigen
  GIT_REPO
  https://gitlab.com/libeigen/eigen
  GIT_TAG
  1da510b8155f8fa81924616dcc517ecc4202e4d3
  TARGETS
  Eigen3::Eigen
  EXCLUDE_FROM_ALL
)

# ==================== PYBIND11 ================================================
option(MUJOCO_PYTHON_USE_SYSTEM_PYBIND11 "Use installed pybind11 version." OFF)
findorfetch(
  USE_SYSTEM_PACKAGE
  MUJOCO_PYTHON_USE_SYSTEM_PYBIND11
  PACKAGE_NAME
  pybind11
  LIBRARY_NAME
  pybind11
  GIT_REPO
  https://github.com/pybind/pybind11
  GIT_TAG
  97bf890db679505a14dfe547a5e77bb2bd05dc90 # v3.1.0
  TARGETS
  pybind11::pybind11_headers
  EXCLUDE_FROM_ALL
)

# ==================== MUJOCO PYTHON BINDINGS ==================================
add_subdirectory(util)

add_library(mujoco_private_api_header INTERFACE)
target_sources(mujoco_private_api_header INTERFACE private.h)
set_target_properties(mujoco_private_api_header PROPERTIES PUBLIC_HEADER private.h)
target_link_libraries(mujoco_private_api_header INTERFACE mujoco)

add_library(raw INTERFACE)
target_sources(raw INTERFACE raw.h)
set_target_properties(raw PROPERTIES PUBLIC_HEADER raw.h)
target_link_libraries(raw INTERFACE mujoco)

add_library(structs_header INTERFACE)
target_sources(
  structs_header
  INTERFACE indexer_xmacro.h
            indexers.h
            structs.h
)
set_target_properties(structs_header PROPERTIES PUBLIC_HEADER structs.h)
target_include_directories(structs_header INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  structs_header
  INTERFACE absl::flat_hash_map
            absl::span
            crossplatform
            mujoco
            raw
)

include(CheckAvxSupport)
get_avx_compile_options(AVX_COMPILE_OPTIONS)

macro(mujoco_pybind11_module name)
  pybind11_add_module(${name} ${ARGN})
  target_compile_options(${name} PRIVATE ${AVX_COMPILE_OPTIONS})
  if(NOT MSVC)
    target_compile_options(
      ${name}
      PRIVATE -Werror
              -Wall
              -Wimplicit-fallthrough
              -Wunused
              -Wno-int-in-bool-context
              -Wno-sign-compare
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # Only allow fallthrough annotation via __attribute__.
      target_compile_options(${name} PRIVATE -Wimplicit-fallthrough=5)

      # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109561
      if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11)
        target_compile_options(${name} PRIVATE -Wno-maybe-uninitialized)
      endif()
    endif()
  endif()
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  if(APPLE)
    add_custom_command(
      TARGET ${name}
      POST_BUILD
      COMMAND
        install_name_tool -change
        $<TARGET_PROPERTY:mujoco,INSTALL_NAME_DIR>/$<TARGET_FILE_NAME:mujoco>
        @rpath/$<TARGET_FILE_NAME:mujoco> -add_rpath @loader_path $<TARGET_FILE:${name}>
    )
  elseif(NOT WIN32)
    set_target_properties(${name} PROPERTIES INSTALL_RPATH $ORIGIN BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
endmacro()

if(MUJOCO_PYTHON_BUILD_CORE)
  set(SIMULATE_BUILD_EXECUTABLE OFF)
  set(SIMULATE_GLFW_DYNAMIC_SYMBOLS ON)
  add_subdirectory(simulate)

  # enum_traits.h, function_traits.h, specs.cc.inc and
  # render_filament_generated.cc.inc are pre-generated into the sdist by
  # make_sdist.sh, so only the EXISTS branches below are taken by any build we
  # run: CI and Kokoro both build the wheel from an sdist.
  #
  # The else() branches are stale and are not known to work. The generators do
  # `from mujoco.introspect import ...`, which needs the *parent* of the mujoco
  # package on PYTHONPATH, not the package itself as set below; and importing
  # mujoco.introspect executes mujoco/__init__.py, which imports the native
  # _specs and _structs extensions that do not exist yet at codegen time. Making
  # these branches usable therefore needs more than a corrected PYTHONPATH --
  # see the TODO in .github/workflows/README.md.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/enum_traits.h)
    add_library(enum_traits INTERFACE)
    target_sources(enum_traits INTERFACE enum_traits.h)
  else()
    add_custom_command(
      OUTPUT enum_traits.h
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${mujoco_SOURCE_DIR}/mujoco/python/mujoco ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_enum_traits.py > enum_traits.h
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_enum_traits.py
    )
    add_library(enum_traits INTERFACE)
    target_sources(enum_traits INTERFACE enum_traits.h)
    target_include_directories(
      enum_traits INTERFACE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
  target_link_libraries(enum_traits INTERFACE mujoco absl::core_headers)

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/function_traits.h)
    add_library(function_traits INTERFACE)
    target_sources(function_traits INTERFACE function_traits.h)
  else()
    add_custom_command(
      OUTPUT function_traits.h
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${mujoco_SOURCE_DIR}/mujoco/python/mujoco ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_function_traits.py > function_traits.h
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_function_traits.py
    )
    add_library(function_traits INTERFACE)
    target_sources(function_traits INTERFACE function_traits.h)
    target_include_directories(
      function_traits INTERFACE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
  target_link_libraries(function_traits INTERFACE mujoco absl::core_headers)

  add_library(errors_header INTERFACE)
  target_sources(errors_header INTERFACE errors.h)
  set_target_properties(errors_header PROPERTIES PUBLIC_HEADER errors.h)
  target_link_libraries(
    errors_header
    INTERFACE crossplatform
              func_wrap
              mujoco
              mujoco_private_api_header
  )

  add_library(structs_wrappers STATIC
    structs_wrappers.cc
    serialization.h
    indexers.cc
  )
  target_include_directories(structs_wrappers PRIVATE ${Python3_INCLUDE_DIRS})
  target_link_libraries(
    structs_wrappers
    PRIVATE absl::flat_hash_map
            absl::span
            crossplatform
            mujoco
            raw
            structs_header
            pybind11::headers
            Eigen3::Eigen
  )

  add_library(functions_header INTERFACE)
  target_sources(functions_header INTERFACE functions.h)
  set_target_properties(functions_header PROPERTIES PUBLIC_HEADER functions.h)
  target_link_libraries(
    functions_header
    INTERFACE array_traits
              crossplatform
              Eigen3::Eigen
              errors_header
              func_wrap
              structs_header
              tuple_tools
  )

  mujoco_pybind11_module(_callbacks callbacks.cc)
  target_link_libraries(
    _callbacks
    PRIVATE errors_header
            mujoco
            raw
            structs_header
  )

  mujoco_pybind11_module(_constants constants.cc)
  target_link_libraries(_constants PRIVATE mujoco)

  mujoco_pybind11_module(_enums enums.cc)
  target_link_libraries(
    _enums
    PRIVATE crossplatform
            enum_traits
            mujoco
            tuple_tools
  )

  mujoco_pybind11_module(_errors errors.cc)
  target_link_libraries(_errors PRIVATE errors_header)

  mujoco_pybind11_module(_functions functions.cc)
  target_link_libraries(
    _functions
    PRIVATE Eigen3::Eigen
            functions_header
            function_traits
            mujoco
            raw
  )
  if(APPLE)
    # C++17 aligned allocation is not available until macOS 10.14.
    target_compile_options(_functions PRIVATE -fno-aligned-allocation)
  endif()

  mujoco_pybind11_module(_render render.cc)
  target_link_libraries(
    _render
    PRIVATE Eigen3::Eigen
            errors_header
            functions_header
            function_traits
            mujoco
            raw
            structs_header
  )

  mujoco_pybind11_module(_batch batch.cc)

  mujoco_pybind11_module(_rollout rollout.cc threadpool.cc)

  target_link_libraries(_batch PRIVATE functions_header mujoco raw)
  target_link_libraries(_rollout PRIVATE functions_header mujoco raw)

  mujoco_pybind11_module(
    _structs
    indexers.cc
    serialization.h
    structs.cc
  )
  target_link_libraries(
    _structs
    PRIVATE absl::flat_hash_map
            crossplatform
            mujoco
            mujoco_private_api_header
            raw
            errors_header
            func_wrap
            function_traits
            structs_header
            structs_wrappers
  )

  if(APPLE)
    set_target_properties(_structs PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
  endif()

  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/specs.cc.inc)
    add_custom_command(
      OUTPUT specs.cc.inc
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${mujoco_SOURCE_DIR}/mujoco/python ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_spec_bindings.py > specs.cc.inc
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_spec_bindings.py
    )
  endif()

  mujoco_pybind11_module(
    _specs
    specs.cc
    specs_wrapper.h
    specs_wrapper.cc
    specs.cc.inc
  )
  target_link_libraries(
    _specs
    PRIVATE mujoco
            Eigen3::Eigen
            errors_header
            raw
            structs_header
  )

  mujoco_pybind11_module(_simulate simulate.cc)
  target_link_libraries(
    _simulate
    PRIVATE mujoco
            mujoco::libmujoco_simulate
            glfw
            errors_header
            raw
            structs_header
  )
endif()

# The Filament rendering API binding. It needs the Filament C API library, the
# Filament engine archives and the source-tree headers under
# MUJOCO_INCLUDE/mujoco, all of which the Studio platform target carries in its
# interface (includes, C++20, dependency archives); a Studio SDK is what makes
# the binding buildable, so key on that target.
if(MUJOCO_PYTHON_BUILD_STUDIO AND TARGET mujoco::platform)
  add_library(_render_filament_header INTERFACE)
  target_sources(
    _render_filament_header
    INTERFACE render_filament.h
  )
  set_target_properties(_render_filament_header PROPERTIES PUBLIC_HEADER render_filament.h)
  target_include_directories(_render_filament_header INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(
    _render_filament_header
    INTERFACE mujoco
              structs_header
              mujoco::platform
  )

  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/render_filament_generated.cc.inc)
    add_custom_command(
      OUTPUT render_filament_generated.cc.inc
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${mujoco_SOURCE_DIR}/mujoco/python ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_mjrf_bindings.py > render_filament_generated.cc.inc
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_mjrf_bindings.py
    )
  endif()

  mujoco_pybind11_module(
    _render_filament
    render_filament.cc
    render_filament_generated.cc.inc
  )
  target_link_libraries(
    _render_filament
    PRIVATE Eigen3::Eigen
            mujoco
            mujoco::platform
            raw
            structs_header
            _render_filament_header
  )
endif()

if(MUJOCO_PYTHON_BUILD_STUDIO)
  add_subdirectory(studio)
endif()

set(MUJOCO_CORE_WHEEL_TARGETS)
if(MUJOCO_PYTHON_BUILD_CORE)
  list(APPEND MUJOCO_CORE_WHEEL_TARGETS
      _batch
      _callbacks
      _constants
      _enums
      _errors
      _functions
      _render
      _rollout
      _simulate
      _specs
      _structs
  )
endif()
if(TARGET _render_filament)
  list(APPEND MUJOCO_CORE_WHEEL_TARGETS _render_filament)
endif()

set(LIBRARIES_FOR_WHEEL)
if(MUJOCO_PYTHON_BUILD_CORE)
  list(APPEND LIBRARIES_FOR_WHEEL "$<TARGET_FILE:mujoco>")
endif()
foreach(target IN LISTS MUJOCO_CORE_WHEEL_TARGETS MUJOCO_STUDIO_TARGETS)
  list(APPEND LIBRARIES_FOR_WHEEL "$<TARGET_FILE:${target}>")
endforeach()

if(MUJOCO_PYTHON_BUILD_CORE AND APPLE)
  add_executable(mjpython mjpython/mjpython.mm)
  target_include_directories(mjpython PRIVATE "${Python3_INCLUDE_DIRS}")
  target_link_libraries(mjpython PRIVATE "-framework Cocoa")
  set(LIBRARIES_FOR_WHEEL "${LIBRARIES_FOR_WHEEL}" "$<TARGET_FILE:mjpython>")
endif()

if(MUJOCO_PYTHON_MAKE_WHEEL)
  add_custom_target(
    wheel ALL
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${CMAKE_CURRENT_BINARY_DIR}/dist"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/.."
            "${CMAKE_CURRENT_BINARY_DIR}/dist"
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE"
            "${CMAKE_CURRENT_BINARY_DIR}/dist/LICENSE"
    COMMAND "${CMAKE_COMMAND}" -E copy ${LIBRARIES_FOR_WHEEL}
            "${CMAKE_CURRENT_BINARY_DIR}/dist/mujoco"
    COMMAND "${Python3_EXECUTABLE}" -m pip wheel --wheel-dir "${CMAKE_BINARY_DIR}" --no-deps -vvv
            "${CMAKE_CURRENT_BINARY_DIR}/dist"
  )

  if(MUJOCO_PYTHON_BUILD_CORE)
    add_dependencies(wheel mujoco)
    if(APPLE)
      add_dependencies(wheel mjpython)
    endif()
  endif()
  if(MUJOCO_CORE_WHEEL_TARGETS)
    add_dependencies(wheel ${MUJOCO_CORE_WHEEL_TARGETS})
  endif()
  if(MUJOCO_STUDIO_TARGETS)
    add_dependencies(wheel ${MUJOCO_STUDIO_TARGETS})
  endif()
endif()
