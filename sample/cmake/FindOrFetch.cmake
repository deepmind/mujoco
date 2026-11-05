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

#.rst:
# FindOrFetch
# ----------------------
#
# Find or fetch a package in order to satisfy target dependencies.
#
#   FindOrFetch([USE_SYSTEM_PACKAGE [ON/OFF]]
#               [PACKAGE_NAME [name]]
#               [LIBRARY_NAME [name]]
#               [GIT_REPO [repo]]
#               [GIT_TAG [tag]]
#               [CUSTOM_CMAKE [path]]
#               [PATCH_COMMAND [cmd] [args]]
#               [TARGETS [targets]]
#               [EXCLUDE_FROM_ALL])
#
# The command has the following parameters:
#
# Arguments:
#  - ``USE_SYSTEM_PACKAGE`` one-value argument on whether to search for the
#    package in the system (ON) or whether to fetch the library using
#    FetchContent from the specified Git repository (OFF). Note that
#    FetchContent variables will override this behaviour.
#  - ``PACKAGE_NAME`` name of the system-package. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``OFF``.
#  - ``LIBRARY_NAME`` name of the library. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``GIT_REPO`` git repository to fetch the library from. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``GIT_TAG`` tag reference when fetching the library from the git
#    repository. Ignored if ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``CUSTOM_CMAKE`` path to a custom CMakeLists.txt file to be used when
#    fetching the library from the git repository. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``PATCH_COMMAND`` Specifies a custom command to patch the sources after an
#    update. See https://cmake.org/cmake/help/latest/module/ExternalProject.html#command:externalproject_add
#    for details on the parameter.
#  - ``TARGETS`` list of targets to be satisfied. If any of these targets are
#    not currently defined, this macro will attempt to either find or fetch the
#    package.
#  - ``EXCLUDE_FROM_ALL`` if specified, the targets are not added to the ``all``
#    metatarget.
#
# Note: if ``USE_SYSTEM_PACKAGE`` is ``OFF``, FetchContent will be used to
# retrieve the specified targets. It is possible to specify any variable in
# https://cmake.org/cmake/help/latest/module/FetchContent.html#variables to
# override this macro behaviour.

if(NOT COMMAND _mujoco_ensure_git_retry_script)
  macro(_mujoco_ensure_git_retry_script)
    set(_MUJOCO_GIT_RETRY_SCRIPT "${CMAKE_BINARY_DIR}/CMakeFiles/mujoco_git_retry.cmake")
    if(NOT EXISTS "${_MUJOCO_GIT_RETRY_SCRIPT}")
      file(WRITE "${_MUJOCO_GIT_RETRY_SCRIPT}" [=[
set(_cmd)
set(_in_cmd FALSE)
foreach(_i RANGE 3 ${CMAKE_ARGC})
  if(_i EQUAL ${CMAKE_ARGC})
    break()
  endif()
  set(_arg "${CMAKE_ARGV${_i}}")
  if(_in_cmd)
    list(APPEND _cmd "${_arg}")
  elseif("${_arg}" STREQUAL "--")
    set(_in_cmd TRUE)
  endif()
endforeach()
set(_max_attempts 3)
set(_delay 5)
foreach(_attempt RANGE 1 ${_max_attempts})
  execute_process(COMMAND ${_cmd} RESULT_VARIABLE _res)
  if(_res EQUAL 0)
    return()
  endif()
  if(_attempt LESS ${_max_attempts})
    message(WARNING "Command failed (attempt ${_attempt}/${_max_attempts}, exit ${_res}); retrying in ${_delay}s: ${_cmd}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep ${_delay})
    math(EXPR _delay "${_delay} * 2")
  endif()
endforeach()
message(FATAL_ERROR "Command failed after ${_max_attempts} attempts: ${_cmd}")
]=])
    endif()
  endmacro()
endif()

if(NOT COMMAND FindOrFetch)
  macro(FindOrFetch)
    if(NOT FetchContent)
      include(FetchContent)
    endif()

    # Parse arguments.
    set(options EXCLUDE_FROM_ALL)
    set(one_value_args
        USE_SYSTEM_PACKAGE
        PACKAGE_NAME
        LIBRARY_NAME
        GIT_REPO
        GIT_TAG
        CUSTOM_CMAKE
    )
    set(multi_value_args PATCH_COMMAND TARGETS)
    cmake_parse_arguments(
      _ARGS
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
    )

    # Check if all targets are found.
    if(NOT _ARGS_TARGETS)
      message(FATAL_ERROR "mujoco::FindOrFetch: TARGETS must be specified.")
    endif()

    set(targets_found TRUE)
    message(CHECK_START
            "mujoco::FindOrFetch: checking for targets in package `${_ARGS_PACKAGE_NAME}`"
    )
    foreach(target ${_ARGS_TARGETS})
      if(NOT TARGET ${target})
        message(CHECK_FAIL "target `${target}` not defined.")
        set(targets_found FALSE)
        break()
      endif()
    endforeach()

    # If targets are not found, use `find_package` or `FetchContent...` to get it.
    if(NOT targets_found)
      if(${_ARGS_USE_SYSTEM_PACKAGE})
        message(CHECK_START
                "mujoco::FindOrFetch: finding `${_ARGS_PACKAGE_NAME}` in system packages..."
        )
        find_package(${_ARGS_PACKAGE_NAME} REQUIRED)
        message(CHECK_PASS "found")
      else()
        set(USE_LOCAL_TARBALL FALSE)
        if(MUJOCO_CMAKE_DEP_CACHE)
          # Try to find versioned library tarball.
          set(TARBALL_PATH "${MUJOCO_CMAKE_DEP_CACHE}/${_ARGS_LIBRARY_NAME}-${_ARGS_GIT_TAG}.tar.gz")
          if(EXISTS "${TARBALL_PATH}")
            set(USE_LOCAL_TARBALL TRUE)
          endif()
        endif()
        if(_ARGS_PATCH_COMMAND)
          set(_WRAPPED_PATCH_COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR} ${_ARGS_PATCH_COMMAND})
        else()
          set(_WRAPPED_PATCH_COMMAND)
        endif()

        set(FETCHCONTENT_QUIET OFF)
        if(USE_LOCAL_TARBALL)
          message(STATUS "mujoco::FindOrFetch: Using package cache for ${_ARGS_LIBRARY_NAME}: ${TARBALL_PATH}")
          FetchContent_Declare(
            ${_ARGS_LIBRARY_NAME}
            URL ${TARBALL_PATH}
            PATCH_COMMAND ${_WRAPPED_PATCH_COMMAND}
          )
        else()
          find_package(Git REQUIRED)
          message(STATUS "mujoco::FindOrFetch: Using FetchContent for ${_ARGS_LIBRARY_NAME}: ${_ARGS_GIT_REPO}")
          _mujoco_ensure_git_retry_script()
          FetchContent_Declare(
            ${_ARGS_LIBRARY_NAME}
            DOWNLOAD_COMMAND
              ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>
              COMMAND ${GIT_EXECUTABLE} init <SOURCE_DIR>
              COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> remote add origin ${_ARGS_GIT_REPO}
              COMMAND ${CMAKE_COMMAND} -P "${_MUJOCO_GIT_RETRY_SCRIPT}" -- ${GIT_EXECUTABLE} -C <SOURCE_DIR> fetch --depth 1 origin ${_ARGS_GIT_TAG}
              COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> -c advice.detachedHead=false checkout FETCH_HEAD
              COMMAND ${CMAKE_COMMAND} -P "${_MUJOCO_GIT_RETRY_SCRIPT}" -- ${GIT_EXECUTABLE} -C <SOURCE_DIR> submodule update --init --recursive --depth 1
            PATCH_COMMAND ${_WRAPPED_PATCH_COMMAND}
            UPDATE_DISCONNECTED TRUE
          )
        endif()
        if(${_ARGS_EXCLUDE_FROM_ALL})
          FetchContent_GetProperties(${_ARGS_LIBRARY_NAME})
          if(NOT ${${_ARGS_LIBRARY_NAME}_POPULATED})
            FetchContent_Populate(${_ARGS_LIBRARY_NAME})
            if(NOT "${_ARGS_CUSTOM_CMAKE}" STREQUAL "")
              file(COPY
                "${_ARGS_CUSTOM_CMAKE}"
                DESTINATION "${${_ARGS_LIBRARY_NAME}_SOURCE_DIR}"
              )
            endif()
            add_subdirectory(
              ${${_ARGS_LIBRARY_NAME}_SOURCE_DIR} ${${_ARGS_LIBRARY_NAME}_BINARY_DIR}
              EXCLUDE_FROM_ALL
            )
          endif()
        else()
          FetchContent_MakeAvailable(${_ARGS_LIBRARY_NAME})
        endif()
        message(CHECK_PASS "Done")
      endif()
    else()
      message(CHECK_PASS "found")
    endif()
  endmacro()
endif()


#.rst:
# FetchPackage
# ----------------------
#
# Fetches a package from a Git repository in order to satisfy dependencies.
#
#   FetchPackage([PACKAGE_NAME [name]]
#               [GIT_REPO [repo]]
#               [GIT_TAG [tag]]
#               [PATCH_COMMAND [cmd] [args]]
#              )
#
# The command has the following parameters:
#
# Arguments:
#  - ``PACKAGE_NAME`` name of the package.
#  - ``GIT_REPO`` git repository to fetch the library from.
#  - ``GIT_TAG`` tag reference when fetching the library from the git repo.
#  - ``CUSTOM_CMAKE`` path to a custom CMakeLists.txt file to be used when
#    fetching the library from the git repository.
#  - ``PATCH_COMMAND`` Specifies a custom command to patch the sources after an
#    update. See https://cmake.org/cmake/help/latest/module/ExternalProject.html#command:externalproject_add
#    for details on the parameter.
#  - ``TARGETS`` (optional) list of targets to be satisfied. If any of these
#    targets are not currently defined, this macro will attempt to fetch the
#    package. If not specified, the package name will be used as the target.
#
# Note: This is a wrapper around FindOrFetch that sets EXCLUDE_FROM_ALL and
# USE_SYSTEM_PACKAGE to OFF.
if(NOT COMMAND FetchPackage)
  function(FetchPackage)
    cmake_parse_arguments(
      _ARGS
      "EXCLUDE_FROM_ALL"
      "PACKAGE_NAME;GIT_REPO;GIT_TAG;CUSTOM_CMAKE"
      "PATCH_COMMAND;TARGETS"
      ${ARGN}
    )
    if(NOT _ARGS_TARGETS)
      set(_ARGS_TARGETS ${_ARGS_PACKAGE_NAME})
    endif()

    FindOrFetch(
      PACKAGE_NAME ${_ARGS_PACKAGE_NAME}
      LIBRARY_NAME ${_ARGS_PACKAGE_NAME}
      GIT_REPO ${_ARGS_GIT_REPO}
      GIT_TAG ${_ARGS_GIT_TAG}
      CUSTOM_CMAKE ${_ARGS_CUSTOM_CMAKE}
      PATCH_COMMAND ${_ARGS_PATCH_COMMAND}
      TARGETS ${_ARGS_TARGETS}
      USE_SYSTEM_PACKAGE OFF
      EXCLUDE_FROM_ALL
    )
  endfunction()
endif()
