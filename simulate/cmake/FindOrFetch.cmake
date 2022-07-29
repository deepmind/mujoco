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
#
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

if(COMMAND FindOrFetch)
  return()
endif()

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
      message(CHECK_START
              "mujoco::FindOrFetch: Using FetchContent to retrieve `${_ARGS_LIBRARY_NAME}`"
      )
      FetchContent_Declare(
        ${_ARGS_LIBRARY_NAME}
        GIT_REPOSITORY ${_ARGS_GIT_REPO}
        GIT_TAG ${_ARGS_GIT_TAG}
        GIT_SHALLOW FALSE
        PATCH_COMMAND ${_ARGS_PATCH_COMMAND}
      )
      if(${_ARGS_EXCLUDE_FROM_ALL})
        FetchContent_GetProperties(${_ARGS_LIBRARY_NAME})
        if(NOT ${${_ARGS_LIBRARY_NAME}_POPULATED})
          FetchContent_Populate(${_ARGS_LIBRARY_NAME})
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
