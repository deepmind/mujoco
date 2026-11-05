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

set(MUJOCO_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(MUJOCO_TEST_WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(GoogleTest)

# This macro is used to add a C++ test to MuJoCo CMake build system
#
# The CMake targets linked by default with target_link_libraries are:
#   mujoco fixture gmock
#
# The macro supports the following parameters:
#   * PROPERTIES: multiple value parameter, its value is set as test properties
#                 via the set_tests_properties(<test> PROPERTIES <argument>) function
#   * ADDITIONAL_LINK_LIBRARIES: multiple value parameter, additional libraries linked
#                                to the test via target_link_libraries
#   * MAIN_TARGET: single value parameter, used to specify the target linked to the tests
#                  that define the main entry poiny. If not indicated, gest_main is used
macro(mujoco_test name)
  set(options)
  set(oneValueArgs MAIN_TARGET)
  set(multiValueArgs PROPERTIES ADDITIONAL_LINK_LIBRARIES)
  cmake_parse_arguments(
    _ARGS
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  add_executable(${name} ${name}.cc)
  target_link_options(${name} PRIVATE ${EXTRA_LINK_OPTIONS})
  target_link_libraries(${name} mujoco fixture gmock)
  if(_ARGS_MAIN_TARGET)
    target_link_libraries(${name} ${_ARGS_MAIN_TARGET})
  else()
    target_link_libraries(${name} gtest_main)
  endif()
  if(_ARGS_ADDITIONAL_LINK_LIBRARIES)
    target_link_libraries(${name} ${_ARGS_ADDITIONAL_LINK_LIBRARIES})
  endif()
  target_include_directories(${name} PRIVATE ${MUJOCO_TEST_INCLUDE})
  if(MUJOCO_WITH_USD AND USD_INSTALL_ROOT)
    set(_test_rpath "${CMAKE_LIBRARY_OUTPUT_DIRECTORY};${USD_INSTALL_ROOT}")
    if(Python3_LIBRARY_DIRS)
      list(APPEND _test_rpath ${Python3_LIBRARY_DIRS})
    endif()
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${_test_rpath}")
  else()
    set_target_properties(${name} PROPERTIES BUILD_RPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  endif()
  # gtest_discover_tests is recommended over gtest_add_tests, but has issues on Windows.
  # It also requires a GoogleTest binary, so we fall back for custom MAIN_TARGET (e.g. benchmarks).
  if(WIN32 OR APPLE OR _ARGS_MAIN_TARGET)
    gtest_add_tests(
      TARGET ${name}
      SOURCES ${name}.cc
      WORKING_DIRECTORY ${MUJOCO_TEST_WORKING_DIR}
      TEST_LIST testList
    )
    # gtest_add_tests marks a test skipped if any of its cases skipped, regardless of the exit
    # code, which hides failures in the other cases of a TEST_P. Mark it skipped only if every
    # case skipped: gtest reports 0 passed tests, followed by nothing but skipped tests.
    set_tests_properties(
      ${testList} PROPERTIES SKIP_REGULAR_EXPRESSION
      "\\[  PASSED  \\] 0 tests\\.\n(\\[  SKIPPED \\] [^\n]*\n)+$"
    )
    if(WIN32)
      set(_win_test_path "$<TARGET_FILE_DIR:mujoco>")
      if(MUJOCO_WITH_USD AND USD_INSTALL_ROOT)
        list(APPEND _win_test_path "${USD_INSTALL_ROOT}")
        if(EXISTS "${pxr_DIR}/bin")
          list(APPEND _win_test_path "${pxr_DIR}/bin")
        endif()
        if(Python3_RUNTIME_LIBRARY_DIRS)
          list(APPEND _win_test_path ${Python3_RUNTIME_LIBRARY_DIRS})
        endif()
      endif()
      list(APPEND _win_test_path "$ENV{PATH}")
      string(REPLACE ";" "\\;" _win_test_path_escaped "${_win_test_path}")
      set_tests_properties(${testList} PROPERTIES ENVIRONMENT "PATH=${_win_test_path_escaped}")
    endif()
    if(_ARGS_PROPERTIES)
      set_tests_properties(${testList} PROPERTIES ${_ARGS_PROPERTIES})
    endif()
  else()
    if(_ARGS_PROPERTIES)
      gtest_discover_tests(
        ${name}
        WORKING_DIRECTORY ${MUJOCO_TEST_WORKING_DIR}
        PROPERTIES ${_ARGS_PROPERTIES}
      )
    else()
      gtest_discover_tests(
        ${name}
        WORKING_DIRECTORY ${MUJOCO_TEST_WORKING_DIR}
      )
    endif()
  endif()

endmacro()

add_library(fixture STATIC fixture.h fixture.cc)
target_include_directories(fixture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(fixture PUBLIC MJSTATIC)

target_link_libraries(
  fixture
  PUBLIC absl::core_headers
         absl::synchronization
         absl::flat_hash_map
         absl::flat_hash_set
         gtest
         gmock
         mujoco::mujoco
)
target_include_directories(fixture PRIVATE ${mujoco_SOURCE_DIR}/include gmock)

add_library(compare_model STATIC compare_model.h compare_model.cc)
target_include_directories(compare_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(compare_model PUBLIC MJSTATIC)
target_link_libraries(compare_model PUBLIC mujoco::mujoco)
target_include_directories(compare_model PRIVATE ${mujoco_SOURCE_DIR}/include)

add_library(compare_spec STATIC compare_spec.h compare_spec.cc)
target_include_directories(compare_spec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(compare_spec PUBLIC MJSTATIC)
target_link_libraries(compare_spec PUBLIC mujoco::mujoco)
target_include_directories(compare_spec PRIVATE ${mujoco_SOURCE_DIR}/include)

mujoco_test(fixture_test)

mujoco_test(compare_spec_test ADDITIONAL_LINK_LIBRARIES compare_spec)

mujoco_test(header_test)

mujoco_test(mjspecmacro_test)

mujoco_test(pipeline_test)

add_subdirectory(batch)
add_subdirectory(benchmark)
add_subdirectory(engine)
add_subdirectory(sample)
add_subdirectory(user)
add_subdirectory(xml)
add_subdirectory(plugin/actuator)
if(MUJOCO_WITH_USD)
  add_subdirectory(plugin/usd_decoder)
endif()
add_subdirectory(plugin/sdf)
add_subdirectory(experimental)
add_subdirectory(doc)
