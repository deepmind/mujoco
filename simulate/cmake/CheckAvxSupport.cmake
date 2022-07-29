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

include(CheckCSourceCompiles)

# Assigns compiler options to the given variable based on availability of AVX.
function(get_avx_compile_options OUTPUT_VAR)
  message(VERBOSE "Checking if AVX is available...")

  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
  else()
    set(CMAKE_REQUIRED_FLAGS "-mavx")
  endif()

  if(APPLE AND "x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES)
    message(STATUS "Building x86_64 on macOS, forcing CAN_BUILD_AVX to TRUE.")
    set(CAN_BUILD_AVX TRUE)
  else()
    check_c_source_compiles(
      "
      #include <immintrin.h>
      int main(int argc, char* argv[]) {
        __m256d ymm;
        return 0;
      }
    "
      CAN_BUILD_AVX
    )
  endif()

  if(CAN_BUILD_AVX)
    message(VERBOSE "Checking if AVX is available... AVX available.")
    set("${OUTPUT_VAR}"
        ${CMAKE_REQUIRED_FLAGS}
        PARENT_SCOPE
    )
  else()
    message(VERBOSE "Checking if AVX is available... AVX not available.")
    set("${OUTPUT_VAR}" PARENT_SCOPE)
  endif()
endfunction()
