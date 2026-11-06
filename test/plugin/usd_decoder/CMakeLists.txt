# Copyright 2026 DeepMind Technologies Limited
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

if(UNIX AND NOT APPLE)
  if(CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+" OR NOT CMAKE_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
    return()
  endif()
endif()
find_package(Python3 3.11...<3.14 COMPONENTS Interpreter Development QUIET)
mujoco_test(
  usd_decoder_test
  ADDITIONAL_LINK_LIBRARIES
  ${PXR_LIBRARIES}
  usd
  sdf
  tf
  arch
  mujoco_usd
  $<$<TARGET_EXISTS:Python3::Python>:Python3::Python>
)
if(UNIX AND NOT APPLE)
  target_link_options(usd_decoder_test PRIVATE -Wl,--no-as-needed)
endif()
add_dependencies(usd_decoder_test usd_decoder_plugin)
target_compile_options(usd_decoder_test PRIVATE -UPXR_PYTHON_SUPPORT_ENABLED)
target_compile_definitions(usd_decoder_test PRIVATE
  USD_DECODER_PLUGIN_PATH="$<TARGET_FILE:usd_decoder_plugin>"
)
