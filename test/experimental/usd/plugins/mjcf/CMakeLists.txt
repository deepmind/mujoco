# Copyright 2025 DeepMind Technologies Limited
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

mujoco_test(mjcf_file_format_test ADDITIONAL_LINK_LIBRARIES
  ${PXR_LIBRARIES}
  usd
  tf
  usdGeom
  usdPhysics
  usdShade
  usd_test_utils
  mjcPhysics
  $<$<TARGET_EXISTS:Python3::Python>:Python3::Python>
  )
if(UNIX AND NOT APPLE)
  target_link_options(mjcf_file_format_test PRIVATE -Wl,--no-as-needed)
endif()
add_dependencies(mjcf_file_format_test usdMjcf)
target_compile_options(mjcf_file_format_test PRIVATE -UPXR_PYTHON_SUPPORT_ENABLED)
