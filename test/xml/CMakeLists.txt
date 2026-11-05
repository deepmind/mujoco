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

mujoco_test(xml_api_test)

mujoco_test(xml_urdf_test)

mujoco_test(
  xml_native_reader_test
  ADDITIONAL_LINK_LIBRARIES compare_model
)

mujoco_test(schema_defaults_test)

mujoco_test(xml_utils_test)

mujoco_test(
  xml_native_writer_test
  ADDITIONAL_LINK_LIBRARIES
  absl::flat_hash_set
  compare_model
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:actuator>"
)
add_dependencies(xml_native_writer_test actuator sensor sdf_plugin)

mujoco_test(
  xml_write_read_test
  ADDITIONAL_LINK_LIBRARIES
  absl::flat_hash_set
  compare_model
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:actuator>"
)
add_dependencies(xml_write_read_test actuator sensor sdf_plugin)

add_subdirectory(mjz)
