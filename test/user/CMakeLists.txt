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

mujoco_test(
  user_model_test
  ADDITIONAL_LINK_LIBRARIES absl::str_format compare_model compare_spec
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:actuator>"
)

mujoco_test(user_objects_test)

mujoco_test(
  user_texture_test
  ADDITIONAL_LINK_LIBRARIES compare_model
)

mujoco_test(
  user_api_test
  ADDITIONAL_LINK_LIBRARIES compare_model compare_spec
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:actuator>"
)

mujoco_test(
  user_recompile_test
  ADDITIONAL_LINK_LIBRARIES compare_model compare_spec
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:actuator>"
)

mujoco_test(
  user_flex_test
)

mujoco_test(
  user_mesh_test
  ADDITIONAL_LINK_LIBRARIES absl::str_format
)

mujoco_test(user_resource_test)

mujoco_test(user_vfs_test)

mujoco_test(user_util_test)
target_sources(user_util_test PRIVATE ${mujoco_SOURCE_DIR}/src/user/user_util.cc)
