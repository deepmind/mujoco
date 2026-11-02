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

# the pre-rewrite box-box collider, linked only into the box-box tests so the rewrite is
# measured against it rather than asserted to be better; see boxbox_legacy.h
add_library(boxbox_legacy STATIC boxbox_legacy.h boxbox_legacy.c)
target_include_directories(boxbox_legacy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_include_directories(boxbox_legacy PRIVATE ${mujoco_SOURCE_DIR}/include ${mujoco_SOURCE_DIR}/src)
target_compile_definitions(boxbox_legacy PUBLIC MJSTATIC)
target_link_libraries(boxbox_legacy PUBLIC mujoco::mujoco)

mujoco_test(engine_collision_box_test ADDITIONAL_LINK_LIBRARIES ccd boxbox_legacy)

mujoco_test(engine_collision_box_fuzz_test ADDITIONAL_LINK_LIBRARIES ccd)

mujoco_test(engine_collision_continuous_test ADDITIONAL_LINK_LIBRARIES ccd)

mujoco_test(engine_collision_convex_test ADDITIONAL_LINK_LIBRARIES ccd)

mujoco_test(engine_collision_driver_test)

mujoco_test(engine_collision_gjk_test ADDITIONAL_LINK_LIBRARIES ccd)

mujoco_test(
  engine_collision_sdf_test
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:sdf_plugin>"
)
add_dependencies(engine_collision_sdf_test sdf_plugin)

mujoco_test(engine_core_constraint_test)

mujoco_test(engine_core_smooth_test ADDITIONAL_LINK_LIBRARIES absl::span)

mujoco_test(engine_core_util_test)

mujoco_test(engine_derivative_test)

mujoco_test(engine_forward_test)

mujoco_test(engine_inverse_test)

mujoco_test(engine_island_test)

mujoco_test(engine_io_test ADDITIONAL_LINK_LIBRARIES absl::str_format)

mujoco_test(engine_ipc_test)

mujoco_test(
  engine_plugin_test
  ADDITIONAL_LINK_LIBRARIES
  absl::str_format
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:actuator>"
)
add_dependencies(engine_plugin_test actuator sensor sdf_plugin)

mujoco_test(engine_passive_test)

mujoco_test(engine_print_test)

mujoco_test(
  engine_ray_test
  PROPERTIES
  ENVIRONMENT
  "MUJOCO_PLUGIN_DIR=$<TARGET_FILE_DIR:sdf_plugin>")

mujoco_test(engine_sensor_test)

mujoco_test(engine_setconst_test)

mujoco_test(engine_sleep_test)

mujoco_test(engine_solver_test)

mujoco_test(engine_sort_test)

mujoco_test(engine_support_test)

mujoco_test(engine_thread_test)

mujoco_test(engine_util_blas_test)

mujoco_test(engine_util_errmem_test)

mujoco_test(engine_util_misc_test)

mujoco_test(engine_util_solve_test)

mujoco_test(engine_util_sparse_test)

mujoco_test(engine_util_spatial_test)

mujoco_test(engine_vis_visualize_test)
