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

set(MUJOCO_ENGINE_SRCS
    engine_array_safety.h
    engine_callback.c
    engine_callback.h
    engine_collision_box.c
    engine_collision_continuous.c
    engine_collision_continuous.h
    engine_collision_convex.c
    engine_collision_convex.h
    engine_collision_driver.c
    engine_collision_driver.h
    engine_collision_flex.c
    engine_collision_flex.h
    engine_collision_gjk.c
    engine_collision_gjk.h
    engine_collision_primitive.c
    engine_collision_primitive.h
    engine_collision_sdf.c
    engine_collision_sdf.h
    engine_core_constraint.c
    engine_core_constraint.h
    engine_core_util.c
    engine_core_util.h
    engine_core_smooth.c
    engine_core_smooth.h
    engine_crossplatform.cc
    engine_crossplatform.h
    engine_derivative.c
    engine_derivative.h
    engine_derivative_fd.c
    engine_derivative_fd.h
    engine_forward.c
    engine_forward.h
    engine_global_table.h
    engine_inverse.c
    engine_inverse.h
    engine_ipc.c
    engine_ipc.h
    engine_init.c
    engine_init.h
    engine_inline.h
    engine_island.c
    engine_island.h
    engine_io.c
    engine_io.h
    engine_macro.h
    engine_memory.c
    engine_memory.h
    engine_metric.c
    engine_metric.h
    engine_name.c
    engine_name.h
    engine_passive.c
    engine_passive.h
    engine_plugin.cc
    engine_plugin.h
    engine_print.c
    engine_print.h
    engine_ray.c
    engine_ray.h
    engine_sensor.c
    engine_sensor.h
    engine_setconst.c
    engine_setconst.h
    engine_sleep.c
    engine_sleep.h
    engine_solver.c
    engine_solver.h
    engine_sort.h
    engine_support.c
    engine_support.h
    engine_thread.cc
    engine_thread.h
    engine_util_blas.c
    engine_util_blas.h
    engine_util_blas_avx.h
    engine_util_errmem.c
    engine_util_errmem.h
    engine_util_misc.c
    engine_util_misc.h
    engine_util_solve.c
    engine_util_solve.h
    engine_util_sparse.c
    engine_util_sparse.h
    engine_util_sparse_avx.h
    engine_util_spatial.c
    engine_util_spatial.h
    engine_util_spatial_avx.h
    engine_vis_init.c
    engine_vis_init.h
    engine_vis_interact.c
    engine_vis_interact.h
    engine_vis_visualize.c
    engine_vis_visualize.h
)

target_sources(mujoco PRIVATE ${MUJOCO_ENGINE_SRCS})
