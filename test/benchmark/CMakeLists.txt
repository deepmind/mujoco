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
  ccd_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers ccd
)

mujoco_test(
  cone_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  step_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  thread_performance_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  parse_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  spec_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  engine_util_spatial_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  factorI_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  inertia_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  iszero_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  solveLD_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  engine_util_solve_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  crb_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  engine_util_sparse_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)

mujoco_test(
  chol_benchmark_test
  MAIN_TARGET benchmark::benchmark_main
  ADDITIONAL_LINK_LIBRARIES benchmark::benchmark absl::core_headers
)
