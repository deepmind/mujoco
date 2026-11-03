#!/bin/bash
# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ -n "${TEST_SRCDIR:-}" ]; then
  readonly MODEL="${TEST_SRCDIR}/${TEST_WORKSPACE}/model/sleep/dominos.xml"
  readonly TARGET_BINARY="${TEST_SRCDIR}/${TEST_WORKSPACE}/sample/testspeed"
else
  MODEL="${CMAKE_SOURCE_DIR}/model/sleep/dominos.xml"
fi

die() { echo "$*" 1>&2 ; exit 1; }

if [ -z "$TARGET_BINARY" ]; then
  die "Expecting environment variable TARGET_BINARY."
fi

if [ -n "$MUJOCO_DLL_DIR" ]; then
  # Extend PATH to include the directory containing the mujoco DLL.
  # This is needed on Windows.
  PATH=$PATH:$MUJOCO_DLL_DIR
fi

# Run testspeed with override flags and verify they get applied
echo "Running testspeed with command-line overrides on dominos.xml..."
echo "Command: $TARGET_BINARY --nstep=10 --nthread=1 --noisestd=0.02 --noiserate=0.2 --solver=PGS --jacobian=sparse --integrator=rk4 --iterations=12 --tolerance=1e-7 --sleep_tolerance=2e-4 --noslip_iterations=5 $MODEL"

OUTPUT=$("$TARGET_BINARY" --nstep=10 --nthread=1 --noisestd=0.02 --noiserate=0.2 --solver=PGS --jacobian=sparse --integrator=rk4 --iterations=12 --tolerance=1e-7 --sleep_tolerance=2e-4 --noslip_iterations=5 "$MODEL") || die "testspeed failed"

# Verify option printing works.
# Dominos.xml has:
#   cone="elliptic" -> cone: Elliptic
#   impratio="10" -> impratio: 10
#   <flag sleep="enable"/> -> Sleep: Enabled
#
# Overridden via command line options:
#   solver=PGS
#   jacobian=sparse -> jacobian: Sparse
#   integrator=rk4 -> integrator: RK4
#   iterations=12
#   tolerance=1e-7 -> tolerance: 1e-07
#   sleep_tolerance=2e-4 (overwrites XML's 3e-4) -> sleep_tolerance: 0.0002
#   noslip_iterations=5

echo "Full testspeed Output:"
echo "--------------------------------------------------------"
echo "$OUTPUT"
echo "--------------------------------------------------------"

echo "Verifying printed non-default option block:"
echo "$OUTPUT" | grep -A 10 "Physics options (non-default):" || die "Missing physics options block!"
echo "--------------------------------------------------------"

echo "$OUTPUT" | grep -F "Physics options (non-default):" || die "Missing non-default header"
echo "$OUTPUT" | grep -F "  cone              : Elliptic" || die "Missing cone print"
echo "$OUTPUT" | grep -F "  impratio          : 10" || die "Missing impratio print"
echo "$OUTPUT" | grep -E "  sleep_tolerance   : (0.0002|2e-0?4)" || die "Missing sleep_tolerance print"
echo "$OUTPUT" | grep -F "  Sleep             : Enabled" || die "Missing Sleep print"
echo "$OUTPUT" | grep -F "  solver            : PGS" || die "Missing solver print"
echo "$OUTPUT" | grep -F "  jacobian          : Sparse" || die "Missing jacobian print"
echo "$OUTPUT" | grep -F "  integrator        : RK4" || die "Missing integrator print"
echo "$OUTPUT" | grep -F "  iterations        : 12" || die "Missing iterations print"
echo "$OUTPUT" | grep -E "  tolerance         : (1e-0?7|0.0000001|1e-7)" || die "Missing tolerance print"
echo "$OUTPUT" | grep -F "  noslip_iterations : 5" || die "Missing noslip_iterations print"
if echo "$OUTPUT" | grep -Fq "worker thread"; then
  die "Unexpected worker thread banner with default --nenginethread=0"
fi

# Verify --nenginethread worker thread banner for 1 and 2 worker threads.
OUTPUT_T1=$("$TARGET_BINARY" --nstep=2 --nenginethread=1 "$MODEL") || die "testspeed --nenginethread=1 failed"
echo "$OUTPUT_T1" | grep -F ", using 1 worker thread..." || die "Missing 1 worker thread banner"

OUTPUT_T2=$("$TARGET_BINARY" --nstep=2 --nenginethread=2 "$MODEL") || die "testspeed --nenginethread=2 failed"
echo "$OUTPUT_T2" | grep -F ", using 2 worker threads..." || die "Missing 2 worker threads banner"

echo "PASS"
