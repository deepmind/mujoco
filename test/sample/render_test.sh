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

MODEL="${CMAKE_SOURCE_DIR}/model/humanoid/humanoid.xml"

die() { echo "$*" 1>&2 ; exit 1; }

if [ -z "$TARGET_BINARY" ]; then
  die "Expecting environment variable TARGET_BINARY."
fi

if [ -n "$MUJOCO_DLL_DIR" ]; then
  # Extend PATH to include the directory containing the mujoco DLL.
  # This is needed on Windows.
  if command -v cygpath >/dev/null 2>&1; then
    PATH="$PATH:$(cygpath -u "$MUJOCO_DLL_DIR")"
  else
    PATH="$PATH:$MUJOCO_DLL_DIR"
  fi
fi

# Test help with no arguments (expect success and usage message)
status=0
OUTPUT=$("$TARGET_BINARY" 2>&1) || status=$?
if [ $status -ne 0 ]; then
  if [ $status -eq 127 ] && { [ -n "$MUJOCO_DLL_DIR" ] || [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ]; }; then
    echo "Skipping render_test: TARGET_BINARY cannot be executed in this environment (exit code 127, missing graphical runtime libraries like opengl32.dll)"
    exit 0
  fi
  die "render failed with no arguments (exit code $status): $OUTPUT"
fi
echo "$OUTPUT" | grep "Usage:" > /dev/null || die "Expected usage message with no arguments"

# Test help with --help (expect success and usage message)
OUTPUT=$("$TARGET_BINARY" --help 2>&1) || die "render failed with --help (exit code $?): $OUTPUT"
echo "$OUTPUT" | grep "Usage:" > /dev/null || die "Expected usage message with --help"

# Test help with -h (expect success and usage message)
OUTPUT=$("$TARGET_BINARY" -h 2>&1) || die "render failed with -h (exit code $?): $OUTPUT"
echo "$OUTPUT" | grep "Usage:" > /dev/null || die "Expected usage message with -h"

# Test nonexistent model (expect failure)
"$TARGET_BINARY" "${TEST_TMPDIR}/nonexistent_model.xml" > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for nonexistent model"
fi

# Test invalid width (expect failure)
"$TARGET_BINARY" "$MODEL" --width=-10 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for negative width"
fi

# Test invalid geomgroup (expect failure)
"$TARGET_BINARY" "$MODEL" --geomgroup=123 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for invalid geomgroup length"
fi

# Test invalid backend (expect failure)
"$TARGET_BINARY" "$MODEL" --backend=invalid_backend > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for invalid backend"
fi

# Test invalid label (expect failure)
"$TARGET_BINARY" "$MODEL" --label=invalid_label > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for invalid label"
fi

# Test invalid frame (expect failure)
"$TARGET_BINARY" "$MODEL" --frame=invalid_frame > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for invalid frame"
fi

# Test negative duration (expect failure)
"$TARGET_BINARY" "$MODEL" --duration=-1 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for negative duration"
fi

# Test zero fps (expect failure)
"$TARGET_BINARY" "$MODEL" --fps=0 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for zero fps"
fi

# Test multi-frame recording with .png output (expect failure)
"$TARGET_BINARY" "$MODEL" "${TEST_TMPDIR}/bad.png" --duration=0.1 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  die "Expected failure for multi-frame recording to .png"
fi

# Test that the Filament backend actually renders. A headless environment
# without a render context (e.g. CI without a GPU) fails to initialize, which is
# tolerated here; only a missing-asset failure -- which passing arg validation
# alone would not catch -- is fatal.
OUT_PNG="${TEST_TMPDIR}/render_out.png"
RENDER_LOG=$("$TARGET_BINARY" "$MODEL" "$OUT_PNG" --width=64 --height=48 2>&1)
if echo "$RENDER_LOG" | grep -q "Failed to open filament asset"; then
  die "Filament asset resolution failed: $RENDER_LOG"
fi
if [ -f "$OUT_PNG" ]; then
  echo "Filament render produced an output image"
  OUT_RGB="${TEST_TMPDIR}/render_out.rgb"
  "$TARGET_BINARY" "$MODEL" "$OUT_RGB" --width=64 --height=48 --duration=0.1 --fps=20 > /dev/null 2>&1 \
    || die "Filament multi-frame recording failed"
  RGB_BYTES=$(wc -c < "$OUT_RGB" | tr -d ' ')
  EXPECTED_BYTES=$((64 * 48 * 3 * 2))
  if [ "$RGB_BYTES" -ne "$EXPECTED_BYTES" ]; then
    die "Expected $EXPECTED_BYTES RGB bytes, got $RGB_BYTES"
  fi
else
  echo "Skipping render output check (no render context available)"
fi
