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

# Checks that the shared MuJoCo library exports only the C API that MuJoCo's
# headers declare with MJAPI (like CMake's libmujoco) plus the private
# _mjPRIVATE_* functions used by the Python bindings, so that engine internals
# and the bundled third-party libraries stay hidden; and that it exports every
# public function in introspect/functions.py (parsed from mujoco.h by clang).
#
# Usage: mujoco_shared_exports_test.sh [--allow=REGEX] [--exclude=REGEX] <exports> <functions_py> <header>...
#
# <exports> lists the library's exported symbols, one per line. <functions_py>
# is python/mujoco/introspect/functions.py. The headers are all of MuJoCo's
# headers. --allow permits additional exported symbols matching REGEX;
# --exclude skips public function names matching REGEX that the library isn't
# expected to export.

set -euo pipefail

allow=""
exclude=""
positional=()
for arg in "$@"; do
  case "${arg}" in
    --allow=*) allow="${arg#--allow=}" ;;
    --exclude=*) exclude="${arg#--exclude=}" ;;
    *) positional+=("${arg}") ;;
  esac
done
exports="${positional[0]}"
functions_py="${positional[1]}"
headers=("${positional[@]:2}")

symbols="$(sort -u "${exports}")"

# Any C API name declared with MJAPI in the headers may be exported, including
# names that depend on the build (e.g. the stack functions with
# AddressSanitizer) and the _mjPRIVATE_* functions used by the Python bindings.
allowed="$(awk '
  /^MJAPI/ {
    s = $0
    sub(/^MJAPI[ \t]+/, "", s)
    if (match(s, /\(\*[ \t]*[A-Za-z_][A-Za-z0-9_]*/)) {
      name = substr(s, RSTART + 2, RLENGTH - 2)
      gsub(/[ \t]/, "", name)
    } else {
      sub(/[(\[;].*/, "", s)
      name = s
      sub(/.*[^A-Za-z0-9_]/, "", name)
    }
    if (name ~ /^(mj|_mjPRIVATE_)/) print name
  }' "${headers[@]}" | sort -u)"

# Ignore the __start_*/__stop_* section bounds the linker itself adds for named
# sections in linked-in code (e.g. malloc hooks); they aren't part of the API.
# GNU gold (used by Bazel on Ubuntu 24.04) also keeps non-static thread_local
# variables with R_X86_64_TLSGD relocations in .dynsym even with
# -fvisibility=hidden and a version script, so allow those in MuJoCo's own
# namespace (_ZN6mujoco..., _ZZN6mujoco...) and the thread-local test hook
# mj_nesterov_momentum; unlike third-party symbols, they can't clash with other
# libraries.
unexpected="$(comm -23 <(echo "${symbols}") <(echo "${allowed}") |
  grep -v -E '^__(start|stop)_' |
  grep -v -E '^_ZZ?N6mujoco' |
  grep -v -x 'mj_nesterov_momentum' || true)"
if [[ -n "${allow}" ]]; then
  unexpected="$(grep -v -E "${allow}" <<< "${unexpected}" || true)"
fi
if [[ -n "${unexpected}" ]]; then
  echo "FAIL: the library exports symbols that aren't MJAPI or _mjPRIVATE_*:"
  echo "${unexpected}" | head -50
  exit 1
fi

# Every unconditional public function in introspect/functions.py (parsed from
# mujoco.h by clang) must be declared MJAPI and exported. mj_markStack and
# mj_freeStack are inline wrappers around mj__markStack and mj__freeStack under
# ASan.
api="$(awk -F"'" '/FunctionDecl\(/ {getline; print $2}' "${functions_py}" |
  grep -v -E '^mj_(mark|free)Stack$' | sort -u)"
if [[ -n "${exclude}" ]]; then
  api="$(grep -v -E "${exclude}" <<< "${api}" || true)"
fi

# Guard against a broken functions.py parse.
if [[ "$(wc -l <<< "${api}")" -lt 100 ]]; then
  echo "FAIL: found only $(wc -l <<< "${api}") functions in ${functions_py}"
  exit 1
fi

unmarked="$(comm -23 <(echo "${api}") <(echo "${allowed}"))"
if [[ -n "${unmarked}" ]]; then
  echo "FAIL: these public functions in ${functions_py} are not declared MJAPI:"
  echo "${unmarked}" | head -50
  exit 1
fi

missing="$(comm -23 <(echo "${api}") <(echo "${symbols}"))"
if [[ -n "${missing}" ]]; then
  echo "FAIL: the library doesn't export these public functions from ${functions_py}:"
  echo "${missing}" | head -50
  exit 1
fi

echo "PASS: the library exports $(wc -l <<< "${symbols}") symbols, all MJAPI or" \
  "_mjPRIVATE_*, including all $(wc -l <<< "${api}") functions checked from functions.py"
