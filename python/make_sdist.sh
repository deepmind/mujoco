#!/bin/bash -xe
# Copyright 2022 DeepMind Technologies Limited
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

if [[ -z ${VIRTUAL_ENV} ]] && [[ -z ${CONDA_DEFAULT_ENV} ]]; then
  echo "This script must be run from within a Python virtual environment"
  exit 1
fi

# Figure out the path to this script (https://stackoverflow.com/a/246128).
package_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ "$(uname)" == CYGWIN* || "$(uname)" == MINGW* ]]; then
  package_dir="$(cygpath -m ${package_dir})"
  readonly tmp_dir="$(TMPDIR="${LOCALAPPDATA//\\/$'/'}/Temp" mktemp -d)"
else
  readonly tmp_dir="$(mktemp -d)"
fi
trap 'rm -rf "${tmp_dir}"' EXIT

python -m pip install --no-build-isolation \
    --index-url "${PYPI_INDEX_URL:-https://pypi.org/simple/}" \
    --upgrade --require-hashes \
    -r ${package_dir}/make_sdist_requirements.txt
pushd ${tmp_dir}
for item in "${package_dir}"/*; do
  if [[ -d "${item}" ]]; then
    case "${item##*/}" in
      build*|cmake-build*|dist) continue ;;
    esac
  fi
  cp -r "${item}" .
done
rm -rf mujoco_full studio usd mujoco/mujoco_full mujoco/sample/studio mujoco/studio mujoco/usd/MjcPhysics
rm -f mujoco/render_filament.cc mujoco/render_filament.h
rm -f mujoco/usd/*.cc mujoco/usd/*.cpp mujoco/usd/*.h \
      mujoco/usd/CMakeLists.txt mujoco/usd/MANIFEST.in \
      mujoco/usd/pyproject.toml mujoco/usd/setup.py \
      mujoco/usd/make_sdist.sh mujoco/usd/_usd_init.py

# Generate header files.
old_pythonpath="${PYTHONPATH}"
if [[ "$(uname)" == CYGWIN* || "$(uname)" == MINGW* ]]; then
  export PYTHONPATH="${old_pythonpath};${package_dir}/mujoco/python/.."
else
  export PYTHONPATH="${old_pythonpath}:${package_dir}/mujoco/python/.."
fi
python "${package_dir}"/mujoco/codegen/generate_enum_traits.py > \
    mujoco/enum_traits.h
python "${package_dir}"/mujoco/codegen/generate_function_traits.py > \
    mujoco/function_traits.h
python "${package_dir}"/mujoco/codegen/generate_spec_bindings.py > \
    mujoco/specs.cc.inc
export PYTHONPATH="${old_pythonpath}"

# Copy over the LICENSE file.
cp "${package_dir}"/../LICENSE .

# Copy over CMake scripts.
mkdir -p mujoco/cmake
cp -r "${package_dir}"/../cmake/* mujoco/cmake

# Copy over Simulate source code (excluding local build trees).
mkdir -p mujoco/simulate
for item in "${package_dir}"/../simulate/*; do
  if [[ -d "${item}" ]]; then
    case "${item##*/}" in
      build*|cmake-build*) continue ;;
    esac
  fi
  cp -r "${item}" mujoco/simulate/
done

# Copy over documentation chapters and agent skills (excluding images and scripts).
mkdir -p mujoco/doc
(
  cd "${package_dir}/../doc"
  find . -path ./internal -prune -o -type f \( -name '*.rst' -o -path './skills/*.md' \) -print0 |
    while IFS= read -r -d '' f; do
      mkdir -p "${tmp_dir}/mujoco/doc/${f%/*}"
      cp "${f}" "${tmp_dir}/mujoco/doc/${f}"
    done
)

python -m build --no-isolation . --sdist
tar -tf dist/mujoco_core-*.tar.gz
popd

mkdir -p "${package_dir}"/dist
mv "${tmp_dir}"/dist/* "${package_dir}"/dist
