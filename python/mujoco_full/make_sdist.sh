#!/bin/bash -xe
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

if [[ -z ${VIRTUAL_ENV} ]] && [[ -z ${CONDA_DEFAULT_ENV} ]]; then
  echo "This script must be run from within a Python virtual environment"
  exit 1
fi

package_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
python_dir="$(cd -- "${package_dir}/.." >/dev/null 2>&1 && pwd)"

if [[ "$(uname)" == CYGWIN* || "$(uname)" == MINGW* ]]; then
  package_dir="$(cygpath -m ${package_dir})"
  python_dir="$(cygpath -m ${python_dir})"
  readonly tmp_dir="$(TMPDIR="${LOCALAPPDATA//\\/$'/'}/Temp" mktemp -d)"
else
  readonly tmp_dir="$(mktemp -d)"
fi
trap 'rm -rf "${tmp_dir}"' EXIT

python -m pip install --no-build-isolation \
    --index-url "${PYPI_INDEX_URL:-https://pypi.org/simple/}" \
    --upgrade --require-hashes \
    -r "${python_dir}"/make_sdist_requirements.txt
pushd "${tmp_dir}"
cp "${package_dir}"/{MANIFEST.in,README.md,pyproject.toml,setup.py} .
if [[ -f "${python_dir}/LICENSES_THIRD_PARTY.md" ]]; then
  cp "${python_dir}/LICENSES_THIRD_PARTY.md" .
fi

mkdir -p mujoco
cp "${python_dir}"/mujoco/__init__.py mujoco/__init__.py

# Copy over the LICENSE file.
cp "${python_dir}"/../LICENSE .

python -m build --no-isolation . --sdist
tar -tf dist/mujoco-*.tar.gz
popd

mkdir -p "${package_dir}"/dist
mv "${tmp_dir}"/dist/* "${package_dir}"/dist
