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
cp "${package_dir}"/pyproject.toml "${package_dir}"/setup.py \
   "${package_dir}"/MANIFEST.in "${package_dir}"/README.md .

mkdir -p mujoco/_usd mujoco/usd/MjcPhysics mujoco/util mujoco/cmake

cp "${python_dir}"/mujoco/CMakeLists.txt \
   "${python_dir}"/mujoco/errors.h \
   "${python_dir}"/mujoco/gil.h \
   "${python_dir}"/mujoco/indexer_xmacro.h \
   "${python_dir}"/mujoco/indexers.h \
   "${python_dir}"/mujoco/private.h \
   "${python_dir}"/mujoco/raw.h \
   "${python_dir}"/mujoco/structs.h \
   mujoco/

cp -r "${python_dir}"/mujoco/util/* mujoco/util/
cp "${python_dir}"/mujoco/usd/_usd_init.py mujoco/_usd/__init__.py
cp "${python_dir}"/mujoco/usd/CMakeLists.txt \
   "${python_dir}"/mujoco/usd/layer_sink_module.cc \
   "${python_dir}"/mujoco/usd/pybind.h \
   "${python_dir}"/mujoco/usd/pybind_boost_py_utils.h \
   "${python_dir}"/mujoco/usd/writer_module.cc \
   mujoco/usd/
cp -r "${python_dir}"/mujoco/usd/MjcPhysics/* mujoco/usd/MjcPhysics/

cp "${python_dir}"/../LICENSE .
cp -r "${python_dir}"/../cmake/* mujoco/cmake/

python -m build --no-isolation . --sdist
tar -tf dist/mujoco_usd-*.tar.gz
popd

mkdir -p "${package_dir}"/dist
mv "${tmp_dir}"/dist/* "${package_dir}"/dist
