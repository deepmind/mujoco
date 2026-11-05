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
# ==============================================================================
"""Install script for the MuJoCo Python metapackage."""

import os
import setuptools


def get_long_description():
  """Creates a long description for the package from bundled markdown files."""
  current_dir = os.path.abspath(os.path.dirname(__file__))
  with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    description = f.read()
  try:
    with open(
        os.path.join(current_dir, 'LICENSES_THIRD_PARTY.md'), encoding='utf-8'
    ) as f:
      description = f'{description}\n{f.read()}'
  except FileNotFoundError:
    pass
  return description


setuptools.setup(
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
)
