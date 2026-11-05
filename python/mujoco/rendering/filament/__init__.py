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

"""Filament rendering package for MuJoCo.

Package marker required for Python package discovery and wheel distribution.

Note:
  When using Filament materials or assets, ensure that resource providers are
  registered by importing `mujoco.studio.window` before creating
  a Filament context. `mujoco.rendering.filament.renderer.Renderer` does not
  register them: it wraps a context the caller has already created.
"""
