# Copyright 2026 DeepMind Technologies Limited
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
"""Built-in MuJoCo Studio application."""

# pylint: disable=g-importing-member
from mujoco.studio.app import inputs
from mujoco.studio.app import plugins
from mujoco.studio.app.plugins import App
from mujoco.studio.app.plugins import Config
from mujoco.studio.app.plugins import InitEvent
from mujoco.studio.app.plugins import install_console_hooks
from mujoco.studio.app.plugins import SIM_PLUGINS

__all__ = [
    'App',
    'Config',
    'InitEvent',
    'SIM_PLUGINS',
    'inputs',
    'install_console_hooks',
    'plugins',
]
