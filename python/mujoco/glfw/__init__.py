# Copyright 2017 The dm_control Authors
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
"""An OpenGL context created via GLFW."""

import contextlib

import glfw


class GLContext:
  """An OpenGL context created via GLFW."""

  def __init__(self, max_width, max_height):
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, 0)
    self._context = glfw.create_window(width=max_width, height=max_height,
                                       title='Invisible window', monitor=None,
                                       share=None)

  def make_current(self):
    glfw.make_context_current(self._context)

  @contextlib.contextmanager
  def as_current(self):
    """Temporarily makes this context current, restoring any prior context."""
    if not self._context:
      yield
      return
    prev_ctx = glfw.get_current_context()
    if prev_ctx == self._context:
      yield
      return
    self.make_current()
    try:
      yield
    finally:
      glfw.make_context_current(prev_ctx)

  def free(self):
    if self._context:
      if glfw.get_current_context() == self._context:
        glfw.make_context_current(None)
      glfw.destroy_window(self._context)
    self._context = None

  def __del__(self):
    self.free()
