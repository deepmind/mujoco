# Copyright 2018 The dm_control Authors
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
"""An OSMesa context for software-based OpenGL rendering."""

import os

PYOPENGL_PLATFORM = os.environ.get('PYOPENGL_PLATFORM')

if not PYOPENGL_PLATFORM:
  os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
elif PYOPENGL_PLATFORM.lower() != 'osmesa':
  raise ImportError(
      'Cannot use OSMesa rendering platform. '
      'The PYOPENGL_PLATFORM environment variable is set to {!r} '
      '(should be either unset or \'osmesa\').'
      .format(PYOPENGL_PLATFORM))

# pylint: disable=g-import-not-at-top
from OpenGL import GL
from OpenGL import osmesa
from OpenGL.GL import arrays

_DEPTH_BITS = 24
_STENCIL_BITS = 8
_ACCUM_BITS = 0


class GLContext:
  """An OSMesa context for software-based OpenGL rendering."""

  def __init__(self, max_width, max_height):
    """Initializes this OSMesa context."""
    self._context = osmesa.OSMesaCreateContextExt(
        osmesa.OSMESA_RGBA,
        _DEPTH_BITS,
        _STENCIL_BITS,
        _ACCUM_BITS,
        None,  # sharelist
    )
    if not self._context:
      raise RuntimeError('Failed to create OSMesa GL context.')

    self._height = max_height
    self._width = max_width

    # Allocate a buffer to render into.
    self._buffer = arrays.GLfloatArray.zeros((max_height, max_width, 4))

  def make_current(self):
    if self._context:
      success = osmesa.OSMesaMakeCurrent(
          self._context,
          self._buffer,
          GL.GL_FLOAT,
          self._width,
          self._height)
      if not success:
        raise RuntimeError('Failed to make OSMesa context current.')

  def free(self):
    """Frees resources associated with this context."""
    if self._context and self._context == osmesa.OSMesaGetCurrentContext():
      osmesa.OSMesaMakeCurrent(None, None, GL.GL_FLOAT, 0, 0)
    osmesa.OSMesaDestroyContext(self._context)
    self._buffer = None
    self._context = None

  def __del__(self):
    self.free()
