# Copyright 2023 DeepMind Technologies Limited
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
"""An Apple CGL context for offscreen rendering on macOS."""

import atexit
import contextlib
import ctypes
import os
import threading
import weakref

from mujoco.cgl import cgl

_ATTRIB = cgl.CGLPixelFormatAttribute
_PROFILE = cgl.CGLOpenGLProfile

_current_context = threading.local()


class GLContext:
  """An EGL context for headless accelerated OpenGL rendering on GPU devices."""

  def __init__(self, max_width, max_height):
    del max_width, max_height  # unused
    attrib_values = (
        _ATTRIB.CGLPFAOpenGLProfile, _PROFILE.CGLOGLPVersion_Legacy,
        _ATTRIB.CGLPFAColorSize, 24,
        _ATTRIB.CGLPFAAlphaSize, 8,
        _ATTRIB.CGLPFADepthSize, 24,
        _ATTRIB.CGLPFAStencilSize, 8,
        _ATTRIB.CGLPFAMultisample,
        _ATTRIB.CGLPFASampleBuffers, 1,
        _ATTRIB.CGLPFASample, 4,
        _ATTRIB.CGLPFAAccelerated,
        0,
    )
    attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
    self._pix = cgl.CGLPixelFormatObj()
    npix = cgl.GLint()
    cgl.CGLChoosePixelFormat(
        attribs, ctypes.byref(self._pix), ctypes.byref(npix)
    )

    self._context = cgl.CGLContextObj()
    cgl.CGLCreateContext(self._pix, 0, ctypes.byref(self._context))

  def make_current(self):
    cgl.CGLSetCurrentContext(self._context)
    cgl.CGLLockContext(self._context)
    _current_context.ref = weakref.ref(self)

  @contextlib.contextmanager
  def as_current(self):
    """Temporarily makes this context current, restoring any prior context."""
    if not self._context:
      yield
      return
    prev_ref = getattr(_current_context, 'ref', None)
    prev = prev_ref() if prev_ref is not None else None
    if prev is self:
      yield
      return
    self.make_current()
    try:
      yield
    finally:
      prev_ctx = (
          prev._context if prev is not None else None  # pylint: disable=protected-access
      )
      if prev is not None and prev_ctx:
        prev.make_current()
      else:
        cgl.CGLSetCurrentContext(None)
        _current_context.ref = None

  def free(self):
    """Frees resources associated with this context."""
    try:
      if self._context:
        cgl.CGLUnlockContext(self._context)
        prev_ref = getattr(_current_context, 'ref', None)
        if prev_ref is None or prev_ref() is self:
          cgl.CGLSetCurrentContext(None)
          _current_context.ref = None
        cgl.CGLReleaseContext(self._context)
      self._context = None

      if self._pix:
        cgl.CGLReleasePixelFormat(self._pix)
      self._pix = None
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def __del__(self):
    self.free()
