# Copyright 2022 DeepMind Technologies Limited
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
"""Tests for MuJoCo Python rendering."""

import os
import pathlib
import struct
import subprocess
import sys
import types
from unittest import mock
import zlib

from absl import flags
from absl.testing import absltest
import mujoco
import numpy as np

_EGL_ENABLED = (
    getattr(getattr(mujoco, 'GLContext', None), '__module__', None)
    == 'mujoco.egl'
)


def _png(pixels: np.ndarray) -> bytes:
  """Encodes an (H, W, 3) uint8 array as an RGB PNG."""

  def chunk(tag, payload):
    data = tag + payload
    return struct.pack('>I', len(payload)) + data + struct.pack(
        '>I', zlib.crc32(data)
    )

  h, w, _ = pixels.shape
  raw = b''.join(b'\x00' + pixels[i].tobytes() for i in range(h))
  return (
      b'\x89PNG\r\n\x1a\n'
      + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
      + chunk(b'IDAT', zlib.compress(raw))
      + chunk(b'IEND', b'')
  )


def _textured_plane_xml(png: str, passthrough: str) -> str:
  return f"""
  <mujoco>
    <compiler passthrough="{passthrough}"/>
    <asset>
      <texture name="tex" type="2d" file="{png}"/>
      <material name="tex" texture="tex"/>
    </asset>
    <worldbody>
      <light pos="0 0 3"/>
      <camera pos="0 0 2"/>
      <geom type="plane" size="1 1 .1" material="tex"/>
    </worldbody>
  </mujoco>
  """


@absltest.skipUnless(
    hasattr(mujoco, 'GLContext'), 'MuJoCo rendering is disabled'
)
class MuJoCoRenderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mark flags as parsed to avoid pytest errors about unparsed flags.
    # This is needed for `create_tempdir()` calls below.
    flags.FLAGS.mark_as_parsed()
    self.gl = mujoco.GLContext(640, 480)
    self.gl.make_current()

  def tearDown(self):
    super().tearDown()
    del self.gl

  def test_can_render(self):
    """Test that the bindings can successfully render a simple image.

    This test sets up a basic MuJoCo rendering context similar to the example in
    https://mujoco.readthedocs.io/en/latest/programming#visualization
    It calls `mjr_rectangle` rather than `mjr_render` so that we can assert an
    exact rendered image without needing golden data. The purpose of this test
    is to ensure that the bindings can correctly return pixels in Python, rather
    than to test MuJoCo's rendering pipeline itself.
    """

    self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    self.data = mujoco.MjData(self.model)

    scene = mujoco.MjvScene(self.model, maxgeom=0)
    mujoco.mjv_updateScene(
        self.model,
        self.data,
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

    # MuJoCo's default render buffer size is 640x480.
    full_rect = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjr_rectangle(full_rect, 0, 0, 0, 1)

    blue_rect = mujoco.MjrRect(56, 67, 234, 123)
    mujoco.mjr_rectangle(blue_rect, 0, 0, 1, 1)

    expected_upside_down_image = np.zeros((480, 640, 3), dtype=np.uint8)
    expected_upside_down_image[67 : 67 + 123, 56 : 56 + 234, 2] = 255

    upside_down_image = np.empty((480, 640, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(upside_down_image, None, full_rect, context)
    np.testing.assert_array_equal(upside_down_image, expected_upside_down_image)

    # Check that mjr_readPixels can accept a flattened array.
    upside_down_image[:] = 0
    mujoco.mjr_readPixels(
        np.reshape(upside_down_image, -1), None, full_rect, context
    )
    np.testing.assert_array_equal(upside_down_image, expected_upside_down_image)
    context.free()

  def test_figure_renders_after_hidden_skin(self):
    xml = """
    <mujoco>
      <worldbody>
        <body name="box" pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <deformable>
        <skin name="triangle" group="3"
              vertex="0 0 0  0.2 0 0  0 0.2 0" face="0 1 2">
          <bone body="box" bindpos="0 0 0" bindquat="1 0 0 0"
                vertid="0 1 2" vertweight="1 1 1"/>
        </skin>
      </deformable>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scene = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    full_rect = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjr_render(full_rect, scene, context)

    figure = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(figure)
    figure.flg_legend = 0
    figure.flg_extend = 0
    figure.range[0][:] = (-1.0, 1.0)
    figure.range[1][:] = (-1.0, 1.0)
    figure.linergb[0] = (1.0, 0.0, 1.0)
    figure.linedata[0][0:4] = (-1.0, -1.0, 1.0, 1.0)
    figure.linepnt[0] = 2
    mujoco.mjr_figure(full_rect, figure, context)

    image = np.empty((480, 640, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(image, None, full_rect, context)
    magenta = (image[:, :, 0] > 200) & (image[:, :, 1] < 100) & (
        image[:, :, 2] > 200
    )
    self.assertGreater(np.count_nonzero(magenta), 0)
    context.free()

  def test_safe_to_free_context_twice(self):
    self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    self.data = mujoco.MjData(self.model)

    scene = mujoco.MjvScene(self.model, maxgeom=0)
    mujoco.mjv_updateScene(
        self.model,
        self.data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

    context.free()
    context.free()

  def _render(self, model, context=None):
    """Renders model from its first camera, in context or a context of its own."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=10)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = 0
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    own_context = context is None
    if own_context:
      context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    full_rect = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjr_render(full_rect, scene, context)
    image = np.empty((480, 640, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(image, None, full_rect, context)
    if own_context:
      context.free()
    return image

  def _write_png(self, name, color):
    """Writes a 4x4 PNG of one color to the test tmpdir, returns its path."""
    if not hasattr(self, '_tmpdir'):
      # one directory per test, so that a second write replaces the first
      self._tmpdir = self.create_tempdir().full_path
    path = os.path.join(self._tmpdir, name)
    with open(path, 'wb') as f:
      f.write(_png(np.full((4, 4, 3), color, dtype=np.uint8)))
    return path

  def test_by_reference_texture_renders_as_embedded(self):
    png = self._write_png('tex.png', (200, 50, 50))
    eager = mujoco.MjModel.from_xml_string(_textured_plane_xml(png, 'none'))
    by_ref = mujoco.MjModel.from_xml_string(
        _textured_plane_xml(png, 'textures')
    )
    self.assertEqual(eager.tex_adr[0], 0)
    self.assertEqual(by_ref.tex_adr[0], -1)

    warning_messages = []
    old_cb = mujoco.get_mju_user_warning()
    try:
      mujoco.set_mju_user_warning(warning_messages.append)
      image_by_ref = self._render(by_ref)
    finally:
      mujoco.set_mju_user_warning(old_cb)
    self.assertEmpty(warning_messages)
    image_eager = self._render(eager)
    self.assertGreater(np.count_nonzero(image_eager), 0)
    np.testing.assert_allclose(image_by_ref, image_eager, atol=1)

  def test_by_reference_texture_missing_file_warns(self):
    png = os.path.join(absltest.get_default_test_tmpdir(), 'missing.png')
    model = mujoco.MjModel.from_xml_string(
        _textured_plane_xml(png, 'textures')
    )
    self.assertEqual(model.tex_adr[0], -1)

    # one warning, and the material renders untextured rather than crashing
    warning_messages = []
    old_cb = mujoco.get_mju_user_warning()
    try:
      mujoco.set_mju_user_warning(warning_messages.append)
      image = self._render(model)
    finally:
      mujoco.set_mju_user_warning(old_cb)
    self.assertLen(warning_messages, 1)
    self.assertIn('texture 0: not uploaded', warning_messages[0])
    self.assertGreater(np.count_nonzero(image), 0)

  def test_upload_texture_rereads_by_reference_file(self):
    png = self._write_png('tex_reread.png', (200, 50, 50))
    model = mujoco.MjModel.from_xml_string(
        _textured_plane_xml(png, 'textures')
    )
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    image_red = self._render(model, context)

    # overwrite the file and upload again: the new pixels are picked up
    self._write_png('tex_reread.png', (50, 50, 200))
    mujoco.mjr_uploadTexture(model, context, 0)
    image_blue = self._render(model, context)
    context.free()
    self.assertFalse(np.array_equal(image_red, image_blue))

    png_blue = self._write_png('tex_blue.png', (50, 50, 200))
    eager = mujoco.MjModel.from_xml_string(
        _textured_plane_xml(png_blue, 'none')
    )
    np.testing.assert_allclose(image_blue, self._render(eager), atol=1)

  def test_mjrrect_repr(self):
    rect = mujoco.MjrRect(1, 2, 3, 4)
    rect_repr = repr(rect)
    self.assertIn('MjrRect', rect_repr)
    self.assertIn('left: 1', rect_repr)

  def test_pixel_buffer_validation(self):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    try:
      mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
      rect = mujoco.MjrRect(0, 0, 10, 10)

      with self.assertRaisesRegex(ValueError, 'rgb buffer is too small'):
        mujoco.mjr_readPixels(
            np.empty(10 * 10 * 3 - 1, dtype=np.uint8), None, rect, context
        )
      with self.assertRaisesRegex(ValueError, 'depth buffer is too small'):
        mujoco.mjr_readPixels(
            None, np.empty(10 * 10 - 1, dtype=np.float32), rect, context
        )
      with self.assertRaisesRegex(ValueError, 'rgb buffer must be C-contiguous'):
        mujoco.mjr_readPixels(
            np.empty((20, 20, 3), dtype=np.uint8)[::2, ::2, :],
            None,
            rect,
            context,
        )
      with self.assertRaisesRegex(
          ValueError, 'depth buffer must be C-contiguous'
      ):
        mujoco.mjr_readPixels(
            None,
            np.empty((20, 20), dtype=np.float32)[::2, ::2],
            rect,
            context,
        )

      with self.assertRaisesRegex(ValueError, 'rgb buffer is too small'):
        mujoco.mjr_drawPixels(
            np.empty(10 * 10 * 3 - 1, dtype=np.uint8), None, rect, context
        )
      with self.assertRaisesRegex(ValueError, 'depth buffer is too small'):
        mujoco.mjr_drawPixels(
            None, np.empty(10 * 10 - 1, dtype=np.float32), rect, context
        )

      neg_rect = mujoco.MjrRect(0, 0, -1, 10)
      with self.assertRaisesRegex(ValueError, 'non-negative'):
        mujoco.mjr_readPixels(None, None, neg_rect, context)
      with self.assertRaisesRegex(ValueError, 'non-negative'):
        mujoco.mjr_drawPixels(None, None, neg_rect, context)

      # Dimensions whose product overflows 32-bit int (65536 * 65536 = 2^32).
      huge_rect = mujoco.MjrRect(0, 0, 65536, 65536)
      with self.assertRaisesRegex(ValueError, 'rgb buffer is too small'):
        mujoco.mjr_readPixels(
            np.empty(3, dtype=np.uint8), None, huge_rect, context
        )
      with self.assertRaisesRegex(ValueError, 'depth buffer is too small'):
        mujoco.mjr_drawPixels(
            None, np.empty(1, dtype=np.float32), huge_rect, context
        )
    finally:
      context.free()


@absltest.skipUnless(_EGL_ENABLED, 'MuJoCo is not using the EGL backend')
class EglDisplayTest(absltest.TestCase):
  """Tests for the lifetime of the shared EGL display."""

  def setUp(self):
    super().setUp()
    self.egl = sys.modules['mujoco.egl']

  def test_free_after_display_terminated(self):
    gl = mujoco.GLContext(64, 64)
    gl.make_current()
    # This is what the atexit hook runs before contexts still alive are freed.
    self.egl._terminate_display()  # pylint: disable=protected-access
    self.assertIsNone(self.egl.EGL_DISPLAY)
    gl.free()

    # New contexts initialize the display again.
    gl = mujoco.GLContext(64, 64)
    gl.make_current()
    gl.free()

  def test_renderer_close_after_display_terminated(self):
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    renderer = mujoco.Renderer(model, 64, 64)
    self.egl._terminate_display()  # pylint: disable=protected-access
    self.assertIsNone(self.egl.EGL_DISPLAY)
    renderer.close()

    with mujoco.Renderer(model, 64, 64):
      self.assertIsNotNone(self.egl.EGL_DISPLAY)

  def test_free_non_current_context_preserves_current(self):
    gl1 = mujoco.GLContext(64, 64)
    gl2 = mujoco.GLContext(64, 64)
    try:
      gl2.make_current()
      expected_addr = self.egl.EGL.eglGetCurrentContext().address
      gl1.free()
      current = self.egl.EGL.eglGetCurrentContext()
      self.assertTrue(bool(current))
      self.assertEqual(current.address, expected_addr)
    finally:
      gl1.free()
      gl2.free()

  def test_failed_display_init_is_retried(self):
    no_display = mock.patch.object(
        self.egl,
        'create_initialized_egl_device_display',
        return_value=self.egl.EGL.EGL_NO_DISPLAY,
    )
    with mock.patch.object(self.egl, 'EGL_DISPLAY', None):
      with no_display:
        for _ in range(2):
          with self.assertRaises(ImportError):
            mujoco.GLContext(64, 64)
          self.assertIsNone(self.egl.EGL_DISPLAY)

      # Once a display is available, initialization succeeds.
      gl = mujoco.GLContext(64, 64)
      gl.make_current()
      gl.free()

  @absltest.skipIf(not sys.executable, 'needs a Python interpreter binary')
  def test_no_error_at_exit_with_live_context(self):
    code = 'import mujoco; gl = mujoco.GLContext(64, 64); gl.make_current()'
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True
    )
    self.assertNotIn('Exception ignored', result.stderr)


class ExternalGLContextLazyImportTest(absltest.TestCase):
  """Tests the external gl_context.py block when deps are missing."""

  def _exec_external_gl_context(self, env, modules):
    gl_context_path = (
        pathlib.Path(mujoco.__file__).parent
        / 'rendering'
        / 'classic'
        / 'gl_context.py'
    )
    lines = gl_context_path.read_text().splitlines()
    if '# _SYSTEM = platform.system()' in lines:
      start = lines.index('# _SYSTEM = platform.system()')
      external_code = '\n'.join(
          line[2:] if line.startswith('# ') else line[1:]
          for line in lines[start:-1]
      )
    else:
      external_code = '\n'.join(lines)
    ns = {
        'ctypes': __import__('ctypes'),
        'os': os,
        'platform': __import__('platform'),
    }
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(
        os.environ, env, clear=True
    ):
      exec(external_code, ns)  # pylint: disable=exec-used
    return ns

  def test_missing_backend_defers_import_error_until_instantiation(self):
    fake_glfw = types.ModuleType('mujoco.glfw')

    def _raise_missing_glfw(name):
      if name == 'GLContext':
        raise ModuleNotFoundError("No module named 'glfw'", name='glfw')
      raise AttributeError(name)

    fake_glfw.__getattr__ = _raise_missing_glfw
    ns = self._exec_external_gl_context(
        {'MUJOCO_GL': 'glfw'}, {'mujoco.glfw': fake_glfw}
    )
    self.assertIn('GLContext', ns)
    with self.assertRaisesRegex(
        ImportError, r"Ensure that 'glfw' and 'pyopengl' are installed"
    ) as ctx:
      ns['GLContext'](64, 64)
    self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)

  def test_disabled_mujoco_gl_leaves_gl_context_undefined(self):
    ns = self._exec_external_gl_context({'MUJOCO_GL': 'disable'}, {})
    self.assertNotIn('GLContext', ns)


if __name__ == '__main__':
  absltest.main()
