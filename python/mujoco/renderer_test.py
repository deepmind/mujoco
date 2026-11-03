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
"""Tests for the MuJoCo renderer."""

import gc
import sys

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import numpy as np


@absltest.skipUnless(
    hasattr(mujoco, 'GLContext'), 'MuJoCo rendering is disabled'
)
class MuJoCoRendererTest(parameterized.TestCase):

  def test_renderer_info_binding(self):
    info = mujoco.MjrRendererInfo()
    mujoco.mjr_getRendererInfo(info)

    self.assertIn(info.renderer, ('classic', 'filament', 'noop'))
    self.assertIn(info.backend, ('', 'opengl', 'vulkan', 'unknown'))

  def test_renderer_unknown_camera_name(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="a"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, 50, 50) as renderer:
      mujoco.mj_forward(model, data)
      with self.assertRaisesRegex(ValueError, r'camera "b" does not exist'):
        renderer.update_scene(data, 'b')

  def test_renderer_camera_under_range(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="a"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, 50, 50) as renderer:
      mujoco.mj_forward(model, data)
      with self.assertRaisesRegex(ValueError, '-2 is out of range'):
        renderer.update_scene(data, -2)

  def test_renderer_camera_over_range(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="a"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, 50, 50) as renderer:
      mujoco.mj_forward(model, data)
      with self.assertRaisesRegex(ValueError, '1 is out of range'):
        renderer.update_scene(data, 1)

  def test_renderer_renders_scene(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="closeup" pos="0 -6 0" xyaxes="1 0 0 0 1 100"/>
    <geom name="white_box" type="box" size="1 1 1" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, 50, 50) as renderer:
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, 'closeup')

      pixels = renderer.render().flatten()
      not_all_black = False

      # Pixels should all be a neutral color.
      for pixel in pixels:
        if pixel > 0:
          not_all_black = True
          break
      self.assertTrue(not_all_black)

  def test_renderer_output_without_out(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="closeup" pos="0 -6 0" xyaxes="1 0 0 0 1 100"/>
    <geom name="white_box" type="box" size="1 1 1" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, 50, 50) as renderer:
      renderer.update_scene(data, 'closeup')
      pixels = [renderer.render()]

      colors = (
          (1.0, 0.0, 0.0, 1.0),
          (0.0, 1.0, 0.0, 1.0),
          (0.0, 0.0, 1.0, 1.0),
      )

      for i, color in enumerate(colors):
        model.geom_rgba[0, :] = color
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, 'closeup')
        pixels.append(renderer.render())
        self.assertIsNot(pixels[-2], pixels[-1])

        # Pixels should change over steps.
        self.assertFalse((pixels[i + 1] == pixels[i]).all())

  def test_renderer_output_with_out(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="closeup" pos="0 -6 0" xyaxes="1 0 0 0 1 100"/>
    <geom name="white_box" type="box" size="1 1 1" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""
    render_size = (50, 50)
    render_out = np.zeros((*render_size, 3), np.uint8)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, *render_size) as renderer:
      renderer.update_scene(data, 'closeup')

      self.assertTrue(np.all(render_out == 0))

      pixels = renderer.render(out=render_out)

      # Pixels should always refer to the same `render_out` array.
      self.assertIs(pixels, render_out)
      self.assertFalse(np.all(render_out == 0))

      failing_render_size = (10, 10)
      self.assertNotEqual(failing_render_size, render_size)
      with self.assertRaises(ValueError):
        renderer.render(out=np.zeros((*failing_render_size, 3), np.uint8))

  def test_renderer_output_with_cast_and_non_contiguous_out(self):
    xml = """
<mujoco>
  <worldbody>
    <camera name="closeup" pos="0 -6 0" xyaxes="1 0 0 0 1 100"/>
    <geom name="white_box" type="box" size="1 1 1" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""
    render_size = (50, 50)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, *render_size) as renderer:
      renderer.update_scene(data, 'closeup')
      expected_rgb = renderer.render()

      rgb_int32 = np.zeros((*render_size, 3), dtype=np.int32)
      pixels = renderer.render(out=rgb_int32)
      self.assertIs(pixels, rgb_int32)
      np.testing.assert_array_equal(rgb_int32, expected_rgb.astype(np.int32))

      rgb_fortran = np.asfortranarray(
          np.zeros((*render_size, 3), dtype=np.uint8)
      )
      pixels = renderer.render(out=rgb_fortran)
      self.assertIs(pixels, rgb_fortran)
      np.testing.assert_array_equal(rgb_fortran, expected_rgb)

      renderer.enable_depth_rendering()
      expected_depth = renderer.render()
      depth_float64 = np.zeros(render_size, dtype=np.float64)
      depth_pixels = renderer.render(out=depth_float64)
      self.assertIs(depth_pixels, depth_float64)
      np.testing.assert_array_equal(
          depth_float64, expected_depth.astype(np.float64)
      )

      renderer.enable_segmentation_rendering()
      expected_seg = renderer.render()
      seg_int64 = np.zeros((*render_size, 2), dtype=np.int64)
      seg_pixels = renderer.render(out=seg_int64)
      self.assertIs(seg_pixels, seg_int64)
      np.testing.assert_array_equal(seg_int64, expected_seg.astype(np.int64))

  def test_renderer_restores_scene_flags_on_exception(self):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    with mujoco.Renderer(model, 50, 50) as renderer:
      renderer.enable_depth_rendering()
      original_flags = renderer.scene.flags.copy()
      with self.assertRaises(ValueError):
        renderer.render(out=np.zeros((10, 10), dtype=np.float32))
      np.testing.assert_array_equal(renderer.scene.flags, original_flags)

      renderer.close()
      with self.assertRaises(RuntimeError):
        renderer.render()
      np.testing.assert_array_equal(renderer.scene.flags, original_flags)

  def test_renderer_depth_rendering_camera_projection(self):
    """Tests orthographic vs perspective camera depth rendering projections.

    Orthographic depth uses a linear NDC mapping (glOrtho); perspective depth
    uses a hyperbolic mapping (glFrustum).

    Render the same tilted plane using identically positioned orthographic and
    perspective cameras. The two should agree on the depth at the center pixel
    (actual distance), but they should have different depth values at non-center
    pixels because the rays of the perspective camera diverge while those of the
    orthographic camera remain parallel.
    """
    distance = 3.0
    ortho_fovy = 2.0 * distance * np.tan(np.deg2rad(45.0 / 2.0))
    xml = f"""
<mujoco>
  <statistic extent="1"/>
  <visual>
    <quality offsamples="0"/>
  </visual>
  <worldbody>
    <camera name="ortho" pos="0 0 {distance}" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="{ortho_fovy}"/>
    <camera name="persp" pos="0 0 {distance}" xyaxes="1 0 0 0 1 0" projection="perspective" fovy="45"/>
    <geom name="floor" type="plane" size="5 5 0.1" zaxis="0 0.3 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Odd resolution so a single pixel sits exactly on the optical axis.
    def render_depth(camera_name):
      with mujoco.Renderer(model, 51, 51) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera_name)
        return renderer.scene.camera[0].orthographic, renderer.render().copy()

    is_ortho, ortho_depth = render_depth('ortho')
    is_persp, persp_depth = render_depth('persp')

    # Check that the scenes are actually rendered with the expected cameras.
    self.assertTrue(is_ortho)
    self.assertFalse(is_persp)

    # Center pixels of both cameras should show the actual distance
    center = ortho_depth.shape[0] // 2, ortho_depth.shape[1] // 2
    self.assertAlmostEqual(ortho_depth[center], distance, delta=1e-2)
    self.assertAlmostEqual(persp_depth[center], distance, delta=1e-2)

    # Depth values at the corner should differ
    corner = (0, 0)
    self.assertNotAlmostEqual(
        ortho_depth[corner], persp_depth[corner], delta=1e-2
    )

  @parameterized.parameters('persp', 'ortho')
  def test_mjr_read_pixels_depth_map(self, camera_name):
    """Tests mjr_readPixels raw depth mapping and far-plane clipping."""
    znear = 0.1
    zfar = 10.0
    xml = f"""
<mujoco>
  <statistic extent="1"/>
  <visual>
    <map znear="{znear}" zfar="{zfar}"/>
    <quality offsamples="0"/>
  </visual>
  <worldbody>
    <camera name="persp" pos="0 0 3" xyaxes="1 0 0 0 1 0"
      projection="perspective" fovy="45"/>
    <camera name="ortho" pos="0 0 3" xyaxes="1 0 0 0 1 0"
      projection="orthographic" fovy="2"/>
    <geom name="near_box" type="box" pos="0 0 1" size="0.3 0.3 0.5"/>
    <geom name="beyond_far" type="box" pos="0 0 -9" size="10 10 0.5"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    size = 51
    gl = mujoco.GLContext(size, size) if mujoco.GLContext is not None else None
    try:
      if gl is not None:
        gl.make_current()
      ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
      try:
        self.assertEqual(ctx.readDepthMap, mujoco.mjtDepthMap.mjDEPTH_ZERONEAR)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, ctx)

        scene = mujoco.MjvScene(model=model, maxgeom=10)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA.value, camera_name
        )
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )

        rect = mujoco.MjrRect(0, 0, size, size)
        mujoco.mjr_render(rect, scene, ctx)

        depth_zeronear = np.empty((size, size), dtype=np.float32)
        mujoco.mjr_readPixels(None, depth_zeronear, rect, ctx)

        ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        depth_zerofar = np.empty((size, size), dtype=np.float32)
        mujoco.mjr_readPixels(None, depth_zerofar, rect, ctx)
      finally:
        ctx.free()
    finally:
      if gl is not None:
        gl.free()

    np.testing.assert_allclose(depth_zeronear + depth_zerofar, 1.0, atol=1e-6)

    # Top face of near_box is at world z = 1.5, so planar camera depth is 1.5.
    z_box = 1.5
    if camera_name == 'persp':
      expected_zerofar = znear * (zfar - z_box) / (z_box * (zfar - znear))
    else:
      expected_zerofar = (zfar - z_box) / (zfar - znear)
    expected_zeronear = 1.0 - expected_zerofar

    center = (size // 2, size // 2)
    self.assertAlmostEqual(depth_zerofar[center], expected_zerofar, delta=1e-4)
    self.assertAlmostEqual(
        depth_zeronear[center], expected_zeronear, delta=1e-4
    )

    # Geometry beyond zfar (z_cam = 11.5 > 10.0) must be clipped at the far
    # plane, leaving background corners at far depth.
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
      self.assertAlmostEqual(depth_zerofar[corner], 0.0, delta=1e-6)
      self.assertAlmostEqual(depth_zeronear[corner], 1.0, delta=1e-6)

  def test_renderer_del_safe_when_init_fails_early(self):
    """Regression test for #3213.

    Renderer.__del__ must be safe on a partially-constructed instance when
    __init__ raises before the rendering contexts are assigned. Previously,
    AttributeError from __del__ masked the real __init__ exception.
    """
    xml = """
<mujoco>
  <visual>
    <global offwidth="50" offheight="50"/>
  </visual>
  <worldbody/>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)

    # Capture any exception raised from __del__ on the partially-constructed
    # Renderer. Without the fix, __del__ raises AttributeError, which is
    # funneled through sys.unraisablehook.
    unraisable = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = lambda args: unraisable.append(args)
    try:
      # width > offwidth raises ValueError in __init__ before
      # self._gl_context is assigned.
      with self.assertRaises(ValueError):
        mujoco.Renderer(model, height=50, width=200)
      gc.collect()
    finally:
      sys.unraisablehook = old_hook

    self.assertEqual(
        [u.exc_type.__name__ for u in unraisable],
        [],
        msg=(
            'Renderer.__del__ raised on a partially-constructed instance; '
            'see #3213.'
        ),
    )

  def test_renderer_close_preserves_active_context(self):
    """Closing or GC-collecting a Renderer must not clobber another context."""
    xml = """
<mujoco>
  <worldbody>
    <camera name="closeup" pos="0 -6 0" xyaxes="1 0 0 0 1 100"/>
    <geom name="green_box" type="box" size="1 1 1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer1 = mujoco.Renderer(model, 50, 50)
    renderer1.update_scene(data, 'closeup')
    renderer1.render()

    with mujoco.Renderer(model, 50, 50) as renderer2:
      renderer2.update_scene(data, 'closeup')
      expected = renderer2.render().copy()
      self.assertGreater(np.count_nonzero(expected), 0)

      # Closing renderer1 while renderer2's GL context is current must not
      # delete renderer2's offscreen FBO.
      renderer1.close()
      out = np.zeros_like(expected)
      renderer2.render(out=out)
      np.testing.assert_array_equal(out, expected)

      # Finalizing an unclosed Renderer via cyclic GC while renderer2 is
      # current must likewise preserve renderer2's FBO and active context.
      cycle = []
      cycle.append((cycle, mujoco.Renderer(model, 50, 50)))
      renderer2.render()
      del cycle
      gc.collect()
      out.fill(0)
      renderer2.render(out=out)
      np.testing.assert_array_equal(out, expected)

      # PEP 442 does not guarantee finalizer order within an unreachable cycle
      # (e.g. free-threaded Python can run `_gl_context.__del__` before
      # `Renderer.__del__`). Verify that order explicitly on all builds.
      unclosed = mujoco.Renderer(model, 50, 50)
      renderer2.render()
      if unclosed._gl_context is not None:  # pylint: disable=protected-access
        unclosed._gl_context.__del__()
      unclosed.__del__()
      del unclosed
      out.fill(0)
      renderer2.render(out=out)
      np.testing.assert_array_equal(out, expected)

    # Also verify that a standalone GLContext stays bound across Renderer.close
    # without needing another make_current() call.
    gl = mujoco.GLContext(50, 50)
    try:
      gl.make_current()
      ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
      try:
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, ctx)
        scene = mujoco.MjvScene(model=model, maxgeom=100)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = 0
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )

        other = mujoco.Renderer(model, 50, 50)
        gl.make_current()
        other.close()

        rect = mujoco.MjrRect(0, 0, 50, 50)
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        mujoco.mjr_render(rect, scene, ctx)
        mujoco.mjr_readPixels(rgb, None, rect, ctx)
        self.assertGreater(np.count_nonzero(rgb), 0)
      finally:
        ctx.free()
    finally:
      gl.free()


if __name__ == '__main__':
  absltest.main()
