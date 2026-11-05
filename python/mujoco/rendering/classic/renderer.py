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
"""Defines a renderer class for the MuJoCo Python native bindings."""

import functools
from typing import Optional, Union

import mujoco
from mujoco.rendering.classic import gl_context
import numpy as np

GLContext = gl_context.GLContext if hasattr(gl_context, 'GLContext') else None


@functools.lru_cache(maxsize=8)
def _owned_gl_context_cls(cls):
  """Returns a `GLContext` subclass whose lifetime is managed by `Renderer`."""
  if not isinstance(cls, type):
    return cls
  # Suppress independent `__del__` during PEP 442 cyclic GC (e.g. under
  # free-threaded Python) so `Renderer.__del__` always binds this context via
  # `as_current()` before freeing `MjrContext`, and only then calls `free()`.
  return type(f'_Owned{cls.__name__}', (cls,), {'__del__': lambda self: None})


class Renderer:
  """Renders MuJoCo scenes."""

  def __init__(
      self,
      model: mujoco.MjModel,
      height: int = 240,
      width: int = 320,
      max_geom: int = 10000,
      font_scale: mujoco.mjtFontScale = mujoco.mjtFontScale.mjFONTSCALE_150,
  ) -> None:
    """Initializes a new `Renderer`.

    Args:
      model: an MjModel instance.
      height: image height in pixels.
      width: image width in pixels.
      max_geom: Optional integer specifying the maximum number of geoms that can
        be rendered in the same scene. If None this will be chosen automatically
        based on the estimated maximum number of renderable geoms in the model.
      font_scale: Optional enum specifying the font scale for text.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
    # Pre-initialize context attributes so __del__ -> close() is safe even if
    # __init__ raises below before they are assigned. See #3213.
    self._gl_context = None  # type: ignore
    self._mjr_context = None

    buffer_width = model.vis.global_.offwidth
    buffer_height = model.vis.global_.offheight
    if width > buffer_width:
      raise ValueError(f"""
Image width {width} > framebuffer width {buffer_width}. Either reduce the image
width or specify a larger offscreen framebuffer in the model XML using the
clause:
<visual>
  <global offwidth="my_width"/>
</visual>""".lstrip())

    if height > buffer_height:
      raise ValueError(f"""
Image height {height} > framebuffer height {buffer_height}. Either reduce the
image height or specify a larger offscreen framebuffer in the model XML using
the clause:
<visual>
  <global offheight="my_height"/>
</visual>""".lstrip())

    self._width = width
    self._height = height
    self._model = model

    self._scene = mujoco.MjvScene(model=model, maxgeom=max_geom)
    self._scene_option = mujoco.MjvOption()

    self._rect = mujoco.MjrRect(0, 0, self._width, self._height)

    # Create render contexts.
    # TODO(nimrod): Figure out why pytype doesn't like gl_context.GLContext
    if gl_context.GLContext is not None:
      gl_context_cls = _owned_gl_context_cls(gl_context.GLContext)
      self._gl_context = gl_context_cls(width, height)
    if self._gl_context:
      self._gl_context.make_current()
    self._mjr_context = mujoco.MjrContext(model, font_scale.value)
    mujoco.mjr_setBuffer(
        mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context
    )
    self._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR  # pyrefly: ignore[bad-assignment]

    # Default render flags.
    self._depth_rendering = False
    self._segmentation_rendering = False

  @property
  def model(self):
    return self._model

  @property
  def scene(self) -> mujoco.MjvScene:
    return self._scene

  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  def enable_depth_rendering(self):
    self._segmentation_rendering = False
    self._depth_rendering = True

  def disable_depth_rendering(self):
    self._depth_rendering = False

  def enable_segmentation_rendering(self):
    self._segmentation_rendering = True
    self._depth_rendering = False

  def disable_segmentation_rendering(self):
    self._segmentation_rendering = False

  def render(self, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Renders the scene as a numpy array of pixel values.

    Args:
      out: Alternative output array in which to place the resulting pixels. It
        must have the same shape as the expected output but the type will be
        cast if necessary. The expected shape depends on the rendering mode:
        `(height, width)` when `self._depth_rendering` is `True`,
        `(height, width, 2)` when `self._segmentation_rendering` is `True`, and
        `(height, width, 3)` otherwise.

    Returns:
      A new numpy array holding the pixels with shape `(H, W)`, `(H, W, 2)`, or
      `(H, W, 3)` depending on the rendering mode unless `out is not None`, in
      which case a reference to `out` is returned.

    Raises:
      RuntimeError: if this method is called after the close method.
    """
    original_flags = self._scene.flags.copy()
    try:
      # Using segmented rendering for depth makes the calculated depth more
      # accurate at far distances.
      if self._depth_rendering or self._segmentation_rendering:
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True

      if self._mjr_context is None:
        raise RuntimeError('render cannot be called after close.')

      if self._gl_context:
        self._gl_context.make_current()

      if self._depth_rendering:
        out_shape = (self._height, self._width)
        out_dtype = np.float32
      elif self._segmentation_rendering:
        out_shape = (self._height, self._width, 2)
        out_dtype = np.int32
      else:
        out_shape = (self._height, self._width, 3)
        out_dtype = np.uint8

      if out is not None and out.shape != out_shape:
        raise ValueError(
            f'Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}`'
            ' instead. When using depth rendering, the out array should be of'
            ' shape `(height, width)`, when using segmentation rendering'
            ' `(height, width, 2)`, and otherwise `(height, width, 3)`.'
            f' Got `(self.height, self.width)={(self.height, self.width)}`,'
            f' `self._depth_rendering={self._depth_rendering}`, and'
            f' `self._segmentation_rendering={self._segmentation_rendering}`.'
        )

      if out is None or out.dtype != out_dtype or not out.flags.c_contiguous:
        buffer = np.empty(out_shape, dtype=out_dtype)
      else:
        buffer = out

      # Render scene and read contents of RGB and depth buffers.
      mujoco.mjr_render(self._rect, self._scene, self._mjr_context)
      if self._depth_rendering:
        mujoco.mjr_readPixels(None, buffer, self._rect, self._mjr_context)

        # Get the distances to the near and far clipping planes.
        extent = self._model.stat.extent
        near = self._model.vis.map.znear * extent
        far = self._model.vis.map.zfar * extent

        # Calculate OpenGL perspective matrix values in float32 precision
        # so they are close to what glFrustum returns
        # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
        zfar = np.float32(far)
        znear = np.float32(near)

        # We need 64 bits to convert Z from ndc to metric depth without noticeable
        # losses in precision
        out_64 = buffer.astype(np.float64)

        if self._scene.camera[0].orthographic:
          # Orthographic cameras: mapping is linear
          out_64 = zfar - out_64 * (zfar - znear)
        else:
          # Perspective cameras: mapping is hyperbolic
          c_coef = -(zfar + znear) / (zfar - znear)
          d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)

          # In reverse Z mode the perspective matrix is transformed by
          c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
          d_coef = np.float32(-0.5) * d_coef

          # Undo OpenGL projection
          # Note: We do not need to take action to convert from window coordinates
          # to normalized device coordinates because in reversed Z mode
          # the mapping is identity
          out_64 = d_coef / (out_64 + c_coef)

        # Cast result back to float32 for backwards compatibility
        # This has a small accuracy cost
        buffer[:] = out_64.astype(np.float32)
      elif self._segmentation_rendering:
        rgb_buffer = np.empty((self._height, self._width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_buffer, None, self._rect, self._mjr_context)

        # Convert 3-channel uint8 to 1-channel uint32.
        image3 = rgb_buffer.astype(np.uint32)
        segimage = (
            image3[:, :, 0]
            + image3[:, :, 1] * (2**8)
            + image3[:, :, 2] * (2**16)
        )
        # Remap segid to 2-channel (object ID, object type) pair.
        # Seg ID 0 is background -- will be remapped to (-1, -1).
        ngeoms = self._scene.ngeom
        segid2output = np.full(
            (ngeoms + 1, 2), fill_value=-1, dtype=np.int32
        )  # Seg id cannot be > ngeom + 1.
        visible_geoms = [g for g in self._scene.geoms[:ngeoms] if g.segid != -1]
        visible_segids = np.array(
            [g.segid + 1 for g in visible_geoms], np.int32
        )
        visible_objid = np.array([g.objid for g in visible_geoms], np.int32)
        visible_objtype = np.array([g.objtype for g in visible_geoms], np.int32)
        segid2output[visible_segids, 0] = visible_objid
        segid2output[visible_segids, 1] = visible_objtype
        buffer[:] = segid2output[segimage]
      else:
        mujoco.mjr_readPixels(buffer, None, self._rect, self._mjr_context)

      if self._gl_context:
        # If using EGL, OSMesa, or GLFW, the output image is flipped vertically.
        # No flip is needed for Filament.
        buffer[:] = np.flipud(buffer)

      if out is not None:
        if out is not buffer:
          np.copyto(out, buffer, casting='unsafe')
        return out
      return buffer
    finally:
      np.copyto(self._scene.flags, original_flags)

  def update_scene(
      self,
      data: mujoco.MjData,
      camera: Union[int, str, mujoco.MjvCamera] = -1,
      scene_option: Optional[mujoco.MjvOption] = None,
  ):
    """Updates geometry used for rendering.

    Args:
      data: An instance of `MjData`.
      camera: An instance of `MjvCamera`, a string or an integer
      scene_option: A custom `MjvOption` instance to use to render the scene
        instead of the default.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if camera does
        not exist.
    """
    if not isinstance(camera, mujoco.MjvCamera):
      camera_id = camera
      if isinstance(camera_id, str):
        camera_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA.value, camera_id
        )
        if camera_id == -1:
          raise ValueError(f'The camera "{camera}" does not exist.')
      if camera_id < -1 or camera_id >= self._model.ncam:
        raise ValueError(
            f'The camera id {camera_id} is out of'
            f' range [-1, {self._model.ncam}).'
        )

      # Render camera.
      camera = mujoco.MjvCamera()
      camera.fixedcamid = camera_id

      # Defaults to mjCAMERA_FREE, otherwise mjCAMERA_FIXED refers to a
      # camera explicitly defined in the model.
      if camera_id == -1:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE  # pyrefly: ignore[bad-assignment]
        mujoco.mjv_defaultFreeCamera(self._model, camera)
      else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED  # pyrefly: ignore[bad-assignment]

    scene_option = scene_option or self._scene_option
    mujoco.mjv_updateScene(
        self._model,
        data,
        scene_option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL.value,
        self._scene,
    )

  def close(self) -> None:
    """Frees the resources used by the renderer.

    This method can be used directly:

    ```python
    renderer = Renderer(...)
    # Use renderer.
    renderer.close()
    ```

    or via a context manager:

    ```python
    with Renderer(...) as renderer:
      # Use renderer.
    ```
    """
    if self._mjr_context:
      if self._gl_context:
        with self._gl_context.as_current():
          self._mjr_context.free()
      else:
        self._mjr_context.free()
    self._mjr_context = None
    if self._gl_context:
      self._gl_context.free()
    self._gl_context = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self) -> None:
    self.close()
