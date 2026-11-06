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
"""Class for coordinating the rendering of scenes to render targets."""

import dataclasses
from typing import Any, Protocol

import mujoco
from mujoco import mjrf
import numpy as np
from typing_extensions import Buffer


class SupportsDLPack(Protocol):
  """Protocol for objects supporting the DLPack tensor exchange protocol."""

  def __dlpack__(self, /, *, stream: Any = None) -> Any:
    ...

  def __dlpack_device__(self, /) -> tuple[int, int]:
    ...


BufferType = Buffer | SupportsDLPack


def _extract_buffer(buffer: BufferType) -> Buffer:
  """Extracts a writable CPU buffer from a Buffer or DLPack object."""
  if isinstance(buffer, Buffer):
    return buffer
  return np.from_dlpack(buffer, copy=False)


def _format_info(pixel_format: mjrf.PixelFormat) -> tuple[int, np.dtype]:
  match pixel_format:
    case mjrf.PixelFormat.PIXEL_FORMAT_R8:
      return 1, np.dtype(np.uint8)
    case mjrf.PixelFormat.PIXEL_FORMAT_RGB8:
      return 3, np.dtype(np.uint8)
    case mjrf.PixelFormat.PIXEL_FORMAT_RGBA8:
      return 4, np.dtype(np.uint8)
    case mjrf.PixelFormat.PIXEL_FORMAT_R32F:
      return 1, np.dtype(np.float32)
    case mjrf.PixelFormat.PIXEL_FORMAT_DEPTH32F:
      return 1, np.dtype(np.float32)
    case _:
      raise ValueError(f"Unsupported pixel format: {pixel_format}")


def _validate_buffer(
    raw_buffer: Buffer,
    pixel_format: mjrf.PixelFormat,
    name: str,
) -> tuple[int, int]:
  """Validates raw_buffer shape and dtype, returning (width, height)."""
  channels, expected_dtype = _format_info(pixel_format)
  mv = memoryview(raw_buffer)
  shape = mv.shape
  if channels == 1:
    valid_shape = shape is not None and (
        len(shape) == 2 or (len(shape) == 3 and shape[2] == 1)
    )
    expected_shape = "(H, W) or (H, W, 1)"
  else:
    valid_shape = shape is not None and len(shape) == 3 and shape[2] == channels
    expected_shape = f"(H, W, {channels})"
  if shape is None or not valid_shape:
    raise ValueError(
        f"Buffer shape {shape} does not match expected shape"
        f" {expected_shape} for target '{name}'"
    )
  height, width = shape[0], shape[1]
  if width <= 0 or height <= 0:
    raise ValueError(
        f"Buffer dimensions must be positive, got shape {shape} for target"
        f" '{name}'"
    )
  if np.dtype(mv.format) != expected_dtype:
    raise ValueError(
        f"Buffer dtype {np.dtype(mv.format)} does not match expected dtype"
        f" {expected_dtype} for target '{name}'"
    )
  return width, height


@dataclasses.dataclass
class View:
  """A representation of an mjrf.RenderRequest that works with the Renderer."""
  scene: str
  target: str
  viewport: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
  camera: mjrf.Camera | mujoco.MjvGLCamera | None = None
  draw_mode: mjrf.DrawMode = mjrf.DrawMode.DRAW_MODE_DEFAULT
  post_processing: bool = True
  reflections: bool = True
  shadows: bool = True
  skybox: bool = True
  haze: bool = True
  headlight: bool = False
  headlight_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
  headlight_intensity: float = 0.0


class Renderer:
  """Coordinates the rendering of scenes to render targets.

  Internally, the Renderer manages the mjrf objects (i.e. Scene, Camera, and
  RenderTarget) that are needed to render a MuJoCo model. The Scene defines
  the entities from the model that are to be rendered. The Camera defines the
  viewpoint from which to render the model. And, finally, the RenderTarget
  defines the memory buffers into which the final rendered image will be
  written.

  Each object owned by the Renderer is associated with a unique name. This name
  is used to identify the object when making rendering calls.

  The Renderer does not specify the contents of the Scene; it is up to the user
  to populate the Scene with the entities that they want to render.

  The Renderer is not limited to rendering a single scene to a single target.
  You can configure the Renderer to manage multiple scenes from multiple
  cameras into multiple targets. Each such render operation is specified using
  the view() method.
  """

  def __init__(self, ctx: mjrf.Context):
    self._ctx = ctx
    self._scenes: dict[str, mjrf.Scene] = {}
    self._reads: dict[str, Renderer._Read] = {}
    self._ctx.set_clear_color([0.0, 0.0, 0.0])

  @property
  def context(self) -> mjrf.Context:
    """Returns the underlying mjrf.Context."""
    return self._ctx

  def scene(self, name: str) -> mjrf.Scene:
    """Returns the scene with the given name, creating it if needed."""
    if name not in self._scenes:
      self._scenes[name] = self._ctx.create_scene(mjrf.SceneParams())
    return self._scenes[name]

  def target(
      self,
      name: str,
      buffer: BufferType,
      pixel_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8,
  ):
    """Defines a render target with the given name and destination buffer."""
    raw_buffer = _extract_buffer(buffer)
    size = _validate_buffer(raw_buffer, pixel_format, name)

    if name not in self._reads:
      target = self._ctx.create_render_target(
          mjrf.RenderTargetConfig(
              color_format=pixel_format,
          )
      )
      read = self._Read(
          size=size,
          target=target,
          format=pixel_format,
          request=mjrf.ReadRequest(target=target),
          buffer=(buffer, raw_buffer),
      )
      read.request.set_buffer(raw_buffer)
      self._reads[name] = read
    else:
      read = self._reads[name]
      if read.format != pixel_format:
        raise ValueError(
            f"Target {name} already exists with a different pixel format."
        )
      read.request.set_buffer(raw_buffer)
      read.buffer = (buffer, raw_buffer)

    read.size = size
    read.target.resize(size[0], size[1])
    return read.target

  def render(self, views: list[View]):
    """Renders all active views."""
    requests: list[mjrf.RenderRequest] = []
    reads: dict[str, mjrf.ReadRequest] = {}
    for view in views:
      read = self._reads.get(view.target, None)
      if read is None:
        raise ValueError(f"Target not found: {view.target}")
      reads[view.target] = read.request

      scene = self._scenes.get(view.scene, None)
      if scene is None:
        raise ValueError(f"Scene not found: {view.scene}")

      req = mjrf.RenderRequest()
      req.scene = scene
      req.target = read.request.target
      req.camera = self._to_mjrf_camera(view.camera)
      req.viewport.left = int(view.viewport[0] * read.size[0])
      req.viewport.bottom = int(view.viewport[1] * read.size[1])
      req.viewport.width = int(view.viewport[2] * read.size[0])
      req.viewport.height = int(view.viewport[3] * read.size[1])
      req.draw_mode = view.draw_mode
      req.post_processing = view.post_processing
      req.reflections = view.reflections
      req.shadows = view.shadows
      req.skybox = view.skybox
      req.haze = view.haze
      req.headlight = view.headlight
      req.headlight_color = view.headlight_color
      req.headlight_intensity = view.headlight_intensity
      requests.append(req)

    frame = self._ctx.render(requests, reads.values())
    self._ctx.wait_for_frame(frame)

  def _to_mjrf_camera(
      self,
      camera: mjrf.Camera | mujoco.MjvGLCamera | None,
  ) -> mjrf.Camera:
    """Converts a MjvGLCamera to an mjrf.Camera."""
    if camera is None:
      return mjrf.Camera()
    if isinstance(camera, mujoco.MjvGLCamera):
      # In C, mjrCamera is a type alias for mjvGLCamera. However, they are
      # bound as distinct types in Python, so we convert from MjvGLCamera to
      # mjrf.Camera here.
      camera = mjrf.Camera(
          pos=camera.pos,
          forward=camera.forward,
          up=camera.up,
          frustum_bottom=camera.frustum_bottom,
          frustum_top=camera.frustum_top,
          frustum_near=camera.frustum_near,
          frustum_far=camera.frustum_far,
      )
    return camera

  @dataclasses.dataclass
  class _Read:
    """Internal wrapper around mjrf.ReadRequest."""
    size: tuple[int, int]
    target: mjrf.RenderTarget
    format: mjrf.PixelFormat
    request: mjrf.ReadRequest
    buffer: tuple[BufferType, Buffer]
