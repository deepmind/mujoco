# Copyright 2026 The Newton Developers
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

import mujoco
import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import bvh
from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src import warp_util
from mujoco.mjx.third_party.mujoco_warp._src.types import ProjectionType
from mujoco.mjx.third_party.mujoco_warp._src.types import RenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def _convert_texture_data(
  # In:
  width: int,
  adr: int,
  nc: int,
  tex_data_in: wp.array[wp.uint8],
  # Out:
  tex_data_out: wp.array3d[float],
):
  """Convert uint8 texture data to vec4 format for efficient sampling."""
  x, y = wp.tid()
  offset = adr + (y * width + x) * nc
  r = tex_data_in[offset + 0] if nc > 0 else wp.uint8(0)
  g = tex_data_in[offset + 1] if nc > 1 else wp.uint8(0)
  b = tex_data_in[offset + 2] if nc > 2 else wp.uint8(0)
  a = tex_data_in[offset + 3] if nc > 3 else wp.uint8(255)

  tex_data_out[y, x, 0] = float(r) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 1] = float(g) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 2] = float(b) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 3] = float(a) * wp.static(1.0 / 255.0)


def texture_pixels(mjm: mujoco.MjModel, tex_id: int) -> np.ndarray:
  """Host pixels of a MuJoCo model texture as a (height, width, nchannel) uint8 array.

  A by-reference texture (tex_adr == -1) is decoded from its file; a load failure raises.
  """
  tex_adr = mjm.tex_adr[tex_id]
  if tex_adr == -1:
    return mujoco.mj_loadTexture(mjm, tex_id)[0]
  shape = (mjm.tex_height[tex_id], mjm.tex_width[tex_id], mjm.tex_nchannel[tex_id])
  return mjm.tex_data[tex_adr : tex_adr + int(np.prod(shape))].reshape(shape)


def create_warp_texture(mjm: mujoco.MjModel, tex_id: int) -> wp.Texture2D:
  """Create a Warp texture from a MuJoCo model texture data."""
  pixels = texture_pixels(mjm, tex_id)
  tex_height, tex_width, nchannel = pixels.shape
  tex_data = wp.zeros((tex_height, tex_width, 4), dtype=float)

  wp.launch(
    _convert_texture_data,
    dim=(tex_width, tex_height),
    inputs=[tex_width, 0, nchannel, wp.array(pixels.ravel(), dtype=wp.uint8)],
    outputs=[tex_data],
  )
  return wp.Texture2D(tex_data, filter_mode=wp.TextureFilterMode.LINEAR)


@wp.func
def compute_ray(
  # In:
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  px: int,
  py: int,
  znear: float,
) -> tuple[wp.vec3, wp.vec3]:
  """Compute ray vector for a pixel with per-world camera parameters.

  This combines _camera_frustum_bounds and build_primary_rays logic for use
  inside a kernel when camera parameters are batched/randomized across worlds.

  Returns:
    Direction of the ray in camera space, and the offset of the ray from the
    camera's center. The latter is only used for orthographic cameras.
  """
  inv_img_h = 1.0 / float(img_h)

  if projection == ProjectionType.ORTHOGRAPHIC:
    # Compute ray direction
    direction = wp.vec3(0.0, 0.0, -1.0)  # always pointing forward

    # Compute ray offset from center
    aspect = float(img_w) * inv_img_h
    sensor_h = fovy
    sensor_w = sensor_h * aspect
    left = -0.5 * sensor_w
    top = 0.5 * sensor_h
    bottom = -top
    u = (float(px) + 0.5) / float(img_w)
    v = (float(py) + 0.5) * inv_img_h
    x = left + sensor_w * u
    y = top + (bottom - top) * v
    offset = wp.vec3(x, y, 0.0)

  else:  # projection == ProjectionType.PERSPECTIVE:
    # Compute ray direction
    aspect = float(img_w) * inv_img_h
    sensor_h = sensorsize[1]

    # Check if we have intrinsics (sensorsize[1] != 0)
    if sensor_h != 0.0:
      fx = intrinsic[0]
      fy = intrinsic[1]
      cx = intrinsic[2]
      cy = intrinsic[3]
      sensor_w = sensorsize[0]

      target_aspect = aspect
      sensor_aspect = sensor_w / sensor_h
      if target_aspect > sensor_aspect:
        sensor_h = sensor_w / target_aspect
      elif target_aspect < sensor_aspect:
        sensor_w = sensor_h * target_aspect

      inv_fx_znear = znear / fx
      inv_fy_znear = znear / fy
      half_sensor_w = 0.5 * sensor_w
      half_sensor_h = 0.5 * sensor_h
      left = -inv_fx_znear * (half_sensor_w - cx)
      right = inv_fx_znear * (half_sensor_w + cx)
      top = inv_fy_znear * (half_sensor_h - cy)
      bottom = -inv_fy_znear * (half_sensor_h + cy)
    else:
      fovy_rad = fovy * wp.static(wp.pi / 180.0)
      half_height = znear * wp.tan(0.5 * fovy_rad)
      half_width = half_height * aspect
      left = -half_width
      right = half_width
      top = half_height
      bottom = -half_height

    u = (float(px) + 0.5) / float(img_w)
    v = (float(py) + 0.5) * inv_img_h
    x = left + (right - left) * u
    y = top + (bottom - top) * v

    direction = wp.normalize(wp.vec3(x, y, -znear))

    # Ray offset from center not used for perspective cameras
    offset = wp.vec3(0.0, 0.0, 0.0)

  return direction, offset


@wp.func
def pack_rgba_to_uint32(r: float, g: float, b: float, a: float) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return wp.uint32((int(a) << int(24)) | (int(r) << int(16)) | (int(g) << int(8)) | int(b))


@wp.kernel
def unpack_rgb_kernel(
  # In:
  packed: wp.array2d[wp.uint32],
  rgb_adr: wp.array[int],
  camera_index: int,
  # Out:
  rgb_out: wp.array3d[wp.vec3],
):
  """Unpack ABGR uint32 packed pixel data into separate R, G, and B channels."""
  worldid, pixelid = wp.tid()

  xid = pixelid % rgb_out.shape[2]
  yid = pixelid // rgb_out.shape[2]

  rgb_adr_offset = rgb_adr[camera_index]
  val = packed[worldid, rgb_adr_offset + pixelid]
  b = wp.float32(val & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
  g = wp.float32((val >> wp.uint32(8)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
  r = wp.float32((val >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
  rgb_out[worldid, yid, xid] = wp.vec3(r, g, b)


@wp.kernel
def extract_depth_kernel(
  # In:
  depth_data: wp.array2d[float],
  depth_adr: wp.array[int],
  camera_index: int,
  depth_scale: float,
  # Out:
  depth_out: wp.array3d[float],
):
  """Extract the depth data from the render context buffers for a given camera index."""
  worldid, pixelid = wp.tid()
  xid = pixelid % depth_out.shape[2]
  yid = pixelid // depth_out.shape[2]

  depth_adr_offset = depth_adr[camera_index]
  val = depth_data[worldid, depth_adr_offset + pixelid]
  depth_out[worldid, yid, xid] = wp.clamp(val / depth_scale, 0.0, 1.0)


def get_rgb(rc: RenderContext, camera_index: int, rgb_out: wp.array3d[wp.vec3]):
  """Get the RGB data output from the render context buffers for a given camera index.

  Args:
    rc: The render context on device.
    camera_index: The index of the camera to get the RGB data for.
    rgb_out: The output array to store the RGB data in, with shape (nworld, height, width).
  """
  wp.launch(
    unpack_rgb_kernel,
    dim=(rgb_out.shape[0], rgb_out.shape[1] * rgb_out.shape[2]),
    inputs=[rc.rgb_data, rc.rgb_adr, camera_index],
    outputs=[rgb_out],
  )


def get_depth(rc: RenderContext, camera_index: int, depth_scale: float, depth_out: wp.array3d[float]):
  """Get the depth data output from the render context buffers for a given camera index.

  Args:
    rc: The render context on device.
    camera_index: The index of the camera to get the depth data for.
    depth_scale: The scale factor to apply to the depth data.
    depth_out: The output array to store the scaled and clamped depth data in
      with shape (nworld, height, width).
  """
  wp.launch(
    extract_depth_kernel,
    dim=(depth_out.shape[0], depth_out.shape[1] * depth_out.shape[2]),
    inputs=[rc.depth_data, rc.depth_adr, camera_index, depth_scale],
    outputs=[depth_out],
  )


@wp.kernel
def _extract_seg_kernel(
  # In:
  seg_data: wp.array2d[wp.vec2i],
  seg_adr: wp.array[int],
  camera_index: int,
  # Out:
  seg_out: wp.array3d[wp.vec2i],
):
  """Extract per-pixel `(object_id, object_type)` pairs for a camera."""
  worldid, pixelid = wp.tid()
  xid = pixelid % seg_out.shape[2]
  yid = pixelid // seg_out.shape[2]

  seg_adr_offset = seg_adr[camera_index]
  seg_out[worldid, yid, xid] = seg_data[worldid, seg_adr_offset + pixelid]


def get_segmentation(rc: RenderContext, camera_index: int, seg_out: wp.array3d[wp.vec2i]):
  """Get the segmentation data from the render context buffers for a given camera index.

  Each pixel stores MuJoCo-style `(object_id, object_type)` data. Background
  pixels are `(-1, -1)`. Regular geometry hits are `(geom_id, mjOBJ_GEOM)`.
  Flex hits are `(flex_id, mjOBJ_FLEX)`.

  Args:
    rc: The render context on device.
    camera_index: The index of the camera to get the segmentation data for.
    seg_out: The output array to store segmentation data in, with shape
      `(nworld, height, width)` and dtype `wp.vec2i`.
  """
  wp.launch(
    _extract_seg_kernel,
    dim=(seg_out.shape[0], seg_out.shape[1] * seg_out.shape[2]),
    inputs=[rc.seg_data, rc.seg_adr, camera_index],
    outputs=[seg_out],
  )


@wp.kernel
def _build_rays(
  # In:
  offset: int,
  img_w: int,
  img_h: int,
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  znear: float,
  n: int,
  sx: int,
  sy: int,
  # Out:
  ray_out: wp.array[wp.vec3],
  ray_offset_out: wp.array[wp.vec3],
):
  xid, yid = wp.tid()
  ray_dir, ray_offset = compute_ray(
    projection, fovy, sensorsize, intrinsic, img_w * n, img_h * n, xid * n + sx, yid * n + sy, znear
  )
  idx = offset + xid + yid * img_w
  ray_out[idx] = ray_dir
  ray_offset_out[idx] = ray_offset


def create_render_context(
  mjm: mujoco.MjModel,
  nworld: int = 1,
  cam_res: list[tuple[int, int]] | tuple[int, int] | None = None,
  render_rgb: list[bool] | bool | None = None,
  render_depth: list[bool] | bool | None = None,
  render_seg: list[bool] | bool | None = None,
  use_textures: bool = True,
  use_fast_math: bool = True,
  use_shadows: bool = False,
  use_ambient_lighting: bool = True,
  enabled_geom_groups: list[int] = [0, 1, 2],
  cam_active: list[bool] | list[str] | list[int] | None = None,
  background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
  flex_render_smooth: bool = True,
  use_precomputed_rays: bool = True,
  render_skybox: bool = False,
  enable_backface_culling: bool = True,
  shadow_light_fraction: float = 0.3,
  samples_per_pixel: int = 1,
  enable_vertex_normals: bool = True,
  enable_specular: bool = True,
  enable_emission: bool = True,
  enable_per_light_ambient: bool = True,
  splat_position: np.ndarray | None = None,
  splat_rotation: np.ndarray | None = None,
  splat_scale: np.ndarray | None = None,
  splat_rgba: np.ndarray | None = None,
  splat_adr: np.ndarray | None = None,
  splat_group_id: np.ndarray | None = None,
  deterministic: bool = False,
) -> RenderContext:
  """Creates a render context on device.

  Args:
    mjm: The model containing kinematic and dynamic information on host.
    nworld: The number of worlds.
    cam_res: The width and height to render each camera image. If None, uses the
             MuJoCo model values.
    render_rgb: Whether to render RGB images. If None, uses the MuJoCo model values.
    render_depth: Whether to render depth images. If None, uses the MuJoCo model values.
    render_seg: Whether to render segmentation (per-pixel object ID/type pairs).
      If None, uses the MuJoCo model values.
    use_textures: Whether to use textures.
    use_fast_math: Whether to enable fast math for the render kernel.
    use_shadows: Whether to use shadows.
    use_ambient_lighting: Top-level ambient switch. When False, skips all
                          ambient contributions, including headlight ambient,
                          the no-light fallback, and per-light ambient.
    enabled_geom_groups: The geom groups to render.
    cam_active: List of booleans, camera names (str), or camera indices (int) indicating
                which cameras to include in rendering. If None, all cameras are included.
                An empty list includes no cameras.
    flex_render_smooth: Whether to render flex meshes smoothly.
    use_precomputed_rays: Use precomputed rays instead of computing during rendering.
                          When using domain randomization for camera intrinsics, set to False.
    render_skybox: Whether to shade missed rays with the MuJoCo skybox texture.
                   Requires the model to contain a texture with type `mjTEXTURE_SKYBOX`.
    shadow_light_fraction: Fraction of a light's direct contribution reaching an
      occluded point. 0 is a true shadow.
    samples_per_pixel: Sub-pixel samples per axis, averaged. Costs n*n renders.
    enable_vertex_normals: Shade meshes from their authored vertex normals,
      matching mjr_uploadMesh. When False, use the face normal.
    enable_backface_culling: Drop primitive-ray hits whose normal faces away from
                             the ray (ray origin inside the geom). Matches MuJoCo's
                             mesh-ray rule. Default True. Disable for a small
                             performance gain when no camera is ever inside a geom.
    background_color: The color to use for background pixels when no skybox is rendered.
    enable_specular: Evaluate specular highlights per light. When False the
                     half-vector normalize and shininess `pow` are dropped at
                     compile time. Disable for performance when no specular is present.
    enable_emission: Add `mat_emission * base_color` per shaded pixel. When
                     False the term is dropped at compile time. Disable for performance
                     when no emission is present.
    enable_per_light_ambient: When ambient lighting is enabled, sum each
                              light's `ambient` color into shaded pixels
                              even outside its cone or in shadow. When False
                              the per-light ambient pass is removed at compile
                              time. Disable for performance when model lights
                              do not use ambient colors.
    splat_position: Splat centers in world coordinates (nsplat, 3).
    splat_rotation: Splat rotations as (w, x, y, z) (nsplat, 4).
    splat_scale: Splat scales as standard deviation in each dimension (nsplat, 3).
    splat_rgba: Splat color and opacity (nsplat, 4).
    splat_adr: Offset of each splat in the splat attribute arrays,
               if None then all splats are in one group.
    splat_group_id: Splat id for each world (nworld,). If None then all worlds
                    use the first splat group.
    deterministic: Whether to use deterministic execution for flex BVH vertex
                   normal accumulation.

  Returns:
    The render context containing rendering fields and output arrays on device.
  """
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  constructor = "cubql"

  # Build grouped splat BVH.
  splat_attribute = (splat_position, splat_rotation, splat_scale, splat_rgba)
  if splat_position is None:
    if any(value is not None for value in (*splat_attribute[1:], splat_adr, splat_group_id)):
      raise ValueError("splat attributes, offsets, and group IDs must be supplied together")

    splat_position = wp.empty(0, dtype=wp.vec3)
    splat_rotation = wp.empty(0, dtype=wp.quat)
    splat_scale = wp.empty(0, dtype=wp.vec3)
    splat_rgba = wp.empty(0, dtype=wp.vec4)
    splat_bvh = None
    splat_bvh_id = wp.uint64(0)
    splat_lower = wp.empty(0, dtype=wp.vec3)
    splat_upper = wp.empty(0, dtype=wp.vec3)
    splat_group_root = wp.empty(nworld, dtype=int)
    splat_count = 0
  else:
    nsplat = splat_position.shape[0]
    if (
      splat_position.shape != (nsplat, 3)
      or splat_rotation.shape != (nsplat, 4)
      or splat_scale.shape != (nsplat, 3)
      or splat_rgba.shape != (nsplat, 4)
    ):
      raise ValueError("splat attributes must have shapes (nsplat, 3), (nsplat, 4), (nsplat, 3), and (nsplat, 4)")
    if splat_adr is not None and splat_adr.ndim != 1:
      raise ValueError("splat_adr must be one-dimensional")
    if splat_group_id is not None and splat_group_id.shape != (nworld,):
      raise ValueError("splat_group_id must be of shape (nworld,)")

    if splat_adr is None:
      splat_adr = np.array([0, splat_position.shape[0]], dtype=np.int32)
    if splat_group_id is None:
      splat_group_id = np.zeros(nworld, dtype=np.int32)
    (
      splat_position,
      splat_rotation,
      splat_scale,
      splat_rgba,
      splat_bvh,
      splat_bvh_id,
      splat_lower,
      splat_upper,
      splat_group_root,
      splat_count,
    ) = bvh.build_splat_bvh(*splat_attribute, splat_adr, splat_group_id, constructor="sah")

  # Mesh BVHs – build for all meshes so per-world variants are available
  nmesh = mjm.nmesh
  geom_enabled_mask = np.isin(mjm.geom_group, list(enabled_geom_groups))
  geom_enabled_idx = np.nonzero(geom_enabled_mask)[0]

  mesh_registry = {}
  mesh_bvh_id = [wp.uint64(0) for _ in range(nmesh)]
  mesh_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nmesh)]

  for mid in range(nmesh):
    mesh, half = bvh.build_mesh_bvh(mjm, mid, constructor=constructor)
    mesh_registry[mesh.id] = mesh
    mesh_bvh_id[mid] = mesh.id
    mesh_bounds_size[mid] = half

  mesh_bvh_id_arr = wp.array(mesh_bvh_id, dtype=wp.uint64)
  mesh_bounds_size_arr = wp.array(mesh_bounds_size, dtype=wp.vec3)

  # HField BVHs
  nhfield = mjm.nhfield
  hfield_geom_mask = geom_enabled_mask & (mjm.geom_type == types.GeomType.HFIELD) & (mjm.geom_dataid >= 0)
  used_hfield_id = set(mjm.geom_dataid[hfield_geom_mask].astype(int))
  hfield_registry = {}
  hfield_bvh_id = [wp.uint64(0) for _ in range(nhfield)]
  hfield_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nhfield)]

  for hid in used_hfield_id:
    hmesh, hhalf = bvh.build_hfield_bvh(mjm, hid, constructor=constructor)
    hfield_registry[hmesh.id] = hmesh
    hfield_bvh_id[hid] = hmesh.id
    hfield_bounds_size[hid] = hhalf

  hfield_bvh_id_arr = wp.array(hfield_bvh_id, dtype=wp.uint64)
  hfield_bounds_size_arr = wp.array(hfield_bounds_size, dtype=wp.vec3)

  # Flex BVHs
  nflex = mjm.nflex
  flex_registry = {}

  # Scene BVH flex primitives: 1D → one capsule per edge, 2D/3D → one box per flex
  flex_geom_flexid = []
  flex_geom_edgeid = []
  flex_bvh_id = np.full(nflex, 0, dtype=np.uint64)
  # Indexed later as [worldid, flexid].
  flex_group_root = np.full((nworld, nflex), -1, dtype=int)

  for f in range(nflex):
    if mjm.flex_dim[f] == 1:
      edge_adr = mjm.flex_edgeadr[f]
      flex_geom_flexid.extend([f] * mjm.flex_edgenum[f])
      flex_geom_edgeid.extend([edge_adr + e for e in range(mjm.flex_edgenum[f])])
    else:
      flex_geom_flexid.append(f)
      flex_geom_edgeid.append(-1)
      fmesh, group_root = bvh.build_flex_bvh(mjm, mjd, nworld, f, deterministic=bool(deterministic))
      flex_registry[f] = fmesh
      flex_bvh_id[f] = fmesh.id
      flex_group_root[:, f] = group_root.numpy()

  # Locate skybox texture
  skybox_tex_ids = np.nonzero(mjm.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)[0] if mjm.ntex else np.array([], dtype=int)
  has_skybox = render_skybox and skybox_tex_ids.size > 0
  if has_skybox:
    skybox_tex_id_np = np.array([skybox_tex_ids[0]], dtype=int)
  else:
    render_skybox = False
    skybox_tex_id_np = np.array([-1], dtype=int)

  textures_registry = []
  # Only materialize GPU textures when the caller actually needs them.
  if use_textures or has_skybox:
    empty_tex = None if use_textures else wp.Texture2D(width=1, height=1, num_channels=4, dtype=wp.float32)
    skybox_id = skybox_tex_ids[0] if has_skybox else -1
    for i in range(mjm.ntex):
      if use_textures or i == skybox_id:
        textures_registry.append(create_warp_texture(mjm, i))
      else:
        textures_registry.append(empty_tex)
  textures = wp.array(textures_registry, dtype=wp.Texture2D)
  # face width from the uploaded texture: tex_width is 0 for a by-reference skybox
  skybox_face_width_np = np.array([textures_registry[skybox_id].width if has_skybox else 1], dtype=int)

  # Filter active cameras
  if cam_active is not None:
    if len(cam_active) == 0:
      # Empty selection renders no cameras, and is the only valid mask when ncam == 0.
      active_cam_indices = []
    elif isinstance(cam_active[0], (bool, np.bool_)):
      assert len(cam_active) == mjm.ncam, f"cam_active must have length {mjm.ncam} (got {len(cam_active)})"
      active_cam_indices = [int(i) for i in np.nonzero(cam_active)[0]]
    elif isinstance(cam_active[0], str):
      active_cam_indices = []
      for name in cam_active:
        cid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid == -1:
          raise ValueError(f"Camera '{name}' not found in model.")
        active_cam_indices.append(cid)
    elif isinstance(cam_active[0], (int, np.integer)):
      active_cam_indices = [int(x) for x in cam_active]
    else:
      raise ValueError(f"Invalid cam_active format: {cam_active}")
  else:
    active_cam_indices = list(range(mjm.ncam))

  ncam = len(active_cam_indices)

  if cam_res is not None:
    if isinstance(cam_res, tuple):
      cam_res = [cam_res] * ncam
    elif isinstance(cam_res, list) and len(cam_res) == 1 and ncam > 1:
      cam_res = cam_res * ncam
    if len(cam_res) != ncam:
      raise ValueError(f"Camera resolutions count ({len(cam_res)}) does not match active camera count ({ncam}).")
    active_cam_res = cam_res
  else:
    active_cam_res = mjm.cam_resolution[active_cam_indices]

  cam_res_arr = wp.array(active_cam_res, dtype=wp.vec2i)

  if render_rgb is None:
    render_rgb = [bool(np.any(mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_RGB)) for i in active_cam_indices]
  elif isinstance(render_rgb, bool):
    render_rgb = [render_rgb] * ncam
  elif isinstance(render_rgb, (list, np.ndarray)):
    if len(render_rgb) == mjm.ncam and ncam != mjm.ncam:
      render_rgb = [bool(render_rgb[i]) for i in active_cam_indices]
    elif len(render_rgb) == 1 and ncam > 1:
      render_rgb = [bool(render_rgb[0])] * ncam
    else:
      render_rgb = [bool(x) for x in render_rgb]

  if render_depth is None:
    render_depth = [bool(np.any(mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_DEPTH)) for i in active_cam_indices]
  elif isinstance(render_depth, bool):
    render_depth = [render_depth] * ncam
  elif isinstance(render_depth, (list, np.ndarray)):
    if len(render_depth) == mjm.ncam and ncam != mjm.ncam:
      render_depth = [bool(render_depth[i]) for i in active_cam_indices]
    elif len(render_depth) == 1 and ncam > 1:
      render_depth = [bool(render_depth[0])] * ncam
    else:
      render_depth = [bool(x) for x in render_depth]

  if render_seg is None:
    render_seg = [bool(np.any(mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_SEG)) for i in active_cam_indices]
  elif isinstance(render_seg, bool):
    render_seg = [render_seg] * ncam
  elif isinstance(render_seg, (list, np.ndarray)):
    if len(render_seg) == mjm.ncam and ncam != mjm.ncam:
      render_seg = [bool(render_seg[i]) for i in active_cam_indices]
    elif len(render_seg) == 1 and ncam > 1:
      render_seg = [bool(render_seg[0])] * ncam
    else:
      render_seg = [bool(x) for x in render_seg]

  if len(render_rgb) != ncam:
    raise ValueError(f"render_rgb length ({len(render_rgb)}) does not match active camera count ({ncam}).")
  if len(render_depth) != ncam:
    raise ValueError(f"render_depth length ({len(render_depth)}) does not match active camera count ({ncam}).")
  if len(render_seg) != ncam:
    raise ValueError(f"render_seg length ({len(render_seg)}) does not match active camera count ({ncam}).")

  rgb_adr = -1 * np.ones(ncam, dtype=int)
  depth_adr = -1 * np.ones(ncam, dtype=int)
  seg_adr = -1 * np.ones(ncam, dtype=int)
  cam_res_np = cam_res_arr.numpy()
  ri = 0
  di = 0
  si = 0
  total = 0

  for idx in range(ncam):
    if render_rgb[idx]:
      rgb_adr[idx] = ri
      ri += cam_res_np[idx][0] * cam_res_np[idx][1]
    if render_depth[idx]:
      depth_adr[idx] = di
      di += cam_res_np[idx][0] * cam_res_np[idx][1]
    if render_seg[idx]:
      seg_adr[idx] = si
      si += cam_res_np[idx][0] * cam_res_np[idx][1]

    total += cam_res_np[idx][0] * cam_res_np[idx][1]

  znear = float(mjm.vis.map.znear * mjm.stat.extent)
  zfar = float(mjm.vis.map.zfar * mjm.stat.extent)

  if samples_per_pixel < 1:
    raise ValueError("samples_per_pixel must be at least 1.")
  if samples_per_pixel > 1 and not use_precomputed_rays:
    raise ValueError("samples_per_pixel > 1 requires use_precomputed_rays=True: dynamic rays carry no sub-pixel jitter.")
  if samples_per_pixel > 1 and ri == 0:
    raise ValueError("samples_per_pixel > 1 requires at least one camera with render_rgb=True.")
  nsamples = samples_per_pixel * samples_per_pixel
  ray = wp.zeros(int(total) * nsamples, dtype=wp.vec3)
  ray_offset = wp.zeros(int(total) * nsamples, dtype=wp.vec3)

  cam_projection = mjm.cam_projection

  for sample in range(nsamples):
    offset = sample * int(total)
    for idx, cam_id_val in enumerate(active_cam_indices):
      cam_id = int(cam_id_val)
      img_w = int(cam_res_np[idx][0])
      img_h = int(cam_res_np[idx][1])
      wp.launch(
        kernel=_build_rays,
        dim=(img_w, img_h),
        inputs=[
          offset,
          img_w,
          img_h,
          int(mjm.cam_projection[cam_id]),
          float(mjm.cam_fovy[cam_id]),
          wp.vec2(float(mjm.cam_sensorsize[cam_id, 0]), float(mjm.cam_sensorsize[cam_id, 1])),
          wp.vec4(
            float(mjm.cam_intrinsic[cam_id, 0]),
            float(mjm.cam_intrinsic[cam_id, 1]),
            float(mjm.cam_intrinsic[cam_id, 2]),
            float(mjm.cam_intrinsic[cam_id, 3]),
          ),
          znear,
          samples_per_pixel,
          sample % samples_per_pixel,
          sample // samples_per_pixel,
        ],
        outputs=[ray, ray_offset],
      )
      offset += img_w * img_h

  aa_accum = wp.zeros((nworld, ri if nsamples > 1 else 1), dtype=wp.vec3)

  bvh_ngeom = len(geom_enabled_idx)

  # Geom types present among enabled geoms, plus FLEX when flex primitives exist.
  # Used to statically eliminate unused intersection branches in the ray-cast kernels.
  geom_ray_types = set(int(t) for t in mjm.geom_type[geom_enabled_idx])
  if len(flex_geom_flexid) > 0:
    geom_ray_types.add(int(types.GeomType.FLEX))
  geom_ray_types = tuple(sorted(geom_ray_types))
  if mjm.nlight == 0:
    light_attenuation_is_default = True
    has_spot_lights = False
  else:
    atten = np.asarray(mjm.light_attenuation, dtype=np.float32).reshape(-1, 3)
    light_attenuation_is_default = bool(np.allclose(atten, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    has_spot_lights = bool((np.asarray(mjm.light_type) == int(mujoco.mjtLightType.mjLIGHT_SPOT)).any())

  has_orthographic_camera = any(
    int(mjm.cam_projection[cam_id]) == int(ProjectionType.ORTHOGRAPHIC) for cam_id in active_cam_indices
  )

  rc = RenderContext(
    nrender=ncam,
    cam_res=cam_res_arr,
    cam_id_map=wp.array(active_cam_indices, dtype=int),
    use_textures=use_textures,
    use_fast_math=use_fast_math,
    use_shadows=use_shadows,
    use_ambient_lighting=use_ambient_lighting,
    background_color=pack_rgba_to_uint32(
      background_color[0] * 255.0, background_color[1] * 255.0, background_color[2] * 255.0, background_color[3] * 255.0
    ),
    use_precomputed_rays=use_precomputed_rays,
    render_skybox=render_skybox,
    skybox_tex_id=wp.array(skybox_tex_id_np, dtype=int),
    skybox_face_width=wp.array(skybox_face_width_np, dtype=int),
    headlight_active=bool(mjm.vis.headlight.active),
    headlight_ambient=wp.vec3(mjm.vis.headlight.ambient),
    headlight_diffuse=wp.vec3(mjm.vis.headlight.diffuse),
    headlight_specular=wp.vec3(mjm.vis.headlight.specular),
    bvh_ngeom=bvh_ngeom,
    enabled_geom_ids=wp.array(geom_enabled_idx, dtype=int),
    mesh_registry=mesh_registry,
    mesh_bvh_id=mesh_bvh_id_arr,
    mesh_bounds_size=mesh_bounds_size_arr,
    mesh_texcoord=wp.array(mjm.mesh_texcoord, dtype=wp.vec2),
    mesh_texcoord_offsets=wp.array(mjm.mesh_texcoordadr, dtype=int),
    mesh_facetexcoord=wp.array(mjm.mesh_facetexcoord, dtype=wp.vec3i),
    mesh_facenormal=wp.array(mjm.mesh_facenormal, dtype=wp.vec3i),
    textures=textures,
    textures_registry=textures_registry,
    hfield_registry=hfield_registry,
    hfield_bvh_id=hfield_bvh_id_arr,
    hfield_bounds_size=hfield_bounds_size_arr,
    flex_mesh_registry=flex_registry,
    flex_rgba=wp.array(mjm.flex_rgba, dtype=wp.vec4),
    flex_bvh_id=wp.array(flex_bvh_id, dtype=wp.uint64),
    flex_group_root=wp.array(flex_group_root, dtype=int),
    flex_render_smooth=flex_render_smooth,
    bvh_nflexgeom=len(flex_geom_flexid),
    flex_dim_np=mjm.flex_dim,
    flex_geom_flexid=wp.array(flex_geom_flexid, dtype=int),
    flex_geom_edgeid=wp.array(flex_geom_edgeid, dtype=int),
    bvh=None,
    bvh_id=None,
    lower=wp.zeros(nworld * (bvh_ngeom + len(flex_geom_flexid)), dtype=wp.vec3),
    upper=wp.zeros(nworld * (bvh_ngeom + len(flex_geom_flexid)), dtype=wp.vec3),
    group=wp.zeros(nworld * (bvh_ngeom + len(flex_geom_flexid)), dtype=int),
    group_root=wp.zeros(nworld, dtype=int),
    ray=ray,
    ray_offset=ray_offset,
    rgb_data=wp.zeros((nworld, ri), dtype=wp.uint32),
    rgb_adr=wp.array(rgb_adr, dtype=int),
    depth_data=wp.zeros((nworld, di), dtype=wp.float32),
    depth_adr=wp.array(depth_adr, dtype=int),
    render_rgb=wp.array(render_rgb, dtype=bool),
    render_depth=wp.array(render_depth, dtype=bool),
    seg_data=wp.zeros((nworld, max(si, 1)), dtype=wp.vec2i),
    seg_adr=wp.array(seg_adr, dtype=int),
    render_seg=wp.array(render_seg, dtype=bool),
    znear=znear,
    zfar=zfar,
    total_rays=int(total),
    enable_backface_culling=enable_backface_culling,
    shadow_light_fraction=shadow_light_fraction,
    samples_per_pixel=samples_per_pixel,
    aa_accum=aa_accum,
    geom_ray_types=geom_ray_types,
    enable_vertex_normals=enable_vertex_normals,
    enable_specular=enable_specular,
    enable_emission=enable_emission,
    enable_per_light_ambient=enable_per_light_ambient,
    light_attenuation_is_default=light_attenuation_is_default,
    has_spot_lights=has_spot_lights,
    has_orthographic_camera=has_orthographic_camera,
    splat_position=splat_position,
    splat_rotation=splat_rotation,
    splat_scale=splat_scale,
    splat_rgba=splat_rgba,
    splat_bvh=splat_bvh,
    splat_bvh_id=splat_bvh_id,
    splat_lower=splat_lower,
    splat_upper=splat_upper,
    splat_group_root=splat_group_root,
    splat_count=splat_count,
  )

  bvh.build_scene_bvh(mjm, mjd, rc, nworld)

  warp_util.mark_batched(rc)
  return rc
