# Copyright 2025 The Newton Developers
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

from typing import Any, Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import history
from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import ray
from mujoco.mjx.third_party.mujoco_warp._src import smooth
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src.collision_sdf import get_sdf_params
from mujoco.mjx.third_party.mujoco_warp._src.collision_sdf import sdf
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import ConDataField
from mujoco.mjx.third_party.mujoco_warp._src.types import ConeType
from mujoco.mjx.third_party.mujoco_warp._src.types import ConstraintType
from mujoco.mjx.third_party.mujoco_warp._src.types import ContactType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DataType
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import JointType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.third_party.mujoco_warp._src.types import ProjectionType
from mujoco.mjx.third_party.mujoco_warp._src.types import RayDataField
from mujoco.mjx.third_party.mujoco_warp._src.types import SensorType
from mujoco.mjx.third_party.mujoco_warp._src.types import Stage
from mujoco.mjx.third_party.mujoco_warp._src.types import TrnType
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.types import vec6
from mujoco.mjx.third_party.mujoco_warp._src.types import vec8
from mujoco.mjx.third_party.mujoco_warp._src.types import vec8i
from mujoco.mjx.third_party.mujoco_warp._src.types import vec_pluginattr
from mujoco.mjx.third_party.mujoco_warp._src.util_misc import inside_geom
from mujoco.mjx.third_party.mujoco_warp._src.util_misc import poly_potential
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def _write_scalar(
  # Model:
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  # In:
  sensorid: int,
  sensor: Any,
  # Out:
  out: wp.array[float],
):
  adr = sensor_adr[sensorid]
  cutoff = sensor_cutoff[sensorid]

  if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):
    datatype = sensor_datatype[sensorid]
    if datatype == DataType.REAL:
      out[adr] = wp.clamp(sensor, -cutoff, cutoff)
      return
    elif datatype == DataType.POSITIVE:
      out[adr] = wp.min(sensor, cutoff)
      return

  out[adr] = sensor


@wp.func
def _write_vector(
  # Model:
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  # In:
  sensorid: int,
  sensordim: int,
  sensor: Any,
  # Out:
  out: wp.array[float],
):
  adr = sensor_adr[sensorid]
  cutoff = sensor_cutoff[sensorid]

  if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):
    datatype = sensor_datatype[sensorid]
    if datatype == DataType.REAL:
      for i in range(sensordim):
        out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)
      return
    elif datatype == DataType.POSITIVE:
      for i in range(sensordim):
        out[adr + i] = wp.min(sensor[i], cutoff)
      return

  for i in range(sensordim):
    out[adr + i] = sensor[i]


@wp.func
def _clamp_real(val: float, cutoff: float) -> float:
  if cutoff > 0.0:
    return wp.clamp(val, -cutoff, cutoff)
  return val


@wp.func
def _magnetometer(
  # Model:
  opt_magnetic: wp.array[wp.vec3],
  # Data in:
  site_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  magnetic = opt_magnetic[worldid % opt_magnetic.shape[0]]
  return wp.transpose(site_xmat_in[worldid, objid]) @ magnetic


@wp.func
def _cam_projection(
  # Model:
  cam_fovy: wp.array2d[float],
  cam_resolution: wp.array[wp.vec2i],
  cam_sensorsize: wp.array[wp.vec2],
  cam_intrinsic: wp.array2d[wp.vec4],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  objid: int,
  refid: int,
) -> wp.vec2:
  sensorsize = cam_sensorsize[refid]
  intrinsic = cam_intrinsic[worldid % cam_intrinsic.shape[0], refid]
  fovy = cam_fovy[worldid % cam_fovy.shape[0], refid]
  res = cam_resolution[refid]

  target_xpos = site_xpos_in[worldid, objid]
  xpos = cam_xpos_in[worldid, refid]
  xmat = cam_xmat_in[worldid, refid]

  # Transform target position into camera-local frame: v = xmat^T @ (target - cam_pos)
  v = wp.transpose(xmat) @ (target_xpos - xpos)

  # Compute focal lengths
  if sensorsize[0] != 0.0 and sensorsize[1] != 0.0:
    fx = intrinsic[0] / (sensorsize[0] + MJ_MINVAL) * float(res[0])
    fy = intrinsic[1] / (sensorsize[1] + MJ_MINVAL) * float(res[1])
  else:
    f = 0.5 / wp.tan(fovy * wp.static(wp.pi / 360.0)) * float(res[1])
    fx = f
    fy = f

  # Compute homogeneous coordinates
  pixel_x = -fx * v[0]
  pixel_y = fy * v[1]

  denom = v[2]
  if wp.abs(denom) < MJ_MINVAL:
    if denom < 0.0:
      denom = -MJ_MINVAL
    else:
      denom = MJ_MINVAL

  return wp.vec2(pixel_x, pixel_y) / denom + 0.5 * wp.vec2(float(res[0]), float(res[1]))


@wp.kernel
def _sensor_rangefinder_init(
  # Model:
  cam_projection: wp.array[int],
  cam_fovy: wp.array2d[float],
  cam_resolution: wp.array[wp.vec2i],
  cam_sensorsize: wp.array[wp.vec2],
  cam_intrinsic: wp.array2d[wp.vec4],
  sensor_objtype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_rangefinder_adr: wp.array[int],
  rangefinder_sensor_adr: wp.array[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  # Out:
  pnt_out: wp.array2d[wp.vec3],
  vec_out: wp.array2d[wp.vec3],
):
  worldid, rfid = wp.tid()
  sensorid = sensor_rangefinder_adr[rfid]
  objtype = sensor_objtype[sensorid]
  objid = sensor_objid[sensorid]

  if objtype == ObjType.SITE:
    # site-attached rangefinder: single ray from site origin along +z
    pnt_out[worldid, rfid] = site_xpos_in[worldid, objid]
    vec_out[worldid, rfid] = site_xmat_in[worldid, objid][:, 2]
  else:
    # camera-attached rangefinder: row-major pixel index within sensor
    res = cam_resolution[objid]
    width = float(res[0])
    height = float(res[1])
    pixelid = rfid - rangefinder_sensor_adr[sensorid]
    row = pixelid // res[0]
    col = pixelid - row * res[0]

    cam_xpos = cam_xpos_in[worldid, objid]
    cam_xmat = cam_xmat_in[worldid, objid]
    proj = cam_projection[objid]

    # compute camera intrinsics (focal lengths and principal point in pixels)
    fx = 0.5 * width
    fy = 0.5 * height
    cx = fx
    cy = fy

    if proj == ProjectionType.PERSPECTIVE:
      sensorsize = cam_sensorsize[objid]
      if sensorsize[0] != 0.0 and sensorsize[1] != 0.0:
        intrinsic = cam_intrinsic[worldid % cam_intrinsic.shape[0], objid]
        inv_sx = math.safe_div(width, sensorsize[0])
        inv_sy = math.safe_div(height, sensorsize[1])
        fx = intrinsic[0] * inv_sx
        fy = intrinsic[1] * inv_sy
        cx = intrinsic[2] * inv_sx
        cy = intrinsic[3] * inv_sy
      else:
        fovy = cam_fovy[worldid % cam_fovy.shape[0], objid]
        fx = math.safe_div(fy, wp.tan(fovy * wp.static(wp.pi / 360.0)))
        fy = fx

    # normalized pixel ray offsets in camera frame (+x right, +y up, -z forward)
    dx = math.safe_div(float(col) + 0.5 - cx, fx)
    dy = -math.safe_div(float(row) + 0.5 - cy, fy)

    # perspective rays fan from camera origin; orthographic rays shift origin across image plane
    if proj == ProjectionType.PERSPECTIVE:
      pnt_out[worldid, rfid] = cam_xpos
      vec_out[worldid, rfid] = wp.normalize(cam_xmat @ wp.vec3(dx, dy, -1.0))
    else:
      half_extent = 0.5 * cam_fovy[worldid % cam_fovy.shape[0], objid]
      pnt_out[worldid, rfid] = cam_xpos + cam_xmat @ wp.vec3(dx * half_extent, dy * half_extent, 0.0)
      vec_out[worldid, rfid] = -cam_xmat[:, 2]


@wp.kernel
def _sensor_rangefinder(
  # Model:
  sensor_objtype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_intprm: wp.array2d[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_rangefinder_adr: wp.array[int],
  rangefinder_sensor_adr: wp.array[int],
  # Data in:
  cam_xmat_in: wp.array2d[wp.mat33],
  # In:
  rangefinder_pnt_in: wp.array2d[wp.vec3],
  rangefinder_vec_in: wp.array2d[wp.vec3],
  rangefinder_dist_in: wp.array2d[float],
  rangefinder_normal_in: wp.array2d[wp.vec3],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, rfid = wp.tid()
  sensorid = sensor_rangefinder_adr[rfid]
  dataspec = sensor_intprm[sensorid, 0]
  pixelid = rfid - rangefinder_sensor_adr[sensorid]

  # unpack requested ray data fields from bitmask
  has_dist = bool(dataspec & (1 << RayDataField.DIST))
  has_dir = bool(dataspec & (1 << RayDataField.DIR))
  has_origin = bool(dataspec & (1 << RayDataField.ORIGIN))
  has_point = bool(dataspec & (1 << RayDataField.POINT))
  has_normal = bool(dataspec & (1 << RayDataField.NORMAL))
  has_depth = bool(dataspec & (1 << RayDataField.DEPTH))

  # compute per-ray output stride and starting sensordata address
  stride = (
    (1 if has_dist else 0)
    + (3 if has_dir else 0)
    + (3 if has_origin else 0)
    + (3 if has_point else 0)
    + (3 if has_normal else 0)
    + (1 if has_depth else 0)
  )
  adr = sensor_adr[sensorid] + pixelid * stride
  ptr = adr
  out = sensordata_out[worldid]

  dist = rangefinder_dist_in[worldid, rfid]
  hit = dist >= 0.0
  is_camera = sensor_objtype[sensorid] == ObjType.CAMERA
  need_dir = has_dir or has_point or (has_depth and is_camera)

  # load ray direction and origin only when required by requested fields
  direction = rangefinder_vec_in[worldid, rfid] if (hit and need_dir) else wp.vec3(0.0)
  origin = rangefinder_pnt_in[worldid, rfid] if (has_origin or (hit and has_point)) else wp.vec3(0.0)
  point = origin + direction * dist if (hit and has_point) else wp.vec3(0.0)

  cutoff = sensor_cutoff[sensorid]

  # write requested fields in ascending mjtRayDataField bit order, clamped to [-cutoff, cutoff]
  if has_dist:
    out[ptr] = _clamp_real(dist, cutoff)
    ptr += 1

  if has_dir:
    out[ptr + 0] = _clamp_real(direction[0], cutoff)
    out[ptr + 1] = _clamp_real(direction[1], cutoff)
    out[ptr + 2] = _clamp_real(direction[2], cutoff)
    ptr += 3

  if has_origin:
    out[ptr + 0] = _clamp_real(origin[0], cutoff)
    out[ptr + 1] = _clamp_real(origin[1], cutoff)
    out[ptr + 2] = _clamp_real(origin[2], cutoff)
    ptr += 3

  if has_point:
    out[ptr + 0] = _clamp_real(point[0], cutoff)
    out[ptr + 1] = _clamp_real(point[1], cutoff)
    out[ptr + 2] = _clamp_real(point[2], cutoff)
    ptr += 3

  if has_normal:
    normal = rangefinder_normal_in[worldid, rfid] if hit else wp.vec3(0.0)
    out[ptr + 0] = _clamp_real(normal[0], cutoff)
    out[ptr + 1] = _clamp_real(normal[1], cutoff)
    out[ptr + 2] = _clamp_real(normal[2], cutoff)
    ptr += 3

  if has_depth:
    # depth is z-distance along camera optical axis (-z), or ray distance for sites
    if not hit:
      depth = -1.0
    elif is_camera:
      objid = sensor_objid[sensorid]
      depth = -dist * wp.dot(direction, cam_xmat_in[worldid, objid][:, 2])
    else:
      depth = dist
    out[ptr] = _clamp_real(depth, cutoff)


@wp.func
def _joint_pos(jnt_qposadr: wp.array[int], qpos_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return qpos_in[worldid, jnt_qposadr[objid]]


@wp.func
def _tendon_pos(ten_length_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return ten_length_in[worldid, objid]


@wp.func
def _actuator_pos(actuator_length_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return actuator_length_in[worldid, objid]


@wp.func
def _ball_quat(jnt_qposadr: wp.array[int], qpos_in: wp.array2d[float], worldid: int, objid: int) -> wp.quat:
  adr = jnt_qposadr[objid]
  quat = wp.quat(
    qpos_in[worldid, adr + 0],
    qpos_in[worldid, adr + 1],
    qpos_in[worldid, adr + 2],
    qpos_in[worldid, adr + 3],
  )
  return wp.normalize(quat)


@wp.kernel
def _limit_pos(
  # Model:
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_limitpos_adr: wp.array[int],
  # Data in:
  ne_in: wp.array[int],
  nf_in: wp.array[int],
  nl_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_pos_in: wp.array2d[float],
  efc_margin_in: wp.array2d[float],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, efcid, limitposid = wp.tid()

  ne = ne_in[worldid]
  nf = nf_in[worldid]
  nl = nl_in[worldid]

  # skip if not limit
  if efcid < ne + nf or efcid >= ne + nf + nl:
    return

  sensorid = sensor_limitpos_adr[limitposid]
  if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:
    efc_type = efc_type_in[worldid, efcid]
    if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:
      val = efc_pos_in[worldid, efcid] - efc_margin_in[worldid, efcid]
      _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, sensordata_out[worldid])


@wp.func
def _get_pos(
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  site_xpos_in: wp.array2d[wp.vec3],
  cam_xpos_in: wp.array2d[wp.vec3],
  # In:
  worldid: int,
  objtype: int,
  objid: int,
) -> wp.vec3:
  if objtype == ObjType.BODY:
    return xipos_in[worldid, objid]
  elif objtype == ObjType.XBODY:
    return xpos_in[worldid, objid]
  elif objtype == ObjType.GEOM:
    return geom_xpos_in[worldid, objid]
  elif objtype == ObjType.SITE:
    return site_xpos_in[worldid, objid]
  elif objtype == ObjType.CAMERA:
    return cam_xpos_in[worldid, objid]
  else:
    return wp.vec3(0.0)


@wp.func
def _get_mat(
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  ximat_in: wp.array2d[wp.mat33],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  objtype: int,
  objid: int,
) -> wp.mat33:
  if objtype == ObjType.BODY:
    return ximat_in[worldid, objid]
  elif objtype == ObjType.XBODY:
    return xmat_in[worldid, objid]
  elif objtype == ObjType.GEOM:
    return geom_xmat_in[worldid, objid]
  elif objtype == ObjType.SITE:
    return site_xmat_in[worldid, objid]
  elif objtype == ObjType.CAMERA:
    return cam_xmat_in[worldid, objid]
  else:
    return wp.identity(3, dtype=wp.float32)


@wp.func
def _get_body_id(
  # Model:
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  # In:
  objtype: int,
  objid: int,
) -> int:
  if objtype == ObjType.BODY or objtype == ObjType.XBODY:
    return objid
  elif objtype == ObjType.GEOM:
    return geom_bodyid[objid]
  elif objtype == ObjType.SITE:
    return site_bodyid[objid]
  elif objtype == ObjType.CAMERA:
    return cam_bodyid[objid]
  else:
    return 0


@wp.func
def _get_quat(
  # Model:
  body_iquat: wp.array2d[wp.quat],
  geom_bodyid: wp.array[int],
  geom_quat: wp.array2d[wp.quat],
  site_bodyid: wp.array[int],
  site_quat: wp.array2d[wp.quat],
  cam_bodyid: wp.array[int],
  cam_quat: wp.array2d[wp.quat],
  # Data in:
  xquat_in: wp.array2d[wp.quat],
  # In:
  worldid: int,
  objtype: int,
  objid: int,
) -> wp.quat:
  if objtype == ObjType.BODY:
    body_iquat_id = worldid % body_iquat.shape[0]
    return math.mul_quat(xquat_in[worldid, objid], body_iquat[body_iquat_id, objid])
  elif objtype == ObjType.XBODY:
    return xquat_in[worldid, objid]
  elif objtype == ObjType.GEOM:
    geom_quat_id = worldid % geom_quat.shape[0]
    return math.mul_quat(xquat_in[worldid, geom_bodyid[objid]], geom_quat[geom_quat_id, objid])
  elif objtype == ObjType.SITE:
    site_quat_id = worldid % site_quat.shape[0]
    return math.mul_quat(xquat_in[worldid, site_bodyid[objid]], site_quat[site_quat_id, objid])
  elif objtype == ObjType.CAMERA:
    cam_quat_id = worldid % cam_quat.shape[0]
    return math.mul_quat(xquat_in[worldid, cam_bodyid[objid]], cam_quat[cam_quat_id, objid])
  else:
    return wp.quat(1.0, 0.0, 0.0, 0.0)


@wp.func
def _frame_pos(
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.vec3:
  xpos = _get_pos(xpos_in, xipos_in, geom_xpos_in, site_xpos_in, cam_xpos_in, worldid, objtype, objid)
  if refid == -1:
    return xpos

  xpos_ref = _get_pos(xpos_in, xipos_in, geom_xpos_in, site_xpos_in, cam_xpos_in, worldid, reftype, refid)
  xmat_ref = _get_mat(xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, reftype, refid)
  return wp.transpose(xmat_ref) @ (xpos - xpos_ref)


@wp.func
def _frame_axis(
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  ximat_in: wp.array2d[wp.mat33],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
  frame_axis: int,
) -> wp.vec3:
  xmat = _get_mat(xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, objtype, objid)
  axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])

  if refid == -1:
    return axis

  xmat_ref = _get_mat(xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, reftype, refid)
  return wp.transpose(xmat_ref) @ axis


@wp.func
def _frame_quat(
  # Model:
  body_iquat: wp.array2d[wp.quat],
  geom_bodyid: wp.array[int],
  geom_quat: wp.array2d[wp.quat],
  site_bodyid: wp.array[int],
  site_quat: wp.array2d[wp.quat],
  cam_bodyid: wp.array[int],
  cam_quat: wp.array2d[wp.quat],
  # Data in:
  xquat_in: wp.array2d[wp.quat],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.quat:
  quat = _get_quat(
    body_iquat,
    geom_bodyid,
    geom_quat,
    site_bodyid,
    site_quat,
    cam_bodyid,
    cam_quat,
    xquat_in,
    worldid,
    objtype,
    objid,
  )

  if refid == -1:
    return quat

  refquat = _get_quat(
    body_iquat,
    geom_bodyid,
    geom_quat,
    site_bodyid,
    site_quat,
    cam_bodyid,
    cam_quat,
    xquat_in,
    worldid,
    reftype,
    refid,
  )

  return math.mul_quat(math.quat_inv(refquat), quat)


@wp.func
def _subtree_com(subtree_com_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:
  return subtree_com_in[worldid, objid]


@wp.func
def _clock(time_in: wp.array[float], worldid: int) -> float:
  return time_in[worldid]


@wp.kernel
def _sensor_pos(
  # Model:
  ngeom: int,
  opt_magnetic: wp.array[wp.vec3],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  body_iquat: wp.array2d[wp.quat],
  body_mass: wp.array2d[float],
  body_subtreemass: wp.array2d[float],
  jnt_qposadr: wp.array[int],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_quat: wp.array2d[wp.quat],
  site_type: wp.array[int],
  site_bodyid: wp.array[int],
  site_size: wp.array[wp.vec3],
  site_quat: wp.array2d[wp.quat],
  cam_bodyid: wp.array[int],
  cam_quat: wp.array2d[wp.quat],
  cam_fovy: wp.array2d[float],
  cam_resolution: wp.array[wp.vec2i],
  cam_sensorsize: wp.array[wp.vec2],
  cam_intrinsic: wp.array2d[wp.vec4],
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_objtype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_reftype: wp.array[int],
  sensor_refid: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  nxn_pairid: wp.array[wp.vec2i],
  sensor_pos_adr: wp.array[int],
  # Data in:
  time_in: wp.array[float],
  energy_in: wp.array[wp.vec2],
  qpos_in: wp.array2d[float],
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  xmat_in: wp.array2d[wp.mat33],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  ten_length_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  # In:
  sensor_collision_in: wp.array4d[float],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, posid = wp.tid()
  sensorid = sensor_pos_adr[posid]
  sensortype = sensor_type[sensorid]
  objid = sensor_objid[sensorid]
  out = sensordata_out[worldid]

  if sensortype == SensorType.MAGNETOMETER:
    vec3 = _magnetometer(opt_magnetic, site_xmat_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.CAMPROJECTION:
    refid = sensor_refid[sensorid]
    vec2 = _cam_projection(
      cam_fovy, cam_resolution, cam_sensorsize, cam_intrinsic, site_xpos_in, cam_xpos_in, cam_xmat_in, worldid, objid, refid
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 2, vec2, out)
  # rangefinder sensors are written by _sensor_rangefinder
  elif sensortype == SensorType.JOINTPOS:
    val = _joint_pos(jnt_qposadr, qpos_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.TENDONPOS:
    val = _tendon_pos(ten_length_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.ACTUATORPOS:
    val = _actuator_pos(actuator_length_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.BALLQUAT:
    quat = _ball_quat(jnt_qposadr, qpos_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)
  elif sensortype == SensorType.FRAMEPOS:
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    vec3 = _frame_pos(
      xpos_in,
      xmat_in,
      xipos_in,
      ximat_in,
      geom_xpos_in,
      geom_xmat_in,
      site_xpos_in,
      site_xmat_in,
      cam_xpos_in,
      cam_xmat_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.FRAMEXAXIS or sensortype == SensorType.FRAMEYAXIS or sensortype == SensorType.FRAMEZAXIS:
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    if sensortype == SensorType.FRAMEXAXIS:
      axis = 0
    elif sensortype == SensorType.FRAMEYAXIS:
      axis = 1
    elif sensortype == SensorType.FRAMEZAXIS:
      axis = 2
    vec3 = _frame_axis(
      xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, objid, objtype, refid, reftype, axis
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.FRAMEQUAT:
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    quat = _frame_quat(
      body_iquat,
      geom_bodyid,
      geom_quat,
      site_bodyid,
      site_quat,
      cam_bodyid,
      cam_quat,
      xquat_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)
  elif sensortype == SensorType.SUBTREECOM:
    vec3 = _subtree_com(subtree_com_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.GEOMDIST or sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:
    objtype = sensor_objtype[sensorid]
    objid = sensor_objid[sensorid]
    reftype = sensor_reftype[sensorid]
    refid = sensor_refid[sensorid]

    # initialize
    dist = float(sensor_cutoff[sensorid])
    pos = wp.vec3()
    normal = wp.vec3()
    flip = bool(False)

    # check for flip direction
    if objtype == int(ObjType.BODY.value):
      n1 = body_geomnum[objid]
      id1 = body_geomadr[objid]
    else:
      n1 = 1
      id1 = objid
    if reftype == int(ObjType.BODY.value):
      n2 = body_geomnum[refid]
      id2 = body_geomadr[refid]
    else:
      n2 = 1
      id2 = refid

    for geom1 in range(n1):
      geomid1 = id1 + geom1
      for geom2 in range(n2):
        geomid2 = id2 + geom2

        if geomid1 <= geomid2:
          pairid = math.upper_tri_index(ngeom, geomid1, geomid2)
        else:
          pairid = math.upper_tri_index(ngeom, geomid2, geomid1)
        collisionid = nxn_pairid[pairid][1]

        for i in range(8):
          dist_new = sensor_collision_in[worldid, collisionid, i, 0]

          if dist_new < dist:
            dist = dist_new

            if sensortype == SensorType.GEOMFROMTO:
              pos = wp.vec3(
                sensor_collision_in[worldid, collisionid, i, 1],
                sensor_collision_in[worldid, collisionid, i, 2],
                sensor_collision_in[worldid, collisionid, i, 3],
              )
            if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:
              normal = wp.vec3(
                sensor_collision_in[worldid, collisionid, i, 4],
                sensor_collision_in[worldid, collisionid, i, 5],
                sensor_collision_in[worldid, collisionid, i, 6],
              )

            if geom_type[geomid1] > geom_type[geomid2]:
              flip = True
            elif geom_type[geomid1] == geom_type[geomid2]:
              flip = geomid1 > geomid2
            else:
              flip = False
    # the segment from geom 1 to geom 2, which points along the contact normal or against it
    if dist >= sensor_cutoff[sensorid] or dist == 0.0:
      segment = wp.vec3()
    else:
      segment = wp.where(flip != (dist < 0.0), -normal, normal)

    if sensortype == int(SensorType.GEOMDIST.value):
      _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, dist, out)
    elif sensortype == int(SensorType.GEOMNORMAL.value):
      _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, segment, out)
    elif sensortype == int(SensorType.GEOMFROMTO.value):
      if dist < sensor_cutoff[sensorid]:
        half = 0.5 * wp.abs(dist) * segment
        fromto = vec6(
          pos[0] - half[0], pos[1] - half[1], pos[2] - half[2], pos[0] + half[0], pos[1] + half[1], pos[2] + half[2]
        )
      else:
        fromto = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 6, fromto, out)
  elif sensortype == SensorType.INSIDESITE:
    objtype = sensor_objtype[sensorid]
    if objtype == ObjType.XBODY:
      xpos = xpos_in[worldid, objid]
    elif objtype == ObjType.BODY:
      # for massless bodies with positive subtree mass (e.g., flex parents),
      # xipos is the static body frame origin; use subtree_com instead
      if objid > 0:
        if (
          body_mass[worldid % body_mass.shape[0], objid] < MJ_MINVAL
          and body_subtreemass[worldid % body_subtreemass.shape[0], objid] >= MJ_MINVAL
        ):
          xpos = subtree_com_in[worldid, objid]
        else:
          xpos = xipos_in[worldid, objid]
      else:
        xpos = xipos_in[worldid, objid]
    elif objtype == ObjType.GEOM:
      xpos = geom_xpos_in[worldid, objid]
    elif objtype == ObjType.SITE:
      xpos = site_xpos_in[worldid, objid]
    elif objtype == ObjType.CAMERA:
      xpos = cam_xpos_in[worldid, objid]
    else:
      return  # should not occur
    refid = sensor_refid[sensorid]
    val_bool = inside_geom(site_xpos_in[worldid, refid], site_xmat_in[worldid, refid], site_size[refid], site_type[refid], xpos)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, float(val_bool), out)
  elif sensortype == SensorType.E_POTENTIAL:
    val = energy_in[worldid][0]
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.E_KINETIC:
    val = energy_in[worldid][1]
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.CLOCK:
    val = _clock(time_in, worldid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)


@wp.kernel
def _sensor_collision(
  # Model:
  ngeom: int,
  nxn_pairid: wp.array[wp.vec2i],
  # Data in:
  contact_dist_in: wp.array[float],
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  contact_geomcollisionid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  sensor_collision_out: wp.array4d[float],
):
  conid = wp.tid()

  if conid >= nacon_in[0]:
    return

  if not (contact_type_in[conid] & ContactType.SENSOR):
    return

  geom = contact_geom_in[conid]
  if geom[0] <= geom[1]:
    pairid = math.upper_tri_index(ngeom, geom[0], geom[1])
  else:
    pairid = math.upper_tri_index(ngeom, geom[1], geom[0])

  worldid = contact_worldid_in[conid]
  collisionid = nxn_pairid[pairid][1]
  geomcollisionid = contact_geomcollisionid_in[conid]

  dist = contact_dist_in[conid]
  pos = contact_pos_in[conid]
  frame = contact_frame_in[conid]

  sensor_collision_out[worldid, collisionid, geomcollisionid, 0] = dist
  sensor_collision_out[worldid, collisionid, geomcollisionid, 1] = pos[0]
  sensor_collision_out[worldid, collisionid, geomcollisionid, 2] = pos[1]
  sensor_collision_out[worldid, collisionid, geomcollisionid, 3] = pos[2]
  sensor_collision_out[worldid, collisionid, geomcollisionid, 4] = frame[0, 0]
  sensor_collision_out[worldid, collisionid, geomcollisionid, 5] = frame[0, 1]
  sensor_collision_out[worldid, collisionid, geomcollisionid, 6] = frame[0, 2]


@event_scope
def sensor_pos(m: Model, d: Data):
  """Compute position-dependent sensor values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return

  # rangefinder
  if m.nrangefinder > 0:
    rangefinder_pnt = wp.empty((d.nworld, m.nrangefinder), dtype=wp.vec3)
    rangefinder_vec = wp.empty((d.nworld, m.nrangefinder), dtype=wp.vec3)
    rangefinder_dist = wp.empty((d.nworld, m.nrangefinder), dtype=float)
    rangefinder_geomid = wp.empty((d.nworld, m.nrangefinder), dtype=int)
    rangefinder_normal = wp.empty((d.nworld, m.nrangefinder), dtype=wp.vec3)

    # get position and direction
    wp.launch(
      _sensor_rangefinder_init,
      dim=(d.nworld, m.nrangefinder),
      inputs=[
        m.cam_projection,
        m.cam_fovy,
        m.cam_resolution,
        m.cam_sensorsize,
        m.cam_intrinsic,
        m.sensor_objtype,
        m.sensor_objid,
        m.sensor_rangefinder_adr,
        m.rangefinder_sensor_adr,
        d.site_xpos,
        d.site_xmat,
        d.cam_xpos,
        d.cam_xmat,
      ],
      outputs=[rangefinder_pnt, rangefinder_vec],
    )

    # get distances
    # TODO(team): use BVH for high-resolution camera rangefinders
    ray.rays(
      m,
      d,
      rangefinder_pnt,
      rangefinder_vec,
      vec6(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL),
      True,
      m.sensor_rangefinder_bodyid,
      rangefinder_dist,
      rangefinder_geomid,
      rangefinder_normal,
    )

    wp.launch(
      _sensor_rangefinder,
      dim=(d.nworld, m.nrangefinder),
      inputs=[
        m.sensor_objtype,
        m.sensor_objid,
        m.sensor_intprm,
        m.sensor_adr,
        m.sensor_cutoff,
        m.sensor_rangefinder_adr,
        m.rangefinder_sensor_adr,
        d.cam_xmat,
        rangefinder_pnt,
        rangefinder_vec,
        rangefinder_dist,
        rangefinder_normal,
      ],
      outputs=[d.sensordata],
    )

  if m.sensor_e_potential:
    energy_pos(m, d)

  if m.sensor_e_kinetic:
    energy_vel(m, d)

  # collision sensors (distance, normal, fromto)
  sensor_collision = wp.full((d.nworld, m.nsensorcollision, 8, 7), 1.0e32, dtype=float)
  if m.nsensorcollision:
    wp.launch(
      _sensor_collision,
      dim=d.naconmax,
      inputs=[
        m.ngeom,
        m.nxn_pairid,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.geom,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.nacon,
      ],
      outputs=[sensor_collision],
    )

  wp.launch(
    _sensor_pos,
    dim=(d.nworld, m.sensor_pos_adr.size),
    inputs=[
      m.ngeom,
      m.opt.magnetic,
      m.body_geomnum,
      m.body_geomadr,
      m.body_iquat,
      m.body_mass,
      m.body_subtreemass,
      m.jnt_qposadr,
      m.geom_type,
      m.geom_bodyid,
      m.geom_quat,
      m.site_type,
      m.site_bodyid,
      m.site_size,
      m.site_quat,
      m.cam_bodyid,
      m.cam_quat,
      m.cam_fovy,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.nxn_pairid,
      m.sensor_pos_adr,
      d.time,
      d.energy,
      d.qpos,
      d.xpos,
      d.xquat,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.subtree_com,
      d.ten_length,
      d.actuator_length,
      sensor_collision,
    ],
    outputs=[d.sensordata],
  )

  # jointlimitpos and tendonlimitpos
  wp.launch(
    _limit_pos,
    dim=(d.nworld, d.njmax, m.sensor_limitpos_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_limitpos_adr,
      d.ne,
      d.nf,
      d.nl,
      d.efc.type,
      d.efc.id,
      d.efc.pos,
      d.efc.margin,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  _eval_sensor_contact(m, d, m.sensor_contact_pos_adr)

  # apply sensor delay/interval for position sensors
  history.apply_sensor_delay(m, d, m.sensor_stage_pos_adr)

  if m.callback.sensor:
    m.callback.sensor(m, d, Stage.POS)


@wp.func
def _velocimeter(
  # Model:
  body_rootid: wp.array[int],
  site_bodyid: wp.array[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  pos = site_xpos_in[worldid, objid]
  rot = site_xmat_in[worldid, objid]
  cvel = cvel_in[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel)
  subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
  dif = pos - subtree_com
  return wp.transpose(rot) @ (lin - wp.cross(dif, ang))


@wp.func
def _gyro(
  # Model:
  site_bodyid: wp.array[int],
  # Data in:
  site_xmat_in: wp.array2d[wp.mat33],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  rot = site_xmat_in[worldid, objid]
  cvel = cvel_in[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  return wp.transpose(rot) @ ang


@wp.func
def _joint_vel(jnt_dofadr: wp.array[int], qvel_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return qvel_in[worldid, jnt_dofadr[objid]]


@wp.func
def _tendon_vel(ten_velocity_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return ten_velocity_in[worldid, objid]


@wp.func
def _actuator_vel(actuator_velocity_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return actuator_velocity_in[worldid, objid]


@wp.func
def _ball_ang_vel(jnt_dofadr: wp.array[int], qvel_in: wp.array2d[float], worldid: int, objid: int) -> wp.vec3:
  adr = jnt_dofadr[objid]
  return wp.vec3(qvel_in[worldid, adr + 0], qvel_in[worldid, adr + 1], qvel_in[worldid, adr + 2])


@wp.kernel
def _limit_vel(
  # Model:
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_limitvel_adr: wp.array[int],
  # Data in:
  ne_in: wp.array[int],
  nf_in: wp.array[int],
  nl_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_vel_in: wp.array2d[float],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, efcid, limitvelid = wp.tid()

  ne = ne_in[worldid]
  nf = nf_in[worldid]
  nl = nl_in[worldid]

  # skip if not limit
  if efcid < ne + nf or efcid >= ne + nf + nl:
    return

  sensorid = sensor_limitvel_adr[limitvelid]
  if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:
    efc_type = efc_type_in[worldid, efcid]
    if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:
      _write_scalar(
        sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, efc_vel_in[worldid, efcid], sensordata_out[worldid]
      )


@wp.func
def _cvel_offset(
  # Model:
  body_rootid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  site_xpos_in: wp.array2d[wp.vec3],
  cam_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objtype: int,
  objid: int,
) -> Tuple[wp.spatial_vector, wp.vec3]:
  if objtype == ObjType.BODY:
    pos = xipos_in[worldid, objid]
    bodyid = objid
  elif objtype == ObjType.XBODY:
    pos = xpos_in[worldid, objid]
    bodyid = objid
  elif objtype == ObjType.GEOM:
    pos = geom_xpos_in[worldid, objid]
    bodyid = geom_bodyid[objid]
  elif objtype == ObjType.SITE:
    pos = site_xpos_in[worldid, objid]
    bodyid = site_bodyid[objid]
  elif objtype == ObjType.CAMERA:
    pos = cam_xpos_in[worldid, objid]
    bodyid = cam_bodyid[objid]
  else:  # UNKNOWN
    pos = wp.vec3(0.0)
    bodyid = 0

  return cvel_in[worldid, bodyid], pos - subtree_com_in[worldid, body_rootid[bodyid]]


@wp.func
def _frame_linvel(
  # Model:
  body_rootid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.vec3:
  if objtype == ObjType.BODY:
    xpos = xipos_in[worldid, objid]
  elif objtype == ObjType.XBODY:
    xpos = xpos_in[worldid, objid]
  elif objtype == ObjType.GEOM:
    xpos = geom_xpos_in[worldid, objid]
  elif objtype == ObjType.SITE:
    xpos = site_xpos_in[worldid, objid]
  elif objtype == ObjType.CAMERA:
    xpos = cam_xpos_in[worldid, objid]
  else:  # UNKNOWN
    xpos = wp.vec3(0.0)

  if reftype == ObjType.BODY:
    xposref = xipos_in[worldid, refid]
    xmatref = ximat_in[worldid, refid]
  elif reftype == ObjType.XBODY:
    xposref = xpos_in[worldid, refid]
    xmatref = xmat_in[worldid, refid]
  elif reftype == ObjType.GEOM:
    xposref = geom_xpos_in[worldid, refid]
    xmatref = geom_xmat_in[worldid, refid]
  elif reftype == ObjType.SITE:
    xposref = site_xpos_in[worldid, refid]
    xmatref = site_xmat_in[worldid, refid]
  elif reftype == ObjType.CAMERA:
    xposref = cam_xpos_in[worldid, refid]
    xmatref = cam_xmat_in[worldid, refid]
  else:  # UNKNOWN
    xposref = wp.vec3(0.0)
    xmatref = wp.identity(3, dtype=float)

  cvel, offset = _cvel_offset(
    body_rootid,
    geom_bodyid,
    site_bodyid,
    cam_bodyid,
    xpos_in,
    xipos_in,
    geom_xpos_in,
    site_xpos_in,
    cam_xpos_in,
    subtree_com_in,
    cvel_in,
    worldid,
    objtype,
    objid,
  )
  cvelref, offsetref = _cvel_offset(
    body_rootid,
    geom_bodyid,
    site_bodyid,
    cam_bodyid,
    xpos_in,
    xipos_in,
    geom_xpos_in,
    site_xpos_in,
    cam_xpos_in,
    subtree_com_in,
    cvel_in,
    worldid,
    reftype,
    refid,
  )
  clinvel = wp.spatial_bottom(cvel)
  cangvel = wp.spatial_top(cvel)
  cangvelref = wp.spatial_top(cvelref)
  xlinvel = clinvel - wp.cross(offset, cangvel)

  if refid > -1:
    clinvelref = wp.spatial_bottom(cvelref)
    xlinvelref = clinvelref - wp.cross(offsetref, cangvelref)
    rvec = xpos - xposref
    rel_vel = xlinvel - xlinvelref + wp.cross(rvec, cangvelref)
    return wp.transpose(xmatref) @ rel_vel
  else:
    return xlinvel


@wp.func
def _frame_angvel(
  # Model:
  body_rootid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.vec3:
  cvel, _ = _cvel_offset(
    body_rootid,
    geom_bodyid,
    site_bodyid,
    cam_bodyid,
    xpos_in,
    xipos_in,
    geom_xpos_in,
    site_xpos_in,
    cam_xpos_in,
    subtree_com_in,
    cvel_in,
    worldid,
    objtype,
    objid,
  )
  cangvel = wp.spatial_top(cvel)

  if refid > -1:
    if reftype == ObjType.BODY:
      xmatref = ximat_in[worldid, refid]
    elif reftype == ObjType.XBODY:
      xmatref = xmat_in[worldid, refid]
    elif reftype == ObjType.GEOM:
      xmatref = geom_xmat_in[worldid, refid]
    elif reftype == ObjType.SITE:
      xmatref = site_xmat_in[worldid, refid]
    elif reftype == ObjType.CAMERA:
      xmatref = cam_xmat_in[worldid, refid]
    else:  # UNKNOWN
      xmatref = wp.identity(3, dtype=float)

    cvelref, _ = _cvel_offset(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xipos_in,
      geom_xpos_in,
      site_xpos_in,
      cam_xpos_in,
      subtree_com_in,
      cvel_in,
      worldid,
      reftype,
      refid,
    )
    cangvelref = wp.spatial_top(cvelref)

    return wp.transpose(xmatref) @ (cangvel - cangvelref)
  else:
    return cangvel


@wp.func
def _subtree_linvel(subtree_linvel_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:
  return subtree_linvel_in[worldid, objid]


@wp.func
def _subtree_angmom(subtree_angmom_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:
  return subtree_angmom_in[worldid, objid]


@wp.kernel
def _sensor_vel(
  # Model:
  body_rootid: wp.array[int],
  jnt_dofadr: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_objtype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_reftype: wp.array[int],
  sensor_refid: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_vel_adr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  ten_velocity_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  cvel_in: wp.array2d[wp.spatial_vector],
  subtree_linvel_in: wp.array2d[wp.vec3],
  subtree_angmom_in: wp.array2d[wp.vec3],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, velid = wp.tid()
  sensorid = sensor_vel_adr[velid]
  sensortype = sensor_type[sensorid]
  objid = sensor_objid[sensorid]
  out = sensordata_out[worldid]

  if sensortype == SensorType.VELOCIMETER:
    vec3 = _velocimeter(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.GYRO:
    vec3 = _gyro(site_bodyid, site_xmat_in, cvel_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.JOINTVEL:
    val = _joint_vel(jnt_dofadr, qvel_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.TENDONVEL:
    val = _tendon_vel(ten_velocity_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.ACTUATORVEL:
    val = _actuator_vel(actuator_velocity_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.BALLANGVEL:
    vec3 = _ball_ang_vel(jnt_dofadr, qvel_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.FRAMELINVEL:
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    frame_linvel = _frame_linvel(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xmat_in,
      xipos_in,
      ximat_in,
      geom_xpos_in,
      geom_xmat_in,
      site_xpos_in,
      site_xmat_in,
      cam_xpos_in,
      cam_xmat_in,
      subtree_com_in,
      cvel_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_linvel, out)
  elif sensortype == SensorType.FRAMEANGVEL:
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    frame_angvel = _frame_angvel(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xmat_in,
      xipos_in,
      ximat_in,
      geom_xpos_in,
      geom_xmat_in,
      site_xpos_in,
      site_xmat_in,
      cam_xpos_in,
      cam_xmat_in,
      subtree_com_in,
      cvel_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_angvel, out)
  elif sensortype == SensorType.SUBTREELINVEL:
    vec3 = _subtree_linvel(subtree_linvel_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.SUBTREEANGMOM:
    vec3 = _subtree_angmom(subtree_angmom_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)


@event_scope
def sensor_vel(m: Model, d: Data):
  """Compute velocity-dependent sensor values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return

  if m.sensor_subtree_vel:
    smooth.subtree_vel(m, d)

  wp.launch(
    _sensor_vel,
    dim=(d.nworld, m.sensor_vel_adr.size),
    inputs=[
      m.body_rootid,
      m.jnt_dofadr,
      m.geom_bodyid,
      m.site_bodyid,
      m.cam_bodyid,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_vel_adr,
      d.qvel,
      d.xpos,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.subtree_com,
      d.ten_velocity,
      d.actuator_velocity,
      d.cvel,
      d.subtree_linvel,
      d.subtree_angmom,
    ],
    outputs=[d.sensordata],
  )

  wp.launch(
    _limit_vel,
    dim=(d.nworld, d.njmax, m.sensor_limitvel_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_limitvel_adr,
      d.ne,
      d.nf,
      d.nl,
      d.efc.type,
      d.efc.id,
      d.efc.vel,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  if m.nsensortaxel:
    weld_geom_count = wp.empty((d.nworld, m.ntactileweld), dtype=int)
    weld_geom_list = wp.empty((d.nworld, m.ntactileweld, MJ_MAXCONPAIR), dtype=int)
    nwords = (m.ngeom + 31) // 32
    weld_geom_seen = wp.zeros((d.nworld, m.ntactileweld, nwords), dtype=wp.uint32)
    wp.launch(
      _mark_tactile_contacts,
      dim=d.naconmax,
      inputs=[
        m.body_weldid,
        m.geom_bodyid,
        m.weld_tactile_id,
        d.contact.geom,
        d.contact.worldid,
        d.nacon,
      ],
      outputs=[weld_geom_seen],
    )
    wp.launch(
      _collect_tactile_contacts_kernel(int(m.opt.warn_overflow)),
      dim=(d.nworld, m.ntactileweld),
      inputs=[m.ngeom, weld_geom_seen],
      outputs=[
        d.overflow,
        weld_geom_count,
        weld_geom_list,
      ],
    )

    wp.launch(
      _sensor_tactile,
      dim=(d.nworld, m.nsensortaxel),
      inputs=[
        m.body_rootid,
        m.body_weldid,
        m.oct_child,
        m.oct_aabb,
        m.oct_coeff,
        m.geom_type,
        m.geom_bodyid,
        m.geom_dataid,
        m.geom_size,
        m.mesh_vertadr,
        m.mesh_vertnum,
        m.mesh_octadr,
        m.mesh_normaladr,
        m.mesh_normalnum,
        m.mesh_vert,
        m.mesh_normal,
        m.mesh_pos,
        m.mesh_quat,
        m.sensor_objid,
        m.sensor_refid,
        m.sensor_dim,
        m.sensor_adr,
        m.sensor_cutoff,
        m.plugin,
        m.plugin_attr,
        m.geom_plugin_index,
        m.weld_tactile_id,
        m.taxel_vertadr,
        m.taxel_sensorid,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cvel,
        weld_geom_count,
        weld_geom_list,
      ],
      outputs=[
        d.sensordata,
      ],
    )

  _eval_sensor_contact(m, d, m.sensor_contact_vel_adr)

  # apply sensor delay/interval for velocity sensors
  history.apply_sensor_delay(m, d, m.sensor_stage_vel_adr)

  if m.callback.sensor:
    m.callback.sensor(m, d, Stage.VEL)


@wp.func
def _accelerometer(
  # Model:
  body_rootid: wp.array[int],
  site_bodyid: wp.array[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  cacc_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  rot = site_xmat_in[worldid, objid]
  rotT = wp.transpose(rot)
  cvel = cvel_in[worldid, bodyid]
  cvel_top = wp.spatial_top(cvel)
  cvel_bottom = wp.spatial_bottom(cvel)
  cacc = cacc_in[worldid, bodyid]
  cacc_top = wp.spatial_top(cacc)
  cacc_bottom = wp.spatial_bottom(cacc)
  dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]
  ang = rotT @ cvel_top
  lin = rotT @ (cvel_bottom - wp.cross(dif, cvel_top))
  acc = rotT @ (cacc_bottom - wp.cross(dif, cacc_top))
  correction = wp.cross(ang, lin)
  return acc + correction


@wp.func
def _force(
  # Model:
  site_bodyid: wp.array[int],
  # Data in:
  site_xmat_in: wp.array2d[wp.mat33],
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  cfrc_int = cfrc_int_in[worldid, bodyid]
  site_xmat = site_xmat_in[worldid, objid]
  return wp.transpose(site_xmat) @ wp.spatial_bottom(cfrc_int)


@wp.func
def _torque(
  # Model:
  body_rootid: wp.array[int],
  site_bodyid: wp.array[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  cfrc_int = cfrc_int_in[worldid, bodyid]
  site_xmat = site_xmat_in[worldid, objid]
  dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]
  return wp.transpose(site_xmat) @ (wp.spatial_top(cfrc_int) - wp.cross(dif, wp.spatial_bottom(cfrc_int)))


@wp.func
def _actuator_force(actuator_force_in: wp.array2d[float], worldid: int, objid: int) -> float:
  return actuator_force_in[worldid, objid]


@wp.func
def _joint_actuator_force(
  # Model:
  jnt_dofadr: wp.array[int],
  # Data in:
  qfrc_actuator_in: wp.array2d[float],
  # In:
  worldid: int,
  objid: int,
) -> float:
  return qfrc_actuator_in[worldid, jnt_dofadr[objid]]


@cache_kernel
def _tendon_actuator_force(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    sensor_objid: wp.array[int],
    sensor_adr: wp.array[int],
    sensor_tendonactfrc_adr: wp.array[int],
    # Data in:
    actuator_force_in: wp.array2d[float],
    # Data out:
    sensordata_out: wp.array2d[float],
  ):
    worldid, tenactfrcid, actid = wp.tid()
    sensorid = sensor_tendonactfrc_adr[tenactfrcid]

    if actuator_trntype[actid] == TrnType.TENDON and actuator_trnid[actid][0] == sensor_objid[sensorid]:
      adr = sensor_adr[sensorid]
      wp.atomic_add(sensordata_out, worldid, adr, actuator_force_in[worldid, actid])

  return kernel


@wp.kernel
def _tendon_actuator_force_cutoff(
  # Model:
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_tendonactfrc_adr: wp.array[int],
  # Data in:
  sensordata_in: wp.array2d[float],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, tenactfrcid = wp.tid()
  sensorid = sensor_tendonactfrc_adr[tenactfrcid]
  adr = sensor_adr[sensorid]
  val = sensordata_in[worldid, adr]

  _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, sensordata_out[worldid])


@wp.kernel
def _limit_frc(
  # Model:
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_limitfrc_adr: wp.array[int],
  # Data in:
  ne_in: wp.array[int],
  nf_in: wp.array[int],
  nl_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_force_in: wp.array2d[float],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, efcid, limitfrcid = wp.tid()

  ne = ne_in[worldid]
  nf = nf_in[worldid]
  nl = nl_in[worldid]

  # skip if not limit
  if efcid < ne + nf or efcid >= ne + nf + nl:
    return

  sensorid = sensor_limitfrc_adr[limitfrcid]
  if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:
    efc_type = efc_type_in[worldid, efcid]
    if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:
      _write_scalar(
        sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, efc_force_in[worldid, efcid], sensordata_out[worldid]
      )


@wp.func
def _framelinacc(
  # Model:
  body_rootid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  site_xpos_in: wp.array2d[wp.vec3],
  cam_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  cacc_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
) -> wp.vec3:
  if objtype == ObjType.BODY:
    bodyid = objid
    pos = xipos_in[worldid, objid]
  elif objtype == ObjType.XBODY:
    bodyid = objid
    pos = xpos_in[worldid, objid]
  elif objtype == ObjType.GEOM:
    bodyid = geom_bodyid[objid]
    pos = geom_xpos_in[worldid, objid]
  elif objtype == ObjType.SITE:
    bodyid = site_bodyid[objid]
    pos = site_xpos_in[worldid, objid]
  elif objtype == ObjType.CAMERA:
    bodyid = cam_bodyid[objid]
    pos = cam_xpos_in[worldid, objid]
  else:  # UNKNOWN
    bodyid = 0
    pos = wp.vec3(0.0)

  cacc = cacc_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  offset = pos - subtree_com_in[worldid, body_rootid[bodyid]]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel) - wp.cross(offset, ang)
  acc = wp.spatial_bottom(cacc) - wp.cross(offset, wp.spatial_top(cacc))
  correction = wp.cross(ang, lin)

  return acc + correction


@wp.func
def _frameangacc(
  # Model:
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  # Data in:
  cacc_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objid: int,
  objtype: int,
) -> wp.vec3:
  if objtype == ObjType.BODY or objtype == ObjType.XBODY:
    bodyid = objid
  elif objtype == ObjType.GEOM:
    bodyid = geom_bodyid[objid]
  elif objtype == ObjType.SITE:
    bodyid = site_bodyid[objid]
  elif objtype == ObjType.CAMERA:
    bodyid = cam_bodyid[objid]
  else:  # UNKNOWN
    bodyid = 0

  return wp.spatial_top(cacc_in[worldid, bodyid])


@wp.kernel
def _sensor_acc(
  # Model:
  body_rootid: wp.array[int],
  jnt_dofadr: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  cam_bodyid: wp.array[int],
  sensor_type: wp.array[int],
  sensor_datatype: wp.array[int],
  sensor_objtype: wp.array[int],
  sensor_objid: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  sensor_acc_adr: wp.array[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  site_xpos_in: wp.array2d[wp.vec3],
  site_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  actuator_force_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  cacc_in: wp.array2d[wp.spatial_vector],
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, accid = wp.tid()
  sensorid = sensor_acc_adr[accid]
  sensortype = sensor_type[sensorid]
  objid = sensor_objid[sensorid]
  out = sensordata_out[worldid]

  if sensortype == SensorType.ACCELEROMETER:
    vec3 = _accelerometer(
      body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, cacc_in, worldid, objid
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.FORCE:
    vec3 = _force(site_bodyid, site_xmat_in, cfrc_int_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.TORQUE:
    vec3 = _torque(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cfrc_int_in, worldid, objid)
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.ACTUATORFRC:
    val = _actuator_force(actuator_force_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.JOINTACTFRC:
    val = _joint_actuator_force(jnt_dofadr, qfrc_actuator_in, worldid, objid)
    _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == SensorType.FRAMELINACC:
    objtype = sensor_objtype[sensorid]
    vec3 = _framelinacc(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xipos_in,
      geom_xpos_in,
      site_xpos_in,
      cam_xpos_in,
      subtree_com_in,
      cvel_in,
      cacc_in,
      worldid,
      objid,
      objtype,
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == SensorType.FRAMEANGACC:
    objtype = sensor_objtype[sensorid]
    vec3 = _frameangacc(
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      cacc_in,
      worldid,
      objid,
      objtype,
    )
    _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)


@cache_kernel
def _sensor_touch(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_cone: int,
    geom_bodyid: wp.array[int],
    site_type: wp.array[int],
    site_bodyid: wp.array[int],
    site_size: wp.array[wp.vec3],
    sensor_objid: wp.array[int],
    sensor_adr: wp.array[int],
    sensor_touch_adr: wp.array[int],
    # Data in:
    site_xpos_in: wp.array2d[wp.vec3],
    site_xmat_in: wp.array2d[wp.mat33],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_force_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # Data out:
    sensordata_out: wp.array2d[float],
  ):
    conid, sensortouchadrid = wp.tid()

    if conid >= nacon_in[0]:
      return

    sensorid = sensor_touch_adr[sensortouchadrid]

    objid = sensor_objid[sensorid]
    bodyid = site_bodyid[objid]

    # find contact in sensor zone, add normal force

    # contacting bodies
    geom = contact_geom_in[conid]
    conbody = wp.vec2i(geom_bodyid[geom[0]], geom_bodyid[geom[1]])

    # select contacts involving sensorized body
    worldid = contact_worldid_in[conid]
    efc_address0 = contact_efc_address_in[conid, 0]
    if efc_address0 >= 0 and (bodyid == conbody[0] or bodyid == conbody[1]):
      # get contact normal force
      normalforce = efc_force_in[worldid, efc_address0]

      if opt_cone == ConeType.PYRAMIDAL:
        dim = contact_dim_in[conid]
        for i in range(1, 2 * (dim - 1)):
          normalforce += efc_force_in[worldid, contact_efc_address_in[conid, i]]

      if normalforce <= 0.0:
        return

      # convert contact normal force to global frame, normalize
      frame = contact_frame_in[conid]
      conray = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2]) * normalforce
      conray, _ = math.normalize_with_norm(conray)

      # flip ray direction if sensor is on body2
      if bodyid == conbody[1]:
        conray = -conray

      # add if ray-zone intersection (always true when contact.pos inside zone)
      dist, normal = ray.ray_geom(
        site_xpos_in[worldid, objid],
        site_xmat_in[worldid, objid],
        site_size[objid],
        contact_pos_in[conid],
        conray,
        site_type[objid],
      )
      if dist >= 0.0:
        adr = sensor_adr[sensorid]
        wp.atomic_add(sensordata_out[worldid], adr, normalforce)

  return kernel


@wp.func
def _transform_spatial(vec: wp.spatial_vector, dif: wp.vec3) -> wp.vec3:
  return wp.spatial_bottom(vec) - wp.cross(dif, wp.spatial_top(vec))


@wp.kernel(enable_backward=False, grid_stride=True)
def _mark_tactile_contacts(
  # Model:
  body_weldid: wp.array[int],
  geom_bodyid: wp.array[int],
  weld_tactile_id: wp.array[int],
  # Data in:
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  weld_geom_seen_out: wp.array3d[wp.uint32],
):
  conid = wp.tid()
  if conid >= wp.min(nacon_in[0], contact_geom_in.shape[0]):
    return

  contact_geom = contact_geom_in[conid]
  geom1 = contact_geom[0]
  geom2 = contact_geom[1]

  # TODO(team): support flex body contacts (MuJoCo C resolves via mj_flexBody).
  # Flex body contacts have negative geom indices; skip for now.
  if geom1 < 0 or geom2 < 0:
    return

  worldid = contact_worldid_in[conid]
  weld1 = body_weldid[geom_bodyid[geom1]]
  weld2 = body_weldid[geom_bodyid[geom2]]

  for side in range(2):
    weld = weld1 if side == 0 else weld2
    geom = geom2 if side == 0 else geom1

    # Skip bodies without tactile sensors
    tactile_idx = weld_tactile_id[weld]
    if tactile_idx < 0:
      continue

    word_idx = geom // 32
    bit_mask = wp.uint32(1) << wp.uint32(geom % 32)
    wp.atomic_or(weld_geom_seen_out[worldid, tactile_idx], word_idx, bit_mask)


@cache_kernel
def _collect_tactile_contacts_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def _collect_tactile_contacts(
    # Model:
    ngeom: int,
    # In:
    weld_geom_seen_in: wp.array3d[wp.uint32],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    weld_geom_count_out: wp.array2d[int],
    weld_geom_list_out: wp.array3d[int],
  ):
    worldid, tactile_idx = wp.tid()
    idx = int(0)
    nwords = (ngeom + 31) // 32
    for word_idx in range(nwords):
      word = weld_geom_seen_in[worldid, tactile_idx, word_idx]
      if word == wp.uint32(0):
        continue
      for bit in range(32):
        geom = word_idx * 32 + bit
        if geom < ngeom and (word & (wp.uint32(1) << wp.uint32(bit))) != wp.uint32(0):
          if idx < MJ_MAXCONPAIR:
            weld_geom_list_out[worldid, tactile_idx, idx] = geom
          elif idx == MJ_MAXCONPAIR:
            if wp.static(bool(warn_overflow & OverflowType.TACTILE)):
              wp.printf(
                "tactile sensor collision pair overflow: number of contacting geoms >= %u\n"
                "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.TACTILE (or = 0 for all)\n",
                MJ_MAXCONPAIR,
              )
            wp.atomic_or(overflow_out, worldid, int(OverflowType.TACTILE))
          idx += 1
    weld_geom_count_out[worldid, tactile_idx] = idx

  return _collect_tactile_contacts


@wp.kernel
def _sensor_tactile(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_octadr: wp.array[int],
  mesh_normaladr: wp.array[int],
  mesh_normalnum: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_normal: wp.array[wp.vec3],
  mesh_pos: wp.array[wp.vec3],
  mesh_quat: wp.array[wp.quat],
  sensor_objid: wp.array[int],
  sensor_refid: wp.array[int],
  sensor_dim: wp.array[int],
  sensor_adr: wp.array[int],
  sensor_cutoff: wp.array[float],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  geom_plugin_index: wp.array[int],
  weld_tactile_id: wp.array[int],
  taxel_vertadr: wp.array[int],
  taxel_sensorid: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  weld_geom_count_in: wp.array2d[int],
  weld_geom_list_in: wp.array3d[int],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, taxelid = wp.tid()

  sensor_id = taxel_sensorid[taxelid]
  mesh_id = sensor_objid[sensor_id]
  geom_id = sensor_refid[sensor_id]
  parent_body = geom_bodyid[geom_id]
  parent_weld = body_weldid[parent_body]
  tactile_idx = weld_tactile_id[parent_weld]

  ncon = mesh_vertnum[mesh_id]
  nchannel = sensor_dim[sensor_id] // ncon
  vertid = taxel_vertadr[taxelid] - mesh_vertadr[mesh_id]

  # Always clear output for this taxel first (matching MuJoCo C mju_zero)
  for c in range(nchannel):
    sensordata_out[worldid, sensor_adr[sensor_id] + c * ncon + vertid] = 0.0

  if tactile_idx < 0:
    return

  geom_count = weld_geom_count_in[worldid, tactile_idx]
  if geom_count == 0:
    return

  # vertex local position
  pos = mesh_vert[vertid + mesh_vertadr[mesh_id]]

  # position in global frame
  sensor_mat = geom_xmat_in[worldid, geom_id]
  xpos = sensor_mat @ pos + geom_xpos_in[worldid, geom_id]

  has_frame = (mesh_normalnum[mesh_id] == 3 * ncon) and (nchannel >= 3)
  offset = int(0)
  if has_frame:
    offset = mesh_normaladr[mesh_id] + 3 * vertid

  vel_sensor = _transform_spatial(
    cvel_in[worldid, parent_weld],
    xpos - subtree_com_in[worldid, body_rootid[parent_weld]],
  )

  max_depth = float(0.0)
  tang1_acc = float(0.0)
  tang2_acc = float(0.0)

  max_geoms = wp.min(geom_count, MJ_MAXCONPAIR)
  for g in range(max_geoms):
    geom = weld_geom_list_in[worldid, tactile_idx, g]
    # Unlike MuJoCo C which excludes self-geom during preprocessing, weld_geom_list is shared
    # across all sensors on the same weld body. Skipping geom == geom_id here ensures each
    # sensor on the weld body filters its own geom without affecting others.
    if geom < 0 or geom == geom_id:
      continue

    contact_type = geom_type[geom]
    other_dataid = geom_dataid[worldid % geom_dataid.shape[0], geom]

    # Gracefully skip mesh geoms without octrees (matches MuJoCo C)
    if contact_type == GeomType.MESH and mesh_octadr[other_dataid] == -1:
      continue

    body = geom_bodyid[geom]

    tmp = xpos - geom_xpos_in[worldid, geom]
    lpos = wp.transpose(geom_xmat_in[worldid, geom]) @ tmp

    plugin_id = geom_plugin_index[geom]
    # SDF plugins are in the original mesh frame (matches MuJoCo C)
    if contact_type == GeomType.SDF and plugin_id != -1 and other_dataid >= 0:
      lpos = math.rot_vec_quat(lpos, mesh_quat[other_dataid]) + mesh_pos[other_dataid]

    plugin_attributes, plugin_index, volume_data, mesh_data = get_sdf_params(
      oct_child,
      oct_aabb,
      oct_coeff,
      mesh_octadr,
      plugin,
      plugin_attr,
      contact_type,
      geom_size[worldid % geom_size.shape[0], geom],
      plugin_id,
      other_dataid,
    )

    depth = wp.min(sdf(contact_type, lpos, plugin_attributes, plugin_index, volume_data, mesh_data), 0.0)
    if depth >= 0.0:
      continue

    if -depth > max_depth:
      max_depth = -depth

    if has_frame:
      vel_other = _transform_spatial(
        cvel_in[worldid, body],
        xpos - subtree_com_in[worldid, body_rootid[body]],
      )
      vel_rel = vel_sensor - vel_other
      vel_rel_geom = wp.transpose(sensor_mat) @ vel_rel
      tang1_acc += wp.abs(wp.dot(vel_rel_geom, mesh_normal[offset + 1]))
      tang2_acc += wp.abs(wp.dot(vel_rel_geom, mesh_normal[offset + 2]))

  cutoff = sensor_cutoff[sensor_id]
  if cutoff > 0.0:
    max_depth = wp.clamp(max_depth, -cutoff, cutoff)
    tang1_acc = wp.clamp(tang1_acc, -cutoff, cutoff)
    tang2_acc = wp.clamp(tang2_acc, -cutoff, cutoff)

  sensordata_out[worldid, sensor_adr[sensor_id] + 0 * ncon + vertid] = max_depth
  if has_frame:
    sensordata_out[worldid, sensor_adr[sensor_id] + 1 * ncon + vertid] = tang1_acc
    sensordata_out[worldid, sensor_adr[sensor_id] + 2 * ncon + vertid] = tang2_acc


@wp.func
def _check_match(body_parentid: wp.array[int], body: int, geom: int, objtype: int, objid: int) -> bool:
  """Check if a contact body/geom matches a sensor spec (objtype, objid)."""
  if objtype == ObjType.UNKNOWN:
    return True
  if objtype == ObjType.SITE:
    return True  # already passed site filter test
  if objtype == ObjType.GEOM:
    return objid == geom
  if objtype == ObjType.BODY:
    return objid == body
  if objtype == ObjType.XBODY:
    # traverse up the tree from body, return true if we land on id
    while body > objid:
      body = body_parentid[body]
    return body == objid
  return False


@cache_kernel
def _contact_match_builder(warn_overflow: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def _contact_match(
    # Model:
    opt_cone: int,
    opt_contact_sensor_maxmatch: int,
    body_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    site_type: wp.array[int],
    site_size: wp.array[wp.vec3],
    sensor_objtype: wp.array[int],
    sensor_objid: wp.array[int],
    sensor_reftype: wp.array[int],
    sensor_refid: wp.array[int],
    sensor_intprm: wp.array2d[int],
    # Data in:
    site_xpos_in: wp.array2d[wp.vec3],
    site_xmat_in: wp.array2d[wp.mat33],
    contact_dist_in: wp.array[float],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    contact_type_in: wp.array[int],
    contact_adhesion_in: wp.array[float],
    efc_force_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    # In:
    sensor_contact_adr_in: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    sensor_contact_nmatch_out: wp.array2d[int],
    sensor_contact_matchid_out: wp.array3d[int],
    sensor_contact_criteria_out: wp.array3d[float],
  ):
    contactsensorid, contactid = wp.tid()
    sensorid = sensor_contact_adr_in[contactsensorid]

    if contactid >= wp.min(nacon_in[0], contact_geom_in.shape[0]):
      return

    if not (contact_type_in[contactid] & ContactType.CONSTRAINT):
      return

    # sensor information
    objtype = sensor_objtype[sensorid]
    objid = sensor_objid[sensorid]
    reftype = sensor_reftype[sensorid]
    refid = sensor_refid[sensorid]
    reduce = sensor_intprm[sensorid, 1]

    worldid = contact_worldid_in[contactid]

    # site filter
    if objtype == ObjType.SITE:
      if not inside_geom(
        site_xpos_in[worldid, objid],
        site_xmat_in[worldid, objid],
        site_size[objid],
        site_type[objid],
        contact_pos_in[contactid],
      ):
        return

    # unknown-unknown match
    if objtype == ObjType.UNKNOWN and reftype == ObjType.UNKNOWN:
      dir = 1.0
    else:
      # contact information
      geom = contact_geom_in[contactid]
      geom1 = geom[0]
      geom2 = geom[1]
      body1 = geom_bodyid[geom1]
      body2 = geom_bodyid[geom2]

      # check match of sensor objects with contact objects
      match11 = _check_match(body_parentid, body1, geom1, objtype, objid)
      match12 = _check_match(body_parentid, body2, geom2, objtype, objid)
      match21 = _check_match(body_parentid, body1, geom1, reftype, refid)
      match22 = _check_match(body_parentid, body2, geom2, reftype, refid)

      # if a sensor object is specified, it must be involved in the contact
      if not match11 and not match12:
        return
      if not match21 and not match22:
        return

      # determine direction
      dir = 1.0
      if objtype != ObjType.UNKNOWN and reftype != ObjType.UNKNOWN:
        # both obj1 and obj2 specified: direction depends on order
        order_regular = match11 and match22
        order_reverse = match12 and match21
        if not order_regular and not order_reverse:
          return
        if order_reverse and not order_regular:
          dir = -1.0
      elif objtype != ObjType.UNKNOWN:
        if not match11:
          dir = -1.0
      elif reftype != ObjType.UNKNOWN:
        if not match22:
          dir = -1.0

    contactmatchid = wp.atomic_add(sensor_contact_nmatch_out[worldid], contactsensorid, 1)

    if contactmatchid >= opt_contact_sensor_maxmatch:
      if wp.static(bool(warn_overflow & OverflowType.CONTACT_MATCH)):
        wp.printf(
          "contact match overflow: please increase Option.contact_sensor_maxmatch beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CONTACT_MATCH (or = 0 for all)\n",
          opt_contact_sensor_maxmatch,
        )
      wp.atomic_or(overflow_out, worldid, OverflowType.CONTACT_MATCH)
      return

    # Keep direction attached to the ID through sorting without a second payload.
    # Complement encodes a negative direction even for contact zero (~0 == -1).
    sensor_contact_matchid_out[worldid, contactsensorid, contactmatchid] = ~contactid if dir < 0.0 else contactid

    if reduce == 1:  # mindist
      sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = contact_dist_in[contactid]
    elif reduce == 2:  # maxforce
      contact_force = support.contact_force_fn(
        opt_cone,
        contact_frame_in,
        contact_friction_in,
        contact_dim_in,
        contact_efc_address_in,
        contact_adhesion_in,
        efc_force_in,
        njmax_in,
        nacon_in,
        worldid,
        contactid,
        False,
      )
      force_magnitude = (
        contact_force[0] * contact_force[0] + contact_force[1] * contact_force[1] + contact_force[2] * contact_force[2]
      )
      sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = -force_magnitude

  return _contact_match


@cache_kernel
def _contact_sort(maxmatch: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def contact_sort(
    # Model:
    sensor_intprm: wp.array2d[int],
    # In:
    sensor_contact_adr_in: wp.array[int],
    sensor_contact_nmatch_in: wp.array2d[int],
    sensor_contact_matchid_in: wp.array3d[int],
    sensor_contact_criteria_in: wp.array3d[float],
    # Out:
    sensor_contact_matchid_out: wp.array3d[int],
  ):
    worldid, contactsensorid = wp.tid()
    sensorid = sensor_contact_adr_in[contactsensorid]

    reduce = sensor_intprm[sensorid, 1]
    if reduce == 0 or reduce == 3:  # none or netforce
      return

    nmatch = sensor_contact_nmatch_in[worldid, contactsensorid]

    # skip sort
    if nmatch <= 1:
      return

    criteria_tile = wp.tile_load(sensor_contact_criteria_in[worldid, contactsensorid], shape=maxmatch)
    matchid_tile = wp.tile_load(sensor_contact_matchid_in[worldid, contactsensorid], shape=maxmatch)
    wp.tile_sort(criteria_tile, matchid_tile)
    wp.tile_store(sensor_contact_matchid_out[worldid, contactsensorid], matchid_tile)

  return contact_sort


@wp.kernel
def _sensor_contact(
  # Model:
  opt_cone: int,
  opt_contact_sensor_maxmatch: int,
  body_rootid: wp.array[int],
  geom_bodyid: wp.array[int],
  sensor_intprm: wp.array2d[int],
  sensor_dim: wp.array[int],
  sensor_adr: wp.array[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  contact_dist_in: wp.array[float],
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_friction_in: wp.array[vec5],
  contact_dim_in: wp.array[int],
  contact_geom_in: wp.array[wp.vec2i],
  contact_efc_address_in: wp.array2d[int],
  contact_adhesion_in: wp.array[float],
  efc_force_in: wp.array2d[float],
  njmax_in: int,
  nacon_in: wp.array[int],
  # In:
  sensor_contact_adr_in: wp.array[int],
  sensor_contact_nmatch_in: wp.array2d[int],
  sensor_contact_matchid_in: wp.array3d[int],
  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, contactsensorid = wp.tid()
  sensorid = sensor_contact_adr_in[contactsensorid]
  out = sensordata_out[worldid]

  dataspec = sensor_intprm[sensorid, 0]
  reduce = sensor_intprm[sensorid, 1]
  dim = sensor_dim[sensorid]

  # TODO(thowell): precompute slot size
  found = bool(dataspec & (1 << ConDataField.FOUND))
  force = bool(dataspec & (1 << ConDataField.FORCE))
  torque = bool(dataspec & (1 << ConDataField.TORQUE))
  dist = bool(dataspec & (1 << ConDataField.DIST))
  pos = bool(dataspec & (1 << ConDataField.POS))
  normal = bool(dataspec & (1 << ConDataField.NORMAL))
  tangent = bool(dataspec & (1 << ConDataField.TANGENT))
  linvel = bool(dataspec & (1 << ConDataField.LINVEL))
  angvel = bool(dataspec & (1 << ConDataField.ANGVEL))

  size = (
    (1 if found else 0)
    + (3 if force else 0)
    + (3 if torque else 0)
    + (1 if dist else 0)
    + (3 if pos else 0)
    + (3 if normal else 0)
    + (3 if tangent else 0)
    + (3 if linvel else 0)
    + (3 if angvel else 0)
  )

  num = dim // size  # number of slots

  adr = sensor_adr[sensorid]
  nmatch_raw = sensor_contact_nmatch_in[worldid, contactsensorid]
  nmatch = wp.min(nmatch_raw, opt_contact_sensor_maxmatch)

  if reduce == 3:  # netforce
    if nmatch == 0:
      for i in range(dim):
        out[adr + i] = 0.0
      return

    net_pos = wp.vec3(0.0)
    net_force = wp.vec3(0.0)
    net_torque = wp.vec3(0.0)
    total_force_magnitude = float(0.0)

    for i in range(nmatch):
      matchid = sensor_contact_matchid_in[worldid, contactsensorid, i]
      cid = ~matchid if matchid < 0 else matchid
      dir = -1.0 if matchid < 0 else 1.0

      contact_forcetorque = support.contact_force_fn(
        opt_cone,
        contact_frame_in,
        contact_friction_in,
        contact_dim_in,
        contact_efc_address_in,
        contact_adhesion_in,
        efc_force_in,
        njmax_in,
        nacon_in,
        worldid,
        cid,
        False,
      )

      # Accumulate for centroid computation (unsigned force magnitude)
      weight = wp.norm_l2(wp.spatial_top(contact_forcetorque))
      contact_pos = contact_pos_in[cid]
      net_pos += weight * contact_pos
      total_force_magnitude += weight

      # Apply direction and transform to global frame
      contact_forcetorque *= dir
      force_local = wp.spatial_top(contact_forcetorque)
      torque_local = wp.spatial_bottom(contact_forcetorque)

      frame = contact_frame_in[cid]
      frameT = wp.transpose(frame)

      force_global = frameT @ force_local
      torque_global = frameT @ torque_local

      # Accumulate force and torque (about origin for now)
      net_force += force_global
      net_torque += torque_global
      # Accumulate moment contribution: will adjust after centroid is computed
      net_torque += wp.cross(contact_pos, force_global)

    # Finalize centroid
    net_pos /= wp.max(total_force_magnitude, MJ_MINVAL)

    # Adjust torque: subtract moment from centroid (since we accumulated about origin)
    # torque_about_centroid = torque_about_origin - centroid x total_force
    net_torque -= wp.cross(net_pos, net_force)

    adr_slot = adr

    if found:
      out[adr_slot] = float(nmatch_raw)
      adr_slot += 1
    if force:
      out[adr_slot + 0] = net_force[0]
      out[adr_slot + 1] = net_force[1]
      out[adr_slot + 2] = net_force[2]
      adr_slot += 3
    if torque:
      out[adr_slot + 0] = net_torque[0]
      out[adr_slot + 1] = net_torque[1]
      out[adr_slot + 2] = net_torque[2]
      adr_slot += 3
    if dist:
      out[adr_slot] = 0.0
      adr_slot += 1
    if pos:
      out[adr_slot + 0] = net_pos[0]
      out[adr_slot + 1] = net_pos[1]
      out[adr_slot + 2] = net_pos[2]
      adr_slot += 3
    if normal:
      out[adr_slot + 0] = 1.0
      out[adr_slot + 1] = 0.0
      out[adr_slot + 2] = 0.0
      adr_slot += 3
    if tangent:
      out[adr_slot + 0] = 0.0
      out[adr_slot + 1] = 1.0
      out[adr_slot + 2] = 0.0

    # zero remaining slots
    for i in range(size, dim):
      out[adr + i] = 0.0
  else:
    nslots = wp.min(nmatch, num)
    for i in range(nslots):
      # Decode only valid matches; unused padded slots are bounded by nslots.
      matchid = sensor_contact_matchid_in[worldid, contactsensorid, i]
      cid = ~matchid if matchid < 0 else matchid
      dir = -1.0 if matchid < 0 else 1.0

      adr_slot = adr + i * size

      if found:
        out[adr_slot] = float(nmatch_raw)
        adr_slot += 1
      if force or torque:
        contact_forcetorque = support.contact_force_fn(
          opt_cone,
          contact_frame_in,
          contact_friction_in,
          contact_dim_in,
          contact_efc_address_in,
          contact_adhesion_in,
          efc_force_in,
          njmax_in,
          nacon_in,
          worldid,
          cid,
          False,
        )
      if force:
        out[adr_slot + 0] = contact_forcetorque[0]
        out[adr_slot + 1] = contact_forcetorque[1]
        out[adr_slot + 2] = dir * contact_forcetorque[2]
        adr_slot += 3
      if torque:
        out[adr_slot + 0] = contact_forcetorque[3]
        out[adr_slot + 1] = contact_forcetorque[4]
        out[adr_slot + 2] = dir * contact_forcetorque[5]
        adr_slot += 3
      if dist:
        out[adr_slot] = contact_dist_in[cid]
        adr_slot += 1
      if pos or linvel:
        contact_pos = contact_pos_in[cid]
      if pos:
        out[adr_slot + 0] = contact_pos[0]
        out[adr_slot + 1] = contact_pos[1]
        out[adr_slot + 2] = contact_pos[2]
        adr_slot += 3
      if normal or tangent or linvel or angvel:
        frame = contact_frame_in[cid]
      if normal:
        contact_normal = frame[0]
        out[adr_slot + 0] = dir * contact_normal[0]
        out[adr_slot + 1] = dir * contact_normal[1]
        out[adr_slot + 2] = dir * contact_normal[2]
        adr_slot += 3
      if tangent:
        contact_tangent = frame[1]
        out[adr_slot + 0] = dir * contact_tangent[0]
        out[adr_slot + 1] = dir * contact_tangent[1]
        out[adr_slot + 2] = dir * contact_tangent[2]
        adr_slot += 3
      if linvel or angvel:
        # velocity of geom[1] relative to geom[0] at the contact point, in the contact frame
        con_geom = contact_geom_in[cid]
        body0 = geom_bodyid[con_geom[0]]
        body1 = geom_bodyid[con_geom[1]]
        cvel0 = cvel_in[worldid, body0]
        cvel1 = cvel_in[worldid, body1]
      if linvel:
        root0 = body_rootid[body0]
        root1 = body_rootid[body1]
        com0 = subtree_com_in[worldid, root0]
        if root1 == root0:
          com1 = com0
        else:
          com1 = subtree_com_in[worldid, root1]
        rel_lin = _transform_spatial(cvel1, contact_pos - com1) - _transform_spatial(cvel0, contact_pos - com0)
        lin_local = frame @ rel_lin
        out[adr_slot + 0] = lin_local[0]
        out[adr_slot + 1] = lin_local[1]
        out[adr_slot + 2] = dir * lin_local[2]
        adr_slot += 3
      if angvel:
        ang_local = frame @ (wp.spatial_top(cvel1) - wp.spatial_top(cvel0))
        out[adr_slot + 0] = ang_local[0]
        out[adr_slot + 1] = ang_local[1]
        out[adr_slot + 2] = dir * ang_local[2]

    # zero remaining slots
    for i in range(nslots, num):
      for j in range(size):
        out[adr + i * size + j] = 0.0


def _eval_sensor_contact(m: Model, d: Data, sensor_contact_adr: wp.array[int]):
  if not sensor_contact_adr.size:
    return

  ncontact = sensor_contact_adr.size
  maxmatch = m.opt.contact_sensor_maxmatch
  sensor_contact_nmatch = wp.zeros((d.nworld, ncontact), dtype=int)
  sensor_contact_matchid = wp.full((d.nworld, ncontact, maxmatch), -1, dtype=int)
  sensor_contact_criteria = wp.full((d.nworld, ncontact, maxmatch), 1.0e32, dtype=float)

  wp.launch(
    _contact_match_builder(int(m.opt.warn_overflow), bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(ncontact, d.naconmax),
    inputs=[
      m.opt.cone,
      maxmatch,
      m.body_parentid,
      m.geom_bodyid,
      m.site_type,
      m.site_size,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_intprm,
      d.site_xpos,
      d.site_xmat,
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.friction,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.type,
      d.contact.adhesion,
      d.efc.force,
      d.njmax,
      d.nacon,
      sensor_contact_adr,
    ],
    outputs=[d.overflow, sensor_contact_nmatch, sensor_contact_matchid, sensor_contact_criteria],
  )

  # sorting
  wp.launch_tiled(
    _contact_sort(maxmatch),
    dim=(d.nworld, ncontact),
    inputs=[
      m.sensor_intprm,
      sensor_contact_adr,
      sensor_contact_nmatch,
      sensor_contact_matchid,
      sensor_contact_criteria,
    ],
    outputs=[sensor_contact_matchid],
    block_dim=m.block_dim.contact_sort,
  )

  wp.launch(
    _sensor_contact,
    dim=(d.nworld, ncontact),
    inputs=[
      m.opt.cone,
      maxmatch,
      m.body_rootid,
      m.geom_bodyid,
      m.sensor_intprm,
      m.sensor_dim,
      m.sensor_adr,
      d.subtree_com,
      d.cvel,
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.friction,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.adhesion,
      d.efc.force,
      d.njmax,
      d.nacon,
      sensor_contact_adr,
      sensor_contact_nmatch,
      sensor_contact_matchid,
    ],
    outputs=[d.sensordata],
  )


@event_scope
def sensor_acc(m: Model, d: Data, skip_rne_postconstraint: bool = False):
  """Compute acceleration-dependent sensor values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return

  wp.launch(
    _sensor_touch(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.naconmax, m.sensor_touch_adr.size),
    inputs=[
      m.opt.cone,
      m.geom_bodyid,
      m.site_type,
      m.site_bodyid,
      m.site_size,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_touch_adr,
      d.site_xpos,
      d.site_xmat,
      d.contact.pos,
      d.contact.frame,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.efc.force,
      d.nacon,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  _eval_sensor_contact(m, d, m.sensor_contact_acc_adr)

  if not skip_rne_postconstraint and m.sensor_rne_postconstraint:
    smooth.rne_postconstraint(m, d)

  wp.launch(
    _sensor_acc,
    dim=(d.nworld, m.sensor_acc_adr.size),
    inputs=[
      m.body_rootid,
      m.jnt_dofadr,
      m.geom_bodyid,
      m.site_bodyid,
      m.cam_bodyid,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_acc_adr,
      d.xpos,
      d.xipos,
      d.geom_xpos,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.subtree_com,
      d.cvel,
      d.actuator_force,
      d.qfrc_actuator,
      d.cacc,
      d.cfrc_int,
    ],
    outputs=[d.sensordata],
  )

  wp.launch(
    _tendon_actuator_force(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.sensor_tendonactfrc_adr.size, m.nactuator),
    inputs=[
      m.actuator_trntype,
      m.actuator_trnid,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_tendonactfrc_adr,
      d.actuator_force,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  wp.launch(
    _tendon_actuator_force_cutoff,
    dim=(d.nworld, m.sensor_tendonactfrc_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_tendonactfrc_adr,
      d.sensordata,
    ],
    outputs=[d.sensordata],
  )

  wp.launch(
    _limit_frc,
    dim=(d.nworld, d.njmax, m.sensor_limitfrc_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_limitfrc_adr,
      d.ne,
      d.nf,
      d.nl,
      d.efc.type,
      d.efc.id,
      d.efc.force,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  # apply sensor delay/interval for acceleration sensors
  history.apply_sensor_delay(m, d, m.sensor_stage_acc_adr)

  if m.callback.sensor:
    m.callback.sensor(m, d, Stage.ACC)


@wp.kernel
def _energy_pos_zero(
  # Data out:
  energy_out: wp.array[wp.vec2],
):
  worldid = wp.tid()
  energy_out[worldid][0] = 0.0


@cache_kernel
def _energy_pos_gravity(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_gravity: wp.array[wp.vec3],
    body_mass: wp.array2d[float],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    # Data out:
    energy_out: wp.array[wp.vec2],
  ):
    worldid, bodyid = wp.tid()
    gravity = opt_gravity[worldid % opt_gravity.shape[0]]
    bodyid += 1  # skip world body

    energy = wp.vec2(
      body_mass[worldid % body_mass.shape[0], bodyid] * wp.dot(gravity, xipos_in[worldid, bodyid]),
      0.0,
    )

    wp.atomic_sub(energy_out, worldid, energy)

  return kernel


@cache_kernel
def _energy_pos_passive_joint(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    qpos_spring: wp.array2d[float],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_plasticadr: wp.array[int],
    jnt_stiffness: wp.array2d[float],
    jnt_stiffnesspoly: wp.array2d[wp.vec2],
    # Data in:
    qpos_in: wp.array2d[float],
    plastic_in: wp.array2d[float],
    # Data out:
    energy_out: wp.array[wp.vec2],
  ):
    worldid, jntid = wp.tid()
    jnt_stiffness_id = worldid % jnt_stiffness.shape[0]
    stiffness = jnt_stiffness[jnt_stiffness_id, jntid]
    spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]

    if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:
      return

    padr = jnt_qposadr[jntid]
    plasticadr = jnt_plasticadr[jntid]
    jnttype = jnt_type[jntid]
    qpos_spring_id = worldid % qpos_spring.shape[0]

    if jnttype == JointType.FREE:
      if plasticadr >= 0:
        ref_pos = wp.vec3(
          plastic_in[worldid, plasticadr + 0],
          plastic_in[worldid, plasticadr + 1],
          plastic_in[worldid, plasticadr + 2],
        )
        quat_spring = wp.quat(
          plastic_in[worldid, plasticadr + 3],
          plastic_in[worldid, plasticadr + 4],
          plastic_in[worldid, plasticadr + 5],
          plastic_in[worldid, plasticadr + 6],
        )
      else:
        ref_pos = wp.vec3(
          qpos_spring[qpos_spring_id, padr + 0],
          qpos_spring[qpos_spring_id, padr + 1],
          qpos_spring[qpos_spring_id, padr + 2],
        )
        quat_spring = wp.quat(
          qpos_spring[qpos_spring_id, padr + 3],
          qpos_spring[qpos_spring_id, padr + 4],
          qpos_spring[qpos_spring_id, padr + 5],
          qpos_spring[qpos_spring_id, padr + 6],
        )

      dif0 = wp.vec3(
        qpos_in[worldid, padr + 0] - ref_pos[0],
        qpos_in[worldid, padr + 1] - ref_pos[1],
        qpos_in[worldid, padr + 2] - ref_pos[2],
      )

      # convert quaternion difference into angular "velocity"
      quat1 = wp.quat(
        qpos_in[worldid, padr + 3],
        qpos_in[worldid, padr + 4],
        qpos_in[worldid, padr + 5],
        qpos_in[worldid, padr + 6],
      )
      quat1 = wp.normalize(quat1)

      dif1 = math.quat_sub(quat1, quat_spring)

      r0 = wp.length(dif0)
      r1 = wp.length(dif1)

      energy = wp.vec2(
        poly_potential(stiffness, spoly, r0, 0) + poly_potential(stiffness, spoly, r1, 0),
        0.0,
      )

      wp.atomic_add(energy_out, worldid, energy)

    elif jnttype == JointType.BALL:
      quat = wp.quat(
        qpos_in[worldid, padr + 0],
        qpos_in[worldid, padr + 1],
        qpos_in[worldid, padr + 2],
        qpos_in[worldid, padr + 3],
      )
      quat = wp.normalize(quat)

      if plasticadr >= 0:
        quat_spring = wp.quat(
          plastic_in[worldid, plasticadr + 0],
          plastic_in[worldid, plasticadr + 1],
          plastic_in[worldid, plasticadr + 2],
          plastic_in[worldid, plasticadr + 3],
        )
      else:
        quat_spring = wp.quat(
          qpos_spring[qpos_spring_id, padr + 0],
          qpos_spring[qpos_spring_id, padr + 1],
          qpos_spring[qpos_spring_id, padr + 2],
          qpos_spring[qpos_spring_id, padr + 3],
        )

      dif = math.quat_sub(quat, quat_spring)
      r = wp.length(dif)
      energy = wp.vec2(
        poly_potential(stiffness, spoly, r, 0),
        0.0,
      )
      wp.atomic_add(energy_out, worldid, energy)
    elif jnttype == JointType.SLIDE or jnttype == JointType.HINGE:
      ref_scalar = plastic_in[worldid, plasticadr] if plasticadr >= 0 else qpos_spring[qpos_spring_id, padr]
      dif_ = qpos_in[worldid, padr] - ref_scalar
      energy = wp.vec2(
        poly_potential(stiffness, spoly, dif_, 0),
        0.0,
      )
      wp.atomic_add(energy_out, worldid, energy)

  return kernel


@cache_kernel
def _energy_pos_passive_tendon(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    tendon_plasticadr: wp.array[int],
    tendon_stiffness: wp.array2d[float],
    tendon_stiffnesspoly: wp.array2d[wp.vec2],
    tendon_lengthspring: wp.array2d[wp.vec2],
    # Data in:
    plastic_in: wp.array2d[float],
    ten_length_in: wp.array2d[float],
    # Data out:
    energy_out: wp.array[wp.vec2],
  ):
    worldid, tenid = wp.tid()

    tendon_stiffness_id = worldid % tendon_stiffness.shape[0]
    stiffness = tendon_stiffness[tendon_stiffness_id, tenid]
    spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]

    if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:
      return

    length = ten_length_in[worldid, tenid]

    # compute spring displacement
    padr = tendon_plasticadr[tenid]
    if padr >= 0:
      lower = plastic_in[worldid, padr + 0]
      upper = plastic_in[worldid, padr + 1]
    else:
      tendon_lengthspring_id = worldid % tendon_lengthspring.shape[0]
      lengthspring = tendon_lengthspring[tendon_lengthspring_id, tenid]
      lower = lengthspring[0]
      upper = lengthspring[1]

    if length > upper:
      x = length - upper
    elif length < lower:
      x = length - lower
    else:
      x = 0.0

    energy = wp.vec2(poly_potential(stiffness, spoly, x, 0), 0.0)
    wp.atomic_add(energy_out, worldid, energy)

  return kernel


@cache_kernel
def _energy_pos_passive_flex(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    flex_interp: wp.array[int],
    flexedge_length0: wp.array[float],
    flex_edgestiffness: wp.array[float],
    flexedge_rigid: wp.array[bool],
    flex_edgeflexid: wp.array[int],
    # Data in:
    flexedge_length_in: wp.array2d[float],
    # Data out:
    energy_out: wp.array[wp.vec2],
  ):
    worldid, edgeid = wp.tid()

    if flexedge_rigid[edgeid]:
      return

    f = flex_edgeflexid[edgeid]
    if flex_interp[f] != 0:
      return

    stiffness = flex_edgestiffness[f]
    if stiffness == 0.0:
      return

    displacement = flexedge_length0[edgeid] - flexedge_length_in[worldid, edgeid]
    if displacement == 0.0:
      return

    energy = wp.vec2(0.5 * stiffness * displacement * displacement, 0.0)
    wp.atomic_add(energy_out, worldid, energy)

  return kernel


def energy_pos(m: Model, d: Data):
  """Position-dependent energy (potential)."""
  wp.launch(_energy_pos_zero, dim=d.nworld, outputs=[d.energy])

  # init potential energy: -sum_i(body_i.mass * dot(gravity, body_i.pos))
  if not (m.opt.disableflags & DisableBit.GRAVITY):
    wp.launch(
      _energy_pos_gravity(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.nbody - 1),
      inputs=[m.opt.gravity, m.body_mass, d.xipos],
      outputs=[d.energy],
    )

  if not (m.opt.disableflags & DisableBit.SPRING):
    # add joint-level springs
    wp.launch(
      _energy_pos_passive_joint(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.njnt),
      inputs=[
        m.qpos_spring,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_plasticadr,
        m.jnt_stiffness,
        m.jnt_stiffnesspoly,
        d.qpos,
        d.plastic,
      ],
      outputs=[d.energy],
    )

    # add tendon-level springs
    if m.ntendon:
      wp.launch(
        _energy_pos_passive_tendon(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
        dim=(d.nworld, m.ntendon),
        inputs=[
          m.tendon_plasticadr,
          m.tendon_stiffness,
          m.tendon_stiffnesspoly,
          m.tendon_lengthspring,
          d.plastic,
          d.ten_length,
        ],
        outputs=[d.energy],
      )

    # add flex-level edge springs
    wp.launch(
      _energy_pos_passive_flex(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.flex_interp,
        m.flexedge_length0,
        m.flex_edgestiffness,
        m.flexedge_rigid,
        m.flex_edgeflexid,
        d.flexedge_length,
      ],
      outputs=[d.energy],
    )


@cache_kernel
def _energy_vel_kinetic(nv: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def energy_vel_kinetic(
    # Data in:
    qvel_in: wp.array2d[float],
    # In:
    Mqvel: wp.array2d[float],
    # Data out:
    energy_out: wp.array[wp.vec2],
  ):
    worldid = wp.tid()

    qvel_tile = wp.tile_load(qvel_in[worldid], shape=wp.static(nv))
    Mqvel_tile = wp.tile_load(Mqvel[worldid], shape=wp.static(nv))

    # qvel * (M @ qvel)
    qvelMqvel_tile = wp.tile_map(wp.mul, qvel_tile, Mqvel_tile)

    # sum(qvel * (M @ qvel))
    quadratic_tile = wp.tile_reduce(wp.add, qvelMqvel_tile)

    energy_out[worldid][1] = 0.5 * quadratic_tile[0]

  return energy_vel_kinetic


def energy_vel(m: Model, d: Data):
  """Velocity-dependent energy (kinetic)."""
  # kinetic energy: 0.5 * qvel.T @ M @ qvel

  # M @ qvel
  mv = wp.zeros((d.nworld, m.nv), dtype=float)
  support.mul_m(m, d, mv, d.qvel)

  wp.launch_tiled(
    _energy_vel_kinetic(m.nv),
    dim=d.nworld,
    inputs=[d.qvel, mv],
    outputs=[d.energy],
    block_dim=m.block_dim.energy_vel_kinetic,
  )
