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
"""Sensor functions."""

import jax
from jax import numpy as jp
import mujoco

# pylint: disable=g-importing-member
from mujoco.mjx._src import math
from mujoco.mjx._src import ray
from mujoco.mjx._src import smooth
from mujoco.mjx._src import support
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DataJAX
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ModelJAX
from mujoco.mjx._src.types import ObjType
from mujoco.mjx._src.types import SensorType
from mujoco.mjx._src.types import TrnType

# pylint: enable=g-importing-member
import numpy as np


def _apply_cutoff(
    sensor: jax.Array, cutoff: jax.Array, data_type: int
) -> jax.Array:
  """Clip sensor to cutoff value."""

  @jax.vmap
  def fn(sensor, cutoff):
    if data_type == mujoco.mjtDataType.mjDATATYPE_REAL:
      return jp.where(cutoff > 0, jp.clip(sensor, -cutoff, cutoff), sensor)
    elif data_type == mujoco.mjtDataType.mjDATATYPE_POSITIVE:
      return jp.where(cutoff > 0, jp.minimum(sensor, cutoff), sensor)
    else:
      return sensor

  return fn(sensor, cutoff)


def _sensor_contact(
    m: Model,
    d: Data,
    idx: np.ndarray,
    contact_force: jax.Array | None = None,
) -> tuple[list[jax.Array], list[np.ndarray]]:
  """Compute contact sensors matching idx mask."""
  sensors, adrs = [], []
  objid = m.sensor_objid[idx]
  adr = m.sensor_adr[idx]

  # maximum number of contacts
  ncon = d._impl.ncon  # pyrefly: ignore[missing-attribute]

  # active contacts
  dist = d._impl.contact.dist  # pyrefly: ignore[missing-attribute]
  pos = dist - d._impl.contact.includemargin  # pyrefly: ignore[missing-attribute]
  is_contact = pos < 0

  contact_intprm = m.sensor_intprm[idx]
  contact_maxforce = (contact_intprm[:, 1] == 2).any()

  # reduction criteria
  if contact_maxforce:
    # compute force magnitude for each contact
    force_mag = jax.vmap(
        lambda forcetorque: jp.dot(forcetorque[:3], forcetorque[:3])
    )(
        contact_force
    )  # pyrefly: ignore[unbound-name]

  def _reduce(reduction, mask):
    if reduction == 1:  # mindist
      return jp.argsort(pos * mask, descending=False)
    if reduction == 2:  # maxforce
      return jp.argsort(force_mag * mask, descending=True)  # pyrefly: ignore[unbound-name]
    return jp.arange(mask.size)

  # number of data elements per slot
  def nslotdata(dataspec):
    size = 0
    # found, force, torque, dist, pos, normal, tangent, linvel, angvel
    # TODO(taylorhowell): get sizes from mjCONDATA_SIZE
    for i, size_i in enumerate([1, 3, 3, 1, 3, 3, 3, 3, 3]):
      if dataspec & (1 << i):
        size += size_i
    return size

  dataspecs, reduces, _ = m.sensor_intprm[idx].T
  dims = m.sensor_dim[idx]
  objtypes = m.sensor_objtype[idx]
  refid = m.sensor_refid[idx]
  reftypes = m.sensor_reftype[idx]

  for dataspec, reduce, objtype, reftype, dim in set(
      zip(dataspecs, reduces, objtypes, reftypes, dims)
  ):
    idx_ds = (
        (dataspec == dataspecs)
        & (reduce == reduces)
        & (objtype == objtypes)
        & (reftype == reftypes)
        & (dim == dims)
    )

    # TODO(taylorhowell): site filter

    size = nslotdata(dataspec)
    num = np.minimum(int(dim / size), ncon)
    nsensor = idx_ds.sum()

    if objtype == ObjType.UNKNOWN and reftype == ObjType.UNKNOWN:
      # all contacts match
      match = np.ones(ncon, dtype=np.bool)

      # matched and reduced contact ids
      sort = _reduce(reduce, match)
      cid = sort[:num]

      # number of contacts per sensor
      nfound = sum(is_contact)

      # if duplicate sensor
      cid = jp.tile(cid, (nsensor,))
      nfound = jp.tile(nfound, (nsensor,))
      flip = jp.ones((cid.size, 3))
    elif objtype == ObjType.GEOM or reftype == ObjType.GEOM:
      sensorid1 = objid[idx_ds]
      sensorid2 = refid[idx_ds]
      geomid0 = d._impl.contact.geom[:, 0]  # pyrefly: ignore[missing-attribute]
      geomid1 = d._impl.contact.geom[:, 1]  # pyrefly: ignore[missing-attribute]

      # match sensor ids and contact geom ids
      geom0id1 = geomid0 == sensorid1[:, None]
      geom0id2 = geomid0 == sensorid2[:, None]
      geom1id1 = geomid1 == sensorid1[:, None]
      geom1id2 = geomid1 == sensorid2[:, None]

      if objtype == ObjType.GEOM and reftype == ObjType.UNKNOWN:  # geom1
        mask12 = geom0id1
        mask21 = geom1id1
      elif objtype == ObjType.UNKNOWN and reftype == ObjType.GEOM:  # geom2
        mask12 = geom0id2
        mask21 = geom1id2
      else:  # geom1, geom2
        mask12 = geom0id1 & geom1id2
        mask21 = geom0id2 & geom1id1

      match = mask12 | mask21

      # matched and reduced contact ids
      cid = jax.vmap(lambda x: _reduce(reduce, x))(match)[:, :num]
      cid = cid.reshape(-1)

      # flip direction for force, torque, normal, tangent
      if reftype == ObjType.UNKNOWN:  # geom1
        is_flip = (geomid1[cid] == np.repeat(sensorid1, num))[:, None]
      elif objtype == ObjType.UNKNOWN:  # geom2
        is_flip = (geomid0[cid] == np.repeat(sensorid2, num))[:, None]
      else:  # geom1, geom2
        is_flip = np.repeat(sensorid1 > sensorid2, num)[:, None]

      flip = jp.where(
          is_flip,
          jp.array([[1, 1, -1]]),
          jp.array([[1, 1, 1]]),
      )

      # number of contacts per sensor
      nfound = (match * is_contact[None, :]).sum(axis=1)

    # TODO(taylorhowell): matching criteria: body, subtree

    else:
      raise NotImplementedError(
          f'Unsupported contact sensor semantics: {objtype} {reftype}.'
      )

    slot = []

    if dataspec & (1 << 0):  # found
      slot.append(jp.repeat(nfound, num)[:, None])

    if dataspec & (1 << 1):  # force
      slot.append(flip * contact_force[cid, :3])  # pyrefly: ignore[index-error, unsupported-operation]

    if dataspec & (1 << 2):  # torque
      slot.append(flip * contact_force[cid, 3:])  # pyrefly: ignore[index-error, unsupported-operation]

    if dataspec & (1 << 3):  # dist
      slot.append(dist[cid, None])

    if dataspec & (1 << 4):  # pos
      slot.append(d._impl.contact.pos[cid])  # pyrefly: ignore[missing-attribute]

    if dataspec & (1 << 5):  # normal
      slot.append(flip[:, 2, None] * d._impl.contact.frame[cid, 0])  # pyrefly: ignore[missing-attribute]

    if dataspec & (1 << 6):  # tangent
      slot.append(flip[:, 2, None] * d._impl.contact.frame[cid, 1])  # pyrefly: ignore[missing-attribute]

    if dataspec & ((1 << 7) | (1 << 8)):  # linvel or angvel
      con_pos = d._impl.contact.pos[cid]  # pyrefly: ignore[missing-attribute]
      con_frame = d._impl.contact.frame[cid]  # pyrefly: ignore[missing-attribute]
      con_geom = d._impl.contact.geom[cid]  # pyrefly: ignore[missing-attribute]
      bid0 = jp.array(m.geom_bodyid)[con_geom[:, 0]]
      bid1 = jp.array(m.geom_bodyid)[con_geom[:, 1]]
      cvel0 = d.cvel[bid0]
      cvel1 = d.cvel[bid1]

      if dataspec & (1 << 7):  # linvel
        root0 = jp.array(m.body_rootid)[bid0]
        root1 = jp.array(m.body_rootid)[bid1]
        v0 = cvel0[:, 3:] - jp.cross(con_pos - d.subtree_com[root0], cvel0[:, :3])
        v1 = cvel1[:, 3:] - jp.cross(con_pos - d.subtree_com[root1], cvel1[:, :3])
        linvel = jax.vmap(lambda f, v: f @ v)(con_frame, v1 - v0)
        slot.append(flip * linvel)

      if dataspec & (1 << 8):  # angvel
        angvel = jax.vmap(lambda f, w: f @ w)(
            con_frame, cvel1[:, :3] - cvel0[:, :3]
        )
        slot.append(flip * angvel)

    found = jp.tile(jp.arange(num), nsensor) < jp.repeat(nfound, num)
    sensors.append((found[:, None] * jp.hstack(slot)).reshape(-1))
    adrs.append(
        (adr[idx_ds][:, None] + np.arange(num * size)[None]).reshape(-1)
    )

  return sensors, adrs


def sensor_pos(m: Model, d: Data) -> Data:
  """Compute position-dependent sensors values."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('sensor_pos requires JAX backend implementation.')

  if m.opt.disableflags & DisableBit.SENSOR:
    return d

  # position and orientation by object type
  objtype_data = {
      int(ObjType.UNKNOWN): (
          np.zeros((1, 3)),
          np.expand_dims(np.eye(3), axis=0),
      ),  # world
      int(ObjType.BODY): (d.xipos, d.ximat),
      int(ObjType.XBODY): (d.xpos, d.xmat),
      int(ObjType.GEOM): (d.geom_xpos, d.geom_xmat),
      int(ObjType.SITE): (d.site_xpos, d.site_xmat),
      int(ObjType.CAMERA): (d.cam_xpos, d.cam_xmat),
  }

  # frame axis indexing
  frame_axis = {
      int(SensorType.FRAMEXAXIS): 0,
      int(SensorType.FRAMEYAXIS): 1,
      int(SensorType.FRAMEZAXIS): 2,
  }

  stage_pos = m.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS
  sensors, adrs = [], []

  for sensor_type in set(m.sensor_type[stage_pos]):
    idx = (m.sensor_type == sensor_type) & stage_pos
    objid = m.sensor_objid[idx]
    objtype = m.sensor_objtype[idx]
    refid = m.sensor_refid[idx]
    reftype = m.sensor_reftype[idx]
    adr = m.sensor_adr[idx]
    cutoff = m.sensor_cutoff[idx]
    data_type = m.sensor_datatype[idx]

    if sensor_type == SensorType.MAGNETOMETER:
      sensor = jax.vmap(lambda xmat: xmat.T @ m.opt.magnetic)(
          d.site_xmat[objid]
      )
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.CAMPROJECTION:

      @jax.vmap
      def _cam_project(
          target_xpos, xpos, xmat, res, fovy, intrinsic, sensorsize, focal_flag
      ):
        translation = jp.eye(4).at[0:3, 3].set(-xpos)
        rotation = jp.eye(4).at[:3, :3].set(xmat.T)

        # focal transformation matrix (3 x 4)
        f = 0.5 / jp.tan(fovy * jp.pi / 360.0) * res[1]
        fx, fy = jp.where(
            focal_flag,
            intrinsic[:2] / (sensorsize[:2] + mujoco.mjMINVAL) * res[:2],
            f,
        )  # add mjMINVAL to denominator to prevent divide by zero warning

        focal = jp.array([[-fx, 0, 0, 0], [0, fy, 0, 0], [0, 0, 1.0, 0]])

        # image matrix (3 x 3)
        image = jp.eye(3).at[:2, 2].set(res[0:2] / 2.0)

        # projection matrix (3 x 4): product of all 4 matrices
        proj = image @ focal @ rotation @ translation

        # projection matrix multiplies homogenous [x, y, z, 1] vectors
        pos_hom = jp.append(target_xpos, 1.0)

        # project world coordinates into pixel space, see:
        # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
        pixel_coord_hom = proj @ pos_hom

        # avoid dividing by tiny numbers
        denom = pixel_coord_hom[2]
        denom = jp.where(
            jp.abs(denom) < mujoco.mjMINVAL,
            jp.clip(denom, -mujoco.mjMINVAL, mujoco.mjMINVAL),
            denom,
        )

        # compute projection
        sensor = pixel_coord_hom / denom

        return sensor[:2]

      sensorsize = m.cam_sensorsize[refid]
      intrinsic = m.cam_intrinsic[refid]
      fovy = m.cam_fovy[refid]
      res = m.cam_resolution[refid]
      focal_flag = np.logical_and(sensorsize[:, 0] != 0, sensorsize[:, 1] != 0)

      target_xpos = d.site_xpos[objid]
      xpos = d.cam_xpos[refid]
      xmat = d.cam_xmat[refid]

      sensor = _cam_project(
          target_xpos, xpos, xmat, res, fovy, intrinsic, sensorsize, focal_flag
      )
      adr = (adr[:, None] + np.arange(2)[None]).reshape(-1)
    elif sensor_type == SensorType.RANGEFINDER:
      site_bodyid = m.site_bodyid[objid]
      for sid in set(site_bodyid):
        idxs = sid == site_bodyid
        objids = objid[idxs]
        site_xpos = d.site_xpos[objids]
        site_mat = d.site_xmat[objids].reshape((-1, 9))[:, np.array([2, 5, 8])]
        cutoffs = cutoff[idxs]
        sensor, _ = jax.vmap(
            ray.ray, in_axes=(None, None, 0, 0, None, None, None)
        )(m, d, site_xpos, site_mat, (), True, sid)
        sensors.append(_apply_cutoff(sensor, cutoffs, data_type[0]))
        adrs.append(adr[idxs])
      continue  # avoid adding to sensors/adrs list a second time
    elif sensor_type == SensorType.JOINTPOS:
      sensor = d.qpos[m.jnt_qposadr[objid]]
    elif sensor_type == SensorType.TENDONPOS:
      sensor = d.ten_length[objid]
    elif sensor_type == SensorType.ACTUATORPOS:
      sensor = d.actuator_length[objid]
    elif sensor_type == SensorType.BALLQUAT:
      jnt_qposadr = m.jnt_qposadr[objid, None] + np.arange(4)[None]
      quat = d.qpos[jnt_qposadr]
      sensor = jax.vmap(math.normalize)(quat)
      adr = (adr[:, None] + np.arange(4)[None]).reshape(-1)
    elif sensor_type == SensorType.FRAMEPOS:

      def _framepos(xpos, xpos_ref, xmat_ref, refid):
        return jp.where(refid == -1, xpos, xmat_ref.T @ (xpos - xpos_ref))

      # evaluate for valid object and reference object type pairs
      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        refidt = refid[idxt]
        xpos, _ = objtype_data[int(ot)]
        xpos_ref, xmat_ref = objtype_data[int(rt)]
        xpos = xpos[objid[idxt]]
        xpos_ref = xpos_ref[refidt]
        xmat_ref = xmat_ref[refidt]
        cutofft = cutoff[idxt]
        sensor = jax.vmap(_framepos)(xpos, xpos_ref, xmat_ref, refidt)
        adrt = adr[idxt, None] + np.arange(3)[None]
        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue  # avoid adding to sensors/adrs list a second time
    elif sensor_type in frame_axis:

      def _frameaxis(xmat, xmat_ref, refid):
        axis = xmat[:, frame_axis[sensor_type]]
        return jp.where(refid == -1, axis, xmat_ref.T @ axis)

      # evaluate for valid object and reference object type pairs
      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        refidt = refid[idxt]
        _, xmat = objtype_data[int(ot)]
        _, xmat_ref = objtype_data[int(rt)]
        xmat = xmat[objid[idxt]]
        xmat_ref = xmat_ref[refidt]
        cutofft = cutoff[idxt]
        sensor = jax.vmap(_frameaxis)(xmat, xmat_ref, refidt)
        adrt = adr[idxt, None] + np.arange(3)[None]
        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue  # avoid adding to sensors/adrs list a second time
    elif sensor_type == SensorType.FRAMEQUAT:

      def _quat(otype, oid):
        otype = int(otype)
        if otype == ObjType.XBODY:
          return d.xquat[oid]
        elif otype == ObjType.BODY:
          return jax.vmap(math.quat_mul)(d.xquat[oid], m.body_iquat[oid])
        elif otype == ObjType.GEOM:
          return jax.vmap(math.quat_mul)(
              d.xquat[m.geom_bodyid[oid]], m.geom_quat[oid]
          )
        elif otype == ObjType.SITE:
          return jax.vmap(math.quat_mul)(
              d.xquat[m.site_bodyid[oid]], m.site_quat[oid]
          )
        elif otype == ObjType.CAMERA:
          return jax.vmap(math.quat_mul)(
              d.xquat[m.cam_bodyid[oid]], m.cam_quat[oid]
          )
        elif otype == ObjType.UNKNOWN:
          return jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), (oid.size, 1))
        else:
          raise ValueError(f'Unknown object type: {otype}')

      # evaluate for valid object and reference object type pairs
      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        objidt = objid[idxt]
        refidt = refid[idxt]
        quat = _quat(ot, objidt)
        refquat = _quat(rt, refidt)
        cutofft = cutoff[idxt]
        sensor = jax.vmap(
            lambda q, r, rid: jp.where(
                rid == -1, q, math.quat_mul(math.quat_inv(r), q)
            )
        )(quat, refquat, refidt)
        adrt = adr[idxt, None] + np.arange(4)[None]
        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue  # avoid adding to sensors/adrs list a second time
    elif sensor_type == SensorType.SUBTREECOM:
      sensor = d.subtree_com[objid]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.CLOCK:
      sensor = jp.repeat(d.time, sum(idx))
    elif sensor_type == SensorType.CONTACT:
      s, a = _sensor_contact(m, d, idx)
      sensors.extend(s)
      adrs.extend(a)
      continue
    else:
      # TODO(taylorhowell): raise error after adding sensor check to io.py
      continue  # unsupported sensor type

    sensors.append(_apply_cutoff(sensor, cutoff, data_type[0]).reshape(-1))
    adrs.append(adr)

  if not adrs:
    return d

  sensordata = d.sensordata.at[np.concatenate(adrs)].set(
      jp.concatenate(sensors)
  )

  return d.replace(sensordata=sensordata)


def sensor_vel(m: Model, d: Data) -> Data:
  """Compute velocity-dependent sensors values."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('sensor_vel requires JAX backend implementation.')

  if m.opt.disableflags & DisableBit.SENSOR:
    return d

  # position and orientation by object type
  objtype_data = {
      int(ObjType.UNKNOWN): (
          np.zeros((1, 3)),
          np.expand_dims(np.eye(3), axis=0),
          np.arange(1),
      ),  # world
      int(ObjType.BODY): (d.xipos, d.ximat, np.arange(m.nbody)),
      int(ObjType.XBODY): (d.xpos, d.xmat, np.arange(m.nbody)),
      int(ObjType.GEOM): (d.geom_xpos, d.geom_xmat, m.geom_bodyid),
      int(ObjType.SITE): (d.site_xpos, d.site_xmat, m.site_bodyid),
      int(ObjType.CAMERA): (d.cam_xpos, d.cam_xmat, m.cam_bodyid),
  }

  stage_vel = m.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL
  sensor_types = set(m.sensor_type[stage_vel])

  if sensor_types & {SensorType.SUBTREELINVEL, SensorType.SUBTREEANGMOM}:
    d = smooth.subtree_vel(m, d)

  sensors, adrs = [], []
  for sensor_type in sensor_types:
    idx = (m.sensor_type == sensor_type) & stage_vel
    objid = m.sensor_objid[idx]
    adr = m.sensor_adr[idx]
    cutoff = m.sensor_cutoff[idx]
    data_type = m.sensor_datatype[idx]

    if sensor_type == SensorType.VELOCIMETER:
      bodyid = m.site_bodyid[objid]
      pos = d.site_xpos[objid]
      rot = d.site_xmat[objid]
      cvel = d.cvel[bodyid]
      subtree_com = d.subtree_com[m.body_rootid[bodyid]]
      sensor = jax.vmap(
          lambda vec, dif, rot: rot.T @ (vec[3:] - jp.cross(dif, vec[:3]))
      )(cvel, pos - subtree_com, rot)
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.GYRO:
      bodyid = m.site_bodyid[objid]
      rot = d.site_xmat[objid]
      ang = d.cvel[bodyid, :3]
      sensor = jax.vmap(lambda ang, rot: rot.T @ ang)(ang, rot)
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.JOINTVEL:
      sensor = d.qvel[m.jnt_dofadr[objid]]
    elif sensor_type == SensorType.TENDONVEL:
      sensor = d._impl.ten_velocity[objid]  # pyrefly: ignore[missing-attribute]
    elif sensor_type == SensorType.ACTUATORVEL:
      sensor = d._impl.actuator_velocity[objid]  # pyrefly: ignore[missing-attribute]
    elif sensor_type == SensorType.BALLANGVEL:
      jnt_dotadr = m.jnt_dofadr[objid, None] + np.arange(3)[None]
      sensor = d.qvel[jnt_dotadr]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type in {SensorType.FRAMELINVEL, SensorType.FRAMEANGVEL}:
      objtype = m.sensor_objtype[idx]
      reftype = m.sensor_reftype[idx]
      refid = m.sensor_refid[idx]

      # evaluate for valid object and reference object type pairs
      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        objidt = objid[idxt]
        refidt = refid[idxt]
        cutofft = cutoff[idxt]

        xpos, _, _ = objtype_data[int(ot)]
        xposref, xmatref, _ = objtype_data[int(rt)]
        xpos = xpos[objidt]
        xposref = xposref[refidt]
        xmatref = xmatref[refidt]

        def _cvel_offset(otype, oid):
          pos, _, bodyid = objtype_data[int(otype)]
          pos = pos[oid]
          bodyid = bodyid[oid]
          return d.cvel[bodyid], pos - d.subtree_com[m.body_rootid[bodyid]]

        cvel, offset = _cvel_offset(ot, objidt)
        cvelref, offsetref = _cvel_offset(rt, refidt)
        cangvel = cvel[:, :3]
        cangvelref = cvelref[:, :3]

        if sensor_type == SensorType.FRAMELINVEL:
          clinvel = cvel[:, 3:]
          clinvelref = cvelref[:, 3:]
          xlinvel = clinvel - jp.cross(offset, cangvel)
          xlinvelref = clinvelref - jp.cross(offsetref, cangvelref)
          rvec = xpos - xposref
          rel_vel = xlinvel - xlinvelref + jp.cross(rvec, cangvelref)
          sensor = jp.where(
              (refidt > -1)[:, None],
              jax.vmap(lambda mat, vec: mat.T @ vec)(xmatref, rel_vel),
              xlinvel,
          )
        elif sensor_type == SensorType.FRAMEANGVEL:
          rel_vel = cangvel - cangvelref
          sensor = jp.where(
              (refidt > -1)[:, None],
              jax.vmap(lambda mat, vec: mat.T @ vec)(xmatref, rel_vel),
              cangvel,
          )
        else:
          raise ValueError(f'Unknown sensor type: {sensor_type}')

        adrt = adr[idxt, None] + np.arange(3)[None]

        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue  # avoid adding to sensors/adrs list a second time
    elif sensor_type == SensorType.SUBTREELINVEL:
      sensor = d._impl.subtree_linvel[objid]  # pyrefly: ignore[missing-attribute]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.SUBTREEANGMOM:
      sensor = d._impl.subtree_angmom[objid]  # pyrefly: ignore[missing-attribute]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.CONTACT:
      s, a = _sensor_contact(m, d, idx)
      sensors.extend(s)
      adrs.extend(a)
      continue
    else:
      # TODO(taylorhowell): raise error after adding sensor check to io.py
      continue  # unsupported sensor type

    sensors.append(_apply_cutoff(sensor, cutoff, data_type[0]).reshape(-1))
    adrs.append(adr)

  if not adrs:
    return d

  sensordata = d.sensordata.at[np.concatenate(adrs)].set(
      jp.concatenate(sensors)
  )

  return d.replace(sensordata=sensordata)


def sensor_acc(m: Model, d: Data) -> Data:
  """Compute acceleration/force-dependent sensors values."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('sensor_acc requires JAX backend implementation.')

  if m.opt.disableflags & DisableBit.SENSOR:
    return d

  # position and bodyid by object type
  objtype_data = {
      int(ObjType.UNKNOWN): (np.zeros((1, 3)), np.arange(1)),
      int(ObjType.BODY): (d.xipos, np.arange(m.nbody)),
      int(ObjType.XBODY): (d.xpos, np.arange(m.nbody)),
      int(ObjType.GEOM): (d.geom_xpos, m.geom_bodyid),
      int(ObjType.SITE): (d.site_xpos, m.site_bodyid),
      int(ObjType.CAMERA): (d.cam_xpos, m.cam_bodyid),
  }

  stage_acc = m.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC
  sensor_types = {int(st) for st in set(m.sensor_type[stage_acc])}

  if sensor_types & {
      int(SensorType.ACCELEROMETER),
      int(SensorType.FORCE),
      int(SensorType.TORQUE),
      int(SensorType.FRAMELINACC),
      int(SensorType.FRAMEANGACC),
  }:
    d = smooth.rne_postconstraint(m, d)

  contact_acc = (m.sensor_type == SensorType.CONTACT) & stage_acc
  contact_intprm = m.sensor_intprm[contact_acc]
  contact_maxforce = (contact_intprm[:, 1] == 2).any()
  contact_dataforce = (contact_intprm[:, 0] & (1 << 1)).any()
  contact_datatorque = (contact_intprm[:, 0] & (1 << 2)).any()
  if (m.sensor_type[stage_acc] == int(SensorType.TOUCH)).any() | (
      contact_acc.any()
      and (contact_maxforce | contact_dataforce | contact_datatorque)
  ):
    # compute contact forces
    contact_force = []
    condim_ids = []
    for dim in set(d._impl.contact.dim):  # pyrefly: ignore[missing-attribute]
      force, condim_id = support.contact_force_dim(m, d, dim)
      contact_force.append(force)
      condim_ids.append(condim_id)
    contact_force = jp.concatenate(contact_force)[
        np.argsort(np.concatenate(condim_ids))
    ]

  sensors, adrs = [], []

  for sensor_type in sensor_types:
    idx = (m.sensor_type == sensor_type) & stage_acc
    objid = m.sensor_objid[idx]
    adr = m.sensor_adr[idx]
    cutoff = m.sensor_cutoff[idx]
    data_type = m.sensor_datatype[idx]

    if sensor_type == SensorType.TOUCH:
      # get bodies of contact geoms
      conbody = jp.array(m.geom_bodyid)[d._impl.contact.geom]  # pyrefly: ignore[missing-attribute]

      # get site information
      site_bodyid = m.site_bodyid[objid]
      site_size = m.site_size[objid]
      site_xpos = d.site_xpos[objid]
      site_xmat = d.site_xmat[objid]
      site_type = m.site_type[objid]
      conbody0 = site_bodyid[:, None] == conbody[:, 0]
      conbody1 = site_bodyid[:, None] == conbody[:, 1]
      contacts = (d._impl.contact.efc_address >= 0)[None] & (  # pyrefly: ignore[missing-attribute]
          conbody0 | conbody1
      )

      # compute conray, flip if second body
      conray = jax.vmap(
          lambda frame, force: math.normalize(frame[0] * force[0])
      )(
          d._impl.contact.frame, contact_force  # pyrefly: ignore[missing-attribute, unbound-name]
      )
      conray = jp.where(conbody1[..., None], -conray, conray)

      # compute distance, mapping over sites and contacts
      def _distance(site_size, site_xpos, site_xmat, site_type, pos, conray):
        def dist(size, xpos, xmat, conray):
          # Avoid matmul here: its batching rule can reorder nested vmap axes.
          pnt = jp.sum((pos - xpos)[..., :, None] * xmat, axis=-2)
          vec = jp.sum(conray[..., :, None] * xmat, axis=-2)
          ray_geom_ = lambda pnt, vec: ray.ray_geom(size, pnt, vec, site_type)
          return jax.vmap(ray_geom_)(pnt, vec)

        return jax.vmap(dist)(site_size, site_xpos, site_xmat, conray)

      dist = []
      dist_id = []
      for st in set(site_type):
        (dist_id_site,) = np.nonzero(st == site_type)
        dist_site = _distance(
            site_size[dist_id_site],
            site_xpos[dist_id_site],
            site_xmat[dist_id_site],
            st,
            d._impl.contact.pos,  # pyrefly: ignore[missing-attribute]
            conray[dist_id_site],
        )
        dist.append(jp.where(jp.isinf(dist_site), 0, dist_site))
        dist_id.append(dist_id_site)
      dist = jp.vstack(dist)[np.argsort(np.concatenate(dist_id))]

      # accumulate normal forces for each site
      sensor = jp.dot((dist > 0) & contacts, contact_force[:, 0])  # pyrefly: ignore[index-error]
    elif sensor_type == SensorType.CONTACT:
      s, a = _sensor_contact(m, d, idx, contact_force)  # pyrefly: ignore[unbound-name]
      sensors.extend(s)
      adrs.extend(a)
      continue

    elif sensor_type == SensorType.ACCELEROMETER:

      @jax.vmap
      def _accelerometer(cvel, cacc, diff, rot):
        ang = rot.T @ cvel[:3]
        lin = rot.T @ (cvel[3:] - jp.cross(diff, cvel[:3]))
        acc = rot.T @ (cacc[3:] - jp.cross(diff, cacc[:3]))
        correction = jp.cross(ang, lin)
        return acc + correction

      bodyid = m.site_bodyid[objid]
      rot = d.site_xmat[objid]
      cvel = d.cvel[bodyid]
      cacc = d._impl.cacc[bodyid]  # pyrefly: ignore[missing-attribute]
      dif = d.site_xpos[objid] - d.subtree_com[m.body_rootid[bodyid]]

      sensor = _accelerometer(cvel, cacc, dif, rot)
      # dof-less body (static or mocap): match mj_objectAcceleration
      dofless = m.body_dofnum[m.body_weldid[bodyid]] == 0
      sensor = jp.where(dofless[:, None], 0.0, sensor)
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.FORCE:
      bodyid = m.site_bodyid[objid]
      cfrc_int = d._impl.cfrc_int[bodyid]  # pyrefly: ignore[missing-attribute]
      site_xmat = d.site_xmat[objid]
      sensor = jax.vmap(lambda mat, vec: mat.T @ vec)(
          site_xmat, cfrc_int[:, 3:]
      )
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.TORQUE:
      bodyid = m.site_bodyid[objid]
      rootid = m.body_rootid[bodyid]
      cfrc_int = d._impl.cfrc_int[bodyid]  # pyrefly: ignore[missing-attribute]
      site_xmat = d.site_xmat[objid]
      dif = d.site_xpos[objid] - d.subtree_com[rootid]
      sensor = jax.vmap(
          lambda vec, dif, rot: rot.T @ (vec[:3] - jp.cross(dif, vec[3:]))
      )(cfrc_int, dif, site_xmat)
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.ACTUATORFRC:
      sensor = d.actuator_force[objid]
    elif sensor_type == SensorType.JOINTACTFRC:
      sensor = d.qfrc_actuator[m.jnt_dofadr[objid]]
    elif sensor_type == SensorType.TENDONACTFRC:
      force_mask = [
          (m.actuator_trntype == TrnType.TENDON)
          & (m.actuator_trnid[:, 0] == tendon_id)
          for tendon_id in objid
      ]
      force_ids = np.concatenate([np.nonzero(mask)[0] for mask in force_mask])
      force_mat = np.array(force_mask)[:, force_ids]
      sensor = force_mat @ d.actuator_force[force_ids]
    elif sensor_type in (SensorType.FRAMELINACC, SensorType.FRAMEANGACC):
      objtype = m.sensor_objtype[idx]

      for ot in set(objtype):
        idxt = objtype == ot
        objidt = objid[idxt]
        pos, bodyid = objtype_data[int(ot)]
        pos = pos[objidt]
        bodyid = bodyid[objidt]
        cacc = d._impl.cacc[bodyid]  # pyrefly: ignore[missing-attribute]

        if sensor_type == SensorType.FRAMELINACC:

          @jax.vmap
          def _framelinacc(cvel, cacc, offset):
            ang = cvel[:3]
            lin = cvel[3:] - jp.cross(offset, cvel[:3])
            acc = cacc[3:] - jp.cross(offset, cacc[:3])
            correction = jp.cross(ang, lin)
            return acc + correction

          cvel = d.cvel[bodyid]
          offset = pos - d.subtree_com[m.body_rootid[bodyid]]

          sensor = _framelinacc(cvel, cacc, offset).reshape(-1)
        elif sensor_type == SensorType.FRAMEANGACC:
          sensor = cacc[:, :3].reshape(-1)
        else:
          raise ValueError(f'Unknown sensor type: {sensor_type}')

        # dof-less body (static or mocap): match mj_objectAcceleration
        dofless = m.body_dofnum[m.body_weldid[bodyid]] == 0
        sensor = jp.where(dofless[:, None], 0.0, sensor.reshape(-1, 3))
        sensor = sensor.reshape(-1)

        adrt = adr[idxt, None] + np.arange(3)[None]

        sensors.append(sensor.reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue  # avoid adding to sensors/adrs list a second time
    else:
      # TODO(taylorhowell): raise error after adding sensor check to io.py
      continue  # unsupported sensor type

    sensors.append(_apply_cutoff(sensor, cutoff, data_type[0]).reshape(-1))  # pyrefly: ignore[bad-argument-type]
    adrs.append(adr)

  if not adrs:
    return d

  sensordata = d.sensordata.at[np.concatenate(adrs)].set(
      jp.concatenate(sensors)
  )

  return d.replace(sensordata=sensordata)
