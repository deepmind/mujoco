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
"""Engine support functions."""

from collections.abc import Iterable, Sequence
from typing import Optional, Tuple, Union

import jax
from jax import numpy as jp
import mujoco
from mujoco.introspect import mjxmacro
from mujoco.mjx._src import math
from mujoco.mjx._src import scan
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import ConeType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import JacobianType
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


def is_sparse(m: Union[mujoco.MjModel, Model]) -> bool:
  """Return True if this model should create sparse mass matrices.

  Args:
    m: a MuJoCo or MJX model

  Returns:
    True if provided model should create sparse mass matrices

  Modern TPUs have specialized hardware for rapidly operating over sparse
  matrices, whereas GPUs tend to be faster with dense matrices as long as they
  fit onto the device.  As such, the default behavior in MJX (via
  ``JacobianType.AUTO``) is sparse if ``nv`` is >= 60 or MJX detects a TPU as
  the default backend, otherwise dense.
  """
  # AUTO is a rough heuristic - you may see better performance for your workload
  # and compute by explicitly setting jacobian to dense or sparse
  if m.opt.jacobian == JacobianType.AUTO:
    return m.nv >= 60 or jax.default_backend() == 'tpu'
  return m.opt.jacobian == JacobianType.SPARSE


def make_m(
    m: Model, a: jax.Array, b: jax.Array, d: Optional[jax.Array] = None
) -> jax.Array:
  """Computes M = a @ b.T + diag(d)."""

  i = np.repeat(np.arange(m.nv), m.M_rownnz)
  j = m.M_colind

  if not is_sparse(m):
    qm = a @ b.T
    if d is not None:
      qm += jp.diag(d)
    mask = jp.zeros((m.nv, m.nv), dtype=bool).at[(i, j)].set(True)
    qm = qm * mask
    qm = qm + jp.tril(qm, -1).T
    return qm

  a_i = jp.take(a, i, axis=0)
  b_j = jp.take(b, j, axis=0)
  qm = jax.vmap(jp.dot)(a_i, b_j)

  if d is not None:
    diag_adr = m.M_rowadr + m.M_rownnz - 1
    qm = qm.at[diag_adr].add(d)

  return qm


def full_m(m: Model, d: Data) -> jax.Array:
  """Reconstitute dense mass matrix from M."""

  if not is_sparse(m):
    return d._impl.M  # pyrefly: ignore[missing-attribute]

  i = np.repeat(np.arange(m.nv), m.M_rownnz)
  j = m.M_colind

  mat = jp.zeros((m.nv, m.nv)).at[(i, j)].set(d._impl.M)  # pyrefly: ignore[missing-attribute]

  # also set upper triangular
  mat = mat + jp.tril(mat, -1).T

  return mat


def mul_m(m: Model, d: Data, vec: jax.Array) -> jax.Array:
  """Multiply vector by inertia matrix."""

  if not is_sparse(m):
    return d._impl.M @ vec  # pyrefly: ignore[missing-attribute]

  diag_adr = m.M_rowadr + m.M_rownnz - 1
  diag_mul = d._impl.M[diag_adr] * vec  # pyrefly: ignore[missing-attribute]

  is_, js, madr_ijs = [], [], []
  for i in range(m.nv):
    adr = m.M_rowadr[i]
    for k in range(m.M_rownnz[i] - 1):
      is_.append(i)
      js.append(m.M_colind[adr + k])
      madr_ijs.append(adr + k)

  i, j, madr_ij = (np.array(x, dtype=np.int32) for x in (is_, js, madr_ijs))

  out = diag_mul.at[i].add(d._impl.M[madr_ij] * vec[j])  # pyrefly: ignore[missing-attribute]
  out = out.at[j].add(d._impl.M[madr_ij] * vec[i])  # pyrefly: ignore[missing-attribute]

  return out


def jac(
    m: Model, d: Data, point: jax.Array, body_id: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Compute pair of (NV, 3) Jacobians of global point attached to body."""
  # TODO(taylorhowell): statically construct mask
  fn = lambda carry, b: b if carry is None else b + carry
  mask = (jp.arange(m.nbody) == body_id) * 1
  mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
  mask = mask[jp.array(m.dof_bodyid)] > 0

  offset = point - d.subtree_com[jp.array(m.body_rootid)[body_id]]
  jacp = jax.vmap(lambda a, b=offset: a[3:] + jp.cross(a[:3], b))(d.cdof)
  jacp = jax.vmap(jp.multiply)(jacp, mask)
  jacr = jax.vmap(jp.multiply)(d.cdof[:, :3], mask)

  return jacp, jacr


def jac_dot(
    m: Model, d: Data, point: jax.Array, body_id: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Compute pair of (NV, 3) Jacobian time derivatives of global point attached to body."""
  # TODO(taylorhowell): statically construct mask
  fn = lambda carry, b: b if carry is None else b + carry
  mask = (jp.arange(m.nbody) == body_id) * 1
  mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
  mask = mask[jp.array(m.dof_bodyid)] > 0

  offset = point - d.subtree_com[jp.array(m.body_rootid)[body_id]]
  pvel_lin = d.cvel[body_id][3:] - jp.cross(offset, d.cvel[body_id][:3])

  cdof = d.cdof
  cdof_dot = d.cdof_dot

  # check for quaternion
  jnt_type = m.jnt_type[m.dof_jntid]
  dof_adr = m.jnt_dofadr[m.dof_jntid]
  is_quat = (jnt_type == JointType.BALL) | (
      (jnt_type == JointType.FREE) & (np.arange(m.nv) >= dof_adr + 3)
  )

  # compute cdof_dot for quaternion (use current body cvel)
  cdof_dot_quat = jax.vmap(math.motion_cross)(d.cvel[m.dof_bodyid], cdof)
  cdof_dot = jp.where(is_quat[:, None], cdof_dot_quat, cdof_dot)

  jacp = jax.vmap(
      lambda a, b: a[3:] + jp.cross(a[:3], offset) + jp.cross(b[:3], pvel_lin)
  )(cdof_dot, cdof)
  jacp = jax.vmap(jp.multiply)(jacp, mask)
  jacr = jax.vmap(jp.multiply)(cdof_dot[:, :3], mask)

  return jacp, jacr


def apply_ft(
    m: Model,
    d: Data,
    force: jax.Array,
    torque: jax.Array,
    point: jax.Array,
    body_id: jax.Array,
) -> jax.Array:
  """Apply Cartesian force and torque."""
  jacp, jacr = jac(m, d, point, body_id)
  return jacp @ force + jacr @ torque


def xfrc_accumulate(m: Model, d: Data) -> jax.Array:
  """Accumulate xfrc_applied into a qfrc."""
  qfrc = jax.vmap(apply_ft, in_axes=(None, None, 0, 0, 0, 0))(
      m,
      d,
      d.xfrc_applied[:, :3],
      d.xfrc_applied[:, 3:],
      d.xipos,
      jp.arange(m.nbody),
  )
  return jp.sum(qfrc, axis=0)


def local_to_global(
    world_pos: jax.Array,
    world_quat: jax.Array,
    local_pos: jax.Array,
    local_quat: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Converts local position/orientation to world frame."""
  pos = world_pos + math.rotate(local_pos, world_quat)
  mat = math.quat_to_mat(math.quat_mul(world_quat, local_quat))
  return pos, mat


def _getnum(m: Union[Model, mujoco.MjModel], obj: mujoco._enums.mjtObj) -> int:
  """Gets the number of objects for the given object type."""
  counts = {
      mujoco.mjtObj.mjOBJ_BODY: m.nbody,
      mujoco.mjtObj.mjOBJ_JOINT: m.njnt,
      mujoco.mjtObj.mjOBJ_GEOM: m.ngeom,
      mujoco.mjtObj.mjOBJ_SITE: m.nsite,
      mujoco.mjtObj.mjOBJ_CAMERA: m.ncam,
      mujoco.mjtObj.mjOBJ_MESH: m.nmesh,
      mujoco.mjtObj.mjOBJ_HFIELD: m.nhfield,
      mujoco.mjtObj.mjOBJ_PAIR: m.npair,
      mujoco.mjtObj.mjOBJ_EQUALITY: m.neq,
      mujoco.mjtObj.mjOBJ_TENDON: m.ntendon,
      mujoco.mjtObj.mjOBJ_ACTUATOR: m.nu,
      mujoco.mjtObj.mjOBJ_SENSOR: m.nsensor,
      mujoco.mjtObj.mjOBJ_NUMERIC: m.nnumeric,
      mujoco.mjtObj.mjOBJ_TUPLE: m.ntuple,
      mujoco.mjtObj.mjOBJ_KEY: m.nkey,
      mujoco.mjtObj.mjOBJ_FLEX: m.nflex,
  }
  return {int(k): v for k, v in counts.items()}.get(int(obj), 0)


def _getadr(
    m: Union[Model, mujoco.MjModel], obj: mujoco._enums.mjtObj
) -> np.ndarray:
  """Gets the name addresses for the given object type."""
  adrs = {
      mujoco.mjtObj.mjOBJ_BODY: m.name_bodyadr,
      mujoco.mjtObj.mjOBJ_JOINT: m.name_jntadr,
      mujoco.mjtObj.mjOBJ_GEOM: m.name_geomadr,
      mujoco.mjtObj.mjOBJ_SITE: m.name_siteadr,
      mujoco.mjtObj.mjOBJ_CAMERA: m.name_camadr,
      mujoco.mjtObj.mjOBJ_MESH: m.name_meshadr,
      mujoco.mjtObj.mjOBJ_HFIELD: m.name_hfieldadr,
      mujoco.mjtObj.mjOBJ_PAIR: m.name_pairadr,
      mujoco.mjtObj.mjOBJ_EQUALITY: m.name_eqadr,
      mujoco.mjtObj.mjOBJ_TENDON: m.name_tendonadr,
      mujoco.mjtObj.mjOBJ_ACTUATOR: m.name_actuatoradr,
      mujoco.mjtObj.mjOBJ_SENSOR: m.name_sensoradr,
      mujoco.mjtObj.mjOBJ_NUMERIC: m.name_numericadr,
      mujoco.mjtObj.mjOBJ_TUPLE: m.name_tupleadr,
      mujoco.mjtObj.mjOBJ_KEY: m.name_keyadr,
      mujoco.mjtObj.mjOBJ_FLEX: m.name_flexadr,
  }
  return {int(k): v for k, v in adrs.items()}[int(obj)]


def id2name(
    m: Union[Model, mujoco.MjModel], typ: mujoco._enums.mjtObj, i: int
) -> Optional[str]:
  """Gets the name of an object with the specified mjtObj type and ids.

  See mujoco.id2name for more info.

  Args:
    m: mujoco.MjModel or mjx.Model
    typ: mujoco.mjtObj type
    i: the id

  Returns:
    the name string, or None if not found
  """
  num = _getnum(m, typ)
  if i < 0 or i >= num:
    return None

  adr = _getadr(m, typ)
  name = m.names[adr[i] :].decode('utf-8').split('\x00', 1)[0]
  return name or None


def name2id(
    m: Union[Model, mujoco.MjModel], typ: mujoco._enums.mjtObj, name: str
) -> int:
  """Gets the id of an object with the specified mjtObj type and name.

  See mujoco.mj_name2id for more info.

  Args:
    m: mujoco.MjModel or mjx.Model
    typ: mujoco.mjtObj type
    name: the name of the object

  Returns:
   the id, or -1 if not found
  """
  num = _getnum(m, typ)
  adr = _getadr(m, typ)

  # TODO: consider using MjModel.names_map instead
  names_map = {
      m.names[adr[i] :].decode('utf-8').split('\x00', 1)[0]: i
      for i in range(num)
  }

  return names_map.get(name, -1)


class BindModel(object):
  """Class holding the requested MJX Model and spec id for binding a spec to Model."""

  def __init__(self, model: Model, specs: Sequence[mujoco.MjStruct]):
    self.model = model
    self.prefix = ''
    ids = []
    for spec in specs:
      if model.signature != spec.signature:
        raise ValueError(
            'mjSpec signature does not match mjx.Model signature:'
            f' {spec.signature} != {model.signature}'
        )
      elif spec.id < 0:
        raise KeyError(f'invalid id: {spec.id}')
      elif isinstance(spec, mujoco.MjsBody):
        self.prefix = 'body_'
      elif isinstance(spec, mujoco.MjsJoint):
        self.prefix = 'jnt_'
      elif isinstance(spec, mujoco.MjsGeom):
        self.prefix = 'geom_'
      elif isinstance(spec, mujoco.MjsSite):
        self.prefix = 'site_'
      elif isinstance(spec, mujoco.MjsLight):
        self.prefix = 'light_'
      elif isinstance(spec, mujoco.MjsCamera):
        self.prefix = 'cam_'
      elif isinstance(spec, mujoco.MjsMesh):
        self.prefix = 'mesh_'
      elif isinstance(spec, mujoco.MjsHField):
        self.prefix = 'hfield_'
      elif isinstance(spec, mujoco.MjsPair):
        self.prefix = 'pair_'
      elif isinstance(spec, mujoco.MjsTendon):
        self.prefix = 'tendon_'
      elif isinstance(spec, mujoco.MjsActuator):
        self.prefix = 'actuator_'
      elif isinstance(spec, mujoco.MjsSensor):
        self.prefix = 'sensor_'
      elif isinstance(spec, mujoco.MjsNumeric):
        self.prefix = 'numeric_'
      elif isinstance(spec, mujoco.MjsText):
        self.prefix = 'text_'
      elif isinstance(spec, mujoco.MjsTuple):
        self.prefix = 'tuple_'
      elif isinstance(spec, mujoco.MjsKey):
        self.prefix = 'key_'
      elif isinstance(spec, mujoco.MjsEquality):
        self.prefix = 'eq_'
      elif isinstance(spec, mujoco.MjsExclude):
        self.prefix = 'exclude_'
      elif isinstance(spec, mujoco.MjsSkin):
        self.prefix = 'skin_'
      elif isinstance(spec, mujoco.MjsMaterial):
        self.prefix = 'material_'
      else:
        raise ValueError('invalid spec type')
      ids.append(spec.id)
    if len(ids) == 1:
      self.id = ids[0]
    else:
      self.id = ids

  def _slice(self, name: str, idx: Union[int, slice, Sequence[int]]):
    _, expected_dim = mjxmacro.MJMODEL[name]
    var = getattr(self.model, name)
    if expected_dim == '1':
      return var[..., idx]
    elif expected_dim == '9':
      return var[..., idx, :, :]
    return var[..., idx, :]

  def __getattr__(self, name: str):
    return self._slice(self.prefix + name, self.id)


def _bind_model(
    self: Model, obj: mujoco.MjStruct | Iterable[mujoco.MjStruct]
) -> BindModel:
  """Bind a Mujoco spec to an MJX Model."""
  if isinstance(obj, mujoco.MjStruct):
    obj = (obj,)
  else:
    obj = tuple(obj)
  return BindModel(self, obj)


class BindData(object):
  """Class holding the requested MJX Data and spec id for binding a spec to Data."""

  def __init__(
      self, data: Data, model: Model, specs: Sequence[mujoco.MjStruct]
  ):
    self.data = data
    self.model = model
    self.prefix = ''
    ids = []
    for spec in specs:
      if model.signature != spec.signature:
        raise ValueError(
            'mjSpec signature does not match mjx.Model signature:'
            f' {spec.signature} != {model.signature}'
        )
      if spec.id < 0:
        raise KeyError(f'invalid id: {spec.id}')
      elif isinstance(spec, mujoco.MjsBody):
        pass
      elif isinstance(spec, mujoco.MjsJoint):
        self.prefix = 'jnt_'
      elif isinstance(spec, mujoco.MjsGeom):
        self.prefix = 'geom_'
      elif isinstance(spec, mujoco.MjsSite):
        self.prefix = 'site_'
      elif isinstance(spec, mujoco.MjsLight):
        self.prefix = 'light_'
      elif isinstance(spec, mujoco.MjsCamera):
        self.prefix = 'cam_'
      elif isinstance(spec, mujoco.MjsTendon):
        self.prefix = 'ten_'
      elif isinstance(spec, mujoco.MjsActuator):
        self.prefix = 'actuator_'
      elif isinstance(spec, mujoco.MjsSensor):
        self.prefix = 'sensor_'
      elif isinstance(spec, mujoco.MjsEquality):
        self.prefix = 'eq_'
      else:
        raise ValueError('invalid spec type')
      ids.append(spec.id)
    if len(ids) == 1:
      self.id = ids[0]
    else:
      self.id = ids

  def __getname(self, name: str):
    """Get the name of the attribute and check if the type is correct."""
    if name == 'sensordata':
      if self.prefix == 'sensor_':
        return name
      else:
        raise AttributeError('sensordata is not available for this type')
    if name == 'ctrl':
      if self.prefix == 'actuator_':
        return name
      else:
        raise AttributeError('ctrl is not available for this type')
    if (
        name == 'qpos'
        or name == 'qvel'
        or name == 'qacc'
        or name.startswith('qfrc_')
    ):
      if self.prefix == 'jnt_':
        return name
      else:
        raise AttributeError(
            'qpos, qvel, qacc, qfrc are not available for this type'
        )
    else:
      return self.prefix + name

  def _slice(self, name: str, idx: Union[int, slice, Sequence[int]]):
    _, expected_dim = mjxmacro.MJDATA[name]
    var = getattr(self.data, name)
    if expected_dim == '1':
      return var[..., idx]
    elif expected_dim == '9':
      return var[..., idx, :, :]
    return var[..., idx, :]

  def __getattr__(self, name: str):
    if name in ('sensordata', 'qpos', 'qvel', 'qacc') or (
        name.startswith('qfrc_')
    ):
      adr = num = 0
      if name == 'sensordata':
        adr = self.model.sensor_adr[self.id]
        num = self.model.sensor_dim[self.id]
      elif name == 'qpos':
        adr = self.model.jnt_qposadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.qpos_width() for jt in JointType)
      elif name == 'qvel' or name == 'qacc' or name.startswith('qfrc_'):
        adr = self.model.jnt_dofadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.dof_width() for jt in JointType)
      if isinstance(self.id, list):
        idx = []
        for a, n in zip(adr, num):  # pyrefly: ignore[bad-argument-type]
          idx.extend(a + j for j in range(n))
        return self._slice(self.__getname(name), idx)
      elif num > 1:
        return self._slice(self.__getname(name), slice(adr, adr + num))
      else:
        return self._slice(self.__getname(name), adr)  # pyrefly: ignore[bad-argument-type]
    elif name in ('mocap_pos', 'mocap_quat'):
      return self._slice(self.__getname(name), self.model.body_mocapid[self.id])  # pyrefly: ignore[bad-argument-type]
    return self._slice(self.__getname(name), self.id)

  def set(self, name: str, value: jax.Array) -> Data:
    """Set the value of an array in an MJX Data."""
    if name == 'sensordata':
      raise AttributeError('sensordata is readonly')
    array = getattr(self.data, self.__getname(name))
    dim = 1 if len(array.shape) == 1 else array.shape[-1]
    try:
      iter(value)
    except TypeError:
      value = [value]  # pyrefly: ignore[bad-assignment]
    if name in ('qpos', 'qvel', 'qacc', 'mocap_pos', 'mocap_quat'):
      adr = num = 0
      if name == 'qpos':
        adr = self.model.jnt_qposadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.qpos_width() for jt in JointType)
      elif name == 'qvel' or name == 'qacc':
        adr = self.model.jnt_dofadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.dof_width() for jt in JointType)
      elif name == 'mocap_pos':
        adr = self.model.body_mocapid[self.id] * 3
        num = np.ones_like(self.id, dtype=int) * 3
      elif name == 'mocap_quat':
        adr = self.model.body_mocapid[self.id] * 4
        num = np.ones_like(self.id, dtype=int) * 4
      if not isinstance(self.id, list):
        adr = [adr]
        num = [num]
    elif isinstance(self.id, list):
      adr = (np.array(self.id) * dim).tolist()
      num = [dim for _ in range(len(self.id))]
    else:
      adr = [self.id * dim]
      num = [dim]
    i = 0
    value = jax.numpy.array(value).flatten()
    for a, n in zip(adr, num):  # pyrefly: ignore[bad-argument-type]
      shape = array.shape
      array = array.flatten().at[a : a + n].set(value[i : i + n]).reshape(shape)
      i += n
    return self.data.replace(**{self.__getname(name): array})


def _bind_data(
    self: Data, model: Model, obj: mujoco.MjStruct | Iterable[mujoco.MjStruct]
) -> BindData:
  """Bind a Mujoco spec to an MJX Data."""
  if isinstance(obj, mujoco.MjStruct):
    obj = (obj,)
  else:
    obj = tuple(obj)
  return BindData(self, model, obj)


Model.bind = _bind_model
Data.bind = _bind_data


def _decode_pyramid(
    pyramid: jax.Array, mu: jax.Array, condim: int
) -> jax.Array:
  """Converts pyramid representation to contact force."""
  force = jp.zeros(6, dtype=float)
  if condim == 1:
    return force.at[0].set(pyramid[0])

  # force_normal = sum(pyramid0_i + pyramid1_i)
  force = force.at[0].set(pyramid[0 : 2 * (condim - 1)].sum())

  # force_tangent_i = (pyramid0_i - pyramid1_i) * mu_i
  i = np.arange(0, condim - 1)
  force = force.at[i + 1].set((pyramid[2 * i] - pyramid[2 * i + 1]) * mu[i])

  return force


def contact_force(
    m: Model, d: Data, contact_id: int, to_world_frame: bool = False
) -> jax.Array:
  """Extract 6D force:torque for one contact, in contact frame by default."""
  efc_address = d._impl.contact.efc_address[contact_id]  # pyrefly: ignore[missing-attribute]
  condim = d._impl.contact.dim[contact_id]  # pyrefly: ignore[missing-attribute]
  if m.opt.cone == ConeType.PYRAMIDAL:
    force = _decode_pyramid(
        d._impl.efc_force[efc_address:], d._impl.contact.friction[contact_id], condim  # pyrefly: ignore[missing-attribute]
    )
  elif m.opt.cone == ConeType.ELLIPTIC:
    force = d._impl.efc_force[efc_address : efc_address + condim]  # pyrefly: ignore[missing-attribute]
    force = jp.concatenate([force, jp.zeros((6 - condim))])
  else:
    raise ValueError(f'Unknown cone type: {m.opt.cone}')

  if to_world_frame:
    force = force.reshape((-1, 3)) @ d._impl.contact.frame[contact_id]  # pyrefly: ignore[missing-attribute]
    force = force.reshape(-1)

  return force * (efc_address >= 0)


def contact_force_dim(
    m: Model, d: Data, dim: int
) -> Tuple[jax.Array, np.ndarray]:
  """Extract 6D force:torque for contacts with dimension dim."""
  # valid contact and condim indices
  idx_dim = (d._impl.contact.efc_address >= 0) & (d._impl.contact.dim == dim)  # pyrefly: ignore[missing-attribute]

  # contact force from efc
  if m.opt.cone == ConeType.PYRAMIDAL:
    efc_address = (
        d._impl.contact.efc_address[idx_dim, None]  # pyrefly: ignore[missing-attribute]
        + np.arange(np.where(dim == 1, 1, 2 * (dim - 1)))[None]  # pyrefly: ignore[no-matching-overload]
    )
    efc_force = d._impl.efc_force[efc_address]  # pyrefly: ignore[missing-attribute]
    force = jax.vmap(_decode_pyramid, in_axes=(0, 0, None))(
        efc_force, d._impl.contact.friction[idx_dim], dim  # pyrefly: ignore[missing-attribute]
    )
  elif m.opt.cone == ConeType.ELLIPTIC:
    efc_address = d._impl.contact.efc_address[idx_dim, None] + np.arange(dim)[None]  # pyrefly: ignore[missing-attribute]
    force = d._impl.efc_force[efc_address]  # pyrefly: ignore[missing-attribute]
    force = jp.hstack([force, jp.zeros((force.shape[0], 6 - dim))])
  else:
    raise ValueError(f'Unknown cone type: {m.opt.cone}.')
  return force, np.where(idx_dim)[0]


def _length_circle(
    p0: jax.Array, p1: jax.Array, ind: jax.Array, rad: jax.Array
) -> jax.Array:
  """Compute length of circle."""
  # compute angle between 0 and pi
  p0n = math.normalize(p0).reshape(-1)
  p1n = math.normalize(p1).reshape(-1)

  # clip input to closed interval for jp.arccos to prevent potential nan
  # TODO(taylorhowell): add test for case where clip is necessary
  angle = jp.arccos(jp.clip(jp.dot(p0n, p1n), -1, 1))

  # flip if necessary
  cross = p0[1] * p1[0] - p0[0] * p1[1]
  flip = ((cross > 0) & (ind != 0)) | ((cross < 0) & (ind == 0))
  angle = jp.where(flip, 2 * jp.pi - angle, angle)

  return rad * angle


def _is_intersect(
    p1: jax.Array, p2: jax.Array, p3: jax.Array, p4: jax.Array
) -> jax.Array:
  """Check for intersection between two lines defined by their endpoints."""
  # compute determinant
  det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])

  # compute intersection point on each line
  a = math.safe_div(
      (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0]), det
  )
  b = math.safe_div(
      (p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0]), det
  )

  return jp.where(
      jp.abs(det) < mujoco.mjMINVAL,
      0,
      (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1),
  )


def _winding_angle(p: jax.Array, q: jax.Array) -> jax.Array:
  """Signed angle swept from p to q around the origin, in (-pi, pi]."""
  return jp.arctan2(p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1])


def wrap_circle(
    d: jax.Array, sd: jax.Array, sidesite: jax.Array, rad: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Compute circle wrap arc length and end points."""
  # check cases
  sqlen0 = d[0] ** 2 + d[1] ** 2
  sqlen1 = d[2] ** 2 + d[3] ** 2
  sqrad = rad * rad
  dif = jp.array([d[2] - d[0], d[3] - d[1]])
  dd = dif[0] ** 2 + dif[1] ** 2
  a = jp.clip(
      -(dif[0] * d[0] + dif[1] * d[1]) / jp.maximum(mujoco.mjMINVAL, dd), 0, 1
  )
  seg = jp.array([a * dif[0] + d[0], a * dif[1] + d[1]])

  point_inside0 = sqlen0 < sqrad
  point_inside1 = sqlen1 < sqrad
  circle_too_small = rad < mujoco.mjMINVAL
  points_too_close = dd < mujoco.mjMINVAL

  # check for intersection
  clear = (seg[0] ** 2 + seg[1] ** 2) > sqrad

  # side: the path is homotopic to the polyline d0 -> sd -> d1, whose winding
  # angle around the center determines the wrap orientation; the segment
  # d0 -> d1 is in the same homotopy class iff its winding angle is the same
  # (they differ by multiples of 2*pi)
  wside = _winding_angle(d[:2], sd) + _winding_angle(sd, d[2:])
  wline = _winding_angle(d[:2], d[2:])
  straight = jp.where(sidesite, clear & (jp.abs(wside - wline) < jp.pi), clear)

  # construct the two solutions, compute goodness
  def _sol(sgn):
    sqrt0 = jp.sqrt(jp.maximum(mujoco.mjMINVAL, sqlen0 - sqrad))
    sqrt1 = jp.sqrt(jp.maximum(mujoco.mjMINVAL, sqlen1 - sqrad))

    d00 = (d[0] * sqrad + sgn * rad * d[1] * sqrt0) / jp.maximum(
        mujoco.mjMINVAL, sqlen0
    )
    d01 = (d[1] * sqrad - sgn * rad * d[0] * sqrt0) / jp.maximum(
        mujoco.mjMINVAL, sqlen0
    )
    d10 = (d[2] * sqrad - sgn * rad * d[3] * sqrt1) / jp.maximum(
        mujoco.mjMINVAL, sqlen1
    )
    d11 = (d[3] * sqrad + sgn * rad * d[2] * sqrt1) / jp.maximum(
        mujoco.mjMINVAL, sqlen1
    )

    sol = jp.array([[d00, d01], [d10, d11]])

    # goodness: shorter path
    tmp = (sol[0] - sol[1]).reshape(-1)
    good = -jp.dot(tmp, tmp)

    # penalize for intersection
    intersect = _is_intersect(d[:2], sol[0], d[2:], sol[1])
    good = jp.where(intersect, -10000, good)

    return sol, good

  sol, good = jax.vmap(_sol)(jp.array([1, -1]))

  # select the solution: with side, by orientation (i=1 is counterclockwise),
  # otherwise the better
  i = jp.where(sidesite, wside > 0, good[0] <= good[1]).astype(int)
  i = i.reshape(())
  sol = sol[i]
  pnt = sol.reshape(-1)

  # check for intersection
  intersect = _is_intersect(d[:2], pnt[:2], d[2:], pnt[2:])

  # compute curve length; side: the path winds like the polyline, so the arc
  # angle is its winding minus that of the tangent segments
  arc = wside - _winding_angle(d[:2], sol[0]) - _winding_angle(sol[1], d[2:])
  wlen = jp.where(
      sidesite, rad * jp.abs(arc), _length_circle(sol[0], sol[1], i, rad)
  )

  # check cases
  invalid = (
      point_inside0
      | point_inside1
      | circle_too_small
      | points_too_close
      | straight
      | intersect
  )

  wlen = jp.where(invalid, -1, wlen)
  pnt = jp.where(invalid, jp.zeros(4), pnt)

  return wlen, pnt


def wrap_inside(
    end: jax.Array,
    radius: jax.Array,
    maxiter: int,
    tolerance: float,
    z_init: float,
) -> Tuple[jax.Array, jax.Array]:
  """Compute 2D inside wrap point.

  Args:
    end: 2D points
    radius: radius of circle

  Returns:
    status: 0 if wrap, else -1
    concatentated 2D wrap points: jax.Array
  """
  mjMINVAL = mujoco.mjMINVAL  # pylint: disable=invalid-name

  # constants
  len0 = math.norm(end[:2])
  len1 = math.norm(end[2:])
  dif = jp.array([end[2] - end[0], end[3] - end[1]])
  dd = dif[0] * dif[0] + dif[1] * dif[1]

  # either point inside circle or circle too small: no wrap
  no_wrap0 = (
      (len0 <= radius)
      | (len1 <= radius)
      | (radius < mjMINVAL)
      | (len0 < mjMINVAL)
      | (len1 < mjMINVAL)
  )

  # find nearest point on line segment to origin: d0 + a*dif
  a = -1 * (dif[0] * end[0] + dif[1] * end[1]) / jp.maximum(mjMINVAL, dd)
  tmp = end[:2] + a * dif

  # segment-circle intersection: no wrap
  no_wrap1 = (dd > mjMINVAL) & (a > 0) & (a < 1) & (math.norm(tmp) <= radius)

  # prepare default in case of numerical failure: average
  pnt_avg = 0.5 * jp.array([end[0] + end[2], end[1] + end[3]])
  pnt_avg = radius * math.normalize(pnt_avg)

  # compute function parameters: asin(A*z) + asin(B*z) - 2*asin(z) + G = 0
  A = radius / jp.maximum(mjMINVAL, len0)  # pylint: disable=invalid-name
  B = radius / jp.maximum(mjMINVAL, len1)  # pylint: disable=invalid-name
  cosG = (len0 * len0 + len1 * len1 - dd) / jp.maximum(mjMINVAL, 2 * len0 * len1)  # pylint: disable=invalid-name

  no_wrap2 = cosG < -1 + mjMINVAL
  early_return0 = cosG > 1 - mjMINVAL

  G = jp.arccos(cosG)  # pylint: disable=invalid-name

  # initialize solver
  z = jp.array([z_init])
  f = jp.arcsin(A * z) + jp.arcsin(B * z) - 2 * jp.arcsin(z) + G

  # make sure initialization is not on the other side
  early_return1 = f > 0

  # iteratively solve with Newton's method
  def _newton(carry, _):
    # unpack
    z, f, status_prev = carry

    # check current solution
    converged = jp.abs(f) <= tolerance

    # compute derivative
    df = (
        A / jp.maximum(mjMINVAL, jp.sqrt(1 - z * z * A * A))
        + B / jp.maximum(mjMINVAL, jp.sqrt(1 - z * z * B * B))
        - 2 / jp.maximum(mjMINVAL, jp.sqrt(1 - z * z))
    )

    # check sign; SHOULD NOT OCCUR
    status0 = df > -mjMINVAL

    # new point
    z_next = z - (1 - converged) * math.safe_div(f, df)

    # make sure we are moving to the left; SHOULD NOT OCCUR
    status1 = z_next > z

    # evaluate solution
    f_next = (
        jp.arcsin(A * z_next)
        + jp.arcsin(B * z_next)
        - 2 * jp.arcsin(z_next)
        + G
    )

    # exit if positive; SHOULD NOT OCCUR
    status2 = f_next > tolerance

    return (
        z_next,
        f_next,
        status_prev | status0 | status1 | status2,
    ), None

  # TODO(taylorhowell): compare performance of jax.lax.scan and jax.lax.while_loop
  z, _, early_return2 = jax.lax.scan(
      _newton, (z, f, jp.array([False])), None, maxiter
  )[0]

  # finalize: rotation by ang from vec = a or b, depending on cross(a,b) sign
  sign = end[0] * end[3] - end[1] * end[2] > 0
  vec = jp.where(sign, end[:2], end[2:])
  vec = math.normalize(vec)
  ang = jp.arcsin(z) - jp.where(sign, jp.arcsin(A * z), jp.arcsin(B * z))
  pnt_sol = radius * jp.array([
      jp.cos(ang) * vec[0] - jp.sin(ang) * vec[1],
      jp.sin(ang) * vec[0] + jp.cos(ang) * vec[1],
  ]).reshape(-1)

  no_wrap = no_wrap0 | no_wrap1 | no_wrap2
  early_return = early_return0 | early_return1 | early_return2
  status = -1 * no_wrap * jp.ones(1)

  pnt = jp.where(early_return, pnt_avg, pnt_sol)
  pnt = jp.where(no_wrap, jp.zeros(2), pnt)

  return status, jp.concatenate([pnt, pnt])


def wrap(
    x0: jax.Array,
    x1: jax.Array,
    xpos: jax.Array,
    xmat: jax.Array,
    size: jax.Array,
    side: jax.Array,
    sidesite: jax.Array,
    is_sphere: jax.Array,
    is_wrap_inside: bool,
    wrap_inside_maxiter: int,
    wrap_inside_tolerance: float,
    wrap_inside_z_init: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Wrap tendon around sphere or cylinder."""
  # map sites to wrap object's local frame
  p0 = xmat.T @ (x0 - xpos)
  p1 = xmat.T @ (x1 - xpos)

  close_to_origin = (jp.linalg.norm(p0) < mujoco.mjMINVAL) | (
      jp.linalg.norm(p1) < mujoco.mjMINVAL
  )

  # compute axes for sphere
  # 1st axis
  axis0 = p0
  axis0 = math.normalize(axis0)

  # compute normal to p0-0-p1 plane = cross(p0, p1)
  normal = jp.cross(p0, p1)
  normal, nrm = math.normalize_with_norm(normal)

  # compute alternative normal (if (p0, p1) are parallel)
  # find max component of axis0
  axis_alt = jp.ones(3).at[jp.argmax(axis0)].set(0)
  normal_alt = jp.cross(axis0, axis_alt)
  normal_alt = math.normalize(normal_alt)

  normal = jp.where(nrm < mujoco.mjMINVAL, normal_alt, normal)

  # 2nd axis
  axis1 = jp.cross(normal, axis0)
  axis1 = math.normalize(axis1)

  # set geom dependent axes
  axis0 = jp.where(is_sphere, axis0, jp.array([1.0, 0.0, 0.0]))
  axis1 = jp.where(is_sphere, axis1, jp.array([0.0, 1.0, 0.0]))

  # project points in 2D frame: p => d
  d = jp.array([
      jp.dot(p0, axis0),
      jp.dot(p0, axis1),
      jp.dot(p1, axis0),
      jp.dot(p1, axis1),
  ])

  # compute sidesite projection
  s = xmat.T @ (side - xpos)
  sd = jp.array([jp.dot(s, axis0), jp.dot(s, axis1)])
  sd, sdnrm = math.normalize_with_norm(sd)
  sd = jp.where(sdnrm < mujoco.mjMINVAL, jp.array([1.0, 0.0]), sd) * size

  if is_wrap_inside:
    wlen, pnt = wrap_inside(
        d, size, wrap_inside_maxiter, wrap_inside_tolerance, wrap_inside_z_init
    )
  else:
    wlen, pnt = wrap_circle(d, sd, sidesite, size)

  no_wrap = wlen < 0

  # reconstruct 3D points in local frame: res
  res0 = axis0 * pnt[0] + axis1 * pnt[1]
  res1 = axis0 * pnt[2] + axis1 * pnt[3]
  res = jp.concatenate([res0, res1])

  # perform correction for cylinder case
  l0 = jp.sqrt(
      (p0[0] - res[0]) * (p0[0] - res[0]) + (p0[1] - res[1]) * (p0[1] - res[1])
  )
  l1 = jp.sqrt(
      (p1[0] - res[3]) * (p1[0] - res[3]) + (p1[1] - res[4]) * (p1[1] - res[4])
  )
  r2 = p0[2] + (p1[2] - p0[2]) * math.safe_div(l0, l0 + wlen + l1)
  r5 = p0[2] + (p1[2] - p0[2]) * math.safe_div(l0 + wlen, l0 + wlen + l1)
  height = jp.abs(r5 - r2)

  wlen = jp.where(is_sphere, wlen, jp.sqrt(wlen * wlen + height * height))
  res = jp.where(
      is_sphere, res, res.at[jp.array([2, 5])].set(jp.concatenate([r2, r5]))
  )

  # map wrap points back to global frame
  wpnt0 = xmat @ res[:3] + xpos
  wpnt1 = xmat @ res[3:] + xpos

  # check cases for no wrap
  invalid = close_to_origin | no_wrap

  wlen = jp.where(invalid, -1, wlen)
  wpnt0 = jp.where(invalid, jp.zeros(3), wpnt0)
  wpnt1 = jp.where(invalid, jp.zeros(3), wpnt1)

  return wlen, wpnt0, wpnt1


def muscle_gain_length(
    length: jax.Array, lmin: jax.Array, lmax: jax.Array
) -> jax.Array:
  """Normalized muscle length-gain curve."""
  # mid-ranges (maximum is at 1.0)
  a = 0.5 * (lmin + 1)
  b = 0.5 * (1 + lmax)

  out0 = 0.5 * jp.square(
      (length - lmin) / jp.maximum(mujoco.mjMINVAL, a - lmin)
  )
  out1 = 1 - 0.5 * jp.square((1 - length) / jp.maximum(mujoco.mjMINVAL, 1 - a))
  out2 = 1 - 0.5 * jp.square((length - 1) / jp.maximum(mujoco.mjMINVAL, b - 1))
  out3 = 0.5 * jp.square(
      (lmax - length) / jp.maximum(mujoco.mjMINVAL, lmax - b)
  )

  out = jp.where(length <= b, out2, out3)
  out = jp.where(length <= 1, out1, out)
  out = jp.where(length <= a, out0, out)
  out = jp.where((lmin <= length) & (length <= lmax), out, 0.0)

  return out


def muscle_gain(
    length: jax.Array,
    vel: jax.Array,
    lengthrange: jax.Array,
    acc0: jax.Array,
    prm: jax.Array,
) -> jax.Array:
  """Muscle active force."""
  # unpack parameters
  lrange = prm[:2]
  force, scale, lmin, lmax, vmax, _, fvmax = prm[2:9]

  force = jp.where(force < 0, scale / jp.maximum(mujoco.mjMINVAL, acc0), force)

  # optimum length
  L0 = (lengthrange[1] - lengthrange[0]) / jp.maximum(  # pylint:disable=invalid-name
      mujoco.mjMINVAL, lrange[1] - lrange[0]
  )

  # normalized length and velocity
  L = lrange[0] + (length - lengthrange[0]) / jp.maximum(mujoco.mjMINVAL, L0)  # pylint:disable=invalid-name
  V = vel / jp.maximum(mujoco.mjMINVAL, L0 * vmax)  # pylint:disable=invalid-name

  # length curve
  FL = muscle_gain_length(L, lmin, lmax)  # pylint:disable=invalid-name

  # velocity curve
  y = fvmax - 1
  FV = jp.where(  # pylint:disable=invalid-name
      V <= y, fvmax - jp.square(y - V) / jp.maximum(mujoco.mjMINVAL, y), fvmax
  )
  FV = jp.where(V <= 0, jp.square(V + 1), FV)  # pylint:disable=invalid-name
  FV = jp.where(V <= -1, 0, FV)  # pylint:disable=invalid-name

  # compute FVL and scale, make it negative
  return -force * FL * FV


def muscle_bias(
    length: jax.Array, lengthrange: jax.Array, acc0: jax.Array, prm: jax.Array
) -> jax.Array:
  """Muscle passive force."""
  # unpack parameters
  lrange = prm[:2]
  force, scale, _, lmax, _, fpmax = prm[2:8]

  force = jp.where(force < 0, scale / jp.maximum(mujoco.mjMINVAL, acc0), force)

  # optimum length
  L0 = (lengthrange[1] - lengthrange[0]) / jp.maximum(  # pylint:disable=invalid-name
      mujoco.mjMINVAL, lrange[1] - lrange[0]
  )

  # normalized length
  L = lrange[0] + (length - lengthrange[0]) / jp.maximum(mujoco.mjMINVAL, L0)  # pylint:disable=invalid-name

  # half-quadratic to (L0 + lmax) / 2, linear beyond
  b = 0.5 * (1 + lmax)

  out1 = (
      -force
      * fpmax
      * 0.5
      * jp.square((L - 1) / jp.maximum(mujoco.mjMINVAL, b - 1))
  )
  out2 = -force * fpmax * (0.5 + (L - b) / jp.maximum(mujoco.mjMINVAL, b - 1))

  out = jp.where(L <= b, out1, out2)
  out = jp.where(L <= 1, 0.0, out)

  return out


def muscle_dynamics_timescale(
    dctrl: jax.Array,
    tau_act: jax.Array,
    tau_deact: jax.Array,
    smoothing_width: jax.Array,
) -> jax.Array:
  """Muscle time constant with optional smoothing."""
  # hard switching
  tau_hard = jp.where(dctrl > 0, tau_act, tau_deact)

  def _sigmoid(x):
    # sigmoid function over 0 <= x <= 1 using quintic polynomial
    # sigmoid: f(x) = 6 * x^5 - 15 * x^4 + 10 * x^3
    # solution of f(0) = f'(0) = f''(0) = 0, f(1) = 1, f'(1) = f''(1) = 0
    x = jp.clip(x, 0, 1)
    return x * x * x * (3 * x * (2 * x - 5) + 10)

  # smooth switching
  # scale by width, center around 0.5 midpoint, rescale to bounds
  tau_smooth = tau_deact + (tau_act - tau_deact) * _sigmoid(
      math.safe_div(dctrl, smoothing_width) + 0.5
  )

  return jp.where(smoothing_width < mujoco.mjMINVAL, tau_hard, tau_smooth)


def muscle_dynamics(
    ctrl: jax.Array, act: jax.Array, prm: jax.Array
) -> jax.Array:
  """Muscle activation dynamics."""
  # clamp control
  ctrlclamp = jp.clip(ctrl, 0, 1)

  # clamp activation
  actclamp = jp.clip(act, 0, 1)

  # compute timescales as in Millard et at. (2013)
  # https://doi.org/10.1115/1.4023390
  tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale
  tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale
  smoothing_width = prm[2]  # width of smoothing sigmoid
  dctrl = ctrlclamp - act  # excess excitation

  tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smoothing_width)

  # filter output
  return dctrl / jp.maximum(mujoco.mjMINVAL, tau)
