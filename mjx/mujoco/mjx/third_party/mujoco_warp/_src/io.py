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

import copy
import dataclasses
import warnings
from typing import Any, Optional, Sequence

import mujoco
import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import history
from mujoco.mjx.third_party.mujoco_warp._src import sleep
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src import util_pkg
from mujoco.mjx.third_party.mujoco_warp._src import warp_util

wp.set_module_options({"default_grid_stride": False})

# MuJoCo contact exclusion flags (mjdata.h):
# 0: include, 1: in gap, 2: fused, 3: no dofs, 4: passive.
CONTACT_EXCLUDE_INCLUDE = 0
CONTACT_EXCLUDE_PASSIVE = 4


def _create_array(data: Any, spec, sizes: dict[str, int], batch_size: int = 1) -> wp.array | None:
  """Creates a warp array and populates it with data.

  The array shape is determined by a field spec referencing MjModel / MjData array sizes.
  """
  spec_shape = getattr(spec, "shape", (0,))
  if spec_shape == (0,):
    if data is None:
      return None
    return wp.array(np.array(data), dtype=spec.dtype)

  shape = tuple(
    batch_size
    if dim == "*"
    else (int(dim) if isinstance(dim, str) and dim.isdigit() else (sizes[dim] if isinstance(dim, str) else dim))
    for dim in spec_shape
  )

  is_batched = spec_shape[0] in ("*", "nworld")

  if data is None:
    array = wp.zeros(shape, dtype=spec.dtype)
  else:
    data = np.array(data)
    if is_batched and shape[0] != 1:
      target_shape = shape + getattr(spec.dtype, "_shape_", ())
      if data.shape != target_shape:
        tail_shape = target_shape[1:]
        if data.size == np.prod(tail_shape):
          data = data.reshape(tail_shape)
        data = np.broadcast_to(data, target_shape).copy()
    array = wp.array(data, dtype=spec.dtype, shape=shape)

  return array


def _create_constraint(
  mjm,
  nworld: int,
  njmax: int,
  sizes: dict,
  mjd=None,
) -> types.Constraint:
  """Construct a types.Constraint with standard and island local fields allocated properly."""
  efc_kwargs = {"J_rownnz": None, "J_rowadr": None, "J_colind": None, "J": None}
  sparse = is_sparse(mjm)
  # The JTDAJ block list is only consumed by the sparse Newton Hessian assembly (_JTDACJ_sparse).
  jtdaj_active = sparse and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON

  efc_D_src = mjd.efc_D if mjd is not None else None
  efc_aref_src = mjd.efc_aref if mjd is not None else None
  if mjd is not None and mjd.nefc > 0 and mjm.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE:
    mjm_tmp = copy.copy(mjm)
    mjm_tmp.opt.disableflags |= (
      mujoco.mjtDisableBit.mjDSBL_SPRING | mujoco.mjtDisableBit.mjDSBL_DAMPER | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    )
    mjm_tmp.opt.density = 0.0
    mjm_tmp.opt.viscosity = 0.0
    mjd_tmp = copy.copy(mjd)
    mujoco.mj_forwardSkip(mjm_tmp, mjd_tmp, mujoco.mjtStage.mjSTAGE_NONE, 1)
    if mjd_tmp.nefc == mjd.nefc:
      efc_D_src = mjd_tmp.efc_D
      efc_aref_src = mjd_tmp.efc_aref

  for f in dataclasses.fields(types.Constraint):
    if f.name in ("jtdaj_adr", "jtdaj_nrow"):
      efc_kwargs[f.name] = wp.empty((nworld, njmax if jtdaj_active else 0), dtype=int)
    elif f.name == "jtdaj_nblock":
      efc_kwargs[f.name] = wp.empty((nworld,), dtype=int)
    else:
      if f.name in efc_kwargs:
        continue

    if mjd is not None:
      shape = tuple(sizes[dim] if isinstance(dim, str) else dim for dim in f.type.shape)
      val = np.full(shape, -1 if f.name == "island" else 0, dtype=wp.dtype_to_numpy(f.type.dtype))
      if f.name == "D":
        val[:, : mjd.nefc] = np.tile(efc_D_src, (nworld, 1))
      elif f.name == "aref":
        val[:, : mjd.nefc] = np.tile(efc_aref_src, (nworld, 1))
      elif f.name in ("type", "id", "pos", "margin", "vel", "frictionloss", "force", "island"):
        val[:, : mjd.nefc] = np.tile(getattr(mjd, "efc_" + f.name), (nworld, 1))
      efc_kwargs[f.name] = wp.array(val, dtype=f.type.dtype)
    else:
      efc_kwargs[f.name] = _create_array(None, f.type, sizes)

  return types.Constraint(**efc_kwargs)


def _jtdaj_groups(mjd: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
  """Group loaded efc rows into JTDAJ blocks: maximal runs sharing (efc_type, efc_id).

  MuJoCo lays each constraint's rows out contiguously, so this reproduces the block list
  make_constraint builds in-kernel. Returns block start rows (adr) and lengths (nrow).
  DOF friction rows get no block, matching make_constraint.
  """
  nefc = mjd.nefc
  if nefc == 0:
    return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
  etype = mjd.efc_type[:nefc]
  eid = mjd.efc_id[:nefc]
  boundary = np.ones(nefc, dtype=bool)
  boundary[1:] = (etype[1:] != etype[:-1]) | (eid[1:] != eid[:-1])
  adr = np.flatnonzero(boundary)
  nrow = np.diff(np.append(adr, nefc))
  keep = etype[adr] != mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
  return adr[keep], nrow[keep]


def _flex_has_stiffness(mjm: mujoco.MjModel, f: int) -> bool:
  """Whether flex f has a nonzero stretch stiffness record, see mj_flexHasStiffness."""
  adr = int(mjm.flex_stiffnessadr[f])
  if adr < 0:
    return False
  order = int(mjm.flex_interp[f])
  if order < 0:
    return True
  if order == 0:
    return bool(mjm.flex_stiffness[adr] != 0)
  cx, cy, cz = mjm.flex_cellnum[f]
  npe = (order + 1) ** 3
  nelem = int(cx) * int(cy) * int(cz)
  stride = 9 * npe * npe
  return any(mjm.flex_stiffness[adr + e * stride] != 0 for e in range(nelem))


def _get_nflexintcell(mjm: mujoco.MjModel) -> int:
  nflexintcell = 0
  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      order = int(mjm.flex_interp[fi])
      if order == 0 or mjm.flex_edgeequality[fi] == 3 or not _flex_has_stiffness(mjm, fi):
        continue
      cx, cy, cz = (int(x) for x in mjm.flex_cellnum[fi])
      if order < 0:
        nflexintcell += 2 * (cy * cz + cx * cz + cx * cy)
      else:
        nflexintcell += cx * cy * cz
  return nflexintcell


def _get_nflexface(mjm: mujoco.MjModel) -> int:
  nflexface = 0
  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      order = mjm.flex_interp[fi]
      if order >= 0:
        continue
      cx, cy, cz = mjm.flex_cellnum[fi]
      nfaces = 2 * (cy * cz + cx * cz + cx * cy)
      nflexface += int(nfaces)
  return nflexface


def _flex_body_simple(mjm: mujoco.MjModel, body: int) -> bool:
  b = int(mjm.body_weldid[body])
  if mjm.body_dofnum[b] != 3 or mjm.body_jntnum[b] != 3:
    return False
  if mjm.body_dofnum[mjm.body_weldid[mjm.body_parentid[b]]] != 0:
    return False
  jadr = int(mjm.body_jntadr[b])
  for j in range(3):
    if mjm.jnt_type[jadr + j] != mujoco.mjtJoint.mjJNT_SLIDE:
      return False
    for k in range(3):
      target = 1.0 if j == k else 0.0
      if abs(float(mjm.jnt_axis[jadr + j, k]) - target) > types.MJ_MINVAL:
        return False
  return True


def _flex_simple(mjm: mujoco.MjModel, f: int) -> bool:
  for v in range(int(mjm.flex_vertadr[f]), int(mjm.flex_vertadr[f] + mjm.flex_vertnum[f])):
    bodyid = int(mjm.flex_vertbodyid[v])
    if bodyid < 0:
      continue
    b = int(mjm.body_weldid[bodyid])
    if mjm.body_dofnum[b] != 0 and not _flex_body_simple(mjm, b):
      return False
  return True


def _eff_flex_contact_possible(mjm: mujoco.MjModel, f: int) -> bool:
  """Whether flex f uses penalty passive contact in the metric, see mj_effFlexContactPossible."""
  return bool(
    mjm.flex_passive[f]
    and not mjm.flex_rigid[f]
    and mjm.flex_dim[f] >= 2
    and not (mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_IPC)
  )


def _eff_flex_stiff_possible(mjm: mujoco.MjModel, f: int) -> bool:
  """Whether flex f contributes elastic stiffness to the metric, see mj_effFlexStiffPossible."""
  if mjm.flex_rigid[f]:
    return False
  badr = int(mjm.flex_bendingadr[f])
  if mjm.flex_dim[f] == 1:
    return badr >= 0
  if mjm.flex_edgeequality[f] != 3 and _flex_has_stiffness(mjm, f):
    return True
  if badr < 0:
    return False
  end = mjm.nflexbending
  for g in range(f + 1, mjm.nflex):
    if mjm.flex_bendingadr[g] >= 0:
      end = int(mjm.flex_bendingadr[g])
      break
  return bool(np.any(mjm.flex_bending[badr:end] != 0))


def _is_open_rod_chain(mjm: mujoco.MjModel, fi: int) -> tuple[bool, int]:
  """Returns (is_open_chain, nhinge) for single-hinge open 1D rod flex `fi`."""
  if types.ROD_HINGE_SIZE != 10:
    return False, 0
  bendingadr = int(mjm.flex_bendingadr[fi])
  if mjm.flex_dim[fi] != 1 or mjm.flex_interp[fi] != 0 or bendingadr < 0:
    return False, 0
  nhinge = int(mjm.flex_bending[bendingadr])
  if nhinge <= 0:
    return False, 0
  records = mjm.flex_bending[bendingadr + 1 : bendingadr + 1 + 10 * nhinge].reshape(nhinge, 10)
  if (records[:, 3] >= 0).any():
    return False, 0
  for h in range(1, nhinge):
    if int(records[h, 0]) != int(records[h - 1, 1]) or int(records[h, 1]) != int(records[h - 1, 2]):
      return False, 0
  if int(records[-1, 1]) == int(records[0, 0]) or int(records[-1, 2]) == int(records[0, 0]):
    return False, 0
  return True, nhinge


def _get_nflexrod_counts(mjm: mujoco.MjModel) -> tuple[int, int]:
  """Returns (nflexrodhinge, nflexrodedge) across all flexes."""
  nflexrodhinge = 0
  nflexrodedge = 0
  for fi in range(mjm.nflex):
    bendingadr = int(mjm.flex_bendingadr[fi])
    if mjm.flex_dim[fi] == 1 and mjm.flex_interp[fi] == 0 and not mjm.flex_rigid[fi] and bendingadr >= 0:
      nhinge = int(mjm.flex_bending[bendingadr])
      if nhinge > 0:
        nflexrodhinge += nhinge
    is_open, nhinge_open = _is_open_rod_chain(mjm, fi)
    if is_open:
      nflexrodedge += nhinge_open + 1
  return nflexrodhinge, nflexrodedge


def _mjm_rod_vert_elems(mjm: mujoco.MjModel, fi: int) -> list[int]:
  """Element of 1D flex `fi` incident to each local vertex, or -1; matches mj_rodVertElem.

  A chain vertex maps to the element of the chain edge leaving it (the last edge at the far end),
  any other vertex to the first element containing it. O(nvert + nelem) via lookup tables.
  """
  nvert = int(mjm.flex_vertnum[fi])
  elemnum = int(mjm.flex_elemnum[fi])
  res = [-1] * nvert
  if mjm.flex_dim[fi] != 1 or elemnum <= 0:
    return res
  ed_base = int(mjm.flex_elemdataadr[fi])
  elem = [(int(mjm.flex_elem[ed_base + 2 * e]), int(mjm.flex_elem[ed_base + 2 * e + 1])) for e in range(elemnum)]
  first_elem = {}
  pair_elem = {}
  for e, (ev0, ev1) in enumerate(elem):
    first_elem.setdefault(ev0, e)
    first_elem.setdefault(ev1, e)
    pair_elem.setdefault((min(ev0, ev1), max(ev0, ev1)), e)

  chain = []
  bendingadr = int(mjm.flex_bendingadr[fi])
  if bendingadr >= 0 and types.ROD_HINGE_SIZE == 10:
    nhinge = int(mjm.flex_bending[bendingadr])
    if nhinge > 0 and mjm.flex_bending[bendingadr + 1 + 3] < 0:
      chain = [int(mjm.flex_bending[bendingadr + 1 + 10 * h]) for h in range(nhinge)]
      chain += [int(mjm.flex_bending[bendingadr + 1 + 10 * (nhinge - 1) + 1 + i]) for i in range(2)]
  chain_index = {}
  for j, cv in enumerate(chain):
    chain_index.setdefault(cv, j)

  for v in range(nvert):
    k = v if (v < len(chain) and chain[v] == v) else chain_index.get(v, -1)
    if k >= 0:
      c = min(k, len(chain) - 2)
      c0, c1 = chain[c], chain[c + 1]
      if c < elemnum and set(elem[c]) == {c0, c1}:
        res[v] = c
        continue
      e = pair_elem.get((min(c0, c1), max(c0, c1)), -1)
      if e >= 0:
        res[v] = e
        continue
    res[v] = first_elem.get(v, -1)
  return res


def _get_efm_data(mjm: mujoco.MjModel) -> mujoco.MjData | None:
  """Runs position kinematics under discrete integrator on a temp MjData for flex stiffness."""
  if mjm.nflex == 0:
    return None
  mjm_tmp = copy.copy(mjm)
  mjm_tmp.opt.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
  mjm_tmp.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  if mjm_tmp.nflexvert > 0:
    mjm_tmp.flex_vertbodyid[:] = np.where(
      mjm_tmp.flex_vertbodyid >= 0, mjm_tmp.body_weldid[mjm_tmp.flex_vertbodyid], mjm_tmp.flex_vertbodyid
    )
  for f in range(mjm_tmp.nflex):
    if not _flex_simple(mjm_tmp, f):
      mjm_tmp.flex_rigid[f] = True
  d = mujoco.MjData(mjm_tmp)
  mujoco.mj_fwdPosition(mjm_tmp, d)
  return d


def is_sparse(mjm: mujoco.MjModel) -> bool:
  if mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    if mjm.nv > 32:
      return True
    else:
      return False
  else:
    return bool(mujoco.mj_isSparse(mjm))


def _m_blocks(mjm: mujoco.MjModel):
  """The (start, size) diagonal blocks of M: the kinematic trees, each a contiguous dof range.

  M couples a dof only with its tree ancestors, so its diagonal blocks are exactly the trees.
  (dof_simplenum is not used to classify blocks: it is a contiguous-suffix run-length, so it
  misses interspersed decoupled dofs; the M_rownnz coupling check in m_block_layout catches those.)
  """
  return [(int(adr), int(num)) for adr, num in zip(mjm.tree_dofadr, mjm.tree_dofnum) if num > 0]


def m_block_layout(mjm: mujoco.MjModel) -> dict:
  """Per-block dense/sparse layout for M's diagonal blocks.

  Blocks use scalar Cholesky through six DOFs, tile Cholesky through M_BLOCK_DENSE_MAX, and sparse
  LDL beyond that. Compact diagonal blocks also use the scalar path without allocating a factor.
  """
  nv = mjm.nv
  blocks = _m_blocks(mjm)
  dof_adr = np.full(nv, types.Q_LD_BLOCK_SPARSE, dtype=np.int32)
  scalar_tiles = {}
  gather_tiles = {}
  off = 0
  for start, size in blocks:
    last = start + size - 1
    madr = int(mjm.M_rowadr[start])
    nnz = int(mjm.M_rowadr[last] + mjm.M_rownnz[last] - madr)
    compact = nnz == size
    triangular = nnz == size * (size + 1) // 2

    if size <= types.M_BLOCK_SCALAR_MAX and (compact or triangular):
      scalar_tiles.setdefault(size, []).append(start)
      if compact:
        dof_adr[start : start + size] = types.Q_LD_BLOCK_COMPACT
      else:
        dof_adr[start : start + size] = off
        off += size * size
    elif size <= types.M_BLOCK_DENSE_MAX:
      gather_tiles.setdefault(size, []).append(start)
      dof_adr[start : start + size] = off
      off += size * size
  for starts in scalar_tiles.values():
    starts.sort(key=lambda start: dof_adr[start] >= 0)
  return {
    "total": off,
    "dof_adr": dof_adr,
    "scalar_tiles": scalar_tiles,
    "gather_tiles": gather_tiles,
    "has_sparse": bool(np.any(dof_adr == types.Q_LD_BLOCK_SPARSE)),
  }


def put_model(mjm: mujoco.MjModel, batch_sizes: dict[str, int] | None = None) -> types.Model:
  """Creates a model on device.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    batch_sizes: Optional per-field leading batch sizes for `Model` fields whose
      array spec starts with `*`. Fields not listed here keep the default shared
      leading dimension of 1.

  Returns:
    The model containing kinematic and dynamic information (device).
  """
  # check for compatible cuda toolkit and driver versions
  warp_util.check_toolkit_driver()

  batch_sizes = batch_sizes or {}
  model_fields = {f.name: f.type for f in dataclasses.fields(types.Model) if warp_util.is_array_spec(f.type)}
  for name, size in batch_sizes.items():
    field_type = model_fields.get(name)
    spec_shape = getattr(field_type, "shape", ())
    if not spec_shape or spec_shape[0] != "*":
      raise ValueError(f"Model field {name!r} is not a batched array field.")
    if size < 1:
      raise ValueError(f"batch_sizes[{name!r}] must be positive, got {size}.")

  # model: check supported features in array types
  for field, field_type, mj_type in (
    (mjm.actuator_trntype, types.TrnType, mujoco.mjtTrn),
    (mjm.actuator_dyntype, types.DynType, mujoco.mjtDyn),
    (mjm.actuator_gaintype, types.GainType, mujoco.mjtGain),
    (mjm.actuator_biastype, types.BiasType, mujoco.mjtBias),
    (mjm.eq_type, types.EqType, mujoco.mjtEq),
    (mjm.geom_type, types.GeomType, mujoco.mjtGeom),
    (mjm.sensor_type, types.SensorType, mujoco.mjtSensor),
    (mjm.wrap_type, types.WrapType, mujoco.mjtWrap),
    (mjm.tree_sleep_policy, types.SleepPolicy, mujoco.mjtSleepPolicy),
  ):
    missing = ~np.isin(field, field_type)
    if missing.any():
      names = [mj_type(v).name for v in field[missing]]
      raise NotImplementedError(f"{names} not supported.")

  # opt: check supported features in scalar types
  for field, field_type, mj_type in (
    (mjm.opt.integrator, types.IntegratorType, mujoco.mjtIntegrator),
    (mjm.opt.cone, types.ConeType, mujoco.mjtCone),
    (mjm.opt.solver, types.SolverType, mujoco.mjtSolver),
  ):
    if field not in set(field_type):
      raise NotImplementedError(f"{mj_type(field).name} is unsupported.")

  # opt: check supported features in scalar flag types
  for field, field_type, mj_type in (
    (mjm.opt.disableflags, types.DisableBit, mujoco.mjtDisableBit),
    (mjm.opt.enableflags, types.EnableBit, mujoco.mjtEnableBit),
  ):
    unsupported = field & ~np.bitwise_or.reduce(field_type)
    if unsupported:
      raise NotImplementedError(f"{mj_type(unsupported).name} is unsupported.")

  if (mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and (mjm.eq_type == mujoco.mjtEq.mjEQ_FLEX).any():
    raise NotImplementedError("Flex equality constraints are not supported with sleeping enabled.")

  if mjm.opt.noslip_iterations > 0:
    if mjm.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE:
      raise NotImplementedError("noslip solver not implemented for the discrete integrator.")

  if (mjm.body_plugin != -1).any():
    raise NotImplementedError("Body plugins not supported.")

  if (mjm.actuator_plugin != -1).any():
    raise NotImplementedError("Actuator plugins not supported.")

  if (mjm.sensor_plugin != -1).any():
    raise NotImplementedError("Sensor plugins not supported.")

  # TODO(team): contact sensor linvel and angvel for flex contacts (contact Jacobian)
  contact_dataspec = mjm.sensor_intprm[mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT, 0]
  if mjm.nflex and (contact_dataspec & ((1 << types.ConDataField.LINVEL) | (1 << types.ConDataField.ANGVEL))).any():
    raise NotImplementedError("Contact sensor linvel and angvel not supported for models with flexes.")

  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      if abs(mjm.flex_interp[fi]) == 2:
        raise NotImplementedError("Quadratic flex interpolation (dof=quadratic) is not supported.")
      if mjm.flex_interp[fi] >= 0:
        continue
      bendingadr = mjm.flex_bendingadr[fi]
      if bendingadr < 0:
        continue
      nedge = int(mjm.flex_bending[bendingadr])
      if nedge > 0 and mjm.flex_damping[fi] > 0.0:
        warnings.warn("Bending damping is not yet supported for interpolated flex shells.")

  # array sizes may change in the future
  if mujoco.mjNPOLY != 2:
    warnings.warn(f"mujoco.mjNPOLY is {mujoco.mjNPOLY}, expected 2. Higher order polynomials may not be supported correctly.")

  # TODO(team): remove after _update_gradient for Newton uses tile operations for islands
  nv_max = 60
  if mjm.nv > nv_max and mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
    raise ValueError(f"Dense is unsupported for nv > {nv_max} (nv = {mjm.nv}).")

  # sleeping is supported via a dof-compaction approach.  awake dofs are compacted into dense
  # nvmax-sized arrays.  nvmax is chosen to fit the worst-case active dof set. sleeping is only
  # supported for Newton solver and requires nv <= nvmax.
  if (mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and mjm.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
    raise ValueError(f"sleeping requires the Newton solver (got solver={types.SolverType(mjm.opt.solver).name})")

  collision_sensors = (mujoco.mjtSensor.mjSENS_GEOMDIST, mujoco.mjtSensor.mjSENS_GEOMNORMAL, mujoco.mjtSensor.mjSENS_GEOMFROMTO)
  is_collision_sensor = np.isin(mjm.sensor_type, collision_sensors)

  def not_implemented(objtype, objid, geomtype):
    if objtype == mujoco.mjtObj.mjOBJ_BODY:
      geomnum = mjm.body_geomnum[objid]
      geomadr = mjm.body_geomadr[objid]
      for geomid in range(geomadr, geomadr + geomnum):
        if mjm.geom_type[geomid] == geomtype:
          return True
    elif objtype == mujoco.mjtObj.mjOBJ_GEOM:
      if mjm.geom_type[objid] == geomtype:
        return True
    return False

  def _check_friction(name: str, id_: int, condim: int, friction, checks):
    for min_condim, indices in checks:
      if condim >= min_condim:
        for idx in indices:
          if friction[idx] < types.MJ_MINMU:
            warnings.warn(
              f"{name} {id_}: friction[{idx}] ({friction[idx]}) < MJ_MINMU ({types.MJ_MINMU}) "
              f"with condim={condim} may cause NaN"
            )

  for geomid in range(mjm.ngeom):
    _check_friction("geom", geomid, mjm.geom_condim[geomid], mjm.geom_friction[geomid], [(3, [0]), (4, [1]), (6, [2])])

  for pairid in range(mjm.npair):
    _check_friction("pair", pairid, mjm.pair_dim[pairid], mjm.pair_friction[pairid], [(3, [0]), (4, [1, 2]), (6, [3, 4])])

  # create opt
  opt_kwargs = {f.name: getattr(mjm.opt, f.name, None) for f in dataclasses.fields(types.Option)}
  if hasattr(mjm.opt, "impratio"):
    opt_kwargs["impratio_invsqrt"] = 1.0 / np.sqrt(np.maximum(mjm.opt.impratio, mujoco.mjMINVAL))
  opt = types.Option(**opt_kwargs)

  # C MuJoCo tolerance was chosen for float64 architecture, but we default to float32 on GPU
  # adjust the tolerance for lower precision, to avoid the solver spending iterations needlessly
  # bouncing around the optimal solution
  opt.tolerance = max(opt.tolerance, 1e-6)

  # warp only fields
  opt.broadphase = types.BroadphaseType.NXN
  opt.broadphase_filter = types.BroadphaseFilter.PLANE | types.BroadphaseFilter.SPHERE | types.BroadphaseFilter.OBB
  opt.graph_conditional = True
  opt.run_collision_detection = True
  opt.warn_overflow = int(types.OverflowType.ALL)
  opt.deterministic = int(types.DeterminismType.NONE)
  opt.run_rne_postconstraint = False
  opt.run_qacc_smooth = True
  contact_sensor_maxmatch_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_NUMERIC, "contact_sensor_maxmatch")
  if contact_sensor_maxmatch_id > -1:
    opt.contact_sensor_maxmatch = mjm.numeric_data[mjm.numeric_adr[contact_sensor_maxmatch_id]]
  else:
    opt.contact_sensor_maxmatch = 64

  # place opt on device
  for f in dataclasses.fields(types.Option):
    if warp_util.is_array_spec(f.type):
      setattr(opt, f.name, _create_array(getattr(opt, f.name), f.type, {"*": 1}))
    else:
      setattr(opt, f.name, f.type(getattr(opt, f.name)))

  # create model
  m = types.Model(**{f.name: getattr(mjm, f.name, None) for f in dataclasses.fields(types.Model)})
  # TODO(team): remove version check once mujoco >= 3.16.1 is required.
  if not util_pkg.check_version("mujoco>=3.16.1"):
    m.eq_data = np.pad(mjm.eq_data, ((0, 0), (0, 1)))
  if not util_pkg.check_version("mujoco>=3.17.1"):
    if not hasattr(mjm, "nplastic"):
      m.nplastic = 0
    if not hasattr(mjm, "jnt_plasticadr"):
      m.jnt_plasticadr = np.full(mjm.njnt, -1, dtype=np.int32)
    if not hasattr(mjm, "tendon_plasticadr"):
      m.tendon_plasticadr = np.full(mjm.ntendon, -1, dtype=np.int32)
    if not hasattr(mjm, "flex_plasticadr"):
      m.flex_plasticadr = np.full(mjm.nflex, -1, dtype=np.int32)
    if not hasattr(mjm, "flex_plasticmode"):
      m.flex_plasticmode = np.zeros(mjm.nflex, dtype=np.int32)

  m.opt = opt
  m.callback = types.Callback()

  m.nv_pad = _get_padded_sizes(
    mjm.nv,
    0,
    is_sparse(mjm),
    types.TILE_SIZE_JTDAJ_SPARSE if is_sparse(mjm) else types.TILE_SIZE_JTDAJ_DENSE,
    augment_cholesky=mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON and mjm.nv > 32,
  )[1]
  m.nacttrnbody = (mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY).sum()
  m.nsensortaxel = mjm.mesh_vertnum[mjm.sensor_objid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE]].sum()
  # rangefinder ray counts (1 per site sensor, width * height per camera sensor)
  rf_sensors = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)[0]
  rf_nrays = np.array(
    [
      1 if mjm.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_SITE else mjm.cam_resolution[mjm.sensor_objid[i]].prod()
      for i in rf_sensors
    ],
    dtype=int,
  )
  m.nrangefinder = int(rf_nrays.sum())
  condim_arrays = [np.array([0]), mjm.geom_condim, mjm.pair_dim]
  if mjm.nflex > 0:
    condim_arrays.append(mjm.flex_condim)
    if (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any():
      raise NotImplementedError("Flex-SDF collision is not implemented.")
    if (mjm.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD).any():
      raise NotImplementedError("Flex-HField collision is not implemented.")
    if hasattr(mjm, "flex_internal") and (mjm.flex_internal != 0).any():
      raise NotImplementedError("Flex internal collisions are not implemented.")
    if (mjm.flex_rigid != 0).any():
      raise NotImplementedError("Rigid flexes are not implemented.")
    # TODO(team): skip element pairs driven by a common grid node, as in mjc_ElemSharedNode
    if np.any((mjm.flex_interp != 0) & (mjm.flex_selfcollide != 0) & ((mjm.flex_contype & mjm.flex_conaffinity) != 0)):
      raise NotImplementedError("Self-collision of interpolated flexes (dof=trilinear or quadratic) is not implemented.")
    if np.any((mjm.flex_interp != 0) & (mjm.flex_passive != 0) & (mjm.flex_dim >= 2)):
      raise NotImplementedError("Passive contact of interpolated flexes (dof=trilinear or quadratic) is not implemented.")
  m.nmaxcondim = np.concatenate(condim_arrays).max()
  m.nmaxpyramid = np.maximum(1, 2 * (m.nmaxcondim - 1))
  m.has_sdf_geom = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  m.has_ellipsoid_geom = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID).any()
  m.has_plane_geom = bool(mjm.ngeom > 0 and (mjm.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any())
  m.has_flex_selfcollide = bool(
    mjm.nflex > 0 and np.any((mjm.flex_selfcollide != 0) & ((mjm.flex_contype & mjm.flex_conaffinity) != 0))
  )
  m.has_flex_passive = bool(
    mjm.nflex > 0 and np.any((mjm.flex_passive != 0) & (mjm.flex_rigid == 0) & (mjm.flex_interp == 0) & (mjm.flex_dim >= 2))
  )
  m.has_flex_snh = bool(
    types.FLEX_STIFFNESS_3D == 24
    and any(
      mjm.flex_interp[f] == 0
      and mjm.flex_dim[f] == 3
      and mjm.flex_stiffnessadr[f] >= 0
      and mjm.flex_elemnum[f] > 0
      and mjm.flex_stiffness[mjm.flex_stiffnessadr[f] + 21] != 0
      for f in range(mjm.nflex)
    )
  )
  if m.has_flex_snh and mjm.opt.integrator != mujoco.mjtIntegrator.mjINT_DISCRETE:
    raise ValueError("stable Neo-Hookean elasticity requires integrator='discrete'")
  flex_simple = np.array([_flex_simple(mjm, f) for f in range(mjm.nflex)], dtype=bool)
  m.flex_simple = flex_simple
  m.has_non_simple_flex = bool(
    mjm.nflex > 0
    and any(
      mjm.flex_interp[f] == 0
      and not mjm.flex_rigid[f]
      and not flex_simple[f]
      and (
        (mjm.flex_dim[f] == 1 and mjm.flex_bendingadr[f] >= 0)
        or (
          mjm.flex_dim[f] >= 2
          and (
            mjm.flex_bendingadr[f] >= 0 or (mjm.flex_stiffnessadr[f] >= 0 and mjm.flex_stiffness[mjm.flex_stiffnessadr[f]] != 0)
          )
        )
      )
      for f in range(mjm.nflex)
    )
  )
  m.has_flex_edge_stiff_damp = bool(
    any(not mjm.flex_rigid[f] and (mjm.flex_edgestiffness[f] > 0 or mjm.flex_edgedamping[f] > 0) for f in range(mjm.nflex))
  )
  if (
    mjm.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
    and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    and m.has_non_simple_flex
  ):
    raise ValueError("discrete integrator: flex with general attachments requires solver='CG'")
  m.has_tendon_stiffness = bool(mjm.ntendon > 0 and (np.any(m.tendon_stiffness != 0) or np.any(m.tendon_stiffnesspoly != 0)))
  tendon_actuator = mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
  m.has_tendon_damping = bool(
    mjm.ntendon > 0
    and (
      np.any(m.tendon_damping != 0)
      or np.any(m.tendon_dampingpoly != 0)
      or np.any(mjm.actuator_damping[tendon_actuator] != 0)
      or np.any(mjm.actuator_dampingpoly[tendon_actuator] != 0)
    )
  )
  m.has_efm_actuator = bool(
    mjm.nu > 0
    and np.any(
      (
        (mjm.actuator_gaintype == mujoco.mjtGain.mjGAIN_AFFINE)
        & ((mjm.actuator_gainprm[:, 1] != 0) | (mjm.actuator_gainprm[:, 2] != 0))
      )
      | (
        (mjm.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        & ((mjm.actuator_biasprm[:, 1] != 0) | (mjm.actuator_biasprm[:, 2] != 0))
      )
      | (mjm.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
      | (mjm.actuator_biastype == mujoco.mjtBias.mjBIAS_MUSCLE)
      | (mjm.actuator_gaintype == mujoco.mjtGain.mjGAIN_DCMOTOR)
      | (mjm.actuator_biastype == mujoco.mjtBias.mjBIAS_DCMOTOR)
      | (mjm.actuator_gaintype == mujoco.mjtGain.mjGAIN_SO3)
      | (mjm.actuator_biastype == mujoco.mjtBias.mjBIAS_SO3)
    )
  )
  has_actuator_wrap = False
  for i in range(mjm.nactuator):
    dyntype = int(mjm.actuator_dyntype[i])
    if dyntype in (
      int(mujoco.mjtDyn.mjDYN_INTEGRATOR),
      int(mujoco.mjtDyn.mjDYN_FILTER),
      int(mujoco.mjtDyn.mjDYN_FILTEREXACT),
    ):
      gaintype = int(mjm.actuator_gaintype[i])
      biastype = int(mjm.actuator_biastype[i])
      if (
        gaintype == int(mujoco.mjtGain.mjGAIN_FIXED)
        and biastype == int(mujoco.mjtBias.mjBIAS_AFFINE)
        and mjm.actuator_gainprm[i, 0] == -mjm.actuator_biasprm[i, 1]
      ):
        trntype = int(mjm.actuator_trntype[i])
        if trntype in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
          jntid = int(mjm.actuator_trnid[i, 0])
          jnt_type = int(mjm.jnt_type[jntid])
          if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL) and np.any(mjm.actuator_gear[i, :3] != 0.0):
            has_actuator_wrap = True
            break
        elif (
          trntype == int(mujoco.mjtTrn.mjTRN_SITE)
          and mjm.actuator_trnid[i, 1] >= 0
          and np.all(mjm.actuator_gear[i, :3] == 0.0)
          and np.any(mjm.actuator_gear[i, 3:6] != 0.0)
        ):
          has_actuator_wrap = True
          break
  m.has_actuator_wrap = has_actuator_wrap
  m.has_actuator_dcmotor = bool(mjm.nactuator > 0 and np.any(mjm.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_DCMOTOR)))
  m.has_1d_flex = bool(mjm.nflex > 0 and np.any(mjm.flex_dim == 1))
  m.has_2d_flex = bool(mjm.nflex > 0 and np.any(mjm.flex_dim == 2))
  m.has_3d_flex = bool(mjm.nflex > 0 and np.any(mjm.flex_dim == 3))
  m.max_flex_dim = int(np.max(mjm.flex_dim)) if mjm.nflex > 0 else 0
  m.block_dim = types.BlockDim()
  # Derive CG solver block_dim from nv: clamp(round_up_to_32(nv), 32, 256)
  _nv_block = max(32, min(256, ((mjm.nv + 31) // 32) * 32))
  m.block_dim.update_gradient_grad = _nv_block
  m.block_dim.solve_beta_accumulate = _nv_block
  m.block_dim.solve_search_update_cg = _nv_block
  m.block_dim.solve_init_search_cg = _nv_block
  if mjm.nv > 500:
    m.block_dim.linesearch_iterative = 256
  m.is_sparse = is_sparse(mjm)
  m.has_fluid = bool(mjm.opt.wind.any() or mjm.opt.density > 0 or mjm.opt.viscosity > 0)
  m.nflexintcell = _get_nflexintcell(mjm)
  efm_d = _get_efm_data(mjm)
  m.nefmK = int(efm_d.nefmK) if efm_d else 0
  m.efm_K_rownnz = efm_d.efm_K_rownnz if efm_d else np.zeros(mjm.nv, dtype=np.int32)
  m.efm_K_rowadr = efm_d.efm_K_rowadr if efm_d else np.zeros(mjm.nv, dtype=np.int32)
  m.efm_K_colind = efm_d.efm_K_colind if (efm_d and efm_d.efm_K_colind is not None) else np.zeros(0, dtype=np.int32)
  has_stretch_or_interp = any(
    not mjm.flex_rigid[f]
    and mjm.flex_edgeequality[f] != 3
    and (
      _flex_has_stiffness(mjm, f)
      if mjm.flex_interp[f] != 0
      else (
        mjm.flex_stiffnessadr[f] >= 0
        and np.any(
          mjm.flex_stiffness[
            mjm.flex_stiffnessadr[f] : mjm.flex_stiffnessadr[f]
            + (types.FLEX_STIFFNESS_3D if mjm.flex_dim[f] == 3 else 21) * mjm.flex_elemnum[f]
          ]
          != 0
        )
      )
    )
    for f in range(mjm.nflex)
  )
  m.efm0_active = bool(mjm.nefm0dof > 0 and not has_stretch_or_interp and not m.has_flex_passive and not m.has_non_simple_flex)
  m.nefmdof = int(efm_d.nefmdof) if (efm_d and not m.efm0_active) else 0
  m.efm_dofid = (
    efm_d.efm_dofid
    if (efm_d and not m.efm0_active and efm_d.efm_dofid is not None)
    else np.zeros(0, dtype=np.int32)
  )

  flex_interp_assemblable = True
  for fi in range(mjm.nflex):
    if mjm.flex_interp[fi] != 0:
      if not mjm.flex_rigid[fi] and mjm.flex_edgeequality[fi] != 3 and _flex_has_stiffness(mjm, fi):
        nodeadr = int(mjm.flex_nodeadr[fi])
        for n in range(int(mjm.flex_nodenum[fi])):
          bid = int(mjm.flex_nodebodyid[nodeadr + n])
          dofnum = int(mjm.body_dofnum[bid])
          if dofnum != 0 and (int(mjm.body_simple[bid]) != 2 or dofnum != 3):
            flex_interp_assemblable = False
            break
  m.flex_interp_assemblable = flex_interp_assemblable
  m.has_unsupported_flex_interp = bool(any(int(x) < -1 or int(x) >= 2 for x in mjm.flex_interp))

  # Precompute flex_cell_map
  flex_cell_map = []
  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      order = int(mjm.flex_interp[fi])
      if order == 0 or mjm.flex_edgeequality[fi] == 3 or not _flex_has_stiffness(mjm, fi):
        continue
      cx, cy, cz = (int(x) for x in mjm.flex_cellnum[fi])
      if order < 0:
        nfaces = 2 * (cy * cz + cx * cz + cx * cy)
        for fe in range(nfaces):
          flex_cell_map.append((fi, fe, 0, 0))
      else:
        for ci in range(cx):
          for cj in range(cy):
            for ck in range(cz):
              flex_cell_map.append((fi, ci, cj, ck))
  if not flex_cell_map:
    m.flex_cell_map = np.zeros((0, 4), dtype=np.int32)
  else:
    m.flex_cell_map = np.array(flex_cell_map, dtype=np.int32)

  m.max_ten_J_rownnz = int(mjm.ten_J_rownnz.max()) if mjm.ntendon else 0

  # body ids grouped by tree level (depth-based traversal)
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  m.body_tree = tuple(wp.array(bodies[i], dtype=int) for i in sorted(bodies))

  # Subtree sizes include the body itself; parents precede their descendants.
  m.body_subtreenum = np.ones(mjm.nbody, dtype=int)
  for i in reversed(range(1, mjm.nbody)):
    m.body_subtreenum[mjm.body_parentid[i]] += m.body_subtreenum[i]

  # branch-based traversal data
  children_count = np.bincount(mjm.body_parentid[1:], minlength=mjm.nbody)
  ancestor_chain = lambda b: ancestor_chain(mjm.body_parentid[b]) + [b] if b else []
  branches = [ancestor_chain(l) for l in np.where(children_count[1:] == 0)[0] + 1]
  m.nbranch = len(branches)

  body_branches = []
  body_branch_start = []
  offset = 0

  for branch in branches:
    body_branches.extend(branch)
    body_branch_start.append(offset)
    offset += len(branch)
  body_branch_start.append(offset)

  m.body_branches = np.array(body_branches, dtype=int)
  m.body_branch_start = np.array(body_branch_start, dtype=int)

  m.mocap_bodyid = np.arange(mjm.nbody)[mjm.body_mocapid >= 0]
  m.mocap_bodyid = m.mocap_bodyid[mjm.body_mocapid[mjm.body_mocapid >= 0].argsort()]
  m.body_fluid_ellipsoid = np.zeros(mjm.nbody, dtype=bool)
  has_fluid = mjm.geom_fluid.reshape(mjm.ngeom, mujoco.mjNFLUID)[:, 0] > 0
  if np.any(has_fluid):
    fluid_bodyids = mjm.geom_bodyid[has_fluid]
    valid_mass = mjm.body_mass[fluid_bodyids] >= types.MJ_MINVAL
    m.body_fluid_ellipsoid[fluid_bodyids[valid_mass]] = True
  m.body_fluid_ellipsoid_adr = np.nonzero(m.body_fluid_ellipsoid)[0]
  body_is_free = np.zeros(mjm.nbody, dtype=bool)
  has_one_jnt = mjm.body_jntnum == 1
  if np.any(has_one_jnt):
    b_indices = np.nonzero(has_one_jnt)[0]
    jnt_adrs = mjm.body_jntadr[b_indices]
    is_free_jnt = mjm.jnt_type[jnt_adrs] == mujoco.mjtJoint.mjJNT_FREE
    if np.any(is_free_jnt):
      b_free = b_indices[is_free_jnt]
      jnt_free_adrs = jnt_adrs[is_free_jnt]
      dof_adrs = mjm.jnt_dofadr[jnt_free_adrs]
      tree_ids = mjm.dof_treeid[dof_adrs]
      tree_dof6 = mjm.tree_dofnum[tree_ids] == 6
      body_is_free[b_free] = tree_dof6
  m.body_is_free = body_is_free
  m.body_freeadr = np.nonzero(m.body_is_free)[0]
  body_fluid_box = np.zeros(mjm.nbody, dtype=bool)
  for b in range(1, mjm.nbody):
    if not m.body_fluid_ellipsoid[b] and mjm.body_mass[b] >= types.MJ_MINVAL:
      body_fluid_box[b] = True
  m.body_fluid_box_adr = np.nonzero(body_fluid_box)[0]
  jnt_limited_slide_hinge = mjm.jnt_limited & np.isin(mjm.jnt_type, (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE))
  m.jnt_limited_slide_hinge_adr = np.nonzero(jnt_limited_slide_hinge)[0]
  m.jnt_limited_ball_adr = np.nonzero(mjm.jnt_limited & (mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL))[0]
  m.dof_tri_row, m.dof_tri_col = np.triu_indices(mjm.nv)

  # precompute body_isdofancestor: which DOFs affect each body
  # TODO: Investigate alternative approach such as bitmap
  body_isdofancestor = np.zeros((mjm.nbody, m.nv_pad), dtype=np.int32)
  for bodyid in range(mjm.nbody):
    b = bodyid
    while b > 0 and mjm.body_dofnum[b] == 0:
      b = mjm.body_parentid[b]
    if mjm.body_dofnum[b] == 0:
      continue
    dofid = mjm.body_dofadr[b] + mjm.body_dofnum[b] - 1
    while dofid >= 0:
      body_isdofancestor[bodyid, dofid] = 1
      dofid = mjm.dof_parentid[dofid]
  m.body_isdofancestor = body_isdofancestor

  # precalculated geom pairs
  filterparent = not (mjm.opt.disableflags & types.DisableBit.FILTERPARENT)

  geom1, geom2 = np.triu_indices(mjm.ngeom, k=1)
  m.nxn_geom_pair = np.stack((geom1, geom2), axis=1)

  bodyid1 = mjm.geom_bodyid[geom1]
  bodyid2 = mjm.geom_bodyid[geom2]
  contype1 = mjm.geom_contype[geom1]
  contype2 = mjm.geom_contype[geom2]
  conaffinity1 = mjm.geom_conaffinity[geom1]
  conaffinity2 = mjm.geom_conaffinity[geom2]
  weldid1 = mjm.body_weldid[bodyid1]
  weldid2 = mjm.body_weldid[bodyid2]
  weld_parentid1 = mjm.body_weldid[mjm.body_parentid[weldid1]]
  weld_parentid2 = mjm.body_weldid[mjm.body_parentid[weldid2]]

  self_collision = weldid1 == weldid2
  # both weld bodies dof-less: no forces can act, skip (matches filterBodyPair)
  no_dofs = (mjm.body_dofnum[weldid1] == 0) & (mjm.body_dofnum[weldid2] == 0)
  parent_child_collision = (
    filterparent & (weldid1 != 0) & (weldid2 != 0) & ((weldid1 == weld_parentid2) | (weldid2 == weld_parentid1))
  )
  mask = np.array((contype1 & conaffinity2) | (contype2 & conaffinity1), dtype=bool)
  exclude = np.isin((bodyid1 << 16) + bodyid2, mjm.exclude_signature)

  nxn_pairid_contact = -1 * np.ones(len(geom1), dtype=int)
  nxn_pairid_contact[~(mask & ~self_collision & ~no_dofs & ~parent_child_collision & ~exclude)] = -2

  # contact pairs
  def upper_tri_index(n, i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * n - i - 3)) // 2 + j - 1

  for i in range(mjm.npair):
    nxn_pairid_contact[upper_tri_index(mjm.ngeom, mjm.pair_geom1[i], mjm.pair_geom2[i])] = i

  sensor_collision_adr = np.nonzero(is_collision_sensor)[0]
  collision_sensor_adr = np.full(mjm.nsensor, -1)
  collision_sensor_adr[sensor_collision_adr] = np.arange(len(sensor_collision_adr))

  nxn_pairid_collision = -1 * np.ones(len(geom1), dtype=int)
  pairids = []
  sensor_collision_start_adr = []
  for i in range(sensor_collision_adr.size):
    sensorid = sensor_collision_adr[i]
    objtype = mjm.sensor_objtype[sensorid]
    objid = mjm.sensor_objid[sensorid]
    reftype = mjm.sensor_reftype[sensorid]
    refid = mjm.sensor_refid[sensorid]

    # get lists of geoms to collide
    if objtype == types.ObjType.BODY:
      n1 = mjm.body_geomnum[objid]
      id1 = mjm.body_geomadr[objid]
    else:
      n1 = 1
      id1 = objid
    if reftype == types.ObjType.BODY:
      n2 = mjm.body_geomnum[refid]
      id2 = mjm.body_geomadr[refid]
    else:
      n2 = 1
      id2 = refid

    # collide all pairs
    for geom1id in range(id1, id1 + n1):
      for geom2id in range(id2, id2 + n2):
        pairid = upper_tri_index(mjm.ngeom, geom1id, geom2id)

        if pairid in pairids:
          sensor_collision_start_adr.append(nxn_pairid_collision[pairid])
        else:
          npairids = len(pairids)
          nxn_pairid_collision[pairid] = npairids
          sensor_collision_start_adr.append(npairids)
          pairids.append(pairid)

  m.nsensorcollision = (nxn_pairid_collision >= 0).sum()
  m.sensor_collision_start_adr = np.array(sensor_collision_start_adr)
  nxn_include = (nxn_pairid_contact > -2) | (nxn_pairid_collision >= 0)

  if nxn_include.sum() < 250_000:
    opt.broadphase = types.BroadphaseType.NXN
  elif mjm.ngeom < 1000:
    opt.broadphase = types.BroadphaseType.SAP_TILE
  else:
    opt.broadphase = types.BroadphaseType.SAP_SEGMENTED

  m.nxn_geom_pair_filtered = m.nxn_geom_pair[nxn_include]
  m.nxn_pairid = np.hstack([nxn_pairid_contact.reshape((-1, 1)), nxn_pairid_collision.reshape((-1, 1))])
  m.nxn_pairid_filtered = m.nxn_pairid[nxn_include]

  # count contact pair types
  def geom_trid_index(i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * len(types.GeomType) - i - 1)) // 2 + j

  m.geom_pair_type_count = tuple(
    np.bincount(
      [geom_trid_index(mjm.geom_type[geom1[i]], mjm.geom_type[geom2[i]]) for i in np.arange(len(geom1)) if nxn_include[i]],
      minlength=len(types.GeomType) * (len(types.GeomType) + 1) // 2,
    )
  )

  # filter plugins for only geom plugins, drop the rest
  m.plugin, m.plugin_attr = [], []
  m.geom_plugin_index = np.full_like(mjm.geom_type, -1)

  for i in range(len(mjm.geom_plugin)):
    if mjm.geom_plugin[i] == -1:
      continue
    p = mjm.geom_plugin[i]
    m.geom_plugin_index[i] = len(m.plugin)
    m.plugin.append(mjm.plugin[p])
    start = mjm.plugin_attradr[p]
    end = mjm.plugin_attradr[p + 1] if p + 1 < mjm.nplugin else len(mjm.plugin_attr)
    values = mjm.plugin_attr[start:end]
    attr_values = []
    current = []
    for v in values:
      if v == 0:
        if current:
          s = "".join(chr(int(x)) for x in current)
          attr_values.append(float(s))
          current = []
      else:
        current.append(v)
    if len(attr_values) > types._NPLUGINATTR:
      raise ValueError(f"Plugin has {len(attr_values)} attributes, which exceeds the maximum of {types._NPLUGINATTR}. ")
    # pad with zeros to _NPLUGINATTR
    attr_values += [0.0] * (types._NPLUGINATTR - len(attr_values))
    m.plugin_attr.append(attr_values[: types._NPLUGINATTR])

  # equality constraint addresses
  m.eq_connect_adr = np.nonzero(mjm.eq_type == types.EqType.CONNECT)[0]
  m.eq_wld_adr = np.nonzero(mjm.eq_type == types.EqType.WELD)[0]
  m.eq_jnt_adr = np.nonzero(mjm.eq_type == types.EqType.JOINT)[0]
  m.eq_ten_adr = np.nonzero(mjm.eq_type == types.EqType.TENDON)[0]
  m.eq_flex_adr = np.nonzero(mjm.eq_type == types.EqType.FLEX)[0]
  m.eq_flexstrain_adr = np.nonzero(mjm.eq_type == types.EqType.FLEXSTRAIN)[0]
  m.neq_flexstrain = m.eq_flexstrain_adr.size
  # Precompute flex strain Jacobian sparsity pattern
  flexstrain_J_rownnz = []
  flexstrain_J_colind = []

  if m.neq_flexstrain > 0:
    for eqstrainid, eqid in enumerate(m.eq_flexstrain_adr):
      f = int(mjm.eq_obj1id[eqid])
      order = int(mjm.flex_interp[f])
      order_abs = abs(order)
      ci = int(mjm.eq_data[eqid, 0])
      cj = int(mjm.eq_data[eqid, 1])
      ck = int(mjm.eq_data[eqid, 2])

      cellnum = mjm.flex_cellnum[f]
      cy = cellnum[1]
      cz = cellnum[2]
      nstart = mjm.flex_nodeadr[f]
      ny_g = cy * order_abs + 1
      nz_g = cz * order_abs + 1

      if order < 0:
        # Shell mode: 2D bilinear quad
        npc = (order_abs + 1) * (order_abs + 1)
        node_bodies = []
        for idx in range(npc):
          gidx = support.gather_face_node_index(int(cellnum[0]), int(cy), int(cz), ci, idx, order_abs)
          node_bodies.append(mjm.flex_nodebodyid[nstart + gidx])
      else:
        # Solid mode: 3D trilinear voxel
        node_bodies = [
          mjm.flex_nodebodyid[
            nstart + (ci * order_abs + li) * ny_g * nz_g + (cj * order_abs + lj) * nz_g + (ck * order_abs + lk)
          ]
          for li in range(order_abs + 1)
          for lj in range(order_abs + 1)
          for lk in range(order_abs + 1)
        ]

      active_dof_mask = np.any(body_isdofancestor[node_bodies, :] != 0, axis=0)
      sorted_dofs = np.nonzero(active_dof_mask)[0].tolist()
      flexstrain_J_rownnz.append(len(sorted_dofs))
      flexstrain_J_colind.extend(sorted_dofs)

    m.flexstrain_J_rownnz = np.array(flexstrain_J_rownnz, dtype=np.int32)
    m.flexstrain_J_colind = np.array(flexstrain_J_colind, dtype=np.int32)
    m.flexstrain_J_rowadr = np.cumsum([0] + flexstrain_J_rownnz[:-1], dtype=np.int32)
  else:
    m.flexstrain_J_rownnz = np.zeros((0,), dtype=np.int32)
    m.flexstrain_J_rowadr = np.zeros((0,), dtype=np.int32)
    m.flexstrain_J_colind = np.zeros((0,), dtype=np.int32)

  m.nJfs = m.flexstrain_J_colind.size

  # tendon type: older mujoco versions imply it from the first wrap object
  if not util_pkg.check_version("mujoco>=3.17.1"):
    is_fixed = mjm.wrap_type[mjm.tendon_adr] == mujoco.mjtWrap.mjWRAP_JOINT
    m.tendon_type = np.where(is_fixed, types.TendonType.FIXED, types.TendonType.SPATIAL)

  # fixed tendon
  m.tendon_jnt_adr, m.wrap_jnt_adr = [], []
  for i in range(mjm.ntendon):
    adr = mjm.tendon_adr[i]
    if m.tendon_type[i] == types.TendonType.FIXED:
      tendon_num = mjm.tendon_num[i]
      for j in range(tendon_num):
        m.tendon_jnt_adr.append(i)
        m.wrap_jnt_adr.append(adr + j)

  # spatial tendon
  m.tendon_site_pair_adr, m.tendon_geom_adr = [], []
  m.ten_wrapadr_site, m.ten_wrapnum_site = [0], []
  for i, tendon_num in enumerate(mjm.tendon_num):
    adr = mjm.tendon_adr[i]
    # sites
    if (mjm.wrap_type[adr : adr + tendon_num] == mujoco.mjtWrap.mjWRAP_SITE).all():
      if i < mjm.ntendon:
        m.ten_wrapadr_site.append(m.ten_wrapadr_site[-1] + tendon_num)
      m.ten_wrapnum_site.append(tendon_num)
    else:
      if i < mjm.ntendon:
        m.ten_wrapadr_site.append(m.ten_wrapadr_site[-1])
      m.ten_wrapnum_site.append(0)

    # geoms
    for j in range(tendon_num):
      wrap_type = mjm.wrap_type[adr + j]
      if j < tendon_num - 1:
        next_wrap_type = mjm.wrap_type[adr + j + 1]
        if wrap_type == mujoco.mjtWrap.mjWRAP_SITE and next_wrap_type == mujoco.mjtWrap.mjWRAP_SITE:
          m.tendon_site_pair_adr.append(i)
      if wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE or wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER:
        m.tendon_geom_adr.append(i)

  m.tendon_limited_adr = np.nonzero(mjm.tendon_limited)[0]
  m.wrap_site_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_SITE)[0]
  m.wrap_site_pair_adr = np.setdiff1d(m.wrap_site_adr[np.nonzero(np.diff(m.wrap_site_adr) == 1)[0]], mjm.tendon_adr[1:] - 1)
  m.wrap_geom_adr = np.nonzero(np.isin(mjm.wrap_type, [mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER]))[0]

  # pulley scaling
  m.wrap_pulley_scale = np.ones(mjm.nwrap, dtype=float)
  pulley_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_PULLEY)[0]
  for tadr, tnum in zip(mjm.tendon_adr, mjm.tendon_num):
    for padr in pulley_adr:
      if tadr <= padr < tadr + tnum:
        m.wrap_pulley_scale[padr : tadr + tnum] = 1.0 / mjm.wrap_prm[padr]

  m.actuator_trntype_body_adr = np.nonzero(mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY)[0]

  # sensor addresses
  is_contact = mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT
  m.sensor_pos_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITPOS)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITPOS)
    & ~is_contact
  )[0]
  m.sensor_limitpos_adr = np.nonzero(
    (mjm.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITPOS) | (mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONLIMITPOS)
  )[0]
  m.sensor_vel_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITVEL)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITVEL)
    & ~is_contact
  )[0]
  m.sensor_limitvel_adr = np.nonzero(
    (mjm.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITVEL) | (mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONLIMITVEL)
  )[0]
  acc_excluded_sensors = (
    mujoco.mjtSensor.mjSENS_TOUCH,
    mujoco.mjtSensor.mjSENS_CONTACT,
    mujoco.mjtSensor.mjSENS_JOINTLIMITFRC,
    mujoco.mjtSensor.mjSENS_TENDONLIMITFRC,
    mujoco.mjtSensor.mjSENS_TENDONACTFRC,
  )
  m.sensor_acc_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC) & ~np.isin(mjm.sensor_type, acc_excluded_sensors)
  )[0]
  m.sensor_stage_pos_adr = np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS)[0]
  m.sensor_stage_vel_adr = np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL)[0]
  m.sensor_stage_acc_adr = np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC)[0]
  # map each ray to its sensor id and each sensor to its starting ray index
  m.sensor_rangefinder_adr = np.repeat(rf_sensors, rf_nrays)
  m.rangefinder_sensor_adr = np.full(mjm.nsensor, -1)
  m.rangefinder_sensor_adr[rf_sensors] = np.cumsum(rf_nrays) - rf_nrays
  m.collision_sensor_adr = np.full(mjm.nsensor, -1)
  m.collision_sensor_adr[sensor_collision_adr] = np.arange(len(sensor_collision_adr))
  m.sensor_touch_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)[0]
  limitfrc_sensors = (mujoco.mjtSensor.mjSENS_JOINTLIMITFRC, mujoco.mjtSensor.mjSENS_TENDONLIMITFRC)
  m.sensor_limitfrc_adr = np.nonzero(np.isin(mjm.sensor_type, limitfrc_sensors))[0]
  m.sensor_e_potential = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_E_POTENTIAL).any()
  m.sensor_e_kinetic = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_E_KINETIC).any()
  m.sensor_tendonactfrc_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONACTFRC)[0]
  subtreevel_sensors = (mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtSensor.mjSENS_SUBTREEANGMOM)
  m.sensor_subtree_vel = np.isin(mjm.sensor_type, subtreevel_sensors).any()
  m.sensor_contact_pos_adr = np.nonzero(is_contact & (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS))[0]
  m.sensor_contact_vel_adr = np.nonzero(is_contact & (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL))[0]
  m.sensor_contact_acc_adr = np.nonzero(is_contact & (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC))[0]
  m.sensor_rne_postconstraint = np.isin(
    mjm.sensor_type,
    [
      mujoco.mjtSensor.mjSENS_ACCELEROMETER,
      mujoco.mjtSensor.mjSENS_FORCE,
      mujoco.mjtSensor.mjSENS_TORQUE,
      mujoco.mjtSensor.mjSENS_FRAMELINACC,
      mujoco.mjtSensor.mjSENS_FRAMEANGACC,
    ],
  ).any()
  # exclude attached site or camera body from ray intersection
  rf_bodyids = np.array(
    [
      mjm.site_bodyid[mjm.sensor_objid[i]]
      if mjm.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_SITE
      else mjm.cam_bodyid[mjm.sensor_objid[i]]
      for i in rf_sensors
    ],
    dtype=int,
  )
  m.sensor_rangefinder_bodyid = np.repeat(rf_bodyids, rf_nrays)
  m.taxel_vertadr = [
    j + mjm.mesh_vertadr[mjm.sensor_objid[i]]
    for i in range(mjm.nsensor)
    if mjm.sensor_type[i] == mujoco.mjtSensor.mjSENS_TACTILE
    for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
  ]
  m.taxel_sensorid = [
    i
    for i in range(mjm.nsensor)
    if mjm.sensor_type[i] == mujoco.mjtSensor.mjSENS_TACTILE
    for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
  ]
  tactile_geomid = mjm.sensor_refid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE]
  tactile_weldid = mjm.body_weldid[mjm.geom_bodyid[tactile_geomid]]
  unique_tactile_welds = np.unique(tactile_weldid)
  m.ntactileweld = int(len(unique_tactile_welds))
  weld_tactile_id = np.full(mjm.nbody, -1, dtype=np.int32)
  for idx, weld in enumerate(unique_tactile_welds):
    weld_tactile_id[weld] = idx
  m.weld_tactile_id = weld_tactile_id

  # Per-block scalar/tile/sparse layout (see m_block_layout).
  _lay = m_block_layout(mjm)
  m.qLD_block_total = _lay["total"]  # packed dense region length / offset of the LDL region
  m.qLD_block_adr = _lay["dof_adr"]

  scalar_tiles = [
    types.TileSet(
      adr=wp.array(_lay["scalar_tiles"][size], dtype=int),
      size=size,
    )
    for size in sorted(_lay["scalar_tiles"])
  ]
  gather_tiles = [
    types.TileSet(adr=wp.array(_lay["gather_tiles"][size], dtype=int), size=size) for size in sorted(_lay["gather_tiles"])
  ]
  m.M_tiles = tuple(scalar_tiles + gather_tiles)

  # Group sparse LDL updates by tree depth. Block-path DOFs never touch the LDL region.
  sparse_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1

  for k in range(mjm.nv):
    # skip diagonal rows
    if mjm.M_rownnz[k] == 1:
      continue
    dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
    if _lay["dof_adr"][k] != types.Q_LD_BLOCK_SPARSE:
      continue
    i = mjm.dof_parentid[k]
    diag_k = mjm.M_rowadr[k] + mjm.M_rownnz[k] - 1
    Madr_ki = diag_k - 1
    while i > -1:
      sparse_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
      i = mjm.dof_parentid[i]
      Madr_ki -= 1
  m.qLD_updates = tuple(wp.array(sparse_updates[i], dtype=wp.vec3i) for i in sorted(sparse_updates))

  # Build concatenated updates for fused kernel
  all_updates_flat = []
  level_offsets = [0]
  for level in sorted(sparse_updates):
    all_updates_flat.extend(sparse_updates[level])
    level_offsets.append(len(all_updates_flat))
  m.qLD_all_updates = all_updates_flat if all_updates_flat else [(0, 0, 0)]
  m.qLD_level_offsets = level_offsets

  # Indices for sparse M_fullm (used in solver). M_fullm_i/j are built by
  # walking dof_parentid for each dof, so for joint types whose internal block
  # MuJoCo stores diagonal-only in the compact (M_rownnz, M_rowadr) layout
  # (e.g. free joints), the chain-aware layout here has more entries per row
  # than the compact layout.
  m.M_fullm_i, m.M_fullm_j = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      m.M_fullm_i.append(i)
      m.M_fullm_j.append(j)
      j = mjm.dof_parentid[j]
  # M_elemid maps (row, col) -> madr index in the native CSR M layout
  M_elemid = np.full((mjm.nv, mjm.nv), -1, dtype=np.int32)
  # M_hinit_i: row index of each CSR entry (its madr is the flat index). The dense Newton H-init
  # uses (M_hinit_i, M_colind) to scatter M's upper triangle into the dense H tile from CSR.
  M_hinit_i = np.zeros(mjm.nC, dtype=np.int32)
  for i in range(mjm.nv):
    rowadr = mjm.M_rowadr[i]
    rownnz = mjm.M_rownnz[i]
    for k in range(rownnz):
      madr = rowadr + k
      col = int(mjm.M_colind[madr])
      M_elemid[i, col] = madr
      M_hinit_i[madr] = i
  m.M_elemid = M_elemid
  m.M_hinit_i = M_hinit_i

  # Precompute per-block gather indices for the dense-block densify (tile_load_indexed). For each
  # dense block and flat slot (row, col), store the CSR address of M[max(i,j), min(i,j)], or nC
  # (out of bounds -> read as 0) for structurally absent pairs. Laid out [block, slot] so the kernel
  # reads slice (block_size^2,) at offset blk * block_size^2.
  for tile in gather_tiles:
    sz = tile.size
    starts = np.array(_lay["gather_tiles"][sz], dtype=np.int32)
    dofs = starts[:, None] + np.arange(sz)[None, :]  # (nblock, sz) global dof per block row
    gi = dofs[:, :, None]  # (nblock, sz, 1)
    gj = dofs[:, None, :]  # (nblock, 1, sz)
    elemid = M_elemid[np.maximum(gi, gj), np.minimum(gi, gj)]  # (nblock, sz, sz), -1 if absent
    elemid = np.where(elemid >= 0, elemid, mjm.nC)
    tile.elemid = wp.array(elemid.reshape(-1).astype(np.int32), dtype=int)

  # indices for sparse qD_fullm (used in RNE derivatives)
  # D-structure is the full square sparsity pattern (both upper and lower triangle)
  m.qD_fullm_i, m.qD_fullm_j = [], []
  for i in range(mjm.nv):
    rowadr = mjm.D_rowadr[i]
    rownnz = mjm.D_rownnz[i]
    for k in range(rownnz):
      m.qD_fullm_i.append(i)
      m.qD_fullm_j.append(int(mjm.D_colind[rowadr + k]))
  m.nD = mjm.nD

  # Gather-based sparse mul_m: for each row, all (col, madr) including diagonal
  row_elements = [[] for _ in range(mjm.nv)]

  for i in range(mjm.nv):
    rowadr = mjm.M_rowadr[i]
    rownnz = mjm.M_rownnz[i]
    for k in range(rownnz):
      madr = rowadr + k
      col = int(mjm.M_colind[madr])
      row_elements[i].append((col, madr))  # row i gathers M[i,col] * vec[col]
      if i != col:
        row_elements[col].append((i, madr))  # row col gathers M[i,col] * vec[i]

  # Flatten into CSR-like arrays
  m.M_mulm_rowadr = [0]
  m.M_mulm_col = []
  m.M_mulm_madr = []
  for i in range(mjm.nv):
    for col, madr in row_elements[i]:
      m.M_mulm_col.append(col)
      m.M_mulm_madr.append(madr)
    m.M_mulm_rowadr.append(len(m.M_mulm_col))

  m.flexedge_J_rownnz = mjm.flexedge_J_rownnz
  m.flexedge_J_rowadr = mjm.flexedge_J_rowadr
  m.flexedge_J_colind = mjm.flexedge_J_colind.reshape(-1)

  # Populate lookup maps
  flex_elemflexid = np.zeros(mjm.nflexelem, dtype=np.int32)
  flex_edgeflexid = np.zeros(mjm.nflexedge, dtype=np.int32)
  flex_shellflexid = np.zeros(mjm.nflexshelldata, dtype=np.int32)
  flex_vertflexid = np.zeros(mjm.nflexvert, dtype=np.int32)
  flex_shelladr = np.zeros(mjm.nflex, dtype=np.int32)
  flex_efcnum = np.zeros(mjm.nflex, dtype=np.int32)
  flex_efcnnz = np.zeros(mjm.nflex, dtype=np.int32)
  flexedge_efcadr = np.full(mjm.nflexedge, -1, dtype=np.int32)
  flexedge_efcnnzadr = np.zeros(mjm.nflexedge, dtype=np.int32)

  if mjm.nflex > 0:
    shell_offset = 0
    for fi in range(mjm.nflex):
      elem_start = mjm.flex_elemadr[fi]
      elem_num = mjm.flex_elemnum[fi]
      flex_elemflexid[elem_start : elem_start + elem_num] = fi

      edge_start = mjm.flex_edgeadr[fi]
      edge_num = mjm.flex_edgenum[fi]
      flex_edgeflexid[edge_start : edge_start + edge_num] = fi

      # edge equality rows of a flex are allocated as one block: rigid edges and interpolated
      # flexes have no rows
      if mjm.flex_interp[fi] == 0:
        edges = np.arange(edge_start, edge_start + edge_num)
        edges = edges[~mjm.flexedge_rigid[edges].astype(bool)]
        edge_nnz = mjm.flexedge_J_rownnz[edges]
        flexedge_efcadr[edges] = np.arange(edges.size)
        flexedge_efcnnzadr[edges] = np.cumsum(edge_nnz) - edge_nnz
        flex_efcnum[fi] = edges.size
        flex_efcnnz[fi] = edge_nnz.sum()

      flex_shelladr[fi] = shell_offset
      shell_num = mjm.flex_shellnum[fi]
      flex_shellflexid[shell_offset : shell_offset + shell_num] = fi
      shell_offset += shell_num

      vert_start = mjm.flex_vertadr[fi]
      vert_num = mjm.flex_vertnum[fi]
      flex_vertflexid[vert_start : vert_start + vert_num] = fi

  m.flex_elemflexid = flex_elemflexid
  m.flex_edgeflexid = flex_edgeflexid
  m.flex_shellflexid = flex_shellflexid
  m.flex_vertflexid = flex_vertflexid
  m.flex_shelladr = flex_shelladr
  m.flex_efcnum = flex_efcnum
  m.flex_efcnnz = flex_efcnnz
  m.flexedge_efcadr = flexedge_efcadr
  m.flexedge_efcnnzadr = flexedge_efcnnzadr

  flex_bend_interp_map = []
  flex_face_map = []
  flex_faceadr = np.zeros(mjm.nflex, dtype=np.int32)
  if mjm.nflex > 0:
    face_offset = 0
    for fi in range(mjm.nflex):
      flex_faceadr[fi] = face_offset
      order = mjm.flex_interp[fi]
      if order >= 0:
        continue

      bendingadr = mjm.flex_bendingadr[fi]
      if bendingadr >= 0:
        nedge = int(mjm.flex_bending[bendingadr])
        for e in range(nedge):
          flex_bend_interp_map.append((fi, e))

      cx, cy, cz = mjm.flex_cellnum[fi]
      nfaces = 2 * (cy * cz + cx * cz + cx * cy)
      for face_idx in range(nfaces):
        flex_face_map.append((fi, face_idx))
      face_offset += nfaces

  # 1D rod bending hinges: (flex id, address of the hinge data in flex_bending)
  flex_rodhinge_map = []
  flex_rod_hingeadr = np.full(mjm.nflex, -1, dtype=np.int32)
  for fi in range(mjm.nflex):
    bendingadr = int(mjm.flex_bendingadr[fi])
    if mjm.flex_dim[fi] != 1 or mjm.flex_interp[fi] != 0 or mjm.flex_rigid[fi] or bendingadr < 0:
      continue
    nhinge = int(mjm.flex_bending[bendingadr])
    if nhinge > 0:
      flex_rod_hingeadr[fi] = len(flex_rodhinge_map)
      for h in range(nhinge):
        flex_rodhinge_map.append((fi, bendingadr + 1 + types.ROD_HINGE_SIZE * h))
  m.nflexrodhinge = len(flex_rodhinge_map)
  m.flex_rodhinge_map = np.array(flex_rodhinge_map, dtype=np.int32).reshape(-1, 2)
  m.flex_rod_hingeadr = flex_rod_hingeadr

  nflex = mjm.nflex
  nflexrodedge = 0

  flex_rod_is_chain = np.zeros(nflex, dtype=bool)
  flex_rod_has_twist = np.zeros(nflex, dtype=bool)
  flex_rod_is_box = np.zeros(nflex, dtype=bool)
  flex_rod_root_clamped = np.zeros(nflex, dtype=bool)
  flex_rod_tail_clamped = np.zeros(nflex, dtype=bool)
  flex_eff_possible = np.array(
    [_eff_flex_stiff_possible(mjm, fi) or _eff_flex_contact_possible(mjm, fi) for fi in range(nflex)],
    dtype=bool,
  )
  flex_rod_box_size = np.zeros((nflex, 2), dtype=np.float32)
  flex_rod_edgeadr = np.zeros(nflex, dtype=np.int32)
  flex_rod_edgenum = np.zeros(nflex, dtype=np.int32)

  rodedge_vert_list = []
  rodedge_jnt_list = []

  flex_elem_rodedge = np.empty((mjm.nflexelem, 2), dtype=np.int32)
  flex_elem_rodedge[:, 0] = -1
  flex_elem_rodedge[:, 1] = 1
  flex_vert_rodelem = np.full(mjm.nflexvert, -1, dtype=np.int32)

  for fi in range(nflex):
    va = int(mjm.flex_vertadr[fi])
    if mjm.flex_dim[fi] == 1:
      flex_vert_rodelem[va : va + int(mjm.flex_vertnum[fi])] = _mjm_rod_vert_elems(mjm, fi)

    bendingadr = int(mjm.flex_bendingadr[fi])
    if mjm.flex_dim[fi] == 1 and bendingadr >= 0 and types.ROD_HINGE_SIZE == 10:
      nhinge_raw = int(mjm.flex_bending[bendingadr])
      if nhinge_raw > 0:
        h0 = mjm.flex_bending[bendingadr + 1 : bendingadr + 1 + 10]
        if h0[3] < 0 and h0[4] > 0 and h0[9] != 0:
          flex_rod_is_box[fi] = True
          b_half = abs(float(h0[9]))
          a_half = b_half * np.sqrt(max(1.0, 1.0 + float(h0[5]) / float(h0[4])))
          flex_rod_box_size[fi, 0] = a_half
          flex_rod_box_size[fi, 1] = b_half

    is_open, nhinge = _is_open_rod_chain(mjm, fi)
    if not is_open:
      continue

    records = mjm.flex_bending[bendingadr + 1 : bendingadr + 1 + 10 * nhinge].reshape(nhinge, 10)
    has_twist = bool(mjm.flex_elemnum[fi] > 0 and (records[0, 7] >= 0 or records[-1, 7] >= 0 or records[0, 9] != 0))
    flex_rod_has_twist[fi] = has_twist

    b0 = int(mjm.flex_vertbodyid[va + int(records[0, 0])])
    b1 = int(mjm.flex_vertbodyid[va + int(records[0, 1])])
    root_clamped = b0 == b1
    flex_rod_root_clamped[fi] = root_clamped

    bT_prev = int(mjm.flex_vertbodyid[va + int(records[-1, 1])])
    bT = int(mjm.flex_vertbodyid[va + int(records[-1, 2])])
    tail_clamped = bool(nhinge >= 2 and bT_prev == bT)
    flex_rod_tail_clamped[fi] = tail_clamped

    nedge = nhinge + 1
    flex_rod_edgeadr[fi] = nflexrodedge
    flex_rod_edgenum[fi] = nedge

    v_chain = [int(records[h, 0]) for h in range(nhinge)] + [int(records[-1, 1]), int(records[-1, 2])]
    for e in range(nedge):
      rodedge_vert_list.append((v_chain[e], v_chain[e + 1]))
      if e == 0:
        jnt1 = int(records[0, 7])
        jnt = (jnt1 - 1) if (not root_clamped and jnt1 >= 1) else -1
      else:
        jnt = int(records[e - 1, 7])
      if jnt < 0 or jnt >= mjm.njnt:
        jnt = -1
      rodedge_jnt_list.append(jnt)

    elem_start = int(mjm.flex_elemadr[fi])
    elem_data_start = int(mjm.flex_elemdataadr[fi])
    for e in range(int(mjm.flex_elemnum[fi])):
      ed = elem_data_start + 2 * e
      ev0 = int(mjm.flex_elem[ed])
      ev1 = int(mjm.flex_elem[ed + 1])
      edge_idx = -1
      edge_sign = 1
      for k in range(nedge):
        idx = e if (k == 0 and e < nedge) else (0 if (k == e and e < nedge) else k)
        v0, v1 = rodedge_vert_list[nflexrodedge + idx]
        if ev0 == v0 and ev1 == v1:
          edge_idx = idx
          edge_sign = 1
          break
        elif ev0 == v1 and ev1 == v0:
          edge_idx = idx
          edge_sign = -1
          break
      flex_elem_rodedge[elem_start + e, 0] = edge_idx
      flex_elem_rodedge[elem_start + e, 1] = edge_sign

    any_aniso = bool((records[:, 5] != 0).any() or (records[:, 9] != 0).any())
    any_twist_jnt = bool((records[:, 7] >= 0).any())
    flex_rod_is_chain[fi] = bool(not mjm.flex_rigid[fi] and (any_aniso or any_twist_jnt))
    nflexrodedge += nedge

  m.nflexrodedge = nflexrodedge
  m.efm_con_max_dof = _calculate_efm_con_max_dof(mjm)
  m.flex_rod_is_chain = flex_rod_is_chain
  m.flex_rod_has_twist = flex_rod_has_twist
  m.flex_rod_is_box = flex_rod_is_box
  m.flex_rod_root_clamped = flex_rod_root_clamped
  m.flex_rod_tail_clamped = flex_rod_tail_clamped
  m.flex_eff_possible = flex_eff_possible
  m.flex_rod_box_size = flex_rod_box_size
  m.flex_rod_edgeadr = flex_rod_edgeadr
  m.flex_rod_edgenum = flex_rod_edgenum
  m.flex_rodedge_vert = (
    np.array(rodedge_vert_list, dtype=np.int32).reshape(-1, 2) if rodedge_vert_list else np.zeros((0, 2), dtype=np.int32)
  )
  m.flex_rodedge_jnt = np.array(rodedge_jnt_list, dtype=np.int32)
  m.flex_elem_rodedge = flex_elem_rodedge
  m.flex_vert_rodelem = flex_vert_rodelem

  if not flex_bend_interp_map:
    m.nflexbend_interp = 0
    m.flex_bend_interp_map = np.zeros((0, 2), dtype=np.int32)
  else:
    m.nflexbend_interp = len(flex_bend_interp_map)
    m.flex_bend_interp_map = np.array(flex_bend_interp_map, dtype=np.int32)

  if not flex_face_map:
    m.nflexface = 0
    m.flex_face_map = np.zeros((0, 2), dtype=np.int32)
  else:
    m.nflexface = len(flex_face_map)
    m.flex_face_map = np.array(flex_face_map, dtype=np.int32)
  m.flex_faceadr = flex_faceadr

  if m.nflexface > 0:
    flex_face = np.zeros((m.nflexface, 9), dtype=np.int32)
    for face_id, (fi, face_elem_idx) in enumerate(flex_face_map):
      order = mjm.flex_interp[fi]
      order_abs = -order
      cx, cy, cz = mjm.flex_cellnum[fi]
      nstart = mjm.flex_nodeadr[fi]
      npc = (order_abs + 1) * (order_abs + 1)

      for local_idx in range(9):
        if local_idx < npc:
          gidx = support.gather_face_node_index(int(cx), int(cy), int(cz), int(face_elem_idx), int(local_idx), int(order_abs))
          flex_face[face_id, local_idx] = nstart + gidx
        else:
          flex_face[face_id, local_idx] = -1
    m.flex_face = flex_face
  else:
    m.flex_face = np.zeros((0, 9), dtype=np.int32)

  sizes = {f.name: getattr(m, f.name) for f in dataclasses.fields(types.Model) if f.type is int}
  sizes.update(
    {
      "nbody_branches": len(m.body_branches),
      "nbranch_start": len(m.body_branch_start),
      "nbodyfree": sum(body_is_free),
      "nbody_fluid_ellipsoid": len(m.body_fluid_ellipsoid_adr),
      "nbody_fluid_box": len(m.body_fluid_box_adr),
      "njnt_limited_slide_hinge": len(m.jnt_limited_slide_hinge_adr),
      "njnt_limited_ball": len(m.jnt_limited_ball_adr),
      "ndof_tri": len(m.dof_tri_row),
      "nnxn_geom_pair": len(m.nxn_geom_pair),
      "nnxn_geom_pair_filtered": len(m.nxn_geom_pair_filtered),
      "neq_connect": len(m.eq_connect_adr),
      "neq_wld": len(m.eq_wld_adr),
      "neq_jnt": len(m.eq_jnt_adr),
      "neq_ten": len(m.eq_ten_adr),
      "neq_flex": len(m.eq_flex_adr),
      "ntendon_jnt": len(m.tendon_jnt_adr),
      "ntendon_site_pair": len(m.tendon_site_pair_adr),
      "ntendon_geom": len(m.tendon_geom_adr),
      "ntendon_limited": len(m.tendon_limited_adr),
      "nten_wrapadr_site": len(m.ten_wrapadr_site),
      "nwrap_jnt": len(m.wrap_jnt_adr),
      "nwrap_site": len(m.wrap_site_adr),
      "nwrap_site_pair": len(m.wrap_site_pair_adr),
      "nwrap_geom": len(m.wrap_geom_adr),
      "nsensor_pos": len(m.sensor_pos_adr),
      "nsensor_limitpos": len(m.sensor_limitpos_adr),
      "nsensor_vel": len(m.sensor_vel_adr),
      "nsensor_limitvel": len(m.sensor_limitvel_adr),
      "nsensor_acc": len(m.sensor_acc_adr),
      "nsensor_stage_pos": len(m.sensor_stage_pos_adr),
      "nsensor_stage_vel": len(m.sensor_stage_vel_adr),
      "nsensor_stage_acc": len(m.sensor_stage_acc_adr),
      "nsensor_touch": len(m.sensor_touch_adr),
      "nsensor_limitfrc": len(m.sensor_limitfrc_adr),
      "nsensor_tendonactfrc": len(m.sensor_tendonactfrc_adr),
      "nsensor_contact_pos": len(m.sensor_contact_pos_adr),
      "nsensor_contact_vel": len(m.sensor_contact_vel_adr),
      "nsensor_contact_acc": len(m.sensor_contact_acc_adr),
      "nsensor_collision_start_adr": len(m.sensor_collision_start_adr),
      "nqLD_all_updates": len(m.qLD_all_updates),
      "nqLD_level_offsets": len(m.qLD_level_offsets),
      "nM_fullm": len(m.M_fullm_i),
      "nqD_fullm": len(m.qD_fullm_i),
      "nv_plus_1": len(m.M_mulm_rowadr),
      "nM_mulm": len(m.M_mulm_col),
    }
  )
  for f in dataclasses.fields(types.Model):
    if warp_util.is_array_spec(f.type):
      batch_size = batch_sizes.get(f.name, 1)
      setattr(m, f.name, _create_array(getattr(m, f.name), f.type, sizes, batch_size))

  warp_util.mark_batched(m)
  return m


def _get_padded_sizes(nv: int, njmax: int, is_sparse: bool, tile_size: int, augment_cholesky: bool = False):
  def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

  njmax_padded = round_up(njmax, tile_size)
  nv_padded = round_up(nv + int(augment_cholesky), tile_size) if (is_sparse or nv > 32) else round_up(nv, 4)

  return njmax_padded, nv_padded


def _nvmax_pad(nvmax: int) -> int:
  """Reserve an augmented column and round nvmax up to the dense tile size."""
  t = types.TILE_SIZE_JTDAJ_DENSE
  return ((max(nvmax, 1) + t) // t) * t


def _default_nconmax(mjm: mujoco.MjModel, mjd: Optional[mujoco.MjData] = None) -> int:
  """Returns a default guess for an ideal nconmax given a Model and optional Data.

  This guess is based off a very simple heuristic, and may need to be manually raised if MJWarp
  reports ncon overflow, or lowered in order to get the very best performance.
  """
  valid_sizes = (2 + (np.arange(19) % 2)) * (2 ** (np.arange(19) // 2 + 3))  # 16, 24, 32, 48, ... 8192
  has_sdf = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  has_flex = mjm.nflex > 0
  nconmax = max(mjm.nv * 0.35 * (mjm.nhfield > 0) * 10 + 45, 256 * has_flex, 64 * has_sdf, mjd.ncon if mjd else 0)
  if nconmax > valid_sizes[-1]:
    return int(nconmax)
  return int(valid_sizes[np.searchsorted(valid_sizes, nconmax)])


def _default_njmax(mjm: mujoco.MjModel, mjd: Optional[mujoco.MjData] = None) -> int:
  """Returns a default guess for an ideal njmax given a Model and optional Data.

  This guess is based off a very simple heuristic, and may need to be manually raised if MJWarp
  reports ncon overflow, or lowered in order to get the very best performance.
  """
  valid_sizes = (2 + (np.arange(19) % 2)) * (2 ** (np.arange(19) // 2 + 3))  # 16, 24, 32, 48, ... 8192
  has_sdf = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  has_flex = mjm.nflex > 0
  njmax = max(mjm.nv * 2.26 * (mjm.nhfield > 0) * 18 + 53, 512 * has_flex, 256 * has_sdf, mjd.nefc if mjd else 0)
  if njmax > valid_sizes[-1]:
    return int(njmax)
  return int(valid_sizes[np.searchsorted(valid_sizes, njmax)])


def _mjm_rod_has_twist(mjm: mujoco.MjModel, fi: int) -> bool:
  """Returns True when flex `fi` is an open 1D DER rod chain with twist DOFs or box section."""
  if mjm.flex_elemnum[fi] <= 0:
    return False
  is_open, nhinge = _is_open_rod_chain(mjm, fi)
  if not is_open:
    return False
  bendingadr = int(mjm.flex_bendingadr[fi])
  records = mjm.flex_bending[bendingadr + 1 : bendingadr + 1 + 10 * nhinge].reshape(nhinge, 10)
  return bool(records[0, 7] >= 0 or records[-1, 7] >= 0 or records[0, 9] != 0)


def _mjm_rod_elem_twist_jnts(mjm: mujoco.MjModel, fi: int) -> list[int]:
  """Twist joint of each element of open 1D DER rod `fi`, or -1; O(nelem) via a chain-edge table."""
  elemnum = int(mjm.flex_elemnum[fi])
  res = [-1] * elemnum
  if not _mjm_rod_has_twist(mjm, fi):
    return res
  bendingadr = int(mjm.flex_bendingadr[fi])
  nhinge = int(mjm.flex_bending[bendingadr])
  va = int(mjm.flex_vertadr[fi])
  root_clamped = (
    mjm.flex_vertbodyid[va + int(mjm.flex_bending[bendingadr + 1])]
    == mjm.flex_vertbodyid[va + int(mjm.flex_bending[bendingadr + 2])]
  )
  edge_jnt = {}
  for c in range(nhinge + 1):
    if c < nhinge:
      cv0 = int(mjm.flex_bending[bendingadr + 1 + 10 * c])
      cv1 = int(mjm.flex_bending[bendingadr + 1 + 10 * c + 1])
    else:
      cv0 = int(mjm.flex_bending[bendingadr + 1 + 10 * (nhinge - 1) + 1])
      cv1 = int(mjm.flex_bending[bendingadr + 1 + 10 * (nhinge - 1) + 2])
    if c == 0:
      jnt1 = int(mjm.flex_bending[bendingadr + 1 + 7])
      jnt = (jnt1 - 1) if (not root_clamped and jnt1 >= 1) else -1
    else:
      jnt = int(mjm.flex_bending[bendingadr + 1 + 10 * (c - 1) + 7])
    edge_jnt.setdefault((min(cv0, cv1), max(cv0, cv1)), jnt if (0 <= jnt < mjm.njnt) else -1)
  ed_base = int(mjm.flex_elemdataadr[fi])
  for e in range(elemnum):
    v0, v1 = int(mjm.flex_elem[ed_base + 2 * e]), int(mjm.flex_elem[ed_base + 2 * e + 1])
    res[e] = edge_jnt.get((min(v0, v1), max(v0, v1)), -1)
  return res


def _calculate_efm_con_max_dof(mjm: mujoco.MjModel) -> int:
  """Returns maximum unique DOFs affected by any passive flex contact in `mjm`."""
  if mjm.nflex == 0 or mjm.nv == 0:
    return 1
  geom_nnz = [_body_set_nnz(mjm, {mjm.geom_bodyid[g]}) for g in range(mjm.ngeom)]
  non_interp_flex_nnz = []
  passive_flex_nnz = []
  for fi in range(mjm.nflex):
    if mjm.flex_interp[fi] != 0 or mjm.flex_rigid[fi]:
      continue
    vert_start = int(mjm.flex_vertadr[fi])
    dim = int(mjm.flex_dim[fi])
    elem_num = int(mjm.flex_elemnum[fi])
    elem_data_start = int(mjm.flex_elemdataadr[fi])
    f_nnz = 0
    if elem_num > 0:
      has_rod_twist = _mjm_rod_has_twist(mjm, fi)
      elem_jnt = _mjm_rod_elem_twist_jnts(mjm, fi) if has_rod_twist else []
      for e in range(elem_num):
        elem_bodies = {
          mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + e * (dim + 1) + k]] for k in range(dim + 1)
        }
        if has_rod_twist:
          jnt_e = elem_jnt[e]
          if 0 <= jnt_e < mjm.njnt:
            elem_bodies.add(int(mjm.jnt_bodyid[jnt_e]))
        f_nnz = max(f_nnz, _body_set_nnz(mjm, elem_bodies))
      if has_rod_twist:
        vert_elem = _mjm_rod_vert_elems(mjm, fi)
        for v in range(int(mjm.flex_vertnum[fi])):
          e_rod = vert_elem[v]
          vert_bodies = {mjm.flex_vertbodyid[vert_start + v]}
          if e_rod >= 0:
            vert_bodies.add(mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + 2 * e_rod]])
            vert_bodies.add(mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + 2 * e_rod + 1]])
            jnt_e = elem_jnt[e_rod]
            if 0 <= jnt_e < mjm.njnt:
              vert_bodies.add(int(mjm.jnt_bodyid[jnt_e]))
          f_nnz = max(f_nnz, _body_set_nnz(mjm, vert_bodies))
    else:
      for v in range(int(mjm.flex_vertnum[fi])):
        f_nnz = max(f_nnz, _body_set_nnz(mjm, {mjm.flex_vertbodyid[vert_start + v]}))
    non_interp_flex_nnz.append(f_nnz)
    if mjm.flex_passive[fi] != 0 and dim >= 2:
      passive_flex_nnz.append(f_nnz)
  if not passive_flex_nnz:
    max_other = max([0] + non_interp_flex_nnz + geom_nnz)
    return int(min(max(2 * max_other, 1), max(mjm.nv, 1)))
  max_passive = max(passive_flex_nnz)
  max_other = max([0] + non_interp_flex_nnz + geom_nnz)
  return int(min(max(max_passive + max_other, 1), max(mjm.nv, 1)))


def _body_pair_nnz(mjm: mujoco.MjModel, body1: int, body2: int) -> int:
  """Returns the number of unique DOFs in the kinematic tree union of two bodies."""
  body1 = mjm.body_weldid[body1]
  body2 = mjm.body_weldid[body2]
  da1 = mjm.body_dofadr[body1] + mjm.body_dofnum[body1] - 1
  da2 = mjm.body_dofadr[body2] + mjm.body_dofnum[body2] - 1
  nnz = 0
  while da1 >= 0 or da2 >= 0:
    da = max(da1, da2)
    if da1 == da:
      da1 = mjm.dof_parentid[da1]
    if da2 == da:
      da2 = mjm.dof_parentid[da2]
    nnz += 1
  return nnz


def _body_set_nnz(mjm: mujoco.MjModel, bodies) -> int:
  """Returns the number of unique DOFs in the kinematic tree union of a set of bodies."""
  active_dofs = set()
  for b in bodies:
    b = mjm.body_weldid[b]
    da = mjm.body_dofadr[b] + mjm.body_dofnum[b] - 1
    while da >= 0:
      active_dofs.add(da)
      da = mjm.dof_parentid[da]
  return len(active_dofs)


def _calculate_max_contact_nnz(mjm: mujoco.MjModel) -> int:
  """Returns the maximum number of non-zeros for a single contact constraint."""
  max_contact_nnz = 0

  # contact pairs
  for i in range(mjm.npair):
    g1, g2 = mjm.pair_geom1[i], mjm.pair_geom2[i]
    b1, b2 = mjm.geom_bodyid[g1], mjm.geom_bodyid[g2]
    max_contact_nnz = max(max_contact_nnz, _body_pair_nnz(mjm, b1, b2))

  # filter geom-geom pairs (unique body pairs, filtered)
  body_pair_seen = set()
  for i in range(mjm.ngeom):
    bi = mjm.geom_bodyid[i]
    cti, cai = mjm.geom_contype[i], mjm.geom_conaffinity[i]
    for j in range(i + 1, mjm.ngeom):
      bj = mjm.geom_bodyid[j]
      if bi == bj:
        continue
      if mjm.body_weldid[bi] == 0 and mjm.body_weldid[bj] == 0:
        continue
      bp = (min(bi, bj), max(bi, bj))
      if bp in body_pair_seen:
        continue
      ctj, caj = mjm.geom_contype[j], mjm.geom_conaffinity[j]
      if not ((cti & caj) or (ctj & cai)):
        continue
      body_pair_seen.add(bp)
      max_contact_nnz = max(max_contact_nnz, _body_pair_nnz(mjm, bi, bj))

  if mjm.nflex == 0:
    return max_contact_nnz

  # Compute upper bound NNZ contribution for each flex individually
  flex_nnz = [0] * mjm.nflex
  for fi in range(mjm.nflex):
    if mjm.flex_interp[fi] == 0:
      vert_start = mjm.flex_vertadr[fi]
      dim = mjm.flex_dim[fi]
      elem_num = mjm.flex_elemnum[fi]
      elem_data_start = mjm.flex_elemdataadr[fi]
      if elem_num > 0:
        has_rod_twist = _mjm_rod_has_twist(mjm, fi)
        elem_jnt = _mjm_rod_elem_twist_jnts(mjm, fi) if has_rod_twist else []
        for e in range(elem_num):
          elem_bodies = {
            mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + e * (dim + 1) + k]] for k in range(dim + 1)
          }
          if has_rod_twist:
            jnt_e = elem_jnt[e]
            if 0 <= jnt_e < mjm.njnt:
              elem_bodies.add(int(mjm.jnt_bodyid[jnt_e]))
          flex_nnz[fi] = max(flex_nnz[fi], _body_set_nnz(mjm, elem_bodies))
        if has_rod_twist:
          vert_elem = _mjm_rod_vert_elems(mjm, fi)
          for v in range(mjm.flex_vertnum[fi]):
            e_rod = vert_elem[v]
            vert_bodies = {mjm.flex_vertbodyid[vert_start + v]}
            if e_rod >= 0:
              vert_bodies.add(mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + 2 * e_rod]])
              vert_bodies.add(mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + 2 * e_rod + 1]])
              jnt_e = elem_jnt[e_rod]
              if 0 <= jnt_e < mjm.njnt:
                vert_bodies.add(int(mjm.jnt_bodyid[jnt_e]))
            flex_nnz[fi] = max(flex_nnz[fi], _body_set_nnz(mjm, vert_bodies))
      else:
        for v in range(mjm.flex_vertnum[fi]):
          flex_nnz[fi] = max(
            flex_nnz[fi],
            _body_set_nnz(mjm, {mjm.flex_vertbodyid[vert_start + v]}),
          )
    else:
      order = abs(mjm.flex_interp[fi])
      is_shell = mjm.flex_interp[fi] < 0
      cx, cy, cz = mjm.flex_cellnum[fi]
      nstart = mjm.flex_nodeadr[fi]
      dim = mjm.flex_dim[fi]
      nx = cx * order + 1
      ny = cy * order + 1 if dim > 1 else 1
      nz = cz * order + 1 if dim > 2 else 1

      ci, cj, ck = cx // 2, cy // 2, cz // 2
      cell_bodies = set()

      for li in range(order + 1):
        for lj in range(order + 1 if dim > 1 else 1):
          for lk in range(order + 1 if dim > 2 else 1):
            gi = ci + li
            gj = cj + lj
            gk = ck + lk

            is_interior = False
            if is_shell:
              is_interior = (
                (gi > 0 and gi < cx * order)
                and (gj > 0 and gj < cy * order if dim > 1 else True)
                and (gk > 0 and gk < cz * order if dim > 2 else True)
              )

            if is_interior:
              for bi in (0, gi, nx - 1):
                for bj in (0, gj, ny - 1 if dim > 1 else 0):
                  for bk in (0, gk, nz - 1 if dim > 2 else 0):
                    if (
                      bi == 0
                      or bi == nx - 1
                      or (dim > 1 and (bj == 0 or bj == ny - 1))
                      or (dim > 2 and (bk == 0 or bk == nz - 1))
                    ):
                      node_idx = bi * ny * nz + bj * nz + bk
                      cell_bodies.add(mjm.flex_nodebodyid[nstart + node_idx])
            else:
              node_idx = gi * ny * nz + gj * nz + gk
              cell_bodies.add(mjm.flex_nodebodyid[nstart + node_idx])

      flex_nnz[fi] = _body_set_nnz(mjm, cell_bodies)

  geom_nnz = [_body_set_nnz(mjm, {mjm.geom_bodyid[g]}) for g in range(mjm.ngeom)]

  for fi in range(mjm.nflex):
    fct = mjm.flex_contype[fi]
    fca = mjm.flex_conaffinity[fi]

    # flex-geom contacts
    for g in range(mjm.ngeom):
      ct, ca = mjm.geom_contype[g], mjm.geom_conaffinity[g]
      if (fct & ca) or (ct & fca):
        max_contact_nnz = max(max_contact_nnz, flex_nnz[fi] + geom_nnz[g])

    # flex self-collision
    if mjm.flex_selfcollide[fi] and (fct & fca):
      max_contact_nnz = max(max_contact_nnz, 2 * flex_nnz[fi])

    # flex-flex collision
    for fj in range(fi + 1, mjm.nflex):
      fct_j = mjm.flex_contype[fj]
      fca_j = mjm.flex_conaffinity[fj]
      if (fct & fca_j) or (fct_j & fca):
        max_contact_nnz = max(max_contact_nnz, flex_nnz[fi] + flex_nnz[fj])

  return max_contact_nnz


def _default_njmax_nnz(mjm: mujoco.MjModel, nconmax: int, njmax: int) -> int:
  """Returns a heuristic estimate for the number of non-zeros in the sparse constraint Jacobian.

  Assumes all equality, friction, and limit constraints are active and computes
  their non-zeros. For contacts, assumes njmax contact rows at the maximum
  body-pair non-zeros from all enabled collision pairs.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    nconmax: Maximum number of contacts per world.
    njmax: Maximum number of constraint rows per world.

  Returns:
    Estimated number of non-zeros in the constraint Jacobian.
  """
  total_nnz = 0

  def _eq_bodies(i):
    """Returns body pair for equality constraint i."""
    obj1id, obj2id = mjm.eq_obj1id[i], mjm.eq_obj2id[i]
    if mjm.eq_objtype[i] == mujoco.mjtObj.mjOBJ_SITE:
      return mjm.site_bodyid[obj1id], mjm.site_bodyid[obj2id]
    return obj1id, obj2id

  # equality constraints (assume all active)
  for i in range(mjm.neq):
    eq_type = mjm.eq_type[i]

    if eq_type == mujoco.mjtEq.mjEQ_CONNECT:
      total_nnz += 3 * _body_pair_nnz(mjm, *_eq_bodies(i))

    elif eq_type == mujoco.mjtEq.mjEQ_WELD:
      total_nnz += 6 * _body_pair_nnz(mjm, *_eq_bodies(i))

    elif eq_type == mujoco.mjtEq.mjEQ_JOINT:
      total_nnz += 2 if mjm.eq_obj2id[i] >= 0 else 1

    elif eq_type == mujoco.mjtEq.mjEQ_TENDON:
      obj1id = mjm.eq_obj1id[i]
      obj2id = mjm.eq_obj2id[i]
      rownnz1 = mjm.ten_J_rownnz[obj1id] if obj1id < mjm.ntendon else 0
      if obj2id >= 0 and obj2id < mjm.ntendon:
        rowadr1 = mjm.ten_J_rowadr[obj1id]
        rowadr2 = mjm.ten_J_rowadr[obj2id]
        rownnz2 = mjm.ten_J_rownnz[obj2id]
        cols = set()
        for j in range(rownnz1):
          cols.add(mjm.ten_J_colind[rowadr1 + j])
        for j in range(rownnz2):
          cols.add(mjm.ten_J_colind[rowadr2 + j])
        total_nnz += len(cols)
      else:
        total_nnz += rownnz1

    elif eq_type == mujoco.mjtEq.mjEQ_FLEX:
      obj1id = mjm.eq_obj1id[i]
      if obj1id < mjm.nflex:
        edge_start = mjm.flex_edgeadr[obj1id]
        edge_count = mjm.flex_edgenum[obj1id]
        for e in range(edge_count):
          if mjm.flexedge_rigid is not None and mjm.flexedge_rigid[edge_start + e]:
            continue
          total_nnz += mjm.flexedge_J_rownnz[edge_start + e]

    elif eq_type == mujoco.mjtEq.mjEQ_FLEXSTRAIN:
      # strain constraints: each cell produces neig rows, each dense (nv)
      obj1id = mjm.eq_obj1id[i]
      if obj1id < mjm.nflex:
        # estimate neig from stiffness data
        adr = mjm.flex_stiffnessadr[obj1id]
        neig = int(mjm.flex_stiffness[adr])
        total_nnz += neig * mjm.nv

  # friction constraints
  total_nnz += (mjm.dof_frictionloss > 0).sum()
  for i in range(mjm.ntendon):
    if mjm.tendon_frictionloss[i] > 0:
      total_nnz += mjm.ten_J_rownnz[i]

  # limit constraints (assume all active)
  for i in range(mjm.njnt):
    if mjm.jnt_limited[i]:
      jnt_type = int(mjm.jnt_type[i])
      if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        total_nnz += 3
      elif jnt_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
        total_nnz += 1
  for i in range(mjm.ntendon):
    if mjm.tendon_limited[i]:
      total_nnz += mjm.ten_J_rownnz[i]

  # contact constraints: njmax rows at max contact non-zeros
  max_contact_nnz = _calculate_max_contact_nnz(mjm)
  total_nnz += njmax * max_contact_nnz

  return int(min(max(total_nnz, 1), njmax * mjm.nv))


def _resolve_batch_size(na: int | None, n: int | None, nworld: int, default: int) -> int:
  if na is not None:
    return na
  if n is not None:
    return n * nworld
  return default


def _allocate_island_arrays(
  mjm: mujoco.MjModel,
  d: types.Data,
  nworld: int,
  njmax: int,
  mjd: mujoco.MjData,
):
  ntree = mjm.ntree
  nv = mjm.nv
  nisland = int(mjd.nisland)
  nidof = int(mjd.nidof)
  nefc = int(mjd.nefc)

  d.nisland = wp.array(np.full(nworld, nisland, dtype=np.int32), dtype=int)
  d.nidof = wp.array(np.full(nworld, nidof, dtype=np.int32), dtype=int)

  tree_island = np.tile(mjd.tree_island[:ntree], (nworld, 1)) if nisland > 0 else np.full((nworld, ntree), -1, dtype=np.int32)
  d.tree_island = wp.array(tree_island, dtype=int)
  dof_island = np.tile(mjd.dof_island[:nv], (nworld, 1)) if nisland > 0 else np.full((nworld, nv), -1, dtype=np.int32)
  d.dof_island = wp.array(dof_island, dtype=int)

  # Island arrays sized by ntree
  for name in (
    "island_dofadr",
    "island_idofadr",
    "island_nv",
    "island_nefc",
    "island_ne",
    "island_nf",
    "island_iefcadr",
  ):
    arr = np.zeros((nworld, ntree), dtype=np.int32)
    if nisland > 0:
      arr[:, :nisland] = getattr(mjd, name)[:nisland]
    setattr(d, name, wp.array(arr, dtype=int))

  # DOF mapping arrays sized by nv
  if nisland > 0:
    map_dof2idof = np.tile(mjd.map_dof2idof[:nv], (nworld, 1))
    map_idof2dof = np.tile(mjd.map_idof2dof[:nv], (nworld, 1))
  else:
    map_dof2idof = np.zeros((nworld, nv), dtype=np.int32)
    map_idof2dof = np.zeros((nworld, nv), dtype=np.int32)
  dof_islandid = np.full((nworld, nv), -1, dtype=np.int32)
  if nisland > 0 and nidof > 0:
    dof_islandid[:, :nidof] = mjd.dof_island[mjd.map_idof2dof[:nidof]]

  d.map_dof2idof = wp.array(map_dof2idof, dtype=int)
  d.map_idof2dof = wp.array(map_idof2dof, dtype=int)
  d.dof_islandid = wp.array(dof_islandid, dtype=int)

  # Constraint mapping arrays sized by njmax
  map_efc2iefc = np.zeros((nworld, njmax), dtype=np.int32)
  map_iefc2efc = np.zeros((nworld, njmax), dtype=np.int32)
  efc_islandid = np.full((nworld, njmax), -1, dtype=np.int32)

  if nefc > 0:
    map_efc2iefc[:, :nefc] = mjd.map_efc2iefc[:nefc]
    map_iefc2efc[:, :nefc] = mjd.map_iefc2efc[:nefc]
    if nisland > 0:
      total_iefc = int(mjd.island_iefcadr[nisland - 1] + mjd.island_nefc[nisland - 1])
      efc_islandid[:, :total_iefc] = mjd.efc_island[mjd.map_iefc2efc[:total_iefc]]

  d.map_efc2iefc = wp.array(map_efc2iefc, dtype=int)
  d.map_iefc2efc = wp.array(map_iefc2efc, dtype=int)
  d.efc_islandid = wp.array(efc_islandid, dtype=int)


_COMPACT_DATA_FIELDS: tuple[str, ...] = (
  "cdof_tri_row",
  "cdof_tri_col",
  "cM",
  "cqLD",
  "crhs",
  "cx",
  "cJ",
  "cMa",
  "cqfrc_smooth",
  "cqacc_smooth",
  "cqacc_warmstart",
  "cqacc",
  "cqfrc_constraint",
)


def _allocate_compact_arrays(
  mjm: mujoco.MjModel,
  d: types.Data,
  nworld: int,
  nvmax_pad: int,
  njmax_pad: int,
  compact: bool,
):
  """Allocate workspace for the compacted dense factor/solve (when nvmax is requested).

  Mirrors the island-local ``i*`` Data fields with a ``c*`` (compact) prefix. The
  constant dof-pair indices are derived on the host since
  they depend only on nvmax_pad; the workspace shadows are sized by nvmax_pad so the
  blocked Cholesky never reads out of bounds on its partial tile. When the user does not
  request compaction (nvmax is None) everything is allocated empty.

  TODO(team): once the compact path replaces the island solver, the whole forward
  pipeline can run in compacted space and ``d.M`` / ``d.qacc`` etc. become nvmax-sized
  directly, collapsing these ``c*`` shadows into the primary Data fields.
  """
  nw = nworld if compact else 0
  nvp = nvmax_pad if compact else 0
  njp = njmax_pad if compact else 0

  if compact:
    # all (i, j) DOF pairs of the nvmax_pad-wide compacted Hessian (the global dof_tri,
    # triu over full nv, would index out of bounds).
    idx = np.arange(nvmax_pad, dtype=np.int32)
    d.cdof_tri_row = wp.array(np.repeat(idx, nvmax_pad), dtype=int)
    d.cdof_tri_col = wp.array(np.tile(idx, nvmax_pad), dtype=int)
  else:
    d.cdof_tri_row = wp.empty(0, dtype=int)
    d.cdof_tri_col = wp.empty(0, dtype=int)

  alloc_cJ = compact and not is_sparse(mjm)

  d.cM = wp.empty((nw, nvp, nvp), dtype=float)
  d.cqLD = wp.empty((nw, nvp, nvp), dtype=float)
  d.crhs = wp.empty((nw, nvp, 1), dtype=float)
  d.cx = wp.empty((nw, nvp, 1), dtype=float)
  d.cJ = wp.empty((nw, njp, nvp), dtype=float) if alloc_cJ else wp.empty((0, 0, 0), dtype=float)
  d.cMa = wp.empty((nw, nvp), dtype=float)
  d.cqfrc_smooth = wp.empty((nw, nvp), dtype=float)
  d.cqacc_smooth = wp.empty((nw, nvp), dtype=float)
  d.cqacc_warmstart = wp.empty((nw, nvp), dtype=float)
  d.cqacc = wp.empty((nw, nvp), dtype=float)
  d.cqfrc_constraint = wp.empty((nw, nvp), dtype=float)


def _initial_body_awake(mjm: mujoco.MjModel, nworld: int, init_asleep: bool) -> np.ndarray:
  """Returns the initial body awake array."""
  body_awake_np = np.zeros((nworld, mjm.nbody), dtype=np.int32)
  for b in range(mjm.nbody):
    tree = mjm.body_treeid[b]
    if tree < 0:
      root = mjm.body_rootid[b]
      mocap = mjm.body_mocapid[root]
      if mocap >= 0:
        body_awake_np[:, b] = int(types.SleepState.AWAKE)
      else:
        body_awake_np[:, b] = int(types.SleepState.STATIC)
    else:
      body_awake_np[:, b] = int(types.SleepState.ASLEEP) if init_asleep else int(types.SleepState.AWAKE)
  return body_awake_np


def make_data(
  mjm: mujoco.MjModel,
  nworld: int = 1,
  nconmax: Optional[int] = None,
  nccdmax: Optional[int] = None,
  njmax: Optional[int] = None,
  njmax_nnz: Optional[int] = None,
  naconmax: Optional[int] = None,
  naccdmax: Optional[int] = None,
  nvmax: Optional[int] = None,
) -> types.Data:
  """Creates a data object on device.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    nworld: Number of worlds.
    nconmax: Number of contacts to allocate per world. Contacts exist in large
             heterogeneous arrays: one world may have more than nconmax contacts.
    nccdmax: Number of CCD contacts to allocate per world. Same semantics as nconmax.
    njmax: Number of constraints to allocate per world. Constraint arrays are
           batched by world: no world may have more than njmax constraints.
    njmax_nnz: Number of non-zeros in constraint Jacobian (sparse). Defaults to njmax * nv.
    naconmax: Number of contacts to allocate for all worlds. Overrides nconmax.
    naccdmax: Maximum number of CCD contacts. Defaults to naconmax.
    nvmax: Capacity for compacted active DOFs per world. Defaults to nv.

  Returns:
    The data object containing the current state and output arrays (device).
  """
  # TODO(team): move nconmax, njmax to Model?
  if nconmax is None:
    nconmax = _default_nconmax(mjm)

  if njmax is None:
    njmax = _default_njmax(mjm)

  sleep_enabled = bool(mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and not bool(
    mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ISLAND
  )
  compact_alloc = sleep_enabled or (nvmax is not None)

  if nvmax is None:
    nvmax = mjm.nv

  if nconmax < 0:
    raise ValueError("nconmax must be >= 0")

  if njmax < 0:
    raise ValueError("njmax must be >= 0")

  if nvmax < 0 or nvmax > mjm.nv:
    raise ValueError(f"nvmax ({nvmax}) must be in [0, nv ({mjm.nv})]")

  if nworld < 1:
    raise ValueError(f"nworld must be >= 1")

  naconmax = _resolve_batch_size(naconmax, nconmax, nworld, 0)
  if naconmax < 0:
    raise ValueError("naconmax must be >= 0")

  naccdmax = _resolve_batch_size(naccdmax, nccdmax, nworld, naconmax)
  if naccdmax < 0:
    raise ValueError("naccdmax must be >= 0")
  elif naccdmax > naconmax:
    raise ValueError(f"naccdmax ({naccdmax}) must be <= naconmax ({naconmax})")

  if nccdmax is None:
    nccdmax = nconmax
  else:
    if nccdmax < 0:
      raise ValueError("nccdmax must be >= 0")
    elif nccdmax > nconmax:
      raise ValueError(f"nccdmax ({nccdmax}) must be <= nconmax ({nconmax})")

  nv_compact = nvmax < mjm.nv

  sizes = dict(
    {"*": 1},
    **{
      f.name: (getattr(mjm, f.name, 0) if f.name == "nplastic" else getattr(mjm, f.name, None))
      for f in dataclasses.fields(types.Model)
      if f.type is int
    },
  )
  condim_arrays = [np.array([0]), mjm.geom_condim, mjm.pair_dim]
  if mjm.nflex > 0:
    condim_arrays.append(mjm.flex_condim)
  sizes["nmaxcondim"] = np.concatenate(condim_arrays).max()
  sizes["nmaxpyramid"] = np.maximum(1, 2 * (sizes["nmaxcondim"] - 1))
  tile_size = types.TILE_SIZE_JTDAJ_SPARSE if is_sparse(mjm) else types.TILE_SIZE_JTDAJ_DENSE
  sizes["njmax_pad"], sizes["nv_pad"] = _get_padded_sizes(
    mjm.nv,
    njmax,
    is_sparse(mjm),
    tile_size,
    augment_cholesky=mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON and mjm.nv > 32,
  )
  sizes["nworld"] = nworld
  sizes["naconmax"] = naconmax
  sizes["njmax"] = njmax
  sizes["nvmax"] = nvmax
  sizes["nvmax_pad"] = _nvmax_pad(nvmax)
  sizes["nvmax_pad_sq"] = sizes["nvmax_pad"] * sizes["nvmax_pad"]
  sizes["nflexintcell"] = _get_nflexintcell(mjm)
  sizes["nflexface"] = _get_nflexface(mjm)
  sizes["nflexrodhinge"], sizes["nflexrodedge"] = _get_nflexrod_counts(mjm)
  efm_d = _get_efm_data(mjm)
  sizes["nefmK"] = int(efm_d.nefmK) if efm_d else 0

  # qLD holds the factor: a packed dense region for dense blocks followed
  # by an nC-length LDL region for sparse blocks (present only when some block is sparse). Either
  # region may be empty (pure dense / pure sparse).
  _lay = m_block_layout(mjm)
  qld_total = _lay["total"] + (mjm.nC if _lay["has_sparse"] else 0)
  sizes["qld_total"] = qld_total

  if njmax_nnz is None:
    if is_sparse(mjm):
      njmax_nnz = _default_njmax_nnz(mjm, nconmax, njmax)
    else:
      njmax_nnz = njmax * mjm.nv

  contact_kwargs = {}
  for f in dataclasses.fields(types.Contact):
    if f.name in ["flex", "elem", "vert"]:
      contact_kwargs[f.name] = wp.full(naconmax, wp.vec2i(-1, -1)) if mjm.nflex > 0 else wp.empty(0, dtype=wp.vec2i)
    else:
      contact_kwargs[f.name] = _create_array(None, f.type, sizes)
  contact = types.Contact(**contact_kwargs)
  contact.efc_address = wp.array(np.full((naconmax, sizes["nmaxpyramid"]), -1, dtype=int), dtype=int)

  efc = _create_constraint(mjm, nworld, njmax, sizes)

  if is_sparse(mjm):
    efc.J_rownnz = wp.zeros((nworld, njmax), dtype=int)
    efc.J_rowadr = wp.zeros((nworld, njmax), dtype=int)
    efc.J_colind = wp.zeros((nworld, 1, njmax_nnz), dtype=int)
    efc.J = wp.zeros((nworld, 1, njmax_nnz), dtype=float)
  else:
    efc.J_rownnz = wp.zeros((nworld, 0), dtype=int)
    efc.J_rowadr = wp.zeros((nworld, 0), dtype=int)
    efc.J_colind = wp.zeros((nworld, 0, 0), dtype=int)
    efc.J = wp.zeros((nworld, sizes["njmax_pad"], sizes["nv_pad"]), dtype=float)

  # Compute initial kinematic state. Static geom positions (geom_xpos, geom_xmat) are set here
  # and never updated by the physics loop (see smooth.py geom_kinematics), so this call is the
  # only place they are initialized. Also seeds body poses (xquat, xmat, ximat) at qpos0.
  mjd = mujoco.MjData(mjm)
  mujoco.mj_kinematics(mjm, mjd)

  # mocap
  mocap_body = np.nonzero(mjm.body_mocapid >= 0)[0]
  mocap_id = mjm.body_mocapid[mocap_body]

  d_kwargs = {
    "qpos": wp.array(np.tile(mjm.qpos0, nworld), shape=(nworld, mjm.nq), dtype=float),
    "contact": contact,
    "efc": efc,
    "nworld": nworld,
    "naconmax": naconmax,
    "naccdmax": naccdmax,
    "njmax": njmax,
    "nvmax": nvmax,
    "nvmax_pad": sizes["nvmax_pad"],
    "njmax_pad": sizes["njmax_pad"],
    "njmax_nnz": njmax_nnz,
    # world body
    "xquat": wp.array(np.tile(mjd.xquat, (nworld, 1)), shape=(nworld, mjm.nbody), dtype=wp.quat),
    "xmat": wp.array(np.tile(mjd.xmat, (nworld, 1)), shape=(nworld, mjm.nbody), dtype=wp.mat33),
    "ximat": wp.array(np.tile(mjd.ximat, (nworld, 1)), shape=(nworld, mjm.nbody), dtype=wp.mat33),
    # static geoms
    "geom_xpos": wp.array(np.tile(mjd.geom_xpos, (nworld, 1)), shape=(nworld, mjm.ngeom), dtype=wp.vec3),
    "geom_xmat": wp.array(np.tile(mjd.geom_xmat, (nworld, 1)), shape=(nworld, mjm.ngeom), dtype=wp.mat33),
    # mocap
    "mocap_pos": wp.array(np.tile(mjm.body_pos[mocap_body[mocap_id]], (nworld, 1)), shape=(nworld, mjm.nmocap), dtype=wp.vec3),
    "mocap_quat": wp.array(
      np.tile(mjm.body_quat[mocap_body[mocap_id]], (nworld, 1)), shape=(nworld, mjm.nmocap), dtype=wp.quat
    ),
    # equality constraints
    "eq_active": wp.array(np.tile(mjm.eq_active0.astype(bool), (nworld, 1)), shape=(nworld, mjm.neq), dtype=bool),
    # history
    "history": (
      wp.array(np.tile(mjd.history, (nworld, 1)), shape=(nworld, mjm.nhistory), dtype=float)
      if mjm.nhistory > 0
      else wp.zeros((nworld, 0), dtype=float)
    ),
    # plastic
    "plastic": (
      wp.array(np.tile(mjd.plastic, (nworld, 1)), shape=(nworld, mjm.nplastic), dtype=float)
      if getattr(mjm, "nplastic", 0) > 0
      else wp.zeros((nworld, 0), dtype=float)
    ),
    # island arrays
    "nisland": None,
    "tree_island": None,
    "dof_island": None,
    "island_dofadr": None,
    "island_idofadr": None,
    "island_nv": None,
    "island_nefc": None,
    "island_ne": None,
    "island_nf": None,
    "island_iefcadr": None,
    "nidof": None,
    "map_dof2idof": None,
    "map_idof2dof": None,
    "map_efc2iefc": None,
    "map_iefc2efc": None,
    "dof_islandid": None,
    "efc_islandid": None,
    # compact arrays (populated by _allocate_compact_arrays; skip eager allocation)
    **{name: None for name in _COMPACT_DATA_FIELDS},
    "tree_asleep": wp.array(np.full((nworld, mjm.ntree), -(1 + types.MJ_MINAWAKE), dtype=np.int32), dtype=int),
    "tree_awake": wp.array(np.ones((nworld, mjm.ntree), dtype=np.int32), dtype=int),
    "body_awake": wp.array(_initial_body_awake(mjm, nworld, False), dtype=int),
  }
  for f in dataclasses.fields(types.Data):
    if f.name in d_kwargs:
      continue
    d_kwargs[f.name] = _create_array(None, f.type, sizes)

  d = types.Data(**d_kwargs)

  _allocate_island_arrays(mjm, d, nworld, njmax, mjd)
  _allocate_compact_arrays(mjm, d, nworld, sizes["nvmax_pad"], sizes["njmax_pad"], compact_alloc)
  d.ncdof.zero_()
  d.dof_cdof.fill_(-1)
  d.cdof_dof.fill_(-1)

  warp_util.mark_batched(d)
  return d


def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int = 1,
  nconmax: Optional[int] = None,
  nccdmax: Optional[int] = None,
  njmax: Optional[int] = None,
  njmax_nnz: Optional[int] = None,
  naconmax: Optional[int] = None,
  naccdmax: Optional[int] = None,
  nvmax: Optional[int] = None,
) -> types.Data:
  """Moves data from host to a device.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    mjd: The data object containing current state and output arrays (host).
    nworld: The number of worlds.
    nconmax: Number of contacts to allocate per world.  Contacts exist in large
             heterogenous arrays: one world may have more than nconmax contacts.
    nccdmax: Number of CCD contacts to allocate per world. Same semantics as nconmax.
    njmax: Number of constraints to allocate per world.  Constraint arrays are
           batched by world: no world may have more than njmax constraints.
    njmax_nnz: Number of non-zeros in constraint Jacobian (sparse). Defaults to njmax * nv.
    naconmax: Number of contacts to allocate for all worlds. Overrides nconmax.
    naccdmax: Maximum number of CCD contacts. Defaults to naconmax.
    nvmax: Capacity for compacted active DOFs per world. Defaults to nv.

  Returns:
    The data object containing the current state and output arrays (device).
  """
  # TODO(team): move nconmax and njmax to Model?
  # TODO(team): decide what to do about uninitialized warp-only fields created by put_data
  #             we need to ensure these are only workspace fields and don't carry state

  if nconmax is None:
    nconmax = _default_nconmax(mjm, mjd)

  if njmax is None:
    njmax = _default_njmax(mjm, mjd)

  sleep_enabled = bool(mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and not bool(
    mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ISLAND
  )
  compact_alloc = sleep_enabled or (nvmax is not None)

  if nvmax is None:
    nvmax = mjm.nv

  if nconmax < 0:
    raise ValueError("nconmax must be >= 0")

  if njmax < 0:
    raise ValueError("njmax must be >= 0")

  if nvmax < 0 or nvmax > mjm.nv:
    raise ValueError(f"nvmax ({nvmax}) must be in [0, nv ({mjm.nv})]")

  if nworld < 1:
    raise ValueError(f"nworld must be >= 1")

  naconmax_is_input = naconmax is not None
  naconmax = _resolve_batch_size(naconmax, nconmax, nworld, 0)
  if naconmax < 0:
    raise ValueError("naconmax must be >= 0")

  if not naconmax_is_input and mjd.ncon > nconmax:
    raise ValueError(f"nconmax overflow (nconmax must be >= {mjd.ncon})")
  elif naconmax < mjd.ncon * nworld:
    raise ValueError(f"naconmax overflow (naconmax must be >= {mjd.ncon * nworld})")

  naccdmax = _resolve_batch_size(naccdmax, nccdmax, nworld, naconmax)

  if naccdmax < 0:
    raise ValueError("naccdmax must be >= 0")
  elif naccdmax > naconmax:
    raise ValueError(f"naccdmax ({naccdmax}) must be <= naconmax ({naconmax})")

  if nccdmax is None:
    nccdmax = nconmax
  else:
    if nccdmax < 0:
      raise ValueError("nccdmax must be >= 0")
    elif nccdmax > nconmax:
      raise ValueError(f"nccdmax ({nccdmax}) must be <= nconmax ({nconmax})")

  if mjd.nefc > njmax:
    raise ValueError(f"njmax overflow (njmax must be >= {mjd.nefc})")

  nv_compact = nvmax < mjm.nv

  sizes = dict(
    {"*": 1},
    **{
      f.name: (getattr(mjm, f.name, 0) if f.name == "nplastic" else getattr(mjm, f.name, None))
      for f in dataclasses.fields(types.Model)
      if f.type is int
    },
  )
  condim_arrays = [np.array([0]), mjm.geom_condim, mjm.pair_dim]
  if mjm.nflex > 0:
    condim_arrays.append(mjm.flex_condim)
  sizes["nmaxcondim"] = np.concatenate(condim_arrays).max()
  sizes["nmaxpyramid"] = np.maximum(1, 2 * (sizes["nmaxcondim"] - 1))
  tile_size = types.TILE_SIZE_JTDAJ_SPARSE if is_sparse(mjm) else types.TILE_SIZE_JTDAJ_DENSE
  sizes["njmax_pad"], sizes["nv_pad"] = _get_padded_sizes(
    mjm.nv,
    njmax,
    is_sparse(mjm),
    tile_size,
    augment_cholesky=mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON and mjm.nv > 32,
  )
  sizes["nworld"] = nworld
  sizes["naconmax"] = naconmax
  sizes["njmax"] = njmax
  sizes["nvmax"] = nvmax
  sizes["nvmax_pad"] = _nvmax_pad(nvmax)
  sizes["nvmax_pad_sq"] = sizes["nvmax_pad"] * sizes["nvmax_pad"]
  sizes["nflexface"] = _get_nflexface(mjm)
  sizes["nflexrodhinge"], sizes["nflexrodedge"] = _get_nflexrod_counts(mjm)
  efm_d = _get_efm_data(mjm)
  sizes["nefmK"] = int(efm_d.nefmK) if efm_d else 0
  lay = m_block_layout(mjm)
  qld_total = lay["total"] + (mjm.nC if lay["has_sparse"] else 0)
  sizes["qld_total"] = qld_total

  if njmax_nnz is None:
    if is_sparse(mjm):
      njmax_nnz = _default_njmax_nnz(mjm, nconmax, njmax)
    else:
      njmax_nnz = njmax * mjm.nv

  # Capture sleep state before mj_kinematics, which resets tree_asleep as a side effect.
  tree_asleep_init = mjd.tree_asleep.copy()
  body_awake_init = mjd.body_awake.copy()

  # Ensure kinematic state is populated. mujoco.MjData() does not call mj_kinematics, so a freshly
  # created mjd has zero geom positions. Static geoms are never updated by the physics loop
  # (see smooth.py geom_kinematics), so without this call they would remain at (0,0,0).
  mujoco.mj_kinematics(mjm, mjd)
  # Populate smooth position-dependent terms and mass matrix without invoking mj_collision.
  if np.all(mjd.M == 0):
    mujoco.mj_comPos(mjm, mjd)
    mujoco.mj_camlight(mjm, mjd)
    mujoco.mj_tendon(mjm, mjd)
    mujoco.mj_crb(mjm, mjd)
    mujoco.mj_factorM(mjm, mjd)

  # create contact
  contact_kwargs = {"efc_address": None, "worldid": None, "type": None, "geomcollisionid": None}
  for f in dataclasses.fields(types.Contact):
    if f.name in contact_kwargs:
      continue
    if f.name in ["flex", "elem", "vert"] and mjm.nflex == 0:
      contact_kwargs[f.name] = wp.empty(0, dtype=wp.vec2i)
      continue
    if f.name == "surfacevel" and not util_pkg.check_version("mujoco>=3.17.1"):
      val = None
      if mjm.flg_surfacevel and mjd.ncon > 0:
        svel = np.zeros((mjd.ncon, 3))
        sang = np.zeros((mjd.ncon, 3))
        for side, sgn in ((0, -1.0), (1, 1.0)):
          geom = mjd.contact.geom[: mjd.ncon, side]
          valid = geom >= 0
          g = geom[valid]
          sv = mjm.geom_surfacevel[g]
          xmat = mjd.geom_xmat[g].reshape((-1, 3, 3))
          ang = np.einsum("nij,nj->ni", xmat, sv[:, 3:])
          lin = np.einsum("nij,nj->ni", xmat, sv[:, :3]) + np.cross(ang, mjd.contact.pos[: mjd.ncon][valid] - mjd.geom_xpos[g])
          svel[valid] += sgn * lin
          sang[valid] += sgn * ang
        frame = mjd.contact.frame[: mjd.ncon].reshape((-1, 3, 3))
        val = np.stack(
          (
            np.einsum("ni,ni->n", frame[:, 1], svel),
            np.einsum("ni,ni->n", frame[:, 2], svel),
            np.einsum("ni,ni->n", frame[:, 0], sang),
          ),
          axis=-1,
        )
        val = np.tile(val, (nworld, 1))
        val = np.pad(val, ((0, naconmax - val.shape[0]), (0, 0)))
      contact_kwargs[f.name] = _create_array(val, f.type, sizes)
      continue
    val = getattr(mjd.contact, f.name)
    val = np.tile(val, (nworld,) + (1,) * (val.ndim - 1))
    width = ((0, naconmax - val.shape[0]),) + ((0, 0),) * (val.ndim - 1)
    val = np.pad(val, width)
    contact_kwargs[f.name] = _create_array(val, f.type, sizes)

  contact = types.Contact(**contact_kwargs)

  contact.efc_address = np.full((naconmax, sizes["nmaxpyramid"]), -1, dtype=int)
  for i in range(mjd.ncon):
    efc_address = mjd.contact.efc_address[i]
    if efc_address == -1:
      continue
    condim = mjd.contact.dim[i]
    ndim = max(1, 2 * (condim - 1)) if mjm.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL else condim
    for j in range(nworld):
      contact.efc_address[j * mjd.ncon + i, :ndim] = efc_address + np.arange(ndim)

  contact.efc_address = wp.array(contact.efc_address, dtype=int)
  contact.worldid = np.pad(np.repeat(np.arange(nworld), mjd.ncon), (0, naconmax - nworld * mjd.ncon))
  contact.worldid = wp.array(contact.worldid, dtype=int)
  con_type = np.ones(mjd.ncon, dtype=int)
  if mjd.ncon > 0:
    con_type[mjd.contact.exclude[: mjd.ncon] == CONTACT_EXCLUDE_PASSIVE] = int(types.ContactType.PASSIVE)
  contact.type = wp.array(
    np.pad(np.tile(con_type, nworld), (0, naconmax - nworld * mjd.ncon), constant_values=1),
    dtype=int,
  )
  contact.geomcollisionid = wp.empty((naconmax,), dtype=int)  # TODO(team): set values

  # create efc
  efc = _create_constraint(mjm, nworld, njmax, sizes, mjd)

  # make_constraint builds the block list in-kernel; put_data does not run it, so build it here
  # -- otherwise solving a put_data state would assemble an empty J^T D J.
  if is_sparse(mjm) and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
    jtdaj_adr, jtdaj_nrow = _jtdaj_groups(mjd)
    nblock = jtdaj_adr.shape[0]
    adr_row = np.zeros(njmax, dtype=int)
    nrow_row = np.zeros(njmax, dtype=int)
    adr_row[:nblock] = jtdaj_adr
    nrow_row[:nblock] = jtdaj_nrow
    efc.jtdaj_adr = wp.array(np.tile(adr_row, (nworld, 1)), dtype=int)
    efc.jtdaj_nrow = wp.array(np.tile(nrow_row, (nworld, 1)), dtype=int)
    efc.jtdaj_nblock = wp.array(np.full(nworld, nblock, dtype=int), dtype=int)

  if is_sparse(mjm):
    J_rownnz = np.zeros(njmax, dtype=np.int32)
    J_rowadr = np.zeros(njmax, dtype=np.int32)
    J_colind = np.zeros(njmax_nnz, dtype=np.int32)
    J = np.zeros(njmax_nnz, dtype=np.float64)
    if mjd.nefc:
      if mujoco.mj_isSparse(mjm):
        J_rownnz[: mjd.nefc] = mjd.efc_J_rownnz[: mjd.nefc]
        J_rowadr[: mjd.nefc] = mjd.efc_J_rowadr[: mjd.nefc]
        nnz = int(mjd.efc_J_rownnz[: mjd.nefc].sum())
        J_colind[:nnz] = mjd.efc_J_colind[:nnz]
        J[:nnz] = mjd.efc_J[:nnz]
      else:
        dense_J = mjd.efc_J.reshape((-1, mjm.nv))[: mjd.nefc]
        mujoco.mju_dense2sparse(
          J[: mjd.nefc * mjm.nv], dense_J, J_rownnz[: mjd.nefc], J_rowadr[: mjd.nefc], J_colind[: mjd.nefc * mjm.nv]
        )

    efc.J_rownnz = wp.array(np.tile(J_rownnz, (nworld, 1)), dtype=int)
    efc.J_rowadr = wp.array(np.tile(J_rowadr, (nworld, 1)), dtype=int)
    efc.J_colind = wp.array(np.tile(J_colind, (nworld, 1)).reshape((nworld, 1, -1)), dtype=int)
    efc.J = wp.array(np.tile(J, (nworld, 1)).reshape((nworld, 1, -1)), dtype=float)
  else:
    efc.J_rownnz = wp.zeros((nworld, 0), dtype=int)
    efc.J_rowadr = wp.zeros((nworld, 0), dtype=int)
    efc.J_colind = wp.zeros((nworld, 0, 0), dtype=int)

    mj_efc_J = np.zeros((mjd.nefc, mjm.nv))
    if mjd.nefc:
      if mujoco.mj_isSparse(mjm):
        mujoco.mju_sparse2dense(mj_efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        mj_efc_J = mjd.efc_J.reshape((-1, mjm.nv))[: mjd.nefc]
    efc_J = np.zeros((nworld, sizes["njmax_pad"], sizes["nv_pad"]), dtype=float)
    efc_J[:, : mjd.nefc, : mjm.nv] = np.tile(mj_efc_J, (nworld, 1, 1))
    efc.J = wp.array(efc_J, dtype=float)

  # create data
  d_kwargs = {
    "contact": contact,
    "efc": efc,
    "nworld": nworld,
    "naconmax": naconmax,
    "naccdmax": naccdmax,
    "njmax": njmax,
    "nvmax": nvmax,
    "nvmax_pad": sizes["nvmax_pad"],
    "njmax_pad": sizes["njmax_pad"],
    "njmax_nnz": njmax_nnz,
    # fields set after initialization:
    "solver_niter": None,
    "qLD": None,
    "nacon": None,
    # island arrays
    "nisland": None,
    "tree_island": None,
    "dof_island": None,
    "island_dofadr": None,
    "island_idofadr": None,
    "island_nv": None,
    "island_nefc": None,
    "island_ne": None,
    "island_nf": None,
    "island_iefcadr": None,
    "nidof": None,
    "map_dof2idof": None,
    "map_idof2dof": None,
    "map_efc2iefc": None,
    "map_iefc2efc": None,
    "dof_islandid": None,
    "efc_islandid": None,
    # compact arrays (populated by _allocate_compact_arrays; skip eager allocation)
    **{name: None for name in _COMPACT_DATA_FIELDS},
    "tree_asleep": wp.array(np.tile(tree_asleep_init, (nworld, 1)), dtype=int),
    "tree_awake": wp.array(np.tile((tree_asleep_init < 0).astype(np.int32), (nworld, 1)), dtype=int),
    "body_awake": wp.array(np.tile(body_awake_init.astype(np.int32), (nworld, 1)), dtype=int),
    # plastic
    "plastic": (
      wp.array(np.tile(mjd.plastic, (nworld, 1)), shape=(nworld, getattr(mjm, "nplastic", 0)), dtype=float)
      if getattr(mjm, "nplastic", 0) > 0 and hasattr(mjd, "plastic")
      else wp.zeros((nworld, 0), dtype=float)
    ),
  }
  for f in dataclasses.fields(types.Data):
    if f.name in d_kwargs:
      continue
    val = getattr(mjd, f.name, None)
    if f.name == "efm_K_val" and val is not None and val.size != sizes["nefmK"]:
      val = efm_d.efm_K_val if efm_d else None
    elif f.name == "efm_ts":
      val = np.zeros(mjm.ntendon, dtype=np.float32)
      if mjd.nefmT > 0 and mjd.efm_ts is not None and mjd.efm_tid is not None:
        val[mjd.efm_tid[: mjd.nefmT]] = mjd.efm_ts[: mjd.nefmT]
    elif f.name == "dof_frictionadr":
      # efc address of each DOF's friction row, as make_constraint writes it
      val = np.full(mjm.nv, -1, dtype=np.int32)
      rows = np.flatnonzero(mjd.efc_type[: min(mjd.nefc, njmax)] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
      val[mjd.efc_id[rows]] = rows
    elif f.name == "efm_as":
      val = np.zeros(mjm.nactuator, dtype=np.float32)
      if mjd.nefmA > 0 and mjd.efm_as is not None and mjd.efm_aid is not None:
        val[mjd.efm_aid[: mjd.nefmA]] = mjd.efm_as[: mjd.nefmA]
    d_kwargs[f.name] = _create_array(val, f.type, sizes)

  d = types.Data(**d_kwargs)
  d.solver_niter = wp.full((nworld,), mjd.solver_niter[0], dtype=int)

  # qLD = [packed block Cholesky | nC LDL region]. Block factors store their upper Cholesky
  # packed; the LDL region (present iff some block is sparse) holds MuJoCo's full L'DL factor (only
  # its sparse-block entries are read by the solve).
  qLD = np.zeros(qld_total, dtype=np.float32)
  if lay["total"]:
    Mfull = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sym2dense(Mfull, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    for size, starts in list(lay["scalar_tiles"].items()) + list(lay["gather_tiles"].items()):
      for start in starts:
        off = lay["dof_adr"][start]
        if off < 0:
          continue
        blk = Mfull[start : start + size, start : start + size]
        if blk.any():
          qLD[off : off + size * size] = np.linalg.cholesky(blk).T.reshape(-1)
  if lay["has_sparse"]:
    qLD[lay["total"] :] = mjd.qLD
  d.qLD = wp.array(np.full((nworld, qld_total), qLD), dtype=float)

  _allocate_island_arrays(mjm, d, nworld, njmax, mjd)
  _allocate_compact_arrays(mjm, d, nworld, sizes["nvmax_pad"], sizes["njmax_pad"], compact_alloc)
  d.ncdof.zero_()
  d.dof_cdof.fill_(-1)
  d.cdof_dof.fill_(-1)
  d.nacon = wp.array([mjd.ncon * nworld], dtype=int)

  warp_util.mark_batched(d)
  return d


def get_data_into(
  result: mujoco.MjData,
  mjm: mujoco.MjModel,
  d: types.Data,
  world_id: int = 0,
):
  """Gets data from a device into an existing mujoco.MjData.

  Args:
    result: The data object containing the current state and output arrays (host).
    mjm: The model containing kinematic and dynamic information (host).
    d: The data object containing the current state and output arrays (device).
    world_id: The id of the world to get the data from.
  """
  # nacon and nefc can overflow.  in that case, only pull up to the max contacts and constraints
  nacon = min(d.nacon.numpy()[0], d.naconmax)
  nefc = min(d.nefc.numpy()[world_id], d.njmax)

  contact_worldid = d.contact.worldid.numpy()
  contact_type = d.contact.type.numpy()
  ncon_filter = np.zeros_like(contact_worldid, dtype=bool)
  exported_type = types.ContactType.CONSTRAINT | types.ContactType.PASSIVE
  is_exported = (contact_type[:nacon] & exported_type) != 0
  ncon_filter[:nacon] = (contact_worldid[:nacon] == world_id) & is_exported
  ncon = ncon_filter.sum()

  if ncon != result.ncon or nefc != result.nefc:
    # TODO(team): if sparse, set nJ based on sparse efc_J
    mujoco._functions._realloc_con_efc(result, ncon=ncon, nefc=nefc, nJ=nefc * mjm.nv)

  # Check island compatibility between MjModel and Data
  nisland = int(d.nisland.numpy()[world_id])
  nidof = int(d.nidof.numpy()[world_id]) if nisland > 0 else 0
  if nisland > 0:
    needs_realloc = (
      result.island_idofadr.shape[0] < nisland or result.ifrc_smooth.shape[0] < nidof or result.map_efc2iefc.shape[0] < nefc
    )
    if needs_realloc:
      mujoco._functions._realloc_island(result, nisland=nisland, nidof=nidof)

  ne = d.ne.numpy()[world_id]
  nf = d.nf.numpy()[world_id]
  nl = d.nl.numpy()[world_id]

  # efc indexing
  # mujoco expects contiguous efc ordering for contacts
  # this ordering is not guaranteed with mujoco warp, we enforce order here
  if ncon > 0:
    efc_idx_efl = np.arange(ne + nf + nl)

    contact_dim = d.contact.dim.numpy()[ncon_filter]
    contact_efc_address = d.contact.efc_address.numpy()[ncon_filter]

    efc_idx_c = []
    contact_efc_address_ordered = []
    efc_offset = ne + nf + nl
    for i in range(ncon):
      if contact_efc_address[i, 0] < 0:
        contact_efc_address_ordered.append(-1)
        continue
      dim = contact_dim[i]
      if mjm.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
        ndim = np.maximum(1, 2 * (dim - 1))
      else:
        ndim = dim
      efc_idx_c.append(contact_efc_address[i, :ndim])
      contact_efc_address_ordered.append(efc_offset)
      efc_offset += ndim
    efc_idx = np.concatenate((efc_idx_efl, *efc_idx_c)) if efc_idx_c else efc_idx_efl
    contact_efc_address_ordered = np.array(contact_efc_address_ordered)
  else:
    efc_idx = np.array(np.arange(nefc))
    contact_efc_address_ordered = np.empty(0)

  efc_idx = efc_idx[:nefc]  # dont emit indices for overflow constraints

  result.solver_niter[0] = d.solver_niter.numpy()[world_id]
  result.ncon = ncon
  result.ne = ne
  result.nf = nf
  result.nl = nl
  result.time = d.time.numpy()[world_id]
  result.energy[:] = d.energy.numpy()[world_id]
  result.qpos[:] = d.qpos.numpy()[world_id]
  result.qvel[:] = d.qvel.numpy()[world_id]
  result.act[:] = d.act.numpy()[world_id]
  result.qacc_warmstart[:] = d.qacc_warmstart.numpy()[world_id]
  result.ctrl[:] = d.ctrl.numpy()[world_id]
  result.qfrc_applied[:] = d.qfrc_applied.numpy()[world_id]
  result.xfrc_applied[:] = d.xfrc_applied.numpy()[world_id]
  result.eq_active[:] = d.eq_active.numpy()[world_id]
  result.mocap_pos[:] = d.mocap_pos.numpy()[world_id]
  result.mocap_quat[:] = d.mocap_quat.numpy()[world_id]
  result.qacc[:] = d.qacc.numpy()[world_id]
  result.act_dot[:] = d.act_dot.numpy()[world_id]
  result.xpos[:] = d.xpos.numpy()[world_id]
  result.xquat[:] = d.xquat.numpy()[world_id]
  result.xmat[:] = d.xmat.numpy()[world_id].reshape((-1, 9))
  result.xipos[:] = d.xipos.numpy()[world_id]
  result.ximat[:] = d.ximat.numpy()[world_id].reshape((-1, 9))
  result.xanchor[:] = d.xanchor.numpy()[world_id]
  result.xaxis[:] = d.xaxis.numpy()[world_id]
  result.geom_xpos[:] = d.geom_xpos.numpy()[world_id]
  result.geom_xmat[:] = d.geom_xmat.numpy()[world_id].reshape((-1, 9))
  result.site_xpos[:] = d.site_xpos.numpy()[world_id]
  result.site_xmat[:] = d.site_xmat.numpy()[world_id].reshape((-1, 9))
  result.cam_xpos[:] = d.cam_xpos.numpy()[world_id]
  result.cam_xmat[:] = d.cam_xmat.numpy()[world_id].reshape((-1, 9))
  result.light_xpos[:] = d.light_xpos.numpy()[world_id]
  result.light_xdir[:] = d.light_xdir.numpy()[world_id]
  result.subtree_com[:] = d.subtree_com.numpy()[world_id]
  result.cdof[:] = d.cdof.numpy()[world_id]
  result.cinert[:] = d.cinert.numpy()[world_id]
  result.flexvert_xpos[:] = d.flexvert_xpos.numpy()[world_id]
  if mjm.nflexedge > 0:
    result.flexedge_J[:] = d.flexedge_J.numpy()[world_id].reshape(-1)
  result.flexedge_length[:] = d.flexedge_length.numpy()[world_id]
  result.flexedge_velocity[:] = d.flexedge_velocity.numpy()[world_id]
  if mjm.nflex > 0 and util_pkg.check_version("mujoco>=3.14.1.dev990351372"):
    result.flex_hessian_valid[:] = d.flex_hessian_valid.numpy()[world_id]
    result.flexvert_hessian[:] = d.flexvert_hessian.numpy()[world_id]
    result.flexedge_hessian[:] = d.flexedge_hessian.numpy()[world_id].reshape((-1, 9))
  result.actuator_length[:] = d.actuator_length.numpy()[world_id]
  result.moment_rownnz[:] = d.moment_rownnz.numpy()[world_id]
  result.moment_rowadr[:] = d.moment_rowadr.numpy()[world_id]
  if mjm.nu:
    result.moment_colind[:] = d.moment_colind.numpy()[world_id]
    result.actuator_moment[:] = d.actuator_moment.numpy()[world_id]
  result.crb[:] = d.crb.numpy()[world_id]
  result.qLDiagInv[:] = d.qLDiagInv.numpy()[world_id]
  result.ten_velocity[:] = d.ten_velocity.numpy()[world_id]
  result.actuator_velocity[:] = d.actuator_velocity.numpy()[world_id]
  result.cvel[:] = d.cvel.numpy()[world_id]
  result.cdof_dot[:] = d.cdof_dot.numpy()[world_id]
  result.qfrc_bias[:] = d.qfrc_bias.numpy()[world_id]
  result.qfrc_spring[:] = d.qfrc_spring.numpy()[world_id]
  result.qfrc_damper[:] = d.qfrc_damper.numpy()[world_id]
  result.qfrc_gravcomp[:] = d.qfrc_gravcomp.numpy()[world_id]
  result.qfrc_fluid[:] = d.qfrc_fluid.numpy()[world_id]
  result.qfrc_adhesion[:] = d.qfrc_adhesion.numpy()[world_id]
  result.qfrc_passive[:] = d.qfrc_passive.numpy()[world_id]
  result.subtree_linvel[:] = d.subtree_linvel.numpy()[world_id]
  result.subtree_angmom[:] = d.subtree_angmom.numpy()[world_id]
  result.actuator_force[:] = d.actuator_force.numpy()[world_id]
  result.qfrc_actuator[:] = d.qfrc_actuator.numpy()[world_id]
  result.qfrc_smooth[:] = d.qfrc_smooth.numpy()[world_id]
  result.qacc_smooth[:] = d.qacc_smooth.numpy()[world_id]
  result.qfrc_constraint[:] = d.qfrc_constraint.numpy()[world_id]
  result.qfrc_inverse[:] = d.qfrc_inverse.numpy()[world_id]

  if mjm.nhistory > 0:
    result.history[:] = d.history.numpy()[world_id]

  if getattr(mjm, "nplastic", 0) > 0:
    result.plastic[:] = d.plastic.numpy()[world_id]

  if mjm.nuserdata > 0:
    result.userdata[:] = d.userdata.numpy()[world_id]

  # contact
  result.contact.dist[:ncon] = d.contact.dist.numpy()[ncon_filter]
  result.contact.pos[:ncon] = d.contact.pos.numpy()[ncon_filter]
  result.contact.frame[:ncon] = d.contact.frame.numpy()[ncon_filter].reshape((-1, 9))
  result.contact.includemargin[:ncon] = d.contact.includemargin.numpy()[ncon_filter]
  result.contact.friction[:ncon] = d.contact.friction.numpy()[ncon_filter]
  result.contact.solref[:ncon] = d.contact.solref.numpy()[ncon_filter]
  result.contact.solreffriction[:ncon] = d.contact.solreffriction.numpy()[ncon_filter]
  result.contact.solimp[:ncon] = d.contact.solimp.numpy()[ncon_filter]
  result.contact.adhesion[:ncon] = d.contact.adhesion.numpy()[ncon_filter]
  if util_pkg.check_version("mujoco>=3.17.1"):
    result.contact.surfacevel[:ncon] = d.contact.surfacevel.numpy()[ncon_filter]
  result.contact.dim[:ncon] = d.contact.dim.numpy()[ncon_filter]
  result.contact.geom[:ncon] = d.contact.geom.numpy()[ncon_filter]
  is_passive = (contact_type[ncon_filter] & types.ContactType.PASSIVE) != 0
  result.contact.exclude[:ncon] = np.where(is_passive, CONTACT_EXCLUDE_PASSIVE, CONTACT_EXCLUDE_INCLUDE)
  if mjm.nflex > 0:
    result.contact.flex[:ncon] = d.contact.flex.numpy()[ncon_filter]
    result.contact.elem[:ncon] = d.contact.elem.numpy()[ncon_filter]
    result.contact.vert[:ncon] = d.contact.vert.numpy()[ncon_filter]
  result.contact.efc_address[:ncon] = contact_efc_address_ordered[:ncon]

  result.M[:] = d.M.numpy()[world_id]
  _lay = m_block_layout(mjm)
  if _lay["scalar_tiles"] or _lay["gather_tiles"]:
    # Block factors do not use MuJoCo's LDL representation.
    if not np.all(result.M == 0):
      mujoco.mj_factorM(mjm, result)
  else:
    # Pure sparse: qLD is exactly MuJoCo's nC LDL factor.
    result.qLD[:] = d.qLD.numpy()[world_id]

  if nefc > 0:
    if is_sparse(mjm):
      efc_J = np.zeros((nefc, mjm.nv))
      mujoco.mju_sparse2dense(
        efc_J,
        d.efc.J.numpy()[world_id, 0],
        d.efc.J_rownnz.numpy()[world_id, :nefc],
        d.efc.J_rowadr.numpy()[world_id, :nefc],
        d.efc.J_colind.numpy()[world_id, 0],
      )
    else:
      efc_J = d.efc.J.numpy()[world_id, :nefc, : mjm.nv]

    # write to mujoco result (format depends on mj_isSparse)
    if mujoco.mj_isSparse(mjm):
      mujoco.mju_dense2sparse(
        result.efc_J,
        efc_J[efc_idx],
        result.efc_J_rownnz,
        result.efc_J_rowadr,
        result.efc_J_colind,
      )
    else:
      result.efc_J[: nefc * mjm.nv] = efc_J[efc_idx].flatten()

  # efc
  result.efc_type[:] = d.efc.type.numpy()[world_id, efc_idx]
  result.efc_id[:] = d.efc.id.numpy()[world_id, efc_idx]
  result.efc_pos[:] = d.efc.pos.numpy()[world_id, efc_idx]
  result.efc_margin[:] = d.efc.margin.numpy()[world_id, efc_idx]
  result.efc_D[:] = d.efc.D.numpy()[world_id, efc_idx]
  result.efc_vel[:] = d.efc.vel.numpy()[world_id, efc_idx]
  result.efc_aref[:] = d.efc.aref.numpy()[world_id, efc_idx]
  result.efc_frictionloss[:] = d.efc.frictionloss.numpy()[world_id, efc_idx]
  result.efc_state[:] = d.efc.state.numpy()[world_id, efc_idx]
  result.efc_force[:] = d.efc.force.numpy()[world_id, efc_idx]

  # rne_postconstraint
  result.cacc[:] = d.cacc.numpy()[world_id]
  result.cfrc_int[:] = d.cfrc_int.numpy()[world_id]
  result.cfrc_ext[:] = d.cfrc_ext.numpy()[world_id]

  # tendon
  result.ten_length[:] = d.ten_length.numpy()[world_id]
  if mjm.ntendon > 0:
    result.ten_J[:] = d.ten_J.numpy()[world_id]
  result.ten_wrapadr[:] = d.ten_wrapadr.numpy()[world_id]
  result.ten_wrapnum[:] = d.ten_wrapnum.numpy()[world_id]
  result.wrap_obj[:] = d.wrap_obj.numpy()[world_id]
  result.wrap_xpos[:] = d.wrap_xpos.numpy()[world_id]

  # sensors
  result.sensordata[:] = d.sensordata.numpy()[world_id]

  # sleep
  result.tree_asleep[:] = d.tree_asleep.numpy()[world_id]
  result.tree_awake[:] = d.tree_awake.numpy()[world_id]
  result.body_awake[:] = d.body_awake.numpy()[world_id]

  # discrete integrator
  if mjm.opt.integrator == types.IntegratorType.DISCRETE:
    result.efm_active = 1
    if result.efm_c is not None and result.efm_c.size == mjm.nv:
      result.efm_c[:] = d.efm_c.numpy()[world_id]
    if result.efm_diag is not None and result.efm_diag.size == mjm.nv:
      result.efm_diag[:] = d.efm_diag.numpy()[world_id]

  # islands
  result.nisland = nisland
  result.nidof = d.nidof.numpy()[world_id]
  if d.tree_island.shape[1] > 0 and nisland:
    result.tree_island[:] = d.tree_island.numpy()[world_id]
    result.dof_island[:] = d.dof_island.numpy()[world_id]
    result.island_idofadr[:nisland] = d.island_idofadr.numpy()[world_id, :nisland]
    result.island_dofadr[:nisland] = d.island_dofadr.numpy()[world_id, :nisland]
    result.island_nv[:nisland] = d.island_nv.numpy()[world_id, :nisland]
    result.island_nefc[:nisland] = d.island_nefc.numpy()[world_id, :nisland]
    result.island_ne[:nisland] = d.island_ne.numpy()[world_id, :nisland]
    result.island_nf[:nisland] = d.island_nf.numpy()[world_id, :nisland]
    result.island_iefcadr[:nisland] = d.island_iefcadr.numpy()[world_id, :nisland]
    nv = mjm.nv
    result.map_dof2idof[:nv] = d.map_dof2idof.numpy()[world_id, :nv]
    result.map_idof2dof[:nv] = d.map_idof2dof.numpy()[world_id, :nv]
    result.efc_island[:] = d.efc.island.numpy()[world_id, efc_idx]
    result.map_efc2iefc[:nefc] = d.map_efc2iefc.numpy()[world_id, :nefc]
    result.map_iefc2efc[:nefc] = d.map_iefc2efc.numpy()[world_id, :nefc]


def reset_data(m: types.Model, d: types.Data, reset: Optional[wp.array] = None):
  """Clear data, set defaults; optionally by world.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    reset: Per-world bitmask (bool or integer array). Reset if nonzero/True.

  Raises:
    ValueError: If reset is specified but its shape is not (d.nworld,) or its
      dtype is not bool or integer.
  """
  sleep_enabled = bool(m.opt.enableflags & types.EnableBit.SLEEP)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_xfrc_applied(reset_in: wp.array[bool], xfrc_applied_out: wp.array2d[wp.spatial_vector]):
    worldid, bodyid, elemid = wp.tid()

    if wp.static(reset is not None):
      if not reset_in[worldid]:
        return

    xfrc_applied_out[worldid, bodyid][elemid] = 0.0

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_M(reset_in: wp.array[bool], M_out: wp.array2d[float]):
    worldid, elemid = wp.tid()

    if wp.static(reset is not None):
      if not reset_in[worldid]:
        return

    M_out[worldid, elemid] = 0.0

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_nworld(
    # Model:
    nq: int,
    nv: int,
    nu: int,
    na: int,
    nbody: int,
    ntree: int,
    nflex: int,
    neq: int,
    nuserdata: int,
    nsensordata: int,
    qpos0: wp.array2d[float],
    eq_active0: wp.array[bool],
    # Data in:
    nworld_in: int,
    # In:
    reset_in: wp.array[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    ne_out: wp.array[int],
    nf_out: wp.array[int],
    nl_out: wp.array[int],
    nefc_out: wp.array[int],
    ntree_awake_out: wp.array[int],
    nbody_awake_out: wp.array[int],
    nv_awake_out: wp.array[int],
    time_out: wp.array[float],
    energy_out: wp.array[wp.vec2],
    qpos_out: wp.array2d[float],
    qvel_out: wp.array2d[float],
    act_out: wp.array2d[float],
    qacc_warmstart_out: wp.array2d[float],
    ctrl_out: wp.array2d[float],
    qfrc_applied_out: wp.array2d[float],
    eq_active_out: wp.array2d[bool],
    qacc_out: wp.array2d[float],
    act_dot_out: wp.array2d[float],
    userdata_out: wp.array2d[float],
    sensordata_out: wp.array2d[float],
    flex_hessian_valid_out: wp.array2d[bool],
    nacon_out: wp.array[int],
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()

    if wp.static(reset is not None):
      if not reset_in[worldid]:
        return

    solver_niter_out[worldid] = 0
    if worldid == 0:
      nacon_out[0] = 0
    ne_out[worldid] = 0
    nf_out[worldid] = 0
    nl_out[worldid] = 0
    nefc_out[worldid] = 0
    time_out[worldid] = 0.0
    energy_out[worldid] = wp.vec2(0.0, 0.0)
    ntree_awake_out[worldid] = ntree
    nbody_awake_out[worldid] = nbody
    nv_awake_out[worldid] = nv
    qpos0_id = worldid % qpos0.shape[0]
    for i in range(nq):
      qpos_out[worldid, i] = qpos0[qpos0_id, i]
      if i < nv:
        qvel_out[worldid, i] = 0.0
        qacc_warmstart_out[worldid, i] = 0.0
        qfrc_applied_out[worldid, i] = 0.0
        qacc_out[worldid, i] = 0.0
    for i in range(nu):
      ctrl_out[worldid, i] = 0.0
    for i in range(na):
      act_out[worldid, i] = 0.0
      act_dot_out[worldid, i] = 0.0
    for i in range(neq):
      eq_active_out[worldid, i] = eq_active0[i]
    for i in range(nflex):
      flex_hessian_valid_out[worldid, i] = False
    for i in range(nsensordata):
      sensordata_out[worldid, i] = 0.0
    for i in range(nuserdata):
      userdata_out[worldid, i] = 0.0
    overflow_out[worldid] = 0

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_mocap(
    # Model:
    body_mocapid: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    # In:
    reset_in: wp.array[bool],
    # Data out:
    mocap_pos_out: wp.array2d[wp.vec3],
    mocap_quat_out: wp.array2d[wp.quat],
  ):
    worldid, bodyid = wp.tid()

    if wp.static(reset is not None):
      if not reset_in[worldid]:
        return

    mocapid = body_mocapid[bodyid]

    if mocapid >= 0:
      mocap_pos_out[worldid, mocapid] = body_pos[worldid % body_pos.shape[0], bodyid]
      mocap_quat_out[worldid, mocapid] = body_quat[worldid % body_quat.shape[0], bodyid]

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_contact(
    # Data in:
    nacon_in: wp.array[int],
    # In:
    reset_in: wp.array[bool],
    nefcaddress: int,
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[types.vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[types.vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_flex_out: wp.array[wp.vec2i],
    contact_elem_out: wp.array[wp.vec2i],
    contact_vert_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    contact_adhesion_out: wp.array[float],
    contact_surfacevel_out: wp.array[wp.vec3],
  ):
    conid = wp.tid()

    if conid >= nacon_in[0]:
      return

    worldid = contact_worldid_out[conid]
    if wp.static(reset is not None):
      if worldid >= 0:
        if not reset_in[worldid]:
          return

    contact_dist_out[conid] = 0.0
    contact_pos_out[conid] = wp.vec3(0.0)
    contact_frame_out[conid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    contact_includemargin_out[conid] = 0.0
    contact_friction_out[conid] = types.vec5(0.0, 0.0, 0.0, 0.0, 0.0)
    contact_solref_out[conid] = wp.vec2(0.0, 0.0)
    contact_solreffriction_out[conid] = wp.vec2(0.0, 0.0)
    contact_solimp_out[conid] = types.vec5(0.0, 0.0, 0.0, 0.0, 0.0)
    contact_dim_out[conid] = 0
    contact_geom_out[conid] = wp.vec2i(0, 0)
    if contact_flex_out.shape[0] > 0:
      contact_flex_out[conid] = wp.vec2i(0, 0)
    if contact_elem_out.shape[0] > 0:
      contact_elem_out[conid] = wp.vec2i(0, 0)
    if contact_vert_out.shape[0] > 0:
      contact_vert_out[conid] = wp.vec2i(0, 0)
    for i in range(nefcaddress):
      contact_efc_address_out[conid, i] = -1
    contact_worldid_out[conid] = 0
    contact_type_out[conid] = 0
    contact_geomcollisionid_out[conid] = 0
    contact_adhesion_out[conid] = 0.0
    contact_surfacevel_out[conid] = wp.vec3(0.0)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_sleep(
    # Model:
    nv: int,
    nbody: int,
    ntree: int,
    body_mocapid: wp.array[int],
    body_treeid: wp.array[int],
    # In:
    mj_minawake: int,
    reset_in: wp.array[bool],
    # Data out:
    tree_asleep_out: wp.array2d[int],
    tree_awake_out: wp.array2d[int],
    body_awake_out: wp.array2d[int],
    body_awake_ind_out: wp.array2d[int],
    dof_awake_ind_out: wp.array2d[int],
  ):
    worldid, elemid = wp.tid()

    if wp.static(reset is not None):
      if not reset_in[worldid]:
        return

    if elemid < ntree:
      tree_asleep_out[worldid, elemid] = -(1 + mj_minawake)
      tree_awake_out[worldid, elemid] = 1

    if elemid < nbody:
      if body_treeid[elemid] < 0:
        if body_mocapid[elemid] >= 0:
          body_awake_out[worldid, elemid] = int(types.SleepState.AWAKE)
        else:
          body_awake_out[worldid, elemid] = int(types.SleepState.STATIC)
      else:
        body_awake_out[worldid, elemid] = int(types.SleepState.AWAKE)
      body_awake_ind_out[worldid, elemid] = elemid

    if elemid < nv:
      dof_awake_ind_out[worldid, elemid] = elemid

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def reset_plastic(
    # Model:
    njnt: int,
    ntendon: int,
    nplastic: int,
    qpos_spring: wp.array2d[float],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_plasticadr: wp.array[int],
    tendon_plasticadr: wp.array[int],
    tendon_lengthspring: wp.array2d[wp.vec2],
    # In:
    reset_in: wp.array[bool],
    # Data out:
    plastic_out: wp.array2d[float],
  ):
    worldid = wp.tid()

    if wp.static(reset is not None):
      if not reset_in[worldid]:
        return

    for i in range(nplastic):
      plastic_out[worldid, i] = 0.0

    qpos_spring_id = worldid % qpos_spring.shape[0]
    for j in range(njnt):
      padr = jnt_plasticadr[j]
      if padr >= 0:
        qadr = jnt_qposadr[j]
        jtype = jnt_type[j]
        nq_j = 4 if jtype == types.JointType.BALL else 1
        for k in range(nq_j):
          plastic_out[worldid, padr + k] = qpos_spring[qpos_spring_id, qadr + k]

    tendon_lengthspring_id = worldid % tendon_lengthspring.shape[0]
    for t in range(ntendon):
      padr = tendon_plasticadr[t]
      if padr >= 0:
        lengthspring = tendon_lengthspring[tendon_lengthspring_id, t]
        plastic_out[worldid, padr + 0] = lengthspring[0]
        plastic_out[worldid, padr + 1] = lengthspring[1]

  if reset is None:
    reset_input = wp.ones(d.nworld, dtype=bool)
  elif isinstance(reset, wp.array):
    if reset.shape != (d.nworld,):
      raise ValueError(f"reset array must have shape ({d.nworld},), got {reset.shape}.")
    if reset.dtype == wp.bool:
      reset_input = reset
    elif wp.types.type_is_int(reset.dtype):
      reset_input = wp.empty(d.nworld, dtype=bool)
      wp.utils.array_cast(reset, reset_input)
    else:
      raise ValueError(f"reset array must be of bool or integer type, got {reset.dtype}.")
  else:
    raise ValueError(f"reset must be None or a wp.array, got {type(reset)}.")

  wp.launch(reset_xfrc_applied, dim=(d.nworld, m.nbody, 6), inputs=[reset_input], outputs=[d.xfrc_applied])
  wp.launch(
    reset_M,
    dim=(d.nworld, d.M.shape[1]),
    inputs=[reset_input],
    outputs=[d.M],
  )

  # set mocap_pos/quat = body_pos/quat for mocap bodies
  wp.launch(
    reset_mocap,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_mocapid, m.body_pos, m.body_quat, reset_input],
    outputs=[d.mocap_pos, d.mocap_quat],
  )

  # clear contacts
  wp.launch(
    reset_contact,
    dim=d.naconmax,
    inputs=[d.nacon, reset_input, d.contact.efc_address.shape[1]],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.flex,
      d.contact.elem,
      d.contact.vert,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.contact.adhesion,
      d.contact.surfacevel,
    ],
  )

  wp.launch(
    reset_sleep,
    dim=(d.nworld, max(m.ntree, m.nbody, m.nv)),
    inputs=[m.nv, m.nbody, m.ntree, m.body_mocapid, m.body_treeid, types.MJ_MINAWAKE, reset_input],
    outputs=[
      d.tree_asleep,
      d.tree_awake,
      d.body_awake,
      d.body_awake_ind,
      d.dof_awake_ind,
    ],
  )

  wp.launch(
    reset_nworld,
    dim=d.nworld,
    inputs=[
      m.nq,
      m.nv,
      m.nu,
      m.na,
      m.nbody,
      m.ntree,
      m.nflex,
      m.neq,
      m.nuserdata,
      m.nsensordata,
      m.qpos0,
      m.eq_active0,
      d.nworld,
      reset_input,
    ],
    outputs=[
      d.solver_niter,
      d.ne,
      d.nf,
      d.nl,
      d.nefc,
      d.ntree_awake,
      d.nbody_awake,
      d.nv_awake,
      d.time,
      d.energy,
      d.qpos,
      d.qvel,
      d.act,
      d.qacc_warmstart,
      d.ctrl,
      d.qfrc_applied,
      d.eq_active,
      d.qacc,
      d.act_dot,
      d.userdata,
      d.sensordata,
      d.flex_hessian_valid,
      d.nacon,
      d.overflow,
    ],
  )

  if m.nhistory > 0:
    history.reset_history(m, d, reset=reset_input)

  if m.nplastic > 0:
    wp.launch(
      reset_plastic,
      dim=d.nworld,
      inputs=[
        m.njnt,
        m.ntendon,
        m.nplastic,
        m.qpos_spring,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_plasticadr,
        m.tendon_plasticadr,
        m.tendon_lengthspring,
        reset_input,
      ],
      outputs=[d.plastic],
    )

  if sleep_enabled:
    sleep.update_sleep(m, d)


def reset_data_keyframe(m: types.Model, d: types.Data, key: int | wp.array):
  """Reset data, set fields from specified keyframe.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    key: The keyframe index to initialize the data with. If a plain integer is
      given, all worlds are reset to that keyframe (in this case, if keyframe
      index is < 0 or >= m.nkey, a ValueError is raised). If an array of
      integers is given, each value of the array indicates the target
      keyframe of the corresponding world (in this case, worlds whose
      keyframe index is < 0 or >= m.nkey are not reset).

  Raises:
    ValueError: If key is an int and key<0 or key>=m.nkey.
    ValueError: If key is a wp.array but its shape is not (d.nworld,) or its
      dtype is not int.
  """
  # Resolve target keyframe index
  if isinstance(key, wp.array):
    if key.shape != (d.nworld,):
      raise ValueError(f"key array must have shape ({d.nworld},), got {key.shape}.")
    if not wp.types.type_is_int(key.dtype):
      raise ValueError(f"key array must be of integer type, got {key.dtype}.")
    key_input = key
  elif isinstance(key, (int, np.integer)):
    key = int(key)
    if key < 0 or key >= m.nkey:
      raise ValueError(f"key ({key}) must be in [0, {m.nkey}).")
    key_input = wp.full(d.nworld, key, dtype=int)
  else:
    raise ValueError(f"key must be an int or a wp.array, got {type(key)}.")

  # Worlds whose keyframe index is out of bounds are left untouched.
  @wp.kernel(module="unique", enable_backward=False)
  def valid_key_mask(
    # Model:
    nkey: int,
    # In:
    key_in: wp.array[int],
    # Out:
    mask_out: wp.array[bool],
  ):
    worldid = wp.tid()
    key = key_in[worldid]
    mask_out[worldid] = key >= 0 and key < nkey

  reset_mask = wp.empty(d.nworld, dtype=bool)
  wp.launch(
    valid_key_mask,
    dim=d.nworld,
    inputs=[m.nkey, key_input],
    outputs=[reset_mask],
  )

  # Call normal reset using the mask
  reset_data(m, d, reset_mask)

  # Set time, qpos, qvel, act, ctrl, mocap_pos, mocap_quat, plastic from the keyframe.
  @wp.kernel(module="unique", enable_backward=False)
  def reset_keyframe_data(
    # Model:
    nq: int,
    nv: int,
    nu: int,
    na: int,
    nmocap: int,
    nplastic: int,
    key_time: wp.array[float],
    key_qpos: wp.array2d[float],
    key_qvel: wp.array2d[float],
    key_act: wp.array2d[float],
    key_mpos: wp.array2d[wp.vec3],
    key_mquat: wp.array2d[wp.quat],
    key_ctrl: wp.array2d[float],
    key_plastic: wp.array2d[float],
    # In:
    key_in: wp.array[int],
    reset_in: wp.array[bool],
    # Data out:
    time_out: wp.array[float],
    qpos_out: wp.array2d[float],
    qvel_out: wp.array2d[float],
    act_out: wp.array2d[float],
    plastic_out: wp.array2d[float],
    ctrl_out: wp.array2d[float],
    mocap_pos_out: wp.array2d[wp.vec3],
    mocap_quat_out: wp.array2d[wp.quat],
  ):
    worldid = wp.tid()

    if not reset_in[worldid]:
      return

    key = key_in[worldid]
    time_out[worldid] = key_time[key]
    for i in range(nq):
      qpos_out[worldid, i] = key_qpos[key, i]
    for i in range(nv):
      qvel_out[worldid, i] = key_qvel[key, i]
    for i in range(na):
      act_out[worldid, i] = key_act[key, i]
    for i in range(nmocap):
      mocap_pos_out[worldid, i] = key_mpos[key, i]
      mocap_quat_out[worldid, i] = key_mquat[key, i]
    for i in range(nu):
      ctrl_out[worldid, i] = key_ctrl[key, i]
    for i in range(nplastic):
      plastic_out[worldid, i] = key_plastic[key, i]

  wp.launch(
    reset_keyframe_data,
    dim=d.nworld,
    inputs=[
      m.nq,
      m.nv,
      m.nu,
      m.na,
      m.nmocap,
      m.nplastic,
      m.key_time,
      m.key_qpos,
      m.key_qvel,
      m.key_act,
      m.key_mpos,
      m.key_mquat,
      m.key_ctrl,
      m.key_plastic,
      key_input,
      reset_mask,
    ],
    outputs=[
      d.time,
      d.qpos,
      d.qvel,
      d.act,
      d.plastic,
      d.ctrl,
      d.mocap_pos,
      d.mocap_quat,
    ],
  )


def override_model(model: types.Model | mujoco.MjModel, overrides: dict[str, Any] | Sequence[str]):
  """Overrides model parameters.

  Overrides are of the format:
    opt.iterations = 1
    opt.cone = pyramidal
    opt.disableflags = contact | spring
  """
  enum_fields = {
    "opt.broadphase": types.BroadphaseType,
    "opt.broadphase_filter": types.BroadphaseFilter,
    "opt.cone": types.ConeType,
    "opt.disableflags": types.DisableBit,
    "opt.enableflags": types.EnableBit,
    "opt.integrator": types.IntegratorType,
    "opt.solver": types.SolverType,
    "opt.warn_overflow": types.OverflowType,
    "opt.deterministic": types.DeterminismType,
  }
  # MuJoCo pybind11 enums don't support iteration, so we provide explicit mappings
  mj_enum_fields = {
    "opt.jacobian": {
      "DENSE": mujoco.mjtJacobian.mjJAC_DENSE,
      "SPARSE": mujoco.mjtJacobian.mjJAC_SPARSE,
      "AUTO": mujoco.mjtJacobian.mjJAC_AUTO,
    },
  }
  mjw_only_fields = {
    "opt.broadphase",
    "opt.broadphase_filter",
    "opt.graph_conditional",
    "opt.contact_sensor_maxmatch",
    "opt.warn_overflow",
    "opt.deterministic",
    "opt.run_collision_detection",
    "opt.run_rne_postconstraint",
    "opt.run_qacc_smooth",
  }
  mj_only_fields = {"opt.jacobian", "vis.quality.offsamples"}

  if not isinstance(overrides, dict):
    overrides_dict = {}
    for override in overrides:
      if "=" not in override:
        raise ValueError(f"Invalid override format: {override}")
      k, v = override.split("=", 1)
      overrides_dict[k.strip()] = v.strip()
    overrides = overrides_dict

  for key, val in overrides.items():
    if key == "opt.ls_parallel":
      raise ValueError("ls_parallel was removed in MuJoCo Warp 3.9.1.")
    if key == "opt.ls_parallel_min_step":
      raise ValueError("ls_parallel_min_step was removed in MuJoCo Warp 3.9.1.")

    # skip overrides on MjModel for properties that are only on mjw.Model
    if key in mjw_only_fields and isinstance(model, mujoco.MjModel):
      continue
    if key in mj_only_fields and isinstance(model, types.Model):
      continue

    obj, attrs = model, key.split(".")
    for i, attr in enumerate(attrs):
      if not hasattr(obj, attr):
        raise ValueError(f"Unrecognized model field: {key}")
      if i < len(attrs) - 1:
        obj = getattr(obj, attr)
        continue

      typ = type(getattr(obj, attr))

      if key == "opt.deterministic":
        if isinstance(val, str) and val.upper() in ("TRUE", "FALSE"):
          val = val.upper() == "TRUE"
        if isinstance(val, (bool, np.bool_)):
          val = int(types.DeterminismType.ALL if val else types.DeterminismType.NONE)
        elif isinstance(val, str) and val.isdigit():
          val = int(val)

      if key in mj_enum_fields and isinstance(val, str):
        enum_member = val.strip().upper()
        if enum_member not in mj_enum_fields[key]:
          raise ValueError(f"Unrecognized enum value for {key}: {enum_member}")
        val = mj_enum_fields[key][enum_member]
      elif key in enum_fields and isinstance(val, str):
        # special case: enum value
        enum_members = val.split("|")
        enum_cls = enum_fields[key]
        val = int(getattr(obj, attr)) if any(m.strip().startswith("~") for m in enum_members) else 0
        for enum_member in enum_members:
          enum_member = enum_member.strip().upper()
          is_negated = enum_member.startswith("~")
          name = enum_member[1:].strip() if is_negated else enum_member
          if name not in enum_cls.__members__:
            raise ValueError(f"Unrecognized enum value for {enum_cls.__name__}: {enum_member}")
          if is_negated:
            val &= ~int(enum_cls[name])
          else:
            val |= int(enum_cls[name])
      elif typ is bool and isinstance(val, str):
        # special case: "true", "TRUE", "false", "FALSE" etc.
        if val.upper() not in ("TRUE", "FALSE"):
          raise ValueError(f"Unrecognized value for field: {key}")
        val = val.upper() == "TRUE"
      elif typ is wp.array and isinstance(val, str):
        arr = getattr(obj, attr)
        floats = [float(p) for p in val.strip("[]").split()]
        val = wp.array([arr.dtype(*floats)], dtype=arr.dtype)
      elif typ is np.ndarray and isinstance(val, str):
        arr = getattr(obj, attr)
        val = np.array([float(p) for p in val.strip("[]").split()], dtype=arr.dtype)
      elif typ is np.ndarray:
        arr = getattr(obj, attr)
        val = np.asarray(val, dtype=arr.dtype)
      else:
        val = typ(val)

      setattr(obj, attr, val)


def find_keys(model: mujoco.MjModel, keyname_prefix: str) -> list[int]:
  """Finds keyframes that start with keyname_prefix."""
  keys = []

  for keyid in range(model.nkey):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, keyid)
    if name.startswith(keyname_prefix):
      keys.append(keyid)

  return keys


def make_trajectory(model: mujoco.MjModel, keys: list[int]) -> np.ndarray:
  """Make a ctrl trajectory with linear interpolation."""
  ctrls = []
  prev_ctrl_key = np.zeros(model.nu, dtype=np.float64)
  prev_time, time = 0.0, 0.0

  for key in keys:
    ctrl_key, ctrl_time = model.key_ctrl[key], model.key_time[key]
    if not ctrls and ctrl_time != 0.0:
      raise ValueError("first keyframe must have time 0.0")
    elif ctrls and ctrl_time <= prev_time:
      raise ValueError("keyframes must be in time order")

    while time < ctrl_time:
      frac = (time - prev_time) / (ctrl_time - prev_time)
      ctrls.append(prev_ctrl_key * (1 - frac) + ctrl_key * frac)
      time += model.opt.timestep

    ctrls.append(ctrl_key)
    time += model.opt.timestep
    prev_ctrl_key = ctrl_key
    prev_time = time

  return np.array(ctrls)


def load_trajectory(npz_path: str, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> np.ndarray:
  """Load ctrl sequence from NPZ and interpolate to model timestep.

  Controls are sampled on the model timestep using zero-order hold. ``times``
  may contain one timestamp per control or interval boundaries with one extra
  final timestamp. For per-control timestamps, the final interval repeats the
  preceding interval duration, or one model timestep for a single control.
  Sub-timestep control intervals may be skipped when no model-timestep sample
  falls within them.

  The NPZ file should contain:
    - 'ctrl': array of shape (nstep, nu) with ctrl values
    - 'times': array of shape (nstep,) or (nstep + 1,) with timestamps
    - 'qpos' (optional): array of shape (1, nq) - initial state
    - 'qvel' (optional): array of shape (1, nv) - initial state
  """
  data = np.load(npz_path)
  ctrl = data["ctrl"]
  times = data["times"]

  if ctrl.ndim != 2 or len(ctrl) == 0:
    raise ValueError(f"ctrl must have shape (nstep, nu) with nstep > 0, got {ctrl.shape}")
  if ctrl.shape[1] != mjm.nu:
    raise ValueError(f"ctrl shape {ctrl.shape} does not match model nu={mjm.nu}")
  if times.ndim != 1 or len(times) not in (len(ctrl), len(ctrl) + 1):
    raise ValueError(f"times shape {times.shape} must contain {len(ctrl)} or {len(ctrl) + 1} timestamps")
  if not np.all(np.isfinite(times)):
    raise ValueError("times must be finite")

  intervals = np.diff(times)
  if np.any(intervals <= 0):
    raise ValueError("times must be strictly increasing")

  # set initial state from first frame if available
  if "qpos" in data and data["qpos"].shape[1] == mjm.nq:
    mjd.qpos[:] = data["qpos"][0]
  if "qvel" in data and data["qvel"].shape[1] == mjm.nv:
    mjd.qvel[:] = data["qvel"][0]

  if len(times) == len(ctrl):
    final_dt = intervals[-1] if len(intervals) else mjm.opt.timestep
    times = np.append(times, times[-1] + final_dt)

  n_steps = int(np.round((times[-1] - times[0]) / mjm.opt.timestep))
  sample_times = times[0] + (np.arange(n_steps) + 1e-7) * mjm.opt.timestep
  ctrl_indices = np.searchsorted(times, sample_times, side="right") - 1
  return ctrl[ctrl_indices]
