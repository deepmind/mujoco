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
"""Many simulations of one model, stepped on a thread pool.

Each simulation is its mjSTATE_INTEGRATION row; a call loads the row into an
MjData, runs, copies out the bound derived fields, and saves the row back, on
nthread threads including the caller's. Results are bit-identical to a loop over
one MjData per simulation at any thread count.

  batch = Batch(model, nsim=1024, nthread=8)
  qpos, ctrl, xpos = batch.bind('qpos'), batch.bind('ctrl'), batch.bind('xpos')
  for _ in range(1000):
    ctrl[:] = policy(qpos)
    batch.step()

bind() returns a live (nsim, ...) array. State fields (qpos, qvel, ctrl,
mocap_pos, ...) are views of the state rows: a write is the simulation's state
at its next call. Derived fields (xpos, sensordata, ...) are refreshed after
every call on a simulation, one substep behind the state after step(), as with
mj_step; a derived field bound between calls is zero until the next call.
joint(name), body(name), sensor(name) and the other named views slice the bound
and expanded fields per object, with the attributes of MjData's and MjModel's:
batch.joint('hinge').qpos is (nsim, 1), and batch.body('pole').mass = masses
writes per-simulation masses. set_models(models) runs each simulation on a model
of the caller's instead of expanding fields. get_model(sim) and get_data(sim)
materialize one simulation, to render it or to pass it to any function. Fields
are also attributes, as on MjData: batch.xpos binds xpos, and batch.ctrl = u
writes every simulation's controls. DATA_FIELDS and MODEL_FIELDS name the fields
bind() and expand() accept.

By default an MjData serves many simulations, so memory scales with threads;
only the fields a call computes are then meaningful, and a field it does not
compute (qfrc_inverse after step, quantities computed only for the sensors that
need them) holds another simulation's value. persistent=True keeps one MjData
per simulation instead, which every bound field then reflects, and is required
for models with sleep or IPC enabled. Likewise, engine plugins that keep state
outside plugin_state share it among the simulations a thread runs, unless the
batch is persistent.

The batch's threads run across simulations; the engine's thread pool
(mju_threadpool) runs within one, and the two do not combine. Physics callbacks
(mjcb_*) must be C functions, installed through ctypes: a Python function
installed as one makes the batch's calls raise ValueError.
"""

import copy
import difflib
import functools
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence, Tuple, Union

import mujoco
from mujoco import _batch
import numpy as np
from numpy import typing as npt

# MjData fields stored in the state rows, by state component
_STATE_FIELDS = {
    'time': mujoco.mjtState.mjSTATE_TIME,
    'qpos': mujoco.mjtState.mjSTATE_QPOS,
    'qvel': mujoco.mjtState.mjSTATE_QVEL,
    'act': mujoco.mjtState.mjSTATE_ACT,
    'history': mujoco.mjtState.mjSTATE_HISTORY,
    'qacc_warmstart': mujoco.mjtState.mjSTATE_WARMSTART,
    'ctrl': mujoco.mjtState.mjSTATE_CTRL,
    'qfrc_applied': mujoco.mjtState.mjSTATE_QFRC_APPLIED,
    'xfrc_applied': mujoco.mjtState.mjSTATE_XFRC_APPLIED,
    'eq_active': mujoco.mjtState.mjSTATE_EQ_ACTIVE,
    'mocap_pos': mujoco.mjtState.mjSTATE_MOCAP_POS,
    'mocap_quat': mujoco.mjtState.mjSTATE_MOCAP_QUAT,
    'userdata': mujoco.mjtState.mjSTATE_USERDATA,
    'plugin_state': mujoco.mjtState.mjSTATE_PLUGIN,
    'plastic': mujoco.mjtState.mjSTATE_PLASTIC,
}

# MjData named-view attributes computed from several fields, not slices of one:
# never probed, since the template MjData is not forwarded (uninitialized)
_COMPUTED_VIEWS = frozenset({('actuator', 'moment')})

Ids = Optional[npt.ArrayLike]
RecordItem = Union[str, int, mujoco.mjtState]


class SimulationError(RuntimeError):
  """MuJoCo raised an error in some simulations of a call.

  Those simulations kept the state they had before the call, and the others ran
  to completion. ids lists them; Batch.error(i) has each message.
  """

  def __init__(
      self, ids: np.ndarray, message: str, ncall: Optional[int] = None
  ):
    self.ids = ids
    shown = ', '.join(str(i) for i in ids[:5]) + (
        ', ...' if len(ids) > 5 else ''
    )
    of = '' if ncall is None else f' of {ncall}'
    super().__init__(
        f'MuJoCo raised in {len(ids)}{of} simulations ({shown}); simulation'
        f' {ids[0]}: {message}'
    )


class Batch:
  """nsim simulations of a copy of model, on nthread threads (0: all CPUs).

  With persistent=True, the batch keeps one MjData per simulation.

  A Batch must not be used from several Python threads at once: its calls are
  serialized, but each reports its failures from status and error(), which the
  next call overwrites.
  """

  def __init__(
      self,
      model: mujoco.MjModel,
      nsim: int,
      nthread: int = 0,
      persistent: bool = False,
  ):
    self._b = _batch.Batch(model, nsim, nthread, persistent)
    self.model = model
    # caller models, kept alive while the batch runs on them
    self._models: Union[None, mujoco.MjModel, list[mujoco.MjModel]] = None
    self._nkey = model.nkey  # keyframes reset accepts: the most of any model's
    # by name, so each is registered once
    self._bound: Dict[str, np.ndarray] = {}
    self._expanded: Dict[str, np.ndarray] = {}
    self.state = self._b.state()
    self.warning = self._b.warning()
    self.status = self._b.status()
    self.status.flags.writeable = False

  @functools.cached_property
  def _data(self) -> mujoco.MjData:
    """An MjData of the model, for field shapes and dtypes."""
    # made on first use: it costs more than a short-lived batch itself
    return mujoco.MjData(self.model)

  def __getattr__(self, name: str) -> np.ndarray:
    # consulted only for names Batch does not have: MjData fields, as bind()
    if name in DATA_FIELDS:
      return self.bind(name)
    hint = ''
    if not name.startswith('__'):
      hint = _hint(name, DATA_FIELDS, model_hint=True)
    raise AttributeError(f"'Batch' object has no attribute {name!r}{hint}")

  def __setattr__(self, name: str, value: Any) -> None:
    if name in DATA_FIELDS:  # write every simulation's value, as with MjData
      if name not in _STATE_FIELDS:
        raise _output_error(name)
      self.bind(name)[...] = value
    else:
      object.__setattr__(self, name, value)

  @property
  def nsim(self) -> int:
    return self._b.nsim

  @property
  def nthread(self) -> int:
    return self._b.nthread

  @property
  def nstate(self) -> int:
    return self._b.nstate

  @property
  def persistent(self) -> bool:
    return self._b.persistent

  def error(self, sim: int) -> str:
    """Message of the error simulation sim raised in its last call, or ''."""
    return self._b.error(sim)

  def bind(self, name: str) -> np.ndarray:
    """Return a live (nsim, ...) array over an MjData field.

    State fields are views of the rows, which hold mjtNum (float64 in the
    standard build), so eq_active binds as mjtNum rather than as MjData's uint8.
    Other fields are outputs, copied out after every call, and read-only.

    Args:
      name: the MjData field, one of DATA_FIELDS.

    Returns:
      The (nsim, ...) array, cached so that each field is registered once.
    """
    if name in _STATE_FIELDS:
      offset, size = self._component(_STATE_FIELDS[name])
      view = self.state[:, offset : offset + size]
      if name == 'time':
        return view[:, 0]
      return view.reshape(self.nsim, *getattr(self._data, name).shape)
    if name in self._bound:
      return self._bound[name]
    template = _template(self._data, name)
    raw = self._b.output(name)
    if raw is None or template is None:
      raise _not_bindable(self._data, name)
    out = raw.view(template.dtype).reshape(self.nsim, *template.shape)
    # an output: a write would be lost at the next call
    out.flags.writeable = False
    self._bound[name] = out
    return out

  def actuator(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'actuator', key)

  def body(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'body', key)

  def camera(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'camera', key)

  def equality(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'equality', key, data=False)

  def geom(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'geom', key)

  def joint(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'joint', key)

  def keyframe(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'keyframe', key, data=False)

  def light(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'light', key)

  def material(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'material', key, data=False)

  def numeric(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'numeric', key, data=False)

  def pair(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'pair', key, data=False)

  def sensor(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'sensor', key)

  def site(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'site', key)

  def tendon(self, key: Union[int, str]) -> '_Named':
    return _Named(self, 'tendon', key)

  def expand(self, name: str) -> np.ndarray:
    """Give a model field per-simulation values; return them, (nsim, ...).

    Array fields are named as in MjModel, MjOption fields as 'opt.<name>'.
    bvh_aabb holds the bodies' bounding volumes, which precede the meshes'.
    After writing a field that mj_setConst reads, one of
    mujoco.MjModel._setconst_inputs or an MjOption field, call set_const() so
    that the constants derived from it follow.

    Args:
      name: the MjModel field, one of MODEL_FIELDS.

    Returns:
      The live (nsim, ...) array of per-simulation values.
    """
    if name in self._expanded:
      return self._expanded[name]
    if name.startswith('opt.'):
      template = np.asarray(getattr(self.model.opt, name[4:], None))
    else:
      template = getattr(self.model, name, None)
    raw = self._b.expand(name)
    if raw is None or not isinstance(template, np.ndarray):
      raise _not_expandable(self.model, name)
    dtype = template.dtype
    if name.startswith('opt.'):
      # mjOption fields are stored as mjtNum or int
      dtype = mujoco.MJTNUM_DTYPE if dtype.kind == 'f' else np.int32
    out = _per_simulation(raw.view(dtype), template)
    self._expanded[name] = out
    return out

  def expanded(self, name: str) -> Optional[np.ndarray]:
    """Return a model field's read-only per-simulation values, or None.

    Returns the per-simulation values of a field that expand() or set_const()
    expanded, without expanding it if it is still shared.

    Args:
      name: the MjModel field, one of MODEL_FIELDS.

    Returns:
      The read-only (nsim, ...) array, or None if the field is not expanded.
    """
    if name not in MODEL_FIELDS:
      raise _not_expandable(self.model, name)
    raw = self._b.expanded(name)
    if raw is None:
      return None
    if name.startswith('opt.'):
      template = np.asarray(getattr(self.model.opt, name[4:]))
      dtype = mujoco.MJTNUM_DTYPE if template.dtype.kind == 'f' else np.int32
    else:
      template = getattr(self.model, name)
      dtype = template.dtype
    return _per_simulation(raw.view(dtype), template)

  def step(
      self,
      ids: Ids = None,
      nstep: int = 1,
      stage: int = mujoco.mjtStage.mjSTAGE_NONE.value,
  ) -> None:
    """Advance each simulation by nstep calls to mj_step, then compute stages.

    As with mj_step, derived fields describe the state before the last substep.
    stage computes the forward stages up to it at the new state: mjSTAGE_VEL
    makes position- and velocity-dependent fields and sensors current, as
    mj_step1 does; mjSTAGE_ACC runs mj_forward, whose acceleration-dependent
    values use controls that will not be applied. A persistent batch continues
    its next step from those stages, so that mjSTAGE_VEL costs nothing extra.

    Args:
      ids: the simulations, or None for all.
      nstep: number of substeps.
      stage: mjtStage, the stages to compute at the new state.
    """
    ids = _ids(ids, self.nsim)
    self._check(ids, self._b.step(ids, nstep, int(stage)))

  def forward(
      self, ids: Ids = None, stage: int = mujoco.mjtStage.mjSTAGE_ACC.value
  ) -> None:
    """Compute the forward stages up to stage, by default mj_forward."""
    ids = _ids(ids, self.nsim)
    self._check(ids, self._b.forward(ids, int(stage)))

  def reset(
      self,
      ids: Ids = None,
      keyframe: npt.ArrayLike = -1,
      stage: int = mujoco.mjtStage.mjSTAGE_ACC.value,
  ) -> None:
    """Reset to the model's defaults or to keyframes, then compute stages.

    Writes to the reset simulations' state made before the call are discarded.
    By default the reset runs mj_forward; stage=mjSTAGE_NONE skips it, for a
    reset that is followed by writes to the state and a forward().

    Args:
      ids: the simulations, or None for all.
      keyframe: a keyframe for every simulation of the call, or one each, (n,);
        -1 for the model's defaults.
      stage: mjtStage, the stages to compute after the reset.
    """
    ids = _ids(ids, self.nsim)
    n = self.nsim if ids is None else len(ids)
    # with caller models, the C call checks each simulation's keyframes exactly
    key = _int32(keyframe, (n,), -1, self._nkey, 'keyframe')
    if (key == -1).all():
      key = None
    self._check(ids, self._b.reset(ids, key, int(stage)))

  def get_model(
      self, sim: int, model: Optional[mujoco.MjModel] = None
  ) -> mujoco.MjModel:
    """Simulation sim's model: the batch's, with its per-simulation fields.

    With caller models set (set_models), a copy of simulation sim's model.

    Args:
      sim: the simulation.
      model: an MjModel with the sizes of simulation sim's model to write into,
        or None for a copy of that model. The batch shares that model's assets
        (meshes, textures and so on) and writes everything else.

    Returns:
      The model, for rendering or for any mj_ function.
    """
    if model is None:
      model = copy.copy(self._model_of(sim))
    self._b.get_model(sim, model)
    return model

  def set_models(
      self,
      models: Union[None, mujoco.MjModel, Sequence[mujoco.MjModel]],
  ) -> None:
    """Run simulation i on models[i] instead of the batch's model.

    Each model must give MjData the layout of the batch's model: the same sizes
    of everything MjData holds, as models compiled from one spec with different
    parameters, or with meshes of other shapes, do. The batch keeps them alive
    and reads them; set them again after editing one. Caller models replace
    per-simulation fields: expand() and set_const() raise while they are set,
    and set_models raises if fields are expanded. Named views read model
    attributes from each simulation's model.

    Args:
      models: an MjModel per simulation, one MjModel for all of them, or None
        for the batch's model.
    """
    if models is not None and not isinstance(models, mujoco.MjModel):
      models = list(models)
    self._b.set_models(models)
    if models is None:
      self._nkey = self.model.nkey
    elif isinstance(models, mujoco.MjModel):
      self._nkey = models.nkey
    else:
      self._nkey = max(m.nkey for m in models)
    self._models = models

  @property
  def models(self) -> Union[None, mujoco.MjModel, Tuple[mujoco.MjModel, ...]]:
    """The caller models set with set_models, or None for the batch's model.

    An MjModel if one was set for every simulation, else a tuple with
    simulation i's model at index i.
    """
    if self._models is None or isinstance(self._models, mujoco.MjModel):
      return self._models
    return tuple(self._models)

  def _model_of(self, sim: int) -> mujoco.MjModel:
    """The model simulation sim runs on, but for its per-simulation fields."""
    if self._models is None:
      return self.model
    if isinstance(self._models, mujoco.MjModel):
      return self._models
    return self._models[sim]

  def get_data(
      self, sim: int, data: Optional[mujoco.MjData] = None
  ) -> mujoco.MjData:
    """Simulation sim's state, and in a persistent batch its whole MjData.

    Derived fields are those of the simulation's last call in a persistent
    batch; otherwise run mj_forward with get_model(sim) to compute them.

    Args:
      sim: the simulation.
      data: an MjData with the batch's sizes to write into, or None for a new
        one.

    Returns:
      The data.
    """
    if data is None:
      data = mujoco.MjData(self.model)
    self._b.get_data(sim, data)
    return data

  def set_const(self, ids: Ids = None) -> None:
    """Run mj_setConst per simulation; derived fields become per-simulation."""
    ids = _ids(ids, self.nsim)
    self._check(ids, self._b.set_const(ids))

  def rollout(
      self,
      nstep: int,
      control: Optional[npt.ArrayLike] = None,
      control_spec: int = mujoco.mjtState.mjSTATE_CTRL.value,
      record: Sequence[RecordItem] = ('qpos',),
      ids: Ids = None,
      out: Optional[Dict[RecordItem, np.ndarray]] = None,
  ) -> Dict[RecordItem, np.ndarray]:
    """Advance each simulation by nstep substeps, recording after each.

    As mujoco.rollout: before substep k, the state components in control_spec
    are set from control, (n, nstep, ncontrol) for the n simulations of the
    call, or the inputs are left as they are when control is None. A simulation
    whose warning counters rise stops stepping, and its remaining records repeat
    its last ones.

    Args:
      nstep: number of substeps.
      control: (n, nstep, mj_stateSize(model, control_spec)) or None.
      control_spec: mjtState bits within mjSTATE_USER.
      record: MjData field names, or mjtState specs for state vectors. Fields
        are recorded with MjData's types, so eq_active is uint8 here though
        bind() gives it as mjtNum.
      ids: the simulations, or None for all.
      out: arrays to record into, by record item, of the shapes and dtypes this
        call returns; the items not in it get new arrays.

    Returns:
      A dict from each record item to its (n, nstep, ...) array.
    """
    ids = _ids(ids, self.nsim)
    n = self.nsim if ids is None else len(ids)
    if control is not None:
      ncontrol = mujoco.mj_stateSize(self.model, int(control_spec))
      control = np.ascontiguousarray(
          np.broadcast_to(control, (n, nstep, ncontrol)),
          dtype=mujoco.MJTNUM_DTYPE,
      )
    given, out, records = out or {}, {}, []
    for item in record:
      if isinstance(item, str):
        template = _template(self._data, item)
        if template is None:
          raise _not_bindable(self._data, item)
        shape, dtype = (n, nstep, *template.shape), template.dtype
      else:
        shape = (n, nstep, mujoco.mj_stateSize(self.model, int(item)))
        dtype = np.dtype(mujoco.MJTNUM_DTYPE)
      buf = given.get(item)
      if buf is None:
        buf = np.empty(shape, dtype)
      elif (
          not isinstance(buf, np.ndarray)
          or buf.shape != shape
          or buf.dtype != dtype
          or not buf.flags.c_contiguous
          or not buf.flags.writeable
      ):
        raise ValueError(
            f'out[{item!r}] must be a writable C-contiguous {dtype} array of'
            f' shape {shape}'
        )
      if isinstance(item, str):
        records.append((item, 0, buf))
      else:
        records.append((None, int(item), buf))
      out[item] = buf
    self._check(
        ids,
        self._b.rollout(ids, nstep, control, int(control_spec), records),
    )
    return out

  def jac(
      self, body: npt.ArrayLike, point: npt.ArrayLike, ids: Ids = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Translational and rotational Jacobians of a world point on a body.

    Runs mj_kinematics and mj_comPos per simulation, from its state, then
    mj_jac; the state and bound fields are untouched. In a persistent batch the
    simulation's MjData keeps these kinematics, as get_data() shows, and its
    next step starts afresh rather than continuing.

    Args:
      body: body id, or one per simulation of the call, (n,).
      point: world-frame points, (3,) or (n, 3) for the n simulations of the
        call.
      ids: the simulations, or None for all.

    Returns:
      jacp, jacr: (n, 3, nv) each.
    """
    ids = _ids(ids, self.nsim)
    n = self.nsim if ids is None else len(ids)
    num = mujoco.MJTNUM_DTYPE
    body = _int32(body, (n,), 0, self.model.nbody, 'body')
    point = np.ascontiguousarray(np.broadcast_to(point, (n, 3)), num)
    jacp = np.zeros((n, 3, self.model.nv), num)
    jacr = np.zeros((n, 3, self.model.nv), num)
    self._check(ids, self._b.jac(ids, body, point, jacp, jacr))
    return jacp, jacr

  def ray(
      self,
      pnt: npt.ArrayLike,
      vec: npt.ArrayLike,
      geomgroup: Optional[npt.ArrayLike] = None,
      flg_static: bool = True,
      bodyexclude: npt.ArrayLike = -1,
      ids: Ids = None,
      normal: bool = False,
  ) -> Tuple[np.ndarray, ...]:
    """Intersect rays with each simulation's geoms, as mj_ray.

    Runs mj_kinematics (and mj_flex) per simulation, from its state; the state
    and bound fields are untouched. In a persistent batch the simulation's
    MjData keeps these kinematics, as get_data() shows, and its next step starts
    afresh rather than continuing.

    Args:
      pnt: ray origins, broadcast to (n, nray, 3) for the n simulations of the
        call: (3,) is one ray, (nray, 3) the same rays in every simulation.
      vec: ray directions, broadcast likewise.
      geomgroup: geom groups to include, (mjNGROUP,), or None for all.
      flg_static: whether to include static geoms.
      bodyexclude: body whose geoms a ray ignores, or -1; broadcast to (n,
        nray).
      ids: the simulations, or None for all.
      normal: whether to return the surface normal at each hit.

    Returns:
      dist, geomid: (n, nray) each; dist is -1 and geomid -1 for a miss. With
      normal, also the normals, (n, nray, 3), zero for a miss.
    """
    ids = _ids(ids, self.nsim)
    n = self.nsim if ids is None else len(ids)
    num = mujoco.MJTNUM_DTYPE
    pnt = np.atleast_2d(np.asarray(pnt, num))
    vec = np.atleast_2d(np.asarray(vec, num))
    nray = np.broadcast_shapes(pnt.shape[:-1], vec.shape[:-1])[-1]
    pnt = np.ascontiguousarray(np.broadcast_to(pnt, (n, nray, 3)))
    vec = np.ascontiguousarray(np.broadcast_to(vec, (n, nray, 3)))
    exclude = _int32(
        bodyexclude, (n, nray), -1, self.model.nbody, 'bodyexclude'
    )
    if geomgroup is not None:
      geomgroup = np.ascontiguousarray(geomgroup, dtype=np.uint8)
    dist = np.zeros((n, nray), num)
    geomid = np.zeros((n, nray), np.int32)
    normals = np.zeros((n, nray, 3), num) if normal else None
    self._check(
        ids,
        self._b.ray(
            ids, pnt, vec, geomgroup, flg_static, exclude, dist, geomid, normals
        ),
    )
    if normals is None:
      return dist, geomid
    return dist, geomid, normals

  def _component(self, component: Any) -> Tuple[int, int]:
    return self._b.state_component(int(component))

  def _check(self, ids: Optional[np.ndarray], nfail: int) -> None:
    if not nfail:
      return
    sel = np.arange(self.nsim) if ids is None else ids
    failed = sel[self.status[sel] != 0]
    raise SimulationError(failed, self.error(int(failed[0])), len(sel))


class _Named:
  """One object's slices of per-simulation fields, as MjData's and MjModel's.

  batch.joint('hinge').qpos is the part of batch.bind('qpos') that
  data.joint('hinge').qpos is of data.qpos, and batch.body('pole').mass the part
  of batch.expand('body_mass') that model.body('pole').mass is of
  model.body_mass. Data attributes bind their field. Model attributes read
  without expanding: the per-simulation values of a field that expand() or
  set_const() expanded, or else the model's value for every simulation,
  read-only unless expanded by expand(). Assignment expands the field and writes
  every simulation's value. Structural fields (addresses, counts, ids and types)
  are the same in every simulation and cannot be assigned. Equalities, pairs,
  materials, numerics and keyframes have only MjModel attributes.
  """

  # (kind, attribute, model or data) -> the field that holds it; shared by all
  # batches, since which field holds an attribute doesn't depend on the model
  _fields: Dict[Tuple[str, str, bool], str] = {}

  _batch: 'Batch'
  _kind: str
  _data_view: Any  # None for a kind which MjData doesn't have
  _model_view: Any
  id: int
  name: str

  def __init__(
      self, batch: Batch, kind: str, key: Union[int, str], data: bool = True
  ):
    model_view = getattr(batch.model, kind)(key)
    data_view = None
    if data:
      data_view = getattr(batch._data, kind)(key)  # pylint: disable=protected-access
    object.__setattr__(self, '_batch', batch)
    object.__setattr__(self, '_kind', kind)
    object.__setattr__(self, '_data_view', data_view)
    object.__setattr__(self, '_model_view', model_view)
    object.__setattr__(self, 'id', model_view.id)
    object.__setattr__(self, 'name', model_view.name)

  def __getattr__(self, attr: str) -> np.ndarray:
    return self._slice(attr, write=False)

  def __setattr__(self, attr: str, value: Any) -> None:
    """Write value to every simulation's slice: batch.joint('j').qpos = 0.5."""
    if attr in ('id', 'name') or attr.startswith('_'):
      raise AttributeError(f'{attr} is read-only')
    self._slice(attr, write=True)[...] = value

  def _slice(self, attr: str, write: bool) -> np.ndarray:
    """The (nsim, ...) slice of attribute attr, writable if write."""
    batch = self._batch
    computed = (self._kind, attr) in _COMPUTED_VIEWS
    data_view = None if computed else self._data_view
    sources = (
        (False, data_view, batch._data, DATA_FIELDS),  # pylint: disable=protected-access
        (True, self._model_view, batch.model, MODEL_FIELDS),
    )
    unheld = computed  # an attribute of a view, but in no field of the batch
    for is_model, view, obj, names in sources:
      part = getattr(view, attr, None) if view is not None else None
      if not isinstance(part, np.ndarray):
        continue
      if not part.size:  # MjData and MjModel views are empty here too
        out = np.empty((batch.nsim, *part.shape), part.dtype)
        out.flags.writeable = False
        return out
      key = (self._kind, attr, is_model)
      if key not in self._fields:
        field = _containing(obj, names, part)
        if field is None:
          unheld = True
          continue
        self._fields[key] = field
      field = self._fields[key]
      if not is_model:
        if write and field not in _STATE_FIELDS:
          raise _output_error(f'{self._kind} {attr}')
        rows = batch.bind(field)
      elif batch._models is not None:  # pylint: disable=protected-access
        if write:
          raise ValueError(
              f'{field} comes from the caller models (set_models): edit them'
              ' and call set_models again'
          )
        # pylint: disable-next=protected-access
        models = [batch._model_of(i) for i in range(batch.nsim)]
        out = np.stack(
            [getattr(getattr(m, self._kind)(self.id), attr) for m in models]
        )
        out.flags.writeable = False
        return out
      elif field in batch._expanded:  # pylint: disable=protected-access
        rows = batch.expand(field)
      elif _structural(field, part):
        if write:
          raise ValueError(
              f'{field} is structural: it cannot differ between simulations'
          )
        out = np.broadcast_to(part, (batch.nsim, *part.shape))
        return out
      elif write:
        rows = batch.expand(field)
      else:
        # unexpanded read: set_const's per-simulation values, or the model's
        rows = batch.expanded(field)
        if rows is None:
          return np.broadcast_to(part, (batch.nsim, *part.shape))
      full = getattr(obj, field)
      start = (part.ctypes.data - full.ctypes.data) // full.itemsize
      flat = rows.reshape(batch.nsim, -1)
      return flat[:, start : start + part.size].reshape(batch.nsim, *part.shape)
    if unheld:
      raise AttributeError(
          f'{self._kind} {attr} is not a slice of a field that the batch binds'
          ' or expands'
      )
    names = dir(self._model_view)
    if self._data_view is not None:
      names += dir(self._data_view)
    raise AttributeError(
        f'{attr} is not a field of a {self._kind}' + _did_you_mean(attr, names)
    )


def _per_simulation(values: np.ndarray, template: np.ndarray) -> np.ndarray:
  """(nsim, n) values of a model field as (nsim, *template.shape).

  bvh_aabb holds only the bodies' hierarchies, a prefix of the model's rows,
  possibly none of them, so the number of rows is taken from values.

  Args:
    values: the per-simulation values, one row of elements per simulation.
    template: the model's array of the field.

  Returns:
    values, reshaped.
  """
  nsim, n = values.shape
  if n == template.size:
    return values.reshape(nsim, *template.shape)
  row = int(np.prod(template.shape[1:]))
  return values.reshape(nsim, n // row, *template.shape[1:])


def _structural(field: str, part: np.ndarray) -> bool:
  """Whether model field holds addresses, counts, ids or types: structure."""
  return (
      part.dtype.kind in 'iu'
      and field.endswith(('adr', 'num', 'id', 'type'))
      and not field.endswith('contype')
  )


def _containing(
    obj: Any, names: FrozenSet[str], part: np.ndarray
) -> Optional[str]:
  """The field of obj among names whose memory holds the array part."""
  for name in names:
    full = getattr(obj, name, None)
    if not isinstance(full, np.ndarray) or full.dtype != part.dtype:
      continue
    offset = part.ctypes.data - full.ctypes.data
    if 0 <= offset and offset + part.nbytes <= full.nbytes and part.size:
      return name
  return None


def _ids(ids: Ids, nsim: int) -> Optional[np.ndarray]:
  """Validated int32 ids from integer ids or a boolean mask; None for all.

  Shape, dtype, range, order and uniqueness are checked before narrowing to
  int32, so that no id is truncated or wrapped into a valid one.

  Args:
    ids: sorted unique integer ids, a boolean mask of shape (nsim,), or None.
    nsim: the number of simulations.

  Returns:
    The int32 ids, or None for all simulations.
  """
  if ids is None:
    return None
  ids = np.asarray(ids)
  if ids.dtype == bool:
    if ids.shape != (nsim,):
      raise ValueError(f'a boolean mask must have shape ({nsim},)')
    if ids.all():
      return None
    return np.flatnonzero(ids).astype(np.int32)
  if ids.ndim != 1:
    raise ValueError('ids must be one-dimensional')
  if ids.size == 0:
    return np.zeros(0, np.int32)
  if not np.issubdtype(ids.dtype, np.integer):
    raise ValueError(f'ids must be integers, got {ids.dtype}')
  # compared, not subtracted, so that unsigned ids cannot wrap around
  if ids.size > 1 and not (ids[1:] > ids[:-1]).all():
    k = np.flatnonzero(ids[:-1] >= ids[1:])[0]
    raise ValueError(
        f'ids must be sorted and unique, got {ids[k]} before {ids[k + 1]}'
    )
  # sorted, so only the first and last can be out of range
  if ids[0] < 0 or ids[-1] >= nsim:
    bad = ids[0] if ids[0] < 0 else ids[-1]
    raise ValueError(f'ids must be in [0, {nsim}), got {bad}')
  if ids.dtype == np.int32 and ids.flags.c_contiguous:
    return ids
  return np.ascontiguousarray(ids, dtype=np.int32)


def _int32(
    x: npt.ArrayLike, shape: Tuple[int, ...], lo: int, hi: int, name: str
) -> np.ndarray:
  """Integers x in [lo, hi), broadcast to shape, checked, then as int32."""
  x = np.asarray(x)
  if not np.issubdtype(x.dtype, np.integer):
    raise ValueError(f'{name} must be integers, got {x.dtype}')
  bad = x[(x < lo) | (x >= hi)]
  if bad.size:
    raise ValueError(f'{name} must be in [{lo}, {hi}), got {bad.flat[0]}')
  return np.ascontiguousarray(np.broadcast_to(x, shape), np.int32)


def _did_you_mean(name: str, names: Iterable[str]) -> str:
  """' (did you mean ...?)' listing close matches of name among names, or ''."""
  close = difflib.get_close_matches(name, list(names), n=3)
  return f" (did you mean {', '.join(map(repr, close))}?)" if close else ''


def _hint(name: str, names: Iterable[str], model_hint: bool = False) -> str:
  """Where name belongs if it is a model or option field, else close matches."""
  if model_hint and name in MODEL_FIELDS:
    return f'; {name} is an MjModel field: use expand({name!r})'
  if 'opt.' + name in MODEL_FIELDS:
    return f"; {name} is an MjOption field: use expand('opt.{name}')"
  return _did_you_mean(name, names)


def _template(data: mujoco.MjData, name: str) -> Optional[np.ndarray]:
  """MjData field name as an array, an int scalar as a 0-d int32; else None."""
  value = getattr(data, name, None)
  if isinstance(value, np.ndarray):
    return value
  if isinstance(value, int) and not isinstance(value, bool):
    return np.zeros((), np.int32)
  return None


def _not_bindable(data: mujoco.MjData, name: str) -> ValueError:
  """The error for a name that bind() does not accept; data is a template."""
  if name in MODEL_FIELDS:
    return ValueError(f'{name} is an MjModel field: use expand({name!r})')
  if _template(data, name) is not None and not isinstance(
      getattr(data, name), np.ndarray
  ):
    counts = sorted(n for n in DATA_FIELDS if isinstance(getattr(data, n), int))
    return ValueError(
        f'{name} is not one of the MjData scalars bind() accepts:'
        f' {", ".join(counts)}'
    )
  if isinstance(getattr(data, name, None), np.ndarray):
    # arena arrays: efc_*, island arrays and the like
    return ValueError(
        f'{name} is in the arena, whose size changes from step to step, and'
        ' cannot be bound'
    )
  return ValueError(
      f'{name} is not an MjData array field' + _hint(name, DATA_FIELDS)
  )


def _not_expandable(model: mujoco.MjModel, name: str) -> ValueError:
  """The error for a name that expand() does not accept."""
  if name in DATA_FIELDS:
    return ValueError(f'{name} is an MjData field: use bind({name!r})')
  if isinstance(getattr(model, name, None), np.ndarray):
    return ValueError(
        f'{name} is asset data, shared by all simulations, and cannot be'
        ' expanded'
    )
  return ValueError(
      f'{name} is not an expandable MjModel field' + _hint(name, MODEL_FIELDS)
  )


def _output_error(name: str) -> ValueError:
  """The error for a write to an output, which every call overwrites."""
  return ValueError(
      f'{name} is computed by every call and cannot be written; write state'
      ' fields (qpos, ctrl, ...) or expand model fields'
  )


def _field_names() -> Tuple[FrozenSet[str], FrozenSet[str]]:
  """The names bind() and expand() accept, as the batch's own tables decide."""
  model = mujoco.MjModel.from_xml_string('<mujoco/>')
  data = mujoco.MjData(model)
  probe = _batch.Batch(model, 1, 1)
  public = lambda obj: [n for n in dir(obj) if not n.startswith('_')]
  data_fields = set(_STATE_FIELDS) | {
      n
      for n in public(data)
      if n not in _STATE_FIELDS
      and _template(data, n) is not None
      and probe.output(n) is not None
  }
  candidates = [
      n
      for n in public(model)
      if isinstance(getattr(model, n, None), np.ndarray)
  ] + [
      'opt.' + n
      for n in public(model.opt)
      if not callable(getattr(model.opt, n))
  ]
  model_fields = {n for n in candidates if probe.expand(n) is not None}
  return frozenset(data_fields), frozenset(model_fields)


# MjData fields that bind() accepts, and MjModel fields that expand() accepts
DATA_FIELDS, MODEL_FIELDS = _field_names()
