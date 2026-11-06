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

"""Trajectory data containers for system identification."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib

from absl import logging
import mujoco
import mujoco.rollout as mj_rollout
from mujoco.sysid._src import timeseries
import numpy as np


@dataclasses.dataclass(frozen=True)
class SystemTrajectory:
  """Encapsulates a trajectory rolled out from a system.

  Attributes:
    model: MuJoCo model used to simulate the trajectory.
    control: A TimeSeries instance containing control signals.
    sensordata: A TimeSeries instance containing sensor data.
    initial_state: Initial state of the simulation. Shape (n_state,).
    state: Simulation states over time. Shape (n_steps, n_state). Optional for
      real robot trajectories.
  """

  model: mujoco.MjModel
  control: timeseries.TimeSeries
  sensordata: timeseries.TimeSeries
  initial_state: np.ndarray
  state: timeseries.TimeSeries | None

  def replace(self, **kwargs) -> SystemTrajectory:
    """Return a copy with the specified fields replaced."""
    return dataclasses.replace(self, **kwargs)

  def get_sensordata_slice(self, sensor: str = "joint_pos") -> np.ndarray:
    """Extract contiguous sensor columns by type.

    Args:
      sensor: One of ``"joint_pos"``, ``"joint_vel"``, or ``"joint_torque"``.

    Returns:
      2-D array of shape ``(n_steps, total_sensor_dim)``.
    """
    if sensor == "joint_pos":
      sensor_type = mujoco.mjtSensor.mjSENS_JOINTPOS
    elif sensor == "joint_vel":
      sensor_type = mujoco.mjtSensor.mjSENS_JOINTVEL
    elif sensor == "joint_torque":
      sensor_type = mujoco.mjtSensor.mjSENS_JOINTACTFRC
    else:
      raise ValueError(f"Unsupported sensor type: {sensor}")
    columns = []
    for i in range(self.model.nsensor):
      if self.model.sensor_type[i] == sensor_type:
        start = self.model.sensor_adr[i]
        columns.extend(range(start, start + self.model.sensor_dim[i]))
    return self.sensordata.data[:, columns]

  @property
  def sensordim(self) -> int:
    """Total number of scalar sensor outputs in the model."""
    return self.model.nsensordata

  def __len__(self) -> int:
    """Number of time steps in the trajectory."""
    return len(self.sensordata)

  def save_to_disk(self, path: pathlib.Path) -> None:
    """Save this trajectory to disk as a compressed NumPy archive."""
    save_dict = {
        "control_times": self.control.times,
        "control_data": self.control.data,
        "sensordata_times": self.sensordata.times,
        "sensordata_data": self.sensordata.data,
        "initial_state": self.initial_state,
    }
    if self.state is not None:
      save_dict["state_times"] = self.state.times
      save_dict["state_data"] = self.state.data
      if self.state.signal_mapping:
        save_dict["state_signal_mapping"] = np.asarray(
            timeseries.encode_signal_mapping(self.state.signal_mapping)
        )

    if self.control.signal_mapping:
      save_dict["control_signal_mapping"] = np.asarray(
          timeseries.encode_signal_mapping(self.control.signal_mapping)
      )

    if self.sensordata.signal_mapping:
      save_dict["sensordata_signal_mapping"] = np.asarray(
          timeseries.encode_signal_mapping(self.sensordata.signal_mapping)
      )

    np.savez(path, **save_dict)  # type: ignore

  @classmethod
  def load_from_disk(
      cls,
      path: pathlib.Path,
      model: mujoco.MjModel,
      allow_missing_sensors: bool = False,
  ) -> SystemTrajectory:
    """Load a trajectory from a compressed NumPy archive."""
    with np.load(path, allow_pickle=False) as npz:
      control_times = npz["control_times"]
      control_data = npz["control_data"]
      sensordata_times = npz["sensordata_times"]
      sensordata_data = npz["sensordata_data"]
      initial_state = npz["initial_state"]
      state_times = npz.get("state_times", None)
      state_data = npz.get("state_data", None)

      control_signal_mapping = None
      if "control_signal_mapping" in npz:
        control_signal_mapping = timeseries.decode_signal_mapping(
            str(npz["control_signal_mapping"])
        )

      sensordata_signal_mapping = None
      if "sensordata_signal_mapping" in npz:
        sensordata_signal_mapping = timeseries.decode_signal_mapping(
            str(npz["sensordata_signal_mapping"])
        )

      state_signal_mapping = None
      if "state_signal_mapping" in npz:
        state_signal_mapping = timeseries.decode_signal_mapping(
            str(npz["state_signal_mapping"])
        )

    predicted_rollout = cls(
        model=model,
        control=timeseries.TimeSeries(
            control_times, control_data, signal_mapping=control_signal_mapping
        ),
        sensordata=timeseries.TimeSeries(
            sensordata_times,
            sensordata_data,
            signal_mapping=sensordata_signal_mapping,
        ),
        initial_state=initial_state,
        state=timeseries.TimeSeries(
            state_times, state_data, signal_mapping=state_signal_mapping
        )
        if state_times is not None
        else None,
    )
    predicted_rollout.check_compatible(allow_missing_sensors)
    return predicted_rollout

  def check_compatible(self, allow_missing_sensors: bool = False) -> None:
    """Validate that data dimensions match the model.

    Checks sensor, control, state, and initial-state dimensions.

    Args:
      allow_missing_sensors: If True, a sensor dimension mismatch is logged as a
        warning instead of raising.
    """
    if self.sensordata.data.shape[1] != self.model.nsensordata:
      if not allow_missing_sensors:
        raise ValueError(
            f"Sensor data dimension {self.sensordata.data.shape[1]} does not"
            f" match model sensor dimension {self.model.nsensordata}"
        )
      else:
        logging.warning(
            "Sensor data dimension %d does not match model sensor dimension %d",
            self.sensordata.data.shape[1],
            self.model.nsensordata,
        )

    if self.control.data.shape[1] != self.model.nu:
      raise ValueError(
          f"Control data dimension {self.control.data.shape[1]} does not"
          f" match model control dimension {self.model.nu}"
      )

    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS.value
    state_size = mujoco.mj_stateSize(self.model, state_spec)
    if self.state is not None:
      if self.state.data.shape[1] != state_size:
        raise ValueError(
            f"State dimension {self.state.data.shape[1]} does not match "
            f"model state dimension {state_size}"
        )
    if self.initial_state.shape[0] != state_size:
      raise ValueError(
          f"Initial state dimension {self.initial_state.shape[0]} does not"
          f" match model state dimension {state_size}"
      )

  def split(self, chunk_size: int) -> list[SystemTrajectory]:
    """Split into consecutive non-overlapping chunks of *chunk_size* steps.

    Incomplete trailing steps are discarded.  Requires ``state`` to be set
    (needed to extract the initial state for each chunk).

    Args:
      chunk_size: Number of time steps per chunk.

    Returns:
      A list of SystemTrajectory chunks.
    """
    if self.state is None:
      raise ValueError("Cannot split rollout with missing state field.")
    steps = len(self.sensordata.times)
    n_complete_chunks = steps // chunk_size
    control_times = self.control.times
    control_data = self.control.data
    sensordata_times = self.sensordata.times
    sensordata_data = self.sensordata.data
    trajectories = []
    for i in range(n_complete_chunks):
      start_idx = i * chunk_size
      end_idx = start_idx + chunk_size
      initial_state = (
          self.initial_state
          if start_idx == 0
          else self.state.data[start_idx - 1]
      )
      control_times_chunk = control_times[start_idx:end_idx]
      control_data_chunk = control_data[start_idx:end_idx]
      sensordata_times_chunk = sensordata_times[start_idx:end_idx]
      sensordata_data_chunk = sensordata_data[start_idx:end_idx]
      trajectories.append(
          SystemTrajectory(
              model=self.model,
              control=timeseries.TimeSeries(
                  control_times_chunk, control_data_chunk
              ),
              sensordata=timeseries.TimeSeries(
                  sensordata_times_chunk, sensordata_data_chunk
              ),
              initial_state=initial_state,
              state=timeseries.TimeSeries(
                  times=self.state.times[start_idx:end_idx],
                  data=self.state.data[start_idx:end_idx],
                  signal_mapping=self.state.signal_mapping,
              ),
          )
      )
    return trajectories

  def render(
      self,
      height: int = 240,
      width: int = 320,
      camera: str | int = -1,
      fps: int = 30,
  ) -> list[np.ndarray]:
    """Render this trajectory to a list of RGB frames.

    Requires ``state`` to be set.  Delegates to
    :func:`~mujoco_sysid._src.plotting.render_rollout`.

    Args:
      height: Height of the rendered frames.
      width: Width of the rendered frames.
      camera: Camera index or name.
      fps: Frames per second.

    Returns:
      A list of RGB frames as numpy arrays.
    """
    if self.state is None:
      raise ValueError("Cannot render rollout with missing state field.")

    from mujoco.sysid._src.plotting import render_rollout

    # Adapt state to batch format (nbatch=1, nsteps, nstate)
    state_batch = self.state.data[np.newaxis, :, :]

    data = mujoco.MjData(self.model)

    return render_rollout(
        model=self.model,
        data=data,
        state=state_batch,
        framerate=fps,
        camera=camera,
        width=width,
        height=height,
    )

def _map_states(from_array, to_array, from_names, to_mapping, map_offset):
  i = 0
  for name in from_names:
    _, indices = to_mapping[name]
    width = indices.shape[0]
    to_array[indices - map_offset] = from_array[i:i+width]
    i += width
  return to_array

def create_initial_state(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
    act: np.ndarray | None = None,
    qpos_names: Sequence[str] | None = None,
    qvel_names: Sequence[str] | None = None,
    act_names: Sequence[str] | None = None,
) -> np.ndarray:
  """Build a ``mjSTATE_FULLPHYSICS`` initial-state vector from components.

  Args:
    model: MuJoCo model.
    qpos: Joint positions, shape ``(nq,)``.
    qvel: Joint velocities, shape ``(nv,)``.  Defaults to zero.
    act: Actuator activations, shape ``(na,)``.  Defaults to zero.
    qpos_names: Names to map elements of qpos to specific MuJoCo states.
      If None, assumes qpos is in MuJoCo's order.
    qvel_names: Names to map elements of qvel to specific MuJoCo states.
      If None, assumes qvel is in MuJoCo's order.
    act_names: Actuator names to map elements of act to specific MuJoCo
      actuators. If None, assumes act is in MuJoCo's order.

  Returns:
    Flat state vector suitable for ``mujoco.rollout``.
  """
  data = mujoco.MjData(model)

  if qpos_names is not None and len(qpos_names) != qpos.shape[0]:
    raise ValueError(
      f"Expected qpos to have shape {len(qpos_names)}, got {qpos.shape[0]}"
    )
  if qvel is None and qvel_names is not None:
    raise ValueError("Expected qvel to not be None when qvel_names is not None")
  if qvel_names is not None and qvel is not None and len(qvel_names) != qvel.shape[0]:
    raise ValueError(
        f"Expected qvel to have shape {len(qvel_names)}, got {qvel.shape[0]}"
    )
  if act_names is not None:
    if act is None:
      raise ValueError("Expected act to not be None when act_names is not None")
    if len(act_names) != act.shape[0]:
      raise ValueError(
          f"Expected act to have shape {len(act_names)}, got {act.shape[0]}"
      )

  if (qpos_names is not None
      or qvel_names is not None
      or act_names is not None):
    qpos_map, qvel_map, act_map, _ = timeseries.TimeSeries.compute_all_state_mappings(model)
    if qpos_names is not None:
      qpos = _map_states(qpos, np.copy(data.qpos), qpos_names, qpos_map, 0)
    if qvel_names is not None:
      indices_offset = data.qpos.shape[0]
      qvel = _map_states(
          qvel, np.copy(data.qvel), qvel_names, qvel_map, indices_offset
      )
    if act_names is not None:
      indices_offset = data.qpos.shape[0] + data.qvel.shape[0]
      act = _map_states(
          act, np.copy(data.act), act_names, act_map, indices_offset
      )

  if qpos.shape[0] != model.nq:
    raise ValueError(
        f"Expected qpos to have shape {model.nq}, got {qpos.shape[0]}."
    )
  data.qpos[:] = qpos
  if qvel is not None:
    if qvel.shape[0] != model.nv:
      raise ValueError(
          f"Expected qvel to have shape {model.nv}, got {qvel.shape[0]}."
      )
    data.qvel[:] = qvel
  if act is not None:
    if act.shape[0] != model.na:
      raise ValueError(
          f"Expected act to have shape {model.na}, got {act.shape[0]}."
      )
    data.act[:] = act

  initial_state = np.empty((
      mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value),
  ))
  mujoco.mj_getState(
      model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
  )
  return initial_state


class ModelSequences:
  """A model spec paired with one or more measured trajectory sequences.

  Groups a single ``MjSpec`` (the model to be identified) with the
  corresponding measured data so that the residual pipeline can iterate
  over all sequences for that model.

  Args:
    name: Identifier for this model group (used for file-naming on save).
    spec: MjSpec that will be recompiled with candidate parameters.
    sequence_name: Name(s) identifying each measured sequence.
    initial_state: Initial state(s) for each sequence.
    control: Measured control TimeSeries for each sequence.
    sensordata: Measured sensor TimeSeries for each sequence.
    allow_missing_sensors: Passed through to
      :meth:`SystemTrajectory.check_compatible`.
  """

  def __init__(
      self,
      name: str,
      spec: mujoco.MjSpec,
      sequence_name: str | Sequence[str],
      initial_state: np.ndarray | Sequence[np.ndarray],
      control: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
      sensordata: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
      allow_missing_sensors: bool = False,
  ):
    self.name = name
    self.spec = spec
    self.allow_missing_sensors = allow_missing_sensors

    self.gt_model = self.spec.compile()

    self.sequence_name: list[str] = (
        [sequence_name]
        if isinstance(sequence_name, str)
        else list(sequence_name)
    )
    self.initial_state: list[np.ndarray] = (
        [initial_state]
        if isinstance(initial_state, np.ndarray)
        else list(initial_state)
    )
    self.control: list[timeseries.TimeSeries] = (
        [control]
        if isinstance(control, timeseries.TimeSeries)
        else list(control)
    )
    self.sensordata: list[timeseries.TimeSeries] = (
        [sensordata]
        if isinstance(sensordata, timeseries.TimeSeries)
        else list(sensordata)
    )

    self.measured_rollout: list[SystemTrajectory] = []
    for initial_state_, control_, sensordata_ in zip(
        self.initial_state, self.control, self.sensordata, strict=True
    ):
      measured_rollout_ = SystemTrajectory(
          model=self.gt_model,
          control=control_,
          sensordata=sensordata_,
          initial_state=initial_state_,
          state=None,
      )
      measured_rollout_.check_compatible(
          allow_missing_sensors=allow_missing_sensors
      )
      self.measured_rollout.append(measured_rollout_)

  def __getitem__(self, key):
    return ModelSequences(
        self.name,
        self.spec,
        self.sequence_name[key],
        self.initial_state[key],
        self.control[key],
        self.sensordata[key],
        self.allow_missing_sensors,
    )


def timeseries2array(
    control_signal: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
) -> tuple[np.ndarray, np.ndarray]:
  """Convert control TimeSeries to stacked arrays, dropping the last step.

  Args:
    control_signal: Control TimeSeries or sequence of TimeSeries.

  Returns:
    ``(control_array, control_times)`` with the last time step removed.
  """
  if isinstance(control_signal, timeseries.TimeSeries):
    control = control_signal.data
    control_times = control_signal.times
  else:
    control = np.stack([ts.data for ts in control_signal], axis=0)
    control_times = np.stack([ts.times for ts in control_signal], axis=0)
  # The measured data has N sensor measurements and N controls,
  # where the first sensor measurement corresponds to the initial
  # condition. Thus we don't have ground truth for the N+1'th state
  # produced by the N'th control and so there is no point in
  # simulating it.
  if control.ndim == 3:
    control_applied_times = control_times[:, :-1]
    control_applied = control[:, :-1, :]
  else:
    control_applied_times = control_times[:-1]
    control_applied = control[:-1, :]
  return control_applied, control_applied_times


def sequence2array(
    initial_states: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
  """Stack a sequence of initial-state vectors into a single array.

  Args:
    initial_states: Single state array or sequence of state arrays.
  """
  if isinstance(initial_states, np.ndarray):
    return initial_states
  return np.stack(initial_states, axis=0)


def arrays2traj(
    models: mujoco.MjModel | Sequence[mujoco.MjModel],
    initial_states: np.ndarray | Sequence[np.ndarray],
    control: np.ndarray,
    control_times: np.ndarray,
    state: np.ndarray,
    sensordata: np.ndarray,
    signal_mapping: timeseries.SignalMappingType,
    state_mapping: timeseries.SignalMappingType,
    ctrl_mapping: timeseries.SignalMappingType,
) -> Sequence[SystemTrajectory]:
  """Convert raw rollout arrays into a list of SystemTrajectory objects.

  Args:
    models: Single model or sequence of models (one per batch element).
    initial_states: Initial state array(s).
    control: Control array, shape ``(nbatch, nsteps, nu)``.
    control_times: Control timestamps, shape ``(nbatch, nsteps)``.
    state: State array, shape ``(nbatch, nsteps, nstate)``.
    sensordata: Sensor data array, shape ``(nbatch, nsteps, nsensordata)``.
    signal_mapping: Signal mapping for sensor data.
    state_mapping: Signal mapping for state data.
    ctrl_mapping: Signal mapping for control data.
  """
  nbatch = state.shape[0]
  # TODO(kevin): When is np.tile/atleast_2d/etc necessary?
  # initial_states = np.tile(initial_states, (nbatch, 1))
  # control = np.tile(control, (nbatch, 1, 1))
  # control_times = np.tile(control_times, (nbatch, 1))
  initial_states = np.atleast_2d(initial_states)
  if control.ndim == 2:
    control = control[np.newaxis, :, :]
  control_times = np.atleast_2d(control_times)

  if isinstance(models, mujoco.MjModel):
    models_list = [models] * nbatch
  else:
    models_list = list(models)

  return [
      SystemTrajectory(
          model=models_list[i],
          control=timeseries.TimeSeries(
              control_times[i], control[i], signal_mapping=ctrl_mapping
          ),
          # NOTE(kevin): When using mjSTATE_FULLPHYSICS, the first element of
          # the state corresponds to the simulation time. The reason we do not
          # use control_times[i] is because sensordata times are shifted by
          # one time step.
          sensordata=timeseries.TimeSeries(
              state[i][:, 0], sensordata[i], signal_mapping
          ),
          initial_state=initial_states[i],
          state=timeseries.TimeSeries(
              times=state[i][:, 0], data=state[i], signal_mapping=state_mapping
          ),
      )
      for i in range(nbatch)
  ]


def sysid_rollout(
    models: mujoco.MjModel | Sequence[mujoco.MjModel],
    datas: mujoco.MjData | Sequence[mujoco.MjData],
    control_signal: Sequence[timeseries.TimeSeries] | timeseries.TimeSeries,
    initial_states: np.ndarray | Sequence[np.ndarray],
    rollout_signal_mapping: timeseries.SignalMappingType | None = None,
    rollout_state_mapping: timeseries.SignalMappingType | None = None,
    ctrl_mapping: timeseries.SignalMappingType | None = None,
) -> Sequence[SystemTrajectory]:
  """Rollout trajectories in parallel for the given models and controls.

  Args:
    models: MuJoCo model or sequence of models.
    datas: MuJoCo data or sequence of data.
    control_signal: Control signals as TimeSeries or sequence of TimeSeries.
    initial_states: Initial states of the simulation. Shape (n_state,) or
      (n_batch, n_state).
    rollout_signal_mapping: Optional signal mapping for sensordata.
    rollout_state_mapping: Optional signal mapping for state.
    ctrl_mapping: Optional signal mapping for controls.

  Returns:
    Sequence of SystemTrajectory instances containing the simulation results.
  """

  # if the user does not supply it, we create it.
  # Note that this will impact perf.
  if (
      not rollout_signal_mapping
      or not rollout_state_mapping
      or not ctrl_mapping
  ):
    if isinstance(models, mujoco.MjModel):
      model0 = models
    else:
      model0 = models[0]
    qpos_map, qvel_map, act_map, ctrl_mapping = (
        timeseries.TimeSeries.compute_all_state_mappings(model0)
    )
    rollout_state_mapping = qpos_map | qvel_map | act_map
    rollout_signal_mapping = timeseries.TimeSeries.compute_all_sensor_mapping(
        model0
    )

  control, control_times = timeseries2array(control_signal)
  initial_states = sequence2array(initial_states)
  state, sensordata = mj_rollout.rollout(models, datas, initial_states, control)
  assert isinstance(state, np.ndarray)
  assert isinstance(sensordata, np.ndarray)

  return arrays2traj(
      models,
      initial_states,
      control,
      control_times,
      state,
      sensordata,
      rollout_signal_mapping,
      rollout_state_mapping,
      ctrl_mapping,
  )
