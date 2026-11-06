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

"""Time series utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import json
import pathlib
from typing import Literal, TypeAlias

import mujoco
import numpy as np
import scipy.interpolate


class SignalType(Enum):
  MjSensor = 0
  CustomObs = 1
  MjStateQPos = 2
  MjStateQVel = 3
  MjStateAct = 4
  MjCtrl = 5


SignalMappingType: TypeAlias = dict[str, tuple[SignalType, np.ndarray]]


def encode_signal_mapping(signal_mapping: SignalMappingType) -> str:
  """Serialize a signal mapping to a JSON string.

  This avoids storing object arrays (which require ``allow_pickle=True`` to
  load) on disk. The mapping is plain data: enum values and integer indices.
  """
  return json.dumps({
      name: [sig_type.value, np.asarray(indices).astype(int).tolist()]
      for name, (sig_type, indices) in signal_mapping.items()
  })


def decode_signal_mapping(encoded: str) -> SignalMappingType:
  """Inverse of :func:`encode_signal_mapping`."""
  raw = json.loads(encoded)
  return {
      name: (SignalType(sig_type), np.asarray(indices, dtype=int))
      for name, (sig_type, indices) in raw.items()
  }

InterpolationMethod = Literal[
    "linear", "cubic", "quadratic", "quintic", "zero_order_hold", "zoh"
]


def _resolve_signals(
    model: mujoco.MjModel,
    names: Sequence[str | tuple[str, SignalType]],
    allowed_types: set[SignalType],
) -> SignalMappingType:
  """Resolves signal names to (canonical_name, type, indices) mappings.

  Each name can be a string or (name, SignalType) tuple for disambiguation.

  Args:
    model: MuJoCo model used to look up sensor metadata.
    names: Signal names or (name, type) tuples.
    allowed_types: Set of acceptable signal types.

  Returns:
    A dictionary mapping signal names to (type, indices) tuples.
  """
  result: SignalMappingType = {}
  idx = 0

  for item in names:
    name, hint = item if isinstance(item, tuple) else (item, None)
    resolved = _resolve_one(model, name, hint, allowed_types)

    if resolved is None:
      if hint is not None and hint not in allowed_types:
        raise ValueError(
            f"Signal '{name}' has type {hint.name} which is not allowed."
        )
      raise ValueError(
          f"Could not resolve signal '{item}' with allowed types"
          f" {[t.name for t in allowed_types]}."
      )

    canon_name, sig_type, width = resolved
    result[canon_name] = (sig_type, np.arange(idx, idx + width))
    idx += width

  return result


# Suffix conventions for state/control signals
_SUFFIXES = {
    SignalType.MjStateQPos: "_qpos",
    SignalType.MjStateQVel: "_qvel",
    SignalType.MjStateAct: "_act",
    SignalType.MjCtrl: "_ctrl",
}


def _strip_suffix(name: str, suffix: str) -> str:
  """Strip suffix from name if present."""
  return name[: -len(suffix)] if name.endswith(suffix) else name


def _resolve_one(
    model: mujoco.MjModel,
    name: str,
    hint: SignalType | None,
    allowed: set[SignalType],
) -> tuple[str, SignalType, int] | None:
  """Resolve a single signal name to (canonical_name, type, width)."""

  # 1. Sensor
  if _type_allowed(hint, SignalType.MjSensor, allowed):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR.value, name)
    if sid >= 0:
      return (name, SignalType.MjSensor, model.sensor_dim[sid])

  # 2. Control
  if _type_allowed(hint, SignalType.MjCtrl, allowed):
    base = _strip_suffix(name, "_ctrl")
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, base)
    if aid >= 0:
      return (base + "_ctrl", SignalType.MjCtrl, 1)

  # 3. State (qpos/qvel)
  for sig_type in (SignalType.MjStateQPos, SignalType.MjStateQVel):
    if _type_allowed(hint, sig_type, allowed):
      base = _strip_suffix(name, _SUFFIXES[sig_type])
      width = _joint_or_body_width(model, base, sig_type)
      if width > 0:
        return (base + _SUFFIXES[sig_type], sig_type, width)

  # 4. Actuator state (act)
  if _type_allowed(hint, SignalType.MjStateAct, allowed):
    base = _strip_suffix(name, "_act")
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, base)
    if aid >= 0 and model.actuator_actnum[aid] > 0:
      return (base + "_act", SignalType.MjStateAct, model.actuator_actnum[aid])

  return None


def _type_allowed(
    hint: SignalType | None, target: SignalType, allowed: set[SignalType]
) -> bool:
  """Check if target type is allowed given hint and allowed set."""
  return (hint is None or hint == target) and target in allowed


def _joint_or_body_width(
    model: mujoco.MjModel, name: str, sig_type: SignalType
) -> int:
  """Get state width for a joint or free body."""
  # Joint widths by type
  QPOS_WIDTHS = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
  QVEL_WIDTHS = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}

  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name)
  if jid >= 0:
    jtype = model.jnt_type[jid]
    widths = QPOS_WIDTHS if sig_type == SignalType.MjStateQPos else QVEL_WIDTHS
    return widths.get(jtype, 1)

  # Free body
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name)
  if bid >= 0 and model.body_dofnum[bid] == 6:
    return 7 if sig_type == SignalType.MjStateQPos else 6

  return 0


@dataclass(frozen=True)
class TimeSeries:
  """A utility class for working with time-series data.

  Attributes:
    times: 1D array of timestamps.
    data: Array of signal data. The first axis corresponds to time.
    signal_mapping: Dict of tuples that maps the signal type and its signal
      fields in data
  """

  times: np.ndarray
  data: np.ndarray
  signal_mapping: SignalMappingType | None = None

  @staticmethod
  def compute_all_sensor_mapping(model: mujoco.MjModel) -> SignalMappingType:
    """Computes mapping for all sensors in the model."""
    signal_mapping = {}
    for sensor_id in range(model.nsensor):
      addr = model.sensor_adr[sensor_id]
      dim = model.sensor_dim[sensor_id]
      name = mujoco.mj_id2name(
          model, mujoco.mjtObj.mjOBJ_SENSOR.value, sensor_id
      )
      indices = np.arange(addr, addr + dim)
      signal_mapping[name] = (SignalType.MjSensor, indices)
    return signal_mapping

  @staticmethod
  def compute_all_control_mapping(model: mujoco.MjModel) -> SignalMappingType:
    """Computes mapping for all controls (actuators) in the model."""
    ctrl_map: SignalMappingType = {}
    for act_id in range(model.nu):
      act_name = model.actuator(act_id).name
      ctrl_indices = np.arange(act_id, act_id + 1)
      ctrl_map[f"{act_name}_ctrl"] = (SignalType.MjCtrl, ctrl_indices)
    return ctrl_map

  @staticmethod
  def compute_all_state_mappings(
      model: mujoco.MjModel,
  ) -> tuple[
      SignalMappingType, SignalMappingType, SignalMappingType, SignalMappingType
  ]:
    """Computes mappings for all state components (qpos, qvel, act) + ctrl."""
    qpos_map: SignalMappingType = {}
    qvel_map: SignalMappingType = {}
    act_map: SignalMappingType = {}

    nq = model.nq
    nv = model.nv

    # Bodies with free joints, named by body rather than joint.
    for body_id in range(model.nbody):
      b = model.body(body_id)
      body_name = b.name
      dof_adr = model.body_dofadr[body_id]

      if dof_adr >= 0 and b.dofnum[0] == 6:
        # Use the body's first joint qposadr for qpos. Free joints have
        # 7 qpos elements but only 6 dofs, so dofadr and qposadr diverge
        # for subsequent entries.
        first_jnt = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[first_jnt]
        qpos_indices = np.arange(qpos_adr, qpos_adr + 7)
        qpos_map[f"{body_name}_qpos"] = (SignalType.MjStateQPos, qpos_indices)
        qvel_indices = np.arange(dof_adr + nq, dof_adr + nq + 6)
        qvel_map[f"{body_name}_qvel"] = (SignalType.MjStateQVel, qvel_indices)

    # Joints, excluding free joints which are handled above.
    for jnt_id in range(model.njnt):
      jnt_name = model.joint(jnt_id).name
      jnt_type = model.jnt_type[jnt_id]

      qpos_width = 1
      qvel_width = 1
      if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        qpos_width = 4
        qvel_width = 3
      elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        continue

      qpos_adr = model.jnt_qposadr[jnt_id]
      qpos_indices = np.arange(qpos_adr, qpos_adr + qpos_width)
      qpos_map[f"{jnt_name}_qpos"] = (SignalType.MjStateQPos, qpos_indices)
      dof_adr = model.jnt_dofadr[jnt_id]
      qvel_indices = np.arange(dof_adr + nq, dof_adr + nq + qvel_width)
      qvel_map[f"{jnt_name}_qvel"] = (SignalType.MjStateQVel, qvel_indices)

    # Actuators
    for act_id in range(model.nu):
      act_name = model.actuator(act_id).name
      start_index = model.actuator_actadr[act_id]
      num_vals = model.actuator_actnum[act_id]
      # if index is -1, the actuator is stateless.
      if start_index != -1:
        indices = np.arange(
            start_index + nq + nv, start_index + nq + nv + num_vals
        )
        act_map[f"{act_name}_act"] = (SignalType.MjStateAct, indices)

    ctrl_map = TimeSeries.compute_all_control_mapping(model)

    return qpos_map, qvel_map, act_map, ctrl_map

  @classmethod
  def from_custom_map(
      cls,
      times: np.ndarray,
      data: np.ndarray,
      signals: Sequence[str | tuple[str, int, SignalType]],
  ) -> TimeSeries:
    """Construct a TimeSeries from custom data with explicit signal definitions.

    Use this when you have custom signal types (e.g., from a custom
    modify_residual function) that are not auto-resolved from a MuJoCo model.
    You must explicitly specify the signal names, widths, and types.

    Args:
      times: 1-D timestamp array of length N.
      data: 2-D array of shape ``(N, D)``.
      signals: Defines the layout of the columns in ``data``.

    Returns:
      A TimeSeries object with the constructed signal mapping.
    """
    if data.ndim != 2:
      raise ValueError(
          "The 'data' array must be 2-dimensional (Time x Features)."
      )

    signal_mapping_dict: SignalMappingType = {}
    current_index = 0
    total_width = 0

    for item in signals:
      if isinstance(item, str):
        name = item
        width = 1
        sig_type = SignalType.CustomObs
      else:
        name, width, sig_type = item

      if width <= 0:
        raise ValueError(
            f"Signal '{name}' must have positive width, got {width}."
        )

      indices = np.arange(current_index, current_index + width)
      signal_mapping_dict[name] = (sig_type, indices)
      current_index += width
      total_width += width

    if total_width != data.shape[1]:
      raise ValueError(
          f"Total width of signals ({total_width}) does not match "
          f"data columns ({data.shape[1]})."
      )

    return cls(times=times, data=data, signal_mapping=signal_mapping_dict)

  @classmethod
  def from_names(
      cls,
      times: np.ndarray,
      data: np.ndarray,
      model: mujoco.MjModel,
      names: Sequence[str | tuple[str, SignalType]] | None = None,
  ) -> TimeSeries:
    """Construct a TimeSeries for observations from the model.

    This method automatically resolves signal names (sensors, qpos, qvel, act)
    from
    the MuJoCo model, determining their types and data layout. Use this for
    standard
    observation signals that are defined in the model.

    Args:
      times: 1-D timestamps of length N.
      data: 2-D array of shape (N, D).
      model: MuJoCo model used to auto-resolve signal names and types.
      names: Signal names to map. Can be strings or (name, SignalType) tuples.
        If None, maps ALL model sensors in sensor address order.

    Warning:
      When names=None, data columns MUST match the model's sensor layout
      (i.e., data[:, i] corresponds to model.sensordata[i] during simulation).
      If your data is in a different order, pass explicit names.

    Raises:
      ValueError: If MjCtrl signals are passed (use from_control_names).
    """
    if data.ndim != 2:
      raise ValueError(
          "The 'data' array must be 2-dimensional (Time x Features)."
      )

    if names is None:
      signal_mapping = cls.compute_all_sensor_mapping(model)
      # Verify width: assumes data contains ALL sensors in sensor_adr order
      if model.nsensordata != data.shape[1]:
        raise ValueError(
            f"Data columns ({data.shape[1]}) do not match model sensors dim"
            f" ({model.nsensordata})."
        )
    else:
      signal_mapping = _resolve_signals(
          model,
          names,
          allowed_types={
              SignalType.MjSensor,
              SignalType.MjStateQPos,
              SignalType.MjStateQVel,
              SignalType.MjStateAct,
          },
      )
      # Verify total resolved width
      max_idx = 0
      for _, indices in signal_mapping.values():
        if indices.size:
          max_idx = max(max_idx, indices[-1] + 1)
      if max_idx != data.shape[1]:
        raise ValueError(
            f"Resolved signal width ({max_idx}) does not match data columns"
            f" ({data.shape[1]})."
        )

    return cls(times=times, data=data, signal_mapping=signal_mapping)

  @classmethod
  def from_control_names(
      cls,
      times: np.ndarray,
      data: np.ndarray,
      model: mujoco.MjModel,
      names: Sequence[str | tuple[str, SignalType]] | None = None,
  ) -> TimeSeries:
    """Construct a TimeSeries for control signals from the model.

    This method automatically resolves control/actuator names from the MuJoCo
    model,
    determining their layout. Use this for control signals (MjCtrl type).

    Args:
      times: 1-D timestamps of length N.
      data: 2-D array of shape (N, model.nu).
      model: MuJoCo model used to auto-resolve actuator names.
      names: Actuator names to map. If None, maps ALL actuators in order.

    Warning:
      When names=None, data columns MUST match actuator order in the model
      (i.e., data[:, i] corresponds to actuator i). Pass explicit names
      if your data is in a different order.
    """
    if data.ndim != 2:
      raise ValueError(
          "The 'data' array must be 2-dimensional (Time x Features)."
      )

    if names is None:
      signal_mapping = cls.compute_all_control_mapping(model)
      if model.nu != data.shape[1]:
        raise ValueError(
            f"Data columns ({data.shape[1]}) do not match model controls"
            f" ({model.nu})."
        )
    else:
      signal_mapping = _resolve_signals(
          model, names, allowed_types={SignalType.MjCtrl}
      )
      max_idx = 0
      for _, indices in signal_mapping.values():
        if indices.size:
          max_idx = max(max_idx, indices[-1] + 1)
      if max_idx != data.shape[1]:
        raise ValueError(
            f"Resolved signal width ({max_idx}) does not match data columns"
            f" ({data.shape[1]})."
        )

    return cls(times=times, data=data, signal_mapping=signal_mapping)

  def get_indices(self, obs_name: str) -> tuple[SignalType, np.ndarray]:
    """Look up the signal type and column indices for a named observation.

    Args:
      obs_name: Name of the observation signal.
    """
    assert self.signal_mapping is not None
    if obs_name not in self.signal_mapping:
      raise ValueError(
          f"{obs_name} observation is not in the observation name map."
      )
    return self.signal_mapping[obs_name]

  @classmethod
  def create(
      cls,
      times: np.ndarray,
      data: np.ndarray,
      signal_mapping: dict[
          str, tuple[SignalType, np.ndarray | list[int] | int]
      ],
  ) -> TimeSeries:
    """Construct a TimeSeries, normalizing index entries to ``np.ndarray``.

    Args:
      times: 1-D timestamp array.
      data: Data array with first axis corresponding to time.
      signal_mapping: Dict mapping signal names to ``(type, indices)`` tuples.
        Index entries are coerced to ``np.ndarray``.
    """
    normalized: SignalMappingType = {}
    for key in signal_mapping:
      signal_type, indices = signal_mapping[key]
      normalized[key] = (signal_type, np.atleast_1d(indices))

    return cls(times, data, normalized)

  @classmethod
  def slice_by_name(
      cls, ts: TimeSeries, enabled_sensors: list[str]
  ) -> TimeSeries:
    """Return a new TimeSeries containing only the named signals.

    Columns are re-indexed so the resulting ``signal_mapping`` has contiguous
    indices starting from 0.

    Args:
      ts: The source TimeSeries.
      enabled_sensors: Names of signals to keep.
    """
    if not ts.signal_mapping:
      return ts

    original_indices_to_keep = []
    original_to_new_index_map = {}
    all_original_indices = []
    for name in ts.signal_mapping:
      all_original_indices.extend(ts.signal_mapping[name][1])

    # Build a set of indices to keep for quick lookups
    kept_indices_set = set()
    for name in enabled_sensors:
      if name not in ts.signal_mapping:
        raise ValueError(
            f"Attemping to slice TimeSeries failed. {name} is not in"
            f" {ts.signal_mapping}."
        )
      kept_indices_set.update(ts.signal_mapping[name][1])
    new_index_counter = 0
    for original_index in all_original_indices:
      if original_index in kept_indices_set:
        original_to_new_index_map[original_index] = new_index_counter
        new_index_counter += 1
        original_indices_to_keep.append(original_index)

    data = ts.data[..., original_indices_to_keep]

    trimmed_signal_mapping = {}
    for name in enabled_sensors:
      metadata, original_indices = ts.signal_mapping[name]

      new_indices = []
      for original_index in original_indices:
        new_indices.append(original_to_new_index_map[original_index])

      trimmed_signal_mapping[name] = (metadata, np.asarray(new_indices))

    return cls(ts.times, data, trimmed_signal_mapping)

  def __post_init__(self):
    """Validate the time series data after initialization.

    Raises:
      ValueError: If times is not 1D, if lengths don't match, if times
        is not strictly increasing, or if arrays are empty.
    """
    if self.times.size == 0:
      raise ValueError("Empty arrays are not allowed in TimeSeries")
    if self.times.ndim != 1:
      raise ValueError(
          f"times must be a 1D array, got {self.times.ndim}D array"
      )
    if len(self.times) != len(self.data):
      raise ValueError(
          f"Length of times ({len(self.times)}) and data ({len(self.data)})"
          " must match"
      )
    if not np.all(np.diff(self.times) > 0):
      raise ValueError("times must be strictly increasing")

  def __len__(self) -> int:
    return len(self.data)

  def save_to_disk(self, path: str | pathlib.Path) -> None:
    """Save the time series data to disk.

    Args:
      path: Path where the data will be saved.
    """
    save_dict = {"times": self.times, "data": self.data}
    if self.signal_mapping:
      save_dict["signal_mapping"] = np.asarray(
          encode_signal_mapping(self.signal_mapping)
      )
    np.savez(path, **save_dict)

  def save_to_csv(self, path: str | pathlib.Path) -> None:
    """Save the time series data to a CSV file.

    Args:
      path: Path where the CSV file will be written.
    """
    np.savetxt(
        path,
        np.column_stack((self.times, self.data)),
        delimiter=",",
    )

  @classmethod
  def load_from_disk(cls, path: str | pathlib.Path) -> TimeSeries:
    """Load time series data from disk.

    Args:
      path: Path to the saved data.

    Returns:
      A new TimeSeries object.
    """
    with np.load(path, allow_pickle=False) as npz:
      times = npz["times"]
      data = npz["data"]
      if "signal_mapping" in npz:
        signal_mapping = decode_signal_mapping(str(npz["signal_mapping"]))
      else:
        signal_mapping = None

    return cls(times=times, data=data, signal_mapping=signal_mapping)

  def interpolate(
      self, t: float | np.ndarray, method: InterpolationMethod = "linear"
  ) -> np.ndarray:
    """Interpolate data at specified time(s).

    This is the core interpolation function used by both get() and resample().

    Args:
      t: Time point(s) at which to interpolate data.
      method: Interpolation method to use.

    Returns:
      Interpolated data values.
    """
    t = np.atleast_1d(np.asarray(t))

    if method in ("zero_order_hold", "zoh"):
      indices = np.searchsorted(self.times, t, side="right") - 1
      indices = np.clip(indices, 0, len(self.times) - 1)
      return self.data[indices]

    return scipy.interpolate.interp1d(
        self.times,
        self.data,
        kind=method,
        axis=0,
        bounds_error=False,
        fill_value=(self.data[0], self.data[-1]),  # pyright: ignore
        assume_sorted=True,
    )(t)

  def get(
      self, t: float | np.ndarray, method: InterpolationMethod = "linear"
  ) -> tuple[np.ndarray, np.ndarray]:
    """Get interpolated data at specified time(s).

    This method is useful for querying data at specific timestamps without
    creating a new TimeSeries object.

    Args:
      t: Time point(s) at which to get data.
      method: Interpolation method to use.

    Returns:
      Tuple of (times, interpolated_data).
    """
    t_orig = np.asarray(t)
    t_shape = t_orig.shape
    result = self.interpolate(t_orig, method=method)
    if not t_shape:
      result = result.squeeze(axis=0)
    return t_orig, result

  def resample(
      self,
      new_times: np.ndarray | None = None,
      target_dt: float | None = None,
      method: InterpolationMethod = "linear",
  ) -> TimeSeries:
    """Resample the time series to new timestamps or a specific time interval.

    This method creates a new TimeSeries object with data interpolated at the
    specified timestamps.

    Args:
      new_times: Optional array of new timestamps. If provided, target_dt is
        ignored.
      target_dt: Optional time interval for regular resampling. Only used if
        new_times is None.
      method: Interpolation method to use.

    Returns:
      A new TimeSeries object with resampled data.

    Raises:
      ValueError: If neither new_times nor target_dt is provided, or if
        new_times is not strictly increasing.
    """
    # Generate new times if target_dt is provided.
    if new_times is None:
      if target_dt is None:
        raise ValueError("Either new_times or target_dt must be provided")
      if target_dt <= 0:
        raise ValueError("target_dt must be a positive float")

      # Create evenly spaced timestamps.
      new_nsteps = (
          int(np.ceil((self.times[-1] - self.times[0]) / target_dt)) + 1
      )
      new_times = np.linspace(
          self.times[0], self.times[-1], new_nsteps, endpoint=True
      )
    else:
      # Make sure new_times is valid.
      if new_times.ndim != 1:
        raise ValueError("new_times must be a 1D array")
      if not np.all(np.diff(new_times) > 0):
        raise ValueError("new_times must be strictly increasing")

    assert new_times is not None
    new_data = self.interpolate(new_times, method=method)
    return TimeSeries(
        times=new_times, data=new_data, signal_mapping=self.signal_mapping
    )

  def remove_from_beginning(self, time_to_remove_s: float) -> TimeSeries:
    """Remove time from the beginning of the time series.

    Args:
      time_to_remove_s: Time to remove from the beginning of the time series.

    Returns:
      A new TimeSeries object with the specified time removed.
    """
    if time_to_remove_s < 0:
      raise ValueError("time_to_remove_s must be non-negative")
    if time_to_remove_s > self.times[-1]:
      raise ValueError(
          "time_to_remove_s is greater than the duration of the time series"
      )
    idx = np.searchsorted(self.times, time_to_remove_s)
    times_shifted = self.times[idx:] - self.times[idx]
    return TimeSeries(
        times=times_shifted,
        data=self.data[idx:],
        signal_mapping=self.signal_mapping,
    )

  def dt_statistics(self) -> dict[str, float]:
    """Calculate statistics about the time intervals.

    Returns:
      Dictionary with mean, median, std, min, and max of time intervals.

    Raises:
      ValueError: If there are fewer than two timestamps.
    """
    if self.times.size < 2:
      raise ValueError(
          "Must have at least two timestamps to compute dt statistics."
      )
    dt_values = np.diff(self.times)
    stats = {}
    for fn in ["mean", "median", "std", "min", "max"]:
      stats[fn] = float(getattr(np, fn)(dt_values))
    return stats

  def __repr__(self) -> str:
    """Return a string representation of the TimeSeries object."""
    t_start, t_end = self.times[0], self.times[-1]
    duration = t_end - t_start

    data_shape = self.data.shape
    n_samples = len(self)

    dt_stats = self.dt_statistics()
    mean_dt = dt_stats["mean"]
    min_dt = dt_stats["min"]
    max_dt = dt_stats["max"]

    is_uniform = dt_stats["std"] / mean_dt < 0.01  # Less than 1% variation.

    # Calculate data range (min/max values).
    data_min = np.min(self.data)
    data_max = np.max(self.data)
    data_range = f"[{data_min:.3g}, {data_max:.3g}]"

    parts = [
        "TimeSeries(",
        f"  samples={n_samples}",
        f"  shape={data_shape}",
        f"  time_range=[{t_start:.3g}, {t_end:.3g}] (duration={duration:.3g})",
        f"  dt={mean_dt:.3g}"
        + (
            " (uniform)"
            if is_uniform
            else f" (min={min_dt:.3g}, max={max_dt:.3g})"
        ),
        f"  data_range={data_range}",
        f"  signal_mapping={self.signal_mapping}",
        ")",
    ]

    return "\n".join(parts)
