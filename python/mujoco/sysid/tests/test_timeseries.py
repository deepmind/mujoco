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
"""Tests for the TimeSeries class and factory methods."""

import mujoco
from mujoco.sysid._src.timeseries import SignalType
from mujoco.sysid._src.timeseries import TimeSeries
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scalar_ts():
  """y = x^2."""
  times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  data = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
  return TimeSeries(times=times, data=data)


@pytest.fixture
def multi_ts():
  """y = [x^2, 2*x^2]."""
  times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  data = np.array([
      [0.0, 0.0],
      [1.0, 2.0],
      [4.0, 8.0],
      [9.0, 18.0],
      [16.0, 32.0],
  ])
  return TimeSeries(times=times, data=data)


# ---------------------------------------------------------------------------
# Core TimeSeries tests
# ---------------------------------------------------------------------------


def test_basics(scalar_ts, multi_ts):
  """Basic properties: length, times array, and data array are all accessible."""
  assert len(scalar_ts) == 5
  assert len(multi_ts) == 5
  np.testing.assert_array_equal(scalar_ts.times, [0, 1, 2, 3, 4])
  np.testing.assert_array_equal(scalar_ts.data, [0, 1, 4, 9, 16])


@pytest.mark.parametrize(
    "times, data, match",
    [
        (np.array([]), np.array([]), "Empty"),
        (np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), "1D"),
        (np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), "Length"),
        (
            np.array([0.0, 2.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
            "strictly increasing",
        ),
    ],
)
def test_validation(times, data, match):
  """Bad inputs (empty, non-1D times, length mismatch, non-monotonic) are rejected."""
  with pytest.raises(ValueError, match=match):
    TimeSeries(times=times, data=data)


def test_zero_column_data():
  """Zero-column data is valid (state-based models with no sensors)."""
  times = np.array([0.0, 1.0, 2.0])
  data = np.empty((3, 0))
  ts = TimeSeries(times=times, data=data)
  assert len(ts) == 3
  assert ts.data.shape == (3, 0)


def test_save_and_load(scalar_ts, multi_ts, tmp_path):
  """Saving to .npz and loading back recovers identical times and data."""
  path = tmp_path / "test.npz"
  scalar_ts.save_to_disk(path)
  loaded = TimeSeries.load_from_disk(path)
  np.testing.assert_array_equal(loaded.times, scalar_ts.times)
  np.testing.assert_array_equal(loaded.data, scalar_ts.data)

  path2 = tmp_path / "test_multi.npz"
  multi_ts.save_to_disk(path2)
  loaded2 = TimeSeries.load_from_disk(path2)
  np.testing.assert_array_equal(loaded2.times, multi_ts.times)
  np.testing.assert_array_equal(loaded2.data, multi_ts.data)


@pytest.mark.parametrize(
    "nsteps, ncolumns, scalar",
    [(3, 1, True), (1, 1, True), (3, 1, False), (3, 2, False), (3, 0, False)],
)
def test_save_to_csv(tmp_path, nsteps, ncolumns, scalar):
  """CSV output contains timestamps followed by each signal column."""
  times = np.arange(nsteps) * 0.5
  columns = np.arange(nsteps * ncolumns).reshape(nsteps, ncolumns)
  data = columns[:, 0] if scalar else columns
  ts = TimeSeries(times=times, data=data)
  path = tmp_path / "timeseries.csv"

  ts.save_to_csv(path)

  loaded = np.loadtxt(path, delimiter=",", ndmin=2)
  assert loaded.shape == (nsteps, ncolumns + 1)
  np.testing.assert_array_equal(loaded[:, 0], times)
  np.testing.assert_array_equal(loaded[:, 1:], columns)


def test_save_and_load_with_signal_mapping(tmp_path):
  """Save/load also preserves the signal_mapping (sensor name -> column index map)."""
  times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  data = np.array([
      [0.0, 0.0],
      [1.0, 2.0],
      [4.0, 8.0],
      [9.0, 18.0],
      [16.0, 32.0],
  ])
  signal_mapping = {
      "signal1": (SignalType.MjSensor, np.array([0])),
      "signal2": (SignalType.MjSensor, np.array([1])),
  }
  ts = TimeSeries(
      times=times, data=data, signal_mapping=signal_mapping
  )

  path = tmp_path / "test_signal_mapping.npz"
  ts.save_to_disk(path)
  loaded = TimeSeries.load_from_disk(path)

  np.testing.assert_array_equal(loaded.times, times)
  np.testing.assert_array_equal(loaded.data, data)
  assert loaded.signal_mapping is not None
  assert loaded.signal_mapping.keys() == signal_mapping.keys()
  for key in signal_mapping:
    val_type, val_indices = signal_mapping[key]
    loaded_type, loaded_indices = loaded.signal_mapping[key]
    assert loaded_type == val_type
    np.testing.assert_array_equal(loaded_indices, val_indices)


@pytest.mark.parametrize(
    "method, expected",
    [
        ("linear", 6.5),
        ("cubic", 6.25),
        ("quadratic", 6.25),
        ("zero_order_hold", 4.0),
        ("zoh", 4.0),
    ],
)
def test_interpolate_scalar(scalar_ts, method, expected):
  """Each interpolation method (linear, cubic, ZOH, etc.) gives the expected midpoint value."""
  result = scalar_ts.interpolate(2.5, method=method)
  assert result[0] == pytest.approx(expected, abs=1e-5)


def test_interpolate_array(scalar_ts, multi_ts):
  """Interpolating at multiple times simultaneously works for scalar and multi-column data."""
  t_values = np.array([0.5, 1.5, 2.5, 3.5])
  expected = np.array([0.5, 2.5, 6.5, 12.5])
  result = scalar_ts.interpolate(t_values, method="linear")
  np.testing.assert_allclose(result, expected, rtol=1e-5)

  expected_multi = np.array([
      [0.5, 1.0],
      [2.5, 5.0],
      [6.5, 13.0],
      [12.5, 25.0],
  ])
  result_multi = multi_ts.interpolate(t_values, method="linear")
  np.testing.assert_allclose(result_multi, expected_multi, rtol=1e-5)


def test_resample_with_new_times(scalar_ts):
  """Resampling onto a finer time grid via explicit new_times gives correct values."""
  new_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
  expected = np.array([0.0, 0.5, 1.0, 2.5, 4.0, 6.5, 9.0, 12.5, 16.0])
  resampled = scalar_ts.resample(new_times=new_times, method="linear")
  np.testing.assert_array_equal(resampled.times, new_times)
  np.testing.assert_allclose(resampled.data, expected, rtol=1e-5)

  with pytest.raises(ValueError):
    scalar_ts.resample(new_times=np.array([0.0, 2.0, 1.0]))

  with pytest.raises(ValueError):
    scalar_ts.resample(new_times=np.array([[0.0], [1.0]]))


def test_resample_with_target_dt(scalar_ts):
  """Resampling by specifying a target timestep generates the right uniform grid."""
  resampled = scalar_ts.resample(target_dt=0.5, method="linear")
  expected_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
  expected_data = np.array([0.0, 0.5, 1.0, 2.5, 4.0, 6.5, 9.0, 12.5, 16.0])
  np.testing.assert_allclose(resampled.times, expected_times, rtol=1e-5)
  np.testing.assert_allclose(resampled.data, expected_data, rtol=1e-5)

  with pytest.raises(ValueError):
    scalar_ts.resample(target_dt=-0.5)

  with pytest.raises(ValueError):
    scalar_ts.resample()


# ---------------------------------------------------------------------------
# TimeSeries factory method tests
# ---------------------------------------------------------------------------


def test_from_model_controls_auto_resolution():
  """Without explicit names, all model actuators are auto-discovered and mapped."""
  xml = """
  <mujoco>
    <worldbody>
      <body>
        <joint name="j1"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="m1" joint="j1"/>
      <motor name="m2" joint="j1"/>
    </actuator>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  times = np.linspace(0, 1, 100)
  data = np.random.randn(100, 2)

  ts = TimeSeries.from_control_names(times, data, model)
  assert ts.signal_mapping is not None
  assert "m1_ctrl" in ts.signal_mapping
  assert "m2_ctrl" in ts.signal_mapping
  assert ts.signal_mapping["m1_ctrl"][0] == SignalType.MjCtrl
  assert ts.signal_mapping["m2_ctrl"][0] == SignalType.MjCtrl


def test_from_model_controls_explicit_names():
  """Explicit actuator names are resolved; invalid or wrong-type names are rejected."""
  xml = """
  <mujoco>
    <worldbody>
      <body>
        <joint name="j1"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="m1" joint="j1"/>
    </actuator>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  times = np.linspace(0, 1, 10)
  data = np.zeros((10, 1))

  ts = TimeSeries.from_control_names(times, data, model, names=["m1"])
  assert ts.signal_mapping is not None
  assert "m1_ctrl" in ts.signal_mapping

  with pytest.raises(ValueError, match="Could not resolve signal"):
    TimeSeries.from_control_names(times, data, model, names=["invalid"])

  with pytest.raises(ValueError, match="not allowed"):
    TimeSeries.from_control_names(
        times, data, model, names=[("m1", SignalType.MjSensor)]
    )


def test_from_model_auto_resolution_sensors():
  """Without explicit names, all model sensors are auto-discovered with correct dimensions."""
  xml = """
  <mujoco>
    <worldbody>
      <body name="b1">
        <geom size="0.1" mass="1"/>
        <site name="s1"/>
      </body>
    </worldbody>
    <sensor>
      <accelerometer name="acc1" site="s1"/>
      <gyro name="gyro1" site="s1"/>
    </sensor>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  times = np.linspace(0, 1, 10)
  data = np.zeros((10, 6))

  ts = TimeSeries.from_names(times, data, model)
  assert ts.signal_mapping is not None
  assert "acc1" in ts.signal_mapping
  assert "gyro1" in ts.signal_mapping
  assert ts.signal_mapping["acc1"][0] == SignalType.MjSensor
  assert ts.signal_mapping["gyro1"][0] == SignalType.MjSensor


def test_from_model_state_resolution():
  """State signals (qpos, qvel) can be mapped by passing (name, SignalType) tuples."""
  xml = """
  <mujoco>
    <worldbody>
      <body name="b1">
        <joint name="j1" type="hinge"/>
        <joint name="j2" type="slide"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  times = np.linspace(0, 1, 10)
  data = np.zeros((10, 2))

  names = [("j1", SignalType.MjStateQPos), ("j2", SignalType.MjStateQPos)]
  ts = TimeSeries.from_names(times, data, model, names=names)
  assert ts.signal_mapping is not None
  assert "j1_qpos" in ts.signal_mapping
  assert "j2_qpos" in ts.signal_mapping


def test_from_custom():
  """Custom signal definitions (name strings and dimension tuples) are mapped correctly."""
  times = np.linspace(0, 1, 10)
  data = np.zeros((10, 3))
  signals = ["a", ("b", 2, SignalType.CustomObs)]

  ts = TimeSeries.from_custom_map(times, data, signals)
  assert ts.signal_mapping is not None
  assert "a" in ts.signal_mapping
  assert "b" in ts.signal_mapping
  assert ts.signal_mapping["a"][1].size == 1
  assert ts.signal_mapping["b"][1].size == 2


# ---------------------------------------------------------------------------
# compute_all_state_mappings correctness tests
# ---------------------------------------------------------------------------


def _verify_state_mapping(model):
  """Set known qpos/qvel values and verify mappings recover them correctly."""
  qpos_map, qvel_map, act_map, _ = TimeSeries.compute_all_state_mappings(model)

  data = mujoco.MjData(model)
  # Fill with distinct values so misalignment is detectable.
  data.qpos[:] = 100 + np.arange(model.nq)
  data.qvel[:] = 200 + np.arange(model.nv)

  state = np.empty(
      mujoco.mj_stateSize(
          model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
      )
  )
  mujoco.mj_getState(
      model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
  )
  # Strip the leading time element (mjSTATE_FULLPHYSICS includes time).
  state_no_time = state[1:]

  # Verify every qpos mapping entry.
  for name, (sig_type, indices) in qpos_map.items():
    assert sig_type == SignalType.MjStateQPos
    values = state_no_time[indices]
    # All qpos values should be in [100, 100+nq).
    assert np.all(values >= 100) and np.all(values < 100 + model.nq), (
        f"{name}: got {values}"
    )

  # Verify every qvel mapping entry.
  for name, (sig_type, indices) in qvel_map.items():
    assert sig_type == SignalType.MjStateQVel
    values = state_no_time[indices]
    # All qvel values should be in [200, 200+nv).
    assert np.all(values >= 200) and np.all(values < 200 + model.nv), (
        f"{name}: got {values}"
    )

  # Verify total coverage.
  all_qpos_indices = np.concatenate([v[1] for v in qpos_map.values()])
  all_qvel_indices = np.concatenate([v[1] for v in qvel_map.values()])
  assert len(all_qpos_indices) == model.nq
  assert len(all_qvel_indices) == model.nv
  assert len(np.unique(all_qpos_indices)) == model.nq, "Duplicate qpos indices"
  assert len(np.unique(all_qvel_indices)) == model.nv, "Duplicate qvel indices"


def test_state_mapping_hinge_only():
  """All-hinge model: qposadr == dofadr, so mapping is straightforward."""
  xml = """
  <mujoco>
    <worldbody>
      <body><joint name="j1" type="hinge"/><geom size="0.1" mass="1"/>
        <body><joint name="j2" type="hinge"/><geom size="0.1" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  _verify_state_mapping(mujoco.MjModel.from_xml_string(xml))


def test_state_mapping_free_plus_hinge():
  """Free body + hinge joints: jnt_qposadr != jnt_dofadr for the hinges."""
  xml = """
  <mujoco>
    <worldbody>
      <body name="box" pos="0 0 1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
      <body>
        <joint name="h1" type="hinge"/><geom size="0.1" mass="1"/>
        <body>
          <joint name="h2" type="hinge"/><geom size="0.1" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  _verify_state_mapping(mujoco.MjModel.from_xml_string(xml))


def test_state_mapping_two_free_bodies():
  """Two free bodies: body_dofadr != jnt_qposadr for the second body."""
  xml = """
  <mujoco>
    <worldbody>
      <body name="a" pos="0 0 1">
        <freejoint/><geom type="box" size=".1 .1 .1"/>
      </body>
      <body name="b" pos="1 0 1">
        <freejoint/><geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  _verify_state_mapping(mujoco.MjModel.from_xml_string(xml))


def test_state_mapping_ball_plus_hinge():
  """Ball joint + hinge: ball takes 4 qpos / 3 qvel, offsetting the hinge."""
  xml = """
  <mujoco>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/><geom size="0.1" mass="1"/>
        <body>
          <joint name="h" type="hinge"/><geom size="0.1" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  _verify_state_mapping(mujoco.MjModel.from_xml_string(xml))
