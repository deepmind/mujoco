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
"""Tests for the SystemTrajectory class."""

from unittest import mock

import mujoco
from mujoco.sysid._src import timeseries
from mujoco.sysid._src.trajectory import create_initial_state
from mujoco.sysid._src.trajectory import SystemTrajectory
import numpy as np
import pytest


@pytest.fixture
def mock_model():
  model = mock.Mock(spec=mujoco.MjModel)
  model.nsensordata = 2
  model.nu = 1
  model.nq = 1
  model.nv = 1
  return model


@pytest.fixture
def sample_trajectory(mock_model):
  """Create a sample SystemTrajectory for testing."""
  with mock.patch.object(
      SystemTrajectory, "check_compatible", return_value=None
  ):
    times = np.array([0.0, 1.0, 2.0])
    control_mapping = {"ctrl1": (timeseries.SignalType.MjCtrl, np.array([0]))}
    sensordata_mapping = {
        "sensor1": (timeseries.SignalType.MjSensor, np.array([0])),
        "sensor2": (timeseries.SignalType.MjSensor, np.array([1])),
    }
    state_mapping = {
        "qpos1": (timeseries.SignalType.MjStateQPos, np.array([0]))
    }

    control = timeseries.TimeSeries(
        times=times,
        data=np.array([[1.0], [2.0], [3.0]]),
        signal_mapping=control_mapping,
    )
    sensordata = timeseries.TimeSeries(
        times=times,
        data=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        signal_mapping=sensordata_mapping,
    )
    state = timeseries.TimeSeries(
        times=times,
        data=np.array([[0.01], [0.02], [0.03]]),
        signal_mapping=state_mapping,
    )

    traj = SystemTrajectory(
        model=mock_model,
        control=control,
        sensordata=sensordata,
        initial_state=np.array([0.0]),
        state=state,
    )
    yield traj, control_mapping, sensordata_mapping, state_mapping


def test_save_and_load_with_signal_mapping(
    sample_trajectory, mock_model, tmp_path
):
  """Saving and loading a trajectory preserves all signal mappings (control, sensor, state)."""
  traj, control_mapping, sensordata_mapping, state_mapping = sample_trajectory

  path = tmp_path / "test_traj.npz"
  traj.save_to_disk(path)

  with mock.patch.object(
      SystemTrajectory, "check_compatible", return_value=None
  ):
    loaded = SystemTrajectory.load_from_disk(path, mock_model)

  ctrl_map = loaded.control.signal_mapping
  assert ctrl_map is not None
  assert ctrl_map.keys() == control_mapping.keys()
  for key in control_mapping:
    assert ctrl_map[key][0] == control_mapping[key][0]
    np.testing.assert_array_equal(ctrl_map[key][1], control_mapping[key][1])

  sensor_map = loaded.sensordata.signal_mapping
  assert sensor_map is not None
  assert sensor_map.keys() == sensordata_mapping.keys()
  for key in sensordata_mapping:
    assert sensor_map[key][0] == sensordata_mapping[key][0]
    np.testing.assert_array_equal(
        sensor_map[key][1], sensordata_mapping[key][1]
    )

  assert loaded.state is not None
  state_map = loaded.state.signal_mapping
  assert state_map is not None
  assert state_map.keys() == state_mapping.keys()
  for key in state_mapping:
    assert state_map[key][0] == state_mapping[key][0]
    np.testing.assert_array_equal(state_map[key][1], state_mapping[key][1])


def test_get_sensordata_slice_interleaved(sample_trajectory):
  model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
      <sensor>
        <jointpos joint="j"/><jointvel joint="j"/><jointpos joint="j"/>
      </sensor>
    </mujoco>
  """)
  values = np.arange(9).reshape(3, 3)
  traj, *_ = sample_trajectory
  traj = traj.replace(model=model, sensordata=timeseries.TimeSeries(
      traj.sensordata.times, values
  ))
  np.testing.assert_array_equal(traj.get_sensordata_slice("joint_pos"),
                                values[:, [0, 2]])
  assert traj.get_sensordata_slice("joint_torque").shape == (3, 0)


def test_create_initial_state(box_model):
  """The initial MuJoCo state (qpos, qvel, act) is packed into a flat vector for rollout."""
  qpos = np.zeros(box_model.nq)
  qvel = np.zeros(box_model.nv)
  state = create_initial_state(box_model, qpos, qvel)
  expected_size = mujoco.mj_stateSize(
      box_model, mujoco.mjtState.mjSTATE_FULLPHYSICS
  )
  assert state.shape == (expected_size,)


def test_create_initial_state_wrong_qpos(box_model):
  """Wrong-sized qpos is caught early rather than causing a silent rollout bug."""
  with pytest.raises(ValueError, match="qpos"):
    create_initial_state(box_model, np.zeros(999))


def test_create_initial_state_with_names():
  """Named mapping places qpos/qvel into correct slots for a subset of joints."""
  xml = """
  <mujoco>
    <worldbody>
      <body name="box" pos="0 0 1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
      <body>
        <joint name="h1" type="hinge"/>
        <geom size="0.1" mass="1"/>
        <body>
          <joint name="h2" type="hinge"/>
          <geom size="0.1" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)

  # Map only the two hinge joints by name, leaving the free body at defaults.
  qpos_subset = np.array([1.1, 2.2])
  qvel_subset = np.array([3.3, 4.4])
  state = create_initial_state(
      model, qpos_subset, qvel_subset,
      qpos_names=["h1_qpos", "h2_qpos"],
      qvel_names=["h1_qvel", "h2_qvel"],
  )

  # Unpack the state to verify values ended up in the right slots.
  mujoco.mj_setState(
      model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
  )
  # Free body qpos (indices 0-6) should be at model defaults.
  # Hinge joints at qposadr 7 and 8.
  np.testing.assert_allclose(data.qpos[7], 1.1)
  np.testing.assert_allclose(data.qpos[8], 2.2)
  # Hinge joints at dofadr 6 and 7.
  np.testing.assert_allclose(data.qvel[6], 3.3)
  np.testing.assert_allclose(data.qvel[7], 4.4)


def test_split(sample_trajectory):
  """A long trajectory can be split into smaller chunks for batched optimization."""
  traj, *_ = sample_trajectory
  chunks = traj.split(chunk_size=1)
  assert len(chunks) == 3
  assert len(chunks[0].sensordata) == 1


def test_check_compatible_sensor_mismatch(box_model):
  """Mismatched sensor dimensions between data and model are caught before rollout."""
  times = np.array([0.0, 0.01, 0.02])
  sensordata = timeseries.TimeSeries(times, np.ones((3, 5)))
  control = timeseries.TimeSeries(times, np.ones((3, box_model.nu)))
  initial_state = np.zeros(
      mujoco.mj_stateSize(box_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  )
  traj = SystemTrajectory(
      model=box_model,
      control=control,
      sensordata=sensordata,
      initial_state=initial_state,
      state=None,
  )
  with pytest.raises(ValueError, match="Sensor data dimension"):
    traj.check_compatible()
