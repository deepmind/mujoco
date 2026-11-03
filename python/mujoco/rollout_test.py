# Copyright 2022 DeepMind Technologies Limited
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
"""tests for rollout function."""

import concurrent.futures
import copy
import os
import threading

from absl.testing import absltest
from absl.testing import parameterized

import mujoco
from mujoco import rollout

import numpy as np


DTYPE = mujoco.MJTNUM_DTYPE


def randn(*shape):
  return np.random.standard_normal(shape).astype(DTYPE)


def empty(*shape):
  return np.empty(shape, dtype=DTYPE)


def zeros(*shape):
  return np.zeros(shape, dtype=DTYPE)

# -------------------------- models used for testing ---------------------------

TEST_XML = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="pitch" gainprm="100"/>
    <general joint="yaw" dyntype="filter" dynprm="1" gainprm="100"/>
  </actuator>
  <sensor>
    <accelerometer site="site"/>
  </sensor>
</mujoco>
"""

TEST_XML_NO_SENSORS = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="pitch" gainprm="100"/>
    <general joint="yaw" dyntype="filter" dynprm="1" gainprm="100"/>
  </actuator>
</mujoco>
"""

TEST_XML_NO_ACTUATORS = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer site="site"/>
  </sensor>
</mujoco>
"""

TEST_XML_MOCAP = r"""
<mujoco>
  <worldbody>
    <body name="1" mocap="true">
    </body>
    <body name="2" mocap="true">
    </body>
  </worldbody>
  <sensor>
    <framepos objtype="xbody" objname="1"/>
    <framequat objtype="xbody" objname="1"/>
  </sensor>
</mujoco>
"""

TEST_XML_EMPTY = r"""
<mujoco>
</mujoco>
"""

TEST_XML_DIVERGE = r"""
<mujoco>
  <option>
    <flag gravity="disable"/>
  </option>

  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 -.3" euler="30 45 90">
      <freejoint/>
      <geom type="box" size=".1 .2 .4"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="non-diverging" qpos="0 0 .5 1 0 0 0"/>
  </keyframe>
</mujoco>
"""

ALL_MODELS = {
    'TEST_XML': TEST_XML,
    'TEST_XML_NO_SENSORS': TEST_XML_NO_SENSORS,
    'TEST_XML_NO_ACTUATORS': TEST_XML_NO_ACTUATORS,
    'TEST_XML_EMPTY': TEST_XML_EMPTY,
}

# ------------------------------ tests -----------------------------------------


class MuJoCoRolloutTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)

  # ----------------------------- test basic operation

  @parameterized.parameters(ALL_MODELS.keys())
  def test_single_step(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = randn(nstate)
    control = randn(model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    mujoco.mj_resetData(model, data)
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_one_rollout(self, model_name):
    nstep = 3  # number of timesteps

    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = randn(nstate)
    control = randn(nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_step(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    mujoco.mj_resetData(model, data)
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nbatch_initial_state(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = randn(nbatch, nstate)
    control = randn(nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    mujoco.mj_resetData(model, data)
    control = np.tile(control, (nbatch, 1, 1))
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nbatch_control(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = randn(nstate)
    control = randn(nbatch, nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    mujoco.mj_resetData(model, data)
    initial_state = np.tile(initial_state, (nbatch, 1))
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nbatch_warmstart(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = randn(nstate)
    control = randn(nstep, model.nu)
    initial_warmstart = np.tile(data.qacc_warmstart.copy(), (nbatch, 1))
    state, sensordata = rollout.rollout(
        model, data, initial_state, control, initial_warmstart=initial_warmstart
    )

    mujoco.mj_resetData(model, data)
    initial_state = np.tile(initial_state, (nbatch, 1))
    control = np.tile(control, (nbatch, 1, 1))
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nbatch_state(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = randn(nstate)
    control = randn(nstep, model.nu)
    state = empty(nbatch, nstep, nstate)
    state, sensordata = rollout.rollout(
        model, data, initial_state, control, state=state
    )

    mujoco.mj_resetData(model, data)
    initial_state = np.tile(initial_state, (nbatch, 1))
    control = np.tile(control, (nbatch, 1, 1))
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nbatch_sensordata(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = randn(nstate)
    control = randn(nstep, model.nu)
    sensordata = empty(nbatch, nstep, model.nsensordata)
    state, sensordata = rollout.rollout(
        model, data, initial_state, control, sensordata=sensordata
    )

    mujoco.mj_resetData(model, data)
    initial_state = np.tile(initial_state, (nbatch, 1))
    control = np.tile(control, (nbatch, 1, 1))
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_one_rollout_fixed_ctrl(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 1  # number of rollouts
    nstep = 3  # number of steps

    initial_state = randn(nstate)
    control = randn(model.nu)
    state = empty(nbatch, nstep, nstate)
    sensordata = empty(nbatch, nstep, model.nsensordata)
    rollout.rollout(
        model, data, initial_state, control, state=state, sensordata=sensordata
    )

    control = np.tile(control, (nstep, 1))
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_rollout(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 2  # number of initial states
    nstep = 3  # number of timesteps

    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_model(self, model_name):
    nbatch = 3  # number of initial states and models
    nstep = 3  # number of timesteps

    spec = mujoco.MjSpec.from_string(ALL_MODELS[model_name])

    if len(spec.bodies) > 1:
      model = []
      for i in range(nbatch):
        body = spec.bodies[1]
        assert body.name != 'world'
        body.pos = body.pos + i
        model.append(spec.compile())
    else:
      model = [spec.compile() for _ in range(nbatch)]

    nstate = mujoco.mj_stateSize(model[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model[0])

    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, nstep, model[0].nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_rollout_fixed_ctrl_infer_from_output(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 2  # number of rollouts
    nstep = 3  # number of timesteps

    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, 1, model.nu)
    state = empty(nbatch, nstep, nstate)
    state, sensordata = rollout.rollout(
        model, data, initial_state, control, state=state
    )

    control = np.repeat(control, nstep, axis=1)
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  @parameterized.parameters(ALL_MODELS.keys())
  def test_py_rollout_generalized_control(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 4  # number of rollouts
    nstep = 3  # number of timesteps

    initial_state = randn(nbatch, nstate)

    control_spec = (
        mujoco.mjtState.mjSTATE_CTRL
        | mujoco.mjtState.mjSTATE_QFRC_APPLIED
        | mujoco.mjtState.mjSTATE_XFRC_APPLIED
    )
    ncontrol = mujoco.mj_stateSize(model, control_spec)
    control = randn(nbatch, nstep, ncontrol)

    state, sensordata = rollout.rollout(
        model, data, initial_state, control, control_spec=control_spec
    )

    py_state, py_sensordata = py_rollout(
        model, data, initial_state, control, control_spec=control_spec
    )
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  def test_detect_divergence(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_DIVERGE)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 4  # number of rollouts
    initial_state = empty(nbatch, nstate)

    # get diverging (0, 2) and non-diverging (1, 3) states
    mujoco.mj_getState(
        model, data, initial_state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    mujoco.mj_getState(
        model, data, initial_state[2], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    mujoco.mj_resetDataKeyframe(model, data, 0)  # keyframe 0 does not diverge
    mujoco.mj_getState(
        model, data, initial_state[1], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    mujoco.mj_getState(
        model, data, initial_state[3], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )

    nstep = 10000  # divergence after ~15s, timestep = 2e-3

    state = randn(nbatch, nstep, nstate)

    rollout.rollout(model, data, initial_state, state=state)

    # initial_state[0,2] diverged, final timesteps are identical
    assert state[0][-1][0] == state[0][-2][0]
    assert state[2][-1][0] == state[2][-2][0]

    # initial_state[1,3] did not diverge, final timesteps are different
    assert state[1][-1][0] != state[1][-2][0]
    assert state[3][-1][0] != state[3][-2][0]

  # ----------------------------- test threaded operation

  def test_threading(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    num_workers = 32
    nbatch = 100
    nstep = 5
    initial_state = randn(nbatch, nstate)
    state = empty(nbatch, nstep, nstate)
    sensordata = empty(nbatch, nstep, model.nsensordata)
    control = randn(nbatch, nstep, model.nu)

    thread_local = threading.local()

    def thread_initializer():
      thread_local.data = mujoco.MjData(model)

    model_list = [copy.copy(model) for _ in range(nbatch)]

    def call_rollout(initial_state, control, state, sensordata):
      rollout.rollout(
          model_list,
          [thread_local.data],
          initial_state,
          control,
          skip_checks=True,
          nstep=nstep,
          state=state,
          sensordata=sensordata,
      )

    n = nbatch // num_workers  # integer division
    chunks = []  # a list of tuples, one per worker
    for i in range(num_workers - 1):
      chunks.append((
          initial_state[i * n : (i + 1) * n],
          control[i * n : (i + 1) * n],
          state[i * n : (i + 1) * n],
          sensordata[i * n : (i + 1) * n],
      ))

    # last chunk, absorbing the remainder:
    chunks.append((
        initial_state[(num_workers - 1) * n :],
        control[(num_workers - 1) * n :],
        state[(num_workers - 1) * n :],
        sensordata[(num_workers - 1) * n :],
    ))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers, initializer=thread_initializer
    ) as executor:
      futures = []
      for chunk in chunks:
        futures.append(executor.submit(call_rollout, *chunk))
      for future in concurrent.futures.as_completed(futures):
        future.result()

    data = mujoco.MjData(model)
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  def test_threading_native(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    num_workers = 32
    nbatch = 100
    nstep = 5
    initial_state = randn(nbatch, nstate)
    state = empty(nbatch, nstep, nstate)
    sensordata = empty(nbatch, nstep, model.nsensordata)
    control = randn(nbatch, nstep, model.nu)

    model_list = [copy.copy(model) for _ in range(nbatch)]
    data_list = [mujoco.MjData(model) for _ in range(num_workers)]

    rollout.rollout(
        model_list,
        data_list,
        initial_state,
        control,
        nstep=nstep,
        state=state,
        sensordata=sensordata,
    )

    data = mujoco.MjData(model)
    py_state, py_sensordata = py_rollout(model, data, initial_state, control)
    np.testing.assert_array_equal(state, py_state)
    np.testing.assert_array_equal(sensordata, py_sensordata)

  def test_threading_native_persistent_object(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    num_workers = 32
    nbatch = 100
    nstep = 5
    initial_state = randn(nbatch, nstate)
    state = empty(nbatch, nstep, nstate)
    sensordata = empty(nbatch, nstep, model.nsensordata)
    control = randn(nbatch, nstep, model.nu)

    model_list = [copy.copy(model) for _ in range(nbatch)]
    data_list = [mujoco.MjData(model) for _ in range(num_workers)]

    with rollout.Rollout(nthread=num_workers) as rollout_:
      for _ in range(2):
        rollout_.rollout(
            model_list,
            data_list,
            initial_state,
            control,
            nstep=nstep,
            state=state,
            sensordata=sensordata,
        )

      data = mujoco.MjData(model)
      py_state, py_sensordata = py_rollout(model, data, initial_state, control)
      np.testing.assert_array_equal(state, py_state)
      np.testing.assert_array_equal(sensordata, py_sensordata)

    rollout_ = rollout.Rollout(nthread=num_workers)
    for _ in range(2):
      rollout_.rollout(
          model_list,
          data_list,
          initial_state,
          control,
          nstep=nstep,
          state=state,
          sensordata=sensordata,
      )

      data = mujoco.MjData(model)
      py_state, py_sensordata = py_rollout(model, data, initial_state, control)
      np.testing.assert_array_equal(state, py_state)
      np.testing.assert_array_equal(sensordata, py_sensordata)
    rollout_.close()

  def test_threading_native_persistent_function(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    num_workers = 32
    nbatch = 100
    nstep = 5
    initial_state = randn(nbatch, nstate)
    state = empty(nbatch, nstep, nstate)
    sensordata = empty(nbatch, nstep, model.nsensordata)
    control = randn(nbatch, nstep, model.nu)

    model_list = [copy.copy(model) for _ in range(nbatch)]
    data_list = [mujoco.MjData(model) for _ in range(num_workers)]

    for _ in range(2):
      rollout.rollout(
          model_list,
          data_list,
          initial_state,
          control,
          nstep=nstep,
          state=state,
          sensordata=sensordata,
          persistent_pool=True,
      )

      data = mujoco.MjData(model)
      py_state, py_sensordata = py_rollout(model, data, initial_state, control)
      np.testing.assert_array_equal(state, py_state)
      np.testing.assert_array_equal(sensordata, py_sensordata)
    rollout.shutdown_persistent_pool()

  # ---------------------------- test advanced operation

  def test_warmstart(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    # take one step, save the state
    state0 = zeros(nstate)
    control = zeros(model.nu)
    state1, _ = step(model, data, state0, control)

    # save qacc_warmstart
    initial_warmstart = data.qacc_warmstart.copy()

    # take one more step (uses correct warmstart)
    state2, _ = step(model, data, state1[0], control)

    # take step using rollout, don't take warmstart into account
    state, _ = rollout.rollout(model, data, state1[0], control)

    # assert that stepping without warmstarts is not exact
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, state, state2
    )

    # take step using rollout, take warmstart into account
    state, _ = rollout.rollout(
        model, data, state1, control, initial_warmstart=initial_warmstart
    )

    # assert exact equality
    np.testing.assert_array_equal(state, np.expand_dims(state2, axis=0))

  def test_mocap(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_MOCAP)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = zeros(nstate)

    control_spec = (
        mujoco.mjtState.mjSTATE_MOCAP_POS | mujoco.mjtState.mjSTATE_MOCAP_QUAT
    )

    pos1 = np.array((1.0, 2.0, 3.0), dtype=DTYPE)
    quat1 = np.array((1.0, 2.0, 3.0, 4.0), dtype=DTYPE)
    quat1 /= np.linalg.norm(quat1)
    pos2 = np.array((2.0, 3.0, 4.0), dtype=DTYPE)
    quat2 = np.array((2.0, 3.0, 4.0, 5.0), dtype=DTYPE)
    quat2 /= np.linalg.norm(quat2)
    control = np.hstack((pos1, pos2, quat1, quat2))

    _, sensordata = rollout.rollout(
        model, data, initial_state, control, control_spec=control_spec
    )

    np.testing.assert_array_almost_equal(sensordata[0][0][:3], pos1)
    np.testing.assert_array_almost_equal(sensordata[0][0][3:], quat1)

  # ---------------------------- test correctness

  def test_intercept_mj_errors(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 1
    nstep = 3

    initial_state = zeros(nbatch, nstate)
    ctrl = zeros(nbatch, nstep, model.nu)

    model.opt.solver = 10  # invalid solver type
    with self.assertRaisesWithLiteralMatch(
        mujoco.FatalError, 'mj_fwdConstraint: unknown solver type 10'
    ):
      rollout.rollout(model, data, initial_state, ctrl)

  def test_invalid(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 1

    initial_state = zeros(nbatch, nstate)

    control = 'string'
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'control must be a numpy array or float'
    ):
      rollout.rollout(model, data, initial_state, control)

    control = zeros(2, 3, 4, 5)
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'control can have at most 3 dimensions'
    ):
      rollout.rollout(model, data, initial_state, control)

  def test_threadpool_rejected(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = [mujoco.MjData(model) for _ in range(2)]
    mujoco.mju_threadpool(data[1], 2)
    initial_state = zeros(4, nstate)
    with self.assertRaisesRegex(ValueError, r'data\[1\] has a thread pool'):
      rollout.rollout(model, data, initial_state)
    with self.assertRaisesRegex(ValueError, r'data\[0\] has a thread pool'):
      rollout.rollout(model, data[1], initial_state)

  def test_bad_sizes(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nbatch = 1
    nstep = 3

    initial_state = randn(nbatch, nstate + 1)
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'trailing dimension of initial_state must be 6, got 7'
    ):
      rollout.rollout(model, data, initial_state)

    initial_state = randn(nbatch, nstate)
    control = randn(1, nstep, model.nu + 1)
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'trailing dimension of control must be 2, got 3'
    ):
      rollout.rollout(model, data, initial_state, control)

    control = randn(nbatch, nstep, model.nu)
    state = randn(nbatch, nstep + 1, nstate)  # incompatible nstep
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'dimension 1 inferred as 3 but state has 4'
    ):
      rollout.rollout(model, data, initial_state, control, state=state)

    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, nstep, model.nu)
    bad_spec = mujoco.mjtState.mjSTATE_ACT
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'control_spec can only contain bits in mjSTATE_USER'
    ):
      rollout.rollout(
          model, data, initial_state, control, control_spec=bad_spec
      )

  def test_nonpositive_nstep(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)
    initial_state = zeros(1, nstate)

    # a nonpositive nstep used to alias "infer the step count from control"
    for nstep in (0, -1):
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'nstep must be positive'
      ):
        rollout.rollout(model, data, initial_state, nstep=nstep)

  def test_noninteger_nstep_chunk_size(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)
    initial_state = zeros(1, nstate)

    # falsy non-integers used to slip through the truthiness-guarded checks
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'nstep must be an integer'
    ):
      rollout.rollout(model, data, initial_state, nstep='')
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'chunk_size must be an integer'
    ):
      rollout.rollout(model, data, initial_state, chunk_size='')

  def test_nonpositive_chunk_size(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = [mujoco.MjData(model) for _ in range(2)]

    nbatch = 4
    nstep = 5
    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, nstep, model.nu)

    # at nbatch=4, 0 divided by zero, -1 and -2 deadlocked in
    # ThreadPool::WaitCount, and -3 scheduled an overlapping job; -5 and
    # -100 used to run by accident because abs(chunk_size) > nbatch
    for chunk_size in (0, -1, -2, -3, -5, -100):
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'chunk_size must be positive'
      ):
        rollout.rollout(
            model, data, initial_state, control, chunk_size=chunk_size
        )

      # skip_checks bypasses the Python wrapper, so this covers the same
      # argument reaching the C++ rollout unchecked
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'chunk_size must be positive'
      ):
        rollout.rollout(
            [model] * nbatch,
            data,
            initial_state,
            control,
            nstep=nstep,
            chunk_size=chunk_size,
            skip_checks=True,
        )

    # the argument is rejected even when the thread pool is not used
    single_data = mujoco.MjData(model)
    for chunk_size in (0, -1):
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'chunk_size must be positive'
      ):
        rollout.rollout(
            model, single_data, initial_state, control, chunk_size=chunk_size
        )

  def test_chunk_size(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    num_workers = 4
    nbatch = 10
    nstep = 5
    initial_state = randn(nbatch, nstate)
    control = randn(nbatch, nstep, model.nu)
    data = [mujoco.MjData(model) for _ in range(num_workers)]

    py_state, py_sensordata = py_rollout(
        model, mujoco.MjData(model), initial_state, control
    )
    for chunk_size in (1, 3, nbatch, nbatch + 1):
      state, sensordata = rollout.rollout(
          model, data, initial_state, control, chunk_size=chunk_size
      )
      np.testing.assert_array_equal(state, py_state)
      np.testing.assert_array_equal(sensordata, py_sensordata)

  def test_stateless(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    # step with a clean mjData
    initial_state = randn(nstate)
    control = randn(3, 3, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    # fill user fields with random values
    for attr in [
        'ctrl',
        'qfrc_applied',
        'xfrc_applied',
        'mocap_pos',
        'mocap_quat',
    ]:
      setattr(data, attr, randn(*getattr(data, attr).shape))

    # roll out again
    state2, sensordata2 = rollout.rollout(model, data, initial_state, control)

    # assert that we still get the same outputs
    np.testing.assert_array_equal(state, state2)
    np.testing.assert_array_equal(sensordata, sensordata2)

  def test_length_one_model_list(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = randn(nstate)
    control = randn(3, 3, model.nu)

    state, sensordata = rollout.rollout(model, data, initial_state, control)
    state2, sensordata2 = rollout.rollout([model], data, initial_state, control)

    # assert that we get same outputs
    np.testing.assert_array_equal(state, state2)
    np.testing.assert_array_equal(sensordata, sensordata2)

  def test_data_sizes(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = randn(nstate)
    control = randn(3, 3, model.nu)

    # Test passing empty lists for data
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'The list of data instances is empty'
    ):
      rollout.rollout(model, [], initial_state, control)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'The list of data instances is empty'
    ):
      with rollout.Rollout(nthread=0) as rollout_:
        rollout_.rollout(model, [], initial_state, control)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'The list of data instances is empty'
    ):
      with rollout.Rollout(nthread=1) as rollout_:
        rollout_.rollout(model, [], initial_state, control)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'The list of data instances is empty'
    ):
      with rollout.Rollout(nthread=2) as rollout_:
        rollout_.rollout(model, [], initial_state, control)

    # Test checking that len(data) equals nthread
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'More than one data instance passed but rollout is configured to run on'
        ' main thread',
    ):
      with rollout.Rollout(nthread=0) as rollout_:
        rollout_.rollout(
            model, [copy.copy(data) for _ in range(2)], initial_state, control
        )

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Length of data: 1 not equal to nthread: 2'
    ):
      with rollout.Rollout(nthread=2) as rollout_:
        rollout_.rollout(model, data, initial_state, control)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Length of data: 1 not equal to nthread: 2'
    ):
      with rollout.Rollout(nthread=2) as rollout_:
        rollout_.rollout(model, [data], initial_state, control)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Length of data: 3 not equal to nthread: 2'
    ):
      with rollout.Rollout(nthread=2) as rollout_:
        rollout_.rollout(
            model, [copy.copy(data) for _ in range(3)], initial_state, control
        )

  @absltest.skip(reason='Takes a long time to run')
  def test_large_state(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nthread = os.cpu_count()
    nbatch = nthread

    nstep = ((2**31) // (nstate * nbatch)) + 2
    assert nstep * nstate * nbatch > 2**31

    initial_state = randn(nbatch, nstate)
    rollout.rollout(
        model,
        [copy.copy(data) for _ in range(nthread)],
        initial_state,
        nstep=nstep,
    )


# -------------- Python implementation of rollout functionality ----------------


def get_state(model, data):
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  state = empty(nstate)
  mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  return state.reshape((1, nstate))


def step(
    model, data, state, control, control_spec=mujoco.mjtState.mjSTATE_CTRL
):
  if state is not None:
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  mujoco.mj_setState(model, data, control, control_spec)
  mujoco.mj_step(model, data)
  return (get_state(model, data), data.sensordata)


def one_rollout(
    model,
    data,
    initial_state,
    control,
    control_spec=mujoco.mjtState.mjSTATE_CTRL,
):
  nstep = control.shape[0]
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  state = empty(nstep, nstate)
  sensordata = empty(nstep, model.nsensordata)

  mujoco.mj_resetData(model, data)
  for t in range(nstep):
    state[t], sensordata[t] = step(
        model, data, initial_state if t == 0 else None, control[t], control_spec
    )
  return state, sensordata


def ensure_2d(arg):
  if arg is None:
    return None
  else:
    return np.ascontiguousarray(np.atleast_2d(arg), dtype=DTYPE)


def ensure_3d(arg):
  if arg is None:
    return None
  else:
    # np.atleast_3d adds both leading and trailing dims, we want only leading
    if arg.ndim == 0:
      arg = arg[np.newaxis, np.newaxis, np.newaxis, ...]
    elif arg.ndim == 1:
      arg = arg[np.newaxis, np.newaxis, ...]
    elif arg.ndim == 2:
      arg = arg[np.newaxis, ...]
    return np.ascontiguousarray(arg, dtype=DTYPE)


def py_rollout(
    model,
    data,
    initial_state,
    control,
    control_spec=mujoco.mjtState.mjSTATE_CTRL,
):
  initial_state = ensure_2d(initial_state)
  control = ensure_3d(control)
  nbatch = initial_state.shape[0]
  nstep = control.shape[1]

  if isinstance(model, mujoco.MjModel):
    model = [copy.copy(model) for _ in range(nbatch)]

  nstate = mujoco.mj_stateSize(model[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)

  state = empty(nbatch, nstep, nstate)
  sensordata = empty(nbatch, nstep, model[0].nsensordata)
  for r in range(nbatch):
    state_r, sensordata_r = one_rollout(
        model[r], data, initial_state[r], control[r], control_spec
    )
    state[r] = state_r
    sensordata[r] = sensordata_r
  return state, sensordata


if __name__ == '__main__':
  absltest.main()
