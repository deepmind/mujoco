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
"""Tests for sensor functions."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import test_util
from mujoco.mjx._src.types import ConeType
import numpy as np

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr)


class SensorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(314)

  @parameterized.product(
      filename=['sensor/model.xml', 'sensor/sensor.xml'],
      cone_type=list(ConeType),
  )
  def test_sensor(self, filename, cone_type):
    """Tests MJX sensor functions match MuJoCo sensor functions."""
    m = test_util.load_test_file(filename)
    m.opt.cone = cone_type
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = 0.1 * np.random.random(m.nv)
    # apply external forces
    d.xfrc_applied = 0.1 * np.random.random(d.xfrc_applied.shape)
    # apply control for activation dynamics
    d.ctrl = np.clip(
        0.1 * np.random.random(m.nu),
        m.actuator_ctrlrange[:, 0],
        m.actuator_ctrlrange[:, 1],
    )
    mujoco.mj_step(m, d, 100)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d).tree_replace({
        'sensordata': jp.zeros_like(d.sensordata),
        '_impl.subtree_linvel': jp.zeros_like(d.subtree_linvel),
        '_impl.subtree_angmom': jp.zeros_like(d.subtree_angmom),
        '_impl.cacc': jp.zeros_like(d.cacc),
        '_impl.cfrc_int': jp.zeros_like(d.cfrc_int),
        '_impl.cfrc_ext': jp.zeros_like(d.cfrc_ext),
    })
    dx = jax.jit(mjx.sensor_pos)(mx, dx)
    dx = jax.jit(mjx.sensor_vel)(mx, dx)
    dx = jax.jit(mjx.sensor_acc)(mx, dx)

    _assert_eq(d.sensordata, dx.sensordata, 'sensordata')

  def test_touch_sensor_nested_vmap(self):
    """Tests touch sensors compose under nested vmap."""
    m = test_util.load_test_file('sensor/sensor.xml')
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 10)
    mx = mjx.put_model(m)
    dx = mjx.forward(mx, mjx.put_data(m, d))

    def sensor_fn(offset):
      data = jax.tree.map(
          lambda x: x + offset * jp.zeros_like(x)
          if isinstance(x, jax.Array) and jp.issubdtype(x.dtype, jp.floating)
          else x,
          dx,
      )
      return mjx.sensor_acc(mx, data).sensordata

    offsets = jp.zeros((2, 4))
    actual = jax.jit(jax.vmap(jax.vmap(sensor_fn)))(offsets)
    expected = np.asarray(dx.sensordata)
    _assert_eq(actual, np.broadcast_to(expected, actual.shape), 'sensordata')

  def test_acc_sensors_dofless_body(self):
    """Tests acceleration sensors on static and mocap bodies are zero."""
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <site name="world" pos=".1 0 0"/>
          <body name="mocap" mocap="true" pos="0 0 1">
            <site name="mocap"/>
          </body>
          <body name="static" pos="0 1 1">
            <geom size=".1"/>
            <site name="static"/>
          </body>
          <body pos="1 0 1">
            <joint type="hinge" axis="0 1 0"/>
            <geom size=".1" pos=".2 0 0"/>
            <site name="hinge" pos=".2 0 0"/>
          </body>
        </worldbody>
        <sensor>
          <accelerometer site="world"/>
          <accelerometer site="mocap"/>
          <accelerometer site="static"/>
          <accelerometer site="hinge"/>
          <framelinacc objtype="site" objname="world"/>
          <framelinacc objtype="body" objname="mocap"/>
          <frameangacc objtype="xbody" objname="static"/>
          <framelinacc objtype="site" objname="hinge"/>
        </sensor>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = jax.jit(mjx.forward)(mx, mjx.put_data(m, d))
    _assert_eq(d.sensordata, dx.sensordata, 'sensordata')
    np.testing.assert_array_equal(d.sensordata[:9], 0.0)

  def test_disable_sensor(self):
    """Tests disabling sensor."""
    m = test_util.load_test_file('sensor/sensor.xml')
    # disable sensors
    m.opt.disableflags = m.opt.disableflags | mjx.DisableBit.SENSOR
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    # random sensor values
    random_sensor = jp.array(np.random.random(dx.sensordata.shape))
    dx = dx.replace(sensordata=random_sensor)
    # call sensor functions
    dx = jax.jit(mjx.forward)(mx, dx)
    # sensor values
    _assert_eq(random_sensor, dx.sensordata, 'sensordata')

  @parameterized.parameters(
      'type="sphere" size=".1"',
      'type="sphere" size=".05" margin=".045"',
      'type="capsule" size=".1 .1" euler="0 89 89"',
      'type="box" size=".1 .1 .1" euler=".05 .075 .1"',
  )
  def test_sensor_contact(self, geom):
    """Tests contact sensor."""

    field = ['found', 'force', 'torque', 'dist', 'pos', 'normal', 'tangent']
    datas = itertools.chain.from_iterable(
        itertools.combinations(field, i) for i in range(len(field) + 1)
    )
    datas = list(datas)

    contact_sensors = ''
    for num in [1, 2, 4, 5]:
      for data in datas:
        data = ' '.join(data)
        for reduce in ['mindist', 'maxforce']:
          for match in [
              '',
              'geom1="plane"',
              'geom1="geom1"',
              'geom1="sphere2"',
              'geom2="plane"',
              'geom2="geom1"',
              'geom2="sphere2"',
              'geom1="plane" geom2="geom1"',
              'geom1="geom1" geom2="plane"',
              'geom1="plane" geom2="sphere2"',
              'geom1="geom1" geom2="sphere2"',
          ]:
            contact_sensors += (
                f'<contact {match} num="{num}" data="{data}"'
                f' reduce="{reduce}"/>\n'
            )

    _MJCF = f"""
      <mujoco>
        <compiler angle="degree"/>
        <worldbody>
          <geom name="plane" type="plane" size="10 10 .001"/>
          <body>
            <geom name="geom1" {geom}/>
            <joint type="slide" axis="0 0 1"/>
          </body>
          <body>
            <geom name="sphere2" type="sphere" size=".1"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
        <sensor>
          {contact_sensors}
        </sensor>
        <keyframe>
          <key qpos=".09 1"/>
        </keyframe>
      </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(_MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_forward(m, d)
    dx = mjx.forward(mx, dx)

    _assert_eq(dx.sensordata, d.sensordata, 'sensordata')

  def test_sensor_contact_velocity(self):
    """Tests contact sensor linvel and angvel with nonzero velocities."""
    contact_sensors = ''
    for data in ['linvel', 'angvel', 'found linvel angvel', 'dist linvel']:
      for match in [
          '',
          'geom1="plane"',
          'geom1="sphere1"',
          'geom1="sphere1" geom2="sphere2"',
          'geom1="sphere2" geom2="sphere1"',
      ]:
        for reduce in ['mindist', 'maxforce']:
          contact_sensors += (
              f'<contact {match} num="2" data="{data}" reduce="{reduce}"/>\n'
          )

    m = mujoco.MjModel.from_xml_string(f"""
      <mujoco>
        <worldbody>
          <geom name="plane" type="plane" size="10 10 .001"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom name="sphere1" type="sphere" size=".1"/>
          </body>
          <body pos=".02 0 .27">
            <freejoint/>
            <geom name="sphere2" type="sphere" size=".1"/>
          </body>
        </worldbody>
        <sensor>
          {contact_sensors}
        </sensor>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    d.qvel = [.3, -.2, .1, .5, -.4, .7, -.1, .4, -.3, -.6, .2, .8]

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_forward(m, d)
    dx = jax.jit(mjx.forward)(mx, dx)

    self.assertTrue(np.any(d.sensordata != 0))
    _assert_eq(dx.sensordata, d.sensordata, 'sensordata')

  def test_unsupported_sensor(self):
    """Tests unsupported sensor raises error."""
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <joint type="hinge"/>
            <geom name="geom0" size="0.1"/>
            <body>
              <joint type="hinge"/>
              <geom name="geom1" size="0.25"/>
            </body>
          </body>
        </worldbody>
        <sensor>
          <distance name="distance" geom1="geom0" geom2="geom1"/>
        </sensor>
      </mujoco>
    """)
    with self.assertRaises(NotImplementedError):
      mjx.put_model(m)


if __name__ == '__main__':
  absltest.main()
