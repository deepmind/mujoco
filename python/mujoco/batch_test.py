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
"""mujoco.batch against MjData, and its errors, views and callbacks."""

import copy
import ctypes
import gc

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import batch
from mujoco import rollout
import numpy as np

NUM = mujoco.MJTNUM_DTYPE  # float64, or float32 in a single-precision build
# a floating-point type of another element size
OTHER = np.float32 if NUM == np.float64 else np.float64

XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 .05">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="mocap" mocap="true" pos="1 0 1">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="5"/>
    <general joint="hinge" dyntype="filter" dynprm="0.02" gainprm="5"
      biastype="affine" biasprm="0 -5 0"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <accelerometer site="tip"/>
  </sensor>
  <keyframe>
    <key qpos="0.3 0.5" ctrl="0.1 0.2" act="0.2"/>
  </keyframe>
</mujoco>
"""

SLEEP_XML = """
<mujoco>
  <option integrator="implicitfast" viscosity="10" sleep_tolerance="0.01">
    <flag sleep="enable" gravity="disable" constraint="disable"
      contact="disable"/>
  </option>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="box" size=".1 .2 .3" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
NSIM = 8


class BatchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model = mujoco.MjModel.from_xml_string(XML)

  @parameterized.parameters((1, False), (3, False), (3, True))
  def test_lockstep(self, nthread, persistent):
    m = self.model
    b = batch.Batch(m, NSIM, nthread, persistent)
    self.assertEqual(b.nthread, nthread)
    self.assertEqual(b.persistent, persistent)
    qpos, ctrl = b.bind('qpos'), b.bind('ctrl')
    sensordata, xpos = b.bind('sensordata'), b.bind('xpos')
    ref = [mujoco.MjData(m) for _ in range(NSIM)]
    rng = np.random.default_rng(0)
    for call in range(30):
      u = rng.uniform(-1, 1, (NSIM, m.nu))
      ctrl[:] = u
      for i, d in enumerate(ref):
        d.ctrl[:] = u[i]
      if call == 12:
        ids = np.array([1, 4])
        b.reset(ids, keyframe=0)
        for i in ids:
          mujoco.mj_resetDataKeyframe(m, ref[i], 0)
          mujoco.mj_forward(m, ref[i])
        ran = ids
      else:
        mask = np.arange(NSIM) % 3 != 2 if call % 5 == 4 else None
        b.step(mask, nstep=2)
        ran = np.arange(NSIM) if mask is None else np.flatnonzero(mask)
        for i in ran:
          for _ in range(2):
            mujoco.mj_step(m, ref[i])
      for i in ran:
        np.testing.assert_array_equal(qpos[i], ref[i].qpos)
        np.testing.assert_array_equal(sensordata[i], ref[i].sensordata)
        np.testing.assert_array_equal(xpos[i], ref[i].xpos)

  def test_bind_shapes(self):
    b = batch.Batch(self.model, NSIM)
    self.assertEqual(b.bind('time').shape, (NSIM,))
    self.assertEqual(b.bind('qpos').shape, (NSIM, self.model.nq))
    self.assertEqual(b.bind('mocap_quat').shape, (NSIM, 1, 4))
    self.assertEqual(b.bind('xmat').shape, (NSIM, self.model.nbody, 9))
    self.assertEqual(b.state.shape, (NSIM, b.nstate))
    with self.assertRaises(ValueError):
      b.bind('nope')
    with self.assertRaises(ValueError):
      b.status[0] = 1  # read-only

  def test_named(self):
    m = self.model
    b = batch.Batch(m, NSIM, 2)
    b.reset(keyframe=0)
    hinge = b.joint('hinge')
    self.assertEqual((hinge.id, hinge.name), (1, 'hinge'))
    hinge.qpos[:, 0] = np.arange(NSIM) * 0.1  # a view of the state rows
    b.actuator(0).ctrl[:, 0] = 0.5
    views = [
        (kind, key, attr, getattr(getattr(b, kind)(key), attr))
        for kind, key, attr in (
            ('joint', 'hinge', 'qpos'),
            ('joint', 'hinge', 'qacc'),
            ('body', 'pole', 'xpos'),
            ('body', 'pole', 'xquat'),
            ('site', 'tip', 'xmat'),
            ('sensor', 1, 'data'),
            ('actuator', 1, 'force'),
        )
    ]
    b.step()

    data = mujoco.MjData(m)
    for i in range(NSIM):
      mujoco.mj_resetDataKeyframe(m, data, 0)
      mujoco.mj_forward(m, data)
      data.qpos[1] = i * 0.1
      data.ctrl[0] = 0.5
      mujoco.mj_step(m, data)
      for kind, key, attr, got in views:
        expected = getattr(getattr(data, kind)(key), attr)
        np.testing.assert_array_equal(got[i], expected, f'{kind} {attr}')
    with self.assertRaises(AttributeError):
      _ = b.joint('hinge').nope
    with self.assertRaises(KeyError):
      b.body('nope')

  @parameterized.parameters(False, True)
  def test_step_stage(self, persistent):
    m = self.model
    b = batch.Batch(m, NSIM, 2, persistent)
    xpos, qpos = b.bind('xpos'), b.bind('qpos')
    b.bind('qvel')[:] = 1
    d = mujoco.MjData(m)

    def fresh(i):
      d.qpos[:] = qpos[i]
      mujoco.mj_kinematics(m, d)
      return np.array_equal(xpos[i], d.xpos)

    b.step(nstep=2, stage=mujoco.mjtStage.mjSTAGE_VEL)
    # positions of the new state
    self.assertTrue(all(fresh(i) for i in range(NSIM)))
    b.step(nstep=2)  # the default: those before the last substep, as mj_step
    self.assertFalse(any(fresh(i) for i in range(NSIM)))

  def test_reset_stage(self):
    m = self.model
    full, none = batch.Batch(m, NSIM, 2), batch.Batch(m, NSIM, 2)
    for b, stage in (
        (full, mujoco.mjtStage.mjSTAGE_ACC),
        (none, mujoco.mjtStage.mjSTAGE_NONE),
    ):
      b.step(nstep=3)
      b.reset(keyframe=0, stage=stage)
      b.qpos[:, 1] = np.linspace(0, 1, NSIM)  # set after the reset
      b.forward()
    np.testing.assert_array_equal(none.state, full.state)
    np.testing.assert_array_equal(none.xpos, full.xpos)

  def test_field_sets_and_attributes(self):
    m = self.model
    b = batch.Batch(m, NSIM)
    self.assertIn('xpos', batch.DATA_FIELDS)
    self.assertIn('qpos', batch.DATA_FIELDS)
    self.assertNotIn('contact', batch.DATA_FIELDS)  # not a fixed-size field
    self.assertIn('body_mass', batch.MODEL_FIELDS)
    self.assertIn('opt.gravity', batch.MODEL_FIELDS)
    self.assertNotIn('mesh_vert', batch.MODEL_FIELDS)  # shared, not expandable
    for name in batch.DATA_FIELDS:
      self.assertEqual(b.bind(name).shape[0], NSIM, name)
    for name in batch.MODEL_FIELDS:
      self.assertEqual(b.expand(name).shape[0], NSIM, name)

    # attribute access binds
    np.testing.assert_array_equal(b.xpos, b.bind('xpos'))
    b.ctrl = 0.5  # and writes through, as with MjData
    np.testing.assert_array_equal(b.bind('ctrl'), 0.5)
    b.qpos[:, 1] = 0.25
    np.testing.assert_array_equal(b.bind('qpos')[:, 1], 0.25)
    with self.assertRaises(AttributeError):
      b.nope  # pylint: disable=pointless-statement
    with self.assertRaises(AttributeError):
      b.contact  # pylint: disable=pointless-statement

  def test_derived_fields_are_read_only(self):
    b = batch.Batch(self.model, NSIM)
    with self.assertRaises(ValueError):
      b.xpos[0, 0, 0] = 1  # an output: a write would be lost at the next call
    with self.assertRaises(ValueError):
      b.body('pole').xpos = 0
    b.qpos[0, 0] = 1  # inputs stay writable

  @parameterized.parameters(False, True)
  def test_get_model_and_data(self, persistent):
    m = self.model
    b = batch.Batch(m, NSIM, 2, persistent)
    pole = m.body('pole').id
    b.expand('body_mass')[:, pole] = np.linspace(0.1, 1, NSIM)
    b.set_const()
    b.qpos[:, 1] = np.linspace(-1, 1, NSIM)
    xpos = b.xpos  # bound before the call that fills it
    b.forward()
    for sim in (0, 5):
      ms = b.get_model(sim)
      self.assertEqual(ms.body_mass[pole], b.expand('body_mass')[sim, pole])
      ds = b.get_data(sim)
      np.testing.assert_array_equal(ds.qpos, b.qpos[sim])
      mujoco.mj_forward(ms, ds)
      np.testing.assert_array_equal(ds.xpos, xpos[sim])
    # into given objects
    ms, ds = mujoco.MjModel.from_xml_string(XML), mujoco.MjData(m)
    self.assertIs(b.get_model(5, ms), ms)
    self.assertIs(b.get_data(5, ds), ds)
    with self.assertRaises(Exception):
      b.get_data(NSIM)

  def test_named_model_fields(self):
    m = self.model
    b = batch.Batch(m, NSIM)
    pole = b.body('pole')
    np.testing.assert_array_equal(
        pole.mass, np.full((NSIM, 1), m.body('pole').mass)
    )
    # per simulation, as expand()
    pole.mass = np.arange(1, NSIM + 1).reshape(NSIM, 1)
    np.testing.assert_array_equal(
        b.expand('body_mass')[:, pole.id], np.arange(1, NSIM + 1)
    )
    hinge = b.joint('hinge')  # joint attributes in dof arrays
    hinge.damping = 0.5
    np.testing.assert_array_equal(
        b.expand('dof_damping')[:, m.joint('hinge').dofadr[0]], 0.5
    )
    self.assertEqual(b.geom(0).size.shape, (NSIM, 3))
    # data attributes still bind
    np.testing.assert_array_equal(hinge.qpos, b.qpos[:, 1:2])

  def test_named_model_reads_do_not_expand(self):
    m = self.model
    b = batch.Batch(m, NSIM)
    # the model's value, read-only, and nothing expanded
    mass = b.body('pole').mass
    np.testing.assert_array_equal(mass, np.full((NSIM, 1), m.body('pole').mass))
    self.assertFalse(mass.flags.writeable)
    self.assertNotIn('body_mass', b._expanded)
    b.body('pole').mass = 2  # assignment expands
    np.testing.assert_array_equal(b.body('pole').mass, 2)
    self.assertTrue(b.body('pole').mass.flags.writeable)

  def test_named_reads_leave_derived_fields_derived(self):
    m = self.model
    b = batch.Batch(m, NSIM)
    pole = b.body('pole')
    pole.mass = np.arange(1, NSIM + 1).reshape(NSIM, 1)
    b.set_const()  # derives invweight0 per simulation
    invweight = pole.invweight0  # set_const's values, read-only
    self.assertFalse(invweight.flags.writeable)
    self.assertGreater(invweight[0, 0], invweight[-1, 0])
    before = invweight.copy()
    pole.mass = 2 * np.arange(1, NSIM + 1).reshape(NSIM, 1)
    b.set_const()  # still derived after the read: recomputed
    self.assertFalse(np.array_equal(pole.invweight0, before))

  def test_expanded_reads_without_expanding(self):
    b = batch.Batch(self.model, NSIM)
    self.assertIsNone(b.expanded('body_mass'))
    self.assertIsNone(b.expanded('body_invweight0'))
    self.assertIsNone(b.expanded('opt.gravity'))

    b.expand('body_mass')[:, 1] = np.arange(1, NSIM + 1)
    b.expand('opt.gravity')[:, 2] = -np.arange(1, NSIM + 1)
    b.set_const()

    mass = b.expanded('body_mass')
    self.assertIsNotNone(mass)
    self.assertFalse(mass.flags.writeable)
    np.testing.assert_array_equal(mass[:, 1], np.arange(1, NSIM + 1))

    invweight = b.expanded('body_invweight0')
    self.assertIsNotNone(invweight)
    self.assertFalse(invweight.flags.writeable)
    self.assertNotIn('body_invweight0', b._expanded)

    gravity = b.expanded('opt.gravity')
    self.assertIsNotNone(gravity)
    self.assertFalse(gravity.flags.writeable)
    np.testing.assert_array_equal(gravity[:, 2], -np.arange(1, NSIM + 1))

    with self.assertRaises(ValueError):
      b.expanded('qpos')
    with self.assertRaises(ValueError):
      b.expanded('nope')

  def test_named_structural_fields(self):
    m = self.model
    b = batch.Batch(m, NSIM)
    qposadr = b.joint('hinge').qposadr  # the same in every simulation
    np.testing.assert_array_equal(
        qposadr, np.full((NSIM, 1), m.joint('hinge').qposadr)
    )
    self.assertFalse(qposadr.flags.writeable)
    with self.assertRaisesRegex(ValueError, 'structural'):
      b.joint('hinge').qposadr = 0
    self.assertNotIn('jnt_qposadr', b._expanded)

  def test_named_contype_and_pre_expanded_type_fields(self):
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1" contype="1"/>
        <body name="ball" pos="0 0 .5">
          <freejoint/>
          <geom name="ball" size=".1" contype="1"/>
        </body>
      </worldbody>
      <equality>
        <connect name="conn" body1="ball" anchor="0 0 0" active="false"/>
      </equality>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    b = batch.Batch(m, NSIM)

    b.geom('floor').contype = np.arange(NSIM, dtype=np.int32).reshape(NSIM, 1)
    self.assertIn('geom_contype', b._expanded)
    np.testing.assert_array_equal(
        b.geom('floor').contype[:, 0], np.arange(NSIM, dtype=np.int32)
    )
    self.assertFalse(
        batch._structural('flex_contype', np.zeros(1, dtype=np.int32))
    )

    eq_type = b.expand('eq_type')
    eq_type[:, 0] = m.eq_type[0]
    self.assertTrue(b.equality('conn').type.flags.writeable)
    np.testing.assert_array_equal(b.equality('conn').type[:, 0], eq_type[:, 0])

  def test_named_empty_slices(self):
    b = batch.Batch(self.model, NSIM)
    # nuser_body is 0: empty, as model.body('pole').user
    user = b.body('pole').user
    self.assertEqual(user.shape, (NSIM, *self.model.body('pole').user.shape))
    self.assertEqual(user.size, 0)

  def test_named_model_only_views(self):
    m = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <asset>
        <material name="red" rgba="1 0 0 1"/>
      </asset>
      <custom>
        <numeric name="gains" data="1 2 3"/>
      </custom>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="ball" pos="0 0 .1">
          <freejoint/>
          <geom name="ball" size=".05" material="red"/>
        </body>
      </worldbody>
      <contact>
        <pair name="touch" geom1="floor" geom2="ball"/>
      </contact>
      <equality>
        <weld name="hold" body1="ball"/>
      </equality>
      <keyframe>
        <key name="up" qpos="0 0 1 1 0 0 0"/>
      </keyframe>
    </mujoco>
    """)
    b = batch.Batch(m, NSIM)
    for kind, key, attr in (
        ('equality', 'hold', 'solref'),
        ('pair', 'touch', 'margin'),
        ('material', 'red', 'rgba'),
        ('numeric', 'gains', 'data'),
        ('keyframe', 'up', 'qpos'),
    ):
      named = getattr(b, kind)(key)
      self.assertEqual((named.id, named.name), (0, key))
      expected = getattr(getattr(m, kind)(key), attr)
      np.testing.assert_array_equal(
          getattr(named, attr),
          np.broadcast_to(expected, (NSIM, *expected.shape)),
      )
    solref = np.stack(
        [np.linspace(0.01, 0.08, NSIM), np.ones(NSIM)], axis=1
    ).astype(NUM)
    b.equality('hold').solref = solref  # assignment expands
    np.testing.assert_array_equal(b.expand('eq_solref')[:, 0], solref)
    np.testing.assert_array_equal(b.get_model(3).eq_solref[0], solref[3])
    with self.assertRaisesRegex(AttributeError, 'not a field of a equality'):
      _ = b.equality('hold').xpos  # no MjData attributes
    with self.assertRaises(KeyError):
      b.pair('nope')

  def test_named_attribute_in_no_field(self):
    b = batch.Batch(self.model, NSIM)
    # MjData's moment is a dense copy of sparse rows, not a slice of a field
    with self.assertRaisesRegex(AttributeError, 'not a slice of a field'):
      _ = b.actuator(1).moment

  def test_named_mimo_actuators(self):
    m = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size=".1"/>
          <body>
            <joint name="ball" type="ball"/>
            <geom size=".1" pos=".2 0 0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="before" joint="slide" gear="2"/>
        <orientation name="orient" joint="ball" kp="1" input="quat"/>
        <motor name="after" joint="slide" gear="3"/>
      </actuator>
    </mujoco>
    """)
    b = batch.Batch(m, NSIM)
    ctrl, force = b.bind('ctrl'), b.bind('actuator_force')
    ctrl[:] = np.arange(NSIM * m.nu).reshape(NSIM, m.nu)
    b.forward()
    # slices at the actuators' addresses, with all of their controls and outputs
    np.testing.assert_array_equal(b.actuator('orient').ctrl, ctrl[:, 1:5])
    np.testing.assert_array_equal(b.actuator('after').ctrl, ctrl[:, 5:6])
    np.testing.assert_array_equal(b.actuator('orient').force, force[:, 1:4])
    np.testing.assert_array_equal(b.actuator('after').force, force[:, 4:5])
    self.assertEqual(b.actuator('orient').gear.shape, (NSIM, 3, 6))
    self.assertEqual(b.actuator('after').gear.shape, (NSIM, 6))
    b.actuator('after').gear = np.arange(NSIM * 6).reshape(NSIM, 6)
    np.testing.assert_array_equal(
        b.expand('actuator_gear')[:, 4], np.arange(NSIM * 6).reshape(NSIM, 6)
    )

  def test_reset_keyframe_per_simulation(self):
    xml = XML.replace(
        '</keyframe>',
        '<key qpos="-0.3 -0.5" ctrl="0.2 0.1" act="0.1"/></keyframe>',
    )
    m = mujoco.MjModel.from_xml_string(xml)
    b = batch.Batch(m, NSIM, 2)
    b.step(nstep=5)
    keys = np.array([0, -1, 1, 1, 0, -1, 0, 1])
    b.reset(keyframe=keys)
    d = mujoco.MjData(m)
    for i, key in enumerate(keys):
      if key < 0:
        mujoco.mj_resetData(m, d)
      else:
        mujoco.mj_resetDataKeyframe(m, d, key)
      mujoco.mj_forward(m, d)
      np.testing.assert_array_equal(b.qpos[i], d.qpos)
    b.reset([2, 3], keyframe=1)  # one keyframe for every simulation of the call
    np.testing.assert_array_equal(b.qpos[2], b.qpos[3])
    with self.assertRaises(Exception):
      b.reset(keyframe=2)

  def test_named_assignment(self):
    b = batch.Batch(self.model, NSIM)
    b.joint('hinge').qpos = 0.5  # writes through, as with MjData
    np.testing.assert_array_equal(b.bind('qpos')[:, 1], 0.5)
    b.actuator(1).ctrl = np.arange(NSIM).reshape(NSIM, 1)
    np.testing.assert_array_equal(b.bind('ctrl')[:, 1], np.arange(NSIM))
    with self.assertRaises(AttributeError):
      b.joint('hinge').id = 0
    with self.assertRaises(AttributeError):
      b.joint('hinge').nope = 0

  def test_expand_and_set_const(self):
    m = self.model
    pole = m.body('pole').id
    b = batch.Batch(m, NSIM, 4)
    mass = b.expand('body_mass')
    mass[:, pole] = np.linspace(0.1, 1.0, NSIM)
    b.set_const()
    np.testing.assert_array_equal(
        b.expand('body_subtreemass')[:, pole], mass[:, pole]
    )
    b.bind('ctrl')[:, 0] = 1.0
    b.step(nstep=10)
    for i in range(NSIM):
      mi = mujoco.MjModel.from_xml_string(XML)
      mi.body_mass[pole] = mass[i, pole]
      di = mujoco.MjData(mi)
      mujoco.mj_setConst(mi, di)
      mujoco.mj_resetData(mi, di)
      di.ctrl[0] = 1.0
      for _ in range(10):
        mujoco.mj_step(mi, di)
      np.testing.assert_array_equal(b.bind('qvel')[i], di.qvel)

  def test_set_const_refits_body_hierarchies(self):
    m = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <asset>
        <mesh name="tet" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
      </asset>
      <worldbody>
        <body>
          <freejoint/>
          <geom name="ball" size=".05" pos=".1 0 0"/>
          <geom size=".05" pos="-.1 0 0"/>
          <geom type="mesh" mesh="tet"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    ball = m.geom('ball').id
    b = batch.Batch(m, NSIM, 2)
    b.geom('ball').size = np.linspace(0.05, 0.2, NSIM).reshape(NSIM, 1)
    b.set_const()
    self.assertEqual(b.expand('bvh_aabb').shape, (NSIM, m.mesh_bvhadr[0], 6))
    size = b.expand('geom_size')
    for i in range(NSIM):
      mi = copy.copy(m)
      mi.geom_size[ball] = size[i, ball]
      mujoco.mj_setConst(mi, mujoco.MjData(mi))
      np.testing.assert_array_equal(b.get_model(i).bvh_aabb, mi.bvh_aabb)

  def test_expand_bvh_without_body_hierarchies(self):
    # a mesh, but no body with geoms, so with a hierarchy: no rows per
    # simulation
    m = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <asset>
        <mesh name="tet" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
      </asset>
      <worldbody>
        <body>
          <freejoint/>
          <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    self.assertGreater(m.nbvhstatic, 0)
    self.assertTrue(np.all(m.body_bvhadr < 0), m.body_bvhadr)
    b = batch.Batch(m, NSIM)
    self.assertEqual(b.expand('bvh_aabb').shape, (NSIM, 0, 6))
    b.set_const()  # reads the expanded values back
    self.assertEqual(b.expanded('bvh_aabb').shape, (NSIM, 0, 6))

  def test_expand_option(self):
    m = self.model
    b = batch.Batch(m, NSIM, 2)
    gravity = b.expand('opt.gravity')
    self.assertEqual(gravity.shape, (NSIM, 3))
    np.testing.assert_array_equal(gravity, np.tile(m.opt.gravity, (NSIM, 1)))
    iterations = b.expand('opt.iterations')
    self.assertEqual(iterations.shape, (NSIM,))
    self.assertEqual(iterations.dtype, np.int32)
    gravity[:, 2] = -np.arange(NSIM)
    b.step(nstep=5)
    for i in range(NSIM):
      mi = mujoco.MjModel.from_xml_string(XML)
      mi.opt.gravity[2] = -i
      di = mujoco.MjData(mi)
      for _ in range(5):
        mujoco.mj_step(mi, di)
      np.testing.assert_array_equal(b.bind('qpos')[i], di.qpos)

  def test_errors_name_every_simulation(self):
    xml = XML.replace(
        '</worldbody>',
        '</worldbody><equality><weld body1="mocap" body2="pole"/></equality>',
    )
    m = mujoco.MjModel.from_xml_string(xml)
    b = batch.Batch(m, NSIM, 2)
    eq_type = b.expand('eq_type')
    eq_type[[3, 6]] = 99  # an unknown constraint type: mj_step raises
    time = b.bind('time')
    with self.assertRaisesRegex(
        batch.SimulationError, r'2 of 8 simulations \(3, 6\); simulation 3'
    ) as e:
      b.step()
    np.testing.assert_array_equal(e.exception.ids, [3, 6])
    np.testing.assert_array_equal(np.flatnonzero(b.status), [3, 6])
    self.assertNotEmpty(b.error(3))
    self.assertEmpty(b.error(0))
    self.assertEqual(time[3], 0)  # it kept its state
    self.assertTrue(np.all(np.delete(time, [3, 6]) > 0))  # the others ran
    eq_type[[3, 6]] = m.eq_type[0]
    b.step()
    self.assertFalse(np.any(b.status))

  def test_set_const_errors_name_its_simulations(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <joint type="slide"/>
            <geom size=".1" mass="1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
      </mujoco>""")
    b = batch.Batch(m, 3)
    b.expand('body_mass')[:, 1] = [0, 2, 0]  # zero mass: mj_setConst raises
    with self.assertRaises(batch.SimulationError) as e:
      b.set_const([0, 1])
    np.testing.assert_array_equal(e.exception.ids, [0])
    self.assertEqual(b.status[2], 0)  # not recomputed

  def test_error_messages(self):
    b = batch.Batch(self.model, NSIM)
    cases = (
        (lambda: b.bind('body_mass'), "use expand\\('body_mass'\\)"),
        (lambda: b.bind('timestep'), "use expand\\('opt.timestep'\\)"),
        (lambda: b.bind('efc_force'), 'arena'),
        (lambda: b.bind('qpso'), "did you mean 'qpos'"),
        (lambda: b.expand('xpos'), "use bind\\('xpos'\\)"),
        (lambda: b.expand('timestep'), "use expand\\('opt.timestep'\\)"),
        (lambda: b.expand('mesh_vert'), 'asset data'),
        (lambda: b.expand('body_mas'), "did you mean 'body_mass'"),
        (lambda: b.rollout(1, record=('qpso',)), "did you mean 'qpos'"),
        (lambda: b.step([0, NSIM]), f'in \\[0, {NSIM}\\), got {NSIM}'),
        (lambda: b.step([2, 1]), 'got 2 before 1'),
        (lambda: b.get_data(NSIM), f'got {NSIM}'),
    )
    for call, message in cases:
      with self.subTest(message):
        with self.assertRaisesRegex((ValueError, IndexError), message):
          call()
    with self.assertRaisesRegex(ValueError, 'computed by every call'):
      b.xpos = 0
    with self.assertRaisesRegex(ValueError, 'computed by every call'):
      b.body('pole').xpos = 0
    with self.assertRaisesRegex(AttributeError, "did you mean 'qpos'"):
      _ = b.joint('hinge').qpso
    with self.assertRaisesRegex(AttributeError, "use expand\\('body_mass'\\)"):
      _ = b.body_mass
    with self.assertRaisesRegex(ValueError, r'shape \(8, 3\), got \(8, 2\)'):
      b._b.jac(
          None, np.zeros(NSIM, np.int32), np.zeros((NSIM, 2), NUM), None, None
      )

  def test_bad_ids(self):
    b = batch.Batch(self.model, NSIM)
    time = b.bind('time')
    # rejected before narrowing to int32, rather than wrapped or truncated to 0
    for bad in (
        [2**32],
        [0.9],
        [[0]],
        [NSIM],
        [-1],
        np.ones(NSIM + 1, bool),
        [3, 1],
        [2, 2],
        np.array([2, 1], np.uint32),  # unsigned: a difference would wrap around
    ):
      with self.assertRaises(ValueError, msg=str(bad)):
        b.step(bad)
    np.testing.assert_array_equal(time, 0)
    b.step([])  # an empty selection runs nothing
    np.testing.assert_array_equal(time, 0)
    b.step()  # still usable
    self.assertTrue(np.all(time > 0))

  def test_mask_ids(self):
    # an all-true mask selects every simulation, as None does
    self.assertIsNone(batch._ids(np.ones(NSIM, bool), NSIM))  # pylint: disable=protected-access
    np.testing.assert_array_equal(
        batch._ids(np.arange(NSIM) != 2, NSIM),  # pylint: disable=protected-access
        np.delete(np.arange(NSIM), 2),
    )

  def test_caller_models(self):
    m = self.model
    pole = m.body('pole').id
    variants = []
    for i in range(NSIM):
      variant = copy.copy(m)
      variant.body_mass[pole] = 0.1 * (i + 1)
      mujoco.mj_setConst(variant, mujoco.MjData(variant))
      variants.append(variant)
    b = batch.Batch(m, NSIM, 3)
    self.assertIsNone(b.models)
    b.set_models(variants)
    self.assertEqual(b.models, tuple(variants))
    b.ctrl[:, 0] = 0.5
    b.step(nstep=10)
    for i in range(NSIM):
      d = mujoco.MjData(m)
      d.ctrl[0] = 0.5
      for _ in range(10):
        mujoco.mj_step(variants[i], d)
      np.testing.assert_array_equal(b.qpos[i], d.qpos)
      self.assertEqual(
          b.get_model(i).body_mass[pole], variants[i].body_mass[pole]
      )
    # named views read each simulation's model, and cannot write it
    np.testing.assert_array_equal(
        b.body('pole').mass[:, 0], [v.body_mass[pole] for v in variants]
    )
    with self.assertRaisesRegex(ValueError, 'set_models'):
      b.body('pole').mass = 1
    with self.assertRaisesRegex(mujoco.FatalError, 'caller'):
      b.expand('body_mass')
    with self.assertRaisesRegex(mujoco.FatalError, 'caller'):
      b.set_const()
    with self.assertRaisesRegex(ValueError, 'per simulation'):
      b.set_models(variants[:-1])
    b.set_models(variants[0])  # one model for every simulation
    self.assertIs(b.models, variants[0])
    self.assertEqual(b.get_model(NSIM - 1).body_mass[pole], 0.1)
    b.set_models(None)
    self.assertIsNone(b.models)
    b.expand('body_mass')
    with self.assertRaisesRegex(mujoco.FatalError, 'per-simulation'):
      b.set_models(variants)

  def test_count_outputs(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .1">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
          <body pos="0 0 .35">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>""")
    b = batch.Batch(m, NSIM, 2)
    ncon, nefc = b.bind('ncon'), b.nefc
    self.assertEqual(ncon.shape, (NSIM,))
    self.assertEqual(ncon.dtype, np.int32)
    qpos = b.bind('qpos')
    qpos[1::2, 2] += 1  # lift the box of every other simulation, apart
    qpos[1::2, 9] += 3
    state = b.state.copy()
    out = b.rollout(2, record=('ncon',))
    self.assertEqual(out['ncon'].shape, (NSIM, 2))
    np.testing.assert_array_equal(out['ncon'][:, -1], ncon)
    d = mujoco.MjData(m)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for i in range(NSIM):
      mujoco.mj_setState(m, d, state[i], spec)
      mujoco.mj_step(m, d)
      mujoco.mj_step(m, d)
      self.assertEqual(ncon[i], d.ncon, i)
      self.assertEqual(nefc[i], d.nefc, i)
    self.assertGreater(ncon[0], 0)
    self.assertEqual(ncon[1], 0)
    with self.assertRaisesRegex(ValueError, 'scalars bind\\(\\) accepts: nA'):
      b.bind('pstack')

  def test_sleep_needs_persistent(self):
    m = mujoco.MjModel.from_xml_string(SLEEP_XML)
    with self.assertRaisesRegex(ValueError, 'persistent'):
      batch.Batch(m, NSIM)
    b = batch.Batch(m, NSIM, 3, persistent=True)
    qvel = b.bind('qvel')
    ref = [mujoco.MjData(m) for _ in range(NSIM)]
    for i, d in enumerate(ref):
      qvel[i] = d.qvel[:] = 0.1 * (i + 1)
    for _ in range(50):
      b.step(nstep=10)
      for d in ref:
        for _ in range(10):
          mujoco.mj_step(m, d)
    self.assertTrue(any(d.ntree_awake < m.ntree for d in ref))
    for i, d in enumerate(ref):
      np.testing.assert_array_equal(qvel[i], d.qvel)
      np.testing.assert_array_equal(b.bind('qpos')[i], d.qpos)

  @parameterized.parameters(1, 3)
  def test_rollout_matches_mujoco_rollout(self, nthread):
    m = self.model
    nstep, ids = 7, np.array([0, 2, 5])
    b = batch.Batch(m, NSIM, nthread)
    rng = np.random.default_rng(1)
    control = rng.uniform(-1, 1, (len(ids), nstep, m.nu))
    full = mujoco.mjtState.mjSTATE_FULLPHYSICS
    out = b.rollout(nstep, control, record=(full, 'sensordata'), ids=ids)

    nstate = mujoco.mj_stateSize(m, full)
    initial = np.empty((len(ids), nstate), NUM)
    d = mujoco.MjData(m)
    mujoco.mj_getState(m, d, initial[0], full)
    initial[:] = initial[0]
    state, sensordata = rollout.rollout(m, d, initial, control)
    np.testing.assert_array_equal(out[full], state)
    np.testing.assert_array_equal(out['sensordata'], sensordata)
    # the rows continue from the last state; the others did not move
    np.testing.assert_array_equal(b.bind('time')[ids], state[:, -1, 0])
    self.assertEqual(b.bind('time')[1], 0)

  def test_rollout_records(self):
    b = batch.Batch(self.model, NSIM)
    out = b.rollout(3, record=('qpos', 'xmat', mujoco.mjtState.mjSTATE_QVEL))
    self.assertEqual(out['qpos'].shape, (NSIM, 3, self.model.nq))
    self.assertEqual(out['xmat'].shape, (NSIM, 3, self.model.nbody, 9))
    self.assertEqual(
        out[mujoco.mjtState.mjSTATE_QVEL].shape, (NSIM, 3, self.model.nv)
    )
    with self.assertRaises(ValueError):
      b.rollout(3, record=('nope',))

  def test_rollout_out(self):
    b = batch.Batch(self.model, NSIM)
    qvel = mujoco.mjtState.mjSTATE_QVEL
    qpos = np.empty((NSIM, 3, self.model.nq), NUM)
    out = b.rollout(3, record=('qpos', qvel), out={'qpos': qpos})
    self.assertIs(out['qpos'], qpos)  # recorded into the given array
    self.assertEqual(out[qvel].shape, (NSIM, 3, self.model.nv))
    np.testing.assert_array_equal(qpos[:, -1], b.bind('qpos'))
    with self.assertRaises(ValueError):
      b.rollout(
          3,
          record=('qpos',),
          out={'qpos': np.empty((NSIM, 2, self.model.nq), NUM)},
      )
    with self.assertRaises(ValueError):
      b.rollout(3, record=('qpos',), out={'qpos': qpos.astype(OTHER)})

  def test_rollout_checks_buffers(self):
    b = batch.Batch(self.model, NSIM)
    m = self.model
    ctrl = int(mujoco.mjtState.mjSTATE_CTRL)
    good = np.empty((NSIM, 3, m.nq), NUM)
    b._b.rollout(None, 3, None, ctrl, [('qpos', 0, good)])
    for record in (
        ('qpos', 0, np.empty((NSIM, 1, m.nq), NUM)),  # too short
        ('qpos', 0, np.empty((NSIM, 3, m.nq), OTHER)),  # wrong element size
        (
            'qpos',
            0,
            np.empty(
                (NSIM, 3, m.nq), np.int64 if NUM == np.float64 else np.int32
            ),
        ),
        ('qpos', 0, np.empty((NSIM, 3, m.nq), np.dtype(NUM).newbyteorder())),
        ('solver_niter', 0, np.empty((NSIM, 3, mujoco.mjNISLAND), np.float32)),
        (
            None,
            int(mujoco.mjtState.mjSTATE_QVEL),
            np.empty((NSIM, 3, m.nv + 1), NUM),
        ),
        (None, -1, good),  # invalid spec
        ('nope', 0, good),
    ):
      with self.assertRaises(ValueError):
        b._b.rollout(None, 3, None, ctrl, [record])
    with self.assertRaises(ValueError):  # control too short
      b._b.rollout(None, 3, np.zeros((NSIM, 2, m.nu), NUM), ctrl, [])
    # not an array: its buffer would be a temporary
    with self.assertRaises(ValueError):
      b._b.rollout(None, 3, None, ctrl, [('qpos', 0, good.tolist())])

  def test_private_binding_does_not_convert(self):
    m = self.model
    b = batch.Batch(m, NSIM)
    pole = m.body('pole').id
    body = np.full(NSIM, pole, np.int32)
    point = np.zeros((NSIM, 3), NUM)
    jac = np.zeros((NSIM, 3, m.nv), NUM)
    b._b.jac(None, body, point, jac, None)
    for args in (
        # an output of another type
        (None, body, point, jac.astype(OTHER), None),
        (
            None,
            body,
            point,
            np.zeros((NSIM, m.nv, 3), NUM).transpose(0, 2, 1),
            None,
        ),
        (None, body.astype(np.int64), point, jac, None),
        (np.arange(NSIM, dtype=np.int64), body, point, jac, None),  # int64 ids
        ([0, 1], body[:2], point[:2], jac[:2], None),  # a list of ids
    ):
      with self.assertRaises(TypeError):
        b._b.jac(*args)

  def test_rollout_records_fixed_size_fields(self):
    m = copy.copy(self.model)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    b = batch.Batch(m, NSIM, 2)
    b.bind('qvel')[:, 1] = np.linspace(-1, 1, NSIM)
    qvel = b.bind('qvel').copy()
    out = b.rollout(3, record=('energy', 'solver_niter'))
    self.assertEqual(out['solver_niter'].dtype, np.int32)
    for i in range(NSIM):
      d = mujoco.MjData(m)
      d.qvel = qvel[i]
      for k in range(3):
        mujoco.mj_step(m, d)
        np.testing.assert_array_equal(out['energy'][i, k], d.energy)
        np.testing.assert_array_equal(out['solver_niter'][i, k], d.solver_niter)

  def test_integer_arguments_checked(self):
    m = self.model
    b = batch.Batch(m, NSIM, 2)
    pole = m.body('pole').id
    for key in (2**32, 0.5, m.nkey, -2):
      with self.assertRaisesRegex(ValueError, 'keyframe'):
        b.reset(keyframe=key)
    for body in (2**32, 0.9, m.nbody, -1):
      with self.assertRaisesRegex(ValueError, 'body'):
        b.jac(body, np.zeros(3))
    for exclude in (2**32, 0.5, m.nbody, -2):
      with self.assertRaisesRegex(ValueError, 'bodyexclude'):
        b.ray(np.zeros(3), [0, 0, 1], bodyexclude=exclude)
    b.jac(np.full(NSIM, pole, np.int64), np.zeros(3))
    b.ray(np.zeros(3), [0, 0, 1], bodyexclude=np.int64(pole))

    # the private binding checks the ids it indexes with
    jac = np.zeros((1, 3, m.nv), NUM)
    for bad in (-1, NSIM):
      with self.assertRaises(ValueError):
        b._b.jac(
            np.array([bad], np.int32),
            np.array([pole], np.int32),
            np.zeros((1, 3), NUM),
            jac,
            None,
        )

  def test_python_callbacks_rejected(self):
    b = batch.Batch(self.model, NSIM, 2)
    b.step()  # validates the installed callbacks
    previous = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(lambda m, d: None)
    try:
      with self.assertRaisesRegex(ValueError, 'mjcb_control'):
        b.step()
    finally:
      mujoco.set_mjcb_control(previous)
    b.step()

    # a C function, here a ctypes thunk, gets the batch's model and data
    calls = []
    cfunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
        lambda m, d: calls.append(1)
    )
    previous = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(cfunc)
    try:
      b.step()
    finally:
      mujoco.set_mjcb_control(previous)
    self.assertLen(calls, NSIM)

  def test_callbacks_checked_on_construction(self):
    # trees that start asleep make mj_makeData run mj_forward and the callbacks
    m = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <option>
        <flag sleep="enable"/>
      </option>
      <worldbody>
        <body sleep="init">
          <freejoint/>
          <geom size=".1"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    previous = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(lambda m, d: None)
    try:
      with self.assertRaisesRegex(ValueError, 'mjcb_control'):
        batch.Batch(m, 1, 1, persistent=True)
    finally:
      mujoco.set_mjcb_control(previous)

    # a C callback runs on the workers, which must not wait for the GIL
    calls = []
    cfunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
        lambda m, d: calls.append(1)
    )
    previous = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(cfunc)
    try:
      batch.Batch(m, 4, 2, persistent=True)
    finally:
      mujoco.set_mjcb_control(previous)
    self.assertGreaterEqual(len(calls), 4)

  def test_views_outlive_the_batch(self):
    b = batch.Batch(self.model, NSIM, 2)
    b.step(nstep=3)
    qpos, xpos = b.qpos, b.xpos
    expected = qpos.copy(), xpos.copy()
    del b
    gc.collect()
    np.testing.assert_array_equal(qpos, expected[0])
    np.testing.assert_array_equal(xpos, expected[1])
    qpos[:] = 0  # still writable memory

  def test_eq_active_dtypes(self):
    xml = XML.replace(
        '</worldbody>',
        '</worldbody><equality><weld body1="mocap" body2="pole"/></equality>',
    )
    m = mujoco.MjModel.from_xml_string(xml)
    b = batch.Batch(m, NSIM)
    self.assertEqual(b.eq_active.dtype, NUM)  # rows hold mjtNum
    out = b.rollout(2, record=('eq_active',))
    self.assertEqual(out['eq_active'].dtype, mujoco.MjData(m).eq_active.dtype)

  def test_named_multidimensional_slices(self):
    xml = XML.replace(
        '<body name="mocap"',
        '<body name="ball" pos="0 1 1"><freejoint name="free"/><geom'
        ' size=".1"/></body><body name="mocap"',
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    b = batch.Batch(m, NSIM)
    accel = b.sensor(1)  # the accelerometer, three-dimensional
    # bound before the call fills it
    self.assertEqual(accel.data.shape, (NSIM, 3))
    b.forward()
    mujoco.mj_forward(m, d)
    self.assertEqual(b.joint('free').qpos.shape, (NSIM, 7))
    self.assertEqual(b.joint('free').qvel.shape, (NSIM, 6))
    np.testing.assert_array_equal(b.joint('free').qpos[3], d.joint('free').qpos)
    np.testing.assert_array_equal(accel.data[0], d.sensor(1).data)

  @parameterized.parameters(False, True)
  def test_jac_and_ray_match_mujoco(self, persistent):
    m = self.model
    b = batch.Batch(m, NSIM, 2, persistent=persistent)
    b.bind('qpos')[:, 1] = np.linspace(-1, 1, NSIM)
    qpos = b.bind('qpos').copy()
    pole = m.body('pole').id
    jacp, jacr = b.jac(pole, np.zeros(3))
    dist, geomid = b.ray([0.0, -1.0, 0.3], [0.0, 1.0, 0.0])
    # the state is unchanged
    np.testing.assert_array_equal(b.bind('qpos'), qpos)
    d = mujoco.MjData(m)
    for i in range(NSIM):
      d.qpos = qpos[i]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      jp, jr = np.zeros((3, m.nv), NUM), np.zeros((3, m.nv), NUM)
      mujoco.mj_jac(m, d, jp, jr, np.zeros(3, NUM), pole)
      np.testing.assert_array_equal(jacp[i], jp)
      np.testing.assert_array_equal(jacr[i], jr)
      gid = np.zeros(1, np.int32)
      ref = mujoco.mj_ray(
          m,
          d,
          np.array([0.0, -1.0, 0.3], NUM),
          np.array([0.0, 1.0, 0.0], NUM),
          None,
          1,
          -1,
          gid,
      )
      self.assertEqual(dist[i, 0], ref)
      self.assertEqual(geomid[i, 0], gid[0])

  def test_warning_counters_and_rollout_stop(self):
    m = self.model
    b = batch.Batch(m, NSIM, 1)
    self.assertEqual(b.warning.shape, (NSIM, mujoco.mjtWarning.mjNWARNING, 2))
    # simulation 3 resets with a warning at its next step
    b.bind('qvel')[3, 0] = np.inf
    spec = mujoco.mjtState.mjSTATE_TIME.value
    out = {'time': b.rollout(4, record=(spec,))[spec][..., 0]}
    bad = int(mujoco.mjtWarning.mjWARN_BADQVEL)
    self.assertEqual(b.warning[3, bad, 1], 1)  # (lastinfo, number)
    self.assertEqual(b.warning[0, bad, 1], 0)
    # stopped, repeated
    np.testing.assert_array_equal(out['time'][3], out['time'][3, 0])
    self.assertGreater(out['time'][0, -1], out['time'][0, 0])

  def test_jac(self):
    m = self.model
    b = batch.Batch(m, NSIM, 3)
    qpos = b.bind('qpos')
    qpos[:, 1] = np.linspace(-1, 1, NSIM)
    pole = m.body('pole').id
    point = np.random.default_rng(2).uniform(-1, 1, (NSIM, 3))
    jacp, jacr = b.jac(pole, point)
    d = mujoco.MjData(m)
    for i in range(NSIM):
      d.qpos[:] = qpos[i]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      ep, er = np.zeros((3, m.nv), NUM), np.zeros((3, m.nv), NUM)
      mujoco.mj_jac(m, d, ep, er, point[i], pole)
      np.testing.assert_array_equal(jacp[i], ep)
      np.testing.assert_array_equal(jacr[i], er)
    sub = np.array([1, 6])
    jacp_sub, _ = b.jac(pole, point[sub], ids=sub)
    np.testing.assert_array_equal(jacp_sub, jacp[sub])

    # a body per simulation
    cart = m.body('cart').id
    bodies = np.where(np.arange(NSIM) % 2, pole, cart)
    jacp_mixed, _ = b.jac(bodies, point)
    for i in range(NSIM):
      d.qpos[:] = qpos[i]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      ep = np.zeros((3, m.nv), NUM)
      mujoco.mj_jac(m, d, ep, None, point[i], bodies[i])
      np.testing.assert_array_equal(jacp_mixed[i], ep)
    with self.assertRaises(ValueError):
      b.jac(m.nbody, point)

  def test_ray(self):
    m = self.model
    b = batch.Batch(m, NSIM, 3)
    b.bind('qpos')[:, 0] = np.linspace(-0.5, 0.5, NSIM)
    nray = 5
    pnt = np.tile([0.0, -1.0, 0.12], (NSIM, nray, 1))
    pnt[:, :, 0] = np.linspace(-0.6, 0.6, nray)
    vec = np.tile([0.0, 1.0, 0.0], (NSIM, nray, 1))
    dist, geomid = b.ray(pnt, vec)
    d = mujoco.MjData(m)
    hits = 0
    for i in range(NSIM):
      d.qpos[:] = b.bind('qpos')[i]
      mujoco.mj_kinematics(m, d)
      for k in range(nray):
        gid = np.zeros(1, np.int32)
        expected = mujoco.mj_ray(
            m, d, pnt[i, k], vec[i, k], None, 1, -1, gid, None
        )
        self.assertEqual(dist[i, k], expected)
        self.assertEqual(geomid[i, k], gid[0])
        hits += gid[0] >= 0
    self.assertGreater(hits, 0)

    # normals, and a body excluded per ray
    pole = m.body('pole').id
    exclude = np.where(np.arange(nray) % 2, pole, -1)
    dist2, geomid2, normal = b.ray(pnt, vec, bodyexclude=exclude, normal=True)
    for i in range(NSIM):
      d.qpos[:] = b.bind('qpos')[i]
      mujoco.mj_kinematics(m, d)
      for k in range(nray):
        gid, nrm = np.zeros(1, np.int32), np.zeros(3, NUM)
        expected = mujoco.mj_ray(
            m, d, pnt[i, k], vec[i, k], None, 1, exclude[k], gid, nrm
        )
        self.assertEqual(dist2[i, k], expected)
        self.assertEqual(geomid2[i, k], gid[0])
        np.testing.assert_array_equal(normal[i, k], nrm)

    # rays broadcast: one ray, and the same rays in every simulation
    one, _ = b.ray(pnt[0, 0], vec[0, 0])
    self.assertEqual(one.shape, (NSIM, 1))
    np.testing.assert_array_equal(one[:, 0], dist[:, 0])
    same, _ = b.ray(pnt[0], vec[0])
    self.assertEqual(same.shape, (NSIM, nray))
    # pnt, vec are the same in every simulation
    np.testing.assert_array_equal(same, dist)
    sub = np.array([2, 5])
    dist_sub, geomid_sub = b.ray(pnt[sub], vec[sub], ids=sub)
    np.testing.assert_array_equal(dist_sub, dist[sub])
    np.testing.assert_array_equal(geomid_sub, geomid[sub])


if __name__ == '__main__':
  absltest.main()
