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
"""Tests for io functions."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import io as mjx_io
from mujoco.mjx._src import test_util
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import ConeType
from mujoco.mjx._src.types import EnableBit
from mujoco.mjx._src.types import Impl
from mujoco.mjx._src.types import IntegratorType
from mujoco.mjx._src.types import JacobianType
# pylint: enable=g-importing-member
import mujoco.mjx.warp as mjxw
from mujoco.mjx.warp import types as mjxw_types
from mujoco.mjx.warp import warp as wp  # pylint: disable=g-importing-member
import numpy as np


_MULTIPLE_CONVEX_OBJECTS = """
  <mujoco>
    <option timestep="0.001" jacobian="dense"/>
    <default>
      <geom solref=".006 1"/>
    </default>
    <asset>
      <mesh name="box" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1" scale="1 1 .1"/>
      <mesh name="boxoid" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 .5 -1 -1 2" scale=".3 .2 .1"/>
      <mesh name="pentaprism" vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0 1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1" scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom type="plane" pos="0 0 -.5" size="3 3 .01"/>
      <geom type="mesh" mesh="box" pos="0 0 -.15" euler="3 7 30"/>
      <body pos="-.3 -.3 .3">
        <freejoint/>
        <geom type="mesh" mesh="boxoid" rgba=".8 0 0 1" euler="3 5 -130"/>
      </body>
      <body pos=".3 .3 0.3">
        <freejoint/>
        <geom type="box" euler="3 5 -80" size=".3 .2 .1" rgba="0 .8 0 1"/>
      </body>
      <body pos=".3 .3 .6">
        <freejoint/>
        <geom type="mesh" mesh="pentaprism" rgba="0 0 .8 1"/>
      </body>
      <body pos=".6 -.3 .3">
        <joint name="joint" axis="1 0 0" type="hinge" range="-45 45"/>
        <geom type="capsule" size=".2 .05" rgba=".6 0 .6 1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="joint"/>
    </actuator>
  </mujoco>
"""

_MULTIPLE_CONSTRAINTS = """
  <mujoco>
    <worldbody>
      <geom type="plane" size="3 3 .01" condim="6"/>
      <body name="cap1" pos="-.3 -.3 .2">
        <freejoint/>
        <geom type="capsule" size=".2 .05"/>
        <body name="cap2" pos=".6 -.3 .3">
          <joint name="joint1" axis="0 1 0" type="hinge" range="-45 45"/>
          <joint name="joint2" axis="1 0 0" type="hinge" range="-0.001 0.001"/>
          <geom type="capsule" size=".2 .05"/>
          <site pos="-0.214 -0.078 0" quat="0.664 0.664 -0.242 -0.242"/>
        </body>
      </body>
    </worldbody>
    <equality>
      <connect body1="cap2" anchor="0 0 1"/>
    </equality>
    <tendon>
      <fixed name="tendon_1" limited="false" stiffness=".1" damping=".2">
        <joint joint="joint1" coef=".1"/>
        <joint joint="joint2" coef="-.2"/>
      </fixed>
    </tendon>
  </mujoco>
"""

_SIMPLE_BODY = """
  <mujoco>
    <worldbody>
      <!-- nC < nM because this body has an inertia-aligned free joint -->
      <body name="simplebody">
        <freejoint/>
        <geom type="sphere" size="0.01"/>
      </body>
    </worldbody>
  </mujoco>
"""

_LIGHTS = """
  <mujoco>
    <worldbody>
      <light name="inactive" active="false"/>
      <light name="active" active="true"/>
    </worldbody>
  </mujoco>
"""


def _get_name_from_path(path: jax.tree_util.KeyPath) -> str:
  """Returns a flattened name from a jax.tree_util.KeyPath."""
  if any(isinstance(p, jax.tree_util.SequenceKey) for p in path):
    is_seq_key = [isinstance(p, jax.tree_util.SequenceKey) for p in path]
    path = path[: is_seq_key.index(True)]
  assert all(isinstance(p, jax.tree_util.GetAttrKey) for p in path)
  path = [p for p in path if p.name != '_impl']
  attr = '__'.join(p.name for p in path)
  return attr


class ModelIOTest(parameterized.TestCase):
  """IO tests for mjx.Model."""

  def setUp(self):
    super().setUp()
    if mjxw.WARP_INSTALLED:
      self.tempdir = tempfile.TemporaryDirectory()
      wp.config.kernel_cache_dir = self.tempdir.name

  @parameterized.product(
      xml=(_MULTIPLE_CONVEX_OBJECTS, _MULTIPLE_CONSTRAINTS),
      impl=('jax', 'warp', 'cpp'),
  )
  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_put_model(self, xml, impl):
    if impl == 'warp' and not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if impl == 'warp' and not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device available.')

    m = mujoco.MjModel.from_xml_string(xml)
    mx = mjx.put_model(m, impl=impl)

    def assert_not_weak_type(x):
      if isinstance(x, jax.Array):
        assert not x.weak_type

    jax.tree_util.tree_map(assert_not_weak_type, mx)
    self.assertEqual(mx.nq, m.nq)
    self.assertEqual(mx.nv, m.nv)
    self.assertEqual(mx.nu, m.nu)
    self.assertEqual(mx.na, m.na)
    self.assertEqual(mx.nbody, m.nbody)
    self.assertEqual(mx.njnt, m.njnt)
    self.assertEqual(mx.ngeom, m.ngeom)
    self.assertEqual(mx.nmesh, m.nmesh)
    self.assertEqual(mx.npair, m.npair)
    self.assertEqual(mx.nexclude, m.nexclude)
    self.assertEqual(mx.neq, m.neq)
    self.assertEqual(mx.nnumeric, m.nnumeric)
    self.assertEqual(mx.nM, m.nM)
    self.assertAlmostEqual(mx.opt.timestep, m.opt.timestep)

    if impl == 'jax':
      # fields restricted to MuJoCo should not be populated
      self.assertFalse(hasattr(mx, 'bvh_aabb'))

    elif impl == 'warp':
      # Fields private to Warp backend impl are populated.
      self.assertTrue(hasattr(mx._impl, 'nxn_geom_pair'))
    elif impl == 'cpp':
      self.assertTrue(hasattr(mx._impl, 'pointer_lo'))
      self.assertTrue(hasattr(mx._impl, 'pointer_hi'))

    np.testing.assert_allclose(mx.body_parentid, m.body_parentid)
    np.testing.assert_allclose(mx.geom_type, m.geom_type)
    np.testing.assert_allclose(mx.geom_bodyid, m.geom_bodyid)
    np.testing.assert_almost_equal(mx.geom_solref, m.geom_solref)
    np.testing.assert_almost_equal(mx.geom_pos, m.geom_pos)

    np.testing.assert_allclose(mx.jnt_type, m.jnt_type)
    np.testing.assert_allclose(mx.jnt_dofadr, m.jnt_dofadr)
    np.testing.assert_allclose(mx.jnt_bodyid, m.jnt_bodyid)
    np.testing.assert_allclose(mx.jnt_limited, m.jnt_limited)
    np.testing.assert_almost_equal(mx.jnt_axis, m.jnt_axis)

    np.testing.assert_allclose(mx.actuator_trntype, m.actuator_trntype)
    np.testing.assert_allclose(mx.actuator_dyntype, m.actuator_dyntype)
    np.testing.assert_allclose(mx.actuator_gaintype, m.actuator_gaintype)
    np.testing.assert_allclose(mx.actuator_biastype, m.actuator_biastype)
    np.testing.assert_allclose(mx.actuator_trnid, m.actuator_trnid)

    np.testing.assert_equal(mx.mesh_faceadr, m.mesh_faceadr)
    np.testing.assert_equal(mx.mesh_facenum, m.mesh_facenum)

    np.testing.assert_equal(mx.tendon_type, m.tendon_type)
    np.testing.assert_equal(mx.wrap_type, m.wrap_type)
    np.testing.assert_equal(mx.wrap_objid, m.wrap_objid)
    np.testing.assert_almost_equal(mx.wrap_prm, m.wrap_prm)

  @parameterized.parameters('jax', 'warp', 'cpp')
  def test_put_model_light_active(self, impl):
    """Tests that light_active is copied to the public Model field."""
    if impl == 'warp' and not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')

    m = mujoco.MjModel.from_xml_string(_LIGHTS)
    mx = mjx.put_model(m, impl=impl)

    self.assertIn('light_active', [f.name for f in mx.fields()])
    np.testing.assert_array_equal(
        np.asarray(mx.light_active),
        m.light_active.astype(np.asarray(mx.light_active).dtype),
    )

  def test_fluid_params(self):
    """Test that has_fluid_params is set when fluid params are present."""
    m = mjx.put_model(
        mujoco.MjModel.from_xml_string(
            '<mujoco><option viscosity="3.0"/><worldbody/></mujoco>'
        ),
        impl='jax',
    )
    self.assertTrue(m.opt._impl.has_fluid_params)

  def test_implicit_not_implemented(self):
    """Test that MJX guards against models with unimplemented features."""

    with self.assertRaises(NotImplementedError):
      mjx.put_model(
          mujoco.MjModel.from_xml_string(
              '<mujoco><option integrator="implicit"/><worldbody/></mujoco>'
          )
      )

  def test_discrete_integrator(self):
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><option integrator="discrete"/><worldbody/></mujoco>'
    )
    mx = mjx.put_model(m)
    self.assertEqual(mx.opt.integrator, IntegratorType.DISCRETE)

  def test_ipc_enableflag(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody/>
      </mujoco>
    """)
    mx = mjx.put_model(m)
    self.assertTrue(mx.opt.enableflags & EnableBit.IPC)

  def test_pgs_not_implemented(self):
    with self.assertRaises(NotImplementedError):
      mjx.put_model(
          mujoco.MjModel.from_xml_string(
              '<mujoco><option solver="PGS"/><worldbody/></mujoco>'
          )
      )

  def test_margin_gap_mesh_not_implemented(self):
    with self.assertRaises(NotImplementedError):
      mjx.put_model(
          mujoco.MjModel.from_xml_string("""
        <mujoco>
          <asset>
            <mesh name="box" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1" scale="1 1 1"/>
          </asset>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="mesh" mesh="box" margin="0.3"/>
            </body>
            <body>
              <freejoint/>
              <geom size="0.05"/>
            </body>
          </worldbody>
        </mujoco>"""),
          impl='jax',
      )

  def test_implicitfast_fluid_not_implemented(self):
    with self.assertRaises(NotImplementedError):
      mjx.put_model(
          mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option viscosity="3.0" integrator="implicitfast"/>
          <worldbody/>
        </mujoco>"""),
          impl='jax',
      )

  def test_wrap_inside(self):
    m = test_util.load_test_file('tendon/wrap_sidesite.xml')
    mx0 = mjx.put_model(m, impl='jax')
    np.testing.assert_equal(
        mx0._impl.is_wrap_inside,
        np.array([1, 0, 1, 0, 1, 1, 0]),
    )
    m.site_pos[2] = m.site_pos[1]
    mx1 = mjx.put_model(m, impl='jax')
    np.testing.assert_equal(
        mx1._impl.is_wrap_inside,
        np.array([0, 0, 1, 0, 1, 0, 0]),
    )

  @parameterized.parameters(
      '<contact site="site"/>',
      '<contact reduce="netforce"/>',
      '<contact body1="body"/>',
      '<contact body2="body"/>',
      '<contact body1="body" body2="body"/>'
      '<contact subtree1="body"/>',
      '<contact subtree2="body"/>',
      '<contact subtree1="body" subtree2="body"/>',
  )
  def test_contact_sensor_jax(self, contact_sensor):
    m = mujoco.MjModel.from_xml_string(f"""
      <mujoco>
        <worldbody>
          <site name="site"/>
          <geom name="plane" type="plane" size="10 10 .001"/>
          <body name="body">
            <geom type="sphere" size=".1"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
        <sensor>
          {contact_sensor}
        </sensor>
      </mujoco>
    """)
    with self.assertRaises(NotImplementedError):
      mjx.put_model(m, impl='jax')

  def test_put_model_warp_has_expected_shapes(self):
    """Tests that put_model produces expected shapes for MuJoCo Warp."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device available.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    mx = mjx.put_model(m, impl='warp')

    def check_ndim(path, x):
      k = _get_name_from_path(path)
      if k not in mjxw_types._NDIM['Model']:
        return
      is_batched = mjxw_types._BATCH_DIM['Model'][k]
      expected_ndim = mjxw_types._NDIM['Model'][k] - is_batched
      if not hasattr(x, 'ndim'):
        return
      msg = f'Field {k} has ndim {x.ndim} but expected {expected_ndim}'
      self.assertEqual(x.ndim, expected_ndim, msg)

    _ = jax.tree.map_with_path(check_ndim, mx)

  def test_put_model_warp_batch_sizes(self):
    """Tests put_model can add an nworld axis to selected Warp model fields."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if not hasattr(mjxw_types.GraphMode, 'WARP'):
      self.skipTest('Warp JAX FFI graph modes unavailable.')

    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <asset>
          <texture name="red" type="2d" builtin="flat" width="4" height="4"
            rgb1="1 0 0" rgb2="1 0 0"/>
          <material name="mat" texture="red" rgba="0.5 0.6 0.7 1"/>
        </asset>
        <worldbody>
          <geom type="sphere" size="0.1" material="mat"/>
        </worldbody>
      </mujoco>
    """)

    nworld = 3
    mx = mjx.put_model(m, impl='warp', batch_sizes={'mat_texid': nworld})

    self.assertEqual(mx.mat_texid.shape, (nworld,) + m.mat_texid.shape)
    self.assertEqual(mx.geom_pos.shape, (m.ngeom, 3))
    self.assertEqual(mx.geom_size.shape, (m.ngeom, 3))
    self.assertEqual(mx.mat_rgba.shape, (m.nmat, 4))
    self.assertEqual(mx.geom_type.shape, (m.ngeom,))
    self.assertEqual(mx.geom_dataid.shape, (m.ngeom,))

    np.testing.assert_array_equal(
        np.asarray(mx.mat_texid), np.repeat(m.mat_texid[None], nworld, axis=0)
    )
    np.testing.assert_allclose(np.asarray(mx.geom_pos), m.geom_pos)
    np.testing.assert_allclose(np.asarray(mx.geom_size), m.geom_size)
    np.testing.assert_allclose(np.asarray(mx.mat_rgba), m.mat_rgba)

  @parameterized.parameters('JAX', 'WARP', None)
  def test_put_model_warp_graph_mode(self, mode: str | None):
    """Tests that put_model accepts graph_mode parameter."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device available.')

    if mode is None:
      graph_mode = None
    else:
      graph_mode = getattr(mjxw_types.GraphMode, mode)

    m = mujoco.MjModel.from_xml_string(_SIMPLE_BODY)
    mx = mjx.put_model(m, impl='warp', graph_mode=graph_mode)

    expected = graph_mode or mjxw_types.GraphMode.WARP
    self.assertEqual(mx.opt._impl.graph_mode, expected)

  def test_unsupported_contact_types(self):
    """Tests that unsupported contact types raise an error."""
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <asset>
          <mesh name="box" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1" scale="1 1 .1"/>
        </asset>
        <worldbody>
        <body name="meshbox">
          <freejoint/>
          <geom type="mesh" mesh="box" pos="0 0 -.15" euler="3 7 30"/>
        </body>
        <body name="cylinder">
          <freejoint/>
          <geom type="cylinder" size="1.0 0.1"/>
        </body>
        </worldbody>
      </mujoco>
    """)

    with self.assertRaises(ValueError):
      mjx.make_data(m, impl='jax')


class DataIOTest(parameterized.TestCase):
  """IO tests for mjx.Data."""

  def setUp(self):
    super().setUp()
    if mjxw.WARP_INSTALLED:
      self.tempdir = tempfile.TemporaryDirectory()
      wp.config.kernel_cache_dir = self.tempdir.name

  @parameterized.parameters('jax', 'cpp')
  def test_make_data(self, impl: str):
    """Test that make_data returns the correct shapes."""
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONVEX_OBJECTS)
    d = mjx.make_data(m, impl=impl)

    nq = 22
    nbody = 5
    ncon = 46
    nm = 64
    nv = 19
    nefc = 185

    # Check public Data fields that exist for all impls
    self.assertEqual(d.qpos.shape, (nq,))
    self.assertEqual(d.qvel.shape, (nv,))
    self.assertEqual(d.act.shape, (0,))
    self.assertEqual(d.qacc_warmstart.shape, (nv,))
    self.assertEqual(d.ctrl.shape, (1,))
    self.assertEqual(d.qfrc_applied.shape, (nv,))
    self.assertEqual(d.xfrc_applied.shape, (nbody, 6))
    self.assertEqual(d.eq_active.shape, (0,))
    self.assertEqual(d.qacc.shape, (nv,))
    self.assertEqual(d.act_dot.shape, (0,))
    self.assertEqual(d.xpos.shape, (nbody, 3))
    self.assertEqual(d.xquat.shape, (nbody, 4))
    self.assertEqual(d.xmat.shape, (nbody, 3, 3))
    self.assertEqual(d.xipos.shape, (nbody, 3))
    self.assertEqual(d.ximat.shape, (nbody, 3, 3))
    self.assertEqual(d.xanchor.shape, (4, 3))
    self.assertEqual(d.xaxis.shape, (4, 3))
    self.assertEqual(d.geom_xpos.shape, (6, 3))
    self.assertEqual(d.geom_xmat.shape, (6, 3, 3))
    self.assertEqual(d.subtree_com.shape, (nbody, 3))
    self.assertEqual(d.cdof.shape, (nv, 6))
    self.assertEqual(d.actuator_length.shape, (1,))

    if impl == 'cpp':
      self.assertTrue(hasattr(d._impl, 'pointer_lo'))
      self.assertTrue(hasattr(d._impl, 'pointer_hi'))
      return  # cpp does not populate other _impl fields

    self.assertEqual(d._impl.nefc, nefc)
    self.assertEqual(d._impl.cinert.shape, (nbody, 10))
    self.assertEqual(d._impl.crb.shape, (nbody, 10))
    if impl == 'jax':
      self.assertEqual(d._impl.actuator_moment.shape, (1, nv))

    self.assertEqual(d._impl.contact.dist.shape, (ncon,))
    self.assertEqual(d._impl.contact.pos.shape, (ncon, 3))
    self.assertEqual(d._impl.contact.frame.shape, (ncon, 3, 3))
    self.assertEqual(d._impl.contact.solref.shape, (ncon, 2))
    self.assertEqual(d._impl.contact.solimp.shape, (ncon, 5))
    self.assertEqual(d._impl.contact.geom1.shape, (ncon,))
    self.assertEqual(d._impl.contact.geom2.shape, (ncon,))
    self.assertEqual(d._impl.efc_J.shape, (nefc, nv))
    self.assertEqual(d._impl.efc_frictionloss.shape, (nefc,))
    self.assertEqual(d._impl.efc_D.shape, (nefc,))
    self.assertEqual(d._impl.actuator_velocity.shape, (1,))
    self.assertEqual(d.cvel.shape, (nbody, 6))
    self.assertEqual(d.cdof_dot.shape, (nv, 6))
    self.assertEqual(d.qfrc_bias.shape, (nv,))
    self.assertEqual(d.qfrc_passive.shape, (nv,))
    self.assertEqual(d._impl.efc_aref.shape, (nefc,))
    self.assertEqual(d.qfrc_actuator.shape, (nv,))
    self.assertEqual(d.qfrc_smooth.shape, (nv,))
    self.assertEqual(d.qacc_smooth.shape, (nv,))
    self.assertEqual(d.qfrc_constraint.shape, (nv,))
    self.assertEqual(d.qfrc_inverse.shape, (nv,))
    self.assertEqual(d._impl.efc_force.shape, (nefc,))

    if impl == 'jax':
      self.assertEqual(d._impl.M.shape, (nv, nv))
      self.assertEqual(d._impl.qLD.shape, (nv, nv))
      self.assertEqual(d._impl.qLDiagInv.shape, (0,))

    # test sparse
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mjx.make_data(m, impl=impl)
    self.assertEqual(d._impl.M.shape, (m.nC,))
    self.assertEqual(d._impl.qLD.shape, (m.nC,))
    self.assertEqual(d._impl.qLDiagInv.shape, (nv,))


  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_make_data_warp(self):
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device.')
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONVEX_OBJECTS)
    d = mjx.make_data(m, impl='warp', naconmax=9, naccdmax=7, njmax=23, nvmax=8)
    self.assertEqual(d._impl.contact__dist.shape[0], 9)
    self.assertEqual(d._impl.naccdmax, 7)
    self.assertEqual(d._impl.efc__pos.shape[0], 23)
    self.assertEqual(d._impl.nvmax, 8)

  def test_put_data_warp(self):
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device.')
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONVEX_OBJECTS)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d, impl='warp', naccdmax=7, nvmax=8)
    self.assertEqual(dx._impl.naccdmax, 7)
    self.assertEqual(dx._impl.nvmax, 8)

  @parameterized.parameters('jax', 'cpp', 'warp')
  def test_put_data(self, impl: str):
    """Test that put_data puts the correct data for dense and sparse."""
    if impl == 'warp':
      if not mjxw.WARP_INSTALLED:
        self.skipTest('Warp is not installed.')
      if not mjx_io.has_cuda_gpu_device():
        self.skipTest('No CUDA GPU device.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d, impl=impl)

    # check a few fields
    np.testing.assert_allclose(dx.qpos, d.qpos)
    np.testing.assert_allclose(dx.xpos, d.xpos)
    np.testing.assert_allclose(dx.cvel, d.cvel)
    np.testing.assert_allclose(dx.cdof_dot, d.cdof_dot)

    # check that there are no weak types
    self.assertFalse(
        any(
            jax.tree_util.tree_flatten(
                jax.tree_util.tree_map(lambda x: x.weak_type, dx)
            )[0]
        )
    )

    # xmat, ximat, geom_xmat are all shape transformed
    np.testing.assert_allclose(dx.xmat.reshape((-1, 9)), d.xmat)
    np.testing.assert_allclose(dx.ximat.reshape((-1, 9)), d.ximat)
    np.testing.assert_allclose(dx.geom_xmat.reshape((-1, 9)), d.geom_xmat)
    np.testing.assert_allclose(dx.site_xmat.reshape((-1, 9)), d.site_xmat)

    # tendon length is correct
    np.testing.assert_allclose(dx.ten_length, d.ten_length)

    if impl == 'jax':
      # check that qM is transformed properly
      qm = np.zeros((m.nv, m.nv), dtype=np.float64)
      mujoco.mju_sym2dense(qm, d.M, m.M_rownnz, m.M_rowadr, m.M_colind)
      np.testing.assert_allclose(qm, mjx.full_m(mjx.put_model(m), dx))

    elif impl == 'cpp':
      self.assertTrue(hasattr(dx._impl, 'pointer_lo'))
      self.assertTrue(hasattr(dx._impl, 'pointer_hi'))
      return  # cpp does not populate other fields in _impl
    elif impl == 'warp':
      qm = np.zeros((m.nv, m.nv), dtype=np.float64)
      mujoco.mju_sym2dense(qm, d.M, m.M_rownnz, m.M_rowadr, m.M_colind)
      warp_M = np.zeros((m.nv, m.nv))
      mujoco.mju_sym2dense(
          warp_M,
          np.array(dx._impl.M),
          m.M_rownnz,
          m.M_rowadr,
          m.M_colind,
      )
      np.testing.assert_allclose(warp_M, qm)
      # TODO(taylorhowell): test efc__J
      np.testing.assert_allclose(dx._impl.efc__aref[:3], d.efc_aref[:3])

    # tendon impl data is correct
    np.testing.assert_equal(dx._impl.ten_wrapadr, np.zeros((1,)))
    np.testing.assert_equal(dx._impl.ten_wrapnum, np.zeros((1,)))
    np.testing.assert_equal(dx._impl.wrap_obj, np.zeros((2, 2)))
    np.testing.assert_equal(dx._impl.wrap_xpos, np.zeros((2, 6)))

    if impl == 'warp':
      return

    # 4 contacts, 2 for each capsule against the plane
    self.assertEqual(dx._impl.contact.dist.shape, (4,))
    self.assertEqual(d.ncon, 1)  # however only 1 contact in this step
    np.testing.assert_allclose(dx._impl.contact.dist[0], d.contact.dist[0])
    self.assertTrue((dx._impl.contact.dist[1:] > 0).all())
    self.assertEqual(dx._impl.contact.frame.shape, (4, 3, 3))
    np.testing.assert_allclose(
        dx._impl.contact.frame[0].reshape(9), d.contact.frame[0]
    )
    np.testing.assert_allclose(dx._impl.contact.frame[1:], 0)

    # efc_ are also shape transformed and padded
    self.assertEqual(dx._impl.efc_J.shape, (45, 8))  # nefc, nv
    d_efc_j = d.efc_J.reshape((-1, 8))
    np.testing.assert_allclose(dx._impl.efc_J[:3], d_efc_j[:3])  # connect eq
    np.testing.assert_allclose(
        dx._impl.efc_J[3], d_efc_j[3]
    )  # one active limit
    np.testing.assert_allclose(dx._impl.efc_J[4], 0)  # one inactive limit
    np.testing.assert_allclose(dx._impl.efc_J[5:15], d_efc_j[4:14])  # contact
    np.testing.assert_allclose(dx._impl.efc_J[15:], 0)  # no contact

    # check another efc_ too
    self.assertEqual(dx._impl.efc_aref.shape, (45,))  # nefc
    np.testing.assert_allclose(dx._impl.efc_aref[:3], d.efc_aref[:3])
    np.testing.assert_allclose(dx._impl.efc_aref[3], d.efc_aref[3])
    np.testing.assert_allclose(dx._impl.efc_aref[4], 0)
    np.testing.assert_allclose(dx._impl.efc_aref[5:15], d.efc_aref[4:14])
    np.testing.assert_allclose(dx._impl.efc_aref[15:], 0)

    # check sparse transform is correct
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx_sparse = mjx.put_data(m, d, impl=impl)
    np.testing.assert_allclose(
        dx_sparse._impl.efc_J, dx._impl.efc_J, atol=1e-8
    )

    # check sparse mass matrices are correct
    np.testing.assert_allclose(dx_sparse._impl.M, d.M, atol=1e-8)
    np.testing.assert_allclose(dx_sparse._impl.qLD, d.qLD, atol=1e-8)
    np.testing.assert_allclose(
        dx_sparse._impl.qLDiagInv, d.qLDiagInv, atol=1e-8
    )

    # check dense mass matrices are correct
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx_from_dense = mjx.put_data(m, d, impl=impl)
    if impl == 'jax':
      qm = np.zeros((m.nv, m.nv))
      mujoco.mju_sym2dense(qm, d.M, m.M_rownnz, m.M_rowadr, m.M_colind)
      np.testing.assert_allclose(dx_from_dense._impl.M, qm, atol=1e-8)


  def test_put_data_warp_ndim(self):
    """Tests that put_data produces expected dimensions for Warp fields."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d, impl='warp')

    def check_ndim(path, x):
      k = _get_name_from_path(path)
      if k not in mjxw_types._NDIM['Data']:
        return
      is_batched = mjxw_types._BATCH_DIM['Data'][k]
      expected_ndim = mjxw_types._NDIM['Data'][k] - is_batched
      if not hasattr(x, 'ndim'):
        return
      msg = f'Field {k} has ndim {x.ndim} but expected {expected_ndim}'
      self.assertEqual(x.ndim, expected_ndim, msg)

    _ = jax.tree.map_with_path(check_ndim, dx)

  @parameterized.parameters(
      ('jax', False), ('jax', True)
  )
  def test_get_data(self, impl: str, sparse: bool):
    """Test that get_data makes correct MjData."""
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    if sparse:
      m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d, impl=impl)
    d_2: mujoco.MjData = mjx.get_data(m, dx)

    # check a few fields
    np.testing.assert_allclose(d_2.qpos, d.qpos)
    np.testing.assert_allclose(d_2.xpos, d.xpos)
    np.testing.assert_allclose(d_2.cvel, d.cvel)
    np.testing.assert_allclose(d_2.cdof_dot, d.cdof_dot)
    np.testing.assert_allclose(d_2.qLD, d.qLD, atol=1e-6)
    np.testing.assert_allclose(d_2.qLDiagInv, d.qLDiagInv, atol=1e-6)

    # only 1 contact active
    self.assertEqual(d_2.contact.dist.shape, (1,))
    self.assertEqual(d_2.ncon, 1)
    np.testing.assert_allclose(d_2.contact.dist, d.contact.dist)
    self.assertEqual(d_2.contact.frame.shape, (1, 9))
    np.testing.assert_allclose(d_2.contact.frame, d.contact.frame)

    # xmat, ximat, geom_xmat, site_xmat are all shape transformed
    self.assertEqual(d_2.xmat.shape, (3, 9))
    self.assertEqual(d_2.ximat.shape, (3, 9))
    self.assertEqual(d_2.geom_xmat.shape, (3, 9))
    self.assertEqual(d_2.site_xmat.shape, (1, 9))
    np.testing.assert_allclose(d_2.xmat, d.xmat)
    np.testing.assert_allclose(d_2.ximat, d.ximat)
    np.testing.assert_allclose(d_2.geom_xmat, d.geom_xmat)
    np.testing.assert_allclose(d_2.site_xmat, d.site_xmat)

    # efc_* are also shape transformed and filtered
    self.assertEqual(d_2.nefc, 14)
    if sparse:
      efc_j = np.zeros((d.nefc, m.nv))
      mujoco.mju_sparse2dense(
          efc_j,
          d.efc_J,
          d.efc_J_rownnz,
          d.efc_J_rowadr,
          d.efc_J_colind,
      )
      efc_j2 = np.zeros((d_2.nefc, m.nv))
      mujoco.mju_sparse2dense(
          efc_j2,
          d_2.efc_J,
          d_2.efc_J_rownnz,
          d_2.efc_J_rowadr,
          d_2.efc_J_colind,
      )
      np.testing.assert_allclose(efc_j, efc_j2)
    else:
      self.assertEqual(d_2.efc_J.shape, (112,))  # nefc * nv
      np.testing.assert_allclose(d_2.efc_J, d.efc_J)
    self.assertEqual(d_2.efc_aref.shape, (14,))  # nefc
    np.testing.assert_allclose(d_2.efc_aref, d.efc_aref)
    np.testing.assert_allclose(d_2.contact.efc_address, d.contact.efc_address)


  def test_get_data_simplebody(self):
    """Test that get_data works with simple bodies where nC < nM."""
    m = mujoco.MjModel.from_xml_string(_SIMPLE_BODY)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d)
    d_2: mujoco.MjData = mjx.get_data(m, dx)
    np.testing.assert_allclose(d_2.qLD, d.qLD, atol=1e-6)
    np.testing.assert_allclose(d_2.qLDiagInv, d.qLDiagInv, atol=1e-6)

  def test_get_data_runs(self):
    xml = """
      <mujoco>
        <compiler autolimits="true"/>
        <worldbody>
          <body name="box">
            <joint name="slide1" type="slide" axis="1 0 0" />
            <geom type="box" size=".05 .05 .05" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="slide1"/>
        </actuator>
      </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    mjx.get_data(m, dx)

  @parameterized.parameters(('jax',))
  def test_get_data_batched(self, impl):
    """Test that get_data makes correct List[MjData] for batched Data."""

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d, impl=impl)
    # second data in batch has contact dist > 0, disables contact
    dx_b = jax.tree_util.tree_map(lambda x: jp.stack((x, x + 0.05)), dx)
    ds = mjx.get_data(m, dx_b)
    self.assertLen(ds, 2)
    np.testing.assert_allclose(ds[0].qpos, d.qpos)
    np.testing.assert_allclose(ds[1].qpos, d.qpos + 0.05, atol=1e-8)
    self.assertEqual(ds[0].ncon, 1)
    self.assertEqual(ds[1].ncon, 0)

  @parameterized.parameters('jax', 'cpp')
  def test_get_data_into(self, impl):
    """Test that get_data_into correctly populates an MjData."""

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    keepalive = {} if impl == 'cpp' else None
    dx = mjx.put_data(m, d, impl=impl, keepalive_refs=keepalive)
    d_2 = mujoco.MjData(m)
    mjx.get_data_into(d_2, m, dx, keepalive_refs=keepalive)

    # check a few fields
    np.testing.assert_allclose(d_2.qpos, d.qpos)
    np.testing.assert_allclose(d_2.xpos, d.xpos)

    # only 1 contact active
    self.assertEqual(d_2.contact.dist.shape, (1,))
    self.assertEqual(d_2.ncon, 1)
    np.testing.assert_allclose(d_2.contact.dist, d.contact.dist)
    self.assertEqual(d_2.contact.frame.shape, (1, 9))
    np.testing.assert_allclose(d_2.contact.frame, d.contact.frame)

  def test_get_data_into_warp(self):
    """Test get_data_into for impl='warp'."""

    # TODO(taylorhowell): After put_data supports impl='warp' update test above
    # and remove this test.
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device.')

    # Use a model with at least one kinematic tree so that island
    # fields (e.g. dof_island, tree_island) are populated on the host
    # MjData.
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    mx = mjx.put_model(m, impl='warp')
    dx = mjx.make_data(m, impl='warp')
    mjx.get_data_into(d, mx, dx)

  @parameterized.parameters(('jax',))
  def test_get_data_into_wrong_shape(self, impl):
    """Tests that get_data_into throwsif input and output shapes don't match."""

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d, impl=impl)
    m_2 = mujoco.MjModel.from_xml_string(_MULTIPLE_CONVEX_OBJECTS)
    d_2 = mujoco.MjData(m_2)
    with self.assertRaisesRegex(ValueError, r'Input field.*has shape.*'):
      mjx.get_data_into(d_2, m, dx)

  @parameterized.parameters(('jax',))
  def test_make_matches_put(self, impl):
    """Test that make_data produces a pytree that matches put_data."""
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 2)
    dx = mjx.put_data(m, d, impl=impl)

    step_fn = lambda d: d.replace(time=d.time + 1)
    step_fn_jit = jax.jit(step_fn).lower(dx).compile()

    # placing an MjData onto device should yield the same treedef mjx.Data as
    # calling make_data.  they should be interchangeable for jax functions:
    step_fn_jit(mjx.make_data(m, impl=impl))

  def test_contact_elliptic_condim1(self):
    """Test that condim=1 with ConeType.ELLIPTIC is not implemented."""
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <geom size="0 0 1e-5" type="plane" condim="1"/>
          <body>
            <freejoint/>
            <geom size="0.1" condim="1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    m.opt.cone = ConeType.ELLIPTIC
    with self.assertRaises(NotImplementedError):
      mjx.make_data(m)

  @parameterized.parameters(JacobianType.DENSE, JacobianType.SPARSE)
  def test_m_mapm2m(self, jacobian):
    """Test that M matches MuJoCo."""
    m = test_util.load_test_file('humanoid/humanoid.xml')
    m.opt.jacobian = jacobian
    d = mujoco.MjData(m)
    mx = mjx.put_model(m, impl='jax')
    dx = mjx.make_data(m, impl='jax')
    dx = mjx.forward(mx, dx)

    mjx.get_data_into(d, m, dx)

    res_mj = np.zeros((1, m.nv))
    mujoco.mj_solveM(m, d, res_mj, np.ones((1, m.nv)))

    res = mjx._src.smooth.solve_m(mx, dx, jp.ones(m.nv))

    np.testing.assert_allclose(res_mj[0], res, rtol=1e-3, atol=1e-3)

  def test_make_data_warp_has_expected_shapes(self):
    """Tests that make_data produces expected shapes for MuJoCo Warp."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    dx = mjx.make_data(m, impl='warp')

    def check_ndim(path, x):
      k = _get_name_from_path(path)
      if k not in mjxw_types._NDIM['Data']:
        return
      is_batched = mjxw_types._BATCH_DIM['Data'][k]
      expected_ndim = mjxw_types._NDIM['Data'][k] - is_batched
      if not hasattr(x, 'ndim'):
        return
      msg = f'Field {k} has ndim {x.ndim} but expected {expected_ndim}'
      self.assertEqual(x.ndim, expected_ndim, msg)

    _ = jax.tree.map_with_path(check_ndim, dx)

  @parameterized.parameters('jax', 'warp', 'cpp')
  def test_data_slice(self, impl):
    """Tests that slice on Data works as expected."""
    if impl == 'warp' and not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')
    if impl == 'warp' and not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    dx = jax.vmap(lambda x: mjx.make_data(m, impl=impl))(jp.arange(10))

    self.assertEqual(dx.qpos.shape, (10, m.nq))
    self.assertEqual(dx[0].qpos.shape, (m.nq,))

    if impl == 'warp':
      self.assertEqual(dx._impl.contact__dist.shape, (dx._impl.naconmax,))
      self.assertEqual(dx[0]._impl.contact__dist.shape, (dx._impl.naconmax,))

  def test_put_data_cpp(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body1" pos="0 0 1">
            <joint type="free"/>
            <geom type="sphere" size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    d.qpos[0] = 1.0
    unused_mjx_data = mjx_io.put_data(m, d, impl='cpp')

    def put_data(dummy_arg_for_batching):
      dx = mjx_io.put_data(
          m, d, impl='cpp', dummy_arg_for_batching=dummy_arg_for_batching
      )
      return dx

    vmjx_data = jax.vmap(put_data, in_axes=0, out_axes=0)(
        jp.zeros(2, dtype=jp.uint32)
    )

    self.assertEqual(vmjx_data.qpos.shape, (2, m.nq))
    lo = vmjx_data._impl.pointer_lo
    hi = vmjx_data._impl.pointer_hi
    addr0 = int(lo[0]) | (int(hi[0]) << 32)
    addr1 = int(lo[1]) | (int(hi[1]) << 32)
    self.assertNotEqual(addr0, addr1)

    jit_vmjx_data = jax.jit(jax.vmap(put_data, in_axes=0, out_axes=0))(
        jp.zeros(2, dtype=jp.uint32)
    )
    np.testing.assert_allclose(jit_vmjx_data.qpos[:, 0], [1.0, 1.0])


# Test cases for `_resolve_impl_and_device` where the device is
# specified by the user and the device is available.
_DEVICE_TEST_CASES = [
    # Arguments use the following format:
    # (device_type_str, impl_str,
    #  (expected_device, expected_impl)))
    # No backend specified.
    ('cpu', None, ('cpu', Impl.JAX)),
    ('gpu-notnvidia', None, ('gpu', Impl.JAX)),
    ('gpu-nvidia', None, ('gpu', Impl.WARP)),
    ('tpu', None, ('tpu', Impl.JAX)),
    # JAX backend specified.
    ('cpu', 'jax', ('cpu', Impl.JAX)),
    ('gpu-notnvidia', 'jax', ('gpu', Impl.JAX)),
    ('gpu-nvidia', 'jax', ('gpu', Impl.JAX)),
    ('tpu', 'jax', ('tpu', Impl.JAX)),
    # WARP backend specified.
    ('cpu', 'warp', ('cpu', Impl.WARP)),
    ('gpu-notnvidia', 'warp', ('gpu', 'error')),
    ('gpu-nvidia', 'warp', ('gpu', Impl.WARP)),
    ('tpu', 'warp', ('tpu', 'error')),
    # CPP backend specified.
    ('cpu', 'cpp', ('cpu', Impl.CPP)),
    ('gpu-notnvidia', 'cpp', ('cpu', 'error')),
    ('gpu-nvidia', 'cpp', ('cpu', 'error')),
    ('tpu', 'cpp', ('tpu', 'error')),
]

# Test cases for `_resolve_impl_and_device` where the user does NOT
# specify a device. We mock the JAX default device.
_DEFAULT_DEVICE_TEST_CASES = [
    # Arguments use the following format:
    # (jax.default_device, impl_str,
    #  (expected_device, expected_impl))
    # No backend impl specified.
    ('cpu', None, ('cpu', Impl.JAX)),
    ('gpu-notnvidia', None, ('gpu', Impl.JAX)),
    ('gpu-nvidia', None, ('gpu', Impl.WARP)),
    ('tpu', None, ('tpu', Impl.JAX)),
    # JAX backend impl specified.
    ('cpu', 'jax', ('cpu', Impl.JAX)),
    ('gpu-notnvidia', 'jax', ('gpu', Impl.JAX)),
    ('gpu-nvidia', 'jax', ('gpu', Impl.JAX)),
    ('tpu', 'jax', ('tpu', Impl.JAX)),
    # WARP backend impl specified.
    ('cpu', 'warp', ('cpu', Impl.WARP)),
    ('gpu-notnvidia', 'warp', ('cpu', Impl.WARP)),
    ('gpu-nvidia', 'warp', ('gpu', Impl.WARP)),
    ('tpu', 'warp', ('cpu', Impl.WARP)),
    # CPP backend impl specified, CPU should always be available.
    ('cpu', 'cpp', ('cpu', Impl.CPP)),
    ('gpu-notnvidia', 'cpp', ('cpu', Impl.CPP)),
    ('gpu-nvidia', 'cpp', ('cpu', Impl.CPP)),
    ('tpu', 'cpp', ('cpu', Impl.CPP)),
]


class ResolveImplAndDeviceTest(parameterized.TestCase):
  """Tests for the _resolve_impl_and_device function."""

  def setUp(self):
    super().setUp()

    # Create mock devices
    self.mock_cpu = mock.Mock(spec=jax.Device)
    self.mock_cpu.platform = 'cpu'
    self.mock_cpu.device_kind = 'Mock CPU'
    self.mock_cpu.id = 0

    self.mock_nvidia_gpu = mock.Mock(spec=jax.Device)
    self.mock_nvidia_gpu.platform = 'gpu'
    self.mock_nvidia_gpu.device_kind = 'NVIDIA Mocked GPU'
    self.mock_nvidia_gpu.id = 0

    self.mock_other_gpu = mock.Mock(spec=jax.Device)
    self.mock_other_gpu.platform = 'gpu'
    self.mock_other_gpu.device_kind = 'Other Mocked GPU'
    self.mock_other_gpu.id = 1

    self.mock_tpu = mock.Mock(spec=jax.Device)
    self.mock_tpu.platform = 'tpu'
    self.mock_tpu.device_kind = 'Mock TPU'
    self.mock_tpu.id = 0

    # Patch jax.devices for the entire test class using enter_context
    self.mock_jax_devices = self.enter_context(mock.patch('jax.devices'))
    self.mock_jax_backends = self.enter_context(
        mock.patch('jax.extend.backend.backends')
    )
    self.mock_default_backend = self.enter_context(
        mock.patch('jax.default_backend')
    )

  @parameterized.named_parameters(
      (f'{str(args[0])}_{str(args[1])}', *args) for args in _DEVICE_TEST_CASES
  )
  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_resolve_with_device(
      self,
      device_type_str,
      impl_str,
      expected,
  ):
    """Tests various combinations of device and backend impls."""
    input_device = {
        'cpu': self.mock_cpu,
        'gpu-nvidia': self.mock_nvidia_gpu,
        'gpu-notnvidia': self.mock_other_gpu,
        'tpu': self.mock_tpu,
    }[device_type_str]

    def devices_side_effect(backend=None):
      # assume the user-specified device is always available
      if backend == 'cpu':
        return [self.mock_cpu]
      elif backend == 'gpu':
        if device_type_str == 'gpu-nvidia':
          return [self.mock_nvidia_gpu]
        return [self.mock_other_gpu]
      elif backend == 'tpu':
        return [self.mock_tpu]
      elif backend == 'cuda':
        return [self.mock_nvidia_gpu]

      raise AssertionError('Should not be called.')
    self.mock_jax_devices.side_effect = devices_side_effect

    def backends_side_effect():
      if device_type_str == 'gpu-nvidia':
        return ['cuda', 'cpu']
      if 'tpu' in device_type_str:
        return ['tpu', 'cpu']
      if 'gpu' in device_type_str:
        return ['gpu', 'cpu']
      return ['cpu']

    self.mock_jax_backends.side_effect = backends_side_effect

    expected_device, expected_impl = expected
    if expected_impl == 'error':
      with self.assertRaises(AssertionError):
        mjx_io._resolve_impl_and_device(
            impl=impl_str, device=input_device
        )
      return

    if impl_str == 'warp' and not mjxw.WARP_INSTALLED:
      with self.assertRaisesRegex(RuntimeError, 'not installed'):
        mjx_io._resolve_impl_and_device(impl=impl_str, device=input_device)
      return

    if impl_str is None and not mjxw.WARP_INSTALLED:
      expected_impl = Impl.JAX

    actual_impl, actual_device = (
        mjx_io._resolve_impl_and_device(
            impl=impl_str, device=input_device
        )
    )

    self.assertEqual(actual_impl, expected_impl)
    self.assertIsNotNone(actual_device)
    self.assertEqual(actual_device.platform, expected_device)

  @parameterized.named_parameters(
      (f'{str(args[0])}_{str(args[1])}', *args)
      for args in _DEFAULT_DEVICE_TEST_CASES
  )
  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_resolve_without_device(
      self,
      default_device_str,
      impl_str,
      expected,
  ):
    """Tests various combinations of jax.default_device and backend impls."""
    default_devices = {
        'cpu': [self.mock_cpu],
        'gpu-nvidia': [self.mock_nvidia_gpu, self.mock_cpu],
        'gpu-notnvidia': [self.mock_other_gpu, self.mock_cpu],
        'tpu': [self.mock_tpu, self.mock_cpu],
    }[default_device_str]

    def devices_side_effect(backend=None):
      if backend == 'cpu':
        return [self.mock_cpu]  # CPU is always available
      if backend == 'gpu' and default_device_str == 'gpu-notnvidia':
        return [self.mock_other_gpu]
      if backend == 'gpu' and default_device_str == 'gpu-nvidia':
        return [self.mock_nvidia_gpu]
      if backend == 'cuda' and default_device_str == 'gpu-nvidia':
        return [self.mock_nvidia_gpu]
      if backend == 'tpu' and default_device_str == 'tpu':
        return [self.mock_tpu]
      if backend is None:
        return default_devices
      if backend == 'cuda':
        raise RuntimeError('cuda backend not supported')
      raise AssertionError('jax.devices error')
    self.mock_jax_devices.side_effect = devices_side_effect

    default_device_side_effect_str = {
        'cpu': 'cpu',
        'gpu-nvidia': 'gpu',
        'gpu-notnvidia': 'gpu',
        'tpu': 'tpu',
    }[default_device_str]
    self.mock_default_backend.side_effect = (
        lambda: default_device_side_effect_str
    )

    def backends_side_effect():
      if default_device_str == 'gpu-nvidia':
        return ['cuda', 'cpu']
      if 'tpu' in default_device_str:
        return ['tpu', 'cpu']
      if 'gpu' in default_device_str:
        return ['gpu', 'cpu']
      return ['cpu']

    self.mock_jax_backends.side_effect = backends_side_effect

    expected_device, expected_impl = expected
    if expected_impl == 'error':
      with self.assertRaises(AssertionError):
        mjx_io._resolve_impl_and_device(impl=impl_str, device=None)
      return

    if impl_str == 'warp' and not mjxw.WARP_INSTALLED:
      with self.assertRaises(RuntimeError):
        mjx_io._resolve_impl_and_device(impl=impl_str, device=None)
      return

    if impl_str is None and not mjxw.WARP_INSTALLED:
      expected_impl = Impl.JAX

    actual_impl, actual_device = mjx_io._resolve_impl_and_device(
        impl=impl_str, device=None
    )

    self.assertEqual(actual_impl, expected_impl)
    self.assertIsNotNone(actual_device)
    self.assertEqual(actual_device.platform, expected_device)

  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'false'})
  def test_resolve_warp_disabled(self):
    """Tests behavior when MJX_GPU_DEFAULT_WARP is false."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp is not installed.')

    self.mock_jax_devices.side_effect = lambda backend=None: (
        [self.mock_nvidia_gpu, self.mock_cpu]
        if backend is None
        else ([self.mock_nvidia_gpu] if backend == 'gpu' else [self.mock_cpu])
    )
    self.mock_default_backend.side_effect = lambda: 'gpu'

    # Default to JAX instead of WARP on NVIDIA GPU.
    impl, device = mjx_io._resolve_impl_and_device(
        impl=None, device=None
    )
    self.assertEqual(impl, Impl.JAX)
    self.assertEqual(device.platform, 'gpu')

    # Specifying an NVIDIA GPU should still choose JAX.
    impl, device = mjx_io._resolve_impl_and_device(
        impl=None, device=self.mock_nvidia_gpu
    )
    self.assertEqual(impl, Impl.JAX)
    self.assertEqual(device.platform, 'gpu')


class StateIOTest(parameterized.TestCase):

  @parameterized.parameters(
      mujoco.mjtState.mjSTATE_TIME,
      mujoco.mjtState.mjSTATE_QPOS,
      mujoco.mjtState.mjSTATE_QVEL,
      mujoco.mjtState.mjSTATE_ACT,
      mujoco.mjtState.mjSTATE_WARMSTART,
      mujoco.mjtState.mjSTATE_CTRL,
      mujoco.mjtState.mjSTATE_QFRC_APPLIED,
      mujoco.mjtState.mjSTATE_XFRC_APPLIED,
      mujoco.mjtState.mjSTATE_EQ_ACTIVE,
      mujoco.mjtState.mjSTATE_INTEGRATION,
  )
  def test_state_size(self, spec):
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    mx = mjx.put_model(m)
    self.assertEqual(mjx.state_size(mx, spec), mujoco.mj_stateSize(m, spec))

  def test_put_data_cpp(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body1" pos="0 0 1">
            <joint type="free"/>
            <geom type="sphere" size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    d.qpos[0] = 1.0
    unused_mjx_data = mjx_io.put_data(m, d, impl='cpp')

    def put_data(dummy_arg_for_batching):
      dx = mjx_io.put_data(
          m, d, impl='cpp', dummy_arg_for_batching=dummy_arg_for_batching
      )
      return dx

    vmjx_data = jax.vmap(put_data, in_axes=0, out_axes=0)(
        jp.zeros(2, dtype=jp.uint32)
    )

    self.assertEqual(vmjx_data.qpos.shape, (2, m.nq))
    lo = vmjx_data._impl.pointer_lo
    hi = vmjx_data._impl.pointer_hi
    addr0 = int(lo[0]) | (int(hi[0]) << 32)
    addr1 = int(lo[1]) | (int(hi[1]) << 32)
    self.assertNotEqual(addr0, addr1)

  def test_get_set_state(self):
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    # give the model a little kick to get some non-zero state
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    # test full state
    spec_full = mujoco.mjtState.mjSTATE_INTEGRATION
    state = mjx.get_state(mx, dx, spec_full)
    state_mj = np.empty(state.shape, dtype=np.float64)
    mujoco.mj_getState(m, d, state_mj, int(spec_full))
    np.testing.assert_allclose(state, state_mj, atol=1e-6)
    dx2 = mjx.set_state(mx, mjx.make_data(m), state, spec_full)
    np.testing.assert_allclose(dx.qpos, dx2.qpos)
    np.testing.assert_allclose(dx.qvel, dx2.qvel)
    np.testing.assert_allclose(dx.act, dx2.act)
    np.testing.assert_allclose(dx.qacc_warmstart, dx2.qacc_warmstart)
    np.testing.assert_allclose(dx.ctrl, dx2.ctrl)
    np.testing.assert_allclose(dx.qfrc_applied, dx2.qfrc_applied)
    np.testing.assert_allclose(dx.xfrc_applied, dx2.xfrc_applied)
    np.testing.assert_allclose(dx.eq_active, dx2.eq_active)

    # test single state
    for spec in [
        mujoco.mjtState.mjSTATE_TIME,
        mujoco.mjtState.mjSTATE_QPOS,
        mujoco.mjtState.mjSTATE_QVEL,
        mujoco.mjtState.mjSTATE_ACT,
        mujoco.mjtState.mjSTATE_WARMSTART,
        mujoco.mjtState.mjSTATE_CTRL,
        mujoco.mjtState.mjSTATE_QFRC_APPLIED,
        mujoco.mjtState.mjSTATE_XFRC_APPLIED,
        mujoco.mjtState.mjSTATE_EQ_ACTIVE,
    ]:
      state = mjx.get_state(mx, dx, spec)
      state_mj = np.empty(state.shape, dtype=np.float64)
      mujoco.mj_getState(m, d, state_mj, int(spec))
      np.testing.assert_allclose(state, state_mj)
      dx2 = mjx.set_state(mx, mjx.make_data(m), state, spec)
      np.testing.assert_allclose(
          getattr(dx, mjx_io._STATE_MAP[spec]),
          getattr(dx2, mjx_io._STATE_MAP[spec]),
      )

    # test partial state
    spec = (
        mujoco.mjtState.mjSTATE_QPOS
        | mujoco.mjtState.mjSTATE_QVEL
    )
    state = mjx.get_state(mx, dx, spec)
    state_mj = np.empty(state.shape, dtype=np.float64)
    mujoco.mj_getState(m, d, state_mj, int(spec))
    np.testing.assert_allclose(state, state_mj)

    # check that we only set qpos/qvel and other values are at init
    dx_init = mjx.make_data(m)
    dx2 = mjx.set_state(mx, dx_init, state, spec)
    np.testing.assert_allclose(dx.qpos, dx2.qpos)
    np.testing.assert_allclose(dx.qvel, dx2.qvel)
    np.testing.assert_allclose(dx_init.time, dx2.time)
    np.testing.assert_allclose(dx_init.act, dx2.act)

  def test_jit(self):
    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONSTRAINTS)
    d = mujoco.MjData(m)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION

    get_state_jit = jax.jit(mjx.get_state, static_argnames='spec')
    state = get_state_jit(mx, dx, spec)
    state_nojit = mjx.get_state(mx, dx, spec)
    np.testing.assert_allclose(state, state_nojit)

    set_state_jit = jax.jit(mjx.set_state, static_argnames='spec')
    dx2 = set_state_jit(mx, dx, state, spec)
    dx2_nojit = mjx.set_state(mx, dx, state, spec)
    np.testing.assert_allclose(dx2.qpos, dx2_nojit.qpos)


if __name__ == '__main__':
  absltest.main()
