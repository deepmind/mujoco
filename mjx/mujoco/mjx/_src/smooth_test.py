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
"""Tests for smooth dynamics functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import test_util
from mujoco.mjx._src.types import ConeType  # pylint: disable=g-importing-member
from mujoco.mjx._src.types import JacobianType  # pylint: disable=g-importing-member
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


def _sidesite_wrap_xml(geomtype: str) -> str:
  """Returns a model with wrapping tendons, with and without sidesites."""
  # wrap geom in a rotated frame with radius 0.5, and unrotated geoms with
  # radius 0.5 and 1
  pos = np.array([0.1, -0.2, 0.3])
  quat = np.array([0.8, 0.2, -0.3, 0.4])
  quat /= np.linalg.norm(quat)
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, quat)
  mat = mat.reshape(3, 3)

  # local endpoints and sidesite, geom name, tendon name
  cases = []

  # wraps constructed from their arcs (start angle a, wrap angle th,
  # direction), including arcs longer than pi, with the sidesite at the arc
  # midpoint and the segment cutting the circle
  r = 0.5
  for a in (0.1, 1.3, 2.9, 4.4, 5.8):
    for th in (0.4, 1.5, 2.8, 3.5, 4.5, 5.5):
      for direction in (1, -1):
        for len0, len1 in ((0.1, 0.15), (0.6, 0.8), (2.0, 1.7), (0.6, 1.7)):
          b = a + direction * th
          t0 = r * np.array([np.cos(a), np.sin(a)])
          t1 = r * np.array([np.cos(b), np.sin(b)])
          p0 = t0 + len0 * direction * np.array([np.sin(a), -np.cos(a)])
          p1 = t1 - len1 * direction * np.array([np.sin(b), -np.cos(b)])

          # keep segments that cut the circle with a margin
          dif = p1 - p0
          s = np.clip(-np.dot(dif, p0) / np.dot(dif, dif), 0, 1)
          if np.linalg.norm(p0 + s * dif) > 0.9 * r:
            continue

          m = a + direction * th / 2
          side = [2 * r * np.cos(m), 2 * r * np.sin(m), 0]
          cases.append(([*p0, 0], [*p1, 0], side, 'rotated', ''))

  # sidesite more than 90 degrees from the segment: no wrap
  cases.append((
      [-4.17, -0.44, 0], [-1.13, -1.53, 0], [-0.79 * 1.5, 0.61 * 1.5, 0],
      'r1', 'nowrap',
  ))

  # segment through the center wraps on the side of the sidesite
  cases.append(([0, 2, 0], [0, -1, 0], [1.5, 0, 0], 'r05', ''))
  cases.append(([0, 2, 0], [0, -1, 0], [-1.5, 0, 0], 'r05', ''))

  # wraps close to 2*pi
  cases.append(([1.001, -1e-2, 0], [1.001, 1e-2, 0], [-2, 0, 0], 'r1', ''))
  cases.append(([1, -1e-7, 0], [1, 1e-7, 0], [-2, 0, 0], 'r1', 'fullwrap'))

  sites, tendons = [], []
  for i, (p0, p1, side, geom, name) in enumerate(cases):
    p0, p1, side = (np.array(p, dtype=float) for p in (p0, p1, side))

    # cylinder: endpoints at different heights
    if geomtype == 'cylinder':
      p0[2], p1[2] = 0.3, -0.2

    if geom == 'rotated':
      p0, p1, side = (mat @ p + pos for p in (p0, p1, side))

    for site, p in (('a', p0), ('b', p1), ('side', side)):
      p_str = ' '.join(f'{x:.17g}' for x in p)
      sites.append(f'<site name="{site}{i}" pos="{p_str}"/>')
    # no sidesite: skip endpoints on the circle, whose wrap is below float32
    # resolution
    sidesites = [f'sidesite="side{i}"'] + ([] if name == 'fullwrap' else [''])
    for sidesite in sidesites:
      tendon_name = f'name="{name}"' if name and sidesite else ''
      path = (
          f'<site site="a{i}"/><geom geom="{geom}" {sidesite}/>'
          f'<site site="b{i}"/>'
      )
      tendons.append(f'<spatial {tendon_name}>{path}</spatial>')

  pos_str = ' '.join(f'{x:.17g}' for x in pos)
  quat_str = ' '.join(f'{x:.17g}' for x in quat)
  sites_str = '\n'.join(sites)
  tendons_str = '\n'.join(tendons)
  return f"""
    <mujoco>
      <worldbody>
        <geom name="rotated" type="{geomtype}" size="0.5 1" pos="{pos_str}"
              quat="{quat_str}"/>
        <geom name="r05" type="{geomtype}" size="0.5 1"/>
        <geom name="r1" type="{geomtype}" size="1 1"/>
        {sites_str}
        <body>
          <joint type="slide"/>
          <geom size="0.1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      <tendon>
        {tendons_str}
      </tendon>
    </mujoco>
  """


class SmoothTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # although we already have generous padding of thresholds, it doesn't hurt
    # to also fix the seed to reduce test flakiness
    np.random.seed(0)

  def test_smooth(self):
    """Tests MJX smooth functions match MuJoCo smooth functions."""

    m = test_util.load_test_file('pendula.xml')
    # tell MJX to use sparse mass matrices:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    # randomize mocap
    d.mocap_pos = np.random.random(d.mocap_pos.shape)
    d.mocap_quat = np.random.random(d.mocap_quat.shape)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)

    # kinematics
    dx = jax.jit(mjx.kinematics)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'xanchor')
    _assert_attr_eq(d, dx, 'xaxis')
    _assert_attr_eq(d, dx, 'xpos')
    _assert_attr_eq(d, dx, 'xquat')
    _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
    _assert_attr_eq(d, dx, 'xipos')
    _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
    _assert_attr_eq(d, dx, 'geom_xpos')
    _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
    _assert_attr_eq(d, dx, 'site_xpos')
    _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')
    # com_pos
    dx = jax.jit(mjx.com_pos)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'subtree_com')
    _assert_attr_eq(d, dx._impl, 'cinert')
    _assert_attr_eq(d, dx, 'cdof')
    # camlight
    dx = jax.jit(mjx.camlight)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'cam_xpos')
    _assert_eq(d.cam_xmat.reshape((-1, 3, 3)), dx.cam_xmat, 'cam_xmat')
    # crb
    dx = jax.jit(mjx.crb)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx._impl, 'crb')
    _assert_attr_eq(d, dx._impl, 'M')
    # factor_m
    dx = jax.jit(mjx.factor_m)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx._impl, 'qLD')
    _assert_attr_eq(d, dx._impl, 'qLDiagInv')
    # com_vel
    dx = jax.jit(mjx.com_vel)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'cvel')
    _assert_attr_eq(d, dx, 'cdof_dot')
    # rne
    dx = jax.jit(mjx.rne)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'qfrc_bias')
    # rne (flg_acc=True)
    qfrc_bias = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 1, qfrc_bias)
    dx = jax.jit(mjx.rne, static_argnums=(2,))(
        mx, mjx.put_data(m, d), flg_acc=True
    )
    _assert_eq(dx.qfrc_bias, qfrc_bias, 'qfrc_bias')

    # set dense jacobian for tendon:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)
    # tendon
    dx = jax.jit(mjx.tendon)(mx, mjx.put_data(m, d))
    ten_J = np.zeros((m.ntendon, m.nv))
    mujoco.mju_sparse2dense(
        ten_J,
        d.ten_J,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
    )
    _assert_eq(ten_J, dx._impl.ten_J, 'ten_J')
    _assert_attr_eq(d, dx, 'ten_length')
    # transmission
    dx = jax.jit(mjx.transmission)(mx, dx)
    _assert_attr_eq(d, dx, 'actuator_length')

    # convert sparse actuator_moment to dense representation
    moment = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(
        moment,
        d.actuator_moment,
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
    )
    _assert_eq(moment, dx._impl.actuator_moment, 'actuator_moment')

  def test_disable_gravity(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option>
            <flag gravity="disable"/>
          </option>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    dx = jax.jit(mjx.rne)(mx, dx)
    np.testing.assert_allclose(dx.qfrc_bias, 0)

  def test_site_transmission(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
        <compiler autolimits="true"/>
        <worldbody>
          <body quat="1 1 0 0">
            <joint type="free"/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
            <site name="site1" quat="1 0 1 0"/>
            <body quat="1 0 0 1">
              <joint type="hinge"/>
              <geom size="0.1" mass="1"/>
              <site name="site2" pos="0.1 0.2 0.3" quat="0 1 1 0"/>
            </body>
          </body>
          <body pos="1 0 0">
            <joint name="slide" type="hinge"/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <position site="site1" kv="0.1" gear="1 2 3 0 0 0"/>
          <position site="site1" kv="0.2" gear="0 0 0 1 2 3"/>
          <position site="site2" kv="0.3" gear="0 3 0 0 0 1"/>
          <position joint="slide" kv="0.05" />
          <position site="site2" refsite="site1" gear="1 2 3 0.5 0.4 0.6"/>
        </actuator>
        </mujoco>
      """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_transmission(m, d)
    dx = jax.jit(mjx.transmission)(mx, dx)
    _assert_attr_eq(d, dx, 'actuator_length')

    # convert sparse actuator_moment to dense representation
    moment = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(
        moment,
        d.actuator_moment,
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
    )
    _assert_eq(moment, dx._impl.actuator_moment, 'actuator_moment')

  def test_subtree_vel(self):
    """Tests MJX subtree_vel function matches MuJoCo mj_subtreeVel."""

    m = test_util.load_test_file('humanoid/humanoid.xml')
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    # subtree velocity
    mujoco.mj_subtreeVel(m, d)
    dx = jax.jit(mjx.subtree_vel)(mx, dx)

    _assert_attr_eq(d, dx._impl, 'subtree_linvel')
    _assert_attr_eq(d, dx._impl, 'subtree_angmom')


class RnePostConstraintTest(parameterized.TestCase):
  _CONNECT_SITE = """
    <equality>
      <connect site1="site1" site2="site2"/>
    </equality>
    """
  _CONNECT_BODY = """
    <equality>
      <connect body1="body1" body2="body2" anchor="1 2 3"/>
    </equality>
    """
  _WELD_SITE = """
    <equality>
      <weld site1="site1" site2="site2"/>
    </equality>
    """
  _WELD_BODY = """
    <equality>
      <weld body1="body1" body2="body2"/>
    </equality>
    """
  _CONNECT_SITE_WELD_SITE = """
    <equality>
      <connect site1="site1" site2="site2"/>
      <weld site1="site1" site2="site2"/>
    </equality>
    """
  _WELD_SITE_CONNECT_SITE = """
    <equality>
      <weld site1="site1" site2="site2"/>
      <connect site1="site1" site2="site2"/>
    </equality>
    """
  _WELD_SITE_CONNECT_SITE_WELD_BODY = """
    <equality>
      <weld site1="site1" site2="site2"/>
      <connect site1="site1" site2="site2"/>
      <weld body1="body1" body2="body2"/>
    </equality>
    """
  _CONNECT_SITE_WELD_SITE_WELD_BODY = """
    <equality>
      <connect site1="site1" site2="site2"/>
      <weld site1="site1" site2="site2"/>
      <weld body1="body1" body2="body2"/>
    </equality>
    """
  _CONNECT_SITE_CONNECT_BODY_CONNECT_WELD = """
    <equality>
      <connect site1="site1" site2="site2"/>
      <connect body1="body1" body2="body2" anchor="1 2 3"/>
      <weld body1="body1" body2="body2"/>
    </equality>
    """

  @parameterized.parameters(
      ('', ConeType.PYRAMIDAL, None),
      ('', ConeType.ELLIPTIC, None),
      (_CONNECT_SITE, ConeType.PYRAMIDAL, None),
      (_CONNECT_BODY, ConeType.PYRAMIDAL, None),
      (_WELD_SITE, ConeType.PYRAMIDAL, None),
      (_WELD_BODY, ConeType.PYRAMIDAL, None),
      (_CONNECT_SITE_WELD_SITE, ConeType.PYRAMIDAL, None),
      (
          _WELD_SITE_CONNECT_SITE,
          ConeType.PYRAMIDAL,
          np.array([6, 7, 8, 0, 1, 2, 3, 4, 5]),
      ),
      (
          _WELD_SITE_CONNECT_SITE_WELD_BODY,
          ConeType.PYRAMIDAL,
          np.array([6, 7, 8, 0, 1, 2, 3, 4, 5]),
      ),
      (_CONNECT_SITE_WELD_SITE_WELD_BODY, ConeType.PYRAMIDAL, None),
      (_CONNECT_SITE_CONNECT_BODY_CONNECT_WELD, ConeType.PYRAMIDAL, None),
  )
  def test_rnepostconstraint(self, equality, cone_type, efc_map):
    """Tests MJX rne_postconstraint function to match MuJoCo mj_rnePostConstraint."""

    m = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <worldbody>
            <geom name="floor" size="10 10 .05" type="plane"/>
            <site name="site1"/>
            <body name="body1">
            </body>
            <body pos="0 0 1" name="body2">
              <joint type="ball" damping="1"/>
              <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0" condim="1"/>
              <body pos="0.5 0 0">
                <joint type="ball" damping="1"/>
                <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0"  condim="3"/>
                <site name="site2"/>
              </body>
            </body>
            <body pos="0 1 1">
              <joint type="ball" damping="1"/>
              <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0" condim="6"/>
              <body pos="0.5 0 0">
                <joint type="ball" damping="1"/>
                <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0"  condim="3"/>
              </body>
            </body>
          </worldbody>
          {equality}
          <keyframe>
            <key qpos='0.424577 0.450592 0.451703 -0.642391 0.729379 0.545151 0.407756 0.0674697 0.424577 1.450592 0.451703 -0.642391 0.729379 0.545151 0.407756 0.0674697'/>
          </keyframe>
        </mujoco>
    """)
    # set cone type
    m.opt.cone = cone_type
    # create data and set to keyframe
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    # apply external forces
    d.xfrc_applied = 0.001 * np.ones(d.xfrc_applied.shape)
    mujoco.mj_step(m, d, 2)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    if efc_map is not None:
      efc_force = d.efc_force.copy()
      efc_force[: len(efc_map)] = d.efc_force[efc_map]
      dx = dx.tree_replace({'_impl.efc_force': jp.array(efc_force)})

    # rne postconstraint
    mujoco.mj_rnePostConstraint(m, d)
    dx = jax.jit(mjx.rne_postconstraint)(mx, dx)

    _assert_eq(d.cacc, dx._impl.cacc, 'cacc')
    _assert_eq(d.cfrc_ext, dx._impl.cfrc_ext, 'cfrc_ext')
    _assert_eq(d.cfrc_int, dx._impl.cfrc_int, 'cfrc_int')


class TendonTest(parameterized.TestCase):

  @parameterized.parameters(
      'tendon/fixed.xml',
      'tendon/fixed_site.xml',
      'tendon/fixed_site_wrap.xml',
      'tendon/pulley_fixed_site_wrap.xml',
      'tendon/pulley_site.xml',
      'tendon/pulley_site_wrap.xml',
      'tendon/pulley_wrap.xml',
      'tendon/no_tendon.xml',
      'tendon/site.xml',
      'tendon/site_wrap.xml',
      'tendon/tendon.xml',
      'tendon/wrap_sidesite.xml',
  )
  def test_tendon(self, filename):
    """Tests MJX tendon function matches MuJoCo mj_tendon."""
    m = test_util.load_test_file(filename)
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_forward(m, d)
    dx = jax.jit(mjx.forward)(mx, dx)

    _assert_eq(d.ten_length, dx.ten_length, 'ten_length')
    ten_J = np.zeros((m.ntendon, m.nv))
    mujoco.mju_sparse2dense(
        ten_J,
        d.ten_J,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
    )
    _assert_eq(ten_J, dx._impl.ten_J, 'ten_J')
    _assert_eq(d.ten_wrapnum, dx._impl.ten_wrapnum, 'ten_wrapnum')
    _assert_eq(d.ten_wrapadr, dx._impl.ten_wrapadr, 'ten_wrapadr')
    _assert_eq(d.wrap_obj, dx._impl.wrap_obj, 'wrap_obj')
    _assert_eq(d.wrap_xpos, dx._impl.wrap_xpos, 'wrap_xpos')

  @parameterized.parameters(JacobianType.DENSE, JacobianType.SPARSE)
  def test_tendon_armature(self, jacobian):
    """Tests MJX tendon armature matches MuJoCo."""
    m = test_util.load_test_file('tendon/armature.xml')
    m.opt.jacobian = jacobian
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    dx = dx.tree_replace(
        {'_impl.M': jp.zeros((m.nv, m.nv)), 'qfrc_bias': jp.zeros(m.nv)}
    )

    dx = mjx.crb(mx, dx)
    dx = mjx.tendon_armature(mx, dx)

    if jacobian == JacobianType.DENSE:
      M = np.zeros((m.nv, m.nv))  # pylint: disable=invalid-name
      mujoco.mju_sym2dense(M, d.M, m.M_rownnz, m.M_rowadr, m.M_colind)
    else:
      M = d.M  # pylint: disable=invalid-name
    _assert_eq(dx._impl.M, M, 'M')

    dx = mjx.rne(mx, dx)
    dx = mjx.tendon_bias(mx, dx)
    _assert_eq(dx.qfrc_bias, d.qfrc_bias, 'qfrc_bias')

  @parameterized.parameters('sphere', 'cylinder')
  def test_tendon_wrap_sidesite(self, geomtype):
    """Tests that the sidesite selects the wrap homotopy class as in MuJoCo."""
    m = mujoco.MjModel.from_xml_string(_sidesite_wrap_xml(geomtype))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    dx = dx.tree_replace({
        'ten_length': jp.zeros_like(dx.ten_length),
        '_impl.wrap_xpos': jp.zeros_like(dx._impl.wrap_xpos),
    })

    dx = jax.jit(mjx.tendon)(mx, dx)

    _assert_eq(d.ten_length, dx.ten_length, 'ten_length')
    _assert_eq(d.ten_wrapnum, dx._impl.ten_wrapnum, 'ten_wrapnum')
    _assert_eq(d.ten_wrapadr, dx._impl.ten_wrapadr, 'ten_wrapadr')
    _assert_eq(d.wrap_obj, dx._impl.wrap_obj, 'wrap_obj')
    _assert_eq(d.wrap_xpos, dx._impl.wrap_xpos, 'wrap_xpos')

    # most tendons wrap, including wraps longer than pi
    self.assertGreater(np.sum(d.ten_wrapnum == 4), m.ntendon // 2)

    # sidesite more than 90 degrees from the segment: no wrap
    nowrap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, 'nowrap')
    self.assertEqual(d.ten_wrapnum[nowrap], 2)

    # sidesite on the far side: wrap close to 2*pi
    fullwrap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, 'fullwrap')
    self.assertGreater(dx.ten_length[fullwrap], 2 * np.pi - 1e-3)


if __name__ == '__main__':
  absltest.main()
