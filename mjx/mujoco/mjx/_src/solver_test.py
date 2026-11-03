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
"""Tests for constraint functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
from mujoco.mjx._src import test_util
from mujoco.mjx._src.types import ConeType
import numpy as np


# tolerance for difference between MuJoCo and MJX solver calculations,
# mostly due to float precision
_TOLERANCE = 5e-3


def _assert_eq(a, b, name, tol=_TOLERANCE):
  tol = tol * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr, tol=_TOLERANCE):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr, tol=tol)


class SolverTest(parameterized.TestCase):

  @parameterized.parameters(
      # these scene challenges the solver, with CG you need to crank up
      # the iterations, otherwise it diverges
      (mujoco.mjtSolver.mjSOL_CG, mujoco.mjtCone.mjCONE_PYRAMIDAL, 100),
      (mujoco.mjtSolver.mjSOL_CG, mujoco.mjtCone.mjCONE_ELLIPTIC, 100),
      # Newton converges much more quickly, lower iterations to demonstrate
      # mgrad is being calculated optimally
      (mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_PYRAMIDAL, 2),
      (mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_ELLIPTIC, 2),
  )
  def test_solver(self, solver_, cone, iterations):
    """Test newton, CG solver with pyramidal, elliptic cones."""
    m = test_util.load_test_file('constraints.xml')
    m.opt.solver = solver_
    m.opt.cone = cone
    m.opt.iterations = iterations

    # with islanding on, MuJoCo CG converges *much* faster,
    # too fast for the low-iteration comparison to be meaningful,
    # so we disable islanding at low iteration count
    if iterations < 5:
      m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND

    d = mujoco.MjData(m)

    def cost(qacc):
      jaref = np.zeros(d.nefc, dtype=float)
      cost = np.zeros(1)
      mujoco.mj_mulJacVec(m, d, jaref, qacc)
      mujoco.mj_constraintUpdate(m, d, jaref - d.efc_aref, cost, 0)
      return cost

    # sample a mix of active/inactive constraints at different timesteps
    for key in range(0, 3):
      mujoco.mj_resetDataKeyframe(m, d, key)
      mujoco.mj_step(m, d)  # step to generate warmstart

      # compare costs
      mj_cost = cost(d.qacc)
      ctx = solver.Context.create(mjx.put_model(m), mjx.put_data(m, d))
      mjx_cost = ctx.cost - ctx.gauss
      _assert_eq(mj_cost, mjx_cost, 'cost')

      mujoco.mj_forward(m, d)
      dx = jax.jit(mjx.solve)(mjx.put_model(m), mjx.put_data(m, d))

      # MJX finds very similar solutions with the newton solver
      if solver_ == mujoco.mjtSolver.mjSOL_NEWTON:
        nnz = dx._impl.efc_J.any(axis=1)
        _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force')
        _assert_attr_eq(d, dx, 'qfrc_constraint')
        _assert_attr_eq(d, dx, 'qacc')

      # both CG and Newton find costs that are nearly the same as MuJoCo, often
      # lower (due to slight differences in the MJX linsearch algorithm)
      mj_cost = cost(d.qacc)
      mjx_cost = cost(dx.qacc)
      self.assertLess(mjx_cost, mj_cost * 1.015)

  def test_no_warmstart(self):
    """Test no warmstart."""
    m = test_util.load_test_file('constraints.xml')
    d = mujoco.MjData(m)
    # significant constraint forces keyframe 2
    mujoco.mj_resetDataKeyframe(m, d, 2)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = jax.jit(mjx.solve)(mx, mjx.put_data(m, d))
    nnz = dx._impl.efc_J.any(axis=1)
    # even without warmstart, newton converges quickly
    _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force', tol=2e-4)

  @parameterized.parameters(*ConeType)
  def test_single_iteration(self, cone):
    """Test one Newton iteration matches MuJoCo C on a perturbed state."""
    m = test_util.load_test_file('constraints.xml')
    m.opt.cone = cone
    m.opt.iterations = 1
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    solve = jax.jit(mjx.solve)
    free = 7 if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0
    mujoco.mj_resetDataKeyframe(m, d, 1)
    d.qpos[free:] += 0.05 * np.sin(248 * np.arange(m.nq - free))
    d.qvel += 0.5 * np.cos(152 * np.arange(m.nv))
    # step to generate warmstart
    mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    dx = solve(mx, mjx.put_data(m, d))
    # linesearch must accept a converged candidate
    # regardless of its derivative's sign
    _assert_eq(d.qacc, dx.qacc, 'qacc')

  def test_sparse(self):
    """Test solver works with sparse mass matrices."""
    m = test_util.load_test_file('constraints.xml')
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    # significant constraint forces keyframe 2
    mujoco.mj_resetDataKeyframe(m, d, 2)

    mujoco.mj_forward(m, d)

    dx = jax.jit(mjx.solve)(mjx.put_model(m), mjx.put_data(m, d))

    _assert_attr_eq(d, dx, 'qacc')
    _assert_attr_eq(d, dx, 'qfrc_constraint')
    nnz = dx._impl.efc_J.any(axis=1)
    _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force')

  def test_quad_frictionloss(self):
    """Test a case with quadratic frictionloss constraints."""
    m = test_util.load_test_file('quadratic_frictionloss.xml')
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    dx = jax.jit(mjx.solve)(mjx.put_model(m), mjx.put_data(m, d))

    _assert_attr_eq(d, dx, 'qacc')
    _assert_attr_eq(d, dx, 'qfrc_constraint')
    nnz = dx._impl.efc_J.any(axis=1)
    _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force')

  # TODO(taylorhowell): condim=1 with ConeType.ELLIPTIC
  @parameterized.product(condim=(3, 4, 6), cone=tuple(ConeType))
  def test_condim(self, condim, cone):
    """Test contact dimension."""
    m = mujoco.MjModel.from_xml_string(f"""
       <mujoco>
          <worldbody>
            <geom size="0 0 1e-5" type="plane" condim="1"/>
            <body pos="0 0 0.09">
              <freejoint/>
              <geom size="0.1" condim="{condim}"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    m.opt.cone = cone
    solver.solve(mjx.put_model(m), mjx.put_data(m, mujoco.MjData(m)))

  def test_elliptic_linesearch_cancellation(self):
    """Elliptic linesearch preserves orthogonal tangential curvature under cancellation."""
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option cone="elliptic"/>
        <worldbody>
          <geom type="plane" size="1 1 1"/>
          <body pos="0 0 0.09">
            <freejoint/>
            <geom type="sphere" size="0.1" friction="1 1 1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    ctx = solver.Context.create(mx, dx)

    # Set u = [0, 1, 2^-13, 0, 0, 0] and search so jv = [0, -1, 0].
    # In float32, 1 + 2^-26 rounds to 1.0, so naive uu + alpha*(2*uv + alpha*vv)
    # vanishes at alpha = 1 and loses the middle-zone curvature.
    eps_perp = np.float32(2.0**-13)
    u = jax.numpy.array([[0.0, 1.0, eps_perp, 0.0, 0.0, 0.0]])
    fri = jax.numpy.ones((1, 6))
    dm = jax.numpy.array([1.0])
    ctx = ctx.replace(
        Jaref=jax.numpy.array([0.0, 1.0, eps_perp]),
        u=u,
        fri=fri,
        dm=dm,
    )
    jv = jax.numpy.array([0.0, -1.0, 0.0])
    quad = jax.numpy.zeros((3, 3))
    quad_gauss = jax.numpy.zeros(3)

    u_tan = u[:, 1:]
    v_tan = jax.numpy.array([[-1.0, 0.0, 0.0, 0.0, 0.0]])
    wedge = (
        u_tan[:, :, None] * v_tan[:, None, :]
        - u_tan[:, None, :] * v_tan[:, :, None]
    )
    det = 0.5 * jax.numpy.sum(jax.numpy.square(wedge), axis=(1, 2))
    vv = jax.numpy.sum(v_tan * v_tan, axis=1)
    v_norm = jax.numpy.sqrt(vv)
    u_par = jax.numpy.sum(u_tan * v_tan, axis=1) / v_norm
    u_perp_sq = det / vv

    pt = solver._LSPoint.create(
        mx,
        dx,
        ctx,
        jax.numpy.array(1.0),
        jv,
        quad,
        quad_gauss,
        u_perp_sq,
        jax.numpy.array([0.0]),
        u_par,
        v_norm,
    )
    np.testing.assert_allclose(pt.cost, 0.5 * eps_perp**2, rtol=1e-5)
    np.testing.assert_allclose(pt.deriv_0, 0.0, atol=1e-7)
    np.testing.assert_allclose(pt.deriv_1, 1.0, rtol=1e-5)

  @parameterized.parameters(1e-6, 1e-12, 1e-17)
  def test_elliptic_hessian_small_tangent(self, t):
    """Elliptic cone Hessian is exact when T^3 < mjMINVAL."""
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option cone="elliptic" solver="Newton"/>
        <worldbody>
          <geom type="plane" size="1 1 1"/>
          <body pos="0 0 0.09">
            <freejoint/>
            <geom type="sphere" size="0.1" friction="1 1 1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    ctx = solver.Context.create(mx, dx)

    # middle zone with tiny tangent t along the first tangent direction
    fri = np.asarray(ctx.fri[0])
    mu = fri[0]
    n = -0.5 * t / mu
    jaref = jax.numpy.array([n / mu, t / fri[1], 0.0])
    ctx = solver._update_constraint(mx, dx, ctx.replace(Jaref=jaref))

    # H = Dm * F * [[1, -mu, 0], [-mu, mu^2, 0], [0, 0, mu^2 - mu*n/t]] * F
    h = np.zeros((3, 3))
    h[0, 0] = 1.0
    h[0, 1] = h[1, 0] = -mu
    h[1, 1] = mu * mu
    h[2, 2] = mu * mu - mu * n / t
    h *= float(ctx.dm[0]) * np.outer(fri[:3], fri[:3])
    np.testing.assert_allclose(ctx.h[0, :3, :3], h, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
