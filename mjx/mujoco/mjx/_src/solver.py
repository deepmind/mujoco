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
"""Constraint solvers."""

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import math
from mujoco.mjx._src import smooth
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src.types import ConeType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DataJAX
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ModelJAX
from mujoco.mjx._src.types import OptionJAX
from mujoco.mjx._src.types import SolverType
# pylint: enable=g-importing-member


class Context(PyTreeNode):
  """Data updated during each solver iteration.

  Attributes:
    qacc: acceleration (from Data)                    (nv,)
    qfrc_constraint: constraint force (from Data)     (nv,)
    Jaref: Jac*qacc - aref                            (nefc,)
    efc_force: constraint force in constraint space   (nefc,)
    Ma: M*qacc                                        (nv,)
    grad: gradient of master cost                     (nv,)
    Mgrad: M / grad                                   (nv,)
    search: linesearch vector                         (nv,)
    gauss: gauss Cost
    cost: constraint + Gauss cost
    prev_cost: cost from previous iter
    solver_niter: number of solver iterations
    active: active (quadratic) constraints            (nefc,)
    fri: friction of regularized cone                 (num(con.dim > 1), 6)
    dm: regularized constraint mass                   (num(con.dim > 1))
    u: friction cone (normal and tangents)            (num(con.dim > 1), 6)
    h: cone hessian                                   (num(con.dim > 1), 6, 6)
  """

  qacc: jax.Array
  qfrc_constraint: jax.Array
  Jaref: jax.Array  # pylint: disable=invalid-name
  efc_force: jax.Array
  Ma: jax.Array  # pylint: disable=invalid-name
  grad: jax.Array
  Mgrad: jax.Array  # pylint: disable=invalid-name
  search: jax.Array
  gauss: jax.Array
  cost: jax.Array
  prev_cost: jax.Array
  solver_niter: jax.Array
  active: jax.Array
  fri: jax.Array
  dm: jax.Array
  u: jax.Array
  h: jax.Array

  @classmethod
  def create(cls, m: Model, d: Data, grad: bool = True) -> 'Context':
    if not isinstance(d._impl, DataJAX) or not isinstance(
        m.opt._impl, OptionJAX
    ):
      raise ValueError(
          'Constraint context requires JAX backend implementation.'
      )

    jaref = d._impl.efc_J @ d.qacc - d._impl.efc_aref
    # TODO(robotics-team): determine nv at which sparse mul is faster
    ma = support.mul_m(m, d, d.qacc)
    nv_0 = jp.zeros(m.nv)
    fri = 0.0
    if m.opt.cone == ConeType.ELLIPTIC:
      friction = d._impl.contact.friction[d._impl.contact.dim > 1]
      dim = d._impl.contact.dim[d._impl.contact.dim > 1]
      mu = friction[:, 0] / jp.sqrt(m.opt.impratio)
      fri = jp.concatenate((mu[:, None], friction), axis=1)
      for condim in (3, 4, 6):
        fri = fri.at[dim == condim, condim:].set(0)

    ctx = Context(
        qacc=d.qacc,
        qfrc_constraint=d.qfrc_constraint,
        Jaref=jaref,
        efc_force=d._impl.efc_force,
        Ma=ma,
        grad=nv_0,
        Mgrad=nv_0,
        search=nv_0,
        gauss=0.0,
        cost=jp.inf,
        prev_cost=0.0,
        solver_niter=0,
        active=0.0,
        fri=fri,
        dm=0.0,
        u=0.0,
        h=0.0,
    )
    ctx = _update_constraint(m, d, ctx)
    if grad:
      ctx = _update_gradient(m, d, ctx)
      ctx = ctx.replace(search=-ctx.Mgrad)  # start with preconditioned gradient

    return ctx


class _LSPoint(PyTreeNode):
  """Line search evaluation point.

  Attributes:
    alpha: step size that reduces f(x + alpha * p) given search direction p
    cost: line search cost
    deriv_0: first derivative of quadratic
    deriv_1: second derivative of quadratic
  """

  alpha: jax.Array
  cost: jax.Array
  deriv_0: jax.Array
  deriv_1: jax.Array

  @classmethod
  def create(
      cls,
      m: Model,
      d: Data,
      ctx: Context,
      alpha: jax.Array,
      jv: jax.Array,
      quad: jax.Array,
      quad_gauss: jax.Array,
      u_perp_sq: jax.Array,
      v0: jax.Array,
      u_par: jax.Array,
      v_norm: jax.Array,
  ) -> '_LSPoint':
    """Creates a linesearch point with first and second derivatives."""
    # roughly corresponds to CGEval in mujoco/src/engine/engine_solver.c
    if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
      raise ValueError('LSPoint requires JAX backend implementation.')

    cost, deriv_0, deriv_1 = 0.0, 0.0, 0.0
    quad_total = quad_gauss
    x = ctx.Jaref + alpha * jv
    active = (x < 0).at[: d._impl.ne + d._impl.nf].set(True)

    dof_fl, ten_fl = m._impl.dof_hasfrictionloss, m._impl.tendon_hasfrictionloss
    if (dof_fl.any() or ten_fl.any()) and not (
        m.opt.disableflags & DisableBit.FRICTIONLOSS
    ):
      f = d._impl.efc_frictionloss
      r = 1.0 / (d._impl.efc_D + (d._impl.efc_D == 0.0) * mujoco.mjMINVAL)
      rf, z = r * f, jp.zeros_like(f)
      linear_neg = (x <= -rf)[:, None]
      linear_pos = (x >= rf)[:, None]
      qf = linear_neg * jp.array([f * (-0.5 * rf - ctx.Jaref), -f * jv, z]).T
      qf += linear_pos * jp.array([f * (-0.5 * rf + ctx.Jaref), f * jv, z]).T
      quad = jp.where((linear_neg | linear_pos) & (f[:, None] > 0), qf, quad)

    if m.opt.cone == ConeType.ELLIPTIC:
      mu, u0 = ctx.fri[:, 0], ctx.u[:, 0]
      n = u0 + alpha * v0
      w_par = u_par + alpha * v_norm
      tsqr = jp.square(w_par) + u_perp_sq
      t = jp.sqrt(tsqr)  # tangential force

      bottom_zone = ((tsqr <= 0) & (n < 0)) | ((tsqr > 0) & ((mu * n + t) <= 0))
      middle_zone = (tsqr > 0) & (n < (mu * t)) & ((mu * n + t) > 0)

      # quadratic cost for equality, friction, limits, frictionless contacts
      dim1 = d._impl.contact.efc_address[d._impl.contact.dim == 1]
      nefl = d._impl.ne + d._impl.nf + d._impl.nl
      active = active.at[nefl:].set(False).at[dim1].set(active[dim1])
      quad_efld = jax.vmap(jp.multiply)(quad, active)
      quad_total += jp.sum(quad_efld, axis=0)
      # elliptic bottom zone: quadratic cost
      efc_elliptic = d._impl.contact.efc_address[d._impl.contact.dim > 1]
      quad_c = jax.vmap(jp.multiply)(quad[efc_elliptic], bottom_zone)
      quad_total += jp.sum(quad_c, axis=0)
      # elliptic middle zone
      t += (t == 0) * mujoco.mjMINVAL
      tsqr += (tsqr == 0) * mujoco.mjMINVAL
      n1 = v0
      t1 = w_par * v_norm / t
      # squared norm of (unit tangent) ^ v, folded with 1/T into (N/T - mu)
      det = (u_perp_sq / tsqr) * jp.square(v_norm)
      dm = ctx.dm * middle_zone
      nmt = n - mu * t
      cost = 0.5 * jp.sum(dm * jp.square(nmt))
      deriv_0 = jp.sum(dm * nmt * (n1 - mu * t1))
      deriv_1 = jp.sum(
          dm * (jp.square(n1 - mu * t1) + (n / t - mu) * (-mu * det))
      )
    elif m.opt.cone == ConeType.PYRAMIDAL:
      quad = jax.vmap(jp.multiply)(quad, active)  # only active
      quad_total += jp.sum(quad, axis=0)
    else:
      raise NotImplementedError(f'unsupported cone type: {m.opt.cone}')

    alpha_sq = alpha * alpha
    cost += alpha_sq * quad_total[2] + alpha * quad_total[1] + quad_total[0]
    deriv_0 += 2 * alpha * quad_total[2] + quad_total[1]
    deriv_1 += 2 * quad_total[2] + (quad_total[2] == 0) * mujoco.mjMINVAL

    return _LSPoint(alpha=alpha, cost=cost, deriv_0=deriv_0, deriv_1=deriv_1)


class _LSContext(PyTreeNode):
  """Data updated during each line search iteration.

  Attributes:
    lo: low point bounding the line search interval
    hi: high point bounding the line search interval
    swap: True if low or hi was swapped in the line search iteration
    ls_iter: number of linesearch iterations
  """

  lo: _LSPoint
  hi: _LSPoint
  swap: jax.Array
  ls_iter: jax.Array


def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""

  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]


def _update_constraint(m: Model, d: Data, ctx: Context) -> Context:
  """Updates constraint force and resulting cost given last solver iteration.

  Corresponds to CGupdateConstraint in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest qacc and smooth terms
    ctx: current solver context

  Returns:
    context with new constraint force and costs
  """
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('_update_constraint requires JAX backend implementation.')

  # ne constraints are always active, nf are conditionally active, others are
  # non-negative constraints.
  active = (ctx.Jaref < 0).at[: d._impl.ne + d._impl.nf].set(True)

  floss_force, floss_cost = jp.zeros(d._impl.nefc), 0.0
  dof_fl, ten_fl = m._impl.dof_hasfrictionloss, m._impl.tendon_hasfrictionloss
  if (dof_fl.any() or ten_fl.any()) and not (
      m.opt.disableflags & DisableBit.FRICTIONLOSS
  ):
    f = d._impl.efc_frictionloss
    r = 1.0 / (d._impl.efc_D + (d._impl.efc_D == 0.0) * mujoco.mjMINVAL)
    linear_neg = (ctx.Jaref <= -r * f) * (f > 0)
    linear_pos = (ctx.Jaref >= r * f) * (f > 0)
    active = active & ~linear_neg & ~linear_pos
    floss_force = linear_neg * f + linear_pos * -f
    floss_cost = linear_neg * (-0.5 * r * f * f - f * ctx.Jaref)
    floss_cost += linear_pos * (-0.5 * r * f * f + f * ctx.Jaref)
    floss_cost = floss_cost.sum()

  if m.opt.cone == ConeType.PYRAMIDAL:
    efc_force = d._impl.efc_D * -ctx.Jaref * active + floss_force
    cost = 0.5 * jp.sum(d._impl.efc_D * ctx.Jaref * ctx.Jaref * active)
    dm, u, h = 0.0, 0.0, 0.0
  elif m.opt.cone == ConeType.ELLIPTIC:
    friction = d._impl.contact.friction[d._impl.contact.dim > 1]
    efc_address = d._impl.contact.efc_address[d._impl.contact.dim > 1]
    dim = d._impl.contact.dim[d._impl.contact.dim > 1]
    # to prevent out of range append zeros to ctx.Jaref
    slice_fn = jax.vmap(
        lambda x: jax.lax.dynamic_slice(
            jp.concatenate((ctx.Jaref, jp.zeros((3)))), (x,), (6,)
        )
    )
    u = slice_fn(efc_address) * ctx.fri
    mu, n = ctx.fri[:, 0], u[:, 0]
    # exact tangent norm, gradient-safe at 0 (math.norm flushes |x| <= 1e-8)
    tsqr = jp.sum(jp.square(u[:, 1:]), axis=1)
    t = jp.where(tsqr > 0, jp.sqrt(jp.where(tsqr > 0, tsqr, 1.0)), 0.0)

    # bottom zone: quadratic
    bottom_zone = ((t <= 0) & (n < 0)) | ((t > 0) & ((mu * n + t) <= 0))
    adr_i, adr_j = [], []
    for i, (condim, addr) in enumerate(zip(dim, efc_address)):
      adr_i.extend(range(addr, addr + condim))
      adr_j.extend([i] * condim)
    active = active.at[jp.array(adr_i)].set(bottom_zone[jp.array(adr_j)])
    efc_force = d._impl.efc_D * -ctx.Jaref * active + floss_force
    cost = 0.5 * jp.sum(d._impl.efc_D * ctx.Jaref * ctx.Jaref * active)

    # middle zone: cone
    middle_zone = (t > 0) & (n < (mu * t)) & ((mu * n + t) > 0)
    dm = d._impl.efc_D[efc_address] / jp.maximum(
        mu * mu * (1 + mu * mu), mujoco.mjMINVAL
    )
    nmt = n - mu * t
    cost += 0.5 * jp.sum(dm * nmt * nmt * middle_zone)
    # tangent and friction for middle zone:
    force = -dm * nmt * mu * middle_zone
    force_fri = -force / (t + ~middle_zone * mujoco.mjMINVAL)
    force_fri = force_fri[:, None] * u[:, 1:] * friction
    efc_force = efc_force.at[efc_address].add(force)
    efc_adr, adr_i, adr_j = [], [], []
    for i, (condim, addr) in enumerate(zip(dim, efc_address)):
      efc_adr.extend(range(addr + 1, addr + condim))
      adr_i.extend([i] * (condim - 1))
      adr_j.extend(range(condim - 1))
    efc_adr, adr_i, adr_j = jp.array(efc_adr), jp.array(adr_i), jp.array(adr_j)
    efc_force = efc_force.at[efc_adr].add(force_fri[(adr_i, adr_j)])

    # cone hessian
    h = 0.0
    if m.opt.solver == SolverType.NEWTON:
      t_safe = jp.where(middle_zone, t, 1.0)
      # unit tangent: normalize before forming products so T^3 never appears
      uhat = u / t_safe[:, None]
      # h = (mu*N/T) * Uhat*Uhat'
      h = jax.vmap(lambda x, y: x * jp.outer(y, y.T))(mu * n / t_safe, uhat)
      # add to diagonal: (mu^2 - mu*N/T) * I
      h += jax.vmap(lambda x: x * jp.eye(6, 6))(mu * mu - mu * n / t_safe)
      # set first row: (1, -mu * Uhat)
      h_0 = jax.vmap(lambda mu, uhat: jp.append(1, -mu * uhat[1:]))(mu, uhat)
      h = h.at[:, 0].set(h_0).at[:, :, 0].set(h_0)
      # pre and post multiply by diag(mu, friction), scale by Dm
      h *= jax.vmap(lambda d, f: d * jp.outer(f, f.T))(dm, ctx.fri)
      # only cone constraints
      h = jp.where(middle_zone[:, None, None], h, 0.0)
  else:
    raise NotImplementedError(f'unsupported cone type: {m.opt.cone}')

  qfrc_constraint = d._impl.efc_J.T @ efc_force
  gauss = 0.5 * jp.dot(ctx.Ma - d.qfrc_smooth, ctx.qacc - d.qacc_smooth)
  ctx = ctx.replace(
      qfrc_constraint=qfrc_constraint,
      gauss=gauss,
      cost=cost + gauss + floss_cost,
      prev_cost=ctx.cost,
      efc_force=efc_force,
      active=active,
      dm=dm,
      u=u,
      h=h,
  )

  return ctx


def _update_gradient(m: Model, d: Data, ctx: Context) -> Context:
  """Updates grad and M / grad given latest solver iteration.

  Corresponds to CGupdateGradient in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest smooth terms
    ctx: current solver context

  Returns:
    context with new grad and M / grad
  Raises:
    NotImplementedError: for unsupported solver type
  """
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('_update_gradient requires JAX backend implementation.')

  grad = ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint

  if m.opt.solver == SolverType.CG:
    mgrad = smooth.solve_m(m, d, grad)
  elif m.opt.solver == SolverType.NEWTON:
    if m.opt.cone == ConeType.ELLIPTIC:
      cm = jp.diag(d._impl.efc_D * ctx.active)
      efc_address = d._impl.contact.efc_address[d._impl.contact.dim > 1]
      dim = d._impl.contact.dim[d._impl.contact.dim > 1]
      # set efc of cone H along diagonal
      for i, (condim, addr) in enumerate(zip(dim, efc_address)):
        h_cone = ctx.h[i, :condim, :condim]
        cm = cm.at[addr : addr + condim, addr : addr + condim].add(h_cone)
      h = d._impl.efc_J.T @ cm @ d._impl.efc_J
    else:
      h = (d._impl.efc_J.T * d._impl.efc_D * ctx.active) @ d._impl.efc_J
    h = support.full_m(m, d) + h
    # Symmetrize to reduce the chance of numerical issues in cholesky.
    h_sym = (h + h.T) * 0.5
    h_ = jax.scipy.linalg.cho_factor(h_sym)
    mgrad = jax.scipy.linalg.cho_solve(h_, grad)
  else:
    raise NotImplementedError(f'unsupported solver type: {m.opt.solver}')

  ctx = ctx.replace(grad=grad, Mgrad=mgrad)

  return ctx


def _trace_m(m: Model, d: Data) -> jax.Array:
  if not isinstance(d._impl, DataJAX):
    raise ValueError('_trace_m requires JAX backend implementation.')
  if not m.nv:
    return jp.array(mujoco.mjMINVAL)
  trace = (
      d._impl.M[m.M_rowadr + m.M_rownnz - 1].sum()
      if support.is_sparse(m)
      else jp.trace(d._impl.M)
  )
  return jp.maximum(mujoco.mjMINVAL, trace)


def _rescale(m: Model, d: Data, value: jax.Array) -> jax.Array:
  return value / _trace_m(m, d)


def _linesearch(m: Model, d: Data, ctx: Context) -> Context:
  """Performs a zoom linesearch to find optimal search step size.

  Args:
    m: model defining search options and other needed terms
    d: data with inertia matrix and other needed terms
    ctx: current solver context

  Returns:
    updated context with new qacc, Ma, Jaref
  """
  if (
      not isinstance(m._impl, ModelJAX)
      or not isinstance(d._impl, DataJAX)
      or not isinstance(m.opt._impl, OptionJAX)
  ):
    raise ValueError('_lineasearch requires JAX backend implementation.')

  smag = math.norm(ctx.search) * _trace_m(m, d)
  gtol = m.opt.tolerance * m.opt.ls_tolerance * smag

  # compute Mv, Jv
  mv = support.mul_m(m, d, ctx.search)
  jv = d._impl.efc_J @ ctx.search

  # prepare quadratics
  quad_gauss = jp.stack((
      ctx.gauss,
      jp.dot(ctx.search, ctx.Ma) - jp.dot(ctx.search, d.qfrc_smooth),
      0.5 * jp.dot(ctx.search, mv),
  ))
  quad = jp.stack((0.5 * ctx.Jaref * ctx.Jaref, jv * ctx.Jaref, 0.5 * jv * jv))
  quad = (quad * d._impl.efc_D).T
  u_perp_sq, v0, u_par, v_norm = 0.0, 0.0, 0.0, 0.0
  if m.opt.cone == ConeType.ELLIPTIC:
    mask = d._impl.contact.dim > 1
    # complete vector quadratic (for bottom zone)
    efc_con, efc_fri = [], []
    for condim, addr in zip(
        d._impl.contact.dim[mask], d._impl.contact.efc_address[mask]
    ):
      efc_con.extend([addr] * (condim - 1))
      efc_fri.extend(range(addr + 1, addr + condim))
    quad = quad.at[jp.array(efc_con)].add(quad[jp.array(efc_fri)])

    # rescale to make primal cone circular
    # to prevent out of range append zeros to jv
    jv_fn = jax.vmap(
        lambda x: jax.lax.dynamic_slice(
            jp.concatenate((jv, jp.zeros(3))), (x,), (6,)
        )
    )
    efc_elliptic = d._impl.contact.efc_address[mask]
    v = jv_fn(efc_elliptic) * ctx.fri
    u_tan, v_tan = ctx.u[:, 1:], v[:, 1:]
    uu = jp.sum(u_tan * u_tan, axis=1)
    v0 = v[:, 0]
    uv = jp.sum(u_tan * v_tan, axis=1)
    vv = jp.sum(v_tan * v_tan, axis=1)

    # decompose tangential u into components parallel and orthogonal to v
    wedge = (
        u_tan[:, :, None] * v_tan[:, None, :]
        - u_tan[:, None, :] * v_tan[:, :, None]
    )
    det = 0.5 * jp.sum(jp.square(wedge), axis=(1, 2))
    v_pos = vv > 0
    vv_safe = jp.where(v_pos, vv, 1.0)
    v_norm = jp.where(v_pos, jp.sqrt(vv_safe), 0.0)
    v_norm_safe = jp.where(v_pos, v_norm, 1.0)
    u_par = jp.where(v_pos, uv / v_norm_safe, 0.0)
    u_perp_sq = jp.where(v_pos, det / vv_safe, uu)

  point_fn = lambda a: _LSPoint.create(
      m, d, ctx, a, jv, quad, quad_gauss, u_perp_sq, v0, u_par, v_norm  # pyrefly: ignore[bad-argument-type]
  )
  # initialize interval
  p0 = point_fn(jp.array(0.0))

  # set acceptance tolerance to avoid exceeding gtol in f32: Cauchy-Schwarz
  # bound on the derivative magnitude of the active constraint rows at alpha = 0
  rows_0 = jp.maximum(p0.cost - quad_gauss[0], 0.0)
  rows_2 = jp.maximum(0.5 * p0.deriv_1 - quad_gauss[2], 0.0)
  dmag = 2.0 * jp.sqrt(rows_0 * rows_2) + jp.abs(quad_gauss[1])
  gtol_accept = jp.maximum(gtol, 8 * jp.finfo(dmag.dtype).eps * dmag)

  def cond(ctx: _LSContext) -> jax.Array:
    done = ctx.ls_iter >= m.opt.ls_iterations
    done |= ~ctx.swap  # if we did not adjust the interval
    done |= (jp.abs(ctx.lo.deriv_0) < gtol_accept) & (ctx.lo.cost < p0.cost)
    done |= (jp.abs(ctx.hi.deriv_0) < gtol_accept) & (ctx.hi.cost < p0.cost)

    return ~done

  def body(ctx: _LSContext) -> _LSContext:
    # always compute new bracket boundaries and a midpoint
    lo, hi = ctx.lo, ctx.hi
    lo_next = point_fn(lo.alpha - lo.deriv_0 / lo.deriv_1)
    hi_next = point_fn(hi.alpha - hi.deriv_0 / hi.deriv_1)
    mid = point_fn(0.5 * (lo.alpha + hi.alpha))

    # swap lo/hi if the derivative points to a narrower bracket width
    in_bracket = lambda x, y: ((x < y) & (y < 0) | (x > y) & (y > 0))
    swap_lo_next = in_bracket(lo.deriv_0, lo_next.deriv_0)
    lo = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_lo_next, y, x), lo, lo_next
    )
    swap_lo_mid = in_bracket(lo.deriv_0, mid.deriv_0)
    lo = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_lo_mid, y, x), lo, mid
    )
    swap_lo_hi_next = in_bracket(lo.deriv_0, hi_next.deriv_0)
    lo = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_lo_hi_next, y, x), lo, hi_next
    )
    # also accept a Newton step that crosses the minimizer from an undershooting
    # hi, turning the one-sided search into an opposite-sign bracket
    swap_hi_next = in_bracket(hi.deriv_0, hi_next.deriv_0) | (
        (hi.deriv_0 < 0) & (hi_next.deriv_0 > 0)
    )
    hi = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_hi_next, y, x), hi, hi_next
    )
    swap_hi_mid = in_bracket(hi.deriv_0, mid.deriv_0)
    hi = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_hi_mid, y, x), hi, mid
    )
    swap_hi_lo_next = in_bracket(hi.deriv_0, lo_next.deriv_0)
    hi = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_hi_lo_next, y, x), hi, lo_next
    )
    swap = swap_lo_next | swap_lo_mid | swap_lo_hi_next
    swap = swap | swap_hi_next | swap_hi_mid | swap_hi_lo_next

    # accept the lowest-cost converged candidate regardless of derivative sign
    cand = jax.tree_util.tree_map(lambda *x: jp.stack(x), lo_next, hi_next, mid)
    converged = (jp.abs(cand.deriv_0) < gtol_accept) & (cand.cost < p0.cost)
    best = jp.argmin(jp.where(converged, cand.cost, jp.inf))
    best = jax.tree_util.tree_map(lambda x: x[best], cand)
    done = converged.any()
    lo = jax.tree_util.tree_map(lambda x, y: jp.where(done, y, x), lo, best)
    hi = jax.tree_util.tree_map(lambda x, y: jp.where(done, y, x), hi, best)
    swap = swap & ~done

    ctx = ctx.replace(lo=lo, hi=hi, swap=swap, ls_iter=ctx.ls_iter + 1)

    return ctx

  lo = point_fn(p0.alpha - p0.deriv_0 / p0.deriv_1)
  lesser_fn = lambda x, y: jp.where(lo.deriv_0 < p0.deriv_0, x, y)
  hi = jax.tree_util.tree_map(lesser_fn, p0, lo)
  lo = jax.tree_util.tree_map(lesser_fn, lo, p0)
  ls_ctx = _LSContext(lo=lo, hi=hi, swap=jp.array(True), ls_iter=0)
  ls_ctx = _while_loop_scan(cond, body, ls_ctx, m.opt.ls_iterations)

  # move to new solution if improved
  lo, hi = ls_ctx.lo, ls_ctx.hi
  improved = (lo.cost < p0.cost) | (hi.cost < p0.cost)
  alpha = jp.where(lo.cost < hi.cost, lo.alpha, hi.alpha)
  qacc = ctx.qacc + improved * ctx.search * alpha
  ma = ctx.Ma + improved * mv * alpha
  jaref = ctx.Jaref + improved * jv * alpha

  ctx = ctx.replace(qacc=qacc, Ma=ma, Jaref=jaref)

  return ctx


def solve(m: Model, d: Data) -> Data:
  """Finds forces that satisfy constraints using conjugate gradient descent."""
  if not isinstance(m.opt._impl, OptionJAX):
    raise ValueError('solve requires JAX backend implementation.')

  def cond(ctx: Context) -> jax.Array:
    improvement = _rescale(m, d, ctx.prev_cost - ctx.cost)
    gradient = _rescale(m, d, math.norm(ctx.grad))

    done = ctx.solver_niter >= m.opt.iterations
    done |= improvement < m.opt.tolerance
    done |= gradient < m.opt.tolerance

    return ~done

  def body(ctx: Context) -> Context:
    ctx = _linesearch(m, d, ctx)
    prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad  # pylint: disable=invalid-name
    ctx = _update_constraint(m, d, ctx)
    ctx = _update_gradient(m, d, ctx)

    if m.opt.solver == SolverType.NEWTON:
      search = -ctx.Mgrad
    else:
      # polak-ribiere:
      beta = jp.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
      beta = beta / jp.maximum(mujoco.mjMINVAL, jp.dot(prev_grad, prev_Mgrad))
      beta = jp.maximum(0, beta)
      search = -ctx.Mgrad + beta * ctx.search
    ctx = ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

    return ctx

  # warmstart:
  qacc = d.qacc_smooth
  if not m.opt.disableflags & DisableBit.WARMSTART:
    warm = Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
    smth = Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
    qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
  d = d.replace(qacc=qacc)

  ctx = Context.create(m, d)
  if m.opt.iterations == 1:
    ctx = body(ctx)
  else:
    ctx = jax.lax.while_loop(cond, body, ctx)

  d = d.tree_replace({  # pyrefly: ignore[bad-assignment]
      'qfrc_constraint': ctx.qfrc_constraint,
      'qacc': ctx.qacc,
      '_impl.efc_force': ctx.efc_force,
  })

  return d
