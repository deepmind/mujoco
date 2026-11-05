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
"""Core smooth dynamics functions."""

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import math
from mujoco.mjx._src import scan
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import CamLightType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DataJAX
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import EqType
from mujoco.mjx._src.types import Impl
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ModelJAX
from mujoco.mjx._src.types import ObjType
from mujoco.mjx._src.types import TendonType
from mujoco.mjx._src.types import TrnType
from mujoco.mjx._src.types import WrapType
# pylint: enable=g-importing-member
import mujoco.mjx.warp as mjxw
import numpy as np


def kinematics(m: Model, d: Data) -> Data:
  """Converts position/velocity from generalized coordinates to maximal."""
  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
    from mujoco.mjx.warp import smooth as mjxw_smooth  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
    return mjxw_smooth.kinematics(m, d)

  def fn(carry, jnt_typs, jnt_pos, jnt_axis, qpos, qpos0, pos, quat):
    # calculate joint anchors, axes, body pos and quat in global frame
    # also normalize qpos while we're at it

    if carry is not None:
      _, _, _, parent_pos, parent_quat, _ = carry
      pos = parent_pos + math.rotate(pos, parent_quat)
      quat = math.quat_mul(parent_quat, quat)

    anchors, axes = [], []

    qpos_i = 0
    for i, jnt_typ in enumerate(jnt_typs):
      if jnt_typ == JointType.FREE:
        anchor, axis = qpos[qpos_i : qpos_i + 3], jp.array([0.0, 0.0, 1.0])
      else:
        anchor = math.rotate(jnt_pos[i], quat) + pos
        axis = math.rotate(jnt_axis[i], quat)
      anchors, axes = anchors + [anchor], axes + [axis]

      if jnt_typ == JointType.FREE:
        pos = qpos[qpos_i : qpos_i + 3]
        quat = math.normalize(qpos[qpos_i + 3 : qpos_i + 7])
        qpos = qpos.at[qpos_i + 3 : qpos_i + 7].set(quat)
        qpos_i += 7
      elif jnt_typ == JointType.BALL:
        qloc = math.normalize(qpos[qpos_i : qpos_i + 4])
        qpos = qpos.at[qpos_i : qpos_i + 4].set(qloc)
        quat = math.quat_mul(quat, qloc)
        pos = anchor - math.rotate(jnt_pos[i], quat)  # off-center rotation
        qpos_i += 4
      elif jnt_typ == JointType.HINGE:
        angle = qpos[qpos_i] - qpos0[qpos_i]
        qloc = math.axis_angle_to_quat(jnt_axis[i], angle)
        quat = math.quat_mul(quat, qloc)
        pos = anchor - math.rotate(jnt_pos[i], quat)  # off-center rotation
        qpos_i += 1
      elif jnt_typ == JointType.SLIDE:
        pos += axis * (qpos[qpos_i] - qpos0[qpos_i])
        qpos_i += 1
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

    anchor = jp.stack(anchors) if anchors else jp.empty((0, 3))
    axis = jp.stack(axes) if axes else jp.empty((0, 3))
    mat = math.quat_to_mat(quat)

    return qpos, anchor, axis, pos, quat, mat

  qpos, xanchor, xaxis, xpos, xquat, xmat = scan.body_tree(
      m,
      fn,
      'jjjqqbb',
      'qjjbbb',
      m.jnt_type,
      m.jnt_pos,
      m.jnt_axis,
      d.qpos,
      m.qpos0,
      m.body_pos,
      m.body_quat,
  )

  if m.nmocap:
    xpos = xpos.at[m.body_mocapid >= 0].set(d.mocap_pos)
    mocap_quat = jax.vmap(math.normalize)(d.mocap_quat)
    xquat = xquat.at[m.body_mocapid >= 0].set(mocap_quat)
    xmat = xmat.at[m.body_mocapid >= 0].set(
        jax.vmap(math.quat_to_mat)(mocap_quat)
    )

  v_local_to_global = jax.vmap(support.local_to_global)

  # TODO(erikfrey): confirm that quats are more performant for mjx than mats
  xipos, ximat = v_local_to_global(xpos, xquat, m.body_ipos, m.body_iquat)
  d = d.replace(qpos=qpos, xanchor=xanchor, xaxis=xaxis, xpos=xpos)
  d = d.replace(xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)

  if m.ngeom:
    geom_xpos, geom_xmat = v_local_to_global(
        xpos[m.geom_bodyid], xquat[m.geom_bodyid], m.geom_pos, m.geom_quat
    )
    d = d.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)

  if m.nsite:
    site_xpos, site_xmat = v_local_to_global(
        xpos[m.site_bodyid], xquat[m.site_bodyid], m.site_pos, m.site_quat
    )
    d = d.replace(site_xpos=site_xpos, site_xmat=site_xmat)

  return d


def com_pos(m: Model, d: Data) -> Data:
  """Maps inertias and motion dofs to global frame centered at subtree-CoM."""
  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
    from mujoco.mjx.warp import smooth as mjxw_smooth  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

    return mjxw_smooth.com_pos(m, d)

  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('com_pos requires JAX backend implementation.')

  # calculate center of mass of each subtree
  def subtree_sum(carry, xipos, body_mass):
    pos, mass = xipos * body_mass, body_mass
    if carry is not None:
      subtree_pos, subtree_mass = carry
      pos, mass = pos + subtree_pos, mass + subtree_mass
    return pos, mass

  pos, mass = scan.body_tree(
      m, subtree_sum, 'bb', 'bb', d.xipos, m.body_mass, reverse=True
  )
  cond = jp.tile(mass < mujoco.mjMINVAL, (3, 1)).T
  # take maximum to avoid NaN in gradient of jp.where
  subtree_com = jax.vmap(jp.divide)(pos, jp.maximum(mass, mujoco.mjMINVAL))
  subtree_com = jp.where(cond, d.xipos, subtree_com)
  d = d.replace(subtree_com=subtree_com)

  # map inertias to frame centered at subtree_com
  @jax.vmap
  def inert_com(inert, ximat, off, mass):
    h = jp.cross(off, -jp.eye(3))
    inert = math.matmul_unroll((ximat * inert), ximat.T)
    inert += math.matmul_unroll(h, h.T) * mass
    # cinert is triu(inert), mass * off, mass
    inert = inert[([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])]
    return jp.concatenate([inert, off * mass, mass[None]])

  root_com = subtree_com[m.body_rootid]
  offset = d.xipos - root_com
  cinert = inert_com(m.body_inertia, d.ximat, offset, m.body_mass)
  d = d.tree_replace({'_impl.cinert': cinert})  # pyrefly: ignore[bad-assignment]

  # map motion dofs to global frame centered at subtree_com
  def cdof_fn(jnt_typs, root_com, xmat, xanchor, xaxis):
    cdofs = []

    dof_com_fn = lambda a, o: jp.concatenate([a, jp.cross(a, o)])

    for i, jnt_typ in enumerate(jnt_typs):
      offset = root_com - xanchor[i]
      if jnt_typ == JointType.FREE:
        cdofs.append(jp.eye(3, 6, 3))  # free translation
        cdofs.append(jax.vmap(dof_com_fn, in_axes=(0, None))(xmat.T, offset))
      elif jnt_typ == JointType.BALL:
        cdofs.append(jax.vmap(dof_com_fn, in_axes=(0, None))(xmat.T, offset))
      elif jnt_typ == JointType.HINGE:
        cdof = dof_com_fn(xaxis[i], offset)
        cdofs.append(jp.expand_dims(cdof, 0))
      elif jnt_typ == JointType.SLIDE:
        cdof = jp.concatenate((jp.zeros((3,)), xaxis[i]))
        cdofs.append(jp.expand_dims(cdof, 0))
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

    cdof = jp.concatenate(cdofs) if cdofs else jp.empty((0, 6))

    return cdof

  cdof = scan.flat(
      m,
      cdof_fn,
      'jbbjj',
      'v',
      m.jnt_type,
      root_com,
      d.xmat,
      d.xanchor,
      d.xaxis,
  )
  d = d.tree_replace({'cdof': cdof})  # pyrefly: ignore[bad-assignment]

  return d


def camlight(m: Model, d: Data) -> Data:
  """Computes camera and light positions and orientations."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('camlight requires JAX backend implementation.')

  if m.ncam == 0:
    return d.replace(cam_xpos=jp.zeros((0, 3)), cam_xmat=jp.zeros((0, 3, 3)))

  # use target body only if target body is specified
  is_target_cam = (m.cam_mode == CamLightType.TARGETBODY) | (
      m.cam_mode == CamLightType.TARGETBODYCOM
  )
  cam_mode = np.where(
      is_target_cam & (m.cam_targetbodyid < 0), CamLightType.FIXED, m.cam_mode
  )

  cam_xpos, cam_xmat = jax.vmap(support.local_to_global)(
      d.xpos[m.cam_bodyid], d.xquat[m.cam_bodyid], m.cam_pos, m.cam_quat
  )

  def fn(
      camid,
      cam_mode,
      cam_xpos,
      cam_xmat,
      body_xpos,
      subtree_com,
      target_body_xpos,
      target_subtree_com,
  ):
    if cam_mode == CamLightType.TRACK:
      cam_xmat = m.cam_mat0[camid]
      cam_xpos = body_xpos + m.cam_pos0[camid]
    elif cam_mode == CamLightType.TRACKCOM:
      cam_xmat = m.cam_mat0[camid]
      cam_xpos = subtree_com + m.cam_poscom0[camid]
    elif cam_mode in (CamLightType.TARGETBODY, CamLightType.TARGETBODYCOM):
      # get position to look at
      pos = target_body_xpos
      if cam_mode == CamLightType.TARGETBODYCOM:
        pos = target_subtree_com
      # zaxis = -desired camera direction, in global frame
      mat_3 = math.normalize(cam_xpos - pos)
      # xaxis: orthogonal to zaxis and to (0,0,1)
      mat_1 = math.normalize(jp.cross(jp.array([0.0, 0.0, 1.0]), mat_3))
      mat_2 = math.normalize(jp.cross(mat_3, mat_1))
      cam_xmat = jp.array([mat_1, mat_2, mat_3]).T
    return cam_xpos, cam_xmat

  cam_xpos, cam_xmat = scan.flat(
      m,
      fn,
      'c' * 8,
      'cc',
      jp.arange(m.ncam),
      cam_mode,
      cam_xpos,
      cam_xmat,
      d.xpos[m.cam_bodyid],
      d.subtree_com[m.cam_bodyid],
      d.xpos[m.cam_targetbodyid],
      d.subtree_com[m.cam_targetbodyid],
      group_by='c',
  )

  d = d.replace(
      cam_xpos=cam_xpos,
      cam_xmat=cam_xmat,
  )

  return d


def crb(m: Model, d: Data) -> Data:
  """Runs composite rigid body inertia algorithm."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('crb requires JAX backend implementation.')

  def crb_fn(crb_child, crb_body):
    if crb_child is not None:
      crb_body += crb_child
    return crb_body

  crb_body = scan.body_tree(m, crb_fn, 'b', 'b', d._impl.cinert, reverse=True)
  crb_body = crb_body.at[0].set(0.0)
  d = d.tree_replace({'_impl.crb': crb_body})  # pyrefly: ignore[bad-assignment]

  crb_dof = jp.take(crb_body, jp.array(m.dof_bodyid), axis=0)
  crb_cdof = jax.vmap(math.inert_mul)(crb_dof, d.cdof)
  qm = support.make_m(m, crb_cdof, d.cdof, m.dof_armature)
  d = d.tree_replace({'_impl.M': qm})  # pyrefly: ignore[bad-assignment]
  return d


def factor_m(m: Model, d: Data) -> Data:
  """Gets factorizaton of inertia-like matrix M, assumed spd."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('factor_m requires JAX backend implementation.')

  if not support.is_sparse(m):
    qh, _ = jax.scipy.linalg.cho_factor(d._impl.M)
    d = d.tree_replace({'_impl.qLD': qh})  # pyrefly: ignore[bad-assignment]
    return d

  depth = []
  for i in range(m.nv):
    depth.append(depth[m.dof_parentid[i]] + 1 if m.dof_parentid[i] != -1 else 0)
  updates = {}
  diag_ds = []
  for i in range(m.nv):
    diag_i = m.M_rowadr[i] + m.M_rownnz[i] - 1
    start_i = m.M_rowadr[i]
    for k in range(m.M_rownnz[i]):
      diag_ds.append(diag_i)
    for k in range(m.M_rownnz[i] - 1):
      adr = start_i + k
      j = m.M_colind[adr]
      out_beg = m.M_rowadr[j]
      width = m.M_rownnz[j]
      out_end = out_beg + width
      updates.setdefault(depth[j], []).append(
          (out_beg, out_end, diag_i, adr, start_i)
      )

  qld = d._impl.M

  for _, updates in sorted(updates.items(), reverse=True):
    rows = []
    madr_ijs = []
    pivots = []
    out = []

    for b, e, piv_i, madr_ij, start_i in updates:
      width = e - b
      rows.append(np.arange(start_i, start_i + width))
      madr_ijs.append(np.full((width,), madr_ij))
      pivots.append(np.full((width,), piv_i))
      out.append(np.arange(b, e))
    if out:
      rows = np.concatenate(rows)
      madr_ijs = np.concatenate(madr_ijs)
      pivots = np.concatenate(pivots)
      out = np.concatenate(out)

      qld = qld.at[out].add(-(qld[madr_ijs] / qld[pivots]) * qld[rows])

  diag_adr = m.M_rowadr + m.M_rownnz - 1
  qld_diag = qld[diag_adr]
  qld = (qld / qld[np.array(diag_ds)]).at[diag_adr].set(qld_diag)

  d = d.tree_replace({'_impl.qLD': qld, '_impl.qLDiagInv': 1 / qld_diag})  # pyrefly: ignore[bad-assignment]
  return d


def solve_m(m: Model, d: Data, x: jax.Array) -> jax.Array:
  """Computes sparse backsubstitution:  x = inv(L'*D*L)*y ."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('solve_m requires JAX backend implementation.')

  if not support.is_sparse(m):
    return jax.scipy.linalg.cho_solve((d._impl.qLD, False), x)

  depth = []
  for i in range(m.nv):
    depth.append(depth[m.dof_parentid[i]] + 1 if m.dof_parentid[i] != -1 else 0)

  updates_i, updates_j = {}, {}
  for i in range(m.nv):
    start_i = m.M_rowadr[i]
    for k in range(m.M_rownnz[i] - 1):
      adr = start_i + k
      j = m.M_colind[adr]
      updates_i.setdefault(depth[i], []).append((i, adr, j))
      updates_j.setdefault(depth[j], []).append((j, adr, i))

  # x <- inv(L') * x
  for _, vals in sorted(updates_j.items(), reverse=True):
    j, madr_ij, i = np.array(vals).T
    x = x.at[j].add(-d._impl.qLD[madr_ij] * x[i])

  # x <- inv(D) * x
  x = x * d._impl.qLDiagInv

  # x <- inv(L) * x
  for _, vals in sorted(updates_i.items()):
    i, madr_ij, j = np.array(vals).T
    x = x.at[i].add(-d._impl.qLD[madr_ij] * x[j])

  return x


def com_vel(m: Model, d: Data) -> Data:
  """Computes cvel, cdof_dot."""

  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('com_vel requires JAX backend implementation.')

  # forward scan down tree: accumulate link center of mass velocity
  def fn(parent, jnt_typs, cdof, qvel):
    cvel = jp.zeros((6,)) if parent is None else parent[0]

    cross_fn = jax.vmap(math.motion_cross, in_axes=(None, 0))
    cdof_x_qvel = jax.vmap(jp.multiply)(cdof, qvel)

    dof_beg = 0
    cdof_dots = []
    for jnt_typ in jnt_typs:
      dof_end = dof_beg + JointType(jnt_typ).dof_width()
      if jnt_typ == JointType.FREE:
        cvel += jp.sum(cdof_x_qvel[:3], axis=0)
        cdof_ang_dot = cross_fn(cvel, cdof[3:])
        cvel += jp.sum(cdof_x_qvel[3:], axis=0)
        cdof_dots.append(jp.concatenate((jp.zeros((3, 6)), cdof_ang_dot)))
      else:
        cdof_dots.append(cross_fn(cvel, cdof[dof_beg:dof_end]))
        cvel += jp.sum(cdof_x_qvel[dof_beg:dof_end], axis=0)
      dof_beg = dof_end

    cdof_dot = jp.concatenate(cdof_dots) if cdof_dots else jp.empty((0, 6))
    return cvel, cdof_dot

  cvel, cdof_dot = scan.body_tree(
      m,
      fn,
      'jvv',
      'bv',
      m.jnt_type,
      d.cdof,
      d.qvel,
  )

  d = d.tree_replace({'cvel': cvel, 'cdof_dot': cdof_dot})  # pyrefly: ignore[bad-assignment]

  return d


def subtree_vel(m: Model, d: Data) -> Data:
  """Subtree linear velocity and angular momentum."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('subtree_vel requires JAX backend implementation.')

  # bodywise quantities
  def _forward(cvel, xipos, ximat, subtree_com_root, mass, inertia):
    ang, lin = jp.split(cvel, 2)

    # update linear velocity
    lin = lin - jp.cross(xipos - subtree_com_root, ang)

    subtree_linvel = mass * lin
    subtree_angmom = inertia * ximat @ ximat.T @ ang
    body_vel = jp.concatenate([ang, lin])

    return body_vel, subtree_linvel, subtree_angmom

  body_vel, subtree_linvel, subtree_angmom = jax.vmap(_forward)(
      d.cvel,
      d.xipos,
      d.ximat,
      d.subtree_com[m.body_rootid],
      m.body_mass,
      m.body_inertia,
  )

  # sum body linear momentum recursively up the kinematic tree
  subtree_linvel = scan.body_tree(
      m,
      lambda x, y: y if x is None else x + y,
      'bb',
      'b',
      subtree_linvel,
      reverse=True,
  )

  subtree_linvel /= jp.maximum(mujoco.mjMINVAL, m.body_subtreemass)[:, None]

  def _subtree_angmom(
      carry,
      angmom,
      com,
      com_parent,
      linvel,
      linvel_parent,
      subtreemass,
      xipos,
      vel,
      mass,
      mask,
  ):

    def _momentum(x0, x1, v0, v1, m):
      dx = x0 - x1
      dv = v0 - v1
      dp = dv * m
      return jp.cross(dx, dp)

    # momentum wrt current body
    mom = mask * _momentum(xipos, com, vel[3:], linvel, mass)

    # momentum wrt parent
    mom_parent = mask * _momentum(
        com, com_parent, linvel, linvel_parent, subtreemass
    )

    if carry is None:
      return angmom + mom, mom_parent
    else:
      angmom_child, mom_parent_child = carry
      return angmom + mom + angmom_child + mom_parent_child, mom_parent

  subtree_angmom, _ = scan.body_tree(
      m,
      _subtree_angmom,
      'bbbbbbbbbb',
      'bb',
      subtree_angmom,
      d.subtree_com,
      d.subtree_com[m.body_parentid],
      subtree_linvel,
      subtree_linvel[m.body_parentid],
      m.body_subtreemass,
      d.xipos,
      body_vel,
      m.body_mass,
      jp.ones(m.nbody).at[0].set(0),
      reverse=True,
  )

  return d.tree_replace({  # pyrefly: ignore[bad-return]
      '_impl.subtree_linvel': subtree_linvel,
      '_impl.subtree_angmom': subtree_angmom,
  })


def rne(m: Model, d: Data, flg_acc: bool = False) -> Data:
  """Computes inverse dynamics using the recursive Newton-Euler algorithm.

  flg_acc=False removes inertial term.
  """
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('rne requires JAX backend implementation.')

  # forward scan over tree: accumulate link center of mass acceleration
  def cacc_fn(cacc, cdof_dot, qvel, cdof, qacc):
    if cacc is None:
      if m.opt.disableflags & DisableBit.GRAVITY:
        cacc = jp.zeros((6,))
      else:
        cacc = jp.concatenate((jp.zeros((3,)), -m.opt.gravity))

    cacc += jp.sum(jax.vmap(jp.multiply)(cdof_dot, qvel), axis=0)

    # cacc += cdof * qacc
    if flg_acc:
      cacc += jp.sum(jax.vmap(jp.multiply)(cdof, qacc), axis=0)

    return cacc

  cacc = scan.body_tree(
      m, cacc_fn, 'vvvv', 'b', d.cdof_dot, d.qvel, d.cdof, d.qacc
  )

  def frc(cinert, cacc, cvel):
    frc = math.inert_mul(cinert, cacc)
    frc += math.motion_cross_force(cvel, math.inert_mul(cinert, cvel))

    return frc

  loc_cfrc = jax.vmap(frc)(d._impl.cinert, cacc, d.cvel)

  # backward scan up tree: accumulate body forces
  def cfrc_fn(cfrc_child, cfrc):
    if cfrc_child is not None:
      cfrc += cfrc_child
    return cfrc

  cfrc = scan.body_tree(m, cfrc_fn, 'b', 'b', loc_cfrc, reverse=True)
  qfrc_bias = jax.vmap(jp.dot)(d.cdof, cfrc[jp.array(m.dof_bodyid)])

  d = d.replace(qfrc_bias=qfrc_bias)

  return d


def rne_postconstraint(m: Model, d: Data) -> Data:
  """RNE with complete data: compute cacc, cfrc_ext, cfrc_int."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('rne_postconstraint requires JAX backend implementation.')

  def _transform_force(frc, offset):
    force, torque = jp.split(frc, 2)
    torque -= jp.cross(offset, force)
    # spatial motion vector layout is flipped: (torque, force)
    return jp.concatenate([torque, force])

  # cfrc_ext = perturb
  cfrc_ext = jp.vstack([
      jp.zeros((1, 6)),  # world body
      jax.vmap(_transform_force)(
          d.xfrc_applied[1:], d.subtree_com[m.body_rootid][1:] - d.xipos[1:]
      ),
  ])

  # cfrc_ext += contacts

  # compute contact forces for each condim
  forces = []
  condim_idx = []
  for dim in set(d._impl.contact.dim):
    force, idx = support.contact_force_dim(m, d, dim)
    forces.append(force)
    condim_idx.append(idx)

  # update cfrc_ext with contact forces
  if forces:

    @jax.vmap
    def _contact_force_to_cfrc_ext(force, pos, frame, id1, id2, com1, com2):
      # force: contact to world frame
      force = force.reshape((-1, 3)) @ frame
      force = force.reshape(-1)

      # contact force on bodies
      cfrc_com1 = _transform_force(force, com1 - pos)
      cfrc_com2 = _transform_force(force, com2 - pos)

      # mask
      mask1 = id1 != 0
      mask2 = id2 != 0

      return jp.vstack([-1 * cfrc_com1 * mask1, cfrc_com2 * mask2]), jp.array(
          [id1, id2]
      )

    condim_idx = jp.concatenate(condim_idx)
    frame = d._impl.contact.frame[condim_idx]
    pos = d._impl.contact.pos[condim_idx]
    id1 = jp.array(m.geom_bodyid)[d._impl.contact.geom[condim_idx, 0]]
    id2 = jp.array(m.geom_bodyid)[d._impl.contact.geom[condim_idx, 1]]
    com1 = d.subtree_com[jp.array(m.body_rootid)][id1]
    com2 = d.subtree_com[jp.array(m.body_rootid)][id2]

    cfrc_contact, cfrc_idx = _contact_force_to_cfrc_ext(
        jp.concatenate(forces), pos, frame, id1, id2, com1, com2
    )

    cfrc_ext = cfrc_ext.at[cfrc_idx.reshape(-1)].add(
        cfrc_contact.reshape((-1, 6))
    )

  # cfrc_ext += connect, weld
  cfrc_ext_equality = []
  cfrc_ext_equality_adr = []

  connect_id = m.eq_type == EqType.CONNECT
  nconnect = connect_id.sum()

  if nconnect:
    cfrc_connect_force = d._impl.efc_force[: 3 * nconnect].reshape(
        (nconnect, 3)
    )

    is_site = m.eq_objtype == ObjType.SITE
    body1id = np.copy(m.eq_obj1id)
    body2id = np.copy(m.eq_obj2id)
    pos1 = m.eq_data[:, :3]
    pos2 = m.eq_data[:, 3:6]

    if m.nsite:
      body1id[is_site] = m.site_bodyid[m.eq_obj1id[is_site]]
      body2id[is_site] = m.site_bodyid[m.eq_obj2id[is_site]]
      pos1 = jp.where(is_site[:, None], m.site_pos[m.eq_obj1id], pos1)
      pos2 = jp.where(is_site[:, None], m.site_pos[m.eq_obj2id], pos2)

    # body 1
    k1_connect = body1id[connect_id]
    k1_connect_mask = k1_connect != 0
    offset1_connect = pos1[connect_id]

    pos1_connect = jax.vmap(lambda pnt, mat, vec: mat @ pnt + vec)(
        offset1_connect, d.xmat[k1_connect], d.xpos[k1_connect]
    )
    subtree_com1_connect = d.subtree_com[jp.array(m.body_rootid)[k1_connect]]
    cfrc_com1_connect = jax.vmap(
        lambda dif, frc, mask: mask * jp.concatenate([-jp.cross(dif, frc), frc])
    )(subtree_com1_connect - pos1_connect, cfrc_connect_force, k1_connect_mask)

    # body 2
    k2_connect = body2id[connect_id]
    k2_connect_mask = -1 * (k2_connect != 0)
    offset2_connect = pos2[connect_id]

    pos2_connect = jax.vmap(lambda pnt, mat, vec: mat @ pnt + vec)(
        offset2_connect, d.xmat[k2_connect], d.xpos[k2_connect]
    )
    subtree_com2_connect = d.subtree_com[jp.array(m.body_rootid)[k2_connect]]
    cfrc_com2_connect = jax.vmap(
        lambda dif, frc, mask: mask * jp.concatenate([-jp.cross(dif, frc), frc])
    )(subtree_com2_connect - pos2_connect, cfrc_connect_force, k2_connect_mask)

    cfrc_ext_equality.append(jp.vstack([cfrc_com1_connect, cfrc_com2_connect]))
    cfrc_ext_equality_adr.append(jp.concatenate([k1_connect, k2_connect]))

  weld_id = m.eq_type == EqType.WELD
  nweld = weld_id.sum()

  if nweld:
    cfrc_weld = d._impl.efc_force[
        3 * nconnect : 3 * nconnect + 6 * nweld
    ].reshape((nweld, 6))
    cfrc_weld_force = cfrc_weld[:, :3]
    cfrc_weld_torque = cfrc_weld[:, 3:]

    is_site = m.eq_objtype == ObjType.SITE
    body1id = np.copy(m.eq_obj1id)
    body2id = np.copy(m.eq_obj2id)
    pos1 = m.eq_data[:, 3:6]
    pos2 = m.eq_data[:, :3]

    if m.nsite:
      body1id[is_site] = m.site_bodyid[m.eq_obj1id[is_site]]
      body2id[is_site] = m.site_bodyid[m.eq_obj2id[is_site]]
      pos1 = jp.where(is_site[:, None], m.site_pos[m.eq_obj1id], pos1)
      pos2 = jp.where(is_site[:, None], m.site_pos[m.eq_obj2id], pos2)

    quat1 = jax.vmap(math.quat_mul)(d.xquat[body1id], m.eq_data[:, 6:10])
    quat2 = d.xquat[body2id]

    if m.nsite:
      quat1_site = jax.vmap(math.quat_mul)(
          d.xquat[body1id], m.site_quat[m.eq_obj1id]
      )
      quat2_site = jax.vmap(math.quat_mul)(
          d.xquat[body2id], m.site_quat[m.eq_obj2id]
      )
      quat1 = jp.where(is_site[:, None], quat1_site, quat1)
      quat2 = jp.where(is_site[:, None], quat2_site, quat2)

    # the rotational rows are 0.5*torquescale * neg(q2)*(jac1-jac2)*q1, so the
    # torque is the adjoint applied to the multiplier: 0.5*torquescale * q2*f*neg(q1)
    def _weld_torque(q1, q2, frc, torquescale):
      quat = math.quat_mul(math.quat_mul_axis(q2, frc), math.quat_inv(q1))
      return 0.5 * torquescale * quat[1:]

    cfrc_weld_torque = jax.vmap(_weld_torque)(
        quat1[weld_id],
        quat2[weld_id],
        cfrc_weld_torque,
        m.eq_data[weld_id, 10],
    )

    # body 1
    k1_weld = body1id[weld_id]
    k1_weld_mask = k1_weld != 0
    offset1_weld = pos1[weld_id]

    pos1_weld = jax.vmap(lambda pnt, mat, vec: mat @ pnt + vec)(
        offset1_weld, d.xmat[k1_weld], d.xpos[k1_weld]
    )
    subtree_com1_weld = d.subtree_com[jp.array(m.body_rootid)[k1_weld]]
    cfrc_com1_weld = jax.vmap(
        lambda dif, frc, trq, mask: mask
        * jp.concatenate([trq - jp.cross(dif, frc), frc])
    )(
        subtree_com1_weld - pos1_weld,
        cfrc_weld_force,
        cfrc_weld_torque,
        k1_weld_mask,
    )

    # body 2
    k2_weld = body2id[weld_id]
    k2_weld_mask = -1 * (k2_weld != 0)
    offset2_weld = pos2[weld_id]

    pos2_weld = jax.vmap(lambda pnt, mat, vec: mat @ pnt + vec)(
        offset2_weld, d.xmat[k2_weld], d.xpos[k2_weld]
    )
    subtree_com2_weld = d.subtree_com[jp.array(m.body_rootid)[k2_weld]]
    cfrc_com2_weld = jax.vmap(
        lambda dif, frc, trq, mask: mask
        * jp.concatenate([trq - jp.cross(dif, frc), frc])
    )(
        subtree_com2_weld - pos2_weld,
        cfrc_weld_force,
        cfrc_weld_torque,
        k2_weld_mask,
    )

    cfrc_ext_equality.append(jp.vstack([cfrc_com1_weld, cfrc_com2_weld]))
    cfrc_ext_equality_adr.append(jp.concatenate([k1_weld, k2_weld]))

  if nconnect or nweld:
    cfrc_ext = cfrc_ext.at[jp.concatenate(cfrc_ext_equality_adr)].add(
        jp.vstack(cfrc_ext_equality)
    )

  # forward pass over bodies: compute cacc, cfrc_int
  def _forward(carry, cfrc_ext, cinert, cvel, body_dofadr, body_dofnum):
    if carry is None:
      if m.opt.disableflags & DisableBit.GRAVITY:
        cacc0 = jp.zeros(6)
      else:
        cacc0 = jp.concatenate((jp.zeros(3), -m.opt.gravity))
      return cacc0, jp.zeros(6)
    else:
      cacc_parent, _ = carry

    # create dof mask
    indices = jp.arange(m.nv)
    mask = jp.logical_and(
        indices >= body_dofadr, indices < body_dofadr + body_dofnum
    )

    # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
    cacc_vel = d.cdof_dot.T @ (mask * d.qvel)
    cacc_acc = d.cdof.T @ (mask * d.qacc)
    cacc = cacc_parent + cacc_vel + cacc_acc

    # cfrc_body = cinert * cacc + cvel x (cinert * cvel)
    cfrc_body = math.inert_mul(cinert, cacc)
    cfrc_corr = math.inert_mul(cinert, cvel)
    cfrc = math.motion_cross_force(cvel, cfrc_corr)
    cfrc_body = cfrc_body + cfrc
    cfrc_int = cfrc_body - cfrc_ext

    return cacc, cfrc_int

  cacc, cfrc_int = scan.body_tree(
      m,
      _forward,
      'bbbbb',
      'bb',
      cfrc_ext,
      d._impl.cinert,
      d.cvel,
      jp.array(m.body_dofadr),
      jp.array(m.body_dofnum),
  )

  # backward pass over bodies: accumulate cfrc_int from children
  cfrc_int = scan.body_tree(
      m,
      lambda c, p: p + c if c is not None else p,  # add child to parent
      'b',
      'b',
      cfrc_int,
      reverse=True,
  )

  # update data
  return d.tree_replace({  # pyrefly: ignore[bad-return]
      '_impl.cacc': cacc,
      '_impl.cfrc_int': cfrc_int,
      '_impl.cfrc_ext': cfrc_ext,
  })


def tendon(m: Model, d: Data) -> Data:
  """Computes tendon lengths and moments."""
  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
    from mujoco.mjx.warp import smooth as mjxw_smooth  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
    return mjxw_smooth.tendon(m, d)

  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('tendon requires JAX backend implementation.')

  if not m.ntendon:
    return d

  # process joint tendons
  (wrap_id_jnt,) = np.nonzero(m.wrap_type == WrapType.JOINT)
  (tendon_id_jnt,) = np.nonzero(m.tendon_type == TendonType.FIXED)

  ntendon_jnt = tendon_id_jnt.size
  wrap_objid_jnt = m.wrap_objid[wrap_id_jnt]
  tendon_num_jnt = m.tendon_num[tendon_id_jnt]

  moment_jnt = m.wrap_prm[wrap_id_jnt]
  length_jnt = jax.ops.segment_sum(
      moment_jnt * d.qpos[m.jnt_qposadr[wrap_objid_jnt]],
      np.repeat(np.arange(ntendon_jnt), tendon_num_jnt),
      ntendon_jnt,
  )

  adr_moment_jnt = np.repeat(tendon_id_jnt, tendon_num_jnt)
  dofadr_moment_jnt = m.jnt_dofadr[wrap_objid_jnt]

  # process pulleys
  (wrap_id_pulley,) = np.nonzero(m.wrap_type == WrapType.PULLEY)
  nwrap_pulley = wrap_id_pulley.size

  tendon_wrapnum_pulley = np.array([
      sum((wrap_id_pulley >= adr) & (wrap_id_pulley < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])

  divisor = np.ones(m.nwrap)
  for adr, num in zip(m.tendon_adr, m.tendon_num):
    for id_pulley in wrap_id_pulley:
      if adr <= id_pulley < adr + num:
        divisor[id_pulley : adr + num] = np.maximum(
            mujoco.mjMINVAL, m.wrap_prm[id_pulley]
        )

  # process spatial tendon sites
  (wrap_id_site,) = np.nonzero(m.wrap_type == WrapType.SITE)
  nwrap_site = wrap_id_site.size

  # find consecutive sites, skipping tendon transitions
  (pair_id,) = np.nonzero(np.diff(wrap_id_site) == 1)
  wrap_id_site_pair = np.setdiff1d(wrap_id_site[pair_id], m.tendon_adr[1:] - 1)
  wrap_objid_site0 = m.wrap_objid[wrap_id_site_pair]
  wrap_objid_site1 = m.wrap_objid[wrap_id_site_pair + 1]

  @jax.vmap
  def _length_moment(pnt0, pnt1, body0, body1):
    dif = pnt1 - pnt0
    length = math.norm(dif)
    vec = jp.where(
        length < mujoco.mjMINVAL,
        jp.array([1.0, 0.0, 0.0]),
        math.safe_div(dif, length),
    )

    jacp1, _ = support.jac(m, d, pnt0, body0)
    jacp2, _ = support.jac(m, d, pnt1, body1)
    jacdif = jacp2 - jacp1
    moment = jp.where(body0 != body1, jacdif @ vec, jp.zeros(m.nv))

    return length, moment

  lengths_site, moments_site = _length_moment(
      d.site_xpos[wrap_objid_site0],
      d.site_xpos[wrap_objid_site1],
      m.site_bodyid[wrap_objid_site0],
      m.site_bodyid[wrap_objid_site1],
  )

  if wrap_id_site_pair.size:
    divisor_site_pair = divisor[wrap_id_site_pair]
    lengths_site /= divisor_site_pair
    moments_site /= divisor_site_pair[:, None]

  tendon_nsite = np.array([
      sum((wrap_id_site_pair >= adr) & (wrap_id_site_pair < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])
  tendon_wrapnum_site = np.array([
      sum((wrap_id_site >= adr) & (wrap_id_site < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])
  tendon_has_site = tendon_nsite > 0
  (tendon_id_site,) = np.nonzero(tendon_has_site)
  tendon_nsite = tendon_nsite[tendon_has_site]
  tendon_with_site = tendon_nsite.size
  ten_site_id = np.repeat(np.arange(tendon_with_site), tendon_nsite)

  length_site = jax.ops.segment_sum(lengths_site, ten_site_id, tendon_with_site)
  moment_site = jax.ops.segment_sum(moments_site, ten_site_id, tendon_with_site)

  # process spatial sphere/cylinder wrap
  (wrap_id_geom,) = np.nonzero(
      (m.wrap_type == WrapType.SPHERE) | (m.wrap_type == WrapType.CYLINDER)
  )

  # get objid for site-geom-site instances
  wrap_id_sitegeomsite = wrap_id_geom[:, None] + np.array([-1, 0, 1])[None]
  wrap_objid_site0, wrap_objid_geom, wrap_objid_site1 = m.wrap_objid[
      wrap_id_sitegeomsite
  ].T

  # get site positions before and after geom
  site_pnt0 = d.site_xpos[wrap_objid_site0]
  site_pnt1 = d.site_xpos[wrap_objid_site1]

  # get geom information
  geom_xpos = d.geom_xpos[wrap_objid_geom]
  geom_xmat = d.geom_xmat[wrap_objid_geom]
  geom_size = m.geom_size[wrap_objid_geom, 0]
  geom_type = m.wrap_type[wrap_id_geom]
  is_sphere = geom_type == WrapType.SPHERE

  # get body ids for site-geom-site instances
  body_id_site0 = m.site_bodyid[wrap_objid_site0]
  body_id_geom = m.geom_bodyid[wrap_objid_geom]
  body_id_site1 = m.site_bodyid[wrap_objid_site1]

  # find wrap object sidesites (if any exist)
  side_id = np.round(m.wrap_prm[wrap_id_geom]).astype(int)
  side = d.site_xpos[side_id]
  has_sidesite = np.expand_dims(np.array(side_id >= 0), -1)

  # wrap inside
  # TODO(taylorhowell): check that is_wrap_inside is consistent with
  # site and geom relative positions
  (wrap_inside_id,) = np.nonzero(m._impl.is_wrap_inside)
  (wrap_outside_id,) = np.nonzero(~m._impl.is_wrap_inside)

  # compute geom wrap length and connect points (if wrap occurs)
  v_wrap = jax.vmap(
      support.wrap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None)
  )
  lengths_inside, pnt0_inside, pnt1_inside = v_wrap(
      site_pnt0[wrap_inside_id],
      site_pnt1[wrap_inside_id],
      geom_xpos[wrap_inside_id],
      geom_xmat[wrap_inside_id],
      geom_size[wrap_inside_id],
      side[wrap_inside_id],
      has_sidesite[wrap_inside_id],
      is_sphere[wrap_inside_id],
      True,
      m._impl.wrap_inside_maxiter,
      m._impl.wrap_inside_tolerance,
      m._impl.wrap_inside_z_init,
  )

  lengths_outside, pnt0_outside, pnt1_outside = v_wrap(
      site_pnt0[wrap_outside_id],
      site_pnt1[wrap_outside_id],
      geom_xpos[wrap_outside_id],
      geom_xmat[wrap_outside_id],
      geom_size[wrap_outside_id],
      side[wrap_outside_id],
      has_sidesite[wrap_outside_id],
      is_sphere[wrap_outside_id],
      False,
      m._impl.wrap_inside_maxiter,
      m._impl.wrap_inside_tolerance,
      m._impl.wrap_inside_z_init,
  )

  wrap_id = np.argsort(np.concatenate([wrap_inside_id, wrap_outside_id]))
  vstack_ = lambda x, y: jp.vstack([x, y])[wrap_id]
  lengths_geomgeom = vstack_(lengths_inside, lengths_outside)
  geom_pnt0 = vstack_(pnt0_inside, pnt0_outside)
  geom_pnt1 = vstack_(pnt1_inside, pnt1_outside)

  lengths_geomgeom = lengths_geomgeom.reshape(-1)

  # identify geoms where wrap does not occur
  no_geom_wrap = lengths_geomgeom < 0
  wrap_objid_geom_skip = jp.where(no_geom_wrap, 0, wrap_objid_geom)

  # compute lengths for site-site (no wrap), site-geom, and geom-site segments
  def _distance(p0, p1):
    return jax.vmap(lambda x, y: math.norm(x - y))(p0, p1)

  lengths_sitesite = _distance(site_pnt0, site_pnt1)
  lengths_sitegeom = _distance(site_pnt0, geom_pnt0)
  lengths_geomsite = _distance(geom_pnt1, site_pnt1)

  # select length segments according to geom wrap
  lengths_geom = jp.where(
      no_geom_wrap,
      lengths_sitesite,
      lengths_sitegeom + lengths_geomgeom + lengths_geomsite,
  )

  # compute moments for site-site (no wrap), site-geom, geom-geom, and geom-site
  # segments
  _, moments_sitesite = _length_moment(
      site_pnt0, site_pnt1, body_id_site0, body_id_site1
  )
  _, moments_sitegeom = _length_moment(
      site_pnt0, geom_pnt0, body_id_site0, body_id_geom
  )
  _, moments_geomgeom = _length_moment(
      geom_pnt0, geom_pnt1, body_id_geom, body_id_geom
  )
  _, moments_geomsite = _length_moment(
      geom_pnt1, site_pnt1, body_id_geom, body_id_site1
  )

  # select moment segments according to geom wrap
  moments_geom = jp.where(
      no_geom_wrap[:, None],
      moments_sitesite,
      moments_sitegeom + moments_geomgeom + moments_geomsite,
  )

  if wrap_id_geom.size:
    divisor_geom = divisor[wrap_id_geom]
    lengths_geom /= divisor_geom
    moments_geom /= divisor_geom[:, None]

  # construct number of site-geom-site instances per tendon
  tendon_ngeom = np.array([
      sum((wrap_id_geom >= adr) & (wrap_id_geom < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])
  tendon_has_geom = tendon_ngeom > 0
  tendon_ngeom = tendon_ngeom[tendon_has_geom]

  # identify tendons with at least one site-geom-site instance
  (tendon_id_geom,) = np.nonzero(tendon_has_geom)

  # combine site-geom-site segment lengths and moments for each tendon
  tendon_with_geom = tendon_ngeom.size
  ten_geom_id = np.repeat(np.arange(tendon_with_geom), tendon_ngeom)

  length_geom = jax.ops.segment_sum(lengths_geom, ten_geom_id, tendon_with_geom)
  moment_geom = jax.ops.segment_sum(moments_geom, ten_geom_id, tendon_with_geom)

  # calculate number of wrap objects per tendon, based on geom wrap
  wrapnums_geom = jp.where(no_geom_wrap, 0, 2)
  tendon_wrapnum_geom = jax.ops.segment_sum(
      wrapnums_geom, ten_geom_id, tendon_with_geom
  )

  # assemble length and moment
  ten_length = (
      jp.zeros_like(d.ten_length).at[tendon_id_jnt].set(length_jnt)
  )
  ten_length = ten_length.at[tendon_id_site].add(length_site)
  ten_length = ten_length.at[tendon_id_geom].add(length_geom)

  ten_moment = (
      jp.zeros_like(d._impl.ten_J)
      .at[adr_moment_jnt, dofadr_moment_jnt]
      .set(moment_jnt)
  )
  ten_moment = ten_moment.at[tendon_id_site].add(moment_site)
  ten_moment = ten_moment.at[tendon_id_geom].add(moment_geom)

  # construct wrap addresses
  wrap_adr_pulley = []
  wrap_adr_site = []
  wrap_adr_geom = []

  count = 0
  for wrap_type in m.wrap_type:
    if wrap_type == WrapType.PULLEY:
      wrap_adr_pulley.append(count)
      count += 1
    elif wrap_type == WrapType.SITE:
      wrap_adr_site.append(count)
      count += 1
    elif wrap_type in (WrapType.SPHERE, WrapType.CYLINDER):
      wrap_adr_geom.append(count)
      wrap_adr_geom.append(count + 1)
      count += 2

  wrap_adr_pulley = np.array(wrap_adr_pulley).astype(int)
  wrap_adr_site = np.array(wrap_adr_site).astype(int)
  wrap_adr_geom = np.array(wrap_adr_geom).astype(int)
  wrap_adr = np.concatenate([wrap_adr_pulley, wrap_adr_site, wrap_adr_geom])

  ten_wrapnum = jp.array(tendon_wrapnum_pulley + tendon_wrapnum_site)
  ten_wrapnum = ten_wrapnum.at[tendon_id_geom].add(tendon_wrapnum_geom)

  ten_wrapadr = jp.concatenate([jp.array([0]), jp.cumsum(ten_wrapnum)[:-1]])

  xpos_site = d.site_xpos[m.wrap_objid[wrap_id_site]]
  xpos_geom = jp.hstack([geom_pnt0, geom_pnt1]).reshape((-1, 3))

  # sort objects, moving no wrap geoms to bottom rows
  wrap_adr_sort = np.argsort(wrap_adr)
  skipped = (
      jp.zeros(count, dtype=bool)
      .at[wrap_adr_geom]
      .set(jp.repeat(no_geom_wrap, 2).reshape(-1))
  )
  sort = jp.argsort(skipped)

  wrap_xpos = jp.concatenate(
      [jp.zeros((nwrap_pulley, 3)), xpos_site, xpos_geom]
  )[wrap_adr_sort]
  wrap_xpos = jp.concatenate(
      [wrap_xpos[sort], jp.zeros((2 * m.nwrap - count, 3))]
  ).reshape((m.nwrap, 6))

  wrap_obj = jp.concatenate([
      -2 * jp.ones(nwrap_pulley, dtype=int),
      -1 * jp.ones(nwrap_site, dtype=int),
      jp.repeat(wrap_objid_geom_skip, 2).reshape(-1),
  ])[wrap_adr_sort]
  wrap_obj = jp.concatenate(
      [wrap_obj[sort], jp.zeros(2 * m.nwrap - count, dtype=int)]
  ).reshape((m.nwrap, 2))

  return d.tree_replace({  # pyrefly: ignore[bad-return]
      'ten_length': ten_length,
      '_impl.ten_J': ten_moment,
      '_impl.ten_wrapadr': jp.array(ten_wrapadr, dtype=int),
      '_impl.ten_wrapnum': jp.array(ten_wrapnum, dtype=int),
      '_impl.wrap_xpos': wrap_xpos,
      '_impl.wrap_obj': jp.array(wrap_obj, dtype=int),
  })


def _site_dof_mask(m: Model) -> np.ndarray:
  """Creates a dof mask for site transmissions."""
  mask = np.ones((m.nu, m.nv))
  for i in np.nonzero(m.actuator_trnid[:, 1] != -1)[0]:
    id_, refid = m.actuator_trnid[i]
    # initialize last dof address for each body
    b0 = m.body_weldid[m.site_bodyid[id_]]
    b1 = m.body_weldid[m.site_bodyid[refid]]
    dofadr0 = m.body_dofadr[b0] + m.body_dofnum[b0] - 1
    dofadr1 = m.body_dofadr[b1] + m.body_dofnum[b1] - 1

    # find common ancestral dof, if any
    while dofadr0 != dofadr1:
      if dofadr0 < dofadr1:
        dofadr1 = m.dof_parentid[dofadr1]
      else:
        dofadr0 = m.dof_parentid[dofadr0]
      if dofadr0 == -1 or dofadr1 == -1:
        break

    # if common ancestral dof was found, clear the columns of its parental chain
    da = dofadr0 if dofadr0 == dofadr1 else -1
    while da >= 0:
      mask[i, da] = 0.0
      da = m.dof_parentid[da]

  return mask


def transmission(m: Model, d: Data) -> Data:
  """Computes actuator/transmission lengths and moments."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('transmission requires JAX backend implementation.')

  if not m.nu:
    return d

  def fn(
      trntype,
      trnid,
      gear,
      jnt_typ,
      m_j,
      qpos,
      has_refsite,
      site_dof_mask,
      site_xpos,
      site_xmat,
      site_quat,
  ):
    if trntype in (TrnType.JOINT, TrnType.JOINTINPARENT):
      if jnt_typ == JointType.FREE:
        length = jp.zeros(1)
        moment = gear
        if trntype == TrnType.JOINTINPARENT:
          quat_neg = math.quat_inv(qpos[3:])
          gearaxis = math.rotate(gear[3:], quat_neg)
          moment = moment.at[3:].set(gearaxis)
        m_j = m_j + jp.arange(6)
      elif jnt_typ == JointType.BALL:
        axis, angle = math.quat_to_axis_angle(qpos)
        gearaxis = gear[:3]
        if trntype == TrnType.JOINTINPARENT:
          quat_neg = math.quat_inv(qpos)
          gearaxis = math.rotate(gear[:3], quat_neg)
        length = jp.dot(axis * angle, gearaxis)[None]
        moment = gearaxis
        m_j = m_j + jp.arange(3)
      elif jnt_typ in (JointType.SLIDE, JointType.HINGE):
        length = qpos * gear[0]
        moment = gear[:1]
        m_j = m_j[None]
      else:
        raise RuntimeError(f'unrecognized joint type: {JointType(jnt_typ)}')

      moment = jp.zeros((m.nv,)).at[m_j].set(moment)
    elif trntype == TrnType.SITE:
      length = jp.zeros(1)
      id_, refid = jp.array(m.site_bodyid)[trnid]
      jacp, jacr = support.jac(m, d, site_xpos[0], id_)
      frame_xmat = site_xmat[0]
      if has_refsite:
        vecp = site_xmat[1].T @ (site_xpos[0] - site_xpos[1])
        vecr = math.quat_sub(site_quat[0], site_quat[1])
        length += jp.dot(jp.concatenate([vecp, vecr]), gear)
        jacrefp, jacrefr = support.jac(m, d, site_xpos[1], refid)
        jacp, jacr = jacp - jacrefp, jacr - jacrefr
        frame_xmat = site_xmat[1]

      jac = jp.concatenate((jacp, jacr), axis=1) * site_dof_mask[:, None]
      wrench = jp.concatenate((frame_xmat @ gear[:3], frame_xmat @ gear[3:]))
      moment = jac @ wrench
    elif trntype == TrnType.TENDON:
      length = d.ten_length[trnid[0]] * gear[:1]
      moment = d._impl.ten_J[trnid[0]] * gear[0]  # pyrefly: ignore[missing-attribute]
    else:
      raise RuntimeError(f'unrecognized trntype: {TrnType(trntype)}')

    return length, moment

  # pre-compute values for site transmissions
  has_refsite = m.actuator_trnid[:, 1] != -1
  site_dof_mask = _site_dof_mask(m)
  site_quat = jax.vmap(math.quat_mul)(d.xquat[m.site_bodyid], m.site_quat)

  length, moment = scan.flat(
      m,
      fn,
      'uuujjquusss',
      'uu',
      m.actuator_trntype,
      jp.array(m.actuator_trnid),
      m.actuator_gear,
      m.jnt_type,
      jp.array(m.jnt_dofadr),
      d.qpos,
      has_refsite,
      jp.array(site_dof_mask),
      d.site_xpos,
      d.site_xmat,
      site_quat,
      group_by='u',
  )
  length = length.reshape((m.nu,))
  moment = moment.reshape((m.nu, m.nv))

  d = d.tree_replace(  # pyrefly: ignore[bad-assignment]
      {'actuator_length': length, '_impl.actuator_moment': moment}
  )
  return d


def tendon_armature(m: Model, d: Data) -> Data:
  """Add tendon armature to M."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('tendon_armature requires JAX backend implementation.')

  if not m.ntendon:
    return d

  # TODO(taylorhowell): if sparse, compute sparse JTAJ
  JTAJ = d._impl.ten_J.T @ jax.vmap(jp.multiply)(
      d._impl.ten_J, m.tendon_armature
  )

  if support.is_sparse(m):
    i = np.repeat(np.arange(m.nv), m.M_rownnz)
    j = m.M_colind
    JTAJ = JTAJ[(i, j)]

  return d.tree_replace({'_impl.M': d._impl.M + JTAJ})  # pyrefly: ignore[bad-return]


def tendon_dot(m: Model, d: Data) -> jax.Array:
  """Compute time derivative of dense tendon Jacobian for one tendon."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('tendon_dot requires JAX backend implementation.')

  ten_Jdot = jp.zeros((m.ntendon, m.nv))  # pylint: disable=invalid-name

  if not m.ntendon:
    return ten_Jdot

  # process pulleys
  (wrap_id_pulley,) = np.nonzero(m.wrap_type == WrapType.PULLEY)

  divisor = np.ones(m.nwrap)
  for adr, num in zip(m.tendon_adr, m.tendon_num):
    for id_pulley in wrap_id_pulley:
      if adr <= id_pulley < adr + num:
        divisor[id_pulley : adr + num] = np.maximum(
            mujoco.mjMINVAL, m.wrap_prm[id_pulley]
        )

  # process spatial tendon sites
  (wrap_id_site,) = np.nonzero(m.wrap_type == WrapType.SITE)

  # find consecutive sites, skipping tendon transitions
  (pair_id,) = np.nonzero(np.diff(wrap_id_site) == 1)
  wrap_id_site_pair = np.setdiff1d(wrap_id_site[pair_id], m.tendon_adr[1:] - 1)
  wrap_objid_site0 = m.wrap_objid[wrap_id_site_pair]
  wrap_objid_site1 = m.wrap_objid[wrap_id_site_pair + 1]
  site_bodyid0 = m.site_bodyid[wrap_objid_site0]
  site_bodyid1 = m.site_bodyid[wrap_objid_site1]
  site_xpos0 = d.site_xpos[wrap_objid_site0]
  site_xpos1 = d.site_xpos[wrap_objid_site1]
  subtree_com0 = d.subtree_com[m.body_rootid[site_bodyid0]]
  subtree_com1 = d.subtree_com[m.body_rootid[site_bodyid1]]
  site_vel0 = jax.vmap(lambda a, b: a[3:] - jp.cross(b, a[:3]))(
      d.cvel[site_bodyid0], site_xpos0 - subtree_com0
  )
  site_vel1 = jax.vmap(lambda a, b: a[3:] - jp.cross(b, a[:3]))(
      d.cvel[site_bodyid1], site_xpos1 - subtree_com1
  )

  @jax.vmap
  def _momentdot(wpnt0, wpnt1, wvel0, wvel1, body0, body1):
    # dpnt = 3D position difference, normalize
    dpnt = wpnt1 - wpnt0
    norm = math.norm(dpnt)
    dpnt = jp.where(
        norm < mujoco.mjMINVAL,
        jp.array([1.0, 0.0, 0.0]),
        math.safe_div(dpnt, norm),
    )

    # dvel = d / dt(dpnt)
    dvel = wvel1 - wvel0
    dot = jp.dot(dpnt, dvel)
    dvel += dpnt * -dot
    dvel = jp.where(norm > mujoco.mjMINVAL, math.safe_div(dvel, norm), 0.0)

    # get endpoint JacobianDots, subtract
    jacp1, _ = support.jac_dot(m, d, wpnt0, body0)
    jacp2, _ = support.jac_dot(m, d, wpnt1, body1)
    jacdif = jacp2 - jacp1

    # chain rule, first term: Jdot += d / dt(jac2 - jac1) * dpnt
    tmp0 = jacdif @ dpnt

    # get endpoint Jacobians, subtract
    jacp1, _ = support.jac(m, d, wpnt0, body0)
    jacp2, _ = support.jac(m, d, wpnt1, body1)
    jacdif = jacp2 - jacp1

    # chain rule, second term: Jdot += (jac2 - jac1) * d/dt (dpnt)
    tmp1 = jacdif @ dvel

    return jp.where(body0 != body1, tmp0 + tmp1, jp.zeros(m.nv))

  momentdots = _momentdot(
      site_xpos0,
      site_xpos1,
      site_vel0,
      site_vel1,
      site_bodyid0,
      site_bodyid1,
  )

  if wrap_id_site_pair.size:
    divisor_site_pair = divisor[wrap_id_site_pair]
    momentdots /= divisor_site_pair[:, None]

  tendon_nsite = np.array([
      sum((wrap_id_site_pair >= adr) & (wrap_id_site_pair < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])
  tendon_has_site = tendon_nsite > 0
  (tendon_id_site,) = np.nonzero(tendon_has_site)
  tendon_nsite = tendon_nsite[tendon_has_site]
  tendon_with_site = tendon_nsite.size
  ten_site_id = np.repeat(np.arange(tendon_with_site), tendon_nsite)

  momentdot = jax.ops.segment_sum(momentdots, ten_site_id, tendon_with_site)
  ten_Jdot = ten_Jdot.at[tendon_id_site].set(momentdot)  # pylint: disable=invalid-name

  # TODO(taylorhowell): time derivatives for geoms

  return ten_Jdot


def tendon_bias(m: Model, d: Data) -> Data:
  """Add bias force due to tendon armature."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('tendon_bias requires JAX backend implementation.')

  if not m.ntendon:
    return d

  # get dense d/dt(tendon Jacobian) for each tendon
  ten_Jdot = tendon_dot(m, d)  # pylint: disable=invalid-name

  # add bias term: qfrc += ten_J * armature * ten_Jdot @ qvel
  coef = m.tendon_armature * jp.dot(ten_Jdot, d.qvel)

  return d.tree_replace({  # pyrefly: ignore[bad-return]
      'qfrc_bias': (
          d.qfrc_bias
          + jp.sum(jax.vmap(jp.multiply)(d._impl.ten_J, coef), axis=0)
      )
  })
