# Copyright 2025 The Newton Developers
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

from typing import Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src.types import ConstraintType
from mujoco.mjx.third_party.mujoco_warp._src.types import ContactType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.types import vec12
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def _zero_constraint_counts(
  # Data out:
  ne_out: wp.array[int],
  nf_out: wp.array[int],
  nl_out: wp.array[int],
  nefc_out: wp.array[int],
  efc_jtdaj_nblock_out: wp.array[int],
  # Out:
  efc_nnz_out: wp.array[int],
):
  worldid = wp.tid()

  # Zero all constraint counters
  ne_out[worldid] = 0
  nf_out[worldid] = 0
  nl_out[worldid] = 0
  nefc_out[worldid] = 0
  efc_jtdaj_nblock_out[worldid] = 0
  efc_nnz_out[worldid] = 0


@wp.func
def _reserve_efc_nnz(
  # Data in:
  nefc_in: wp.array[int],
  njmax_in: int,
  njmax_nnz_in: int,
  # In:
  worldid: int,
  nrow: int,
  rownnz: int,
  # Out:
  efc_nnz_out: wp.array[int],
) -> int:
  """Reserves sparse Jacobian storage for a block of rows that is not yet allocated.

  Storage must be reserved before rows are allocated: a row counted in nefc without storage would
  keep a stale sparse layout from a previous step.

  Returns:
    Address of the reserved storage, or -1 if it does not fit and the rows must be dropped.
  """
  # nefc only grows, so at most this many rows of the block will be allocated below njmax
  nrow = wp.min(nrow, njmax_in - nefc_in[worldid])
  if nrow <= 0:
    return 0

  return _reserve_nnz(njmax_nnz_in, worldid, nrow * rownnz, efc_nnz_out)


@wp.func
def _reserve_nnz(
  # Data in:
  njmax_nnz_in: int,
  # In:
  worldid: int,
  nnz: int,
  # Out:
  efc_nnz_out: wp.array[int],
) -> int:
  """Reserves nnz sparse Jacobian entries, returns their address or -1 if they do not fit."""
  rowadr = wp.atomic_add(efc_nnz_out, worldid, nnz)  # kernel_analyzer: ignore[atomic]
  if rowadr + nnz > njmax_nnz_in:
    return -1
  return rowadr


@cache_kernel
def _efc_nnz_overflow(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    njmax_nnz_in: int,
    # In:
    efc_nnz_in: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()

    # the counter only exceeds njmax_nnz once a reservation failed and its rows were dropped
    if efc_nnz_in[worldid] > njmax_nnz_in:
      if wp.static(bool(warn_overflow & OverflowType.NJMAX_NNZ)):
        wp.printf(
          "njmax_nnz overflow - please increase njmax_nnz beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NJMAX_NNZ (or = 0 for all)\n",
          njmax_nnz_in,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.NJMAX_NNZ

  return kernel


@wp.func
def _contact_kbimp(
  # Model:
  opt_disableflags: int,
  # In:
  timestep: float,
  solref: wp.vec2,
  solimp: vec5,
  pos_imp: float,
  is_discrete: bool,
) -> wp.vec3:
  """Computes the constraint (k, b, impedance) from solref/solimp at impedance position pos_imp."""
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width_raw = solimp[2]
  width = width_raw
  mid = solimp[3]
  power = solimp[4]

  if not (opt_disableflags & DisableBit.REFSAFE) and not is_discrete:
    timeconst = wp.max(timeconst, 2.0 * timestep)

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  # see https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  dmax_sq = dmax * dmax
  # use a branch because Warp differentiates both wp.where operands; the unused time-constant
  # formula for direct-format solref can divide by zero or produce NaN gradients
  if solref[0] <= 0.0:
    k = -solref[0] / wp.max(types.MJ_MINVAL, dmax_sq)
  else:
    k = 1.0 / wp.max(types.MJ_MINVAL, dmax_sq * timeconst * timeconst * dampratio * dampratio)
  if solref[1] <= 0.0:
    b = -solref[1] / wp.max(types.MJ_MINVAL, dmax)
  else:
    b = 2.0 / wp.max(types.MJ_MINVAL, dmax * timeconst)

  imp_x = wp.abs(pos_imp) / width
  # evaluate only the active polynomial because the unused branch can raise FE_INVALID or produce
  # NaN gradients; the constant branch gives zero gradient at and beyond the transition endpoints
  if dmin == dmax or width_raw <= types.MJ_MINVAL:
    imp = 0.5 * (dmin + dmax)
  elif imp_x <= 0.0:
    imp = dmin
  elif imp_x >= 1.0:
    imp = dmax
  elif power == 1.0:
    imp = dmin + imp_x * (dmax - dmin)
  elif imp_x <= mid:
    imp_y = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
    imp = wp.clamp(dmin + imp_y * (dmax - dmin), dmin, dmax)
  else:
    imp_y = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
    imp = wp.clamp(dmin + imp_y * (dmax - dmin), dmin, dmax)

  return wp.vec3(k, b, imp)


@wp.func
def _efc_D(invweight: float, imp: float) -> float:
  return 1.0 / wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)


@wp.func
def _efc_row(
  # Model:
  opt_disableflags: int,
  # In:
  worldid: int,
  timestep: float,
  efcid: int,
  pos_aref: float,
  pos_imp: float,
  invweight: float,
  solref: wp.vec2,
  solimp: vec5,
  margin: float,
  vel: float,
  frictionloss: float,
  type: int,
  id: int,
  is_discrete: bool,
  aref_shift: float,
  # Out:
  type_out: wp.array2d[int],
  id_out: wp.array2d[int],
  pos_out: wp.array2d[float],
  margin_out: wp.array2d[float],
  D_out: wp.array2d[float],
  vel_out: wp.array2d[float],
  aref_out: wp.array2d[float],
  frictionloss_out: wp.array2d[float],
):
  # calculate kbi
  kbimp = _contact_kbimp(opt_disableflags, timestep, solref, solimp, pos_imp, is_discrete)
  k = 0.0 if (type == ConstraintType.FRICTION_DOF or type == ConstraintType.FRICTION_TENDON) else kbimp[0]
  b = kbimp[1]
  imp = kbimp[2]

  if is_discrete:
    timestep_sq = timestep * timestep
    # discrete refsafe: a contact or limit row whose spring the step cannot resolve (h^2*K*I > 1)
    if (
      not (opt_disableflags & DisableBit.REFSAFE)
      and solref[0] > 0.0
      and k > 0.0
      and (
        type == ConstraintType.LIMIT_JOINT
        or type == ConstraintType.LIMIT_TENDON
        or type == ConstraintType.CONTACT_FRICTIONLESS
        or type == ConstraintType.CONTACT_PYRAMIDAL
        or type == ConstraintType.CONTACT_ELLIPTIC
      )
    ):
      excess = timestep_sq * k * imp
      if excess > 1.0:
        fmax = math.safe_div(types.MJ_MAXIMP * (1.0 - imp), imp * (1.0 - types.MJ_MAXIMP))
        k = 1.0 / (timestep_sq * imp)
        b = wp.min(b / wp.sqrt(excess), wp.max(0.0, fmax - 2.0) / timestep)

    f = 1.0 + timestep * b + timestep_sq * k * imp
    R0 = wp.max(types.MJ_MINVAL, (1.0 - imp) * invweight / imp)
    Rmin = wp.max(types.MJ_MINVAL, (1.0 - types.MJ_MAXIMP) * invweight / types.MJ_MAXIMP)
    R = wp.max(R0 / f, Rmin)
    D_out[worldid, efcid] = 1.0 / R
    aref_out[worldid, efcid] = (-k * imp * pos_aref - (b + timestep * k * imp) * vel - aref_shift) / f
  else:
    D_out[worldid, efcid] = _efc_D(invweight, imp)
    aref_out[worldid, efcid] = -k * imp * pos_aref - b * vel - aref_shift

  vel_out[worldid, efcid] = vel
  pos_out[worldid, efcid] = pos_aref + margin
  margin_out[worldid, efcid] = margin
  frictionloss_out[worldid, efcid] = frictionloss
  type_out[worldid, efcid] = type
  id_out[worldid, efcid] = id


@cache_kernel
def _equality_connect(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    nsite: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_invweight0: wp.array2d[wp.vec2],
    jnt_type: wp.array[int],
    jnt_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_jntid: wp.array[int],
    dof_parentid: wp.array[int],
    site_bodyid: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_objtype: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec12],
    body_isdofancestor: wp.array2d[int],
    eq_connect_adr: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    eq_active_in: wp.array2d[bool],
    xpos_in: wp.array2d[wp.vec3],
    xmat_in: wp.array2d[wp.mat33],
    site_xpos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    cvel_in: wp.array2d[wp.spatial_vector],
    cdof_dot_in: wp.array2d[wp.spatial_vector],
    subtree_linvel_in: wp.array2d[wp.vec3],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    ne_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    """Calculates constraint rows for connect equality constraints."""
    worldid, eqconnectid = wp.tid()
    eqid = eq_connect_adr[eqconnectid]

    if not eq_active_in[worldid, eqid]:
      return

    data = eq_data[worldid % eq_data.shape[0], eqid]
    anchor1 = wp.vec3f(data[0], data[1], data[2])
    anchor2 = wp.vec3f(data[3], data[4], data[5])

    obj1id = eq_obj1id[eqid]
    obj2id = eq_obj2id[eqid]

    if nsite > 0 and eq_objtype[eqid] == types.ObjType.SITE:
      body1 = site_bodyid[obj1id]
      body2 = site_bodyid[obj2id]
      pos1 = site_xpos_in[worldid, obj1id]
      pos2 = site_xpos_in[worldid, obj2id]
    else:
      body1 = obj1id
      body2 = obj2id
      pos1 = xpos_in[worldid, body1] + xmat_in[worldid, body1] @ anchor1
      pos2 = xpos_in[worldid, body2] + xmat_in[worldid, body2] @ anchor2

    # error is difference in global positions
    pos = pos1 - pos2

    rownnz = int(0)
    rowadr = int(0)
    if wp.static(is_sparse):
      # TODO(team): pre-compute number of non-zeros
      body1 = body_weldid[body1]
      body2 = body_weldid[body2]

      # count non-zeros
      pda1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
      pda2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)
      while pda1 >= 0 or pda2 >= 0:
        da = wp.max(pda1, pda2)
        if pda1 == da:
          pda1 = dof_parentid[pda1]
        if pda2 == da:
          pda2 = dof_parentid[pda2]
        rownnz += 1

      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 3, rownnz, efc_nnz_out)
      if rowadr < 0:
        return

    wp.atomic_add(ne_out, worldid, 3)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, 3)  # kernel_analyzer: ignore[atomic]

    if efcid >= njmax_in:
      return

    # rows past njmax are dropped
    nrow = wp.min(3, njmax_in - efcid)

    if wp.static(is_sparse and newton):
      jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efc_jtdaj_adr_out[worldid, jgid] = efcid
      efc_jtdaj_nrow_out[worldid, jgid] = nrow

    # compute Jacobian difference (opposite of contact: 0 - 1)
    Jqvel = wp.vec3f(0.0, 0.0, 0.0)
    Jdotv = wp.vec3f(0.0, 0.0, 0.0)

    if wp.static(is_sparse):
      da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
      da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)

      for i in range(3):
        if i < nrow:
          efc_J_rowadr_out[worldid, efcid + i] = rowadr + i * rownnz
          efc_J_rownnz_out[worldid, efcid + i] = rownnz

      # compute J and colind
      nnz = int(0)
      while da1 >= 0 or da2 >= 0:
        da = wp.max(da1, da2)
        if da1 == da:
          da1 = dof_parentid[da1]
        if da2 == da:
          da2 = dof_parentid[da2]

        jacp1, _ = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos1,
          body1,
          da,
          worldid,
        )
        jacp2, _ = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos2,
          body2,
          da,
          worldid,
        )
        j1mj2 = jacp1 - jacp2

        jacp1_dot, _ = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos1,
          body1,
          da,
          worldid,
        )
        jacp2_dot, _ = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos2,
          body2,
          da,
          worldid,
        )
        j1mj2_dot = jacp1_dot - jacp2_dot

        # constant indices keep j1mj2 in registers
        for i in range(3):
          if i < nrow:
            sparseid = rowadr + i * rownnz + nnz
            efc_J_colind_out[worldid, 0, sparseid] = da
            efc_J_out[worldid, 0, sparseid] = j1mj2[i]

        qvel = qvel_in[worldid, da]
        Jqvel += j1mj2 * qvel
        Jdotv += j1mj2_dot * qvel

        nnz += 1
    else:
      # TODO(team): dof tree traversal
      for dofid in range(nv):
        jacp1, _ = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos1,
          body1,
          dofid,
          worldid,
        )
        jacp2, _ = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos2,
          body2,
          dofid,
          worldid,
        )
        j1mj2 = jacp1 - jacp2

        jacp1_dot, _ = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos1,
          body1,
          dofid,
          worldid,
        )
        jacp2_dot, _ = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos2,
          body2,
          dofid,
          worldid,
        )
        j1mj2_dot = jacp1_dot - jacp2_dot

        for i in range(3):
          if i < nrow:
            efc_J_out[worldid, efcid + i, dofid] = j1mj2[i]

        qvel = qvel_in[worldid, dofid]
        Jqvel += j1mj2 * qvel
        Jdotv += j1mj2_dot * qvel

    body_invweight0_id = worldid % body_invweight0.shape[0]
    invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]
    pos_imp = wp.length(pos)

    solref = eq_solref[worldid % eq_solref.shape[0], eqid]
    solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    for i in range(nrow):
      efcidi = efcid + i

      _efc_row(
        opt_disableflags,
        worldid,
        timestep,
        efcidi,
        pos[i],
        pos_imp,
        invweight,
        solref,
        solimp,
        0.0,
        Jqvel[i],
        0.0,
        ConstraintType.EQUALITY,
        eqid,
        is_discrete,
        Jdotv[i],
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
      )

  return kernel


@cache_kernel
def _equality_joint(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    qpos0: wp.array2d[float],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    dof_invweight0: wp.array2d[float],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec12],
    eq_jnt_adr: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    eq_active_in: wp.array2d[bool],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    ne_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, eqjntid = wp.tid()
    eqid = eq_jnt_adr[eqjntid]

    if not eq_active_in[worldid, eqid]:
      return

    jntid_1 = eq_obj1id[eqid]
    jntid_2 = eq_obj2id[eqid]

    rownnz = int(0)
    rowadr = int(0)
    if wp.static(is_sparse):
      rownnz = wp.where(jntid_2 > -1, 2, 1)
      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, rownnz, efc_nnz_out)
      if rowadr < 0:
        return

    wp.atomic_add(ne_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

    if efcid >= njmax_in:
      return

    if wp.static(is_sparse and newton):
      jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efc_jtdaj_adr_out[worldid, jgid] = efcid
      efc_jtdaj_nrow_out[worldid, jgid] = 1

    data = eq_data[worldid % eq_data.shape[0], eqid]
    dofadr1 = jnt_dofadr[jntid_1]
    qposadr1 = jnt_qposadr[jntid_1]
    qpos0_id = worldid % qpos0.shape[0]
    dof_invweight0_id = worldid % dof_invweight0.shape[0]

    if wp.static(is_sparse):
      efc_J_rownnz_out[worldid, efcid] = rownnz
      efc_J_rowadr_out[worldid, efcid] = rowadr
      efc_J_colind_out[worldid, 0, rowadr] = dofadr1
      efc_J_out[worldid, 0, rowadr] = 1.0
    else:
      for i in range(nv):
        efc_J_out[worldid, efcid, i] = 0.0
      efc_J_out[worldid, efcid, dofadr1] = 1.0

    if jntid_2 > -1:
      # Two joint constraint
      qposadr2 = jnt_qposadr[jntid_2]
      dofadr2 = jnt_dofadr[jntid_2]
      dif = qpos_in[worldid, qposadr2] - qpos0[qpos0_id, qposadr2]

      # Horner's method for polynomials
      rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))
      deriv_2 = data[1] + dif * (2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4]))

      pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - rhs
      Jqvel = qvel_in[worldid, dofadr1] - qvel_in[worldid, dofadr2] * deriv_2
      invweight = dof_invweight0[dof_invweight0_id, dofadr1] + dof_invweight0[dof_invweight0_id, dofadr2]

      if wp.static(is_sparse):
        sparseid = rowadr + 1
        efc_J_colind_out[worldid, 0, sparseid] = dofadr2
        efc_J_out[worldid, 0, sparseid] = -deriv_2
      else:
        efc_J_out[worldid, efcid, dofadr2] = -deriv_2
    else:
      # Single joint constraint
      pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - data[0]
      Jqvel = qvel_in[worldid, dofadr1]
      invweight = dof_invweight0[dof_invweight0_id, dofadr1]

    # Update constraint parameters
    _efc_row(
      opt_disableflags,
      worldid,
      opt_timestep[worldid % opt_timestep.shape[0]],
      efcid,
      pos,
      pos,
      invweight,
      eq_solref[worldid % eq_solref.shape[0], eqid],
      eq_solimp[worldid % eq_solimp.shape[0], eqid],
      0.0,
      Jqvel,
      0.0,
      ConstraintType.EQUALITY,
      eqid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

  return kernel


@cache_kernel
def _equality_tendon(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec12],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_length0: wp.array2d[float],
    tendon_invweight0: wp.array2d[float],
    eq_ten_adr: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    eq_active_in: wp.array2d[bool],
    ten_J_in: wp.array2d[float],
    ten_length_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    ne_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, eqtenid = wp.tid()
    eqid = eq_ten_adr[eqtenid]

    if not eq_active_in[worldid, eqid]:
      return

    obj1id = eq_obj1id[eqid]
    obj2id = eq_obj2id[eqid]

    data = eq_data[worldid % eq_data.shape[0], eqid]
    solref = eq_solref[worldid % eq_solref.shape[0], eqid]
    solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]
    tendon_length0_id = worldid % tendon_length0.shape[0]
    tendon_invweight0_id = worldid % tendon_invweight0.shape[0]
    pos1 = ten_length_in[worldid, obj1id] - tendon_length0[tendon_length0_id, obj1id]

    if obj2id > -1:
      invweight = tendon_invweight0[tendon_invweight0_id, obj1id] + tendon_invweight0[tendon_invweight0_id, obj2id]

      pos2 = ten_length_in[worldid, obj2id] - tendon_length0[tendon_length0_id, obj2id]

      dif = pos2
      dif2 = dif * dif
      dif3 = dif2 * dif
      dif4 = dif3 * dif

      pos = pos1 - (data[0] + data[1] * dif + data[2] * dif2 + data[3] * dif3 + data[4] * dif4)
      deriv = data[1] + 2.0 * data[2] * dif + 3.0 * data[3] * dif2 + 4.0 * data[4] * dif3
    else:
      invweight = tendon_invweight0[tendon_invweight0_id, obj1id]
      pos = pos1 - data[0]
      deriv = 0.0

    rownnz1 = ten_J_rownnz[obj1id]
    rowadr1 = ten_J_rowadr[obj1id]
    rownnz2 = 0
    rowadr2 = 0

    if deriv != 0.0:
      rownnz2 = ten_J_rownnz[obj2id]
      rowadr2 = ten_J_rowadr[obj2id]

    rownnz = int(0)
    rowadr = int(0)
    if wp.static(is_sparse):
      # TODO(team): pre-compute rownnz
      # count unique dofs
      p1, p2 = int(0), int(0)
      while p1 < rownnz1 or p2 < rownnz2:
        col1 = nv
        col2 = nv
        if p1 < rownnz1:
          col1 = ten_J_colind[rowadr1 + p1]
        if p2 < rownnz2:
          col2 = ten_J_colind[rowadr2 + p2]
        if col1 <= col2:
          p1 += 1
        if col2 <= col1:
          p2 += 1
        rownnz += 1

      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, rownnz, efc_nnz_out)
      if rowadr < 0:
        return

    wp.atomic_add(ne_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

    if efcid >= njmax_in:
      return

    if wp.static(is_sparse and newton):
      jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efc_jtdaj_adr_out[worldid, jgid] = efcid
      efc_jtdaj_nrow_out[worldid, jgid] = 1

    if wp.static(is_sparse):
      efc_J_rowadr_out[worldid, efcid] = rowadr

    ptr1 = int(0)
    ptr2 = int(0)

    Jqvel = float(0.0)

    nnz = int(0)
    for i in range(nv):
      J1 = float(0.0)
      if ptr1 < rownnz1:
        sparseid1 = rowadr1 + ptr1
        if ten_J_colind[sparseid1] == i:
          J1 = ten_J_in[worldid, sparseid1]
          ptr1 += 1

      J = J1
      if deriv != 0.0:
        J2 = float(0.0)
        if ptr2 < rownnz2:
          sparseid2 = rowadr2 + ptr2
          if ten_J_colind[sparseid2] == i:
            J2 = ten_J_in[worldid, sparseid2]
            ptr2 += 1
        J += J2 * -deriv

      if wp.static(is_sparse):
        if J != 0.0:
          sparseid = rowadr + nnz
          efc_J_colind_out[worldid, 0, sparseid] = i
          efc_J_out[worldid, 0, sparseid] = J
          nnz += 1
      else:
        efc_J_out[worldid, efcid, i] = J

      Jqvel += J * qvel_in[worldid, i]

    if wp.static(is_sparse):
      efc_J_rownnz_out[worldid, efcid] = nnz

    _efc_row(
      opt_disableflags,
      worldid,
      opt_timestep[worldid % opt_timestep.shape[0]],
      efcid,
      pos,
      pos,
      invweight,
      solref,
      solimp,
      0.0,
      Jqvel,
      0.0,
      ConstraintType.EQUALITY,
      eqid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

  return kernel


@cache_kernel
def _equality_flex_alloc(is_sparse: bool, newton: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    flex_edgeadr: wp.array[int],
    flex_edgenum: wp.array[int],
    flexedge_J_rownnz: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_flex_adr: wp.array[int],
    flex_efcnum: wp.array[int],
    flex_efcnnz: wp.array[int],
    flexedge_efcadr: wp.array[int],
    # Data in:
    eq_active_in: wp.array2d[bool],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    ne_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_jtdaj_nblock_out: wp.array[int],
    # Out:
    block_out: wp.array2d[wp.vec3i],
    efc_nnz_out: wp.array[int],
  ):
    """Allocates the edge equality rows of a flex as one block.

    block_out holds the first efc row, sparse Jacobian address, and jtdaj block of the rows, or -1
    as first efc row if no rows were allocated.
    """
    worldid, eqflexid = wp.tid()
    block_out[worldid, eqflexid] = wp.vec3i(-1, 0, 0)

    eqid = eq_flex_adr[eqflexid]
    if not eq_active_in[worldid, eqid]:
      return

    flexid = eq_obj1id[eqid]
    nrow = flex_efcnum[flexid]
    if nrow == 0:
      return

    rowadr = int(0)
    if wp.static(is_sparse):
      # nefc only grows, so at most this many rows of the block will be allocated below njmax
      nfit = wp.min(nrow, njmax_in - nefc_out[worldid])
      if nfit > 0:
        nnz = int(0)
        if nfit < nrow:
          edgeadr = flex_edgeadr[flexid]
          for edgeid in range(edgeadr, edgeadr + flex_edgenum[flexid]):
            row = flexedge_efcadr[edgeid]
            if row >= 0 and row < nfit:
              nnz += flexedge_J_rownnz[edgeid]
        else:
          nnz = flex_efcnnz[flexid]
        rowadr = _reserve_nnz(njmax_nnz_in, worldid, nnz, efc_nnz_out)
        if rowadr < 0:
          return

    wp.atomic_add(ne_out, worldid, nrow)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, nrow)  # kernel_analyzer: ignore[atomic]
    if efcid >= njmax_in:
      return

    jtdajadr = int(0)
    if wp.static(is_sparse and newton):
      jtdajadr = wp.atomic_add(efc_jtdaj_nblock_out, worldid, wp.min(nrow, njmax_in - efcid))  # kernel_analyzer: ignore[atomic]

    block_out[worldid, eqflexid] = wp.vec3i(efcid, rowadr, jtdajadr)

  return kernel


@cache_kernel
def _equality_flex(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    flex_edgeadr: wp.array[int],
    flex_edgenum: wp.array[int],
    flexedge_length0: wp.array[float],
    flexedge_invweight0: wp.array[float],
    flexedge_J_rownnz: wp.array[int],
    flexedge_J_rowadr: wp.array[int],
    flexedge_J_colind: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_flex_adr: wp.array[int],
    flexedge_efcadr: wp.array[int],
    flexedge_efcnnzadr: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    flexedge_J_in: wp.array2d[float],
    flexedge_length_in: wp.array2d[float],
    njmax_in: int,
    # In:
    block_in: wp.array2d[wp.vec3i],
    # Data out:
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
  ):
    worldid, eqflexid, edgeid = wp.tid()

    # rows were allocated by _equality_flex_alloc
    block = block_in[worldid, eqflexid]
    if block[0] < 0:
      return

    eqid = eq_flex_adr[eqflexid]
    flexid = eq_obj1id[eqid]
    edgeadr = flex_edgeadr[flexid]
    if edgeid < edgeadr or edgeid >= edgeadr + flex_edgenum[flexid]:
      return

    # rigid edges and interpolated flexes have no rows
    row = flexedge_efcadr[edgeid]
    if row < 0:
      return

    efcid = block[0] + row
    if efcid >= njmax_in:
      return

    if wp.static(is_sparse and newton):
      jgid = block[2] + row
      efc_jtdaj_adr_out[worldid, jgid] = efcid
      efc_jtdaj_nrow_out[worldid, jgid] = 1

    rownnz = flexedge_J_rownnz[edgeid]
    efc_rowadr = int(0)
    if wp.static(is_sparse):
      efc_rowadr = block[1] + flexedge_efcnnzadr[edgeid]

    pos = flexedge_length_in[worldid, edgeid] - flexedge_length0[edgeid]
    solref = eq_solref[worldid % eq_solref.shape[0], eqid]
    solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]

    Jqvel = float(0.0)

    flex_rowadr = flexedge_J_rowadr[edgeid]

    if wp.static(is_sparse):
      efc_J_rownnz_out[worldid, efcid] = rownnz
      efc_J_rowadr_out[worldid, efcid] = efc_rowadr
      for i in range(rownnz):
        flex_sparseid = flex_rowadr + i
        efc_sparseid = efc_rowadr + i
        colind = flexedge_J_colind[flex_sparseid]
        J = flexedge_J_in[worldid, flex_sparseid]
        efc_J_colind_out[worldid, 0, efc_sparseid] = colind
        efc_J_out[worldid, 0, efc_sparseid] = J
        Jqvel += J * qvel_in[worldid, colind]
    else:
      for i in range(nv):
        efc_J_out[worldid, efcid, i] = 0.0
      for i in range(rownnz):
        flex_sparseid = flex_rowadr + i
        colind = flexedge_J_colind[flex_sparseid]
        J = flexedge_J_in[worldid, flex_sparseid]
        efc_J_out[worldid, efcid, colind] = J
        Jqvel += J * qvel_in[worldid, colind]

    _efc_row(
      opt_disableflags,
      worldid,
      opt_timestep[worldid % opt_timestep.shape[0]],
      efcid,
      pos,
      pos,
      flexedge_invweight0[edgeid],
      solref,
      solimp,
      0.0,
      Jqvel,
      0.0,
      ConstraintType.EQUALITY,
      eqid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

  return kernel


@cache_kernel
def _equality_weld(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    nsite: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_invweight0: wp.array2d[wp.vec2],
    jnt_type: wp.array[int],
    jnt_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_jntid: wp.array[int],
    dof_parentid: wp.array[int],
    site_bodyid: wp.array[int],
    site_quat: wp.array2d[wp.quat],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_objtype: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec12],
    body_isdofancestor: wp.array2d[int],
    eq_wld_adr: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    eq_active_in: wp.array2d[bool],
    xpos_in: wp.array2d[wp.vec3],
    xquat_in: wp.array2d[wp.quat],
    xmat_in: wp.array2d[wp.mat33],
    site_xpos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    cvel_in: wp.array2d[wp.spatial_vector],
    cdof_dot_in: wp.array2d[wp.spatial_vector],
    subtree_linvel_in: wp.array2d[wp.vec3],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    ne_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, eqweldid = wp.tid()
    eqid = eq_wld_adr[eqweldid]

    if not eq_active_in[worldid, eqid]:
      return

    is_site = eq_objtype[eqid] == types.ObjType.SITE and nsite > 0

    obj1id = eq_obj1id[eqid]
    obj2id = eq_obj2id[eqid]

    data = eq_data[worldid % eq_data.shape[0], eqid]
    anchor1 = wp.vec3(data[0], data[1], data[2])
    anchor2 = wp.vec3(data[3], data[4], data[5])
    relpose = wp.quat(data[6], data[7], data[8], data[9])
    torquescale = data[10]

    if is_site:
      body1 = site_bodyid[obj1id]
      body2 = site_bodyid[obj2id]
      pos1 = site_xpos_in[worldid, obj1id]
      pos2 = site_xpos_in[worldid, obj2id]

      site_quat_id = worldid % site_quat.shape[0]
      quat = math.mul_quat(xquat_in[worldid, body1], site_quat[site_quat_id, obj1id])
      quat1 = math.quat_inv(math.mul_quat(xquat_in[worldid, body2], site_quat[site_quat_id, obj2id]))

    else:
      body1 = obj1id
      body2 = obj2id
      pos1 = xpos_in[worldid, body1] + xmat_in[worldid, body1] @ anchor2
      pos2 = xpos_in[worldid, body2] + xmat_in[worldid, body2] @ anchor1

      quat = math.mul_quat(xquat_in[worldid, body1], relpose)
      quat1 = math.quat_inv(xquat_in[worldid, body2])

      # quat1 = quat_inv(xquat_in[worldid, body2])
      q2 = xquat_in[worldid, body2]
      quat1 = wp.quat(q2[0], -q2[1], -q2[2], -q2[3])

    # compute rotational Jdotv helper quaternions
    omega1 = wp.spatial_top(cvel_in[worldid, body1])
    omega2 = wp.spatial_top(cvel_in[worldid, body2])
    domega = omega1 - omega2

    omega1_q = wp.quat(0.0, omega1[0], omega1[1], omega1[2])
    omega2_q = wp.quat(0.0, omega2[0], omega2[1], omega2[2])
    domega_q = wp.quat(0.0, domega[0], domega[1], domega[2])

    if is_site:
      qdot0r = math.mul_quat(omega1_q, quat) * 0.5
      qfull1 = math.mul_quat(xquat_in[worldid, body2], site_quat[site_quat_id, obj2id])
      qdot1 = math.mul_quat(omega2_q, qfull1) * 0.5

      negqdot1 = math.quat_inv(qdot1)
      negq1 = wp.quat(qfull1[0], -qfull1[1], -qfull1[2], -qfull1[3])

    else:
      # qdot0 = mul_quat(xquat_in[worldid, body1], omega1_q) * 0.5
      u7 = xquat_in[worldid, body1]
      qdot0 = math.mul_quat(omega1_q, xquat_in[worldid, body1]) * 0.5
      qdot0r = math.mul_quat(qdot0, relpose)
      q1_non_site = xquat_in[worldid, body2]
      qdot1 = math.mul_quat(omega2_q, q1_non_site) * 0.5

      negqdot1 = math.quat_inv(qdot1)
      negq1 = wp.quat(q1_non_site[0], -q1_non_site[1], -q1_non_site[2], -q1_non_site[3])

    rownnz = int(0)
    rowadr = int(0)
    if wp.static(is_sparse):
      # TODO(team): pre-compute number of non-zeros
      body1 = body_weldid[body1]
      body2 = body_weldid[body2]

      # count non-zeros
      pda1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
      pda2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)
      while pda1 >= 0 or pda2 >= 0:
        da = wp.max(pda1, pda2)
        if pda1 == da:
          pda1 = dof_parentid[da]
        if pda2 == da:
          pda2 = dof_parentid[da]
        rownnz += 1

      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 6, rownnz, efc_nnz_out)
      if rowadr < 0:
        return

    wp.atomic_add(ne_out, worldid, 6)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, 6)  # kernel_analyzer: ignore[atomic]

    if efcid >= njmax_in:
      return

    # rows past njmax are dropped
    nrow = wp.min(6, njmax_in - efcid)

    if wp.static(is_sparse and newton):
      jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efc_jtdaj_adr_out[worldid, jgid] = efcid
      efc_jtdaj_nrow_out[worldid, jgid] = nrow

    # compute Jacobian difference (opposite of contact: 0 - 1)
    Jqvelp = wp.vec3f(0.0, 0.0, 0.0)
    Jqvelr = wp.vec3f(0.0, 0.0, 0.0)
    Jdotv_p = wp.vec3f(0.0, 0.0, 0.0)
    Jdotv_r0 = wp.vec3f(0.0, 0.0, 0.0)

    if wp.static(is_sparse):
      da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
      da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)

      for i in range(6):
        if i < nrow:
          efc_J_rowadr_out[worldid, efcid + i] = rowadr + i * rownnz
          efc_J_rownnz_out[worldid, efcid + i] = rownnz

      # compute J and colind
      nnz = int(0)
      while da1 >= 0 or da2 >= 0:
        da = wp.max(da1, da2)
        if da1 == da:
          da1 = dof_parentid[da]
        if da2 == da:
          da2 = dof_parentid[da]

        jacp1, jacr1 = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos1,
          body1,
          da,
          worldid,
        )
        jacp2, jacr2 = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos2,
          body2,
          da,
          worldid,
        )

        jacdifp = jacp1 - jacp2

        jacdifr = (jacr1 - jacr2) * torquescale
        jacdifrq = math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)
        jacdifr = 0.5 * wp.vec3(jacdifrq[1], jacdifrq[2], jacdifrq[3])

        jacp1_dot, jacr1_dot = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos1,
          body1,
          da,
          worldid,
        )
        jacp2_dot, jacr2_dot = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos2,
          body2,
          da,
          worldid,
        )

        jacdifp_dot = jacp1_dot - jacp2_dot
        jacdifr_dot = jacr1_dot - jacr2_dot

        # constant indices keep jacdifp and jacdifr in registers
        for i in range(3):
          if i < nrow:
            sparseid = rowadr + i * rownnz + nnz
            efc_J_colind_out[worldid, 0, sparseid] = da
            efc_J_out[worldid, 0, sparseid] = jacdifp[i]
          if 3 + i < nrow:
            sparseid = rowadr + (3 + i) * rownnz + nnz
            efc_J_colind_out[worldid, 0, sparseid] = da
            efc_J_out[worldid, 0, sparseid] = jacdifr[i]

        Jqvelp += jacdifp * qvel_in[worldid, da]
        Jqvelr += jacdifr * qvel_in[worldid, da]
        Jdotv_p += jacdifp_dot * qvel_in[worldid, da]
        Jdotv_r0 += jacdifr_dot * qvel_in[worldid, da]

        nnz += 1
    else:
      for dofid in range(nv):
        jacp1, jacr1 = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos1,
          body1,
          dofid,
          worldid,
        )
        jacp2, jacr2 = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          pos2,
          body2,
          dofid,
          worldid,
        )

        jacdifp = jacp1 - jacp2

        for i in range(3):
          if i < nrow:
            efc_J_out[worldid, efcid + i, dofid] = jacdifp[i]

        jacdifr = (jacr1 - jacr2) * torquescale
        jacdifrq = math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)
        jacdifr = 0.5 * wp.vec3(jacdifrq[1], jacdifrq[2], jacdifrq[3])

        jacp1_dot, jacr1_dot = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos1,
          body1,
          dofid,
          worldid,
        )
        jacp2_dot, jacr2_dot = support.jac_dot_dof(
          body_parentid,
          body_rootid,
          jnt_type,
          jnt_dofadr,
          dof_bodyid,
          dof_jntid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          cvel_in,
          cdof_dot_in,
          pos2,
          body2,
          dofid,
          worldid,
        )

        jacdifp_dot = jacp1_dot - jacp2_dot
        jacdifr_dot = jacr1_dot - jacr2_dot

        for i in range(3):
          if 3 + i < nrow:
            efc_J_out[worldid, efcid + 3 + i, dofid] = jacdifr[i]

        Jqvelp += jacdifp * qvel_in[worldid, dofid]
        Jqvelr += jacdifr * qvel_in[worldid, dofid]
        Jdotv_p += jacdifp_dot * qvel_in[worldid, dofid]
        Jdotv_r0 += jacdifr_dot * qvel_in[worldid, dofid]

    # error is difference in global position and orientation
    cpos = pos1 - pos2

    crotq = math.mul_quat(quat1, quat)  # copy axis components
    crot = wp.vec3(crotq[1], crotq[2], crotq[3]) * torquescale

    body_invweight0_id = worldid % body_invweight0.shape[0]
    invweight_t = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]

    pos_imp = wp.sqrt(wp.length_sq(cpos) + wp.length_sq(crot))

    solref = eq_solref[worldid % eq_solref.shape[0], eqid]
    solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    djrdv_q = wp.quat(0.0, Jdotv_r0[0], Jdotv_r0[1], Jdotv_r0[2])

    # Term 1: negqdot1 * domega * q0r
    t1a = math.mul_quat(negqdot1, domega_q)
    t1 = math.mul_quat(t1a, quat)

    # Term 2: negq1 * djrdv * q0r
    t2a = math.mul_quat(negq1, djrdv_q)
    t2 = math.mul_quat(t2a, quat)

    # Term 3: negq1 * domega * qdot0r
    t3a = math.mul_quat(negq1, domega_q)
    t3 = math.mul_quat(t3a, qdot0r)

    Jdotv_r = wp.vec3(t1[1] + t2[1] + t3[1], t1[2] + t2[2] + t3[2], t1[3] + t2[3] + t3[3]) * 0.5 * torquescale

    for i in range(3):
      if i < nrow:
        _efc_row(
          opt_disableflags,
          worldid,
          timestep,
          efcid + i,
          cpos[i],
          pos_imp,
          invweight_t,
          solref,
          solimp,
          0.0,
          Jqvelp[i],
          0.0,
          ConstraintType.EQUALITY,
          eqid,
          is_discrete,
          Jdotv_p[i],
          efc_type_out,
          efc_id_out,
          efc_pos_out,
          efc_margin_out,
          efc_D_out,
          efc_vel_out,
          efc_aref_out,
          efc_frictionloss_out,
        )

    invweight_r = body_invweight0[body_invweight0_id, body1][1] + body_invweight0[body_invweight0_id, body2][1]

    for i in range(3):
      if 3 + i < nrow:
        _efc_row(
          opt_disableflags,
          worldid,
          timestep,
          efcid + 3 + i,
          crot[i],
          pos_imp,
          invweight_r,
          solref,
          solimp,
          0.0,
          Jqvelr[i],
          0.0,
          ConstraintType.EQUALITY,
          eqid,
          is_discrete,
          Jdotv_r[i],
          efc_type_out,
          efc_id_out,
          efc_pos_out,
          efc_margin_out,
          efc_D_out,
          efc_vel_out,
          efc_aref_out,
          efc_frictionloss_out,
        )

  return kernel


@cache_kernel
def _equality_flexstrain(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_invweight0: wp.array2d[wp.vec2],
    dof_bodyid: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_nodenum: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_node: wp.array[wp.vec3],
    flex_node0: wp.array[wp.vec3],
    flex_stiffness: wp.array[float],
    flex_centered: wp.array[bool],
    eq_obj1id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec12],
    body_isdofancestor: wp.array2d[int],
    eq_flexstrain_adr: wp.array[int],
    flexstrain_J_rownnz: wp.array[int],
    flexstrain_J_rowadr: wp.array[int],
    flexstrain_J_colind: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    eq_active_in: wp.array2d[bool],
    xpos_in: wp.array2d[wp.vec3],
    xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    njmax_in: int,
    njmax_nnz_in: int,
    flexnode_xpos_in: wp.array2d[wp.vec3],
    face_quat_in: wp.array2d[wp.quat],
    # Data out:
    ne_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, eqstrainid = wp.tid()
    eqid = eq_flexstrain_adr[eqstrainid]

    if not eq_active_in[worldid, eqid]:
      return

    f = eq_obj1id[eqid]
    order = flex_interp[f]
    if order == 0:
      return

    # nodes per cell
    npc = wp.where(order < 0, 4, 8)

    # cell indices from eq_data
    data = eq_data[worldid % eq_data.shape[0], eqid]
    ci = int(data[0])
    cj = int(data[1])
    ck = int(data[2])

    cellnum = flex_cellnum[f]
    cx = cellnum[0]
    cy = cellnum[1]
    cz = cellnum[2]
    nstart = flex_nodeadr[f]
    ny_g = cy + 1
    nz_g = cz + 1

    ndof_cell = 3 * npc

    # read eigenmode data from flex_stiffness
    cell_idx = wp.where(order < 0, ci, ci * cy * cz + cj * cz + ck)
    normal_axis = 0
    g_fixed = 0
    q0 = 0
    q1 = 0
    ny_g_face = 0
    nz_g_face = 0
    if order < 0:
      normal_axis, g_fixed, q0, q1, ny_g_face, nz_g_face = support.get_face_metadata(cx, cy, cz, cell_idx, 1)
    k_base = flex_stiffnessadr[f] + cell_idx * ndof_cell * ndof_cell
    neig = int(flex_stiffness[k_base])

    if neig == 0:
      return

    solref = eq_solref[worldid % eq_solref.shape[0], eqid]
    solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]

    # gather cell node positions and reference positions (order==1: npc=8)
    # We compute the corotational quaternion from the deformation gradient
    # at the cell center (0.5, 0.5, 0.5)

    cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)
    if order < 0:
      cell_quat = face_quat_in[worldid, cell_idx]
    else:
      cell_quat = support.compute_interp_cell_quat(
        flex_node0,
        flexnode_xpos_in,
        1,
        ci,
        cj,
        ck,
        cy,
        cz,
        ny_g,
        nz_g,
        nstart,
        worldid,
      )
    cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])

    # Compute average invweight across cell nodes (translation component)
    avg_invweight = float(0.0)
    idx_iw = int(0)
    for li_iw in range(2):
      for lj_iw in range(2):
        for lk_iw in range(2):
          if idx_iw < npc:
            gidx_iw = wp.where(
              order < 0,
              support.gather_face_node_index_fast(
                normal_axis,
                g_fixed,
                q0,
                q1,
                ny_g_face,
                nz_g_face,
                idx_iw,
                1,
              ),
              (ci + li_iw) * ny_g * nz_g + (cj + lj_iw) * nz_g + (ck + lk_iw),
            )
            bodyid_iw = flex_nodebodyid[nstart + gidx_iw]
            avg_invweight += body_invweight0[worldid % body_invweight0.shape[0], bodyid_iw][0]
            idx_iw += 1
    avg_invweight /= float(npc)

    rownnz = flexstrain_J_rownnz[eqstrainid]
    fs_rowadr = flexstrain_J_rowadr[eqstrainid]

    # the eigenmode rows of a cell share their columns: allocate them as one block
    efc_rowadr0 = int(0)
    if wp.static(is_sparse):
      efc_rowadr0 = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, neig, rownnz, efc_nnz_out)
      if efc_rowadr0 < 0:
        return

    wp.atomic_add(ne_out, worldid, neig)  # kernel_analyzer: ignore[atomic]
    efcid0 = wp.atomic_add(nefc_out, worldid, neig)  # kernel_analyzer: ignore[atomic]

    if efcid0 >= njmax_in:
      return

    # rows past njmax are dropped
    nrow = wp.min(neig, njmax_in - efcid0)

    if wp.static(is_sparse and newton):
      jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efc_jtdaj_adr_out[worldid, jgid] = efcid0
      efc_jtdaj_nrow_out[worldid, jgid] = nrow

    # Loop over eigenmodes
    for eig in range(nrow):
      efcid = efcid0 + eig
      efc_rowadr = efc_rowadr0 + eig * rownnz

      # Read eigenvector from stiffness data
      eigvec_base = k_base + 1 + eig * ndof_cell

      # Compute constraint residual: dot(eigvec, displacement_in_corot_frame)
      residual = float(0.0)
      idx2 = int(0)
      for li2 in range(2):
        for lj2 in range(2):
          for lk2 in range(2):
            if idx2 < npc:
              gidx2 = wp.where(
                order < 0,
                support.gather_face_node_index_fast(
                  normal_axis,
                  g_fixed,
                  q0,
                  q1,
                  ny_g_face,
                  nz_g_face,
                  idx2,
                  1,
                ),
                (ci + li2) * ny_g * nz_g + (cj + lj2) * nz_g + (ck + lk2),
              )

              xpos_n = flexnode_xpos_in[worldid, nstart + gidx2]
              refpos_n = flex_node0[nstart + gidx2]

              # Rotate xpos to corotational frame
              xrot = wp.quat_rotate(cell_quat_inv, xpos_n)
              displ = wp.vec3(xrot[0] - refpos_n[0], xrot[1] - refpos_n[1], xrot[2] - refpos_n[2])

              for c in range(3):
                residual += flex_stiffness[eigvec_base + 3 * idx2 + c] * displ[c]

              idx2 += 1

      efc_pos_out[worldid, efcid] = residual

      # Compute Jacobian: J_q = dSdx_world * dxpos/dq
      # dSdx_world = R_inv * eigvec (rotate eigenvector to world frame)
      # Then contract with per-node body Jacobians

      Jqvel = float(0.0)
      invweight = float(0.0)

      if wp.static(is_sparse):
        efc_J_rownnz_out[worldid, efcid] = rownnz
        efc_J_rowadr_out[worldid, efcid] = efc_rowadr
      else:
        for q in range(nv):
          efc_J_out[worldid, efcid, q] = 0.0

      for sparseid in range(rownnz):
        q = flexstrain_J_colind[fs_rowadr + sparseid]
        J_val = float(0.0)
        idx3 = int(0)
        for li3 in range(2):
          for lj3 in range(2):
            for lk3 in range(2):
              if idx3 < npc:
                gidx3 = wp.where(
                  order < 0,
                  support.gather_face_node_index_fast(
                    normal_axis,
                    g_fixed,
                    q0,
                    q1,
                    ny_g_face,
                    nz_g_face,
                    idx3,
                    1,
                  ),
                  (ci + li3) * ny_g * nz_g + (cj + lj3) * nz_g + (ck + lk3),
                )

                bodyid3 = flex_nodebodyid[nstart + gidx3]
                xpos_n3 = flexnode_xpos_in[worldid, nstart + gidx3]

                if body_isdofancestor[bodyid3, q] != 0:
                  # Rotate eigenvector component to world frame
                  eig_local = wp.vec3(
                    flex_stiffness[eigvec_base + 3 * idx3 + 0],
                    flex_stiffness[eigvec_base + 3 * idx3 + 1],
                    flex_stiffness[eigvec_base + 3 * idx3 + 2],
                  )
                  eig_world = wp.quat_rotate(cell_quat, eig_local)

                  jacp, jacr = support.jac_dof(
                    body_parentid,
                    body_rootid,
                    dof_bodyid,
                    body_isdofancestor,
                    subtree_com_in,
                    cdof_in,
                    xpos_n3,
                    bodyid3,
                    q,
                    worldid,
                  )
                  J_val += wp.dot(eig_world, jacp)
                idx3 += 1

        if wp.static(is_sparse):
          efc_J_colind_out[worldid, 0, efc_rowadr + sparseid] = q
          efc_J_out[worldid, 0, efc_rowadr + sparseid] = J_val
        else:
          efc_J_out[worldid, efcid, q] = J_val
        Jqvel += J_val * qvel_in[worldid, q]

      invweight = avg_invweight

      _efc_row(
        opt_disableflags,
        worldid,
        opt_timestep[worldid % opt_timestep.shape[0]],
        efcid,
        residual,
        residual,
        invweight,
        solref,
        solimp,
        0.0,
        Jqvel,
        0.0,
        ConstraintType.EQUALITY,
        eqid,
        is_discrete,
        0.0,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
      )

  return kernel


@cache_kernel
def _friction_dof(is_sparse: bool, is_discrete: bool = False):
  # sparse Newton adds these rows to the Hessian diagonal directly, so they get no JTDAJ block
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    dof_solref: wp.array2d[wp.vec2],
    dof_solimp: wp.array2d[vec5],
    dof_frictionloss: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    # Data in:
    qvel_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    nf_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    dof_frictionadr_out: wp.array2d[int],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, dofid = wp.tid()

    dof_frictionloss_id = worldid % dof_frictionloss.shape[0]

    if dof_frictionloss[dof_frictionloss_id, dofid] <= 0.0:
      dof_frictionadr_out[worldid, dofid] = -1
      return

    rowadr = int(0)
    if wp.static(is_sparse):
      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, 1, efc_nnz_out)
      if rowadr < 0:
        dof_frictionadr_out[worldid, dofid] = -1
        return

    wp.atomic_add(nf_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

    if efcid >= njmax_in:
      dof_frictionadr_out[worldid, dofid] = -1
      return

    if wp.static(is_sparse):
      efc_J_rownnz_out[worldid, efcid] = 1
      efc_J_rowadr_out[worldid, efcid] = rowadr
      efc_J_colind_out[worldid, 0, rowadr] = dofid
      efc_J_out[worldid, 0, rowadr] = 1.0
    else:
      for i in range(nv):
        efc_J_out[worldid, efcid, i] = 0.0
      efc_J_out[worldid, efcid, dofid] = 1.0

    dof_frictionadr_out[worldid, dofid] = efcid

    Jqvel = qvel_in[worldid, dofid]

    dof_invweight0_id = worldid % dof_invweight0.shape[0]
    dof_solref_id = worldid % dof_solref.shape[0]
    dof_solimp_id = worldid % dof_solimp.shape[0]
    _efc_row(
      opt_disableflags,
      worldid,
      opt_timestep[worldid % opt_timestep.shape[0]],
      efcid,
      0.0,
      0.0,
      dof_invweight0[dof_invweight0_id, dofid],
      dof_solref[dof_solref_id, dofid],
      dof_solimp[dof_solimp_id, dofid],
      0.0,
      Jqvel,
      dof_frictionloss[dof_frictionloss_id, dofid],
      ConstraintType.FRICTION_DOF,
      dofid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

  return kernel


@cache_kernel
def _friction_tendon(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_solref_fri: wp.array2d[wp.vec2],
    tendon_solimp_fri: wp.array2d[vec5],
    tendon_frictionloss: wp.array2d[float],
    tendon_invweight0: wp.array2d[float],
    # Data in:
    qvel_in: wp.array2d[float],
    ten_J_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    nf_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, tenid = wp.tid()

    tendon_frictionloss_id = worldid % tendon_frictionloss.shape[0]

    frictionloss = tendon_frictionloss[tendon_frictionloss_id, tenid]
    if frictionloss <= 0.0:
      return

    rownnz_tenJ = ten_J_rownnz[tenid]
    rowadr_efc = int(0)
    if wp.static(is_sparse):
      rowadr_efc = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, rownnz_tenJ, efc_nnz_out)
      if rowadr_efc < 0:
        return

    wp.atomic_add(nf_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
    efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

    if efcid >= njmax_in:
      return

    if wp.static(is_sparse and newton):
      jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efc_jtdaj_adr_out[worldid, jgid] = efcid
      efc_jtdaj_nrow_out[worldid, jgid] = 1

    Jqvel = float(0.0)

    rowadr_tenJ = ten_J_rowadr[tenid]
    if wp.static(is_sparse):
      efc_J_rownnz_out[worldid, efcid] = rownnz_tenJ
      efc_J_rowadr_out[worldid, efcid] = rowadr_efc

      for i in range(rownnz_tenJ):
        sparseid_ten = rowadr_tenJ + i
        sparseid_efc = rowadr_efc + i
        colind = ten_J_colind[sparseid_ten]
        J = ten_J_in[worldid, sparseid_ten]
        efc_J_colind_out[worldid, 0, sparseid_efc] = colind
        efc_J_out[worldid, 0, sparseid_efc] = J
        Jqvel += J * qvel_in[worldid, colind]
    else:
      nnz = int(0)
      colind = ten_J_colind[rowadr_tenJ]
      for i in range(nv):
        if nnz < rownnz_tenJ and i == colind:
          J = ten_J_in[worldid, rowadr_tenJ + nnz]
          efc_J_out[worldid, efcid, i] = J
          Jqvel += J * qvel_in[worldid, i]
          nnz += 1
          if nnz < rownnz_tenJ:
            colind = ten_J_colind[rowadr_tenJ + nnz]
        else:
          efc_J_out[worldid, efcid, i] = 0.0

    tendon_invweight0_id = worldid % tendon_invweight0.shape[0]
    tendon_solref_fri_id = worldid % tendon_solref_fri.shape[0]
    tendon_solimp_fri_id = worldid % tendon_solimp_fri.shape[0]
    _efc_row(
      opt_disableflags,
      worldid,
      opt_timestep[worldid % opt_timestep.shape[0]],
      efcid,
      0.0,
      0.0,
      tendon_invweight0[tendon_invweight0_id, tenid],
      tendon_solref_fri[tendon_solref_fri_id, tenid],
      tendon_solimp_fri[tendon_solimp_fri_id, tenid],
      0.0,
      Jqvel,
      frictionloss,
      ConstraintType.FRICTION_TENDON,
      tenid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

  return kernel


@cache_kernel
def _limit_slide_hinge(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_solimp: wp.array2d[vec5],
    jnt_range: wp.array2d[wp.vec2],
    jnt_margin: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    jnt_limited_slide_hinge_adr: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    nl_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, jntlimitedid = wp.tid()
    jntid = jnt_limited_slide_hinge_adr[jntlimitedid]
    jnt_range_id = worldid % jnt_range.shape[0]
    jntrange = jnt_range[jnt_range_id, jntid]

    qpos = qpos_in[worldid, jnt_qposadr[jntid]]
    jnt_margin_id = worldid % jnt_margin.shape[0]
    jntmargin = jnt_margin[jnt_margin_id, jntid]
    dist_min, dist_max = qpos - jntrange[0], jntrange[1] - qpos
    pos = wp.min(dist_min, dist_max) - jntmargin
    active = pos < 0

    if active:
      rowadr = int(0)
      if wp.static(is_sparse):
        rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, 1, efc_nnz_out)
        if rowadr < 0:
          return

      wp.atomic_add(nl_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

      if efcid >= njmax_in:
        return

      if wp.static(is_sparse and newton):
        jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
        efc_jtdaj_adr_out[worldid, jgid] = efcid
        efc_jtdaj_nrow_out[worldid, jgid] = 1

      dofadr = jnt_dofadr[jntid]

      J = float(dist_min < dist_max) * 2.0 - 1.0

      if wp.static(is_sparse):
        efc_J_rownnz_out[worldid, efcid] = 1
        efc_J_rowadr_out[worldid, efcid] = rowadr
        efc_J_colind_out[worldid, 0, rowadr] = dofadr
        efc_J_out[worldid, 0, rowadr] = J
      else:
        for i in range(nv):
          efc_J_out[worldid, efcid, i] = 0.0
        efc_J_out[worldid, efcid, dofadr] = J

      Jqvel = J * qvel_in[worldid, dofadr]

      dof_invweight0_id = worldid % dof_invweight0.shape[0]
      jnt_solref_id = worldid % jnt_solref.shape[0]
      jnt_solimp_id = worldid % jnt_solimp.shape[0]
      _efc_row(
        opt_disableflags,
        worldid,
        opt_timestep[worldid % opt_timestep.shape[0]],
        efcid,
        pos,
        pos,
        dof_invweight0[dof_invweight0_id, dofadr],
        jnt_solref[jnt_solref_id, jntid],
        jnt_solimp[jnt_solimp_id, jntid],
        jntmargin,
        Jqvel,
        0.0,
        ConstraintType.LIMIT_JOINT,
        jntid,
        is_discrete,
        0.0,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
      )

  return kernel


@cache_kernel
def _limit_ball(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_solimp: wp.array2d[vec5],
    jnt_range: wp.array2d[wp.vec2],
    jnt_margin: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    jnt_limited_ball_adr: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    nl_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, jntlimitedid = wp.tid()
    jntid = jnt_limited_ball_adr[jntlimitedid]
    qposadr = jnt_qposadr[jntid]

    qpos = qpos_in[worldid]
    jnt_quat = wp.quat(qpos[qposadr + 0], qpos[qposadr + 1], qpos[qposadr + 2], qpos[qposadr + 3])
    jnt_quat = wp.normalize(jnt_quat)
    axis_angle = math.quat_to_vel(jnt_quat)
    jnt_range_id = worldid % jnt_range.shape[0]
    jntrange = jnt_range[jnt_range_id, jntid]
    axis, angle = math.normalize_with_norm(axis_angle)
    jnt_margin_id = worldid % jnt_margin.shape[0]
    jntmargin = jnt_margin[jnt_margin_id, jntid]

    pos = wp.max(jntrange[0], jntrange[1]) - angle - jntmargin
    active = pos < 0

    if active:
      rowadr = int(0)
      if wp.static(is_sparse):
        rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, 3, efc_nnz_out)
        if rowadr < 0:
          return

      wp.atomic_add(nl_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

      if efcid >= njmax_in:
        return

      if wp.static(is_sparse and newton):
        jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
        efc_jtdaj_adr_out[worldid, jgid] = efcid
        efc_jtdaj_nrow_out[worldid, jgid] = 1

      dofadr = jnt_dofadr[jntid]
      dof0 = dofadr + 0
      dof1 = dofadr + 1
      dof2 = dofadr + 2

      if wp.static(is_sparse):
        efc_J_rownnz_out[worldid, efcid] = 3
        efc_J_rowadr_out[worldid, efcid] = rowadr

        sparseid0 = rowadr + 0
        sparseid1 = rowadr + 1
        sparseid2 = rowadr + 2

        efc_J_colind_out[worldid, 0, sparseid0] = dof0
        efc_J_colind_out[worldid, 0, sparseid1] = dof1
        efc_J_colind_out[worldid, 0, sparseid2] = dof2

        efc_J_out[worldid, 0, sparseid0] = -axis[0]
        efc_J_out[worldid, 0, sparseid1] = -axis[1]
        efc_J_out[worldid, 0, sparseid2] = -axis[2]
      else:
        for i in range(nv):
          efc_J_out[worldid, efcid, i] = 0.0
        efc_J_out[worldid, efcid, dof0] = -axis[0]
        efc_J_out[worldid, efcid, dof1] = -axis[1]
        efc_J_out[worldid, efcid, dof2] = -axis[2]

      Jqvel = -axis[0] * qvel_in[worldid, dof0]
      Jqvel -= axis[1] * qvel_in[worldid, dof1]
      Jqvel -= axis[2] * qvel_in[worldid, dof2]

      dof_invweight0_id = worldid % dof_invweight0.shape[0]
      jnt_solref_id = worldid % jnt_solref.shape[0]
      jnt_solimp_id = worldid % jnt_solimp.shape[0]
      _efc_row(
        opt_disableflags,
        worldid,
        opt_timestep[worldid % opt_timestep.shape[0]],
        efcid,
        pos,
        pos,
        dof_invweight0[dof_invweight0_id, dofadr],
        jnt_solref[jnt_solref_id, jntid],
        jnt_solimp[jnt_solimp_id, jntid],
        jntmargin,
        Jqvel,
        0.0,
        ConstraintType.LIMIT_JOINT,
        jntid,
        is_discrete,
        0.0,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
      )

  return kernel


@cache_kernel
def _limit_tendon(is_sparse: bool, newton: bool, is_discrete: bool = False):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_solref_lim: wp.array2d[wp.vec2],
    tendon_solimp_lim: wp.array2d[vec5],
    tendon_range: wp.array2d[wp.vec2],
    tendon_margin: wp.array2d[float],
    tendon_invweight0: wp.array2d[float],
    tendon_limited_adr: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    ten_J_in: wp.array2d[float],
    ten_length_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    nl_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    worldid, tenlimitedid = wp.tid()
    tenid = tendon_limited_adr[tenlimitedid]

    tendon_range_id = worldid % tendon_range.shape[0]
    tenrange = tendon_range[tendon_range_id, tenid]
    length = ten_length_in[worldid, tenid]
    dist_min, dist_max = length - tenrange[0], tenrange[1] - length
    tendon_margin_id = worldid % tendon_margin.shape[0]
    tenmargin = tendon_margin[tendon_margin_id, tenid]
    pos = wp.min(dist_min, dist_max) - tenmargin
    active = pos < 0

    if active:
      rownnz_tenJ = ten_J_rownnz[tenid]
      rowadr_efc = int(0)
      if wp.static(is_sparse):
        rowadr_efc = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, 1, rownnz_tenJ, efc_nnz_out)
        if rowadr_efc < 0:
          return

      wp.atomic_add(nl_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
      efcid = wp.atomic_add(nefc_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

      if efcid >= njmax_in:
        return

      if wp.static(is_sparse and newton):
        jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
        efc_jtdaj_adr_out[worldid, jgid] = efcid
        efc_jtdaj_nrow_out[worldid, jgid] = 1

      Jqvel = float(0.0)
      scl = float(dist_min < dist_max) * 2.0 - 1.0

      rowadr_tenJ = ten_J_rowadr[tenid]
      if wp.static(is_sparse):
        efc_J_rownnz_out[worldid, efcid] = rownnz_tenJ
        efc_J_rowadr_out[worldid, efcid] = rowadr_efc

        for i in range(rownnz_tenJ):
          sparseid_ten = rowadr_tenJ + i
          sparseid_efc = rowadr_efc + i
          colind = ten_J_colind[sparseid_ten]
          J = scl * ten_J_in[worldid, sparseid_ten]
          efc_J_colind_out[worldid, 0, sparseid_efc] = colind
          efc_J_out[worldid, 0, sparseid_efc] = J
          Jqvel += J * qvel_in[worldid, colind]
      else:
        nnz = int(0)
        colind = ten_J_colind[rowadr_tenJ]
        for i in range(nv):
          if nnz < rownnz_tenJ and i == colind:
            J = scl * ten_J_in[worldid, rowadr_tenJ + nnz]
            efc_J_out[worldid, efcid, i] = J
            Jqvel += J * qvel_in[worldid, i]
            nnz += 1
            if nnz < rownnz_tenJ:
              colind = ten_J_colind[rowadr_tenJ + nnz]
          else:
            efc_J_out[worldid, efcid, i] = 0.0

      tendon_invweight0_id = worldid % tendon_invweight0.shape[0]
      tendon_solref_lim_id = worldid % tendon_solref_lim.shape[0]
      tendon_solimp_lim_id = worldid % tendon_solimp_lim.shape[0]
      _efc_row(
        opt_disableflags,
        worldid,
        opt_timestep[worldid % opt_timestep.shape[0]],
        efcid,
        pos,
        pos,
        tendon_invweight0[tendon_invweight0_id, tenid],
        tendon_solref_lim[tendon_solref_lim_id, tenid],
        tendon_solimp_lim[tendon_solimp_lim_id, tenid],
        tenmargin,
        Jqvel,
        0.0,
        ConstraintType.LIMIT_TENDON,
        tenid,
        is_discrete,
        0.0,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
      )

  return kernel


@cache_kernel
def _efc_contact_init(cone_type: types.ConeType, is_sparse: bool, newton: bool, flg_adhesion: bool):
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  IS_SPARSE = is_sparse

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    # Data in:
    njmax_in: int,
    njmax_nnz_in: int,
    nacon_in: wp.array[int],
    # In:
    dist_in: wp.array[float],
    condim_in: wp.array[int],
    includemargin_in: wp.array[float],
    adhesion_in: wp.array[float],
    worldid_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    type_in: wp.array[int],
    # Data out:
    nefc_out: wp.array[int],
    contact_efc_address_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_Jqvel_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    conid = wp.tid()

    if conid >= nacon_in[0]:
      return

    if not (type_in[conid] & ContactType.CONSTRAINT):
      return

    condim = condim_in[conid]

    includemargin = includemargin_in[conid]
    pos = dist_in[conid] - includemargin
    if wp.static(flg_adhesion):
      active = (pos < 0.0) or (adhesion_in[conid] != 0.0)
    else:
      active = pos < 0.0

    if not active:
      return

    if wp.static(IS_ELLIPTIC):
      ndim = condim
    else:
      if condim == 1:
        ndim = 1
      else:
        ndim = 2 * (condim - 1)

    worldid = worldid_in[conid]

    rownnz = int(0)
    rowadr = int(0)
    if wp.static(IS_SPARSE):
      geom = geom_in[conid]
      body1 = body_weldid[geom_bodyid[geom[0]]]
      body2 = body_weldid[geom_bodyid[geom[1]]]

      da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
      da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)

      # count non-zeros
      while da1 >= 0 or da2 >= 0:
        da = wp.max(da1, da2)
        # skip common dofs
        if da1 == da and da2 == da:
          break
        if da1 == da:
          da1 = dof_parentid[da1]
        if da2 == da:
          da2 = dof_parentid[da2]
        rownnz += 1

      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, ndim, rownnz, efc_nnz_out)
      if rowadr < 0:
        for dim in range(ndim):
          contact_efc_address_out[conid, dim] = -1
        return

    # Allocate contiguous block of efcids for all dimids
    base_efcid = wp.atomic_add(nefc_out, worldid, ndim)  # kernel_analyzer: ignore[atomic]
    for dim in range(ndim):
      efcid = base_efcid + dim
      if efcid >= njmax_in:
        contact_efc_address_out[conid, dim] = -1
      else:
        contact_efc_address_out[conid, dim] = efcid
        # This is redundant with the _efc_row call later but needed for the jac calculation
        efc_id_out[worldid, efcid] = conid
        if wp.static(IS_SPARSE):
          # the sparse Jacobian kernel accumulates into this
          efc_Jqvel_out[worldid, efcid] = 0.0

    if wp.static(is_sparse and newton):
      if base_efcid < njmax_in:
        jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
        efc_jtdaj_adr_out[worldid, jgid] = base_efcid
        efc_jtdaj_nrow_out[worldid, jgid] = wp.min(ndim, njmax_in - base_efcid)

    if wp.static(IS_SPARSE):
      for dim in range(ndim):
        efcid = base_efcid + dim
        if efcid < njmax_in:
          efc_J_rowadr_out[worldid, efcid] = rowadr + dim * rownnz
          efc_J_rownnz_out[worldid, efcid] = rownnz

  return kernel


@cache_kernel
def _efc_contact_init_flex(cone_type: types.ConeType, is_sparse: bool, newton: bool, flg_adhesion: bool):
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  IS_SPARSE = is_sparse
  HAS_FLEX = True

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_vert0: wp.array[wp.vec3],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_jnt: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_vert_rodelem: wp.array[int],
    # Data in:
    flexvert_xpos_in: wp.array2d[wp.vec3],
    njmax_in: int,
    njmax_nnz_in: int,
    nacon_in: wp.array[int],
    # In:
    dist_in: wp.array[float],
    condim_in: wp.array[int],
    includemargin_in: wp.array[float],
    adhesion_in: wp.array[float],
    worldid_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    flex_in: wp.array[wp.vec2i],
    elem_in: wp.array[wp.vec2i],
    vert_in: wp.array[wp.vec2i],
    pos_in: wp.array[wp.vec3],
    type_in: wp.array[int],
    # Data out:
    nefc_out: wp.array[int],
    contact_efc_address_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    # Out:
    efc_nnz_out: wp.array[int],
  ):
    conid = wp.tid()

    if conid >= nacon_in[0]:
      return

    if not (type_in[conid] & ContactType.CONSTRAINT):
      return

    condim = condim_in[conid]

    geom = geom_in[conid]
    includemargin = includemargin_in[conid]
    pos = dist_in[conid] - includemargin
    active = pos < 0.0
    if wp.static(flg_adhesion):
      active = active or (adhesion_in[conid] != 0.0)

    if not active:
      return

    if wp.static(IS_ELLIPTIC):
      ndim = condim
    else:
      if condim == 1:
        ndim = 1
      else:
        ndim = 2 * (condim - 1)

    worldid = worldid_in[conid]

    rownnz = int(0)
    rowadr = int(0)
    if wp.static(IS_SPARSE):
      geom = geom_in[conid]
      flex = flex_in[conid]
      elem = elem_in[conid]
      vert = vert_in[conid]
      con_pos = pos_in[conid]

      is_interp = False
      if wp.static(HAS_FLEX):
        if geom[0] < 0 and flex[0] >= 0 and (vert[0] >= 0 or elem[0] >= 0):
          if flex_interp[flex[0]] != 0:
            is_interp = True
        if geom[1] < 0 and flex[1] >= 0 and (vert[1] >= 0 or elem[1] >= 0):
          if flex_interp[flex[1]] != 0:
            is_interp = True

      if is_interp:
        # Interpolated flex: sum of all contributing body dofnums after de-duplication
        local_bodies = support.vec_interp_body(-1)
        local_weights = support.vec_interp_weight(0.0)
        local_nb = int(0)

        for side in range(2):
          if geom[side] >= 0:
            b = body_weldid[geom_bodyid[geom[side]]]
            local_nb, local_bodies, local_weights = support.add_interp_weight(
              local_nb, local_bodies, local_weights, b, 1.0
            )
          elif flex[side] >= 0:
            f = flex[side]
            if flex_interp[f] != 0:
              local_nb, local_bodies, local_weights = support.add_flex_interp_weights(
                body_weldid,
                flex_dim,
                flex_interp,
                flex_cellnum,
                flex_nodeadr,
                flex_vertadr,
                flex_elemdataadr,
                flex_nodebodyid,
                flex_elem,
                flex_vert0,
                flexvert_xpos_in,
                f,
                elem[side],
                vert[side],
                con_pos,
                worldid,
                1.0,
                True,
                local_nb,
                local_bodies,
                local_weights,
              )
            else:
              body_ids, _weights = support.contact_bodies_and_weights(
                jnt_bodyid,
                geom_bodyid,
                flex_dim,
                flex_interp,
                flex_cellnum,
                flex_nodeadr,
                flex_vertadr,
                flex_elemadr,
                flex_elemnum,
                flex_elemdataadr,
                flex_nodebodyid,
                flex_vertbodyid,
                flex_elem,
                flex_vert0,
                flex_rod_has_twist,
                flex_rod_edgeadr,
                flex_rod_edgenum,
                flex_rodedge_jnt,
                flex_elem_rodedge,
                flex_vert_rodelem,
                flexvert_xpos_in,
                conid,
                side,
                geom,
                flex,
                elem,
                vert,
                con_pos,
                worldid,
              )
              for vi in range(4):
                if body_ids[vi] >= 0:
                  b = body_weldid[body_ids[vi]]
                  local_nb, local_bodies, local_weights = support.add_interp_weight(
                    local_nb, local_bodies, local_weights, b, 1.0
                  )

        # sum dofnums for unique bodies
        rownnz = int(0)
        for i in range(local_nb):
          rownnz += body_dofnum[local_bodies[i]]
      else:
        # Standard path (including elements up to 4 bodies)
        body_ids1, weights1 = support.contact_bodies_and_weights(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          conid,
          0,
          geom,
          flex,
          elem,
          vert,
          con_pos,
          worldid,
        )
        body_ids2, weights2 = support.contact_bodies_and_weights(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          conid,
          1,
          geom,
          flex,
          elem,
          vert,
          con_pos,
          worldid,
        )

        b1_0 = body_weldid[body_ids1[0]]
        b1_1 = body_weldid[body_ids1[1]] if body_ids1[1] >= 0 else -1
        b1_2 = body_weldid[body_ids1[2]] if body_ids1[2] >= 0 else -1
        b1_3 = body_weldid[body_ids1[3]] if body_ids1[3] >= 0 else -1

        b2_0 = body_weldid[body_ids2[0]]
        b2_1 = body_weldid[body_ids2[1]] if body_ids2[1] >= 0 else -1
        b2_2 = body_weldid[body_ids2[2]] if body_ids2[2] >= 0 else -1
        b2_3 = body_weldid[body_ids2[3]] if body_ids2[3] >= 0 else -1

        dof1_0 = int(body_dofadr[b1_0] + body_dofnum[b1_0] - 1) if b1_0 >= 0 else -1
        dof1_1 = int(body_dofadr[b1_1] + body_dofnum[b1_1] - 1) if b1_1 >= 0 else -1
        dof1_2 = int(body_dofadr[b1_2] + body_dofnum[b1_2] - 1) if b1_2 >= 0 else -1
        dof1_3 = int(body_dofadr[b1_3] + body_dofnum[b1_3] - 1) if b1_3 >= 0 else -1

        dof2_0 = int(body_dofadr[b2_0] + body_dofnum[b2_0] - 1) if b2_0 >= 0 else -1
        dof2_1 = int(body_dofadr[b2_1] + body_dofnum[b2_1] - 1) if b2_1 >= 0 else -1
        dof2_2 = int(body_dofadr[b2_2] + body_dofnum[b2_2] - 1) if b2_2 >= 0 else -1
        dof2_3 = int(body_dofadr[b2_3] + body_dofnum[b2_3] - 1) if b2_3 >= 0 else -1

        # count non-zeros
        rownnz = int(0)
        while (
          dof1_0 >= 0 or dof1_1 >= 0 or dof1_2 >= 0 or dof1_3 >= 0 or dof2_0 >= 0 or dof2_1 >= 0 or dof2_2 >= 0 or dof2_3 >= 0
        ):
          da1_max = wp.max(dof1_0, wp.max(dof1_1, wp.max(dof1_2, dof1_3)))
          da2_max = wp.max(dof2_0, wp.max(dof2_1, wp.max(dof2_2, dof2_3)))
          da = wp.max(da1_max, da2_max)

          if dof1_0 == da:
            dof1_0 = dof_parentid[dof1_0]
          if dof1_1 == da:
            dof1_1 = dof_parentid[dof1_1]
          if dof1_2 == da:
            dof1_2 = dof_parentid[dof1_2]
          if dof1_3 == da:
            dof1_3 = dof_parentid[dof1_3]

          if dof2_0 == da:
            dof2_0 = dof_parentid[dof2_0]
          if dof2_1 == da:
            dof2_1 = dof_parentid[dof2_1]
          if dof2_2 == da:
            dof2_2 = dof_parentid[dof2_2]
          if dof2_3 == da:
            dof2_3 = dof_parentid[dof2_3]

          rownnz += 1

      rowadr = _reserve_efc_nnz(nefc_out, njmax_in, njmax_nnz_in, worldid, ndim, rownnz, efc_nnz_out)
      if rowadr < 0:
        for dim in range(ndim):
          contact_efc_address_out[conid, dim] = -1
        return

    # Allocate contiguous block of efcids for all dimids
    base_efcid = wp.atomic_add(nefc_out, worldid, ndim)  # kernel_analyzer: ignore[atomic]
    for dim in range(ndim):
      efcid = base_efcid + dim
      if efcid >= njmax_in:
        contact_efc_address_out[conid, dim] = -1
      else:
        contact_efc_address_out[conid, dim] = efcid
        # This is redundant with the _efc_row call later but needed for the jac calculation
        efc_id_out[worldid, efcid] = conid

    if wp.static(is_sparse and newton):
      if base_efcid < njmax_in:
        jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
        efc_jtdaj_adr_out[worldid, jgid] = base_efcid
        efc_jtdaj_nrow_out[worldid, jgid] = wp.min(ndim, njmax_in - base_efcid)

    if wp.static(IS_SPARSE):
      for dim in range(ndim):
        efcid = base_efcid + dim
        if efcid < njmax_in:
          efc_J_rowadr_out[worldid, efcid] = rowadr + dim * rownnz
          efc_J_rownnz_out[worldid, efcid] = rownnz

  return kernel


# lanes per contact in _efc_contact_jac_sparse
_CONTACT_JAC_LANES = 8


@wp.func
def _dof_chain_advance(dof_parentid: wp.array[int], da1: int, da2: int, n: int) -> Tuple[int, int]:
  """Advance the merged DOF chains of two bodies by n nonzeros."""
  for _ in range(n):
    da = wp.max(da1, da2)
    # common dofs and the ends of the chains are not nonzeros
    if da < 0 or da1 == da2:
      break
    if da1 == da:
      da1 = dof_parentid[da1]
    if da2 == da:
      da2 = dof_parentid[da2]
  return da1, da2


@cache_kernel
def _efc_contact_jac_sparse(cone_type: types.ConeType, nmaxdim: int, max_records: int = 0, deterministic: bool = False):
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  vecdim = wp.types.vector(length=nmaxdim, dtype=float)
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_records

  # Lane k of a contact computes the k-th nonzero for all of the contact's rows, so lanes write
  # consecutive addresses; one thread per row would scatter every store across sectors.
  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    qvel_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    contact_efc_address_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    naconmax_in: int,
    nacon_in: wp.array[int],
    # In:
    condim_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    pos_in: wp.array[wp.vec3],
    frame_in: wp.array2d[wp.vec3],
    friction_in: wp.array2d[float],
    worldid_in: wp.array[int],
    ncontact_per_pass: int,
    # Data out:
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_Jqvel_out: wp.array2d[float],
  ):
    tid = wp.tid()
    lane = tid % _CONTACT_JAC_LANES

    for conid in range(tid / _CONTACT_JAC_LANES, wp.min(nacon_in[0], naconmax_in), ncontact_per_pass):
      efcid0 = contact_efc_address_in[conid, 0]
      if efcid0 < 0:
        continue

      worldid = worldid_in[conid]

      # all rows of a contact share the same nonzeros
      rownnz = efc_J_rownnz_in[worldid, efcid0]
      if lane >= rownnz:
        continue

      condim = condim_in[conid]

      geom = geom_in[conid]
      body1 = body_weldid[geom_bodyid[geom[0]]]
      body2 = body_weldid[geom_bodyid[geom[1]]]

      con_pos = pos_in[conid]
      if not wp.static(IS_ELLIPTIC):
        frame_0 = frame_in[conid, 0]

      da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
      da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)
      da1, da2 = _dof_chain_advance(dof_parentid, da1, da2, lane)

      # per-lane partial sums, so a lane issues one atomic per row rather than one per nonzero
      Jqvel = vecdim()
      for nnz in range(lane, rownnz, _CONTACT_JAC_LANES):
        dofid = wp.max(da1, da2)
        jac1p, jac1r = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          con_pos,
          body1,
          dofid,
          worldid,
        )
        jac2p, jac2r = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          con_pos,
          body2,
          dofid,
          worldid,
        )

        jacp_dif = jac2p - jac1p
        jacr_dif = jac2r - jac1r
        qvel = qvel_in[worldid, dofid]

        for dimid in range(nmaxdim):
          efcid = contact_efc_address_in[conid, dimid]
          if efcid < 0:
            continue

          if wp.static(IS_ELLIPTIC):
            J = float(0.0)
            if dimid < 3:
              frame_row = frame_in[conid, dimid]
              for xyz in range(3):
                J += frame_row[xyz] * jacp_dif[xyz]
            else:
              frame_row = frame_in[conid, dimid - 3]
              for xyz in range(3):
                J += frame_row[xyz] * jacr_dif[xyz]
          else:
            J = float(0.0)
            Ji = float(0.0)
            dimid2 = dimid / 2 + 1

            for xyz in range(3):
              J += frame_0[xyz] * jacp_dif[xyz]

              if condim > 1:
                if dimid2 < 3:
                  Ji += frame_in[conid, dimid2][xyz] * jacp_dif[xyz]
                else:
                  Ji += frame_in[conid, dimid2 - 3][xyz] * jacr_dif[xyz]

            if condim > 1:
              frii = friction_in[conid, dimid2 - 1]
              if dimid % 2 == 0:
                J += Ji * frii
              else:
                J -= Ji * frii

          sparseid = efc_J_rowadr_in[worldid, efcid] + nnz
          efc_J_colind_out[worldid, 0, sparseid] = dofid
          efc_J_out[worldid, 0, sparseid] = J
          Jqvel[dimid] += J * qvel

        if nnz + _CONTACT_JAC_LANES < rownnz:
          da1, da2 = _dof_chain_advance(dof_parentid, da1, da2, _CONTACT_JAC_LANES)

      for dimid in range(nmaxdim):
        efcid = contact_efc_address_in[conid, dimid]
        if efcid >= 0:
          # rows are zeroed by _efc_contact_init
          wp.atomic_add(efc_Jqvel_out, worldid, efcid, Jqvel[dimid])

  return kernel


@wp.func
def _rod_side_dof_jac_delta(
  # Model:
  body_rootid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  dofid: int,
  sgn: float,
  con_pos: wp.vec3,
  dof_twist: int,
  dp_twist: wp.vec3,
  dr_twist: wp.vec3,
  active_rod: bool,
  b0: int,
  b1: int,
  x0: wp.vec3,
  x1: wp.vec3,
  w0: float,
  w1: float,
  inv_le: float,
  t: wp.vec3,
  dvec: wp.vec3,
  t_cross_r: wp.vec3,
  g_psi: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  dp = wp.vec3(0.0)
  dr = wp.vec3(0.0)
  if dofid == dof_twist:
    dp += dp_twist
    dr += dr_twist
  if active_rod:
    cdof = cdof_in[worldid, dofid]
    if b0 >= 0 and body_isdofancestor[b0, dofid] != 0:
      dp0, dr0 = support.rod_contact_dof_delta(
        cdof,
        x0 - subtree_com_in[worldid, body_rootid[b0]],
        con_pos - x0,
        t,
        dvec,
        t_cross_r,
        g_psi,
        sgn * w0,
        -sgn,
        inv_le,
      )
      dp += dp0
      dr += dr0
    if b1 >= 0 and body_isdofancestor[b1, dofid] != 0:
      dp1, dr1 = support.rod_contact_dof_delta(
        cdof,
        x1 - subtree_com_in[worldid, body_rootid[b1]],
        con_pos - x1,
        t,
        dvec,
        t_cross_r,
        g_psi,
        sgn * w1,
        sgn,
        inv_le,
      )
      dp += dp1
      dr += dr1
  return dp, dr


@wp.func
def _tile_rod_twist_jac(tid_idx: int, local_twist: int, d_twist: wp.vec3) -> wp.vec3:
  if tid_idx == local_twist:
    return d_twist
  return wp.vec3(0.0)


@wp.func
def _tile_rod_jacp(jv: wp.vec3, cdof_ang: wp.vec3, M_p: wp.mat33, arm_s: wp.vec3) -> wp.vec3:
  return M_p @ jv - wp.cross(cdof_ang, arm_s)


@wp.func
def _tile_rod_jacr(
  # In:
  jv: wp.vec3,
  cdof_ang: wp.vec3,
  t_s: wp.vec3,
  M_psi: wp.mat33,
  sgn_wk: float,
) -> wp.vec3:
  return wp.cross(t_s, jv) + M_psi @ jv - sgn_wk * cdof_ang


@cache_kernel
def _efc_contact_jac_sparse_flex(cone_type: types.ConeType):
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  HAS_FLEX = True

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_vert0: wp.array[wp.vec3],
    body_isdofancestor: wp.array2d[int],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_jnt: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_vert_rodelem: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    plastic_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    contact_efc_address_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    nacon_in: wp.array[int],
    # In:
    dist_in: wp.array[float],
    condim_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    flex_in: wp.array[wp.vec2i],
    elem_in: wp.array[wp.vec2i],
    vert_in: wp.array[wp.vec2i],
    pos_in: wp.array[wp.vec3],
    frame_in: wp.array2d[wp.vec3],
    friction_in: wp.array2d[float],
    worldid_in: wp.array[int],
    # Data out:
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_Jqvel_out: wp.array2d[float],
  ):
    conid, dimid = wp.tid()

    if conid >= nacon_in[0]:
      return

    efcid = contact_efc_address_in[conid, dimid]
    if efcid < 0:
      return

    worldid = worldid_in[conid]
    condim = condim_in[conid]

    geom = geom_in[conid]
    flex = flex_in[conid]
    elem = elem_in[conid]
    vert = vert_in[conid]
    con_pos = pos_in[conid]

    is_interp = False
    if wp.static(HAS_FLEX):
      if geom[0] < 0 and flex[0] >= 0 and (vert[0] >= 0 or elem[0] >= 0):
        if flex_interp[flex[0]] != 0:
          is_interp = True
      if geom[1] < 0 and flex[1] >= 0 and (vert[1] >= 0 or elem[1] >= 0):
        if flex_interp[flex[1]] != 0:
          is_interp = True

    if not wp.static(IS_ELLIPTIC):
      frame_0 = frame_in[conid, 0]
      if condim > 1:
        dimid2 = dimid / 2 + 1
        frii = friction_in[conid, dimid2 - 1]

    rowadr = efc_J_rowadr_in[worldid, efcid]
    rownnz = efc_J_rownnz_in[worldid, efcid]

    if is_interp:
      # Interpolated flex path: accumulate unique bodies and signed weights
      local_bodies = support.vec_interp_body(-1)
      local_weights = support.vec_interp_weight(0.0)
      local_nb = int(0)

      for side in range(2):
        sign = float(-1.0) if side == 0 else float(1.0)

        if geom[side] >= 0:
          b = body_weldid[geom_bodyid[geom[side]]]
          local_nb, local_bodies, local_weights = support.add_interp_weight(
            local_nb, local_bodies, local_weights, b, sign
          )
        elif flex[side] >= 0:
          f = flex[side]
          if flex_interp[f] != 0:
            local_nb, local_bodies, local_weights = support.add_flex_interp_weights(
              body_weldid,
              flex_dim,
              flex_interp,
              flex_cellnum,
              flex_nodeadr,
              flex_vertadr,
              flex_elemdataadr,
              flex_nodebodyid,
              flex_elem,
              flex_vert0,
              flexvert_xpos_in,
              f,
              elem[side],
              vert[side],
              con_pos,
              worldid,
              sign,
              True,
              local_nb,
              local_bodies,
              local_weights,
            )
          else:
            # Non-interpolated flex path: use pre-computed bodies/weights
            body_ids, weights = support.contact_bodies_and_weights(
              jnt_bodyid,
              geom_bodyid,
              flex_dim,
              flex_interp,
              flex_cellnum,
              flex_nodeadr,
              flex_vertadr,
              flex_elemadr,
              flex_elemnum,
              flex_elemdataadr,
              flex_nodebodyid,
              flex_vertbodyid,
              flex_elem,
              flex_vert0,
              flex_rod_has_twist,
              flex_rod_edgeadr,
              flex_rod_edgenum,
              flex_rodedge_jnt,
              flex_elem_rodedge,
              flex_vert_rodelem,
              flexvert_xpos_in,
              conid,
              side,
              geom,
              flex,
              elem,
              vert,
              con_pos,
              worldid,
            )
            for vi in range(4):
              if body_ids[vi] >= 0:
                b = body_weldid[body_ids[vi]]
                local_nb, local_bodies, local_weights = support.add_interp_weight(
                  local_nb, local_bodies, local_weights, b, weights[vi] * sign
                )

      # Evaluate Jacobians for unique bodies and write to sparse J
      nnz = int(0)
      Jqvel = float(0.0)

      for i in range(local_nb):
        b = local_bodies[i]
        w_sign = local_weights[i]

        dof_start = body_dofadr[b]
        ndof = body_dofnum[b]
        for di in range(ndof):
          dofid = dof_start + di
          jacp, jacr = support.jac_dof(
            body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b, dofid, worldid
          )

          if wp.static(IS_ELLIPTIC):
            J = float(0.0)
            if dimid < 3:
              frame_row = frame_in[conid, dimid]
              for xyz in range(3):
                J += frame_row[xyz] * jacp[xyz] * w_sign
            else:
              frame_row = frame_in[conid, dimid - 3]
              for xyz in range(3):
                J += frame_row[xyz] * jacr[xyz] * w_sign
          else:
            J = float(0.0)
            Ji = float(0.0)
            for xyz in range(3):
              J += frame_0[xyz] * jacp[xyz] * w_sign
              if condim > 1:
                if dimid2 < 3:
                  Ji += frame_in[conid, dimid2][xyz] * jacp[xyz] * w_sign
                else:
                  Ji += frame_in[conid, dimid2 - 3][xyz] * jacr[xyz] * w_sign
            if condim > 1:
              if dimid % 2 == 0:
                J += Ji * frii
              else:
                J -= Ji * frii

          if nnz < rownnz:
            sparseid = rowadr + nnz
            efc_J_colind_out[worldid, 0, sparseid] = dofid
            efc_J_out[worldid, 0, sparseid] = J
            Jqvel += J * qvel_in[worldid, dofid]
            nnz += 1

      efc_Jqvel_out[worldid, efcid] = Jqvel

    else:
      # Non-interpolated flex path: use HEAD's 4-body lockstep tree-walk
      body_ids1, weights1 = support.contact_bodies_and_weights(
        jnt_bodyid,
        geom_bodyid,
        flex_dim,
        flex_interp,
        flex_cellnum,
        flex_nodeadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_nodebodyid,
        flex_vertbodyid,
        flex_elem,
        flex_vert0,
        flex_rod_has_twist,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        flex_vert_rodelem,
        flexvert_xpos_in,
        conid,
        0,
        geom,
        flex,
        elem,
        vert,
        con_pos,
        worldid,
      )
      body_ids2, weights2 = support.contact_bodies_and_weights(
        jnt_bodyid,
        geom_bodyid,
        flex_dim,
        flex_interp,
        flex_cellnum,
        flex_nodeadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_nodebodyid,
        flex_vertbodyid,
        flex_elem,
        flex_vert0,
        flex_rod_has_twist,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        flex_vert_rodelem,
        flexvert_xpos_in,
        conid,
        1,
        geom,
        flex,
        elem,
        vert,
        con_pos,
        worldid,
      )

      con_dist = dist_in[conid]
      con_normal = frame_in[conid, 0]
      (
        dof_tw0,
        dp_tw0,
        dr_tw0,
        active_rod0,
        rb0_0,
        rb1_0,
        rx0_0,
        rx1_0,
        rw0_0,
        rw1_0,
        rinv_le0,
        rt0,
        rdvec0,
        rt_cross_r0,
        rg_psi0,
      ) = support.rod_contact_side_params(
        jnt_dofadr,
        geom_type,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_plasticadr,
        flex_vertbodyid,
        flex_elem,
        flex_rod_has_twist,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        flex_vert_rodelem,
        plastic_in,
        flexvert_xpos_in,
        worldid,
        0,
        geom,
        flex,
        elem,
        vert,
        con_dist,
        con_pos,
        con_normal,
      )
      (
        dof_tw1,
        dp_tw1,
        dr_tw1,
        active_rod1,
        rb0_1,
        rb1_1,
        rx0_1,
        rx1_1,
        rw0_1,
        rw1_1,
        rinv_le1,
        rt1,
        rdvec1,
        rt_cross_r1,
        rg_psi1,
      ) = support.rod_contact_side_params(
        jnt_dofadr,
        geom_type,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_plasticadr,
        flex_vertbodyid,
        flex_elem,
        flex_rod_has_twist,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        flex_vert_rodelem,
        plastic_in,
        flexvert_xpos_in,
        worldid,
        1,
        geom,
        flex,
        elem,
        vert,
        con_dist,
        con_pos,
        con_normal,
      )

      # skip fixed bodies
      b1_0 = body_weldid[body_ids1[0]]
      b1_1 = body_weldid[body_ids1[1]] if body_ids1[1] >= 0 else -1
      b1_2 = body_weldid[body_ids1[2]] if body_ids1[2] >= 0 else -1
      b1_3 = body_weldid[body_ids1[3]] if body_ids1[3] >= 0 else -1

      b2_0 = body_weldid[body_ids2[0]]
      b2_1 = body_weldid[body_ids2[1]] if body_ids2[1] >= 0 else -1
      b2_2 = body_weldid[body_ids2[2]] if body_ids2[2] >= 0 else -1
      b2_3 = body_weldid[body_ids2[3]] if body_ids2[3] >= 0 else -1

      dof1_0 = int(body_dofadr[b1_0] + body_dofnum[b1_0] - 1) if b1_0 >= 0 else -1
      dof1_1 = int(body_dofadr[b1_1] + body_dofnum[b1_1] - 1) if b1_1 >= 0 else -1
      dof1_2 = int(body_dofadr[b1_2] + body_dofnum[b1_2] - 1) if b1_2 >= 0 else -1
      dof1_3 = int(body_dofadr[b1_3] + body_dofnum[b1_3] - 1) if b1_3 >= 0 else -1
      da1 = wp.max(dof1_0, wp.max(dof1_1, wp.max(dof1_2, dof1_3)))

      dof2_0 = int(body_dofadr[b2_0] + body_dofnum[b2_0] - 1) if b2_0 >= 0 else -1
      dof2_1 = int(body_dofadr[b2_1] + body_dofnum[b2_1] - 1) if b2_1 >= 0 else -1
      dof2_2 = int(body_dofadr[b2_2] + body_dofnum[b2_2] - 1) if b2_2 >= 0 else -1
      dof2_3 = int(body_dofadr[b2_3] + body_dofnum[b2_3] - 1) if b2_3 >= 0 else -1
      da2 = wp.max(dof2_0, wp.max(dof2_1, wp.max(dof2_2, dof2_3)))

      da = wp.max(da1, da2)

      Jqvel = float(0.0)
      nnz = int(0)
      dofid = int(da)

      while True:
        if nnz >= rownnz:
          break

        if dofid == da:
          jac1p = wp.vec3(0.0)
          jac1r = wp.vec3(0.0)
          if dof1_0 == da and weights1[0] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b1_0, dofid, worldid
            )
            jac1p += jp * weights1[0]
            jac1r += jr * weights1[0]
          if dof1_1 == da and weights1[1] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b1_1, dofid, worldid
            )
            jac1p += jp * weights1[1]
            jac1r += jr * weights1[1]
          if dof1_2 == da and weights1[2] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b1_2, dofid, worldid
            )
            jac1p += jp * weights1[2]
            jac1r += jr * weights1[2]
          if dof1_3 == da and weights1[3] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b1_3, dofid, worldid
            )
            jac1p += jp * weights1[3]
            jac1r += jr * weights1[3]

          jac2p = wp.vec3(0.0)
          jac2r = wp.vec3(0.0)
          if dof2_0 == da and weights2[0] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b2_0, dofid, worldid
            )
            jac2p += jp * weights2[0]
            jac2r += jr * weights2[0]
          if dof2_1 == da and weights2[1] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b2_1, dofid, worldid
            )
            jac2p += jp * weights2[1]
            jac2r += jr * weights2[1]
          if dof2_2 == da and weights2[2] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b2_2, dofid, worldid
            )
            jac2p += jp * weights2[2]
            jac2r += jr * weights2[2]
          if dof2_3 == da and weights2[3] != 0.0:
            jp, jr = support.jac_dof(
              body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, con_pos, b2_3, dofid, worldid
            )
            jac2p += jp * weights2[3]
            jac2r += jr * weights2[3]

          dp_r0, dr_r0 = _rod_side_dof_jac_delta(
            body_rootid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            worldid,
            dofid,
            -1.0,
            con_pos,
            dof_tw0,
            dp_tw0,
            dr_tw0,
            active_rod0,
            rb0_0,
            rb1_0,
            rx0_0,
            rx1_0,
            rw0_0,
            rw1_0,
            rinv_le0,
            rt0,
            rdvec0,
            rt_cross_r0,
            rg_psi0,
          )
          dp_r1, dr_r1 = _rod_side_dof_jac_delta(
            body_rootid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            worldid,
            dofid,
            1.0,
            con_pos,
            dof_tw1,
            dp_tw1,
            dr_tw1,
            active_rod1,
            rb0_1,
            rb1_1,
            rx0_1,
            rx1_1,
            rw0_1,
            rw1_1,
            rinv_le1,
            rt1,
            rdvec1,
            rt_cross_r1,
            rg_psi1,
          )

          jacp_dif = jac2p - jac1p + dp_r0 + dp_r1
          jacr_dif = jac2r - jac1r + dr_r0 + dr_r1

          if wp.static(IS_ELLIPTIC):
            J = float(0.0)
            if dimid < 3:
              frame_row = frame_in[conid, dimid]
              for xyz in range(3):
                J += frame_row[xyz] * jacp_dif[xyz]
            else:
              frame_row = frame_in[conid, dimid - 3]
              for xyz in range(3):
                J += frame_row[xyz] * jacr_dif[xyz]
          else:
            J = float(0.0)
            Ji = float(0.0)

            for xyz in range(3):
              J += frame_0[xyz] * jacp_dif[xyz]

              if condim > 1:
                if dimid2 < 3:
                  Ji += frame_in[conid, dimid2][xyz] * jacp_dif[xyz]
                else:
                  Ji += frame_in[conid, dimid2 - 3][xyz] * jacr_dif[xyz]

            if condim > 1:
              if dimid % 2 == 0:
                J += Ji * frii
              else:
                J -= Ji * frii

          sparseid = rowadr + nnz
          efc_J_colind_out[worldid, 0, sparseid] = dofid
          efc_J_out[worldid, 0, sparseid] = J
          nnz += 1
          Jqvel += J * qvel_in[worldid, dofid]

          # Advance tree pointers
          if dof1_0 == da:
            dof1_0 = dof_parentid[dof1_0]
          if dof1_1 == da:
            dof1_1 = dof_parentid[dof1_1]
          if dof1_2 == da:
            dof1_2 = dof_parentid[dof1_2]
          if dof1_3 == da:
            dof1_3 = dof_parentid[dof1_3]
          da1 = wp.max(dof1_0, wp.max(dof1_1, wp.max(dof1_2, dof1_3)))

          if dof2_0 == da:
            dof2_0 = dof_parentid[dof2_0]
          if dof2_1 == da:
            dof2_1 = dof_parentid[dof2_1]
          if dof2_2 == da:
            dof2_2 = dof_parentid[dof2_2]
          if dof2_3 == da:
            dof2_3 = dof_parentid[dof2_3]
          da2 = wp.max(dof2_0, wp.max(dof2_1, wp.max(dof2_2, dof2_3)))

          da = wp.max(da1, da2)
          dofid = da

      efc_Jqvel_out[worldid, efcid] = Jqvel

  return kernel


@cache_kernel
def _efc_contact_jac_dense(tile_size: int, cone_type: types.ConeType, njmax: int, deterministic: bool = False):
  TILE_SIZE = tile_size
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = njmax

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    geom_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nl_in: wp.array[int],
    nefc_in: wp.array[int],
    qvel_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    contact_efc_address_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    njmax_in: int,
    # In:
    nv_padded: int,
    condim_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    pos_in: wp.array[wp.vec3],
    frame_in: wp.array2d[wp.vec3],
    friction_in: wp.array2d[float],
    # Data out:
    efc_J_out: wp.array3d[float],
    efc_Jqvel_out: wp.array2d[float],
  ):
    worldid, dof_block_id, tid = wp.tid()

    dof_start = dof_block_id * wp.static(TILE_SIZE)
    if dof_start >= nv_padded:
      return

    cdof_tile = wp.tile_load(cdof_in[worldid], shape=TILE_SIZE, offset=dof_start, bounds_check=True)
    qvel_tile = wp.tile_load(qvel_in[worldid], shape=TILE_SIZE, offset=dof_start, bounds_check=True)

    efcid_start = ne_in[worldid] + nf_in[worldid] + nl_in[worldid]
    efcid_end = wp.min(nefc_in[worldid], njmax_in)

    prev_conid = int(-1)
    condim = int(0)

    for efcid in range(efcid_start, efcid_end):
      conid = efc_id_in[worldid, efcid]

      # Recompute per-contact data only when contact changes
      if conid != prev_conid:
        prev_conid = conid
        condim = condim_in[conid]
        first_efcid = contact_efc_address_in[conid, 0]

        geom = geom_in[conid]
        body1 = geom_bodyid[geom[0]]
        body2 = geom_bodyid[geom[1]]

        con_pos = pos_in[conid]
        offset1 = con_pos - subtree_com_in[worldid, body_rootid[body1]]
        affects1_tile = wp.tile_load(body_isdofancestor[body1], shape=TILE_SIZE, offset=dof_start, bounds_check=True)
        jacp1_tile = wp.tile_map(support._compute_jacp, cdof_tile, offset1, affects1_tile)
        jacr1_tile = wp.tile_map(support._compute_jacr, cdof_tile, affects1_tile)

        offset2 = con_pos - subtree_com_in[worldid, body_rootid[body2]]
        affects2_tile = wp.tile_load(body_isdofancestor[body2], shape=TILE_SIZE, offset=dof_start, bounds_check=True)
        jacp2_tile = wp.tile_map(support._compute_jacp, cdof_tile, offset2, affects2_tile)
        jacr2_tile = wp.tile_map(support._compute_jacr, cdof_tile, affects2_tile)

        jacp_dif_tile = wp.tile_map(wp.sub, jacp2_tile, jacp1_tile)
        jacr_dif_tile = wp.tile_map(wp.sub, jacr2_tile, jacr1_tile)

        if not wp.static(IS_ELLIPTIC):
          frame_0 = frame_in[conid, 0]
          Ji_0p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_0)

          if condim > 1:
            Ji_0r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_0)
            frame_1 = frame_in[conid, 1]
            Ji_1p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_1)
            Ji_1r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_1)
            frame_2 = frame_in[conid, 2]
            Ji_2p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_2)
            Ji_2r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_2)

      if wp.static(IS_ELLIPTIC):
        dimid = efcid - first_efcid
        if dimid < 3:
          frame_idx = dimid
        else:
          frame_idx = dimid - 3

        frame_row = frame_in[conid, frame_idx]

        if dimid < 3:
          J_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_row)
        else:
          J_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_row)
      else:
        J_tile = Ji_0p_tile
        if condim > 1:
          dimid = efcid - first_efcid
          dimid2 = dimid / 2 + 1
          frii = friction_in[conid, dimid2 - 1]
          frii_sign = frii * (1.0 - 2.0 * float(dimid & 1))

          if dimid2 == 1:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_1p_tile, frii_sign))
          elif dimid2 == 2:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_2p_tile, frii_sign))
          elif dimid2 == 3:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_0r_tile, frii_sign))
          elif dimid2 == 4:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_1r_tile, frii_sign))
          else:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_2r_tile, frii_sign))

      wp.tile_store(efc_J_out[worldid, efcid], J_tile, offset=dof_start, bounds_check=True)

      Jqvel_tile = wp.tile_map(wp.mul, J_tile, qvel_tile)
      Jqvel_sum = wp.tile_reduce(wp.add, Jqvel_tile)
      # Tile atomics bypass Warp's deterministic accumulation.
      if wp.static(deterministic):
        if tid == 0:
          wp.atomic_add(efc_Jqvel_out[worldid], efcid, wp.tile_extract(Jqvel_sum, 0))
      else:
        wp.tile_atomic_add(efc_Jqvel_out[worldid], Jqvel_sum, offset=efcid)

  return kernel


@cache_kernel
def _efc_contact_jac_dense_flex(tile_size: int, cone_type: types.ConeType, njmax: int, deterministic: bool = False):
  TILE_SIZE = tile_size
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = njmax

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_vert0: wp.array[wp.vec3],
    body_isdofancestor: wp.array2d[int],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_jnt: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_vert_rodelem: wp.array[int],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nl_in: wp.array[int],
    nefc_in: wp.array[int],
    qvel_in: wp.array2d[float],
    plastic_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    contact_efc_address_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    njmax_in: int,
    # In:
    nv_padded: int,
    dist_in: wp.array[float],
    condim_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    flex_in: wp.array[wp.vec2i],
    elem_in: wp.array[wp.vec2i],
    vert_in: wp.array[wp.vec2i],
    pos_in: wp.array[wp.vec3],
    frame_in: wp.array2d[wp.vec3],
    friction_in: wp.array2d[float],
    # Data out:
    efc_J_out: wp.array3d[float],
    efc_Jqvel_out: wp.array2d[float],
  ):
    worldid, dof_block_id, tid = wp.tid()

    dof_start = dof_block_id * wp.static(TILE_SIZE)
    if dof_start >= nv_padded:
      return

    cdof_tile = wp.tile_load(cdof_in[worldid], shape=TILE_SIZE, offset=dof_start, bounds_check=True)
    qvel_tile = wp.tile_load(qvel_in[worldid], shape=TILE_SIZE, offset=dof_start, bounds_check=True)
    tid_tile = wp.tile_arange(TILE_SIZE, dtype=int)

    efcid_start = ne_in[worldid] + nf_in[worldid] + nl_in[worldid]
    efcid_end = wp.min(nefc_in[worldid], njmax_in)

    prev_conid = int(-1)
    condim = int(0)

    for efcid in range(efcid_start, efcid_end):
      conid = efc_id_in[worldid, efcid]

      # Recompute per-contact data only when contact changes
      if conid != prev_conid:
        prev_conid = conid
        condim = condim_in[conid]
        first_efcid = contact_efc_address_in[conid, 0]

        geom = geom_in[conid]
        flex = flex_in[conid]
        elem = elem_in[conid]
        vert = vert_in[conid]
        con_pos = pos_in[conid]

        body_ids1, weights1 = support.contact_bodies_and_weights(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          conid,
          0,
          geom,
          flex,
          elem,
          vert,
          con_pos,
          worldid,
        )
        body_ids2, weights2 = support.contact_bodies_and_weights(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          conid,
          1,
          geom,
          flex,
          elem,
          vert,
          con_pos,
          worldid,
        )

        b1_0 = body_ids1[0]
        b1_1 = body_ids1[1]
        b1_2 = body_ids1[2]
        b1_3 = body_ids1[3]

        b2_0 = body_ids2[0]
        b2_1 = body_ids2[1]
        b2_2 = body_ids2[2]
        b2_3 = body_ids2[3]

        # Weighted jacp for side 1
        jacp1_tile = wp.tile_map(
          support._compute_jacp,
          cdof_tile,
          con_pos - subtree_com_in[worldid, body_rootid[b1_0]],
          wp.tile_load(body_isdofancestor[b1_0], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
        )
        jacp1_tile = wp.tile_map(wp.mul, jacp1_tile, weights1[0])
        if b1_1 >= 0 and weights1[1] != 0.0:
          t_jacp = wp.tile_map(
            support._compute_jacp,
            cdof_tile,
            con_pos - subtree_com_in[worldid, body_rootid[b1_1]],
            wp.tile_load(body_isdofancestor[b1_1], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacp1_tile = wp.tile_map(wp.add, jacp1_tile, wp.tile_map(wp.mul, t_jacp, weights1[1]))
        if b1_2 >= 0 and weights1[2] != 0.0:
          t_jacp = wp.tile_map(
            support._compute_jacp,
            cdof_tile,
            con_pos - subtree_com_in[worldid, body_rootid[b1_2]],
            wp.tile_load(body_isdofancestor[b1_2], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacp1_tile = wp.tile_map(wp.add, jacp1_tile, wp.tile_map(wp.mul, t_jacp, weights1[2]))
        if b1_3 >= 0 and weights1[3] != 0.0:
          t_jacp = wp.tile_map(
            support._compute_jacp,
            cdof_tile,
            con_pos - subtree_com_in[worldid, body_rootid[b1_3]],
            wp.tile_load(body_isdofancestor[b1_3], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacp1_tile = wp.tile_map(wp.add, jacp1_tile, wp.tile_map(wp.mul, t_jacp, weights1[3]))

        # Weighted jacp for side 2
        jacp2_tile = wp.tile_map(
          support._compute_jacp,
          cdof_tile,
          con_pos - subtree_com_in[worldid, body_rootid[b2_0]],
          wp.tile_load(body_isdofancestor[b2_0], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
        )
        jacp2_tile = wp.tile_map(wp.mul, jacp2_tile, weights2[0])
        if b2_1 >= 0 and weights2[1] != 0.0:
          t_jacp = wp.tile_map(
            support._compute_jacp,
            cdof_tile,
            con_pos - subtree_com_in[worldid, body_rootid[b2_1]],
            wp.tile_load(body_isdofancestor[b2_1], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacp2_tile = wp.tile_map(wp.add, jacp2_tile, wp.tile_map(wp.mul, t_jacp, weights2[1]))
        if b2_2 >= 0 and weights2[2] != 0.0:
          t_jacp = wp.tile_map(
            support._compute_jacp,
            cdof_tile,
            con_pos - subtree_com_in[worldid, body_rootid[b2_2]],
            wp.tile_load(body_isdofancestor[b2_2], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacp2_tile = wp.tile_map(wp.add, jacp2_tile, wp.tile_map(wp.mul, t_jacp, weights2[2]))
        if b2_3 >= 0 and weights2[3] != 0.0:
          t_jacp = wp.tile_map(
            support._compute_jacp,
            cdof_tile,
            con_pos - subtree_com_in[worldid, body_rootid[b2_3]],
            wp.tile_load(body_isdofancestor[b2_3], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacp2_tile = wp.tile_map(wp.add, jacp2_tile, wp.tile_map(wp.mul, t_jacp, weights2[3]))

        jacp_dif_tile = wp.tile_map(wp.sub, jacp2_tile, jacp1_tile)

        # Weighted jacr for side 1
        jacr1_tile = wp.tile_map(
          support._compute_jacr,
          cdof_tile,
          wp.tile_load(body_isdofancestor[b1_0], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
        )
        jacr1_tile = wp.tile_map(wp.mul, jacr1_tile, weights1[0])
        if b1_1 >= 0 and weights1[1] != 0.0:
          t_jacr = wp.tile_map(
            support._compute_jacr,
            cdof_tile,
            wp.tile_load(body_isdofancestor[b1_1], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacr1_tile = wp.tile_map(wp.add, jacr1_tile, wp.tile_map(wp.mul, t_jacr, weights1[1]))
        if b1_2 >= 0 and weights1[2] != 0.0:
          t_jacr = wp.tile_map(
            support._compute_jacr,
            cdof_tile,
            wp.tile_load(body_isdofancestor[b1_2], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacr1_tile = wp.tile_map(wp.add, jacr1_tile, wp.tile_map(wp.mul, t_jacr, weights1[2]))
        if b1_3 >= 0 and weights1[3] != 0.0:
          t_jacr = wp.tile_map(
            support._compute_jacr,
            cdof_tile,
            wp.tile_load(body_isdofancestor[b1_3], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacr1_tile = wp.tile_map(wp.add, jacr1_tile, wp.tile_map(wp.mul, t_jacr, weights1[3]))

        # Weighted jacr for side 2
        jacr2_tile = wp.tile_map(
          support._compute_jacr,
          cdof_tile,
          wp.tile_load(body_isdofancestor[b2_0], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
        )
        jacr2_tile = wp.tile_map(wp.mul, jacr2_tile, weights2[0])
        if b2_1 >= 0 and weights2[1] != 0.0:
          t_jacr = wp.tile_map(
            support._compute_jacr,
            cdof_tile,
            wp.tile_load(body_isdofancestor[b2_1], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacr2_tile = wp.tile_map(wp.add, jacr2_tile, wp.tile_map(wp.mul, t_jacr, weights2[1]))
        if b2_2 >= 0 and weights2[2] != 0.0:
          t_jacr = wp.tile_map(
            support._compute_jacr,
            cdof_tile,
            wp.tile_load(body_isdofancestor[b2_2], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacr2_tile = wp.tile_map(wp.add, jacr2_tile, wp.tile_map(wp.mul, t_jacr, weights2[2]))
        if b2_3 >= 0 and weights2[3] != 0.0:
          t_jacr = wp.tile_map(
            support._compute_jacr,
            cdof_tile,
            wp.tile_load(body_isdofancestor[b2_3], shape=TILE_SIZE, offset=dof_start, bounds_check=True),
          )
          jacr2_tile = wp.tile_map(wp.add, jacr2_tile, wp.tile_map(wp.mul, t_jacr, weights2[3]))

        jacr_dif_tile = wp.tile_map(wp.sub, jacr2_tile, jacr1_tile)

        con_dist = dist_in[conid]
        con_normal = frame_in[conid, 0]
        for side_rod in range(2):
          (
            dof_tw,
            dp_tw,
            dr_tw,
            active_rod,
            rb0,
            rb1,
            rx0,
            rx1,
            rw0,
            rw1,
            rinv_le,
            rt,
            rdvec,
            rt_cross_r,
            rg_psi,
          ) = support.rod_contact_side_params(
            jnt_dofadr,
            geom_type,
            flex_vertadr,
            flex_elemadr,
            flex_elemnum,
            flex_elemdataadr,
            flex_plasticadr,
            flex_vertbodyid,
            flex_elem,
            flex_rod_has_twist,
            flex_rod_edgeadr,
            flex_rod_edgenum,
            flex_rodedge_jnt,
            flex_elem_rodedge,
            flex_vert_rodelem,
            plastic_in,
            flexvert_xpos_in,
            worldid,
            side_rod,
            geom,
            flex,
            elem,
            vert,
            con_dist,
            con_pos,
            con_normal,
          )
          if dof_tw >= dof_start and dof_tw < dof_start + wp.static(TILE_SIZE):
            local_tw = dof_tw - dof_start
            jacp_dif_tile = wp.tile_map(
              wp.add,
              jacp_dif_tile,
              wp.tile_map(_tile_rod_twist_jac, tid_tile, local_tw, dp_tw),
            )
            jacr_dif_tile = wp.tile_map(
              wp.add,
              jacr_dif_tile,
              wp.tile_map(_tile_rod_twist_jac, tid_tile, local_tw, dr_tw),
            )
          if active_rod:
            rsgn = 1.0 if side_rod == 1 else -1.0
            for k_end in range(2):
              rbk = rb0 if k_end == 0 else rb1
              rxk = rx0 if k_end == 0 else rx1
              rwk = rw0 if k_end == 0 else rw1
              resgn = -1.0 if k_end == 0 else 1.0
              if rbk >= 0:
                aff_tile = wp.tile_load(
                  body_isdofancestor[rbk],
                  shape=TILE_SIZE,
                  offset=dof_start,
                  bounds_check=True,
                )
                jv_tile = wp.tile_map(
                  support._compute_jacp,
                  cdof_tile,
                  rxk - subtree_com_in[worldid, body_rootid[rbk]],
                  aff_tile,
                )
                ang_tile = wp.tile_map(support._compute_jacr, cdof_tile, aff_tile)
                sgn_esgn = rsgn * resgn
                sgn_wk = rsgn * rwk
                ra = sgn_esgn * rinv_le * wp.dot(rdvec, rt)
                M_p = (
                  wp.outer(rt, (-sgn_esgn * rinv_le) * rdvec) + wp.diag(wp.vec3(ra)) + wp.outer(rt_cross_r, sgn_esgn * rg_psi)
                )
                arm_s = sgn_wk * (con_pos - rxk)
                jacp_dif_tile = wp.tile_map(
                  wp.add,
                  jacp_dif_tile,
                  wp.tile_map(_tile_rod_jacp, jv_tile, ang_tile, M_p, arm_s),
                )
                t_s = (sgn_esgn * rinv_le) * rt
                M_psi = wp.outer(rt, sgn_esgn * rg_psi)
                jacr_dif_tile = wp.tile_map(
                  wp.add,
                  jacr_dif_tile,
                  wp.tile_map(
                    _tile_rod_jacr,
                    jv_tile,
                    ang_tile,
                    t_s,
                    M_psi,
                    sgn_wk,
                  ),
                )

        if not wp.static(IS_ELLIPTIC):
          frame_0 = frame_in[conid, 0]
          Ji_0p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_0)

          if condim > 1:
            Ji_0r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_0)
            frame_1 = frame_in[conid, 1]
            Ji_1p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_1)
            Ji_1r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_1)
            frame_2 = frame_in[conid, 2]
            Ji_2p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_2)
            Ji_2r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_2)

      if wp.static(IS_ELLIPTIC):
        dimid = efcid - first_efcid
        if dimid < 3:
          frame_idx = dimid
        else:
          frame_idx = dimid - 3

        frame_row = frame_in[conid, frame_idx]

        if dimid < 3:
          J_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_row)
        else:
          J_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_row)
      else:
        J_tile = Ji_0p_tile
        if condim > 1:
          dimid = efcid - first_efcid
          dimid2 = dimid / 2 + 1
          frii = friction_in[conid, dimid2 - 1]
          frii_sign = frii * (1.0 - 2.0 * float(dimid & 1))

          if dimid2 == 1:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_1p_tile, frii_sign))
          elif dimid2 == 2:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_2p_tile, frii_sign))
          elif dimid2 == 3:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_0r_tile, frii_sign))
          elif dimid2 == 4:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_1r_tile, frii_sign))
          else:
            J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_2r_tile, frii_sign))

      wp.tile_store(efc_J_out[worldid, efcid], J_tile, offset=dof_start, bounds_check=True)

      Jqvel_tile = wp.tile_map(wp.mul, J_tile, qvel_tile)
      Jqvel_sum = wp.tile_reduce(wp.add, Jqvel_tile)
      # Tile atomics bypass Warp's deterministic accumulation.
      if wp.static(deterministic):
        if tid == 0:
          wp.atomic_add(efc_Jqvel_out[worldid], efcid, wp.tile_extract(Jqvel_sum, 0))
      else:
        wp.tile_atomic_add(efc_Jqvel_out[worldid], Jqvel_sum, offset=efcid)

  return kernel


@cache_kernel
def _efc_contact_update(cone_type: types.ConeType, flg_adhesion: bool, is_discrete: bool = False):
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    body_invweight0: wp.array2d[wp.vec2],
    geom_bodyid: wp.array[int],
    # Data in:
    contact_efc_address_in: wp.array2d[int],
    efc_Jqvel_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    dist_in: wp.array[float],
    condim_in: wp.array[int],
    includemargin_in: wp.array[float],
    worldid_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    friction_in: wp.array[vec5],
    solref_in: wp.array[wp.vec2],
    solreffriction_in: wp.array[wp.vec2],
    solimp_in: wp.array[vec5],
    adhesion_in: wp.array[float],
    type_in: wp.array[int],
    # Data out:
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
  ):
    conid, dimid = wp.tid()

    if conid >= nacon_in[0]:
      return

    if not (type_in[conid] & ContactType.CONSTRAINT):
      return

    condim = condim_in[conid]

    if wp.static(IS_ELLIPTIC):
      if dimid > condim - 1:
        return
    else:
      if condim == 1 and dimid > 0:
        return
      elif condim > 1 and dimid >= 2 * (condim - 1):
        return

    efcid = contact_efc_address_in[conid, dimid]
    if efcid < 0:
      return

    worldid = worldid_in[conid]
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    includemargin = includemargin_in[conid]
    pos = dist_in[conid] - includemargin

    geom = geom_in[conid]
    Jqvel = efc_Jqvel_in[worldid, efcid]

    body1 = geom_bodyid[geom[0]]
    body2 = geom_bodyid[geom[1]]

    body_invweight0_id = worldid % body_invweight0.shape[0]
    invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]

    invweight_scale = 1.0
    ref = solref_in[conid]
    pos_aref = pos

    if wp.static(IS_ELLIPTIC):
      if dimid > 0:
        solreffriction = solreffriction_in[conid]

        # non-normal directions use solreffriction (if non-zero)
        if solreffriction[0] or solreffriction[1]:
          ref = solreffriction

        invweight_scale = impratio_invsqrt * impratio_invsqrt
        friction = friction_in[conid]

        if dimid > 1:
          fri0 = friction[0]
          frii = friction[dimid - 1]
          fri = fri0 * fri0 / (frii * frii)
          invweight_scale *= fri

        pos_aref = 0.0
    else:
      if condim > 1:
        friction = friction_in[conid]
        fri0 = friction[0]
        invweight = invweight + fri0 * fri0 * invweight
        invweight = invweight * 2.0 * fri0 * fri0 * impratio_invsqrt * impratio_invsqrt

    if condim == 1:
      efc_type = ConstraintType.CONTACT_FRICTIONLESS
    elif wp.static(IS_ELLIPTIC):
      efc_type = ConstraintType.CONTACT_ELLIPTIC
    else:
      efc_type = ConstraintType.CONTACT_PYRAMIDAL

    _efc_row(
      opt_disableflags,
      worldid,
      timestep,
      efcid,
      pos_aref,
      pos,
      invweight,
      solref_in[conid],
      solimp_in[conid],
      includemargin,
      Jqvel,
      0.0,
      efc_type,
      conid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

    if wp.static(IS_ELLIPTIC):
      if dimid > 0:
        # Preserve elliptic friction ratios after _efc_row floors normal regularization.
        efc_D_out[worldid, efcid] /= invweight_scale

        b_fri = _contact_kbimp(opt_disableflags, timestep, ref, solimp_in[conid], pos, is_discrete)[1]
        f_fri = 1.0 + timestep * b_fri if is_discrete else 1.0
        efc_aref_out[worldid, efcid] = -b_fri * Jqvel / f_fri

    if wp.static(flg_adhesion):
      if adhesion_in[conid] != 0.0 and (dimid == 0 or not wp.static(IS_ELLIPTIC)):
        efc_D = efc_D_out[worldid, efcid]
        if efc_D > 0.0:
          adhesion = adhesion_in[conid]
          if not wp.static(IS_ELLIPTIC) and condim > 1:
            adhesion = adhesion / float(2 * (condim - 1))
          efc_aref_out[worldid, efcid] += (1.0 / efc_D) * adhesion

  return kernel


@cache_kernel
def _efc_contact_update_flex(cone_type: types.ConeType, flg_adhesion: bool = False, is_discrete: bool = False):
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    body_invweight0: wp.array2d[wp.vec2],
    jnt_bodyid: wp.array[int],
    geom_bodyid: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_vert0: wp.array[wp.vec3],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_jnt: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_vert_rodelem: wp.array[int],
    # Data in:
    flexvert_xpos_in: wp.array2d[wp.vec3],
    contact_efc_address_in: wp.array2d[int],
    efc_Jqvel_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    dist_in: wp.array[float],
    pos_in: wp.array[wp.vec3],
    condim_in: wp.array[int],
    includemargin_in: wp.array[float],
    worldid_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    flex_in: wp.array[wp.vec2i],
    elem_in: wp.array[wp.vec2i],
    vert_in: wp.array[wp.vec2i],
    friction_in: wp.array[vec5],
    solref_in: wp.array[wp.vec2],
    solreffriction_in: wp.array[wp.vec2],
    solimp_in: wp.array[vec5],
    adhesion_in: wp.array[float],
    type_in: wp.array[int],
    # Data out:
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
  ):
    conid, dimid = wp.tid()

    if conid >= nacon_in[0]:
      return

    if not (type_in[conid] & ContactType.CONSTRAINT):
      return

    condim = condim_in[conid]

    if wp.static(IS_ELLIPTIC):
      if dimid > condim - 1:
        return
    else:
      if condim == 1 and dimid > 0:
        return
      elif condim > 1 and dimid >= 2 * (condim - 1):
        return

    efcid = contact_efc_address_in[conid, dimid]
    if efcid < 0:
      return

    worldid = worldid_in[conid]
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    includemargin = includemargin_in[conid]
    pos = dist_in[conid] - includemargin

    geom = geom_in[conid]
    flex = flex_in[conid]
    elem = elem_in[conid]
    vert = vert_in[conid]
    con_pos = pos_in[conid]
    Jqvel = efc_Jqvel_in[worldid, efcid]

    body_invweight0_id = worldid % body_invweight0.shape[0]
    invweight1 = float(0.0)

    if geom[0] >= 0:
      b1 = geom_bodyid[geom[0]]
      invweight1 = body_invweight0[body_invweight0_id, b1][0]
    elif flex[0] >= 0:
      f1 = flex[0]
      if flex_interp[f1] != 0:
        # Interpolated path: de-duplicate nodes in the cell
        local_bodies = support.vec_interp_body(-1)
        local_weights = support.vec_interp_weight(0.0)
        local_nb = int(0)
        local_nb, local_bodies, local_weights = support.add_flex_interp_weights(
          flex_nodebodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_elem,
          flex_vert0,
          flexvert_xpos_in,
          f1,
          elem[0],
          vert[0],
          con_pos,
          worldid,
          1.0,
          False,
          local_nb,
          local_bodies,
          local_weights,
        )

        # Accumulate linear weights for unique bodies
        for i in range(local_nb):
          b = local_bodies[i]
          w = local_weights[i]
          invweight1 += body_invweight0[body_invweight0_id, b][0] * w
      else:
        body_ids, weights = support.contact_bodies_and_weights(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          conid,
          0,
          geom,
          flex,
          elem,
          vert,
          con_pos,
          worldid,
        )
        b0 = body_ids[0]
        b1 = body_ids[1]
        b2 = body_ids[2]
        b3 = body_ids[3]
        invweight1 = weights[0] * body_invweight0[body_invweight0_id, b0][0]
        if b1 >= 0:
          invweight1 += weights[1] * body_invweight0[body_invweight0_id, b1][0]
        if b2 >= 0:
          invweight1 += weights[2] * body_invweight0[body_invweight0_id, b2][0]
        if b3 >= 0:
          invweight1 += weights[3] * body_invweight0[body_invweight0_id, b3][0]

    invweight2 = float(0.0)

    if geom[1] >= 0:
      b2 = geom_bodyid[geom[1]]
      invweight2 = body_invweight0[body_invweight0_id, b2][0]
    elif flex[1] >= 0:
      f2 = flex[1]
      if flex_interp[f2] != 0:
        # Interpolated path: de-duplicate nodes in the cell
        local_bodies = support.vec_interp_body(-1)
        local_weights = support.vec_interp_weight(0.0)
        local_nb = int(0)
        local_nb, local_bodies, local_weights = support.add_flex_interp_weights(
          flex_nodebodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_elem,
          flex_vert0,
          flexvert_xpos_in,
          f2,
          elem[1],
          vert[1],
          con_pos,
          worldid,
          1.0,
          False,
          local_nb,
          local_bodies,
          local_weights,
        )

        # Accumulate linear weights for unique bodies
        for i in range(local_nb):
          b = local_bodies[i]
          w = local_weights[i]
          invweight2 += body_invweight0[body_invweight0_id, b][0] * w
      else:
        body_ids, weights = support.contact_bodies_and_weights(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          conid,
          1,
          geom,
          flex,
          elem,
          vert,
          con_pos,
          worldid,
        )
        b0 = body_ids[0]
        b1 = body_ids[1]
        b2 = body_ids[2]
        b3 = body_ids[3]
        invweight2 = weights[0] * body_invweight0[body_invweight0_id, b0][0]
        if b1 >= 0:
          invweight2 += weights[1] * body_invweight0[body_invweight0_id, b1][0]
        if b2 >= 0:
          invweight2 += weights[2] * body_invweight0[body_invweight0_id, b2][0]
        if b3 >= 0:
          invweight2 += weights[3] * body_invweight0[body_invweight0_id, b3][0]

    invweight = invweight1 + invweight2

    invweight_scale = 1.0
    ref = solref_in[conid]
    pos_aref = pos

    if wp.static(IS_ELLIPTIC):
      if dimid > 0:
        solreffriction = solreffriction_in[conid]

        # non-normal directions use solreffriction (if non-zero)
        if solreffriction[0] or solreffriction[1]:
          ref = solreffriction

        invweight_scale = impratio_invsqrt * impratio_invsqrt
        friction = friction_in[conid]

        if dimid > 1:
          fri0 = friction[0]
          frii = friction[dimid - 1]
          fri = fri0 * fri0 / (frii * frii)
          invweight_scale *= fri

        pos_aref = 0.0
    else:
      if condim > 1:
        friction = friction_in[conid]
        fri0 = friction[0]
        invweight = invweight + fri0 * fri0 * invweight
        invweight = invweight * 2.0 * fri0 * fri0 * impratio_invsqrt * impratio_invsqrt

    if condim == 1:
      efc_type = ConstraintType.CONTACT_FRICTIONLESS
    elif wp.static(IS_ELLIPTIC):
      efc_type = ConstraintType.CONTACT_ELLIPTIC
    else:
      efc_type = ConstraintType.CONTACT_PYRAMIDAL

    _efc_row(
      opt_disableflags,
      worldid,
      timestep,
      efcid,
      pos_aref,
      pos,
      invweight,
      solref_in[conid],
      solimp_in[conid],
      includemargin,
      Jqvel,
      0.0,
      efc_type,
      conid,
      is_discrete,
      0.0,
      efc_type_out,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_vel_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

    if wp.static(IS_ELLIPTIC):
      if dimid > 0:
        # Preserve elliptic friction ratios after _efc_row floors normal regularization.
        efc_D_out[worldid, efcid] /= invweight_scale

        b_fri = _contact_kbimp(opt_disableflags, timestep, ref, solimp_in[conid], pos, is_discrete)[1]
        f_fri = 1.0 + timestep * b_fri if is_discrete else 1.0
        efc_aref_out[worldid, efcid] = -b_fri * Jqvel / f_fri

    if wp.static(flg_adhesion):
      if adhesion_in[conid] != 0.0 and (dimid == 0 or not wp.static(IS_ELLIPTIC)):
        efc_D = efc_D_out[worldid, efcid]
        if efc_D > 0.0:
          adhesion = adhesion_in[conid]
          if not wp.static(IS_ELLIPTIC) and condim > 1:
            adhesion = adhesion / float(2 * (condim - 1))
          efc_aref_out[worldid, efcid] += (1.0 / efc_D) * adhesion

  return kernel


@cache_kernel
def _add_surface_vel(is_pyramidal: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    contact_type_in: wp.array[int],
    contact_surfacevel_in: wp.array[wp.vec3],
    naconmax_in: int,
    nacon_in: wp.array[int],
    # Data out:
    efc_Jqvel_out: wp.array2d[float],
  ):
    conid = wp.tid()

    if conid >= wp.min(nacon_in[0], naconmax_in):
      return

    if not (contact_type_in[conid] & ContactType.CONSTRAINT):
      return

    dim = contact_dim_in[conid]
    if dim <= 1:
      return

    if contact_efc_address_in[conid, 0] < 0:
      return

    sv = contact_surfacevel_in[conid]
    if sv[0] == 0.0 and sv[1] == 0.0 and sv[2] == 0.0:
      return

    worldid = contact_worldid_in[conid]
    ndim = wp.min(dim, 4)

    if not wp.static(is_pyramidal):
      for j in range(1, ndim):
        cs = sv[j - 1]
        if cs != 0.0:
          efcid = contact_efc_address_in[conid, j]
          if efcid >= 0:
            efc_Jqvel_out[worldid, efcid] += cs
    else:
      friction = contact_friction_in[conid]
      for k in range(1, ndim):
        cs = sv[k - 1]
        if cs != 0.0:
          mu_cs = friction[k - 1] * cs
          efcid_pos = contact_efc_address_in[conid, 2 * (k - 1)]
          if efcid_pos >= 0:
            efc_Jqvel_out[worldid, efcid_pos] += mu_cs
          efcid_neg = contact_efc_address_in[conid, 2 * (k - 1) + 1]
          if efcid_neg >= 0:
            efc_Jqvel_out[worldid, efcid_neg] -= mu_cs

  return kernel


def _sort_constraints(m: types.Model, d: types.Data):
  """Order constraint rows canonically for determinism."""
  # TODO(team): Implementation of constraint row sorting.
  pass


@event_scope
def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""
  newton = m.opt.solver == types.SolverType.NEWTON
  is_discrete = m.opt.integrator == types.IntegratorType.DISCRETE
  efc_nnz = wp.empty((d.nworld,), dtype=int)

  wp.launch(
    _zero_constraint_counts,
    dim=d.nworld,
    inputs=[d.ne, d.nf, d.nl, d.nefc, d.efc.jtdaj_nblock, efc_nnz],
  )

  if not (m.opt.disableflags & types.DisableBit.CONSTRAINT):
    if not (m.opt.disableflags & types.DisableBit.EQUALITY):
      wp.launch(
        _equality_connect(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.eq_connect_adr.size),
        inputs=[
          m.nv,
          m.nsite,
          m.opt.timestep,
          m.opt.disableflags,
          m.body_parentid,
          m.body_rootid,
          m.body_weldid,
          m.body_dofnum,
          m.body_dofadr,
          m.body_invweight0,
          m.jnt_type,
          m.jnt_dofadr,
          m.dof_bodyid,
          m.dof_jntid,
          m.dof_parentid,
          m.site_bodyid,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_objtype,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.body_isdofancestor,
          m.eq_connect_adr,
          d.qvel,
          d.eq_active,
          d.xpos,
          d.xmat,
          d.site_xpos,
          d.subtree_com,
          d.cdof,
          d.cvel,
          d.cdof_dot,
          d.subtree_linvel,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.ne,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )
      wp.launch(
        _equality_weld(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.eq_wld_adr.size),
        inputs=[
          m.nv,
          m.nsite,
          m.opt.timestep,
          m.opt.disableflags,
          m.body_parentid,
          m.body_rootid,
          m.body_weldid,
          m.body_dofnum,
          m.body_dofadr,
          m.body_invweight0,
          m.jnt_type,
          m.jnt_dofadr,
          m.dof_bodyid,
          m.dof_jntid,
          m.dof_parentid,
          m.site_bodyid,
          m.site_quat,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_objtype,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.body_isdofancestor,
          m.eq_wld_adr,
          d.qvel,
          d.eq_active,
          d.xpos,
          d.xquat,
          d.xmat,
          d.site_xpos,
          d.subtree_com,
          d.cdof,
          d.cvel,
          d.cdof_dot,
          d.subtree_linvel,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.ne,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )
      wp.launch(
        _equality_joint(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.eq_jnt_adr.size),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.qpos0,
          m.jnt_qposadr,
          m.jnt_dofadr,
          m.dof_invweight0,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.eq_jnt_adr,
          d.qpos,
          d.qvel,
          d.eq_active,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.ne,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )
      wp.launch(
        _equality_tendon(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.eq_ten_adr.size),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
          m.tendon_length0,
          m.tendon_invweight0,
          m.eq_ten_adr,
          d.qvel,
          d.eq_active,
          d.ten_J,
          d.ten_length,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.ne,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )

      if m.nflex > 0:
        flex_block = wp.empty((d.nworld, m.eq_flex_adr.size), dtype=wp.vec3i)
        wp.launch(
          _equality_flex_alloc(m.is_sparse, newton),
          dim=(d.nworld, m.eq_flex_adr.size),
          inputs=[
            m.flex_edgeadr,
            m.flex_edgenum,
            m.flexedge_J_rownnz,
            m.eq_obj1id,
            m.eq_flex_adr,
            m.flex_efcnum,
            m.flex_efcnnz,
            m.flexedge_efcadr,
            d.eq_active,
            d.njmax,
            d.njmax_nnz,
          ],
          outputs=[d.ne, d.nefc, d.efc.jtdaj_nblock, flex_block, efc_nnz],
        )
        wp.launch(
          _equality_flex(m.is_sparse, newton, is_discrete),
          dim=(d.nworld, m.eq_flex_adr.size, m.nflexedge),
          inputs=[
            m.nv,
            m.opt.timestep,
            m.opt.disableflags,
            m.flex_edgeadr,
            m.flex_edgenum,
            m.flexedge_length0,
            m.flexedge_invweight0,
            m.flexedge_J_rownnz,
            m.flexedge_J_rowadr,
            m.flexedge_J_colind,
            m.eq_obj1id,
            m.eq_solref,
            m.eq_solimp,
            m.eq_flex_adr,
            m.flexedge_efcadr,
            m.flexedge_efcnnzadr,
            d.qvel,
            d.flexedge_J,
            d.flexedge_length,
            d.njmax,
            flex_block,
          ],
          outputs=[
            d.efc.type,
            d.efc.id,
            d.efc.jtdaj_adr,
            d.efc.jtdaj_nrow,
            d.efc.J_rownnz,
            d.efc.J_rowadr,
            d.efc.J_colind,
            d.efc.J,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.vel,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )

        if m.eq_flexstrain_adr.size:
          wp.launch(
            _equality_flexstrain(m.is_sparse, newton, is_discrete),
            dim=(d.nworld, m.eq_flexstrain_adr.size),
            inputs=[
              m.nv,
              m.opt.timestep,
              m.opt.disableflags,
              m.body_parentid,
              m.body_rootid,
              m.body_dofnum,
              m.body_dofadr,
              m.body_invweight0,
              m.dof_bodyid,
              m.flex_interp,
              m.flex_cellnum,
              m.flex_nodeadr,
              m.flex_nodenum,
              m.flex_stiffnessadr,
              m.flex_nodebodyid,
              m.flex_node,
              m.flex_node0,
              m.flex_stiffness,
              m.flex_centered,
              m.eq_obj1id,
              m.eq_solref,
              m.eq_solimp,
              m.eq_data,
              m.body_isdofancestor,
              m.eq_flexstrain_adr,
              m.flexstrain_J_rownnz,
              m.flexstrain_J_rowadr,
              m.flexstrain_J_colind,
              d.qvel,
              d.eq_active,
              d.xpos,
              d.xmat,
              d.subtree_com,
              d.cdof,
              d.njmax,
              d.njmax_nnz,
              d.flexnode_xpos,
              d.face_quat,
            ],
            outputs=[
              d.ne,
              d.nefc,
              d.efc.type,
              d.efc.id,
              d.efc.jtdaj_adr,
              d.efc.jtdaj_nrow,
              d.efc.jtdaj_nblock,
              d.efc.J_rownnz,
              d.efc.J_rowadr,
              d.efc.J_colind,
              d.efc.J,
              d.efc.pos,
              d.efc.margin,
              d.efc.D,
              d.efc.vel,
              d.efc.aref,
              d.efc.frictionloss,
              efc_nnz,
            ],
          )

    if not (m.opt.disableflags & types.DisableBit.FRICTIONLOSS):
      wp.launch(
        _friction_dof(m.is_sparse, is_discrete),
        dim=(d.nworld, m.nv),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.dof_solref,
          m.dof_solimp,
          m.dof_frictionloss,
          m.dof_invweight0,
          d.qvel,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.nf,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          d.dof_frictionadr,
          efc_nnz,
        ],
      )

      wp.launch(
        _friction_tendon(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.ntendon),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
          m.tendon_solref_fri,
          m.tendon_solimp_fri,
          m.tendon_frictionloss,
          m.tendon_invweight0,
          d.qvel,
          d.ten_J,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.nf,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )

    # limit
    if not (m.opt.disableflags & types.DisableBit.LIMIT):
      wp.launch(
        _limit_ball(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.jnt_limited_ball_adr.size),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.jnt_qposadr,
          m.jnt_dofadr,
          m.jnt_solref,
          m.jnt_solimp,
          m.jnt_range,
          m.jnt_margin,
          m.dof_invweight0,
          m.jnt_limited_ball_adr,
          d.qpos,
          d.qvel,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.nl,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )

      wp.launch(
        _limit_slide_hinge(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.jnt_limited_slide_hinge_adr.size),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.jnt_qposadr,
          m.jnt_dofadr,
          m.jnt_solref,
          m.jnt_solimp,
          m.jnt_range,
          m.jnt_margin,
          m.dof_invweight0,
          m.jnt_limited_slide_hinge_adr,
          d.qpos,
          d.qvel,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.nl,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )

      wp.launch(
        _limit_tendon(m.is_sparse, newton, is_discrete),
        dim=(d.nworld, m.tendon_limited_adr.size),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.opt.disableflags,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
          m.tendon_solref_lim,
          m.tendon_solimp_lim,
          m.tendon_range,
          m.tendon_margin,
          m.tendon_invweight0,
          m.tendon_limited_adr,
          d.qvel,
          d.ten_J,
          d.ten_length,
          d.njmax,
          d.njmax_nnz,
        ],
        outputs=[
          d.nl,
          d.nefc,
          d.efc.type,
          d.efc.id,
          d.efc.jtdaj_adr,
          d.efc.jtdaj_nrow,
          d.efc.jtdaj_nblock,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.vel,
          d.efc.aref,
          d.efc.frictionloss,
          efc_nnz,
        ],
      )

    # contact
    if not (m.opt.disableflags & types.DisableBit.CONTACT):
      nmaxdim = int(m.nmaxpyramid) if m.opt.cone == types.ConeType.PYRAMIDAL else int(m.nmaxcondim)

      # Reinterpret to avoid unnecessary loads
      contact_frame_2d = wp.array(
        ptr=d.contact.frame.ptr,
        dtype=wp.vec3,
        shape=(d.naconmax, 3),
        device=d.contact.frame.device,
        copy=False,
      )
      contact_friction_2d = wp.array(
        ptr=d.contact.friction.ptr,
        dtype=float,
        shape=(d.naconmax, 5),
        device=d.contact.friction.device,
        copy=False,
      )

      has_flex = m.nflex > 0
      if has_flex:
        wp.launch(
          _efc_contact_init_flex(m.opt.cone, m.is_sparse, newton, m.flg_adhesion),
          dim=d.naconmax,
          inputs=[
            m.body_parentid,
            m.body_weldid,
            m.body_dofnum,
            m.body_dofadr,
            m.jnt_bodyid,
            m.dof_parentid,
            m.geom_bodyid,
            m.flex_dim,
            m.flex_interp,
            m.flex_cellnum,
            m.flex_nodeadr,
            m.flex_vertadr,
            m.flex_elemadr,
            m.flex_elemnum,
            m.flex_elemdataadr,
            m.flex_nodebodyid,
            m.flex_vertbodyid,
            m.flex_elem,
            m.flex_vert0,
            m.flex_rod_has_twist,
            m.flex_rod_edgeadr,
            m.flex_rod_edgenum,
            m.flex_rodedge_jnt,
            m.flex_elem_rodedge,
            m.flex_vert_rodelem,
            d.flexvert_xpos,
            d.njmax,
            d.njmax_nnz,
            d.nacon,
            d.contact.dist,
            d.contact.dim,
            d.contact.includemargin,
            d.contact.adhesion,
            d.contact.worldid,
            d.contact.geom,
            d.contact.flex,
            d.contact.elem,
            d.contact.vert,
            d.contact.pos,
            d.contact.type,
          ],
          outputs=[
            d.nefc,
            d.contact.efc_address,
            d.efc.id,
            d.efc.jtdaj_adr,
            d.efc.jtdaj_nrow,
            d.efc.jtdaj_nblock,
            d.efc.J_rownnz,
            d.efc.J_rowadr,
            efc_nnz,
          ],
        )
      else:
        wp.launch(
          _efc_contact_init(m.opt.cone, m.is_sparse, newton, m.flg_adhesion),
          dim=d.naconmax,
          inputs=[
            m.body_weldid,
            m.body_dofnum,
            m.body_dofadr,
            m.dof_parentid,
            m.geom_bodyid,
            d.njmax,
            d.njmax_nnz,
            d.nacon,
            d.contact.dist,
            d.contact.dim,
            d.contact.includemargin,
            d.contact.adhesion,
            d.contact.worldid,
            d.contact.geom,
            d.contact.type,
          ],
          outputs=[
            d.nefc,
            d.contact.efc_address,
            d.efc.id,
            d.efc.jtdaj_adr,
            d.efc.jtdaj_nrow,
            d.efc.jtdaj_nblock,
            d.efc.J_rownnz,
            d.efc.J_rowadr,
            d.efc.Jqvel,
            efc_nnz,
          ],
        )

      det_atomics = bool(m.opt.deterministic & types.DeterminismType.ATOMICS)
      if m.is_sparse:
        if has_flex:
          wp.launch(
            _efc_contact_jac_sparse_flex(m.opt.cone),
            dim=(d.naconmax, nmaxdim),
            inputs=[
              m.body_parentid,
              m.body_rootid,
              m.body_weldid,
              m.body_dofnum,
              m.body_dofadr,
              m.jnt_dofadr,
              m.jnt_bodyid,
              m.dof_bodyid,
              m.dof_parentid,
              m.geom_type,
              m.geom_bodyid,
              m.flex_dim,
              m.flex_interp,
              m.flex_cellnum,
              m.flex_nodeadr,
              m.flex_vertadr,
              m.flex_elemadr,
              m.flex_elemnum,
              m.flex_elemdataadr,
              m.flex_plasticadr,
              m.flex_nodebodyid,
              m.flex_vertbodyid,
              m.flex_elem,
              m.flex_vert0,
              m.body_isdofancestor,
              m.flex_rod_has_twist,
              m.flex_rod_edgeadr,
              m.flex_rod_edgenum,
              m.flex_rodedge_jnt,
              m.flex_elem_rodedge,
              m.flex_vert_rodelem,
              d.qvel,
              d.plastic,
              d.subtree_com,
              d.cdof,
              d.flexvert_xpos,
              d.contact.efc_address,
              d.efc.J_rownnz,
              d.efc.J_rowadr,
              d.nacon,
              d.contact.dist,
              d.contact.dim,
              d.contact.geom,
              d.contact.flex,
              d.contact.elem,
              d.contact.vert,
              d.contact.pos,
              contact_frame_2d,
              contact_friction_2d,
              d.contact.worldid,
            ],
            outputs=[
              d.efc.J_colind,
              d.efc.J,
              d.efc.Jqvel,
            ],
          )
        else:
          # Loop over the active contacts with a GPU-sized grid rather than launching naconmax
          # threads; several waves, since longer per-thread loops over contacts are slower.
          ncontact_per_pass = d.naconmax
          if wp.get_device().is_cuda:
            ncontact_per_pass = min(ncontact_per_pass, 1024 * wp.get_device().sm_count)
          ncontact_stride = max(ncontact_per_pass, 1)
          max_records = ((d.naconmax + ncontact_stride - 1) // ncontact_stride) * nmaxdim if det_atomics else 0
          wp.launch(
            _efc_contact_jac_sparse(m.opt.cone, nmaxdim, max_records, det_atomics),
            dim=ncontact_stride * _CONTACT_JAC_LANES,
            inputs=[
              m.body_parentid,
              m.body_rootid,
              m.body_weldid,
              m.body_dofnum,
              m.body_dofadr,
              m.dof_bodyid,
              m.dof_parentid,
              m.geom_bodyid,
              m.body_isdofancestor,
              d.qvel,
              d.subtree_com,
              d.cdof,
              d.contact.efc_address,
              d.efc.J_rownnz,
              d.efc.J_rowadr,
              d.naconmax,
              d.nacon,
              d.contact.dim,
              d.contact.geom,
              d.contact.pos,
              contact_frame_2d,
              contact_friction_2d,
              d.contact.worldid,
              ncontact_per_pass,
            ],
            outputs=[
              d.efc.J_colind,
              d.efc.J,
              d.efc.Jqvel,
            ],
          )
      else:
        d.efc.Jqvel.zero_()
        tile_size = m.block_dim.contact_jac_tiled
        n_dof_blocks = (m.nv_pad + tile_size - 1) // tile_size

        if has_flex:
          wp.launch_tiled(
            _efc_contact_jac_dense_flex(
              tile_size,
              m.opt.cone,
              d.njmax if det_atomics else 0,
              det_atomics,
            ),
            dim=(d.nworld, n_dof_blocks),
            inputs=[
              m.body_rootid,
              m.jnt_dofadr,
              m.jnt_bodyid,
              m.geom_type,
              m.geom_bodyid,
              m.flex_dim,
              m.flex_interp,
              m.flex_cellnum,
              m.flex_nodeadr,
              m.flex_vertadr,
              m.flex_elemadr,
              m.flex_elemnum,
              m.flex_elemdataadr,
              m.flex_plasticadr,
              m.flex_nodebodyid,
              m.flex_vertbodyid,
              m.flex_elem,
              m.flex_vert0,
              m.body_isdofancestor,
              m.flex_rod_has_twist,
              m.flex_rod_edgeadr,
              m.flex_rod_edgenum,
              m.flex_rodedge_jnt,
              m.flex_elem_rodedge,
              m.flex_vert_rodelem,
              d.ne,
              d.nf,
              d.nl,
              d.nefc,
              d.qvel,
              d.plastic,
              d.subtree_com,
              d.cdof,
              d.flexvert_xpos,
              d.contact.efc_address,
              d.efc.id,
              d.njmax,
              m.nv_pad,
              d.contact.dist,
              d.contact.dim,
              d.contact.geom,
              d.contact.flex,
              d.contact.elem,
              d.contact.vert,
              d.contact.pos,
              contact_frame_2d,
              contact_friction_2d,
            ],
            outputs=[
              d.efc.J,
              d.efc.Jqvel,
            ],
            block_dim=tile_size,
          )
        else:
          wp.launch_tiled(
            _efc_contact_jac_dense(tile_size, m.opt.cone, d.njmax if det_atomics else 0, det_atomics),
            dim=(d.nworld, n_dof_blocks),
            inputs=[
              m.body_rootid,
              m.geom_bodyid,
              m.body_isdofancestor,
              d.ne,
              d.nf,
              d.nl,
              d.nefc,
              d.qvel,
              d.subtree_com,
              d.cdof,
              d.contact.efc_address,
              d.efc.id,
              d.njmax,
              m.nv_pad,
              d.contact.dim,
              d.contact.geom,
              d.contact.pos,
              contact_frame_2d,
              contact_friction_2d,
            ],
            outputs=[
              d.efc.J,
              d.efc.Jqvel,
            ],
            block_dim=tile_size,
          )

      if m.flg_surfacevel:
        wp.launch(
          _add_surface_vel(m.opt.cone == types.ConeType.PYRAMIDAL),
          dim=d.naconmax,
          inputs=[
            d.contact.friction,
            d.contact.dim,
            d.contact.efc_address,
            d.contact.worldid,
            d.contact.type,
            d.contact.surfacevel,
            d.naconmax,
            d.nacon,
          ],
          outputs=[
            d.efc.Jqvel,
          ],
        )

      if has_flex:
        wp.launch(
          _efc_contact_update_flex(m.opt.cone, m.flg_adhesion, is_discrete),
          dim=(d.naconmax, nmaxdim),
          inputs=[
            m.opt.timestep,
            m.opt.disableflags,
            m.opt.impratio_invsqrt,
            m.body_invweight0,
            m.jnt_bodyid,
            m.geom_bodyid,
            m.flex_dim,
            m.flex_interp,
            m.flex_cellnum,
            m.flex_nodeadr,
            m.flex_vertadr,
            m.flex_elemadr,
            m.flex_elemnum,
            m.flex_elemdataadr,
            m.flex_nodebodyid,
            m.flex_vertbodyid,
            m.flex_elem,
            m.flex_vert0,
            m.flex_rod_has_twist,
            m.flex_rod_edgeadr,
            m.flex_rod_edgenum,
            m.flex_rodedge_jnt,
            m.flex_elem_rodedge,
            m.flex_vert_rodelem,
            d.flexvert_xpos,
            d.contact.efc_address,
            d.efc.Jqvel,
            d.nacon,
            d.contact.dist,
            d.contact.pos,
            d.contact.dim,
            d.contact.includemargin,
            d.contact.worldid,
            d.contact.geom,
            d.contact.flex,
            d.contact.elem,
            d.contact.vert,
            d.contact.friction,
            d.contact.solref,
            d.contact.solreffriction,
            d.contact.solimp,
            d.contact.adhesion,
            d.contact.type,
          ],
          outputs=[
            d.efc.type,
            d.efc.id,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.vel,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )
      else:
        wp.launch(
          _efc_contact_update(m.opt.cone, m.flg_adhesion, is_discrete),
          dim=(d.naconmax, nmaxdim),
          inputs=[
            m.opt.timestep,
            m.opt.disableflags,
            m.opt.impratio_invsqrt,
            m.body_invweight0,
            m.geom_bodyid,
            d.contact.efc_address,
            d.efc.Jqvel,
            d.nacon,
            d.contact.dist,
            d.contact.dim,
            d.contact.includemargin,
            d.contact.worldid,
            d.contact.geom,
            d.contact.friction,
            d.contact.solref,
            d.contact.solreffriction,
            d.contact.solimp,
            d.contact.adhesion,
            d.contact.type,
          ],
          outputs=[
            d.efc.type,
            d.efc.id,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.vel,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )

  if m.is_sparse:
    wp.launch(
      _efc_nnz_overflow(int(m.opt.warn_overflow)),
      dim=d.nworld,
      inputs=[d.njmax_nnz, efc_nnz],
      outputs=[d.overflow],
    )

  if m.opt.deterministic & types.DeterminismType.CONSTRAINT:
    _sort_constraints(m, d)


@cache_kernel
def _regularize_constraint(is_sparse: bool, flg_adhesion: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # Data in:
    nefc_in: wp.array[int],
    M_in: wp.array2d[float],
    qH_in: wp.array2d[float],
    # In:
    type_in: wp.array2d[int],
    id_in: wp.array2d[int],
    J_rownnz_in: wp.array2d[int],
    J_rowadr_in: wp.array2d[int],
    J_colind_in: wp.array3d[int],
    J_in: wp.array3d[float],
    condim_in: wp.array[int],
    efc_address_in: wp.array2d[int],
    adhesion_in: wp.array[float],
    # Out:
    D_out: wp.array2d[float],
    aref_out: wp.array2d[float],
  ):
    worldid, r = wp.tid()
    if r >= nefc_in[worldid]:
      return

    etype = type_in[worldid, r]
    conid = -1
    r_jac = r
    if (
      etype == types.ConstraintType.CONTACT_FRICTIONLESS
      or etype == types.ConstraintType.CONTACT_PYRAMIDAL
      or etype == types.ConstraintType.CONTACT_ELLIPTIC
    ):
      conid = id_in[worldid, r]
      r_jac = efc_address_in[conid, 0]

    num = float(0.0)
    den = float(0.0)
    rownnz = J_rownnz_in[worldid, r_jac] if wp.static(is_sparse) else nv
    rowadr = J_rowadr_in[worldid, r_jac] if wp.static(is_sparse) else 0
    for a in range(rownnz):
      J = J_in[worldid, 0, rowadr + a] if wp.static(is_sparse) else J_in[worldid, r_jac, a]
      if J == 0.0:
        continue
      c = J_colind_in[worldid, 0, rowadr + a] if wp.static(is_sparse) else a
      diag_adr = M_rowadr[c] + M_rownnz[c] - 1
      Mdiag = M_in[worldid, diag_adr]
      qHdiag = qH_in[worldid, diag_adr]
      J2 = J * J
      num += math.safe_div(J2, qHdiag)
      den += math.safe_div(J2, Mdiag)

    if den > types.MJ_MINVAL:
      scale = num / den
      D_old = D_out[worldid, r]
      D_new = wp.min(math.safe_div(D_old, scale), 1.0 / types.MJ_MINVAL)
      D_out[worldid, r] = D_new

      if wp.static(flg_adhesion) and conid >= 0 and D_old > types.MJ_MINVAL and D_new > types.MJ_MINVAL:
        adhesion = adhesion_in[conid]
        if adhesion != 0.0:
          if etype == types.ConstraintType.CONTACT_FRICTIONLESS or (
            etype == types.ConstraintType.CONTACT_ELLIPTIC and r == r_jac
          ):
            aref_out[worldid, r] += (1.0 / D_new - 1.0 / D_old) * adhesion
          elif etype == types.ConstraintType.CONTACT_PYRAMIDAL:
            condim = condim_in[conid]
            if condim > 1:
              aref_out[worldid, r] += (1.0 / D_new - 1.0 / D_old) * (adhesion / float(2 * (condim - 1)))

  return kernel


@event_scope
def regularize_constraint(m: types.Model, d: types.Data):
  """Regularizes constraints against effective metric (discrete integrator)."""
  wp.launch(
    _regularize_constraint(m.is_sparse, m.flg_adhesion),
    dim=(d.nworld, d.efc.D.shape[1]),
    inputs=[
      m.nv,
      m.M_rownnz,
      m.M_rowadr,
      d.nefc,
      d.M,
      d.qH,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.contact.dim,
      d.contact.efc_address,
      d.contact.adhesion,
    ],
    outputs=[d.efc.D, d.efc.aref],
  )
