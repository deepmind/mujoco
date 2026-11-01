# Copyright 2026 The Newton Developers
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

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import EqType
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import SleepPolicy
from mujoco.mjx.third_party.mujoco_warp._src.types import SleepState
from mujoco.mjx.third_party.mujoco_warp._src.types import WrapType
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# tree_asleep value for fully awake tree
K_AWAKE_VAL = -(1 + types.MJ_MINAWAKE)


@wp.func
def _sleep_cycle(tree_asleep: wp.array2d[int], ntree: int, worldid: int, treeid: int) -> int:  # kernel_analyzer: ignore
  if treeid < 0 or treeid >= ntree:
    return -1

  smallest = int(treeid)
  current = int(treeid)
  count = int(0)

  for step in range(ntree + 1):  # safe upper bound
    next_tree = tree_asleep[worldid, current]
    if next_tree < 0 or next_tree >= ntree:
      return -1

    if next_tree < smallest:
      smallest = next_tree

    current = next_tree
    count += 1
    if current == treeid:
      break

  return smallest


@wp.func
def _tendon_limit_active(
  tendon_limited: wp.array[int],  # kernel_analyzer: ignore
  tendon_range: wp.array2d[wp.vec2],  # kernel_analyzer: ignore
  tendon_margin: wp.array2d[float],  # kernel_analyzer: ignore
  ten_length: wp.array2d[float],  # kernel_analyzer: ignore
  worldid: int,  # kernel_analyzer: ignore
  tenid: int,
) -> bool:
  if tendon_limited[tenid] == 0:
    return False

  length = ten_length[worldid, tenid]
  margin = tendon_margin[worldid % tendon_margin.shape[0], tenid]
  r = tendon_range[worldid % tendon_range.shape[0], tenid]

  # low limit
  dist_low = length - r[0]
  if dist_low < margin:
    return True

  # high limit
  dist_high = r[1] - length
  if dist_high < margin:
    return True

  return False


@wp.kernel
def _zero_sleep_counters(
  # Data out:
  ntree_awake_out: wp.array[int],
  nbody_awake_out: wp.array[int],
  nv_awake_out: wp.array[int],
):
  worldid = wp.tid()
  ntree_awake_out[worldid] = 0
  nbody_awake_out[worldid] = 0
  nv_awake_out[worldid] = 0


@wp.kernel
def _update_sleep_trees(
  # Data in:
  tree_asleep_in: wp.array2d[int],
  # Data out:
  ntree_awake_out: wp.array[int],
  tree_awake_out: wp.array2d[int],
):
  worldid, treeid = wp.tid()
  is_awake = int(tree_asleep_in[worldid, treeid] < 0)
  tree_awake_out[worldid, treeid] = is_awake
  if is_awake == 1:
    wp.atomic_add(ntree_awake_out, worldid, 1)  # kernel_analyzer: ignore[atomic]


@cache_kernel
def _update_sleep_bodies(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mocapid: wp.array[int],
    body_treeid: wp.array[int],
    # Data in:
    tree_awake_in: wp.array2d[int],
    # In:
    flg_staticawake: int,
    # Data out:
    nbody_awake_out: wp.array[int],
    body_awake_out: wp.array2d[int],
    body_awake_ind_out: wp.array2d[int],
  ):
    worldid, bodyid = wp.tid()

    tree = body_treeid[bodyid]

    # check static
    if tree < 0:
      root = body_rootid[bodyid]
      mocap = body_mocapid[root]
      if mocap >= 0:
        state = SleepState.AWAKE
      else:
        state = SleepState.AWAKE if flg_staticawake != 0 else SleepState.STATIC
    else:
      state = SleepState.AWAKE if tree_awake_in[worldid, tree] == 1 else SleepState.ASLEEP

    body_awake_out[worldid, bodyid] = state

    if state != SleepState.ASLEEP:
      idx = wp.atomic_add(nbody_awake_out, worldid, 1)
      body_awake_ind_out[worldid, idx] = bodyid

  return kernel


@cache_kernel
def _update_sleep_dofs(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_treeid: wp.array[int],
    dof_bodyid: wp.array[int],
    # Data in:
    body_awake_in: wp.array2d[int],
    # Data out:
    nv_awake_out: wp.array[int],
    dof_awake_ind_out: wp.array2d[int],
  ):
    worldid, dofid = wp.tid()
    bodyid = dof_bodyid[dofid]
    if body_treeid[bodyid] >= 0 and body_awake_in[worldid, bodyid] == SleepState.AWAKE:
      idx = wp.atomic_add(nv_awake_out, worldid, 1)
      dof_awake_ind_out[worldid, idx] = dofid

  return kernel


@event_scope
def update_sleep(m: types.Model, d: types.Data, flg_staticawake: int = 0):
  """Computes sleeping arrays from tree_asleep."""
  wp.launch(
    _zero_sleep_counters,
    dim=d.nworld,
    inputs=[],
    outputs=[d.ntree_awake, d.nbody_awake, d.nv_awake],
  )

  wp.launch(
    _update_sleep_trees,
    dim=(d.nworld, m.ntree),
    inputs=[d.tree_asleep],
    outputs=[d.ntree_awake, d.tree_awake],
  )

  wp.launch(
    _update_sleep_bodies(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.nbody),
    inputs=[
      m.body_parentid,
      m.body_rootid,
      m.body_mocapid,
      m.body_treeid,
      d.tree_awake,
      flg_staticawake,
    ],
    outputs=[
      d.nbody_awake,
      d.body_awake,
      d.body_awake_ind,
    ],
  )

  wp.launch(
    _update_sleep_dofs(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.nv),
    inputs=[
      m.body_treeid,
      m.dof_bodyid,
      d.body_awake,
    ],
    outputs=[d.nv_awake, d.dof_awake_ind],
  )


@event_scope
def update_sleep_trees(m: types.Model, d: types.Data):
  """Lightweight update of tree_awake array only, avoiding body/dof kernel launches."""
  wp.launch(
    _zero_sleep_counters,
    dim=d.nworld,
    inputs=[],
    outputs=[d.ntree_awake, d.nbody_awake, d.nv_awake],
  )

  wp.launch(
    _update_sleep_trees,
    dim=(d.nworld, m.ntree),
    inputs=[d.tree_asleep],
    outputs=[d.ntree_awake, d.tree_awake],
  )


@wp.func
def _wake_tree(
  # Model:
  ntree: int,
  # In:
  worldid: int,
  treeid: int,
  wakeval: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Wakes tree treeid and its associated cycle, returning number of woke trees."""
  if treeid < 0 or treeid >= ntree:
    return 0

  asleep_val = tree_asleep_out[worldid, treeid]
  if asleep_val < 0:
    if wakeval < asleep_val:
      tree_asleep_out[worldid, treeid] = wakeval
    return 0

  nwoke = int(0)
  current = int(treeid)
  for step in range(ntree + 1):  # safe upper bound
    next_tree = tree_asleep_out[worldid, current]
    if next_tree < 0 or next_tree >= ntree:
      break

    tree_asleep_out[worldid, current] = wakeval
    nwoke += 1
    current = next_tree
    if current == treeid:
      break

  return nwoke


@wp.func
def _tree_can_sleep(
  # Model:
  nbody: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  treeid: int,
  sleep_tolerance: float,
) -> bool:
  policy = tree_sleep_policy[treeid]
  if policy == SleepPolicy.AUTO_NEVER:
    return False

  # check xfrc_applied
  for b in range(nbody):
    if body_treeid[b] == treeid:
      xfrc = xfrc_applied_in[worldid, b]
      for i in range(6):
        if xfrc[i] != 0.0:
          return False

  # check qfrc_applied
  dofadr = tree_dofadr[treeid]
  dofnum = tree_dofnum[treeid]
  for d in range(dofnum):
    if qfrc_applied_in[worldid, dofadr + d] != 0.0:
      return False

  # check qvel
  for d in range(dofnum):
    dof_idx = dofadr + d
    v = qvel_in[worldid, dof_idx]
    weight = dof_length[dof_idx]
    if sleep_tolerance > 0.0:
      if wp.abs(weight * v) >= sleep_tolerance:
        return False
    else:
      if v != 0.0:
        return False

  return True


@wp.kernel
def _wake_kernel(
  # Model:
  nbody: int,
  ntree: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  tree_awake_in: wp.array2d[int],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  worldid, treeid = wp.tid()

  asleep = int(tree_asleep_out[worldid, treeid] >= 0)
  if asleep == 0:
    return

  # if tree_awake mismatch or cannot sleep: wake up
  if tree_awake_in[worldid, treeid] == 1 or not _tree_can_sleep(
    nbody,
    body_treeid,
    dof_length,
    tree_dofadr,
    tree_dofnum,
    tree_sleep_policy,
    qvel_in,
    qfrc_applied_in,
    xfrc_applied_in,
    worldid,
    treeid,
    0.0,  # zero tolerance
  ):
    _wake_tree(ntree, worldid, treeid, K_AWAKE_VAL, tree_asleep_out)


@wp.kernel
def _wake_collision_kernel(
  # Model:
  ntree: int,
  body_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  conid = wp.tid()
  if conid >= nacon_in[0]:
    return

  geom_pair = contact_geom_in[conid]
  g1 = geom_pair[0]
  g2 = geom_pair[1]

  if g1 < 0 or g2 < 0:
    return

  b1 = geom_bodyid[g1]
  b2 = geom_bodyid[g2]
  tree1 = body_treeid[b1]
  tree2 = body_treeid[b2]

  if tree1 < 0 or tree2 < 0:
    return

  worldid = contact_worldid_in[conid]
  awake1 = tree_awake_in[worldid, tree1]
  awake2 = tree_awake_in[worldid, tree2]

  if awake1 == 1 and awake2 == 1:
    return

  if awake1 == 0 and awake2 == 0:
    return

  # wake sleeping tree
  sleeping_tree = tree2 if awake1 == 1 else tree1
  wakeval = tree_asleep_out[worldid, tree1] if awake1 == 1 else tree_asleep_out[worldid, tree2]

  _wake_tree(ntree, worldid, sleeping_tree, wakeval, tree_asleep_out)


@wp.kernel
def _wake_tendon_kernel(
  # Model:
  ntree: int,
  ntendon: int,
  opt_disableflags: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  tendon_limited: wp.array[int],
  tendon_range: wp.array2d[wp.vec2],
  tendon_margin: wp.array2d[float],
  tendon_frictionloss: wp.array2d[float],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  ten_length_in: wp.array2d[float],
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  worldid, tenid = wp.tid()

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]

  # Pass 1: Check if any tree involved in the tendon is awake
  any_awake = int(0)
  wakeval = int(K_AWAKE_VAL)

  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 1:
        any_awake = 1
        val = tree_asleep_out[worldid, t]
        if val < wakeval:
          wakeval = val

  # Pass 2: If at least one tree is awake and a limit or frictionloss constraint is active, wake all sleeping trees
  if any_awake == 1:
    has_limit = not (opt_disableflags & DisableBit.LIMIT) and _tendon_limit_active(
      tendon_limited, tendon_range, tendon_margin, ten_length_in, worldid, tenid
    )
    has_friction = (
      not (opt_disableflags & DisableBit.FRICTIONLOSS)
      and tendon_frictionloss[worldid % tendon_frictionloss.shape[0], tenid] > 0.0
    )
    if has_limit or has_friction:
      for i in range(num):
        idx = adr + i
        t_type = wrap_type[idx]
        t_objid = wrap_objid[idx]
        t = int(-1)
        if t_type == WrapType.JOINT:
          t = body_treeid[jnt_bodyid[t_objid]]
        elif t_type == WrapType.SITE:
          t = body_treeid[site_bodyid[t_objid]]
        elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
          t = body_treeid[geom_bodyid[t_objid]]

        if t >= 0:
          if tree_awake_in[worldid, t] == 0:
            _wake_tree(ntree, worldid, t, wakeval, tree_asleep_out)


@wp.func
def _tendon_wake_val(
  # Model:
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Returns the minimum (most-awake) tree_asleep value if any tree in tendon is awake, else 0."""
  if tenid < 0:
    return 0

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]
  wakeval = int(0)

  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 1:
        val = tree_asleep_out[worldid, t]
        if wakeval == 0 or val < wakeval:
          wakeval = val

  return wakeval


@wp.func
def _wake_tendon_trees(
  # Model:
  ntree: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  wakeval: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  """Wakes up all sleeping trees associated with a tendon."""
  if tenid < 0:
    return

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]
  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 0:
        _wake_tree(ntree, worldid, t, wakeval, tree_asleep_out)


@wp.kernel
def _wake_equality_kernel(
  # Model:
  ntree: int,
  neq: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  eq_active_in: wp.array2d[bool],  # kernel_analyzer: ignore
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  worldid, eqid = wp.tid()

  if not eq_active_in[worldid, eqid]:
    return

  eqtype = eq_type[eqid]
  id1 = eq_obj1id[eqid]
  id2 = eq_obj2id[eqid]

  if eqtype == EqType.CONNECT or eqtype == EqType.WELD or eqtype == EqType.JOINT:
    tree1 = int(-1)
    tree2 = int(-1)

    if eqtype == EqType.CONNECT or eqtype == EqType.WELD:
      if eq_objtype[eqid] == ObjType.BODY:
        tree1 = body_treeid[id1]
        tree2 = body_treeid[id2]
      else:
        tree1 = body_treeid[site_bodyid[id1]]
        tree2 = body_treeid[site_bodyid[id2]]
    elif eqtype == EqType.JOINT:
      tree1 = body_treeid[jnt_bodyid[id1]] if id1 >= 0 else -1
      tree2 = body_treeid[jnt_bodyid[id2]] if id2 >= 0 else -1

    s1 = tree_awake_in[worldid, tree1] if tree1 >= 0 else SleepState.STATIC
    s2 = tree_awake_in[worldid, tree2] if tree2 >= 0 else SleepState.STATIC

    if s1 != SleepState.ASLEEP and s2 != SleepState.ASLEEP:
      return
    if s1 == SleepState.STATIC or s2 == SleepState.STATIC:
      return
    if tree1 == tree2:
      return

    if s1 == SleepState.ASLEEP and s2 == SleepState.ASLEEP:
      cycle1 = _sleep_cycle(tree_asleep_out, ntree, worldid, tree1)
      cycle2 = _sleep_cycle(tree_asleep_out, ntree, worldid, tree2)
      if cycle1 != cycle2:
        _wake_tree(ntree, worldid, tree1, K_AWAKE_VAL, tree_asleep_out)
        _wake_tree(ntree, worldid, tree2, K_AWAKE_VAL, tree_asleep_out)
    else:
      sleeping_tree = tree1 if s1 == SleepState.ASLEEP else tree2
      _wake_tree(ntree, worldid, sleeping_tree, K_AWAKE_VAL, tree_asleep_out)

  elif eqtype == EqType.TENDON:
    ten1 = id1
    ten2 = id2
    w1 = _tendon_wake_val(
      body_treeid,
      jnt_bodyid,
      geom_bodyid,
      site_bodyid,
      tendon_adr,
      tendon_num,
      wrap_type,
      wrap_objid,
      tree_awake_in,
      worldid,
      ten1,
      tree_asleep_out,
    )
    w2 = _tendon_wake_val(
      body_treeid,
      jnt_bodyid,
      geom_bodyid,
      site_bodyid,
      tendon_adr,
      tendon_num,
      wrap_type,
      wrap_objid,
      tree_awake_in,
      worldid,
      ten2,
      tree_asleep_out,
    )

    if w1 < 0 or w2 < 0:
      wakeval = int(K_AWAKE_VAL)
      if w1 < 0 and w1 < wakeval:
        wakeval = w1
      if w2 < 0 and w2 < wakeval:
        wakeval = w2

      _wake_tendon_trees(
        ntree,
        body_treeid,
        jnt_bodyid,
        geom_bodyid,
        site_bodyid,
        tendon_adr,
        tendon_num,
        wrap_type,
        wrap_objid,
        tree_awake_in,
        worldid,
        ten1,
        wakeval,
        tree_asleep_out,
      )
      _wake_tendon_trees(
        ntree,
        body_treeid,
        jnt_bodyid,
        geom_bodyid,
        site_bodyid,
        tendon_adr,
        tendon_num,
        wrap_type,
        wrap_objid,
        tree_awake_in,
        worldid,
        ten2,
        wakeval,
        tree_asleep_out,
      )

  # TODO(team): Implement waking for EqType.FLEX constraints.


@event_scope
def wake(m: types.Model, d: types.Data):
  """Wakes sleeping trees due to user changes/perturbations."""
  wp.launch(
    _wake_kernel,
    dim=(d.nworld, m.ntree),
    inputs=[
      m.nbody,
      m.ntree,
      m.body_treeid,
      m.dof_length,
      m.tree_dofadr,
      m.tree_dofnum,
      m.tree_sleep_policy,
      d.qvel,
      d.qfrc_applied,
      d.xfrc_applied,
      d.tree_awake,
    ],
    outputs=[d.tree_asleep],
  )


@event_scope
def wake_collision(m: types.Model, d: types.Data):
  """Wakes sleeping trees that touch awake trees."""
  wp.launch(
    _wake_collision_kernel,
    dim=d.naconmax,
    inputs=[
      m.ntree,
      m.body_treeid,
      m.geom_bodyid,
      d.tree_awake,
      d.contact.geom,
      d.contact.worldid,
      d.nacon,
    ],
    outputs=[d.tree_asleep],
  )


@event_scope
def wake_tendon(m: types.Model, d: types.Data):
  """Wakes sleeping trees with constrained tendons."""
  if m.ntendon == 0 or (m.opt.disableflags & DisableBit.CONSTRAINT):
    return

  wp.launch(
    _wake_tendon_kernel,
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.ntree,
      m.ntendon,
      m.opt.disableflags,
      m.body_treeid,
      m.jnt_bodyid,
      m.geom_bodyid,
      m.site_bodyid,
      m.tendon_adr,
      m.tendon_num,
      m.tendon_limited,
      m.tendon_range,
      m.tendon_margin,
      m.tendon_frictionloss,
      m.wrap_type,
      m.wrap_objid,
      d.ten_length,
      d.tree_awake,
    ],
    outputs=[d.tree_asleep],
  )


@event_scope
def wake_equality(m: types.Model, d: types.Data):
  """Wakes sleeping trees connected by equality constraints."""
  if m.neq == 0:
    return

  wp.launch(
    _wake_equality_kernel,
    dim=(d.nworld, m.neq),
    inputs=[
      m.ntree,
      m.neq,
      m.body_treeid,
      m.jnt_bodyid,
      m.geom_bodyid,
      m.site_bodyid,
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.tendon_adr,
      m.tendon_num,
      m.wrap_type,
      m.wrap_objid,
      d.eq_active,
      d.tree_awake,
    ],
    outputs=[d.tree_asleep],
  )


@wp.kernel
def _sweep_awake_trees(  # kernel_analyzer: ignore
  # Model:
  nbody: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # In:
  opt_sleep_tolerance: wp.array[float],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  worldid, treeid = wp.tid()
  sleep_tolerance = opt_sleep_tolerance[worldid % opt_sleep_tolerance.shape[0]]
  as_val = tree_asleep_out[worldid, treeid]
  if as_val >= 0:
    return

  if _tree_can_sleep(
    nbody,
    body_treeid,
    dof_length,
    tree_dofadr,
    tree_dofnum,
    tree_sleep_policy,
    qvel_in,
    qfrc_applied_in,
    xfrc_applied_in,
    worldid,
    treeid,
    sleep_tolerance,
  ):
    if as_val < -1:
      tree_asleep_out[worldid, treeid] = as_val + 1
  else:
    tree_asleep_out[worldid, treeid] = K_AWAKE_VAL


@wp.kernel
def _check_island_can_sleep(
  # Model:
  ntree: int,
  # Data in:
  nisland_in: wp.array[int],
  tree_asleep_in: wp.array2d[int],
  tree_island_in: wp.array2d[int],
  # Out:
  island_can_sleep_out: wp.array2d[int],
):
  worldid, treeid = wp.tid()
  nisland = nisland_in[worldid]
  island_id = tree_island_in[worldid, treeid]
  if island_id >= 0 and island_id < nisland:
    as_val = tree_asleep_in[worldid, treeid]
    if as_val < -1:
      # Not ready to sleep yet
      wp.atomic_min(island_can_sleep_out, worldid, island_id, 0)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _build_cycles(  # kernel_analyzer: ignore
  # Model:
  ntree: int,
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nisland_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  # In:
  island_can_sleep_in: wp.array2d[int],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  qvel_out: wp.array2d[float],  # kernel_analyzer: ignore
  qacc_out: wp.array2d[float],  # kernel_analyzer: ignore
):
  worldid = wp.tid()

  num_islands = nisland_in[worldid]
  for island_id in range(num_islands):
    if island_can_sleep_in[worldid, island_id] == 1:
      first_tree = int(-1)
      prev_tree = int(-1)
      for t in range(ntree):
        if tree_island_in[worldid, t] == island_id:
          if first_tree == -1:
            first_tree = t
          if prev_tree != -1:
            tree_asleep_out[worldid, prev_tree] = t
          prev_tree = t

          # Zero velocities and accelerations
          dofadr = tree_dofadr[t]
          dofnum = tree_dofnum[t]
          for d in range(dofnum):
            qvel_out[worldid, dofadr + d] = 0.0
            qacc_out[worldid, dofadr + d] = 0.0

      if first_tree != -1:
        tree_asleep_out[worldid, prev_tree] = first_tree

  # Sleep unconstrained trees
  for t in range(ntree):
    island_id = tree_island_in[worldid, t]
    if island_id < 0 or island_id >= num_islands:
      if tree_asleep_out[worldid, t] == -1:
        tree_asleep_out[worldid, t] = t  # self-cycle

      # Ensure sleeping tree dof velocity and acceleration remain exactly zero
      if tree_asleep_out[worldid, t] >= 0:
        dofadr = tree_dofadr[t]
        dofnum = tree_dofnum[t]
        for d in range(dofnum):
          # TODO(team): shouldn't be necessary to zero if island is already asleep
          qvel_out[worldid, dofadr + d] = 0.0
          qacc_out[worldid, dofadr + d] = 0.0


@event_scope
def sleep(m: types.Model, d: types.Data):
  """Puts trees to sleep according to velocity tolerance."""
  # 1. Sweep over awake trees and increment counter if they can sleep
  wp.launch(
    _sweep_awake_trees,
    dim=(d.nworld, m.ntree),
    inputs=[
      m.nbody,
      m.body_treeid,
      m.dof_length,
      m.tree_dofadr,
      m.tree_dofnum,
      m.tree_sleep_policy,
      d.qvel,
      d.qfrc_applied,
      d.xfrc_applied,
      m.opt.sleep_tolerance,
    ],
    outputs=[d.tree_asleep],
  )

  # 2. Check which constraint islands can sleep (all trees in island must be asleep)
  island_can_sleep = wp.ones((d.nworld, m.ntree), dtype=int)
  wp.launch(
    _check_island_can_sleep,
    dim=(d.nworld, m.ntree),
    inputs=[
      m.ntree,
      d.nisland,
      d.tree_asleep,
      d.tree_island,
    ],
    outputs=[island_can_sleep],
  )

  # 3. Build sleep cycles for sleeping islands and sleep unconstrained trees
  wp.launch(
    _build_cycles,
    dim=d.nworld,
    inputs=[
      m.ntree,
      m.tree_dofadr,
      m.tree_dofnum,
      d.nisland,
      d.tree_island,
      island_can_sleep,
    ],
    outputs=[
      d.tree_asleep,
      d.qvel,
      d.qacc,
    ],
  )
