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

from typing import Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src.types import ConstraintType
from mujoco.mjx.third_party.mujoco_warp._src.types import EqType
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"default_grid_stride": False})


@wp.func
def _dsu_find(parent: wp.array2d[int], worldid: int, tree: int):
  """Find a root while shortening its strictly decreasing parent path."""
  current = tree
  while True:
    parent_current = parent[worldid, current]
    if parent_current == current:
      return current
    grandparent = parent[worldid, parent_current]
    if grandparent < parent_current:
      wp.atomic_min(parent, worldid, current, grandparent)  # kernel_analyzer: ignore[atomic]
    current = parent_current
  return current


@wp.func
def _dsu_activate(worldid: int, tree: int, tree_island_out: wp.array2d[int]):
  if tree >= 0:
    wp.atomic_max(tree_island_out, worldid, tree, 0)  # kernel_analyzer: ignore[atomic]


@wp.func
def _dsu_union(parent: wp.array2d[int], worldid: int, tree0: int, tree1: int, tree_island_out: wp.array2d[int]):
  """Activate endpoints and atomically hook the higher root to the lower root."""
  if tree0 < 0:
    tree0 = tree1
    tree1 = -1
  if tree0 < 0:
    return
  _dsu_activate(worldid, tree0, tree_island_out)
  if tree1 < 0:
    return
  _dsu_activate(worldid, tree1, tree_island_out)

  while True:
    root0 = _dsu_find(parent, worldid, tree0)
    root1 = _dsu_find(parent, worldid, tree1)
    if root0 == root1:
      return
    low_root = wp.min(root0, root1)
    high_root = wp.max(root0, root1)
    previous = wp.atomic_cas(parent, worldid, high_root, high_root, low_root)  # kernel_analyzer: ignore[atomic]
    if previous == high_root:
      return


@wp.kernel
def _reset_dsu(
  # Data out:
  nisland_out: wp.array[int],
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Reset identity parents and active/output markers in parallel."""
  worldid, treeid = wp.tid()
  island_parent_out[worldid, treeid] = treeid
  tree_island_out[worldid, treeid] = -1
  if treeid == 0:
    nisland_out[worldid] = 0


@wp.func
def _is_repeated_fixed_support_efc(
  # Data in:
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  # In:
  worldid: int,
  efcid: int,
) -> bool:
  """Return whether the preceding scalar row has the same fixed tree support."""
  if efcid == 0:
    return False
  return (
    efc_type_in[worldid, efcid - 1] == efc_type_in[worldid, efcid]
    and efc_id_in[worldid, efcid - 1] == efc_id_in[worldid, efcid]
  )


@wp.func
def _flex_contact_bodies(
  # Model:
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
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
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  # In:
  worldid: int,
  conid: int,
) -> Tuple[wp.vec4i, wp.vec4i]:
  """Return the bodies of both sides of a flex contact, -1 padded.

  The rows of a flex contact can vanish (e.g. contact point on a hinge axis), so its trees
  are taken from its bodies, as for its Jacobian, rather than from the values of its rows.
  """
  geom = contact_geom_in[conid]
  flex = contact_flex_in[conid]
  elem = contact_elem_in[conid]
  vert = contact_vert_in[conid]
  pos = contact_pos_in[conid]
  bodies0, _weights0 = support.contact_bodies_and_weights(
    jnt_bodyid,
    geom_bodyid,
    flex_dim,
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
    pos,
    worldid,
  )
  bodies1, _weights1 = support.contact_bodies_and_weights(
    jnt_bodyid,
    geom_bodyid,
    flex_dim,
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
    pos,
    worldid,
  )
  return bodies0, bodies1


@wp.kernel
def _flex_island_dsu(
  # Model:
  opt_enableflags: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  flex_interp: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_eff_possible: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  tree_awake_in: wp.array2d[int],
  # In:
  sleep_metric_flex: bool,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Unions awake dynamic trees of each metric-coupled flex (vertices, nodes, twist joints)."""
  worldid, f = wp.tid()
  if not flex_eff_possible[f]:
    return
  if nefc_in[worldid] == 0 and not sleep_metric_flex:
    return
  sleep_enabled = (opt_enableflags & types.EnableBit.SLEEP) != 0
  use_interp = flex_interp[f] != 0
  num = flex_nodenum[f] if use_interp else flex_vertnum[f]
  adr = flex_nodeadr[f] if use_interp else flex_vertadr[f]
  tree1 = int(-1)
  for j in range(num):
    b = flex_nodebodyid[adr + j] if use_interp else flex_vertbodyid[adr + j]
    tree2 = body_treeid[b]
    if tree2 < 0 or tree2 == tree1 or (sleep_enabled and tree_awake_in[worldid, tree2] == 0):
      continue
    if tree1 < 0:
      tree1 = tree2
    else:
      _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)
  if flex_rod_has_twist[f]:
    eadr = flex_rod_edgeadr[f]
    for e in range(flex_rod_edgenum[f]):
      jnt = flex_rodedge_jnt[eadr + e]
      if jnt < 0:
        continue
      tree2 = body_treeid[jnt_bodyid[jnt]]
      if tree2 < 0 or tree2 == tree1 or (sleep_enabled and tree_awake_in[worldid, tree2] == 0):
        continue
      if tree1 < 0:
        tree1 = tree2
      else:
        _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)


@wp.kernel
def _passive_contact_island_dsu(
  # Model:
  opt_enableflags: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
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
  nefc_in: wp.array[int],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  tree_awake_in: wp.array2d[int],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  sleep_metric_flex: bool,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Unions awake dynamic trees coupled by passive flex contacts under the discrete integrator."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  if (contact_type_in[cid] & int(types.ContactType.PASSIVE)) == 0:
    return
  worldid = contact_worldid_in[cid]
  if nefc_in[worldid] == 0 and not sleep_metric_flex:
    return
  sleep_enabled = (opt_enableflags & types.EnableBit.SLEEP) != 0
  bodies0, bodies1 = _flex_contact_bodies(
    jnt_bodyid,
    geom_bodyid,
    flex_dim,
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
    contact_pos_in,
    contact_geom_in,
    contact_flex_in,
    contact_elem_in,
    contact_vert_in,
    worldid,
    cid,
  )
  tree1 = int(-1)
  for k in range(8):
    body = bodies0[k % 4] if k < 4 else bodies1[k % 4]
    if body < 0:
      continue
    tree2 = body_treeid[body]
    if tree2 < 0 or tree2 == tree1 or (sleep_enabled and tree_awake_in[worldid, tree2] == 0):
      continue
    if tree1 < 0:
      tree1 = tree2
    else:
      _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)


@wp.kernel(module="unique")
def _island_dsu(
  # Model:
  nv: int,
  opt_enableflags: int,
  body_treeid: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  dof_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  flex_dim: wp.array[int],
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
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  is_sparse: bool,
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  tree_awake_in: wp.array2d[int],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  njmax_in: int,
  # In:
  chunk_size: int,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Process one chunk of one world's active EFC prefix with one strided CUDA warp.

  Blocks are laid out over (world, chunk) rather than world alone, so resident
  parallelism tracks total constraint capacity instead of batch size. A single world
  with a long prefix fills the device the same way a large batch of short ones does.
  """
  worldid, chunkid, lane = wp.tid()
  sleep_enabled = (opt_enableflags & types.EnableBit.SLEEP) != 0
  chunk_beg = chunkid * chunk_size
  chunk_end = wp.min(chunk_beg + chunk_size, wp.min(njmax_in, nefc_in[worldid]))
  for efcid in range(chunk_beg + lane, chunk_end, wp.block_dim()):
    efc_type = efc_type_in[worldid, efcid]
    efc_id = efc_id_in[worldid, efcid]
    repeated = _is_repeated_fixed_support_efc(efc_type_in, efc_id_in, worldid, efcid)
    tree0 = int(-1)
    tree1 = int(-1)
    use_generic = int(0)

    if efc_type == ConstraintType.EQUALITY:
      eq_t = eq_type[efc_id]
      if eq_t == EqType.CONNECT or eq_t == EqType.WELD:
        if repeated:
          continue
        body0 = eq_obj1id[efc_id]
        body1 = eq_obj2id[efc_id]
        if eq_objtype[efc_id] == ObjType.SITE:
          body0 = site_bodyid[body0]
          body1 = site_bodyid[body1]
        tree0 = body_treeid[body0]
        tree1 = body_treeid[body1]
      else:
        use_generic = 1
    elif efc_type == ConstraintType.FRICTION_DOF:
      if repeated:
        continue
      tree0 = dof_treeid[efc_id]
    elif efc_type == ConstraintType.LIMIT_JOINT:
      if repeated:
        continue
      tree0 = dof_treeid[jnt_dofadr[efc_id]]
    elif (
      efc_type == ConstraintType.CONTACT_FRICTIONLESS
      or efc_type == ConstraintType.CONTACT_PYRAMIDAL
      or efc_type == ConstraintType.CONTACT_ELLIPTIC
    ):
      if repeated:
        continue
      geom_pair = contact_geom_in[efc_id]
      if geom_pair[0] >= 0 and geom_pair[1] >= 0:
        tree0 = body_treeid[geom_bodyid[geom_pair[0]]]
        tree1 = body_treeid[geom_bodyid[geom_pair[1]]]
      else:
        # flex: union the trees of the bodies of both sides
        bodies0, bodies1 = _flex_contact_bodies(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
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
          contact_pos_in,
          contact_geom_in,
          contact_flex_in,
          contact_elem_in,
          contact_vert_in,
          worldid,
          efc_id,
        )
        first_tree = int(-1)
        for k in range(8):
          body = bodies0[k % 4]
          if k >= 4:
            body = bodies1[k % 4]
          if body < 0:
            continue
          tree = body_treeid[body]
          if tree < 0 or (sleep_enabled and tree_awake_in[worldid, tree] == 0):
            continue
          if first_tree < 0:
            first_tree = tree
            _dsu_union(island_parent_out, worldid, tree, -1, tree_island_out)
          elif tree != first_tree:
            _dsu_union(island_parent_out, worldid, first_tree, tree, tree_island_out)
        continue
    else:
      use_generic = 1

    if use_generic == 0:
      if sleep_enabled:
        if tree0 >= 0 and tree_awake_in[worldid, tree0] == 0:
          tree0 = -1
        if tree1 >= 0 and tree_awake_in[worldid, tree1] == 0:
          tree1 = -1
      _dsu_union(island_parent_out, worldid, tree0, tree1, tree_island_out)
      continue

    first_tree = int(-1)
    count = nv
    rowadr = int(0)
    if is_sparse:
      count = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]

    for index in range(count):
      dof = index
      if is_sparse:
        dof = efc_J_colind_in[worldid, 0, rowadr + index]
      elif efc_J_in[worldid, efcid, dof] == 0.0:
        continue
      tree = dof_treeid[dof]
      if tree < 0 or (sleep_enabled and tree_awake_in[worldid, tree] == 0):
        continue
      if first_tree < 0:
        first_tree = tree
        _dsu_union(island_parent_out, worldid, tree, -1, tree_island_out)
      else:
        _dsu_union(island_parent_out, worldid, first_tree, tree, tree_island_out)


@wp.kernel
def _compress_roots(
  # Data in:
  tree_island_in: wp.array2d[int],
  # Data out:
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Point every active tree straight at its component's minimum-tree root."""
  worldid, treeid = wp.tid()
  if tree_island_in[worldid, treeid] >= 0:
    island_parent_out[worldid, treeid] = _dsu_find(island_parent_out, worldid, treeid)


@wp.kernel
def _label_roots(
  # Model:
  ntree: int,
  # Data in:
  # In:
  island_parent_in: wp.array2d[int],
  # Data out:
  nisland_out: wp.array[int],
  tree_island_out: wp.array2d[int],
):
  """Number the roots in ascending tree order, which is MuJoCo's island order.

  Ranking roots is the one inherently sequential step: island i is the i-th smallest
  root, so it depends on every preceding tree. It reads two ints per tree and does no
  pointer chasing, leaving the compress and propagate passes free to run per tree.
  """
  worldid = wp.tid()
  nisland = int(0)
  for treeid in range(ntree):
    if tree_island_out[worldid, treeid] >= 0 and island_parent_in[worldid, treeid] == treeid:
      tree_island_out[worldid, treeid] = nisland
      nisland += 1
  nisland_out[worldid] = nisland


@wp.kernel
def _propagate_labels(
  # Data in:
  # In:
  island_parent_in: wp.array2d[int],
  # Data out:
  tree_island_out: wp.array2d[int],
):
  """Copy each root's label down to the trees that point at it.

  A root reads and writes its own slot with the same value, so the only concurrent
  writes to any slot store identical data.
  """
  worldid, treeid = wp.tid()
  if tree_island_out[worldid, treeid] >= 0:
    tree_island_out[worldid, treeid] = tree_island_out[worldid, island_parent_in[worldid, treeid]]


# Smallest EFC chunk worth its own block, and the resident-block count worth reaching for.
_DSU_MIN_CHUNK = 256
_DSU_TARGET_BLOCKS = 2048


@event_scope
def direct_dsu(m: types.Model, d: types.Data, island_parent: wp.array2d[int]):
  """Discover islands with EFC-parallel atomic minimum-root hooks.

  `island_parent` is the (nworld, ntree) disjoint-set workspace, owned by the caller.
  """
  # Discovery blocks are laid out over (world, chunk). A batch large enough to occupy the
  # device keeps one block per world, since extra chunks would mostly launch past the
  # active prefix; a small batch with a long prefix is split until there is resident work.
  max_chunks = max(1, -(-d.njmax // _DSU_MIN_CHUNK))
  nchunk = min(max(1, -(-_DSU_TARGET_BLOCKS // d.nworld)), max_chunks)
  chunk_size = -(-d.njmax // nchunk)
  wp.launch(
    _reset_dsu,
    dim=(d.nworld, m.ntree),
    inputs=[d.nisland, d.tree_island, island_parent],
  )
  if m.nflex > 0:
    has_passive = not ((m.opt.disableflags & types.DisableBit.SPRING) and (m.opt.disableflags & types.DisableBit.DAMPER))
    sleep_metric_flex = (
      bool(m.opt.enableflags & types.EnableBit.SLEEP) and m.opt.integrator == types.IntegratorType.DISCRETE and has_passive
    )
    wp.launch(
      _flex_island_dsu,
      dim=(d.nworld, m.nflex),
      inputs=[
        m.opt.enableflags,
        m.body_treeid,
        m.jnt_bodyid,
        m.flex_interp,
        m.flex_nodeadr,
        m.flex_nodenum,
        m.flex_vertadr,
        m.flex_vertnum,
        m.flex_nodebodyid,
        m.flex_vertbodyid,
        m.flex_rod_has_twist,
        m.flex_eff_possible,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        d.nefc,
        d.tree_awake,
        sleep_metric_flex,
      ],
      outputs=[d.tree_island, island_parent],
    )
    if m.opt.integrator == types.IntegratorType.DISCRETE and m.has_flex_passive:
      wp.launch(
        _passive_contact_island_dsu,
        dim=d.naconmax,
        inputs=[
          m.opt.enableflags,
          m.body_treeid,
          m.jnt_bodyid,
          m.geom_bodyid,
          m.flex_dim,
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
          d.nefc,
          d.flexvert_xpos,
          d.tree_awake,
          d.contact.pos,
          d.contact.geom,
          d.contact.flex,
          d.contact.elem,
          d.contact.vert,
          d.contact.worldid,
          d.contact.type,
          d.nacon,
          sleep_metric_flex,
        ],
        outputs=[d.tree_island, island_parent],
      )

  wp.launch_tiled(
    _island_dsu,
    dim=(d.nworld, nchunk),
    inputs=[
      m.nv,
      m.opt.enableflags,
      m.body_treeid,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.dof_treeid,
      m.geom_bodyid,
      m.site_bodyid,
      m.flex_dim,
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
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.is_sparse,
      m.flex_rod_has_twist,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      m.flex_vert_rodelem,
      d.nefc,
      d.flexvert_xpos,
      d.tree_awake,
      d.contact.pos,
      d.contact.geom,
      d.contact.flex,
      d.contact.elem,
      d.contact.vert,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.njmax,
      chunk_size,
      d.tree_island,
      island_parent,
    ],
    block_dim=types.BlockDim.island_dsu,
  )
  wp.launch(
    _compress_roots,
    dim=(d.nworld, m.ntree),
    inputs=[d.tree_island, island_parent],
  )
  wp.launch(
    _label_roots,
    dim=d.nworld,
    inputs=[m.ntree, island_parent, d.nisland, d.tree_island],
  )
  wp.launch(
    _propagate_labels,
    dim=(d.nworld, m.ntree),
    inputs=[island_parent, d.tree_island],
  )


@wp.kernel
def _zero_island_counts(
  # Data out:
  nisland_out: wp.array[int],
  nidof_out: wp.array[int],
):
  worldid = wp.tid()
  nisland_out[worldid] = 0
  nidof_out[worldid] = 0


@event_scope
def island(m: types.Model, d: types.Data):
  """Discover constraint islands."""
  if m.ntree == 0:
    wp.launch(
      _zero_island_counts,
      dim=d.nworld,
      outputs=[d.nisland, d.nidof],
    )
    return

  direct_dsu(m, d, wp.empty((d.nworld, m.ntree), dtype=int))


@wp.kernel
def _island_count_dofs(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  tree_island_in: wp.array2d[int],
  # Data out:
  dof_island_out: wp.array2d[int],
  island_nv_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()

  island_id = tree_island_in[worldid, dof_treeid[dofid]]
  dof_island_out[worldid, dofid] = island_id
  if island_id >= 0:
    wp.atomic_add(island_nv_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _island_scan_sizes(
  # Data in:
  nisland_in: wp.array[int],
  island_nv_in: wp.array2d[int],
  # Data out:
  nidof_out: wp.array[int],
  island_idofadr_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_iefcadr_out: wp.array2d[int],
):
  worldid = wp.tid()

  nisland = nisland_in[worldid]
  if nisland == 0:
    nidof_out[worldid] = 0
    return

  # Scan DOFs and Constraints
  island_idofadr_out[worldid, 0] = 0
  island_iefcadr_out[worldid, 0] = 0
  for i in range(1, nisland):
    island_idofadr_out[worldid, i] = island_idofadr_out[worldid, i - 1] + island_nv_in[worldid, i - 1]
    island_iefcadr_out[worldid, i] = island_iefcadr_out[worldid, i - 1] + island_nefc_out[worldid, i - 1]

  nidof = island_idofadr_out[worldid, nisland - 1] + island_nv_in[worldid, nisland - 1]
  nidof_out[worldid] = nidof

  # Reset for recount ( island_nv is final after _island_count_dofs and stays put )
  for i in range(nisland):
    island_nefc_out[worldid, i] = 0


@wp.kernel
def _island_map_dofs(
  # Model:
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nidof_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  island_idofadr_in: wp.array2d[int],
  # Data out:
  island_dofadr_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
):
  """Map DOFs in ascending global order for deterministic solver inputs.

  A DOF's rank inside its island is fixed by the tree layout: DOFs are contiguous per
  tree and trees are ordered, so the rank is the DOF count of every preceding tree
  carrying the same label. That closed form makes the ascending order reproducible
  without a serial per-world scan.
  """
  worldid, dofid = wp.tid()

  tree = dof_treeid[dofid]
  island_id = tree_island_in[worldid, tree]

  local_idx = dofid - tree_dofadr[tree]
  for t in range(tree):
    if tree_island_in[worldid, t] == island_id:
      local_idx += tree_dofnum[t]

  if island_id >= 0:
    idof = island_idofadr_in[worldid, island_id] + local_idx
    idof_islandid_out[worldid, idof] = island_id
    if local_idx == 0:
      island_dofadr_out[worldid, island_id] = dofid
  else:
    idof = nidof_in[worldid] + local_idx

  map_dof2idof_out[worldid, dofid] = idof
  map_idof2dof_out[worldid, idof] = dofid


@wp.kernel
def _island_count_constraints(
  # Data in:
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  tree_island_in: wp.array2d[int],
  njmax_in: int,
  # In:
  efc_tree_in: wp.array2d[int],
  # Data out:
  efc_island_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_ne_out: wp.array2d[int],
  island_nf_out: wp.array2d[int],
):
  worldid, efcid = wp.tid()
  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  efc_tree = efc_tree_in[worldid, efcid]
  if efc_tree < 0:
    efc_island_out[worldid, efcid] = -1
    return
  island_id = tree_island_in[worldid, efc_tree]
  efc_island_out[worldid, efcid] = island_id

  if island_id >= 0:
    wp.atomic_add(island_nefc_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]

    efc_type = efc_type_in[worldid, efcid]
    if efc_type == ConstraintType.EQUALITY:
      wp.atomic_add(island_ne_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]
    elif efc_type == ConstraintType.FRICTION_DOF or efc_type == ConstraintType.FRICTION_TENDON:
      wp.atomic_add(island_nf_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _island_map_constraints(
  # Data in:
  nefc_in: wp.array[int],
  nisland_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_island_in: wp.array2d[int],
  island_ne_in: wp.array2d[int],
  island_nf_in: wp.array2d[int],
  island_iefcadr_in: wp.array2d[int],
  # In:
  njmax_in: int,
  # Data out:
  island_nefc_out: wp.array2d[int],
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  iefc_islandid_out: wp.array2d[int],
):
  """Map constraints in ascending EFC order within each MuJoCo category.

  One thread owns one island and keeps its three category counters in registers, so the
  ascending scan stays serial per island (and therefore reproducible) while different
  islands and worlds advance in parallel.
  """
  worldid, islandid = wp.tid()
  if islandid >= nisland_in[worldid]:
    return

  iefcadr = island_iefcadr_in[worldid, islandid]
  fadr = iefcadr + island_ne_in[worldid, islandid]
  oadr = fadr + island_nf_in[worldid, islandid]
  ne_mapped = int(0)
  nf_mapped = int(0)
  nother_mapped = int(0)

  nefc = wp.min(njmax_in, nefc_in[worldid])
  for efcid in range(nefc):
    if efc_island_in[worldid, efcid] == islandid:
      efc_type = efc_type_in[worldid, efcid]

      # 1. Determine absolute index ic based on category
      if efc_type == ConstraintType.EQUALITY:
        ic = iefcadr + ne_mapped
        ne_mapped += 1
      elif efc_type == ConstraintType.FRICTION_DOF or efc_type == ConstraintType.FRICTION_TENDON:
        ic = fadr + nf_mapped
        nf_mapped += 1
      else:
        ic = oadr + nother_mapped
        nother_mapped += 1

      # 2. Store mappings
      map_efc2iefc_out[worldid, efcid] = ic
      map_iefc2efc_out[worldid, ic] = efcid
      iefc_islandid_out[worldid, ic] = islandid

  # 3. Reconstruct d.island_nefc from the three category counts
  island_nefc_out[worldid, islandid] = ne_mapped + nf_mapped + nother_mapped


@wp.kernel
def _compute_efc_tree(
  # Model:
  nv: int,
  body_treeid: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  dof_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  flex_dim: wp.array[int],
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
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  is_sparse: bool,
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  njmax_in: int,
  # Out:
  efc_tree_out: wp.array2d[int],
):
  """Compute the first non-negative tree for each constraint."""
  worldid, efcid = wp.tid()

  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  efc_type = efc_type_in[worldid, efcid]
  efc_id = efc_id_in[worldid, efcid]

  tree = int(-1)
  use_generic = int(0)

  # equality (connect/weld)
  if efc_type == ConstraintType.EQUALITY:
    eq_t = eq_type[efc_id]

    if eq_t == EqType.CONNECT or eq_t == EqType.WELD:
      b1 = eq_obj1id[efc_id]
      b2 = eq_obj2id[efc_id]

      if eq_objtype[efc_id] == ObjType.SITE:
        b1 = site_bodyid[b1]
        b2 = site_bodyid[b2]

      t1 = body_treeid[b1]
      t2 = body_treeid[b2]
      if t1 >= 0:
        tree = t1
      else:
        tree = t2
    else:
      # JOINT, TENDON, FLEX: generic scan
      use_generic = 1

  # joint friction
  elif efc_type == ConstraintType.FRICTION_DOF:
    tree = dof_treeid[efc_id]

  # joint limit
  elif efc_type == ConstraintType.LIMIT_JOINT:
    tree = dof_treeid[jnt_dofadr[efc_id]]

  # contact
  elif (
    efc_type == ConstraintType.CONTACT_FRICTIONLESS
    or efc_type == ConstraintType.CONTACT_PYRAMIDAL
    or efc_type == ConstraintType.CONTACT_ELLIPTIC
  ):
    geom_pair = contact_geom_in[efc_id]
    g1 = geom_pair[0]
    g2 = geom_pair[1]

    if g1 >= 0 and g2 >= 0:
      t1 = body_treeid[geom_bodyid[g1]]
      t2 = body_treeid[geom_bodyid[g2]]
      if t1 >= 0:
        tree = t1
      else:
        tree = t2
    else:
      # flex: first tree of the bodies of both sides
      bodies0, bodies1 = _flex_contact_bodies(
        jnt_bodyid,
        geom_bodyid,
        flex_dim,
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
        contact_pos_in,
        contact_geom_in,
        contact_flex_in,
        contact_elem_in,
        contact_vert_in,
        worldid,
        efc_id,
      )
      for k in range(8):
        body = bodies0[k % 4]
        if k >= 4:
          body = bodies1[k % 4]
        if body >= 0:
          t = body_treeid[body]
          if t >= 0:
            tree = t
            break

  else:
    # generic: scan Jacobian row
    use_generic = 1

  if use_generic:
    count = nv
    rowadr = 0
    if is_sparse:
      count = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]

    for i in range(count):
      dof = i
      if is_sparse:
        sparseid = rowadr + i
        dof = efc_J_colind_in[worldid, 0, sparseid]
      else:
        J_val = efc_J_in[worldid, efcid, dof]
        if J_val == 0.0:
          continue

      t = dof_treeid[dof]
      if t >= 0:
        tree = t
        break

  efc_tree_out[worldid, efcid] = tree


@wp.kernel
def _init_island_arrays(
  # Data out:
  nidof_out: wp.array[int],
  island_idofadr_out: wp.array2d[int],
  island_nv_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_ne_out: wp.array2d[int],
  island_nf_out: wp.array2d[int],
  island_iefcadr_out: wp.array2d[int],
):
  worldid, islandid = wp.tid()
  island_nv_out[worldid, islandid] = 0
  island_nefc_out[worldid, islandid] = 0
  island_ne_out[worldid, islandid] = 0
  island_nf_out[worldid, islandid] = 0
  island_idofadr_out[worldid, islandid] = 0
  island_iefcadr_out[worldid, islandid] = 0
  if islandid == 0:
    nidof_out[worldid] = 0


@wp.kernel
def _init_dof_arrays(
  # Data out:
  dof_island_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()
  dof_island_out[worldid, dofid] = -1
  map_dof2idof_out[worldid, dofid] = 0
  map_idof2dof_out[worldid, dofid] = 0
  idof_islandid_out[worldid, dofid] = -1


@wp.kernel
def _init_efc_arrays(
  # Data out:
  efc_island_out: wp.array2d[int],
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  iefc_islandid_out: wp.array2d[int],
  efc_tree_out: wp.array2d[int],
):
  worldid, efcid = wp.tid()
  efc_island_out[worldid, efcid] = -1
  map_efc2iefc_out[worldid, efcid] = 0
  map_iefc2efc_out[worldid, efcid] = 0
  iefc_islandid_out[worldid, efcid] = -1
  efc_tree_out[worldid, efcid] = -1


@event_scope
def compute_island_mapping(m: types.Model, d: types.Data):
  """Compute DOF/constraint island mappings after island discovery.

  Populates d.dof_island, d.efc.island, d.island_idofadr, d.island_dofadr,
  d.island_nv, d.island_nefc, d.island_ne, d.island_nf, d.island_iefcadr,
  d.nidof, d.map_dof2idof, d.map_idof2dof, d.dof_islandid, d.map_efc2iefc,
  d.map_iefc2efc, d.efc_islandid.

  Args:
    m: Model.
    d: Data.
  """
  # Ensure dof_islandid / efc_islandid are allocated at the right shape
  if d.dof_islandid.shape[1] != m.nv:
    d.dof_islandid = wp.empty((d.nworld, m.nv), dtype=int)
  if d.efc_islandid.shape[1] != d.njmax:
    d.efc_islandid = wp.empty((d.nworld, d.njmax), dtype=int)
  if d.island_idofadr.shape[1] != m.ntree:
    d.island_idofadr = wp.empty((d.nworld, m.ntree), dtype=int)

  wp.launch(
    _init_island_arrays,
    dim=(d.nworld, m.ntree),
    inputs=[],
    outputs=[d.nidof, d.island_idofadr, d.island_nv, d.island_nefc, d.island_ne, d.island_nf, d.island_iefcadr],
  )
  wp.launch(
    _init_dof_arrays,
    dim=(d.nworld, m.nv),
    inputs=[],
    outputs=[d.dof_island, d.map_dof2idof, d.map_idof2dof, d.dof_islandid],
  )
  efc_tree = wp.empty((d.nworld, d.njmax), dtype=int)
  wp.launch(
    _init_efc_arrays,
    dim=(d.nworld, d.njmax),
    inputs=[],
    outputs=[d.efc.island, d.map_efc2iefc, d.map_iefc2efc, d.efc_islandid, efc_tree],
  )

  wp.launch(
    _compute_efc_tree,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.nv,
      m.body_treeid,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.dof_treeid,
      m.geom_bodyid,
      m.site_bodyid,
      m.flex_dim,
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
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.is_sparse,
      m.flex_rod_has_twist,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      m.flex_vert_rodelem,
      d.nefc,
      d.flexvert_xpos,
      d.contact.pos,
      d.contact.geom,
      d.contact.flex,
      d.contact.elem,
      d.contact.vert,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.njmax,
    ],
    outputs=[efc_tree],
  )

  # 1. Count DOFs per island
  wp.launch(
    _island_count_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, d.tree_island],
    outputs=[d.dof_island, d.island_nv],
  )

  # 2. Count Constraints per island
  wp.launch(
    _island_count_constraints,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.type, d.tree_island, d.njmax, efc_tree],
    outputs=[d.efc.island, d.island_nefc, d.island_ne, d.island_nf],
  )

  # 3. Scan sizes and reset counters for mapping
  wp.launch(
    _island_scan_sizes,
    dim=d.nworld,
    inputs=[d.nisland, d.island_nv],
    outputs=[d.nidof, d.island_idofadr, d.island_nefc, d.island_iefcadr],
  )

  # 4. Map DOFs
  d.island_dofadr.fill_(m.nv)
  wp.launch(
    _island_map_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, m.tree_dofadr, m.tree_dofnum, d.nidof, d.tree_island, d.island_idofadr],
    outputs=[d.island_dofadr, d.map_dof2idof, d.map_idof2dof, d.dof_islandid],
  )

  # 5. Map Constraints
  wp.launch(
    _island_map_constraints,
    dim=(d.nworld, m.ntree),
    inputs=[d.nefc, d.nisland, d.efc.type, d.efc.island, d.island_ne, d.island_nf, d.island_iefcadr, d.njmax],
    outputs=[d.island_nefc, d.map_efc2iefc, d.map_iefc2efc, d.efc_islandid],
  )


# Active-DOF compaction (nvmax < nv).
#
# The active set is tracked per kinematic tree (the unit of coupling in the mass matrix)
# via the same tree_awake bookkeeping used by the sleep solver. Each step the active trees'
# DOFs are packed into a contiguous [0, ncdof) range so the dense factor/solve can run at
# size nvmax instead of nv. This is the active-set analog of compute_island_mapping.


@wp.kernel
def _reset_compact_maps(
  # Model:
  nv: int,
  # Data in:
  nvmax_pad_in: int,
  # Data out:
  dof_cdof_out: wp.array2d[int],
  cdof_dof_out: wp.array2d[int],
):
  worldid, idx = wp.tid()
  if idx < nv:
    dof_cdof_out[worldid, idx] = -1
  # cdof_dof is nvmax_pad-wide: clear the whole row so the padded tail [ncdof, nvmax_pad)
  # reads as -1 (the gather/solve run over nvmax_pad, not just the active ncdof).
  if idx < nvmax_pad_in:
    cdof_dof_out[worldid, idx] = -1


@cache_kernel
def _compact_dofs_builder(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def _compact_dofs(
    # Model:
    ntree: int,
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    # Data in:
    tree_awake_in: wp.array2d[int],
    nvmax_in: int,
    # Data out:
    ncdof_out: wp.array[int],
    dof_cdof_out: wp.array2d[int],
    cdof_dof_out: wp.array2d[int],
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()
    count = int(0)
    for t in range(ntree):
      if tree_awake_in[worldid, t] == 1:
        adr = tree_dofadr[t]
        num = tree_dofnum[t]
        for j in range(num):
          dof = adr + j
          if count < nvmax_in:
            dof_cdof_out[worldid, dof] = count
            cdof_dof_out[worldid, count] = dof
          count += 1

    if count > nvmax_in:
      if wp.static(bool(warn_overflow & OverflowType.NVMAX)):
        wp.printf(
          "nvmax overflow: world %d needs %d active DOFs but nvmax = %d (behavior undefined)\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NVMAX (or = 0 for all)\n",
          worldid,
          count,
          nvmax_in,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.NVMAX
      ncdof_out[worldid] = nvmax_in
    else:
      ncdof_out[worldid] = count

  return _compact_dofs


@event_scope
def update_active_dofs(m: types.Model, d: types.Data):
  """Rebuild the compaction maps (dof_cdof / cdof_dof) from tree_awake."""
  wp.launch(
    _reset_compact_maps,
    dim=(d.nworld, max(m.nv, d.nvmax_pad)),
    inputs=[m.nv, d.nvmax_pad],
    outputs=[d.dof_cdof, d.cdof_dof],
  )
  wp.launch(
    _compact_dofs_builder(int(m.opt.warn_overflow)),
    dim=(d.nworld,),
    inputs=[m.ntree, m.tree_dofadr, m.tree_dofnum, d.tree_awake, d.nvmax],
    outputs=[d.ncdof, d.dof_cdof, d.cdof_dof, d.overflow],
  )
