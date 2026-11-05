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

"""Core collision types and utilities shared across collision modules."""

import dataclasses
from typing import Any, Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.math import safe_div
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINMU
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import ContactType
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import mat63
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class Geom:
  """Geom properties for pairwise collision detection.

  Bundles a geometry's pose, size, surface normal, and mesh topology data into
  a single struct that can be passed to Warp collision kernels.
  """

  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  margin: float
  polyvert: mat63
  vertadr: int
  vertnum: int
  vert: wp.array[wp.vec3]
  graphadr: int
  graph: wp.array[int]
  mesh_polynum: int
  mesh_polyadr: int
  mesh_polynormal: wp.array[wp.vec3]
  mesh_polyvertadr: wp.array[int]
  mesh_polyvertnum: wp.array[int]
  mesh_polyvert: wp.array[int]
  mesh_polymapadr: wp.array[int]
  mesh_polymapnum: wp.array[int]
  mesh_polymap: wp.array[int]
  index: int


@wp.func
def geom_collision_pair_from_types(
  # Model:
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  geom_type1: int,
  geom_type2: int,
  geoms: wp.vec2i,
  worldid: int,
) -> Tuple[Geom, Geom]:
  geom1 = Geom()
  geom2 = Geom()

  g1 = geoms[0]
  g2 = geoms[1]

  geom1.pos = geom_xpos_in[worldid, g1]
  geom1.rot = geom_xmat_in[worldid, g1]
  geom1.size = geom_size[worldid % geom_size.shape[0], g1]
  # z-axis of the rotation matrix, used as the surface normal for plane collisions
  geom1.normal = wp.vec3(geom1.rot[0, 2], geom1.rot[1, 2], geom1.rot[2, 2])

  geom2.pos = geom_xpos_in[worldid, g2]
  geom2.rot = geom_xmat_in[worldid, g2]
  geom2.size = geom_size[worldid % geom_size.shape[0], g2]
  # z-axis of the rotation matrix, used as the surface normal for plane collisions
  geom2.normal = wp.vec3(geom2.rot[0, 2], geom2.rot[1, 2], geom2.rot[2, 2])

  dataid_setid = worldid % geom_dataid.shape[0]

  if geom_type1 == GeomType.MESH:
    dataid = geom_dataid[dataid_setid, g1]
    geom1.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)
    geom1.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)
    geom1.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)
    geom1.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)
    geom1.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)

    geom1.vert = mesh_vert
    geom1.graph = mesh_graph
    geom1.mesh_polynormal = mesh_polynormal
    geom1.mesh_polyvertadr = mesh_polyvertadr
    geom1.mesh_polyvertnum = mesh_polyvertnum
    geom1.mesh_polyvert = mesh_polyvert
    geom1.mesh_polymapadr = mesh_polymapadr
    geom1.mesh_polymapnum = mesh_polymapnum
    geom1.mesh_polymap = mesh_polymap

  if geom_type2 == GeomType.MESH:
    dataid = geom_dataid[dataid_setid, g2]
    geom2.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)
    geom2.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)
    geom2.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)
    geom2.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)
    geom2.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)

    geom2.vert = mesh_vert
    geom2.graph = mesh_graph
    geom2.mesh_polynormal = mesh_polynormal
    geom2.mesh_polyvertadr = mesh_polyvertadr
    geom2.mesh_polyvertnum = mesh_polyvertnum
    geom2.mesh_polyvert = mesh_polyvert
    geom2.mesh_polymapadr = mesh_polymapadr
    geom2.mesh_polymapnum = mesh_polymapnum
    geom2.mesh_polymap = mesh_polymap

  geom1.index = -1
  geom1.margin = 0.0

  geom2.index = -1
  geom2.margin = 0.0

  return geom1, geom2


@wp.func
def geom_collision_pair(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  geoms: wp.vec2i,
  worldid: int,
) -> Tuple[Geom, Geom]:
  geom_type1 = geom_type[geoms[0]]
  geom_type2 = geom_type[geoms[1]]
  return geom_collision_pair_from_types(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    mesh_polynum,
    mesh_polyadr,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    mesh_polymapadr,
    mesh_polymapnum,
    mesh_polymap,
    geom_xpos_in,
    geom_xmat_in,
    geom_type1,
    geom_type2,
    geoms,
    worldid,
  )


@wp.func
def write_contact(
  # Data in:
  naconmax_in: int,
  # In:
  id_: int,
  dist_in: float,
  pos_in: wp.vec3,
  frame_in: wp.mat33,
  margin_in: float,
  gap_in: float,
  condim_in: int,
  friction_in: vec5,
  solref_in: wp.vec2,
  solreffriction_in: wp.vec2,
  solimp_in: vec5,
  adhesion_in: float,
  geoms_in: wp.vec2i,
  pairid_in: wp.vec2i,
  worldid_in: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_efc_address_out: wp.array2d[int],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  contact_adhesion_out: wp.array[float],
  nacon_out: wp.array[int],
) -> int:
  """Atomically write a detected contact into the contact output arrays.

  Returns 1 if the contact is active (dist < margin or in-gap adhesive), 0 otherwise.
  """
  active = (dist_in < margin_in) or ((adhesion_in != 0.0) and (dist_in < margin_in + gap_in))
  detected = dist_in < margin_in + gap_in
  condim = wp.where((adhesion_in != 0.0) and (dist_in >= margin_in), 1, condim_in)

  # skip contact and no collision sensor
  if (pairid_in[0] == -2 or not detected) and pairid_in[1] == -1:
    return 0

  contact_type = 0

  if pairid_in[0] >= -1 and detected:
    contact_type |= ContactType.CONSTRAINT

  if pairid_in[1] >= 0:
    contact_type |= ContactType.SENSOR

  cid = wp.atomic_add(nacon_out, 0, 1)  # kernel_analyzer: ignore[atomic]
  if cid < naconmax_in:
    contact_dist_out[cid] = dist_in
    contact_pos_out[cid] = pos_in
    contact_frame_out[cid] = frame_in
    contact_geom_out[cid] = geoms_in
    contact_worldid_out[cid] = worldid_in
    includemargin = margin_in
    contact_includemargin_out[cid] = includemargin
    contact_dim_out[cid] = condim
    contact_friction_out[cid] = friction_in
    contact_solref_out[cid] = solref_in
    contact_solreffriction_out[cid] = solreffriction_in
    contact_solimp_out[cid] = solimp_in
    contact_adhesion_out[cid] = adhesion_in
    contact_type_out[cid] = contact_type
    contact_geomcollisionid_out[cid] = id_
    for i in range(contact_efc_address_out.shape[1]):
      contact_efc_address_out[cid, i] = -1
    return int(active)
  return 0


@wp.func
def contact_margin_gap(
  # Model:
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  pair_margin: wp.array2d[float],
  pair_gap: wp.array2d[float],
  # In:
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
) -> Tuple[float, float]:
  if pairid > -1:
    margin = pair_margin[worldid % pair_margin.shape[0], pairid]
    gap = pair_gap[worldid % pair_gap.shape[0], pairid]
  else:
    g1 = geoms[0]
    g2 = geoms[1]
    margin_id = worldid % geom_margin.shape[0]
    gap_id = worldid % geom_gap.shape[0]
    margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]
    gap = geom_gap[gap_id, g1] + geom_gap[gap_id, g2]

  return margin, gap


@wp.func
def contact_material_params(
  # Model:
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_friction: wp.array2d[wp.vec3],
  geom_adhesion: wp.array2d[float],
  pair_dim: wp.array[int],
  pair_solref: wp.array2d[wp.vec2],
  pair_solreffriction: wp.array2d[wp.vec2],
  pair_solimp: wp.array2d[vec5],
  pair_adhesion: wp.array2d[float],
  pair_friction: wp.array2d[vec5],
  # In:
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
):
  if pairid > -1:
    condim = pair_dim[pairid]
    friction = pair_friction[worldid % pair_friction.shape[0], pairid]
    solref = pair_solref[worldid % pair_solref.shape[0], pairid]
    solreffriction = pair_solreffriction[worldid % pair_solreffriction.shape[0], pairid]
    solimp = pair_solimp[worldid % pair_solimp.shape[0], pairid]
    adhesion = pair_adhesion[worldid % pair_adhesion.shape[0], pairid]
  else:
    g1 = geoms[0]
    g2 = geoms[1]
    solmix_id = worldid % geom_solmix.shape[0]
    friction_id = worldid % geom_friction.shape[0]
    solref_id = worldid % geom_solref.shape[0]
    solimp_id = worldid % geom_solimp.shape[0]
    adhesion_id = worldid % geom_adhesion.shape[0]

    solmix1 = geom_solmix[solmix_id, g1]
    solmix2 = geom_solmix[solmix_id, g2]

    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]

    # priority
    p1 = geom_priority[g1]
    p2 = geom_priority[g2]
    a1 = geom_adhesion[adhesion_id, g1]
    a2 = geom_adhesion[adhesion_id, g2]

    if p1 > p2:
      mix = 1.0
      condim = condim1
      max_geom_friction = geom_friction[friction_id, g1]
      adhesion = a1
    elif p2 > p1:
      mix = 0.0
      condim = condim2
      max_geom_friction = geom_friction[friction_id, g2]
      adhesion = a2
    else:
      mix = safe_div(solmix1, solmix1 + solmix2)
      mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
      mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
      mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
      condim = wp.max(condim1, condim2)
      max_geom_friction = wp.max(geom_friction[friction_id, g1], geom_friction[friction_id, g2])
      adhesion = a1 + a2

    friction = vec5(
      max_geom_friction[0],
      max_geom_friction[0],
      max_geom_friction[1],
      max_geom_friction[2],
      max_geom_friction[2],
    )

    if geom_solref[solref_id, g1][0] > 0.0 and geom_solref[solref_id, g2][0] > 0.0:
      solref = mix * geom_solref[solref_id, g1] + (1.0 - mix) * geom_solref[solref_id, g2]
    else:
      solref = wp.min(geom_solref[solref_id, g1], geom_solref[solref_id, g2])

    solreffriction = wp.vec2(0.0, 0.0)
    solimp = mix * geom_solimp[solimp_id, g1] + (1.0 - mix) * geom_solimp[solimp_id, g2]

  friction = vec5(
    wp.max(MJ_MINMU, friction[0]),
    wp.max(MJ_MINMU, friction[1]),
    wp.max(MJ_MINMU, friction[2]),
    wp.max(MJ_MINMU, friction[3]),
    wp.max(MJ_MINMU, friction[4]),
  )

  return condim, friction, solref, solreffriction, solimp, adhesion


# TODO(team): early return if collision sensor but no contact
@wp.func
def contact_params(
  # Model:
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  geom_adhesion: wp.array2d[float],
  pair_dim: wp.array[int],
  pair_solref: wp.array2d[wp.vec2],
  pair_solreffriction: wp.array2d[wp.vec2],
  pair_solimp: wp.array2d[vec5],
  pair_margin: wp.array2d[float],
  pair_gap: wp.array2d[float],
  pair_adhesion: wp.array2d[float],
  pair_friction: wp.array2d[vec5],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  cid: int,
  worldid: int,
):
  """Resolve contact parameters for a collision pair.

  Uses explicit pair overrides when available, otherwise mixes geom-level
  properties by priority and solmix weights.
  """
  geoms = collision_pair_in[cid]
  pairid = collision_pairid_in[cid][0]

  margin, gap = contact_margin_gap(geom_margin, geom_gap, pair_margin, pair_gap, geoms, pairid, worldid)
  condim, friction, solref, solreffriction, solimp, adhesion = contact_material_params(
    geom_condim,
    geom_priority,
    geom_solmix,
    geom_solref,
    geom_solimp,
    geom_friction,
    geom_adhesion,
    pair_dim,
    pair_solref,
    pair_solreffriction,
    pair_solimp,
    pair_adhesion,
    pair_friction,
    geoms,
    pairid,
    worldid,
  )

  return geoms, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion


@dataclasses.dataclass
class CollisionContext:
  """Collision driver intermediate arrays.

  Attributes:
    collision_pair: collision pairs from broadphase             (naconmax, 2)
    collision_pairid: ids from broadphase                       (naconmax, 2)
    collision_worldid: collision world ids from broadphase      (naconmax,)
  """

  collision_pair: wp.array
  collision_pairid: wp.array
  collision_worldid: wp.array


@wp.func
def sap_binary_search(values: wp.array[Any], value: Any, lower: int, upper: int) -> int:
  """Binary search for the first element > value in sorted array."""
  while lower < upper:
    mid = (lower + upper) >> 1
    if values[mid] > value:
      upper = mid
    else:
      lower = mid + 1
  return upper


@wp.kernel
def sap_range(
  # In:
  n: int,
  lower_in: wp.array2d[float],
  upper_in: wp.array2d[float],
  sort_index_in: wp.array2d[int],
  # Out:
  range_out: wp.array2d[int],
):
  """Compute the sweep range for each sorted element."""
  worldid, sortedid = wp.tid()

  idx = sort_index_in[worldid, sortedid]
  upper = upper_in[worldid, idx]

  limit = sap_binary_search(lower_in[worldid], upper, sortedid + 1, n)
  limit = wp.min(n - 1, limit)

  range_out[worldid, sortedid] = limit - sortedid


def create_collision_context(naconmax: int) -> CollisionContext:
  """Create a CollisionContext with allocated arrays."""
  return CollisionContext(
    collision_pair=wp.empty(naconmax, dtype=wp.vec2i),
    collision_pairid=wp.empty(naconmax, dtype=wp.vec2i),
    collision_worldid=wp.empty(naconmax, dtype=int),
  )
