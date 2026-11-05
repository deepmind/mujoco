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

from mujoco.mjx.third_party.mujoco_warp._src.collision_core import CollisionContext
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import Geom
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import contact_margin_gap
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import contact_material_params
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import contact_params
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import geom_collision_pair
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import geom_collision_pair_from_types
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import write_contact
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import _gjk_simplex_to_epa
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import ccd
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import epa_phase
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import gjk_phase
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import multicontact
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import support
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive import Geom
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive import contact_params
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive import geom_collision_pair
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive import write_contact
from mujoco.mjx.third_party.mujoco_warp._src.math import make_frame
from mujoco.mjx.third_party.mujoco_warp._src.math import upper_trid_index
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAX_EPAFACES
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAX_EPAHORIZON
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.third_party.mujoco_warp._src.types import mat43
from mujoco.mjx.third_party.mujoco_warp._src.types import mat63
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

vec_maxconpair = wp.types.vector(length=MJ_MAXCONPAIR, dtype=float)
mat_maxconpair = wp.types.matrix(shape=(MJ_MAXCONPAIR, 3), dtype=float)


@wp.func
def _hfield_filter(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rbound: wp.array2d[float],
  geom_margin: wp.array2d[float],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  hfield_size: wp.array[wp.vec4],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  g1: int,
  g2: int,
) -> Tuple[bool, float, float, float, float, float, float]:
  """Filter for height field collisions.

  See MuJoCo mjc_ConvexHField.
  """
  # height field info
  dataid_setid = worldid % geom_dataid.shape[0]
  hfdataid = geom_dataid[dataid_setid, g1]
  size1 = hfield_size[hfdataid]

  # geom info
  rbound_id = worldid % geom_rbound.shape[0]
  margin_id = worldid % geom_margin.shape[0]
  size_id = worldid % geom_size.shape[0]

  pos1 = geom_xpos_in[worldid, g1]
  mat1 = geom_xmat_in[worldid, g1]
  mat1T = wp.transpose(mat1)
  pos2 = geom_xpos_in[worldid, g2]
  pos = mat1T @ (pos2 - pos1)
  r2 = geom_rbound[rbound_id, g2]

  # TODO(team): margin?
  margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]

  # box-sphere test: horizontal plane
  for i in range(2):
    if (size1[i] < pos[i] - r2 - margin) or (-size1[i] > pos[i] + r2 + margin):
      return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL

  # box-sphere test: vertical direction
  if size1[2] < pos[2] - r2 - margin:  # up
    return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL

  if -size1[3] > pos[2] + r2 + margin:  # down
    return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL

  mat2 = geom_xmat_in[worldid, g2]
  mat = mat1T @ mat2

  # create geom in height field frame for support function queries
  geom2 = Geom()
  geom2.pos = pos
  geom2.rot = mat
  geom2.size = geom_size[size_id, g2]
  geom2.margin = 0.0  # margin handled separately
  geom2.index = -1

  geomtype2 = geom_type[g2]

  # load mesh vertex data for support function queries
  if geomtype2 == GeomType.MESH:
    dataid = geom_dataid[dataid_setid, g2]
    geom2.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)
    geom2.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)
    geom2.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)
    geom2.vert = mesh_vert
    geom2.graph = mesh_graph

  # use support functions for tight AABB bounds
  xmax = support(geom2, geomtype2, wp.vec3(1.0, 0.0, 0.0)).point[0]
  xmin = support(geom2, geomtype2, wp.vec3(-1.0, 0.0, 0.0)).point[0]
  ymax = support(geom2, geomtype2, wp.vec3(0.0, 1.0, 0.0)).point[1]
  ymin = support(geom2, geomtype2, wp.vec3(0.0, -1.0, 0.0)).point[1]
  zmax = support(geom2, geomtype2, wp.vec3(0.0, 0.0, 1.0)).point[2]
  zmin = support(geom2, geomtype2, wp.vec3(0.0, 0.0, -1.0)).point[2]

  # box-box test
  if (
    (xmin - margin > size1[0])
    or (xmax + margin < -size1[0])
    or (ymin - margin > size1[1])
    or (ymax + margin < -size1[1])
    or (zmin - margin > size1[2])
    or (zmax + margin < -size1[3])
  ):
    return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL
  else:
    return False, xmin, xmax, ymin, ymax, zmin, zmax


@cache_kernel
def ccd_hfield_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  geomgeomid: int,
  warn_overflow: int,
):
  """Kernel builder for heightfield CCD collisions (no multiccd args)."""

  # runs convex collision on a set of geom pairs to recover contact info
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def ccd_hfield_kernel(
    # Model:
    opt_ccd_tolerance: wp.array[float],
    geom_type: wp.array[int],
    geom_condim: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_size: wp.array2d[wp.vec3],
    geom_rbound: wp.array2d[float],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    geom_adhesion: wp.array2d[float],
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
    hfield_size: wp.array[wp.vec4],
    hfield_nrow: wp.array[int],
    hfield_ncol: wp.array[int],
    hfield_adr: wp.array[int],
    hfield_data: wp.array[float],
    pair_dim: wp.array[int],
    pair_solref: wp.array2d[wp.vec2],
    pair_solreffriction: wp.array2d[wp.vec2],
    pair_solimp: wp.array2d[vec5],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    pair_adhesion: wp.array2d[float],
    pair_friction: wp.array2d[vec5],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    naconmax_in: int,
    naccdmax_in: int,
    ncollision_in: wp.array[int],
    # In:
    collision_pair_in: wp.array[wp.vec2i],
    collision_pairid_in: wp.array[wp.vec2i],
    collision_worldid_in: wp.array[int],
    epa_vert_in: wp.array2d[wp.vec3],
    epa_vert_index_in: wp.array2d[int],
    epa_face_in: wp.array2d[int],
    epa_pr_in: wp.array2d[wp.vec3],
    epa_norm2_in: wp.array2d[float],
    epa_horizon_in: wp.array2d[int],
    nccd_in: wp.array[int],
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
    # Data out:
    overflow_out: wp.array[int],
  ):
    collisionid = wp.tid()
    if collisionid >= ncollision_in[0]:
      return

    geoms = collision_pair_in[collisionid]
    g1 = geoms[0]
    g2 = geoms[1]

    if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
      return

    worldid = collision_worldid_in[collisionid]

    # height field filter
    no_hf_collision, xmin, xmax, ymin, ymax, zmin, zmax = _hfield_filter(
      geom_type,
      geom_dataid,
      geom_size,
      geom_rbound,
      geom_margin,
      mesh_vertadr,
      mesh_vertnum,
      mesh_graphadr,
      mesh_vert,
      mesh_graph,
      hfield_size,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      g1,
      g2,
    )
    if no_hf_collision:
      return

    ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)  # kernel_analyzer: ignore[atomic]
    if ccdid >= naccdmax_in:
      if wp.static(bool(warn_overflow & OverflowType.CCD)):
        wp.printf(
          "CCD overflow - please increase naccdmax beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
          naccdmax_in,
        )
      wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.CCD))
      return

    _, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = contact_params(
      geom_condim,
      geom_priority,
      geom_solmix,
      geom_solref,
      geom_solimp,
      geom_friction,
      geom_margin,
      geom_gap,
      geom_adhesion,
      pair_dim,
      pair_solref,
      pair_solreffriction,
      pair_solimp,
      pair_margin,
      pair_gap,
      pair_adhesion,
      pair_friction,
      collision_pair_in,
      collision_pairid_in,
      collisionid,
      worldid,
    )

    geom1, geom2 = geom_collision_pair(
      geom_type,
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
      geoms,
      worldid,
    )

    # transform geom2 into heightfield frame
    hf_pos = geom_xpos_in[worldid, g1]
    hf_mat = geom_xmat_in[worldid, g1]
    hf_matT = wp.transpose(hf_mat)

    geom2.pos = hf_matT @ (geom2.pos - hf_pos)
    geom2.rot = hf_matT @ geom2.rot

    # geom1 has identity pose
    geom1.pos = wp.vec3(0.0, 0.0, 0.0)
    geom1.rot = wp.identity(n=3, dtype=float)

    # see MuJoCo mjc_ConvexHField
    geom1_dataid = geom_dataid[worldid % geom_dataid.shape[0], g1]

    # height field subgrid
    nrow = hfield_nrow[geom1_dataid]
    ncol = hfield_ncol[geom1_dataid]
    size = hfield_size[geom1_dataid]

    # subgrid
    x_scale = 0.5 * float(ncol - 1) / size[0]
    y_scale = 0.5 * float(nrow - 1) / size[1]
    cmin = wp.max(0, int(wp.floor((xmin + size[0]) * x_scale)))
    cmax = wp.min(ncol - 1, int(wp.ceil((xmax + size[0]) * x_scale)))
    rmin = wp.max(0, int(wp.floor((ymin + size[1]) * y_scale)))
    rmax = wp.min(nrow - 1, int(wp.ceil((ymax + size[1]) * y_scale)))

    dx = (2.0 * size[0]) / float(ncol - 1)
    dy = (2.0 * size[1]) / float(nrow - 1)
    dr = wp.vec2i(1, 0)

    prism = mat63()

    # set zbottom value using base size
    prism[0, 2] = -size[3]
    prism[1, 2] = -size[3]
    prism[2, 2] = -size[3]

    adr = hfield_adr[geom1_dataid]

    hfield_contact_dist = vec_maxconpair()
    hfield_contact_pos = mat_maxconpair()
    hfield_contact_normal = mat_maxconpair()
    min_dist = float(MJ_MAXVAL)
    min_normal = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
    min_pos = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
    min_id = int(-1)

    # geom1 margin added to z-axis of hfield prism top points below
    geom2.margin = margin

    # EPA memory
    epa_vert = epa_vert_in[ccdid]
    epa_vert_index = epa_vert_index_in[ccdid]
    epa_face = epa_face_in[ccdid]
    epa_pr = epa_pr_in[ccdid]
    epa_norm2 = epa_norm2_in[ccdid]
    epa_horizon = epa_horizon_in[ccdid]

    collision_pairid = collision_pairid_in[collisionid]

    # process all prisms in subgrid
    count = int(0)
    noverflow = int(0)
    for r in range(rmin, rmax):
      if noverflow > 0:
        break

      # pre-initialize first 2 vertices
      for init_i in range(2):
        x = dx * float(cmin) - size[0]
        y = dy * float(r + dr[init_i]) - size[1]
        z = hfield_data[adr + (r + dr[init_i]) * ncol + cmin] * size[2] + margin

        prism[0] = prism[1]
        prism[1] = prism[2]
        prism[3] = prism[4]
        prism[4] = prism[5]

        prism[2, 0] = x
        prism[5, 0] = x
        prism[2, 1] = y
        prism[5, 1] = y
        prism[5, 2] = z

      for c in range(cmin + 1, cmax + 1):
        if noverflow > 0:
          break

        # add both triangles from this cell
        for i in range(2):
          if count >= MJ_MAXCONPAIR:
            noverflow = 1
            break

          # add vert
          x = dx * float(c) - size[0]
          y = dy * float(r + dr[i]) - size[1]
          z = hfield_data[adr + (r + dr[i]) * ncol + c] * size[2] + margin

          prism[0] = prism[1]
          prism[1] = prism[2]
          prism[3] = prism[4]
          prism[4] = prism[5]

          prism[2, 0] = x
          prism[5, 0] = x
          prism[2, 1] = y
          prism[5, 1] = y
          prism[5, 2] = z

          # prism height test
          if prism[3, 2] < zmin and prism[4, 2] < zmin and prism[5, 2] < zmin:
            continue

          geom1.polyvert = prism

          # prism center
          x1 = geom1.pos + geom1.rot @ (prism[0] + prism[1] + prism[2] + prism[3] + prism[4] + prism[5]) * wp.static(1.0 / 6.0)

          dist, ncontact, w1, w2, idx = ccd(
            opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],
            0.0,
            gjk_iterations,
            epa_iterations,
            geom1,
            geom2,
            geomtype1,
            geomtype2,
            x1,
            geom2.pos,
            epa_vert,
            epa_vert_index,
            epa_face,
            epa_pr,
            epa_norm2,
            epa_horizon,
            wp.static(bool(warn_overflow & OverflowType.EPA_HORIZON)),
            worldid,
            overflow_out,
          )

          # skip prisms not in contact: ccd reports ncontact == 1 for separated geoms
          if ncontact == 0 or dist >= margin + gap:
            continue

          # cache contact information
          hfield_contact_dist[count] = dist

          # transform contact to global frame
          pos_local = 0.5 * (w1 + w2)
          pos = hf_mat @ pos_local + hf_pos
          hfield_contact_pos[count, 0] = pos[0]
          hfield_contact_pos[count, 1] = pos[1]
          hfield_contact_pos[count, 2] = pos[2]

          frame_local = make_frame(w1 - w2)
          normal_local = wp.vec3(frame_local[0, 0], frame_local[0, 1], frame_local[0, 2])
          normal = hf_mat @ normal_local
          hfield_contact_normal[count, 0] = normal[0]
          hfield_contact_normal[count, 1] = normal[1]
          hfield_contact_normal[count, 2] = normal[2]

          # contact with minimum distance
          if dist < min_dist:
            min_dist = dist
            min_normal = normal
            min_pos = pos
            min_id = count

          count += 1

    # warn once per pair rather than once per prism that did not fit
    if noverflow > 0:
      if wp.static(bool(warn_overflow & OverflowType.HFIELD)):
        wp.printf(
          "height field collision overflow, number of collisions >= %u - please adjust resolution: \n"
          "decrease the number of hfield rows/cols or modify size of colliding geom\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.HFIELD (or = 0 for all)\n",
          MJ_MAXCONPAIR,
        )
      wp.atomic_or(overflow_out, worldid, OverflowType.HFIELD)

    # skip write when no prisms collide
    if count == 0:
      return

    # contact 0: minimum distance
    write_contact(
      naconmax_in,
      0,
      min_dist,
      min_pos,
      make_frame(min_normal),
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      adhesion,
      geoms,
      collision_pairid,
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_efc_address_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      contact_adhesion_out,
      nacon_out,
    )

    # TODO(team): routine for select subset of contacts
    if wp.static(True):
      MIN_DIST_TO_NEXT_CONTACT = 1.0e-3

      # contact 1: furthest from minimum distance contact
      id1 = int(-1)
      dist1 = float(-MJ_MAXVAL)
      for i in range(count):
        if i == min_id:
          continue

        hf_pos = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
        dist = wp.norm_l2(hf_pos - min_pos)

        if dist > dist1:
          id1 = i
          dist1 = dist

      if id1 == -1 or (0.0 < dist1 and dist1 < MIN_DIST_TO_NEXT_CONTACT):
        return

      pos1 = wp.vec3(hfield_contact_pos[id1, 0], hfield_contact_pos[id1, 1], hfield_contact_pos[id1, 2])
      normal1 = wp.vec3(hfield_contact_normal[id1, 0], hfield_contact_normal[id1, 1], hfield_contact_normal[id1, 2])

      write_contact(
        naconmax_in,
        1,
        hfield_contact_dist[id1],
        pos1,
        make_frame(normal1),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        adhesion,
        geoms,
        collision_pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_efc_address_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        contact_adhesion_out,
        nacon_out,
      )

      # contact 2: point furthest from min_pos - pos1 line
      dist_min1 = wp.cross(min_normal, min_pos - pos1)

      id2 = int(-1)
      dist_12 = float(-MJ_MAXVAL)
      for i in range(count):
        if i == min_id or i == id1:
          continue

        hf_pos = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
        dist = wp.abs(wp.dot(hf_pos - min_pos, dist_min1))

        if dist > dist_12:
          id2 = i
          dist_12 = dist

      if id2 == -1 or (0.0 < dist_12 and dist_12 < MIN_DIST_TO_NEXT_CONTACT):
        return

      pos2 = wp.vec3(hfield_contact_pos[id2, 0], hfield_contact_pos[id2, 1], hfield_contact_pos[id2, 2])
      normal2 = wp.vec3(hfield_contact_normal[id2, 0], hfield_contact_normal[id2, 1], hfield_contact_normal[id2, 2])

      write_contact(
        naconmax_in,
        2,
        hfield_contact_dist[id2],
        pos2,
        make_frame(normal2),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        adhesion,
        geoms,
        collision_pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_efc_address_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        contact_adhesion_out,
        nacon_out,
      )

      # contact 3: point furthest from other triangle edge
      vec_min2 = wp.cross(min_normal, min_pos - pos2)
      vec_12 = wp.cross(min_normal, pos1 - pos2)

      id3 = int(-1)
      dist3 = float(-MJ_MAXVAL)
      for i in range(count):
        if i == min_id or i == id1 or i == id2:
          continue

        hf_pos = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
        dist = wp.abs(wp.dot(hf_pos - min_pos, vec_min2)) + wp.abs(wp.dot(pos1 - hf_pos, vec_12))

        if dist > dist3:
          id3 = i
          dist3 = dist

      if id3 == -1 or (0.0 < dist3 and dist3 < MIN_DIST_TO_NEXT_CONTACT):
        return

      pos3 = wp.vec3(hfield_contact_pos[id3, 0], hfield_contact_pos[id3, 1], hfield_contact_pos[id3, 2])
      normal3 = wp.vec3(hfield_contact_normal[id3, 0], hfield_contact_normal[id3, 1], hfield_contact_normal[id3, 2])

      write_contact(
        naconmax_in,
        3,
        hfield_contact_dist[id3],
        pos3,
        make_frame(normal3),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        adhesion,
        geoms,
        collision_pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_efc_address_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        contact_adhesion_out,
        nacon_out,
      )

  return ccd_hfield_kernel


_CCD_OVERSUBSCRIBE_WAVES = 2
_CCD_MIN_BLOCKS = 8


@cache_kernel
def ccd_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  use_multiccd: bool,
  geomgeomid: int,
  block_dim: int,
  warn_overflow: int,
):
  """Kernel builder for non-heightfield CCD collisions (no hfield args)."""

  @wp.func
  def eval_ccd_write_contact(
    # Model:
    opt_ccd_tolerance: wp.array[float],
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
    # Data in:
    naconmax_in: int,
    naccdmax_in: int,
    # In:
    epa_vert_in: wp.array2d[wp.vec3],
    epa_vert_index_in: wp.array2d[int],
    epa_face_in: wp.array2d[int],
    epa_pr_in: wp.array2d[wp.vec3],
    epa_norm2_in: wp.array2d[float],
    epa_horizon_in: wp.array2d[int],
    multiccd_polygon_in: wp.array2d[wp.vec3],
    multiccd_clipped_in: wp.array2d[wp.vec3],
    multiccd_pnormal_in: wp.array2d[wp.vec3],
    multiccd_pdist_in: wp.array2d[float],
    multiccd_idx1_in: wp.array2d[int],
    multiccd_idx2_in: wp.array2d[int],
    multiccd_n1_in: wp.array2d[wp.vec3],
    multiccd_n2_in: wp.array2d[wp.vec3],
    multiccd_endvert_in: wp.array2d[wp.vec3],
    multiccd_face1_in: wp.array2d[wp.vec3],
    multiccd_face2_in: wp.array2d[wp.vec3],
    geom1: Geom,
    geom2: Geom,
    geoms: wp.vec2i,
    worldid: int,
    nccd_in: wp.array[int],
    margin: float,
    gap: float,
    pairid: wp.vec2i,
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
    overflow_out: wp.array[int],
  ):
    geom1.margin = margin
    geom2.margin = margin
    tolerance = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]
    is_collision_sensor = pairid[1] >= 0
    if is_collision_sensor:
      cutoff = 1.0e32
    else:
      cutoff = gap
    # Shift origin to geom1 position for translation invariance with numerical precision.
    origin = geom1.pos
    geom1.pos = wp.vec3(0.0)
    geom2.pos = geom2.pos - origin
    needs_epa, dist, ncollision, w1, w2, gjk_result, geom1, geom2 = gjk_phase(
      tolerance,
      cutoff,
      gjk_iterations,
      geom1,
      geom2,
      geomtype1,
      geomtype2,
      geom1.pos,
      geom2.pos,
    )

    ccdid = int(-1)
    multiccd_idx = int(-1)

    if needs_epa:
      ccdid = wp.atomic_add(nccd_in, geomgeomid, 1)  # kernel_analyzer: ignore[atomic]
      if ccdid >= naccdmax_in:
        if wp.static(bool(warn_overflow & OverflowType.CCD)):
          wp.printf(
            "CCD overflow - please increase naccdmax beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
            naccdmax_in,
          )
        wp.atomic_or(overflow_out, worldid, OverflowType.CCD)
        return
      dist, ncollision, w1, w2, multiccd_idx = epa_phase(
        tolerance,
        epa_iterations,
        gjk_result,
        geom1,
        geom2,
        geomtype1,
        geomtype2,
        epa_vert_in[ccdid],
        epa_vert_index_in[ccdid],
        epa_face_in[ccdid],
        epa_pr_in[ccdid],
        epa_norm2_in[ccdid],
        epa_horizon_in[ccdid],
        wp.static(bool(warn_overflow & OverflowType.EPA_HORIZON)),
        worldid,
        overflow_out,
      )

    if dist >= gap and not is_collision_sensor:
      return

    # CCD operates on margin-inflated shapes (support() inflates each geom by
    # 0.5 * margin).  The returned dist is therefore relative to the inflated
    # geometry.  Correct back to the true surface-to-surface distance so that
    # the constraint pipeline (pos = dist - includemargin) works consistently
    # with the primitive narrowphase, which reports un-inflated distances.
    dist += margin

    witness1 = mat43()
    witness2 = mat43()
    dists = wp.vec4()
    witness1[0] = w1
    witness2[0] = w2

    has_face1 = wp.static(geomtype1 == GeomType.BOX or geomtype1 == GeomType.MESH or geomtype1 == GeomType.CYLINDER)
    has_face2 = wp.static(geomtype2 == GeomType.BOX or geomtype2 == GeomType.MESH or geomtype2 == GeomType.CYLINDER)
    has_edge1 = wp.static(geomtype1 == GeomType.CAPSULE)
    has_edge2 = wp.static(geomtype2 == GeomType.CAPSULE)
    if wp.static(
      (geomtype1 == GeomType.BOX and geomtype2 == GeomType.BOX)
      or (use_multiccd and ((has_face1 and has_face2) or (has_edge1 and has_face2) or (has_face1 and has_edge2)))
    ):
      if not needs_epa and dist < margin and gjk_result.dim > 0:
        ccdid = wp.atomic_add(nccd_in, geomgeomid, 1)  # kernel_analyzer: ignore[atomic]
        if ccdid >= naccdmax_in:
          if wp.static(bool(warn_overflow & OverflowType.CCD)):
            wp.printf(
              "CCD overflow - please increase naccdmax beyond %u\n"
              "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
              naccdmax_in,
            )
          wp.atomic_or(overflow_out, worldid, OverflowType.CCD)
          return
        multiccd_idx = _gjk_simplex_to_epa(gjk_result, epa_vert_in[ccdid], epa_vert_index_in[ccdid], epa_face_in[ccdid])

      if wp.static(geomtype1 == GeomType.MESH):
        # verify that geom1 mesh data is present for multicontact
        if geom1.mesh_polyadr < 0:
          multiccd_idx = -1

      if wp.static(geomtype2 == GeomType.MESH):
        # verify that geom2 mesh data is present for multicontact
        if geom2.mesh_polyadr < 0:
          multiccd_idx = -1

      dists = wp.vec4()
      if multiccd_idx > -1:
        ncollision, witness1, witness2, dists = multicontact(
          multiccd_polygon_in[ccdid],
          multiccd_clipped_in[ccdid],
          multiccd_pnormal_in[ccdid],
          multiccd_pdist_in[ccdid],
          multiccd_idx1_in[ccdid],
          multiccd_idx2_in[ccdid],
          multiccd_n1_in[ccdid],
          multiccd_n2_in[ccdid],
          multiccd_endvert_in[ccdid],
          multiccd_face1_in[ccdid],
          multiccd_face2_in[ccdid],
          epa_vert_in[ccdid],
          epa_vert_index_in[ccdid],
          epa_face_in[ccdid, multiccd_idx],
          w1,
          w2,
          geom1,
          geom2,
          geomtype1,
          geomtype2,
        )

        # multicontact clipping may produce 0 contacts; fall back to the
        # single EPA witness pair
        if ncollision < 1:
          ncollision = 1
          witness1[0] = w1
          witness2[0] = w2

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
      pairid[0],
      worldid,
    )

    # When shapes overlap (dist <= margin, i.e. inflated dist <= 0), witness1 has crossed
    # past witness2, so witness1 - witness2 points from geom[0] to geom[1].
    # When shapes are separated in the gap band (dist > margin, i.e. inflated dist > 0),
    # witness1 is on geom[0] and witness2 is on geom[1], so witness2 - witness1 points
    # from geom[0] to geom[1].
    if dist <= margin:
      frame = make_frame(witness1[0] - witness2[0])
    else:
      frame = make_frame(witness2[0] - witness1[0])

    for i in range(ncollision):
      write_contact(
        naconmax_in,
        i,
        dists[i] + margin if ncollision > 1 else dist,
        0.5 * (witness1[i] + witness2[i]) + origin,
        frame,
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        adhesion,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_efc_address_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        contact_adhesion_out,
        nacon_out,
      )

  # runs convex collision on a set of geom pairs to recover contact info (non-heightfield)
  # Compile occupancy queries and launches with the same block dimension.
  @wp.kernel(
    module="unique",
    module_options={"block_dim": block_dim},
    enable_backward=False,
    grid_stride=False,
    launch_bounds=(block_dim, _CCD_MIN_BLOCKS),
  )
  def ccd_kernel(
    # Model:
    opt_ccd_tolerance: wp.array[float],
    geom_type: wp.array[int],
    geom_condim: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_size: wp.array2d[wp.vec3],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    geom_adhesion: wp.array2d[float],
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
    pair_dim: wp.array[int],
    pair_solref: wp.array2d[wp.vec2],
    pair_solreffriction: wp.array2d[wp.vec2],
    pair_solimp: wp.array2d[vec5],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    pair_adhesion: wp.array2d[float],
    pair_friction: wp.array2d[vec5],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    naconmax_in: int,
    naccdmax_in: int,
    ncollision_in: wp.array[int],
    # In:
    grid_stride_in: int,
    collision_pair_in: wp.array[wp.vec2i],
    collision_pairid_in: wp.array[wp.vec2i],
    collision_worldid_in: wp.array[int],
    epa_vert_in: wp.array2d[wp.vec3],
    epa_vert_index_in: wp.array2d[int],
    epa_face_in: wp.array2d[int],
    epa_pr_in: wp.array2d[wp.vec3],
    epa_norm2_in: wp.array2d[float],
    epa_horizon_in: wp.array2d[int],
    multiccd_polygon_in: wp.array2d[wp.vec3],
    multiccd_clipped_in: wp.array2d[wp.vec3],
    multiccd_pnormal_in: wp.array2d[wp.vec3],
    multiccd_pdist_in: wp.array2d[float],
    multiccd_idx1_in: wp.array2d[int],
    multiccd_idx2_in: wp.array2d[int],
    multiccd_n1_in: wp.array2d[wp.vec3],
    multiccd_n2_in: wp.array2d[wp.vec3],
    multiccd_endvert_in: wp.array2d[wp.vec3],
    multiccd_face1_in: wp.array2d[wp.vec3],
    multiccd_face2_in: wp.array2d[wp.vec3],
    nccd_in: wp.array[int],
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
    # Data out:
    overflow_out: wp.array[int],
  ):
    tid = wp.tid()
    for collisionid in range(tid, wp.min(ncollision_in[0], naconmax_in), grid_stride_in):
      geoms = collision_pair_in[collisionid]
      g1 = geoms[0]
      g2 = geoms[1]

      if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
        continue

      worldid = collision_worldid_in[collisionid]
      pairid = collision_pairid_in[collisionid]
      margin, gap = contact_margin_gap(
        geom_margin,
        geom_gap,
        pair_margin,
        pair_gap,
        geoms,
        pairid[0],
        worldid,
      )

      geom1, geom2 = geom_collision_pair_from_types(
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
        geomtype1,
        geomtype2,
        geoms,
        worldid,
      )

      eval_ccd_write_contact(
        opt_ccd_tolerance,
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
        naconmax_in,
        naccdmax_in,
        epa_vert_in,
        epa_vert_index_in,
        epa_face_in,
        epa_pr_in,
        epa_norm2_in,
        epa_horizon_in,
        multiccd_polygon_in,
        multiccd_clipped_in,
        multiccd_pnormal_in,
        multiccd_pdist_in,
        multiccd_idx1_in,
        multiccd_idx2_in,
        multiccd_n1_in,
        multiccd_n2_in,
        multiccd_endvert_in,
        multiccd_face1_in,
        multiccd_face2_in,
        geom1,
        geom2,
        geoms,
        worldid,
        nccd_in,
        margin,
        gap,
        pairid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_efc_address_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        contact_adhesion_out,
        nacon_out,
        overflow_out,
      )

  return ccd_kernel


def _ccd_grid_size(kernel, naconmax: int, device) -> int:
  # Grid-stride launch width for the CCD kernel: a few device waves, capped at the contact
  # capacity. The kernel strides over the actual candidate count, so we avoid launching one
  # (mostly idle) thread per naconmax slot.
  if device.is_cpu:
    # Warp forces CPU block_dim to 1 and has no CUDA occupancy information.
    return naconmax

  block_size, min_grid_size = wp.get_suggested_block_size(kernel, device)
  return max(1, min(naconmax, _CCD_OVERSUBSCRIBE_WAVES * block_size * min_grid_size))


@event_scope
def convex_narrowphase(m: Model, d: Data, ctx: CollisionContext, collision_table: list[tuple[GeomType, GeomType]]):
  """Runs narrowphase collision detection for convex geom pairs.

  This function handles collision detection for pairs of convex geometries that were
  identified during the broadphase. It uses the Gilbert-Johnson-Keerthi (GJK) algorithm to
  determine the distance between shapes and the Expanding Polytope Algorithm (EPA) to find
  the penetration depth and contact normal for colliding pairs.

  The convex geom types handled by this function are `SPHERE`, `CAPSULE`, `ELLIPSOID`, `CYLINDER`,
  `BOX`, `MESH`, `HFIELD`.

  To optimize performance, this function dynamically builds and launches a specialized
  kernel for each type of convex collision pair present in the model, avoiding unnecessary
  computations for non-existent pair types.
  """

  def _pair_count(p1: int, p2: int) -> Tuple[int, int]:
    idx = upper_trid_index(len(GeomType), p1, p2)
    return m.geom_pair_type_count[idx], idx

  ncollision = sum(_pair_count(g[0].value, g[1].value)[0] for g in collision_table)
  # no convex collisions, early return

  if ncollision == 0:
    return

  # compute npolygonmax and nmeshdegmax given the geom pairs for the model
  nboxbox, _ = _pair_count(GeomType.BOX.value, GeomType.BOX.value)
  if (GeomType.BOX, GeomType.BOX) not in collision_table:
    nboxbox = 0
  nboxmesh, _ = _pair_count(GeomType.BOX.value, GeomType.MESH.value)
  nmeshmesh, _ = _pair_count(GeomType.MESH.value, GeomType.MESH.value)
  ncylcyl, _ = _pair_count(GeomType.CYLINDER.value, GeomType.CYLINDER.value)
  ncylbox, _ = _pair_count(GeomType.CYLINDER.value, GeomType.BOX.value)
  ncylmesh, _ = _pair_count(GeomType.CYLINDER.value, GeomType.MESH.value)
  ncapcyl, _ = _pair_count(GeomType.CAPSULE.value, GeomType.CYLINDER.value)
  ncapmesh, _ = _pair_count(GeomType.CAPSULE.value, GeomType.MESH.value)

  epa_iterations = 16 if nboxbox == ncollision else m.opt.ccd_iterations

  # set to true to enable multiccd
  use_multiccd = (m.opt.disableflags & DisableBit.MULTICCD) == 0

  # note: box<->box multicontact is independent of the use_multiccd flag
  # need at least 4 (square sides) if there's a box collision needing multiccd
  npolygonmax = 4 if nboxbox > 0 else 0
  nmeshdegmax = 3 if nboxbox > 0 else 0

  # need to allocate more memory if there are cylinders or meshes
  if use_multiccd and ncylcyl + ncylbox + ncylmesh + ncapcyl > 0:
    npolygonmax = max(m.npolygonmax, 16)
    nmeshdegmax = max(m.nmeshdegmax, 3)
  elif use_multiccd and nmeshmesh + nboxmesh + ncapmesh > 0:
    minval = 4 if nboxmesh else npolygonmax
    npolygonmax = max(m.npolygonmax, minval)
    nmeshdegmax = max(m.nmeshdegmax, 3)

  # ccd collider count
  nccd = wp.zeros(len(GeomType) * (len(GeomType) + 1) // 2, dtype=int)

  # epa_vert: vertices in EPA polytope
  epa_vert = wp.empty(shape=(d.naccdmax, 10 + 2 * epa_iterations), dtype=wp.vec3)
  # epa_vert_index: vertex indices in EPA polytope
  epa_vert_index = wp.empty(shape=(d.naccdmax, 10 + 2 * epa_iterations), dtype=int)
  # epa_face: faces of polytope represented by three indices
  epa_face = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=int)
  # epa_pr: projection of origin on polytope faces
  epa_pr = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=wp.vec3)
  # epa_norm2: epa_pr * epa_pr
  epa_norm2 = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=float)
  # epa_horizon: index pair (i j) of edges on horizon
  epa_horizon = wp.empty(shape=(d.naccdmax, MJ_MAX_EPAHORIZON), dtype=int)

  # Contact outputs
  contact_outputs = [
    d.contact.dist,
    d.contact.pos,
    d.contact.frame,
    d.contact.includemargin,
    d.contact.friction,
    d.contact.solref,
    d.contact.solreffriction,
    d.contact.solimp,
    d.contact.dim,
    d.contact.geom,
    d.contact.efc_address,
    d.contact.worldid,
    d.contact.type,
    d.contact.geomcollisionid,
    d.contact.adhesion,
    d.nacon,
  ]

  # Launch heightfield collision kernels (no multiccd args, 72 args total)
  for geom_pair in collision_table:
    g1 = geom_pair[0].value
    g2 = geom_pair[1].value
    count, geomgeomid = _pair_count(g1, g2)
    if (g1 == GeomType.HFIELD or g2 == GeomType.HFIELD) and count:
      wp.launch(
        ccd_hfield_kernel_builder(g1, g2, m.opt.ccd_iterations, epa_iterations, geomgeomid, int(m.opt.warn_overflow)),
        dim=d.naconmax,
        inputs=[
          m.opt.ccd_tolerance,
          m.geom_type,
          m.geom_condim,
          m.geom_dataid,
          m.geom_priority,
          m.geom_solmix,
          m.geom_solref,
          m.geom_solimp,
          m.geom_size,
          m.geom_rbound,
          m.geom_friction,
          m.geom_margin,
          m.geom_gap,
          m.geom_adhesion,
          m.mesh_vertadr,
          m.mesh_vertnum,
          m.mesh_graphadr,
          m.mesh_vert,
          m.mesh_graph,
          m.mesh_polynum,
          m.mesh_polyadr,
          m.mesh_polynormal,
          m.mesh_polyvertadr,
          m.mesh_polyvertnum,
          m.mesh_polyvert,
          m.mesh_polymapadr,
          m.mesh_polymapnum,
          m.mesh_polymap,
          m.hfield_size,
          m.hfield_nrow,
          m.hfield_ncol,
          m.hfield_adr,
          m.hfield_data,
          m.pair_dim,
          m.pair_solref,
          m.pair_solreffriction,
          m.pair_solimp,
          m.pair_margin,
          m.pair_gap,
          m.pair_adhesion,
          m.pair_friction,
          d.geom_xpos,
          d.geom_xmat,
          d.naconmax,
          d.naccdmax,
          d.ncollision,
          ctx.collision_pair,
          ctx.collision_pairid,
          ctx.collision_worldid,
          epa_vert,
          epa_vert_index,
          epa_face,
          epa_pr,
          epa_norm2,
          epa_horizon,
          nccd,
        ],
        outputs=contact_outputs + [d.overflow],
      )

  # Allocate multiccd arrays only for non-heightfield collisions
  # multiccd_polygon: clipped contact surface
  multiccd_polygon = wp.empty(shape=(d.naccdmax, 2 * npolygonmax), dtype=wp.vec3)
  # multiccd_clipped: clipped contact surface (intermediate)
  multiccd_clipped = wp.empty(shape=(d.naccdmax, 2 * npolygonmax), dtype=wp.vec3)
  # multiccd_pnormal: plane normal of clipping polygon
  multiccd_pnormal = wp.empty(shape=(d.naccdmax, npolygonmax), dtype=wp.vec3)
  # multiccd_pdist: plane distance of clipping polygon
  multiccd_pdist = wp.empty(shape=(d.naccdmax, npolygonmax), dtype=float)
  # multiccd_idx1: list of normal index candidates for Geom 1
  multiccd_idx1 = wp.empty(shape=(d.naccdmax, nmeshdegmax), dtype=int)
  # multiccd_idx2: list of normal index candidates for Geom 2
  multiccd_idx2 = wp.empty(shape=(d.naccdmax, nmeshdegmax), dtype=int)
  # multiccd_n1: list of normal candidates for Geom 1
  multiccd_n1 = wp.empty(shape=(d.naccdmax, nmeshdegmax), dtype=wp.vec3)
  # multiccd_n2: list of normal candidates for Geom 1
  multiccd_n2 = wp.empty(shape=(d.naccdmax, nmeshdegmax), dtype=wp.vec3)
  # multiccd_endvert: list of edge vertices candidates
  multiccd_endvert = wp.empty(shape=(d.naccdmax, nmeshdegmax), dtype=wp.vec3)
  # multiccd_face1: contact face
  multiccd_face1 = wp.empty(shape=(d.naccdmax, npolygonmax), dtype=wp.vec3)
  # multiccd_face2: contact face
  multiccd_face2 = wp.empty(shape=(d.naccdmax, npolygonmax), dtype=wp.vec3)

  # Launch non-heightfield collision kernels (no hfield args, 78 args total)
  for geom_pair in collision_table:
    g1 = geom_pair[0].value
    g2 = geom_pair[1].value
    count, geomgeomid = _pair_count(g1, g2)
    if g1 != GeomType.HFIELD and g2 != GeomType.HFIELD and count:
      ccd_k = ccd_kernel_builder(
        g1,
        g2,
        m.opt.ccd_iterations,
        epa_iterations,
        use_multiccd,
        geomgeomid,
        m.block_dim.convex_ccd,
        int(m.opt.warn_overflow),
      )
      ccd_grid = _ccd_grid_size(ccd_k, d.naconmax, d.ncollision.device)
      wp.launch(
        ccd_k,
        dim=ccd_grid,
        block_dim=m.block_dim.convex_ccd,
        inputs=[
          m.opt.ccd_tolerance,
          m.geom_type,
          m.geom_condim,
          m.geom_dataid,
          m.geom_priority,
          m.geom_solmix,
          m.geom_solref,
          m.geom_solimp,
          m.geom_size,
          m.geom_friction,
          m.geom_margin,
          m.geom_gap,
          m.geom_adhesion,
          m.mesh_vertadr,
          m.mesh_vertnum,
          m.mesh_graphadr,
          m.mesh_vert,
          m.mesh_graph,
          m.mesh_polynum,
          m.mesh_polyadr,
          m.mesh_polynormal,
          m.mesh_polyvertadr,
          m.mesh_polyvertnum,
          m.mesh_polyvert,
          m.mesh_polymapadr,
          m.mesh_polymapnum,
          m.mesh_polymap,
          m.pair_dim,
          m.pair_solref,
          m.pair_solreffriction,
          m.pair_solimp,
          m.pair_margin,
          m.pair_gap,
          m.pair_adhesion,
          m.pair_friction,
          d.geom_xpos,
          d.geom_xmat,
          d.naconmax,
          d.naccdmax,
          d.ncollision,
          ccd_grid,
          ctx.collision_pair,
          ctx.collision_pairid,
          ctx.collision_worldid,
          epa_vert,
          epa_vert_index,
          epa_face,
          epa_pr,
          epa_norm2,
          epa_horizon,
          multiccd_polygon,
          multiccd_clipped,
          multiccd_pnormal,
          multiccd_pdist,
          multiccd_idx1,
          multiccd_idx2,
          multiccd_n1,
          multiccd_n2,
          multiccd_endvert,
          multiccd_face1,
          multiccd_face2,
          nccd,
        ],
        outputs=contact_outputs + [d.overflow],
      )
