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
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import contact_params
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import geom_collision_pair
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import write_contact
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import box_box
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import capsule_box
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import capsule_capsule
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import plane_box
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import plane_capsule
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import plane_cylinder
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import plane_ellipsoid
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import plane_sphere
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import sphere_box
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import sphere_capsule
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import sphere_cylinder
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import sphere_sphere
from mujoco.mjx.third_party.mujoco_warp._src.math import make_frame
from mujoco.mjx.third_party.mujoco_warp._src.math import upper_trid_index
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import mat43
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _area4(
  # In:
  vert: wp.array[wp.vec3],
  vertadr: int,
  polyvert: wp.array[int],
  face_vertadr: int,
  a: int,
  b: int,
  c: int,
  d: int,
) -> float:
  """Returns the area of a quadrilateral on a mesh polygon."""
  va = vert[vertadr + polyvert[face_vertadr + a]]
  vb = vert[vertadr + polyvert[face_vertadr + b]]
  vc = vert[vertadr + polyvert[face_vertadr + c]]
  vd = vert[vertadr + polyvert[face_vertadr + d]]
  return 0.5 * wp.length(wp.cross(va - vc, vb - vd))


@wp.func
def _hull4(
  # In:
  vert: wp.array[wp.vec3],
  vertadr: int,
  polyvert: wp.array[int],
  face_vertadr: int,
  nhull: int,
  a: int,
) -> Tuple[int, wp.vec4i]:
  """Prunes a convex polygon to at most four vertices, keeping anchor `a` (MuJoCo's hull4f)."""
  b = (a + 1) % nhull
  c = (a + 2) % nhull
  d = (a + 3) % nhull
  res = wp.vec4i(a, b, c, d)
  if nhull <= 4:
    return nhull, res

  m = _area4(vert, vertadr, polyvert, face_vertadr, a, b, c, d)
  while True:
    d_next = (d + 1) % nhull
    m_next = _area4(vert, vertadr, polyvert, face_vertadr, a, b, c, d_next)
    if m_next <= m:
      break
    d = d_next
    m = m_next
    res = wp.vec4i(a, b, c, d)
    while True:
      c_next = (c + 1) % nhull
      m_next = _area4(vert, vertadr, polyvert, face_vertadr, a, b, c_next, d)
      if m_next <= m:
        break
      c = c_next
      m = m_next
      res = wp.vec4i(a, b, c, d)
    while True:
      b_next = (b + 1) % nhull
      m_next = _area4(vert, vertadr, polyvert, face_vertadr, a, b_next, c, d)
      if m_next <= m:
        break
      b = b_next
      m = m_next
      res = wp.vec4i(a, b, c, d)
  return 4, res


@wp.func
def plane_convex(plane_normal: wp.vec3, plane_pos: wp.vec3, convex: Geom, margin: float) -> Tuple[wp.vec4, mat43, wp.vec3]:
  """Core contact geometry calculation for plane-convex collision."""
  contact_dist = wp.vec4(MJ_MAXVAL)
  contact_pos = mat43()

  # Rank support in the mesh frame, independently of the plane's position.
  local_normal = wp.transpose(convex.rot) @ plane_normal
  local_dir = -local_normal
  vertindex = int(-1)
  max_support = wp.float32(-MJ_MAXVAL)

  if convex.graphadr == -1 or convex.vertnum < 10:
    for i in range(convex.vertnum):
      support = wp.dot(local_dir, convex.vert[convex.vertadr + i])
      if support > max_support:
        max_support = support
        vertindex = i
  else:
    numvert = convex.graph[convex.graphadr]
    vert_edgeadr = convex.graphadr + 2
    vert_globalid = convex.graphadr + 2 + numvert
    edge_localid = convex.graphadr + 2 + 2 * numvert

    prev = int(-1)
    imax = int(0)
    max_support = wp.dot(local_dir, convex.vert[convex.vertadr + convex.graph[vert_globalid + 0]])
    while imax != prev:
      prev = imax
      i = convex.graph[vert_edgeadr + imax]
      while convex.graph[edge_localid + i] >= 0:
        subidx = convex.graph[edge_localid + i]
        idx = convex.graph[vert_globalid + subidx]
        support = wp.dot(local_dir, convex.vert[convex.vertadr + idx])
        if support > max_support:
          max_support = support
          imax = int(subidx)
        i += int(1)
    vertindex = convex.graph[vert_globalid + imax]

  if vertindex < 0:
    return contact_dist, contact_pos, plane_normal

  v0 = convex.pos + convex.rot @ convex.vert[convex.vertadr + vertindex]
  dist0 = wp.dot(plane_normal, v0 - plane_pos)
  contact_dist[0] = dist0
  contact_pos[0] = v0 - 0.5 * dist0 * plane_normal

  # Separated past margin, write_contact drops every contact: skip the incident face.
  if convex.mesh_polynum <= 0 or dist0 > margin:
    return contact_dist, contact_pos, plane_normal

  # Choose the incident face most anti-aligned with the plane normal.
  polymapadr = convex.mesh_polymapadr[convex.vertadr + vertindex]
  polymapnum = convex.mesh_polymapnum[convex.vertadr + vertindex]

  best_poly = int(-1)
  best_dot = wp.float32(1.0)
  for i in range(polymapnum):
    idx = convex.mesh_polymap[polymapadr + i]
    pn = convex.mesh_polynormal[convex.mesh_polyadr + idx]
    ndot = wp.dot(pn, local_normal)
    if ndot < best_dot:
      best_dot = ndot
      best_poly = idx

  if best_poly < 0:
    return contact_dist, contact_pos, plane_normal

  face_vertadr = convex.mesh_polyvertadr[convex.mesh_polyadr + best_poly]
  face_vertnum = convex.mesh_polyvertnum[convex.mesh_polyadr + best_poly]

  a = int(0)
  for i in range(face_vertnum):
    if convex.mesh_polyvert[face_vertadr + i] == vertindex:
      a = i
      break

  n, indices = _hull4(convex.vert, convex.vertadr, convex.mesh_polyvert, face_vertadr, face_vertnum, a)

  ncon = int(1)
  for i in range(1, n):
    v_idx = convex.mesh_polyvert[face_vertadr + indices[i]]
    pnt = convex.pos + convex.rot @ convex.vert[convex.vertadr + v_idx]
    # Match MuJoCo's mesh-center filter; write_contact applies margin + gap.
    if wp.dot(plane_normal, pnt - convex.pos) > 0.0:
      continue
    vdist = wp.dot(plane_normal, pnt - plane_pos)
    contact_dist[ncon] = vdist
    contact_pos[ncon] = pnt - 0.5 * vdist * plane_normal
    ncon += 1

  return contact_dist, contact_pos, plane_normal


@wp.func
def plane_sphere_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  plane: Geom,
  sphere: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contact between a sphere and a plane."""
  normal = plane.normal
  dist, pos = plane_sphere(normal, plane.pos, sphere.pos, sphere.size[0])

  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(normal),
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


@wp.func
def sphere_sphere_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  sphere1: Geom,
  sphere2: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contact between two spheres."""
  dist, pos, normal = sphere_sphere(sphere1.pos, sphere1.size[0], sphere2.pos, sphere2.size[0])

  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(normal),
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


@wp.func
def sphere_capsule_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  sphere: Geom,
  cap: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates one contact between a sphere and a capsule."""
  # capsule axis
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])

  dist, pos, normal = sphere_capsule(sphere.pos, sphere.size[0], cap.pos, axis, cap.size[0], cap.size[1])

  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(normal),
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


@wp.func
def capsule_capsule_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  cap1: Geom,
  cap2: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between two capsules."""
  # capsule axes
  cap1_axis = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
  cap2_axis = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])

  dist, pos, normal = capsule_capsule(
    cap1.pos,
    cap1_axis,
    cap1.size[0],  # radius1
    cap1.size[1],  # half_length1
    cap2.pos,
    cap2_axis,
    cap2.size[0],  # radius2
    cap2.size[1],  # half_length2
  )

  for i in range(2):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),
      make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),
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


@wp.func
def plane_capsule_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  plane: Geom,
  cap: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between a capsule and a plane."""
  # capsule axis
  capsule_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])

  dist, pos, frame = plane_capsule(
    plane.normal,
    plane.pos,
    cap.pos,
    capsule_axis,
    cap.size[0],  # radius
    cap.size[1],  # half_length
  )

  for i in range(2):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      pos[i],
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


@wp.func
def plane_ellipsoid_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  plane: Geom,
  ellipsoid: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between an ellipsoid and a plane."""
  dist, pos, normal = plane_ellipsoid(plane.normal, plane.pos, ellipsoid.pos, ellipsoid.rot, ellipsoid.size)

  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(normal),
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


@wp.func
def plane_box_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  plane: Geom,
  box: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between a box and a plane."""
  dist, pos, normal = plane_box(plane.normal, plane.pos, box.pos, box.rot, box.size)
  frame = make_frame(normal)

  for i in range(8):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      pos[i],
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


@wp.func
def plane_convex_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  plane: Geom,
  convex: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between a plane and a convex object."""
  dist, pos, normal = plane_convex(plane.normal, plane.pos, convex, margin + gap)

  frame = make_frame(normal)
  for i in range(4):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      pos[i],
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


@wp.func
def sphere_cylinder_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  sphere: Geom,
  cylinder: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between a sphere and a cylinder."""
  # cylinder axis
  cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])

  dist, pos, normal = sphere_cylinder(
    sphere.pos,
    sphere.size[0],  # sphere radius
    cylinder.pos,
    cylinder_axis,
    cylinder.size[0],  # cylinder radius
    cylinder.size[1],  # cylinder half_height
  )

  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(normal),
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


@wp.func
def plane_cylinder_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  plane: Geom,
  cylinder: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between a cylinder and a plane."""
  # cylinder axis
  cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])

  dist, pos, normal = plane_cylinder(
    plane.normal,
    plane.pos,
    cylinder.pos,
    cylinder_axis,
    cylinder.size[0],  # radius
    cylinder.size[1],  # half_height
  )

  frame = make_frame(normal)
  for i in range(4):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      pos[i],
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


@wp.func
def sphere_box_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  sphere: Geom,
  box: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  dist, pos, normal = sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot, box.size)

  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(normal),
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


@wp.func
def capsule_box_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  cap: Geom,
  box: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between a capsule and a box."""
  # Extract capsule axis
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])

  # Call the core function to get contact geometry
  dist, pos, normal = capsule_box(
    cap.pos,
    axis,
    cap.size[0],  # capsule radius
    cap.size[1],  # capsule half length
    box.pos,
    box.rot,
    box.size,
  )

  # Loop over the contacts and write them
  for i in range(2):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      pos[i],
      make_frame(normal[i]),
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


@wp.func
def box_box_wrapper(
  # Data in:
  naconmax_in: int,
  # In:
  box1: Geom,
  box2: Geom,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  adhesion: float,
  geoms: wp.vec2i,
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
):
  """Calculates contacts between two boxes."""
  # Call the core function to get contact geometry; as in MuJoCo, the collider's acceptance
  # band is margin + gap
  ncon, dist, pos, normal = box_box(
    box1.pos,
    box1.rot,
    box1.size,
    box2.pos,
    box2.rot,
    box2.size,
    margin + gap,
  )
  if ncon == 0:
    return
  frame = make_frame(normal)

  for i in range(ncon):
    write_contact(
      naconmax_in,
      i,
      dist[i],
      pos[i],
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


# Map of supported primitive collision functions
_PRIMITIVE_COLLISIONS = {
  (GeomType.PLANE, GeomType.SPHERE): plane_sphere_wrapper,
  (GeomType.PLANE, GeomType.CAPSULE): plane_capsule_wrapper,
  (GeomType.PLANE, GeomType.ELLIPSOID): plane_ellipsoid_wrapper,
  (GeomType.PLANE, GeomType.CYLINDER): plane_cylinder_wrapper,
  (GeomType.PLANE, GeomType.BOX): plane_box_wrapper,
  (GeomType.PLANE, GeomType.MESH): plane_convex_wrapper,
  (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere_wrapper,
  (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule_wrapper,
  (GeomType.SPHERE, GeomType.CYLINDER): sphere_cylinder_wrapper,
  (GeomType.SPHERE, GeomType.BOX): sphere_box_wrapper,
  (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule_wrapper,
  (GeomType.CAPSULE, GeomType.BOX): capsule_box_wrapper,
  (GeomType.BOX, GeomType.BOX): box_box_wrapper,
}


@cache_kernel
def _primitive_narrowphase(primitive_collisions_types, primitive_collisions_func):
  @wp.kernel(module="unique", enable_backward=False)
  def primitive_narrowphase(
    # Model:
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
    ncollision_in: wp.array[int],
    # In:
    collision_pair_in: wp.array[wp.vec2i],
    collision_pairid_in: wp.array[wp.vec2i],
    collision_worldid_in: wp.array[int],
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
  ):
    tid = wp.tid()

    if tid >= ncollision_in[0]:
      return

    geoms = collision_pair_in[tid]
    worldid = collision_worldid_in[tid]

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
      tid,
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

    for i in range(wp.static(len(primitive_collisions_func))):
      collision_type1 = wp.static(primitive_collisions_types[i][0])
      collision_type2 = wp.static(primitive_collisions_types[i][1])
      type1 = geom_type[geoms[0]]
      type2 = geom_type[geoms[1]]
      if collision_type1 == type1 and collision_type2 == type2:
        wp.static(primitive_collisions_func[i])(
          naconmax_in,
          geom1,
          geom2,
          worldid,
          margin,
          gap,
          condim,
          friction,
          solref,
          solreffriction,
          solimp,
          adhesion,
          geoms,
          collision_pairid_in[tid],
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

  return primitive_narrowphase


@event_scope
def primitive_narrowphase(m: Model, d: Data, ctx: CollisionContext, collision_table: list[tuple[GeomType, GeomType]]):
  """Runs collision detection on primitive geom pairs discovered during broadphase.

  This function processes collision pairs involving primitive shapes that were
  identified during the broadphase stage. It computes detailed contact information
  such as distance, position, and frame, and populates the `d.contact` array.

  The primitive geom types: `PLANE`, `SPHERE`, `CAPSULE`, `CYLINDER`, and `BOX`.

  Additionally, collisions between planes and convex hulls.

  To improve performance, it dynamically builds and launches a kernel tailored to
  the specific primitive collision types present in the model, avoiding
  unnecessary checks for non-existent collision pairs.
  """
  # TODO(team): keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.

  # the pair types are computed per call (rather than accumulated globally) so
  # that pair types of one model do not leak into the kernels of other models
  primitive_collision_types = []
  primitive_collision_func = []
  for types, func in _PRIMITIVE_COLLISIONS.items():
    if types not in collision_table:
      continue
    idx = upper_trid_index(len(GeomType), types[0].value, types[1].value)
    if m.geom_pair_type_count[idx]:
      primitive_collision_types.append(types)
      primitive_collision_func.append(func)

  wp.launch(
    _primitive_narrowphase(primitive_collision_types, primitive_collision_func),
    dim=d.naconmax,
    inputs=[
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
      d.ncollision,
      ctx.collision_pair,
      ctx.collision_pairid,
      ctx.collision_worldid,
    ],
    outputs=[
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
    ],
  )
