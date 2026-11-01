# Copyright 2025 The Physics-Next Project Developers
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
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import contact_params
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import geom_collision_pair
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import write_contact
from mujoco.mjx.third_party.mujoco_warp._src.math import make_frame
from mujoco.mjx.third_party.mujoco_warp._src.ray import ray_mesh
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.types import vec8
from mujoco.mjx.third_party.mujoco_warp._src.types import vec8i
from mujoco.mjx.third_party.mujoco_warp._src.types import vec_pluginattr
from mujoco.mjx.third_party.mujoco_warp._src.util_misc import halton
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class OptimizationParams:
  rel_mat: wp.mat33
  rel_pos: wp.vec3
  attr1: vec_pluginattr
  attr2: vec_pluginattr


@wp.struct
class AABB:
  min: wp.vec3
  max: wp.vec3


@wp.struct
class VolumeData:
  center: wp.vec3
  half_size: wp.vec3
  oct_aabb: wp.array2d[wp.vec3]
  oct_child: wp.array[vec8i]
  oct_coeff: wp.array[vec8]
  root: int = 0
  valid: bool = False


@wp.struct
class MeshData:
  nmeshface: int
  mesh_vertadr: wp.array[int]
  mesh_vert: wp.array[wp.vec3]
  mesh_faceadr: wp.array[int]
  mesh_face: wp.array[wp.vec3i]
  data_id: int
  pos: wp.vec3
  mat: wp.mat33
  size: wp.vec3
  pnt: wp.vec3
  vec: wp.vec3
  valid: bool = False


@wp.func
def get_sdf_params(
  # Model:
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  mesh_octadr: wp.array[int],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  # In:
  g_type: int,
  g_size: wp.vec3,
  plugin_id: int,
  mesh_id: int,
) -> Tuple[vec_pluginattr, int, VolumeData, MeshData]:
  # default attributes from geom size, first 3 values copied
  attributes = vec_pluginattr()
  attributes[0] = g_size[0]
  attributes[1] = g_size[1]
  attributes[2] = g_size[2]
  plugin_index = -1
  volume_data = VolumeData()

  if g_type == GeomType.SDF and plugin_id != -1:
    attributes = plugin_attr[plugin_id]
    plugin_index = plugin[plugin_id]

  elif g_type == GeomType.SDF and mesh_id != -1:
    octadr = mesh_octadr[mesh_id]
    volume_data.center = oct_aabb[octadr, 0]
    volume_data.half_size = oct_aabb[octadr, 1]
    volume_data.root = octadr
    volume_data.oct_aabb = oct_aabb
    volume_data.oct_child = oct_child
    volume_data.oct_coeff = oct_coeff
    volume_data.valid = True

  elif g_type == GeomType.MESH and mesh_id != -1 and mesh_octadr[mesh_id] != -1:
    octadr = mesh_octadr[mesh_id]
    volume_data.center = oct_aabb[octadr, 0]
    volume_data.half_size = oct_aabb[octadr, 1]
    volume_data.root = octadr
    volume_data.oct_aabb = oct_aabb
    volume_data.oct_child = oct_child
    volume_data.oct_coeff = oct_coeff
    volume_data.valid = True

  return attributes, plugin_index, volume_data, MeshData()


@wp.func
def transform_aabb(aabb_pos: wp.vec3, aabb_size: wp.vec3, pos: wp.vec3, ori: wp.mat33) -> AABB:
  aabb = AABB()
  aabb.max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
  aabb.min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)
  for i in range(8):
    vec = wp.vec3(
      aabb_size.x * (1.0 if (i & 1) else -1.0),
      aabb_size.y * (1.0 if (i & 2) else -1.0),
      aabb_size.z * (1.0 if (i & 4) else -1.0),
    )
    frame_vec = ori * (vec + aabb_pos) + pos
    aabb.min = wp.min(aabb.min, frame_vec)
    aabb.max = wp.max(aabb.max, frame_vec)
  return aabb


@wp.func
def radial_field(a: wp.vec3, x: wp.vec3, size: wp.vec3) -> wp.vec3:
  field = wp.cw_div(-size, a)
  field = wp.normalize(field)
  field[0] *= wp.sign(x[0])
  field[1] *= wp.sign(x[1])
  field[2] *= wp.sign(x[2])
  return field


@wp.func
def sphere(p: wp.vec3, size: wp.vec3) -> float:
  return wp.length(p) - size[0]


@wp.func
def box(p: wp.vec3, size: wp.vec3) -> float:
  a = wp.abs(p) - size
  if a[0] >= 0 or a[1] >= 0 or a[2] >= 0:
    z = wp.vec3(0.0, 0.0, 0.0)
    b = wp.max(a, z)
    return wp.norm_l2(b) + wp.min(wp.max(a), 0.0)
  b = radial_field(a, p, size)
  t = -wp.cw_div(a, wp.abs(b))
  return -wp.min(t) * wp.norm_l2(b)


@wp.func
def ellipsoid(p: wp.vec3, size: wp.vec3) -> float:
  scaled_p = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  k0 = wp.length(scaled_p)
  k1 = wp.length(wp.vec3(p[0] / (size[0] ** 2.0), p[1] / (size[1] ** 2.0), p[2] / (size[2] ** 2.0)))
  if k1 < MJ_MINVAL:
    return -wp.min(size)
  return k0 * (k0 - 1.0) / k1


@wp.func
def capsule(p: wp.vec3, size: wp.vec3) -> float:
  r = size[0]
  h = size[1]
  pz_clamped = wp.clamp(p[2], -h, h)
  diff = wp.vec3(p[0], p[1], p[2] - pz_clamped)
  return wp.length(diff) - r


@wp.func
def cylinder(p: wp.vec3, size: wp.vec3) -> float:
  r = size[0]
  h = size[1]
  dx = wp.length(wp.vec2(p[0], p[1])) - r
  dy = wp.abs(p[2]) - h
  return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def grad_sphere(p: wp.vec3) -> wp.vec3:
  c = wp.length(p)
  if c > 1e-9:
    return p / c
  else:
    return wp.vec3(0.0)


@wp.func
def grad_box(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  a = wp.abs(p) - size
  if wp.max(a) < 0.0:
    return radial_field(a, p, size)
  b = wp.max(a, wp.vec3(0.0))
  c = wp.norm_l2(b)
  g = wp.vec3(0.0)
  if c == 0.0:
    for j in range(3):
      if a[j] == 0.0:
        g[j] = -1.0 if p[j] < 0.0 else 1.0
    return wp.normalize(g)
  for j in range(3):
    if a[j] > 0.0:
      g[j] = (b[j] / c) * (-1.0 if p[j] < 0.0 else 1.0)
  return g


@wp.func
def grad_ellipsoid(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  a = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  b = wp.vec3(a[0] / size[0], a[1] / size[1], a[2] / size[2])
  k0 = wp.length(a)
  k1 = wp.length(b)
  if k0 < MJ_MINVAL or k1 < MJ_MINVAL:
    return wp.vec3(0.0)
  invK0 = 1.0 / k0
  invK1 = 1.0 / k1
  gk0 = b * invK0
  gk1 = wp.vec3(
    b[0] * invK1 / (size[0] * size[0]),
    b[1] * invK1 / (size[1] * size[1]),
    b[2] * invK1 / (size[2] * size[2]),
  )
  df_dk0 = (2.0 * k0 - 1.0) * invK1
  df_dk1 = k0 * (k0 - 1.0) * invK1 * invK1
  raw_grad = gk0 * df_dk0 - gk1 * df_dk1
  norm = wp.length(raw_grad)
  if norm > MJ_MINVAL:
    return raw_grad / norm
  return wp.vec3(0.0)


@wp.func
def grad_capsule(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  h = size[1]
  pz_clamped = wp.clamp(p[2], -h, h)
  diff = wp.vec3(p[0], p[1], p[2] - pz_clamped)
  c = wp.length(diff)
  if c > MJ_MINVAL:
    return diff / c
  else:
    return wp.vec3(0.0)


@wp.func
def grad_cylinder(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  r = size[0]
  h = size[1]

  radial_dist = wp.length(wp.vec2(p[0], p[1]))
  if radial_dist > MJ_MINVAL:
    u = wp.vec3(p[0] / radial_dist, p[1] / radial_dist, 0.0)
  else:
    u = wp.vec3(0.0)

  w = wp.vec3(0.0, 0.0, wp.sign(p[2]))

  dx = radial_dist - r
  dy = wp.abs(p[2]) - h

  if dx > 0.0 and dy > 0.0:
    v = wp.vec2(dx, dy)
    len_v = wp.length(v)
    if len_v > MJ_MINVAL:
      return u * (dx / len_v) + w * (dy / len_v)
    else:
      return wp.vec3(0.0)
  elif dx > 0.0:
    return u
  elif dy > 0.0:
    return w
  else:
    if dx > dy:
      return u
    else:
      return w


@wp.func
def user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float:
  """User-defined SDF function.

  Access attributes via attr[i] where i is the attribute index (0 to _NPLUGINATTR-1).
  """
  wp.printf("ERROR: user_sdf function must be implemented by user code\n")
  return 0.0


@wp.func
def user_sdf_grad(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> wp.vec3:
  """User-defined SDF gradient function.

  Access attributes via attr[i] where i is the attribute index (0 to _NPLUGINATTR-1).
  """
  wp.printf("ERROR: user_sdf_grad function must be implemented by user code\n")
  return wp.vec3(0.0)


@wp.func
def find_oct(
  oct_child: wp.array[vec8i], oct_aabb: wp.array2d[wp.vec3], p: wp.vec3, grad: bool, root: int
) -> Tuple[int, Tuple[vec8, vec8, vec8]]:
  stack = root
  niter = int(100)
  rx = vec8(0.0)
  ry = vec8(0.0)
  rz = vec8(0.0)
  eps = 1e-6

  while niter > 0:
    niter -= 1
    node = stack

    if node == -1:
      wp.printf("ERROR: Invalid node number\n")
      return -1, (rx, ry, rz)

    vmin = oct_aabb[node, 0] - oct_aabb[node, 1]
    vmax = oct_aabb[node, 0] + oct_aabb[node, 1]

    if (
      p[0] + eps < vmin[0]
      or p[0] - eps > vmax[0]
      or p[1] + eps < vmin[1]
      or p[1] - eps > vmax[1]
      or p[2] + eps < vmin[2]
      or p[2] - eps > vmax[2]
    ):
      continue

    coord = wp.cw_div(p - vmin, vmax - vmin)

    # check if the node is a leaf
    # child indices are relative to root (mesh_octadr offset)
    child0 = oct_child[node][0]
    # Evaluate this hot leaf predicate eagerly to avoid branch-heavy codegen.
    if (
      int(child0 == -1)
      & int(oct_child[node][1] == -1)
      & int(oct_child[node][2] == -1)
      & int(oct_child[node][3] == -1)
      & int(oct_child[node][4] == -1)
      & int(oct_child[node][5] == -1)
      & int(oct_child[node][6] == -1)
      & int(oct_child[node][7] == -1)
    ) != 0:
      if not grad:
        for j in range(8):
          rx[j] = (
            (coord[0] if j & 1 else 1.0 - coord[0])
            * (coord[1] if j & 2 else 1.0 - coord[1])
            * (coord[2] if j & 4 else 1.0 - coord[2])
          )
      else:
        # Convert derivatives of cell coordinates to derivatives of physical coordinates.
        inv_cell = wp.cw_div(wp.vec3(1.0), vmax - vmin)
        for j in range(8):
          # fmt: off
          rx[j] = ((1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1])
                   * (coord[2] if j & 4 else 1.0 - coord[2]) * inv_cell[0])
          ry[j] = ((coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0)
                   * (coord[2] if j & 4 else 1.0 - coord[2]) * inv_cell[1])
          rz[j] = ((coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1])
                   * (1.0 if j & 4 else -1.0) * inv_cell[2])
          # fmt: on
      return node, (rx, ry, rz)

    # compute which of 8 children to visit next
    # child indices are stored relative to mesh_octadr, add root offset
    x = 0 if coord[0] < 0.5 else 1
    y = 0 if coord[1] < 0.5 else 1
    z = 0 if coord[2] < 0.5 else 1
    child = oct_child[node][4 * z + 2 * y + x]
    stack = child + root if child != -1 else -1

  wp.print("ERROR: Node not found\n")
  return -1, (rx, ry, rz)


@wp.func
def box_project(center: wp.vec3, half_size: wp.vec3, xyz: wp.vec3) -> Tuple[float, wp.vec3]:
  r = xyz - center
  q = wp.vec3(wp.abs(r[0]) - half_size[0], wp.abs(r[1]) - half_size[1], wp.abs(r[2]) - half_size[2])

  if q[0] <= 0.0 and q[1] <= 0.0 and q[2] <= 0.0:
    return 0.0, xyz

  else:
    dist_sqr = 0.0
    eps = 1e-4
    point = wp.vec3(xyz[0], xyz[1], xyz[2])

    if q[0] >= 0.0:
      dist_sqr += q[0] * q[0]
      if r[0] > 0.0:
        point = wp.vec3(point[0] - (q[0] + eps), point[1], point[2])
      else:
        point = wp.vec3(point[0] + (q[0] + eps), point[1], point[2])

    if q[1] >= 0.0:
      dist_sqr += q[1] * q[1]
      if r[1] > 0.0:
        point = wp.vec3(point[0], point[1] - (q[1] + eps), point[2])
      else:
        point = wp.vec3(point[0], point[1] + (q[1] + eps), point[2])

    if q[2] >= 0.0:
      dist_sqr += q[2] * q[2]
      if r[2] > 0.0:
        point = wp.vec3(point[0], point[1], point[2] - (q[2] + eps))
      else:
        point = wp.vec3(point[0], point[1], point[2] + (q[2] + eps))

    return wp.sqrt(dist_sqr), point


@wp.func
def sample_volume_sdf(xyz: wp.vec3, volume_data: VolumeData) -> float:
  dist0, point = box_project(volume_data.center, volume_data.half_size, xyz)
  node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=False, root=volume_data.root)
  return dist0 + wp.dot(weights[0], volume_data.oct_coeff[node])


@wp.func
def sample_volume_grad(xyz: wp.vec3, volume_data: VolumeData) -> wp.vec3:
  dist0, point = box_project(volume_data.center, volume_data.half_size, xyz)
  if dist0 > 0:
    h = 1e-4
    dx = wp.vec3(h, 0.0, 0.0)
    dy = wp.vec3(0.0, h, 0.0)
    dz = wp.vec3(0.0, 0.0, h)
    f = sample_volume_sdf(xyz, volume_data)
    grad_x = (sample_volume_sdf(xyz + dx, volume_data) - f) / h
    grad_y = (sample_volume_sdf(xyz + dy, volume_data) - f) / h
    grad_z = (sample_volume_sdf(xyz + dz, volume_data) - f) / h
    return wp.vec3(grad_x, grad_y, grad_z)
  node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=True, root=volume_data.root)
  grad_x = wp.dot(weights[0], volume_data.oct_coeff[node])
  grad_y = wp.dot(weights[1], volume_data.oct_coeff[node])
  grad_z = wp.dot(weights[2], volume_data.oct_coeff[node])
  return wp.vec3(grad_x, grad_y, grad_z)


@wp.func
def sdf(type: int, p: wp.vec3, attr: vec_pluginattr, sdf_type: int, volume_data: VolumeData, mesh_data: MeshData) -> float:
  # extract first 3 elements as vec3 for primitive sdf functions
  attr_vec3 = wp.vec3(attr[0], attr[1], attr[2])
  if type == GeomType.PLANE:
    return p[2]
  elif type == GeomType.SPHERE:
    return sphere(p, attr_vec3)
  elif type == GeomType.CAPSULE:
    return capsule(p, attr_vec3)
  elif type == GeomType.CYLINDER:
    return cylinder(p, attr_vec3)
  elif type == GeomType.BOX:
    return box(p, attr_vec3)
  elif type == GeomType.ELLIPSOID:
    return ellipsoid(p, attr_vec3)
  elif type == GeomType.MESH and mesh_data.valid:
    mesh_data.pnt = p
    mesh_data.vec = -wp.normalize(p)
    dist, normal = ray_mesh(
      mesh_data.nmeshface,
      mesh_data.mesh_vertadr,
      mesh_data.mesh_faceadr,
      mesh_data.mesh_vert,
      mesh_data.mesh_face,
      mesh_data.data_id,
      mesh_data.pos,
      mesh_data.mat,
      mesh_data.size,
      mesh_data.pnt,
      mesh_data.vec,
    )
    if dist > wp.norm_l2(p):
      dist, normal = ray_mesh(
        mesh_data.nmeshface,
        mesh_data.mesh_vertadr,
        mesh_data.mesh_faceadr,
        mesh_data.mesh_vert,
        mesh_data.mesh_face,
        mesh_data.data_id,
        mesh_data.pos,
        mesh_data.mat,
        mesh_data.size,
        mesh_data.pnt,
        -mesh_data.vec,
      )
      return -dist
    return dist
  elif type == GeomType.SDF:
    if sdf_type == -1:
      return sample_volume_sdf(p, volume_data)
    else:
      return user_sdf(p, attr, sdf_type)
  elif type == GeomType.MESH and volume_data.valid:
    return sample_volume_sdf(p, volume_data)
  wp.printf("ERROR: SDF type not implemented\n")
  return 0.0


@wp.func
def sdf_grad(
  type: int, p: wp.vec3, attr: vec_pluginattr, sdf_type: int, volume_data: VolumeData, mesh_data: MeshData
) -> wp.vec3:
  # extract first 3 elements as vec3 for primitive sdf functions
  attr_vec3 = wp.vec3(attr[0], attr[1], attr[2])
  if type == GeomType.PLANE:
    grad = wp.vec3(0.0, 0.0, 1.0)
    return grad
  elif type == GeomType.SPHERE:
    return grad_sphere(p)
  elif type == GeomType.CAPSULE:
    return grad_capsule(p, attr_vec3)
  elif type == GeomType.CYLINDER:
    return grad_cylinder(p, attr_vec3)
  elif type == GeomType.BOX:
    return grad_box(p, attr_vec3)
  elif type == GeomType.ELLIPSOID:
    return grad_ellipsoid(p, attr_vec3)
  elif type == GeomType.MESH and mesh_data.valid:
    mesh_data.pnt = p
    mesh_data.vec = -wp.normalize(p)
    dist, normal = ray_mesh(
      mesh_data.nmeshface,
      mesh_data.mesh_vertadr,
      mesh_data.mesh_faceadr,
      mesh_data.mesh_vert,
      mesh_data.mesh_face,
      mesh_data.data_id,
      mesh_data.pos,
      mesh_data.mat,
      mesh_data.size,
      mesh_data.pnt,
      mesh_data.vec,
    )
    if dist > wp.norm_l2(p):
      return wp.vec3(1.0)
    else:
      return wp.vec3(-1.0)

  elif type == GeomType.SDF:
    if sdf_type == -1:
      return sample_volume_grad(p, volume_data)
    else:
      return user_sdf_grad(p, attr, sdf_type)
  elif type == GeomType.MESH and volume_data.valid:
    return sample_volume_grad(p, volume_data)
  wp.printf("ERROR: SDF grad type not implemented\n")
  return wp.vec3(0.0)


@wp.func
def clearance(
  # In:
  type1: int,
  p1: wp.vec3,
  p2: wp.vec3,
  s1: vec_pluginattr,
  s2: vec_pluginattr,
  sdf_type1: int,
  sdf_type2: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> float:
  sdf1 = sdf(type1, p1, s1, sdf_type1, volume_data1, mesh_data1)
  sdf2 = sdf(GeomType.SDF, p2, s2, sdf_type2, volume_data2, mesh_data2)
  if sfd_intersection:
    return wp.max(sdf1, sdf2)
  else:
    return sdf1 + sdf2 + wp.abs(wp.max(sdf1, sdf2))


@wp.func
def compute_grad(
  # In:
  type1: int,
  p1: wp.vec3,
  p2: wp.vec3,
  params: OptimizationParams,
  sdf_type1: int,
  sdf_type2: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> wp.vec3:
  A = sdf(type1, p1, params.attr1, sdf_type1, volume_data1, mesh_data1)
  B = sdf(GeomType.SDF, p2, params.attr2, sdf_type2, volume_data2, mesh_data2)
  grad1 = sdf_grad(type1, p1, params.attr1, sdf_type1, volume_data1, mesh_data1)
  grad2 = sdf_grad(GeomType.SDF, p2, params.attr2, sdf_type2, volume_data2, mesh_data2)
  grad1_transformed = wp.transpose(params.rel_mat) * grad1
  if sfd_intersection:
    if A > B:
      return grad1_transformed
    else:
      return grad2
  else:
    gradient = grad2 + grad1_transformed
    max_val = wp.max(A, B)
    if A > B:
      max_grad = grad1_transformed
    else:
      max_grad = grad2
    sign = wp.sign(max_val)
    gradient += max_grad * sign
    return gradient


@wp.func
def gradient_step(
  # In:
  type1: int,
  x: wp.vec3,
  params: OptimizationParams,
  sdf_type1: int,
  sdf_type2: int,
  niter: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> Tuple[float, wp.vec3]:
  amin = 1e-4
  rho = 0.5
  c = 0.1
  dist = float(1e10)
  for i in range(niter):
    alpha = float(2.0)
    x2 = wp.vec3(x[0], x[1], x[2])
    x1 = params.rel_mat * x2 + params.rel_pos
    grad = compute_grad(
      type1, x1, x2, params, sdf_type1, sdf_type2, sfd_intersection, volume_data1, volume_data2, mesh_data1, mesh_data2
    )
    dist0 = clearance(
      type1,
      x1,
      x,
      params.attr1,
      params.attr2,
      sdf_type1,
      sdf_type2,
      sfd_intersection,
      volume_data1,
      volume_data2,
      mesh_data1,
      mesh_data2,
    )
    grad_dot = wp.dot(grad, grad)
    if grad_dot < 1e-12:
      return dist0, x
    wolfe = -c * alpha * grad_dot
    while True:
      alpha *= rho
      wolfe *= rho
      x = x2 - grad * alpha
      x1 = params.rel_mat * x + params.rel_pos
      dist = clearance(
        type1,
        x1,
        x,
        params.attr1,
        params.attr2,
        sdf_type1,
        sdf_type2,
        sfd_intersection,
        volume_data1,
        volume_data2,
        mesh_data1,
        mesh_data2,
      )
      if alpha <= amin or (dist - dist0) <= wolfe:
        break
    if dist > dist0:
      return dist0, x2
  return dist, x


@wp.func
def gradient_descent(
  # In:
  type1: int,
  x0_initial: wp.vec3,
  attr1: vec_pluginattr,
  attr2: vec_pluginattr,
  pos1: wp.vec3,
  rot1: wp.mat33,
  pos2: wp.vec3,
  rot2: wp.mat33,
  sdf_type1: int,
  sdf_type2: int,
  sdf_iterations: int,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> Tuple[float, wp.vec3, wp.vec3]:
  params = OptimizationParams()
  params.rel_mat = wp.transpose(rot1) * rot2
  params.rel_pos = wp.transpose(rot1) * (pos2 - pos1)
  params.attr1 = attr1
  params.attr2 = attr2
  dist, x = gradient_step(
    type1, x0_initial, params, sdf_type1, sdf_type2, sdf_iterations, False, volume_data1, volume_data2, mesh_data1, mesh_data2
  )
  dist, x = gradient_step(type1, x, params, sdf_type1, sdf_type2, 1, True, volume_data1, volume_data2, mesh_data1, mesh_data2)
  x_1 = params.rel_mat * x + params.rel_pos
  grad1 = sdf_grad(type1, x_1, params.attr1, sdf_type1, volume_data1, mesh_data1)
  grad1 = wp.transpose(params.rel_mat) * grad1
  grad1 = wp.normalize(grad1)
  grad2 = sdf_grad(GeomType.SDF, x, params.attr2, sdf_type2, volume_data2, mesh_data2)
  grad2 = wp.normalize(grad2)
  n = grad1 - grad2
  n = wp.normalize(n)
  pos = rot2 * x + pos2
  n = rot2 * n
  pos3 = pos - n * dist / 2.0
  return dist, pos3, n


@wp.kernel
def _sdf_narrowphase(
  # Model:
  nmeshface: int,
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  geom_type: wp.array[int],
  geom_condim: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_aabb: wp.array3d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  geom_adhesion: wp.array2d[float],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_octadr: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
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
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  geom_plugin_index: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  sdf_initpoints: int,
  sdf_iterations: int,
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
  i, contact_tid = wp.tid()
  if i >= sdf_initpoints:
    return
  if contact_tid >= ncollision_in[0]:
    return
  geoms = collision_pair_in[contact_tid]
  g2 = geoms[1]
  type2 = geom_type[g2]
  if type2 != GeomType.SDF:
    return
  worldid = collision_worldid_in[contact_tid]
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
    contact_tid,
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

  aabb_id = worldid % geom_aabb.shape[0]
  g1 = geoms[0]
  type1 = geom_type[g1]

  g1_plugin = geom_plugin_index[g1]
  g2_plugin = geom_plugin_index[g2]

  g1_to_g2_rot = wp.transpose(geom1.rot) * geom2.rot
  g1_to_g2_pos = wp.transpose(geom1.rot) * (geom2.pos - geom1.pos)
  aabb_pos = geom_aabb[aabb_id, g1, 0]
  aabb_size = geom_aabb[aabb_id, g1, 1]
  identity = wp.identity(3, dtype=float)
  aabb1 = transform_aabb(aabb_pos, aabb_size, wp.vec3(0.0), identity)
  aabb_pos = geom_aabb[aabb_id, g2, 0]
  aabb_size = geom_aabb[aabb_id, g2, 1]
  aabb2 = transform_aabb(aabb_pos, aabb_size, g1_to_g2_pos, g1_to_g2_rot)
  aabb_intersection = AABB()
  aabb_intersection.min = wp.max(aabb1.min, aabb2.min)
  aabb_intersection.max = wp.min(aabb1.max, aabb2.max)
  # A broadphase candidate gap can admit pairs with no geometric intersection.
  if (
    aabb_intersection.min[0] > aabb_intersection.max[0]
    or aabb_intersection.min[1] > aabb_intersection.max[1]
    or aabb_intersection.min[2] > aabb_intersection.max[2]
  ):
    return

  pos2 = geom2.pos
  rot2 = geom2.rot
  pos1 = geom1.pos
  rot1 = geom1.rot

  dataid_setid = worldid % geom_dataid.shape[0]

  attr1, g1_plugin_id, volume_data1, mesh_data1 = get_sdf_params(
    oct_child,
    oct_aabb,
    oct_coeff,
    mesh_octadr,
    plugin,
    plugin_attr,
    type1,
    geom1.size,
    g1_plugin,
    geom_dataid[dataid_setid, g1],
  )

  attr2, g2_plugin_id, volume_data2, mesh_data2 = get_sdf_params(
    oct_child,
    oct_aabb,
    oct_coeff,
    mesh_octadr,
    plugin,
    plugin_attr,
    type2,
    geom2.size,
    g2_plugin,
    geom_dataid[dataid_setid, g2],
  )

  mesh_data1.nmeshface = nmeshface
  mesh_data1.mesh_vertadr = mesh_vertadr
  mesh_data1.mesh_vert = mesh_vert
  mesh_data1.mesh_faceadr = mesh_faceadr
  mesh_data1.mesh_face = mesh_face
  mesh_data1.data_id = geom_dataid[dataid_setid, g1]
  mesh_data1.pos = geom1.pos
  mesh_data1.mat = geom1.rot
  mesh_data1.size = geom1.size
  mesh_data1.pnt = wp.vec3(-1.0)
  mesh_data1.vec = wp.vec3(0.0)
  mesh_data1.valid = True

  mesh_data2.nmeshface = nmeshface
  mesh_data2.mesh_vertadr = mesh_vertadr
  mesh_data2.mesh_vert = mesh_vert
  mesh_data2.mesh_faceadr = mesh_faceadr
  mesh_data2.mesh_face = mesh_face
  mesh_data2.data_id = geom_dataid[dataid_setid, g2]
  mesh_data2.pos = geom2.pos
  mesh_data2.mat = geom2.rot
  mesh_data2.size = geom2.size
  mesh_data2.pnt = wp.vec3(-1.0)
  mesh_data2.vec = wp.vec3(0.0)
  mesh_data2.valid = True

  x_g2 = wp.vec3(
    aabb_intersection.min[0] + (aabb_intersection.max[0] - aabb_intersection.min[0]) * halton(i, 2),
    aabb_intersection.min[1] + (aabb_intersection.max[1] - aabb_intersection.min[1]) * halton(i, 3),
    aabb_intersection.min[2] + (aabb_intersection.max[2] - aabb_intersection.min[2]) * halton(i, 5),
  )
  x = geom1.rot * x_g2 + geom1.pos
  x0_initial = wp.transpose(rot2) * (x - pos2)
  dist, pos, n = gradient_descent(
    type1,
    x0_initial,
    attr1,
    attr2,
    pos1,
    rot1,
    pos2,
    rot2,
    g1_plugin_id,
    g2_plugin_id,
    sdf_iterations,
    volume_data1,
    volume_data2,
    mesh_data1,
    mesh_data2,
  )
  write_contact(
    naconmax_in,
    0,
    dist,
    pos,
    make_frame(n),
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    adhesion,
    geoms,
    collision_pairid_in[contact_tid],
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


@event_scope
def sdf_narrowphase(m: Model, d: Data, ctx: CollisionContext):
  wp.launch(
    _sdf_narrowphase,
    dim=(m.opt.sdf_initpoints, d.naconmax),
    inputs=[
      m.nmeshface,
      m.oct_child,
      m.oct_aabb,
      m.oct_coeff,
      m.geom_type,
      m.geom_condim,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_aabb,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.geom_adhesion,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_faceadr,
      m.mesh_octadr,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_face,
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
      m.plugin,
      m.plugin_attr,
      m.geom_plugin_index,
      d.geom_xpos,
      d.geom_xmat,
      d.naconmax,
      d.ncollision,
      ctx.collision_pair,
      ctx.collision_pairid,
      ctx.collision_worldid,
      m.opt.sdf_initpoints,
      m.opt.sdf_iterations,
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
