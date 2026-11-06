# Copyright 2026 DeepMind Technologies Limited
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

"""DO NOT EDIT. This file is auto-generated."""
import dataclasses
import functools
import jax
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi
import mujoco.mjx.third_party.mujoco_warp as mjwarp
from mujoco.mjx.third_party.mujoco_warp._src import types as mjwp_types
import warp as wp


_m = mjwarp.Model(
    **{f.name: None for f in dataclasses.fields(mjwarp.Model) if f.init}
)
_d = mjwarp.Data(
    **{f.name: None for f in dataclasses.fields(mjwarp.Data) if f.init}
)
_o = mjwarp.Option(
    **{f.name: None for f in dataclasses.fields(mjwarp.Option) if f.init}
)
_c = mjwarp.Contact(
    **{f.name: None for f in dataclasses.fields(mjwarp.Contact) if f.init}
)
_e = mjwarp.Constraint(
    **{f.name: None for f in dataclasses.fields(mjwarp.Constraint) if f.init}
)
_cb = mjwp_types.Callback(
    **{f.name: None for f in dataclasses.fields(mjwp_types.Callback) if f.init}
)

@ffi.format_args_for_warp
def _collision_shim(
    # Model
    nworld: int,
    block_dim: mjwp_types.BlockDim,
    body_weldid: wp.array[int],
    flex_activelayers: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_condim: wp.array[int],
    flex_contype: wp.array[int],
    flex_dim: wp.array[int],
    flex_elem: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elemflexid: wp.array[int],
    flex_elemlayer: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_friction: wp.array[wp.vec3],
    flex_gap: wp.array[float],
    flex_interp: wp.array[int],
    flex_margin: wp.array[float],
    flex_passive: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_priority: wp.array[int],
    flex_radius: wp.array[float],
    flex_rod_box_size: wp.array[wp.vec2],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rod_is_box: wp.array[bool],
    flex_rod_root_clamped: wp.array[bool],
    flex_rod_tail_clamped: wp.array[bool],
    flex_rodedge_jnt: wp.array[int],
    flex_rodedge_vert: wp.array[wp.vec2i],
    flex_selfcollide: wp.array[int],
    flex_solimp: wp.array[mjwp_types.vec5],
    flex_solmix: wp.array[float],
    flex_solref: wp.array[wp.vec2],
    flex_vert_rodelem: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_vertflexid: wp.array[int],
    flex_vertnum: wp.array[int],
    flg_surfacevel: bool,
    geom_aabb: wp.array3d[wp.vec3],
    geom_adhesion: wp.array2d[float],
    geom_bodyid: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_condim: wp.array[int],
    geom_contype: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_friction: wp.array2d[wp.vec3],
    geom_gap: wp.array2d[float],
    geom_margin: wp.array2d[float],
    geom_pair_type_count: tuple[int, ...],
    geom_plugin_index: wp.array[int],
    geom_priority: wp.array[int],
    geom_rbound: wp.array2d[float],
    geom_size: wp.array2d[wp.vec3],
    geom_solimp: wp.array2d[mjwp_types.vec5],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_surfacevel: wp.array2d[mjwp_types.vec6],
    geom_type: wp.array[int],
    has_1d_flex: bool,
    has_3d_flex: bool,
    has_ellipsoid_geom: bool,
    has_flex_selfcollide: bool,
    has_plane_geom: bool,
    has_sdf_geom: bool,
    hfield_adr: wp.array[int],
    hfield_data: wp.array[float],
    hfield_ncol: wp.array[int],
    hfield_nrow: wp.array[int],
    hfield_size: wp.array[wp.vec4],
    jnt_qposadr: wp.array[int],
    mesh_face: wp.array[wp.vec3i],
    mesh_faceadr: wp.array[int],
    mesh_graph: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_octadr: wp.array[int],
    mesh_polyadr: wp.array[int],
    mesh_polymap: wp.array[int],
    mesh_polymapadr: wp.array[int],
    mesh_polymapnum: wp.array[int],
    mesh_polynormal: wp.array[wp.vec3],
    mesh_polynum: wp.array[int],
    mesh_polyvert: wp.array[int],
    mesh_polyvertadr: wp.array[int],
    mesh_polyvertnum: wp.array[int],
    mesh_pos: wp.array[wp.vec3],
    mesh_vert: wp.array[wp.vec3],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    nbody: int,
    nflex: int,
    nflexelem: int,
    nflexvert: int,
    ngeom: int,
    nmesh: int,
    nmeshdegmax: int,
    nmeshface: int,
    npair: int,
    npolygonmax: int,
    nxn_geom_pair_filtered: wp.array[wp.vec2i],
    nxn_pairid: wp.array[wp.vec2i],
    nxn_pairid_filtered: wp.array[wp.vec2i],
    oct_aabb: wp.array2d[wp.vec3],
    oct_child: wp.array[mjwp_types.vec8i],
    oct_coeff: wp.array[mjwp_types.vec8],
    pair_adhesion: wp.array2d[float],
    pair_dim: wp.array[int],
    pair_friction: wp.array2d[mjwp_types.vec5],
    pair_gap: wp.array2d[float],
    pair_geom1: wp.array[int],
    pair_geom2: wp.array[int],
    pair_margin: wp.array2d[float],
    pair_solimp: wp.array2d[mjwp_types.vec5],
    pair_solref: wp.array2d[wp.vec2],
    pair_solreffriction: wp.array2d[wp.vec2],
    plugin: wp.array[int],
    plugin_attr: wp.array[mjwp_types.vec_pluginattr],
    qpos0: wp.array2d[float],
    opt__broadphase: int,
    opt__broadphase_filter: int,
    opt__ccd_iterations: int,
    opt__ccd_tolerance: wp.array[float],
    opt__deterministic: int,
    opt__disableflags: int,
    opt__enableflags: int,
    opt__graph_conditional: bool,
    opt__integrator: int,
    opt__sdf_initpoints: int,
    opt__sdf_iterations: int,
    opt__warn_overflow: int,
    # Data
    naccdmax: int,
    naconmax: int,
    body_awake: wp.array2d[int],
    flex_aabb_max: wp.array2d[wp.vec3],
    flex_aabb_min: wp.array2d[wp.vec3],
    flex_rod_frame0_loc: wp.array3d[wp.vec3],
    flex_rod_frame0_ref: wp.array3d[wp.vec3],
    flex_rod_tail_loc: wp.array3d[wp.vec3],
    flexvert_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    geom_xpos: wp.array2d[wp.vec3],
    nacon: wp.array[int],
    ncollision: wp.array[int],
    overflow: wp.array[int],
    plastic: wp.array2d[float],
    qpos: wp.array2d[float],
    xmat: wp.array2d[wp.mat33],
    contact__adhesion: wp.array[float],
    contact__dim: wp.array[int],
    contact__dist: wp.array[float],
    contact__efc_address: wp.array2d[int],
    contact__elem: wp.array[wp.vec2i],
    contact__flex: wp.array[wp.vec2i],
    contact__frame: wp.array[wp.mat33],
    contact__friction: wp.array[mjwp_types.vec5],
    contact__geom: wp.array[wp.vec2i],
    contact__geomcollisionid: wp.array[int],
    contact__includemargin: wp.array[float],
    contact__pos: wp.array[wp.vec3],
    contact__solimp: wp.array[mjwp_types.vec5],
    contact__solref: wp.array[wp.vec2],
    contact__solreffriction: wp.array[wp.vec2],
    contact__surfacevel: wp.array[wp.vec3],
    contact__type: wp.array[int],
    contact__vert: wp.array[wp.vec2i],
    contact__worldid: wp.array[int],
):
  _m.opt = _o
  _m.callback = _cb
  _d.efc = _e
  _d.contact = _c
  _m.block_dim = block_dim
  _m.body_weldid = body_weldid
  _m.flex_activelayers = flex_activelayers
  _m.flex_conaffinity = flex_conaffinity
  _m.flex_condim = flex_condim
  _m.flex_contype = flex_contype
  _m.flex_dim = flex_dim
  _m.flex_elem = flex_elem
  _m.flex_elem_rodedge = flex_elem_rodedge
  _m.flex_elemadr = flex_elemadr
  _m.flex_elemdataadr = flex_elemdataadr
  _m.flex_elemflexid = flex_elemflexid
  _m.flex_elemlayer = flex_elemlayer
  _m.flex_elemnum = flex_elemnum
  _m.flex_friction = flex_friction
  _m.flex_gap = flex_gap
  _m.flex_interp = flex_interp
  _m.flex_margin = flex_margin
  _m.flex_passive = flex_passive
  _m.flex_plasticadr = flex_plasticadr
  _m.flex_priority = flex_priority
  _m.flex_radius = flex_radius
  _m.flex_rod_box_size = flex_rod_box_size
  _m.flex_rod_edgeadr = flex_rod_edgeadr
  _m.flex_rod_edgenum = flex_rod_edgenum
  _m.flex_rod_is_box = flex_rod_is_box
  _m.flex_rod_root_clamped = flex_rod_root_clamped
  _m.flex_rod_tail_clamped = flex_rod_tail_clamped
  _m.flex_rodedge_jnt = flex_rodedge_jnt
  _m.flex_rodedge_vert = flex_rodedge_vert
  _m.flex_selfcollide = flex_selfcollide
  _m.flex_solimp = flex_solimp
  _m.flex_solmix = flex_solmix
  _m.flex_solref = flex_solref
  _m.flex_vert_rodelem = flex_vert_rodelem
  _m.flex_vertadr = flex_vertadr
  _m.flex_vertbodyid = flex_vertbodyid
  _m.flex_vertflexid = flex_vertflexid
  _m.flex_vertnum = flex_vertnum
  _m.flg_surfacevel = flg_surfacevel
  _m.geom_aabb = geom_aabb
  _m.geom_adhesion = geom_adhesion
  _m.geom_bodyid = geom_bodyid
  _m.geom_conaffinity = geom_conaffinity
  _m.geom_condim = geom_condim
  _m.geom_contype = geom_contype
  _m.geom_dataid = geom_dataid
  _m.geom_friction = geom_friction
  _m.geom_gap = geom_gap
  _m.geom_margin = geom_margin
  _m.geom_pair_type_count = geom_pair_type_count
  _m.geom_plugin_index = geom_plugin_index
  _m.geom_priority = geom_priority
  _m.geom_rbound = geom_rbound
  _m.geom_size = geom_size
  _m.geom_solimp = geom_solimp
  _m.geom_solmix = geom_solmix
  _m.geom_solref = geom_solref
  _m.geom_surfacevel = geom_surfacevel
  _m.geom_type = geom_type
  _m.has_1d_flex = has_1d_flex
  _m.has_3d_flex = has_3d_flex
  _m.has_ellipsoid_geom = has_ellipsoid_geom
  _m.has_flex_selfcollide = has_flex_selfcollide
  _m.has_plane_geom = has_plane_geom
  _m.has_sdf_geom = has_sdf_geom
  _m.hfield_adr = hfield_adr
  _m.hfield_data = hfield_data
  _m.hfield_ncol = hfield_ncol
  _m.hfield_nrow = hfield_nrow
  _m.hfield_size = hfield_size
  _m.jnt_qposadr = jnt_qposadr
  _m.mesh_face = mesh_face
  _m.mesh_faceadr = mesh_faceadr
  _m.mesh_graph = mesh_graph
  _m.mesh_graphadr = mesh_graphadr
  _m.mesh_octadr = mesh_octadr
  _m.mesh_polyadr = mesh_polyadr
  _m.mesh_polymap = mesh_polymap
  _m.mesh_polymapadr = mesh_polymapadr
  _m.mesh_polymapnum = mesh_polymapnum
  _m.mesh_polynormal = mesh_polynormal
  _m.mesh_polynum = mesh_polynum
  _m.mesh_polyvert = mesh_polyvert
  _m.mesh_polyvertadr = mesh_polyvertadr
  _m.mesh_polyvertnum = mesh_polyvertnum
  _m.mesh_pos = mesh_pos
  _m.mesh_vert = mesh_vert
  _m.mesh_vertadr = mesh_vertadr
  _m.mesh_vertnum = mesh_vertnum
  _m.nbody = nbody
  _m.nflex = nflex
  _m.nflexelem = nflexelem
  _m.nflexvert = nflexvert
  _m.ngeom = ngeom
  _m.nmesh = nmesh
  _m.nmeshdegmax = nmeshdegmax
  _m.nmeshface = nmeshface
  _m.npair = npair
  _m.npolygonmax = npolygonmax
  _m.nxn_geom_pair_filtered = nxn_geom_pair_filtered
  _m.nxn_pairid = nxn_pairid
  _m.nxn_pairid_filtered = nxn_pairid_filtered
  _m.oct_aabb = oct_aabb
  _m.oct_child = oct_child
  _m.oct_coeff = oct_coeff
  _m.opt.broadphase = opt__broadphase
  _m.opt.broadphase_filter = opt__broadphase_filter
  _m.opt.ccd_iterations = opt__ccd_iterations
  _m.opt.ccd_tolerance = opt__ccd_tolerance
  _m.opt.deterministic = opt__deterministic
  _m.opt.disableflags = opt__disableflags
  _m.opt.enableflags = opt__enableflags
  _m.opt.graph_conditional = opt__graph_conditional
  _m.opt.integrator = opt__integrator
  _m.opt.sdf_initpoints = opt__sdf_initpoints
  _m.opt.sdf_iterations = opt__sdf_iterations
  _m.opt.warn_overflow = opt__warn_overflow
  _m.pair_adhesion = pair_adhesion
  _m.pair_dim = pair_dim
  _m.pair_friction = pair_friction
  _m.pair_gap = pair_gap
  _m.pair_geom1 = pair_geom1
  _m.pair_geom2 = pair_geom2
  _m.pair_margin = pair_margin
  _m.pair_solimp = pair_solimp
  _m.pair_solref = pair_solref
  _m.pair_solreffriction = pair_solreffriction
  _m.plugin = plugin
  _m.plugin_attr = plugin_attr
  _m.qpos0 = qpos0
  _d.body_awake = body_awake
  _d.contact.adhesion = contact__adhesion
  _d.contact.dim = contact__dim
  _d.contact.dist = contact__dist
  _d.contact.efc_address = contact__efc_address
  _d.contact.elem = contact__elem
  _d.contact.flex = contact__flex
  _d.contact.frame = contact__frame
  _d.contact.friction = contact__friction
  _d.contact.geom = contact__geom
  _d.contact.geomcollisionid = contact__geomcollisionid
  _d.contact.includemargin = contact__includemargin
  _d.contact.pos = contact__pos
  _d.contact.solimp = contact__solimp
  _d.contact.solref = contact__solref
  _d.contact.solreffriction = contact__solreffriction
  _d.contact.surfacevel = contact__surfacevel
  _d.contact.type = contact__type
  _d.contact.vert = contact__vert
  _d.contact.worldid = contact__worldid
  _d.flex_aabb_max = flex_aabb_max
  _d.flex_aabb_min = flex_aabb_min
  _d.flex_rod_frame0_loc = flex_rod_frame0_loc
  _d.flex_rod_frame0_ref = flex_rod_frame0_ref
  _d.flex_rod_tail_loc = flex_rod_tail_loc
  _d.flexvert_xpos = flexvert_xpos
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.naccdmax = naccdmax
  _d.nacon = nacon
  _d.naconmax = naconmax
  _d.ncollision = ncollision
  _d.overflow = overflow
  _d.plastic = plastic
  _d.qpos = qpos
  _d.xmat = xmat
  _d.nworld = nworld
  mjwarp.collision(_m, _d)


def _collision_jax_impl(m: types.Model, d: types.Data):
  output_dims = {
      'flex_aabb_max': d._impl.flex_aabb_max.shape,
      'flex_aabb_min': d._impl.flex_aabb_min.shape,
      'nacon': d._impl.nacon.shape,
      'ncollision': d._impl.ncollision.shape,
      'overflow': d._impl.overflow.shape,
      'contact__adhesion': d._impl.contact__adhesion.shape,
      'contact__dim': d._impl.contact__dim.shape,
      'contact__dist': d._impl.contact__dist.shape,
      'contact__efc_address': d._impl.contact__efc_address.shape,
      'contact__elem': d._impl.contact__elem.shape,
      'contact__flex': d._impl.contact__flex.shape,
      'contact__frame': d._impl.contact__frame.shape,
      'contact__friction': d._impl.contact__friction.shape,
      'contact__geom': d._impl.contact__geom.shape,
      'contact__geomcollisionid': d._impl.contact__geomcollisionid.shape,
      'contact__includemargin': d._impl.contact__includemargin.shape,
      'contact__pos': d._impl.contact__pos.shape,
      'contact__solimp': d._impl.contact__solimp.shape,
      'contact__solref': d._impl.contact__solref.shape,
      'contact__solreffriction': d._impl.contact__solreffriction.shape,
      'contact__surfacevel': d._impl.contact__surfacevel.shape,
      'contact__type': d._impl.contact__type.shape,
      'contact__vert': d._impl.contact__vert.shape,
      'contact__worldid': d._impl.contact__worldid.shape,
  }
  jf = ffi.jax_callable_variadic_tuple(
      _collision_shim,
      num_outputs=24,
      output_dims=output_dims,
      vmap_method=None,
      in_out_argnames=set([
          'flex_aabb_max',
          'flex_aabb_min',
          'nacon',
          'ncollision',
          'overflow',
          'contact__adhesion',
          'contact__dim',
          'contact__dist',
          'contact__efc_address',
          'contact__elem',
          'contact__flex',
          'contact__frame',
          'contact__friction',
          'contact__geom',
          'contact__geomcollisionid',
          'contact__includemargin',
          'contact__pos',
          'contact__solimp',
          'contact__solref',
          'contact__solreffriction',
          'contact__surfacevel',
          'contact__type',
          'contact__vert',
          'contact__worldid',
      ]),
      stage_in_argnames=set([
          'body_awake',
          'contact__adhesion',
          'contact__dim',
          'contact__dist',
          'contact__efc_address',
          'contact__elem',
          'contact__flex',
          'contact__frame',
          'contact__friction',
          'contact__geom',
          'contact__geomcollisionid',
          'contact__includemargin',
          'contact__pos',
          'contact__solimp',
          'contact__solref',
          'contact__solreffriction',
          'contact__surfacevel',
          'contact__type',
          'contact__vert',
          'contact__worldid',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_tail_loc',
          'flexvert_xpos',
          'geom_aabb',
          'geom_adhesion',
          'geom_friction',
          'geom_gap',
          'geom_margin',
          'geom_rbound',
          'geom_size',
          'geom_solimp',
          'geom_solmix',
          'geom_solref',
          'geom_surfacevel',
          'geom_xmat',
          'geom_xpos',
          'hfield_data',
          'nacon',
          'ncollision',
          'opt__ccd_tolerance',
          'overflow',
          'pair_adhesion',
          'pair_friction',
          'pair_gap',
          'pair_margin',
          'pair_solimp',
          'pair_solref',
          'pair_solreffriction',
          'plastic',
          'qpos',
          'qpos0',
          'xmat',
      ]),
      stage_out_argnames=set([
          'contact__adhesion',
          'contact__dim',
          'contact__dist',
          'contact__efc_address',
          'contact__elem',
          'contact__flex',
          'contact__frame',
          'contact__friction',
          'contact__geom',
          'contact__geomcollisionid',
          'contact__includemargin',
          'contact__pos',
          'contact__solimp',
          'contact__solref',
          'contact__solreffriction',
          'contact__surfacevel',
          'contact__type',
          'contact__vert',
          'contact__worldid',
          'flex_aabb_max',
          'flex_aabb_min',
          'nacon',
          'ncollision',
          'overflow',
      ]),
      graph_mode=m.opt._impl.graph_mode,
      has_side_effect=False,
  )
  out = jf(
      d.qpos.shape[0],
      m._impl.block_dim,
      m.body_weldid,
      m._impl.flex_activelayers,
      m._impl.flex_conaffinity,
      m._impl.flex_condim,
      m._impl.flex_contype,
      m._impl.flex_dim,
      m._impl.flex_elem,
      m._impl.flex_elem_rodedge,
      m._impl.flex_elemadr,
      m._impl.flex_elemdataadr,
      m._impl.flex_elemflexid,
      m._impl.flex_elemlayer,
      m._impl.flex_elemnum,
      m._impl.flex_friction,
      m._impl.flex_gap,
      m.flex_interp,
      m._impl.flex_margin,
      m._impl.flex_passive,
      m._impl.flex_plasticadr,
      m._impl.flex_priority,
      m._impl.flex_radius,
      m._impl.flex_rod_box_size,
      m._impl.flex_rod_edgeadr,
      m._impl.flex_rod_edgenum,
      m._impl.flex_rod_is_box,
      m._impl.flex_rod_root_clamped,
      m._impl.flex_rod_tail_clamped,
      m._impl.flex_rodedge_jnt,
      m._impl.flex_rodedge_vert,
      m._impl.flex_selfcollide,
      m._impl.flex_solimp,
      m._impl.flex_solmix,
      m._impl.flex_solref,
      m._impl.flex_vert_rodelem,
      m.flex_vertadr,
      m._impl.flex_vertbodyid,
      m._impl.flex_vertflexid,
      m.flex_vertnum,
      m.flg_surfacevel,
      m.geom_aabb,
      m._impl.geom_adhesion,
      m.geom_bodyid,
      m.geom_conaffinity,
      m.geom_condim,
      m.geom_contype,
      jax.numpy.expand_dims(m.geom_dataid, 0),
      m.geom_friction,
      m.geom_gap,
      m.geom_margin,
      m._impl.geom_pair_type_count,
      m._impl.geom_plugin_index,
      m.geom_priority,
      m.geom_rbound,
      m.geom_size,
      m.geom_solimp,
      m.geom_solmix,
      m.geom_solref,
      m._impl.geom_surfacevel,
      m.geom_type,
      m._impl.has_1d_flex,
      m._impl.has_3d_flex,
      m._impl.has_ellipsoid_geom,
      m._impl.has_flex_selfcollide,
      m._impl.has_plane_geom,
      m._impl.has_sdf_geom,
      m.hfield_adr,
      m.hfield_data,
      m.hfield_ncol,
      m.hfield_nrow,
      m.hfield_size,
      m.jnt_qposadr,
      m.mesh_face,
      m.mesh_faceadr,
      m.mesh_graph,
      m.mesh_graphadr,
      m.mesh_octadr,
      m._impl.mesh_polyadr,
      m._impl.mesh_polymap,
      m._impl.mesh_polymapadr,
      m._impl.mesh_polymapnum,
      m._impl.mesh_polynormal,
      m._impl.mesh_polynum,
      m._impl.mesh_polyvert,
      m._impl.mesh_polyvertadr,
      m._impl.mesh_polyvertnum,
      m.mesh_pos,
      m.mesh_vert,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.nbody,
      m.nflex,
      m._impl.nflexelem,
      m._impl.nflexvert,
      m.ngeom,
      m.nmesh,
      m._impl.nmeshdegmax,
      m.nmeshface,
      m.npair,
      m._impl.npolygonmax,
      m._impl.nxn_geom_pair_filtered,
      m._impl.nxn_pairid,
      m._impl.nxn_pairid_filtered,
      m._impl.oct_aabb,
      m._impl.oct_child,
      m._impl.oct_coeff,
      m._impl.pair_adhesion,
      m.pair_dim,
      m.pair_friction,
      m.pair_gap,
      m.pair_geom1,
      m.pair_geom2,
      m.pair_margin,
      m.pair_solimp,
      m.pair_solref,
      m.pair_solreffriction,
      m._impl.plugin,
      m._impl.plugin_attr,
      m.qpos0,
      m.opt._impl.broadphase,
      m.opt._impl.broadphase_filter,
      m.opt._impl.ccd_iterations,
      m.opt._impl.ccd_tolerance,
      m.opt._impl.deterministic,
      m.opt.disableflags,
      m.opt.enableflags,
      m.opt._impl.graph_conditional,
      m.opt.integrator,
      m.opt._impl.sdf_initpoints,
      m.opt._impl.sdf_iterations,
      m.opt._impl.warn_overflow,
      d._impl.naccdmax,
      d._impl.naconmax,
      d._impl.body_awake,
      d._impl.flex_aabb_max,
      d._impl.flex_aabb_min,
      d._impl.flex_rod_frame0_loc,
      d._impl.flex_rod_frame0_ref,
      d._impl.flex_rod_tail_loc,
      d._impl.flexvert_xpos,
      d.geom_xmat,
      d.geom_xpos,
      d._impl.nacon,
      d._impl.ncollision,
      d._impl.overflow,
      d.plastic,
      d.qpos,
      d.xmat,
      d._impl.contact__adhesion,
      d._impl.contact__dim,
      d._impl.contact__dist,
      d._impl.contact__efc_address,
      d._impl.contact__elem,
      d._impl.contact__flex,
      d._impl.contact__frame,
      d._impl.contact__friction,
      d._impl.contact__geom,
      d._impl.contact__geomcollisionid,
      d._impl.contact__includemargin,
      d._impl.contact__pos,
      d._impl.contact__solimp,
      d._impl.contact__solref,
      d._impl.contact__solreffriction,
      d._impl.contact__surfacevel,
      d._impl.contact__type,
      d._impl.contact__vert,
      d._impl.contact__worldid,
  )
  d = d.tree_replace({
      '_impl.flex_aabb_max': out[0],
      '_impl.flex_aabb_min': out[1],
      '_impl.nacon': out[2],
      '_impl.ncollision': out[3],
      '_impl.overflow': out[4],
      '_impl.contact__adhesion': out[5],
      '_impl.contact__dim': out[6],
      '_impl.contact__dist': out[7],
      '_impl.contact__efc_address': out[8],
      '_impl.contact__elem': out[9],
      '_impl.contact__flex': out[10],
      '_impl.contact__frame': out[11],
      '_impl.contact__friction': out[12],
      '_impl.contact__geom': out[13],
      '_impl.contact__geomcollisionid': out[14],
      '_impl.contact__includemargin': out[15],
      '_impl.contact__pos': out[16],
      '_impl.contact__solimp': out[17],
      '_impl.contact__solref': out[18],
      '_impl.contact__solreffriction': out[19],
      '_impl.contact__surfacevel': out[20],
      '_impl.contact__type': out[21],
      '_impl.contact__vert': out[22],
      '_impl.contact__worldid': out[23],
  })
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def collision(m: types.Model, d: types.Data):
  return _collision_jax_impl(m, d)


@collision.def_vmap
@ffi.marshal_custom_vmap
def collision_vmap(unused_axis_size, is_batched, m: types.Model, d: types.Data):
  d = collision(m, d)
  return d, is_batched[1]
