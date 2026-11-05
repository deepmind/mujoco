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
def _forward_shim(
    # Model
    nworld: int,
    M_colind: wp.array[int],
    M_elemid: wp.array2d[int],
    M_fullm_i: wp.array[int],
    M_fullm_j: wp.array[int],
    M_hinit_i: wp.array[int],
    M_mulm_col: wp.array[int],
    M_mulm_madr: wp.array[int],
    M_mulm_rowadr: wp.array[int],
    M_rowadr: wp.array[int],
    M_rownnz: wp.array[int],
    M_tiles: tuple[mjwp_types.TileSet, ...],
    actuator_acc0: wp.array2d[float],
    actuator_actadr: wp.array[int],
    actuator_actearly: wp.array[bool],
    actuator_actlimited: wp.array[bool],
    actuator_actnum: wp.array[int],
    actuator_actrange: wp.array2d[wp.vec2],
    actuator_armature: wp.array2d[float],
    actuator_biasprm: wp.array2d[mjwp_types.vec10],
    actuator_biastype: wp.array[int],
    actuator_cranklength: wp.array2d[float],
    actuator_ctrladr: wp.array[int],
    actuator_ctrllimited: wp.array[bool],
    actuator_ctrlnum: wp.array[int],
    actuator_ctrlrange: wp.array2d[wp.vec2],
    actuator_ctrlspec: wp.array[int],
    actuator_damping: wp.array2d[float],
    actuator_dampingpoly: wp.array2d[wp.vec2],
    actuator_delay: wp.array2d[float],
    actuator_dynprm: wp.array2d[mjwp_types.vec10],
    actuator_dyntype: wp.array[int],
    actuator_forcelimited: wp.array[bool],
    actuator_forcerange: wp.array2d[wp.vec2],
    actuator_gainprm: wp.array2d[mjwp_types.vec10],
    actuator_gaintype: wp.array[int],
    actuator_gear: wp.array2d[wp.spatial_vector],
    actuator_history: wp.array2d[wp.vec2i],
    actuator_historyadr: wp.array2d[int],
    actuator_lengthrange: wp.array2d[wp.vec2],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_trntype: wp.array[int],
    actuator_trntype_body_adr: wp.array[int],
    block_dim: mjwp_types.BlockDim,
    body_branch_start: wp.array[int],
    body_branches: wp.array[int],
    body_dofadr: wp.array[int],
    body_dofnum: wp.array[int],
    body_fluid_box_adr: wp.array[int],
    body_fluid_ellipsoid: wp.array[bool],
    body_fluid_ellipsoid_adr: wp.array[int],
    body_freeadr: wp.array[int],
    body_geomadr: wp.array[int],
    body_geomnum: wp.array[int],
    body_gravcomp: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    body_invweight0: wp.array2d[wp.vec2],
    body_ipos: wp.array2d[wp.vec3],
    body_iquat: wp.array2d[wp.quat],
    body_is_free: wp.array[bool],
    body_isdofancestor: wp.array2d[int],
    body_jntadr: wp.array[int],
    body_jntnum: wp.array[int],
    body_mass: wp.array2d[float],
    body_mocapid: wp.array[int],
    body_parentid: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    body_rootid: wp.array[int],
    body_simple: wp.array[int],
    body_subtreemass: wp.array2d[float],
    body_subtreenum: wp.array[int],
    body_tree: tuple[wp.array[int], ...],
    body_treeid: wp.array[int],
    body_weldid: wp.array[int],
    cam_bodyid: wp.array[int],
    cam_fovy: wp.array2d[float],
    cam_intrinsic: wp.array2d[wp.vec4],
    cam_mat0: wp.array2d[wp.mat33],
    cam_mode: wp.array[int],
    cam_pos: wp.array2d[wp.vec3],
    cam_pos0: wp.array2d[wp.vec3],
    cam_poscom0: wp.array2d[wp.vec3],
    cam_projection: wp.array[int],
    cam_quat: wp.array2d[wp.quat],
    cam_resolution: wp.array[wp.vec2i],
    cam_sensorsize: wp.array[wp.vec2],
    cam_targetbodyid: wp.array[int],
    dof_armature: wp.array2d[float],
    dof_bodyid: wp.array[int],
    dof_damping: wp.array2d[float],
    dof_dampingpoly: wp.array2d[wp.vec2],
    dof_frictionloss: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    dof_jntid: wp.array[int],
    dof_length: wp.array[float],
    dof_parentid: wp.array[int],
    dof_solimp: wp.array2d[mjwp_types.vec5],
    dof_solref: wp.array2d[wp.vec2],
    dof_treeid: wp.array[int],
    dof_tri_col: wp.array[int],
    dof_tri_row: wp.array[int],
    efm0_L: wp.array[float],
    efm0_L_colind: wp.array[int],
    efm0_L_rowadr: wp.array[int],
    efm0_L_rownnz: wp.array[int],
    efm0_active: bool,
    efm0_dofid: wp.array[int],
    efm_K_colind: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_rownnz: wp.array[int],
    efm_con_max_dof: int,
    efm_dofblk: wp.array[int],
    efm_dofid: wp.array[int],
    eq_connect_adr: wp.array[int],
    eq_data: wp.array2d[mjwp_types.vec12],
    eq_flex_adr: wp.array[int],
    eq_flexstrain_adr: wp.array[int],
    eq_jnt_adr: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_objtype: wp.array[int],
    eq_solimp: wp.array2d[mjwp_types.vec5],
    eq_solref: wp.array2d[wp.vec2],
    eq_ten_adr: wp.array[int],
    eq_type: wp.array[int],
    eq_wld_adr: wp.array[int],
    flex_activelayers: wp.array[int],
    flex_bend_interp_map: wp.array[wp.vec2i],
    flex_bending: wp.array[float],
    flex_bendingadr: wp.array[int],
    flex_cell_map: wp.array[wp.vec4i],
    flex_cellnum: wp.array[wp.vec3i],
    flex_centered: wp.array[bool],
    flex_conaffinity: wp.array[int],
    flex_condim: wp.array[int],
    flex_contype: wp.array[int],
    flex_damping: wp.array[float],
    flex_dim: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_edgeadr: wp.array[int],
    flex_edgedamping: wp.array[float],
    flex_edgeequality: wp.array[int],
    flex_edgeflap: wp.array[wp.vec2i],
    flex_edgeflexid: wp.array[int],
    flex_edgenum: wp.array[int],
    flex_edgestiffness: wp.array[float],
    flex_eff_possible: wp.array[bool],
    flex_elem: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elemedge: wp.array[int],
    flex_elemedgeadr: wp.array[int],
    flex_elemflexid: wp.array[int],
    flex_elemlayer: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_face: wp.array2d[int],
    flex_face_map: wp.array[wp.vec2i],
    flex_faceadr: wp.array[int],
    flex_friction: wp.array[wp.vec3],
    flex_gap: wp.array[float],
    flex_interp: wp.array[int],
    flex_interp_assemblable: bool,
    flex_margin: wp.array[float],
    flex_node: wp.array[wp.vec3],
    flex_node0: wp.array[wp.vec3],
    flex_nodeadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_nodenum: wp.array[int],
    flex_passive: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_plasticmode: wp.array[int],
    flex_priority: wp.array[int],
    flex_radius: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_rod_box_size: wp.array[wp.vec2],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_hingeadr: wp.array[int],
    flex_rod_is_box: wp.array[bool],
    flex_rod_is_chain: wp.array[bool],
    flex_rod_root_clamped: wp.array[bool],
    flex_rod_tail_clamped: wp.array[bool],
    flex_rodedge_jnt: wp.array[int],
    flex_rodedge_vert: wp.array[wp.vec2i],
    flex_rodhinge_map: wp.array[wp.vec2i],
    flex_selfcollide: wp.array[int],
    flex_simple: wp.array[bool],
    flex_size: wp.array[wp.vec3],
    flex_solimp: wp.array[mjwp_types.vec5],
    flex_solmix: wp.array[float],
    flex_solref: wp.array[wp.vec2],
    flex_stiffness: wp.array[float],
    flex_stiffnessadr: wp.array[int],
    flex_vert: wp.array[wp.vec3],
    flex_vert0: wp.array[wp.vec3],
    flex_vert_rodelem: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_vertcell: wp.array[int],
    flex_vertflexid: wp.array[int],
    flex_vertnum: wp.array[int],
    flexedge_J_colind: wp.array[int],
    flexedge_J_rowadr: wp.array[int],
    flexedge_J_rownnz: wp.array[int],
    flexedge_invweight0: wp.array[float],
    flexedge_length0: wp.array[float],
    flexedge_rigid: wp.array[bool],
    flexstrain_J_colind: wp.array[int],
    flexstrain_J_rowadr: wp.array[int],
    flexstrain_J_rownnz: wp.array[int],
    flg_adhesion: bool,
    flg_surfacevel: bool,
    geom_aabb: wp.array3d[wp.vec3],
    geom_adhesion: wp.array2d[float],
    geom_bodyid: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_condim: wp.array[int],
    geom_contype: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_fluid: wp.array2d[float],
    geom_friction: wp.array2d[wp.vec3],
    geom_gap: wp.array2d[float],
    geom_group: wp.array[int],
    geom_margin: wp.array2d[float],
    geom_matid: wp.array2d[int],
    geom_pair_type_count: tuple[int, ...],
    geom_plugin_index: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_priority: wp.array[int],
    geom_quat: wp.array2d[wp.quat],
    geom_rbound: wp.array2d[float],
    geom_rgba: wp.array2d[wp.vec4],
    geom_size: wp.array2d[wp.vec3],
    geom_solimp: wp.array2d[mjwp_types.vec5],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_surfacevel: wp.array2d[mjwp_types.vec6],
    geom_type: wp.array[int],
    has_1d_flex: bool,
    has_3d_flex: bool,
    has_efm_actuator: bool,
    has_ellipsoid_geom: bool,
    has_flex_edge_stiff_damp: bool,
    has_flex_passive: bool,
    has_flex_selfcollide: bool,
    has_flex_snh: bool,
    has_fluid: bool,
    has_non_simple_flex: bool,
    has_plane_geom: bool,
    has_sdf_geom: bool,
    has_tendon_damping: bool,
    has_tendon_stiffness: bool,
    has_unsupported_flex_interp: bool,
    hfield_adr: wp.array[int],
    hfield_data: wp.array[float],
    hfield_ncol: wp.array[int],
    hfield_nrow: wp.array[int],
    hfield_size: wp.array[wp.vec4],
    is_sparse: bool,
    jnt_actfrclimited: wp.array[bool],
    jnt_actfrcrange: wp.array2d[wp.vec2],
    jnt_actgravcomp: wp.array[int],
    jnt_actuatorid: wp.array[int],
    jnt_axis: wp.array2d[wp.vec3],
    jnt_bodyid: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_limited_ball_adr: wp.array[int],
    jnt_limited_slide_hinge_adr: wp.array[int],
    jnt_margin: wp.array2d[float],
    jnt_plasticadr: wp.array[int],
    jnt_pos: wp.array2d[wp.vec3],
    jnt_qposadr: wp.array[int],
    jnt_range: wp.array2d[wp.vec2],
    jnt_solimp: wp.array2d[mjwp_types.vec5],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_stiffness: wp.array2d[float],
    jnt_stiffnesspoly: wp.array2d[wp.vec2],
    jnt_type: wp.array[int],
    light_bodyid: wp.array[int],
    light_dir: wp.array2d[wp.vec3],
    light_dir0: wp.array2d[wp.vec3],
    light_mode: wp.array[int],
    light_pos: wp.array2d[wp.vec3],
    light_pos0: wp.array2d[wp.vec3],
    light_poscom0: wp.array2d[wp.vec3],
    light_targetbodyid: wp.array[int],
    mat_rgba: wp.array2d[wp.vec4],
    max_ten_J_rownnz: int,
    mesh_face: wp.array[wp.vec3i],
    mesh_faceadr: wp.array[int],
    mesh_graph: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_normal: wp.array[wp.vec3],
    mesh_normaladr: wp.array[int],
    mesh_normalnum: wp.array[int],
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
    mesh_quat: wp.array[wp.quat],
    mesh_vert: wp.array[wp.vec3],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    nC: int,
    nJten: int,
    na: int,
    nacttrnbody: int,
    nactuator: int,
    nbody: int,
    nbranch: int,
    ncam: int,
    nefm0dof: int,
    nefmK: int,
    nefmdof: int,
    neq: int,
    nflex: int,
    nflexbend_interp: int,
    nflexedge: int,
    nflexelem: int,
    nflexface: int,
    nflexintcell: int,
    nflexnode: int,
    nflexrodedge: int,
    nflexrodhinge: int,
    nflexvert: int,
    ngeom: int,
    nhistory: int,
    njnt: int,
    nlight: int,
    nmaxcondim: int,
    nmaxpyramid: int,
    nmesh: int,
    nmeshdegmax: int,
    nmeshface: int,
    npair: int,
    npolygonmax: int,
    nrangefinder: int,
    nsensorcollision: int,
    nsensortaxel: int,
    nsite: int,
    ntactileweld: int,
    ntendon: int,
    ntree: int,
    nu: int,
    nv: int,
    nv_pad: int,
    nwrap: int,
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
    qLD_all_updates: wp.array[wp.vec3i],
    qLD_block_adr: wp.array[int],
    qLD_block_total: int,
    qLD_level_offsets: wp.array[int],
    qLD_updates: tuple[wp.array[wp.vec3i], ...],
    qpos0: wp.array2d[float],
    qpos_spring: wp.array2d[float],
    rangefinder_sensor_adr: wp.array[int],
    sensor_acc_adr: wp.array[int],
    sensor_adr: wp.array[int],
    sensor_contact_acc_adr: wp.array[int],
    sensor_contact_pos_adr: wp.array[int],
    sensor_cutoff: wp.array[float],
    sensor_datatype: wp.array[int],
    sensor_delay: wp.array2d[float],
    sensor_dim: wp.array[int],
    sensor_e_kinetic: bool,
    sensor_e_potential: bool,
    sensor_history: wp.array2d[wp.vec2i],
    sensor_historyadr: wp.array2d[int],
    sensor_interval: wp.array2d[wp.vec2],
    sensor_intprm: wp.array2d[int],
    sensor_limitfrc_adr: wp.array[int],
    sensor_limitpos_adr: wp.array[int],
    sensor_limitvel_adr: wp.array[int],
    sensor_objid: wp.array[int],
    sensor_objtype: wp.array[int],
    sensor_pos_adr: wp.array[int],
    sensor_rangefinder_adr: wp.array[int],
    sensor_rangefinder_bodyid: wp.array[int],
    sensor_refid: wp.array[int],
    sensor_reftype: wp.array[int],
    sensor_rne_postconstraint: bool,
    sensor_stage_acc_adr: wp.array[int],
    sensor_stage_pos_adr: wp.array[int],
    sensor_stage_vel_adr: wp.array[int],
    sensor_subtree_vel: bool,
    sensor_tendonactfrc_adr: wp.array[int],
    sensor_touch_adr: wp.array[int],
    sensor_type: wp.array[int],
    sensor_vel_adr: wp.array[int],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    site_quat: wp.array2d[wp.quat],
    site_size: wp.array[wp.vec3],
    site_type: wp.array[int],
    taxel_sensorid: wp.array[int],
    taxel_vertadr: wp.array[int],
    ten_J_colind: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_rownnz: wp.array[int],
    tendon_actfrclimited: wp.array[bool],
    tendon_actfrcrange: wp.array2d[wp.vec2],
    tendon_actuatorid: wp.array[int],
    tendon_adr: wp.array[int],
    tendon_armature: wp.array2d[float],
    tendon_damping: wp.array2d[float],
    tendon_dampingpoly: wp.array2d[wp.vec2],
    tendon_frictionloss: wp.array2d[float],
    tendon_geom_adr: wp.array[int],
    tendon_invweight0: wp.array2d[float],
    tendon_jnt_adr: wp.array[int],
    tendon_length0: wp.array2d[float],
    tendon_lengthspring: wp.array2d[wp.vec2],
    tendon_limited: wp.array[int],
    tendon_limited_adr: wp.array[int],
    tendon_margin: wp.array2d[float],
    tendon_num: wp.array[int],
    tendon_plasticadr: wp.array[int],
    tendon_range: wp.array2d[wp.vec2],
    tendon_site_pair_adr: wp.array[int],
    tendon_solimp_fri: wp.array2d[mjwp_types.vec5],
    tendon_solimp_lim: wp.array2d[mjwp_types.vec5],
    tendon_solref_fri: wp.array2d[wp.vec2],
    tendon_solref_lim: wp.array2d[wp.vec2],
    tendon_stiffness: wp.array2d[float],
    tendon_stiffnesspoly: wp.array2d[wp.vec2],
    tendon_type: wp.array[int],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    tree_sleep_policy: wp.array[int],
    weld_tactile_id: wp.array[int],
    wrap_geom_adr: wp.array[int],
    wrap_jnt_adr: wp.array[int],
    wrap_objid: wp.array[int],
    wrap_prm: wp.array[float],
    wrap_pulley_scale: wp.array[float],
    wrap_site_pair_adr: wp.array[int],
    wrap_type: wp.array[int],
    opt__broadphase: int,
    opt__broadphase_filter: int,
    opt__ccd_iterations: int,
    opt__ccd_tolerance: wp.array[float],
    opt__cone: int,
    opt__contact_sensor_maxmatch: int,
    opt__density: wp.array[float],
    opt__deterministic: int,
    opt__disableflags: int,
    opt__enableflags: int,
    opt__graph_conditional: bool,
    opt__gravity: wp.array[wp.vec3],
    opt__impratio_invsqrt: wp.array[float],
    opt__integrator: int,
    opt__iterations: int,
    opt__ls_iterations: int,
    opt__ls_tolerance: wp.array[float],
    opt__magnetic: wp.array[wp.vec3],
    opt__noslip_iterations: int,
    opt__noslip_tolerance: wp.array[float],
    opt__run_collision_detection: bool,
    opt__run_rne_postconstraint: bool,
    opt__sdf_initpoints: int,
    opt__sdf_iterations: int,
    opt__solver: int,
    opt__timestep: wp.array[float],
    opt__tolerance: wp.array[float],
    opt__viscosity: wp.array[float],
    opt__warn_overflow: int,
    opt__wind: wp.array[wp.vec3],
    # Data
    naccdmax: int,
    naconmax: int,
    njmax: int,
    njmax_nnz: int,
    nvmax: int,
    nvmax_pad: int,
    M: wp.array2d[float],
    act: wp.array2d[float],
    act_dot: wp.array2d[float],
    actuator_force: wp.array2d[float],
    actuator_length: wp.array2d[float],
    actuator_moment: wp.array2d[float],
    actuator_velocity: wp.array2d[float],
    body_awake: wp.array2d[int],
    body_awake_ind: wp.array2d[int],
    cJ: wp.array3d[float],
    cM: wp.array3d[float],
    cMa: wp.array2d[float],
    cacc: wp.array2d[wp.spatial_vector],
    cam_xmat: wp.array2d[wp.mat33],
    cam_xpos: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    cdof_dof: wp.array2d[int],
    cdof_dot: wp.array2d[wp.spatial_vector],
    cdof_tri_col: wp.array[int],
    cdof_tri_row: wp.array[int],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    cfrc_int: wp.array2d[wp.spatial_vector],
    cinert: wp.array2d[mjwp_types.vec10],
    cqLD: wp.array3d[float],
    cqacc: wp.array2d[float],
    cqacc_smooth: wp.array2d[float],
    cqacc_warmstart: wp.array2d[float],
    cqfrc_constraint: wp.array2d[float],
    cqfrc_smooth: wp.array2d[float],
    crb: wp.array2d[mjwp_types.vec10],
    crhs: wp.array3d[float],
    ctrl: wp.array2d[float],
    cvel: wp.array2d[wp.spatial_vector],
    cx: wp.array3d[float],
    dof_awake_ind: wp.array2d[int],
    dof_cdof: wp.array2d[int],
    dof_island: wp.array2d[int],
    dof_islandid: wp.array2d[int],
    efc_islandid: wp.array2d[int],
    efm_K_val: wp.array2d[float],
    efm_L: wp.array2d[float],
    efm_as: wp.array2d[float],
    efm_c: wp.array2d[float],
    efm_ca: wp.array2d[float],
    efm_diag: wp.array2d[float],
    efm_fluid: wp.array2d[float],
    efm_ts: wp.array2d[float],
    energy: wp.array[wp.vec2],
    eq_active: wp.array2d[bool],
    face_quat: wp.array2d[wp.quat],
    face_xpos: wp.array3d[wp.vec3],
    flex_aabb_max: wp.array2d[wp.vec3],
    flex_aabb_min: wp.array2d[wp.vec3],
    flex_hessian_valid: wp.array2d[bool],
    flex_rod_chain_ok: wp.array2d[bool],
    flex_rod_chain_valid: wp.array2d[bool],
    flex_rod_end_torque: wp.array2d[wp.spatial_vector],
    flex_rod_frame0_loc: wp.array3d[wp.vec3],
    flex_rod_frame0_ref: wp.array3d[wp.vec3],
    flex_rod_ref_ok: wp.array2d[bool],
    flex_rod_tail_loc: wp.array3d[wp.vec3],
    flex_rodedge_frc: wp.array2d[wp.vec3],
    flex_rodedge_grad: wp.array2d[float],
    flex_rodedge_ktheta: wp.array2d[float],
    flex_rodedge_tridiag: wp.array2d[wp.vec2],
    flex_rodhinge_Wd: wp.array2d[wp.mat33],
    flex_rodhinge_Wk: wp.array2d[wp.mat33],
    flex_rodhinge_g: wp.array2d[wp.vec3],
    flex_rodhinge_om0: wp.array2d[wp.vec2],
    flexedge_J: wp.array2d[float],
    flexedge_hessian: wp.array2d[wp.mat33],
    flexedge_length: wp.array2d[float],
    flexedge_velocity: wp.array2d[float],
    flexnode_xpos: wp.array2d[wp.vec3],
    flexvert_hessian: wp.array2d[mjwp_types.vec6],
    flexvert_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    geom_xpos: wp.array2d[wp.vec3],
    history: wp.array2d[float],
    island_dofadr: wp.array2d[int],
    island_idofadr: wp.array2d[int],
    island_iefcadr: wp.array2d[int],
    island_ne: wp.array2d[int],
    island_nefc: wp.array2d[int],
    island_nf: wp.array2d[int],
    island_nv: wp.array2d[int],
    light_xdir: wp.array2d[wp.vec3],
    light_xpos: wp.array2d[wp.vec3],
    map_dof2idof: wp.array2d[int],
    map_efc2iefc: wp.array2d[int],
    map_idof2dof: wp.array2d[int],
    map_iefc2efc: wp.array2d[int],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    moment_colind: wp.array2d[int],
    moment_rowadr: wp.array2d[int],
    moment_rownnz: wp.array2d[int],
    nacon: wp.array[int],
    nbody_awake: wp.array[int],
    ncdof: wp.array[int],
    ncollision: wp.array[int],
    ne: wp.array[int],
    nefc: wp.array[int],
    nf: wp.array[int],
    nidof: wp.array[int],
    nisland: wp.array[int],
    nl: wp.array[int],
    ntree_awake: wp.array[int],
    nv_awake: wp.array[int],
    overflow: wp.array[int],
    plastic: wp.array2d[float],
    qH: wp.array2d[float],
    qHDiagInv: wp.array2d[float],
    qHLD: wp.array2d[float],
    qLD: wp.array2d[float],
    qLDiagInv: wp.array2d[float],
    qacc: wp.array2d[float],
    qacc_smooth: wp.array2d[float],
    qacc_warmstart: wp.array2d[float],
    qfrc_actuator: wp.array2d[float],
    qfrc_adhesion: wp.array2d[float],
    qfrc_applied: wp.array2d[float],
    qfrc_bias: wp.array2d[float],
    qfrc_constraint: wp.array2d[float],
    qfrc_damper: wp.array2d[float],
    qfrc_fluid: wp.array2d[float],
    qfrc_gravcomp: wp.array2d[float],
    qfrc_passive: wp.array2d[float],
    qfrc_smooth: wp.array2d[float],
    qfrc_spring: wp.array2d[float],
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    sensordata: wp.array2d[float],
    site_xmat: wp.array2d[wp.mat33],
    site_xpos: wp.array2d[wp.vec3],
    solver_niter: wp.array[int],
    subtree_angmom: wp.array2d[wp.vec3],
    subtree_com: wp.array2d[wp.vec3],
    subtree_linvel: wp.array2d[wp.vec3],
    ten_J: wp.array2d[float],
    ten_length: wp.array2d[float],
    ten_velocity: wp.array2d[float],
    ten_wrapadr: wp.array2d[int],
    ten_wrapnum: wp.array2d[int],
    time: wp.array[float],
    tree_asleep: wp.array2d[int],
    tree_awake: wp.array2d[int],
    tree_island: wp.array2d[int],
    wrap_obj: wp.array2d[wp.vec2i],
    wrap_xpos: wp.array2d[wp.spatial_vector],
    xanchor: wp.array2d[wp.vec3],
    xaxis: wp.array2d[wp.vec3],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    ximat: wp.array2d[wp.mat33],
    xipos: wp.array2d[wp.vec3],
    xmat: wp.array2d[wp.mat33],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
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
    efc__D: wp.array2d[float],
    efc__J: wp.array3d[float],
    efc__J_colind: wp.array3d[int],
    efc__J_rowadr: wp.array2d[int],
    efc__J_rownnz: wp.array2d[int],
    efc__Jqvel: wp.array2d[float],
    efc__Ma: wp.array2d[float],
    efc__aref: wp.array2d[float],
    efc__force: wp.array2d[float],
    efc__frictionloss: wp.array2d[float],
    efc__id: wp.array2d[int],
    efc__island: wp.array2d[int],
    efc__jtdaj_adr: wp.array2d[int],
    efc__jtdaj_nblock: wp.array[int],
    efc__jtdaj_nrow: wp.array2d[int],
    efc__margin: wp.array2d[float],
    efc__pos: wp.array2d[float],
    efc__state: wp.array2d[int],
    efc__type: wp.array2d[int],
    efc__vel: wp.array2d[float],
):
  _m.opt = _o
  _m.callback = _cb
  _d.efc = _e
  _d.contact = _c
  _m.M_colind = M_colind
  _m.M_elemid = M_elemid
  _m.M_fullm_i = M_fullm_i
  _m.M_fullm_j = M_fullm_j
  _m.M_hinit_i = M_hinit_i
  _m.M_mulm_col = M_mulm_col
  _m.M_mulm_madr = M_mulm_madr
  _m.M_mulm_rowadr = M_mulm_rowadr
  _m.M_rowadr = M_rowadr
  _m.M_rownnz = M_rownnz
  _m.M_tiles = M_tiles
  _m.actuator_acc0 = actuator_acc0
  _m.actuator_actadr = actuator_actadr
  _m.actuator_actearly = actuator_actearly
  _m.actuator_actlimited = actuator_actlimited
  _m.actuator_actnum = actuator_actnum
  _m.actuator_actrange = actuator_actrange
  _m.actuator_armature = actuator_armature
  _m.actuator_biasprm = actuator_biasprm
  _m.actuator_biastype = actuator_biastype
  _m.actuator_cranklength = actuator_cranklength
  _m.actuator_ctrladr = actuator_ctrladr
  _m.actuator_ctrllimited = actuator_ctrllimited
  _m.actuator_ctrlnum = actuator_ctrlnum
  _m.actuator_ctrlrange = actuator_ctrlrange
  _m.actuator_ctrlspec = actuator_ctrlspec
  _m.actuator_damping = actuator_damping
  _m.actuator_dampingpoly = actuator_dampingpoly
  _m.actuator_delay = actuator_delay
  _m.actuator_dynprm = actuator_dynprm
  _m.actuator_dyntype = actuator_dyntype
  _m.actuator_forcelimited = actuator_forcelimited
  _m.actuator_forcerange = actuator_forcerange
  _m.actuator_gainprm = actuator_gainprm
  _m.actuator_gaintype = actuator_gaintype
  _m.actuator_gear = actuator_gear
  _m.actuator_history = actuator_history
  _m.actuator_historyadr = actuator_historyadr
  _m.actuator_lengthrange = actuator_lengthrange
  _m.actuator_trnid = actuator_trnid
  _m.actuator_trntype = actuator_trntype
  _m.actuator_trntype_body_adr = actuator_trntype_body_adr
  _m.block_dim = block_dim
  _m.body_branch_start = body_branch_start
  _m.body_branches = body_branches
  _m.body_dofadr = body_dofadr
  _m.body_dofnum = body_dofnum
  _m.body_fluid_box_adr = body_fluid_box_adr
  _m.body_fluid_ellipsoid = body_fluid_ellipsoid
  _m.body_fluid_ellipsoid_adr = body_fluid_ellipsoid_adr
  _m.body_freeadr = body_freeadr
  _m.body_geomadr = body_geomadr
  _m.body_geomnum = body_geomnum
  _m.body_gravcomp = body_gravcomp
  _m.body_inertia = body_inertia
  _m.body_invweight0 = body_invweight0
  _m.body_ipos = body_ipos
  _m.body_iquat = body_iquat
  _m.body_is_free = body_is_free
  _m.body_isdofancestor = body_isdofancestor
  _m.body_jntadr = body_jntadr
  _m.body_jntnum = body_jntnum
  _m.body_mass = body_mass
  _m.body_mocapid = body_mocapid
  _m.body_parentid = body_parentid
  _m.body_pos = body_pos
  _m.body_quat = body_quat
  _m.body_rootid = body_rootid
  _m.body_simple = body_simple
  _m.body_subtreemass = body_subtreemass
  _m.body_subtreenum = body_subtreenum
  _m.body_tree = body_tree
  _m.body_treeid = body_treeid
  _m.body_weldid = body_weldid
  _m.cam_bodyid = cam_bodyid
  _m.cam_fovy = cam_fovy
  _m.cam_intrinsic = cam_intrinsic
  _m.cam_mat0 = cam_mat0
  _m.cam_mode = cam_mode
  _m.cam_pos = cam_pos
  _m.cam_pos0 = cam_pos0
  _m.cam_poscom0 = cam_poscom0
  _m.cam_projection = cam_projection
  _m.cam_quat = cam_quat
  _m.cam_resolution = cam_resolution
  _m.cam_sensorsize = cam_sensorsize
  _m.cam_targetbodyid = cam_targetbodyid
  _m.dof_armature = dof_armature
  _m.dof_bodyid = dof_bodyid
  _m.dof_damping = dof_damping
  _m.dof_dampingpoly = dof_dampingpoly
  _m.dof_frictionloss = dof_frictionloss
  _m.dof_invweight0 = dof_invweight0
  _m.dof_jntid = dof_jntid
  _m.dof_length = dof_length
  _m.dof_parentid = dof_parentid
  _m.dof_solimp = dof_solimp
  _m.dof_solref = dof_solref
  _m.dof_treeid = dof_treeid
  _m.dof_tri_col = dof_tri_col
  _m.dof_tri_row = dof_tri_row
  _m.efm0_L = efm0_L
  _m.efm0_L_colind = efm0_L_colind
  _m.efm0_L_rowadr = efm0_L_rowadr
  _m.efm0_L_rownnz = efm0_L_rownnz
  _m.efm0_active = efm0_active
  _m.efm0_dofid = efm0_dofid
  _m.efm_K_colind = efm_K_colind
  _m.efm_K_rowadr = efm_K_rowadr
  _m.efm_K_rownnz = efm_K_rownnz
  _m.efm_con_max_dof = efm_con_max_dof
  _m.efm_dofblk = efm_dofblk
  _m.efm_dofid = efm_dofid
  _m.eq_connect_adr = eq_connect_adr
  _m.eq_data = eq_data
  _m.eq_flex_adr = eq_flex_adr
  _m.eq_flexstrain_adr = eq_flexstrain_adr
  _m.eq_jnt_adr = eq_jnt_adr
  _m.eq_obj1id = eq_obj1id
  _m.eq_obj2id = eq_obj2id
  _m.eq_objtype = eq_objtype
  _m.eq_solimp = eq_solimp
  _m.eq_solref = eq_solref
  _m.eq_ten_adr = eq_ten_adr
  _m.eq_type = eq_type
  _m.eq_wld_adr = eq_wld_adr
  _m.flex_activelayers = flex_activelayers
  _m.flex_bend_interp_map = flex_bend_interp_map
  _m.flex_bending = flex_bending
  _m.flex_bendingadr = flex_bendingadr
  _m.flex_cell_map = flex_cell_map
  _m.flex_cellnum = flex_cellnum
  _m.flex_centered = flex_centered
  _m.flex_conaffinity = flex_conaffinity
  _m.flex_condim = flex_condim
  _m.flex_contype = flex_contype
  _m.flex_damping = flex_damping
  _m.flex_dim = flex_dim
  _m.flex_edge = flex_edge
  _m.flex_edgeadr = flex_edgeadr
  _m.flex_edgedamping = flex_edgedamping
  _m.flex_edgeequality = flex_edgeequality
  _m.flex_edgeflap = flex_edgeflap
  _m.flex_edgeflexid = flex_edgeflexid
  _m.flex_edgenum = flex_edgenum
  _m.flex_edgestiffness = flex_edgestiffness
  _m.flex_eff_possible = flex_eff_possible
  _m.flex_elem = flex_elem
  _m.flex_elem_rodedge = flex_elem_rodedge
  _m.flex_elemadr = flex_elemadr
  _m.flex_elemdataadr = flex_elemdataadr
  _m.flex_elemedge = flex_elemedge
  _m.flex_elemedgeadr = flex_elemedgeadr
  _m.flex_elemflexid = flex_elemflexid
  _m.flex_elemlayer = flex_elemlayer
  _m.flex_elemnum = flex_elemnum
  _m.flex_face = flex_face
  _m.flex_face_map = flex_face_map
  _m.flex_faceadr = flex_faceadr
  _m.flex_friction = flex_friction
  _m.flex_gap = flex_gap
  _m.flex_interp = flex_interp
  _m.flex_interp_assemblable = flex_interp_assemblable
  _m.flex_margin = flex_margin
  _m.flex_node = flex_node
  _m.flex_node0 = flex_node0
  _m.flex_nodeadr = flex_nodeadr
  _m.flex_nodebodyid = flex_nodebodyid
  _m.flex_nodenum = flex_nodenum
  _m.flex_passive = flex_passive
  _m.flex_plasticadr = flex_plasticadr
  _m.flex_plasticmode = flex_plasticmode
  _m.flex_priority = flex_priority
  _m.flex_radius = flex_radius
  _m.flex_rigid = flex_rigid
  _m.flex_rod_box_size = flex_rod_box_size
  _m.flex_rod_edgeadr = flex_rod_edgeadr
  _m.flex_rod_edgenum = flex_rod_edgenum
  _m.flex_rod_has_twist = flex_rod_has_twist
  _m.flex_rod_hingeadr = flex_rod_hingeadr
  _m.flex_rod_is_box = flex_rod_is_box
  _m.flex_rod_is_chain = flex_rod_is_chain
  _m.flex_rod_root_clamped = flex_rod_root_clamped
  _m.flex_rod_tail_clamped = flex_rod_tail_clamped
  _m.flex_rodedge_jnt = flex_rodedge_jnt
  _m.flex_rodedge_vert = flex_rodedge_vert
  _m.flex_rodhinge_map = flex_rodhinge_map
  _m.flex_selfcollide = flex_selfcollide
  _m.flex_simple = flex_simple
  _m.flex_size = flex_size
  _m.flex_solimp = flex_solimp
  _m.flex_solmix = flex_solmix
  _m.flex_solref = flex_solref
  _m.flex_stiffness = flex_stiffness
  _m.flex_stiffnessadr = flex_stiffnessadr
  _m.flex_vert = flex_vert
  _m.flex_vert0 = flex_vert0
  _m.flex_vert_rodelem = flex_vert_rodelem
  _m.flex_vertadr = flex_vertadr
  _m.flex_vertbodyid = flex_vertbodyid
  _m.flex_vertcell = flex_vertcell
  _m.flex_vertflexid = flex_vertflexid
  _m.flex_vertnum = flex_vertnum
  _m.flexedge_J_colind = flexedge_J_colind
  _m.flexedge_J_rowadr = flexedge_J_rowadr
  _m.flexedge_J_rownnz = flexedge_J_rownnz
  _m.flexedge_invweight0 = flexedge_invweight0
  _m.flexedge_length0 = flexedge_length0
  _m.flexedge_rigid = flexedge_rigid
  _m.flexstrain_J_colind = flexstrain_J_colind
  _m.flexstrain_J_rowadr = flexstrain_J_rowadr
  _m.flexstrain_J_rownnz = flexstrain_J_rownnz
  _m.flg_adhesion = flg_adhesion
  _m.flg_surfacevel = flg_surfacevel
  _m.geom_aabb = geom_aabb
  _m.geom_adhesion = geom_adhesion
  _m.geom_bodyid = geom_bodyid
  _m.geom_conaffinity = geom_conaffinity
  _m.geom_condim = geom_condim
  _m.geom_contype = geom_contype
  _m.geom_dataid = geom_dataid
  _m.geom_fluid = geom_fluid
  _m.geom_friction = geom_friction
  _m.geom_gap = geom_gap
  _m.geom_group = geom_group
  _m.geom_margin = geom_margin
  _m.geom_matid = geom_matid
  _m.geom_pair_type_count = geom_pair_type_count
  _m.geom_plugin_index = geom_plugin_index
  _m.geom_pos = geom_pos
  _m.geom_priority = geom_priority
  _m.geom_quat = geom_quat
  _m.geom_rbound = geom_rbound
  _m.geom_rgba = geom_rgba
  _m.geom_size = geom_size
  _m.geom_solimp = geom_solimp
  _m.geom_solmix = geom_solmix
  _m.geom_solref = geom_solref
  _m.geom_surfacevel = geom_surfacevel
  _m.geom_type = geom_type
  _m.has_1d_flex = has_1d_flex
  _m.has_3d_flex = has_3d_flex
  _m.has_efm_actuator = has_efm_actuator
  _m.has_ellipsoid_geom = has_ellipsoid_geom
  _m.has_flex_edge_stiff_damp = has_flex_edge_stiff_damp
  _m.has_flex_passive = has_flex_passive
  _m.has_flex_selfcollide = has_flex_selfcollide
  _m.has_flex_snh = has_flex_snh
  _m.has_fluid = has_fluid
  _m.has_non_simple_flex = has_non_simple_flex
  _m.has_plane_geom = has_plane_geom
  _m.has_sdf_geom = has_sdf_geom
  _m.has_tendon_damping = has_tendon_damping
  _m.has_tendon_stiffness = has_tendon_stiffness
  _m.has_unsupported_flex_interp = has_unsupported_flex_interp
  _m.hfield_adr = hfield_adr
  _m.hfield_data = hfield_data
  _m.hfield_ncol = hfield_ncol
  _m.hfield_nrow = hfield_nrow
  _m.hfield_size = hfield_size
  _m.is_sparse = is_sparse
  _m.jnt_actfrclimited = jnt_actfrclimited
  _m.jnt_actfrcrange = jnt_actfrcrange
  _m.jnt_actgravcomp = jnt_actgravcomp
  _m.jnt_actuatorid = jnt_actuatorid
  _m.jnt_axis = jnt_axis
  _m.jnt_bodyid = jnt_bodyid
  _m.jnt_dofadr = jnt_dofadr
  _m.jnt_limited_ball_adr = jnt_limited_ball_adr
  _m.jnt_limited_slide_hinge_adr = jnt_limited_slide_hinge_adr
  _m.jnt_margin = jnt_margin
  _m.jnt_plasticadr = jnt_plasticadr
  _m.jnt_pos = jnt_pos
  _m.jnt_qposadr = jnt_qposadr
  _m.jnt_range = jnt_range
  _m.jnt_solimp = jnt_solimp
  _m.jnt_solref = jnt_solref
  _m.jnt_stiffness = jnt_stiffness
  _m.jnt_stiffnesspoly = jnt_stiffnesspoly
  _m.jnt_type = jnt_type
  _m.light_bodyid = light_bodyid
  _m.light_dir = light_dir
  _m.light_dir0 = light_dir0
  _m.light_mode = light_mode
  _m.light_pos = light_pos
  _m.light_pos0 = light_pos0
  _m.light_poscom0 = light_poscom0
  _m.light_targetbodyid = light_targetbodyid
  _m.mat_rgba = mat_rgba
  _m.max_ten_J_rownnz = max_ten_J_rownnz
  _m.mesh_face = mesh_face
  _m.mesh_faceadr = mesh_faceadr
  _m.mesh_graph = mesh_graph
  _m.mesh_graphadr = mesh_graphadr
  _m.mesh_normal = mesh_normal
  _m.mesh_normaladr = mesh_normaladr
  _m.mesh_normalnum = mesh_normalnum
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
  _m.mesh_quat = mesh_quat
  _m.mesh_vert = mesh_vert
  _m.mesh_vertadr = mesh_vertadr
  _m.mesh_vertnum = mesh_vertnum
  _m.nC = nC
  _m.nJten = nJten
  _m.na = na
  _m.nacttrnbody = nacttrnbody
  _m.nactuator = nactuator
  _m.nbody = nbody
  _m.nbranch = nbranch
  _m.ncam = ncam
  _m.nefm0dof = nefm0dof
  _m.nefmK = nefmK
  _m.nefmdof = nefmdof
  _m.neq = neq
  _m.nflex = nflex
  _m.nflexbend_interp = nflexbend_interp
  _m.nflexedge = nflexedge
  _m.nflexelem = nflexelem
  _m.nflexface = nflexface
  _m.nflexintcell = nflexintcell
  _m.nflexnode = nflexnode
  _m.nflexrodedge = nflexrodedge
  _m.nflexrodhinge = nflexrodhinge
  _m.nflexvert = nflexvert
  _m.ngeom = ngeom
  _m.nhistory = nhistory
  _m.njnt = njnt
  _m.nlight = nlight
  _m.nmaxcondim = nmaxcondim
  _m.nmaxpyramid = nmaxpyramid
  _m.nmesh = nmesh
  _m.nmeshdegmax = nmeshdegmax
  _m.nmeshface = nmeshface
  _m.npair = npair
  _m.npolygonmax = npolygonmax
  _m.nrangefinder = nrangefinder
  _m.nsensorcollision = nsensorcollision
  _m.nsensortaxel = nsensortaxel
  _m.nsite = nsite
  _m.ntactileweld = ntactileweld
  _m.ntendon = ntendon
  _m.ntree = ntree
  _m.nu = nu
  _m.nv = nv
  _m.nv_pad = nv_pad
  _m.nwrap = nwrap
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
  _m.opt.cone = opt__cone
  _m.opt.contact_sensor_maxmatch = opt__contact_sensor_maxmatch
  _m.opt.density = opt__density
  _m.opt.deterministic = opt__deterministic
  _m.opt.disableflags = opt__disableflags
  _m.opt.enableflags = opt__enableflags
  _m.opt.graph_conditional = opt__graph_conditional
  _m.opt.gravity = opt__gravity
  _m.opt.impratio_invsqrt = opt__impratio_invsqrt
  _m.opt.integrator = opt__integrator
  _m.opt.iterations = opt__iterations
  _m.opt.ls_iterations = opt__ls_iterations
  _m.opt.ls_tolerance = opt__ls_tolerance
  _m.opt.magnetic = opt__magnetic
  _m.opt.noslip_iterations = opt__noslip_iterations
  _m.opt.noslip_tolerance = opt__noslip_tolerance
  _m.opt.run_collision_detection = opt__run_collision_detection
  _m.opt.run_rne_postconstraint = opt__run_rne_postconstraint
  _m.opt.sdf_initpoints = opt__sdf_initpoints
  _m.opt.sdf_iterations = opt__sdf_iterations
  _m.opt.solver = opt__solver
  _m.opt.timestep = opt__timestep
  _m.opt.tolerance = opt__tolerance
  _m.opt.viscosity = opt__viscosity
  _m.opt.warn_overflow = opt__warn_overflow
  _m.opt.wind = opt__wind
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
  _m.qLD_all_updates = qLD_all_updates
  _m.qLD_block_adr = qLD_block_adr
  _m.qLD_block_total = qLD_block_total
  _m.qLD_level_offsets = qLD_level_offsets
  _m.qLD_updates = qLD_updates
  _m.qpos0 = qpos0
  _m.qpos_spring = qpos_spring
  _m.rangefinder_sensor_adr = rangefinder_sensor_adr
  _m.sensor_acc_adr = sensor_acc_adr
  _m.sensor_adr = sensor_adr
  _m.sensor_contact_acc_adr = sensor_contact_acc_adr
  _m.sensor_contact_pos_adr = sensor_contact_pos_adr
  _m.sensor_cutoff = sensor_cutoff
  _m.sensor_datatype = sensor_datatype
  _m.sensor_delay = sensor_delay
  _m.sensor_dim = sensor_dim
  _m.sensor_e_kinetic = sensor_e_kinetic
  _m.sensor_e_potential = sensor_e_potential
  _m.sensor_history = sensor_history
  _m.sensor_historyadr = sensor_historyadr
  _m.sensor_interval = sensor_interval
  _m.sensor_intprm = sensor_intprm
  _m.sensor_limitfrc_adr = sensor_limitfrc_adr
  _m.sensor_limitpos_adr = sensor_limitpos_adr
  _m.sensor_limitvel_adr = sensor_limitvel_adr
  _m.sensor_objid = sensor_objid
  _m.sensor_objtype = sensor_objtype
  _m.sensor_pos_adr = sensor_pos_adr
  _m.sensor_rangefinder_adr = sensor_rangefinder_adr
  _m.sensor_rangefinder_bodyid = sensor_rangefinder_bodyid
  _m.sensor_refid = sensor_refid
  _m.sensor_reftype = sensor_reftype
  _m.sensor_rne_postconstraint = sensor_rne_postconstraint
  _m.sensor_stage_acc_adr = sensor_stage_acc_adr
  _m.sensor_stage_pos_adr = sensor_stage_pos_adr
  _m.sensor_stage_vel_adr = sensor_stage_vel_adr
  _m.sensor_subtree_vel = sensor_subtree_vel
  _m.sensor_tendonactfrc_adr = sensor_tendonactfrc_adr
  _m.sensor_touch_adr = sensor_touch_adr
  _m.sensor_type = sensor_type
  _m.sensor_vel_adr = sensor_vel_adr
  _m.site_bodyid = site_bodyid
  _m.site_pos = site_pos
  _m.site_quat = site_quat
  _m.site_size = site_size
  _m.site_type = site_type
  _m.taxel_sensorid = taxel_sensorid
  _m.taxel_vertadr = taxel_vertadr
  _m.ten_J_colind = ten_J_colind
  _m.ten_J_rowadr = ten_J_rowadr
  _m.ten_J_rownnz = ten_J_rownnz
  _m.tendon_actfrclimited = tendon_actfrclimited
  _m.tendon_actfrcrange = tendon_actfrcrange
  _m.tendon_actuatorid = tendon_actuatorid
  _m.tendon_adr = tendon_adr
  _m.tendon_armature = tendon_armature
  _m.tendon_damping = tendon_damping
  _m.tendon_dampingpoly = tendon_dampingpoly
  _m.tendon_frictionloss = tendon_frictionloss
  _m.tendon_geom_adr = tendon_geom_adr
  _m.tendon_invweight0 = tendon_invweight0
  _m.tendon_jnt_adr = tendon_jnt_adr
  _m.tendon_length0 = tendon_length0
  _m.tendon_lengthspring = tendon_lengthspring
  _m.tendon_limited = tendon_limited
  _m.tendon_limited_adr = tendon_limited_adr
  _m.tendon_margin = tendon_margin
  _m.tendon_num = tendon_num
  _m.tendon_plasticadr = tendon_plasticadr
  _m.tendon_range = tendon_range
  _m.tendon_site_pair_adr = tendon_site_pair_adr
  _m.tendon_solimp_fri = tendon_solimp_fri
  _m.tendon_solimp_lim = tendon_solimp_lim
  _m.tendon_solref_fri = tendon_solref_fri
  _m.tendon_solref_lim = tendon_solref_lim
  _m.tendon_stiffness = tendon_stiffness
  _m.tendon_stiffnesspoly = tendon_stiffnesspoly
  _m.tendon_type = tendon_type
  _m.tree_dofadr = tree_dofadr
  _m.tree_dofnum = tree_dofnum
  _m.tree_sleep_policy = tree_sleep_policy
  _m.weld_tactile_id = weld_tactile_id
  _m.wrap_geom_adr = wrap_geom_adr
  _m.wrap_jnt_adr = wrap_jnt_adr
  _m.wrap_objid = wrap_objid
  _m.wrap_prm = wrap_prm
  _m.wrap_pulley_scale = wrap_pulley_scale
  _m.wrap_site_pair_adr = wrap_site_pair_adr
  _m.wrap_type = wrap_type
  _d.M = M
  _d.act = act
  _d.act_dot = act_dot
  _d.actuator_force = actuator_force
  _d.actuator_length = actuator_length
  _d.actuator_moment = actuator_moment
  _d.actuator_velocity = actuator_velocity
  _d.body_awake = body_awake
  _d.body_awake_ind = body_awake_ind
  _d.cJ = cJ
  _d.cM = cM
  _d.cMa = cMa
  _d.cacc = cacc
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.cdof = cdof
  _d.cdof_dof = cdof_dof
  _d.cdof_dot = cdof_dot
  _d.cdof_tri_col = cdof_tri_col
  _d.cdof_tri_row = cdof_tri_row
  _d.cfrc_ext = cfrc_ext
  _d.cfrc_int = cfrc_int
  _d.cinert = cinert
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
  _d.cqLD = cqLD
  _d.cqacc = cqacc
  _d.cqacc_smooth = cqacc_smooth
  _d.cqacc_warmstart = cqacc_warmstart
  _d.cqfrc_constraint = cqfrc_constraint
  _d.cqfrc_smooth = cqfrc_smooth
  _d.crb = crb
  _d.crhs = crhs
  _d.ctrl = ctrl
  _d.cvel = cvel
  _d.cx = cx
  _d.dof_awake_ind = dof_awake_ind
  _d.dof_cdof = dof_cdof
  _d.dof_island = dof_island
  _d.dof_islandid = dof_islandid
  _d.efc.D = efc__D
  _d.efc.J = efc__J
  _d.efc.J_colind = efc__J_colind
  _d.efc.J_rowadr = efc__J_rowadr
  _d.efc.J_rownnz = efc__J_rownnz
  _d.efc.Jqvel = efc__Jqvel
  _d.efc.Ma = efc__Ma
  _d.efc.aref = efc__aref
  _d.efc.force = efc__force
  _d.efc.frictionloss = efc__frictionloss
  _d.efc.id = efc__id
  _d.efc.island = efc__island
  _d.efc.jtdaj_adr = efc__jtdaj_adr
  _d.efc.jtdaj_nblock = efc__jtdaj_nblock
  _d.efc.jtdaj_nrow = efc__jtdaj_nrow
  _d.efc.margin = efc__margin
  _d.efc.pos = efc__pos
  _d.efc.state = efc__state
  _d.efc.type = efc__type
  _d.efc.vel = efc__vel
  _d.efc_islandid = efc_islandid
  _d.efm_K_val = efm_K_val
  _d.efm_L = efm_L
  _d.efm_as = efm_as
  _d.efm_c = efm_c
  _d.efm_ca = efm_ca
  _d.efm_diag = efm_diag
  _d.efm_fluid = efm_fluid
  _d.efm_ts = efm_ts
  _d.energy = energy
  _d.eq_active = eq_active
  _d.face_quat = face_quat
  _d.face_xpos = face_xpos
  _d.flex_aabb_max = flex_aabb_max
  _d.flex_aabb_min = flex_aabb_min
  _d.flex_hessian_valid = flex_hessian_valid
  _d.flex_rod_chain_ok = flex_rod_chain_ok
  _d.flex_rod_chain_valid = flex_rod_chain_valid
  _d.flex_rod_end_torque = flex_rod_end_torque
  _d.flex_rod_frame0_loc = flex_rod_frame0_loc
  _d.flex_rod_frame0_ref = flex_rod_frame0_ref
  _d.flex_rod_ref_ok = flex_rod_ref_ok
  _d.flex_rod_tail_loc = flex_rod_tail_loc
  _d.flex_rodedge_frc = flex_rodedge_frc
  _d.flex_rodedge_grad = flex_rodedge_grad
  _d.flex_rodedge_ktheta = flex_rodedge_ktheta
  _d.flex_rodedge_tridiag = flex_rodedge_tridiag
  _d.flex_rodhinge_Wd = flex_rodhinge_Wd
  _d.flex_rodhinge_Wk = flex_rodhinge_Wk
  _d.flex_rodhinge_g = flex_rodhinge_g
  _d.flex_rodhinge_om0 = flex_rodhinge_om0
  _d.flexedge_J = flexedge_J
  _d.flexedge_hessian = flexedge_hessian
  _d.flexedge_length = flexedge_length
  _d.flexedge_velocity = flexedge_velocity
  _d.flexnode_xpos = flexnode_xpos
  _d.flexvert_hessian = flexvert_hessian
  _d.flexvert_xpos = flexvert_xpos
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.history = history
  _d.island_dofadr = island_dofadr
  _d.island_idofadr = island_idofadr
  _d.island_iefcadr = island_iefcadr
  _d.island_ne = island_ne
  _d.island_nefc = island_nefc
  _d.island_nf = island_nf
  _d.island_nv = island_nv
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  _d.map_dof2idof = map_dof2idof
  _d.map_efc2iefc = map_efc2iefc
  _d.map_idof2dof = map_idof2dof
  _d.map_iefc2efc = map_iefc2efc
  _d.mocap_pos = mocap_pos
  _d.mocap_quat = mocap_quat
  _d.moment_colind = moment_colind
  _d.moment_rowadr = moment_rowadr
  _d.moment_rownnz = moment_rownnz
  _d.naccdmax = naccdmax
  _d.nacon = nacon
  _d.naconmax = naconmax
  _d.nbody_awake = nbody_awake
  _d.ncdof = ncdof
  _d.ncollision = ncollision
  _d.ne = ne
  _d.nefc = nefc
  _d.nf = nf
  _d.nidof = nidof
  _d.nisland = nisland
  _d.njmax = njmax
  _d.njmax_nnz = njmax_nnz
  _d.nl = nl
  _d.ntree_awake = ntree_awake
  _d.nv_awake = nv_awake
  _d.nvmax = nvmax
  _d.nvmax_pad = nvmax_pad
  _d.overflow = overflow
  _d.plastic = plastic
  _d.qH = qH
  _d.qHDiagInv = qHDiagInv
  _d.qHLD = qHLD
  _d.qLD = qLD
  _d.qLDiagInv = qLDiagInv
  _d.qacc = qacc
  _d.qacc_smooth = qacc_smooth
  _d.qacc_warmstart = qacc_warmstart
  _d.qfrc_actuator = qfrc_actuator
  _d.qfrc_adhesion = qfrc_adhesion
  _d.qfrc_applied = qfrc_applied
  _d.qfrc_bias = qfrc_bias
  _d.qfrc_constraint = qfrc_constraint
  _d.qfrc_damper = qfrc_damper
  _d.qfrc_fluid = qfrc_fluid
  _d.qfrc_gravcomp = qfrc_gravcomp
  _d.qfrc_passive = qfrc_passive
  _d.qfrc_smooth = qfrc_smooth
  _d.qfrc_spring = qfrc_spring
  _d.qpos = qpos
  _d.qvel = qvel
  _d.sensordata = sensordata
  _d.site_xmat = site_xmat
  _d.site_xpos = site_xpos
  _d.solver_niter = solver_niter
  _d.subtree_angmom = subtree_angmom
  _d.subtree_com = subtree_com
  _d.subtree_linvel = subtree_linvel
  _d.ten_J = ten_J
  _d.ten_length = ten_length
  _d.ten_velocity = ten_velocity
  _d.ten_wrapadr = ten_wrapadr
  _d.ten_wrapnum = ten_wrapnum
  _d.time = time
  _d.tree_asleep = tree_asleep
  _d.tree_awake = tree_awake
  _d.tree_island = tree_island
  _d.wrap_obj = wrap_obj
  _d.wrap_xpos = wrap_xpos
  _d.xanchor = xanchor
  _d.xaxis = xaxis
  _d.xfrc_applied = xfrc_applied
  _d.ximat = ximat
  _d.xipos = xipos
  _d.xmat = xmat
  _d.xpos = xpos
  _d.xquat = xquat
  _d.nworld = nworld
  mjwarp.forward(_m, _d)


def _forward_jax_impl(m: types.Model, d: types.Data):
  output_dims = {
      'M': d._impl.M.shape,
      'act_dot': d.act_dot.shape,
      'actuator_force': d.actuator_force.shape,
      'actuator_length': d.actuator_length.shape,
      'actuator_moment': d._impl.actuator_moment.shape,
      'actuator_velocity': d._impl.actuator_velocity.shape,
      'body_awake': d._impl.body_awake.shape,
      'body_awake_ind': d._impl.body_awake_ind.shape,
      'cJ': d._impl.cJ.shape,
      'cM': d._impl.cM.shape,
      'cacc': d._impl.cacc.shape,
      'cam_xmat': d.cam_xmat.shape,
      'cam_xpos': d.cam_xpos.shape,
      'cdof': d.cdof.shape,
      'cdof_dof': d._impl.cdof_dof.shape,
      'cdof_dot': d.cdof_dot.shape,
      'cfrc_ext': d._impl.cfrc_ext.shape,
      'cfrc_int': d._impl.cfrc_int.shape,
      'cinert': d._impl.cinert.shape,
      'cqLD': d._impl.cqLD.shape,
      'cqacc_smooth': d._impl.cqacc_smooth.shape,
      'cqacc_warmstart': d._impl.cqacc_warmstart.shape,
      'cqfrc_smooth': d._impl.cqfrc_smooth.shape,
      'crb': d._impl.crb.shape,
      'crhs': d._impl.crhs.shape,
      'cvel': d.cvel.shape,
      'cx': d._impl.cx.shape,
      'dof_awake_ind': d._impl.dof_awake_ind.shape,
      'dof_cdof': d._impl.dof_cdof.shape,
      'dof_island': d._impl.dof_island.shape,
      'dof_islandid': d._impl.dof_islandid.shape,
      'efc_islandid': d._impl.efc_islandid.shape,
      'efm_K_val': d._impl.efm_K_val.shape,
      'efm_L': d._impl.efm_L.shape,
      'efm_as': d._impl.efm_as.shape,
      'efm_c': d._impl.efm_c.shape,
      'efm_ca': d._impl.efm_ca.shape,
      'efm_diag': d._impl.efm_diag.shape,
      'efm_fluid': d._impl.efm_fluid.shape,
      'efm_ts': d._impl.efm_ts.shape,
      'energy': d._impl.energy.shape,
      'face_quat': d._impl.face_quat.shape,
      'face_xpos': d._impl.face_xpos.shape,
      'flex_aabb_max': d._impl.flex_aabb_max.shape,
      'flex_aabb_min': d._impl.flex_aabb_min.shape,
      'flex_rod_chain_ok': d._impl.flex_rod_chain_ok.shape,
      'flex_rod_chain_valid': d._impl.flex_rod_chain_valid.shape,
      'flex_rod_end_torque': d._impl.flex_rod_end_torque.shape,
      'flex_rod_frame0_loc': d._impl.flex_rod_frame0_loc.shape,
      'flex_rod_frame0_ref': d._impl.flex_rod_frame0_ref.shape,
      'flex_rod_ref_ok': d._impl.flex_rod_ref_ok.shape,
      'flex_rod_tail_loc': d._impl.flex_rod_tail_loc.shape,
      'flex_rodedge_frc': d._impl.flex_rodedge_frc.shape,
      'flex_rodedge_grad': d._impl.flex_rodedge_grad.shape,
      'flex_rodedge_ktheta': d._impl.flex_rodedge_ktheta.shape,
      'flex_rodedge_tridiag': d._impl.flex_rodedge_tridiag.shape,
      'flex_rodhinge_Wd': d._impl.flex_rodhinge_Wd.shape,
      'flex_rodhinge_Wk': d._impl.flex_rodhinge_Wk.shape,
      'flex_rodhinge_g': d._impl.flex_rodhinge_g.shape,
      'flex_rodhinge_om0': d._impl.flex_rodhinge_om0.shape,
      'flexedge_J': d._impl.flexedge_J.shape,
      'flexedge_hessian': d._impl.flexedge_hessian.shape,
      'flexedge_length': d._impl.flexedge_length.shape,
      'flexedge_velocity': d._impl.flexedge_velocity.shape,
      'flexnode_xpos': d._impl.flexnode_xpos.shape,
      'flexvert_hessian': d._impl.flexvert_hessian.shape,
      'flexvert_xpos': d._impl.flexvert_xpos.shape,
      'geom_xmat': d.geom_xmat.shape,
      'geom_xpos': d.geom_xpos.shape,
      'history': d.history.shape,
      'island_dofadr': d._impl.island_dofadr.shape,
      'island_idofadr': d._impl.island_idofadr.shape,
      'island_iefcadr': d._impl.island_iefcadr.shape,
      'island_ne': d._impl.island_ne.shape,
      'island_nefc': d._impl.island_nefc.shape,
      'island_nf': d._impl.island_nf.shape,
      'island_nv': d._impl.island_nv.shape,
      'light_xdir': d._impl.light_xdir.shape,
      'light_xpos': d._impl.light_xpos.shape,
      'map_dof2idof': d._impl.map_dof2idof.shape,
      'map_efc2iefc': d._impl.map_efc2iefc.shape,
      'map_idof2dof': d._impl.map_idof2dof.shape,
      'map_iefc2efc': d._impl.map_iefc2efc.shape,
      'moment_colind': d._impl.moment_colind.shape,
      'moment_rowadr': d._impl.moment_rowadr.shape,
      'moment_rownnz': d._impl.moment_rownnz.shape,
      'nacon': d._impl.nacon.shape,
      'nbody_awake': d._impl.nbody_awake.shape,
      'ncdof': d._impl.ncdof.shape,
      'ncollision': d._impl.ncollision.shape,
      'ne': d._impl.ne.shape,
      'nefc': d._impl.nefc.shape,
      'nf': d._impl.nf.shape,
      'nidof': d._impl.nidof.shape,
      'nisland': d._impl.nisland.shape,
      'nl': d._impl.nl.shape,
      'ntree_awake': d._impl.ntree_awake.shape,
      'nv_awake': d._impl.nv_awake.shape,
      'overflow': d._impl.overflow.shape,
      'qH': d._impl.qH.shape,
      'qHDiagInv': d._impl.qHDiagInv.shape,
      'qHLD': d._impl.qHLD.shape,
      'qLD': d._impl.qLD.shape,
      'qLDiagInv': d._impl.qLDiagInv.shape,
      'qacc': d.qacc.shape,
      'qacc_smooth': d.qacc_smooth.shape,
      'qfrc_actuator': d.qfrc_actuator.shape,
      'qfrc_adhesion': d._impl.qfrc_adhesion.shape,
      'qfrc_bias': d.qfrc_bias.shape,
      'qfrc_constraint': d.qfrc_constraint.shape,
      'qfrc_damper': d._impl.qfrc_damper.shape,
      'qfrc_fluid': d.qfrc_fluid.shape,
      'qfrc_gravcomp': d.qfrc_gravcomp.shape,
      'qfrc_passive': d.qfrc_passive.shape,
      'qfrc_smooth': d.qfrc_smooth.shape,
      'qfrc_spring': d._impl.qfrc_spring.shape,
      'qvel': d.qvel.shape,
      'sensordata': d.sensordata.shape,
      'site_xmat': d.site_xmat.shape,
      'site_xpos': d.site_xpos.shape,
      'solver_niter': d._impl.solver_niter.shape,
      'subtree_angmom': d._impl.subtree_angmom.shape,
      'subtree_com': d.subtree_com.shape,
      'subtree_linvel': d._impl.subtree_linvel.shape,
      'ten_J': d._impl.ten_J.shape,
      'ten_length': d.ten_length.shape,
      'ten_velocity': d._impl.ten_velocity.shape,
      'ten_wrapadr': d._impl.ten_wrapadr.shape,
      'ten_wrapnum': d._impl.ten_wrapnum.shape,
      'tree_asleep': d._impl.tree_asleep.shape,
      'tree_awake': d._impl.tree_awake.shape,
      'tree_island': d._impl.tree_island.shape,
      'wrap_obj': d._impl.wrap_obj.shape,
      'wrap_xpos': d._impl.wrap_xpos.shape,
      'xanchor': d.xanchor.shape,
      'xaxis': d.xaxis.shape,
      'ximat': d.ximat.shape,
      'xipos': d.xipos.shape,
      'xmat': d.xmat.shape,
      'xpos': d.xpos.shape,
      'xquat': d.xquat.shape,
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
      'efc__D': d._impl.efc__D.shape,
      'efc__J': d._impl.efc__J.shape,
      'efc__J_colind': d._impl.efc__J_colind.shape,
      'efc__J_rowadr': d._impl.efc__J_rowadr.shape,
      'efc__J_rownnz': d._impl.efc__J_rownnz.shape,
      'efc__Jqvel': d._impl.efc__Jqvel.shape,
      'efc__Ma': d._impl.efc__Ma.shape,
      'efc__aref': d._impl.efc__aref.shape,
      'efc__force': d._impl.efc__force.shape,
      'efc__frictionloss': d._impl.efc__frictionloss.shape,
      'efc__id': d._impl.efc__id.shape,
      'efc__island': d._impl.efc__island.shape,
      'efc__jtdaj_adr': d._impl.efc__jtdaj_adr.shape,
      'efc__jtdaj_nblock': d._impl.efc__jtdaj_nblock.shape,
      'efc__jtdaj_nrow': d._impl.efc__jtdaj_nrow.shape,
      'efc__margin': d._impl.efc__margin.shape,
      'efc__pos': d._impl.efc__pos.shape,
      'efc__state': d._impl.efc__state.shape,
      'efc__type': d._impl.efc__type.shape,
      'efc__vel': d._impl.efc__vel.shape,
  }
  jf = ffi.jax_callable_variadic_tuple(
      _forward_shim,
      num_outputs=180,
      output_dims=output_dims,
      vmap_method=None,
      in_out_argnames=set([
          'M',
          'act_dot',
          'actuator_force',
          'actuator_length',
          'actuator_moment',
          'actuator_velocity',
          'body_awake',
          'body_awake_ind',
          'cJ',
          'cM',
          'cacc',
          'cam_xmat',
          'cam_xpos',
          'cdof',
          'cdof_dof',
          'cdof_dot',
          'cfrc_ext',
          'cfrc_int',
          'cinert',
          'cqLD',
          'cqacc_smooth',
          'cqacc_warmstart',
          'cqfrc_smooth',
          'crb',
          'crhs',
          'cvel',
          'cx',
          'dof_awake_ind',
          'dof_cdof',
          'dof_island',
          'dof_islandid',
          'efc_islandid',
          'efm_K_val',
          'efm_L',
          'efm_as',
          'efm_c',
          'efm_ca',
          'efm_diag',
          'efm_fluid',
          'efm_ts',
          'energy',
          'face_quat',
          'face_xpos',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_rod_chain_ok',
          'flex_rod_chain_valid',
          'flex_rod_end_torque',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_ref_ok',
          'flex_rod_tail_loc',
          'flex_rodedge_frc',
          'flex_rodedge_grad',
          'flex_rodedge_ktheta',
          'flex_rodedge_tridiag',
          'flex_rodhinge_Wd',
          'flex_rodhinge_Wk',
          'flex_rodhinge_g',
          'flex_rodhinge_om0',
          'flexedge_J',
          'flexedge_hessian',
          'flexedge_length',
          'flexedge_velocity',
          'flexnode_xpos',
          'flexvert_hessian',
          'flexvert_xpos',
          'geom_xmat',
          'geom_xpos',
          'history',
          'island_dofadr',
          'island_idofadr',
          'island_iefcadr',
          'island_ne',
          'island_nefc',
          'island_nf',
          'island_nv',
          'light_xdir',
          'light_xpos',
          'map_dof2idof',
          'map_efc2iefc',
          'map_idof2dof',
          'map_iefc2efc',
          'moment_colind',
          'moment_rowadr',
          'moment_rownnz',
          'nacon',
          'nbody_awake',
          'ncdof',
          'ncollision',
          'ne',
          'nefc',
          'nf',
          'nidof',
          'nisland',
          'nl',
          'ntree_awake',
          'nv_awake',
          'overflow',
          'qH',
          'qHDiagInv',
          'qHLD',
          'qLD',
          'qLDiagInv',
          'qacc',
          'qacc_smooth',
          'qfrc_actuator',
          'qfrc_adhesion',
          'qfrc_bias',
          'qfrc_constraint',
          'qfrc_damper',
          'qfrc_fluid',
          'qfrc_gravcomp',
          'qfrc_passive',
          'qfrc_smooth',
          'qfrc_spring',
          'qvel',
          'sensordata',
          'site_xmat',
          'site_xpos',
          'solver_niter',
          'subtree_angmom',
          'subtree_com',
          'subtree_linvel',
          'ten_J',
          'ten_length',
          'ten_velocity',
          'ten_wrapadr',
          'ten_wrapnum',
          'tree_asleep',
          'tree_awake',
          'tree_island',
          'wrap_obj',
          'wrap_xpos',
          'xanchor',
          'xaxis',
          'ximat',
          'xipos',
          'xmat',
          'xpos',
          'xquat',
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
          'efc__D',
          'efc__J',
          'efc__J_colind',
          'efc__J_rowadr',
          'efc__J_rownnz',
          'efc__Jqvel',
          'efc__Ma',
          'efc__aref',
          'efc__force',
          'efc__frictionloss',
          'efc__id',
          'efc__island',
          'efc__jtdaj_adr',
          'efc__jtdaj_nblock',
          'efc__jtdaj_nrow',
          'efc__margin',
          'efc__pos',
          'efc__state',
          'efc__type',
          'efc__vel',
      ]),
      stage_in_argnames=set([
          'M',
          'act',
          'act_dot',
          'actuator_acc0',
          'actuator_actrange',
          'actuator_armature',
          'actuator_biasprm',
          'actuator_cranklength',
          'actuator_ctrlrange',
          'actuator_damping',
          'actuator_dampingpoly',
          'actuator_delay',
          'actuator_dynprm',
          'actuator_force',
          'actuator_forcerange',
          'actuator_gainprm',
          'actuator_gear',
          'actuator_history',
          'actuator_historyadr',
          'actuator_length',
          'actuator_lengthrange',
          'actuator_moment',
          'actuator_velocity',
          'body_awake',
          'body_awake_ind',
          'body_gravcomp',
          'body_inertia',
          'body_invweight0',
          'body_ipos',
          'body_iquat',
          'body_mass',
          'body_pos',
          'body_quat',
          'body_subtreemass',
          'cJ',
          'cM',
          'cMa',
          'cacc',
          'cam_fovy',
          'cam_intrinsic',
          'cam_mat0',
          'cam_pos',
          'cam_pos0',
          'cam_poscom0',
          'cam_quat',
          'cam_xmat',
          'cam_xpos',
          'cdof',
          'cdof_dof',
          'cdof_dot',
          'cdof_tri_col',
          'cdof_tri_row',
          'cfrc_ext',
          'cfrc_int',
          'cinert',
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
          'cqLD',
          'cqacc',
          'cqacc_smooth',
          'cqacc_warmstart',
          'cqfrc_constraint',
          'cqfrc_smooth',
          'crb',
          'crhs',
          'ctrl',
          'cvel',
          'cx',
          'dof_armature',
          'dof_awake_ind',
          'dof_cdof',
          'dof_damping',
          'dof_dampingpoly',
          'dof_frictionloss',
          'dof_invweight0',
          'dof_island',
          'dof_islandid',
          'dof_solimp',
          'dof_solref',
          'efc__D',
          'efc__J',
          'efc__J_colind',
          'efc__J_rowadr',
          'efc__J_rownnz',
          'efc__Jqvel',
          'efc__Ma',
          'efc__aref',
          'efc__force',
          'efc__frictionloss',
          'efc__id',
          'efc__island',
          'efc__jtdaj_adr',
          'efc__jtdaj_nblock',
          'efc__jtdaj_nrow',
          'efc__margin',
          'efc__pos',
          'efc__state',
          'efc__type',
          'efc__vel',
          'efc_islandid',
          'efm_K_val',
          'efm_L',
          'efm_as',
          'efm_c',
          'efm_ca',
          'efm_diag',
          'efm_fluid',
          'efm_ts',
          'energy',
          'eq_active',
          'eq_data',
          'eq_solimp',
          'eq_solref',
          'face_quat',
          'face_xpos',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_hessian_valid',
          'flex_rod_chain_ok',
          'flex_rod_chain_valid',
          'flex_rod_end_torque',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_ref_ok',
          'flex_rod_tail_loc',
          'flex_rodedge_frc',
          'flex_rodedge_grad',
          'flex_rodedge_ktheta',
          'flex_rodedge_tridiag',
          'flex_rodhinge_Wd',
          'flex_rodhinge_Wk',
          'flex_rodhinge_g',
          'flex_rodhinge_om0',
          'flexedge_J',
          'flexedge_hessian',
          'flexedge_length',
          'flexedge_velocity',
          'flexnode_xpos',
          'flexvert_hessian',
          'flexvert_xpos',
          'geom_aabb',
          'geom_adhesion',
          'geom_friction',
          'geom_gap',
          'geom_margin',
          'geom_matid',
          'geom_pos',
          'geom_quat',
          'geom_rbound',
          'geom_rgba',
          'geom_size',
          'geom_solimp',
          'geom_solmix',
          'geom_solref',
          'geom_surfacevel',
          'geom_xmat',
          'geom_xpos',
          'hfield_data',
          'history',
          'island_dofadr',
          'island_idofadr',
          'island_iefcadr',
          'island_ne',
          'island_nefc',
          'island_nf',
          'island_nv',
          'jnt_actfrcrange',
          'jnt_axis',
          'jnt_margin',
          'jnt_pos',
          'jnt_range',
          'jnt_solimp',
          'jnt_solref',
          'jnt_stiffness',
          'jnt_stiffnesspoly',
          'light_dir',
          'light_dir0',
          'light_pos',
          'light_pos0',
          'light_poscom0',
          'light_xdir',
          'light_xpos',
          'map_dof2idof',
          'map_efc2iefc',
          'map_idof2dof',
          'map_iefc2efc',
          'mat_rgba',
          'mocap_pos',
          'mocap_quat',
          'moment_colind',
          'moment_rowadr',
          'moment_rownnz',
          'nacon',
          'nbody_awake',
          'ncdof',
          'ncollision',
          'ne',
          'nefc',
          'nf',
          'nidof',
          'nisland',
          'nl',
          'ntree_awake',
          'nv_awake',
          'opt__ccd_tolerance',
          'opt__density',
          'opt__gravity',
          'opt__impratio_invsqrt',
          'opt__ls_tolerance',
          'opt__magnetic',
          'opt__noslip_tolerance',
          'opt__timestep',
          'opt__tolerance',
          'opt__viscosity',
          'opt__wind',
          'overflow',
          'pair_adhesion',
          'pair_friction',
          'pair_gap',
          'pair_margin',
          'pair_solimp',
          'pair_solref',
          'pair_solreffriction',
          'plastic',
          'qH',
          'qHDiagInv',
          'qHLD',
          'qLD',
          'qLDiagInv',
          'qacc',
          'qacc_smooth',
          'qacc_warmstart',
          'qfrc_actuator',
          'qfrc_adhesion',
          'qfrc_applied',
          'qfrc_bias',
          'qfrc_constraint',
          'qfrc_damper',
          'qfrc_fluid',
          'qfrc_gravcomp',
          'qfrc_passive',
          'qfrc_smooth',
          'qfrc_spring',
          'qpos',
          'qpos0',
          'qpos_spring',
          'qvel',
          'sensor_delay',
          'sensor_history',
          'sensor_historyadr',
          'sensor_interval',
          'sensordata',
          'site_pos',
          'site_quat',
          'site_xmat',
          'site_xpos',
          'solver_niter',
          'subtree_angmom',
          'subtree_com',
          'subtree_linvel',
          'ten_J',
          'ten_length',
          'ten_velocity',
          'ten_wrapadr',
          'ten_wrapnum',
          'tendon_actfrcrange',
          'tendon_armature',
          'tendon_damping',
          'tendon_dampingpoly',
          'tendon_frictionloss',
          'tendon_invweight0',
          'tendon_length0',
          'tendon_lengthspring',
          'tendon_margin',
          'tendon_range',
          'tendon_solimp_fri',
          'tendon_solimp_lim',
          'tendon_solref_fri',
          'tendon_solref_lim',
          'tendon_stiffness',
          'tendon_stiffnesspoly',
          'time',
          'tree_asleep',
          'tree_awake',
          'tree_island',
          'wrap_obj',
          'wrap_xpos',
          'xanchor',
          'xaxis',
          'xfrc_applied',
          'ximat',
          'xipos',
          'xmat',
          'xpos',
          'xquat',
      ]),
      stage_out_argnames=set([
          'M',
          'act_dot',
          'actuator_force',
          'actuator_length',
          'actuator_moment',
          'actuator_velocity',
          'body_awake',
          'body_awake_ind',
          'cJ',
          'cM',
          'cacc',
          'cam_xmat',
          'cam_xpos',
          'cdof',
          'cdof_dof',
          'cdof_dot',
          'cfrc_ext',
          'cfrc_int',
          'cinert',
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
          'cqLD',
          'cqacc_smooth',
          'cqacc_warmstart',
          'cqfrc_smooth',
          'crb',
          'crhs',
          'cvel',
          'cx',
          'dof_awake_ind',
          'dof_cdof',
          'dof_island',
          'dof_islandid',
          'efc__D',
          'efc__J',
          'efc__J_colind',
          'efc__J_rowadr',
          'efc__J_rownnz',
          'efc__Jqvel',
          'efc__Ma',
          'efc__aref',
          'efc__force',
          'efc__frictionloss',
          'efc__id',
          'efc__island',
          'efc__jtdaj_adr',
          'efc__jtdaj_nblock',
          'efc__jtdaj_nrow',
          'efc__margin',
          'efc__pos',
          'efc__state',
          'efc__type',
          'efc__vel',
          'efc_islandid',
          'efm_K_val',
          'efm_L',
          'efm_as',
          'efm_c',
          'efm_ca',
          'efm_diag',
          'efm_fluid',
          'efm_ts',
          'energy',
          'face_quat',
          'face_xpos',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_rod_chain_ok',
          'flex_rod_chain_valid',
          'flex_rod_end_torque',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_ref_ok',
          'flex_rod_tail_loc',
          'flex_rodedge_frc',
          'flex_rodedge_grad',
          'flex_rodedge_ktheta',
          'flex_rodedge_tridiag',
          'flex_rodhinge_Wd',
          'flex_rodhinge_Wk',
          'flex_rodhinge_g',
          'flex_rodhinge_om0',
          'flexedge_J',
          'flexedge_hessian',
          'flexedge_length',
          'flexedge_velocity',
          'flexnode_xpos',
          'flexvert_hessian',
          'flexvert_xpos',
          'geom_xmat',
          'geom_xpos',
          'history',
          'island_dofadr',
          'island_idofadr',
          'island_iefcadr',
          'island_ne',
          'island_nefc',
          'island_nf',
          'island_nv',
          'light_xdir',
          'light_xpos',
          'map_dof2idof',
          'map_efc2iefc',
          'map_idof2dof',
          'map_iefc2efc',
          'moment_colind',
          'moment_rowadr',
          'moment_rownnz',
          'nacon',
          'nbody_awake',
          'ncdof',
          'ncollision',
          'ne',
          'nefc',
          'nf',
          'nidof',
          'nisland',
          'nl',
          'ntree_awake',
          'nv_awake',
          'overflow',
          'qH',
          'qHDiagInv',
          'qHLD',
          'qLD',
          'qLDiagInv',
          'qacc',
          'qacc_smooth',
          'qfrc_actuator',
          'qfrc_adhesion',
          'qfrc_bias',
          'qfrc_constraint',
          'qfrc_damper',
          'qfrc_fluid',
          'qfrc_gravcomp',
          'qfrc_passive',
          'qfrc_smooth',
          'qfrc_spring',
          'qvel',
          'sensordata',
          'site_xmat',
          'site_xpos',
          'solver_niter',
          'subtree_angmom',
          'subtree_com',
          'subtree_linvel',
          'ten_J',
          'ten_length',
          'ten_velocity',
          'ten_wrapadr',
          'ten_wrapnum',
          'tree_asleep',
          'tree_awake',
          'tree_island',
          'wrap_obj',
          'wrap_xpos',
          'xanchor',
          'xaxis',
          'ximat',
          'xipos',
          'xmat',
          'xpos',
          'xquat',
      ]),
      graph_mode=m.opt._impl.graph_mode,
      has_side_effect=False,
  )
  out = jf(
      d.qpos.shape[0],
      m.M_colind,
      m._impl.M_elemid,
      m._impl.M_fullm_i,
      m._impl.M_fullm_j,
      m._impl.M_hinit_i,
      m._impl.M_mulm_col,
      m._impl.M_mulm_madr,
      m._impl.M_mulm_rowadr,
      m.M_rowadr,
      m.M_rownnz,
      m._impl.M_tiles,
      m.actuator_acc0,
      m.actuator_actadr,
      m.actuator_actearly,
      m.actuator_actlimited,
      m.actuator_actnum,
      m.actuator_actrange,
      m._impl.actuator_armature,
      m.actuator_biasprm,
      m.actuator_biastype,
      m.actuator_cranklength,
      m._impl.actuator_ctrladr,
      m.actuator_ctrllimited,
      m._impl.actuator_ctrlnum,
      m.actuator_ctrlrange,
      m._impl.actuator_ctrlspec,
      m._impl.actuator_damping,
      m._impl.actuator_dampingpoly,
      m._impl.actuator_delay,
      m.actuator_dynprm,
      m.actuator_dyntype,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      m.actuator_gainprm,
      m.actuator_gaintype,
      m.actuator_gear,
      m._impl.actuator_history,
      m._impl.actuator_historyadr,
      m.actuator_lengthrange,
      m.actuator_trnid,
      m.actuator_trntype,
      m._impl.actuator_trntype_body_adr,
      m._impl.block_dim,
      m._impl.body_branch_start,
      m._impl.body_branches,
      m.body_dofadr,
      m.body_dofnum,
      m._impl.body_fluid_box_adr,
      m._impl.body_fluid_ellipsoid,
      m._impl.body_fluid_ellipsoid_adr,
      m._impl.body_freeadr,
      m.body_geomadr,
      m.body_geomnum,
      m.body_gravcomp,
      m.body_inertia,
      m.body_invweight0,
      m.body_ipos,
      m.body_iquat,
      m._impl.body_is_free,
      m._impl.body_isdofancestor,
      m.body_jntadr,
      m.body_jntnum,
      m.body_mass,
      m.body_mocapid,
      m.body_parentid,
      m.body_pos,
      m.body_quat,
      m.body_rootid,
      m.body_simple,
      m.body_subtreemass,
      m._impl.body_subtreenum,
      m._impl.body_tree,
      m.body_treeid,
      m.body_weldid,
      m.cam_bodyid,
      m.cam_fovy,
      m.cam_intrinsic,
      m.cam_mat0,
      m.cam_mode,
      m.cam_pos,
      m.cam_pos0,
      m.cam_poscom0,
      m._impl.cam_projection,
      m.cam_quat,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_targetbodyid,
      m.dof_armature,
      m.dof_bodyid,
      m.dof_damping,
      m.dof_dampingpoly,
      m.dof_frictionloss,
      m.dof_invweight0,
      m.dof_jntid,
      m._impl.dof_length,
      m.dof_parentid,
      m.dof_solimp,
      m.dof_solref,
      m.dof_treeid,
      m._impl.dof_tri_col,
      m._impl.dof_tri_row,
      m._impl.efm0_L,
      m._impl.efm0_L_colind,
      m._impl.efm0_L_rowadr,
      m._impl.efm0_L_rownnz,
      m._impl.efm0_active,
      m._impl.efm0_dofid,
      m._impl.efm_K_colind,
      m._impl.efm_K_rowadr,
      m._impl.efm_K_rownnz,
      m._impl.efm_con_max_dof,
      m._impl.efm_dofblk,
      m._impl.efm_dofid,
      m._impl.eq_connect_adr,
      m.eq_data,
      m._impl.eq_flex_adr,
      m._impl.eq_flexstrain_adr,
      m._impl.eq_jnt_adr,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.eq_solimp,
      m.eq_solref,
      m._impl.eq_ten_adr,
      m.eq_type,
      m._impl.eq_wld_adr,
      m._impl.flex_activelayers,
      m._impl.flex_bend_interp_map,
      m._impl.flex_bending,
      m._impl.flex_bendingadr,
      m._impl.flex_cell_map,
      m._impl.flex_cellnum,
      m._impl.flex_centered,
      m._impl.flex_conaffinity,
      m._impl.flex_condim,
      m._impl.flex_contype,
      m._impl.flex_damping,
      m._impl.flex_dim,
      m._impl.flex_edge,
      m._impl.flex_edgeadr,
      m._impl.flex_edgedamping,
      m._impl.flex_edgeequality,
      m._impl.flex_edgeflap,
      m._impl.flex_edgeflexid,
      m._impl.flex_edgenum,
      m._impl.flex_edgestiffness,
      m._impl.flex_eff_possible,
      m._impl.flex_elem,
      m._impl.flex_elem_rodedge,
      m._impl.flex_elemadr,
      m._impl.flex_elemdataadr,
      m._impl.flex_elemedge,
      m._impl.flex_elemedgeadr,
      m._impl.flex_elemflexid,
      m._impl.flex_elemlayer,
      m._impl.flex_elemnum,
      m._impl.flex_face,
      m._impl.flex_face_map,
      m._impl.flex_faceadr,
      m._impl.flex_friction,
      m._impl.flex_gap,
      m.flex_interp,
      m._impl.flex_interp_assemblable,
      m._impl.flex_margin,
      m._impl.flex_node,
      m.flex_node0,
      m.flex_nodeadr,
      m.flex_nodebodyid,
      m.flex_nodenum,
      m._impl.flex_passive,
      m._impl.flex_plasticadr,
      m._impl.flex_plasticmode,
      m._impl.flex_priority,
      m._impl.flex_radius,
      m._impl.flex_rigid,
      m._impl.flex_rod_box_size,
      m._impl.flex_rod_edgeadr,
      m._impl.flex_rod_edgenum,
      m._impl.flex_rod_has_twist,
      m._impl.flex_rod_hingeadr,
      m._impl.flex_rod_is_box,
      m._impl.flex_rod_is_chain,
      m._impl.flex_rod_root_clamped,
      m._impl.flex_rod_tail_clamped,
      m._impl.flex_rodedge_jnt,
      m._impl.flex_rodedge_vert,
      m._impl.flex_rodhinge_map,
      m._impl.flex_selfcollide,
      m._impl.flex_simple,
      m._impl.flex_size,
      m._impl.flex_solimp,
      m._impl.flex_solmix,
      m._impl.flex_solref,
      m._impl.flex_stiffness,
      m._impl.flex_stiffnessadr,
      m._impl.flex_vert,
      m.flex_vert0,
      m._impl.flex_vert_rodelem,
      m.flex_vertadr,
      m._impl.flex_vertbodyid,
      m._impl.flex_vertcell,
      m._impl.flex_vertflexid,
      m.flex_vertnum,
      m._impl.flexedge_J_colind,
      m._impl.flexedge_J_rowadr,
      m._impl.flexedge_J_rownnz,
      m._impl.flexedge_invweight0,
      m._impl.flexedge_length0,
      m._impl.flexedge_rigid,
      m._impl.flexstrain_J_colind,
      m._impl.flexstrain_J_rowadr,
      m._impl.flexstrain_J_rownnz,
      m._impl.flg_adhesion,
      m.flg_surfacevel,
      m.geom_aabb,
      m._impl.geom_adhesion,
      m.geom_bodyid,
      m.geom_conaffinity,
      m.geom_condim,
      m.geom_contype,
      jax.numpy.expand_dims(m.geom_dataid, 0),
      m.geom_fluid,
      m.geom_friction,
      m.geom_gap,
      m.geom_group,
      m.geom_margin,
      m.geom_matid,
      m._impl.geom_pair_type_count,
      m._impl.geom_plugin_index,
      m.geom_pos,
      m.geom_priority,
      m.geom_quat,
      m.geom_rbound,
      m.geom_rgba,
      m.geom_size,
      m.geom_solimp,
      m.geom_solmix,
      m.geom_solref,
      m._impl.geom_surfacevel,
      m.geom_type,
      m._impl.has_1d_flex,
      m._impl.has_3d_flex,
      m._impl.has_efm_actuator,
      m._impl.has_ellipsoid_geom,
      m._impl.has_flex_edge_stiff_damp,
      m._impl.has_flex_passive,
      m._impl.has_flex_selfcollide,
      m._impl.has_flex_snh,
      m._impl.has_fluid,
      m._impl.has_non_simple_flex,
      m._impl.has_plane_geom,
      m._impl.has_sdf_geom,
      m._impl.has_tendon_damping,
      m._impl.has_tendon_stiffness,
      m._impl.has_unsupported_flex_interp,
      m.hfield_adr,
      m.hfield_data,
      m.hfield_ncol,
      m.hfield_nrow,
      m.hfield_size,
      m._impl.is_sparse,
      m.jnt_actfrclimited,
      m.jnt_actfrcrange,
      m.jnt_actgravcomp,
      m._impl.jnt_actuatorid,
      m.jnt_axis,
      m.jnt_bodyid,
      m.jnt_dofadr,
      m._impl.jnt_limited_ball_adr,
      m._impl.jnt_limited_slide_hinge_adr,
      m.jnt_margin,
      m._impl.jnt_plasticadr,
      m.jnt_pos,
      m.jnt_qposadr,
      m.jnt_range,
      m.jnt_solimp,
      m.jnt_solref,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.jnt_type,
      m._impl.light_bodyid,
      m.light_dir,
      m.light_dir0,
      m.light_mode,
      m.light_pos,
      m.light_pos0,
      m.light_poscom0,
      m._impl.light_targetbodyid,
      m.mat_rgba,
      m._impl.max_ten_J_rownnz,
      m.mesh_face,
      m.mesh_faceadr,
      m.mesh_graph,
      m.mesh_graphadr,
      m.mesh_normal,
      m.mesh_normaladr,
      m.mesh_normalnum,
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
      m.mesh_quat,
      m.mesh_vert,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.nC,
      m.nJten,
      m.na,
      m._impl.nacttrnbody,
      m._impl.nactuator,
      m.nbody,
      m._impl.nbranch,
      m.ncam,
      m._impl.nefm0dof,
      m._impl.nefmK,
      m._impl.nefmdof,
      m.neq,
      m.nflex,
      m._impl.nflexbend_interp,
      m._impl.nflexedge,
      m._impl.nflexelem,
      m._impl.nflexface,
      m._impl.nflexintcell,
      m._impl.nflexnode,
      m._impl.nflexrodedge,
      m._impl.nflexrodhinge,
      m._impl.nflexvert,
      m.ngeom,
      m.nhistory,
      m.njnt,
      m.nlight,
      m._impl.nmaxcondim,
      m._impl.nmaxpyramid,
      m.nmesh,
      m._impl.nmeshdegmax,
      m.nmeshface,
      m.npair,
      m._impl.npolygonmax,
      m._impl.nrangefinder,
      m._impl.nsensorcollision,
      m._impl.nsensortaxel,
      m.nsite,
      m._impl.ntactileweld,
      m.ntendon,
      m._impl.ntree,
      m.nu,
      m.nv,
      m._impl.nv_pad,
      m.nwrap,
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
      m._impl.qLD_all_updates,
      m._impl.qLD_block_adr,
      m._impl.qLD_block_total,
      m._impl.qLD_level_offsets,
      m._impl.qLD_updates,
      m.qpos0,
      m.qpos_spring,
      m._impl.rangefinder_sensor_adr,
      m._impl.sensor_acc_adr,
      m.sensor_adr,
      m._impl.sensor_contact_acc_adr,
      m._impl.sensor_contact_pos_adr,
      m.sensor_cutoff,
      m.sensor_datatype,
      m._impl.sensor_delay,
      m.sensor_dim,
      m._impl.sensor_e_kinetic,
      m._impl.sensor_e_potential,
      m._impl.sensor_history,
      m._impl.sensor_historyadr,
      m._impl.sensor_interval,
      m.sensor_intprm,
      m._impl.sensor_limitfrc_adr,
      m._impl.sensor_limitpos_adr,
      m._impl.sensor_limitvel_adr,
      m.sensor_objid,
      m.sensor_objtype,
      m._impl.sensor_pos_adr,
      m._impl.sensor_rangefinder_adr,
      m._impl.sensor_rangefinder_bodyid,
      m.sensor_refid,
      m.sensor_reftype,
      m._impl.sensor_rne_postconstraint,
      m._impl.sensor_stage_acc_adr,
      m._impl.sensor_stage_pos_adr,
      m._impl.sensor_stage_vel_adr,
      m._impl.sensor_subtree_vel,
      m._impl.sensor_tendonactfrc_adr,
      m._impl.sensor_touch_adr,
      m.sensor_type,
      m._impl.sensor_vel_adr,
      m.site_bodyid,
      m.site_pos,
      m.site_quat,
      m.site_size,
      m.site_type,
      m._impl.taxel_sensorid,
      m._impl.taxel_vertadr,
      m._impl.ten_J_colind,
      m._impl.ten_J_rowadr,
      m._impl.ten_J_rownnz,
      m.tendon_actfrclimited,
      m.tendon_actfrcrange,
      m._impl.tendon_actuatorid,
      m.tendon_adr,
      m.tendon_armature,
      m.tendon_damping,
      m.tendon_dampingpoly,
      m.tendon_frictionloss,
      m._impl.tendon_geom_adr,
      m.tendon_invweight0,
      m._impl.tendon_jnt_adr,
      m.tendon_length0,
      m.tendon_lengthspring,
      m.tendon_limited,
      m._impl.tendon_limited_adr,
      m.tendon_margin,
      m.tendon_num,
      m._impl.tendon_plasticadr,
      m.tendon_range,
      m._impl.tendon_site_pair_adr,
      m.tendon_solimp_fri,
      m.tendon_solimp_lim,
      m.tendon_solref_fri,
      m.tendon_solref_lim,
      m.tendon_stiffness,
      m.tendon_stiffnesspoly,
      m.tendon_type,
      m._impl.tree_dofadr,
      m._impl.tree_dofnum,
      m._impl.tree_sleep_policy,
      m._impl.weld_tactile_id,
      m._impl.wrap_geom_adr,
      m._impl.wrap_jnt_adr,
      m.wrap_objid,
      m.wrap_prm,
      m._impl.wrap_pulley_scale,
      m._impl.wrap_site_pair_adr,
      m.wrap_type,
      m.opt._impl.broadphase,
      m.opt._impl.broadphase_filter,
      m.opt._impl.ccd_iterations,
      m.opt._impl.ccd_tolerance,
      m.opt.cone,
      m.opt._impl.contact_sensor_maxmatch,
      m.opt.density,
      m.opt._impl.deterministic,
      m.opt.disableflags,
      m.opt.enableflags,
      m.opt._impl.graph_conditional,
      m.opt.gravity,
      m.opt._impl.impratio_invsqrt,
      m.opt.integrator,
      m.opt.iterations,
      m.opt.ls_iterations,
      m.opt.ls_tolerance,
      m.opt.magnetic,
      m.opt._impl.noslip_iterations,
      m.opt._impl.noslip_tolerance,
      m.opt._impl.run_collision_detection,
      m.opt._impl.run_rne_postconstraint,
      m.opt._impl.sdf_initpoints,
      m.opt._impl.sdf_iterations,
      m.opt.solver,
      m.opt.timestep,
      m.opt.tolerance,
      m.opt.viscosity,
      m.opt._impl.warn_overflow,
      m.opt.wind,
      d._impl.naccdmax,
      d._impl.naconmax,
      d._impl.njmax,
      d._impl.njmax_nnz,
      d._impl.nvmax,
      d._impl.nvmax_pad,
      d._impl.M,
      d.act,
      d.act_dot,
      d.actuator_force,
      d.actuator_length,
      d._impl.actuator_moment,
      d._impl.actuator_velocity,
      d._impl.body_awake,
      d._impl.body_awake_ind,
      d._impl.cJ,
      d._impl.cM,
      d._impl.cMa,
      d._impl.cacc,
      d.cam_xmat,
      d.cam_xpos,
      d.cdof,
      d._impl.cdof_dof,
      d.cdof_dot,
      d._impl.cdof_tri_col,
      d._impl.cdof_tri_row,
      d._impl.cfrc_ext,
      d._impl.cfrc_int,
      d._impl.cinert,
      d._impl.cqLD,
      d._impl.cqacc,
      d._impl.cqacc_smooth,
      d._impl.cqacc_warmstart,
      d._impl.cqfrc_constraint,
      d._impl.cqfrc_smooth,
      d._impl.crb,
      d._impl.crhs,
      d.ctrl,
      d.cvel,
      d._impl.cx,
      d._impl.dof_awake_ind,
      d._impl.dof_cdof,
      d._impl.dof_island,
      d._impl.dof_islandid,
      d._impl.efc_islandid,
      d._impl.efm_K_val,
      d._impl.efm_L,
      d._impl.efm_as,
      d._impl.efm_c,
      d._impl.efm_ca,
      d._impl.efm_diag,
      d._impl.efm_fluid,
      d._impl.efm_ts,
      d._impl.energy,
      d.eq_active,
      d._impl.face_quat,
      d._impl.face_xpos,
      d._impl.flex_aabb_max,
      d._impl.flex_aabb_min,
      d._impl.flex_hessian_valid,
      d._impl.flex_rod_chain_ok,
      d._impl.flex_rod_chain_valid,
      d._impl.flex_rod_end_torque,
      d._impl.flex_rod_frame0_loc,
      d._impl.flex_rod_frame0_ref,
      d._impl.flex_rod_ref_ok,
      d._impl.flex_rod_tail_loc,
      d._impl.flex_rodedge_frc,
      d._impl.flex_rodedge_grad,
      d._impl.flex_rodedge_ktheta,
      d._impl.flex_rodedge_tridiag,
      d._impl.flex_rodhinge_Wd,
      d._impl.flex_rodhinge_Wk,
      d._impl.flex_rodhinge_g,
      d._impl.flex_rodhinge_om0,
      d._impl.flexedge_J,
      d._impl.flexedge_hessian,
      d._impl.flexedge_length,
      d._impl.flexedge_velocity,
      d._impl.flexnode_xpos,
      d._impl.flexvert_hessian,
      d._impl.flexvert_xpos,
      d.geom_xmat,
      d.geom_xpos,
      d.history,
      d._impl.island_dofadr,
      d._impl.island_idofadr,
      d._impl.island_iefcadr,
      d._impl.island_ne,
      d._impl.island_nefc,
      d._impl.island_nf,
      d._impl.island_nv,
      d._impl.light_xdir,
      d._impl.light_xpos,
      d._impl.map_dof2idof,
      d._impl.map_efc2iefc,
      d._impl.map_idof2dof,
      d._impl.map_iefc2efc,
      d.mocap_pos,
      d.mocap_quat,
      d._impl.moment_colind,
      d._impl.moment_rowadr,
      d._impl.moment_rownnz,
      d._impl.nacon,
      d._impl.nbody_awake,
      d._impl.ncdof,
      d._impl.ncollision,
      d._impl.ne,
      d._impl.nefc,
      d._impl.nf,
      d._impl.nidof,
      d._impl.nisland,
      d._impl.nl,
      d._impl.ntree_awake,
      d._impl.nv_awake,
      d._impl.overflow,
      d.plastic,
      d._impl.qH,
      d._impl.qHDiagInv,
      d._impl.qHLD,
      d._impl.qLD,
      d._impl.qLDiagInv,
      d.qacc,
      d.qacc_smooth,
      d.qacc_warmstart,
      d.qfrc_actuator,
      d._impl.qfrc_adhesion,
      d.qfrc_applied,
      d.qfrc_bias,
      d.qfrc_constraint,
      d._impl.qfrc_damper,
      d.qfrc_fluid,
      d.qfrc_gravcomp,
      d.qfrc_passive,
      d.qfrc_smooth,
      d._impl.qfrc_spring,
      d.qpos,
      d.qvel,
      d.sensordata,
      d.site_xmat,
      d.site_xpos,
      d._impl.solver_niter,
      d._impl.subtree_angmom,
      d.subtree_com,
      d._impl.subtree_linvel,
      d._impl.ten_J,
      d.ten_length,
      d._impl.ten_velocity,
      d._impl.ten_wrapadr,
      d._impl.ten_wrapnum,
      d.time,
      d._impl.tree_asleep,
      d._impl.tree_awake,
      d._impl.tree_island,
      d._impl.wrap_obj,
      d._impl.wrap_xpos,
      d.xanchor,
      d.xaxis,
      d.xfrc_applied,
      d.ximat,
      d.xipos,
      d.xmat,
      d.xpos,
      d.xquat,
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
      d._impl.efc__D,
      d._impl.efc__J,
      d._impl.efc__J_colind,
      d._impl.efc__J_rowadr,
      d._impl.efc__J_rownnz,
      d._impl.efc__Jqvel,
      d._impl.efc__Ma,
      d._impl.efc__aref,
      d._impl.efc__force,
      d._impl.efc__frictionloss,
      d._impl.efc__id,
      d._impl.efc__island,
      d._impl.efc__jtdaj_adr,
      d._impl.efc__jtdaj_nblock,
      d._impl.efc__jtdaj_nrow,
      d._impl.efc__margin,
      d._impl.efc__pos,
      d._impl.efc__state,
      d._impl.efc__type,
      d._impl.efc__vel,
  )
  d = d.tree_replace({
      '_impl.M': out[0],
      'act_dot': out[1],
      'actuator_force': out[2],
      'actuator_length': out[3],
      '_impl.actuator_moment': out[4],
      '_impl.actuator_velocity': out[5],
      '_impl.body_awake': out[6],
      '_impl.body_awake_ind': out[7],
      '_impl.cJ': out[8],
      '_impl.cM': out[9],
      '_impl.cacc': out[10],
      'cam_xmat': out[11],
      'cam_xpos': out[12],
      'cdof': out[13],
      '_impl.cdof_dof': out[14],
      'cdof_dot': out[15],
      '_impl.cfrc_ext': out[16],
      '_impl.cfrc_int': out[17],
      '_impl.cinert': out[18],
      '_impl.cqLD': out[19],
      '_impl.cqacc_smooth': out[20],
      '_impl.cqacc_warmstart': out[21],
      '_impl.cqfrc_smooth': out[22],
      '_impl.crb': out[23],
      '_impl.crhs': out[24],
      'cvel': out[25],
      '_impl.cx': out[26],
      '_impl.dof_awake_ind': out[27],
      '_impl.dof_cdof': out[28],
      '_impl.dof_island': out[29],
      '_impl.dof_islandid': out[30],
      '_impl.efc_islandid': out[31],
      '_impl.efm_K_val': out[32],
      '_impl.efm_L': out[33],
      '_impl.efm_as': out[34],
      '_impl.efm_c': out[35],
      '_impl.efm_ca': out[36],
      '_impl.efm_diag': out[37],
      '_impl.efm_fluid': out[38],
      '_impl.efm_ts': out[39],
      '_impl.energy': out[40],
      '_impl.face_quat': out[41],
      '_impl.face_xpos': out[42],
      '_impl.flex_aabb_max': out[43],
      '_impl.flex_aabb_min': out[44],
      '_impl.flex_rod_chain_ok': out[45],
      '_impl.flex_rod_chain_valid': out[46],
      '_impl.flex_rod_end_torque': out[47],
      '_impl.flex_rod_frame0_loc': out[48],
      '_impl.flex_rod_frame0_ref': out[49],
      '_impl.flex_rod_ref_ok': out[50],
      '_impl.flex_rod_tail_loc': out[51],
      '_impl.flex_rodedge_frc': out[52],
      '_impl.flex_rodedge_grad': out[53],
      '_impl.flex_rodedge_ktheta': out[54],
      '_impl.flex_rodedge_tridiag': out[55],
      '_impl.flex_rodhinge_Wd': out[56],
      '_impl.flex_rodhinge_Wk': out[57],
      '_impl.flex_rodhinge_g': out[58],
      '_impl.flex_rodhinge_om0': out[59],
      '_impl.flexedge_J': out[60],
      '_impl.flexedge_hessian': out[61],
      '_impl.flexedge_length': out[62],
      '_impl.flexedge_velocity': out[63],
      '_impl.flexnode_xpos': out[64],
      '_impl.flexvert_hessian': out[65],
      '_impl.flexvert_xpos': out[66],
      'geom_xmat': out[67],
      'geom_xpos': out[68],
      'history': out[69],
      '_impl.island_dofadr': out[70],
      '_impl.island_idofadr': out[71],
      '_impl.island_iefcadr': out[72],
      '_impl.island_ne': out[73],
      '_impl.island_nefc': out[74],
      '_impl.island_nf': out[75],
      '_impl.island_nv': out[76],
      '_impl.light_xdir': out[77],
      '_impl.light_xpos': out[78],
      '_impl.map_dof2idof': out[79],
      '_impl.map_efc2iefc': out[80],
      '_impl.map_idof2dof': out[81],
      '_impl.map_iefc2efc': out[82],
      '_impl.moment_colind': out[83],
      '_impl.moment_rowadr': out[84],
      '_impl.moment_rownnz': out[85],
      '_impl.nacon': out[86],
      '_impl.nbody_awake': out[87],
      '_impl.ncdof': out[88],
      '_impl.ncollision': out[89],
      '_impl.ne': out[90],
      '_impl.nefc': out[91],
      '_impl.nf': out[92],
      '_impl.nidof': out[93],
      '_impl.nisland': out[94],
      '_impl.nl': out[95],
      '_impl.ntree_awake': out[96],
      '_impl.nv_awake': out[97],
      '_impl.overflow': out[98],
      '_impl.qH': out[99],
      '_impl.qHDiagInv': out[100],
      '_impl.qHLD': out[101],
      '_impl.qLD': out[102],
      '_impl.qLDiagInv': out[103],
      'qacc': out[104],
      'qacc_smooth': out[105],
      'qfrc_actuator': out[106],
      '_impl.qfrc_adhesion': out[107],
      'qfrc_bias': out[108],
      'qfrc_constraint': out[109],
      '_impl.qfrc_damper': out[110],
      'qfrc_fluid': out[111],
      'qfrc_gravcomp': out[112],
      'qfrc_passive': out[113],
      'qfrc_smooth': out[114],
      '_impl.qfrc_spring': out[115],
      'qvel': out[116],
      'sensordata': out[117],
      'site_xmat': out[118],
      'site_xpos': out[119],
      '_impl.solver_niter': out[120],
      '_impl.subtree_angmom': out[121],
      'subtree_com': out[122],
      '_impl.subtree_linvel': out[123],
      '_impl.ten_J': out[124],
      'ten_length': out[125],
      '_impl.ten_velocity': out[126],
      '_impl.ten_wrapadr': out[127],
      '_impl.ten_wrapnum': out[128],
      '_impl.tree_asleep': out[129],
      '_impl.tree_awake': out[130],
      '_impl.tree_island': out[131],
      '_impl.wrap_obj': out[132],
      '_impl.wrap_xpos': out[133],
      'xanchor': out[134],
      'xaxis': out[135],
      'ximat': out[136],
      'xipos': out[137],
      'xmat': out[138],
      'xpos': out[139],
      'xquat': out[140],
      '_impl.contact__adhesion': out[141],
      '_impl.contact__dim': out[142],
      '_impl.contact__dist': out[143],
      '_impl.contact__efc_address': out[144],
      '_impl.contact__elem': out[145],
      '_impl.contact__flex': out[146],
      '_impl.contact__frame': out[147],
      '_impl.contact__friction': out[148],
      '_impl.contact__geom': out[149],
      '_impl.contact__geomcollisionid': out[150],
      '_impl.contact__includemargin': out[151],
      '_impl.contact__pos': out[152],
      '_impl.contact__solimp': out[153],
      '_impl.contact__solref': out[154],
      '_impl.contact__solreffriction': out[155],
      '_impl.contact__surfacevel': out[156],
      '_impl.contact__type': out[157],
      '_impl.contact__vert': out[158],
      '_impl.contact__worldid': out[159],
      '_impl.efc__D': out[160],
      '_impl.efc__J': out[161],
      '_impl.efc__J_colind': out[162],
      '_impl.efc__J_rowadr': out[163],
      '_impl.efc__J_rownnz': out[164],
      '_impl.efc__Jqvel': out[165],
      '_impl.efc__Ma': out[166],
      '_impl.efc__aref': out[167],
      '_impl.efc__force': out[168],
      '_impl.efc__frictionloss': out[169],
      '_impl.efc__id': out[170],
      '_impl.efc__island': out[171],
      '_impl.efc__jtdaj_adr': out[172],
      '_impl.efc__jtdaj_nblock': out[173],
      '_impl.efc__jtdaj_nrow': out[174],
      '_impl.efc__margin': out[175],
      '_impl.efc__pos': out[176],
      '_impl.efc__state': out[177],
      '_impl.efc__type': out[178],
      '_impl.efc__vel': out[179],
  })
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def forward(m: types.Model, d: types.Data):
  return _forward_jax_impl(m, d)


@forward.def_vmap
@ffi.marshal_custom_vmap
def forward_vmap(unused_axis_size, is_batched, m: types.Model, d: types.Data):
  d = forward(m, d)
  return d, is_batched[1]



@ffi.format_args_for_warp
def _step_shim(
    # Model
    nworld: int,
    D_colind: wp.array[int],
    D_diag: wp.array[int],
    D_rowadr: wp.array[int],
    D_rownnz: wp.array[int],
    M_colind: wp.array[int],
    M_elemid: wp.array2d[int],
    M_fullm_i: wp.array[int],
    M_fullm_j: wp.array[int],
    M_hinit_i: wp.array[int],
    M_mulm_col: wp.array[int],
    M_mulm_madr: wp.array[int],
    M_mulm_rowadr: wp.array[int],
    M_rowadr: wp.array[int],
    M_rownnz: wp.array[int],
    M_tiles: tuple[mjwp_types.TileSet, ...],
    actuator_acc0: wp.array2d[float],
    actuator_actadr: wp.array[int],
    actuator_actearly: wp.array[bool],
    actuator_actlimited: wp.array[bool],
    actuator_actnum: wp.array[int],
    actuator_actrange: wp.array2d[wp.vec2],
    actuator_armature: wp.array2d[float],
    actuator_biasprm: wp.array2d[mjwp_types.vec10],
    actuator_biastype: wp.array[int],
    actuator_cranklength: wp.array2d[float],
    actuator_ctrladr: wp.array[int],
    actuator_ctrllimited: wp.array[bool],
    actuator_ctrlnum: wp.array[int],
    actuator_ctrlrange: wp.array2d[wp.vec2],
    actuator_ctrlspec: wp.array[int],
    actuator_damping: wp.array2d[float],
    actuator_dampingpoly: wp.array2d[wp.vec2],
    actuator_delay: wp.array2d[float],
    actuator_dynprm: wp.array2d[mjwp_types.vec10],
    actuator_dyntype: wp.array[int],
    actuator_forcelimited: wp.array[bool],
    actuator_forcerange: wp.array2d[wp.vec2],
    actuator_gainprm: wp.array2d[mjwp_types.vec10],
    actuator_gaintype: wp.array[int],
    actuator_gear: wp.array2d[wp.spatial_vector],
    actuator_history: wp.array2d[wp.vec2i],
    actuator_historyadr: wp.array2d[int],
    actuator_lengthrange: wp.array2d[wp.vec2],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_trntype: wp.array[int],
    actuator_trntype_body_adr: wp.array[int],
    block_dim: mjwp_types.BlockDim,
    body_branch_start: wp.array[int],
    body_branches: wp.array[int],
    body_dofadr: wp.array[int],
    body_dofnum: wp.array[int],
    body_fluid_box_adr: wp.array[int],
    body_fluid_ellipsoid: wp.array[bool],
    body_fluid_ellipsoid_adr: wp.array[int],
    body_freeadr: wp.array[int],
    body_geomadr: wp.array[int],
    body_geomnum: wp.array[int],
    body_gravcomp: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    body_invweight0: wp.array2d[wp.vec2],
    body_ipos: wp.array2d[wp.vec3],
    body_iquat: wp.array2d[wp.quat],
    body_is_free: wp.array[bool],
    body_isdofancestor: wp.array2d[int],
    body_jntadr: wp.array[int],
    body_jntnum: wp.array[int],
    body_mass: wp.array2d[float],
    body_mocapid: wp.array[int],
    body_parentid: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    body_rootid: wp.array[int],
    body_simple: wp.array[int],
    body_subtreemass: wp.array2d[float],
    body_subtreenum: wp.array[int],
    body_tree: tuple[wp.array[int], ...],
    body_treeid: wp.array[int],
    body_weldid: wp.array[int],
    cam_bodyid: wp.array[int],
    cam_fovy: wp.array2d[float],
    cam_intrinsic: wp.array2d[wp.vec4],
    cam_mat0: wp.array2d[wp.mat33],
    cam_mode: wp.array[int],
    cam_pos: wp.array2d[wp.vec3],
    cam_pos0: wp.array2d[wp.vec3],
    cam_poscom0: wp.array2d[wp.vec3],
    cam_projection: wp.array[int],
    cam_quat: wp.array2d[wp.quat],
    cam_resolution: wp.array[wp.vec2i],
    cam_sensorsize: wp.array[wp.vec2],
    cam_targetbodyid: wp.array[int],
    dof_armature: wp.array2d[float],
    dof_bodyid: wp.array[int],
    dof_damping: wp.array2d[float],
    dof_dampingpoly: wp.array2d[wp.vec2],
    dof_frictionloss: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    dof_jntid: wp.array[int],
    dof_length: wp.array[float],
    dof_parentid: wp.array[int],
    dof_solimp: wp.array2d[mjwp_types.vec5],
    dof_solref: wp.array2d[wp.vec2],
    dof_treeid: wp.array[int],
    dof_tri_col: wp.array[int],
    dof_tri_row: wp.array[int],
    efm0_L: wp.array[float],
    efm0_L_colind: wp.array[int],
    efm0_L_rowadr: wp.array[int],
    efm0_L_rownnz: wp.array[int],
    efm0_active: bool,
    efm0_dofid: wp.array[int],
    efm_K_colind: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_rownnz: wp.array[int],
    efm_con_max_dof: int,
    efm_dofblk: wp.array[int],
    efm_dofid: wp.array[int],
    eq_connect_adr: wp.array[int],
    eq_data: wp.array2d[mjwp_types.vec12],
    eq_flex_adr: wp.array[int],
    eq_flexstrain_adr: wp.array[int],
    eq_jnt_adr: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_objtype: wp.array[int],
    eq_solimp: wp.array2d[mjwp_types.vec5],
    eq_solref: wp.array2d[wp.vec2],
    eq_ten_adr: wp.array[int],
    eq_type: wp.array[int],
    eq_wld_adr: wp.array[int],
    flex_activelayers: wp.array[int],
    flex_bend_interp_map: wp.array[wp.vec2i],
    flex_bending: wp.array[float],
    flex_bendingadr: wp.array[int],
    flex_cell_map: wp.array[wp.vec4i],
    flex_cellnum: wp.array[wp.vec3i],
    flex_centered: wp.array[bool],
    flex_conaffinity: wp.array[int],
    flex_condim: wp.array[int],
    flex_contype: wp.array[int],
    flex_damping: wp.array[float],
    flex_dim: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_edgeadr: wp.array[int],
    flex_edgedamping: wp.array[float],
    flex_edgeequality: wp.array[int],
    flex_edgeflap: wp.array[wp.vec2i],
    flex_edgeflexid: wp.array[int],
    flex_edgenum: wp.array[int],
    flex_edgestiffness: wp.array[float],
    flex_eff_possible: wp.array[bool],
    flex_elem: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elemedge: wp.array[int],
    flex_elemedgeadr: wp.array[int],
    flex_elemflexid: wp.array[int],
    flex_elemlayer: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_face: wp.array2d[int],
    flex_face_map: wp.array[wp.vec2i],
    flex_faceadr: wp.array[int],
    flex_friction: wp.array[wp.vec3],
    flex_gap: wp.array[float],
    flex_interp: wp.array[int],
    flex_interp_assemblable: bool,
    flex_margin: wp.array[float],
    flex_node: wp.array[wp.vec3],
    flex_node0: wp.array[wp.vec3],
    flex_nodeadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_nodenum: wp.array[int],
    flex_passive: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_plasticmode: wp.array[int],
    flex_priority: wp.array[int],
    flex_radius: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_rod_box_size: wp.array[wp.vec2],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_hingeadr: wp.array[int],
    flex_rod_is_box: wp.array[bool],
    flex_rod_is_chain: wp.array[bool],
    flex_rod_root_clamped: wp.array[bool],
    flex_rod_tail_clamped: wp.array[bool],
    flex_rodedge_jnt: wp.array[int],
    flex_rodedge_vert: wp.array[wp.vec2i],
    flex_rodhinge_map: wp.array[wp.vec2i],
    flex_selfcollide: wp.array[int],
    flex_simple: wp.array[bool],
    flex_size: wp.array[wp.vec3],
    flex_solimp: wp.array[mjwp_types.vec5],
    flex_solmix: wp.array[float],
    flex_solref: wp.array[wp.vec2],
    flex_stiffness: wp.array[float],
    flex_stiffnessadr: wp.array[int],
    flex_vert: wp.array[wp.vec3],
    flex_vert0: wp.array[wp.vec3],
    flex_vert_rodelem: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_vertcell: wp.array[int],
    flex_vertflexid: wp.array[int],
    flex_vertnum: wp.array[int],
    flex_yield: wp.array[wp.vec2],
    flexedge_J_colind: wp.array[int],
    flexedge_J_rowadr: wp.array[int],
    flexedge_J_rownnz: wp.array[int],
    flexedge_invweight0: wp.array[float],
    flexedge_length0: wp.array[float],
    flexedge_rigid: wp.array[bool],
    flexstrain_J_colind: wp.array[int],
    flexstrain_J_rowadr: wp.array[int],
    flexstrain_J_rownnz: wp.array[int],
    flg_adhesion: bool,
    flg_surfacevel: bool,
    geom_aabb: wp.array3d[wp.vec3],
    geom_adhesion: wp.array2d[float],
    geom_bodyid: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_condim: wp.array[int],
    geom_contype: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_fluid: wp.array2d[float],
    geom_friction: wp.array2d[wp.vec3],
    geom_gap: wp.array2d[float],
    geom_group: wp.array[int],
    geom_margin: wp.array2d[float],
    geom_matid: wp.array2d[int],
    geom_pair_type_count: tuple[int, ...],
    geom_plugin_index: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_priority: wp.array[int],
    geom_quat: wp.array2d[wp.quat],
    geom_rbound: wp.array2d[float],
    geom_rgba: wp.array2d[wp.vec4],
    geom_size: wp.array2d[wp.vec3],
    geom_solimp: wp.array2d[mjwp_types.vec5],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_surfacevel: wp.array2d[mjwp_types.vec6],
    geom_type: wp.array[int],
    has_1d_flex: bool,
    has_3d_flex: bool,
    has_actuator_dcmotor: bool,
    has_actuator_wrap: bool,
    has_efm_actuator: bool,
    has_ellipsoid_geom: bool,
    has_flex_edge_stiff_damp: bool,
    has_flex_passive: bool,
    has_flex_selfcollide: bool,
    has_flex_snh: bool,
    has_fluid: bool,
    has_non_simple_flex: bool,
    has_plane_geom: bool,
    has_sdf_geom: bool,
    has_tendon_damping: bool,
    has_tendon_stiffness: bool,
    has_unsupported_flex_interp: bool,
    hfield_adr: wp.array[int],
    hfield_data: wp.array[float],
    hfield_ncol: wp.array[int],
    hfield_nrow: wp.array[int],
    hfield_size: wp.array[wp.vec4],
    is_sparse: bool,
    jnt_actfrclimited: wp.array[bool],
    jnt_actfrcrange: wp.array2d[wp.vec2],
    jnt_actgravcomp: wp.array[int],
    jnt_actuatorid: wp.array[int],
    jnt_axis: wp.array2d[wp.vec3],
    jnt_bodyid: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_limited_ball_adr: wp.array[int],
    jnt_limited_slide_hinge_adr: wp.array[int],
    jnt_margin: wp.array2d[float],
    jnt_plasticadr: wp.array[int],
    jnt_pos: wp.array2d[wp.vec3],
    jnt_qposadr: wp.array[int],
    jnt_range: wp.array2d[wp.vec2],
    jnt_solimp: wp.array2d[mjwp_types.vec5],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_springyield: wp.array2d[float],
    jnt_stiffness: wp.array2d[float],
    jnt_stiffnesspoly: wp.array2d[wp.vec2],
    jnt_type: wp.array[int],
    light_bodyid: wp.array[int],
    light_dir: wp.array2d[wp.vec3],
    light_dir0: wp.array2d[wp.vec3],
    light_mode: wp.array[int],
    light_pos: wp.array2d[wp.vec3],
    light_pos0: wp.array2d[wp.vec3],
    light_poscom0: wp.array2d[wp.vec3],
    light_targetbodyid: wp.array[int],
    mapM2D: wp.array[int],
    mat_rgba: wp.array2d[wp.vec4],
    max_ten_J_rownnz: int,
    mesh_face: wp.array[wp.vec3i],
    mesh_faceadr: wp.array[int],
    mesh_graph: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_normal: wp.array[wp.vec3],
    mesh_normaladr: wp.array[int],
    mesh_normalnum: wp.array[int],
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
    mesh_quat: wp.array[wp.quat],
    mesh_vert: wp.array[wp.vec3],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    nC: int,
    nD: int,
    nJten: int,
    nM: int,
    na: int,
    nacttrnbody: int,
    nactuator: int,
    nbody: int,
    nbranch: int,
    ncam: int,
    nefm0dof: int,
    nefmK: int,
    nefmdof: int,
    neq: int,
    nflex: int,
    nflexbend_interp: int,
    nflexedge: int,
    nflexelem: int,
    nflexface: int,
    nflexintcell: int,
    nflexnode: int,
    nflexrodedge: int,
    nflexrodhinge: int,
    nflexvert: int,
    ngeom: int,
    nhistory: int,
    njnt: int,
    nlight: int,
    nmaxcondim: int,
    nmaxpyramid: int,
    nmesh: int,
    nmeshdegmax: int,
    nmeshface: int,
    npair: int,
    nplastic: int,
    npolygonmax: int,
    nrangefinder: int,
    nsensorcollision: int,
    nsensortaxel: int,
    nsite: int,
    ntactileweld: int,
    ntendon: int,
    ntree: int,
    nu: int,
    nv: int,
    nv_pad: int,
    nwrap: int,
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
    qD_fullm_i: wp.array[int],
    qD_fullm_j: wp.array[int],
    qLD_all_updates: wp.array[wp.vec3i],
    qLD_block_adr: wp.array[int],
    qLD_block_total: int,
    qLD_level_offsets: wp.array[int],
    qLD_updates: tuple[wp.array[wp.vec3i], ...],
    qpos0: wp.array2d[float],
    qpos_spring: wp.array2d[float],
    rangefinder_sensor_adr: wp.array[int],
    sensor_acc_adr: wp.array[int],
    sensor_adr: wp.array[int],
    sensor_contact_acc_adr: wp.array[int],
    sensor_contact_pos_adr: wp.array[int],
    sensor_cutoff: wp.array[float],
    sensor_datatype: wp.array[int],
    sensor_delay: wp.array2d[float],
    sensor_dim: wp.array[int],
    sensor_e_kinetic: bool,
    sensor_e_potential: bool,
    sensor_history: wp.array2d[wp.vec2i],
    sensor_historyadr: wp.array2d[int],
    sensor_interval: wp.array2d[wp.vec2],
    sensor_intprm: wp.array2d[int],
    sensor_limitfrc_adr: wp.array[int],
    sensor_limitpos_adr: wp.array[int],
    sensor_limitvel_adr: wp.array[int],
    sensor_objid: wp.array[int],
    sensor_objtype: wp.array[int],
    sensor_pos_adr: wp.array[int],
    sensor_rangefinder_adr: wp.array[int],
    sensor_rangefinder_bodyid: wp.array[int],
    sensor_refid: wp.array[int],
    sensor_reftype: wp.array[int],
    sensor_rne_postconstraint: bool,
    sensor_stage_acc_adr: wp.array[int],
    sensor_stage_pos_adr: wp.array[int],
    sensor_stage_vel_adr: wp.array[int],
    sensor_subtree_vel: bool,
    sensor_tendonactfrc_adr: wp.array[int],
    sensor_touch_adr: wp.array[int],
    sensor_type: wp.array[int],
    sensor_vel_adr: wp.array[int],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    site_quat: wp.array2d[wp.quat],
    site_size: wp.array[wp.vec3],
    site_type: wp.array[int],
    taxel_sensorid: wp.array[int],
    taxel_vertadr: wp.array[int],
    ten_J_colind: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_rownnz: wp.array[int],
    tendon_actfrclimited: wp.array[bool],
    tendon_actfrcrange: wp.array2d[wp.vec2],
    tendon_actuatorid: wp.array[int],
    tendon_adr: wp.array[int],
    tendon_armature: wp.array2d[float],
    tendon_damping: wp.array2d[float],
    tendon_dampingpoly: wp.array2d[wp.vec2],
    tendon_frictionloss: wp.array2d[float],
    tendon_geom_adr: wp.array[int],
    tendon_invweight0: wp.array2d[float],
    tendon_jnt_adr: wp.array[int],
    tendon_length0: wp.array2d[float],
    tendon_lengthspring: wp.array2d[wp.vec2],
    tendon_limited: wp.array[int],
    tendon_limited_adr: wp.array[int],
    tendon_margin: wp.array2d[float],
    tendon_num: wp.array[int],
    tendon_plasticadr: wp.array[int],
    tendon_range: wp.array2d[wp.vec2],
    tendon_site_pair_adr: wp.array[int],
    tendon_solimp_fri: wp.array2d[mjwp_types.vec5],
    tendon_solimp_lim: wp.array2d[mjwp_types.vec5],
    tendon_solref_fri: wp.array2d[wp.vec2],
    tendon_solref_lim: wp.array2d[wp.vec2],
    tendon_springyield: wp.array2d[float],
    tendon_stiffness: wp.array2d[float],
    tendon_stiffnesspoly: wp.array2d[wp.vec2],
    tendon_type: wp.array[int],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    tree_sleep_policy: wp.array[int],
    weld_tactile_id: wp.array[int],
    wrap_geom_adr: wp.array[int],
    wrap_jnt_adr: wp.array[int],
    wrap_objid: wp.array[int],
    wrap_prm: wp.array[float],
    wrap_pulley_scale: wp.array[float],
    wrap_site_pair_adr: wp.array[int],
    wrap_type: wp.array[int],
    opt__broadphase: int,
    opt__broadphase_filter: int,
    opt__ccd_iterations: int,
    opt__ccd_tolerance: wp.array[float],
    opt__cone: int,
    opt__contact_sensor_maxmatch: int,
    opt__density: wp.array[float],
    opt__deterministic: int,
    opt__disableflags: int,
    opt__enableflags: int,
    opt__graph_conditional: bool,
    opt__gravity: wp.array[wp.vec3],
    opt__impratio_invsqrt: wp.array[float],
    opt__integrator: int,
    opt__iterations: int,
    opt__ls_iterations: int,
    opt__ls_tolerance: wp.array[float],
    opt__magnetic: wp.array[wp.vec3],
    opt__noslip_iterations: int,
    opt__noslip_tolerance: wp.array[float],
    opt__run_collision_detection: bool,
    opt__run_rne_postconstraint: bool,
    opt__sdf_initpoints: int,
    opt__sdf_iterations: int,
    opt__sleep_tolerance: wp.array[float],
    opt__solver: int,
    opt__timestep: wp.array[float],
    opt__tolerance: wp.array[float],
    opt__viscosity: wp.array[float],
    opt__warn_overflow: int,
    opt__wind: wp.array[wp.vec3],
    # Data
    naccdmax: int,
    naconmax: int,
    njmax: int,
    njmax_nnz: int,
    nvmax: int,
    nvmax_pad: int,
    M: wp.array2d[float],
    act: wp.array2d[float],
    act_dot: wp.array2d[float],
    actuator_force: wp.array2d[float],
    actuator_length: wp.array2d[float],
    actuator_moment: wp.array2d[float],
    actuator_velocity: wp.array2d[float],
    body_awake: wp.array2d[int],
    body_awake_ind: wp.array2d[int],
    cJ: wp.array3d[float],
    cM: wp.array3d[float],
    cMa: wp.array2d[float],
    cacc: wp.array2d[wp.spatial_vector],
    cam_xmat: wp.array2d[wp.mat33],
    cam_xpos: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    cdof_dof: wp.array2d[int],
    cdof_dot: wp.array2d[wp.spatial_vector],
    cdof_tri_col: wp.array[int],
    cdof_tri_row: wp.array[int],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    cfrc_int: wp.array2d[wp.spatial_vector],
    cinert: wp.array2d[mjwp_types.vec10],
    cqLD: wp.array3d[float],
    cqacc: wp.array2d[float],
    cqacc_smooth: wp.array2d[float],
    cqacc_warmstart: wp.array2d[float],
    cqfrc_constraint: wp.array2d[float],
    cqfrc_smooth: wp.array2d[float],
    crb: wp.array2d[mjwp_types.vec10],
    crhs: wp.array3d[float],
    ctrl: wp.array2d[float],
    cvel: wp.array2d[wp.spatial_vector],
    cx: wp.array3d[float],
    dof_awake_ind: wp.array2d[int],
    dof_cdof: wp.array2d[int],
    dof_island: wp.array2d[int],
    dof_islandid: wp.array2d[int],
    efc_islandid: wp.array2d[int],
    efm_K_val: wp.array2d[float],
    efm_L: wp.array2d[float],
    efm_as: wp.array2d[float],
    efm_c: wp.array2d[float],
    efm_ca: wp.array2d[float],
    efm_diag: wp.array2d[float],
    efm_fluid: wp.array2d[float],
    efm_ts: wp.array2d[float],
    energy: wp.array[wp.vec2],
    eq_active: wp.array2d[bool],
    face_quat: wp.array2d[wp.quat],
    face_xpos: wp.array3d[wp.vec3],
    flex_aabb_max: wp.array2d[wp.vec3],
    flex_aabb_min: wp.array2d[wp.vec3],
    flex_hessian_valid: wp.array2d[bool],
    flex_rod_chain_ok: wp.array2d[bool],
    flex_rod_chain_valid: wp.array2d[bool],
    flex_rod_end_torque: wp.array2d[wp.spatial_vector],
    flex_rod_frame0_loc: wp.array3d[wp.vec3],
    flex_rod_frame0_ref: wp.array3d[wp.vec3],
    flex_rod_ref_ok: wp.array2d[bool],
    flex_rod_tail_loc: wp.array3d[wp.vec3],
    flex_rodedge_frc: wp.array2d[wp.vec3],
    flex_rodedge_grad: wp.array2d[float],
    flex_rodedge_ktheta: wp.array2d[float],
    flex_rodedge_tridiag: wp.array2d[wp.vec2],
    flex_rodhinge_Wd: wp.array2d[wp.mat33],
    flex_rodhinge_Wk: wp.array2d[wp.mat33],
    flex_rodhinge_g: wp.array2d[wp.vec3],
    flex_rodhinge_om0: wp.array2d[wp.vec2],
    flexedge_J: wp.array2d[float],
    flexedge_hessian: wp.array2d[wp.mat33],
    flexedge_length: wp.array2d[float],
    flexedge_velocity: wp.array2d[float],
    flexnode_xpos: wp.array2d[wp.vec3],
    flexvert_hessian: wp.array2d[mjwp_types.vec6],
    flexvert_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    geom_xpos: wp.array2d[wp.vec3],
    history: wp.array2d[float],
    island_dofadr: wp.array2d[int],
    island_idofadr: wp.array2d[int],
    island_iefcadr: wp.array2d[int],
    island_ne: wp.array2d[int],
    island_nefc: wp.array2d[int],
    island_nf: wp.array2d[int],
    island_nv: wp.array2d[int],
    light_xdir: wp.array2d[wp.vec3],
    light_xpos: wp.array2d[wp.vec3],
    map_dof2idof: wp.array2d[int],
    map_efc2iefc: wp.array2d[int],
    map_idof2dof: wp.array2d[int],
    map_iefc2efc: wp.array2d[int],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    moment_colind: wp.array2d[int],
    moment_rowadr: wp.array2d[int],
    moment_rownnz: wp.array2d[int],
    nacon: wp.array[int],
    nbody_awake: wp.array[int],
    ncdof: wp.array[int],
    ncollision: wp.array[int],
    ne: wp.array[int],
    nefc: wp.array[int],
    nf: wp.array[int],
    nidof: wp.array[int],
    nisland: wp.array[int],
    nl: wp.array[int],
    ntree_awake: wp.array[int],
    nv_awake: wp.array[int],
    overflow: wp.array[int],
    plastic: wp.array2d[float],
    qH: wp.array2d[float],
    qHDiagInv: wp.array2d[float],
    qHLD: wp.array2d[float],
    qLD: wp.array2d[float],
    qLDiagInv: wp.array2d[float],
    qLU: wp.array2d[float],
    qacc: wp.array2d[float],
    qacc_smooth: wp.array2d[float],
    qacc_warmstart: wp.array2d[float],
    qfrc_actuator: wp.array2d[float],
    qfrc_adhesion: wp.array2d[float],
    qfrc_applied: wp.array2d[float],
    qfrc_bias: wp.array2d[float],
    qfrc_constraint: wp.array2d[float],
    qfrc_damper: wp.array2d[float],
    qfrc_fluid: wp.array2d[float],
    qfrc_gravcomp: wp.array2d[float],
    qfrc_passive: wp.array2d[float],
    qfrc_smooth: wp.array2d[float],
    qfrc_spring: wp.array2d[float],
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    sensordata: wp.array2d[float],
    site_xmat: wp.array2d[wp.mat33],
    site_xpos: wp.array2d[wp.vec3],
    solver_niter: wp.array[int],
    subtree_angmom: wp.array2d[wp.vec3],
    subtree_com: wp.array2d[wp.vec3],
    subtree_linvel: wp.array2d[wp.vec3],
    ten_J: wp.array2d[float],
    ten_length: wp.array2d[float],
    ten_velocity: wp.array2d[float],
    ten_wrapadr: wp.array2d[int],
    ten_wrapnum: wp.array2d[int],
    time: wp.array[float],
    tree_asleep: wp.array2d[int],
    tree_awake: wp.array2d[int],
    tree_island: wp.array2d[int],
    wrap_obj: wp.array2d[wp.vec2i],
    wrap_xpos: wp.array2d[wp.spatial_vector],
    xanchor: wp.array2d[wp.vec3],
    xaxis: wp.array2d[wp.vec3],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    ximat: wp.array2d[wp.mat33],
    xipos: wp.array2d[wp.vec3],
    xmat: wp.array2d[wp.mat33],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
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
    efc__D: wp.array2d[float],
    efc__J: wp.array3d[float],
    efc__J_colind: wp.array3d[int],
    efc__J_rowadr: wp.array2d[int],
    efc__J_rownnz: wp.array2d[int],
    efc__Jqvel: wp.array2d[float],
    efc__Ma: wp.array2d[float],
    efc__aref: wp.array2d[float],
    efc__force: wp.array2d[float],
    efc__frictionloss: wp.array2d[float],
    efc__id: wp.array2d[int],
    efc__island: wp.array2d[int],
    efc__jtdaj_adr: wp.array2d[int],
    efc__jtdaj_nblock: wp.array[int],
    efc__jtdaj_nrow: wp.array2d[int],
    efc__margin: wp.array2d[float],
    efc__pos: wp.array2d[float],
    efc__state: wp.array2d[int],
    efc__type: wp.array2d[int],
    efc__vel: wp.array2d[float],
):
  _m.opt = _o
  _m.callback = _cb
  _d.efc = _e
  _d.contact = _c
  _m.D_colind = D_colind
  _m.D_diag = D_diag
  _m.D_rowadr = D_rowadr
  _m.D_rownnz = D_rownnz
  _m.M_colind = M_colind
  _m.M_elemid = M_elemid
  _m.M_fullm_i = M_fullm_i
  _m.M_fullm_j = M_fullm_j
  _m.M_hinit_i = M_hinit_i
  _m.M_mulm_col = M_mulm_col
  _m.M_mulm_madr = M_mulm_madr
  _m.M_mulm_rowadr = M_mulm_rowadr
  _m.M_rowadr = M_rowadr
  _m.M_rownnz = M_rownnz
  _m.M_tiles = M_tiles
  _m.actuator_acc0 = actuator_acc0
  _m.actuator_actadr = actuator_actadr
  _m.actuator_actearly = actuator_actearly
  _m.actuator_actlimited = actuator_actlimited
  _m.actuator_actnum = actuator_actnum
  _m.actuator_actrange = actuator_actrange
  _m.actuator_armature = actuator_armature
  _m.actuator_biasprm = actuator_biasprm
  _m.actuator_biastype = actuator_biastype
  _m.actuator_cranklength = actuator_cranklength
  _m.actuator_ctrladr = actuator_ctrladr
  _m.actuator_ctrllimited = actuator_ctrllimited
  _m.actuator_ctrlnum = actuator_ctrlnum
  _m.actuator_ctrlrange = actuator_ctrlrange
  _m.actuator_ctrlspec = actuator_ctrlspec
  _m.actuator_damping = actuator_damping
  _m.actuator_dampingpoly = actuator_dampingpoly
  _m.actuator_delay = actuator_delay
  _m.actuator_dynprm = actuator_dynprm
  _m.actuator_dyntype = actuator_dyntype
  _m.actuator_forcelimited = actuator_forcelimited
  _m.actuator_forcerange = actuator_forcerange
  _m.actuator_gainprm = actuator_gainprm
  _m.actuator_gaintype = actuator_gaintype
  _m.actuator_gear = actuator_gear
  _m.actuator_history = actuator_history
  _m.actuator_historyadr = actuator_historyadr
  _m.actuator_lengthrange = actuator_lengthrange
  _m.actuator_trnid = actuator_trnid
  _m.actuator_trntype = actuator_trntype
  _m.actuator_trntype_body_adr = actuator_trntype_body_adr
  _m.block_dim = block_dim
  _m.body_branch_start = body_branch_start
  _m.body_branches = body_branches
  _m.body_dofadr = body_dofadr
  _m.body_dofnum = body_dofnum
  _m.body_fluid_box_adr = body_fluid_box_adr
  _m.body_fluid_ellipsoid = body_fluid_ellipsoid
  _m.body_fluid_ellipsoid_adr = body_fluid_ellipsoid_adr
  _m.body_freeadr = body_freeadr
  _m.body_geomadr = body_geomadr
  _m.body_geomnum = body_geomnum
  _m.body_gravcomp = body_gravcomp
  _m.body_inertia = body_inertia
  _m.body_invweight0 = body_invweight0
  _m.body_ipos = body_ipos
  _m.body_iquat = body_iquat
  _m.body_is_free = body_is_free
  _m.body_isdofancestor = body_isdofancestor
  _m.body_jntadr = body_jntadr
  _m.body_jntnum = body_jntnum
  _m.body_mass = body_mass
  _m.body_mocapid = body_mocapid
  _m.body_parentid = body_parentid
  _m.body_pos = body_pos
  _m.body_quat = body_quat
  _m.body_rootid = body_rootid
  _m.body_simple = body_simple
  _m.body_subtreemass = body_subtreemass
  _m.body_subtreenum = body_subtreenum
  _m.body_tree = body_tree
  _m.body_treeid = body_treeid
  _m.body_weldid = body_weldid
  _m.cam_bodyid = cam_bodyid
  _m.cam_fovy = cam_fovy
  _m.cam_intrinsic = cam_intrinsic
  _m.cam_mat0 = cam_mat0
  _m.cam_mode = cam_mode
  _m.cam_pos = cam_pos
  _m.cam_pos0 = cam_pos0
  _m.cam_poscom0 = cam_poscom0
  _m.cam_projection = cam_projection
  _m.cam_quat = cam_quat
  _m.cam_resolution = cam_resolution
  _m.cam_sensorsize = cam_sensorsize
  _m.cam_targetbodyid = cam_targetbodyid
  _m.dof_armature = dof_armature
  _m.dof_bodyid = dof_bodyid
  _m.dof_damping = dof_damping
  _m.dof_dampingpoly = dof_dampingpoly
  _m.dof_frictionloss = dof_frictionloss
  _m.dof_invweight0 = dof_invweight0
  _m.dof_jntid = dof_jntid
  _m.dof_length = dof_length
  _m.dof_parentid = dof_parentid
  _m.dof_solimp = dof_solimp
  _m.dof_solref = dof_solref
  _m.dof_treeid = dof_treeid
  _m.dof_tri_col = dof_tri_col
  _m.dof_tri_row = dof_tri_row
  _m.efm0_L = efm0_L
  _m.efm0_L_colind = efm0_L_colind
  _m.efm0_L_rowadr = efm0_L_rowadr
  _m.efm0_L_rownnz = efm0_L_rownnz
  _m.efm0_active = efm0_active
  _m.efm0_dofid = efm0_dofid
  _m.efm_K_colind = efm_K_colind
  _m.efm_K_rowadr = efm_K_rowadr
  _m.efm_K_rownnz = efm_K_rownnz
  _m.efm_con_max_dof = efm_con_max_dof
  _m.efm_dofblk = efm_dofblk
  _m.efm_dofid = efm_dofid
  _m.eq_connect_adr = eq_connect_adr
  _m.eq_data = eq_data
  _m.eq_flex_adr = eq_flex_adr
  _m.eq_flexstrain_adr = eq_flexstrain_adr
  _m.eq_jnt_adr = eq_jnt_adr
  _m.eq_obj1id = eq_obj1id
  _m.eq_obj2id = eq_obj2id
  _m.eq_objtype = eq_objtype
  _m.eq_solimp = eq_solimp
  _m.eq_solref = eq_solref
  _m.eq_ten_adr = eq_ten_adr
  _m.eq_type = eq_type
  _m.eq_wld_adr = eq_wld_adr
  _m.flex_activelayers = flex_activelayers
  _m.flex_bend_interp_map = flex_bend_interp_map
  _m.flex_bending = flex_bending
  _m.flex_bendingadr = flex_bendingadr
  _m.flex_cell_map = flex_cell_map
  _m.flex_cellnum = flex_cellnum
  _m.flex_centered = flex_centered
  _m.flex_conaffinity = flex_conaffinity
  _m.flex_condim = flex_condim
  _m.flex_contype = flex_contype
  _m.flex_damping = flex_damping
  _m.flex_dim = flex_dim
  _m.flex_edge = flex_edge
  _m.flex_edgeadr = flex_edgeadr
  _m.flex_edgedamping = flex_edgedamping
  _m.flex_edgeequality = flex_edgeequality
  _m.flex_edgeflap = flex_edgeflap
  _m.flex_edgeflexid = flex_edgeflexid
  _m.flex_edgenum = flex_edgenum
  _m.flex_edgestiffness = flex_edgestiffness
  _m.flex_eff_possible = flex_eff_possible
  _m.flex_elem = flex_elem
  _m.flex_elem_rodedge = flex_elem_rodedge
  _m.flex_elemadr = flex_elemadr
  _m.flex_elemdataadr = flex_elemdataadr
  _m.flex_elemedge = flex_elemedge
  _m.flex_elemedgeadr = flex_elemedgeadr
  _m.flex_elemflexid = flex_elemflexid
  _m.flex_elemlayer = flex_elemlayer
  _m.flex_elemnum = flex_elemnum
  _m.flex_face = flex_face
  _m.flex_face_map = flex_face_map
  _m.flex_faceadr = flex_faceadr
  _m.flex_friction = flex_friction
  _m.flex_gap = flex_gap
  _m.flex_interp = flex_interp
  _m.flex_interp_assemblable = flex_interp_assemblable
  _m.flex_margin = flex_margin
  _m.flex_node = flex_node
  _m.flex_node0 = flex_node0
  _m.flex_nodeadr = flex_nodeadr
  _m.flex_nodebodyid = flex_nodebodyid
  _m.flex_nodenum = flex_nodenum
  _m.flex_passive = flex_passive
  _m.flex_plasticadr = flex_plasticadr
  _m.flex_plasticmode = flex_plasticmode
  _m.flex_priority = flex_priority
  _m.flex_radius = flex_radius
  _m.flex_rigid = flex_rigid
  _m.flex_rod_box_size = flex_rod_box_size
  _m.flex_rod_edgeadr = flex_rod_edgeadr
  _m.flex_rod_edgenum = flex_rod_edgenum
  _m.flex_rod_has_twist = flex_rod_has_twist
  _m.flex_rod_hingeadr = flex_rod_hingeadr
  _m.flex_rod_is_box = flex_rod_is_box
  _m.flex_rod_is_chain = flex_rod_is_chain
  _m.flex_rod_root_clamped = flex_rod_root_clamped
  _m.flex_rod_tail_clamped = flex_rod_tail_clamped
  _m.flex_rodedge_jnt = flex_rodedge_jnt
  _m.flex_rodedge_vert = flex_rodedge_vert
  _m.flex_rodhinge_map = flex_rodhinge_map
  _m.flex_selfcollide = flex_selfcollide
  _m.flex_simple = flex_simple
  _m.flex_size = flex_size
  _m.flex_solimp = flex_solimp
  _m.flex_solmix = flex_solmix
  _m.flex_solref = flex_solref
  _m.flex_stiffness = flex_stiffness
  _m.flex_stiffnessadr = flex_stiffnessadr
  _m.flex_vert = flex_vert
  _m.flex_vert0 = flex_vert0
  _m.flex_vert_rodelem = flex_vert_rodelem
  _m.flex_vertadr = flex_vertadr
  _m.flex_vertbodyid = flex_vertbodyid
  _m.flex_vertcell = flex_vertcell
  _m.flex_vertflexid = flex_vertflexid
  _m.flex_vertnum = flex_vertnum
  _m.flex_yield = flex_yield
  _m.flexedge_J_colind = flexedge_J_colind
  _m.flexedge_J_rowadr = flexedge_J_rowadr
  _m.flexedge_J_rownnz = flexedge_J_rownnz
  _m.flexedge_invweight0 = flexedge_invweight0
  _m.flexedge_length0 = flexedge_length0
  _m.flexedge_rigid = flexedge_rigid
  _m.flexstrain_J_colind = flexstrain_J_colind
  _m.flexstrain_J_rowadr = flexstrain_J_rowadr
  _m.flexstrain_J_rownnz = flexstrain_J_rownnz
  _m.flg_adhesion = flg_adhesion
  _m.flg_surfacevel = flg_surfacevel
  _m.geom_aabb = geom_aabb
  _m.geom_adhesion = geom_adhesion
  _m.geom_bodyid = geom_bodyid
  _m.geom_conaffinity = geom_conaffinity
  _m.geom_condim = geom_condim
  _m.geom_contype = geom_contype
  _m.geom_dataid = geom_dataid
  _m.geom_fluid = geom_fluid
  _m.geom_friction = geom_friction
  _m.geom_gap = geom_gap
  _m.geom_group = geom_group
  _m.geom_margin = geom_margin
  _m.geom_matid = geom_matid
  _m.geom_pair_type_count = geom_pair_type_count
  _m.geom_plugin_index = geom_plugin_index
  _m.geom_pos = geom_pos
  _m.geom_priority = geom_priority
  _m.geom_quat = geom_quat
  _m.geom_rbound = geom_rbound
  _m.geom_rgba = geom_rgba
  _m.geom_size = geom_size
  _m.geom_solimp = geom_solimp
  _m.geom_solmix = geom_solmix
  _m.geom_solref = geom_solref
  _m.geom_surfacevel = geom_surfacevel
  _m.geom_type = geom_type
  _m.has_1d_flex = has_1d_flex
  _m.has_3d_flex = has_3d_flex
  _m.has_actuator_dcmotor = has_actuator_dcmotor
  _m.has_actuator_wrap = has_actuator_wrap
  _m.has_efm_actuator = has_efm_actuator
  _m.has_ellipsoid_geom = has_ellipsoid_geom
  _m.has_flex_edge_stiff_damp = has_flex_edge_stiff_damp
  _m.has_flex_passive = has_flex_passive
  _m.has_flex_selfcollide = has_flex_selfcollide
  _m.has_flex_snh = has_flex_snh
  _m.has_fluid = has_fluid
  _m.has_non_simple_flex = has_non_simple_flex
  _m.has_plane_geom = has_plane_geom
  _m.has_sdf_geom = has_sdf_geom
  _m.has_tendon_damping = has_tendon_damping
  _m.has_tendon_stiffness = has_tendon_stiffness
  _m.has_unsupported_flex_interp = has_unsupported_flex_interp
  _m.hfield_adr = hfield_adr
  _m.hfield_data = hfield_data
  _m.hfield_ncol = hfield_ncol
  _m.hfield_nrow = hfield_nrow
  _m.hfield_size = hfield_size
  _m.is_sparse = is_sparse
  _m.jnt_actfrclimited = jnt_actfrclimited
  _m.jnt_actfrcrange = jnt_actfrcrange
  _m.jnt_actgravcomp = jnt_actgravcomp
  _m.jnt_actuatorid = jnt_actuatorid
  _m.jnt_axis = jnt_axis
  _m.jnt_bodyid = jnt_bodyid
  _m.jnt_dofadr = jnt_dofadr
  _m.jnt_limited_ball_adr = jnt_limited_ball_adr
  _m.jnt_limited_slide_hinge_adr = jnt_limited_slide_hinge_adr
  _m.jnt_margin = jnt_margin
  _m.jnt_plasticadr = jnt_plasticadr
  _m.jnt_pos = jnt_pos
  _m.jnt_qposadr = jnt_qposadr
  _m.jnt_range = jnt_range
  _m.jnt_solimp = jnt_solimp
  _m.jnt_solref = jnt_solref
  _m.jnt_springyield = jnt_springyield
  _m.jnt_stiffness = jnt_stiffness
  _m.jnt_stiffnesspoly = jnt_stiffnesspoly
  _m.jnt_type = jnt_type
  _m.light_bodyid = light_bodyid
  _m.light_dir = light_dir
  _m.light_dir0 = light_dir0
  _m.light_mode = light_mode
  _m.light_pos = light_pos
  _m.light_pos0 = light_pos0
  _m.light_poscom0 = light_poscom0
  _m.light_targetbodyid = light_targetbodyid
  _m.mapM2D = mapM2D
  _m.mat_rgba = mat_rgba
  _m.max_ten_J_rownnz = max_ten_J_rownnz
  _m.mesh_face = mesh_face
  _m.mesh_faceadr = mesh_faceadr
  _m.mesh_graph = mesh_graph
  _m.mesh_graphadr = mesh_graphadr
  _m.mesh_normal = mesh_normal
  _m.mesh_normaladr = mesh_normaladr
  _m.mesh_normalnum = mesh_normalnum
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
  _m.mesh_quat = mesh_quat
  _m.mesh_vert = mesh_vert
  _m.mesh_vertadr = mesh_vertadr
  _m.mesh_vertnum = mesh_vertnum
  _m.nC = nC
  _m.nD = nD
  _m.nJten = nJten
  _m.nM = nM
  _m.na = na
  _m.nacttrnbody = nacttrnbody
  _m.nactuator = nactuator
  _m.nbody = nbody
  _m.nbranch = nbranch
  _m.ncam = ncam
  _m.nefm0dof = nefm0dof
  _m.nefmK = nefmK
  _m.nefmdof = nefmdof
  _m.neq = neq
  _m.nflex = nflex
  _m.nflexbend_interp = nflexbend_interp
  _m.nflexedge = nflexedge
  _m.nflexelem = nflexelem
  _m.nflexface = nflexface
  _m.nflexintcell = nflexintcell
  _m.nflexnode = nflexnode
  _m.nflexrodedge = nflexrodedge
  _m.nflexrodhinge = nflexrodhinge
  _m.nflexvert = nflexvert
  _m.ngeom = ngeom
  _m.nhistory = nhistory
  _m.njnt = njnt
  _m.nlight = nlight
  _m.nmaxcondim = nmaxcondim
  _m.nmaxpyramid = nmaxpyramid
  _m.nmesh = nmesh
  _m.nmeshdegmax = nmeshdegmax
  _m.nmeshface = nmeshface
  _m.npair = npair
  _m.nplastic = nplastic
  _m.npolygonmax = npolygonmax
  _m.nrangefinder = nrangefinder
  _m.nsensorcollision = nsensorcollision
  _m.nsensortaxel = nsensortaxel
  _m.nsite = nsite
  _m.ntactileweld = ntactileweld
  _m.ntendon = ntendon
  _m.ntree = ntree
  _m.nu = nu
  _m.nv = nv
  _m.nv_pad = nv_pad
  _m.nwrap = nwrap
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
  _m.opt.cone = opt__cone
  _m.opt.contact_sensor_maxmatch = opt__contact_sensor_maxmatch
  _m.opt.density = opt__density
  _m.opt.deterministic = opt__deterministic
  _m.opt.disableflags = opt__disableflags
  _m.opt.enableflags = opt__enableflags
  _m.opt.graph_conditional = opt__graph_conditional
  _m.opt.gravity = opt__gravity
  _m.opt.impratio_invsqrt = opt__impratio_invsqrt
  _m.opt.integrator = opt__integrator
  _m.opt.iterations = opt__iterations
  _m.opt.ls_iterations = opt__ls_iterations
  _m.opt.ls_tolerance = opt__ls_tolerance
  _m.opt.magnetic = opt__magnetic
  _m.opt.noslip_iterations = opt__noslip_iterations
  _m.opt.noslip_tolerance = opt__noslip_tolerance
  _m.opt.run_collision_detection = opt__run_collision_detection
  _m.opt.run_rne_postconstraint = opt__run_rne_postconstraint
  _m.opt.sdf_initpoints = opt__sdf_initpoints
  _m.opt.sdf_iterations = opt__sdf_iterations
  _m.opt.sleep_tolerance = opt__sleep_tolerance
  _m.opt.solver = opt__solver
  _m.opt.timestep = opt__timestep
  _m.opt.tolerance = opt__tolerance
  _m.opt.viscosity = opt__viscosity
  _m.opt.warn_overflow = opt__warn_overflow
  _m.opt.wind = opt__wind
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
  _m.qD_fullm_i = qD_fullm_i
  _m.qD_fullm_j = qD_fullm_j
  _m.qLD_all_updates = qLD_all_updates
  _m.qLD_block_adr = qLD_block_adr
  _m.qLD_block_total = qLD_block_total
  _m.qLD_level_offsets = qLD_level_offsets
  _m.qLD_updates = qLD_updates
  _m.qpos0 = qpos0
  _m.qpos_spring = qpos_spring
  _m.rangefinder_sensor_adr = rangefinder_sensor_adr
  _m.sensor_acc_adr = sensor_acc_adr
  _m.sensor_adr = sensor_adr
  _m.sensor_contact_acc_adr = sensor_contact_acc_adr
  _m.sensor_contact_pos_adr = sensor_contact_pos_adr
  _m.sensor_cutoff = sensor_cutoff
  _m.sensor_datatype = sensor_datatype
  _m.sensor_delay = sensor_delay
  _m.sensor_dim = sensor_dim
  _m.sensor_e_kinetic = sensor_e_kinetic
  _m.sensor_e_potential = sensor_e_potential
  _m.sensor_history = sensor_history
  _m.sensor_historyadr = sensor_historyadr
  _m.sensor_interval = sensor_interval
  _m.sensor_intprm = sensor_intprm
  _m.sensor_limitfrc_adr = sensor_limitfrc_adr
  _m.sensor_limitpos_adr = sensor_limitpos_adr
  _m.sensor_limitvel_adr = sensor_limitvel_adr
  _m.sensor_objid = sensor_objid
  _m.sensor_objtype = sensor_objtype
  _m.sensor_pos_adr = sensor_pos_adr
  _m.sensor_rangefinder_adr = sensor_rangefinder_adr
  _m.sensor_rangefinder_bodyid = sensor_rangefinder_bodyid
  _m.sensor_refid = sensor_refid
  _m.sensor_reftype = sensor_reftype
  _m.sensor_rne_postconstraint = sensor_rne_postconstraint
  _m.sensor_stage_acc_adr = sensor_stage_acc_adr
  _m.sensor_stage_pos_adr = sensor_stage_pos_adr
  _m.sensor_stage_vel_adr = sensor_stage_vel_adr
  _m.sensor_subtree_vel = sensor_subtree_vel
  _m.sensor_tendonactfrc_adr = sensor_tendonactfrc_adr
  _m.sensor_touch_adr = sensor_touch_adr
  _m.sensor_type = sensor_type
  _m.sensor_vel_adr = sensor_vel_adr
  _m.site_bodyid = site_bodyid
  _m.site_pos = site_pos
  _m.site_quat = site_quat
  _m.site_size = site_size
  _m.site_type = site_type
  _m.taxel_sensorid = taxel_sensorid
  _m.taxel_vertadr = taxel_vertadr
  _m.ten_J_colind = ten_J_colind
  _m.ten_J_rowadr = ten_J_rowadr
  _m.ten_J_rownnz = ten_J_rownnz
  _m.tendon_actfrclimited = tendon_actfrclimited
  _m.tendon_actfrcrange = tendon_actfrcrange
  _m.tendon_actuatorid = tendon_actuatorid
  _m.tendon_adr = tendon_adr
  _m.tendon_armature = tendon_armature
  _m.tendon_damping = tendon_damping
  _m.tendon_dampingpoly = tendon_dampingpoly
  _m.tendon_frictionloss = tendon_frictionloss
  _m.tendon_geom_adr = tendon_geom_adr
  _m.tendon_invweight0 = tendon_invweight0
  _m.tendon_jnt_adr = tendon_jnt_adr
  _m.tendon_length0 = tendon_length0
  _m.tendon_lengthspring = tendon_lengthspring
  _m.tendon_limited = tendon_limited
  _m.tendon_limited_adr = tendon_limited_adr
  _m.tendon_margin = tendon_margin
  _m.tendon_num = tendon_num
  _m.tendon_plasticadr = tendon_plasticadr
  _m.tendon_range = tendon_range
  _m.tendon_site_pair_adr = tendon_site_pair_adr
  _m.tendon_solimp_fri = tendon_solimp_fri
  _m.tendon_solimp_lim = tendon_solimp_lim
  _m.tendon_solref_fri = tendon_solref_fri
  _m.tendon_solref_lim = tendon_solref_lim
  _m.tendon_springyield = tendon_springyield
  _m.tendon_stiffness = tendon_stiffness
  _m.tendon_stiffnesspoly = tendon_stiffnesspoly
  _m.tendon_type = tendon_type
  _m.tree_dofadr = tree_dofadr
  _m.tree_dofnum = tree_dofnum
  _m.tree_sleep_policy = tree_sleep_policy
  _m.weld_tactile_id = weld_tactile_id
  _m.wrap_geom_adr = wrap_geom_adr
  _m.wrap_jnt_adr = wrap_jnt_adr
  _m.wrap_objid = wrap_objid
  _m.wrap_prm = wrap_prm
  _m.wrap_pulley_scale = wrap_pulley_scale
  _m.wrap_site_pair_adr = wrap_site_pair_adr
  _m.wrap_type = wrap_type
  _d.M = M
  _d.act = act
  _d.act_dot = act_dot
  _d.actuator_force = actuator_force
  _d.actuator_length = actuator_length
  _d.actuator_moment = actuator_moment
  _d.actuator_velocity = actuator_velocity
  _d.body_awake = body_awake
  _d.body_awake_ind = body_awake_ind
  _d.cJ = cJ
  _d.cM = cM
  _d.cMa = cMa
  _d.cacc = cacc
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.cdof = cdof
  _d.cdof_dof = cdof_dof
  _d.cdof_dot = cdof_dot
  _d.cdof_tri_col = cdof_tri_col
  _d.cdof_tri_row = cdof_tri_row
  _d.cfrc_ext = cfrc_ext
  _d.cfrc_int = cfrc_int
  _d.cinert = cinert
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
  _d.cqLD = cqLD
  _d.cqacc = cqacc
  _d.cqacc_smooth = cqacc_smooth
  _d.cqacc_warmstart = cqacc_warmstart
  _d.cqfrc_constraint = cqfrc_constraint
  _d.cqfrc_smooth = cqfrc_smooth
  _d.crb = crb
  _d.crhs = crhs
  _d.ctrl = ctrl
  _d.cvel = cvel
  _d.cx = cx
  _d.dof_awake_ind = dof_awake_ind
  _d.dof_cdof = dof_cdof
  _d.dof_island = dof_island
  _d.dof_islandid = dof_islandid
  _d.efc.D = efc__D
  _d.efc.J = efc__J
  _d.efc.J_colind = efc__J_colind
  _d.efc.J_rowadr = efc__J_rowadr
  _d.efc.J_rownnz = efc__J_rownnz
  _d.efc.Jqvel = efc__Jqvel
  _d.efc.Ma = efc__Ma
  _d.efc.aref = efc__aref
  _d.efc.force = efc__force
  _d.efc.frictionloss = efc__frictionloss
  _d.efc.id = efc__id
  _d.efc.island = efc__island
  _d.efc.jtdaj_adr = efc__jtdaj_adr
  _d.efc.jtdaj_nblock = efc__jtdaj_nblock
  _d.efc.jtdaj_nrow = efc__jtdaj_nrow
  _d.efc.margin = efc__margin
  _d.efc.pos = efc__pos
  _d.efc.state = efc__state
  _d.efc.type = efc__type
  _d.efc.vel = efc__vel
  _d.efc_islandid = efc_islandid
  _d.efm_K_val = efm_K_val
  _d.efm_L = efm_L
  _d.efm_as = efm_as
  _d.efm_c = efm_c
  _d.efm_ca = efm_ca
  _d.efm_diag = efm_diag
  _d.efm_fluid = efm_fluid
  _d.efm_ts = efm_ts
  _d.energy = energy
  _d.eq_active = eq_active
  _d.face_quat = face_quat
  _d.face_xpos = face_xpos
  _d.flex_aabb_max = flex_aabb_max
  _d.flex_aabb_min = flex_aabb_min
  _d.flex_hessian_valid = flex_hessian_valid
  _d.flex_rod_chain_ok = flex_rod_chain_ok
  _d.flex_rod_chain_valid = flex_rod_chain_valid
  _d.flex_rod_end_torque = flex_rod_end_torque
  _d.flex_rod_frame0_loc = flex_rod_frame0_loc
  _d.flex_rod_frame0_ref = flex_rod_frame0_ref
  _d.flex_rod_ref_ok = flex_rod_ref_ok
  _d.flex_rod_tail_loc = flex_rod_tail_loc
  _d.flex_rodedge_frc = flex_rodedge_frc
  _d.flex_rodedge_grad = flex_rodedge_grad
  _d.flex_rodedge_ktheta = flex_rodedge_ktheta
  _d.flex_rodedge_tridiag = flex_rodedge_tridiag
  _d.flex_rodhinge_Wd = flex_rodhinge_Wd
  _d.flex_rodhinge_Wk = flex_rodhinge_Wk
  _d.flex_rodhinge_g = flex_rodhinge_g
  _d.flex_rodhinge_om0 = flex_rodhinge_om0
  _d.flexedge_J = flexedge_J
  _d.flexedge_hessian = flexedge_hessian
  _d.flexedge_length = flexedge_length
  _d.flexedge_velocity = flexedge_velocity
  _d.flexnode_xpos = flexnode_xpos
  _d.flexvert_hessian = flexvert_hessian
  _d.flexvert_xpos = flexvert_xpos
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.history = history
  _d.island_dofadr = island_dofadr
  _d.island_idofadr = island_idofadr
  _d.island_iefcadr = island_iefcadr
  _d.island_ne = island_ne
  _d.island_nefc = island_nefc
  _d.island_nf = island_nf
  _d.island_nv = island_nv
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  _d.map_dof2idof = map_dof2idof
  _d.map_efc2iefc = map_efc2iefc
  _d.map_idof2dof = map_idof2dof
  _d.map_iefc2efc = map_iefc2efc
  _d.mocap_pos = mocap_pos
  _d.mocap_quat = mocap_quat
  _d.moment_colind = moment_colind
  _d.moment_rowadr = moment_rowadr
  _d.moment_rownnz = moment_rownnz
  _d.naccdmax = naccdmax
  _d.nacon = nacon
  _d.naconmax = naconmax
  _d.nbody_awake = nbody_awake
  _d.ncdof = ncdof
  _d.ncollision = ncollision
  _d.ne = ne
  _d.nefc = nefc
  _d.nf = nf
  _d.nidof = nidof
  _d.nisland = nisland
  _d.njmax = njmax
  _d.njmax_nnz = njmax_nnz
  _d.nl = nl
  _d.ntree_awake = ntree_awake
  _d.nv_awake = nv_awake
  _d.nvmax = nvmax
  _d.nvmax_pad = nvmax_pad
  _d.overflow = overflow
  _d.plastic = plastic
  _d.qH = qH
  _d.qHDiagInv = qHDiagInv
  _d.qHLD = qHLD
  _d.qLD = qLD
  _d.qLDiagInv = qLDiagInv
  _d.qLU = qLU
  _d.qacc = qacc
  _d.qacc_smooth = qacc_smooth
  _d.qacc_warmstart = qacc_warmstart
  _d.qfrc_actuator = qfrc_actuator
  _d.qfrc_adhesion = qfrc_adhesion
  _d.qfrc_applied = qfrc_applied
  _d.qfrc_bias = qfrc_bias
  _d.qfrc_constraint = qfrc_constraint
  _d.qfrc_damper = qfrc_damper
  _d.qfrc_fluid = qfrc_fluid
  _d.qfrc_gravcomp = qfrc_gravcomp
  _d.qfrc_passive = qfrc_passive
  _d.qfrc_smooth = qfrc_smooth
  _d.qfrc_spring = qfrc_spring
  _d.qpos = qpos
  _d.qvel = qvel
  _d.sensordata = sensordata
  _d.site_xmat = site_xmat
  _d.site_xpos = site_xpos
  _d.solver_niter = solver_niter
  _d.subtree_angmom = subtree_angmom
  _d.subtree_com = subtree_com
  _d.subtree_linvel = subtree_linvel
  _d.ten_J = ten_J
  _d.ten_length = ten_length
  _d.ten_velocity = ten_velocity
  _d.ten_wrapadr = ten_wrapadr
  _d.ten_wrapnum = ten_wrapnum
  _d.time = time
  _d.tree_asleep = tree_asleep
  _d.tree_awake = tree_awake
  _d.tree_island = tree_island
  _d.wrap_obj = wrap_obj
  _d.wrap_xpos = wrap_xpos
  _d.xanchor = xanchor
  _d.xaxis = xaxis
  _d.xfrc_applied = xfrc_applied
  _d.ximat = ximat
  _d.xipos = xipos
  _d.xmat = xmat
  _d.xpos = xpos
  _d.xquat = xquat
  _d.nworld = nworld
  mjwarp.step(_m, _d)


def _step_jax_impl(m: types.Model, d: types.Data):
  output_dims = {
      'M': d._impl.M.shape,
      'act': d.act.shape,
      'act_dot': d.act_dot.shape,
      'actuator_force': d.actuator_force.shape,
      'actuator_length': d.actuator_length.shape,
      'actuator_moment': d._impl.actuator_moment.shape,
      'actuator_velocity': d._impl.actuator_velocity.shape,
      'body_awake': d._impl.body_awake.shape,
      'body_awake_ind': d._impl.body_awake_ind.shape,
      'cJ': d._impl.cJ.shape,
      'cM': d._impl.cM.shape,
      'cacc': d._impl.cacc.shape,
      'cam_xmat': d.cam_xmat.shape,
      'cam_xpos': d.cam_xpos.shape,
      'cdof': d.cdof.shape,
      'cdof_dof': d._impl.cdof_dof.shape,
      'cdof_dot': d.cdof_dot.shape,
      'cfrc_ext': d._impl.cfrc_ext.shape,
      'cfrc_int': d._impl.cfrc_int.shape,
      'cinert': d._impl.cinert.shape,
      'cqLD': d._impl.cqLD.shape,
      'cqacc_smooth': d._impl.cqacc_smooth.shape,
      'cqacc_warmstart': d._impl.cqacc_warmstart.shape,
      'cqfrc_smooth': d._impl.cqfrc_smooth.shape,
      'crb': d._impl.crb.shape,
      'crhs': d._impl.crhs.shape,
      'cvel': d.cvel.shape,
      'cx': d._impl.cx.shape,
      'dof_awake_ind': d._impl.dof_awake_ind.shape,
      'dof_cdof': d._impl.dof_cdof.shape,
      'dof_island': d._impl.dof_island.shape,
      'dof_islandid': d._impl.dof_islandid.shape,
      'efc_islandid': d._impl.efc_islandid.shape,
      'efm_K_val': d._impl.efm_K_val.shape,
      'efm_L': d._impl.efm_L.shape,
      'efm_as': d._impl.efm_as.shape,
      'efm_c': d._impl.efm_c.shape,
      'efm_ca': d._impl.efm_ca.shape,
      'efm_diag': d._impl.efm_diag.shape,
      'efm_fluid': d._impl.efm_fluid.shape,
      'efm_ts': d._impl.efm_ts.shape,
      'energy': d._impl.energy.shape,
      'face_quat': d._impl.face_quat.shape,
      'face_xpos': d._impl.face_xpos.shape,
      'flex_aabb_max': d._impl.flex_aabb_max.shape,
      'flex_aabb_min': d._impl.flex_aabb_min.shape,
      'flex_hessian_valid': d._impl.flex_hessian_valid.shape,
      'flex_rod_chain_ok': d._impl.flex_rod_chain_ok.shape,
      'flex_rod_chain_valid': d._impl.flex_rod_chain_valid.shape,
      'flex_rod_end_torque': d._impl.flex_rod_end_torque.shape,
      'flex_rod_frame0_loc': d._impl.flex_rod_frame0_loc.shape,
      'flex_rod_frame0_ref': d._impl.flex_rod_frame0_ref.shape,
      'flex_rod_ref_ok': d._impl.flex_rod_ref_ok.shape,
      'flex_rod_tail_loc': d._impl.flex_rod_tail_loc.shape,
      'flex_rodedge_frc': d._impl.flex_rodedge_frc.shape,
      'flex_rodedge_grad': d._impl.flex_rodedge_grad.shape,
      'flex_rodedge_ktheta': d._impl.flex_rodedge_ktheta.shape,
      'flex_rodedge_tridiag': d._impl.flex_rodedge_tridiag.shape,
      'flex_rodhinge_Wd': d._impl.flex_rodhinge_Wd.shape,
      'flex_rodhinge_Wk': d._impl.flex_rodhinge_Wk.shape,
      'flex_rodhinge_g': d._impl.flex_rodhinge_g.shape,
      'flex_rodhinge_om0': d._impl.flex_rodhinge_om0.shape,
      'flexedge_J': d._impl.flexedge_J.shape,
      'flexedge_hessian': d._impl.flexedge_hessian.shape,
      'flexedge_length': d._impl.flexedge_length.shape,
      'flexedge_velocity': d._impl.flexedge_velocity.shape,
      'flexnode_xpos': d._impl.flexnode_xpos.shape,
      'flexvert_hessian': d._impl.flexvert_hessian.shape,
      'flexvert_xpos': d._impl.flexvert_xpos.shape,
      'geom_xmat': d.geom_xmat.shape,
      'geom_xpos': d.geom_xpos.shape,
      'history': d.history.shape,
      'island_dofadr': d._impl.island_dofadr.shape,
      'island_idofadr': d._impl.island_idofadr.shape,
      'island_iefcadr': d._impl.island_iefcadr.shape,
      'island_ne': d._impl.island_ne.shape,
      'island_nefc': d._impl.island_nefc.shape,
      'island_nf': d._impl.island_nf.shape,
      'island_nv': d._impl.island_nv.shape,
      'light_xdir': d._impl.light_xdir.shape,
      'light_xpos': d._impl.light_xpos.shape,
      'map_dof2idof': d._impl.map_dof2idof.shape,
      'map_efc2iefc': d._impl.map_efc2iefc.shape,
      'map_idof2dof': d._impl.map_idof2dof.shape,
      'map_iefc2efc': d._impl.map_iefc2efc.shape,
      'moment_colind': d._impl.moment_colind.shape,
      'moment_rowadr': d._impl.moment_rowadr.shape,
      'moment_rownnz': d._impl.moment_rownnz.shape,
      'nacon': d._impl.nacon.shape,
      'nbody_awake': d._impl.nbody_awake.shape,
      'ncdof': d._impl.ncdof.shape,
      'ncollision': d._impl.ncollision.shape,
      'ne': d._impl.ne.shape,
      'nefc': d._impl.nefc.shape,
      'nf': d._impl.nf.shape,
      'nidof': d._impl.nidof.shape,
      'nisland': d._impl.nisland.shape,
      'nl': d._impl.nl.shape,
      'ntree_awake': d._impl.ntree_awake.shape,
      'nv_awake': d._impl.nv_awake.shape,
      'overflow': d._impl.overflow.shape,
      'plastic': d.plastic.shape,
      'qH': d._impl.qH.shape,
      'qHDiagInv': d._impl.qHDiagInv.shape,
      'qHLD': d._impl.qHLD.shape,
      'qLD': d._impl.qLD.shape,
      'qLDiagInv': d._impl.qLDiagInv.shape,
      'qLU': d._impl.qLU.shape,
      'qacc': d.qacc.shape,
      'qacc_smooth': d.qacc_smooth.shape,
      'qacc_warmstart': d.qacc_warmstart.shape,
      'qfrc_actuator': d.qfrc_actuator.shape,
      'qfrc_adhesion': d._impl.qfrc_adhesion.shape,
      'qfrc_bias': d.qfrc_bias.shape,
      'qfrc_constraint': d.qfrc_constraint.shape,
      'qfrc_damper': d._impl.qfrc_damper.shape,
      'qfrc_fluid': d.qfrc_fluid.shape,
      'qfrc_gravcomp': d.qfrc_gravcomp.shape,
      'qfrc_passive': d.qfrc_passive.shape,
      'qfrc_smooth': d.qfrc_smooth.shape,
      'qfrc_spring': d._impl.qfrc_spring.shape,
      'qpos': d.qpos.shape,
      'qvel': d.qvel.shape,
      'sensordata': d.sensordata.shape,
      'site_xmat': d.site_xmat.shape,
      'site_xpos': d.site_xpos.shape,
      'solver_niter': d._impl.solver_niter.shape,
      'subtree_angmom': d._impl.subtree_angmom.shape,
      'subtree_com': d.subtree_com.shape,
      'subtree_linvel': d._impl.subtree_linvel.shape,
      'ten_J': d._impl.ten_J.shape,
      'ten_length': d.ten_length.shape,
      'ten_velocity': d._impl.ten_velocity.shape,
      'ten_wrapadr': d._impl.ten_wrapadr.shape,
      'ten_wrapnum': d._impl.ten_wrapnum.shape,
      'time': d.time.shape,
      'tree_asleep': d._impl.tree_asleep.shape,
      'tree_awake': d._impl.tree_awake.shape,
      'tree_island': d._impl.tree_island.shape,
      'wrap_obj': d._impl.wrap_obj.shape,
      'wrap_xpos': d._impl.wrap_xpos.shape,
      'xanchor': d.xanchor.shape,
      'xaxis': d.xaxis.shape,
      'ximat': d.ximat.shape,
      'xipos': d.xipos.shape,
      'xmat': d.xmat.shape,
      'xpos': d.xpos.shape,
      'xquat': d.xquat.shape,
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
      'efc__D': d._impl.efc__D.shape,
      'efc__J': d._impl.efc__J.shape,
      'efc__J_colind': d._impl.efc__J_colind.shape,
      'efc__J_rowadr': d._impl.efc__J_rowadr.shape,
      'efc__J_rownnz': d._impl.efc__J_rownnz.shape,
      'efc__Jqvel': d._impl.efc__Jqvel.shape,
      'efc__Ma': d._impl.efc__Ma.shape,
      'efc__aref': d._impl.efc__aref.shape,
      'efc__force': d._impl.efc__force.shape,
      'efc__frictionloss': d._impl.efc__frictionloss.shape,
      'efc__id': d._impl.efc__id.shape,
      'efc__island': d._impl.efc__island.shape,
      'efc__jtdaj_adr': d._impl.efc__jtdaj_adr.shape,
      'efc__jtdaj_nblock': d._impl.efc__jtdaj_nblock.shape,
      'efc__jtdaj_nrow': d._impl.efc__jtdaj_nrow.shape,
      'efc__margin': d._impl.efc__margin.shape,
      'efc__pos': d._impl.efc__pos.shape,
      'efc__state': d._impl.efc__state.shape,
      'efc__type': d._impl.efc__type.shape,
      'efc__vel': d._impl.efc__vel.shape,
  }
  jf = ffi.jax_callable_variadic_tuple(
      _step_shim,
      num_outputs=187,
      output_dims=output_dims,
      vmap_method=None,
      in_out_argnames=set([
          'M',
          'act',
          'act_dot',
          'actuator_force',
          'actuator_length',
          'actuator_moment',
          'actuator_velocity',
          'body_awake',
          'body_awake_ind',
          'cJ',
          'cM',
          'cacc',
          'cam_xmat',
          'cam_xpos',
          'cdof',
          'cdof_dof',
          'cdof_dot',
          'cfrc_ext',
          'cfrc_int',
          'cinert',
          'cqLD',
          'cqacc_smooth',
          'cqacc_warmstart',
          'cqfrc_smooth',
          'crb',
          'crhs',
          'cvel',
          'cx',
          'dof_awake_ind',
          'dof_cdof',
          'dof_island',
          'dof_islandid',
          'efc_islandid',
          'efm_K_val',
          'efm_L',
          'efm_as',
          'efm_c',
          'efm_ca',
          'efm_diag',
          'efm_fluid',
          'efm_ts',
          'energy',
          'face_quat',
          'face_xpos',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_hessian_valid',
          'flex_rod_chain_ok',
          'flex_rod_chain_valid',
          'flex_rod_end_torque',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_ref_ok',
          'flex_rod_tail_loc',
          'flex_rodedge_frc',
          'flex_rodedge_grad',
          'flex_rodedge_ktheta',
          'flex_rodedge_tridiag',
          'flex_rodhinge_Wd',
          'flex_rodhinge_Wk',
          'flex_rodhinge_g',
          'flex_rodhinge_om0',
          'flexedge_J',
          'flexedge_hessian',
          'flexedge_length',
          'flexedge_velocity',
          'flexnode_xpos',
          'flexvert_hessian',
          'flexvert_xpos',
          'geom_xmat',
          'geom_xpos',
          'history',
          'island_dofadr',
          'island_idofadr',
          'island_iefcadr',
          'island_ne',
          'island_nefc',
          'island_nf',
          'island_nv',
          'light_xdir',
          'light_xpos',
          'map_dof2idof',
          'map_efc2iefc',
          'map_idof2dof',
          'map_iefc2efc',
          'moment_colind',
          'moment_rowadr',
          'moment_rownnz',
          'nacon',
          'nbody_awake',
          'ncdof',
          'ncollision',
          'ne',
          'nefc',
          'nf',
          'nidof',
          'nisland',
          'nl',
          'ntree_awake',
          'nv_awake',
          'overflow',
          'plastic',
          'qH',
          'qHDiagInv',
          'qHLD',
          'qLD',
          'qLDiagInv',
          'qLU',
          'qacc',
          'qacc_smooth',
          'qacc_warmstart',
          'qfrc_actuator',
          'qfrc_adhesion',
          'qfrc_bias',
          'qfrc_constraint',
          'qfrc_damper',
          'qfrc_fluid',
          'qfrc_gravcomp',
          'qfrc_passive',
          'qfrc_smooth',
          'qfrc_spring',
          'qpos',
          'qvel',
          'sensordata',
          'site_xmat',
          'site_xpos',
          'solver_niter',
          'subtree_angmom',
          'subtree_com',
          'subtree_linvel',
          'ten_J',
          'ten_length',
          'ten_velocity',
          'ten_wrapadr',
          'ten_wrapnum',
          'time',
          'tree_asleep',
          'tree_awake',
          'tree_island',
          'wrap_obj',
          'wrap_xpos',
          'xanchor',
          'xaxis',
          'ximat',
          'xipos',
          'xmat',
          'xpos',
          'xquat',
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
          'efc__D',
          'efc__J',
          'efc__J_colind',
          'efc__J_rowadr',
          'efc__J_rownnz',
          'efc__Jqvel',
          'efc__Ma',
          'efc__aref',
          'efc__force',
          'efc__frictionloss',
          'efc__id',
          'efc__island',
          'efc__jtdaj_adr',
          'efc__jtdaj_nblock',
          'efc__jtdaj_nrow',
          'efc__margin',
          'efc__pos',
          'efc__state',
          'efc__type',
          'efc__vel',
      ]),
      stage_in_argnames=set([
          'M',
          'act',
          'act_dot',
          'actuator_acc0',
          'actuator_actrange',
          'actuator_armature',
          'actuator_biasprm',
          'actuator_cranklength',
          'actuator_ctrlrange',
          'actuator_damping',
          'actuator_dampingpoly',
          'actuator_delay',
          'actuator_dynprm',
          'actuator_force',
          'actuator_forcerange',
          'actuator_gainprm',
          'actuator_gear',
          'actuator_history',
          'actuator_historyadr',
          'actuator_length',
          'actuator_lengthrange',
          'actuator_moment',
          'actuator_velocity',
          'body_awake',
          'body_awake_ind',
          'body_gravcomp',
          'body_inertia',
          'body_invweight0',
          'body_ipos',
          'body_iquat',
          'body_mass',
          'body_pos',
          'body_quat',
          'body_subtreemass',
          'cJ',
          'cM',
          'cMa',
          'cacc',
          'cam_fovy',
          'cam_intrinsic',
          'cam_mat0',
          'cam_pos',
          'cam_pos0',
          'cam_poscom0',
          'cam_quat',
          'cam_xmat',
          'cam_xpos',
          'cdof',
          'cdof_dof',
          'cdof_dot',
          'cdof_tri_col',
          'cdof_tri_row',
          'cfrc_ext',
          'cfrc_int',
          'cinert',
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
          'cqLD',
          'cqacc',
          'cqacc_smooth',
          'cqacc_warmstart',
          'cqfrc_constraint',
          'cqfrc_smooth',
          'crb',
          'crhs',
          'ctrl',
          'cvel',
          'cx',
          'dof_armature',
          'dof_awake_ind',
          'dof_cdof',
          'dof_damping',
          'dof_dampingpoly',
          'dof_frictionloss',
          'dof_invweight0',
          'dof_island',
          'dof_islandid',
          'dof_solimp',
          'dof_solref',
          'efc__D',
          'efc__J',
          'efc__J_colind',
          'efc__J_rowadr',
          'efc__J_rownnz',
          'efc__Jqvel',
          'efc__Ma',
          'efc__aref',
          'efc__force',
          'efc__frictionloss',
          'efc__id',
          'efc__island',
          'efc__jtdaj_adr',
          'efc__jtdaj_nblock',
          'efc__jtdaj_nrow',
          'efc__margin',
          'efc__pos',
          'efc__state',
          'efc__type',
          'efc__vel',
          'efc_islandid',
          'efm_K_val',
          'efm_L',
          'efm_as',
          'efm_c',
          'efm_ca',
          'efm_diag',
          'efm_fluid',
          'efm_ts',
          'energy',
          'eq_active',
          'eq_data',
          'eq_solimp',
          'eq_solref',
          'face_quat',
          'face_xpos',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_hessian_valid',
          'flex_rod_chain_ok',
          'flex_rod_chain_valid',
          'flex_rod_end_torque',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_ref_ok',
          'flex_rod_tail_loc',
          'flex_rodedge_frc',
          'flex_rodedge_grad',
          'flex_rodedge_ktheta',
          'flex_rodedge_tridiag',
          'flex_rodhinge_Wd',
          'flex_rodhinge_Wk',
          'flex_rodhinge_g',
          'flex_rodhinge_om0',
          'flexedge_J',
          'flexedge_hessian',
          'flexedge_length',
          'flexedge_velocity',
          'flexnode_xpos',
          'flexvert_hessian',
          'flexvert_xpos',
          'geom_aabb',
          'geom_adhesion',
          'geom_friction',
          'geom_gap',
          'geom_margin',
          'geom_matid',
          'geom_pos',
          'geom_quat',
          'geom_rbound',
          'geom_rgba',
          'geom_size',
          'geom_solimp',
          'geom_solmix',
          'geom_solref',
          'geom_surfacevel',
          'geom_xmat',
          'geom_xpos',
          'hfield_data',
          'history',
          'island_dofadr',
          'island_idofadr',
          'island_iefcadr',
          'island_ne',
          'island_nefc',
          'island_nf',
          'island_nv',
          'jnt_actfrcrange',
          'jnt_axis',
          'jnt_margin',
          'jnt_pos',
          'jnt_range',
          'jnt_solimp',
          'jnt_solref',
          'jnt_springyield',
          'jnt_stiffness',
          'jnt_stiffnesspoly',
          'light_dir',
          'light_dir0',
          'light_pos',
          'light_pos0',
          'light_poscom0',
          'light_xdir',
          'light_xpos',
          'map_dof2idof',
          'map_efc2iefc',
          'map_idof2dof',
          'map_iefc2efc',
          'mat_rgba',
          'mocap_pos',
          'mocap_quat',
          'moment_colind',
          'moment_rowadr',
          'moment_rownnz',
          'nacon',
          'nbody_awake',
          'ncdof',
          'ncollision',
          'ne',
          'nefc',
          'nf',
          'nidof',
          'nisland',
          'nl',
          'ntree_awake',
          'nv_awake',
          'opt__ccd_tolerance',
          'opt__density',
          'opt__gravity',
          'opt__impratio_invsqrt',
          'opt__ls_tolerance',
          'opt__magnetic',
          'opt__noslip_tolerance',
          'opt__sleep_tolerance',
          'opt__timestep',
          'opt__tolerance',
          'opt__viscosity',
          'opt__wind',
          'overflow',
          'pair_adhesion',
          'pair_friction',
          'pair_gap',
          'pair_margin',
          'pair_solimp',
          'pair_solref',
          'pair_solreffriction',
          'plastic',
          'qH',
          'qHDiagInv',
          'qHLD',
          'qLD',
          'qLDiagInv',
          'qLU',
          'qacc',
          'qacc_smooth',
          'qacc_warmstart',
          'qfrc_actuator',
          'qfrc_adhesion',
          'qfrc_applied',
          'qfrc_bias',
          'qfrc_constraint',
          'qfrc_damper',
          'qfrc_fluid',
          'qfrc_gravcomp',
          'qfrc_passive',
          'qfrc_smooth',
          'qfrc_spring',
          'qpos',
          'qpos0',
          'qpos_spring',
          'qvel',
          'sensor_delay',
          'sensor_history',
          'sensor_historyadr',
          'sensor_interval',
          'sensordata',
          'site_pos',
          'site_quat',
          'site_xmat',
          'site_xpos',
          'solver_niter',
          'subtree_angmom',
          'subtree_com',
          'subtree_linvel',
          'ten_J',
          'ten_length',
          'ten_velocity',
          'ten_wrapadr',
          'ten_wrapnum',
          'tendon_actfrcrange',
          'tendon_armature',
          'tendon_damping',
          'tendon_dampingpoly',
          'tendon_frictionloss',
          'tendon_invweight0',
          'tendon_length0',
          'tendon_lengthspring',
          'tendon_margin',
          'tendon_range',
          'tendon_solimp_fri',
          'tendon_solimp_lim',
          'tendon_solref_fri',
          'tendon_solref_lim',
          'tendon_springyield',
          'tendon_stiffness',
          'tendon_stiffnesspoly',
          'time',
          'tree_asleep',
          'tree_awake',
          'tree_island',
          'wrap_obj',
          'wrap_xpos',
          'xanchor',
          'xaxis',
          'xfrc_applied',
          'ximat',
          'xipos',
          'xmat',
          'xpos',
          'xquat',
      ]),
      stage_out_argnames=set([
          'M',
          'act',
          'act_dot',
          'actuator_force',
          'actuator_length',
          'actuator_moment',
          'actuator_velocity',
          'body_awake',
          'body_awake_ind',
          'cJ',
          'cM',
          'cacc',
          'cam_xmat',
          'cam_xpos',
          'cdof',
          'cdof_dof',
          'cdof_dot',
          'cfrc_ext',
          'cfrc_int',
          'cinert',
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
          'cqLD',
          'cqacc_smooth',
          'cqacc_warmstart',
          'cqfrc_smooth',
          'crb',
          'crhs',
          'cvel',
          'cx',
          'dof_awake_ind',
          'dof_cdof',
          'dof_island',
          'dof_islandid',
          'efc__D',
          'efc__J',
          'efc__J_colind',
          'efc__J_rowadr',
          'efc__J_rownnz',
          'efc__Jqvel',
          'efc__Ma',
          'efc__aref',
          'efc__force',
          'efc__frictionloss',
          'efc__id',
          'efc__island',
          'efc__jtdaj_adr',
          'efc__jtdaj_nblock',
          'efc__jtdaj_nrow',
          'efc__margin',
          'efc__pos',
          'efc__state',
          'efc__type',
          'efc__vel',
          'efc_islandid',
          'efm_K_val',
          'efm_L',
          'efm_as',
          'efm_c',
          'efm_ca',
          'efm_diag',
          'efm_fluid',
          'efm_ts',
          'energy',
          'face_quat',
          'face_xpos',
          'flex_aabb_max',
          'flex_aabb_min',
          'flex_hessian_valid',
          'flex_rod_chain_ok',
          'flex_rod_chain_valid',
          'flex_rod_end_torque',
          'flex_rod_frame0_loc',
          'flex_rod_frame0_ref',
          'flex_rod_ref_ok',
          'flex_rod_tail_loc',
          'flex_rodedge_frc',
          'flex_rodedge_grad',
          'flex_rodedge_ktheta',
          'flex_rodedge_tridiag',
          'flex_rodhinge_Wd',
          'flex_rodhinge_Wk',
          'flex_rodhinge_g',
          'flex_rodhinge_om0',
          'flexedge_J',
          'flexedge_hessian',
          'flexedge_length',
          'flexedge_velocity',
          'flexnode_xpos',
          'flexvert_hessian',
          'flexvert_xpos',
          'geom_xmat',
          'geom_xpos',
          'history',
          'island_dofadr',
          'island_idofadr',
          'island_iefcadr',
          'island_ne',
          'island_nefc',
          'island_nf',
          'island_nv',
          'light_xdir',
          'light_xpos',
          'map_dof2idof',
          'map_efc2iefc',
          'map_idof2dof',
          'map_iefc2efc',
          'moment_colind',
          'moment_rowadr',
          'moment_rownnz',
          'nacon',
          'nbody_awake',
          'ncdof',
          'ncollision',
          'ne',
          'nefc',
          'nf',
          'nidof',
          'nisland',
          'nl',
          'ntree_awake',
          'nv_awake',
          'overflow',
          'plastic',
          'qH',
          'qHDiagInv',
          'qHLD',
          'qLD',
          'qLDiagInv',
          'qLU',
          'qacc',
          'qacc_smooth',
          'qacc_warmstart',
          'qfrc_actuator',
          'qfrc_adhesion',
          'qfrc_bias',
          'qfrc_constraint',
          'qfrc_damper',
          'qfrc_fluid',
          'qfrc_gravcomp',
          'qfrc_passive',
          'qfrc_smooth',
          'qfrc_spring',
          'qpos',
          'qvel',
          'sensordata',
          'site_xmat',
          'site_xpos',
          'solver_niter',
          'subtree_angmom',
          'subtree_com',
          'subtree_linvel',
          'ten_J',
          'ten_length',
          'ten_velocity',
          'ten_wrapadr',
          'ten_wrapnum',
          'time',
          'tree_asleep',
          'tree_awake',
          'tree_island',
          'wrap_obj',
          'wrap_xpos',
          'xanchor',
          'xaxis',
          'ximat',
          'xipos',
          'xmat',
          'xpos',
          'xquat',
      ]),
      graph_mode=m.opt._impl.graph_mode,
      has_side_effect=False,
  )
  out = jf(
      d.qpos.shape[0],
      m._impl.D_colind,
      m._impl.D_diag,
      m._impl.D_rowadr,
      m._impl.D_rownnz,
      m.M_colind,
      m._impl.M_elemid,
      m._impl.M_fullm_i,
      m._impl.M_fullm_j,
      m._impl.M_hinit_i,
      m._impl.M_mulm_col,
      m._impl.M_mulm_madr,
      m._impl.M_mulm_rowadr,
      m.M_rowadr,
      m.M_rownnz,
      m._impl.M_tiles,
      m.actuator_acc0,
      m.actuator_actadr,
      m.actuator_actearly,
      m.actuator_actlimited,
      m.actuator_actnum,
      m.actuator_actrange,
      m._impl.actuator_armature,
      m.actuator_biasprm,
      m.actuator_biastype,
      m.actuator_cranklength,
      m._impl.actuator_ctrladr,
      m.actuator_ctrllimited,
      m._impl.actuator_ctrlnum,
      m.actuator_ctrlrange,
      m._impl.actuator_ctrlspec,
      m._impl.actuator_damping,
      m._impl.actuator_dampingpoly,
      m._impl.actuator_delay,
      m.actuator_dynprm,
      m.actuator_dyntype,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      m.actuator_gainprm,
      m.actuator_gaintype,
      m.actuator_gear,
      m._impl.actuator_history,
      m._impl.actuator_historyadr,
      m.actuator_lengthrange,
      m.actuator_trnid,
      m.actuator_trntype,
      m._impl.actuator_trntype_body_adr,
      m._impl.block_dim,
      m._impl.body_branch_start,
      m._impl.body_branches,
      m.body_dofadr,
      m.body_dofnum,
      m._impl.body_fluid_box_adr,
      m._impl.body_fluid_ellipsoid,
      m._impl.body_fluid_ellipsoid_adr,
      m._impl.body_freeadr,
      m.body_geomadr,
      m.body_geomnum,
      m.body_gravcomp,
      m.body_inertia,
      m.body_invweight0,
      m.body_ipos,
      m.body_iquat,
      m._impl.body_is_free,
      m._impl.body_isdofancestor,
      m.body_jntadr,
      m.body_jntnum,
      m.body_mass,
      m.body_mocapid,
      m.body_parentid,
      m.body_pos,
      m.body_quat,
      m.body_rootid,
      m.body_simple,
      m.body_subtreemass,
      m._impl.body_subtreenum,
      m._impl.body_tree,
      m.body_treeid,
      m.body_weldid,
      m.cam_bodyid,
      m.cam_fovy,
      m.cam_intrinsic,
      m.cam_mat0,
      m.cam_mode,
      m.cam_pos,
      m.cam_pos0,
      m.cam_poscom0,
      m._impl.cam_projection,
      m.cam_quat,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_targetbodyid,
      m.dof_armature,
      m.dof_bodyid,
      m.dof_damping,
      m.dof_dampingpoly,
      m.dof_frictionloss,
      m.dof_invweight0,
      m.dof_jntid,
      m._impl.dof_length,
      m.dof_parentid,
      m.dof_solimp,
      m.dof_solref,
      m.dof_treeid,
      m._impl.dof_tri_col,
      m._impl.dof_tri_row,
      m._impl.efm0_L,
      m._impl.efm0_L_colind,
      m._impl.efm0_L_rowadr,
      m._impl.efm0_L_rownnz,
      m._impl.efm0_active,
      m._impl.efm0_dofid,
      m._impl.efm_K_colind,
      m._impl.efm_K_rowadr,
      m._impl.efm_K_rownnz,
      m._impl.efm_con_max_dof,
      m._impl.efm_dofblk,
      m._impl.efm_dofid,
      m._impl.eq_connect_adr,
      m.eq_data,
      m._impl.eq_flex_adr,
      m._impl.eq_flexstrain_adr,
      m._impl.eq_jnt_adr,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.eq_solimp,
      m.eq_solref,
      m._impl.eq_ten_adr,
      m.eq_type,
      m._impl.eq_wld_adr,
      m._impl.flex_activelayers,
      m._impl.flex_bend_interp_map,
      m._impl.flex_bending,
      m._impl.flex_bendingadr,
      m._impl.flex_cell_map,
      m._impl.flex_cellnum,
      m._impl.flex_centered,
      m._impl.flex_conaffinity,
      m._impl.flex_condim,
      m._impl.flex_contype,
      m._impl.flex_damping,
      m._impl.flex_dim,
      m._impl.flex_edge,
      m._impl.flex_edgeadr,
      m._impl.flex_edgedamping,
      m._impl.flex_edgeequality,
      m._impl.flex_edgeflap,
      m._impl.flex_edgeflexid,
      m._impl.flex_edgenum,
      m._impl.flex_edgestiffness,
      m._impl.flex_eff_possible,
      m._impl.flex_elem,
      m._impl.flex_elem_rodedge,
      m._impl.flex_elemadr,
      m._impl.flex_elemdataadr,
      m._impl.flex_elemedge,
      m._impl.flex_elemedgeadr,
      m._impl.flex_elemflexid,
      m._impl.flex_elemlayer,
      m._impl.flex_elemnum,
      m._impl.flex_face,
      m._impl.flex_face_map,
      m._impl.flex_faceadr,
      m._impl.flex_friction,
      m._impl.flex_gap,
      m.flex_interp,
      m._impl.flex_interp_assemblable,
      m._impl.flex_margin,
      m._impl.flex_node,
      m.flex_node0,
      m.flex_nodeadr,
      m.flex_nodebodyid,
      m.flex_nodenum,
      m._impl.flex_passive,
      m._impl.flex_plasticadr,
      m._impl.flex_plasticmode,
      m._impl.flex_priority,
      m._impl.flex_radius,
      m._impl.flex_rigid,
      m._impl.flex_rod_box_size,
      m._impl.flex_rod_edgeadr,
      m._impl.flex_rod_edgenum,
      m._impl.flex_rod_has_twist,
      m._impl.flex_rod_hingeadr,
      m._impl.flex_rod_is_box,
      m._impl.flex_rod_is_chain,
      m._impl.flex_rod_root_clamped,
      m._impl.flex_rod_tail_clamped,
      m._impl.flex_rodedge_jnt,
      m._impl.flex_rodedge_vert,
      m._impl.flex_rodhinge_map,
      m._impl.flex_selfcollide,
      m._impl.flex_simple,
      m._impl.flex_size,
      m._impl.flex_solimp,
      m._impl.flex_solmix,
      m._impl.flex_solref,
      m._impl.flex_stiffness,
      m._impl.flex_stiffnessadr,
      m._impl.flex_vert,
      m.flex_vert0,
      m._impl.flex_vert_rodelem,
      m.flex_vertadr,
      m._impl.flex_vertbodyid,
      m._impl.flex_vertcell,
      m._impl.flex_vertflexid,
      m.flex_vertnum,
      m._impl.flex_yield,
      m._impl.flexedge_J_colind,
      m._impl.flexedge_J_rowadr,
      m._impl.flexedge_J_rownnz,
      m._impl.flexedge_invweight0,
      m._impl.flexedge_length0,
      m._impl.flexedge_rigid,
      m._impl.flexstrain_J_colind,
      m._impl.flexstrain_J_rowadr,
      m._impl.flexstrain_J_rownnz,
      m._impl.flg_adhesion,
      m.flg_surfacevel,
      m.geom_aabb,
      m._impl.geom_adhesion,
      m.geom_bodyid,
      m.geom_conaffinity,
      m.geom_condim,
      m.geom_contype,
      jax.numpy.expand_dims(m.geom_dataid, 0),
      m.geom_fluid,
      m.geom_friction,
      m.geom_gap,
      m.geom_group,
      m.geom_margin,
      m.geom_matid,
      m._impl.geom_pair_type_count,
      m._impl.geom_plugin_index,
      m.geom_pos,
      m.geom_priority,
      m.geom_quat,
      m.geom_rbound,
      m.geom_rgba,
      m.geom_size,
      m.geom_solimp,
      m.geom_solmix,
      m.geom_solref,
      m._impl.geom_surfacevel,
      m.geom_type,
      m._impl.has_1d_flex,
      m._impl.has_3d_flex,
      m._impl.has_actuator_dcmotor,
      m._impl.has_actuator_wrap,
      m._impl.has_efm_actuator,
      m._impl.has_ellipsoid_geom,
      m._impl.has_flex_edge_stiff_damp,
      m._impl.has_flex_passive,
      m._impl.has_flex_selfcollide,
      m._impl.has_flex_snh,
      m._impl.has_fluid,
      m._impl.has_non_simple_flex,
      m._impl.has_plane_geom,
      m._impl.has_sdf_geom,
      m._impl.has_tendon_damping,
      m._impl.has_tendon_stiffness,
      m._impl.has_unsupported_flex_interp,
      m.hfield_adr,
      m.hfield_data,
      m.hfield_ncol,
      m.hfield_nrow,
      m.hfield_size,
      m._impl.is_sparse,
      m.jnt_actfrclimited,
      m.jnt_actfrcrange,
      m.jnt_actgravcomp,
      m._impl.jnt_actuatorid,
      m.jnt_axis,
      m.jnt_bodyid,
      m.jnt_dofadr,
      m._impl.jnt_limited_ball_adr,
      m._impl.jnt_limited_slide_hinge_adr,
      m.jnt_margin,
      m._impl.jnt_plasticadr,
      m.jnt_pos,
      m.jnt_qposadr,
      m.jnt_range,
      m.jnt_solimp,
      m.jnt_solref,
      m._impl.jnt_springyield,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.jnt_type,
      m._impl.light_bodyid,
      m.light_dir,
      m.light_dir0,
      m.light_mode,
      m.light_pos,
      m.light_pos0,
      m.light_poscom0,
      m._impl.light_targetbodyid,
      m._impl.mapM2D,
      m.mat_rgba,
      m._impl.max_ten_J_rownnz,
      m.mesh_face,
      m.mesh_faceadr,
      m.mesh_graph,
      m.mesh_graphadr,
      m.mesh_normal,
      m.mesh_normaladr,
      m.mesh_normalnum,
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
      m.mesh_quat,
      m.mesh_vert,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.nC,
      m.nD,
      m.nJten,
      m.nM,
      m.na,
      m._impl.nacttrnbody,
      m._impl.nactuator,
      m.nbody,
      m._impl.nbranch,
      m.ncam,
      m._impl.nefm0dof,
      m._impl.nefmK,
      m._impl.nefmdof,
      m.neq,
      m.nflex,
      m._impl.nflexbend_interp,
      m._impl.nflexedge,
      m._impl.nflexelem,
      m._impl.nflexface,
      m._impl.nflexintcell,
      m._impl.nflexnode,
      m._impl.nflexrodedge,
      m._impl.nflexrodhinge,
      m._impl.nflexvert,
      m.ngeom,
      m.nhistory,
      m.njnt,
      m.nlight,
      m._impl.nmaxcondim,
      m._impl.nmaxpyramid,
      m.nmesh,
      m._impl.nmeshdegmax,
      m.nmeshface,
      m.npair,
      m.nplastic,
      m._impl.npolygonmax,
      m._impl.nrangefinder,
      m._impl.nsensorcollision,
      m._impl.nsensortaxel,
      m.nsite,
      m._impl.ntactileweld,
      m.ntendon,
      m._impl.ntree,
      m.nu,
      m.nv,
      m._impl.nv_pad,
      m.nwrap,
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
      m._impl.qD_fullm_i,
      m._impl.qD_fullm_j,
      m._impl.qLD_all_updates,
      m._impl.qLD_block_adr,
      m._impl.qLD_block_total,
      m._impl.qLD_level_offsets,
      m._impl.qLD_updates,
      m.qpos0,
      m.qpos_spring,
      m._impl.rangefinder_sensor_adr,
      m._impl.sensor_acc_adr,
      m.sensor_adr,
      m._impl.sensor_contact_acc_adr,
      m._impl.sensor_contact_pos_adr,
      m.sensor_cutoff,
      m.sensor_datatype,
      m._impl.sensor_delay,
      m.sensor_dim,
      m._impl.sensor_e_kinetic,
      m._impl.sensor_e_potential,
      m._impl.sensor_history,
      m._impl.sensor_historyadr,
      m._impl.sensor_interval,
      m.sensor_intprm,
      m._impl.sensor_limitfrc_adr,
      m._impl.sensor_limitpos_adr,
      m._impl.sensor_limitvel_adr,
      m.sensor_objid,
      m.sensor_objtype,
      m._impl.sensor_pos_adr,
      m._impl.sensor_rangefinder_adr,
      m._impl.sensor_rangefinder_bodyid,
      m.sensor_refid,
      m.sensor_reftype,
      m._impl.sensor_rne_postconstraint,
      m._impl.sensor_stage_acc_adr,
      m._impl.sensor_stage_pos_adr,
      m._impl.sensor_stage_vel_adr,
      m._impl.sensor_subtree_vel,
      m._impl.sensor_tendonactfrc_adr,
      m._impl.sensor_touch_adr,
      m.sensor_type,
      m._impl.sensor_vel_adr,
      m.site_bodyid,
      m.site_pos,
      m.site_quat,
      m.site_size,
      m.site_type,
      m._impl.taxel_sensorid,
      m._impl.taxel_vertadr,
      m._impl.ten_J_colind,
      m._impl.ten_J_rowadr,
      m._impl.ten_J_rownnz,
      m.tendon_actfrclimited,
      m.tendon_actfrcrange,
      m._impl.tendon_actuatorid,
      m.tendon_adr,
      m.tendon_armature,
      m.tendon_damping,
      m.tendon_dampingpoly,
      m.tendon_frictionloss,
      m._impl.tendon_geom_adr,
      m.tendon_invweight0,
      m._impl.tendon_jnt_adr,
      m.tendon_length0,
      m.tendon_lengthspring,
      m.tendon_limited,
      m._impl.tendon_limited_adr,
      m.tendon_margin,
      m.tendon_num,
      m._impl.tendon_plasticadr,
      m.tendon_range,
      m._impl.tendon_site_pair_adr,
      m.tendon_solimp_fri,
      m.tendon_solimp_lim,
      m.tendon_solref_fri,
      m.tendon_solref_lim,
      m._impl.tendon_springyield,
      m.tendon_stiffness,
      m.tendon_stiffnesspoly,
      m.tendon_type,
      m._impl.tree_dofadr,
      m._impl.tree_dofnum,
      m._impl.tree_sleep_policy,
      m._impl.weld_tactile_id,
      m._impl.wrap_geom_adr,
      m._impl.wrap_jnt_adr,
      m.wrap_objid,
      m.wrap_prm,
      m._impl.wrap_pulley_scale,
      m._impl.wrap_site_pair_adr,
      m.wrap_type,
      m.opt._impl.broadphase,
      m.opt._impl.broadphase_filter,
      m.opt._impl.ccd_iterations,
      m.opt._impl.ccd_tolerance,
      m.opt.cone,
      m.opt._impl.contact_sensor_maxmatch,
      m.opt.density,
      m.opt._impl.deterministic,
      m.opt.disableflags,
      m.opt.enableflags,
      m.opt._impl.graph_conditional,
      m.opt.gravity,
      m.opt._impl.impratio_invsqrt,
      m.opt.integrator,
      m.opt.iterations,
      m.opt.ls_iterations,
      m.opt.ls_tolerance,
      m.opt.magnetic,
      m.opt._impl.noslip_iterations,
      m.opt._impl.noslip_tolerance,
      m.opt._impl.run_collision_detection,
      m.opt._impl.run_rne_postconstraint,
      m.opt._impl.sdf_initpoints,
      m.opt._impl.sdf_iterations,
      m.opt._impl.sleep_tolerance,
      m.opt.solver,
      m.opt.timestep,
      m.opt.tolerance,
      m.opt.viscosity,
      m.opt._impl.warn_overflow,
      m.opt.wind,
      d._impl.naccdmax,
      d._impl.naconmax,
      d._impl.njmax,
      d._impl.njmax_nnz,
      d._impl.nvmax,
      d._impl.nvmax_pad,
      d._impl.M,
      d.act,
      d.act_dot,
      d.actuator_force,
      d.actuator_length,
      d._impl.actuator_moment,
      d._impl.actuator_velocity,
      d._impl.body_awake,
      d._impl.body_awake_ind,
      d._impl.cJ,
      d._impl.cM,
      d._impl.cMa,
      d._impl.cacc,
      d.cam_xmat,
      d.cam_xpos,
      d.cdof,
      d._impl.cdof_dof,
      d.cdof_dot,
      d._impl.cdof_tri_col,
      d._impl.cdof_tri_row,
      d._impl.cfrc_ext,
      d._impl.cfrc_int,
      d._impl.cinert,
      d._impl.cqLD,
      d._impl.cqacc,
      d._impl.cqacc_smooth,
      d._impl.cqacc_warmstart,
      d._impl.cqfrc_constraint,
      d._impl.cqfrc_smooth,
      d._impl.crb,
      d._impl.crhs,
      d.ctrl,
      d.cvel,
      d._impl.cx,
      d._impl.dof_awake_ind,
      d._impl.dof_cdof,
      d._impl.dof_island,
      d._impl.dof_islandid,
      d._impl.efc_islandid,
      d._impl.efm_K_val,
      d._impl.efm_L,
      d._impl.efm_as,
      d._impl.efm_c,
      d._impl.efm_ca,
      d._impl.efm_diag,
      d._impl.efm_fluid,
      d._impl.efm_ts,
      d._impl.energy,
      d.eq_active,
      d._impl.face_quat,
      d._impl.face_xpos,
      d._impl.flex_aabb_max,
      d._impl.flex_aabb_min,
      d._impl.flex_hessian_valid,
      d._impl.flex_rod_chain_ok,
      d._impl.flex_rod_chain_valid,
      d._impl.flex_rod_end_torque,
      d._impl.flex_rod_frame0_loc,
      d._impl.flex_rod_frame0_ref,
      d._impl.flex_rod_ref_ok,
      d._impl.flex_rod_tail_loc,
      d._impl.flex_rodedge_frc,
      d._impl.flex_rodedge_grad,
      d._impl.flex_rodedge_ktheta,
      d._impl.flex_rodedge_tridiag,
      d._impl.flex_rodhinge_Wd,
      d._impl.flex_rodhinge_Wk,
      d._impl.flex_rodhinge_g,
      d._impl.flex_rodhinge_om0,
      d._impl.flexedge_J,
      d._impl.flexedge_hessian,
      d._impl.flexedge_length,
      d._impl.flexedge_velocity,
      d._impl.flexnode_xpos,
      d._impl.flexvert_hessian,
      d._impl.flexvert_xpos,
      d.geom_xmat,
      d.geom_xpos,
      d.history,
      d._impl.island_dofadr,
      d._impl.island_idofadr,
      d._impl.island_iefcadr,
      d._impl.island_ne,
      d._impl.island_nefc,
      d._impl.island_nf,
      d._impl.island_nv,
      d._impl.light_xdir,
      d._impl.light_xpos,
      d._impl.map_dof2idof,
      d._impl.map_efc2iefc,
      d._impl.map_idof2dof,
      d._impl.map_iefc2efc,
      d.mocap_pos,
      d.mocap_quat,
      d._impl.moment_colind,
      d._impl.moment_rowadr,
      d._impl.moment_rownnz,
      d._impl.nacon,
      d._impl.nbody_awake,
      d._impl.ncdof,
      d._impl.ncollision,
      d._impl.ne,
      d._impl.nefc,
      d._impl.nf,
      d._impl.nidof,
      d._impl.nisland,
      d._impl.nl,
      d._impl.ntree_awake,
      d._impl.nv_awake,
      d._impl.overflow,
      d.plastic,
      d._impl.qH,
      d._impl.qHDiagInv,
      d._impl.qHLD,
      d._impl.qLD,
      d._impl.qLDiagInv,
      d._impl.qLU,
      d.qacc,
      d.qacc_smooth,
      d.qacc_warmstart,
      d.qfrc_actuator,
      d._impl.qfrc_adhesion,
      d.qfrc_applied,
      d.qfrc_bias,
      d.qfrc_constraint,
      d._impl.qfrc_damper,
      d.qfrc_fluid,
      d.qfrc_gravcomp,
      d.qfrc_passive,
      d.qfrc_smooth,
      d._impl.qfrc_spring,
      d.qpos,
      d.qvel,
      d.sensordata,
      d.site_xmat,
      d.site_xpos,
      d._impl.solver_niter,
      d._impl.subtree_angmom,
      d.subtree_com,
      d._impl.subtree_linvel,
      d._impl.ten_J,
      d.ten_length,
      d._impl.ten_velocity,
      d._impl.ten_wrapadr,
      d._impl.ten_wrapnum,
      d.time,
      d._impl.tree_asleep,
      d._impl.tree_awake,
      d._impl.tree_island,
      d._impl.wrap_obj,
      d._impl.wrap_xpos,
      d.xanchor,
      d.xaxis,
      d.xfrc_applied,
      d.ximat,
      d.xipos,
      d.xmat,
      d.xpos,
      d.xquat,
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
      d._impl.efc__D,
      d._impl.efc__J,
      d._impl.efc__J_colind,
      d._impl.efc__J_rowadr,
      d._impl.efc__J_rownnz,
      d._impl.efc__Jqvel,
      d._impl.efc__Ma,
      d._impl.efc__aref,
      d._impl.efc__force,
      d._impl.efc__frictionloss,
      d._impl.efc__id,
      d._impl.efc__island,
      d._impl.efc__jtdaj_adr,
      d._impl.efc__jtdaj_nblock,
      d._impl.efc__jtdaj_nrow,
      d._impl.efc__margin,
      d._impl.efc__pos,
      d._impl.efc__state,
      d._impl.efc__type,
      d._impl.efc__vel,
  )
  d = d.tree_replace({
      '_impl.M': out[0],
      'act': out[1],
      'act_dot': out[2],
      'actuator_force': out[3],
      'actuator_length': out[4],
      '_impl.actuator_moment': out[5],
      '_impl.actuator_velocity': out[6],
      '_impl.body_awake': out[7],
      '_impl.body_awake_ind': out[8],
      '_impl.cJ': out[9],
      '_impl.cM': out[10],
      '_impl.cacc': out[11],
      'cam_xmat': out[12],
      'cam_xpos': out[13],
      'cdof': out[14],
      '_impl.cdof_dof': out[15],
      'cdof_dot': out[16],
      '_impl.cfrc_ext': out[17],
      '_impl.cfrc_int': out[18],
      '_impl.cinert': out[19],
      '_impl.cqLD': out[20],
      '_impl.cqacc_smooth': out[21],
      '_impl.cqacc_warmstart': out[22],
      '_impl.cqfrc_smooth': out[23],
      '_impl.crb': out[24],
      '_impl.crhs': out[25],
      'cvel': out[26],
      '_impl.cx': out[27],
      '_impl.dof_awake_ind': out[28],
      '_impl.dof_cdof': out[29],
      '_impl.dof_island': out[30],
      '_impl.dof_islandid': out[31],
      '_impl.efc_islandid': out[32],
      '_impl.efm_K_val': out[33],
      '_impl.efm_L': out[34],
      '_impl.efm_as': out[35],
      '_impl.efm_c': out[36],
      '_impl.efm_ca': out[37],
      '_impl.efm_diag': out[38],
      '_impl.efm_fluid': out[39],
      '_impl.efm_ts': out[40],
      '_impl.energy': out[41],
      '_impl.face_quat': out[42],
      '_impl.face_xpos': out[43],
      '_impl.flex_aabb_max': out[44],
      '_impl.flex_aabb_min': out[45],
      '_impl.flex_hessian_valid': out[46],
      '_impl.flex_rod_chain_ok': out[47],
      '_impl.flex_rod_chain_valid': out[48],
      '_impl.flex_rod_end_torque': out[49],
      '_impl.flex_rod_frame0_loc': out[50],
      '_impl.flex_rod_frame0_ref': out[51],
      '_impl.flex_rod_ref_ok': out[52],
      '_impl.flex_rod_tail_loc': out[53],
      '_impl.flex_rodedge_frc': out[54],
      '_impl.flex_rodedge_grad': out[55],
      '_impl.flex_rodedge_ktheta': out[56],
      '_impl.flex_rodedge_tridiag': out[57],
      '_impl.flex_rodhinge_Wd': out[58],
      '_impl.flex_rodhinge_Wk': out[59],
      '_impl.flex_rodhinge_g': out[60],
      '_impl.flex_rodhinge_om0': out[61],
      '_impl.flexedge_J': out[62],
      '_impl.flexedge_hessian': out[63],
      '_impl.flexedge_length': out[64],
      '_impl.flexedge_velocity': out[65],
      '_impl.flexnode_xpos': out[66],
      '_impl.flexvert_hessian': out[67],
      '_impl.flexvert_xpos': out[68],
      'geom_xmat': out[69],
      'geom_xpos': out[70],
      'history': out[71],
      '_impl.island_dofadr': out[72],
      '_impl.island_idofadr': out[73],
      '_impl.island_iefcadr': out[74],
      '_impl.island_ne': out[75],
      '_impl.island_nefc': out[76],
      '_impl.island_nf': out[77],
      '_impl.island_nv': out[78],
      '_impl.light_xdir': out[79],
      '_impl.light_xpos': out[80],
      '_impl.map_dof2idof': out[81],
      '_impl.map_efc2iefc': out[82],
      '_impl.map_idof2dof': out[83],
      '_impl.map_iefc2efc': out[84],
      '_impl.moment_colind': out[85],
      '_impl.moment_rowadr': out[86],
      '_impl.moment_rownnz': out[87],
      '_impl.nacon': out[88],
      '_impl.nbody_awake': out[89],
      '_impl.ncdof': out[90],
      '_impl.ncollision': out[91],
      '_impl.ne': out[92],
      '_impl.nefc': out[93],
      '_impl.nf': out[94],
      '_impl.nidof': out[95],
      '_impl.nisland': out[96],
      '_impl.nl': out[97],
      '_impl.ntree_awake': out[98],
      '_impl.nv_awake': out[99],
      '_impl.overflow': out[100],
      'plastic': out[101],
      '_impl.qH': out[102],
      '_impl.qHDiagInv': out[103],
      '_impl.qHLD': out[104],
      '_impl.qLD': out[105],
      '_impl.qLDiagInv': out[106],
      '_impl.qLU': out[107],
      'qacc': out[108],
      'qacc_smooth': out[109],
      'qacc_warmstart': out[110],
      'qfrc_actuator': out[111],
      '_impl.qfrc_adhesion': out[112],
      'qfrc_bias': out[113],
      'qfrc_constraint': out[114],
      '_impl.qfrc_damper': out[115],
      'qfrc_fluid': out[116],
      'qfrc_gravcomp': out[117],
      'qfrc_passive': out[118],
      'qfrc_smooth': out[119],
      '_impl.qfrc_spring': out[120],
      'qpos': out[121],
      'qvel': out[122],
      'sensordata': out[123],
      'site_xmat': out[124],
      'site_xpos': out[125],
      '_impl.solver_niter': out[126],
      '_impl.subtree_angmom': out[127],
      'subtree_com': out[128],
      '_impl.subtree_linvel': out[129],
      '_impl.ten_J': out[130],
      'ten_length': out[131],
      '_impl.ten_velocity': out[132],
      '_impl.ten_wrapadr': out[133],
      '_impl.ten_wrapnum': out[134],
      'time': out[135],
      '_impl.tree_asleep': out[136],
      '_impl.tree_awake': out[137],
      '_impl.tree_island': out[138],
      '_impl.wrap_obj': out[139],
      '_impl.wrap_xpos': out[140],
      'xanchor': out[141],
      'xaxis': out[142],
      'ximat': out[143],
      'xipos': out[144],
      'xmat': out[145],
      'xpos': out[146],
      'xquat': out[147],
      '_impl.contact__adhesion': out[148],
      '_impl.contact__dim': out[149],
      '_impl.contact__dist': out[150],
      '_impl.contact__efc_address': out[151],
      '_impl.contact__elem': out[152],
      '_impl.contact__flex': out[153],
      '_impl.contact__frame': out[154],
      '_impl.contact__friction': out[155],
      '_impl.contact__geom': out[156],
      '_impl.contact__geomcollisionid': out[157],
      '_impl.contact__includemargin': out[158],
      '_impl.contact__pos': out[159],
      '_impl.contact__solimp': out[160],
      '_impl.contact__solref': out[161],
      '_impl.contact__solreffriction': out[162],
      '_impl.contact__surfacevel': out[163],
      '_impl.contact__type': out[164],
      '_impl.contact__vert': out[165],
      '_impl.contact__worldid': out[166],
      '_impl.efc__D': out[167],
      '_impl.efc__J': out[168],
      '_impl.efc__J_colind': out[169],
      '_impl.efc__J_rowadr': out[170],
      '_impl.efc__J_rownnz': out[171],
      '_impl.efc__Jqvel': out[172],
      '_impl.efc__Ma': out[173],
      '_impl.efc__aref': out[174],
      '_impl.efc__force': out[175],
      '_impl.efc__frictionloss': out[176],
      '_impl.efc__id': out[177],
      '_impl.efc__island': out[178],
      '_impl.efc__jtdaj_adr': out[179],
      '_impl.efc__jtdaj_nblock': out[180],
      '_impl.efc__jtdaj_nrow': out[181],
      '_impl.efc__margin': out[182],
      '_impl.efc__pos': out[183],
      '_impl.efc__state': out[184],
      '_impl.efc__type': out[185],
      '_impl.efc__vel': out[186],
  })
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def step(m: types.Model, d: types.Data):
  return _step_jax_impl(m, d)


@step.def_vmap
@ffi.marshal_custom_vmap
def step_vmap(unused_axis_size, is_batched, m: types.Model, d: types.Data):
  d = step(m, d)
  return d, is_batched[1]
