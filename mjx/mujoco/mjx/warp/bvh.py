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
from mujoco.mjx.warp.render_context import _MJX_RENDER_CONTEXT_BUFFERS
from mujoco.mjx.warp.render_context import RenderContextPytree
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
def _refit_bvh_shim(
    # Model
    nworld: int,
    flex_dim: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_elem: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_radius: wp.array[float],
    flex_shell: wp.array[int],
    flex_shelldataadr: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertnum: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_type: wp.array[int],
    nflex: int,
    nflexelem: int,
    opt__deterministic: int,
    # Data
    flexvert_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    geom_xpos: wp.array2d[wp.vec3],
    _jax_token: wp.array[int],
    # Registry
    rc_id: int,
    # Output token
    output_token: wp.array[int],
):
  _m.opt = _o
  _m.callback = _cb
  _d.efc = _e
  _d.contact = _c
  _m.flex_dim = flex_dim
  _m.flex_edge = flex_edge
  _m.flex_elem = flex_elem
  _m.flex_elemadr = flex_elemadr
  _m.flex_elemdataadr = flex_elemdataadr
  _m.flex_elemnum = flex_elemnum
  _m.flex_radius = flex_radius
  _m.flex_shell = flex_shell
  _m.flex_shelldataadr = flex_shelldataadr
  _m.flex_vertadr = flex_vertadr
  _m.flex_vertnum = flex_vertnum
  _m.geom_dataid = geom_dataid
  _m.geom_size = geom_size
  _m.geom_type = geom_type
  _m.nflex = nflex
  _m.nflexelem = nflexelem
  _m.opt.deterministic = opt__deterministic
  _d.flexvert_xpos = flexvert_xpos
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.nworld = nworld
  render_context = _MJX_RENDER_CONTEXT_BUFFERS[(rc_id, wp.get_device().ordinal)]
  output_token.zero_()
  mjwarp.refit_bvh(_m, _d, render_context)


def _refit_bvh_jax_impl(
    m: types.Model, d: types.Data, ctx: RenderContextPytree
):
  render_ctx = _MJX_RENDER_CONTEXT_BUFFERS[(ctx.key, None)]
  output_dims = {'output_token': (d.qpos.shape[0],)}
  jf = ffi.jax_callable_variadic_tuple(
      _refit_bvh_shim,
      num_outputs=1,
      output_dims=output_dims,
      vmap_method=None,
      in_out_argnames=set([]),
      stage_in_argnames=set(
          ['flexvert_xpos', 'geom_size', 'geom_xmat', 'geom_xpos']
      ),
      stage_out_argnames=set([]),
      graph_mode=m.opt._impl.graph_mode,
      has_side_effect=True,
  )
  out = jf(
      render_ctx.nworld,
      m._impl.flex_dim,
      m._impl.flex_edge,
      m._impl.flex_elem,
      m._impl.flex_elemadr,
      m._impl.flex_elemdataadr,
      m._impl.flex_elemnum,
      m._impl.flex_radius,
      m._impl.flex_shell,
      m._impl.flex_shelldataadr,
      m.flex_vertadr,
      m.flex_vertnum,
      jax.numpy.expand_dims(m.geom_dataid, 0),
      m.geom_size,
      m.geom_type,
      m.nflex,
      m._impl.nflexelem,
      m.opt._impl.deterministic,
      d._impl.flexvert_xpos,
      d.geom_xmat,
      d.geom_xpos,
      d._impl._jax_token,
      ctx.key,
  )
  d = d.tree_replace({'_impl._jax_token': out[0]})
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def refit_bvh(m: types.Model, d: types.Data, ctx: RenderContextPytree):
  return _refit_bvh_jax_impl(m, d, ctx)


@refit_bvh.def_vmap
@ffi.marshal_custom_vmap
def refit_bvh_vmap(
    unused_axis_size,
    is_batched,
    m: types.Model,
    d: types.Data,
    ctx: RenderContextPytree,
):
  d = refit_bvh(m, d, ctx)
  return d, is_batched[1]
