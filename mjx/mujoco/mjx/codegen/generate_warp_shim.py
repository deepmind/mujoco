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
"""Creates a shim between JAX and Warp for a given function."""

import enum
import inspect
import re
from typing import Dict, List, Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
from mujoco.mjx._src import types as mjx_types
from mujoco.mjx.codegen import file
from mujoco.mjx.codegen import trace
from mujoco.mjx.warp import types as mjxw_types
import jax
from mujoco.mjx.third_party import mujoco_warp  # pylint: disable=unused-import


_MJWARP_FUNCTION = flags.DEFINE_string(
    'mjwarp_function',
    'third_party/py/mujoco_warp/_src/smooth.py:kinematics',
    'Function to create the shim for.',
)
_MJWARP_TYPES = flags.DEFINE_string(
    'mjwarp_types',
    'third_party/py/mujoco_warp/_src/types.py',
    'Path to the mjwarp types file.',
)
_MJX_WARP_OUTPUT_PATH = flags.DEFINE_string(
    'mjx_warp_output_path',
    'third_party/py/mujoco/mjx/warp',
    'Path to the output file.',
)
_ONLY_PUBLIC_OUTPUT_FIELDS = flags.DEFINE_bool(
    'only_public_output_fields',
    False,
    'Whether to keep only public fields in the output.',
)
_APPEND_TO_OUTPUT_FILE = flags.DEFINE_bool(
    'append_to_output_file',
    False,
    'Whether to append to the output file.',
)

_RENDER_CONTEXT_BUFFER_NAME = '_MJX_RENDER_CONTEXT_BUFFERS'


def _clean_type(type_: str):
  # check for enums
  if type_ in [
      name
      for name, obj in inspect.getmembers(mujoco_warp, inspect.isclass)
      if issubclass(obj, (enum.IntEnum, enum.IntFlag))
  ]:
    return 'int'

  types_to_prefix = (
      'vec5',
      'vec6',
      'vec6i',
      'vec8',
      'vec8i',
      'vec10',
      'vec10f',
      'vec12',
      'vec16',
      'vec16f',
      'vec16i',
      'vec128',
      'mat23',
      'mat23f',
      'mat43',
      'mat43f',
      'mat63',
      'mat63f',
      'TileSet',
      'BlockDim',
      'vec_pluginattr',
  )
  # Match and convert custom mujoco_warp array annotations (e.g. within tuple[...]).
  def _tuple_array(m):
    args = [a.strip() for a in m.group(1).split(',')]
    ndim, dtype = len(args) - 1, args[-1]
    dims = {1: '', 2: '2d', 3: '3d', 4: '4d'}
    if ndim not in dims:
      raise ValueError(f'Unsupported array ndim: {ndim} for type: {dtype}')
    return f'wp.array{dims[ndim]}[{dtype}]'

  type_ = re.sub(r'array\((.*?)\)', _tuple_array, type_)

  for t in types_to_prefix:
    type_ = re.sub(rf'\b{t}\b', f'mjwp_types.{t}', type_)
  return type_


def _get_stage_fields(
    field_usage: trace.FieldUsage,
) -> tuple[list[str], list[str]]:
  """Returns stage_in and stage_out fields after tracing.

  stage_in:
    * Model/ModelWarp jax.Array input fields
    * Data jax.Array input fields
    * Option/OptionWarp jax.Array input fields

  stage_out:
    * Data jax.Array output fields

  Args:
    field_usage: FieldUsage object

  Returns:
    A tuple of (stage_in, stage_out) field name lists.
  """
  stage_in = []
  stage_out = []

  def is_jax_array(cls, field):
    if cls is None:
      return False
    return cls.__annotations__.get(field) is jax.Array

  ModelWarp = getattr(mjxw_types, 'ModelWarp', None)
  OptionWarp = getattr(mjxw_types, 'OptionWarp', None)
  DataWarp = getattr(mjxw_types, 'DataWarp', None)

  # stage_in: Model/ModelWarp jax.Array input fields
  for field in field_usage.model_fields:
    if is_jax_array(mjx_types.Model, field) or is_jax_array(ModelWarp, field):
      stage_in.append(field)
    # stage_in: Option/OptionWarp jax.Array input fields
    elif field.startswith('opt__'):
      sub_field = field.split('opt__')[-1]
      if is_jax_array(mjx_types.Option, sub_field) or is_jax_array(OptionWarp, sub_field):
        stage_in.append(field)

  # stage_in: Data/DataWarp jax.Array input fields
  for field in field_usage.data_fields:
    if is_jax_array(mjx_types.Data, field) or is_jax_array(DataWarp, field):
      stage_in.append(field)

  # stage_out: Data/DataWarp jax.Array output fields
  for field in field_usage.data_out_fields:
    if is_jax_array(mjx_types.Data, field) or is_jax_array(DataWarp, field):
      stage_out.append(field)

  return sorted(stage_in), sorted(stage_out)


def _top_level_imports(field_usage: trace.FieldUsage):
  """Returns top-level imports."""
  imports = '''
"""DO NOT EDIT. This file is auto-generated."""
import dataclasses
import functools
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi
import mujoco.mjx.third_party.mujoco_warp as mjwarp
import warp as wp
import jax
from mujoco.mjx.third_party.mujoco_warp._src import types as mjwp_types
'''

  if field_usage.render_context_in_caller:
    imports += (
        """
from mujoco.mjx.warp.render_context import """
        + _RENDER_CONTEXT_BUFFER_NAME
        + """
from mujoco.mjx.warp.render_context import RenderContextPytree
"""
    )

  return imports


def _global_assignments():
  """Returns global assignments."""
  assignments = ''
  for attr, cls in (
      ('_m', 'mjwarp.Model'),
      ('_d', 'mjwarp.Data'),
      ('_o', 'mjwarp.Option'),
      ('_c', 'mjwarp.Contact'),
      ('_e', 'mjwarp.Constraint'),
      ('_cb', 'mjwp_types.Callback'),
  ):
    assignments += (
        f'{attr} = {cls}(**{{f.name: None for f in dataclasses.fields({cls}) if'
        ' f.init})\n'
    )
  return assignments


def _warp_function(
    fn_name: str,
    field_usage: trace.FieldUsage,
    mjwarp_field_info: Dict[str, trace.FieldInfo],
    mjx_warp_field_info: Dict[str, trace.FieldInfo],
):
  """Returns warp function arguments, assignments, and call."""
  # create warp function.
  fn_args_model, fn_assignments = [('nworld: int,', (-1, ''))], []
  if field_usage.model_fields:
    for f in field_usage.model_fields:
      if f not in mjwarp_field_info:
        raise AssertionError(f'Field {f} not found in mjwarp_field_info.')
      info = mjwarp_field_info[f]
      expected_type = _clean_type(info.expected_type)
      fn_args_model.append((f'{f}: {expected_type},', info.param_order))
      fn_assignments.append(f"  _m.{f.replace('__', '.')} = {f}")
    fn_args_model = sorted(fn_args_model, key=lambda x: x[1])
    fn_args_model = ['# Model'] + [f[0] for f in fn_args_model]

  fn_args_data = []
  if field_usage.data_fields:
    for f in field_usage.data_fields:
      if f not in mjwarp_field_info:
        raise AssertionError(f'Field {f} not found in mjwarp_field_info.')
      if f == 'nworld':
        continue  # this gets set manually below
      j = trace.FieldInfo(f, 'jax.Array', (0, ''))
      is_jax_arr = mjx_warp_field_info.get(f, j).expected_type == 'jax.Array'
      is_out = is_jax_arr
      info = mjwarp_field_info[f]
      param_order = info.param_order
      expected_type = _clean_type(info.expected_type)
      fn_args_data.append((f'{f}: {expected_type},', (is_out, param_order)))
      fn_assignments.append(f"  _d.{f.replace('__', '.')} = {f}")
    fn_args_data = sorted(fn_args_data, key=lambda x: x[1])
    fn_args_data = ['# Data'] + [f[0] for f in fn_args_data]

  fn_assignments.append('  _d.nworld = nworld')

  render_context_args = []
  render_context_call_arg = ''
  if field_usage.render_context_in_caller:
    render_context_args = ['# Registry', 'rc_id: int,']
    render_context_call_arg = ', render_context'
    fn_assignments.append(
        f'  render_context = {_RENDER_CONTEXT_BUFFER_NAME}[(rc_id, wp.get_device().ordinal)]'
    )

    if fn_name == 'render':
      render_context_args.append('rgb: wp.array2d[wp.uint32],')
      render_context_args.append('depth: wp.array2d[wp.float32],')
      render_context_args.append('seg: wp.array2d[wp.vec2i],')
      render_context_args.append('output_token: wp.array[int],')
      fn_assignments.append('  render_context.rgb_data = rgb')
      fn_assignments.append('  render_context.depth_data = depth')
      fn_assignments.append('  render_context.seg_data = seg')
      fn_assignments.append('  output_token.zero_()')
    else:
      fn_assignments.append('  output_token.zero_()')

  token_args = []
  if field_usage.render_context_in_caller:
    token_args.append('_jax_token: wp.array[int],')

  fn_call = f'mjwarp.{fn_name}(_m, _d{render_context_call_arg})'
  fn_args_raw = fn_args_model + fn_args_data + token_args + render_context_args

  # create an output token if there are no output fields
  needs_output_token = not field_usage.data_out_fields
  if needs_output_token and fn_name != 'render':
    fn_args_raw.append('# Output token')  # pyrefly: ignore[bad-argument-type]
    fn_args_raw.append('output_token: wp.array[int],')  # pyrefly: ignore[bad-argument-type]

  return fn_args_raw, fn_assignments, fn_call


def _jax_shim_fn(
    fn_name: str,
    field_usage: trace.FieldUsage,
    warp_fn_args: List[str],
    mjwarp_field_info: Dict[str, trace.FieldInfo],
):
  """Generates a JAX shim for the Warp function."""
  num_outputs = 0
  output_dims = []
  jax_args = []
  tree_replace = []
  in_out_argnames = []
  has_side_effect = False
  stage_in_fields, stage_out_fields = _get_stage_fields(field_usage)
  stage_in_argnames = [f"'{f}'" for f in stage_in_fields]
  stage_out_argnames = [f"'{f}'" for f in stage_out_fields]

  for arg in warp_fn_args:
    if 'nworld' in arg:
      if field_usage.render_context_in_caller:
        jax_args.append('render_ctx.nworld')
      else:
        jax_args.append('d.qpos.shape[0]')
      continue

    if arg in ('rc_id', 'output_token', '_jax_token'):
      continue

    if arg in ('rgb', 'depth', 'seg') and fn_name == 'render':
      num_outputs += 1
      continue

    arg_jax = arg
    if mjwarp_field_info[arg].param_source == 'Data':
      if arg.split('__')[0] not in mjx_types.Data.__annotations__:
        arg_jax = f'_impl.{arg}'
      arg_jax = 'd.' + arg_jax
    elif mjwarp_field_info[arg].param_source == 'Model':
      arg_jax = (
          arg.replace('__', '.')
          if arg.startswith('opt')
          else arg
      )
      public_field = arg.split('__')[0] in mjx_types.Model.__annotations__
      if not public_field:
        arg_jax = f'_impl.{arg_jax}'
      if (
          arg.startswith('opt')
          and arg.split('__')[-1] not in mjx_types.Option.__annotations__
      ):
        arg_jax = arg_jax.replace('opt', 'opt._impl')
      arg_jax = 'm.' + arg_jax
    else:
      raise ValueError(
          f'Unknown param source: {mjwarp_field_info[arg].param_source}'
      )

    # Only treat as output if it is a Warp array (excludes scalars like
    # naconmax).
    if (
        arg in field_usage.data_out_fields
        and 'array' in mjwarp_field_info[arg].expected_type
    ):
      # all out fields are in_out, since JAX already allocated them
      in_out_argnames.append(f"'{arg}'")
      num_outputs += 1
      output_dims.append(f"'{arg}': {arg_jax}.shape")

      if '_impl' not in arg_jax or not _ONLY_PUBLIC_OUTPUT_FIELDS.value:
        tree_replace.append(f'"{arg_jax[2:]}": out[{num_outputs - 1}]')

    if arg == 'geom_dataid':
      jax_args.append(f'jax.numpy.expand_dims({arg_jax}, 0)')
    else:
      jax_args.append(arg_jax)

  if field_usage.render_context_in_caller:
    jax_args.append('d._impl._jax_token')
    jax_args.append('ctx.key')

  # If there are no Warp array outputs, we need an output token for JAX FFI.
  needs_output_token = not any(
      'array' in mjwarp_field_info[f].expected_type
      for f in field_usage.data_out_fields
  )
  if needs_output_token and fn_name != 'render':
    num_outputs = 1
    output_dims = ["'output_token': (d.qpos.shape[0],)"]
    if field_usage.render_context_in_caller:
      tree_replace = ['"_impl._jax_token": out[0]']
    has_side_effect = True

  if fn_name == 'render':
    num_outputs = 4
    output_dims = [
        "'rgb': render_ctx.rgb_data_shape",
        "'depth': render_ctx.depth_data_shape",
        "'seg': render_ctx.seg_data_shape",
        "'output_token': (d.qpos.shape[0],)",
    ]
    tree_replace = []
    has_side_effect = True

  render_ctx_param = (
      'ctx: RenderContextPytree' if field_usage.render_context_in_caller else ''
  )
  fn_args = ['m: types.Model', 'd: types.Data']

  if render_ctx_param:
    fn_args.append(render_ctx_param)

  return (
      fn_args,
      jax_args,
      output_dims,
      num_outputs,
      tree_replace,
      in_out_argnames,
      stage_in_argnames,
      stage_out_argnames,
      has_side_effect,
  )


def create_jax_warp_shim(
    fn_name: str,
    field_usage: trace.FieldUsage,
    mjwarp_field_info: Dict[str, trace.FieldInfo],
    mjx_warp_field_info: Dict[str, trace.FieldInfo],
    out_fpath: epath.Path,
):
  """Creates a JAX-wrapped MJWarp function."""
  src = ''
  old_src = (
      out_fpath.read_text()
      if out_fpath.exists()
      else ''
      if out_fpath.exists()
      else ''
  )

  # create top-level imports.
  if not _APPEND_TO_OUTPUT_FILE.value:
    src += _top_level_imports(field_usage) + '\n\n'

  # create global assignments.
  assignments = _global_assignments()
  already_in_src = re.sub(r'\s+', '', assignments) in re.sub(
      r'\s+', '', old_src
  )
  if not already_in_src or not _APPEND_TO_OUTPUT_FILE.value:
    src += assignments

  # create warp function.
  fn_args_raw, fn_assignments, fn_call = _warp_function(
      fn_name, field_usage, mjwarp_field_info, mjx_warp_field_info
  )
  fn_args_raw_str = '\n'.join(['    ' + arg for arg in fn_args_raw])
  warp_fn_args = [arg.split(':')[0] for arg in fn_args_raw if '#' not in arg]

  fn_assignments_str = '\n'.join(fn_assignments)
  src += f"""
@ffi.format_args_for_warp
def _{fn_name}_shim(
{fn_args_raw_str}
):
  _m.opt = _o
  _m.callback = _cb
  _d.efc = _e
  _d.contact = _c
{fn_assignments_str}
  {fn_call}
  """
  src += '\n\n'

  # create private jax function.
  (
      fn_args,
      jax_args,
      output_dims,
      num_outputs,
      tree_replace,
      in_out_argnames,
      stage_in_argnames,
      stage_out_argnames,
      has_side_effect,
  ) = _jax_shim_fn(fn_name, field_usage, warp_fn_args, mjwarp_field_info)
  render_ctx_line = ''
  return_stmt = 'return d'
  if field_usage.render_context_in_caller:
    render_ctx_line = f'  render_ctx = _MJX_RENDER_CONTEXT_BUFFERS[(ctx.key, None)]\n'
  if fn_name == 'render':
    return_stmt = 'return out'
  output_dims_str = '{' + ','.join(output_dims) + '}'
  data_tree_replace = f"d = d.tree_replace({{ {','.join(tree_replace)} }})"
  src += f"""
def _{fn_name}_jax_impl({','.join(fn_args)}):
{render_ctx_line}  output_dims = {output_dims_str}
  jf = ffi.jax_callable_variadic_tuple(
      _{fn_name}_shim, num_outputs={num_outputs},
      output_dims=output_dims,
      vmap_method=None,
      in_out_argnames=set([{','.join(in_out_argnames)}]),
      stage_in_argnames=set([{','.join(stage_in_argnames)}]),
      stage_out_argnames=set([{','.join(stage_out_argnames)}]),
      graph_mode=m.opt._impl.graph_mode,
      has_side_effect={has_side_effect},
  )
  out = jf({','.join(jax_args)})
  {data_tree_replace}
  {return_stmt}
"""
  src += '\n'

  # create public jax functions.
  fn_args_no_annotation = [arg.split(':')[0] for arg in fn_args]
  fn_call_str = ','.join(fn_args_no_annotation)

  marshal_decorator = '@ffi.marshal_jax_warp_callable'
  marshal_vmap_decorator = '@ffi.marshal_custom_vmap'
  vmap_return_stmt = f'd = {fn_name}({fn_call_str})\n  return d, is_batched[1]'
  if fn_name == 'render':
    marshal_decorator = (
        '@functools.partial('
        'ffi.marshal_jax_warp_callable, tree_map_output=True)'
    )
    marshal_vmap_decorator = (
        '@functools.partial(ffi.marshal_custom_vmap, tree_map_output=True)'
    )
    vmap_return_stmt = (
        f'out = {fn_name}({fn_call_str})\n  return out, [True, True, True,'
        ' is_batched[1]._impl._jax_token]'
    )

  src += f"""
@jax.custom_batching.custom_vmap
{marshal_decorator}
def {fn_name}({','.join(fn_args)}):
  return _{fn_name}_jax_impl({','.join(fn_args_no_annotation)})
@{fn_name}.def_vmap
{marshal_vmap_decorator}
def {fn_name}_vmap(unused_axis_size, is_batched, {','.join(fn_args)}):
  {vmap_return_stmt}
"""
  src += '\n'

  if _APPEND_TO_OUTPUT_FILE.value:
    src = old_src + '\n\n' + src

  out_fpath.write_text(src)


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.DEBUG)

  # Get mjwarp field annotations.
  fpath = epath.Path(_MJWARP_TYPES.value)
  mjwarp_field_info = trace.get_mjwarp_field_info(
      fpath.read_text(), file.get_cls_type_annotations
  )

  # Trace function to get field usage.
  fpath, fn_name = _MJWARP_FUNCTION.value.split(':')
  field_usage = trace.trace_function(fpath, fn_name, mjwarp_field_info)

  base_path = file.get_base_path()
  types_fpath = base_path / _MJX_WARP_OUTPUT_PATH.value / 'types.py'
  mjx_warp_field_info = trace.get_mjx_warp_field_info(
      types_fpath.read_text(), file.get_cls_type_annotations
  )

  base_path = file.get_base_path()
  target_fpath = (
      base_path / _MJX_WARP_OUTPUT_PATH.value / epath.Path(fpath).name
  )
  create_jax_warp_shim(
      fn_name, field_usage, mjwarp_field_info, mjx_warp_field_info, target_fpath
  )

  if not _APPEND_TO_OUTPUT_FILE.value:
    file.write_license(target_fpath)
  file.format_file(target_fpath)


if __name__ == '__main__':
  app.run(main)
