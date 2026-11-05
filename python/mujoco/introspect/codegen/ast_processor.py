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
"""Parses MuJoCo enums, structs, and functions from a Clang AST JSON dump.

The JSON input can be generated with, e.g:
  clang -Xclang -ast-dump=json -fsyntax-only -fparse-all-comments -x c mujoco.h
"""

import itertools
import json
import os
import re
from typing import Any, Mapping, Union

from introspect import ast_nodes
from introspect import type_parsing


_ANONYMOUS_KEY_PATTERN = re.compile(r'\d+:\d+(?=\))')

_ARRAY_COMMENT_PATTERN = re.compile(r'(.+?)\s+\(([^\(\)]+) x ([^\(\)]+)\)\Z')

ClangJsonNode = Mapping[str, Any]


class _AnonymousTypePlaceholder(ast_nodes.ValueType):

  def __init__(self, anonymous_key: str):
    self.name = anonymous_key
    self.is_const = False
    self.is_volatile = False


class AstProcessor:
  """A Clang AST JSON node visitor for."""

  def __init__(self, raw_headers: Mapping[str, str], excluded: list[str]):
    self._enums = {}
    self._exported_enums = {}
    self._raw_headers = raw_headers
    self._excluded = set(excluded)
    self._current_source = ''
    self._exported_functions = {}
    self._structs = {}
    self._anonymous = {}
    self._exported_structs = {}
    self._type_aliases = {}

  @property
  def exported_enums(self) -> Mapping[str, ast_nodes.EnumDecl]:
    return self._exported_enums

  @property
  def exported_functions(self) -> Mapping[str, ast_nodes.FunctionDecl]:
    return self._exported_functions

  @property
  def exported_structs(self) -> Mapping[str, ast_nodes.StructDecl]:
    return self._exported_structs

  @property
  def type_aliases(self) -> Mapping[str, str]:
    return self._type_aliases

  def visit(self, node: ClangJsonNode) -> None:
    """Visits a JSON node and stores information about API elements."""
    if 'loc' in node and 'file' in node['loc']:
      self._current_source = os.path.basename(node['loc']['file'])

    node_kind = node.get('kind')
    node_name = node.get('name', '')

    if (
        node_kind == 'FunctionDecl'
        and node_name.startswith('mj')
        and node_name not in self._excluded
    ):
      func_decl = self._make_function(node)
      self._exported_functions[func_decl.name] = func_decl
    elif node_kind == 'EnumDecl' and node_name.startswith('mj'):
      enum_decl = self._make_enum(node)
      self._enums[enum_decl.name] = enum_decl
    elif node_kind == 'RecordDecl' and self._is_mujoco_type(node):
      struct_decl = self._make_struct(node)
      if hasattr(struct_decl, 'name'):
        self._structs[struct_decl.name] = struct_decl
      else:
        anonymous_key = self._make_anonymous_key(node)
        if anonymous_key in self._anonymous:
          raise RuntimeError(
              f'duplicate key for anonymous struct: {anonymous_key}')
        self._anonymous[anonymous_key] = struct_decl
    elif (node_kind == 'TypedefDecl' and
          node['type']['qualType'].startswith('enum mj')):
      enum = self._enums[node['type']['qualType']]
      self._exported_enums[node_name] = ast_nodes.EnumDecl(
          name=node_name, declname=enum.declname, values=dict(enum.values))
    elif (node_kind == 'TypedefDecl' and
          node['type']['qualType'].startswith('struct mj') and
          node_name not in self._excluded):
      declname = node['type']['qualType']
      try:
        struct = self._structs[declname]
      except KeyError:
        self._exported_structs[node_name] = ast_nodes.StructDecl(
            name=node_name, declname=declname, fields=())
      else:
        self._exported_structs[node_name] = ast_nodes.StructDecl(
            name=node_name, declname=struct.declname, fields=struct.fields)
    elif (node_kind == 'TypedefDecl' and node_name.startswith('mj')):
      qual_type = str(node['type']['qualType'])
      self._type_aliases[node_name] = qual_type

  def resolve(self) -> None:
    """Replaces anonymous struct placeholders with corresponding decl."""
    for struct in itertools.chain(
        self._structs.values(), self._exported_structs.values()
    ):
      fields = []
      for field in struct.fields:
        if isinstance(field, _AnonymousTypePlaceholder):
          fields.append(self._anonymous[field.name])
        elif isinstance(field.type, _AnonymousTypePlaceholder):
          fields.append(
              ast_nodes.StructFieldDecl(
                  name=field.name,
                  type=self._anonymous[field.type.name],
                  doc=field.doc,
              )
          )
        else:
          fields.append(field)
      struct.fields = tuple(fields)

  def _make_enum(self, node: ClangJsonNode) -> ast_nodes.EnumDecl:
    """Makes a EnumDecl from a Clang AST EnumDecl node."""
    name = f"enum {node['name']}"
    values = []
    for child in node['inner']:
      child_kind = child.get('kind')
      if child_kind == 'EnumConstantDecl':
        next_idx = values[-1][1] + 1 if values else 0
        if 'inner' in child:
          value = int(child['inner'][0].get('value', next_idx))
        else:
          value = next_idx
        values.append((child['name'], value))
    return ast_nodes.EnumDecl(name=name, declname=name, values=dict(values))

  def _make_function(self, node: ClangJsonNode) -> ast_nodes.FunctionDecl:
    """Makes a FunctionDecl from a Clang AST FunctionDecl node."""
    name = node['name']
    return_type = type_parsing.parse_function_return_type(
        node['type']['qualType'])
    parameters = []
    comments = []
    nullable_params = set()

    for child in node['inner']:
      child_kind = child.get('kind')
      if child_kind == 'FullComment':
        comments.append(self._make_function_comment(child))
        nullable_params.update(self._find_nullable_params(child))
    comment = ' '.join(comments).strip()

    for child in node['inner']:
      child_kind = child.get('kind')
      if child_kind == 'ParmVarDecl':
        parameters.append(self._make_parameter(child, nullable_params))

    return ast_nodes.FunctionDecl(
        name=name, return_type=return_type, parameters=parameters, doc=comment)

  def _find_nullable_params(self, node: ClangJsonNode) -> set[str]:
    """Finds the names of parameters that are marked as nullable."""
    nullable_params = set()
    for child in node['inner']:
      child_kind = child.get('kind')
      if child_kind == 'ParagraphComment':
        nullable_params.update(self._find_nullable_params(child))
      if child_kind == 'TextComment':
        if 'Nullable' in child['text']:
          for param in child['text'].split(':')[1].split(','):
            nullable_params.add(param.strip())
    return nullable_params

  def _make_parameter(
      self, node: ClangJsonNode, nullable_params: set[str]
  ) -> ast_nodes.FunctionParameterDecl:
    """Makes a ParameterDecl from a Clang AST ParmVarDecl node."""
    name = node['name']
    type_name = node['type']['qualType']
    nullable = name in nullable_params

    # For a pointer parameters, look up in the original header to see if
    # n array extent was declared there.
    if type_name.endswith('*'):
      decl_begin = node['range']['begin']['offset']
      decl_end = node['range']['end']['offset'] + node['range']['end']['tokLen']
      decl = self._raw_headers[self._current_source][decl_begin:decl_end]
      name_begin = node['loc']['offset'] - decl_begin
      name_end = name_begin + node['loc']['tokLen']
      type_name = decl[:name_begin] + decl[name_end:]

    return ast_nodes.FunctionParameterDecl(
        nullable=nullable,
        name=name,
        type=type_parsing.parse_type(type_name),
    )

  def _make_function_comment(self, node: ClangJsonNode) -> str:
    """Makes a comment string from a Clang AST FullComment node."""
    kind = node.get('kind')
    if kind == 'TextComment':
      return node['text'].replace('\N{NO-BREAK SPACE}', '&nbsp;')
    else:
      strings = []
      for child in node['inner']:
        comment = self._make_function_comment(child)
        nullable_index = comment.find('Nullable:')
        if nullable_index != -1:
          comment = comment[:nullable_index]
        strings.append(comment)
      return ''.join(strings)

  def _make_struct_comment(
      self, node: ClangJsonNode, strip: bool = True
  ) -> str:
    """Makes a comment string from a Clang AST FullComment node."""
    kind = node.get('kind')
    if kind == 'TextComment':
      retval = node['text'].replace('\N{NO-BREAK SPACE}', '&nbsp;')
    else:
      strings = []
      for child in node['inner']:
        strings.append(self._make_struct_comment(child, strip=False))
      retval = ''.join(strings)
    if strip:
      retval = retval.strip()
    return retval

  def _normalize_type(
      self, declname: str
  ) -> Union[
      ast_nodes.ValueType, ast_nodes.PointerType, ast_nodes.ArrayType]:
    """Resolves anonymous structs/unions and looks up existing typedefs."""
    # Check for anonymous struct/union.
    if '(unnamed ' in declname:
      m = _ANONYMOUS_KEY_PATTERN.search(declname)
      if not m:
        raise RuntimeError('cannot parse anonymous key from {m!r}')
      return _AnonymousTypePlaceholder(m.group(0))

    # Lookup typedef name and use it instead if one exists.
    for k, v in self._exported_structs.items():
      if declname == v.declname:
        return type_parsing.parse_type(k)

    # No valid normalization, just parse the declname.
    return type_parsing.parse_type(declname)

  def _make_field(
      self, node: ClangJsonNode
  ) -> Union[ast_nodes.StructFieldDecl, _AnonymousTypePlaceholder]:
    """Makes a StructFieldDecl object from a Clang AST FieldDecl node."""
    doc = ''
    for child in node.get('inner', ()):
      if child['kind'] == 'FullComment':
        doc = self._make_struct_comment(child)
    if 'name' in node:
      field_type = self._normalize_type(node['type']['qualType'])
      m = _ARRAY_COMMENT_PATTERN.match(doc)
      if m is None:
        array_extent = None
      else:
        doc = m.group(1)
        array_extent_0 = m.group(2)
        array_extent_1 = m.group(3)
        try:
          array_extent_1 = int(array_extent_1)
        except ValueError:
          pass
        if array_extent_1 == 1:
          array_extent = (array_extent_0,)
        else:
          array_extent = (array_extent_0, array_extent_1)
      return ast_nodes.StructFieldDecl(
          name=node['name'], type=field_type, doc=doc,
          array_extent=array_extent)
    else:
      return _AnonymousTypePlaceholder(self._make_anonymous_key(node))

  def _make_struct(
      self, node: ClangJsonNode
  ) -> Union[ast_nodes.AnonymousStructDecl, ast_nodes.StructDecl]:
    """Makes a Decl object from a Clang AST RecordDecl node."""
    name = f"{node['tagUsed']} {node['name']}" if 'name' in node else ''
    fields = []
    for child in node.get('inner', ()):
      child_kind = child.get('kind')
      if child_kind == 'FieldDecl':
        fields.append(self._make_field(child))

    if name:
      return ast_nodes.StructDecl(name=name, declname=name, fields=fields)
    elif node['tagUsed'] == 'union':
      return ast_nodes.AnonymousUnionDecl(fields=fields)
    else:
      return ast_nodes.AnonymousStructDecl(fields=fields)

  def _is_mujoco_type(self, node: ClangJsonNode) -> bool:
    node_name = node.get('name', '')
    included_from = os.path.basename(
        node['loc'].get('includedFrom', {}).get('file', '')
    )
    return node_name not in self._excluded and (
        node_name.startswith('mj')
        or included_from == 'mujoco.h'
        or included_from.startswith('mj')
    )

  def _make_anonymous_key(self, node: ClangJsonNode) -> str:
    loc = node['loc']
    if 'line' not in loc:
      # Clang elides 'line' when unchanged from the previous node. This only
      # occurs for system-header structs never referenced by MuJoCo types, so
      # the key merely needs to be unique.
      return f"offset{loc.get('offset')}:{loc.get('col')}"
    return f"{loc['line']}:{loc['col']}"


def _traverse(node, visitor):
  visitor.visit(node)

  children = node.get('inner', [])
  for child in children:
    _traverse(child, visitor)


def process(
    input_json: str,
    headers: list[str] | None = None,
    excluded: list[str] | None = None,
) -> AstProcessor:
  """Parses a Clang AST JSON file into individual AST types."""
  with open(input_json, 'r', encoding='utf-8') as f:
    root = json.load(f)

  raw_headers = {}
  for p in headers or []:
    with open(p, 'r', encoding='utf-8', newline='') as f:
      raw_headers[os.path.basename(p)] = f.read()

  visitor = AstProcessor(raw_headers, excluded or [])
  _traverse(root, visitor)
  visitor.resolve()
  return visitor
