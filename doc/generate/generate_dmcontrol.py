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
"""Generates dm_control's schema.xml dialect from mjcf.schema.

dm_control's PyMJCF library builds its element classes from a hand-maintained
schema.xml; this emitter replaces that file with a generated one so the MJCF
facts live in one place. The element surface is deliberately frozen to what
dm_control supports today: new MJCF elements are listed in EXCLUDED_ELEMENTS
and not emitted, while attributes of supported elements are emitted in full
from the schema, so keyword sets, arities and defaults can no longer drift.

The dialect is dm_control's: a nested element tree (one node per context,
recursion expressed with recursive="true"), attribute types keyword / array /
int / float / string / identifier / reference / basepath / file, and element
flags repeated / on_demand / namespace. Facts PyMJCF needs that mjcf.schema
deliberately does not model -- attach-merge conflict policy, on-demand
construction, and dm_control's namespace grouping -- are carried as overlay
tables below, harvested once from the hand-maintained file.

The generated file is checked in as src/xml/generated/dmcontrol_schema.xml
and gated by test/doc/doc_test.py; dm_control vendors it at pin bumps.
"""

import os
import sys
from xml.sax import saxutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
  import resource_loader  # pyrefly: ignore[missing-import]
  import generate_xsd  # pyrefly: ignore[missing-import]
  import mjcf_schema  # pyrefly: ignore[missing-import]
except ImportError:
  raise

SCHEMA_PATH = str(resource_loader.resolve_path('src/xml/mjcf.schema'))

# elements dm_control does not support; the surface is frozen, so new MJCF
# elements are added here rather than emitted
EXCLUDED_ELEMENTS = {
    'pid', 'dcmotor', 'replicate', 'frame', 'attach', 'model',
}
# (parent element, child element) pairs excluded in that context only
EXCLUDED_CHILDREN = {('worldbody', 'plugin')}

# PyMJCF attach-merge policy: (element, attribute) -> conflict_behavior
# (None = conflict_allowed with no behavior). Harvested from dm_control.
CONFLICTS = {
    ('compiler', 'assetdir'): None,
    ('compiler', 'meshdir'): None,
    ('compiler', 'texturedir'): None,
    ('map', 'zfar'): 'max',
    ('map', 'znear'): 'min',
    ('mujoco', 'model'): None,
    ('option', 'ccd_iterations'): 'max',
    ('option', 'ccd_tolerance'): 'min',
    ('option', 'iterations'): 'max',
    ('option', 'ls_iterations'): 'max',
    ('option', 'ls_tolerance'): 'min',
    ('option', 'noslip_iterations'): 'max',
    ('option', 'noslip_tolerance'): 'min',
    ('option', 'sdf_initpoints'): 'max',
    ('option', 'sdf_iterations'): 'max',
    ('option', 'timestep'): 'min',
    ('option', 'tolerance'): 'min',
    ('size', 'memory'): 'max_bytes',
    ('size', 'nconmax'): 'max',
    ('size', 'njmax'): 'max',
}

# (parent, tag) pairs PyMJCF exposes as singletons even where MJCF admits
# repetition (repeated sections merge at load): the flag is API surface
# (root.option.timestep vs a list view), harvested from the hand file
SINGLETONS = {
    ('body', 'freejoint'), ('body', 'inertial'),
    ('compiler', 'lengthrange'),
    ('default', 'adhesion'), ('default', 'camera'), ('default', 'cylinder'),
    ('default', 'damper'), ('default', 'equality'), ('default', 'general'),
    ('default', 'geom'), ('default', 'intvelocity'), ('default', 'joint'),
    ('default', 'light'), ('default', 'material'), ('default', 'mesh'),
    ('default', 'motor'), ('default', 'muscle'), ('default', 'orientation'),
    ('default', 'pair'), ('default', 'position'), ('default', 'site'),
    ('default', 'tendon'), ('default', 'velocity'),
    ('mujoco', 'actuator'), ('mujoco', 'asset'), ('mujoco', 'compiler'),
    ('mujoco', 'contact'), ('mujoco', 'custom'), ('mujoco', 'default'),
    ('mujoco', 'deformable'), ('mujoco', 'equality'),
    ('mujoco', 'extension'), ('mujoco', 'keyframe'), ('mujoco', 'option'),
    ('mujoco', 'sensor'), ('mujoco', 'size'), ('mujoco', 'statistic'),
    ('mujoco', 'tendon'), ('mujoco', 'visual'), ('mujoco', 'worldbody'),
    ('option', 'flag'),
    ('visual', 'global'), ('visual', 'headlight'), ('visual', 'map'),
    ('visual', 'quality'), ('visual', 'rgba'), ('visual', 'scale'),
}

# elements PyMJCF constructs on demand rather than eagerly
ON_DEMAND = {'inertial', 'freejoint'}

# dm_control groups some identifier namespaces more coarsely than the
# schema's id<ns> targets; keyed by element name
NAMESPACE_OVERRIDES = {
    'exclude': 'contact',
    'flex': 'deformable',
    'flexcomp': 'flexcomp',
    'instance': 'plugin',
    'pair': 'contact',
}
# per-context namespaces: the same schema element namescopes differently by
# parent (dm_control's asset skin lives in 'skin', deformable skin in
# 'deformable')
CONTEXT_NAMESPACE = {('deformable', 'skin'): 'deformable'}
# attributes that are PyMJCF identifiers although the schema types them as
# plain strings
IDENTIFIER_OVERRIDES: set[tuple[str, str]] = set()
# the same grouping applied to reference targets
REF_NS_MAP = {'instance': 'plugin', 'flex': 'deformable',
              'skin': 'deformable', 'pair': 'contact', 'exclude': 'contact'}

# compiler directory attributes are PyMJCF basepaths
BASEPATHS = {'meshdir': 'mesh', 'texturedir': 'texture', 'assetdir': 'asset'}
# path namespace for file-typed attributes, by owning element. In MJCF, meshdir
# is the base directory for meshes, heightfields, and skins.
FILE_NS = {'mesh': 'mesh', 'texture': 'texture', 'hfield': 'mesh',
           'skin': 'mesh'}


def _format_number(value):
  """Format a numeric default the shortest exact way: 2.0 -> '2'."""
  if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
    return str(int(value))
  return repr(value)


def _fmt_default(attr):
  """Format a declared default the way dm_control's file writes values."""
  d = attr.default
  if d is None:
    return None
  if isinstance(d, tuple):
    return ' '.join(_format_number(v) for v in d)
  if isinstance(d, (int, float)):
    return _format_number(d)
  return str(d)


class _Emitter:
  """Emits the dm_control schema tree by walking mjcf.schema from mujoco."""

  def __init__(self, schema):
    self.schema = schema
    self.dims = generate_xsd.parse_dims()
    self.lines = []
    self.ref_namespaces = set()   # (element, attr, ns) of emitted references
    self.id_namespaces = set()    # namespaces populated by emitted elements

  def out(self, indent, text):
    self.lines.append(' ' * indent + text)

  def resolve(self, bound):
    if isinstance(bound, str):
      return self.dims[bound]
    return bound

  def element_namespace(self, element, parent):
    """Returns the dm_control namespace for an element.

    Any identifier-typed attribute makes the element identified (their
    parser accepts any; default's identifier is 'class', not 'name').

    Args:
      element: The mjcf.schema element to inspect.
      parent: Optional name of the parent element context.

    Returns:
      The string namespace name, or None.
    """
    if (parent, element.name) in CONTEXT_NAMESPACE:
      return CONTEXT_NAMESPACE[(parent, element.name)]
    if element.name in NAMESPACE_OVERRIDES:
      return NAMESPACE_OVERRIDES[element.name]
    for a in self.schema.expanded_attrs(element):
      if a.type == 'id' or (element.name, a.name) in IDENTIFIER_OVERRIDES:
        return a.target if a.type == 'id' else element.name
    return None

  def attr_parts(self, element, attr, attr_names):
    """Returns the type-describing XML attribute string for one attribute.

    Args:
      element: The owning element.
      attr: The mjcf.schema attribute.
      attr_names: Set of attribute names for the element.

    Returns:
      A formatted XML attribute string.
    """
    lo, hi = attr.arity.lo, self.resolve(attr.arity.hi)
    # sensor-style object references are typed by a sibling attribute; the
    # schema leaves these as strings (the target kind is value-dependent),
    # but PyMJCF's attachment namespacing needs the dynamic reference
    if attr.name == 'objname' and 'objtype' in attr_names:
      return 'type="reference" reference_namespace="attrib:objtype"'
    if attr.name == 'refname' and 'reftype' in attr_names:
      return 'type="reference" reference_namespace="attrib:reftype"'
    if (element.name, attr.name) in IDENTIFIER_OVERRIDES:
      return 'type="identifier"'
    if element.name == 'mujoco' and attr.name == 'model':
      return 'type="string"'  # the model's own name, not a reference
    if attr.name in BASEPATHS and element.name == 'compiler':
      return f'type="basepath" path_namespace="{BASEPATHS[attr.name]}"'
    if attr.type == 'file':
      ns = FILE_NS.get(element.name)
      ns_part = f' path_namespace="{ns}"' if ns else ''
      return f'type="file"{ns_part}'
    if attr.type == 'enum':
      keywords = ' '.join(k for k, _ in self.schema.enums[attr.target].items)
      return f'type="keyword" valid_values={saxutils.quoteattr(keywords)}'
    if attr.type == 'bool':
      return 'type="keyword" valid_values="false true"'
    if attr.type == 'id':
      return 'type="identifier"'
    if attr.type == 'ref':
      ns = REF_NS_MAP.get(attr.target, attr.target)
      self.ref_namespaces.add((element.name, attr.name, ns))
      return f'type="reference" reference_namespace="{ns}"'
    if attr.type in ('string', 'chars', 'flags'):
      # flags (multi-keyword) attributes are opaque strings to PyMJCF
      return 'type="string"'
    if attr.type in ('double', 'float', 'int'):
      base = 'int' if attr.type == 'int' else 'float'
      if (lo, hi) == (1, 1):
        return f'type="{base}"'
      size = f' array_size="{hi}"' if hi is not None else ''
      return f'type="array" array_type="{base}"{size}'
    raise ValueError(f'{element.name}.{attr.name}: unmapped type {attr.type}')

  def emit_attr(self, indent, element, attr, attr_names):
    """Emits a single <attribute .../> XML element.

    Args:
      indent: Indentation level in spaces.
      element: The owning element.
      attr: The mjcf.schema attribute.
      attr_names: Set of attribute names for the element.
    """
    parts = [f'<attribute name="{attr.name}"',
             self.attr_parts(element, attr, attr_names)]
    if attr.facets.get('required'):
      parts.append('required="true"')
    default = _fmt_default(attr)
    if default is not None:
      parts.append(f'default={saxutils.quoteattr(default)}')
    behavior = CONFLICTS.get((element.name, attr.name), 'ABSENT')
    if behavior != 'ABSENT':
      parts.append('conflict_allowed="true"')
      if behavior is not None:
        parts.append(f'conflict_behavior="{behavior}"')
    self.out(indent, ' '.join(parts) + '/>')

  def emit_element(self, element, tag, card, projected, indent, ancestry,
                   parent=None):
    """Emits one element node.

    Self-recursion is expressed dm_control's way: the full definition
    carries recursive="true" and lists no self-child -- their parser links
    children[name] back to the element itself. Non-self cycles cannot occur
    on the emitted surface (frame and replicate are excluded), which the
    ancestry assertion enforces.

    Args:
      element: The mjcf.schema element to emit.
      tag: XML element tag name.
      card: Cardinality string ('!', '?', '*', 'R').
      projected: Whether defaults are projected.
      indent: Current indentation level in spaces.
      ancestry: Set of parent element names in current stack.
      parent: Optional name of parent element.
    """
    assert (
        element.name not in ancestry
        or (element.name == 'default' and parent == 'default')
    ), f'unexpected cycle at {element.name}'
    self_recursive = any(c.name == element.name for c in element.children())
    # the top-level default is a singleton in PyMJCF (root.default.geom...)
    # while nested defaults are repeated: one recursive spec cannot carry
    # both flags, so the top node is emitted unflagged with an explicit
    # nested recursive+repeated copy, exactly dm_control's historical shape
    top_default = element.name == 'default' and parent == 'mujoco'
    parts = [f'<element name="{tag}"']
    if self_recursive and not top_default:
      parts.append('recursive="true"')
    if (card in ('*', 'R') and not top_default and
        (parent, tag) not in SINGLETONS):
      parts.append('repeated="true"')
    if element.name in ON_DEMAND:
      parts.append('on_demand="true"')
    ns = self.element_namespace(element, parent)
    if ns:
      self.id_namespaces.add(ns)
    if ns and ns != tag:
      # the parser defaults an identified element's namespace to its name
      parts.append(f'namespace="{ns}"')
    head = ' '.join(parts) + '>'
    self.out(indent, head)

    attrs = self.schema.expanded_attrs(element)
    if projected:
      attrs = [a for a in attrs if a.name not in ('name', 'class')
               and not a.facets.get('nodefault')]
    attr_names = {a.name for a in attrs}
    if attrs:
      self.out(indent + 2, '<attributes>')
      for attr in attrs:
        self.emit_attr(indent + 4, element, attr, attr_names)
      self.out(indent + 2, '</attributes>')

    children = []
    for child in element.children():
      if child.name == element.name:
        if top_default:
          # the nested recursive+repeated copy of the default subtree
          children.append((element, tag, child.card, projected))
        continue  # otherwise self-recursion is the recursive flag
      target = self.schema.elements[child.name]
      child_tag = target.xml_name()
      if element.name == 'mujoco' and child.name == 'body':
        target, child_tag = self.schema.elements['worldbody'], 'worldbody'
      if (target.name in EXCLUDED_ELEMENTS or
          (element.name, target.name) in EXCLUDED_CHILDREN):
        continue
      child_projected = (projected or
                         (element.name == 'default' and
                          not child.name.startswith('default_') and
                          child.name != 'default'))
      if projected and child.name == 'plugin':
        continue
      children.append((target, child_tag, child.card, child_projected))
    if children:
      self.out(indent + 2, '<children>')
      for target, child_tag, card_, child_projected in children:
        self.emit_element(target, child_tag, card_, child_projected,
                          indent + 4, ancestry | {element.name},
                          parent=element.name)
      self.out(indent + 2, '</children>')
    self.out(indent, '</element>')

  def generate(self):
    """Generate the full schema XML as a string."""
    self.out(0, '<!-- Generated by generate_dmcontrol.py from mjcf.schema;')
    self.out(0, '     do not edit by hand. The element surface is frozen to')
    self.out(0, '     what dm_control supports (see EXCLUDED_ELEMENTS);')
    self.out(0, '     attribute facts follow mjcf.schema. -->')
    self.emit_element(self.schema.elements['mujoco'], 'mujoco', '!',
                      False, 0, frozenset())
    dangling = {r for r in self.ref_namespaces
                if r[2] not in self.id_namespaces}
    if dangling:
      raise ValueError('references into namespaces no emitted element '
                       f'populates: {sorted(dangling)}')
    return '\n'.join(self.lines) + '\n'


def generate():
  """Generate the dmcontrol_schema.xml content as a string."""
  schema = mjcf_schema.parse_file(SCHEMA_PATH)
  return _Emitter(schema).generate()


def main():
  """CLI entry point: generate dmcontrol_schema.xml to stdout or a file."""
  if len(sys.argv) > 2:
    sys.exit('usage: generate_dmcontrol.py [output.xml]')
  text = generate()
  if len(sys.argv) == 2:
    with open(sys.argv[1], 'w', encoding='utf-8') as f:
      f.write(text)
  else:
    sys.stdout.write(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
