# Copyright 2025 DeepMind Technologies Limited
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

"""Helper class to build code string line by line with indentation."""

INDENT = "  "


class CodeBuilder:
  """Helper class to build code string line by line with indentation."""

  def __init__(self, indent_str: str = INDENT):
    self._lines = []
    self._indent_level = 0
    self._indent_str = indent_str

  def license(self, year: int, comment: str = "//"):
    """Adds the standard MuJoCo license to the code."""
    # pylint: disable=line-too-long
    self.line(f"{comment} Copyright {year} DeepMind Technologies Limited")
    self.line(f"{comment}")
    self.line(f"{comment} Licensed under the Apache License, Version 2.0 (the \"License\");")
    self.line(f"{comment} you may not use this file except in compliance with the License.")
    self.line(f"{comment} You may obtain a copy of the License at")
    self.line(f"{comment}")
    self.line(f"{comment}     http://www.apache.org/licenses/LICENSE-2.0")
    self.line(f"{comment}")
    self.line(f"{comment} Unless required by applicable law or agreed to in writing, software")
    self.line(f"{comment} distributed under the License is distributed on an \"AS IS\" BASIS,")
    self.line(f"{comment} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.")
    self.line(f"{comment} See the License for the specific language governing permissions and")
    self.line(f"{comment} limitations under the License.")
    self.newline()
    # pylint: enable=line-too-long

  def line(self, line_content: str) -> None:
    """Adds a line with indentation, special-casing "private:" and "public:"."""
    indent = self._indent_str * self._indent_level
    content = line_content.strip()
    if content == "private:" or content == "public:":
      self._lines.append(indent[:-1] + line_content)
    elif content:
      self._lines.append(indent + line_content)
    else:
      self._lines.append("")

  def newline(self) -> None:
    """Adds a newline."""
    self.line("")

  def to_string(self) -> str:
    """Returns the complete code string."""
    return "\n".join(self._lines)

  class IndentBlock:
    """Helper class to manage indentation within a `with` statement."""

    def __init__(
        self,
        builder: "CodeBuilder",
        header_line="",
        footer_line="",
        braces=True,
    ):
      self._builder = builder
      self._header_line = header_line
      self._footer_line = footer_line
      self._braces = braces

    def __enter__(self):
      line = self._header_line
      if self._braces:
        line += " {" if line else "{"
      if line:
        self._builder.line(line)
      self._builder._indent_level += 1
      self._line_count_enter = len(self._builder._lines)
      return self._builder

    def __exit__(self, exc_type, exc_val, exc_tb):
      if self._builder._indent_level > 0:
        self._builder._indent_level -= 1
      if self._line_count_enter == len(self._builder._lines):
        self._builder._lines[-1] += "}"
      else:
        if self._braces:
          self._builder.line("}" + self._footer_line)

  def block(
      self,
      header_line="",
      footer_line="",
      *,
      braces=True,
  ) -> IndentBlock:
    """Creates a block including braces and an optional headers/footers.

    Use via a `with` statement.

    Args:
        header_line: Optional header line to add before the opening brace.
        footer_line: Optional footer line to add after the closing brace.
        braces: Whether to include opening and closing braces.

    Returns:
        An IndentBlock instance that manages the indentation.
    """
    return self.IndentBlock(self, header_line, footer_line, braces)

  def function(self, signature="") -> IndentBlock:
    """Creates a function."""
    return self.block(signature)

  def struct(self, name="") -> IndentBlock:
    """Creates a struct."""
    return self.block(f"struct {name}")

  def private(self) -> None:
    """Creates a private section."""
    self.newline()
    self.line("private:")

  def public(self) -> None:
    """Creates a public section."""
    self.newline()
    self.line("public:")
