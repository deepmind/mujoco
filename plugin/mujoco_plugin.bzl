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

"""Builds MuJoCo plugins as shared libraries."""

load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")

_MUJOCO_SHARED = Label("//:mujoco_shared")
_LDS = Label("//:mujoco_plugin.lds")
_EXPORTED_SYMBOLS = Label("//:mujoco_plugin.exported_symbols")
_MACOS = Label("@platforms//os:macos")

def _mujoco_plugin_impl(name, visibility, deps, library_name, **kwargs):
    library_name = library_name or "lib" + name
    cc_shared_library(
        name = name,
        additional_linker_inputs = select({
            _MACOS: [_EXPORTED_SYMBOLS],
            "//conditions:default": [_LDS],
        }),
        dynamic_deps = [_MUJOCO_SHARED],
        shared_lib_name = select({
            _MACOS: library_name + ".dylib",
            "//conditions:default": library_name + ".so",
        }),
        user_link_flags = select({
            _MACOS: ["-Wl,-exported_symbols_list,$(execpath %s)" % _EXPORTED_SYMBOLS],
            "//conditions:default": [
                "-Wl,--version-script=$(execpath %s)" % _LDS,
            ],
        }),
        visibility = visibility,
        deps = deps,
        **kwargs
    )

mujoco_plugin = macro(
    doc = """Builds a MuJoCo plugin as a shared library, like the bundled plugins.

The library links MuJoCo dynamically from `//:mujoco_shared` and exports no
symbols: plugins register themselves with `mjPLUGIN_LIB_INIT` when the library
is loaded, e.g. with `mj_loadPluginLibrary`. Not supported on Windows.

Example:

    cc_library(
        name = "my_plugin_lib",
        srcs = ["my_plugin.cc"],
        deps = ["@mujoco"],
        alwayslink = True,
    )

    mujoco_plugin(
        name = "my_plugin",
        deps = [":my_plugin_lib"],
    )
""",
    implementation = _mujoco_plugin_impl,
    attrs = {
        "deps": attr.label_list(
            mandatory = True,
            doc = "The plugin's libraries. They should be `alwayslink`, so " +
                  "that the registration code is linked in.",
        ),
        "library_name": attr.string(
            configurable = False,
            doc = "The library's file name without the extension, which " +
                  "depends on the platform (`.so` or `.dylib`). Defaults to " +
                  "`lib<name>`. May include a directory, e.g. `plugin/libfoo`.",
        ),
    },
    inherit_attrs = "common",
)
