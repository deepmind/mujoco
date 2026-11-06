# MuJoCo Core Python Bindings (`mujoco-core`)

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/mujoco-core
[pypi-badge]: https://badge.fury.io/py/mujoco-core.svg
[pypi]: https://pypi.org/project/mujoco-core/

This package contains the core Python bindings (`mujoco-core`) for the
[MuJoCo physics engine](https://github.com/google-deepmind/mujoco).
These bindings are developed and maintained by Google DeepMind, and are kept
up-to-date with the latest developments in MuJoCo itself.

The `mujoco-core` package provides direct access to raw MuJoCo C API functions,
structs, constants, and enumerations, with only `numpy` as a required runtime
dependency. It is automatically installed when you install the full `mujoco`
package (`pip install mujoco`), or can be installed standalone in minimal
headless simulation environments (`pip install mujoco-core`). To switch an
existing `mujoco` installation to `mujoco-core` only, run:

```sh
pip uninstall -y mujoco mujoco-studio && pip install --force-reinstall --no-deps mujoco-core
```

It is not the aim of this package to provide fully fledged
scene/environment/game authoring API, as there are already a number of existing
packages that do this well. However, this package does provide a number of
lower-level components outside of MuJoCo itself that are likely to be useful to
most users who access MuJoCo through Python. For example, the `egl`, `glfw`, and
`osmesa` subpackages contain utilities for setting up OpenGL rendering contexts.

## Installation

The recommended way to install this package is via [PyPI](https://pypi.org/project/mujoco/):

```sh
pip install mujoco
```

A copy of the MuJoCo library is provided as part of the package and does **not**
need to be downloaded or installed separately.

### Source

**IMPORTANT:** Building from source is only necessary if you are modifying the
Python bindings (or are trying to run on exceptionally old Linux systems).
If that's not the case, then we recommend installing the prebuilt binaries from
PyPI.

If you need to build the Python bindings from source, please consult
[the documentation](https://mujoco.readthedocs.io/en/latest/python.html#building-from-source).

## Usage

Once installed, the package can be imported via `import mujoco`. Please consult
our [documentation](https://mujoco.readthedocs.io/en/stable/python.html) for
further detail on the package's API.

We recommend going through the tutorial notebook which covers the basics of
MuJoCo using Python:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)

## Versioning

The `major.minor.micro` portion of the version number matches the version of
MuJoCo that the bindings provide. Optionally, if we release updates to the
Python bindings themselves that target the same version of MuJoCo, a `.postN`
suffix is added, for example `2.1.2.post2` represents the second update to the
bindings for MuJoCo 2.1.2.

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited

MuJoCo and its Python bindings are licensed under the Apache License,
Version 2.0. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
