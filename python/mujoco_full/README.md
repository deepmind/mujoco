# MuJoCo Python Bindings

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/mujoco
[pypi-badge]: https://badge.fury.io/py/mujoco.svg
[pypi]: https://pypi.org/project/mujoco/

This package is the canonical Python distribution for the
[MuJoCo physics engine](https://github.com/google-deepmind/mujoco).
These bindings are developed and maintained by Google DeepMind, and are kept
up-to-date with the latest developments in MuJoCo itself.

Installing `mujoco` installs:

- **`mujoco-core`**: Core C API functions, structs, constants, enumerations,
  `MjSpec`, classic OpenGL rendering (`egl`, `glfw`, `osmesa`), and the classic
  `mujoco.viewer`.
- **`mujoco-studio`**: MuJoCo Studio desktop and web viewers (`mujoco.studio`)
  and Filament rendering bindings (`mujoco.mjrf`).
- **`glfw`** and **`pyopengl`**: Default OpenGL context and viewer dependencies.

For minimal headless simulation environments (e.g., RL training or MJX/Warp
workers) that do not need Studio, Filament, or OpenGL, you can install
`mujoco-core` directly (`pip install mujoco-core`), which depends only on
`numpy`. To switch an existing `mujoco` installation to `mujoco-core` only, run:

```sh
pip uninstall -y mujoco mujoco-studio && pip install --force-reinstall --no-deps mujoco-core
```

## Installation

The recommended way to install this package is via [PyPI](https://pypi.org/project/mujoco/):

```sh
pip install mujoco
```

A copy of the MuJoCo library is provided as part of `mujoco-core` and does
**not** need to be downloaded or installed separately.

## Usage

Once installed, the package can be imported via `import mujoco`. Please consult
our [documentation](https://mujoco.readthedocs.io/en/stable/python.html) for
further detail on the package's API.

We recommend going through the tutorial notebook which covers the basics of
MuJoCo using Python:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)

## Versioning

The `major.minor.micro` portion of the version number matches the version of
MuJoCo that the bindings provide.

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited

MuJoCo and its Python bindings are licensed under the Apache License,
Version 2.0. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
