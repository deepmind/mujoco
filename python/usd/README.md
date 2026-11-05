# MuJoCo OpenUSD Bindings (`mujoco-usd`)

[![PyPI version](https://img.shields.io/pypi/v/mujoco-usd)](https://pypi.org/project/mujoco-usd/)

**`mujoco-usd`** provides the C++ OpenUSD importer (`usd_decoder`), exporter
(`mujoco.usd.writer`, `mujoco.usd.layer_sink`), and custom physics schemas
(`mujoco.usd.MjcPhysics`, `usdMjcf`) for [MuJoCo](https://mujoco.org/).

```bash
pip install mujoco-usd
```

## Version Alignment

`mujoco-usd` pins the exact matching version of `mujoco-core`
(`mujoco-core==<version>`) and `usd-core` (`usd-core==25.11`) that its C++
extensions and OpenUSD plugins were compiled against.
