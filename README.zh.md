<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<h1>
  <a href="#"><img alt="MuJoCo" src="banner.png" width="100%"/></a>
</h1>

<p>
  <a href="https://github.com/google-deepmind/mujoco/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco/build.yml?branch=main">
  </a>
  <a href="https://mujoco.readthedocs.io/" alt="Documentation">
    <img src="https://readthedocs.org/projects/mujoco/badge/?version=latest">
  </a>
  <a href="https://github.com/google-deepmind/mujoco/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco">
  </a>
</p>

**MuJoCo** 是 **Mu**lti-**Jo**int dynamics with **Co**ntact（带接触的多关节动力学）的缩写。它是一款通用物理仿真引擎，旨在促进机器人学、生物力学、图形与动画、机器学习，以及其他需要对与环境交互的铰接式关节结构进行快速精准仿真的领域的研究与开发。

本代码库由 [Google DeepMind](https://www.deepmind.com/) 维护。

MuJoCo 提供底层 C API，专为科研人员与开发者设计。其运行时仿真模块经过极致调优以最大化性能，直接在由内置 XML 编译器预先分配的底层数据结构上运算。算法库包含基于 OpenGL 渲染的原生图形界面（GUI）交互式可视化工具。此外，MuJoCo 还提供了丰富的实用函数，用于计算各类物理动力学物理量。

我们还提供官方 [Python 语言绑定 (Python bindings)] 以及面向 [Unity] 游戏引擎的插件。

## 文档 (Documentation)

MuJoCo 的完整官方文档可在 [mujoco.readthedocs.io] 查阅。下一个版本即将发布的新功能特性可查阅 "latest" 分支中的[变更日志 (changelog)]。

## 快速入门 (Getting Started)

上手体验 MuJoCo 有两种便捷方式：

1. **在本地运行 `simulate` 可视化工具**
[此视频](https://www.youtube.com/watch?v=P83tKA1iz2Y) 展示了 MuJoCo 原生交互式可视化查看器 `simulate` 的录屏。请按照官方文档中的[快速入门 (Getting Started)]章节指引，在您的机器上编译并运行 `simulate`。

2. **探索 Google Colab 在线交互式 Notebook**
如果您是 Python 用户，建议从运行在 Google Colab 上的官方教程 Notebook 开始：

 - **基础入门教程 (Introductory)**：讲解 MuJoCo 核心基础：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)
 - **程序化模型编辑 (Model Editing)**：演示如何使用代码程序化创建与修改模型：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb)
 - **轨迹推演教程 (Rollout)**：演示如何使用多线程 `rollout` 模块进行大规模动力学推演：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb)
 - **LQR 控制教程 (LQR)**：合成线性二次型调节器 (LQR)，实现人形机器人单腿平衡控制：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb)
 - **非线性最小二乘 (Least-squares)**：讲解如何使用基于 Python 的非线性最小二乘求解器：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb)
 - **MJX 硬件加速教程**：提供 [MuJoCo XLA](https://mujoco.readthedocs.io/en/stable/mjx.html)（MuJoCo 的 JAX 原生实现分支，支持 GPU/TPU 批处理物理仿真）的使用示例：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb)
 - **可微物理仿真 (Differentiable physics)**：利用从 MuJoCo 物理步长中自动推导的解析梯度训练机器人物理步态运动策略：
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb)

## 安装指南 (Installation)

### 预编译二进制文件

各个正式版本的预编译二进制包可在 GitHub [Releases 页面][releases page] 下载，支持 Linux (x86-64 与 AArch64)、Windows (仅 x86-64) 以及 macOS (通用架构 Universal)。这是使用本软件的推荐方式。

### 源码编译安装

希望从源码构建 MuJoCo 的用户，请查阅官方文档的[源码构建 (build from source)]章节。请注意，`main` 分支最新 Commit 可能处于开发不稳定状态。

### Python 环境 (>= 3.10)

原生 Python 绑定已打包内置了 MuJoCo 运行时，可通过 [PyPI] 直接安装：

```bash
pip install mujoco
```

请注意，预编译的 Linux Wheel 针对 `manylinux2014` 构建，兼容的发行版列表详见[此处](https://github.com/pypa/manylinux)。有关从源码构建 Python 绑定的更多信息，请参阅官方文档的 [Python 绑定 (Python bindings)] 章节。

## 版本发布规范 (Versioning)

我们通常在每月的第一个星期发布 MuJoCo 新版本。从 3.5.0 版本开始，版本规范已调整为修订版语义化版本控制（Semantic Versioning），详情请参阅 [版本规范说明 (versioning)](VERSIONING.md)。

## 参与贡献 (Contributing)

我们非常欢迎开源社区的积极参与：提问、求助、Bug 反馈与新特性建议。关于 Bug 报告、功能请求以及代码贡献的更多细节，请参阅我们的[贡献指南 (contributors guide)](CONTRIBUTING.md)与[代码风格指南 (style guide)](STYLEGUIDE.md)。

## 交流求助 (Asking Questions)

欢迎在 GitHub 的 [“Asking for Help” 讨论区](https://github.com/google-deepmind/mujoco/discussions/categories/asking-for-help) 中提出问题或寻求协助，请尽量聚焦于具体的工程或技术问题。

## 缺陷报告与功能建议 (Bug reports and feature requests)

GitHub [Issues](https://github.com/google-deepmind/mujoco/issues) 专门用于跟踪缺陷报告、功能需求以及其他开发相关事项。

## 关联生态与第三方软件 (Related software)

MuJoCo 是众多强化学习与机器人仿真环境的底层核心基石。以下列出了部分官方与社区语言绑定和格式转换工具：

### 语言绑定 (Bindings)

这些软件包为不同编程语言的用户提供了访问 MuJoCo 功能的接口：

#### 官方第一方绑定：

- [Python 绑定](https://mujoco.readthedocs.io/en/stable/python.html)
  - [dm_control](https://github.com/google-deepmind/dm_control)：Google DeepMind 的强化学习环境技术栈，包含用于程序化操作和构造 MuJoCo 模型的 [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md)。
- [JavaScript 绑定与 WebAssembly 支持](/wasm/README.md)（受 [stillonearth](https://github.com/stillonearth) 与 [zalo](https://github.com/zalo) 社区项目启发；[mjswan](https://github.com/ttktjmt/mjswan) 进一步扩展了实时策略控制与交互式施力功能）。
- [C# 绑定与 Unity 插件](https://mujoco.readthedocs.io/en/stable/unity.html)

#### 社区第三方绑定：

- **MATLAB Simulink**：[Simulink Blockset for MuJoCo Simulator](https://github.com/mathworks-robotics/mujoco-simulink-blockset)（由 [Manoj Velmurugan](https://github.com/vmanoj1996) 开发）
- **Swift**：[swift-mujoco](https://github.com/liuliu/swift-mujoco)
- **Java**：[mujoco-java](https://github.com/CommonWealthRobotics/mujoco-java)
- **Julia**：[MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl)
- **Rust**：[MuJoCo-rs](https://github.com/davidhozic/mujoco-rs)

### 模型格式转换工具 (Converters)

- **OpenSim**：[MyoConverter](https://github.com/MyoHub/myoconverter) 将 OpenSim 生物力学模型转换为 MJCF 格式。
- **SDFormat**：[gz-mujoco](https://github.com/gazebosim/gz-mujoco/) 是 SDFormat 与 MJCF 之间的双向模型转换工具。
- **OBJ**：[obj2mjcf](https://github.com/kevinzakka/obj2mjcf) 用于将复合 OBJ 3D 模型转换为可加载的 MJCF 模型。
- **Onshape**：[Onshape to Robot](https://github.com/rhoban/onshape-to-robot) 将 [Onshape](https://www.onshape.com/en/) CAD 装配体转换为 MJCF。

## 引用 (Citation)

如果您在发表的研究成果中使用了 MuJoCo，请引用以下学术论文：

```bibtex
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

## 开源许可与免责声明 (License and Disclaimer)

Copyright 2021 DeepMind Technologies Limited.

包围盒碰撞检测代码（[`engine_collision_box.c`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c)）版权属于 2016 Svetoslav Kolev。

`doc` 目录下的 reStructuredText 文档、图像及视频遵循知识共享署名 4.0 (CC BY 4.0) 国际许可协议。协议副本可访问 https://creativecommons.org/licenses/by/4.0/legalcode。

源代码遵循 Apache License, Version 2.0 开源许可证。协议副本可访问 https://www.apache.org/licenses/LICENSE-2.0。

*本产品非 Google 官方正式支持的产品。*

[build from source]: https://mujoco.readthedocs.io/en/latest/programming#building-from-source
[Getting Started]: https://mujoco.readthedocs.io/en/latest/programming#getting-started
[Unity]: https://unity.com/
[releases page]: https://github.com/google-deepmind/mujoco/releases
[mujoco.readthedocs.io]: https://mujoco.readthedocs.io
[changelog]: https://mujoco.readthedocs.io/en/latest/changelog.html
[Python bindings]: https://mujoco.readthedocs.io/en/stable/python.html#python-bindings
[PyPI]: https://pypi.org/project/mujoco/

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月20日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
