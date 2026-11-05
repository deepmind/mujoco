# GitHub Actions & build scripts

This folder holds MuJoCo's continuous-integration workflows and the shared build
script they call, [`build_steps.sh`](build_steps.sh). It doubles as the entry
point for building MuJoCo from source for local development: **every CI step is a
`build_steps.sh` function you can run yourself**, so a green CI run is
reproducible on your machine by calling the same functions in order.

> [!NOTE]
> These are build/CI notes for people developing MuJoCo on GitHub. End-user
> documentation — installing the wheel, using the Python / JavaScript APIs, and
> [MuJoCo Studio](https://mujoco.readthedocs.io/en/latest/studio/index.html) —
> lives at <https://mujoco.readthedocs.io>, and the build instructions collected
> here are intended to move there over time. Treat this file as the interim home
> while the build story stabilises.

## What's in this folder

| File | Purpose |
|------|---------|
| [`build.yml`](build.yml) | Main CI. The `mujoco` compiler matrix (including Studio) plus the `wasm`, `mjx` and `bazel` jobs. |
| [`build_steps.sh`](build_steps.sh) | Every build/test step, as a shell function. CI calls these; so can you. |
| [`build_matrix.json`](build_matrix.json) | The CMake and Bazel matrices `build.yml` expands (a "core" subset on PRs, the full sweep on push). |
| [`lint.yml`](lint.yml) | Pre-commit checks (whitespace, EOF, `clang-format`) on PRs. |
| [`live.yml`](live.yml) / [`update_live.yml`](update_live.yml) | Build (`build_mujoco_live`) + deploy [MuJoCo Live](https://live.mujoco.org) to GitHub Pages. |
| [`publish-wasm.yml`](publish-wasm.yml) | Publish the `@mujoco/mujoco` npm package. |

## The `build_steps.sh` convention

> [!IMPORTANT]
> Run every command below from the repository **top-level directory**. Build
> trees land in top-level folders (`build`, `build_simulate`, `build_host`,
> `build_wasm`, `build_python`) and packaging output in `python/dist`.

Run one step with:

```sh
bash .github/workflows/build_steps.sh <function_name>
```

Each CI job in `build.yml` is just an ordered sequence of these calls. The
sections below group the functions into the artifact they build.

<details>
<summary><b>CI matrix verification (MuJoCo engine and Python bindings)</b></summary>

In GitHub Actions (`build.yml`), the `mujoco` matrix job verifies compilation
and testing of the core MuJoCo C physics engine (`libmujoco`), Studio (on
`MUJOCO_BUILD_STUDIO=ON` entries), samples, `simulate`, and the Python bindings
across Linux, macOS and Windows under several compilers (GCC, Clang, MSVC). CI
runs these steps in order:

```sh
bash .github/workflows/build_steps.sh configure_mujoco         # -> build/
bash .github/workflows/build_steps.sh build_mujoco_core        # libmujoco + engine plugins (run in build/)
bash .github/workflows/build_steps.sh build_mujoco_studio      # mujoco_studio binary when MUJOCO_BUILD_STUDIO=ON
bash .github/workflows/build_steps.sh build_mujoco             # remaining targets + C++ test binaries
bash .github/workflows/build_steps.sh test_mujoco              # parallel engine ctest suite
bash .github/workflows/build_steps.sh install_mujoco           # -> ${TMPDIR}/mujoco_install
bash .github/workflows/build_steps.sh copy_plugins_posix       # or copy_plugins_window on Windows
bash .github/workflows/build_steps.sh install_studio_platform  # Studio SDK/assets when MUJOCO_BUILD_STUDIO=ON
bash .github/workflows/build_steps.sh configure_samples        # standalone sample/ build check
bash .github/workflows/build_steps.sh build_samples
bash .github/workflows/build_steps.sh configure_simulate       # standalone simulate/ build check
bash .github/workflows/build_steps.sh build_simulate
bash .github/workflows/build_steps.sh make_python_sdist        # -> python/dist/*.tar.gz
bash .github/workflows/build_steps.sh build_python_bindings    # -> python/dist/*.whl
bash .github/workflows/build_steps.sh install_python_bindings
bash .github/workflows/build_steps.sh test_python_bindings     # pytest + Studio import smoke test
```

`build_python_bindings` / `make_python_sdist` expect a virtualenv; CI creates one
under `${TMPDIR}/venv` via `prepare_python`.

> [!TIP]
> **Local development**: to iterate on the physics engine and the
> Python bindings without running the individual CI steps yourself, run:
>
> ```sh
> bash .github/workflows/build_steps.sh build_engine           # builds and installs into the active venv
> bash .github/workflows/build_steps.sh test_python_bindings   # runs pytest
> ```
>
> For the complete self-contained distribution wheel (Studio and the web
> viewer included), see the [developer workflows](#developer-workflows).
</details>

<details>
<summary><b>Studio CI coverage (desktop, Python, WebViewer WASM, and MuJoCo Live)</b></summary>

**Studio** ([documentation](https://mujoco.readthedocs.io/en/latest/studio/index.html))
replaces the legacy `simulate` architecture with a modular visualization stack
built on [Dear ImGui](https://github.com/ocornut/imgui) and Filament, providing
a standalone **C++ Studio viewer** (`mujoco_studio`), a **Python Studio viewer**
(`python -m mujoco.studio.viewer`), a headless **WebViewer** (`--gfx=web`), and
**[MuJoCo Live](https://live.mujoco.org)** in the browser.

In GitHub Actions:

- **Desktop C++ & Python extensions (`build.yml` `mujoco` matrix)**: Matrix
  entries with `MUJOCO_BUILD_STUDIO=ON` compile `mujoco_studio`
  (`build_mujoco_studio`), stage the Studio platform SDK
  (`install_studio_platform`), compile the Studio Python extensions into the
  wheel (`build_python_bindings`), and smoke-import them in
  `test_python_bindings`.
- **WebViewer browser client (`build.yml` `wasm` job)**: Runs
  `build_studio_wasm` alongside the JavaScript WASM bindings check.
- **MuJoCo Live (`live.yml`)**: Compiles the standalone desktop (`build_wasm/`)
  and iOS (`build_wasm_ios/`) WebAssembly apps via `build_mujoco_live` and
  deploys them to GitHub Pages.

For local incremental Studio builds, use the `build_studio_*` steps under
[Developer workflows](#developer-workflows) below.
</details>

<details>
<summary><b>Build the WASM / JavaScript bindings (<code>@mujoco/mujoco</code>)</b></summary>

These are the browser bindings that compile the MuJoCo engine to WebAssembly and
expose it to JavaScript/TypeScript.

#### 1. Build the WASM bindings

Install the dependencies and compile the bindings into `wasm/dist/`
(multi-threaded under `wasm/dist/mt`, single-threaded in `wasm/dist`):

```sh
bash .github/workflows/build_steps.sh npm_ci          # installs wasm/ node deps
bash .github/workflows/build_steps.sh setup_emsdk      # installs Emscripten 4.0.10 into ./emsdk
bash .github/workflows/build_steps.sh build_test_wasm  # CI: builds and tests both threading modes
bash .github/workflows/build_steps.sh build_wasm       # local: builds both modes, no tests
```

To build by hand, first `source ./emsdk/emsdk_env.sh`, then:

```sh
emcmake cmake -B build && cmake --build build           # single-threaded
emcmake cmake -B build -DMUJOCO_WASM_THREADS=ON && cmake --build build   # multi-threaded
```

#### 2. Iterating on the sandbox and the demo app

The sandbox testbed and the Three.js demo app both consume the compiled
bindings from `wasm/dist/` and run locally on Vite dev servers with hot module
replacement:

- **Sandbox** (`wasm/tests/sandbox/main.ts`): `npm run dev:sandbox --prefix ./wasm`
  starts a dev server (at `http://localhost:5173/`) with the
  `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers the
  multi-threaded build needs; edits to `tests/sandbox/main.ts` reload live.
- **Demo app** (`wasm/demo_app/app.ts`): `npm run dev:demo --prefix ./wasm`
  starts the Three.js demo with live reload.
- **C++ bindings or engine code**: recompile the WASM binary and the dev server
  picks up the new module from `wasm/dist/`:
  ```sh
  source ./emsdk/emsdk_env.sh
  cmake --build build_wasm_st   # or build_wasm_mt
  ```

#### 3. Testing the WASM bindings

Testing is separate from compilation:

```sh
bash .github/workflows/build_steps.sh test_wasm        # Jasmine suite against wasm/dist (npm run test --prefix ./wasm)
```

> [!NOTE]
> CI uses `build_test_wasm`, which builds and tests the multi-threaded and the
> single-threaded configuration in sequence.

> [!TIP]
> The full JavaScript **API reference and user guide** (named access, memory
> management, out-parameters, threading headers, …) lives in
> [`wasm/README.md`](../../wasm/README.md), which is also the README shipped with
> the npm package. We may in future move that README into this file.
</details>

<details>
<summary><b>MJX and Bazel CI jobs</b></summary>

- **MJX (`build.yml` `mjx` job)**: Builds `libmujoco` and the Python bindings
  once on Linux (`-DMUJOCO_BUILD_TESTS=OFF`), then runs `package_mjx`,
  `install_mjx`, and `test_mjx` inside `mjx/`.
- **Bazel (`build.yml` `bazel` matrix)**: Expands the `bazel` entries in
  `build_matrix.json` across Linux and macOS (pinned `.bazelversion` on PRs,
  plus Bazel 8.x and `latest` on pushes to `main`) and runs:
  ```sh
  bash .github/workflows/build_steps.sh test_bazel  # bazel test //... + sample/ Bzlmod check
  ```
</details>

## Developer workflows

The Studio `build_*` steps in `build_steps.sh` are idempotent and incremental:
they can be run in any order, and the native build trees are reused, so a
repeated step only rebuilds what changed. Steps described as **installed** end
by packaging a wheel and installing it into your virtualenv; that part is not
incremental, so any source edit — Python or C++ — needs the step re-run.

> [!IMPORTANT]
> **TODO: make the Python steps incremental via editable installs.**
>
> Every step that installs Python does so by building a wheel from a freshly
> packaged sdist, in pip's own temporary tree. Nothing is carried over between
> runs, so a one-line edit to a binding costs a full rebuild of the extensions
> (ccache makes this survivable, not fast).
>
> An editable install (`pip install -e python/`) with a persistent CMake tree
> would turn that into a single-module recompile, and would let pure-Python
> edits take effect with no build at all. The steps this would speed up:
>
> | Step | Affects | Today | With an editable install |
> |------|---------|-------|--------------------------|
> | `build_engine` | the CI-matrix tip above | full rebuild of the core bindings | one module recompiled |
> | `build_studio_python` | workflows 4 and 5 | full rebuild of the Studio extensions | one module recompiled; pure-Python edits need no build |
> | `build_studio_all` | workflow 6 | as above, plus the browser client | as above; the client is already incremental |
>
> Workflows 1, 2, 3 and 7 are unaffected: 1 to 3 never touch Python, and 7
> deliberately builds a clean wheel from scratch.

#### Prerequisites

- **Platforms**: Linux (Ubuntu 24.04 with clang-18, as in CI) or macOS
  (Apple Silicon). Windows is untested.
- **Tools**: a native CMake 3.26 or newer (a pip-installed `cmake` cannot be
  used from pip's isolated build environment, which `build_wheel` relies on;
  older CMake places the C++-standard flag after spirv-cross's own
  `-std=c++11 -Werror`, which fails under clang-18), Ninja, git.
- **Browser client**: Node.js and npm, used by `setup_emsdk` (installs
  Emscripten 4.0.10 into `./emsdk`; `build_studio_wasm` runs it when `emsdk/`
  is missing).
- **Python** (workflows 4 to 8): an active virtualenv (`VIRTUAL_ENV`) with
  `pip install -r python/build_requirements.txt`. Without one, the install
  part of a step is skipped with a note.
- **ccache** is strongly recommended: the wheel is compiled from a fresh sdist
  on every install, and ccache is what keeps that to a tolerable length.
- **Compiler selection** goes through `CMAKE_ARGS`, e.g.
  `CMAKE_ARGS="-DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18"`.

| Step | What it builds | Output |
|------|----------------|--------|
| `build_engine` | `libmujoco` + engine plugins (with tests and simulate), then the core Python bindings, installed | `build/`, `${TMPDIR:-build}/mujoco_install/`, `python/dist/` |
| `build_simulate_app` | the engine and the classic `simulate` app, no Filament | `build_simulate/bin/simulate` |
| `build_studio_cpp_viewer` | the C++ Studio viewer app with its plugins, in the host tree | `build_host/bin/mujoco_studio` |
| `build_studio_platform_libs` | the Studio platform without the app: platform library (and the Filament tools), staged assets, engine plugins | `build_host/` |
| `build_studio_wasm` | the web viewer browser client (WASM/JS/HTML + assets) via Emscripten | `python/mujoco/studio/web/dist/` |
| `build_studio_python` | platform libraries + SDK, then the Studio Python extensions (`_studio` — bundling `ux`, `sim`, `window`, `native_viewer_cc`, and on POSIX `headless_ui` + `state_payload` — plus `_render_filament`, `dear_imgui`, `implot`), installed | `build/mujoco_install/`, `python/dist/` |
| `build_studio_all` | `build_studio_python` plus the browser client | everything above |
| `build_wheel` | platform libraries + SDK + browser client, then a clean sdist and the wheel from it (no install, no app) | `python/dist/mujoco-*.whl` |
| `build_mujoco_live` | standalone [MuJoCo Live](https://live.mujoco.org) WASM app (desktop + iOS profiles, used by `live.yml`; run `setup_emsdk` once first) | `build_wasm/bin/`, `build_wasm_ios/bin/` |

Host-build reuse: the Studio steps look for the host build in
`MUJOCO_NATIVE_BUILD_DIR`, then in `build/` when it was configured with
`MUJOCO_BUILD_STUDIO=ON`, and only then use `build_host/`.

<details>
<summary><b>1. Developing C engine code using the C++ Studio Viewer</b></summary>

```sh
bash .github/workflows/build_steps.sh build_studio_cpp_viewer
./build_host/bin/mujoco_studio model/humanoid/humanoid.xml
```

An edit under `src/engine/` recompiles that translation unit and relinks
`libmujoco` and the app; Filament and the Studio UI are not recompiled.
</details>

<details>
<summary><b>2. Developing C engine code using the C++ Simulate Viewer</b></summary>

```sh
bash .github/workflows/build_steps.sh build_simulate_app
./build_simulate/bin/simulate model/humanoid/humanoid.xml
```

Its own tree, `build_simulate/`, with only the engine, GLFW and lodepng:
no Filament, Dear ImGui, Emscripten or Python. Rebuilds take seconds. (CI's
`configure_simulate` / `build_simulate` are a different thing: they build the
`simulate/` sample against an installed MuJoCo.)
</details>

<details>
<summary><b>3. Developing the C++ Studio Viewer</b></summary>

Same step as workflow 1: `build_studio_cpp_viewer`. Edits under
`studio/ux/`, `studio/hal/` or `studio/main.cc`
recompile and relink in seconds; no Python environment or Emscripten needed.
</details>

<details>
<summary><b>4. Developing the Python Studio Viewer</b></summary>

```sh
source /path/to/venv/bin/activate
pip install -r python/build_requirements.txt
bash .github/workflows/build_steps.sh build_studio_python
python -m mujoco.studio.viewer --model=model/humanoid/humanoid.xml
```

`build_studio_python` builds the Studio platform, then packages and installs a
wheel. Any edit — Python (`viewer.py`, plugins) or C++ binding (`sim.cc`,
`ux.cc`, `headless_ui.cc`, ...) — needs the step re-run before it takes effect.
The platform libraries are reused across runs and ccache absorbs
most of the recompilation, but the extensions themselves are rebuilt from a
fresh sdist every time, so expect minutes rather than seconds. The C++ app
(`main.cc`, `launcher.cc`) is never compiled by this workflow.

> [!NOTE]
> **TODO(matijak):** Re-export Studio Python tests (`studio/*_test.py`) so
> `test_python_bindings` runs them via `pytest` in addition to its Studio
> module import smoke checks (`build.yml` matrix jobs with
> `MUJOCO_BUILD_STUDIO=ON` already build and install the Studio C++ extensions
> via `install_studio_platform`).
</details>

<details>
<summary><b>5. Developing the Python Studio sample code</b></summary>

```sh
source /path/to/venv/bin/activate
pip install -r python/build_requirements.txt
bash .github/workflows/build_steps.sh build_studio_python

python python/mujoco/studio/viewer.py --model=model/humanoid/humanoid.xml --plugins=python/mujoco/sample/studio/ghost.py
python python/mujoco/studio/viewer.py --model=model/humanoid/humanoid.xml --plugins=python/mujoco/sample/studio/implot.py
python python/mujoco/sample/studio/custom_sim_loop.py
python python/mujoco/sample/studio/filament_render.py model/humanoid/humanoid.xml
python python/mujoco/sample/studio/filament_camera.py model/humanoid/humanoid.xml
python python/mujoco/sample/studio/filament_multiview.py model/humanoid/humanoid.xml
python python/mujoco/sample/studio/filament_multimodel.py model/humanoid/humanoid.xml model/cartpole/cartpole.xml
```

The samples under `python/mujoco/sample/studio/` show ghost
overlays, ImPlot integration, driving `launch_web.launch` from a custom
simulation loop, and the Filament Python rendering API
(`mujoco.rendering.filament`, backed by `_render_filament` and `_studio`'s
`window` submodule that `build_studio_python` installs).
The samples are run from the source tree, so editing one takes effect
immediately; editing the library behind it needs the step re-run.
</details>

<details>
<summary><b>6. Developing the Python Studio Web Viewer browser (WASM) client</b></summary>

```sh
bash .github/workflows/build_steps.sh build_studio_all
python -m mujoco.studio.viewer --model=model/humanoid/humanoid.xml --gfx=web
```

`build_studio_all` runs `build_studio_wasm`, which compiles the client with
Emscripten and stages it into `python/mujoco/studio/web/dist/`,
and then installs a wheel carrying a copy of that directory. Because the
browser loads the client from the installed wheel, a client rebuild only takes
effect after the reinstall — so run the whole step rather than
`build_studio_wasm` on its own.
</details>

<details>
<summary><b>7. Compiling a self-contained MuJoCo Python wheel for downstream projects</b></summary>

```sh
bash .github/workflows/build_steps.sh build_wheel
pip install python/dist/mujoco-*.whl
```

Platform libraries, SDK and browser client, then a clean sdist and the wheel
compiled from it in pip's own temporary tree, with the engine, the Python
bindings, the Studio modules, the plugins and the pre-built browser client.
MuJoCo has one distribution wheel; this is it.
</details>

<details>
<summary><b>8. Sharing the Web Viewer with other machines (Cloudflare quick tunnel)</b></summary>

```sh
python -m mujoco.studio.viewer --model=model/humanoid/humanoid.xml --gfx=web
cloudflared tunnel --url http://localhost:8080 --no-autoupdate
```

The viewer server listens on all interfaces, but a machine on another network
(or behind WSL2's own network) cannot reach `localhost:8080`. A Cloudflare
quick tunnel publishes it without any account or router configuration: the
second command prints a random `https://<words>.trycloudflare.com` URL that
anyone can open, from anywhere. The client switches to `wss://` under https,
and Cloudflare passes the viewer's WebSocket connections through; the first
browser to connect becomes the controller, later ones are viewers.

`cloudflared` is a single binary from
https://github.com/cloudflare/cloudflared/releases (Linux, macOS, Windows);
put it on your `PATH`. Run it from the same machine as the viewer server, and
leave both running; the tunnel closes when `cloudflared` exits.

Caveats: the URL is public, not limited to your network, and unauthenticated,
so share it only with the people meant to see the simulation. Quick tunnels
carry no uptime guarantee and the URL changes on every restart of
`cloudflared`. For a fixed, memorable hostname use a named tunnel on a domain
in your own Cloudflare account (`cloudflared tunnel login`, then
`cloudflared tunnel create` / `route dns` / `run`); that setup is outside this
document.

> [!TIP]
> **A stable, memorable URL without a domain.** URL shorteners refuse
> `trycloudflare.com` targets (they blacklist redirect services), but a GitHub
> Pages project site can forward to the tunnel for free. Create a public
> repository named, say, `robosim`, containing a single `index.html`:
>
> ```html
> <meta http-equiv="refresh" content="0; url=https://<words>.trycloudflare.com/">
> ```
>
> and enable Pages from its `main` branch:
>
> ```sh
> gh repo create <user>/robosim --public --source . --push
> echo '{"source":{"branch":"main","path":"/"}}' | gh api -X POST repos/<user>/robosim/pages --input -
> ```
>
> A minute later `https://<user>.github.io/robosim` forwards every visitor to
> the viewer. When the tunnel restarts, put the new hostname in that one line
> and push; the site rebuilds within a minute.
</details>
