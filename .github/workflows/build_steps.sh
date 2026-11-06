#!/bin/bash
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

# TODO(matijak): Make all cmake commands run from the top-level directory, and
# consider making the builds parallel.


# Wrap the compiler with ccache when it is available (set up by ccache-action in
# CI). This makes warm rebuilds - including the expensive, pinned Filament build -
# much faster. ccache is content-addressed on the full compiler invocation, so
# changing a flag or source forces a recompile: a stale object is never reused.
# Guarded by `command -v` so the script still works locally without ccache.
CCACHE_ARGS=""
if command -v ccache >/dev/null 2>&1; then
    CCACHE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

# Portable parallel job count. getconf works on Linux and macOS; on Windows we
# fall back to the NUMBER_OF_PROCESSORS environment variable, then to 4.
NJOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-4}")"


# Emit the build matrix for build.yml as a step output. On pull_request we run
# only the representative "core" compiler set; on push (e.g. to main) we run the
# full compiler sweep. Tiers are defined in build_matrix.json.
generate_matrix() {
    echo "Generating build matrix for event '${GITHUB_EVENT_NAME}'..."
    local file=".github/workflows/build_matrix.json"
    local matrix
    local bazel_matrix
    if [[ "${GITHUB_EVENT_NAME}" == "pull_request" ]]; then
        matrix="$(jq -c '{include: [.include[] | select(.tier == "core") | del(.tier)]}' "${file}")"
        bazel_matrix="$(jq -c '{include: [.bazel[] | select(.tier == "core") | del(.tier)]}' "${file}")"
    else
        matrix="$(jq -c '{include: [.include[] | del(.tier)]}' "${file}")"
        bazel_matrix="$(jq -c '{include: [.bazel[] | del(.tier)]}' "${file}")"
    fi
    echo "matrix=${matrix}" >> "${GITHUB_OUTPUT}"
    echo "bazel_matrix=${bazel_matrix}" >> "${GITHUB_OUTPUT}"
    echo "${matrix}" | jq .
    echo "${bazel_matrix}" | jq .
}


prepare_linux() {
    echo "Preparing Linux..."
    sudo apt-get update
    local extra_pkgs=()
    if apt-cache show libclang-rt-18-dev >/dev/null 2>&1; then
        extra_pkgs+=(libclang-rt-18-dev)
    fi
    sudo apt-get install -y \
        libgl1-mesa-dev \
        libwayland-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxkbcommon-dev \
        libxrandr-dev \
        libxi-dev \
        ninja-build \
        "${extra_pkgs[@]}"
}


# Install a hash-pinned requirements file into the active virtualenv. When uv
# is available, try an offline install from UV_CACHE_DIR first so warm CI runs
# make zero network requests to PyPI, falling back to an online install on
# cache misses.
_pip_install_requirements() {
    local req_file="$1"
    if command -v uv > /dev/null 2>&1; then
        uv pip install --offline --require-hashes -r "${req_file}" || \
            uv pip install --require-hashes -r "${req_file}"
    else
        python -m pip install --upgrade --require-hashes -r "${req_file}"
    fi
}


prepare_python() {
    echo "Preparing Python..."
    repo="${PWD}"
    pushd "${TMPDIR}" > /dev/null
    python -m venv venv
    if [[ $RUNNER_OS == "Windows" ]]; then
        mkdir -p venv/bin
        if ! grep -q "cygpath" venv/Scripts/activate; then
            fixpath="$(grep '^VIRTUAL_ENV=' venv/Scripts/activate | cut -d= -f2- | tr -d '\"')"
            if [[ -n "${fixpath}" ]]; then
                sed -i "s#$(printf "%q" "${fixpath}")#$(cygpath "${fixpath}")#g" venv/Scripts/activate
            fi
        fi
        ln -sf ../Scripts/activate venv/bin/activate
    fi
    source venv/bin/activate
    # Bootstrap the hash-pinned uv binary into the venv on POSIX (restoring from
    # UV_CACHE_DIR on cache hits so no network request is made). Windows stays on
    # pip to keep the venv activate fixup path untouched there. The venv is still
    # created by `python -m venv`, so pip stays available for later steps (pip
    # wheel / python -m build).
    local uv_req="${repo}/.github/workflows/uv_requirements.txt"
    if [[ "${RUNNER_OS:-}" != "Windows" && -f "${uv_req}" ]]; then
        if [[ -n "${UV_CACHE_DIR:-}" && -x "${UV_CACHE_DIR}/bin/uv" && -f "${UV_CACHE_DIR}/bin/uv_requirements.txt" ]] && \
           cmp -s "${uv_req}" "${UV_CACHE_DIR}/bin/uv_requirements.txt"; then
            cp -p "${UV_CACHE_DIR}/bin/uv" venv/bin/uv
        else
            python -m pip install --no-deps --require-hashes -r "${uv_req}"
            if [[ -n "${UV_CACHE_DIR:-}" && -x venv/bin/uv ]]; then
                mkdir -p "${UV_CACHE_DIR}/bin"
                cp -p venv/bin/uv "${UV_CACHE_DIR}/bin/uv"
                cp -p "${uv_req}" "${UV_CACHE_DIR}/bin/uv_requirements.txt"
            fi
        fi
    fi
    _pip_install_requirements "${repo}/python/build_requirements.txt"
    _pip_install_requirements "${repo}/python/build_requirements_usd.txt"
    popd > /dev/null
}


npm_ci() {
    echo "Installing NPM dependencies for WASM bindings..."
    pushd wasm
    npm ci
    popd
}


setup_emsdk() {
    echo "Setting up Emscripten..."
    git clone https://github.com/emscripten-core/emsdk.git
    ./emsdk/emsdk install 4.0.10
    ./emsdk/emsdk activate 4.0.10
    # Force installing emscripten's typescript dependencies. This is a
    # workaround for the github update to a newer typescript, which gives an
    # error on the deprecated `--outFile` flag.
    pushd emsdk/upstream/emscripten
    npm i || true
    popd
}


# Populate ext_deps_args with -DFETCHCONTENT_SOURCE_DIR_<DEP>=<path> for each
# cached dependency in $2 (defaults to EXTERNAL_DEPS_DIR). When a target build
# directory ($1) is given, stage each cached dependency into $1/_deps/<dep>-src
# so targets and helper scripts that reference ${CMAKE_BINARY_DIR}/_deps/<dep>-src
# directly (e.g. Studio fonts, Filament WASM host imports,
# install_studio_platform) find the sources at their standard _deps/<dep>-src
# paths.
_setup_ext_deps_args() {
    local build_dir="${1:-}"
    local ext_deps_dir="${2:-${EXTERNAL_DEPS_DIR:-${TMPDIR:-/tmp}/external_deps}}"
    ext_deps_args=()
    if [[ ! -d "${ext_deps_dir}" ]]; then
        return 0
    fi
    ext_deps_dir="$(cd "${ext_deps_dir}" && pwd)"
    local target_deps_dir=""
    if [[ -n "${build_dir}" ]]; then
        mkdir -p "${build_dir}/_deps"
        target_deps_dir="$(cd "${build_dir}/_deps" && pwd)"
    fi
    for dep_path in "${ext_deps_dir}"/*; do
        if [[ -d "${dep_path}" ]]; then
            local dep_name
            dep_name="$(basename "${dep_path}")"
            if [[ "${dep_name}" == *-build || "${dep_name}" == *-subbuild ]]; then
                continue
            fi
            dep_name="${dep_name%-src}"
            if [[ "${dep_name}" == "mujoco" ]]; then
                continue
            fi
            local dep_upper
            dep_upper="$(echo "${dep_name}" | tr '[:lower:]' '[:upper:]')"
            local use_path="${dep_path}"
            if [[ -n "${target_deps_dir}" ]]; then
                local staged_dir="${target_deps_dir}/${dep_name}-src"
                if [[ ! -d "${staged_dir}" ]]; then
                    cp -al "${dep_path}" "${staged_dir}" 2>/dev/null || \
                        cp -RLp "${dep_path}" "${staged_dir}" || \
                        cp -rp "${dep_path}" "${staged_dir}"
                fi
                use_path="${staged_dir}"
            fi
            if command -v cygpath >/dev/null 2>&1; then
                use_path="$(cygpath -m "${use_path}")"
            fi
            echo "Using cached ${dep_name} from ${use_path}"
            ext_deps_args+=("-DFETCHCONTENT_SOURCE_DIR_${dep_upper}=${use_path}")
            # python/mujoco/CMakeLists.txt declares "eigen" instead of "Eigen3".
            if [[ "${dep_upper}" == "EIGEN3" ]]; then
                ext_deps_args+=("-DFETCHCONTENT_SOURCE_DIR_EIGEN=${use_path}")
            elif [[ "${dep_upper}" == "EIGEN" ]]; then
                ext_deps_args+=("-DFETCHCONTENT_SOURCE_DIR_EIGEN3=${use_path}")
            fi
        fi
    done
}


# Copy any newly fetched dependencies from $1/_deps/<dep>-src into
# EXTERNAL_DEPS_DIR so actions/cache can persist them across runs.
_populate_ext_deps_cache() {
    local build_dir="${1:-.}"
    local ext_deps_dir="${EXTERNAL_DEPS_DIR:-${TMPDIR:-/tmp}/external_deps}"
    for src in "${build_dir}/_deps"/*-src; do
        if [[ ! -d "${src}" ]]; then
            continue
        fi
        local dep_dir="${src##*/}"
        local dst_name="${dep_dir%-src}"
        if [[ "${dst_name}" == "mujoco" ]]; then
            continue
        fi
        if [[ "${dst_name}" == "eigen" ]]; then
            dst_name="eigen3"
        fi
        local dst="${ext_deps_dir}/${dst_name}"
        if [[ ! -d "${dst}" ]]; then
            echo "Populating external dependencies cache for ${dst_name}..."
            mkdir -p "${ext_deps_dir}"
            rm -rf "${src}/.git"
            cp -RLp "${src}" "${dst}" || cp -rp "${src}" "${dst}"
        fi
    done
}


configure_mujoco() {
    echo "Configuring MuJoCo..."
    # Disable global IPO/LTO to avoid expensive whole-program optimization link
    # passes on third-party dependencies, Studio, and the 65 test binaries. On
    # Windows MSVC, CMakeLists.txt keeps INTERPROCEDURAL_OPTIMIZATION enabled on
    # the core mujoco DLL target to avoid an MSVC /EHsc + longjmp unwinding crash
    # in SetConstTest.SleepingNotAllowed.
    local ipo_off="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF"

    # Use cached external dependencies from actions/cache.
    local ext_deps_args=()
    mkdir -p build
    _setup_ext_deps_args build

    cd build &&
    cmake .. \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        ${ipo_off} \
        -DCMAKE_INSTALL_PREFIX:STRING=${TMPDIR}/mujoco_install \
        -DMUJOCO_BUILD_EXAMPLES:BOOL=OFF \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS} \
        "${ext_deps_args[@]}" &&
    _populate_ext_deps_cache .
}


build_mujoco_core() {
    echo "Building MuJoCo core and plugins..."
    cmake --build . --config=Release --target mujoco actuator sensor sdf_plugin ${CMAKE_BUILD_ARGS}
}


build_mujoco_studio() {
    echo "Building MuJoCo Studio..."
    cmake --build . --config=Release --target mujoco_studio ${CMAKE_BUILD_ARGS}
}


build_mujoco() {
    echo "Building MuJoCo..."
    cmake --build . --config=Release ${CMAKE_BUILD_ARGS}
}


test_mujoco() {
    echo "Testing MuJoCo..."
    # ctest defaults to serial. The suite is ~1650 independent tests that use
    # unique temp files (mkstemp / testing::TempDir) and declare no RUN_SERIAL /
    # RESOURCE_LOCK, so running them in parallel is safe.
    ctest -C Release --output-on-failure --parallel "${NJOBS}" .
}


install_mujoco() {
    echo "Installing MuJoCo..."
    cmake --install .
}


copy_plugins_posix() {
    echo "Copying plugins..."
    mkdir -p ${TMPDIR}/mujoco_install/mujoco_plugin &&
    find lib -maxdepth 2 -name "libactuator.*" -exec cp {} ${TMPDIR}/mujoco_install/mujoco_plugin \; &&
    find lib -maxdepth 2 -name "libsensor.*" -exec cp {} ${TMPDIR}/mujoco_install/mujoco_plugin \; &&
    find lib -maxdepth 2 -name "libsdf_plugin.*" -exec cp {} ${TMPDIR}/mujoco_install/mujoco_plugin \;
}


copy_plugins_window() {
    echo "Copying plugins..."
    mkdir -p ${TMPDIR}/mujoco_install/mujoco_plugin &&
    local dll_dir="bin"
    if [[ -d "bin/Release" ]]; then
        dll_dir="bin/Release"
    fi
    cp ${dll_dir}/actuator.dll ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp ${dll_dir}/sensor.dll ${TMPDIR}/mujoco_install/mujoco_plugin
}


configure_samples() {
    echo "Configuring samples..."
    # Samples are tiny, so they keep the default IPO/LTO: disabling it saves no
    # meaningful build time and would expose the same gcc -Werror false positives
    # that -O3-without-LTO triggers (see configure_mujoco).
    local ext_deps_args=()
    mkdir -p build
    _setup_ext_deps_args build

    cd build &&
    cmake .. \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -Dmujoco_ROOT:STRING=${TMPDIR}/mujoco_install \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS} \
        "${ext_deps_args[@]}" &&
    _populate_ext_deps_cache .
}


build_samples() {
    echo "Building samples..."
    cmake --build . --config=Release ${CMAKE_BUILD_ARGS}
}


configure_simulate() {
    echo "Configuring simulate..."
    # See configure_samples: keep the default IPO/LTO for this small build.
    local ext_deps_args=()
    mkdir -p build
    _setup_ext_deps_args build

    cd build &&
    cmake .. \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -Dmujoco_ROOT:STRING=${TMPDIR}/mujoco_install \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS} \
        "${ext_deps_args[@]}" &&
    _populate_ext_deps_cache .
}


build_simulate() {
    echo "Building simulate..."
    cmake --build . --config=Release ${CMAKE_BUILD_ARGS}
}


configure_studio() {
    echo "Configuring Studio..."
    cmake -B build \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DUSE_STATIC_LIBCXX=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_TEST_PYTHON_UTIL=OFF \
        -DMUJOCO_WITH_USD=OFF \
        -DMUJOCO_USE_FILAMENT=ON \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS}
    echo "Configuring Studio... DONE"
}


build_studio() {
    echo "Building Studio..."
    cmake --build build --config=Release --target mujoco_studio --parallel
    echo "Building Studio... DONE"
}


make_python_sdist() {
    echo "Making Python sdists..."
    source ${TMPDIR}/venv/bin/activate &&
    ./make_sdist.sh &&
    (cd studio && ./make_sdist.sh) &&
    (cd usd && ./make_sdist.sh) &&
    (cd mujoco_full && ./make_sdist.sh) &&
    mv studio/dist/mujoco_studio-*.tar.gz usd/dist/mujoco_usd-*.tar.gz mujoco_full/dist/mujoco-*.tar.gz dist/
}


build_python_bindings() {
    echo "Building Python bindings..."
    source ${TMPDIR}/venv/bin/activate
    # pip unpacks the sdist into a randomized temp dir every run, so the absolute
    # source/include paths differ each time and defeat ccache (0% hit, full
    # recompile). CCACHE_BASEDIR rewrites absolute paths under it to paths relative
    # to the (also-in-temp) build cwd, cancelling the random component so objects
    # hash identically across runs. CCACHE_SLOPPINESS ignores timestamp/path noise.
    #
    # Do NOT add system_headers here: CMake adds the imported mujoco target's
    # include dir (MUJOCO_PATH/include) as -isystem, so ccache would treat the
    # public MuJoCo headers as system headers and skip hashing them. A change that
    # lives only in those headers (a new mjData field, a new enum value) would then
    # go undetected and ccache would reuse an object compiled against the old struct
    # layout, producing an ABI-mismatched binding (wrong field offsets, stale
    # mjNENABLE, signature mismatch). The mtime/ctime flags are kept: they handle the
    # temp-dir churn without affecting header content detection.
    # On Windows (Git Bash / MSYS2), TMPDIR is an MSYS path (/c/Temp) that MSYS2
    # does not auto-convert in custom env vars passed to native Win32 processes,
    # so convert it to a Windows path (C:/Temp) for ccache.exe. On macOS, /tmp is
    # a symlink to /private/tmp, so resolve symlinks with pwd -P to match the
    # canonicalized compiler paths in pip's temp build directory.
    if command -v cygpath >/dev/null 2>&1; then
        export CCACHE_BASEDIR="$(cygpath -m "${TMPDIR}")"
    else
        export CCACHE_BASEDIR="$(cd "${TMPDIR}" && pwd -P)"
    fi
    export CCACHE_NOHASHDIR=1
    export CCACHE_SLOPPINESS="time_macros,include_file_mtime,include_file_ctime,pch_defines,locale"
    local py_deps_dir="${CCACHE_BASEDIR}/python_cmake_deps"
    rm -rf "${py_deps_dir}"
    mkdir -p "${py_deps_dir}/_deps"
    local ext_deps_args=()
    _setup_ext_deps_args
    local sdists=(mujoco_core-*.tar.gz)
    if [[ -f "${TMPDIR}/mujoco_install/lib/libmujoco_studio_platform.a" || \
          -f "${TMPDIR}/mujoco_install/lib/mujoco_studio_platform.lib" || \
          -f "${TMPDIR}/mujoco_install/bin/Release/mujoco_studio_platform.lib" ]]; then
        sdists+=(mujoco_studio-*.tar.gz mujoco-*.tar.gz)
    fi
    if [[ -f "${TMPDIR}/mujoco_install/lib/libmujoco_usd.so" || \
          -f "${TMPDIR}/mujoco_install/lib64/libmujoco_usd.so" || \
          -f "${TMPDIR}/mujoco_install/lib/libmujoco_usd.dylib" || \
          -f "${TMPDIR}/mujoco_install/lib/mujoco_usd.lib" || \
          -f "${TMPDIR}/mujoco_install/bin/mujoco_usd.dll" || \
          -f "${TMPDIR}/mujoco_install/bin/Release/mujoco_usd.dll" ]]; then
        sdists+=(mujoco_usd-*.tar.gz)
    fi
    MUJOCO_PATH="${TMPDIR}/mujoco_install" \
    MUJOCO_PLUGIN_PATH="${TMPDIR}/mujoco_install/mujoco_plugin" \
    MUJOCO_CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF -DFETCHCONTENT_BASE_DIR=${py_deps_dir}/_deps ${CCACHE_ARGS} ${CMAKE_ARGS} ${ext_deps_args[*]}" \
    pip wheel -v --no-deps --no-build-isolation "${sdists[@]}"
    _populate_ext_deps_cache "${py_deps_dir}"
    rm -rf "${py_deps_dir}"
}


install_python_bindings() {
    echo "Installing Python bindings..."
    source ${TMPDIR}/venv/bin/activate &&
    pip install --no-deps --no-index mujoco_core-*.whl &&
    if compgen -G "mujoco_studio-*.whl" > /dev/null; then
        pip install --no-deps --no-index mujoco_studio-*.whl mujoco-*.whl
    fi &&
    pip check &&
    if compgen -G "mujoco_usd-*.whl" > /dev/null; then
        pip install --no-deps --no-index mujoco_usd-*.whl
    fi
}


smoke_test_python_bindings() {
    echo "Smoke testing Python bindings..."
    source ${TMPDIR}/venv/bin/activate
    local doc_dir=""
    for d in "${TMPDIR}"/venv/[lL]ib*/python*/site-packages/mujoco/doc \
             "${TMPDIR}"/venv/[lL]ib*/pypy*/site-packages/mujoco/doc \
             "${TMPDIR}"/venv/[lL]ib/site-packages/mujoco/doc; do
        if [[ -d "${d}" ]]; then
            doc_dir="${d}"
            break
        fi
    done
    [[ -f "${doc_dir}/index.rst" ]]
    [[ -f "${doc_dir}/APIreference/functions.rst" ]]
    [[ -f "${doc_dir}/skills/python/SKILL.md" ]]
    python -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.mj_step(m, mujoco.MjData(m))"
    if _has_studio_platform build; then
        echo "Smoke testing Studio Python modules..."
        python -c "import mujoco._render_filament, mujoco.studio.window, mujoco.studio.ux, mujoco.studio.sim, mujoco.studio.native_viewer"
        if [[ "${RUNNER_OS:-}" != "Windows" ]]; then
            python -c "import mujoco.studio.web.headless_ui, mujoco.studio.web.state_payload"
        fi
    fi
}


test_python_bindings() {
    echo "Testing Python bindings..."
    source ${TMPDIR}/venv/bin/activate
    pytest -v --pyargs mujoco
    smoke_test_python_bindings
}


# Builds the WASM bindings for one threading mode into wasm/dist.
_build_wasm_threads() {  # $1 = ON | OFF
    local mode=$1 dir
    [[ "${mode}" == "ON" ]] && dir=build_wasm_mt || dir=build_wasm_st
    local ext_deps_args=()
    _setup_ext_deps_args "${dir}"
    echo "Build MuJoCo with Emscripten (MUJOCO_WASM_THREADS=${mode})..."
    emcmake cmake -B "${dir}" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DMUJOCO_WASM_THREADS=${mode} \
        ${CCACHE_ARGS} \
        $WASM_CMAKE_ARGS \
        "${ext_deps_args[@]}"
    _populate_ext_deps_cache "${dir}"
    cmake --build "${dir}" --parallel $(nproc)
}


# Runs the bindings tests against whatever is in wasm/dist.
test_wasm() {
    echo "Testing WASM bindings..."
    export PATH="$(pwd)/node_modules/.bin:$PATH"
    npm run test --prefix ./wasm
}


# Builds both threading modes without testing: multi-threaded under
# wasm/dist/mt, single-threaded in wasm/dist.
build_wasm() {
    echo "Building WASM bindings..."
    source emsdk/emsdk_env.sh
    export PATH="$(pwd)/node_modules/.bin:$PATH"
    _build_wasm_threads ON
    echo "Moving Multi-Thread version under mt subfolder..."
    mkdir -p wasm/dist/mt
    mv wasm/dist/mujoco.* wasm/dist/mt/
    _build_wasm_threads OFF
}


# CI: builds and tests both threading modes, testing each before the next
# build replaces wasm/dist.
build_test_wasm() {
    echo "Building and testing WASM bindings..."
    source emsdk/emsdk_env.sh
    export PATH="$(pwd)/node_modules/.bin:$PATH"
    _build_wasm_threads ON
    echo "Run bindings tests for Multi-Threaded version..."
    test_wasm
    echo "Moving Multi-Thread version under mt subfolder..."
    mkdir -p wasm/dist/mt
    mv wasm/dist/mujoco.* wasm/dist/mt/
    _build_wasm_threads OFF
    echo "Run bindings tests for Single-Threaded version..."
    test_wasm
}

package_wasm() {
    echo "Publishing WASM bindings..."
    cp wasm/package.npm.json wasm/dist/package.json
    cp wasm/README.md wasm/dist/README.md
    VERSION="${VERSION:-${GITHUB_REF#refs/tags/}}"
    npm --prefix wasm/dist version "${VERSION}" --no-git-tag-version
    npm pack --dry-run ./wasm/dist
    npm publish ./wasm/dist --access public --provenance
}


package_mjx() {
    echo "Packaging MJX..."
    source ${TMPDIR}/venv/bin/activate &&
    python -m build --no-isolation .
}


install_mjx() {
    echo "Installing MJX..."
    source ${TMPDIR}/venv/bin/activate
    # The MJX requirements (jax, jaxlib, scipy, ...) are a big install; use uv when
    # available. Keep pip for the local --no-index wheel.
    _pip_install_requirements requirements.txt
    pip install --no-deps --no-index dist/mujoco_mjx-*.whl
    pip check
}


test_mjx() {
    echo "Testing MJX..."
    source ${TMPDIR}/venv/bin/activate &&
    pytest -n auto -v -k 'not IntegrationTest' --pyargs mujoco.mjx
}





build_mujoco_live() {
    echo "Setting up Emscripten SDK..."
    source emsdk/emsdk_env.sh

    echo "Building Filament tools, targeting host platform..."
    local ext_deps_args=()
    _setup_ext_deps_args build_host
    cmake -S . -B build_host -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_STATIC_LIBCXX=OFF \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_TEST_PYTHON_UTIL=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF \
        "${ext_deps_args[@]}"
    _populate_ext_deps_cache build_host
    cmake --build build_host --target matc resgen cmgen mujoco_filament_assets -j"${NJOBS}"

    echo "Building desktop WASM app..."
    _setup_ext_deps_args build_wasm
    emcmake cmake -S . -B build_wasm -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS_WASM=OFF \
        -DMUJOCO_NATIVE_BUILD_DIR=$(pwd)/build_host \
        "${ext_deps_args[@]}"
    cmake --build build_wasm --target mujoco_studio -j"${NJOBS}"

    # iOS WebKit cannot reliably reserve the desktop build's 4 GiB shared
    # memory. Keep a separate non-threaded profile below its practical limit.
    echo "Building iOS WASM app..."
    _setup_ext_deps_args build_wasm_ios
    emcmake cmake -S . -B build_wasm_ios -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS_WASM=OFF \
        -DMUJOCO_WASM_THREADS=OFF \
        -DMUJOCO_STUDIO_WASM_INITIAL_MEMORY=256mb \
        -DMUJOCO_STUDIO_WASM_MAXIMUM_MEMORY=512mb \
        -DMUJOCO_NATIVE_BUILD_DIR=$(pwd)/build_host \
        "${ext_deps_args[@]}"
    cmake --build build_wasm_ios --target mujoco_studio -j"${NJOBS}"
}


# ---------------------------------------------------------------------------
# Modular developer steps for MuJoCo Studio and the self-contained wheel.
#
# All of them run from the repository top level and use these trees:
#   build/                the engine build (build_engine) or a configure_studio tree
#   build_simulate/       the engine + classic simulate app, no Filament
#   build_host/           the native Studio host build: platform library, Filament
#                         tools, staged assets, engine plugins, the mujoco_studio app
#   build_wasm/           the Emscripten build of the browser client
#   build/mujoco_install/ the SDK the Python extensions compile against
#
# The primitives are idempotent and incremental, so the workflows are flat
# lists of primitives, any step can be run in any order, and a repeated step
# costs seconds rather than a rebuild.
# ---------------------------------------------------------------------------

# --- primitives ------------------------------------------------------------

# Packages an sdist and builds the wheel from it against the SDK at $1, into
# python/dist. The wheel is always compiled from the sdist in pip's own
# temporary tree, so this is a full build rather than an incremental one.
_build_python_wheel() {
    local prefix="$1"
    # A clean python/dist keeps the wheel glob below unambiguous.
    rm -rf python/dist python/studio/dist python/usd/dist python/mujoco_full/dist
    (cd python && ./make_sdist.sh)
    local sdists=(python/dist/mujoco_core-*.tar.gz)
    if _has_studio_platform "${prefix}"; then
        (cd python/studio && ./make_sdist.sh)
        (cd python/mujoco_full && ./make_sdist.sh)
        mv python/studio/dist/mujoco_studio-*.tar.gz python/mujoco_full/dist/mujoco-*.tar.gz python/dist/
        sdists+=(python/dist/mujoco_studio-*.tar.gz python/dist/mujoco-*.tar.gz)
    fi
    if [[ -f "${prefix}/lib/libmujoco_usd.so" || \
          -f "${prefix}/lib64/libmujoco_usd.so" || \
          -f "${prefix}/lib/libmujoco_usd.dylib" || \
          -f "${prefix}/lib/mujoco_usd.lib" || \
          -f "${prefix}/bin/mujoco_usd.dll" || \
          -f "${prefix}/bin/Release/mujoco_usd.dll" ]]; then
        (cd python/usd && ./make_sdist.sh)
        mv python/usd/dist/mujoco_usd-*.tar.gz python/dist/
        sdists+=(python/dist/mujoco_usd-*.tar.gz)
    fi

    # See build_python_bindings for why CCACHE_BASEDIR/SLOPPINESS are set.
    # ccache is what keeps repeated wheel builds bearable.
    if command -v cygpath >/dev/null 2>&1; then
        export CCACHE_BASEDIR="$(cygpath -m "${TMPDIR:-$(pwd)}")"
    else
        export CCACHE_BASEDIR="$(cd "${TMPDIR:-$(pwd)}" && pwd -P)"
    fi
    export CCACHE_NOHASHDIR=1
    export CCACHE_SLOPPINESS="time_macros,include_file_mtime,include_file_ctime,pch_defines,locale"
    local py_deps_dir="${CCACHE_BASEDIR}/python_cmake_deps"
    rm -rf "${py_deps_dir}"
    mkdir -p "${py_deps_dir}/_deps"
    local ext_deps_args=()
    _setup_ext_deps_args
    MUJOCO_PATH="${prefix}" \
    MUJOCO_PLUGIN_PATH="${prefix}/mujoco_plugin" \
    MUJOCO_CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF -DFETCHCONTENT_BASE_DIR=${py_deps_dir}/_deps ${CCACHE_ARGS} ${CMAKE_ARGS} ${ext_deps_args[*]}" \
    pip wheel -v --no-deps -w python/dist "${sdists[@]}"
    _populate_ext_deps_cache "${py_deps_dir}"
    rm -rf "${py_deps_dir}"
}


# _build_python_wheel followed by an install into the active virtualenv
# (VIRTUAL_ENV or CONDA_DEFAULT_ENV); without one it prints a note and skips.
_install_python_wheel() {
    local prefix="$1"
    if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
        echo "NOTE: no active virtualenv (VIRTUAL_ENV / CONDA_DEFAULT_ENV); skipping the Python install."
        return 0
    fi
    echo "Building and installing the Python bindings against ${prefix}..."
    _build_python_wheel "${prefix}"
    # Install build/runtime dependencies (including websockets for the web
    # viewer server) with hash verification, then --force-reinstall --no-deps
    # --no-index to replace whatever mujoco is installed with the one just built
    # without disturbing the rest of the environment.
    _pip_install_requirements python/build_requirements.txt
    pip install --force-reinstall --no-deps --no-index python/dist/mujoco_core-*.whl
    if compgen -G "python/dist/mujoco_studio-*.whl" > /dev/null; then
        pip install --force-reinstall --no-deps --no-index python/dist/mujoco_studio-*.whl python/dist/mujoco-*.whl
    fi
    if compgen -G "python/dist/mujoco_usd-*.whl" > /dev/null; then
        pip install --force-reinstall --no-deps --no-index python/dist/mujoco_usd-*.whl
    fi
}



# True when the build tree $1 holds the Studio platform library.
_has_studio_platform() {
    [[ -f "$1/lib/libmujoco_studio_platform.a" || \
       -f "$1/lib/mujoco_studio_platform.lib" || \
       -f "$1/bin/Release/mujoco_studio_platform.lib" ]]
}


# Prints the tree holding the native Studio host build (platform library and
# Filament tools): MUJOCO_NATIVE_BUILD_DIR when set, else build/ when it was
# configured with Studio, else build_host/.
_resolve_studio_host_dir() {
    if [[ -n "${MUJOCO_NATIVE_BUILD_DIR:-}" && -d "${MUJOCO_NATIVE_BUILD_DIR}" ]]; then
        echo "${MUJOCO_NATIVE_BUILD_DIR}"
    elif _has_studio_platform build; then
        echo "$(pwd)/build"
    else
        echo "$(pwd)/build_host"
    fi
}


# Configures build_host/ once (no-op afterwards).
_configure_studio_host() {
    if [[ -f build_host/CMakeCache.txt ]]; then
        return 0
    fi
    echo "Configuring the native Studio host build..."
    local ext_deps_args=()
    _setup_ext_deps_args build_host
    cmake -S . -B build_host -G Ninja \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DUSE_STATIC_LIBCXX=OFF \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_TEST_PYTHON_UTIL=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS} \
        "${ext_deps_args[@]}"
    _populate_ext_deps_cache build_host
}


# The native Studio platform without the app: the platform library (its
# closure includes the Filament tools matc, resgen and cmgen the browser
# client build imports), the staged runtime assets (fonts + compiled
# materials, target mujoco_studio_assets) and the engine plugins the wheel
# ships. main.cc / launcher.cc are never compiled here. A host build provided
# by MUJOCO_NATIVE_BUILD_DIR or by a Studio build in build/ is used as is.
build_studio_platform_libs() {
    local host_dir
    host_dir="$(_resolve_studio_host_dir)"
    if [[ "${host_dir}" != "$(pwd)/build_host" ]]; then
        echo "Using the existing Studio host build in ${host_dir}"
        return 0
    fi
    _configure_studio_host
    echo "Building the Studio platform libraries, assets and engine plugins..."
    cmake --build build_host -j"${NJOBS}" \
        --target mujoco_studio_platform mujoco_studio_assets matc resgen cmgen actuator sensor sdf_plugin
}


# The C++ Studio viewer app (mujoco_studio) with its plugins, in the host
# tree. The fast path for C engine and C++ GUI work: an edit recompiles its
# translation unit and relinks.
build_studio_cpp_viewer() {
    local host_dir
    host_dir="$(_resolve_studio_host_dir)"
    if [[ "${host_dir}" == "$(pwd)/build_host" ]]; then
        _configure_studio_host
    fi
    echo "Building the C++ Studio viewer app (mujoco_studio) in ${host_dir}..."
    cmake --build "${host_dir}" -j"${NJOBS}" \
        --target mujoco_studio actuator sensor sdf_plugin
}


# Gathers the headers, static archives, plugins and assets the Python
# extensions compile against from the host build into ${TMPDIR:-build}/mujoco_install/.
install_studio_platform() {
    echo "Gathering the headers + libraries the Studio Python modules compile against..."
    local build_dir install_dir prefix deps plugin_ext plugin_dir
    build_dir="$(_resolve_studio_host_dir)"
    install_dir="${TMPDIR:-build}/mujoco_install"
    # Recreate the prefix every time. cmake --install and the copies below only
    # ever add files, so after a version bump the previous engine library
    # (libmujoco.so.<old version>) would stay here next to the new one, and
    # setup.py, which packages every libmujoco.so.* it finds, would ship both.
    # The copies keep timestamps, so repeated Python builds stay incremental.
    rm -rf "${install_dir}"
    mkdir -p "${install_dir}"
    prefix="$(cd "${install_dir}" && pwd)"
    # Per-OS engine-plugin layout.
    case "$(uname -s)" in
        Darwin) plugin_ext="dylib"; plugin_dir="${build_dir}/lib" ;;
        MINGW*|MSYS*|CYGWIN*) plugin_ext="dll"; plugin_dir="${build_dir}/bin" ;;
        *) plugin_ext="so"; plugin_dir="${build_dir}/lib" ;;
    esac

    # Portable copy helpers. They preserve timestamps (cp -p): the Python
    # extensions depend on these headers, so a copy that looked new would make
    # every rebuild recompile everything.
    _copy_headers() {  # $1=src dir, $2=dst dir — only *.h/*.inl, keep subdirs
        local s="${1%/}"
        (cd "${s}" && find . \( -name '*.h' -o -name '*.inl' \) -print0 |
            while IFS= read -r -d '' f; do
                mkdir -p "$2/${f%/*}" && cp -p "${f}" "$2/${f}"
            done)
    }
    _copy_tree() {  # $1=src dir, $2=dst dir — the whole subtree
        local s="${1%/}"
        mkdir -p "$2" && cp -rp "${s}/." "$2/"
    }

    # 1. Standard install: libmujoco (runtime), headers + archives (dev), the
    #    export and models (Unspecified). Not the studio component: it holds
    #    the app, which the Python builds do not need and may not have built.
    local component
    for component in runtime dev Unspecified; do
        cmake --install "${build_dir}" --prefix "${prefix}" --component "${component}"
    done

    # 2. Engine plugins (setup.py packages them from MUJOCO_PLUGIN_PATH).
    mkdir -p "${prefix}/mujoco_plugin"
    for plugin in actuator sensor sdf_plugin; do
        find "${plugin_dir}" -name "*${plugin}.${plugin_ext}" \
            -exec cp {} "${prefix}/mujoco_plugin/" \; 2>/dev/null || true
    done

    # 3. Static archives: mujoco_platform + every dependency archive; the Python
    #    build looks each one up by name with find_library().
    mkdir -p "${prefix}/lib"
    find "${build_dir}" -path "${prefix}" -prune -o \
        \( -name "*.a" -o -name "*.lib" \) ! -name "mujoco.lib" \
        -exec cp -p {} "${prefix}/lib/" \;

    # 4. Source-tree headers for studio / platform / filament-compat / render.
    _copy_headers studio "${prefix}/include/mujoco/studio"
    _copy_headers src/experimental "${prefix}/include/mujoco/experimental"
    _copy_headers src/render "${prefix}/include/mujoco/render"

    # 5. Third-party headers.
    deps="${build_dir}/_deps"
    # Dear ImGui.
    cp -p "${deps}/dear_imgui-src/"im*.h "${prefix}/include/"
    mkdir -p "${prefix}/include/misc/cpp"
    cp -p "${deps}/dear_imgui-src/misc/cpp/imgui_stdlib.h" "${prefix}/include/misc/cpp/"
    mkdir -p "${prefix}/include/backends"
    cp -p "${deps}/dear_imgui-src/backends/"imgui_impl_{sdl2,opengl3}.h \
        "${prefix}/include/backends/" 2>/dev/null || true
    # ImPlot.
    cp -p "${deps}/implot-src/"implot*.h "${prefix}/include/"
    # SDL2.
    mkdir -p "${prefix}/include/SDL2"
    cp -p "${deps}/sdl2-src/include/"*.h "${prefix}/include/SDL2/"
    cp -fp "${deps}/sdl2-build/include/"*.h "${prefix}/include/SDL2/" 2>/dev/null || true
    cp -fp "${deps}/sdl2-build/include-config-"*/*.h "${prefix}/include/SDL2/" 2>/dev/null || true
    # Filament support libraries (math/, utils/, filament/, backend/, ...).
    for lib in math utils filament backend filabridge ibl; do
        [[ -d "${deps}/filament-src/libs/${lib}/include/" ]] &&
            _copy_tree "${deps}/filament-src/libs/${lib}/include" "${prefix}/include"
    done
    _copy_tree "${deps}/filament-src/filament/include" "${prefix}/include"
    _copy_tree "${deps}/filament-src/filament/backend/include" "${prefix}/include"

    # 6. Studio runtime assets (fonts + Filament materials), staged by the
    #    mujoco_studio_assets target.
    if [[ ! -d "${build_dir}/bin/assets" ]]; then
        echo "ERROR: ${build_dir}/bin/assets is missing; build the mujoco_studio_assets target first" >&2
        return 1
    fi
    mkdir -p "${prefix}/assets"
    cp -rp "${build_dir}/bin/assets/." "${prefix}/assets/"

    echo "Gathered Studio platform compile inputs at ${prefix}"
}


# The web viewer browser client (WASM) in
# python/mujoco/studio/web/dist/. The output is
# platform-independent. It needs the host Filament tools, so the platform
# libraries come first (a no-op once built). Installs Emscripten into ./emsdk
# when it is missing. Configures build_wasm/ once; Ninja re-runs CMake by
# itself when a CMakeLists changes, and the tree is reconfigured when it was
# built against another host tree.
build_studio_wasm() {
    if [[ ! -d emsdk ]]; then
        setup_emsdk
    fi
    source emsdk/emsdk_env.sh

    local host_dir
    host_dir="$(_resolve_studio_host_dir)"
    if [[ ! -x "${host_dir}/bin/matc" || ! -x "${host_dir}/bin/resgen" || ! -x "${host_dir}/bin/cmgen" || ! -d "${host_dir}/src/render/filament/assets" ]]; then
        if [[ "${host_dir}" == "$(pwd)/build_host" ]]; then
            _configure_studio_host
        fi
        echo "Building Filament host tools and materials in ${host_dir}..."
        cmake --build "${host_dir}" -j"${NJOBS}" --target matc resgen cmgen mujoco_filament_assets
    fi

    if [[ ! -f build_wasm/CMakeCache.txt ]] ||
       ! grep -q "^MUJOCO_NATIVE_BUILD_DIR:[A-Z]*=${host_dir}\$" build_wasm/CMakeCache.txt; then
        echo "Configuring the web viewer browser client (WASM) with host tools from ${host_dir}..."
        local ext_deps_args=()
        _setup_ext_deps_args build_wasm "${host_dir}/_deps"
        emcmake cmake -S . -B build_wasm -G Ninja \
            -DCMAKE_BUILD_TYPE:STRING=Release \
            -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
            -DMUJOCO_BUILD_STUDIO=ON \
            -DMUJOCO_USE_FILAMENT=ON \
            -DMUJOCO_BUILD_TESTS_WASM=OFF \
            -DMUJOCO_NATIVE_BUILD_DIR:PATH="${host_dir}" \
            ${CCACHE_ARGS} \
            "${ext_deps_args[@]}"
    fi
    echo "Building the web viewer browser client (WASM)..."
    cmake --build build_wasm --target web_client -j"${NJOBS}"
}


# --- workflows -------------------------------------------------------------

# Builds libmujoco into build/, installs it with the engine plugins under
# ${TMPDIR:-build}/mujoco_install, then builds and installs the core Python
# bindings (Studio modules are skipped: this SDK has no Studio platform).
build_engine() {
    echo "Building the MuJoCo engine..."
    local prefix="${TMPDIR:-$(pwd)/build}/mujoco_install"
    # See configure_mujoco for why global IPO is off while CMakeLists.txt keeps
    # IPO on the core mujoco target on Windows MSVC.
    local ipo_off="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF"
    local ext_deps_args=()
    _setup_ext_deps_args build
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        ${ipo_off} \
        -DCMAKE_INSTALL_PREFIX:PATH="${prefix}" \
        -DMUJOCO_BUILD_EXAMPLES:BOOL=OFF \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS} \
        "${ext_deps_args[@]}"
    _populate_ext_deps_cache build
    cmake --build build -j"${NJOBS}"
    cmake --install build --prefix "${prefix}"

    mkdir -p "${prefix}/mujoco_plugin"
    local plugin
    for plugin in actuator sensor sdf_plugin; do
        find build/lib build/bin -maxdepth 2 \
            \( -name "lib${plugin}.so" -o -name "lib${plugin}.dylib" -o -name "${plugin}.dll" \) \
            -exec cp {} "${prefix}/mujoco_plugin/" \; 2>/dev/null || true
    done

    _install_python_wheel "${prefix}"
}


# The engine and the classic simulate app, no Filament, Dear ImGui, Emscripten
# or Python: the fastest loop for C engine work. Its own tree, build_simulate/,
# so it never collides with build/ (engine + tests, or a configure_studio
# tree). CI's configure_simulate / build_simulate are a different thing: they
# build the simulate/ sample against an installed MuJoCo.
build_simulate_app() {
    if [[ ! -f build_simulate/CMakeCache.txt ]]; then
        echo "Configuring the engine + simulate build..."
        local ipo_off="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF"
        cmake -S . -B build_simulate -G Ninja \
            -DCMAKE_BUILD_TYPE:STRING=Release \
            ${ipo_off} \
            -DMUJOCO_BUILD_SIMULATE=ON \
            -DMUJOCO_BUILD_STUDIO=OFF \
            -DMUJOCO_BUILD_TESTS=OFF \
            -DMUJOCO_TEST_PYTHON_UTIL=OFF \
            -DMUJOCO_BUILD_EXAMPLES=OFF \
            ${CCACHE_ARGS} \
            ${CMAKE_ARGS}
    fi
    echo "Building the engine and simulate..."
    cmake --build build_simulate --target simulate -j"${NJOBS}"
    echo "Built build_simulate/bin/simulate"
}


# The Python Studio viewer: platform libraries, SDK, then the Studio Python
# modules (ux, sim, window, dear_imgui, implot, headless_ui, state_payload,
# _render_filament) built into a wheel and installed. The browser client and
# the C++ app are not built.
build_studio_python() {
    echo "Building the Studio platform for Python..."
    build_studio_platform_libs
    install_studio_platform
    _install_python_wheel "$(cd "${TMPDIR:-build}/mujoco_install" && pwd)"
}


# Everything Python-facing: build_studio_python plus the browser client, so the
# installed wheel also carries web/dist.
build_studio_all() {
    echo "Building the complete Studio suite..."
    build_studio_platform_libs
    build_studio_wasm
    install_studio_platform
    _install_python_wheel "$(cd "${TMPDIR:-build}/mujoco_install" && pwd)"
}


# Builds the self-contained wheel: platform libraries, SDK and browser client,
# then a clean sdist (it carries web/dist, see MANIFEST.in) and the wheel from
# it. Unlike build_studio_all it does not install the result. The C++ app is
# not built.
build_wheel() {
    echo "Building the self-contained MuJoCo Python wheel..."
    build_studio_platform_libs
    install_studio_platform
    build_studio_wasm
    _build_python_wheel "$(cd "${TMPDIR:-build}/mujoco_install" && pwd)"
}


test_bazel() {
    echo "Building and testing MuJoCo with Bazel..."
    local test_flags=()
    if [[ "$(uname)" == "Darwin" ]]; then
        # GitHub's macOS runners have no hardware OpenGL, which mujoco.cgl
        # requires, so skip the rendering tests there (like the CMake job's
        # MUJOCO_GL=disable). On Linux they use headless EGL.
        test_flags+=(--test_env=MUJOCO_GL=disable)
    fi
    bazel test --test_output=errors "${test_flags[@]}" //...

    echo "Building standalone sample Bzlmod module..."
    pushd sample > /dev/null
    bazel build //...
    popd > /dev/null
}


# Discover functions defined in this script by finding identifiers followed by
# "()" and capturing the identifier as a valid function name.
VALID_FUNCTIONS=()
while IFS= read -r func_name; do
  VALID_FUNCTIONS+=("$func_name")
done < <(grep -E '^[[:alnum:]_]+\(\)' "$0" | sed 's/().*$//')

# Exit with an error if the requested function is not found.
if [[ ! " ${VALID_FUNCTIONS[*]} " =~ " ${1} " ]]; then
    echo "Usage: $0 {$(IFS='|'; echo "${VALID_FUNCTIONS[*]}")}, got '$1'"
    exit 1
fi

# Set options to print the commands being run, and cause the script to exit with
# an error code if any command fails. Note we do this just before executing
# the requested function to avoid cluttering the output with the above command
# discovery code.
set -xe

# Execute the requested function.
"$1"
