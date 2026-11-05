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


# Detect the operating system
OS="$(uname -s)"
case "${OS}" in
    Linux*)     export OS_NAME="linux" ;;
    Darwin*)    export OS_NAME="macos" ;;
    MINGW* | CYGWIN* | MSYS*) export OS_NAME="windows" ;;
    *)          echo "Unsupported OS: ${OS}" >&2; exit 1 ;;
esac

# Parse arguments
do_configure=false
do_build=false
build_type="Release"
explicit_action=false

for arg in "$@"; do
    case "$arg" in
        configure|config)
            do_configure=true
            explicit_action=true
            ;;
        build)
            do_build=true
            explicit_action=true
            ;;
        debug|dbg)
            build_type="Debug"
            ;;
    esac
done

# If no specific action (configure/build) was requested, do both by default.
if [[ "$explicit_action" == false ]]; then
    do_configure=true
    do_build=true
fi

GIT_ROOT="${GIT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${GIT_ROOT}"

# Configure MuJoCo Studio
if [[ "$do_configure" == true ]]; then
    echo "Configuring MuJoCo Studio (${build_type})..."
    CMAKE_CONFIG_ARGS=(
        "-B build"
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DUSE_STATIC_LIBCXX=OFF"
        "-DBUILD_SHARED_LIB=OFF"
        "-DMUJOCO_USE_FILAMENT=ON"
        "-DMUJOCO_BUILD_EXAMPLES=OFF"
        "-DMUJOCO_BUILD_SIMULATE=OFF"
        "-DMUJOCO_BUILD_TESTS=OFF"
        "-DMUJOCO_TEST_PYTHON_UTIL=OFF"
        "-DMUJOCO_WITH_USD=OFF"
        "-DMUJOCO_BUILD_STUDIO=ON"
        "-DFILAMENT_SKIP_SAMPLES=ON"
        # Several dependencies generate deprecated warnings on MacOS.
        "-DCMAKE_CXX_FLAGS=\"-Wno-error=deprecated-declarations\""
        # This flag defines _ITERATOR_DEBUG_LEVEL=0 which conflicts with debug
        # builds.
        "-DFILAMENT_SHORTEN_MSVC_COMPILATION=OFF"
    )

    # Add user-defined CMAKE_ARGS at the end so they override other settings.
    if [[ -n "${CMAKE_ARGS}" ]]; then
        read -a cmake_args_arr <<<"$CMAKE_ARGS"
        CMAKE_CONFIG_ARGS+=("${cmake_args_arr[@]}")
    fi

    cmake ${CMAKE_CONFIG_ARGS[@]}

    echo "Configuring MuJoCo Studio (${build_type})... DONE"
fi

# Build MuJoCo Studio
if [[ "$do_build" == true ]]; then
    echo "Building MuJoCo Studio..."
    cmake --build build --config=${build_type} --target mujoco_studio --parallel
    echo "Building MuJoCo Studio... DONE"

    # Print the command to run the built MuJoCo Studio from the right directory.
    echo "Use the following command to run mujoco_studio"
    echo ""
    if [[ "${OS_NAME}" == "windows" ]]; then
        echo " cd ${GIT_ROOT}/build/bin && ./${build_type}/mujoco_studio.exe "
    else
        echo " cd ${GIT_ROOT}/build/bin && ./mujoco_studio "
    fi
fi
