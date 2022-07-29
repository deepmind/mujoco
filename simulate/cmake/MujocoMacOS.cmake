# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(APPLE)
  # 10.12 is the oldest version of macOS that supports C++17, launched 2016.
  set(MUJOCO_MACOSX_VERSION_MIN 10.12)

  # We are setting the -mmacosx-version-min compiler flag directly rather than using the
  # CMAKE_OSX_DEPLOYMENT_TARGET variable since we do not want to affect choice of SDK,
  # and also we only want to apply the version restriction locally.
  set(MUJOCO_MACOS_COMPILE_OPTIONS -mmacosx-version-min=${MUJOCO_MACOSX_VERSION_MIN}
                                   -Werror=partial-availability -Werror=unguarded-availability
  )
  set(MUJOCO_MACOS_LINK_OPTIONS -mmacosx-version-min=${MUJOCO_MACOSX_VERSION_MIN}
                                -Wl,-no_weak_imports
  )
else()
  set(MUJOCO_MACOS_COMPILE_OPTIONS "")
  set(MUJOCO_MACOS_LINK_OPTIONS "")
endif()

function(enforce_mujoco_macosx_min_version)
  if(APPLE)
    add_compile_options(${MUJOCO_MACOS_COMPILE_OPTIONS})
    add_link_options(${MUJOCO_MACOS_LINK_OPTIONS})
  endif()
endfunction()
