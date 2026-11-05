# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../..;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs;@loader_path/../Frameworks;@loader_path/../../Frameworks;${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../..;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs;${CMAKE_INSTALL_LIBDIR}")
else()
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
endif()
if(UNIX AND NOT APPLE)
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  foreach(
    _flag
    "-stdlib=libc++"
    "-nostdlib++"
    "-rtlib=compiler-rt"
    "-Wl,-Bstatic"
    "-lc++"
    "-Wl,-Bdynamic"
  )
    string(REPLACE "${_flag}" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  endforeach()
endif()
set(USD_DECODER_SRCS
    newton_tokens.cc
    newton_tokens.h
    usd_decoder.cc
    kinematic_tree.cc
    kinematic_tree.h
    material_parsing.cc
    material_parsing.h
)

add_library(usd_decoder_plugin SHARED ${USD_DECODER_SRCS})
target_include_directories(usd_decoder_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
if(APPLE)
  set_target_properties(usd_decoder_plugin PROPERTIES
    BUILD_RPATH "@loader_path;@loader_path/..;@loader_path/../..;@loader_path/../../pxr;@loader_path/../../pxr/.dylibs;@loader_path/../Frameworks;@loader_path/../../Frameworks;${USD_INSTALL_ROOT}"
  )
elseif(UNIX)
  set_target_properties(usd_decoder_plugin PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../..;$ORIGIN/../../pxr;$ORIGIN/../../usd_core.libs;${USD_INSTALL_ROOT}"
  )
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(usd_decoder_plugin PRIVATE
    -Wl,-lstdc++
    -Wl,--exclude-libs,libstdc++_nonshared.a
  )
endif()
target_link_libraries(usd_decoder_plugin PRIVATE
  mujoco
  mujoco_usd
  mjcPhysics
)
target_compile_options(usd_decoder_plugin PRIVATE
  ${AVX_COMPILE_OPTIONS}
  ${MUJOCO_MACOS_COMPILE_OPTIONS}
  ${EXTRA_COMPILE_OPTIONS}
  ${MUJOCO_CXX_FLAGS}
  -UPXR_PYTHON_SUPPORT_ENABLED
  -Wno-deprecated # pxr Tf lib uses deprecated header
)
target_link_options(usd_decoder_plugin PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)

# Install to mujoco_plugin directory in bin location so that it is picked up by simulate
# on startup.
install(
  TARGETS usd_decoder_plugin
  LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}/mujoco_plugin"
)

target_link_libraries(usd_decoder_plugin PRIVATE ${PXR_LIBRARIES} usd usdGeom usdPhysics usdShade gf tf ar vt kind)
