# Copyright 2026 DeepMind Technologies Limited
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

if(EMSCRIPTEN)
  # Browser WASM client for the Web Viewer
  add_executable(web_client
    logging.h
    role_transitions.cc
    role_transitions.h
    state_payload.cc
    state_payload.h
    ui_codec.cc
    ui_codec.h
    web_client.cc
    web_client_local_ui.cc
    web_client_local_ui.h
    web_client_remote_ui.cc
    web_client_remote_ui.h
    web_client_session.cc
    web_client_session.h
    web_client_websocket.cc
    web_client_websocket.h
  )

  target_include_directories(web_client
    PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src
  )

  target_link_libraries(web_client
    PRIVATE
      dear_imgui
      implot
      mujoco_filament
      mujoco::mujoco
      mujoco::platform
      mujoco::render_noop
      SDL2::SDL2-static
  )

  # Ensure resource decoder plugins are linked into the WASM binary.
  target_link_options(web_client PRIVATE
    -Wl,--whole-archive $<TARGET_FILE:mujoco> -Wl,--no-whole-archive
  )

  # Compile with exception support so the try/catch around MainLoop can catch
  # throws escaping the requestAnimationFrame callback. Emscripten disables
  # exception handling by default at compile time; the -fexceptions in the
  # link options below only covers linking.
  target_compile_options(web_client PRIVATE -fexceptions)

  # Filament's OpenGL backend requires WebGL2 / full ES3 bindings.
  target_link_options(web_client PRIVATE
    --bind
    # The browser WebSockets (emscripten/websocket.h) for /ui and /state.
    -lwebsocket.js
    -sASSERTIONS=0
    -sGL_ASSERTIONS=0
    -sFETCH=1
    -sALLOW_MEMORY_GROWTH=1
    # embind (EMSCRIPTEN_BINDINGS registerAsset/startApp) needs eval enabled.
    -sDYNAMIC_EXECUTION=1
    -sUSE_WEBGL2=1
    -sFULL_ES3=1
    -sMIN_WEBGL_VERSION=2
    -sMAX_WEBGL_VERSION=2
    -sSTACK_SIZE=32mb
    # INITIAL_MEMORY=3072mb reserves a 3GB contiguous virtual address block upfront. This means
    # large models do not fail dynamic memory.grow calls due to browser heap fragmentation. OS
    # lazy page allocation for anonymous virtual memory means reserving 3GB of virtual address
    # space consumes much less physical RAM when loading small XML models. Therefore, reducing
    # INITIAL_MEMORY is not necessary for small models. For models that don't fit in 32-bit
    # address space (~4GB) we can switch to 64-bit WASM.
    -sINITIAL_MEMORY=3072mb
    -sMAXIMUM_MEMORY=4gb
    -fexceptions
    # Filament's GL backend calls glGetError() on every draw call via even in non-debug builds
    # (the functions are compiled into the library unconditionally). In WebGL, each glGetError()
    # forces a synchronous GPU pipeline flush across the JS-WASM bridge (~0.2ms each, hundreds
    # per frame). Wrapping it at link time replaces all call sites with a no-op.
    -Wl,--wrap=glGetError
  )

  # Stage the Studio fonts in WEB_CLIENT_FONTS_DIR so the browser can load them.
  set(WEB_CLIENT_FONTS_DIR ${CMAKE_CURRENT_BINARY_DIR}/web_fonts)
  add_custom_target(web_client_fonts
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_FONTS_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_BINARY_DIR}/_deps/atkinson_hyperlegible_next-src/fonts/variable/AtkinsonHyperlegibleNext[wght].ttf"
      "${WEB_CLIENT_FONTS_DIR}/AtkinsonHyperlegibleNext[wght].ttf"
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/_deps/font_awesome-src/fonts/fontawesome-webfont.ttf
      ${WEB_CLIENT_FONTS_DIR}/fontawesome-webfont.ttf
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_BINARY_DIR}/_deps/atkinson_hyperlegible_mono-src/fonts/ttf/AtkinsonHyperlegibleMono-Regular.ttf"
      ${WEB_CLIENT_FONTS_DIR}/AtkinsonHyperlegibleMono-Regular.ttf
    COMMENT "Staging web viewer UI fonts"
  )

  # The fonts and Filament assets are served as plain files under dist/assets.
  # The page fetches them and pushes them into the client via registerAsset()
  # before startApp() runs (see index.html and web_client.cc).
  set(WEB_CLIENT_FILAMENT_ASSETS_DIR
      ${CMAKE_BINARY_DIR}/src/render/filament/assets
  )
  add_dependencies(web_client mujoco_filament_assets web_client_fonts)

  # Stage the browser artifacts into the source-tree dist/ folder, which
  # web_server.py serves by default (see _find_static_files_dir). Assets go in
  # dist/assets/ so the page can fetch them at /assets/<name>.
  set(WEB_CLIENT_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
  add_custom_command(
    TARGET web_client
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_DIST_DIR}/assets
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE_DIR:web_client>/web_client.js
        $<TARGET_FILE_DIR:web_client>/web_client.wasm
        ${CMAKE_CURRENT_SOURCE_DIR}/index.html
        ${PROJECT_SOURCE_DIR}/studio/favicon.ico
        ${WEB_CLIENT_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${WEB_CLIENT_FILAMENT_ASSETS_DIR} ${WEB_CLIENT_DIST_DIR}/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${WEB_CLIENT_FONTS_DIR} ${WEB_CLIENT_DIST_DIR}/assets
    COMMENT "Staging web viewer artifacts + assets into ${WEB_CLIENT_DIST_DIR}"
  )
else()
  if(WIN32)
    # Windows wheels ship without the browser WASM client (Kokoro Windows jobs
    # have no Emscripten toolchain).
    message(STATUS "Windows: skipping the Web Viewer Python modules.")
    return()
  endif()

  if(NOT TARGET mujoco::dep::implot)
    message(WARNING "mujoco::dep::implot target not found. The web viewer "
                    "modules will not be built.")
    return()
  endif()

  target_sources(_studio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/headless_ui.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/headless_ui_session.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/headless_ui_session.h
    ${CMAKE_CURRENT_SOURCE_DIR}/logging.h
    ${CMAKE_CURRENT_SOURCE_DIR}/state_payload_py.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/state_payload.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/state_payload.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ui_codec.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ui_codec.h
  )
  target_link_libraries(_studio PRIVATE
    mujoco::dep::dear_imgui
    mujoco::dep::implot
  )
  target_compile_definitions(_studio PRIVATE MUJOCO_STUDIO_WITH_WEB)
endif()
