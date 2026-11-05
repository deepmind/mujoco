# Copyright 2025 DeepMind Technologies Limited
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

cmake_minimum_required(VERSION 3.16)

include(third_party_deps/atkinson_hyperlegible_mono)
include(third_party_deps/atkinson_hyperlegible_next)
include(third_party_deps/dear_imgui)
include(third_party_deps/font_awesome)
include(third_party_deps/implot)
include(third_party_deps/libwebp)
include(third_party_deps/sdl2)

set(MUJOCO_STUDIO_PLATFORM_TARGET_NAME mujoco_studio_platform)

add_library(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME} STATIC)

target_sources(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME}
    PRIVATE
        hal/filament_renderer.cc
        hal/filament_renderer.h
        hal/graphics_mode.cc
        hal/graphics_mode.h
        hal/window.cc
        hal/window.h
        io/resources.cc
        io/resources.h
        render/model_scene.cc
        render/model_scene.h
        sim/model_holder.cc
        sim/model_holder.h
        sim/sim_history.cc
        sim/sim_history.h
        sim/sim_profiler.cc
        sim/sim_profiler.h
        sim/step_control.cc
        sim/step_control.h
        ux/console.cc
        ux/console.h
        ux/enum_utils.h
        ux/file_dialog.h
        ux/fonts.cc
        ux/fonts.h
        ux/gui.cc
        ux/gui.h
        ux/gui_helpers.cc
        ux/gui_helpers.h
        ux/gui_spec.cc
        ux/gui_spec.h
        ux/imgui_widgets.cc
        ux/imgui_widgets.h
        ux/interaction.cc
        ux/interaction.h
        ux/picture_gui.h
        ux/picture_gui.cc
        ux/plugin.cc
        ux/plugin.h
        ux/spec_editor.cc
        ux/spec_editor.h
        ux/well_tempered.h
)

if(NOT WIN32)
  target_sources(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME}
      PRIVATE
          ux/object_launcher_plugin.cc
  )
endif()

if(APPLE)
    set_source_files_properties(hal/window_osx.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++")
    target_sources(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME} PRIVATE
        hal/window_osx.mm
        ux/file_dialog_cocoa.mm
    )
elseif(UNIX AND NOT APPLE)
    target_sources(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME} PRIVATE
        ux/file_dialog_zenity.cc
    )
elseif(WIN32)
    target_sources(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME} PRIVATE
        ux/file_dialog_win.cc
    )
else()
    target_sources(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME} PRIVATE
        ux/file_dialog.cc
    )
endif()

target_include_directories(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

mujoco_add_dear_imgui_sdl2_backend()

target_link_libraries(${MUJOCO_STUDIO_PLATFORM_TARGET_NAME}
    dear_imgui
    dear_imgui_SDL2
    dear_imgui_OpenGL3
    dear_imgui_stdlib
    implot
    mujoco::filament
    SDL2::SDL2-static
)

add_library(mujoco::platform ALIAS ${MUJOCO_STUDIO_PLATFORM_TARGET_NAME})

if(NOT TARGET mujoco_studio_assets)
    set(STUDIO_ASSETS_STAGE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets)
    add_custom_target(mujoco_studio_assets
        COMMAND ${CMAKE_COMMAND} -E make_directory "${STUDIO_ASSETS_STAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_BINARY_DIR}/_deps/atkinson_hyperlegible_next-src/fonts/variable/AtkinsonHyperlegibleNext[wght].ttf"
                "${CMAKE_BINARY_DIR}/_deps/font_awesome-src/fonts/fontawesome-webfont.ttf"
                "${CMAKE_BINARY_DIR}/_deps/atkinson_hyperlegible_mono-src/fonts/ttf/AtkinsonHyperlegibleMono-Regular.ttf"
                "${STUDIO_ASSETS_STAGE_DIR}"
        COMMENT "Staging Studio fonts into ${STUDIO_ASSETS_STAGE_DIR}")
    if(MUJOCO_USE_FILAMENT)
        add_custom_command(TARGET mujoco_studio_assets POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../src/render/filament/assets
                    "${STUDIO_ASSETS_STAGE_DIR}")
        add_dependencies(mujoco_studio_assets mujoco_filament_assets)
    endif()
endif()

# Common configuration shared between mujoco_studio and mujoco_live.
function(configure_studio_target TARGET_NAME)
    target_sources(${TARGET_NAME}
        PRIVATE
            app.cc
            app.h
    )

    target_include_directories(${TARGET_NAME}
        PUBLIC
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    if (WIN32)
        target_compile_definitions(${TARGET_NAME}
            PRIVATE
            -D_USE_MATH_DEFINES
        )
        set_target_properties(${TARGET_NAME}
            PROPERTIES
                VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
    endif()

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            absl::flags
            absl::flags_parse
            dear_imgui
            implot
            mujoco::mujoco
            mujoco::platform
            webp
    )

    # Self-registering plugin TUs (mjPLUGIN_LIB_INIT) have no referenced symbols,
    # so the linker discards them when they arrive inside a static archive.
    if(APPLE)
        target_link_options(${TARGET_NAME} PRIVATE
            "-Wl,-force_load,$<TARGET_FILE:mujoco_studio_platform>")
    elseif(NOT MSVC)
        target_link_options(${TARGET_NAME} PRIVATE
            "-Wl,--whole-archive,$<TARGET_FILE:mujoco_studio_platform>,--no-whole-archive")
    else()
        target_link_options(${TARGET_NAME} PRIVATE
            "/WHOLEARCHIVE:$<TARGET_FILE:mujoco_studio_platform>")
    endif()

    add_dependencies(${TARGET_NAME} mujoco_studio_assets)

    get_target_property(IS_BUNDLE ${TARGET_NAME} MACOSX_BUNDLE)
    if(IS_BUNDLE)
        add_custom_command(
            TARGET ${TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${STUDIO_ASSETS_STAGE_DIR}"
                    $<TARGET_FILE_DIR:${TARGET_NAME}>/../Resources/assets
            COMMENT "Copying staged assets to bundle Resources"
        )
    endif()
endfunction()

# Desktop Studio target.
if(NOT EMSCRIPTEN)
    if(APPLE)
        set(STUDIO_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco_studio.icns)
    elseif(WIN32)
        set(STUDIO_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco_studio.rc)
    else()
        set(STUDIO_RESOURCE_FILES "")
    endif()

    add_executable(mujoco_studio
        main.cc
        launcher.cc
        launcher.h
        ${STUDIO_RESOURCE_FILES}
    )
    if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
        set_target_properties(mujoco_studio PROPERTIES
            MACOSX_BUNDLE TRUE
            INSTALL_RPATH "@executable_path/../Frameworks;@executable_path/../../.."
            BUILD_WITH_INSTALL_RPATH TRUE
            RESOURCE "${STUDIO_RESOURCE_FILES}"
            MACOSX_BUNDLE_ICON_FILE "mujoco_studio.icns"
        )
    endif()
    if(WIN32)
        add_custom_command(
            TARGET mujoco_studio
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco_studio.ico
                    ${CMAKE_CURRENT_SOURCE_DIR}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_studio.ico
        )
    endif()
    configure_studio_target(mujoco_studio)
    if(NOT (APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS))
        include(TargetAddRpath)
        target_add_rpath(
          TARGETS
          mujoco_studio
          INSTALL_DIRECTORY
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
          LIB_DIRS
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
          DEPENDS
          MUJOCO_ENABLE_RPATH
        )
    endif()
    install(
      TARGETS mujoco_studio
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT studio
      BUNDLE DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT studio
    )
    get_target_property(IS_BUNDLE mujoco_studio MACOSX_BUNDLE)
    if(NOT IS_BUNDLE)
        install(
          DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets"
          DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT studio
        )
    endif()
endif()

# Browser WASM client for the Web Viewer.
if(EMSCRIPTEN)
    add_subdirectory(
        ${PROJECT_SOURCE_DIR}/python/mujoco/studio/web
        ${CMAKE_CURRENT_BINARY_DIR}/web
    )
endif()

# WASM Live target.
if(EMSCRIPTEN)
    set(
      MUJOCO_STUDIO_WASM_INITIAL_MEMORY
      "1024mb"
      CACHE STRING "Initial WebAssembly memory for MuJoCo Live"
    )
    set(
      MUJOCO_STUDIO_WASM_MAXIMUM_MEMORY
      "4gb"
      CACHE STRING "Maximum WebAssembly memory for MuJoCo Live"
    )

    add_executable(mujoco_studio)
    target_sources(mujoco_studio PRIVATE emscripten.cc)
    target_compile_options(mujoco_studio PRIVATE -g)
    configure_studio_target(mujoco_studio)
    target_link_libraries(mujoco_studio PRIVATE mujoco::render_noop)

    # Ensure resource decoder plugins are linked into the WASM binary.
    target_link_options(mujoco_studio PRIVATE
        "-Wl,--whole-archive,$<TARGET_FILE:mujoco>,--no-whole-archive"
    )

    # Filament's OpenGL backend requires WebGL2 / full ES3 bindings.
    target_link_options(mujoco_studio PRIVATE
        --bind
        -sUSE_WEBGL2=1
        -sFULL_ES3
        -sMIN_WEBGL_VERSION=2
        -sMAX_WEBGL_VERSION=2
        -sALLOW_MEMORY_GROWTH=1
        -sASYNCIFY=1
        -sFETCH=1
        -sGL_PREINITIALIZED_CONTEXT=1
        -sSTACK_SIZE=32mb
        -sINITIAL_MEMORY=${MUJOCO_STUDIO_WASM_INITIAL_MEMORY}
        -sMAXIMUM_MEMORY=${MUJOCO_STUDIO_WASM_MAXIMUM_MEMORY}
        -sASSERTIONS=1
        -fexceptions
        -g
    )
endif()
