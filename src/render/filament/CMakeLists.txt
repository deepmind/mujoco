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

set(MUJOCO_FILAMENT_TARGET_NAME mujoco_filament)

add_library(${MUJOCO_FILAMENT_TARGET_NAME} STATIC)
target_compile_definitions(${MUJOCO_FILAMENT_TARGET_NAME} PRIVATE MJ_STATIC)

target_sources(${MUJOCO_FILAMENT_TARGET_NAME}
    PRIVATE
        mjrfilament.cc
        core/builtins.cc
        core/builtins.h
        core/color_grading_options.cc
        core/color_grading_options.h
        core/filament_context.cc
        core/filament_context.h
        core/filament_platform_factory.cc
        core/filament_platform_factory.h
        core/imgui_editor.h
        core/imgui_editor.cc
        core/light.cc
        core/light.h
        core/material_manager.cc
        core/material_manager.h
        core/mesh.cc
        core/mesh.h
        core/object_manager.cc
        core/object_manager.h
        core/outliner.cc
        core/outliner.h
        core/reflection_manager.cc
        core/reflection_manager.h
        core/releaser.h
        core/render_target.cc
        core/render_target.h
        core/renderable.cc
        core/renderable.h
        core/scene_view.cc
        core/scene_view.h
        core/texture.cc
        core/texture.h
        core/types.h
        support/filament_util.h
        support/filament_util.cc
        support/imgui_bridge.h
        support/imgui_bridge.cc
        support/mesh_util.h
        support/mesh_util.cc
        support/mjrf_ptr.h
        support/model_decorations.h
        support/model_decorations.cc
        support/model_lights.h
        support/model_lights.cc
        support/model_renderables.h
        support/model_renderables.cc
        support/model_resources.h
        support/model_resources.cc
        support/renderer.h
        support/renderer.cc
)

target_include_directories(${MUJOCO_FILAMENT_TARGET_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

include(third_party_deps/dear_imgui)
include(third_party_deps/filament)
target_link_libraries(${MUJOCO_FILAMENT_TARGET_NAME}
    dear_imgui
    filament
    image       # provided by filament
    ktxreader   # provided by filament
)

add_library(mujoco::filament ALIAS ${MUJOCO_FILAMENT_TARGET_NAME})

# The "mujoco::filament" library needs access to some assets (e.g. filament
# material shaders) in order to do rendering. As such, you must also include
# the assets under MUJOCO_FILAMENT_ASSETS_DIR in your project. As part of
# the filament library configuration, you will provide a function that the
# library will use to load these assets.
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(OUTPUT_ASSETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets")
file(MAKE_DIRECTORY ${OUTPUT_ASSETS_DIR})

file(COPY
    "${ASSETS_DIR}/ibl.ktx"
    DESTINATION ${OUTPUT_ASSETS_DIR}
)

set(MATC_EXECUTABLE matc)
file(GLOB MATERIAL_FILES CONFIGURE_DEPENDS "${ASSETS_DIR}/*.mat")

foreach(INPUT_FILE ${MATERIAL_FILES})
    get_filename_component(MATERIAL_NAME ${INPUT_FILE} NAME_WE)
    get_filename_component(MATERIAL_FILE ${INPUT_FILE} NAME)
    set(OUTPUT_FILE "${OUTPUT_ASSETS_DIR}/${MATERIAL_NAME}.filamat")

    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        set(PRECOMPILED_FILE "${MUJOCO_NATIVE_BUILD_DIR}/src/render/filament/assets/${MATERIAL_NAME}.filamat")
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy
                ${PRECOMPILED_FILE}
                ${OUTPUT_FILE}
            DEPENDS ${PRECOMPILED_FILE}
            COMMENT "Copying precompiled material ${MATERIAL_NAME}.filamat"
        )
    else()
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${MATC_EXECUTABLE}
                --platform=all
                --api=vulkan
                --api=opengl
                --variant-filter skinning
                --optimize-size
                --output ${OUTPUT_FILE}
                ${INPUT_FILE}
            DEPENDS ${INPUT_FILE}
            DEPENDS matc
            COMMENT "Compiling ${MATERIAL_FILE}"
        )
    endif()

    list(APPEND MUJOCO_FILAMENT_ASSET_FILES ${OUTPUT_FILE})
endforeach()

list(APPEND MUJOCO_FILAMENT_ASSET_FILES "${OUTPUT_ASSETS_DIR}/ibl.ktx")

add_custom_target(mujoco_filament_assets DEPENDS ${MUJOCO_FILAMENT_ASSET_FILES})
add_dependencies(${MUJOCO_FILAMENT_TARGET_NAME} mujoco_filament_assets)

set(MUJOCO_FILAMENT_ASSETS_DIR ${OUTPUT_ASSETS_DIR})
