// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_STUDIO_UX_PLUGIN_H_
#define MUJOCO_STUDIO_UX_PLUGIN_H_

#include <functional>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Registers plugins with the global registry. The plugins must have a
// case-insensitive unique name for the plugin type. Note that plugins are
// copied by value, so do not use inheritance.
template <typename T>
void RegisterPlugin(T plugin);

// Executes the given function for each registered plugin of type T.
template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn);

// Plugin for processing custom UI windows. The plugin will be listed in the
// "Plugins" main menu and, when selected, an ImGui window will be opened with
// the name of the plugin as the title. The `update` function can then be used
// to process the GUI. All functions will be called by passing `this` as the
// first argument.
struct GuiPlugin final {
  using UpdateFn = void (*)(GuiPlugin* self);

  // Whether or not to display the plugin window.
  bool active = false;

  // The name of the plugin; must be unique.
  const char* name = "";

  // The function that will update the plugin's window. Plugin GUI updates
  // happen when the window is `active` and after all other Studio GUI updates.
  UpdateFn update = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin for adding additional scene geoms.
struct ScenePlugin final {
  using EnhanceSceneFn = void (*)(ScenePlugin* self, const mjModel* model, mjData* data,
                                  mjvScene* scene);

  // The name of the plugin; must be unique.
  const char* name = "";

  // This function will add additional mjvGeom objects to the mjvScene.
  EnhanceSceneFn enhance_scene = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin for loading and updating models.
struct ModelPlugin final {
  using GetModelToLoadFn = const char* (*)(ModelPlugin* self, int* size, char* content_type,
                                           int content_type_size, char* model_name,
                                           int model_name_size);
  using PostModelLoadedFn = void (*)(ModelPlugin* self, const mjModel* model,
                                     const char* model_path);
  using DoUpdateFn = bool (*)(ModelPlugin* self, mjModel* model, mjData* data);
  using PreStepFn = void (*)(ModelPlugin* self, const mjModel* model, mjData* data);
  using PostStepFn = void (*)(ModelPlugin* self, const mjModel* model, mjData* data);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback for when the plugin wants to load a new model. This function will
  // return a buffer containing the model data as well as the content type of
  // the buffer. If buf == model_name, then we assume that model_name stores the
  // path of the model to load. Returns nullptr if no model needs to be loaded.
  GetModelToLoadFn get_model_to_load = nullptr;

  // Callback when a new model is loaded.
  PostModelLoadedFn post_model_loaded = nullptr;

  // Callback when the physics simulation is updated. Returns true if the
  // simulation should be stepped.
  DoUpdateFn do_update = nullptr;

  // Callback immediately before mj_step is called on the model. This may be
  // called multiple times per update/frame.
  PreStepFn pre_step = nullptr;

  // Callback immediately after mj_step is called on the model. This may be
  // called multiple times per update/frame.
  PostStepFn post_step = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin for handling custom keyboard events.
struct KeyHandlerPlugin final {
  using OnKeyPressedFn = void (*)(KeyHandlerPlugin* self);

  // The name of the plugin; must be unique.
  const char* name = "";

  // The ImGui key codes for the key combination that triggers the plugin.
  int key_chord = 0;

  // The function to be called when the above key combination is pressed.
  OnKeyPressedFn on_key_pressed = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin for editing the mjSpec.
struct SpecEditorPlugin final {
  using PreCompileFn = bool (*)(SpecEditorPlugin* self, mjSpec* spec, const mjModel* model,
                                const mjData* data, const mjvCamera* camera);
  using PostCompileFn = void (*)(SpecEditorPlugin* self, const mjSpec* spec, const mjModel* model,
                                 mjData* data);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback that edits the spec. If it returns true, then the spec will be
  // recompiled and `post_compile` will be called with the result.
  PreCompileFn pre_compile = nullptr;

  // Callback that is called after the spec has been recompiled.
  PostCompileFn post_compile = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_PLUGIN_H_
