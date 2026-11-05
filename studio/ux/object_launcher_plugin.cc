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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/plugin.h"

namespace mujoco::studio {

inline constexpr char kObjectLauncherName[] = "ObjectLauncher";

class ObjectLauncher {
 public:
  ObjectLauncher() : rng_(std::random_device{}()) {}

  void UpdateGui() {
    constexpr mjtNum kLifeDefault = 2.0;
    ImGui_SliderLog("Size", &size_, size_seed_ * 0.1, size_seed_ * 10);
    ImGui::BeginDisabled(size_ == size_seed_);
    if (ImGui_ResetButton("Size")) size_ = size_seed_;
    ImGui::EndDisabled();
    ImGui_SliderLog("Speed", &speed_, speed_seed_ * 0.1, speed_seed_ * 10);
    ImGui::BeginDisabled(speed_ == speed_seed_);
    if (ImGui_ResetButton("Speed")) speed_ = speed_seed_;
    ImGui::EndDisabled();
    ImGui_SliderLog("Mass", &mass_, mass_seed_ * 0.1, mass_seed_ * 100);
    ImGui::BeginDisabled(mass_ == mass_seed_);
    if (ImGui_ResetButton("Mass")) mass_ = mass_seed_;
    ImGui::EndDisabled();
    ImGui_SliderLog("Life", &lifetime_, 0.5, 50.0);
    ImGui::BeginDisabled(lifetime_ == kLifeDefault);
    if (ImGui_ResetButton("Life")) lifetime_ = kLifeDefault;
    ImGui::EndDisabled();

    int shape = type_ == mjGEOM_BOX ? 0 : 1;
    const char* names[] = {"Box", "Sphere"};
    ImGui::Combo("Shape", &shape, names, 2);
    type_ = shape == 0 ? mjGEOM_BOX : mjGEOM_SPHERE;

    if (ImGui::Button("Launch (Enter)", ImVec2(-1.0f, 0.0f))) {
      active_ = true;
    }

    if (ImGui::Button("Clear")) {
      for (auto& object : objects_) {
        object.expiration = -1;
      }
    }
  }

  void HandleKeyboardEvent() {
    // The key binding is armed only while the plugin window is open.
    ForEachPlugin<GuiPlugin>([this](auto* gui) {
      if (gui->active && !std::strcmp(gui->name, kObjectLauncherName)) {
        active_ = true;
      }
    });
  }

  void OnModelLoaded(const mjModel* model) {
    if (!model) return;
    // Geometric mean of two length estimates -- the mean geom size, and the
    // length implied by the mean mass at density 1000 -- so a bad meansize or
    // meanmass only half-poisons the seed (they agree for a consistent model).
    const mjtNum len_size = model->stat.meansize;
    const mjtNum len_mass = std::cbrt(model->stat.meanmass / 1000.0);
    mjtNum len = mju_sqrt(len_size * len_mass);
    if (!(len > 0)) len = len_size > 0 ? len_size : len_mass;
    const mjtNum size = 0.5 * len;
    Seed(size, &size_, &size_seed_);
    // Mass giving the default box (volume (2*size)^3) density 1000, MuJoCo's
    // default geom density. Units are unspecified, so 1000 is just a
    // convention.
    const mjtNum box_volume = 8.0 * size * size * size;
    Seed(1000.0 * box_volume, &mass_, &mass_seed_);
    const mjtNum gravity = mju_norm3(model->opt.gravity);
    const mjtNum g = gravity > mjMINVAL ? gravity : 9.81;
    // Ballistic scale: the launch arc spans a few model extents.
    Seed(3.0 * mju_sqrt(model->stat.extent * g), &speed_, &speed_seed_);
  }

  bool UpdateSpecPreCompile(mjSpec* spec, const mjModel* model, const mjData* data,
                            const mjvCamera* camera) {
    // Objects live on [creation_time, expiration]: drop them when the lifetime
    // elapses, or when the sim rewinds before launch (so a reset clears them).
    auto it = std::remove_if(objects_.begin(), objects_.end(), [&](const ObjectInfo& o) {
      if (o.body_id < 0) {
        return false;  // not yet compiled into the model
      }
      const bool expired = o.expiration != 0 && o.expiration < data->time;
      const bool rewound = data->time < o.creation_time;
      const bool remove = expired || rewound;
      if (remove) {
        mjsBody* body = mjs_findBody(spec, o.name.c_str());
        if (body) {
          mjs_delete(spec, body->element);
        }
      }
      return remove;
    });
    if (it != objects_.end()) {
      objects_.erase(it, objects_.end());
      return true;
    }

    if (!active_) return false;
    active_ = false;

    mjsBody* world = mjs_findBody(spec, "world");
    if (!world) return false;
    mjsBody* body = mjs_addBody(world, nullptr);
    if (!body) return false;
    mjsJoint* joint = mjs_addJoint(body, nullptr);
    if (!joint) return false;
    mjsGeom* geom = mjs_addGeom(body, nullptr);
    if (!geom) return false;

    ObjectInfo& object = objects_.emplace_back();
    object.name = "projectile" + std::to_string(counter_++);
    object.creation_time = data->time;
    object.expiration = data->time + lifetime_;

    mjtNum pos[3];
    mjtNum dir[3];
    mjv_cameraFrame(pos, dir, nullptr, nullptr, data, camera);
    mjs_setName(body->element, object.name.c_str());

    joint->type = mjJNT_FREE;
    body->mass = mass_;
    geom->type = type_;
    geom->size[0] = size_;
    geom->size[1] = size_;
    geom->size[2] = size_;
    // Slightly in front of the camera.
    body->pos[0] = pos[0] + (dir[0] * 0.1);
    body->pos[1] = pos[1] + (dir[1] * 0.1);
    body->pos[2] = pos[2] + (dir[2] * 0.1);
    // Randomize the color.
    geom->rgba[0] = std::uniform_real_distribution<float>(0.3f, 1.0f)(rng_);
    geom->rgba[1] = std::uniform_real_distribution<float>(0.3f, 1.0f)(rng_);
    geom->rgba[2] = std::uniform_real_distribution<float>(0.3f, 1.0f)(rng_);
    geom->rgba[3] = 1.0;
    launch_vel_[0] = dir[0] * speed_;
    launch_vel_[1] = dir[1] * speed_;
    launch_vel_[2] = dir[2] * speed_;
    return true;
  }

  void UpdateSpecPostCompile(const mjSpec* spec, const mjModel* model, mjData* data) {
    if (objects_.empty()) {
      return;
    }

    ObjectInfo& object = objects_.back();
    if (object.launched) {
      return;
    }
    object.launched = true;

    const int body_id = mj_name2id(model, mjOBJ_BODY, object.name.c_str());
    if (body_id < 0) {
      return;
    }
    int joint_id = model->body_jntadr[body_id];
    if (joint_id < 0 || model->jnt_type[joint_id] != mjJNT_FREE) {
      return;
    }
    int qvel_addr = model->jnt_dofadr[joint_id];
    if (qvel_addr < 0) {
      return;
    }
    object.body_id = body_id;
    data->qvel[qvel_addr + 0] = launch_vel_[0];
    data->qvel[qvel_addr + 1] = launch_vel_[1];
    data->qvel[qvel_addr + 2] = launch_vel_[2];
  }

 private:
  // Update value with the new seed, unless the user has edited it.
  static void Seed(mjtNum seed, mjtNum* value, mjtNum* prev_seed) {
    if (seed <= 0 || !std::isfinite(seed)) return;
    if (*value == *prev_seed) *value = seed;
    *prev_seed = seed;
  }

  struct ObjectInfo {
    std::string name;
    int body_id = -1;
    mjtNum creation_time = 0;
    mjtNum expiration = 0;
    bool launched = false;
  };

  std::mt19937 rng_;
  int counter_ = 0;
  bool active_ = false;
  mjtNum size_ = 0.1;
  mjtNum speed_ = 10.0;
  mjtNum mass_ = 10.0;
  mjtNum lifetime_ = 2.0;
  mjtNum size_seed_ = 0.1;
  mjtNum speed_seed_ = 10.0;
  mjtNum mass_seed_ = 10.0;
  mjtGeom type_ = mjGEOM_BOX;
  mjtNum launch_vel_[3] = {0, 0, 0};
  std::vector<ObjectInfo> objects_;
};

}  // namespace mujoco::studio

mjPLUGIN_LIB_INIT(object_launcher) {
  using mujoco::studio::ObjectLauncher;

  static ObjectLauncher plugin;

  mujoco::studio::GuiPlugin gui;
  gui.data = &plugin;
  gui.name = mujoco::studio::kObjectLauncherName;
  gui.update = [](mujoco::studio::GuiPlugin* self) {
    auto* plugin = static_cast<mujoco::studio::ObjectLauncher*>(self->data);
    plugin->UpdateGui();
  };
  mujoco::studio::RegisterPlugin(gui);

  mujoco::studio::KeyHandlerPlugin key_handler;
  key_handler.data = &plugin;
  key_handler.name = mujoco::studio::kObjectLauncherName;
  key_handler.key_chord = ImGuiKey_Enter;
  key_handler.on_key_pressed = [](mujoco::studio::KeyHandlerPlugin* self) {
    auto* plugin = static_cast<mujoco::studio::ObjectLauncher*>(self->data);
    plugin->HandleKeyboardEvent();
  };
  mujoco::studio::RegisterPlugin(key_handler);

  mujoco::studio::ModelPlugin model_plugin;
  model_plugin.data = &plugin;
  model_plugin.name = mujoco::studio::kObjectLauncherName;
  model_plugin.post_model_loaded = [](mujoco::studio::ModelPlugin* self, const mjModel* model,
                                      const char* model_path) {
    auto* plugin = static_cast<mujoco::studio::ObjectLauncher*>(self->data);
    plugin->OnModelLoaded(model);
  };
  mujoco::studio::RegisterPlugin(model_plugin);

  mujoco::studio::SpecEditorPlugin spec_editor;
  spec_editor.data = &plugin;
  spec_editor.name = mujoco::studio::kObjectLauncherName;
  spec_editor.pre_compile = [](mujoco::studio::SpecEditorPlugin* self, mjSpec* spec,
                               const mjModel* model, const mjData* data, const mjvCamera* camera) {
    auto* plugin = static_cast<mujoco::studio::ObjectLauncher*>(self->data);
    return plugin->UpdateSpecPreCompile(spec, model, data, camera);
  };
  spec_editor.post_compile = [](mujoco::studio::SpecEditorPlugin* self, const mjSpec* spec,
                                const mjModel* model, mjData* data) {
    auto* plugin = static_cast<mujoco::studio::ObjectLauncher*>(self->data);
    return plugin->UpdateSpecPostCompile(spec, model, data);
  };
  mujoco::studio::RegisterPlugin(spec_editor);
}
