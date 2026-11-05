// Copyright 2025 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_BUILTINS_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_BUILTINS_H_

#include <memory>

#include <filament/Engine.h>
#include "render/filament/core/mesh.h"

namespace mujoco {

// A collection of meshes that "built in" to the renderer.
class Builtins {
 public:
  Builtins(filament::Engine* engine, int nstack, int nslice, int nquad);

  // Returns a mesh for the corresponding built-in shape.
  const Mesh* Line();
  const Mesh* LineBox();
  const Mesh* Plane();
  const Mesh* Haze();
  const Mesh* Triangle();
  const Mesh* Box();
  const Mesh* Sphere();
  const Mesh* Cone();
  const Mesh* Disk();
  const Mesh* DomeTop();
  const Mesh* DomeBottom();
  const Mesh* Tube();

 private:
  std::unique_ptr<Mesh> line_;
  std::unique_ptr<Mesh> line_box_;
  std::unique_ptr<Mesh> plane_;
  std::unique_ptr<Mesh> haze_;
  std::unique_ptr<Mesh> triangle_;
  std::unique_ptr<Mesh> box_;
  std::unique_ptr<Mesh> sphere_;
  std::unique_ptr<Mesh> cone_;
  std::unique_ptr<Mesh> disk_;
  std::unique_ptr<Mesh> dome_top_;
  std::unique_ptr<Mesh> dome_bottom_;
  std::unique_ptr<Mesh> tube_;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_BUILTINS_H_
