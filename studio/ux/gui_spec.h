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

#ifndef MUJOCO_STUDIO_UX_GUI_SPEC_H_
#define MUJOCO_STUDIO_UX_GUI_SPEC_H_

#include <mujoco/mujoco.h>
#include "studio/ux/spec_editor.h"

namespace mujoco::studio {

// Displaying the mjSpec as a tree. `element` is the currently selected
// element and will be updated if a new element is selected. The `editor` is
// optional; if provided, it will allow the user to modify the spec (e.g.
// add/delete elements) on the assumption that `element` and `spec` were
// obtained from the editor.
void SpecTreeGui(mjsElement** element, mjSpec* spec, SpecEditor* editor = nullptr);

// Displays the properties of the given element in the table. If `editor` is
// provided, the user will be able to modify the properties of the element,
// updating the editor of any changes on the assumption that `element` was
// obtained from the editor.
void ElementSpecGui(mjsElement* element, SpecEditor* editor = nullptr);

// Displays a (read-only) data table of the mjData values that correspond to the
// given element.
void ElementDataGui(const mjData* data, mjsElement* element);

// Displays a (read-only) data table of the mjModel values that correspond to
// the given element.
void ElementModelGui(const mjModel* model, mjsElement* element);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_GUI_SPEC_H_
