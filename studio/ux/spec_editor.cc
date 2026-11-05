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

#include "studio/ux/spec_editor.h"

#include <memory>
#include <utility>

#include <mujoco/mujoco.h>
#include "studio/sim/model_holder.h"

namespace mujoco::studio {

SpecEditor::SpecEditor(int history_size) : capacity_(history_size) {}

void SpecEditor::Reset() {
  history_.clear();
  ref_spec_.reset();
  active_spec_.reset();
  active_map_.Clear();
  ref_map_.Clear();
  active_element_ = nullptr;
  ref_element_ = nullptr;
  active_element_key_ = kInvalidElementKey;
  next_element_key_ = 1;
  cursor_ = 0;
}

void SpecEditor::Reset(const mjSpec& spec) {
  Reset();

  ref_spec_ = Copy(&spec);
  active_spec_ = Copy(&spec);

  // Build the initial element maps.
  for (int i = mjOBJ_UNKNOWN + 1; i < mjNOBJECT; ++i) {
    const mjtObj type = static_cast<mjtObj>(i);
    mjsElement* element = mjs_firstElement(ref_spec_.get(), type);
    while (element) {
      ref_map_.Append(type, next_element_key_++);
      element = mjs_nextElement(ref_spec_.get(), element);
    }
  }
  active_map_ = ref_map_;

  // Seed the history buffer with the initial spec.
  history_.push_back(HistoryEntry{
      .spec = Copy(ref_spec_.get()),
      .op = kInitialize,
      .key = kInvalidElementKey,
      .type_index = kInvalidTypeIndex,
  });
  cursor_ = 0;
}

std::unique_ptr<ModelHolder> SpecEditor::Compile(mjVFS* vfs) {
  if (!active_spec_) {
    return nullptr;
  }

  auto holder = ModelHolder::FromSpec(mj_copySpec(active_spec_.get()), vfs);
  if (holder->ok()) {
    ref_spec_ = Copy(active_spec_.get());
    ref_map_ = active_map_;
    UpdateReferenceElement();
  }
  return holder;
}

mjSpec* SpecEditor::GetActiveSpec() const { return active_spec_.get(); }

mjsElement* SpecEditor::AddElement(mjtObj type) {
  if (!active_spec_) {
    return nullptr;
  }

  // TODO: check that type is only for spec elements.
  mjsElement* element = AddElementToSpec(active_spec_.get(), type);
  if (element) {
    const ElementKey key = next_element_key_++;
    const int index = active_map_.Append(element->elemtype, key);
    AppendHistory(HistoryEntry{
        .spec = Copy(active_spec_.get()),
        .op = kAdd,
        .key = key,
        .type_index = {element->elemtype, index},
    });
  }
  return element;
}

mjsElement* SpecEditor::AddBodyElement(mjsBody* body, mjtObj type) {
  if (!active_spec_) {
    return nullptr;
  }

  // TODO: check that type is only for body elements.
  mjsElement* element = AddElementToSpec(active_spec_.get(), type, body);
  if (element) {
    const ElementKey key = next_element_key_++;
    const int index = active_map_.Append(element->elemtype, key);
    AppendHistory(HistoryEntry{
        .spec = Copy(active_spec_.get()),
        .op = kAdd,
        .key = key,
        .type_index = {element->elemtype, index},
    });
  }
  return element;
}

void SpecEditor::DeleteActiveElement() {
  if (!active_spec_) {
    return;
  }

  if (active_element_) {
    const mjtObj type = active_element_->elemtype;
    const int index = active_map_.Remove(active_element_key_);
    mjs_delete(active_spec_.get(), active_element_);
    AppendHistory(HistoryEntry{
        .spec = Copy(active_spec_.get()),
        .op = kDelete,
        .key = active_element_key_,
        .type_index = {type, index},
    });

    ref_element_ = nullptr;
    active_element_ = nullptr;
    active_element_key_ = kInvalidElementKey;
  }
}

void SpecEditor::SetActiveElement(mjsElement* element) {
  if (!active_spec_) {
    return;
  }

  if (element == nullptr) {
    ref_element_ = nullptr;
    active_element_ = nullptr;
    active_element_key_ = kInvalidElementKey;
    return;
  }

  mjSpec* spec = mjs_getSpec(element);
  if (spec != active_spec_.get()) {
    mju_warning("Element is not owned by the active spec.");
    return;
  }

  const mjtObj type = element->elemtype;
  active_element_ = element;

  int index = 0;
  for (mjsElement* iter = mjs_firstElement(spec, type); iter != nullptr;
       iter = mjs_nextElement(spec, iter), ++index) {
    if (iter == element) {
      active_element_key_ = active_map_.LookupElementKey({type, index});
      if (active_element_key_ == kInvalidElementKey) {
        mju_warning("Element not found in active spec.");
      }
      UpdateReferenceElement();
      return;
    }
  }

  mju_warning("Element not found in active spec.");
  ref_element_ = nullptr;
  active_element_ = nullptr;
  active_element_key_ = kInvalidElementKey;
}

void SpecEditor::UpdateReferenceElement() {
  if (!active_spec_) {
    return;
  }

  if (active_element_ == nullptr) {
    ref_element_ = nullptr;
  } else {
    const mjtObj type = active_element_->elemtype;
    ref_element_ = ref_map_.Resolve(ref_spec_.get(), active_element_key_);
    if (ref_element_ == nullptr) {
      ref_element_ = mjs_firstElement(dummy_spec_.get(), type);
    }
    if (ref_element_ == nullptr) {
      ref_element_ = AddElementToSpec(dummy_spec_.get(), type);
    }
    if (ref_element_ == nullptr) {
      mju_warning("Failed to create reference element.");
      ref_element_ = active_element_;
    }
  }
}

mjsElement* SpecEditor::GetActiveElement() const { return active_element_; }

mjsElement* SpecEditor::GetRefElement() const { return ref_element_; }

void SpecEditor::CommitChanges(mjsElement* element) {
  if (!active_spec_) {
    return;
  }

  if (element == nullptr) {
    mju_warning("Element is null.");
    return;
  }
  if (mjs_getSpec(element) != active_spec_.get()) {
    mju_warning("Element is not owned by the active spec.");
    return;
  }

  AppendHistory({
      .spec = Copy(active_spec_.get()),
      .op = kModify,
      .key = active_element_key_,
      .type_index = active_map_.LookupTypeIndex(active_element_key_),
  });
}

void SpecEditor::Undo() {
  if (CanUndo()) {
    const auto& undone = history_[cursor_];
    if (undone.op == kAdd) {
      active_map_.Remove(undone.key);
    } else if (undone.op == kDelete) {
      active_map_.Insert(undone.key, undone.type_index);
    }

    --cursor_;

    auto& entry = history_[cursor_];
    active_spec_ = Copy(entry.spec.get());

    active_element_ = active_map_.Resolve(active_spec_.get(), active_element_key_);
    UpdateReferenceElement();
  }
}

bool SpecEditor::CanUndo() const { return cursor_ > 0; }

void SpecEditor::Redo() {
  if (CanRedo()) {
    ++cursor_;

    auto& entry = history_[cursor_];
    active_spec_ = Copy(entry.spec.get());
    if (entry.op == kAdd) {
      active_map_.Insert(entry.key, entry.type_index);
    } else if (entry.op == kDelete) {
      active_map_.Remove(entry.key);
    }

    active_element_ = active_map_.Resolve(active_spec_.get(), active_element_key_);
    UpdateReferenceElement();
  }
}

bool SpecEditor::CanRedo() const { return cursor_ + 1 < static_cast<int>(history_.size()); }

void SpecEditor::AppendHistory(HistoryEntry entry) {
  if (!active_spec_) {
    return;
  }

  ++cursor_;
  while (static_cast<int>(history_.size()) > cursor_) {
    history_.pop_back();
  }
  history_.push_back(std::move(entry));
  if (static_cast<int>(history_.size()) > capacity_) {
    history_.pop_front();
    --cursor_;
  }
}

void SpecEditor::ElementKeyMap::Clear() {
  for (auto& list : keys_) {
    list.clear();
  }
}

int SpecEditor::ElementKeyMap::Append(mjtObj type, ElementKey key) {
  const int index = keys_[type].size();
  keys_[type].push_back(key);
  return index;
}

int SpecEditor::ElementKeyMap::Remove(ElementKey key) {
  const TypeIndex type_index = LookupTypeIndex(key);
  auto& list = keys_[type_index.type];
  list.erase(list.begin() + type_index.index);
  return type_index.index;
}

void SpecEditor::ElementKeyMap::Insert(ElementKey key, TypeIndex type_index) {
  auto& list = keys_[type_index.type];
  list.insert(list.begin() + type_index.index, key);
}

SpecEditor::TypeIndex SpecEditor::ElementKeyMap::LookupTypeIndex(ElementKey key) const {
  for (int type = mjOBJ_UNKNOWN + 1; type < mjNOBJECT; ++type) {
    const auto& list = keys_[type];
    for (int index = 0; index < list.size(); ++index) {
      if (list[index] == key) {
        return {static_cast<mjtObj>(type), index};
      }
    }
  }
  return kInvalidTypeIndex;
}

SpecEditor::ElementKey SpecEditor::ElementKeyMap::LookupElementKey(TypeIndex type_index) const {
  if (type_index.type == mjOBJ_UNKNOWN) {
    return kInvalidElementKey;
  }
  auto& list = keys_[type_index.type];
  if (type_index.index >= list.size()) {
    return kInvalidElementKey;
  }
  return list[type_index.index];
}

mjsElement* SpecEditor::ElementKeyMap::Resolve(mjSpec* spec, ElementKey key) const {
  const TypeIndex type_index = LookupTypeIndex(key);
  if (type_index.type == mjOBJ_UNKNOWN) {
    return nullptr;
  }

  mjsElement* element = mjs_firstElement(spec, type_index.type);
  for (int i = 0; i < type_index.index && element; ++i) {
    element = mjs_nextElement(spec, element);
  }
  return element;
}

SpecEditor::SpecPtr SpecEditor::Copy(const mjSpec* spec) {
  return SpecPtr(mj_copySpec(spec), mj_deleteSpec);
}

mjsElement* SpecEditor::AddElementToSpec(mjSpec* spec, mjtObj type, mjsBody* body) {
  if (spec == nullptr || type == mjOBJ_UNKNOWN) {
    return nullptr;
  }
  if (body == nullptr) {
    body = mjs_asBody(mjs_firstElement(spec, mjOBJ_BODY));
  }

  const mjsDefault* def = nullptr;
  switch (type) {
    case mjOBJ_ACTUATOR:
      return mjs_addActuator(spec, def)->element;
    case mjOBJ_EQUALITY:
      return mjs_addEquality(spec, def)->element;
    case mjOBJ_EXCLUDE:
      return mjs_addExclude(spec)->element;
    case mjOBJ_FLEX:
      return mjs_addFlex(spec)->element;
    case mjOBJ_HFIELD:
      return mjs_addHField(spec)->element;
    case mjOBJ_KEY:
      return mjs_addKey(spec)->element;
    case mjOBJ_MATERIAL:
      return mjs_addMaterial(spec, def)->element;
    case mjOBJ_MESH:
      return mjs_addMesh(spec, def)->element;
    case mjOBJ_NUMERIC:
      return mjs_addNumeric(spec)->element;
    case mjOBJ_PAIR:
      return mjs_addPair(spec, def)->element;
    case mjOBJ_SENSOR:
      return mjs_addSensor(spec)->element;
    case mjOBJ_SKIN:
      return mjs_addSkin(spec)->element;
    case mjOBJ_TENDON:
      return mjs_addTendon(spec, def)->element;
    case mjOBJ_TEXT:
      return mjs_addText(spec)->element;
    case mjOBJ_TEXTURE:
      return mjs_addTexture(spec, def)->element;
    case mjOBJ_TUPLE:
      return mjs_addTuple(spec)->element;
    case mjOBJ_PLUGIN:
      return mjs_addPlugin(spec)->element;
    case mjOBJ_BODY:
      return mjs_addBody(body, def)->element;
    case mjOBJ_SITE:
      return mjs_addSite(body, def)->element;
    case mjOBJ_JOINT:
      return mjs_addJoint(body, def)->element;
    case mjOBJ_GEOM:
      return mjs_addGeom(body, def)->element;
    case mjOBJ_CAMERA:
      return mjs_addCamera(body, def)->element;
    case mjOBJ_LIGHT:
      return mjs_addLight(body, def)->element;
    default:
      mju_warning("Unsupported element type: %d", type);
      return nullptr;
  }
}

}  // namespace mujoco::studio
