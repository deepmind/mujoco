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

#ifndef MUJOCO_STUDIO_UX_SPEC_EDITOR_H_
#define MUJOCO_STUDIO_UX_SPEC_EDITOR_H_

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <vector>

#include <mujoco/mujoco.h>
#include "studio/sim/model_holder.h"

namespace mujoco::studio {

// Manages everything related to mjSpec editing including compiling, undo/redo,
// etc.
//
// The primary responsibility of the SpecEditor is to manage an "active" spec
// and element. This is the spec/element that is currently being edited. It
// also keeps track of the most recently compiled spec which is used as a
// reference spec for editing. Finally, it tracks a history of specs in
// order to support undo/redo operations.
class SpecEditor {
 public:
  explicit SpecEditor(int history_size = 256);

  // Resets the editor with an empty spec. No editing will be allowed.
  void Reset();

  // Resets the editor to manage the given spec, clearing all cached data.
  // Assumes the given spec is correctly formed (i.e. compilable).
  void Reset(const mjSpec& spec);

  // Attempts to compile the active spec, returning a ModelHolder. If the
  // compilation was successfully, it will also update the reference spec to be
  // the source spec.
  std::unique_ptr<ModelHolder> Compile(mjVFS* vfs);

  // Returns the active spec being edited. That said, users should not directly
  // modify this spec. Instead, they should use the add/delete operations below.
  mjSpec* GetActiveSpec() const;

  // Adds an element to the active spec.
  mjsElement* AddElement(mjtObj type);

  // Adds an element to the given body of the active spec.
  mjsElement* AddBodyElement(mjsBody* body, mjtObj type);

  // Deletes the currently active element from the active spec.
  void DeleteActiveElement();

  // Marks the given element as the active element being edited. This element
  // must be owned by the active spec (i.e. the spec at the current point in
  // the history buffer). Note that some operations (e.g. DeleteElement) will
  // set the active element to nullptr.
  void SetActiveElement(mjsElement* element);

  // Returns the element that is currently being edited. If this element is
  // modified, then the caller should call CommitChanges() below (otherwise
  // the change will not be saved in the history buffer).
  mjsElement* GetActiveElement() const;

  // Commits any changes made to the given element. This effectively updates
  // the history buffer to support undo/redo operations.
  void CommitChanges(mjsElement* element);

  // Undoes the last change, making a new spec the active spec.
  void Undo();
  bool CanUndo() const;

  // Redoes the last change, making a new spec the active spec.
  void Redo();
  bool CanRedo() const;

  // Returns the reference element against which the active element is being
  // edited. This is either an element from the "reference" spec that matches
  // the active element, or, in the case of a newly added element, a default
  // element of the same type.
  mjsElement* GetRefElement() const;

 private:
  // A unique pointer to an mjSpec that will be deleted by mj_deleteSpec.
  using SpecPtr = std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)>;

  // A simple tuple of mjtObj type and int index.
  struct TypeIndex {
    mjtObj type;
    int index;
  };
  static constexpr TypeIndex kInvalidTypeIndex = {mjOBJ_UNKNOWN, 0};

  // Every element encountered by this class is assigned a unique ElementKey.
  // This allows us to track elements across copies of the spec.
  using ElementKey = std::size_t;
  static constexpr ElementKey kInvalidElementKey = 0;

  // A bidirectional mapping of ElementKey to TypeIndex. This allows us to
  // efficiently track elements across copies of a spec.
  //
  // The map should be initialized from a base spec, assigning a unique key
  // to every element. When elements are added or removed (usually from copies
  // of the spec), this mapping can be updated to reflect the changes. We can
  // then "resolve" an element from a spec based on its key.
  class ElementKeyMap {
   public:
    void Clear();

    int Append(mjtObj type, ElementKey key);

    int Remove(ElementKey key);
    void Insert(ElementKey key, TypeIndex type_index);

    TypeIndex LookupTypeIndex(ElementKey key) const;
    ElementKey LookupElementKey(TypeIndex type_index) const;

    // Returns the element in the spec that corresponds to the given key.
    mjsElement* Resolve(mjSpec* spec, ElementKey key) const;

   private:
    std::array<std::vector<ElementKey>, mjNOBJECT> keys_;
  };

  // The operation that was performed on the spec, used for undo/redo.
  enum Operation {
    kInitialize,
    kModify,
    kAdd,
    kDelete,
  };

  // A single entry in the history buffer.
  struct HistoryEntry {
    SpecPtr spec;          // A full copy of a spec.
    Operation op;          // The operation performed on the spec.
    ElementKey key;        // The key of the element being modified.
    TypeIndex type_index;  // The type/index of the element after the change.
  };

  // Appends a new HistoryEntry to the history buffer.
  void AppendHistory(HistoryEntry entry);

  // Updates the reference element to be the element in the reference spec that
  // corresponds the active element.
  void UpdateReferenceElement();

  // Adds an element to the spec with the given type and (optional) body.
  static mjsElement* AddElementToSpec(mjSpec* spec, mjtObj type, mjsBody* body = nullptr);

  // Creates a copy of the given spec.
  static SpecPtr Copy(const mjSpec* spec);

  std::deque<HistoryEntry> history_;
  SpecPtr ref_spec_ = SpecPtr(nullptr, mj_deleteSpec);
  SpecPtr active_spec_ = SpecPtr(nullptr, mj_deleteSpec);
  SpecPtr dummy_spec_ = SpecPtr(mj_makeSpec(), mj_deleteSpec);
  ElementKeyMap active_map_;
  ElementKeyMap ref_map_;
  mjsElement* active_element_ = nullptr;
  mjsElement* ref_element_ = nullptr;
  ElementKey active_element_key_ = kInvalidElementKey;
  ElementKey next_element_key_ = 1;
  int capacity_ = 0;
  int cursor_ = 0;
};
}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_SPEC_EDITOR_H_
