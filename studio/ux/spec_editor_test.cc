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

#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>
#include "studio/sim/model_holder.h"

namespace mujoco::studio {
namespace {

using SpecPtr = std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)>;

SpecPtr MakeSpecWithGeomMass(double mass) {
  SpecPtr spec(mj_makeSpec(), mj_deleteSpec);
  mjsBody* world = mjs_findBody(spec.get(), "world");
  mjsGeom* geom = mjs_addGeom(world, nullptr);
  geom->type = mjGEOM_SPHERE;
  geom->size[0] = 0.1;
  geom->mass = mass;
  return spec;
}

TEST(SpecEditorTest, InitialUndoStateAndSingleCommitUndoRedo) {
  SpecEditor editor;
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  SpecPtr spec = MakeSpecWithGeomMass(2.0);
  editor.Reset(*spec);
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  mjsElement* geom_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(geom_el, nullptr);
  editor.SetActiveElement(geom_el);

  mjsGeom* geom = mjs_asGeom(geom_el);
  ASSERT_NE(geom, nullptr);
  EXPECT_DOUBLE_EQ(geom->mass, 2.0);

  geom->mass = 3.0;
  editor.CommitChanges(geom_el);
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  editor.Undo();
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_TRUE(editor.CanRedo());

  mjsGeom* undone_geom = mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM));
  ASSERT_NE(undone_geom, nullptr);
  EXPECT_DOUBLE_EQ(undone_geom->mass, 2.0);

  editor.Redo();
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  mjsGeom* redone_geom = mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM));
  ASSERT_NE(redone_geom, nullptr);
  EXPECT_DOUBLE_EQ(redone_geom->mass, 3.0);
}

TEST(SpecEditorTest, UndoReversesUndoneOperationInElementMap) {
  SpecEditor editor;
  SpecPtr spec = MakeSpecWithGeomMass(2.0);
  editor.Reset(*spec);

  // Add a second geom and select it.
  mjsElement* added_el = editor.AddElement(mjOBJ_GEOM);
  ASSERT_NE(added_el, nullptr);
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());
  editor.SetActiveElement(added_el);
  ASSERT_EQ(editor.GetActiveElement(), added_el);

  // Undo the add: the added geom's key must be removed from active_map_,
  // clearing the active element selection.
  editor.Undo();
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_TRUE(editor.CanRedo());
  EXPECT_EQ(editor.GetActiveElement(), nullptr);
  mjsElement* first_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(first_el, nullptr);
  EXPECT_EQ(mjs_nextElement(editor.GetActiveSpec(), first_el), nullptr);

  // Redo the add, commit a mass change on the second geom, then delete the
  // first geom and undo the delete.
  editor.Redo();
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());
  first_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(first_el, nullptr);
  mjsElement* second_el = mjs_nextElement(editor.GetActiveSpec(), first_el);
  ASSERT_NE(second_el, nullptr);
  editor.SetActiveElement(second_el);
  mjs_asGeom(second_el)->mass = 5.0;
  editor.CommitChanges(second_el);

  editor.SetActiveElement(first_el);
  editor.DeleteActiveElement();
  EXPECT_EQ(editor.GetActiveElement(), nullptr);

  // Undo the delete: first_el's key must be re-inserted into active_map_ at
  // index 0 so both geoms can be selected and resolved.
  editor.Undo();
  first_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(first_el, nullptr);
  second_el = mjs_nextElement(editor.GetActiveSpec(), first_el);
  ASSERT_NE(second_el, nullptr);
  EXPECT_DOUBLE_EQ(mjs_asGeom(first_el)->mass, 2.0);
  EXPECT_DOUBLE_EQ(mjs_asGeom(second_el)->mass, 5.0);

  editor.SetActiveElement(first_el);
  EXPECT_EQ(editor.GetActiveElement(), first_el);
  ASSERT_NE(editor.GetRefElement(), nullptr);
  EXPECT_DOUBLE_EQ(mjs_asGeom(editor.GetRefElement())->mass, 2.0);

  // Undoing the earlier mass change on second_el while first_el is selected
  // must resolve first_el's restored key in the new active_spec_.
  editor.Undo();
  first_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(first_el, nullptr);
  EXPECT_EQ(editor.GetActiveElement(), first_el);
  EXPECT_DOUBLE_EQ(mjs_asGeom(editor.GetActiveElement())->mass, 2.0);

  editor.Redo();
  first_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(first_el, nullptr);
  second_el = mjs_nextElement(editor.GetActiveSpec(), first_el);
  ASSERT_NE(second_el, nullptr);
  editor.SetActiveElement(second_el);
  EXPECT_EQ(editor.GetActiveElement(), second_el);
  ASSERT_NE(editor.GetRefElement(), nullptr);
}

TEST(SpecEditorTest, HistoryCapacityEvictsOldestEntries) {
  SpecEditor editor(/*history_size=*/3);
  SpecPtr spec = MakeSpecWithGeomMass(1.0);
  editor.Reset(*spec);

  for (double mass : {2.0, 3.0, 4.0}) {
    mjsElement* geom_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
    ASSERT_NE(geom_el, nullptr);
    editor.SetActiveElement(geom_el);
    mjs_asGeom(geom_el)->mass = mass;
    editor.CommitChanges(geom_el);
  }

  EXPECT_TRUE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  // First undo: 4.0 -> 3.0.
  editor.Undo();
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_TRUE(editor.CanRedo());
  EXPECT_DOUBLE_EQ(mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM))->mass, 3.0);

  // Second undo: 3.0 -> 2.0. The initial state (1.0) was evicted due to
  // history_size=3, so CanUndo() must now be false.
  editor.Undo();
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_TRUE(editor.CanRedo());
  EXPECT_DOUBLE_EQ(mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM))->mass, 2.0);

  // Calling Undo() when CanUndo() is false is a no-op.
  editor.Undo();
  EXPECT_DOUBLE_EQ(mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM))->mass, 2.0);

  // Redo back to 3.0 and 4.0.
  editor.Redo();
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_TRUE(editor.CanRedo());
  EXPECT_DOUBLE_EQ(mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM))->mass, 3.0);

  editor.Redo();
  EXPECT_TRUE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());
  EXPECT_DOUBLE_EQ(mjs_asGeom(mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM))->mass, 4.0);
}

TEST(SpecEditorTest, EmptyAndRepeatedResetClearStateAndMaps) {
  SpecEditor editor;
  SpecPtr spec1 = MakeSpecWithGeomMass(2.0);
  // Add a second geom to spec1 so its element map has 2 geoms.
  mjsBody* world1 = mjs_findBody(spec1.get(), "world");
  mjsGeom* extra_geom = mjs_addGeom(world1, nullptr);
  extra_geom->mass = 9.0;

  editor.Reset(*spec1);
  mjsElement* geom_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(geom_el, nullptr);
  editor.SetActiveElement(geom_el);
  mjs_asGeom(geom_el)->mass = 4.0;
  editor.CommitChanges(geom_el);

  ASSERT_TRUE(editor.CanUndo());
  ASSERT_NE(editor.GetActiveElement(), nullptr);
  ASSERT_NE(editor.GetRefElement(), nullptr);

  // Empty reset must clear specs, selection pointers, and undo/redo state.
  editor.Reset();
  EXPECT_EQ(editor.GetActiveSpec(), nullptr);
  EXPECT_EQ(editor.GetActiveElement(), nullptr);
  EXPECT_EQ(editor.GetRefElement(), nullptr);
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  // Re-initializing twice with specs must clear undo/redo state and must not
  // retain stale element map keys from the earlier spec.
  editor.Reset(*spec1);
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());
  editor.AddElement(mjOBJ_GEOM);
  editor.AddElement(mjOBJ_GEOM);
  editor.Undo();
  ASSERT_TRUE(editor.CanUndo());
  ASSERT_TRUE(editor.CanRedo());

  SpecPtr spec2 = MakeSpecWithGeomMass(7.0);
  editor.Reset(*spec2);
  EXPECT_FALSE(editor.CanUndo());
  EXPECT_FALSE(editor.CanRedo());

  mjsElement* spec2_geom = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(spec2_geom, nullptr);
  editor.SetActiveElement(spec2_geom);
  ASSERT_NE(editor.GetRefElement(), nullptr);
  EXPECT_DOUBLE_EQ(mjs_asGeom(editor.GetRefElement())->mass, 7.0);
}

TEST(SpecEditorTest, CompileRefreshesRefElement) {
  SpecEditor editor;
  SpecPtr spec = MakeSpecWithGeomMass(2.0);
  editor.Reset(*spec);

  mjsElement* geom_el = mjs_firstElement(editor.GetActiveSpec(), mjOBJ_GEOM);
  ASSERT_NE(geom_el, nullptr);
  editor.SetActiveElement(geom_el);
  ASSERT_NE(editor.GetRefElement(), nullptr);
  EXPECT_DOUBLE_EQ(mjs_asGeom(editor.GetRefElement())->mass, 2.0);

  mjs_asGeom(geom_el)->mass = 6.0;
  editor.CommitChanges(geom_el);

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  auto holder = editor.Compile(&vfs);
  ASSERT_NE(holder, nullptr);
  ASSERT_TRUE(holder->ok()) << holder->error();

  // Accessing GetRefElement() after Compile() must not dereference freed
  // ref_spec_ memory and must reflect the newly compiled reference spec.
  mjsElement* ref_el = editor.GetRefElement();
  ASSERT_NE(ref_el, nullptr);
  EXPECT_DOUBLE_EQ(mjs_asGeom(ref_el)->mass, 6.0);

  mj_deleteVFS(&vfs);
}

TEST(SpecEditorTest, CompileAccessesVfsOnlyMeshAfterCacheClear) {
  static constexpr char kCubeObj[] =
      "v 0 0 0\n"
      "v 1 0 0\n"
      "v 0 1 0\n"
      "v 0 0 1\n"
      "f 1 3 2\n"
      "f 1 2 4\n"
      "f 1 4 3\n"
      "f 2 3 4\n";
  static constexpr char kXml[] = R"(
    <mujoco>
      <asset>
        <mesh name="tetra" file="tetra_vfs_only.obj"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="tetra"/>
      </worldbody>
    </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  ASSERT_EQ(mj_addBufferVFS(&vfs, "tetra_vfs_only.obj", kCubeObj, sizeof(kCubeObj) - 1), 0);

  char error[1000] = "";
  SpecPtr spec(mj_parseXMLString(kXml, &vfs, error, sizeof(error)), mj_deleteSpec);
  ASSERT_NE(spec, nullptr) << error;

  auto initial_holder = ModelHolder::FromSpec(mj_copySpec(spec.get()), &vfs);
  ASSERT_NE(initial_holder, nullptr);
  ASSERT_TRUE(initial_holder->ok()) << initial_holder->error();

  SpecEditor editor;
  editor.Reset(*spec);

  // Clear the compiler asset cache so recompilation must read from the VFS.
  mj_clearCache(mj_getCache());

  auto recompiled_holder = editor.Compile(&vfs);
  ASSERT_NE(recompiled_holder, nullptr);
  EXPECT_TRUE(recompiled_holder->ok()) << recompiled_holder->error();

  mj_deleteVFS(&vfs);
}

}  // namespace
}  // namespace mujoco::studio
