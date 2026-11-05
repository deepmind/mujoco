// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace Mujoco {

[TestFixture]
public class MjHeightFieldShapeTests {
  private MjGeom _geom;
  private string _tempPngPath;

  [SetUp]
  public void SetUp() {
    _geom = new GameObject("hfield_geom", typeof(MjGeom)).GetComponent<MjGeom>();
    _geom.ShapeType = MjShapeComponent.ShapeTypes.HField;
    _tempPngPath = Path.Combine(Path.GetTempPath(), $"mujoco_hfield_{Guid.NewGuid():N}.png");
  }

  [TearDown]
  public void TearDown() {
    if (!string.IsNullOrEmpty(_tempPngPath) && File.Exists(_tempPngPath)) {
      File.Delete(_tempPngPath);
    }
    if (_geom != null) {
      if (_geom.HField.Terrain != null && _geom.HField.Terrain.terrainData != null) {
        UnityEngine.Object.DestroyImmediate(_geom.HField.Terrain.terrainData);
      }
      UnityEngine.Object.DestroyImmediate(_geom.gameObject);
    }
    if (MjScene.InstanceExists) {
      UnityEngine.Object.DestroyImmediate(MjScene.Instance.gameObject);
    }
  }

  [Test]
  public void AddTerrainCreatesAlignedChildTerrain() {
    _geom.HField.AddTerrain(_geom.transform);

    Assert.That(_geom.HField.Terrain != null, Is.True);
    Assert.That(_geom.HField.Terrain.gameObject.name, Is.EqualTo("Terrain"));
    Assert.That(_geom.HField.Terrain.transform.parent, Is.EqualTo(_geom.transform));
    Assert.That(_geom.HField.Terrain.terrainData.size, Is.EqualTo(new Vector3(100f, 1f, 100f)));
    Assert.That(
        _geom.HField.Terrain.transform.localPosition, Is.EqualTo(new Vector3(-50f, 0f, -50f)));
  }

  [Test]
  public void ContextMenuRegistersItemForHFieldOnlyAndSupportsUndo() {
    var serializedObject = new SerializedObject(_geom);
    var hFieldMenu = new GenericMenu();
    EditorApplication.contextualPropertyMenu?.Invoke(
        hFieldMenu, serializedObject.FindProperty("HField"));
    Assert.That(hFieldMenu.GetItemCount(), Is.EqualTo(1));

    var nonHFieldMenu = new GenericMenu();
    EditorApplication.contextualPropertyMenu?.Invoke(
        nonHFieldMenu, serializedObject.FindProperty("Mass"));
    Assert.That(nonHFieldMenu.GetItemCount(), Is.EqualTo(0));

    Undo.IncrementCurrentGroup();
    int undoGroup = Undo.GetCurrentGroup();
    MjGeomEditor.AddUnityTerrainWithUndo(_geom);
    Undo.CollapseUndoOperations(undoGroup);

    Assert.That(_geom.HField.Terrain != null, Is.True);
    Assert.That(_geom.transform.childCount, Is.EqualTo(1));

    Undo.PerformUndo();

    Assert.That(_geom.HField.Terrain == null, Is.True);
    Assert.That(_geom.transform.childCount, Is.EqualTo(0));
  }

  [Test]
  public void DynamicHeightFieldUsesHeightMapScaleAndBaseHeightEvenWhenFlat() {
    _geom.HField.AddTerrain(_geom.transform);
    var terrainData = _geom.HField.Terrain.terrainData;
    terrainData.heightmapResolution = 33;
    terrainData.size = new Vector3(20f, 4f, 10f);

    // Regression case for GitHub #1930: constant non-zero elevation everywhere (min == max).
    var heights = new float[33, 33];
    for (int r = 0; r < 33; r++) {
      for (int c = 0; c < 33; c++) {
        heights[r, c] = 0.7f;
      }
    }
    terrainData.SetHeights(0, 0, heights);
    _geom.HField.BaseHeight = 0.25f;
    _geom.HField.HeightMapExportPath = "";

    var doc = MjScene.Instance.CreateScene(skipCompile: true);
    var hfieldElem = doc.SelectSingleNode("//asset/hfield");

    Assert.That(hfieldElem, Is.Not.Null);
    Assert.That(hfieldElem.Attributes["size"].Value, Is.EqualTo("10 5 4 0.25"));
    Assert.That(hfieldElem.Attributes["nrow"].Value, Is.EqualTo("33"));
    Assert.That(hfieldElem.Attributes["ncol"].Value, Is.EqualTo("33"));
  }

  [Test]
  public void ExportedPngHeightFieldUsesMaximumHeightAndWarnsWhenMinNonZero() {
    _geom.HField.AddTerrain(_geom.transform);
    var terrainData = _geom.HField.Terrain.terrainData;
    terrainData.heightmapResolution = 33;
    terrainData.size = new Vector3(20f, 10f, 10f);

    var heights = new float[33, 33];
    heights[16, 16] = 0.3f;
    terrainData.SetHeights(0, 0, heights);

    _geom.HField.BaseHeight = 0.2f;
    _geom.HField.HeightMapExportPath = _tempPngPath;

    var doc = MjScene.Instance.CreateScene(skipCompile: true);
    var hfieldElem = doc.SelectSingleNode("//asset/hfield");

    Assert.That(hfieldElem, Is.Not.Null);
    Assert.That(hfieldElem.Attributes["content_type"].Value, Is.EqualTo("image/png"));
    Assert.That(hfieldElem.Attributes["nrow"].Value, Is.EqualTo("0"));
    Assert.That(hfieldElem.Attributes["ncol"].Value, Is.EqualTo("0"));
    var sizeParts = hfieldElem.Attributes["size"].Value.Split(' ');
    Assert.That(sizeParts.Length, Is.EqualTo(4));
    Assert.That(float.Parse(sizeParts[0], CultureInfo.InvariantCulture),
        Is.EqualTo(10f).Within(1e-3f));
    Assert.That(float.Parse(sizeParts[1], CultureInfo.InvariantCulture),
        Is.EqualTo(5f).Within(1e-3f));
    Assert.That(float.Parse(sizeParts[2], CultureInfo.InvariantCulture),
        Is.EqualTo(3f).Within(1e-3f));
    Assert.That(float.Parse(sizeParts[3], CultureInfo.InvariantCulture),
        Is.EqualTo(0.2f).Within(1e-3f));
    Assert.That(File.Exists(_tempPngPath), Is.True);
    Assert.That(new FileInfo(_tempPngPath).Length, Is.GreaterThan(0));

    // Raise minimum height above zero and confirm warning is emitted on PNG export.
    for (int r = 0; r < 33; r++) {
      for (int c = 0; c < 33; c++) {
        heights[r, c] = 0.5f;
      }
    }
    terrainData.SetHeights(0, 0, heights);
    LogAssert.Expect(LogType.Warning, new Regex("minimum heightmap value of 0"));
    _geom.HField.PrepareHeightMap();
  }
}
}
