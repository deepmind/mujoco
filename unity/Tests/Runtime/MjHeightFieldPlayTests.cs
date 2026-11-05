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

#if UNITY_EDITOR

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Mujoco {

[TestFixture]
public class MjHeightFieldPlayTests {
  private MjScene _scene;
  private MjGeom _terrainGeom;
  private readonly List<GameObject> _spawnedObjects = new List<GameObject>();
  private string _tempPngPath;

  [SetUp]
  public void SetUp() {
    _scene = MjScene.Instance;
    _terrainGeom = new GameObject("hfield_geom", typeof(MjGeom)).GetComponent<MjGeom>();
    _terrainGeom.ShapeType = MjShapeComponent.ShapeTypes.HField;
    _terrainGeom.HField.AddTerrain(_terrainGeom.transform);
    _tempPngPath = Path.Combine(Path.GetTempPath(), $"mujoco_play_hfield_{Guid.NewGuid():N}.png");
  }

  [TearDown]
  public void TearDown() {
    if (!string.IsNullOrEmpty(_tempPngPath) && File.Exists(_tempPngPath)) {
      File.Delete(_tempPngPath);
    }
    foreach (var obj in _spawnedObjects) {
      if (obj != null) {
        UnityEngine.Object.DestroyImmediate(obj);
      }
    }
    _spawnedObjects.Clear();
    if (_terrainGeom != null) {
      if (_terrainGeom.HField.Terrain != null &&
          _terrainGeom.HField.Terrain.terrainData != null) {
        UnityEngine.Object.DestroyImmediate(_terrainGeom.HField.Terrain.terrainData);
      }
      UnityEngine.Object.DestroyImmediate(_terrainGeom.gameObject);
    }
    if (_scene != null) {
      UnityEngine.Object.DestroyImmediate(_scene.gameObject);
    }
  }

  private MjBody CreateDynamicSphere(string name, Vector3 position, float radius) {
    var body = new GameObject(name).AddComponent<MjBody>();
    body.transform.position = position;
    var joint = new GameObject($"{name}_free").AddComponent<MjFreeJoint>();
    joint.transform.parent = body.transform;
    joint.transform.localPosition = Vector3.zero;
    var geom = new GameObject($"{name}_geom").AddComponent<MjGeom>();
    geom.transform.parent = body.transform;
    geom.transform.localPosition = Vector3.zero;
    geom.ShapeType = MjShapeComponent.ShapeTypes.Sphere;
    geom.Sphere.Radius = radius;
    _spawnedObjects.Add(body.gameObject);
    return body;
  }

  [UnityTest]
  public IEnumerator ConstantNonZeroHeightFieldMatchesUnitySampleHeight() {
    // End-to-end physics regression test for GitHub #1930: constant 0.7 elevation across terrain.
    var terrainData = _terrainGeom.HField.Terrain.terrainData;
    terrainData.heightmapResolution = 33;
    terrainData.size = new Vector3(16f, 1f, 16f);
    _terrainGeom.HField.Terrain.transform.localPosition = new Vector3(-8f, 0f, -8f);
    var heights = new float[33, 33];
    for (int r = 0; r < 33; r++) {
      for (int c = 0; c < 33; c++) {
        heights[r, c] = 0.7f;
      }
    }
    terrainData.SetHeights(0, 0, heights);
    _terrainGeom.HField.BaseHeight = 0.1f;

    const float radius = 0.2f;
    var ball = CreateDynamicSphere("ball", new Vector3(0f, 1.3f, 0f), radius);

    _scene.CreateScene();
    yield return null;
    for (int i = 0; i < 400; i++) {
      _scene.StepScene();
    }

    float expectedY = _terrainGeom.HField.Terrain.SampleHeight(ball.transform.position) + radius;
    Assert.That(expectedY, Is.EqualTo(0.9f).Within(1e-3f));
    Assert.That(ball.transform.position.y, Is.EqualTo(expectedY).Within(2e-2f));
  }

  [UnityTest]
  public IEnumerator AsymmetricQuadrantHeightFieldMatchesSampleHeight([Values(false, true)] bool exportPng) {
    // Distinct plateau heights in all 4 (+/-X, +/-Z) quadrants with max < TerrainData.size.y.
    var terrainData = _terrainGeom.HField.Terrain.terrainData;
    terrainData.heightmapResolution = 33;
    terrainData.size = new Vector3(16f, 10f, 16f);
    _terrainGeom.HField.Terrain.transform.localPosition = new Vector3(-8f, 0f, -8f);
    var heights = new float[33, 33];
    for (int r = 0; r < 33; r++) {
      for (int c = 0; c < 33; c++) {
        if (r < 16 && c < 16) {
          heights[r, c] = 0.0f;   // (-X, -Z) -> 0.0 m
        } else if (r < 16 && c >= 16) {
          heights[r, c] = 0.1f;   // (+X, -Z) -> 1.0 m
        } else if (r >= 16 && c < 16) {
          heights[r, c] = 0.25f;  // (-X, +Z) -> 2.5 m
        } else {
          heights[r, c] = 0.4f;   // (+X, +Z) -> 4.0 m
        }
      }
    }
    terrainData.SetHeights(0, 0, heights);
    _terrainGeom.HField.BaseHeight = 0.2f;
    _terrainGeom.HField.HeightMapExportPath = exportPng ? _tempPngPath : "";

    const float radius = 0.2f;
    var testPoints = new[] {
      new Vector3(-4f, 0.8f, -4f),
      new Vector3(4f, 1.8f, -4f),
      new Vector3(-4f, 3.3f, 4f),
      new Vector3(4f, 4.8f, 4f),
    };
    var balls = new List<MjBody>();
    for (int i = 0; i < testPoints.Length; i++) {
      balls.Add(CreateDynamicSphere($"quadrant_ball_{i}", testPoints[i], radius));
    }

    _scene.CreateScene();
    yield return null;
    for (int step = 0; step < 500; step++) {
      _scene.StepScene();
    }

    foreach (var ball in balls) {
      float expectedY = _terrainGeom.HField.Terrain.SampleHeight(ball.transform.position) + radius;
      Assert.That(ball.transform.position.y, Is.EqualTo(expectedY).Within(3e-2f));
    }
  }

  [UnityTest]
  public IEnumerator DynamicHeightFieldUpdatesElevationInPlaceAtRuntime() {
    var terrainData = _terrainGeom.HField.Terrain.terrainData;
    terrainData.heightmapResolution = 33;
    terrainData.size = new Vector3(16f, 4f, 16f);
    _terrainGeom.HField.Terrain.transform.localPosition = new Vector3(-8f, 0f, -8f);
    _terrainGeom.HField.UpdateLimit = 1;
    _terrainGeom.HField.BaseHeight = 0.1f;

    const float radius = 0.2f;
    var ball = CreateDynamicSphere("dynamic_ball", new Vector3(0f, 3.0f, 0f), radius);

    _scene.CreateScene();
    yield return null;

    // Raise terrain height at runtime from 0 m to 2.0 m before letting the ball fall onto it.
    var raisedHeights = new float[33, 33];
    for (int r = 0; r < 33; r++) {
      for (int c = 0; c < 33; c++) {
        raisedHeights[r, c] = 0.5f;
      }
    }
    terrainData.SetHeights(0, 0, raisedHeights);
    _terrainGeom.HField.RebuildHeightField();

    for (int step = 0; step < 500; step++) {
      _scene.StepScene();
    }

    Assert.That(ball.transform.position.y, Is.EqualTo(2.0f + radius).Within(2e-2f));
  }
}
}

#endif
