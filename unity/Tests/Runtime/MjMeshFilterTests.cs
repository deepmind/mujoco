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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Mujoco {

[TestFixture]
public class MjMeshFilterTests {

  private MjGeom _geom;
  private MeshFilter _meshFilter;

  [SetUp]
  public void SetUp() {
    var gameObject = new GameObject(
        "object", typeof(MjGeom), typeof(MeshFilter), typeof(MjMeshFilter));
    _geom = gameObject.GetComponent<MjGeom>();
    _meshFilter = gameObject.GetComponent<MeshFilter>();
  }

  [TearDown]
  public void TearDown() {
    UnityEngine.Object.DestroyImmediate(_geom.gameObject);
    GameObject.DestroyImmediate(MjScene.Instance);
  }

  [UnityTest]
  public IEnumerator NotMakingUnnecesaryMeshChanges() {
    yield return null;
    var firstMesh = _meshFilter.mesh;
    yield return null;
    var secondMesh = _meshFilter.mesh;
    Assert.That(firstMesh, Is.EqualTo(secondMesh));
  }

  [UnityTest]
  public IEnumerator DetectingChangesOfGeomShape() {
    yield return null;
    var firstMesh = _meshFilter.mesh;
    _geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
    yield return null;
    var secondMesh = _meshFilter.mesh;
    Assert.That(firstMesh, Is.Not.EqualTo(secondMesh));
  }

  [UnityTest]
  public IEnumerator SwitchingToHFieldDisposesMeshWithoutThrowing() {
    yield return null;
    Assert.That(_meshFilter.sharedMesh, Is.Not.Null);
    _geom.ShapeType = MjShapeComponent.ShapeTypes.HField;
    yield return null;
    Assert.That(_meshFilter.sharedMesh == null, Is.True);
  }
}
}
