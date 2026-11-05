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
using UnityEngine;

namespace Mujoco {

[RequireComponent(typeof(MjShapeComponent))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteInEditMode]
public class MjMeshFilter : MonoBehaviour {
  private MjShapeComponent _geom;
  private Vector4 _shapeChangeStamp;
  private MeshFilter _meshFilter;

  protected void Awake() {
    _geom = GetComponent<MjShapeComponent>();
    _shapeChangeStamp = new Vector4(0, 0, 0, -1);
    _meshFilter = GetComponent<MeshFilter>();
    _meshFilter.mesh = new Mesh();
  }

  protected void Update() {
    var currentChangeStamp = _geom.GetChangeStamp();
    if ((_shapeChangeStamp - currentChangeStamp).magnitude <= 1e-3f) {
      return;
    }

    if(_geom.ShapeType == MjShapeComponent.ShapeTypes.Mesh) {
      MjMeshShape meshShape = _geom.Shape as MjMeshShape;
      _meshFilter.sharedMesh = meshShape.Mesh;
      return;
    }

    _shapeChangeStamp = currentChangeStamp;
    Tuple<Vector3[], int[]> meshData = _geom.BuildMesh();
    DisposeCurrentMesh();
    if (meshData == null) {
      _meshFilter.sharedMesh = null;
      return;
    }

    var mesh = new Mesh();
    // Name this mesh to easily track resources in Unity analysis tools.
    mesh.name = $"Mujoco mesh for {gameObject.name}, id:{mesh.GetInstanceID()}";
    _meshFilter.sharedMesh = mesh;
    mesh.vertices = meshData.Item1;
    mesh.triangles = meshData.Item2;
    Vector2[] uvs = new Vector2[mesh.vertices.Length];
    for (int i = 0; i < uvs.Length; i++){
      uvs[i] = new Vector2(mesh.vertices[i].x, mesh.vertices[i].z);
    }
    mesh.uv = uvs;
    mesh.RecalculateNormals();
    mesh.RecalculateTangents();
  }

  protected void OnDestroy() {
    DisposeCurrentMesh();
  }

  // Dynamically created meshes with no references are only disposed automatically on scene changes.
  // This prevents resource leaks in case the host environment doesn't reload scenes.
  private void DisposeCurrentMesh() {
    if (_meshFilter.sharedMesh != null && _geom.ShapeType != MjShapeComponent.ShapeTypes.Mesh) {
#if UNITY_EDITOR
      DestroyImmediate(_meshFilter.sharedMesh);
#else
      Destroy(_meshFilter.sharedMesh);
#endif
    }
  }
}
}
