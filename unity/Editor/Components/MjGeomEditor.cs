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

using UnityEditor;
using UnityEngine;

namespace Mujoco {

[InitializeOnLoad]
[CustomEditor(typeof(MjGeom), true)]
[CanEditMultipleObjects]
public class MjGeomEditor : MjShapeComponentEditor {

  static MjGeomEditor() {
    EditorApplication.contextualPropertyMenu += OnHeightFieldContextMenu;
  }

  public override void OnInspectorGUI() {
    serializedObject.Update();
    EditorGUILayout.PropertyField(serializedObject.FindProperty("Mass"));
    EditorGUILayout.PropertyField(serializedObject.FindProperty("Density"));
    EditorGUILayout.PropertyField(serializedObject.FindProperty("Settings"));
    serializedObject.ApplyModifiedProperties();
    base.OnInspectorGUI();
  }

  [MenuItem("CONTEXT/MjGeom/Add mesh renderer")]
  private static void AddMeshComponents(MenuCommand menuCommand) {
    var geom = menuCommand.context as MjGeom;
    geom.gameObject.AddComponent<MjMeshFilter>();
    geom.gameObject.AddComponent<MeshRenderer>().material = new Material(Shader.Find("Diffuse"));
  }

  [MenuItem("CONTEXT/MjGeom/Convert to free object")]
  private static void ConvertToFreeObject(MenuCommand menuCommand) {
    var geom = menuCommand.context as MjGeom;
    if (geom.GetComponentInParent<MjBody>()) {
      Debug.LogError("This geom already has a Body parent.", geom.GetComponentInParent<MjBody>());
      return;
    }
    var parent = new GameObject(geom.gameObject.name + " Body").AddComponent<MjBody>().transform;
    parent.position = geom.transform.position;
    geom.transform.parent = parent;

    var root = geom.transform.root;
    if (root != geom.transform) {
      parent.parent = geom.transform.parent;
    }

    var joint = new GameObject("Free Joint").AddComponent<MjFreeJoint>().transform;
    joint.position = geom.transform.position;
    joint.parent = parent;
  }

  [MenuItem("CONTEXT/Collider/Add a matching MuJoCo geom")]
  private static void AddMatchingGeom(MenuCommand menuCommand) {
    var collider = menuCommand.context as Collider;
    collider.enabled = false;
    var geom = collider.gameObject.AddComponent<MjGeom>();
    if (collider as BoxCollider) {
      geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
    } else if (collider as SphereCollider) {
      geom.ShapeType = MjShapeComponent.ShapeTypes.Sphere;
    } else if (collider as CapsuleCollider) {
      geom.ShapeType = MjShapeComponent.ShapeTypes.Capsule;
    } else if (collider as MeshCollider) {
      geom.ShapeType = MjShapeComponent.ShapeTypes.Mesh;
      ((MjMeshShape)geom.Shape).Mesh = ((MeshCollider)collider).sharedMesh;
    } else {
      Debug.LogError("Collider type not supported yet.", collider);
      collider.enabled = true;
      GameObject.DestroyImmediate(geom);
    }
  }

  public static void AddUnityTerrainWithUndo(MjGeom geom) {
    Undo.RecordObject(geom, "Add Unity Terrain");
    geom.HField.AddTerrain(geom.transform);
    Undo.RegisterCreatedObjectUndo(geom.HField.Terrain.gameObject, "Add Unity Terrain");
    EditorUtility.SetDirty(geom);
  }

  private static void OnHeightFieldContextMenu(GenericMenu menu, SerializedProperty property) {
    if (property.type != nameof(MjHeightFieldShape)) return;
    if (!(property.serializedObject.targetObject is MjGeom geom)) return;

    menu.AddItem(new GUIContent("Add Unity Terrain"), false, () => AddUnityTerrainWithUndo(geom));
  }
}
}
