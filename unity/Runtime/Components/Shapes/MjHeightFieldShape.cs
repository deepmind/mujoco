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
using System.IO;
using System.Linq;
using System.Xml;
using UnityEngine;
// internal imports
namespace Mujoco {

[Serializable]
public class MjHeightFieldShape : IMjShape {

  [Tooltip("Terrain's heightmap should have a minimum value of zero (fully black). Right click " +
           "HField to add Unity Terrain.")]
  public Terrain Terrain;

  [Tooltip("The path, relative to Application.dataPath, where the heightmap " +
           "data will be saved/exported in PNG format. Leave blank if hfield data should " +
           "be set instead programmatically (faster).")]
  public string HeightMapExportPath;

  [Tooltip("This thickness will be added below the 0 height in the hfield, to prevent " +
           "penetration.")]
  public float BaseHeight = 0.1f;

  public bool ExportImage => !string.IsNullOrEmpty(HeightMapExportPath);

  public string FullHeightMapPath => Path.GetFullPath(Path.Combine(Application.dataPath,
      HeightMapExportPath));

  public int HeightMapWidth => Terrain.terrainData.heightmapResolution;
  public int HeightMapLength => Terrain.terrainData.heightmapResolution;
  public Vector3 HeightMapScale => Terrain.terrainData.heightmapScale;

  [Tooltip("At least this many frames will have to pass before the scene is rebuilt with an " +
           "updated heightmap. Leave as 0 to not update the hfield during simulation. " +
           "If nonzero, increasing this can improve performance.")]
  public int UpdateLimit;

  private int _updateCountdown;
  private Action _cleanupSceneSubscriptions;

  [HideInInspector]
  public float MaximumHeight { get; private set; }

  public int HeightFieldId { get; private set; }

  public unsafe void ToMjcf(XmlElement mjcf, Transform transform) {
    _cleanupSceneSubscriptions?.Invoke();
    _cleanupSceneSubscriptions = null;

    if (Terrain.transform.parent != transform)
      Debug.LogWarning(
          $"The terrain of heightfield {transform.name} needs to be parented to the Geom " +
          "for proper rendering.");
    else {
      if ((Terrain.transform.localPosition - new Vector3(
              -(HeightMapLength - 1) * HeightMapScale.x / 2f,
              Terrain.transform.localPosition.y,
              -(HeightMapWidth - 1) * HeightMapScale.z / 2)).magnitude > 0.001) {
        Debug.LogWarning($"Terrain of heightfield {transform.name} not aligned with geom. The " +
                         " terrain will be moved to accurately represent the simulated position.");
      }
      Terrain.transform.localPosition = new Vector3(-(HeightMapLength - 1) * HeightMapScale.x / 2,
          Terrain.transform.localPosition.y,
          -(HeightMapWidth - 1) * HeightMapScale.z / 2);
    }
    var scene = MjScene.Instance;
    var assetName = scene.GenerationContext.AddHeightFieldAsset(this);

    EventHandler<MjStepArgs> onPostInit = null;
    EventHandler<MjStepArgs> onPreUpdate = null;
    TerrainCallbacks.HeightmapChangedCallback onHeightmapChanged = null;
    EventHandler<MjStepArgs> onPreDestroy = null;

    if (Application.isPlaying) {
      onPostInit = (unused_first, unused_second) => {
        if (scene == null || scene.Model == null || Terrain == null) return;
        HeightFieldId =
            MujocoLib.mj_name2id(scene.Model, (int)MujocoLib.mjtObj.mjOBJ_HFIELD, assetName);
        if (!ExportImage) {
          UpdateHeightFieldData();
        }
      };
      scene.postInitEvent += onPostInit;

      if (UpdateLimit > 0) {
        _updateCountdown = UpdateLimit;
        if (UpdateLimit > 1) {
          onPreUpdate = (unused_first, unused_second) => CountdownUpdateCondition();
          scene.preUpdateEvent += onPreUpdate;
        }
        onHeightmapChanged = (unused_first, unused_second, unused_third) => RebuildHeightField();
        TerrainCallbacks.heightmapChanged += onHeightmapChanged;
      }

      onPreDestroy = (unused_first, unused_second) => _cleanupSceneSubscriptions?.Invoke();
      _cleanupSceneSubscriptions = () => {
        if (scene != null) {
          if (onPostInit != null) scene.postInitEvent -= onPostInit;
          if (onPreUpdate != null) scene.preUpdateEvent -= onPreUpdate;
          scene.preDestroyEvent -= onPreDestroy;
        }
        if (onHeightmapChanged != null) TerrainCallbacks.heightmapChanged -= onHeightmapChanged;
        _cleanupSceneSubscriptions = null;
      };
      scene.preDestroyEvent += onPreDestroy;
    }

    mjcf.SetAttribute("hfield", assetName);
    PrepareHeightMap();
  }

  public void FromMjcf(XmlElement mjcf) {
  }

  public void PrepareHeightMap() {
    int width = HeightMapWidth;
    int length = HeightMapLength;
    float[,] heights = Terrain.terrainData.GetHeights(0, 0, width, length);
    float minNormalized = float.PositiveInfinity;
    float maxNormalized = float.NegativeInfinity;
    for (int r = 0; r < length; r++) {
      for (int c = 0; c < width; c++) {
        float h = heights[r, c];
        if (h < minNormalized) minNormalized = h;
        if (h > maxNormalized) maxNormalized = h;
      }
    }
    MaximumHeight = maxNormalized * HeightMapScale.y;
    var minimumHeight = minNormalized * HeightMapScale.y;

    if (ExportImage) {
      if (minimumHeight > 0.0001f) {
        Debug.LogWarning("Due to assumptions in MuJoCo heightfields, terrains should have a " +
                         "minimum heightmap value of 0.");
      }
      var texture = new Texture2D(width, length);
      try {
        var pixels = new Color[width * length];
        for (int r = 0; r < length; r++) {
          for (int c = 0; c < width; c++) {
            float grey = maxNormalized > 0f ? heights[r, c] / maxNormalized : 0f;
            pixels[r * width + c] = new Color(grey, grey, grey, 1f);
          }
        }
        texture.SetPixels(pixels);
        texture.Apply();
        File.WriteAllBytes(FullHeightMapPath, texture.EncodeToPNG());
      } finally {
        UnityEngine.Object.DestroyImmediate(texture);
      }
    }
  }

  public unsafe void UpdateHeightFieldData() {
    if (Terrain == null || MjScene.Instance == null || MjScene.Instance.Model == null) return;

    int width = HeightMapWidth;
    int length = HeightMapLength;
    float[,] heights = Terrain.terrainData.GetHeights(0, 0, width, length);
    int adr = MjScene.Instance.Model->hfield_adr[HeightFieldId];
    for (int r = 0; r < length; r++) {
      for (int c = 0; c < width; c++) {
        MjScene.Instance.Model->hfield_data[adr + r * width + c] = heights[r, c];
      }
    }
  }

  public void CountdownUpdateCondition() {
    if (_updateCountdown < 1) return;

    _updateCountdown -= 1;
  }

  public void RebuildHeightField() {
    // The update rate limiting countdown goes from UpdateLimit + 1 to 1, since if the Update
    // limit is at 1, we can update on every frame and don't need a countdown.
    if (_updateCountdown > 1) return;
    if (!Application.isPlaying || !MjScene.InstanceExists) return;

    if (ExportImage) {
      // If we export an image, it needs to be read by the compiler so we might as well rebuild
      // the scene.
      MjScene.Instance.SceneRecreationAtLateUpdateRequested = true;
    } else {
      UpdateHeightFieldData();
    }
    _updateCountdown = UpdateLimit + 1;
  }

  public Vector4 GetChangeStamp() {
    return Vector4.one;
  }

  public Tuple<Vector3[], int[]> BuildMesh() {
    return null;
  }

  public void DebugDraw(Transform transform) {}

  public void AddTerrain(Transform parent) {
    var terrainData = new TerrainData();
    var terrainObject = Terrain.CreateTerrainGameObject(terrainData);
    terrainObject.name = "Terrain";
    var newTerrain = terrainObject.GetComponent<Terrain>();
    newTerrain.terrainData.size = new Vector3(100, 1, 100);
    terrainObject.transform.parent = parent;
    Terrain = newTerrain;
    terrainObject.transform.localPosition = new Vector3(-50, 0, -50);
  }
}
}
