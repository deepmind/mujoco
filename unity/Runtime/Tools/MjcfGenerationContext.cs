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
using System.Text;
using System.Xml;
using UnityEngine;

namespace Mujoco {

// An instance of this class will be used to collect dynamic dependencies produced by a Mujoco
// model during scene generation.
public class MjcfGenerationContext {

  public int NUserSensor {
    get { return _nuserSensor; }
    set { _nuserSensor = Math.Max(_nuserSensor, value); }
  }

  private int _nuserSensor;
  private int _numGeneratedNames = 0;
  private Dictionary<Mesh, string> _meshAssets = new Dictionary<Mesh, string>();

  private Dictionary<MjHeightFieldShape, string> _hFieldAssets =
      new Dictionary<MjHeightFieldShape, string>();

  public void GenerateMjcf(XmlElement mjcf) {
    GenerateConfigurationMjcf(mjcf);
    GenerateAssetsMjcf(mjcf);
  }

  // Adds a mesh to the list of assets that will be added to the generated MJCF.
  // Returns the unique name that identifies this new asset.
  public string AddMeshAsset(Mesh mesh) {
    if (!_meshAssets.ContainsKey(mesh)) {
      var uniqueName = $"mesh_{_numGeneratedNames}";
      _numGeneratedNames++;
      _meshAssets.Add(mesh, uniqueName);
    }
    return _meshAssets[mesh];
  }

  // Adds a mesh to the list of assets that will be added to the generated MJCF.
  // Returns the unique name that identifies this new asset.
  public string AddHeightFieldAsset(MjHeightFieldShape hField) {
    if (!_hFieldAssets.ContainsKey(hField)) {
      var uniqueName = $"hfield_{_numGeneratedNames}";
      _numGeneratedNames++;
      _hFieldAssets.Add(hField, uniqueName);
    }
    return _hFieldAssets[hField];
  }

  public string GenerateName(Component comp) {
    var settings = MjGlobalSettings.Instance;
    if (settings?.UseRawGameObjectNames ?? false) {
      return comp.gameObject.name;
    } else {
      var name = $"{comp.gameObject.name}_{_numGeneratedNames}";
      _numGeneratedNames++;
      return name;
    }
  }

  private void GenerateConfigurationMjcf(XmlElement mjcf) {
    var doc = mjcf.OwnerDocument;
    var compilerMjcf = (XmlElement)mjcf.AppendChild(doc.CreateElement("compiler"));
    compilerMjcf.SetAttribute("coordinate", "local");

    var optionMjcf = (XmlElement)mjcf.AppendChild(doc.CreateElement("option"));
    optionMjcf.SetAttribute(
        "gravity", MjEngineTool.Vector3ToMjcf(MjEngineTool.MjVector3(Physics.gravity)));
    optionMjcf.SetAttribute("timestep", MjEngineTool.MakeLocaleInvariant($"{Time.fixedDeltaTime}"));

    var settings = MjGlobalSettings.Instance;
    if (settings) {
      settings.GlobalsToMjcf(mjcf);
    }
  }

  private void GenerateAssetsMjcf(XmlElement mjcf) {
    var doc = mjcf.OwnerDocument;
    var assetMjcf = (XmlElement)mjcf.AppendChild(doc.CreateElement("asset"));
    foreach (var meshAsset in _meshAssets) {
      var meshMjcf = (XmlElement)assetMjcf.AppendChild(doc.CreateElement("mesh"));
      meshMjcf.SetAttribute("name", meshAsset.Value);
      GenerateMeshMjcf(meshAsset.Key, meshMjcf);
    }
    foreach (var hFieldAsset in _hFieldAssets) {
      var hFieldMjcf = (XmlElement)assetMjcf.AppendChild(doc.CreateElement("hfield"));
      hFieldMjcf.SetAttribute("name", hFieldAsset.Value);
      GenerateHeightFieldMjcf(hFieldAsset.Key, hFieldMjcf);
    }
  }

  private static void GenerateMeshMjcf(Mesh mesh, XmlElement mjcf) {
    var vertexPositionsStr = new StringBuilder();
    foreach (var unityVertex in mesh.vertices) {
      var mjVertex = MjEngineTool.MjVector3(unityVertex);
      vertexPositionsStr.Append(MjEngineTool.Vector3ToMjcf(mjVertex));
      vertexPositionsStr.Append(" ");
    }
    mjcf.SetAttribute("vertex", vertexPositionsStr.ToString());
  }

  private static void GenerateHeightFieldMjcf(MjHeightFieldShape hFieldComponent, XmlElement mjcf) {
    if (hFieldComponent.ExportImage) {
      mjcf.SetAttribute("content_type", "image/png");
      mjcf.SetAttribute("file", hFieldComponent.FullHeightMapPath);
      mjcf.SetAttribute("nrow", "0");
      mjcf.SetAttribute("ncol", "0");
    } else {
      mjcf.SetAttribute("nrow", hFieldComponent.HeightMapLength.ToString());
      mjcf.SetAttribute("ncol", hFieldComponent.HeightMapWidth.ToString());
    }

    var baseHeight = hFieldComponent.BaseHeight;
    var heightRange = hFieldComponent.ExportImage
        ? Mathf.Max(hFieldComponent.MaximumHeight, 0.00001f)
        : hFieldComponent.HeightMapScale.y;
    mjcf.SetAttribute(
        "size",
        MjEngineTool.MakeLocaleInvariant(
            $@"{hFieldComponent.HeightMapScale.x * (hFieldComponent.HeightMapLength - 1) / 2f} {
              hFieldComponent.HeightMapScale.z * (hFieldComponent.HeightMapWidth - 1) / 2f} {
                heightRange} {
                  baseHeight}"));
  }
}
}
