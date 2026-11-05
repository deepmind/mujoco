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
using System.IO;
using System.Linq;
using System.Reflection;
using System.Xml;
using UnityEditor;
using UnityEngine;

namespace Mujoco {

// API for importing Mujoco XML files into Unity scenes.
public class MjImporterWithAssets : MjcfImporter {
  private string _sourceMeshesDir;
  private string _targetMeshesDir;
  private string _targetAssetDir;
  private unsafe MujocoLib.mjModel_* _mjModel = null;

  // Imports the scene from the specified file, which should be a well-formed MJCF document.
  // The imported scene will be placed under a single node with the requested name assigned.
  //
  // Args:
  //   filePath: Path to XML file.
  //
  // Throws:
  //   Exception if the parsed XML is malformed or contains rough errors. If an exception is thrown,
  //   the entire imported scene will be automatically deleted.
  // TODO(etom) - reconsider unencouraged pattern of validation through exception side-effects.
  public unsafe GameObject ImportFile(string filePath) {
    // If MuJoCo can't parse the mjcfString, we abort the entire process.
    // MjEngineTool.LoadModelFromString throws an exception when MuJoCo fails to parse the provided
    // mjcfString.
    var name = Path.GetFileNameWithoutExtension(filePath) + $"{UnityEngine.Random.Range(0,999)}";
    string newPath = Path.Combine(Application.temporaryCachePath, $"{name}.xml");
    MjEngineTool.LoadPlugins();
    _mjModel = MjEngineTool.LoadModelFromFile(filePath);
    MjEngineTool.SaveModelToFile(newPath, _mjModel);
    Debug.Log($"Imported MJCF loaded, saved to {newPath}");
    string mjcfString = File.ReadAllText(newPath);
    GameObject root = null;
    try {
      root = ImportString(mjcfString, name, filePath);
    } finally {
      MujocoLib.mj_deleteModel(_mjModel);
    }
    return root;
  }

  public GameObject ImportString(
      string mjcfString, string name = null, string filePath = null) {
    var mjcfXml = new XmlDocument();
    mjcfXml.LoadXml(mjcfString);
    // Determine the folder where the meshes are stored.
    ConfigureMeshPath(filePath, name, mjcfXml);
    GameObject sceneRoot = null;
    try {
      sceneRoot = ImportXml(mjcfXml, name);
    } catch (Exception) {
      // We consider any error as critical, and end the import process immediately, cleaning up
      // any assets created.
      AssetDatabase.DeleteAsset(_targetAssetDir);
    }
    return sceneRoot;
  }

  protected override void ParseRoot(GameObject parentObject, XmlElement parentNode) {

    // This makes no references, and should be parsed first.
    var assetNode = parentNode.SelectSingleNode("asset") as XmlElement;
    if (assetNode != null) {
      ParseAssets(assetNode);
    }
    base.ParseRoot(parentObject, parentNode);
  }

  protected override void ParseGeom(GameObject parentObject, XmlElement child) {
    var gameObject = CreateGameObjectWithUniqueName<MjGeom>(parentObject, child);
    if (gameObject.GetComponent<MjGeom>().ShapeType != MjShapeComponent.ShapeTypes.HField) {
      gameObject.AddComponent<MjMeshFilter>();
      var renderer = gameObject.AddComponent<MeshRenderer>();
      ResolveOrCreateMaterial(renderer, child);
    }
  }

  private void ConfigureMeshPath(string path, string projectName, XmlDocument mjcf) {
    // Crate the mesh target directory.
    _targetMeshesDir = Path.Combine(
        Application.dataPath, "Local", "MjImports", projectName, "Resources");
    _targetAssetDir = Path.Combine("Assets", "Local", "MjImports", projectName, "Resources");
    if (!Directory.Exists(_targetMeshesDir)) {
      Directory.CreateDirectory(_targetMeshesDir);
    }
    if (string.IsNullOrEmpty(path)) { // we're loading a string
      return;
    }
    _sourceMeshesDir = Path.GetDirectoryName(path);
    var compilerNode = mjcf.SelectSingleNode("/mujoco/compiler") as XmlElement;
    if (compilerNode != null) {
      // Parse the location of meshes
      var relativeMeshDir = compilerNode.GetStringAttribute("meshdir", defaultValue: string.Empty);
      _sourceMeshesDir = Path.Combine(_sourceMeshesDir, relativeMeshDir);
    }
    Debug.Log(
        $"Meshes locations: source = {_sourceMeshesDir}, target = {_targetMeshesDir}");
  }

  private void ParseAssets(XmlElement parentNode) {
    foreach (var child in parentNode.SelectNodes("descendant::mesh").OfType<XmlElement>()) {
      _modifiers.ApplyModifiersToElement(child);
      ParseMesh(child);
    }
    foreach (var child in parentNode.SelectNodes("descendant::material").OfType<XmlElement>()) {
      _modifiers.ApplyModifiersToElement(child);
      ParseMaterial(child);
    }
    AssetDatabase.SaveAssets();
  }

  private void ImportMeshFromModel(int meshIndex) {
  }

  // Using mesh assets involves:
  // (1) Copying the asset to a Resources folder, and rescaling it during that operation.
  // (2) Allowing Unity to parse it using a registered asset importer (STLMeshImporter).
  // (3) Loading that asset as a Mesh resource when the referring geom is parsed.
  //
  // We're also using a dedicated folder to deploy the meshes that are being imported, so that the
  // user can find all meshes loaded by importing a specific model.
  private void ParseMesh(XmlElement parentNode) {
    if (parentNode.HasAttribute("vertex")) {
      throw new NotImplementedException("XML with explicit mesh info not supported yet.");
    }
    var fileName = parentNode.GetStringAttribute("file");
    // If we want to use the element name as the asset name, we must sanitize it:
    var unsanitizedAssetReferenceName =
      parentNode.GetStringAttribute("name", defaultValue: string.Empty);
    var assetReferenceName = MjEngineTool.Sanitize(unsanitizedAssetReferenceName);
    var sourceFilePath = Path.Combine(_sourceMeshesDir, fileName);
    var extension = Path.GetExtension(sourceFilePath).ToLowerInvariant();

    if (extension != ".obj" && extension != ".stl") {
      throw new NotImplementedException("Type of mesh file not yet supported. " +
                                        "Please convert to binary STL or OBJ. " +
                                        $"Attempted to load: {sourceFilePath}");
    }

    var targetFilePath =
        Path.Combine(_targetMeshesDir, assetReferenceName + Path.GetExtension(sourceFilePath));
    if (File.Exists(targetFilePath)) {
      File.Delete(targetFilePath);
    }
    var scale = MjEngineTool.UnityVector3(
        parentNode.GetVector3Attribute("scale", defaultValue: Vector3.one));
    CopyMeshAndRescale(sourceFilePath, targetFilePath, scale);
    var assetPath = Path.Combine(_targetAssetDir, assetReferenceName + Path.GetExtension(sourceFilePath));
    // This asset path should be available because the MuJoCo compiler guarantees element names
    // are unique, but check for completeness (and in case sanitizing the name broke uniqueness):
    if (AssetDatabase.LoadMainAssetAtPath(assetPath) != null) {
      throw new Exception(
        $"Trying to import mesh {unsanitizedAssetReferenceName} but {assetPath} already exists.");
    }

    AssetDatabase.ImportAsset(assetPath);
    ModelImporter importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
    if (importer != null && !importer.isReadable) {
      importer.isReadable = true;
      importer.SaveAndReimport();
    }

    var copiedMesh = AssetDatabase.LoadAssetAtPath<Mesh>(assetPath);
    if (copiedMesh == null) {
      throw new Exception($"Mesh {assetPath} was not imported.");
    }
    copiedMesh.RecalculateNormals();
    copiedMesh.RecalculateTangents();
    copiedMesh.RecalculateBounds();
  }

  private void CopyMeshAndRescale(
      string sourceFilePath, string targetFilePath, Vector3 scale) {
    var originalMeshBytes = File.ReadAllBytes(sourceFilePath);
    var extension = Path.GetExtension(sourceFilePath).ToLowerInvariant();
    if (extension == ".stl") {
      var mesh = StlMeshParser.ParseBinary(originalMeshBytes, scale);
      var rescaledMeshBytes = StlMeshParser.SerializeBinary(mesh);
      File.WriteAllBytes(targetFilePath, rescaledMeshBytes);
    } else if (extension == ".obj") {
      ObjMeshImportUtility.CopyAndScaleOBJFile(sourceFilePath, targetFilePath, scale);
    } else {
      throw new NotImplementedException($"Extension {Path.GetExtension(sourceFilePath)} " +
                                        $"not yet supported for MuJoCo mesh asset.");
    }
  }

  private static Material LoadTemplateMaterial(string templateName) {
    var guids = AssetDatabase.FindAssets(templateName);
    foreach (var guid in guids) {
      var path = AssetDatabase.GUIDToAssetPath(guid);
      if (Path.GetFileNameWithoutExtension(path) == templateName) {
        var mat = AssetDatabase.LoadMainAssetAtPath(path) as Material;
        if (mat != null) {
          return mat;
        }
      }
    }
    return null;
  }

  private void ParseMaterial(XmlElement parentNode) {
    var rgba = parentNode.GetFloatArrayAttribute(
      "rgba", defaultValue: new float[] {1.0f, 1.0f, 1.0f, 1.0f});
    var emission = parentNode.GetFloatAttribute("emission", defaultValue: 0.0f);
    var reflectance = parentNode.GetFloatAttribute("reflectance", defaultValue: 0.0f);
    var specular = parentNode.GetFloatAttribute("specular", defaultValue: 0.5f);
    var shininess = parentNode.GetFloatAttribute("shininess", defaultValue: 0.5f);
    var unsanitizedName = parentNode.GetStringAttribute("name", defaultValue: string.Empty);
    var name = MjEngineTool.Sanitize(unsanitizedName);
    var albedo = new Color(rgba[0], rgba[1], rgba[2], rgba[3]);

    // Mujoco uses a Blinn/Phong shading model with the addition of reflectance. Unfortunately, at
    // the moment of writing this comment, Unity does not come with a compatible shader. The closest
    // results can be achieved using the Standard shader that implements the Cook-Torrence shading
    // model.
    Material material;
    if (rgba[3] < 1f) {
      var template = LoadTemplateMaterial(SemiTransparentMaterialName);
      material = template != null ? new Material(template) : new Material(GetLitShader());
    } else {
      material = new Material(GetLitShader());
    }
    material.SetColor("_Color", albedo);
    material.SetColor("_BaseColor", albedo);
    material.SetFloat("_Metallic", reflectance);

    // In order to convert the specular/shininess parameters into glossiness/roughness,
    // we perform a coarse approximation.
    // In Blinn/Phong model, Shininess corresponds to the width of the specular spot, while
    // the Specularity corresponds to its strength (how visible it is). In order to approximate it
    // with a single parameter Glossiness, we will assume that the largest value wins. The model
    // will break at the extremes (Spec~=1 & Shin~=0, Spec~=0 & Shin~=1), however it should be
    // representative for the intermediate values.
    float glossiness = Math.Max(specular, shininess);
    // We observe that any reflective material is automatically glossy, by the property of not
    // having rough surface that would scatter the incoming light.
    // Instead of modifying the Shininess parameter however, we're dirrectly modifying
    // the glossiness by bringing the value closer to the upper boundary.
    glossiness = (1.0f - reflectance) * glossiness + reflectance;
    material.SetFloat("_Glossiness", glossiness);

    // We choose to define a simple emission model that only emits light, without scaling
    // the brightness of the defined color. If the user requires, they should tweak the material
    // settings manually.
    if (emission > 0.5f) {
      material.EnableKeyword("_EMISSION");
      material.SetColor("_EmissionColor", albedo);
    }
    var assetPath = Path.Combine(_targetAssetDir, name + ".mat");
    if (AssetDatabase.LoadMainAssetAtPath(assetPath) != null) {
      throw new Exception(
        $"Trying to create material {unsanitizedName} but {assetPath} already exists.");
    }
    AssetDatabase.CreateAsset(material, assetPath);
  }

  // Loads a named material asset, or creates an ad-hoc material asset for the specific node.
  private void ResolveOrCreateMaterial(MeshRenderer renderer, XmlElement parentNode) {
    // When the asset was parsed and stored its name was sanitized, so we should load it using a
    // sanitized name:
    var materialName =
      MjEngineTool.Sanitize(parentNode.GetStringAttribute("material", defaultValue: string.Empty));
    Material material = null;
    if (!string.IsNullOrEmpty(materialName)) {
      var assetPath = Path.Combine(_targetAssetDir, materialName + ".mat");
      material = (Material)AssetDatabase.LoadAssetAtPath(assetPath, typeof(Material));
    } else {
      // Nodes may contain inlined color definitions, which override the assigned material colors.
      if (parentNode.HasAttribute("rgba")) {
        // We need a bespoke copy of the material from the database for this particular node.
        var rgba = parentNode.GetFloatArrayAttribute(
          "rgba", defaultValue: new float[] {1.0f, 1.0f, 1.0f, 1.0f});
        if (rgba[3] < 1f) {
          var template = LoadTemplateMaterial(SemiTransparentMaterialName);
          material = template != null ? new Material(template) : new Material(GetLitShader());
        } else {
          material = new Material(GetLitShader());
        }
        var color = new Color(rgba[0], rgba[1], rgba[2], rgba[3]);
        material.SetColor("_Color", color);
        material.SetColor("_BaseColor", color);
        // We use the geom's name, guaranteed to be unique, as the asset name.
        // If geom is nameless, use a random number.
        var name =
          MjEngineTool.Sanitize(parentNode.GetStringAttribute(
            "name", defaultValue: $"{UnityEngine.Random.Range(0, int.MaxValue)}"));
        var assetPath = Path.Combine(_targetAssetDir, name+".mat");
        var suffix = 0;
        while (AssetDatabase.LoadMainAssetAtPath(assetPath) != null) {
          assetPath = Path.Combine(_targetAssetDir, $"{name}_{++suffix}.mat");
        }
        AssetDatabase.CreateAsset(material, assetPath);
        AssetDatabase.SaveAssets();
        material = AssetDatabase.LoadMainAssetAtPath(assetPath) as Material;
      } else {
        material = DefaultMujocoMaterial;
      }
    }
    if (parentNode.GetFloatAttribute("group") > 2) renderer.enabled = false;
    renderer.sharedMaterial = material;
  }
}
}
