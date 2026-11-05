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
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using UnityEngine;
using UnityEngine.Rendering;

namespace Mujoco {
// API for importing Mujoco XML files into Unity scenes.
public class MjcfImporter {
  // Any materials created at runtime will only be automatically cleaned up if no other objects
  // reference it AND Unity changes scenes. In long-running worlds with no scene resets, this means
  // those materials are never cleaned even after their MjBodies are destroyed, so we cache a
  // single shared material to use for all generated meshes.
  public static Material DefaultMujocoMaterial {
    get {
      if (_DefaultMujocoMaterial == null) {
        _DefaultMujocoMaterial = new Material(GetLitShader());
      }

      return _DefaultMujocoMaterial;
    }
    set {
      _DefaultMujocoMaterial = value;
    }
  }

  // There are many configurations that need to be changed in a material to reliably render as
  // transparent, so we use template materials to ensure all default settings are correct.
  public static string SemiTransparentMaterialName {
    get {
      string defaultName = "mujoco_semitransparent_template";
      var pipeline = GraphicsSettings.currentRenderPipeline ?? GraphicsSettings.defaultRenderPipeline;
      if (pipeline != null) {
        string pipelineType = pipeline.GetType().ToString();

        if (pipelineType.Contains("HighDefinition")) {
          defaultName = "mujoco_semitransparent_template_hd";
        } else if (pipelineType.Contains("Universal")) {
          defaultName = "mujoco_semitransparent_template_urp";
        }
      }
      return defaultName;
    }
  }

  // Modifiers that change the settings of parsed nodes. They're limited to the scope of ParseRoot
  // method.
  protected MjXmlModifiers _modifiers;

  private static Material _DefaultMujocoMaterial;

  private const string _angleTypeDegree = "degree";

  private int _numGeneratedNames = 0;

  // A list of callbacks on how to handle custom node types.
  private IDictionary<string, Action<XmlElement, GameObject>> _customNodeHandlers =
      new Dictionary<string, Action<XmlElement, GameObject>>();

  public void AddNodeHandler(string name, Action<XmlElement, GameObject> handler) {
    _customNodeHandlers[name] = handler;
  }

  public MjcfImporter() {}

  public GameObject ImportString(string mjcfString, string name = null) {
    var mjcfXml = new XmlDocument();
    mjcfXml.LoadXml(mjcfString);
    return ImportXml(mjcfXml, name);
  }

  // A generic version of CreateGameObjectWithUniqueName.
  public GameObject CreateGameObjectWithUniqueName<T>(
      GameObject parentObject, XmlElement parentNode) where T : MjComponent {
    return CreateGameObjectWithUniqueName(parentObject, parentNode, typeof(T));
  }

  public string GenerateUniqueName(XmlElement parentNode) {
    var name = string.Empty;
    if (parentNode.HasAttribute("name")) {
      name = parentNode.GetAttribute("name");
    }
    if (string.IsNullOrEmpty(name)) {
      name = parentNode.Name + "_" + _numGeneratedNames++;
    }
    return name;
  }

  public unsafe GameObject ImportXml(XmlDocument mjcfXml, string name = null) {
    // Reset the numbering of anonymous nodes to ensure deterministic naming of loaded scenes.
    _numGeneratedNames = 0;

    _modifiers = new MjXmlModifiers(mjcfXml);

    ConfigureAngleType(mjcfXml);

    // We're adopting the approach of building the scene hierarchy as we parse the Mjcf.
    // Should we encounter any error, we'll terminate and clean it immediately.
    // Having a single root object for that will be very helpful, since we only need to delete that
    // object.
    var sceneRoot = new GameObject(name ?? "default");

    // We'll use Exceptions as a way to communicate errors encountered during parsing.
    try {
      ParseRoot(sceneRoot, mjcfXml.SelectSingleNode("/mujoco") as XmlElement);
    } catch (Exception ex) {
      // We consider any error as critical, and end the import process immediately, cleaning up
      // what we have already built.
      Debug.LogException(ex);
      GameObject.DestroyImmediate(sceneRoot);
      throw ex;
    } finally {
      _modifiers = null;
    }

    return sceneRoot;
  }

  // Creates a new GameObject, assigns it a unique name, and attaches a component of the specified
  // type.  If the requested component type is abstract, the method will attempt to instantiate one
  // of its implementations that passes the parser test.
  protected GameObject CreateGameObjectWithUniqueName(
      GameObject parentObject, XmlElement parentNode, Type componentType) {
    var name = GenerateUniqueName(parentNode);
    var gameObject = new GameObject(name);
    gameObject.transform.parent = parentObject.transform;
    var component = gameObject.AddComponent(componentType) as MjComponent;
    component.ParseMjcf(parentNode);
    return gameObject;
  }

  protected void ConfigureAngleType(XmlDocument mjcfXml) {
    var compilerNode = mjcfXml.SelectSingleNode("/mujoco/compiler") as XmlElement;
    if (compilerNode != null) {
      // Parse the angle type.
      var angleSetting = compilerNode.GetStringAttribute("angle", defaultValue: _angleTypeDegree);
      MjSceneImportSettings.AnglesInDegrees = angleSetting == _angleTypeDegree;
    }
  }

  protected virtual void ParseRoot(GameObject rootObject, XmlElement mujocoNode) {
    if (mujocoNode.SelectSingleNode("option") != null
        || mujocoNode.SelectSingleNode("size") != null
        || mujocoNode.SelectSingleNode("custom") != null) {
      var globalsObject = CreateGameObjectInParent("Global Settings", rootObject);
      var settingsComponent = globalsObject.AddComponent<MjGlobalSettings>();
      settingsComponent.ParseGlobalMjcfSections(mujocoNode);
    }

    // This makes references to assets.
    var worldBodyNode = mujocoNode.SelectSingleNode("worldbody") as XmlElement;
    ParseBodyChildren(rootObject, worldBodyNode);

    // This section references bodies, must be parsed after worldbody.
    var excludeNode = mujocoNode.SelectSingleNode("contact") as XmlElement;
    if (excludeNode != null) {
      var excludesParentObject = CreateGameObjectInParent("excludes", rootObject);
      foreach (var child in excludeNode.OfType<XmlElement>()) {
        if (child.Name != "exclude") {
          Debug.LogWarning(
              $"Only 'exclude' is supported - {child.Name} isn't supported yet.",
              rootObject);
        } else {
          _modifiers.ApplyModifiersToElement(child);
          CreateGameObjectWithUniqueName<MjExclude>(excludesParentObject, child);
        }
      }
    }

    // This section references joints/sites/geoms, must be parsed after worldbody.
    var tendonNode = mujocoNode.SelectSingleNode("tendon") as XmlElement;
    if (tendonNode != null) {
      var tendonsParentObject = CreateGameObjectInParent("tendons", rootObject);
      foreach (var child in tendonNode.OfType<XmlElement>()) {
        _modifiers.ApplyModifiersToElement(child, elementName: "tendon");
        if (child.Name == "fixed") {
          CreateGameObjectWithUniqueName<MjFixedTendon>(tendonsParentObject, child);
        } else if (child.Name == "spatial") {
          CreateGameObjectWithUniqueName<MjSpatialTendon>(tendonsParentObject, child);
        } else {
          Debug.Log($"Tendon type {child.Name} not supported.");
        }
      }
    }

    // This section references worldbody elements + tendons, must be parsed after them.
    var equalityNode = mujocoNode.SelectSingleNode("equality") as XmlElement;
    if (equalityNode != null) {
      var equalitiesParentObject = CreateGameObjectInParent("equality constraints", rootObject);
      foreach (var child in equalityNode.OfType<XmlElement>()) {
        var equalityType = ParseEqualityType(child);
        _modifiers.ApplyModifiersToElement(child);
        CreateGameObjectWithUniqueName(equalitiesParentObject, child, equalityType);
      }
    }

    // This section references joints and tendons, must be parsed after worldbody and tendon.
    var actuatorNode = mujocoNode.SelectSingleNode("actuator") as XmlElement;
    if (actuatorNode != null) {
      var actuatorsParentObject = CreateGameObjectInParent("actuators", rootObject);
      foreach (var child in actuatorNode.OfType<XmlElement>()) {
        _modifiers.ApplyModifiersToElement(child);
        CreateGameObjectWithUniqueName<MjActuator>(actuatorsParentObject, child);
      }
    }

    // This section references tendons, actuators and worldbody elements, must be parsed last.
    var sensorNode = mujocoNode.SelectSingleNode("sensor") as XmlElement;
    if (sensorNode != null) {
      var sensorParentObject = CreateGameObjectInParent("sensors", rootObject);
      foreach (var child in sensorNode.OfType<XmlElement>()) {
        _modifiers.ApplyModifiersToElement(child);
        var sensorType = ParseSensorType(child);
        if (sensorType != null) {
          CreateGameObjectWithUniqueName(sensorParentObject, child, sensorType);
        }
      }
    }
  }

  protected virtual void ParseGeom(GameObject parentObject, XmlElement child) {
    var gameObject = CreateGameObjectWithUniqueName<MjGeom>(parentObject, child);
    if (gameObject.GetComponent<MjGeom>().ShapeType != MjShapeComponent.ShapeTypes.HField) {
      // Add the visuals.
      gameObject.AddComponent<MjMeshFilter>();
      var renderer = gameObject.AddComponent<MeshRenderer>();
      renderer.sharedMaterial = DefaultMujocoMaterial;
    }
  }

  private GameObject CreateGameObjectInParent(string name, GameObject parentObject) {
    var createdObject = new GameObject(name);
    createdObject.transform.parent = parentObject.transform;
    return createdObject;
  }

  private void ParseBodyChildren(GameObject parentObject, XmlElement parentNode) {
    foreach (var child in parentNode.Cast<XmlNode>().OfType<XmlElement>()) {
      _modifiers.ApplyModifiersToElement(child);

      if (_customNodeHandlers.TryGetValue(child.Name, out var handler)) {
        handler?.Invoke(child, parentObject);
      } else {
        ParseBodyChild(child, parentObject);
      }
    }
  }

  // Called by ParseBodyChildren for each XML node, overridable by inheriting classes.
  private void ParseBodyChild(XmlElement child, GameObject parentObject) {
    switch (child.Name) {
      case "geom": {
        // this one's different because it's overwritten in MjImporterWithAssets.
        ParseGeom(parentObject, child);
        break;
      }
      case "inertial": {
        CreateGameObjectWithUniqueName<MjInertial>(parentObject, child);
        break;
      }
      case "body": {
        GameObject childObject = null;
        const string mocapAttribute = "mocap";
        if (child.HasAttribute(mocapAttribute)) {
          var mocapValueStr = child.GetAttribute(mocapAttribute);
          var mocapValue = bool.Parse(mocapValueStr);
          if (mocapValue) {
            childObject = CreateGameObjectWithUniqueName<MjMocapBody>(parentObject, child);
          }
        }
        if (childObject == null) {
          childObject = CreateGameObjectWithUniqueName<MjBody>(parentObject, child);
        }
        ParseBodyChildren(childObject, child);
        break;
      }
      case "joint": {
        var jointType = ParseJointType(child);
        CreateGameObjectWithUniqueName(parentObject, child, jointType);
        break;
      }
      case "freejoint": {
        CreateGameObjectWithUniqueName<MjFreeJoint>(parentObject, child);
        break;
      }
      case "site": {
        CreateGameObjectWithUniqueName<MjSite>(parentObject, child);
        break;
      }
      case "camera": {
        CreateGameObjectWithCamera(parentObject, child);
        break;
      }
      case "plugin": {
        Debug.Log($"Plugin elements are only partially supported.");
        break;
      }

      default: {
        Debug.Log($"The importer does not yet support tags <{child.Name}>.");
        break;
      }
    }
  }

  private static Type ParseEqualityType(XmlElement node) {
    Type equalityType = null;
    switch (node.Name) {
      case "connect":
        equalityType = typeof(MjConnect);
        break;
      case "weld":
        equalityType = typeof(MjWeld);
        break;
      case "joint":
        equalityType = typeof(MjJointConstraint);
        break;
      case "tendon":
        equalityType = typeof(MjTendonConstraint);
        break;
      default:
        Debug.Log($"The importer does not yet support equality <{node.Name}>.");
        break;
    }
    return equalityType;
  }

  private static Type ParseSensorType(XmlElement node) {
    Type sensorType = null;
    switch (node.Name) {
      case "actuatorpos":
      case "actuatorvel":
      case "actuatorfrc":
        sensorType = typeof(MjActuatorScalarSensor);
        break;
      case "subtreecom":
      case "subtreelinvel":
      case "subtreeangmom":
        sensorType = typeof(MjBodyVectorSensor);
        break;
      case "jointpos":
      case "jointvel":
      case "jointlimitpos":
      case "jointlimitvel":
      case "jointlimitfrc":
        sensorType = typeof(MjJointScalarSensor);
        break;
      case "touch":
      case "rangefinder":
        sensorType = typeof(MjSiteScalarSensor);
        break;
      case "accelerometer":
      case "velocimeter":
      case "force":
      case "torque":
      case "gyro":
      case "magnetometer":
        sensorType = typeof(MjSiteVectorSensor);
        break;
      case "framequat":
        switch (node.GetAttribute("objtype")) {
          case "body":
          case "xbody":
            sensorType = typeof(MjBodyQuaternionSensor);
            break;
          case "geom":
            sensorType = typeof(MjGeomQuaternionSensor);
            break;
          case "site":
            sensorType = typeof(MjSiteQuaternionSensor);
            break;
          default:
            Debug.Log($"camera sensors unsupported <{node.Name}>.");
            break;
        }
        break;
      case "framepos":
      case "framexaxis":
      case "frameyaxis":
      case "framezaxis":
      case "framelinvel":
      case "frameangvel":
      case "framelinacc":
      case "frameangacc":
        switch (node.GetAttribute("objtype")) {
          case "body":
          case "xbody":
            sensorType = typeof(MjBodyVectorSensor);
            break;
          case "geom":
            sensorType = typeof(MjGeomVectorSensor);
            break;
          case "site":
            sensorType = typeof(MjSiteVectorSensor);
            break;
          default:
            Debug.Log($"camera sensors unsupported <{node.Name}>.");
            break;
        }
        break;
      case "user":
        sensorType = typeof(MjUserSensor);
        break;
      default:
        Debug.Log($"The importer does not yet support sensor <{node.Name}>.");
        break;
    }
    return sensorType;
  }

  private Type ParseJointType(XmlElement parentNode) {
    Type jointType = null;
    switch (parentNode.GetStringAttribute("type", defaultValue: string.Empty)) {
      case "hinge":
        jointType = typeof(MjHingeJoint);
        break;
      case "ball":
        jointType = typeof(MjBallJoint);
        break;
      case "slide":
        jointType = typeof(MjSlideJoint);
        break;
      case "free":
        jointType = typeof(MjFreeJoint);
        break;
      default:
        // According to http://mujoco.org/book/XMLreference.html#joint, the default type for a joint
        // is Hinge.
        jointType = typeof(MjHingeJoint);
        break;
    }
    return jointType;
  }

  private GameObject CreateGameObjectWithCamera(
      GameObject parentObject, XmlElement parentNode) {
    var name = GenerateUniqueName(parentNode);
    var gameObject = new GameObject(name);
    gameObject.transform.parent = parentObject.transform;
    MjEngineTool.ParseTransformMjcf(parentNode, gameObject.transform);
    if (parentNode.GetStringAttribute("mode") == "targetbody") {
      // Look at the parent body:
      gameObject.transform.localRotation = Quaternion.LookRotation(
          gameObject.transform.InverseTransformPoint(parentObject.transform.position));
    } else {
      // MuJoCo's camera convention is to look down the negative z, which post-swizzle became
      // negative y, while Unity looks down positive z:
      gameObject.transform.localRotation *=
        Quaternion.FromToRotation(Vector3.forward, -Vector3.up);
    }
    var camera = gameObject.AddComponent<Camera>();
    camera.fieldOfView = parentNode.GetFloatAttribute("fovy", defaultValue: 45.0f);
    camera.nearClipPlane = 0.01f;  // MuJoCo default, TODO(etom): get from visual/map/znear
    return gameObject;
  }

  public static Shader GetLitShader() {
    string shaderName = "Standard";

    var pipeline = GraphicsSettings.currentRenderPipeline ?? GraphicsSettings.defaultRenderPipeline;
    if (pipeline != null) {
      string pipelineType = pipeline.GetType().ToString();

      if (pipelineType.Contains("Universal")) {
        shaderName = "Universal Render Pipeline/Lit";
      } else if (pipelineType.Contains("HighDefinition")) {
        shaderName = "HDRP/Lit";
      }
    }
    Shader shader = Shader.Find(shaderName) ?? Shader.Find("Standard");
    return shader;
  }
}

}
