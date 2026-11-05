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
public class MjcfImporterTests {

  private GameObject _sceneRoot;
  private MjcfImporter _importer;

  [SetUp]
  public void SetUp() {
    _importer = new MjcfImporter();
  }

  [TearDown]
  public void TearDown() {
    if (_sceneRoot != null) {
      UnityEngine.Object.DestroyImmediate(_sceneRoot);
    }
  }

  [Test]
  public void SceneRootIsGivenRequestedName() {
    var rootName = "rootName";
    var mjcfString = "<mujoco><worldbody/></mujoco>";
    _sceneRoot = _importer.ImportString(name: rootName, mjcfString: mjcfString);
    Assert.That(_sceneRoot.name, Is.EqualTo(rootName));
  }

  [Test]
  public void ParsingAGeom() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <geom size='1'/>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    Assert.That(_sceneRoot.transform.childCount, Is.EqualTo(1));
    var geomChild = _sceneRoot.transform.GetChild(0).gameObject;
    Assert.That(geomChild, Is.Not.Null);
    Assert.That(geomChild.GetComponent<MjGeom>(), Is.Not.Null);
    Assert.That(geomChild.GetComponent<MjMeshFilter>(), Is.Not.Null);
    Assert.That(geomChild.GetComponent<MeshRenderer>(), Is.Not.Null);
  }

  [Test]
  public void ParsingHFieldGeomSkipsMeshFilterAndRenderer() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <geom type='hfield'/>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    Assert.That(_sceneRoot.transform.childCount, Is.EqualTo(1));
    var geomChild = _sceneRoot.transform.GetChild(0).gameObject;
    var geom = geomChild.GetComponent<MjGeom>();
    Assert.That(geom, Is.Not.Null);
    Assert.That(geom.ShapeType, Is.EqualTo(MjShapeComponent.ShapeTypes.HField));
    Assert.That(geomChild.GetComponent<MjMeshFilter>(), Is.Null);
    Assert.That(geomChild.GetComponent<MeshRenderer>(), Is.Null);
  }

  [TestCase("hinge", typeof(MjHingeJoint))]
  [TestCase("slide", typeof(MjSlideJoint))]
  [TestCase("free", typeof(MjFreeJoint))]
  public void ParsingJoints(string typeId, Type jointType) {
    var mjcfString = string.Format(@"<mujoco>
      <worldbody>
        <body>
          <joint type=""{0}""/>
          <geom size='1'/>
        </body>
      </worldbody>
    </mujoco>", typeId);
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    Assert.That(_sceneRoot.transform.childCount, Is.EqualTo(1));
    var bodyChild = _sceneRoot.transform.GetChild(0).gameObject;
    Assert.That(bodyChild.transform.childCount, Is.EqualTo(2));
    var jointChild = bodyChild.transform.GetChild(0).gameObject;
    Assert.That(jointChild, Is.Not.Null);
    Assert.That(jointChild.GetComponent(jointType), Is.Not.Null);
  }

  [Test]
  public void ParsingAFreeJointStandaloneNode() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body>
          <freejoint/>
          <geom size='1'/>
        </body>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    Assert.That(_sceneRoot.transform.childCount, Is.EqualTo(1));
    var bodyChild = _sceneRoot.transform.GetChild(0).gameObject;
    Assert.That(bodyChild.transform.childCount, Is.EqualTo(2));
    var jointChild = bodyChild.transform.GetChild(0).gameObject;
    Assert.That(jointChild, Is.Not.Null);
    Assert.That(jointChild.GetComponent<MjFreeJoint>(), Is.Not.Null);
  }

  [Test]
  public void ParsingABody() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body/>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    Assert.That(_sceneRoot.transform.childCount, Is.EqualTo(1));
    var bodyChild = _sceneRoot.transform.GetChild(0).gameObject;
    Assert.That(bodyChild, Is.Not.Null);
    Assert.That(bodyChild.GetComponent<MjBody>(), Is.Not.Null);
  }

  [Test]
  public void ParsingChildrenOfABody() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body>
          <geom size='1'/>
          <body/>
          <joint type=""hinge""/>
        </body>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    Assert.That(_sceneRoot.transform.childCount, Is.EqualTo(1));
    var bodyChild = _sceneRoot.transform.GetChild(0).gameObject;
    Assert.That(bodyChild, Is.Not.Null);
    Assert.That(bodyChild.transform.childCount, Is.EqualTo(3));
  }

  [Test]
  public void AnonymousNodeReceivesGeneratedName() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body/>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var bodyChild = _sceneRoot.transform.GetChild(0).gameObject;
    Assert.That(bodyChild.name, Is.EqualTo("body_0"));
  }

  [Test]
  public void AnonymousNodesReceiveTheSameNamesWithEveryImport() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body/>
      </worldbody>
    </mujoco>";
    var scenes = new GameObject[] {
      _importer.ImportString(name: string.Empty, mjcfString: mjcfString),
      _importer.ImportString(name: string.Empty, mjcfString: mjcfString)
    };
    // Ensure that the objects will be destroyed after the test is completed.
    _sceneRoot = new GameObject();
    foreach (var scene in scenes) {
      scene.transform.parent = _sceneRoot.transform;
    }
    var children = scenes.Select(parent => parent.transform.GetChild(0).gameObject).ToArray();
    Assert.That(children[0].name, Is.EqualTo(children[1].name));
  }

  [Test]
  public void ApplyingDefaultOutsideWorldbody() {
    var mjcfString = @"<mujoco>
      <default>
        <general ctrllimited='true' ctrlrange='-1 1'/>
      </default>
      <worldbody>
        <body>
          <geom name=""root_geom"" size='1 2'/>
          <joint name='j'/>
        </body>
      </worldbody>
      <actuator>
        <general joint='j'/>
      </actuator>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var actuator = _sceneRoot.GetComponentInChildren<MjActuator>();
    Assert.That(actuator.CommonParams.CtrlLimited, Is.True);
  }

  [Test]
  public void ApplyingDefaultSettingsToNodeWithoutClass() {
    var mjcfString = @"<mujoco>
      <default>
        <geom type=""cylinder""/>
      </default>
      <worldbody>
        <geom name=""root_geom"" size='1 2'/>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var rootGeom = _sceneRoot.GetComponentInChildren<MjGeom>();
    Assert.That(rootGeom.ShapeType, Is.EqualTo(MjShapeComponent.ShapeTypes.Cylinder));
  }

  [Test]
  public void ApplyingHierarchicalDefaultSettingsToNodeWithClass() {
    var mjcfString = @"<mujoco>
      <default>
        <geom type=""cylinder""/>
        <default class=""child"">
          <geom type=""box""/>
        </default>
      </default>
      <worldbody>
        <body childclass=""child"">
          <geom name=""body_geom"" size='1 2 3'/>
        </body>
      </worldbody>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var bodyGeom = _sceneRoot.GetComponentInChildren<MjGeom>();
    Assert.That(bodyGeom.ShapeType, Is.EqualTo(MjShapeComponent.ShapeTypes.Box));
  }

  [Test]
  public void ReadingOption() {
    var mjcfString = @"<mujoco>
      <option impratio='1.2345'/>
      <worldbody/>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var settings = _sceneRoot.GetComponentInChildren<MjGlobalSettings>();
    Assert.That(settings.GlobalOptions.ImpRatio, Is.EqualTo(1.2345f));
  }

  [Test]
  public void ReadingSize() {
    var mjcfString = @"<mujoco>
      <size memory='1M'/>
      <worldbody/>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var settings = _sceneRoot.GetComponentInChildren<MjGlobalSettings>();
    Assert.That(settings.GlobalSizes.Memory, Is.EqualTo("1M"));
  }

  [Test]
  public void ReadingOptionAndSize() {
    var mjcfString = @"<mujoco>
      <option impratio='5.4321'/>
      <size memory='16K'/>
      <worldbody/>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var settings = _sceneRoot.GetComponentInChildren<MjGlobalSettings>();
    Assert.That(settings.GlobalOptions.ImpRatio, Is.EqualTo(5.4321f));
    Assert.That(settings.GlobalSizes.Memory, Is.EqualTo("16K"));
  }

  [Test]
  public void ReadingOptionFlags() {
    var mjcfString = @"<mujoco>
      <option>
        <flag gravity='disable' contact='disable'/>
      </option>
      <worldbody/>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var settings = _sceneRoot.GetComponentInChildren<MjGlobalSettings>();
    Assert.That(settings.GlobalOptions.Flag.Gravity, Is.EqualTo(EnableDisableFlag.disable));
    Assert.That(settings.GlobalOptions.Flag.Contact, Is.EqualTo(EnableDisableFlag.disable));
  }

  [Test]
  public void MultiCcdDefaultsToEnabledWhenFlagAbsent() {
    var mjcfString = @"<mujoco>
      <option/>
      <worldbody/>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var settings = _sceneRoot.GetComponentInChildren<MjGlobalSettings>();
    Assert.That(settings, Is.Not.Null);
    Assert.That(settings.GlobalOptions.Flag.MultiCCD, Is.EqualTo(EnableDisableFlag.enable));
  }

  [Test]
  public void MultiCcdDisabledFlagIsPreserved() {
    var mjcfString = @"<mujoco>
      <option>
        <flag multiccd='disable'/>
      </option>
      <worldbody/>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var settings = _sceneRoot.GetComponentInChildren<MjGlobalSettings>();
    Assert.That(settings, Is.Not.Null);
    Assert.That(settings.GlobalOptions.Flag.MultiCCD, Is.EqualTo(EnableDisableFlag.disable));
  }

  [Test]
  public void ParsedActuatorsAddedToDedicatedGameObjectAggregate() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body>
          <geom size='1'/>
          <joint name='1'/>
        </body>
      </worldbody>
      <actuator>
        <motor joint='1'/>
      </actuator>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var actuator = _sceneRoot.GetComponentInChildren<MjActuator>();
    Assert.That(actuator, Is.Not.Null);
    Assert.That(actuator.transform.parent.gameObject.name, Is.EqualTo("actuators"));
  }

  [Test]
  public void ParsedSensorsAddedToDedicatedGameObjectAggregate() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body name='1'/>
      </worldbody>
      <sensor>
        <framepos objtype='body' objname='1'/>
      </sensor>
    </mujoco>";
    _sceneRoot = _importer.ImportString(
        name: string.Empty, mjcfString: mjcfString);
    var sensor = _sceneRoot.GetComponentInChildren<MjBaseSensor>();
    Assert.That(sensor, Is.Not.Null);
    Assert.That(sensor.transform.parent.gameObject.name, Is.EqualTo("sensors"));
  }

  [Test]
  public void NodeHandlerCalledForMatchingelements() {
    var mjcfString = @"<mujoco>
      <worldbody>
        <body>
          <CustomTestNode/>
        </body>
      </worldbody>
    </mujoco>";
    var customImporter = new MjcfImporter();
    bool ranCustomFunction = false;
    customImporter.AddNodeHandler("CustomTestNode", (element, o) => {
      ranCustomFunction = true;
    });
    var mjcfXml = new XmlDocument();
    mjcfXml.LoadXml(mjcfString);
    customImporter.ImportXml(mjcfXml);
    Assert.That(ranCustomFunction, Is.True);
  }
}
}
