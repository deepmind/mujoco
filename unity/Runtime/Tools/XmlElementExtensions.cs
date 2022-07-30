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
using System.Linq;
using System.Xml;
using UnityEngine;

namespace Mujoco {
// This class helps us manipulate MuJoCo-bound XML files. It is only intended to be used
// within the context of MuJoCo - elsewhere the preferred format is JSON.
public static class XmlElementExtensions {

  public static bool GetBoolAttribute(
      this XmlElement element, string name, bool defaultValue = false) {
    if (!element.HasAttribute(name)) {
      return defaultValue;
    }
    var strValue = element.GetAttribute(name);
    bool parsedValue;
    if (bool.TryParse(strValue, out parsedValue)) {
      return parsedValue;
    } else {
      throw new ArgumentException($"'{strValue}' is not a bool.");
    }
  }

  public static float GetFloatAttribute(
      this XmlElement element, string name, float defaultValue = 0.0f) {
    if (!element.HasAttribute(name)) {
      return defaultValue;
    }
    var strValue = element.GetAttribute(name);
    float parsedValue;
    if (float.TryParse(strValue, out parsedValue)) {
      return parsedValue;
    } else {
      throw new ArgumentException($"'{strValue}' is not a float.");
    }
  }

  // The MuJoCo parser is case-sensitive.
  public static T GetEnumAttribute<T>(
      this XmlElement element, string name, T defaultValue,
      bool ignoreCase = false) where T : struct, IConvertible {
    if (!typeof(T).IsEnum) {
      throw new ArgumentException("T must be an enumerated type.");
    }
    if (!element.HasAttribute(name)) {
      return defaultValue;
    }
    var strValue = element.GetAttribute(name);
    T parsedValue;
    if (Enum.TryParse<T>(strValue, ignoreCase, out parsedValue)) {
      return parsedValue;
    } else {
      throw new ArgumentException(
        $"Reading attribute {name}: '{strValue}' is not a value of enum {typeof(T)}.");
    }
  }

  public static string GetStringAttribute(
      this XmlElement element, string name, string defaultValue = null) {
    if (!element.HasAttribute(name)) {
      return defaultValue;
    }
    return element.GetAttribute(name);
  }

  public static T GetObjectReferenceAttribute<T>(
    this XmlElement mjcf, string attributeName) where T : MjComponent {
    var objectName = mjcf.GetStringAttribute(attributeName);
    if (string.IsNullOrEmpty(objectName)) {
      return null;
    }
    T foundObject = MjHierarchyTool.FindComponentOfTypeAndName<T>(objectName);
    if (foundObject == null) {
      throw new ArgumentException(
        $"Object {objectName} of type {typeof(T).ToString()} referred by {attributeName} " +
        "doesn't exist.");
    }
    return foundObject;
  }

  public static Vector2 GetVector2Attribute(
      this XmlElement element, string name, Vector2 defaultValue) {
    var defaultValues = new float[] {defaultValue.x, defaultValue.y};
    var components = element.GetFloatArrayAttribute(name, defaultValues, fillMissingValues: false);
    if (components.Length != 2) {
      throw new ArgumentException("Invalid Vector2 string representation.");
    }
    return new Vector2(components[0], components[1]);
  }

  public static Vector3 GetVector3Attribute(
      this XmlElement element, string name, Vector3 defaultValue) {
    var defaultValues = new float[] {
        defaultValue.x, defaultValue.y, defaultValue.z};
    var components = element.GetFloatArrayAttribute(name, defaultValues, fillMissingValues: false);
    if (components.Length == 1) { return new Vector3(components[0], components[0], components[0]); }
    if (components.Length != 3) {
      throw new ArgumentException("Invalid Vector3 string representation.");
    }
    return new Vector3(components[0], components[1], components[2]);
  }

  public static Quaternion GetQuaternionAttribute(
      this XmlElement element, string name, Quaternion defaultValue) {
    var mjDefault = new float[] {
        defaultValue.w, defaultValue.x, defaultValue.y, defaultValue.z };
    var components = element.GetFloatArrayAttribute(name, mjDefault, fillMissingValues: false);
    if (components.Length != 4) {
      throw new ArgumentException("Invalid Quaternion string representation.");
    }
    return new Quaternion(w:components[0], x:components[1], y:components[2], z:components[3]);
  }

  // Parses an array of whitespace separated floating points.
  //
  // Args:
  // . element: XmlElement that contains the attribute to be parsed.
  // . name: Name of the attribute to be parsed.
  // . defaultValue: An array of floats, or null. A default value to be returned in case the
  // the attribute is missing.
  // . fillMissingValues: If a default value was provided, and it has more components than the value
  // parsed from the attribute, the missing components will be copied from the defaultValue.
  public static float[] GetFloatArrayAttribute(
      this XmlElement element, string name, float[] defaultValue, bool fillMissingValues = true) {
    if (!element.HasAttribute(name)) {
      return defaultValue;
    }
    var strValue = element.GetAttribute(name);
    var components = strValue.Split(new char[] {' '}, StringSplitOptions.RemoveEmptyEntries);
    var resultLength = components.Length;
    if (fillMissingValues && defaultValue != null) {
      // If filling of the missing values was enabled, and a default value was provided,
      // allocate an array large enough to store a value of this length, in case when the parsed
      // value has fewer components.
      resultLength = Math.Max(resultLength, defaultValue.Length);
    }
    var result = new float[resultLength];
    for (var i = 0; i < components.Length; ++i) {
      float componentValue;
      if (float.TryParse(components[i], out componentValue)) {
        result[i] = componentValue;
      } else {
        throw new ArgumentException($"'{components[i]}' is not a float.");
      }
    }
    for (var i = components.Length; i < resultLength; ++i) {
      // Fill the missing values with defaults.
      result[i] = defaultValue[i];
    }
    return result;
  }
}
}
