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
using UnityEngine;

namespace Mujoco {

  public class MjXmlModifiers {
    private XmlDocument _root;

    public MjXmlModifiers(XmlDocument root) {
      _root = root;
    }

    public void ApplyModifiersToElement(XmlElement element) {
      // Combine all defaults into one. At this stage, we want to overwrite attributes defined by
      // the previous defaults.
      var aggregateDefaults = _root.CreateElement("aggregate");
      // Root default leaf should be processed only once, and handled first (so it's overriden).
      var rootDefaultLeaf = _root.SelectSingleNode($"/mujoco/default/{element.Name}") as XmlElement;
      if (rootDefaultLeaf != null) {
        CopyAttributes(rootDefaultLeaf, aggregateDefaults);
      }
      // Order matters - reverse class list so that most relevant is last.
      var classes = GetApplicableDefaultClasses(element).Reverse();
      foreach (var className in classes) {
        var defaultClassElement =
            _root.SelectSingleNode($"descendant::default[@class='{className}']") as XmlElement;
        // Ancestry iterates up in the tree, but we want to apply changes from remote to specific.
        if (defaultClassElement == null) { continue; }
        var ancestors = GetDefaultAncestry(defaultClassElement, element.Name).Reverse();
        foreach (var defaultAncestor in ancestors) {
          CopyAttributesOverwriteExisting(defaultAncestor, aggregateDefaults);
        }
      }
      // Add the merged attribiutes to the node, without overwriting the existing values.
      CopyAttributes(aggregateDefaults, element);
    }

    private IEnumerable<XmlElement> GetDefaultAncestry(XmlElement classElement, string nodeType) {
      var defaultElement = classElement;
      var top = _root.SelectSingleNode("/mujoco/default");
      while (defaultElement != top) {
        foreach (var element in defaultElement.ChildNodes) {
          if (((XmlElement)element).Name == nodeType) {
            yield return (XmlElement)element;
          }
        }
        defaultElement = defaultElement.ParentNode as XmlElement;
      }
    }

    // Return a list of names of all the applicable classes.  Implemented by climbing up the tree,
    // hence results are in order of relevance, from the most specific to most remotely-inherited.
    public static IEnumerable<string> GetApplicableDefaultClasses(XmlElement modifiedElement) {
      var nodeType = modifiedElement.Name;

      // Build the inheritance list for the element that's meant to be modified.
      // It will consist of all qualifiers supplied by the element itself and its parents
      // using "class" and "childclass" values.
      var inheritance = new List<string>();
      var element = modifiedElement;
      while (element != null) {
        var className = element.GetStringAttribute("childclass", string.Empty);
        if (element == modifiedElement) {
          // "class" attribute does not partcipate in building of the inheritance tree, and so we're
          // only considering it if it's defined on the very element we're trying to modify.
          className = element.GetStringAttribute("class", className);
        }
        if (!string.IsNullOrEmpty(className) && !inheritance.Contains(className)) {
          inheritance.Add(className);
        }
        element = element.ParentNode as XmlElement;
      }
      return inheritance;
    }

    private static string GetElementClass(XmlElement element) {
      var elementClass = element.GetStringAttribute("class", string.Empty);
      if (string.IsNullOrEmpty(elementClass)) {
        elementClass = element.GetStringAttribute("childclass", string.Empty);
      }
      return elementClass;
    }

    public static void CopyAttributes(XmlElement from, XmlElement to) {
      foreach (XmlAttribute attribute in from.Attributes) {
        if (to.HasAttribute(attribute.Name) == false) {
          to.SetAttribute(attribute.Name, attribute.Value);
        }
      }
    }

    public static void CopyAttributesOverwriteExisting(XmlElement from, XmlElement to) {
      foreach (XmlAttribute attribute in from.Attributes) {
        to.SetAttribute(attribute.Name, attribute.Value);
      }
    }
  }
}
