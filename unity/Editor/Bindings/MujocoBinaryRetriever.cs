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
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;

namespace Mujoco {
public class MujocoBinaryRetriever {

  [InitializeOnLoadMethod]
  static void SubscribeToEvent() {
    // This causes the method to be invoked after the Editor registers the new list of packages.
    Events.registeredPackages += RegisteredPackagesEventHandler;
  }

  static void RegisteredPackagesEventHandler(
      PackageRegistrationEventArgs packageRegistrationEventArgs) {
    foreach (var packageInfo in packageRegistrationEventArgs.added) {
      if (packageInfo.name.Equals("org.mujoco")) {
        var mujocoPath = packageInfo.assetPath;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
          if (AssetDatabase.LoadMainAssetAtPath(mujocoPath + "/mujoco.dylib") == null) {
            File.Copy(
                "/Applications/MuJoCo.app/Contents/Frameworks" +
                "/mujoco.framework/Versions/Current/libmujoco.4.0.0.dylib",
                mujocoPath + "/mujoco.dylib");
            AssetDatabase.Refresh();
          }
        } else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
          if (AssetDatabase.LoadMainAssetAtPath(mujocoPath + "/libmujoco.so") == null) {
            File.Copy(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile) +
                "/.mujoco/mujoco-4.0.0/lib/libmujoco.so.4.0.0",
                mujocoPath + "/libmujoco.so");
            AssetDatabase.Refresh();
          }
        } else {
          if (AssetDatabase.LoadMainAssetAtPath(mujocoPath + "/mujoco.dll") == null) {
            File.Copy(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile) +
                "\\MuJoCo\\bin\\mujoco.dll",
                mujocoPath + "\\mujoco.dll");
            AssetDatabase.Refresh();
          }
        }
      }
    }
  }
}
}
