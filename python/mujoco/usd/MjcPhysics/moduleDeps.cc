// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/scriptModuleLoader.h>
#include <pxr/base/tf/token.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfScriptModuleLoader) {
  // List of direct dependencies for this library.
  const std::vector<TfToken> reqs = {
      TfToken("usd"),
  };
  TfScriptModuleLoader::GetInstance().RegisterLibrary(
      TfToken("mjcPhysics"), TfToken("mujoco.usd.MjcPhysics"), reqs);
}

PXR_NAMESPACE_CLOSE_SCOPE
