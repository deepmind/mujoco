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

#include "pxr/base/tf/pyModule.h"
#include "pxr/pxr.h"

PXR_NAMESPACE_USING_DIRECTIVE

TF_WRAP_MODULE {
  TF_WRAP(MjcPhysicsActuator);
  TF_WRAP(MjcPhysicsCollisionAPI);
  TF_WRAP(MjcPhysicsEqualityAPI);
  TF_WRAP(MjcPhysicsEqualityConnectAPI);
  TF_WRAP(MjcPhysicsEqualityJointAPI);
  TF_WRAP(MjcPhysicsEqualityWeldAPI);
  TF_WRAP(MjcPhysicsImageableAPI);
  TF_WRAP(MjcPhysicsJointAPI);
  TF_WRAP(MjcPhysicsKeyframe);
  TF_WRAP(MjcPhysicsMaterialAPI);
  TF_WRAP(MjcPhysicsMeshCollisionAPI);
  TF_WRAP(MjcPhysicsSceneAPI);
  TF_WRAP(MjcPhysicsSiteAPI);
  TF_WRAP(MjcPhysicsTendon);
  TF_WRAP(MjcPhysicsTokens);
}
