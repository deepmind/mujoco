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

import 'jasmine';

import { MainModule } from "../dist/mujoco"
import loadMujoco from "../dist/mujoco.js"

let mujoco: MainModule;

describe('Enums', () => {
  beforeAll(async () => {
    mujoco = await loadMujoco();
  }, 60_000);

  it('mjtDisableBit should exist', () => {
    expect(mujoco.mjtDisableBit).toBeDefined();
  });

  it('mjtEnableBit should exist', () => {
    expect(mujoco.mjtEnableBit).toBeDefined();
  });

  it('mjtJoint should exist', () => {
    expect(mujoco.mjtJoint).toBeDefined();
  });

  it('mjtGeom should exist', () => {
    expect(mujoco.mjtGeom).toBeDefined();
  });

  it('mjtProjection should exist', () => {
    expect(mujoco.mjtProjection).toBeDefined();
  });

  it('mjtCamLight should exist', () => {
    expect(mujoco.mjtCamLight).toBeDefined();
  });

  it('mjtLightType should exist', () => {
    expect(mujoco.mjtLightType).toBeDefined();
  });

  it('mjtTexture should exist', () => {
    expect(mujoco.mjtTexture).toBeDefined();
  });

  it('mjtTextureRole should exist', () => {
    expect(mujoco.mjtTextureRole).toBeDefined();
  });

  it('mjtColorSpace should exist', () => {
    expect(mujoco.mjtColorSpace).toBeDefined();
  });

  it('mjtIntegrator should exist', () => {
    expect(mujoco.mjtIntegrator).toBeDefined();
  });

  it('mjtCone should exist', () => {
    expect(mujoco.mjtCone).toBeDefined();
  });

  it('mjtJacobian should exist', () => {
    expect(mujoco.mjtJacobian).toBeDefined();
  });

  it('mjtSolver should exist', () => {
    expect(mujoco.mjtSolver).toBeDefined();
  });

  it('mjtEq should exist', () => {
    expect(mujoco.mjtEq).toBeDefined();
  });

  it('mjtWrap should exist', () => {
    expect(mujoco.mjtWrap).toBeDefined();
  });

  it('mjtTrn should exist', () => {
    expect(mujoco.mjtTrn).toBeDefined();
  });

  it('mjtDyn should exist', () => {
    expect(mujoco.mjtDyn).toBeDefined();
  });

  it('mjtGain should exist', () => {
    expect(mujoco.mjtGain).toBeDefined();
  });

  it('mjtBias should exist', () => {
    expect(mujoco.mjtBias).toBeDefined();
  });

  it('mjtObj should exist', () => {
    expect(mujoco.mjtObj).toBeDefined();
  });

  it('mjtSensor should exist', () => {
    expect(mujoco.mjtSensor).toBeDefined();
  });

  it('mjtStage should exist', () => {
    expect(mujoco.mjtStage).toBeDefined();
  });

  it('mjtDataType should exist', () => {
    expect(mujoco.mjtDataType).toBeDefined();
  });

  it('mjtConDataField should exist', () => {
    expect(mujoco.mjtConDataField).toBeDefined();
  });

  it('mjtRayDataField should exist', () => {
    expect(mujoco.mjtRayDataField).toBeDefined();
  });

  it('mjtCamOutBit should exist', () => {
    expect(mujoco.mjtCamOutBit).toBeDefined();
  });

  it('mjtSameFrame should exist', () => {
    expect(mujoco.mjtSameFrame).toBeDefined();
  });

  it('mjtSleepPolicy should exist', () => {
    expect(mujoco.mjtSleepPolicy).toBeDefined();
  });

  it('mjtLRMode should exist', () => {
    expect(mujoco.mjtLRMode).toBeDefined();
  });

  it('mjtFlexSelf should exist', () => {
    expect(mujoco.mjtFlexSelf).toBeDefined();
  });

  it('mjtSDFType should exist', () => {
    expect(mujoco.mjtSDFType).toBeDefined();
  });

  it('mjtState should exist', () => {
    expect(mujoco.mjtState).toBeDefined();
  });

  it('mjtConstraint should exist', () => {
    expect(mujoco.mjtConstraint).toBeDefined();
  });

  it('mjtConstraintState should exist', () => {
    expect(mujoco.mjtConstraintState).toBeDefined();
  });

  it('mjtWarning should exist', () => {
    expect(mujoco.mjtWarning).toBeDefined();
  });

  it('mjtTimer should exist', () => {
    expect(mujoco.mjtTimer).toBeDefined();
  });

  it('mjtSleepState should exist', () => {
    expect(mujoco.mjtSleepState).toBeDefined();
  });

  it('mjtGeomInertia should exist', () => {
    expect(mujoco.mjtGeomInertia).toBeDefined();
  });

  it('mjtMeshInertia should exist', () => {
    expect(mujoco.mjtMeshInertia).toBeDefined();
  });

  it('mjtMeshBuiltin should exist', () => {
    expect(mujoco.mjtMeshBuiltin).toBeDefined();
  });

  it('mjtFlexSection should exist', () => {
    expect(mujoco.mjtFlexSection).toBeDefined();
  });

  it('mjtBuiltin should exist', () => {
    expect(mujoco.mjtBuiltin).toBeDefined();
  });

  it('mjtMark should exist', () => {
    expect(mujoco.mjtMark).toBeDefined();
  });

  it('mjtLimited should exist', () => {
    expect(mujoco.mjtLimited).toBeDefined();
  });

  it('mjtAlignFree should exist', () => {
    expect(mujoco.mjtAlignFree).toBeDefined();
  });

  it('mjtInertiaFromGeom should exist', () => {
    expect(mujoco.mjtInertiaFromGeom).toBeDefined();
  });

  it('mjtOrientation should exist', () => {
    expect(mujoco.mjtOrientation).toBeDefined();
  });

  it('mjtCatBit should exist', () => {
    expect(mujoco.mjtCatBit).toBeDefined();
  });

  it('mjtMouse should exist', () => {
    expect(mujoco.mjtMouse).toBeDefined();
  });

  it('mjtPertBit should exist', () => {
    expect(mujoco.mjtPertBit).toBeDefined();
  });

  it('mjtCamera should exist', () => {
    expect(mujoco.mjtCamera).toBeDefined();
  });

  it('mjtLabel should exist', () => {
    expect(mujoco.mjtLabel).toBeDefined();
  });

  it('mjtFrame should exist', () => {
    expect(mujoco.mjtFrame).toBeDefined();
  });

  it('mjtVisFlag should exist', () => {
    expect(mujoco.mjtVisFlag).toBeDefined();
  });

  it('mjtRndFlag should exist', () => {
    expect(mujoco.mjtRndFlag).toBeDefined();
  });

  it('mjtStereo should exist', () => {
    expect(mujoco.mjtStereo).toBeDefined();
  });

  it('mjtPluginCapabilityBit should exist', () => {
    expect(mujoco.mjtPluginCapabilityBit).toBeDefined();
  });

  it('mjtGridPos should exist', () => {
    expect(mujoco.mjtGridPos).toBeDefined();
  });

  it('mjtFramebuffer should exist', () => {
    expect(mujoco.mjtFramebuffer).toBeDefined();
  });

  it('mjtDepthMap should exist', () => {
    expect(mujoco.mjtDepthMap).toBeDefined();
  });

  it('mjtFontScale should exist', () => {
    expect(mujoco.mjtFontScale).toBeDefined();
  });

  it('mjtFont should exist', () => {
    expect(mujoco.mjtFont).toBeDefined();
  });

  it('mjtButton should exist', () => {
    expect(mujoco.mjtButton).toBeDefined();
  });

  it('mjtEvent should exist', () => {
    expect(mujoco.mjtEvent).toBeDefined();
  });

  it('mjtItem should exist', () => {
    expect(mujoco.mjtItem).toBeDefined();
  });

  it('mjtSection should exist', () => {
    expect(mujoco.mjtSection).toBeDefined();
  });
});
