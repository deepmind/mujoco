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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/experimental/usd/mjcPhysics/sceneAPI.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/vt/types.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdPhysics/scene.h>

namespace mujoco {
namespace {

using pxr::SdfPath;
using MjcPhysicsSceneTest = MujocoTest;
using testing::NotNull;

// clang-format off
#define EXPECT_TYPE_USD_FALLBACK_EQ_MODEL_DEFAULT(type, usd_attr, mjc_attr) \
  {                                                                   \
    type value;                                                     \
    mjc_phys_scene.Get##usd_attr##Attr().Get(&value);                 \
    EXPECT_EQ((mjtNum)value, default_model->opt.mjc_attr);            \
  }


#define EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_attr) \
    EXPECT_TYPE_USD_FALLBACK_EQ_MODEL_DEFAULT(double, usd_attr, mjc_attr)

#define EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_flag) \
  {                                                                         \
    bool flag;                                                              \
    mjc_phys_scene.Get##usd_attr##Attr().Get(&flag);                        \
    EXPECT_NE(flag, default_model->opt.disableflags & (mjc_flag));          \
  }

#define EXPECT_ENABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_flag) \
  {                                                                         \
    bool flag;                                                              \
    mjc_phys_scene.Get##usd_attr##Attr().Get(&flag);                        \
    EXPECT_EQ(flag, default_model->opt.enableflags & (mjc_flag));          \
  }


#define EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_attr) \
    EXPECT_TYPE_USD_FALLBACK_EQ_MODEL_DEFAULT(int, usd_attr, mjc_attr)

#define EXPECT_VEC3_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_attr) \
  {                                                                   \
    pxr::GfVec3d value;                                               \
    mjc_phys_scene.Get##usd_attr##Attr().Get(&value);                 \
    EXPECT_EQ(value[0], default_model->opt.mjc_attr[0]);              \
    EXPECT_EQ(value[1], default_model->opt.mjc_attr[1]);              \
    EXPECT_EQ(value[2], default_model->opt.mjc_attr[2]);              \
  }

#define EXPECT_TYPE_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(type, usd_attr, mjc_attr)\
  {                                                                      \
    pxr::Vt##type##Array mjc_attr;                                         \
    mjc_phys_scene.Get##usd_attr##Attr().Get(&mjc_attr);                 \
    EXPECT_THAT(                                                         \
            mjc_attr,                                                    \
            testing::ElementsAreArray(default_model->opt.mjc_attr)       \
    );                                                                   \
  }

#define EXPECT_REAL_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_attr)\
     EXPECT_TYPE_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(Double, usd_attr, mjc_attr)

#define EXPECT_INT_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(usd_attr, mjc_attr)\
     EXPECT_TYPE_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(Int, usd_attr, mjc_attr)
// clang-format on

TEST_F(MjcPhysicsSceneTest, TestDefaults) {
  auto stage = pxr::UsdStage::CreateInMemory();

  auto physics_scene =
      pxr::UsdPhysicsScene::Define(stage, SdfPath("/World/PhysicsScene"));

  auto mjc_phys_scene = pxr::MjcPhysicsSceneAPI::Apply(physics_scene.GetPrim());

  mjSpec* empty_spec = mj_makeSpec();
  mjModel* default_model = mj_compile(empty_spec, nullptr);
  EXPECT_THAT(default_model, NotNull());

  // Check that all the USD schema fallback values are the same
  // as the model defaults.
  // If this test is failing due to an update of defaults in Mujoco you need to
  // update mjcPhysics/schema.usda.
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(Timestep, timestep);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(ImpRatio, impratio);

  EXPECT_VEC3_USD_FALLBACK_EQ_MODEL_DEFAULT(Wind, wind);
  EXPECT_VEC3_USD_FALLBACK_EQ_MODEL_DEFAULT(Magnetic, magnetic);

  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(Density, density);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(Viscosity, viscosity);

  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(OMargin, o_margin);
  EXPECT_REAL_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(OSolRef, o_solref);
  EXPECT_REAL_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(OSolImp, o_solimp);
  EXPECT_REAL_ARR_USD_FALLBACK_EQ_MODEL_DEFAULT(OFriction, o_friction);

  EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(Iterations, iterations);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(Tolerance, tolerance);
  EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(LSIterations, ls_iterations);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(LSTolerance, ls_tolerance);
  EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(NoslipIterations, noslip_iterations);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(NoslipTolerance, noslip_tolerance);
  EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(CCDIterations, ccd_iterations);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(CCDTolerance, ccd_tolerance);
  EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(SDFIterations, sdf_iterations);
  EXPECT_INT_USD_FALLBACK_EQ_MODEL_DEFAULT(SDFInitPoints, sdf_initpoints);
  EXPECT_REAL_USD_FALLBACK_EQ_MODEL_DEFAULT(SleepTolerance, sleep_tolerance);

  // We store the actuator disable groups as an array of integers, but the
  // model stores it as a bitmask.
  pxr::VtIntArray actuator_group_disable;
  mjc_phys_scene.GetActuatorGroupDisableAttr().Get(&actuator_group_disable);
  int bitmask = 0;
  for (int ind : actuator_group_disable) {
    bitmask |= 1 << ind;
  }
  EXPECT_EQ(default_model->opt.disableactuator, bitmask);

  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(ConstraintFlag,
                                                    mjDSBL_CONSTRAINT);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(EqualityFlag,
                                                    mjDSBL_EQUALITY);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(FrictionLossFlag,
                                                    mjDSBL_FRICTIONLOSS);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(LimitFlag, mjDSBL_LIMIT);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(ContactFlag,
                                                    mjDSBL_CONTACT);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(SpringFlag, mjDSBL_SPRING);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(DamperFlag, mjDSBL_DAMPER);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(GravityFlag,
                                                    mjDSBL_GRAVITY);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(ClampCtrlFlag,
                                                    mjDSBL_CLAMPCTRL);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(WarmStartFlag,
                                                    mjDSBL_WARMSTART);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(FilterParentFlag,
                                                    mjDSBL_FILTERPARENT);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(ActuationFlag,
                                                    mjDSBL_ACTUATION);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(RefSafeFlag,
                                                    mjDSBL_REFSAFE);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(SensorFlag, mjDSBL_SENSOR);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(MidPhaseFlag,
                                                    mjDSBL_MIDPHASE);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(NativeCCDFlag,
                                                    mjDSBL_NATIVECCD);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(EulerDampFlag,
                                                    mjDSBL_EULERDAMP);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(AutoResetFlag,
                                                    mjDSBL_AUTORESET);
  EXPECT_DISABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(MultiCCDFlag,
                                                    mjDSBL_MULTICCD);

  EXPECT_ENABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(OverrideFlag,
                                                   mjENBL_OVERRIDE);
  EXPECT_ENABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(EnergyFlag, mjENBL_ENERGY);
  EXPECT_ENABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(FwdinvFlag, mjENBL_FWDINV);
  EXPECT_ENABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(InvDiscreteFlag,
                                                   mjENBL_INVDISCRETE);
  EXPECT_ENABLE_FLAG_USD_FALLBACK_EQ_MODEL_DEFAULT(SleepFlag, mjENBL_SLEEP);

  mj_deleteModel(default_model);
  mj_deleteSpec(empty_spec);
}

}  // namespace
}  // namespace mujoco
