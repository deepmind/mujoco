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

#include "engine/engine_init.h"

#include <string.h>

#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_util_blas.h"

#ifdef _MSC_VER
  #pragma warning (disable: 4305)  // disable MSVC warning: truncation from 'double' to 'float'
#endif



//------------------------------- solver parameters ------------------------------------------------

// set default solver parameters
void mj_defaultSolRefImp(mjtNum* solref, mjtNum* solimp) {
  if (solref) {
    solref[0] = 0.02;       // timeconst
    solref[1] = 1;          // dampratio
  }

  if (solimp) {
    solimp[0] = 0.9;        // dmin
    solimp[1] = 0.95;       // dmax
    solimp[2] = 0.001;      // width
    solimp[3] = 0.5;        // midpoint
    solimp[4] = 2;          // power
  }
}


//------------------------------- mjOption ---------------------------------------------------------

// set model options to default values
void mj_defaultOption(mjOption* opt) {
  // fill opt with zeros in case struct is padded
  memset(opt, 0, sizeof(mjOption));

  // timing parameters
  opt->timestep           = 0.002;

  // solver parameters
  opt->impratio           = 1;
  opt->tolerance          = 1e-8;
  opt->ls_tolerance       = 0.01;
  opt->noslip_tolerance   = 1e-6;
  opt->ccd_tolerance      = 1e-6;

  // sleep settings
  opt->sleep_tolerance    = 1e-3;

  // physical constants
  opt->gravity[0]         = 0;
  opt->gravity[1]         = 0;
  opt->gravity[2]         = -9.81;
  opt->wind[0]            = 0;
  opt->wind[1]            = 0;
  opt->wind[2]            = 0;
  opt->magnetic[0]        = 0;
  opt->magnetic[1]        = -0.5;
  opt->magnetic[2]        = 0;
  opt->density            = 0;
  opt->viscosity          = 0;

  // solver overrides
  opt->o_margin           = 0;
  mj_defaultSolRefImp(opt->o_solref, opt->o_solimp);
  opt->o_friction[0] = 1;
  opt->o_friction[1] = 1;
  opt->o_friction[2] = 0.005;
  opt->o_friction[3] = 0.0001;
  opt->o_friction[4] = 0.0001;

  // discrete options
  opt->integrator         = mjINT_EULER;
  opt->cone               = mjCONE_PYRAMIDAL;
  opt->jacobian           = mjJAC_AUTO;
  opt->solver             = mjSOL_NEWTON;
  opt->iterations         = 100;
  opt->ls_iterations      = 50;
  opt->noslip_iterations  = 0;
  opt->ccd_iterations     = 35;
  opt->disableflags       = 0;
  opt->enableflags        = 0;
  opt->disableactuator    = 0;

  // sdf collisions
  opt->sdf_initpoints     = 40;
  opt->sdf_iterations     = 10;
}


//------------------------------- mjStatistic ------------------------------------------------------

// set statistics to default values; compute later in compiler
void mj_defaultStatistic(mjStatistic* stat) {
  mju_zero3(stat->center);
  stat->extent = 2;
  stat->meaninertia = 1;
  stat->meanmass = 1;
  stat->meansize = 0.2;
}


//------------------------------- mjVisual ---------------------------------------------------------

// set 4 floats
static void setf4(float* rgba, float r, float g, float b, float a) {
  rgba[0] = r;
  rgba[1] = g;
  rgba[2] = b;
  rgba[3] = a;
}


// set visual options to default values
void mj_defaultVisual(mjVisual* vis) {
  // global
  vis->global.cameraid            = -1;
  vis->global.orthographic        = 0;
  vis->global.fovy                = 45;
  vis->global.ipd                 = 0.068;
  vis->global.azimuth             = 90;
  vis->global.elevation           = -45;
  vis->global.linewidth           = 1.0;
  vis->global.glow                = 0.3;
  vis->global.offwidth            = 640;
  vis->global.offheight           = 480;
  vis->global.realtime            = 1.0;
  vis->global.ellipsoidinertia    = 0;
  vis->global.bvactive            = 0;

  // rendering quality
  vis->quality.shadowsize         = 4096;
  vis->quality.offsamples         = 4;
  vis->quality.numslices          = 28;
  vis->quality.numstacks          = 16;
  vis->quality.numquads           = 4;

  // head light
  vis->headlight.ambient[0]       = 0.1;
  vis->headlight.ambient[1]       = 0.1;
  vis->headlight.ambient[2]       = 0.1;
  vis->headlight.diffuse[0]       = 0.4;
  vis->headlight.diffuse[1]       = 0.4;
  vis->headlight.diffuse[2]       = 0.4;
  vis->headlight.specular[0]      = 0.5;
  vis->headlight.specular[1]      = 0.5;
  vis->headlight.specular[2]      = 0.5;
  vis->headlight.active           = 2;

  // map parameters
  vis->map.stiffness              = 100;
  vis->map.stiffnessrot           = 500;
  vis->map.force                  = 0.005;
  vis->map.torque                 = 0.1;
  vis->map.alpha                  = 0.3;
  vis->map.fogstart               = 3.0;
  vis->map.fogend                 = 10.0;
  vis->map.znear                  = 0.01;
  vis->map.zfar                   = 50.0;
  vis->map.haze                   = 0.3;
  vis->map.shadowclip             = 1.0;
  vis->map.shadowscale            = 0.6;
  vis->map.actuatortendon         = 2.0;

  // size parameters
  vis->scale.forcewidth           = 0.1;
  vis->scale.contactwidth         = 0.3;
  vis->scale.contactheight        = 0.1;
  vis->scale.connect              = 0.2;
  vis->scale.com                  = 0.4;
  vis->scale.camera               = 0.3;
  vis->scale.light                = 0.3;
  vis->scale.selectpoint          = 0.2;
  vis->scale.jointlength          = 1.0;
  vis->scale.jointwidth           = 0.1;
  vis->scale.actuatorlength       = 0.7;
  vis->scale.actuatorwidth        = 0.2;
  vis->scale.framelength          = 1.0;
  vis->scale.framewidth           = 0.1;
  vis->scale.constraint           = 0.1;
  vis->scale.slidercrank          = 0.2;
  vis->scale.frustum             = 10.0;

  // colors
  setf4(vis->rgba.fog,              0., 0., 0., 1.);
  setf4(vis->rgba.haze,             1., 1., 1., 1.);
  setf4(vis->rgba.force,            1., .5, .5, 1.);
  setf4(vis->rgba.inertia,          .8, .2, .2, .6);
  setf4(vis->rgba.joint,            .2, .6, .8, 1.);
  setf4(vis->rgba.actuator,         .2, .25, .2, 1);
  setf4(vis->rgba.actuatornegative, .2, .6, .9, 1.);
  setf4(vis->rgba.actuatorpositive, .9, .4, .2, 1.);
  setf4(vis->rgba.com,              .9, .9, .9, 1.);
  setf4(vis->rgba.camera,           .6, .9, .6, 1);
  setf4(vis->rgba.light,            .6, .6, .9, 1.);
  setf4(vis->rgba.selectpoint,      .9, .9, .1, 1.);
  setf4(vis->rgba.connect,          .2, .2, .8, 1.);
  setf4(vis->rgba.contactpoint,     .9, .6, .2, 1.);
  setf4(vis->rgba.contactforce,     .7, .9, .9, 1.);
  setf4(vis->rgba.contactfriction,  .9, .8, .4, 1.);
  setf4(vis->rgba.contacttorque,    .9, .7, .9, 1.);
  setf4(vis->rgba.contactgap,       .5, .8, .9, 1.);
  setf4(vis->rgba.rangefinder,      1., 1., .1, 1.);
  setf4(vis->rgba.constraint,       .9, .0, .0, 1.);
  setf4(vis->rgba.slidercrank,      .5, .3, .8, 1.);
  setf4(vis->rgba.crankbroken,      .9, .0, .0, 1.);
  setf4(vis->rgba.frustum,          1., 1., .0, .2);
  setf4(vis->rgba.bv,               0., 1., .0, .5);
  setf4(vis->rgba.bvactive,         1., 0., .0, .5);
}


//------------------------------ mjLROpt -----------------------------------------------------------

// set default options for length range computation
void mj_defaultLROpt(mjLROpt* opt) {
  opt->mode           = mjLRMODE_MUSCLE;
  opt->useexisting    = 1;
  opt->uselimit       = 0;

  opt->accel          = 20;
  opt->maxforce       = 0;
  opt->timeconst      = 1;
  opt->timestep       = 0.01;
  opt->inttotal       = 10;
  opt->interval       = 2;
  opt->tolrange       = 0.05;
}
