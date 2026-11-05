// Copyright 2024 DeepMind Technologies Limited
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

#include <string.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjspec.h>
#include "engine/engine_init.h"
#include "engine/engine_io.h"
#include "user/user_api.h"


// default model attributes
void mjs_defaultSpec(mjSpec* spec) {
  memset(spec, 0, sizeof(mjSpec));

  // default statistics
  spec->stat.meaninertia = mjNAN;
  spec->stat.meanmass    = mjNAN;
  spec->stat.meansize    = mjNAN;
  spec->stat.extent      = mjNAN;
  spec->stat.center[0]   = mjNAN;

  // compiler settings
  spec->compiler.autolimits           = 1;
  spec->compiler.settotalmass         = -1;
  spec->compiler.degree               = 1;
  spec->compiler.eulerseq[0]          = 'x';
  spec->compiler.eulerseq[1]          = 'y';
  spec->compiler.eulerseq[2]          = 'z';
  spec->compiler.usethread            = 1;
  spec->compiler.inertiafromgeom      = mjINERTIAFROMGEOM_AUTO;
  spec->compiler.inertiagrouprange[1] = mjNGROUP - 1;
  spec->compiler.saveinertial         = 0;
  spec->compiler.savecompiled         = 1;
  spec->compiler.savecanonical        = 1;
  spec->compiler.passthrough          = mjPASSTHROUGH_NONE;
  mj_defaultLROpt(&spec->compiler.LRopt);

  // engine data
  mj_defaultOption(&spec->option);
  mj_defaultVisual(&spec->visual);
  spec->memory  = -1;
  spec->njmax   = -1;
  spec->nconmax = -1;
  spec->nstack  = -1;

  // user fields
  spec->nuser_body     = -1;
  spec->nuser_jnt      = -1;
  spec->nuser_geom     = -1;
  spec->nuser_site     = -1;
  spec->nuser_cam      = -1;
  spec->nuser_tendon   = -1;
  spec->nuser_actuator = -1;
  spec->nuser_sensor   = -1;
}


// default orientation attributes
void mjs_defaultOrientation(mjsOrientation* orient) {
  memset(orient, 0, sizeof(mjsOrientation));
}


// default body attributes
void mjs_defaultBody(mjsBody* body) {
  memset(body, 0, sizeof(mjsBody));

  // body frame
  body->quat[0] = 1;

  // inertial frame
  body->ipos[0]        = mjNAN;
  body->iquat[0]       = 1;
  body->fullinertia[0] = mjNAN;

  // simple optimization: auto
  body->simple = 1;

  // fuse with parent when static: auto
  body->fuse = 1;
}


// default frame attributes
void mjs_defaultFrame(mjsFrame* frame) {
  memset(frame, 0, sizeof(mjsFrame));
  frame->quat[0]  = 1;
  frame->scale[0] = frame->scale[1] = frame->scale[2] = 1;
}


// default joint attributes
void mjs_defaultJoint(mjsJoint* joint) {
  memset(joint, 0, sizeof(mjsJoint));
  joint->type          = mjJNT_HINGE;
  joint->axis[2]       = 1;
  joint->limited       = mjLIMITED_AUTO;
  joint->actfrclimited = mjLIMITED_AUTO;
  joint->align         = mjALIGNFREE_AUTO;
  mj_defaultSolRefImp(joint->solref_limit, joint->solimp_limit);
  mj_defaultSolRefImp(joint->solref_friction, joint->solimp_friction);
}


// default geom attributes
void mjs_defaultGeom(mjsGeom* geom) {
  memset(geom, 0, sizeof(mjsGeom));

  // type
  geom->type = mjGEOM_SPHERE;

  // frame
  geom->quat[0]   = 1;
  geom->fromto[0] = mjNAN;

  // contact-related
  geom->contype     = 1;
  geom->conaffinity = 1;
  geom->condim      = 3;
  geom->friction[0] = 1;
  geom->friction[1] = 0.005;
  geom->friction[2] = 0.0001;
  geom->solmix      = 1.0;
  mj_defaultSolRefImp(geom->solref, geom->solimp);

  // inertia-related
  geom->mass        = mjNAN;
  geom->density     = 1000;  // water density (1000 Kg / m^3)
  geom->typeinertia = mjINERTIA_VOLUME;

  // color
  geom->rgba[0] = geom->rgba[1] = geom->rgba[2] = 0.5f;
  geom->rgba[3]                                 = 1.0f;
  geom->castshadow                              = 1;

  // fluid forces
  geom->fluid_coefs[0] = 0.5;   // blunt drag
  geom->fluid_coefs[1] = 0.25;  // slender drag
  geom->fluid_coefs[2] = 1.5;   // angular drag
  geom->fluid_coefs[3] = 1.0;   // kutta lift
  geom->fluid_coefs[4] = 1.0;   // magnus lift

  // other
  geom->fitscale = 1;
}


// default site attributes
void mjs_defaultSite(mjsSite* site) {
  memset(site, 0, sizeof(mjsSite));

  // type
  site->type = mjGEOM_SPHERE;

  // frame
  site->quat[0] = 1;
  site->size[0] = site->size[1] = site->size[2] = 0.005;
  site->fromto[0]                               = mjNAN;

  // color
  site->rgba[0] = site->rgba[1] = site->rgba[2] = 0.5f;
  site->rgba[3]                                 = 1.0f;
}


// default cam attributes
void mjs_defaultCamera(mjsCamera* cam) {
  memset(cam, 0, sizeof(mjsCamera));

  // mode
  cam->mode = mjCAMLIGHT_FIXED;

  // extrinsics
  cam->quat[0] = 1;

  // intrinsics
  cam->fovy          = 45;
  cam->ipd           = 0.068;
  cam->resolution[0] = cam->resolution[1] = 1;
  cam->output                             = mjCAMOUT_RGB;
}


// default light attributes
void mjs_defaultLight(mjsLight* light) {
  memset(light, 0, sizeof(mjsLight));

  // mode
  light->mode = mjCAMLIGHT_FIXED;

  // extrinsics
  light->dir[2] = -1;

  // intrinsics
  light->castshadow     = 1;
  light->bulbradius     = 0.02;
  light->intensity      = 0.0;
  light->range          = 10.0;
  light->active         = 1;
  light->attenuation[0] = 1;
  light->cutoff         = 45;
  light->softness       = 0.2;
  light->exponent       = 10;
  light->diffuse[0] = light->diffuse[1] = light->diffuse[2] = 0.7;
  light->specular[0] = light->specular[1] = light->specular[2] = 0.3;
}


// default flex attributes
void mjs_defaultFlex(mjsFlex* flex) {
  memset(flex, 0, sizeof(mjsFlex));

  // set contact defaults
  flex->contype     = 1;
  flex->conaffinity = 1;
  flex->condim      = 3;
  flex->friction[0] = 1;
  flex->friction[1] = 0.005;
  flex->friction[2] = 0.0001;
  flex->solmix      = 1.0;
  mj_defaultSolRefImp(flex->solref, flex->solimp);

  // set other defaults
  flex->dim          = 2;
  flex->radius       = 0.005;
  flex->cellcount[0] = 1;
  flex->cellcount[1] = 1;
  flex->cellcount[2] = 1;
  flex->selfcollide  = mjFLEXSELF_AUTO;
  flex->activelayers = 1;
  flex->rgba[0] = flex->rgba[1] = flex->rgba[2] = 0.5f;
  flex->rgba[3]                                 = 1.0f;
  flex->thickness                               = -1;
}


// default mesh attributes
void mjs_defaultMesh(mjsMesh* mesh) {
  memset(mesh, 0, sizeof(mjsMesh));
  mesh->refquat[0] = 1;
  mesh->scale[0] = mesh->scale[1] = mesh->scale[2] = 1;
  mesh->maxhullvert                                = -1;
  mesh->inertia                                    = mjMESH_INERTIA_LEGACY;
  mesh->octree_maxdepth                            = 6;
}


// default height field attributes
void mjs_defaultHField(mjsHField* hfield) {
  memset(hfield, 0, sizeof(mjsHField));
}


// default skin attributes
void mjs_defaultSkin(mjsSkin* skin) {
  memset(skin, 0, sizeof(mjsSkin));
  skin->rgba[0] = skin->rgba[1] = skin->rgba[2] = 0.5f;
  skin->rgba[3]                                 = 1.0f;
}


// default texture attributes
void mjs_defaultTexture(mjsTexture* texture) {
  memset(texture, 0, sizeof(mjsTexture));
  texture->type       = mjTEXTURE_CUBE;
  texture->colorspace = mjCOLORSPACE_AUTO;
  texture->rgb1[0] = texture->rgb1[1] = texture->rgb1[2] = 0.8;
  texture->rgb2[0] = texture->rgb2[1] = texture->rgb2[2] = 0.5;
  texture->random                                        = 0.01;
  texture->gridsize[0] = texture->gridsize[1] = 1;
  texture->nchannel                           = 3;
  texture->embed                              = mjEMBED_AUTO;
  memcpy(texture->gridlayout, "............", sizeof(texture->gridlayout));
}


// default material attributes
void mjs_defaultMaterial(mjsMaterial* material) {
  memset(material, 0, sizeof(mjsMaterial));
  material->texrepeat[0] = 1;
  material->texrepeat[1] = 1;
  material->specular     = 0.5;
  material->shininess    = 0.5;
  material->metallic     = -1.0;
  material->roughness    = -1.0;
  material->rgba[0] = material->rgba[1] = material->rgba[2] = material->rgba[3] = 1;
}


// default pair attributes
void mjs_defaultPair(mjsPair* pair) {
  memset(pair, 0, sizeof(mjsPair));
  pair->condim = 3;
  mj_defaultSolRefImp(pair->solref, pair->solimp);
  pair->friction[0] = 1;
  pair->friction[1] = 1;
  pair->friction[2] = 0.005;
  pair->friction[3] = 0.0001;
  pair->friction[4] = 0.0001;
}


// default equality attributes
void mjs_defaultEquality(mjsEquality* equality) {
  memset(equality, 0, sizeof(mjsEquality));
  equality->type   = mjEQ_CONNECT;
  equality->active = 1;
  mj_defaultSolRefImp(equality->solref, equality->solimp);
  equality->data[1]  = 1;
  equality->data[10] = 1;  // torque:force ratio
}


// default tendon attributes
void mjs_defaultTendon(mjsTendon* tendon) {
  memset(tendon, 0, sizeof(mjsTendon));
  tendon->limited         = mjLIMITED_AUTO;
  tendon->actfrclimited   = mjLIMITED_AUTO;
  tendon->springlength[0] = tendon->springlength[1] = -1;
  mj_defaultSolRefImp(tendon->solref_limit, tendon->solimp_limit);
  mj_defaultSolRefImp(tendon->solref_friction, tendon->solimp_friction);
  tendon->width   = 0.003;
  tendon->rgba[0] = 0.5f;
  tendon->rgba[1] = 0.5f;
  tendon->rgba[2] = 0.5f;
  tendon->rgba[3] = 1.0f;
}


// default actuator attributes
void mjs_defaultActuator(mjsActuator* actuator) {
  memset(actuator, 0, sizeof(mjsActuator));

  // gain, bias
  actuator->gaintype   = mjGAIN_FIXED;
  actuator->gainprm[0] = 1;
  actuator->biastype   = mjBIAS_NONE;

  // activation state
  actuator->dyntype     = mjDYN_NONE;
  actuator->dynprm[0]   = 1;
  actuator->actdim      = -1;
  actuator->ctrlspec    = 0;
  actuator->velrange[0] = actuator->velrange[1] = 0;
  actuator->ffrange[0] = actuator->ffrange[1] = 0;

  // transmission
  actuator->trntype = mjTRN_UNDEFINED;
  actuator->gear[0] = 1;

  // input/output clamping
  actuator->ctrllimited  = mjLIMITED_AUTO;
  actuator->forcelimited = mjLIMITED_AUTO;
  actuator->actlimited   = mjLIMITED_AUTO;
}


// default sensor attributes
void mjs_defaultSensor(mjsSensor* sensor) {
  memset(sensor, 0, sizeof(mjsSensor));

  sensor->type      = mjSENS_TOUCH;
  sensor->datatype  = mjDATATYPE_REAL;
  sensor->needstage = mjSTAGE_ACC;
}


// Default numeric attributes.
void mjs_defaultNumeric(mjsNumeric* numeric) {
  memset(numeric, 0, sizeof(mjsNumeric));
}


// Default text attributes.
void mjs_defaultText(mjsText* text) {
  memset(text, 0, sizeof(mjsText));
}


// Default tuple attributes.
void mjs_defaultTuple(mjsTuple* tuple) {
  memset(tuple, 0, sizeof(mjsTuple));
}


// Default keyframe attributes.
void mjs_defaultKey(mjsKey* key) {
  memset(key, 0, sizeof(mjsKey));
}


// default plugin attributes
void mjs_defaultPlugin(mjsPlugin* plugin) {
  memset(plugin, 0, sizeof(mjsPlugin));
}
