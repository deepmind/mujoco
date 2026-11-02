// Copyright 2026 DeepMind Technologies Limited
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

// GENERATED FILE, DO NOT EDIT. Generated from src/xml/mjcf.schema by
// doc/generate/generate_default_table.py; test/doc/doc_test.py checks
// freshness.
//
// One row per bound numeric schema attribute -- total coverage, whether or
// not a default is declared. SchemaDefaultsTest compares every row against a
// freshly-constructed spec: declared values must match the C
// default-constructors, values beyond ndecl must be zero, and rows marked
// unset must hold the mjNAN sentinel. A nonzero constructor default with no
// schema declaration is therefore a test failure: the schema cannot silently
// under-declare. Rows are {attr, offset, kind, len, ndecl, unset, values};
// kind: 0=double 1=float 2=int 3=byte 4=mjtNum.

// clang-format off
struct mjXDefaultEntry {
  const char* attr;
  int offset;
  int kind;
  int len;
  int ndecl;
  int unset;
  double value[8];
};

struct mjXDefaultTable {
  const char* structname;
  const mjXDefaultEntry* entries;
  int n;
};


static const mjXDefaultEntry kDefaults_mjLROpt[] = {
  {"mode", (int)offsetof(mjLROpt, mode), 2, 1, 1, 0, {(double)mjLRMODE_MUSCLE}},
  {"useexisting", (int)offsetof(mjLROpt, useexisting), 2, 1, 1, 0, {1}},
  {"uselimit", (int)offsetof(mjLROpt, uselimit), 2, 1, 0, 0, {0}},
  {"accel", (int)offsetof(mjLROpt, accel), 4, 1, 1, 0, {20.0}},
  {"maxforce", (int)offsetof(mjLROpt, maxforce), 4, 1, 0, 0, {0}},
  {"timeconst", (int)offsetof(mjLROpt, timeconst), 4, 1, 1, 0, {1.0}},
  {"timestep", (int)offsetof(mjLROpt, timestep), 4, 1, 1, 0, {0.01}},
  {"inttotal", (int)offsetof(mjLROpt, inttotal), 4, 1, 1, 0, {10.0}},
  {"interval", (int)offsetof(mjLROpt, interval), 4, 1, 1, 0, {2.0}},
  {"tolrange", (int)offsetof(mjLROpt, tolrange), 4, 1, 1, 0, {0.05}},
};

static const mjXDefaultEntry kDefaults_mjOption[] = {
  {"timestep", (int)offsetof(mjOption, timestep), 4, 1, 1, 0, {0.002}},
  {"impratio", (int)offsetof(mjOption, impratio), 4, 1, 1, 0, {1.0}},
  {"tolerance", (int)offsetof(mjOption, tolerance), 4, 1, 1, 0, {1e-08}},
  {"ls_tolerance", (int)offsetof(mjOption, ls_tolerance), 4, 1, 1, 0, {0.01}},
  {"noslip_tolerance", (int)offsetof(mjOption, noslip_tolerance), 4, 1, 1, 0, {1e-06}},
  {"ccd_tolerance", (int)offsetof(mjOption, ccd_tolerance), 4, 1, 1, 0, {1e-06}},
  {"sleep_tolerance", (int)offsetof(mjOption, sleep_tolerance), 4, 1, 1, 0, {0.001}},
  {"gravity", (int)offsetof(mjOption, gravity), 4, 3, 3, 0, {0.0, 0.0, -9.81}},
  {"wind", (int)offsetof(mjOption, wind), 4, 3, 3, 0, {0.0, 0.0, 0.0}},
  {"magnetic", (int)offsetof(mjOption, magnetic), 4, 3, 3, 0, {0.0, -0.5, 0.0}},
  {"density", (int)offsetof(mjOption, density), 4, 1, 0, 0, {0}},
  {"viscosity", (int)offsetof(mjOption, viscosity), 4, 1, 0, 0, {0}},
  {"o_margin", (int)offsetof(mjOption, o_margin), 4, 1, 0, 0, {0}},
  {"o_solref", (int)offsetof(mjOption, o_solref), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"o_solimp", (int)offsetof(mjOption, o_solimp), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"o_friction", (int)offsetof(mjOption, o_friction), 4, 5, 5, 0, {1.0, 1.0, 0.005, 0.0001, 0.0001}},
  {"integrator", (int)offsetof(mjOption, integrator), 2, 1, 1, 0, {(double)mjINT_EULER}},
  {"cone", (int)offsetof(mjOption, cone), 2, 1, 1, 0, {(double)mjCONE_PYRAMIDAL}},
  {"jacobian", (int)offsetof(mjOption, jacobian), 2, 1, 1, 0, {(double)mjJAC_AUTO}},
  {"solver", (int)offsetof(mjOption, solver), 2, 1, 1, 0, {(double)mjSOL_NEWTON}},
  {"iterations", (int)offsetof(mjOption, iterations), 2, 1, 1, 0, {100.0}},
  {"ls_iterations", (int)offsetof(mjOption, ls_iterations), 2, 1, 1, 0, {50.0}},
  {"noslip_iterations", (int)offsetof(mjOption, noslip_iterations), 2, 1, 0, 0, {0}},
  {"ccd_iterations", (int)offsetof(mjOption, ccd_iterations), 2, 1, 1, 0, {35.0}},
  {"sdf_iterations", (int)offsetof(mjOption, sdf_iterations), 2, 1, 1, 0, {10.0}},
  {"sdf_initpoints", (int)offsetof(mjOption, sdf_initpoints), 2, 1, 1, 0, {40.0}},
};

static const mjXDefaultEntry kDefaults_mjSpec[] = {
  {"njmax", (int)offsetof(mjSpec, njmax), 2, 1, 1, 0, {-1.0}},
  {"nconmax", (int)offsetof(mjSpec, nconmax), 2, 1, 1, 0, {-1.0}},
  {"nstack", (int)offsetof(mjSpec, nstack), 2, 1, 1, 0, {-1.0}},
  {"nuserdata", (int)offsetof(mjSpec, nuserdata), 2, 1, 0, 0, {0}},
  {"nkey", (int)offsetof(mjSpec, nkey), 2, 1, 0, 0, {0}},
  {"nuser_body", (int)offsetof(mjSpec, nuser_body), 2, 1, 1, 0, {-1.0}},
  {"nuser_jnt", (int)offsetof(mjSpec, nuser_jnt), 2, 1, 1, 0, {-1.0}},
  {"nuser_geom", (int)offsetof(mjSpec, nuser_geom), 2, 1, 1, 0, {-1.0}},
  {"nuser_site", (int)offsetof(mjSpec, nuser_site), 2, 1, 1, 0, {-1.0}},
  {"nuser_cam", (int)offsetof(mjSpec, nuser_cam), 2, 1, 1, 0, {-1.0}},
  {"nuser_tendon", (int)offsetof(mjSpec, nuser_tendon), 2, 1, 1, 0, {-1.0}},
  {"nuser_actuator", (int)offsetof(mjSpec, nuser_actuator), 2, 1, 1, 0, {-1.0}},
  {"nuser_sensor", (int)offsetof(mjSpec, nuser_sensor), 2, 1, 1, 0, {-1.0}},
};

static const mjXDefaultEntry kDefaults_mjStatistic[] = {
  {"meaninertia", (int)offsetof(mjStatistic, meaninertia), 4, 1, 0, 1, {0}},
  {"meanmass", (int)offsetof(mjStatistic, meanmass), 4, 1, 0, 1, {0}},
  {"meansize", (int)offsetof(mjStatistic, meansize), 4, 1, 0, 1, {0}},
  {"extent", (int)offsetof(mjStatistic, extent), 4, 1, 0, 1, {0}},
  {"center", (int)offsetof(mjStatistic, center), 4, 3, 0, 1, {0}},
};

static const mjXDefaultEntry kDefaults_mjVisual_global[] = {
  {"cameraid", (int)offsetof(mjVisual, global.cameraid), 2, 1, 1, 0, {-1.0}},
  {"orthographic", (int)offsetof(mjVisual, global.orthographic), 2, 1, 1, 0, {0}},
  {"fovy", (int)offsetof(mjVisual, global.fovy), 1, 1, 1, 0, {45.0}},
  {"ipd", (int)offsetof(mjVisual, global.ipd), 1, 1, 1, 0, {0.068}},
  {"azimuth", (int)offsetof(mjVisual, global.azimuth), 1, 1, 1, 0, {90.0}},
  {"elevation", (int)offsetof(mjVisual, global.elevation), 1, 1, 1, 0, {-45.0}},
  {"linewidth", (int)offsetof(mjVisual, global.linewidth), 1, 1, 1, 0, {1.0}},
  {"glow", (int)offsetof(mjVisual, global.glow), 1, 1, 1, 0, {0.3}},
  {"offwidth", (int)offsetof(mjVisual, global.offwidth), 2, 1, 1, 0, {640.0}},
  {"offheight", (int)offsetof(mjVisual, global.offheight), 2, 1, 1, 0, {480.0}},
  {"realtime", (int)offsetof(mjVisual, global.realtime), 1, 1, 1, 0, {1.0}},
  {"ellipsoidinertia", (int)offsetof(mjVisual, global.ellipsoidinertia), 2, 1, 1, 0, {0}},
  {"bvactive", (int)offsetof(mjVisual, global.bvactive), 2, 1, 1, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjVisual_headlight[] = {
  {"ambient", (int)offsetof(mjVisual, headlight.ambient), 1, 3, 3, 0, {0.1, 0.1, 0.1}},
  {"diffuse", (int)offsetof(mjVisual, headlight.diffuse), 1, 3, 3, 0, {0.4, 0.4, 0.4}},
  {"specular", (int)offsetof(mjVisual, headlight.specular), 1, 3, 3, 0, {0.5, 0.5, 0.5}},
  {"active", (int)offsetof(mjVisual, headlight.active), 2, 1, 1, 0, {1.0}},
};

static const mjXDefaultEntry kDefaults_mjVisual_map[] = {
  {"stiffness", (int)offsetof(mjVisual, map.stiffness), 1, 1, 1, 0, {100.0}},
  {"stiffnessrot", (int)offsetof(mjVisual, map.stiffnessrot), 1, 1, 1, 0, {500.0}},
  {"force", (int)offsetof(mjVisual, map.force), 1, 1, 1, 0, {0.005}},
  {"torque", (int)offsetof(mjVisual, map.torque), 1, 1, 1, 0, {0.1}},
  {"alpha", (int)offsetof(mjVisual, map.alpha), 1, 1, 1, 0, {0.3}},
  {"fogstart", (int)offsetof(mjVisual, map.fogstart), 1, 1, 1, 0, {3.0}},
  {"fogend", (int)offsetof(mjVisual, map.fogend), 1, 1, 1, 0, {10.0}},
  {"znear", (int)offsetof(mjVisual, map.znear), 1, 1, 1, 0, {0.01}},
  {"zfar", (int)offsetof(mjVisual, map.zfar), 1, 1, 1, 0, {50.0}},
  {"haze", (int)offsetof(mjVisual, map.haze), 1, 1, 1, 0, {0.3}},
  {"shadowclip", (int)offsetof(mjVisual, map.shadowclip), 1, 1, 1, 0, {1.0}},
  {"shadowscale", (int)offsetof(mjVisual, map.shadowscale), 1, 1, 1, 0, {0.6}},
  {"actuatortendon", (int)offsetof(mjVisual, map.actuatortendon), 1, 1, 1, 0, {2.0}},
};

static const mjXDefaultEntry kDefaults_mjVisual_quality[] = {
  {"shadowsize", (int)offsetof(mjVisual, quality.shadowsize), 2, 1, 1, 0, {4096.0}},
  {"offsamples", (int)offsetof(mjVisual, quality.offsamples), 2, 1, 1, 0, {4.0}},
  {"numslices", (int)offsetof(mjVisual, quality.numslices), 2, 1, 1, 0, {28.0}},
  {"numstacks", (int)offsetof(mjVisual, quality.numstacks), 2, 1, 1, 0, {16.0}},
  {"numquads", (int)offsetof(mjVisual, quality.numquads), 2, 1, 1, 0, {4.0}},
};

static const mjXDefaultEntry kDefaults_mjVisual_rgba[] = {
  {"fog", (int)offsetof(mjVisual, rgba.fog), 1, 4, 4, 0, {0.0, 0.0, 0.0, 1.0}},
  {"haze", (int)offsetof(mjVisual, rgba.haze), 1, 4, 4, 0, {1.0, 1.0, 1.0, 1.0}},
  {"force", (int)offsetof(mjVisual, rgba.force), 1, 4, 4, 0, {1.0, 0.5, 0.5, 1.0}},
  {"inertia", (int)offsetof(mjVisual, rgba.inertia), 1, 4, 4, 0, {0.8, 0.2, 0.2, 0.6}},
  {"joint", (int)offsetof(mjVisual, rgba.joint), 1, 4, 4, 0, {0.2, 0.6, 0.8, 1.0}},
  {"actuator", (int)offsetof(mjVisual, rgba.actuator), 1, 4, 4, 0, {0.2, 0.25, 0.2, 1.0}},
  {"actuatornegative", (int)offsetof(mjVisual, rgba.actuatornegative), 1, 4, 4, 0, {0.2, 0.6, 0.9, 1.0}},
  {"actuatorpositive", (int)offsetof(mjVisual, rgba.actuatorpositive), 1, 4, 4, 0, {0.9, 0.4, 0.2, 1.0}},
  {"com", (int)offsetof(mjVisual, rgba.com), 1, 4, 4, 0, {0.9, 0.9, 0.9, 1.0}},
  {"camera", (int)offsetof(mjVisual, rgba.camera), 1, 4, 4, 0, {0.6, 0.9, 0.6, 1.0}},
  {"light", (int)offsetof(mjVisual, rgba.light), 1, 4, 4, 0, {0.6, 0.6, 0.9, 1.0}},
  {"selectpoint", (int)offsetof(mjVisual, rgba.selectpoint), 1, 4, 4, 0, {0.9, 0.9, 0.1, 1.0}},
  {"connect", (int)offsetof(mjVisual, rgba.connect), 1, 4, 4, 0, {0.2, 0.2, 0.8, 1.0}},
  {"contactpoint", (int)offsetof(mjVisual, rgba.contactpoint), 1, 4, 4, 0, {0.9, 0.6, 0.2, 1.0}},
  {"contactforce", (int)offsetof(mjVisual, rgba.contactforce), 1, 4, 4, 0, {0.7, 0.9, 0.9, 1.0}},
  {"contactfriction", (int)offsetof(mjVisual, rgba.contactfriction), 1, 4, 4, 0, {0.9, 0.8, 0.4, 1.0}},
  {"contacttorque", (int)offsetof(mjVisual, rgba.contacttorque), 1, 4, 4, 0, {0.9, 0.7, 0.9, 1.0}},
  {"contactgap", (int)offsetof(mjVisual, rgba.contactgap), 1, 4, 4, 0, {0.5, 0.8, 0.9, 1.0}},
  {"rangefinder", (int)offsetof(mjVisual, rgba.rangefinder), 1, 4, 4, 0, {1.0, 1.0, 0.1, 1.0}},
  {"constraint", (int)offsetof(mjVisual, rgba.constraint), 1, 4, 4, 0, {0.9, 0.0, 0.0, 1.0}},
  {"slidercrank", (int)offsetof(mjVisual, rgba.slidercrank), 1, 4, 4, 0, {0.5, 0.3, 0.8, 1.0}},
  {"crankbroken", (int)offsetof(mjVisual, rgba.crankbroken), 1, 4, 4, 0, {0.9, 0.0, 0.0, 1.0}},
  {"frustum", (int)offsetof(mjVisual, rgba.frustum), 1, 4, 4, 0, {1.0, 1.0, 0.0, 0.2}},
  {"bv", (int)offsetof(mjVisual, rgba.bv), 1, 4, 4, 0, {0.0, 1.0, 0.0, 0.5}},
  {"bvactive", (int)offsetof(mjVisual, rgba.bvactive), 1, 4, 4, 0, {1.0, 0.0, 0.0, 0.5}},
};

static const mjXDefaultEntry kDefaults_mjVisual_scale[] = {
  {"forcewidth", (int)offsetof(mjVisual, scale.forcewidth), 1, 1, 1, 0, {0.1}},
  {"contactwidth", (int)offsetof(mjVisual, scale.contactwidth), 1, 1, 1, 0, {0.3}},
  {"contactheight", (int)offsetof(mjVisual, scale.contactheight), 1, 1, 1, 0, {0.1}},
  {"connect", (int)offsetof(mjVisual, scale.connect), 1, 1, 1, 0, {0.2}},
  {"com", (int)offsetof(mjVisual, scale.com), 1, 1, 1, 0, {0.4}},
  {"camera", (int)offsetof(mjVisual, scale.camera), 1, 1, 1, 0, {0.3}},
  {"light", (int)offsetof(mjVisual, scale.light), 1, 1, 1, 0, {0.3}},
  {"selectpoint", (int)offsetof(mjVisual, scale.selectpoint), 1, 1, 1, 0, {0.2}},
  {"jointlength", (int)offsetof(mjVisual, scale.jointlength), 1, 1, 1, 0, {1.0}},
  {"jointwidth", (int)offsetof(mjVisual, scale.jointwidth), 1, 1, 1, 0, {0.1}},
  {"actuatorlength", (int)offsetof(mjVisual, scale.actuatorlength), 1, 1, 1, 0, {0.7}},
  {"actuatorwidth", (int)offsetof(mjVisual, scale.actuatorwidth), 1, 1, 1, 0, {0.2}},
  {"framelength", (int)offsetof(mjVisual, scale.framelength), 1, 1, 1, 0, {1.0}},
  {"framewidth", (int)offsetof(mjVisual, scale.framewidth), 1, 1, 1, 0, {0.1}},
  {"constraint", (int)offsetof(mjVisual, scale.constraint), 1, 1, 1, 0, {0.1}},
  {"slidercrank", (int)offsetof(mjVisual, scale.slidercrank), 1, 1, 1, 0, {0.2}},
  {"frustum", (int)offsetof(mjVisual, scale.frustum), 1, 1, 1, 0, {10.0}},
};

static const mjXDefaultEntry kDefaults_mjsActuator[] = {
  {"group", (int)offsetof(mjsActuator, group), 2, 1, 0, 0, {0}},
  {"nsample", (int)offsetof(mjsActuator, nsample), 2, 1, 0, 0, {0}},
  {"interp", (int)offsetof(mjsActuator, interp), 2, 1, 0, 0, {0}},
  {"delay", (int)offsetof(mjsActuator, delay), 0, 1, 0, 0, {0}},
  {"ctrlrange", (int)offsetof(mjsActuator, ctrlrange), 0, 2, 0, 0, {0}},
  {"ctrllimited", (int)offsetof(mjsActuator, ctrllimited), 2, 1, 1, 0, {(double)2}},
  {"forcelimited", (int)offsetof(mjsActuator, forcelimited), 2, 1, 1, 0, {(double)2}},
  {"actlimited", (int)offsetof(mjsActuator, actlimited), 2, 1, 1, 0, {(double)2}},
  {"forcerange", (int)offsetof(mjsActuator, forcerange), 0, 2, 0, 0, {0}},
  {"actrange", (int)offsetof(mjsActuator, actrange), 0, 2, 0, 0, {0}},
  {"lengthrange", (int)offsetof(mjsActuator, lengthrange), 0, 2, 0, 0, {0}},
  {"gear", (int)offsetof(mjsActuator, gear), 0, 6, 6, 0, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
  {"damping", (int)offsetof(mjsActuator, damping), 0, mjNPOLY+1, 0, 0, {0}},
  {"armature", (int)offsetof(mjsActuator, armature), 0, 1, 0, 0, {0}},
  {"cranklength", (int)offsetof(mjsActuator, cranklength), 0, 1, 0, 0, {0}},
  {"actdim", (int)offsetof(mjsActuator, actdim), 2, 1, 1, 0, {-1.0}},
  {"velrange", (int)offsetof(mjsActuator, velrange), 0, 2, 0, 0, {0}},
  {"ffrange", (int)offsetof(mjsActuator, ffrange), 0, 2, 0, 0, {0}},
  {"inheritrange", (int)offsetof(mjsActuator, inheritrange), 0, 1, 0, 0, {0}},
  {"dyntype", (int)offsetof(mjsActuator, dyntype), 2, 1, 1, 0, {(double)mjDYN_NONE}},
  {"gaintype", (int)offsetof(mjsActuator, gaintype), 2, 1, 1, 0, {(double)mjGAIN_FIXED}},
  {"biastype", (int)offsetof(mjsActuator, biastype), 2, 1, 1, 0, {(double)mjBIAS_NONE}},
  {"dynprm", (int)offsetof(mjsActuator, dynprm), 0, mjNDYN, 1, 0, {1.0}},
  {"gainprm", (int)offsetof(mjsActuator, gainprm), 0, mjNGAIN, 1, 0, {1.0}},
  {"biasprm", (int)offsetof(mjsActuator, biasprm), 0, mjNBIAS, 0, 0, {0}},
  {"actearly", (int)offsetof(mjsActuator, actearly), 3, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsBody[] = {
  {"pos", (int)offsetof(mjsBody, pos), 0, 3, 3, 0, {0.0, 0.0, 0.0}},
  {"quat", (int)offsetof(mjsBody, quat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"mocap", (int)offsetof(mjsBody, mocap), 3, 1, 0, 0, {0}},
  {"gravcomp", (int)offsetof(mjsBody, gravcomp), 0, 1, 0, 0, {0}},
  {"sleep", (int)offsetof(mjsBody, sleep), 2, 1, 0, 0, {0}},
  {"simple", (int)offsetof(mjsBody, simple), 3, 1, 1, 0, {(double)1}},
  {"fuse", (int)offsetof(mjsBody, fuse), 3, 1, 1, 0, {(double)1}},
  {"pos", (int)offsetof(mjsBody, ipos), 0, 3, 0, 1, {0}},
  {"quat", (int)offsetof(mjsBody, iquat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"mass", (int)offsetof(mjsBody, mass), 0, 1, 0, 0, {0}},
  {"diaginertia", (int)offsetof(mjsBody, inertia), 0, 3, 0, 0, {0}},
  {"fullinertia", (int)offsetof(mjsBody, fullinertia), 0, 6, 0, 1, {0}},
};

static const mjXDefaultEntry kDefaults_mjsCamera[] = {
  {"projection", (int)offsetof(mjsCamera, proj), 2, 1, 0, 0, {0}},
  {"fovy", (int)offsetof(mjsCamera, fovy), 0, 1, 1, 0, {45.0}},
  {"ipd", (int)offsetof(mjsCamera, ipd), 0, 1, 1, 0, {0.068}},
  {"resolution", (int)offsetof(mjsCamera, resolution), 2, 2, 2, 0, {1.0, 1.0}},
  {"pos", (int)offsetof(mjsCamera, pos), 0, 3, 3, 0, {0.0, 0.0, 0.0}},
  {"quat", (int)offsetof(mjsCamera, quat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"mode", (int)offsetof(mjsCamera, mode), 2, 1, 1, 0, {(double)mjCAMLIGHT_FIXED}},
  {"focal", (int)offsetof(mjsCamera, focal_length), 1, 2, 0, 0, {0}},
  {"focalpixel", (int)offsetof(mjsCamera, focal_pixel), 1, 2, 0, 0, {0}},
  {"principal", (int)offsetof(mjsCamera, principal_length), 1, 2, 0, 0, {0}},
  {"principalpixel", (int)offsetof(mjsCamera, principal_pixel), 1, 2, 0, 0, {0}},
  {"sensorsize", (int)offsetof(mjsCamera, sensor_size), 1, 2, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsCompiler[] = {
  {"autolimits", (int)offsetof(mjsCompiler, autolimits), 3, 1, 1, 0, {1}},
  {"boundmass", (int)offsetof(mjsCompiler, boundmass), 0, 1, 0, 0, {0}},
  {"boundinertia", (int)offsetof(mjsCompiler, boundinertia), 0, 1, 0, 0, {0}},
  {"settotalmass", (int)offsetof(mjsCompiler, settotalmass), 0, 1, 1, 0, {-1.0}},
  {"balanceinertia", (int)offsetof(mjsCompiler, balanceinertia), 3, 1, 0, 0, {0}},
  {"angle", (int)offsetof(mjsCompiler, degree), 3, 1, 1, 0, {(double)1}},
  {"fitaabb", (int)offsetof(mjsCompiler, fitaabb), 3, 1, 0, 0, {0}},
  {"discardvisual", (int)offsetof(mjsCompiler, discardvisual), 3, 1, 0, 0, {0}},
  {"usethread", (int)offsetof(mjsCompiler, usethread), 3, 1, 1, 0, {1}},
  {"fusestatic", (int)offsetof(mjsCompiler, fusestatic), 3, 1, 0, 0, {0}},
  {"inertiafromgeom", (int)offsetof(mjsCompiler, inertiafromgeom), 2, 1, 1, 0, {(double)2}},
  {"inertiagrouprange", (int)offsetof(mjsCompiler, inertiagrouprange), 2, 2, 2, 0, {0.0, 5.0}},
  {"saveinertial", (int)offsetof(mjsCompiler, saveinertial), 3, 1, 0, 0, {0}},
  {"savecompiled", (int)offsetof(mjsCompiler, savecompiled), 3, 1, 1, 0, {1}},
  {"savecanonical", (int)offsetof(mjsCompiler, savecanonical), 3, 1, 1, 0, {1}},
  {"alignfree", (int)offsetof(mjsCompiler, alignfree), 3, 1, 0, 0, {0}},
  {"conflict", (int)offsetof(mjsCompiler, conflict), 2, 1, 0, 0, {0}},
  {"passthrough", (int)offsetof(mjsCompiler, passthrough), 2, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsEquality[] = {
  {"active", (int)offsetof(mjsEquality, active), 3, 1, 1, 0, {1}},
  {"solref", (int)offsetof(mjsEquality, solref), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimp", (int)offsetof(mjsEquality, solimp), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
};

static const mjXDefaultEntry kDefaults_mjsFlex[] = {
  {"young", (int)offsetof(mjsFlex, young), 0, 1, 0, 0, {0}},
  {"poisson", (int)offsetof(mjsFlex, poisson), 0, 1, 0, 0, {0}},
  {"shear", (int)offsetof(mjsFlex, shear), 0, 1, 0, 0, {0}},
  {"damping", (int)offsetof(mjsFlex, damping), 0, 1, 0, 0, {0}},
  {"thickness", (int)offsetof(mjsFlex, thickness), 0, 1, 1, 0, {-1.0}},
  {"yield", (int)offsetof(mjsFlex, stressyield), 0, 1, 0, 0, {0}},
  {"elastic2d", (int)offsetof(mjsFlex, elastic2d), 2, 1, 0, 0, {0}},
  {"section", (int)offsetof(mjsFlex, section), 2, 1, 0, 0, {0}},
  {"sectionsize", (int)offsetof(mjsFlex, sectionsize), 0, 2, 0, 0, {0}},
  {"contype", (int)offsetof(mjsFlex, contype), 2, 1, 1, 0, {1.0}},
  {"conaffinity", (int)offsetof(mjsFlex, conaffinity), 2, 1, 1, 0, {1.0}},
  {"condim", (int)offsetof(mjsFlex, condim), 2, 1, 1, 0, {3.0}},
  {"priority", (int)offsetof(mjsFlex, priority), 2, 1, 0, 0, {0}},
  {"friction", (int)offsetof(mjsFlex, friction), 0, 3, 3, 0, {1.0, 0.005, 0.0001}},
  {"solmix", (int)offsetof(mjsFlex, solmix), 0, 1, 1, 0, {1.0}},
  {"solref", (int)offsetof(mjsFlex, solref), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimp", (int)offsetof(mjsFlex, solimp), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"margin", (int)offsetof(mjsFlex, margin), 0, 1, 0, 0, {0}},
  {"gap", (int)offsetof(mjsFlex, gap), 0, 1, 0, 0, {0}},
  {"selfcollide", (int)offsetof(mjsFlex, selfcollide), 2, 1, 1, 0, {(double)mjFLEXSELF_AUTO}},
  {"activelayers", (int)offsetof(mjsFlex, activelayers), 2, 1, 1, 0, {1.0}},
  {"passive", (int)offsetof(mjsFlex, passive), 2, 1, 0, 0, {0}},
  {"group", (int)offsetof(mjsFlex, group), 2, 1, 0, 0, {0}},
  {"dim", (int)offsetof(mjsFlex, dim), 2, 1, 1, 0, {2.0}},
  {"radius", (int)offsetof(mjsFlex, radius), 0, 1, 1, 0, {0.005}},
  {"rgba", (int)offsetof(mjsFlex, rgba), 1, 4, 4, 0, {0.5, 0.5, 0.5, 1.0}},
  {"flatskin", (int)offsetof(mjsFlex, flatskin), 3, 1, 0, 0, {0}},
  {"cellcount", (int)offsetof(mjsFlex, cellcount), 2, 3, 3, 0, {1.0, 1.0, 1.0}},
  {"stiffness", (int)offsetof(mjsFlex, edgestiffness), 0, 1, 0, 0, {0}},
  {"damping", (int)offsetof(mjsFlex, edgedamping), 0, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsFrame[] = {
  {"pos", (int)offsetof(mjsFrame, pos), 0, 3, 0, 0, {0}},
  {"quat", (int)offsetof(mjsFrame, quat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"scale", (int)offsetof(mjsFrame, scale), 0, 3, 3, 0, {1.0, 1.0, 1.0}},
  {"scaling", (int)offsetof(mjsFrame, scaling), 2, 1, 1, 0, {(double)mjSCALING_SIMILAR}},
  {"massscale", (int)offsetof(mjsFrame, massscale), 0, 1, 1, 0, {0.0}},
};

static const mjXDefaultEntry kDefaults_mjsGeom[] = {
  {"type", (int)offsetof(mjsGeom, type), 2, 1, 1, 0, {(double)mjGEOM_SPHERE}},
  {"contype", (int)offsetof(mjsGeom, contype), 2, 1, 1, 0, {1.0}},
  {"conaffinity", (int)offsetof(mjsGeom, conaffinity), 2, 1, 1, 0, {1.0}},
  {"condim", (int)offsetof(mjsGeom, condim), 2, 1, 1, 0, {3.0}},
  {"group", (int)offsetof(mjsGeom, group), 2, 1, 0, 0, {0}},
  {"priority", (int)offsetof(mjsGeom, priority), 2, 1, 0, 0, {0}},
  {"size", (int)offsetof(mjsGeom, size), 0, 3, 0, 0, {0}},
  {"friction", (int)offsetof(mjsGeom, friction), 0, 3, 3, 0, {1.0, 0.005, 0.0001}},
  {"mass", (int)offsetof(mjsGeom, mass), 0, 1, 0, 1, {0}},
  {"density", (int)offsetof(mjsGeom, density), 0, 1, 1, 0, {1000.0}},
  {"shellinertia", (int)offsetof(mjsGeom, typeinertia), 2, 1, 0, 0, {0}},
  {"solmix", (int)offsetof(mjsGeom, solmix), 0, 1, 1, 0, {1.0}},
  {"solref", (int)offsetof(mjsGeom, solref), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimp", (int)offsetof(mjsGeom, solimp), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"margin", (int)offsetof(mjsGeom, margin), 0, 1, 0, 0, {0}},
  {"gap", (int)offsetof(mjsGeom, gap), 0, 1, 0, 0, {0}},
  {"surfacevel", (int)offsetof(mjsGeom, surfacevel), 0, 6, 0, 0, {0}},
  {"adhesion", (int)offsetof(mjsGeom, adhesion), 0, 1, 0, 0, {0}},
  {"fromto", (int)offsetof(mjsGeom, fromto), 0, 6, 0, 1, {0}},
  {"pos", (int)offsetof(mjsGeom, pos), 0, 3, 3, 0, {0.0, 0.0, 0.0}},
  {"quat", (int)offsetof(mjsGeom, quat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"fitscale", (int)offsetof(mjsGeom, fitscale), 0, 1, 1, 0, {1.0}},
  {"rgba", (int)offsetof(mjsGeom, rgba), 1, 4, 4, 0, {0.5, 0.5, 0.5, 1.0}},
  {"castshadow", (int)offsetof(mjsGeom, castshadow), 3, 1, 1, 0, {1}},
  {"fluidshape", (int)offsetof(mjsGeom, fluid_ellipsoid), 4, 1, 0, 0, {0}},
  {"fluidcoef", (int)offsetof(mjsGeom, fluid_coefs), 4, 5, 5, 0, {0.5, 0.25, 1.5, 1.0, 1.0}},
};

static const mjXDefaultEntry kDefaults_mjsHField[] = {
  {"nrow", (int)offsetof(mjsHField, nrow), 2, 1, 0, 0, {0}},
  {"ncol", (int)offsetof(mjsHField, ncol), 2, 1, 0, 0, {0}},
  {"size", (int)offsetof(mjsHField, size), 0, 4, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsJoint[] = {
  {"type", (int)offsetof(mjsJoint, type), 2, 1, 1, 0, {(double)mjJNT_HINGE}},
  {"group", (int)offsetof(mjsJoint, group), 2, 1, 0, 0, {0}},
  {"pos", (int)offsetof(mjsJoint, pos), 0, 3, 0, 0, {0}},
  {"axis", (int)offsetof(mjsJoint, axis), 0, 3, 3, 0, {0.0, 0.0, 1.0}},
  {"springdamper", (int)offsetof(mjsJoint, springdamper), 0, 2, 0, 0, {0}},
  {"limited", (int)offsetof(mjsJoint, limited), 2, 1, 1, 0, {(double)2}},
  {"actuatorfrclimited", (int)offsetof(mjsJoint, actfrclimited), 2, 1, 1, 0, {(double)2}},
  {"solreflimit", (int)offsetof(mjsJoint, solref_limit), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimplimit", (int)offsetof(mjsJoint, solimp_limit), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"solreffriction", (int)offsetof(mjsJoint, solref_friction), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimpfriction", (int)offsetof(mjsJoint, solimp_friction), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"stiffness", (int)offsetof(mjsJoint, stiffness), 0, mjNPOLY+1, 0, 0, {0}},
  {"range", (int)offsetof(mjsJoint, range), 0, 2, 0, 0, {0}},
  {"actuatorfrcrange", (int)offsetof(mjsJoint, actfrcrange), 0, 2, 0, 0, {0}},
  {"actuatorgravcomp", (int)offsetof(mjsJoint, actgravcomp), 3, 1, 0, 0, {0}},
  {"margin", (int)offsetof(mjsJoint, margin), 0, 1, 0, 0, {0}},
  {"ref", (int)offsetof(mjsJoint, ref), 0, 1, 0, 0, {0}},
  {"springref", (int)offsetof(mjsJoint, springref), 0, 1, 0, 0, {0}},
  {"springyield", (int)offsetof(mjsJoint, springyield), 0, 1, 0, 0, {0}},
  {"armature", (int)offsetof(mjsJoint, armature), 0, 1, 0, 0, {0}},
  {"damping", (int)offsetof(mjsJoint, damping), 0, mjNPOLY+1, 0, 0, {0}},
  {"frictionloss", (int)offsetof(mjsJoint, frictionloss), 0, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsKey[] = {
  {"time", (int)offsetof(mjsKey, time), 0, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsLight[] = {
  {"type", (int)offsetof(mjsLight, type), 2, 1, 0, 0, {0}},
  {"castshadow", (int)offsetof(mjsLight, castshadow), 3, 1, 1, 0, {1}},
  {"active", (int)offsetof(mjsLight, active), 3, 1, 1, 0, {1}},
  {"pos", (int)offsetof(mjsLight, pos), 0, 3, 3, 0, {0.0, 0.0, 0.0}},
  {"dir", (int)offsetof(mjsLight, dir), 0, 3, 3, 0, {0.0, 0.0, -1.0}},
  {"bulbradius", (int)offsetof(mjsLight, bulbradius), 1, 1, 1, 0, {0.02}},
  {"intensity", (int)offsetof(mjsLight, intensity), 1, 1, 0, 0, {0}},
  {"range", (int)offsetof(mjsLight, range), 1, 1, 1, 0, {10.0}},
  {"attenuation", (int)offsetof(mjsLight, attenuation), 1, 3, 3, 0, {1.0, 0.0, 0.0}},
  {"cutoff", (int)offsetof(mjsLight, cutoff), 1, 1, 1, 0, {45.0}},
  {"softness", (int)offsetof(mjsLight, softness), 1, 1, 1, 0, {0.2}},
  {"exponent", (int)offsetof(mjsLight, exponent), 1, 1, 1, 0, {10.0}},
  {"ambient", (int)offsetof(mjsLight, ambient), 1, 3, 0, 0, {0}},
  {"diffuse", (int)offsetof(mjsLight, diffuse), 1, 3, 3, 0, {0.7, 0.7, 0.7}},
  {"specular", (int)offsetof(mjsLight, specular), 1, 3, 3, 0, {0.3, 0.3, 0.3}},
  {"mode", (int)offsetof(mjsLight, mode), 2, 1, 1, 0, {(double)mjCAMLIGHT_FIXED}},
};

static const mjXDefaultEntry kDefaults_mjsMaterial[] = {
  {"texrepeat", (int)offsetof(mjsMaterial, texrepeat), 1, 2, 2, 0, {1.0, 1.0}},
  {"texuniform", (int)offsetof(mjsMaterial, texuniform), 3, 1, 0, 0, {0}},
  {"emission", (int)offsetof(mjsMaterial, emission), 1, 1, 0, 0, {0}},
  {"specular", (int)offsetof(mjsMaterial, specular), 1, 1, 1, 0, {0.5}},
  {"shininess", (int)offsetof(mjsMaterial, shininess), 1, 1, 1, 0, {0.5}},
  {"reflectance", (int)offsetof(mjsMaterial, reflectance), 1, 1, 0, 0, {0}},
  {"metallic", (int)offsetof(mjsMaterial, metallic), 1, 1, 1, 0, {-1.0}},
  {"roughness", (int)offsetof(mjsMaterial, roughness), 1, 1, 1, 0, {-1.0}},
  {"rgba", (int)offsetof(mjsMaterial, rgba), 1, 4, 4, 0, {1.0, 1.0, 1.0, 1.0}},
};

static const mjXDefaultEntry kDefaults_mjsMesh[] = {
  {"refpos", (int)offsetof(mjsMesh, refpos), 0, 3, 0, 0, {0}},
  {"refquat", (int)offsetof(mjsMesh, refquat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"scale", (int)offsetof(mjsMesh, scale), 0, 3, 3, 0, {1.0, 1.0, 1.0}},
  {"smoothnormal", (int)offsetof(mjsMesh, smoothnormal), 3, 1, 0, 0, {0}},
  {"maxhullvert", (int)offsetof(mjsMesh, maxhullvert), 2, 1, 1, 0, {-1.0}},
  {"inertia", (int)offsetof(mjsMesh, inertia), 2, 1, 1, 0, {(double)mjMESH_INERTIA_LEGACY}},
};

static const mjXDefaultEntry kDefaults_mjsNumeric[] = {
  {"size", (int)offsetof(mjsNumeric, size), 2, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsPair[] = {
  {"condim", (int)offsetof(mjsPair, condim), 2, 1, 1, 0, {3.0}},
  {"friction", (int)offsetof(mjsPair, friction), 0, 5, 5, 0, {1.0, 1.0, 0.005, 0.0001, 0.0001}},
  {"solref", (int)offsetof(mjsPair, solref), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solreffriction", (int)offsetof(mjsPair, solreffriction), 4, mjNREF, 0, 0, {0}},
  {"solimp", (int)offsetof(mjsPair, solimp), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"gap", (int)offsetof(mjsPair, gap), 0, 1, 0, 0, {0}},
  {"margin", (int)offsetof(mjsPair, margin), 0, 1, 0, 0, {0}},
  {"adhesion", (int)offsetof(mjsPair, adhesion), 0, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsSensor[] = {
  {"nsample", (int)offsetof(mjsSensor, nsample), 2, 1, 0, 0, {0}},
  {"interp", (int)offsetof(mjsSensor, interp), 2, 1, 0, 0, {0}},
  {"delay", (int)offsetof(mjsSensor, delay), 0, 1, 0, 0, {0}},
  {"interval", (int)offsetof(mjsSensor, interval), 0, 2, 0, 0, {0}},
  {"cutoff", (int)offsetof(mjsSensor, cutoff), 0, 1, 0, 0, {0}},
  {"noise", (int)offsetof(mjsSensor, noise), 0, 1, 0, 0, {0}},
  {"objtype", (int)offsetof(mjsSensor, objtype), 2, 1, 0, 0, {0}},
  {"reftype", (int)offsetof(mjsSensor, reftype), 2, 1, 0, 0, {0}},
  {"enclosed", (int)offsetof(mjsSensor, intprm), 2, mjNSENS, 1, 0, {0}},
  {"datatype", (int)offsetof(mjsSensor, datatype), 2, 1, 1, 0, {(double)mjDATATYPE_REAL}},
  {"needstage", (int)offsetof(mjsSensor, needstage), 2, 1, 1, 0, {(double)mjSTAGE_ACC}},
  {"dim", (int)offsetof(mjsSensor, dim), 2, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsSite[] = {
  {"type", (int)offsetof(mjsSite, type), 2, 1, 1, 0, {(double)mjGEOM_SPHERE}},
  {"group", (int)offsetof(mjsSite, group), 2, 1, 0, 0, {0}},
  {"pos", (int)offsetof(mjsSite, pos), 0, 3, 3, 0, {0.0, 0.0, 0.0}},
  {"quat", (int)offsetof(mjsSite, quat), 0, 4, 4, 0, {1.0, 0.0, 0.0, 0.0}},
  {"size", (int)offsetof(mjsSite, size), 0, 3, 3, 0, {0.005, 0.005, 0.005}},
  {"fromto", (int)offsetof(mjsSite, fromto), 0, 6, 0, 1, {0}},
  {"rgba", (int)offsetof(mjsSite, rgba), 1, 4, 4, 0, {0.5, 0.5, 0.5, 1.0}},
};

static const mjXDefaultEntry kDefaults_mjsSkin[] = {
  {"rgba", (int)offsetof(mjsSkin, rgba), 1, 4, 4, 0, {0.5, 0.5, 0.5, 1.0}},
  {"inflate", (int)offsetof(mjsSkin, inflate), 1, 1, 0, 0, {0}},
  {"group", (int)offsetof(mjsSkin, group), 2, 1, 0, 0, {0}},
};

static const mjXDefaultEntry kDefaults_mjsTendon[] = {
  {"group", (int)offsetof(mjsTendon, group), 2, 1, 0, 0, {0}},
  {"limited", (int)offsetof(mjsTendon, limited), 2, 1, 1, 0, {(double)2}},
  {"actuatorfrclimited", (int)offsetof(mjsTendon, actfrclimited), 2, 1, 1, 0, {(double)2}},
  {"range", (int)offsetof(mjsTendon, range), 0, 2, 0, 0, {0}},
  {"actuatorfrcrange", (int)offsetof(mjsTendon, actfrcrange), 0, 2, 0, 0, {0}},
  {"solreflimit", (int)offsetof(mjsTendon, solref_limit), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimplimit", (int)offsetof(mjsTendon, solimp_limit), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"solreffriction", (int)offsetof(mjsTendon, solref_friction), 4, mjNREF, 2, 0, {0.02, 1.0}},
  {"solimpfriction", (int)offsetof(mjsTendon, solimp_friction), 4, mjNIMP, 5, 0, {0.9, 0.95, 0.001, 0.5, 2.0}},
  {"frictionloss", (int)offsetof(mjsTendon, frictionloss), 0, 1, 0, 0, {0}},
  {"springlength", (int)offsetof(mjsTendon, springlength), 0, 2, 2, 0, {-1.0, -1.0}},
  {"springyield", (int)offsetof(mjsTendon, springyield), 0, 1, 0, 0, {0}},
  {"width", (int)offsetof(mjsTendon, width), 0, 1, 1, 0, {0.003}},
  {"margin", (int)offsetof(mjsTendon, margin), 0, 1, 0, 0, {0}},
  {"stiffness", (int)offsetof(mjsTendon, stiffness), 0, mjNPOLY+1, 0, 0, {0}},
  {"damping", (int)offsetof(mjsTendon, damping), 0, mjNPOLY+1, 0, 0, {0}},
  {"armature", (int)offsetof(mjsTendon, armature), 0, 1, 0, 0, {0}},
  {"rgba", (int)offsetof(mjsTendon, rgba), 1, 4, 4, 0, {0.5, 0.5, 0.5, 1.0}},
};

static const mjXDefaultEntry kDefaults_mjsTexture[] = {
  {"type", (int)offsetof(mjsTexture, type), 2, 1, 1, 0, {(double)mjTEXTURE_CUBE}},
  {"colorspace", (int)offsetof(mjsTexture, colorspace), 2, 1, 1, 0, {(double)mjCOLORSPACE_AUTO}},
  {"gridsize", (int)offsetof(mjsTexture, gridsize), 2, 2, 2, 0, {1.0, 1.0}},
  {"builtin", (int)offsetof(mjsTexture, builtin), 2, 1, 0, 0, {0}},
  {"rgb1", (int)offsetof(mjsTexture, rgb1), 0, 3, 3, 0, {0.8, 0.8, 0.8}},
  {"rgb2", (int)offsetof(mjsTexture, rgb2), 0, 3, 3, 0, {0.5, 0.5, 0.5}},
  {"mark", (int)offsetof(mjsTexture, mark), 2, 1, 0, 0, {0}},
  {"markrgb", (int)offsetof(mjsTexture, markrgb), 0, 3, 0, 0, {0}},
  {"random", (int)offsetof(mjsTexture, random), 0, 1, 1, 0, {0.01}},
  {"width", (int)offsetof(mjsTexture, width), 2, 1, 0, 0, {0}},
  {"height", (int)offsetof(mjsTexture, height), 2, 1, 0, 0, {0}},
  {"hflip", (int)offsetof(mjsTexture, hflip), 3, 1, 0, 0, {0}},
  {"vflip", (int)offsetof(mjsTexture, vflip), 3, 1, 0, 0, {0}},
  {"nchannel", (int)offsetof(mjsTexture, nchannel), 2, 1, 1, 0, {3.0}},
  {"embed", (int)offsetof(mjsTexture, embed), 2, 1, 1, 0, {(double)2}},
};

static const mjXDefaultTable kDefaultTables[] = {
  {"mjLROpt", kDefaults_mjLROpt, (int)(sizeof(kDefaults_mjLROpt) / sizeof(kDefaults_mjLROpt[0]))},
  {"mjOption", kDefaults_mjOption, (int)(sizeof(kDefaults_mjOption) / sizeof(kDefaults_mjOption[0]))},
  {"mjSpec", kDefaults_mjSpec, (int)(sizeof(kDefaults_mjSpec) / sizeof(kDefaults_mjSpec[0]))},
  {"mjStatistic", kDefaults_mjStatistic, (int)(sizeof(kDefaults_mjStatistic) / sizeof(kDefaults_mjStatistic[0]))},
  {"mjVisual", kDefaults_mjVisual_global, (int)(sizeof(kDefaults_mjVisual_global) / sizeof(kDefaults_mjVisual_global[0]))},
  {"mjVisual", kDefaults_mjVisual_headlight, (int)(sizeof(kDefaults_mjVisual_headlight) / sizeof(kDefaults_mjVisual_headlight[0]))},
  {"mjVisual", kDefaults_mjVisual_map, (int)(sizeof(kDefaults_mjVisual_map) / sizeof(kDefaults_mjVisual_map[0]))},
  {"mjVisual", kDefaults_mjVisual_quality, (int)(sizeof(kDefaults_mjVisual_quality) / sizeof(kDefaults_mjVisual_quality[0]))},
  {"mjVisual", kDefaults_mjVisual_rgba, (int)(sizeof(kDefaults_mjVisual_rgba) / sizeof(kDefaults_mjVisual_rgba[0]))},
  {"mjVisual", kDefaults_mjVisual_scale, (int)(sizeof(kDefaults_mjVisual_scale) / sizeof(kDefaults_mjVisual_scale[0]))},
  {"mjsActuator", kDefaults_mjsActuator, (int)(sizeof(kDefaults_mjsActuator) / sizeof(kDefaults_mjsActuator[0]))},
  {"mjsBody", kDefaults_mjsBody, (int)(sizeof(kDefaults_mjsBody) / sizeof(kDefaults_mjsBody[0]))},
  {"mjsCamera", kDefaults_mjsCamera, (int)(sizeof(kDefaults_mjsCamera) / sizeof(kDefaults_mjsCamera[0]))},
  {"mjsCompiler", kDefaults_mjsCompiler, (int)(sizeof(kDefaults_mjsCompiler) / sizeof(kDefaults_mjsCompiler[0]))},
  {"mjsEquality", kDefaults_mjsEquality, (int)(sizeof(kDefaults_mjsEquality) / sizeof(kDefaults_mjsEquality[0]))},
  {"mjsFlex", kDefaults_mjsFlex, (int)(sizeof(kDefaults_mjsFlex) / sizeof(kDefaults_mjsFlex[0]))},
  {"mjsFrame", kDefaults_mjsFrame, (int)(sizeof(kDefaults_mjsFrame) / sizeof(kDefaults_mjsFrame[0]))},
  {"mjsGeom", kDefaults_mjsGeom, (int)(sizeof(kDefaults_mjsGeom) / sizeof(kDefaults_mjsGeom[0]))},
  {"mjsHField", kDefaults_mjsHField, (int)(sizeof(kDefaults_mjsHField) / sizeof(kDefaults_mjsHField[0]))},
  {"mjsJoint", kDefaults_mjsJoint, (int)(sizeof(kDefaults_mjsJoint) / sizeof(kDefaults_mjsJoint[0]))},
  {"mjsKey", kDefaults_mjsKey, (int)(sizeof(kDefaults_mjsKey) / sizeof(kDefaults_mjsKey[0]))},
  {"mjsLight", kDefaults_mjsLight, (int)(sizeof(kDefaults_mjsLight) / sizeof(kDefaults_mjsLight[0]))},
  {"mjsMaterial", kDefaults_mjsMaterial, (int)(sizeof(kDefaults_mjsMaterial) / sizeof(kDefaults_mjsMaterial[0]))},
  {"mjsMesh", kDefaults_mjsMesh, (int)(sizeof(kDefaults_mjsMesh) / sizeof(kDefaults_mjsMesh[0]))},
  {"mjsNumeric", kDefaults_mjsNumeric, (int)(sizeof(kDefaults_mjsNumeric) / sizeof(kDefaults_mjsNumeric[0]))},
  {"mjsPair", kDefaults_mjsPair, (int)(sizeof(kDefaults_mjsPair) / sizeof(kDefaults_mjsPair[0]))},
  {"mjsSensor", kDefaults_mjsSensor, (int)(sizeof(kDefaults_mjsSensor) / sizeof(kDefaults_mjsSensor[0]))},
  {"mjsSite", kDefaults_mjsSite, (int)(sizeof(kDefaults_mjsSite) / sizeof(kDefaults_mjsSite[0]))},
  {"mjsSkin", kDefaults_mjsSkin, (int)(sizeof(kDefaults_mjsSkin) / sizeof(kDefaults_mjsSkin[0]))},
  {"mjsTendon", kDefaults_mjsTendon, (int)(sizeof(kDefaults_mjsTendon) / sizeof(kDefaults_mjsTendon[0]))},
  {"mjsTexture", kDefaults_mjsTexture, (int)(sizeof(kDefaults_mjsTexture) / sizeof(kDefaults_mjsTexture[0]))},
};
static const int kDefaultTablesN = (int)(sizeof(kDefaultTables) / sizeof(kDefaultTables[0]));
// clang-format on
