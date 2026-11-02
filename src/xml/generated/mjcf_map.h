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
// doc/generate/generate_mjcf_map.py; test/doc/doc_test.py checks freshness.
//
// Keyword maps, one per schema enum, shared by the reader, the writer and
// the generated tables. Inline variables: including this header is all a
// translation unit needs.

#ifndef MUJOCO_SRC_XML_GENERATED_MJCF_MAP_H_
#define MUJOCO_SRC_XML_GENERATED_MJCF_MAP_H_

#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include "user/user_flexcomp.h"
#include "xml/xml_util.h"

// clang-format off

// keywords of the built-in bool type (not a schema enum)
inline constexpr mjMap bool_map[] = {
  {"false",   0},
  {"true",    1},
};

// enum coordinate
inline constexpr mjMap coordinate_map[] = {
  {"local",   0},
  {"global",  1},
};
inline constexpr int coordinate_sz = 2;

// enum angle
inline constexpr mjMap angle_map[] = {
  {"radian",  0},
  {"degree",  1},
};
inline constexpr int angle_sz = 2;

// enum fluidshape
inline constexpr mjMap fluidshape_map[] = {
  {"none",       0},
  {"ellipsoid",  1},
};
inline constexpr int fluidshape_sz = 2;

// enum enable
inline constexpr mjMap enable_map[] = {
  {"disable",  0},
  {"enable",   1},
};
inline constexpr int enable_sz = 2;

// enum FalseTrueAuto
inline constexpr mjMap FalseTrueAuto_map[] = {
  {"false",  0},
  {"true",   1},
  {"auto",   2},
};
inline constexpr int FalseTrueAuto_sz = 3;

// enum FalseAuto
inline constexpr mjMap FalseAuto_map[] = {
  {"false",  0},
  {"auto",   1},
};
inline constexpr int FalseAuto_sz = 2;

// enum bodysleep
inline constexpr mjMap bodysleep_map[] = {
  {"auto",     mjSLEEP_AUTO},
  {"never",    mjSLEEP_NEVER},
  {"allowed",  mjSLEEP_ALLOWED},
  {"init",     mjSLEEP_INIT},
};
inline constexpr int bodysleep_sz = 4;

// enum jointtype
inline constexpr mjMap jointtype_map[] = {
  {"free",   mjJNT_FREE},
  {"ball",   mjJNT_BALL},
  {"slide",  mjJNT_SLIDE},
  {"hinge",  mjJNT_HINGE},
};
inline constexpr int jointtype_sz = 4;

// enum geomtype
inline constexpr mjMap geomtype_map[] = {
  {"plane",      mjGEOM_PLANE},
  {"hfield",     mjGEOM_HFIELD},
  {"sphere",     mjGEOM_SPHERE},
  {"capsule",    mjGEOM_CAPSULE},
  {"ellipsoid",  mjGEOM_ELLIPSOID},
  {"cylinder",   mjGEOM_CYLINDER},
  {"box",        mjGEOM_BOX},
  {"mesh",       mjGEOM_MESH},
  {"sdf",        mjGEOM_SDF},
};
inline constexpr int geomtype_sz = 9;

// enum projection
inline constexpr mjMap projection_map[] = {
  {"perspective",   mjPROJ_PERSPECTIVE},
  {"orthographic",  mjPROJ_ORTHOGRAPHIC},
};
inline constexpr int projection_sz = 2;

// enum camlight
inline constexpr mjMap camlight_map[] = {
  {"fixed",          mjCAMLIGHT_FIXED},
  {"track",          mjCAMLIGHT_TRACK},
  {"trackcom",       mjCAMLIGHT_TRACKCOM},
  {"targetbody",     mjCAMLIGHT_TARGETBODY},
  {"targetbodycom",  mjCAMLIGHT_TARGETBODYCOM},
};
inline constexpr int camlight_sz = 5;

// enum lighttype
inline constexpr mjMap lighttype_map[] = {
  {"spot",         mjLIGHT_SPOT},
  {"directional",  mjLIGHT_DIRECTIONAL},
  {"point",        mjLIGHT_POINT},
  {"image",        mjLIGHT_IMAGE},
};
inline constexpr int lighttype_sz = 4;

// enum texrole
inline constexpr mjMap texrole_map[] = {
  {"rgb",        mjTEXROLE_RGB},
  {"occlusion",  mjTEXROLE_OCCLUSION},
  {"roughness",  mjTEXROLE_ROUGHNESS},
  {"metallic",   mjTEXROLE_METALLIC},
  {"normal",     mjTEXROLE_NORMAL},
  {"opacity",    mjTEXROLE_OPACITY},
  {"emissive",   mjTEXROLE_EMISSIVE},
  {"rgba",       mjTEXROLE_RGBA},
  {"orm",        mjTEXROLE_ORM},
};
inline constexpr int texrole_sz = 9;

// enum integrator
inline constexpr mjMap integrator_map[] = {
  {"Euler",         mjINT_EULER},
  {"RK4",           mjINT_RK4},
  {"implicit",      mjINT_IMPLICIT},
  {"implicitfast",  mjINT_IMPLICITFAST},
  {"discrete",      mjINT_DISCRETE},
};
inline constexpr int integrator_sz = 5;

// enum cone
inline constexpr mjMap cone_map[] = {
  {"pyramidal",  mjCONE_PYRAMIDAL},
  {"elliptic",   mjCONE_ELLIPTIC},
};
inline constexpr int cone_sz = 2;

// enum jacobian
inline constexpr mjMap jacobian_map[] = {
  {"dense",   mjJAC_DENSE},
  {"sparse",  mjJAC_SPARSE},
  {"auto",    mjJAC_AUTO},
};
inline constexpr int jacobian_sz = 3;

// enum solver
inline constexpr mjMap solver_map[] = {
  {"PGS",     mjSOL_PGS},
  {"CG",      mjSOL_CG},
  {"Newton",  mjSOL_NEWTON},
};
inline constexpr int solver_sz = 3;

// enum equality
inline constexpr mjMap equality_map[] = {
  {"connect",     mjEQ_CONNECT},
  {"weld",        mjEQ_WELD},
  {"joint",       mjEQ_JOINT},
  {"tendon",      mjEQ_TENDON},
  {"flex",        mjEQ_FLEX},
  {"flexvert",    mjEQ_FLEXVERT},
  {"flexstrain",  mjEQ_FLEXSTRAIN},
  {"distance",    mjEQ_DISTANCE},
};
inline constexpr int equality_sz = 8;

// enum texture
inline constexpr mjMap texture_map[] = {
  {"2d",      mjTEXTURE_2D},
  {"cube",    mjTEXTURE_CUBE},
  {"skybox",  mjTEXTURE_SKYBOX},
};
inline constexpr int texture_sz = 3;

// enum colorspace
inline constexpr mjMap colorspace_map[] = {
  {"auto",    mjCOLORSPACE_AUTO},
  {"linear",  mjCOLORSPACE_LINEAR},
  {"sRGB",    mjCOLORSPACE_SRGB},
};
inline constexpr int colorspace_sz = 3;

// enum builtin
inline constexpr mjMap builtin_map[] = {
  {"none",      mjBUILTIN_NONE},
  {"gradient",  mjBUILTIN_GRADIENT},
  {"checker",   mjBUILTIN_CHECKER},
  {"flat",      mjBUILTIN_FLAT},
};
inline constexpr int builtin_sz = 4;

// enum mark
inline constexpr mjMap mark_map[] = {
  {"none",    mjMARK_NONE},
  {"edge",    mjMARK_EDGE},
  {"cross",   mjMARK_CROSS},
  {"random",  mjMARK_RANDOM},
};
inline constexpr int mark_sz = 4;

// enum actuatortype
inline constexpr mjMap actuatortype_map[] = {
  {"general",      mjACTUATOR_GENERAL},
  {"motor",        mjACTUATOR_MOTOR},
  {"position",     mjACTUATOR_POSITION},
  {"velocity",     mjACTUATOR_VELOCITY},
  {"intvelocity",  mjACTUATOR_INTVELOCITY},
  {"damper",       mjACTUATOR_DAMPER},
  {"cylinder",     mjACTUATOR_CYLINDER},
  {"muscle",       mjACTUATOR_MUSCLE},
  {"adhesion",     mjACTUATOR_ADHESION},
  {"pid",          mjACTUATOR_PID},
  {"orientation",  mjACTUATOR_ORIENTATION},
  {"dcmotor",      mjACTUATOR_DCMOTOR},
};
inline constexpr int actuatortype_sz = 12;

// enum dyn
inline constexpr mjMap dyn_map[] = {
  {"none",         mjDYN_NONE},
  {"integrator",   mjDYN_INTEGRATOR},
  {"filter",       mjDYN_FILTER},
  {"filterexact",  mjDYN_FILTEREXACT},
  {"muscle",       mjDYN_MUSCLE},
  {"dcmotor",      mjDYN_DCMOTOR},
  {"pid",          mjDYN_PID},
  {"user",         mjDYN_USER},
};
inline constexpr int dyn_sz = 8;

// enum inputkeyword
inline constexpr mjMap inputkeyword_map[] = {
  {"none",  mjINPUT_NONE},
};
inline constexpr int inputkeyword_sz = 1;

// enum gain
inline constexpr mjMap gain_map[] = {
  {"fixed",    mjGAIN_FIXED},
  {"affine",   mjGAIN_AFFINE},
  {"muscle",   mjGAIN_MUSCLE},
  {"dcmotor",  mjGAIN_DCMOTOR},
  {"so3",      mjGAIN_SO3},
  {"pid",      mjGAIN_PID},
  {"user",     mjGAIN_USER},
};
inline constexpr int gain_sz = 7;

// enum inputchart
inline constexpr mjMap inputchart_map[] = {
  {"expmap",  mjCHART_EXPMAP},
  {"quat",    mjCHART_QUAT},
};
inline constexpr int inputchart_sz = 2;

// enum inputbit
inline constexpr mjMap inputbit_map[] = {
  {"pos",       mjINPUT_POS},
  {"vel",       mjINPUT_VEL},
  {"ff",        mjINPUT_FF},
  {"voltage",   mjINPUT_VOLTAGE},
  {"pressure",  mjINPUT_PRESSURE},
};
inline constexpr int inputbit_sz = 5;

// enum bias
inline constexpr mjMap bias_map[] = {
  {"none",     mjBIAS_NONE},
  {"affine",   mjBIAS_AFFINE},
  {"muscle",   mjBIAS_MUSCLE},
  {"dcmotor",  mjBIAS_DCMOTOR},
  {"so3",      mjBIAS_SO3},
  {"user",     mjBIAS_USER},
};
inline constexpr int bias_sz = 6;

// enum interp
inline constexpr mjMap interp_map[] = {
  {"zoh",     0},
  {"linear",  1},
  {"cubic",   2},
};
inline constexpr int interp_sz = 3;

// enum stage
inline constexpr mjMap stage_map[] = {
  {"none",  mjSTAGE_NONE},
  {"pos",   mjSTAGE_POS},
  {"vel",   mjSTAGE_VEL},
  {"acc",   mjSTAGE_ACC},
};
inline constexpr int stage_sz = 4;

// enum datatype
inline constexpr mjMap datatype_map[] = {
  {"real",        mjDATATYPE_REAL},
  {"positive",    mjDATATYPE_POSITIVE},
  {"axis",        mjDATATYPE_AXIS},
  {"quaternion",  mjDATATYPE_QUATERNION},
};
inline constexpr int datatype_sz = 4;

// enum frameobj
inline constexpr mjMap frameobj_map[] = {
  {"body",    mjOBJ_BODY},
  {"xbody",   mjOBJ_XBODY},
  {"geom",    mjOBJ_GEOM},
  {"site",    mjOBJ_SITE},
  {"camera",  mjOBJ_CAMERA},
};
inline constexpr int frameobj_sz = 5;

// enum condata
inline constexpr mjMap condata_map[] = {
  {"found",    mjCONDATA_FOUND},
  {"force",    mjCONDATA_FORCE},
  {"torque",   mjCONDATA_TORQUE},
  {"dist",     mjCONDATA_DIST},
  {"pos",      mjCONDATA_POS},
  {"normal",   mjCONDATA_NORMAL},
  {"tangent",  mjCONDATA_TANGENT},
  {"linvel",   mjCONDATA_LINVEL},
  {"angvel",   mjCONDATA_ANGVEL},
};
inline constexpr int condata_sz = 9;

// enum raydata
inline constexpr mjMap raydata_map[] = {
  {"dist",    mjRAYDATA_DIST},
  {"dir",     mjRAYDATA_DIR},
  {"origin",  mjRAYDATA_ORIGIN},
  {"point",   mjRAYDATA_POINT},
  {"normal",  mjRAYDATA_NORMAL},
  {"depth",   mjRAYDATA_DEPTH},
};
inline constexpr int raydata_sz = 6;

// enum camout
inline constexpr mjMap camout_map[] = {
  {"rgb",           mjCAMOUT_RGB},
  {"depth",         mjCAMOUT_DEPTH},
  {"distance",      mjCAMOUT_DIST},
  {"normal",        mjCAMOUT_NORMAL},
  {"segmentation",  mjCAMOUT_SEG},
};
inline constexpr int camout_sz = 5;

// enum reduce
inline constexpr mjMap reduce_map[] = {
  {"none",      0},
  {"mindist",   1},
  {"maxforce",  2},
  {"netforce",  3},
};
inline constexpr int reduce_sz = 4;

// enum conflict
inline constexpr mjMap conflict_map[] = {
  {"warning",  mjCONFLICT_WARNING},
  {"merge",    mjCONFLICT_MERGE},
  {"error",    mjCONFLICT_ERROR},
};
inline constexpr int conflict_sz = 3;

// enum passthrough
inline constexpr mjMap passthrough_map[] = {
  {"none",      mjPASSTHROUGH_NONE},
  {"textures",  mjPASSTHROUGH_TEXTURES},
};
inline constexpr int passthrough_sz = 2;

// enum scaling
inline constexpr mjMap scaling_map[] = {
  {"similar",   mjSCALING_SIMILAR},
  {"geometry",  mjSCALING_GEOMETRY},
};
inline constexpr int scaling_sz = 2;

// enum lrmode
inline constexpr mjMap lrmode_map[] = {
  {"none",        mjLRMODE_NONE},
  {"muscle",      mjLRMODE_MUSCLE},
  {"muscleuser",  mjLRMODE_MUSCLEUSER},
  {"all",         mjLRMODE_ALL},
};
inline constexpr int lrmode_sz = 4;

// enum meshinertia
inline constexpr mjMap meshinertia_map[] = {
  {"convex",  mjMESH_INERTIA_CONVEX},
  {"legacy",  mjMESH_INERTIA_LEGACY},
  {"exact",   mjMESH_INERTIA_EXACT},
  {"shell",   mjMESH_INERTIA_SHELL},
};
inline constexpr int meshinertia_sz = 4;

// enum meshbuiltin
inline constexpr mjMap meshbuiltin_map[] = {
  {"none",         mjMESH_BUILTIN_NONE},
  {"sphere",       mjMESH_BUILTIN_SPHERE},
  {"hemisphere",   mjMESH_BUILTIN_HEMISPHERE},
  {"cone",         mjMESH_BUILTIN_CONE},
  {"supertorus",   mjMESH_BUILTIN_SUPERTORUS},
  {"supersphere",  mjMESH_BUILTIN_SUPERSPHERE},
  {"wedge",        mjMESH_BUILTIN_WEDGE},
  {"plate",        mjMESH_BUILTIN_PLATE},
  {"grid",         mjMESH_BUILTIN_GRID},
  {"box",          mjMESH_BUILTIN_BOX},
  {"cylinder",     mjMESH_BUILTIN_CYLINDER},
  {"ellipsoid",    mjMESH_BUILTIN_ELLIPSOID},
  {"square",       mjMESH_BUILTIN_SQUARE},
  {"disc",         mjMESH_BUILTIN_DISC},
};
inline constexpr int meshbuiltin_sz = 14;

// enum fcomp
inline constexpr mjMap fcomp_map[] = {
  {"grid",       mjFCOMPTYPE_GRID},
  {"box",        mjFCOMPTYPE_BOX},
  {"cylinder",   mjFCOMPTYPE_CYLINDER},
  {"ellipsoid",  mjFCOMPTYPE_ELLIPSOID},
  {"square",     mjFCOMPTYPE_SQUARE},
  {"disc",       mjFCOMPTYPE_DISC},
  {"circle",     mjFCOMPTYPE_CIRCLE},
  {"mesh",       mjFCOMPTYPE_MESH},
  {"gmsh",       mjFCOMPTYPE_GMSH},
  {"direct",     mjFCOMPTYPE_DIRECT},
};
inline constexpr int fcomp_sz = 10;

// enum fdof
inline constexpr mjMap fdof_map[] = {
  {"full",       mjFCOMPDOF_FULL},
  {"radial",     mjFCOMPDOF_RADIAL},
  {"trilinear",  mjFCOMPDOF_TRILINEAR},
  {"quadratic",  mjFCOMPDOF_QUADRATIC},
  {"2d",         mjFCOMPDOF_2D},
};
inline constexpr int fdof_sz = 5;

// enum flexself
inline constexpr mjMap flexself_map[] = {
  {"none",    mjFLEXSELF_NONE},
  {"narrow",  mjFLEXSELF_NARROW},
  {"bvh",     mjFLEXSELF_BVH},
  {"sap",     mjFLEXSELF_SAP},
  {"auto",    mjFLEXSELF_AUTO},
};
inline constexpr int flexself_sz = 5;

// enum elastic2d
inline constexpr mjMap elastic2d_map[] = {
  {"none",     0},
  {"bend",     1},
  {"stretch",  2},
  {"both",     3},
};
inline constexpr int elastic2d_sz = 4;

// enum flexsection
inline constexpr mjMap flexsection_map[] = {
  {"circle",  mjFLEXSECTION_CIRCLE},
  {"box",     mjFLEXSECTION_BOX},
};
inline constexpr int flexsection_sz = 2;

// enum flexeq
inline constexpr mjMap flexeq_map[] = {
  {"false",   0},
  {"true",    1},
  {"strain",  3},
};
inline constexpr int flexeq_sz = 3;
// clang-format on

#endif  // MUJOCO_SRC_XML_GENERATED_MJCF_MAP_H_
