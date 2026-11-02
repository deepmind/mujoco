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
// doc/generate/generate_read_table.py; test/doc/doc_test.py checks freshness.
//
// Typed attribute rows for table-driven reading (mjXReader::ReadAttrTable),
// one array per migrated element. Rows are {attr, kind, len, exact,
// required, nodefault, handwrite, offset}; keyword maps are defined in
// mjcf_map.h.

// clang-format off

// compiler (mjsCompiler)
inline constexpr mjXAttr kCompilerAttrs[] = {
  {"autolimits", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, autolimits)},
  {"boundmass", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsCompiler, boundmass)},
  {"boundinertia", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsCompiler, boundinertia)},
  {"settotalmass", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsCompiler, settotalmass)},
  {"balanceinertia", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, balanceinertia)},
  {"angle", mjXAttr::kEnumByte, 1, true, false, false, false, (int)offsetof(mjsCompiler, degree), angle_map, angle_sz},
  {"fitaabb", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, fitaabb)},
  {"eulerseq", mjXAttr::kChars, 3, true, false, false, false, (int)offsetof(mjsCompiler, eulerseq)},
  {"meshdir", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsCompiler, meshdir)},
  {"texturedir", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsCompiler, texturedir)},
  {"discardvisual", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, discardvisual)},
  {"usethread", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, usethread)},
  {"fusestatic", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, fusestatic)},
  {"inertiafromgeom", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsCompiler, inertiafromgeom), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"inertiagrouprange", mjXAttr::kInt, 2, true, false, false, false, (int)offsetof(mjsCompiler, inertiagrouprange)},
  {"saveinertial", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, saveinertial)},
  {"savecompiled", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, savecompiled)},
  {"savecanonical", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, savecanonical)},
  {"alignfree", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsCompiler, alignfree)},
  {"conflict", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsCompiler, conflict), conflict_map, conflict_sz},
  {"passthrough", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsCompiler, passthrough), passthrough_map, passthrough_sz},
};
inline constexpr int kCompilerAttrsN = sizeof(kCompilerAttrs) / sizeof(kCompilerAttrs[0]);

// lengthrange (mjLROpt)
inline constexpr mjXAttr kLengthrangeAttrs[] = {
  {"mode", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjLROpt, mode), lrmode_map, lrmode_sz},
  {"useexisting", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjLROpt, useexisting), bool_map, 2},
  {"uselimit", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjLROpt, uselimit), bool_map, 2},
  {"accel", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, accel)},
  {"maxforce", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, maxforce)},
  {"timeconst", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, timeconst)},
  {"timestep", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, timestep)},
  {"inttotal", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, inttotal)},
  {"interval", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, interval)},
  {"tolrange", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjLROpt, tolrange)},
};
inline constexpr int kLengthrangeAttrsN = sizeof(kLengthrangeAttrs) / sizeof(kLengthrangeAttrs[0]);

// option (mjOption)
inline constexpr mjXAttr kOptionAttrs[] = {
  {"timestep", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, timestep)},
  {"impratio", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, impratio)},
  {"tolerance", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, tolerance)},
  {"ls_tolerance", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, ls_tolerance)},
  {"noslip_tolerance", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, noslip_tolerance)},
  {"ccd_tolerance", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, ccd_tolerance)},
  {"sleep_tolerance", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, sleep_tolerance)},
  {"gravity", mjXAttr::kNum, 3, true, false, false, false, (int)offsetof(mjOption, gravity)},
  {"wind", mjXAttr::kNum, 3, true, false, false, false, (int)offsetof(mjOption, wind)},
  {"magnetic", mjXAttr::kNum, 3, true, false, false, false, (int)offsetof(mjOption, magnetic)},
  {"density", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, density)},
  {"viscosity", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, viscosity)},
  {"o_margin", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjOption, o_margin)},
  {"o_solref", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjOption, o_solref)},
  {"o_solimp", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjOption, o_solimp)},
  {"o_friction", mjXAttr::kNum, 5, false, false, false, false, (int)offsetof(mjOption, o_friction)},
  {"integrator", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjOption, integrator), integrator_map, integrator_sz},
  {"cone", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjOption, cone), cone_map, cone_sz},
  {"jacobian", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjOption, jacobian), jacobian_map, jacobian_sz},
  {"solver", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjOption, solver), solver_map, solver_sz},
  {"iterations", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjOption, iterations)},
  {"ls_iterations", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjOption, ls_iterations)},
  {"noslip_iterations", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjOption, noslip_iterations)},
  {"ccd_iterations", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjOption, ccd_iterations)},
  {"sdf_iterations", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjOption, sdf_iterations)},
  {"sdf_initpoints", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjOption, sdf_initpoints)},
};
inline constexpr int kOptionAttrsN = sizeof(kOptionAttrs) / sizeof(kOptionAttrs[0]);

// size (mjSpec)
inline constexpr mjXAttr kSizeAttrs[] = {
  {"nuserdata", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuserdata)},
  {"nkey", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nkey)},
  {"nuser_body", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_body)},
  {"nuser_jnt", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_jnt)},
  {"nuser_geom", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_geom)},
  {"nuser_site", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_site)},
  {"nuser_cam", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_cam)},
  {"nuser_tendon", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_tendon)},
  {"nuser_actuator", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_actuator)},
  {"nuser_sensor", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjSpec, nuser_sensor)},
};
inline constexpr int kSizeAttrsN = sizeof(kSizeAttrs) / sizeof(kSizeAttrs[0]);

// statistic (mjStatistic)
inline constexpr mjXAttr kStatisticAttrs[] = {
  {"meaninertia", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjStatistic, meaninertia)},
  {"meanmass", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjStatistic, meanmass)},
  {"meansize", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjStatistic, meansize)},
  {"extent", mjXAttr::kNum, 1, true, false, false, false, (int)offsetof(mjStatistic, extent)},
  {"center", mjXAttr::kNum, 3, true, false, false, false, (int)offsetof(mjStatistic, center)},
};
inline constexpr int kStatisticAttrsN = sizeof(kStatisticAttrs) / sizeof(kStatisticAttrs[0]);

// global (mjVisual)
inline constexpr mjXAttr kGlobalAttrs[] = {
  {"cameraid", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, global.cameraid)},
  {"orthographic", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjVisual, global.orthographic), bool_map, 2},
  {"fovy", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.fovy)},
  {"ipd", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.ipd)},
  {"azimuth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.azimuth)},
  {"elevation", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.elevation)},
  {"linewidth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.linewidth)},
  {"glow", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.glow)},
  {"offwidth", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, global.offwidth)},
  {"offheight", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, global.offheight)},
  {"realtime", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, global.realtime)},
  {"ellipsoidinertia", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjVisual, global.ellipsoidinertia), bool_map, 2},
  {"bvactive", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjVisual, global.bvactive), bool_map, 2},
};
inline constexpr int kGlobalAttrsN = sizeof(kGlobalAttrs) / sizeof(kGlobalAttrs[0]);

// quality (mjVisual)
inline constexpr mjXAttr kQualityAttrs[] = {
  {"shadowsize", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, quality.shadowsize)},
  {"offsamples", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, quality.offsamples)},
  {"numslices", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, quality.numslices)},
  {"numstacks", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, quality.numstacks)},
  {"numquads", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, quality.numquads)},
};
inline constexpr int kQualityAttrsN = sizeof(kQualityAttrs) / sizeof(kQualityAttrs[0]);

// headlight (mjVisual)
inline constexpr mjXAttr kHeadlightAttrs[] = {
  {"ambient", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjVisual, headlight.ambient)},
  {"diffuse", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjVisual, headlight.diffuse)},
  {"specular", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjVisual, headlight.specular)},
  {"active", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjVisual, headlight.active)},
};
inline constexpr int kHeadlightAttrsN = sizeof(kHeadlightAttrs) / sizeof(kHeadlightAttrs[0]);

// map (mjVisual)
inline constexpr mjXAttr kMapAttrs[] = {
  {"stiffness", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.stiffness)},
  {"stiffnessrot", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.stiffnessrot)},
  {"force", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.force)},
  {"torque", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.torque)},
  {"alpha", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.alpha)},
  {"fogstart", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.fogstart)},
  {"fogend", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.fogend)},
  {"znear", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.znear)},
  {"zfar", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.zfar)},
  {"haze", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.haze)},
  {"shadowclip", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.shadowclip)},
  {"shadowscale", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.shadowscale)},
  {"actuatortendon", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, map.actuatortendon)},
};
inline constexpr int kMapAttrsN = sizeof(kMapAttrs) / sizeof(kMapAttrs[0]);

// scale (mjVisual)
inline constexpr mjXAttr kScaleAttrs[] = {
  {"forcewidth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.forcewidth)},
  {"contactwidth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.contactwidth)},
  {"contactheight", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.contactheight)},
  {"connect", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.connect)},
  {"com", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.com)},
  {"camera", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.camera)},
  {"light", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.light)},
  {"selectpoint", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.selectpoint)},
  {"jointlength", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.jointlength)},
  {"jointwidth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.jointwidth)},
  {"actuatorlength", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.actuatorlength)},
  {"actuatorwidth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.actuatorwidth)},
  {"framelength", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.framelength)},
  {"framewidth", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.framewidth)},
  {"constraint", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.constraint)},
  {"slidercrank", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.slidercrank)},
  {"frustum", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjVisual, scale.frustum)},
};
inline constexpr int kScaleAttrsN = sizeof(kScaleAttrs) / sizeof(kScaleAttrs[0]);

// rgba (mjVisual)
inline constexpr mjXAttr kRgbaAttrs[] = {
  {"fog", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.fog)},
  {"haze", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.haze)},
  {"force", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.force)},
  {"inertia", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.inertia)},
  {"joint", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.joint)},
  {"actuator", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.actuator)},
  {"actuatornegative", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.actuatornegative)},
  {"actuatorpositive", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.actuatorpositive)},
  {"com", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.com)},
  {"camera", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.camera)},
  {"light", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.light)},
  {"selectpoint", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.selectpoint)},
  {"connect", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.connect)},
  {"contactpoint", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.contactpoint)},
  {"contactforce", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.contactforce)},
  {"contactfriction", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.contactfriction)},
  {"contacttorque", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.contacttorque)},
  {"contactgap", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.contactgap)},
  {"rangefinder", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.rangefinder)},
  {"constraint", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.constraint)},
  {"slidercrank", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.slidercrank)},
  {"crankbroken", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.crankbroken)},
  {"frustum", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.frustum)},
  {"bv", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.bv)},
  {"bvactive", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjVisual, rgba.bvactive)},
};
inline constexpr int kRgbaAttrsN = sizeof(kRgbaAttrs) / sizeof(kRgbaAttrs[0]);

// mesh (mjsMesh)
inline constexpr mjXAttr kMeshAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"content_type", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsMesh, content_type)},
  {"vertex", mjXAttr::kFloatVec, 1, true, false, true, false, (int)offsetof(mjsMesh, uservert)},
  {"normal", mjXAttr::kFloatVec, 1, true, false, true, false, (int)offsetof(mjsMesh, usernormal)},
  {"texcoord", mjXAttr::kFloatVec, 1, true, false, true, false, (int)offsetof(mjsMesh, usertexcoord)},
  {"face", mjXAttr::kIntVec, 1, true, false, true, false, (int)offsetof(mjsMesh, userface)},
  {"refpos", mjXAttr::kDouble, 3, true, false, true, false, (int)offsetof(mjsMesh, refpos)},
  {"refquat", mjXAttr::kDouble, 4, true, false, true, false, (int)offsetof(mjsMesh, refquat)},
  {"scale", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsMesh, scale)},
  {"smoothnormal", mjXAttr::kBool, 1, true, false, true, false, (int)offsetof(mjsMesh, smoothnormal)},
  {"inertia", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsMesh, inertia), meshinertia_map, meshinertia_sz},
  {"material", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsMesh, material)},
};
inline constexpr int kMeshAttrsN = sizeof(kMeshAttrs) / sizeof(kMeshAttrs[0]);

// hfield (mjsHField)
inline constexpr mjXAttr kHfieldAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"content_type", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsHField, content_type)},
  {"nrow", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsHField, nrow)},
  {"ncol", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsHField, ncol)},
  {"size", mjXAttr::kDouble, 4, true, true, false, false, (int)offsetof(mjsHField, size)},
};
inline constexpr int kHfieldAttrsN = sizeof(kHfieldAttrs) / sizeof(kHfieldAttrs[0]);

// skin (mjsSkin)
inline constexpr mjXAttr kSkinAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"material", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSkin, material)},
  {"rgba", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjsSkin, rgba)},
  {"inflate", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsSkin, inflate)},
  {"vertex", mjXAttr::kFloatVec, 1, true, false, false, false, (int)offsetof(mjsSkin, vert)},
  {"texcoord", mjXAttr::kFloatVec, 1, true, false, false, false, (int)offsetof(mjsSkin, texcoord)},
  {"face", mjXAttr::kIntVec, 1, true, false, false, false, (int)offsetof(mjsSkin, face)},
};
inline constexpr int kSkinAttrsN = sizeof(kSkinAttrs) / sizeof(kSkinAttrs[0]);

// texture (mjsTexture)
inline constexpr mjXAttr kTextureAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"type", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTexture, type), texture_map, texture_sz},
  {"colorspace", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTexture, colorspace), colorspace_map, colorspace_sz},
  {"content_type", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsTexture, content_type)},
  {"gridsize", mjXAttr::kInt, 2, true, false, false, false, (int)offsetof(mjsTexture, gridsize)},
  {"gridlayout", mjXAttr::kChars, 12, false, false, false, false, (int)offsetof(mjsTexture, gridlayout)},
  {"builtin", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTexture, builtin), builtin_map, builtin_sz},
  {"rgb1", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsTexture, rgb1)},
  {"rgb2", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsTexture, rgb2)},
  {"mark", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTexture, mark), mark_map, mark_sz},
  {"markrgb", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsTexture, markrgb)},
  {"random", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTexture, random)},
  {"width", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsTexture, width)},
  {"height", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsTexture, height)},
  {"hflip", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsTexture, hflip)},
  {"vflip", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsTexture, vflip)},
  {"nchannel", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsTexture, nchannel)},
  {"embed", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTexture, embed), FalseTrueAuto_map, FalseTrueAuto_sz},
};
inline constexpr int kTextureAttrsN = sizeof(kTextureAttrs) / sizeof(kTextureAttrs[0]);

// material (mjsMaterial)
inline constexpr mjXAttr kMaterialAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"texrepeat", mjXAttr::kFloat, 2, true, false, false, false, (int)offsetof(mjsMaterial, texrepeat)},
  {"texuniform", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsMaterial, texuniform)},
  {"emission", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsMaterial, emission)},
  {"specular", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsMaterial, specular)},
  {"shininess", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsMaterial, shininess)},
  {"reflectance", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsMaterial, reflectance)},
  {"metallic", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsMaterial, metallic)},
  {"roughness", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsMaterial, roughness)},
  {"rgba", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjsMaterial, rgba)},
};
inline constexpr int kMaterialAttrsN = sizeof(kMaterialAttrs) / sizeof(kMaterialAttrs[0]);

// body (mjsBody)
inline constexpr mjXAttr kBodyAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"pos", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsBody, pos)},
  {"mocap", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsBody, mocap)},
  {"gravcomp", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsBody, gravcomp)},
  {"sleep", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsBody, sleep), bodysleep_map, bodysleep_sz},
  {"simple", mjXAttr::kEnumByte, 1, true, false, false, false, (int)offsetof(mjsBody, simple), FalseAuto_map, FalseAuto_sz},
  {"fuse", mjXAttr::kEnumByte, 1, true, false, false, false, (int)offsetof(mjsBody, fuse), FalseAuto_map, FalseAuto_sz},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsBody, userdata)},
};
inline constexpr int kBodyAttrsN = sizeof(kBodyAttrs) / sizeof(kBodyAttrs[0]);

// inertial (mjsBody)
inline constexpr mjXAttr kInertialAttrs[] = {
  {"pos", mjXAttr::kDouble, 3, true, true, false, false, (int)offsetof(mjsBody, ipos)},
  {"mass", mjXAttr::kDouble, 1, true, true, false, false, (int)offsetof(mjsBody, mass)},
  {"diaginertia", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsBody, inertia)},
};
inline constexpr int kInertialAttrsN = sizeof(kInertialAttrs) / sizeof(kInertialAttrs[0]);

// joint (mjsJoint)
inline constexpr mjXAttr kJointAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"type", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsJoint, type), jointtype_map, jointtype_sz},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsJoint, group)},
  {"pos", mjXAttr::kDouble, 3, true, false, false, true, (int)offsetof(mjsJoint, pos)},
  {"axis", mjXAttr::kDouble, 3, true, false, false, true, (int)offsetof(mjsJoint, axis)},
  {"springdamper", mjXAttr::kDouble, 2, true, false, false, true, (int)offsetof(mjsJoint, springdamper)},
  {"limited", mjXAttr::kEnum, 1, true, false, false, true, (int)offsetof(mjsJoint, limited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"actuatorfrclimited", mjXAttr::kEnum, 1, true, false, false, true, (int)offsetof(mjsJoint, actfrclimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"solreflimit", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsJoint, solref_limit)},
  {"solimplimit", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsJoint, solimp_limit)},
  {"solreffriction", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsJoint, solref_friction)},
  {"solimpfriction", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsJoint, solimp_friction)},
  {"stiffness", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsJoint, stiffness)},
  {"range", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsJoint, range)},
  {"actuatorfrcrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsJoint, actfrcrange)},
  {"actuatorgravcomp", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsJoint, actgravcomp)},
  {"margin", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsJoint, margin)},
  {"ref", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsJoint, ref)},
  {"springref", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsJoint, springref)},
  {"springyield", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsJoint, springyield)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsJoint, armature)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsJoint, damping)},
  {"frictionloss", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsJoint, frictionloss)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsJoint, userdata)},
};
inline constexpr int kJointAttrsN = sizeof(kJointAttrs) / sizeof(kJointAttrs[0]);

// geom (mjsGeom)
inline constexpr mjXAttr kGeomAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"type", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsGeom, type), geomtype_map, geomtype_sz},
  {"contype", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsGeom, contype)},
  {"conaffinity", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsGeom, conaffinity)},
  {"condim", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsGeom, condim)},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsGeom, group)},
  {"priority", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsGeom, priority)},
  {"size", mjXAttr::kDouble, 3, false, false, false, true, (int)offsetof(mjsGeom, size)},
  {"material", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsGeom, material)},
  {"friction", mjXAttr::kDouble, 3, false, false, false, false, (int)offsetof(mjsGeom, friction)},
  {"mass", mjXAttr::kDouble, 1, true, false, false, true, (int)offsetof(mjsGeom, mass)},
  {"density", mjXAttr::kDouble, 1, true, false, false, true, (int)offsetof(mjsGeom, density)},
  {"shellinertia", mjXAttr::kEnum, 1, true, false, false, true, (int)offsetof(mjsGeom, typeinertia), bool_map, 2},
  {"solmix", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsGeom, solmix)},
  {"solref", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsGeom, solref)},
  {"solimp", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsGeom, solimp)},
  {"margin", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsGeom, margin)},
  {"gap", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsGeom, gap)},
  {"surfacevel", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsGeom, surfacevel)},
  {"adhesion", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsGeom, adhesion)},
  {"fromto", mjXAttr::kDouble, 6, true, false, false, true, (int)offsetof(mjsGeom, fromto)},
  {"pos", mjXAttr::kDouble, 3, true, false, false, true, (int)offsetof(mjsGeom, pos)},
  {"hfield", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsGeom, hfieldname)},
  {"mesh", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsGeom, meshname)},
  {"fitscale", mjXAttr::kDouble, 1, true, false, false, true, (int)offsetof(mjsGeom, fitscale)},
  {"rgba", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjsGeom, rgba)},
  {"castshadow", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsGeom, castshadow)},
  {"fluidcoef", mjXAttr::kNum, 5, false, false, false, false, (int)offsetof(mjsGeom, fluid_coefs)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsGeom, userdata)},
};
inline constexpr int kGeomAttrsN = sizeof(kGeomAttrs) / sizeof(kGeomAttrs[0]);

// site (mjsSite)
inline constexpr mjXAttr kSiteAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"type", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSite, type), geomtype_map, geomtype_sz},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsSite, group)},
  {"pos", mjXAttr::kDouble, 3, true, false, false, true, (int)offsetof(mjsSite, pos)},
  {"material", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSite, material)},
  {"mesh", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSite, meshname)},
  {"size", mjXAttr::kDouble, 3, false, false, false, true, (int)offsetof(mjsSite, size)},
  {"fromto", mjXAttr::kDouble, 6, true, false, false, true, (int)offsetof(mjsSite, fromto)},
  {"rgba", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjsSite, rgba)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsSite, userdata)},
};
inline constexpr int kSiteAttrsN = sizeof(kSiteAttrs) / sizeof(kSiteAttrs[0]);

// camera (mjsCamera)
inline constexpr mjXAttr kCameraAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"projection", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsCamera, proj), projection_map, projection_sz},
  {"fovy", mjXAttr::kDouble, 1, true, false, false, true, (int)offsetof(mjsCamera, fovy)},
  {"ipd", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsCamera, ipd)},
  {"resolution", mjXAttr::kInt, 2, true, false, false, false, (int)offsetof(mjsCamera, resolution)},
  {"output", mjXAttr::kFlags, 1, true, false, false, false, (int)offsetof(mjsCamera, output), camout_map, camout_sz},
  {"pos", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsCamera, pos)},
  {"mode", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsCamera, mode), camlight_map, camlight_sz},
  {"target", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsCamera, targetbody)},
  {"focal", mjXAttr::kFloat, 2, true, false, false, true, (int)offsetof(mjsCamera, focal_length)},
  {"focalpixel", mjXAttr::kFloat, 2, true, false, false, true, (int)offsetof(mjsCamera, focal_pixel)},
  {"principal", mjXAttr::kFloat, 2, true, false, false, true, (int)offsetof(mjsCamera, principal_length)},
  {"principalpixel", mjXAttr::kFloat, 2, true, false, false, true, (int)offsetof(mjsCamera, principal_pixel)},
  {"sensorsize", mjXAttr::kFloat, 2, true, false, false, true, (int)offsetof(mjsCamera, sensor_size)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsCamera, userdata)},
};
inline constexpr int kCameraAttrsN = sizeof(kCameraAttrs) / sizeof(kCameraAttrs[0]);

// light (mjsLight)
inline constexpr mjXAttr kLightAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"castshadow", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsLight, castshadow)},
  {"active", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsLight, active)},
  {"pos", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsLight, pos)},
  {"dir", mjXAttr::kDouble, 3, true, false, false, false, (int)offsetof(mjsLight, dir)},
  {"bulbradius", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsLight, bulbradius)},
  {"intensity", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsLight, intensity)},
  {"range", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsLight, range)},
  {"attenuation", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjsLight, attenuation)},
  {"cutoff", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsLight, cutoff)},
  {"softness", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsLight, softness)},
  {"exponent", mjXAttr::kFloat, 1, true, false, false, false, (int)offsetof(mjsLight, exponent)},
  {"ambient", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjsLight, ambient)},
  {"diffuse", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjsLight, diffuse)},
  {"specular", mjXAttr::kFloat, 3, true, false, false, false, (int)offsetof(mjsLight, specular)},
  {"mode", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsLight, mode), camlight_map, camlight_sz},
  {"target", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsLight, targetbody)},
  {"texture", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsLight, texture)},
};
inline constexpr int kLightAttrsN = sizeof(kLightAttrs) / sizeof(kLightAttrs[0]);

// elasticity (mjsFlex)
inline constexpr mjXAttr kElasticityAttrs[] = {
  {"young", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, young)},
  {"poisson", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, poisson)},
  {"shear", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, shear)},
  {"damping", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, damping)},
  {"thickness", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, thickness)},
  {"yield", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, stressyield)},
  {"elastic2d", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsFlex, elastic2d), elastic2d_map, elastic2d_sz},
  {"section", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsFlex, section), flexsection_map, flexsection_sz},
  {"sectionsize", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsFlex, sectionsize)},
};
inline constexpr int kElasticityAttrsN = sizeof(kElasticityAttrs) / sizeof(kElasticityAttrs[0]);

// flexcomp_contact (mjsFlex)
inline constexpr mjXAttr kFlexcomp_contactAttrs[] = {
  {"contype", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, contype)},
  {"conaffinity", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, conaffinity)},
  {"condim", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, condim)},
  {"priority", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, priority)},
  {"friction", mjXAttr::kDouble, 3, false, false, false, false, (int)offsetof(mjsFlex, friction)},
  {"solmix", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, solmix)},
  {"solref", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsFlex, solref)},
  {"solimp", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsFlex, solimp)},
  {"margin", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, margin)},
  {"gap", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, gap)},
  {"selfcollide", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsFlex, selfcollide), flexself_map, flexself_sz},
  {"activelayers", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, activelayers)},
  {"passive", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsFlex, passive), bool_map, 2},
};
inline constexpr int kFlexcomp_contactAttrsN = sizeof(kFlexcomp_contactAttrs) / sizeof(kFlexcomp_contactAttrs[0]);

// flex (mjsFlex)
inline constexpr mjXAttr kFlexAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, group)},
  {"dim", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsFlex, dim)},
  {"radius", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, radius)},
  {"material", mjXAttr::kStringVec, 1, true, false, false, true, (int)offsetof(mjsFlex, material)},
  {"rgba", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjsFlex, rgba)},
  {"flatskin", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsFlex, flatskin)},
  {"body", mjXAttr::kStringVec, 1, true, true, false, false, (int)offsetof(mjsFlex, vertbody)},
  {"vertex", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsFlex, vert)},
  {"element", mjXAttr::kIntVec, 1, true, true, false, false, (int)offsetof(mjsFlex, elem)},
  {"texcoord", mjXAttr::kFloatVec, 1, true, false, false, false, (int)offsetof(mjsFlex, texcoord)},
  {"elemtexcoord", mjXAttr::kIntVec, 1, true, false, false, false, (int)offsetof(mjsFlex, elemtexcoord)},
  {"node", mjXAttr::kStringVec, 1, true, false, false, false, (int)offsetof(mjsFlex, nodebody)},
  {"nodecoord", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsFlex, node)},
};
inline constexpr int kFlexAttrsN = sizeof(kFlexAttrs) / sizeof(kFlexAttrs[0]);

// flex_edge (mjsFlex)
inline constexpr mjXAttr kFlex_edgeAttrs[] = {
  {"stiffness", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, edgestiffness)},
  {"damping", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsFlex, edgedamping)},
};
inline constexpr int kFlex_edgeAttrsN = sizeof(kFlex_edgeAttrs) / sizeof(kFlex_edgeAttrs[0]);

// pair (mjsPair)
inline constexpr mjXAttr kPairAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"geom1", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsPair, geomname1)},
  {"geom2", mjXAttr::kString, 1, true, false, true, false, (int)offsetof(mjsPair, geomname2)},
  {"condim", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsPair, condim)},
  {"friction", mjXAttr::kDouble, 5, false, false, false, false, (int)offsetof(mjsPair, friction)},
  {"solref", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsPair, solref)},
  {"solreffriction", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsPair, solreffriction)},
  {"solimp", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsPair, solimp)},
  {"gap", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsPair, gap)},
  {"margin", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsPair, margin)},
  {"adhesion", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsPair, adhesion)},
};
inline constexpr int kPairAttrsN = sizeof(kPairAttrs) / sizeof(kPairAttrs[0]);

// exclude (mjsExclude)
inline constexpr mjXAttr kExcludeAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"body1", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsExclude, bodyname1)},
  {"body2", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsExclude, bodyname2)},
};
inline constexpr int kExcludeAttrsN = sizeof(kExcludeAttrs) / sizeof(kExcludeAttrs[0]);

// spatial (mjsTendon)
inline constexpr mjXAttr kSpatialAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsTendon, group)},
  {"limited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTendon, limited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"actuatorfrclimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTendon, actfrclimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"range", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsTendon, range)},
  {"actuatorfrcrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsTendon, actfrcrange)},
  {"solreflimit", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsTendon, solref_limit)},
  {"solimplimit", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsTendon, solimp_limit)},
  {"solreffriction", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsTendon, solref_friction)},
  {"solimpfriction", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsTendon, solimp_friction)},
  {"frictionloss", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, frictionloss)},
  {"springyield", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, springyield)},
  {"width", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, width)},
  {"material", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsTendon, material)},
  {"margin", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, margin)},
  {"stiffness", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsTendon, stiffness)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsTendon, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, armature)},
  {"rgba", mjXAttr::kFloat, 4, true, false, false, false, (int)offsetof(mjsTendon, rgba)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsTendon, userdata)},
};
inline constexpr int kSpatialAttrsN = sizeof(kSpatialAttrs) / sizeof(kSpatialAttrs[0]);

// fixed (mjsTendon)
inline constexpr mjXAttr kFixedAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsTendon, group)},
  {"limited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTendon, limited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"actuatorfrclimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsTendon, actfrclimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"range", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsTendon, range)},
  {"actuatorfrcrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsTendon, actfrcrange)},
  {"solreflimit", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsTendon, solref_limit)},
  {"solimplimit", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsTendon, solimp_limit)},
  {"solreffriction", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsTendon, solref_friction)},
  {"solimpfriction", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsTendon, solimp_friction)},
  {"frictionloss", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, frictionloss)},
  {"springyield", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, springyield)},
  {"margin", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, margin)},
  {"stiffness", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsTendon, stiffness)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsTendon, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsTendon, armature)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsTendon, userdata)},
};
inline constexpr int kFixedAttrsN = sizeof(kFixedAttrs) / sizeof(kFixedAttrs[0]);

// general (mjsActuator)
inline constexpr mjXAttr kGeneralAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"actlimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, actlimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"actrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, actrange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
  {"actdim", mjXAttr::kInt, 1, true, false, false, true, (int)offsetof(mjsActuator, actdim)},
  {"velrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, velrange)},
  {"ffrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ffrange)},
  {"inheritrange", mjXAttr::kDouble, 1, true, false, false, true, (int)offsetof(mjsActuator, inheritrange)},
  {"dyntype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, dyntype), dyn_map, dyn_sz},
  {"gaintype", mjXAttr::kEnum, 1, true, false, false, true, (int)offsetof(mjsActuator, gaintype), gain_map, gain_sz},
  {"biastype", mjXAttr::kEnum, 1, true, false, false, true, (int)offsetof(mjsActuator, biastype), bias_map, bias_sz},
  {"dynprm", mjXAttr::kDouble, mjNDYN, false, false, false, false, (int)offsetof(mjsActuator, dynprm)},
  {"gainprm", mjXAttr::kDouble, mjNGAIN, false, false, false, true, (int)offsetof(mjsActuator, gainprm)},
  {"biasprm", mjXAttr::kDouble, mjNBIAS, false, false, false, true, (int)offsetof(mjsActuator, biasprm)},
  {"actearly", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsActuator, actearly)},
};
inline constexpr int kGeneralAttrsN = sizeof(kGeneralAttrs) / sizeof(kGeneralAttrs[0]);

// motor (mjsActuator)
inline constexpr mjXAttr kMotorAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kMotorAttrsN = sizeof(kMotorAttrs) / sizeof(kMotorAttrs[0]);

// position (mjsActuator)
inline constexpr mjXAttr kPositionAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kPositionAttrsN = sizeof(kPositionAttrs) / sizeof(kPositionAttrs[0]);

// velocity (mjsActuator)
inline constexpr mjXAttr kVelocityAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kVelocityAttrsN = sizeof(kVelocityAttrs) / sizeof(kVelocityAttrs[0]);

// intvelocity (mjsActuator)
inline constexpr mjXAttr kIntvelocityAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"actlimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, actlimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"actrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, actrange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kIntvelocityAttrsN = sizeof(kIntvelocityAttrs) / sizeof(kIntvelocityAttrs[0]);

// orientation (mjsActuator)
inline constexpr mjXAttr kOrientationAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
};
inline constexpr int kOrientationAttrsN = sizeof(kOrientationAttrs) / sizeof(kOrientationAttrs[0]);

// pid (mjsActuator)
inline constexpr mjXAttr kPidAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"velrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, velrange)},
  {"ffrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ffrange)},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kPidAttrsN = sizeof(kPidAttrs) / sizeof(kPidAttrs[0]);

// damper (mjsActuator)
inline constexpr mjXAttr kDamperAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kDamperAttrsN = sizeof(kDamperAttrs) / sizeof(kDamperAttrs[0]);

// cylinder (mjsActuator)
inline constexpr mjXAttr kCylinderAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kCylinderAttrsN = sizeof(kCylinderAttrs) / sizeof(kCylinderAttrs[0]);

// muscle (mjsActuator)
inline constexpr mjXAttr kMuscleAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kMuscleAttrsN = sizeof(kMuscleAttrs) / sizeof(kMuscleAttrs[0]);

// adhesion (mjsActuator)
inline constexpr mjXAttr kAdhesionAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"forcelimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, forcelimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"forcerange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, forcerange)},
};
inline constexpr int kAdhesionAttrsN = sizeof(kAdhesionAttrs) / sizeof(kAdhesionAttrs[0]);

// dcmotor (mjsActuator)
inline constexpr mjXAttr kDcmotorAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"group", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, group)},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsActuator, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, delay)},
  {"ctrlrange", mjXAttr::kDouble, 2, true, false, false, false, (int)offsetof(mjsActuator, ctrlrange)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsActuator, userdata)},
  {"ctrllimited", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsActuator, ctrllimited), FalseTrueAuto_map, FalseTrueAuto_sz},
  {"lengthrange", mjXAttr::kDouble, 2, true, false, true, false, (int)offsetof(mjsActuator, lengthrange)},
  {"gear", mjXAttr::kDouble, 6, false, false, false, false, (int)offsetof(mjsActuator, gear)},
  {"damping", mjXAttr::kDouble, mjNPOLY+1, false, false, false, false, (int)offsetof(mjsActuator, damping)},
  {"armature", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsActuator, armature)},
};
inline constexpr int kDcmotorAttrsN = sizeof(kDcmotorAttrs) / sizeof(kDcmotorAttrs[0]);

// touch (mjsSensor)
inline constexpr mjXAttr kTouchAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TOUCH},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTouchAttrsN = sizeof(kTouchAttrs) / sizeof(kTouchAttrs[0]);

// accelerometer (mjsSensor)
inline constexpr mjXAttr kAccelerometerAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_ACCELEROMETER},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kAccelerometerAttrsN = sizeof(kAccelerometerAttrs) / sizeof(kAccelerometerAttrs[0]);

// velocimeter (mjsSensor)
inline constexpr mjXAttr kVelocimeterAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_VELOCIMETER},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kVelocimeterAttrsN = sizeof(kVelocimeterAttrs) / sizeof(kVelocimeterAttrs[0]);

// gyro (mjsSensor)
inline constexpr mjXAttr kGyroAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_GYRO},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kGyroAttrsN = sizeof(kGyroAttrs) / sizeof(kGyroAttrs[0]);

// force (mjsSensor)
inline constexpr mjXAttr kForceAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FORCE},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kForceAttrsN = sizeof(kForceAttrs) / sizeof(kForceAttrs[0]);

// torque (mjsSensor)
inline constexpr mjXAttr kTorqueAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TORQUE},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTorqueAttrsN = sizeof(kTorqueAttrs) / sizeof(kTorqueAttrs[0]);

// magnetometer (mjsSensor)
inline constexpr mjXAttr kMagnetometerAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_MAGNETOMETER},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kMagnetometerAttrsN = sizeof(kMagnetometerAttrs) / sizeof(kMagnetometerAttrs[0]);

// camprojection (mjsSensor)
inline constexpr mjXAttr kCamprojectionAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_CAMPROJECTION},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_SITE},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), nullptr, 0, mjOBJ_CAMERA},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"camera", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kCamprojectionAttrsN = sizeof(kCamprojectionAttrs) / sizeof(kCamprojectionAttrs[0]);

// jointpos (mjsSensor)
inline constexpr mjXAttr kJointposAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_JOINTPOS},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kJointposAttrsN = sizeof(kJointposAttrs) / sizeof(kJointposAttrs[0]);

// jointvel (mjsSensor)
inline constexpr mjXAttr kJointvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_JOINTVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kJointvelAttrsN = sizeof(kJointvelAttrs) / sizeof(kJointvelAttrs[0]);

// tendonpos (mjsSensor)
inline constexpr mjXAttr kTendonposAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TENDONPOS},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_TENDON},
  {"tendon", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTendonposAttrsN = sizeof(kTendonposAttrs) / sizeof(kTendonposAttrs[0]);

// tendonvel (mjsSensor)
inline constexpr mjXAttr kTendonvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TENDONVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_TENDON},
  {"tendon", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTendonvelAttrsN = sizeof(kTendonvelAttrs) / sizeof(kTendonvelAttrs[0]);

// actuatorpos (mjsSensor)
inline constexpr mjXAttr kActuatorposAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_ACTUATORPOS},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_ACTUATOR},
  {"actuator", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kActuatorposAttrsN = sizeof(kActuatorposAttrs) / sizeof(kActuatorposAttrs[0]);

// actuatorvel (mjsSensor)
inline constexpr mjXAttr kActuatorvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_ACTUATORVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_ACTUATOR},
  {"actuator", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kActuatorvelAttrsN = sizeof(kActuatorvelAttrs) / sizeof(kActuatorvelAttrs[0]);

// actuatorfrc (mjsSensor)
inline constexpr mjXAttr kActuatorfrcAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_ACTUATORFRC},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_ACTUATOR},
  {"actuator", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kActuatorfrcAttrsN = sizeof(kActuatorfrcAttrs) / sizeof(kActuatorfrcAttrs[0]);

// jointactuatorfrc (mjsSensor)
inline constexpr mjXAttr kJointactuatorfrcAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_JOINTACTFRC},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kJointactuatorfrcAttrsN = sizeof(kJointactuatorfrcAttrs) / sizeof(kJointactuatorfrcAttrs[0]);

// tendonactuatorfrc (mjsSensor)
inline constexpr mjXAttr kTendonactuatorfrcAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TENDONACTFRC},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_TENDON},
  {"tendon", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTendonactuatorfrcAttrsN = sizeof(kTendonactuatorfrcAttrs) / sizeof(kTendonactuatorfrcAttrs[0]);

// ballquat (mjsSensor)
inline constexpr mjXAttr kBallquatAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_BALLQUAT},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kBallquatAttrsN = sizeof(kBallquatAttrs) / sizeof(kBallquatAttrs[0]);

// ballangvel (mjsSensor)
inline constexpr mjXAttr kBallangvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_BALLANGVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kBallangvelAttrsN = sizeof(kBallangvelAttrs) / sizeof(kBallangvelAttrs[0]);

// jointlimitpos (mjsSensor)
inline constexpr mjXAttr kJointlimitposAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_JOINTLIMITPOS},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kJointlimitposAttrsN = sizeof(kJointlimitposAttrs) / sizeof(kJointlimitposAttrs[0]);

// jointlimitvel (mjsSensor)
inline constexpr mjXAttr kJointlimitvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_JOINTLIMITVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kJointlimitvelAttrsN = sizeof(kJointlimitvelAttrs) / sizeof(kJointlimitvelAttrs[0]);

// jointlimitfrc (mjsSensor)
inline constexpr mjXAttr kJointlimitfrcAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_JOINTLIMITFRC},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_JOINT},
  {"joint", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kJointlimitfrcAttrsN = sizeof(kJointlimitfrcAttrs) / sizeof(kJointlimitfrcAttrs[0]);

// tendonlimitpos (mjsSensor)
inline constexpr mjXAttr kTendonlimitposAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TENDONLIMITPOS},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_TENDON},
  {"tendon", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTendonlimitposAttrsN = sizeof(kTendonlimitposAttrs) / sizeof(kTendonlimitposAttrs[0]);

// tendonlimitvel (mjsSensor)
inline constexpr mjXAttr kTendonlimitvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TENDONLIMITVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_TENDON},
  {"tendon", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTendonlimitvelAttrsN = sizeof(kTendonlimitvelAttrs) / sizeof(kTendonlimitvelAttrs[0]);

// tendonlimitfrc (mjsSensor)
inline constexpr mjXAttr kTendonlimitfrcAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_TENDONLIMITFRC},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_TENDON},
  {"tendon", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kTendonlimitfrcAttrsN = sizeof(kTendonlimitfrcAttrs) / sizeof(kTendonlimitfrcAttrs[0]);

// framepos (mjsSensor)
inline constexpr mjXAttr kFrameposAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEPOS},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFrameposAttrsN = sizeof(kFrameposAttrs) / sizeof(kFrameposAttrs[0]);

// framequat (mjsSensor)
inline constexpr mjXAttr kFramequatAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEQUAT},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFramequatAttrsN = sizeof(kFramequatAttrs) / sizeof(kFramequatAttrs[0]);

// framexaxis (mjsSensor)
inline constexpr mjXAttr kFramexaxisAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEXAXIS},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFramexaxisAttrsN = sizeof(kFramexaxisAttrs) / sizeof(kFramexaxisAttrs[0]);

// frameyaxis (mjsSensor)
inline constexpr mjXAttr kFrameyaxisAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEYAXIS},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFrameyaxisAttrsN = sizeof(kFrameyaxisAttrs) / sizeof(kFrameyaxisAttrs[0]);

// framezaxis (mjsSensor)
inline constexpr mjXAttr kFramezaxisAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEZAXIS},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFramezaxisAttrsN = sizeof(kFramezaxisAttrs) / sizeof(kFramezaxisAttrs[0]);

// framelinvel (mjsSensor)
inline constexpr mjXAttr kFramelinvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMELINVEL},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFramelinvelAttrsN = sizeof(kFramelinvelAttrs) / sizeof(kFramelinvelAttrs[0]);

// frameangvel (mjsSensor)
inline constexpr mjXAttr kFrameangvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEANGVEL},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"reftype", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), frameobj_map, frameobj_sz},
  {"refname", mjXAttr::kString, 1, true, false, false, false, (int)offsetof(mjsSensor, refname)},
};
inline constexpr int kFrameangvelAttrsN = sizeof(kFrameangvelAttrs) / sizeof(kFrameangvelAttrs[0]);

// framelinacc (mjsSensor)
inline constexpr mjXAttr kFramelinaccAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMELINACC},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kFramelinaccAttrsN = sizeof(kFramelinaccAttrs) / sizeof(kFramelinaccAttrs[0]);

// frameangacc (mjsSensor)
inline constexpr mjXAttr kFrameangaccAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_FRAMEANGACC},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kFrameangaccAttrsN = sizeof(kFrameangaccAttrs) / sizeof(kFrameangaccAttrs[0]);

// subtreecom (mjsSensor)
inline constexpr mjXAttr kSubtreecomAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_SUBTREECOM},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_BODY},
  {"body", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kSubtreecomAttrsN = sizeof(kSubtreecomAttrs) / sizeof(kSubtreecomAttrs[0]);

// subtreelinvel (mjsSensor)
inline constexpr mjXAttr kSubtreelinvelAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_SUBTREELINVEL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_BODY},
  {"body", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kSubtreelinvelAttrsN = sizeof(kSubtreelinvelAttrs) / sizeof(kSubtreelinvelAttrs[0]);

// subtreeangmom (mjsSensor)
inline constexpr mjXAttr kSubtreeangmomAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_SUBTREEANGMOM},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_BODY},
  {"body", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
};
inline constexpr int kSubtreeangmomAttrsN = sizeof(kSubtreeangmomAttrs) / sizeof(kSubtreeangmomAttrs[0]);

// insidesite (mjsSensor)
inline constexpr mjXAttr kInsidesiteAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_INSIDESITE},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, reftype), nullptr, 0, mjOBJ_SITE},
  {"site", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, refname)},
  {"objtype", mjXAttr::kEnum, 1, true, true, false, false, (int)offsetof(mjsSensor, objtype), frameobj_map, frameobj_sz},
  {"objname", mjXAttr::kString, 1, true, true, false, false, (int)offsetof(mjsSensor, objname)},
  {"enclosed", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, intprm), bool_map, 2},
};
inline constexpr int kInsidesiteAttrsN = sizeof(kInsidesiteAttrs) / sizeof(kInsidesiteAttrs[0]);

// e_potential (mjsSensor)
inline constexpr mjXAttr kE_potentialAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_E_POTENTIAL},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_UNKNOWN},
};
inline constexpr int kE_potentialAttrsN = sizeof(kE_potentialAttrs) / sizeof(kE_potentialAttrs[0]);

// e_kinetic (mjsSensor)
inline constexpr mjXAttr kE_kineticAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_E_KINETIC},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_UNKNOWN},
};
inline constexpr int kE_kineticAttrsN = sizeof(kE_kineticAttrs) / sizeof(kE_kineticAttrs[0]);

// clock (mjsSensor)
inline constexpr mjXAttr kClockAttrs[] = {
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, type), nullptr, 0, mjSENS_CLOCK},
  {nullptr, mjXAttr::kConst, 1, true, false, false, false, (int)offsetof(mjsSensor, objtype), nullptr, 0, mjOBJ_UNKNOWN},
};
inline constexpr int kClockAttrsN = sizeof(kClockAttrs) / sizeof(kClockAttrs[0]);

// key (mjsKey)
inline constexpr mjXAttr kKeyAttrs[] = {
  {"time", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsKey, time)},
  {"qpos", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, qpos)},
  {"qvel", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, qvel)},
  {"act", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, act)},
  {"mpos", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, mpos)},
  {"mquat", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, mquat)},
  {"ctrl", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, ctrl)},
  {"plastic", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsKey, plastic)},
};
inline constexpr int kKeyAttrsN = sizeof(kKeyAttrs) / sizeof(kKeyAttrs[0]);

// sensors fully described by the schema: dispatch by tag
struct mjXSensorEntry { const char* tag; const mjXAttr* rows; int n; };
inline constexpr mjXSensorEntry kSensorDispatch[] = {
  {"touch", kTouchAttrs, kTouchAttrsN},
  {"accelerometer", kAccelerometerAttrs, kAccelerometerAttrsN},
  {"velocimeter", kVelocimeterAttrs, kVelocimeterAttrsN},
  {"gyro", kGyroAttrs, kGyroAttrsN},
  {"force", kForceAttrs, kForceAttrsN},
  {"torque", kTorqueAttrs, kTorqueAttrsN},
  {"magnetometer", kMagnetometerAttrs, kMagnetometerAttrsN},
  {"camprojection", kCamprojectionAttrs, kCamprojectionAttrsN},
  {"jointpos", kJointposAttrs, kJointposAttrsN},
  {"jointvel", kJointvelAttrs, kJointvelAttrsN},
  {"tendonpos", kTendonposAttrs, kTendonposAttrsN},
  {"tendonvel", kTendonvelAttrs, kTendonvelAttrsN},
  {"actuatorpos", kActuatorposAttrs, kActuatorposAttrsN},
  {"actuatorvel", kActuatorvelAttrs, kActuatorvelAttrsN},
  {"actuatorfrc", kActuatorfrcAttrs, kActuatorfrcAttrsN},
  {"jointactuatorfrc", kJointactuatorfrcAttrs, kJointactuatorfrcAttrsN},
  {"tendonactuatorfrc", kTendonactuatorfrcAttrs, kTendonactuatorfrcAttrsN},
  {"ballquat", kBallquatAttrs, kBallquatAttrsN},
  {"ballangvel", kBallangvelAttrs, kBallangvelAttrsN},
  {"jointlimitpos", kJointlimitposAttrs, kJointlimitposAttrsN},
  {"jointlimitvel", kJointlimitvelAttrs, kJointlimitvelAttrsN},
  {"jointlimitfrc", kJointlimitfrcAttrs, kJointlimitfrcAttrsN},
  {"tendonlimitpos", kTendonlimitposAttrs, kTendonlimitposAttrsN},
  {"tendonlimitvel", kTendonlimitvelAttrs, kTendonlimitvelAttrsN},
  {"tendonlimitfrc", kTendonlimitfrcAttrs, kTendonlimitfrcAttrsN},
  {"framepos", kFrameposAttrs, kFrameposAttrsN},
  {"framequat", kFramequatAttrs, kFramequatAttrsN},
  {"framexaxis", kFramexaxisAttrs, kFramexaxisAttrsN},
  {"frameyaxis", kFrameyaxisAttrs, kFrameyaxisAttrsN},
  {"framezaxis", kFramezaxisAttrs, kFramezaxisAttrsN},
  {"framelinvel", kFramelinvelAttrs, kFramelinvelAttrsN},
  {"frameangvel", kFrameangvelAttrs, kFrameangvelAttrsN},
  {"framelinacc", kFramelinaccAttrs, kFramelinaccAttrsN},
  {"frameangacc", kFrameangaccAttrs, kFrameangaccAttrsN},
  {"insidesite", kInsidesiteAttrs, kInsidesiteAttrsN},
  {"subtreecom", kSubtreecomAttrs, kSubtreecomAttrsN},
  {"subtreelinvel", kSubtreelinvelAttrs, kSubtreelinvelAttrsN},
  {"subtreeangmom", kSubtreeangmomAttrs, kSubtreeangmomAttrsN},
  {"e_potential", kE_potentialAttrs, kE_potentialAttrsN},
  {"e_kinetic", kE_kineticAttrs, kE_kineticAttrsN},
  {"clock", kClockAttrs, kClockAttrsN},
};
inline constexpr int kSensorDispatchN = sizeof(kSensorDispatch) / sizeof(kSensorDispatch[0]);

// actuator tags and their mechanical rows: dispatch by tag or type
struct mjXActuatorEntry { const char* tag; int type; const mjXAttr* rows; int n; };
inline constexpr mjXActuatorEntry kActuatorDispatch[] = {
  {"general", mjACTUATOR_GENERAL, kGeneralAttrs, kGeneralAttrsN},
  {"motor", mjACTUATOR_MOTOR, kMotorAttrs, kMotorAttrsN},
  {"position", mjACTUATOR_POSITION, kPositionAttrs, kPositionAttrsN},
  {"velocity", mjACTUATOR_VELOCITY, kVelocityAttrs, kVelocityAttrsN},
  {"intvelocity", mjACTUATOR_INTVELOCITY, kIntvelocityAttrs, kIntvelocityAttrsN},
  {"damper", mjACTUATOR_DAMPER, kDamperAttrs, kDamperAttrsN},
  {"cylinder", mjACTUATOR_CYLINDER, kCylinderAttrs, kCylinderAttrsN},
  {"muscle", mjACTUATOR_MUSCLE, kMuscleAttrs, kMuscleAttrsN},
  {"adhesion", mjACTUATOR_ADHESION, kAdhesionAttrs, kAdhesionAttrsN},
  {"pid", mjACTUATOR_PID, kPidAttrs, kPidAttrsN},
  {"orientation", mjACTUATOR_ORIENTATION, kOrientationAttrs, kOrientationAttrsN},
  {"dcmotor", mjACTUATOR_DCMOTOR, kDcmotorAttrs, kDcmotorAttrsN},
};
inline constexpr int kActuatorDispatchN = sizeof(kActuatorDispatch) / sizeof(kActuatorDispatch[0]);

// group equality_base (mjsEquality)
inline constexpr mjXAttr kEqualityBaseAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"active", mjXAttr::kBool, 1, true, false, false, false, (int)offsetof(mjsEquality, active)},
  {"solref", mjXAttr::kNum, mjNREF, false, false, false, false, (int)offsetof(mjsEquality, solref)},
  {"solimp", mjXAttr::kNum, mjNIMP, false, false, false, false, (int)offsetof(mjsEquality, solimp)},
};
inline constexpr int kEqualityBaseAttrsN = sizeof(kEqualityBaseAttrs) / sizeof(kEqualityBaseAttrs[0]);

// group sensor_base (mjsSensor)
inline constexpr mjXAttr kSensorBaseAttrs[] = {
  {"name", mjXAttr::kName, 1, true, false, true, false, -1},
  {"nsample", mjXAttr::kInt, 1, true, false, false, false, (int)offsetof(mjsSensor, nsample)},
  {"interp", mjXAttr::kEnum, 1, true, false, false, false, (int)offsetof(mjsSensor, interp), interp_map, interp_sz},
  {"delay", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsSensor, delay)},
  {"interval", mjXAttr::kDouble, 2, false, false, false, false, (int)offsetof(mjsSensor, interval)},
  {"cutoff", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsSensor, cutoff)},
  {"noise", mjXAttr::kDouble, 1, true, false, false, false, (int)offsetof(mjsSensor, noise)},
  {"user", mjXAttr::kDoubleVec, 1, true, false, false, false, (int)offsetof(mjsSensor, userdata)},
};
inline constexpr int kSensorBaseAttrsN = sizeof(kSensorBaseAttrs) / sizeof(kSensorBaseAttrs[0]);

// clang-format on
