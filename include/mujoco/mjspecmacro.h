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

#ifndef MUJOCO_MJSPECMACRO_H_
#define MUJOCO_MJSPECMACRO_H_


//-------------------------------- mjsElement ------------------------------------------------------

#define MJSELEMENT_FIELDS       \
    X( mjtObj,   elemtype,  1 ) \
    X( uint64_t, signature, 1 )


//-------------------------------- mjsCompiler -----------------------------------------------------

#define MJSCOMPILER_FIELDS                    \
  X(mjtBool,            autolimits,        1) \
  X(double,             boundmass,         1) \
  X(double,             boundinertia,      1) \
  X(double,             settotalmass,      1) \
  X(mjtBool,            balanceinertia,    1) \
  X(mjtBool,            fitaabb,           1) \
  X(mjtBool,            degree,            1) \
  XVEC(char,            eulerseq,          3) \
  X(mjtBool,            discardvisual,     1) \
  X(mjtBool,            usethread,         1) \
  X(mjtBool,            fusestatic,        1) \
  X(mjtInertiaFromGeom, inertiafromgeom,   1) \
  XVEC(int,             inertiagrouprange, 2) \
  X(mjtBool,            saveinertial,      1) \
  X(mjtBool,            savecompiled,      1) \
  X(mjtBool,            savecanonical,     1) \
  X(mjtBool,            alignfree,         1) \
  X(mjtConflict,        conflict,          1) \
  X(mjtPassthrough,     passthrough,       1) \
  X(mjLROpt,            LRopt,             1) \
  X(mjString*,          meshdir,           1) \
  X(mjString*,          texturedir,        1) \
  X(uint64_t,           authored,          1)

//-------------------------------- mjSpec ----------------------------------------------------------

#define MJSPEC_FIELDS                          \
    X( mjsElement*, element,               1 ) \
    X( mjString*,   modelname,             1 ) \
    X( mjsCompiler, compiler,              1 ) \
    X( mjtBool,     strippath,             1 ) \
    X( mjOption,    option,                1 ) \
    X( mjVisual,    visual,                1 ) \
    X( mjStatistic, stat,                  1 ) \
    X( mjtSize,     memory,                1 ) \
    X( int,         nemax,                 1 ) \
    X( int,         nuserdata,             1 ) \
    X( int,         nuser_body,            1 ) \
    X( int,         nuser_jnt,             1 ) \
    X( int,         nuser_geom,            1 ) \
    X( int,         nuser_site,            1 ) \
    X( int,         nuser_cam,             1 ) \
    X( int,         nuser_tendon,          1 ) \
    X( int,         nuser_actuator,        1 ) \
    X( int,         nuser_sensor,          1 ) \
    X( int,         nkey,                  1 ) \
    X( int,         njmax,                 1 ) \
    X( int,         nconmax,               1 ) \
    X( mjtSize,     nstack,                1 ) \
    X( mjString*,   comment,               1 ) \
    X( mjString*,   modelfiledir,          1 ) \
    X( mjtBool,     hasImplicitPluginElem, 1 )


//-------------------------------- mjsOrientation --------------------------------------------------

#define MJSORIENTATION_FIELDS            \
    X   ( mjtOrientation, type,      1 ) \
    XVEC( double,         axisangle, 4 ) \
    XVEC( double,         xyaxes,    6 ) \
    XVEC( double,         zaxis,     3 ) \
    XVEC( double,         euler,     3 )


//-------------------------------- mjsPlugin -------------------------------------------------------

#define MJSPLUGIN_FIELDS             \
    X( mjsElement*, element,     1 ) \
    X( mjString*,   name,        1 ) \
    X( mjString*,   plugin_name, 1 ) \
    X( mjtBool,     active,      1 ) \
    X( mjString*,   info,        1 )


//-------------------------------- mjsBody ---------------------------------------------------------

#define MJSBODY_FIELDS                           \
    X   ( mjsElement*,     element,          1 ) \
    X   ( mjString*,       childclass,       1 ) \
    XVEC( double,          pos,              3 ) \
    XVEC( double,          quat,             4 ) \
    X   ( mjsOrientation,  alt,              1 ) \
    X   ( double,          mass,             1 ) \
    XVEC( double,          ipos,             3 ) \
    XVEC( double,          iquat,            4 ) \
    XVEC( double,          inertia,          3 ) \
    X   ( mjsOrientation,  ialt,             1 ) \
    XVEC( double,          fullinertia,      6 ) \
    X   ( mjtBool,         mocap,            1 ) \
    X   ( double,          gravcomp,         1 ) \
    X   ( mjtSleepPolicy,  sleep,            1 ) \
    X   ( mjtByte,         simple,           1 ) \
    X   ( mjtByte,         fuse,             1 ) \
    X   ( mjDoubleVec*,    userdata,         1 ) \
    X   ( mjtBool,         explicitinertial, 1 ) \
    X   ( mjsPlugin,       plugin,           1 ) \
    X   ( mjString*,       info,             1 )


//-------------------------------- mjsFrame --------------------------------------------------------

#define MJSFRAME_FIELDS                   \
    X   ( mjsElement*,    element,    1 ) \
    X   ( mjString*,      childclass, 1 ) \
    XVEC( double,         pos,        3 ) \
    XVEC( double,         quat,       4 ) \
    X   ( mjsOrientation, alt,        1 ) \
    XVEC( double,         scale,      3 ) \
    X   ( mjtScaling,     scaling,    1 ) \
    X   ( double,         massscale,  1 ) \
    X   ( mjString*,      info,       1 )


//-------------------------------- mjsJoint --------------------------------------------------------

#define MJSJOINT_FIELDS                               \
    X   ( mjsElement*,   element,         1         ) \
    X   ( mjtJoint,      type,            1         ) \
    XVEC( double,        pos,             3         ) \
    XVEC( double,        axis,            3         ) \
    X   ( double,        ref,             1         ) \
    X   ( mjtAlignFree,  align,           1         ) \
    XVEC( double,        stiffness,       mjNPOLY+1 ) \
    X   ( double,        springref,       1         ) \
    X   ( double,        springyield,     1         ) \
    XVEC( double,        springdamper,    2         ) \
    X   ( mjtLimited,    limited,         1         ) \
    XVEC( double,        range,           2         ) \
    X   ( double,        margin,          1         ) \
    XVEC( mjtNum,        solref_limit,    mjNREF    ) \
    XVEC( mjtNum,        solimp_limit,    mjNIMP    ) \
    X   ( mjtLimited,    actfrclimited,   1         ) \
    XVEC( double,        actfrcrange,     2         ) \
    X   ( double,        armature,        1         ) \
    XVEC( double,        damping,         mjNPOLY+1 ) \
    X   ( double,        frictionloss,    1         ) \
    XVEC( mjtNum,        solref_friction, mjNREF    ) \
    XVEC( mjtNum,        solimp_friction, mjNIMP    ) \
    X   ( int,           group,           1         ) \
    X   ( mjtBool,       actgravcomp,     1         ) \
    X   ( mjDoubleVec*,  userdata,        1         ) \
    X   ( mjString*,     info,            1         )


//-------------------------------- mjsGeom ---------------------------------------------------------

#define MJSGEOM_FIELDS                              \
    X   ( mjsElement*,    element,         1      ) \
    X   ( mjtGeom,        type,            1      ) \
    XVEC( double,         pos,             3      ) \
    XVEC( double,         quat,            4      ) \
    X   ( mjsOrientation, alt,             1      ) \
    XVEC( double,         fromto,          6      ) \
    XVEC( double,         size,            3      ) \
    X   ( int,            contype,         1      ) \
    X   ( int,            conaffinity,     1      ) \
    X   ( int,            condim,          1      ) \
    X   ( int,            priority,        1      ) \
    XVEC( double,         friction,        3      ) \
    X   ( double,         solmix,          1      ) \
    XVEC( mjtNum,         solref,          mjNREF ) \
    XVEC( mjtNum,         solimp,          mjNIMP ) \
    X   ( double,         margin,          1      ) \
    X   ( double,         gap,             1      ) \
    XVEC( double,         surfacevel,      6      ) \
    X   ( double,         adhesion,        1      ) \
    X   ( double,         mass,            1      ) \
    X   ( double,         density,         1      ) \
    X   ( mjtGeomInertia, typeinertia,     1      ) \
    X   ( mjtNum,         fluid_ellipsoid, 1      ) \
    XVEC( mjtNum,         fluid_coefs,     5      ) \
    X   ( mjString*,      material,        1      ) \
    XVEC( float,          rgba,            4      ) \
    X   ( int,            group,           1      ) \
    X   ( mjString*,      hfieldname,      1      ) \
    X   ( mjString*,      meshname,        1      ) \
    X   ( double,         fitscale,        1      ) \
    X   ( mjDoubleVec*,   userdata,        1      ) \
    X   ( mjsPlugin,      plugin,          1      ) \
    X   ( mjString*,      info,            1      )


//-------------------------------- mjsSite ---------------------------------------------------------

#define MJSSITE_FIELDS                  \
    X   ( mjsElement*,    element,  1 ) \
    XVEC( double,         pos,      3 ) \
    XVEC( double,         quat,     4 ) \
    X   ( mjsOrientation, alt,      1 ) \
    XVEC( double,         fromto,   6 ) \
    XVEC( double,         size,     3 ) \
    X   ( mjtGeom,        type,     1 ) \
    X   ( mjString*,      material, 1 ) \
    X   ( int,            group,    1 ) \
    XVEC( float,          rgba,     4 ) \
    X   ( mjString*,      meshname, 1 ) \
    X   ( mjDoubleVec*,   userdata, 1 ) \
    X   ( mjString*,      info,     1 )


//-------------------------------- mjsCamera -------------------------------------------------------

#define MJSCAMERA_FIELDS                        \
    X   ( mjsElement*,    element,          1 ) \
    XVEC( double,         pos,              3 ) \
    XVEC( double,         quat,             4 ) \
    X   ( mjsOrientation, alt,              1 ) \
    X   ( mjtCamLight,    mode,             1 ) \
    X   ( mjString*,      targetbody,       1 ) \
    X   ( mjtProjection,  proj,             1 ) \
    XVEC( int,            resolution,       2 ) \
    X   ( int,            output,           1 ) \
    X   ( double,         fovy,             1 ) \
    X   ( double,         ipd,              1 ) \
    XVEC( float,          intrinsic,        4 ) \
    XVEC( float,          sensor_size,      2 ) \
    XVEC( float,          focal_length,     2 ) \
    XVEC( float,          focal_pixel,      2 ) \
    XVEC( float,          principal_length, 2 ) \
    XVEC( float,          principal_pixel,  2 ) \
    X   ( mjDoubleVec*,   userdata,         1 ) \
    X   ( mjString*,      info,             1 )


//-------------------------------- mjsLight --------------------------------------------------------

#define MJSLIGHT_FIELDS                  \
    X   ( mjsElement*,  element,     1 ) \
    XVEC( double,       pos,         3 ) \
    XVEC( double,       dir,         3 ) \
    X   ( mjtCamLight,  mode,        1 ) \
    X   ( mjString*,    targetbody,  1 ) \
    X   ( mjtBool,      active,      1 ) \
    X   ( mjtLightType, type,        1 ) \
    X   ( mjString*,    texture,     1 ) \
    X   ( mjtBool,      castshadow,  1 ) \
    X   ( float,        bulbradius,  1 ) \
    X   ( float,        intensity,   1 ) \
    X   ( float,        range,       1 ) \
    XVEC( float,        attenuation, 3 ) \
    X   ( float,        cutoff,      1 ) \
    X   ( float,        softness,    1 ) \
    X   ( float,        exponent,    1 ) \
    XVEC( float,        ambient,     3 ) \
    XVEC( float,        diffuse,     3 ) \
    XVEC( float,        specular,    3 ) \
    X   ( mjString*,    info,        1 )


//-------------------------------- mjsFlex ---------------------------------------------------------

#define MJSFLEX_FIELDS                           \
    X   ( mjsElement*,   element,       1      ) \
    X   ( int,            contype,       1      ) \
    X   ( int,            conaffinity,   1      ) \
    X   ( int,            condim,        1      ) \
    X   ( int,            priority,      1      ) \
    XVEC( double,         friction,      3      ) \
    X   ( double,         solmix,        1      ) \
    XVEC( mjtNum,         solref,        mjNREF ) \
    XVEC( mjtNum,         solimp,        mjNIMP ) \
    X   ( double,         margin,        1      ) \
    X   ( double,         gap,           1      ) \
    X   ( int,            dim,           1      ) \
    X   ( double,         radius,        1      ) \
    XVEC( double,         size,          3      ) \
    X   ( mjtBool,        flatskin,      1      ) \
    X   ( mjtFlexSelf,    selfcollide,   1      ) \
    X   ( int,            passive,       1      ) \
    X   ( int,            activelayers,  1      ) \
    X   ( int,            group,         1      ) \
    X   ( double,         edgestiffness, 1      ) \
    X   ( double,         edgedamping,   1      ) \
    XVEC( float,          rgba,          4      ) \
    X   ( mjStringVec*,   material,      1      ) \
    X   ( double,         young,         1      ) \
    X   ( double,         poisson,       1      ) \
    X   ( double,         shear,         1      ) \
    X   ( double,         damping,       1      ) \
    X   ( double,         thickness,     1      ) \
    X   ( double,         stressyield,   1      ) \
    X   ( int,            elastic2d,     1      ) \
    X   ( int,            elastic3d,     1      ) \
    X   ( mjtFlexSection, section,       1      ) \
    XVEC( double,         sectionsize,   2      ) \
    XVEC( int,            cellcount,     3      ) \
    X   ( int,            order,         1      ) \
    X   ( mjStringVec*,   nodebody,      1      ) \
    X   ( mjStringVec*,   vertbody,      1      ) \
    X   ( mjDoubleVec*,   node,          1      ) \
    X   ( mjDoubleVec*,   vert,          1      ) \
    X   ( mjIntVec*,      elem,          1      ) \
    X   ( mjFloatVec*,    texcoord,      1      ) \
    X   ( mjIntVec*,      elemtexcoord,  1      ) \
    X   ( mjString*,      info,          1      )


//-------------------------------- mjsMesh ---------------------------------------------------------

#define MJSMESH_FIELDS                           \
    X   ( mjsElement*,     element,          1 ) \
    X   ( mjString*,       content_type,     1 ) \
    X   ( mjString*,       file,             1 ) \
    XVEC( double,          refpos,           3 ) \
    XVEC( double,          refquat,          4 ) \
    XVEC( double,          scale,            3 ) \
    X   ( mjtMeshInertia,  inertia,          1 ) \
    X   ( mjtBool,         smoothnormal,     1 ) \
    X   ( mjtBool,         needsdf,          1 ) \
    X   ( int,             maxhullvert,      1 ) \
    X   ( mjFloatVec*,     uservert,         1 ) \
    X   ( mjFloatVec*,     usernormal,       1 ) \
    X   ( mjFloatVec*,     usertexcoord,     1 ) \
    X   ( mjIntVec*,       userface,         1 ) \
    X   ( mjIntVec*,       userfacenormal,   1 ) \
    X   ( mjIntVec*,       userfacetexcoord, 1 ) \
    X   ( mjDoubleVec*,    usernode,         1 ) \
    X   ( mjIntVec*,       usertet,          1 ) \
    X   ( mjsPlugin,       plugin,           1 ) \
    X   ( mjString*,       material,         1 ) \
    X   ( int,             octree_maxdepth,  1 ) \
    X   ( mjString*,       info,             1 )


//-------------------------------- mjsHField -------------------------------------------------------

#define MJSHFIELD_FIELDS                  \
    X   ( mjsElement*,  element,      1 ) \
    X   ( mjString*,    content_type, 1 ) \
    X   ( mjString*,    file,         1 ) \
    XVEC( double,       size,         4 ) \
    X   ( int,          nrow,         1 ) \
    X   ( int,          ncol,         1 ) \
    X   ( mjFloatVec*,  userdata,     1 ) \
    X   ( mjString*,    info,         1 )


//-------------------------------- mjsSkin ---------------------------------------------------------

#define MJSSKIN_FIELDS                    \
    X   ( mjsElement*,    element,    1 ) \
    X   ( mjString*,      file,       1 ) \
    X   ( mjString*,      material,   1 ) \
    XVEC( float,          rgba,       4 ) \
    X   ( float,          inflate,    1 ) \
    X   ( int,            group,      1 ) \
    X   ( mjFloatVec*,    vert,       1 ) \
    X   ( mjFloatVec*,    texcoord,   1 ) \
    X   ( mjIntVec*,      face,       1 ) \
    X   ( mjStringVec*,   bodyname,   1 ) \
    X   ( mjFloatVec*,    bindpos,    1 ) \
    X   ( mjFloatVec*,    bindquat,   1 ) \
    X   ( mjIntVecVec*,   vertid,     1 ) \
    X   ( mjFloatVecVec*, vertweight, 1 ) \
    X   ( mjString*,      info,       1 )


//-------------------------------- mjsTexture ------------------------------------------------------

#define MJSTEXTURE_FIELDS                    \
    X   ( mjsElement*,    element,      1  ) \
    X   ( mjtTexture,     type,         1  ) \
    X   ( mjtColorSpace,  colorspace,   1  ) \
    X   ( mjtBuiltin,     builtin,      1  ) \
    X   ( mjtMark,        mark,         1  ) \
    XVEC( double,         rgb1,         3  ) \
    XVEC( double,         rgb2,         3  ) \
    XVEC( double,         markrgb,      3  ) \
    X   ( double,         random,       1  ) \
    X   ( int,            height,       1  ) \
    X   ( int,            width,        1  ) \
    X   ( int,            nchannel,     1  ) \
    X   ( mjString*,      content_type, 1  ) \
    X   ( mjString*,      file,         1  ) \
    XVEC( int,            gridsize,     2  ) \
    XVEC( char,           gridlayout,   12 ) \
    X   ( mjStringVec*,   cubefiles,    1  ) \
    X   ( mjByteVec*,     data,         1  ) \
    X   ( mjtBool,        hflip,        1  ) \
    X   ( mjtBool,        vflip,        1  ) \
    X   ( mjtEmbed,       embed,        1  ) \
    X   ( mjString*,      info,         1  )


//-------------------------------- mjsMaterial -----------------------------------------------------

#define MJSMATERIAL_FIELDS               \
    X   ( mjsElement*,  element,     1 ) \
    X   ( mjStringVec*, textures,    1 ) \
    X   ( mjtBool,      texuniform,  1 ) \
    XVEC( float,        texrepeat,   2 ) \
    X   ( float,        emission,    1 ) \
    X   ( float,        specular,    1 ) \
    X   ( float,        shininess,   1 ) \
    X   ( float,        reflectance, 1 ) \
    X   ( float,        metallic,    1 ) \
    X   ( float,        roughness,   1 ) \
    XVEC( float,        rgba,        4 ) \
    X   ( mjString*,    info,        1 )


//-------------------------------- mjsPair ---------------------------------------------------------

#define MJSPAIR_FIELDS                          \
    X   ( mjsElement*, element,        1      ) \
    X   ( mjString*,   geomname1,      1      ) \
    X   ( mjString*,   geomname2,      1      ) \
    X   ( int,         condim,         1      ) \
    XVEC( mjtNum,      solref,         mjNREF ) \
    XVEC( mjtNum,      solreffriction, mjNREF ) \
    XVEC( mjtNum,      solimp,         mjNIMP ) \
    X   ( double,      margin,         1      ) \
    X   ( double,      gap,            1      ) \
    X   ( double,      adhesion,       1      ) \
    XVEC( double,      friction,       5      ) \
    X   ( mjString*,   info,           1      )


//-------------------------------- mjsExclude ------------------------------------------------------

#define MJSEXCLUDE_FIELDS          \
    X( mjsElement*, element,   1 ) \
    X( mjString*,   bodyname1, 1 ) \
    X( mjString*,   bodyname2, 1 ) \
    X( mjString*,   info,      1 )


//-------------------------------- mjsEquality -----------------------------------------------------

#define MJSEQUALITY_FIELDS                  \
    X   ( mjsElement*, element, 1         ) \
    X   ( mjtEq,       type,    1         ) \
    XVEC( double,      data,    mjNEQDATA ) \
    X   ( mjtBool,     active,  1         ) \
    X   ( mjString*,   name1,   1         ) \
    X   ( mjString*,   name2,   1         ) \
    X   ( mjtObj,      objtype, 1         ) \
    XVEC( mjtNum,      solref,  mjNREF    ) \
    XVEC( mjtNum,      solimp,  mjNIMP    ) \
    X   ( mjString*,   info,    1         )


//-------------------------------- mjsTendon -------------------------------------------------------

#define MJSTENDON_FIELDS                              \
    X   ( mjsElement*,   element,         1         ) \
    XVEC( double,        stiffness,       mjNPOLY+1 ) \
    XVEC( double,        springlength,    2         ) \
    X   ( double,        springyield,     1         ) \
    XVEC( double,        damping,         mjNPOLY+1 ) \
    X   ( double,        frictionloss,    1         ) \
    XVEC( mjtNum,        solref_friction, mjNREF    ) \
    XVEC( mjtNum,        solimp_friction, mjNIMP    ) \
    X   ( double,        armature,        1         ) \
    X   ( mjtLimited,    limited,         1         ) \
    X   ( mjtLimited,    actfrclimited,   1         ) \
    XVEC( double,        range,           2         ) \
    XVEC( double,        actfrcrange,     2         ) \
    X   ( double,        margin,          1         ) \
    XVEC( mjtNum,        solref_limit,    mjNREF    ) \
    XVEC( mjtNum,        solimp_limit,    mjNIMP    ) \
    X   ( mjString*,     material,        1         ) \
    X   ( double,        width,           1         ) \
    XVEC( float,         rgba,            4         ) \
    X   ( int,           group,           1         ) \
    X   ( mjDoubleVec*,  userdata,        1         ) \
    X   ( mjString*,     info,            1         )


//-------------------------------- mjsWrap ---------------------------------------------------------

#define MJSWRAP_FIELDS           \
    X( mjsElement*, element, 1 ) \
    X( mjtWrap,     type,    1 ) \
    X( mjString*,   info,    1 )


//-------------------------------- mjsActuator -----------------------------------------------------

#define MJSACTUATOR_FIELDS                          \
    X   ( mjsElement*,   element,       1         ) \
    X   ( mjtActuator,   type,          1         ) \
    X   ( mjtGain,       gaintype,      1         ) \
    XVEC( double,        gainprm,       mjNGAIN   ) \
    X   ( mjtBias,       biastype,      1         ) \
    XVEC( double,        biasprm,       mjNBIAS   ) \
    X   ( mjtDyn,        dyntype,       1         ) \
    XVEC( double,        dynprm,        mjNDYN    ) \
    X   ( int,           actdim,        1         ) \
    X   ( int,           ctrlspec,      1         ) \
    XVEC( double,        velrange,      2         ) \
    XVEC( double,        ffrange,       2         ) \
    X   ( mjtBool,       actearly,      1         ) \
    X   ( mjtTrn,        trntype,       1         ) \
    XVEC( double,        gear,          6         ) \
    X   ( mjString*,     target,        1         ) \
    X   ( mjString*,     refsite,       1         ) \
    X   ( mjString*,     slidersite,    1         ) \
    X   ( double,        cranklength,   1         ) \
    XVEC( double,        lengthrange,   2         ) \
    X   ( double,        inheritrange,  1         ) \
    XVEC( double,        damping,       mjNPOLY+1 ) \
    X   ( double,        armature,      1         ) \
    X   ( mjtLimited,    ctrllimited,   1         ) \
    XVEC( double,        ctrlrange,     2         ) \
    X   ( mjtLimited,    forcelimited,  1         ) \
    XVEC( double,        forcerange,    2         ) \
    X   ( mjtLimited,    actlimited,    1         ) \
    XVEC( double,        actrange,      2         ) \
    X   ( int,           group,         1         ) \
    X   ( int,           nsample,       1         ) \
    X   ( int,           interp,        1         ) \
    X   ( double,        delay,         1         ) \
    X   ( mjDoubleVec*,  userdata,      1         ) \
    X   ( mjsPlugin,     plugin,        1         ) \
    X   ( mjString*,     info,          1         )


//-------------------------------- mjsSensor -------------------------------------------------------

#define MJSSENSOR_FIELDS                      \
    X   ( mjsElement*,   element,   1       ) \
    X   ( mjtSensor,     type,      1       ) \
    X   ( mjtObj,        objtype,   1       ) \
    X   ( mjString*,     objname,   1       ) \
    X   ( mjtObj,        reftype,   1       ) \
    X   ( mjString*,     refname,   1       ) \
    XVEC( int,           intprm,    mjNSENS ) \
    X   ( mjtDataType,   datatype,  1       ) \
    X   ( mjtStage,      needstage, 1       ) \
    X   ( int,           dim,       1       ) \
    X   ( double,        cutoff,    1       ) \
    X   ( double,        noise,     1       ) \
    X   ( int,           nsample,   1       ) \
    X   ( int,           interp,    1       ) \
    X   ( double,        delay,     1       ) \
    XVEC( double,        interval,  2       ) \
    X   ( mjDoubleVec*,  userdata,  1       ) \
    X   ( mjsPlugin,     plugin,    1       ) \
    X   ( mjString*,     info,      1       )


//-------------------------------- mjsNumeric ------------------------------------------------------

#define MJSNUMERIC_FIELDS         \
    X( mjsElement*,  element, 1 ) \
    X( mjDoubleVec*, data,    1 ) \
    X( int,          size,    1 ) \
    X( mjString*,    info,    1 )


//-------------------------------- mjsText ---------------------------------------------------------

#define MJSTEXT_FIELDS           \
    X( mjsElement*, element, 1 ) \
    X( mjString*,   data,    1 ) \
    X( mjString*,   info,    1 )


//-------------------------------- mjsTuple --------------------------------------------------------

#define MJSTUPLE_FIELDS           \
    X( mjsElement*,  element, 1 ) \
    X( mjIntVec*,    objtype, 1 ) \
    X( mjStringVec*, objname, 1 ) \
    X( mjDoubleVec*, objprm,  1 ) \
    X( mjString*,    info,    1 )


//-------------------------------- mjsKey ----------------------------------------------------------

#define MJSKEY_FIELDS             \
    X( mjsElement*,  element, 1 ) \
    X( double,       time,    1 ) \
    X( mjDoubleVec*, qpos,    1 ) \
    X( mjDoubleVec*, qvel,    1 ) \
    X( mjDoubleVec*, act,     1 ) \
    X( mjDoubleVec*, mpos,    1 ) \
    X( mjDoubleVec*, mquat,   1 ) \
    X( mjDoubleVec*, ctrl,    1 ) \
    X( mjDoubleVec*, plastic, 1 ) \
    X( mjString*,    info,    1 )


//-------------------------------- mjsDefault ------------------------------------------------------

#define MJSDEFAULT_FIELDS          \
    X( mjsElement*,  element,  1 ) \
    X( mjsJoint*,    joint,    1 ) \
    X( mjsGeom*,     geom,     1 ) \
    X( mjsSite*,     site,     1 ) \
    X( mjsCamera*,   camera,   1 ) \
    X( mjsLight*,    light,    1 ) \
    X( mjsFlex*,     flex,     1 ) \
    X( mjsMesh*,     mesh,     1 ) \
    X( mjsTexture*,  texture,  1 ) \
    X( mjsMaterial*, material, 1 ) \
    X( mjsPair*,     pair,     1 ) \
    X( mjsEquality*, equality, 1 ) \
    X( mjsTendon*,   tendon,   1 ) \
    X( mjsActuator*, actuator, 1 )


#endif  // MUJOCO_MJSPECMACRO_H_
