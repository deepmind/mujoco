// Copyright 2022 DeepMind Technologies Limited
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

#ifndef MUJOCO_PYTHON_INDEXER_XMACRO_H_
#define MUJOCO_PYTHON_INDEXER_XMACRO_H_

#include <mujoco/mjxmacro.h>  // IWYU pragma: keep

#define MJMODEL_ACTUATOR                                                 \
  X( int,     actuator_, trntype,      nactuator, 1                    ) \
  X( int,     actuator_, dyntype,      nactuator, 1                    ) \
  X( int,     actuator_, gaintype,     nactuator, 1                    ) \
  X( int,     actuator_, biastype,     nactuator, 1                    ) \
  X( int,     actuator_, trnid,        nactuator, 2                    ) \
  X( int,     actuator_, actadr,       nactuator, 1                    ) \
  X( int,     actuator_, actnum,       nactuator, 1                    ) \
  X( int,     actuator_, group,        nactuator, 1                    ) \
  X( mjtByte, actuator_, ctrllimited,  nu,        1                    ) \
  X( mjtByte, actuator_, forcelimited, nactuator, 1                    ) \
  X( mjtByte, actuator_, actlimited,   nactuator, 1                    ) \
  X( mjtNum,  actuator_, dynprm,       nactuator, mjNDYN               ) \
  X( mjtNum,  actuator_, gainprm,      nactuator, mjNGAIN              ) \
  X( mjtNum,  actuator_, biasprm,      nactuator, mjNBIAS              ) \
  X( mjtNum,  actuator_, ctrlrange,    nu,        2                    ) \
  X( mjtNum,  actuator_, forcerange,   nactuator, 2                    ) \
  X( mjtNum,  actuator_, actrange,     nactuator, 2                    ) \
  X( mjtNum,  actuator_, gear,         nout,      6                    ) \
  X( mjtNum,  actuator_, cranklength,  nactuator, 1                    ) \
  X( mjtNum,  actuator_, acc0,         nout,      1                    ) \
  X( mjtNum,  actuator_, length0,      nout,      1                    ) \
  X( mjtNum,  actuator_, lengthrange,  nout,      2                    ) \
  X( mjtNum,  actuator_, user,         nactuator, MJ_M(nuser_actuator) )

#define MJMODEL_BODY                                        \
  X( int,     body_, parentid,    nbody, 1                ) \
  X( int,     body_, rootid,      nbody, 1                ) \
  X( int,     body_, weldid,      nbody, 1                ) \
  X( int,     body_, mocapid,     nbody, 1                ) \
  X( int,     body_, jntnum,      nbody, 1                ) \
  X( int,     body_, jntadr,      nbody, 1                ) \
  X( int,     body_, dofnum,      nbody, 1                ) \
  X( int,     body_, dofadr,      nbody, 1                ) \
  X( int,     body_, geomnum,     nbody, 1                ) \
  X( int,     body_, geomadr,     nbody, 1                ) \
  X( mjtByte, body_, simple,      nbody, 1                ) \
  X( mjtByte, body_, sameframe,   nbody, 1                ) \
  X( mjtNum,  body_, pos,         nbody, 3                ) \
  X( mjtNum,  body_, quat,        nbody, 4                ) \
  X( mjtNum,  body_, ipos,        nbody, 3                ) \
  X( mjtNum,  body_, iquat,       nbody, 4                ) \
  X( mjtNum,  body_, mass,        nbody, 1                ) \
  X( mjtNum,  body_, subtreemass, nbody, 1                ) \
  X( mjtNum,  body_, inertia,     nbody, 3                ) \
  X( mjtNum,  body_, invweight0,  nbody, 2                ) \
  X( mjtNum,  body_, user,        nbody, MJ_M(nuser_body) )

#define MJMODEL_CAMERA                                   \
  X( int,    cam_, mode,         ncam, 1               ) \
  X( int,    cam_, bodyid,       ncam, 1               ) \
  X( int,    cam_, targetbodyid, ncam, 1               ) \
  X( mjtNum, cam_, pos,          ncam, 3               ) \
  X( mjtNum, cam_, quat,         ncam, 4               ) \
  X( mjtNum, cam_, poscom0,      ncam, 3               ) \
  X( mjtNum, cam_, pos0,         ncam, 3               ) \
  X( mjtNum, cam_, mat0,         ncam, 9               ) \
  X( mjtNum, cam_, fovy,         ncam, 1               ) \
  X( mjtNum, cam_, ipd,          ncam, 1               ) \
  X( mjtNum, cam_, user,         ncam, MJ_M(nuser_cam) )

#define MJMODEL_EQUALITY                     \
  X( int,     eq_, type,    neq, 1         ) \
  X( int,     eq_, obj1id,  neq, 1         ) \
  X( int,     eq_, obj2id,  neq, 1         ) \
  X( mjtByte, eq_, active0, neq, 1         ) \
  X( mjtNum,  eq_, solref,  neq, mjNREF    ) \
  X( mjtNum,  eq_, solimp,  neq, mjNIMP    ) \
  X( mjtNum,  eq_, data,    neq, mjNEQDATA )

#define MJMODEL_EXCLUDE                      \
  X( int, exclude_, signature, nexclude, 1 )

#define MJMODEL_GEOM                                        \
  X( int,     geom_, type,        ngeom, 1                ) \
  X( int,     geom_, contype,     ngeom, 1                ) \
  X( int,     geom_, conaffinity, ngeom, 1                ) \
  X( int,     geom_, condim,      ngeom, 1                ) \
  X( int,     geom_, bodyid,      ngeom, 1                ) \
  X( int,     geom_, dataid,      ngeom, 1                ) \
  X( int,     geom_, matid,       ngeom, 1                ) \
  X( int,     geom_, group,       ngeom, 1                ) \
  X( int,     geom_, priority,    ngeom, 1                ) \
  X( mjtByte, geom_, sameframe,   ngeom, 1                ) \
  X( mjtByte, geom_, castshadow,  ngeom, 1                ) \
  X( mjtNum,  geom_, solmix,      ngeom, 1                ) \
  X( mjtNum,  geom_, solref,      ngeom, mjNREF           ) \
  X( mjtNum,  geom_, solimp,      ngeom, mjNIMP           ) \
  X( mjtNum,  geom_, size,        ngeom, 3                ) \
  X( mjtNum,  geom_, rbound,      ngeom, 1                ) \
  X( mjtNum,  geom_, pos,         ngeom, 3                ) \
  X( mjtNum,  geom_, quat,        ngeom, 4                ) \
  X( mjtNum,  geom_, friction,    ngeom, 3                ) \
  X( mjtNum,  geom_, margin,      ngeom, 1                ) \
  X( mjtNum,  geom_, gap,         ngeom, 1                ) \
  X( mjtNum,  geom_, user,        ngeom, MJ_M(nuser_geom) ) \
  X( float,   geom_, rgba,        ngeom, 4                )

#define MJMODEL_HFIELD                          \
  X( mjtNum, hfield_, size,    nhfield,     4 ) \
  X( int,    hfield_, nrow,    nhfield,     1 ) \
  X( int,    hfield_, ncol,    nhfield,     1 ) \
  X( int,    hfield_, adr,     nhfield,     1 ) \
  X( float,  hfield_, data,    nhfielddata, 1 ) \
  X( int,    hfield_, pathadr, nhfield,     1 )

#define MJMODEL_JOINT                                     \
  X( mjtNum,  ,     qpos0,        nq,   1               ) \
  X( mjtNum,  ,     qpos_spring,  nq,   1               ) \
  X( int,     jnt_, type,         njnt, 1               ) \
  X( int,     jnt_, qposadr,      njnt, 1               ) \
  X( int,     jnt_, dofadr,       njnt, 1               ) \
  X( int,     jnt_, group,        njnt, 1               ) \
  X( int,     jnt_, plasticadr,   njnt, 1               ) \
  X( mjtByte, jnt_, limited,      njnt, 1               ) \
  X( mjtNum,  jnt_, pos,          njnt, 3               ) \
  X( mjtNum,  jnt_, axis,         njnt, 3               ) \
  X( mjtNum,  jnt_, stiffness,    njnt, 1               ) \
  X( mjtNum,  jnt_, stiffnesspoly, njnt, mjNPOLY        ) \
  X( mjtNum,  jnt_, springyield,  njnt, 1               ) \
  X( mjtNum,  jnt_, range,        njnt, 2               ) \
  X( mjtNum,  jnt_, margin,       njnt, 1               ) \
  X( mjtNum,  jnt_, user,         njnt, MJ_M(nuser_jnt) ) \
  X( int,     dof_, bodyid,       nv,   1               ) \
  X( int,     dof_, jntid,        nv,   1               ) \
  X( int,     dof_, parentid,     nv,   1               ) \
  X( int,     dof_, Madr,         nv,   1               ) \
  X( int,     dof_, simplenum,    nv,   1               ) \
  X( mjtNum,  jnt_, solref,       njnt, mjNREF          ) \
  X( mjtNum,  jnt_, solimp,       njnt, mjNIMP          ) \
  X( mjtNum,  dof_, frictionloss, nv,   1               ) \
  X( mjtNum,  dof_, armature,     nv,   1               ) \
  X( mjtNum,  dof_, damping,      nv,   1               ) \
  X( mjtNum,  dof_, dampingpoly,  nv,   mjNPOLY         ) \
  X( mjtNum,  dof_, invweight0,   nv,   1               ) \
  X( mjtNum,  dof_, M0,           nv,   1               )

#define MJMODEL_LIGHT                             \
  X( int,     light_, mode,          nlight,  1 ) \
  X( int,     light_, bodyid,        nlight,  1 ) \
  X( int,     light_, targetbodyid,  nlight,  1 ) \
  X( int,     light_, type,          nlight,  1 ) \
  X( mjtByte, light_, castshadow,    nlight,  1 ) \
  X( mjtByte, light_, active,        nlight,  1 ) \
  X( mjtNum,  light_, pos,           nlight,  3 ) \
  X( mjtNum,  light_, dir,           nlight,  3 ) \
  X( mjtNum,  light_, poscom0,       nlight,  3 ) \
  X( mjtNum,  light_, pos0,          nlight,  3 ) \
  X( mjtNum,  light_, dir0,          nlight,  3 ) \
  X( float,   light_, attenuation,   nlight,  3 ) \
  X( float,   light_, cutoff,        nlight,  1 ) \
  X( float,   light_, exponent,      nlight,  1 ) \
  X( float,   light_, ambient,       nlight,  3 ) \
  X( float,   light_, diffuse,       nlight,  3 ) \
  X( float,   light_, specular,      nlight,  3 )

#define MJMODEL_MATERIAL                   \
  X( int,     mat_, texid,       nmat, mjNTEXROLE ) \
  X( mjtByte, mat_, texuniform,  nmat, 1 ) \
  X( float,   mat_, texrepeat,   nmat, 2 ) \
  X( float,   mat_, emission,    nmat, 1 ) \
  X( float,   mat_, specular,    nmat, 1 ) \
  X( float,   mat_, shininess,   nmat, 1 ) \
  X( float,   mat_, reflectance, nmat, 1 ) \
  X( float,   mat_, rgba,        nmat, 4 )

#define MJMODEL_MESH                   \
  X( int, mesh_, vertadr,     nmesh, 1 ) \
  X( int, mesh_, vertnum,     nmesh, 1 ) \
  X( int, mesh_, texcoordadr, nmesh, 1 ) \
  X( int, mesh_, faceadr,     nmesh, 1 ) \
  X( int, mesh_, facenum,     nmesh, 1 ) \
  X( int, mesh_, graphadr,    nmesh, 1 ) \
  X( int, mesh_, pathadr,     nmesh, 1 )

#define MJMODEL_NUMERIC                        \
  X( int,    numeric_, adr,  nnumeric,     1 ) \
  X( int,    numeric_, size, nnumeric,     1 ) \
  X( mjtNum, numeric_, data, nnumericdata, 1 )

#define MJMODEL_PAIR                            \
  X( int,     pair_, dim,       npair, 1      ) \
  X( int,     pair_, geom1,     npair, 1      ) \
  X( int,     pair_, geom2,     npair, 1      ) \
  X( int,     pair_, signature, npair, 1      ) \
  X( mjtNum,  pair_, solref,    npair, mjNREF ) \
  X( mjtNum,  pair_, solimp,    npair, mjNIMP ) \
  X( mjtNum,  pair_, margin,    npair, 1      ) \
  X( mjtNum,  pair_, gap,       npair, 1      ) \
  X( mjtNum,  pair_, friction,  npair, 5      )

#define MJMODEL_SENSOR                                         \
  X( int,    sensor_, type,      nsensor, 1                  ) \
  X( int,    sensor_, datatype,  nsensor, 1                  ) \
  X( int,    sensor_, needstage, nsensor, 1                  ) \
  X( int,    sensor_, objtype,   nsensor, 1                  ) \
  X( int,    sensor_, objid,     nsensor, 1                  ) \
  X( int,    sensor_, reftype,   nsensor, 1                  ) \
  X( int,    sensor_, refid,     nsensor, 1                  ) \
  X( int,    sensor_, dim,       nsensor, 1                  ) \
  X( int,    sensor_, adr,       nsensor, 1                  ) \
  X( mjtNum, sensor_, cutoff,    nsensor, 1                  ) \
  X( mjtNum, sensor_, noise,     nsensor, 1                  ) \
  X( mjtNum, sensor_, user,      nsensor, MJ_M(nuser_sensor) )

#define MJMODEL_SITE                                      \
  X( int,     site_, type,      nsite, 1                ) \
  X( int,     site_, bodyid,    nsite, 1                ) \
  X( int,     site_, matid,     nsite, 1                ) \
  X( int,     site_, group,     nsite, 1                ) \
  X( mjtByte, site_, sameframe, nsite, 1                ) \
  X( mjtNum,  site_, size,      nsite, 3                ) \
  X( mjtNum,  site_, pos,       nsite, 3                ) \
  X( mjtNum,  site_, quat,      nsite, 4                ) \
  X( mjtNum,  site_, user,      nsite, MJ_M(nuser_site) ) \
  X( float,   site_, rgba,      nsite, 4                )

#define MJMODEL_SKIN                       \
  X( int,   skin_, matid,       nskin, 1 ) \
  X( float, skin_, rgba,        nskin, 4 ) \
  X( float, skin_, inflate,     nskin, 1 ) \
  X( int,   skin_, vertadr,     nskin, 1 ) \
  X( int,   skin_, vertnum,     nskin, 1 ) \
  X( int,   skin_, texcoordadr, nskin, 1 ) \
  X( int,   skin_, faceadr,     nskin, 1 ) \
  X( int,   skin_, facenum,     nskin, 1 ) \
  X( int,   skin_, boneadr,     nskin, 1 ) \
  X( int,   skin_, bonenum,     nskin, 1 ) \
  X( int,   skin_, pathadr,     nskin, 1 )

#define MJMODEL_TENDON                                             \
  X( int,     tendon, _type,         ntendon, 1                  ) \
  X( int,     tendon, _adr,          ntendon, 1                  ) \
  X( int,     tendon, _num,          ntendon, 1                  ) \
  X( int,     tendon, _matid,        ntendon, 1                  ) \
  X( int,     tendon, _group,        ntendon, 1                  ) \
  X( int,     tendon_, plasticadr,   ntendon, 1                  ) \
  X( mjtByte, tendon, _limited,      ntendon, 1                  ) \
  X( mjtNum,  tendon, _width,        ntendon, 1                  ) \
  X( mjtNum,  tendon, _solref_lim,   ntendon, mjNREF             ) \
  X( mjtNum,  tendon, _solimp_lim,   ntendon, mjNIMP             ) \
  X( mjtNum,  tendon, _solref_fri,   ntendon, mjNREF             ) \
  X( mjtNum,  tendon, _solimp_fri,   ntendon, mjNIMP             ) \
  X( mjtNum,  tendon, _range,        ntendon, 2                  ) \
  X( mjtNum,  tendon, _margin,       ntendon, 1                  ) \
  X( mjtNum,  tendon, _stiffness,    ntendon, 1                  ) \
  X( mjtNum,  tendon, _stiffnesspoly, ntendon, mjNPOLY           ) \
  X( mjtNum,  tendon, _damping,      ntendon, 1                  ) \
  X( mjtNum,  tendon, _dampingpoly,  ntendon, mjNPOLY            ) \
  X( mjtNum,  tendon, _frictionloss, ntendon, 1                  ) \
  X( mjtNum,  tendon, _lengthspring, ntendon, 1                  ) \
  X( mjtNum,  tendon_, springyield,  ntendon, 1                  ) \
  X( mjtNum,  tendon, _length0,      ntendon, 1                  ) \
  X( mjtNum,  tendon, _invweight0,   ntendon, 1                  ) \
  X( mjtNum,  tendon, _user,         ntendon, MJ_M(nuser_tendon) ) \
  X( float,   tendon, _rgba,         ntendon, 4                  ) \
  X( int,     ten_,   J_rownnz,      ntendon, 1                  ) \
  X( int,     ten_,   J_rowadr,      ntendon, 1                  ) \
  X( int,     ten_,   J_colind,      ntendon, MJ_M(nv)           )

#define MJMODEL_TEXTURE                         \
  X( int,     tex_, type,           ntex,     1 ) \
  X( int,     tex_, colorspace,     ntex,     1 ) \
  X( int,     tex_, height,         ntex,     1 ) \
  X( int,     tex_, width,          ntex,     1 ) \
  X( int,     tex_, nchannel,       ntex,     1 ) \
  X( mjtSize, tex_, adr,            ntex,     1 ) \
  X( mjtByte, tex_, data,           ntexdata, 1 ) \
  X( int,     tex_, pathadr,        ntex,     1 ) \
  X( int,     tex_, contenttypeadr, ntex,     1 )

#define MJMODEL_TUPLE                         \
  X( int,    tuple_, adr,     ntuple,     1 ) \
  X( int,    tuple_, size,    ntuple,     1 ) \
  X( int,    tuple_, objtype, ntupledata, 1 ) \
  X( int,    tuple_, objid,   ntupledata, 1 ) \
  X( mjtNum, tuple_, objprm,  ntupledata, 1 )

#define MJMODEL_KEYFRAME                         \
  X( mjtNum, key_, time,  nkey, 1              ) \
  X( mjtNum, key_, qpos,  nkey, MJ_M(nq)       ) \
  X( mjtNum, key_, qvel,  nkey, MJ_M(nv)       ) \
  X( mjtNum, key_, act,   nkey, MJ_M(na)       ) \
  X( mjtNum, key_, mpos,  nkey, MJ_M(nmocap)*3 ) \
  X( mjtNum, key_, mquat,   nkey, MJ_M(nmocap)*4 ) \
  X( mjtNum, key_, ctrl,    nkey, MJ_M(nu)       ) \
  X( mjtNum, key_, plastic, nkey, MJ_M(nplastic) )

#define MJMODEL_VIEW_GROUPS                                             \
  XGROUP( MjModelActuatorViews, actuator, nactuator, MJMODEL_ACTUATOR ) \
  XGROUP( MjModelBodyViews,     body,     nbody,     MJMODEL_BODY     ) \
  XGROUP( MjModelCameraViews,   cam,      ncam,      MJMODEL_CAMERA   ) \
  XGROUP( MjModelEqualityViews, eq,       neq,       MJMODEL_EQUALITY ) \
  XGROUP( MjModelExcludeViews,  exclude,  nexclude,  MJMODEL_EXCLUDE  ) \
  XGROUP( MjModelGeomViews,     geom,     ngeom,     MJMODEL_GEOM     ) \
  XGROUP( MjModelHfieldViews,   hfield,   nhfield,   MJMODEL_HFIELD   ) \
  XGROUP( MjModelJointViews,    jnt,      njnt,      MJMODEL_JOINT    ) \
  XGROUP( MjModelLightViews,    light,    nlight,    MJMODEL_LIGHT    ) \
  XGROUP( MjModelMaterialViews, mat,      nmat,      MJMODEL_MATERIAL ) \
  XGROUP( MjModelMeshViews,     mesh,     nmesh,     MJMODEL_MESH     ) \
  XGROUP( MjModelNumericViews,  numeric,  nnumeric,  MJMODEL_NUMERIC  ) \
  XGROUP( MjModelPairViews,     pair,     npair,     MJMODEL_PAIR     ) \
  XGROUP( MjModelSensorViews,   sensor,   nsensor,   MJMODEL_SENSOR   ) \
  XGROUP( MjModelSiteViews,     site,     nsite,     MJMODEL_SITE     ) \
  XGROUP( MjModelSkinViews,     skin,     nskin,     MJMODEL_SKIN     ) \
  XGROUP( MjModelTendonViews,   tendon,   ntendon,   MJMODEL_TENDON   ) \
  XGROUP( MjModelTextureViews,  tex,      ntex,      MJMODEL_TEXTURE  ) \
  XGROUP( MjModelTupleViews,    tuple,    ntuple,    MJMODEL_TUPLE    ) \
  XGROUP( MjModelKeyframeViews, key,      nkey,      MJMODEL_KEYFRAME )

#define MJMODEL_BIND_GROUPS      \
  XGROUP( mjsActuator, actuator) \
  XGROUP( mjsBody,     body    ) \
  XGROUP( mjsCamera,   cam     ) \
  XGROUP( mjsEquality, eq      ) \
  XGROUP( mjsExclude,  exclude ) \
  XGROUP( mjsGeom,     geom    ) \
  XGROUP( mjsHField,   hfield  ) \
  XGROUP( mjsJoint,    jnt     ) \
  XGROUP( mjsLight,    light   ) \
  XGROUP( mjsMaterial, mat     ) \
  XGROUP( mjsMesh,     mesh    ) \
  XGROUP( mjsNumeric,  numeric ) \
  XGROUP( mjsPair,     pair    ) \
  XGROUP( mjsSensor,   sensor  ) \
  XGROUP( mjsSite,     site    ) \
  XGROUP( mjsSkin,     skin    ) \
  XGROUP( mjsTendon,   tendon  ) \
  XGROUP( mjsTexture,  tex     ) \
  XGROUP( mjsTuple,    tuple   ) \
  XGROUP( mjsKey,      key     )

#define MJMODEL_VIEW_GROUPS_ALTNAMES        \
  XGROUP( cam, camera,   MJMODEL_CAMERA   ) \
  XGROUP( eq,  equality, MJMODEL_EQUALITY ) \
  XGROUP( jnt, joint,    MJMODEL_JOINT    ) \
  XGROUP( mat, material, MJMODEL_MATERIAL ) \
  XGROUP( tex, texture,  MJMODEL_TEXTURE  ) \
  XGROUP( key, keyframe, MJMODEL_KEYFRAME )

// actuator_moment is sparse: the moment view is computed, in structs.cc
#define MJDATA_ACTUATOR                            \
  X( mjtNum, ,          ctrl,     nu,   1        ) \
  X( mjtNum, actuator_, length,   nout, 1        ) \
  X( mjtNum, actuator_, velocity, nout, 1        ) \
  X( mjtNum, actuator_, force,    nout, 1        )

#define MJDATA_BODY                        \
  X( mjtNum, , xfrc_applied,   nbody, 6  ) \
  X( mjtNum, , xpos,           nbody, 3  ) \
  X( mjtNum, , xquat,          nbody, 4  ) \
  X( mjtNum, , xmat,           nbody, 9  ) \
  X( mjtNum, , xipos,          nbody, 3  ) \
  X( mjtNum, , xiquat,         nbody, 4  ) \
  X( mjtNum, , ximat,          nbody, 9  ) \
  X( mjtNum, , subtree_com,    nbody, 3  ) \
  X( mjtNum, , cinert,         nbody, 10 ) \
  X( mjtNum, , crb,            nbody, 10 ) \
  X( mjtNum, , cvel,           nbody, 6  ) \
  X( mjtNum, , subtree_linvel, nbody, 3  ) \
  X( mjtNum, , subtree_angmom, nbody, 3  ) \
  X( mjtNum, , cacc,           nbody, 6  ) \
  X( mjtNum, , cfrc_int,       nbody, 6  ) \
  X( mjtNum, , cfrc_ext,       nbody, 6  )

#define MJDATA_CAMERA              \
  X( mjtNum, cam_, xpos, ncam, 3 ) \
  X( mjtNum, cam_, xmat, ncam, 9 )

#define MJDATA_GEOM                   \
  X( mjtNum, geom_, xpos,  ngeom, 3 ) \
  X( mjtNum, geom_, xquat, ngeom, 4 ) \
  X( mjtNum, geom_, xmat,  ngeom, 9 )

#define MJDATA_JOINT                                \
  X( mjtNum, , qpos,            nq,   1           ) \
  X( mjtNum, , qvel,            nv,   1           ) \
  X( mjtNum, , qacc_warmstart,  nv,   1           ) \
  X( mjtNum, , qfrc_applied,    nv,   1           ) \
  X( mjtNum, , qacc,            nv,   1           ) \
  X( mjtNum, , xanchor,         njnt, 3           ) \
  X( mjtNum, , xaxis,           njnt, 3           ) \
  X( mjtNum, , cdof,            nv,   6           ) \
  X( mjtNum, , qLDiagInv,       nv,   1           ) \
  X( mjtNum, , cdof_dot,        nv,   6           ) \
  X( mjtNum, , qfrc_bias,       nv,   1           ) \
  X( mjtNum, , qfrc_passive,    nv,   1           ) \
  X( mjtNum, , qfrc_actuator,   nv,   1           ) \
  X( mjtNum, , qfrc_smooth,     nv,   1           ) \
  X( mjtNum, , qacc_smooth,     nv,   1           ) \
  X( mjtNum, , qfrc_constraint, nv,       1           ) \
  X( mjtNum, , qfrc_inverse,    nv,       1           ) \
  X( mjtNum, , plastic,         nplastic, 1           )

#define MJDATA_LIGHT                   \
  X( mjtNum, light_, xpos, nlight, 3 ) \
  X( mjtNum, light_, xdir, nlight, 3 )

#define MJDATA_SENSOR                       \
  X( mjtNum, sensor, data, nsensordata, 1 )

#define MJDATA_SITE                   \
  X( mjtNum, site_, xpos,  nsite, 3 ) \
  X( mjtNum, site_, xquat, nsite, 4 ) \
  X( mjtNum, site_, xmat,  nsite, 9 )

#define MJDATA_TENDON                            \
  X( int, ten_,    wrapadr , ntendon,  1       ) \
  X( int, ten_,    wrapnum , ntendon,  1       ) \
  X( mjtNum, ten_, length  , ntendon,  1       ) \
  X( mjtNum, ten_, velocity, ntendon,  1       ) \
  X( mjtNum, ,     plastic , nplastic, 2       )

#define MJDATA_VIEW_GROUPS                                            \
  XGROUP( MjDataActuatorViews, actuator, nactuator, MJDATA_ACTUATOR ) \
  XGROUP( MjDataBodyViews,     body,     nbody,     MJDATA_BODY     ) \
  XGROUP( MjDataCameraViews,   cam,      ncam,      MJDATA_CAMERA   ) \
  XGROUP( MjDataGeomViews,     geom,     ngeom,     MJDATA_GEOM     ) \
  XGROUP( MjDataJointViews,    jnt,      njnt,      MJDATA_JOINT    ) \
  XGROUP( MjDataLightViews,    light,    nlight,    MJDATA_LIGHT    ) \
  XGROUP( MjDataSensorViews,   sensor,   nsensor,   MJDATA_SENSOR   ) \
  XGROUP( MjDataSiteViews,     site,     nsite,     MJDATA_SITE     ) \
  XGROUP( MjDataTendonViews,   tendon,   ntendon,   MJDATA_TENDON   )

#define MJDATA_BIND_GROUPS       \
  XGROUP( mjsActuator, actuator) \
  XGROUP( mjsBody,     body    ) \
  XGROUP( mjsCamera,   cam     ) \
  XGROUP( mjsGeom,     geom    ) \
  XGROUP( mjsJoint,    jnt     ) \
  XGROUP( mjsLight,    light   ) \
  XGROUP( mjsSensor,   sensor  ) \
  XGROUP( mjsSite,     site    ) \
  XGROUP( mjsTendon,   tendon  )

#define MJDATA_VIEW_GROUPS_ALTNAMES       \
  XGROUP( cam,    camera, MJDATA_CAMERA ) \
  XGROUP( jnt,    joint,  MJDATA_JOINT  ) \
  XGROUP( tendon, ten,    MJDATA_TENDON )

#endif  // MUJOCO_PYTHON_INDEXER_XMACRO_H_
