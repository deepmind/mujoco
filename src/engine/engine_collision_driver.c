// Copyright 2021 DeepMind Technologies Limited
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

#include "engine/engine_collision_driver.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_callback.h"
#include "engine/engine_collision_continuous.h"
#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_flex.h"
#include "engine/engine_collision_gjk.h"
#include "engine/engine_collision_primitive.h"
#include "engine/engine_collision_sdf.h"
#include "engine/engine_thread.h"
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_util.h"
#include "engine/engine_inline.h"
#include "engine/engine_macro.h"
#include "engine/engine_memory.h"
#include "engine/engine_sleep.h"
#include "engine/engine_sort.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_spatial.h"


// table of pairwise collision functions
mjfCollision mjCOLLISIONFUNC[mjNGEOMTYPES][mjNGEOMTYPES] = {
  /*              PLANE  HFIELD  SPHERE            CAPSULE             ELLIPSOID         CYLINDER            BOX               MESH              SDF */
  /*PLANE     */ {0,     0,      mjc_PlaneSphere,  mjc_PlaneCapsule,   mjc_PlaneConvex,  mjc_PlaneCylinder,  mjc_PlaneBox,     mjc_PlaneConvex,  mjc_PlaneConvex},
  /*HFIELD    */ {0,     0,      mjc_ConvexHField, mjc_ConvexHField,   mjc_ConvexHField, mjc_ConvexHField,   mjc_ConvexHField, mjc_ConvexHField, mjc_HFieldSDF},
  /*SPHERE    */ {0,     0,      mjc_SphereSphere, mjc_SphereCapsule,  mjc_Convex,       mjc_SphereCylinder, mjc_SphereBox,    mjc_Convex,       mjc_SDF},
  /*CAPSULE   */ {0,     0,      0,                mjc_CapsuleCapsule, mjc_Convex,       mjc_Convex,         mjc_CapsuleBox,   mjc_Convex,       mjc_SDF},
  /*ELLIPSOID */ {0,     0,      0,                0,                  mjc_Convex,       mjc_Convex,         mjc_Convex,       mjc_Convex,       mjc_SDF},
  /*CYLINDER  */ {0,     0,      0,                0,                  0,                mjc_Convex,         mjc_Convex,       mjc_Convex,       mjc_SDF},
  /*BOX       */ {0,     0,      0,                0,                  0,                0,                  mjc_BoxBox,       mjc_Convex,       mjc_SDF},
  /*MESH      */ {0,     0,      0,                0,                  0,                0,                  0,                mjc_Convex,       mjc_MeshSDF},
  /*SDF       */ {0,     0,      0,                0,                  0,                0,                  0,                0,                mjc_SDF}
};



//------------------------------------ utility functions ------------------------------------------


// return the maximum number of contacts that can be generated between two geoms
// has_margin is ignored
int mj_maxContact(const mjModel* m, int g1, int g2, int has_margin) {
  int type1 = m->geom_type[g1];
  int type2 = m->geom_type[g2];

  if (type1 == mjGEOM_SDF || type2 == mjGEOM_SDF) {
    if (type1 == mjGEOM_PLANE || type2 == mjGEOM_PLANE) {
      return 4;
    }
    if (type1 == mjGEOM_MESH || type2 == mjGEOM_MESH) {
      return mjMAXCONPAIR;
    }
    if (type1 == mjGEOM_HFIELD || type2 == mjGEOM_HFIELD) {
      return 0;
    }
    return mjMIN(mjMAXCONPAIR, mjMAX(0, m->opt.sdf_initpoints));
  }

  if (type1 == mjGEOM_HFIELD || type2 == mjGEOM_HFIELD) {
    int type = (type1 == mjGEOM_HFIELD) ? type2 : type1;
    return (type != mjGEOM_PLANE && type != mjGEOM_HFIELD) ? mjMAXCONPAIR : 0;
  }

  // spheres and ellipsoids always generate a single contact
  if (type1 == mjGEOM_SPHERE || type1 == mjGEOM_ELLIPSOID ||
      type2 == mjGEOM_SPHERE || type2 == mjGEOM_ELLIPSOID) {
    return 1;
  }

  // box-box primitive collider
  if (type1 == mjGEOM_BOX && type2 == mjGEOM_BOX) {
    return 8;
  }

  // capsule-capsule primitive collider
  if (type1 == mjGEOM_CAPSULE && type2 == mjGEOM_CAPSULE) {
    return 2;
  }

  // capsule-box primitive collider
  if ((type1 == mjGEOM_CAPSULE && type2 == mjGEOM_BOX) ||
      (type1 == mjGEOM_BOX && type2 == mjGEOM_CAPSULE)) {
    return 4;
  }

  // geoms with flat faces
  int hasface1 = (type1 == mjGEOM_BOX || type1 == mjGEOM_MESH || type1 == mjGEOM_CYLINDER);
  int hasface2 = (type2 == mjGEOM_BOX || type2 == mjGEOM_MESH || type2 == mjGEOM_CYLINDER);

  // geoms with only edges
  int hasedge1 = (type1 == mjGEOM_CAPSULE);
  int hasedge2 = (type2 == mjGEOM_CAPSULE);

  // the remaining plane cases
  if (type1 == mjGEOM_PLANE || type2 == mjGEOM_PLANE) {
    if (hasface1 || hasface2) {
      return 4;
    }
    if (hasedge1 || hasedge2) {
      return 2;
    }
    return 0;
  }

  int is_multiccd = !mjDISABLED(mjDSBL_MULTICCD);
  if (!is_multiccd) {
    return 1;
  }

  if (mjDISABLED(mjDSBL_NATIVECCD)) {
    return 5;
  }

  // colliding geoms with flat faces
  if (hasface1 && hasface2) {
    return 4;
  }

  // colliding geoms with edges
  if ((hasedge1 && hasface2) || (hasface1 && hasedge2) || (hasedge1 && hasedge2)) {
    return 2;
  }

  return 1;
}


// return the margin for a given geom pair
static inline mjtNum getMargin(const mjModel* m, int g1, int g2, int ipair) {
  if (ipair >= 0) {
    return mj_assignMargin(m, m->pair_margin[ipair]);
  }
  return mj_assignMargin(m, m->geom_margin[g1] + m->geom_margin[g2]);
}


// return the gap for a given geom pair
static inline mjtNum getGap(const mjModel* m, int g1, int g2, int ipair) {
  if (ipair >= 0) {
    return m->pair_gap[ipair];
  }
  return m->geom_gap[g1] + m->geom_gap[g2];
}


// move arena pointer back to the end of the contact array
static inline void resetArena(mjData* d) {
  d->parena = d->ncon * sizeof(mjContact);
#ifdef mjUSEASAN
    ASAN_POISON_MEMORY_REGION(
      (char*)d->arena + d->parena, d->narena - d->pstack - d->parena);
#endif
}


// realign the arena pointer to the specified alignment
static inline size_t alignArena(mjData* d, size_t alignment) {
  size_t misalignment = d->parena % alignment;
  if (misalignment) {
    d->parena += alignment - misalignment;
  }
  return d->parena;
}


// plane to geom_center squared distance, g1 is a plane
static mjtNum planeGeomDist(const mjModel* m, mjData* d, int g1, int g2) {
  mjtNum* mat1 = d->geom_xmat + 9*g1;
  mjtNum norm[3] = {mat1[2], mat1[5], mat1[8]};
  mjtNum dif[3];

  mju_sub3(dif, d->geom_xpos + 3*g2, d->geom_xpos + 3*g1);
  return mju_dot3(dif, norm);
}


// return 1 if body has plane geom, 0 otherwise
static int hasPlane(const mjModel* m, int body) {
  int start = m->body_geomadr[body];
  int end = m->body_geomadr[body] + m->body_geomnum[body];

  // scan geoms belonging to body
  int g;
  for (g=start; g < end; g++) {
    if (m->geom_type[g] == mjGEOM_PLANE) {
      return 1;
    }
  }

  return 0;
}


// filter contact based on type and affinity
static int filterBitmask(int contype1, int conaffinity1,
                         int contype2, int conaffinity2) {
  return !(contype1 & conaffinity2) && !(contype2 & conaffinity1);
}


// filter contact based on global AABBs
static int filterBox(const mjtNum aabb1[6], const mjtNum aabb2[6], mjtNum margin) {
  if (aabb1[0]+aabb1[3]+margin < aabb2[0]-aabb2[3]) return 1;
  if (aabb1[1]+aabb1[4]+margin < aabb2[1]-aabb2[4]) return 1;
  if (aabb1[2]+aabb1[5]+margin < aabb2[2]-aabb2[5]) return 1;
  if (aabb2[0]+aabb2[3]+margin < aabb1[0]-aabb1[3]) return 1;
  if (aabb2[1]+aabb2[4]+margin < aabb1[1]-aabb1[4]) return 1;
  if (aabb2[2]+aabb2[5]+margin < aabb1[2]-aabb1[5]) return 1;
  return 0;
}


// filter contact based sphere-box test, treating sphere as box
static int filterSphereBox(const mjtNum s[3], mjtNum bound, const mjtNum aabb[6]) {
  if (s[0]+bound < aabb[0]-aabb[3]) return 1;
  if (s[1]+bound < aabb[1]-aabb[4]) return 1;
  if (s[2]+bound < aabb[2]-aabb[5]) return 1;
  if (s[0]-bound > aabb[0]+aabb[3]) return 1;
  if (s[1]-bound > aabb[1]+aabb[4]) return 1;
  if (s[2]-bound > aabb[2]+aabb[5]) return 1;
  return 0;
}


// return 1 if speculative collisions are enabled
static inline int speculativeEnabled(const mjModel* m) {
  return mjENABLED(mjENBL_SPECULATIVE) && !mjDISABLED(mjDSBL_NATIVECCD);
}


// return minimum AABB half-thickness of geom g
static inline mjtNum geomMinSize(const mjModel* m, int g) {
  const mjtNum* size = m->geom_aabb + 6*g + 3;
  return mju_min(size[0], mju_min(size[1], size[2]));
}


// return minimum half-thickness across a geom pair
static inline mjtNum pairMinSize(const mjModel* m, int g1, int g2) {
  if (m->geom_type[g1] == mjGEOM_PLANE) return geomMinSize(m, g2);
  if (m->geom_type[g2] == mjGEOM_PLANE) return geomMinSize(m, g1);
  return mju_min(geomMinSize(m, g1), geomMinSize(m, g2));
}


// return step sweep distance if geom g moves faster than min_size
static mjtNum geomStepSweep(const mjModel* m, const mjData* d, int g, mjtNum min_size,
                            mjtNum dx[3], mjtNum dw[3]) {
  if (dx) mji_zero3(dx);
  if (dw) mji_zero3(dw);
  if (g < 0 || m->geom_type[g] == mjGEOM_PLANE || !mjc_GeomSupported(m->geom_type[g])) {
    return 0;
  }
  mjtNum dx_copy[3], dw_copy[3];
  mjtNum s = mjc_geomStepMotion(m, d, g, m->opt.timestep, dx_copy, dw_copy);

  // if the geom is moving slower than min_size, we do not need to consider it
  // for continuous collision detection
  if (s < min_size) {
    return 0;
  }

  if (dx) mji_copy3(dx, dx_copy);
  if (dw) mji_copy3(dw, dw_copy);
  return s;
}


// return 1 if any geom in body pair (b1, b2) sweeps past the pair's minimum geom half-thickness
static int bodyPairHasFastGeom(const mjModel* m, const mjData* d, int b1, int b2) {
  int adr1 = m->body_geomadr[b1], num1 = m->body_geomnum[b1];
  int adr2 = m->body_geomadr[b2], num2 = m->body_geomnum[b2];
  mjtNum min_size = mjMAXVAL;
  for (int i = 0; i < num1; i++) {
    int g = adr1 + i;
    if (m->geom_type[g] != mjGEOM_PLANE && mjc_GeomSupported(m->geom_type[g])) {
      min_size = mju_min(min_size, geomMinSize(m, g));
    }
  }
  for (int i = 0; i < num2; i++) {
    int g = adr2 + i;
    if (m->geom_type[g] != mjGEOM_PLANE && mjc_GeomSupported(m->geom_type[g])) {
      min_size = mju_min(min_size, geomMinSize(m, g));
    }
  }
  if (min_size >= mjMAXVAL) {
    return 0;
  }
  for (int i = 0; i < num1; i++) {
    if (geomStepSweep(m, d, adr1 + i, min_size, NULL, NULL) > 0) {
      return 1;
    }
  }
  for (int i = 0; i < num2; i++) {
    if (geomStepSweep(m, d, adr2 + i, min_size, NULL, NULL) > 0) {
      return 1;
    }
  }
  return 0;
}


// filter contact based on bounding sphere test (swept over step for fast geoms)
static int mj_filterSphere(const mjModel* m, mjData* d, int g1, int g2, mjtNum margin) {
  mjtNum dx1[3], dx2[3];
  mjtNum s1 = 0, s2 = 0;

  // if speculative collisions are enabled, compute the step sweep distance for each geom
  if (speculativeEnabled(m)) {
    mjtNum min_size = pairMinSize(m, g1, g2);
    s1 = geomStepSweep(m, d, g1, min_size, dx1, NULL);
    s2 = geomStepSweep(m, d, g2, min_size, dx2, NULL);
  }

  // neither geom is a plane
  if (m->geom_rbound[g1] > 0 && m->geom_rbound[g2] > 0) {
    mjtNum bound = m->geom_rbound[g1] + m->geom_rbound[g2] + margin;
    const mjtNum* pos1 = d->geom_xpos + 3*g1;
    const mjtNum* pos2 = d->geom_xpos + 3*g2;
    mjtNum diff[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};

    // no sweeping motion needed, check distance between centers
    if (s1 == 0 && s2 == 0) {
      return diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] > bound*bound;
    }

    // find closest approach time tau in [0, 1] (fraction of timestep) and check against
    // combined radius
    mjtNum v[3], closest[3];
    mju_sub3(v, dx2, dx1);
    mjtNum vv = mju_dot3(v, v);
    // tau is the minimum of |diff + t*v|^2 on the line segment [0, 1]
    mjtNum tau = (vv > mjMINVAL) ? mju_clip(-mju_dot3(diff, v) / vv, 0.0, 1.0) : 0.0;
    mji_addScl3(closest, diff, v, tau);
    return mju_dot3(closest, closest) > bound * bound;
  }

  // one geom is a plane
  if (m->geom_type[g1] == mjGEOM_PLANE && m->geom_rbound[g2] > 0) {
    mjtNum dist = planeGeomDist(m, d, g1, g2);
    if (s2 > 0) {
      const mjtNum* mat1 = d->geom_xmat + 9*g1;
      mjtNum n1[3] = {mat1[2], mat1[5], mat1[8]};
      dist = mju_min(dist, dist + mju_dot3(n1, dx2));
    }
    if (dist > margin + m->geom_rbound[g2]) {
      return 1;
    }
  }
  if (m->geom_type[g2] == mjGEOM_PLANE && m->geom_rbound[g1] > 0) {
    mjtNum dist = planeGeomDist(m, d, g2, g1);
    if (s1 > 0) {
      const mjtNum* mat2 = d->geom_xmat + 9*g2;
      mjtNum n2[3] = {mat2[2], mat2[5], mat2[8]};
      dist = mju_min(dist, dist + mju_dot3(n2, dx1));
    }
    if (dist > margin + m->geom_rbound[g1]) {
      return 1;
    }
  }
  return 0;
}


// filter body pair; 1: discard, 0: proceed
static int filterBodyPair(int weldbody1, int weldparent1, int asleep1, int dofnum1,
                          int weldbody2, int weldparent2, int asleep2, int dofnum2,
                          int dsbl_filterparent) {
  // same weldbody check
  if (weldbody1 == weldbody2) {
    return 1;
  }

  // both dof-less: no forces can act, skip
  if (dofnum1 == 0 && dofnum2 == 0) {
    return 1;
  }

  // both asleep check
  if (asleep1 && asleep2) {
    return 1;
  }

  // asleep and world-static check (mocap has weldbody != 0, still triggers wake)
  if ((asleep1 && !weldbody2) || (asleep2 && !weldbody1)) {
    return 1;
  }

  // weldparent check
  if ((!dsbl_filterparent && weldbody1 != 0 && weldbody2 != 0) &&
      (weldbody1 == weldparent2 || weldbody2 == weldparent1)) {
    return 1;
  }

  // all tests passed
  return 0;
}


// return 1 if bodyflex can collide, 0 otherwise
static int canCollide(const mjModel* m, int bf) {
  if (bf < m->nbody) {
    return (m->body_contype[bf] || m->body_conaffinity[bf]);
  } else {
    int f = bf - m->nbody;
    return (m->flex_contype[f] || m->flex_conaffinity[f]);
  }
}


// return 1 if two bodyflexes can collide, 0 otherwise
static int canCollide2(const mjModel* m, int bf1, int bf2) {
  int nbody = m->nbody;
  int contype1 = (bf1 < nbody) ? m->body_contype[bf1] : m->flex_contype[bf1-nbody];
  int conaffinity1 = (bf1 < nbody) ? m->body_conaffinity[bf1] : m->flex_conaffinity[bf1-nbody];
  int contype2 = (bf2 < nbody) ? m->body_contype[bf2] : m->flex_contype[bf2-nbody];
  int conaffinity2 = (bf2 < nbody) ? m->body_conaffinity[bf2] : m->flex_conaffinity[bf2-nbody];

  // opposite of bitmask filter
  return (!filterBitmask(contype1, conaffinity1, contype2, conaffinity2));
}


// return 1 if bodyflex has dofs and they are awake, 0 otherwise
static int hasAwakeDofs(const mjModel* m, const mjData* d, int bf, int sleep_filter) {
  // body
  if (bf < m->nbody) {
    return m->body_treeid[bf] >= 0 && (!sleep_filter || d->body_awake[bf] == mjS_AWAKE);
  }

  // flex: static if none of its vertices (or nodes) are in a body with dofs
  mjtSleepState state = mj_sleepState(m, d, mjOBJ_FLEX, bf - m->nbody);
  return state != mjS_STATIC && (!sleep_filter || state == mjS_AWAKE);
}


// return 1 if element is active, 0 otherwise
int mj_isElemActive(const mjModel* m, int f, int e) {
  if (m->flex_dim[f] < 3) {
    return 1;
  } else {
    return (m->flex_elemlayer[m->flex_elemadr[f]+e] < m->flex_activelayers[f]);
  }
}


//----------------------------- collision detection entry point ------------------------------------

// internal structure for storing pairs meant for narrowphase collision detection
typedef struct {
  // type of collision pair
  enum {
    mjCPAIR_GEOM_GEOM = 0,  // mjCOLLISIONFUNC[type1][type2]
    mjCPAIR_GEOM_FLEX,      // mjc_PlaneFlex, mjc_FlexSDF
    mjCPAIR_GEOM_ELEM,      // mjc_GeomElem
    mjCPAIR_ELEM_ELEM,      // mjc_ElemElem
  } type;
  int conpos;
  int group;                // filter group (-1: no filtering)
  union {
    struct { int g1, g2, ipair, spec; } geom_geom;
    struct { int g, f; } geom_flex;
    struct { int g, f, e; } geom_elem;
    struct { int f1, e1, f2, e2; } elem_elem;
  };
} mjcPair;


// initialize a default collision pair
static inline void defaultPair(mjcPair* pair, int type) {
  pair->type = type;
  pair->group = -1;
  // clear all fields in the union
  pair->elem_elem.f1 = -1;
  pair->elem_elem.e1 = -1;
  pair->elem_elem.f2 = -1;
  pair->elem_elem.e2 = -1;
}


// binary search between two bodyflex trees
static int mj_collideTree(const mjModel* m, mjData* d, int bf1, int bf2,
                          int merged, int startadr, int pairadr,
                          size_t parena, int npair, int group);

// struct for storing pair of 16-bit unsigned integers
typedef struct {
  uint16_t hi, lo;
} mjPacked32;

// broadphase collision detection; return number of bodyflex pairs
static int mj_broadphase(const mjModel* m, mjData* d, mjPacked32* bfpair, int maxpair);

// compute contacts for a batch of collision pairs on the arena starting at parena
static void mj_narrowphase(const mjModel* m, mjData* d, int npair, size_t parena);

// test active element self-collisions with SAP, pushing pairs onto the arena
static int mj_collideFlexSAPPairs(const mjModel* m, mjData* d, int f, int group, int npair);

// test active element self-collisions of an interpolated flex by grouping elements into cells
static int mj_collideFlexCellPairs(const mjModel* m, mjData* d, int f, int group,
                                   size_t parena, int npair);

// get colliding object IDs for a collision pair
static inline void pairObjects(const mjcPair* p, int* obj1, int* obj2) {
  switch (p->type) {
    case mjCPAIR_GEOM_GEOM:
      *obj1 = p->geom_geom.g1;
      *obj2 = p->geom_geom.g2;
      break;

    case mjCPAIR_GEOM_FLEX:
      *obj1 = p->geom_flex.g;
      *obj2 = p->geom_flex.f;
      break;

    case mjCPAIR_GEOM_ELEM:
      *obj1 = p->geom_elem.g;
      *obj2 = p->geom_elem.e;
      break;

    case mjCPAIR_ELEM_ELEM:
      *obj1 = p->elem_elem.e1;
      *obj2 = p->elem_elem.e2;
      break;

    default:
      *obj1 = -1;
      *obj2 = -1;
      break;
  }
}


// compare two collision pairs for ordering
static inline int pairCompare(const mjcPair* p1, const mjcPair* p2, void* context) {
  (void) context;

  // if both have a valid group, keep pairs in the same group contiguous
  if (p1->group >= 0 && p2->group >= 0 && p1->group != p2->group) {
    return p1->group < p2->group ? -1 : 1;
  }

  // sort by type
  if (p1->type != p2->type) {
    return p1->type < p2->type ? -1 : 1;
  }

  // sort by colliding objects
  int obj1_1, obj1_2, obj2_1, obj2_2;
  pairObjects(p1, &obj1_1, &obj1_2);
  pairObjects(p2, &obj2_1, &obj2_2);

  if (obj1_1 < obj2_1) return -1;
  if (obj1_1 > obj2_1) return 1;
  if (obj1_2 < obj2_2) return -1;
  if (obj1_2 > obj2_2) return 1;
  return 0;
}

// define pairSort function for sorting collision pairs
mjSORT(pairSort, mjcPair, pairCompare);




// push a candidate collision pair onto the arena
static int pushPairArena(mjData* d, const mjcPair* pair, int npair) {
  // allocate geom pair on the arena
  mjcPair* new_pair = (mjcPair*) mj_arenaAllocByte(d, sizeof(mjcPair), _Alignof(mjcPair));
  if (!new_pair) {
    mjERROR("arena too small to allocate geom pair");
  }
  *new_pair = *pair;
  return npair + 1;
}


// push a geom-geom collision pair onto the arena
static int pushGeomGeom(const mjModel* m, mjData* d, int g1, int g2, int ipair, int npair) {
  if (m->geom_type[g1] > m->geom_type[g2]) {
    int tmp = g1;
    g1 = g2;
    g2 = tmp;
  }

  mjcPair pair;
  defaultPair(&pair, mjCPAIR_GEOM_GEOM);
  pair.geom_geom.g1 = g1;
  pair.geom_geom.g2 = g2;
  pair.geom_geom.ipair = ipair;
  pair.geom_geom.spec = 0;
  return pushPairArena(d, &pair, npair);
}


// flex with few dofs (rigid or interpolated): its contacts can be reduced to mjMAXCONPAIR
static inline int flexReducible(const mjModel* m, int f) {
  return m->flex_rigid[f] || m->flex_interp[f];
}


// filter group of geom g in a body with first geom geomadr (-1: no filtering)
static inline int geomGroup(int group, int g, int geomadr) {
  return group < 0 ? -1 : group + g - geomadr;
}


// push a geom-flex collision pair onto the arena
static int pushGeomFlex(mjData* d, int g, int f, int group, int npair) {
  mjcPair pair;
  defaultPair(&pair, mjCPAIR_GEOM_FLEX);
  pair.group = group;
  pair.geom_flex.g = g;
  pair.geom_flex.f = f;
  return pushPairArena(d, &pair, npair);
}


// push a geom-elem collision pair onto the arena
static int pushGeomElem(mjData* d, int g, int f, int e, int group, int npair) {
  mjcPair pair;
  defaultPair(&pair, mjCPAIR_GEOM_ELEM);
  pair.group = group;
  pair.geom_elem.g = g;
  pair.geom_elem.f = f;
  pair.geom_elem.e = e;
  return pushPairArena(d, &pair, npair);
}


// push an elem-elem collision pair onto the arena
static int pushElemElem(mjData* d, int f1, int e1, int f2, int e2, int group, int npair) {
  mjcPair pair;
  defaultPair(&pair, mjCPAIR_ELEM_ELEM);
  pair.group = group;
  pair.elem_elem.f1 = f1;
  pair.elem_elem.e1 = e1;
  pair.elem_elem.f2 = f2;
  pair.elem_elem.e2 = e2;
  return pushPairArena(d, &pair, npair);
}


// push an elem-elem self-collision pair of flex f onto the arena, unless it is excluded
static int pushSelfElemElem(const mjModel* m, mjData* d, int f, int e1, int e2, int group,
                            int npair) {
  // interpolated flex: skip elements driven by a common grid node
  if (mjc_ElemSharedNode(m, f, e1, e2)) {
    return npair;
  }
  return pushElemElem(d, f, e1, f, e2, group, npair);
}


// filter candidate collision pair; return 0 if pair should be discarded
static int filterCollisionPair(const mjModel* m, mjData* d, int g1, int g2, int ipair,
                               int merged, int startadr, int pairadr) {
  // merged, find matching pair
  if (merged) {
    for (int k=startadr; k < pairadr; k++) {
      if ((m->pair_geom1[k] == g1 && m->pair_geom2[k] == g2) ||
          (m->pair_geom1[k] == g2 && m->pair_geom2[k] == g1)) {
        return 0;
      }
    }
  }

  if (ipair >= 0) {
    if (mjENABLED(mjENBL_SLEEP)) {
      int b1 = m->geom_bodyid[g1];
      int b2 = m->geom_bodyid[g2];
      if (d->body_awake[b1] != mjS_AWAKE && d->body_awake[b2] != mjS_AWAKE) {
        return 0;
      }
    }
  }

  if (ipair < 0) {
    if (mjcb_contactfilter) {
      if (mjcb_contactfilter(m, d, g1, g2)) {
        return 0;
      }
    } else if (filterBitmask(m->geom_contype[g1], m->geom_conaffinity[g1],
                             m->geom_contype[g2], m->geom_conaffinity[g2])) {
      return 0;
    }
  }

  // bounding sphere filter
  mjtNum margin = getMargin(m, g1, g2, ipair);
  mjtNum gap = getGap(m, g1, g2, ipair);
  if (mj_filterSphere(m, d, g1, g2, margin + gap)) {
    return 0;
  }

  // highly unlikely, but check collision function is well-defined
  int type1 = mjMIN(m->geom_type[g1], m->geom_type[g2]);
  int type2 = mjMAX(m->geom_type[g1], m->geom_type[g2]);
  return mjCOLLISIONFUNC[type1][type2] != NULL;
}


// main collision function
void mj_collision(const mjModel* m, mjData* d) {
  TM_START1;

  int nexclude = m->nexclude, npair = m->npair, nbody = m->nbody;
  int nbodyflex = m->nbody + m->nflex;
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < nbody;

  // reset the size of the contact array and invalidate efc arrays
  d->ncon = 0;
  resetArena(d);
  mj_clearEfc(d);

  // return if disabled
  if (mjDISABLED(mjDSBL_CONSTRAINT) || mjDISABLED(mjDSBL_CONTACT) || nbodyflex < 2) {
    TM_END1(mjTIMER_POS_COLLISION);
    return;
  }

  // reset the visualization flags
  if (m->vis.global.bvactive) {
    memset(d->bvh_active, 0, m->nbvh);
  }

  mj_markStack(d);

  // broadphase collision detector
  TM_START;
  int nmaxpairs = (nbodyflex*(nbodyflex - 1))/2;
  mjPacked32* broadphasepair = mjSTACKALLOC(d, nmaxpairs, mjPacked32);
  int nbfpair = mj_broadphase(m, d, broadphasepair, nmaxpairs);
  unsigned int last_signature = -1;
  TM_END(mjTIMER_COL_BROAD);

  // narrowphase and midphase collision detector
  TM_RESTART;

  // process bodyflex pairs returned by broadphase, merge with predefined geom pairs
  int pairadr = 0, ncandidate = 0, ngroup = 0, g1, g2;

  // align the arena for candidate collision pairs for narrowphase
  size_t parena = alignArena(d, _Alignof(mjcPair));

  for (int i=0; i < nbfpair; i++) {
    // reconstruct bodyflex pair ids
    int bf1 = broadphasepair[i].hi;
    int bf2 = broadphasepair[i].lo;

    // compute signature for this bodyflex pair
    unsigned int signature = ((unsigned int)bf1 << 16) + bf2;
    // pairs come sorted by signature, but may not be unique
    // if signature is repeated, skip it
    if (signature == last_signature) {
      continue;
    }
    last_signature = signature;

    // merge predefined geom pairs
    int merged = 0;
    int startadr = pairadr;
    // test all predefined pairs for which pair_signature <= signature
    for (; pairadr < npair && m->pair_signature[pairadr] <= signature; pairadr++) {
      merged = (m->pair_signature[pairadr] == signature);
      g1 = m->pair_geom1[pairadr], g2 = m->pair_geom2[pairadr];
      if (filterCollisionPair(m, d, g1, g2, pairadr, 0, 0, 0)) {
        ncandidate = pushGeomGeom(m, d, g1, g2, pairadr, ncandidate);
      }
    }

    // apply bitmask filtering at the bodyflex level
    if (!canCollide2(m, bf1, bf2)) {
      continue;
    }

    // under the ipc flag the IPC step resolves two dim-2 flexes itself
    if (bf1 >= nbody && bf2 >= nbody && mjc_ipcOwnsFlexFlex(m, bf1 - nbody, bf2 - nbody)) {
      continue;
    }

    // handle body pair exclusion
    int exadr = 0;
    if (nexclude) {
      // advance exadr while exclude_signature < signature
      while (exadr < nexclude && m->exclude_signature[exadr] < signature) {
        exadr++;
      }

      // skip this bodyflex pair if its signature is found in exclude array
      if (exadr < nexclude && m->exclude_signature[exadr] == signature) {
        continue;
      }
    }

    // get bodyflex info
    int isbody1 = (bf1 < nbody);
    int isbody2 = (bf2 < nbody);
    int bvh1 = (isbody1 ? m->body_bvhadr[bf1] : m->flex_bvhadr[bf1-nbody]);
    int bvh2 = (isbody2 ? m->body_bvhadr[bf2] : m->flex_bvhadr[bf2-nbody]);
    int geomadr1 = (isbody1 ? m->body_geomadr[bf1] : -1);
    int geomadr2 = (isbody2 ? m->body_geomadr[bf2] : -1);

    // process bodyflex pair: two single-geom bodies
    if (isbody1 && isbody2 && m->body_geomnum[bf1] == 1 && m->body_geomnum[bf2] == 1) {
      if (filterCollisionPair(m, d, geomadr1, geomadr2, -1, merged, startadr, pairadr)) {
        ncandidate = pushGeomGeom(m, d, geomadr1, geomadr2, -1, ncandidate);
      }
    }

    // process bodyflex pair: midphase
    else if (!mjDISABLED(mjDSBL_MIDPHASE) && bvh1 >= 0 && bvh2 >= 0 &&
             !(speculativeEnabled(m) && isbody1 && isbody2 &&
               bodyPairHasFastGeom(m, d, bf1, bf2))) {
      // filter groups for reducible flexes: one per geom for body:flex, one for flex:flex
      int group = -1;
      if (isbody1 && !isbody2) {
        if (flexReducible(m, bf2-nbody)) {
          group = ngroup;
          ngroup += m->body_geomnum[bf1];
        }
      } else if (!isbody1 && flexReducible(m, bf1-nbody) &&
                 flexReducible(m, bf2-nbody)) {
        group = ngroup++;
      }
      ncandidate = mj_collideTree(m, d, bf1, bf2, merged, startadr, pairadr, parena, ncandidate, group);
    }

    // process bodyflex pair: all-to-all
    else {
      int geomadr_end1 = (isbody1 ? geomadr1 + m->body_geomnum[bf1] : -1);
      int geomadr_end2 = (isbody2 ? geomadr2 + m->body_geomnum[bf2] : -1);

      // body : body
      if (isbody1 && isbody2) {
        for (g1=geomadr1; g1 < geomadr_end1; g1++) {
          for (g2=geomadr2; g2 < geomadr_end2; g2++) {
            if (filterCollisionPair(m, d, g1, g2, -1, merged, startadr, pairadr)) {
              ncandidate = pushGeomGeom(m, d, g1, g2, -1, ncandidate);
            }
          }
        }
      }

      // body : flex
      else if (isbody1) {
        int f = bf2 - nbody;

        // process body geoms
        for (int g=m->body_geomadr[bf1]; g < geomadr_end1; g++) {
          // bitmask filtering at the geom-flex level
          if (filterBitmask(m->geom_contype[g], m->geom_conaffinity[g],
                            m->flex_contype[f], m->flex_conaffinity[f])) {
            continue;
          }

          if (mjc_ipcOwnsFlexGeom(m, f, g)) {
            continue;
          }

          int group = flexReducible(m, f) ? ngroup++ : -1;

          // plane special processing
          if (m->geom_type[g] == mjGEOM_PLANE) {
            ncandidate = pushGeomFlex(d, g, f, group, ncandidate);
            continue;
          }

          // SDF special processing
          if (m->geom_type[g] == mjGEOM_SDF) {
            if (m->flex_dim[f] == 2) {
              ncandidate = pushGeomFlex(d, g, f, group, ncandidate);
            }
            continue;
          }

          // collide geom with flex elements
          int elemnum = m->flex_elemnum[f];
          for (int e=0; e < elemnum; e++) {
            if (mj_isElemActive(m, f, e)) {
              ncandidate = pushGeomElem(d, g, f, e, group, ncandidate);
            }
          }
        }
      }

      // flex : flex
      else {
        int f1 = bf1 - nbody;
        int f2 = bf2 - nbody;
        int group = (flexReducible(m, f1) && flexReducible(m, f2)) ? ngroup++ : -1;

        // collide elements of two flexes
        for (int e1=0; e1 < m->flex_elemnum[f1]; e1++) {
          if (mj_isElemActive(m, f1, e1)) {
            for (int e2=0; e2 < m->flex_elemnum[f2]; e2++) {
              if (mj_isElemActive(m, f2, e2)) {
                ncandidate = pushElemElem(d, f1, e1, f2, e2, group, ncandidate);
              }
            }
          }
        }
      }
    }
  }
  mj_freeStack(d);

  // finish merging predefined geom pairs
  for (; pairadr < npair; pairadr++) {
    g1 = m->pair_geom1[pairadr], g2 = m->pair_geom2[pairadr];
    if (filterCollisionPair(m, d, g1, g2, pairadr, 0, 0, 0)) {
      ncandidate = pushGeomGeom(m, d, g1, g2, pairadr, ncandidate);
    }
  }

  // flex self-collisions
  for (int f=0; f < m->nflex; f++) {
    if (!m->flex_rigid[f] && (m->flex_contype[f] & m->flex_conaffinity[f])) {
      // skip if flex has no dofs or is asleep
      if (!hasAwakeDofs(m, d, nbody+f, sleep_filter)) continue;
      // under the ipc flag the IPC step resolves a dim-2 flex's self-contact itself
      if (mjc_ipcOwnsFlexFlex(m, f, f)) continue;

      // active element collisions; for an interpolated flex with at most 2 cells along every
      // axis, all cells are neighbors, so every pair is excluded by mjc_ElemSharedNode
      const int* cellnum = m->flex_cellnum + 3*f;
      int all_excluded = m->flex_interp[f] && cellnum[0] <= 2 && cellnum[1] <= 2 &&
                         cellnum[2] <= 2;
      if (m->flex_selfcollide[f] != mjFLEXSELF_NONE && !all_excluded) {
        int group = flexReducible(m, f) ? ngroup++ : -1;

        // element-element: midphase
        if (!mjDISABLED(mjDSBL_MIDPHASE) &&
            m->flex_selfcollide[f] != mjFLEXSELF_NARROW &&
            (m->flex_bvhadr[f] >= 0 ||
             (m->flex_selfcollide[f] == mjFLEXSELF_AUTO && m->flex_interp[f]))) {
          // select midphase mode
          if (m->flex_selfcollide[f] == mjFLEXSELF_AUTO && m->flex_interp[f]) {
            ncandidate = mj_collideFlexCellPairs(m, d, f, group, parena, ncandidate);
          } else if (m->flex_selfcollide[f] == mjFLEXSELF_BVH ||
                     (m->flex_selfcollide[f] == mjFLEXSELF_AUTO && m->flex_dim[f] == 3)) {
            ncandidate = mj_collideTree(m, d, nbody+f, nbody+f, 0, 0, 0, parena, ncandidate,
                                        group);
          } else {
            ncandidate = mj_collideFlexSAPPairs(m, d, f, group, ncandidate);
          }
        }

        // element-element: direct
        else {
          int flex_elemnum = m->flex_elemnum[f];
          for (int e1=0; e1 < flex_elemnum; e1++) {
            if (mj_isElemActive(m, f, e1)) {
              for (int e2=e1+1; e2 < flex_elemnum; e2++) {
                if (mj_isElemActive(m, f, e2)) {
                  ncandidate = pushSelfElemElem(m, d, f, e1, e2, group, ncandidate);
                }
              }
            }
          }
        }
      }
    }
  }

  // narrowphase: compute contacts for all candidate pairs in a single global call
  if (ncandidate > 0) {
    mj_narrowphase(m, d, ncandidate, parena);
  }

  // end narrowphase and midphase timer
  TM_END(mjTIMER_COL_NARROW);
  TM_END1(mjTIMER_POS_COLLISION);
}


//------------------------------------ binary tree search ------------------------------------------

// collision tree node
typedef struct  {
  int node1;
  int node2;
} mjCollisionTree;

// oriented bounding boxes collision (see Gottschalk et al.)
int mj_collideOBB(const mjtNum aabb1[6], const mjtNum aabb2[6],
                  const mjtNum xpos1[3], const mjtNum xmat1[9],
                  const mjtNum xpos2[3], const mjtNum xmat2[9], mjtNum margin,
                  mjtNum product[36], mjtNum offset[12], mjtBool* initialize) {
  // get infinite dimensions (planes only)
  mjtBool inf1[3] = {aabb1[3] >= mjMAXVAL, aabb1[4] >= mjMAXVAL, aabb1[5] >= mjMAXVAL};
  mjtBool inf2[3] = {aabb2[3] >= mjMAXVAL, aabb2[4] >= mjMAXVAL, aabb2[5] >= mjMAXVAL};

  // if a bounding box is infinite, there must be a collision
  if ((inf1[0] && inf1[1] && inf1[2]) || (inf2[0] && inf2[1] && inf2[2])) {
    return 1;
  }

  const mjtNum* aabb[2] = {aabb1, aabb2};
  const mjtNum *xmat[2] = {xmat1, xmat2};
  const mjtNum *xpos[2] = {xpos1, xpos2};
  mjtNum xcenter[2][3], normal[2][3][3];
  mjtNum proj[2], radius[2];
  mjtBool infinite[2] = {inf1[0] || inf1[1] || inf1[2], inf2[0] || inf2[1] || inf2[2]};

  // compute centers in local coordinates
  if (product == NULL) {
    for (int i=0; i < 2; i++) {  // bounding boxes
      if (xmat[i]) {
        mju_mulMatVec3(xcenter[i], xmat[i], aabb[i]);
      } else {
        mju_copy3(xcenter[i], aabb[i]);
      }

      if (xpos[i]) {
        mju_addTo3(xcenter[i], xpos[i]);
      }
    }
  }

  // compute normals in global coordinates
  for (int i=0; i < 2; i++) {  // bounding boxes
    for (int j=0; j < 3; j++) {  // faces
      for (int k=0; k < 3; k++) {  // world axes
        if (xmat[i]) {
          normal[i][j][k] = xmat[i][3*k+j];
        } else {
          normal[i][j][k] = (j == k);
        }
      }
    }
  }

  // precompute dot products
  if (product && offset && *initialize) {
    for (int i=0; i < 2; i++) {  // bodies
      for (int j=0; j < 2; j++) {  // bodies
        for (int k=0; k < 3; k++) {  // axes
          for (int l=0; l < 3; l++) {  // axes
            product[18*i + 9*j + 3*k + l] = mju_dot3(normal[i][l], normal[j][k]);
          }
          offset[6*i + 3*j + k] = xpos[i] ? mju_dot3(xpos[i], normal[j][k]) : 0;
        }
      }
    }
    *initialize = 0;
  }

  // check intersections
  for (int j=0; j < 2; j++) {  // bounding boxes
    if (infinite[1-j]) {
      continue;  // skip test against an infinite body
    }
    for (int k=0; k < 3; k++) {  // face
      for (int i=0; i < 2; i++) {  // bounding boxes
        if (product == NULL) {
          proj[i] = mju_dot3(xcenter[i], normal[j][k]);
          radius[i] = mju_abs(aabb[i][3]*mju_dot3(normal[i][0], normal[j][k])) +
                      mju_abs(aabb[i][4]*mju_dot3(normal[i][1], normal[j][k])) +
                      mju_abs(aabb[i][5]*mju_dot3(normal[i][2], normal[j][k]));
        } else {
          int adr = 18*i + 9*j + 3*k;
          proj[i] = aabb[i][0] * product[adr + 0] +
                    aabb[i][1] * product[adr + 1] +
                    aabb[i][2] * product[adr + 2] +
                    offset[6*i + 3*j + k];
          radius[i] = mju_abs(aabb[i][3]*product[adr + 0]) +
                      mju_abs(aabb[i][4]*product[adr + 1]) +
                      mju_abs(aabb[i][5]*product[adr + 2]);
        }
      }

      if (radius[0]+radius[1]+margin < mju_abs(proj[1]-proj[0])) {
        return 0;
      }
    }
  }

  return 1;
}


// binary search between two bodyflex trees
//  body:flex pairs of geom g are in filter group: group + g - body_geomadr[bf1]
static int mj_collideTree(const mjModel* m, mjData* d, int bf1, int bf2,
                          int merged, int startadr, int pairadr,
                          size_t parena, int npair, int group) {
  int nbody = m->nbody, nbvhstatic = m->nbvhstatic;
  mjtBool isbody1 = (bf1 < nbody);
  mjtBool isbody2 = (bf2 < nbody);
  int f1 = isbody1 ? -1 : bf1 - nbody;
  int f2 = isbody2 ? -1 : bf2 - nbody;
  int geomadr1 = isbody1 ? m->body_geomadr[bf1] : -1;
  int mark_active = m->vis.global.bvactive;
  const int bvhadr1 = isbody1 ? m->body_bvhadr[bf1] : m->flex_bvhadr[f1];
  const int bvhadr2 = isbody2 ? m->body_bvhadr[bf2] : m->flex_bvhadr[f2];
  const int* child1 = m->bvh_child + 2*bvhadr1;
  const int* child2 = m->bvh_child + 2*bvhadr2;
  const mjtNum* bvh1 = isbody1 ? (m->bvh_aabb     + 6*bvhadr1) :
                                 (d->bvh_aabb_dyn + 6*(bvhadr1 - nbvhstatic));
  const mjtNum* bvh2 = isbody2 ? (m->bvh_aabb     + 6*bvhadr2) :
                                 (d->bvh_aabb_dyn + 6*(bvhadr2 - nbvhstatic));

  // used with rotated bounding boxes (when bodies are involved)
  mjtNum product[36];  // 2 bb x 2 bb x 3 axes (body) x 3 axes (world)
  mjtNum offset[12];   // 2 bb x 2 bb x 3 axes (world)
  mjtBool initialize = true;

  // bitmask filter for bodyflex pair
  if (!canCollide2(m, bf1, bf2)) {
    return npair;
  }

  mj_markStack(d);
  // TODO(b/273737633): Store bvh max depths to make this bound tighter.
  const int max_stack = (isbody1 ? m->body_bvhnum[bf1] : m->flex_bvhnum[f1]) +
                        (isbody2 ? m->body_bvhnum[bf2] : m->flex_bvhnum[f2]);
  mjCollisionTree* stack = mjSTACKALLOC(d, max_stack, mjCollisionTree);

  int nstack = 1;
  stack[0].node1 = stack[0].node2 = 0;
  int start_pair = npair;

  // for body:flex, if dof-less body has planes, call pushGeomFlex
  if (isbody1 && !isbody2 && m->body_dofnum[m->body_weldid[bf1]] == 0) {
    for (int i=geomadr1; i < geomadr1+m->body_geomnum[bf1]; i++) {
      if (m->geom_type[i] == mjGEOM_PLANE) {
        if (!filterBitmask(m->geom_contype[i], m->geom_conaffinity[i],
                           m->flex_contype[f2], m->flex_conaffinity[f2]) &&
            !mjc_ipcOwnsFlexGeom(m, f2, i)) {
          npair = pushGeomFlex(d, i, f2, geomGroup(group, i, geomadr1), npair);
        }
      }
    }
  }

  // for body:flex, if body has SDFs, call pushGeomFlex
  if (isbody1 && !isbody2) {
    for (int i=geomadr1; i < geomadr1+m->body_geomnum[bf1]; i++) {
      if (m->geom_type[i] == mjGEOM_SDF) {
        if (m->flex_dim[f2] == 2 &&
            !filterBitmask(m->geom_contype[i], m->geom_conaffinity[i],
                           m->flex_contype[f2], m->flex_conaffinity[f2])) {
          npair = pushGeomFlex(d, i, f2, geomGroup(group, i, geomadr1), npair);
        }
      }
    }
  }

  // collide trees
  while (nstack) {
    // pop from stack
    nstack--;
    int node1 = stack[nstack].node1;
    int node2 = stack[nstack].node2;
    mjtBool isleaf1 = (child1[2*node1] < 0) && (child1[2*node1+1] < 0);
    mjtBool isleaf2 = (child2[2*node2] < 0) && (child2[2*node2+1] < 0);
    int nodeid1 = m->bvh_nodeid[bvhadr1 + node1];
    int nodeid2 = m->bvh_nodeid[bvhadr2 + node2];

    // SHOULD NOT OCCUR
    if ((isleaf1 && nodeid1 < 0) || (isleaf2 && nodeid2 < 0)) {
      mju_error("BVH leaf has invalid node id");
    }

    // self-collision: avoid repeated pairs
    if (bf1 == bf2 && node1 > node2) {
      continue;
    }

    // body : body
    if (isbody1 && isbody2) {
      // both are leaves
      if (isleaf1 && isleaf2) {
        mjtNum margin = mj_assignMargin(m, m->geom_margin[nodeid1] + m->geom_margin[nodeid2]);
        mjtNum gap = m->geom_gap[nodeid1] + m->geom_gap[nodeid2];

        if (!mj_filterSphere(m, d, nodeid1, nodeid2, margin + gap)) {
          if (mj_collideOBB(m->geom_aabb + 6*nodeid1, m->geom_aabb + 6*nodeid2,
                            d->geom_xpos + 3*nodeid1, d->geom_xmat + 9*nodeid1,
                            d->geom_xpos + 3*nodeid2, d->geom_xmat + 9*nodeid2,
                            margin + gap, NULL, NULL, &initialize)) {
            if (filterCollisionPair(m, d, nodeid1, nodeid2, -1, merged, startadr, pairadr)) {
              npair = pushGeomGeom(m, d, nodeid1, nodeid2, -1, npair);
            }
            if (mark_active) {
              d->bvh_active[node1 + bvhadr1] = true;
              d->bvh_active[node2 + bvhadr2] = true;
            }
          }
        }
        continue;
      }

      // if no intersection at intermediate levels, stop
      mjtNum margin = mj_assignMargin(m, m->body_margin[bf1] + m->body_margin[bf2]);
      if (!mj_collideOBB(bvh1 + 6*node1, bvh2 + 6*node2,
                         d->xipos + 3*bf1, d->ximat + 9*bf1,
                         d->xipos + 3*bf2, d->ximat + 9*bf2,
                         margin, product, offset, &initialize)) {
        continue;
      }
    }

    // body : flex
    else if (isbody1 && !isbody2) {
      // both are leaves
      if (isleaf1 && isleaf2) {
        mjtNum margin = mj_assignMargin(m, m->geom_margin[nodeid1] + m->flex_margin[f2]);
        mjtNum gap = m->geom_gap[nodeid1] + m->flex_gap[f2];

        if (!filterBitmask(m->geom_contype[nodeid1], m->geom_conaffinity[nodeid1],
                           m->flex_contype[f2], m->flex_conaffinity[f2]) &&
            !filterSphereBox(d->geom_xpos + 3*nodeid1, m->geom_rbound[nodeid1] + margin + gap,
                             bvh2 + 6*node2)) {
          if (mj_collideOBB(m->geom_aabb + 6*nodeid1, bvh2 + 6*node2,
                            d->geom_xpos + 3*nodeid1, d->geom_xmat + 9*nodeid1,
                            NULL, NULL,
                            margin + gap, NULL, NULL, &initialize)) {
            // collide unless geom is plane or SDF (handled separately)
            if (m->geom_type[nodeid1] != mjGEOM_PLANE &&
                m->geom_type[nodeid1] != mjGEOM_SDF) {
              if (!mjc_ipcOwnsFlexGeom(m, f2, nodeid1)) {
                npair = pushGeomElem(d, nodeid1, f2, nodeid2, geomGroup(group, nodeid1, geomadr1),
                                     npair);
              }
            }
            if (mark_active) {
              d->bvh_active[node1 + bvhadr1] = true;
              d->bvh_active[node2 + bvhadr2] = true;
            }
          }
        }
        continue;
      }

      // if no intersection at intermediate levels, stop
      mjtNum margin = mj_assignMargin(m, m->body_margin[bf1] + m->flex_margin[f2]) +
                      m->flex_gap[f2];
      if (!mj_collideOBB(bvh1 + 6*node1, bvh2 + 6*node2,
                         d->xipos + 3*bf1, d->ximat + 9*bf1,
                         NULL, NULL,
                         margin, product, offset, &initialize)) {
        continue;
      }
    }

    // flex : body  SHOULD NOT OCCUR
    else if (!isbody1 && isbody2) {
      mjERROR("BVH flex : body collision should not occur");
    }

    // flex : flex
    else {
      // both are leaves: same box filter as mjc_ElemElem (margin and gap ignored in
      // self-collisions), bitmask filter applied earlier
      if (isleaf1 && isleaf2) {
        mjtNum margin = 0;
        if (bf1 != bf2) {
          margin = mj_assignMargin(m, m->flex_margin[f1] + m->flex_margin[f2]) +
                   m->flex_gap[f1] + m->flex_gap[f2];
        }
        if (!filterBox(d->flexelem_aabb + 6*(m->flex_elemadr[f1] + nodeid1),
                       d->flexelem_aabb + 6*(m->flex_elemadr[f2] + nodeid2), margin)) {
          int npair_old = npair;
          if (bf1 != bf2) {
            npair = pushElemElem(d, f1, nodeid1, f2, nodeid2, group, npair);
          } else if (nodeid1 != nodeid2) {
            npair = pushSelfElemElem(m, d, f1, nodeid1, nodeid2, group, npair);
          }
          if (mark_active && npair > npair_old) {
            d->bvh_active[node1 + bvhadr1] = true;
            d->bvh_active[node2 + bvhadr2] = true;
          }
        }
        continue;
      }

      // if no intersection at intermediate levels, stop
      mjtNum margin = mj_assignMargin(m, m->flex_margin[f1] + m->flex_margin[f2]) +
                      (bf1 != bf2 ? m->flex_gap[f1] + m->flex_gap[f2] : 0);
      if (filterBox(bvh1 + 6*node1, bvh2 + 6*node2, margin)) {
        continue;
      }
    }

    if (mark_active) {
      d->bvh_active[node1 + bvhadr1] = true;
      d->bvh_active[node2 + bvhadr2] = true;
    }

    // keep traversing the tree
    if (!isleaf1 && isleaf2) {
      for (int i=0; i < 2; i++) {
        if (child1[2*node1+i] != -1) {
          if (nstack >= max_stack) {
            mjERROR("BVH stack depth exceeded.");  // SHOULD NOT OCCUR
          }
          stack[nstack].node1 = child1[2*node1+i];
          stack[nstack].node2 = node2;
          nstack++;
        }
      }
    } else if (isleaf1 && !isleaf2) {
      for (int i=0; i < 2; i++) {
        if (child2[2*node2+i] != -1) {
          if (nstack >= max_stack) {
            mjERROR("BVH stack depth exceeded.");  // SHOULD NOT OCCUR
          }
          stack[nstack].node1 = node1;
          stack[nstack].node2 = child2[2*node2+i];
          nstack++;
        }
      }
    } else {
      // compute surface areas of bounding boxes (center, size): use half-sizes
      mjtNum x1 = bvh1[6*node1+3];
      mjtNum y1 = bvh1[6*node1+4];
      mjtNum z1 = bvh1[6*node1+5];
      mjtNum x2 = bvh2[6*node2+3];
      mjtNum y2 = bvh2[6*node2+4];
      mjtNum z2 = bvh2[6*node2+5];
      mjtNum surface1 = x1*y1 + y1*z1 + z1*x1;
      mjtNum surface2 = x2*y2 + y2*z2 + z2*x2;

      // traverse the hierarchy whose bounding box has the larger surface area
      if (surface1 > surface2) {
        for (int i = 0; i < 2; i++) {
          if (child1[2 * node1 + i] != -1) {
            if (nstack >= max_stack) {
              mjERROR("BVH stack depth exceeded.");  // SHOULD NOT OCCUR
            }
            stack[nstack].node1 = child1[2 * node1 + i];
            stack[nstack].node2 = node2;
            nstack++;
          }
        }
      } else {
        for (int i = 0; i < 2; i++) {
          if (child2[2 * node2 + i] != -1) {
            if (nstack >= max_stack) {
              mjERROR("BVH stack depth exceeded.");  // SHOULD NOT OCCUR
            }
            stack[nstack].node1 = node1;
            stack[nstack].node2 = child2[2*node2+i];
            nstack++;
          }
        }
      }
    }
  }
  // sort pairs generated by this tree collision
  int n = npair - start_pair;
  if (n > 1) {
    mjcPair* buf = mjSTACKALLOC(d, n, mjcPair);
    mjcPair* pairs = (mjcPair*) ((char*) d->arena + parena);
    pairSort(pairs + start_pair, buf, n, (void*)m);
  }

  mj_freeStack(d);
  return npair;
}


//----------------------------- broad-phase collision detection ------------------------------------

// make AAMM (xmin[3], xmax[3]) for one bodyflex, in the frame whose axes are the rows of frame
static void makeAAMM(const mjModel* m, mjData* d,
                     mjtNum* x_min, mjtNum* y_min, mjtNum* z_min,
                     mjtNum* x_max, mjtNum* y_max, mjtNum* z_max,
                     int bf, const mjtNum* frame) {
  mjtNum aamm[6];
  mjtNum override_margin = mjENABLED(mjENBL_OVERRIDE) ? 0.5 * m->opt.o_margin : 0;

  // body
  if (bf < m->nbody) {
    int body = bf;
    int body_geomnum = m->body_geomnum[body];

    // process all body geoms (body is collidable, should have geoms)
    for (int i=0; i < body_geomnum; i++) {
      int geom = m->body_geomadr[body]+i;
      mjtNum margin = override_margin ? override_margin
                                       : m->geom_margin[geom] + m->geom_gap[geom];
      mjtNum _aamm[6];

      const mjtNum* aabb = m->geom_aabb + 6*geom;
      const mjtNum* size = m->geom_aabb + 6*geom + 3;
      const mjtNum* xpos = d->geom_xpos + 3*geom;
      const mjtNum* xmat = d->geom_xmat + 9*geom;

      // compute center in global coordinates
      mjtNum pos[3];
      mji_mulMatVec3(pos, xmat, aabb);
      mju_addTo3(pos, xpos);

      mjtNum axis[9];
      mji_transpose3(axis, xmat);
      mjtNum r_half = m->geom_rbound[geom];
      mjtNum dx[3], dw[3];
      mjtNum s = speculativeEnabled(m) ? geomStepSweep(m, d, geom, 0, dx, dw) : 0;
      mjtNum s_rot = s > 0 ? mju_norm3(dw) * r_half : 0;

      for (int j=0; j < 3; j++) {
        const mjtNum* frame_j = frame + 3*j;
        mjtNum aabb_cen = mju_dot3(pos, frame_j);
        mjtNum aabb_half = mju_abs(size[0] * mju_dot3(axis + 0, frame_j))
                         + mju_abs(size[1] * mju_dot3(axis + 3, frame_j))
                         + mju_abs(size[2] * mju_dot3(axis + 6, frame_j))
                         + s_rot;
        mjtNum r_cen = mju_dot3(xpos, frame_j);
        mjtNum lo = mju_max(r_cen - r_half, aabb_cen - aabb_half);
        mjtNum hi = mju_min(r_cen + r_half, aabb_cen + aabb_half);
        if (s > 0) {
          mjtNum d_proj = mju_dot3(dx, frame_j);
          lo += mju_min(0, d_proj);
          hi += mju_max(0, d_proj);
        }
        _aamm[j + 0] = lo - margin;
        _aamm[j + 3] = hi + margin;
      }

      // update body aamm
      if (i == 0) {
        mju_copy(aamm, _aamm, 6);
      } else {
        for (int j=0; j < 3; j++) {
          aamm[j]   = mju_min(aamm[j],   _aamm[j]);
          aamm[j+3] = mju_max(aamm[j+3], _aamm[j+3]);
        }
      }
    }
  }

  // flex
  else {
    int f = bf - m->nbody;
    int flex_vertnum = m->flex_vertnum[f];
    const mjtNum* vbase = d->flexvert_xpos + 3*m->flex_vertadr[f];

    // process flex vertices
    for (int i=0; i < flex_vertnum; i++) {
      mjtNum v[3];

      // compute vertex coordinates in given frame
      mju_mulMatVec(v, frame, vbase+3*i, 3, 3);

      // update aamm
      if (i == 0) {
        mju_copy3(aamm, v);
        mju_copy3(aamm+3, v);
      } else {
        for (int j=0; j < 3; j++) {
          aamm[j]   = mju_min(aamm[j], v[j]);
          aamm[j+3] = mju_max(aamm[j+3], v[j]);
        }
      }
    }

    // correct for flex radius and margin
    mjtNum margin = override_margin ? override_margin
                                     : m->flex_margin[f] + m->flex_gap[f];
    mjtNum bound = m->flex_radius[f] + margin;
    aamm[0] -= bound;
    aamm[1] -= bound;
    aamm[2] -= bound;
    aamm[3] += bound;
    aamm[4] += bound;
    aamm[5] += bound;
  }

  // assign outputs
  *x_min = aamm[0];
  *y_min = aamm[1];
  *z_min = aamm[2];
  *x_max = aamm[3];
  *y_max = aamm[4];
  *z_max = aamm[5];
}


// add bodyflex pair in buffer; do not filter if m is NULL
static void add_pair(const mjModel* m, int bf1, int bf2,
                     int* npair, mjPacked32* pair, int maxpair) {
  // add pair if there is room in buffer
  if ((*npair) < maxpair) {
    // contact filtering if m is not NULL
    if (m) {
      int nbody = m->nbody;
      int contype1, conaffinity1, contype2, conaffinity2;

      // get contype and conaffinity for bodyflex 1
      if (bf1 < nbody) {
        int body_geomadr1 = m->body_geomadr[bf1];
        int body_geomnum1 = m->body_geomnum[bf1];
        contype1 = conaffinity1 = 0;
        for (int i=body_geomadr1; i < body_geomadr1+body_geomnum1; i++) {
          contype1 |= m->geom_contype[i];
          conaffinity1 |= m->geom_conaffinity[i];
        }
      } else {
        contype1 = m->flex_contype[bf1-nbody];
        conaffinity1 = m->flex_conaffinity[bf1-nbody];
      }

      // get contype and conaffinity for bodyflex 2
      if (bf2 < nbody) {
        int body_geomadr2 = m->body_geomadr[bf2];
        int body_geomnum2 = m->body_geomnum[bf2];
        contype2 = conaffinity2 = 0;
        for (int i=body_geomadr2; i < body_geomadr2+body_geomnum2; i++) {
          contype2 |= m->geom_contype[i];
          conaffinity2 |= m->geom_conaffinity[i];
        }
      } else {
        contype2 = m->flex_contype[bf2-nbody];
        conaffinity2 = m->flex_conaffinity[bf2-nbody];
      }

      // compatibility check
      if (!(contype1 & conaffinity2) && !(contype2 & conaffinity1)) {
        return;
      }
    }

    // add pair
    int n = (*npair)++;
    pair[n].hi = (bf1 < bf2) ? bf1 : bf2;
    pair[n].lo = (bf1 < bf2) ? bf2 : bf1;
  } else {
    mjERROR("broadphase buffer full");
  }
}


//----------------------------- general Sweep and Prune algorithm ----------------------------------

// helper structure for SAP sorting
struct _mjtSAP {
  float value;
  int id_ismax;
};
typedef struct _mjtSAP mjtSAP;



// comparison function for SAP
static inline int SAPcmp(mjtSAP* obj1, mjtSAP* obj2, void* context) {
  if (obj1->value < obj2->value) {
    return -1;
  } else if (obj1->value == obj2->value) {
    return (obj1->id_ismax & 0x10000) - (obj2->id_ismax & 0x10000);
  } else {
    return 1;
  }
}

// define SAPsort function for sorting SAP sorting
mjSORT(SAPsort, mjtSAP, SAPcmp);


// given list of axis-aligned bounding boxes in AAMM (min[3], max[3]) format,
// return number of pairs that can collide, using sweep-and-prune along
// specified x axis (0, 1 or 2).
static int mj_SAP(mjData* d, const mjtNum* aamm, int n, int axis_x, mjPacked32* pair, int maxpair) {
  // check inputs
  if (n >= 0x10000 || axis_x < 0 || axis_x > 2 || maxpair < 1) {
    return -1;
  }

  // allocate sort buffer
  mjtSAP* sortbuf = mjSTACKALLOC(d, 2*n, mjtSAP);    // sorted min and max values projected onto x axis
  mjtSAP* activebuf = mjSTACKALLOC(d, 2*n, mjtSAP);  // list of all AAMMs currently intersecting the sweep plane

  // get AAMM pointers for primary "x" axis
  const mjtNum* x_min = aamm + n*(axis_x + 0);
  const mjtNum* x_max = aamm + n*(axis_x + 3);

  // init sortbuf with specified axis
  for (int i=0; i < n; i++) {
    sortbuf[2*i].id_ismax = i;
    sortbuf[2*i].value = (float)x_min[i];
    sortbuf[2*i+1].id_ismax = i + 0x10000;
    sortbuf[2*i+1].value = (float)x_max[i];
  }

  // sort along specified axis
  mjtSAP* buf = mjSTACKALLOC(d, 2*n, mjtSAP);
  SAPsort(sortbuf, buf, 2*n, NULL);

  // define the other two axes
  int axis_y, axis_z;
  if (axis_x == 0) {
    axis_y = 1;
    axis_z = 2;
  } else if (axis_x == 1) {
    axis_y = 0;
    axis_z = 2;
  } else {
    axis_y = 0;
    axis_z = 1;
  }

  // get AAMM pointers to secondary "y, z" axes
  const mjtNum* y_min = aamm + n*(axis_y + 0);
  const mjtNum* y_max = aamm + n*(axis_y + 3);
  const mjtNum* z_min = aamm + n*(axis_z + 0);
  const mjtNum* z_max = aamm + n*(axis_z + 3);

  // sweep and prune
  int cnt = 0;    // size of active list
  int npair = 0;  // number of pairs added
  for (int i=0; i < 2*n; i++) {
    // min value: collide with all in list, add
    if (!(sortbuf[i].id_ismax & 0x10000)) {
      for (int j=0; j < cnt; j++) {
        // get ids: no need to mask ismax because activebuf entries never have the ismax bit,
        // and sortbuf[i].id_ismax is tested above
        int id1 = activebuf[j].id_ismax;
        int id2 = sortbuf[i].id_ismax;

        // use the other two axes to prune if possible
        if (y_min[id1] > y_max[id2] ||
            y_min[id2] > y_max[id1] ||
            z_min[id1] > z_max[id2] ||
            z_min[id2] > z_max[id1]) {
          continue;
        }

        // add pair, check buffer size
        pair[npair].hi = id1;
        pair[npair].lo = id2;
        npair++;
        if (npair >= maxpair) {
          return maxpair;
        }
      }

      // add to list
      activebuf[cnt] = sortbuf[i];
      cnt++;
    }

    // max value: remove corresponding min value from list
    else {
      int toremove = sortbuf[i].id_ismax & 0xFFFF;
      for (int j=0; j < cnt; j++) {
        if (activebuf[j].id_ismax == toremove) {
          if (j < cnt-1) {
            memmove(activebuf+j, activebuf+j+1, sizeof(mjtSAP)*(cnt-1-j));
          }
          cnt--;
          break;
        }
      }
    }
  }

  return npair;
}


// add vector to covariance
static void updateCov(mjtNum cov[9], const mjtNum vec[3], const mjtNum cen[3]) {
  mjtNum dif[3] = {vec[0]-cen[0], vec[1]-cen[1], vec[2]-cen[2]};
  mjtNum D00 = dif[0]*dif[0];
  mjtNum D01 = dif[0]*dif[1];
  mjtNum D02 = dif[0]*dif[2];
  mjtNum D11 = dif[1]*dif[1];
  mjtNum D12 = dif[1]*dif[2];
  mjtNum D22 = dif[2]*dif[2];
  cov[0] += D00;
  cov[1] += D01;
  cov[2] += D02;
  cov[3] += D01;
  cov[4] += D11;
  cov[5] += D12;
  cov[6] += D02;
  cov[7] += D12;
  cov[8] += D22;
}


// comparison function for packed 32-bit unsigned integers (lexicographical)
static inline int mjPacked32cmp(const mjPacked32* i, const mjPacked32* j, void* context) {
  if (i->hi < j->hi) return -1;
  if (i->hi > j->hi) return 1;
  if (i->lo < j->lo) return -1;
  if (i->lo > j->lo) return 1;
  return 0;
}

// define bfsort function for sorting bodyflex pairs
mjSORT(bfsort, mjPacked32, mjPacked32cmp);


// broadphase collision detector
static int mj_broadphase(const mjModel* m, mjData* d, mjPacked32* bfpair, int maxpair) {
  int npair = 0, nbody = m->nbody, ngeom = m->ngeom;
  int nvert = m->nflexvert, nflex = m->nflex, nbodyflex = m->nbody + m->nflex;
  int dsbl_filterparent = mjDISABLED(mjDSBL_FILTERPARENT);
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < nbody;
  mjtNum cov[9], cen[3], eigval[3], eigvec[9], frame[9], quat[4];

  // init with pairs involving always-colliding bodies
  for (int b1=0; b1 < nbody; b1++) {
    // cannot collide
    if (!canCollide(m, b1)) {
      continue;
    }

    // b1 is world body with geoms, or dof-less body with plane (static or mocap)
    if ((b1 == 0 && m->body_geomnum[b1] > 0) ||
        (m->body_dofnum[m->body_weldid[b1]] == 0 && hasPlane(m, b1))) {
      int weld1 = m->body_weldid[b1];
      int parent_weld1 = m->body_weldid[m->body_parentid[weld1]];
      int dofnum1 = m->body_dofnum[weld1];

      // add b1:b2 pairs that are not welded together
      for (int b2=0; b2 < nbody; b2++) {
        // cannot collide
        if (!canCollide(m, b2)) {
          continue;
        }

        int weld2 = m->body_weldid[b2];
        int parent_weld2 = m->body_weldid[m->body_parentid[weld2]];
        int asleep2 = sleep_filter ? d->body_awake[b2] == mjS_ASLEEP : 0;
        int dofnum2 = m->body_dofnum[weld2];
        if (filterBodyPair(weld1, parent_weld1, 0, dofnum1,
                           weld2, parent_weld2, asleep2, dofnum2,
                           dsbl_filterparent)) {
          continue;
        }

        // add pair
        add_pair(m, b1, b2, &npair, bfpair, maxpair);
      }

      // add body:flex pairs, skip if flex has no dofs (like b1) or is asleep
      for (int f=0; f < nflex; f++) {
        if (!hasAwakeDofs(m, d, nbody+f, sleep_filter)) continue;
        add_pair(m, b1, nbody+f, &npair, bfpair, maxpair);
      }
    }
  }

  // find center of non-world geoms and flex vertices; return if none
  int cnt = 0;
  mju_zero3(cen);
  for (int i=0; i < ngeom; i++) {
    if (m->geom_bodyid[i]) {
      mju_addTo3(cen, d->geom_xpos+3*i);
      cnt++;
    }
  }
  for (int i=0; i < nvert; i++) {
    if (m->flex_vertbodyid[i]) {
      mju_addTo3(cen, d->flexvert_xpos+3*i);
      cnt++;
    }
  }
  if (cnt == 0) {
    return npair;
  }
  mju_scl3(cen, cen, 1.0/cnt);

  // compute covariance
  mju_zero(cov, 9);
  for (int i=0; i < ngeom; i++) {
    if (m->geom_bodyid[i]) {
      updateCov(cov, d->geom_xpos+3*i, cen);
    }
  }
  for (int i=0; i < nvert; i++) {
    if (m->flex_vertbodyid[i]) {
      updateCov(cov, d->flexvert_xpos+3*i, cen);
    }
  }
  mju_scl(cov, cov, 1.0/cnt, 9);

  // construct covariance-aligned 3D frame: eigenvectors are columns of eigvec, rows of frame
  // SAP is exact in any frame, a rough one prunes as well
  mju_eig3Tol(eigval, eigvec, quat, cov, 1e-3);
  mji_transpose3(frame, eigvec);

  // allocate collidable bodyflex ids, construct list
  mj_markStack(d);
  int* bfid = mjSTACKALLOC(d, nbodyflex, int);
  int ncollide = 0;
  for (int i=1; i < nbodyflex; i++) {
    if (canCollide(m, i)) {
      bfid[ncollide++] = i;
    }
  }

  if (ncollide > 1) {
    // allocate and construct AAMMs for collidable only
    mjtNum* aamm = mjSTACKALLOC(d, 6*ncollide, mjtNum);
    for (int i=0; i < ncollide; i++) {
      // aamm is column-major (grouped per-axis rather than per-box for better cache locality)
      makeAAMM(m, d, aamm + 0*ncollide + i,
                     aamm + 1*ncollide + i,
                     aamm + 2*ncollide + i,
                     aamm + 3*ncollide + i,
                     aamm + 4*ncollide + i,
                     aamm + 5*ncollide + i, bfid[i], frame);
    }

    // call SAP
    int maxsappair = ncollide*(ncollide-1)/2;
    mjPacked32* sappair = mjSTACKALLOC(d, maxsappair, mjPacked32);
    int nsappair = mj_SAP(d, aamm, ncollide, 0, sappair, maxsappair);
    if (nsappair < 0) {
      mjERROR("SAP failed");
    }

    // filter SAP pairs, convert to bodyflex pairs
    for (int i=0; i < nsappair; i++) {
      int bf1 = bfid[sappair[i].hi];
      int bf2 = bfid[sappair[i].lo];

      // body pair: prune based on sleep filter and weld filter
      if (bf1 < nbody && bf2 < nbody) {
        int weld1 = m->body_weldid[bf1];
        int weld2 = m->body_weldid[bf2];
        int parent_weld1 = m->body_weldid[m->body_parentid[weld1]];
        int parent_weld2 = m->body_weldid[m->body_parentid[weld2]];
        int asleep1 = sleep_filter ? d->body_awake[bf1] == mjS_ASLEEP : 0;
        int asleep2 = sleep_filter ? d->body_awake[bf2] == mjS_ASLEEP : 0;
        int dofnum1 = m->body_dofnum[weld1];
        int dofnum2 = m->body_dofnum[weld2];
        if (filterBodyPair(weld1, parent_weld1, asleep1, dofnum1,
                           weld2, parent_weld2, asleep2, dofnum2,
                           dsbl_filterparent)) {
          continue;
        }
      }

      // flex pair: skip if neither side has awake dofs
      // (smaller id first: a body is cheaper to test than a flex)
      else if (!hasAwakeDofs(m, d, mjMIN(bf1, bf2), sleep_filter) &&
               !hasAwakeDofs(m, d, mjMAX(bf1, bf2), sleep_filter)) {
        continue;
      }

      // add bodyflex pair if there is room in buffer
      add_pair(m, bf1, bf2, &npair, bfpair, maxpair);
    }
  }

  // sort bodyflex pairs by signature
  if (npair > 1) {
    mjPacked32* buf = mjSTACKALLOC(d, npair, mjPacked32);
    bfsort(bfpair, buf, npair, NULL);
  }

  mj_freeStack(d);
  return npair;
}


//----------------------------- narrow-phase collision detection ----------------------------------


// set contact parameters
static void mj_setContact(const mjModel* m, const mjData* d, mjContact* con,
                          int condim, mjtNum includemargin,
                          const mjtNum* solref, const mjtNum* solreffriction,
                          const mjtNum* solimp, const mjtNum* friction,
                          mjtNum adhesion, int speculative) {
  // set parameters
  con->dim = condim;
  con->includemargin = includemargin;
  con->adhesion = adhesion;
  mj_assignRef(m, con->solref, solref);
  mj_assignRef(m, con->solreffriction, solreffriction);
  mj_assignImp(m, con->solimp, solimp);
  mj_assignFriction(m, con->friction, friction);

  // exclude in gap; adhesive and speculative contacts remain active when dist >= includemargin
  con->exclude = (con->dist >= includemargin) && !adhesion && !speculative;

  // in-gap adhesive contacts have no surface contact and therefore no
  // friction: reduce to normal-only rows (with tangential rows present, fast
  // slip would recruit normal force and brake tangential motion at a distance).
  // Speculative contacts represent a surface impact during this step (where
  // dist >= includemargin only indicates t >= 1 - b0), so they keep condim.
  if (adhesion && !speculative && con->dist >= includemargin) {
    con->dim = 1;
  }

  // complete frame
  mju_makeFrame(con->frame);

  // compute relative surface velocity in contact frame: (tangent1, tangent2, spin)
  mji_zero3(con->surfacevel);
  if (m->flg_surfacevel) {
    mjtNum svel[3] = {0, 0, 0}, sang[3] = {0, 0, 0};
    int active = 0;
    for (int side=0; side < 2; side++) {
      int g = con->geom[side];
      if (g < 0) {
        // TODO(team): support flex
        continue;
      }
      const mjtNum* sv = m->geom_surfacevel + 6*g;

      // skip geom with no surface velocity
      if (!sv[0] && !sv[1] && !sv[2] && !sv[3] && !sv[4] && !sv[5]) {
        continue;
      }
      active = 1;

      // rotate to world frame, add angular contribution at contact point
      mjtNum sgn = side ? 1 : -1;
      mjtNum vw[3], ww[3];
      mj_geomSurfaceVelocity(m, d, g, con->pos, vw, ww);
      mji_addToScl3(svel, vw, sgn);
      mji_addToScl3(sang, ww, sgn);
    }
    if (active) {
      con->surfacevel[0] = mju_dot3(con->frame + 3, svel);
      con->surfacevel[1] = mju_dot3(con->frame + 6, svel);
      con->surfacevel[2] = mju_dot3(con->frame + 0, sang);
    }
  }

  // clear fields that are computed later
  con->efc_address = -1;
  con->mu = 0;
  mju_zero(con->H, 36);

  // set deprecated fields
  con->geom1 = con->geom[0];
  con->geom2 = con->geom[1];
}


// struct for collision task
typedef struct {
  mjPreContact* conbuffer;  // pre-contact buffer returned by collision functions
  int* flexbuffer;          // buffer for flex vertex or element IDs
  int* nconbuffer;          // contact count for each collision pair
  char* epabuffer;          // buffer for nativeccd
  int ccd_size;             // size of nativeccd buffer
  mjcPair* pairbuffer;      // collision pairs
  int npair;                // number of collision pairs
  int chunksize;            // number of pairs to process per task
  int maxcon;               // maximum number of contacts (size of conbuffer)
} mjContactArg;


// return the maximum number of contacts for a collision pair
static inline int pairMaxContact(const mjModel* m, const mjcPair* pair) {
  switch (pair->type) {
    case mjCPAIR_GEOM_GEOM: {
      int g1 = pair->geom_geom.g1;
      int g2 = pair->geom_geom.g2;
      int ipair = pair->geom_geom.ipair;
      mjtNum margin = getMargin(m, g1, g2, ipair);
      mjtNum gap = getGap(m, g1, g2, ipair);
      return mj_maxContact(m, g1, g2, margin + gap > 0);
    }

    case mjCPAIR_GEOM_FLEX: {
      int g = pair->geom_flex.g;
      int f = pair->geom_flex.f;
      return (m->geom_type[g] == mjGEOM_PLANE) ? m->flex_vertnum[f] : mjMAXCONPAIR;
    }

    case mjCPAIR_GEOM_ELEM: {
      int g = pair->geom_elem.g;
      int f = pair->geom_elem.f;
      int dim = m->flex_dim[f];
      int type = m->geom_type[g];
      if (type == mjGEOM_HFIELD ||
          (dim == 2 && (type == mjGEOM_BOX || type == mjGEOM_CAPSULE))) {
        return mjMAXCONPAIR;
      }
      if (dim == 1 && (type == mjGEOM_CAPSULE || type == mjGEOM_BOX)) {
        return 2;
      }
      return 1;
    }

    case mjCPAIR_ELEM_ELEM: {
      int dim1 = m->flex_dim[pair->elem_elem.f1];
      int dim2 = m->flex_dim[pair->elem_elem.f2];
      return (dim1 == 1 && dim2 == 1) ? 2 : 1;
    }

    default:
      return 0;
  }
}


static void collisionTask(const mjModel* m, mjData* d, void* arg, int thread_id, int idx) {
  mjContactArg* conargs = (mjContactArg*)arg;
  mjPreContact* conbuffer = conargs->conbuffer;
  char* epabuffer = conargs->epabuffer;
  int chunksize = conargs->chunksize;
  int globalidx = chunksize * idx;
  mjcPair* pair = conargs->pairbuffer + globalidx;
  int* ncon = conargs->nconbuffer + chunksize * idx;

  int npair = conargs->npair;
  int n = mjMIN(chunksize, npair - globalidx);

  mjc_setCCDBuffer(epabuffer + thread_id * conargs->ccd_size);
  for (int i = 0; i < n; i++) {
    int conpos = pair[i].conpos;

    switch (pair[i].type) {
      case mjCPAIR_GEOM_GEOM: {
        int g1 = pair[i].geom_geom.g1;
        int g2 = pair[i].geom_geom.g2;
        int ipair = pair[i].geom_geom.ipair;
        mjfCollision collision_func = mjCOLLISIONFUNC[m->geom_type[g1]][m->geom_type[g2]];
        mjtNum margin = getMargin(m, g1, g2, ipair);
        mjtNum gap = getGap(m, g1, g2, ipair);
        ncon[i] = collision_func(m, d, conbuffer + conpos, g1, g2, margin + gap);
        if (speculativeEnabled(m)) {
          // check for active surface contact (in-gap contacts are only replaced if
          // CCD finds a surface hit)
          int active = 0;
          for (int k = 0; k < ncon[i]; k++) {
            if (conbuffer[conpos + k].dist < margin) {
              active = 1;
              break;
            }
          }
          mjtNum min_size = pairMinSize(m, g1, g2);
          if (!active && (geomStepSweep(m, d, g1, min_size, NULL, NULL) > 0 ||
                          geomStepSweep(m, d, g2, min_size, NULL, NULL) > 0)) {
            int nccd = mjc_advanceGeomPair(m, d, g1, g2, margin, conbuffer + conpos,
                                           epabuffer + thread_id * conargs->ccd_size);
            if (nccd > 0) {
              ncon[i] = nccd;
              pair[i].geom_geom.spec = 1;
            }
          }
        }
        break;
      }

      case mjCPAIR_GEOM_FLEX: {
        int g = pair[i].geom_flex.g;
        int f = pair[i].geom_flex.f;
        mjtNum margin = mj_assignMargin(m, m->geom_margin[g] + m->flex_margin[f]);
        mjtNum gap = m->geom_gap[g] + m->flex_gap[f];
        if (m->geom_type[g] == mjGEOM_PLANE) {
          ncon[i] = mjc_PlaneFlex(m, d, conbuffer + conpos, conargs->flexbuffer + conpos,
                                  g, f, margin + gap);
        } else if (m->geom_type[g] == mjGEOM_SDF) {
          ncon[i] = mjc_FlexSDF(m, d, conbuffer + conpos, conargs->flexbuffer + conpos,
                                g, f, margin + gap);
        } else {
          ncon[i] = 0;
        }
        break;
      }

      case mjCPAIR_GEOM_ELEM: {
        int g = pair[i].geom_elem.g;
        int f = pair[i].geom_elem.f;
        int e = pair[i].geom_elem.e;
        mjtNum margin = mj_assignMargin(m, m->geom_margin[g] + m->flex_margin[f]);
        mjtNum gap = m->geom_gap[g] + m->flex_gap[f];
        ncon[i] = mjc_GeomElem(m, d, conbuffer + conpos, g, f, e, margin + gap);
        break;
      }

      case mjCPAIR_ELEM_ELEM: {
        int f1 = pair[i].elem_elem.f1;
        int e1 = pair[i].elem_elem.e1;
        int f2 = pair[i].elem_elem.f2;
        int e2 = pair[i].elem_elem.e2;
        mjtNum margin = mj_assignMargin(m, m->flex_margin[f1] + m->flex_margin[f2]);
        mjtNum gap = m->flex_gap[f1] + m->flex_gap[f2];
        if (f1 == f2) {
          margin = 0;
        }
        ncon[i] = mjc_ElemElem(m, d, conbuffer + conpos, f1, e1, f2, e2, margin + gap);
        break;
      }

      default:
        ncon[i] = 0;
        break;
    }

    // SHOULD NOT OCCUR
    int expected_max = (globalidx + i + 1 < npair ? pair[i+1].conpos : conargs->maxcon) - conpos;
    if (ncon[i] > expected_max) {
      mjERROR("collision function returned %d contacts, "
              "expected at most %d from pairMaxContact", ncon[i], expected_max);
    }
  }
  mjc_setCCDBuffer(NULL);
}


// populate contacts from mjPreContact array; allocate in arena
static void addPairContacts(const mjModel* m, mjData* d, const mjPreContact* precon, int ncon,
                            const mjcPair* pair, const int* elem, const int* vert) {
  if (!ncon) {
    return;
  }

  mjContact* con = (mjContact*) mj_arenaAllocByte(d, sizeof(mjContact) * ncon, _Alignof(mjContact));
  if (!con) {
    mj_warning(d, mjWARN_CONTACTFULL, d->ncon);
    return;
  }
  d->ncon += ncon;

  int condim = 0;
  mjtNum margin = 0;
  mjtNum friction[5] = {0}, solref[mjNREF] = {0}, solimp[mjNIMP] = {0}, adhesion = 0;
  mjtNum solreffriction[mjNREF] = {0};

  int g1 = -1, g2 = -1, f1 = -1, f2 = -1;
  int e1 = -1, e2 = -1, v1 = -1, v2 = -1;
  int spec = 0;

  switch (pair->type) {
    case mjCPAIR_GEOM_GEOM: {
      g1 = pair->geom_geom.g1;
      g2 = pair->geom_geom.g2;
      int ipair = pair->geom_geom.ipair;
      spec = pair->geom_geom.spec;
      margin = getMargin(m, g1, g2, ipair);
      if (ipair >= 0) {
        condim = m->pair_dim[ipair];
        mju_copy(solref, m->pair_solref + mjNREF * ipair, mjNREF);
        mju_copy(solimp, m->pair_solimp + mjNIMP * ipair, mjNIMP);
        mju_copy(friction, m->pair_friction + 5 * ipair, 5);
        adhesion = m->pair_adhesion[ipair];
        if (m->pair_solreffriction[mjNREF * ipair] || m->pair_solreffriction[mjNREF * ipair + 1]) {
          mju_copy(solreffriction, m->pair_solreffriction + mjNREF * ipair, mjNREF);
        }
      } else {
        mj_contactParam(m, &condim, solref, solimp, friction, &adhesion, g1, g2, -1, -1);
      }
      break;
    }

    case mjCPAIR_GEOM_FLEX:
      g1 = pair->geom_flex.g;
      f2 = pair->geom_flex.f;
      margin = mj_assignMargin(m, m->geom_margin[g1] + m->flex_margin[f2]);
      mj_contactParam(m, &condim, solref, solimp, friction, &adhesion, g1, -1, -1, f2);
      break;

    case mjCPAIR_GEOM_ELEM:
      g1 = pair->geom_elem.g;
      f2 = pair->geom_elem.f;
      e2 = pair->geom_elem.e;
      margin = mj_assignMargin(m, m->geom_margin[g1] + m->flex_margin[f2]);
      mj_contactParam(m, &condim, solref, solimp, friction, &adhesion, g1, -1, -1, f2);
      break;

    case mjCPAIR_ELEM_ELEM:
      f1 = pair->elem_elem.f1;
      e1 = pair->elem_elem.e1;
      f2 = pair->elem_elem.f2;
      e2 = pair->elem_elem.e2;
      margin = (f1 == f2) ? 0 : mj_assignMargin(m, m->flex_margin[f1] + m->flex_margin[f2]);
      mj_contactParam(m, &condim, solref, solimp, friction, &adhesion, -1, -1, f1, f2);
      break;
  }

  for (int i = 0; i < ncon; i++) {
    con[i].dist = precon[i].dist;
    mji_copy3(con[i].pos, precon[i].pos);
    mji_copy3(con[i].frame + 0, precon[i].normal);
    mji_copy3(con[i].frame + 3, precon[i].tangent);

    con[i].geom[0] = g1;
    con[i].geom[1] = g2;
    con[i].flex[0] = f1;
    con[i].flex[1] = f2;
    con[i].elem[0] = e1;
    con[i].elem[1] = elem ? elem[i] : e2;
    con[i].vert[0] = v1;
    con[i].vert[1] = vert ? vert[i] : v2;

    mj_setContact(m, d, con + i, condim, margin, solref, solreffriction, solimp, friction,
                  adhesion, spec);
  }
}


// filter precontacts across a group of collision pairs down to mjMAXCONPAIR
// the pairs of a group are contiguous, the first one is at start
// removes unselected precontacts in-place, resets group to -1, and updates nconbuffer
static inline void filterPreContacts(mjData* d, mjContactArg* arg, int start, int group) {
  // find the end of the group, reset group and count precontacts
  int end = start;
  int count = 0;
  while (end < arg->npair && arg->pairbuffer[end].group == group) {
    arg->pairbuffer[end].group = -1;
    count += arg->nconbuffer[end];
    end++;
  }

  // if total contacts <= mjMAXCONPAIR, return
  if (count <= mjMAXCONPAIR) {
    return;
  }

  mj_markStack(d);
  mjtByte* selected = mjSTACKALLOC(d, count, mjtByte);
  mjtNum* min_dist = mjSTACKALLOC(d, count, mjtNum);
  memset(selected, 0, count);
  for (int i = 0; i < count; i++) {
    min_dist[i] = mjMAXVAL;
  }

  // start with deepest penetrating pre-contact (skip uninitialized pre-contacts)
  int best = -1, best_k = -1, best_c = -1;
  mjtNum bestdist = -mjMAXVAL;
  int idx = 0;
  for (int k = start; k < end; k++) {
    int conpos = arg->pairbuffer[k].conpos;
    for (int c = 0; c < arg->nconbuffer[k]; c++, idx++) {
      mjtNum dist = -arg->conbuffer[conpos + c].dist;
      if (dist > bestdist) {
        bestdist = dist;
        best = idx;
        best_k = k;
        best_c = c;
      }
    }
  }

  // greedy furthest-point selection
  int nselected = 0;
  while (nselected < mjMAXCONPAIR && best >= 0) {
    selected[best] = 1;
    const mjtNum* bestpos = arg->conbuffer[arg->pairbuffer[best_k].conpos + best_c].pos;

    int nextbest = -1, nextbest_k = -1, nextbest_c = -1;
    mjtNum nextbestdist = -1;
    idx = 0;
    for (int k = start; k < end; k++) {
      int conpos = arg->pairbuffer[k].conpos;
      for (int c = 0; c < arg->nconbuffer[k]; c++, idx++) {
        if (selected[idx]) {
          continue;
        }

        const mjtNum* pos = arg->conbuffer[conpos + c].pos;
        mjtNum dx = pos[0] - bestpos[0];
        mjtNum dy = pos[1] - bestpos[1];
        mjtNum dz = pos[2] - bestpos[2];
        mjtNum d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < min_dist[idx]) {
          min_dist[idx] = d2;
        }
        if (min_dist[idx] > nextbestdist) {
          nextbestdist = min_dist[idx];
          nextbest = idx;
          nextbest_k = k;
          nextbest_c = c;
        }
      }
    }

    nselected++;
    best = nextbest;
    best_k = nextbest_k;
    best_c = nextbest_c;
  }

  // compact selected precontacts in-place and update nconbuffer
  idx = 0;
  for (int k = start; k < end; k++) {
    int conpos = arg->pairbuffer[k].conpos;
    int new_ncon = 0;
    for (int c = 0; c < arg->nconbuffer[k]; c++, idx++) {
      if (selected[idx]) {
        if (new_ncon != c) {
          arg->conbuffer[conpos + new_ncon] = arg->conbuffer[conpos + c];
          if (arg->flexbuffer) {
            arg->flexbuffer[conpos + new_ncon] = arg->flexbuffer[conpos + c];
          }
        }
        new_ncon++;
      }
    }
    arg->nconbuffer[k] = new_ncon;
  }

  mj_freeStack(d);
}


// compute contacts for a batch of collision pairs on the arena starting at parena
static void mj_narrowphase(const mjModel* m, mjData* d, int npair, size_t parena) {
  int nthread = mju_numThread(d);
  int npolygonmax = mjDISABLED(mjDSBL_MULTICCD) ? 0 : m->npolygonmax;
  int nmeshdegmax = mjDISABLED(mjDSBL_MULTICCD) ? 0 : m->nmeshdegmax;
  int ccd_size = mjc_ccdSize(npolygonmax, nmeshdegmax, m->opt.ccd_iterations);
  // try to balance load of 5 chunks per thread
  int chunksize = (nthread == 1) ? npair : mjMAX(1, npair / (5 * nthread));

  int nchunk = (npair + chunksize - 1) / chunksize;

  const mjcPair* buffer = (const mjcPair*) ((char*) d->arena + parena);

  mj_markStack(d);

  // buffer for pair data
  mjcPair* pairbuffer = mjSTACKALLOC(d, npair, mjcPair);
  int maxcon = 0;
  int has_flex = 0;
  for (int i = 0; i < npair; i++) {
    pairbuffer[i] = buffer[i];
    pairbuffer[i].conpos = maxcon;
    maxcon += pairMaxContact(m, buffer + i);
    if (buffer[i].type == mjCPAIR_GEOM_FLEX) {
      has_flex = 1;
    }
  }

  // buffer data has been copied to metadata on the stack;
  // reclaim arena space so contacts can overwrite the buffer region
  d->parena = parena;

  mjContactArg arg;
  arg.ccd_size = ccd_size;
  arg.pairbuffer = pairbuffer;
  arg.nconbuffer = mjSTACKALLOC(d, npair, int);
  arg.conbuffer = mjSTACKALLOC(d, maxcon, mjPreContact);
  arg.flexbuffer = has_flex ? mjSTACKALLOC(d, maxcon, int) : NULL;
  arg.npair = npair;
  arg.chunksize = chunksize;
  arg.maxcon = maxcon;

  // dispatch narrowphase to threads with local stack allocation for EPA
  {
    mj_markStack(d);
    int nactive = (nchunk < 2) ? 1 : nthread;
    arg.epabuffer = mj_stackAllocByte(d, ccd_size * nactive, sizeof(mjtNum));
    mju_dispatch(m, d, collisionTask, &arg, nchunk);
    mj_freeStack(d);
  }

  // fill in contact data
  int lastgroup = -1;
  for (int i = 0; i < npair; i++) {
    if (pairbuffer[i].group >= 0) {
      // groups are contiguous and increasing: SHOULD NOT OCCUR
      if (pairbuffer[i].group <= lastgroup) {
        mjERROR("filter group %d is not contiguous", pairbuffer[i].group);
      }
      lastgroup = pairbuffer[i].group;
      filterPreContacts(d, &arg, i, pairbuffer[i].group);
    }

    int ncon = arg.nconbuffer[i];
    if (!ncon) {
      continue;
    }

    const int* vert = (pairbuffer[i].type == mjCPAIR_GEOM_FLEX &&
                       m->geom_type[pairbuffer[i].geom_flex.g] == mjGEOM_PLANE)
                      ? arg.flexbuffer + pairbuffer[i].conpos : NULL;
    const int* elem = (pairbuffer[i].type == mjCPAIR_GEOM_FLEX &&
                       m->geom_type[pairbuffer[i].geom_flex.g] == mjGEOM_SDF)
                      ? arg.flexbuffer + pairbuffer[i].conpos : NULL;
    addPairContacts(m, d, arg.conbuffer + pairbuffer[i].conpos, ncon,
                    pairbuffer + i, elem, vert);
  }
  mj_freeStack(d);
}


// test active element self-collisions with SAP, pushing pairs onto the arena
static int mj_collideFlexSAPPairs(const mjModel* m, mjData* d, int f, int group, int npair) {
  mj_markStack(d);

  // allocate and construct active element ids
  int* elid = mjSTACKALLOC(d, m->flex_elemnum[f], int);
  int nactive = 0;
  int flex_elemnum = m->flex_elemnum[f];
  for (int i=0; i < flex_elemnum; i++) {
    if (mj_isElemActive(m, f, i)) {
      elid[nactive++] = i;
    }
  }

  // nothing active
  if (nactive < 2) {
    mj_freeStack(d);
    return npair;
  }

  // allocate and construct AAMMs for active elements
  mjtNum* aamm = mjSTACKALLOC(d, 6*nactive, mjtNum);
  const mjtNum* elemaabb = d->flexelem_aabb + 6*m->flex_elemadr[f];
  for (int i=0; i < nactive; i++) {
    const mjtNum* center = elemaabb+6*elid[i];
    const mjtNum* radius = elemaabb+6*elid[i]+3;
    aamm[nactive*0 + i] = center[0] - radius[0];
    aamm[nactive*1 + i] = center[1] - radius[1];
    aamm[nactive*2 + i] = center[2] - radius[2];
    aamm[nactive*3 + i] = center[0] + radius[0];
    aamm[nactive*4 + i] = center[1] + radius[1];
    aamm[nactive*5 + i] = center[2] + radius[2];
  }

  // select largest axis from flex bvh
  const mjtNum* bvh = d->bvh_aabb_dyn + 6*(m->flex_bvhadr[f] - m->nbvhstatic);
  int axis = (bvh[3] > bvh[4] && bvh[3] > bvh[5]) ? 0 : (bvh[4] > bvh[5] ? 1 : 2);

  // call SAP; hard limit on number of pairs to avoid out-of-memory
  int maxsappair = mjMIN(nactive*(nactive-1)/2, 1000000);
  mjPacked32* sappair = mjSTACKALLOC(d, maxsappair, mjPacked32);
  int nsappair = mj_SAP(d, aamm, nactive, axis, sappair, maxsappair);
  if (nsappair < 0) {
    mjERROR("SAP failed");
  }

  // push SAP pairs
  for (int i=0; i < nsappair; i++) {
    int e1 = elid[sappair[i].hi];
    int e2 = elid[sappair[i].lo];
    npair = pushSelfElemElem(m, d, f, e1, e2, group, npair);
  }

  mj_freeStack(d);
  return npair;
}


// test active element self-collisions of an interpolated flex by grouping elements into cells
static int mj_collideFlexCellPairs(const mjModel* m, mjData* d, int f, int group,
                                   size_t parena, int npair) {
  const int* cellnum = m->flex_cellnum + 3*f;
  int nx = cellnum[0], ny = cellnum[1], nz = cellnum[2];
  int ncell = nx * ny * nz;
  int nvert = m->flex_vertnum[f];
  int nelem = m->flex_elemnum[f];
  int dim = m->flex_dim[f];
  if (nelem < 2 || ncell < 1) {
    return npair;
  }

  mj_markStack(d);
  int start_pair = npair;

  // compute cell range [lo, hi] once per vertex
  int* vert_lo = mjSTACKALLOC(d, 3*nvert, int);
  int* vert_hi = mjSTACKALLOC(d, 3*nvert, int);
  for (int v = 0; v < nvert; v++) {
    mjc_vertCellRange(m, f, v, vert_lo + 3*v, vert_hi + 3*v);
  }

  // assign each active element to the lo-cell of its first vertex, and record its [elo, ehi] range
  int* elem_cell = mjSTACKALLOC(d, nelem, int);
  int* elem_lo = mjSTACKALLOC(d, 3*nelem, int);
  int* elem_hi = mjSTACKALLOC(d, 3*nelem, int);
  int* cell_count = mjSTACKALLOC(d, ncell, int);
  mju_zeroInt(cell_count, ncell);

  const int* elemdata = m->flex_elem + m->flex_elemdataadr[f];
  int nactive = 0;
  for (int e = 0; e < nelem; e++) {
    if (!mj_isElemActive(m, f, e)) {
      elem_cell[e] = -1;
      continue;
    }
    const int* edata = elemdata + e*(dim + 1);
    int v0 = edata[0];
    int elo[3], ehi[3];
    for (int k = 0; k < 3; k++) {
      elo[k] = vert_lo[3*v0 + k];
      ehi[k] = vert_hi[3*v0 + k];
    }
    int c = (elo[0]*ny + elo[1])*nz + elo[2];
    for (int i = 1; i <= dim; i++) {
      int v = edata[i];
      for (int k = 0; k < 3; k++) {
        elo[k] = mjMIN(elo[k], vert_lo[3*v + k]);
        ehi[k] = mjMAX(ehi[k], vert_hi[3*v + k]);
      }
    }
    elem_cell[e] = c;
    for (int k = 0; k < 3; k++) {
      elem_lo[3*e + k] = elo[k];
      elem_hi[3*e + k] = ehi[k];
    }
    cell_count[c]++;
    nactive++;
  }

  if (nactive < 2) {
    mj_freeStack(d);
    return npair;
  }

  // compact non-empty cells and build CSR offsets into cell_elems
  int nnonempty = 0;
  for (int c = 0; c < ncell; c++) {
    if (cell_count[c] > 0) {
      nnonempty++;
    }
  }
  int* cell_slot = mjSTACKALLOC(d, ncell, int);
  int* cell_offset = mjSTACKALLOC(d, nnonempty + 1, int);
  int* cell_cursor = mjSTACKALLOC(d, nnonempty, int);
  int* cell_coord = mjSTACKALLOC(d, 3*nnonempty, int);
  int* cell_lo = mjSTACKALLOC(d, 3*nnonempty, int);
  int* cell_hi = mjSTACKALLOC(d, 3*nnonempty, int);
  mjtNum* cell_min = mjSTACKALLOC(d, 3*nnonempty, mjtNum);
  mjtNum* cell_max = mjSTACKALLOC(d, 3*nnonempty, mjtNum);
  mjtNum* cell_aabb = mjSTACKALLOC(d, 6*nnonempty, mjtNum);

  int offset = 0, slot = 0;
  for (int c = 0; c < ncell; c++) {
    if (cell_count[c] > 0) {
      cell_slot[c] = slot;
      cell_offset[slot] = offset;
      cell_cursor[slot] = offset;
      offset += cell_count[c];

      int cx = c / (ny * nz);
      int cy = (c / nz) % ny;
      int cz = c % nz;
      cell_coord[3*slot + 0] = cx;
      cell_coord[3*slot + 1] = cy;
      cell_coord[3*slot + 2] = cz;
      cell_lo[3*slot + 0] = cx;
      cell_lo[3*slot + 1] = cy;
      cell_lo[3*slot + 2] = cz;
      cell_hi[3*slot + 0] = cx;
      cell_hi[3*slot + 1] = cy;
      cell_hi[3*slot + 2] = cz;
      for (int k = 0; k < 3; k++) {
        cell_min[3*slot + k] = mjMAXVAL;
        cell_max[3*slot + k] = -mjMAXVAL;
      }
      slot++;
    } else {
      cell_slot[c] = -1;
    }
  }
  cell_offset[nnonempty] = offset;

  // populate cell_elems, cell_lo/hi, and cell bounding boxes in a single O(nelem) pass
  int* cell_elems = mjSTACKALLOC(d, nactive, int);
  const mjtNum* elemaabb = d->flexelem_aabb + 6*m->flex_elemadr[f];
  for (int e = 0; e < nelem; e++) {
    int c = elem_cell[e];
    if (c < 0) {
      continue;
    }
    int s = cell_slot[c];
    cell_elems[cell_cursor[s]++] = e;
    for (int k = 0; k < 3; k++) {
      cell_lo[3*s + k] = mjMIN(cell_lo[3*s + k], elem_lo[3*e + k]);
      cell_hi[3*s + k] = mjMAX(cell_hi[3*s + k], elem_hi[3*e + k]);
      mjtNum center = elemaabb[6*e + k];
      mjtNum radius = elemaabb[6*e + 3 + k];
      cell_min[3*s + k] = mju_min(cell_min[3*s + k], center - radius);
      cell_max[3*s + k] = mju_max(cell_max[3*s + k], center + radius);
    }
  }

  for (int s = 0; s < nnonempty; s++) {
    for (int k = 0; k < 3; k++) {
      cell_aabb[6*s + k] = 0.5 * (cell_max[3*s + k] + cell_min[3*s + k]);
      cell_aabb[6*s + 3 + k] = 0.5 * (cell_max[3*s + k] - cell_min[3*s + k]);
    }
  }

  // test pairs of non-empty cells (interpolated cages are coarse in practice, typically 3^3 to 6^3)
  for (int s1 = 0; s1 < nnonempty; s1++) {
    const int* coord1 = cell_coord + 3*s1;
    const int* lo1 = cell_lo + 3*s1;
    const int* hi1 = cell_hi + 3*s1;
    const mjtNum* aabb1 = cell_aabb + 6*s1;
    int start1 = cell_offset[s1], end1 = cell_offset[s1 + 1];

    // elements in the same cell s1 share a node via their first vertex (both have vert_lo == coord1)
    for (int s2 = s1 + 1; s2 < nnonempty; s2++) {
      const int* coord2 = cell_coord + 3*s2;

      // if first-vertex cells are neighbors (|coord1 - coord2|_inf <= 1), every pair is excluded
      if (mjc_cellRangeShared(coord1, coord1, coord2, coord2)) {
        continue;
      }

      // skip if cell AABBs do not overlap
      if (filterBox(aabb1, cell_aabb + 6*s2, 0)) {
        continue;
      }

      // if cell ranges are separated by > 1 along some axis, no pair in (s1, s2) shares a node
      const int* lo2 = cell_lo + 3*s2;
      const int* hi2 = cell_hi + 3*s2;
      int disjoint = !mjc_cellRangeShared(lo1, hi1, lo2, hi2);

      int start2 = cell_offset[s2], end2 = cell_offset[s2 + 1];
      for (int i1 = start1; i1 < end1; i1++) {
        int e1 = cell_elems[i1];
        const mjtNum* eaabb1 = elemaabb + 6*e1;
        for (int i2 = start2; i2 < end2; i2++) {
          int e2 = cell_elems[i2];
          if (!filterBox(eaabb1, elemaabb + 6*e2, 0)) {
            int emin = mjMIN(e1, e2);
            int emax = mjMAX(e1, e2);
            if (!disjoint) {
              // check if any vertex of e1 and any vertex of e2 share a cell node
              const int* edata1 = elemdata + e1*(dim + 1);
              const int* edata2 = elemdata + e2*(dim + 1);
              int shared = 0;
              for (int v1 = 0; v1 <= dim && !shared; v1++) {
                const int* vlo1 = vert_lo + 3*edata1[v1];
                const int* vhi1 = vert_hi + 3*edata1[v1];
                for (int v2 = 0; v2 <= dim; v2++) {
                  const int* vlo2 = vert_lo + 3*edata2[v2];
                  const int* vhi2 = vert_hi + 3*edata2[v2];
                  if (mjc_cellRangeShared(vlo1, vhi1, vlo2, vhi2)) {
                    shared = 1;
                    break;
                  }
                }
              }
              if (shared) {
                continue;
              }
            }
            npair = pushElemElem(d, f, emin, f, emax, group, npair);
          }
        }
      }
    }
  }

  // sort pairs generated by this flex self-collision
  int n = npair - start_pair;
  if (n > 1) {
    mjcPair* buf = mjSTACKALLOC(d, n, mjcPair);
    mjcPair* pairs = (mjcPair*) ((char*) d->arena + parena);
    pairSort(pairs + start_pair, buf, n, (void*)m);
  }

  mj_freeStack(d);
  return npair;
}
