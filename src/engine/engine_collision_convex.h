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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_CONVEX_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_CONVEX_H_

// libCCD has an unconditional `#define _CRT_SECURE_NO_WARNINGS` on Windows.
// TODO(stunya): Remove once https://github.com/danfis/libccd/pull/77 is merged
#ifdef _CRT_SECURE_NO_WARNINGS
#undef _CRT_SECURE_NO_WARNINGS
#endif

#include <ccd/vec3.h>

#include <mujoco/mjexport.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

// minimum number of vertices to use hill-climbing in mesh support
#define mjMESH_HILLCLIMB_MIN 10

#ifdef __cplusplus
extern "C" {
#endif

// internal object type for convex collision detection
struct _mjCCDObj {
  int geom;
  int geom_type;
  mjtNum size[4];
  mjtNum pos[3];
  mjtNum mat[9];
  int vertindex;
  int meshindex;
  int flex;
  int elem;
  int vert;
  mjtNum margin;
  mjtNum rotate[4];
  mjtNum suppvert[12];
  int suppind[4];
  int nsupp;
  union {
    // mesh data from mjModel
    struct {
      int nvert;
      int mesh_polynum;
      const float* vert;
      const int* mpolymapadr;
      const int* mpolymapnum;
      const int* polymap;
      const int* polyvertadr;
      const int* polyvertnum;
      const int* polyvert;
      const mjtNum* polynormal;
      const int* graph;
      const int* extrema;
    } mesh;

    // hfield prism data
    struct {
      mjtNum prism[6][3];
      const float* hfield_data;
      int hfield_nrow;
      int hfield_ncol;
    } hfield;

    // flex data from mjModel and mjData
    struct {
      const int* elem;
      const int* dim;
      const mjtNum* aabb;
      const int* elemadr;
      const int* elemdataadr;
      const mjtNum* vert_xpos;
      const int* vertadr;
      const mjtNum* xradius;
    } flex;
  } data;
  void (*center)(mjtNum res[3], const struct _mjCCDObj* obj);
  void (*support)(mjtNum res[3], struct _mjCCDObj* obj, const mjtNum dir[3]);
};
typedef struct _mjCCDObj mjCCDObj;

// initialize a CCD object
MJAPI void mjc_initCCDObj(mjCCDObj* obj, const mjModel* m, const mjData* d, int g, mjtNum margin);

// initialize a CCD object from a site
MJAPI void mjc_initCCDObjSite(mjCCDObj* obj, const mjModel* m, const mjData* d, int s,
                              mjtNum margin);

// center function for convex collision algorithms
MJAPI void mjc_center(mjtNum res[3], const mjCCDObj *obj);

// libccd center function
MJAPI void mjccd_center(const void *obj, ccd_vec3_t *center);

// libccd support function
MJAPI void mjccd_support(const void *obj, const ccd_vec3_t *dir, ccd_vec3_t *vec);

// support function for point
void mjc_pointSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]);

// support function for line (capsule)
void mjc_lineSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]);

// pairwise geom collision functions using ccd
int mjc_PlaneConvex(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin);
int mjc_ConvexHField(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin);
MJAPI int mjc_Convex(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin);

// geom-elem or elem-elem or vert-elem collision function using ccd
int mjc_ConvexElem(const mjModel* m, mjData* d, mjPreContact* con, int g1, int f1, int e1, int v1,
                   int f2, int e2, mjtNum margin);

// heightfield-elem collision function using ccd
int mjc_HFieldElem(const mjModel* m, mjData* d, mjPreContact* con, int g, int f, int e,
                   mjtNum margin);

// fix contact frame normal
void mjc_fixNormal(const mjModel* m, const mjData* d, mjPreContact* con, int g1, int g2);

// set CCD internal buffer
void mjc_setCCDBuffer(void* buffer);

// returns approximation (lower bound) of directed Hausdorff distance between two
// compact convex geoms; if distance is positive then obj1 is guaranteed to not be enclosed in obj2
MJAPI mjtNum mjc_hausdorff(mjCCDObj* obj1, mjCCDObj* obj2, int nitermax,
                           mjtNum stepsize, mjtNum tolerance);

#ifdef __cplusplus
}
#endif
#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_CONVEX_H_
