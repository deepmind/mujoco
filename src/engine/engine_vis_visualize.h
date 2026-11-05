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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_VIS_VISUALIZE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_VIS_VISUALIZE_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>

#ifdef __cplusplus
extern "C" {
#endif

#define mjMAXCURVE 64  // maximum number of points along a curve (catenary or geodesic)

// set (type, size, pos, mat) connector-type geom between given points
//  assume that mjv_initGeom was already called to set all other properties
MJAPI void mjv_connector(mjvGeom* geom, int type, mjtNum width,
                         const mjtNum from[3], const mjtNum to[3]);

// initialize given fields when not NULL, set the rest to their default values
MJAPI void mjv_initGeom(mjvGeom* geom, int type, const mjtNum* size,
                        const mjtNum* pos, const mjtNum* mat, const float* rgba);

// update entire scene
MJAPI void mjv_updateScene(const mjModel* m, mjData* d, const mjvOption* opt,
                           const mjvPerturb* pert, mjvCamera* cam, int catmask, mjvScene* scn);

// add geoms from selected categories to existing scene
MJAPI void mjv_addGeoms(const mjModel* m, mjData* d, const mjvOption* opt,
                        const mjvPerturb* pert, int catmask, mjvScene* scn);

// make list of lights only
MJAPI void mjv_makeLights(const mjModel* m, const mjData* d, mjvScene* scn);

// update camera only
MJAPI void mjv_updateCamera(const mjModel* m, const mjData* d, mjvCamera* cam, mjvScene* scn);

// update visible flexes only
MJAPI void mjv_updateActiveFlex(const mjModel* m, mjData* d, mjvScene* scn, const mjvOption* opt);

// update skins only
MJAPI void mjv_updateSkin(const mjModel* m, const mjData* d, mjvScene* scn);

// update visible skins only
MJAPI void mjv_updateActiveSkin(const mjModel* m, const mjData* d, mjvScene* scn,
                                const mjvOption* opt);

// computes the camera position, forward, up, and right vectors
MJAPI void mjv_cameraFrame(mjtNum headpos[3], mjtNum forward[3], mjtNum up[3], mjtNum right[3],
                           const mjData* d, const mjvCamera* cam);

// computes the camera frustums, i.e. vertical, horizontal, and clip planes
MJAPI void mjv_cameraFrustum(float zver[2], float zhor[2], float zclip[2],  const mjModel* m,
                             const mjvCamera* cam);

// determines if tendon should be rendered as a catenary curve and, if so, computes its length
MJAPI int mjv_isCatenary(const mjModel* m, const mjData* d, int i, mjtNum* length);

// computes points along a catenary curve
MJAPI int mjv_catenary(const mjtNum x0[3], const mjtNum x1[3], const mjtNum gravity[3],
                       mjtNum length, mjtNum* catenary, int ncatenary);

// computes points along a wrapped geodesic curve on a sphere or cylinder
MJAPI int mjv_geodesic(mjtNum* pts, int npts, const mjtNum xpos[3], const mjtNum xmat[9],
                       mjtNum radius, int type, const mjtNum wpnt[12]);

// convert HSV to RGB
MJAPI void mjv_hsv2rgb(float *RGB, float H, float S, float V);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_VIS_VISUALIZE_H_
