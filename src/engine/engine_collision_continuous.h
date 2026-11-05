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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_CONTINUOUS_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_CONTINUOUS_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// Continuous collision for deformables. The discrete pipeline (engine_collision_*) generates
// contact points at a configuration; this module prices gaps along trajectories: differentiable
// vertex-triangle / edge-edge / vertex-geom distance kernels with closest-point barycentrics,
// swept-volume candidate generation over the flex BVH, per-pair gap evaluation with the gradient's
// vertex weights, and a conservative advancement (CCD) that bounds each pair's time of impact.
// A consumer supplies two lengths:
//   cap   -- the standoff ceiling: a pair's rest gap is min(band, cap), so thin participants keep
//            a proportionally thin skin and thick ones do not carry a fat layer;
//   band  -- the detection reach: how early the broad phase starts tracking a pair. Not a physical
//            length; wider costs candidates, narrower hands pairs to the solver later.
// ALL pair types measure their gap at the flex MIDSURFACE: mjc_pairGap does not subtract the flex
// radii, because where the geometry is tighter than the radii (a string threaded through a hem, a
// liner resting on its bin) a skin-to-skin gap is permanently negative and the pair would be
// discarded as invalid -- no CCD coverage, so the region could tunnel. For types 2/3/4 the gap is
// flex-midsurface to geom SURFACE (a geom's surface is its geometry; it has no radius to fold).
// The broad phase adds the flex radii back into its reach so detection range is unchanged (see
// addCand).

// Pair types for continuous collision. A geom's features are its box corners and edges, its mesh
// vertices and hull edges, a sphere's centre and a capsule's axis endpoints and axis; the last
// three stand the geom's radius inside its surface, which the gap subtracts.
typedef enum {
  mjcFLEX_VERT_TRI   = 0,  // flex vertex vs flex triangle
  mjcFLEX_EDGE_EDGE  = 1,  // flex edge vs flex edge
  mjcFLEX_VERT_GEOM  = 2,  // flex vertex vs geom surface
  mjcGEOM_CORNER_TRI = 3,  // geom corner feature vs flex triangle
  mjcGEOM_EDGE_EDGE  = 4,  // geom edge feature vs flex edge
} mjcFlexPairType;

// One candidate contact pair: the geometric identity only.
// idx holds free-point indices (types 0/1: all four; type 2: idx[0];
// types 3/4: the feature index in idx[0], then the flex-side points), g the geom for types 2-4.
// Solver state (multipliers, ages) and any cached linearization of the gap belong to the
// consumer, not to this struct.
typedef struct {
  mjcFlexPairType type;  // pair type
  int idx[4];  // participant free-point indices, meaning per type (see mjc_pairGap)
  int g;       // geom id, types 2-4 only
} mjcFlexPair;

// the standoff of a pair whose detection band is `band`: min(band, cap)
mjtNum mjc_standoff(mjtNum band, mjtNum cap);

// the involved free-point indices of a pair (up to 4), for iterating its vertices
int mjc_pairVerts(int* v, const mjcFlexPair* pair);

// per-pair detection band: min over the pair's flex radii and `band` (see the header note on
// midsurface gaps for why the radii enter the band and not the gap)
mjtNum mjc_pairBand(const mjcFlexPair* pair, const mjtNum* radii, mjtNum band);

// Gap of a pair at configuration x, plus the gradient's direction n and its vertex weights:
// dg/d(vertex idv[p]) = cw[p]*n, p < *nidx. gv/ge are the precomputed world-space geom corners and
// edges (mjc_GeomVerts/mjc_GeomEdges), radii the per-free-point radii. Early-out beyond cutoff.
MJAPI mjtNum mjc_pairGap(const mjcFlexPair* pair, const mjModel* m, const mjData* d,
                         const mjtNum* x, const mjtNum* gv, const mjtNum* ge, const mjtNum* radii,
                         mjtNum* n, int* idv, mjtNum* cw, int* nidx, mjtNum cutoff);

// Swept candidate generation: all pairs whose gap can enter the detection band along the segment
// dfrom -> dto, gathered over the flex BVH (self and cross-flex) and the geom features (gv, ge,
// with the owning geom of each in gvgeom, gegeom), under the native collision filtering: none
// with contact disabled, the contype/conaffinity rule per pair, selfcollide for self-contact.
// thresh / threshGeom are the base flex-flex / flex-geom reach at the query configuration x;
// each query extends it by the travel of the points involved along the sweep, and the BVH
// queries by the distance of the flex from the configuration its tree was built at (the state's),
// so no pair that the sweep can bring into the band is missed. ghat is the detection band. The
// candidates are pushed onto the mjData arena one at a time, so they form a contiguous array that
// lives until the arena is next reset (the position stage); *cand receives its start and the
// count is returned. If the arena fills, the remaining pairs are dropped with a warning: raise
// the model's memory attribute. A point meets the static geoms if it is free (fidx >= 0) or
// rides a moving body (pjnv[point] > 0; pjnv may be NULL when no point does): a point pinned to
// a static body never moves towards them.
// Under the ipc flag the IPC step detects and resolves these pairs itself, so the native pipeline
// neither generates nor keeps a contact for them: two dim-2 flexes (a flex with itself included),
// and a dim-2 flex against a static geom of a type the mode supports. Flex against a moving body,
// against a dim-1 flex and against the other geom types stay native, with their friction.
MJAPI int mjc_ipcOwnsFlexFlex(const mjModel* m, int f1, int f2);
MJAPI int mjc_ipcOwnsFlexGeom(const mjModel* m, int f, int g);

MJAPI int mjc_candidates(const mjModel* m, mjData* d, const mjtNum* x, const mjtNum* gv,
                         const mjtNum* ge, const int* gvgeom, const int* gegeom, int ngv, int nge,
                         const mjtNum* radii, mjtNum thresh, mjtNum threshGeom,
                         const mjtNum* dfrom, const mjtNum* dto, mjtNum ghat, int nfv, int npt,
                         const int* fidx, const int* pjnv, const int* flist, const int* fxadr,
                         int nfd, const int* pt2flex, mjcFlexPair** cand);

// Conservative advancement: the largest alpha in [0, 1] such that moving every point from x by
// alpha*dxw (per point, 3 per point; a point that does not move has zeros) keeps every candidate's
// gap above a fraction of its value at x (no pair's gap is closed by more than 80%), so the
// advanced configuration stays intersection-free. cgap holds each candidate's gap at x (from
// mjc_pairGap). Optional outputs: approut[c] = 1 if the full step closes candidate c into its
// active zone; toiout[c] = candidate c's own time of impact (1 if it does not collide this step).
MJAPI mjtNum mjc_advance(const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* dxw,
                         const mjtNum* gv, const mjtNum* ge, const mjtNum* radii, int nfv,
                         const mjcFlexPair* cand, int ncand, const mjtNum* cgap,
                         const int* pt2flex, int* approut, mjtNum* toiout);

// point-triangle distance (closest point cp and barycentric weights w)
MJAPI mjtNum mjc_PtTri(const mjtNum* p, const mjtNum* a, const mjtNum* b, const mjtNum* c,
                       mjtNum* cp, mjtNum* w);

// segment-segment distance (closest points and line parameters st)
MJAPI mjtNum mjc_SegSeg(const mjtNum* p1, const mjtNum* p2, const mjtNum* q1, const mjtNum* q2,
                        mjtNum* cp1, mjtNum* cp2, mjtNum* st);

// is geom type supported by mjc_GeomDist (plane, sphere, capsule, ellipsoid, cylinder, box, mesh)
MJAPI int mjc_GeomSupported(int type);

// signed distance (+ outward unit normal n) from geom gi's surface, at pose
// gpos/gmat, to world point x; early-out beyond distmax
MJAPI mjtNum mjc_GeomDist(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                          const mjtNum* x, mjtNum* n, mjtNum distmax);

// world-space corner features of geom gi at pose gpos/gmat (out sized by the caller): box corners,
// mesh vertices, a sphere's centre, a capsule's axis endpoints; return count
MJAPI int mjc_GeomVerts(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                        mjtNum* out);

// world-space edge features of geom gi at pose gpos/gmat (out sized by the caller): box edges,
// mesh hull edges, a capsule's axis; return count
MJAPI int mjc_GeomEdges(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                        mjtNum* out);

// compute the step displacement (dx, dw) over the time interval [0, t] and return the
// upper bound on the distance traveled by any point on the geom's surface
MJAPI mjtNum mjc_geomStepMotion(const mjModel* m, const mjData* d, int g, mjtNum t,
                                mjtNum dx[3], mjtNum dw[3]);

// conservative advancement between two geoms g1 and g2 over the timestep;
// write up to 4 contacts into con and return number of contacts
MJAPI int mjc_advanceGeomPair(const mjModel* m, const mjData* d, int g1, int g2,
                              mjtNum margin, mjPreContact* con, void* buffer);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_CONTINUOUS_H_
