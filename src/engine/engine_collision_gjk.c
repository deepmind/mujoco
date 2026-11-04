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

#include "engine/engine_collision_gjk.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mjtype.h>
#include "engine/engine_collision_convex.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"

#define mjMINVAL2 (mjMINVAL * mjMINVAL)
#define mjMAXVAL2 (mjMAXVAL * mjMAXVAL)

// numerical limits for single and double precision
#ifdef mjUSESINGLE
  // minimal distance squared for origin inside tetrahedron in polytope2
  #define mjMINDIST2 1e-10f
  // minimal distance squared for origin inside tetrahedron in polytope3
  #define mjMINDIST3 1e-10f
  // minimal distance squared for origin inside tetrahedron in polytope4
  #define mjMINDIST4 1e-17f
  // minimal tolerance for EPA
  #define mjMINEPATOL 1e-7f
#else
  #define mjMINDIST2 mjMINVAL2
  #define mjMINDIST3 mjMINVAL2
  #define mjMINDIST4 mjMINVAL2
  #define mjMINEPATOL mjMINVAL
#endif  // mjUSESINGLE

// align memory size on 8-byte boundary; needed for single precision
static inline size_t align8(size_t size) {
  return ((size + 7) / 8) * 8;
}

// vertex in a polytope
typedef struct {
  mjtNum vert[3];   // v1 - v2; vertex in Minkowski sum making up polytope
  mjtNum vert1[3];  // vertex of polytope in obj1
  mjtNum vert2[3];  // vertex of polytope in obj2
  int index1;       // vertex index in mesh 1
  int index2;       // vertex index in mesh 2
} Vertex;

// subdistance algorithm for GJK that computes the barycentric coordinates of the point in a
// simplex closest to the origin
// implementation adapted from Montanari et al, ToG 2017
static void subdistance(mjtNum lambda[4], int n, const mjtNum simplex[12]);

// compute the barycentric coordinates of the closest point to the origin in the n-simplex,
// for n = 3, 2, 1 respectively
static void S3D(mjtNum lambda[4], const mjtNum s1[3], const mjtNum s2[3], const mjtNum s3[3],
                const mjtNum s4[3]);
static void S2D(mjtNum lambda[3], const mjtNum s1[3], const mjtNum s2[3], const mjtNum s3[3]);
static void S1D(mjtNum lambda[2], const mjtNum s1[3], const mjtNum s2[3]);

// compute the support point for GJK
static void gjkSupport(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                       const mjtNum x_k[3], mjtNum x_norm);

// compute the linear combination of up to 4 3D vectors
static inline void lincomb(mjtNum res[3], const mjtNum* coef, int n, const mjtNum v1[3],
                           const mjtNum v2[3], const mjtNum v3[3], const mjtNum v4[3]);

// one face in a polytope
typedef struct {
  int verts;         // indices of the three vertices of the face in the polytope
  int adj[3];        // adjacent faces, one for each edge: [v1,v2], [v2,v3], [v3,v1]
  mjtNum v[3];       // projection of the origin on face, can be used as face normal
  mjtNum dist2;      // squared norm of v; negative if deleted
  int index;         // index in map; -1: not in map, -2: deleted from polytope
} Face;

// return int[3] with lower 30 bits split between three int
#define EPA_VERT_EXPAND(n)                     \
  { (n) & 0x3FF,                               \
    (n) >> 10 & 0x3FF,                         \
    (n) >> 20 & 0x3FF }


// polytope used in the Expanding Polytope Algorithm (EPA)
typedef struct {
  Vertex* verts;      // list of vertices that make up the polytope
  int nverts;         // number of vertices
  Face* faces;        // list of faces that make up the polytope
  int nfaces;         // number of faces
  int maxfaces;       // max number of faces that can be stored in polytope
  mjtNum center[3];   // center of the polytope
  Face** map;         // linear map storing faces
  int nmap;           // number of faces in map
  struct Horizon {    // polytope boundary edges that can be seen from w
    int* indices;     // indices of faces on horizon
    int* edges;       // corresponding edge of each face on the horizon
    int nedges;       // number of edges in horizon
    int maxedges;     // max number of edges in horizon
    const mjtNum* w;  // point where horizon is created
  } horizon;
} Polytope;

// compute the support point for EPA
static int epaSupport(Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2,
                      const mjtNum d[3], mjtNum dnorm);

// copy support vertex i from obj1 and obj2 into polytope and return its index
static int insertVertex(Polytope* pt, const mjCCDObj* obj1, const mjCCDObj* obj2, int i);

// attach a face to the polytope with the given vertex indices; return squared distance to origin
static mjtNum attachFace(Polytope* pt, int v1, int v2, int v3, int adj1, int adj2, int adj3);

// return 1 if objects are in contact; 0 if not; -1 if inconclusive
// status must have initial tetrahedrons
static int gjkIntersect(mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);

// create initial polytope for EPA for a 2, 3, or 4-simplex
static int polytope2(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);
static int polytope3(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);
static int polytope4(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);

// return a face of the expanded polytope that best approximates the pentration depth
// witness points are in status->{x1, x2}
static Face* epa(mjCCDStatus* status, Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2);

// -------------------------------- inlined  3D vector utils --------------------------------------

// res = v1 + v2
static inline void add3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  res[0] = v1[0] + v2[0], res[1] = v1[1] + v2[1], res[2] = v1[2] + v2[2];
}

// res = v1 - v2
static inline void sub3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  res[0] = v1[0] - v2[0], res[1] = v1[1] - v2[1], res[2] = v1[2] - v2[2];
}

// dot product
static inline mjtNum dot3(const mjtNum v1[3], const mjtNum v2[3]) {
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

// norm of v
static inline mjtNum norm3(const mjtNum v[3]) {
  return mju_sqrt(dot3(v, v));
}

// res = v
static inline void copy3(mjtNum res[3], const mjtNum v[3]) {
  res[0] = v[0], res[1] = v[1], res[2] = v[2];
}

// scalar product: res = s*v
static inline void scl3(mjtNum res[3], const mjtNum v[3], mjtNum s) {
  res[0] = s*v[0], res[1] = s*v[1], res[2] = s*v[2];
}

// res = v1 + s*v2
static inline void addScl3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3], mjtNum s) {
  res[0] = v1[0] + s*v2[0], res[1] = v1[1] + s*v2[1], res[2] = v1[2] + s*v2[2];
}

// cross product: res = v1 x v2
static inline void cross3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  res[0] = v1[1]*v2[2] - v1[2]*v2[1];
  res[1] = v1[2]*v2[0] - v1[0]*v2[2];
  res[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// return determinant of the 3x3 matrix with columns v1, v2, v3
static inline mjtNum det3(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3]) {
  // v1 * (v2 x v3)
  return v1[0]*(v2[1]*v3[2] - v2[2]*v3[1])
       + v1[1]*(v2[2]*v3[0] - v2[0]*v3[2])
       + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0]);
}

// ---------------------------------------- GJK ---------------------------------------------------


// return true if object is a discrete shape or shrunk to a point/line with no margin
static int discreteGeom(const mjCCDObj* obj) {
  // non-zero margin makes geoms smooth
  if (obj->margin != 0) return 0;

  // point and line supports are discrete
  if (obj->support == mjc_pointSupport || obj->support == mjc_lineSupport) return 1;

  // mesh, box, and hfield supports are discrete
  int g = obj->geom_type;
  return (g == mjGEOM_MESH || g == mjGEOM_BOX || g == mjGEOM_HFIELD);
}


// GJK algorithm
static void gjk(mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  int get_dist = status->dist_cutoff > 0;  // need to recover geom distances if not in contact
  int backup_gjk = !get_dist;              // use gjkIntersect if no geom distances needed
  mjtNum simplex[12];                      // Minkowski difference vertices in the simplex
  mjtNum* simplex1 = obj1->suppvert;       // obj1 support vertices in the simplex
  mjtNum* simplex2 = obj2->suppvert;       // obj2 support vertices in the simplex
  int* index1 = obj1->suppind;             // obj1 support vertex indices
  int* index2 = obj2->suppind;             // obj2 support vertex indices
  int n = 0;                               // number of vertices in the simplex
  int k = 0;                               // current iteration
  int kmax = status->max_iterations;       // max number of iterations
  mjtNum* x1_k = status->x1;               // the kth approximation point for obj1
  mjtNum* x2_k = status->x2;               // the kth approximation point for obj2
  mjtNum x_k[3];                           // the kth approximation point in Minkowski difference
  mjtNum lambda[4];                        // barycentric coordinates for x_k
  mjtNum tol2 = status->tolerance * status->tolerance;
  status->separated = 0;
  obj1->nsupp = 0;
  obj2->nsupp = 0;

  // if both geoms are discrete, finite convergence is guaranteed; set tolerance to 0
  int discrete = discreteGeom(obj1) && discreteGeom(obj2);
  mjtNum epsilon = discrete ? 0 : 0.5 * tol2;

  // minimum norm of x_k; resolve distances below tolerance for discrete shapes or distance queries
  mjtNum min_norm = (discrete || get_dist) ? mjMINVAL : status->tolerance;

  // set initial guess
  sub3(x_k, x1_k, x2_k);
  mjtNum x_norm = norm3(x_k), x_norm_prev = 0;  // set to 0 for no-op on first iteration

  for (; k < kmax; k++) {
    // in tolerance for geoms to be in contact, or x_norm has stagnated
    if (x_norm < min_norm || mju_abs(x_norm_prev - x_norm) < mjMINVAL) {
      break;
    }

    // compute the kth support point
    Vertex v;
    gjkSupport(&v, obj1, obj2, x_k, x_norm);
    mjtNum* s_k = simplex + 3*n;
    copy3(s_k, v.vert);
    copy3(simplex1 + 3*n, v.vert1);
    copy3(simplex2 + 3*n, v.vert2);
    index1[n] = v.index1;
    index2[n] = v.index2;

    // stopping criteria using the Frank-Wolfe duality gap given by
    //  |f(x_k) - f(x_min)|^2 <= < grad f(x_k), (x_k - s_k) >
    mjtNum diff[3] = {x_k[0] - s_k[0], x_k[1] - s_k[1], x_k[2] - s_k[2]};
    if (dot3(x_k, diff) < epsilon) {
      break;
    }

    // the lower bound on distance between the two geoms is (lower / x_norm)
    // if lower > 0, then the geoms are separated
    mjtNum lower = dot3(x_k, s_k);
    if (!get_dist) {
      if (lower > 0) {
        status->separated = 1;
        status->gjk_iterations = k;
        status->nx = 0;
        status->dist[0] = mjMAX_LIMIT;
        return;
      }
    } else if (status->dist_cutoff < mjMAX_LIMIT) {
      if (lower > 0 && lower >= status->dist_cutoff * x_norm) {
        status->separated = 1;
        status->gjk_iterations = k;
        status->nx = 0;
        status->dist[0] = mjMAX_LIMIT;
        return;
      }
    }

    // tetrahedron is generated and only need contact info; fallback to gjkIntersect to
    // determine contact
    if (n == 3 && backup_gjk) {
      status->gjk_iterations = k;
      int ret = gjkIntersect(status, obj1, obj2);
      if (ret != -1) {
        status->nx = 0;
        status->separated = ret == 0;
        status->dist[0] = ret > 0 ? 0 : mjMAX_LIMIT;
        return;
      }
      k = status->gjk_iterations;
      backup_gjk = 0;
    }

    // run the distance subalgorithm to compute the barycentric coordinates
    // of the closest point to the origin in the simplex
    subdistance(lambda, n + 1, simplex);

    // remove vertices from the simplex no longer needed
    n = 0;
    for (int i = 0; i < 4; i++) {
      if (!lambda[i]) continue;
      if (n != i) {
        copy3(simplex + 3*n, simplex + 3*i);
        copy3(simplex1 + 3*n, simplex1 + 3*i);
        copy3(simplex2 + 3*n, simplex2 + 3*i);
        index1[n] = index1[i];
        index2[n] = index2[i];
      }
      lambda[n++] = lambda[i];
    }

    // SHOULD NOT OCCUR
    if (n < 1) {
      status->gjk_iterations = k;
      status->nx = 0;
      status->dist[0] = mjMAX_LIMIT;
      status->separated = 1;
      return;
    }

    // get the next iteration of x_k, save previous x_norm
    lincomb(x_k, lambda, n, simplex + 0, simplex + 3, simplex + 6, simplex + 9);
    x_norm_prev = x_norm;
    x_norm = norm3(x_k);

    // we should have a tetrahedron containing the origin so return early
    if (n == 4) {
      break;
    }
  }

  // compute the approximate witness points
  if (n > 0) {
    lincomb(x1_k, lambda, n, simplex1 + 0, simplex1 + 3, simplex1 + 6, simplex1 + 9);
    lincomb(x2_k, lambda, n, simplex2 + 0, simplex2 + 3, simplex2 + 6, simplex2 + 9);
  }

  // GJK exited early; do a final separation check
  Vertex tmp;
  gjkSupport(&tmp, obj1, obj2, x_k, x_norm);
  if (dot3(x_k, tmp.vert) > 0) {
    status->separated = 1;
  }

  status->nx = 1;
  status->gjk_iterations = k;
  obj1->nsupp = n;
  obj2->nsupp = n;

  // if 3-simplex and not separated, then the origin is contained in the simplex
  status->dist[0] = (n == 4 && !status->separated) ? 0 : x_norm;
}


// compute the support point in obj1 and obj2 for Minkowski difference
static inline void support(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                           const mjtNum dir[3], const mjtNum dir_neg[3]) {
  // obj1
  obj1->support(v->vert1, obj1, dir);
  if (obj1->margin > 0 && obj1->geom >= 0) {
    mjtNum margin = 0.5 * obj1->margin;
    v->vert1[0] += dir[0] * margin;
    v->vert1[1] += dir[1] * margin;
    v->vert1[2] += dir[2] * margin;
  }

  // obj2
  obj2->support(v->vert2, obj2, dir_neg);
  if (obj2->margin > 0 && obj2->geom >= 0) {
    mjtNum margin = 0.5 * obj2->margin;
    v->vert2[0] += dir_neg[0] * margin;
    v->vert2[1] += dir_neg[1] * margin;
    v->vert2[2] += dir_neg[2] * margin;
  }

  // compute S_{A-B}(dir) = S_A(dir) - S_B(-dir)
  sub3(v->vert, v->vert1, v->vert2);

  // copy vertex indices of discrete geoms
  v->index1 = obj1->vertindex;
  v->index2 = obj2->vertindex;
}


// compute the support points in obj1 and obj2 for the kth approximation point
static inline void gjkSupport(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                              const mjtNum x_k[3], mjtNum x_norm) {
  mjtNum dir[3], dir_neg[3];

  // mjc_support requires a normalized direction
  scl3(dir_neg, x_k, 1 / x_norm);
  scl3(dir, dir_neg, -1);
  support(v, obj1, obj2, dir, dir_neg);
}


// compute support points in Minkowski difference, return index of new vertex in polytope
static int epaSupport(Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2,
                     const mjtNum d[3], mjtNum dnorm) {
  mjtNum dir[3] = {1, 0, 0}, dir_neg[3] = {-1, 0, 0};

  // mjc_support assumes a normalized direction
  if (dnorm > mjMINVAL) {
    dir[0] = d[0] / dnorm;
    dir[1] = d[1] / dnorm;
    dir[2] = d[2] / dnorm;
    scl3(dir_neg, dir, -1);
  }

  int n = pt->nverts++;
  Vertex* v = pt->verts + n;
  support(v, obj1, obj2, dir, dir_neg);
  return n;
}


// compute the support point in the Minkowski difference for gjkIntersect (without normalization)
static void gjkIntersectSupport(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                                const mjtNum dir[3]) {
  mjtNum dir_neg[3] = {-dir[0], -dir[1], -dir[2]};
  support(v, obj1, obj2, dir, dir_neg);
}


// compute the signed distance of a face along with the normal
static inline mjtNum signedDistance(mjtNum normal[3], const Vertex* v1, const Vertex* v2,
                                    const Vertex* v3) {
  mjtNum diff1[3], diff2[3];
  sub3(diff1, v3->vert, v1->vert);
  sub3(diff2, v2->vert, v1->vert);
  cross3(normal, diff1, diff2);
  mjtNum norm2 = dot3(normal, normal);
  if (norm2 > mjMINVAL2 && norm2 < mjMAXVAL2) {
    scl3(normal, normal, 1 / mju_sqrt(norm2));
    return dot3(normal, v1->vert);
  }
  return mjMAX_LIMIT;  // cannot recover normal (ignore face)
}


// return 1 if objects are in contact; 0 if not; -1 if inconclusive
static int gjkIntersect(mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  Vertex simplex[4];
  for (int i = 0; i < 4; i++) {
    copy3(simplex[i].vert1, obj1->suppvert + 3*i);
    copy3(simplex[i].vert2, obj2->suppvert + 3*i);
    sub3(simplex[i].vert, simplex[i].vert1, simplex[i].vert2);
    simplex[i].index1 = obj1->suppind[i];
    simplex[i].index2 = obj2->suppind[i];
  }
  int s[4] = {0, 1, 2, 3};

  int k = status->gjk_iterations, kmax = status->max_iterations;
  for (; k < kmax; k++) {
    // compute the signed distance to each face in the simplex along with normals
    mjtNum dist[4], normals[12];
    dist[0] = signedDistance(&normals[0], simplex + s[2], simplex + s[1], simplex + s[3]);
    dist[1] = signedDistance(&normals[3], simplex + s[0], simplex + s[2], simplex + s[3]);
    dist[2] = signedDistance(&normals[6], simplex + s[1], simplex + s[0], simplex + s[3]);
    dist[3] = signedDistance(&normals[9], simplex + s[0], simplex + s[1], simplex + s[2]);

    // if origin is on any affine hull, convergence will fail
    if (!dist[3] || !dist[2] || !dist[1] || !dist[0]) {
      status->gjk_iterations = k;
      return -1;
    }

    // find the face with the smallest distance to the origin
    int i = (dist[0] < dist[1]) ? 0 : 1;
    int j = (dist[2] < dist[3]) ? 2 : 3;
    int index = (dist[i] < dist[j]) ? i : j;

    // origin inside of simplex (run EPA for contact information)
    if (dist[index] > 0) {
      obj1->nsupp = 4;
      obj2->nsupp = 4;
      for (int m = 0; m < 4; m++) {
        copy3(obj1->suppvert + 3*m, simplex[s[m]].vert1);
        copy3(obj2->suppvert + 3*m, simplex[s[m]].vert2);
        obj1->suppind[m] = simplex[s[m]].index1;
        obj2->suppind[m] = simplex[s[m]].index2;
      }
      status->gjk_iterations = k;
      return 1;
    }

    // replace worst vertex (farthest from origin) with new candidate
    gjkIntersectSupport(simplex + s[index], obj1, obj2, normals + 3*index);

    // found origin outside the Minkowski difference (return no collision)
    if (dot3(&normals[3*index], simplex[s[index]].vert) < 0) {
      obj1->nsupp = 0;
      obj2->nsupp = 0;
      status->gjk_iterations = k;
      return 0;
    }

    // swap vertices in the simplex to retain orientation
    i = (index + 1) & 3;
    j = (index + 2) & 3;
    int swap = s[i];
    s[i] = s[j];
    s[j] = swap;
  }
  status->gjk_iterations = k;
  return -1;  // never found origin
}


// compute the linear combination of up to 4 3D vectors
static inline void lincomb(mjtNum res[3], const mjtNum* coef, int n, const mjtNum v1[3],
                           const mjtNum v2[3], const mjtNum v3[3], const mjtNum v4[3]) {
  switch (n) {
    case 0:
      res[0] = res[1] = res[2] = 0;
      break;
    case 1:
      res[0] = coef[0]*v1[0];
      res[1] = coef[0]*v1[1];
      res[2] = coef[0]*v1[2];
      break;
    case 2:
      res[0] = coef[0]*v1[0] + coef[1]*v2[0];
      res[1] = coef[0]*v1[1] + coef[1]*v2[1];
      res[2] = coef[0]*v1[2] + coef[1]*v2[2];
      break;
    case 3:
      res[0] = coef[0]*v1[0] + coef[1]*v2[0] + coef[2]*v3[0];
      res[1] = coef[0]*v1[1] + coef[1]*v2[1] + coef[2]*v3[1];
      res[2] = coef[0]*v1[2] + coef[1]*v2[2] + coef[2]*v3[2];
      break;
    case 4:
      res[0] = coef[0]*v1[0] + coef[1]*v2[0] + coef[2]*v3[0] + coef[3]*v4[0];
      res[1] = coef[0]*v1[1] + coef[1]*v2[1] + coef[2]*v3[1] + coef[3]*v4[1];
      res[2] = coef[0]*v1[2] + coef[1]*v2[2] + coef[2]*v3[2] + coef[3]*v4[2];
      break;
  }
}


// res = origin projected onto plane defined by v1, v2, v3
static int projectOriginPlane(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3],
                              const mjtNum v3[3]) {
  mjtNum diff21[3], diff31[3], diff32[3], n[3], nv, nn;
  sub3(diff21, v2, v1);
  sub3(diff31, v3, v1);
  sub3(diff32, v3, v2);

  // n = (v1 - v2) x (v3 - v2)
  cross3(n, diff32, diff21);
  nv = dot3(n, v2);
  nn = dot3(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > mjMINVAL) {
    scl3(res, n, nv / nn);
    return 0;
  }

  // n = (v2 - v1) x (v3 - v1)
  cross3(n, diff21, diff31);
  nv = dot3(n, v1);
  nn = dot3(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > mjMINVAL) {
    scl3(res, n, nv / nn);
    return 0;
  }

  // n = (v1 - v3) x (v2 - v3)
  cross3(n, diff31, diff32);
  nv = dot3(n, v3);
  nn = dot3(n, n);
  scl3(res, n, nv / nn);
  return 0;
}


// res = origin projected onto line defined by v1, v2
static inline void projectOriginLine(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  // res = v2 - <v2, v2 - v1> / <v2 - v1, v2 - v1> * (v2 - v1)
  mjtNum diff[3];
  sub3(diff, v2, v1);
  mjtNum scl = -(dot3(v2, diff) / dot3(diff, diff));
  res[0] = v2[0] + scl*diff[0];
  res[1] = v2[1] + scl*diff[1];
  res[2] = v2[2] + scl*diff[2];
}


// return 1 if both numbers are positive, -1 if both negative and 0 otherwise
static inline int sameSign2(mjtNum a, mjtNum b) {
  if (a > 0 && b > 0) return 1;
  if (a < 0 && b < 0) return -1;
  return 0;
}


// subdistance algorithm for GJK that computes the barycentric coordinates of the point in a
// simplex closest to the origin
// implementation adapted from Montanari et al, ToG 2017
static inline void subdistance(mjtNum lambda[4], int n, const mjtNum simplex[12]) {
  lambda[0] = lambda[1] = lambda[2] = lambda[3] = 0;
  const mjtNum* s1 = simplex + 0;
  const mjtNum* s2 = simplex + 3;
  const mjtNum* s3 = simplex + 6;
  const mjtNum* s4 = simplex + 9;

  switch (n) {
  case 4:
     S3D(lambda, s1, s2, s3, s4);
     break;
  case 3:
     S2D(lambda, s1, s2, s3);
      break;
  case 2:
     S1D(lambda, s1, s2);
     break;
  default:
    lambda[0] = 1;
    break;
  }
}


static void S3D(mjtNum lambda[4], const mjtNum s1[3], const mjtNum s2[3],
                const mjtNum s3[3], const mjtNum s4[3]) {
  // the matrix M is given by
  //  [[ s1_x, s2_x, s3_x, s4_x ],
  //   [ s1_y, s2_y, s3_y, s4_y ],
  //   [ s1_z, s2_z, s3_z, s4_z ],
  //   [ 1,    1,    1,    1    ]]
  // we want to solve M*lambda = P, where P = [p_x, p_y, p_z, 1] with [p_x, p_y, p_z] is the
  // origin projected onto the simplex

  // compute cofactors to find det(M)
  mjtNum C41 = -det3(s2, s3, s4);
  mjtNum C42 =  det3(s1, s3, s4);
  mjtNum C43 = -det3(s1, s2, s4);
  mjtNum C44 =  det3(s1, s2, s3);

  // note that m_det = 6*SignVol(simplex) with C4i corresponding to the volume of the 3-simplex
  // with vertices {s1, s2, s3, 0} - si
  mjtNum m_det = C41 + C42 + C43 + C44;

  int comp1 = sameSign2(m_det, C41),
      comp2 = sameSign2(m_det, C42),
      comp3 = sameSign2(m_det, C43),
      comp4 = sameSign2(m_det, C44);

  // if all signs are the same then the origin is inside the simplex
  if (comp1 && comp2 && comp3 && comp4) {
    lambda[0] = C41 / m_det;
    lambda[1] = C42 / m_det;
    lambda[2] = C43 / m_det;
    lambda[3] = C44 / m_det;
    return;
  }

  // find the smallest distance, and use the corresponding barycentric coordinates
  mjtNum dmin = mjMAX_LIMIT;

  if (!comp1) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s2, s3, s4);
    lincomb(x, lambda_2d, 3, s2, s3, s4, NULL);
    mjtNum d = dot3(x, x);
    lambda[0] = 0;
    lambda[1] = lambda_2d[0];
    lambda[2] = lambda_2d[1];
    lambda[3] = lambda_2d[2];
    dmin = d;
  }

  if (!comp2) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s3, s4);
    lincomb(x, lambda_2d, 3, s1, s3, s4, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = 0;
      lambda[2] = lambda_2d[1];
      lambda[3] = lambda_2d[2];
      dmin = d;
    }
  }

  if (!comp3) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s2, s4);
    lincomb(x, lambda_2d, 3, s1, s2, s4, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = lambda_2d[1];
      lambda[2] = 0;
      lambda[3] = lambda_2d[2];
      dmin = d;
    }
  }

  if (!comp4) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s2, s3);
    lincomb(x, lambda_2d, 3, s1, s2, s3, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = lambda_2d[1];
      lambda[2] = lambda_2d[2];
      lambda[3] = 0;
    }
  }
}


static void S2D(mjtNum lambda[3], const mjtNum s1[3], const mjtNum s2[3], const mjtNum s3[3]) {
  // project origin onto affine hull of the simplex
  mjtNum p_o[3];
  if (projectOriginPlane(p_o, s1, s2, s3)) {
    S1D(lambda, s1, s2);
    lambda[2] = 0;
    return;
  }

  // Below are the minors M_i4 of the matrix M given by
  // [[ s1_x, s2_x, s3_x, s4_x ],
  //  [ s1_y, s2_y, s3_y, s4_y ],
  //  [ s1_z, s2_z, s3_z, s4_z ],
  //  [ 1,    1,    1,    1    ]]
  mjtNum M_14 = s2[1]*s3[2] - s2[2]*s3[1] - s1[1]*s3[2] + s1[2]*s3[1] + s1[1]*s2[2] - s1[2]*s2[1];
  mjtNum M_24 = s2[0]*s3[2] - s2[2]*s3[0] - s1[0]*s3[2] + s1[2]*s3[0] + s1[0]*s2[2] - s1[2]*s2[0];
  mjtNum M_34 = s2[0]*s3[1] - s2[1]*s3[0] - s1[0]*s3[1] + s1[1]*s3[0] + s1[0]*s2[1] - s1[1]*s2[0];

  // exclude the axis with the largest projection of the simplex using the computed minors
  mjtNum M_max = 0;
  mjtNum s1_2D[2], s2_2D[2], s3_2D[2], p_o_2D[2];
  mjtNum mu1 = mju_abs(M_14), mu2 = mju_abs(M_24), mu3 = mju_abs(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    s1_2D[0] = s1[1];
    s1_2D[1] = s1[2];

    s2_2D[0] = s2[1];
    s2_2D[1] = s2[2];

    s3_2D[0] = s3[1];
    s3_2D[1] = s3[2];

    p_o_2D[0] = p_o[1];
    p_o_2D[1] = p_o[2];
  } else if (mu2 >= mu3) {
    M_max = M_24;
    s1_2D[0] = s1[0];
    s1_2D[1] = s1[2];

    s2_2D[0] = s2[0];
    s2_2D[1] = s2[2];

    s3_2D[0] = s3[0];
    s3_2D[1] = s3[2];

    p_o_2D[0] = p_o[0];
    p_o_2D[1] = p_o[2];
  } else {
    M_max = M_34;
    s1_2D[0] = s1[0];
    s1_2D[1] = s1[1];

    s2_2D[0] = s2[0];
    s2_2D[1] = s2[1];

    s3_2D[0] = s3[0];
    s3_2D[1] = s3[1];

    p_o_2D[0] = p_o[0];
    p_o_2D[1] = p_o[1];
  }

  // compute the cofactors C3i of the following matrix:
  // [[ s1_2D[0] - p_o_2D[0], s2_2D[0] - p_o_2D[0], s3_2D[0] - p_o_2D[0] ],
  //  [ s1_2D[1] - p_o_2D[1], s2_2D[1] - p_o_2D[1], s3_2D[1] - p_o_2D[1] ],
  //  [ 1,                    1,                    1                    ]]

  // C31 corresponds to the signed area of 2-simplex: (p_o_2D, s2_2D, s3_2D)
  mjtNum C31 = p_o_2D[0]*s2_2D[1] + p_o_2D[1]*s3_2D[0] + s2_2D[0]*s3_2D[1]
             - p_o_2D[0]*s3_2D[1] - p_o_2D[1]*s2_2D[0] - s3_2D[0]*s2_2D[1];

  // C32 corresponds to the signed area of 2-simplex: (_po_2D, s1_2D, s3_2D)
  mjtNum C32 = p_o_2D[0]*s3_2D[1] + p_o_2D[1]*s1_2D[0] + s3_2D[0]*s1_2D[1]
             - p_o_2D[0]*s1_2D[1] - p_o_2D[1]*s3_2D[0] - s1_2D[0]*s3_2D[1];

  // C33 corresponds to the signed area of 2-simplex: (p_o_2D, s1_2D, s2_2D)
  mjtNum C33 = p_o_2D[0]*s1_2D[1] + p_o_2D[1]*s2_2D[0] + s1_2D[0]*s2_2D[1]
             - p_o_2D[0]*s2_2D[1] - p_o_2D[1]*s1_2D[0] - s2_2D[0]*s1_2D[1];

  int comp1 = sameSign2(M_max, C31),
      comp2 = sameSign2(M_max, C32),
      comp3 = sameSign2(M_max, C33);

  // all the same sign, p_o is inside the 2-simplex
  if (comp1 && comp2 && comp3) {
    lambda[0] = C31 / M_max;
    lambda[1] = C32 / M_max;
    lambda[2] = C33 / M_max;
    return;
  }

  // find the smallest distance, and use the corresponding barycentric coordinates
  mjtNum dmin = mjMAX_LIMIT;

  if (!comp1) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s2, s3);
    lincomb(x, lambda_1d, 2, s2, s3, NULL, NULL);
    mjtNum d = dot3(x, x);
    lambda[0] = 0;
    lambda[1] = lambda_1d[0];
    lambda[2] = lambda_1d[1];
    dmin = d;
  }

  if (!comp2) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s1, s3);
    lincomb(x, lambda_1d, 2, s1, s3, NULL, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_1d[0];
      lambda[1] = 0;
      lambda[2] = lambda_1d[1];
      dmin = d;
    }
  }

  if (!comp3) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s1, s2);
    lincomb(x, lambda_1d, 2, s1, s2, NULL, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_1d[0];
      lambda[1] = lambda_1d[1];
      lambda[2] = 0;
    }
  }
}


static void S1D(mjtNum lambda[2], const mjtNum s1[3], const mjtNum s2[3]) {
  // find projection of origin onto the 1-simplex:
  mjtNum p_o[3];
  projectOriginLine(p_o, s1, s2);

  // find the axis with the largest projection "shadow" of the simplex
  mjtNum mu = s1[0] - s2[0];
  mjtNum mu_max = mu;
  int index = 0;

  mu = s1[1] - s2[1];
  if (mju_abs(mu) >= mju_abs(mu_max)) {
    mu_max = mu;
    index = 1;
  }

  mu = s1[2] - s2[2];
  if (mju_abs(mu) >= mju_abs(mu_max)) {
    mu_max = mu;
    index = 2;
  }

  mjtNum C1 = p_o[index] - s2[index];
  mjtNum C2 = s1[index] - p_o[index];

  // determine if projection of origin lies inside 1-simplex
  int same = sameSign2(mu_max, C1) && sameSign2(mu_max, C2);
  lambda[0] = same ? C1 / mu_max : 0;
  lambda[1] = same ? C2 / mu_max : 1;
}


// ---------------------------------------- EPA ---------------------------------------------------

// replace a 3-simplex with one of its faces
static inline void replaceSimplex3(Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2,
                                   int v1, int v2, int v3) {
  int v[3] = {v1, v2, v3};
  obj1->nsupp = 3;
  obj2->nsupp = 3;
  for (int i = 0; i < 3; i++) {
    copy3(obj1->suppvert + 3*i, pt->verts[v[i]].vert1);
    copy3(obj2->suppvert + 3*i, pt->verts[v[i]].vert2);
    obj1->suppind[i] = pt->verts[v[i]].index1;
    obj2->suppind[i] = pt->verts[v[i]].index2;
  }

  // reset polytope
  pt->nfaces = 0;
  pt->nverts = 0;
  pt->nmap = 0;
}


// return 1 if the origin and p3 are on the same side of the plane defined by p0, p1, p2
static int sameSide(const mjtNum p0[3], const mjtNum p1[3],
                    const mjtNum p2[3], const mjtNum p3[3]) {
    mjtNum diff1[3], diff2[3], diff3[3], diff4[3], n[3];
    sub3(diff1, p1, p0);
    sub3(diff2, p2, p0);
    cross3(n, diff1, diff2);

    sub3(diff3, p3, p0);
    mjtNum dot1 = dot3(n, diff3);

    scl3(diff4, p0, -1);
    mjtNum dot2 = dot3(n, diff4);
    if (dot1 > 0 && dot2 > 0) return 1;
    if (dot1 < 0 && dot2 < 0) return 1;
    return 0;
}


// return 1 if the origin is contained in the tetrahedron, 0 otherwise
static int testTetra(const mjtNum p0[3], const mjtNum p1[3],
                     const mjtNum p2[3], const mjtNum p3[3]) {
  return sameSide(p0, p1, p2, p3)
      && sameSide(p1, p2, p3, p0)
      && sameSide(p2, p3, p0, p1)
      && sameSide(p3, p0, p1, p2);
}


// matrix for 120 degrees rotation around given axis
static void rotmat(mjtNum R[9], const mjtNum axis[3]) {
  mjtNum n = norm3(axis);
  mjtNum u1 = axis[0] / n, u2 = axis[1] / n, u3 = axis[2] / n;
  const mjtNum sin = 0.86602540378;  // sin(120 deg)
  const mjtNum cos = -0.5;           // cos(120 deg)
  R[0] = cos + u1*u1*(1 - cos);
  R[1] = u1*u2*(1 - cos) - u3*sin;
  R[2] = u1*u3*(1 - cos) + u2*sin;
  R[3] = u2*u1*(1 - cos) + u3*sin;
  R[4] = cos + u2*u2*(1 - cos);
  R[5] = u2*u3*(1 - cos) - u1*sin;
  R[6] = u1*u3*(1 - cos) - u2*sin;
  R[7] = u2*u3*(1 - cos) + u1*sin;
  R[8] = cos + u3*u3*(1 - cos);
}


// return nonzero if the ray v1v2 intersects the triangle v3v4v5
static inline int rayTriangle(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3],
                              const mjtNum v4[3], const mjtNum v5[3]) {
  mjtNum diff12[3], diff13[3], diff14[3], diff15[3];
  sub3(diff12, v2, v1);
  sub3(diff13, v3, v1);
  sub3(diff14, v4, v1);
  sub3(diff15, v5, v1);

  mjtNum vol1 = det3(diff13, diff14, diff12);
  mjtNum vol2 = det3(diff14, diff15, diff12);
  mjtNum vol3 = det3(diff15, diff13, diff12);

  if (vol1 >= 0 && vol2 >= 0 && vol3 >= 0) return 1;
  if (vol1 <= 0 && vol2 <= 0 && vol3 <= 0) return -1;
  return 0;
}


// create a polytope from a 1-simplex (returns 0 on success)
static int polytope2(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  int v1i = insertVertex(pt, obj1, obj2, 0);
  int v2i = insertVertex(pt, obj1, obj2, 1);
  mjtNum *v1 = pt->verts[v1i].vert, *v2 = pt->verts[v2i].vert;

  // set the polytope center
  add3(pt->center, v1, v2);
  scl3(pt->center, pt->center, 0.5);

  mjtNum diff[3];
  sub3(diff, v2, v1);

  // find component with smallest magnitude (so cross product is largest)
  mjtNum value = mjMAX_LIMIT;
  int index = 0;
  for (int i = 0; i < 3; i++) {
    if (mju_abs(diff[i]) < value) {
      value = mju_abs(diff[i]);
      index = i;
    }
  }

  // cross product with best coordinate axis
  mjtNum e[3] = {0, 0, 0};
  e[index] = 1;
  mjtNum d1[3], d2[3], d3[3];
  cross3(d1, e, diff);

  // rotate around the line segment to get three more points spaced 120 degrees apart
  mjtNum R[9];
  rotmat(R, diff);

  mju_mulMatVec3(d2, R, d1);
  mju_mulMatVec3(d3, R, d2);

  // save vertices and get indices for each one
  int v3i = epaSupport(pt, obj1, obj2, d1, norm3(d1));
  int v4i = epaSupport(pt, obj1, obj2, d2, norm3(d2));
  int v5i = epaSupport(pt, obj1, obj2, d3, norm3(d3));

  mjtNum* v3 = pt->verts[v3i].vert;
  mjtNum* v4 = pt->verts[v4i].vert;
  mjtNum* v5 = pt->verts[v5i].vert;

  // build hexahedron
  if (attachFace(pt, v1i, v3i, v4i, 1, 3, 2) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v1i, v3i, v4i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1i, v5i, v3i, 2, 4, 0) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v1i, v5i, v3i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1i, v4i, v5i, 0, 5, 1) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v1i, v4i, v5i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v2i, v4i, v3i, 5, 0, 4) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v2i, v4i, v3i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v2i, v3i, v5i, 3, 1, 5) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v2i, v3i, v5i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v2i, v5i, v4i, 4, 2, 3) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v2i, v5i, v4i);
    return polytope3(pt, status, obj1, obj2);
  }

  // check hexahedron is convex
  if (!rayTriangle(v1, v2, v3, v4, v5)) {
    return mjEPA_P2_NONCONVEX;
  }

  // populate face map
  for (int i = 0; i < 6; i++) {
    pt->map[i] = pt->faces + i;
    pt->faces[i].index = i;
  }
  pt->nmap = 6;
  return 0;
}


// compute the affine coordinates of p on the triangle v1v2v3
static void triAffineCoord(mjtNum lambda[3], const mjtNum v1[3], const mjtNum v2[3],
                           const mjtNum v3[3], const mjtNum p[3]) {
  // compute minors as in S2D
  mjtNum M_14 = v2[1]*v3[2] - v2[2]*v3[1] - v1[1]*v3[2] + v1[2]*v3[1] + v1[1]*v2[2] - v1[2]*v2[1];
  mjtNum M_24 = v2[0]*v3[2] - v2[2]*v3[0] - v1[0]*v3[2] + v1[2]*v3[0] + v1[0]*v2[2] - v1[2]*v2[0];
  mjtNum M_34 = v2[0]*v3[1] - v2[1]*v3[0] - v1[0]*v3[1] + v1[1]*v3[0] + v1[0]*v2[1] - v1[1]*v2[0];

  // exclude one of the axes with the largest projection of the simplex using the computed minors
  mjtNum M_max = 0;
  int x, y;
  mjtNum mu1 = mju_abs(M_14), mu2 = mju_abs(M_24), mu3 = mju_abs(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    x = 1;
    y = 2;
  } else if (mu2 >= mu3) {
    M_max = M_24;
    x = 0;
    y = 2;
  } else {
    M_max = M_34;
    x = 0;
    y = 1;
  }

  // C31 corresponds to the signed area of 2-simplex: (v, s2, s3)
  mjtNum C31 = p[x]*v2[y] + p[y]*v3[x] + v2[x]*v3[y]
             - p[x]*v3[y] - p[y]*v2[x] - v3[x]*v2[y];

  // C32 corresponds to the signed area of 2-simplex: (v, s1, s3)
  mjtNum C32 = p[x]*v3[y] + p[y]*v1[x] + v3[x]*v1[y]
             - p[x]*v1[y] - p[y]*v3[x] - v1[x]*v3[y];

  // C33 corresponds to the signed area of 2-simplex: (v, s1, s2)
  mjtNum C33 = p[x]*v1[y] + p[y]*v2[x] + v1[x]*v2[y]
             - p[x]*v2[y] - p[y]*v1[x] - v2[x]*v1[y];

  // compute affine coordinates
  lambda[0] = C31 / M_max;
  lambda[1] = C32 / M_max;
  lambda[2] = C33 / M_max;
}


// return true if point p and triangle v1v2v3 intersect
static int triPointIntersect(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3],
                             const mjtNum p[3]) {
  mjtNum lambda[3];
  triAffineCoord(lambda, v1, v2, v3, p);
  if (lambda[0] < 0 || lambda[1] < 0 || lambda[2] < 0) {
    return 0;
  }
  mjtNum pr[3], diff[3];
  pr[0] = v1[0]*lambda[0] + v2[0]*lambda[1] + v3[0]*lambda[2];
  pr[1] = v1[1]*lambda[0] + v2[1]*lambda[1] + v3[1]*lambda[2];
  pr[2] = v1[2]*lambda[0] + v2[2]*lambda[1] + v3[2]*lambda[2];
  sub3(diff, pr, p);
  return norm3(diff) < mjMINVAL;
}


// create a polytope from a 2-simplex (returns 0 on success)
static int polytope3(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  // save vertices and get indices for each one
  int v1i = insertVertex(pt, obj1, obj2, 0);
  int v2i = insertVertex(pt, obj1, obj2, 1);
  int v3i = insertVertex(pt, obj1, obj2, 2);
  const mjtNum *v1 = pt->verts[v1i].vert,
               *v2 = pt->verts[v2i].vert,
               *v3 = pt->verts[v3i].vert;

  // set the polytope center
  add3(pt->center, v1, v2);
  add3(pt->center, pt->center, v3);
  scl3(pt->center, pt->center, 1.0 / 3.0);

  // get normals in both directions
  mjtNum diff1[3], diff2[3], n[3], n_neg[3];
  sub3(diff1, v2, v1);
  sub3(diff2, v3, v1);
  cross3(n, diff1, diff2);
  mjtNum n_norm = norm3(n);
  if (n_norm < mjMINVAL) {
    return mjEPA_P3_BAD_NORMAL;
  }

  // negative of triangle normal n
  scl3(n_neg, n, -1);

  int v5i = epaSupport(pt, obj1, obj2, n_neg, n_norm);
  int v4i = epaSupport(pt, obj1, obj2, n, n_norm);
  mjtNum* v4 = pt->verts[v4i].vert;
  mjtNum* v5 = pt->verts[v5i].vert;

  // check that v4 is not contained in the 2-simplex
  if (triPointIntersect(v1, v2, v3, v4)) {
    return mjEPA_P3_INVALID_V4;
  }

  // check that v5 is not contained in the 2-simplex
  if (triPointIntersect(v1, v2, v3, v5)) {
    return mjEPA_P3_INVALID_V5;
  }

  // check that v4 and v5 are not coplanar with the 2-simplex
  mjtNum d4[3], d5[3];
  sub3(d4, v4, v1);
  sub3(d5, v5, v1);
  if (dot3(n, d4) < mjMINEPATOL*n_norm || dot3(n_neg, d5) < mjMINEPATOL*n_norm) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }

  // if origin does not lie on simplex then we need to check that the hexahedron contains the
  // origin
  //
  // TODO(kylebayes): It's possible for GJK to return a 2-simplex with the origin not contained in
  // it but within tolerance from it. In that case the hexahedron could possibly be constructed
  // that doesn't contain the origin, but nonetheless there is penetration depth.
  if (status->dist[0] > 10*mjMINVAL && !testTetra(v1, v2, v3, v4) && !testTetra(v1, v2, v3, v5)) {
    return mjEPA_P3_MISSING_ORIGIN;
  }

  // create hexahedron for EPA
  if (attachFace(pt, v4i, v1i, v2i, 1, 3, 2) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v4i, v3i, v1i, 2, 4, 0) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v4i, v2i, v3i, 0, 5, 1) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v2i, v1i, 5, 0, 4) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v1i, v3i, 3, 1, 5) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v3i, v2i, 4, 2, 3) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }

  // populate face map
  for (int i = 0; i < 6; i++) {
    pt->map[i] = pt->faces + i;
    pt->faces[i].index = i;
  }
  pt->nmap = 6;
  return 0;
}


// create a polytope from a 3-simplex (returns 0 on success)
static int polytope4(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  int v1 = insertVertex(pt, obj1, obj2, 0);
  int v2 = insertVertex(pt, obj1, obj2, 1);
  int v3 = insertVertex(pt, obj1, obj2, 2);
  int v4 = insertVertex(pt, obj1, obj2, 3);

  // set the polytope center
  add3(pt->center, pt->verts[v1].vert, pt->verts[v2].vert);
  add3(pt->center, pt->center, pt->verts[v3].vert);
  add3(pt->center, pt->center, pt->verts[v4].vert);
  scl3(pt->center, pt->center, 0.25);

  // if the origin is on a face, replace the 3-simplex with a 2-simplex
  if (attachFace(pt, v1, v2, v3, 1, 3, 2) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v1, v2, v3);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1, v4, v2, 2, 3, 0) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v1, v4, v2);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1, v3, v4, 0, 3, 1) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v1, v3, v4);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v4, v3, v2, 2, 0, 1) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v4, v3, v2);
    return polytope3(pt, status, obj1, obj2);
  }

  // numerically verify that the origin lies in the tetrahedron interior
  if (!testTetra(pt->verts[v1].vert, pt->verts[v2].vert, pt->verts[v3].vert, pt->verts[v4].vert)) {
    return mjEPA_P4_MISSING_ORIGIN;
  }

  // populate face map
  for (int i = 0; i < 4; i++) {
    pt->map[i] = pt->faces + i;
    pt->faces[i].index = i;
  }
  pt->nmap = 4;
  return 0;
}


// copy support vertex i from obj1 and obj2 into polytope and return its index
static inline int insertVertex(Polytope* pt, const mjCCDObj* obj1, const mjCCDObj* obj2, int i) {
  int n = pt->nverts++;
  Vertex* v = &pt->verts[n];
  copy3(v->vert1, obj1->suppvert + 3*i);
  copy3(v->vert2, obj2->suppvert + 3*i);
  sub3(v->vert, v->vert1, v->vert2);
  v->index1 = obj1->suppind[i];
  v->index2 = obj2->suppind[i];
  return n;
}


// delete face from map (return non-zero on error)
static void deleteFace(Polytope* pt, Face* face) {
  if (face->index >= 0) {
    pt->map[face->index] = pt->map[--pt->nmap];
    pt->map[face->index]->index = face->index;
  }
  face->index = -2;  // mark face as deleted from map and polytope
}


// return max number of faces that can be stored in polytope
static inline int maxFaces(Polytope* pt) {
  return pt->maxfaces - pt->nfaces;
}


// attach a face to the polytope with the given vertex indices; return squared distance to origin
static inline mjtNum attachFace(Polytope* pt, int v1, int v2, int v3,
                                int adj1, int adj2, int adj3) {
  Face* face = &pt->faces[pt->nfaces++];
  face->verts = v1 + (v2 << 10) + (v3 << 20);

  // adjacent faces
  face->adj[0] = adj1;
  face->adj[1] = adj2;
  face->adj[2] = adj3;

  // compute witness point v
  int ret = projectOriginPlane(face->v, pt->verts[v3].vert, pt->verts[v2].vert, pt->verts[v1].vert);
  if (ret) {
    return 0;
  }

  // ensure projection points outward from the polytope
  mjtNum outward[3];
  sub3(outward, pt->verts[v1].vert, pt->center);
  if (dot3(face->v, outward) < 0) {
    scl3(face->v, face->v, -1);
  }

  face->dist2 = dot3(face->v, face->v);
  face->index = -1;

  return face->dist2;
}


// add an edge to the horizon; return number of edges added
static inline int addEdge(Polytope* pt, int index, int edge) {
  // always increment nedges so we can produce a useful warning if threshold is exceeded
  int i = pt->horizon.nedges++;
  if (i < pt->horizon.maxedges) {
    pt->horizon.edges[i] = edge;
    pt->horizon.indices[i] = index;
    return 1;
  }
  return 0;
}


// get edge index where vertex lies
static inline int getEdge(Face* face, int vertex) {
  int verts[3] = EPA_VERT_EXPAND(face->verts);
  if (verts[0] == vertex) return 0;
  if (verts[1] == vertex) return 1;
  return 2;
}


// recursive call to build horizon; return 1 if face is visible from w otherwise 0
static int horizonRec(Polytope* pt, Face* face, int e) {
    // v is visible from w so it is deleted and adjacent faces are checked
    if (dot3(face->v, pt->horizon.w) - face->dist2 > mjMINVAL) {
      int verts[3] = EPA_VERT_EXPAND(face->verts);
      deleteFace(pt, face);
      int ret = 1;

      // recursively search the adjacent faces on the next two edges
      for (int k = 1; k < 3; k++) {
        int i = (e + k) % 3;
        Face* adjFace = &pt->faces[face->adj[i]];
        if (adjFace->index > -2) {
          int adjEdge = getEdge(adjFace, verts[(i + 1) % 3]);
          int r = horizonRec(pt, adjFace, adjEdge);
          if (r < 0 || (!r && !addEdge(pt, face->adj[i], adjEdge))) {
            ret = -1;
          }
        }
      }
      return ret;
    }
  return 0;
}


// create horizon given the face as starting point; return 0 on success
static int horizon(Polytope* pt, Face* face) {
  deleteFace(pt, face);
  int verts[3] = EPA_VERT_EXPAND(face->verts);
  int ret = 0;
  for (int i = 0; i < 3; i++) {
    Face* adjFace = &pt->faces[face->adj[i]];
    if (adjFace->index > -2) {
      int adjEdge = getEdge(adjFace, verts[(i + 1) % 3]);
      int r = horizonRec(pt, adjFace, adjEdge);
      if (r < 0 || (!r && !addEdge(pt, face->adj[i], adjEdge))) {
        ret = -1;
      }
    }
  }
  return ret;
}


// recover witness points from EPA polytope, return signed distance between witness points
static mjtNum epaWitness(const Polytope* pt, const Face* face, mjtNum x1[3], mjtNum x2[3]) {
  // compute affine coordinates for witness points on plane defined by face
  mjtNum lambda[3];
  int verts[3] = EPA_VERT_EXPAND(face->verts);
  Vertex* v1 = pt->verts + verts[0];
  Vertex* v2 = pt->verts + verts[1];
  Vertex* v3 = pt->verts + verts[2];
  triAffineCoord(lambda, v1->vert, v2->vert, v3->vert, face->v);

  // witness point on geom 1
  lincomb(x1, lambda, 3, v1->vert1, v2->vert1, v3->vert1, NULL);

  // witness point on geom 2
  lincomb(x2, lambda, 3, v1->vert2, v2->vert2, v3->vert2, NULL);

  return -mju_sqrt(face->dist2);
}


// return a face of the expanded polytope that best approximates the pentration depth
// witness points are in status->{x1, x2}
static Face* epa(mjCCDStatus* status, Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2) {
  mjtNum upper = mjMAX_LIMIT, upper2 = mjMAX_LIMIT, lower2;
  Face* face = NULL, *pface = NULL;  // face closest to origin
  int discrete = discreteGeom(obj1) && discreteGeom(obj2);

  // discrete geoms return in a finite number of iterations, a non-zero tolerance avoids
  // absurdly small lower and upper bounds
  mjtNum tolerance = discrete ? mjMINEPATOL : status->tolerance;

  int k, kmax = status->max_iterations < 1000 ? status->max_iterations : 1000;
  for (k = 0; k < kmax; k++) {
    pface = face;

    // find the face closest to the origin (lower bound for penetration depth)
    lower2 = mjMAX_LIMIT;
    for (int i = 0; i < pt->nmap; i++) {
      if (pt->map[i]->dist2 < lower2) {
        face = pt->map[i];
        lower2 = face->dist2;
      }
    }

    // face not valid, return previous face
    if (lower2 > upper2 || !face) {
      face = pface;
      break;
    }

    // check if lower bound is 0
    if (lower2 <= 0) {
      mju_warning("EPA: origin lies on affine hull of face");
      break;
    }

    // compute support point w from the closest face's normal
    mjtNum lower = mju_sqrt(lower2);
    int wi = epaSupport(pt, obj1, obj2, face->v, lower);
    const Vertex* w = pt->verts + wi;
    mjtNum upper_k = dot3(face->v, w->vert) / lower;  // upper bound for kth iteration
    if (upper_k < upper) {
      upper = upper_k;
      upper2 = upper * upper;
    }
    if (upper - lower < tolerance) {
      // terminate without contact when upper < lower on first iteration
      if (k == 0 && upper < lower - 1e-10) {
        face = NULL;
        status->epa_status = mjEPA_NOCONTACT;
      }
      break;
    }

    // check if vertex w is a repeated support point
    if (discrete) {
      int i = 0, nverts = pt->nverts - 1;
      for (; i < nverts; i++) {
        if (w->index1 == pt->verts[i].index1 && w->index2 == pt->verts[i].index2) {
          break;
        }
      }
      if (i != nverts) {
        break;
      }
    }

    pt->horizon.w = w->vert;
    if (horizon(pt, face) < 0) {
      mju_warning(
          "EPA: out of memory for horizon edges on expanding polytope, "
          "set ccd_iterations to at least %d",
          pt->horizon.nedges + 11);
      face = NULL;
      break;
    }

    // unrecoverable numerical issue; at least one face was deleted so nedges is 3 or more
    if (pt->horizon.nedges < 3) {
      face = NULL;
      break;
    }

    // insert w as new vertex and attach faces along the horizon
    int nfaces = pt->nfaces, nedges = pt->horizon.nedges;

    // check if there's enough memory to store new faces
    if (nedges > maxFaces(pt)) {
      mju_warning("EPA: out of memory for faces on expanding polytope");
      break;
    }

    // attach first face
    int hznIndex = pt->horizon.indices[0], hznEdge = pt->horizon.edges[0];
    Face* hznFace = &pt->faces[hznIndex];
    int hznVerts[3] = EPA_VERT_EXPAND(hznFace->verts);
    int v1 = hznVerts[hznEdge],
        v2 = hznVerts[(hznEdge + 1) % 3];
    hznFace->adj[hznEdge] = nfaces;
    mjtNum dist2 = attachFace(pt, wi, v2, v1, nfaces + nedges - 1, hznIndex, nfaces + 1);

    // unrecoverable numerical issue
    if (dist2 == 0) {
      face = NULL;
      break;
    }

    // store face in map
    if (dist2 >= lower2 && dist2 <= upper2) {
      int i = pt->nmap++;
      pt->map[i] = &pt->faces[pt->nfaces - 1];
      pt->map[i]->index = i;
    }

    // attach remaining faces
    for (int i = 1; i < nedges; i++) {
      int cur = nfaces + i;                  // index of attached face
      int next = nfaces + (i + 1) % nedges;  // index of next face

      hznIndex = pt->horizon.indices[i], hznEdge = pt->horizon.edges[i];
      hznFace = &pt->faces[hznIndex];
      int hznVerts2[3] = EPA_VERT_EXPAND(hznFace->verts);
      v1 = hznVerts2[hznEdge];
      v2 = hznVerts2[(hznEdge + 1) % 3];
      hznFace->adj[hznEdge] = cur;
      dist2 = attachFace(pt, wi, v2, v1, cur - 1, hznIndex, next);

      // unrecoverable numerical issue
      if (dist2 == 0) {
        face = NULL;
        break;
      }

      // store face in map
      if (dist2 >= lower2 && dist2 <= upper2) {
        int idx = pt->nmap++;
        pt->map[idx] = &pt->faces[pt->nfaces - 1];
        pt->map[idx]->index = idx;
      }
    }
    pt->horizon.nedges = 0;  // clear horizon

    // no face candidates left
    if (!pt->nmap || !face) {
      break;
    }
  }

  status->epa_iterations = k;
  if (face) {
    status->dist[0] = epaWitness(pt, face, status->x1, status->x2);
    status->nx = 1;
    int verts[3] = EPA_VERT_EXPAND(face->verts);
    obj1->nsupp = 3;
    obj2->nsupp = 3;
    for (int i = 0; i < 3; i++) {
      copy3(obj1->suppvert + 3*i, pt->verts[verts[i]].vert1);
      copy3(obj2->suppvert + 3*i, pt->verts[verts[i]].vert2);
      obj1->suppind[i] = pt->verts[verts[i]].index1;
      obj2->suppind[i] = pt->verts[verts[i]].index2;
    }
  } else {
    status->nx = 0;
    status->dist[0] = 0;
  }
  return face;
}


// ------------------------------------- MultiCCD -------------------------------------------------

// compute area of a quadrilateral (helper for hull4)
static inline mjtNum area4(const mjtNum* hull, int a, int b, int c, int d) {
  mjtNum ca[3], db[3], cross[3];
  sub3(ca, hull + 3*a, hull + 3*c);
  sub3(db, hull + 3*b, hull + 3*d);
  cross3(cross, ca, db);
  return 0.5 * norm3(cross);
}


// prune a convex hull to a maximum area convex quadrilateral
static inline void hull4(int res[4], const mjtNum* hull, int nhull) {
  int a = 0, b = 1, c = 2, d = 3;
  res[0] = 0, res[1] = 1, res[2] = 2, res[3] = 3;
  mjtNum m = area4(hull, a, b, c, d), m_next;
  for (; a < nhull; a++) {
    while (1) {
      int d_next = (d + 1) % nhull;
      m_next = area4(hull, a, b, c, d_next);
      if (m_next <= m) {
        break;
      }
      d = d_next, m = m_next;
      res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      while (1) {
        int c_next = (c + 1) % nhull;
        m_next = area4(hull, a, b, c_next, d);
        if (m_next <= m) {
          break;
        }
        c = c_next, m = m_next;
        res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      }
      while (1) {
        int b_next = (b + 1) % nhull;
        m_next = area4(hull, a, b_next, c, d);
        if (m_next <= m) {
          break;
        }
        b = b_next, m = m_next;
        res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      }
    }
    if (b == a) {
      b = (b + 1) % nhull;
      if (c == b) {
        c = (c + 1) % nhull;
        if (d == c) {
          d = (d + 1) % nhull;
        }
      }
    }
  }
}


// find the normal of a plane perpendicular to the face (given by its normal n) and intersecting the
// face edge (v1, v2)
static mjtNum planeNormal(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3],
                          const mjtNum n[3]) {
  mjtNum v3[3], diff1[3], diff2[3];
  add3(v3, v1, n);
  sub3(diff1, v2, v1);
  sub3(diff2, v3, v1);
  cross3(res, diff1, diff2);

  // normalize isn't needed (cancelled out), but done to avoid asymmetric rounding later on
  mju_normalize3(res);
  return dot3(res, v1);
}


// find what side of a plane a point p lies
static int halfspace(const mjtNum a[3], const mjtNum n[3], const mjtNum p[3]) {
  mjtNum diff[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  return dot3(diff, n) > -mjMINVAL;
}


// compute witness points on face (given by point p and normal n) from clipped vertex
static inline mjtNum witnessOnFace(mjtNum w1[3], mjtNum w2[3], const mjtNum v[3],
                                   const mjtNum* p, const mjtNum n[3], const mjtNum dir[3]) {
  mjtNum d[3] = {v[0] - p[0], v[1] - p[1], v[2] - p[2]};
  addScl3(w1, v, n, -dot3(d, n));
  sub3(d, w1, v);
  mjtNum dist = dot3(d, dir);
  addScl3(w2, w1, dir, -dist);
  return dist;
}


// clip a polygon against another polygon, return number of contacts written
static int polygonClip(mjtNum* x1, mjtNum* x2, mjtNum* dist, int nconmax,
                       const mjtNum* face1, int nface1,
                       const mjtNum* face2, int nface2, const mjtNum n[3],
                       const mjtNum dir[3], mjtNum* buffer, int npolygonmax,
                       mjtNum margin) {
  // clipping face needs to be at least a triangle, or a plane (nface1 == 0)
  if (nface1 != 0 && nface1 < 3) {
    return 0;
  }
  mjtNum* polygon = buffer;
  mjtNum* clipped = polygon + 6 * npolygonmax;
  mjtNum* pn = clipped + 6 * npolygonmax;
  mjtNum* pd = pn + 3 * npolygonmax;

  // compute plane normal and distance to plane for each vertex
  if (nface1 > 0) {
    for (int i = 0; i < nface1 - 1; i++) {
      pd[i] = planeNormal(&pn[3*i], &face1[3*i], &face1[3*i + 3], n);
    }
    pd[nface1 - 1] = planeNormal(&pn[3*(nface1 - 1)], &face1[3*(nface1 - 1)], &face1[0], n);
  }

  // reserve 2 * max_sides as max sides for a clipped polygon
  int npolygon = nface2, nclipped = 0;

  for (int i = 0; i < nface2; i++) {
    copy3(polygon + 3*i, face2 + 3*i);
  }

  // clip the polygon by one edge e at a time
  for (int e = 0; e < (3 * nface1); e += 3) {
    for (int i = 0; i < npolygon; i++) {
      // get edge PQ of the polygon
      mjtNum *P = polygon + 3*i;
      mjtNum *Q = (i < npolygon - 1) ? polygon + 3*(i+1) : polygon;
      mjtNum PQ[3] = {Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]};

      // determine if P and Q are in the halfspace of the clipping edge
      int inside1 = halfspace(face1 + e, pn + e, P);
      int inside2 = halfspace(face1 + e, pn + e, Q);

      // PQ entirely outside the clipping edge, skip
      if (!inside1 && !inside2) {
        continue;
      }

      // edge PQ is inside the clipping edge, add Q
      if (inside1 && inside2) {
        copy3(clipped + 3*nclipped++, Q);
        continue;
      }

      // add new vertex to clipped polygon where PQ intersects the clipping edge
      mjtNum tmp = dot3(pn + e, PQ);
      if (tmp != 0.0) {
        mjtNum t = (pd[e/3] - dot3(pn + e, P)) / tmp;
        if (t >= 0.0 && t <= 1.0) {
          addScl3(clipped + 3*nclipped++, P, PQ, t);
        }
      }

      // add Q as PQ is now back inside the clipping edge
      if (inside2) {
        copy3(clipped + 3*nclipped++, Q);
      }
    }

    // swap clipped and polygon
    mjtNum* tmp = polygon;
    polygon = clipped;
    clipped = tmp;
    npolygon = nclipped;
    nclipped = 0;
  }

  // prune out vertices with distance from the face exceeding margin
  int m = npolygon;
  npolygon = 0;
  for (int i = 0; i < m; i++) {
    mjtNum diff[3];
    sub3(diff, polygon + 3*i, face1);
    if (dot3(diff, n) <= margin) {
      if (npolygon != i) {
        copy3(polygon + 3*npolygon, polygon + 3*i);
      }
      npolygon++;
    }
  }

  if (npolygon < 1) {
    return 0;
  }

  // copy final clipped polygon
  if (nconmax < 5 && npolygon > 4) {
    int idx[4];
    hull4(idx, polygon, npolygon);
    for (int i = 0; i < 4; i++) {
      mjtNum d = witnessOnFace(x1 + 3*i, x2 + 3*i, polygon + 3*idx[i], face1, n, dir);
      if (dist) dist[i] = d;
    }
    return 4;
  }

  // if the face is an edge, remove potential duplicates
  if (nface2 == 2 && npolygon > 2) {
    // find the two most distant vertices in the polygon
    int best1 = 0, best2 = 1;
    mjtNum d = 0;
    for (int i = 0; i < npolygon; i++) {
      for (int j = i + 1; j < npolygon; j++) {
        mjtNum diff[3];
        sub3(diff, polygon + 3*j, polygon + 3*i);
        mjtNum d2 = dot3(diff, diff);
        if (d2 > d) {
          d = d2;
          best1 = i;
          best2 = j;
        }
      }
    }
    mjtNum d0 = witnessOnFace(x1, x2, polygon + 3*best1, face1, n, dir);
    mjtNum d1 = witnessOnFace(x1 + 3, x2 + 3, polygon + 3*best2, face1, n, dir);
    if (dist) {
      dist[0] = d0;
      dist[1] = d1;
    }
    return 2;
  }

  // no pruning needed (cap to max contacts)
  npolygon = (npolygon < nconmax) ? npolygon : nconmax;
  for (int i = 0; i < npolygon; i++) {
    mjtNum d = witnessOnFace(x1 + 3*i, x2 + 3*i, polygon + 3*i, face1, n, dir);
    if (dist) dist[i] = d;
  }
  return npolygon;
}


// compute global coordinates of a local point (l1, l2, l3)
static inline void globalcoord(mjtNum res[3], const mjtNum mat[9], const mjtNum pos[3],
                              mjtNum l1, mjtNum l2, mjtNum l3) {
  // perform mat * (l1, l2, l3) + pos
  res[0] = mat[0]*l1 + mat[1]*l2 + mat[2]*l3;
  res[1] = mat[3]*l1 + mat[4]*l2 + mat[5]*l3;
  res[2] = mat[6]*l1 + mat[7]*l2 + mat[8]*l3;
  if (pos) {
    res[0] += pos[0];
    res[1] += pos[1];
    res[2] += pos[2];
  }
}


// find up to n <= 2 common integers of two arrays, return n
static int intersect(int res[2], const int* arr1, const int* arr2, int n, int m) {
  int count = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (arr1[i] == arr2[j]) {
        res[count++] = arr1[i];
        if (count == 2) return 2;
      }
    }
  }
  return count;
}


// compute possible polygon normals of a mesh given up to 3 vertices
static int meshNormals(mjtNum* res, int resind[3], int dim, mjCCDObj* obj, const int vi[3]) {
  int v1 = vi[0], v2 = vi[1], v3 = vi[2];
  const int* polymap = obj->data.mesh.polymap;
  const mjtNum* polynormal = obj->data.mesh.polynormal;

  if (dim == 3) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    int v2_adr = obj->data.mesh.mpolymapadr[v2];
    int v2_num = obj->data.mesh.mpolymapnum[v2];

    int v3_adr = obj->data.mesh.mpolymapadr[v3];
    int v3_num = obj->data.mesh.mpolymapnum[v3];

    int edgeset[2], faceset[2];
    int n = intersect(edgeset, polymap + v1_adr, polymap + v2_adr, v1_num, v2_num);
    if (n == 0) return 0;
    n = intersect(faceset, edgeset, polymap + v3_adr, n, v3_num);
    if (n == 0) return 0;

    // three vertices on mesh define a unique face
    const mjtNum* normal = polynormal + 3*faceset[0];
    globalcoord(res, obj->mat, NULL, normal[0], normal[1], normal[2]);
    resind[0] = faceset[0];
    return 1;
  }

  if (dim == 2) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    int v2_adr = obj->data.mesh.mpolymapadr[v2];
    int v2_num = obj->data.mesh.mpolymapnum[v2];

    // up to two faces as vertices on mesh define an edge
    int edgeset[2];
    int n = intersect(edgeset, polymap + v1_adr, polymap + v2_adr, v1_num, v2_num);
    if (n == 0) return 0;
    for (int i = 0; i < n; i++) {
      const mjtNum* normal = polynormal + 3*edgeset[i];
      globalcoord(res + 3*i, obj->mat, NULL, normal[0], normal[1], normal[2]);
      resind[i] = edgeset[i];
    }
    return n;
  }

  if (dim == 1) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    // cap number of possible faces intersecting at a vertex
    for (int i = 0; i < v1_num; i++) {
      int index = polymap[v1_adr + i];
      const mjtNum* normal = polynormal + 3*index;
      globalcoord(res + 3*i, obj->mat, NULL, normal[0], normal[1], normal[2]);
      resind[i] = index;
    }
    return v1_num;
  }
  return 0;
}


// compute normal directional vectors along possible edges given by up to two vertices
static int meshEdgeNormals(mjtNum* res, mjtNum* endverts, int dim, mjCCDObj* obj,
                           mjtNum v[9], const int vi[3]) {
  mjtNum* v1 = v;
  mjtNum* v2 = v + 3;
  int v1i = vi[0], v2i = vi[1];
  // mesh data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const float* vert = obj->data.mesh.vert;
  const int* polyvert = obj->data.mesh.polyvert;
  globalcoord(v1, mat, pos, vert[3*v1i + 0], vert[3*v1i + 1], vert[3*v1i + 2]);

  // only one edge
  if (dim == 2) {
    globalcoord(v2, mat, pos, vert[3*v2i + 0], vert[3*v2i + 1], vert[3*v2i + 2]);
    copy3(endverts, v2);
    sub3(res, v2, v1);
    mju_normalize3(res);
    return 1;
  }

  if (dim == 1) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1i];
    int v1_num = obj->data.mesh.mpolymapnum[v1i];

    // loop through all faces with vertex v1
    for (int i = 0; i < v1_num; i++) {
      int idx = obj->data.mesh.polymap[v1_adr + i];
      int adr = obj->data.mesh.polyvertadr[idx];
      int nvert = obj->data.mesh.polyvertnum[idx];
      // find previous vertex in polygon to form edge
      for (int j = 0; j < nvert; j++) {
        if (polyvert[adr + j] == v1i) {
          int k = (j == 0) ? nvert - 1 : j - 1;
          const float* vert_k = vert + 3*polyvert[adr + k];
          globalcoord(endverts + 3*i, mat, pos, vert_k[0], vert_k[1], vert_k[2]);
          sub3(res + 3*i, endverts + 3*i, v1);
          mju_normalize3(res + 3*i);
          break;
        }
      }
    }
    return v1_num;
  }
  return 0;
}


// try recovering the cylinder normal from vertex index
static int cylinderNormals(mjtNum res[9], int resind[3], int dim, mjCCDObj* obj,
                           const int vi[3], const mjtNum dir[3]) {
  if (dim == 1) {
    globalcoord(res, obj->mat, NULL, 0, 0, vi[0] ? -1.0 : 1.0);
    resind[0] = vi[0];
    return 1;
  }
  return 0;
}


// try recovering box normal from collision normal
static int boxNormals2(mjtNum res[9], int resind[3], const mjtNum mat[9], const mjtNum n[3]) {
  // list of box face normals
  mjtNum normals[18] = {1, 0, 0,    -1,  0,  0,
                        0, 1, 0,     0, -1,  0,
                        0, 0, 1,     0,  0, -1};

  // get local coordinates of the normal
  mjtNum local_n[3];
  local_n[0] = mat[0]*n[0] + mat[3]*n[1] + mat[6]*n[2];
  local_n[1] = mat[1]*n[0] + mat[4]*n[1] + mat[7]*n[2];
  local_n[2] = mat[2]*n[0] + mat[5]*n[1] + mat[8]*n[2];
  scl3(local_n, local_n, 1/mju_sqrt(dot3(local_n, local_n)));

  // determine if there is a side close to the normal
  for (int i = 0; i < 6; i++) {
    if (dot3(local_n, normals + 3*i) > mjFACE_TOL) {
      globalcoord(res, mat, NULL, normals[3*i], normals[3*i + 1], normals[3*i + 2]);
      resind[0] = i;
      return 1;
    }
  }
  return 0;
}


// compute possible face normals of a box given up to 3 vertices
static int boxNormals(mjtNum res[9], int resind[3], int dim, mjCCDObj* obj,
                      const int vi[3], const mjtNum dir[3]) {
  int v1 = vi[0], v2 = vi[1], v3 = vi[2];
  const mjtNum* mat = obj->mat;
  if (dim == 3) {
    int c = 0;
    int x = ((v1 & 1) && (v2 & 1) && (v3 & 1)) - (!(v1 & 1) && !(v2 & 1) && !(v3 & 1));
    int y = ((v1 & 2) && (v2 & 2) && (v3 & 2)) - (!(v1 & 2) && !(v2 & 2) && !(v3 & 2));
    int z = ((v1 & 4) && (v2 & 4) && (v3 & 4)) - (!(v1 & 4) && !(v2 & 4) && !(v3 & 4));
    globalcoord(res, mat, NULL, x, y, z);
    int sgn = x + y + z;
    if (x) resind[c++] = 0;
    if (y) resind[c++] = 2;
    if (z) resind[c++] = 4;
    if (sgn == -1) resind[0]++;
    return c == 1 ? 1 : boxNormals2(res, resind, mat, dir);
  }

  if (dim == 2) {
    int c = 0;
    int x = ((v1 & 1) && (v2 & 1)) - (!(v1 & 1) && !(v2 & 1));
    int y = ((v1 & 2) && (v2 & 2)) - (!(v1 & 2) && !(v2 & 2));
    int z = ((v1 & 4) && (v2 & 4)) - (!(v1 & 4) && !(v2 & 4));
    if (x) {
      globalcoord(res, mat, NULL, x, 0, 0);
      resind[c++] = (x > 0) ? 0 : 1;
    }
    if (y) {
      globalcoord(res + 3*c, mat, NULL, 0, y, 0);
      resind[c++] = (y > 0) ? 2 : 3;
    }
    if (z) {
      globalcoord(res + 3*c, mat, NULL, 0, 0, z);
      resind[c++] = (z > 0) ? 4 : 5;
    }
    return c > 0 ? c : boxNormals2(res, resind, mat, dir);
  }

  if (dim == 1) {
    mjtNum x = (v1 & 1) ? 1 : -1;
    mjtNum y = (v1 & 2) ? 1 : -1;
    mjtNum z = (v1 & 4) ? 1 : -1;
    globalcoord(res + 0, mat, NULL, x, 0, 0);
    globalcoord(res + 3, mat, NULL, 0, y, 0);
    globalcoord(res + 6, mat, NULL, 0, 0, z);
    resind[0] = (x > 0) ? 0 : 1;
    resind[1] = (y > 0) ? 2 : 3;
    resind[2] = (z > 0) ? 4 : 5;
    return 3;
  }
  return 0;
}


// compute possible edge normals for box for edge collisions
static int boxEdgeNormals(mjtNum res[9], mjtNum endverts[9], int dim, mjCCDObj* obj,
                          mjtNum v[9], const int vi[3]) {
  mjtNum* v1 = v;
  mjtNum* v2 = v + 3;
  int v1i = vi[0], v2i = vi[1];
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;
  mjtNum x = (v1i & 1) ? size[0] : -size[0];
  mjtNum y = (v1i & 2) ? size[1] : -size[1];
  mjtNum z = (v1i & 4) ? size[2] : -size[2];
  globalcoord(v1, mat, pos, x, y, z);

  if (dim == 2) {
    mjtNum x2 = (v2i & 1) ? size[0] : -size[0];
    mjtNum y2 = (v2i & 2) ? size[1] : -size[1];
    mjtNum z2 = (v2i & 4) ? size[2] : -size[2];
    globalcoord(v2, mat, pos, x2, y2, z2);
    copy3(endverts, v2);
    sub3(res, v2, v1);
    mju_normalize3(res);
    return 1;
  }

  // return 3 adjacent vertices
  if (dim == 1) {
    globalcoord(endverts, mat, pos, -x, y, z);
    sub3(res, endverts, v1);
    mju_normalize3(res);

    globalcoord(endverts + 3, mat, pos, x, -y, z);
    sub3(res + 3, endverts + 3, v1);
    mju_normalize3(res + 3);

    globalcoord(endverts + 6, mat, pos, x, y, -z);
    sub3(res + 6, endverts + 6, v1);
    mju_normalize3(res + 6);
    return 3;
  }
  return 0;
}


// recover edge of a cylinder from witness point
static int cylinderEdgeNormals(mjtNum res[9], mjtNum endverts[9], int dim, mjCCDObj* obj,
                               mjtNum v[9], int v1i) {
  if (dim == 1 || dim == 2) {
    mjtNum sgn = v1i ? 1.0 : -1.0;
    res[0] = sgn * obj->mat[2];
    res[1] = sgn * obj->mat[5];
    res[2] = sgn * obj->mat[8];

    if (obj->margin > 0 && obj->geom >= 0) {
      mjtNum diff[3], local[2];
      sub3(diff, v, obj->pos);
      local[0] = obj->mat[0]*diff[0] + obj->mat[3]*diff[1] + obj->mat[6]*diff[2];
      local[1] = obj->mat[1]*diff[0] + obj->mat[4]*diff[1] + obj->mat[7]*diff[2];
      mjtNum n2 = local[0]*local[0] + local[1]*local[1];
      mjtNum scl = n2 >= mjMINVAL2 ? obj->size[0] / mju_sqrt(n2) : 0;
      globalcoord(v, obj->mat, obj->pos, scl * local[0], scl * local[1], -sgn * obj->size[1]);
    }

    // compute endverts = v1 + res * 2 * half_length
    addScl3(endverts, v, res, 2 * obj->size[1]);
    return 1;
  }
  return 0;
}


// recover edge of a capsule from witness point
static int capsuleEdgeNormals(mjtNum res[3], mjtNum endverts[3], mjCCDObj* obj,
                              mjtNum v[3], const mjtNum w[3]) {
  res[0] = obj->mat[2];
  res[1] = obj->mat[5];
  res[2] = obj->mat[8];

  mjtNum tmp[3];
  sub3(tmp, w, obj->pos);
  mjtNum pr = dot3(tmp, res);
  mjtNum length = obj->size[1];
  addScl3(v, w, res, -pr - length);
  addScl3(endverts, w, res, -pr + length);
  return 1;
}

// recover face of a cylinder (approximated as a 16-gon) from its index
static int cylinderFace(mjtNum res[48], mjCCDObj* obj, int idx) {
  static const mjtNum cos_16[16] = {
    1.000000000000000,  0.923879532511287,  0.707106781186548,  0.382683432365090,
    0.000000000000000, -0.382683432365090, -0.707106781186547, -0.923879532511287,
   -1.000000000000000, -0.923879532511287, -0.707106781186548, -0.382683432365090,
    0.000000000000000,  0.382683432365090,  0.707106781186547,  0.923879532511287
  };
  static const mjtNum sin_16[16] = {
    0.000000000000000,  0.382683432365090,  0.707106781186547,  0.923879532511287,
    1.000000000000000,  0.923879532511287,  0.707106781186548,  0.382683432365090,
    0.000000000000000, -0.382683432365090, -0.707106781186547, -0.923879532511287,
   -1.000000000000000, -0.923879532511287, -0.707106781186548, -0.382683432365090
  };

  // compute 16-gon cylinder face (size[0]: radius, size[1]: half-length)
  mjtNum sgn = idx ? -1.0 : 1.0;
  for (int i = 0; i < 16; i++) {
    mjtNum x = cos_16[i] * obj->size[0];
    mjtNum y = -sin_16[i] * obj->size[0] * sgn;  // use sgn for correct orientation
    globalcoord(res + 3*i, obj->mat, obj->pos, x, y, sgn * obj->size[1]);
  }
  return 16;
}


// recover face of a box from its index
static int boxFace(mjtNum res[12], mjCCDObj* obj, int idx) {
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // compute global coordinates of the box face and face normal
  switch (idx) {
    case 0:  // right
      globalcoord(res + 0, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord(res + 3, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord(res + 6, mat, pos,  size[0], -size[1], -size[2]);
      globalcoord(res + 9, mat, pos,  size[0], -size[1],  size[2]);
      return 4;
     case 1:  // left
      globalcoord(res + 0, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord(res + 3, mat, pos, -size[0],  size[1],  size[2]);
      globalcoord(res + 6, mat, pos, -size[0], -size[1],  size[2]);
      globalcoord(res + 9, mat, pos, -size[0], -size[1], -size[2]);
      return 4;
    case 2:  // top
      globalcoord(res + 0, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord(res + 3, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord(res + 6, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord(res + 9, mat, pos, -size[0],  size[1],  size[2]);
      return 4;
    case 3:  // bottom
      globalcoord(res + 0, mat, pos, -size[0], -size[1],  size[2]);
      globalcoord(res + 3, mat, pos,  size[0], -size[1],  size[2]);
      globalcoord(res + 6, mat, pos,  size[0], -size[1], -size[2]);
      globalcoord(res + 9, mat, pos, -size[0], -size[1], -size[2]);
      return 4;
    case 4:  // front
      globalcoord(res + 0, mat, pos, -size[0],  size[1],  size[2]);
      globalcoord(res + 3, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord(res + 6, mat, pos,  size[0], -size[1],  size[2]);
      globalcoord(res + 9, mat, pos, -size[0], -size[1],  size[2]);
      return 4;
    case 5:  // back
      globalcoord(res + 0, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord(res + 3, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord(res + 6, mat, pos, -size[0], -size[1], -size[2]);
      globalcoord(res + 9, mat, pos,  size[0], -size[1], -size[2]);
      return 4;
  }
  return 0;
}


// recover mesh polygon from its index, return number of edges
static int meshFace(mjtNum* res, mjCCDObj* obj, int idx) {
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;

  int adr = obj->data.mesh.polyvertadr[idx], j = 0;
  int nvert = obj->data.mesh.polyvertnum[idx];
  const float* vert = obj->data.mesh.vert;
  const int* polyvert = obj->data.mesh.polyvert + adr;
  for (int i = nvert - 1; i >= 0; i--) {
    const float* v = vert + 3*polyvert[i];
    globalcoord(res + 3*j++, mat, pos, v[0], v[1], v[2]);
  }
  return nvert;
}


// recover plane normal
static int planeNormals(mjtNum res[3], int resind[1], mjCCDObj* obj) {
  res[0] = obj->mat[2];
  res[1] = obj->mat[5];
  res[2] = obj->mat[8];
  resind[0] = 0;
  return 1;
}


// recover plane point (0 edges as infinite plane needs no edge clipping)
static int planeFace(mjtNum res[3], mjCCDObj* obj) {
  copy3(res, obj->pos);
  return 0;
}


// find two normals that are facing each other within a tolerance, return 1 if found
static inline int alignedFaces(int res[2], const mjtNum* v, int nv,
                               const mjtNum* w, int nw, const mjtNum dir[3],
                               mjtNum face_tol) {
  int found = 0;
  mjtNum best = -face_tol;
  mjtNum dir_tol = mjEDGE_TOL * norm3(dir);
  for (int i = 0; i < nv; i++) {
    if (dot3(v + 3*i, dir) >= -dir_tol) {
      continue;
    }
    for (int j = 0; j < nw; j++) {
      if (dot3(w + 3*j, dir) <= dir_tol) {
        continue;
      }
      mjtNum tmp = dot3(v + 3*i, w + 3*j);
      if (tmp < best) {
        best = tmp;
        res[0] = i;
        res[1] = j;
        found = 1;
      }
    }
  }
  return found;
}


// find two normals that are perpendicular to each other within a tolerance, return 1 if found
static inline int alignedFaceEdge(int res[2], const mjtNum* edge, int nedge,
                                  const mjtNum* face, int nface, const mjtNum dir[3]) {
  for (int i = 0; i < nface; i++) {
    // ignore faces pointing away from the collision direction (negative dot product)
    if (dot3(face + 3*i, dir) <= mjMINVAL) {
      continue;
    }

    for (int j = 0; j < nedge; j++) {
      if (mju_abs(dot3(edge + 3*j, face + 3*i)) < mjEDGE_TOL) {
        res[0] = j;
        res[1] = i;
        return 1;
      }
    }
  }
  return 0;
}


// return number (1, 2 or 3) of dimensions of a simplex; reorder vertices if necessary
static inline int simplexDim(int vi[3], mjtNum v[9]) {
  if (vi[0] != vi[1]) {
    return (vi[2] == vi[0] || vi[2] == vi[1]) ? 2 : 3;
  }
  if (vi[0] != vi[2]) {
    vi[1] = vi[2];
    copy3(v + 3, v + 6);
    return 2;
  }
  return 1;
}


// inflate contacts by margin along normal (computed from witness points if normal is NULL)
static inline void inflate(int nx, mjtNum* x1, mjtNum* x2, mjtNum* dist,
                           mjtNum margin1, mjtNum margin2, const mjtNum normal[3]) {
  if (!margin1 && !margin2) {
    return;
  }
  mjtNum n[3];
  if (normal) {
    copy3(n, normal);
  } else {
    sub3(n, x2, x1);
    mju_normalize3(n);
  }
  for (int i = 0; i < nx; i++) {
    if (margin1) addScl3(x1 + 3*i, x1 + 3*i, n, margin1);
    if (margin2) addScl3(x2 + 3*i, x2 + 3*i, n, -margin2);
    if (dist) dist[i] -= (margin1 + margin2);
  }
}


// compute multiple contacts from GJK/EPA support vertices, assuming 1 initial contact in x1, x2
// return number of contacts
int mjc_multicontact(mjCCDObj* obj1, mjCCDObj* obj2, mjtNum* x1, mjtNum* x2, mjtNum* dist,
                     int nconmax, int nmeshdegmax, int npolygonmax, uint8_t* buffer) {
  if ((obj1->geom_type != mjGEOM_PLANE && obj1->nsupp < 1) ||
      (obj2->geom_type != mjGEOM_PLANE && obj2->nsupp < 1)) {
    return 1;
  }
  if (obj1->geom_type == mjGEOM_MESH && !obj1->data.mesh.mesh_polynum) {
    return 1;
  }
  if (obj2->geom_type == mjGEOM_MESH && !obj2->data.mesh.mesh_polynum) {
    return 1;
  }
  if (obj1->geom_type == mjGEOM_PLANE || obj2->geom_type == mjGEOM_PLANE) {
    nmeshdegmax = nmeshdegmax < 1 ? 1 : nmeshdegmax;
    npolygonmax = npolygonmax < 1 ? 1 : npolygonmax;
  }
  if (obj1->geom_type == mjGEOM_CYLINDER || obj2->geom_type == mjGEOM_CYLINDER) {
    nmeshdegmax = nmeshdegmax < 2 ? 2 : nmeshdegmax;
    npolygonmax = npolygonmax < 16 ? 16 : npolygonmax;
  }
  if (obj1->geom_type == mjGEOM_BOX || obj2->geom_type == mjGEOM_BOX) {
    nmeshdegmax = nmeshdegmax < 3 ? 3 : nmeshdegmax;
    npolygonmax = npolygonmax < 4 ? 4 : npolygonmax;
  }
  if (obj1->geom_type == mjGEOM_CAPSULE || obj2->geom_type == mjGEOM_CAPSULE) {
    nmeshdegmax = nmeshdegmax < 2 ? 2 : nmeshdegmax;
    npolygonmax = npolygonmax < 2 ? 2 : npolygonmax;
  }

  int triface1i[3] = {0}, triface2i[3] = {0};
  mjtNum triface1[9] = {0}, triface2[9] = {0};
  int nface1 = 3, nface2 = 3;
  if (obj1->geom_type != mjGEOM_PLANE) {
    int s1_1 = obj1->nsupp > 1 ? 1 : 0, s1_2 = obj1->nsupp > 2 ? 2 : 0;
    triface1i[0] = obj1->suppind[0];
    triface1i[1] = obj1->suppind[s1_1];
    triface1i[2] = obj1->suppind[s1_2];
    copy3(triface1 + 0, obj1->suppvert + 0);
    copy3(triface1 + 3, obj1->suppvert + 3*s1_1);
    copy3(triface1 + 6, obj1->suppvert + 3*s1_2);
    nface1 = simplexDim(triface1i, triface1);
  }
  if (obj2->geom_type != mjGEOM_PLANE) {
    int s2_1 = obj2->nsupp > 1 ? 1 : 0, s2_2 = obj2->nsupp > 2 ? 2 : 0;
    triface2i[0] = obj2->suppind[0];
    triface2i[1] = obj2->suppind[s2_1];
    triface2i[2] = obj2->suppind[s2_2];
    copy3(triface2 + 0, obj2->suppvert + 0);
    copy3(triface2 + 3, obj2->suppvert + 3*s2_1);
    copy3(triface2 + 6, obj2->suppvert + 3*s2_2);
    nface2 = simplexDim(triface2i, triface2);
  }

  uint8_t* ptr = buffer;
  int* idx1 = (int*)ptr;           ptr += align8(sizeof(int) * nmeshdegmax);
  int* idx2 = (int*)ptr;           ptr += align8(sizeof(int) * nmeshdegmax);
  mjtNum* n1 = (mjtNum*)ptr;       ptr += align8(sizeof(mjtNum) * 3 * nmeshdegmax);
  mjtNum* n2 = (mjtNum*)ptr;       ptr += align8(sizeof(mjtNum) * 3 * nmeshdegmax);
  mjtNum* endverts = (mjtNum*)ptr; ptr += align8(sizeof(mjtNum) * 3 * nmeshdegmax);
  mjtNum* face1 = (mjtNum*)ptr;    ptr += align8(sizeof(mjtNum) * 3 * npolygonmax);
  mjtNum* face2 = (mjtNum*)ptr;    ptr += align8(sizeof(mjtNum) * 3 * npolygonmax);
  mjtNum* polygon = (mjtNum*)ptr;  // buffer for polygonClip

  int nnorms1 = 0, nnorms2 = 0;

  mjtNum dir[3], dir_neg[3];
  if (obj1->geom_type == mjGEOM_PLANE) {
    dir_neg[0] = obj1->mat[2];
    dir_neg[1] = obj1->mat[5];
    dir_neg[2] = obj1->mat[8];
    scl3(dir, dir_neg, -1.0);
  } else if (obj2->geom_type == mjGEOM_PLANE) {
    dir[0] = obj2->mat[2];
    dir[1] = obj2->mat[5];
    dir[2] = obj2->mat[8];
    scl3(dir_neg, dir, -1.0);
  } else {
    sub3(dir, x2, x1);
    sub3(dir_neg, x1, x2);
  }

  // get all possible face normals for each geom
  if (obj1->geom_type == mjGEOM_BOX) {
    nnorms1 = boxNormals(n1, idx1, nface1, obj1, triface1i, dir_neg);
  } else if (obj1->geom_type == mjGEOM_MESH) {
    nnorms1 = meshNormals(n1, idx1, nface1, obj1, triface1i);
  } else if (obj1->geom_type == mjGEOM_CYLINDER) {
    nnorms1 = cylinderNormals(n1, idx1, nface1, obj1, triface1i, dir_neg);
  } else if (obj1->geom_type == mjGEOM_PLANE) {
    nnorms1 = planeNormals(n1, idx1, obj1);
  }
  if (obj2->geom_type == mjGEOM_BOX) {
    nnorms2 = boxNormals(n2, idx2, nface2, obj2, triface2i, dir);
  } else if (obj2->geom_type == mjGEOM_MESH) {
    nnorms2 = meshNormals(n2, idx2, nface2, obj2, triface2i);
  } else if (obj2->geom_type == mjGEOM_CYLINDER) {
    nnorms2 = cylinderNormals(n2, idx2, nface2, obj2, triface2i, dir);
  } else if (obj2->geom_type == mjGEOM_PLANE) {
    nnorms2 = planeNormals(n2, idx2, obj2);
  }

  // determine if any two face normals match (planes skip face tolerance)
  int res[2], edgecon1 = 0, edgecon2 = 0;
  mjtNum face_tol = (obj1->geom_type == mjGEOM_PLANE || obj2->geom_type == mjGEOM_PLANE)
                    ? 0 : mjFACE_TOL;
  if (!alignedFaces(res, n1, nnorms1, n2, nnorms2, dir, face_tol)) {
    // check if edge-face collision
    if (nface1 < 3 && nnorms2 && (nface1 <= nface2 || !nnorms1)) {
      nnorms1 = 0;
      if (obj1->geom_type == mjGEOM_BOX) {
        nnorms1 = boxEdgeNormals(n1, endverts, nface1, obj1, triface1, triface1i);
      } else if (obj1->geom_type == mjGEOM_MESH) {
        nnorms1 = meshEdgeNormals(n1, endverts, nface1, obj1, triface1, triface1i);
      } else if (obj1->geom_type == mjGEOM_CYLINDER) {
        nnorms1 = cylinderEdgeNormals(n1, endverts, nface1, obj1, triface1, triface1i[0]);
      } else if (obj1->geom_type == mjGEOM_CAPSULE) {
        nnorms1 = capsuleEdgeNormals(n1, endverts, obj1, triface1, x1);
      }
      if (!alignedFaceEdge(res, n1, nnorms1, n2, nnorms2, dir)) return 1;
      edgecon1 = 1;

    // check if face-edge collision
    } else if (nface2 < 3) {
      nnorms2 = 0;
      if (obj2->geom_type == mjGEOM_BOX) {
        nnorms2 = boxEdgeNormals(n2, endverts, nface2, obj2, triface2, triface2i);
      } else if (obj2->geom_type == mjGEOM_MESH) {
        nnorms2 = meshEdgeNormals(n2, endverts, nface2, obj2, triface2, triface2i);
      } else if (obj2->geom_type == mjGEOM_CYLINDER) {
        nnorms2 = cylinderEdgeNormals(n2, endverts, nface2, obj2, triface2, triface2i[0]);
      } else if (obj2->geom_type == mjGEOM_CAPSULE) {
        nnorms2 = capsuleEdgeNormals(n2, endverts, obj2, triface2, x2);
      }
      if (!alignedFaceEdge(res, n2, nnorms2, n1, nnorms1, dir_neg)) return 1;
      edgecon2 = 1;
    } else {
      // no multi-contact
      return 1;
    }
  }
  int i = res[0], j = res[1];

  // recover geom1 matching edge or face
  if (edgecon1) {
    copy3(face1, triface1);
    copy3(face1 + 3, endverts + 3*i);
    nface1 = 2;
  } else {
    if (obj1->geom_type == mjGEOM_BOX) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = boxFace(face1, obj1, ind);
    } else if (obj1->geom_type == mjGEOM_MESH) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = meshFace(face1, obj1, ind);
    } else if (obj1->geom_type == mjGEOM_CYLINDER) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = cylinderFace(face1, obj1, ind);
    } else if (obj1->geom_type == mjGEOM_PLANE) {
      nface1 = planeFace(face1, obj1);
    }
  }

  // recover geom2 matching edge or face
  if (edgecon2) {
    copy3(face2, triface2);
    copy3(face2 + 3, endverts + 3*i);
    nface2 = 2;
  } else {
    if (obj2->geom_type == mjGEOM_BOX) {
      nface2 = boxFace(face2, obj2, idx2[j]);
    } else if (obj2->geom_type == mjGEOM_MESH) {
      nface2 = meshFace(face2, obj2, idx2[j]);
    } else if (obj2->geom_type == mjGEOM_CYLINDER) {
      nface2 = cylinderFace(face2, obj2, idx2[j]);
    } else if (obj2->geom_type == mjGEOM_PLANE) {
      nface2 = planeFace(face2, obj2);
    }
  }

  // normal direction for witness recovery and contact inflation
  mjtNum wit_dir[3], normal[3];

  // remaining uninflated margins (capsule edges are already inflated via x1/x2)
  mjtNum margin1 = (obj1->geom >= 0 && obj1->geom_type != mjGEOM_CAPSULE) ? 0.5 * obj1->margin : 0;
  mjtNum margin2 = (obj2->geom >= 0 && obj2->geom_type != mjGEOM_CAPSULE) ? 0.5 * obj2->margin : 0;
  mjtNum margin = margin1 + margin2;

  int nx;
  if (edgecon1 || obj2->geom_type == mjGEOM_PLANE) {
    // face1 is an edge or face2 is a plane; clip face1 against face2
    scl3(wit_dir, n2 + 3*j, -1.0);
    copy3(normal, wit_dir);
    nx = polygonClip(x2, x1, dist, nconmax, face2, nface2, face1, nface1, n2 + 3*j, wit_dir,
                     polygon, npolygonmax, margin);
  } else if (edgecon2 || obj1->geom_type == mjGEOM_PLANE) {
    // face2 is an edge or face1 is a plane; clip face2 against face1
    int ind = edgecon2 ? j : i;
    scl3(wit_dir, n1 + 3*ind, -1.0);
    copy3(normal, n1 + 3*ind);
    nx = polygonClip(x1, x2, dist, nconmax, face1, nface1, face2, nface2, n1 + 3*ind, wit_dir,
                     polygon, npolygonmax, margin);
  } else {
    // face-face collision
    copy3(wit_dir, n2 + 3*j);
    scl3(normal, n2 + 3*j, -1.0);
    nx = polygonClip(x1, x2, dist, nconmax, face1, nface1, face2, nface2, n1 + 3*i, wit_dir,
                     polygon, npolygonmax, margin);
  }

  if (nx > 0) {
    inflate(nx, x1, x2, dist, margin1, margin2, normal);
    return nx;
  }
  return 1;
}


// return size in bytes of the buffer needed for mjc_ccd for a given number of iterations
size_t mjc_ccdSize(int npolygonmax, int nmeshdegmax, int iterations) {
  int maxedges = 24 + (iterations > 35 ? iterations - 35 : 0);
  size_t epa_size = align8(sizeof(Vertex) * (5 + iterations))   // vertices in polytope
                  + align8(sizeof(Face) * 6 * iterations)       // faces in polytope
                  + align8(sizeof(Face*) * 6 * iterations)      // map in polytope
                  + align8(sizeof(int) * maxedges)              // horizon indices
                  + align8(sizeof(int) * maxedges);             // horizon edges

  // allocate room for primitive geoms (multicontact is hardwired for primitive collisions)
  npolygonmax = npolygonmax < 16 ? 16 : npolygonmax;
  nmeshdegmax = nmeshdegmax < 3 ? 3 : nmeshdegmax;
  size_t multiccd_size = align8(sizeof(mjtNum) * 3 * 2 * npolygonmax)
                       + align8(sizeof(mjtNum) * 3 * 2 * npolygonmax)
                       + align8(sizeof(mjtNum) * 3 * npolygonmax)
                       + align8(sizeof(mjtNum) * npolygonmax)
                       + align8(sizeof(int) * nmeshdegmax)
                       + align8(sizeof(int) * nmeshdegmax)
                       + align8(sizeof(mjtNum) * 3 * nmeshdegmax)
                       + align8(sizeof(mjtNum) * 3 * nmeshdegmax)
                       + align8(sizeof(mjtNum) * 3 * nmeshdegmax)
                       + align8(sizeof(mjtNum) * 3 * npolygonmax)
                       + align8(sizeof(mjtNum) * 3 * npolygonmax);

  return epa_size > multiccd_size ? epa_size : multiccd_size;
}


// general convex collision detection
mjtNum mjc_ccd(const mjCCDConfig* config, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  // setup
  obj1->center(status->x1, obj1);
  obj2->center(status->x2, obj2);
  status->gjk_iterations = 0;
  status->epa_iterations = 0;
  status->epa_status = mjEPA_NOCONTACT;
  status->tolerance = config->tolerance;
  status->max_iterations = config->max_iterations;
  int maxcon = sizeof(status->dist) / sizeof(status->dist[0]);
  status->max_contacts = config->max_contacts < maxcon ? config->max_contacts : maxcon;
  status->dist_cutoff = config->dist_cutoff;

  // special handling for sphere, capsule, and margin (shrink to underlying geom)
  if (obj1->geom_type == mjGEOM_SPHERE || obj2->geom_type == mjGEOM_SPHERE ||
      obj1->geom_type == mjGEOM_CAPSULE || obj2->geom_type == mjGEOM_CAPSULE ||
      (obj1->geom >= 0 && obj2->geom >= 0 && (obj1->margin > 0 || obj2->margin > 0))) {
    void (*support1)(mjtNum*, struct _mjCCDObj*, const mjtNum*) = obj1->support;
    void (*support2)(mjtNum*, struct _mjCCDObj*, const mjtNum*) = obj2->support;
    mjtNum full_margin1 = 0, full_margin2 = 0;
    mjtNum margin1 = obj1->margin, margin2 = obj2->margin;

    if (obj1->geom >= 0) {
      full_margin1 = 0.5 * margin1;
      obj1->margin = 0;
    }
    if (obj1->geom_type == mjGEOM_SPHERE) {
      full_margin1 += obj1->size[0];
      obj1->support = mjc_pointSupport;
    } else if (obj1->geom_type == mjGEOM_CAPSULE) {
      full_margin1 += obj1->size[0];
      obj1->support = mjc_lineSupport;
    }

    if (obj2->geom >= 0) {
      full_margin2 = 0.5 * margin2;
      obj2->margin = 0;
    }
    if (obj2->geom_type == mjGEOM_SPHERE) {
      full_margin2 += obj2->size[0];
      obj2->support = mjc_pointSupport;
    } else if (obj2->geom_type == mjGEOM_CAPSULE) {
      full_margin2 += obj2->size[0];
      obj2->support = mjc_lineSupport;
    }

    status->dist_cutoff += full_margin1 + full_margin2;
    gjk(status, obj1, obj2);
    status->dist_cutoff = config->dist_cutoff;

    // restore original margin and support
    obj1->margin = margin1;
    obj2->margin = margin2;
    obj1->support = support1;
    obj2->support = support2;

    // shallow penetration, inflate contact
    if (status->dist[0] > status->tolerance) {
      inflate(status->nx, status->x1, status->x2, status->dist, full_margin1, full_margin2, NULL);
      if (status->dist[0] > status->dist_cutoff) {
        status->dist[0] = mjMAX_LIMIT;
      } else if (config->max_contacts > 1 && status->dist[0] < 0
                 && config->buffer && obj1->nsupp > 0) {
        status->nx = mjc_multicontact(obj1, obj2, status->x1, status->x2, status->dist,
                                      status->max_contacts, config->nmeshdegmax,
                                      config->npolygonmax, config->buffer);
        mjtNum min_dist = status->dist[0];
        for (int i = 1; i < status->nx; i++) {
          if (status->dist[i] < min_dist) {
            min_dist = status->dist[i];
          }
        }
        return min_dist;
      }
      return status->dist[0];
    }

    // contact not needed
    if (!config->max_contacts) {
      status->nx = 0;
      status->dist[0] = 0;
      return 0;
    }

    // deep penetration, reset initial conditions and rerun GJK + EPA
    status->gjk_iterations = 0;
    obj1->center(status->x1, obj1);
    obj2->center(status->x2, obj2);
  }
  gjk(status, obj1, obj2);

  // penetration recovery for contacts not needed
  if (!config->max_contacts) {
    return status->dist[0];
  }

  if (status->dist[0] <= config->tolerance && obj1->nsupp > 1
      && config->buffer && !status->separated) {
    mjtNum gjk_dist = status->dist[0];
    status->dist[0] = 0;  // assume touching
    Polytope pt;
    pt.nfaces = pt.nmap = pt.nverts = pt.horizon.nedges = 0;

    // allocate memory for polytope
    int N = config->max_iterations;
    pt.maxfaces = 6 * N;
    pt.horizon.maxedges = 24 + (N > 35 ? N - 35 : 0);
    uint8_t* buffer = config->buffer;
    pt.verts = (Vertex*)buffer;
    buffer += align8(sizeof(Vertex) * (5 + N));
    pt.faces = (Face*)buffer;
    buffer += align8(sizeof(Face) * (6 * N));
    pt.map = (Face**)buffer;
    buffer += align8(sizeof(Face*) * (6 * N));
    pt.horizon.indices = (int*)buffer;
    buffer += align8(sizeof(int) * pt.horizon.maxedges);
    pt.horizon.edges = (int*)buffer;

    int ret;
    if (obj1->nsupp == 2) {
      ret = polytope2(&pt, status, obj1, obj2);
    } else if (obj1->nsupp == 3) {
      ret = polytope3(&pt, status, obj1, obj2);
    } else {
      ret = polytope4(&pt, status, obj1, obj2);
    }
    status->epa_status = ret;

    // simplex not on boundary (objects are penetrating)
    if (!ret) {
      Face* face = epa(status, &pt, obj1, obj2);

      // get multiple contacts if requested
      if (config->max_contacts > 1 && face) {
        status->nx = mjc_multicontact(obj1, obj2, status->x1, status->x2, status->dist,
                                      status->max_contacts, config->nmeshdegmax,
                                      config->npolygonmax, config->buffer);
      }

      // if EPA found no contact during a distance query, restore GJK distance and witness count
      if (!face && config->dist_cutoff > 0 && status->epa_status == mjEPA_NOCONTACT) {
        status->nx = 1;
        status->dist[0] = gjk_dist;
      }
    }
  }
  mjtNum min_dist = status->dist[0];
  for (int i = 1; i < status->nx; i++) {
    if (status->dist[i] < min_dist) {
      min_dist = status->dist[i];
    }
  }
  return min_dist;
}


// warm-started GJK distance query (no EPA), returns positive distance or 0 if penetrating
mjtNum mjc_gjk(mjCCDObj* obj1, mjCCDObj* obj2, mjtNum x1[3], mjtNum x2[3],
               mjtNum dist_cutoff, mjtNum tolerance, int max_iterations) {
  mjCCDStatus status = {0};
  mju_copy3(status.x1, x1);
  mju_copy3(status.x2, x2);
  status.dist_cutoff = dist_cutoff;
  status.tolerance = tolerance;
  status.max_iterations = max_iterations;

  gjk(&status, obj1, obj2);
  mju_copy3(x1, status.x1);
  mju_copy3(x2, status.x2);
  return status.dist[0];
}

#undef EPA_VERT_EXPAND
