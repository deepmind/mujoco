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

// Continuous-collision geometry for deformables; see the header for scope.

#include "engine/engine_collision_continuous.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_gjk.h"
#include "engine/engine_collision_primitive.h"
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_util.h"
#include "engine/engine_crossplatform.h"  // mj_atomic_add_size_t
#include "engine/engine_init.h"
#include "engine/engine_inline.h"
#include "engine/engine_memory.h"        // mj_arenaAllocByte, mjSTACKALLOC
#include "engine/engine_ray.h"
#include "engine/engine_thread.h"        // mju_numThread, mju_dispatch
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"


// point-triangle: distance, closest point cp, barycentric weights w of cp (for the barrier
// gradient).
mjtNum mjc_PtTri(const mjtNum* p, const mjtNum* a, const mjtNum* b, const mjtNum* c, mjtNum* cp,
                 mjtNum* w) {
  mjc_triWeights(w, p, a, b, c);
  for (int k=0; k < 3; k++) {
    cp[k] = w[0] * a[k] + w[1] * b[k] + w[2] * c[k];
  }
  mjtNum dd[3];
  mju_sub3(dd, p, cp);
  return mju_sqrt(mju_dot3(dd, dd));
}


// closest distance between segment p1p2 and segment q1q2; closest points cp1, cp2 and the segment
// parameters st = {s, t} (cp1 = p1+s*(p2-p1), cp2 = q1+t*(q2-q1)).
mjtNum mjc_SegSeg(const mjtNum* p1, const mjtNum* p2, const mjtNum* q1, const mjtNum* q2,
                  mjtNum* cp1, mjtNum* cp2, mjtNum* st) {
  mjtNum w[4];
  mjc_segWeights(w, p1, p2, q1, q2);
  st[0] = w[0];
  st[1] = w[1];
  for (int k=0; k < 3; k++) {
    cp1[k] = p1[k] + st[0] * (p2[k] - p1[k]);
    cp2[k] = q1[k] + st[1] * (q2[k] - q1[k]);
  }
  mjtNum dd[3];
  mju_sub3(dd, cp1, cp2);
  return mju_sqrt(mju_dot3(dd, dd));
}


// geom types mjc_GeomDist supports; a flex vertex has no continuous contact with the others
int mjc_GeomSupported(int type) {
  return type == mjGEOM_PLANE || type == mjGEOM_SPHERE || type == mjGEOM_CAPSULE ||
         type == mjGEOM_ELLIPSOID || type == mjGEOM_CYLINDER || type == mjGEOM_BOX ||
         type == mjGEOM_MESH;
}


// signed distance from a static geom's surface to world point x (positive outside) + outward unit
// normal n. Closed-form for plane/sphere/capsule/box, GJK for mesh/cylinder/ellipsoid; returns
// +large (no contact) for other types.
mjtNum mjc_GeomDist(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                    const mjtNum* x, mjtNum* n, mjtNum distmax) {
  int type = m->geom_type[gi];
  const mjtNum* size = m->geom_size + 3 * gi;
  mjtNum dx[3] = {x[0] - gpos[0], x[1] - gpos[1], x[2] - gpos[2]};
  n[0] = 0, n[1] = 0, n[2] = 1;

  if (type == mjGEOM_PLANE) {
    n[0] = gmat[2];
    n[1] = gmat[5];
    n[2] = gmat[8];  // local +z axis in world coords
    return mju_dot3(n, dx);
  }
  if (type == mjGEOM_SPHERE) {
    mjtNum L = sqrt(mju_dot3(dx, dx));
    if (L < 1e-12) {
      return -size[0];
    }
    mji_scl3(n, dx, 1.0 / L);
    return L - size[0];
  }
  mjtNum p[3];
  mju_mulMatTVec3(p, gmat, dx);  // world -> geom-local
  mjtNum nl[3] = {0, 0, 0}, dist;
  if (type == mjGEOM_CAPSULE) {
    mjtNum hz = size[1], zc = p[2] > hz ? hz : (p[2] < -hz ? -hz : p[2]);
    mjtNum q[3] = {p[0], p[1], p[2] - zc};  // vector from nearest axis point
    mjtNum L = sqrt(mju_dot3(q, q));
    if (L < 1e-12) {
      nl[0] = 1;
    } else {
      mji_scl3(nl, q, 1.0 / L);
    }
    dist = L - size[0];
  } else if (type == mjGEOM_BOX) {
    mjtNum q[3];
    int outside = 0;
    for (int k=0; k < 3; k++) {
      mjtNum c = p[k] > size[k] ? size[k] : (p[k] < -size[k] ? -size[k] : p[k]);
      q[k] = p[k] - c;
      if (q[k] != 0) outside = 1;
    }
    if (outside) {
      mjtNum L = sqrt(mju_dot3(q, q));
      dist = L;
      mji_scl3(nl, q, 1.0 / L);
    } else {  // inside: least-penetration face
      int ax = 0;
      mjtNum best = 1e30;
      for (int k=0; k < 3; k++) {
        mjtNum pen = size[k] - (p[k] < 0 ? -p[k] : p[k]);
        if (pen < best) {
          best = pen;
          ax = k;
        }
      }
      dist = -best;
      nl[ax] = p[ax] > 0 ? 1 : -1;
    }
  } else if (type == mjGEOM_MESH || type == mjGEOM_CYLINDER || type == mjGEOM_ELLIPSOID) {
    if (type == mjGEOM_MESH && m->mesh_polynum[m->geom_dataid[gi]] <= 0) {
      return 1e30;
    }
    mjCCDObj obj_pt = {.pos = {x[0], x[1], x[2]}, .support = mjc_pointSupport};
    mjCCDObj obj_geom;
    mjc_initCCDObj(&obj_geom, m, NULL, gi, 0);
    mju_copy3(obj_geom.pos, gpos);
    mju_copy9(obj_geom.mat, gmat);

    mjtNum x1[3] = {x[0], x[1], x[2]};
    mjtNum x2[3] = {gpos[0], gpos[1], gpos[2]};
    mjtNum gdist = mjc_gjk(&obj_pt, &obj_geom, x1, x2, distmax, m->opt.ccd_tolerance,
                           m->opt.ccd_iterations);
    if (gdist > distmax) {
      return 1e30;
    }
    if (gdist > 0) {
      mjtNum diff[3] = {x[0] - x2[0], x[1] - x2[1], x[2] - x2[2]};
      if (mju_normalize3(diff) > 0) {
        mju_copy3(n, diff);
        return gdist;
      }
    }

    // inside or on surface: closed-form penetration depth and outward normal
    if (type == mjGEOM_MESH) {
      int meshid = m->geom_dataid[gi];
      const float* vbase = m->mesh_vert + 3 * m->mesh_vertadr[meshid];
      int polyadr = m->mesh_polyadr[meshid], pn = m->mesh_polynum[meshid];
      dist = -1e30;
      const mjtNum* bestn = m->mesh_polynormal + 3 * polyadr;
      for (int i = 0; i < pn; i++) {
        const mjtNum* pnl = m->mesh_polynormal + 3 * (polyadr + i);
        const float* v0 = vbase + 3 * m->mesh_polyvert[m->mesh_polyvertadr[polyadr + i]];
        mjtNum c = pnl[0] * v0[0] + pnl[1] * v0[1] + pnl[2] * v0[2];
        mjtNum dd = mju_dot3(pnl, p) - c;
        if (dd > dist) {
          dist = dd;
          bestn = pnl;
        }
      }
      mju_copy3(nl, bestn);
    } else if (type == mjGEOM_CYLINDER) {
      mjtNum r = mju_sqrt(p[0] * p[0] + p[1] * p[1]);
      mjtNum pr = size[0] - r;
      mjtNum pz = size[1] - mju_abs(p[2]);
      if (pr < pz) {
        dist = -pr;
        if (r > mjMINVAL) {
          nl[0] = p[0] / r;
          nl[1] = p[1] / r;
        } else {
          nl[0] = 1;
        }
      } else {
        dist = -pz;
        nl[2] = p[2] >= 0 ? 1 : -1;
      }
    } else {
      // approximate ellipsoid penetration depth and normal
      mjtNum q[3] = {p[0] / size[0], p[1] / size[1], p[2] / size[2]};
      mjtNum qnorm = mju_norm3(q);
      mjtNum smin = mju_min(size[0], mju_min(size[1], size[2]));
      dist = (qnorm - 1.0) * smin;
      nl[0] = p[0] / (size[0] * size[0]);
      nl[1] = p[1] / (size[1] * size[1]);
      nl[2] = p[2] / (size[2] * size[2]);
      if (mju_normalize3(nl) == 0) nl[2] = 1;
    }
  } else {
    return 1e30;  // unsupported geom -> no barrier
  }
  mju_mulMatVec3(n, gmat, nl);  // geom-local normal -> world
  return dist;
}


// the radius a geom's features carry: a sphere's centre and a capsule's axis stand that far
// inside the surface; a box's or mesh's corners and edges lie on it
static mjtNum featRadius(const mjModel* m, int gi) {
  if (gi < 0) return 0;
  int type = m->geom_type[gi];
  return (type == mjGEOM_SPHERE || type == mjGEOM_CAPSULE) ? m->geom_size[3 * gi] : 0;
}


// world-space VERTICES of a static geom (the features that can poke through a flex triangle
// between its vertices): box -> 8 corners; mesh -> all its vertices; sphere -> its centre;
// capsule -> its two axis endpoints (both carrying the geom's radius, see featRadius); the plane
// none. Returns the count.
int mjc_GeomVerts(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat, mjtNum* out) {
  int type = m->geom_type[gi];
  if (type == mjGEOM_SPHERE) {
    for (int k=0; k < 3; k++) out[k] = gpos[k];
    return 1;
  }
  if (type == mjGEOM_CAPSULE) {
    mjtNum axis[3] = {0, 0, m->geom_size[3 * gi + 1]}, wa[3];  // half-length along the local z
    mju_mulMatVec3(wa, gmat, axis);
    for (int k=0; k < 3; k++) {
      out[k] = gpos[k] - wa[k];
      out[3 + k] = gpos[k] + wa[k];
    }
    return 2;
  }
  if (type == mjGEOM_BOX) {
    const mjtNum* size = m->geom_size + 3 * gi;
    int n = 0;
    for (int sx=-1; sx <= 1; sx += 2)
      for (int sy=-1; sy <= 1; sy += 2)
        for (int sz=-1; sz <= 1; sz += 2) {
          mjtNum loc[3] = {sx * size[0], sy * size[1], sz * size[2]}, wc[3];
          mju_mulMatVec3(wc, gmat, loc);
          for (int k=0; k < 3; k++) out[3 * n + k] = gpos[k] + wc[k];
          n++;
        }
    return n;
  }
  if (type == mjGEOM_MESH) {
    int mid = m->geom_dataid[gi], nv = m->mesh_vertnum[mid];
    const float* vb = m->mesh_vert + 3 * m->mesh_vertadr[mid];
    for (int i=0; i < nv; i++) {
      mjtNum lv[3] = {vb[3 * i], vb[3 * i + 1], vb[3 * i + 2]}, wv[3];
      mju_mulMatVec3(wv, gmat, lv);
      for (int k=0; k < 3; k++) out[3 * i + k] = gpos[k] + wv[k];
    }
    return nv;
  }
  return 0;
}


// world-space EDGES of a static geom (a geom edge can slice through a flex triangle between flex
// vertices): box -> 12 edges; mesh -> its convex-polygon edges (deduped: each shared edge emitted
// once, by the polygon traversing it low->high index); capsule -> its axis (carrying the geom's
// radius, see featRadius). Each edge = two endpoints. Returns count.
int mjc_GeomEdges(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat, mjtNum* out) {
  int type = m->geom_type[gi];
  if (type == mjGEOM_CAPSULE) {
    return mjc_GeomVerts(m, gi, gpos, gmat, out) == 2 ? 1 : 0;  // the two endpoints, one edge
  }
  if (type == mjGEOM_BOX) {
    const mjtNum* size = m->geom_size + 3 * gi;
    int n = 0;
    for (int axis=0; axis < 3; axis++) {
      int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
      for (int s1=-1; s1 <= 1; s1 += 2)
        for (int s2=-1; s2 <= 1; s2 += 2) {
          mjtNum lo[3], hi[3], w1[3], w2[3];
          lo[axis] = -size[axis];
          hi[axis] = size[axis];
          lo[a1] = hi[a1] = s1 * size[a1];
          lo[a2] = hi[a2] = s2 * size[a2];
          mju_mulMatVec3(w1, gmat, lo);
          mju_mulMatVec3(w2, gmat, hi);
          for (int k=0; k < 3; k++) {
            out[6 * n + k] = gpos[k] + w1[k];
            out[6 * n + 3 + k] = gpos[k] + w2[k];
          }
          n++;
        }
    }
    return n;
  }
  if (type == mjGEOM_MESH) {
    int mid = m->geom_dataid[gi];
    const float* vb = m->mesh_vert + 3 * m->mesh_vertadr[mid];
    int pa = m->mesh_polyadr[mid], pn = m->mesh_polynum[mid], n = 0;
    for (int p=0; p < pn; p++) {
      int adr = m->mesh_polyvertadr[pa + p], nvp = m->mesh_polyvertnum[pa + p];
      for (int j=0; j < nvp; j++) {
        int a = m->mesh_polyvert[adr + j], b = m->mesh_polyvert[adr + (j + 1) % nvp];
        if (a >= b) continue;  // dedup: emit each hull edge once (the low->high traversal)
        mjtNum la[3] = {vb[3 * a], vb[3 * a + 1], vb[3 * a + 2]},
               lb[3] = {vb[3 * b], vb[3 * b + 1], vb[3 * b + 2]}, wa[3], wb[3];
        mju_mulMatVec3(wa, gmat, la);
        mju_mulMatVec3(wb, gmat, lb);
        for (int k=0; k < 3; k++) {
          out[6 * n + k] = gpos[k] + wa[k];
          out[6 * n + 3 + k] = gpos[k] + wb[k];
        }
        n++;
      }
    }
    return n;
  }
  return 0;
}


// contact standoff from a pair's detection band: the rest gap min(band, cap). See the header for
// the standoff/band semantics; cap is the caller's skin thickness ceiling.
mjtNum mjc_standoff(mjtNum band, mjtNum cap) {
  return band < cap ? band : cap;
}


// unit gap normal from a closest-point pair. Coincident features (dd == 0) have no defined
// direction, so the normal is zeroed and the contact goes inert for this linearization instead of
// emitting NaN; it is re-linearized at the next configuration. The CCD certificate keeps dd > 0,
// so this only guards configurations that are already degenerate on entry.
static void gapNormal(mjtNum* n, const mjtNum* p1, const mjtNum* p2, mjtNum dd) {
  if (dd < mjMINVAL) {
    n[0] = n[1] = n[2] = 0;
    return;
  }
  for (int k=0; k < 3; k++) n[k] = (p1[k] - p2[k]) / dd;
}


// gap g of a contact at configuration x, plus the barrier gradient direction n, the involved flex
// vertices idv[*nidx] and their weights cw (dg/d(vertex_p) = cw[p]*n). gv/ge are the precomputed
// world-space static-geom corners/edges. Single source of the per-type contact geometry.
// The engine's collision generators only emit an mjContact (dist/normal); this
// returns gap + normal
// + barycentric weights cw (dg/dx) for the barrier gradient/Hessian, which they do not expose.
mjtNum mjc_pairGap(const mjcFlexPair* pair, const mjModel* m, const mjData* d, const mjtNum* x,
                  const mjtNum* gv, const mjtNum* ge, const mjtNum* radii, mjtNum* n,
                  int* idv, mjtNum* cw, int* nidx, mjtNum cutoff) {
  switch (pair->type) {
    case mjcFLEX_VERT_TRI: {  // flex self-contact or cross-flex (vertex v vs triangle A,B,C)
      int v = pair->idx[0], A = pair->idx[1], B = pair->idx[2], C = pair->idx[3];
      mjtNum cp[3], w[3], dd = mjc_PtTri(&x[3 * v], &x[3 * A], &x[3 * B], &x[3 * C], cp, w);
      gapNormal(n, &x[3 * v], cp, dd);
      idv[0] = v;
      idv[1] = A;
      idv[2] = B;
      idv[3] = C;
      cw[0] = 1;
      cw[1] = -w[0];
      cw[2] = -w[1];
      cw[3] = -w[2];
      *nidx = 4;
      return dd;  // MIDSURFACE distance: radii NOT subtracted (delta unchanged; see header)
    }
    case mjcFLEX_EDGE_EDGE: {  // edge a1b1 against edge a2b2: flex self OR cross-flex
      int a1 = pair->idx[0], b1 = pair->idx[1], a2 = pair->idx[2], b2 = pair->idx[3];
      mjtNum cp1[3], cp2[3], st[2],
          dd = mjc_SegSeg(&x[3 * a1], &x[3 * b1], &x[3 * a2], &x[3 * b2], cp1, cp2, st);
      gapNormal(n, cp1, cp2, dd);
      idv[0] = a1;
      idv[1] = b1;
      idv[2] = a2;
      idv[3] = b2;
      cw[0] = 1 - st[0];
      cw[1] = st[0];
      cw[2] = -(1 - st[1]);
      cw[3] = -st[1];
      *nidx = 4;
      return dd;  // MIDSURFACE distance: radii NOT subtracted (delta unchanged; see header)
    }
    case mjcFLEX_VERT_GEOM: {  // flex vertex v vs static geom g surface
      int v = pair->idx[0], gi = pair->g;
      mjtNum dd = mjc_GeomDist(m, gi, d->geom_xpos + 3 * gi, d->geom_xmat + 9 * gi, &x[3 * v], n,
                               cutoff + radii[v]);
      idv[0] = v;
      cw[0] = 1;
      *nidx = 1;
      return dd;  // MIDSURFACE distance: radii NOT subtracted (delta unchanged; see header)
    }
    case mjcGEOM_CORNER_TRI: {  // static geom corner gv[idx0] vs flex triangle A,B,C
      const mjtNum* corner = &gv[3 * pair->idx[0]];
      int A = pair->idx[1], B = pair->idx[2], C = pair->idx[3];
      mjtNum cp[3], w[3], dd = mjc_PtTri(corner, &x[3 * A], &x[3 * B], &x[3 * C], cp, w);
      gapNormal(n, corner, cp, dd);
      idv[0] = A;
      idv[1] = B;
      idv[2] = C;
      cw[0] = -w[0];
      cw[1] = -w[1];
      cw[2] = -w[2];
      *nidx = 3;

      // MIDSURFACE distance: flex radii NOT subtracted (delta unchanged; see header); the geom's
      // own radius is, for a sphere centre or capsule endpoint standing in for a corner
      return dd - featRadius(m, pair->g);
    }
    case mjcGEOM_EDGE_EDGE: {  // static geom edge ge[idx0] vs flex edge a,b
      const mjtNum* eg = &ge[6 * pair->idx[0]];
      int a = pair->idx[1], b = pair->idx[2];
      mjtNum cp1[3], cp2[3], st[2], dd = mjc_SegSeg(eg, eg + 3, &x[3 * a], &x[3 * b], cp1, cp2, st);
      gapNormal(n, cp1, cp2, dd);
      idv[0] = a;
      idv[1] = b;
      cw[0] = -(1 - st[1]);
      cw[1] = -st[1];
      *nidx = 2;
      return dd - featRadius(m, pair->g);  // as above, a capsule axis standing in for an edge
    }
    default:
      mju_error("mjc_pairGap: unknown pair type %d", pair->type);
      return 0;
  }
}


// copy flex pair vertices and return number of vertices; geom features (fixed)
// are excluded -- types 3/4 store the geom index in idx[0], so the flex
// vertices start at idx[1]
int mjc_pairVerts(int* v, const mjcFlexPair* pair) {
  int nv = 2, off = 0;
  switch (pair->type) {
    case mjcFLEX_VERT_TRI:
    case mjcFLEX_EDGE_EDGE:
      nv = 4;
      break;
    case mjcFLEX_VERT_GEOM:
      nv = 1;
      break;
    case mjcGEOM_CORNER_TRI:
      nv = 3;
      off = 1;
      break;
    case mjcGEOM_EDGE_EDGE:
      nv = 2;
      off = 1;
      break;
  }

  for (int i = 0; i < nv; i++) {
    v[i] = pair->idx[off + i];
  }
  return nv;
}


// per-contact barrier activation distance d_hat. Never exceeds the global band, but shrinks to the
// thinnest participating radius: a thin flex (e.g. a drawstring sitting in a thick bag's sleeve)
// then gets a proportionally thin barrier zone instead of resting deep inside the thick neighbour's
// zone (which keeps it permanently active + ratchets kappa -> ill-conditioned). Geom features carry
// no radius (excluded by mjc_pairVerts), so geom/flex and sphere/flex contacts keep the global
// band.
mjtNum mjc_pairBand(const mjcFlexPair* pair, const mjtNum* radii, mjtNum band) {
  int vv[4];
  int nvv = mjc_pairVerts(vv, pair);
  mjtNum g = band;
  for (int q=0; q < nvv; q++) {
    if (radii[vv[q]] < g) {
      g = radii[vv[q]];
    }
  }
  return g;
}


// surface gap of a contact with its flex vertices advanced by t*dxw (geom features fixed);
// gap-only, for the CCD conservative advancement (recomputes the closest feature at the advanced
// configuration). The engine has no advanced-configuration gap evaluator for
// conservative advancement (mjc_ccd is single-config).
static mjtNum conGapAdv(const mjcFlexPair* con, const mjModel* m, const mjData* d, const mjtNum* x,
                        const mjtNum* dxw, mjtNum t, const mjtNum* gv, const mjtNum* ge,
                        const mjtNum* radii) {
  int v[4];
  int nv = mjc_pairVerts(v, con);
  mjtNum P[4][3];

  // the conditional keeps the multiply-add from being contracted into an FMA, so a point's
  // advanced position is bit for bit what it was when the displacement was indexed per free dof
  for (int q=0; q < nv; q++)
    for (int k=0; k < 3; k++) {
      mjtNum dk = dxw[3 * v[q] + k];
      P[q][k] = x[3 * v[q] + k] + (dk != 0 ? t * dk : 0.0);
    }
  mjtNum cp[3], w[3], c1[3], c2[3], st[2];
  switch (con->type) {
    case mjcFLEX_VERT_TRI:
      return mjc_PtTri(P[0], P[1], P[2], P[3], cp, w);  // midsurface, matching mjc_pairGap
    case mjcFLEX_EDGE_EDGE:
      return mjc_SegSeg(P[0], P[1], P[2], P[3], c1, c2, st);  // midsurface, matching mjc_pairGap
    case mjcFLEX_VERT_GEOM: {
      mjtNum nn[3];
      return mjc_GeomDist(m, con->g, d->geom_xpos + 3 * con->g, d->geom_xmat + 9 * con->g, P[0],
                          nn, 1e30);  // midsurface, matching mjc_pairGap
    }
    case mjcGEOM_CORNER_TRI:  // midsurface less the feature's radius, matching mjc_pairGap
      return mjc_PtTri(&gv[3 * con->idx[0]], P[0], P[1], P[2], cp, w) - featRadius(m, con->g);
    case mjcGEOM_EDGE_EDGE: {
      const mjtNum* eg = &ge[6 * con->idx[0]];
      return mjc_SegSeg(eg, eg + 3, P[0], P[1], c1, c2, st) - featRadius(m, con->g);
    }
    default:
      mju_error("conGapAdv: unknown pair type %d", con->type);
      return 0;
  }
}


mjtNum mjc_advance(const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* dxw,
                   const mjtNum* gv, const mjtNum* ge, const mjtNum* radii, int nfv,
                   const mjcFlexPair* cand, int ncand, const mjtNum* cgap, const int* pt2flex,
                   int* approut, mjtNum* toiout) {
  mjtNum alpha = 1.0;
  if (approut)
    for (int c=0; c < ncand; c++)
      approut[c] = 0;  // Alg.3: per-pair "the proxy approaches this contact"
  if (toiout)
    for (int c=0; c < ncand; c++)
      toiout[c] = 1.0;  // Alg.3: per-pair collision time, 1 == does not collide this step
  for (int c=0; c < ncand; c++) {
    const mjcFlexPair* con = &cand[c];
    // mean-removal (don't throttle COHERENT motion) is valid ONLY for a true SAME-FLEX
    // self-contact. For INTER-flex (e.g. drawstring vs bag) the two sides move independently, so
    // removing the mean underestimates the closing speed and lets one tunnel through the other.
    // Gate it on same-flex (both sides in the same flex via pt2flex).
    int other = (con->type == mjcFLEX_VERT_TRI) ? con->idx[1] : con->idx[2];
    int v[4],
        self = (con->type <= mjcFLEX_EDGE_EDGE) && (con->idx[0] < nfv) && (other < nfv) &&
               (pt2flex[con->idx[0]] == pt2flex[other]);
    int nv = mjc_pairVerts(v, con);
    mjtNum dp[4][3], mean[3] = {0, 0, 0};
    for (int q=0; q < nv; q++)
      for (int k=0; k < 3; k++) dp[q][k] = dxw[3 * v[q] + k];
    if (self) {
      for (int q=0; q < nv; q++)
        for (int k=0; k < 3; k++) mean[k] += dp[q][k];
      for (int k=0; k < 3; k++) mean[k] /= nv;
      for (int q=0; q < nv; q++)
        for (int k=0; k < 3; k++) dp[q][k] -= mean[k];
    }
    mjtNum l;  // bound on the gap-shrink rate per unit alpha
    if (con->type == mjcFLEX_VERT_TRI) {
      mjtNum a0 = sqrt(mju_dot3(dp[0], dp[0])), b0 = 0;
      for (int q=1; q < 4; q++) {
        mjtNum s = sqrt(mju_dot3(dp[q], dp[q]));
        if (s > b0) b0 = s;
      }
      l = a0 + b0;
    } else if (con->type == mjcFLEX_EDGE_EDGE) {
      mjtNum a0 = 0, b0 = 0;
      for (int q=0; q < 2; q++) {
        mjtNum s = sqrt(mju_dot3(dp[q], dp[q]));
        if (s > a0) a0 = s;
      }
      for (int q=2; q < 4; q++) {
        mjtNum s = sqrt(mju_dot3(dp[q], dp[q]));
        if (s > b0) b0 = s;
      }
      l = a0 + b0;
    } else {
      l = 0;
      for (int q=0; q < nv; q++) {
        mjtNum s = sqrt(mju_dot3(dp[q], dp[q]));
        if (s > l) l = s;
      }
    }
    if (l < 1e-12) continue;
    mjtNum g0 =
        cgap[c];  // true gap at x (>= standoff delta at rest, so the CCD always has room: no lock)
    if (g0 <= 0) continue;  // already at/under the surface: the contact solver owns it
    if (l <= 0.8 * g0)
      continue;  // full alpha=1 step shrinks gap by <= l, stays above the 20% floor
    if (approut)
      approut[c] =
          1;  // reaches the bisection -> the proxy closes this pair's gap this step (Alg.3 add)

    // this pair's own collision time, found on its own and capped at the full step: the earliest
    // filter compares these across pairs, so none may be cut short at the running minimum of
    // the others, which would tie every later pair with the earliest one
    mjtNum gtarget = 0.2 * g0, t = 0;
    for (int it=0; it < 32; it++) {
      mjtNum g = conGapAdv(con, m, d, x, dxw, t, gv, ge, radii);
      mjtNum room = g - gtarget;
      if (room <= 1e-9 * g0) break;
      t += room / l;
      if (t >= 1.0) {
        t = 1.0;
        break;
      }
    }
    if (toiout) {
      toiout[c] = t;
    }
    if (t < alpha) {
      alpha = t;
    }
  }
  return alpha;
}


// the radii a midsurface gap reads ABOVE the old skin gap. Used ONLY to keep the broad-phase
// detection band at its pre-midsurface reach; every pair type measures at the flex midsurface,
// so each flex participant contributes its radius (geoms have none: their surface is the surface).
static mjtNum bandOffset(const mjcFlexPair* con, const mjtNum* radii) {
  switch (con->type) {
    case mjcFLEX_VERT_TRI:
      return radii[con->idx[0]] + radii[con->idx[1]];  // vertex + triangle flex radius
    case mjcFLEX_EDGE_EDGE:
      return radii[con->idx[0]] + radii[con->idx[2]];  // both edges (flex) radii
    case mjcFLEX_VERT_GEOM:
      return radii[con->idx[0]];  // flex vertex radius
    case mjcGEOM_CORNER_TRI:
      return radii[con->idx[1]];  // flex triangle radius
    case mjcGEOM_EDGE_EDGE:
      return radii[con->idx[1]];  // flex edge radius
    default:
      return 0.0;
  }
}


#define CAND_BLOCK_CAP 64

// one fixed-size block of candidates in the stack pool of a threaded broad phase
typedef struct {
  mjcFlexPair pairs[CAND_BLOCK_CAP];
  int n;
  int next;
} candBlock;

// shared pool of candidate blocks on the mjData stack for a threaded broad phase
typedef struct {
  candBlock* blocks;
  size_t next;
  size_t cap;
} candBlockPool;

// where addCand puts a candidate: the arena (pool == NULL; the serial broad phase and the merge of
// a threaded one), or a linked list of stack blocks (pool != NULL; one per task of a threaded
// broad phase)
typedef struct {
  mjData* d;             // arena mode: arena owner
  mjcFlexPair** base;    // arena mode: start of the pushed array
  int* nc;               // arena mode: pushed count
  int full;              // set once the arena or block pool is full
  candBlockPool* pool;   // block mode: shared stack block pool (NULL in arena mode)
  int head, tail;        // block mode: first and last block indices (-1 when empty)
} candSink;


// push one candidate onto the sink
static void sinkPush(candSink* s, const mjcFlexPair* con) {
  if (s->pool) {
    if (s->full) return;
    int b = s->tail;
    if (b < 0 || s->pool->blocks[b].n == CAND_BLOCK_CAP) {
      size_t idx = mj_atomic_add_size_t(&s->pool->next, 1);
      if (idx >= s->pool->cap) {
        s->full = 1;
        return;
      }
      b = (int)idx;
      s->pool->blocks[b].n = 0;
      s->pool->blocks[b].next = -1;
      if (s->tail >= 0) {
        s->pool->blocks[s->tail].next = b;
      } else {
        s->head = b;
      }
      s->tail = b;
    }
    candBlock* blk = &s->pool->blocks[b];
    blk->pairs[blk->n++] = *con;
    return;
  }
  mjcFlexPair* p =
      (mjcFlexPair*)mj_arenaAllocByte(s->d, sizeof(mjcFlexPair), _Alignof(mjcFlexPair));
  if (!p) {
    s->full = 1;
    return;
  }
  *p = *con;
  if (*s->nc == 0) *s->base = p;
  (*s->nc)++;
}


// append a candidate contact if its gap at x is below the (margin-inflated) detection threshold.
// Candidates are pushed onto the arena one at a time, as mj_collision pushes its pairs: consecutive
// pushes of one type are adjacent, so *base + *nc indexes them as an array. Once the arena is full
// sink->full is set and nothing more is pushed. (A threaded broad phase pushes into per-task block
// lists instead, see candSink.)
static void addCand(mjcFlexPair con, const mjModel* m, const mjData* d, const mjtNum* x,
                    const mjtNum* gv, const mjtNum* ge, const mjtNum* radii, mjtNum thresh,
                    const mjtNum* dfrom, const mjtNum* dto, mjtNum ghat, candSink* sink) {
  if (sink->full) return;
  // flex-flex pre-test: the distance between the two features' bounding boxes bounds their
  // (midsurface) gap from below, so a pair whose boxes are beyond the detection threshold below
  // fails it in any case. The margin keeps the test clear of the exact distance's round-off
  if (con.type <= mjcFLEX_EDGE_EDGE) {
    int n1 = con.type == mjcFLEX_EDGE_EDGE ? 2 : 1;
    mjtNum lim = bandOffset(&con, radii) + thresh;
    lim += 1e-9 * (lim > 0 ? lim : -lim) + 1e-12;
    mjtNum dist2 = 0;
    for (int c=0; c < 3; c++) {
      mjtNum lo1 = x[3 * con.idx[0] + c], hi1 = lo1;
      for (int q=1; q < n1; q++) {
        mjtNum v = x[3 * con.idx[q] + c];
        if (v < lo1) lo1 = v;
        if (v > hi1) hi1 = v;
      }
      mjtNum lo2 = x[3 * con.idx[n1] + c], hi2 = lo2;
      for (int q=n1+1; q < 4; q++) {
        mjtNum v = x[3 * con.idx[q] + c];
        if (v < lo2) lo2 = v;
        if (v > hi2) hi2 = v;
      }
      mjtNum gap = lo2 - hi1 > lo1 - hi2 ? lo2 - hi1 : lo1 - hi2;
      if (gap > 0) dist2 += gap * gap;
    }
    if (lim > 0 && dist2 > lim * lim) return;
  }
  mjtNum n[3], cw[4];
  int idv[4], nidx;
  mjtNum g = mjc_pairGap(&con, m, d, x, gv, ge, radii, n, idv, cw, &nidx, thresh);
  // DETECTION band. A flex-flex gap is now the midsurface distance, so it reads r1+r2 larger than
  // the pre-midsurface convention; add that back HERE (and only here) so the broad-phase reaches
  // exactly as far as it used to. This offset belongs to the band, NOT to the rest target
  // -- the rest target (mjc_standoff(mjc_pairBand)) is unchanged and must stay that way.
  mjtNum bo = bandOffset(&con, radii);
  if (g >= bo + thresh) return;

  // KEEP penetrating geom pairs (g < 0): the normal is still defined there and the solver's
  // restoring force is what pushes the flex back out, so dropping them leaves the penetration
  // unopposed; mjc_advance (g0 <= 0 -> no cap) and the merge (cgap <= 0 -> admit) already expect
  // them. Flex-flex distances are UNSIGNED, so g <= 0 means coincident features with no defined
  // normal.
  if (con.type <= mjcFLEX_EDGE_EDGE && g <= 0) return;

  // closing-bound prune: the gap is a minimum over the two features, each the convex hull of its
  // vertices, so over the step it shrinks by at most the largest displacement of a vertex of one
  // feature relative to a vertex of the other (a geom feature is static); a pair beyond its band
  // plus that bound cannot become active this step -> drop it. Bulk motion cancels in the
  // relative displacement, so a settling bag does not flood the list. The closest points' weights
  // at x are no bound: under deformation the closest points move to another region of the
  // features (ContinuousCollisionTest.CandidatesCoverARotatingTriangle).
  int vv[4], nvv = mjc_pairVerts(vv, &con);
  int n1 = con.type == mjcFLEX_EDGE_EDGE ? 2
           : (con.type == mjcFLEX_VERT_TRI || con.type == mjcFLEX_VERT_GEOM) ? 1 : 0;
  mjtNum dv[5][3] = {{0}};  // per pair vertex, the fifth entry standing in for a static feature
  for (int q=0; q < nvv; q++)
    for (int c=0; c < 3; c++) dv[q][c] = dto[3 * vv[q] + c] - dfrom[3 * vv[q] + c];
  int i0 = n1 ? 0 : 4, i1 = n1 ? n1 : 5, j0 = nvv > n1 ? n1 : 4, j1 = nvv > n1 ? nvv : 5;
  mjtNum bound = 0;
  for (int i=i0; i < i1; i++)
    for (int j=j0; j < j1; j++) {
      mjtNum dd[3];
      mju_sub3(dd, dv[i], dv[j]);
      mjtNum l = mju_sqrt(mju_dot3(dd, dd));
      if (l > bound) bound = l;
    }
  if (g < bo + mjc_pairBand(&con, radii, ghat) + bound) {
    sinkPush(sink, &con);
  }
}


// descend flex f's element BVH (built and AABB-refreshed by mj_flex at the step's xold), collecting
// the leaf element ids whose (radius-inflated) node AABB overlaps the query box [c +/- h]. Replaces
// the hand-rolled uniform spatial hash: same "nearby elements" query, but the engine's hierarchy.
// The node AABBs already include flex_radius, so a query half of thresh is a conservative superset
// (no pair within thresh is missed; the narrowphase then filters). stack must hold flex_bvhnum
// ints.
static int bvhBox(const mjModel* m, const mjData* d, int f, const mjtNum* c, const mjtNum* h,
                  int* stack, int* out, int maxout) {
  int bvhadr = m->flex_bvhadr[f];
  if (bvhadr < 0) return 0;
  const int* child = m->bvh_child + 2 * bvhadr;
  const int* nodeid = m->bvh_nodeid + bvhadr;
  const mjtNum* aabb = d->bvh_aabb_dyn + 6 * (bvhadr - m->nbvhstatic);
  int ns = 0, nout = 0;
  stack[ns++] = 0;
  while (ns) {
    int node = stack[--ns];
    const mjtNum* na = aabb + 6 * node;  // [center(3), halfsize(3)]
    if (mju_abs(na[0] - c[0]) > na[3] + h[0] || mju_abs(na[1] - c[1]) > na[4] + h[1] ||
        mju_abs(na[2] - c[2]) > na[5] + h[2])
      continue;  // box-box separation -> prune
    int c0 = child[2 * node], c1 = child[2 * node + 1];
    if (c0 < 0 && c1 < 0) {
      if (nout < maxout) out[nout++] = nodeid[node];
    }  // leaf -> element id
    else {
      if (c0 >= 0) stack[ns++] = c0;
      if (c1 >= 0) stack[ns++] = c1;
    }
  }
  return nout;
}


// contype/conaffinity filter, the rule of the native pipeline (filterBitmask in
// engine_collision_driver.c): a pair is filtered unless one side's type meets the other's affinity
static int maskFiltered(int contype1, int conaffinity1, int contype2, int conaffinity2) {
  return !(contype1 & conaffinity2) && !(contype2 & conaffinity1);
}


// the pairs the IPC step owns under the ipc flag (see the header)
int mjc_ipcOwnsFlexFlex(const mjModel* m, int f1, int f2) {
  return mjENABLED(mjENBL_IPC) && m->flex_dim[f1] == 2 && m->flex_dim[f2] == 2;
}


int mjc_ipcOwnsFlexGeom(const mjModel* m, int f, int g) {
  return mjENABLED(mjENBL_IPC) && m->flex_dim[f] == 2 &&
         m->body_weldid[m->geom_bodyid[g]] == 0 && mjc_GeomSupported(m->geom_type[g]);
}


// whether points a and b ride on one body (ptbody is -1 for points without a body)
static inline int sameBody(const int* ptbody, int a, int b) {
  return ptbody[a] >= 0 && ptbody[a] == ptbody[b];
}


// the broad phase's inputs and derived per-call tables, shared read-only by its work units
typedef struct {
  const mjModel* m;
  const mjData* d;
  const mjtNum* x;
  const mjtNum* gv;
  const mjtNum* ge;
  const int* gvgeom;
  const int* gegeom;
  int ngv, nge;
  const mjtNum* radii;
  mjtNum thresh, threshGeom;
  const mjtNum* dfrom;
  const mjtNum* dto;
  mjtNum ghat;
  int nfv;
  const int* fidx;
  const int* pjnv;
  const int* flist;
  const int* fxadr;
  int nfd;
  const int* pt2flex;
  const mjtNum* dl;      // per point travel                                         (nfv)
  const mjtNum* fsweep;  // per flex largest travel                                  (nfd)
  const mjtNum* fstale;  // per flex largest distance from the BVH's configuration   (nfd)
  const int* fok;        // flex-pair filter                                         (nfd*nfd)
  const int* ptbody;     // body of each point                                       (npt_all)
  // the points that meet the static geoms: their bounding box and largest margin (padded), so a
  // geom whose box misses the padded point box has no candidate; npg: number of such points
  mjtNum pglo[3], pghi[3], pgmarg;
  int npg;
} candCtx;

// a unit's BVH query scratch; stampG dedups the edges of one query (stamp == qid)
typedef struct {
  int* stk;
  int* outel;
  int* stampG;
  int qid;
} candScratch;

// work units of the broad phase, in the serial order: the point-vs-static-geom tests over geoms
// [lo, hi), then per queried flex k its geom corners, geom edges, vertices and, per flex kj <= k,
// the edges of flex kj, each over the index range [lo, hi)
enum { candGEOMVERT = 0, candCORNER, candGEDGE, candVERT, candEDGE };
typedef struct {
  int kind, k, kj, lo, hi;
} candUnit;


// run one work unit, pushing its candidates onto sink in the serial order
static void candRun(const candCtx* cx, const candUnit* u, candScratch* w, candSink* sink) {
  const mjModel* m = cx->m;
  const mjData* d = cx->d;
  const mjtNum* x = cx->x;
  const mjtNum* gv = cx->gv;
  const mjtNum* ge = cx->ge;
  const mjtNum* radii = cx->radii;
  const mjtNum* dfrom = cx->dfrom;
  const mjtNum* dto = cx->dto;
  const mjtNum* dl = cx->dl;
  mjtNum ghat = cx->ghat;
  int nfd = cx->nfd;

  if (u->kind == candGEOMVERT) {
    for (int gi=u->lo; gi < u->hi; gi++) {  // free point (flex vert) vs STATIC geom
      if (m->geom_contype[gi] == 0 && m->geom_conaffinity[gi] == 0) continue;  // non-colliding
      if (!mjc_GeomSupported(m->geom_type[gi])) continue;  // its native rows are kept instead
      if (m->body_weldid[m->geom_bodyid[gi]] != 0)
        continue;  // STATIC geoms ONLY: a MOVABLE rigid geom (ball/limb) is a soft type-3
                   // contact, never a frozen penetration-free obstacle
      // geom-level cull: bound the points by the geom's true world AABB (the rotated local
      // geom_aabb) + per-point margin, before the per-face distance. Much tighter than the
      // bounding sphere for the elongated convex-decomposition slabs. Planes are infinite -> no
      // cull (their distance is O(1) anyway).
      int isplane = (m->geom_type[gi] == mjGEOM_PLANE);
      mjtNum wc[3], wh[3];
      if (!isplane) {
        const mjtNum* la = m->geom_aabb + 6 * gi;
        const mjtNum* gp = d->geom_xpos + 3 * gi;
        const mjtNum* gR = d->geom_xmat + 9 * gi;
        mju_mulMatVec3(wc, gR, la);
        for (int k=0; k < 3; k++) wc[k] += gp[k];  // world AABB center
        for (int k=0; k < 3; k++)
          wh[k] = mju_abs(gR[3 * k]) * la[3] + mju_abs(gR[3 * k + 1]) * la[4] +
                  mju_abs(gR[3 * k + 2]) * la[5];
        // geom-level cull against the box of all the points: a point in the geom's box widened
        // by its margin lies in the point box, so a geom whose box misses it has no candidate
        int miss = 0;
        for (int k=0; k < 3; k++) {
          if (wc[k] - wh[k] - cx->pgmarg > cx->pghi[k] || wc[k] + wh[k] + cx->pgmarg < cx->pglo[k])
            miss = 1;
        }
        if (miss) continue;
      }
      if (!cx->npg) continue;
      for (int v=0; v < cx->nfv; v++) {  // free points and points riding a moving body
        if (cx->fidx[v] < 0 && !(cx->pjnv && cx->pjnv[v] > 0)) continue;
        int f = cx->flist[cx->pt2flex[v]];
        if (maskFiltered(m->geom_contype[gi], m->geom_conaffinity[gi], m->flex_contype[f],
                         m->flex_conaffinity[f]))
          continue;
        mjtNum reach = cx->thresh + dl[v];  // the point's own travel: the geom is static
        mjtNum marg = reach + radii[v];
        if (!isplane) {  // world-AABB cull
          if (mju_abs(x[3 * v] - wc[0]) > wh[0] + marg ||
              mju_abs(x[3 * v + 1] - wc[1]) > wh[1] + marg ||
              mju_abs(x[3 * v + 2] - wc[2]) > wh[2] + marg)
            continue;
        }
        mjcFlexPair con = {mjcFLEX_VERT_GEOM, {v, 0, 0, 0}, gi};
        addCand(con, m, d, x, gv, ge, radii, reach, dfrom, dto, ghat, sink);
      }
    }
    return;
  }

  // query flex fk's element BVH
  int k = u->k;
  int fk = cx->flist[k];
  int ne_k = m->flex_elemnum[fk], ea_k = m->flex_edgeadr[fk], off_k = cx->fxadr[k];
  const int* el_k = m->flex_elem + m->flex_elemdataadr[fk];
  const int* eme_k = m->flex_elemedge + m->flex_elemedgeadr[fk];
  mjtNum rk = m->flex_radius[fk];
  int ct_k = m->flex_contype[fk], ca_k = m->flex_conaffinity[fk];
  mjtNum fsweep_k = cx->fsweep[k], fstale_k = cx->fstale[k];
  int* stk = w->stk;
  int* outel = w->outel;
  int* stampG = w->stampG;

  // (rigid sphere-vs-flex-triangle hard contact removed in increment A -- see note above.)
  // geom-corner vs flex triangle (type 3); geom is static (one-sided) -> tighter threshGeom (a
  // convex-decomposition bin's ~1600 edges otherwise produce hundreds of thousands of
  // candidates). Only the flex side moves: its reach grows by the flex's travel, and its BVH
  // query by the staleness of the tree as well.
  // A feature carries its geom (con.g) for the masks and for the radius a sphere centre or
  // capsule axis stands inside the surface, which widens its query box.
  mjtNum reachG = cx->threshGeom + fsweep_k, boxG = reachG + rk + fstale_k;
  if (u->kind == candCORNER) {
    for (int c=u->lo; c < u->hi; c++) {
      int gi = cx->gvgeom[c];
      if (maskFiltered(m->geom_contype[gi], m->geom_conaffinity[gi], ct_k, ca_k)) continue;
      mjtNum bx = boxG + featRadius(m, gi);
      mjtNum qhvG[3] = {bx, bx, bx};
      int n = bvhBox(m, d, fk, &gv[3 * c], qhvG, stk, outel, ne_k);
      for (int i=0; i < n; i++) {
        int e = outel[i];
        mjcFlexPair con = {mjcGEOM_CORNER_TRI,
                           {c, off_k + el_k[3 * e], off_k + el_k[3 * e + 1],
                            off_k + el_k[3 * e + 2]},
                           gi};
        addCand(con, m, d, x, gv, ge, radii, reachG, dfrom, dto, ghat, sink);
      }
    }
    return;
  }

  // geom-edge vs flex edge (type 4); dedup the shared triangle edges per query via stampG.
  if (u->kind == candGEDGE) {
    for (int c=u->lo; c < u->hi; c++) {
      int gi = cx->gegeom[c];
      if (maskFiltered(m->geom_contype[gi], m->geom_conaffinity[gi], ct_k, ca_k)) continue;
      const mjtNum* p0 = &ge[6 * c];
      const mjtNum* p1 = &ge[6 * c + 3];
      mjtNum qc[3], qh[3], bx = boxG + featRadius(m, gi);
      for (int kk=0; kk < 3; kk++) {
        qc[kk] = 0.5 * (p0[kk] + p1[kk]);
        qh[kk] = 0.5 * mju_abs(p1[kk] - p0[kk]) + bx;
      }
      int n = bvhBox(m, d, fk, qc, qh, stk, outel, ne_k);
      int qid = ++w->qid;
      for (int i=0; i < n; i++) {
        int e = outel[i];
        for (int j=0; j < 3; j++) {
          int e2 = eme_k[3 * e + j];
          if (stampG[e2] == qid) continue;
          stampG[e2] = qid;
          mjcFlexPair con = {mjcGEOM_EDGE_EDGE,
                             {c, off_k + m->flex_edge[2 * (ea_k + e2)],
                              off_k + m->flex_edge[2 * (ea_k + e2) + 1], 0},
                             gi};
          addCand(con, m, d, x, gv, ge, radii, reachG, dfrom, dto, ghat, sink);
        }
      }
    }
    return;
  }

  // flex vertex vs flex triangle (type 0): self (same flex) + inter-flex, both under fok.
  // Asymmetric (vert vs tri), so all verts query every flex's BVH -- both directions are
  // distinct contacts.
  if (u->kind == candVERT) {
    for (int v=u->lo; v < u->hi; v++) {
      int kv = cx->pt2flex[v];
      if (!cx->fok[k * nfd + kv]) continue;  // filtered flex pair
      // per-pair band: the thinner flex sets it, capped by the global band; the reach adds the
      // travel of both sides, the BVH query the tree's staleness as well
      mjtNum thv = 3.0 * mju_min(ghat, mju_min(radii[v], rk)) + dl[v] + fsweep_k;
      mjtNum bv = thv + radii[v] + fstale_k;
      mjtNum qh[3] = {bv, bv, bv};
      int n = bvhBox(m, d, fk, &x[3 * v], qh, stk, outel, ne_k);
      for (int i=0; i < n; i++) {
        int e = outel[i];
        int A = off_k + el_k[3 * e], B = off_k + el_k[3 * e + 1], C = off_k + el_k[3 * e + 2];
        if (kv == k && (v == A || v == B || v == C)) continue;  // the self-adjacent triangle
        if (sameBody(cx->ptbody, v, A) || sameBody(cx->ptbody, v, B) ||
            sameBody(cx->ptbody, v, C)) {
          continue;  // vertex rides on the body of a triangle vertex
        }
        mjcFlexPair con = {mjcFLEX_VERT_TRI, {v, A, B, C}, -1};
        addCand(con, m, d, x, gv, ge, radii, thv, dfrom, dto, ghat, sink);
      }
    }
    return;
  }

  // flex edge vs flex edge (type 1): symmetric, so canonical -- querying flex kj <= k, and e2 >
  // e1 within a flex. Self (kj==k) and inter-flex (kj<k) both under fok.
  int kj = u->kj;
  int self = (kj == k);
  if (!cx->fok[k * nfd + kj]) return;  // filtered flex pair
  int fj = cx->flist[kj], ea_j = m->flex_edgeadr[fj], off_j = cx->fxadr[kj];
  for (int e1=u->lo; e1 < u->hi; e1++) {
    int a1 = off_j + m->flex_edge[2 * (ea_j + e1)],
        b1 = off_j + m->flex_edge[2 * (ea_j + e1) + 1];
    // per-pair band plus the travel of both sides; the query adds the tree's staleness
    mjtNum the = 3.0 * mju_min(ghat, mju_min(radii[a1], rk)) + (dl[a1] > dl[b1] ? dl[a1] : dl[b1]) +
                 fsweep_k;
    mjtNum qc[3], qh[3];
    for (int kk=0; kk < 3; kk++) {
      qc[kk] = 0.5 * (x[3 * a1 + kk] + x[3 * b1 + kk]);
      qh[kk] = 0.5 * mju_abs(x[3 * a1 + kk] - x[3 * b1 + kk]) + the + radii[a1] + fstale_k;
    }
    int n = bvhBox(m, d, fk, qc, qh, stk, outel, ne_k);
    int qid = ++w->qid;
    for (int i=0; i < n; i++) {
      int e = outel[i];
      for (int j=0; j < 3; j++) {
        int e2 = eme_k[3 * e + j];
        if (self && e2 <= e1) continue;  // canonical within a flex
        if (stampG[e2] == qid) continue;
        stampG[e2] = qid;
        int a2 = off_k + m->flex_edge[2 * (ea_k + e2)],
            b2 = off_k + m->flex_edge[2 * (ea_k + e2) + 1];
        if (a1 == a2 || a1 == b2 || b1 == a2 || b1 == b2)
          continue;  // shared vertex -> adjacent, skip
        if (sameBody(cx->ptbody, a1, a2) || sameBody(cx->ptbody, a1, b2) ||
            sameBody(cx->ptbody, b1, a2) || sameBody(cx->ptbody, b1, b2)) {
          continue;  // endpoints ride on one body
        }
        mjcFlexPair con = {mjcFLEX_EDGE_EDGE, {a1, b1, a2, b2}, -1};
        addCand(con, m, d, x, gv, ge, radii, the, dfrom, dto, ghat, sink);
      }
    }
  }
}


// append the units of one segment [0, n) in chunks of at most chunk (count only if out is NULL)
static int candSegment(candUnit* out, int nu, int kind, int k, int kj, int n, int chunk) {
  for (int lo=0; lo < n; lo += chunk) {
    if (out) {
      candUnit u = {kind, k, kj, lo, n - lo < chunk ? n : lo + chunk};
      out[nu] = u;
    }
    nu++;
  }
  return nu;
}


// the BVH-query units of the broad phase in the serial order (count only if out is NULL); every
// segment is split into chunks of about n / nsplit (at least minchunk)
static int candUnits(const candCtx* cx, candUnit* out, int nsplit, int minchunk) {
  const mjModel* m = cx->m;
  int nu = 0;
#define CAND_CHUNK(n) ((n) / nsplit > minchunk ? ((n) + nsplit - 1) / nsplit : minchunk)
  for (int k=0; k < cx->nfd; k++) {
    int fk = cx->flist[k];
    if (m->flex_bvhadr[fk] < 0 || m->flex_elemnum[fk] == 0) continue;
    nu = candSegment(out, nu, candCORNER, k, 0, cx->ngv, CAND_CHUNK(cx->ngv));
    nu = candSegment(out, nu, candGEDGE, k, 0, cx->nge, CAND_CHUNK(cx->nge));
    nu = candSegment(out, nu, candVERT, k, 0, cx->nfv, CAND_CHUNK(cx->nfv));
    for (int kj=0; kj <= k; kj++) {
      if (!cx->fok[k * cx->nfd + kj]) continue;
      int en = m->flex_edgenum[cx->flist[kj]];
      nu = candSegment(out, nu, candEDGE, k, kj, en, CAND_CHUNK(en));
    }
  }
#undef CAND_CHUNK
  return nu;
}


// threaded broad phase: unit task_id into its own block list, with thread thread_id's scratch
typedef struct {
  const candCtx* cx;
  const candUnit* units;
  candScratch* scratch;  // per thread
  candSink* sinks;       // per unit, block mode
} candTaskArg;

static void candTask(const mjModel* m, mjData* unused, void* arg, int thread_id, int task_id) {
  (void)m;
  (void)unused;
  candTaskArg* a = (candTaskArg*)arg;
  candRun(a->cx, a->units + task_id, a->scratch + thread_id, a->sinks + task_id);
}


// build the candidate-contact list: every pair whose gap can enter the detection band along the
// sweep dfrom -> dto. The reach of each query is the base threshold plus the travel of the points
// involved, taken per point (|dto - dfrom|) and per queried flex (the largest travel of its points,
// plus how far its points sit from the configuration its BVH was built at, the state's), so the
// bounds are conservative for the sweep however far a proposal flings a vertex, without inflating
// every query by the fastest vertex anywhere. Flex-flex and geom-feature-vs-flex pairs are found by
// querying the flex element BVH (bvhBox). The native collision filtering applies: nothing with
// contact (or constraints) disabled, pairs under the contype/conaffinity rule, self-contact under
// the flex's selfcollide, and no flex-flex pair between features with points on one body.
// When a thread pool is bound to d, the work units run in parallel into per-unit block lists on the
// stack and are then pushed onto the arena in the serial order.
int mjc_candidates(const mjModel* m, mjData* d, const mjtNum* x, const mjtNum* gv,
                   const mjtNum* ge, const int* gvgeom, const int* gegeom, int ngv, int nge,
                   const mjtNum* radii, mjtNum thresh, mjtNum threshGeom, const mjtNum* dfrom,
                   const mjtNum* dto, mjtNum ghat, int nfv, int npt, const int* fidx,
                   const int* pjnv, const int* flist, const int* fxadr, int nfd,
                   const int* pt2flex, mjcFlexPair** cand) {
  (void)npt;  // increment A: candidates are flex-only (npt==nfv); rigid bodies have no hard contact
  int nc = 0, full = 0;
  *cand = NULL;
  if (mjDISABLED(mjDSBL_CONSTRAINT) || mjDISABLED(mjDSBL_CONTACT)) return 0;  // as mj_collision
  mj_markStack(d);
  // the sweep: travel per point, and per flex the largest travel of its points (fsweep) and the
  // largest distance of its points from the BVH's configuration (fstale)
  mjtNum* dl = mjSTACKALLOC(d, nfv > 0 ? nfv : 1, mjtNum);
  mjtNum* fsweep = mjSTACKALLOC(d, nfd > 0 ? nfd : 1, mjtNum);
  mjtNum* fstale = mjSTACKALLOC(d, nfd > 0 ? nfd : 1, mjtNum);
  for (int k=0; k < nfd; k++) fsweep[k] = fstale[k] = 0;
  for (int v=0; v < nfv; v++) {
    int k = pt2flex[v];
    int vg = m->flex_vertadr[flist[k]] + v - fxadr[k];
    mjtNum dd[3], ds[3];
    for (int c=0; c < 3; c++) {
      dd[c] = dto[3 * v + c] - dfrom[3 * v + c];
      ds[c] = x[3 * v + c] - d->flexvert_xpos[3 * vg + c];
    }
    dl[v] = mju_sqrt(mju_dot3(dd, dd));
    mjtNum stale = mju_sqrt(mju_dot3(ds, ds));
    if (dl[v] > fsweep[k]) fsweep[k] = dl[v];
    if (stale > fstale[k]) fstale[k] = stale;
  }
  candCtx cx = {m, d, x, gv, ge, gvgeom, gegeom, ngv, nge, radii, thresh, threshGeom, dfrom, dto,
                ghat, nfv, fidx, pjnv, flist, fxadr, nfd, pt2flex, dl, fsweep, fstale, NULL, NULL,
                {0, 0, 0}, {0, 0, 0}, 0, 0};
  for (int v=0; v < nfv; v++) {
    if (fidx[v] < 0 && !(pjnv && pjnv[v] > 0)) continue;
    mjtNum marg = thresh + dl[v] + radii[v];
    if (!cx.npg || marg > cx.pgmarg) cx.pgmarg = marg;
    for (int k=0; k < 3; k++) {
      if (!cx.npg || x[3 * v + k] < cx.pglo[k]) cx.pglo[k] = x[3 * v + k];
      if (!cx.npg || x[3 * v + k] > cx.pghi[k]) cx.pghi[k] = x[3 * v + k];
    }
    cx.npg++;
  }
  // pad clear of the per-point test's round-off
  cx.pgmarg += 1e-9 * mju_abs(cx.pgmarg) + 1e-9;
  candSink arena = {d, cand, &nc, 0, NULL, -1, -1};
  int nthread = mju_numThread(d);

  // free point (flex vert) vs STATIC geom; serially before the scratch below is allocated, as the
  // arena it fills ends at the stack
  if (nthread < 2) {
    candUnit u = {candGEOMVERT, 0, 0, 0, m->ngeom};
    candRun(&cx, &u, NULL, &arena);
  }

  // (Rigid sphere-sphere and sphere-vs-flex HARD contacts removed in increment A: rigid bodies live
  // in their own solver block and carry no continuous contact yet. Soft flex-rigid contact is a
  // separate next increment.)

  // ---- BVH-based candidates over ALL dim-2 flexes: geom-feature and flex-vs-flex VT/EE.
  // Each query is against one flex's element BVH (bvhBox); triangle/edge vertices are mapped
  // from the queried flex's local indices to the combined free-point space (fxadr[k] + local).
  // flex-vs-flex contact is SELF when the querying vertex/edge is in the queried flex (gated by
  // that flex's selfcollide and self-compatible masks) and INTER-FLEX otherwise (gated by the
  // two flexes' masks). Scratch buffers are sized for the largest flex. ----
  int maxbvh = 1, maxel = 1, maxen = 1;
  for (int k=0; k < nfd; k++) {
    int fk = flist[k];
    if (m->flex_bvhnum[fk] > maxbvh) maxbvh = m->flex_bvhnum[fk];
    if (m->flex_elemnum[fk] > maxel) maxel = m->flex_elemnum[fk];
    if (m->flex_edgenum[fk] > maxen) maxen = m->flex_edgenum[fk];
  }
  // flex-pair filter fok[k*nfd+kj]: flex flist[kj] against flex flist[k]'s elements
  int* fok = mjSTACKALLOC(d, nfd > 0 ? nfd * nfd : 1, int);
  for (int k=0; k < nfd; k++) {
    for (int kj=0; kj < nfd; kj++) {
      int fk = flist[k], fj = flist[kj];
      int ok = !maskFiltered(m->flex_contype[fk], m->flex_conaffinity[fk], m->flex_contype[fj],
                             m->flex_conaffinity[fj]);
      if (kj == k) ok = ok && (m->flex_selfcollide[fk] != mjFLEXSELF_NONE);
      fok[k * nfd + kj] = ok;
    }
  }

  // body of each point (-1 for interpolated flexes, whose vertices have no body). Two features
  // whose points share a body move rigidly together, e.g. a vertex pinned onto another flex's
  // vertex body or two vertices pinned to one parent: like geoms on one body, they do not collide
  int npt_all = 0;
  for (int k=0; k < nfd; k++) {
    int top = fxadr[k] + m->flex_vertnum[flist[k]];
    if (top > npt_all) npt_all = top;
  }
  int* ptbody = mjSTACKALLOC(d, npt_all > 0 ? npt_all : 1, int);
  for (int k=0; k < nfd; k++) {
    int fk = flist[k];
    for (int lv=0; lv < m->flex_vertnum[fk]; lv++) {
      ptbody[fxadr[k] + lv] = m->flex_vertbodyid[m->flex_vertadr[fk] + lv];
    }
  }
  cx.fok = fok;
  cx.ptbody = ptbody;

  if (nthread < 2) {
    // serial: each segment as a single unit, onto the arena
    int* stk = mjSTACKALLOC(d, maxbvh, int);
    int* outel = mjSTACKALLOC(d, maxel, int);
    int* stampG = mjSTACKALLOC(d, maxen, int);
    for (int e=0; e < maxen; e++) stampG[e] = -1;
    candScratch w = {stk, outel, stampG, 0};
    int nu = candUnits(&cx, NULL, 1, INT_MAX);
    candUnit* units = mjSTACKALLOC(d, nu > 0 ? nu : 1, candUnit);
    candUnits(&cx, units, 1, INT_MAX);
    for (int i=0; i < nu && !arena.full; i++) candRun(&cx, units + i, &w, &arena);
  } else {
    // threaded: the point-vs-geom units (one per geom chunk), then the BVH units, each into its
    // own block list on the stack; after dispatch, blocks are copied onto the arena in serial order
    int gchunk = m->ngeom / (4*nthread) > 1 ? (m->ngeom + 4*nthread - 1) / (4*nthread) : 1;
    int ng = candSegment(NULL, 0, candGEOMVERT, 0, 0, m->ngeom, gchunk);
    int nb = candUnits(&cx, NULL, 4*nthread, 16);
    int nu = ng + nb;
    candUnit* units = mjSTACKALLOC(d, nu > 0 ? nu : 1, candUnit);
    candSegment(units, 0, candGEOMVERT, 0, 0, m->ngeom, gchunk);
    candUnits(&cx, units + ng, 4*nthread, 16);
    candSink* sinks = mjSTACKALLOC(d, nu > 0 ? nu : 1, candSink);
    candScratch* scratch = mjSTACKALLOC(d, nthread, candScratch);
    size_t nscr = (size_t)nthread * (size_t)(maxbvh + maxel + maxen);
    int* sbuf = mjSTACKALLOC(d, nscr > 0 ? nscr : 1, int);
    for (int t=0; t < nthread; t++) {
      int* base = sbuf + (size_t)t * (size_t)(maxbvh + maxel + maxen);
      candScratch w = {base, base + maxbvh, base + maxbvh + maxel, 0};
      for (int e=0; e < maxen; e++) w.stampG[e] = -1;
      scratch[t] = w;
    }
    // reserve up to half of the remaining stack space for candidate blocks so the other half stays
    // free for the contiguous arena merge (and the inner mju_dispatch stack frame)
    size_t avail = mj_stackBytesAvailable(d);
    size_t frame = 2 * mj_stackFrameBytes() + mj_stackBytes(sizeof(candBlock), _Alignof(candBlock));
    size_t maxblocks = avail > frame ? (avail - frame) / (2 * sizeof(candBlock)) : 0;
    candBlockPool pool = {NULL, 0, maxblocks};
    if (maxblocks > 0) {
      pool.blocks = mjSTACKALLOC(d, maxblocks, candBlock);
    } else if (nu > 0) {
      full = 1;
    }
    for (int i=0; i < nu; i++) {
      candSink s = {d, NULL, NULL, 0, &pool, -1, -1};
      sinks[i] = s;
    }
    candTaskArg targ = {&cx, units, scratch, sinks};
    if (!full) {
      mju_dispatch(m, d, candTask, &targ, nu);
    }
    for (int i=0; i < nu && !arena.full; i++) {
      for (int b=sinks[i].head; b >= 0 && !arena.full; b = pool.blocks[b].next) {
        const candBlock* blk = &pool.blocks[b];
        for (int j=0; j < blk->n && !arena.full; j++) {
          sinkPush(&arena, &blk->pairs[j]);
        }
      }
    }
    for (int i=0; i < nu; i++) {
      if (sinks[i].full) full = 1;
    }
  }
  if (arena.full) full = 1;
  mj_freeStack(d);
  if (full) {
    mj_warning(d, mjWARN_CONTACTFULL, nc + 1);
    mju_warning("IPC: arena full after %d candidate pairs; the rest were dropped, so geometry "
                "there can pass through. Increase the model's memory attribute. Time = %.4f",
                nc, d->time);
  }
  return nc;
}


// compute the step displacement (dx, dw) over the time interval [0, t] and return the
// upper bound on the distance traveled by any point on the geom's surface
mjtNum mjc_geomStepMotion(const mjModel* m, const mjData* d, int g, mjtNum t,
                          mjtNum dx[3], mjtNum dw[3]) {
  mju_zero3(dx);
  mju_zero3(dw);
  if (g < 0 || m->body_weldid[m->geom_bodyid[g]] == 0) {
    return 0;
  }
  int bodyid = m->geom_bodyid[g];

  // compute 6D spatial velocity cvel at subtree_com from cdof * qvel
  mjtNum cvel[6] = {0, 0, 0, 0, 0, 0};
  for (int b = m->body_weldid[bodyid]; b > 0; b = m->body_parentid[b]) {
    int bda = m->body_dofadr[b];
    int dofnum = m->body_dofnum[b];
    for (int j = 0; j < dofnum; j++) {
      mjtNum qv = d->qvel[bda + j];
      if (qv != 0) {
        const mjtNum* cdof = d->cdof + 6 * (bda + j);
        for (int k = 0; k < 6; k++) {
          cvel[k] += cdof[k] * qv;
        }
      }
    }
  }

  // transform 6D spatial velocity from subtree_com to geom_xpos
  mjtNum gvel[6];
  mju_transformSpatial(gvel, cvel, 0, d->geom_xpos + 3 * g,
                       d->subtree_com + 3 * m->body_rootid[bodyid], NULL);

  if (m->geom_type[g] != mjGEOM_SPHERE) {
    mji_scl3(dw, gvel, t);
  }
  mji_scl3(dx, gvel + 3, t);

  // for any point q on the geom surface, its velocity is v + w x (q - xpos)
  // where v is the linear velocity and w is the angular velocity
  // the upper bound on the distance traveled for any point on the surface is
  // lin_dist + ang_dist * max_q |q - xpos| via the triangle inequality
  return mju_norm3(dx) + mju_norm3(dw) * m->geom_rbound[g];
}


// advance rigid geom pose (obj->pos, obj->mat) by step * (dx, dw)
static inline void advanceGeomPose(mjCCDObj* obj, const mjtNum dx[3], const mjtNum dw[3],
                                   mjtNum step) {
  mji_addToScl3(obj->pos, dx, step);

  // guard against near-zero rotation angles
  mjtNum wnorm = mju_norm3(dw);
  if (wnorm * step < mjMINVAL) {
    return;
  }

  mjtNum axis[3], dq[4], dmat[9], mat[9];
  mju_copy9(mat, obj->mat);
  mji_scl3(axis, dw, 1.0 / wnorm);
  mju_axisAngle2Quat(dq, axis, wnorm * step);
  mju_quat2Mat(dmat, dq);
  mju_mulMatMat3(obj->mat, dmat, mat);
}


// evaluate distance and witnesses (in/out x1, x2 seeded by caller; normal from g1 to g2)
// at current obj1/obj2 poses
static inline mjtNum evalGeomPairDist(const mjModel* m, mjCCDObj* obj1, mjCCDObj* obj2,
                                      mjtNum distmax, mjtNum x1[3], mjtNum x2[3],
                                      mjtNum normal[3]) {
  // radii of spheres and capsules (shrunk to point and line segment for GJK)
  int type1 = obj1->geom_type, type2 = obj2->geom_type;
  mjtNum r1 = (type1 == mjGEOM_SPHERE || type1 == mjGEOM_CAPSULE) ? obj1->size[0] : 0;
  mjtNum r2 = (type2 == mjGEOM_SPHERE || type2 == mjGEOM_CAPSULE) ? obj2->size[0] : 0;

  // distance returned by GJK and cutoff distance for GJK
  mjtNum dist = 0, cutoff = distmax + r1 + r2;

  // planes not supported by GJK
  int has_plane = (type1 == mjGEOM_PLANE) - (type2 == mjGEOM_PLANE);
  if (has_plane) {
    mjCCDObj* plane = has_plane > 0 ? obj1 : obj2;
    mjCCDObj* other = has_plane > 0 ? obj2 : obj1;
    mjtNum* x = has_plane > 0 ? x2 : x1;
    mjtNum n[3] = {plane->mat[2], plane->mat[5], plane->mat[8]};

    mjtNum s[3], dir[3] = {-n[0], -n[1], -n[2]};
    other->support(s, other, dir);
    other->nsupp = 1;
    other->suppind[0] = other->vertindex;
    mji_copy3(other->suppvert, s);

    // account for sphere or capsule radius
    mjtNum r = has_plane > 0 ? r2 : r1;
    mji_addScl3(x, s, n, -r);

    dist = mju_dot3(n, x) - mju_dot3(n, plane->pos);
    mji_addScl3(has_plane > 0 ? x1 : x2, x, n, -dist);
    mji_scl3(normal, n, has_plane);
    return dist;
  }

  normal[0] = 0, normal[1] = 0, normal[2] = 1;

  dist = mjc_gjk(obj1, obj2, x1, x2, cutoff, m->opt.ccd_tolerance, m->opt.ccd_iterations);
  if (dist <= cutoff) {
    mjtNum diff[3] = {x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]};
    if (mju_normalize3(diff) > 0) {
      mju_copy3(normal, diff);
    }
    mji_addToScl3(x1, normal, r1);
    mji_addToScl3(x2, normal, -r2);
  }
  return dist - (r1 + r2);
}


// compute effective contact distance so that mj_referenceConstraint arrests (1 - t) of the
// normal velocity over this step and lands the geom at margin at t = 1
// TODO(kylebayes): consider reusing the solver directly
static inline mjtNum contactDist(const mjModel* m, int g1, int g2, mjtNum margin, mjtNum t,
                                 mjtNum closing_dist) {
  // compute effective solref and solimp for g1 and g2
  mjtNum solref[mjNREF], solimp[mjNIMP], friction[5], adhesion;
  int condim, ipair = -1;
  for (int i = 0; i < m->npair; i++) {
    if ((m->pair_geom1[i] == g1 && m->pair_geom2[i] == g2) ||
        (m->pair_geom1[i] == g2 && m->pair_geom2[i] == g1)) {
      ipair = i;
      break;
    }
  }
  if (ipair >= 0) {
    mju_copy(solref, m->pair_solref + mjNREF * ipair, mjNREF);
    mju_copy(solimp, m->pair_solimp + mjNIMP * ipair, mjNIMP);
  } else {
    mj_contactParam(m, &condim, solref, solimp, friction, &adhesion, g1, g2, -1, -1);
  }
  mj_assignRef(m, solref, solref);
  mj_assignImp(m, solimp, solimp);
  if ((solref[0] > 0) ^ (solref[1] > 0)) {
    mj_defaultSolRefImp(solref, NULL);
  }

  // compute dimensionless step stiffness (k0 = k * dt^2) and damping (b0 = b * dt) from solref,
  // either in standard (timeconst, dampratio) mode or direct (-stiffness, -damping) mode
  mjtNum dt = m->opt.timestep;
  mjtNum imp = mju_min(mjMAXIMP, mju_max(mjMINIMP, solimp[1]));
  mjtNum delta = mju_max(1e-4, solimp[2]);
  if (dt > 0) {
    mjtNum k0 = 0, b0 = 0;
    if (solref[0] > 0) {
      mjtNum tau_ref = (!mjDISABLED(mjDSBL_REFSAFE) && !mj_isMetric(m))
                       ? mju_max(2.0 * dt, solref[0])
                       : mju_max(mjMINVAL, solref[0]);
      mjtNum d_ratio = mju_max(mjMINVAL, solref[1]);
      k0 = (dt * dt) / mju_max(mjMINVAL, tau_ref * tau_ref * d_ratio * d_ratio);
      b0 = 2.0 * dt / tau_ref;
    } else {
      k0 = -solref[0] * dt * dt;
      b0 = -solref[1] * dt;
    }

    // solve for virtual penetration delta so that mj_referenceConstraint's spring-damper response
    // absorbs the remaining (1 - t) fraction of closing_dist after damping is accounted for
    if (mj_isMetric(m)) {
      if (!mjDISABLED(mjDSBL_REFSAFE) && solref[0] > 0 && k0 > imp) {
        mjtNum excess = k0 / imp;
        mjtNum fmax = mjMAXIMP * (1.0 - imp) / mju_max(mjMINVAL, imp * (1.0 - mjMAXIMP));
        k0 = imp;
        b0 = mju_min(b0 / mju_sqrt(excess), imp * mju_max(0.0, fmax - 2.0));
      }
      if (k0 > mjMINVAL) {
        mjtNum rem = 1.0 - t - t * (b0 + k0);
        delta = (rem / k0) * closing_dist;
      }
    } else if (k0 > mjMINVAL) {
      mjtNum rem = 1.0 - t - b0;
      delta = (rem / k0) * closing_dist;
    }
  }
  return margin - delta;
}


// map witness point x_t from advanced geom pose (obj->pos, obj->mat) back to t = 0 geom pose
static inline void pullbackGeomPoint(mjtNum x_0[3], const mjtNum x_t[3],
                                     const mjCCDObj* obj, const mjData* d) {
  int g = obj->geom;
  mjtNum diff[3], local[3];
  mji_sub3(diff, x_t, obj->pos);
  mju_mulMatTVec3(local, obj->mat, diff);
  mji_mulMatVec3(x_0, d->geom_xmat + 9 * g, local);
  mju_addTo3(x_0, d->geom_xpos + 3 * g);
}


// return 1 if body pair (b1, b2) is excluded via <contact><exclude>
static inline int isBodyPairExcluded(const mjModel* m, int b1, int b2) {
  // TODO(kylebayes): consider binary search over m->exclude_signature
  unsigned int sig = ((unsigned int)mjMIN(b1, b2) << 16) + mjMAX(b1, b2);
  for (int i = 0; i < m->nexclude; i++) {
    if ((unsigned int)m->exclude_signature[i] == sig) {
      return 1;
    }
  }
  return 0;
}


// return <contact><pair> index for geoms (g1, g2), or -1 if none
static inline int findExplicitGeomPair(const mjModel* m, int g1, int g2) {
  // TODO(kylebayes): consider binary search over m->pair_signature
  for (int i = 0; i < m->npair; i++) {
    if ((m->pair_geom1[i] == g1 && m->pair_geom2[i] == g2) ||
        (m->pair_geom1[i] == g2 && m->pair_geom2[i] == g1)) {
      return i;
    }
  }
  return -1;
}


// return 1 if geoms g1 and g2 are within margin at t = 0
static inline int isGeomPairWithinMargin(const mjModel* m, const mjData* d,
                                         int g1, int g2, mjtNum margin) {
  int type1 = m->geom_type[g1], type2 = m->geom_type[g2];
  if (type1 == mjGEOM_PLANE && type2 == mjGEOM_PLANE) {
    return 0;
  }
  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, 0);
  mjc_initCCDObj(&obj2, m, d, g2, 0);
  if (type1 == mjGEOM_SPHERE) obj1.support = mjc_pointSupport;
  if (type1 == mjGEOM_CAPSULE) obj1.support = mjc_lineSupport;
  if (type2 == mjGEOM_SPHERE) obj2.support = mjc_pointSupport;
  if (type2 == mjGEOM_CAPSULE) obj2.support = mjc_lineSupport;

  mjtNum tol = margin + m->opt.ccd_tolerance;
  mjtNum x1[3], x2[3], normal[3];
  mji_copy3(x1, obj1.pos);
  mji_copy3(x2, obj2.pos);
  return evalGeomPairDist(m, &obj1, &obj2, tol, x1, x2, normal) <= tol;
}


// return 1 if segment [a, b] swept by geom g1 (paired with g2) is occluded by another geom
static inline int isRayOccluded(const mjModel* m, const mjData* d, int g1, int g2,
                                const mjtNum a[3], const mjtNum b[3]) {
  mjtNum ab[3];
  mji_sub3(ab, b, a);
  mjtNum len2 = mju_dot3(ab, ab);
  if (len2 < mjMINVAL * mjMINVAL) {
    return 0;
  }

  // shrink segment slightly to avoid self-hits on geoms touching a or b from outside
  const mjtNum eps = 1e-5;
  mjtNum pnt[3], vec[3];
  mji_addScl3(pnt, a, ab, eps);
  mji_scl3(vec, ab, 1.0 - 2.0 * eps);

  mjtNum mid[3];
  mju_addScl3(mid, a, ab, 0.5);
  mjtNum half_len = 0.5 * mju_sqrt(len2);

  int b1 = m->geom_bodyid[g1], b2 = m->geom_bodyid[g2];
  int weld1 = m->body_weldid[b1];
  int weld2 = m->body_weldid[b2];
  int p1 = m->body_weldid[m->body_parentid[weld1]];
  int fltparent = !mjDISABLED(mjDSBL_FILTERPARENT);
  int ct1 = m->geom_contype[g1], ca1 = m->geom_conaffinity[g1];

  // TODO(kylebayes): consider doing this as a post-processing step on the
  // speculative contacts in engine_collision_driver (which also respects
  // mjcb_contactfilter on the main thread) instead of iterating over all geoms here
  for (int g3 = 0; g3 < m->ngeom; g3++) {
    int type = m->geom_type[g3];
    if (g3 == g1 || g3 == g2 || !mjc_GeomSupported(type)) {
      continue;
    }

    // skip g3 if its bounding sphere cannot overlap the segment's sphere
    if (m->geom_rbound[g3] > 0) {
      mjtNum diff[3];
      mji_sub3(diff, d->geom_xpos + 3 * g3, mid);
      mjtNum max_r = m->geom_rbound[g3] + half_len;
      if (mju_dot3(diff, diff) > max_r * max_r) {
        continue;
      }
    }

    // g3 must not share a weld with g1 (or a moving weld with g2) and must collide with g1
    int b3 = m->geom_bodyid[g3];
    int weld3 = m->body_weldid[b3];
    if (weld3 == weld1 || (weld3 == weld2 && weld2 != 0)) {
      continue;
    }
    int p3 = m->body_weldid[m->body_parentid[weld3]];
    int ct3 = m->geom_contype[g3], ca3 = m->geom_conaffinity[g3];
    int ipair = findExplicitGeomPair(m, g1, g3);
    if (ipair < 0) {
      if (!((ct1 & ca3) || (ct3 & ca1)) || isBodyPairExcluded(m, b1, b3) ||
          (fltparent && weld1 != 0 && weld3 != 0 && (weld1 == p3 || weld3 == p1))) {
        continue;
      }
    }

    int hit = 0;
    if (type == mjGEOM_PLANE) {
      const mjtNum* gpos = d->geom_xpos + 3 * g3;
      const mjtNum* gmat = d->geom_xmat + 9 * g3;
      mjtNum n[3] = {gmat[2], gmat[5], gmat[8]};
      mjtNum diff[3];
      mji_sub3(diff, a, gpos);
      mjtNum da = mju_dot3(diff, n);
      mjtNum db = da + mju_dot3(ab, n);
      hit = (da * db <= 0 && (mju_abs(da) > eps || mju_abs(db) > eps));
    } else if (type == mjGEOM_MESH) {
      mjtNum x = mj_rayMesh(m, d, g3, pnt, vec, NULL);
      hit = (x >= 0 && x <= 1);
    } else {
      mjtNum x = mju_rayGeom(d->geom_xpos + 3 * g3, d->geom_xmat + 9 * g3,
                             m->geom_size + 3 * g3, pnt, vec, type, NULL);
      hit = (x >= 0 && x <= 1);
    }

    if (hit) {
      mjtNum margin3 = mj_assignMargin(
          m, ipair >= 0 ? m->pair_margin[ipair] : m->geom_margin[g1] + m->geom_margin[g3]);
      if (!isGeomPairWithinMargin(m, d, g1, g3, margin3)) {
        return 1;
      }
    }
  }
  return 0;
}


// conservative advancement between two geoms g1 and g2 over the timestep;
// write up to 4 contacts into con and return number of contacts
int mjc_advanceGeomPair(const mjModel* m, const mjData* d, int g1, int g2,
                        mjtNum margin, mjPreContact* con, void* buffer) {
  int type1 = g1 >= 0 ? m->geom_type[g1] : -1, type2 = g2 >= 0 ? m->geom_type[g2] : -1;
  if (!mjc_GeomSupported(type1) || !mjc_GeomSupported(type2) ||
      (type1 == mjGEOM_PLANE && type2 == mjGEOM_PLANE)) {
    return 0;
  }

  mjtNum dx1[3], dw1[3], dx2[3], dw2[3];
  mjtNum s1 = mjc_geomStepMotion(m, d, g1, m->opt.timestep, dx1, dw1);
  mjtNum s2 = mjc_geomStepMotion(m, d, g2, m->opt.timestep, dx2, dw2);
  mjtNum sweep = s1 + s2;
  if (sweep < mjMINVAL) {
    return 0;
  }

  mjtNum dx[3];
  mji_sub3(dx, dx1, dx2);
  mjtNum rot_sweep = (s1 - mju_norm3(dx1)) + (s2 - mju_norm3(dx2));

  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, 0);
  mjc_initCCDObj(&obj2, m, d, g2, 0);
  if (type1 == mjGEOM_SPHERE) obj1.support = mjc_pointSupport;
  if (type1 == mjGEOM_CAPSULE) obj1.support = mjc_lineSupport;
  if (type2 == mjGEOM_SPHERE) obj2.support = mjc_pointSupport;
  if (type2 == mjGEOM_CAPSULE) obj2.support = mjc_lineSupport;

  mjtNum r1 = (type1 == mjGEOM_SPHERE || type1 == mjGEOM_CAPSULE) ? obj1.size[0] : 0;
  mjtNum r2 = (type2 == mjGEOM_SPHERE || type2 == mjGEOM_CAPSULE) ? obj2.size[0] : 0;
  mjtNum distmax = sweep + margin;

  mjtNum x1[3], x2[3], normal[3];
  mji_copy3(x1, obj1.pos);
  mji_copy3(x2, obj2.pos);
  mjtNum dist0 = evalGeomPairDist(m, &obj1, &obj2, distmax, x1, x2, normal);

  // already in contact or too far apart (no extra collision needed)
  if (dist0 <= margin || dist0 - margin > sweep) {
    return 0;
  }

  // target distance to shrink to (2% of initial distance to always get witness
  // points from GJK)
  mjtNum target = 0.02 * (dist0 - margin) + margin;

  // relative tolerance for early exit
  mjtNum rel_tol = mju_max(m->opt.ccd_tolerance, 0.5 * (target - margin));
  mjtNum t = 0, dist = dist0;

  // directed conservative advancement (capped at 16 steps)
  for (int it = 0; it < 16; it++) {
    mjtNum clearance = dist - target;
    if (clearance <= rel_tol) {
      break;
    }

    mjtNum dir_sweep = mju_dot3(normal, dx) + rot_sweep;
    if (dir_sweep <= mjMINVAL) {
      return 0;
    }

    mjtNum step = clearance / dir_sweep;
    t += step;
    if (t >= 1.0) {
      return 0;
    }

    // transport un-inflated core witness points into advanced geom frames
    mjtNum core1[3], core2[3], diff1[3], diff2[3], local1[3], local2[3];
    mji_addScl3(core1, x1, normal, -r1);
    mji_addScl3(core2, x2, normal, r2);
    mji_sub3(diff1, core1, obj1.pos);
    mji_sub3(diff2, core2, obj2.pos);
    mju_mulMatTVec3(local1, obj1.mat, diff1);
    mju_mulMatTVec3(local2, obj2.mat, diff2);

    advanceGeomPose(&obj1, dx1, dw1, step);
    advanceGeomPose(&obj2, dx2, dw2, step);

    mjtNum tmp1[3], tmp2[3];
    mji_mulMatVec3(tmp1, obj1.mat, local1);
    mju_add3(x1, obj1.pos, tmp1);
    mji_mulMatVec3(tmp2, obj2.mat, local2);
    mju_add3(x2, obj2.pos, tmp2);

    dist = evalGeomPairDist(m, &obj1, &obj2, distmax, x1, x2, normal);

    // remaining motion cannot reach margin (also guarantees dist <= distmax, valid normal)
    if (dist - margin > (1.0 - t) * sweep) {
      return 0;
    }
  }

  // full-step displacements of witness points x1, x2: dp = dx + dw x (x - pos)
  mjtNum rel1[3], rel2[3], rot1[3], rot2[3], dp1[3], dp2[3], drel[3];
  mji_sub3(rel1, x1, obj1.pos);
  mji_sub3(rel2, x2, obj2.pos);
  mji_cross(rot1, dw1, rel1);
  mji_cross(rot2, dw2, rel2);
  mju_add3(dp1, dx1, rot1);
  mju_add3(dp2, dx2, rot2);

  // normal closing distance over the step (max of instantaneous and average closing rates)
  mji_sub3(drel, dp1, dp2);
  mjtNum vn = mju_dot3(normal, drel);
  mjtNum closing_dist = mju_max(vn, (dist0 - dist) / mju_max(t, mjMINVAL));

  // linear extrapolation fallback if 16-step cap was reached (non-conservative under rotation)
  if (dist - target > rel_tol) {
    if (vn <= mjMINVAL || dist - margin > (1.0 - t) * vn) {
      return 0;
    }
    t += (dist - target) / vn;
  }
  if (closing_dist <= 0) {
    return 0;
  }

  int ncon = 0;
  if (con) {
    int nx = 1;
    mjtNum x1_all[12], x2_all[12], dist_all[4];
    mji_copy3(x1_all, x1);
    mji_copy3(x2_all, x2);
    dist_all[0] = dist;

    // plane geoms always use multicontact
    int has_plane = (type1 == mjGEOM_PLANE || type2 == mjGEOM_PLANE);
    if (buffer && (has_plane || !mjDISABLED(mjDSBL_MULTICCD))) {
      mjtNum pad = 2.0 * mju_max(0.0, dist) + (1.0 - t) * closing_dist +
                   m->opt.ccd_tolerance;
      obj1.margin = pad;
      obj2.margin = pad;
      mji_addToScl3(x1_all, normal, 0.5 * pad);
      mji_addToScl3(x2_all, normal, -0.5 * pad);
      nx = mjc_multicontact(&obj1, &obj2, x1_all, x2_all, dist_all, 4,
                            m->nmeshdegmax, m->npolygonmax, buffer);
      for (int i = 0; i < nx; i++) {
        mji_addToScl3(x1_all + 3*i, normal, -0.5 * pad);
        mji_addToScl3(x2_all + 3*i, normal, 0.5 * pad);
      }
    }

    mjtNum cdist = contactDist(m, g1, g2, margin, t, closing_dist);
    // TODO(kylebayes): slower body gets a lever-arm error proportional to tangential travel;
    // the real fix is per-body contact points in the Jacobian
    mjtNum w1 = s1 / sweep, w2 = s2 / sweep;
    int g_fast = s1 >= s2 ? g1 : g2, g_slow = s1 >= s2 ? g2 : g1;
    for (int i = 0; i < nx; i++) {
      const mjtNum* x1_t = x1_all + 3*i;
      const mjtNum* x2_t = x2_all + 3*i;
      mjtNum x1_0[3], x2_0[3];
      pullbackGeomPoint(x1_0, x1_t, &obj1, d);
      pullbackGeomPoint(x2_0, x2_t, &obj2, d);
      if (isRayOccluded(m, d, g1, g2, x1_0, x1_t) ||
          isRayOccluded(m, d, g2, g1, x2_0, x2_t) ||
          isRayOccluded(m, d, g_fast, g_slow, x1_t, x2_t)) {
        continue;
      }
      con[ncon].pos[0] = w1 * x1_0[0] + w2 * x2_0[0];
      con[ncon].pos[1] = w1 * x1_0[1] + w2 * x2_0[1];
      con[ncon].pos[2] = w1 * x1_0[2] + w2 * x2_0[2];
      con[ncon].dist = cdist;
      mju_copy3(con[ncon].normal, normal);
      mju_zero3(con[ncon].tangent);
      ncon++;
    }
  } else {
    int g_fast = s1 >= s2 ? g1 : g2, g_slow = s1 >= s2 ? g2 : g1;
    mjtNum x1_0[3], x2_0[3];
    pullbackGeomPoint(x1_0, x1, &obj1, d);
    pullbackGeomPoint(x2_0, x2, &obj2, d);
    if (!isRayOccluded(m, d, g1, g2, x1_0, x1) &&
        !isRayOccluded(m, d, g2, g1, x2_0, x2) &&
        !isRayOccluded(m, d, g_fast, g_slow, x1, x2)) {
      ncon = 1;
    }
  }

  return ncon;
}
