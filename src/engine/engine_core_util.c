// Copyright 2025 DeepMind Technologies Limited
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

#include "engine/engine_core_util.h"

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include "engine/engine_collision_primitive.h"
#include "engine/engine_inline.h"
#include "engine/engine_memory.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"



// determine type of constraint Jacobian
int mj_isSparse(const mjModel* m) {
  if (m->opt.jacobian == mjJAC_SPARSE ||
      (m->opt.jacobian == mjJAC_AUTO && m->nv >= 60)) {
    return 1;
  } else {
    return 0;
  }
}


// determine type of friction cone
int mj_isPyramidal(const mjModel* m) {
  if (m->opt.cone == mjCONE_PYRAMIDAL) {
    return 1;
  } else {
    return 0;
  }
}


// determine type of solver
int mj_isDual(const mjModel* m) {
  if (m->opt.solver == mjSOL_PGS || m->opt.noslip_iterations > 0) {
    return 1;
  } else {
    return 0;
  }
}


//-------------------------- sparse chains ---------------------------------------------------------

// merge dof chains for two bodies
int mj_mergeChain(const mjModel* m, int* chain, int b1, int b2, int flg_skipcommon) {
  int da1, da2, NV = 0;

  // skip fixed bodies
  b1 = m->body_weldid[b1];
  b2 = m->body_weldid[b2];

  // neither weld root has dofs: empty chain
  if (m->body_dofnum[b1] == 0 && m->body_dofnum[b2] == 0) {
    return 0;
  }

  // initialize last dof address for each body
  da1 = m->body_dofadr[b1] + m->body_dofnum[b1] - 1;
  da2 = m->body_dofadr[b2] + m->body_dofnum[b2] - 1;

  // merge chains
  while (da1 >= 0 || da2 >= 0) {
    int da = mjMAX(da1, da2);
    if (flg_skipcommon && da1 == da && da2 == da) {
      break;
    }
    chain[NV] = da;
    if (da1 == da) {
      da1 = m->dof_parentid[da1];
    }
    if (da2 == da) {
      da2 = m->dof_parentid[da2];
    }
    NV++;
  }

  // reverse order of chain: make it increasing
  for (int i=0; i < NV/2; i++) {
    int tmp = chain[i];
    chain[i] = chain[NV-i-1];
    chain[NV-i-1] = tmp;
  }

  return NV;
}


// merge dof chains for two different simple bodies
int mj_mergeChainSimple(const mjModel* m, int* chain, int b1, int b2) {
  // swap bodies if wrong order
  if (b1 > b2) {
    int tmp = b1;
    b1 = b2;
    b2 = tmp;
  }

  // init
  int n1 = m->body_dofnum[b1];
  int n2 = m->body_dofnum[b2];

  // both fixed: nothing to do
  if (n1 == 0 && n2 == 0) {
    return 0;
  }

  // copy b1 dofs
  for (int i=0; i < n1; i++) {
    chain[i] = m->body_dofadr[b1] + i;
  }

  // copy b2 dofs
  for (int i=0; i < n2; i++) {
    chain[n1+i] = m->body_dofadr[b2] + i;
  }

  return (n1+n2);
}


// get body chain
int mj_bodyChain(const mjModel* m, int body, int* chain) {
  // simple body
  if (m->body_simple[body]) {
    int dofnum = m->body_dofnum[body];
    for (int i=0; i < dofnum; i++) {
      chain[i] = m->body_dofadr[body] + i;
    }
    return dofnum;
  }

  // general case
  else {
    // skip fixed bodies
    body = m->body_weldid[body];

    // weld root has no dofs: empty chain
    if (m->body_dofnum[body] == 0) {
      return 0;
    }

    // initialize last dof
    int da = m->body_dofadr[body] + m->body_dofnum[body] - 1;
    int NV = 0;

    // construct chain from child to parent
    while (da >= 0) {
      chain[NV++] = da;
      da = m->dof_parentid[da];
    }

    // reverse order of chain: make it increasing
    for (int i=0; i < NV/2; i++) {
      int tmp = chain[i];
      chain[i] = chain[NV-i-1];
      chain[NV-i-1] = tmp;
    }

    return NV;
  }
}


//-------------------------- Jacobians -------------------------------------------------------------

// compute 3/6-by-nv Jacobian of global point attached to given body
void mj_jac(const mjModel* m, const mjData* d,
            mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body) {
  int nv = m->nv;
  mjtNum offset[3];

  // clear jacobians, compute offset if required
  if (jacp) {
    mju_zero(jacp, 3*nv);
    mju_sub3(offset, point, d->subtree_com+3*m->body_rootid[body]);
  }
  if (jacr) {
    mju_zero(jacr, 3*nv);
  }

  // skip fixed bodies
  body = m->body_weldid[body];

  // weld root has no dofs: nothing to do
  if (m->body_dofnum[body] == 0) {
    return;
  }

  // get last dof that affects this (as well as the original) body
  int i = m->body_dofadr[body] + m->body_dofnum[body] - 1;

  // backward pass over dof ancestor chain
  while (i >= 0) {
    mjtNum* cdof = d->cdof+6*i;

    // construct rotation jacobian
    if (jacr) {
      jacr[i+0*nv] = cdof[0];
      jacr[i+1*nv] = cdof[1];
      jacr[i+2*nv] = cdof[2];
    }

    // construct translation jacobian (correct for rotation)
    if (jacp) {
      mjtNum tmp[3];
      mji_cross(tmp, cdof, offset);
      jacp[i+0*nv] = cdof[3] + tmp[0];
      jacp[i+1*nv] = cdof[4] + tmp[1];
      jacp[i+2*nv] = cdof[5] + tmp[2];
    }

    // advance to parent dof
    i = m->dof_parentid[i];
  }
}


// compute body Jacobian
void mj_jacBody(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int body) {
  mj_jac(m, d, jacp, jacr, d->xpos+3*body, body);
}


// compute body-com Jacobian
void mj_jacBodyCom(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int body) {
  mj_jac(m, d, jacp, jacr, d->xipos+3*body, body);
}


// compute subtree-com Jacobian
void mj_jacSubtreeCom(const mjModel* m, mjData* d, mjtNum* jacp, int body) {
  int nv = m->nv;
  mj_markStack(d);
  mjtNum* jacp_b = mjSTACKALLOC(d, 3*nv, mjtNum);

  // clear output
  mju_zero(jacp, 3*nv);

  // forward pass starting from body
  for (int b=body; b < m->nbody; b++) {
    // end of body subtree, break from the loop
    if (b > body && m->body_parentid[b] < body) {
      break;
    }

    // b is in the body subtree, add mass-weighted Jacobian into jacp
    mj_jac(m, d, jacp_b, NULL, d->xipos+3*b, b);
    mju_addToScl(jacp, jacp_b, m->body_mass[b], 3*nv);
  }

  // normalize by subtree mass
  mju_scl(jacp, jacp, 1/m->body_subtreemass[body], 3*nv);

  mj_freeStack(d);
}


// compute geom Jacobian
void mj_jacGeom(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int geom) {
  mj_jac(m, d, jacp, jacr, d->geom_xpos + 3*geom, m->geom_bodyid[geom]);
}


// compute site Jacobian
void mj_jacSite(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int site) {
  mj_jac(m, d, jacp, jacr, d->site_xpos + 3*site, m->site_bodyid[site]);
}


// compute translation Jacobian of point, and rotation Jacobian of axis
void mj_jacPointAxis(const mjModel* m, mjData* d, mjtNum* jacPoint, mjtNum* jacAxis,
                     const mjtNum point[3], const mjtNum axis[3], int body) {
  int nv = m->nv;

  // get full Jacobian of point
  mj_markStack(d);
  mjtNum* jacp = (jacPoint ? jacPoint : mjSTACKALLOC(d, 3*nv, mjtNum));
  mjtNum* jacr = mjSTACKALLOC(d, 3*nv, mjtNum);
  mj_jac(m, d, jacp, jacr, point, body);

  // jacAxis_col = cross(jacr_col, axis)
  if (jacAxis) {
    for (int i=0; i < nv; i++) {
      jacAxis[     i] = jacr[  nv+i]*axis[2] - jacr[2*nv+i]*axis[1];
      jacAxis[  nv+i] = jacr[2*nv+i]*axis[0] - jacr[     i]*axis[2];
      jacAxis[2*nv+i] = jacr[     i]*axis[1] - jacr[  nv+i]*axis[0];
    }
  }

  mj_freeStack(d);
}


// compute 3/6-by-nv sparse Jacobian of global point attached to given body
void mj_jacSparse(const mjModel* m, const mjData* d,
                  mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body,
                  int NV, const int* chain, int flg_skipcommon) {
  // clear jacobians
  if (jacp) {
    mju_zero(jacp, 3*NV);
  }
  if (jacr) {
    mju_zero(jacr, 3*NV);
  }

  // compute point-com offset
  mjtNum offset[3];
  mju_sub3(offset, point, d->subtree_com+3*m->body_rootid[body]);

  // skip fixed bodies
  body = m->body_weldid[body];

  // weld root has no dofs: nothing to do
  if (m->body_dofnum[body] == 0) {
    return;
  }

  // get last dof that affects this (as well as the original) body
  int da = m->body_dofadr[body] + m->body_dofnum[body] - 1;

  // start and the end of the chain (chain is in increasing order)
  int ci = NV-1;

  // backward pass over dof ancestor chain
  while (da >= 0) {
    // find chain index for this dof
    while (ci >= 0 && chain[ci] > da) {
      ci--;
    }

    // dof not in chain: skip if shared dofs are excluded, otherwise SHOULD NOT OCCUR
    if (ci < 0 || chain[ci] != da) {
      if (flg_skipcommon) {
        da = m->dof_parentid[da];
        continue;
      }
      mjERROR("dof index %d not found in chain", da);
    }

    const mjtNum* cdof = d->cdof + 6*da;

    // construct rotation jacobian
    if (jacr) {
      jacr[ci+0*NV] = cdof[0];
      jacr[ci+1*NV] = cdof[1];
      jacr[ci+2*NV] = cdof[2];
    }

    // construct translation jacobian (correct for rotation)
    if (jacp) {
      mjtNum tmp[3];
      mji_cross(tmp, cdof, offset);

      jacp[ci+0*NV] = cdof[3] + tmp[0];
      jacp[ci+1*NV] = cdof[4] + tmp[1];
      jacp[ci+2*NV] = cdof[5] + tmp[2];
    }

    // advance to parent dof
    da = m->dof_parentid[da];
  }
}


// sparse Jacobian difference for simple body contacts
void mj_jacSparseSimple(const mjModel* m, const mjData* d,
                        mjtNum* jacdifp, mjtNum* jacdifr, const mjtNum* point,
                        int body, int flg_second, int NV, int start) {
  // compute point-com offset
  mjtNum offset[3];
  mju_sub3(offset, point, d->subtree_com+3*m->body_rootid[body]);

  // skip fixed body
  if (!m->body_dofnum[body]) {
    return;
  }

  // process dofs
  int ci = start;
  int end = m->body_dofadr[body] + m->body_dofnum[body];
  for (int da=m->body_dofadr[body]; da < end; da++) {
    mjtNum *cdof = d->cdof+6*da;

    // construct rotation jacobian
    if (jacdifr) {
      // plus sign
      if (flg_second) {
        jacdifr[ci+0*NV] = cdof[0];
        jacdifr[ci+1*NV] = cdof[1];
        jacdifr[ci+2*NV] = cdof[2];
      }

      // minus sign
      else {
        jacdifr[ci+0*NV] = -cdof[0];
        jacdifr[ci+1*NV] = -cdof[1];
        jacdifr[ci+2*NV] = -cdof[2];
      }
    }

    // construct translation jacobian (correct for rotation)
    if (jacdifp) {
      mjtNum tmp[3];
      mji_cross(tmp, cdof, offset);

      // plus sign
      if (flg_second) {
        jacdifp[ci+0*NV] = (cdof[3] + tmp[0]);
        jacdifp[ci+1*NV] = (cdof[4] + tmp[1]);
        jacdifp[ci+2*NV] = (cdof[5] + tmp[2]);
      }

      // minus sign
      else {
        jacdifp[ci+0*NV] = -(cdof[3] + tmp[0]);
        jacdifp[ci+1*NV] = -(cdof[4] + tmp[1]);
        jacdifp[ci+2*NV] = -(cdof[5] + tmp[2]);
      }
    }

    // advance jacdif counter
    ci++;
  }
}


// dense or sparse Jacobian difference for two body points: pos2 - pos1, global
int mj_jacDifPair(const mjModel* m, const mjData* d, int* chain,
                  int b1, int b2, const mjtNum pos1[3], const mjtNum pos2[3],
                  mjtNum* jac1p, mjtNum* jac2p, mjtNum* jacdifp,
                  mjtNum* jac1r, mjtNum* jac2r, mjtNum* jacdifr,
                  int issparse, int flg_skipcommon) {
  // simple path concatenates the dofs of two simple bodies, so they must differ
  int issimple = (b1 != b2 && m->body_simple[b1] && m->body_simple[b2]);
  int NV = m->nv;

  // skip if no DOFs
  if (!NV) {
    return 0;
  }

  // construct merged chain of body dofs
  if (issparse) {
    if (issimple) {
      NV = mj_mergeChainSimple(m, chain, b1, b2);
    } else {
      NV = mj_mergeChain(m, chain, b1, b2, flg_skipcommon);
    }
  }

  // skip if empty chain
  if (!NV) {
    return 0;
  }

  // count-only mode
  if (!jacdifp && !jacdifr && !jac1p && !jac1r) {
    return NV;
  }

  // sparse case
  if (issparse) {
    // simple: fast processing
    if (issimple) {
      // first body
      mj_jacSparseSimple(m, d, jacdifp, jacdifr, pos1, b1, 0, NV,
                         b1 < b2 ? 0 : m->body_dofnum[b2]);

      // second body
      mj_jacSparseSimple(m, d, jacdifp, jacdifr, pos2, b2, 1, NV,
                         b2 < b1 ? 0 : m->body_dofnum[b1]);
    }

    // regular processing
    else {
      // Jacobians
      mj_jacSparse(m, d, jac1p, jac1r, pos1, b1, NV, chain, flg_skipcommon);
      mj_jacSparse(m, d, jac2p, jac2r, pos2, b2, NV, chain, flg_skipcommon);

      // differences
      if (jacdifp) {
        mju_sub(jacdifp, jac2p, jac1p, 3*NV);
      }
      if (jacdifr) {
        mju_sub(jacdifr, jac2r, jac1r, 3*NV);
      }
    }
  }

  // dense case
  else {
    // Jacobians
    mj_jac(m, d, jac1p, jac1r, pos1, b1);
    mj_jac(m, d, jac2p, jac2r, pos2, b2);

    // differences
    if (jacdifp) {
      mju_sub(jacdifp, jac2p, jac1p, 3*NV);
    }
    if (jacdifr) {
      mju_sub(jacdifr, jac2r, jac1r, 3*NV);
    }
  }

  return NV;
}


// dense or sparse weighted sum of multiple body Jacobians at same point
int mj_jacSum(const mjModel* m, mjData* d, int* chain,
              int n, const int* body, const mjtNum* weight,
              const mjtNum point[3], mjtNum* jacp, mjtNum* jacr, int flg_rot) {
  int nv = m->nv, NV;

  mj_markStack(d);
  mjtNum* jtmp = mjSTACKALLOC(d, flg_rot ? 6*nv : 3*nv, mjtNum);

  // sparse
  if (mj_isSparse(m)) {
    // the sparse merge produces one packed [jacp; jacr] block; split into the outputs at the end
    mjtNum* jac = mjSTACKALLOC(d, flg_rot ? 6*nv : 3*nv, mjtNum);
    mjtNum* buf = mjSTACKALLOC(d, flg_rot ? 6*nv : 3*nv, mjtNum);
    int* buf_ind = mjSTACKALLOC(d, nv, int);
    int* bodychain = mjSTACKALLOC(d, nv, int);

    // set first (rotational rows packed right after the translational rows, at offset 3*NV)
    NV = mj_bodyChain(m, body[0], chain);
    if (NV) {
      // get Jacobian
      mjtNum* jr = flg_rot ? jac + 3*NV : NULL;
      if (m->body_simple[body[0]]) {
        mj_jacSparseSimple(m, d, jac, jr, point, body[0], 1, NV, 0);
      } else {
        mj_jacSparse(m, d, jac, jr, point, body[0], NV, chain, /*flg_skipcommon=*/0);
      }

      // apply weight
      mju_scl(jac, jac, weight[0], flg_rot ? 6*NV : 3*NV);
    }

    // accumulate remaining
    for (int i=1; i < n; i++) {
      // get body chain and Jacobian
      int bodyNV = mj_bodyChain(m, body[i], bodychain);
      if (!bodyNV) {
        continue;
      }
      mjtNum* jr = flg_rot ? jtmp + 3*bodyNV : NULL;
      if (m->body_simple[body[i]]) {
        mj_jacSparseSimple(m, d, jtmp, jr, point, body[i], 1, bodyNV, 0);
      } else {
        mj_jacSparse(m, d, jtmp, jr, point, body[i], bodyNV, bodychain, /*flg_skipcommon=*/0);
      }

      // combine sparse matrices
      NV = mju_addToSparseMat(jac, jtmp, nv, flg_rot ? 6 : 3, weight[i],
                              NV, bodyNV, chain, bodychain, buf, buf_ind);
    }

    // split the packed block into the separate output buffers (each NV-packed)
    mju_copy(jacp, jac, 3*NV);
    if (flg_rot) {
      mju_copy(jacr, jac + 3*NV, 3*NV);
    }
  }

  // dense
  else {
    mjtNum* jr = flg_rot ? jtmp + 3*nv : NULL;

    // set first
    mj_jac(m, d, jacp, flg_rot ? jacr : NULL, point, body[0]);
    mju_scl(jacp, jacp, weight[0], 3*nv);
    if (flg_rot) {
      mju_scl(jacr, jacr, weight[0], 3*nv);
    }

    // accumulate remaining
    for (int i=1; i < n; i++) {
      mj_jac(m, d, jtmp, jr, point, body[i]);
      mju_addToScl(jacp, jtmp, weight[i], 3*nv);
      if (flg_rot) {
        mju_addToScl(jacr, jr, weight[i], 3*nv);
      }
    }

    NV = nv;
  }

  mj_freeStack(d);

  return NV;
}


// compute barycentric weights w of the point closest to p on the simplex with n vertices x
static void simplexWeights(mjtNum w[4], const mjtNum p[3], const mjtNum* x[4], int n) {
  // vertex
  if (n == 1) {
    w[0] = 1;
  }

  // segment
  else if (n == 2) {
    mjtNum ab[3], ap[3];
    mju_sub3(ab, x[1], x[0]);
    mju_sub3(ap, p, x[0]);
    mjtNum len2 = mju_dot3(ab, ab);
    mjtNum t = len2 > mjMINVAL ? mju_clip(mju_dot3(ap, ab) / len2, 0, 1) : 0;
    w[0] = 1 - t;
    w[1] = t;
  }

  // triangle
  else if (n == 3) {
    mjc_triWeights(w, p, x[0], x[1], x[2]);
  }

  // tetrahedron
  else {
    // solve p - x0 = sum_i w_i (x_i - x0) with Cramer's rule, done if p is inside
    mjtNum e1[3], e2[3], e3[3], ep[3], c[3];
    mju_sub3(e1, x[1], x[0]);
    mju_sub3(e2, x[2], x[0]);
    mju_sub3(e3, x[3], x[0]);
    mju_sub3(ep, p, x[0]);
    mji_cross(c, e2, e3);
    mjtNum det = mju_dot3(e1, c);
    if (mju_abs(det) > mjMINVAL) {
      w[1] = mju_dot3(ep, c) / det;
      mji_cross(c, ep, e3);
      w[2] = mju_dot3(e1, c) / det;
      mji_cross(c, e2, ep);
      w[3] = mju_dot3(e1, c) / det;
      w[0] = 1 - w[1] - w[2] - w[3];
      if (w[0] >= 0 && w[1] >= 0 && w[2] >= 0 && w[3] >= 0) {
        return;
      }
    }

    // p is outside: closest point on the faces, face i is opposite vertex i
    static const int face[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
    mjtNum mindist = mjMAXVAL;
    mju_fill(w, 0.25, 4);
    for (int i=0; i < 4; i++) {
      const int* f = face[i];
      mjtNum wf[3], cp[3] = {0, 0, 0};
      mjc_triWeights(wf, p, x[f[0]], x[f[1]], x[f[2]]);
      for (int k=0; k < 3; k++) {
        mju_addToScl3(cp, x[f[k]], wf[k]);
      }
      mjtNum dist = mju_dist3(cp, p);
      if (dist < mindist) {
        mindist = dist;
        w[i] = 0;
        for (int k=0; k < 3; k++) {
          w[f[k]] = wf[k];
        }
      }
    }
  }

  // degenerate simplex: uniform weights
  for (int i=0; i < n; i++) {
    if (mju_isBad(w[i])) {
      mju_fill(w, 1.0/n, n);
      return;
    }
  }
}


// compute element vertices and weights for given contact point, return #vertices
//   weights are the barycentric coordinates of the point on the element closest to the contact
//   if v is one of the element vertices, reduce element to fragment
static int mj_elemVertWeight(const mjModel* m, const mjData* d, int f, int e, int v,
                             const mjtNum point[3], int* vertid, mjtNum* weight) {
  // get flex info
  int dim = m->flex_dim[f];
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim+1);
  const mjtNum* vert = d->flexvert_xpos + 3*m->flex_vertadr[f];

  // save vertex ids and positions, skip vertex v
  int n = 0;
  const mjtNum* x[4];
  for (int i=0; i <= dim; i++) {
    if (edata[i] != v) {
      vertid[n] = m->flex_vertadr[f] + edata[i];
      x[n++] = vert + 3*edata[i];
    }
  }

  // barycentric weights of the closest point on the element or fragment
  simplexWeights(weight, point, x, n);
  return n;
}


// compute body weights for a given contact vertex, return number of bodies
static int mj_vertBodyWeight(const mjModel* m, const mjData* d, int f, int* v,
                             int* body, mjtNum* bweight, const mjtNum* vweight, int nw) {
  if (nw == 0) {
    return 0;
  }

  // compute parametric coordinates
  mjtNum coord[3] = {0, 0, 0};
  for (int i = 0; i < nw; i++) {
    mju_addToScl3(coord, m->flex_vert0 + 3*v[i], vweight[i]);
  }

  int interp = m->flex_interp[f];
  int order = interp < 0 ? -interp : interp;
  int npc = (order+1)*(order+1)*(order+1);  // number of nodes per cell

  // grid dimensions for shell mode
  int nx = 0, ny = 0, nz = 0;
  if (interp < 0) {
    nx = m->flex_cellnum[3*f+0] * order + 1;
    ny = m->flex_cellnum[3*f+1] * order + 1;
    nz = m->flex_cellnum[3*f+2] * order + 1;
  }

  // cell lookup: get local coords and node indices
  const int* cellnum = m->flex_cellnum + 3*f;
  int cell = -1;
  if (nw == 1) {
    cell = m->flex_vertcell[v[0]];
  } else {
    int cx = cellnum[0], cy = cellnum[1], cz = cellnum[2];
    int ci = mjMAX(0, mjMIN(cx - 1, (int)mju_floor(coord[0] * cx)));
    int cj = mjMAX(0, mjMIN(cy - 1, (int)mju_floor(coord[1] * cy)));
    int ck = mjMAX(0, mjMIN(cz - 1, (int)mju_floor(coord[2] * cz)));
    int nat_cell = ci * cy * cz + cj * cz + ck;
    int best = 0;
    for (int i = 0; i < nw; i++) {
      if (m->flex_vertcell[v[i]] == nat_cell) {
        cell = nat_cell;
        break;
      }
      if (vweight[i] > vweight[best]) {
        best = i;
      }
    }
    if (cell < 0) {
      cell = m->flex_vertcell[v[best]];
    }
  }
  mjtNum local[3];
  int nodeindices[27];  // max npc for quadratic: 3^3 = 27
  mju_cellLookup(coord, cellnum, order, cell, local, nodeindices);

  // evaluate basis functions for this cell's local nodes
  int nstart = m->flex_nodeadr[f];
  int nb = 0;

  if (!m->flex_nodebodyid) {
    return 0;
  }

  if (npc > 27) {
    for (int j = 0; j < npc; j++) {
      mjtNum w = mju_evalBasis(local, j, order);
      if (mju_abs(w) < 1e-5) {
        continue;
      }

      int idx = nodeindices[j];

      // shell mode: map interior nodes to boundary
      if (interp < 0) {
        int k_idx = idx % nz;
        int rest = idx / nz;
        int j_idx = rest % ny;
        int i_idx = rest / ny;

        if (i_idx > 0 && i_idx < nx-1 && j_idx > 0 && j_idx < ny-1 && k_idx > 0 && k_idx < nz-1) {
          mju_shellTFIWeights(nx, ny, nz, i_idx, j_idx, k_idx, w, &nb, body, bweight,
                              m->flex_nodebodyid, nstart);
          continue;
        }
      }

      // add node, check for duplicates (especially needed when combining with TFI)
      int b = m->flex_nodebodyid[nstart + idx];
      int found = 0;
      for (int k = 0; k < nb; k++) {
        if (body[k] == b) {
          if (bweight) bweight[k] += w;
          found = 1;
          break;
        }
      }
      if (!found) {
        if (bweight) bweight[nb] = w;
        body[nb++] = b;
      }
    }
  } else {
    mjtNum basis[27];
    mju_evalBasisArray(basis, local, order);

    for (int j = 0; j < npc; j++) {
      mjtNum w = basis[j];
      if (mju_abs(w) < 1e-5) {
        continue;
      }

      int idx = nodeindices[j];

      // shell mode: map interior nodes to boundary
      if (interp < 0) {
        int k_idx = idx % nz;
        int rest = idx / nz;
        int j_idx = rest % ny;
        int i_idx = rest / ny;

        if (i_idx > 0 && i_idx < nx-1 && j_idx > 0 && j_idx < ny-1 && k_idx > 0 && k_idx < nz-1) {
          mju_shellTFIWeights(nx, ny, nz, i_idx, j_idx, k_idx, w, &nb, body, bweight,
                              m->flex_nodebodyid, nstart);
          continue;
        }
      }

      // add node, check for duplicates (especially needed when combining with TFI)
      int b = m->flex_nodebodyid[nstart + idx];
      int found = 0;
      for (int k = 0; k < nb; k++) {
        if (body[k] == b) {
          if (bweight) bweight[k] += w;
          found = 1;
          break;
        }
      }
      if (!found) {
        if (bweight) bweight[nb] = w;
        body[nb++] = b;
      }
    }
  }


  return nb;
}


// compute bodies and weights of one side of a contact, return number of bodies
//   the weights interpolate the side at the contact point; weight can be NULL
int mj_contactBodyWeight(const mjModel* m, const mjData* d, const mjContact* con, int side,
                         int* body, mjtNum* weight) {
  // geom
  if (con->geom[side] >= 0) {
    body[0] = m->geom_bodyid[con->geom[side]];
    if (weight) {
      weight[0] = 1;
    }
    return 1;
  }

  // flex vertex or element
  int f = con->flex[side];
  int nw;
  int vid[4];
  mjtNum vweight[4];
  if (con->vert[side] >= 0) {
    vid[0] = m->flex_vertadr[f] + con->vert[side];
    vweight[0] = 1;
    nw = 1;
  } else {
    int v = (con->flex[1-side] == f ? con->vert[1-side] : -1);

    // vertex flex without weights: element vertex bodies, skipping v
    if (!weight && !m->flex_interp[f]) {
      int dim = m->flex_dim[f];
      const int* edata = m->flex_elem + m->flex_elemdataadr[f] + con->elem[side]*(dim+1);
      int nb = 0;
      for (int k=0; k <= dim; k++) {
        if (edata[k] != v) {
          body[nb++] = m->flex_vertbodyid[m->flex_vertadr[f] + edata[k]];
        }
      }
      return nb;
    }

    nw = mj_elemVertWeight(m, d, f, con->elem[side], v, con->pos, vid, vweight);
  }

  // interpolated flex: node bodies
  if (m->flex_interp[f]) {
    return mj_vertBodyWeight(m, d, f, vid, body, weight, vweight, nw);
  }

  // vertex bodies
  for (int k=0; k < nw; k++) {
    body[k] = m->flex_vertbodyid[vid[k]];
    if (weight) {
      weight[k] = vweight[k];
    }
  }
  return nw;
}


// compute 3/6-by-nv Jacobian time derivative of global point attached to given body
void mj_jacDot(const mjModel* m, const mjData* d,
               mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body) {
  int nv = m->nv;
  mjtNum offset[3];
  mjtNum pvel[6];  // point velocity (rot:lin order)

  // clear jacobians, compute offset and pvel if required
  if (jacp) {
    mju_zero(jacp, 3*nv);
    const mjtNum* com = d->subtree_com+3*m->body_rootid[body];
    mju_sub3(offset, point, com);
    mju_transformSpatial(pvel, d->cvel+6*body, 0, point, com, 0);
  }
  if (jacr) {
    mju_zero(jacr, 3*nv);
  }

  // skip fixed bodies
  body = m->body_weldid[body];

  // weld root has no dofs: nothing to do
  if (m->body_dofnum[body] == 0) {
    return;
  }

  // get last dof that affects this (as well as the original) body
  int i = m->body_dofadr[body] + m->body_dofnum[body] - 1;

  // backward pass over dof ancestor chain
  while (i >= 0) {
    mjtNum cdof_dot[6];
    mji_copy6(cdof_dot, d->cdof_dot+6*i);
    mjtNum* cdof = d->cdof+6*i;

    // check for quaternion
    mjtJoint type = m->jnt_type[m->dof_jntid[i]];
    int dofadr = m->jnt_dofadr[m->dof_jntid[i]];
    int is_quat = type == mjJNT_BALL || (type == mjJNT_FREE && i >= dofadr + 3);

    // compute cdof_dot for quaternion (use current body cvel)
    if (is_quat) {
      mji_crossMotion(cdof_dot, d->cvel+6*m->dof_bodyid[i], cdof);
    }

    // construct rotation jacobian
    if (jacr) {
      jacr[i+0*nv] += cdof_dot[0];
      jacr[i+1*nv] += cdof_dot[1];
      jacr[i+2*nv] += cdof_dot[2];
    }

    // construct translation jacobian (correct for rotation)
    if (jacp) {
      // first correction term, account for varying cdof
      mjtNum tmp1[3];
      mji_cross(tmp1, cdof_dot, offset);

      // second correction term, account for point translational velocity
      mjtNum tmp2[3];
      mji_cross(tmp2, cdof, pvel + 3);

      jacp[i+0*nv] += cdof_dot[3] + tmp1[0] + tmp2[0];
      jacp[i+1*nv] += cdof_dot[4] + tmp1[1] + tmp2[1];
      jacp[i+2*nv] += cdof_dot[5] + tmp1[2] + tmp2[2];
    }

    // advance to parent dof
    i = m->dof_parentid[i];
  }
}


// compute 3/6-by-NV sparse Jacobian time derivative of global point attached to given body
void mj_jacDotSparse(const mjModel* m, const mjData* d,
                     mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body,
                     int NV, const int* chain) {
  mjtNum offset[3];
  mjtNum pvel[6];

  // clear jacobians, compute offset and pvel if required
  if (jacp) {
    mju_zero(jacp, 3*NV);
    const mjtNum* com = d->subtree_com+3*m->body_rootid[body];
    mju_sub3(offset, point, com);
    mju_transformSpatial(pvel, d->cvel+6*body, 0, point, com, 0);
  }
  if (jacr) {
    mju_zero(jacr, 3*NV);
  }

  // skip fixed bodies
  body = m->body_weldid[body];

  // weld root has no dofs: nothing to do
  if (m->body_dofnum[body] == 0) {
    return;
  }

  // get last dof that affects this body
  int da = m->body_dofadr[body] + m->body_dofnum[body] - 1;

  // start at end of chain (chain is in increasing order)
  int ci = NV-1;

  // backward pass over dof ancestor chain
  while (da >= 0) {
    // find chain index for this dof
    while (ci >= 0 && chain[ci] > da) {
      ci--;
    }

    // dof not in chain: SHOULD NOT OCCUR
    if (ci < 0 || chain[ci] != da) {
      mjERROR("dof index %d not found in chain", da);
    }

    mjtNum cdof_dot[6];
    mji_copy6(cdof_dot, d->cdof_dot+6*da);
    mjtNum* cdof = d->cdof+6*da;

    // check for quaternion
    mjtJoint type = m->jnt_type[m->dof_jntid[da]];
    int dofadr = m->jnt_dofadr[m->dof_jntid[da]];
    int is_quat = type == mjJNT_BALL || (type == mjJNT_FREE && da >= dofadr + 3);

    // compute cdof_dot for quaternion (use current body cvel)
    if (is_quat) {
      mji_crossMotion(cdof_dot, d->cvel+6*m->dof_bodyid[da], cdof);
    }

    // construct rotation jacobian
    if (jacr) {
      jacr[ci+0*NV] += cdof_dot[0];
      jacr[ci+1*NV] += cdof_dot[1];
      jacr[ci+2*NV] += cdof_dot[2];
    }

    // construct translation jacobian (correct for rotation)
    if (jacp) {
      // first correction term, account for varying cdof
      mjtNum tmp1[3];
      mji_cross(tmp1, cdof_dot, offset);

      // second correction term, account for point translational velocity
      mjtNum tmp2[3];
      mji_cross(tmp2, cdof, pvel + 3);

      jacp[ci+0*NV] += cdof_dot[3] + tmp1[0] + tmp2[0];
      jacp[ci+1*NV] += cdof_dot[4] + tmp1[1] + tmp2[1];
      jacp[ci+2*NV] += cdof_dot[5] + tmp1[2] + tmp2[2];
    }

    // advance to parent dof
    da = m->dof_parentid[da];
  }
}


// compute subtree angular momentum matrix
void mj_angmomMat(const mjModel* m, mjData* d, mjtNum* mat, int body) {
  int nv = m->nv;
  mj_markStack(d);

  // stack allocations
  mjtNum* jacp = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacr = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* term1 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* term2 = mjSTACKALLOC(d, 3*nv, mjtNum);

  // clear output
  mju_zero(mat, 3*nv);

  // save the location of the subtree COM
  mjtNum subtree_com[3];
  mju_copy3(subtree_com, d->subtree_com+3*body);

  for (int b=body; b < m->nbody; b++) {
    // end of body subtree, break from the loop
    if (b > body && m->body_parentid[b] < body) {
      break;
    }

    // linear and angular velocity Jacobian of the body COM (inertial frame)
    mj_jacBodyCom(m, d, jacp, jacr, b);

    // orientation of the COM (inertial) frame of b-th body
    mjtNum ximat[9];
    mji_copy9(ximat, d->ximat+9*b);

    // save the inertia matrix of b-th body
    mjtNum inertia[9] = {0};
    inertia[0] = m->body_inertia[3*b+0];  // inertia(1,1)
    inertia[4] = m->body_inertia[3*b+1];  // inertia(2,2)
    inertia[8] = m->body_inertia[3*b+2];  // inertia(3,3)

    // term1 = body angular momentum about self COM in world frame
    mjtNum tmp1[9], tmp2[9];
    mji_mulMatMat3(tmp1, ximat, inertia);          // tmp1  = ximat * inertia
    mju_mulMatMatT3(tmp2, tmp1, ximat);            // tmp2  = ximat * inertia * ximat^T
    mju_mulMatMat(term1, tmp2, jacr, 3, 3, nv);    // term1 = ximat * inertia * ximat^T * jacr

    // location of body COM w.r.t subtree COM
    mjtNum com[3];
    mji_sub3(com, d->xipos+3*b, subtree_com);

    // skew symmetric matrix representing body_com vector
    mjtNum com_mat[9] = {0};
    com_mat[1] = -com[2];
    com_mat[2] = com[1];
    com_mat[3] = com[2];
    com_mat[5] = -com[0];
    com_mat[6] = -com[1];
    com_mat[7] = com[0];

    // term2 = moment of linear momentum
    mju_mulMatMat(term2, com_mat, jacp, 3, 3, nv);   // term2 = com_mat * jacp
    mju_scl(term2, term2, m->body_mass[b], 3 * nv);  // term2 = com_mat * jacp * mass

    // mat += term1 + term2
    mju_addTo(mat, term1, 3*nv);
    mju_addTo(mat, term2, 3*nv);
  }

  mj_freeStack(d);
}


//-------------------------- spatial frame utilities -----------------------------------------------

// compute object 6D velocity in object-centered frame, world/local orientation
void mj_objectVelocity(const mjModel* m, const mjData* d,
                       int objtype, int objid, mjtNum res[6], int flg_local) {
  int bodyid = 0;
  const mjtNum *pos = 0, *rot = 0;

  // body-inertial
  if (objtype == mjOBJ_BODY) {
    bodyid = objid;
    pos = d->xipos+3*objid;
    rot = (flg_local ? d->ximat+9*objid : 0);
  }

  // body-regular
  else if (objtype == mjOBJ_XBODY) {
    bodyid = objid;
    pos = d->xpos+3*objid;
    rot = (flg_local ? d->xmat+9*objid : 0);
  }

  // geom
  else if (objtype == mjOBJ_GEOM) {
    bodyid = m->geom_bodyid[objid];
    pos = d->geom_xpos+3*objid;
    rot = (flg_local ? d->geom_xmat+9*objid : 0);
  }

  // site
  else if (objtype == mjOBJ_SITE) {
    bodyid = m->site_bodyid[objid];
    pos = d->site_xpos+3*objid;
    rot = (flg_local ? d->site_xmat+9*objid : 0);
  }

  // camera
  else if (objtype == mjOBJ_CAMERA) {
    bodyid = m->cam_bodyid[objid];
    pos = d->cam_xpos+3*objid;
    rot = (flg_local ? d->cam_xmat+9*objid : 0);
  }

  // object without spatial frame
  else {
    mjERROR("invalid object type %d", objtype);
  }

  // dof-less body (static or mocap): quick return
  if (m->body_dofnum[m->body_weldid[bodyid]] == 0) {
    mju_zero(res, 6);
    return;
  }

  // transform velocity
  mju_transformSpatial(res, d->cvel+6*bodyid, 0, pos, d->subtree_com+3*m->body_rootid[bodyid], rot);
}


// compute material surface velocity of a geom at a point, in the world frame
void mj_geomSurfaceVelocity(const mjModel* m, const mjData* d, int geomid,
                            const mjtNum point[3], mjtNum linear[3], mjtNum angular[3]) {
  const mjtNum* sv = m->geom_surfacevel + 6*geomid;

  // rotate local linear and angular surface velocities to the world frame
  mji_mulMatVec3(linear, d->geom_xmat + 9*geomid, sv);
  mji_mulMatVec3(angular, d->geom_xmat + 9*geomid, sv + 3);

  // add angular velocity contribution (w x r) at the query point
  mjtNum arm[3], wxr[3];
  mji_sub3(arm, point, d->geom_xpos + 3*geomid);
  mji_cross(wxr, angular, arm);
  mji_addTo3(linear, wxr);
}


// compute object 6D acceleration in object-centered frame, world/local orientation
void mj_objectAcceleration(const mjModel* m, const mjData* d,
                           int objtype, int objid, mjtNum res[6], int flg_local) {
  int bodyid = 0;
  const mjtNum *pos = 0, *rot = 0;

  // body-inertial
  if (objtype == mjOBJ_BODY) {
    bodyid = objid;
    pos = d->xipos+3*objid;
    rot = (flg_local ? d->ximat+9*objid : 0);
  }

  // body-regular
  else if (objtype == mjOBJ_XBODY) {
    bodyid = objid;
    pos = d->xpos+3*objid;
    rot = (flg_local ? d->xmat+9*objid : 0);
  }

  // geom
  else if (objtype == mjOBJ_GEOM) {
    bodyid = m->geom_bodyid[objid];
    pos = d->geom_xpos+3*objid;
    rot = (flg_local ? d->geom_xmat+9*objid : 0);
  }

  // site
  else if (objtype == mjOBJ_SITE) {
    bodyid = m->site_bodyid[objid];
    pos = d->site_xpos+3*objid;
    rot = (flg_local ? d->site_xmat+9*objid : 0);
  }

  // camera
  else if (objtype == mjOBJ_CAMERA) {
    bodyid = m->cam_bodyid[objid];
    pos = d->cam_xpos+3*objid;
    rot = (flg_local ? d->cam_xmat+9*objid : 0);
  }

  // object without spatial frame
  else {
    mjERROR("invalid object type %d", objtype);
  }

  // dof-less body (static or mocap): quick return
  if (m->body_dofnum[m->body_weldid[bodyid]] == 0) {
    mju_zero(res, 6);
    return;
  }

  // transform com-based acceleration to local frame
  mju_transformSpatial(res, d->cacc+6*bodyid, 0, pos, d->subtree_com+3*m->body_rootid[bodyid], rot);

  // transform com-based velocity to local frame
  mjtNum vel[6];
  mju_transformSpatial(vel, d->cvel+6*bodyid, 0, pos, d->subtree_com+3*m->body_rootid[bodyid], rot);

  // add Coriolis correction due to rotating frame:  acc_tran += vel_rot x vel_tran
  mjtNum correction[3];
  mji_cross(correction, vel, vel+3);
  mji_addTo3(res+3, correction);
}


// map from body local to global Cartesian position, quaternion and matrix
void mj_local2GlobalQuat(mjData* d, mjtNum* restrict xpos, mjtNum* restrict xquat,
                         mjtNum* restrict xmat, const mjtNum pos[3], const mjtNum quat[4],
                         int body, mjtByte sameframe) {
  mjtSameFrame sf = sameframe;

  // position
  if (xpos && pos) {
    switch (sf) {
    case mjSAMEFRAME_NONE:
    case mjSAMEFRAME_BODYROT:
    case mjSAMEFRAME_INERTIAROT:
      mji_mulMatVec3(xpos, d->xmat+9*body, pos);
      mji_addTo3(xpos, d->xpos+3*body);
      break;
    case mjSAMEFRAME_BODY:
      mji_copy3(xpos, d->xpos+3*body);
      break;
    case mjSAMEFRAME_INERTIA:
      mji_copy3(xpos, d->xipos+3*body);
      break;
    }
  }

  // orientation
  if ((xquat || xmat) && quat) {
    mjtNum tmp[4];
    mjtNum* q = xquat ? xquat : tmp;
    switch (sf) {
    case mjSAMEFRAME_NONE:
      mji_mulQuat(q, d->xquat+4*body, quat);
      if (xmat) mji_quat2Mat(xmat, q);
      break;
    case mjSAMEFRAME_BODY:
    case mjSAMEFRAME_BODYROT:
      if (xquat) mji_copy4(xquat, d->xquat+4*body);
      if (xmat) mji_copy9(xmat, d->xmat+9*body);
      break;
    case mjSAMEFRAME_INERTIA:
    case mjSAMEFRAME_INERTIAROT:
      if (xquat) mji_copy4(xquat, d->xiquat+4*body);
      if (xmat) mji_copy9(xmat, d->ximat+9*body);
      break;
    }
  }
}


// map from body local to global Cartesian coordinates
void mj_local2Global(mjData* d, mjtNum xpos[3], mjtNum xmat[9],
                     const mjtNum pos[3], const mjtNum quat[4],
                     int body, mjtByte sameframe) {
  mj_local2GlobalQuat(d, xpos, NULL, xmat, pos, quat, body, sameframe);
}


//-------------------------- miscellaneous utilities -----------------------------------------------

// Check weld parent for independent ordered XYZ slides. Used only to select fast paths;
// general attachments use the point Jacobian. The same test sizes the constant factor in mjCFlex.
int mj_flexBodySimple(const mjModel* m, int body) {
  body = m->body_weldid[body];
  if (m->body_dofnum[body] != 3 || m->body_jntnum[body] != 3 ||
      m->body_dofnum[m->body_weldid[m->body_parentid[body]]] != 0) {
    return 0;
  }
  int jadr = m->body_jntadr[body];
  for (int j=0; j < 3; j++) {
    if (m->jnt_type[jadr+j] != mjJNT_SLIDE) return 0;
    for (int k=0; k < 3; k++) {
      if (mju_abs(m->jnt_axis[3*(jadr+j)+k] - (j == k)) > mjMINVAL) return 0;
    }
  }
  return 1;
}


// Check for standard fixed-frame three-DOF assembly flex and compatibility with cached factor.
int mj_flexSimple(const mjModel* m, int f) {
  if (m->flex_interp[f]) {
    return 0;
  }
  for (int v=m->flex_vertadr[f]; v < m->flex_vertadr[f]+m->flex_vertnum[f]; v++) {
    int body = m->body_weldid[m->flex_vertbodyid[v]];
    if (m->body_dofnum[body] && !mj_flexBodySimple(m, body)) return 0;
  }
  return 1;
}


// Apply the point Jacobian or its transpose without constructing a matrix. The general
// path walks the same motion axes as mj_jac, including every ancestor and the vertex offset.
// Gather overwrites 3*nvert world components; scatter adds to nv generalized components.
static void flexMap(const mjModel* m, const mjData* d, int f, mjtNum* res,
                    const mjtNum* vec, mjtNum scale, int transpose) {
  int vadr = m->flex_vertadr[f];
  for (int v=0; v < m->flex_vertnum[f]; v++) {
    int body = m->body_weldid[m->flex_vertbodyid[vadr+v]];
    if (!transpose) mji_zero3(res + 3*v);
    if (!m->body_dofnum[body]) continue;
    int da = m->body_dofadr[body];
    if (m->body_simple[body] == 2) {
      // The world-space slide axes are already in cdof, including axis order and signs.
      for (int j=0; j < m->body_dofnum[body]; j++) {
        const mjtNum* axis = d->cdof + 6*(da+j)+3;
        if (transpose) {
          res[da+j] += scale*mju_dot3(axis, vec + 3*v);
        } else {
          mji_addToScl3(res + 3*v, axis, vec[da+j]);
        }
      }
      continue;
    }
    mjtNum offset[3];
    mji_sub3(offset, d->flexvert_xpos + 3*(vadr+v),
             d->subtree_com + 3*m->body_rootid[body]);
    for (int i=da + m->body_dofnum[body]-1; i >= 0; i=m->dof_parentid[i]) {
      mjtNum column[3];
      mji_cross(column, d->cdof + 6*i, offset);
      mji_addTo3(column, d->cdof + 6*i+3);
      if (transpose) {
        res[i] += scale*mju_dot3(column, vec + 3*v);
      } else {
        mji_addToScl3(res + 3*v, column, vec[i]);
      }
    }
  }
}


// Gather an arbitrary generalized vector (including qvel) in world vertex coordinates.
void mj_flexGather(const mjModel* m, const mjData* d, int f, mjtNum* res, const mjtNum* vec) {
  flexMap(m, d, f, res, vec, 1, 0);
}


// Accumulate world vertex forces in generalized coordinates, including pin reactions.
void mj_flexScatter(const mjModel* m, const mjData* d, int f, mjtNum* res,
                    const mjtNum* vec, mjtNum scale) {
  flexMap(m, d, f, res, vec, scale, 1);
}


// gather global node positions and velocities
void mju_flexGatherState(const mjModel* m, const mjData* d, int f, mjtNum* xpos, mjtNum* vel) {
  int nodenum = m->flex_nodenum[f];
  int nstart = m->flex_nodeadr[f];
  int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];

  // compute positions and velocities
  for (int i=0; i < nodenum; i++) {
    int bid = bodyid[i];
    if (m->flex_centered[f] ||
        (m->flex_node[3*(i+nstart)+0] == 0 &&
         m->flex_node[3*(i+nstart)+1] == 0 &&
         m->flex_node[3*(i+nstart)+2] == 0)) {
      mju_copy3(xpos + 3*i, d->xpos + 3*bid);
    } else {
      mju_mulMatVec3(xpos + 3*i, d->xmat + 9*bid, m->flex_node + 3*(i+nstart));
      mju_addTo3(xpos + 3*i, d->xpos + 3*bid);
    }

    if (vel) {
      mjtNum body_vel[6];
      mj_objectVelocity(m, d, mjOBJ_BODY, bid, body_vel, 0);  // returns [omega, v_CoM] in world frame

      // linear velocity at CoM
      mju_copy3(vel + 3*i, body_vel + 3);

      // add omega x (xpos - xipos)
      mjtNum r[3], cross[3];
      mju_sub3(r, xpos + 3*i, d->xipos + 3*bid);
      mju_cross(cross, body_vel, r);
      mju_addTo3(vel + 3*i, cross);
    }
  }

  // shell mode: reconstruct interior node positions and velocities via TFI
  int interp = m->flex_interp[f];
  if (interp < 0) {
    int order = -interp;
    int cx = m->flex_cellnum[3*f+0];
    int cy = m->flex_cellnum[3*f+1];
    int cz = m->flex_cellnum[3*f+2];
    int nx_g = cx * order + 1;
    int ny_g = cy * order + 1;
    int nz_g = cz * order + 1;

    mju_shellTrackInterior(xpos, nx_g, ny_g, nz_g);
    if (vel) {
      mju_shellTrackInterior(vel, nx_g, ny_g, nz_g);
    }
  }
}


// extract 6D force:torque for one contact, in contact frame
void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]) {
  mjContact* con;

  // clear result
  mju_zero(result, 6);

  // make sure contact is valid
  if (id >= 0 && id < d->ncon && d->contact[id].efc_address >= 0) {
    // get contact pointer
    con = d->contact + id;

    if (mj_isPyramidal(m)) {
      mju_decodePyramid(result, d->efc_force + con->efc_address, con->friction, con->dim);
    } else {
      mju_copy(result, d->efc_force + con->efc_address, con->dim);
    }

    // report the net interface force: the solver's cone force minus the adhesive pull
    result[0] -= con->adhesion;
  }
}


// count the number of length limit violations for tendon i (0, 1 or 2)
int tendonLimit(const mjModel* m, const mjtNum* ten_length, int i) {
  if (!m->tendon_limited[i]) {
    return 0;
  }

  int nl = 0;
  mjtNum value = ten_length[i];
  mjtNum margin = m->tendon_margin[i];

  // tendon limits can be bilateral, check both sides
  for (int side = -1; side <= 1; side += 2) {
    mjtNum dist = side * (m->tendon_range[2 * i + (side + 1) / 2] - value);
    if (dist < margin) nl++;
  }

  return nl;
}


// compute spring and damper forces along tendon i, zero when disabled
void mj_tendonSpringDamper(const mjModel* m, const mjData* d, int i,
                           mjtNum* frc_spring, mjtNum* frc_damper) {
  *frc_spring = 0;
  *frc_damper = 0;

  // spring force: displacement outside the spring range
  if (!mjDISABLED(mjDSBL_SPRING)) {
    mjtNum stiffness = m->tendon_stiffness[i];
    const mjtNum* spoly = m->tendon_stiffnesspoly + mjNPOLY*i;
    if (stiffness || !mju_isZero(spoly, mjNPOLY)) {
      mjtNum length = d->ten_length[i];
      const mjtNum* lengthspring = m->tendon_plasticadr[i] >= 0 ?
                                   d->plastic + m->tendon_plasticadr[i] :
                                   m->tendon_lengthspring + 2*i;
      mjtNum lower = lengthspring[0];
      mjtNum upper = lengthspring[1];
      mjtNum x = (length > upper) ? length - upper : (length < lower) ? length - lower : 0;
      *frc_spring = -x * mju_polyForce(stiffness, spoly, x, mjNPOLY, 0);
    }
  }

  // damper force: velocity, damping includes the contribution of actuators
  if (!mjDISABLED(mjDSBL_DAMPER)) {
    mjtNum dpoly[mjNPOLY];
    mju_copy(dpoly, m->tendon_dampingpoly + mjNPOLY*i, mjNPOLY);
    mjtNum damping = m->tendon_damping[i] + mj_actuatorDamping(m, mjOBJ_TENDON, i, dpoly);
    if (damping || !mju_isZero(dpoly, mjNPOLY)) {
      mjtNum v = d->ten_velocity[i];
      *frc_damper = -v * mju_polyForce(damping, dpoly, v, mjNPOLY, 1);
    }
  }
}


// return actuator damping contribution to joint or tendon
mjtNum mj_actuatorDamping(const mjModel* m, mjtObj type, int id, mjtNum poly[mjNPOLY]) {
  if (type != mjOBJ_TENDON && type != mjOBJ_JOINT) {
    mjERROR("only joint and tendon objects can inherit damping from actuators");
    return 0;
  }

  // get actuator id
  int actuatorid = type == mjOBJ_JOINT ? m->jnt_actuatorid[id] : m->tendon_actuatorid[id];

  if (actuatorid == -1) {
    return 0;
  }

  mjtNum damping = 0;

  // single actuator contributes damping
  if (actuatorid >= 0) {
    mjtNum gear2 = m->actuator_gear[6*m->actuator_outadr[actuatorid]] * m->actuator_gear[6*m->actuator_outadr[actuatorid]];
    damping = m->actuator_damping[actuatorid] * gear2;
    for (int k = 0; k < mjNPOLY; k++) {
      poly[k] += m->actuator_dampingpoly[mjNPOLY*actuatorid+k] * gear2;
    }
  }

  // actuatorid < -1: scan all actuators for contributions
  else {
    for (int k = 0; k < m->nactuator; k++) {
      // skip actuators that don't actuate the given joint/tendon
      if (m->actuator_trnid[2*k] != id) {
        continue;
      }
      if (type == mjOBJ_JOINT &&
          m->actuator_trntype[k] != mjTRN_JOINT &&
          m->actuator_trntype[k] != mjTRN_JOINTINPARENT) {
        continue;
      }
      if (type == mjOBJ_TENDON && m->actuator_trntype[k] != mjTRN_TENDON) {
        continue;
      }

      // accumulate damping contribution
      mjtNum gear2 = m->actuator_gear[6*m->actuator_outadr[k]] * m->actuator_gear[6*m->actuator_outadr[k]];
      damping += m->actuator_damping[k] * gear2;
      for (int j = 0; j < mjNPOLY; j++) {
        poly[j] += m->actuator_dampingpoly[mjNPOLY*k+j] * gear2;
      }
    }
  }

  return damping;
}


// return actuator armature contribution to joint or tendon
mjtNum mj_actuatorArmature(const mjModel* m, mjtObj type, int id) {
  if (type != mjOBJ_TENDON && type != mjOBJ_JOINT) {
    mjERROR("only joint and tendon objects can inherit armature from actuators");
    return 0;
  }

  // get actuator id
  int actuatorid = type == mjOBJ_JOINT ? m->jnt_actuatorid[id] : m->tendon_actuatorid[id];

  // no actuator contribution
  if (actuatorid == -1) {
    return 0;
  }

  mjtNum armature = 0;

  // single actuator contributes armature
  if (actuatorid >= 0) {
    mjtNum gear2 = m->actuator_gear[6*m->actuator_outadr[actuatorid]] * m->actuator_gear[6*m->actuator_outadr[actuatorid]];
    armature = m->actuator_armature[actuatorid] * gear2;
  }

  // actuatorid < -1: scan all actuators for contributions
  else {
    for (int k = 0; k < m->nactuator; k++) {
      // skip actuators that don't actuate the given joint/tendon
      if (m->actuator_trnid[2*k] != id) {
        continue;
      }
      if (type == mjOBJ_JOINT &&
          m->actuator_trntype[k] != mjTRN_JOINT &&
          m->actuator_trntype[k] != mjTRN_JOINTINPARENT) {
        continue;
      }
      if (type == mjOBJ_TENDON && m->actuator_trntype[k] != mjTRN_TENDON) {
        continue;
      }

      // accumulate armature contribution
      mjtNum gear2 = m->actuator_gear[6*m->actuator_outadr[k]] * m->actuator_gear[6*m->actuator_outadr[k]];
      armature += m->actuator_armature[k] * gear2;
    }
  }

  return armature;
}


// return DC motor winding resistance at the current temperature
mjtNum mj_dcmotorResistance(const mjModel* m, const mjData* d, int id) {
  const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*id;
  const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*id;
  mjtNum R = gainprm[0];
  mjDCMotorSlots slots = mj_dcmotorSlots(dynprm, gainprm);

  // account for temperature if thermal model is enabled
  if (slots.temperature >= 0) {
    mjtNum T = d->act[m->actuator_actadr[id]+slots.temperature];
    mjtNum alpha = gainprm[2];  // temperature coefficient
    mjtNum T0 = gainprm[3];     // reference temperature
    mjtNum Ta = dynprm[4];      // ambient temperature
    R *= 1 + alpha * (T + Ta - T0);
  }

  return mju_max(mjMINVAL, R);
}


// count warnings, print only the first time
void mj_warning(mjData* d, int warning, int info) {
  // check type
  if (warning < 0 || warning >= mjNWARNING) {
    mjERROR("invalid warning type %d", warning);
  }

  // save info (override previous)
  d->warning[warning].lastinfo = info;

  // print message only the first time this warning is encountered
  if (!d->warning[warning].number) {
    mju_warning("%s Time = %.4f.", mju_warningText(warning, info), d->time);
  }

  // increase counter
  d->warning[warning].number++;
}


//-------------------------- effective-metric predicates ------------------------------------------

// the selected integrator performs the constraint solve in the effective metric.
// The option-level gate decision; d->efm_active reports whether the per-step build ran
int mj_isMetric(const mjModel* m) {
  return m->opt.integrator == mjINT_DISCRETE;
}


// do the tendon and actuator classes enter the metric. Under solver=PGS -- and only
// there -- they are excluded and their forces integrate explicitly: the dual assembles
// its constraint-space AR from the backbone factor, which cannot carry their couplings,
// and a consistent backbone metric beats a solve whose forces and accelerations disagree.
// Noslip atop a primal solver keeps the couplings: the main solve runs in the full
// metric and the post-pass consumes the backbone AR as an approximation. Flex, which is
// too stiff to exclude, is rejected by mj_checkDiscrete instead
int mj_effCouplings(const mjModel* m) {
  return mj_isMetric(m) && m->opt.solver != mjSOL_PGS;
}


// tendon i has a spring: nonzero stiffness or stiffness polynomial
int mj_tendonHasStiffness(const mjModel* m, int i) {
  return m->tendon_stiffness[i] != 0 ||
         !mju_isZero(m->tendon_stiffnesspoly + mjNPOLY*i, mjNPOLY);
}


// tendon i has a damper: nonzero damping, damping polynomial, or an attached actuator
int mj_tendonHasDamping(const mjModel* m, int i) {
  return m->tendon_damping[i] != 0 ||
         !mju_isZero(m->tendon_dampingpoly + mjNPOLY*i, mjNPOLY) ||
         m->tendon_actuatorid[i] != -1;
}


// flex f has a nonzero stretch stiffness record (raw stiffness, or the strain equality
// eigenmodes). Volume-mode interpolated flexes store one dense block per cell and empty cells
// (no mesh content) keep an all-zero block, so the first block alone is not representative
int mj_flexHasStiffness(const mjModel* m, int f) {
  int adr = m->flex_stiffnessadr[f];
  if (adr < 0) {
    return 0;
  }
  const mjtNum* k = m->flex_stiffness + adr;

  // standard flex: every element has stiffness if any does
  int order = m->flex_interp[f];
  if (order == 0) {
    return k[0] != 0;
  }

  // shell-mode interpolated flex (interp < 0): stiffness is allocated only when nonzero
  if (order < 0) {
    return 1;
  }

  // volume-mode interpolated flex: any 3D cell counts
  int cx = m->flex_cellnum[3*f+0];
  int cy = m->flex_cellnum[3*f+1];
  int cz = m->flex_cellnum[3*f+2];
  int npe = (order+1)*(order+1)*(order+1);
  int nelem = cx*cy*cz;
  int stride = 9*npe*npe;
  for (int e=0; e < nelem; e++) {
    if (k[(size_t)e*stride] != 0) {
      return 1;
    }
  }
  return 0;
}


// does flex f use the penalty form of passive contact: a deformable flex of dim >= 2
// that asks for it, and not under the ipc flag, which solves the same law for every supported
// flex itself (running both would apply each pair's force twice)
int mj_effFlexContactPossible(const mjModel* m, int f) {
  return m->flex_passive[f] && !m->flex_rigid[f] && m->flex_dim[f] >= 2 &&
         !mjENABLED(mjENBL_IPC);
}


// does flex f contribute elastic stiffness to the metric. Unlike the assembler gate
// mjd_flexStiff_active (engine_derivative.c), interpolated flexes are included: their
// stiffness is carried matrix-free
int mj_effFlexStiffPossible(const mjModel* m, int f) {
  // rigid flexes do not contribute stiffness
  if (m->flex_rigid[f]) {
    return 0;
  }

  // 1D flexes: rod bending
  if (m->flex_dim[f] == 1) {
    return m->flex_bendingadr[f] >= 0;
  }

  // stretch stiffness present (the strain equality mode stores its constraint
  // eigenmodes in this block instead)
  if (m->flex_edgeequality[f] != 3 && mj_flexHasStiffness(m, f)) {
    return 1;
  }

  // bending: an allocated block does not imply stiffness
  // (strain-constrained and zero-elasticity flexes carry an all-zero block)
  int badr = m->flex_bendingadr[f];
  if (badr < 0) {
    return 0;
  }
  int end = m->nflexbending;
  for (int g=f+1; g < m->nflex; g++) {
    if (m->flex_bendingadr[g] >= 0) {
      end = m->flex_bendingadr[g];
      break;
    }
  }
  return !mju_isZero(m->flex_bending + badr, end - badr);
}


// does flex f need the implicit metric treatment: elastic stiffness or passive contact
int mj_effFlexPossible(const mjModel* m, int f) {
  return mj_effFlexStiffPossible(m, f) || mj_effFlexContactPossible(m, f);
}


// does any flex need the implicit metric treatment: elastic stiffness or passive contact
int mj_effFlexAny(const mjModel* m) {
  for (int f=0; f < m->nflex; f++) {
    if (mj_effFlexPossible(m, f)) {
      return 1;
    }
  }
  return 0;
}


// can this tendon contribute to the metric (model-level; mirrored by island discovery
// and the sleep wake rule)
int mj_effTendonPossible(const mjModel* m, int i) {
  return (!mjDISABLED(mjDSBL_SPRING) && mj_tendonHasStiffness(m, i)) ||
         (!mjDISABLED(mjDSBL_DAMPER) && mj_tendonHasDamping(m, i));
}


// can this actuator contribute to the metric (model-level type check; mirrored by island discovery)
int mj_effActuatorPossible(const mjModel* m, int i) {
  if (mjDISABLED(mjDSBL_ACTUATION)) {
    return 0;
  }
  return m->actuator_biastype[i] == mjBIAS_AFFINE  ||
         m->actuator_biastype[i] == mjBIAS_SO3     ||
         m->actuator_biastype[i] == mjBIAS_DCMOTOR ||
         m->actuator_biastype[i] == mjBIAS_MUSCLE  ||
         m->actuator_gaintype[i] == mjGAIN_AFFINE  ||
         m->actuator_gaintype[i] == mjGAIN_SO3     ||
         m->actuator_gaintype[i] == mjGAIN_MUSCLE  ||
         m->actuator_gaintype[i] == mjGAIN_DCMOTOR;
}


//-------------------------- flex elasticity -------------------------------------------------------

// temporary spectral representation of one projected material Hessian in the
// signed SVD frame F = U diag(sigma) V'; only the assembled Cartesian blocks are cached
typedef struct {
  mjtNum rotation[9];       // U
  mjtNum gradient[4][3];    // V' grad(N_i)
  mjtNum stretch[9];        // projected 3x3 block coupling diagonal variations of F
  mjtNum symmetric[3];      // projected symmetric off-diagonal modes: (01, 02, 12)
  mjtNum skew[3];           // projected skew off-diagonal modes: (01, 02, 12)
} mjSnhHessian;

// project the SNH material Hessian, using existing normalized reference vertices and half sizes
static void snhProject(mjSnhHessian* hessian, const mjtNum k[24], mjtNum edgevec[6][3],
                       const mjtNum elongation[6], const mjtNum* vert0, const int vert[4],
                       const mjtNum size[3]);

// pull the projected material Hessian back to a world-space vertex-pair block
static void snhProjectedBlock(mjtNum block[9], const mjSnhHessian* hessian, int i, int j);


// cache the unscaled Cartesian stretch Hessian of a standard 2D or 3D flex
// StVK retains its tensile geometric term; SNH projects each element's material Hessian to PSD
void mj_flexHessian(const mjModel* m, mjData* d, int f) {
  if (d->flex_hessian_valid[f]) {
    return;
  }

  int va = m->flex_vertadr[f], ea = m->flex_edgeadr[f];
  mjtNum* diagonal = d->flexvert_hessian + 6*va;
  mjtNum* offdiag = d->flexedge_hessian + 9*ea;

  mju_zero(diagonal, 6*m->flex_vertnum[f]);
  mju_zero(offdiag, 9*m->flex_edgenum[f]);

  const mjtNum* k = m->flex_stiffness + m->flex_stiffnessadr[f];
  int dim = m->flex_dim[f];
  int nedge = dim == 2 ? 3 : 6;
  int stride = dim == 2 ? 21 : 24;
  int snh = dim == 3 && k[21] != 0;

  const int (*edge)[2] = mj_stretchEdges[dim-2];
  const int* elem = m->flex_elem + m->flex_elemdataadr[f];
  const int* eelem = m->flex_elemedge + m->flex_elemedgeadr[f];

  const mjtNum* xpos = d->flexvert_xpos + 3*va;
  const mjtNum* length = d->flexedge_length + ea;
  const mjtNum* rest = m->flexedge_length0 + ea;
  const mjtNum* plastic = m->flex_plasticadr[f] >= 0 ? d->plastic + m->flex_plasticadr[f] : NULL;

  for (int t=0; t < m->flex_elemnum[f]; t++) {
    const int* vert = elem + (dim+1)*t;
    const mjtNum* packed = k + stride*t;
    const mjtNum* eplastic = plastic ? plastic + nedge*t : NULL;
    mjtNum edges[6][3], metric[36], tension[6];
    mjSnhHessian projected;
    mj_stretchEdgeVectors(edges, xpos, vert, dim);
    if (snh) {
      mjtNum elongation[6];
      mj_stretchElongation(elongation, eelem + 6*t, length, rest, eplastic, 6);
      snhProject(&projected, packed, edges, elongation, m->flex_vert0 + 3*va, vert,
                 m->flex_size + 3*f);
    } else {
      mj_stretchStiffness(metric, tension, packed, eelem + nedge*t, length, rest, eplastic, nedge);
    }
    // assemble one block per edge, oriented from its first endpoint to its second
    // the opposite block is its transpose by Hessian symmetry
    for (int e=0; e < nedge; e++) {
      int i = edge[e][0], j = edge[e][1];
      int id = eelem[nedge*t+e];
      const int* endpoints = m->flex_edge + 2*(ea+id);
      if (endpoints[0] != vert[i]) {
        int swap = i;
        i = j;
        j = swap;
      }
      mjtNum block[9];
      if (snh) {
        snhProjectedBlock(block, &projected, i, j);
      } else {
        mj_stretchStiffnessBlock(block, metric, tension, edges, dim, i, j, 1);
      }
      mju_addTo(offdiag + 9*id, block, 9);
    }
  }

  // translation invariance gives H_ii = -sum_{j != i} H_ij
  // pack each symmetric diagonal block as (00, 01, 02, 11, 12, 22)
  for (int e=0; e < m->flex_edgenum[f]; e++) {
    const int* vert = m->flex_edge + 2*(ea+e);
    const mjtNum* block = offdiag + 9*e;
    int id = 0;
    for (int r=0; r < 3; r++) {
      for (int c=r; c < 3; c++) {
        diagonal[6*vert[0]+id] -= block[3*r+c];
        diagonal[6*vert[1]+id] -= block[3*c+r];
        id++;
      }
    }
  }
  d->flex_hessian_valid[f] = 1;
}


// add scale * cached Cartesian Hessian * vec to res
void mj_flexHessianMul(const mjModel* m, const mjData* d, int f, mjtNum* res,
                       const mjtNum* vec, mjtNum scale) {
  const mjtNum* diagonal = d->flexvert_hessian + 6*m->flex_vertadr[f];
  int ea = m->flex_edgeadr[f];
  const mjtNum* offdiag = d->flexedge_hessian + 9*ea;
  for (int v=0; v < m->flex_vertnum[f]; v++) {
    const mjtNum* a = diagonal + 6*v;
    const mjtNum* x = vec + 3*v;
    res[3*v]   += scale*(a[0]*x[0] + a[1]*x[1] + a[2]*x[2]);
    res[3*v+1] += scale*(a[1]*x[0] + a[3]*x[1] + a[4]*x[2]);
    res[3*v+2] += scale*(a[2]*x[0] + a[4]*x[1] + a[5]*x[2]);
  }
  for (int e=0; e < m->flex_edgenum[f]; e++) {
    const int* v = m->flex_edge + 2*(ea+e);
    const mjtNum* a = offdiag + 9*e;
    const mjtNum* x = vec + 3*v[0];
    const mjtNum* y = vec + 3*v[1];
    for (int r=0; r < 3; r++) {
      res[3*v[0]+r] += scale*(a[3*r]*y[0] + a[3*r+1]*y[1] + a[3*r+2]*y[2]);
      res[3*v[1]+r] += scale*(a[r]*x[0] + a[3+r]*x[1] + a[6+r]*x[2]);
    }
  }
}


//-------------------------- 1D rod bending --------------------------------------------------------

// smoothing of curved rod references at zero curvature, relative to the reference curvature
#define mjROD_SMOOTH ((mjtNum)1e-4)


// orthonormal axes (u, v) orthogonal to the edge (x1, x2); returns 0 if the edge is degenerate
static int rodEdgeFrame(const mjtNum x1[3], const mjtNum x2[3], mjtNum u[3], mjtNum v[3]) {
  mjtNum frame[9] = {0};
  mji_sub3(frame, x2, x1);
  if (mju_normalize3(frame) < mjMINVAL) {
    return 0;
  }
  mju_makeFrame(frame);
  mji_copy3(u, frame+3);
  mji_copy3(v, frame+6);
  return 1;
}


// curvature binormal of a rod hinge and its Jacobian, see Bergou et al., "Discrete Elastic Rods"
int mj_rodHinge(const mjtNum xa[3], const mjtNum xv[3], const mjtNum xb[3],
                mjtNum kb[3], mjtNum J[27]) {
  mjtNum e0[3], e1[3];
  mji_sub3(e0, xv, xa);
  mji_sub3(e1, xb, xv);
  mjtNum l0 = mju_norm3(e0), l1 = mju_norm3(e1);
  mjtNum denom = l0*l1 + mju_dot3(e0, e1);
  if (l0 < mjMINVAL || l1 < mjMINVAL || denom < mjMINVAL) {
    return 0;
  }
  mjtNum cross[3];
  mji_cross(cross, e0, e1);
  mji_scl3(kb, cross, 2/denom);
  if (!J) {
    return 1;
  }

  // gradients of denom with respect to e0 and e1
  mjtNum g0[3], g1[3];
  mji_scl3(g0, e0, l1/l0);
  mji_addTo3(g0, e1);
  mji_scl3(g1, e1, l0/l1);
  mji_addTo3(g1, e0);

  // A0 = d(kb)/d(e0) = (-2[e1]x - kb g0') / denom,  A1 = d(kb)/d(e1) = (2[e0]x - kb g1') / denom
  mjtNum A0[9] = {0, e1[2], -e1[1], -e1[2], 0, e1[0], e1[1], -e1[0], 0};
  mjtNum A1[9] = {0, -e0[2], e0[1], e0[2], 0, -e0[0], -e0[1], e0[0], 0};
  for (int r=0; r < 3; r++) {
    for (int c=0; c < 3; c++) {
      A0[3*r+c] = (2*A0[3*r+c] - kb[r]*g0[c]) / denom;
      A1[3*r+c] = (2*A1[3*r+c] - kb[r]*g1[c]) / denom;
    }
  }

  // J = [-A0, A0 - A1, A1]
  for (int r=0; r < 3; r++) {
    for (int c=0; c < 3; c++) {
      J[9*r+c]   = -A0[3*r+c];
      J[9*r+3+c] = A0[3*r+c] - A1[3*r+c];
      J[9*r+6+c] = A1[3*r+c];
    }
  }
  return 1;
}


// gradient, Hessian, and damping weights of a single rod hinge in kb space
void mj_rodHingeWeights(mjtNum g[3], mjtNum Wk[9], mjtNum Wd[9],
                        const mjtNum kb[3], mjtNum k, mjtNum kb0) {
  mjtNum r = mju_norm3(kb);
  mjtNum n[3] = {0, 0, 0};
  if (r >= mjMINVAL) {
    mji_scl3(n, kb, 1/r);
  }

  // straight reference: energy k |kb|^2, gradient 2k kb, Hessian 2k I
  if (kb0 == 0) {
    if (g) mji_scl3(g, kb, 2*k);
    for (int i=0; i < 3; i++) {
      for (int j=0; j < 3; j++) {
        mjtNum w = (i == j) ? 2*k : 0;
        if (Wk) Wk[3*i+j] = w;
        if (Wd) Wd[3*i+j] = r < mjMINVAL ? w : 2*k*n[i]*n[j];
      }
    }
    return;
  }

  // curved reference: energy k (rho - gam)^2 kb0^2, with rho = sqrt((|kb|/kb0)^2 + e^2) the
  // smoothed relative curvature and gam = sqrt(1 + e^2) its reference value
  mjtNum e2 = mjROD_SMOOTH*mjROD_SMOOTH;
  mjtNum gam = mju_sqrt(1 + e2);
  mjtNum c = r/kb0;
  mjtNum rho2 = c*c + e2;
  mjtNum rho = mju_sqrt(rho2);
  mjtNum s = 1 - gam/rho;
  if (g) mji_scl3(g, kb, 2*k*s);

  // Hessian: tangential 2k s and radial 2k (1 - gam e^2 / rho^3), clamped to PSD
  mjtNum tang = mju_max(0, s);
  mjtNum rad = mju_max(0, 1 - gam*e2/(rho*rho2));
  for (int i=0; i < 3; i++) {
    for (int j=0; j < 3; j++) {
      mjtNum delta = (i == j) ? 1 : 0;
      if (Wk) {
        Wk[3*i+j] = r < mjMINVAL ? 0 : 2*k*(tang*(delta - n[i]*n[j]) + rad*n[i]*n[j]);
      }
      if (Wd) {
        Wd[3*i+j] = r < mjMINVAL ? 2*k*delta : 2*k*n[i]*n[j];
      }
    }
  }
}


// curvature binormals of a rod hinge pair in a frame orthogonal to their shared edge
int mj_rodPair(const mjtNum x0[3], const mjtNum x1[3], const mjtNum x2[3], const mjtNum x3[3],
               mjtNum om[4], mjtNum J[48]) {
  mjtNum kba[3], kbb[3], Ja[27], Jb[27], u[3], v[3];
  if (!mj_rodHinge(x0, x1, x2, kba, J ? Ja : NULL) ||
      !mj_rodHinge(x1, x2, x3, kbb, J ? Jb : NULL) ||
      !rodEdgeFrame(x1, x2, u, v)) {
    return 0;
  }

  om[0] = mju_dot3(kba, u);
  om[1] = mju_dot3(kba, v);
  om[2] = mju_dot3(kbb, u);
  om[3] = mju_dot3(kbb, v);
  if (!J) {
    return 1;
  }

  // rows of J: u'Ja and v'Ja on columns x0..x2, u'Jb and v'Jb on columns x1..x3
  mju_zero(J, 48);
  for (int k=0; k < 9; k++) {
    J[k]      = u[0]*Ja[k] + u[1]*Ja[9+k] + u[2]*Ja[18+k];
    J[12+k]   = v[0]*Ja[k] + v[1]*Ja[9+k] + v[2]*Ja[18+k];
    J[27+k]   = u[0]*Jb[k] + u[1]*Jb[9+k] + u[2]*Jb[18+k];
    J[39+k]   = v[0]*Jb[k] + v[1]*Jb[9+k] + v[2]*Jb[18+k];
  }
  return 1;
}


// gradient, Hessian, and damping weights of a rod hinge pair in om space
void mj_rodPairWeights(mjtNum g[4], mjtNum Wk[16], mjtNum Wd[16],
                       const mjtNum om[4], mjtNum w0, mjtNum w1, const mjtNum om0[4]) {
  mjtNum w[4] = {w0, w0, w1, w1};
  mjtNum wmax = mju_max(w0, w1);
  if (wmax <= 0) {
    if (g) mju_zero(g, 4);
    if (Wk) mju_zero(Wk, 16);
    if (Wd) mju_zero(Wd, 16);
    return;
  }

  // weights relative to wmax
  mjtNum u0 = w0/wmax, u1 = w1/wmax;

  // damping: 2 diag(w), without the rate of rotation about the shared edge (direction t)
  if (Wd) {
    mjtNum t[4] = {-om[1], om[0], -om[3], om[2]};
    mjtNum ut[4] = {u0*t[0], u0*t[1], u1*t[2], u1*t[3]};
    mjtNum tnorm = mju_sqrt(t[0]*ut[0] + t[1]*ut[1] + t[2]*ut[2] + t[3]*ut[3]);
    if (tnorm >= mjMINVAL) {
      for (int i=0; i < 4; i++) ut[i] /= tnorm;
    }
    mjtNum scl = tnorm < mjMINVAL ? 0 : 2*wmax;
    for (int i=0; i < 4; i++) {
      for (int j=0; j < 4; j++) {
        Wd[4*i+j] = (i == j ? 2*w[i] : 0) - scl*ut[i]*ut[j];
      }
    }
  }

  // straight reference: energy sum_i w_i |om_i|^2
  mjtNum r = u0*(om0[0]*om0[0] + om0[1]*om0[1]) + u1*(om0[2]*om0[2] + om0[3]*om0[3]);
  if (r <= 0) {
    if (g) {
      for (int i=0; i < 4; i++) g[i] = 2*w[i]*om[i];
    }
    if (Wk) {
      for (int i=0; i < 4; i++) {
        for (int j=0; j < 4; j++) Wk[4*i+j] = (i == j) ? 2*w[i] : 0;
      }
    }
    return;
  }

  // weighted reference om0 and om0 rotated by 90 degrees about the edge
  mjtNum c[4] = {u0*om0[0], u0*om0[1], u1*om0[2], u1*om0[3]};
  mjtNum s[4] = {-u0*om0[1], u0*om0[0], -u1*om0[3], u1*om0[2]};

  // cosine and sine terms of the rotation aligning om0 with om, relative to the reference
  mjtNum cs = 0, sn = 0;
  for (int i=0; i < 4; i++) {
    cs += c[i]*om[i];
    sn += s[i]*om[i];
  }
  cs /= r;
  sn /= r;

  // energy sum_i w_i |om_i|^2 - 2 gam wmax r rho, up to a constant, where rho = sqrt(cs^2 + sn^2 +
  // e^2) is the smoothed alignment after the optimal rotation and gam = sqrt(1 + e^2) its reference
  // value
  mjtNum e2 = mjROD_SMOOTH*mjROD_SMOOTH;
  mjtNum gam = mju_sqrt(1 + e2);
  mjtNum rho2 = cs*cs + sn*sn + e2;
  mjtNum rho = mju_sqrt(rho2);
  if (g) {
    mjtNum scl = 2*gam*wmax/rho;
    for (int i=0; i < 4; i++) {
      g[i] = 2*w[i]*om[i] - scl*(cs*c[i] + sn*s[i]);
    }
  }

  // Hessian: 2 diag(w) - 2 gam wmax / (r rho) (q q' + e^2 / rho^2 (c c' + s s')), with
  // q = (cs s - sn c) / rho and gam / rho clamped to 1 so that it is PSD
  if (Wk) {
    mjtNum q[4];
    for (int i=0; i < 4; i++) {
      q[i] = (cs*s[i] - sn*c[i]) / rho;
    }
    mjtNum scl = 2*wmax*mju_min(1, gam/rho) / r;
    mjtNum ratio = e2/rho2;
    for (int i=0; i < 4; i++) {
      for (int j=0; j < 4; j++) {
        mjtNum diag = (i == j) ? 2*w[i] : 0;
        Wk[4*i+j] = diag - scl*(q[i]*q[j] + ratio*(c[i]*c[j] + s[i]*s[j]));
      }
    }
  }
}


//-------------------------- Stable Neo-Hookean tetrahedra -----------------------------------------

// noniterative symmetric eigensystem: an isolated cubic root, followed by a 2x2 solve
// selecting the isolated root avoids an ill-conditioned cross product at a repeated pair
// see Eberly, A Robust Eigensolver for 3x3 Symmetric Matrices, section 5
static void snhEigen3(mjtNum value[3], mjtNum Q[9], const mjtNum A[9]) {
  mjtNum scale = 0, D[9];
  for (int i=0; i < 9; i++) {
    scale = mju_max(scale, mju_abs(A[i]));
  }
  mju_zero(Q, 9);
  Q[0] = Q[4] = Q[8] = 1;
  if (!scale) {
    mju_zero3(value);
    return;
  }
  for (int i=0; i < 9; i++) {
    D[i] = A[i]/scale;
  }
  if (!D[1] && !D[2] && !D[5]) {
    for (int i=0; i < 3; i++) {
      value[i] = A[4*i];
    }
    return;
  }

  // shift by trace mean to reduce the cubic to depressed form: det(B - lambda I) = 0
  mjtNum mean = (D[0]+D[4]+D[8])/3;
  mjtNum B[9];
  mju_copy(B, D, 9);
  for (int i=0; i < 3; i++) {
    B[4*i] -= mean;
  }
  mjtNum bscale = 0;
  for (int i=0; i < 9; i++) {
    bscale = mju_max(bscale, mju_abs(B[i]));
  }
  for (int i=0; i < 9; i++) {
    B[i] /= bscale;
  }
  mjtNum p = mju_sqrt((B[0]*B[0]+B[4]*B[4]+B[8]*B[8]
                       +2*(B[1]*B[1]+B[2]*B[2]+B[5]*B[5]))/6);
  for (int i=0; i < 9; i++) {
    B[i] /= p;
  }
  mjtNum determinant = B[0]*(B[4]*B[8]-B[5]*B[5])
                       - B[1]*(B[1]*B[8]-B[2]*B[5])
                       + B[2]*(B[1]*B[5]-B[2]*B[4]);
  mjtNum r = .5*determinant;

  // solve depressed cubic via trigonometric formula; choose isolated root (extreme eigenvalue)
  mjtNum root = 2*mju_cos(mju_acos(mju_min(1, mju_abs(r)))/3);
  if (r < 0) {
    root = -root;
  }
  for (int i=0; i < 3; i++) {
    B[4*i] -= root;
  }

  // rows of (B - root*I) are perpendicular to the isolated eigenvector;
  // the longest row cross product gives the most numerically stable eigenvector
  mjtNum cross[3][3], norm[3];
  mju_cross(cross[0], B, B+3);
  mju_cross(cross[1], B, B+6);
  mju_cross(cross[2], B+3, B+6);
  int best = 0;
  for (int i=0; i < 3; i++) {
    norm[i] = mju_dot3(cross[i], cross[i]);
    if (norm[i] > norm[best]) {
      best = i;
    }
  }
  mjtNum q[3], u[3], v[3];
  mju_scl3(q, cross[best], 1/mju_sqrt(norm[best]));

  // construct an orthonormal basis {u, v} for the complementary 2D plane
  if (mju_abs(q[0]) > mju_abs(q[1])) {
    mjtNum inv = 1/mju_sqrt(q[0]*q[0]+q[2]*q[2]);
    u[0] = -q[2]*inv;
    u[1] = 0;
    u[2] = q[0]*inv;
  } else {
    mjtNum inv = 1/mju_sqrt(q[1]*q[1]+q[2]*q[2]);
    u[0] = 0;
    u[1] = q[2]*inv;
    u[2] = -q[1]*inv;
  }
  mju_cross(v, q, u);
  mjtNum Du[3], Dv[3], Dq[3];
  mji_mulMatVec3(Du, D, u);
  mji_mulMatVec3(Dv, D, v);
  mji_mulMatVec3(Dq, D, q);

  // diagonalize the remaining 2x2 symmetric system via a 2D Jacobi rotation
  mjtNum a = mju_dot3(u, Du), b = mju_dot3(u, Dv), c = mju_dot3(v, Dv);
  mjtNum cosine = 1, sine = 0, t = 0;
  if (b) {
    mjtNum delta = .5*(c-a);
    mjtNum wscale = mju_max(mju_abs(delta), mju_abs(b));
    mjtNum x = delta/wscale, y = b/wscale;
    t = (x >= 0 ? y : -y)/(mju_abs(x)+mju_sqrt(x*x+y*y));
    cosine = 1/mju_sqrt(1+t*t);
    sine = t*cosine;
  }
  value[0] = mju_dot3(q, Dq)*scale;
  value[1] = (a-t*b)*scale;
  value[2] = (c+t*b)*scale;
  for (int i=0; i < 3; i++) {
    Q[3*i] = q[i];
    Q[3*i+1] = cosine*u[i]-sine*v[i];
    Q[3*i+2] = sine*u[i]+cosine*v[i];
  }
}


// signed SVD F = U diag(sigma) V', with proper U,V even at collapse or inversion
static void snhSVD(mjtNum U[9], mjtNum sigma[3], mjtNum V[9], const mjtNum F[9]) {
  mjtNum scale = 0;
  for (int i=0; i < 9; i++) {
    scale = mju_max(scale, mju_abs(F[i]));
  }

  // one-sided Jacobi rotations on F diagonalize F'F without squaring condition number
  mjtNum A[9];
  for (int i=0; i < 9; i++) {
    A[i] = scale ? F[i]/scale : 0;
  }
  mju_zero(V, 9);
  V[0] = V[4] = V[8] = 1;
#ifdef mjUSESINGLE
  const mjtNum reltol = 2e-6f;
#else
  const mjtNum reltol = 4e-15;
#endif
  for (int sweep=0; sweep < 24; sweep++) {
    int converged = 1;
    for (int p=0; p < 2; p++) {
      for (int q=p+1; q < 3; q++) {
        mjtNum pp = 0, qq = 0, pq = 0;
        for (int r=0; r < 3; r++) {
          pp += A[3*r+p]*A[3*r+p];
          qq += A[3*r+q]*A[3*r+q];
          pq += A[3*r+p]*A[3*r+q];
        }
        if (mju_abs(pq) <= reltol*mju_sqrt(pp)*mju_sqrt(qq)) {
          continue;
        }
        converged = 0;
        mjtNum delta = .5*(qq-pp);
        mjtNum t = (delta >= 0 ? pq : -pq)/(mju_abs(delta)+mju_sqrt(delta*delta+pq*pq));
        mjtNum c = 1/mju_sqrt(1+t*t), s = t*c;
        for (int r=0; r < 3; r++) {
          mjtNum x = A[3*r+p], y = A[3*r+q];
          A[3*r+p] = c*x-s*y;
          A[3*r+q] = s*x+c*y;
          x = V[3*r+p];
          y = V[3*r+q];
          V[3*r+p] = c*x-s*y;
          V[3*r+q] = s*x+c*y;
        }
      }
    }
    if (converged) {
      break;
    }
  }
  for (int i=0; i < 3; i++) {
    sigma[i] = A[i]*A[i]+A[3+i]*A[3+i]+A[6+i]*A[6+i];
  }

  // sort singular values in descending order
  for (int i=0; i < 2; i++) {
    for (int j=i+1; j < 3; j++) {
      if (sigma[j] > sigma[i]) {
        mjtNum tmp = sigma[i];
        sigma[i] = sigma[j];
        sigma[j] = tmp;
        for (int r=0; r < 3; r++) {
          tmp = A[3*r+i];
          A[3*r+i] = A[3*r+j];
          A[3*r+j] = tmp;
          tmp = V[3*r+i];
          V[3*r+i] = V[3*r+j];
          V[3*r+j] = tmp;
        }
      }
    }
  }

  // ensure V is a proper rotation (det(V) = +1); flip third column if reflected
  mjtNum cross[3], a[3] = {V[0], V[3], V[6]}, b[3] = {V[1], V[4], V[7]};
  mju_cross(cross, a, b);
  if (cross[0]*V[2]+cross[1]*V[5]+cross[2]*V[8] < 0) {
    for (int r=0; r < 3; r++) {
      V[3*r+2] = -V[3*r+2];
      A[3*r+2] = -A[3*r+2];
    }
  }

  // normalize columns of A to construct left singular vectors U
  mjtNum u[3][3];
  for (int i=0; i < 3; i++) {
    for (int x=0; x < 3; x++) {
      u[i][x] = A[3*x+i];
    }
  }
  sigma[0] = mju_norm3(u[0]);
  if (!sigma[0]) {
    mju_zero(U, 9);
    U[0] = U[4] = U[8] = 1;
    mju_zero3(sigma);
    return;
  }
  mju_scl3(u[0], u[0], 1/sigma[0]);
  mju_addToScl3(u[1], u[0], -mju_dot3(u[0], u[1]));
  mjtNum norm = mju_norm3(u[1]);
#ifdef mjUSESINGLE
  const mjtNum tol = 1e-6f;
#else
  const mjtNum tol = 1e-12;
#endif
  if (norm > tol*sigma[0]) {
    mju_scl3(u[1], u[1], 1/norm);
  } else {
    // rank one: choose any orthonormal completion of the nonzero column
    int axis = 0;
    for (int x=1; x < 3; x++) {
      if (mju_abs(u[0][x]) < mju_abs(u[0][axis])) {
        axis = x;
      }
    }
    for (int x=0; x < 3; x++) {
      u[1][x] = (x == axis)-u[0][axis]*u[0][x];
    }
    mju_scl3(u[1], u[1], 1/mju_norm3(u[1]));
  }
  mju_cross(u[2], u[0], u[1]);
  sigma[1] = sigma[2] = 0;
  for (int x=0; x < 3; x++) {
    sigma[1] += u[1][x]*A[3*x+1];
    sigma[2] += u[2][x]*A[3*x+2];
    for (int i=0; i < 3; i++) {
      U[3*x+i] = u[i][x];
    }
  }
  mju_scl3(sigma, sigma, scale);
}


// conservative Cholesky check: uncertain pivots fall back to eigendecomposition
static int snhPositive3(const mjtNum A[9]) {
  mjtNum scale = 0;
  for (int i=0; i < 9; i++) {
    scale = mju_max(scale, mju_abs(A[i]));
  }
#ifdef mjUSESINGLE
  mjtNum tol = 1e-5f*scale;
#else
  mjtNum tol = 1e-12*scale;
#endif
  if (A[0] <= tol) {
    return 0;
  }
  mjtNum pivot = A[4]-A[1]*(A[1]/A[0]);
  if (pivot <= tol) {
    return 0;
  }
  mjtNum cross = A[5]-A[1]*(A[2]/A[0]);
  return A[8]-A[2]*(A[2]/A[0])-cross*(cross/pivot) > tol;
}


// add twice the Hessian of gamma*P(s) to the edge metric; its gradient is evaluated
// by mj_snhCubic, shared with the force calculation
static void snhCubicMetric(mjtNum metric[36], const mjtNum s[6], mjtNum gamma) {
  mjtNum a = s[0], b = s[2], c = s[4];
  mjtNum d = .5*(s[0]+s[2]-s[1]);
  mjtNum e = .5*(s[0]+s[4]-s[5]);
  mjtNum f = .5*(s[2]+s[4]-s[3]);

  // columns of the constant map s -> (a,b,c,d,e,f)
  static const mjtNum basis[6][6] = {
    {1, 0, 0, .5, .5, 0}, {0, 0, 0, -.5, 0, 0},
    {0, 1, 0, .5, 0, .5}, {0, 0, 0, 0, 0, -.5},
    {0, 0, 1, 0, .5, .5}, {0, 0, 0, 0, -.5, 0}
  };
  for (int j=0; j < 6; j++) {
    const mjtNum* h = basis[j];
    mjtNum dc[6] = {
      h[1]*c+b*h[2]-2*f*h[5], h[0]*c+a*h[2]-2*e*h[4],
      h[0]*b+a*h[1]-2*d*h[3], h[4]*f+e*h[5]-h[2]*d-c*h[3],
      h[3]*f+d*h[5]-h[1]*e-b*h[4], h[3]*e+d*h[4]-h[0]*f-a*h[5]
    };
    mjtNum dg[6] = {dc[0]+dc[3]+dc[4], -dc[3], dc[1]+dc[3]+dc[5],
                    -dc[5], dc[2]+dc[4]+dc[5], -dc[4]};
    for (int i=0; i <= j; i++) {
      mjtNum value = 2*gamma*dg[i];
      metric[6*i+j] += value;
      if (i != j) metric[6*j+i] += value;
    }
  }
}


// project d2E/dF2, not the vertex Hessian: PSD projection and pullback do not commute
// the signed SVD supplies the principal stretches and frame; contract the edge
// energy's derivatives into one 3x3 stretch block and six scalar off-diagonal modes
// (Smith et al., Stable Neo-Hookean Flesh Simulation, Sec. 4)
static void snhProject(mjSnhHessian* hessian, const mjtNum k[24], mjtNum edgevec[6][3],
                       const mjtNum elongation[6], const mjtNum* vert0, const int vert[4],
                       const mjtNum size[3]) {
  // recover reference shape gradients dN/dX from normalized rest coordinates
  mjtNum rest[3][3], gradient[3][3];
  for (int v=0; v < 3; v++) {
    for (int x=0; x < 3; x++) {
      rest[v][x] = 2*size[x]*(vert0[3*vert[v+1]+x]-vert0[3*vert[0]+x]);
    }
  }
  mju_cross(gradient[0], rest[1], rest[2]);
  mju_cross(gradient[1], rest[2], rest[0]);
  mju_cross(gradient[2], rest[0], rest[1]);
  for (int v=0; v < 3; v++) {
    mju_scl3(gradient[v], gradient[v], k[23]);
  }

  // compute deformation gradient F = sum_e x_e * dN_e' and its signed SVD: F = U * diag(sigma) * V'
  mjtNum F[9], sigma[3], V[9];
  for (int x=0; x < 3; x++) {
    for (int y=0; y < 3; y++) {
      F[3*x+y] = -edgevec[0][x]*gradient[0][y]+edgevec[2][x]*gradient[1][y]
                -edgevec[4][x]*gradient[2][y];
    }
  }
  snhSVD(hessian->rotation, sigma, V, F);
  mju_zero3(hessian->gradient[0]);
  for (int v=1; v < 4; v++) {
    mji_mulMatTVec3(hessian->gradient[v], V, gradient[v-1]);
    mju_subFrom3(hessian->gradient[0], hessian->gradient[v]);
  }

  // compute 1D edge tension (2*dU/ds) and metric (2*d2U/ds2) for quadratic and cubic energy
  mjtNum metric[36], tension[6];
  mj_stretchElasticity(metric, tension, k, elongation, 6);
  mj_snhCubic(tension, elongation, k[21]);
  snhCubicMetric(metric, elongation, k[21]);

  // reference edges r_e in the principal frame; a variation D in that frame
  // changes s_e by 2*(diag(sigma)*r_e)'*D*r_e; its second variation is 2*|D*r_e|^2
  mjtNum vertex[4][3] = {{0}}, reference[6][3], square[3][6], product[3][6], geo[3];
  for (int v=1; v < 4; v++) {
    mji_mulMatTVec3(vertex[v], V, rest[v-1]);
  }
  for (int e=0; e < 6; e++) {
    const int* endpoints = mj_stretchEdges[1][e];
    mju_sub3(reference[e], vertex[endpoints[0]], vertex[endpoints[1]]);
    for (int i=0; i < 3; i++) {
      square[i][e] = reference[e][i]*reference[e][i];
    }
  }
  for (int i=0; i < 3; i++) {
    mj_stretchTension(product[i], metric, square[i], 6);
    geo[i] = mju_dot(tension, square[i], 6);
  }

  // assemble 3x3 stretch block coupling diagonal variations of F, adding volume penalty
  mjtNum J = sigma[0]*sigma[1]*sigma[2];
  mjtNum volume_stiffness = 2*k[22], pressure = volume_stiffness*(J-1);
  mjtNum cof[3] = {sigma[1]*sigma[2], sigma[0]*sigma[2], sigma[0]*sigma[1]};
  mjtNum A[9];
  for (int i=0; i < 3; i++) {
    for (int j=i; j < 3; j++) {
      mjtNum value = 2*sigma[i]*sigma[j]*mju_dot(square[i], product[j], 6)
                     + volume_stiffness*cof[i]*cof[j]
                     + (i == j ? geo[i] : pressure*sigma[3-i-j]);
      A[3*i+j] = A[3*j+i] = value;
    }
  }

  // project 3x3 stretch block to positive semi-definite (PSD) by clamping negative eigenvalues
  if (snhPositive3(A)) {
    mju_copy(hessian->stretch, A, 9);
  } else {
    mjtNum value[3], Q[9];
    snhEigen3(value, Q, A);
    mju_zero(hessian->stretch, 9);
    for (int e=0; e < 3; e++) {
      mjtNum weight = mju_max(0, value[e]);
      for (int i=0; i < 3; i++) {
        for (int j=0; j < 3; j++) {
          hessian->stretch[3*i+j] += weight*Q[3*i+e]*Q[3*j+e];
        }
      }
    }
  }

  // compute and clamp symmetric (shear) and skew (flip) off-diagonal modes to non-negative
  int mode = 0;
  for (int i=0; i < 3; i++) {
    for (int j=i+1; j < 3; j++) {
      mjtNum direction[6], response[6];
      for (int e=0; e < 6; e++) {
        direction[e] = reference[e][i]*reference[e][j];
      }
      mj_stretchTension(response, metric, direction, 6);
      mjtNum material = mju_dot(direction, response, 6);
      mjtNum geometric = .5*(geo[i]+geo[j]);
      mjtNum sum = sigma[i]+sigma[j], diff = sigma[i]-sigma[j];
      mjtNum cross = pressure*sigma[3-i-j];
      hessian->symmetric[mode] = mju_max(0, sum*sum*material+geometric-cross);
      hessian->skew[mode] = mju_max(0, diff*diff*material+geometric+cross);
      mode++;
    }
  }
}


// contract in the SVD frame, then rotate the 3x3 block back to Cartesian coordinates
static void snhProjectedBlock(mjtNum block[9], const mjSnhHessian* hessian, int i, int j) {
  const mjtNum* a = hessian->gradient[i];
  const mjtNum* b = hessian->gradient[j];
  mjtNum local[9], tmp[9];
  for (int r=0; r < 3; r++) {
    for (int c=0; c < 3; c++) {
      local[3*r+c] = hessian->stretch[3*r+c]*a[r]*b[c];
    }
  }
  int mode = 0;
  for (int r=0; r < 3; r++) {
    for (int c=r+1; c < 3; c++) {
      mjtNum sum = .5*(hessian->symmetric[mode]+hessian->skew[mode]);
      mjtNum diff = .5*(hessian->symmetric[mode]-hessian->skew[mode]);
      mode++;
      local[4*r] += sum*a[c]*b[c];
      local[4*c] += sum*a[r]*b[r];
      local[3*r+c] += diff*a[c]*b[r];
      local[3*c+r] += diff*a[r]*b[c];
    }
  }
  mji_mulMatMat3(tmp, hessian->rotation, local);
  mju_mulMatMatT3(block, tmp, hessian->rotation);
}


// cubic Gram determinant P(s) = a*b*c + 2*d*e*f - a*f*f - b*e*e - c*d*d,
// where D(s) = [[a,d,e], [d,b,f], [e,f,c]] is the Gram difference at vertex 0.
// add its gradient to the same edge tension used for the quadratic energy
void mj_snhCubic(mjtNum tension[6], const mjtNum s[6], mjtNum gamma) {
  mjtNum a = s[0], b = s[2], c = s[4];
  mjtNum d = 0.5 * (s[0] + s[2] - s[1]);
  mjtNum e = 0.5 * (s[0] + s[4] - s[5]);
  mjtNum f = 0.5 * (s[2] + s[4] - s[3]);
  mjtNum cof[6] = {b*c-f*f, a*c-e*e, a*b-d*d, e*f-c*d, d*f-b*e, d*e-a*f};
  mjtNum g[6] = {cof[0]+cof[3]+cof[4], -cof[3], cof[1]+cof[3]+cof[5],
                 -cof[5], cof[2]+cof[4]+cof[5], -cof[4]};
  mju_addToScl(tension, g, 2*gamma, 6);
}
