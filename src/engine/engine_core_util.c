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
#include <string.h>

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


// world position of (global) lattice node n of interpolated flex f
void mj_flexNodePos(const mjModel* m, const mjData* d, int f, int n, mjtNum pos[3]) {
  int b = m->flex_nodebodyid[n];
  const mjtNum* off = m->flex_node + 3*n;
  if (m->flex_centered[f] || (off[0] == 0 && off[1] == 0 && off[2] == 0)) {
    mju_copy3(pos, d->xpos + 3*b);
  } else {
    mju_mulMatVec3(pos, d->xmat + 9*b, off);
    mju_addTo3(pos, d->xpos + 3*b);
  }
}


//---------------------------- flex interpolation -------------------------------------------------
// The vertices of an interpolated flex (flex_interp != 0) are a fixed linear combination of its
// lattice nodes: the Lagrange basis of order |flex_interp| of the vertex's cell, where for shell-mode
// flexes (flex_interp < 0) the interior lattice nodes are in turn the Coons TFI of the boundary
// nodes. Vertex positions, gather/scatter and contact weights all evaluate this map through the
// functions in this section (flexVertBasis and flexTFINode), so they agree exactly.

// lattice node counts of interpolated flex f, return interpolation order
static int flexLattice(const mjModel* m, int f, int dims[3]) {
  int interp = m->flex_interp[f];
  int order = interp < 0 ? -interp : interp;
  for (int i=0; i < 3; i++) {
    dims[i] = m->flex_cellnum[3*f+i]*order + 1;
  }
  return order;
}


// Coons transfinite interpolation (TFI) of interior lattice node (i,j,k) of an nx*ny*nz lattice from
// the boundary nodes on the lines through it: 6 face, 12 edge and 8 corner terms
//   idx: flat node indices (i*ny*nz + j*nz + k) of the 26 boundary nodes, w: their weights
//   returns the number of weights (26)
static int flexTFINode(int nx, int ny, int nz, int i, int j, int k, int idx[26], mjtNum w[26]) {
  mjtNum s = (mjtNum)i / (nx-1);
  mjtNum t = (mjtNum)j / (ny-1);
  mjtNum u = (mjtNum)k / (nz-1);
  int I[2] = {0, nx-1}, J[2] = {0, ny-1}, K[2] = {0, nz-1};
  mjtNum ws[2] = {1-s, s}, wt[2] = {1-t, t}, wu[2] = {1-u, u};
  int n = 0;

  // face terms: +1
  for (int a=0; a < 2; a++) {
    idx[n] = I[a]*ny*nz + j*nz + k;  w[n++] = ws[a];
    idx[n] = i*ny*nz + J[a]*nz + k;  w[n++] = wt[a];
    idx[n] = i*ny*nz + j*nz + K[a];  w[n++] = wu[a];
  }

  // edge terms: -1
  for (int a=0; a < 2; a++) {
    for (int b=0; b < 2; b++) {
      idx[n] = i*ny*nz + J[a]*nz + K[b];  w[n++] = -wt[a]*wu[b];
      idx[n] = I[a]*ny*nz + j*nz + K[b];  w[n++] = -ws[a]*wu[b];
      idx[n] = I[a]*ny*nz + J[b]*nz + k;  w[n++] = -ws[a]*wt[b];
    }
  }

  // corner terms: +1
  for (int a=0; a < 2; a++) {
    for (int b=0; b < 2; b++) {
      for (int c=0; c < 2; c++) {
        idx[n] = I[a]*ny*nz + J[b]*nz + K[c];  w[n++] = ws[a]*wt[b]*wu[c];
      }
    }
  }
  return n;
}


// Lagrange basis of a point at [0,1]^3 reference coordinates coord of interpolated flex f in its
// cell (or precomputed cell >= 0), return number of nodes
static int flexPointBasis(const mjModel* m, int f, const mjtNum coord[3], int cell,
                          int nodeindices[27], mjtNum basis[27]) {
  int interp = m->flex_interp[f];
  int order = interp < 0 ? -interp : interp;
  if (order < 1 || order > 2) {
    mjERROR("unsupported interpolation order %d of flex %d", order, f);
  }
  mjtNum local[3];
  int npc = mju_cellLookup(coord, m->flex_cellnum + 3*f, order, cell, local, nodeindices);
  mju_evalBasisArray(basis, local, order);
  return npc;
}


// complete a lattice node vector in place: interior nodes of shell-mode flexes follow the boundary
static void flexInterpNodes(const mjModel* m, int f, mjtNum* nodevec) {
  if (m->flex_interp[f] >= 0) {
    return;
  }
  int dims[3], idx[26];
  mjtNum w[26];
  flexLattice(m, f, dims);
  int nx = dims[0], ny = dims[1], nz = dims[2];
  for (int i=1; i < nx-1; i++) {
    for (int j=1; j < ny-1; j++) {
      for (int k=1; k < nz-1; k++) {
        // interior nodes only depend on boundary nodes, so the update can be done in place
        int n = flexTFINode(nx, ny, nz, i, j, k, idx, w);
        mjtNum* res = nodevec + 3*(i*ny*nz + j*nz + k);
        mju_zero3(res);
        for (int q=0; q < n; q++) {
          mju_addToScl3(res, nodevec + 3*idx[q], w[q]);
        }
      }
    }
  }
}


// vertex vectors of interpolated flex f from a completed lattice node vector
void mj_flexInterpVert(const mjModel* m, int f, const mjtNum* nodevec, mjtNum* vertvec) {
  int vadr = m->flex_vertadr[f];
  for (int v=0; v < m->flex_vertnum[f]; v++) {
    int g = vadr + v;
    int idx[27];
    mjtNum basis[27];
    int npc = flexPointBasis(m, f, m->flex_vert0 + 3*g, m->flex_vertcell[g], idx, basis);
    mjtNum* res = vertvec + 3*v;
    mju_zero3(res);
    for (int j=0; j < npc; j++) {
      mju_addToScl3(res, nodevec + 3*idx[j], basis[j]);
    }
  }
}


// transpose of flexInterpNodes followed by mj_flexInterpVert: accumulate vertex vectors onto the
// lattice nodes, then fold interior nodes of shell-mode flexes onto the boundary
static void flexInterpVertT(const mjModel* m, int f, const mjtNum* vertvec, mjtNum* nodevec) {
  int vadr = m->flex_vertadr[f];
  for (int v=0; v < m->flex_vertnum[f]; v++) {
    const mjtNum* x = vertvec + 3*v;
    if (!x[0] && !x[1] && !x[2]) {
      continue;
    }
    int g = vadr + v;
    int idx[27];
    mjtNum basis[27];
    int npc = flexPointBasis(m, f, m->flex_vert0 + 3*g, m->flex_vertcell[g], idx, basis);
    for (int j=0; j < npc; j++) {
      mju_addToScl3(nodevec + 3*idx[j], x, basis[j]);
    }
  }
  if (m->flex_interp[f] >= 0) {
    return;
  }
  int dims[3], idx[26];
  mjtNum w[26];
  flexLattice(m, f, dims);
  int nx = dims[0], ny = dims[1], nz = dims[2];
  for (int i=1; i < nx-1; i++) {
    for (int j=1; j < ny-1; j++) {
      for (int k=1; k < nz-1; k++) {
        mjtNum* fi = nodevec + 3*(i*ny*nz + j*nz + k);
        if (!fi[0] && !fi[1] && !fi[2]) {
          continue;
        }
        mjtNum val[3] = {fi[0], fi[1], fi[2]};
        mju_zero3(fi);
        int n = flexTFINode(nx, ny, nz, i, j, k, idx, w);
        for (int q=0; q < n; q++) {
          mju_addToScl3(nodevec + 3*idx[q], val, w[q]);
        }
      }
    }
  }
}


// append (k, w) to (key, weight) or add w to existing key; return updated count
static int flexAddWeight(int* key, mjtNum* weight, int n, int cap, int k, mjtNum w) {
  for (int i=0; i < n; i++) {
    if (key[i] == k) {
      weight[i] += w;
      return n;
    }
  }
  if (n >= cap) {
    mjERROR("too many interpolation weights (max %d)", cap);
  }
  key[n] = k;
  weight[n] = w;
  return n + 1;
}


// lattice node weights of a point at [0,1]^3 reference coordinates coord of interpolated flex f,
// consistent with mj_flexInterpVert; node are local lattice node indices (only boundary nodes for
// shell-mode flexes)
//   returns the number of weights, at most mjMAXFLEXINTERPW
int mj_flexInterpWeights(const mjModel* m, int f, const mjtNum coord[3],
                         int* node, mjtNum* weight) {
  int cnode[27];
  mjtNum cweight[27];
  int npc = flexPointBasis(m, f, coord, -1, cnode, cweight);

  // volume mode: the cell nodes are distinct degrees of freedom
  if (m->flex_interp[f] > 0) {
    int nn = 0;
    for (int j=0; j < npc; j++) {
      if (cweight[j]) {
        node[nn] = cnode[j];
        weight[nn] = cweight[j];
        nn++;
      }
    }
    return nn;
  }

  // shell mode: replace interior nodes by their TFI weights on the boundary
  int dims[3];
  flexLattice(m, f, dims);
  int nx = dims[0], ny = dims[1], nz = dims[2];
  int nn = 0;
  for (int q=0; q < npc; q++) {
    if (!cweight[q]) {
      continue;
    }
    int n = cnode[q];
    int k = n % nz, j = (n / nz) % ny, i = n / (ny*nz);
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
      int tidx[26];
      mjtNum tw[26];
      int nt = flexTFINode(nx, ny, nz, i, j, k, tidx, tw);
      for (int t=0; t < nt; t++) {
        nn = flexAddWeight(node, weight, nn, mjMAXFLEXINTERPW, tidx[t], cweight[q]*tw[t]);
      }
    } else {
      nn = flexAddWeight(node, weight, nn, mjMAXFLEXINTERPW, n, cweight[q]);
    }
  }
  return nn;
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
    if (!weight && !m->flex_interp[f] && !mj_rodHasTwist(m, f)) {
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

  // interpolated flex: node bodies weighted at the contact point's blended reference coordinates
  if (m->flex_interp[f]) {
    if (!m->flex_nodebodyid) {
      return 0;
    }
    mjtNum coord[3] = {0, 0, 0};
    for (int i=0; i < nw; i++) {
      mju_addToScl3(coord, m->flex_vert0 + 3*vid[i], vweight[i]);
    }
    int node[mjMAXFLEXINTERPW];
    mjtNum nweight[mjMAXFLEXINTERPW], wtmp[mjMAXFLEXINTERPW];
    int nn = mj_flexInterpWeights(m, f, coord, node, nweight);

    // map lattice nodes to their bodies, merging nodes that share a body
    int nstart = m->flex_nodeadr[f];
    mjtNum* bw = weight ? weight : wtmp;
    int nb = 0;
    for (int q=0; q < nn; q++) {
      if (nweight[q]) {
        nb = flexAddWeight(body, bw, nb, mjMAXFLEXINTERPW,
                           m->flex_nodebodyid[nstart + node[q]], nweight[q]);
      }
    }
    return nb;
  }

  // vertex bodies
  int nb = nw;
  for (int k=0; k < nw; k++) {
    body[k] = m->flex_vertbodyid[vid[k]];
    if (weight) {
      weight[k] = vweight[k];
    }
  }

  // 1D DER rod: append edge twist body and endpoint vertex bodies with 0 translational weight so
  // island discovery, sparse nnz counting, and mj_jacSum include the dofs of mj_rodContactJac
  if (mj_rodHasTwist(m, f) && m->flex_elemnum[f] > 0) {
    int elem = con->elem[side] >= 0 ? con->elem[side] : mj_rodVertElem(m, f, con->vert[side]);
    int v_other = (con->flex[1-side] == f ? con->vert[1-side] : -1);
    const int* edata = m->flex_elem + m->flex_elemdataadr[f] + 2*mjMAX(0, elem);
    if (elem >= 0 && edata[0] != v_other && edata[1] != v_other) {
      int b_extra[3], n_extra = 0;
      b_extra[n_extra++] = m->flex_vertbodyid[m->flex_vertadr[f] + edata[0]];
      b_extra[n_extra++] = m->flex_vertbodyid[m->flex_vertadr[f] + edata[1]];
      int jnt_e = mj_rodEdgeTwistJnt(m, f, elem, NULL);
      if (jnt_e >= 0) {
        b_extra[n_extra++] = m->jnt_bodyid[jnt_e];
      }
      for (int i = 0; i < n_extra; i++) {
        int b = b_extra[i], found = 0;
        for (int k = 0; k < nb; k++) {
          if (body[k] == b) {
            found = 1;
            break;
          }
        }
        if (!found) {
          if (weight) {
            weight[nb] = 0;
          }
          body[nb++] = b;
        }
      }
    }
  }
  return nb;
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

  // transform com-based acceleration to local frame
  mju_transformSpatial(res, d->cacc+6*bodyid, 0, pos, d->subtree_com+3*m->body_rootid[bodyid], rot);

  // dof-less body (static or mocap): zero velocity, quick return
  if (m->body_dofnum[m->body_weldid[bodyid]] == 0) {
    return;
  }

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


// map a single world point attached to body (gather: res += J_p * vec; scatter: res += scale * J_p^T * vec)
static void flexPointMap(const mjModel* m, const mjData* d, int body, const mjtNum point[3],
                         mjtNum* res, const mjtNum* vec, mjtNum scale, int transpose) {
  body = m->body_weldid[body];
  if (!m->body_dofnum[body]) {
    return;
  }
  int da = m->body_dofadr[body];
  if (m->body_simple[body] == 2) {
    // The world-space slide axes are already in cdof, including axis order and signs.
    for (int j=0; j < m->body_dofnum[body]; j++) {
      const mjtNum* axis = d->cdof + 6*(da+j)+3;
      if (transpose) {
        res[da+j] += scale*mju_dot3(axis, vec);
      } else {
        mji_addToScl3(res, axis, vec[da+j]);
      }
    }
    return;
  }
  mjtNum offset[3];
  mji_sub3(offset, point, d->subtree_com + 3*m->body_rootid[body]);
  for (int i=da + m->body_dofnum[body]-1; i >= 0; i=m->dof_parentid[i]) {
    mjtNum column[3];
    mji_cross(column, d->cdof + 6*i, offset);
    mji_addTo3(column, d->cdof + 6*i+3);
    if (transpose) {
      res[i] += scale*mju_dot3(column, vec);
    } else {
      mji_addToScl3(res, column, vec[i]);
    }
  }
}


// Apply the point Jacobian or its transpose without constructing a matrix. The general
// path walks the same motion axes as mj_jac, including every ancestor and the vertex offset.
// Gather overwrites 3*nvert world components; scatter adds to nv generalized components.
// For an interpolated flex (flex_interp != 0), the map factors through the lattice nodes: DOFs <->
// lattice nodes, then lattice nodes <-> vertices via flexInterpNodes, mj_flexInterpVert and
// flexInterpVertT.
static void flexMap(const mjModel* m, mjData* d, int f, mjtNum* res,
                    const mjtNum* vec, mjtNum scale, int transpose) {
  int vadr = m->flex_vertadr[f];
  int vertnum = m->flex_vertnum[f];
  int interp = m->flex_interp[f];
  if (!interp) {
    for (int v=0; v < vertnum; v++) {
      if (!transpose) mji_zero3(res + 3*v);
      flexPointMap(m, d, m->flex_vertbodyid[vadr+v], d->flexvert_xpos + 3*(vadr+v),
                   transpose ? res : res + 3*v, transpose ? vec + 3*v : vec, scale, transpose);
    }
    return;
  }

  int nadr = m->flex_nodeadr[f];
  int nodenum = m->flex_nodenum[f];
  mj_markStack(d);
  mjtNum* node_vec = mjSTACKALLOC(d, 3*nodenum, mjtNum);

  // gather: DOFs -> lattice nodes -> vertices
  if (!transpose) {
    mju_zero(node_vec, 3*nodenum);
    for (int i=0; i < nodenum; i++) {
      mjtNum point[3];
      mj_flexNodePos(m, d, f, nadr + i, point);
      flexPointMap(m, d, m->flex_nodebodyid[nadr + i], point, node_vec + 3*i, vec, 1, 0);
    }
    flexInterpNodes(m, f, node_vec);
    mj_flexInterpVert(m, f, node_vec, res);
  }

  // scatter: vertices -> lattice nodes -> DOFs
  else {
    mju_zero(node_vec, 3*nodenum);
    flexInterpVertT(m, f, vec, node_vec);
    for (int i=0; i < nodenum; i++) {
      if (!node_vec[3*i+0] && !node_vec[3*i+1] && !node_vec[3*i+2]) {
        continue;
      }
      mjtNum point[3];
      mj_flexNodePos(m, d, f, nadr + i, point);
      flexPointMap(m, d, m->flex_nodebodyid[nadr + i], point, res, node_vec + 3*i, scale, 1);
    }
  }

  mj_freeStack(d);
}


// Gather an arbitrary generalized vector (including qvel) in world vertex coordinates.
void mj_flexGather(const mjModel* m, mjData* d, int f, mjtNum* res, const mjtNum* vec) {
  flexMap(m, d, f, res, vec, 1, 0);
}


// Accumulate world vertex forces in generalized coordinates, including pin reactions.
void mj_flexScatter(const mjModel* m, mjData* d, int f, mjtNum* res,
                    const mjtNum* vec, mjtNum scale) {
  flexMap(m, d, f, res, vec, scale, 1);
}


// gather global node positions and velocities
void mju_flexGatherState(const mjModel* m, const mjData* d, int f, mjtNum* xpos, mjtNum* vel) {
  int nodenum = m->flex_nodenum[f];
  int nstart = m->flex_nodeadr[f];

  // compute positions and velocities
  for (int i=0; i < nodenum; i++) {
    int bid = m->flex_nodebodyid[nstart + i];
    mj_flexNodePos(m, d, f, nstart + i, xpos + 3*i);

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

  // shell mode: interior nodes follow the boundary
  flexInterpNodes(m, f, xpos);
  if (vel) {
    flexInterpNodes(m, f, vel);
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


// does the active metric step populate the qH backbone factor (efm_diag or nefmK)
int mj_effUseQH(const mjData* d) {
  return d->efm_active && (d->efm_diag != NULL || d->nefmK > 0);
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
  const mjtNum* plastic = m->flex_plasticadr[f] >= 0 && (m->flex_plasticmode[f] & 2) ?
                          d->plastic + m->flex_plasticadr[f] : NULL;

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
                       const mjtNum om[4], mjtNum w0, mjtNum w1, const mjtNum prm[4]) {
  mjtNum r0a = prm[0], r0b = prm[1], tc = prm[2], ts = prm[3];
  mjtNum kt = mju_sqrt(tc*tc + ts*ts);
  if (w0 <= 0 && w1 <= 0 && kt <= 0) {
    if (g) mju_zero(g, 4);
    if (Wk) mju_zero(Wk, 16);
    if (Wd) mju_zero(Wd, 16);
    return;
  }

  mjtNum e2 = mjROD_SMOOTH*mjROD_SMOOTH;
  mjtNum gam = mju_sqrt(1 + e2);

  // hinge a: smoothed curvature magnitude and unit binormal
  mjtNum ra = mju_sqrt(om[0]*om[0] + om[1]*om[1]);
  mjtNum na[2] = {0, 0};
  if (ra >= mjMINVAL) {
    na[0] = om[0] / ra;
    na[1] = om[1] / ra;
  }
  mjtNum ca = r0a > 0 ? ra / r0a : 0;
  mjtNum rho2a = ca*ca + e2;
  mjtNum rhoa = mju_sqrt(rho2a);
  mjtNum sa = r0a > 0 ? 1 - gam / rhoa : 1;

  // hinge b: smoothed curvature magnitude and unit binormal
  mjtNum rb = mju_sqrt(om[2]*om[2] + om[3]*om[3]);
  mjtNum nb[2] = {0, 0};
  if (rb >= mjMINVAL) {
    nb[0] = om[2] / rb;
    nb[1] = om[3] / rb;
  }
  mjtNum cb = r0b > 0 ? rb / r0b : 0;
  mjtNum rho2b = cb*cb + e2;
  mjtNum rhob = mju_sqrt(rho2b);
  mjtNum sb = r0b > 0 ? 1 - gam / rhob : 1;

  // smoothed unit binormals nta = alpa * oma, ntb = alpb * omb and their 2x2 Jacobians Ma, Mb
  mjtNum alpa = r0a > 0 ? gam / (rhoa * r0a) : 0;
  mjtNum alpb = r0b > 0 ? gam / (rhob * r0b) : 0;
  mjtNum nta[2] = {alpa * om[0], alpa * om[1]};
  mjtNum ntb[2] = {alpb * om[2], alpb * om[3]};
  mjtNum scla = ca * ca / rho2a;
  mjtNum Ma[4] = {
    alpa * (1 - scla * na[0] * na[0]),
    -alpa * scla * na[0] * na[1],
    -alpa * scla * na[1] * na[0],
    alpa * (1 - scla * na[1] * na[1])
  };
  mjtNum sclb = cb * cb / rho2b;
  mjtNum Mb[4] = {
    alpb * (1 - sclb * nb[0] * nb[0]),
    -alpb * sclb * nb[0] * nb[1],
    -alpb * sclb * nb[1] * nb[0],
    alpb * (1 - sclb * nb[1] * nb[1])
  };

  // gradient: radial bending + dihedral torsion kt |R(dphi0) nta - ntb|^2
  if (g) {
    mjtNum da[2] = {
      kt * nta[0] - (tc * ntb[0] + ts * ntb[1]),
      kt * nta[1] - (-ts * ntb[0] + tc * ntb[1])
    };
    mjtNum db[2] = {
      kt * ntb[0] - (tc * nta[0] - ts * nta[1]),
      kt * ntb[1] - (ts * nta[0] + tc * nta[1])
    };
    g[0] = 2 * w0 * sa * om[0] + 2 * (Ma[0] * da[0] + Ma[1] * da[1]);
    g[1] = 2 * w0 * sa * om[1] + 2 * (Ma[2] * da[0] + Ma[3] * da[1]);
    g[2] = 2 * w1 * sb * om[2] + 2 * (Mb[0] * db[0] + Mb[1] * db[1]);
    g[3] = 2 * w1 * sb * om[3] + 2 * (Mb[2] * db[0] + Mb[3] * db[1]);
  }

  // Hessian: clamped radial bending + Gauss-Newton torsion 2 kt Jr' Jr (PSD, exact at reference)
  if (Wk) {
    mjtNum tanga = r0a > 0 ? mju_max(0, sa) : 1;
    mjtNum rada = r0a > 0 ? mju_max(0, 1 - gam * e2 / (rhoa * rho2a)) : 1;
    mjtNum tangb = r0b > 0 ? mju_max(0, sb) : 1;
    mjtNum radb = r0b > 0 ? mju_max(0, 1 - gam * e2 / (rhob * rho2b)) : 1;

    // Ma * T' where T = [tc, -ts; ts, tc]
    mjtNum MaTt[4] = {
      Ma[0] * tc - Ma[1] * ts,  Ma[0] * ts + Ma[1] * tc,
      Ma[2] * tc - Ma[3] * ts,  Ma[2] * ts + Ma[3] * tc
    };

    mju_zero(Wk, 16);
    for (int i=0; i < 2; i++) {
      for (int j=0; j < 2; j++) {
        mjtNum delta = (i == j) ? 1 : 0;
        mjtNum Wka = (r0a > 0 && ra < mjMINVAL) ? 0 :
                     2 * w0 * (tanga * (delta - na[i]*na[j]) + rada * na[i]*na[j]);
        mjtNum Wkb = (r0b > 0 && rb < mjMINVAL) ? 0 :
                     2 * w1 * (tangb * (delta - nb[i]*nb[j]) + radb * nb[i]*nb[j]);
        mjtNum Ma2 = Ma[2*i+0] * Ma[0+j] + Ma[2*i+1] * Ma[2+j];
        mjtNum Mb2 = Mb[2*i+0] * Mb[0+j] + Mb[2*i+1] * Mb[2+j];
        mjtNum Kab = -2 * (MaTt[2*i+0] * Mb[0+j] + MaTt[2*i+1] * Mb[2+j]);

        Wk[4*i + j]         = Wka + 2 * kt * Ma2;
        Wk[4*(i+2) + (j+2)] = Wkb + 2 * kt * Mb2;
        Wk[4*i + (j+2)]     = Kab;
        Wk[4*(j+2) + i]     = Kab;
      }
    }
  }

  // damping: radial curvature rates + dihedral twist rate, orthogonal to edge rotations
  if (Wd) {
    mjtNum qd[4] = {0, 0, 0, 0};
    if (ra >= mjMINVAL && rb >= mjMINVAL && r0a > 0 && r0b > 0) {
      mjtNum scla_d = alpa * (cb / rhob);
      mjtNum sclb_d = alpb * (ca / rhoa);
      qd[0] = scla_d * na[1];
      qd[1] = -scla_d * na[0];
      qd[2] = -sclb_d * nb[1];
      qd[3] = sclb_d * nb[0];
    }
    mju_zero(Wd, 16);
    for (int i=0; i < 2; i++) {
      for (int j=0; j < 2; j++) {
        mjtNum delta = (i == j) ? 1 : 0;
        Wd[4*i + j]         = ra < mjMINVAL ? 2*w0*delta : 2*w0*na[i]*na[j];
        Wd[4*(i+2) + (j+2)] = rb < mjMINVAL ? 2*w1*delta : 2*w1*nb[i]*nb[j];
      }
    }
    for (int i=0; i < 4; i++) {
      for (int j=0; j < 4; j++) {
        Wd[4*i + j] += 2 * kt * qd[i] * qd[j];
      }
    }
  }
}


// nominal pose of body b at qpos0 (slide/hinge joints at 0)
static void rodBodyPose0(const mjModel* m, int b, mjtNum pos[3], mjtNum quat[4]) {
  mju_zero3(pos);
  quat[0] = 1;
  quat[1] = quat[2] = quat[3] = 0;
  for (int cur = b; cur > 0; cur = m->body_parentid[cur]) {
    mjtNum p[3], q[4];
    mju_rotVecQuat(p, pos, m->body_quat + 4*cur);
    mju_add3(pos, p, m->body_pos + 3*cur);
    mju_mulQuat(q, m->body_quat + 4*cur, quat);
    mju_copy4(quat, q);
  }
}


// nominal world position of flex vertex gv at qpos0
static void rodVertPos0(const mjModel* m, int gv, mjtNum xpos0[3], mjtNum xquat0[4]) {
  int b = m->flex_vertbodyid[gv];
  rodBodyPose0(m, b, xpos0, xquat0);
  mjtNum off[3];
  mju_rotVecQuat(off, m->flex_vert + 3*gv, xquat0);
  mju_addTo3(xpos0, off);
}


// parallel-transport unit vector u_in (orthogonal to t_from) to t_to; returns 0 if antipodal
static int rodParallelTransport(const mjtNum t_from[3], const mjtNum t_to[3],
                                const mjtNum u_in[3], mjtNum u_out[3], mjtNum v_out[3]) {
  mjtNum denom = 1 + mju_dot3(t_from, t_to);
  if (denom < mjMINVAL) {
    return 0;
  }
  mjtNum sum[3];
  mji_add3(sum, t_from, t_to);
  mjtNum scale = mju_dot3(t_to, u_in) / denom;
  mji_addScl3(u_out, u_in, sum, -scale);
  mju_normalize3(u_out);
  mji_cross(v_out, t_to, u_out);
  return 1;
}


// twist hinge joint ID of chain edge e (0 <= e <= nhinge), or -1 if none; edge 0's joint (free
// root) is the joint of edge 1 minus one, which the compiler validates
static int rodEdgeTwistJnt(const mjtNum* b, int nhinge, int root_clamped, int e) {
  if (e < 0 || e > nhinge) {
    return -1;
  }
  if (e == 0) {
    int jnt1 = (int)b[1 + 7];
    return (!root_clamped && jnt1 >= 1) ? (jnt1 - 1) : -1;
  }
  return (int)b[1 + mjROD_HINGE_SIZE*(e - 1) + 7];
}


// 1 if the first edge of the 1D rod chain with bending data b is clamped (both vertices on one
// body)
static int rodRootClamped(const mjModel* m, int f, const mjtNum* b) {
  int va = m->flex_vertadr[f];
  return m->flex_vertbodyid[va + (int)b[1 + 0]] == m->flex_vertbodyid[va + (int)b[1 + 1]];
}


// 1 if the last edge of the 1D rod chain with bending data b is clamped (both vertices on one
// body), with at least two hinges so that the first edge's joint convention still holds
static int rodTailClamped(const mjModel* m, int f, const mjtNum* b) {
  int nhinge = (int)b[0];
  if (nhinge < 2) {
    return 0;
  }
  const mjtNum* last = b + 1 + mjROD_HINGE_SIZE*(nhinge - 1);
  int va = m->flex_vertadr[f];
  return m->flex_vertbodyid[va + (int)last[1]] == m->flex_vertbodyid[va + (int)last[2]];
}


// clamped first (root) and last (tail) edges of the 1D DER rod chain of flex f
void mj_rodClampedEnds(const mjModel* m, int f, int* root, int* tail) {
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  *root = rodRootClamped(m, f, b);
  *tail = rodTailClamped(m, f, b);
}


// twist hinge joint ID of chain edge c (0 <= c <= nhinge) of a 1D rod, or -1 if none
int mj_rodChainTwistJnt(const mjModel* m, int f, int c) {
  if (m->flex_dim[f] != 1 || m->flex_bendingadr[f] < 0) {
    return -1;
  }
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  if (nhinge <= 0 || b[1 + 3] >= 0) {
    return -1;
  }
  int jnt = rodEdgeTwistJnt(b, nhinge, rodRootClamped(m, f, b), c);
  return (jnt >= 0 && jnt < m->njnt) ? jnt : -1;
}


// dof address of the twist hinge of chain edge c of a 1D rod, or -1 if none
int mj_rodChainTwistDof(const mjModel* m, int f, int c) {
  int jnt = mj_rodChainTwistJnt(m, f, c);
  return jnt >= 0 ? m->jnt_dofadr[jnt] : -1;
}


// time-parallel state (t_old, u_old) of chain edge e of 1D rod f in d->plastic, NULL if none (zero
// after mj_resetData, or a keyframe without plastic)
static const mjtNum* rodEdgeState(const mjModel* m, const mjData* d, int f, int e) {
  if (m->flex_plasticadr[f] < 0 || !d->plastic) {
    return NULL;
  }
  const mjtNum* s = d->plastic + m->flex_plasticadr[f] + 6*e;
  return (mju_dot3(s, s) > 0.25 && mju_dot3(s + 3, s + 3) > 0.25) ? s : NULL;
}


// some unit normal u of unit vector t, and v = t x u
static void rodAnyNormal(const mjtNum t[3], mjtNum u[3], mjtNum v[3]) {
  mjtNum frame[9] = {t[0], t[1], t[2], 0, 0, 0, 0, 0, 0};
  mju_makeFrame(frame);
  mji_copy3(u, frame + 3);
  mji_cross(v, t, u);
}


// evaluate a 1D DER rod chain of single hinges with time-parallel reference directors and per-edge
// twist hinge DOFs; returns 1 if handled
static int rodChain(const mjModel* m, const mjData* d, mjData* d_stack, int f, mjtNum* energy_out,
                    mjtNum* frc, mjtNum* qfrc_out,
                    mjtNum* Wk_out, mjtNum* Wd_out, mjtNum* state_out,
                    mjtNum* elem_mat_out, mjtNum* ktheta_out) {
  int bendingadr = m->flex_bendingadr[f];
  if (bendingadr < 0) {
    return 0;
  }
  const mjtNum* b = m->flex_bending + bendingadr;
  int nhinge = (int)b[0];
  if (nhinge <= 0) {
    return 0;
  }

  int any_aniso = 0;
  int any_kb0 = 0;
  int any_twist_jnt = 0;
  for (int h = 0; h < nhinge; h++) {
    const mjtNum* hinge = b + 1 + mjROD_HINGE_SIZE*h;
    if (hinge[3] >= 0) {
      return 0;
    }
    if (h > 0) {
      const mjtNum* prev = b + 1 + mjROD_HINGE_SIZE*(h - 1);
      if ((int)hinge[0] != (int)prev[1] || (int)hinge[1] != (int)prev[2]) {
        return 0;
      }
    }
    if (hinge[5] != 0 || hinge[9] != 0) {
      any_aniso = 1;
    }
    if (hinge[6] > 0) {
      any_kb0 = 1;
    }
    if (hinge[7] >= 0) {
      any_twist_jnt = 1;
    }
  }

  // closed-loop chains are not handled by the open-chain DER solver
  const mjtNum* last_hinge = b + 1 + mjROD_HINGE_SIZE*(nhinge - 1);
  if ((int)last_hinge[1] == (int)b[1 + 0] || (int)last_hinge[2] == (int)b[1 + 0]) {
    return 0;
  }
  if (!any_aniso && !any_twist_jnt && !elem_mat_out) {
    return 0;
  }

  int nedge = nhinge + 1;
  int nvert_chain = nhinge + 2;
  int va = m->flex_vertadr[f];
  const mjtNum* xpos = d->flexvert_xpos + 3*va;
  int b0 = m->flex_vertbodyid[va + (int)b[1 + 0]];
  int b1 = m->flex_vertbodyid[va + (int)b[1 + 1]];
  int root_clamped = (b0 == b1);
  int tail_clamped = rodTailClamped(m, f, b);
  int bT = m->flex_vertbodyid[va + (int)last_hinge[2]];

  int need_J = (frc || Wk_out || Wd_out) ? 1 : 0;
  int need_eval = (energy_out || frc || qfrc_out || Wk_out || Wd_out || ktheta_out) ? 1 : 0;
  int need_dpsi = (energy_out || frc || qfrc_out || state_out) ? 1 : 0;
  int need_ref_chain = (need_eval && any_kb0) || tail_clamped;
  int nvert_ref = need_ref_chain ? nvert_chain : 2;
  size_t nnums = (size_t)(3*nvert_ref + 6*nhinge + 27*need_J*nhinge + 25*nedge);
  if (d_stack) {
    mj_markStack(d_stack);
  }
  mjtNum* buf = d_stack ? mjSTACKALLOC(d_stack, nnums, mjtNum)
                        : (mjtNum*)mju_malloc(nnums * sizeof(mjtNum));
  int* v_chain = d_stack ? mjSTACKALLOC(d_stack, nvert_chain, int)
                         : (int*)mju_malloc((size_t)nvert_chain * sizeof(int));
#define mjROD_FREE()          \
  do {                        \
    if (d_stack) {            \
      mj_freeStack(d_stack);  \
    } else {                  \
      mju_free(v_chain);      \
      mju_free(buf);          \
    }                         \
  } while (0)

  mjtNum* ptr = buf;
  for (int h = 0; h < nhinge; h++) {
    v_chain[h] = (int)b[1 + mjROD_HINGE_SIZE*h + 0];
  }
  v_chain[nhinge]     = (int)b[1 + mjROD_HINGE_SIZE*(nhinge - 1) + 1];
  v_chain[nhinge + 1] = (int)b[1 + mjROD_HINGE_SIZE*(nhinge - 1) + 2];

  // reference vertex positions in qpos0 world coordinates (vertices past the first edge only when a
  // curved reference or a clamped tail needs the reference Bishop frame along the chain)
  mjtNum* xpos0 = ptr; ptr += 3*nvert_ref;
  mjtNum p0_b0[3], q0_b0[4], R0_0[9];
  rodBodyPose0(m, b0, p0_b0, q0_b0);
  mju_quat2Mat(R0_0, q0_b0);
  for (int k = 0; k < nvert_ref; k++) {
    mjtNum qk[4];
    rodVertPos0(m, va + v_chain[k], xpos0 + 3*k, qk);
  }

  // initial reference frame (t0_ref, u0_ref, v0_ref) on edge 0
  mjtNum e0_ref[3], frame_loc[9] = {0}, t0_ref[3], u0_ref[3], v0_ref[3];
  mju_sub3(e0_ref, xpos0 + 3, xpos0);
  mju_mulMatTVec3(frame_loc, R0_0, e0_ref);
  if (mju_normalize3(frame_loc) < mjMINVAL) {
    mjROD_FREE();
    return 0;
  }
  mju_makeFrame(frame_loc);
  mjtNum psi0 = (b[1 + 9] < 0) ? 0.5*mjPI : 0;
  mjtNum c0 = mju_cos(psi0), s0 = mju_sin(psi0);
  mjtNum u_loc[3], v_loc[3];
  for (int c = 0; c < 3; c++) {
    u_loc[c] = c0*frame_loc[3 + c] + s0*frame_loc[6 + c];
    v_loc[c] = -s0*frame_loc[3 + c] + c0*frame_loc[6 + c];
  }
  mju_mulMatVec3(t0_ref, d->xmat + 9*b0, frame_loc);
  mju_mulMatVec3(u0_ref, d->xmat + 9*b0, u_loc);
  mju_mulMatVec3(v0_ref, d->xmat + 9*b0, v_loc);

  // propagate reference Bishop frame along reference edges to get om0 and (t_ref, u_ref) per edge
  mjtNum* om0 = ptr; ptr += 2*nhinge;
  mju_zero(om0, 2*nhinge);
  mjtNum* t_ref = ptr; ptr += 3*nedge;
  mjtNum* u_ref = ptr; ptr += 3*nedge;
  mjtNum v_prev_ref[3];
  mju_mulMatVec3(t_ref, R0_0, frame_loc);
  mju_mulMatVec3(u_ref, R0_0, u_loc);
  mju_mulMatVec3(v_prev_ref, R0_0, v_loc);
  for (int h = 0; need_ref_chain && h < nhinge; h++) {
    if (need_eval && b[1 + mjROD_HINGE_SIZE*h + 6] > 0) {
      mjtNum kb_ref[3];
      if (mj_rodHinge(xpos0 + 3*h, xpos0 + 3*(h+1), xpos0 + 3*(h+2), kb_ref, NULL)) {
        om0[2*h + 0] = mju_dot3(kb_ref, u_ref + 3*h);
        om0[2*h + 1] = mju_dot3(kb_ref, v_prev_ref);
      }
    }
    mjtNum* t_next_ref = t_ref + 3*(h+1);
    mju_sub3(t_next_ref, xpos0 + 3*(h+2), xpos0 + 3*(h+1));
    if (mju_normalize3(t_next_ref) < mjMINVAL ||
        !rodParallelTransport(t_ref + 3*h, t_next_ref, u_ref + 3*h, u_ref + 3*(h+1), v_prev_ref)) {
      mjROD_FREE();
      return 0;
    }
  }

  // clamped tail: the reference frame of the last edge rotates with its body, like the root's
  mjtNum tT_ref[3] = {0, 0, 0}, uT_ref[3] = {0, 0, 0};
  if (tail_clamped) {
    mjtNum pT[3], qT[4], R0_T[9], loc[3], R_rel[9];
    rodBodyPose0(m, bT, pT, qT);
    mju_quat2Mat(R0_T, qT);
    mju_mulMatMatT3(R_rel, d->xmat + 9*bT, R0_T);
    mju_mulMatVec3(loc, R_rel, t_ref + 3*nhinge);
    mji_copy3(tT_ref, loc);
    mju_mulMatVec3(loc, R_rel, u_ref + 3*nhinge);
    mji_copy3(uT_ref, loc);
  }

  // deformed edges and exact 6-float time-parallel reference directors in d->plastic (6*nedge)
  mjtNum* t_prev = ptr; ptr += 3*nedge;
  mjtNum* t_edge = ptr; ptr += 3*nedge;
  mjtNum* u_edge = ptr; ptr += 3*nedge;
  mjtNum* v_edge = ptr; ptr += 3*nedge;
  mjtNum* l_edge = ptr; ptr += nedge;
  for (int e = 0; e < nedge; e++) {
    mju_sub3(t_edge + 3*e, xpos + 3*v_chain[e+1], xpos + 3*v_chain[e]);
    l_edge[e] = mju_normalize3(t_edge + 3*e);
    if (l_edge[e] < mjMINVAL) {
      mjROD_FREE();
      return 0;
    }
  }

  const mjtNum* state = (m->flex_plasticadr[f] >= 0 && d->plastic)
                        ? (d->plastic + m->flex_plasticadr[f]) : NULL;
  // reference director of each edge: the time-parallel transport of its state (t_old, u_old) to
  // the current tangent, the clamped first and last edges following their bodies. An edge without
  // state (after mj_resetData, or a keyframe without plastic) or whose transport fails takes the
  // spatial parallel transport of the previous edge's director, and a free first edge its qpos0
  // reference transported to the current tangent; a rod without state thus starts twist-free from
  // edge 0's current frame, and its state always initializes. Such directors are frozen like a
  // state (t_prev = t_edge): the vertex forces do not differentiate through the propagation
  mjtNum* anchored = ptr; ptr += nedge;
  for (int e = 0; e < nedge; e++) {
    mjtNum* te = t_edge + 3*e;
    if ((e == 0 && root_clamped) || (e == nhinge && tail_clamped)) {
      const mjtNum* tc = e ? tT_ref : t0_ref;
      const mjtNum* uc = e ? uT_ref : u0_ref;
      mji_copy3(t_prev + 3*e, tc);
      anchored[e] = 1;
      if (!rodParallelTransport(tc, te, uc, u_edge + 3*e, v_edge + 3*e)) {
        mjROD_FREE();
        return 0;
      }
      continue;
    }
    const mjtNum* state_e = state ? rodEdgeState(m, d, f, e) : NULL;
    if (state_e && rodParallelTransport(state_e, te, state_e + 3, u_edge + 3*e, v_edge + 3*e)) {
      mji_copy3(t_prev + 3*e, state_e);
      anchored[e] = 1;
      continue;
    }
    anchored[e] = 0;
    mji_copy3(t_prev + 3*e, te);
    mjtNum* ue = u_edge + 3*e;
    int ok = (e > 0) ? rodParallelTransport(te - 3, te, ue - 3, ue, v_edge + 3*e)
                     : rodParallelTransport(t_ref, te, u_ref, u_edge, v_edge);
    if (!ok) {
      rodAnyNormal(te, u_edge + 3*e, v_edge + 3*e);
    }
  }

  // reference twist dpsi_h of each hinge: the roll of edge h+1's reference director relative to
  // edge h's transported along the hinge. atan2 gives it modulo 2pi only; when both edges carry
  // time-parallel state, the unwrapped value of the previous step (stored after the frames)
  // selects the branch, so dpsi stays continuous on writhing or tumbling rods and the joint
  // difference theta[h+1] - theta[h] is never wrapped (twist beyond pi per hinge keeps its energy)
  const mjtNum* psi_state = state ? (state + 6*nedge) : NULL;
  mjtNum* dpsi = ptr; ptr += nhinge;
  for (int h = 0; need_dpsi && h < nhinge; h++) {
    mjtNum u_trans[3], v_trans[3];
    if (!rodParallelTransport(t_edge + 3*h, t_edge + 3*(h+1), u_edge + 3*h, u_trans, v_trans)) {
      mjROD_FREE();
      return 0;
    }
    dpsi[h] = mju_atan2(mju_dot3(u_edge + 3*(h+1), v_trans), mju_dot3(u_edge + 3*(h+1), u_trans));
    if (psi_state && anchored[h] && anchored[h+1]) {
      mjtNum jump = dpsi[h] - psi_state[h];
      dpsi[h] -= 2*mjPI * mju_round(jump / (2*mjPI));
    }
  }
  if (state_out) {
    for (int e = 0; e < nedge; e++) {
      mji_copy3(state_out + 6*e, t_edge + 3*e);
      mji_copy3(state_out + 6*e + 3, u_edge + 3*e);
    }
    mju_copy(state_out + 6*nedge, dpsi, nhinge);
  }
  if (!need_eval && !elem_mat_out) {
    mjROD_FREE();
    return 1;
  }

  mjtNum* theta = ptr; ptr += nedge;
  for (int e = 0; e < nedge; e++) {
    int jnt = rodEdgeTwistJnt(b, nhinge, root_clamped, e);
    theta[e] = (jnt >= 0) ? (d->qpos[m->jnt_qposadr[jnt]] - m->qpos0[m->jnt_qposadr[jnt]]) : 0;
  }

  mjtNum energy = 0;
  if (need_eval) {
    // evaluate hinge curvature binormals and Jacobians; this is the last failure exit, so a rod
    // that falls back to the isotropic path never leaves partial forces in frc or qfrc_out
    mjtNum* kb_all = ptr; ptr += 3*nhinge;
    mjtNum* J_all  = need_J ? (ptr += 27*nhinge, ptr - 27*nhinge) : NULL;
    for (int h = 0; h < nhinge; h++) {
      if (!mj_rodHinge(xpos + 3*v_chain[h], xpos + 3*v_chain[h+1], xpos + 3*v_chain[h+2],
                       kb_all + 3*h, J_all ? J_all + 27*h : NULL)) {
        mjROD_FREE();
        return 0;
      }
    }

    // dE/dtheta_e: also the energy's sensitivity to a roll of edge e's reference director
    mjtNum* grad_theta = ptr; ptr += nedge;
    mju_zero(grad_theta, nedge);
    mjtNum* g_edge = frc ? (ptr += 3*nedge, ptr - 3*nedge) : NULL;
    if (g_edge) {
      mju_zero(g_edge, 3*nedge);
    }
    if (ktheta_out) {
      mju_zero(ktheta_out, nedge);
    }

    // evaluate energy, kb-space weights/forces, and local reference director derivatives
    for (int h = 0; h < nhinge; h++) {
      const mjtNum* hinge = b + 1 + mjROD_HINGE_SIZE*h;
      mjtNum k1 = hinge[4], k2 = k1 + hinge[5], kt = hinge[8];
      mjtNum kd1 = Wd_out ? mju_sqrt(k1 * k2) : 0;
      mjtNum w1_0 = om0[2*h + 0], w2_0 = om0[2*h + 1];

      mjtNum g_kb[3] = {0, 0, 0};
      mjtNum Wk_h[9] = {0};
      mjtNum Wd_h[9] = {0};
      for (int side = 0; side < 2; side++) {
        int e = h + side;
        const mjtNum* ue = u_edge + 3*e;
        const mjtNum* ve = v_edge + 3*e;
        mjtNum om_e0 = mju_dot3(kb_all + 3*h, ue);
        mjtNum om_e1 = mju_dot3(kb_all + 3*h, ve);
        mjtNum ce = mju_cos(theta[e]), se = mju_sin(theta[e]);
        mjtNum m1[3], m2[3];
        for (int c = 0; c < 3; c++) {
          m1[c] = ce*ue[c] + se*ve[c];
          m2[c] = -se*ue[c] + ce*ve[c];
        }
        mjtNum w1 = ce*om_e0 + se*om_e1;
        mjtNum w2 = -se*om_e0 + ce*om_e1;
        mjtNum dw1 = w1 - w1_0, dw2 = w2 - w2_0;

        energy += 0.5 * (k1*dw1*dw1 + k2*dw2*dw2);
        if (frc) {
          for (int c = 0; c < 3; c++) {
            g_kb[c] += k1*dw1*m1[c] + k2*dw2*m2[c];
          }
        }
        if (Wk_out) {
          for (int c = 0; c < 3; c++) {
            for (int d_idx = 0; d_idx < 3; d_idx++) {
              Wk_h[3*c + d_idx] += k1*m1[c]*m1[d_idx] + k2*m2[c]*m2[d_idx];
            }
          }
        }
        if (Wd_out) {
          for (int c = 0; c < 3; c++) {
            for (int d_idx = 0; d_idx < 3; d_idx++) {
              Wd_h[3*c + d_idx] += kd1*m1[c]*m1[d_idx] + k2*m2[c]*m2[d_idx];
            }
          }
        }
        mjtNum dE_dtheta = k1*dw1*w2 - k2*dw2*w1;
        grad_theta[e] += dE_dtheta;

        // bending stiffness of theta_e (dw1/dtheta = w2, dw2/dtheta = -w1): the exact second
        // derivative, clamped PSD per side. It equals the Gauss-Newton k1 w2^2 + k2 w1^2 at the
        // reference but, unlike it, vanishes for circular sections with a straight reference,
        // whose energy does not depend on theta: a Gauss-Newton k |w|^2 there would be a twist
        // stiffness absent from the force, i.e. artificial damping of rolling under discrete
        if (ktheta_out && (any_aniso || any_kb0)) {
          mjtNum kth = k1*(w2*w2 - dw1*w1) + k2*(w1*w1 - dw2*w2);
          ktheta_out[e] += mju_max(0, kth);
        }
      }

      if (need_dpsi) {
        mjtNum dtheta = theta[h+1] - theta[h] + dpsi[h];
        energy += kt * dtheta * dtheta;
        mjtNum tau_h = 2 * kt * dtheta;
        grad_theta[h]     -= tau_h;
        grad_theta[h + 1] += tau_h;
        if (frc) {
          mji_addToScl3(g_edge + 3*h,     kb_all + 3*h, tau_h / (2*l_edge[h]));
          mji_addToScl3(g_edge + 3*(h+1), kb_all + 3*h, tau_h / (2*l_edge[h+1]));
        }
      }

      if (Wk_out) {
        mju_copy(Wk_out + 9*h, Wk_h, 9);
      }
      if (Wd_out) {
        mjtNum r = mju_norm3(kb_all + 3*h);
        if (r < mjMINVAL) {
          mju_copy(Wd_out + 9*h, Wd_h, 9);
        } else {
          mjtNum n[3] = {kb_all[3*h+0]/r, kb_all[3*h+1]/r, kb_all[3*h+2]/r};
          mjtNum Wn[3];
          mju_mulMatVec3(Wn, Wd_h, n);
          mjtNum keff2 = mju_dot3(n, Wn);
          for (int r_i = 0; r_i < 3; r_i++) {
            for (int c_i = 0; c_i < 3; c_i++) {
              Wd_out[9*h + 3*r_i + c_i] = keff2 * n[r_i] * n[c_i];
            }
          }
        }
      }
      if (frc) {
        const mjtNum* J = J_all + 27*h;
        for (int i = 0; i < 3; i++) {
          for (int c = 0; c < 3; c++) {
            for (int r = 0; r < 3; r++) {
              frc[3*v_chain[h+i] + c] -= J[9*r + 3*i + c] * g_kb[r];
            }
          }
        }
      }
    }

    if (qfrc_out) {
      for (int e = 0; e < nedge; e++) {
        int jnt = rodEdgeTwistJnt(b, nhinge, root_clamped, e);
        if (jnt >= 0) {
          qfrc_out[m->jnt_dofadr[jnt]] -= grad_theta[e];
        }
      }
    }

    if (frc) {
      for (int e = 0; e < nedge; e++) {
        if (grad_theta[e] != 0) {
          mjtNum cross_e[3];
          mji_cross(cross_e, t_prev + 3*e, t_edge + 3*e);
          mjtNum denom_e = l_edge[e] * (1 + mju_dot3(t_prev + 3*e, t_edge + 3*e));
          if (denom_e >= mjMINVAL) {
            mji_addToScl3(g_edge + 3*e, cross_e, -grad_theta[e] / denom_e);
          }
        }
        mji_addTo3(frc + 3*v_chain[e],     g_edge + 3*e);
        mji_subFrom3(frc + 3*v_chain[e+1], g_edge + 3*e);
      }
    }

    // reaction torque on the bodies of clamped ends: their rotation rolls the end frame by
    // (t_ref + t_edge) / (1 + t_ref.t_edge) per unit rotation vector, at fixed edge tangent
    for (int end = 0; end < 2 && qfrc_out; end++) {
      int e = end ? nhinge : 0;
      if (!(end ? tail_clamped : root_clamped) || grad_theta[e] == 0) {
        continue;
      }
      const mjtNum* tc = end ? tT_ref : t0_ref;
      mjtNum d0 = 1 + mju_dot3(tc, t_edge + 3*e);
      if (d0 >= mjMINVAL) {
        mjtNum sum0[3], tau_b[3];
        mju_add3(sum0, tc, t_edge + 3*e);
        mji_scl3(tau_b, sum0, -grad_theta[e] / d0);
        int bw = m->body_weldid[end ? bT : b0];
        for (int i = m->body_dofadr[bw] + m->body_dofnum[bw] - 1; i >= 0; i = m->dof_parentid[i]) {
          qfrc_out[i] += mju_dot3(d->cdof + 6*i, tau_b);
        }
      }
    }
  }

  // populate per-element 3x3 rotation matrices [m1, m2, t] when requested
  if (elem_mat_out) {
    int elemnum = m->flex_elemnum[f];
    const int* edata = m->flex_elem + m->flex_elemdataadr[f];
    for (int e = 0; e < elemnum; e++) {
      int v0 = edata[2*e + 0], v1 = edata[2*e + 1];
      int c = (e < nedge && v_chain[e] == v0 && v_chain[e+1] == v1) ? e : -1;
      mjtNum sgn = 1;
      if (c < 0) {
        for (int k = 0; k < nedge; k++) {
          if (v_chain[k] == v0 && v_chain[k+1] == v1) {
            c = k;
            sgn = 1;
            break;
          } else if (v_chain[k] == v1 && v_chain[k+1] == v0) {
            c = k;
            sgn = -1;
            break;
          }
        }
      }
      if (c >= 0) {
        mjtNum ce = mju_cos(theta[c]), se = mju_sin(theta[c]);
        const mjtNum* uc = u_edge + 3*c;
        const mjtNum* vc = v_edge + 3*c;
        const mjtNum* tc = t_edge + 3*c;
        for (int r = 0; r < 3; r++) {
          elem_mat_out[9*e + 3*r + 0] = ce*uc[r] + se*vc[r];
          elem_mat_out[9*e + 3*r + 1] = sgn*(-se*uc[r] + ce*vc[r]);
          elem_mat_out[9*e + 3*r + 2] = sgn*tc[r];
        }
      } else {
        mju_zero(elem_mat_out + 9*e, 9);
        elem_mat_out[9*e + 0] = elem_mat_out[9*e + 4] = elem_mat_out[9*e + 8] = 1;
      }
    }
  }

  if (energy_out) {
    *energy_out = energy;
  }
  mjROD_FREE();
#undef mjROD_FREE
  return 1;
}


int mj_rodChain(const mjModel* m, mjData* d, int f, mjtNum* energy_out,
                mjtNum* frc, mjtNum* qfrc_out,
                mjtNum* Wk_out, mjtNum* Wd_out, mjtNum* ktheta_out,
                mjtNum* state_out) {
  return rodChain(m, d, d, f, energy_out, frc, qfrc_out, Wk_out, Wd_out, state_out, NULL,
                  ktheta_out);
}


// return 1 if 1D flex f is an open DER rod chain with per-edge twist DOFs or box section, 0 else
int mj_rodHasTwist(const mjModel* m, int f) {
  if (m->flex_dim[f] != 1 || m->flex_interp[f] || m->flex_elemnum[f] <= 0 ||
      m->flex_bendingadr[f] < 0) {
    return 0;
  }
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  if (nhinge <= 0) {
    return 0;
  }
  const mjtNum* hinge0 = b + 1;
  const mjtNum* last_hinge = b + 1 + mjROD_HINGE_SIZE*(nhinge - 1);
  if (hinge0[3] >= 0 ||
      (int)last_hinge[1] == (int)hinge0[0] || (int)last_hinge[2] == (int)hinge0[0]) {
    return 0;
  }
  return (hinge0[7] >= 0 || last_hinge[7] >= 0 || hinge0[9] != 0) ? 1 : 0;
}


// cross-section half-sizes (a >= b) of a 1D DER rod with a box section; returns 1 if box, 0 else
int mj_rodBoxSize(const mjModel* m, int f, mjtNum size[2]) {
  if (m->flex_dim[f] != 1 || m->flex_bendingadr[f] < 0) {
    return 0;
  }
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  if (nhinge <= 0) {
    return 0;
  }
  const mjtNum* hinge = b + 1;
  if (hinge[3] >= 0 || hinge[4] <= 0 || hinge[9] == 0) {
    return 0;
  }
  mjtNum b_half = mju_abs(hinge[9]);
  mjtNum a_half = b_half * mju_sqrt(mju_max(1, 1 + hinge[5] / hinge[4]));
  if (size) {
    size[0] = a_half;
    size[1] = b_half;
  }
  return 1;
}


// per-element 3x3 rotation matrices [m1, m2, t] and half-sizes (a >= b) of a 1D box-section rod;
// returns 1 if handled, 0 otherwise
int mj_rodBoxEdge(const mjModel* m, const mjData* d, int f, mjtNum* elem_mat, mjtNum size[2]) {
  if (!mj_rodBoxSize(m, f, size)) {
    return 0;
  }
  if (!elem_mat) {
    return 1;
  }
  return rodChain(m, d, NULL, f, NULL, NULL, NULL, NULL, NULL, NULL, elem_mat, NULL);
}


// reference frame (t, u) of the first edge of 1D rod f: rotated with the current pose of its
// first vertex body if use_current_b0 (the clamped root), else at qpos0
static int rodRootDirector(const mjModel* m, const mjData* d, int f, int use_current_b0,
                           mjtNum t_out[3], mjtNum u_out[3]) {
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int va = m->flex_vertadr[f];
  int b0 = m->flex_vertbodyid[va + (int)b[1 + 0]];
  mjtNum p0_b0[3], q0_b0[4], R0_0[9];
  rodBodyPose0(m, b0, p0_b0, q0_b0);
  mju_quat2Mat(R0_0, q0_b0);

  mjtNum x0[3], x1[3], qk[4];
  rodVertPos0(m, va + (int)b[1 + 0], x0, qk);
  rodVertPos0(m, va + (int)b[1 + 1], x1, qk);

  mjtNum e0_ref[3], frame_loc[9] = {0};
  mju_sub3(e0_ref, x1, x0);
  mju_mulMatTVec3(frame_loc, R0_0, e0_ref);
  if (mju_normalize3(frame_loc) < mjMINVAL) {
    return 0;
  }
  mju_makeFrame(frame_loc);
  mjtNum psi0 = (b[1 + 9] < 0) ? 0.5*mjPI : 0;
  mjtNum c0 = mju_cos(psi0), s0 = mju_sin(psi0);
  mjtNum u_loc[3];
  for (int c = 0; c < 3; c++) {
    u_loc[c] = c0*frame_loc[3 + c] + s0*frame_loc[6 + c];
  }
  const mjtNum* R_base = use_current_b0 ? (d->xmat + 9*b0) : R0_0;
  mju_mulMatVec3(t_out, R_base, frame_loc);
  mju_mulMatVec3(u_out, R_base, u_loc);
  return 1;
}


// flex-local vertex k (0 <= k <= nhinge + 1) of the 1D rod chain with bending data b
static inline int rodChainVert(const mjtNum* b, int nhinge, int k) {
  if (k < nhinge) {
    return (int)b[1 + mjROD_HINGE_SIZE*k + 0];
  }
  return (int)b[1 + mjROD_HINGE_SIZE*(nhinge - 1) + 1 + (k - nhinge)];
}


// reference frame (t, u) of the clamped last edge of 1D rod f, rotated with the current pose of
// its body: the qpos0 reference frame propagated along the qpos0 chain, as in rodChain
static int rodTailDirector(const mjModel* m, const mjData* d, int f,
                           mjtNum t_out[3], mjtNum u_out[3]) {
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  int va = m->flex_vertadr[f];
  mjtNum t_cur[3], u_cur[3], v_cur[3], x1[3], qk[4];
  if (!rodRootDirector(m, d, f, 0, t_cur, u_cur)) {
    return 0;
  }
  rodVertPos0(m, va + rodChainVert(b, nhinge, 1), x1, qk);
  for (int k = 1; k <= nhinge; k++) {
    mjtNum x2[3], t_next[3], u_next[3];
    rodVertPos0(m, va + rodChainVert(b, nhinge, k + 1), x2, qk);
    mju_sub3(t_next, x2, x1);
    if (mju_normalize3(t_next) < mjMINVAL ||
        !rodParallelTransport(t_cur, t_next, u_cur, u_next, v_cur)) {
      return 0;
    }
    mji_copy3(t_cur, t_next);
    mji_copy3(u_cur, u_next);
    mji_copy3(x1, x2);
  }
  int bT = m->flex_vertbodyid[va + rodChainVert(b, nhinge, nhinge + 1)];
  mjtNum pT[3], qT[4], R0_T[9], R_rel[9];
  rodBodyPose0(m, bT, pT, qT);
  mju_quat2Mat(R0_T, qT);
  mju_mulMatMatT3(R_rel, d->xmat + 9*bT, R0_T);
  mju_mulMatVec3(t_out, R_rel, t_cur);
  mju_mulMatVec3(u_out, R_rel, u_cur);
  return 1;
}


// current tangent t and reference director (u, v) of chain edge c of 1D rod f, by the rule of
// rodChain: walk back to the nearest edge whose director does not depend on its predecessor (the
// clamped root, an edge with a transportable state, or edge 0), then transport spatially to c;
// O(1) when edge c has a state. Returns 0 if an edge is degenerate
static int rodEdgeDirector(const mjModel* m, const mjData* d, int f, int c,
                           mjtNum t_out[3], mjtNum u_out[3], mjtNum v_out[3]) {
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  const mjtNum* xpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
  int root_clamped = rodRootClamped(m, f, b);
  int tail_clamped = rodTailClamped(m, f, b);

  int k = c;
  for (; k >= 0; k--) {
    mju_sub3(t_out, xpos + 3*rodChainVert(b, nhinge, k + 1), xpos + 3*rodChainVert(b, nhinge, k));
    if (mju_normalize3(t_out) < mjMINVAL) {
      return 0;
    }
    if ((k == 0 && root_clamped) || (k == nhinge && tail_clamped)) {
      mjtNum t0[3], u0[3];
      int ok = k ? rodTailDirector(m, d, f, t0, u0) : rodRootDirector(m, d, f, 1, t0, u0);
      if (!ok || !rodParallelTransport(t0, t_out, u0, u_out, v_out)) {
        return 0;
      }
      break;
    }
    const mjtNum* state_k = rodEdgeState(m, d, f, k);
    if (state_k && rodParallelTransport(state_k, t_out, state_k + 3, u_out, v_out)) {
      break;
    }
    if (k == 0) {
      mjtNum t0[3], u0[3];
      if (!rodRootDirector(m, d, f, 0, t0, u0) ||
          !rodParallelTransport(t0, t_out, u0, u_out, v_out)) {
        rodAnyNormal(t_out, u_out, v_out);
      }
      break;
    }
  }
  for (int e = k + 1; e <= c; e++) {
    mjtNum t_e[3], u_e[3];
    mju_sub3(t_e, xpos + 3*rodChainVert(b, nhinge, e + 1), xpos + 3*rodChainVert(b, nhinge, e));
    if (mju_normalize3(t_e) < mjMINVAL) {
      return 0;
    }
    if (!rodParallelTransport(t_out, t_e, u_out, u_e, v_out)) {
      rodAnyNormal(t_e, u_e, v_out);
    }
    mji_copy3(t_out, t_e);
    mji_copy3(u_out, u_e);
  }
  return 1;
}


// center, 3x3 rotation matrix [m1, m2, t], and half-sizes (a, b, L/2) of element e of a 1D
// box-section rod; returns 1 if handled, 0 otherwise
int mj_rodBoxElem(const mjModel* m, const mjData* d, int f, int e,
                  mjtNum pos[3], mjtNum mat[9], mjtNum size[3]) {
  if (!mj_rodBoxSize(m, f, size)) {
    return 0;
  }
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  const mjtNum* last_hinge = b + 1 + mjROD_HINGE_SIZE*(nhinge - 1);
  if ((int)last_hinge[1] == (int)b[1 + 0] || (int)last_hinge[2] == (int)b[1 + 0]) {
    return 0;
  }

  int nedge = nhinge + 1;
  int va = m->flex_vertadr[f];
  const mjtNum* xpos = d->flexvert_xpos + 3*va;
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + 2*e;
  int v0 = edata[0], v1 = edata[1];
  mju_add3(pos, xpos + 3*v0, xpos + 3*v1);
  mju_scl3(pos, pos, 0.5);
  size[2] = 0.5 * mju_dist3(xpos + 3*v0, xpos + 3*v1);

  int c_target = -1;
  mjtNum sgn = 1;
  for (int step = 0; step <= nedge; step++) {
    int k = (step == 0) ? e : (step - 1);
    if (k < 0 || k >= nedge) {
      continue;
    }
    int cv0 = (k < nhinge) ? (int)b[1 + mjROD_HINGE_SIZE*k + 0] : (int)last_hinge[1];
    int cv1 = (k < nhinge) ? (int)b[1 + mjROD_HINGE_SIZE*k + 1] : (int)last_hinge[2];
    if (cv0 == v0 && cv1 == v1) {
      c_target = k;
      sgn = 1;
      break;
    } else if (cv0 == v1 && cv1 == v0) {
      c_target = k;
      sgn = -1;
      break;
    }
  }
  if (c_target < 0) {
    return 0;
  }

  int root_clamped = rodRootClamped(m, f, b);
  mjtNum t_curr[3], u_curr[3], v_curr[3];
  if (!rodEdgeDirector(m, d, f, c_target, t_curr, u_curr, v_curr)) {
    return 0;
  }

  int jnt = rodEdgeTwistJnt(b, nhinge, root_clamped, c_target);
  mjtNum theta_c = (jnt >= 0) ? (d->qpos[m->jnt_qposadr[jnt]] - m->qpos0[m->jnt_qposadr[jnt]]) : 0;
  mjtNum ce = mju_cos(theta_c), se = mju_sin(theta_c);
  for (int r = 0; r < 3; r++) {
    mat[3*r + 0] = ce*u_curr[r] + se*v_curr[r];
    mat[3*r + 1] = sgn*(-se*u_curr[r] + ce*v_curr[r]);
    mat[3*r + 2] = sgn*t_curr[r];
  }
  return 1;
}


// twist hinge joint ID of element e of a 1D rod (-1 if none), and optional orientation sign (+-1)
int mj_rodEdgeTwistJnt(const mjModel* m, int f, int e, int* sgn_out) {
  if (sgn_out) {
    *sgn_out = 1;
  }
  if (m->flex_dim[f] != 1 || m->flex_bendingadr[f] < 0 || e < 0 || e >= m->flex_elemnum[f]) {
    return -1;
  }
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  if (nhinge <= 0) {
    return -1;
  }
  const mjtNum* last_hinge = b + 1 + mjROD_HINGE_SIZE*(nhinge - 1);
  int nedge = nhinge + 1;
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + 2*e;
  int v0 = edata[0], v1 = edata[1];
  int c_target = -1;
  for (int step = 0; step <= nedge; step++) {
    int k = (step == 0) ? e : (step - 1);
    if (k < 0 || k >= nedge) {
      continue;
    }
    int cv0 = (k < nhinge) ? (int)b[1 + mjROD_HINGE_SIZE*k + 0] : (int)last_hinge[1];
    int cv1 = (k < nhinge) ? (int)b[1 + mjROD_HINGE_SIZE*k + 1] : (int)last_hinge[2];
    if (cv0 == v0 && cv1 == v1) {
      c_target = k;
      if (sgn_out) {
        *sgn_out = 1;
      }
      break;
    } else if (cv0 == v1 && cv1 == v0) {
      c_target = k;
      if (sgn_out) {
        *sgn_out = -1;
      }
      break;
    }
  }
  if (c_target < 0) {
    return -1;
  }
  int va = m->flex_vertadr[f];
  int root_clamped =
      (m->flex_vertbodyid[va + (int)b[1 + 0]] == m->flex_vertbodyid[va + (int)b[1 + 1]]);
  return rodEdgeTwistJnt(b, nhinge, root_clamped, c_target);
}


// element of 1D flex f incident to its flex-local vertex v, -1 if none: for a DER rod chain the
// element of the chain edge leaving v (the last edge for the far end vertex), found through the
// chain order in flex_bending; the vertex and element orders of flexcomp grids make this O(1)
int mj_rodVertElem(const mjModel* m, int f, int v) {
  int elemnum = m->flex_elemnum[f];
  const int* elem = m->flex_elem + m->flex_elemdataadr[f];
  if (m->flex_dim[f] != 1 || v < 0) {
    return -1;
  }
  if (m->flex_bendingadr[f] >= 0) {
    const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
    int nhinge = (int)b[0];
    if (nhinge > 0 && b[1 + 3] < 0) {
      int nvert = nhinge + 2, k = -1;
      if (v < nvert && rodChainVert(b, nhinge, v) == v) {
        k = v;
      } else {
        for (int j = 0; j < nvert; j++) {
          if (rodChainVert(b, nhinge, j) == v) {
            k = j;
            break;
          }
        }
      }
      if (k >= 0) {
        int c = mjMIN(k, nhinge);
        int c0 = rodChainVert(b, nhinge, c), c1 = rodChainVert(b, nhinge, c + 1);
        for (int step = 0; step <= elemnum; step++) {
          int e = (step == 0) ? c : (step - 1);
          if (e >= elemnum) {
            continue;
          }
          const int* ed = elem + 2*e;
          if ((ed[0] == c0 && ed[1] == c1) || (ed[0] == c1 && ed[1] == c0)) {
            return e;
          }
        }
      }
    }
  }
  for (int e = 0; e < elemnum; e++) {
    if (elem[2*e] == v || elem[2*e + 1] == v) {
      return e;
    }
  }
  return -1;
}


// add rotational lever-arm Jacobian terms (axial twist roll, and the pitch/yaw vertex couple of the
// cross-section) for a 1D DER rod on side (0 or 1) of a contact
void mj_rodContactJac(const mjModel* m, const mjData* d, const mjContact* con, int side,
                      int NV, const int* chain, mjtNum* jacdifp, mjtNum* jacdifr) {
  int f = con->flex[side];
  if (f < 0 || !mj_rodHasTwist(m, f)) {
    return;
  }
  int elem = con->elem[side] >= 0 ? con->elem[side] : mj_rodVertElem(m, f, con->vert[side]);
  if (elem < 0) {
    return;
  }
  int v_other = (con->flex[1-side] == f ? con->vert[1-side] : -1);
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + 2*elem;
  if (edata[0] == v_other || edata[1] == v_other) {
    return;
  }

  int va = m->flex_vertadr[f];
  const mjtNum* x0 = d->flexvert_xpos + 3*(va + edata[0]);
  const mjtNum* x1 = d->flexvert_xpos + 3*(va + edata[1]);
  mjtNum t[3];
  mju_sub3(t, x1, x0);
  mjtNum le = mju_normalize3(t);
  if (le < mjMINVAL) {
    return;
  }

  mjtNum w0, w1;
  mjtNum dvec[3], r_perp[3];
  if (con->vert[side] >= 0) {
    w1 = (con->vert[side] == edata[1]) ? 1 : 0;
    w0 = 1 - w1;
    const mjtNum* x_vert = w1 ? x1 : x0;
    mju_sub3(dvec, con->pos, x_vert);
    if (con->geom[1-side] >= 0 && m->geom_type[con->geom[1-side]] == mjGEOM_PLANE) {
      mjtNum nsign = side ? 1 : -1;
      mju_addToScl3(dvec, con->frame, 0.5 * nsign * con->dist);
    }
  } else {
    mjtNum ap[3], p_axis[3];
    mju_sub3(ap, con->pos, x0);
    w1 = mju_clip(mju_dot3(ap, t) / le, 0, 1);
    w0 = 1 - w1;
    for (int c = 0; c < 3; c++) {
      p_axis[c] = w0*x0[c] + w1*x1[c];
    }
    mju_sub3(dvec, con->pos, p_axis);
  }
  mjtNum a = mju_dot3(dvec, t);
  mju_addScl3(r_perp, dvec, t, -a);

  mjtNum sgn = side ? 1 : -1;
  mjtNum t_cross_r[3];
  mji_cross(t_cross_r, t, r_perp);

  // 1. axial twist roll about edge tangent t from explicit twist DOF
  int sgn_chain = 1;
  int jnt_e = mj_rodEdgeTwistJnt(m, f, elem, &sgn_chain);
  if (jnt_e >= 0) {
    int dof_e = m->jnt_dofadr[jnt_e];
    int col_e = -1;
    if (!chain) {
      col_e = dof_e;
    } else {
      for (int c = 0; c < NV; c++) {
        if (chain[c] == dof_e) {
          col_e = c;
          break;
        }
      }
    }
    if (col_e >= 0) {
      mjtNum scl = sgn * sgn_chain;
      if (jacdifp) {
        jacdifp[0*NV + col_e] += scl * t_cross_r[0];
        jacdifp[1*NV + col_e] += scl * t_cross_r[1];
        jacdifp[2*NV + col_e] += scl * t_cross_r[2];
      }
      if (jacdifr) {
        jacdifr[0*NV + col_e] += scl * t[0];
        jacdifr[1*NV + col_e] += scl * t[1];
        jacdifr[2*NV + col_e] += scl * t[2];
      }
    }
  }

  // 2. pitch/yaw tangent rotation and local reference director parallel-transport sensitivity on
  // (x0, x1): the surface point at dvec = r_perp + a t from the axis moves with the element frame,
  // including (t x tdot) x dvec = -(dvec.tdot) t + a tdot (the axial arm a != 0 on spherical caps)
  int b0 = m->flex_vertbodyid[va + edata[0]];
  int b1 = m->flex_vertbodyid[va + edata[1]];
  if (b0 == b1) {
    return;
  }

  mjtNum g_psi[3] = {0, 0, 0};
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  const mjtNum* last_hinge = b + 1 + mjROD_HINGE_SIZE*(nhinge - 1);
  int nedge = nhinge + 1;
  int c_target = -1;
  for (int step = 0; step <= nedge; step++) {
    int k = (step == 0) ? elem : (step - 1);
    if (k < 0 || k >= nedge) {
      continue;
    }
    int cv0 = (k < nhinge) ? (int)b[1 + mjROD_HINGE_SIZE*k + 0] : (int)last_hinge[1];
    int cv1 = (k < nhinge) ? (int)b[1 + mjROD_HINGE_SIZE*k + 1] : (int)last_hinge[2];
    if ((cv0 == edata[0] && cv1 == edata[1]) || (cv0 == edata[1] && cv1 == edata[0])) {
      c_target = k;
      break;
    }
  }
  if (c_target >= 0) {
    // directors without a state are frozen at the current tangent (see rodChain): no sensitivity
    const mjtNum* t_s = rodEdgeState(m, d, f, c_target);
    if (t_s) {
      mjtNum t_chain[3] = {sgn_chain * t[0], sgn_chain * t[1], sgn_chain * t[2]};
      mjtNum cross_st[3];
      mji_cross(cross_st, t_s, t_chain);
      mjtNum denom = le * (1 + mju_dot3(t_s, t_chain));
      if (denom >= mjMINVAL) {
        mji_scl3(g_psi, cross_st, -1 / denom);
      }
    }
  }

  for (int k = 0; k < 2; k++) {
    mjtNum esgn = k ? 1 : -1;
    // mj_jacSum weighted the endpoint bodies by (w0, w1): vertex contacts lie on an endpoint of
    // their incident element (mj_rodVertElem), element contacts use the same barycentric weights
    mjtNum wk = k ? w1 : w0;
    int bk = k ? b1 : b0;
    const mjtNum* xk = k ? x1 : x0;
    mjtNum offset[3], arm_k[3];
    mju_sub3(offset, xk, d->subtree_com + 3*m->body_rootid[bk]);
    mju_sub3(arm_k, con->pos, xk);

    int bw = m->body_weldid[bk];
    int da = m->body_dofadr[bw] + m->body_dofnum[bw] - 1;
    int ci = NV - 1;
    while (da >= 0) {
      if (chain) {
        while (ci >= 0 && chain[ci] > da) {
          ci--;
        }
        if (ci < 0 || chain[ci] != da) {
          da = m->dof_parentid[da];
          continue;
        }
      } else {
        ci = da;
      }

      const mjtNum* cdof = d->cdof + 6*da;
      mjtNum tmp[3], jv[3];
      mji_cross(tmp, cdof, offset);
      mju_add3(jv, cdof + 3, tmp);
      mjtNum dpsi = sgn * esgn * mju_dot3(g_psi, jv);

      if (jacdifp) {
        // remove individual-vertex body rotation around xk added by mj_jacSum at con->pos
        if (wk != 0 && (cdof[0] != 0 || cdof[1] != 0 || cdof[2] != 0)) {
          mjtNum rot_arm[3];
          mji_cross(rot_arm, cdof, arm_k);
          jacdifp[0*NV + ci] -= sgn * wk * rot_arm[0];
          jacdifp[1*NV + ci] -= sgn * wk * rot_arm[1];
          jacdifp[2*NV + ci] -= sgn * wk * rot_arm[2];
        }
        mjtNum coeff_t = -sgn * esgn * (mju_dot3(dvec, jv) / le);
        mjtNum coeff_j = sgn * esgn * (a / le);
        jacdifp[0*NV + ci] += coeff_t * t[0] + coeff_j * jv[0] + dpsi * t_cross_r[0];
        jacdifp[1*NV + ci] += coeff_t * t[1] + coeff_j * jv[1] + dpsi * t_cross_r[1];
        jacdifp[2*NV + ci] += coeff_t * t[2] + coeff_j * jv[2] + dpsi * t_cross_r[2];
      }
      if (jacdifr) {
        if (wk != 0) {
          jacdifr[0*NV + ci] -= sgn * wk * cdof[0];
          jacdifr[1*NV + ci] -= sgn * wk * cdof[1];
          jacdifr[2*NV + ci] -= sgn * wk * cdof[2];
        }
        mjtNum t_cross_jv[3];
        mji_cross(t_cross_jv, t, jv);
        mjtNum scl = sgn * esgn / le;
        jacdifr[0*NV + ci] += scl * t_cross_jv[0] + dpsi * t[0];
        jacdifr[1*NV + ci] += scl * t_cross_jv[1] + dpsi * t[1];
        jacdifr[2*NV + ci] += scl * t_cross_jv[2] + dpsi * t[2];
      }

      da = m->dof_parentid[da];
    }
  }
}


// size of the time-parallel state of a 1D DER rod in d->plastic: 6*nedge director frames followed
// by nhinge unwrapped reference twists, or 0 if none
int mj_rodStateSize(const mjModel* m, int f) {
  if (m->flex_dim[f] != 1 || m->flex_plasticadr[f] < 0 || m->flex_bendingadr[f] < 0) {
    return 0;
  }
  int nhinge = (int)m->flex_bending[m->flex_bendingadr[f]];
  return 6 * (nhinge + 1) + nhinge;
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
