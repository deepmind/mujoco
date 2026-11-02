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

#include "engine/engine_sleep.h"

#include <stdio.h>
#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_core_util.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"


//-------------------------------- update ----------------------------------------------------------

// compute sleeping arrays from tree_asleep, if flg_staticawake is set treat static bodies as awake
void mj_updateSleepInit(const mjModel* m, mjData* d, int flg_staticawake) {
  int ntree = m->ntree, nbody = m->nbody, nv = m->nv;

  // input arrays
  const int* tree_asleep   = d->tree_asleep;  // sleep state source of truth
  const int* body_treeid   = m->body_treeid;
  const int* body_parentid = m->body_parentid;
  const int* body_rootid   = m->body_rootid;
  const int* body_mocapid  = m->body_mocapid;
  const int* dof_bodyid    = m->dof_bodyid;

  // output arrays
  int* tree_awake        = d->tree_awake;
  int* body_awake        = d->body_awake;
  int* dof_awake_ind     = d->dof_awake_ind;
  int* body_awake_ind    = d->body_awake_ind;
  int* parent_awake_ind  = d->parent_awake_ind;

  // tree_awake
  int ntree_awake = 0;
  for (int i=0; i < ntree; i++) {
    tree_awake[i] = tree_asleep[i] < 0;
    ntree_awake += tree_awake[i];
  }
  d->ntree_awake = ntree_awake;

  // {body,parent}_awake_ind
  int nbody_awake = 0;
  int nparent_awake = 0;
  for (int i=0; i < nbody; i++) {
    // static body
    if (body_treeid[i] < 0) {
      if (body_mocapid[body_rootid[i]] >= 0) {
        // mocap body are always awake
        body_awake[i] = mjS_AWAKE;
      } else {
        // mark static body unless flg_staticawake is set
        body_awake[i] = flg_staticawake ? mjS_AWAKE : mjS_STATIC;
      }
    }

    // dynamic body
    else {
      body_awake[i] = tree_awake[body_treeid[i]] ? mjS_AWAKE : mjS_ASLEEP;
    }

    // body_awake_ind: list of awake and static bodies
    if (body_awake[i] != mjS_ASLEEP) {
      body_awake_ind[nbody_awake++] = i;
    }

    // parent_awake_ind: list of bodies with awake or static parents
    if (i && body_awake[body_parentid[i]] != mjS_ASLEEP) {
      parent_awake_ind[nparent_awake++] = i;
    }
  }
  d->nbody_awake = nbody_awake;
  d->nparent_awake = nparent_awake;

  // dof_awake_ind: list of awake degrees of freedom
  int nv_awake = 0;
  for (int i=0; i < nv; i++) {
    int bodyid = dof_bodyid[i];
    if (body_treeid[bodyid] >= 0 && body_awake[bodyid] == mjS_AWAKE) {
      dof_awake_ind[nv_awake++] = i;
    }
  }
  d->nv_awake = nv_awake;
}

// compute sleep arrays from tree_asleep
void mj_updateSleep(const mjModel* m, mjData* d) {
  mj_updateSleepInit(m, d, /*flg_staticawake*/0);
}


//-------------------------------- utilities -------------------------------------------------------

// return 1 if the weighted infinity norm of vec is smaller than tol, 0 otherwise
static int isSmaller(const mjtNum* vec, const mjtNum* weight, int n, mjtNum tol) {
  mjtNum max = 0;
  for (int i=0; i < n; i++) {
    max = mju_max(max, weight[i] * mju_abs(vec[i]));
    if (max >= tol) {
      return 0;
    }
  }
  return 1;
}


// return 1 if tree i can sleep, 0 otherwise
static int treeCanSleep(const mjModel* m, const mjData* d, int i, mjtNum tol) {
  // check sleep policy
  if (m->tree_sleep_policy[i] == mjSLEEP_NEVER ||
      m->tree_sleep_policy[i] == mjSLEEP_AUTO_NEVER) {
    return 0;
  }

  // check xfrc_applied
  int adr = m->tree_bodyadr[i];
  int num = m->tree_bodynum[i];
  if (!mju_isZeroByte((const unsigned char*)(d->xfrc_applied+6*adr), 6*num*sizeof(mjtNum))) {
    return 0;
  }

  // check qfrc_applied
  adr = m->tree_dofadr[i];
  num = m->tree_dofnum[i];
  if (!mju_isZeroByte((const unsigned char*)(d->qfrc_applied+adr), num*sizeof(mjtNum))) {
    return 0;
  }

  // check qvel
  if (tol) {
    return isSmaller(d->qvel+adr, m->dof_length+adr, num, tol);
  } else {
    return mju_isZeroByte((const unsigned char*)(d->qvel+adr), num*sizeof(mjtNum));
  }
}


// return the first tree in the sleep cycle that starts at i, -1 if error
int mj_sleepCycle(const int* tree_asleep, int ntree, int i) {
  if (i < 0 || i >= ntree) {
    return -1;  // index i out of bounds
  }

  int smallest = i;
  int current = i;
  int count = 0;

  do {
    if (count > ntree) {
      return -1;  // cycle detection failed (too many steps)
    }

    int next = tree_asleep[current];

    if (next < 0 || next >= ntree) {
      return -1;  // next index out of bounds
    }

    if (next < smallest) {
      smallest = next;
    }

    current = next;
    count++;
  } while (current != i);

  return smallest;
}


//-------------------------------- wake ------------------------------------------------------------

// helper for pluralizing in log messages
static inline const char* plural(int n) {
  return n > 1 ? "s" : "";
}

#ifndef MJ_DISABLE_DEBUG_TRACING
// append formatted text to buf[pos..sizeof(buf)-1], clamping pos on truncation
#define MJ_APPEND_BUF(buf, pos, ...)                                        \
  do {                                                                      \
    if ((pos) < (int)sizeof(buf) - 1) {                                     \
      int n_ = snprintf((buf) + (pos), sizeof(buf) - (pos), __VA_ARGS__);   \
      if (n_ > 0) {                                                         \
        (pos) = mjMIN((int)sizeof(buf) - 1, (pos) + n_);                    \
      }                                                                     \
    }                                                                       \
  } while (0)
#endif


// wake tree i and its associated cycle, return number of woke trees
int mj_wakeIsland(int* tree_asleep, int ntree, int i, int wakeval, const char* reason, mjtNum time) {
  int nwoke = 0;

  // i is invalid; SHOULD NOT OCCUR
  if (i < 0 || i >= ntree) {
    mjERROR("invalid tree %d", i);
    return nwoke;
  }

  // tree i already awake: set to wakeval if larger than current value
  int asleep_val = tree_asleep[i];
  if (asleep_val < 0) {
    tree_asleep[i] = mjMIN(wakeval, asleep_val);
    return nwoke;
  }

  // tree i asleep: wake up tree and its island cycle
  else {
    int current = i;
    int woke_trees[1024];  // buffer for woke tree indices
    do {
      // get the index of the next tree in the cycle
      int next = tree_asleep[current];

      // next is invalid; SHOULD NOT OCCUR
      if (next < 0 || next >= ntree) {
        mjERROR("invalid sleep state index %d when waking tree %d", next, i);
        return 0;
      }

      // wake the current tree, record index, increment and advance to next
      tree_asleep[current] = wakeval;
      if (nwoke < 1024) woke_trees[nwoke] = current;
      nwoke++;
      current = next;
    } while (current != i && nwoke < ntree);

    // did not come back to tree i, not a cycle; SHOULD NOT OCCUR
    if (current != i) {
      mjERROR("tree %d is not in a cycle", i);
      return 0;
    }

#ifndef MJ_DISABLE_DEBUG_TRACING
    if (reason && mju_isTopicEnabled(mjTOPIC_SLEEP)) {
      int nprint = mjMIN(nwoke, 1024);
      char buf[1024] = "";
      int pos = 0;
      MJ_APPEND_BUF(buf, pos, "t=%6.3g, woke due to %s tree%s ", time, reason, plural(nprint));
      for (int j = 0; j < nprint; j++) {
        MJ_APPEND_BUF(buf, pos, "%d%s", woke_trees[j], (j == nprint - 1) ? "" : " ");
      }
      mjLogMessage msg = {.level = mjLOG_DEBUG, .topic = mjTOPIC_SLEEP, .func = __func__};
      mju_strncpy(msg.subject, buf, sizeof(msg.subject));
      mju_message(&msg);
    }
#endif
  }

  return nwoke;
}


static int kAwake = -(1+mjMINAWAKE);  // tree_asleep value for fully awake tree


// under non-metric integrators, constraint-free flexes cannot sleep: wake sleeping trees and
// reset awake countdowns in a single O(nflexvert) pass
static int wakeNonMetricFlex(const mjModel* m, mjData* d, int flg_wake) {
  int nwoke = 0;
  if (mj_isMetric(m)) {
    return 0;
  }
  for (int f = 0; f < m->nflex; f++) {
    if (m->flex_edgeequality[f] || !mj_effFlexPossible(m, f)) {
      continue;
    }
    int num = m->flex_interp[f] ? m->flex_nodenum[f] : m->flex_vertnum[f];
    const int* bodyid = m->flex_interp[f] ? m->flex_nodebodyid + m->flex_nodeadr[f]
                                          : m->flex_vertbodyid + m->flex_vertadr[f];
    for (int j = 0; j < num; j++) {
      int t = m->body_treeid[bodyid[j]];
      if (t < 0) {
        continue;
      }
      if (d->tree_asleep[t] >= 0) {
        if (flg_wake) {
          nwoke += mj_wakeIsland(d->tree_asleep, m->ntree, t, kAwake, "perturbation", d->time);
        }
      } else {
        d->tree_asleep[t] = kAwake;
      }
    }
  }
  return nwoke;
}


// wake sleeping trees due to changes by user, return number of woke trees
int mj_wake(const mjModel* m, mjData* d) {
  int ntree = m->ntree, nwoke = 0;

  // sleep disabled
  if (!mjENABLED(mjENBL_SLEEP)) {
    // sleep disabled but some trees still asleep: wake all
    if (d->ntree_awake < ntree) {
      mju_fillInt(d->tree_asleep, kAwake, ntree);
    }
    return ntree - d->ntree_awake;
  }

  // wake sleeping passive flexes if integrator is non-metric
  if (d->ntree_awake < ntree) {
    nwoke += wakeNonMetricFlex(m, d, /*flg_wake=*/1);
  }

  // sweep over trees, wake if required
  for (int i=0; i < ntree; i++) {
    int asleep = d->tree_asleep[i] >= 0;

    // awake: nothing to do
    if (!asleep) {
      continue;
    }

    // if qpos mismatch or cannot sleep: wake up
    if (d->tree_awake[i] || !treeCanSleep(m, d, i, 0)) {
      nwoke += mj_wakeIsland(d->tree_asleep, ntree, i, kAwake, "perturbation", d->time);
    }
  }

  return nwoke;
}


// get a representative body from a flex contact side
int mj_flexBody(const mjModel* m, const mjContact* con, int side) {
  int f = con->flex[side];

  // flex vertex contact (non-interpolated)
  if (con->vert[side] >= 0 && m->flex_interp[f] == 0) {
    return m->flex_vertbodyid[m->flex_vertadr[f] + con->vert[side]];
  }

  // flex element contact (non-interpolated): return first dynamic vertex body if any
  if (con->elem[side] >= 0 && m->flex_interp[f] == 0) {
    int dim = m->flex_dim[f];
    const int* edata = m->flex_elem + m->flex_elemdataadr[f] + con->elem[side]*(dim+1);
    const int* vertbodyid = m->flex_vertbodyid + m->flex_vertadr[f];
    for (int j = 0; j <= dim; j++) {
      int b = vertbodyid[edata[j]];
      if (m->body_treeid[b] >= 0) {
        return b;
      }
    }
    return vertbodyid[edata[0]];
  }

  // interpolated flex: return the first dynamic node body if any, else first node body
  int nodenum = m->flex_nodenum[f];
  const int* nodebodyid = m->flex_nodebodyid + m->flex_nodeadr[f];
  for (int j = 0; j < nodenum; j++) {
    int b = nodebodyid[j];
    if (m->body_treeid[b] >= 0) {
      return b;
    }
  }
  return nodebodyid[0];
}


// wake sleeping trees with collision contact, return number of woke trees
int mj_wakeCollision(const mjModel* m, mjData* d) {
  int ntree = m->ntree, ncon = d->ncon, nwoke = 0;

  if (!mjENABLED(mjENBL_SLEEP)) {
    return nwoke;
  }

  // sweep over contacts, wake trees if required
  for (int i=0; i < ncon; i++) {
    const mjContact* con = d->contact + i;

    // resolve body on each side
    int b1 = con->geom[0] >= 0 ? m->geom_bodyid[con->geom[0]] : mj_flexBody(m, con, 0);
    int b2 = con->geom[1] >= 0 ? m->geom_bodyid[con->geom[1]] : mj_flexBody(m, con, 1);

    int tree1 = m->body_treeid[b1];
    int tree2 = m->body_treeid[b2];

    // contact with a dof-less body: wake if it is marked awake (mocap), otherwise nothing to do
    if (tree1 < 0 || tree2 < 0) {
      int tree = tree1 < 0 ? tree2 : tree1;
      int b = tree1 < 0 ? b1 : b2;
      if (tree >= 0 && !d->tree_awake[tree] && d->body_awake[b] == mjS_AWAKE) {
        nwoke += mj_wakeIsland(d->tree_asleep, ntree, tree, kAwake, "mocap contact with", d->time);
      }
      continue;
    }

    int awake1 = d->tree_awake[tree1];
    int awake2 = d->tree_awake[tree2];

    // both trees awake, nothing to do
    if (awake1 && awake2) {
      continue;
    }

    // both trees asleep; SHOULD NOT OCCUR
    if (!awake1 && !awake2) {
      mjERROR("contact between sleeping bodies %d and %d", b1, b2);
    }

    // wake sleeping tree
    int sleeping_tree = awake1 ? tree2 : tree1;
    int wakeval = awake1 ? d->tree_asleep[tree1] : d->tree_asleep[tree2];
    nwoke += mj_wakeIsland(d->tree_asleep, ntree, sleeping_tree, wakeval, "contact", d->time);
  }

  return nwoke;
}


// wake sleeping trees with a constrained or metric tendon to a waking tree
// return number of woke trees
int mj_wakeTendon(const mjModel* m, mjData* d) {
  int ntendon = m->ntendon, nwoke = 0;

  if (!mjENABLED(mjENBL_SLEEP)) {
    return nwoke;
  }

  // sweep over tendons, wake trees if required
  for (int i=0; i < ntendon; i++) {
    // skip tendons with less than two trees
    int treenum = m->tendon_treenum[i];
    if (treenum < 2) {
      continue;
    }

    // skip tendons that are not constrained or do not affect the discrete metric
    int has_limit = !mjDISABLED(mjDSBL_CONSTRAINT) && !mjDISABLED(mjDSBL_LIMIT) &&
                    (treenum == 2) && tendonLimit(m, d->ten_length, i);
    int has_friction = !mjDISABLED(mjDSBL_CONSTRAINT) && !mjDISABLED(mjDSBL_FRICTIONLOSS) &&
                       (treenum == 2) && (m->tendon_frictionloss[i] > 0);
    int is_constraint = has_limit || has_friction;
    int is_metric = mj_isMetric(m) && mj_effTendonPossible(m, i);
    if (!is_constraint && !is_metric) {
      continue;
    }

    // two-tree tendon
    if (treenum == 2) {
      int tree1 = m->tendon_treeid[2*i];
      int tree2 = m->tendon_treeid[2*i+1];
      int awake1 = d->tree_awake[tree1];
      int awake2 = d->tree_awake[tree2];
      if (awake1 != awake2) {
        int sleeping_tree = awake1 ? tree2 : tree1;
        int wakeval = awake1 ? d->tree_asleep[tree1] : d->tree_asleep[tree2];
        const char* reason = is_constraint ? "tendon constraint" : "tendon metric";
        nwoke += mj_wakeIsland(d->tree_asleep, m->ntree, sleeping_tree, wakeval, reason, d->time);
      }
    }

    // multi-tree tendon (> 2 trees): wake all sleeping trees if any tree is awake
    else {
      int any_awake = 0, wakeval = 0;
      int start = m->ten_J_rowadr[i];
      int end = start + m->ten_J_rownnz[i];
      for (int j=start; j < end; j++) {
        int tree = m->dof_treeid[m->ten_J_colind[j]];
        if (d->tree_awake[tree]) {
          any_awake = 1;
          wakeval = mjMIN(wakeval, d->tree_asleep[tree]);
        }
      }
      if (any_awake) {
        for (int j=start; j < end; j++) {
          int tree = m->dof_treeid[m->ten_J_colind[j]];
          if (d->tree_asleep[tree] >= 0) {
            nwoke += mj_wakeIsland(d->tree_asleep, m->ntree, tree, wakeval, "tendon metric",
                                   d->time);
          }
        }
      }
    }
  }

  return nwoke;
}


// wake sleeping trees with an equality to a waking tree, return number of woke trees
int mj_wakeEquality(const mjModel* m, mjData* d) {
  int neq = m->neq, nwoke = 0;

  if (!mjENABLED(mjENBL_SLEEP)) {
    return nwoke;
  }

  // sweep over equalities, wake trees if required
  for (int i=0; i < neq; i++) {
    // skip inactive
    if (!d->eq_active[i]) continue;

    mjtEq eqtype = m->eq_type[i];
    int id1 = m->eq_obj1id[i];
    int id2 = m->eq_obj2id[i];
    int tree1, tree2;
    int body1 = -1, body2 = -1;

    switch (eqtype) {
    case mjEQ_CONNECT:
    case mjEQ_WELD:
      if (m->eq_objtype[i] == mjOBJ_BODY) {
        body1 = id1;
        body2 = id2;
      } else {
        body1 = m->site_bodyid[id1];
        body2 = m->site_bodyid[id2];
      }
      tree1 = m->body_treeid[body1];
      tree2 = m->body_treeid[body2];
      break;

    case mjEQ_JOINT:
      body1 = id1 >= 0 ? m->jnt_bodyid[id1] : -1;
      body2 = id2 >= 0 ? m->jnt_bodyid[id2] : -1;
      tree1 = body1 >= 0 ? m->body_treeid[body1] : -1;
      tree2 = body2 >= 0 ? m->body_treeid[body2] : -1;
      break;

    case mjEQ_TENDON:
      mjERROR("tendon equality does not yet support sleeping");
      continue;

    case mjEQ_FLEX:
    case mjEQ_FLEXSTRAIN: {
      int f = id1;
      int num, adr;
      const int* bodyid;
      if (m->flex_interp[f]) {
        num = m->flex_nodenum[f];
        adr = m->flex_nodeadr[f];
        bodyid = m->flex_nodebodyid;
      } else {
        num = m->flex_vertnum[f];
        adr = m->flex_vertadr[f];
        bodyid = m->flex_vertbodyid;
      }

      // check if any dynamic tree is awake and track most-awake countdown
      int any_awake = 0, wakeval = 0;
      for (int j = 0; j < num; j++) {
        int treeid = m->body_treeid[bodyid[adr+j]];
        if (treeid >= 0 && d->tree_awake[treeid]) {
          any_awake = 1;
          wakeval = mjMIN(wakeval, d->tree_asleep[treeid]);
        }
      }

      // wake all sleeping islands spanned by the flex
      if (any_awake) {
        for (int j = 0; j < num; j++) {
          int treeid = m->body_treeid[bodyid[adr+j]];
          if (treeid >= 0 && d->tree_asleep[treeid] >= 0) {
            nwoke += mj_wakeIsland(d->tree_asleep, m->ntree, treeid, wakeval,
                                   "flex equality", d->time);
          }
        }
      }
      continue;
    }

    default:
      continue;
    }

    // get sleep state; dof-less bodies marked awake (mocap) count as awake
    mjtSleepState s1 = tree1 >= 0 ? (mjtSleepState)d->tree_awake[tree1]
                                  : (body1 >= 0 ? (mjtSleepState)d->body_awake[body1] : mjS_STATIC);
    mjtSleepState s2 = tree2 >= 0 ? (mjtSleepState)d->tree_awake[tree2]
                                  : (body2 >= 0 ? (mjtSleepState)d->body_awake[body2] : mjS_STATIC);

    // neither is asleep, nothing to do
    if (s1 != mjS_ASLEEP && s2 != mjS_ASLEEP) {
      continue;
    }

    // one is static, nothing to do
    if (s1 == mjS_STATIC || s2 == mjS_STATIC) {
      continue;
    }

    // equality within the same tree, nothing to do
    if (tree1 == tree2) {
      continue;
    }

    // both are asleep, wake if in different islands
    if (s1 == mjS_ASLEEP && s2 == mjS_ASLEEP) {
      int cycle1 = mj_sleepCycle(d->tree_asleep, m->ntree, tree1);
      int cycle2 = mj_sleepCycle(d->tree_asleep, m->ntree, tree2);
      if (cycle1 != cycle2) {
        int nwoke1 = mj_wakeIsland(d->tree_asleep, m->ntree, tree1, kAwake, "equality", d->time);
        int nwoke2 = mj_wakeIsland(d->tree_asleep, m->ntree, tree2, kAwake, "equality", d->time);

        nwoke += nwoke1 + nwoke2;
      }
      continue;
    }

    // one is asleep and one is awake, wake the sleeping tree
    int sleeping_tree = s1 == mjS_ASLEEP ? tree1 : tree2;
    nwoke += mj_wakeIsland(d->tree_asleep, m->ntree, sleeping_tree, kAwake, "equality", d->time);
  }

  return nwoke;
}


//-------------------------------- sleep -----------------------------------------------------------

// put n trees to sleep (create cycle), set their velocity and acceleration to zero
static inline void mj_sleepTrees(const mjModel* m, mjData* d, const int* tree, int n) {
  for (int i=0; i < n; i++) {
    // create cycle
    int current = tree[i];
    int next = (i == n - 1) ? tree[0] : tree[i + 1];
    if (d->tree_asleep[current] == -1) {
      d->tree_asleep[current] = next;
    }

    // SHOULD NOT OCCUR
    else if (d->tree_asleep[current] >= 0) {
      mjERROR("trying to sleep tree %d which is already asleep", i);
    } else {
      mjERROR("trying to sleep tree %d which is not ready to sleep", i);
    }

    // set tree velocity and acceleration to zero
    int adr = m->tree_dofadr[current];
    int num = m->tree_dofnum[current];
    mju_zero(d->qvel+adr, num);
    mju_zero(d->qacc+adr, num);
    mju_zero(d->cdof_dot+6*adr, 6*num);

    int badr = m->tree_bodyadr[current];
    int bnum = m->tree_bodynum[current];
    mju_zero(d->cvel+6*badr, 6*bnum);
  }

#ifndef MJ_DISABLE_DEBUG_TRACING
  if (mju_isTopicEnabled(mjTOPIC_SLEEP)) {
    char buf[1024] = "";
    int pos = 0;
    MJ_APPEND_BUF(buf, pos, "t=%6.2g, slept tree%s ", d->time, plural(n));
    for (int i = 0; i < n; i++) {
      MJ_APPEND_BUF(buf, pos, "%d%s", tree[i], (i == n - 1) ? "" : " ");
    }
    mjLogMessage msg = {.level = mjLOG_DEBUG, .topic = mjTOPIC_SLEEP, .func = __func__};
    mju_strncpy(msg.subject, buf, sizeof(msg.subject));
    mju_message(&msg);
  }
#endif
}


// put trees to sleep according to tolerance, return number of slept trees
int mj_sleep(const mjModel* m, mjData* d) {
  int ntree = m->ntree, nisland = d->nisland, nslept = 0;

  // sleep disabled: nothing to do
  if (!mjENABLED(mjENBL_SLEEP)) {
    return nslept;
  }

  // have constraints but no island structure: can't sleep
  if (d->nefc && !nisland) {
    return nslept;
  }

  // sweep over awake trees, increment tree_asleep if under tolerance
  for (int i=0; i < ntree; i++) {
    // skip sleeping tree
    if (d->tree_asleep[i] >= 0) {
      continue;
    }

    // increment tree_asleep if tree can sleep, otherwise wake up
    if (treeCanSleep(m, d, i, m->opt.sleep_tolerance)) {
      d->tree_asleep[i] += (d->tree_asleep[i] < -1);
    } else {
      d->tree_asleep[i] = -(1+mjMINAWAKE);
    }
  }

  // under non-metric integrators, prevent constraint-free flex trees from sleeping
  wakeNonMetricFlex(m, d, /*flg_wake=*/0);

  // sweep over islands, put to sleep if all trees are under tolerance
  for (int i=0; i < nisland; i++) {
    // check if all trees in the island can sleep
    int can_sleep = 1;
    int start = d->island_itreeadr[i];
    int end = start + d->island_ntree[i];
    for (int j=start; j < end; j++) {
      int tree_asleep = d->tree_asleep[d->map_itree2tree[j]];
      if (tree_asleep < -1) {
        can_sleep = 0;
        break;
      }

      // sleeping tree in an island; SHOULD NOT OCCUR
      else if (tree_asleep >= 0) {
        mjERROR("found sleeping tree %d in island %d", d->map_itree2tree[j], i);
      }
    }

    // put island to sleep
    if (can_sleep) {
      const int* tree = d->map_itree2tree + start;
      int n = d->island_ntree[i];
      mj_sleepTrees(m, d, tree, n);
      nslept += n;
    }
  }

  // sleep unconstrained trees (with or without island structure)
  int start = nisland ? d->island_itreeadr[nisland-1] + d->island_ntree[nisland-1] : 0;
  for (int j=start; j < ntree; j++) {
    int i = nisland ? d->map_itree2tree[j] : j;
    if (d->tree_asleep[i] == -1) {
      mj_sleepTrees(m, d, &i, 1);
      nslept++;
    }
  }

  return nslept;
}


//-------------------------------- sleep state -----------------------------------------------------

// return sleep state of tendon i
static mjtSleepState mj_tendonSleepState(const mjModel* m, const mjData* d, int i) {
  int treenum = m->tendon_treenum[i];

  // no trees: tendon is static
  if (treenum == 0) {
    return mjS_STATIC;
  }

  // single tree: awake if tree is awake, asleep otherwise
  int id1 = m->tendon_treeid[2*i];
  if (treenum == 1) {
    return d->tree_awake[id1] ? mjS_AWAKE : mjS_ASLEEP;
  }

  // two trees: asleep only if both are asleep
  int id2 = m->tendon_treeid[2*i+1];
  if (treenum == 2) {
    return (d->tree_awake[id1] || d->tree_awake[id2]) ? mjS_AWAKE : mjS_ASLEEP;
  }

  return mjS_AWAKE;
}


// return sleep state of actuator i
static mjtSleepState mj_actuatorSleepState(const mjModel* m, const mjData* d, int i) {
  mjtSleepState s1, s2;
  int trnid = m->actuator_trnid[i*2];

  switch ((mjtTrn)m->actuator_trntype[i]) {
  case mjTRN_JOINT:
  case mjTRN_JOINTINPARENT:
    return mj_sleepState(m, d, mjOBJ_JOINT, trnid);

  case mjTRN_SLIDERCRANK:
    s1 = mj_sleepState(m, d, mjOBJ_SITE, trnid);
    s2 = mj_sleepState(m, d, mjOBJ_SITE, m->actuator_trnid[i*2+1]);
    return (s1 == mjS_AWAKE || s2 == mjS_AWAKE) ? mjS_AWAKE : mjS_ASLEEP;

  case mjTRN_TENDON:
    return mj_tendonSleepState(m, d, trnid);

  case mjTRN_SITE:
    if (m->actuator_trnid[i*2+1] == -1) {
      return mj_sleepState(m, d, mjOBJ_SITE, trnid);
    }
    s1 = mj_sleepState(m, d, mjOBJ_SITE, trnid);
    s2 = mj_sleepState(m, d, mjOBJ_SITE, m->actuator_trnid[i*2+1]);
    if (s1 == mjS_AWAKE || s2 == mjS_AWAKE) {
      return mjS_AWAKE;
    }
    return (s1 == mjS_ASLEEP || s2 == mjS_ASLEEP) ? mjS_ASLEEP : mjS_STATIC;

  case mjTRN_SO3:
    // ball joint target or site + refsite target
    if (m->actuator_trnid[i*2+1] == -1) {
      return mj_sleepState(m, d, mjOBJ_JOINT, trnid);
    }
    s1 = mj_sleepState(m, d, mjOBJ_SITE, trnid);
    s2 = mj_sleepState(m, d, mjOBJ_SITE, m->actuator_trnid[i*2+1]);
    return (s1 == mjS_AWAKE || s2 == mjS_AWAKE) ? mjS_AWAKE : mjS_ASLEEP;

  case mjTRN_BODY:
    return mj_sleepState(m, d, mjOBJ_BODY, trnid);

  case mjTRN_UNDEFINED:
    return mjS_AWAKE;
  }

  return mjS_AWAKE;
}


// return sleep state of equality i
static mjtSleepState mj_equalitySleepState(const mjModel* m, const mjData* d, int i) {
  mjtEq eqtype = m->eq_type[i];
  mjtObj objtype;

  switch (eqtype) {
    case mjEQ_CONNECT:
    case mjEQ_WELD:
      objtype = m->eq_objtype[i];
      break;
    case mjEQ_JOINT:
      objtype = mjOBJ_JOINT;
      break;
    case mjEQ_TENDON:
      objtype = mjOBJ_TENDON;
      break;
    case mjEQ_FLEX:
    case mjEQ_FLEXSTRAIN:
      objtype = mjOBJ_FLEX;
      break;
    default:
      return mjS_AWAKE;
  }

  int id1 = m->eq_obj1id[i];
  int id2 = m->eq_obj2id[i];
  mjtSleepState s1 = (id1 >= 0) ? mj_sleepState(m, d, objtype, id1) : mjS_STATIC;
  mjtSleepState s2 = (id2 >= 0) ? mj_sleepState(m, d, objtype, id2) : mjS_STATIC;

  // return ASLEEP if both objects are asleep or static, AWAKE otherwise
  int neither_awake = (s1 != mjS_AWAKE && s2 != mjS_AWAKE);
  return neither_awake ? mjS_ASLEEP : mjS_AWAKE;
}


// return sleep state of kinematic subtree rooted at body i
static mjtSleepState mj_subtreeSleepState(const mjModel* m, const mjData* d, int i) {
  // dynamic or mocap root shares sleep state with its entire subtree
  if (d->body_awake[i] != mjS_STATIC) {
    return (mjtSleepState) d->body_awake[i];
  }

  // bodies are in depth-first order: the subtree is contiguous after i
  int has_asleep = 0;
  for (int b = i + 1; b < m->nbody; b++) {
    int p = b;
    while (p > i) {
      p = m->body_parentid[p];
    }
    if (p != i) {
      break;
    }
    if (d->body_awake[b] == mjS_AWAKE) {
      return mjS_AWAKE;
    }
    if (d->body_awake[b] == mjS_ASLEEP) {
      has_asleep = 1;
    }
  }
  return has_asleep ? mjS_ASLEEP : mjS_STATIC;
}


// return sleep state of sensor i (AWAKE or ASLEEP, never STATIC)
static mjtSleepState mj_sensorSleepState(const mjModel* m, const mjData* d, int i) {
  mjtSensor type = m->sensor_type[i];
  mjtObj objtype = m->sensor_objtype[i];
  mjtObj reftype = m->sensor_reftype[i];
  int objid = m->sensor_objid[i];
  int refid = m->sensor_refid[i];

  // special handling for specific sensor types
  switch (type) {

  // USER and PLUGIN sensors: always awake
  case mjSENS_USER:
  case mjSENS_PLUGIN:
    return mjS_AWAKE;

  // contact sensors with site specifiers: always awake
  case mjSENS_CONTACT:
    // site used to define a volume: always awake
    if (objtype == mjOBJ_SITE || reftype == mjOBJ_SITE) {
      return mjS_AWAKE;
    }
    break;

  // rangefinder output does not depend on sleep state: always awake
  case mjSENS_RANGEFINDER:
    return mjS_AWAKE;

  // tactile sensor: sleep iff parent geom is asleep (static geom stays awake)
  case mjSENS_TACTILE:
    return mj_sleepState(m, d, reftype, refid) == mjS_ASLEEP ? mjS_ASLEEP : mjS_AWAKE;

  default:
    break;
  }

  // get sleep state of the primary and reference objects
  mjtSleepState s_obj;
  if ((type == mjSENS_CONTACT && objtype == mjOBJ_XBODY) ||
      (type == mjSENS_INSIDESITE && objtype == mjOBJ_XBODY) ||
      (type == mjSENS_INSIDESITE && objtype == mjOBJ_BODY && objid > 0 &&
       m->body_mass[objid] < mjMINVAL && m->body_subtreemass[objid] >= mjMINVAL)) {
    s_obj = mj_subtreeSleepState(m, d, objid);
  } else {
    s_obj = mj_sleepState(m, d, objtype, objid);
  }

  mjtSleepState s_ref;
  if (type == mjSENS_CONTACT && reftype == mjOBJ_XBODY) {
    s_ref = mj_subtreeSleepState(m, d, refid);
  } else {
    s_ref = mj_sleepState(m, d, reftype, refid);
  }

  // special handling for UNKNOWN objects

  // if both are UNKNOWN, return AWAKE
  if (objtype == mjOBJ_UNKNOWN && reftype == mjOBJ_UNKNOWN) {
    return mjS_AWAKE;
  }

  // if one is UNKNOWN, return the other's sleep state (if STATIC, return AWAKE)
  if (objtype == mjOBJ_UNKNOWN) {
    return s_ref == mjS_ASLEEP ? mjS_ASLEEP : mjS_AWAKE;
  } else if (reftype == mjOBJ_UNKNOWN) {
    return s_obj == mjS_ASLEEP ? mjS_ASLEEP : mjS_AWAKE;
  }

  // if either object is awake, return AWAKE
  if (s_obj == mjS_AWAKE || s_ref == mjS_AWAKE) {
    return mjS_AWAKE;
  }

  // otherwise return ASLEEP
  return mjS_ASLEEP;
}


// return sleep state of object i
mjtSleepState mj_sleepState(const mjModel* m, const mjData* d, mjtObj type, int i) {
  const char* typename;

  switch (type) {

  // simple types
  case mjOBJ_BODY:
  case mjOBJ_XBODY:
    return (mjtSleepState) d->body_awake[i];
  case mjOBJ_JOINT:
    return (mjtSleepState) d->body_awake[m->jnt_bodyid[i]];
  case mjOBJ_SITE:
    return (mjtSleepState) d->body_awake[m->site_bodyid[i]];
  case mjOBJ_DOF:
    return (mjtSleepState) d->body_awake[m->dof_bodyid[i]];
  case mjOBJ_GEOM:
    return (mjtSleepState) d->body_awake[m->geom_bodyid[i]];
  case mjOBJ_CAMERA:
    return (mjtSleepState) d->body_awake[m->cam_bodyid[i]];
  case mjOBJ_LIGHT:
    return (mjtSleepState) d->body_awake[m->light_bodyid[i]];
  case mjOBJ_MESH:
    return mjS_STATIC;

  // complex types
  case mjOBJ_EQUALITY:
    return mj_equalitySleepState(m, d, i);
  case mjOBJ_TENDON:
    return mj_tendonSleepState(m, d, i);
  case mjOBJ_ACTUATOR:
    return mj_actuatorSleepState(m, d, i);
  case mjOBJ_SENSOR:
    return mj_sensorSleepState(m, d, i);

  case mjOBJ_FLEX: {
    // scan all dynamic bodies: awake if any is awake, asleep if all are asleep, else static
    int num, adr;
    const int* bodyid;
    if (m->flex_interp[i]) {
      num = m->flex_nodenum[i];
      adr = m->flex_nodeadr[i];
      bodyid = m->flex_nodebodyid;
    } else {
      num = m->flex_vertnum[i];
      adr = m->flex_vertadr[i];
      bodyid = m->flex_vertbodyid;
    }
    int has_asleep = 0;
    for (int j = 0; j < num; j++) {
      int b = bodyid[adr+j];
      if (m->body_treeid[b] >= 0) {
        if (d->body_awake[b] == mjS_AWAKE) {
          return mjS_AWAKE;
        }
        has_asleep = 1;
      }
    }
    return has_asleep ? mjS_ASLEEP : mjS_STATIC;
  }

  // undefined sleep state, return AWAKE
  case mjOBJ_UNKNOWN:
    return mjS_AWAKE;

  // unsupported
  default:
    typename = mju_type2Str(type);
    if (typename) {
      mjERROR("unsupported object type '%s'", typename);
    } else {
      mjERROR("unsupported object type %d", type);
    }
    return mjS_AWAKE;
  }
}
