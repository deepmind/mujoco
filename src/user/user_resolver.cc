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

#include "user/user_resolver.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "user/user_api.h"

namespace mujoco {
namespace {

// format a numeric value for conflict messages
template <typename T>
std::string fmtVal(T val) {
  if constexpr (std::is_floating_point_v<T>) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", val);
    return buf;
  } else {
    return std::to_string(val);
  }
}

// format an array of numeric values for conflict messages
std::string fmtArr(const mjtNum* val, int n) {
  std::string s;
  for (int i = 0; i < n; i++) {
    if (i) s += ' ';
    s += fmtVal(val[i]);
  }
  return s;
}

// merge strategy for scalar conflict resolution
enum ResolveMerge { kMergeMin, kMergeMax, kMergeError };

// single-pass conflict resolver: accumulates errors, warnings, and deferred
// mutations; commits mutations only if no errors were found
struct Resolver {
  mjtConflict              mode;
  mjSpec*                  parent;
  const mjSpec*            child;
  std::vector<std::string> errs;
  std::vector<std::string> warnings;

  std::vector<std::function<void()>> ops;

  Resolver(mjtConflict mode, mjSpec* parent, const mjSpec* child)
      : mode(mode), parent(parent), child(child) {}

  // resolve a scalar field conflict
  template <typename T>
  void operator()(
      const char* name, T& pval, const T& cval, T dval, ResolveMerge merge = kMergeError) {
    // check authored status
    // fall back to default comparison for fields without authored tracking
    bool child_authored = mjs_isAuthored(child, &cval) || (cval != dval);
    if (!child_authored) return;

    bool parent_authored = mjs_isAuthored(parent, &pval) || (pval != dval);

    if (pval == cval) {
      if (!mjs_isAuthored(parent, &pval) && mode == mjCONFLICT_MERGE) {
        mjSpec* pspec = parent;
        ops.push_back([pspec, &pval]() { mjs_setAuthored(pspec, &pval, 1); });
      }
      return;
    }

    // "FIELD: parent has X, child has Y"
    auto prefix = [&]() {
      std::string p_str = fmtVal(pval) + (parent_authored ? "" : " (default)");
      std::string c_str = fmtVal(cval) + (child_authored ? "" : " (default)");
      return std::string(name) + ": parent has " + p_str + ", child has " + c_str;
    };

    // only child authored: adopt or keep
    if (!parent_authored) {
      std::string p = prefix();
      if (mode == mjCONFLICT_MERGE) {
        mjSpec* pspec = parent;
        ops.push_back([pspec, &pval, cval]() {
          pval = cval;
          mjs_setAuthored(pspec, &pval, 1);
        });
        warnings.push_back(p + ", adopting child value");
      } else {
        warnings.push_back(p + ", keeping parent value");
      }
      return;
    }

    // both authored: dispatch by mode
    switch (mode) {
      case mjCONFLICT_WARNING:
        warnings.push_back(prefix() + ", keeping parent value");
        break;
      case mjCONFLICT_MERGE:
        if (merge == kMergeError) {
          errs.push_back(prefix());
          break;
        }
        {
          T merged = merge == kMergeMin ? std::min(pval, cval) : std::max(pval, cval);

          const char* desc = merge == kMergeMin ? "the minimum" : "the maximum";
          warnings.push_back(
              prefix() + (merged == cval ? std::string(", taking ") + desc
                                         : std::string(", keeping parent value (") + desc + ")"));
          mjSpec* pspec = parent;
          ops.push_back([pspec, &pval, merged]() {
            pval = merged;
            mjs_setAuthored(pspec, &pval, 1);
          });
          break;
        }
      case mjCONFLICT_ERROR:
        errs.push_back(prefix());
        break;
    }
  }

  // resolve an unmergeable array field
  template <int N>
  void operator()(const char* name,
                  mjtNum (&pval)[N],
                  const mjtNum (&cval)[N],
                  const mjtNum (&dval)[N]) {
    // check authored status
    bool child_authored = mjs_isAuthored(child, cval);

    // fall back to default comparison for fields without authored tracking
    if (!child_authored) {
      for (int i = 0; i < N; i++) {
        if (cval[i] != dval[i]) {
          child_authored = true;
          break;
        }
      }
    }
    if (!child_authored) return;

    bool parent_authored = mjs_isAuthored(parent, pval);
    if (!parent_authored) {
      for (int i = 0; i < N; i++) {
        if (pval[i] != dval[i]) {
          parent_authored = true;
          break;
        }
      }
    }

    bool equal = true;
    for (int i = 0; i < N; i++) {
      if (pval[i] != cval[i]) {
        equal = false;
        break;
      }
    }
    if (equal) {
      if (!mjs_isAuthored(parent, pval) && mode == mjCONFLICT_MERGE) {
        mjSpec* pspec = parent;
        ops.push_back([pspec, &pval]() { mjs_setAuthored(pspec, pval, 1); });
      }
      return;
    }

    // "FIELD: parent has X Y Z, child has X Y Z"
    auto prefix = [&]() {
      std::string p_str = fmtArr(pval, N) + (parent_authored ? "" : " (default)");
      std::string c_str = fmtArr(cval, N) + (child_authored ? "" : " (default)");
      return std::string(name) + ": parent has " + p_str + ", child has " + c_str;
    };

    // only child authored: adopt or keep
    if (!parent_authored) {
      std::string p = prefix();
      if (mode == mjCONFLICT_MERGE) {
        std::array<mjtNum, N> vals;
        for (int i = 0; i < N; i++) vals[i] = cval[i];
        mjSpec* pspec = parent;
        ops.push_back([pspec, &pval, vals]() {
          for (int i = 0; i < N; i++) pval[i] = vals[i];
          mjs_setAuthored(pspec, pval, 1);
        });
        warnings.push_back(p + ", adopting child value");
      } else {
        warnings.push_back(p + ", keeping parent value");
      }
      return;
    }

    // both authored: dispatch by mode
    switch (mode) {
      case mjCONFLICT_WARNING:
        warnings.push_back(prefix() + ", keeping parent value");
        break;
      case mjCONFLICT_MERGE:
      case mjCONFLICT_ERROR:
        errs.push_back(prefix());
        break;
    }
  }

  // compute effective authored bitmask, falling back to C-API mutations
  int EffectiveBitAuthored(const mjSpec* spec, const int& val, int auth) const {
    if (mjs_isAuthored(spec, &val)) return ~0;
    return auth | val;
  }

  // resolve bitfield conflicts
  void operator()(
      int& pval, const int& cval, const char* const names[], int nbit, int& pauth, int cauth) {
    // compute effective authored bitmasks
    int eff_cauth = EffectiveBitAuthored(child, cval, cauth);
    if (!eff_cauth) return;

    int eff_pauth = EffectiveBitAuthored(parent, pval, pauth);

    // merge set bits and authored bitmask from child
    if (mode == mjCONFLICT_MERGE) {
      int added = cval & ~pval;
      for (int i = 0; i < nbit; i++) {
        if ((added >> i) & 1) {
          warnings.push_back(std::string("flag '") + names[i] + "': added from child");
        }
      }
      int adopted_auth = cauth | cval;
      if (cval || (adopted_auth & ~pauth)) {
        ops.push_back([&pval, &pauth, cval, adopted_auth]() {
          pval  |= cval;
          pauth |= adopted_auth;
        });
      }
      return;
    }

    // only report conflicts for bits where both parent and child authored
    int both = eff_pauth & eff_cauth;
    int diff = (pval ^ cval) & both;
    for (int i = 0; i < nbit; i++) {
      if ((diff >> i) & 1) {
        bool parent_set = (pval >> i) & 1;

        std::string msg = std::string("flag '") +
                          names[i] +
                          "': parent " +
                          (parent_set ? "set" : "unset") +
                          ", child " +
                          (parent_set ? "unset" : "set");
        if (mode == mjCONFLICT_ERROR) {
          errs.push_back(msg);
        } else {
          warnings.push_back(msg + ", keeping parent");
        }
      }
    }
  }

  // resolve disableactuator bitfield conflicts
  void operator()(int& pval, const int& cval, int& pauth, int cauth) {
    // compute effective authored bitmasks
    int eff_cauth = EffectiveBitAuthored(child, cval, cauth);
    if (!eff_cauth) return;

    int eff_pauth = EffectiveBitAuthored(parent, pval, pauth);

    // merge set bits and authored bitmask from child
    if (mode == mjCONFLICT_MERGE) {
      for (int i = 0; i < mjNGROUP; i++) {
        if (((cval & ~pval) >> i) & 1) {
          char buf[64];
          snprintf(buf, sizeof(buf), "disableactuator group %d: added from child", i);
          warnings.push_back(buf);
        }
      }
      int adopted_auth = cauth | cval;
      if (cval || (adopted_auth & ~pauth)) {
        ops.push_back([&pval, &pauth, cval, adopted_auth]() {
          pval  |= cval;
          pauth |= adopted_auth;
        });
      }
      return;
    }
    int both = eff_pauth & eff_cauth;
    int diff = (pval ^ cval) & both;
    for (int i = 0; i < mjNGROUP; i++) {
      if ((diff >> i) & 1) {
        char buf[128];
        bool parent_set = (pval >> i) & 1;
        snprintf(buf,
                 sizeof(buf),
                 "disableactuator group %d: parent %s, child %s",
                 i,
                 parent_set ? "set" : "unset",
                 parent_set ? "unset" : "set");
        if (mode == mjCONFLICT_ERROR) {
          errs.push_back(buf);
        } else {
          warnings.push_back(std::string(buf) + ", keeping parent");
        }
      }
    }
  }

  // apply deferred mutations (only if no errors)
  bool Apply() {
    if (!errs.empty()) return false;
    for (auto& op : ops) op();
    return true;
  }
};

// enumerate all conflictable fields, dispatching each to the resolver
void VisitConflicts(mjSpec* parent, const mjSpec* child, Resolver& r) {
  // ==== mjOption ====
  mjOption d;
  mj_defaultOption(&d);
  mjOption&       po = parent->option;
  const mjOption& co = child->option;

  // min-merge fields
  r("timestep", po.timestep, co.timestep, d.timestep, kMergeMin);
  r("tolerance", po.tolerance, co.tolerance, d.tolerance, kMergeMin);
  r("ls_tolerance", po.ls_tolerance, co.ls_tolerance, d.ls_tolerance, kMergeMin);
  r("noslip_tolerance", po.noslip_tolerance, co.noslip_tolerance, d.noslip_tolerance, kMergeMin);
  r("ccd_tolerance", po.ccd_tolerance, co.ccd_tolerance, d.ccd_tolerance, kMergeMin);
  r("sleep_tolerance", po.sleep_tolerance, co.sleep_tolerance, d.sleep_tolerance, kMergeMin);

  // max-merge fields
  r("iterations", po.iterations, co.iterations, d.iterations, kMergeMax);
  r("ls_iterations", po.ls_iterations, co.ls_iterations, d.ls_iterations, kMergeMax);
  r("noslip_iterations",
    po.noslip_iterations,
    co.noslip_iterations,
    d.noslip_iterations,
    kMergeMax);
  r("ccd_iterations", po.ccd_iterations, co.ccd_iterations, d.ccd_iterations, kMergeMax);
  r("sdf_iterations", po.sdf_iterations, co.sdf_iterations, d.sdf_iterations, kMergeMax);
  r("sdf_initpoints", po.sdf_initpoints, co.sdf_initpoints, d.sdf_initpoints, kMergeMax);

  // unmergeable scalars
  r("impratio", po.impratio, co.impratio, d.impratio);
  r("density", po.density, co.density, d.density);
  r("viscosity", po.viscosity, co.viscosity, d.viscosity);
  r("o_margin", po.o_margin, co.o_margin, d.o_margin);
  r("integrator", po.integrator, co.integrator, d.integrator);
  r("cone", po.cone, co.cone, d.cone);
  r("jacobian", po.jacobian, co.jacobian, d.jacobian);
  r("solver", po.solver, co.solver, d.solver);

  // unmergeable arrays
  r("gravity", po.gravity, co.gravity, d.gravity);
  r("wind", po.wind, co.wind, d.wind);
  r("magnetic", po.magnetic, co.magnetic, d.magnetic);
  r("o_solref", po.o_solref, co.o_solref, d.o_solref);
  r("o_solimp", po.o_solimp, co.o_solimp, d.o_solimp);
  r("o_friction", po.o_friction, co.o_friction, d.o_friction);

  // bitfields (pass authored bitmasks)
  mjsAuthored&       pa = parent->authored;
  const mjsAuthored& ca = child->authored;
  r(po.disableflags,
    co.disableflags,
    mjDISABLESTRING,
    mjNDISABLE,
    pa.disableflags,
    ca.disableflags);
  r(po.enableflags, co.enableflags, mjENABLESTRING, mjNENABLE, pa.enableflags, ca.enableflags);
  r(po.disableactuator, co.disableactuator, pa.disableactuator, ca.disableactuator);

  // ==== mjVisual ====
  mjVisual dv;
  mj_defaultVisual(&dv);
  mjVisual&       pv = parent->visual;
  const mjVisual& cv = child->visual;

  r("znear", pv.map.znear, cv.map.znear, dv.map.znear, kMergeMin);
  r("realtime", pv.global.realtime, cv.global.realtime, dv.global.realtime, kMergeMin);
  r("zfar", pv.map.zfar, cv.map.zfar, dv.map.zfar, kMergeMax);

  // ==== mjSpec sizes (no authored tracking, uses default fallback) ====
  mjSpec ds;
  mjs_defaultSpec(&ds);
  mjSpec&       ps = *parent;
  const mjSpec& cs = *child;

  r("memory", ps.memory, cs.memory, ds.memory, kMergeMax);
  r("njmax", ps.njmax, cs.njmax, ds.njmax, kMergeMax);
  r("nconmax", ps.nconmax, cs.nconmax, ds.nconmax, kMergeMax);
  r("nuserdata", ps.nuserdata, cs.nuserdata, ds.nuserdata, kMergeMax);
  r("nkey", ps.nkey, cs.nkey, ds.nkey, kMergeMax);
  r("nuser_body", ps.nuser_body, cs.nuser_body, ds.nuser_body, kMergeMax);
  r("nuser_jnt", ps.nuser_jnt, cs.nuser_jnt, ds.nuser_jnt, kMergeMax);
  r("nuser_geom", ps.nuser_geom, cs.nuser_geom, ds.nuser_geom, kMergeMax);
  r("nuser_site", ps.nuser_site, cs.nuser_site, ds.nuser_site, kMergeMax);
  r("nuser_cam", ps.nuser_cam, cs.nuser_cam, ds.nuser_cam, kMergeMax);
  r("nuser_tendon", ps.nuser_tendon, cs.nuser_tendon, ds.nuser_tendon, kMergeMax);
  r("nuser_actuator", ps.nuser_actuator, cs.nuser_actuator, ds.nuser_actuator, kMergeMax);
  r("nuser_sensor", ps.nuser_sensor, cs.nuser_sensor, ds.nuser_sensor, kMergeMax);
}

// compose subject line for conflict messages
std::string ConflictSubject(const mjSpec* parent, const mjSpec* child) {
  const char* mode_str = parent->compiler.conflict == mjCONFLICT_MERGE   ? "merge"
                         : parent->compiler.conflict == mjCONFLICT_ERROR ? "error"
                                                                         : "warning";
  std::string pname    = *parent->modelname;
  std::string cname    = *child->modelname;

  bool has_parent = (pname != "MuJoCo Model" && !pname.empty());
  bool has_child  = (cname != "MuJoCo Model" && !cname.empty());
  if (has_child && has_parent) {
    return "Attach conflict when attaching '" +
           cname +
           "' to '" +
           pname +
           "', policy is '" +
           mode_str +
           "'";
  } else if (has_child) {
    return "Attach conflict when attaching '" + cname + "', policy is '" + mode_str + "'";
  } else if (has_parent) {
    return "Attach conflict when attaching to '" + pname + "', policy is '" + mode_str + "'";
  }
  return std::string("Attach conflict on attach, policy is '") + mode_str + "'";
}

}  // namespace

bool ResolveConflicts(mjSpec*       parent,
                      const mjSpec* child,
                      mjtConflict   mode,
                      std::string*  error_msg,
                      std::string*  warning_subject,
                      std::string*  warning_body) {
  Resolver r(mode, parent, child);
  VisitConflicts(parent, child, r);

  if (!r.Apply()) {
    if (error_msg) {
      *error_msg = ConflictSubject(parent, child);
      for (const auto& e : r.errs) {
        error_msg->append("\n");
        error_msg->append(e);
      }
    }
    return false;
  }

  if (!r.warnings.empty()) {
    if (warning_subject) { *warning_subject = ConflictSubject(parent, child); }
    if (warning_body) {
      warning_body->clear();
      for (size_t i = 0; i < r.warnings.size(); ++i) {
        if (i > 0) { warning_body->append("\n"); }
        warning_body->append(r.warnings[i]);
      }
    }
  }
  return true;
}

}  // namespace mujoco
