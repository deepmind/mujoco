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

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ratio>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

// maximum number of threads
const int maxthread = 512;

// rollout runner state
struct RolloutRunner {
  mjModel* m            = nullptr;
  mjData*  d[maxthread] = {nullptr};

  // per-thread statistics
  int    contacts[maxthread]    = {0};
  int    constraints[maxthread] = {0};
  mjtNum iterations[maxthread]  = {0.0};
  mjtNum simtime[maxthread]     = {0.0};
};

static RolloutRunner runner;

// timer (milliseconds)
mjtNum gettm(void) {
  using Clock        = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<mjtNum, std::milli>;

  static const Clock::time_point tm_start = Clock::now();
  return Milliseconds(Clock::now() - tm_start).count();
}

// deallocate and print message
int finish(const char* msg = NULL, mjModel* m = NULL) {
  // deallocate model
  if (m) { mj_deleteModel(m); }

  // print message
  if (msg) { std::printf("%s\n", msg); }

  return EXIT_SUCCESS;
}

std::vector<mjtNum> CtrlNoise(
    const mjModel* m, int nsteps, mjtNum noise_std, mjtNum noise_rate, int key) {
  std::vector<mjtNum> ctrl;
  ctrl.reserve(nsteps * m->nu);

  // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
  mjtNum rate  = mju_exp(-m->opt.timestep / noise_rate);
  mjtNum scale = noise_std * mju_sqrt(1 - rate * rate);

  for (int step = 0; step < nsteps; step++) {
    for (int i = 0; i < m->nu; i++) {
      mjtNum  midpoint  = 0.0;
      mjtNum  halfrange = 1.0;
      mjtNum* range     = m->actuator_ctrlrange + 2 * i;
      if (m->actuator_ctrllimited[i]) {
        midpoint  = 0.5 * (range[1] + range[0]);
        halfrange = 0.5 * (range[1] - range[0]);
      }

      // overwrite midpoint with keyframe, if given
      if (key >= 0) { midpoint = m->key_ctrl[key * m->nu + i]; }

      // exponential convergence to midpoint at ctrl_noise_rate
      mjtNum ctrl_ =
          step > 0 ? rate * ctrl[(step - 1) * m->nu + i] + (1 - rate) * midpoint : midpoint;

      // add noise
      ctrl_ += scale * halfrange * (2 * mju_Halton(step, i + 2) - 1);

      // clip to range if limited
      if (m->actuator_ctrllimited[i]) { ctrl_ = mju_clip(ctrl_, range[0], range[1]); }

      ctrl.push_back(ctrl_);
    }
  }
  return ctrl;
}

// thread function
void simulate(int id, int nstep, mjtNum* ctrl) {
  // clear statistics
  runner.contacts[id]    = 0;
  runner.constraints[id] = 0;
  runner.iterations[id]  = 0;

  // run and time
  mjtNum start = gettm();
  for (int i = 0; i < nstep; i++) {
    // inject pseudo-random control noise
    mju_copy(runner.d[id]->ctrl, ctrl + i * runner.m->nu, runner.m->nu);

    // advance simulation
    mj_step(runner.m, runner.d[id]);

    // accumulate statistics
    runner.contacts[id]    += runner.d[id]->ncon;
    runner.constraints[id] += runner.d[id]->nefc;
    int nisland             = mjMAX(1, mjMIN(runner.d[id]->nisland, mjNISLAND));
    if (nisland == 1 || nisland == 0) {
      runner.iterations[id] += runner.d[id]->solver_niter[0];
    } else {
      mjtNum niter = 0;
      for (int j = 0; j < nisland; j++) { niter += runner.d[id]->solver_niter[j]; }
      runner.iterations[id] += niter / nisland;
    }
  }
  runner.simtime[id] = 1e-3 * (gettm() - start);
}

// print non-default options
static void PrintOptions(const mjModel* m) {
  mjOption optd;
  mj_defaultOption(&optd);

  bool header_printed = false;
  auto print_header   = [&]() {
    if (!header_printed) {
      std::printf("\nPhysics options (non-default):\n");
      header_printed = true;
    }
  };

#define X(type, name, size)                                                             \
  if (std::strcmp(#name, "disableflags") != 0 &&                                        \
      std::strcmp(#name, "enableflags") != 0 &&                                         \
      std::strcmp(#name, "disableactuator") != 0) {                                     \
    if (m->opt.name != optd.name) {                                                     \
      print_header();                                                                   \
      std::printf("  %-18s: ", #name);                                                  \
      if (std::strcmp(#name, "integrator") == 0) {                                      \
        const char* names[] = {"Euler", "RK4", "Implicit", "ImplicitFast", "Discrete"}; \
        int         val     = (int)m->opt.name;                                         \
        if (val >= 0 && val < 5) {                                                      \
          std::printf("%s", names[val]);                                                \
        } else {                                                                        \
          std::printf("%d", val);                                                       \
        }                                                                               \
      } else if (std::strcmp(#name, "cone") == 0) {                                     \
        const char* names[] = {"Pyramidal", "Elliptic"};                                \
        int         val     = (int)m->opt.name;                                         \
        if (val >= 0 && val < 2) {                                                      \
          std::printf("%s", names[val]);                                                \
        } else {                                                                        \
          std::printf("%d", val);                                                       \
        }                                                                               \
      } else if (std::strcmp(#name, "jacobian") == 0) {                                 \
        const char* names[] = {"Dense", "Sparse", "Auto"};                              \
        int         val     = (int)m->opt.name;                                         \
        if (val >= 0 && val < 3) {                                                      \
          std::printf("%s", names[val]);                                                \
        } else {                                                                        \
          std::printf("%d", val);                                                       \
        }                                                                               \
      } else if (std::strcmp(#name, "solver") == 0) {                                   \
        const char* names[] = {"PGS", "CG", "Newton"};                                  \
        int         val     = (int)m->opt.name;                                         \
        if (val >= 0 && val < 3) {                                                      \
          std::printf("%s", names[val]);                                                \
        } else {                                                                        \
          std::printf("%d", val);                                                       \
        }                                                                               \
      } else {                                                                          \
        if (std::strcmp(#type, "int") == 0) {                                           \
          std::printf("%d", (int)m->opt.name);                                          \
        } else {                                                                        \
          std::printf("%g", (double)m->opt.name);                                       \
        }                                                                               \
      }                                                                                 \
      std::printf("\n");                                                                \
    }                                                                                   \
  }

#define XVEC(type, name, size)                                                       \
  {                                                                                  \
    bool diff = false;                                                               \
    for (int i = 0; i < size; ++i) {                                                 \
      if (m->opt.name[i] != optd.name[i]) {                                          \
        diff = true;                                                                 \
        break;                                                                       \
      }                                                                              \
    }                                                                                \
    if (diff) {                                                                      \
      print_header();                                                                \
      std::printf("  %-18s:", #name);                                                \
      for (int i = 0; i < size; ++i) { std::printf(" %g", (double)m->opt.name[i]); } \
      std::printf("\n");                                                             \
    }                                                                                \
  }

  // option fields
#include <mujoco/mjxmacro.h>  // NOLINT(build/include)
  MJOPTION_FIELDS

#undef X
#undef XVEC

  // disableflags
  for (int i = 0; i < mjNDISABLE; ++i) {
    bool current = (m->opt.disableflags & (1 << i)) != 0;
    bool def     = (optd.disableflags & (1 << i)) != 0;
    if (current != def) {
      print_header();
      std::printf("  %-18s: %s\n", mjDISABLESTRING[i], current ? "Disabled" : "Enabled");
    }
  }

  // enableflags
  for (int i = 0; i < mjNENABLE; ++i) {
    bool current = (m->opt.enableflags & (1 << i)) != 0;
    bool def     = (optd.enableflags & (1 << i)) != 0;
    if (current != def) {
      print_header();
      std::printf("  %-18s: %s\n", mjENABLESTRING[i], current ? "Enabled" : "Disabled");
    }
  }

  // disableactuator
  if (m->opt.disableactuator != optd.disableactuator) {
    print_header();
    std::printf("  disableactuator   :");
    bool first = true;
    for (int i = 0; i < 32; ++i) {
      if (m->opt.disableactuator & (1 << i)) {
        if (!first) std::printf(",");
        std::printf(" %d", i);
        first = false;
      }
    }
    if (first) { std::printf(" none"); }
    std::printf("\n");
  }
}

// helper enum parser: returns the matched enum index from the names vector (case-insensitive)
// or the parsed integer value if it represents a valid number. Returns -1 on failure.
static int ParseEnum(std::string_view val, const std::vector<std::string>& names) {
  int int_val;
  auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), int_val);
  if (ec == std::errc()) { return int_val; }
  for (size_t i = 0; i < names.size(); ++i) {
    if (val.size() == names[i].size() &&
        std::equal(val.begin(), val.end(), names[i].begin(), [](unsigned char a, unsigned char b) {
          return std::tolower(a) == std::tolower(b);
        })) {
      return i;
    }
  }
  return -1;
}

// main function
int main(int argc, char** argv) {
  static const char* help_msg =
      "\n"
      "Usage:  testspeed [options] model\n"
      "\n"
      "  option                 default   semantic\n"
      "  ------                 -------   --------\n"
      "  model                            path to model (required, positional)\n"
      "  --nstep=N              10000     number of steps per rollout\n"
      "  --nthread=N            1         number of threads running parallel rollouts\n"
      "  --noisestd=X           0.01      scale of pseudo-random noise injected into actuators\n"
      "  --noiserate=X          0.1       rate of convergence to ctrl keyframe/midpoint\n"
      "  --nenginethread=N      0         number of worker threads in engine-internal threadpool\n"
      "  --solver=S             Newton    PGS, CG, Newton\n"
      "  --cone=C               Pyramidal Pyramidal, Elliptic\n"
      "  --jacobian=J           Auto      Dense, Sparse, Auto\n"
      "  --integrator=I         Euler     Euler, RK4, Implicit, ImplicitFast, Discrete\n"
      "  --iterations=N         100       solver iterations limit\n"
      "  --tolerance=X          1e-8      solver tolerance\n"
      "  --sleep_tolerance=X    1e-3      sleep tolerance\n"
      "  --noslip_iterations=N  0         noslip solver iterations limit\n"
      "  --help (or no arguments)         print this help message\n"
      "\n"
      "Note: If the model has a keyframe named \"test\", it will be loaded prior to simulation\n";

  // default values
  int nstep = 10000, nthread = 0, nenginethread = 0;
  // inject small noise by default, to avoid fixed contact state
  double noisestd  = 0.01;
  double noiserate = 0.1;

  // option override settings
  bool   set_solver = false, set_cone = false, set_jacobian = false, set_integrator = false;
  bool   set_iterations = false, set_tolerance = false, set_sleep_tolerance = false;
  bool   set_noslip_iterations = false;
  int    opt_solver = -1, opt_cone = -1, opt_jacobian = -1, opt_integrator = -1;
  int    opt_iterations = -1, opt_noslip_iterations = -1;
  double opt_tolerance = -1.0, opt_sleep_tolerance = -1.0;

  const char* model = nullptr;
  for (int i = 1; i < argc; i++) {
    // helper: given "--key=value" or "--key value", extract the value string
    auto getarg = [&](const char* key) -> const char* {
      std::string prefix = std::string("--") + key + "=";
      if (std::strncmp(argv[i], prefix.c_str(), prefix.size()) == 0) {
        return argv[i] + prefix.size();
      }
      if (std::strcmp(argv[i], (std::string("--") + key).c_str()) == 0) {
        if (i + 1 < argc) { return argv[++i]; }
      }
      return nullptr;
    };

    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      return finish(help_msg);
    }

    const char* val;
    if ((val = getarg("nstep"))) {
      if (std::sscanf(val, "%d", &nstep) != 1 || nstep <= 0) {
        return finish("Invalid --nstep argument");
      }
    } else if ((val = getarg("nthread"))) {
      if (std::sscanf(val, "%d", &nthread) != 1) { return finish("Invalid --nthread argument"); }
    } else if ((val = getarg("noisestd"))) {
      if (std::sscanf(val, "%lf", &noisestd) != 1) { return finish("Invalid --noisestd argument"); }
    } else if ((val = getarg("noiserate"))) {
      if (std::sscanf(val, "%lf", &noiserate) != 1) {
        return finish("Invalid --noiserate argument");
      }
    } else if ((val = getarg("nenginethread"))) {
      if (std::sscanf(val, "%d", &nenginethread) != 1) {
        return finish("Invalid --nenginethread argument");
      }
    } else if ((val = getarg("solver"))) {
      int parsed = ParseEnum(val, {"PGS", "CG", "Newton"});
      if (parsed < 0 || parsed > 2) { return finish("Invalid --solver argument"); }
      opt_solver = parsed;
      set_solver = true;
    } else if ((val = getarg("cone"))) {
      int parsed = ParseEnum(val, {"Pyramidal", "Elliptic"});
      if (parsed < 0 || parsed > 1) { return finish("Invalid --cone argument"); }
      opt_cone = parsed;
      set_cone = true;
    } else if ((val = getarg("jacobian"))) {
      int parsed = ParseEnum(val, {"Dense", "Sparse", "Auto"});
      if (parsed < 0 || parsed > 2) { return finish("Invalid --jacobian argument"); }
      opt_jacobian = parsed;
      set_jacobian = true;
    } else if ((val = getarg("integrator"))) {
      int parsed = ParseEnum(val, {"Euler", "RK4", "Implicit", "ImplicitFast", "Discrete"});
      if (parsed < 0 || parsed > 4) { return finish("Invalid --integrator argument"); }
      opt_integrator = parsed;
      set_integrator = true;
    } else if ((val = getarg("iterations"))) {
      if (std::sscanf(val, "%d", &opt_iterations) != 1 || opt_iterations <= 0) {
        return finish("Invalid --iterations argument");
      }
      set_iterations = true;
    } else if ((val = getarg("tolerance"))) {
      if (std::sscanf(val, "%lf", &opt_tolerance) != 1 || opt_tolerance < 0) {
        return finish("Invalid --tolerance argument");
      }
      set_tolerance = true;
    } else if ((val = getarg("sleep_tolerance"))) {
      if (std::sscanf(val, "%lf", &opt_sleep_tolerance) != 1 || opt_sleep_tolerance < 0) {
        return finish("Invalid --sleep_tolerance argument");
      }
      set_sleep_tolerance = true;
    } else if ((val = getarg("noslip_iterations"))) {
      if (std::sscanf(val, "%d", &opt_noslip_iterations) != 1 || opt_noslip_iterations < 0) {
        return finish("Invalid --noslip_iterations argument");
      }
      set_noslip_iterations = true;
    } else if (argv[i][0] != '-') {
      // positional argument: model file
      model = argv[i];
    } else {
      std::printf("Unknown option: %s\n", argv[i]);
      return finish(help_msg);
    }
  }

  // model file is required
  if (!model) { return finish(help_msg); }

  // clamp noisestd to [0.0, 1.0]
  noisestd = mju_clip(noisestd, 0.0, 1.0);

  // clamp noiserate to [0.0, 1.0]
  noiserate = mju_clip(noiserate, 0.0, 1.0);

  // clamp nthread to [1, maxthread], nenginethread to [0, maxthread]
  nthread       = mjMAX(1, mjMIN(maxthread, nthread));
  nenginethread = mjMAX(0, mjMIN(maxthread, nenginethread));

  // get filename, determine file type
  std::string filename(model);
  bool        binary = (filename.find(".mjb") != std::string::npos);  // NOLINT

  // load model
  char error[1000] = "Could not load binary model";
  if (binary) {
    runner.m = mj_loadModel(model, 0);
  } else {
    runner.m = mj_loadXML(model, 0, error, 1000);
  }
  if (!runner.m) { return finish(error); }

  // apply command-line option overrides
  if (set_solver) runner.m->opt.solver = opt_solver;
  if (set_cone) runner.m->opt.cone = opt_cone;
  if (set_jacobian) runner.m->opt.jacobian = opt_jacobian;
  if (set_integrator) runner.m->opt.integrator = opt_integrator;
  if (set_iterations) runner.m->opt.iterations = opt_iterations;
  if (set_tolerance) runner.m->opt.tolerance = opt_tolerance;
  if (set_sleep_tolerance) runner.m->opt.sleep_tolerance = opt_sleep_tolerance;
  if (set_noslip_iterations) runner.m->opt.noslip_iterations = opt_noslip_iterations;

  // make per-thread data
  int testkey = mj_name2id(runner.m, mjOBJ_KEY, "test");
  for (int id = 0; id < nthread; id++) {
    // make mjData(s)
    runner.d[id] = mj_makeData(runner.m);
    if (!runner.d[id]) { return finish("Could not allocate mjData", runner.m); }

    // reset to keyframe
    if (testkey >= 0) { mj_resetDataKeyframe(runner.m, runner.d[id], testkey); }

    // make and bind threadpool
    if (nenginethread > 0) { mju_threadpool(runner.d[id], nenginethread); }
  }

  // install timer callback for profiling
  mjcb_time = gettm;

  // print physics options if not default
  PrintOptions(runner.m);

  // print start
  std::printf("\nRolling out %d steps%s at dt = %g",
              nstep,
              nthread > 1 ? " per thread" : "",
              runner.m->opt.timestep);

  // print precision
  if (sizeof(mjtNum) == 4) {
    std::printf(", using single precision");
  } else {
    std::printf(", using double precision");
  }

  // print thread pool size
  if (nenginethread > 0) {
    std::printf(", using %d worker thread%s", nenginethread, nenginethread > 1 ? "s" : "");
  }
  std::printf("...\n");

  // create pseudo-random control sequence
  std::vector<mjtNum> ctrl = CtrlNoise(runner.m, nstep, noisestd, noiserate, testkey);

  // run simulation, record total time
  std::thread th[maxthread];
  double      starttime = gettm();
  for (int id = 0; id < nthread; id++) { th[id] = std::thread(simulate, id, nstep, ctrl.data()); }
  for (int id = 0; id < nthread; id++) { th[id].join(); }
  double tottime = 1e-3 * (gettm() - starttime);  // total time, in seconds

  // all-thread summary
  constexpr char mu_str[3] = "\u00B5";  // unicode mu character
  if (nthread > 1) {
    std::printf("Summary for all %d threads\n\n", nthread);
    std::printf(" Total simulation time  : %.2f s\n", tottime);
    std::printf(" Total steps per second : %.0f\n", nthread * nstep / tottime);
    std::printf(" Total realtime factor  : %.2f x\n",
                nthread * nstep * runner.m->opt.timestep / tottime);
    std::printf(" Total time per step    : %.1f %ss\n\n",
                1e6 * tottime / (nthread * nstep),
                mu_str);

    std::printf("Details for thread 0\n\n");
  }

  // solver names indexed by mjtSolver
  const char* solver_names[] = {"PGS", "CG", "Newton"};
  const char* solto6[]       = {"   ", "    ", ""};  // complete to 6 characters

  // details for thread 0
  std::printf(" Simulation time      : %.2f s\n", runner.simtime[0]);
  std::printf(" Steps per second     : %.0f\n", nstep / runner.simtime[0]);
  std::printf(" Realtime factor      : %.2f x\n",
              nstep * runner.m->opt.timestep / runner.simtime[0]);
  std::printf(" Time per step        : %.1f %ss\n\n", 1e6 * runner.simtime[0] / nstep, mu_str);
  std::printf(" %s iters / step  %s: %.2f\n",
              solver_names[runner.m->opt.solver],
              solto6[runner.m->opt.solver],
              runner.iterations[0] / nstep);
  std::printf(" Contacts / step      : %.2f\n", static_cast<float>(runner.contacts[0]) / nstep);
  std::printf(" Constraints / step   : %.2f\n", static_cast<float>(runner.constraints[0]) / nstep);
  std::printf(" Degrees of freedom   : %" PRId64 "\n", runner.m->nv);
  std::printf(" Dynamic memory usage : %.1f%% of %s\n\n",
              100 * runner.d[0]->maxuse_arena / (double)(runner.d[0]->narena),
              mju_writeNumBytes(runner.d[0]->narena));

  // profiler, top-level
  printf(" Internal profiler%s, %ss per step\n", nthread > 1 ? " for thread 0" : "", mu_str);
  int    number     = runner.d[0]->timer[mjTIMER_STEP].number;
  mjtNum tstep      = number ? 1000 * runner.d[0]->timer[mjTIMER_STEP].duration / number : 0.0;
  mjtNum components = 0, total = 0;
  for (int i = 0; i <= mjTIMER_ADVANCE; i++) {
    if (runner.d[0]->timer[i].number > 0) {
      int    number  = runner.d[0]->timer[i].number;
      mjtNum istep   = number ? 1000 * runner.d[0]->timer[i].duration / number : 0.0;
      mjtNum percent = number ? 100 * istep / tstep : 0.0;
      std::printf(" %17s : %6.1f  (%6.2f %%)\n", mjTIMERSTRING[i], istep, percent);

      // save step time, add up timing of components
      if (i == 0) total = istep;
      if (i >= mjTIMER_POSITION) { components += istep; }
    }
  }

  // "other" (computation not covered by timers)
  if (tstep > 0) {
    mjtNum other = total - components;
    std::printf(" %17s : %6.1f  (%6.2f %%)\n", "other", other, 100 * other / tstep);
  }

  std::printf("\n");

  // mjTIMER_POSITION and its components
  for (int i : {mjTIMER_POSITION,
                mjTIMER_POS_KINEMATICS,
                mjTIMER_POS_INERTIA,
                mjTIMER_POS_COLLISION,
                mjTIMER_POS_MAKE,
                mjTIMER_POS_PROJECT}) {
    if (runner.d[0]->timer[i].number > 0) {
      mjtNum istep = 1000 * runner.d[0]->timer[i].duration / runner.d[0]->timer[i].number;
      if (i == mjTIMER_POSITION) {
        std::printf("   position total  : %6.1f  (%6.2f %%)\n", istep, 100 * istep / tstep);
      } else {
        std::printf("     %-10s    : %6.1f  (%6.2f %%)\n",
                    mjTIMERSTRING[i] + 4,
                    istep,
                    100 * istep / tstep);
      }
    }

    // components of mjTIMER_POS_COLLISION
    if (i == mjTIMER_POS_COLLISION) {
      for (int j : {mjTIMER_COL_BROAD, mjTIMER_COL_NARROW}) {
        int    number  = runner.d[0]->timer[j].number;
        mjtNum jstep   = number ? 1000 * runner.d[0]->timer[j].duration / number : 0.0;
        mjtNum percent = number ? 100 * jstep / tstep : 0.0;
        std::printf("       %-11s : %6.1f  (%6.2f %%)\n", mjTIMERSTRING[j] + 4, jstep, percent);
      }
    }
  }

  // free per-thread data
  for (int id = 0; id < nthread; id++) { mj_deleteData(runner.d[id]); }

  // finalize
  return finish();
}
