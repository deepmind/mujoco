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

#include "simulate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <ratio>
#include <string>
#include <type_traits>
#include <utility>

#include "lodepng.h"
#include <mujoco/mjdata.h>
#include <mujoco/mjui.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "platform_ui_adapter.h"
#include "array_safety.h"

// When launched via an App Bundle on macOS, the working directory is the path to the App Bundle's
// resource directory. This causes files to be saved into the bundle, which is not the desired
// behavior. Instead, we open a save dialog box to ask the user where to put the file.
// Since the dialog box logic needs to be written in Objective-C, we separate it into a different
// source file.
#ifdef __APPLE__
std::string GetSavePath(const char* filename);
#else
static std::string GetSavePath(const char* filename) {
  return filename;
}
#endif

namespace {
namespace mj  = ::mujoco;
namespace mju = ::mujoco::sample_util;

using Seconds      = std::chrono::duration<double>;
using Milliseconds = std::chrono::duration<double, std::milli>;

template <typename T>
inline bool IsDifferent(const T& a, const T& b) {
  if constexpr (std::is_array_v<T>) {
    static_assert(std::rank_v<T> == 1);
    for (int i = 0; i < std::extent_v<T>; ++i) {
      if (a[i] != b[i]) { return true; }
    }
    return false;
  } else {
    return a != b;
  }
}

template <typename T>
inline void CopyScalar(T& dst, const T& src) {
  dst = src;
}

template <typename T, int N>
inline void CopyArray(T (&dst)[N], const T (&src)[N]) {
  for (int i = 0; i < N; ++i) { dst[i] = src[i]; }
}

template <typename T>
inline void Copy(T& dst, const T& src) {
  if constexpr (std::is_array_v<T>) {
    CopyArray(dst, src);
  } else {
    CopyScalar(dst, src);
  }
}

//------------------------------------------- global -----------------------------------------------

const double zoom_increment = 0.02;  // ratio of one click-wheel zoom increment to vertical extent

// section ids
enum {
  // left ui
  SECT_FILE = 0,
  SECT_OPTION,
  SECT_SIMULATION,
  SECT_WATCH,
  SECT_PHYSICS,
  SECT_RENDERING,
  SECT_VISUALIZATION,
  SECT_GROUP,
  SECT_LOGGING,
  NSECT0,

  // right ui
  SECT_JOINT = 0,
  SECT_CONTROL,
  SECT_EQUALITY,
  NSECT1
};

// file section of UI
const mjuiDef defFile[] = {
    // clang-format off
  {mjITEM_SECTION,   "File",          mjPRESERVE, nullptr, "AF"},
  {mjITEM_BUTTON,    "Save xml",      2, nullptr, ""},
  {mjITEM_BUTTON,    "Save mjb",      2, nullptr, ""},
  {mjITEM_BUTTON,    "Print model",   2, nullptr, "CM"},
  {mjITEM_BUTTON,    "Print data",    2, nullptr, "CD"},
  {mjITEM_BUTTON,    "Quit",          1, nullptr, "CQ"},
  {mjITEM_BUTTON,    "Screenshot",    2, nullptr, "CP"},
  {mjITEM_END}
    // clang-format on
};

// help strings
const char help_content[] =
    "Space\n"
    "+  -\n"
    "Left / Right arrow\n"
    "Tab / Shift-Tab\n"
    "[  ]\n"
    "Esc\n"
    "Double-click\n"
    "Page Up\n"
    "Right double-click\n"
    "Ctrl Right double-click\n"
    "Scroll, middle drag\n"
    "Left drag\n"
    "[Shift] right drag\n"
    "Ctrl [Shift] drag\n"
    "Ctrl [Shift] right drag\n"
    "F1\n"
    "F2\n"
    "F3\n"
    "F4\n"
    "F5\n"
    "UI right-button hold\n"
    "UI title double-click";

const char help_title[] =
    "Play / Pause\n"
    "Speed Up / Down\n"
    "Step Back / Forward\n"
    "Toggle Left / Right UI\n"
    "Cycle cameras\n"
    "Free camera\n"
    "Select\n"
    "Select parent\n"
    "Center camera\n"
    "Tracking camera\n"
    "Zoom\n"
    "View Orbit\n"
    "View Pan\n"
    "Object Rotate\n"
    "Object Translate\n"
    "Help\n"
    "Info\n"
    "Profiler\n"
    "Sensors\n"
    "Full screen\n"
    "Show UI shortcuts\n"
    "Expand/collapse all";


//-------------------------------- profiler, sensor, info, watch -----------------------------------

// number of lines in the Constraint ("Counts") and Cost ("Convergence") figures
static constexpr int kConstraintNum = 5;
static constexpr int kCostNum       = 3;

// init profiler figures
void InitializeProfiler(mj::Simulate* sim) {
  // set figures to default
  mjv_defaultFigure(&sim->figconstraint);
  mjv_defaultFigure(&sim->figcost);
  mjv_defaultFigure(&sim->figtimer);
  mjv_defaultFigure(&sim->figsize);

  // titles
  mju::strcpy_arr(sim->figconstraint.title, "Counts");
  mju::strcpy_arr(sim->figcost.title, "Convergence (log 10)");
  mju::strcpy_arr(sim->figsize.title, "Dimensions");
  mju::strcpy_arr(sim->figtimer.title, "CPU time (msec)");

  // x-labels
  mju::strcpy_arr(sim->figconstraint.xlabel, "Solver iteration");
  mju::strcpy_arr(sim->figcost.xlabel, "Solver iteration");
  mju::strcpy_arr(sim->figsize.xlabel, "Video frame");
  mju::strcpy_arr(sim->figtimer.xlabel, "Video frame");

  // y-tick number formats
  mju::strcpy_arr(sim->figconstraint.yformat, "%.0f");
  mju::strcpy_arr(sim->figcost.yformat, "%.1f");
  mju::strcpy_arr(sim->figsize.yformat, "%.0f");
  mju::strcpy_arr(sim->figtimer.yformat, "%.2f");

  // colors
  sim->figconstraint.figurergba[0] = 0.1f;
  sim->figcost.figurergba[2]       = 0.2f;
  sim->figsize.figurergba[0]       = 0.1f;
  sim->figtimer.figurergba[2]      = 0.2f;
  sim->figconstraint.figurergba[3] = 0.5f;
  sim->figcost.figurergba[3]       = 0.5f;
  sim->figsize.figurergba[3]       = 0.5f;
  sim->figtimer.figurergba[3]      = 0.5f;

  // repeat line colors for constraint and cost figures
  mjvFigure* fig = &sim->figcost;
  for (int i = kCostNum; i < mjMAXLINE; i++) {
    fig->linergb[i][0] = fig->linergb[i - kCostNum][0];
    fig->linergb[i][1] = fig->linergb[i - kCostNum][1];
    fig->linergb[i][2] = fig->linergb[i - kCostNum][2];
  }
  fig = &sim->figconstraint;
  for (int i = kConstraintNum; i < mjMAXLINE; i++) {
    fig->linergb[i][0] = fig->linergb[i - kConstraintNum][0];
    fig->linergb[i][1] = fig->linergb[i - kConstraintNum][1];
    fig->linergb[i][2] = fig->linergb[i - kConstraintNum][2];
  }

  // legends
  mju::strcpy_arr(sim->figconstraint.linename[0], "total");
  mju::strcpy_arr(sim->figconstraint.linename[1], "active");
  mju::strcpy_arr(sim->figconstraint.linename[2], "changed");
  mju::strcpy_arr(sim->figconstraint.linename[3], "evals");
  mju::strcpy_arr(sim->figconstraint.linename[4], "updates");
  mju::strcpy_arr(sim->figcost.linename[0], "improvement");
  mju::strcpy_arr(sim->figcost.linename[1], "gradient");
  mju::strcpy_arr(sim->figcost.linename[2], "lineslope");
  mju::strcpy_arr(sim->figsize.linename[0], "dof");
  mju::strcpy_arr(sim->figsize.linename[1], "body");
  mju::strcpy_arr(sim->figsize.linename[2], "constraint");
  mju::strcpy_arr(sim->figsize.linename[3], "sqrt(nnz)");
  mju::strcpy_arr(sim->figsize.linename[4], "contact");
  mju::strcpy_arr(sim->figsize.linename[5], "iteration");
  mju::strcpy_arr(sim->figtimer.linename[0], "total");
  mju::strcpy_arr(sim->figtimer.linename[1], "collision");
  mju::strcpy_arr(sim->figtimer.linename[2], "prepare");
  mju::strcpy_arr(sim->figtimer.linename[3], "solve");
  mju::strcpy_arr(sim->figtimer.linename[4], "other");

  // grid sizes
  sim->figconstraint.gridsize[0] = 5;
  sim->figconstraint.gridsize[1] = 5;
  sim->figcost.gridsize[0]       = 5;
  sim->figcost.gridsize[1]       = 5;
  sim->figsize.gridsize[0]       = 3;
  sim->figsize.gridsize[1]       = 5;
  sim->figtimer.gridsize[0]      = 3;
  sim->figtimer.gridsize[1]      = 5;

  // minimum ranges
  sim->figconstraint.range[0][0] = 0;
  sim->figconstraint.range[0][1] = 20;
  sim->figconstraint.range[1][0] = 0;
  sim->figconstraint.range[1][1] = 80;
  sim->figcost.range[0][0]       = 0;
  sim->figcost.range[0][1]       = 20;
  sim->figcost.range[1][0]       = -15;
  sim->figcost.range[1][1]       = 5;
  sim->figsize.range[0][0]       = -200;
  sim->figsize.range[0][1]       = 0;
  sim->figsize.range[1][0]       = 0;
  sim->figsize.range[1][1]       = 100;
  sim->figtimer.range[0][0]      = -200;
  sim->figtimer.range[0][1]      = 0;
  sim->figtimer.range[1][0]      = 0;
  sim->figtimer.range[1][1]      = 0.4f;

  // init x axis on history figures (do not show yet)
  for (int n = 0; n < 6; n++) {
    for (int i = 0; i < mjMAXLINEPNT; i++) {
      sim->figtimer.linedata[n][2 * i] = -i;
      sim->figsize.linedata[n][2 * i]  = -i;
    }
  }
}

// update profiler figures
void UpdateProfiler(mj::Simulate* sim, const mjModel* m, const mjData* d) {
  // reset lines in Constraint and Cost figures
  memset(sim->figconstraint.linepnt, 0, mjMAXLINE * sizeof(int));
  memset(sim->figcost.linepnt, 0, mjMAXLINE * sizeof(int));

  // number of islands that have diagnostics
  int nisland = d->nefc ? mjMAX(1, mjMIN(d->nisland, mjNISLAND)) : 0;

  // iterate over islands
  for (int k = 0; k < nisland; k++) {
    // ==== update Constraint ("Counts") figure

    // number of points to plot, starting line
    int npoints = mjMIN(mjMIN(d->solver_niter[k], mjNSOLVER), mjMAXLINEPNT);
    int start   = kConstraintNum * k;

    sim->figconstraint.linepnt[start + 0] = npoints;
    for (int i = 1; i < kConstraintNum; i++) { sim->figconstraint.linepnt[start + i] = npoints; }
    if (m->opt.solver == mjSOL_PGS) {
      sim->figconstraint.linepnt[start + 3] = 0;
      sim->figconstraint.linepnt[start + 4] = 0;
    }
    if (m->opt.solver == mjSOL_CG) { sim->figconstraint.linepnt[start + 4] = 0; }
    for (int i = 0; i < npoints; i++) {
      // x
      sim->figconstraint.linedata[start + 0][2 * i] = i;
      sim->figconstraint.linedata[start + 1][2 * i] = i;
      sim->figconstraint.linedata[start + 2][2 * i] = i;
      sim->figconstraint.linedata[start + 3][2 * i] = i;
      sim->figconstraint.linedata[start + 4][2 * i] = i;

      // y
      int nefc = nisland == 1 ? d->nefc : d->island_nefc[k];
      sim->figconstraint.linedata[start + 0][2 * i + 1] = nefc;
      const mjSolverStat* stat                          = d->solver + k * mjNSOLVER + i;
      sim->figconstraint.linedata[start + 1][2 * i + 1] = stat->nactive;
      sim->figconstraint.linedata[start + 2][2 * i + 1] = stat->nchange;
      sim->figconstraint.linedata[start + 3][2 * i + 1] = stat->neval;
      sim->figconstraint.linedata[start + 4][2 * i + 1] = stat->nupdate;
    }

    // update cost figure
    start                           = kCostNum * k;
    sim->figcost.linepnt[start + 0] = npoints;
    for (int i = 1; i < kCostNum; i++) { sim->figcost.linepnt[start + i] = npoints; }
    if (m->opt.solver == mjSOL_PGS) {
      sim->figcost.linepnt[start + 1] = 0;
      sim->figcost.linepnt[start + 2] = 0;
    }

    for (int i = 0; i < npoints; i++) {
      // x
      sim->figcost.linedata[start + 0][2 * i] = i;
      sim->figcost.linedata[start + 1][2 * i] = i;
      sim->figcost.linedata[start + 2][2 * i] = i;

      // y
      const mjSolverStat* stat                    = d->solver + k * mjNSOLVER + i;
      sim->figcost.linedata[start + 0][2 * i + 1] = mju_log10(mju_max(mjMINVAL, stat->improvement));
      sim->figcost.linedata[start + 1][2 * i + 1] = mju_log10(mju_max(mjMINVAL, stat->gradient));
      sim->figcost.linedata[start + 2][2 * i + 1] = mju_log10(mju_max(mjMINVAL, stat->lineslope));
    }
  }

  // get timers: total, collision, prepare, solve, other
  mjtNum total  = d->timer[mjTIMER_STEP].duration - sim->timer_prev_[mjTIMER_STEP].duration;
  int    number = d->timer[mjTIMER_STEP].number - sim->timer_prev_[mjTIMER_STEP].number;
  int    prev_forward_number   = sim->timer_prev_[mjTIMER_FORWARD].number;
  mjtNum prev_forward_duration = sim->timer_prev_[mjTIMER_FORWARD].duration;
  if (!number) {
    total  = d->timer[mjTIMER_FORWARD].duration - prev_forward_duration;
    number = d->timer[mjTIMER_FORWARD].number - prev_forward_number;
  }

  if (number > 0) {  // skip update if no measurements
    float tdata[5] = {static_cast<float>(total / number),
                      static_cast<float>((d->timer[mjTIMER_POS_COLLISION].duration -
                                          sim->timer_prev_[mjTIMER_POS_COLLISION].duration) /
                                         number),
                      static_cast<float>((d->timer[mjTIMER_POS_MAKE].duration -
                                          sim->timer_prev_[mjTIMER_POS_MAKE].duration +
                                          d->timer[mjTIMER_POS_PROJECT].duration -
                                          sim->timer_prev_[mjTIMER_POS_PROJECT].duration) /
                                         number),
                      static_cast<float>((d->timer[mjTIMER_CONSTRAINT].duration -
                                          sim->timer_prev_[mjTIMER_CONSTRAINT].duration) /
                                         number),
                      0};
    tdata[4]       = tdata[0] - tdata[1] - tdata[2] - tdata[3];

    // update figtimer
    int pnt = mjMIN(201, sim->figtimer.linepnt[0] + 1);
    for (int n = 0; n < 5; n++) {
      // shift data
      for (int i = pnt - 1; i > 0; i--) {
        sim->figtimer.linedata[n][2 * i + 1] = sim->figtimer.linedata[n][2 * i - 1];
      }

      // assign new
      sim->figtimer.linepnt[n]     = pnt;
      sim->figtimer.linedata[n][1] = tdata[n];
    }
  }


  // get total number of iterations and nonzeros
  mjtNum sqrt_nnz     = 0;
  int    solver_niter = 0;
  for (int island = 0; island < nisland; island++) {
    sqrt_nnz     += d->solver_nnz[island];
    solver_niter += d->solver_niter[island];
  }
  sqrt_nnz = mju_sqrt(sqrt_nnz);

  // get sizes: nv, nbody, nefc, sqrt(nnz), ncon, iter
  int   nv       = mjENABLED(mjENBL_SLEEP) ? d->nv_awake : m->nv;
  int   nbody    = mjENABLED(mjENBL_SLEEP) ? d->nbody_awake : m->nbody;
  float sdata[6] = {static_cast<float>(nv),
                    static_cast<float>(nbody),
                    static_cast<float>(d->nefc),
                    static_cast<float>(sqrt_nnz),
                    static_cast<float>(d->ncon),
                    static_cast<float>(solver_niter) / nisland};

  // update figsize
  int pnt = mjMIN(201, sim->figsize.linepnt[0] + 1);
  for (int n = 0; n < 6; n++) {
    // shift data
    for (int i = pnt - 1; i > 0; i--) {
      sim->figsize.linedata[n][2 * i + 1] = sim->figsize.linedata[n][2 * i - 1];
    }

    // assign new
    sim->figsize.linepnt[n]     = pnt;
    sim->figsize.linedata[n][1] = sdata[n];
  }
}

// show profiler figures
void ShowProfiler(mj::Simulate* sim, mjrRect rect) {
  mjrRect viewport = {rect.left + rect.width - rect.width / 4,
                      rect.bottom,
                      rect.width / 4,
                      rect.height / 4};
  mjr_figure(viewport, &sim->figtimer, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height / 4;
  mjr_figure(viewport, &sim->figsize, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height / 4;
  mjr_figure(viewport, &sim->figcost, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height / 4;
  mjr_figure(viewport, &sim->figconstraint, &sim->platform_ui->mjr_context());
}


// init sensor figure
void InitializeSensor(mj::Simulate* sim) {
  mjvFigure& figsensor = sim->figsensor;

  // set figure to default
  mjv_defaultFigure(&figsensor);
  figsensor.figurergba[3] = 0.5f;

  // set flags
  figsensor.flg_extend    = 1;
  figsensor.flg_barplot   = 1;
  figsensor.flg_symmetric = 1;

  // title
  mju::strcpy_arr(figsensor.title, "Sensor data");

  // y-tick number format
  mju::strcpy_arr(figsensor.yformat, "%.1f");

  // grid size
  figsensor.gridsize[0] = 2;
  figsensor.gridsize[1] = 3;

  // minimum range
  figsensor.range[0][0] = 0;
  figsensor.range[0][1] = 0;
  figsensor.range[1][0] = -1;
  figsensor.range[1][1] = 1;
}

// update sensor figure
void UpdateSensor(mj::Simulate* sim, const mjModel* m, const mjData* d) {
  mjvFigure&       figsensor = sim->figsensor;
  static const int maxline   = 10;

  // clear linepnt
  for (int i = 0; i < maxline; i++) { figsensor.linepnt[i] = 0; }

  // start with line 0
  int lineid = 0;

  // loop over sensors
  for (int n = 0; n < m->nsensor; n++) {
    // go to next line if type is different
    if (n > 0 && m->sensor_type[n] != m->sensor_type[n - 1]) {
      lineid = mjMIN(lineid + 1, maxline - 1);
    }

    // get info about this sensor
    mjtNum cutoff = (m->sensor_cutoff[n] > 0 ? m->sensor_cutoff[n] : 1);
    int    adr    = m->sensor_adr[n];
    int    dim    = m->sensor_dim[n];

    // data pointer in line
    int p = figsensor.linepnt[lineid];

    // fill in data for this sensor
    for (int i = 0; i < dim; i++) {
      // check size
      if ((p + 2 * i) >= mjMAXLINEPNT / 2) { break; }

      // x
      figsensor.linedata[lineid][2 * p + 4 * i]     = adr + i;
      figsensor.linedata[lineid][2 * p + 4 * i + 2] = adr + i;

      // y
      figsensor.linedata[lineid][2 * p + 4 * i + 1] = 0;
      figsensor.linedata[lineid][2 * p + 4 * i + 3] = d->sensordata[adr + i] / cutoff;
    }

    // update linepnt
    figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT - 1, figsensor.linepnt[lineid] + 2 * dim);
  }
}

// show sensor figure
void ShowSensor(mj::Simulate* sim, mjrRect rect) {
  // constant width with and without profiler
  int width = sim->profiler ? rect.width / 3 : rect.width / 4;

  // render figure on the right
  mjrRect viewport = {rect.left + rect.width - width, rect.bottom, width, rect.height / 3};

  // if image sensor selected, show sensor image instead
  if (sim->image_sensor_count > 0 && sim->selected_image_sensor >= 0 && sim->sensor_image) {
    // render sensor image - viewport dimensions MUST match image buffer dimensions
    int img_w = sim->sensor_image_width;
    int img_h = sim->sensor_image_height;
    if (img_w > 0 && img_h > 0) {
      // center the image in the available viewport area
      mjrRect img_viewport = {
          viewport.left + (viewport.width - img_w) / 2,
          viewport.bottom + (viewport.height - img_h) / 2,
          img_w,  // MUST match actual buffer width
          img_h   // MUST match actual buffer height
      };
      mjr_drawPixels(sim->sensor_image.get(),
                     nullptr,
                     img_viewport,
                     &sim->platform_ui->mjr_context());
    }
  } else {
    mjr_figure(viewport, &sim->figsensor, &sim->platform_ui->mjr_context());
  }
}

// forward declaration
void InitializeSensorImage(mj::Simulate* sim, const mjModel* m);

// Detect image sensors in model
// A sensor is an image if: mjSENS_USER and intprm[0]*intprm[1]*3 == dim
void DetectImageSensors(mj::Simulate* sim, const mjModel* m) {
  sim->image_sensor_count = 0;
  sim->image_sensor_indices.clear();
  sim->image_sensor_names.clear();
  sim->selected_image_sensor = -1;
  sim->sensor_image.reset();

  if (!m) return;

  for (int i = 0; i < m->nsensor; i++) {
    if (m->sensor_type[i] == mjSENS_USER) {
      // sensor_intprm stores [width, height, unused] (mjNSENS=3 per sensor)
      int width  = m->sensor_intprm[i * mjNSENS];
      int height = m->sensor_intprm[i * mjNSENS + 1];
      int dim    = m->sensor_dim[i];

      // Check if this is an image sensor: width*height*3 == dim
      if (width > 0 && height > 0 && width * height * 3 == dim) {
        const char* name = mj_id2name(m, mjOBJ_SENSOR, i);
        sim->image_sensor_indices.push_back(i);
        sim->image_sensor_names.push_back(name ? name : "");
        sim->image_sensor_count++;
      }
    }
  }

  // Auto-select first image sensor if any found
  if (sim->image_sensor_count > 0) {
    sim->selected_image_sensor = 0;
    InitializeSensorImage(sim, m);
  }
}

// initialize sensor image for selected sensor
// Reads image resolution from sensor_intprm (set by Python code)
void InitializeSensorImage(mj::Simulate* sim, const mjModel* m) {
  if (sim->selected_image_sensor < 0 || sim->selected_image_sensor >= sim->image_sensor_count) {
    sim->sensor_image.reset();
    return;
  }

  int sensor_idx = sim->image_sensor_indices[sim->selected_image_sensor];

  // Read width/height from sensor_intprm
  int width  = m->sensor_intprm[sensor_idx * mjNSENS];
  int height = m->sensor_intprm[sensor_idx * mjNSENS + 1];

  sim->sensor_image_width  = width;
  sim->sensor_image_height = height;

  // Allocate image buffer (RGB)
  int img_size      = width * height * 3;
  sim->sensor_image = std::make_unique<unsigned char[]>(img_size);
}

// update sensor image with current sensor data
// Uses version number (last element) to avoid reading during MuJoCo reset
void UpdateSensorImage(mj::Simulate* sim, const mjModel* m, const mjData* d) {
  if (sim->selected_image_sensor < 0 || !sim->sensor_image) return;

  int sensor_idx = sim->image_sensor_indices[sim->selected_image_sensor];
  int adr        = m->sensor_adr[sensor_idx];
  int w          = sim->sensor_image_width;
  int h          = sim->sensor_image_height;
  int img_size   = w * h * 3;

  // Check version (last element) - 0 means MuJoCo reset, skip copy
  int current_version = (int)d->sensordata[adr + img_size - 1];
  if (current_version == 0 || current_version == sim->sensor_image_last_seq) { return; }
  sim->sensor_image_last_seq = current_version;

  // Copy image data (skip last element = version)
  for (int i = 0; i < img_size - 1; i++) {
    mjtNum val = d->sensordata[adr + i];

    sim->sensor_image[i] = (unsigned char)mjMIN(255, mjMAX(0, (int)val));
  }
}


void ShowFigure(mj::Simulate* sim, mjrRect viewport, mjvFigure* fig) {
  mjr_figure(viewport, fig, &sim->platform_ui->mjr_context());
}

void ShowOverlayText(mj::Simulate* sim,
                     mjrRect       viewport,
                     int           font,
                     int           gridpos,
                     std::string   text1,
                     std::string   text2) {
  mjr_overlay(font,
              gridpos,
              viewport,
              text1.c_str(),
              text2.c_str(),
              &sim->platform_ui->mjr_context());
}

void ShowImage(mj::Simulate* sim, mjrRect viewport, const unsigned char* image) {
  mjr_drawPixels(image, nullptr, viewport, &sim->platform_ui->mjr_context());
}

// load state from history buffer
static void LoadScrubState(mj::Simulate* sim) {
  // get index into circular buffer
  int i = (sim->scrub_index + sim->history_cursor_) % sim->nhistory_;
  i     = (i + sim->nhistory_) % sim->nhistory_;

  // load state
  mjtNum* state = &sim->history_[i * sim->state_size_];
  mj_setState(sim->m_, sim->d_, state, mjSTATE_INTEGRATION);

  // call forward dynamics
  mj_forward(sim->m_, sim->d_);
}

// update an entire section of ui0
static void mjui0_update_section(mj::Simulate* sim, int section) {
  mjui_update(section, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
}

// prepare info text
void UpdateInfoText(mj::Simulate*  sim,
                    const mjModel* m,
                    const mjData*  d,
                    char (&title)[mj::Simulate::kMaxFilenameLength],
                    char (&content)[mj::Simulate::kMaxFilenameLength]) {
  char tmp[20];

  // number of islands with statistics
  int nisland = mjMAX(1, mjMIN(d->nisland, mjNISLAND));

  // compute solver error (maximum over islands)
  mjtNum solerr = 0;
  for (int i = 0; i < nisland; i++) {
    mjtNum solerr_i = 0;
    if (d->solver_niter[i]) {
      int ind = mjMIN(d->solver_niter[i], mjNSOLVER) - 1;

      const mjSolverStat* stat = d->solver + i * mjNSOLVER + ind;

      solerr_i = mju_min(stat->improvement, stat->gradient);
      if (solerr_i == 0) { solerr_i = mju_max(stat->improvement, stat->gradient); }
    }
    solerr = mju_max(solerr, solerr_i);
  }
  solerr = mju_log10(mju_max(mjMINVAL, solerr));

  // format FPS text
  char fps[10];
  if (sim->fps_ < 1) {
    mju::sprintf_arr(fps, "%0.1f ", sim->fps_);
  } else {
    mju::sprintf_arr(fps, "%.0f ", sim->fps_);
  }

  // total iterations of all islands with statistics
  int solver_niter = 0;
  for (int i = 0; i < nisland; i++) { solver_niter += d->solver_niter[i]; }

  mjtNum step_duration = d->timer[mjTIMER_STEP].duration - sim->timer_prev_[mjTIMER_STEP].duration;
  int    step_number   = d->timer[mjTIMER_STEP].number - sim->timer_prev_[mjTIMER_STEP].number;
  mjtNum forward_duration =
      d->timer[mjTIMER_FORWARD].duration - sim->timer_prev_[mjTIMER_FORWARD].duration;
  int forward_number = d->timer[mjTIMER_FORWARD].number - sim->timer_prev_[mjTIMER_FORWARD].number;

  // prepare info text
  mju::strcpy_arr(title, "Time\nSize\nCPU\nSolver   \nFPS\nMemory");
  mju::sprintf_arr(content,
                   "%-9.3f\n%d  (%d con)\n%.3f\n%.1f  (%d it)\n%s\n%.1f%% of %s",
                   d->time,
                   d->nefc,
                   d->ncon,
                   sim->run ? step_duration / mjMAX(1, step_number)
                            : forward_duration / mjMAX(1, forward_number),
                   solerr,
                   solver_niter,
                   fps,
                   100 * d->maxuse_arena / (double)(d->narena),
                   mju_writeNumBytes(d->narena));

  // add Energy if enabled
  {
    if (mjENABLED(mjENBL_ENERGY)) {
      mju::sprintf_arr(tmp, "\n%.3f", d->energy[0] + d->energy[1]);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if (mjENABLED(mjENBL_FWDINV)) {
      mju::sprintf_arr(tmp,
                       "\n%.1f %.1f",
                       mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[0])),
                       mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[1])));
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nFwdInv");
    }

    // add islands if enabled
    if (!mjDISABLED(mjDSBL_ISLAND)) {
      mju::sprintf_arr(tmp, "\n%d", d->nisland);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nIslands");
    }
  }
}

// sprintf forwarding, to avoid compiler warning in x-macro
void PrintField(char (&str)[mjMAXUINAME], void* ptr) {
  mju::sprintf_arr(str, "%g", *static_cast<mjtNum*>(ptr));
}

// update watch
void UpdateWatch(mj::Simulate* sim, const mjModel* m, const mjData* d) {
  // clear
  sim->ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
  mju::strcpy_arr(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

// find specified field in mjData arrays, update value
#define X(TYPE, NAME, NR, NC)                                                            \
  if (!mju::strcmp_arr(#NAME, sim->field) && !mju::strcmp_arr(#TYPE, "mjtNum")) {        \
    if (sim->index >= 0 && sim->index < m->NR * NC) {                                    \
      PrintField(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], d->NAME + sim->index); \
    } else {                                                                             \
      mju::strcpy_arr(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid index"); \
    }                                                                                    \
    return;                                                                              \
  }

  MJDATA_POINTERS
#undef X
}


//---------------------------------- UI construction -----------------------------------------------

// make physics section of UI
void MakePhysicsSection(mj::Simulate* sim) {
  mjOption* opt = sim->is_passive_ ? &sim->m_passive_->opt : &sim->m_->opt;
  // clang-format off
  mjuiDef defPhysics[] = {
    {mjITEM_SECTION,   "Physics",       mjPRESERVE, nullptr,          "AP"},
    {mjITEM_SELECT,    "Integrator",    2, &(opt->integrator),        "Euler\nRK4\nimplicit\nimplicitfast\ndiscrete"},
    {mjITEM_SELECT,    "Cone",          2, &(opt->cone),              "Pyramidal\nElliptic"},
    {mjITEM_SELECT,    "Jacobian",      2, &(opt->jacobian),          "Dense\nSparse\nAuto"},
    {mjITEM_SELECT,    "Solver",        2, &(opt->solver),            "PGS\nCG\nNewton"},
    {mjITEM_SEPARATOR, "Algorithmic Parameters", mjPRESERVE},
    {mjITEM_EDITNUM,   "Timestep",      2, &(opt->timestep),          "1 0 1"},
    {mjITEM_EDITINT,   "Iterations",    2, &(opt->iterations),        "1 0 1000"},
    {mjITEM_EDITNUM,   "Tolerance",     2, &(opt->tolerance),         "1 0 1"},
    {mjITEM_EDITINT,   "LS Iter",       2, &(opt->ls_iterations),     "1 0 100"},
    {mjITEM_EDITNUM,   "LS Tol",        2, &(opt->ls_tolerance),      "1 0 0.1"},
    {mjITEM_EDITINT,   "Noslip Iter",   2, &(opt->noslip_iterations), "1 0 1000"},
    {mjITEM_EDITNUM,   "Noslip Tol",    2, &(opt->noslip_tolerance),  "1 0 1"},
    {mjITEM_EDITINT,   "CCD Iter",      2, &(opt->ccd_iterations),    "1 0 1000"},
    {mjITEM_EDITNUM,   "CCD Tol",       2, &(opt->ccd_tolerance),     "1 0 1"},
    {mjITEM_EDITNUM,   "Sleep Tol",     2, &(opt->sleep_tolerance),   "1 0 1"},
    {mjITEM_EDITINT,   "SDF Iter",      2, &(opt->sdf_iterations),    "1 1 20"},
    {mjITEM_EDITINT,   "SDF Init",      2, &(opt->sdf_initpoints),    "1 1 100"},
    {mjITEM_SEPARATOR, "Physical Parameters", mjPRESERVE},
    {mjITEM_EDITNUM,   "Gravity",       2, opt->gravity,              "3"},
    {mjITEM_EDITNUM,   "Wind",          2, opt->wind,                 "3"},
    {mjITEM_EDITNUM,   "Magnetic",      2, opt->magnetic,             "3"},
    {mjITEM_EDITNUM,   "Density",       2, &(opt->density),           "1"},
    {mjITEM_EDITNUM,   "Viscosity",     2, &(opt->viscosity),         "1"},
    {mjITEM_EDITNUM,   "Imp Ratio",     2, &(opt->impratio),          "1"},
    {mjITEM_SEPARATOR, "Disable Flags", mjPRESERVE},
    {mjITEM_END}
  };
  mjuiDef defEnableFlags[] = {
      {mjITEM_SEPARATOR, "Enable Flags", mjPRESERVE},
      {mjITEM_END}
  };
  mjuiDef defOverride[] = {
    {mjITEM_SEPARATOR, "Contact Override", mjPRESERVE},
    {mjITEM_EDITNUM,   "Margin",        2, &(opt->o_margin),          "1"},
    {mjITEM_EDITNUM,   "Sol Imp",       2, &(opt->o_solimp),          "5"},
    {mjITEM_EDITNUM,   "Sol Ref",       2, &(opt->o_solref),          "2"},
    {mjITEM_EDITNUM,   "Friction",      2, &(opt->o_friction),        "5"},
    {mjITEM_END}
  };
  mjuiDef defDisableActuator[] = {
    {mjITEM_SEPARATOR, "Actuator Group Enable", mjPRESERVE},
    {mjITEM_CHECKBYTE,  "Act Group 0",  2, sim->enableactuator+0,     ""},
    {mjITEM_CHECKBYTE,  "Act Group 1",  2, sim->enableactuator+1,     ""},
    {mjITEM_CHECKBYTE,  "Act Group 2",  2, sim->enableactuator+2,     ""},
    {mjITEM_CHECKBYTE,  "Act Group 3",  2, sim->enableactuator+3,     ""},
    {mjITEM_CHECKBYTE,  "Act Group 4",  2, sim->enableactuator+4,     ""},
    {mjITEM_CHECKBYTE,  "Act Group 5",  2, sim->enableactuator+5,     ""},
    {mjITEM_END}
  };
  // clang-format on

  // add physics
  mjui_add(&sim->ui0, defPhysics);

  // add flags programmatically
  mjuiDef defFlag[] = {
      {mjITEM_CHECKINT, "", 2, nullptr, ""},
      {mjITEM_END}
  };
  for (int i = 0; i < mjNDISABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjDISABLESTRING[i]);
    defFlag[0].pdata = sim->disable + i;
    mjui_add(&sim->ui0, defFlag);
  }
  mjui_add(&sim->ui0, defEnableFlags);
  for (int i = 0; i < mjNENABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjENABLESTRING[i]);
    defFlag[0].pdata = sim->enable + i;
    mjui_add(&sim->ui0, defFlag);
  }
  // add contact override
  mjui_add(&sim->ui0, defOverride);

  // add actuator group enable/disable
  mjui_add(&sim->ui0, defDisableActuator);

  // make some subsections closed by default
  for (int i = 0; i < sim->ui0.sect[SECT_PHYSICS].nitem; i++) {
    mjuiItem* it = sim->ui0.sect[SECT_PHYSICS].item + i;

    // close less useful subsections
    if (it->type == mjITEM_SEPARATOR) {
      if (mju::strcmp_arr(it->name, "Actuator Group Enable") &&
          mju::strcmp_arr(it->name, "Contact Override") &&
          mju::strcmp_arr(it->name, "Physical Parameters")) {
        it->state = mjSEPCLOSED + 1;
      }
    }
  }
}


// make rendering section of UI
void MakeRenderingSection(mj::Simulate* sim, const mjModel* m) {
  mjuiDef defRendering[] = {
      {mjITEM_SECTION, "Rendering", mjPRESERVE, nullptr, "AR"},
      {mjITEM_SELECT, "Camera", 2, &(sim->camera), "Free\nTracking"},
      {mjITEM_SELECT,
       "Label", 2,
       &(sim->opt.label),
       "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\n"
       "Actuator\nConstraint\nFlex\nSkin\nSelection\nSel Pnt\nContact\nForce\nIsland"},
      {mjITEM_SELECT,
       "Frame", 2,
       &(sim->opt.frame),
       "None\nBody\nGeom\nSite\nCamera\nLight\nContact\nWorld"},
      {mjITEM_BUTTON, "Copy camera", 2, nullptr, ""},
      {mjITEM_SEPARATOR, "Model Elements", 1},
      {mjITEM_END}
  };
  mjuiDef defOpenGL[] = {
      {mjITEM_SEPARATOR, "OpenGL Effects", 1},
      {mjITEM_END}
  };

  // add model cameras, up to UI limit
  for (int i = 0; i < mjMIN(m->ncam, mjMAXUIMULTI - 2); i++) {
    // prepare name
    char        camname[mjMAXUINAME] = "\n";
    const char* name                 = mj_id2name(m, mjOBJ_CAMERA, i);
    if (name) {
      mju::strcat_arr(camname, name);
    } else {
      mju::sprintf_arr(camname, "\nCamera %d", i);
    }

    // check string length
    if (mju::strlen_arr(camname) + mju::strlen_arr(defRendering[1].other) >= mjMAXUITEXT - 1) {
      break;
    }

    // add camera
    mju::strcat_arr(defRendering[1].other, camname);
  }

  // add rendering standard
  mjui_add(&sim->ui0, defRendering);

  // add flags programmatically
  mjuiDef defFlag[] = {
      {mjITEM_CHECKBYTE, "", 2, nullptr, ""},
      {mjITEM_END}
  };
  for (int i = 0; i < mjNVISFLAG; i++) {
    // set name
    mju::strcpy_arr(defFlag[0].name, mjVISSTRING[i][0]);

    // set shortcut and data
    if (mjVISSTRING[i][2][0]) {
      mju::sprintf_arr(defFlag[0].other, " %s", mjVISSTRING[i][2]);
    } else {
      mju::sprintf_arr(defFlag[0].other, "");
    }
    defFlag[0].pdata = sim->opt.flags + i;
    mjui_add(&sim->ui0, defFlag);
  }

  // create tree slider
  mjuiDef defTree[] = {
      {mjITEM_SLIDERINT, "Tree depth", 2, &sim->opt.bvh_depth, "0 20"},
      {mjITEM_SLIDERINT, "Flex layer", 2, &sim->opt.flex_layer, "0 10"},
      {mjITEM_END}
  };
  mjui_add(&sim->ui0, defTree);

  // add image sensor selector if image sensors exist
  if (sim->image_sensor_count > 0) {
    // build options string: "All\nSensor1\nSensor2\n..."
    static char sensor_options[mjMAXUITEXT];
    mju::strcpy_arr(sensor_options, "All");
    for (int i = 0; i < sim->image_sensor_count && i < mjMAXUIMULTI - 1; i++) {
      mju::strcat_arr(sensor_options, "\n");
      if ((int)sim->image_sensor_names[i].length() < mjMAXUINAME) {
        mju::strcat_arr(sensor_options, sim->image_sensor_names[i].c_str());
      } else {
        char truncated[mjMAXUINAME];
        snprintf(truncated, mjMAXUINAME, "Image %d", i);
        mju::strcat_arr(sensor_options, truncated);
      }
    }

    // selected_image_sensor is -1 for None, 0+ for actual sensor
    // but mjITEM_SELECT uses 0-based index, so we need to offset
    sim->image_sensor_ui_selection = sim->selected_image_sensor + 1;

    mjuiDef defImageSensor[] = {
        {mjITEM_SELECT, "Sensor", 2, &sim->image_sensor_ui_selection, ""},
        {mjITEM_END}
    };
    // copy options string into the struct's other field
    mju::strcpy_arr(defImageSensor[0].other, sensor_options);
    mjui_add(&sim->ui0, defImageSensor);
  }

  // add rendering flags
  mjui_add(&sim->ui0, defOpenGL);
  for (int i = 0; i < mjNRNDFLAG; i++) {
    // set name
    mju::strcpy_arr(defFlag[0].name, mjRNDSTRING[i][0]);

    // set shortcut and data
    if (mjRNDSTRING[i][2][0]) {
      mju::sprintf_arr(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
    } else {
      mju::sprintf_arr(defFlag[0].other, "");
    }
    defFlag[0].pdata = sim->scn.flags + i;
    mjui_add(&sim->ui0, defFlag);
  }
}

// make visualization section of UI
void MakeVisualizationSection(mj::Simulate* sim, const mjModel* m) {
  mjStatistic* stat = sim->is_passive_ ? &sim->m_passive_->stat : &sim->m_->stat;
  mjVisual*    vis  = sim->is_passive_ ? &sim->m_passive_->vis : &sim->m_->vis;

  // clang-format off
  mjuiDef defVisualization[] = {
    {mjITEM_SECTION,   "Visualization", mjPRESERVE, nullptr, "AV"},
    {mjITEM_SEPARATOR, "Headlight",  1},
    {mjITEM_RADIO,     "Active",          2, &(vis->headlight.active),     "Off\nOn"},
    {mjITEM_EDITFLOAT, "Ambient",         2, &(vis->headlight.ambient),    "3"},
    {mjITEM_EDITFLOAT, "Diffuse",         2, &(vis->headlight.diffuse),    "3"},
    {mjITEM_EDITFLOAT, "Specular",        2, &(vis->headlight.specular),   "3"},
    {mjITEM_SEPARATOR, "Free Camera", 1},
    {mjITEM_RADIO,     "Orthographic",    2, &sim->cam.orthographic,       "No\nYes"},
    {mjITEM_EDITFLOAT, "Field of view",   2, &(vis->global.fovy),          "1"},
    {mjITEM_EDITNUM,   "Center",          2, &(stat->center),              "3"},
    {mjITEM_EDITFLOAT, "Azimuth",         2, &(vis->global.azimuth),       "1"},
    {mjITEM_EDITFLOAT, "Elevation",       2, &(vis->global.elevation),     "1"},
    {mjITEM_BUTTON,    "Align",           2, nullptr,                      "CA"},
    {mjITEM_SEPARATOR, "Global",  1},
    {mjITEM_EDITNUM,   "Extent",          2, &(stat->extent),              "1"},
    {mjITEM_RADIO,     "Inertia",         2, &(vis->global.ellipsoidinertia), "Box\nEllipsoid"},
    {mjITEM_RADIO,     "BVH active",      5, &(vis->global.bvactive), "False\nTrue"},
    {mjITEM_SEPARATOR, "Map",  1},
    {mjITEM_EDITFLOAT, "Stiffness",       2, &(vis->map.stiffness),        "1"},
    {mjITEM_EDITFLOAT, "Rot stiffness",   2, &(vis->map.stiffnessrot),     "1"},
    {mjITEM_EDITFLOAT, "Force",           2, &(vis->map.force),            "1"},
    {mjITEM_EDITFLOAT, "Torque",          2, &(vis->map.torque),           "1"},
    {mjITEM_EDITFLOAT, "Alpha",           2, &(vis->map.alpha),            "1"},
    {mjITEM_EDITFLOAT, "Fog start",       2, &(vis->map.fogstart),         "1"},
    {mjITEM_EDITFLOAT, "Fog end",         2, &(vis->map.fogend),           "1"},
    {mjITEM_EDITFLOAT, "Z near",          2, &(vis->map.znear),            "1"},
    {mjITEM_EDITFLOAT, "Z far",           2, &(vis->map.zfar),             "1"},
    {mjITEM_EDITFLOAT, "Haze",            2, &(vis->map.haze),             "1"},
    {mjITEM_EDITFLOAT, "Shadow clip",     2, &(vis->map.shadowclip),       "1"},
    {mjITEM_EDITFLOAT, "Shadow scale",    2, &(vis->map.shadowscale),      "1"},
    {mjITEM_SEPARATOR, "Scale", mjPRESERVE},
    {mjITEM_EDITNUM,   "All (meansize)",  2, &(stat->meansize),            "1"},
    {mjITEM_EDITFLOAT, "Force width",     2, &(vis->scale.forcewidth),     "1"},
    {mjITEM_EDITFLOAT, "Contact width",   2, &(vis->scale.contactwidth),   "1"},
    {mjITEM_EDITFLOAT, "Contact height",  2, &(vis->scale.contactheight),  "1"},
    {mjITEM_EDITFLOAT, "Connect",         2, &(vis->scale.connect),        "1"},
    {mjITEM_EDITFLOAT, "Com",             2, &(vis->scale.com),            "1"},
    {mjITEM_EDITFLOAT, "Camera",          2, &(vis->scale.camera),         "1"},
    {mjITEM_EDITFLOAT, "Light",           2, &(vis->scale.light),          "1"},
    {mjITEM_EDITFLOAT, "Select point",    2, &(vis->scale.selectpoint),    "1"},
    {mjITEM_EDITFLOAT, "Joint length",    2, &(vis->scale.jointlength),    "1"},
    {mjITEM_EDITFLOAT, "Joint width",     2, &(vis->scale.jointwidth),     "1"},
    {mjITEM_EDITFLOAT, "Actuator length", 2, &(vis->scale.actuatorlength), "1"},
    {mjITEM_EDITFLOAT, "Actuator width",  2, &(vis->scale.actuatorwidth),  "1"},
    {mjITEM_EDITFLOAT, "Frame length",    2, &(vis->scale.framelength),    "1"},
    {mjITEM_EDITFLOAT, "Frame width",     2, &(vis->scale.framewidth),     "1"},
    {mjITEM_EDITFLOAT, "Constraint",      2, &(vis->scale.constraint),     "1"},
    {mjITEM_EDITFLOAT, "Slider-crank",    2, &(vis->scale.slidercrank),    "1"},
    {mjITEM_SEPARATOR, "RGBA", mjPRESERVE},
    {mjITEM_EDITFLOAT, "fog",             2, &(vis->rgba.fog),              "4"},
    {mjITEM_EDITFLOAT, "haze",            2, &(vis->rgba.haze),             "4"},
    {mjITEM_EDITFLOAT, "force",           2, &(vis->rgba.force),            "4"},
    {mjITEM_EDITFLOAT, "inertia",         2, &(vis->rgba.inertia),          "4"},
    {mjITEM_EDITFLOAT, "joint",           2, &(vis->rgba.joint),            "4"},
    {mjITEM_EDITFLOAT, "actuator",        2, &(vis->rgba.actuator),         "4"},
    {mjITEM_EDITFLOAT, "actnegative",     2, &(vis->rgba.actuatornegative), "4"},
    {mjITEM_EDITFLOAT, "actpositive",     2, &(vis->rgba.actuatorpositive), "4"},
    {mjITEM_EDITFLOAT, "com",             2, &(vis->rgba.com),              "4"},
    {mjITEM_EDITFLOAT, "camera",          2, &(vis->rgba.camera),           "4"},
    {mjITEM_EDITFLOAT, "light",           2, &(vis->rgba.light),            "4"},
    {mjITEM_EDITFLOAT, "selectpoint",     2, &(vis->rgba.selectpoint),      "4"},
    {mjITEM_EDITFLOAT, "connect",         2, &(vis->rgba.connect),          "4"},
    {mjITEM_EDITFLOAT, "contactpoint",    2, &(vis->rgba.contactpoint),     "4"},
    {mjITEM_EDITFLOAT, "contactforce",    2, &(vis->rgba.contactforce),     "4"},
    {mjITEM_EDITFLOAT, "contactfriction", 2, &(vis->rgba.contactfriction),  "4"},
    {mjITEM_EDITFLOAT, "contacttorque",   2, &(vis->rgba.contacttorque),    "4"},
    {mjITEM_EDITFLOAT, "contactgap",      2, &(vis->rgba.contactgap),       "4"},
    {mjITEM_EDITFLOAT, "rangefinder",     2, &(vis->rgba.rangefinder),      "4"},
    {mjITEM_EDITFLOAT, "constraint",      2, &(vis->rgba.constraint),       "4"},
    {mjITEM_EDITFLOAT, "slidercrank",     2, &(vis->rgba.slidercrank),      "4"},
    {mjITEM_EDITFLOAT, "crankbroken",     2, &(vis->rgba.crankbroken),      "4"},
    {mjITEM_EDITFLOAT, "frustum",         2, &(vis->rgba.frustum),          "4"},
    {mjITEM_EDITFLOAT, "bv",              2, &(vis->rgba.bv),               "4"},
    {mjITEM_EDITFLOAT, "bvactive",        2, &(vis->rgba.bvactive),         "4"},
    {mjITEM_END}
  };

  // add visualization section
  mjui_add(&sim->ui0, defVisualization);
}

// make group section of UI
void MakeGroupSection(mj::Simulate* sim) {
  mjuiDef defGroup[] = {
    {mjITEM_SECTION,    "Group enable",     mjPRESERVE, nullptr,            "AG"},
    {mjITEM_SEPARATOR,  "Geom groups",  1},
    {mjITEM_CHECKBYTE,  "Geom 0",           2, sim->opt.geomgroup,          " 0"},
    {mjITEM_CHECKBYTE,  "Geom 1",           2, sim->opt.geomgroup+1,        " 1"},
    {mjITEM_CHECKBYTE,  "Geom 2",           2, sim->opt.geomgroup+2,        " 2"},
    {mjITEM_CHECKBYTE,  "Geom 3",           2, sim->opt.geomgroup+3,        " 3"},
    {mjITEM_CHECKBYTE,  "Geom 4",           2, sim->opt.geomgroup+4,        " 4"},
    {mjITEM_CHECKBYTE,  "Geom 5",           2, sim->opt.geomgroup+5,        " 5"},
    {mjITEM_SEPARATOR,  "Site groups",  1},
    {mjITEM_CHECKBYTE,  "Site 0",           2, sim->opt.sitegroup,          "S0"},
    {mjITEM_CHECKBYTE,  "Site 1",           2, sim->opt.sitegroup+1,        "S1"},
    {mjITEM_CHECKBYTE,  "Site 2",           2, sim->opt.sitegroup+2,        "S2"},
    {mjITEM_CHECKBYTE,  "Site 3",           2, sim->opt.sitegroup+3,        "S3"},
    {mjITEM_CHECKBYTE,  "Site 4",           2, sim->opt.sitegroup+4,        "S4"},
    {mjITEM_CHECKBYTE,  "Site 5",           2, sim->opt.sitegroup+5,        "S5"},
    {mjITEM_SEPARATOR,  "Joint groups", 1},
    {mjITEM_CHECKBYTE,  "Joint 0",          2, sim->opt.jointgroup,         ""},
    {mjITEM_CHECKBYTE,  "Joint 1",          2, sim->opt.jointgroup+1,       ""},
    {mjITEM_CHECKBYTE,  "Joint 2",          2, sim->opt.jointgroup+2,       ""},
    {mjITEM_CHECKBYTE,  "Joint 3",          2, sim->opt.jointgroup+3,       ""},
    {mjITEM_CHECKBYTE,  "Joint 4",          2, sim->opt.jointgroup+4,       ""},
    {mjITEM_CHECKBYTE,  "Joint 5",          2, sim->opt.jointgroup+5,       ""},
    {mjITEM_SEPARATOR,  "Tendon groups",    1},
    {mjITEM_CHECKBYTE,  "Tendon 0",         2, sim->opt.tendongroup,        ""},
    {mjITEM_CHECKBYTE,  "Tendon 1",         2, sim->opt.tendongroup+1,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 2",         2, sim->opt.tendongroup+2,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 3",         2, sim->opt.tendongroup+3,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 4",         2, sim->opt.tendongroup+4,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 5",         2, sim->opt.tendongroup+5,      ""},
    {mjITEM_SEPARATOR,  "Actuator groups", 1},
    {mjITEM_CHECKBYTE,  "Actuator 0",       2, sim->opt.actuatorgroup,      ""},
    {mjITEM_CHECKBYTE,  "Actuator 1",       2, sim->opt.actuatorgroup+1,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 2",       2, sim->opt.actuatorgroup+2,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 3",       2, sim->opt.actuatorgroup+3,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 4",       2, sim->opt.actuatorgroup+4,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 5",       2, sim->opt.actuatorgroup+5,    ""},
    {mjITEM_SEPARATOR,  "Flex groups", 1},
    {mjITEM_CHECKBYTE,  "Flex 0",           2, sim->opt.flexgroup,          ""},
    {mjITEM_CHECKBYTE,  "Flex 1",           2, sim->opt.flexgroup+1,        ""},
    {mjITEM_CHECKBYTE,  "Flex 2",           2, sim->opt.flexgroup+2,        ""},
    {mjITEM_CHECKBYTE,  "Flex 3",           2, sim->opt.flexgroup+3,        ""},
    {mjITEM_CHECKBYTE,  "Flex 4",           2, sim->opt.flexgroup+4,        ""},
    {mjITEM_CHECKBYTE,  "Flex 5",           2, sim->opt.flexgroup+5,        ""},
    {mjITEM_SEPARATOR,  "Skin groups", 1},
    {mjITEM_CHECKBYTE,  "Skin 0",           2, sim->opt.skingroup,          ""},
    {mjITEM_CHECKBYTE,  "Skin 1",           2, sim->opt.skingroup+1,        ""},
    {mjITEM_CHECKBYTE,  "Skin 2",           2, sim->opt.skingroup+2,        ""},
    {mjITEM_CHECKBYTE,  "Skin 3",           2, sim->opt.skingroup+3,        ""},
    {mjITEM_CHECKBYTE,  "Skin 4",           2, sim->opt.skingroup+4,        ""},
    {mjITEM_CHECKBYTE,  "Skin 5",           2, sim->opt.skingroup+5,        ""},
    {mjITEM_END}
  };
  // clang-format on

  // add section
  mjui_add(&sim->ui0, defGroup);
}

// make logging section of UI
void MakeLoggingSection(mj::Simulate* sim) {
  mjLogConfig cfg  = mju_getLogConfig();
  sim->log_console = cfg.logto_console;
  sim->log_file    = cfg.logto_file;
  for (int i = 0; i < mjNTOPIC; i++) { sim->log_topics[i] = ((cfg.topics & (1 << i)) != 0); }

  // clang-format off
  mjuiDef defLogging[] = {
    {mjITEM_SECTION,    "Logging",          mjPRESERVE, nullptr,            "AL"},
    {mjITEM_CHECKBYTE,  "Console",          2, &sim->log_console,           ""},
    {mjITEM_CHECKBYTE,  "File",             2, &sim->log_file,              ""},
    {mjITEM_SEPARATOR,  "Info topics",      1},
    {mjITEM_END}
  };
  mjui_add(&sim->ui0, defLogging);

  mjuiDef defTopic[] = {
    {mjITEM_CHECKBYTE,  "",                 2, nullptr,                     ""},
    {mjITEM_END}
  };
  // clang-format on
  for (int i = 0; i < mjNTOPIC; i++) {
    mju::strcpy_arr(defTopic[0].name, mjTOPICSTRING[i]);
    defTopic[0].pdata = sim->log_topics + i;
    mjui_add(&sim->ui0, defTopic);
  }
}

// make joint section of UI
void MakeJointSection(mj::Simulate* sim) {
  mjuiDef defJoint[] = {
      {mjITEM_SECTION, "Joint", mjPRESERVE, nullptr, "AJ"},
      {mjITEM_END}
  };
  mjuiDef defSlider[] = {
      {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
      {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defJoint);
  defSlider[0].state = 4;

  // add scalar joints, exit if UI limit reached
  int itemcnt = 0;
  for (int i = 0; i < sim->jnt_type_.size() && itemcnt < mjMAXUIITEM; i++) {
    if ((sim->jnt_type_[i] == mjJNT_HINGE || sim->jnt_type_[i] == mjJNT_SLIDE)) {
      // skip if joint group is disabled
      if (!sim->opt.jointgroup[mjMAX(0, mjMIN(mjNGROUP - 1, sim->jnt_group_[i]))]) { continue; }

      // set data and name
      if (!sim->is_passive_) {
        defSlider[0].pdata = &sim->d_->qpos[sim->m_->jnt_qposadr[i]];
      } else {
        defSlider[0].pdata = &sim->qpos_[sim->jnt_qposadr_[i]];
      }
      if (!sim->jnt_names_[i].empty()) {
        mju::strcpy_arr(defSlider[0].name, sim->jnt_names_[i].c_str());
      } else {
        mju::sprintf_arr(defSlider[0].name, "joint %d", i);
      }

      // set range
      if (sim->jnt_range_[i].has_value())
        mju::sprintf_arr(defSlider[0].other,
                         "%.17g %.17g",
                         sim->jnt_range_[i]->first,
                         sim->jnt_range_[i]->second);
      else if (sim->jnt_type_[i] == mjJNT_SLIDE) {
        mju::strcpy_arr(defSlider[0].other, "-1 1");
      } else {
        mju::strcpy_arr(defSlider[0].other, "-3.1416 3.1416");
      }

      // add and count
      mjui_add(&sim->ui1, defSlider);
      itemcnt++;
    }
  }
}

// make control section of UI
void MakeControlSection(mj::Simulate* sim) {
  mjuiDef defControl[] = {
      {mjITEM_SECTION, "Control", mjPRESERVE, nullptr, "AC"},
      {mjITEM_BUTTON, "Clear all", 2},
      {mjITEM_END}
  };
  mjuiDef defSlider[] = {
      {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
      {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defControl);

  // add controls, exit if UI limit reached (Clear button already added)
  int itemcnt = 1;
  for (int i = 0; i < sim->actuator_ctrlrange_.size() && itemcnt < mjMAXUIITEM; i++) {
    // skip if actuator vis group is disabled
    int group = sim->actuator_group_[i];
    if (!sim->opt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP - 1, group))]) { continue; }
    // grey out if actuator group is disabled
    if (group >= 0 && group <= 30 && sim->m_->opt.disableactuator & (1 << group)) {
      defSlider[0].state = 0;
    } else {
      defSlider[0].state = 2;
    }

    // set data and name
    if (!sim->is_passive_) {
      defSlider[0].pdata = &sim->d_->ctrl[i];
    } else {
      defSlider[0].pdata = &sim->ctrl_[i];
    }
    if (!sim->actuator_names_[i].empty()) {
      mju::strcpy_arr(defSlider[0].name, sim->actuator_names_[i].c_str());
    } else {
      mju::sprintf_arr(defSlider[0].name, "control %d", i);
    }

    // set range
    if (sim->actuator_ctrlrange_[i].has_value())
      mju::sprintf_arr(defSlider[0].other,
                       "%.17g %.17g",
                       sim->actuator_ctrlrange_[i]->first,
                       sim->actuator_ctrlrange_[i]->second);
    else { mju::strcpy_arr(defSlider[0].other, "-1 1"); }

    // add and count
    mjui_add(&sim->ui1, defSlider);
    itemcnt++;
  }
}

// make equality section of UI
void MakeEqualitySection(mj::Simulate* sim) {
  mjuiDef defEquality[] = {
      {mjITEM_SECTION, "Equality", mjPRESERVE, nullptr, "AE"},
      {mjITEM_END}
  };
  mjuiDef defCheckBox[] = {
      {mjITEM_CHECKBYTE, "", 2, nullptr, ""},
      {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defEquality);

  // add equalities, exit if UI limit reached
  for (int i = 0; i < sim->m_->neq && i < mjMAXUIITEM; i++) {
    // set data
    defCheckBox[0].pdata = &sim->d_->eq_active[i];

    // set name
    if (!sim->equality_names_[i].empty()) {
      mju::strcpy_arr(defCheckBox[0].name, sim->equality_names_[i].c_str());
    } else {
      mju::sprintf_arr(defCheckBox[0].name, "equality %d", i);
    }

    mjui_add(&sim->ui1, defCheckBox);
  }
}

// make model-dependent UI sections
void MakeUiSections(mj::Simulate* sim, const mjModel* m, const mjData* d) {
  // clear model-dependent sections of UI
  sim->ui0.nsect = SECT_PHYSICS;
  sim->ui1.nsect = 0;

  // make
  MakePhysicsSection(sim);
  MakeRenderingSection(sim, m);
  MakeVisualizationSection(sim, m);
  MakeGroupSection(sim);
  MakeLoggingSection(sim);
  MakeJointSection(sim);
  MakeControlSection(sim);
  MakeEqualitySection(sim);
}

//---------------------------------- utility functions ---------------------------------------------

// align and scale view
void AlignAndScaleView(mj::Simulate* sim, const mjModel* m) {
  // if the id is valid, use the initial fixed camera
  if (m->vis.global.cameraid >= 0 && m->vis.global.cameraid < m->ncam) {
    sim->cam.fixedcamid = m->vis.global.cameraid;
    sim->cam.type       = mjCAMERA_FIXED;
  }

  // otherwise use default free camera
  else {
    mjv_defaultFreeCamera(m, &sim->cam);
  }
}


// copy state to clipboard as key
void CopyKey(mj::Simulate* sim, const mjModel* m, const mjData* d, bool fp) {
  char        clipboard[5000] = "<key\n";
  char        buf[200];
  const char  p_regular[] = "%g";
  const char  p_full[]    = "%-22.16g";
  const char* format      = fp ? p_full : p_regular;

  // time
  mju::strcat_arr(clipboard, "  time=\"");
  mju::sprintf_arr(buf, format, d->time);
  mju::strcat_arr(clipboard, buf);

  // qpos
  mju::strcat_arr(clipboard, "\"\n  qpos=\"");
  for (int i = 0; i < m->nq; i++) {
    mju::sprintf_arr(buf, format, d->qpos[i]);
    if (i < m->nq - 1) mju::strcat_arr(buf, " ");
    mju::strcat_arr(clipboard, buf);
  }

  // qvel
  mju::strcat_arr(clipboard, "\"\n  qvel=\"");
  for (int i = 0; i < m->nv; i++) {
    mju::sprintf_arr(buf, format, d->qvel[i]);
    if (i < m->nv - 1) mju::strcat_arr(buf, " ");
    mju::strcat_arr(clipboard, buf);
  }

  // act
  if (m->na > 0) {
    mju::strcat_arr(clipboard, "\"\n  act=\"");
    for (int i = 0; i < m->na; i++) {
      mju::sprintf_arr(buf, format, d->act[i]);
      if (i < m->na - 1) mju::strcat_arr(buf, " ");
      mju::strcat_arr(clipboard, buf);
    }
  }

  // ctrl
  if (m->nu > 0) {
    mju::strcat_arr(clipboard, "\"\n  ctrl=\"");
    for (int i = 0; i < m->nu; i++) {
      mju::sprintf_arr(buf, format, d->ctrl[i]);
      if (i < m->nu - 1) mju::strcat_arr(buf, " ");
      mju::strcat_arr(clipboard, buf);
    }
  }

  if (m->nmocap > 0) {
    // mocap_pos
    mju::strcat_arr(clipboard, "\"\n  mpos=\"");
    for (int i = 0; i < 3 * m->nmocap; i++) {
      mju::sprintf_arr(buf, format, d->mocap_pos[i]);
      if (i < 3 * m->nmocap - 1) mju::strcat_arr(buf, " ");
      mju::strcat_arr(clipboard, buf);
    }

    // mocap_quat
    mju::strcat_arr(clipboard, "\"\n  mquat=\"");
    for (int i = 0; i < 4 * m->nmocap; i++) {
      mju::sprintf_arr(buf, format, d->mocap_quat[i]);
      if (i < 4 * m->nmocap - 1) mju::strcat_arr(buf, " ");
      mju::strcat_arr(clipboard, buf);
    }
  }

  // plastic
  if (m->nplastic > 0) {
    mju::strcat_arr(clipboard, "\"\n  plastic=\"");
    for (int i = 0; i < m->nplastic; i++) {
      mju::sprintf_arr(buf, format, d->plastic[i]);
      if (i < m->nplastic - 1) mju::strcat_arr(buf, " ");
      mju::strcat_arr(clipboard, buf);
    }
  }

  mju::strcat_arr(clipboard, "\"\n/>");

  // copy to clipboard
  sim->platform_ui->SetClipboardString(clipboard);
}

// millisecond timer, for MuJoCo built-in profiler
mjtNum Timer() {
  static auto start   = mj::Simulate::Clock::now();
  auto        elapsed = Milliseconds(mj::Simulate::Clock::now() - start);
  return elapsed.count();
}

// copy current camera to clipboard as MJCF specification
void CopyCamera(mj::Simulate* sim) {
  mjvGLCamera* camera = sim->scn.camera;

  char   clipboard[500];
  mjtNum cam_right[3];
  mjtNum cam_forward[3];
  mjtNum cam_up[3];

  // get camera spec from the GLCamera
  mju_f2n(cam_forward, camera[0].forward, 3);
  mju_f2n(cam_up, camera[0].up, 3);
  mju_cross(cam_right, cam_forward, cam_up);

  // make MJCF camera spec
  mju::sprintf_arr(clipboard,
                   "<camera pos=\"%.3f %.3f %.3f\" xyaxes=\"%.3f %.3f %.3f %.3f %.3f %.3f\"/>\n",
                   (camera[0].pos[0] + camera[1].pos[0]) / 2,
                   (camera[0].pos[1] + camera[1].pos[1]) / 2,
                   (camera[0].pos[2] + camera[1].pos[2]) / 2,
                   cam_right[0],
                   cam_right[1],
                   cam_right[2],
                   camera[0].up[0],
                   camera[0].up[1],
                   camera[0].up[2]);

  // copy spec into clipboard
  sim->platform_ui->SetClipboardString(clipboard);
}

// update UI 0 when MuJoCo structures change (except for joint sliders)
void UpdateSettings(mj::Simulate* sim, const mjModel* m) {
  // physics flags
  for (int i = 0; i < mjNDISABLE; i++) {
    int new_value = ((m->opt.disableflags & (1 << i)) != 0);
    if (sim->disable[i] != new_value) {
      sim->disable[i]                 = new_value;
      sim->pending_.ui_update_physics = true;
    }
  }
  for (int i = 0; i < mjNENABLE; i++) {
    int new_value = ((m->opt.enableflags & (1 << i)) != 0);
    if (sim->enable[i] != new_value) {
      sim->enable[i]                  = new_value;
      sim->pending_.ui_update_physics = true;
    }
  }
  for (int i = 0; i < mjNGROUP; i++) {
    int enabled = ((m->opt.disableactuator & (1 << i)) == 0);
    if (sim->enableactuator[i] != enabled) {
      sim->enableactuator[i]          = enabled;
      sim->pending_.ui_update_physics = true;
      sim->pending_.ui_remake_ctrl    = true;
    }
  }

  // camera
  int old_camera = sim->camera;
  if (sim->cam.type == mjCAMERA_FIXED) {
    sim->camera = 2 + sim->cam.fixedcamid;
  } else if (sim->cam.type == mjCAMERA_TRACKING) {
    sim->camera = 1;
  } else {
    sim->camera = 0;
  }
  if (old_camera != sim->camera) { sim->pending_.ui_update_rendering = true; }

  // logging flags
  mjLogConfig cfg             = mju_getLogConfig();
  bool        logging_changed = false;
  if (sim->log_console != cfg.logto_console) {
    sim->log_console = cfg.logto_console;
    logging_changed  = true;
  }
  if (sim->log_file != cfg.logto_file) {
    sim->log_file   = cfg.logto_file;
    logging_changed = true;
  }
  for (int i = 0; i < mjNTOPIC; i++) {
    int enabled = ((cfg.topics & (1 << i)) != 0);
    if (sim->log_topics[i] != enabled) {
      sim->log_topics[i] = enabled;
      logging_changed    = true;
    }
  }
  if (logging_changed) { sim->pending_.ui_update_logging = true; }
}

// Compute suitable font scale.
int ComputeFontScale(const mj::PlatformUIAdapter& platform_ui) {
  // compute framebuffer-to-window ratio
  auto [buf_width, buf_height] = platform_ui.GetFramebufferSize();
  auto [win_width, win_height] = platform_ui.GetWindowSize();

  double b2w = static_cast<double>(buf_width) / win_width;

  // compute PPI
  double PPI = b2w * platform_ui.GetDisplayPixelsPerInch();

  // estimate font scaling, guard against unrealistic PPI
  int fs;
  if (buf_width > win_width) {
    fs = mju_round(b2w * 100);
  } else if (PPI > 50 && PPI < 350) {
    fs = mju_round(PPI);
  } else {
    fs = 150;
  }
  fs = mju_round(fs * 0.02) * 50;
  fs = mjMIN(300, mjMAX(100, fs));

  return fs;
}


//---------------------------------- UI handlers ---------------------------------------------------

// determine enable/disable item state given category
int UiPredicate(int category, void* userdata) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(userdata);

  switch (category) {
    case 2:  // require model
      return sim->m_ || sim->is_passive_;

    case 3:  // require model and nkey
      return (sim->m_ || sim->is_passive_) && sim->nkey_;

    case 4:  // require model and paused
      return sim->m_ && !sim->run;

    case 5:  // require model and fully managed mode
      return !sim->is_passive_ && sim->m_;

    default:
      return 1;
  }
}

// set window layout
void UiLayout(mjuiState* state) {
  mj::Simulate* sim  = static_cast<mj::Simulate*>(state->userdata);
  mjrRect*      rect = state->rect;

  // set number of rectangles
  state->nrect = 4;

  // rect 1: UI 0
  rect[1].left   = 0;
  rect[1].width  = sim->ui0_enable ? sim->ui0.width : 0;
  rect[1].bottom = 0;
  rect[1].height = rect[0].height;

  // rect 2: UI 1
  rect[2].width  = sim->ui1_enable ? sim->ui1.width : 0;
  rect[2].left   = mjMAX(0, rect[0].width - rect[2].width);
  rect[2].bottom = 0;
  rect[2].height = rect[0].height;

  // rect 3: 3D plot (everything else is an overlay)
  rect[3].left   = rect[1].width;
  rect[3].width  = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
  rect[3].bottom = 0;
  rect[3].height = rect[0].height;
}

// modify UI
void UiModify(mjUI* ui, mjuiState* state, mjrContext* con) {
  mjui_resize(ui, con);

  // remake aux buffer only if missing or different
  int id = ui->auxid;
  if (con->auxFBO[id] == 0 ||
      con->auxFBO_r[id] == 0 ||
      con->auxColor[id] == 0 ||
      con->auxColor_r[id] == 0 ||
      con->auxWidth[id] != ui->width ||
      con->auxHeight[id] != ui->maxheight ||
      con->auxSamples[id] != ui->spacing.samples) {
    mjr_addAux(id, ui->width, ui->maxheight, ui->spacing.samples, con);
  }

  UiLayout(state);
  mjui_update(-1, -1, ui, state, con);
}

// handle UI event
void UiEvent(mjuiState* state) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(state->userdata);

  // call UI 0 if event is directed to it
  if ((state->dragrect == sim->ui0.rectid) ||
      (state->dragrect == 0 && state->mouserect == sim->ui0.rectid) ||
      state->type == mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&sim->ui0, state, &sim->platform_ui->mjr_context());

    // file section
    if (it && it->sectionid == SECT_FILE) {
      switch (it->itemid) {
        case 0:  // Save xml
          sim->pending_.save_xml = GetSavePath("mjmodel.xml");
          break;

        case 1:  // Save mjb
          sim->pending_.save_mjb = GetSavePath("mjmodel.mjb");
          break;

        case 2:  // Print model
          sim->pending_.print_model = GetSavePath("MJMODEL.TXT");
          break;

        case 3:  // Print data
          sim->pending_.print_data = GetSavePath("MJDATA.TXT");
          break;

        case 4:  // Quit
          sim->exitrequest.store(1);
          break;

        case 5:  // Screenshot
          sim->screenshotrequest.store(true);
          break;
      }
    }

    // option section
    else if (it && it->sectionid == SECT_OPTION) {
      if (it->pdata == &sim->info) {
        // clear load error/warning when toggling info panel
        sim->load_error[0] = '\0';
      }
      if (it->pdata == &sim->spacing) {
        sim->ui0.spacing = mjui_themeSpacing(sim->spacing);
        sim->ui1.spacing = mjui_themeSpacing(sim->spacing);
      } else if (it->pdata == &sim->color) {
        sim->ui0.color = mjui_themeColor(sim->color);
        sim->ui1.color = mjui_themeColor(sim->color);
      } else if (it->pdata == &sim->font) {
        mjr_changeFont(50 * (sim->font + 1), &sim->platform_ui->mjr_context());
      } else if (it->pdata == &sim->fullscreen) {
        sim->platform_ui->ToggleFullscreen();
      } else if (it->pdata == &sim->vsync) {
        sim->platform_ui->SetVSync(sim->vsync);
      }

      // modify UI
      UiModify(&sim->ui0, state, &sim->platform_ui->mjr_context());
      UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
    }

    // simulation section
    else if (it && it->sectionid == SECT_SIMULATION) {
      if (it->itemid == 0) {
        // clear load error/warning when toggling play/pause
        sim->load_error[0] = '\0';
      }
      switch (it->itemid) {
        case 1:  // Threadpool
          sim->pending_.update_threadpool = true;
          break;
        case 2:  // Reset
          sim->pending_.reset = true;
          break;

        case 3:  // Reload
          sim->uiloadrequest.fetch_add(1);
          break;

        case 4:  // Align
          sim->pending_.align = true;
          break;

        case 5:  // Copy key
          sim->pending_.copy_key                = true;
          sim->pending_.copy_key_full_precision = sim->platform_ui->IsShiftKeyPressed();
          break;

        case 6:  // Adjust key
        case 7:  // Load key
          sim->pending_.load_key = true;
          break;

        case 8:  // Save key
          sim->pending_.save_key = true;
          break;

        case 12:  // History scrubber
          sim->run = 0;

          sim->pending_.load_from_history = true;
          mjui0_update_section(sim, SECT_SIMULATION);
          break;
      }
    }

    // physics section
    else if (it && it->sectionid == SECT_PHYSICS && sim->m_) {
      mjOption* opt = sim->is_passive_ ? &sim->m_passive_->opt : &sim->m_->opt;

      // update disable flags in mjOption
      opt->disableflags = 0;
      for (int i = 0; i < mjNDISABLE; i++) {
        if (sim->disable[i]) { opt->disableflags |= (1 << i); }
      }

      // update enable flags in mjOption
      opt->enableflags = 0;
      for (int i = 0; i < mjNENABLE; i++) {
        if (sim->enable[i]) { opt->enableflags |= (1 << i); }
      }

      // update disableactuator bitflag in mjOption
      bool group_changed = false;
      for (int i = 0; i < mjNGROUP; i++) {
        if ((!sim->enableactuator[i]) != (opt->disableactuator & (1 << i))) {
          group_changed = true;
          if (!sim->enableactuator[i]) {
            // disable actuator group i
            opt->disableactuator |= (1 << i);
          } else {
            // enable actuator group i
            opt->disableactuator &= ~(1 << i);
          }
        }
      }

      // remake control section if actuator disable group changed
      if (group_changed) { sim->pending_.ui_remake_ctrl = true; }
    }

    // rendering section
    else if (it && it->sectionid == SECT_RENDERING) {
      // only update the camera when the camera itself changed
      if (it->pdata == &sim->camera) {
        if (sim->camera == 0) {
          sim->cam.type = mjCAMERA_FREE;
        } else if (sim->camera == 1) {
          if (sim->pert.select > 0) {
            sim->cam.type        = mjCAMERA_TRACKING;
            sim->cam.trackbodyid = sim->pert.select;
            sim->cam.fixedcamid  = -1;
          } else {
            sim->cam.type = mjCAMERA_FREE;
            sim->camera   = 0;
            mjui0_update_section(sim, SECT_RENDERING);
          }
        } else {
          sim->cam.type       = mjCAMERA_FIXED;
          sim->cam.fixedcamid = sim->camera - 2;
        }
      }

      // copy camera spec to clipboard (as MJCF element)
      if (it->itemid == 3) { CopyCamera(sim); }
    }

    // visualization section
    else if (it && it->sectionid == SECT_VISUALIZATION) {
      if (!mju::strcmp_arr(it->name, "Align")) {
        sim->pending_.align = true;
      } else if (!mju::strcmp_arr(it->name, "Orthographic")) {
        mjVisual* vis = sim->is_passive_ ? (sim->m_passive_ ? &sim->m_passive_->vis : nullptr)
                                         : (sim->m_ ? &sim->m_->vis : nullptr);
        if (vis) { vis->global.orthographic = sim->cam.orthographic; }
      }
    }

    // group section
    else if (it && it->sectionid == SECT_GROUP) {
      // remake joint section if joint group changed
      if (it->name[0] == 'J' && it->name[1] == 'o') {
        sim->ui1.nsect = SECT_JOINT;
        MakeJointSection(sim);
        sim->ui1.nsect = NSECT1;
        UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
      }

      // remake control section if actuator group changed
      if (it->name[0] == 'A' && it->name[1] == 'c') { sim->pending_.ui_remake_ctrl = true; }
    }

    // logging section
    else if (it && it->sectionid == SECT_LOGGING) {
      mjLogConfig cfg   = mju_getLogConfig();
      cfg.logto_console = sim->log_console;
      cfg.logto_file    = sim->log_file;
      cfg.topics        = 0;
      for (int i = 0; i < mjNTOPIC; i++) {
        if (sim->log_topics[i]) { cfg.topics |= (1 << i); }
      }
      mju_setLogConfig(cfg);
    }

    // stop if UI processed event
    if (it != nullptr || (state->type == mjEVENT_KEY && state->key == 0)) { return; }
  }

  // call UI 1 if event is directed to it
  if ((state->dragrect == sim->ui1.rectid) ||
      (state->dragrect == 0 && state->mouserect == sim->ui1.rectid) ||
      state->type == mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&sim->ui1, state, &sim->platform_ui->mjr_context());

    // control section
    if (it && it->sectionid == SECT_CONTROL) {
      // clear controls
      if (it->itemid == 0) { sim->pending_.zero_ctrl = true; }
    }

    // stop if UI processed event
    if (it != nullptr || (state->type == mjEVENT_KEY && state->key == 0)) { return; }
  }

  // shortcut not handled by UI
  if (state->type == mjEVENT_KEY && state->key != 0) {
    switch (state->key) {
      case ' ':  // Mode
        if (!sim->is_passive_ && sim->m_) {
          sim->run           = 1 - sim->run;
          sim->pert.active   = 0;
          sim->load_error[0] = '\0';

          if (sim->run) sim->scrub_index = 0;  // reset scrubber

          mjui0_update_section(sim, -1);
        }
        break;

      case mjKEY_RIGHT:  // step forward
        if (!sim->is_passive_ && sim->m_ && !sim->run) {
          // currently in scrubber: increment scrub, load state, update slider UI
          if (sim->scrub_index < 0) {
            sim->scrub_index++;
            sim->pending_.load_from_history = true;
            mjui0_update_section(sim, SECT_SIMULATION);
          }

          // not in scrubber: step, add to history buffer
          else {
            mj_step(sim->m_, sim->d_);
            sim->AddToHistory();
          }

          UpdateProfiler(sim, sim->m_, sim->d_);
          UpdateSensor(sim, sim->m_, sim->d_);
          UpdateSettings(sim, sim->m_);
        }
        break;

      case mjKEY_LEFT:  // step backward
        if (!sim->is_passive_ && sim->m_) {
          sim->run = 0;

          // decrement scrub, load state
          sim->scrub_index                = mjMAX(sim->scrub_index - 1, 1 - sim->nhistory_);
          sim->pending_.load_from_history = true;

          // update slider UI, profiler, sensor
          mjui0_update_section(sim, SECT_SIMULATION);
          UpdateProfiler(sim, sim->m_, sim->d_);
          UpdateSensor(sim, sim->m_, sim->d_);
        }
        break;

      case mjKEY_PAGE_UP:  // select parent body
        if ((sim->m_ || sim->is_passive_) && sim->pert.select > 0) {
          sim->pert.select     = sim->body_parentid_[sim->pert.select];
          sim->pert.flexselect = -1;
          sim->pert.skinselect = -1;

          // stop perturbation if world reached
          if (sim->pert.select <= 0) { sim->pert.active = 0; }
        }

        break;

      case ']':  // cycle up fixed cameras
        if ((sim->m_ || !sim->is_passive_) && sim->ncam_) {
          sim->cam.type = mjCAMERA_FIXED;
          // camera = {0 or 1} are reserved for the free and tracking cameras
          if (sim->camera < 2 || sim->camera == 2 + sim->ncam_ - 1) {
            sim->camera = 2;
          } else {
            sim->camera += 1;
          }
          sim->cam.fixedcamid = sim->camera - 2;
          mjui0_update_section(sim, SECT_RENDERING);
        }
        break;

      case '[':  // cycle down fixed cameras
        if ((sim->m_ || sim->is_passive_) && sim->ncam_) {
          sim->cam.type = mjCAMERA_FIXED;
          // camera = {0 or 1} are reserved for the free and tracking cameras
          if (sim->camera <= 2) {
            sim->camera = 2 + sim->ncam_ - 1;
          } else {
            sim->camera -= 1;
          }
          sim->cam.fixedcamid = sim->camera - 2;
          mjui0_update_section(sim, SECT_RENDERING);
        }
        break;

      case mjKEY_F6:  // cycle frame visualisation
        if (sim->m_ || sim->is_passive_) {
          sim->opt.frame = (sim->opt.frame + 1) % mjNFRAME;
          mjui0_update_section(sim, SECT_RENDERING);
        }
        break;

      case mjKEY_F7:  // cycle label visualisation
        if (sim->m_ || sim->is_passive_) {
          sim->opt.label = (sim->opt.label + 1) % mjNLABEL;
          mjui0_update_section(sim, SECT_RENDERING);
        }
        break;

      case mjKEY_ESCAPE:  // free camera
        sim->cam.type = mjCAMERA_FREE;
        sim->camera   = 0;
        mjui0_update_section(sim, SECT_RENDERING);
        break;

      case '-':  // slow down
        if (!sim->is_passive_) {
          int numclicks = sizeof(sim->percentRealTime) / sizeof(sim->percentRealTime[0]);
          if (sim->real_time_index < numclicks - 1 && !state->shift) {
            sim->real_time_index++;
            sim->speed_changed = true;
          }
        }
        break;

      case '=':  // speed up
        if (!sim->is_passive_ && sim->real_time_index > 0 && !state->shift) {
          sim->real_time_index--;
          sim->speed_changed = true;
        }
        break;

      case mjKEY_TAB:  // toggle left/right UI
        if (!state->shift) {
          // toggle left UI
          sim->ui0_enable = !sim->ui0_enable;
          UiModify(&sim->ui0, state, &sim->platform_ui->mjr_context());
        } else {
          // toggle right UI
          sim->ui1_enable = !sim->ui1_enable;
          UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
        }
        break;
    }

    return;
  }

  // local pointers used below
  mjModel* model = sim->is_passive_ ? sim->m_passive_ : sim->m_;
  mjData*  data  = sim->is_passive_ ? sim->d_passive_ : sim->d_;

  // 3D scroll
  if (state->type == mjEVENT_SCROLL && state->mouserect == 3 && model) {
    // emulate vertical mouse motion = 2% of window height
    mjv_moveCamera(model, mjMOUSE_ZOOM, 0, -zoom_increment * state->sy, &sim->cam);
    return;
  }

  // 3D press
  if (state->type == mjEVENT_PRESS && state->mouserect == 3) {
    // set perturbation
    int newperturb = 0;
    if (state->control && sim->pert.select > 0 && (sim->m_ || sim->is_passive_)) {
      // right: translate;  left: rotate
      if (state->right) {
        newperturb = mjPERT_TRANSLATE;
      } else if (state->left) {
        newperturb = mjPERT_ROTATE;
      }
      if (newperturb && !sim->pert.active) { sim->pending_.newperturb = newperturb; }
    }

    // handle double-click
    if (state->doubleclick && (sim->m_ || sim->is_passive_)) {
      sim->pending_.select = true;
      std::memcpy(&sim->pending_.select_state, state, sizeof(sim->pending_.select_state));

      // stop perturbation on select
      sim->pert.active         = 0;
      sim->pending_.newperturb = 0;
    }

    return;
  }

  // 3D release
  if (state->type == mjEVENT_RELEASE && state->dragrect == 3 && (sim->m_ || sim->is_passive_)) {
    // stop perturbation
    sim->pert.active         = 0;
    sim->pending_.newperturb = 0;
    return;
  }

  // 3D move
  if (state->type == mjEVENT_MOVE && state->dragrect == 3 && (sim->m_ || sim->is_passive_)) {
    // determine action based on mouse button
    mjtMouse action;
    if (state->right) {
      action = state->shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (state->left) {
      action = state->shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
      action = mjMOUSE_ZOOM;
    }

    // move perturb or camera
    mjrRect r = state->rect[3];
    if (sim->pert.active) {
      mjv_movePerturb(model,
                      data,
                      action,
                      state->dx / r.height,
                      -state->dy / r.height,
                      &sim->scn,
                      &sim->pert);
    } else {
      mjv_moveCamera(model, action, state->dx / r.height, -state->dy / r.height, &sim->cam);
    }
    return;
  }

  // Dropped files
  if (state->type == mjEVENT_FILESDROP && state->dropcount > 0 && !sim->is_passive_) {
    while (sim->droploadrequest.load()) {}
    mju::strcpy_arr(sim->dropfilename, state->droppaths[0]);
    sim->droploadrequest.store(true);
    return;
  }

  // Redraw
  if (state->type == mjEVENT_REDRAW) {
    sim->Render();
    return;
  }
}
}  // namespace

namespace mujoco {
namespace mju = ::mujoco::sample_util;

Simulate::Simulate(std::unique_ptr<PlatformUIAdapter> platform_ui,
                   mjvCamera*                         cam,
                   mjvOption*                         opt,
                   mjvPerturb*                        pert,
                   bool                               is_passive)
    : is_passive_(is_passive),
      cam(*cam),
      opt(*opt),
      pert(*pert),
      platform_ui(std::move(platform_ui)),
      uistate(this->platform_ui->state()) {
  mjv_defaultScene(&scn);
}


//------------------------- Synchronize render and physics threads ---------------------------------

// operations which require holding the mutex, prevents racing with physics thread
void Simulate::Sync(bool state_only) {
  MutexLock lock(this->mtx);

  if (!m_) { return; }

  if (this->exitrequest.load()) { return; }

  bool update_profiler = this->profiler;
  bool update_sensor   = this->sensor;

  for (int i = 0; i < m_->njnt; ++i) {
    std::optional<std::pair<mjtNum, mjtNum>> range;
    if (m_->jnt_limited[i]) { range.emplace(m_->jnt_range[2 * i], m_->jnt_range[2 * i + 1]); }
    if (jnt_range_[i] != range) {
      pending_.ui_update_joint = true;
      jnt_range_[i].swap(range);
    }
  }

  for (int i = 0; i < m_->nu; ++i) {
    std::optional<std::pair<mjtNum, mjtNum>> range;
    // a defined ctrlrange sets the slider range, even when ctrl is not clamped
    if (m_->actuator_ctrlrange[2 * i] < m_->actuator_ctrlrange[2 * i + 1]) {
      range.emplace(m_->actuator_ctrlrange[2 * i], m_->actuator_ctrlrange[2 * i + 1]);
    }
    if (actuator_ctrlrange_[i] != range) {
      pending_.ui_remake_ctrl = true;
      actuator_ctrlrange_[i].swap(range);
    }
  }

  for (int i = 0; i < m_->nq; ++i) {
    if (qpos_[i] != qpos_prev_[i]) {
      d_->qpos[i] = qpos_[i];
    } else {
      qpos_[i] = d_->qpos[i];
    }
    if (qpos_prev_[i] != qpos_[i]) {
      pending_.ui_update_joint = true;
      qpos_prev_[i]            = qpos_[i];
    }
  }

  for (int i = 0; i < m_->nu; ++i) {
    if (ctrl_[i] != ctrl_prev_[i]) {
      d_->ctrl[i] = ctrl_[i];
    } else {
      ctrl_[i] = d_->ctrl[i];
    }
    if (ctrl_prev_[i] != ctrl_[i]) {
      pending_.ui_update_ctrl = true;
      ctrl_prev_[i]           = ctrl_[i];
    }
  }

  for (int i = 0; i < m_->neq; ++i) {
    if (eq_active_[i] != eq_active_prev_[i]) {
      d_->eq_active[i] = eq_active_[i];
    } else {
      eq_active_[i] = d_->eq_active[i];
    }
    if (eq_active_prev_[i] != eq_active_[i]) {
      pending_.ui_update_equality = true;
      eq_active_prev_[i]          = eq_active_[i];
    }
  }

  // in passive mode, synchronize user's mjModel with changes made via the UI
  if (is_passive_) {
    // synchronize mjModel.opt
    if (std::memcmp(&m_passive_->opt, &mjopt_prev_, sizeof(mjOption))) {
      pending_.ui_update_physics = true;

      m_->opt = m_passive_->opt;
    }

    // synchronize mjModel.vis
    if (std::memcmp(&m_passive_->vis, &mjvis_prev_, sizeof(mjVisual))) {
      pending_.ui_update_visualization = true;

      m_->vis = m_passive_->vis;
    }

    // synchronize mjModel.stat
    if (std::memcmp(&m_passive_->stat, &mjstat_prev_, sizeof(mjStatistic))) {
      pending_.ui_update_visualization = true;

      m_->stat = m_passive_->stat;
    }
  }

  if (pending_.save_xml) {
    char err[200];
    if (!pending_.save_xml->empty() && !mj_saveLastXML(pending_.save_xml->c_str(), m_, err, 200)) {
      std::printf("Save XML error: %s", err);
    }
    pending_.save_xml = std::nullopt;
  }

  if (pending_.save_mjb) {
    if (!pending_.save_mjb->empty()) { mj_saveModel(m_, pending_.save_mjb->c_str(), nullptr, 0); }
    pending_.save_mjb = std::nullopt;
  }

  if (pending_.print_model) {
    if (!pending_.print_model->empty()) { mj_printModel(m_, pending_.print_model->c_str()); }
    pending_.print_model = std::nullopt;
  }

  if (pending_.print_data) {
    if (!pending_.print_data->empty()) { mj_printData(m_, d_, pending_.print_data->c_str()); }
    pending_.print_data = std::nullopt;
  }

  if (pending_.update_threadpool) {
    mju_threadpool(d_, nthread);
    pending_.update_threadpool = false;
  }

  if (pending_.reset) {
    mj_resetData(m_, d_);
    memset(timer_prev_, 0, sizeof(timer_prev_));
    mj_forward(m_, d_);
    load_error[0]   = '\0';
    update_profiler = true;
    update_sensor   = true;
    scrub_index     = 0;
    pending_.reset  = false;

    pending_.ui_update_simulation = true;
  }

  if (pending_.align) {
    AlignAndScaleView(this, m_);
    pending_.align                   = false;
    pending_.ui_update_visualization = true;
  }

  if (pending_.copy_key) {
    CopyKey(this, m_, d_, pending_.copy_key_full_precision);
    pending_.copy_key                = false;
    pending_.copy_key_full_precision = false;
  }

  if (pending_.load_from_history) {
    LoadScrubState(this);
    update_profiler            = true;
    update_sensor              = true;
    pending_.load_from_history = false;
  }

  if (pending_.load_key) {
    mj_resetDataKeyframe(m_, d_, this->key);
    mj_forward(m_, d_);
    update_profiler   = true;
    update_sensor     = true;
    pending_.load_key = false;
  }

  if (pending_.save_key) {
    mj_setKeyframe(m_, d_, this->key);
    pending_.save_key = false;
  }

  if (pending_.zero_ctrl) {
    mj_resetCtrl(m_, d_);
    pending_.zero_ctrl = false;
  }

  // perturbation onset: reset reference
  if (pending_.newperturb) {
    mjv_initPerturb(m_, d_, &this->scn, &this->pert);
    this->pert.active   = pending_.newperturb;
    pending_.newperturb = 0;
  }

  if (pending_.select) {
    // determine selection mode
    int selmode;
    if (pending_.select_state.button == mjBUTTON_LEFT) {
      selmode = 1;
    } else if (pending_.select_state.control) {
      selmode = 3;
    } else {
      selmode = 2;
    }

    // find geom and 3D click point, get corresponding body
    mjrRect r = pending_.select_state.rect[3];
    mjtNum  selpnt[3];
    int     selgeom, selflex, selskin;
    int     selbody = mjv_select(m_,
                                 d_,
                                 &this->opt,
                                 static_cast<mjtNum>(r.width) / r.height,
                                 (pending_.select_state.x - r.left) / r.width,
                                 (pending_.select_state.y - r.bottom) / r.height,
                                 &this->scn,
                                 selpnt,
                                 &selgeom,
                                 &selflex,
                                 &selskin);

    // set lookat point, start tracking is requested
    if (selmode == 2 || selmode == 3) {
      // copy selpnt if anything clicked
      if (selbody >= 0) { mju_copy3(this->cam.lookat, selpnt); }

      // switch to tracking camera if dynamic body clicked
      if (selmode == 3 && selbody > 0) {
        // mujoco camera
        this->cam.type        = mjCAMERA_TRACKING;
        this->cam.trackbodyid = selbody;
        this->cam.fixedcamid  = -1;

        // UI camera
        this->camera                 = 1;
        pending_.ui_update_rendering = true;
      }
    }

    // set body selection
    else {
      if (selbody >= 0) {
        // record selection
        this->pert.select     = selbody;
        this->pert.flexselect = selflex;
        this->pert.skinselect = selskin;

        // compute localpos
        mjtNum tmp[3];
        mju_sub3(tmp, selpnt, d_->xpos + 3 * this->pert.select);
        mju_mulMatTVec(this->pert.localpos, d_->xmat + 9 * this->pert.select, tmp, 3, 3);
      } else {
        this->pert.select     = 0;
        this->pert.flexselect = -1;
        this->pert.skinselect = -1;
      }
    }
    pending_.select = false;
  }

  // update scene or sync data from user in passive mode
  if (!is_passive_) {
    mjv_updateScene(m_, d_, &this->opt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);
  } else {
    if (state_only) {
      int     state_size = mj_stateSize(m_, mjSTATE_INTEGRATION);
      mjtNum* state      = new mjtNum[state_size];
      mj_getState(m_, d_, state, mjSTATE_INTEGRATION);
      mj_setState(m_passive_, d_passive_, state, mjSTATE_INTEGRATION);
      mj_forward(m_passive_, d_passive_);
      delete[] state;
    } else {
      mjv_copyModel(m_passive_, m_);
      mjv_copyData(d_passive_, m_passive_, d_);
    }

    // append geoms from user_scn to scratch space
    if (user_scn) {
      user_scn_geoms_.clear();
      user_scn_geoms_.reserve(user_scn->ngeom);
      for (int i = 0; i < user_scn->ngeom; ++i) { user_scn_geoms_.push_back(user_scn->geoms[i]); }
    }

    // pick up rendering flags changed via user_scn
    if (user_scn) {
      for (int i = 0; i < mjNRNDFLAG; ++i) {
        if (user_scn->flags[i] != user_scn_flags_prev_[i]) {
          scn.flags[i]                 = user_scn->flags[i];
          pending_.ui_update_rendering = true;
        }
      }
      Copy(user_scn->flags, scn.flags);
      Copy(user_scn_flags_prev_, user_scn->flags);
    }

    mjopt_prev_  = m_passive_->opt;
    mjvis_prev_  = m_passive_->vis;
    mjstat_prev_ = m_passive_->stat;
  }

  // update settings
  UpdateSettings(this, m_);

  // update watch
  if (this->ui0_enable && this->ui0.sect[SECT_WATCH].state) { UpdateWatch(this, m_, d_); }

  // update info text
  if (this->info) { UpdateInfoText(this, m_, d_, this->info_title, this->info_content); }
  if (update_profiler) { UpdateProfiler(this, m_, d_); }
  if (update_sensor) {
    UpdateSensor(this, m_, d_);

    // check if image sensor selection changed in UI dropdown
    int new_selected = this->image_sensor_ui_selection - 1;  // 0=None(-1), 1+=sensor
    if (new_selected != this->selected_image_sensor) {
      this->selected_image_sensor = new_selected;
      InitializeSensorImage(this, m_);
    }
    UpdateSensorImage(this, m_, d_);
  }

  if (this->run || this->is_passive_) {
    // clear old perturbations, apply new
    mju_zero(d_->xfrc_applied, 6 * m_->nbody);
    mjv_applyPerturbPose(m_, d_, &this->pert, 0);  // mocap bodies only
    mjv_applyPerturbForce(m_, d_, &this->pert);
  } else {
    mjv_applyPerturbPose(m_, d_, &this->pert, 1);  // mocap and dynamic bodies
  }

  for (int i = 0; i < mjNTIMER; i++) { timer_prev_[i] = d_->timer[i]; }
}

//------------------------- Tell the render thread to load a file and wait -------------------------
void Simulate::LoadMessage(const char* displayed_filename) {
  mju::strcpy_arr(this->filename, displayed_filename);

  {
    MutexLock lock(mtx);
    this->loadrequest = 3;
  }
}

void Simulate::Load(mjModel* m, mjData* d, const char* displayed_filename) {
  this->mnew_ = m;
  this->dnew_ = d;
  mju::strcpy_arr(this->filename, displayed_filename);

  {
    MutexLock lock(mtx);
    this->loadrequest = 2;

    // Wait for the render thread to be done loading
    // so that we know the old model and data's memory can
    // be free'd by the other thread (sometimes python)
    cond_loadrequest.wait(lock, [this]() { return this->loadrequest == 0; });
  }
}

void Simulate::LoadMessageClear(void) {
  {
    MutexLock lock(mtx);
    this->loadrequest = 0;
  }
}


//------------------------------------- load mjb or xml model --------------------------------------
void Simulate::LoadOnRenderThread() {
  this->m_ = this->mnew_;
  this->d_ = this->dnew_;

  ncam_ = this->m_->ncam;
  nkey_ = this->m_->nkey;
  body_parentid_.resize(this->m_->nbody);
  std::memcpy(body_parentid_.data(),
              this->m_->body_parentid,
              sizeof(this->m_->body_parentid[0]) * this->m_->nbody);

  jnt_type_.resize(this->m_->njnt);
  std::memcpy(jnt_type_.data(), this->m_->jnt_type, sizeof(this->m_->jnt_type[0]) * this->m_->njnt);

  jnt_group_.resize(this->m_->njnt);
  std::memcpy(jnt_group_.data(),
              this->m_->jnt_group,
              sizeof(this->m_->jnt_group[0]) * this->m_->njnt);

  jnt_qposadr_.resize(this->m_->njnt);
  std::memcpy(jnt_qposadr_.data(),
              this->m_->jnt_qposadr,
              sizeof(this->m_->jnt_qposadr[0]) * this->m_->njnt);

  jnt_range_.clear();
  jnt_range_.reserve(this->m_->njnt);
  for (int i = 0; i < this->m_->njnt; ++i) {
    if (this->m_->jnt_limited[i]) {
      jnt_range_.push_back(
          std::make_pair(this->m_->jnt_range[2 * i], this->m_->jnt_range[2 * i + 1]));
    } else {
      jnt_range_.push_back(std::nullopt);
    }
  }

  jnt_names_.clear();
  jnt_names_.reserve(this->m_->njnt);
  for (int i = 0; i < this->m_->njnt; ++i) {
    const char* name = mj_id2name(this->m_, mjOBJ_JOINT, i);
    jnt_names_.emplace_back(name ? name : "");
  }

  actuator_ctrlrange_.clear();
  actuator_ctrlrange_.reserve(this->m_->nu);
  for (int i = 0; i < this->m_->nu; ++i) {
    // a defined ctrlrange sets the slider range, even when ctrl is not clamped
    if (this->m_->actuator_ctrlrange[2 * i] < this->m_->actuator_ctrlrange[2 * i + 1]) {
      actuator_ctrlrange_.push_back(std::make_pair(this->m_->actuator_ctrlrange[2 * i + 0],
                                                   this->m_->actuator_ctrlrange[2 * i + 1]));
    } else {
      actuator_ctrlrange_.push_back(std::nullopt);
    }
  }

  // per-control group and name; multi-input actuators suffix the input name
  actuator_group_.resize(this->m_->nu);
  actuator_names_.clear();
  actuator_names_.reserve(this->m_->nu);
  for (int i = 0; i < this->m_->nactuator; ++i) {
    const char* actname = mj_id2name(this->m_, mjOBJ_ACTUATOR, i);
    int         ctrlnum = this->m_->actuator_ctrlnum[i];
    for (int k = 0; k < ctrlnum; ++k) {
      actuator_group_[this->m_->actuator_ctrladr[i] + k] = this->m_->actuator_group[i];
      std::string name                                   = actname ? actname : "";
      if (ctrlnum > 1 && actname) {
        const char* input_name = mj_actuatorInputName(this->m_, i, k);

        name += '/';
        name += input_name ? input_name : std::to_string(k);
      }
      actuator_names_.emplace_back(std::move(name));
    }
  }

  equality_names_.clear();
  equality_names_.reserve(this->m_->neq);
  for (int i = 0; i < this->m_->neq; ++i) {
    const char* name = mj_id2name(this->m_, mjOBJ_EQUALITY, i);
    equality_names_.emplace_back(name ? name : "");
  }

  qpos_.resize(this->m_->nq);
  std::memcpy(qpos_.data(), this->d_->qpos, sizeof(this->d_->qpos[0]) * this->m_->nq);
  qpos_prev_ = qpos_;

  ctrl_.resize(this->m_->nu);
  std::memcpy(ctrl_.data(), this->d_->ctrl, sizeof(this->d_->ctrl[0]) * this->m_->nu);
  ctrl_prev_ = ctrl_;

  eq_active_.resize(this->m_->neq);
  std::memcpy(eq_active_.data(),
              this->d_->eq_active,
              sizeof(this->d_->eq_active[0]) * this->m_->neq);
  eq_active_prev_ = eq_active_;

  // allocate history buffer: smaller of {2000 states, 100 MB}
  if (!this->is_passive_) {
    constexpr int kMaxHistoryBytes = 1e8;

    // get state size, size of history buffer
    state_size_        = mj_stateSize(this->m_, mjSTATE_INTEGRATION);
    int state_bytes    = state_size_ * sizeof(mjtNum);
    int history_length = mjMIN(INT_MAX / state_bytes, 2000);
    int history_bytes  = mjMIN(state_bytes * history_length, kMaxHistoryBytes);
    nhistory_          = history_bytes / state_bytes;

    // allocate history buffer, reset cursor and UI slider
    history_.clear();
    history_.resize(nhistory_ * state_size_);
    history_cursor_ = 0;
    scrub_index     = 0;

    // fill buffer with initial state
    mj_getState(this->m_, this->d_, history_.data(), mjSTATE_INTEGRATION);
    for (int i = 1; i < nhistory_; ++i) {
      mju_copy(&history_[i * state_size_], history_.data(), state_size_);
    }
  }

  // re-create scene
  mjv_makeScene(this->m_, &this->scn, kMaxGeom);

  this->platform_ui->RefreshMjrContext(this->m_, 50 * (this->font + 1));
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());

  if (!this->platform_ui->IsGPUAccelerated()) {
    this->scn.flags[mjRND_SHADOW]     = 0;
    this->scn.flags[mjRND_REFLECTION] = 0;
  }

  if (this->user_scn) {
    Copy(this->user_scn->flags, this->scn.flags);
    Copy(this->user_scn_flags_prev_, this->scn.flags);
  }

  // clear perturbation state
  this->pert.active     = 0;
  this->pert.select     = 0;
  this->pert.flexselect = -1;
  this->pert.skinselect = -1;

  memset(timer_prev_, 0, sizeof(timer_prev_));

  // align and scale view unless reloading the same file
  if (this->filename[0] && mju::strcmp_arr(this->filename, this->previous_filename)) {
    AlignAndScaleView(this, this->m_);
    mju::strcpy_arr(this->previous_filename, this->filename);
  }

  // update scene in managed mode, in passive mode copy data from user (update in RenderLoop)
  if (!is_passive_) {
    mjv_updateScene(this->m_, this->d_, &this->opt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);
  } else {
    mjopt_prev_ = m_->opt;
    opt_prev_   = opt;
    cam_prev_   = cam;


    // full copy on init
    m_passive_ = mj_copyModel(nullptr, m_);
    d_passive_ = mj_copyData(nullptr, m_passive_, d_);
  }

  // set window title to model name
  if (this->m_->names) {
    char title[200] = "MuJoCo : ";
    mju::strcat_arr(title, this->m_->names);
    platform_ui->SetWindowTitle(title);
  }

  // set keyframe range and divisions
  this->ui0.sect[SECT_SIMULATION].item[6].slider.range[0]  = 0;
  this->ui0.sect[SECT_SIMULATION].item[6].slider.range[1]  = mjMAX(0, this->m_->nkey - 1);
  this->ui0.sect[SECT_SIMULATION].item[6].slider.divisions = mjMAX(1, this->m_->nkey - 1);

  // set scrubber range and divisions
  this->ui0.sect[SECT_SIMULATION].item[12].slider.range[0]  = 1 - nhistory_;
  this->ui0.sect[SECT_SIMULATION].item[12].slider.divisions = nhistory_;

  // detect image sensors for visualization
  DetectImageSensors(this, this->m_);

  // rebuild UI sections
  MakeUiSections(this, this->m_, this->d_);

  // full ui update
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  UpdateSettings(this, this->m_);

  // clear request
  this->loadrequest = 0;
  cond_loadrequest.notify_all();

  // set real time index
  int   numclicks = sizeof(this->percentRealTime) / sizeof(this->percentRealTime[0]);
  float min_error = 1e6;
  float desired   = mju_log(100 * this->m_->vis.global.realtime);
  for (int click = 0; click < numclicks; click++) {
    float error = mju_abs(mju_log(this->percentRealTime[click]) - desired);
    if (error < min_error) {
      min_error             = error;
      this->real_time_index = click;
    }
  }

  this->mnew_ = nullptr;
  this->dnew_ = nullptr;
}


//------------------------------------------- rendering --------------------------------------------

// render the ui to the window
void Simulate::Render() {
  // update rendering context buffer size if required
  if (this->platform_ui->EnsureContextSize()) {
    UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // get 3D rectangle and reduced for profiler
  mjrRect rect      = this->uistate.rect[3];
  mjrRect smallrect = rect;
  if (this->profiler) { smallrect.width = rect.width - rect.width / 4; }

  // no model
  if (!this->is_passive_ && !this->m_) {
    // blank screen
    mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

    // label
    if (this->loadrequest) {
      mjr_overlay(mjFONT_BIG,
                  mjGRID_TOP,
                  smallrect,
                  "LOADING...",
                  nullptr,
                  &this->platform_ui->mjr_context());
    } else {
      char intro_message[Simulate::kMaxFilenameLength];
      mju::sprintf_arr(intro_message,
                       "MuJoCo version %s\nDrag-and-drop model file here",
                       mj_versionString());
      mjr_overlay(mjFONT_NORMAL,
                  mjGRID_TOPLEFT,
                  rect,
                  intro_message,
                  0,
                  &this->platform_ui->mjr_context());
    }

    // show last loading error
    if (this->load_error[0]) {
      mjr_overlay(mjFONT_NORMAL,
                  mjGRID_BOTTOMLEFT,
                  rect,
                  this->load_error,
                  0,
                  &this->platform_ui->mjr_context());
    }

    // render uis
    if (this->ui0_enable) {
      mjui_render(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    }
    if (this->ui1_enable) {
      mjui_render(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }

    // finalize
    this->platform_ui->SwapBuffers();

    return;
  }

  // update UI sections from last sync
  if (pending_.ui_update_simulation) {
    if (this->ui0_enable && this->ui0.sect[SECT_SIMULATION].state) {
      mjui0_update_section(this, SECT_SIMULATION);
    }
    pending_.ui_update_simulation = false;
  }

  if (this->ui0_enable && this->ui0.sect[SECT_WATCH].state) {
    mjui0_update_section(this, SECT_WATCH);
  }

  if (pending_.ui_update_physics) {
    if (this->ui0_enable && this->ui0.sect[SECT_PHYSICS].state) {
      mjui0_update_section(this, SECT_PHYSICS);
    }
    pending_.ui_update_physics = false;
  }

  if (pending_.ui_update_visualization) {
    if (this->ui0_enable && this->ui0.sect[SECT_VISUALIZATION].state) {
      mjui0_update_section(this, SECT_VISUALIZATION);
    }
    pending_.ui_update_visualization = false;
  }

  if (pending_.ui_update_logging) {
    if (this->ui0_enable && this->ui0.sect[SECT_LOGGING].state) {
      mjui0_update_section(this, SECT_LOGGING);
    }
    pending_.ui_update_logging = false;
  }

  if (is_passive_) {
    if (this->ui0_enable &&
        this->ui0.sect[SECT_RENDERING].state &&
        (cam_prev_.type != cam.type ||
         cam_prev_.fixedcamid != cam.fixedcamid ||
         cam_prev_.trackbodyid != cam.trackbodyid ||
         opt_prev_.label != opt.label ||
         opt_prev_.frame != opt.frame ||
         IsDifferent(opt_prev_.flags, opt.flags))) {
      pending_.ui_update_rendering = true;
    }

    if (this->ui0_enable &&
        this->ui0.sect[SECT_VISUALIZATION].state &&
        cam_prev_.orthographic != cam.orthographic) {
      mjui0_update_section(this, SECT_VISUALIZATION);
    }

    if (this->ui0_enable &&
        this->ui0.sect[SECT_RENDERING].state &&
        (IsDifferent(opt_prev_.geomgroup, opt.geomgroup) ||
         IsDifferent(opt_prev_.sitegroup, opt.sitegroup) ||
         IsDifferent(opt_prev_.jointgroup, opt.jointgroup) ||
         IsDifferent(opt_prev_.tendongroup, opt.tendongroup) ||
         IsDifferent(opt_prev_.actuatorgroup, opt.actuatorgroup) ||
         IsDifferent(opt_prev_.flexgroup, opt.flexgroup) ||
         IsDifferent(opt_prev_.skingroup, opt.skingroup))) {
      mjui0_update_section(this, SECT_GROUP);
    }

    opt_prev_ = opt;
    cam_prev_ = cam;
  }

  if (pending_.ui_update_rendering) {
    if (this->ui0_enable && this->ui0.sect[SECT_RENDERING].state) {
      mjui0_update_section(this, SECT_RENDERING);
    }
    pending_.ui_update_rendering = false;
  }

  if (pending_.ui_update_joint) {
    if (this->ui1_enable && this->ui1.sect[SECT_JOINT].state) {
      mjui_update(SECT_JOINT, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }
    pending_.ui_update_joint = false;
  }

  if (pending_.ui_remake_ctrl) {
    if (this->ui1_enable && this->ui1.sect[SECT_CONTROL].state) {
      this->ui1.nsect = SECT_CONTROL;
      MakeControlSection(this);
      this->ui1.nsect = NSECT1;
      UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }
    pending_.ui_remake_ctrl = false;
  }

  if (pending_.ui_update_ctrl) {
    if (this->ui1_enable && this->ui1.sect[SECT_CONTROL].state) {
      mjui_update(SECT_CONTROL, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }
    pending_.ui_update_ctrl = false;
  }

  if (pending_.ui_update_equality) {
    if (this->ui1_enable && this->ui1.sect[SECT_EQUALITY].state) {
      mjui_update(SECT_EQUALITY, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }
    pending_.ui_update_equality = false;
  }

  // render scene
  mjr_render(rect, &this->scn, &this->platform_ui->mjr_context());

  // show last loading error
  if (this->load_error[0]) {
    mjr_overlay(mjFONT_NORMAL,
                mjGRID_BOTTOMLEFT,
                rect,
                this->load_error,
                0,
                &this->platform_ui->mjr_context());
  }

  // show pause/loading label
  if (!this->run || this->loadrequest) {
    char label[30] = {'\0'};
    if (this->loadrequest) {
      std::snprintf(label, sizeof(label), "LOADING...");
    } else if (this->scrub_index == 0) {
      std::snprintf(label, sizeof(label), "PAUSE");
    } else {
      std::snprintf(label, sizeof(label), "PAUSE (%d)", this->scrub_index);
    }
    mjr_overlay(mjFONT_BIG,
                mjGRID_TOP,
                smallrect,
                label,
                nullptr,
                &this->platform_ui->mjr_context());
  }

  // get desired and actual percent-of-real-time
  float desiredRealtime = this->percentRealTime[this->real_time_index];
  float actualRealtime  = 100 / this->measured_slowdown;

  // if running, check for misalignment of more than 10%
  float realtime_offset = mju_abs(actualRealtime - desiredRealtime);
  bool  misaligned      = this->run && realtime_offset > 0.1 * desiredRealtime;

  // make realtime overlay label
  char rtlabel[30] = {'\0'};
  if (desiredRealtime != 100.0 || misaligned) {
    // print desired realtime
    int labelsize = std::snprintf(rtlabel, sizeof(rtlabel), "%g%%", desiredRealtime);

    // if misaligned, append to label
    if (misaligned) {
      std::snprintf(rtlabel + labelsize,
                    sizeof(rtlabel) - labelsize,
                    " (%-4.1f%%)",
                    actualRealtime);
    }
  }

  // show real-time overlay
  if (rtlabel[0]) {
    mjr_overlay(mjFONT_BIG,
                mjGRID_TOPLEFT,
                smallrect,
                rtlabel,
                nullptr,
                &this->platform_ui->mjr_context());
  }

  // show ui 0
  if (this->ui0_enable) {
    mjui_render(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // show ui 1
  if (this->ui1_enable) {
    mjui_render(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // show help
  if (this->help) {
    mjr_overlay(mjFONT_NORMAL,
                mjGRID_TOPLEFT,
                rect,
                help_title,
                help_content,
                &this->platform_ui->mjr_context());
  }

  // show info
  if (this->info) {
    mjr_overlay(mjFONT_NORMAL,
                mjGRID_BOTTOMLEFT,
                rect,
                this->info_title,
                this->info_content,
                &this->platform_ui->mjr_context());
  }

  // show profiler
  if (this->profiler) { ShowProfiler(this, rect); }

  // show sensor
  if (this->sensor) { ShowSensor(this, smallrect); }

  // take screenshot, save to file
  if (this->screenshotrequest.exchange(false)) {
    const unsigned int h = uistate.rect[0].height;
    const unsigned int w = uistate.rect[0].width;

    std::unique_ptr<unsigned char[]> rgb(new unsigned char[3 * w * h]);
    if (!rgb) { mju_error("could not allocate buffer for screenshot"); }
    mjr_readPixels(rgb.get(), nullptr, uistate.rect[0], &this->platform_ui->mjr_context());

    // flip up-down
    for (int r = 0; r < h / 2; ++r) {
      unsigned char* top_row    = &rgb[3 * w * r];
      unsigned char* bottom_row = &rgb[3 * w * (h - 1 - r)];
      std::swap_ranges(top_row, top_row + 3 * w, bottom_row);
    }

    // save as PNG
    // TODO(b/241577466): Parse the stem of the filename and use a .PNG extension.
    // Unfortunately, if we just yank ".xml"/".mjb" from the filename and append .PNG, the macOS
    // file dialog does not automatically open that location. Thus, we defer to a default
    // "screenshot.png" for now.
    const std::string path = GetSavePath("screenshot.png");
    if (!path.empty()) {
      if (lodepng::encode(path, rgb.get(), w, h, LCT_RGB)) {
        mju_error("could not save screenshot");
      } else {
        std::printf("saved screenshot: %s\n", path.c_str());
      }
    }
  }

  // user figures
  if (this->newfigurerequest.load() == 1) {
    this->user_figures_.clear();
    std::swap(this->user_figures_, this->user_figures_new_);
    int value = 1;
    this->newfigurerequest.compare_exchange_strong(value, 0);
  }
  for (auto& [viewport, figure] : this->user_figures_) { ShowFigure(this, viewport, &figure); }

  // overlay text
  if (this->newtextrequest.load() == 1) {
    this->user_texts_.clear();
    std::swap(this->user_texts_, this->user_texts_new_);
    int value = 1;
    this->newtextrequest.compare_exchange_strong(value, 0);
  }
  for (auto& [font, gridpos, text1, text2] : this->user_texts_) {
    ShowOverlayText(this, rect, font, gridpos, text1, text2);
  }

  // user images
  if (this->newimagerequest.load() == 1) {
    this->user_images_.clear();
    std::swap(this->user_images_, this->user_images_new_);
    int value = 1;
    this->newimagerequest.compare_exchange_strong(value, 0);
  }
  for (auto& [viewport, image] : this->user_images_) { ShowImage(this, viewport, image.get()); }

  // finalize
  this->platform_ui->SwapBuffers();
}


void Simulate::RenderLoop() {
  // Set timer callback (milliseconds)
  mjcb_time = Timer;

  // init abstract visualization
  mjv_defaultCamera(&this->cam);
  mjv_defaultOption(&this->opt);
  InitializeProfiler(this);
  InitializeSensor(this);

  // make empty scene
  if (!is_passive_) {
    mjv_defaultScene(&this->scn);
    mjv_makeScene(nullptr, &this->scn, kMaxGeom);
  }

  if (!this->platform_ui->IsGPUAccelerated()) {
    this->scn.flags[mjRND_SHADOW]     = 0;
    this->scn.flags[mjRND_REFLECTION] = 0;
  }

  // select default font
  int fontscale = ComputeFontScale(*this->platform_ui);
  this->font    = fontscale / 50 - 1;

  // make empty context
  this->platform_ui->RefreshMjrContext(nullptr, fontscale);

  // init state and uis
  std::memset(&this->uistate, 0, sizeof(mjuiState));
  std::memset(&this->ui0, 0, sizeof(mjUI));
  std::memset(&this->ui1, 0, sizeof(mjUI));

  auto [buf_width, buf_height] = this->platform_ui->GetFramebufferSize();
  this->uistate.nrect          = 1;
  this->uistate.rect[0].width  = buf_width;
  this->uistate.rect[0].height = buf_height;

  this->ui0.spacing   = mjui_themeSpacing(this->spacing);
  this->ui0.color     = mjui_themeColor(this->color);
  this->ui0.predicate = UiPredicate;
  this->ui0.rectid    = 1;
  this->ui0.auxid     = 0;

  this->ui1.spacing   = mjui_themeSpacing(this->spacing);
  this->ui1.color     = mjui_themeColor(this->color);
  this->ui1.predicate = UiPredicate;
  this->ui1.rectid    = 2;
  this->ui1.auxid     = 1;

  // set GUI adapter callbacks
  this->uistate.userdata = this;
  this->platform_ui->SetEventCallback(UiEvent);
  this->platform_ui->SetLayoutCallback(UiLayout);

  // populate uis with standard sections, open some sections initially
  this->ui0.userdata = this;
  this->ui1.userdata = this;
  mjui_add(&this->ui0, defFile);
  mjui_add(&this->ui0, this->def_option);
  mjui_add(&this->ui0, this->def_simulation);
  this->ui0.sect[0].state = 1;
  this->ui0.sect[1].state = 1;
  this->ui0.sect[2].state = 1;
  mjui_add(&this->ui0, this->def_watch);
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());

  // set VSync to initial value
  this->platform_ui->SetVSync(this->vsync);

  frames_          = 0;
  last_fps_update_ = mj::Simulate::Clock::now();

  // run event loop
  while (!this->platform_ui->ShouldCloseWindow() && !this->exitrequest.load()) {
    {
      const MutexLock lock(this->mtx);

      // load model (not on first pass, to show "loading" label)
      if (this->loadrequest == 1) {
        this->LoadOnRenderThread();
      } else if (this->loadrequest == 2) {
        this->loadrequest = 1;
      }

      // poll and handle events
      this->platform_ui->PollEvents();

      // upload assets if requested
      bool upload_notify = false;
      if (hfield_upload_ != -1) {
        mjr_uploadHField(m_, &platform_ui->mjr_context(), hfield_upload_);
        hfield_upload_ = -1;
        upload_notify  = true;
      }
      if (mesh_upload_ != -1) {
        mjr_uploadMesh(m_, &platform_ui->mjr_context(), mesh_upload_);
        mesh_upload_  = -1;
        upload_notify = true;
      }
      if (texture_upload_ != -1) {
        mjr_uploadTexture(m_, &platform_ui->mjr_context(), texture_upload_);
        texture_upload_ = -1;
        upload_notify   = true;
      }
      if (upload_notify) { cond_upload_.notify_all(); }

      // update scene, doing a full sync if in fully managed mode
      if (!is_passive_) {
        Sync();
      } else if (m_passive_ && d_passive_) {
        // the user has called Sync() in their code
        mjv_updateScene(m_passive_,
                        d_passive_,
                        &this->opt,
                        &this->pert,
                        &this->cam,
                        mjCAT_ALL,
                        &this->scn);

        // add user geoms to scene
        int nusergeom = user_scn_geoms_.size();
        int ngeom     = std::min(nusergeom, this->scn.maxgeom - this->scn.ngeom);
        if (ngeom < nusergeom) {
          if (!this->scn.status) {
            mju_warning(
                "Pre-allocated visual geom buffer is full. "
                "Increase maxgeom above %d.",
                this->scn.maxgeom);
            this->scn.status = 1;
          }
        }
        std::memcpy(this->scn.geoms + this->scn.ngeom,
                    user_scn_geoms_.data(),
                    ngeom * sizeof(mjvGeom));
        this->scn.ngeom += ngeom;
      }
    }  // MutexLock (unblocks simulation thread)

    // render while simulation is running
    this->Render();

    // update FPS stat, at most 5 times per second
    auto   now      = mj::Simulate::Clock::now();
    double interval = Seconds(now - last_fps_update_).count();
    ++frames_;
    if (interval > 0.2) {
      last_fps_update_ = now;
      fps_             = frames_ / interval;
      frames_          = 0;
    }
  }

  const MutexLock lock(this->mtx);
  mjv_freeScene(&this->scn);
  if (is_passive_) {
    mj_deleteData(d_passive_);
    mj_deleteModel(m_passive_);
  }

  this->exitrequest.store(2);
}

// add state to history buffer
void Simulate::AddToHistory() {
  if (history_.empty()) { return; }

  // circular increment of cursor
  history_cursor_ = (history_cursor_ + 1) % nhistory_;

  // add state at cursor
  mjtNum* state = &history_[state_size_ * history_cursor_];
  mj_getState(m_, d_, state, mjSTATE_INTEGRATION);
}

// inject Brownian noise
void Simulate::InjectNoise(int key) {
  // no noise, return
  if (ctrl_noise_std <= 0) { return; }

  // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
  mjtNum rate  = mju_exp(-m_->opt.timestep / ctrl_noise_rate);
  mjtNum scale = ctrl_noise_std * mju_sqrt(1 - rate * rate);

  for (int i = 0; i < m_->nu; i++) {
    mjtNum bottom = 0, top = 0, midpoint = 0, halfrange = 1;
    if (m_->actuator_ctrllimited[i]) {
      bottom    = m_->actuator_ctrlrange[2 * i];
      top       = m_->actuator_ctrlrange[2 * i + 1];
      midpoint  = 0.5 * (top + bottom);  // target of exponential decay
      halfrange = 0.5 * (top - bottom);  // scales noise
    }

    // overwrite midpoint with keyframe, if given
    if (key >= 0) { midpoint = m_->key_ctrl[key * m_->nu + i]; }

    // exponential convergence to midpoint at ctrl_noise_rate
    d_->ctrl[i] = rate * d_->ctrl[i] + (1 - rate) * midpoint;

    // add noise
    d_->ctrl[i] += scale * halfrange * mju_standardNormal(nullptr);

    // clip to range if limited
    if (m_->actuator_ctrllimited[i]) { d_->ctrl[i] = mju_clip(d_->ctrl[i], bottom, top); }
  }
}

void Simulate::UpdateHField(int hfieldid) {
  MutexLock lock(this->mtx);
  if (!m_ || hfieldid < 0 || hfieldid >= m_->nhfield) { return; }
  hfield_upload_ = hfieldid;
  cond_upload_.wait(lock, [this]() { return hfield_upload_ == -1; });
}

void Simulate::UpdateMesh(int meshid) {
  MutexLock lock(this->mtx);
  if (!m_ || meshid < 0 || meshid >= m_->nmesh) { return; }
  mesh_upload_ = meshid;
  cond_upload_.wait(lock, [this]() { return mesh_upload_ == -1; });
}

void Simulate::UpdateTexture(int texid) {
  MutexLock lock(this->mtx);
  if (!m_ || texid < 0 || texid >= m_->ntex) { return; }
  texture_upload_ = texid;
  cond_upload_.wait(lock, [this]() { return texture_upload_ == -1; });
}
}  // namespace mujoco
