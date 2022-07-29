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
#include "uitools.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <mujoco/mjxmacro.h>
#include "array_safety.h"

// When launched via an App Bundle on macOS, the working directory is the path to the App Bundle's
// resource directory. This causes files to be saved into the bundle, which is not the desired
// behavior. Instead, we open a save dialog box to ask the user where to put the file.
// Since the dialog box logic needs to be written in Objective-C, we separate it into a different
// source file.
#ifdef __APPLE__
std::string getSavePath(const char* filename);
#else
static std::string getSavePath(const char* filename) {
  return filename;
}
#endif

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

//-------------------------------- global -----------------------------------------------

const int maxgeom = 5000;           // preallocated geom array in mjvScene
const int max_slow_down = 128;      // maximum slow-down quotient
const double zoom_increment = 0.02; // ratio of single click-wheel zoom increment to vertical extent

// section ids
enum {
  // left ui
  SECT_FILE   = 0,
  SECT_OPTION,
  SECT_SIMULATION,
  SECT_WATCH,
  SECT_PHYSICS,
  SECT_RENDERING,
  SECT_GROUP,
  NSECT0,

  // right ui
  SECT_JOINT = 0,
  SECT_CONTROL,
  NSECT1
};

// file section of UI
const mjuiDef defFile[] = {
  {mjITEM_SECTION,   "File",          1, nullptr,                    "AF"},
  {mjITEM_BUTTON,    "Save xml",      2, nullptr,                    ""},
  {mjITEM_BUTTON,    "Save mjb",      2, nullptr,                    ""},
  {mjITEM_BUTTON,    "Print model",   2, nullptr,                    "CM"},
  {mjITEM_BUTTON,    "Print data",    2, nullptr,                    "CD"},
  {mjITEM_BUTTON,    "Quit",          1, nullptr,                    "CQ"},
  {mjITEM_END}
};

// help strings
const char help_content[] =
  "Space\n"
  "+  -\n"
  "Right arrow\n"
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
  "UI right hold\n"
  "UI title double-click";

const char help_title[] =
  "Play / Pause\n"
  "Speed up / down\n"
  "Step\n"
  "Cycle cameras\n"
  "Free camera\n"
  "Select\n"
  "Select parent\n"
  "Center\n"
  "Tracking camera\n"
  "Zoom\n"
  "View rotate\n"
  "View translate\n"
  "Object rotate\n"
  "Object translate\n"
  "Help\n"
  "Info\n"
  "Profiler\n"
  "Sensors\n"
  "Full screen\n"
  "Show UI shortcuts\n"
  "Expand/collapse all";


//-------------------------------- profiler, sensor, info, watch -----------------------------------

// init profiler figures
void profilerinit(mj::Simulate* simulate) {
  int i, n;

  // set figures to default
  mjv_defaultFigure(&simulate->figconstraint);
  mjv_defaultFigure(&simulate->figcost);
  mjv_defaultFigure(&simulate->figtimer);
  mjv_defaultFigure(&simulate->figsize);

  // titles
  mju::strcpy_arr(simulate->figconstraint.title, "Counts");
  mju::strcpy_arr(simulate->figcost.title, "Convergence (log 10)");
  mju::strcpy_arr(simulate->figsize.title, "Dimensions");
  mju::strcpy_arr(simulate->figtimer.title, "CPU time (msec)");

  // x-labels
  mju::strcpy_arr(simulate->figconstraint.xlabel, "Solver iteration");
  mju::strcpy_arr(simulate->figcost.xlabel, "Solver iteration");
  mju::strcpy_arr(simulate->figsize.xlabel, "Video frame");
  mju::strcpy_arr(simulate->figtimer.xlabel, "Video frame");

  // y-tick nubmer formats
  mju::strcpy_arr(simulate->figconstraint.yformat, "%.0f");
  mju::strcpy_arr(simulate->figcost.yformat, "%.1f");
  mju::strcpy_arr(simulate->figsize.yformat, "%.0f");
  mju::strcpy_arr(simulate->figtimer.yformat, "%.2f");

  // colors
  simulate->figconstraint.figurergba[0] = 0.1f;
  simulate->figcost.figurergba[2]       = 0.2f;
  simulate->figsize.figurergba[0]       = 0.1f;
  simulate->figtimer.figurergba[2]      = 0.2f;
  simulate->figconstraint.figurergba[3] = 0.5f;
  simulate->figcost.figurergba[3]       = 0.5f;
  simulate->figsize.figurergba[3]       = 0.5f;
  simulate->figtimer.figurergba[3]      = 0.5f;

  // legends
  mju::strcpy_arr(simulate->figconstraint.linename[0], "total");
  mju::strcpy_arr(simulate->figconstraint.linename[1], "active");
  mju::strcpy_arr(simulate->figconstraint.linename[2], "changed");
  mju::strcpy_arr(simulate->figconstraint.linename[3], "evals");
  mju::strcpy_arr(simulate->figconstraint.linename[4], "updates");
  mju::strcpy_arr(simulate->figcost.linename[0], "improvement");
  mju::strcpy_arr(simulate->figcost.linename[1], "gradient");
  mju::strcpy_arr(simulate->figcost.linename[2], "lineslope");
  mju::strcpy_arr(simulate->figsize.linename[0], "dof");
  mju::strcpy_arr(simulate->figsize.linename[1], "body");
  mju::strcpy_arr(simulate->figsize.linename[2], "constraint");
  mju::strcpy_arr(simulate->figsize.linename[3], "sqrt(nnz)");
  mju::strcpy_arr(simulate->figsize.linename[4], "contact");
  mju::strcpy_arr(simulate->figsize.linename[5], "iteration");
  mju::strcpy_arr(simulate->figtimer.linename[0], "total");
  mju::strcpy_arr(simulate->figtimer.linename[1], "collision");
  mju::strcpy_arr(simulate->figtimer.linename[2], "prepare");
  mju::strcpy_arr(simulate->figtimer.linename[3], "solve");
  mju::strcpy_arr(simulate->figtimer.linename[4], "other");

  // grid sizes
  simulate->figconstraint.gridsize[0] = 5;
  simulate->figconstraint.gridsize[1] = 5;
  simulate->figcost.gridsize[0] = 5;
  simulate->figcost.gridsize[1] = 5;
  simulate->figsize.gridsize[0] = 3;
  simulate->figsize.gridsize[1] = 5;
  simulate->figtimer.gridsize[0] = 3;
  simulate->figtimer.gridsize[1] = 5;

  // minimum ranges
  simulate->figconstraint.range[0][0] = 0;
  simulate->figconstraint.range[0][1] = 20;
  simulate->figconstraint.range[1][0] = 0;
  simulate->figconstraint.range[1][1] = 80;
  simulate->figcost.range[0][0] = 0;
  simulate->figcost.range[0][1] = 20;
  simulate->figcost.range[1][0] = -15;
  simulate->figcost.range[1][1] = 5;
  simulate->figsize.range[0][0] = -200;
  simulate->figsize.range[0][1] = 0;
  simulate->figsize.range[1][0] = 0;
  simulate->figsize.range[1][1] = 100;
  simulate->figtimer.range[0][0] = -200;
  simulate->figtimer.range[0][1] = 0;
  simulate->figtimer.range[1][0] = 0;
  simulate->figtimer.range[1][1] = 0.4f;

  // init x axis on history figures (do not show yet)
  for (n=0; n<6; n++)
    for (i=0; i<mjMAXLINEPNT; i++) {
      simulate->figtimer.linedata[n][2*i] = (float)-i;
      simulate->figsize.linedata[n][2*i] = (float)-i;
    }
}

// update profiler figures
void profilerupdate(mj::Simulate* simulate) {
  int i, n;

  // update constraint figure
  simulate->figconstraint.linepnt[0] = mjMIN(mjMIN(simulate->d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
  for (i=1; i<5; i++) {
    simulate->figconstraint.linepnt[i] = simulate->figconstraint.linepnt[0];
  }
  if (simulate->m->opt.solver==mjSOL_PGS) {
    simulate->figconstraint.linepnt[3] = 0;
    simulate->figconstraint.linepnt[4] = 0;
  }
  if (simulate->m->opt.solver==mjSOL_CG) {
    simulate->figconstraint.linepnt[4] = 0;
  }
  for (i=0; i<simulate->figconstraint.linepnt[0]; i++) {
    // x
    simulate->figconstraint.linedata[0][2*i] = (float)i;
    simulate->figconstraint.linedata[1][2*i] = (float)i;
    simulate->figconstraint.linedata[2][2*i] = (float)i;
    simulate->figconstraint.linedata[3][2*i] = (float)i;
    simulate->figconstraint.linedata[4][2*i] = (float)i;

    // y
    simulate->figconstraint.linedata[0][2*i+1] = (float)simulate->d->nefc;
    simulate->figconstraint.linedata[1][2*i+1] = (float)simulate->d->solver[i].nactive;
    simulate->figconstraint.linedata[2][2*i+1] = (float)simulate->d->solver[i].nchange;
    simulate->figconstraint.linedata[3][2*i+1] = (float)simulate->d->solver[i].neval;
    simulate->figconstraint.linedata[4][2*i+1] = (float)simulate->d->solver[i].nupdate;
  }

  // update cost figure
  simulate->figcost.linepnt[0] = mjMIN(mjMIN(simulate->d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
  for (i=1; i<3; i++) {
    simulate->figcost.linepnt[i] = simulate->figcost.linepnt[0];
  }
  if (simulate->m->opt.solver==mjSOL_PGS) {
    simulate->figcost.linepnt[1] = 0;
    simulate->figcost.linepnt[2] = 0;
  }

  for (i=0; i<simulate->figcost.linepnt[0]; i++) {
    // x
    simulate->figcost.linedata[0][2*i] = (float)i;
    simulate->figcost.linedata[1][2*i] = (float)i;
    simulate->figcost.linedata[2][2*i] = (float)i;

    // y
    simulate->figcost.linedata[0][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, simulate->d->solver[i].improvement));
    simulate->figcost.linedata[1][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, simulate->d->solver[i].gradient));
    simulate->figcost.linedata[2][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, simulate->d->solver[i].lineslope));
  }

  // get timers: total, collision, prepare, solve, other
  mjtNum total = simulate->d->timer[mjTIMER_STEP].duration;
  int number = simulate->d->timer[mjTIMER_STEP].number;
  if (!number) {
    total = simulate->d->timer[mjTIMER_FORWARD].duration;
    number = simulate->d->timer[mjTIMER_FORWARD].number;
  }
  number = mjMAX(1, number);
  float tdata[5] = {
    (float)(total/number),
    (float)(simulate->d->timer[mjTIMER_POS_COLLISION].duration/number),
    (float)(simulate->d->timer[mjTIMER_POS_MAKE].duration/number) +
    (float)(simulate->d->timer[mjTIMER_POS_PROJECT].duration/number),
    (float)(simulate->d->timer[mjTIMER_CONSTRAINT].duration/number),
    0
  };
  tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

  // update figtimer
  int pnt = mjMIN(201, simulate->figtimer.linepnt[0]+1);
  for (n=0; n<5; n++) {
    // shift data
    for (i=pnt-1; i>0; i--) {
      simulate->figtimer.linedata[n][2*i+1] = simulate->figtimer.linedata[n][2*i-1];
    }

    // assign new
    simulate->figtimer.linepnt[n] = pnt;
    simulate->figtimer.linedata[n][1] = tdata[n];
  }

  // get sizes: nv, nbody, nefc, sqrt(nnz), ncont, iter
  float sdata[6] = {
    (float)simulate->m->nv,
    (float)simulate->m->nbody,
    (float)simulate->d->nefc,
    (float)mju_sqrt((mjtNum)simulate->d->solver_nnz),
    (float)simulate->d->ncon,
    (float)simulate->d->solver_iter
  };

  // update figsize
  pnt = mjMIN(201, simulate->figsize.linepnt[0]+1);
  for (n=0; n<6; n++) {
    // shift data
    for (i=pnt-1; i>0; i--) {
      simulate->figsize.linedata[n][2*i+1] = simulate->figsize.linedata[n][2*i-1];
    }

    // assign new
    simulate->figsize.linepnt[n] = pnt;
    simulate->figsize.linedata[n][1] = sdata[n];
  }
}

// show profiler figures
void profilershow(mj::Simulate* simulate, mjrRect rect) {
  mjrRect viewport = {
    rect.left + rect.width - rect.width/4,
    rect.bottom,
    rect.width/4,
    rect.height/4
  };
  mjr_figure(viewport, &simulate->figtimer, &simulate->con);
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &simulate->figsize, &simulate->con);
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &simulate->figcost, &simulate->con);
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &simulate->figconstraint, &simulate->con);
}


// init sensor figure
void sensorinit(mj::Simulate* simulate) {
  // set figure to default
  mjv_defaultFigure(&simulate->figsensor);
  simulate->figsensor.figurergba[3] = 0.5f;

  // set flags
  simulate->figsensor.flg_extend = 1;
  simulate->figsensor.flg_barplot = 1;
  simulate->figsensor.flg_symmetric = 1;

  // title
  mju::strcpy_arr(simulate->figsensor.title, "Sensor data");

  // y-tick nubmer format
  mju::strcpy_arr(simulate->figsensor.yformat, "%.0f");

  // grid size
  simulate->figsensor.gridsize[0] = 2;
  simulate->figsensor.gridsize[1] = 3;

  // minimum range
  simulate->figsensor.range[0][0] = 0;
  simulate->figsensor.range[0][1] = 0;
  simulate->figsensor.range[1][0] = -1;
  simulate->figsensor.range[1][1] = 1;
}

// update sensor figure
void sensorupdate(mj::Simulate* simulate) {
  static const int maxline = 10;

  // clear linepnt
  for (int i=0; i<maxline; i++) {
    simulate->figsensor.linepnt[i] = 0;
  }

  // start with line 0
  int lineid = 0;

  // loop over sensors
  for (int n=0; n<simulate->m->nsensor; n++) {
    // go to next line if type is different
    if (n>0 && simulate->m->sensor_type[n]!=simulate->m->sensor_type[n-1]) {
      lineid = mjMIN(lineid+1, maxline-1);
    }

    // get info about this sensor
    mjtNum cutoff = (simulate->m->sensor_cutoff[n]>0 ? simulate->m->sensor_cutoff[n] : 1);
    int adr = simulate->m->sensor_adr[n];
    int dim = simulate->m->sensor_dim[n];

    // data pointer in line
    int p = simulate->figsensor.linepnt[lineid];

    // fill in data for this sensor
    for (int i=0; i<dim; i++) {
      // check size
      if ((p+2*i)>=mjMAXLINEPNT/2) {
        break;
      }

      // x
      simulate->figsensor.linedata[lineid][2*p+4*i] = (float)(adr+i);
      simulate->figsensor.linedata[lineid][2*p+4*i+2] = (float)(adr+i);

      // y
      simulate->figsensor.linedata[lineid][2*p+4*i+1] = 0;
      simulate->figsensor.linedata[lineid][2*p+4*i+3] = (float)(simulate->d->sensordata[adr+i]/cutoff);
    }

    // update linepnt
    simulate->figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT-1,
                                      simulate->figsensor.linepnt[lineid]+2*dim);
  }
}

// show sensor figure
void sensorshow(mj::Simulate* simulate, mjrRect rect) {
  // constant width with and without profiler
  int width = simulate->profiler ? rect.width/3 : rect.width/4;

  // render figure on the right
  mjrRect viewport = {
    rect.left + rect.width - width,
    rect.bottom,
    width,
    rect.height/3
  };
  mjr_figure(viewport, &simulate->figsensor, &simulate->con);
}

// prepare info text
void infotext(mj::Simulate* simulate,
              char (&title)[mj::Simulate::kMaxFilenameLength],
              char (&content)[mj::Simulate::kMaxFilenameLength],
              double interval) {
  char tmp[20];

  // compute solver error
  mjtNum solerr = 0;
  if (simulate->d->solver_iter) {
    int ind = mjMIN(simulate->d->solver_iter-1, mjNSOLVER-1);
    solerr = mju_min(simulate->d->solver[ind].improvement, simulate->d->solver[ind].gradient);
    if (solerr==0) {
      solerr = mju_max(simulate->d->solver[ind].improvement, simulate->d->solver[ind].gradient);
    }
  }
  solerr = mju_log10(mju_max(mjMINVAL, solerr));

  // prepare info text
  const std::string realtime_nominator = simulate->slow_down == 1 ? "" : "1/";
  mju::strcpy_arr(title, "Time\nSize\nCPU\nSolver   \nFPS\nstack\nconbuf\nefcbuf");
  mju::sprintf_arr(content,
                   "%-9.3f %s%d x\n%d  (%d simulate->con)\n%.3f\n%.1f  (%d it)\n%.0f\n%.3f\n%.3f\n%.3f",
                   simulate->d->time, realtime_nominator.c_str(), simulate->slow_down,
                   simulate->d->nefc, simulate->d->ncon,
                   simulate->run ?
                   simulate->d->timer[mjTIMER_STEP].duration / mjMAX(1, simulate->d->timer[mjTIMER_STEP].number) :
                   simulate->d->timer[mjTIMER_FORWARD].duration / mjMAX(1, simulate->d->timer[mjTIMER_FORWARD].number),
                   solerr, simulate->d->solver_iter,
                   1/interval,
                   simulate->d->maxuse_stack/(double)simulate->d->nstack,
                   simulate->d->maxuse_con/(double)simulate->m->nconmax,
                   simulate->d->maxuse_efc/(double)simulate->m->njmax);

  // add Energy if enabled
  {
    mjModel* m = simulate->m; // for mjENABLED
    if (mjENABLED(mjENBL_ENERGY)) {
      mju::sprintf_arr(tmp, "\n%.3f", simulate->d->energy[0]+simulate->d->energy[1]);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if (mjENABLED(mjENBL_FWDINV)) {
      mju::sprintf_arr(tmp, "\n%.1f %.1f",
                       mju_log10(mju_max(mjMINVAL, simulate->d->solver_fwdinv[0])),
                       mju_log10(mju_max(mjMINVAL, simulate->d->solver_fwdinv[1])));
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nFwdInv");
    }
  }
}

// sprintf forwarding, to avoid compiler warning in x-macro
void printfield(char (&str)[mjMAXUINAME], void* ptr) {
  mju::sprintf_arr(str, "%g", *(mjtNum*)ptr);
}

// update watch
void watch(mj::Simulate* simulate) {
  // clear
  simulate->ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
  mju::strcpy_arr(simulate->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

  // prepare symbols needed by xmacro
  MJDATA_POINTERS_PREAMBLE(simulate->m);

  // find specified field in mjData arrays, update value
  #define X(TYPE, NAME, NR, NC)                                                             \
    if (!mju::strcmp_arr(#NAME, simulate->field) &&                                              \
        !mju::strcmp_arr(#TYPE, "mjtNum")) {                                                \
      if (simulate->index>=0 && simulate->index<simulate->m->NR*NC) {                                           \
        printfield(simulate->ui0.sect[SECT_WATCH].item[2].multi.name[0], simulate->d->NAME + simulate->index);  \
      } else {                                                                              \
        mju::strcpy_arr(simulate->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid index");  \
      }                                                                                     \
      return;                                                                               \
    }

    MJDATA_POINTERS
  #undef X
}


//---------------------------------- UI construction -----------------------------------------------

// make physics section of UI
void makephysics(mj::Simulate* simulate, int oldstate) {
  int i;

  mjuiDef defPhysics[] = {
    {mjITEM_SECTION,   "Physics",       oldstate, nullptr,                        "AP"},
    {mjITEM_SELECT,    "Integrator",    2, &(simulate->m->opt.integrator),        "Euler\nRK4\nimplicit"},
    {mjITEM_SELECT,    "Collision",     2, &(simulate->m->opt.collision),         "All\nPair\nDynamic"},
    {mjITEM_SELECT,    "Cone",          2, &(simulate->m->opt.cone),              "Pyramidal\nElliptic"},
    {mjITEM_SELECT,    "Jacobian",      2, &(simulate->m->opt.jacobian),          "Dense\nSparse\nAuto"},
    {mjITEM_SELECT,    "Solver",        2, &(simulate->m->opt.solver),            "PGS\nCG\nNewton"},
    {mjITEM_SEPARATOR, "Algorithmic Parameters", 1},
    {mjITEM_EDITNUM,   "Timestep",      2, &(simulate->m->opt.timestep),          "1 0 1"},
    {mjITEM_EDITINT,   "Iterations",    2, &(simulate->m->opt.iterations),        "1 0 1000"},
    {mjITEM_EDITNUM,   "Tolerance",     2, &(simulate->m->opt.tolerance),         "1 0 1"},
    {mjITEM_EDITINT,   "Noslip Iter",   2, &(simulate->m->opt.noslip_iterations), "1 0 1000"},
    {mjITEM_EDITNUM,   "Noslip Tol",    2, &(simulate->m->opt.noslip_tolerance),  "1 0 1"},
    {mjITEM_EDITINT,   "MRR Iter",      2, &(simulate->m->opt.mpr_iterations),    "1 0 1000"},
    {mjITEM_EDITNUM,   "MPR Tol",       2, &(simulate->m->opt.mpr_tolerance),     "1 0 1"},
    {mjITEM_EDITNUM,   "API Rate",      2, &(simulate->m->opt.apirate),           "1 0 1000"},
    {mjITEM_SEPARATOR, "Physical Parameters", 1},
    {mjITEM_EDITNUM,   "Gravity",       2, simulate->m->opt.gravity,              "3"},
    {mjITEM_EDITNUM,   "Wind",          2, simulate->m->opt.wind,                 "3"},
    {mjITEM_EDITNUM,   "Magnetic",      2, simulate->m->opt.magnetic,             "3"},
    {mjITEM_EDITNUM,   "Density",       2, &(simulate->m->opt.density),           "1"},
    {mjITEM_EDITNUM,   "Viscosity",     2, &(simulate->m->opt.viscosity),         "1"},
    {mjITEM_EDITNUM,   "Imp Ratio",     2, &(simulate->m->opt.impratio),          "1"},
    {mjITEM_SEPARATOR, "Disable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defEnableFlags[] = {
    {mjITEM_SEPARATOR, "Enable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defOverride[] = {
    {mjITEM_SEPARATOR, "Contact Override", 1},
    {mjITEM_EDITNUM,   "Margin",        2, &(simulate->m->opt.o_margin),          "1"},
    {mjITEM_EDITNUM,   "Sol Imp",       2, &(simulate->m->opt.o_solimp),          "5"},
    {mjITEM_EDITNUM,   "Sol Ref",       2, &(simulate->m->opt.o_solref),          "2"},
    {mjITEM_END}
  };

  // add physics
  mjui_add(&simulate->ui0, defPhysics);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKINT,  "", 2, nullptr, ""},
    {mjITEM_END}
  };
  for (i=0; i<mjNDISABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjDISABLESTRING[i]);
    defFlag[0].pdata = simulate->disable + i;
    mjui_add(&simulate->ui0, defFlag);
  }
  mjui_add(&simulate->ui0, defEnableFlags);
  for (i=0; i<mjNENABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjENABLESTRING[i]);
    defFlag[0].pdata = simulate->enable + i;
    mjui_add(&simulate->ui0, defFlag);
  }

  // add contact override
  mjui_add(&simulate->ui0, defOverride);
}



// make rendering section of UI
void makerendering(mj::Simulate* simulate, int oldstate) {
  int i, j;

  mjuiDef defRendering[] = {
    {
      mjITEM_SECTION,
      "Rendering",
      oldstate,
      nullptr,
      "AR"
    },
    {
      mjITEM_SELECT,
      "Camera",
      2,
      &(simulate->camera),
      "Free\nTracking"
    },
    {
      mjITEM_SELECT,
      "Label",
      2,
      &(simulate->vopt.label),
      "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\n"
      "Actuator\nConstraint\nSkin\nSelection\nSel Pnt\nForce"
    },
    {
      mjITEM_SELECT,
      "Frame",
      2,
      &(simulate->vopt.frame),
      "None\nBody\nGeom\nSite\nCamera\nLight\nContact\nWorld"
    },
    {
      mjITEM_BUTTON,
      "Copy camera",
      2,
      nullptr,
      ""
    },
    {
      mjITEM_SEPARATOR,
      "Model Elements",
      1
    },
    {
      mjITEM_END
    }
  };
  mjuiDef defOpenGL[] = {
    {mjITEM_SEPARATOR, "OpenGL Effects", 1},
    {mjITEM_END}
  };

  // add model cameras, up to UI limit
  for (i=0; i<mjMIN(simulate->m->ncam, mjMAXUIMULTI-2); i++) {
    // prepare name
    char camname[mjMAXUITEXT] = "\n";
    if (simulate->m->names[simulate->m->name_camadr[i]]) {
      mju::strcat_arr(camname, simulate->m->names+simulate->m->name_camadr[i]);
    } else {
      mju::sprintf_arr(camname, "\nCamera %d", i);
    }

    // check string length
    if (mju::strlen_arr(camname) + mju::strlen_arr(defRendering[1].other)>=mjMAXUITEXT-1) {
      break;
    }

    // add camera
    mju::strcat_arr(defRendering[1].other, camname);
  }

  // add rendering standard
  mjui_add(&simulate->ui0, defRendering);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKBYTE,  "", 2, nullptr, ""},
    {mjITEM_END}
  };
  for (i=0; i<mjNVISFLAG; i++) {
    // set name, remove "&"
    mju::strcpy_arr(defFlag[0].name, mjVISSTRING[i][0]);
    for (j=0; j<strlen(mjVISSTRING[i][0]); j++)
      if (mjVISSTRING[i][0][j]=='&') {
        mju_strncpy(
          defFlag[0].name+j, mjVISSTRING[i][0]+j+1, mju::sizeof_arr(defFlag[0].name)-j);
        break;
      }

    // set shortcut and data
    mju::sprintf_arr(defFlag[0].other, " %s", mjVISSTRING[i][2]);
    defFlag[0].pdata = simulate->vopt.flags + i;
    mjui_add(&simulate->ui0, defFlag);
  }
  mjui_add(&simulate->ui0, defOpenGL);
  for (i=0; i<mjNRNDFLAG; i++) {
    mju::strcpy_arr(defFlag[0].name, mjRNDSTRING[i][0]);
    mju::sprintf_arr(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
    defFlag[0].pdata = simulate->scn.flags + i;
    mjui_add(&simulate->ui0, defFlag);
  }
}



// make group section of UI
void makegroup(mj::Simulate* simulate, int oldstate) {
  mjuiDef defGroup[] = {
    {mjITEM_SECTION,    "Group enable",     oldstate, nullptr,                    "AG"},
    {mjITEM_SEPARATOR,  "Geom groups",  1},
    {mjITEM_CHECKBYTE,  "Geom 0",           2, simulate->vopt.geomgroup,          " 0"},
    {mjITEM_CHECKBYTE,  "Geom 1",           2, simulate->vopt.geomgroup+1,        " 1"},
    {mjITEM_CHECKBYTE,  "Geom 2",           2, simulate->vopt.geomgroup+2,        " 2"},
    {mjITEM_CHECKBYTE,  "Geom 3",           2, simulate->vopt.geomgroup+3,        " 3"},
    {mjITEM_CHECKBYTE,  "Geom 4",           2, simulate->vopt.geomgroup+4,        " 4"},
    {mjITEM_CHECKBYTE,  "Geom 5",           2, simulate->vopt.geomgroup+5,        " 5"},
    {mjITEM_SEPARATOR,  "Site groups",  1},
    {mjITEM_CHECKBYTE,  "Site 0",           2, simulate->vopt.sitegroup,          "S0"},
    {mjITEM_CHECKBYTE,  "Site 1",           2, simulate->vopt.sitegroup+1,        "S1"},
    {mjITEM_CHECKBYTE,  "Site 2",           2, simulate->vopt.sitegroup+2,        "S2"},
    {mjITEM_CHECKBYTE,  "Site 3",           2, simulate->vopt.sitegroup+3,        "S3"},
    {mjITEM_CHECKBYTE,  "Site 4",           2, simulate->vopt.sitegroup+4,        "S4"},
    {mjITEM_CHECKBYTE,  "Site 5",           2, simulate->vopt.sitegroup+5,        "S5"},
    {mjITEM_SEPARATOR,  "Joint groups", 1},
    {mjITEM_CHECKBYTE,  "Joint 0",          2, simulate->vopt.jointgroup,         ""},
    {mjITEM_CHECKBYTE,  "Joint 1",          2, simulate->vopt.jointgroup+1,       ""},
    {mjITEM_CHECKBYTE,  "Joint 2",          2, simulate->vopt.jointgroup+2,       ""},
    {mjITEM_CHECKBYTE,  "Joint 3",          2, simulate->vopt.jointgroup+3,       ""},
    {mjITEM_CHECKBYTE,  "Joint 4",          2, simulate->vopt.jointgroup+4,       ""},
    {mjITEM_CHECKBYTE,  "Joint 5",          2, simulate->vopt.jointgroup+5,       ""},
    {mjITEM_SEPARATOR,  "Tendon groups",    1},
    {mjITEM_CHECKBYTE,  "Tendon 0",         2, simulate->vopt.tendongroup,        ""},
    {mjITEM_CHECKBYTE,  "Tendon 1",         2, simulate->vopt.tendongroup+1,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 2",         2, simulate->vopt.tendongroup+2,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 3",         2, simulate->vopt.tendongroup+3,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 4",         2, simulate->vopt.tendongroup+4,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 5",         2, simulate->vopt.tendongroup+5,      ""},
    {mjITEM_SEPARATOR,  "Actuator groups", 1},
    {mjITEM_CHECKBYTE,  "Actuator 0",       2, simulate->vopt.actuatorgroup,      ""},
    {mjITEM_CHECKBYTE,  "Actuator 1",       2, simulate->vopt.actuatorgroup+1,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 2",       2, simulate->vopt.actuatorgroup+2,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 3",       2, simulate->vopt.actuatorgroup+3,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 4",       2, simulate->vopt.actuatorgroup+4,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 5",       2, simulate->vopt.actuatorgroup+5,    ""},
    {mjITEM_SEPARATOR,  "Skin groups", 1},
    {mjITEM_CHECKBYTE,  "Skin 0",           2, simulate->vopt.skingroup,          ""},
    {mjITEM_CHECKBYTE,  "Skin 1",           2, simulate->vopt.skingroup+1,        ""},
    {mjITEM_CHECKBYTE,  "Skin 2",           2, simulate->vopt.skingroup+2,        ""},
    {mjITEM_CHECKBYTE,  "Skin 3",           2, simulate->vopt.skingroup+3,        ""},
    {mjITEM_CHECKBYTE,  "Skin 4",           2, simulate->vopt.skingroup+4,        ""},
    {mjITEM_CHECKBYTE,  "Skin 5",           2, simulate->vopt.skingroup+5,        ""},
    {mjITEM_END}
  };

  // add section
  mjui_add(&simulate->ui0, defGroup);
}

// make joint section of UI
void makejoint(mj::Simulate* simulate, int oldstate) {
  int i;

  mjuiDef defJoint[] = {
    {mjITEM_SECTION, "Joint", oldstate, nullptr, "AJ"},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&simulate->ui1, defJoint);
  defSlider[0].state = 4;

  // add scalar joints, exit if UI limit reached
  int itemcnt = 0;
  for (i=0; i<simulate->m->njnt && itemcnt<mjMAXUIITEM; i++)
    if ((simulate->m->jnt_type[i]==mjJNT_HINGE || simulate->m->jnt_type[i]==mjJNT_SLIDE)) {
      // skip if joint group is disabled
      if (!simulate->vopt.jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, simulate->m->jnt_group[i]))]) {
        continue;
      }

      // set data and name
      defSlider[0].pdata = simulate->d->qpos + simulate->m->jnt_qposadr[i];
      if (simulate->m->names[simulate->m->name_jntadr[i]]) {
        mju::strcpy_arr(defSlider[0].name, simulate->m->names+simulate->m->name_jntadr[i]);
      } else {
        mju::sprintf_arr(defSlider[0].name, "joint %d", i);
      }

      // set range
      if (simulate->m->jnt_limited[i])
        mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                         simulate->m->jnt_range[2*i], simulate->m->jnt_range[2*i+1]);
      else if (simulate->m->jnt_type[i]==mjJNT_SLIDE) {
        mju::strcpy_arr(defSlider[0].other, "-1 1");
      } else {
        mju::strcpy_arr(defSlider[0].other, "-3.1416 3.1416");
      }

      // add and count
      mjui_add(&simulate->ui1, defSlider);
      itemcnt++;
    }
}

// make control section of UI
void makecontrol(mj::Simulate* simulate, int oldstate) {
  int i;

  mjuiDef defControl[] = {
    {mjITEM_SECTION, "Control", oldstate, nullptr, "AC"},
    {mjITEM_BUTTON,  "Clear all", 2},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&simulate->ui1, defControl);
  defSlider[0].state = 2;

  // add controls, exit if UI limit reached (Clear button already added)
  int itemcnt = 1;
  for (i=0; i<simulate->m->nu && itemcnt<mjMAXUIITEM; i++) {
    // skip if actuator group is disabled
    if (!simulate->vopt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, simulate->m->actuator_group[i]))]) {
      continue;
    }

    // set data and name
    defSlider[0].pdata = simulate->d->ctrl + i;
    if (simulate->m->names[simulate->m->name_actuatoradr[i]]) {
      mju::strcpy_arr(defSlider[0].name, simulate->m->names+simulate->m->name_actuatoradr[i]);
    } else {
      mju::sprintf_arr(defSlider[0].name, "control %d", i);
    }

    // set range
    if (simulate->m->actuator_ctrllimited[i])
      mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                       simulate->m->actuator_ctrlrange[2*i], simulate->m->actuator_ctrlrange[2*i+1]);
    else {
      mju::strcpy_arr(defSlider[0].other, "-1 1");
    }

    // add and count
    mjui_add(&simulate->ui1, defSlider);
    itemcnt++;
  }
}

// make model-dependent UI sections
void makesections(mj::Simulate* simulate) {
  int i;

  // get section open-close state, UI 0
  int oldstate0[NSECT0];
  for (i=0; i<NSECT0; i++) {
    oldstate0[i] = 0;
    if (simulate->ui0.nsect>i) {
      oldstate0[i] = simulate->ui0.sect[i].state;
    }
  }

  // get section open-close state, UI 1
  int oldstate1[NSECT1];
  for (i=0; i<NSECT1; i++) {
    oldstate1[i] = 0;
    if (simulate->ui1.nsect>i) {
      oldstate1[i] = simulate->ui1.sect[i].state;
    }
  }

  // clear model-dependent sections of UI
  simulate->ui0.nsect = SECT_PHYSICS;
  simulate->ui1.nsect = 0;

  // make
  makephysics(simulate, oldstate0[SECT_PHYSICS]);
  makerendering(simulate, oldstate0[SECT_RENDERING]);
  makegroup(simulate, oldstate0[SECT_GROUP]);
  makejoint(simulate, oldstate1[SECT_JOINT]);
  makecontrol(simulate, oldstate1[SECT_CONTROL]);
}

//---------------------------------- utility functions ---------------------------------------------

// align and scale view
void alignscale(mj::Simulate* simulate) {
  // autoscale
  simulate->cam.lookat[0] = simulate->m->stat.center[0];
  simulate->cam.lookat[1] = simulate->m->stat.center[1];
  simulate->cam.lookat[2] = simulate->m->stat.center[2];
  simulate->cam.distance = 1.5 * simulate->m->stat.extent;

  // set to free camera
  simulate->cam.type = mjCAMERA_FREE;
}

// copy qpos to clipboard as key
void copykey(mj::Simulate* simulate) {
  char clipboard[5000] = "<key qpos='";
  char buf[200];

  // prepare string
  for (int i=0; i<simulate->m->nq; i++) {
    mju::sprintf_arr(buf, i==simulate->m->nq-1 ? "%g" : "%g ", simulate->d->qpos[i]);
    mju::strcat_arr(clipboard, buf);
  }
  mju::strcat_arr(clipboard, "'/>");

  // copy to clipboard
  glfwSetClipboardString(simulate->window, clipboard);
}

// millisecond timer, for MuJoCo built-in profiler
mjtNum timer(void) {
  return (mjtNum)(1000*glfwGetTime());
}

// clear all times
void cleartimers(mjData* d) {
  for (int i=0; i<mjNTIMER; i++) {
    d->timer[i].duration = 0;
    d->timer[i].number = 0;
  }
}

// copy current camera to clipboard as MJCF specification
void copycamera(mj::Simulate* simulate) {
  mjvGLCamera* camera = simulate->scn.camera;

  char clipboard[500];
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
              cam_right[0], cam_right[1], cam_right[2],
              camera[0].up[0], camera[0].up[1], camera[0].up[2]);

  // copy spec into clipboard
  glfwSetClipboardString(simulate->window, clipboard);
}

// update UI 0 when MuJoCo structures change (except for joint sliders)
void updatesettings(mj::Simulate* simulate) {
  int i;

  // physics flags
  for (i=0; i<mjNDISABLE; i++) {
    simulate->disable[i] = ((simulate->m->opt.disableflags & (1<<i)) !=0);
  }
  for (i=0; i<mjNENABLE; i++) {
    simulate->enable[i] = ((simulate->m->opt.enableflags & (1<<i)) !=0);
  }

  // camera
  if (simulate->cam.type==mjCAMERA_FIXED) {
    simulate->camera = 2 + simulate->cam.fixedcamid;
  } else if (simulate->cam.type==mjCAMERA_TRACKING) {
    simulate->camera = 1;
  } else {
    simulate->camera = 0;
  }

  // update UI
  mjui_update(-1, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
}


//---------------------------------- UI hooks (for uitools.c) --------------------------------------

// determine enable/disable item state given category
int uiPredicate(int category, void* userdata) {
  mj::Simulate* simulate = (mj::Simulate*)(userdata);

  switch (category) {
  case 2:                 // require model
    return (simulate->m!=nullptr);

  case 3:                 // require model and nkey
    return (simulate->m && simulate->m->nkey);

  case 4:                 // require model and paused
    return (simulate->m && !simulate->run);

  default:
    return 1;
  }
}

// set window layout
void uiLayout(mjuiState* state) {
  mj::Simulate* simulate = (mj::Simulate*)(state->userdata);

  mjrRect* rect = state->rect;

  // set number of rectangles
  state->nrect = 4;

  // rect 0: entire framebuffer
  rect[0].left = 0;
  rect[0].bottom = 0;
  glfwGetFramebufferSize(simulate->window, &rect[0].width, &rect[0].height);

  // rect 1: UI 0
  rect[1].left = 0;
  rect[1].width = simulate->ui0_enable ? simulate->ui0.width : 0;
  rect[1].bottom = 0;
  rect[1].height = rect[0].height;

  // rect 2: UI 1
  rect[2].width = simulate->ui1_enable ? simulate->ui1.width : 0;
  rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
  rect[2].bottom = 0;
  rect[2].height = rect[0].height;

  // rect 3: 3D plot (everything else is an overlay)
  rect[3].left = rect[1].width;
  rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
  rect[3].bottom = 0;
  rect[3].height = rect[0].height;
}

// handle UI event
void uiEvent(mjuiState* state) {
  mj::Simulate* simulate = (mj::Simulate*)(state->userdata);
  int i;
  char err[200];

  // call UI 0 if event is directed to it
  if ((state->dragrect==simulate->ui0.rectid) ||
      (state->dragrect==0 && state->mouserect==simulate->ui0.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&simulate->ui0, state, &simulate->con);

    // file section
    if (it && it->sectionid==SECT_FILE) {
      switch (it->itemid) {
      case 0:             // Save xml
        {
          const std::string path = getSavePath("mjmodel.xml");
          if (!path.empty() && !mj_saveLastXML(path.c_str(), simulate->m, err, 200)) {
            std::printf("Save XML error: %s", err);
          }
        }
        break;

      case 1:             // Save mjb
        {
          const std::string path = getSavePath("mjmodel.mjb");
          if (!path.empty()) {
            mj_saveModel(simulate->m, path.c_str(), NULL, 0);
          }
        }
        break;

      case 2:             // Print model
        mj_printModel(simulate->m, "MJMODEL.TXT");
        break;

      case 3:             // Print data
        mj_printData(simulate->m, simulate->d, "MJDATA.TXT");
        break;

      case 4:             // Quit
        simulate->exitrequest = 1;
        break;
      }
    }

    // option section
    else if (it && it->sectionid==SECT_OPTION) {
      switch (it->itemid) {
      case 0:             // Spacing
        simulate->ui0.spacing = mjui_themeSpacing(simulate->spacing);
        simulate->ui1.spacing = mjui_themeSpacing(simulate->spacing);
        break;

      case 1:             // Color
        simulate->ui0.color = mjui_themeColor(simulate->color);
        simulate->ui1.color = mjui_themeColor(simulate->color);
        break;

      case 2:             // Font
        mjr_changeFont(50*(simulate->font+1), &simulate->con);
        break;

      case 9:             // Full screen
        if (glfwGetWindowMonitor(simulate->window)) {
          // restore window from saved data
          glfwSetWindowMonitor(simulate->window, nullptr, simulate->windowpos[0], simulate->windowpos[1],
                               simulate->windowsize[0], simulate->windowsize[1], 0);
        }

        // currently windowed: switch to full screen
        else {
          // save window data
          glfwGetWindowPos(simulate->window, simulate->windowpos, simulate->windowpos+1);
          glfwGetWindowSize(simulate->window, simulate->windowsize, simulate->windowsize+1);

          // switch
          glfwSetWindowMonitor(simulate->window, glfwGetPrimaryMonitor(), 0, 0,
                               simulate->vmode.width, simulate->vmode.height, simulate->vmode.refreshRate);
        }

        // reinstante vsync, just in case
        glfwSwapInterval(simulate->vsync);
        break;

      case 10:            // Vertical sync
        glfwSwapInterval(simulate->vsync);
        break;
      }

      // modify UI
      uiModify(simulate->window, &simulate->ui0, state, &simulate->con);
      uiModify(simulate->window, &simulate->ui1, state, &simulate->con);
    }

    // simulation section
    else if (it && it->sectionid==SECT_SIMULATION) {
      switch (it->itemid) {
      case 1:             // Reset
        if (simulate->m) {
          mj_resetData(simulate->m, simulate->d);
          mj_forward(simulate->m, simulate->d);
          profilerupdate(simulate);
          sensorupdate(simulate);
          updatesettings(simulate);
        }
        break;

      case 2:             // Reload
        simulate->uiloadrequest = 1;
        break;

      case 3:             // Align
        alignscale(simulate);
        updatesettings(simulate);
        break;

      case 4:             // Copy pose
        copykey(simulate);
        break;

      case 5:             // Adjust key
      case 6:             // Load key
        i = simulate->key;
        simulate->d->time = simulate->m->key_time[i];
        mju_copy(simulate->d->qpos, simulate->m->key_qpos+i*simulate->m->nq, simulate->m->nq);
        mju_copy(simulate->d->qvel, simulate->m->key_qvel+i*simulate->m->nv, simulate->m->nv);
        mju_copy(simulate->d->act, simulate->m->key_act+i*simulate->m->na, simulate->m->na);
        mju_copy(simulate->d->mocap_pos, simulate->m->key_mpos+i*3*simulate->m->nmocap, 3*simulate->m->nmocap);
        mju_copy(simulate->d->mocap_quat, simulate->m->key_mquat+i*4*simulate->m->nmocap, 4*simulate->m->nmocap);
        mju_copy(simulate->d->ctrl, simulate->m->key_ctrl+i*simulate->m->nu, simulate->m->nu);
        mj_forward(simulate->m, simulate->d);
        profilerupdate(simulate);
        sensorupdate(simulate);
        updatesettings(simulate);
        break;

      case 7:             // Save key
        i = simulate->key;
        simulate->m->key_time[i] = simulate->d->time;
        mju_copy(simulate->m->key_qpos+i*simulate->m->nq, simulate->d->qpos, simulate->m->nq);
        mju_copy(simulate->m->key_qvel+i*simulate->m->nv, simulate->d->qvel, simulate->m->nv);
        mju_copy(simulate->m->key_act+i*simulate->m->na, simulate->d->act, simulate->m->na);
        mju_copy(simulate->m->key_mpos+i*3*simulate->m->nmocap, simulate->d->mocap_pos, 3*simulate->m->nmocap);
        mju_copy(simulate->m->key_mquat+i*4*simulate->m->nmocap, simulate->d->mocap_quat, 4*simulate->m->nmocap);
        mju_copy(simulate->m->key_ctrl+i*simulate->m->nu, simulate->d->ctrl, simulate->m->nu);
        break;
      }
    }

    // physics section
    else if (it && it->sectionid==SECT_PHYSICS) {
      // update disable flags in mjOption
      simulate->m->opt.disableflags = 0;
      for (i=0; i<mjNDISABLE; i++)
        if (simulate->disable[i]) {
          simulate->m->opt.disableflags |= (1<<i);
        }

      // update enable flags in mjOption
      simulate->m->opt.enableflags = 0;
      for (i=0; i<mjNENABLE; i++)
        if (simulate->enable[i]) {
          simulate->m->opt.enableflags |= (1<<i);
        }
    }

    // rendering section
    else if (it && it->sectionid==SECT_RENDERING) {
      // set camera in mjvCamera
      if (simulate->camera==0) {
        simulate->cam.type = mjCAMERA_FREE;
      } else if (simulate->camera==1) {
        if (simulate->pert.select>0) {
          simulate->cam.type = mjCAMERA_TRACKING;
          simulate->cam.trackbodyid = simulate->pert.select;
          simulate->cam.fixedcamid = -1;
        } else {
          simulate->cam.type = mjCAMERA_FREE;
          simulate->camera = 0;
          mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
        }
      } else {
        simulate->cam.type = mjCAMERA_FIXED;
        simulate->cam.fixedcamid = simulate->camera - 2;
      }
      // copy camera spec to clipboard (as MJCF element)
      if (it->itemid == 3) {
        copycamera(simulate);
      }
    }

    // group section
    else if (it && it->sectionid==SECT_GROUP) {
      // remake joint section if joint group changed
      if (it->name[0]=='J' && it->name[1]=='o') {
        simulate->ui1.nsect = SECT_JOINT;
        makejoint(simulate, simulate->ui1.sect[SECT_JOINT].state);
        simulate->ui1.nsect = NSECT1;
        uiModify(simulate->window, &simulate->ui1, state, &simulate->con);
      }

      // remake control section if actuator group changed
      if (it->name[0]=='A' && it->name[1]=='c') {
        simulate->ui1.nsect = SECT_CONTROL;
        makecontrol(simulate, simulate->ui1.sect[SECT_CONTROL].state);
        simulate->ui1.nsect = NSECT1;
        uiModify(simulate->window, &simulate->ui1, state, &simulate->con);
      }
    }

    // stop if UI processed event
    if (it!=nullptr || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // call UI 1 if event is directed to it
  if ((state->dragrect==simulate->ui1.rectid) ||
      (state->dragrect==0 && state->mouserect==simulate->ui1.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&simulate->ui1, state, &simulate->con);

    // control section
    if (it && it->sectionid==SECT_CONTROL) {
      // clear controls
      if (it->itemid==0) {
        mju_zero(simulate->d->ctrl, simulate->m->nu);
        mjui_update(SECT_CONTROL, -1, &simulate->ui1, &simulate->uistate, &simulate->con);
      }
    }

    // stop if UI processed event
    if (it!=nullptr || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // shortcut not handled by UI
  if (state->type==mjEVENT_KEY && state->key!=0) {
    switch (state->key) {
    case ' ':                   // Mode
      if (simulate->m) {
        simulate->run = 1 - simulate->run;
        simulate->pert.active = 0;
        mjui_update(-1, -1, &simulate->ui0, state, &simulate->con);
      }
      break;

    case mjKEY_RIGHT:           // step forward
      if (simulate->m && !simulate->run) {
        cleartimers(simulate->d);
        mj_step(simulate->m, simulate->d);
        profilerupdate(simulate);
        sensorupdate(simulate);
        updatesettings(simulate);
      }
      break;

    case mjKEY_PAGE_UP:         // select parent body
      if (simulate->m && simulate->pert.select>0) {
        simulate->pert.select = simulate->m->body_parentid[simulate->pert.select];
        simulate->pert.skinselect = -1;

        // stop perturbation if world reached
        if (simulate->pert.select<=0) {
          simulate->pert.active = 0;
        }
      }

      break;

    case ']':                   // cycle up fixed cameras
      if (simulate->m && simulate->m->ncam) {
        simulate->cam.type = mjCAMERA_FIXED;
        // simulate->camera = {0 or 1} are reserved for the free and tracking cameras
        if (simulate->camera < 2 || simulate->camera == 2 + simulate->m->ncam-1) {
          simulate->camera = 2;
        } else {
          simulate->camera += 1;
        }
        simulate->cam.fixedcamid = simulate->camera - 2;
        mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
      }
      break;

    case '[':                   // cycle down fixed cameras
      if (simulate->m && simulate->m->ncam) {
        simulate->cam.type = mjCAMERA_FIXED;
        // settings.camera = {0 or 1} are reserved for the free and tracking cameras
        if (simulate->camera <= 2) {
          simulate->camera = 2 + simulate->m->ncam-1;
        } else {
          simulate->camera -= 1;
        }
        simulate->cam.fixedcamid = simulate->camera - 2;
        mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
      }
      break;

    case mjKEY_F6:                   // cycle frame visualisation
      if (simulate->m) {
        simulate->vopt.frame = (simulate->vopt.frame + 1) % mjNFRAME;
        mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
      }
      break;

    case mjKEY_F7:                   // cycle label visualisation
      if (simulate->m) {
        simulate->vopt.label = (simulate->vopt.label + 1) % mjNLABEL;
        mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
      }
      break;

    case mjKEY_ESCAPE:          // free camera
      simulate->cam.type = mjCAMERA_FREE;
      simulate->camera = 0;
      mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
      break;

    case '-':                   // slow down
      if (simulate->slow_down < max_slow_down && !state->shift) {
        simulate->slow_down *= 2;
        simulate->speed_changed = true;
      }
      break;

    case '=':                   // speed up
      if (simulate->slow_down > 1 && !state->shift) {
        simulate->slow_down /= 2;
        simulate->speed_changed = true;
      }
      break;
    }

    return;
  }

  // 3D scroll
  if (state->type==mjEVENT_SCROLL && state->mouserect==3 && simulate->m) {
    // emulate vertical mouse motion = 2% of window height
    mjv_moveCamera(simulate->m, mjMOUSE_ZOOM, 0, -zoom_increment*state->sy, &simulate->scn, &simulate->cam);

    return;
  }

  // 3D press
  if (state->type==mjEVENT_PRESS && state->mouserect==3 && simulate->m) {
    // set perturbation
    int newperturb = 0;
    if (state->control && simulate->pert.select>0) {
      // right: translate;  left: rotate
      if (state->right) {
        newperturb = mjPERT_TRANSLATE;
      } else if (state->left) {
        newperturb = mjPERT_ROTATE;
      }

      // perturbation onset: reset reference
      if (newperturb && !simulate->pert.active) {
        mjv_initPerturb(simulate->m, simulate->d, &simulate->scn, &simulate->pert);
      }
    }
    simulate->pert.active = newperturb;

    // handle double-click
    if (state->doubleclick) {
      // determine selection mode
      int selmode;
      if (state->button==mjBUTTON_LEFT) {
        selmode = 1;
      } else if (state->control) {
        selmode = 3;
      } else {
        selmode = 2;
      }

      // find geom and 3D click point, get corresponding body
      mjrRect r = state->rect[3];
      mjtNum selpnt[3];
      int selgeom, selskin;
      int selbody = mjv_select(simulate->m, simulate->d, &simulate->vopt,
                               (mjtNum)r.width/(mjtNum)r.height,
                               (mjtNum)(state->x-r.left)/(mjtNum)r.width,
                               (mjtNum)(state->y-r.bottom)/(mjtNum)r.height,
                               &simulate->scn, selpnt, &selgeom, &selskin);

      // set lookat point, start tracking is requested
      if (selmode==2 || selmode==3) {
        // copy selpnt if anything clicked
        if (selbody>=0) {
          mju_copy3(simulate->cam.lookat, selpnt);
        }

        // switch to tracking camera if dynamic body clicked
        if (selmode==3 && selbody>0) {
          // mujoco camera
          simulate->cam.type = mjCAMERA_TRACKING;
          simulate->cam.trackbodyid = selbody;
          simulate->cam.fixedcamid = -1;

          // UI camera
          simulate->camera = 1;
          mjui_update(SECT_RENDERING, -1, &simulate->ui0, &simulate->uistate, &simulate->con);
        }
      }

      // set body selection
      else {
        if (selbody>=0) {
          // record selection
          simulate->pert.select = selbody;
          simulate->pert.skinselect = selskin;

          // compute localpos
          mjtNum tmp[3];
          mju_sub3(tmp, selpnt, simulate->d->xpos+3*simulate->pert.select);
          mju_mulMatTVec(simulate->pert.localpos, simulate->d->xmat+9*simulate->pert.select, tmp, 3, 3);
        } else {
          simulate->pert.select = 0;
          simulate->pert.skinselect = -1;
        }
      }

      // stop perturbation on select
      simulate->pert.active = 0;
    }

    return;
  }

  // 3D release
  if (state->type==mjEVENT_RELEASE && state->dragrect==3 && simulate->m) {
    // stop perturbation
    simulate->pert.active = 0;

    return;
  }

  // 3D move
  if (state->type==mjEVENT_MOVE && state->dragrect==3 && simulate->m) {
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
    if (simulate->pert.active)
      mjv_movePerturb(simulate->m, simulate->d, action, state->dx/r.height, -state->dy/r.height,
                      &simulate->scn, &simulate->pert);
    else
      mjv_moveCamera(simulate->m, action, state->dx/r.height, -state->dy/r.height,
                     &simulate->scn, &simulate->cam);

    return;
  }
}

void uiRender(mjuiState* state) {
  mj::Simulate* simulate = (mj::Simulate*)(state->userdata);
  simulate->render();
}

// drop file callback
void drop(mj::Simulate* simulate, int count, const char** paths) {
  // make sure list is non-empty
  if (count>0) {
    mju::strcpy_arr(simulate->dropfilename, paths[0]);
    simulate->droploadrequest = 1;
  }
}

void uiDrop(mjuiState* state, int count, const char** paths) {
  mj::Simulate* simulate = (mj::Simulate*)(state->userdata);
  drop(simulate, count, paths);
}

} // end unnamed namespace

namespace mujoco {
namespace mju = ::mujoco::sample_util;

//---------------------------------- init -------------------------------------------------

// create object and initialize the simulate ui
Simulate::Simulate(void) {
} // TODO constructor is now empty...

//------------------------ apply pose perturbations ----------------------------
void Simulate::applyposepertubations(int flg_paused) {
  if (this->m != nullptr) {
    mjv_applyPerturbPose(this->m, this->d, &this->pert, flg_paused);  // move mocap bodies only
  }
}

//------------------------ apply force perturbations ---------------------------
void Simulate::applyforceperturbations(void) {
  if (this->m != nullptr) {
    mjv_applyPerturbForce(this->m, this->d, &this->pert);
  }
}

//-------------------- Tell the render thread to load a file and wait ----------
void Simulate::load(const char* file,
                    mjModel* mnew,
                    mjData* dnew,
                    bool delete_old_m_d) {
  this->mnew = mnew;
  this->dnew = dnew;
  this->delete_old_m_d = delete_old_m_d;
  mju::strcpy_arr(this->filename, file);
  this->loadrequest = 2;

  // Wait for the render thread to be done loading
  // so that we know the old model and data's memory can
  // be free'd by the other thread (sometimes python)
  while (this->loadrequest > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

//------------------------ load mjb or xml model -------------------------------
void Simulate::loadmodel(void) {
  if (this->delete_old_m_d) {
    // delete old model if requested
    if (this->d) {
      mj_deleteData(d);
    }
    if (this->m) {
      mj_deleteModel(m);
    }
  }

  this->m = this->mnew;
  this->d = this->dnew;

  // re-create scene and context
  mjv_makeScene(this->m, &this->scn, maxgeom);
  mjr_makeContext(this->m, &this->con, 50*(this->font+1));

  // clear perturbation state
  this->pert.active = 0;
  this->pert.select = 0;
  this->pert.skinselect = -1;

  // align and scale view unless reloading the same file
  if (mju::strcmp_arr(this->filename, this->previous_filename)) {
    alignscale(this);
    mju::strcpy_arr(this->previous_filename, this->filename);
  }

  // update scene
  mjv_updateScene(this->m, this->d, &this->vopt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);

  // set window title to model name
  if (this->window && this->m->names) {
    char title[200] = "this : ";
    mju::strcat_arr(title, this->m->names);
    glfwSetWindowTitle(this->window, title);
  }

  // set keyframe range and divisions
  this->ui0.sect[SECT_SIMULATION].item[5].slider.range[0] = 0;
  this->ui0.sect[SECT_SIMULATION].item[5].slider.range[1] = mjMAX(0, this->m->nkey - 1);
  this->ui0.sect[SECT_SIMULATION].item[5].slider.divisions = mjMAX(1, this->m->nkey - 1);

  // rebuild UI sections
  makesections(this);

  // full ui update
  uiModify(this->window, &this->ui0, &this->uistate, &this->con);
  uiModify(this->window, &this->ui1, &this->uistate, &this->con);
  updatesettings(this);

  // clear request
  this->loadrequest = 0;
}


//---------------------------------- rendering --------------------------------------


// prepare to render
void Simulate::prepare(void) {
  // data for FPS calculation
  static double lastupdatetm = 0;

  // update interval, save update time
  double tmnow = glfwGetTime();
  double interval = tmnow - lastupdatetm;
  interval = mjMIN(1, mjMAX(0.0001, interval));
  lastupdatetm = tmnow;

  // no model: nothing to do
  if (!this->m) {
    return;
  }

  // update scene
  mjv_updateScene(this->m, this->d, &this->vopt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);

  // update watch
  if (this->ui0_enable && this->ui0.sect[SECT_WATCH].state) {
    watch(this);
    mjui_update(SECT_WATCH, -1, &this->ui0, &this->uistate, &this->con);
  }

  // update joint
  if (this->ui1_enable && this->ui1.sect[SECT_JOINT].state) {
    mjui_update(SECT_JOINT, -1, &this->ui1, &this->uistate, &this->con);
  }

  // update info text
  if (this->info) {
    infotext(this, this->info_title, this->info_content, interval);
  }

  // update control
  if (this->ui1_enable && this->ui1.sect[SECT_CONTROL].state ) {
    mjui_update(SECT_CONTROL, -1, &this->ui1, &this->uistate, &this->con);
  }

  // update profiler
  if (this->profiler && this->run) {
    profilerupdate(this);
  }

  // update sensor
  if (this->sensor && this->run) {
    sensorupdate(this);
  }

  // clear timers once profiler info has been copied
  cleartimers(this->d);
}

// render the ui to the window
void Simulate::render(void) {
  // get 3D rectangle and reduced for profiler
  mjrRect rect = this->uistate.rect[3];
  mjrRect smallrect = rect;
  if (this->profiler) {
    smallrect.width = rect.width - rect.width/4;
  }

  // no model
  if (!this->m) {
    // blank screen
    mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

    // label
    if (this->loadrequest) {
      mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect, "loading", nullptr, &this->con);
    } else {
      char intro_message[Simulate::kMaxFilenameLength];
      mju::sprintf_arr(intro_message,
                       "MuJoCo version %s\nDrag-and-drop model file here", mj_versionString());
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, intro_message, 0, &this->con);
    }

    // show last loading error
    if (this->loadError[0]) {
      mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->loadError, 0, &this->con);
    }

    // render uis
    if (this->ui0_enable) {
      mjui_render(&this->ui0, &this->uistate, &this->con);
    }
    if (this->ui1_enable) {
      mjui_render(&this->ui1, &this->uistate, &this->con);
    }

    // finalize
    glfwSwapBuffers(this->window);

    return;
  }

  // render scene
  mjr_render(rect, &this->scn, &this->con);

  // show last loading error
  if (this->loadError[0]) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->loadError, 0, &this->con);
  }

  // show pause/loading label
  if (!this->run || this->loadrequest) {
    mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                this->loadrequest ? "loading" : "pause", nullptr, &this->con);
  }

  // show realtime label
  if (this->run && this->slow_down != 1) {
    std::string realtime_label = "1/" + std::to_string(this->slow_down) + " x";
    mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect, realtime_label.c_str(), nullptr, &this->con);
  }

  // show ui 0
  if (this->ui0_enable) {
    mjui_render(&this->ui0, &this->uistate, &this->con);
  }

  // show ui 1
  if (this->ui1_enable) {
    mjui_render(&this->ui1, &this->uistate, &this->con);
  }

  // show help
  if (this->help) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content, &this->con);
  }

  // show info
  if (this->info) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->info_title, this->info_content, &this->con);
  }

  // show profiler
  if (this->profiler) {
    profilershow(this, rect);
  }

  // show sensor
  if (this->sensor) {
    sensorshow(this, smallrect);
  }

  // finalize
  glfwSwapBuffers(this->window);
}


// clear callbacks registered in external structures
void Simulate::clearcallback(void) {
  uiClearCallback(this->window);
}

void Simulate::renderloop(void) {
  // Set timer callback (milliseconds)
  mjcb_time = timer;

  // multisampling
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_VISIBLE, 1);

  // get videomode and save
  this->vmode = *glfwGetVideoMode(glfwGetPrimaryMonitor());

  // create window
  this->window = glfwCreateWindow((2*this->vmode.width)/3, (2*this->vmode.height)/3,
                            "Simulate", nullptr, nullptr);
  if (!this->window) {
    glfwTerminate();
    mju_error("could not create window");
  }

  // save window position and size
  glfwGetWindowPos(this->window, this->windowpos, this->windowpos+1);
  glfwGetWindowSize(this->window, this->windowsize, this->windowsize+1);

  // make context current, set v-sync
  glfwMakeContextCurrent(this->window);
  glfwSwapInterval(this->vsync);

  // init abstract visualization
  mjv_defaultCamera(&this->cam);
  mjv_defaultOption(&this->vopt);
  profilerinit(this);
  sensorinit(this);

  // make empty scene
  mjv_defaultScene(&this->scn);
  mjv_makeScene(nullptr, &this->scn, maxgeom);

  // select default font
  int fontscale = uiFontScale(this->window);
  this->font = fontscale/50 - 1;

  // make empty context
  mjr_defaultContext(&this->con);
  mjr_makeContext(nullptr, &this->con, fontscale);

  // init state and uis
  std::memset(&this->uistate, 0, sizeof(mjuiState));
  std::memset(&this->ui0, 0, sizeof(mjUI));
  std::memset(&this->ui1, 0, sizeof(mjUI));
  this->ui0.spacing = mjui_themeSpacing(this->spacing);
  this->ui0.color = mjui_themeColor(this->color);
  this->ui0.predicate = uiPredicate;
  this->ui0.rectid = 1;
  this->ui0.auxid = 0;
  this->ui1.spacing = mjui_themeSpacing(this->spacing);
  this->ui1.color = mjui_themeColor(this->color);
  this->ui1.predicate = uiPredicate;
  this->ui1.rectid = 2;
  this->ui1.auxid = 1;

  // set GLFW callbacks
  this->uistate.userdata = (void*)(this);
  uiSetCallback(this->window, &this->uistate, uiEvent, uiLayout, uiRender, uiDrop);

  // populate uis with standard sections
  this->ui0.userdata = (void*)(this);
  this->ui1.userdata = (void*)(this);
  mjui_add(&this->ui0, defFile);
  mjui_add(&this->ui0, this->defOption);
  mjui_add(&this->ui0, this->defSimulation);
  mjui_add(&this->ui0, this->defWatch);
  uiModify(this->window, &this->ui0, &this->uistate, &this->con);
  uiModify(this->window, &this->ui1, &this->uistate, &this->con);

  // run event loop
  while (!glfwWindowShouldClose(this->window) && !this->exitrequest) {
    { // start exclusive access (block simulation thread)
      const std::lock_guard<std::mutex> lock(this->mtx);

      // load model (not on first pass, to show "loading" label)
      if (this->loadrequest==1) {
        {
          this->loadmodel();
        }
      } else if (this->loadrequest>1) {
        this->loadrequest = 1;
      }

      // handle events (calls all callbacks)
      glfwPollEvents();

      // prepare to render
      this->prepare();
    } // end exclusive access (allow simulation thread to run)

    // render while simulation is running
    this->render();
  }

  this->clearcallback();
  mjv_freeScene(&this->scn);
  mjr_freeContext(&this->con);
}

}  // namespace mujoco
