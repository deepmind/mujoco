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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <mujoco/mjxmacro.h>
#include "uitools.h"

#include "array_safety.h"
namespace mju = ::mujoco::sample_util;

#include "portable-file-dialogs.h"

//-------------------------------- global -----------------------------------------------

// Default path for file dialogs.
#define DEFAULT_PATH "."

static constexpr int kBufSize = 1000;

// constants
const int maxgeom = 5000;           // preallocated geom array in mjvScene
const double syncmisalign = 0.1;    // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7;   // fraction of refresh available for simulation
const int max_slow_down = 128;      // maximum slow-down quotient
const double zoom_increment = 0.02; // ratio of single click-wheel zoom increment to vertical extent

// model and data
mjModel* m = NULL;
mjData* d = NULL;


// strings
char filename[kBufSize] = "";
char previous_filename[kBufSize] = "";
char loadError[kBufSize] = "";


// control noise variables
mjtNum* ctrlnoise = nullptr;


// abstract visualization
mjvScene scn;
mjvCamera cam;
mjvOption vopt;
mjvPerturb pert;
mjvFigure figconstraint;
mjvFigure figcost;
mjvFigure figtimer;
mjvFigure figsize;
mjvFigure figsensor;


// OpenGL rendering and UI
GLFWvidmode vmode;
int windowpos[2];
int windowsize[2];
mjrContext con;
GLFWwindow* window = NULL;
mjuiState uistate;
mjUI ui0, ui1;


// UI settings not contained in MuJoCo structures
struct {
  // file
  int exitrequest = 0;

  // option
  int spacing = 0;
  int color = 0;
  int font = 0;
  int ui0 = 1;
  int ui1 = 1;
  int help = 0;
  int info = 0;
  int profiler = 0;
  int sensor = 0;
  int fullscreen = 0;
  int vsync = 1;
  int busywait = 0;

  // simulation
  int run = 1;
  int key = 0;
  int loadrequest = 0;
  int slow_down = 1;
  bool speed_changed = true;
  double ctrlnoisestd = 0.0;
  double ctrlnoiserate = 0.0;

  // watch
  char field[mjMAXUITEXT] = "qpos";
  int index = 0;

  // physics: need sync
  int disable[mjNDISABLE];
  int enable[mjNENABLE];

  // rendering: need sync
  int camera = 0;
} settings;


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
  {mjITEM_SECTION,   "File",          1, NULL,                    "AF"},
  {mjITEM_BUTTON,    "Open",          1, NULL,                    ""},
  {mjITEM_BUTTON,    "Save xml",      2, NULL,                    ""},
  {mjITEM_BUTTON,    "Save mjb",      2, NULL,                    ""},
  {mjITEM_BUTTON,    "Print model",   2, NULL,                    "CM"},
  {mjITEM_BUTTON,    "Print data",    2, NULL,                    "CD"},
  {mjITEM_BUTTON,    "Quit",          1, NULL,                    "CQ"},
  {mjITEM_END}
};


// option section of UI
const mjuiDef defOption[] = {
  {mjITEM_SECTION,   "Option",        1, NULL,                    "AO"},
  {mjITEM_SELECT,    "Spacing",       1, &settings.spacing,       "Tight\nWide"},
  {mjITEM_SELECT,    "Color",         1, &settings.color,         "Default\nOrange\nWhite\nBlack"},
  {mjITEM_SELECT,    "Font",          1, &settings.font,          "50 %\n100 %\n150 %\n200 %\n250 %\n300 %"},
  {mjITEM_CHECKINT,  "Left UI (Tab)", 1, &settings.ui0,           " #258"},
  {mjITEM_CHECKINT,  "Right UI",      1, &settings.ui1,           "S#258"},
  {mjITEM_CHECKINT,  "Help",          2, &settings.help,          " #290"},
  {mjITEM_CHECKINT,  "Info",          2, &settings.info,          " #291"},
  {mjITEM_CHECKINT,  "Profiler",      2, &settings.profiler,      " #292"},
  {mjITEM_CHECKINT,  "Sensor",        2, &settings.sensor,        " #293"},
#ifdef __APPLE__
  {mjITEM_CHECKINT,  "Fullscreen",    0, &settings.fullscreen,    " #294"},
#else
  {mjITEM_CHECKINT,  "Fullscreen",    1, &settings.fullscreen,    " #294"},
#endif
  {mjITEM_CHECKINT,  "Vertical Sync", 1, &settings.vsync,         ""},
  {mjITEM_CHECKINT,  "Busy Wait",     1, &settings.busywait,      ""},
  {mjITEM_END}
};


// simulation section of UI
const mjuiDef defSimulation[] = {
  {mjITEM_SECTION,   "Simulation",    1, NULL,                    "AS"},
  {mjITEM_RADIO,     "",              2, &settings.run,           "Pause\nRun"},
  {mjITEM_BUTTON,    "Reset",         2, NULL,                    " #259"},
  {mjITEM_BUTTON,    "Reload",        2, NULL,                    "CL"},
  {mjITEM_BUTTON,    "Align",         2, NULL,                    "CA"},
  {mjITEM_BUTTON,    "Copy pose",     2, NULL,                    "CC"},
  {mjITEM_SLIDERINT, "Key",           3, &settings.key,           "0 0"},
  {mjITEM_BUTTON,    "Load key",      3},
  {mjITEM_BUTTON,    "Save key",      3},
  {mjITEM_SLIDERNUM, "Noise scale",   2, &settings.ctrlnoisestd,  "0 2"},
  {mjITEM_SLIDERNUM, "Noise rate",    2, &settings.ctrlnoiserate, "0 2"},
  {mjITEM_END}
};


// watch section of UI
const mjuiDef defWatch[] = {
  {mjITEM_SECTION,   "Watch",         0, NULL,                    "AW"},
  {mjITEM_EDITTXT,   "Field",         2, settings.field,          "qpos"},
  {mjITEM_EDITINT,   "Index",         2, &settings.index,         "1"},
  {mjITEM_STATIC,    "Value",         2, NULL,                    " "},
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
  "UI title double-click\n"
  "Q";

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
  "Expand/collapse all\n"
  "Quit";


// info strings
char info_title[kBufSize];
char info_content[kBufSize];



//-------------------------------- profiler, sensor, info, watch -----------------------------------

// init profiler figures
void profilerinit(void) {
  int i, n;

  // set figures to default
  mjv_defaultFigure(&figconstraint);
  mjv_defaultFigure(&figcost);
  mjv_defaultFigure(&figtimer);
  mjv_defaultFigure(&figsize);

  // titles
  mju::strcpy_arr(figconstraint.title, "Counts");
  mju::strcpy_arr(figcost.title, "Convergence (log 10)");
  mju::strcpy_arr(figsize.title, "Dimensions");
  mju::strcpy_arr(figtimer.title, "CPU time (msec)");

  // x-labels
  mju::strcpy_arr(figconstraint.xlabel, "Solver iteration");
  mju::strcpy_arr(figcost.xlabel, "Solver iteration");
  mju::strcpy_arr(figsize.xlabel, "Video frame");
  mju::strcpy_arr(figtimer.xlabel, "Video frame");

  // y-tick nubmer formats
  mju::strcpy_arr(figconstraint.yformat, "%.0f");
  mju::strcpy_arr(figcost.yformat, "%.1f");
  mju::strcpy_arr(figsize.yformat, "%.0f");
  mju::strcpy_arr(figtimer.yformat, "%.2f");

  // colors
  figconstraint.figurergba[0] =   0.1f;
  figcost.figurergba[2] =         0.2f;
  figsize.figurergba[0] =         0.1f;
  figtimer.figurergba[2] =        0.2f;
  figconstraint.figurergba[3] =   0.5f;
  figcost.figurergba[3] =         0.5f;
  figsize.figurergba[3] =         0.5f;
  figtimer.figurergba[3] =        0.5f;

  // legends
  mju::strcpy_arr(figconstraint.linename[0], "total");
  mju::strcpy_arr(figconstraint.linename[1], "active");
  mju::strcpy_arr(figconstraint.linename[2], "changed");
  mju::strcpy_arr(figconstraint.linename[3], "evals");
  mju::strcpy_arr(figconstraint.linename[4], "updates");
  mju::strcpy_arr(figcost.linename[0], "improvement");
  mju::strcpy_arr(figcost.linename[1], "gradient");
  mju::strcpy_arr(figcost.linename[2], "lineslope");
  mju::strcpy_arr(figsize.linename[0], "dof");
  mju::strcpy_arr(figsize.linename[1], "body");
  mju::strcpy_arr(figsize.linename[2], "constraint");
  mju::strcpy_arr(figsize.linename[3], "sqrt(nnz)");
  mju::strcpy_arr(figsize.linename[4], "contact");
  mju::strcpy_arr(figsize.linename[5], "iteration");
  mju::strcpy_arr(figtimer.linename[0], "total");
  mju::strcpy_arr(figtimer.linename[1], "collision");
  mju::strcpy_arr(figtimer.linename[2], "prepare");
  mju::strcpy_arr(figtimer.linename[3], "solve");
  mju::strcpy_arr(figtimer.linename[4], "other");

  // grid sizes
  figconstraint.gridsize[0] = 5;
  figconstraint.gridsize[1] = 5;
  figcost.gridsize[0] = 5;
  figcost.gridsize[1] = 5;
  figsize.gridsize[0] = 3;
  figsize.gridsize[1] = 5;
  figtimer.gridsize[0] = 3;
  figtimer.gridsize[1] = 5;

  // minimum ranges
  figconstraint.range[0][0] = 0;
  figconstraint.range[0][1] = 20;
  figconstraint.range[1][0] = 0;
  figconstraint.range[1][1] = 80;
  figcost.range[0][0] = 0;
  figcost.range[0][1] = 20;
  figcost.range[1][0] = -15;
  figcost.range[1][1] = 5;
  figsize.range[0][0] = -200;
  figsize.range[0][1] = 0;
  figsize.range[1][0] = 0;
  figsize.range[1][1] = 100;
  figtimer.range[0][0] = -200;
  figtimer.range[0][1] = 0;
  figtimer.range[1][0] = 0;
  figtimer.range[1][1] = 0.4f;

  // init x axis on history figures (do not show yet)
  for (n=0; n<6; n++)
    for (i=0; i<mjMAXLINEPNT; i++) {
      figtimer.linedata[n][2*i] = (float)-i;
      figsize.linedata[n][2*i] = (float)-i;
    }
}



// update profiler figures
void profilerupdate(void) {
  int i, n;

  // update constraint figure
  figconstraint.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
  for (i=1; i<5; i++) {
    figconstraint.linepnt[i] = figconstraint.linepnt[0];
  }
  if (m->opt.solver==mjSOL_PGS) {
    figconstraint.linepnt[3] = 0;
    figconstraint.linepnt[4] = 0;
  }
  if (m->opt.solver==mjSOL_CG) {
    figconstraint.linepnt[4] = 0;
  }
  for (i=0; i<figconstraint.linepnt[0]; i++) {
    // x
    figconstraint.linedata[0][2*i] = (float)i;
    figconstraint.linedata[1][2*i] = (float)i;
    figconstraint.linedata[2][2*i] = (float)i;
    figconstraint.linedata[3][2*i] = (float)i;
    figconstraint.linedata[4][2*i] = (float)i;

    // y
    figconstraint.linedata[0][2*i+1] = (float)d->nefc;
    figconstraint.linedata[1][2*i+1] = (float)d->solver[i].nactive;
    figconstraint.linedata[2][2*i+1] = (float)d->solver[i].nchange;
    figconstraint.linedata[3][2*i+1] = (float)d->solver[i].neval;
    figconstraint.linedata[4][2*i+1] = (float)d->solver[i].nupdate;
  }

  // update cost figure
  figcost.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
  for (i=1; i<3; i++) {
    figcost.linepnt[i] = figcost.linepnt[0];
  }
  if (m->opt.solver==mjSOL_PGS) {
    figcost.linepnt[1] = 0;
    figcost.linepnt[2] = 0;
  }

  for (i=0; i<figcost.linepnt[0]; i++) {
    // x
    figcost.linedata[0][2*i] = (float)i;
    figcost.linedata[1][2*i] = (float)i;
    figcost.linedata[2][2*i] = (float)i;

    // y
    figcost.linedata[0][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].improvement));
    figcost.linedata[1][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].gradient));
    figcost.linedata[2][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].lineslope));
  }

  // get timers: total, collision, prepare, solve, other
  mjtNum total = d->timer[mjTIMER_STEP].duration;
  int number = d->timer[mjTIMER_STEP].number;
  if (!number) {
    total = d->timer[mjTIMER_FORWARD].duration;
    number = d->timer[mjTIMER_FORWARD].number;
  }
  number = mjMAX(1, number);
  float tdata[5] = {
    (float)(total/number),
    (float)(d->timer[mjTIMER_POS_COLLISION].duration/number),
    (float)(d->timer[mjTIMER_POS_MAKE].duration/number) +
    (float)(d->timer[mjTIMER_POS_PROJECT].duration/number),
    (float)(d->timer[mjTIMER_CONSTRAINT].duration/number),
    0
  };
  tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

  // update figtimer
  int pnt = mjMIN(201, figtimer.linepnt[0]+1);
  for (n=0; n<5; n++) {
    // shift data
    for (i=pnt-1; i>0; i--) {
      figtimer.linedata[n][2*i+1] = figtimer.linedata[n][2*i-1];
    }

    // assign new
    figtimer.linepnt[n] = pnt;
    figtimer.linedata[n][1] = tdata[n];
  }

  // get sizes: nv, nbody, nefc, sqrt(nnz), ncont, iter
  float sdata[6] = {
    (float)m->nv,
    (float)m->nbody,
    (float)d->nefc,
    (float)mju_sqrt((mjtNum)d->solver_nnz),
    (float)d->ncon,
    (float)d->solver_iter
  };

  // update figsize
  pnt = mjMIN(201, figsize.linepnt[0]+1);
  for (n=0; n<6; n++) {
    // shift data
    for (i=pnt-1; i>0; i--) {
      figsize.linedata[n][2*i+1] = figsize.linedata[n][2*i-1];
    }

    // assign new
    figsize.linepnt[n] = pnt;
    figsize.linedata[n][1] = sdata[n];
  }
}



// show profiler figures
void profilershow(mjrRect rect) {
  mjrRect viewport = {
    rect.left + rect.width - rect.width/4,
    rect.bottom,
    rect.width/4,
    rect.height/4
  };
  mjr_figure(viewport, &figtimer, &con);
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &figsize, &con);
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &figcost, &con);
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &figconstraint, &con);
}



// init sensor figure
void sensorinit(void) {
  // set figure to default
  mjv_defaultFigure(&figsensor);
  figsensor.figurergba[3] = 0.5f;

  // set flags
  figsensor.flg_extend = 1;
  figsensor.flg_barplot = 1;
  figsensor.flg_symmetric = 1;

  // title
  mju::strcpy_arr(figsensor.title, "Sensor data");

  // y-tick nubmer format
  mju::strcpy_arr(figsensor.yformat, "%.0f");

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
void sensorupdate(void) {
  static const int maxline = 10;

  // clear linepnt
  for (int i=0; i<maxline; i++) {
    figsensor.linepnt[i] = 0;
  }

  // start with line 0
  int lineid = 0;

  // loop over sensors
  for (int n=0; n<m->nsensor; n++) {
    // go to next line if type is different
    if (n>0 && m->sensor_type[n]!=m->sensor_type[n-1]) {
      lineid = mjMIN(lineid+1, maxline-1);
    }

    // get info about this sensor
    mjtNum cutoff = (m->sensor_cutoff[n]>0 ? m->sensor_cutoff[n] : 1);
    int adr = m->sensor_adr[n];
    int dim = m->sensor_dim[n];

    // data pointer in line
    int p = figsensor.linepnt[lineid];

    // fill in data for this sensor
    for (int i=0; i<dim; i++) {
      // check size
      if ((p+2*i)>=mjMAXLINEPNT/2) {
        break;
      }

      // x
      figsensor.linedata[lineid][2*p+4*i] = (float)(adr+i);
      figsensor.linedata[lineid][2*p+4*i+2] = (float)(adr+i);

      // y
      figsensor.linedata[lineid][2*p+4*i+1] = 0;
      figsensor.linedata[lineid][2*p+4*i+3] = (float)(d->sensordata[adr+i]/cutoff);
    }

    // update linepnt
    figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT-1,
                                      figsensor.linepnt[lineid]+2*dim);
  }
}



// show sensor figure
void sensorshow(mjrRect rect) {
  // constant width with and without profiler
  int width = settings.profiler ? rect.width/3 : rect.width/4;

  // render figure on the right
  mjrRect viewport = {
    rect.left + rect.width - width,
    rect.bottom,
    width,
    rect.height/3
  };
  mjr_figure(viewport, &figsensor, &con);
}



// prepare info text
void infotext(char (&title)[kBufSize], char (&content)[kBufSize], double interval) {
  char tmp[20];

  // compute solver error
  mjtNum solerr = 0;
  if (d->solver_iter) {
    int ind = mjMIN(d->solver_iter-1, mjNSOLVER-1);
    solerr = mju_min(d->solver[ind].improvement, d->solver[ind].gradient);
    if (solerr==0) {
      solerr = mju_max(d->solver[ind].improvement, d->solver[ind].gradient);
    }
  }
  solerr = mju_log10(mju_max(mjMINVAL, solerr));

  // prepare info text
  const std::string realtime_nominator = settings.slow_down == 1 ? "" : "1/";
  mju::strcpy_arr(title, "Time\nSize\nCPU\nSolver   \nFPS\nstack\nconbuf\nefcbuf");
  mju::sprintf_arr(content,
                   "%-9.3f %s%d x\n%d  (%d con)\n%.3f\n%.1f  (%d it)\n%.0f\n%.3f\n%.3f\n%.3f",
                   d->time, realtime_nominator.c_str(), settings.slow_down,
                   d->nefc, d->ncon,
                   settings.run ?
                   d->timer[mjTIMER_STEP].duration / mjMAX(1, d->timer[mjTIMER_STEP].number) :
                   d->timer[mjTIMER_FORWARD].duration / mjMAX(1, d->timer[mjTIMER_FORWARD].number),
                   solerr, d->solver_iter,
                   1/interval,
                   d->maxuse_stack/(double)d->nstack,
                   d->maxuse_con/(double)m->nconmax,
                   d->maxuse_efc/(double)m->njmax);

  // add Energy if enabled
  if (mjENABLED(mjENBL_ENERGY)) {
    mju::sprintf_arr(tmp, "\n%.3f", d->energy[0]+d->energy[1]);
    mju::strcat_arr(content, tmp);
    mju::strcat_arr(title, "\nEnergy");
  }

  // add FwdInv if enabled
  if (mjENABLED(mjENBL_FWDINV)) {
    mju::sprintf_arr(tmp, "\n%.1f %.1f",
                     mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[0])),
                     mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[1])));
    mju::strcat_arr(content, tmp);
    mju::strcat_arr(title, "\nFwdInv");
  }
}



// sprintf forwarding, to avoid compiler warning in x-macro
void printfield(char (&str)[mjMAXUINAME], void* ptr) {
  mju::sprintf_arr(str, "%g", *(mjtNum*)ptr);
}



// update watch
void watch(void) {
  // clear
  ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
  mju::strcpy_arr(ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

  // prepare symbols needed by xmacro
  MJDATA_POINTERS_PREAMBLE(m);

  // find specified field in mjData arrays, update value
  #define X(TYPE, NAME, NR, NC)                                                             \
    if (!mju::strcmp_arr(#NAME, settings.field) &&                                          \
        !mju::strcmp_arr(#TYPE, "mjtNum")) {                                                \
      if (settings.index>=0 && settings.index<m->NR*NC) {                                   \
        printfield(ui0.sect[SECT_WATCH].item[2].multi.name[0], d->NAME + settings.index);   \
      } else {                                                                              \
        mju::strcpy_arr(ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid index");       \
      }                                                                                     \
      return;                                                                               \
    }

    MJDATA_POINTERS
  #undef X
}



//---------------------------------- UI construction -----------------------------------------------

// make physics section of UI
void makephysics(int oldstate) {
  int i;

  mjuiDef defPhysics[] = {
    {mjITEM_SECTION,   "Physics",       oldstate, NULL,                 "AP"},
    {mjITEM_SELECT,    "Integrator",    2, &(m->opt.integrator),        "Euler\nRK4\nimplicit"},
    {mjITEM_SELECT,    "Collision",     2, &(m->opt.collision),         "All\nPair\nDynamic"},
    {mjITEM_SELECT,    "Cone",          2, &(m->opt.cone),              "Pyramidal\nElliptic"},
    {mjITEM_SELECT,    "Jacobian",      2, &(m->opt.jacobian),          "Dense\nSparse\nAuto"},
    {mjITEM_SELECT,    "Solver",        2, &(m->opt.solver),            "PGS\nCG\nNewton"},
    {mjITEM_SEPARATOR, "Algorithmic Parameters", 1},
    {mjITEM_EDITNUM,   "Timestep",      2, &(m->opt.timestep),          "1 0 1"},
    {mjITEM_EDITINT,   "Iterations",    2, &(m->opt.iterations),        "1 0 1000"},
    {mjITEM_EDITNUM,   "Tolerance",     2, &(m->opt.tolerance),         "1 0 1"},
    {mjITEM_EDITINT,   "Noslip Iter",   2, &(m->opt.noslip_iterations), "1 0 1000"},
    {mjITEM_EDITNUM,   "Noslip Tol",    2, &(m->opt.noslip_tolerance),  "1 0 1"},
    {mjITEM_EDITINT,   "MRR Iter",      2, &(m->opt.mpr_iterations),    "1 0 1000"},
    {mjITEM_EDITNUM,   "MPR Tol",       2, &(m->opt.mpr_tolerance),     "1 0 1"},
    {mjITEM_EDITNUM,   "API Rate",      2, &(m->opt.apirate),           "1 0 1000"},
    {mjITEM_SEPARATOR, "Physical Parameters", 1},
    {mjITEM_EDITNUM,   "Gravity",       2, m->opt.gravity,              "3"},
    {mjITEM_EDITNUM,   "Wind",          2, m->opt.wind,                 "3"},
    {mjITEM_EDITNUM,   "Magnetic",      2, m->opt.magnetic,             "3"},
    {mjITEM_EDITNUM,   "Density",       2, &(m->opt.density),           "1"},
    {mjITEM_EDITNUM,   "Viscosity",     2, &(m->opt.viscosity),         "1"},
    {mjITEM_EDITNUM,   "Imp Ratio",     2, &(m->opt.impratio),          "1"},
    {mjITEM_SEPARATOR, "Disable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defEnableFlags[] = {
    {mjITEM_SEPARATOR, "Enable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defOverride[] = {
    {mjITEM_SEPARATOR, "Contact Override", 1},
    {mjITEM_EDITNUM,   "Margin",        2, &(m->opt.o_margin),          "1"},
    {mjITEM_EDITNUM,   "Sol Imp",       2, &(m->opt.o_solimp),          "5"},
    {mjITEM_EDITNUM,   "Sol Ref",       2, &(m->opt.o_solref),          "2"},
    {mjITEM_END}
  };

  // add physics
  mjui_add(&ui0, defPhysics);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKINT,  "", 2, NULL, ""},
    {mjITEM_END}
  };
  for (i=0; i<mjNDISABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjDISABLESTRING[i]);
    defFlag[0].pdata = settings.disable + i;
    mjui_add(&ui0, defFlag);
  }
  mjui_add(&ui0, defEnableFlags);
  for (i=0; i<mjNENABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjENABLESTRING[i]);
    defFlag[0].pdata = settings.enable + i;
    mjui_add(&ui0, defFlag);
  }

  // add contact override
  mjui_add(&ui0, defOverride);
}



// make rendering section of UI
void makerendering(int oldstate) {
  int i, j;

  mjuiDef defRendering[] = {
    {
      mjITEM_SECTION,
      "Rendering",
      oldstate,
      NULL,
      "AR"
    },
    {
      mjITEM_SELECT,
      "Camera",
      2,
      &(settings.camera),
      "Free\nTracking"
    },
    {
      mjITEM_SELECT,
      "Label",
      2,
      &(vopt.label),
      "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\n"
      "Actuator\nConstraint\nSkin\nSelection\nSel Pnt\nForce"
    },
    {
      mjITEM_SELECT,
      "Frame",
      2,
      &(vopt.frame),
      "None\nBody\nGeom\nSite\nCamera\nLight\nContact\nWorld"
    },
    {
      mjITEM_BUTTON,
      "Copy camera",
      2,
      NULL,
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
  for (i=0; i<mjMIN(m->ncam, mjMAXUIMULTI-2); i++) {
    // prepare name
    char camname[mjMAXUITEXT] = "\n";
    if (m->names[m->name_camadr[i]]) {
      mju::strcat_arr(camname, m->names+m->name_camadr[i]);
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
  mjui_add(&ui0, defRendering);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKBYTE,  "", 2, NULL, ""},
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
    defFlag[0].pdata = vopt.flags + i;
    mjui_add(&ui0, defFlag);
  }
  mjui_add(&ui0, defOpenGL);
  for (i=0; i<mjNRNDFLAG; i++) {
    mju::strcpy_arr(defFlag[0].name, mjRNDSTRING[i][0]);
    mju::sprintf_arr(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
    defFlag[0].pdata = scn.flags + i;
    mjui_add(&ui0, defFlag);
  }
}



// make group section of UI
void makegroup(int oldstate) {
  mjuiDef defGroup[] = {
    {mjITEM_SECTION,    "Group enable",     oldstate, NULL,             "AG"},
    {mjITEM_SEPARATOR,  "Geom groups",  1},
    {mjITEM_CHECKBYTE,  "Geom 0",           2, vopt.geomgroup,          " 0"},
    {mjITEM_CHECKBYTE,  "Geom 1",           2, vopt.geomgroup+1,        " 1"},
    {mjITEM_CHECKBYTE,  "Geom 2",           2, vopt.geomgroup+2,        " 2"},
    {mjITEM_CHECKBYTE,  "Geom 3",           2, vopt.geomgroup+3,        " 3"},
    {mjITEM_CHECKBYTE,  "Geom 4",           2, vopt.geomgroup+4,        " 4"},
    {mjITEM_CHECKBYTE,  "Geom 5",           2, vopt.geomgroup+5,        " 5"},
    {mjITEM_SEPARATOR,  "Site groups",  1},
    {mjITEM_CHECKBYTE,  "Site 0",           2, vopt.sitegroup,          "S0"},
    {mjITEM_CHECKBYTE,  "Site 1",           2, vopt.sitegroup+1,        "S1"},
    {mjITEM_CHECKBYTE,  "Site 2",           2, vopt.sitegroup+2,        "S2"},
    {mjITEM_CHECKBYTE,  "Site 3",           2, vopt.sitegroup+3,        "S3"},
    {mjITEM_CHECKBYTE,  "Site 4",           2, vopt.sitegroup+4,        "S4"},
    {mjITEM_CHECKBYTE,  "Site 5",           2, vopt.sitegroup+5,        "S5"},
    {mjITEM_SEPARATOR,  "Joint groups", 1},
    {mjITEM_CHECKBYTE,  "Joint 0",          2, vopt.jointgroup,         ""},
    {mjITEM_CHECKBYTE,  "Joint 1",          2, vopt.jointgroup+1,       ""},
    {mjITEM_CHECKBYTE,  "Joint 2",          2, vopt.jointgroup+2,       ""},
    {mjITEM_CHECKBYTE,  "Joint 3",          2, vopt.jointgroup+3,       ""},
    {mjITEM_CHECKBYTE,  "Joint 4",          2, vopt.jointgroup+4,       ""},
    {mjITEM_CHECKBYTE,  "Joint 5",          2, vopt.jointgroup+5,       ""},
    {mjITEM_SEPARATOR,  "Tendon groups",    1},
    {mjITEM_CHECKBYTE,  "Tendon 0",         2, vopt.tendongroup,        ""},
    {mjITEM_CHECKBYTE,  "Tendon 1",         2, vopt.tendongroup+1,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 2",         2, vopt.tendongroup+2,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 3",         2, vopt.tendongroup+3,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 4",         2, vopt.tendongroup+4,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 5",         2, vopt.tendongroup+5,      ""},
    {mjITEM_SEPARATOR,  "Actuator groups", 1},
    {mjITEM_CHECKBYTE,  "Actuator 0",       2, vopt.actuatorgroup,      ""},
    {mjITEM_CHECKBYTE,  "Actuator 1",       2, vopt.actuatorgroup+1,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 2",       2, vopt.actuatorgroup+2,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 3",       2, vopt.actuatorgroup+3,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 4",       2, vopt.actuatorgroup+4,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 5",       2, vopt.actuatorgroup+5,    ""},
    {mjITEM_SEPARATOR,  "Skin groups", 1},
    {mjITEM_CHECKBYTE,  "Skin 0",           2, vopt.skingroup,          ""},
    {mjITEM_CHECKBYTE,  "Skin 1",           2, vopt.skingroup+1,        ""},
    {mjITEM_CHECKBYTE,  "Skin 2",           2, vopt.skingroup+2,        ""},
    {mjITEM_CHECKBYTE,  "Skin 3",           2, vopt.skingroup+3,        ""},
    {mjITEM_CHECKBYTE,  "Skin 4",           2, vopt.skingroup+4,        ""},
    {mjITEM_CHECKBYTE,  "Skin 5",           2, vopt.skingroup+5,        ""},
    {mjITEM_END}
  };

  // add section
  mjui_add(&ui0, defGroup);
}



// make joint section of UI
void makejoint(int oldstate) {
  int i;

  mjuiDef defJoint[] = {
    {mjITEM_SECTION, "Joint", oldstate, NULL, "AJ"},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&ui1, defJoint);
  defSlider[0].state = 4;

  // add scalar joints, exit if UI limit reached
  int itemcnt = 0;
  for (i=0; i<m->njnt && itemcnt<mjMAXUIITEM; i++)
    if ((m->jnt_type[i]==mjJNT_HINGE || m->jnt_type[i]==mjJNT_SLIDE)) {
      // skip if joint group is disabled
      if (!vopt.jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->jnt_group[i]))]) {
        continue;
      }

      // set data and name
      defSlider[0].pdata = d->qpos + m->jnt_qposadr[i];
      if (m->names[m->name_jntadr[i]]) {
        mju::strcpy_arr(defSlider[0].name, m->names+m->name_jntadr[i]);
      } else {
        mju::sprintf_arr(defSlider[0].name, "joint %d", i);
      }

      // set range
      if (m->jnt_limited[i])
        mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                         m->jnt_range[2*i], m->jnt_range[2*i+1]);
      else if (m->jnt_type[i]==mjJNT_SLIDE) {
        mju::strcpy_arr(defSlider[0].other, "-1 1");
      } else {
        mju::strcpy_arr(defSlider[0].other, "-3.1416 3.1416");
      }

      // add and count
      mjui_add(&ui1, defSlider);
      itemcnt++;
    }
}



// make control section of UI
void makecontrol(int oldstate) {
  int i;

  mjuiDef defControl[] = {
    {mjITEM_SECTION, "Control", oldstate, NULL, "AC"},
    {mjITEM_BUTTON,  "Clear all", 2},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&ui1, defControl);
  defSlider[0].state = 2;

  // add controls, exit if UI limit reached (Clear button already added)
  int itemcnt = 1;
  for (i=0; i<m->nu && itemcnt<mjMAXUIITEM; i++) {
    // skip if actuator group is disabled
    if (!vopt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->actuator_group[i]))]) {
      continue;
    }

    // set data and name
    defSlider[0].pdata = d->ctrl + i;
    if (m->names[m->name_actuatoradr[i]]) {
      mju::strcpy_arr(defSlider[0].name, m->names+m->name_actuatoradr[i]);
    } else {
      mju::sprintf_arr(defSlider[0].name, "control %d", i);
    }

    // set range
    if (m->actuator_ctrllimited[i])
      mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                       m->actuator_ctrlrange[2*i], m->actuator_ctrlrange[2*i+1]);
    else {
      mju::strcpy_arr(defSlider[0].other, "-1 1");
    }

    // add and count
    mjui_add(&ui1, defSlider);
    itemcnt++;
  }
}



// make model-dependent UI sections
void makesections(void) {
  int i;

  // get section open-close state, UI 0
  int oldstate0[NSECT0];
  for (i=0; i<NSECT0; i++) {
    oldstate0[i] = 0;
    if (ui0.nsect>i) {
      oldstate0[i] = ui0.sect[i].state;
    }
  }

  // get section open-close state, UI 1
  int oldstate1[NSECT1];
  for (i=0; i<NSECT1; i++) {
    oldstate1[i] = 0;
    if (ui1.nsect>i) {
      oldstate1[i] = ui1.sect[i].state;
    }
  }

  // clear model-dependent sections of UI
  ui0.nsect = SECT_PHYSICS;
  ui1.nsect = 0;

  // make
  makephysics(oldstate0[SECT_PHYSICS]);
  makerendering(oldstate0[SECT_RENDERING]);
  makegroup(oldstate0[SECT_GROUP]);
  makejoint(oldstate1[SECT_JOINT]);
  makecontrol(oldstate1[SECT_CONTROL]);
}



//---------------------------------- utility functions ---------------------------------------------

// align and scale view
void alignscale(void) {
  // autoscale
  cam.lookat[0] = m->stat.center[0];
  cam.lookat[1] = m->stat.center[1];
  cam.lookat[2] = m->stat.center[2];
  cam.distance = 1.5 * m->stat.extent;

  // set to free camera
  cam.type = mjCAMERA_FREE;
}



// copy qpos to clipboard as key
void copykey(void) {
  char clipboard[5000] = "<key qpos='";
  char buf[200];

  // prepare string
  for (int i=0; i<m->nq; i++) {
    mju::sprintf_arr(buf, i==m->nq-1 ? "%g" : "%g ", d->qpos[i]);
    mju::strcat_arr(clipboard, buf);
  }
  mju::strcat_arr(clipboard, "'/>");

  // copy to clipboard
  glfwSetClipboardString(window, clipboard);
}



// millisecond timer, for MuJoCo built-in profiler
mjtNum timer(void) {
  return (mjtNum)(1000*glfwGetTime());
}



// clear all times
void cleartimers(void) {
  for (int i=0; i<mjNTIMER; i++) {
    d->timer[i].duration = 0;
    d->timer[i].number = 0;
  }
}



// copy current camera to clipboard as MJCF specification
void copycamera(mjvGLCamera* camera) {
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
  glfwSetClipboardString(window, clipboard);
}



// update UI 0 when MuJoCo structures change (except for joint sliders)
void updatesettings(void) {
  int i;

  // physics flags
  for (i=0; i<mjNDISABLE; i++) {
    settings.disable[i] = ((m->opt.disableflags & (1<<i)) !=0);
  }
  for (i=0; i<mjNENABLE; i++) {
    settings.enable[i] = ((m->opt.enableflags & (1<<i)) !=0);
  }

  // camera
  if (cam.type==mjCAMERA_FIXED) {
    settings.camera = 2 + cam.fixedcamid;
  } else if (cam.type==mjCAMERA_TRACKING) {
    settings.camera = 1;
  } else {
    settings.camera = 0;
  }

  // update UI
  mjui_update(-1, -1, &ui0, &uistate, &con);
}



// drop file callback
void drop(GLFWwindow* window, int count, const char** paths) {
  // make sure list is non-empty
  if (count>0) {
    mju::strcpy_arr(filename, paths[0]);
    settings.loadrequest = 1;
  }
}



// load mjb or xml model
void loadmodel(void) {
  // clear request
  settings.loadrequest = 0;

  // make sure filename is not empty
  if (!filename[0]) {
    return;
  }

  // load and compile
  loadError[0] = '\0';
  mjModel* mnew = 0;
  if (mju::strlen_arr(filename)>4 &&
      !std::strncmp(filename+mju::strlen_arr(filename)-4, ".mjb",
                    mju::sizeof_arr(filename)-mju::strlen_arr(filename)+4)) {
    mnew = mj_loadModel(filename, NULL);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, NULL, loadError, kBufSize);
    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length-1] == '\n') {
        loadError[error_length-1] = '\0';
      }
    }
  }
  if (!mnew) {
    std::printf("%s\n", loadError);
    return;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    settings.run = 0;
  }

  // delete old model, assign new
  mj_deleteData(d);
  mj_deleteModel(m);
  m = mnew;
  d = mj_makeData(m);
  mj_forward(m, d);

  // allocate ctrlnoise
  free(ctrlnoise);
  ctrlnoise = (mjtNum*) malloc(sizeof(mjtNum)*m->nu);
  mju_zero(ctrlnoise, m->nu);

  // re-create scene and context
  mjv_makeScene(m, &scn, maxgeom);
  mjr_makeContext(m, &con, 50*(settings.font+1));

  // clear perturbation state
  pert.active = 0;
  pert.select = 0;
  pert.skinselect = -1;

  // align and scale view unless reloading the same file
  if (mju::strcmp_arr(filename, previous_filename)) {
    alignscale();
    mju::strcpy_arr(previous_filename, filename);
  }

  // update scene
  mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

  // set window title to model name
  if (window && m->names) {
    char title[200] = "Simulate : ";
    mju::strcat_arr(title, m->names);
    glfwSetWindowTitle(window, title);
  }

  // set keyframe range and divisions
  ui0.sect[SECT_SIMULATION].item[5].slider.range[0] = 0;
  ui0.sect[SECT_SIMULATION].item[5].slider.range[1] = mjMAX(0, m->nkey - 1);
  ui0.sect[SECT_SIMULATION].item[5].slider.divisions = mjMAX(1, m->nkey - 1);

  // rebuild UI sections
  makesections();

  // full ui update
  uiModify(window, &ui0, &uistate, &con);
  uiModify(window, &ui1, &uistate, &con);
  updatesettings();
}



//---------------------------------- UI hooks (for uitools.c) --------------------------------------

// determine enable/disable item state given category
int uiPredicate(int category, void* userdata) {
  switch (category) {
  case 2:                 // require model
    return (m!=NULL);

  case 3:                 // require model and nkey
    return (m && m->nkey);

  case 4:                 // require model and paused
    return (m && !settings.run);

  default:
    return 1;
  }
}



// set window layout
void uiLayout(mjuiState* state) {
  mjrRect* rect = state->rect;

  // set number of rectangles
  state->nrect = 4;

  // rect 0: entire framebuffer
  rect[0].left = 0;
  rect[0].bottom = 0;
  glfwGetFramebufferSize(window, &rect[0].width, &rect[0].height);

  // rect 1: UI 0
  rect[1].left = 0;
  rect[1].width = settings.ui0 ? ui0.width : 0;
  rect[1].bottom = 0;
  rect[1].height = rect[0].height;

  // rect 2: UI 1
  rect[2].width = settings.ui1 ? ui1.width : 0;
  rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
  rect[2].bottom = 0;
  rect[2].height = rect[0].height;

  // rect 3: 3D plot (everything else is an overlay)
  rect[3].left = rect[1].width;
  rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
  rect[3].bottom = 0;
  rect[3].height = rect[0].height;
}



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



// handle UI event
void uiEvent(mjuiState* state) {
  int i;
  char err[200];

  // call UI 0 if event is directed to it
  if ((state->dragrect==ui0.rectid) ||
      (state->dragrect==0 && state->mouserect==ui0.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&ui0, state, &con);

    // file section
    if (it && it->sectionid==SECT_FILE) {
      switch (it->itemid) {
        // Open XML/MJB
        case 0: {
          auto fileName = pfd::open_file("Open XML/MJB", DEFAULT_PATH, { "All Files", "*", "XML Files", "*.xml", "MJB Files", "*.mjb"}, pfd::opt::none).result();
          if (!fileName.empty()) {
            mju::strcpy_arr(filename, fileName[0].c_str());
            loadmodel();
          }
          break;
        }
        // Save XML
        case 1: {
          auto fileName = pfd::save_file("Save XML", DEFAULT_PATH, { "All Files", "*", "XML Files", "*.xml" }, pfd::opt::none).result();
          if (!fileName.empty()) {
            if(fileName.substr(fileName.find_last_of(".") + 1) != "xml")
              fileName += ".xml";
            if (!mj_saveLastXML(fileName.c_str(), m, err, 200))
              std::printf("Save XML error: %s", err);
          }
          break;
        }
        // Save MJB
        case 2: {
          auto fileName = pfd::save_file("Save MJB", DEFAULT_PATH, { "All Files", "*", "MJB Files", "*.mjb" }, pfd::opt::none).result();
          if (!fileName.empty()) {
            if(fileName.substr(fileName.find_last_of(".") + 1) != "mjb")
              fileName += ".mjb";
            mj_saveModel(m, fileName.c_str(), NULL, 0);
          }
          break;
        }
        // Print model
        case 3: {
          auto fileName = pfd::save_file("Print Model", DEFAULT_PATH, { "All Files", "*", "Text Files", "*.txt" }, pfd::opt::none).result();
          if (!fileName.empty()) {
            if(fileName.substr(fileName.find_last_of(".") + 1) != "txt")
              fileName += ".txt";
            mj_printModel(m, fileName.c_str());
          }
          break;
        }
        // Print data
        case 4: {
          auto fileName = pfd::save_file("Print Data", DEFAULT_PATH, { "All Files", "*", "Text Files", "*.txt" }, pfd::opt::none).result();
          if (!fileName.empty()) {
            if(fileName.substr(fileName.find_last_of(".") + 1) != "txt")
              fileName += ".txt";
            mj_printData(m, d, fileName.c_str());
          }
          break;
        }
        // Quit
        case 5: {
          settings.exitrequest = 1;
          break;
        }
      }
    }

    // option section
    else if (it && it->sectionid==SECT_OPTION) {
      switch (it->itemid) {
      case 0:             // Spacing
        ui0.spacing = mjui_themeSpacing(settings.spacing);
        ui1.spacing = mjui_themeSpacing(settings.spacing);
        break;

      case 1:             // Color
        ui0.color = mjui_themeColor(settings.color);
        ui1.color = mjui_themeColor(settings.color);
        break;

      case 2:             // Font
        mjr_changeFont(50*(settings.font+1), &con);
        break;

      case 9:             // Full screen
        if (glfwGetWindowMonitor(window)) {
          // restore window from saved data
          glfwSetWindowMonitor(window, NULL, windowpos[0], windowpos[1],
                               windowsize[0], windowsize[1], 0);
        }

        // currently windowed: switch to full screen
        else {
          // save window data
          glfwGetWindowPos(window, windowpos, windowpos+1);
          glfwGetWindowSize(window, windowsize, windowsize+1);

          // switch
          glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0,
                               vmode.width, vmode.height, vmode.refreshRate);
        }

        // reinstante vsync, just in case
        glfwSwapInterval(settings.vsync);
        break;

      case 10:            // Vertical sync
        glfwSwapInterval(settings.vsync);
        break;
      }

      // modify UI
      uiModify(window, &ui0, state, &con);
      uiModify(window, &ui1, state, &con);
    }

    // simulation section
    else if (it && it->sectionid==SECT_SIMULATION) {
      switch (it->itemid) {
      case 1:             // Reset
        if (m) {
          mj_resetData(m, d);
          mj_forward(m, d);
          profilerupdate();
          sensorupdate();
          updatesettings();
        }
        break;

      case 2:             // Reload
        settings.loadrequest = 1;
        break;

      case 3:             // Align
        alignscale();
        updatesettings();
        break;

      case 4:             // Copy pose
        copykey();
        break;

      case 5:             // Adjust key
      case 6:             // Load key
        i = settings.key;
        d->time = m->key_time[i];
        mju_copy(d->qpos, m->key_qpos+i*m->nq, m->nq);
        mju_copy(d->qvel, m->key_qvel+i*m->nv, m->nv);
        mju_copy(d->act, m->key_act+i*m->na, m->na);
        mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
        mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
        mju_copy(d->ctrl, m->key_ctrl+i*m->nu, m->nu);
        mj_forward(m, d);
        profilerupdate();
        sensorupdate();
        updatesettings();
        break;

      case 7:             // Save key
        i = settings.key;
        m->key_time[i] = d->time;
        mju_copy(m->key_qpos+i*m->nq, d->qpos, m->nq);
        mju_copy(m->key_qvel+i*m->nv, d->qvel, m->nv);
        mju_copy(m->key_act+i*m->na, d->act, m->na);
        mju_copy(m->key_mpos+i*3*m->nmocap, d->mocap_pos, 3*m->nmocap);
        mju_copy(m->key_mquat+i*4*m->nmocap, d->mocap_quat, 4*m->nmocap);
        mju_copy(m->key_ctrl+i*m->nu, d->ctrl, m->nu);
        break;
      }
    }

    // physics section
    else if (it && it->sectionid==SECT_PHYSICS) {
      // update disable flags in mjOption
      m->opt.disableflags = 0;
      for (i=0; i<mjNDISABLE; i++)
        if (settings.disable[i]) {
          m->opt.disableflags |= (1<<i);
        }

      // update enable flags in mjOption
      m->opt.enableflags = 0;
      for (i=0; i<mjNENABLE; i++)
        if (settings.enable[i]) {
          m->opt.enableflags |= (1<<i);
        }
    }

    // rendering section
    else if (it && it->sectionid==SECT_RENDERING) {
      // set camera in mjvCamera
      if (settings.camera==0) {
        cam.type = mjCAMERA_FREE;
      } else if (settings.camera==1) {
        if (pert.select>0) {
          cam.type = mjCAMERA_TRACKING;
          cam.trackbodyid = pert.select;
          cam.fixedcamid = -1;
        } else {
          cam.type = mjCAMERA_FREE;
          settings.camera = 0;
          mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
        }
      } else {
        cam.type = mjCAMERA_FIXED;
        cam.fixedcamid = settings.camera - 2;
      }
      // copy camera spec to clipboard (as MJCF element)
      if (it->itemid == 3) {
        copycamera(scn.camera);
      }
    }

    // group section
    else if (it && it->sectionid==SECT_GROUP) {
      // remake joint section if joint group changed
      if (it->name[0]=='J' && it->name[1]=='o') {
        ui1.nsect = SECT_JOINT;
        makejoint(ui1.sect[SECT_JOINT].state);
        ui1.nsect = NSECT1;
        uiModify(window, &ui1, state, &con);
      }

      // remake control section if actuator group changed
      if (it->name[0]=='A' && it->name[1]=='c') {
        ui1.nsect = SECT_CONTROL;
        makecontrol(ui1.sect[SECT_CONTROL].state);
        ui1.nsect = NSECT1;
        uiModify(window, &ui1, state, &con);
      }
    }

    // stop if UI processed event
    if (it!=NULL || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // call UI 1 if event is directed to it
  if ((state->dragrect==ui1.rectid) ||
      (state->dragrect==0 && state->mouserect==ui1.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&ui1, state, &con);

    // control section
    if (it && it->sectionid==SECT_CONTROL) {
      // clear controls
      if (it->itemid==0) {
        mju_zero(d->ctrl, m->nu);
        mjui_update(SECT_CONTROL, -1, &ui1, &uistate, &con);
      }
    }

    // stop if UI processed event
    if (it!=NULL || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // shortcut not handled by UI
  if (state->type==mjEVENT_KEY && state->key!=0) {
    switch (state->key) {
    case ' ':                   // Mode
      if (m) {
        settings.run = 1 - settings.run;
        pert.active = 0;
        mjui_update(-1, -1, &ui0, state, &con);
      }
      break;

    case mjKEY_RIGHT:           // step forward
      if (m && !settings.run) {
        cleartimers();
        mj_step(m, d);
        profilerupdate();
        sensorupdate();
        updatesettings();
      }
      break;

    case mjKEY_PAGE_UP:         // select parent body
      if (m && pert.select>0) {
        pert.select = m->body_parentid[pert.select];
        pert.skinselect = -1;

        // stop perturbation if world reached
        if (pert.select<=0) {
          pert.active = 0;
        }
      }

      break;

    case ']':                   // cycle up fixed cameras
      if (m && m->ncam) {
        cam.type = mjCAMERA_FIXED;
        // settings.camera = {0 or 1} are reserved for the free and tracking cameras
        if (settings.camera < 2 || settings.camera == 2 + m->ncam-1) {
          settings.camera = 2;
        } else {
          settings.camera += 1;
        }
        cam.fixedcamid = settings.camera - 2;
        mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
      }
      break;

    case '[':                   // cycle down fixed cameras
      if (m && m->ncam) {
        cam.type = mjCAMERA_FIXED;
        // settings.camera = {0 or 1} are reserved for the free and tracking cameras
        if (settings.camera <= 2) {
          settings.camera = 2 + m->ncam-1;
        } else {
          settings.camera -= 1;
        }
        cam.fixedcamid = settings.camera - 2;
        mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
      }
      break;

    case mjKEY_F6:                   // cycle frame visualisation
      if (m) {
        vopt.frame = (vopt.frame + 1) % mjNFRAME;
        mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
      }
      break;

    case mjKEY_F7:                   // cycle label visualisation
      if (m) {
        vopt.label = (vopt.label + 1) % mjNLABEL;
        mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
      }
      break;

    case mjKEY_ESCAPE:          // free camera
      cam.type = mjCAMERA_FREE;
      settings.camera = 0;
      mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
      break;

    case '-':                   // slow down
      if (settings.slow_down < max_slow_down && !state->shift) {
        settings.slow_down *= 2;
        settings.speed_changed = true;
      }
      break;

    case '=':                   // speed up
      if (settings.slow_down > 1 && !state->shift) {
        settings.slow_down /= 2;
        settings.speed_changed = true;
      }
      break;

    case 'Q':                   // quit
      settings.exitrequest = 1;
      break;
    }

    return;
  }

  // 3D scroll
  if (state->type==mjEVENT_SCROLL && state->mouserect==3 && m) {
    // emulate vertical mouse motion = 2% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -zoom_increment*state->sy, &scn, &cam);

    return;
  }

  // 3D press
  if (state->type==mjEVENT_PRESS && state->mouserect==3 && m) {
    // set perturbation
    int newperturb = 0;
    if (state->control && pert.select>0) {
      // right: translate;  left: rotate
      if (state->right) {
        newperturb = mjPERT_TRANSLATE;
      } else if (state->left) {
        newperturb = mjPERT_ROTATE;
      }

      // perturbation onset: reset reference
      if (newperturb && !pert.active) {
        mjv_initPerturb(m, d, &scn, &pert);
      }
    }
    pert.active = newperturb;

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
      int selbody = mjv_select(m, d, &vopt,
                               (mjtNum)r.width/(mjtNum)r.height,
                               (mjtNum)(state->x-r.left)/(mjtNum)r.width,
                               (mjtNum)(state->y-r.bottom)/(mjtNum)r.height,
                               &scn, selpnt, &selgeom, &selskin);

      // set lookat point, start tracking is requested
      if (selmode==2 || selmode==3) {
        // copy selpnt if anything clicked
        if (selbody>=0) {
          mju_copy3(cam.lookat, selpnt);
        }

        // switch to tracking camera if dynamic body clicked
        if (selmode==3 && selbody>0) {
          // mujoco camera
          cam.type = mjCAMERA_TRACKING;
          cam.trackbodyid = selbody;
          cam.fixedcamid = -1;

          // UI camera
          settings.camera = 1;
          mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
        }
      }

      // set body selection
      else {
        if (selbody>=0) {
          // record selection
          pert.select = selbody;
          pert.skinselect = selskin;

          // compute localpos
          mjtNum tmp[3];
          mju_sub3(tmp, selpnt, d->xpos+3*pert.select);
          mju_mulMatTVec(pert.localpos, d->xmat+9*pert.select, tmp, 3, 3);
        } else {
          pert.select = 0;
          pert.skinselect = -1;
        }
      }

      // stop perturbation on select
      pert.active = 0;
    }

    return;
  }

  // 3D release
  if (state->type==mjEVENT_RELEASE && state->dragrect==3 && m) {
    // stop perturbation
    pert.active = 0;

    return;
  }

  // 3D move
  if (state->type==mjEVENT_MOVE && state->dragrect==3 && m) {
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
    if (pert.active)
      mjv_movePerturb(m, d, action, state->dx/r.height, -state->dy/r.height,
                      &scn, &pert);
    else
      mjv_moveCamera(m, action, state->dx/r.height, -state->dy/r.height,
                     &scn, &cam);

    return;
  }
}



//---------------------------------- rendering and simulation --------------------------------------

// sim thread synchronization
std::mutex mtx;


// prepare to render
void prepare(void) {
  // data for FPS calculation
  static double lastupdatetm = 0;

  // update interval, save update time
  double tmnow = glfwGetTime();
  double interval = tmnow - lastupdatetm;
  interval = mjMIN(1, mjMAX(0.0001, interval));
  lastupdatetm = tmnow;

  // no model: nothing to do
  if (!m) {
    return;
  }

  // update scene
  mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

  // update watch
  if (settings.ui0 && ui0.sect[SECT_WATCH].state) {
    watch();
    mjui_update(SECT_WATCH, -1, &ui0, &uistate, &con);
  }

  // update joint
  if (settings.ui1 && ui1.sect[SECT_JOINT].state) {
    mjui_update(SECT_JOINT, -1, &ui1, &uistate, &con);
  }

  // update info text
  if (settings.info) {
    infotext(info_title, info_content, interval);
  }

  // update control
  if( settings.ui1 && ui1.sect[SECT_CONTROL].state ) {
    mjui_update(SECT_CONTROL, -1, &ui1, &uistate, &con);
  }

  // update profiler
  if (settings.profiler && settings.run) {
    profilerupdate();
  }

  // update sensor
  if (settings.sensor && settings.run) {
    sensorupdate();
  }

  // clear timers once profiler info has been copied
  cleartimers();
}



// render im main thread (while simulating in background thread)
void render(GLFWwindow* window) {
  // get 3D rectangle and reduced for profiler
  mjrRect rect = uistate.rect[3];
  mjrRect smallrect = rect;
  if (settings.profiler) {
    smallrect.width = rect.width - rect.width/4;
  }

  // no model
  if (!m) {
    // blank screen
    mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

    // label
    if (settings.loadrequest) {
      mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect, "loading", NULL, &con);
    } else {
      char intro_message[kBufSize];
      mju::sprintf_arr(intro_message,
                       "MuJoCo version %s\nDrag-and-drop model file here or click Open", mj_versionString());
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, intro_message, 0, &con);
    }

    // show last loading error
    if (loadError[0]) {
      mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, loadError, 0, &con);
    }

    // render uis
    if (settings.ui0) {
      mjui_render(&ui0, &uistate, &con);
    }
    if (settings.ui1) {
      mjui_render(&ui1, &uistate, &con);
    }

    // finalize
    glfwSwapBuffers(window);

    return;
  }

  // render scene
  mjr_render(rect, &scn, &con);

  // show last loading error
  if (loadError[0]) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, loadError, 0, &con);
  }

  // show pause/loading label
  if (!settings.run || settings.loadrequest) {
    mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                settings.loadrequest ? "loading" : "pause", NULL, &con);
  }

  // show realtime label
  if (settings.run && settings.slow_down != 1) {
    std::string realtime_label = "1/" + std::to_string(settings.slow_down) + " x";
    mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect, realtime_label.c_str(), NULL, &con);
  }

  // show ui 0
  if (settings.ui0) {
    mjui_render(&ui0, &uistate, &con);
  }

  // show ui 1
  if (settings.ui1) {
    mjui_render(&ui1, &uistate, &con);
  }

  // show help
  if (settings.help) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content, &con);
  }

  // show info
  if (settings.info) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, info_title, info_content, &con);
  }

  // show profiler
  if (settings.profiler) {
    profilershow(rect);
  }

  // show sensor
  if (settings.sensor) {
    sensorshow(smallrect);
  }

  // finalize
  glfwSwapBuffers(window);
}



// simulate in background thread (while rendering in main thread)
void simulate(void) {
  // cpu-sim syncronization point
  double cpusync = 0;
  mjtNum simsync = 0;

  // run until asked to exit
  while (!settings.exitrequest) {
    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (settings.run && settings.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // start exclusive access
    mtx.lock();

    // run only if model is present
    if (m) {
      // running
      if (settings.run) {
        // record cpu time at start of iteration
        double tmstart = glfwGetTime();

        // inject noise
        if (settings.ctrlnoisestd) {
          // convert rate and scale to discrete time given current timestep
          mjtNum rate = mju_exp(-m->opt.timestep / settings.ctrlnoiserate);
          mjtNum scale = settings.ctrlnoisestd * mju_sqrt(1-rate*rate);

          for (int i=0; i<m->nu; i++) {
            // update noise
            ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);
            // apply noise
            d->ctrl[i] = ctrlnoise[i];
          }
        }

        // out-of-sync (for any reason)
        mjtNum offset = mju_abs((d->time*settings.slow_down-simsync)-(tmstart-cpusync));
        if( d->time*settings.slow_down<simsync || tmstart<cpusync || cpusync==0 ||
            offset > syncmisalign*settings.slow_down || settings.speed_changed) {
          // re-sync
          cpusync = tmstart;
          simsync = d->time*settings.slow_down;
          settings.speed_changed = false;

          // clear old perturbations, apply new
          mju_zero(d->xfrc_applied, 6*m->nbody);
          mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
          mjv_applyPerturbForce(m, d, &pert);

          // run single step, let next iteration deal with timing
          mj_step(m, d);
        }

        // in-sync
        else {
          // step while simtime lags behind cputime, and within safefactor
          while ((d->time*settings.slow_down-simsync) < (glfwGetTime()-cpusync) &&
                 (glfwGetTime()-tmstart) < refreshfactor/vmode.refreshRate) {
            // clear old perturbations, apply new
            mju_zero(d->xfrc_applied, 6*m->nbody);
            mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
            mjv_applyPerturbForce(m, d, &pert);

            // run mj_step
            mjtNum prevtm = d->time*settings.slow_down;
            mj_step(m, d);

            // break on reset
            if (d->time*settings.slow_down<prevtm) {
              break;
            }
          }
        }
      }

      // paused
      else {
        // apply pose perturbation
        mjv_applyPerturbPose(m, d, &pert, 1);      // move mocap and dynamic bodies

        // run mj_forward, to update rendering and joint sliders
        mj_forward(m, d);
      }
    }

    // end exclusive access
    mtx.unlock();
  }
}



//---------------------------------- init and main -------------------------------------------------

// initalize everything
void init(void) {
  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER!=mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // init GLFW, set timer callback (milliseconds)
  if (!glfwInit()) {
    mju_error("could not initialize GLFW");
  }
  mjcb_time = timer;

  // multisampling
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_VISIBLE, 1);

  // get videomode and save
  vmode = *glfwGetVideoMode(glfwGetPrimaryMonitor());

  // create window
  window = glfwCreateWindow((2*vmode.width)/3, (2*vmode.height)/3,
                            "Simulate", NULL, NULL);
  if (!window) {
    glfwTerminate();
    mju_error("could not create window");
  }

  // save window position and size
  glfwGetWindowPos(window, windowpos, windowpos+1);
  glfwGetWindowSize(window, windowsize, windowsize+1);

  // make context current, set v-sync
  glfwMakeContextCurrent(window);
  glfwSwapInterval(settings.vsync);

  // init abstract visualization
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&vopt);
  profilerinit();
  sensorinit();

  // make empty scene
  mjv_defaultScene(&scn);
  mjv_makeScene(NULL, &scn, maxgeom);

  // select default font
  int fontscale = uiFontScale(window);
  settings.font = fontscale/50 - 1;

  // make empty context
  mjr_defaultContext(&con);
  mjr_makeContext(NULL, &con, fontscale);

  // set GLFW callbacks
  uiSetCallback(window, &uistate, uiEvent, uiLayout);
  glfwSetWindowRefreshCallback(window, render);
  glfwSetDropCallback(window, drop);

  // init state and uis
  std::memset(&uistate, 0, sizeof(mjuiState));
  std::memset(&ui0, 0, sizeof(mjUI));
  std::memset(&ui1, 0, sizeof(mjUI));
  ui0.spacing = mjui_themeSpacing(settings.spacing);
  ui0.color = mjui_themeColor(settings.color);
  ui0.predicate = uiPredicate;
  ui0.rectid = 1;
  ui0.auxid = 0;
  ui1.spacing = mjui_themeSpacing(settings.spacing);
  ui1.color = mjui_themeColor(settings.color);
  ui1.predicate = uiPredicate;
  ui1.rectid = 2;
  ui1.auxid = 1;

  // populate uis with standard sections
  mjui_add(&ui0, defFile);
  mjui_add(&ui0, defOption);
  mjui_add(&ui0, defSimulation);
  mjui_add(&ui0, defWatch);
  uiModify(window, &ui0, &uistate, &con);
  uiModify(window, &ui1, &uistate, &con);
}



// run event loop
int main(int argc, const char** argv) {
  // initialize everything
  init();

  // request loadmodel if file given (otherwise drag-and-drop)
  if (argc>1) {
    mju::strcpy_arr(filename, argv[1]);
    settings.loadrequest = 2;
  }

  // start simulation thread
  std::thread simthread(simulate);

  // event loop
  while (!glfwWindowShouldClose(window) && !settings.exitrequest) {
    // start exclusive access (block simulation thread)
    mtx.lock();

    // load model (not on first pass, to show "loading" label)
    if (settings.loadrequest==1) {
      loadmodel();
    } else if (settings.loadrequest>1) {
      settings.loadrequest = 1;
    }

    // handle events (calls all callbacks)
    glfwPollEvents();

    // prepare to render
    prepare();

    // end exclusive access (allow simulation thread to run)
    mtx.unlock();

    // render while simulation is running
    render(window);
  }

  // stop simulation thread
  settings.exitrequest = 1;
  simthread.join();

  // delete everything we allocated
  uiClearCallback(window);
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 0;
}
