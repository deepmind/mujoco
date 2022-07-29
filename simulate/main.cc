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

#include <mujoco/mujoco.h>
#include <mujoco/mjxmacro.h>
#include "simulate.h"

#include "array_safety.h"

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncmisalign = 0.1;    // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7;   // fraction of refresh available for simulation

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

// control noise variables
mjtNum* ctrlnoise = nullptr;

//---------------------------------- simulation --------------------------------------


mjModel* LoadModel(const char* file, mj::Simulate& simulate) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[mj::Simulate::kMaxFilenameLength] = "";
  mjModel* mnew = 0;
  if (mju::strlen_arr(filename)>4 &&
      !std::strncmp(filename+mju::strlen_arr(filename)-4, ".mjb",
                    mju::sizeof_arr(filename)-mju::strlen_arr(filename)+4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, mj::Simulate::kMaxFilenameLength);
    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length-1] == '\n') {
        loadError[error_length-1] = '\0';
      }
    }
  }

  mju::strcpy_arr(simulate.loadError, loadError);

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    simulate.run = 0;
  }

  return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& simulate) {
  // cpu-sim syncronization point
  double cpusync = 0;
  mjtNum simsync = 0;

  // run until asked to exit
  while (!simulate.exitrequest) {

    if (simulate.droploadrequest) {
      mjModel* mnew = LoadModel(simulate.dropfilename, simulate);
      simulate.droploadrequest = 0;
      if (mnew) {
        mjData* dnew = mj_makeData(mnew);
        simulate.load(simulate.dropfilename, mnew, dnew, true);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = (mjtNum*) malloc(sizeof(mjtNum)*m->nu);
        mju_zero(ctrlnoise, m->nu);
      }
    }

    if (simulate.uiloadrequest) {
      mjModel* mnew = LoadModel(simulate.filename, simulate);
      simulate.uiloadrequest = 0;
      if (mnew) {
        mjData* dnew = mj_makeData(mnew);
        simulate.load(simulate.filename, mnew, dnew, true);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = (mjtNum*) malloc(sizeof(mjtNum)*m->nu);
        mju_zero(ctrlnoise, m->nu);
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (simulate.run && simulate.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    { // start exclusive access
      const std::lock_guard<std::mutex> lock(simulate.mtx);

      // run only if model is present
      if (m) {
        // running
        if (simulate.run) {
          // record cpu time at start of iteration
          double tmstart = glfwGetTime();

          // inject noise
          if (simulate.ctrlnoisestd) {
            // convert rate and scale to discrete time given current timestep
            mjtNum rate = mju_exp(-m->opt.timestep / simulate.ctrlnoiserate);
            mjtNum scale = simulate.ctrlnoisestd * mju_sqrt(1-rate*rate);

            for (int i=0; i<m->nu; i++) {
              // update noise
              ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);
              // apply noise
              d->ctrl[i] = ctrlnoise[i];
            }
          }

          // out-of-sync (for any reason)
          mjtNum offset = mju_abs((d->time*simulate.slow_down-simsync)-(tmstart-cpusync));
          if( d->time*simulate.slow_down<simsync || tmstart<cpusync || cpusync==0 ||
              offset > syncmisalign*simulate.slow_down || simulate.speed_changed) {
            // re-sync
            cpusync = tmstart;
            simsync = d->time*simulate.slow_down;
            simulate.speed_changed = false;

            // clear old perturbations, apply new
            mju_zero(d->xfrc_applied, 6*m->nbody);
            simulate.applyposepertubations(0); // move mocap bodies only
            simulate.applyforceperturbations();

            // run single step, let next iteration deal with timing
            mj_step(m, d);
          }

          // in-sync
          else {
            // step while simtime lags behind cputime, and within safefactor
            while ((d->time*simulate.slow_down-simsync) < (glfwGetTime()-cpusync) &&
                   (glfwGetTime()-tmstart) < refreshfactor/simulate.vmode.refreshRate) {
              // clear old perturbations, apply new
              mju_zero(d->xfrc_applied, 6*m->nbody);
              simulate.applyposepertubations(0); // move mocap bodies only
              simulate.applyforceperturbations();

              // run mj_step
              mjtNum prevtm = d->time*simulate.slow_down;
              mj_step(m, d);

              // break on reset
              if (d->time*simulate.slow_down<prevtm) {
                break;
              }
            }
          }
        }

        // paused
        else {
          // apply pose perturbation
          simulate.applyposepertubations(1); // move mocap and dynamic bodies

          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
        }
      }
    } // end exclusive access
  }
}
} // end unnamed namespace

//---------------------------------- physics_thread ---------------------------------------
void PhysicsThread(mj::Simulate* simulate, const char* filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    m = LoadModel(filename, *simulate);
    if (m) {
      d = mj_makeData(m);
      simulate->load(filename, m, d, true);
      mj_forward(m, d);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = (mjtNum*) malloc(sizeof(mjtNum)*m->nu);
      mju_zero(ctrlnoise, m->nu);
    }
  }

  PhysicsLoop(*simulate);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);
}

//---------------------------------- main -------------------------------------------------

// run event loop
int main(int argc, const char** argv) {
  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER!=mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // simulate object encapsulates the UI
  mj::Simulate simulate;

  // init GLFW
  if (!glfwInit()) {
    mju_error("could not initialize GLFW");
  }

  const char* filename = nullptr;
  if (argc >  1) {
    filename = argv[1];
  }

  // start physics thread
  std::thread physicsthreadhandle = std::thread(&PhysicsThread, &simulate, filename);

  // start simulation UI loop (blocking call)
  simulate.renderloop();
  physicsthreadhandle.join();

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 0;
}
