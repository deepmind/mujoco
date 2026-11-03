// Copyright 2022 DeepMind Technologies Limited
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
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

#include <mujoco/mujoco.h>
#include "errors.h"
#include "raw.h"
#include "structs.h"
#include "threadpool.h"
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mujoco::python {

namespace {

namespace py = ::pybind11;

using PyCArray = py::array_t<mjtNum, py::array::c_style>;

// NOLINTBEGIN(whitespace/line_length)

const auto rollout_init_doc = R"(
Construct a rollout object containing a thread pool for parallel rollouts.

  input arguments (optional):
    nthread            integer, number of threads in pool
                       if zero, this pool is not started and rollouts run on the calling thread
)";

const auto rollout_doc = R"(
Roll out batch of trajectories from initial states, get resulting states and sensor values.

  input arguments (required):
    model              list of homogenous MjModel instances of length nbatch
    data               list of compatible MjData instances of length nthread
    nstep              integer, number of steps to be taken for each trajectory
    control_spec       specification of controls, ncontrol = mj_stateSize(m, control_spec)
    state0             (nbatch x nstate) nbatch initial state arrays, where
                           nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS)
  input arguments (optional):
    warmstart0         (nbatch x nv)                  nbatch qacc_warmstart arrays
    control            (nbatch x nstep x ncontrol)    nbatch trajectories of nstep controls
  output arguments (optional):
    state              (nbatch x nstep x nstate)      nbatch nstep states
    sensordata         (nbatch x nstep x nsendordata) nbatch trajectories of nstep sensordata arrays
    chunk_size         integer, positive threadpool chunk size. If unspecified, the default is
                           chunk_size = max(1, nbatch / (nthread * 10))
)";

// C-style rollout function, assumes all arguments are valid
// all input fields of d are initialised, contents at call time do not matter
// after returning, d will contain the last step of the last rollout
void _unsafe_rollout(std::vector<const mjModel*>& m, mjData* d, int start_roll,
                     int end_roll, int nstep, unsigned int control_spec,
                     const mjtNum* state0, const mjtNum* warmstart0,
                     const mjtNum* control, mjtNum* state, mjtNum* sensordata) {
  // sizes
  size_t nstate = static_cast<size_t>(mj_stateSize(m[0], mjSTATE_FULLPHYSICS));
  size_t ncontrol = static_cast<size_t>(mj_stateSize(m[0], control_spec));
  size_t nv = static_cast<size_t>(m[0]->nv);
  int nbody = m[0]->nbody, neq = m[0]->neq;
  size_t nsensordata = static_cast<size_t>(m[0]->nsensordata);

  // clear user inputs if unspecified
  if (!(control_spec & mjSTATE_CTRL)) {
    mju_zero(d->ctrl, m[0]->nu);
  }
  if (!(control_spec & mjSTATE_QFRC_APPLIED)) {
    mju_zero(d->qfrc_applied, nv);
  }
  if (!(control_spec & mjSTATE_XFRC_APPLIED)) {
    mju_zero(d->xfrc_applied, 6*nbody);
  }

  // loop over rollouts
  for (size_t r = start_roll; r < end_roll; r++) {
    // clear user inputs if unspecified
    if (!(control_spec & mjSTATE_MOCAP_POS)) {
      for (int i = 0; i < nbody; i++) {
        int id = m[r]->body_mocapid[i];
        if (id >= 0) mju_copy3(d->mocap_pos+3*id, m[r]->body_pos+3*i);
      }
    }
    if (!(control_spec & mjSTATE_MOCAP_QUAT)) {
      for (int i = 0; i < nbody; i++) {
        int id = m[r]->body_mocapid[i];
        if (id >= 0) mju_copy4(d->mocap_quat+4*id, m[r]->body_quat+4*i);
      }
    }
    if (!(control_spec & mjSTATE_EQ_ACTIVE)) {
      for (int i = 0; i < neq; i++) {
        d->eq_active[i] = m[r]->eq_active0[i];
      }
    }

    // set initial state
    mj_setState(m[r], d, state0 + r*nstate, mjSTATE_FULLPHYSICS);

    // set warmstart accelerations
    if (warmstart0) {
      mju_copy(d->qacc_warmstart, warmstart0 + r*nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }

    // clear warning counters
    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    // roll out trajectory
    for (size_t t = 0; t < nstep; t++) {
      // check for warnings
      bool nwarning = false;
      for (int i = 0; i < mjNWARNING; i++) {
        if (d->warning[i].number) {
          nwarning = true;
          break;
        }
      }

      // if any warnings, fill remaining outputs with current outputs, break
      if (nwarning) {
        for (; t < nstep; t++) {
          size_t step = r*static_cast<size_t>(nstep) + t;
          if (state) {
            mj_getState(m[r], d, state + step*nstate, mjSTATE_FULLPHYSICS);
          }
          if (sensordata) {
            mju_copy(sensordata + step*nsensordata, d->sensordata, nsensordata);
          }
        }
        break;
      }

      size_t step = r*static_cast<size_t>(nstep) + t;

      // controls
      if (control) {
        mj_setState(m[r], d, control + step*ncontrol, control_spec);
      }

      // step
      mj_step(m[r], d);

      // copy out new state
      if (state) {
        mj_getState(m[r], d, state + step*nstate, mjSTATE_FULLPHYSICS);
      }

      // copy out sensor values
      if (sensordata) {
        mju_copy(sensordata + step*nsensordata, d->sensordata, nsensordata);
      }
    }
  }
}

// C-style threaded version of _unsafe_rollout
void _unsafe_rollout_threaded(std::vector<const mjModel*>& m,
                              std::vector<mjData*>& d, int nbatch, int nstep,
                              unsigned int control_spec, const mjtNum* state0,
                              const mjtNum* warmstart0, const mjtNum* control,
                              mjtNum* state, mjtNum* sensordata,
                              ThreadPool* pool, int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  // Reset the pool counter
  pool->ResetCount();

  // schedule all jobs of full (chunk) size
  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &m, &d](void) {
      int id = pool->WorkerId();
      _unsafe_rollout(m, d[id], j*chunk_size, (j+1)*chunk_size,
        nstep, control_spec, state0, warmstart0, control, state, sensordata);
    };
    pool->Schedule(task);
  }

  // schedule any remaining jobs of size < chunk_size
  if (chunk_remainder > 0) {
    auto task = [=, &m, &d](void) {
      _unsafe_rollout(m, d[pool->WorkerId()], nfulljobs*chunk_size,
        nfulljobs*chunk_size+chunk_remainder,
        nstep, control_spec, state0, warmstart0, control, state, sensordata);
    };
    pool->Schedule(task);
  }

  // wait for counter to increment up to number of jobs submitted by this thread
  pool->WaitCount(njobs);
}

// NOLINTEND(whitespace/line_length)

// check size of optional argument to rollout(), return raw pointer
mjtNum* get_array_ptr(std::optional<const py::array_t<mjtNum>> arg,
                      const char* name, int nbatch, int nstep, int dim) {
  // if empty return nullptr
  if (!arg.has_value()) {
    return nullptr;
  }

  // get info
  py::buffer_info info = arg->request();

  // check size
  size_t expected_size = static_cast<size_t>(nbatch) *
                         static_cast<size_t>(nstep) * static_cast<size_t>(dim);
  if (info.size != expected_size) {
    std::ostringstream msg;
    msg << name << ".size should be " << expected_size << ", got " << info.size;
    throw py::value_error(msg.str());
  }
  return static_cast<mjtNum*>(info.ptr);
}

class Rollout {
 public:
  Rollout(int nthread) : nthread_(nthread) {
    if (this->nthread_ > 0) {
      this->pool_ = std::make_shared<ThreadPool>(this->nthread_);
    }
  }

  void rollout(py::list m, py::list d, int nstep, unsigned int control_spec,
               const PyCArray state0, std::optional<const PyCArray> warmstart0,
               std::optional<const PyCArray> control,
               std::optional<const PyCArray> state,
               std::optional<const PyCArray> sensordata,
               std::optional<int> chunk_size) {
    // get raw pointers
    int nbatch = state0.shape(0);
    std::vector<const raw::MjModel*> model_ptrs(nbatch);
    for (int r = 0; r < nbatch; r++) {
      model_ptrs[r] = m[r].cast<const MjModelWrapper*>()->get();
    }

    // check length d and nthread are consistent
    if (py::len(d) == 0) {
      std::ostringstream msg;
      msg << "The list of data instances is empty";
      throw py::value_error(msg.str());
    } else if (this->nthread_ == 0 && py::len(d) > 1) {
      std::ostringstream msg;
      msg << "More than one data instance passed but "
          << "rollout is configured to run on main thread";
      throw py::value_error(msg.str());
    } else if (this->nthread_ > 0 && this->nthread_ != py::len(d)) {
      std::ostringstream msg;
      msg << "Length of data: " << py::len(d)
          << " not equal to nthread: " << this->nthread_;
      throw py::value_error(msg.str());
    }

    std::vector<raw::MjData*> data_ptrs(py::len(d));
    for (int t = 0; t < py::len(d); t++) {
      data_ptrs[t] = d[t].cast<MjDataWrapper*>()->get();
      // rollout threads across simulations; the engine's pool threads within one
      if (data_ptrs[t]->threadpool) {
        std::ostringstream msg;
        msg << "data[" << t << "] has a thread pool installed (mju_threadpool); "
            << "rollout parallelizes across simulations and cannot use it";
        throw py::value_error(msg.str());
      }
    }

    // the thread pool chunks the batch by this size
    if (chunk_size.has_value() && *chunk_size < 1) {
      throw py::value_error("chunk_size must be positive");
    }

    // check that some steps need to be taken, return if not
    if (nstep < 1) {
      return;
    }

    // get sizes
    int nstate = mj_stateSize(model_ptrs[0], mjSTATE_FULLPHYSICS);
    int ncontrol = mj_stateSize(model_ptrs[0], control_spec);

    mjtNum* state0_ptr = get_array_ptr(state0, "state0", nbatch, 1, nstate);
    mjtNum* warmstart0_ptr =
        get_array_ptr(warmstart0, "warmstart0", nbatch, 1, model_ptrs[0]->nv);
    mjtNum* control_ptr =
        get_array_ptr(control, "control", nbatch, nstep, ncontrol);
    mjtNum* state_ptr = get_array_ptr(state, "state", nbatch, nstep, nstate);
    mjtNum* sensordata_ptr = get_array_ptr(sensordata, "sensordata", nbatch,
                                           nstep, model_ptrs[0]->nsensordata);

    // perform rollouts
    {
      // release the GIL
      py::gil_scoped_release no_gil;

      // call unsafe rollout function, multi or single threaded
      if (this->nthread_ > 0 && nbatch > 1) {
        int chunk_size_final = 1;
        if (!chunk_size.has_value()) {
          chunk_size_final = std::max(1, nbatch / (10 * this->nthread_));
        } else {
          chunk_size_final = *chunk_size;
        }
        InterceptMjErrors(_unsafe_rollout_threaded)(
            model_ptrs, data_ptrs, nbatch, nstep, control_spec, state0_ptr,
            warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr,
            this->pool_.get(), chunk_size_final);
      } else {
        InterceptMjErrors(_unsafe_rollout)(
            model_ptrs, data_ptrs[0], 0, nbatch, nstep, control_spec,
            state0_ptr, warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr);
      }
    }
  }

 private:
  int nthread_;
  std::shared_ptr<ThreadPool> pool_;
};

PYBIND11_MODULE(_rollout, pymodule, pybind11::mod_gil_not_used()) {
  namespace py = ::pybind11;

  py::class_<Rollout>(pymodule, "Rollout")
      .def(
        py::init([](int nthread) {
          return std::make_unique<Rollout>(nthread);
        }),
        py::kw_only(),
        py::arg("nthread"),
        py::doc(rollout_init_doc))
      .def(
        "rollout",
        &Rollout::rollout,
        py::arg("model"),
        py::arg("data"),
        py::arg("nstep"),
        py::arg("control_spec"),
        py::arg("state0"),
        py::arg("warmstart0") = py::none(),
        py::arg("control")    = py::none(),
        py::arg("state")      = py::none(),
        py::arg("sensordata") = py::none(),
        py::arg("chunk_size") = py::none(),
        py::doc(rollout_doc));
}

}  // namespace

}  // namespace mujoco::python
