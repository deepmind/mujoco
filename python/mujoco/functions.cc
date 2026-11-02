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

#include <stdalign.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <optional>

#include <Eigen/Core>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "errors.h"
#include "function_traits.h"
#include "functions.h"
#include "private.h"
#include "raw.h"
#include "structs.h"
#include "util/func_wrap.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mujoco::python {
namespace {
PYBIND11_MODULE(_functions, pymodule, pybind11::mod_gil_not_used()) {
  namespace py = ::pybind11;
  namespace traits = python_traits;

  using EigenVectorI = Eigen::Vector<int, Eigen::Dynamic>;
  using EigenVectorX = Eigen::Vector<mjtNum, Eigen::Dynamic>;
  using EigenArrayXX = Eigen::Array<
      mjtNum, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


  // Import the _structs module so that pybind11 knows about Python bindings
  // for MjWrapper types and therefore generates prettier docstrings.
  py::module::import("mujoco._structs");

  // Virtual file system
  // Skipped entire section

  // Asset cache
  Def<traits::mj_getCacheSize>(pymodule, [](void* cache) {
        return mj_getCacheSize(static_cast<mjCache*>(cache));
      });

  Def<traits::mj_getCacheCapacity>(pymodule, [](void* cache) {
        return mj_getCacheCapacity(static_cast<mjCache*>(cache));
      });

  Def<traits::mj_setCacheCapacity>(pymodule, [](void* cache, std::size_t size) {
        return mj_setCacheCapacity(static_cast<mjCache*>(cache), size);
      });

  Def<traits::mj_getCache>(pymodule, []() {
        return static_cast<void*>(mj_getCache());
      });

  Def<traits::mj_clearCache>(pymodule, [](void* cache) {
        return mj_clearCache(static_cast<mjCache*>(cache));
      });

  // Parse and compile
  // Skipped: mj_loadXML (have MjModel.from_xml_string)
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_saveLastXML, "error", "error_sz")(
      pymodule,
      [](const char* filename, const mjModel* m) {
        std::array<char, 1024> error;
        int success = InterceptMjErrors(::mj_saveLastXML)(
            filename, m, error.data(), error.size());
        if (!success) {
          throw FatalError(std::string(error.data()));
        }
      });
  // Skipped: mj_freeLastXML
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_printSchema,
                           "filename", "buffer", "buffer_sz")(
      pymodule, [](bool flg_html, bool flg_pad) {
        constexpr int kBufferSize = 60000;
        auto buffer = std::unique_ptr<char[]>(new char[kBufferSize]);
        const int out_length = InterceptMjErrors(::mj_printSchema)(
            nullptr, buffer.get(), kBufferSize, flg_html, flg_pad);
        if (out_length >= kBufferSize) {
          throw UnexpectedError("output buffer too small");
        }
        return std::string(buffer.get(), out_length);
      });

  DEF_WITH_OMITTED_PY_ARGS(traits::mju_getXMLDependencies,
                           "dependencies")(
      pymodule, [](const char* filename){
        mjStringVec dependencies;
        InterceptMjErrors(::mju_getXMLDependencies)(filename, &dependencies);
        return dependencies;
      });

  // mj_loadTexture returns its out-parameters: a (height, width, nchannel)
  // uint8 array which owns the mju_malloc'ed buffer, and the colorspace
  pymodule.def(
      "mj_loadTexture",
      [](const MjModelWrapper& m, int id) {
        std::array<char, 1024> error;
        mjtByte* data = nullptr;
        int width, height, colorspace, status;
        {
          py::gil_scoped_release no_gil;
          status = InterceptMjErrors(::mj_loadTexture)(
              m.get(), id, &data, &width, &height, &colorspace, error.data(),
              error.size());
        }
        if (status) {
          throw py::value_error(std::string(error.data()));
        }
        int nchannel = m.get()->tex_nchannel[id];
        py::capsule owner(data, [](void* p) { mju_free(p); });
        py::array_t<std::uint8_t> pixels({height, width, nchannel}, data,
                                         owner);
        return std::make_pair(pixels, colorspace);
      },
      py::arg("m"), py::arg("id"),
      py::doc((std::string(traits::mj_loadTexture::doc) +
               std::string(" Returns (pixels, colorspace), pixels an array of "
                           "shape (height, width, nchannel)."))
                  .c_str()));

  // mj_embedAssets returns a new MjModel which owns the model
  py::function mjmodel_from_raw_ptr = py::module::import("mujoco._structs")
                                          .attr("MjModel")
                                          .attr("_from_model_ptr");
  pymodule.def(
      "mj_embedAssets",
      [mjmodel_from_raw_ptr](const MjModelWrapper& m) -> py::object {
        std::array<char, 1024> error;
        raw::MjModel* embedded = nullptr;
        {
          py::gil_scoped_release no_gil;
          embedded = InterceptMjErrors(::mj_embedAssets)(
              m.get(), error.data(), error.size());
        }
        if (!embedded) {
          throw py::value_error(std::string(error.data()));
        }
        // wrapped by _structs, which owns MjModelWrapper's symbols
        return mjmodel_from_raw_ptr(reinterpret_cast<std::uintptr_t>(embedded));
      },
      py::arg("m"), py::doc(traits::mj_embedAssets::doc));

  // Main simulation
  pymodule.def(
      "mj_step",
      InterceptMjErrors(
          [](const MjModelWrapper& m, MjDataWrapper& d, int nstep) {
            const raw::MjModel* const m_ptr = m.get();
            raw::MjData* const d_ptr = d.get();
            for (int i = 0; i < nstep; ++i) {
              ::mj_step(m_ptr, d_ptr);
            }
          }),
      py::arg("m"), py::arg("d"), py::arg_v("nstep", 1),
      py::doc((std::string(traits::mj_step::doc) +
               std::string(" Optionally, repeat nstep times.")).c_str()),
      py::call_guard<py::gil_scoped_release>());
  Def<traits::mj_step1>(pymodule);
  Def<traits::mj_step2>(pymodule);
  Def<traits::mj_forward>(pymodule);
  Def<traits::mj_inverse>(pymodule);
  Def<traits::mj_forwardSkip>(pymodule);
  Def<traits::mj_inverseSkip>(pymodule);

  // Initialization
  Def<traits::mj_defaultLROpt>(pymodule);
  Def<traits::mj_defaultSolRefImp>(
      pymodule,
      [](std::optional<Eigen::Ref<Eigen::Vector<mjtNum, mjNREF>>> solref,
         std::optional<Eigen::Ref<Eigen::Vector<mjtNum, mjNIMP>>> solimp) {
        return InterceptMjErrors(::mj_defaultSolRefImp)(
            solref.has_value() ? solref->data() : nullptr,
            solimp.has_value() ? solimp->data() : nullptr);
      });
  Def<traits::mj_defaultOption>(pymodule);
  Def<traits::mj_defaultVisual>(pymodule);
  // Skipped: mj_copyModel (have MjModel.__copy__, memory managed by MjModel)
  pymodule.def(
      "mj_saveModel",
      [](const MjModelWrapper& m,
         const std::optional<std::string>& filename = std::nullopt,
         std::optional<Eigen::Ref<Eigen::Vector<std::uint8_t, Eigen::Dynamic>>>
             buffer = std::nullopt) {
        void* buffer_ptr = nullptr;
        mjtSize buffer_sz = 0;
        if (buffer.has_value()) {
          buffer_ptr = buffer->data();
          buffer_sz = buffer->size();
        }
        return InterceptMjErrors(::mj_saveModel)(
            m.get(), filename.has_value() ? filename->c_str() : nullptr,
            buffer_ptr, buffer_sz);
      },
      py::arg("m"), py::arg_v("filename", std::nullopt),
      py::arg_v("buffer", std::nullopt), py::doc(traits::mj_saveModel::doc),
      py::call_guard<py::gil_scoped_release>());
  // Skipped: mj_loadModel (have MjModel.from_binary_path)
  // Skipped: mj_deleteModel (have MjModel.__del__)
  Def<traits::mj_sizeModel>(pymodule);
  // Skipped: mj_makeData (have MjData.__init__)
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_copyData)(
      pymodule,
      [](raw::MjData* dest, const raw::MjModel* m, const raw::MjData* src) {
        InterceptMjErrors(::mj_copyData)(dest, m, src);
      });
  Def<traits::mj_resetData>(pymodule);
  Def<traits::mj_resetCtrl>(pymodule);
  Def<traits::mj_resetDataDebug>(pymodule);
  Def<traits::mj_resetDataKeyframe>(pymodule);
  // Skipped: mj_stackAllocByte (doesn't make sense in Python)
  // Skipped: mj_deleteData (have MjData.__del__)
  Def<traits::mj_resetCallbacks>(pymodule);
  Def<traits::mj_setConst>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_setLengthRange, "error", "error_sz")(
      pymodule,
      [](raw::MjModel* m, raw::MjData* d, int index, const raw::MjLROpt* opt) {
        std::array<char, 1024> error;
        int success = InterceptMjErrors(::mj_setLengthRange)(
            m, d, index, opt, error.data(), error.size());
        if (!success) {
          throw FatalError(std::string(error.data()));
        }
      });

  // Printing
  Def<traits::mj_printFormattedModel>(pymodule);
  Def<traits::mj_printModel>(pymodule);
  Def<traits::mj_printFormattedData>(pymodule);
  Def<traits::mj_printData>(pymodule);
  Def<traits::mj_printFormattedScene>(pymodule);
  Def<traits::mj_printScene>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_printMat, "nr", "nc")(
      pymodule,
      [](Eigen::Ref<const EigenArrayXX> mat) {
        return ::mju_printMat(mat.data(), mat.rows(), mat.cols());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_printMatSparse, "nr")(
      pymodule,
      [](Eigen::Ref<const EigenVectorX> mat,
         Eigen::Ref<const Eigen::Vector<int, Eigen::Dynamic>> rownnz,
         Eigen::Ref<const Eigen::Vector<int, Eigen::Dynamic>> rowadr,
         Eigen::Ref<const Eigen::Vector<int, Eigen::Dynamic>> colind) {
        if (rownnz.size() != rowadr.size()) {
          throw py::type_error("size of rownnz should equal size of rowadr");
        }
        const int nnz = rowadr[rowadr.size() - 1] + rownnz[rownnz.size() - 1];
        if (mat.size() != nnz) {
          throw py::type_error(
              "size of mat should equal rownnz[-1] + rowadr[-1]");
        }
        if (colind.size() != nnz) {
          throw py::type_error(
              "size of colind should equal rownnz[-1] + rowadr[-1]");
        }
        return InterceptMjErrors(::mju_printMatSparse)(
            mat.data(), rowadr.size(), rownnz.data(),
            rowadr.data(), colind.data());
      });

  // Components
  Def<traits::mj_fwdKinematics>(pymodule);
  Def<traits::mj_fwdPosition>(pymodule);
  Def<traits::mj_fwdVelocity>(pymodule);
  Def<traits::mj_fwdActuation>(pymodule);
  Def<traits::mj_fwdAcceleration>(pymodule);
  Def<traits::mj_fwdConstraint>(pymodule);
  Def<traits::mj_Euler>(pymodule);
  Def<traits::mj_RungeKutta>(pymodule);
  Def<traits::mj_implicit>(pymodule);
  Def<traits::mj_invPosition>(pymodule);
  Def<traits::mj_invVelocity>(pymodule);
  Def<traits::mj_invConstraint>(pymodule);
  Def<traits::mj_compareFwdInv>(pymodule);

  // Sub components
  Def<traits::mj_sensorPos>(pymodule);
  Def<traits::mj_sensorVel>(pymodule);
  Def<traits::mj_sensorAcc>(pymodule);
  Def<traits::mj_energyPos>(pymodule);
  Def<traits::mj_energyVel>(pymodule);
  Def<traits::mj_checkPos>(pymodule);
  Def<traits::mj_checkVel>(pymodule);
  Def<traits::mj_checkAcc>(pymodule);
  Def<traits::mj_kinematics>(pymodule);
  Def<traits::mj_comPos>(pymodule);
  Def<traits::mj_camlight>(pymodule);
  Def<traits::mj_flex>(pymodule);
  Def<traits::mj_tendon>(pymodule);
  Def<traits::mj_transmission>(pymodule);
  Def<traits::mj_crb>(pymodule);
  Def<traits::mj_makeM>(pymodule);
  Def<traits::mj_factorM>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_solveM, "n")(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, Eigen::Ref<EigenArrayXX> x,
         Eigen::Ref<const EigenArrayXX> y) {
        if (x.rows() != y.rows()) {
          throw py::type_error(
              "the first dimension of x and y should be of the same size");
        }
        if (x.cols() != m->nv) {
          throw py::type_error(
              "the last dimension of x should be of size nv");
        }
        if (y.cols() != m->nv) {
          throw py::type_error(
              "the last dimension of y should be of size nv");
        }
        return InterceptMjErrors(::mj_solveM)(
            m, d, x.data(), y.data(), y.rows());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_solveM2, "n")(
      pymodule, [](const raw::MjModel* m, raw::MjData* d,
                   Eigen::Ref<EigenArrayXX> x, Eigen::Ref<const EigenArrayXX> y,
                   Eigen::Ref<const EigenArrayXX> sqrtInvD) {
        if (x.rows() != y.rows()) {
          throw py::type_error(
              "the first dimension of x and y should be of the same size");
        }
        if (x.cols() != m->nv) {
          throw py::type_error("the last dimension of x should be of size nv");
        }
        if (y.cols() != m->nv) {
          throw py::type_error("the last dimension of y should be of size nv");
        }
        if (sqrtInvD.size() != m->nv) {
          throw py::type_error("the size of sqrtInvD should be nv");
        }
        return InterceptMjErrors(::mj_solveM2)(m, d, x.data(), y.data(),
                                               sqrtInvD.data(), y.rows());
      });
  Def<traits::mj_comVel>(pymodule);
  Def<traits::mj_passive>(pymodule);
  Def<traits::mj_subtreeVel>(pymodule);
  Def<traits::mj_rne>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, int flg_acc,
         Eigen::Ref<EigenVectorX> result) {
        if (result.size() != m->nv) {
          throw py::type_error("result should have length nv");
        }
        return InterceptMjErrors(::mj_rne)(
            m, d, flg_acc, result.data());
      });
  Def<traits::mj_rnePostConstraint>(pymodule);
  Def<traits::mj_maxContact>(pymodule);
  Def<traits::mj_collision>(pymodule);
  Def<traits::mj_makeConstraint>(pymodule);
  Def<traits::mj_island>(pymodule);
  Def<traits::mj_projectConstraint>(pymodule);
  Def<traits::mj_referenceConstraint>(pymodule);
  Def<traits::mj_constraintUpdate>(
      pymodule, [](const raw::MjModel* m, raw::MjData* d,
                   Eigen::Ref<const EigenVectorX> jar,
                   std::optional<Eigen::Ref<Eigen::Vector<mjtNum, 1>>> cost,
                   int flg_coneHessian) {
        if (jar.size() != d->nefc) {
          throw py::type_error("size of jar should equal nefc");
        }
        return InterceptMjErrors(::mj_constraintUpdate)(
            m, d, jar.data(), cost.has_value() ? cost->data() : nullptr,
            flg_coneHessian);
      });

  // Support
  Def<traits::mj_stateSize>(pymodule);
  Def<traits::mj_getState>(
      pymodule,
      [](const raw::MjModel* m, const raw::MjData* d,
         Eigen::Ref<EigenVectorX> state, unsigned int sig) {
        if (state.size() != mj_stateSize(m, sig)) {
          throw py::type_error("state size should equal mj_stateSize(m, sig)");
        }
        return InterceptMjErrors(::mj_getState)(m, d, state.data(), sig);
      });
  Def<traits::mj_extractState>(
    pymodule,
    [](const raw::MjModel* m,
       Eigen::Ref<const EigenVectorX> src, unsigned int srcsig,
       Eigen::Ref<EigenVectorX> dst, unsigned int dstsig) {
        if (src.size() != mj_stateSize(m, srcsig)) {
          throw py::type_error("src size should equal mj_stateSize(m, srcsig)");
        }
        if (dst.size() != mj_stateSize(m, dstsig)) {
          throw py::type_error("dst size should equal mj_stateSize(m, dstsig)");
        }
        return InterceptMjErrors(::mj_extractState)(m, src.data(), srcsig,
                                                    dst.data(), dstsig);
      });
  Def<traits::mj_setState>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         Eigen::Ref<const EigenVectorX> state, unsigned int sig) {
        if (state.size() != mj_stateSize(m, sig)) {
          throw py::type_error("state size should equal mj_stateSize(m, sig)");
        }
        return InterceptMjErrors(::mj_setState)(m, d, state.data(), sig);
      });
  Def<traits::mj_copyState>(pymodule);
  Def<traits::mj_readCtrl>(
      pymodule,
      [](const raw::MjModel* m, const raw::MjData* d, int id, mjtNum time,
         Eigen::Ref<EigenVectorX> result, int interp) {
        if (id < 0 || id >= m->nactuator) {
          throw py::index_error("actuator id out of range");
        }
        int dim = m->actuator_ctrlnum[id];
        if (result.size() != dim) {
          throw py::type_error(
              "result should have length actuator_ctrlnum[id]");
        }
        const mjtNum* ptr = InterceptMjErrors(::mj_readCtrl)(
            m, d, id, time, result.data(), interp);
        if (ptr && ptr != result.data()) {
          for (int i = 0; i < dim; ++i) {
            result[i] = ptr[i];
          }
        }
        return result;
      });
  Def<traits::mj_readSensor>(
      pymodule,
      [](const raw::MjModel* m, const raw::MjData* d, int id, mjtNum time,
         Eigen::Ref<EigenVectorX> result, int interp) {
        if (id < 0 || id >= m->nsensor) {
          throw py::index_error("sensor id out of range");
        }
        int dim = m->sensor_dim[id];
        if (result.size() != dim) {
          throw py::type_error("result should have length sensor_dim[id]");
        }
        const mjtNum* ptr = InterceptMjErrors(::mj_readSensor)(
            m, d, id, time, result.data(), interp);
        if (ptr && ptr != result.data()) {
          for (int i = 0; i < dim; ++i) {
            result[i] = ptr[i];
          }
        }
        return result;
      });
  Def<traits::mj_initCtrlHistory>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, int id,
         std::optional<Eigen::Ref<const EigenVectorX>> times,
         Eigen::Ref<const EigenArrayXX> values) {
        if (id < 0 || id >= m->nactuator) {
          throw py::index_error("actuator id out of range");
        }
        int nhistory = m->actuator_history[2 * id];
        int dim = m->actuator_ctrlnum[id];
        if (times.has_value() && times->size() != nhistory) {
          throw py::type_error(
              "times should have length actuator_history[2*id]");
        }
        if (values.rows() != nhistory || values.cols() != dim) {
          throw py::type_error(
              "values should have shape (actuator_history[2*id], "
              "actuator_ctrlnum[id])");
        }
        return InterceptMjErrors(::mj_initCtrlHistory)(
            m, d, id,
            times.has_value() ? times->data() : nullptr, values.data());
      });
  Def<traits::mj_initSensorHistory>(
      pymodule, [](const raw::MjModel* m, raw::MjData* d, int id,
                   std::optional<Eigen::Ref<const EigenVectorX>> times,
                   Eigen::Ref<const EigenArrayXX> values, mjtNum phase) {
        if (id < 0 || id >= m->nsensor) {
          throw py::index_error("sensor id out of range");
        }
        int nhistory = m->sensor_history[2 * id];
        int dim = m->sensor_dim[id];
        if (times.has_value() && times->size() != nhistory) {
          throw py::type_error("times should have length sensor_history[2*id]");
        }
        if (values.rows() != nhistory || values.cols() != dim) {
          throw py::type_error(
              "values should have shape (sensor_history[2*id], "
              "sensor_dim[id])");
        }
        return InterceptMjErrors(::mj_initSensorHistory)(
            m, d, id, times.has_value() ? times->data() : nullptr,
            values.data(), phase);
      });
  Def<traits::mj_setKeyframe>(pymodule);
  Def<traits::mj_addContact>(pymodule);
  Def<traits::mj_isPyramidal>(pymodule);
  Def<traits::mj_isSparse>(pymodule);
  Def<traits::mj_isDual>(pymodule);
  Def<traits::mj_mulJacVec>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != d->nefc) {
          throw py::type_error("res should be of length nefc");
        }
        if (vec.size() != m->nv) {
          throw py::type_error("vec should be of length nv");
        }
        return InterceptMjErrors(::mj_mulJacVec)(m, d, res.data(), vec.data());
      });
  Def<traits::mj_mulJacTVec>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != m->nv) {
          throw py::type_error("res should be of length nv");
        }
        if (vec.size() != d->nefc) {
          throw py::type_error("vec should be of length nefc");
        }
        return InterceptMjErrors(::mj_mulJacTVec)(m, d, res.data(), vec.data());
      });
  Def<traits::mj_jac>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr,
         const mjtNum (*point)[3], int body) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jac)(
            m, d,
            jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr,
            &(*point)[0], body);
      });
  Def<traits::mj_jacBody>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr, int body) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacBody)(
            m, d, jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr, body);
      });
  Def<traits::mj_jacBodyCom>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr, int body) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacBodyCom)(
            m, d, jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr, body);
      });
  Def<traits::mj_jacSubtreeCom>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp, int body) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacSubtreeCom)(
            m, d, jacp.has_value() ? jacp->data() : nullptr, body);
      });
  Def<traits::mj_jacGeom>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr, int geom) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacGeom)(
            m, d, jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr, geom);
      });
  Def<traits::mj_jacSite>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr, int site) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacSite)(
            m, d, jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr, site);
      });
  Def<traits::mj_jacPointAxis>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr,
         const mjtNum (*point)[3], const mjtNum (*axis)[3], int body) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacPointAxis)(
            m, d, jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr,
            &(*point)[0], &(*axis)[0], body);
      });
  Def<traits::mj_jacDot>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         std::optional<Eigen::Ref<EigenArrayXX>> jacp,
         std::optional<Eigen::Ref<EigenArrayXX>> jacr,
         const mjtNum (*point)[3], int body) {
        if (jacp.has_value() &&
            (jacp->rows() != 3 || jacp->cols() != m->nv)) {
          throw py::type_error("jacp should be of shape (3, nv)");
        }
        if (jacr.has_value() &&
            (jacr->rows() != 3 || jacr->cols() != m->nv)) {
          throw py::type_error("jacr should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_jacDot)(
            m, d,
            jacp.has_value() ? jacp->data() : nullptr,
            jacr.has_value() ? jacr->data() : nullptr,
            &(*point)[0], body);
      });
  Def<traits::mj_angmomMat>(
      pymodule, [](const raw::MjModel* m, raw::MjData* d,
                   Eigen::Ref<EigenArrayXX> mat, int body) {
        if (mat.rows() != 3 || mat.cols() != m->nv) {
          throw py::type_error("mat should be of shape (3, nv)");
        }
        return InterceptMjErrors(::mj_angmomMat)(m, d, mat.data(), body);
      });
  Def<traits::mj_name2id>(pymodule);
  Def<traits::mj_id2name>(pymodule);
  Def<traits::mj_actuatorInputName>(pymodule);
  Def<traits::mj_fullM>(
      pymodule,
      [](const raw::MjModel* m, const raw::MjData* d,
         Eigen::Ref<EigenArrayXX> dst) {
        if (dst.cols() != m->nv || dst.rows() != m->nv) {
          throw py::type_error("dst should be of shape (nv, nv)");
        }
        return ::mj_fullM(m, d, dst.data());
      });
  Def<traits::mj_mulM>(
      pymodule,
      [](const raw::MjModel* m, const raw::MjData* d,
         Eigen::Ref<EigenVectorX> res, Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != m->nv) {
          throw py::type_error("res should be of size nv");
        }
        if (vec.size() != m->nv) {
          throw py::type_error("vec should be of size nv");
        }
        return InterceptMjErrors(::mj_mulM)(m, d, res.data(), vec.data());
      });
  Def<traits::mj_mulM2>(
      pymodule,
      [](const raw::MjModel* m, const raw::MjData* d,
         Eigen::Ref<EigenVectorX> res, Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != m->nv) {
          throw py::type_error("res should be of size nv");
        }
        if (vec.size() != m->nv) {
          throw py::type_error("vec should be of size nv");
        }
        return InterceptMjErrors(::mj_mulM2)(m, d, res.data(), vec.data());
      });
  Def<traits::mj_addM>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, Eigen::Ref<EigenVectorX> dst,
         Eigen::Ref<Eigen::Vector<int, Eigen::Dynamic>> rownnz,
         Eigen::Ref<Eigen::Vector<int, Eigen::Dynamic>> rowadr,
         Eigen::Ref<Eigen::Vector<int, Eigen::Dynamic>> colind) {
        if (dst.size() != m->nM) {
          throw py::type_error("dst should be of size nM");
        }
        if (rownnz.size() != m->nv) {
          throw py::type_error("rownnz should be of size nv");
        }
        if (rowadr.size() != m->nv) {
          throw py::type_error("rowadr should be of size nv");
        }
        if (colind.size() != m->nM) {
          throw py::type_error("colind should be of size nM");
        }
        return InterceptMjErrors(::mj_addM)(
            m, d, dst.data(), rownnz.data(), rowadr.data(), colind.data());
      });
  Def<traits::mj_applyFT>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         const mjtNum (*force)[3], const mjtNum (*torque)[3],
         const mjtNum (*point)[3], int body,
         Eigen::Ref<EigenVectorX> qfrc_target) {
        if (qfrc_target.size() != m->nv) {
          throw py::type_error("qfrc_target should be of size nv");
        }
        return InterceptMjErrors(::mj_applyFT)(
            m, d, &(*force)[0], &(*torque)[0], &(*point)[0],
            body, qfrc_target.data());
      });
  Def<traits::mj_objectVelocity>(pymodule);
  Def<traits::mj_objectAcceleration>(pymodule);
  Def<traits::mj_contactForce>(pymodule);
  Def<traits::mj_geomDistance>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         int geom1, int geom2, mjtNum distmax,
         std::optional<Eigen::Ref<EigenArrayXX>> fromto) {
        if (fromto.has_value() && fromto->size() != 6) {
          throw py::type_error("fromto should be of size 6");
        }
        return InterceptMjErrors(::mj_geomDistance)(
            m, d, geom1, geom2, distmax,
            fromto.has_value() ? fromto->data() : nullptr);
      });
  Def<traits::mj_differentiatePos>(
      pymodule,
      [](const raw::MjModel* m, Eigen::Ref<EigenVectorX> qvel,
         mjtNum dt, Eigen::Ref<const EigenVectorX> qpos1,
         Eigen::Ref<const EigenVectorX> qpos2) {
        if (qvel.size() != m->nv) {
          throw py::type_error("qvel should be of size nv");
        }
        if (qpos1.size() != m->nq) {
          throw py::type_error("qpos1 should be of size nq");
        }
        if (qpos2.size() != m->nq) {
          throw py::type_error("qpos2 should be of size nq");
        }
        return InterceptMjErrors(::mj_differentiatePos)(
            m, qvel.data(), dt, qpos1.data(), qpos2.data());
      });
  Def<traits::mj_integratePos>(
      pymodule,
      [](const raw::MjModel* m, Eigen::Ref<EigenVectorX> qpos,
         Eigen::Ref<const EigenVectorX> qvel, mjtNum dt) {
        if (qpos.size() != m->nq) {
          throw py::type_error("qpos should be of size nq");
        }
        if (qvel.size() != m->nv) {
          throw py::type_error("qvel should be of size nv");
        }
        return InterceptMjErrors(::mj_integratePos)(
            m, qpos.data(), qvel.data(), dt);
      });
  Def<traits::mj_normalizeQuat>(
      pymodule,
      [](const raw::MjModel* m, Eigen::Ref<EigenVectorX> qpos) {
        if (qpos.size() != m->nq) {
          throw py::type_error("qpos should be of size nq");
        }
        return InterceptMjErrors(::mj_normalizeQuat)(m, qpos.data());
      });
  Def<traits::mj_local2Global>(pymodule);
  Def<traits::mj_getTotalmass>(pymodule);
  Def<traits::mj_setTotalmass>(pymodule);
  Def<traits::mj_loadPluginLibrary>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mj_loadAllPluginLibraries, "callback")(
      pymodule,
      [](const std::string& directory) {
        InterceptMjErrors(::mj_loadAllPluginLibraries)(
            directory.c_str(), nullptr);
      });
  Def<traits::mj_version>(pymodule);
  Def<traits::mj_versionString>(pymodule);

  // Thread pool
  Def<traits::mju_threadpool>(pymodule);

  // Ray collision
  Def<traits::mj_multiRay>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, const mjtNum(*pnt)[3],
         Eigen::Ref<const EigenVectorX> vec,
         std::optional<Eigen::Ref<const Eigen::Vector<mjtByte, mjNGROUP>>>
             geomgroup,
         mjtBool flg_static, int bodyexclude, Eigen::Ref<EigenVectorI> geomid,
         Eigen::Ref<EigenVectorX> dist,
         std::optional<Eigen::Ref<EigenVectorX>> normal,
         int nray, mjtNum cutoff) {
        if (dist.size() != nray || geomid.size() != nray) {
          throw py::type_error("dist and geomid should be of size nray");
        }
        if (vec.size() != 3 * nray) {
          throw py::type_error("vec should be of size 3*nray");
        }
        if (normal.has_value() && normal->size() != 3 * nray) {
          throw py::type_error("normal should be of size 3*nray");
        }
        InterceptMjErrors(::mj_multiRay)(
            m, d, &(*pnt)[0], vec.data(),
            geomgroup.has_value() ? geomgroup->data() : nullptr, flg_static,
            bodyexclude, geomid.data(), dist.data(),
            normal.has_value() ? normal->data() : nullptr, nray, cutoff);
      });
  pymodule.def(
      "mj_ray",
      util::UnwrapArgs(
          [](const raw::MjModel* m, const raw::MjData* d, const mjtNum(*pnt)[3],
             const mjtNum(*vec)[3],
             std::optional<Eigen::Ref<const Eigen::Vector<mjtByte, mjNGROUP>>>
                 geomgroup,
             mjtBool flg_static, int bodyexclude,
             std::optional<Eigen::Ref<Eigen::Vector<int, 1>>> geomid,
             std::optional<Eigen::Ref<Eigen::Vector<mjtNum, 3>>> normal) {
            return mj_ray(m, d, &(*pnt)[0], &(*vec)[0],
                          geomgroup.has_value() ? geomgroup->data() : nullptr,
                          flg_static, bodyexclude,
                          geomid.has_value() ? geomid->data() : nullptr,
                          normal.has_value() ? normal->data() : nullptr);
          }),
      py::arg("m"), py::arg("d"), py::arg("pnt"), py::arg("vec"),
      py::arg("geomgroup"), py::arg("flg_static"), py::arg("bodyexclude"),
      py::arg("geomid"), py::arg("normal") = std::nullopt,
      py::doc(traits::mj_ray::doc),
      py::call_guard<py::gil_scoped_release>());
  pymodule.def(
      "mj_rayHfield",
      util::UnwrapArgs(
          [](const raw::MjModel* m, const raw::MjData* d, int geomid,
             const mjtNum(*pnt)[3], const mjtNum(*vec)[3],
             std::optional<Eigen::Ref<Eigen::Vector<mjtNum, 3>>> normal) {
            return mj_rayHfield(m, d, geomid, &(*pnt)[0], &(*vec)[0],
                                normal.has_value() ? normal->data() : nullptr);
          }),
      py::arg("m"), py::arg("d"), py::arg("geomid"), py::arg("pnt"),
      py::arg("vec"), py::arg("normal") = std::nullopt,
      py::doc(traits::mj_rayHfield::doc),
      py::call_guard<py::gil_scoped_release>());
  pymodule.def(
      "mj_rayMesh",
      util::UnwrapArgs(
          [](const raw::MjModel* m, const raw::MjData* d, int geomid,
             const mjtNum(*pnt)[3], const mjtNum(*vec)[3],
             std::optional<Eigen::Ref<Eigen::Vector<mjtNum, 3>>> normal) {
            return mj_rayMesh(m, d, geomid, &(*pnt)[0], &(*vec)[0],
                              normal.has_value() ? normal->data() : nullptr);
          }),
      py::arg("m"), py::arg("d"), py::arg("geomid"), py::arg("pnt"),
      py::arg("vec"), py::arg("normal") = std::nullopt,
      py::doc(traits::mj_rayMesh::doc),
      py::call_guard<py::gil_scoped_release>());
  pymodule.def(
      "mju_rayGeom",
      util::UnwrapArgs(
          [](const mjtNum(*pos)[3], const mjtNum(*mat)[9],
             const mjtNum(*size)[3], const mjtNum(*pnt)[3],
             const mjtNum(*vec)[3], int geomtype,
             std::optional<Eigen::Ref<Eigen::Vector<mjtNum, 3>>> normal) {
            return mju_rayGeom(&(*pos)[0], &(*mat)[0], &(*size)[0], &(*pnt)[0],
                               &(*vec)[0], geomtype,
                               normal.has_value() ? normal->data() : nullptr);
          }),
      py::arg("pos"), py::arg("mat"), py::arg("size"), py::arg("pnt"),
      py::arg("vec"), py::arg("geomtype"), py::arg("normal") = std::nullopt,
      py::doc(traits::mju_rayGeom::doc),
      py::call_guard<py::gil_scoped_release>());
  pymodule.def(
      "mj_rayFlex",
      util::UnwrapArgs(
          [](const raw::MjModel* m, const raw::MjData* d, int flex_layer,
             mjtBool flg_vert, mjtBool flg_edge, mjtBool flg_face,
             mjtBool flg_skin, int flexid, const mjtNum(*pnt)[3],
             const mjtNum(*vec)[3],
             std::optional<Eigen::Ref<Eigen::Vector<int, 1>>> vertid,
             std::optional<Eigen::Ref<Eigen::Vector<mjtNum, 3>>> normal) {
            return mj_rayFlex(m, d, flex_layer, flg_vert, flg_edge, flg_face,
                              flg_skin, flexid, &(*pnt)[0], &(*vec)[0],
                              vertid.has_value() ? vertid->data() : nullptr,
                              normal.has_value() ? normal->data() : nullptr);
          }),
      py::arg("m"), py::arg("d"), py::arg("flex_layer"), py::arg("flg_vert"),
      py::arg("flg_edge"), py::arg("flg_face"), py::arg("flg_skin"),
      py::arg("flexid"), py::arg("pnt"), py::arg("vec"),
      py::arg("vertid") = std::nullopt, py::arg("normal") = std::nullopt,
      py::doc(traits::mj_rayFlex::doc),
      py::call_guard<py::gil_scoped_release>());
  Def<traits::mju_raySkin>(
      pymodule,
      [](int nface, int nvert,
         Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
             face,
         Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
             vert,
         const mjtNum(*pnt)[3], const mjtNum(*vec)[3],
         std::optional<Eigen::Ref<Eigen::Vector<int, 1>>> vertid) {
        if (nface < 0 || nvert < 0) {
          throw py::value_error("nface and nvert must be non-negative");
        }
        if (face.size() != 3 * static_cast<Eigen::Index>(nface)) {
          throw py::type_error("face should be of size 3*nface");
        }
        if (vert.size() != 3 * static_cast<Eigen::Index>(nvert)) {
          throw py::type_error("vert should be of size 3*nvert");
        }
        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            face_copy;
        const int* face_ptr = face.data();
        if (face.outerStride() != face.cols()) {
          face_copy = face;
          face_ptr = face_copy.data();
        }
        Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            vert_copy;
        const float* vert_ptr = vert.data();
        if (vert.outerStride() != vert.cols()) {
          vert_copy = vert;
          vert_ptr = vert_copy.data();
        }
        for (Eigen::Index i = 0; i < face.size(); ++i) {
          if (face_ptr[i] < 0 || face_ptr[i] >= nvert) {
            throw py::value_error("face vertex index out of bounds");
          }
        }
        return InterceptMjErrors(::mju_raySkin)(
            nface, nvert, face_ptr, vert_ptr, &(*pnt)[0], &(*vec)[0],
            vertid.has_value() ? vertid->data() : nullptr);
      });

  // Interaction
  Def<traits::mjv_defaultCamera>(pymodule);
  Def<traits::mjv_defaultFreeCamera>(pymodule);
  Def<traits::mjv_defaultPerturb>(pymodule);
  Def<traits::mjv_room2model>(pymodule);
  Def<traits::mjv_model2room>(pymodule);
  Def<traits::mjv_cameraInModel>(pymodule);
  Def<traits::mjv_cameraInRoom>(pymodule);
  Def<traits::mjv_frustumHeight>(pymodule);
  Def<traits::mjv_alignToCamera>(pymodule);
  Def<traits::mjv_moveCamera>(pymodule);
  Def<traits::mjv_movePerturb>(pymodule);
  Def<traits::mjv_moveModel>(pymodule);
  Def<traits::mjv_initPerturb>(pymodule);
  Def<traits::mjv_applyPerturbPose>(pymodule);
  Def<traits::mjv_applyPerturbForce>(pymodule);
  // Skipped: mjv_averageCamera (defined in structs.cc due to the return type)
  // Skipped: mjv_camera2GLCamera (defined in structs.cc due to the return type)
  Def<traits::mjv_select>(pymodule);

  // Visualization
  Def<traits::mjv_defaultOption>(pymodule);
  Def<traits::mjv_defaultFigure>(pymodule);
  Def<traits::mjv_initGeom>(pymodule);
  Def<traits::mjv_connector>(pymodule);
  // Skipped: mjv_defaultScene (have MjvScene.__init__, memory managed by
  // MjvScene).
  // Skipped: mjv_makeScene (have MjvScene.__init__)
  // Skipped: mjv_freeScene (have MjvScene.__del__)
  Def<traits::mjv_updateScene>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d, const raw::MjvOption* opt,
         const std::optional<raw::MjvPerturb*> pert, raw::MjvCamera* cam,
         int catmask, raw::MjvScene* scn) {
        const raw::MjvPerturb* pert_ptr = pert.has_value() ? *pert : nullptr;
        return mjv_updateScene(m, d, opt, pert_ptr, cam, catmask, scn);
      });
  Def<traits::mjv_addGeoms>(pymodule);
  Def<traits::mjv_makeLights>(pymodule);
  Def<traits::mjv_updateCamera>(pymodule);
  Def<traits::mjv_updateSkin>(pymodule);
  Def<traits::mjv_cameraFrame>(pymodule);
  Def<traits::mjv_cameraFrustum>(pymodule);

  // UI framework
  // Skipped: entire section (can add this if there's demand)

  // Error and memory
  // Skipped: everything other than the function below (Python has exceptions)
  Def<traits::mju_writeLog>(pymodule);

  // Standard math
  // This section consists only of preprocessor macros.

  // Vector math
  Def<traits::mju_zero3>(pymodule);
  Def<traits::mju_copy3>(pymodule);
  Def<traits::mju_scl3>(pymodule);
  Def<traits::mju_add3>(pymodule);
  Def<traits::mju_sub3>(pymodule);
  Def<traits::mju_addTo3>(pymodule);
  Def<traits::mju_subFrom3>(pymodule);
  Def<traits::mju_addToScl3>(pymodule);
  Def<traits::mju_addScl3>(pymodule);
  Def<traits::mju_normalize3>(pymodule);
  Def<traits::mju_norm3>(pymodule);
  Def<traits::mju_dot3>(pymodule);
  Def<traits::mju_dist3>(pymodule);
  Def<traits::mju_mulMatVec3>(pymodule);
  Def<traits::mju_mulMatTVec3>(pymodule);
  Def<traits::mju_cross>(pymodule);
  Def<traits::mju_zero4>(pymodule);
  Def<traits::mju_unit4>(pymodule);
  Def<traits::mju_copy4>(pymodule);
  Def<traits::mju_normalize4>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_zero, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res) {
        return InterceptMjErrors(::mju_zero)(res.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_fill, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res, mjtNum val) {
        return InterceptMjErrors(::mju_fill)(res.data(), val, res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_copy, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> data) {
        if (res.size() != data.size()) {
          throw py::type_error("res and data should have the same size");
        }
        return InterceptMjErrors(::mju_copy)(
            res.data(), data.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_sum, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> vec) {
        return ::mju_sum(vec.data(), vec.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_L1, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> vec) {
        return InterceptMjErrors(::mju_L1)(vec.data(), vec.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_scl, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec, mjtNum scl) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_scl)(
            res.data(), vec.data(), scl, res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_add, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec1,
         Eigen::Ref<const EigenVectorX> vec2) {
        if (res.size() != vec1.size()) {
          throw py::type_error("res and vec1 should have the same size");
        }
        if (res.size() != vec2.size()) {
          throw py::type_error("res and vec2 should have the same size");
        }
        return ::mju_add(res.data(), vec1.data(), vec2.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_sub, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec1,
         Eigen::Ref<const EigenVectorX> vec2) {
        if (res.size() != vec1.size()) {
          throw py::type_error("res and vec1 should have the same size");
        }
        if (res.size() != vec2.size()) {
          throw py::type_error("res and vec2 should have the same size");
        }
        return InterceptMjErrors(::mju_sub)(
            res.data(), vec1.data(), vec2.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_addTo, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_addTo)(
            res.data(), vec.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_subFrom, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_subFrom)(
            res.data(), vec.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_addToScl, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec, mjtNum scl) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return ::mju_addToScl(res.data(), vec.data(), scl, res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_addScl, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenVectorX> vec1,
         Eigen::Ref<const EigenVectorX> vec2, mjtNum scl) {
        if (res.size() != vec1.size()) {
          throw py::type_error("res and vec1 should have the same size");
        }
        if (res.size() != vec2.size()) {
          throw py::type_error("res and vec2 should have the same size");
        }
        return InterceptMjErrors(::mju_addScl)(
            res.data(), vec1.data(), vec2.data(), scl, res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_normalize, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> vec) {
        return InterceptMjErrors(::mju_normalize)(vec.data(), vec.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_norm, "n")(
      pymodule,
      [](Eigen::Ref<const EigenVectorX> vec) {
        return InterceptMjErrors(::mju_norm)(vec.data(), vec.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_dot, "n")(
      pymodule,
      [](Eigen::Ref<const EigenVectorX> vec1,
         Eigen::Ref<const EigenVectorX> vec2) {
        if (vec1.size() != vec2.size()) {
          throw py::type_error("vec1 and vec2 should have the same size");
        }
        return InterceptMjErrors(::mju_dot)(
            vec1.data(), vec2.data(), vec1.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_mulMatVec, "nr", "nc")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenArrayXX> mat,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != mat.rows()) {
          throw py::type_error(
              "size of res should equal the number of rows in mat");
        }
        if (vec.size() != mat.cols()) {
          throw py::type_error(
              "size of vec should equal the number of columns in mat");
        }
        return InterceptMjErrors(::mju_mulMatVec)(
            res.data(), mat.data(), vec.data(), mat.rows(), mat.cols());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_mulMatTVec, "nr", "nc")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenArrayXX> mat,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != mat.cols()) {
          throw py::type_error(
              "size of res should equal the number of columns in mat");
        }
        if (vec.size() != mat.rows()) {
          throw py::type_error(
              "size of vec should equal the number of rows in mat");
        }
        return InterceptMjErrors(::mju_mulMatTVec)(
            res.data(), mat.data(), vec.data(), mat.rows(), mat.cols());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_mulVecMatVec, "n")(
      pymodule,
      [](Eigen::Ref<const EigenVectorX> vec1,
         Eigen::Ref<const EigenArrayXX> mat,
         Eigen::Ref<const EigenVectorX> vec2) {
        if (vec1.size() != vec2.size()) {
          throw py::type_error(
              "size of vec1 should equal the size of vec2");
        }
        if (vec1.size() != mat.cols()) {
          throw py::type_error(
              "size of vectors should equal the number of columns in mat");
        }
        if (vec1.size() != mat.rows()) {
          throw py::type_error(
              "size of vectors should equal the number of rows in mat");
        }
        return InterceptMjErrors(::mju_mulVecMatVec)(
            vec1.data(), mat.data(), vec2.data(), vec1.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_transpose, "nr", "nc")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenArrayXX> mat) {
        if (res.cols() != mat.rows()) {
          throw py::type_error("#columns in res should equal #rows in mat");
        }
        if (res.rows() != mat.cols()) {
          throw py::type_error("#rows in res should equal #columns in mat");
        }
        return InterceptMjErrors(::mju_transpose)(
            res.data(), mat.data(), mat.rows(), mat.cols());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_symmetrize, "n")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenArrayXX> mat) {
        if (mat.cols() != mat.rows()) {
          throw py::type_error("mat should be square");
        }
        if (res.cols() != mat.cols() || res.rows() != mat.rows()) {
          throw py::type_error("res and mat should have the same shape");
        }
        return InterceptMjErrors(::mju_symmetrize)(
            res.data(), mat.data(), mat.rows());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_eye, "n")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> mat) {
        if (mat.cols() != mat.rows()) {
          throw py::type_error("mat should be square");
        }
        return InterceptMjErrors(::mju_eye)(mat.data(), mat.rows());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_mulMatMat, "r1", "c1", "c2")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenArrayXX> mat1,
         Eigen::Ref<const EigenArrayXX> mat2) {
        if (res.rows() != mat1.rows()) {
          throw py::type_error("#rows in res should equal #rows in mat1");
        }
        if (res.cols() != mat2.cols()) {
          throw py::type_error(
              "#columns in res should equal #columns in mat2");
        }
        if (mat1.cols() != mat2.rows()) {
          throw py::type_error("#columns in mat1 should equal #rows in mat2");
        }
        return InterceptMjErrors(::mju_mulMatMat)(
            res.data(), mat1.data(), mat2.data(),
            mat1.rows(), mat1.cols(), mat2.cols());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_mulMatMatT, "r1", "c1", "r2")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenArrayXX> mat1,
         Eigen::Ref<const EigenArrayXX> mat2) {
        if (res.rows() != mat1.rows()) {
          throw py::type_error("#rows in res should equal #rows in mat1");
        }
        if (res.cols() != mat2.rows()) {
          throw py::type_error("#columns in res should equal #rows in mat2");
        }
        if (mat1.cols() != mat2.cols()) {
          throw py::type_error(
              "#columns in mat1 should equal #columns in mat2");
        }
        return InterceptMjErrors(::mju_mulMatMatT)(
            res.data(), mat1.data(), mat2.data(),
            mat1.rows(), mat1.cols(), mat2.rows());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_mulMatTMat, "r1", "c1", "c2")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenArrayXX> mat1,
         Eigen::Ref<const EigenArrayXX> mat2) {
        if (res.rows() != mat1.cols()) {
          throw py::type_error("#rows in res should equal #columns in mat1");
        }
        if (res.cols() != mat2.cols()) {
          throw py::type_error(
              "#columns in res should equal #columns in mat2");
        }
        if (mat1.rows() != mat2.rows()) {
          throw py::type_error("#rows in mat1 should equal #rows in mat2");
        }
        return ::mju_mulMatTMat(res.data(), mat1.data(), mat2.data(),
                                mat1.rows(), mat1.cols(), mat2.cols());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_sqrMatTD, "nr", "nc")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenArrayXX> mat,
         std::optional<Eigen::Ref<EigenVectorX>> diag) {
        if (res.rows() != mat.cols()) {
          throw py::type_error("#rows in res should equal #columns in mat");
        }
        if (res.cols() != mat.cols()) {
          throw py::type_error("#rows in res should equal #columns in mat");
        }
        if (diag.has_value() && diag->size() != mat.rows()) {
          throw py::type_error(
              "size of diag should equal the number of rows in mat");
        }
        return InterceptMjErrors(::mju_sqrMatTD)(
            res.data(), mat.data(),
            diag.has_value() ? diag->data() : nullptr,
            mat.rows(), mat.cols());
      });
  Def<traits::mju_transformSpatial>(pymodule);

  // Sparse math
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_dense2sparse, "nr", "nc", "nnz")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res, Eigen::Ref<const EigenArrayXX> mat,
         Eigen::Ref<EigenVectorI> rownnz, Eigen::Ref<EigenVectorI> rowadr,
         Eigen::Ref<EigenVectorI> colind) {
        if (mat.rows() != rownnz.size()) {
          throw py::type_error("#rows in mat should equal size of rownnz");
        }
        if (mat.rows() != rowadr.size()) {
          throw py::type_error("#rows in mat should equal size of rowadr");
        }
        if (res.size() != colind.size()) {
          throw py::type_error("#size of res should equal size of colind");
        }
        return ::mju_dense2sparse(res.data(), mat.data(), mat.rows(),
                                  mat.cols(), rownnz.data(), rowadr.data(),
                                  colind.data(), res.size());
      });

  DEF_WITH_OMITTED_PY_ARGS(traits::mju_sparse2dense, "nr", "nc")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenVectorX> mat,
         Eigen::Ref<const EigenVectorI> rownnz,
         Eigen::Ref<const EigenVectorI> rowadr,
         Eigen::Ref<const EigenVectorI> colind) {
        if (res.rows() != rownnz.size()) {
          throw py::type_error("#rows in res should equal size of rownnz");
        }
        if (res.rows() != rowadr.size()) {
          throw py::type_error("#rows in res should equal size of rowadr");
        }
        return ::mju_sparse2dense(res.data(), mat.data(), res.rows(),
                                  res.cols(), rownnz.data(), rowadr.data(),
                                  colind.data());
      });

  DEF_WITH_OMITTED_PY_ARGS(traits::mju_sym2dense, "n")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res,
         Eigen::Ref<const EigenVectorX> mat,
         Eigen::Ref<const EigenVectorI> rownnz,
         Eigen::Ref<const EigenVectorI> rowadr,
         Eigen::Ref<const EigenVectorI> colind) {
        if (res.rows() != res.cols()) {
          throw py::type_error("res should be a square matrix");
        }
        if (res.rows() != rownnz.size()) {
          throw py::type_error("#rows in res should equal size of rownnz");
        }
        if (res.rows() != rowadr.size()) {
          throw py::type_error("#rows in res should equal size of rowadr");
        }
        if (res.rows() > 0) {
          int nnz = rowadr.array().tail(1)[0] + rownnz.array().tail(1)[0];
          if (mat.size() < nnz) {
            throw py::type_error("mat size is too small for the given sparse "
                                 "structure");
          }
          if (colind.size() < nnz) {
            throw py::type_error("colind size is too small for the given "
                                 "sparse structure");
          }
        }
        return ::mju_sym2dense(res.data(), mat.data(), res.rows(),
                               rownnz.data(), rowadr.data(), colind.data());
      });

  // Quaternions
  Def<traits::mju_rotVecQuat>(pymodule);
  Def<traits::mju_negQuat>(pymodule);
  Def<traits::mju_mulQuat>(pymodule);
  Def<traits::mju_mulQuatAxis>(pymodule);
  Def<traits::mju_axisAngle2Quat>(pymodule);
  Def<traits::mju_quat2Vel>(pymodule);
  Def<traits::mju_subQuat>(pymodule);
  Def<traits::mju_quat2Mat>(pymodule);
  Def<traits::mju_mat2Quat>(pymodule);
  Def<traits::mju_derivQuat>(pymodule);
  Def<traits::mju_quatIntegrate>(pymodule);
  Def<traits::mju_quatZ2Vec>(pymodule);
  Def<traits::mju_mat2Rot>(pymodule);
  Def<traits::mju_euler2Quat>(pymodule);

  // Poses
  Def<traits::mju_mulPose>(pymodule);
  Def<traits::mju_negPose>(pymodule);
  Def<traits::mju_trnVecPose>(pymodule);

  // Decompositions
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_cholFactor, "n")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> mat, mjtNum mindiag) {
        if (mat.rows() != mat.cols()) {
          throw py::type_error("mat should be a square matrix");
        }
        return InterceptMjErrors(::mju_cholFactor)(
            mat.data(), mat.rows(), mindiag);
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_cholSolve, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const EigenArrayXX> mat,
         Eigen::Ref<const EigenVectorX> vec) {
        if (mat.rows() != mat.cols()) {
          throw py::type_error("mat should be a square matrix");
        }
        if (res.size() != mat.rows()) {
          throw py::type_error(
              "size of res should equal the number of rows in mat");
        }
        if (vec.size() != mat.cols()) {
          throw py::type_error(
              "size of vec should equal the number of rows in mat");
        }
        return InterceptMjErrors(::mju_cholSolve)(
            res.data(), mat.data(), vec.data(), mat.rows());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_cholUpdate, "n")(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> mat, Eigen::Ref<EigenVectorX> x,
         int flg_plus) {
        if (mat.rows() != mat.cols()) {
          throw py::type_error("mat should be a square matrix");
        }
        if (x.size() != mat.rows()) {
          throw py::type_error(
              "size of x should equal the number of rows in mat");
        }
        return InterceptMjErrors(::mju_cholUpdate)(
            mat.data(), x.data(), mat.rows(), flg_plus);
      });
  Def<traits::mju_cholFactorBand>(
      pymodule, [](Eigen::Ref<EigenVectorX> mat, int ntotal, int nband,
                   int ndense, mjtNum diagadd, mjtNum diagmul) {
        int nMat = (ntotal - ndense) * nband + ndense * ntotal;
        if (mat.size() != nMat) {
          throw py::type_error(
              "mat must have size (ntotal-ndense)*nband + ndense*ntotal");
        }
        return InterceptMjErrors(::mju_cholFactorBand)(
            mat.data(), ntotal, nband, ndense, diagadd, diagmul);
      });
  Def<traits::mju_cholSolveBand>(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res, Eigen::Ref<const EigenVectorX> mat,
         Eigen::Ref<const EigenVectorX> vec, int ntotal, int nband,
         int ndense) {
        int nMat = (ntotal - ndense) * nband + ndense * ntotal;
        if (mat.size() != nMat) {
          throw py::type_error(
              "mat must have (ntotal-ndense)*nband + "
              "ndense*ntotal elements");
        }
        if (res.size() != ntotal) {
          throw py::type_error("size of res should equal ntotal");
        }
        if (vec.size() != ntotal) {
          throw py::type_error("size of vec should equal ntotal");
        }
        return InterceptMjErrors(::mju_cholSolveBand)(
            res.data(), mat.data(), vec.data(), ntotal, nband, ndense);
      });
  Def<traits::mju_band2Dense>(
      pymodule,
      [](Eigen::Ref<EigenArrayXX> res, Eigen::Ref<const EigenVectorX> mat,
         int ntotal, int nband, int ndense, mjtBool flg_sym) {
        int nMat = (ntotal - ndense) * nband + ndense * ntotal;
        if (mat.size() != nMat) {
          throw py::type_error(
              "mat must have size (ntotal-ndense)*nband + ndense*ntotal");
        }
        if (res.rows() != ntotal) {
          throw py::type_error("res should have ntotal rows");
        }
        if (res.cols() != ntotal) {
          throw py::type_error("res should have ntotal columns");
        }
        return InterceptMjErrors(::mju_band2Dense)(
            res.data(), mat.data(), ntotal, nband, ndense, flg_sym);
      });
  Def<traits::mju_dense2Band>(pymodule, [](Eigen::Ref<EigenVectorX> res,
                                           Eigen::Ref<const EigenArrayXX> mat,
                                           int ntotal, int nband, int ndense) {
    int nRes = (ntotal - ndense) * nband + ndense * ntotal;
    if (res.size() != nRes) {
      throw py::type_error(
          "res must have size (ntotal-ndense)*nband + ndense*ntotal");
    }
    if (mat.rows() != ntotal) {
      throw py::type_error("mat should have ntotal rows");
    }
    if (mat.cols() != ntotal) {
      throw py::type_error("mat should have ntotal columns");
    }
    return InterceptMjErrors(::mju_dense2Band)(res.data(), mat.data(), ntotal,
                                               nband, ndense);
  });
  Def<traits::mju_bandMulMatVec>(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res, Eigen::Ref<const EigenArrayXX> mat,
         Eigen::Ref<const EigenArrayXX> vec, int ntotal, int nband, int ndense,
         int nVec, mjtBool flg_sym) {
        int nMat = (ntotal - ndense) * nband + ndense * ntotal;
        if (mat.size() != nMat) {
          throw py::type_error(
              "mat must have size (ntotal-ndense)*nband + ndense*ntotal");
        }
        if (res.rows() != ntotal) {
          throw py::type_error("res should have ntotal rows");
        }
        if (res.cols() != nVec) {
          throw py::type_error("res should have nVec columns");
        }
        if (vec.rows() != ntotal) {
          throw py::type_error("vec should have ntotal rows");
        }
        if (vec.cols() != nVec) {
          throw py::type_error("vec should have nVec columns");
        }
        return InterceptMjErrors(::mju_bandMulMatVec)(res.data(), mat.data(),
                                                      vec.data(), ntotal, nband,
                                                      ndense, nVec, flg_sym);
      });
  Def<traits::mju_bandDiag>(pymodule);
  Def<traits::mju_eig3>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_boxQP, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<EigenArrayXX> R,
         std::optional<Eigen::Ref<Eigen::Vector<int, Eigen::Dynamic>>> index,
         Eigen::Ref<const EigenArrayXX> H,
         Eigen::Ref<const EigenVectorX> g,
         std::optional<Eigen::Ref<const EigenVectorX>> lower,
         std::optional<Eigen::Ref<const EigenVectorX>> upper) {
        int n = res.size();
        if (R.size() != n*(n+7)) {
          throw py::type_error("size of R should be n*(n+7)");
        }
        if (index.has_value() && (index->size() != n)) {
          throw py::type_error("size of index should equal n");
        }
        if (H.rows() != n || H.cols() != n) {
          throw py::type_error("H should be of shape (n, n)");
        }
        if (g.size() != n) {
          throw py::type_error("size of g should equal n");
        }
        if (lower.has_value() && (lower->size() != n)) {
          throw py::type_error("size of lower should equal n");
        }
        if (upper.has_value() && (upper->size() != n)) {
          throw py::type_error("size of upper should equal n");
        }
        return InterceptMjErrors(::mju_boxQP)(
            res.data(), R.data(),
            index.has_value() ? index->data() : nullptr,
            H.data(), g.data(), n,
            lower.has_value() ? lower->data() : nullptr,
            upper.has_value() ? upper->data() : nullptr);
      });
  // Miscellaneous
  Def<traits::mju_muscleGain>(pymodule);
  Def<traits::mju_muscleBias>(pymodule);
  Def<traits::mju_muscleDynamics>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_encodePyramid, "dim")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> pyramid, Eigen::Ref<const EigenVectorX> force,
         Eigen::Ref<const EigenVectorX> mu) {
        if (pyramid.size() != 2*mu.size()) {
          throw py::type_error(
              "size of pyramid should be twice as large as size of mu");
        }
        if (force.size() != mu.size() + 1) {
          throw py::type_error(
              "size of force should be exactly one larger than size of mu");
        }
        return InterceptMjErrors(::mju_encodePyramid)(
            pyramid.data(), force.data(), mu.data(), mu.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_decodePyramid, "dim")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> force, Eigen::Ref<const EigenVectorX> pyramid,
         Eigen::Ref<const EigenVectorX> mu) {
        if (pyramid.size() != 2*mu.size()) {
          throw py::type_error(
              "size of pyramid should be twice as large as size of mu");
        }
        if (force.size() != mu.size() + 1) {
          throw py::type_error(
              "size of force should be exactly one larger than size of mu");
        }
        return InterceptMjErrors(::mju_decodePyramid)(
            force.data(), pyramid.data(), mu.data(), mu.size());
      });
  Def<traits::mju_springDamper>(pymodule);
  Def<traits::mju_min>(pymodule);
  Def<traits::mju_max>(pymodule);
  Def<traits::mju_clip>(pymodule);
  Def<traits::mju_sign>(pymodule);
  Def<traits::mju_round>(pymodule);
  Def<traits::mju_type2Str>(pymodule);
  Def<traits::mju_str2Type>(pymodule);
  Def<traits::mju_writeNumBytes>(pymodule);
  Def<traits::mju_warningText>(pymodule);
  Def<traits::mju_isBad>(pymodule);
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_isZero, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> vec) {
        return InterceptMjErrors(::mju_isZero)(vec.data(), vec.size());
      });
  Def<traits::mju_standardNormal>(
      pymodule,
      [](std::optional<mjtNum> num2) {
        return InterceptMjErrors(::mju_standardNormal)(
            num2.has_value() ? &*num2 : nullptr);
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_f2n, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const Eigen::Vector<float, Eigen::Dynamic>> vec) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_f2n)(res.data(), vec.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_n2f, "n")(
      pymodule,
      [](Eigen::Ref<Eigen::Vector<float, Eigen::Dynamic>> res,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_n2f)(res.data(), vec.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_d2n, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res,
         Eigen::Ref<const Eigen::Vector<double, Eigen::Dynamic>> vec) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_d2n)(res.data(), vec.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_n2d, "n")(
      pymodule,
      [](Eigen::Ref<Eigen::Vector<double, Eigen::Dynamic>> res,
         Eigen::Ref<const EigenVectorX> vec) {
        if (res.size() != vec.size()) {
          throw py::type_error("res and vec should have the same size");
        }
        return InterceptMjErrors(::mju_n2d)(res.data(), vec.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_insertionSort, "n")(
      pymodule,
      [](Eigen::Ref<EigenVectorX> res) {
        return InterceptMjErrors(::mju_insertionSort)(res.data(), res.size());
      });
  DEF_WITH_OMITTED_PY_ARGS(traits::mju_insertionSortInt, "n")(
      pymodule,
      [](Eigen::Ref<Eigen::Vector<int, Eigen::Dynamic>> res) {
        return InterceptMjErrors(::mju_insertionSortInt)(
            res.data(), res.size());
      });
  Def<traits::mju_Halton>(pymodule);
  // Skipped: mju_strncpy (doesn't make sense in Python)
  Def<traits::mju_sigmoid>(pymodule);
  Def<traits::mj_insideSite>(pymodule);

  // Derivatives
  Def<traits::mjd_transitionFD>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         mjtNum eps, mjtBool flg_centered,
         std::optional<Eigen::Ref<EigenArrayXX>> A,
         std::optional<Eigen::Ref<EigenArrayXX>> B,
         std::optional<Eigen::Ref<EigenArrayXX>> C,
         std::optional<Eigen::Ref<EigenArrayXX>> D) {
        if (A.has_value() &&
            (A->rows() != 2*m->nv+m->na || A->cols() != 2*m->nv+m->na)) {
          throw py::type_error("A should be of shape (2*nv+na, 2*nv+na)");
        }
        if (B.has_value() &&
            (B->rows() != 2*m->nv+m->na || B->cols() != m->nu)) {
          throw py::type_error("B should be of shape (2*nv+na, nu)");
        }
        if (C.has_value() &&
            (C->rows() != m->nsensordata || C->cols() != 2*m->nv+m->na)) {
          throw py::type_error("C should be of shape (nsensordata, 2*nv+na)");
        }
        if (D.has_value() &&
            (D->rows() != m->nsensordata || D->cols() != m->nu)) {
          throw py::type_error("D should be of shape (nsensordata, nu)");
        }
        return InterceptMjErrors(::mjd_transitionFD)(
            m, d, eps, flg_centered,
            A.has_value() ? A->data() : nullptr,
            B.has_value() ? B->data() : nullptr,
            C.has_value() ? C->data() : nullptr,
            D.has_value() ? D->data() : nullptr);
      });
  Def<traits::mjd_inverseFD>(
      pymodule,
      [](const raw::MjModel* m, raw::MjData* d,
         mjtNum eps, mjtBool flg_actuation,
         std::optional<Eigen::Ref<EigenArrayXX>> DfDq,
         std::optional<Eigen::Ref<EigenArrayXX>> DfDv,
         std::optional<Eigen::Ref<EigenArrayXX>> DfDa,
         std::optional<Eigen::Ref<EigenArrayXX>> DsDq,
         std::optional<Eigen::Ref<EigenArrayXX>> DsDv,
         std::optional<Eigen::Ref<EigenArrayXX>> DsDa,
         std::optional<Eigen::Ref<EigenArrayXX>> DmDq) {
        if (DfDq.has_value() &&
            (DfDq->rows() != m->nv || DfDq->cols() != m->nv)) {
          throw py::type_error("DfDq should be of shape (nv, nv)");
        }
        if (DfDv.has_value() &&
            (DfDv->rows() != m->nv || DfDv->cols() != m->nv)) {
          throw py::type_error("DfDv should be of shape (nv, nv)");
        }
        if (DfDa.has_value() &&
            (DfDa->rows() != m->nv || DfDa->cols() != m->nv)) {
          throw py::type_error("DfDa should be of shape (nv, nv)");
        }
        if (DsDq.has_value() &&
            (DsDq->rows() != m->nv || DsDq->cols() != m->nsensordata)) {
          throw py::type_error("DsDq should be of shape (nv, nsensordata)");
        }
        if (DsDv.has_value() &&
            (DsDv->rows() != m->nv || DsDv->cols() != m->nsensordata)) {
          throw py::type_error("DsDv should be of shape (nv, nsensordata)");
        }
        if (DsDa.has_value() &&
            (DsDa->rows() != m->nv || DsDa->cols() != m->nsensordata)) {
          throw py::type_error("DsDa should be of shape (nv, nsensordata)");
        }
        if (DmDq.has_value() &&
            (DmDq->rows() != m->nv || DmDq->cols() != m->nC)) {
          throw py::type_error("DmDq should be of shape (nv, nC)");
        }
        return InterceptMjErrors(::mjd_inverseFD)(
            m, d, eps, flg_actuation,
            DfDq.has_value() ? DfDq->data() : nullptr,
            DfDv.has_value() ? DfDv->data() : nullptr,
            DfDa.has_value() ? DfDa->data() : nullptr,
            DsDq.has_value() ? DsDq->data() : nullptr,
            DsDv.has_value() ? DsDv->data() : nullptr,
            DsDa.has_value() ? DsDa->data() : nullptr,
            DmDq.has_value() ? DmDq->data() : nullptr);
      });
  Def<traits::mjd_subQuat>(
      pymodule,
      [](Eigen::Ref<const EigenVectorX> qa, Eigen::Ref<const EigenVectorX> qb,
         std::optional<Eigen::Ref<EigenArrayXX>> Da,
         std::optional<Eigen::Ref<EigenArrayXX>> Db) {
        if (qa.size() != 4) {
          throw py::type_error("qa must have size 4");
        }
        if (qb.size() != 4) {
          throw py::type_error("qb must have size 4");
        }
        if (Da.has_value() && Da->size() != 9) {
          throw py::type_error("Da must have size 9");
        }
        if (Db.has_value() && Db->size() != 9) {
          throw py::type_error("Db must have size 9");
        }
        return InterceptMjErrors(::mjd_subQuat)(
            qa.data(), qb.data(),
            Da.has_value() ? Da->data() : nullptr,
            Db.has_value() ? Db->data() : nullptr);
      });
  Def<traits::mjd_quatIntegrate>(pymodule);

  pymodule.def(
      "_realloc_con_efc",
      [](MjDataWrapper& d, int ncon, int nefc, int nJ) {
        raw::MjData* data = d.get();

        // discard all contacts and constraint rows: like mj_clearEfc, clear the
        // arena pointers and every count that sizes or indexes them
        auto cleanup = [](raw::MjData* data) {
#ifdef ADDRESS_SANITIZER
        ASAN_POISON_MEMORY_REGION(
            static_cast<char*>(data->arena),
            data->narena - data->pstack);
#endif
          data->parena = 0;
          data->ncon = 0;
          data->ne = data->nf = data->nl = data->nefc = 0;
          data->nisland = data->nidof = 0;
          data->nJ = data->nY = data->nA = 0;
          data->efm_active = 0;
          data->nefmT = data->nefmA = data->nefmK = data->nefmL = 0;
          data->nefmdof = data->nefmcon = 0;
          data->contact = static_cast<raw::MjContact*>(data->arena);
#define X(type, name, nr, nc) data->name = nullptr;
          MJDATA_ARENA_POINTERS_SOLVER
          MJDATA_ARENA_POINTERS_DUAL
          MJDATA_ARENA_POINTERS_ISLAND
          MJDATA_ARENA_POINTERS_EFM
#undef X
        };

        char error_msg[128];
        error_msg[0] = '\0';
        const char* error_msg_fmt =
            "Insufficient arena memory, currently allocated memory=\"%s\". "
            "Increase using <size memory=\"X\"/>.";
        if (nJ < 0) nJ = data->nJ;  // by default, keep the current size
        cleanup(data);
        data->ncon = ncon;
        data->nefc = nefc;
        data->nJ = nJ;
        data->contact =
            static_cast<raw::MjContact*>(InterceptMjErrors(::mj_arenaAllocByte)(
                data, ncon * sizeof(raw::MjContact), alignof(raw::MjContact)));
        if (!data->contact) {
          cleanup(data);
          std::snprintf(error_msg, sizeof(error_msg), error_msg_fmt,
                        mju_writeNumBytes(data->narena));
          throw FatalError(error_msg);
        }

#undef MJ_M
#define MJ_M(x) d.model().get()->x
#undef MJ_D
#define MJ_D(x) data->x
#define X(type, name, nr, nc)                                             \
  data->name = static_cast<type*>(InterceptMjErrors(::mj_arenaAllocByte)( \
      data, sizeof(type) * (nr) * (nc), alignof(type)));                  \
  if (!data->name) {                                                      \
    cleanup(data);                                                        \
    std::snprintf(error_msg, sizeof(error_msg), error_msg_fmt,            \
                  mju_writeNumBytes(data->narena));                       \
    throw FatalError(error_msg);                                          \
  }

        MJDATA_ARENA_POINTERS_SOLVER
        if (mj_isDual(d.model().get())) {
          MJDATA_ARENA_POINTERS_DUAL
        }
#undef X
#undef MJ_D
#define MJ_D(x) x
#undef MJ_M
#define MJ_M(x) x
      },
      py::arg("d"), py::arg("ncon"), py::arg("nefc"), py::arg("nJ") = -1,
      py::call_guard<py::gil_scoped_release>());

  pymodule.def(
      "_realloc_island",
      [](MjDataWrapper& d, int nisland, int nidof) {
        raw::MjData* data = d.get();

        size_t parena_start = data->parena;
        // Find island block start in arena to reclaim memory on re-allocation.
        char* min_ptr = nullptr;
#define X(type, name, nr, nc)                                        \
  if (data->name &&                                                  \
      (!min_ptr || reinterpret_cast<char*>(data->name) < min_ptr)) { \
    min_ptr = reinterpret_cast<char*>(data->name);                   \
  }
        MJDATA_ARENA_POINTERS_ISLAND
#undef X

#undef MJ_M
#define MJ_M(x) d.model().get()->x
#undef MJ_D
#define MJ_D(x) data->x
        // Reclaim the island block only if no other arena array follows it; a
        // forward pass allocates the dual and effective-metric arrays after it.
        if (min_ptr && data->arena) {
          bool followed = false;
#define X(type, name, nr, nc)                           \
  if (data->name && (nr) * (nc) > 0 &&                  \
      reinterpret_cast<char*>(data->name) >= min_ptr) { \
    followed = true;                                    \
  }
          MJDATA_ARENA_POINTERS_CONTACT
          MJDATA_ARENA_POINTERS_SOLVER
          MJDATA_ARENA_POINTERS_DUAL
          MJDATA_ARENA_POINTERS_EFM
#undef X
          if (!followed) {
            parena_start = min_ptr - static_cast<char*>(data->arena);
          }
        }

        auto cleanup = [](raw::MjData* data, size_t target_parena) {
#define X(type, name, nr, nc) data->name = nullptr;
          MJDATA_ARENA_POINTERS_ISLAND
#undef X
          data->nisland = 0;
          data->nidof = 0;
          data->parena = target_parena;
#ifdef ADDRESS_SANITIZER
          ASAN_POISON_MEMORY_REGION(
              static_cast<char*>(data->arena) + target_parena,
              data->narena - data->pstack - target_parena);
#endif
        };

        cleanup(data, parena_start);

        char error_msg[128];
        error_msg[0] = '\0';
        const char* error_msg_fmt =
            "Insufficient arena memory, currently allocated memory=\"%s\". "
            "Increase using <size memory=\"X\"/>.";

        data->nisland = nisland;
        data->nidof = nidof;

#define X(type, name, nr, nc)                                             \
  data->name = static_cast<type*>(InterceptMjErrors(::mj_arenaAllocByte)( \
      data, sizeof(type) * (nr) * (nc), alignof(type)));                  \
  if (!data->name) {                                                      \
    cleanup(data, parena_start);                                          \
    std::snprintf(error_msg, sizeof(error_msg), error_msg_fmt,            \
                  mju_writeNumBytes(data->narena));                       \
    throw FatalError(error_msg);                                          \
  }

        MJDATA_ARENA_POINTERS_ISLAND
#undef X
#undef MJ_D
#define MJ_D(x) x
#undef MJ_M
#define MJ_M(x) x
      },
      py::arg("d"), py::arg("nisland"), py::arg("nidof"),
      py::call_guard<py::gil_scoped_release>());
}  // PYBIND11_MODULE NOLINT(readability/fn_size)
}  // namespace
}  // namespace mujoco::python
