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

#include "structs.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "errors.h"
#include "function_traits.h"
#include "indexer_xmacro.h"
#include "indexers.h"
#include "raw.h"
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include "vfs.h"

namespace mujoco::python::_impl {

namespace py = ::pybind11;

namespace {
#define PTRDIFF(x, y) \
  reinterpret_cast<const char*>(x) - reinterpret_cast<const char*>(y)

// Returns the shape of a NumPy array given the dimensions from an X Macro.
// If dim1 is a _literal_ constant 1, the resulting array is 1-dimensional of
// length dim0, otherwise the resulting array is 2-dimensional of shape
// (dim0, dim1).
#define X_ARRAY_SHAPE(dim0, dim1) XArrayShapeImpl(#dim1)((dim0), (dim1))

std::vector<mjtSize> XArrayShapeImpl1D(mjtSize dim0, mjtSize dim1) {
  return {dim0};
}

std::vector<mjtSize> XArrayShapeImpl2D(mjtSize dim0, mjtSize dim1) {
  return {dim0, dim1};
}

constexpr auto XArrayShapeImpl(const std::string_view dim1_str) {
  if (dim1_str == "1") {
    return XArrayShapeImpl1D;
  } else {
    return XArrayShapeImpl2D;
  }
}

// Wraps an arena array, or returns an array of the same shape filled with
// `fill` if it is not allocated (e.g. efm_* arrays when the discrete integrator
// is not used).
template <typename T>
py::array_t<T> InitArenaArray(const std::vector<mjtSize>& shape, T* buf,
                              py::handle owner, T fill) {
  if (!buf) {
    py::array_t<T> arr(shape);
    std::fill_n(arr.mutable_data(), arr.size(), fill);
    return arr;
  }
  return InitPyArray(shape, buf, owner);
}

// arena fields where -1 means "none"
constexpr bool IsNoneIndex(std::string_view name) {
  return name == "tendon_efcadr" || name == "tree_island" ||
         name == "dof_island";
}

py::tuple RecompileSpec(raw::MjSpec* spec, const mjVFS* vfs,
                        const MjModelWrapper& old_m,
                        const MjDataWrapper& old_d) {
  raw::MjModel* m = static_cast<raw::MjModel*>(mju_malloc(sizeof(mjModel)));
  m->buffer = nullptr;
  raw::MjData* d = mj_copyData(nullptr, old_m.get(), old_d.get());

  bool compile_failed = false;

  {
    // Release GIL before calling mj_recompile which may spawn threads
    py::gil_scoped_release no_gil;
    if (mj_recompile(spec, vfs, m, d)) {
      compile_failed = true;
    }
  }

  if (compile_failed) {
    throw py::value_error(mjs_getError(spec));
  }

  py::object m_pyobj = py::cast((MjModelWrapper(m)));
  py::object d_pyobj =
      py::cast((MjDataWrapper(py::cast<MjModelWrapper*>(m_pyobj), d)));
  return py::make_tuple(m_pyobj, d_pyobj);
}

// properties of a group of named views that are not views of a field
template <typename GroupedViews>
void DefineComputedViews(py::class_<GroupedViews>&) {}

// an actuator's rows of the sparse actuator_moment, as a dense read-only copy:
// (nv,) for an actuator with one force output, else (outnum, nv)
template <>
void DefineComputedViews(py::class_<MjDataActuatorViews>& views) {
  views.def_property_readonly("moment", [](MjDataActuatorViews& v) {
    const raw::MjModel* m = v.model();
    const raw::MjData* d = v.ptr();
    int adr = m->actuator_outadr[v.index()];
    int num = m->actuator_outnum[v.index()];
    std::vector<py::ssize_t> shape = {num, static_cast<py::ssize_t>(m->nv)};
    if (num == 1) shape.erase(shape.begin());
    py::array_t<mjtNum> out(shape);
    mju_sparse2dense(out.mutable_data(), d->actuator_moment, num, m->nv,
                     d->moment_rownnz + adr, d->moment_rowadr + adr,
                     d->moment_colind);
    out.attr("flags").attr("writeable") = false;
    return out;
  });
}

}  // namespace

PYBIND11_MODULE(_structs, m, pybind11::mod_gil_not_used()) {
  py::module_::import("mujoco._enums");

  // We import _callbacks so that a default timer is installed.
  try {
    py::module_::import("mujoco._callbacks");
  } catch (py::error_already_set&) {
    // In stubgen or isolated extension loading, _callbacks may not be present.
  }

  // ==================== MJOPTION =============================================
  py::class_<MjOptionWrapper> mjOption(m, "MjOption");
  mjOption.def(py::init<>());
  mjOption.def("__copy__", [](const MjOptionWrapper& other) {
    return MjOptionWrapper(other);
  });
  mjOption.def("__deepcopy__", [](const MjOptionWrapper& other, py::dict) {
    return MjOptionWrapper(other);
  });
  mjOption.def_property_readonly_static("_all_fields", [](py::object) {
    std::vector<std::string> fields;
#define X(type, var, dim) fields.push_back(#var);
#define XVEC X
    MJOPTION_FIELDS
#undef XVEC
#undef X
    return py::tuple(py::cast(fields));
  });
  DefineStructFunctions(mjOption);

#define X(type, var, dim)                                          \
  mjOption.def_property(                                           \
      #var, [](const MjOptionWrapper& c) { return c.get()->var; }, \
      [](MjOptionWrapper& c, type rhs) { c.get()->var = rhs; });
#define XVEC(type, var, dim) \
  DefinePyArray(mjOption, #var, &MjOptionWrapper::var);
  MJOPTION_FIELDS
#undef XVEC
#undef X

  mjOption.def_property_readonly_static("_float_fields", [](py::object) {
    std::vector<std::string> field_names;
#define X(type, var, dim) \
  if constexpr (std::is_floating_point_v<type>) field_names.push_back(#var);
#define XVEC(type, var, dim)
    MJOPTION_FIELDS
#undef XVEC
#undef X
    return py::tuple(py::cast(field_names));
  });

  mjOption.def_property_readonly_static("_int_fields", [](py::object) {
    std::vector<std::string> field_names;
#define X(type, var, dim) \
  if constexpr (std::is_integral_v<type>) field_names.push_back(#var);
#define XVEC(type, var, dim)
    MJOPTION_FIELDS
#undef XVEC
#undef X
    return py::tuple(py::cast(field_names));
  });

  mjOption.def_property_readonly_static("_floatarray_fields", [](py::object) {
    std::vector<std::string> field_names;
#define X(type, var, dim)
#define XVEC(type, var, dim) field_names.push_back(#var);
    MJOPTION_FIELDS
#undef XVEC
#undef X
    return py::tuple(py::cast(field_names));
  });

  // ==================== MJVISUAL =============================================
  py::class_<MjVisualWrapper> mjVisual(m, "MjVisual");
  mjVisual.def("__copy__", [](const MjVisualWrapper& other) {
    return MjVisualWrapper(other);
  });
  mjVisual.def("__deepcopy__", [](const MjVisualWrapper& other, py::dict) {
    return MjVisualWrapper(other);
  });
  mjVisual.def("__eq__", StructsEqual<MjVisualWrapper>);
  // Special __repr__ implementation for MjVisual, since:
  // 1. Types under MjVisual confuse StructRepr;
  // 2. StructRepr does not handle the indentation of nested structs well.
  mjVisual.def("__repr__", [](py::object self) {
    std::ostringstream result;
    result << "<"
           << self.attr("__class__").attr("__name__").cast<std::string_view>();

#define X(type, var)          \
  result << "\n  " #var ": "; \
  StructReprImpl<type>(self.attr(#var), result, 2);

    X(raw::MjVisualGlobal, global_)
    X(raw::MjVisualQuality, quality)
    X(MjVisualHeadlightWrapper, headlight)
    X(raw::MjVisualMap, map)
    X(raw::MjVisualScale, scale)
    X(MjVisualRgbaWrapper, rgba)
#undef X
    result << "\n>";
    return result.str();
  });

  py::class_<raw::MjVisualGlobal> mjVisualGlobal(mjVisual, "Global");
  mjVisualGlobal.def("__copy__", [](const raw::MjVisualGlobal& other) {
    return raw::MjVisualGlobal(other);
  });
  mjVisualGlobal.def("__deepcopy__",
                     [](const raw::MjVisualGlobal& other, py::dict) {
                       return raw::MjVisualGlobal(other);
                     });
  DefineStructFunctions(mjVisualGlobal);
#define X(type, var, dim) \
  mjVisualGlobal.def_readwrite(#var, &raw::MjVisualGlobal::var);
  MJVISUAL_GLOBAL_FIELDS
#undef X

  py::class_<raw::MjVisualQuality> mjVisualQuality(mjVisual, "Quality");
  mjVisualQuality.def("__copy__", [](const raw::MjVisualQuality& other) {
    return raw::MjVisualQuality(other);
  });
  mjVisualQuality.def("__deepcopy__",
                      [](const raw::MjVisualQuality& other, py::dict) {
                        return raw::MjVisualQuality(other);
                      });
  DefineStructFunctions(mjVisualQuality);
#define X(type, var, dim) \
  mjVisualQuality.def_readwrite(#var, &raw::MjVisualQuality::var);
  MJVISUAL_QUALITY_FIELDS
#undef X

  py::class_<MjVisualHeadlightWrapper> mjVisualHeadlight(mjVisual, "Headlight");
  mjVisualHeadlight.def("__copy__", [](const MjVisualHeadlightWrapper& other) {
    return MjVisualHeadlightWrapper(other);
  });
  mjVisualHeadlight.def("__deepcopy__",
                        [](const MjVisualHeadlightWrapper& other, py::dict) {
                          return MjVisualHeadlightWrapper(other);
                        });
  DefineStructFunctions(mjVisualHeadlight);
#define X(type, var, dim)                                                   \
  mjVisualHeadlight.def_property(                                           \
      #var, [](const MjVisualHeadlightWrapper& c) { return c.get()->var; }, \
      [](MjVisualHeadlightWrapper& c, type rhs) {                           \
        return c.get()->var = rhs;                                          \
      });
#define XVEC(type, var, dim) \
  DefinePyArray(mjVisualHeadlight, #var, &MjVisualHeadlightWrapper::var);
  MJVISUAL_HEADLIGHT_FIELDS
#undef XVEC
#undef X

  py::class_<raw::MjVisualMap> mjVisualMap(mjVisual, "Map");
  mjVisualMap.def("__copy__", [](const raw::MjVisualMap& other) {
    return raw::MjVisualMap(other);
  });
  mjVisualMap.def("__deepcopy__", [](const raw::MjVisualMap& other, py::dict) {
    return raw::MjVisualMap(other);
  });
  DefineStructFunctions(mjVisualMap);
#define X(type, var, dim) \
  mjVisualMap.def_readwrite(#var, &raw::MjVisualMap::var);
  MJVISUAL_MAP_FIELDS
#undef X

  py::class_<raw::MjVisualScale> mjVisualScale(mjVisual, "Scale");
  mjVisualScale.def("__copy__", [](const raw::MjVisualScale& other) {
    return raw::MjVisualScale(other);
  });
  mjVisualScale.def("__deepcopy__",
                    [](const raw::MjVisualScale& other, py::dict) {
                      return raw::MjVisualScale(other);
                    });
  DefineStructFunctions(mjVisualScale);
#define X(type, var, dim) \
  mjVisualScale.def_readwrite(#var, &raw::MjVisualScale::var);
  MJVISUAL_SCALE_FIELDS
#undef X

  py::class_<MjVisualRgbaWrapper> mjVisualRgba(mjVisual, "Rgba");
  mjVisualRgba.def("__copy__", [](const MjVisualRgbaWrapper& other) {
    return MjVisualRgbaWrapper(other);
  });
  mjVisualRgba.def("__deepcopy__",
                   [](const MjVisualRgbaWrapper& other, py::dict) {
                     return MjVisualRgbaWrapper(other);
                   });
  DefineStructFunctions(mjVisualRgba);
#define XVEC(type, var, dim) \
  DefinePyArray(mjVisualRgba, #var, &MjVisualRgbaWrapper::var);
  MJVISUAL_RGBA_FIELDS
#undef XVEC

#define X(var)                                                              \
  mjVisual.def_property(                                                    \
      #var, [](const MjVisualWrapper& c) -> auto& { return c.get()->var; }, \
      [](MjVisualWrapper& c, const decltype(raw::MjVisual::var)& rhs) {     \
        c.get()->var = rhs;                                                 \
      });
  // mjVisual.global is exposed as "global_" to avoid clash with the Python
  // keyword.
  mjVisual.def_property(
      "global_",
      [](const MjVisualWrapper& c) -> auto& { return c.get()->global; },
      [](MjVisualWrapper& c, const raw::MjVisualGlobal& rhs) {
        c.get()->global = rhs;
      });
  X(quality);
  DefinePyStruct(mjVisual, "headlight", &MjVisualWrapper::headlight);
  X(map);
  X(scale);
  DefinePyStruct(mjVisual, "rgba", &MjVisualWrapper::rgba);
#undef X

  // ==================== MJMODEL ==============================================
  py::class_<MjModelWrapper> mjModel(m, "MjModel");
  mjModel.def_static(
      "from_xml_string", MjModelWrapper::LoadXML, py::arg("xml"),
      py::arg_v("assets", py::none()), py::arg("vfs") = py::none(),
      py::doc(
          R"(Loads an MjModel from an XML string and optional assets dict or VFS.)"));
  mjModel.def_static("_from_model_ptr", [](uintptr_t addr) {
    return MjModelWrapper::WrapRawModel(reinterpret_cast<raw::MjModel*>(addr));
  });
  mjModel.def_static(
      "from_xml_path", MjModelWrapper::LoadXMLFile, py::arg("filename"),
      py::arg_v("assets", py::none()), py::arg("vfs") = py::none(),
      py::doc(
          R"(Loads an MjModel from an XML file and optional assets dict or VFS.)"));
  mjModel.def_static(
      "from_binary_path", &MjModelWrapper::LoadBinaryFile, py::arg("filename"),
      py::arg_v("assets", py::none()), py::arg("vfs") = py::none(),
      py::doc(
          R"(Loads an MjModel from an MJB file and an optional assets dictionary.

The filename for the MJB can also refer to a key in the assets dictionary.
This is useful for example when the MJB is not available as a file on disk.)"));
  mjModel.def_property_readonly("_address", [](const MjModelWrapper& m) {
    return reinterpret_cast<std::uintptr_t>(m.get());
  });
  mjModel.def("__copy__", [](const MjModelWrapper& other) {
    return MjModelWrapper(other);
  });
  mjModel.def("__deepcopy__", [](const MjModelWrapper& other, py::dict) {
    return MjModelWrapper(other);
  });
  mjModel.def(py::pickle(
      [](const MjModelWrapper& m) {  // __getstate__
        std::ostringstream output(std::ios::out | std::ios::binary);
        m.Serialize(output);
        return py::bytes(output.str());
      },
      [](py::bytes b) {  // __setstate__
        std::istringstream input(b, std::ios::in | std::ios::binary);
        return MjModelWrapper::Deserialize(input);
      }));

  DefinePyStruct(mjModel, "opt", &MjModelWrapper::opt);
  DefinePyStruct(mjModel, "vis", &MjModelWrapper::vis);
  DefinePyStruct(mjModel, "stat", &MjModelWrapper::stat);

  mjModel.def_property(
      "flg_gravcomp",
      [](const MjModelWrapper& m) { return m.get()->flg_gravcomp; },
      [](MjModelWrapper& m, bool val) {
        m.get()->flg_gravcomp = val;
        m.get()->ngravcomp = val ? 1 : 0;
      });

  mjModel.def_property(
      "flg_surfacevel",
      [](const MjModelWrapper& m) { return m.get()->flg_surfacevel; },
      [](MjModelWrapper& m, bool val) {
        m.get()->flg_surfacevel = val;
      });

  mjModel.def_property(
      "flg_adhesion",
      [](const MjModelWrapper& m) { return m.get()->flg_adhesion; },
      [](MjModelWrapper& m, bool val) {
        m.get()->flg_adhesion = val;
      });

#define X(var)                   \
  mjModel.def_property_readonly( \
      #var, [](const MjModelWrapper& m) { return m.get()->var; });
  MJMODEL_SIZES
#undef X

  mjModel.def_property_readonly("_sizes", [](const MjModelWrapper& m) {
    int nint = 0;
#define X(var) ++nint;
    MJMODEL_SIZES
#undef X
    py::array_t<std::int64_t> sizes(nint);
    {
      int i = 0;
      auto data = sizes.mutable_unchecked();
#define X(var) data[i++] = m.get()->var;
      MJMODEL_SIZES
#undef X
    }
    py::detail::array_proxy(sizes.ptr())->flags &=
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return sizes;
  });

  mjModel.def_property_readonly_static("_size_fields", [](py::object) {
    std::vector<std::string> fields;
#define X(var) fields.push_back(#var);
    MJMODEL_SIZES
#undef X
    return py::tuple(py::cast(fields));
  });

  mjModel.def_property_readonly_static("_all_fields", [](py::object) {
    std::vector<std::string> fields;
#define X(var) fields.push_back(#var);
    MJMODEL_SIZES
#undef X
#define X(type, name, nr, nc) fields.push_back(#name);
    MJMODEL_POINTERS
#undef X
    return py::tuple(py::cast(fields));
  });

  // arrays which mj_setConst uses, with how each group of its computations
  // (fixed, kin0, inert0, spring) uses them: 0, mjSC_IN, mjSC_OUT or mjSC_INOUT
  mjModel.def_property_readonly_static("_setconst_fields", [](py::object) {
    py::dict fields;
#define X(name, fixed, kin0, inert0, spring) \
    fields[#name] = py::make_tuple(fixed, kin0, inert0, spring);
    MJSETCONST_ARRAYS
#undef X
    return fields;
  });

  // the groups of mj_setConst computations, in the order of the columns
  mjModel.def_property_readonly_static("_setconst_groups", [](py::object) {
    return py::make_tuple("fixed", "kin0", "inert0", "spring");
  });

  // arrays which are inputs of mj_setConst: the first group which uses them
  // reads them before writing them
  mjModel.def_property_readonly_static("_setconst_inputs", [](py::object) {
    std::vector<std::string> inputs;
#define X(name, fixed, kin0, inert0, spring)                    \
    for (int access : {fixed, kin0, inert0, spring}) {          \
      if (access) {                                             \
        if (access & mjSC_IN) inputs.push_back(#name);          \
        break;                                                  \
      }                                                         \
    }
    MJSETCONST_ARRAYS
#undef X
    return py::tuple(py::cast(inputs));
  });

  // arrays which mj_setConst writes
  mjModel.def_property_readonly_static("_setconst_outputs", [](py::object) {
    std::vector<std::string> outputs;
#define X(name, fixed, kin0, inert0, spring) \
    if ((fixed | kin0 | inert0 | spring) & mjSC_OUT) outputs.push_back(#name);
    MJSETCONST_ARRAYS
#undef X
    return py::tuple(py::cast(outputs));
  });

  // scalars which mj_setConst writes
  mjModel.def_property_readonly_static("_setconst_scalars", [](py::object) {
    std::vector<std::string> scalars;
#define X(name) scalars.push_back(#name);
    MJSETCONST_SCALARS
#undef X
    return py::tuple(py::cast(scalars));
  });

#define X(dtype, var, dim0, dim1)                        \
  if constexpr (std::string_view(#var) != "text_data" && \
                std::string_view(#var) != "names" &&     \
                std::string_view(#var) != "paths") {     \
    DefinePyArray(mjModel, #var, &MjModelWrapper::var);  \
  }
  MJMODEL_POINTERS
#undef X

  mjModel.def_property_readonly("text_data",
                                [](const MjModelWrapper& m) -> const auto& {
                                  // Return the full bytes array of concatenated
                                  // text data
                                  return m.text_data_bytes;
                                });
  mjModel.def_property_readonly("names",
                                [](const MjModelWrapper& m) -> const auto& {
                                  // Return the full bytes array of concatenated
                                  // names
                                  return m.names_bytes;
                                });
  mjModel.def_property_readonly("paths",
                                [](const MjModelWrapper& m) -> const auto& {
                                  // Return the full bytes array of concatenated
                                  // paths
                                  return m.paths_bytes;
                                });
  mjModel.def_property_readonly("signature",
                                [](const MjModelWrapper& m) -> const uint64_t& {
                                  return m.get()->signature;
                                });

#define XGROUP(MjModelGroupedViews, field, nfield, FIELD_XMACROS)             \
  mjModel.def(                                                                \
      #field,                                                                 \
      [](MjModelWrapper& m, int i) -> auto& { return m.indexer().field(i); }, \
      py::return_value_policy::reference_internal);                           \
  mjModel.def(                                                                \
      #field,                                                                 \
      [](MjModelWrapper& m, std::string_view name) -> auto& {                 \
        return m.indexer().field##_by_name(name);                             \
      },                                                                      \
      py::return_value_policy::reference_internal, py::arg_v("name", ""));

  MJMODEL_VIEW_GROUPS
#undef XGROUP

#define XGROUP(spectype, field)                                      \
  mjModel.def(                                                       \
      "bind_scalar",                                                 \
      [](MjModelWrapper& m, spectype& spec) -> auto& {               \
        if (mjs_getSpec(spec.element)->element->signature !=         \
            m.get()->signature) {                                    \
          throw py::value_error(                                     \
              "The mjSpec does not match mjModel. Please recompile " \
              "the mjSpec.");                                        \
        }                                                            \
        return m.indexer().field(mjs_getId(spec.element));           \
      },                                                             \
      py::return_value_policy::reference_internal,                   \
      py::arg_v("spec", py::none()));

  MJMODEL_BIND_GROUPS
#undef XGROUP

#define XGROUP(field, altname, FIELD_XMACROS)                                 \
  mjModel.def(                                                                \
      #altname,                                                               \
      [](MjModelWrapper& m, int i) -> auto& { return m.indexer().field(i); }, \
      py::return_value_policy::reference_internal);                           \
  mjModel.def(                                                                \
      #altname,                                                               \
      [](MjModelWrapper& m, std::string_view name) -> auto& {                 \
        return m.indexer().field##_by_name(name);                             \
      },                                                                      \
      py::return_value_policy::reference_internal, py::arg_v("name", ""));

  MJMODEL_VIEW_GROUPS_ALTNAMES
#undef XGROUP

#define XGROUP(MjModelGroupedViews, field, nfield, FIELD_XMACROS)              \
  {                                                                            \
    using GroupedViews = MjModelGroupedViews;                                  \
    py::class_<MjModelGroupedViews> groupedViews(m, "_" #MjModelGroupedViews); \
    FIELD_XMACROS                                                              \
    groupedViews.def("__repr__", MjModelStructRepr<GroupedViews>);             \
    groupedViews.def_property_readonly(                                        \
        "id", [](GroupedViews& views) { return views.index(); });              \
    groupedViews.def_property_readonly(                                        \
        "name", [](GroupedViews& views) { return views.name(); });             \
  }
#define X(type, prefix, var, dim0, dim1)                                    \
  groupedViews.def_property(                                                \
      #var, &GroupedViews::var, [](GroupedViews& views, py::handle rhs) {   \
        (views.var())[py::slice(py::none(), py::none(), py::none())] = rhs; \
      });

  MJMODEL_VIEW_GROUPS
#undef X
#undef XGROUP

  {
    py::handle builtins(PyEval_GetBuiltins());
    builtins[MjModelWrapper::kFromRawPointer] =
        reinterpret_cast<std::uintptr_t>(
            reinterpret_cast<void*>(&MjModelWrapper::FromRawPointer));
  }

  // ==================== MJWARNINGSTAT ========================================
  py::class_<MjWarningStatWrapper> mjWarningStat(m, "MjWarningStat");
  mjWarningStat.def(py::init<>());
  mjWarningStat.def("__copy__", [](const MjWarningStatWrapper& other) {
    return MjWarningStatWrapper(other);
  });
  mjWarningStat.def("__deepcopy__",
                    [](const MjWarningStatWrapper& other, py::dict) {
                      return MjWarningStatWrapper(other);
                    });
  DefineStructFunctions(mjWarningStat);
#define X(var)                                                             \
  mjWarningStat.def_property(                                              \
      #var, [](const MjWarningStatWrapper& d) { return d.get()->var; },    \
      [](MjWarningStatWrapper& d, decltype(raw::MjWarningStat::var) rhs) { \
        d.get()->var = rhs;                                                \
      });
  X(lastinfo);
  X(number);
#undef X

  py::class_<MjWarningStatList> mjWarningStatList(m, "_MjWarningStatList");
  mjWarningStatList.def("__getitem__", &MjWarningStatList::operator[],
                        py::return_value_policy::reference_internal);
  mjWarningStatList.def(
      "__getitem__",
      [](MjWarningStatList& list, ::mjtWarning idx) -> MjWarningStatWrapper& {
        return list[idx];
      },
      py::return_value_policy::reference_internal);
  mjWarningStatList.def("__getitem__", &MjWarningStatList::Slice,
                        py::keep_alive<0, 1>());
  mjWarningStatList.def("__len__", &MjWarningStatList::size);
  DefineStructFunctions(mjWarningStatList);

#define X(type, var) \
  mjWarningStatList.def_readonly(#var, &MjWarningStatList::var)
  X(int, lastinfo);
  X(int, number);
#undef X

  // ==================== MJLOGCONFIG ==========================================
  py::class_<MjLogConfigWrapper> mjLogConfig(m, "MjLogConfig");
  mjLogConfig.def(py::init<>());
  mjLogConfig.def("__copy__", [](const MjLogConfigWrapper& other) {
    return MjLogConfigWrapper(other);
  });
  mjLogConfig.def("__deepcopy__",
                  [](const MjLogConfigWrapper& other, py::dict) {
                    return MjLogConfigWrapper(other);
                  });
  DefineStructFunctions(mjLogConfig);
  mjLogConfig.def_property(
      "logto_console",
      [](const MjLogConfigWrapper& d) { return d.get()->logto_console; },
      [](MjLogConfigWrapper& d, bool rhs) { d.get()->logto_console = rhs; });
  mjLogConfig.def_property(
      "logto_file",
      [](const MjLogConfigWrapper& d) { return d.get()->logto_file; },
      [](MjLogConfigWrapper& d, bool rhs) { d.get()->logto_file = rhs; });
  mjLogConfig.def_property(
      "logfile",
      [](const MjLogConfigWrapper& d) { return std::string(d.get()->logfile); },
      [](MjLogConfigWrapper& d, const std::string& rhs) {
        std::strncpy(d.get()->logfile, rhs.c_str(), 1023);
        d.get()->logfile[1023] = '\0';
      });
  mjLogConfig.def_property(
      "topics", [](const MjLogConfigWrapper& d) { return d.get()->topics; },
      [](MjLogConfigWrapper& d, int rhs) { d.get()->topics = rhs; });
  mjLogConfig.def_static("get", []() {
    MjLogConfigWrapper wrapper;
    *wrapper.get() = mju_getLogConfig();
    return wrapper;
  });
  mjLogConfig.def("set", [](const MjLogConfigWrapper& self) {
    mju_setLogConfig(*self.get());
  });

  // ==================== MJLOGMESSAGE =========================================
  py::class_<MjLogMessageWrapper> mjLogMessage(m, "MjLogMessage");
  mjLogMessage.def(py::init<>());
  mjLogMessage.def("__copy__", [](const MjLogMessageWrapper& other) {
    return MjLogMessageWrapper(other);
  });
  mjLogMessage.def("__deepcopy__",
                   [](const MjLogMessageWrapper& other, py::dict) {
                     return MjLogMessageWrapper(other);
                   });
  DefineStructFunctions(mjLogMessage);
  mjLogMessage.def_property(
      "level", [](const MjLogMessageWrapper& d) { return d.get()->level; },
      [](MjLogMessageWrapper& d, int rhs) { d.get()->level = rhs; });
  mjLogMessage.def_property(
      "topic", [](const MjLogMessageWrapper& d) { return d.get()->topic; },
      [](MjLogMessageWrapper& d, int rhs) { d.get()->topic = rhs; });
  mjLogMessage.def_property(
      "subject",
      [](const MjLogMessageWrapper& d) {
        return std::string(d.get()->subject);
      },
      [](MjLogMessageWrapper& d, const std::string& rhs) {
        std::strncpy(d.get()->subject, rhs.c_str(), 1023);
        d.get()->subject[1023] = '\0';
      });
  mjLogMessage.def_property_readonly(
      "body",
      [](const MjLogMessageWrapper& d) -> py::object {
        if (d.get()->body) return py::str(d.get()->body);
        return py::none();
      });
  mjLogMessage.def_property_readonly(
      "func",
      [](const MjLogMessageWrapper& d) -> py::object {
        if (d.get()->func) return py::str(d.get()->func);
        return py::none();
      });
  mjLogMessage.def_property_readonly(
      "file",
      [](const MjLogMessageWrapper& d) -> py::object {
        if (d.get()->file) return py::str(d.get()->file);
        return py::none();
      });
  mjLogMessage.def_property(
      "line", [](const MjLogMessageWrapper& d) { return d.get()->line; },
      [](MjLogMessageWrapper& d, int rhs) { d.get()->line = rhs; });
  mjLogMessage.def_property(
      "timestamp",
      [](const MjLogMessageWrapper& d) { return d.get()->timestamp; },
      [](MjLogMessageWrapper& d, bool rhs) { d.get()->timestamp = rhs; });

  // ==================== MJTIMERSTAT ==========================================
  py::class_<MjTimerStatWrapper> mjTimerStat(m, "MjTimerStat");
  mjTimerStat.def(py::init<>());
  mjTimerStat.def("__copy__", [](const MjTimerStatWrapper& other) {
    return MjTimerStatWrapper(other);
  });
  mjTimerStat.def("__deepcopy__",
                  [](const MjTimerStatWrapper& other, py::dict) {
                    return MjTimerStatWrapper(other);
                  });
  DefineStructFunctions(mjTimerStat);
#define X(var)                                                         \
  mjTimerStat.def_property(                                            \
      #var, [](const MjTimerStatWrapper& d) { return d.get()->var; },  \
      [](MjTimerStatWrapper& d, decltype(raw::MjTimerStat::var) rhs) { \
        d.get()->var = rhs;                                            \
      });
  X(duration);
  X(number);
#undef X

  py::class_<MjTimerStatList> mjTimerStatList(m, "_MjTimerStatList");
  mjTimerStatList.def("__getitem__", &MjTimerStatList::operator[],
                      py::return_value_policy::reference_internal);
  mjTimerStatList.def(
      "__getitem__",
      [](MjTimerStatList& list, ::mjtTimer idx) -> MjTimerStatWrapper& {
        return list[idx];
      },
      py::return_value_policy::reference_internal);
  mjTimerStatList.def("__getitem__", &MjTimerStatList::Slice,
                      py::keep_alive<0, 1>());
  mjTimerStatList.def("__len__", &MjTimerStatList::size);
  DefineStructFunctions(mjTimerStatList);

#define X(type, var) mjTimerStatList.def_readonly(#var, &MjTimerStatList::var)
  X(mjtNum, duration);
  X(int, number);
#undef X

  // ==================== MJSOLVERSTAT =========================================
  py::class_<MjSolverStatWrapper> mjSolverStat(m, "MjSolverStat");
  mjSolverStat.def(py::init<>());
  mjSolverStat.def("__copy__", [](const MjSolverStatWrapper& other) {
    return MjSolverStatWrapper(other);
  });
  mjSolverStat.def("__deepcopy__",
                   [](const MjSolverStatWrapper& other, py::dict) {
                     return MjSolverStatWrapper(other);
                   });
  DefineStructFunctions(mjSolverStat);
#define X(var)                                                           \
  mjSolverStat.def_property(                                             \
      #var, [](const MjSolverStatWrapper& d) { return d.get()->var; },   \
      [](MjSolverStatWrapper& d, decltype(raw::MjSolverStat::var) rhs) { \
        d.get()->var = rhs;                                              \
      });
  X(improvement);
  X(gradient);
  X(lineslope);
  X(nactive);
  X(nchange);
  X(neval);
  X(nupdate);
#undef X

  py::class_<MjSolverStatList> mjSolverStatList(m, "_MjSolverStatList");
  mjSolverStatList.def("__getitem__", &MjSolverStatList::operator[],
                       py::return_value_policy::reference_internal);
  mjSolverStatList.def("__getitem__", &MjSolverStatList::Slice,
                       py::keep_alive<0, 1>());
  mjSolverStatList.def("__len__", &MjSolverStatList::size);
  DefineStructFunctions(mjSolverStatList);

#define X(type, var) mjSolverStatList.def_readonly(#var, &MjSolverStatList::var)
  X(mjtNum, improvement);
  X(mjtNum, gradient);
  X(mjtNum, lineslope);
  X(int, nactive);
  X(int, nchange);
  X(int, neval);
  X(int, nupdate);
#undef X

  // ==================== MJPRECONTACT =========================================
  py::class_<MjPreContactWrapper> mjPreContact(m, "MjPreContact");
  mjPreContact.def(py::init<>());
  mjPreContact.def("__copy__", [](const MjPreContactWrapper& self) {
    return MjPreContactWrapper(self);
  });
  mjPreContact.def("__deepcopy__",
                   [](const MjPreContactWrapper& self, py::dict) {
                     return MjPreContactWrapper(self);
                   });
  DefineStructFunctions(mjPreContact);

#define X(var)                                                           \
  mjPreContact.def_property(                                             \
      #var, [](const MjPreContactWrapper& c) { return c.get()->var; },   \
      [](MjPreContactWrapper& c, decltype(raw::MjPreContact::var) rhs) { \
        c.get()->var = rhs;                                              \
      })
  X(dist);
#undef X

#define X(var) DefinePyArray(mjPreContact, #var, &MjPreContactWrapper::var)
  X(pos);
  X(normal);
  X(tangent);
#undef X

  // ==================== MJCONTACT ============================================
  py::class_<MjContactWrapper> mjContact(m, "MjContact");
  mjContact.def(py::init<>());
  mjContact.def("__copy__", [](const MjContactWrapper& self) {
    return MjContactWrapper(self);
  });
  mjContact.def("__deepcopy__", [](const MjContactWrapper& self, py::dict) {
    return MjContactWrapper(self);
  });
  DefineStructFunctions(mjContact);

#define X(var)                                                      \
  mjContact.def_property(                                           \
      #var, [](const MjContactWrapper& c) { return c.get()->var; }, \
      [](MjContactWrapper& c, decltype(raw::MjContact::var) rhs) {  \
        c.get()->var = rhs;                                         \
      })
  X(dist);
  X(includemargin);
  X(adhesion);
  X(mu);
  X(dim);
  X(geom1);
  X(geom2);
  X(exclude);
  X(efc_address);
#undef X

#define X(var) DefinePyArray(mjContact, #var, &MjContactWrapper::var)
  X(pos);
  X(frame);
  X(friction);
  X(solref);
  X(solreffriction);
  X(solimp);
  X(surfacevel);
  X(H);
  X(geom);
  X(flex);
  X(elem);
  X(vert);
#undef X

  py::class_<MjContactList> mjContactList(m, "_MjContactList");
  mjContactList.def("__getitem__", &MjContactList::operator[],
                    py::return_value_policy::reference_internal);
  mjContactList.def("__getitem__", &MjContactList::Slice,
                    py::keep_alive<0, 1>());
  mjContactList.def("__len__", &MjContactList::size);
  DefineStructFunctions(mjContactList);

#define X(type, var)                                                      \
  mjContactList.def_property_readonly(#var, [](const MjContactList& c) {  \
    const py::ssize_t n = c.size();                                       \
    const py::ssize_t s =                                                 \
        c.stride() * static_cast<py::ssize_t>(sizeof(raw::MjContact));    \
    return py::array_t<type>(std::vector<py::ssize_t>{n},                 \
                             std::vector<py::ssize_t>{s},                 \
                             n > 0 ? &c.get()->var : nullptr, c.owner()); \
  });
#define XN(type, var)                                                        \
  mjContactList.def_property_readonly(#var, [](const MjContactList& c) {     \
    const py::ssize_t n = c.size();                                          \
    const py::ssize_t dim = sizeof(raw::MjContact::var) / sizeof(type);      \
    const py::ssize_t s =                                                    \
        c.stride() * static_cast<py::ssize_t>(sizeof(raw::MjContact));       \
    return py::array_t<type>(                                                \
        std::vector<py::ssize_t>{n, dim},                                    \
        std::vector<py::ssize_t>{s, static_cast<py::ssize_t>(sizeof(type))}, \
        n > 0 ? &c.get()->var[0] : nullptr, c.owner());                      \
  });
  X(mjtNum, dist);
  XN(mjtNum, pos);
  XN(mjtNum, frame);
  X(mjtNum, includemargin);
  XN(mjtNum, friction);
  XN(mjtNum, solref);
  XN(mjtNum, solreffriction);
  XN(mjtNum, solimp);
  X(mjtNum, adhesion);
  XN(mjtNum, surfacevel);
  X(mjtNum, mu);
  XN(mjtNum, H);
  X(int, dim);
  X(int, geom1);
  X(int, geom2);
  X(int, exclude);
  X(int, efc_address);
  XN(int, geom);
  XN(int, flex);
  XN(int, elem);
  XN(int, vert);
#undef X
#undef XN

  // ==================== MJDATA ===============================================
  py::class_<MjDataWrapper> mjData(m, "MjData");
  mjData.def(py::init([](MjModelWrapper* m) {
    if (!m) {
      throw py::type_error("MjModel cannot be None");
    }
    return MjDataWrapper(m);
  }));
  mjData.def_property_readonly("_address", [](const MjDataWrapper& d) {
    return reinterpret_cast<std::uintptr_t>(d.get());
  });
  mjData.def_property_readonly(
      "model", [](const MjDataWrapper& d) { return &d.model(); });
  mjData.def("__copy__",
             [](const MjDataWrapper& other) { return MjDataWrapper(other); });
  mjData.def("__deepcopy__", [](const MjDataWrapper& other, py::dict memo) {
    // Use copy.deepcopy(model) to make a model that Python is aware of.
    py::object new_model_py =
        py::cast(other.model()).attr("__deepcopy__")(memo);
    return MjDataWrapper(other, new_model_py.cast<MjModelWrapper*>());
  });
  mjData.def_property_readonly_static("_all_fields", [](py::object) {
    std::vector<std::string> fields;
#define X(dtype, name) fields.push_back(#name);
    MJDATA_SCALAR
#undef X
#define X(dtype, name, dim0, dim1) fields.push_back(#name);
    MJDATA_VECTOR
    MJDATA_POINTERS
#undef X
    return py::tuple(py::cast(fields));
  });
  mjData.def(py::pickle(
      [](const MjDataWrapper& d) {  // __getstate__
        std::ostringstream output(std::ios::out | std::ios::binary);
        d.Serialize(output);
        return py::bytes(output.str());
      },
      [](py::bytes b) {  // __setstate__
        std::istringstream input(b, std::ios::in | std::ios::binary);
        return MjDataWrapper::Deserialize(input);
      }));
  mjData.def_property_readonly(
      "signature",
      [](const MjDataWrapper& d) -> uint64_t { return d.get()->signature; });

#define X(type, var)                                             \
  mjData.def_property(                                           \
      #var, [](const MjDataWrapper& d) { return d.get()->var; }, \
      [](MjDataWrapper& d, decltype(raw::MjData::var) rhs) {     \
        d.get()->var = rhs;                                      \
      });
  MJDATA_SCALAR
#undef X

#define X(dtype, var, dim0, dim1) \
  DefinePyArray(mjData, #var, &MjDataWrapper::var);
  MJDATA_POINTERS
  MJDATA_ARENA_POINTERS_CONTACT
#undef X

#undef MJ_M
#define MJ_M(x) d.model().get()->x
#undef MJ_D
#define MJ_D(x) d.get()->x
#define X(dtype, var, dim0, dim1)                                         \
  mjData.def_property_readonly(#var, [](const MjDataWrapper& d) {         \
    return InitArenaArray(X_ARRAY_SHAPE(dim0, dim1), d.get()->var,        \
                          d.owner(),                                      \
                          static_cast<dtype>(IsNoneIndex(#var) ? -1 : 0)); \
  });

  MJDATA_ARENA_POINTERS_SOLVER
  MJDATA_ARENA_POINTERS_DUAL
  MJDATA_ARENA_POINTERS_ISLAND
  MJDATA_ARENA_POINTERS_EFM

#undef MJ_M
#define MJ_M(x) (x)
#undef MJ_D
#define MJ_D(x) (x)
#undef X

#define X(dtype, var, dim0, dim1) \
  DefinePyArray(mjData, #var, &MjDataWrapper::var);
  MJDATA_VECTOR
#undef X

#define XGROUP(MjDataGroupedViews, field, nfield, FIELD_XMACROS)             \
  mjData.def(                                                                \
      #field,                                                                \
      [](MjDataWrapper& d, int i) -> auto& { return d.indexer().field(i); }, \
      py::return_value_policy::reference_internal);                          \
  mjData.def(                                                                \
      #field,                                                                \
      [](MjDataWrapper& d, std::string_view name) -> auto& {                 \
        return d.indexer().field##_by_name(name);                            \
      },                                                                     \
      py::return_value_policy::reference_internal, py::arg_v("name", ""));

  MJDATA_VIEW_GROUPS
#undef XGROUP

#define XGROUP(spectype, field)                                     \
  mjData.def(                                                       \
      "bind_scalar",                                                \
      [](MjDataWrapper& d, spectype& spec) -> auto& {               \
        if (mjs_getSpec(spec.element)->element->signature !=        \
            d.get()->signature) {                                   \
          throw py::value_error(                                    \
              "The mjSpec does not match mjData. Please recompile " \
              "the mjSpec.");                                       \
        }                                                           \
        return d.indexer().field(mjs_getId(spec.element));          \
      },                                                            \
      py::return_value_policy::reference_internal,                  \
      py::arg_v("spec", py::none()));

  MJDATA_BIND_GROUPS
#undef XGROUP

#define XGROUP(field, altname, FIELD_XMACROS)                                \
  mjData.def(                                                                \
      #altname,                                                              \
      [](MjDataWrapper& d, int i) -> auto& { return d.indexer().field(i); }, \
      py::return_value_policy::reference_internal);                          \
  mjData.def(                                                                \
      #altname,                                                              \
      [](MjDataWrapper& d, std::string_view name) -> auto& {                 \
        return d.indexer().field##_by_name(name);                            \
      },                                                                     \
      py::return_value_policy::reference_internal, py::arg_v("name", ""));

  MJDATA_VIEW_GROUPS_ALTNAMES
#undef XGROUP

#define XGROUP(MjDataGroupedViews, field, nfield, FIELD_XMACROS)             \
  {                                                                          \
    using GroupedViews = MjDataGroupedViews;                                 \
    py::class_<MjDataGroupedViews> groupedViews(m, "_" #MjDataGroupedViews); \
    FIELD_XMACROS                                                            \
    DefineComputedViews(groupedViews);                                       \
    groupedViews.def("__repr__", MjDataStructRepr<GroupedViews>);            \
    groupedViews.def_property_readonly(                                      \
        "id", [](GroupedViews& views) { return views.index(); });            \
    groupedViews.def_property_readonly(                                      \
        "name", [](GroupedViews& views) { return views.name(); });           \
  }
#define X(type, prefix, var, dim0, dim1)                                    \
  groupedViews.def_property(                                                \
      #var, &GroupedViews::var, [](GroupedViews& views, py::handle rhs) {   \
        (views.var())[py::slice(py::none(), py::none(), py::none())] = rhs; \
      });

  MJDATA_VIEW_GROUPS
#undef X
#undef XGROUP

  {
    py::handle builtins(PyEval_GetBuiltins());
    builtins[MjDataWrapper::kFromRawPointer] = reinterpret_cast<std::uintptr_t>(
        reinterpret_cast<void*>(&MjDataWrapper::FromRawPointer));
  }

  // ==================== MJSTATISTIC ==========================================
  py::class_<MjStatisticWrapper> mjStatistic(m, "MjStatistic");
  mjStatistic.def(py::init<>());
  mjStatistic.def("__copy__", [](const MjStatisticWrapper& other) {
    return MjStatisticWrapper(other);
  });
  mjStatistic.def("__deepcopy__",
                  [](const MjStatisticWrapper& other, py::dict) {
                    return MjStatisticWrapper(other);
                  });
  DefineStructFunctions(mjStatistic);

#define X(var, dim)                                                    \
  mjStatistic.def_property(                                            \
      #var, [](const MjStatisticWrapper& c) { return c.get()->var; },  \
      [](MjStatisticWrapper& c, decltype(raw::MjStatistic::var) rhs) { \
        c.get()->var = rhs;                                            \
      });
#define XVEC(var, dim) \
  DefinePyArray(mjStatistic, #var, &MjStatisticWrapper::var);

  MJSTATISTIC_FIELDS

#undef XVEC
#undef X

  // ==================== MJLROPT ==============================================
  py::class_<raw::MjLROpt> mjLROpt(m, "MjLROpt");
  mjLROpt.def(py::init<>());
  mjLROpt.def("__copy__",
              [](const raw::MjLROpt& other) { return raw::MjLROpt(other); });
  mjLROpt.def("__deepcopy__", [](const raw::MjLROpt& other, py::dict) {
    return raw::MjLROpt(other);
  });
  DefineStructFunctions(mjLROpt);
#define X(var) mjLROpt.def_readwrite(#var, &raw::MjLROpt::var)
  X(mode);
  X(useexisting);
  X(uselimit);
  X(accel);
  X(maxforce);
  X(timeconst);
  X(timestep);
  X(inttotal);
  X(interval);
  X(tolrange);
#undef X


  // ==================== MJRRECT ==============================================
  py::class_<raw::MjrRect> mjrRect(m, "MjrRect");
  mjrRect.def(py::init([](int left, int bottom, int width, int height) {
                return raw::MjrRect{left, bottom, width, height};
              }),
              py::arg("left"), py::arg("bottom"), py::arg("width"),
              py::arg("height"));
  mjrRect.def("__copy__",
              [](const raw::MjrRect& other) { return raw::MjrRect(other); });
  mjrRect.def("__deepcopy__", [](const raw::MjrRect& other, py::dict) {
    return raw::MjrRect(other);
  });
  DefineStructFunctions(mjrRect);
#define X(var) mjrRect.def_readwrite(#var, &raw::MjrRect::var)
  X(left);
  X(bottom);
  X(width);
  X(height);
#undef X

  // ==================== MJRVERTEXATTRIBUTE ===================================
  py::class_<raw::MjrVertexAttribute> mjrVertexAttribute(m,
                                                         "MjrVertexAttribute");
  mjrVertexAttribute.def(py::init([](int usage, int type) {
                           return raw::MjrVertexAttribute{usage, type};
                         }),
                         py::arg("usage") = 0, py::arg("type") = 0);
  mjrVertexAttribute.def("__copy__", [](const raw::MjrVertexAttribute& other) {
    return raw::MjrVertexAttribute(other);
  });
  mjrVertexAttribute.def("__deepcopy__",
                         [](const raw::MjrVertexAttribute& other, py::dict) {
                           return raw::MjrVertexAttribute(other);
                         });
  DefineStructFunctions(mjrVertexAttribute);
#define X(var) \
  mjrVertexAttribute.def_readwrite(#var, &raw::MjrVertexAttribute::var)
  X(usage);
  X(type);
#undef X

  // ==================== MJVPERTURB ===========================================
  py::class_<MjvPerturbWrapper> mjvPerturb(m, "MjvPerturb");
  mjvPerturb.def(py::init<>());
  mjvPerturb.def("__copy__", [](const MjvPerturbWrapper& other) {
    return MjvPerturbWrapper(other);
  });
  mjvPerturb.def("__deepcopy__", [](const MjvPerturbWrapper& other, py::dict) {
    return MjvPerturbWrapper(other);
  });
  DefineStructFunctions(mjvPerturb);
#define X(var)                                                       \
  mjvPerturb.def_property(                                           \
      #var, [](const MjvPerturbWrapper& c) { return c.get()->var; }, \
      [](MjvPerturbWrapper& c, decltype(raw::MjvPerturb::var) rhs) { \
        c.get()->var = rhs;                                          \
      })
  X(select);
  X(flexselect);
  X(skinselect);
  X(active);
  X(active2);
  X(localmass);
  X(scale);
#undef X

#define X(var) DefinePyArray(mjvPerturb, #var, &MjvPerturbWrapper::var)
  X(refpos);
  X(refquat);
  X(refselpos);
  X(localpos);
#undef X

  // ==================== MJVCAMERA ============================================
  py::class_<MjvCameraWrapper> mjvCamera(m, "MjvCamera");
  mjvCamera.def(py::init<>());
  mjvCamera.def("__copy__", [](const MjvCameraWrapper& other) {
    return MjvCameraWrapper(other);
  });
  mjvCamera.def("__deepcopy__", [](const MjvCameraWrapper& other, py::dict) {
    return MjvCameraWrapper(other);
  });
  DefineStructFunctions(mjvCamera);
#define X(var)                                                      \
  mjvCamera.def_property(                                           \
      #var, [](const MjvCameraWrapper& c) { return c.get()->var; }, \
      [](MjvCameraWrapper& c, decltype(raw::MjvCamera::var) rhs) {  \
        c.get()->var = rhs;                                         \
      })
  X(type);
  X(fixedcamid);
  X(trackbodyid);
  X(distance);
  X(azimuth);
  X(elevation);
  X(orthographic);
#undef X

#define X(var) DefinePyArray(mjvCamera, #var, &MjvCameraWrapper::var)
  X(lookat);
#undef X

  // ==================== MJVGLCAMERA ==========================================
  py::class_<MjvGLCameraWrapper> mjvGLCamera(m, "MjvGLCamera");
  mjvGLCamera.def(py::init<>());
  mjvGLCamera.def("__copy__", [](const MjvGLCameraWrapper& other) {
    return MjvGLCameraWrapper(other);
  });
  mjvGLCamera.def("__deepcopy__",
                  [](const MjvGLCameraWrapper& other, py::dict) {
                    return MjvGLCameraWrapper(other);
                  });
  DefineStructFunctions(mjvGLCamera);
#define X(var)                                                         \
  mjvGLCamera.def_property(                                            \
      #var, [](const MjvGLCameraWrapper& c) { return c.get()->var; },  \
      [](MjvGLCameraWrapper& c, decltype(raw::MjvGLCamera::var) rhs) { \
        c.get()->var = rhs;                                            \
      })
  X(frustum_center);
  X(frustum_width);
  X(frustum_bottom);
  X(frustum_top);
  X(frustum_near);
  X(frustum_far);
  X(orthographic);
#undef X

#define X(var) DefinePyArray(mjvGLCamera, #var, &MjvGLCameraWrapper::var)
  X(pos);
  X(forward);
  X(up);
#undef X

  // ==================== MJVGEOM ==============================================
  py::class_<MjvGeomWrapper> mjvGeom(m, "MjvGeom");
  mjvGeom.def(py::init<>());
  mjvGeom.def("__copy__", [](const MjvGeomWrapper& other) {
    return MjvGeomWrapper(other);
  });
  mjvGeom.def("__deepcopy__", [](const MjvGeomWrapper& other, py::dict) {
    return MjvGeomWrapper(other);
  });
  DefineStructFunctions(mjvGeom);
#define X(var)                                                    \
  mjvGeom.def_property(                                           \
      #var, [](const MjvGeomWrapper& c) { return c.get()->var; }, \
      [](MjvGeomWrapper& c, decltype(raw::MjvGeom::var) rhs) {    \
        c.get()->var = rhs;                                       \
      })
  X(type);
  X(dataid);
  X(objtype);
  X(objid);
  X(category);
  X(matid);
  X(texid);
  X(texuniform);
  X(texcoord);
  X(segid);
  X(emission);
  X(specular);
  X(shininess);
  X(reflectance);
  X(castshadow);
  X(camdist);
  X(modelrbound);
  X(transparent);
#undef X

#define X(var) DefinePyArray(mjvGeom, #var, &MjvGeomWrapper::var)
  X(size);
  X(pos);
  X(mat);
  X(rgba);
  X(texrepeat);
#undef X

  DefinePyStr(mjvGeom, "label", &raw::MjvGeom::label);

  // ==================== MJVLIGHT =============================================
  py::class_<MjvLightWrapper> mjvLight(m, "MjvLight");
  mjvLight.def(py::init<>());
  mjvLight.def("__copy__", [](const MjvLightWrapper& other) {
    return MjvLightWrapper(other);
  });
  mjvLight.def("__deepcopy__", [](const MjvLightWrapper& other, py::dict) {
    return MjvLightWrapper(other);
  });
  DefineStructFunctions(mjvLight);
#define X(var)                                                     \
  mjvLight.def_property(                                           \
      #var, [](const MjvLightWrapper& c) { return c.get()->var; }, \
      [](MjvLightWrapper& c, decltype(raw::MjvLight::var) rhs) {   \
        c.get()->var = rhs;                                        \
      })
  X(id);
  X(cutoff);
  X(exponent);
  X(headlight);
  X(type);
  X(texid);
  X(castshadow);
  X(bulbradius);
  X(intensity);
  X(range);
  X(softness);
#undef X

#define X(var) DefinePyArray(mjvLight, #var, &MjvLightWrapper::var)
  X(pos);
  X(dir);
  X(attenuation);
  X(ambient);
  X(diffuse);
  X(specular);
#undef X

  // ==================== MJVOPTION ============================================
  py::class_<MjvOptionWrapper> mjvOption(m, "MjvOption");
  mjvOption.def(py::init<>());
  mjvOption.def("__copy__", [](const MjvOptionWrapper& other) {
    return MjvOptionWrapper(other);
  });
  mjvOption.def("__deepcopy__", [](const MjvOptionWrapper& other, py::dict) {
    return MjvOptionWrapper(other);
  });
  DefineStructFunctions(mjvOption);
#define X(var)                                                      \
  mjvOption.def_property(                                           \
      #var, [](const MjvOptionWrapper& c) { return c.get()->var; }, \
      [](MjvOptionWrapper& c, decltype(raw::MjvOption::var) rhs) {  \
        c.get()->var = rhs;                                         \
      })
  X(label);
  X(frame);
  X(bvh_depth);
  X(flex_layer);
#undef X

#define X(var) DefinePyArray(mjvOption, #var, &MjvOptionWrapper::var)
  X(geomgroup);
  X(sitegroup);
  X(jointgroup);
  X(tendongroup);
  X(actuatorgroup);
  X(flexgroup);
  X(skingroup);
  X(flags);
#undef X

  // ==================== MJVSCENE =============================================
  py::class_<MjvSceneWrapper> mjvScene(m, "MjvScene");
  mjvScene.def(py::init<>());
  mjvScene.def(py::init<const MjModelWrapper&, int>(), py::arg("model"),
               py::arg("maxgeom"));
  mjvScene.def("__copy__", [](const MjvSceneWrapper& other) {
    return MjvSceneWrapper(other);
  });
  mjvScene.def("__deepcopy__", [](const MjvSceneWrapper& other, py::dict) {
    return MjvSceneWrapper(other);
  });
#define X(var)                                                     \
  mjvScene.def_property(                                           \
      #var, [](const MjvSceneWrapper& c) { return c.get()->var; }, \
      [](MjvSceneWrapper& c, decltype(raw::MjvScene::var) rhs) {   \
        c.get()->var = rhs;                                        \
      })
  X(maxgeom);
  X(ngeom);
  X(nflex);
  X(nskin);
  X(nlight);
  X(flexvertopt);
  X(flexedgeopt);
  X(flexfaceopt);
  X(flexskinopt);
  X(enabletransform);
  X(scale);
  X(stereo);
  X(framewidth);
  X(status);
#undef X

#define X(var) DefinePyArray(mjvScene, #var, &MjvSceneWrapper::var)
  X(geoms);
  X(geomorder);
  X(flexedgeadr);
  X(flexedgenum);
  X(flexvertadr);
  X(flexvertnum);
  X(flexfaceadr);
  X(flexfacenum);
  X(flexfaceused);
  X(flexedge);
  X(flexvert);
  X(flexface);
  X(flexnormal);
  X(flextexcoord);
  X(skinfacenum);
  X(skinvertadr);
  X(skinvertnum);
  X(skinvert);
  X(skinnormal);
  X(lights);
  X(camera);
  X(translate);
  X(rotate);
  X(flags);
  X(framergb);
#undef X

  // ==================== MJVFIGURE ============================================
  py::class_<MjvFigureWrapper> mjvFigure(m, "MjvFigure");
  mjvFigure.def(py::init<>());
  mjvFigure.def("__copy__", [](const MjvFigureWrapper& other) {
    return MjvFigureWrapper(other);
  });
  mjvFigure.def("__deepcopy__", [](const MjvFigureWrapper& other, py::dict) {
    return MjvFigureWrapper(other);
  });
#define X(var)                                                      \
  mjvFigure.def_property(                                           \
      #var, [](const MjvFigureWrapper& c) { return c.get()->var; }, \
      [](MjvFigureWrapper& c, decltype(raw::MjvFigure::var) rhs) {  \
        c.get()->var = rhs;                                         \
      })
  X(flg_legend);
  X(flg_extend);
  X(flg_barplot);
  X(flg_selection);
  X(flg_symmetric);
  X(linewidth);
  X(gridwidth);
  X(legendoffset);
  X(subplot);
  X(highlightid);
  X(selection);
#undef X

#define X(var) DefinePyArray(mjvFigure, #var, &MjvFigureWrapper::var)
  X(flg_ticklabel);
  X(gridsize);
  X(gridrgb);
  X(figurergba);
  X(panergba);
  X(legendrgba);
  X(textrgb);
  X(linergb);
  X(range);
  X(highlight);
  X(linepnt);
  X(linedata);
  X(xaxispixel);
  X(yaxispixel);
  X(xaxisdata);
  X(yaxisdata);
#undef X

#define X(var) DefinePyStr(mjvFigure, #var, &raw::MjvFigure::var);
  X(xformat);
  X(yformat);
  X(minwidth);
  X(title);
  X(xlabel);
#undef X

  mjvFigure.def_readonly("linename", &MjvFigureWrapper::linename);

  // mjv_averageCamera and mjv_camera2GLCamera both return an mjvGLCamera.
  // We need to call the wrapper's constructor on the return value. Defining the
  // binding for these functions in this file to avoid symbol dependency across
  // modules.
  m.def(
      "mjv_averageCamera",
      [](const MjvGLCameraWrapper& cam1, const MjvGLCameraWrapper& cam2) {
        return MjvGLCameraWrapper([&cam1, &cam2]() {
          py::gil_scoped_release no_gil;
          return InterceptMjErrors(mjv_averageCamera)(cam1.get(), cam2.get());
        }());
      },
      py::arg("cam1"), py::arg("cam2"),
      py::doc(python_traits::mjv_averageCamera::doc));

  m.def(
      "mjv_camera2GLCamera",
      [](const MjModelWrapper& m, const MjDataWrapper& d,
         const MjvCameraWrapper& cam) {
        return MjvGLCameraWrapper([&m, &d, &cam]() {
          py::gil_scoped_release no_gil;
          return InterceptMjErrors(mjv_camera2GLCamera)(m.get(), d.get(),
                                                        cam.get());
        }());
      },
      py::arg("m"), py::arg("d"), py::arg("cam"),
      py::doc(python_traits::mjv_camera2GLCamera::doc));

  m.def(
      "_recompile_spec_addr",
      [](uintptr_t spec_addr, const MjModelWrapper& m, const MjDataWrapper& d,
         uintptr_t vfs_addr) {
        return RecompileSpec(reinterpret_cast<raw::MjSpec*>(spec_addr),
                             reinterpret_cast<const mjVFS*>(vfs_addr), m, d);
      },
      py::arg("spec_addr"), py::arg("m"), py::arg("d"),
      py::arg("vfs_addr") = 0);
}  // PYBIND11_MODULE NOLINT(readability/fn_size)
}  // namespace mujoco::python::_impl
