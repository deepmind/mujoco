// Copyright 2024 DeepMind Technologies Limited
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

#include <array>
#include <cstddef>  // IWYU pragma: keep
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>  // IWYU pragma: keep
#include <unordered_map>
#include <utility>
#include <vector>  // IWYU pragma: keep

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <mujoco/mjspec.h>  // IWYU pragma: keep
#include <mujoco/mujoco.h>
#include "errors.h"
#include "indexers.h"  // IWYU pragma: keep
#include "raw.h"
#include "specs_wrapper.h"  // IWYU pragma: keep
#include "structs.h"        // IWYU pragma: keep
#include "vfs.h"
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = ::pybind11;

namespace mujoco::python {
using MjInt2 = Eigen::Map<Eigen::Vector2i>;
using MjInt3 = Eigen::Map<Eigen::Vector3i>;
using MjFloat2 = Eigen::Map<Eigen::Vector2f>;
using MjFloat3 = Eigen::Map<Eigen::Vector3f>;
using MjFloat4 = Eigen::Map<Eigen::Vector4f>;
using MjDouble2 = Eigen::Map<Eigen::Vector2d>;
using MjDouble3 = Eigen::Map<Eigen::Vector3d>;
using MjDouble4 = Eigen::Map<Eigen::Vector4d>;
using MjDouble5 = Eigen::Map<Eigen::Matrix<double, 5, 1>>;
using MjDouble6 = Eigen::Map<Eigen::Matrix<double, 6, 1>>;
using MjDouble9 = Eigen::Map<Eigen::Matrix<double, 9, 1>>;
using MjDouble10 = Eigen::Map<Eigen::Matrix<double, 10, 1>>;
using MjDouble12 = Eigen::Map<Eigen::Matrix<double, 12, 1>>;
using MjDoubleVec = Eigen::Map<Eigen::VectorXd>;
using MjNum2 = Eigen::Map<Eigen::Matrix<mjtNum, 2, 1>>;
using MjNum3 = Eigen::Map<Eigen::Matrix<mjtNum, 3, 1>>;
using MjNum5 = Eigen::Map<Eigen::Matrix<mjtNum, 5, 1>>;

using MjIntRef2 = Eigen::Ref<const Eigen::Vector2i>;
using MjIntRef3 = Eigen::Ref<const Eigen::Vector3i>;
using MjFloatRef2 = Eigen::Ref<const Eigen::Vector2f>;
using MjFloatRef3 = Eigen::Ref<const Eigen::Vector3f>;
using MjFloatRef4 = Eigen::Ref<const Eigen::Vector4f>;
using MjDoubleRef2 = Eigen::Ref<const Eigen::Vector2d>;
using MjDoubleRef3 = Eigen::Ref<const Eigen::Vector3d>;
using MjDoubleRef4 = Eigen::Ref<const Eigen::Vector4d>;
using MjDoubleRef5 = Eigen::Ref<const Eigen::Matrix<double, 5, 1>>;
using MjDoubleRef6 = Eigen::Ref<const Eigen::Matrix<double, 6, 1>>;
using MjDoubleRef9 = Eigen::Ref<const Eigen::Matrix<double, 9, 1>>;
using MjDoubleRef10 = Eigen::Ref<const Eigen::Matrix<double, 10, 1>>;
using MjDoubleRef12 = Eigen::Ref<const Eigen::Matrix<double, 12, 1>>;
using MjDoubleRefVec = Eigen::Ref<const Eigen::VectorXd>;
using MjNumRef2 = Eigen::Ref<const Eigen::Matrix<mjtNum, 2, 1>>;
using MjNumRef3 = Eigen::Ref<const Eigen::Matrix<mjtNum, 3, 1>>;
using MjNumRef5 = Eigen::Ref<const Eigen::Matrix<mjtNum, 5, 1>>;

namespace {
mjVFS* GetOpenVfs(MjVfs* vfs) {
  if (vfs) {
    if (!vfs->is_open()) {
      throw std::runtime_error("VFS is closed");
    }
    return vfs->get();
  }
  return nullptr;
}

mjVFS* GetOpenVfs(std::optional<MjVfs*> vfs) {
  return GetOpenVfs(vfs.value_or(nullptr));
}
}  // namespace

template <typename LoadFunc>
static raw::MjSpec* LoadSpecFileImpl(const std::string& filename,
                                     const std::vector<_impl::VfsAsset>& assets,
                                     LoadFunc&& loadfunc) {
  mjVFS vfs;
  mjVFS* vfs_ptr = nullptr;
  if (!assets.empty()) {
    mj_defaultVFS(&vfs);
    vfs_ptr = &vfs;
    for (const auto& asset : assets) {
      std::string buffer_name = _impl::StripPath(asset.name);
      const int vfs_error = InterceptMjErrors(mj_addBufferVFS)(
          vfs_ptr, buffer_name.c_str(), asset.content, asset.content_size);
      if (vfs_error) {
        mj_deleteVFS(vfs_ptr);
        if (vfs_error == 2) {
          throw py::value_error("Repeated file name in assets dict: " +
                                buffer_name);
        } else {
          throw py::value_error("Asset failed to load: " + buffer_name);
        }
      }
    }
  }

  raw::MjSpec* spec = loadfunc(filename.c_str(), vfs_ptr);
  mj_deleteVFS(vfs_ptr);
  return spec;
}

template <typename T>
struct MjTypeVec {
  MjTypeVec(T* data, int size) : ptr(data), size(size) {}
  T* ptr;
  int size;
};

template <typename T>
void DefineArray(py::module& m, const std::string& typestr) {
  using Class = MjTypeVec<T>;
  py::class_<Class>(m, typestr.c_str())
      .def(py::init([](T* data, int size) { return Class(data, size); }))
      .def("__getitem__",
           [](Class& v, int i) {
             if (i < 0 || i >= v.size) {
               throw py::index_error("Index out of range.");
             }
             return v.ptr[i];
           })
      .def("__setitem__",
           [](Class& v, int i, T c) {
             if (i < 0 || i >= v.size) {
               throw py::index_error("Index out of range.");
             }
             v.ptr[i] = std::move(c);
           })
      .def("__len__", [](Class& v) { return v.size; })
      .def(
          "__iter__",
          [](Class& v) { return py::make_iterator(v.ptr, v.ptr + v.size); },
          py::keep_alive<0, 1>(), py::return_value_policy::reference_internal);
};

py::list FindAllImpl(raw::MjsBody& body, mjtObj objtype, bool recursive) {
  py::list list;
  raw::MjsElement* el = mjs_firstChild(&body, objtype, recursive);
  std::string error = mjs_getError(mjs_getSpec(body.element));
  if (!el && !error.empty()) {
    throw pybind11::value_error(error);
  }
  while (el) {
    switch (objtype) {
      case mjOBJ_BODY:
        list.append(mjs_asBody(el));
        break;
      case mjOBJ_CAMERA:
        list.append(mjs_asCamera(el));
        break;
      case mjOBJ_FRAME:
        list.append(mjs_asFrame(el));
        break;
      case mjOBJ_GEOM:
        list.append(mjs_asGeom(el));
        break;
      case mjOBJ_JOINT:
        list.append(mjs_asJoint(el));
        break;
      case mjOBJ_LIGHT:
        list.append(mjs_asLight(el));
        break;
      case mjOBJ_SITE:
        list.append(mjs_asSite(el));
        break;
      default:
        // this should never happen
        throw pybind11::value_error(
            "body.find_all supports the types: body, frame, geom, site, "
            "joint, light, camera.");
        break;
    }
    el = mjs_nextChild(&body, el, recursive);
  }
  return list;  // list of pointers, so they can be copied
}

static std::string addSuffixBeforeExtension(const std::string& original_path,
                                            const std::string& suffix_to_add) {
  // Find the position of the last dot
  size_t dot_pos = original_path.rfind('.');

  // Check if a dot was found
  if (dot_pos != std::string::npos) {
    std::string new_path = original_path;
    new_path.insert(dot_pos, suffix_to_add);
    return new_path;
  } else {
    // No extension found, just append
    return original_path + suffix_to_add;
  }
}

static std::string addPrefixToFileName(const std::string& original_path,
                                       const std::string& prefix_to_add) {
  // Find the position of the last slash
  size_t slash_pos = original_path.find_last_of("/\\");

  // Check if a slash was found
  if (slash_pos != std::string::npos) {
    std::string new_path = original_path;
    new_path.insert(slash_pos + 1, prefix_to_add);
    return new_path;
  } else {
    // No slash found, just prepend
    return prefix_to_add + original_path;
  }
}

static std::string addPrefixAndSuffix(const std::string& original_path,
                                      const std::string& prefix_to_add,
                                      const std::string& suffix_to_add) {
  std::string prefixed_path = addPrefixToFileName(original_path, prefix_to_add);
  return addSuffixBeforeExtension(prefixed_path, suffix_to_add);
}

PYBIND11_MODULE(_specs, m, pybind11::mod_gil_not_used()) {
  auto structs_m = py::module::import("mujoco._structs");
  py::function mjmodel_from_raw_ptr =
      structs_m.attr("MjModel").attr("_from_model_ptr");
  py::function mjmodel_mjdata_from_spec_ptr =
      structs_m.attr("_recompile_spec_addr");

  py::class_<MjSpec> mjSpec(m, "MjSpec");
  py::class_<raw::MjsElement, MjsElementRef<raw::MjsElement>> mjsElement(
      m, "MjsElement");
  py::class_<raw::MjsDefault, MjsElementRef<raw::MjsDefault>> mjsDefault(
      m, "MjsDefault");
  py::class_<raw::MjsBody, MjsElementRef<raw::MjsBody>> mjsBody(m, "MjsBody");
  py::class_<raw::MjsFrame, MjsElementRef<raw::MjsFrame>> mjsFrame(m,
                                                                   "MjsFrame");
  py::class_<raw::MjsGeom, MjsElementRef<raw::MjsGeom>> mjsGeom(m, "MjsGeom");
  py::class_<raw::MjsJoint, MjsElementRef<raw::MjsJoint>> mjsJoint(m,
                                                                   "MjsJoint");
  py::class_<raw::MjsLight, MjsElementRef<raw::MjsLight>> mjsLight(m,
                                                                   "MjsLight");
  py::class_<raw::MjsMaterial, MjsElementRef<raw::MjsMaterial>> mjsMaterial(
      m, "MjsMaterial");
  py::class_<raw::MjsSite, MjsElementRef<raw::MjsSite>> mjsSite(m, "MjsSite");
  py::class_<raw::MjsMesh, MjsElementRef<raw::MjsMesh>> mjsMesh(m, "MjsMesh");
  py::class_<raw::MjsSkin, MjsElementRef<raw::MjsSkin>> mjsSkin(m, "MjsSkin");
  py::class_<raw::MjsTexture, MjsElementRef<raw::MjsTexture>> mjsTexture(
      m, "MjsTexture");
  py::class_<raw::MjsText, MjsElementRef<raw::MjsText>> mjsText(m, "MjsText");
  py::class_<raw::MjsTuple, MjsElementRef<raw::MjsTuple>> mjsTuple(m,
                                                                   "MjsTuple");
  py::class_<raw::MjsCamera, MjsElementRef<raw::MjsCamera>> mjsCamera(
      m, "MjsCamera");
  py::class_<raw::MjsFlex, MjsElementRef<raw::MjsFlex>> mjsFlex(m, "MjsFlex");
  py::class_<raw::MjsHField, MjsElementRef<raw::MjsHField>> mjsHField(
      m, "MjsHField");
  py::class_<raw::MjsKey, MjsElementRef<raw::MjsKey>> mjsKey(m, "MjsKey");
  py::class_<raw::MjsNumeric, MjsElementRef<raw::MjsNumeric>> mjsNumeric(
      m, "MjsNumeric");
  py::class_<raw::MjsPair, MjsElementRef<raw::MjsPair>> mjsPair(m, "MjsPair");
  py::class_<raw::MjsExclude, MjsElementRef<raw::MjsExclude>> mjsExclude(
      m, "MjsExclude");
  py::class_<raw::MjsEquality, MjsElementRef<raw::MjsEquality>> mjsEquality(
      m, "MjsEquality");
  py::class_<raw::MjsTendon, MjsElementRef<raw::MjsTendon>> mjsTendon(
      m, "MjsTendon");
  py::class_<raw::MjsSensor, MjsElementRef<raw::MjsSensor>> mjsSensor(
      m, "MjsSensor");
  py::class_<raw::MjsActuator, MjsElementRef<raw::MjsActuator>> mjsActuator(
      m, "MjsActuator");
  py::class_<raw::MjsPlugin, MjsElementRef<raw::MjsPlugin>> mjsPlugin(
      m, "MjsPlugin");
  py::class_<raw::MjsOrientation> mjsOrientation(m, "MjsOrientation");
  py::class_<raw::MjsWrap, MjsElementRef<raw::MjsWrap>> mjsWrap(m, "MjsWrap");
  py::class_<raw::MjOption> mjOption(m, "MjOption");
  py::class_<raw::MjStatistic> mjStatistic(m, "MjStatistic");
  py::class_<raw::MjVisual> mjVisual(m, "MjVisual");
  py::class_<raw::MjVisualHeadlight> mjVisualHeadlight(m, "MjVisualHeadlight");
  py::class_<raw::MjVisualRgba> mjVisualRgba(m, "MjVisualRgba");
  py::class_<raw::MjsCompiler> mjsCompiler(m, "MjsCompiler");
  py::class_<raw::MjsAuthored> mjsAuthored(m, "MjsAuthored");
  DefineArray<char>(m, "MjCharVec");
  DefineArray<std::string>(m, "MjStringVec");
  DefineArray<std::byte>(m, "MjByteVec");
  DefineArray<double>(m, "MjDoubleVec");
  DefineArray<float>(m, "MjFloatVec");
  DefineArray<int>(m, "MjIntVec");

  // ============================= MJVFS =====================================
  py::class_<MjVfs>(m, "MjVfs")
      .def(py::init<>())
      .def("close", &MjVfs::Close)
      .def("__enter__", [](MjVfs& self) -> MjVfs& { return self; })
      .def("__exit__", [](MjVfs& self, py::object, py::object,
                          py::object) { self.Close(); })
      .def("__setitem__",
           [](MjVfs& self, const std::string& name, py::bytes data) {
             if (!self.is_open()) {
               throw std::runtime_error("VFS is closed");
             }
             std::string_view buffer = data;
             const int err = mj_addBufferVFS(self.get(), name.c_str(),
                                             buffer.data(), buffer.size());
             if (err == 2) {
               throw py::value_error("Repeated file name in VFS: " + name);
             } else if (err) {
               throw py::value_error("Failed to add buffer to VFS: " + name);
             }
           })
      .def("__delitem__",
           [](MjVfs& self, const std::string& name) {
             if (!self.is_open()) {
               throw std::runtime_error("VFS is closed");
             }
             if (mj_deleteFileVFS(self.get(), name.c_str())) {
               throw py::key_error(name);
             }
           })
      .def("__contains__", [](MjVfs& self, const std::string& name) {
        if (!self.is_open()) {
          throw std::runtime_error("VFS is closed");
        }
        return mj_containsBufferVFS(self.get(), name.c_str()) == 1;
      });

  // ============================= MJSPEC =====================================
  mjSpec.def(py::init<>());
  mjSpec.def_property_readonly(
      "parent", [](MjSpec& self) -> MjSpec* { return self.parent; });
  mjSpec.def_static(
      "from_file",
      [](std::string& filename,
         std::optional<std::unordered_map<std::string, py::bytes>>& include,
         std::optional<py::dict>& assets, MjVfs* vfs) -> MjSpec {
        if (vfs && (include.has_value() || assets.has_value())) {
          throw py::value_error(
              "Cannot specify both 'vfs' and 'include'/'assets'.");
        }
        mjVFS* vfs_ptr = GetOpenVfs(vfs);
        if (vfs_ptr) {
          raw::MjSpec* spec;
          {
            py::gil_scoped_release no_gil;
            char error[1024];
            spec = InterceptMjErrors(mj_parse)(
                filename.c_str(), nullptr, vfs_ptr, error, sizeof(error));
            if (!spec) {
              throw py::value_error(error);
            }
          }
          return MjSpec(spec);
        }
        const auto files = _impl::ConvertAssetsDict(include);
        raw::MjSpec* spec;
        {
          py::gil_scoped_release no_gil;
          char error[1024];
          spec = LoadSpecFileImpl(
              filename, files,
              [&error](const char* filename, const mjVFS* vfs) {
                return InterceptMjErrors(mj_parse)(filename, nullptr, vfs,
                                                   error, sizeof(error));
              });
          if (!spec) {
            throw py::value_error(error);
          }
        }
        if (assets.has_value()) {
          return MjSpec(spec, assets.value());
        }
        return MjSpec(spec);
      },
      py::arg("filename"), py::arg("include") = py::none(),
      py::arg("assets") = py::none(), py::arg("vfs") = py::none(),
      R"mydelimiter(
    Creates a spec from an XML file.

    Parameters
    ----------
    filename : str
        Path to the XML file.
    include : dict, optional
        A dictionary of xml files included by the model. The keys are file names
        and the values are file contents.
    assets : dict, optional
        A dictionary of assets to be used by the spec. The keys are asset names
        and the values are asset contents.
    vfs : MjVfs, optional
        A VFS to use for resolving includes and assets. Cannot be used with
        include or assets.
  )mydelimiter",
      py::return_value_policy::move);
  mjSpec.def_static(
      "from_string",
      [](std::string& xml,
         std::optional<std::unordered_map<std::string, py::bytes>>& include,
         std::optional<py::dict>& assets, MjVfs* vfs) -> MjSpec {
        if (vfs && (include.has_value() || assets.has_value())) {
          throw py::value_error(
              "Cannot specify both 'vfs' and 'include'/'assets'.");
        }
        mjVFS* vfs_ptr = GetOpenVfs(vfs);
        if (vfs_ptr) {
          raw::MjSpec* spec;
          {
            py::gil_scoped_release no_gil;
            char error[1024];
            spec = InterceptMjErrors(mj_parseXMLString)(xml.c_str(), vfs_ptr,
                                                        error, sizeof(error));
            if (!spec) {
              throw py::value_error(error);
            }
          }
          return MjSpec(spec);
        }
        auto files = _impl::ConvertAssetsDict(include);
        raw::MjSpec* spec;
        {
          py::gil_scoped_release no_gil;
          std::string model_filename = "model_.xml";
          if (include.has_value()) {
            while (include->find(model_filename) != include->end()) {
              model_filename =
                  model_filename.substr(0, model_filename.size() - 4) + "_.xml";
            }
          }
          files.emplace_back(model_filename.c_str(), xml.c_str(), xml.length());
          char error[1024];
          spec = LoadSpecFileImpl(
              model_filename, files,
              [&error](const char* filename, const mjVFS* vfs) {
                return InterceptMjErrors(mj_parseXML)(filename, vfs, error,
                                                      sizeof(error));
              });
          if (!spec) {
            throw py::value_error(error);
          }
        }
        if (assets.has_value()) {
          return MjSpec(spec, assets.value());
        }
        return MjSpec(spec);
      },
      py::arg("xml"), py::arg("include") = py::none(),
      py::arg("assets") = py::none(), py::arg("vfs") = py::none(),
      R"mydelimiter(
    Creates a spec from an XML string.

    Parameters
    ----------
    xml : str
        XML string.
    include : dict, optional
        A dictionary of xml files included by the model. The keys are file names
        and the values are file contents.
    assets : dict, optional
        A dictionary of assets to be used by the spec. The keys are asset names
        and the values are asset contents.
    vfs : MjVfs, optional
        A VFS to use for resolving includes and assets. Cannot be used with
        include or assets.
  )mydelimiter",
      py::return_value_policy::move);
  mjSpec.def(
      "recompile",
      [mjmodel_mjdata_from_spec_ptr](MjSpec& self, py::object m, py::object d,
                                     std::optional<MjVfs*> vfs) {
        mjVFS* user_vfs = GetOpenVfs(vfs);
        std::optional<mjVFS> local_vfs;
        mjVFS* vfs_ptr = self.AssetsVfs(user_vfs, local_vfs);
        struct LocalVfsGuard {
          mjVFS* vfs;
          bool active;
          ~LocalVfsGuard() {
            if (active) {
              mj_deleteVFS(vfs);
            }
          }
        } guard{vfs_ptr, local_vfs.has_value()};
        return mjmodel_mjdata_from_spec_ptr(
            reinterpret_cast<uintptr_t>(self.ptr), m, d,
            reinterpret_cast<uintptr_t>(vfs_ptr));
      },
      py::arg("m"), py::arg("d"), py::arg("vfs") = py::none());
  mjSpec.def("copy", [](const MjSpec& self) -> MjSpec { return MjSpec(self); });
  mjSpec.def_property_readonly("_address", [](const MjSpec& self) {
    return reinterpret_cast<uintptr_t>(self.ptr);
  });
  mjSpec.def_property_readonly(
      "timer",
      [](MjSpec& self) -> MjDouble9 {
        return MjDouble9(const_cast<double*>(mjs_getTimer(self.ptr)));
      },
      py::return_value_policy::reference_internal);
  mjSpec.def_property(
      "copy_during_attach",
      [](MjSpec& self) {
        throw pybind11::value_error("copy_during_attach can only be set.");
      },
      [](MjSpec& self, bool deepcopy) {
        return mjs_setDeepCopy(self.ptr, deepcopy);
      });
  mjSpec.def_property_readonly(
      "worldbody",
      [](MjSpec& self) -> raw::MjsBody* {
        return mjs_findBody(self.ptr, "world");
      },
      py::return_value_policy::reference_internal);
  mjSpec.def(
      "find_default",
      [](MjSpec& self, std::string& classname) -> raw::MjsDefault* {
        return mjs_findDefault(self.ptr, classname.c_str());
      },
      py::return_value_policy::reference_internal);
  mjSpec.def(
      "compile",
      [mjmodel_from_raw_ptr](MjSpec& self,
                             std::optional<MjVfs*> vfs) -> py::object {
        mjVFS* vfs_ptr = GetOpenVfs(vfs);
        return mjmodel_from_raw_ptr(
            reinterpret_cast<uintptr_t>(self.Compile(vfs_ptr)));
      },
      py::arg("vfs") = py::none());
  mjSpec.def(
      "fuse_static",
      [](MjSpec& self, std::optional<MjVfs*> vfs) {
        return self.FuseStatic(GetOpenVfs(vfs));
      },
      py::arg("vfs") = py::none(),
      R"mydelimiter(
    Fuses the static bodies of the spec with their parents.

    Each fused body becomes a frame in its parent, holding what the body held.
    Objects referring to the fused bodies must not be used afterwards. Assets
    are read as in compile(). Returns the number of bodies fused.

    Parameters
    ----------
    vfs : MjVfs, optional
        A VFS to read assets from. Cannot be used with assets.
    )mydelimiter");
  mjSpec.def(
      "discard_visual",
      [](MjSpec& self, std::optional<MjVfs*> vfs) {
        return self.DiscardVisual(GetOpenVfs(vfs));
      },
      py::arg("vfs") = py::none(),
      R"mydelimiter(
    Discards the visual elements of the spec.

    All materials and textures are discarded, as are the geoms which do not
    collide and are not referenced, and the meshes which are then not used.
    Objects referring to the discarded elements must not be used afterwards.
    Assets are read as in compile(), and only if body inertia is inferred from
    a discarded geom. Returns the number of elements discarded.

    Parameters
    ----------
    vfs : MjVfs, optional
        A VFS to read assets from. Cannot be used with assets.
    )mydelimiter");
  mjSpec.def(
      "dedup_assets", [](MjSpec& self) { return self.DedupAssets(); },
      R"mydelimiter(
    Merges the duplicate textures and meshes of the spec.

    Textures and meshes which would compile to the same asset, with the same
    resolved file or buffers and the same parameters, are merged: the first of
    each group keeps its name, every reference to the others is rewritten to it
    and the others are deleted. Objects referring to the deleted elements must
    not be used afterwards. Nothing is read from disk. Returns the number of
    assets merged.
    )mydelimiter");
  mjSpec.def(
      "adopt_inertial",
      [](MjSpec& self, raw::MjsBody* body, std::optional<MjVfs*> vfs) {
        self.AdoptInertial(body, GetOpenVfs(vfs));
      },
      py::arg("body"), py::arg("vfs") = py::none(),
      R"mydelimiter(
    Makes the inertial which compilation infers for a body part of the spec.

    The mass, inertia and inertial frame calculated from the geoms of the body
    become its explicit inertial, and no longer follow changes to the geoms.
    Assets are read as in compile().

    Parameters
    ----------
    body : MjsBody
        A body of this spec.
    vfs : MjVfs, optional
        A VFS to read assets from. Cannot be used with assets.
    )mydelimiter");
  mjSpec.def(
      "copy_back",
      [](MjSpec& self, py::object model) {
        auto& wrapper = py::cast<_impl::MjModelWrapper&>(model);
        if (!mj_copyBack(self.ptr, wrapper.get())) {
          throw py::value_error(mjs_getError(self.ptr));
        }
      },
      py::arg("model"),
      R"mydelimiter(
    Writes to the spec what was changed in a model compiled from it.

    Each value of the model which is not the one that the spec was compiled to
    is written to the spec as what compiles to it, so that it is in the models
    which are compiled from the spec afterwards and in the XML which is saved.
    Raises ValueError, and writes nothing, if a change cannot be expressed in
    the spec.

    Parameters
    ----------
    model : MjModel
        The model which was last compiled from this spec.
    )mydelimiter");
  mjSpec.def_property(
      "assets", [](MjSpec& self) -> py::dict { return self.assets; },
      [](MjSpec& self, py::dict& assets) {
        for (auto item : assets) {
          self.assets[item.first] = item.second;
        };
      },
      py::return_value_policy::reference_internal);
  mjSpec.def_property(
      "override_assets",
      [](MjSpec& self) -> bool { return self.override_assets; },
      [](MjSpec& self, bool override_assets) {
        self.override_assets = override_assets;
      });
  mjSpec.def("to_xml", [](MjSpec& self) -> std::string {
    mj_deleteModel(self.Compile());
    std::array<char, 1024> err;
    int size = mj_saveXMLString(self.ptr, nullptr, 0, nullptr, 0);
    std::unique_ptr<char[]> buf(new char[size + 1]);
    buf[0] = '\0';
    err[0] = '\0';
    mj_saveXMLString(self.ptr, buf.get(), size + 1, err.data(), err.size());
    if (err[0] != '\0') {
      throw FatalError(std::string(err.data()));
    }
    return std::string(buf.get());
  });
  mjSpec.def("to_file", [](MjSpec& self, std::string& file) {
    std::array<char, 1024> err;
    if (mj_saveXML(self.ptr, file.c_str(), err.data(), err.size()) < 0) {
      throw FatalError(std::string(err.data()));
    }
  });
  mjSpec.def(
      "encode",
      [](MjSpec& self, std::string filename, std::optional<py::object> model,
         std::optional<std::string> content_type) -> mjtSize {
        raw::MjModel* m = nullptr;
        if (model.has_value() && !model->is_none()) {
          auto& wrapper = py::cast<_impl::MjModelWrapper&>(*model);
          m = wrapper.get();
        }

        mjVFS vfs;
        mjVFS* vfs_ptr = nullptr;
        if (!self.assets.empty()) {
          mj_defaultVFS(&vfs);
          vfs_ptr = &vfs;
          for (const auto& asset : self.assets) {
            std::string buffer_name = py::cast<std::string>(asset.first);
            std::string buffer = py::cast<std::string>(asset.second);
            mj_addBufferVFS(vfs_ptr, buffer_name.c_str(), buffer.c_str(),
                            buffer.size());
          }
        }

        std::array<char, 1024> err;
        err[0] = '\0';
        const char* ct =
            content_type.has_value() ? content_type->c_str() : nullptr;
        mjtSize nbytes = mj_encode(self.ptr, m, filename.c_str(), ct, vfs_ptr,
                                   err.data(), err.size());

        if (vfs_ptr) {
          mj_deleteVFS(vfs_ptr);
        }

        if (nbytes < 0) {
          throw FatalError(std::string(err.data()));
        }
        return nbytes;
      },
      py::arg("filename"), py::arg("model") = py::none(),
      py::arg("content_type") = py::none());
  mjSpec.def(
      "add_default",
      [](MjSpec* spec, std::string& classname,
         raw::MjsDefault* parent) -> raw::MjsDefault* {
        return mjs_addDefault(spec->ptr, classname.c_str(), parent);
      },
      py::return_value_policy::reference_internal);
  mjSpec.def("delete", [](MjSpec& self, raw::MjsDefault& def) {
    if (mjs_delete(self.ptr, def.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjSpec.def_property_readonly(
      "default",
      [](MjSpec& self) -> raw::MjsDefault* {
        return mjs_getSpecDefault(self.ptr);
      },
      py::return_value_policy::reference_internal);
  mjSpec.def("delete", [](MjSpec& self, raw::MjsBody& body) {
    if (mjs_delete(self.ptr, body.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjSpec.def(
      "attach",
      [](MjSpec& self, MjSpec& child, std::optional<std::string>& prefix,
         std::optional<std::string>& suffix, std::optional<py::object>& site,
         std::optional<py::object>& frame) -> raw::MjsFrame* {
        if (!frame.has_value() && !site.has_value()) {
          throw pybind11::value_error(
              "One of frame or site must be specified.");
        }
        if (frame.has_value() && site.has_value()) {
          throw pybind11::value_error(
              "Only one of frame or site can be specified.");
        }
        const char* p = prefix.has_value() ? prefix.value().c_str() : "/";
        const char* s = suffix.has_value() ? suffix.value().c_str() : "";
        raw::MjsElement* parent_elem = nullptr;
        if (frame.has_value()) {
          raw::MjsFrame* frame_ptr = nullptr;
          try {
            frame_ptr = frame->cast<raw::MjsFrame*>();
          } catch (const py::cast_error& e) {
            frame_ptr =
                mjs_findFrame(self.ptr, frame->cast<std::string>().c_str());
          }
          if (!frame_ptr) {
            throw pybind11::value_error("Frame not found.");
          }
          if (mjs_getSpec(frame_ptr->element) != self.ptr) {
            throw pybind11::value_error(
                "Frame spec does not match parent spec.");
          }
          parent_elem = frame_ptr->element;
        }
        if (site.has_value()) {
          raw::MjsSite* site_ptr = nullptr;
          try {
            site_ptr = site->cast<raw::MjsSite*>();
          } catch (const py::cast_error& e) {
            site_ptr = mjs_asSite(mjs_findElement(
                self.ptr, mjOBJ_SITE, site->cast<std::string>().c_str()));
          }
          if (!site_ptr) {
            throw pybind11::value_error("Site not found.");
          }
          if (mjs_getSpec(site_ptr->element) != self.ptr) {
            throw pybind11::value_error(
                "Site spec does not match parent spec.");
          }
          parent_elem = site_ptr->element;
        }
        // add prefix and suffix to the assets keys
        std::string pre(p);
        std::string suf(s);
        std::vector<std::pair<py::str, py::object>> new_assets;
        new_assets.reserve(child.assets.size());
        for (const auto& asset : child.assets) {
          std::string key = asset.first.cast<std::string>();
          std::string asset_name = addPrefixAndSuffix(key, pre, suf);
          py::str py_asset_name(asset_name);
          if (self.assets.contains(py_asset_name) && !self.override_assets) {
            throw pybind11::value_error("Asset " + key +
                                        " already exists in parent spec.");
          }
          new_assets.emplace_back(
              std::move(py_asset_name),
              py::reinterpret_borrow<py::object>(asset.second));
        }
        raw::MjsElement* attached_frame =
            mjs_attach(parent_elem, child.ptr->element, p, s);
        if (!attached_frame) {
          throw pybind11::value_error(mjs_getError(self.ptr));
        }
        for (const auto& [asset_name, asset_val] : new_assets) {
          self.assets[asset_name] = asset_val;
        }
        raw::MjsElement* mesh = mjs_firstElement(child.ptr, mjOBJ_MESH);
        raw::MjsElement* tex = mjs_firstElement(child.ptr, mjOBJ_TEXTURE);
        raw::MjsElement* parent_mesh = mjs_firstElement(self.ptr, mjOBJ_MESH);
        raw::MjsElement* parent_tex = mjs_firstElement(self.ptr, mjOBJ_TEXTURE);
        bool child_has_assets = mesh || tex;
        bool parent_has_assets = parent_mesh || parent_tex;
        bool child_use_asset_dict = !child.assets.empty();
        bool parent_use_asset_dict = !self.assets.empty();
        if (!child_use_asset_dict && child_has_assets &&
            parent_use_asset_dict) {
          PyErr_WarnEx(
              PyExc_Warning,
              "Attaching a child without asset dict to a parent with an "
              "asset dict might result in missing assets when attaching again.",
              1);
        }
        if (!parent_use_asset_dict && parent_has_assets &&
            child_use_asset_dict) {
          PyErr_WarnEx(
              PyExc_Warning,
              "Attaching a child with asset dict to a parent without an "
              "asset dict might result in missing assets when attaching again.",
              1);
        }
        if (child_use_asset_dict) {
          while (mesh) {
            std::string file = mjs_getString(mjs_asMesh(mesh)->file);
            if (!file.empty()) {
              std::string mesh_file = addPrefixAndSuffix(file, pre, suf);
              mjs_setString(mjs_asMesh(mesh)->file, mesh_file.c_str());
            }
            mesh = mjs_nextElement(child.ptr, mesh);
          }
          while (tex) {
            std::string file = mjs_getString(mjs_asTexture(tex)->file);
            if (!file.empty()) {
              std::string tex_file = addPrefixAndSuffix(file, pre, suf);
              mjs_setString(mjs_asTexture(tex)->file, tex_file.c_str());
            }
            tex = mjs_nextElement(child.ptr, tex);
          }
        }
        child.parent = &self;
        return mjs_asFrame(attached_frame);
      },
      py::arg("child"), py::arg("prefix") = py::none(),
      py::arg("suffix") = py::none(), py::arg("site") = py::none(),
      py::arg("frame") = py::none(),
      py::return_value_policy::reference_internal);
  mjSpec.def(
      "activate_plugin",
      [](MjSpec& self, std::string& name) {
        mjs_activatePlugin(self.ptr, name.c_str());
      },
      py::arg("name"), py::return_value_policy::reference_internal);
  mjSpec.def_static(
      "resolve_orientation",
      [](bool degree, const MjTypeVec<char>& sequence,
         const raw::MjsOrientation* orientation) -> std::array<double, 4> {
        std::array<double, 4> quat = {0, 0, 0, 0};
        const char* err = mjs_resolveOrientation(quat.data(), degree,
                                                 sequence.ptr, orientation);
        if (err) {
          throw pybind11::value_error(err);
        }
        return quat;
      },
      py::arg("degree"), py::arg("sequence") = py::none(),
      py::arg("orientation"), py::return_value_policy::copy);
  mjSpec.def_property(
      "meshdir",
      [](MjSpec& self) -> std::string_view {
        return *self.ptr->compiler.meshdir;
      },
      [](MjSpec& self, std::string_view meshdir) {
        *(self.ptr->compiler.meshdir) = meshdir;
      });
  mjSpec.def_property(
      "texturedir",
      [](MjSpec& self) -> std::string_view {
        return *self.ptr->compiler.texturedir;
      },
      [](MjSpec& self, std::string_view texturedir) {
        *(self.ptr->compiler.texturedir) = texturedir;
      });

  // ============================= MJSBODY =====================================
  mjsBody.def(
      "add_freejoint",
      [](raw::MjsBody& self, py::kwargs kwargs) -> raw::MjsJoint* {
        auto out = mjs_addFreeJoint(&self);
        py::dict kwarg_dict = kwargs;
        for (auto item : kwarg_dict) {
          std::string key = py::str(item.first);
          if (key == "align") {
            try {
              out->align =
                  static_cast<mjtAlignFree>(kwargs["align"].cast<int>());
            } catch (const py::cast_error& e) {
              throw pybind11::value_error("align is the wrong type.");
            }
          } else if (key == "name") {
            try {
              mjs_setName(out->element,
                          kwargs["name"].cast<std::string>().c_str());
            } catch (const py::cast_error& e) {
              throw pybind11::value_error("name is the wrong type.");
            }
          } else if (key == "group") {
            try {
              out->group = kwargs["group"].cast<int>();
            } catch (const py::cast_error& e) {
              throw pybind11::value_error("group is the wrong type.");
            }
          } else {
            throw pybind11::type_error(
                "Invalid " + key +
                " keyword argument. Valid options are: align, group, name.");
          }
        }
        return out;
      },
      py::return_value_policy::reference_internal);
  mjsBody.def("set_frame", [](raw::MjsBody& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsBody.def_property(
      "classname",
      [](raw::MjsBody& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsBody& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "find_all",
      [](raw::MjsBody& self, mjtObj objtype) -> py::list {
        return FindAllImpl(self, objtype, true);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "find_all",
      [](raw::MjsBody& self, std::string& name) -> py::list {
        mjtObj objtype = mjOBJ_UNKNOWN;
        if (name == "body") {
          objtype = mjOBJ_BODY;
        } else if (name == "frame") {
          objtype = mjOBJ_FRAME;
        } else if (name == "geom") {
          objtype = mjOBJ_GEOM;
        } else if (name == "site") {
          objtype = mjOBJ_SITE;
        } else if (name == "joint") {
          objtype = mjOBJ_JOINT;
        } else if (name == "light") {
          objtype = mjOBJ_LIGHT;
        } else if (name == "camera") {
          objtype = mjOBJ_CAMERA;
        } else {
          throw pybind11::value_error(
              "body.find_all supports the types: body, frame, geom, site, "
              "joint, light, camera.");
        }
        return FindAllImpl(self, objtype, true);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "make_flex",
      [](raw::MjsBody& self, const std::string& name,
         std::optional<std::string> type, int dim,
         std::optional<std::string> dof, std::optional<std::vector<int>> count,
         std::optional<std::vector<int>> cellcount,
         std::optional<std::vector<double>> spacing,
         std::optional<std::vector<double>> scale, double radius, double mass,
         double inertiabox, int equality, int rigid, int flatskin,
         int elastic2d, std::optional<std::vector<double>> pos,
         std::optional<std::vector<double>> quat,
         std::optional<std::vector<double>> origin,
         std::optional<std::string> file, std::optional<std::string> mesh,
         MjVfs* vfs) -> raw::MjsFlex* {
        const char* type_str = type.has_value() ? type->c_str() : nullptr;
        const char* dof_str = dof.has_value() ? dof->c_str() : nullptr;
        const char* file_str = file.has_value() ? file->c_str() : nullptr;
        if (mesh.has_value()) {
          if (file.has_value()) {
            throw pybind11::value_error(
                "both 'file' and 'mesh' specified");
          }
          if (!mjs_findElement(mjs_getSpec(self.element), mjOBJ_MESH,
                               mesh->c_str())) {
            throw pybind11::value_error("mesh '" + *mesh + "' not found");
          }
          file_str = mesh->c_str();
        }
        const mjVFS* vfs_ptr = GetOpenVfs(vfs);

        int count_arr[3] = {10, 10, 10};
        if (count.has_value()) {
          if (count->size() != 3) {
            throw pybind11::value_error("count must have 3 elements");
          }
          for (int i = 0; i < 3; i++) count_arr[i] = (*count)[i];
        }
        const int* count_ptr = count.has_value() ? count_arr : nullptr;

        int cellcount_arr[3] = {-1, -1, -1};
        if (cellcount.has_value()) {
          if (cellcount->size() != 3) {
            throw pybind11::value_error("cellcount must have 3 elements");
          }
          for (int i = 0; i < 3; i++) cellcount_arr[i] = (*cellcount)[i];
        }
        const int* cellcount_ptr =
            cellcount.has_value() ? cellcount_arr : nullptr;

        double spacing_arr[3] = {0.02, 0.02, 0.02};
        if (spacing.has_value()) {
          if (spacing->size() != 3) {
            throw pybind11::value_error("spacing must have 3 elements");
          }
          for (int i = 0; i < 3; i++) spacing_arr[i] = (*spacing)[i];
        }
        const double* spacing_ptr = spacing.has_value() ? spacing_arr : nullptr;

        double scale_arr[3] = {1, 1, 1};
        if (scale.has_value()) {
          if (scale->size() != 3) {
            throw pybind11::value_error("scale must have 3 elements");
          }
          for (int i = 0; i < 3; i++) scale_arr[i] = (*scale)[i];
        }
        const double* scale_ptr = scale.has_value() ? scale_arr : nullptr;

        double pos_arr[3] = {0, 0, 0};
        if (pos.has_value()) {
          if (pos->size() != 3) {
            throw pybind11::value_error("pos must have 3 elements");
          }
          for (int i = 0; i < 3; i++) pos_arr[i] = (*pos)[i];
        }
        const double* pos_ptr = pos.has_value() ? pos_arr : nullptr;

        double quat_arr[4] = {1, 0, 0, 0};
        if (quat.has_value()) {
          if (quat->size() != 4) {
            throw pybind11::value_error("quat must have 4 elements");
          }
          for (int i = 0; i < 4; i++) quat_arr[i] = (*quat)[i];
        }
        const double* quat_ptr = quat.has_value() ? quat_arr : nullptr;

        double origin_arr[3] = {0, 0, 0};
        if (origin.has_value()) {
          if (origin->size() != 3) {
            throw pybind11::value_error("origin must have 3 elements");
          }
          for (int i = 0; i < 3; i++) origin_arr[i] = (*origin)[i];
        }
        const double* origin_ptr = origin.has_value() ? origin_arr : nullptr;

        auto out =
            mjs_makeFlex(&self, name.c_str(), type_str, dim, dof_str, count_ptr,
                         cellcount_ptr, spacing_ptr, scale_ptr, radius, mass,
                         inertiabox, equality, rigid, flatskin, elastic2d,
                         pos_ptr, quat_ptr, origin_ptr, file_str, vfs_ptr);
        if (!out) {
          raw::MjSpec* spec = mjs_getSpec(self.element);
          throw pybind11::value_error(mjs_getError(spec));
        }
        return out;
      },
      py::arg("name"), py::arg("type") = py::none(), py::arg("dim") = 3,
      py::arg("dof") = py::none(), py::arg("count") = py::none(),
      py::arg("cellcount") = py::none(), py::arg("spacing") = py::none(),
      py::arg("scale") = py::none(), py::arg("radius") = 0.0,
      py::arg("mass") = 1.0, py::arg("inertiabox") = 0.005,
      py::arg("equality") = 0, py::arg("rigid") = 0, py::arg("flatskin") = 0,
      py::arg("elastic2d") = 0, py::arg("pos") = py::none(),
      py::arg("quat") = py::none(), py::arg("origin") = py::none(),
      py::arg("file") = py::none(), py::arg("mesh") = py::none(),
      py::arg("vfs") = py::none(), py::return_value_policy::reference_internal);
  mjsBody.def(
      "find_child",
      [](raw::MjsBody& self, std::string& name) -> raw::MjsBody* {
        return mjs_findChild(&self, name.c_str());
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_body",
      [](raw::MjsBody& self) -> raw::MjsBody* {
        return mjs_asBody(mjs_firstChild(&self, mjOBJ_BODY, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_body",
      [](raw::MjsBody& self, raw::MjsBody& child) -> raw::MjsBody* {
        return mjs_asBody(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "bodies",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_BODY, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_camera",
      [](raw::MjsBody& self) -> raw::MjsCamera* {
        return mjs_asCamera(mjs_firstChild(&self, mjOBJ_CAMERA, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_camera",
      [](raw::MjsBody& self, raw::MjsCamera& child) -> raw::MjsCamera* {
        return mjs_asCamera(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "cameras",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_CAMERA, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_light",
      [](raw::MjsBody& self) -> raw::MjsLight* {
        return mjs_asLight(mjs_firstChild(&self, mjOBJ_LIGHT, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_light",
      [](raw::MjsBody& self, raw::MjsLight& child) -> raw::MjsLight* {
        return mjs_asLight(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "lights",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_LIGHT, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_joint",
      [](raw::MjsBody& self) -> raw::MjsJoint* {
        return mjs_asJoint(mjs_firstChild(&self, mjOBJ_JOINT, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_joint",
      [](raw::MjsBody& self, raw::MjsJoint& child) -> raw::MjsJoint* {
        return mjs_asJoint(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "joints",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_JOINT, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_geom",
      [](raw::MjsBody& self) -> raw::MjsGeom* {
        return mjs_asGeom(mjs_firstChild(&self, mjOBJ_GEOM, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_geom",
      [](raw::MjsBody& self, raw::MjsGeom& child) -> raw::MjsGeom* {
        return mjs_asGeom(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "geoms",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_GEOM, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_site",
      [](raw::MjsBody& self) -> raw::MjsSite* {
        return mjs_asSite(mjs_firstChild(&self, mjOBJ_SITE, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_site",
      [](raw::MjsBody& self, raw::MjsSite& child) -> raw::MjsSite* {
        return mjs_asSite(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "sites",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_SITE, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "first_frame",
      [](raw::MjsBody& self) -> raw::MjsFrame* {
        return mjs_asFrame(mjs_firstChild(&self, mjOBJ_FRAME, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "next_frame",
      [](raw::MjsBody& self, raw::MjsFrame& child) -> raw::MjsFrame* {
        return mjs_asFrame(mjs_nextChild(&self, child.element, false));
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "frames",
      [](raw::MjsBody& self) -> py::list {
        return FindAllImpl(self, mjOBJ_FRAME, false);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "parent",
      [](raw::MjsBody& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "attach_frame",
      [](raw::MjsBody& self, raw::MjsFrame& frame,
         std::optional<std::string>& prefix,
         std::optional<std::string>& suffix) -> raw::MjsFrame* {
        const char* p = prefix.has_value() ? prefix.value().c_str() : "";
        const char* s = suffix.has_value() ? suffix.value().c_str() : "";
        auto new_frame = mjs_attach(self.element, frame.element, p, s);
        if (!new_frame) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
        }
        return mjs_asFrame(new_frame);
      },
      py::arg("frame"), py::arg("prefix") = py::none(),
      py::arg("suffix") = py::none(),
      py::return_value_policy::reference_internal);
  mjsBody.def(
      "to_frame",
      [](raw::MjsBody* self) -> raw::MjsFrame* {
        raw::MjsFrame* frame = mjs_bodyToFrame(&self);
        if (!frame) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
        // TODO: set the body to py::none
        return frame;
      },
      py::return_value_policy::reference_internal);
  mjsBody.def_property_readonly(
      "frame",
      [](raw::MjsBody* self) -> raw::MjsFrame* {
        return mjs_getFrame(self->element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSFRAME ====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsFrame& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsFrame.def("set_frame", [](raw::MjsFrame& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsFrame.def_property_readonly(
      "parent",
      [](raw::MjsFrame& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsFrame.def(
      "attach_body",
      [](raw::MjsFrame& self, raw::MjsBody& body,
         std::optional<std::string>& prefix,
         std::optional<std::string>& suffix) -> raw::MjsBody* {
        const char* p = prefix.has_value() ? prefix.value().c_str() : "";
        const char* s = suffix.has_value() ? suffix.value().c_str() : "";
        auto new_body = mjs_attach(self.element, body.element, p, s);
        if (!new_body) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
        }
        return mjs_asBody(new_body);
      },
      py::arg("body"), py::arg("prefix") = py::none(),
      py::arg("suffix") = py::none(),
      py::return_value_policy::reference_internal);
  mjsFrame.def_property_readonly(
      "frame",
      [](raw::MjsFrame& self) -> raw::MjsFrame* {
        return mjs_getFrame(self.element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSGEOM =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsGeom& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsGeom.def("set_frame", [](raw::MjsGeom& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsGeom.def_property_readonly(
      "parent",
      [](raw::MjsGeom& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsGeom.def_property(
      "classname",
      [](raw::MjsGeom& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsGeom& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsGeom.def_property_readonly(
      "frame",
      [](raw::MjsGeom& self) -> raw::MjsFrame* {
        return mjs_getFrame(self.element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSJOINT ====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsJoint& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsJoint.def("set_frame", [](raw::MjsJoint& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsJoint.def_property_readonly(
      "parent",
      [](raw::MjsJoint& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsJoint.def_property(
      "classname",
      [](raw::MjsJoint& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsJoint& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsJoint.def_property_readonly(
      "frame",
      [](raw::MjsJoint& self) -> raw::MjsFrame* {
        return mjs_getFrame(self.element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSSITE =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsSite& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsSite.def("set_frame", [](raw::MjsSite& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsSite.def_property_readonly(
      "parent",
      [](raw::MjsSite& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsSite.def_property(
      "classname",
      [](raw::MjsSite& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsSite& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsSite.def(
      "attach_body",
      [](raw::MjsSite& self, raw::MjsBody& body,
         std::optional<std::string>& prefix,
         std::optional<std::string>& suffix) -> raw::MjsBody* {
        const char* p = prefix.has_value() ? prefix.value().c_str() : "";
        const char* s = suffix.has_value() ? suffix.value().c_str() : "";
        auto new_body = mjs_attach(self.element, body.element, p, s);
        if (!new_body) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
        }
        return mjs_asBody(new_body);
      },
      py::arg("body"), py::arg("prefix") = py::none(),
      py::arg("suffix") = py::none(),
      py::return_value_policy::reference_internal);
  mjsSite.def_property_readonly(
      "frame",
      [](raw::MjsSite& self) -> raw::MjsFrame* {
        return mjs_getFrame(self.element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSCAMERA ===================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsCamera& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsCamera.def("set_frame", [](raw::MjsCamera& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsCamera.def_property_readonly(
      "parent",
      [](raw::MjsCamera& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsCamera.def_property(
      "classname",
      [](raw::MjsCamera& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsCamera& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsCamera.def_property_readonly(
      "frame",
      [](raw::MjsCamera& self) -> raw::MjsFrame* {
        return mjs_getFrame(self.element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSLIGHT ====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsLight& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsLight.def("set_frame", [](raw::MjsLight& self, raw::MjsFrame& frame) {
    if (mjs_setFrame(self.element, &frame) != 0) {
      throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
    }
  });
  mjsLight.def_property_readonly(
      "parent",
      [](raw::MjsLight& self) -> raw::MjsBody* {
        return mjs_getParent(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsLight.def_property(
      "classname",
      [](raw::MjsLight& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsLight& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsLight.def_property_readonly(
      "frame",
      [](raw::MjsLight& self) -> raw::MjsFrame* {
        return mjs_getFrame(self.element);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSMATERIAL =================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsMaterial& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsMaterial.def_property(
      "classname",
      [](raw::MjsMaterial& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsMaterial& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSMESH =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsMesh& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsMesh.def_property(
      "classname",
      [](raw::MjsMesh& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsMesh& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsMesh.def(
      "make_wedge",
      [](raw::MjsMesh* self, std::array<int, 2>& resolution,
         std::array<double, 2>& fov, double gamma) {
        double params[5] = {static_cast<double>(resolution[0]),
                            static_cast<double>(resolution[1]), fov[0], fov[1],
                            gamma};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_WEDGE, params, 5)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("resolution") = std::array<int, 2>{0, 0},
      py::arg("fov") = std::array<double, 2>{0, 0}, py::arg("gamma") = 0);
  mjsMesh.def(
      "make_sphere",
      [](raw::MjsMesh* self, int subdivision) {
        double params[1] = {static_cast<double>(subdivision)};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_SPHERE, params, 1)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("subdivision"));
  mjsMesh.def(
      "make_hemisphere",
      [](raw::MjsMesh* self, int resolution) {
        double params[1] = {static_cast<double>(resolution)};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_HEMISPHERE, params, 1)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("resolution"));
  mjsMesh.def(
      "make_cone",
      [](raw::MjsMesh* self, int nedge, double radius) {
        double params[2] = {static_cast<double>(nedge), radius};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_CONE, params, 2)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("nedge"), py::arg("radius"));
  mjsMesh.def(
      "make_supersphere",
      [](raw::MjsMesh* self, int resolution, double e, double n) {
        double params[3] = {static_cast<double>(resolution), e, n};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_SUPERSPHERE, params, 3)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("resolution"), py::arg("e"), py::arg("n"));
  mjsMesh.def(
      "make_supertorus",
      [](raw::MjsMesh* self, int resolution, double radius, double s,
         double t) {
        double params[4] = {static_cast<double>(resolution), radius, s, t};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_SUPERTORUS, params, 4)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("resolution"), py::arg("radius"), py::arg("s"), py::arg("t"));
  mjsMesh.def(
      "make_plate",
      [](raw::MjsMesh* self, std::array<int, 2>& resolution) {
        double params[2] = {static_cast<double>(resolution[0]),
                            static_cast<double>(resolution[1])};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_PLATE, params, 2)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("resolution") = std::array<int, 2>{0, 0});
  mjsMesh.def(
      "make_grid",
      [](raw::MjsMesh* self, std::vector<int> count,
         std::vector<double> spacing, int dim) {
        if (count.size() != 3 || spacing.size() != 3) {
          throw pybind11::value_error("count and spacing must have 3 elements");
        }
        double params[7] = {
            static_cast<double>(count[0]), static_cast<double>(count[1]),
            static_cast<double>(count[2]), spacing[0], spacing[1], spacing[2],
            static_cast<double>(dim)};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_GRID, params, 7)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("count"), py::arg("spacing"), py::arg("dim") = 3);
  mjsMesh.def(
      "make_box",
      [](raw::MjsMesh* self, std::vector<int> count,
         std::vector<double> spacing, int dim) {
        if (count.size() != 3 || spacing.size() != 3) {
          throw pybind11::value_error("count and spacing must have 3 elements");
        }
        double params[7] = {
            static_cast<double>(count[0]), static_cast<double>(count[1]),
            static_cast<double>(count[2]), spacing[0], spacing[1], spacing[2],
            static_cast<double>(dim)};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_BOX, params, 7)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("count"), py::arg("spacing"), py::arg("dim") = 3);
  mjsMesh.def(
      "make_cylinder",
      [](raw::MjsMesh* self, std::vector<int> count,
         std::vector<double> spacing, int dim) {
        if (count.size() != 3 || spacing.size() != 3) {
          throw pybind11::value_error("count and spacing must have 3 elements");
        }
        double params[7] = {
            static_cast<double>(count[0]), static_cast<double>(count[1]),
            static_cast<double>(count[2]), spacing[0], spacing[1], spacing[2],
            static_cast<double>(dim)};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_CYLINDER, params, 7)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("count"), py::arg("spacing"), py::arg("dim") = 3);
  mjsMesh.def(
      "make_ellipsoid",
      [](raw::MjsMesh* self, std::vector<int> count,
         std::vector<double> spacing, int dim) {
        if (count.size() != 3 || spacing.size() != 3) {
          throw pybind11::value_error("count and spacing must have 3 elements");
        }
        double params[7] = {
            static_cast<double>(count[0]), static_cast<double>(count[1]),
            static_cast<double>(count[2]), spacing[0], spacing[1], spacing[2],
            static_cast<double>(dim)};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_ELLIPSOID, params, 7)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("count"), py::arg("spacing"), py::arg("dim") = 3);
  mjsMesh.def(
      "make_square",
      [](raw::MjsMesh* self, std::vector<int> count,
         std::vector<double> spacing) {
        if (count.size() < 2 || spacing.size() < 2) {
          throw pybind11::value_error(
              "count and spacing must have at least 2 elements");
        }
        int c2 = (count.size() >= 3) ? count[2] : 1;
        double s2 = (spacing.size() >= 3) ? spacing[2] : 0.0;
        double params[6] = {
            static_cast<double>(count[0]), static_cast<double>(count[1]),
            static_cast<double>(c2), spacing[0], spacing[1], s2};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_SQUARE, params, 6)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("count"), py::arg("spacing"));
  mjsMesh.def(
      "make_disc",
      [](raw::MjsMesh* self, std::vector<int> count,
         std::vector<double> spacing) {
        if (count.size() < 2 || spacing.size() < 2) {
          throw pybind11::value_error(
              "count and spacing must have at least 2 elements");
        }
        int c2 = (count.size() >= 3) ? count[2] : 1;
        double s2 = (spacing.size() >= 3) ? spacing[2] : 0.0;
        double params[6] = {
            static_cast<double>(count[0]), static_cast<double>(count[1]),
            static_cast<double>(c2), spacing[0], spacing[1], s2};
        if (mjs_makeMesh(self, mjMESH_BUILTIN_DISC, params, 6)) {
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self->element)));
        }
      },
      py::arg("count"), py::arg("spacing"));

  // ============================= MJSPAIR =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsPair& obj) {
    mjs_delete(self.ptr, obj.element);
  });
  mjsPair.def_property(
      "classname",
      [](raw::MjsPair& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsPair& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSEQUAL ====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsEquality& obj) {
    mjs_delete(self.ptr, obj.element);
  });
  mjsEquality.def_property(
      "classname",
      [](raw::MjsEquality& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsEquality& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSACTUATOR =================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsActuator& obj) {
    mjs_delete(self.ptr, obj.element);
  });
  mjsActuator.def_property(
      "classname",
      [](raw::MjsActuator& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsActuator& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);
  mjsActuator.def("set_to_motor", [](raw::MjsActuator* self) {
    std::string err = mjs_setToMotor(self);
    if (!err.empty()) {
      throw pybind11::value_error(err);
    }
  });
  mjsActuator.def(
      "set_to_position",
      [](raw::MjsActuator* self, double kp, double kv, double dampratio,
         double timeconst, bool inheritrange) {
        std::string err = mjs_setToPosition(
            self, kp, kv == -1 ? nullptr : &kv,
            dampratio == -1 ? nullptr : &dampratio,
            timeconst == -1 ? nullptr : &timeconst, inheritrange);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("kp"), py::arg("kv") = -1, py::arg("dampratio") = -1,
      py::arg("timeconst") = -1, py::arg("inheritrange") = false);
  mjsActuator.def(
      "set_to_intvelocity",
      [](raw::MjsActuator* self, double kp, double kv, double dampratio,
         double timeconst, bool inheritrange) {
        std::string err = mjs_setToIntVelocity(
            self, kp, kv == -1 ? nullptr : &kv,
            dampratio == -1 ? nullptr : &dampratio,
            timeconst == -1 ? nullptr : &timeconst, inheritrange);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("kp"), py::arg("kv") = -1, py::arg("dampratio") = -1,
      py::arg("timeconst") = -1, py::arg("inheritrange") = false);
  mjsActuator.def(
      "set_to_velocity",
      [](raw::MjsActuator* self, double kv) {
        std::string err = mjs_setToVelocity(self, kv);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("kv"));
  mjsActuator.def(
      "set_to_orientation",
      [](raw::MjsActuator* self, double kp, double kv, double dampratio,
         int ctrlspec) {
        std::string err = mjs_setToOrientation(
            self, kp, kv == -1 ? nullptr : &kv,
            dampratio == -1 ? nullptr : &dampratio, ctrlspec);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("kp"), py::arg("kv") = -1, py::arg("dampratio") = -1,
      py::arg("ctrlspec") = 0);
  mjsActuator.def(
      "set_to_pid",
      [](raw::MjsActuator* self, double kp, double kv, double dampratio,
         double ki, double imax, double slewmax, double inheritrange,
         int ctrlspec) {
        std::string err = mjs_setToPID(
            self, kp, kv == -1 ? nullptr : &kv,
            dampratio == -1 ? nullptr : &dampratio, ki == -1 ? nullptr : &ki,
            imax == -1 ? nullptr : &imax, slewmax == -1 ? nullptr : &slewmax,
            inheritrange, ctrlspec);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("kp"), py::arg("kv") = -1, py::arg("dampratio") = -1,
      py::arg("ki") = -1, py::arg("imax") = -1, py::arg("slewmax") = -1,
      py::arg("inheritrange") = 0, py::arg("ctrlspec") = 0);
  mjsActuator.def(
      "set_to_damper",
      [](raw::MjsActuator* self, double kv) {
        std::string err = mjs_setToDamper(self, kv);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("kv"));
  mjsActuator.def(
      "set_to_cylinder",
      [](raw::MjsActuator* self, double timeconst, double bias, double area,
         double diameter) {
        std::string err =
            mjs_setToCylinder(self, timeconst, bias, area, diameter);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("timeconst"), py::arg("bias"), py::arg("area"),
      py::arg("diameter") = -1);
  mjsActuator.def(
      "set_to_muscle",
      [](raw::MjsActuator* self, std::array<double, 2> timeconst,
         double tausmooth, std::array<double, 2> range, double force,
         double scale, double lmin, double lmax, double vmax, double fpmax,
         double fvmax) {
        std::string err =
            mjs_setToMuscle(self, timeconst.data(), tausmooth, range.data(),
                            force, scale, lmin, lmax, vmax, fpmax, fvmax);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("timeconst") = std::array<double, 2>{-1, -1},
      py::arg("tausmooth"), py::arg("range") = std::array<double, 2>{-1, -1},
      py::arg("force") = -1, py::arg("scale") = -1, py::arg("lmin") = -1,
      py::arg("lmax") = -1, py::arg("vmax") = -1, py::arg("fpmax") = -1,
      py::arg("fvmax") = -1);
  mjsActuator.def(
      "set_to_adhesion",
      [](raw::MjsActuator* self, double gain) {
        std::string err = mjs_setToAdhesion(self, gain);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("gain"));
  mjsActuator.def(
      "set_to_dcmotor",
      [](raw::MjsActuator* self, std::array<double, 2> motorconst,
         double resistance, std::array<double, 3> nominal,
         std::array<double, 3> saturation, std::array<double, 2> inductance,
         std::array<double, 3> cogging, std::array<double, 6> controller,
         std::array<double, 6> thermal, std::array<double, 5> lugre,
         int ctrlspec) {
        std::string err = mjs_setToDCMotor(
            self, motorconst.data(), resistance, nominal.data(),
            saturation.data(), inductance.data(), cogging.data(),
            controller.data(), thermal.data(), lugre.data(), ctrlspec);
        if (!err.empty()) {
          throw pybind11::value_error(err);
        }
      },
      py::arg("motorconst"), py::arg("resistance"),
      py::arg("nominal") = std::array<double, 3>{0, 0, 0},
      py::arg("saturation") = std::array<double, 3>{0, 0, 0},
      py::arg("inductance") = std::array<double, 2>{0, 0},
      py::arg("cogging") = std::array<double, 3>{0, 0, 0},
      py::arg("controller") = std::array<double, 6>{0, 0, 0, 0, 0, 0},
      py::arg("thermal") = std::array<double, 6>{0, 0, 0, 0, 0, 0},
      py::arg("lugre") = std::array<double, 5>{0, 0, 0, 0, 0},
      py::arg("ctrlspec") = 0);

  // ============================= MJSTENDONPATH ===============================
  // helper struct for tendon path indexing
  struct MjsTendonPath {
    raw::MjsTendon* tendon;
  };

  py::class_<MjsTendonPath>(m, "MjsTendonPath")
      .def(
          "__getitem__",
          [](MjsTendonPath& self, int i) -> const raw::MjsWrap* {
            int num_wrap = mjs_getWrapNum(self.tendon);
            if (i < 0 || i >= num_wrap) {
              throw py::index_error("Index out of range.");
            }
            return mjs_getWrap(self.tendon, i);
          },
          py::return_value_policy::reference_internal)
      .def("__len__",
           [](MjsTendonPath& self) { return mjs_getWrapNum(self.tendon); });

  // ============================= MJSWRAP =====================================
  mjsWrap.def_property_readonly(
      "target",
      [](raw::MjsWrap& self) -> py::object {
        raw::MjsElement* target = mjs_getWrapTarget(&self);
        if (!target) {
          return py::none();
        }
        switch (target->elemtype) {
          case mjOBJ_SITE:
            return py::cast(mjs_asSite(target));
          case mjOBJ_GEOM:
            return py::cast(mjs_asGeom(target));
          case mjOBJ_JOINT:
            return py::cast(mjs_asJoint(target));
          default:
            throw pybind11::value_error("Unsupported wrap target type: " +
                                        std::to_string(target->elemtype));
        }
        return py::none();
      },
      py::return_value_policy::reference_internal);

  mjsWrap.def_property_readonly(
      "sidesite",
      [](raw::MjsWrap& self) -> raw::MjsSite* {
        return mjs_getWrapSideSite(&self);
      },
      py::return_value_policy::reference_internal);

  mjsWrap.def_property_readonly(
      "divisor",
      [](raw::MjsWrap& self) -> py::object {
        if (self.type != mjWRAP_PULLEY) {
          return py::none();
        }
        return py::cast(mjs_getWrapDivisor(&self));
      },
      py::return_value_policy::reference_internal);

  mjsWrap.def_property_readonly(
      "coef",
      [](raw::MjsWrap& self) -> py::object {
        if (self.type != mjWRAP_JOINT) {
          return py::none();
        }
        return py::cast(mjs_getWrapCoef(&self));
      },
      py::return_value_policy::reference_internal);

  mjSpec.def("delete", [](MjSpec& self, raw::MjsTendon& obj) {
    mjs_delete(self.ptr, obj.element);
  });

  // ============================= MJSTENDON ===================================
  mjsTendon.def(
      "default",
      [](raw::MjsTendon& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      py::return_value_policy::reference_internal);
  mjsTendon.def_property_readonly(
      "path", [](raw::MjsTendon& self) { return MjsTendonPath{&self}; },
      py::return_value_policy::reference_internal);
  mjsTendon.def(
      "wrap_site",
      [](raw::MjsTendon& self, std::string& name) {
        return mjs_wrapSite(&self, name.c_str());
      },
      py::return_value_policy::reference_internal);
  mjsTendon.def(
      "wrap_geom",
      [](raw::MjsTendon& self, std::string& name, std::string& sidesite) {
        return mjs_wrapGeom(&self, name.c_str(), sidesite.c_str());
      },
      py::return_value_policy::reference_internal);
  mjsTendon.def(
      "wrap_joint",
      [](raw::MjsTendon& self, std::string& name, double coef) {
        return mjs_wrapJoint(&self, name.c_str(), coef);
      },
      py::return_value_policy::reference_internal);
  mjsTendon.def(
      "wrap_pulley",
      [](raw::MjsTendon& self, double divisor) {
        return mjs_wrapPulley(&self, divisor);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSSENSOR ===================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsSensor& obj) {
    mjs_delete(self.ptr, obj.element);
  });
  mjsSensor.def("get_data_size", [](raw::MjsSensor& self) -> int {
    return mjs_sensorDim(&self);
  });

  // ============================= MJSFLEX =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsFlex& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSHFIELD ===================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsHField& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSSKIN =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsSkin& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSTEXTURE ==================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsTexture& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });
  mjsTexture.def_property(
      "classname",
      [](raw::MjsTexture& self) -> raw::MjsDefault* {
        return mjs_getDefault(self.element);
      },
      [](raw::MjsTexture& self, raw::MjsDefault& default_) -> void {
        mjs_setDefault(self.element, &default_);
      },
      py::return_value_policy::reference_internal);

  // ============================= MJSKEY ======================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsKey& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSTEXT =====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsText& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSNUMERIC ==================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsNumeric& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSEXCLUDE ==================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsExclude& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSTUPLE ====================================
  mjSpec.def("delete", [](MjSpec& self, raw::MjsTuple& obj) {
    if (mjs_delete(self.ptr, obj.element) != 0) {
      throw pybind11::value_error(mjs_getError(self.ptr));
    }
  });

  // ============================= MJSPLUGIN ===================================
  mjsPlugin.def_property(
      "id", [](raw::MjsPlugin& self) -> int { return mjs_getId(self.element); },
      [](raw::MjsPlugin& self, raw::MjsPlugin* other) {
        self.element = other->element;
      });
  mjSpec.def("delete", [](MjSpec& self, raw::MjsPlugin& obj) {
    mjs_delete(self.ptr, obj.element);
  });
  mjsPlugin.def_property(
      "config",
      [](raw::MjsPlugin& self) -> py::dict {
        const std::map<std::string, std::string, std::less<>>* config_attribs =
            static_cast<const std::map<std::string, std::string, std::less<>>*>(
                mjs_getPluginAttributes(&self));
        py::dict config;
        for (const auto& [key, value] : *config_attribs) {
          config[py::str(key)] = value;
        }
        return config;
      },
      [](raw::MjsPlugin& self, py::dict& config) {
        std::map<std::string, std::string, std::less<>> config_attribs;
        for (const auto& [key, value] : config) {
          std::string key_str = key.cast<std::string>();
          if (config_attribs.find(key_str) != config_attribs.end()) {
            throw pybind11::value_error("Duplicate config key: " + key_str);
          }
          config_attribs[key_str] = value.cast<std::string>();
        }
        mjs_setPluginAttributes(&self, &config_attribs);
      },
      py::return_value_policy::reference_internal);
  // ============================= MJVISUAL ====================================
  mjVisual.def_property(
      "global_",
      [](raw::MjVisual& self) -> raw::MjVisualGlobal& { return self.global; },
      [](raw::MjVisual& self, raw::MjVisualGlobal& value) {
        self.global = value;
      });

#include "specs.cc.inc"
}  // PYBIND11_MODULE // NOLINT
}  // namespace mujoco::python
