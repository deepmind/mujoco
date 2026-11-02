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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "gil.h"
#include "indexers.h"
#include "raw.h"
#include "util/crossplatform.h"

namespace mujoco::python {

namespace py = ::pybind11;

namespace {
// Parses raw mjModel to create a mapping from names to array indices.
//
// Args:
//   count: Number of names in the map.
//   name_offsets: Array consisting of indices that correspond to the start of
//     each name in the `names` array.
//   names: Character arrays consisting of the concatenation of all names.
template <typename IntPtr, typename CharPtr>
NameToID MakeNameToID(int count, IntPtr name_offsets, CharPtr names)
{
  NameToID name_to_id;
  for (int index = 0; index < count; ++index) {
    const char* name = &names[name_offsets[index]];
    if (name[0] != '\0') {
      name_to_id.insert({name, index});
    }
  }
  return name_to_id;
}

// Parses raw mjModel to create a mapping from array indices to names.
//
// Args:
//   count: Number of names in the map.
//   name_offsets: Array consisting of indices that correspond to the start of
//     each name in the `names` array.
//   names: Character arrays consisting of the concatenation of all names.
template <typename IntPtr, typename CharPtr>
IDToName MakeIDToName(int count, IntPtr name_offsets, CharPtr names) {
  IDToName id_to_name;
  for (int index = 0; index < count; ++index) {
    const char* name = &names[name_offsets[index]];
    id_to_name.emplace_back(name);
  }
  return id_to_name;
};

// Makes an array view into an mjModel/mjData struct field at a given index.
//
// Template args:
//   MjSize: An (int M::*) specifying the number of entities of the same type
//     as the one whose array view is to be created. This is a way to indicate
//     the MuJoCo category of the entity itself, e.g. nbody indicates that the
//     field belongs to a body.
//   T: Scalar data type of the field.
//
// Args:
//   base_ptr: Pointer to the first entry in the entire field.
//   index: Entity ID at which a view is to be created.
//   shape: The shape of the array as defined by MuJoCo. If MjSize corresponds
//     to a ragged or indirect field, the shape may be prepended by additional
//     dimensions as appropriate, e.g. if MjSize is nv then the view gains an
//     additional dimension of the size len(qvel) of the particular joint. For
//     actuator fields (nu, nout), that dimension is added only if the actuator
//     does not have exactly one control or output, so that the views of
//     single-input, single-output actuators keep the shape of a single row.
//   m: Used for dereferencing MjSize.
//   owner: The base object whose lifetime is tied to the returned array.
template <auto MjSize, typename T>
py::array_t<T> MakeArray(T* base_ptr, int index, std::vector<int>&& shape,
                         const raw::MjModel& m, py::handle owner) {
  // number of elements in one row of a ragged field, whose address counts rows
  mjtSize rowsize = 1;
  for (int s : shape) {
    rowsize *= s;
  }
  mjtSize offset;
  if (MjSize == &raw::MjModel::nq) {
    offset = m.jnt_qposadr[index] * rowsize;
    shape.insert(
        shape.begin(),
        ((index < m.njnt-1) ? m.jnt_qposadr[index+1] : m.nq) -
            m.jnt_qposadr[index]);
  } else if (MjSize == &raw::MjModel::nv) {
    offset = m.jnt_dofadr[index] * rowsize;
    shape.insert(
        shape.begin(),
        ((index < m.njnt-1) ? m.jnt_dofadr[index+1] : m.nv) -
            m.jnt_dofadr[index]);
  } else if (MjSize == &raw::MjModel::nu || MjSize == &raw::MjModel::nout) {
    bool ctrl = MjSize == &raw::MjModel::nu;
    offset = (ctrl ? m.actuator_ctrladr : m.actuator_outadr)[index] * rowsize;
    int num = (ctrl ? m.actuator_ctrlnum : m.actuator_outnum)[index];
    if (num != 1 || shape.empty()) {
      shape.insert(shape.begin(), num);
    }
  } else if (MjSize == &raw::MjModel::nhfielddata) {
    offset = m.hfield_adr[index];
    shape.insert(shape.begin(), m.hfield_ncol[index]);
    shape.insert(shape.begin(), m.hfield_nrow[index]);
  } else if (MjSize == &raw::MjModel::ntexdata) {
    // a by-reference texture (tex_adr == -1) has no pixels: an empty view
    offset = m.tex_adr[index] < 0 ? 0 : m.tex_adr[index];
    shape.insert(shape.begin(), m.tex_nchannel[index]);
    shape.insert(shape.begin(), m.tex_width[index]);
    shape.insert(shape.begin(), m.tex_height[index]);
  } else if (MjSize == &raw::MjModel::nsensordata) {
    offset = m.sensor_adr[index];
    shape.insert(shape.begin(), m.sensor_dim[index]);
  } else if (MjSize == &raw::MjModel::nnumericdata) {
    offset = m.numeric_adr[index];
    shape.insert(shape.begin(), m.numeric_size[index]);
  } else if (MjSize == &raw::MjModel::ntupledata) {
    offset = m.tuple_adr[index];
    shape.insert(shape.begin(), m.tuple_size[index]);
  } else if (MjSize == &raw::MjModel::nplastic) {
    if (shape.empty()) {
      int adr = m.jnt_plasticadr[index];
      offset = adr < 0 ? 0 : adr;
      int num = adr < 0 ? 0
                        : (((index < m.njnt - 1) ? m.jnt_qposadr[index + 1]
                                                 : m.nq) -
                           m.jnt_qposadr[index]);
      shape.push_back(num);
    } else {
      int adr = m.tendon_plasticadr[index];
      offset = adr < 0 ? 0 : adr;
      if (adr < 0) {
        shape[0] = 0;
      }
    }
  } else if (MjSize == &raw::MjModel::nJten) {
    int adr = m.ten_J_rowadr[index];
    offset = adr < 0 ? 0 : adr;
    int num = adr < 0 ? 0 : m.ten_J_rownnz[index];
    shape.push_back(num);
  } else {
    // Do not return a NumPy array with shape () since these aren't very nice
    // to work with. Instead, always return singleton arrays with shape (1,).
    if (shape.empty()) {
      shape.push_back(1);
    }
    int size = 1;
    for (int s : shape) {
      size *= s;
    }
    offset = index * size;
  }
  return py::array_t<T>(std::move(shape), base_ptr + offset,
                        py::reinterpret_borrow<py::object>(owner));
}
}  // namespace

NameToIDMappings::NameToIDMappings(const raw::MjModel& m)
    : body(MakeNameToID(m.nbody, m.name_bodyadr, m.names)),
      jnt(MakeNameToID(m.njnt, m.name_jntadr, m.names)),
      geom(MakeNameToID(m.ngeom, m.name_geomadr, m.names)),
      site(MakeNameToID(m.nsite, m.name_siteadr, m.names)),
      cam(MakeNameToID(m.ncam, m.name_camadr, m.names)),
      light(MakeNameToID(m.nlight, m.name_lightadr, m.names)),
      mesh(MakeNameToID(m.nmesh, m.name_meshadr, m.names)),
      skin(MakeNameToID(m.nskin, m.name_skinadr, m.names)),
      hfield(MakeNameToID(m.nhfield, m.name_hfieldadr, m.names)),
      tex(MakeNameToID(m.ntex, m.name_texadr, m.names)),
      mat(MakeNameToID(m.nmat, m.name_matadr, m.names)),
      pair(MakeNameToID(m.npair, m.name_pairadr, m.names)),
      exclude(MakeNameToID(m.nexclude, m.name_excludeadr, m.names)),
      eq(MakeNameToID(m.neq, m.name_eqadr, m.names)),
      tendon(MakeNameToID(m.ntendon, m.name_tendonadr, m.names)),
      actuator(MakeNameToID(m.nactuator, m.name_actuatoradr, m.names)),
      sensor(MakeNameToID(m.nsensor, m.name_sensoradr, m.names)),
      numeric(MakeNameToID(m.nnumeric, m.name_numericadr, m.names)),
      text(MakeNameToID(m.ntext, m.name_textadr, m.names)),
      tuple(MakeNameToID(m.ntuple, m.name_tupleadr, m.names)),
      key(MakeNameToID(m.nkey, m.name_keyadr, m.names)) {}

IDToNameMappings::IDToNameMappings(const raw::MjModel& m)
    : body(MakeIDToName(m.nbody, m.name_bodyadr, m.names)),
      jnt(MakeIDToName(m.njnt, m.name_jntadr, m.names)),
      geom(MakeIDToName(m.ngeom, m.name_geomadr, m.names)),
      site(MakeIDToName(m.nsite, m.name_siteadr, m.names)),
      cam(MakeIDToName(m.ncam, m.name_camadr, m.names)),
      light(MakeIDToName(m.nlight, m.name_lightadr, m.names)),
      mesh(MakeIDToName(m.nmesh, m.name_meshadr, m.names)),
      skin(MakeIDToName(m.nskin, m.name_skinadr, m.names)),
      hfield(MakeIDToName(m.nhfield, m.name_hfieldadr, m.names)),
      tex(MakeIDToName(m.ntex, m.name_texadr, m.names)),
      mat(MakeIDToName(m.nmat, m.name_matadr, m.names)),
      pair(MakeIDToName(m.npair, m.name_pairadr, m.names)),
      exclude(MakeIDToName(m.nexclude, m.name_excludeadr, m.names)),
      eq(MakeIDToName(m.neq, m.name_eqadr, m.names)),
      tendon(MakeIDToName(m.ntendon, m.name_tendonadr, m.names)),
      actuator(MakeIDToName(m.nactuator, m.name_actuatoradr, m.names)),
      sensor(MakeIDToName(m.nsensor, m.name_sensoradr, m.names)),
      numeric(MakeIDToName(m.nnumeric, m.name_numericadr, m.names)),
      text(MakeIDToName(m.ntext, m.name_textadr, m.names)),
      tuple(MakeIDToName(m.ntuple, m.name_tupleadr, m.names)),
      key(MakeIDToName(m.nkey, m.name_keyadr, m.names)) {}

MjModelIndexer::MjModelIndexer(raw::MjModel* m, py::handle owner)
    : m_(m),
      owner_(owner),
      name_to_id_(*m),
      id_to_name_(*m)
#define XGROUP(MjModelFieldGroupedViews, field, nfield, FIELD_XMACROS) \
  , field##_(m->nfield)
      MJMODEL_VIEW_GROUPS
#undef XGROUP
{}

#define XGROUP(MjModelFieldGroupedViews, field, nfield, FIELD_XMACROS) \
  MjModelFieldGroupedViews& MjModelIndexer::field(int i) {             \
    if (i >= field##_.size() || i < 0) {                               \
      throw py::index_error(IndexErrorMessage(i, field##_.size()));    \
    }                                                                  \
    MutexLockIfGilDisabled lock(lazy_init_mutex_);                     \
    auto& indexer = field##_[i];                                       \
    if (!indexer.has_value()) {                                        \
      const std::string& name = id_to_name_.field[i];                  \
      indexer.emplace(i, name, m_, owner_);                            \
    }                                                                  \
    return *indexer;                                                   \
  }
MJMODEL_VIEW_GROUPS
#undef XGROUP

#define XGROUP(MjModelFieldGroupedViews, field, nfield, FIELD_XMACROS) \
  MjModelFieldGroupedViews& MjModelIndexer::field##_by_name(           \
      std::string_view name) {                                         \
    auto item = name_to_id_.field.find(name);                          \
    if (item == name_to_id_.field.end()) {                             \
      throw py::key_error(KeyErrorMessage(name_to_id_.field, name));   \
    }                                                                  \
    return field(item->second);                                        \
  }
MJMODEL_VIEW_GROUPS
#undef XGROUP

MjDataIndexer::MjDataIndexer(raw::MjData* d, const raw::MjModel* m,
                             py::handle owner)
    : d_(d),
      m_(m),
      owner_(owner),
      name_to_id_(*m),
      id_to_name_(*m)
#define XGROUP(MjDataGroupedViews, field, nfield, FIELD_XMACROS) \
  , field##_(m->nfield)
      MJDATA_VIEW_GROUPS
#undef XGROUP
{}

#define XGROUP(MjDataGroupedViews, field, nfield, FIELD_XMACROS)    \
  MjDataGroupedViews& MjDataIndexer::field(int i) {                 \
    if (i >= field##_.size() || i < 0) {                            \
      throw py::index_error(IndexErrorMessage(i, field##_.size())); \
    }                                                               \
    MutexLockIfGilDisabled lock(lazy_init_mutex_);                  \
    auto& indexer = field##_[i];                                    \
    if (!indexer.has_value()) {                                     \
      const std::string& name = id_to_name_.field[i];               \
      indexer.emplace(i, name, d_, m_, owner_);                     \
    }                                                               \
    return *indexer;                                                \
  }
MJDATA_VIEW_GROUPS
#undef XGROUP

#define XGROUP(MjDataGroupedViews, field, nfield, FIELD_XMACROS)              \
  MjDataGroupedViews& MjDataIndexer::field##_by_name(std::string_view name) { \
    auto item = name_to_id_.field.find(name);                                 \
    if (item == name_to_id_.field.end()) {                                    \
      throw py::key_error(KeyErrorMessage(name_to_id_.field, name));          \
    }                                                                         \
    return field(item->second);                                               \
  }
MJDATA_VIEW_GROUPS
#undef XGROUP

#define MAKE_SHAPE(dim)                                              \
  MUJOCO_DIAG_IGNORE_UNUSED_LAMBDA_CAPTURE                           \
    [n = (dim)]() -> std::vector<int> {                              \
    if constexpr (std::string_view(#dim) == std::string_view("1")) { \
      return {};                                                     \
    } else {                                                         \
      return {(int)n};                                               \
    }                                                                \
  }()                                                                \
  MUJOCO_DIAG_UNIGNORE_UNUSED_LAMBDA_CAPTURE

#undef MJ_M
#define MJ_M(n) m_->n
#define X(type, prefix, var, dim0, dim1)                            \
  py::array_t<type> XGROUP::var() {                                 \
    MutexLockIfGilDisabled lock(lazy_init_mutex_);                  \
    if (!var##_.has_value()) {                                      \
      var##_.emplace(MakeArray<&raw::MjModel::dim0>(                \
          m_->prefix##var, index_, MAKE_SHAPE(dim1), *m_, owner_)); \
    }                                                               \
    return *var##_;                                                 \
  }

#define XGROUP MjModelActuatorViews
MJMODEL_ACTUATOR
#undef XGROUP

#define XGROUP MjModelBodyViews
MJMODEL_BODY
#undef XGROUP

#define XGROUP MjModelCameraViews
MJMODEL_CAMERA
#undef XGROUP

#define XGROUP MjModelEqualityViews
MJMODEL_EQUALITY
#undef XGROUP

#define XGROUP MjModelExcludeViews
MJMODEL_EXCLUDE
#undef XGROUP

#define XGROUP MjModelGeomViews
MJMODEL_GEOM
#undef XGROUP

#define XGROUP MjModelHfieldViews
MJMODEL_HFIELD
#undef XGROUP

#define XGROUP MjModelJointViews
MJMODEL_JOINT
#undef XGROUP

#define XGROUP MjModelLightViews
MJMODEL_LIGHT
#undef XGROUP

#define XGROUP MjModelMaterialViews
MJMODEL_MATERIAL
#undef XGROUP

#define XGROUP MjModelMeshViews
MJMODEL_MESH
#undef XGROUP

#define XGROUP MjModelNumericViews
MJMODEL_NUMERIC
#undef XGROUP

#define XGROUP MjModelPairViews
MJMODEL_PAIR
#undef XGROUP

#define XGROUP MjModelSensorViews
MJMODEL_SENSOR
#undef XGROUP

#define XGROUP MjModelSiteViews
MJMODEL_SITE
#undef XGROUP

#define XGROUP MjModelSkinViews
MJMODEL_SKIN
#undef XGROUP

#define XGROUP MjModelTendonViews
MJMODEL_TENDON
#undef XGROUP

#define XGROUP MjModelTextureViews
MJMODEL_TEXTURE
#undef XGROUP

#define XGROUP MjModelTupleViews
MJMODEL_TUPLE
#undef XGROUP

#define XGROUP MjModelKeyframeViews
MJMODEL_KEYFRAME
#undef XGROUP

#undef X
#define MJ_M(n) m_->n
#define X(type, prefix, var, dim0, dim1)                            \
  py::array_t<type> XGROUP::var() {                               \
    MutexLockIfGilDisabled lock(lazy_init_mutex_);                  \
    if (!var##_.has_value()) {                                      \
      var##_.emplace(MakeArray<&raw::MjModel::dim0>(              \
          d_->prefix##var, index_, MAKE_SHAPE(dim1), *m_, owner_)); \
    }                                                               \
    return *var##_;                                                 \
  }

#define XGROUP MjDataActuatorViews
MJDATA_ACTUATOR
#undef XGROUP

#define XGROUP MjDataBodyViews
MJDATA_BODY
#undef XGROUP

#define XGROUP MjDataCameraViews
MJDATA_CAMERA
#undef XGROUP

#define XGROUP MjDataGeomViews
MJDATA_GEOM
#undef XGROUP

#define XGROUP MjDataJointViews
MJDATA_JOINT
#undef XGROUP

#define XGROUP MjDataLightViews
MJDATA_LIGHT
#undef XGROUP

#define XGROUP MjDataSensorViews
MJDATA_SENSOR
#undef XGROUP

#define XGROUP MjDataSiteViews
MJDATA_SITE
#undef XGROUP

#define XGROUP MjDataTendonViews
MJDATA_TENDON
#undef XGROUP

#undef X
#undef MJ_M
#define MJ_M(n) n

// Returns an error message when a non-existent name is requested from an
// indexer, which includes all valid names.
std::string KeyErrorMessage(const NameToID& map, std::string_view name) {
  // Make a sorted list of valid names
  std::vector<std::string_view> valid_names;
  valid_names.reserve(map.size());
  for (const auto& [key, value] : map) {
    valid_names.push_back(key);
  }
  std::sort(valid_names.begin(), valid_names.end());

  // Construct the error message
  std::ostringstream message;
  message << "Invalid name '" << name << "'. Valid names: [";
  int i = 0;
  for (const auto& key : valid_names) {
    message << "'" << key << "'";
    if (i < map.size() - 1) message << ", ";
    i++;
  }
  message << "]";
  return message.str();
}

std::string IndexErrorMessage(int index, int size) {
  std::ostringstream message;
  message << "Invalid index " << index << ". Valid indices from 0 to "
          << size - 1;
  return message.str();
}
}  // namespace mujoco::python
