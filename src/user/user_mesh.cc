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
#include <chrono>
#include <climits>
#include <cmath>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mjspec.h>
#include <mujoco/mjspecmacro.h>
#include "user/user_api.h"

#ifdef MUJOCO_TINYOBJLOADER_IMPL
  #define TINYOBJLOADER_IMPLEMENTATION
#endif

#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
  #pragma clang diagnostic ignored "-Wnested-anon-types"
#elif defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "MC.h"
#if defined(__clang__)
  #pragma clang diagnostic pop
#elif defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include "engine/engine_crossplatform.h"  // IWYU pragma: keep
#include "engine/engine_plugin.h"
#include "engine/engine_util_errmem.h"
#include "user/user_cache.h"
#include "user/user_model.h"
#include "user/user_objects.h"
#include "user/user_resource.h"
#include "user/user_util.h"

extern "C" {
#include "libqhull_r/qhull_ra.h"
}

namespace {
using mujoco::user::FilePath;
using mujoco::user::PairHash;
using std::cos;
using std::max;
using std::pow;
using std::sin;

// Parametrized linear/quintic interpolated nonlinearity.
double Fovea(double x, double gamma) {
  // Quick return.
  if (!gamma) return x;

  // Foveal deformation.
  double g = mjMAX(0, mjMIN(1, gamma));
  return g * pow(x, 5) + (1 - g) * x;
}

// Evenly spaced numbers over a specified interval.
void LinSpace(double lower, double upper, int n, double array[]) {
  double increment = n > 1 ? (upper - lower) / (n - 1) : 0;
  for (int i = 0; i < n; ++i) {
    *array = lower;
    ++array;
    lower += increment;
  }
}

// Make bin edges.
void BinEdges(double* x_edges, double* y_edges, int size[2], double fov[2], double gamma) {
  // Make unit bin edges.
  LinSpace(-1, 1, size[0] + 1, x_edges);
  LinSpace(-1, 1, size[1] + 1, y_edges);

  // Apply foveal deformation.
  for (int i = 0; i < size[0] + 1; i++) { x_edges[i] = Fovea(x_edges[i], gamma); }
  for (int i = 0; i < size[1] + 1; i++) { y_edges[i] = Fovea(y_edges[i], gamma); }

  // Scale by field-of-view.
  mjuu_scalevec(x_edges, x_edges, fov[0] * mjPI / 180, size[0] + 1);
  mjuu_scalevec(y_edges, y_edges, fov[1] * mjPI / 180, size[1] + 1);
}

// Transform spherical (azimuth, elevation, radius) to Cartesian (x,y,z).
void SphericalToCartesian(const double aer[3], float xyz[3]) {
  double a = aer[0], e = aer[1], r = aer[2];
  xyz[0] = r * cos(e) * sin(a);
  xyz[1] = r * sin(e);
  xyz[2] = -r * cos(e) * cos(a);
}

// Tangent frame in Cartesian coordinates.
void TangentFrame(const double aer[3], float mat[9]) {
  double a = aer[0], e = aer[1], r = aer[2];
  double ta[3] = {r * cos(e) * cos(a), 0, r * cos(e) * sin(a)};
  double te[3] = {-r * sin(e) * sin(a), r * cos(e), r * sin(e) * cos(a)};
  double n[3];
  mjuu_normvec(ta, 3);
  mjuu_normvec(te, 3);
  mjuu_copyvec(mat + 3, ta, 3);
  mjuu_copyvec(mat + 6, te, 3);
  mjuu_crossvec(n, te, ta);
  mjuu_copyvec(mat, n, 3);
}

// parametric superellipsoid/supertoroid helper functions
double aux_c(double omega, double m) {
  return std::copysign(pow(std::abs(cos(omega)), m), cos(omega));
}
double aux_s(double omega, double m) {
  return std::copysign(pow(std::abs(sin(omega)), m), sin(omega));
}
}  // namespace

// compute triangle area, surface normal, center
static double triangle(
    double* normal, double* center, const double* v1, const double* v2, const double* v3) {
  double  normal_local[3];  // if normal is nullptr
  double* normal_ptr = (normal) ? normal : normal_local;
  // center
  if (center) {
    center[0] = (v1[0] + v2[0] + v3[0]) / 3;
    center[1] = (v1[1] + v2[1] + v3[1]) / 3;
    center[2] = (v1[2] + v2[2] + v3[2]) / 3;
  }

  // normal = (v2-v1) cross (v3-v1)
  double b[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  double c[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
  mjuu_crossvec(normal_ptr, b, c);

  // get length
  double len = sqrt(mjuu_dot3(normal_ptr, normal_ptr));

  // ignore small faces
  if (len < mjMINVAL) { return 0; }

  // normalize
  if (normal) {
    normal_ptr[0] /= len;
    normal_ptr[1] /= len;
    normal_ptr[2] /= len;
  }

  // return area
  return 0.5 * len;
}


// Read data of type T from a potentially unaligned buffer pointer.
template <typename T>
static void ReadFromBuffer(T* dst, const char* src) {
  std::memcpy(dst, src, sizeof(T));
}


//------------------ class mjCMesh implementation --------------------------------------------------

mjCMesh::mjCMesh(mjCModel* _model, mjCDef* _def) {
  mjs_defaultMesh(&spec);
  elemtype = mjOBJ_MESH;

  // clear internal variables
  mjuu_setvec(pos_, 0, 0, 0);
  mjuu_setvec(quat_, 1, 0, 0, 0);

  mjuu_setvec(boxsz_, 0, 0, 0);
  mjuu_setvec(aamm_, 1e10, 1e10, 1e10);
  mjuu_setvec(aamm_ + 3, -1e10, -1e10, -1e10);
  szgraph_      = 0;
  center_       = nullptr;
  graph_        = nullptr;
  needhull_     = false;
  needsdf_      = false;
  maxhullvert_  = -1;
  processed_    = false;
  needreorient_ = true;

  // reset to default if given
  if (_def) { *this = _def->Mesh(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = (_def ? _def->name : (_model ? "main" : ""));

  // in case this body is not compiled
  CopyFromSpec();

  // point to local
  PointToLocal();
}


mjCMesh::mjCMesh(const mjCMesh& other) {
  *this = other;
}


mjCMesh& mjCMesh::operator=(const mjCMesh& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCMesh_*>(this) = static_cast<const mjCMesh_&>(other);
    *static_cast<mjsMesh*>(this)  = static_cast<const mjsMesh&>(other);
    if (other.center_) {
      size_t ncenter = 3 * other.nface() * sizeof(double);
      this->center_  = (double*)mju_malloc(ncenter);
      memcpy(this->center_, other.center_, ncenter);
    } else {
      this->center_ = nullptr;
    }
    if (other.graph_) {
      size_t szgraph = szgraph_ * sizeof(int);
      this->graph_   = (int*)mju_malloc(szgraph);
      memcpy(this->graph_, other.graph_, szgraph);
    } else {
      this->graph_ = nullptr;
    }
  }
  PointToLocal();
  return *this;
}


void mjCMesh::PointToLocal() {
  spec.element            = static_cast<mjsElement*>(this);
  spec.file               = &spec_file_;
  spec.content_type       = &spec_content_type_;
  spec.uservert           = &spec_vert_;
  spec.usernormal         = &spec_normal_;
  spec.userface           = &spec_face_;
  spec.userfacenormal     = &spec_facenormal_;
  spec.usertexcoord       = &spec_texcoord_;
  spec.userfacetexcoord   = &spec_facetexcoord_;
  spec.usernode           = &spec_node_;
  spec.usertet            = &spec_tet_;
  spec.material           = &spec_material_;
  spec.plugin.plugin_name = &plugin_name;
  spec.plugin.name        = &plugin_instance_name;
  spec.info               = &info;
  file                    = nullptr;
  content_type            = nullptr;
  uservert                = nullptr;
  usernormal              = nullptr;
  userface                = nullptr;
  userfacenormal          = nullptr;
  usertexcoord            = nullptr;
  userfacetexcoord        = nullptr;
  usernode                = nullptr;
  usertet                 = nullptr;
}


void mjCMesh::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!spec_material_.empty() && model != m) {
    spec_material_ = m->prefix + spec_material_ + m->suffix;
  }
  if (!plugin_instance_name.empty()) {
    plugin_instance_name = m->prefix + plugin_instance_name + m->suffix;
  }
}


// the key under which meshes which would compile to the same asset are deduplicated: every field
// of the spec which compilation reads, names excluded, with the file resolved as the compiler
// resolves it, and the sizes of the arrays, whose contents are compared separately; a mesh with a
// plugin is not deduplicated
std::string mjCMesh::AssetKey() const {
#define X(type, name, dim) +1
#define XVEC(type, name, dim) +1
  static_assert((0 MJSMESH_FIELDS) == 22,
                "a field was added to mjsMesh: decide whether AssetKey reads it");
#undef X
#undef XVEC
  const mjsMesh& s = spec;
  std::string    key;
  mjuu_addkey(key, ResolvedSpecFile(mjs_getString(compiler->meshdir), spec_file_));
  mjuu_addkey(key, spec_content_type_);
  mjuu_addkey(key, spec_material_);
  mjuu_addkey(key, s.refpos);
  mjuu_addkey(key, s.refquat);
  mjuu_addkey(key, s.scale);
  mjuu_addkey(key, s.inertia);
  mjuu_addkey(key, s.smoothnormal);
  mjuu_addkey(key, s.needsdf);
  mjuu_addkey(key, s.maxhullvert);
  mjuu_addkey(key, s.octree_maxdepth);
  for (size_t size : {spec_vert_.size(),
                      spec_normal_.size(),
                      spec_texcoord_.size(),
                      spec_face_.size(),
                      spec_facenormal_.size(),
                      spec_facetexcoord_.size(),
                      spec_node_.size(),
                      spec_tet_.size()}) {
    mjuu_addkey(key, size);
  }
  return key;
}


void mjCMesh::CopyFromSpec() {
  *static_cast<mjsMesh*>(this) = spec;
  if (InstanceDiagonal()) {
    for (int i = 0; i < 3; i++) { scale[i] *= instancemap_[4 * i]; }
  }

  file_         = spec_file_;
  content_type_ = spec_content_type_;
  normal_       = spec_normal_;
  face_         = spec_face_;
  node_         = spec_node_;
  tet_          = spec_tet_;
  material_     = spec_material_;
  ProcessVertices(spec_vert_);
  texcoord_          = spec_texcoord_;
  facetexcoord_      = spec_facetexcoord_;
  maxhullvert_       = spec.maxhullvert;
  plugin.active      = spec.plugin.active;
  plugin.element     = spec.plugin.element;
  plugin.plugin_name = spec.plugin.plugin_name;
  plugin.name        = spec.plugin.name;

  // clear precompiled asset. TODO: use asset cache
  if (center_) mju_free(center_);
  if (graph_) mju_free(graph_);
  szgraph_ = 0;
  center_  = nullptr;
  graph_   = nullptr;
  halfedge_.clear();
  tree_ = mjCBoundingVolumeHierarchy();
  face_aabb_.clear();
  octree_.Clear();
  mjuu_setvec(aamm_, 1e10, 1e10, 1e10);
  mjuu_setvec(aamm_ + 3, -1e10, -1e10, -1e10);
}


void mjCMesh::CopyPlugin() {
  model->CopyExplicitPlugin(this);
}


mjCMesh::~mjCMesh() {
  if (center_) mju_free(center_);
  if (graph_) mju_free(graph_);
}


// file as opened by Compile, which has already applied strippath to file_
std::string mjCMesh::ResolvedFilePath() const {
  return ResolvePath(mjs_getString(compiler->meshdir), file_);
}


// generate mesh using marching cubes
void mjCMesh::LoadSDF() {
  if (plugin_name.empty() && plugin_instance_name.empty()) {
    throw mjCError(this,
                   "neither 'plugin' nor 'instance' is specified for mesh '%s', (id = %d)",
                   name.c_str(),
                   id);
  }

  if (instancemap_ != decltype(instancemap_){1, 0, 0, 0, 1, 0, 0, 0, 1}) {
    throw mjCError(this, "SDF mesh '%s' cannot be scaled by a frame", name.c_str());
  }
  if (scale[0] != 1 || scale[1] != 1 || scale[2] != 1) {
    throw mjCError(this,
                   "attribute scale is not compatible with SDFs in mesh '%s', (id = %d)",
                   name.c_str(),
                   id);
  }

  mjCPlugin* plugin_instance = static_cast<mjCPlugin*>(plugin.element);
  model->ResolvePlugin(this, plugin_name, plugin_instance_name, &plugin_instance);
  plugin.element           = plugin_instance;
  const mjpPlugin* pplugin = mjp_getPluginAtSlot(plugin_instance->plugin_slot);
  if (!(pplugin->capabilityflags & mjPLUGIN_SDF)) {
    throw mjCError(this, "plugin '%s' does not support signed distance fields", pplugin->name);
  }

  std::vector<mjtNum>      attributes(pplugin->nattribute, 0);
  std::vector<const char*> names(pplugin->nattribute, 0);
  std::vector<const char*> values(pplugin->nattribute, 0);
  for (int i = 0; i < pplugin->nattribute; i++) {
    // an attribute which is not configured is empty, and is not added to the configuration
    names[i]  = pplugin->attributes[i];
    auto it   = plugin_instance->config_attribs.find(names[i]);
    values[i] = it != plugin_instance->config_attribs.end() ? it->second.c_str() : "";
  }

  if (pplugin->sdf_attribute) {
    pplugin->sdf_attribute(attributes.data(), names.data(), values.data());
  }
  for (int i = 0; i < pplugin->nattribute; i++) {
    if (std::isnan(attributes[i])) {
      throw mjCError(this, "invalid attribute value for '%s'", names[i]);
    }
  }

  mjtNum aabb[6] = {0};
  pplugin->sdf_aabb(aabb, attributes.data());
  mjtNum total = aabb[3] + aabb[4] + aabb[5];

  const double n = 300;
  int          nx, ny, nz;
  nx = floor(n / total * aabb[3]) + 1;
  ny = floor(n / total * aabb[4]) + 1;
  nz = floor(n / total * aabb[5]) + 1;

  MC::MC_FLOAT* field = new MC::MC_FLOAT[nx * ny * nz];

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        mjtNum point[]               = {aabb[0] - aabb[3] + 2 * aabb[3] * i / (nx - 1),
                                        aabb[1] - aabb[4] + 2 * aabb[4] * j / (ny - 1),
                                        aabb[2] - aabb[5] + 2 * aabb[5] * k / (nz - 1)};
        field[(k * ny + j) * nx + i] = pplugin->sdf_staticdistance(point, attributes.data());
      }
    }
  }

  MC::mcMesh mesh;
  MC::marching_cube(field, nx, ny, nz, mesh);
  std::vector<float> uservert;
  std::vector<float> usernormal;
  std::vector<int>   userface;

  uservert.reserve(mesh.vertices.size() * 3);
  usernormal.reserve(mesh.normals.size() * 3);
  userface.reserve(mesh.indices.size());

  for (const auto& vertex : mesh.vertices) {
    uservert.push_back(2 * aabb[3] * vertex.x / (nx - 1) + aabb[0] - aabb[3]);
    uservert.push_back(2 * aabb[4] * vertex.y / (ny - 1) + aabb[1] - aabb[4]);
    uservert.push_back(2 * aabb[5] * vertex.z / (nz - 1) + aabb[2] - aabb[5]);
  }

  for (const auto& normal : mesh.normals) {
    usernormal.push_back(normal.x);
    usernormal.push_back(normal.y);
    usernormal.push_back(normal.z);
  }

  for (unsigned int index : mesh.indices) { userface.push_back(index); }

  needreorient_ = false;
  needsdf       = false;
  needsdf_      = false;
  normal_       = std::move(usernormal);
  face_         = std::move(userface);
  ProcessVertices(uservert);
  delete[] field;
}


void mjCMesh::CacheMesh(mjCCache* cache, const mjResource* resource) {
  if (cache == nullptr) return;

  // cache mesh data into new mesh object
  mjCMesh* mesh = new mjCMesh();

  // mesh properties
  mesh->maxhullvert_    = maxhullvert_;
  mesh->inertia         = inertia;
  mesh->smoothnormal    = smoothnormal;
  mesh->octree_maxdepth = octree_maxdepth;
  std::copy(scale, scale + 3, mesh->scale);
  std::copy(refpos, refpos + 3, mesh->refpos);
  std::copy(refquat, refquat + 4, mesh->refquat);

  // mesh processed data
  mesh->processed_    = processed_;
  mesh->vert_         = vert_;
  mesh->normal_       = normal_;
  mesh->texcoord_     = texcoord_;
  mesh->face_         = face_;
  mesh->facenormal_   = facenormal_;
  mesh->facetexcoord_ = facetexcoord_;
  mesh->node_         = node_;
  mesh->tet_          = tet_;
  mesh->halfedge_     = halfedge_;
  mesh->szgraph_      = szgraph_;
  if (szgraph_) {
    mesh->graph_ = (int*)mju_malloc(szgraph_ * sizeof(int));
    std::copy(graph_, graph_ + szgraph_, mesh->graph_);
    mesh->polygons_        = polygons_;
    mesh->polygon_normals_ = polygon_normals_;
    mesh->polygon_map_     = polygon_map_;
  }
  mesh->surface_  = surface_;
  mesh->volume_   = volume_;
  mesh->material_ = material_;
  std::copy(boxsz_, boxsz_ + 3, mesh->boxsz_);
  std::copy(aamm_, aamm_ + 6, mesh->aamm_);
  std::copy(pos_, pos_ + 3, mesh->pos_);
  std::copy(quat_, quat_ + 4, mesh->quat_);
  int ncenter = face_.size();
  if (ncenter) {
    mesh->center_ = (double*)mju_malloc(ncenter * sizeof(double));
    std::copy(center_, center_ + ncenter, mesh->center_);
  }
  mesh->tree_      = tree_;
  mesh->face_aabb_ = face_aabb_;
  mesh->octree_    = octree_;

  // calculate estimated size of mesh
  std::size_t size = sizeof(mjCMesh) +
                     (sizeof(float) * vert_.size()) +
                     (sizeof(float) * normal_.size()) +
                     (sizeof(float) * texcoord_.size()) +
                     (sizeof(int) * face_.size()) +
                     (sizeof(int) * facenormal_.size()) +
                     (sizeof(int) * facetexcoord_.size()) +
                     (sizeof(int) * 2 * halfedge_.size()) +
                     (sizeof(int) * szgraph_) +
                     (sizeof(int) * npolygonvert()) +
                     (sizeof(double) * polygon_normals_.size()) +
                     (sizeof(int) * (szgraph_ ? npolygonmap() : 0)) +
                     (sizeof(double) * 18) +
                     (sizeof(int) * ncenter) +
                     tree_.Size() +
                     octree_.Size() +
                     (sizeof(double) * face_aabb_.size());

  std::shared_ptr<const void> cached_data(
      mesh,
      +[](const void* data) {
        const mjCMesh* mesh = static_cast<const mjCMesh*>(data);
        delete mesh;
      });
  cache->Insert("", CacheId(resource), resource, cached_data, size);
}


// id of the mesh in the asset cache: that of its file, and its factor for an instance at another
// scale
std::string mjCMesh::CacheId(const mjResource* resource) const {
  const auto& m = instancemap_;
  if (InstanceDiagonal() && m[0] == m[4] && m[4] == m[8]) {
    if (m[0] == 1) { return resource->name; }
    char factor[32];
    std::snprintf(factor, sizeof(factor), "@%a", m[0]);
    return std::string(resource->name) + factor;
  }
  std::string id = std::string(resource->name) + "@";
  for (double x : m) {
    char factor[32];
    std::snprintf(factor, sizeof(factor), "%a,", x);
    id += factor;
  }
  return id;
}


// true if the map of this instance is diagonal, which then multiplies the scale of the mesh
bool mjCMesh::InstanceDiagonal() const {
  const auto& m = instancemap_;
  return !m[1] && !m[2] && !m[3] && !m[5] && !m[6] && !m[7];
}

namespace {

// vertex key for hash map
struct VertexKey {
  float v[3];

  bool operator==(const VertexKey& other) const {
    return (v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2]);
  }

  std::size_t operator()(const VertexKey& vertex) const {
    // combine all three hash values into a single hash value
    return ((std::hash<float>()(vertex.v[0]) ^ (std::hash<float>()(vertex.v[1]) << 1)) >> 1) ^
           (std::hash<float>()(vertex.v[2]) << 1);
  }
};

}  // namespace


// process and remove repeated vertices if requested
void mjCMesh::ProcessVertices(const std::vector<float>& vert, bool remove_repeated) {
  vert_.clear();
  int nvert = vert.size();

  if (nvert % 3) { throw mjCError(this, "vertex data must be a multiple of 3"); }
  if (face_.size() % 3) { throw mjCError(this, "face data must be a multiple of 3"); }

  // convert vertices to double precision, may contain repeated vertices
  if (!remove_repeated) {
    vert_.reserve(nvert);
    for (int i = 0; i < nvert / 3; ++i) {
      const float* v = &vert[3 * i];
      if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        throw mjCError(this, "vertex coordinate %d is not finite", nullptr, i);
      }
      vert_.push_back(v[0]);
      vert_.push_back(v[1]);
      vert_.push_back(v[2]);
    }
    return;
  }

  int index = 0;

  std::unordered_map<VertexKey, int, VertexKey> vertex_map;

  // populate vertex map with new vertex indices
  for (int i = 0; i < nvert; i += 3) {
    const float* v = &vert[i];

    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
      throw mjCError(this, "vertex coordinate %d is not finite", nullptr, i);
    }

    VertexKey key = {v[0], v[1], v[2]};
    if (vertex_map.find(key) == vertex_map.end()) {
      vertex_map.insert({key, index});
      ++index;
    }
  }

  // no repeated vertices (just copy vertex data)
  if (3 * index == nvert) {
    vert_.reserve(nvert);
    for (float v : vert) { vert_.push_back(v); }
    return;
  }

  // update face vertex indices
  for (int i = 0; i < face_.size(); ++i) {
    VertexKey key = {vert[3 * face_[i]], vert[3 * face_[i] + 1], vert[3 * face_[i] + 2]};
    face_[i]      = vertex_map[key];
  }

  // repopulate vertex data
  vert_.resize(3 * index);
  for (const auto& pair : vertex_map) {
    const VertexKey& key   = pair.first;
    int              index = pair.second;

    // double precision
    vert_[3 * index + 0] = key.v[0];
    vert_[3 * index + 1] = key.v[1];
    vert_[3 * index + 2] = key.v[2];
  }
}


bool mjCMesh::IsMSH(std::string_view filename, std::string_view ct) {
  std::string asset_type = GetAssetContentType(filename, ct);
  return asset_type == "model/vnd.mujoco.msh";
}


bool mjCMesh::IsMSH() const {
  return content_type_ == "model/vnd.mujoco.msh";
}

// load mesh using decoder plugin
void mjCMesh::LoadFromDecoder(mjResource* resource, bool remove_repeated) {
  const mjpDecoder* decoder = mjp_findDecoder(resource, content_type_.c_str());
  if (!decoder) { throw mjCError(this, "no decoder found for mesh file '%s'", resource->name); }
  char    error[1024] = "";
  mjSpec* mesh_spec   = decoder->decode(resource, nullptr, error, sizeof(error));
  if (!mesh_spec) {
    if (error[0]) { throw mjCError(this, "%s", error); }
    throw mjCError(this, "decoder failed for mesh file '%s'", resource->name);
  }
  mjsElement* elem = mjs_firstElement(mesh_spec, mjOBJ_MESH);
  if (elem) {
    mjsMesh* src_mesh = mjs_asMesh(elem);
    if (src_mesh) {
      normal_.assign(src_mesh->usernormal->begin(), src_mesh->usernormal->end());
      texcoord_.assign(src_mesh->usertexcoord->begin(), src_mesh->usertexcoord->end());
      face_.assign(src_mesh->userface->begin(), src_mesh->userface->end());
      facenormal_.assign(src_mesh->userfacenormal->begin(), src_mesh->userfacenormal->end());
      facetexcoord_.assign(src_mesh->userfacetexcoord->begin(), src_mesh->userfacetexcoord->end());
      if (src_mesh->usernode) {
        node_.assign(src_mesh->usernode->begin(), src_mesh->usernode->end());
      }
      if (src_mesh->usertet) { tet_.assign(src_mesh->usertet->begin(), src_mesh->usertet->end()); }

      std::vector<float> vert(src_mesh->uservert->begin(), src_mesh->uservert->end());
      mj_deleteSpec(mesh_spec);
      ProcessVertices(vert, remove_repeated);
      return;
    }
  }
  mj_deleteSpec(mesh_spec);
}


// load mesh from resource; throw error on failure
void mjCMesh::LoadFromResource(mjResource* resource, bool remove_repeated) {
  // set content type from resource name
  std::string asset_type = GetAssetContentType(resource->name, content_type_);
  content_type_          = asset_type;

  if (IsMSH()) {
    LoadMSH(resource, remove_repeated);
  } else {
    LoadFromDecoder(resource, remove_repeated);
  }
}


// compiler wrapper
void mjCMesh::Compile(const mjVFS* vfs) {
  try {
    TryCompile(vfs);
  } catch (mjCError err) {
    if (resource_ != nullptr) {
      mju_closeResource(resource_);
      resource_ = nullptr;
    }
    throw err;
  }
}


// compiler
void mjCMesh::TryCompile(const mjVFS* vfs) {
  using Clock   = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;
  std::fill_n(mesh_timer_, mjNCTIMER, 0.0);

  bool fromCache = false;
  CopyFromSpec();
  mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);

  Clock::time_point t0 = Clock::now();

  // load file
  if (!file_.empty()) {
    vert_.clear();
    face_.clear();
    normal_.clear();
    texcoord_.clear();
    facenormal_.clear();
    facetexcoord_.clear();
    node_.clear();
    tet_.clear();
    if (resource_ != nullptr) {
      mju_closeResource(resource_);
      resource_ = nullptr;
    }

    // copy paths from model if not already defined
    mujoco::user::FilePath meshdir_;
    meshdir_ = FilePath(mjs_getString(compiler->meshdir));

    // remove path from file if necessary
    if (model->spec.strippath) { file_ = mjuu_strippath(file_); }

    mjSpec*  owning_spec = model->FindSpec(compiler);
    FilePath filename    = meshdir_ + FilePath(file_);
    resource_            = LoadResource(owning_spec->modelfiledir->c_str(), filename.Str(), vfs);

    // set content type before the cache lookup, so it does not depend on a cache hit
    content_type_ = GetAssetContentType(resource_->name, content_type_);

    // try loading from cache
    if (cache != nullptr && LoadCachedMesh(cache, resource_)) {
      mju_closeResource(resource_);
      resource_ = nullptr;
      fromCache = true;
    }

    if (!fromCache) {
      LoadFromResource(resource_);

      // check repeated mesh data
      if (!normal_.empty() && !spec_normal_.empty()) {
        throw mjCError(this, "repeated normal specification");
      } else if (normal_.empty()) {
        normal_ = spec_normal_;
      }
      if (!texcoord_.empty() && !spec_texcoord_.empty()) {
        throw mjCError(this, "repeated texcoord specification");
      } else if (texcoord_.empty()) {
        texcoord_ = spec_texcoord_;
      }
      if (!face_.empty() && !spec_face_.empty()) {
        throw mjCError(this, "repeated face specification");
      } else if (face_.empty()) {
        face_ = spec_face_;
      }
      if (!vert_.empty() && !spec_vert_.empty()) {
        throw mjCError(this, "repeated vertex specification");
      } else if (vert_.empty()) {
        ProcessVertices(spec_vert_);
      }
      if (!facenormal_.empty() && !spec_facenormal_.empty()) {
        throw mjCError(this, "repeated facenormal specification");
      } else if (facenormal_.empty()) {
        facenormal_ = spec_facenormal_;
      }
      if (!facetexcoord_.empty() && !spec_facetexcoord_.empty()) {
        throw mjCError(this, "repeated facetexcoord specification");
      } else if (facetexcoord_.empty()) {
        facetexcoord_ = spec_facetexcoord_;
      }
      if (!node_.empty() && !spec_node_.empty()) {
        throw mjCError(this, "repeated node specification");
      } else if (node_.empty()) {
        node_ = spec_node_;
      }
      if (!tet_.empty() && !spec_tet_.empty()) {
        throw mjCError(this, "repeated tet specification");
      } else if (tet_.empty()) {
        tet_ = spec_tet_;
      }
    }
  } else if (plugin.active) {
    LoadSDF();  // create using marching cubes
  }

  mesh_timer_[mjCTIMER_MESH_LOAD] = Seconds(Clock::now() - t0).count();

  CheckInitialMesh();

  // compute mesh properties
  if (!fromCache) {
    Process();
    if (!file_.empty()) { CacheMesh(cache, resource_); }
  } else {
    // When a mesh is loaded from the cache, has no octree but needs one,
    // we need to compute it here. If inversely it has an octree but we *do not*
    // need one, we clear it.
    t0 = Clock::now();
    if (!needsdf && !needsdf_) {
      octree_.Clear();
    } else if (octree_.NumNodes() == 0) {
      std::vector<double> dvert(vert_.begin(), vert_.end());
      octree_.SetFace(dvert, face_);
      octree_.SetMaxDepth(spec.octree_maxdepth);
      octree_.CreateOctree(aamm_);
      if (!plugin.active) {
        octree_.ComputeSdfCoeffs(dvert.data(), nvert(), face_.data(), nface(), tree_);
      }
    }
    mesh_timer_[mjCTIMER_MESH_OCTREE] = Seconds(Clock::now() - t0).count();
  }

  // close resource
  if (resource_ != nullptr) {
    mju_closeResource(resource_);
    resource_ = nullptr;
  }
}


// get bounding volume
void mjCMesh::SetBoundingVolume(int faceid, const double* dvert) {
  constexpr double kMaxVal      = std::numeric_limits<double>::max();
  double           face_aamm[6] = {kMaxVal, kMaxVal, kMaxVal, -kMaxVal, -kMaxVal, -kMaxVal};

  for (int j = 0; j < 3; j++) {
    int vertid   = face_[3 * faceid + j];
    face_aamm[0] = std::min(face_aamm[0], dvert[3 * vertid + 0]);
    face_aamm[1] = std::min(face_aamm[1], dvert[3 * vertid + 1]);
    face_aamm[2] = std::min(face_aamm[2], dvert[3 * vertid + 2]);
    face_aamm[3] = std::max(face_aamm[3], dvert[3 * vertid + 0]);
    face_aamm[4] = std::max(face_aamm[4], dvert[3 * vertid + 1]);
    face_aamm[5] = std::max(face_aamm[5], dvert[3 * vertid + 2]);
  }

  face_aabb_.push_back(.5 * (face_aamm[0] + face_aamm[3]));
  face_aabb_.push_back(.5 * (face_aamm[1] + face_aamm[4]));
  face_aabb_.push_back(.5 * (face_aamm[2] + face_aamm[5]));
  face_aabb_.push_back(.5 * (face_aamm[3] - face_aamm[0]));
  face_aabb_.push_back(.5 * (face_aamm[4] - face_aamm[1]));
  face_aabb_.push_back(.5 * (face_aamm[5] - face_aamm[2]));

  tree_.AddBoundingVolume(faceid, 1, 1, center_ + 3 * faceid, nullptr, &face_aabb_[6 * faceid]);
}


double* mjCMesh::GetPosPtr() {
  return pos_;
}


double* mjCMesh::GetQuatPtr() {
  return quat_;
}


bool mjCMesh::HasTexcoord() const {
  return !texcoord_.empty();
}


void mjCMesh::CopyVert(float* arr) const {
  std::copy(vert_.begin(), vert_.end(), arr);
}


void mjCMesh::CopyNormal(float* arr) const {
  std::copy(normal_.begin(), normal_.end(), arr);
}


void mjCMesh::CopyFace(int* arr) const {
  std::copy(face_.begin(), face_.end(), arr);
}


void mjCMesh::CopyFaceTexcoord(int* arr) const {
  std::copy(facetexcoord_.begin(), facetexcoord_.end(), arr);
}


void mjCMesh::CopyFaceNormal(int* arr) const {
  std::copy(facenormal_.begin(), facenormal_.end(), arr);
}


void mjCMesh::CopyTexcoord(float* arr) const {
  std::copy(texcoord_.begin(), texcoord_.end(), arr);
}


void mjCMesh::CopyGraph(int* arr) const {
  std::copy(graph_, graph_ + szgraph_, arr);
}


void mjCMesh::CopyPolygons(int* verts, int* adr, int* num, int poly_adr) const {
  int n = polygons_.size(), count = 0;
  for (int i = 0; i < n; ++i) {
    int m = num[i]  = polygons_[i].size();
    adr[i]          = poly_adr + count;
    count          += m;
    for (int j = 0; j < m; ++j) { verts[adr[i] + j - poly_adr] = polygons_[i][j]; }
  }
}


void mjCMesh::CopyPolygonMap(int* faces, int* adr, int* num, int poly_adr) const {
  int n = polygon_map_.size(), count = 0;
  for (int i = 0; i < n; ++i) {
    int m = num[i]  = polygon_map_[i].size();
    adr[i]          = poly_adr + count;
    count          += m;
    for (int j = 0; j < m; ++j) { faces[adr[i] + j - poly_adr] = polygon_map_[i][j]; }
  }
}


void mjCMesh::CopyPolygonNormals(mjtNum* arr) {
  for (int i = 0; i < polygon_normals_.size(); i += 3) {
    arr[i + 0] = (mjtNum)polygon_normals_[i + 0];
    arr[i + 1] = (mjtNum)polygon_normals_[i + 1];
    arr[i + 2] = (mjtNum)polygon_normals_[i + 2];
  }
}


// set geom size to match mesh
void mjCMesh::FitGeom(mjCGeom* geom, double center[3]) {
  // use inertial box
  if (!model->compiler.fitaabb) {
    // get inertia box type (shell or volume)
    double* boxsz = GetInertiaBoxPtr();
    switch (geom->type) {
      case mjGEOM_SPHERE:
        geom->size[0] = (boxsz[0] + boxsz[1] + boxsz[2]) / 3;
        break;

      case mjGEOM_CAPSULE:
        geom->size[0] = (boxsz[0] + boxsz[1]) / 2;
        geom->size[1] = max(0.0, boxsz[2] - geom->size[0] / 2);
        break;

      case mjGEOM_CYLINDER:
        geom->size[0] = (boxsz[0] + boxsz[1]) / 2;
        geom->size[1] = boxsz[2];
        break;

      case mjGEOM_ELLIPSOID:
      case mjGEOM_BOX:
        geom->size[0] = boxsz[0];
        geom->size[1] = boxsz[1];
        geom->size[2] = boxsz[2];
        break;

      default:
        throw mjCError(this, "invalid geom type in fitting mesh %s", name.c_str());
    }
  }

  // use aamm
  else {
    // find aabb box center and size
    center[0] = (aamm_[0] + aamm_[3]) / 2;
    center[1] = (aamm_[1] + aamm_[4]) / 2;
    center[2] = (aamm_[2] + aamm_[5]) / 2;

    double size[3] = {aamm_[3] - center[0], aamm_[4] - center[1], aamm_[5] - center[2]};

    // compute smallest geom whose aabb contains the mesh aabb
    switch (geom->type) {
      case mjGEOM_SPHERE:
        geom->size[0] = max(max(size[0], size[1]), size[2]);
        break;

      case mjGEOM_CAPSULE:
      case mjGEOM_CYLINDER:
        // find maximum distance in XY, separately in Z
        geom->size[0] = max(size[0], size[1]);
        geom->size[1] = size[2];

        // special handling of capsule: consider curved cap
        if (geom->type == mjGEOM_CAPSULE) { geom->size[1] -= geom->size[0]; }
        break;

      case mjGEOM_ELLIPSOID:
      case mjGEOM_BOX:
        geom->size[0] = size[0];
        geom->size[1] = size[1];
        geom->size[2] = size[2];
        break;

      default:
        throw mjCError(this, "invalid fittype in mesh %s", name.c_str());
    }
  }

  // rescale size
  geom->size[0] *= geom->fitscale;
  geom->size[1] *= geom->fitscale;
  geom->size[2] *= geom->fitscale;
}


// load mesh from cached asset, return true on success
bool mjCMesh::LoadCachedMesh(mjCCache* cache, const mjResource* resource) {
  auto process_mesh = [&](const void* data) {
    const mjCMesh* mesh = static_cast<const mjCMesh*>(data);
    // check if maxhullvert is different
    if (maxhullvert_ != mesh->maxhullvert_) { return false; }

    // check if inertia is different
    if (inertia != mesh->inertia) { return false; }

    // check if normal generation is different
    if (smoothnormal != mesh->smoothnormal) { return false; }

    // check if octree_maxdepth is different
    if (octree_maxdepth != mesh->octree_maxdepth) { return false; }

    // check if scale is different
    if (scale[0] != mesh->scale[0] || scale[1] != mesh->scale[1] || scale[2] != mesh->scale[2]) {
      return false;
    }

    // check if refpos is different
    if (refpos[0] != mesh->refpos[0] ||
        refpos[1] != mesh->refpos[1] ||
        refpos[2] != mesh->refpos[2]) {
      return false;
    }

    // check if refquat is different
    if (refquat[0] != mesh->refquat[0] ||
        refquat[1] != mesh->refquat[1] ||
        refquat[2] != mesh->refquat[2] ||
        refquat[3] != mesh->refquat[3]) {
      return false;
    }

    // check if need hull
    if (needhull_ && !mesh->szgraph_) { return false; }

    processed_    = mesh->processed_;
    vert_         = mesh->vert_;
    normal_       = mesh->normal_;
    texcoord_     = mesh->texcoord_;
    face_         = mesh->face_;
    facenormal_   = mesh->facenormal_;
    facetexcoord_ = mesh->facetexcoord_;
    node_         = mesh->node_;
    tet_          = mesh->tet_;
    halfedge_     = mesh->halfedge_;

    // only copy graph if needed
    if (needhull_ || mesh->face_.empty()) {
      szgraph_ = mesh->szgraph_;
      graph_   = nullptr;
      if (szgraph_) {
        graph_ = (int*)mju_malloc(szgraph_ * sizeof(int));
        std::copy(mesh->graph_, mesh->graph_ + szgraph_, graph_);
      }
    }

    if (szgraph_) {
      polygons_        = mesh->polygons_;
      polygon_normals_ = mesh->polygon_normals_;
      polygon_map_     = mesh->polygon_map_;
    } else {
      polygon_map_.resize(nvert());
    }
    surface_ = mesh->surface_;
    volume_  = mesh->volume_;
    std::copy(mesh->boxsz_, mesh->boxsz_ + 3, boxsz_);
    std::copy(mesh->aamm_, mesh->aamm_ + 6, aamm_);
    std::copy(mesh->pos_, mesh->pos_ + 3, pos_);
    std::copy(mesh->quat_, mesh->quat_ + 4, quat_);

    center_     = nullptr;
    int ncenter = mesh->face_.size();
    if (ncenter) {
      center_ = (double*)mju_malloc(ncenter * sizeof(double));
      std::copy(mesh->center_, mesh->center_ + ncenter, center_);
    }
    tree_      = mesh->tree_;
    face_aabb_ = mesh->face_aabb_;
    octree_    = mesh->octree_;
    return true;
  };

  // check that cached asset has all data
  return cache->PopulateData(CacheId(resource), resource, process_mesh);
}


// load MSH binary mesh
void mjCMesh::LoadMSH(mjResource* resource, bool remove_repeated) {
  // get file data in buffer
  char* buffer    = 0;
  int   buffer_sz = mju_readResource(resource, (const void**)&buffer);

  // still not found
  if (buffer_sz < 0) {
    throw mjCError(this, "could not read MSH file '%s'", resource->name);
  } else if (!buffer_sz) {
    throw mjCError(this, "MSH file '%s' is empty", resource->name);
  }

  // make sure header is present
  if (buffer_sz < 4 * sizeof(int)) {
    throw mjCError(this, "missing header in MSH file '%s'", resource->name);
  }

  // get sizes from header
  int nvbuf = 0, nfbuf = 0, nnbuf = 0, ntbuf = 0;
  ReadFromBuffer(&nvbuf, buffer);
  ReadFromBuffer(&nnbuf, buffer + sizeof(int));
  ReadFromBuffer(&ntbuf, buffer + 2 * sizeof(int));
  ReadFromBuffer(&nfbuf, buffer + 3 * sizeof(int));

  // check sizes
  if (nvbuf < 4 ||
      nfbuf < 0 ||
      nnbuf < 0 ||
      ntbuf < 0 ||
      (nnbuf > 0 && nnbuf != nvbuf) ||
      (ntbuf > 0 && ntbuf != nvbuf)) {
    throw mjCError(this, "invalid sizes in MSH file '%s'", resource->name);
  }

  if (nvbuf >= INT_MAX / sizeof(float) / 3 ||
      nnbuf >= INT_MAX / sizeof(float) / 3 ||
      ntbuf >= INT_MAX / sizeof(float) / 2 ||
      nfbuf >= INT_MAX / sizeof(int) / 3) {
    throw mjCError(this, "too large sizes in MSH file '%s'.", resource->name);
  }
  // check file size
  if (buffer_sz != 4 * sizeof(int) +
                       3 * nvbuf * sizeof(float) +
                       3 * nnbuf * sizeof(float) +
                       2 * ntbuf * sizeof(float) +
                       3 * nfbuf * sizeof(int)) {
    throw mjCError(this, "unexpected file size in MSH file '%s'", resource->name);
  }

  // allocate and copy
  using UnalignedFloat     = char[sizeof(float)];
  auto               fdata = reinterpret_cast<UnalignedFloat*>(buffer + 4 * sizeof(int));
  std::vector<float> vert;
  int                nvert = 0;
  if (nvbuf) {
    vert.assign(3 * nvbuf, 0);
    nvert = 3 * nvbuf;
    memcpy(vert.data(), fdata, nvert * sizeof(float));
    fdata += nvert;
  }
  if (nnbuf) {
    normal_.assign(nvert, 0);
    memcpy(normal_.data(), fdata, nvert * sizeof(float));
    fdata += nvert;
  }
  if (ntbuf) {
    texcoord_.assign(2 * (nvert / 3), 0);
    memcpy(texcoord_.data(), fdata, 2 * (nvert / 3) * sizeof(float));
    fdata += 2 * (nvert / 3);
  }
  if (nfbuf) {
    face_.assign(3 * nfbuf, 0);
    memcpy(face_.data(), fdata, 3 * nfbuf * sizeof(int));
  }
  if (nfbuf && !normal_.empty()) {
    facenormal_.assign(3 * nfbuf, 0);
    memcpy(facenormal_.data(), fdata, 3 * nfbuf * sizeof(int));
  }
  if (nfbuf && !texcoord_.empty()) {
    facetexcoord_.assign(3 * nfbuf, 0);
    memcpy(facetexcoord_.data(), fdata, 3 * nfbuf * sizeof(int));
  }

  ProcessVertices(vert, remove_repeated);
}


// compute the volume and center-of-mass of the mesh given the face centroid
double mjCMesh::ComputeVolume(double CoM[3], const double facecen[3], const double* dvert) const {
  double normal[3], center[3], total_volume = 0;
  CoM[0] = CoM[1] = CoM[2] = 0;
  int        nf            = (inertia == mjMESH_INERTIA_CONVEX) ? graph_[1] : nface();
  const int* f             = (inertia == mjMESH_INERTIA_CONVEX) ? GraphFaces() : face_.data();

  for (int i = 0; i < nf; i++) {
    double area = triangle(normal,
                           center,
                           &dvert[3 * f[3 * i]],
                           &dvert[3 * f[3 * i + 1]],
                           &dvert[3 * f[3 * i + 2]]);

    // compute and add volume
    double vec[3] = {center[0] - facecen[0], center[1] - facecen[1], center[2] - facecen[2]};
    double volume = mjuu_dot3(vec, normal) * area / 3;

    // if legacy computation requested, then always positive
    if (inertia == mjMESH_INERTIA_LEGACY) { volume = std::abs(volume); }

    // add pyramid com
    total_volume += volume;
    CoM[0]       += volume * (center[0] * 3.0 / 4.0 + facecen[0] / 4.0);
    CoM[1]       += volume * (center[1] * 3.0 / 4.0 + facecen[1] / 4.0);
    CoM[2]       += volume * (center[2] * 3.0 / 4.0 + facecen[2] / 4.0);
  }

  // if volume is valid normalize CoM
  if (total_volume >= mjMINVAL) {
    CoM[0] /= total_volume;
    CoM[1] /= total_volume;
    CoM[2] /= total_volume;
  }
  return total_volume;
}


// compute the surface area and center-of-mass of the mesh given the face centroid
double mjCMesh::ComputeSurfaceArea(double        CoM[3],
                                   const double  facecen[3],
                                   const double* dvert) const {
  double surface = 0;
  CoM[0] = CoM[1] = CoM[2] = 0;
  for (int i = 0; i < nface(); i++) {
    double area, center[3];
    area = triangle(nullptr,
                    center,
                    &dvert[3 * face_[3 * i]],
                    &dvert[3 * face_[3 * i + 1]],
                    &dvert[3 * face_[3 * i + 2]]);

    // add pyramid com
    surface += area;
    CoM[0]  += area * (center[0] * 3.0 / 4.0 + facecen[0] / 4.0);
    CoM[1]  += area * (center[1] * 3.0 / 4.0 + facecen[1] / 4.0);
    CoM[2]  += area * (center[2] * 3.0 / 4.0 + facecen[2] / 4.0);
  }

  // if area is valid normalize CoM
  if (surface >= mjMINVAL) {
    CoM[0] /= surface;
    CoM[1] /= surface;
    CoM[2] /= surface;
  }
  return surface;
}


// apply transformations
void mjCMesh::ApplyTransformations(double* dvert) {
  if (refpos[0] != 0 || refpos[1] != 0 || refpos[2] != 0) {
    int nv = nvert();
    for (int i = 0; i < nv; i++) {
      dvert[3 * i + 0] -= refpos[0];
      dvert[3 * i + 1] -= refpos[1];
      dvert[3 * i + 2] -= refpos[2];
    }
  }

  if (refquat[0] != 1 || refquat[1] != 0 || refquat[2] != 0 || refquat[3] != 0) {
    double quat[4] = {refquat[0], refquat[1], refquat[2], refquat[3]};
    double mat[9];
    mjuu_normvec(quat, 4);
    mjuu_quat2mat(mat, quat);

    for (int i = 0; i < nvert(); i++) { mjuu_mulvecmatT(&dvert[3 * i], &dvert[3 * i], mat); }

    // process normals
    for (int i = 0; i < nnormal(); i++) {
      double n1[3], n0[3] = {normal_[3 * i], normal_[3 * i + 1], normal_[3 * i + 2]};
      mjuu_mulvecmatT(n1, n0, mat);
      normal_[3 * i]     = (float)n1[0];
      normal_[3 * i + 1] = (float)n1[1];
      normal_[3 * i + 2] = (float)n1[2];
    }
  }

  // scale
  if (scale[0] != 1 || scale[1] != 1 || scale[2] != 1) {
    for (int i = 0; i < nvert(); i++) {
      dvert[3 * i + 0] *= scale[0];
      dvert[3 * i + 1] *= scale[1];
      dvert[3 * i + 2] *= scale[2];
    }

    // normals are covectors: transform by the inverse scale
    for (int i = 0; i < nnormal(); i++) {
      normal_[3 * i + 0] /= scale[0];
      normal_[3 * i + 1] /= scale[1];
      normal_[3 * i + 2] /= scale[2];
    }
  }

  // the map of an instance which does not multiply the scale; normals by the inverse transpose
  if (!InstanceDiagonal()) {
    const double* m = instancemap_.data();
    for (int i = 0; i < nvert(); i++) { mjuu_mulvecmat(&dvert[3 * i], &dvert[3 * i], m); }
    double cof[9] = {m[4] * m[8] - m[5] * m[7],
                     m[5] * m[6] - m[3] * m[8],
                     m[3] * m[7] - m[4] * m[6],
                     m[2] * m[7] - m[1] * m[8],
                     m[0] * m[8] - m[2] * m[6],
                     m[1] * m[6] - m[0] * m[7],
                     m[1] * m[5] - m[2] * m[4],
                     m[2] * m[3] - m[0] * m[5],
                     m[0] * m[4] - m[1] * m[3]};
    for (int i = 0; i < nnormal(); i++) {
      double n[3] = {normal_[3 * i], normal_[3 * i + 1], normal_[3 * i + 2]};
      mjuu_mulvecmat(n, n, cof);
      for (int k = 0; k < 3; k++) { normal_[3 * i + k] = (float)n[k]; }
    }
  }

  // normalize normals
  for (int i = 0; i < nnormal(); i++) {
    // compute length
    float len = normal_[3 * i] * normal_[3 * i] +
                normal_[3 * i + 1] * normal_[3 * i + 1] +
                normal_[3 * i + 2] * normal_[3 * i + 2];

    // rescale
    if (len > mjMINVAL) {
      float scl           = 1 / sqrtf(len);
      normal_[3 * i + 0] *= scl;
      normal_[3 * i + 1] *= scl;
      normal_[3 * i + 2] *= scl;
    } else {
      normal_[3 * i + 0] = 0;
      normal_[3 * i + 1] = 0;
      normal_[3 * i + 2] = 1;
    }
  }
}


// apply user transformations, then translate by -pos and rotate by quat^-1 (as for vertices)
void mjCMesh::TransformNodes(const double pos[3], const double quat[4]) {
  if (node_.empty()) { return; }

  double refmat[9], mat[9];
  double rquat[4] = {refquat[0], refquat[1], refquat[2], refquat[3]};
  mjuu_normvec(rquat, 4);
  mjuu_quat2mat(refmat, rquat);
  double neg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  mjuu_quat2mat(mat, neg);

  for (int i = 0; i < nnode(); i++) {
    double* p = &node_[3 * i];
    for (int j = 0; j < 3; j++) { p[j] -= refpos[j]; }
    mjuu_mulvecmatT(p, p, refmat);
    for (int j = 0; j < 3; j++) { p[j] = p[j] * scale[j] - pos[j]; }
    mjuu_mulvecmat(p, p, mat);
  }
}


// find centroid of faces, return total area
double mjCMesh::ComputeFaceCentroid(double facecen[3], const double* dvert) const {
  double total_area = 0;

  for (int i = 0; i < nface(); i++) {
    double area, center[3];
    area = triangle(nullptr,
                    center,
                    &dvert[3 * face_[3 * i]],
                    &dvert[3 * face_[3 * i + 1]],
                    &dvert[3 * face_[3 * i + 2]]);

    // accumulate
    facecen[0] += area * center[0];
    facecen[1] += area * center[1];
    facecen[2] += area * center[2];
    total_area += area;
  }

  // finalize centroid of faces
  if (total_area >= mjMINVAL) {
    facecen[0] /= total_area;
    facecen[1] /= total_area;
    facecen[2] /= total_area;
  }
  return total_area;
}


void mjCMesh::Process() {
  std::vector<double> dvert(vert_.begin(), vert_.end());
  using Clock   = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;
  Clock::time_point t0;
  // create half-edge structure (if mesh was in XML)
  if (halfedge_.empty()) {
    for (int i = 0; i < nface(); i++) {
      int v0 = face_[3 * i + 0];
      int v1 = face_[3 * i + 1];
      int v2 = face_[3 * i + 2];
      if (triangle(nullptr, nullptr, &dvert[3 * v0], &dvert[3 * v1], &dvert[3 * v2]) >
          sqrt(mjMINVAL)) {
        halfedge_.push_back({v0, v1});
        halfedge_.push_back({v1, v2});
        halfedge_.push_back({v2, v0});
      } else {
        // TODO(b/255525326)
      }
    }
  }

  // check for inconsistent face orientations
  if (!halfedge_.empty()) {
    std::stable_sort(halfedge_.begin(), halfedge_.end());
    auto iterator = std::adjacent_find(halfedge_.begin(), halfedge_.end());
    if (iterator != halfedge_.end() && inertia == mjMESH_INERTIA_EXACT) {
      throw mjCError(this,
                     "faces of mesh '%s' have inconsistent orientation. Please check the "
                     "faces containing the vertices %d and %d.",
                     name.c_str(),
                     iterator->first + 1,
                     iterator->second + 1);
    }
  }

  t0 = Clock::now();
  // make graph describing convex hull
  if (needhull_ || face_.empty()) { MakeGraph(dvert.data()); }

  // no faces: copy from convex hull
  if (face_.empty()) { CopyGraph(); }
  mesh_timer_[mjCTIMER_MESH_HULL] += Seconds(Clock::now() - t0).count();

  t0 = Clock::now();
  // no normals: make
  if (normal_.empty()) { MakeNormal(dvert.data()); }

  // check facenormal size
  if (!facenormal_.empty() && facenormal_.size() != face_.size()) {
    throw mjCError(this, "face data must have the same size as face normal data");
  }

  // no facetexcoord: copy from faces
  if (facetexcoord_.empty() && !texcoord_.empty()) { facetexcoord_ = face_; }

  // facenormal might not exist if usernormal was specified
  if (facenormal_.empty()) {
    int normal_per_vertex = normal_.size() / vert_.size();
    facenormal_.assign(face_.size(), 0);
    for (int i = 0; i < face_.size(); i++) { facenormal_[i] = normal_per_vertex * face_[i]; }
  }

  if (szgraph_) {
    MakePolygons(dvert.data());
  } else {
    polygon_map_.resize(nvert());
  }

  // correct winding order for left-handed coordinate systems
  bool righthand = scale[0] * scale[1] * scale[2] > 0;
  if (!righthand) {
    for (size_t i = 0; i < face_.size(); i += 3) { std::swap(face_[i + 1], face_[i + 2]); }
    for (size_t i = 0; i < facenormal_.size(); i += 3) {
      std::swap(facenormal_[i + 1], facenormal_[i + 2]);
    }
    for (size_t i = 0; i < facetexcoord_.size(); i += 3) {
      std::swap(facetexcoord_[i + 1], facetexcoord_[i + 2]);
    }
    if (graph_) {
      int* faces = GraphFaces();
      for (int i = 0; i < graph_[1]; i++) { std::swap(faces[3 * i + 1], faces[3 * i + 2]); }
      for (auto& polygon : polygons_) { std::reverse(polygon.begin() + 1, polygon.end()); }
    }
  }

  mesh_timer_[mjCTIMER_MESH_POLYGON] += Seconds(Clock::now() - t0).count();

  t0 = Clock::now();
  // user offset, rotation, scaling
  ApplyTransformations(dvert.data());

  // find centroid of faces
  double facecen[3] = {0, 0, 0};
  if (ComputeFaceCentroid(facecen, dvert.data()) < mjMINVAL) {
    throw mjCError(this, "mesh surface area is too small: %s", name.c_str());
  }

  // compute inertia and transform mesh. The mesh is transformed such that it is
  // centered at the CoM and the axes are the principle axes of inertia
  double CoM[3]   = {0, 0, 0};
  double inert[6] = {0, 0, 0, 0, 0, 0};

  // compute CoM and volume/area
  if (inertia == mjMESH_INERTIA_SHELL) {
    surface_ = ComputeSurfaceArea(CoM, facecen, dvert.data());
    if (surface_ < mjMINVAL) {
      throw mjCError(this, "mesh surface area is too small: %s", name.c_str());
    }
  } else {
    if ((volume_ = ComputeVolume(CoM, facecen, dvert.data())) < mjMINVAL) {
      if (volume_ < 0) {
        throw mjCError(this, "mesh volume is negative (misoriented triangles): %s", name.c_str());
      } else {
        throw mjCError(this,
                       "mesh volume is too small: %s . Try setting inertia to shell",
                       name.c_str());
      }
    }
  }

  // compute inertia
  double total_volume = ComputeInertia(inert, CoM, dvert.data());
  if (inertia == mjMESH_INERTIA_SHELL) {
    surface_ = total_volume;
  } else {
    volume_ = total_volume;
  }

  // get quaternion and diagonal inertia (vertices are float32, so stop at 1e-7 relative)
  double eigval[3], eigvec[9], quattmp[4];
  double full[9] =
      {inert[0], inert[3], inert[4], inert[3], inert[1], inert[5], inert[4], inert[5], inert[2]};
  mjuu_eig3(eigval, eigvec, quattmp, full, 1e-7);

  constexpr double inequality_atol = 1e-9;
  constexpr double inequality_rtol = 1e-6;

  // check eigval - SHOULD NOT OCCUR
  if (eigval[2] <= 0) {
    throw mjCError(this, "eigenvalue of mesh inertia must be positive: %s", name.c_str());
  }

  if (eigval[0] + eigval[1] < eigval[2] * (1.0 - inequality_rtol) - inequality_atol ||
      eigval[0] + eigval[2] < eigval[1] * (1.0 - inequality_rtol) - inequality_atol ||
      eigval[1] + eigval[2] < eigval[0] * (1.0 - inequality_rtol) - inequality_atol) {
    throw mjCError(this, "eigenvalues of mesh inertia violate A + B >= C: %s", name.c_str());
  }

  // compute sizes of equivalent inertia box
  double volume = GetVolumeRef();
  boxsz_[0]     = 0.5 * std::sqrt(6 * (eigval[1] + eigval[2] - eigval[0]) / volume);
  boxsz_[1]     = 0.5 * std::sqrt(6 * (eigval[0] + eigval[2] - eigval[1]) / volume);
  boxsz_[2]     = 0.5 * std::sqrt(6 * (eigval[0] + eigval[1] - eigval[2]) / volume);

  // prevent reorientation if the mesh was autogenerated using marching cubes
  if (!needreorient_) {
    mjuu_setvec(CoM, 0, 0, 0);
    mjuu_setvec(quattmp, 1, 0, 0, 0);
  }

  // transform CoM to origin
  for (int i = 0; i < nvert(); i++) {
    dvert[3 * i + 0] -= CoM[0];
    dvert[3 * i + 1] -= CoM[1];
    dvert[3 * i + 2] -= CoM[2];
  }
  Rotate(quattmp, dvert.data());

  // save the pos and quat that was used to transform the mesh
  mjuu_copyvec(pos_, CoM, 3);
  mjuu_copyvec(quat_, quattmp, 4);

  // nodes (e.g. volumetric meshes) follow the same transformation as vertices
  TransformNodes(CoM, quattmp);

  processed_ = true;

  // no radii: make
  if (!center_) { MakeCenter(dvert.data()); }

  // recompute polygon normals
  MakePolygonNormals(dvert.data());
  mesh_timer_[mjCTIMER_MESH_INERTIA] += Seconds(Clock::now() - t0).count();

  t0 = Clock::now();
  // make bounding volume hierarchy
  if (tree_.Bvh().empty()) {
    face_aabb_.clear();
    face_aabb_.reserve(3 * face_.size());
    tree_.AllocateBoundingVolumes(nface());
    for (int i = 0; i < nface(); i++) { SetBoundingVolume(i, dvert.data()); }
    tree_.CreateBVH(model, this);
  }
  mesh_timer_[mjCTIMER_MESH_BVH] += Seconds(Clock::now() - t0).count();

  t0 = Clock::now();
  // make octree
  if (needsdf || needsdf_) {
    octree_.SetFace(dvert, face_);
    octree_.SetMaxDepth(spec.octree_maxdepth);
    octree_.CreateOctree(aamm_);

    if (!plugin.active) {
      octree_.ComputeSdfCoeffs(dvert.data(), nvert(), face_.data(), nface(), tree_);
    }
  }

  // narrow back to float
  for (int i = 0; i < (int)dvert.size(); i++) { vert_[i] = (float)dvert[i]; }
  mesh_timer_[mjCTIMER_MESH_OCTREE] += Seconds(Clock::now() - t0).count();
}


// compute abstract (unitless) inertia, recompute area / volume
double mjCMesh::ComputeInertia(double inert[6], const double CoM[3], const double* dvert) const {
  double total_volume = 0;

  // copy vertices to avoid modifying the original mesh
  std::vector<double> vert_centered;
  vert_centered.reserve(3 * nvert());

  // translate vertices to origin in order to compute inertia
  for (int i = 0; i < nvert(); i++) {
    vert_centered.push_back(dvert[3 * i + 0] - CoM[0]);
    vert_centered.push_back(dvert[3 * i + 1] - CoM[1]);
    vert_centered.push_back(dvert[3 * i + 2] - CoM[2]);
  }

  // accumulate products of inertia, recompute volume
  const int k[6][2] = {
      {0, 0},
      {1, 1},
      {2, 2},
      {0, 1},
      {0, 2},
      {1, 2}
  };
  double     P[6] = {0, 0, 0, 0, 0, 0};
  int        nf   = (inertia == mjMESH_INERTIA_CONVEX) ? graph_[1] : nface();
  const int* f    = (inertia == mjMESH_INERTIA_CONVEX) ? GraphFaces() : face_.data();
  for (int i = 0; i < nf; i++) {
    const double* D = &vert_centered[3 * f[3 * i + 0]];
    const double* E = &vert_centered[3 * f[3 * i + 1]];
    const double* F = &vert_centered[3 * f[3 * i + 2]];

    // get area, normal and center; update volume
    double normal[3], center[3];
    double volume, area = triangle(normal, center, D, E, F);
    if (inertia == mjMESH_INERTIA_SHELL) {
      volume = area;
    } else {
      volume = mjuu_dot3(center, normal) * area / 3;
    }

    // if legacy computation requested, then always positive
    if (inertia == mjMESH_INERTIA_LEGACY) { volume = abs(volume); }

    // apply formula, accumulate
    total_volume += volume;

    int C = (inertia == mjMESH_INERTIA_SHELL) ? 12 : 20;
    for (int j = 0; j < 6; j++) {
      P[j] += volume /
              C *
              (2 * (D[k[j][0]] * D[k[j][1]] + E[k[j][0]] * E[k[j][1]] + F[k[j][0]] * F[k[j][1]]) +
               D[k[j][0]] * E[k[j][1]] +
               D[k[j][1]] * E[k[j][0]] +
               D[k[j][0]] * F[k[j][1]] +
               D[k[j][1]] * F[k[j][0]] +
               E[k[j][0]] * F[k[j][1]] +
               E[k[j][1]] * F[k[j][0]]);
    }
  }

  // convert from products of inertia to moments of inertia
  inert[0] = P[1] + P[2];
  inert[1] = P[0] + P[2];
  inert[2] = P[0] + P[1];
  inert[3] = -P[3];
  inert[4] = -P[4];
  inert[5] = -P[5];
  return total_volume;
}


void mjCMesh::Rotate(double quat[4], double* dvert) {
  double neg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  double mat[9];
  mjuu_quat2mat(mat, neg);
  int nv = nvert();
  for (int i = 0; i < nv; i++) {
    mjuu_mulvecmat(&dvert[3 * i], &dvert[3 * i], mat);

    aamm_[0] = std::min(aamm_[0], dvert[3 * i + 0]);
    aamm_[3] = std::max(aamm_[3], dvert[3 * i + 0]);
    aamm_[1] = std::min(aamm_[1], dvert[3 * i + 1]);
    aamm_[4] = std::max(aamm_[4], dvert[3 * i + 1]);
    aamm_[2] = std::min(aamm_[2], dvert[3 * i + 2]);
    aamm_[5] = std::max(aamm_[5], dvert[3 * i + 2]);
  }

  for (int i = 0; i < nnormal(); i++) {
    // normals
    const double nrm[3] = {normal_[3 * i], normal_[3 * i + 1], normal_[3 * i + 2]};
    double       res[3];
    mjuu_mulvecmat(res, nrm, mat);
    for (int j = 0; j < 3; j++) { normal_[3 * i + j] = (float)res[j]; }
  }
}


void mjCMesh::CheckInitialMesh() const {
  if (vert_.size() < 12) { throw mjCError(this, "at least 4 vertices required"); }
  if (vert_.size() % 3) { throw mjCError(this, "vertex data must be a multiple of 3"); }
  if (normal_.size() % 3) { throw mjCError(this, "normal data must be a multiple of 3"); }
  if (texcoord_.size() % 2) { throw mjCError(this, "texcoord must be a multiple of 2"); }
  if (face_.size() % 3) { throw mjCError(this, "face data must be a multiple of 3"); }

  // check texcoord size if no face texcoord indices are given
  if (!texcoord_.empty() &&
      texcoord_.size() != 2 * nvert() &&
      facetexcoord_.empty() &&
      content_type_ != "model/obj") {
    throw mjCError(
        this,
        "texcoord must be 2*nv if face texcoord indices are not provided in an OBJ file");
  }

  // require vertices
  if (vert_.empty()) { throw mjCError(this, "no vertices"); }

  // check vertices exist
  int nv = nvert(), nf = face_.size();
  for (int i = 0; i < nf; i++) {
    if (face_[i] >= nv || face_[i] < 0) {
      throw mjCError(this, "in face %d, vertex index %d does not exist", nullptr, i / 3, face_[i]);
    }
  }
}


// return inertia pointer
double* mjCMesh::GetInertiaBoxPtr() {
  return boxsz_;
}


// return volume or surface area
double mjCMesh::GetVolumeRef() const {
  return (inertia == mjMESH_INERTIA_SHELL) ? surface_ : volume_;
}


// make graph describing convex hull
// NOLINTBEGIN(misc-include-cleaner)
void mjCMesh::MakeGraph(const double* dvert) {
  int      adr, ok, curlong, totlong, exitcode;
  facetT * facet, **facetp;
  vertexT *vertex, *vertex1, **vertex1p;

  std::string qhopt = "qhull Qt";
  if (maxhullvert_ > -1) {
    // qhull "Q9" picks the furthest of all furthest points across facets.
    // qhull "TA" actually means "number of vertices added after the initial simplex"
    qhopt += " Q9 TA" + std::to_string(maxhullvert_ - 4);
  }

  // graph not needed for small meshes
  if (nvert() < 4) { return; }

  // check for colocated/collinear/coplanar vertices
  {
    // find second vertex that is distinct from vertex 0
    int    v1   = -1;
    double len1 = 0;
    for (int i = 1; i < nvert(); i++) {
      len1 = mjuu_dist3(dvert + 3 * i, dvert);
      if (len1 > mjMINVAL) {
        v1 = i;
        break;
      }
    }

    // no second vertex found: all vertices are colocated
    if (v1 < 0) {
      throw mjCError(this,
                     "mesh '%s' has colocated vertices, cannot compute convex hull."
                     " Consider using a small sphere instead",
                     name.c_str());
    }

    // find first non-collinear triple to define a plane
    double edge1[3]  = {dvert[3 * v1 + 0] - dvert[0],
                        dvert[3 * v1 + 1] - dvert[1],
                        dvert[3 * v1 + 2] - dvert[2]};
    double normal[3] = {0, 0, 0};
    bool   collinear = true;
    for (int i = 1; i < nvert(); i++) {
      if (i == v1) continue;
      double edge2[3] = {dvert[3 * i + 0] - dvert[0],
                         dvert[3 * i + 1] - dvert[1],
                         dvert[3 * i + 2] - dvert[2]};
      double len2     = sqrt(mjuu_dot3(edge2, edge2));
      if (len2 < mjMINVAL) continue;
      mjuu_crossvec(normal, edge1, edge2);
      double norm = sqrt(mjuu_dot3(normal, normal));
      if (norm > mjMINVAL * len1 * len2) {
        normal[0] /= norm;
        normal[1] /= norm;
        normal[2] /= norm;
        collinear  = false;
        break;
      }
    }

    // vertices are collinear: cannot compute convex hull
    if (collinear) {
      throw mjCError(this,
                     "mesh '%s' has collinear vertices, cannot compute convex hull."
                     " Consider using a thin capsule instead",
                     name.c_str());
    }

    // find first vertex that is not on the plane
    double d        = mjuu_dot3(normal, dvert);
    bool   coplanar = true;
    for (int i = 0; i < nvert(); i++) {
      if (fabs(mjuu_dot3(normal, dvert + 3 * i) - d) > mjMINVAL * len1) {
        coplanar = false;
        break;
      }
    }

    // vertices are coplanar: cannot compute convex hull
    if (coplanar) {
      throw mjCError(this,
                     "mesh '%s' has coplanar vertices, cannot compute convex hull."
                     " Consider using a primitive geom type (plane or thin box) instead",
                     name.c_str());
    }
  }

  qhT  qh_qh;
  qhT* qh = &qh_qh;
  qh_zero(qh, stderr);

  // qhull basic init
  qh_init_A(qh, stdin, stdout, stderr, 0, nullptr);

  // install longjmp error handler
  exitcode      = setjmp(qh->errexit);
  qh->NOerrexit = false;
  if (!exitcode) {
    // actual init
    qh_initflags(qh, const_cast<char*>(qhopt.c_str()));
    qh_init_B(qh, const_cast<double*>(dvert), nvert(), 3, qh_False);

    // construct convex hull
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);

    // allocate graph:
    //  numvert, numface, vert_edgeadr[numvert], vert_globalid[numvert],
    //  edge_localid[numvert+3*numface], face_globalid[3*numface]
    int numvert = qh->num_vertices;
    int numface = qh->num_facets;
    szgraph_    = 2 + 3 * numvert + 6 * numface;
    graph_      = (int*)mju_malloc(szgraph_ * sizeof(int));
    graph_[0]   = numvert;
    graph_[1]   = numface;

    // pointers for convenience
    int* vert_edgeadr  = graph_ + 2;
    int* vert_globalid = graph_ + 2 + numvert;
    int* edge_localid  = graph_ + 2 + 2 * numvert;
    int* face_globalid = graph_ + 2 + 3 * numvert + 3 * numface;

    // fill in graph data
    int i = adr = 0;
    ok          = 1;
    FORALLvertices {
      // point id of this vertex, check
      int pid = qh_pointid(qh, vertex->point);
      if (pid < 0 || pid >= nvert()) {
        ok = 0;
        break;
      }

      // save edge address and global id of this vertex
      vert_edgeadr[i]  = adr;
      vert_globalid[i] = pid;

      // process neighboring faces and their vertices
      int start = adr;
      FOREACHsetelement_(facetT, vertex->neighbors, facet) {
        int cnt = 0;
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          cnt++;

          // point id of face vertex, check
          int pid1 = qh_pointid(qh, vertex1->point);
          if (pid1 < 0 || pid1 >= nvert()) {
            ok = 0;
            break;
          }

          // if different from vertex id, try to insert
          if (pid != pid1) {
            // check for previous record
            int j;
            for (j = start; j < adr; j++)
              if (pid1 == edge_localid[j]) { break; }

            // not found: insert
            if (j >= adr) { edge_localid[adr++] = pid1; }
          }
        }

        // make sure we have triangle: SHOULD NOT OCCUR
        if (cnt != 3) { mju_error("Qhull did not return triangle"); }
      }

      // insert separator, advance to next vertex
      edge_localid[adr++] = -1;
      i++;
    }

    // size check: SHOULD NOT OCCUR
    if (adr != numvert + 3 * numface) { mju_error("Wrong size in convex hull graph"); }

    // add triangle data, reorient faces if flipped
    adr = 0;
    FORALLfacets {
      int ii     = 0;
      int ind[3] = {0, 1, 2};
      if (facet->toporient) {
        ind[0] = 1;
        ind[1] = 0;
      }

      // copy triangle data
      FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
        // make sure we have triangle: SHOULD NOT OCCUR
        if (ii >= 3) { mju_error("Qhull did not return triangle"); }

        face_globalid[adr + ind[ii++]] = qh_pointid(qh, vertex1->point);
      }

      // advance to next triangle
      adr += 3;
    }

    // free all
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);

    // bad graph: delete
    if (!ok) {
      szgraph_ = 0;
      mju_free(graph_);
      graph_ = 0;
      mju_warning("Could not construct convex hull graph");
    }

    // replace global ids with local ids in edge data
    else {
      // invert vert_globalid: point id -> hull vertex index, -1 if not on hull.
      // every vert_globalid entry was range-checked above, so the index is safe.
      std::vector<int> hullid(nvert(), -1);
      for (int adr = 0; adr < numvert; adr++) { hullid[vert_globalid[adr]] = adr; }

      for (int i = 0; i < numvert + 3 * numface; i++) {
        if (edge_localid[i] >= 0) {
          int adr = hullid[edge_localid[i]];

          // make sure we found a match: SHOULD NOT OCCUR
          if (adr < 0) { mju_error("Vertex id not found in convex hull"); }

          edge_localid[i] = adr;
        }
      }
    }
  }

  // longjmp error handler
  else {
    // free all
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (graph_) {
      mju_free(graph_);
      szgraph_ = 0;
    }

    throw mjCError(this, "qhull error");
  }
}
// NOLINTEND(misc-include-cleaner)


// copy graph into face data
void mjCMesh::CopyGraph() {
  // only if face data is missing
  if (!face_.empty()) { return; }

  // get info from graph, allocate
  int numvert = graph_[0];
  face_.assign(3 * graph_[1], 0);

  // copy faces
  for (int i = 0; i < nface(); i++) {
    // address in graph
    int j = 2 + 3 * numvert + 3 * nface() + 3 * i;

    // copy
    face_[3 * i + 0] = graph_[j + 0];
    face_[3 * i + 1] = graph_[j + 1];
    face_[3 * i + 2] = graph_[j + 2];
  }
}


// make a mesh of a hemisphere (quad projected)
void mjCMesh::MakeHemisphere(int res, bool make_faces, bool make_cap) {
  constexpr double kNorthPole[3]  = {0, 0, 1};
  constexpr double kEquator[4][3] = {
      {1,  0,  0},
      {0,  1,  0},
      {-1, 0,  0},
      {0,  -1, 0},
  };

  // allocate vertices
  int nvert  = 1 + 2 * (res + 1) * (res + 2);
  nvert     += make_cap && make_faces;  // add center vertex for bottom cap faces
  std::vector<float> vert(3 * nvert);

  // north pole
  vert[0] = kNorthPole[0];
  vert[1] = kNorthPole[1];
  vert[2] = kNorthPole[2];

  // iterate through rows from north pole to equator, compute vertices
  int v = 1;
  for (int row = 0; row <= res; row++) {
    // iterate through the four sides
    for (int side = 0; side < 4; side++) {
      double factor = static_cast<double>(row + 1) / (res + 1);
      double start[3], end[3];

      // start and end points of current arc
      for (int i = 0; i < 3; i++) {
        start[i] = kNorthPole[i] + factor * (kEquator[side][i] - kNorthPole[i]);
        end[i]   = kNorthPole[i] + factor * (kEquator[(side + 1) % 4][i] - kNorthPole[i]);
      }

      // step size for interpolation along the arc
      double delta[3];
      for (int i = 0; i < 3; i++) { delta[i] = (end[i] - start[i]) / (row + 1); }

      // interpolate points along the arc
      for (int i = 0; i < row + 1; i++) {
        double p[3];
        for (int j = 0; j < 3; j++) { p[j] = start[j] + i * delta[j]; }

        // normalize point to lie on hemisphere surface
        double norm     = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        vert[3 * v + 0] = p[0] / norm;
        vert[3 * v + 1] = p[1] / norm;
        vert[3 * v + 2] = p[2] / norm;
        v++;
      }
    }
  }

  // optional center vertex for bottom cap (for symmetry)
  if (make_faces && make_cap) {
    vert[3 * (nvert - 1) + 0] = 0;
    vert[3 * (nvert - 1) + 1] = 0;
    vert[3 * (nvert - 1) + 2] = 0;
  }

  // save vertices
  mjs_setFloat(spec.uservert, vert.data(), 3 * nvert);

  if (make_faces) {
    // allocate faces
    int nface  = 4 * (res + 1) * (res + 1);
    nface     += make_cap * (4 * (res + 1));  // bottom cap faces
    std::vector<int> face(3 * nface);

    // faces connected to north pole
    int f     = 0;
    face[f++] = 0;
    face[f++] = 1;
    face[f++] = 2;
    face[f++] = 0;
    face[f++] = 2;
    face[f++] = 3;
    face[f++] = 0;
    face[f++] = 3;
    face[f++] = 4;
    face[f++] = 0;
    face[f++] = 4;
    face[f++] = 1;

    // faces on the hemisphere from north pole to equator
    for (int row = 0; row < res; row++) {
      const int start_curr = 2 * row * (row + 1) + 1;
      const int count_curr = 4 * (row + 1);
      const int start_next = start_curr + count_curr;
      const int count_next = 4 * (row + 2);

      for (int side = 0; side < 4; side++) {
        for (int i = 0; i < row + 2; i++) {
          const int v_curr = i + (row + 1) * side;
          const int v_next = i + (row + 2) * side;
          face[f++]        = start_curr + v_curr % count_curr;
          face[f++]        = start_next + v_next % count_next;
          face[f++]        = start_next + (v_next + 1) % count_next;

          if (i < row + 1) {
            face[f++] = start_curr + v_curr % count_curr;
            face[f++] = start_next + (v_next + 1) % count_next;
            face[f++] = start_curr + (v_curr + 1) % count_curr;
          }
        }
      }
    }

    if (make_cap) {
      // add faces for the bottom cap
      const int start = 2 * res * (res + 1) + 1;
      const int count = 4 * (res + 1);
      for (int i = 0; i < count; i++) {
        face[f++] = start + i;
        face[f++] = nvert - 1;
        face[f++] = start + (i + 1) % count;
      }
    }

    // save faces
    mjs_setInt(spec.userface, face.data(), 3 * nface);
  }
}


// make a mesh of a sphere using icosaheral subdivision
void mjCMesh::MakeSphere(int subdiv, bool make_faces) {
  // make icosahedron
  const float        phi  = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<float> vert = {
      -1.0, phi,  0.0,  1.0, phi, 0.0, -1.0, -phi, 0.0,  1.0,  -phi, 0.0,

      0.0,  -1.0, phi,  0.0, 1.0, phi, 0.0,  -1.0, -phi, 0.0,  1.0,  -phi,

      phi,  0.0,  -1.0, phi, 0.0, 1.0, -phi, 0.0,  -1.0, -phi, 0.0,  1.0,
  };

  // normalize vertices to be on a unit sphere
  const double norm = std::sqrt(1.0 + phi * phi);
  for (float& v : vert) { v /= norm; }

  std::vector<int> face = {0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
                           4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
                           6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1};

  // subdivision
  if (subdiv > 0) {
    // helper to get or create a midpoint vertex
    auto get_midpoint =
        [&vert](int v1_idx, int v2_idx, std::map<std::pair<int, int>, int>& cache) -> int {
      // key is the pair of vertex indices, sorted
      std::pair<int, int> key = std::minmax(v1_idx, v2_idx);

      // if midpoint is already in cache, return its index
      auto it = cache.find(key);
      if (it != cache.end()) { return it->second; }

      // otherwise, create it
      const float* v1 = &vert[v1_idx * 3];
      const float* v2 = &vert[v2_idx * 3];

      float mid_x = (v1[0] + v2[0]) / 2.0f;
      float mid_y = (v1[1] + v2[1]) / 2.0f;
      float mid_z = (v1[2] + v2[2]) / 2.0f;

      // normalize the new vertex to put it on the sphere
      float mid_norm  = std::sqrt(mid_x * mid_x + mid_y * mid_y + mid_z * mid_z);
      mid_x          /= mid_norm;
      mid_y          /= mid_norm;
      mid_z          /= mid_norm;

      // add the new vertex to the list
      int new_idx = vert.size() / 3;
      vert.push_back(mid_x);
      vert.push_back(mid_y);
      vert.push_back(mid_z);

      // add to cache
      cache[key] = new_idx;

      return new_idx;
    };

    // subdivision loop
    for (int i = 0; i < subdiv; ++i) {
      std::map<std::pair<int, int>, int> midpoint_cache;
      std::vector<int>                   new_face;
      new_face.reserve(face.size() * 4);
      for (size_t j = 0; j < face.size(); j += 3) {
        int v1 = face[j];
        int v2 = face[j + 1];
        int v3 = face[j + 2];

        int m12 = get_midpoint(v1, v2, midpoint_cache);
        int m23 = get_midpoint(v2, v3, midpoint_cache);
        int m31 = get_midpoint(v3, v1, midpoint_cache);

        new_face.insert(new_face.end(), {v1, m12, m31});
        new_face.insert(new_face.end(), {v2, m23, m12});
        new_face.insert(new_face.end(), {v3, m31, m23});
        new_face.insert(new_face.end(), {m12, m23, m31});
      }
      face = std::move(new_face);
    }
  }

  // save vertices and maybe faces
  mjs_setFloat(spec.uservert, vert.data(), vert.size());
  if (make_faces) mjs_setInt(spec.userface, face.data(), face.size());
}


// make a mesh of a supersphere
void mjCMesh::MakeSupersphere(int res, double e, double n) {
  // allocate vertices and faces
  int                nvert = (res - 1) * res + 2;
  int                nface = 2 * res * (res - 1);
  std::vector<float> vert;
  vert.reserve(3 * nvert);
  std::vector<int> face;
  face.reserve(3 * nface);

  // south pole
  vert.insert(vert.end(), {0.0f, 0.0f, -1.0f});

  // rings
  for (int i = 1; i < res; i++) {
    double v = -mjPI / 2 + i * mjPI / res;
    for (int j = 0; j < res; j++) {
      double u = -mjPI + j * 2 * mjPI / res;
      vert.push_back(aux_c(v, n) * aux_c(u, e));
      vert.push_back(aux_c(v, n) * aux_s(u, e));
      vert.push_back(aux_s(v, n));
    }
  }

  // north pole
  vert.insert(vert.end(), {0.0f, 0.0f, 1.0f});

  // south pole faces
  for (int j = 0; j < res; j++) {
    int v2 = 1 + j;
    int v3 = 1 + (j + 1) % res;
    face.insert(face.end(), {0, v3, v2});
  }

  // ring faces
  for (int i = 0; i < res - 2; i++) {
    for (int j = 0; j < res; j++) {
      int v1 = 1 + i * res + j;
      int v2 = 1 + i * res + (j + 1) % res;
      int v4 = 1 + (i + 1) * res + j;
      int v3 = 1 + (i + 1) * res + (j + 1) % res;
      face.insert(face.end(), {v1, v2, v4});
      face.insert(face.end(), {v2, v3, v4});
    }
  }

  // north pole faces
  int north_pole_idx      = nvert - 1;
  int last_ring_start_idx = 1 + (res - 2) * res;
  for (int j = 0; j < res; j++) {
    int v1 = last_ring_start_idx + j;
    int v2 = last_ring_start_idx + (j + 1) % res;
    face.insert(face.end(), {v1, v2, north_pole_idx});
  }

  // save vertices and faces
  mjs_setFloat(spec.uservert, vert.data(), vert.size());
  mjs_setInt(spec.userface, face.data(), face.size());
}


// make a mesh of a torus (subsumed by supertorus, kept for reference only)
void mjCMesh::MakeTorus(int res, double radius) {
  // allocate vertices and faces
  int                nvert = res * res;
  int                nface = res * res * 2;
  std::vector<float> vert(3 * nvert);
  std::vector<int>   face(3 * nface);

  // generate vertices
  for (int i = 0; i < res; ++i) {
    for (int j = 0; j < res; ++j) {
      double u           = 2 * mjPI * i / res;
      double v           = 2 * mjPI * j / res;
      int    vidx        = i * res + j;
      vert[3 * vidx + 0] = (1 + radius * cos(v)) * cos(u);
      vert[3 * vidx + 1] = (1 + radius * cos(v)) * sin(u);
      vert[3 * vidx + 2] = radius * sin(v);
    }
  }

  // generate faces
  int fidx = 0;
  for (int i = 0; i < res; ++i) {
    for (int j = 0; j < res; ++j) {
      int i_next = (i + 1) % res;
      int j_next = (j + 1) % res;

      int v1 = i * res + j;
      int v2 = i_next * res + j;
      int v3 = i_next * res + j_next;
      int v4 = i * res + j_next;

      // first triangle
      face[3 * fidx + 0] = v1;
      face[3 * fidx + 1] = v2;
      face[3 * fidx + 2] = v4;
      fidx++;

      // second triangle
      face[3 * fidx + 0] = v2;
      face[3 * fidx + 1] = v3;
      face[3 * fidx + 2] = v4;
      fidx++;
    }
  }

  // save vertices and faces
  mjs_setFloat(spec.uservert, vert.data(), vert.size());
  mjs_setInt(spec.userface, face.data(), face.size());
}


// make a mesh of a supertoroid, see https://en.wikipedia.org/wiki/Supertoroid
void mjCMesh::MakeSupertorus(int res, double radius, double s, double t) {
  // allocate vertices and faces
  int                nvert = res * res;
  int                nface = res * res * 2;
  std::vector<float> vert(3 * nvert);
  std::vector<int>   face(3 * nface);

  // generate vertices
  for (int i = 0; i < res; ++i) {
    for (int j = 0; j < res; ++j) {
      double u           = 2 * mjPI * i / res;
      double v           = 2 * mjPI * j / res;
      int    vidx        = i * res + j;
      vert[3 * vidx + 0] = (1 + radius * aux_c(v, s)) * aux_c(u, t);
      vert[3 * vidx + 1] = (1 + radius * aux_c(v, s)) * aux_s(u, t);
      vert[3 * vidx + 2] = radius * aux_s(v, s);
    }
  }

  // generate faces
  int fidx = 0;
  for (int i = 0; i < res; ++i) {
    for (int j = 0; j < res; ++j) {
      int i_next = (i + 1) % res;
      int j_next = (j + 1) % res;

      int v1 = i * res + j;
      int v2 = i_next * res + j;
      int v3 = i_next * res + j_next;
      int v4 = i * res + j_next;

      // first triangle
      face[3 * fidx + 0] = v1;
      face[3 * fidx + 1] = v2;
      face[3 * fidx + 2] = v4;
      fidx++;

      // second triangle
      face[3 * fidx + 0] = v2;
      face[3 * fidx + 1] = v3;
      face[3 * fidx + 2] = v4;
      fidx++;
    }
  }

  // save vertices and faces
  mjs_setFloat(spec.uservert, vert.data(), vert.size());
  mjs_setInt(spec.userface, face.data(), face.size());
}


// make a mesh of a spherical wedge
void mjCMesh::MakeWedge(int resolution[2], double fov[2], double gamma) {
  std::vector<double> x_edges(resolution[0] + 1, 0);
  std::vector<double> y_edges(resolution[1] + 1, 0);
  BinEdges(x_edges.data(), y_edges.data(), resolution, fov, gamma);
  std::vector<float> uservert(3 * resolution[0] * resolution[1], 0);
  std::vector<float> usernormal(9 * resolution[0] * resolution[1], 0);

  for (int i = 0; i < resolution[0]; i++) {
    for (int j = 0; j < resolution[1]; j++) {
      double aer[3];
      aer[0] = 0.5 * (x_edges[i + 1] + x_edges[i]);
      aer[1] = 0.5 * (y_edges[j + 1] + y_edges[j]);
      aer[2] = 1;
      SphericalToCartesian(aer, uservert.data() + 3 * (i * resolution[1] + j));
      TangentFrame(aer, usernormal.data() + 9 * (i * resolution[1] + j));
    }
  }

  mjs_setFloat(spec.uservert, uservert.data(), 3 * resolution[0] * resolution[1]);
  mjs_setFloat(spec.usernormal, usernormal.data(), 9 * resolution[0] * resolution[1]);
}


// make a mesh of a rectangle
void mjCMesh::MakeRect(int resolution[2]) {
  std::vector<double> x_edges(resolution[0] + 1, 0);
  std::vector<double> y_edges(resolution[1] + 1, 0);
  LinSpace(-1, 1, resolution[0] + 1, x_edges.data());
  LinSpace(-1, 1, resolution[1] + 1, y_edges.data());
  std::vector<float> uservert(3 * resolution[0] * resolution[1], 0);
  std::vector<float> usernormal(9 * resolution[0] * resolution[1], 0);
  std::vector<int>   userface(6 * (resolution[0] - 1) * (resolution[1] - 1), 0);
  spec.inertia = mjMESH_INERTIA_SHELL;

  for (int i = 0; i < resolution[0]; i++) {
    for (int j = 0; j < resolution[1]; j++) {
      int    vert              = i * resolution[1] + j;
      mjtNum dx                = 2. / resolution[0];
      mjtNum dy                = 2. / resolution[1];
      uservert[3 * vert + 0]   = -1 + (i + 0.5) * dx;
      uservert[3 * vert + 1]   = -1 + (j + 0.5) * dy;
      uservert[3 * vert + 2]   = -1;
      usernormal[9 * vert + 0] = 1;
      usernormal[9 * vert + 4] = 1;
      usernormal[9 * vert + 8] = 1;
      if (i > 0 && j > 0) {
        int cell               = (i - 1) * (resolution[1] - 1) + j - 1;
        userface[6 * cell + 0] = (i - 1) * resolution[1] + j - 1;
        userface[6 * cell + 1] = (i - 0) * resolution[1] + j - 1;
        userface[6 * cell + 2] = (i - 1) * resolution[1] + j - 0;
        userface[6 * cell + 3] = (i - 0) * resolution[1] + j - 0;
        userface[6 * cell + 4] = (i - 1) * resolution[1] + j - 0;
        userface[6 * cell + 5] = (i - 0) * resolution[1] + j - 1;
      }
    }
  }

  mjs_setFloat(spec.uservert, uservert.data(), 3 * resolution[0] * resolution[1]);
  mjs_setFloat(spec.usernormal, usernormal.data(), 9 * resolution[0] * resolution[1]);
  mjs_setInt(spec.userface, userface.data(), 6 * (resolution[0] - 1) * (resolution[1] - 1));
}


// make a mesh of a generalized discrete cone
void mjCMesh::MakeCone(int nedge, double radius) {
  int                n = 3 * (nedge + (radius > 0 ? nedge : 1));
  std::vector<float> uservert(n, 0);

  // bottom face
  for (int i = 0; i < nedge; i++) {
    uservert[3 * i + 0] = cos(2 * i * mjPI / nedge);
    uservert[3 * i + 1] = sin(2 * i * mjPI / nedge);
    uservert[3 * i + 2] = -1;
  }

  // top face or single point
  if (radius > 0) {
    for (int i = nedge; i < 2 * nedge; i++) {
      uservert[3 * i + 0] = radius * cos(2 * i * mjPI / nedge);
      uservert[3 * i + 1] = radius * sin(2 * i * mjPI / nedge);
      uservert[3 * i + 2] = 1;
    }
  } else {
    uservert[3 * nedge + 2] = 1;
  }

  mjs_setFloat(spec.uservert, uservert.data(), n);
}


// get point id from box coordinates and side
static int BoxID(int ix, int iy, int iz, const int count[3]) {
  // side iz=0
  if (iz == 0) {
    return ix * count[1] + iy + 1;
  }

  // side iz=max
  else if (iz == count[2] - 1) {
    return count[0] * count[1] + ix * count[1] + iy + 1;
  }

  // side iy=0
  else if (iy == 0) {
    return 2 * count[0] * count[1] + ix * (count[2] - 2) + iz - 1 + 1;
  }

  // side iy=max
  else if (iy == count[1] - 1) {
    return 2 * count[0] * count[1] + count[0] * (count[2] - 2) + ix * (count[2] - 2) + iz - 1 + 1;
  }

  // side ix=0
  else if (ix == 0) {
    return 2 * count[0] * count[1] +
           2 * count[0] * (count[2] - 2) +
           (iy - 1) * (count[2] - 2) +
           iz -
           1 +
           1;
  }

  // side ix=max
  else {
    return 2 * count[0] * count[1] +
           2 * count[0] * (count[2] - 2) +
           (count[1] - 2) * (count[2] - 2) +
           (iy - 1) * (count[2] - 2) +
           iz -
           1 +
           1;
  }
}


// shape onto which the surface of a box is projected
enum BoxShape { kBoxShapeBox, kBoxShapeCylinder, kBoxShapeEllipsoid };


// project from box to other shape
static void BoxProject(double*      pos,
                       int          ix,
                       int          iy,
                       int          iz,
                       const int    count[3],
                       const double spacing[3],
                       BoxShape     shape) {
  // init point
  pos[0] = 2.0 * ix / (count[0] - 1) - 1;
  pos[1] = 2.0 * iy / (count[1] - 1) - 1;
  pos[2] = 2.0 * iz / (count[2] - 1) - 1;

  // determine sizes
  double size[3] = {0.5 * spacing[0] * (count[0] - 1),
                    0.5 * spacing[1] * (count[1] - 1),
                    0.5 * spacing[2] * (count[2] - 1)};

  // box
  if (shape == kBoxShapeBox) {
    pos[0] *= size[0];
    pos[1] *= size[1];
    pos[2] *= size[2];
  }

  // cylinder
  else if (shape == kBoxShapeCylinder) {
    double L0 = std::max(std::abs(pos[0]), std::abs(pos[1]));
    mjuu_normvec(pos, 2);
    pos[0] *= size[0] * L0;
    pos[1] *= size[1] * L0;
    pos[2] *= size[2];
  }

  // ellipsoid
  else if (shape == kBoxShapeEllipsoid) {
    mjuu_normvec(pos, 3);
    pos[0] *= size[0];
    pos[1] *= size[1];
    pos[2] *= size[2];
  }
}


static int mat2lin(int ix, int iy, int iz, const int count[3]) {
  return ix * count[1] * count[2] + iy * count[2] + iz;
}


// generate grid points, elements and texcoords:
// 1D points only, 2D triangles or 3D tetrahedra
static void GridData(std::vector<double>& points,
                     std::vector<int>&    elements,
                     std::vector<float>&  texcoords,
                     const int            count[3],
                     const double         spacing[3],
                     int                  dim,
                     bool                 disc,
                     bool                 needtex) {
  // 1D: points only, connectivity is implicit
  if (dim == 1) {
    for (int ix = 0; ix < count[0]; ix++) {
      points.push_back(spacing[0] * (ix - 0.5 * (count[0] - 1)));
      points.push_back(0);
      points.push_back(0);
    }
  }

  // 2D
  else if (dim == 2) {
    for (int ix = 0; ix < count[0]; ix++) {
      for (int iy = 0; iy < count[1]; iy++) {
        int quad2tri[2][3] = {
            {0, 1, 2},
            {0, 2, 3}
        };

        // add point
        double pos[2] = {spacing[0] * (ix - 0.5 * (count[0] - 1)),
                         spacing[1] * (iy - 0.5 * (count[1] - 1))};
        points.push_back(pos[0]);
        points.push_back(pos[1]);
        points.push_back(0);

        // add texture coordinates, if not specified explicitly
        if (needtex) {
          texcoords.push_back(ix / (double)std::max(count[0] - 1, 1));
          texcoords.push_back(iy / (double)std::max(count[1] - 1, 1));
        }

        // flip triangles if radial projection is requested
        if (((pos[0] < -mjEPS && pos[1] > -mjEPS) || (pos[0] > -mjEPS && pos[1] < -mjEPS)) &&
            disc) {
          quad2tri[0][2] = 3;
          quad2tri[1][0] = 1;
        }

        // add elements
        if (ix < count[0] - 1 && iy < count[1] - 1) {
          int vert[4] = {
              count[2] * count[1] * (ix + 0) + count[2] * (iy + 0),
              count[2] * count[1] * (ix + 1) + count[2] * (iy + 0),
              count[2] * count[1] * (ix + 1) + count[2] * (iy + 1),
              count[2] * count[1] * (ix + 0) + count[2] * (iy + 1),
          };
          for (int s = 0; s < 2; s++) {
            for (int v = 0; v < 3; v++) { elements.push_back(vert[quad2tri[s][v]]); }
          }
        }
      }
    }
  }

  // 3D
  else {
    int cube2tets[6][4] = {
        {0, 3, 1, 7},
        {0, 1, 4, 7},
        {1, 3, 2, 7},
        {1, 2, 6, 7},
        {1, 5, 4, 7},
        {1, 6, 5, 7}
    };
    for (int ix = 0; ix < count[0]; ix++) {
      for (int iy = 0; iy < count[1]; iy++) {
        for (int iz = 0; iz < count[2]; iz++) {
          // add point
          points.push_back(spacing[0] * (ix - 0.5 * (count[0] - 1)));
          points.push_back(spacing[1] * (iy - 0.5 * (count[1] - 1)));
          points.push_back(spacing[2] * (iz - 0.5 * (count[2] - 1)));

          // add texture coordinates, if not specified explicitly
          if (needtex) {
            texcoords.push_back(ix / (float)std::max(count[0] - 1, 1));
            texcoords.push_back(iy / (float)std::max(count[1] - 1, 1));
          }

          // add elements
          if (ix < count[0] - 1 && iy < count[1] - 1 && iz < count[2] - 1) {
            int vert[8] = {
                count[2] * count[1] * (ix + 0) + count[2] * (iy + 0) + iz + 0,
                count[2] * count[1] * (ix + 1) + count[2] * (iy + 0) + iz + 0,
                count[2] * count[1] * (ix + 1) + count[2] * (iy + 1) + iz + 0,
                count[2] * count[1] * (ix + 0) + count[2] * (iy + 1) + iz + 0,
                count[2] * count[1] * (ix + 0) + count[2] * (iy + 0) + iz + 1,
                count[2] * count[1] * (ix + 1) + count[2] * (iy + 0) + iz + 1,
                count[2] * count[1] * (ix + 1) + count[2] * (iy + 1) + iz + 1,
                count[2] * count[1] * (ix + 0) + count[2] * (iy + 1) + iz + 1,
            };
            for (int s = 0; s < 6; s++) {
              for (int v = 0; v < 4; v++) { elements.push_back(vert[cube2tets[s][v]]); }
            }
          }
        }
      }
    }
  }
}


// save procedurally generated points, elements and texcoords to a mesh spec
static void SetProcedural(mjsMesh*                   spec,
                          const std::vector<double>& points,
                          const std::vector<int>&    elements,
                          const std::vector<float>&  texcoords,
                          int                        dim) {
  // 1D meshes only have nodes, uservert is needed by surfaces and volumes used by geoms
  if (dim > 1) {
    std::vector<float> points_f(points.begin(), points.end());
    mjs_setFloat(spec->uservert, points_f.data(), points_f.size());
  }
  mjs_setDouble(spec->usernode, points.data(), points.size());
  if (dim == 3) {
    mjs_setInt(spec->usertet, elements.data(), elements.size());
  } else if (dim == 2) {
    mjs_setInt(spec->userface, elements.data(), elements.size());
  }
  if (!texcoords.empty()) { mjs_setFloat(spec->usertexcoord, texcoords.data(), texcoords.size()); }
}


// make a line of nodes (dim=1), a 2D grid of triangles (dim=2) or a 3D grid of
// tetrahedra (dim=3); 1D grids only have nodes since their connectivity is implicit
void mjCMesh::MakeGrid(const int count[3], const double spacing[3], int dim, bool needtex) {
  std::vector<double> points;
  std::vector<int>    elements;
  std::vector<float>  texcoords;
  GridData(points, elements, texcoords, count, spacing, dim, /*disc=*/false, needtex);
  SetProcedural(&spec, points, elements, texcoords, dim);
}


// make a closed circle of count[0]-1 nodes with spacing[0] between consecutive
// nodes; only nodes are generated since the connectivity is implicit; at least
// three nodes (count[0] >= 4) are needed, otherwise nothing is generated
void mjCMesh::MakeCircle(const int count[3], const double spacing[3]) {
  if (count[0] < 4) { return; }
  std::vector<double> points;
  double              theta  = 2 * mjPI / (count[0] - 1);
  double              radius = spacing[0] / std::sin(theta / 2) / 2;
  for (int ix = 0; ix < count[0] - 1; ix++) {
    points.push_back(radius * std::cos(theta * ix));
    points.push_back(radius * std::sin(theta * ix));
    points.push_back(0);
  }
  SetProcedural(&spec, points, {}, {}, 1);
}


// make a 2D disc of triangles by radial projection of a square grid
void mjCMesh::MakeDisc(const int count[3], const double spacing[3], bool needtex) {
  std::vector<double> points;
  std::vector<int>    elements;
  std::vector<float>  texcoords;
  GridData(points, elements, texcoords, count, spacing, 2, /*disc=*/true, needtex);

  // radial projection
  double size[2] = {
      0.5 * spacing[0] * (count[0] - 1),
      0.5 * spacing[1] * (count[1] - 1),
  };
  for (int i = 0; i < points.size() / 3; i++) {
    double* pos = points.data() + 3 * i;
    double  L0  = std::max(std::abs(pos[0]), std::abs(pos[1]));
    mjuu_normvec(pos, 2);
    pos[0] *= size[0] * L0;
    pos[1] *= size[1] * L0;
  }

  SetProcedural(&spec, points, elements, texcoords, 2);
}


// generate points, elements and texcoords of a box projected onto a shape:
// 3D tetrahedra connected to the center (dim=3) or surface triangles (dim=2)
static void BoxData(std::vector<double>& points,
                    std::vector<int>&    elements,
                    std::vector<float>&  texcoords,
                    const int            count[3],
                    const double         spacing[3],
                    int                  dim,
                    bool                 needtex,
                    bool                 open,
                    BoxShape             shape) {
  double pos[3];

  // at least two points per side are needed
  if (count[0] < 2 || count[1] < 2 || count[2] < 2) { return; }

  // add center point and its texture coordinates
  if (dim == 3) {
    points.push_back(0);
    points.push_back(0);
    points.push_back(0);
    if (needtex) {
      texcoords.push_back(0);
      texcoords.push_back(0);
    }
  }

  // add points
  int              n = 0;
  std::vector<int> idx(count[0] * count[1] * count[2]);

  // iz=0/max
  for (int iz = 0; iz < count[2]; iz += count[2] - 1) {
    for (int ix = 0; ix < count[0]; ix++) {
      for (int iy = 0; iy < count[1]; iy++) {
        if (open && dim == 2 && iz != 0) { continue; }

        // add point
        BoxProject(pos, ix, iy, iz, count, spacing, shape);
        points.push_back(pos[0]);
        points.push_back(pos[1]);
        points.push_back(pos[2]);
        idx[mat2lin(ix, iy, iz, count)] = n++;

        // add texture coordinates, if not specified explicitly
        if (needtex) {
          texcoords.push_back(ix / (float)std::max(count[0] - 1, 1));
          texcoords.push_back(iy / (float)std::max(count[1] - 1, 1));
        }
      }
    }
  }

  // iy=0/max
  for (int iy = 0; iy < count[1]; iy += count[1] - 1) {
    for (int ix = 0; ix < count[0]; ix++) {
      for (int iz = 0; iz < count[2]; iz++) {
        // add point
        if (iz > 0 && ((open && dim == 2) || (iz < count[2] - 1))) {
          BoxProject(pos, ix, iy, iz, count, spacing, shape);
          points.push_back(pos[0]);
          points.push_back(pos[1]);
          points.push_back(pos[2]);
          idx[mat2lin(ix, iy, iz, count)] = n++;

          // add texture coordinates
          if (needtex) {
            texcoords.push_back(ix / (float)std::max(count[0] - 1, 1));
            texcoords.push_back(iz / (float)std::max(count[2] - 1, 1));
          }
        }
      }
    }
  }

  // ix=0/max
  for (int ix = 0; ix < count[0]; ix += count[0] - 1) {
    for (int iy = 0; iy < count[1]; iy++) {
      for (int iz = 0; iz < count[2]; iz++) {
        // add point
        if (iz > 0 && ((open && dim == 2) || (iz < count[2] - 1)) && iy > 0 && iy < count[1] - 1) {
          BoxProject(pos, ix, iy, iz, count, spacing, shape);
          points.push_back(pos[0]);
          points.push_back(pos[1]);
          points.push_back(pos[2]);
          idx[mat2lin(ix, iy, iz, count)] = n++;

          // add texture coordinates
          if (needtex) {
            texcoords.push_back(iy / (float)std::max(count[1] - 1, 1));
            texcoords.push_back(iz / (float)std::max(count[2] - 1, 1));
          }
        }
      }
    }
  }

  // add elements

  // iz=0/max
  for (int iz = 0; iz < count[2]; iz += count[2] - 1) {
    for (int ix = 0; ix < count[0]; ix++) {
      for (int iy = 0; iy < count[1]; iy++) {
        if (open && dim == 2 && iz != 0) { continue; }

        if (ix < count[0] - 1 && iy < count[1] - 1) {
          if (dim == 3) {
            elements.push_back(0);
            elements.push_back(BoxID(ix, iy, iz, count));
            elements.push_back(BoxID(ix + 1, iy, iz, count));
            elements.push_back(BoxID(ix + 1, iy + 1, iz, count));

            elements.push_back(0);
            elements.push_back(BoxID(ix, iy, iz, count));
            elements.push_back(BoxID(ix, iy + 1, iz, count));
            elements.push_back(BoxID(ix + 1, iy + 1, iz, count));
          } else {
            int step1 = iz == 0 ? 1 : 0;
            int step2 = iz == 0 ? 0 : 1;
            elements.push_back(idx[mat2lin(ix, iy, iz, count)]);
            elements.push_back(idx[mat2lin(ix + 1, iy + step1, iz, count)]);
            elements.push_back(idx[mat2lin(ix + 1, iy + step2, iz, count)]);

            elements.push_back(idx[mat2lin(ix, iy, iz, count)]);
            elements.push_back(idx[mat2lin(ix + step2, iy + 1, iz, count)]);
            elements.push_back(idx[mat2lin(ix + step1, iy + 1, iz, count)]);
          }
        }
      }
    }
  }

  // iy=0/max
  for (int iy = 0; iy < count[1]; iy += count[1] - 1) {
    for (int ix = 0; ix < count[0]; ix++) {
      for (int iz = 0; iz < count[2]; iz++) {
        if (ix < count[0] - 1 && iz < count[2] - 1) {
          if (dim == 3) {
            elements.push_back(0);
            elements.push_back(BoxID(ix, iy, iz, count));
            elements.push_back(BoxID(ix + 1, iy, iz, count));
            elements.push_back(BoxID(ix + 1, iy, iz + 1, count));

            elements.push_back(0);
            elements.push_back(BoxID(ix, iy, iz, count));
            elements.push_back(BoxID(ix, iy, iz + 1, count));
            elements.push_back(BoxID(ix + 1, iy, iz + 1, count));
          } else {
            int ix0 = iy == 0 ? ix : ix + 1;
            int dx  = iy == 0 ? 1 : -1;
            elements.push_back(idx[mat2lin(ix0, iy, iz, count)]);
            elements.push_back(idx[mat2lin(ix0 + dx, iy, iz, count)]);
            elements.push_back(idx[mat2lin(ix0 + dx, iy, iz + 1, count)]);

            elements.push_back(idx[mat2lin(ix0, iy, iz, count)]);
            elements.push_back(idx[mat2lin(ix0 + dx, iy, iz + 1, count)]);
            elements.push_back(idx[mat2lin(ix0, iy, iz + 1, count)]);
          }
        }
      }
    }
  }

  // ix=0/max
  for (int ix = 0; ix < count[0]; ix += count[0] - 1) {
    for (int iy = 0; iy < count[1]; iy++) {
      for (int iz = 0; iz < count[2]; iz++) {
        if (iy < count[1] - 1 && iz < count[2] - 1) {
          if (dim == 3) {
            elements.push_back(0);
            elements.push_back(BoxID(ix, iy, iz, count));
            elements.push_back(BoxID(ix, iy + 1, iz, count));
            elements.push_back(BoxID(ix, iy + 1, iz + 1, count));

            elements.push_back(0);
            elements.push_back(BoxID(ix, iy, iz, count));
            elements.push_back(BoxID(ix, iy, iz + 1, count));
            elements.push_back(BoxID(ix, iy + 1, iz + 1, count));
          } else {
            int iy0 = ix != 0 ? iy : iy + 1;
            int dy  = ix != 0 ? 1 : -1;
            elements.push_back(idx[mat2lin(ix, iy0, iz, count)]);
            elements.push_back(idx[mat2lin(ix, iy0 + dy, iz, count)]);
            elements.push_back(idx[mat2lin(ix, iy0 + dy, iz + 1, count)]);

            elements.push_back(idx[mat2lin(ix, iy0, iz, count)]);
            elements.push_back(idx[mat2lin(ix, iy0 + dy, iz + 1, count)]);
            elements.push_back(idx[mat2lin(ix, iy0, iz + 1, count)]);
          }
        }
      }
    }
  }
}


// make a box, cylinder or ellipsoid mesh
static void MakeBoxShape(mjsMesh*     spec,
                         const int    count[3],
                         const double spacing[3],
                         int          dim,
                         bool         needtex,
                         bool         open,
                         BoxShape     shape) {
  std::vector<double> points;
  std::vector<int>    elements;
  std::vector<float>  texcoords;
  BoxData(points, elements, texcoords, count, spacing, dim, needtex, open, shape);
  SetProcedural(spec, points, elements, texcoords, dim);
}


// make a box: 3D tetrahedra (dim=3) or surface triangles (dim=2)
void mjCMesh::MakeBox(
    const int count[3], const double spacing[3], int dim, bool needtex, bool open) {
  MakeBoxShape(&spec, count, spacing, dim, needtex, open, kBoxShapeBox);
}


// make a cylinder: 3D tetrahedra (dim=3) or surface triangles (dim=2)
void mjCMesh::MakeCylinder(
    const int count[3], const double spacing[3], int dim, bool needtex, bool open) {
  MakeBoxShape(&spec, count, spacing, dim, needtex, open, kBoxShapeCylinder);
}


// make an ellipsoid: 3D tetrahedra (dim=3) or surface triangles (dim=2)
void mjCMesh::MakeEllipsoid(
    const int count[3], const double spacing[3], int dim, bool needtex, bool open) {
  MakeBoxShape(&spec, count, spacing, dim, needtex, open, kBoxShapeEllipsoid);
}


// compute vertex normals
void mjCMesh::MakeNormal(const double* dvert) {
  // only if normal data is missing
  if (!normal_.empty()) { return; }

  // allocate and clear normals
  normal_.assign(3 * nvert(), 0);

  if (facenormal_.empty()) { facenormal_.assign(3 * nface(), 0); }

  // loop over faces, accumulate vertex normals
  for (int i = 0; i < nface(); i++) {
    // get vertex ids
    int vertid[3];
    for (int j = 0; j < 3; j++) { vertid[j] = face_[3 * i + j]; }

    // get triangle edges
    double vec01[3], vec02[3];
    for (int j = 0; j < 3; j++) {
      vec01[j] = dvert[3 * vertid[1] + j] - dvert[3 * vertid[0] + j];
      vec02[j] = dvert[3 * vertid[2] + j] - dvert[3 * vertid[0] + j];
    }

    // compute face normal
    double nrm[3];
    mjuu_crossvec(nrm, vec01, vec02);
    double area = mjuu_normvec(nrm, 3);

    // add normal to each vertex with weight = area
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) { normal_[3 * vertid[j] + k] += nrm[k] * area; }
      facenormal_[3 * i + j] = vertid[j];
    }
  }

  // remove large-angle faces
  if (!smoothnormal) {
    // allocate removal and clear
    float* nremove = (float*)mju_malloc(3 * nnormal() * sizeof(float));
    memset(nremove, 0, 3 * nnormal() * sizeof(float));

    // remove contributions from faces at large angles with vertex normal
    for (int i = 0; i < nface(); i++) {
      // get vertex ids
      int vertid[3];
      for (int j = 0; j < 3; j++) { vertid[j] = face_[3 * i + j]; }

      // get triangle edges
      double vec01[3], vec02[3];
      for (int j = 0; j < 3; j++) {
        vec01[j] = dvert[3 * vertid[1] + j] - dvert[3 * vertid[0] + j];
        vec02[j] = dvert[3 * vertid[2] + j] - dvert[3 * vertid[0] + j];
      }

      // compute face normal
      double nrm[3];
      mjuu_crossvec(nrm, vec01, vec02);
      double area = mjuu_normvec(nrm, 3);

      // compare to vertex normal, subtract contribution if dot product too small
      for (int j = 0; j < 3; j++) {
        // normalized vertex normal
        double vnrm[3] = {normal_[3 * vertid[j]],
                          normal_[3 * vertid[j] + 1],
                          normal_[3 * vertid[j] + 2]};
        mjuu_normvec(vnrm, 3);

        // dot too small: remove
        if (mjuu_dot3(nrm, vnrm) < 0.8) {
          for (int k = 0; k < 3; k++) { nremove[3 * vertid[j] + k] += nrm[k] * area; }
        }
      }
    }

    // apply removal, free nremove
    for (int i = 0; i < 3 * nnormal(); i++) { normal_[i] -= nremove[i]; }
    mju_free(nremove);
  }

  // normalize normals
  for (int i = 0; i < nnormal(); i++) {
    // compute length
    float len = sqrtf(normal_[3 * i] * normal_[3 * i] +
                      normal_[3 * i + 1] * normal_[3 * i + 1] +
                      normal_[3 * i + 2] * normal_[3 * i + 2]);

    // divide by length
    if (len > mjMINVAL) {
      for (int j = 0; j < 3; j++) { normal_[3 * i + j] /= len; }
    } else {
      normal_[3 * i] = normal_[3 * i + 1] = 0;
      normal_[3 * i + 2]                  = 1;
    }
  }
}


// compute face circumradii
void mjCMesh::MakeCenter(const double* dvert) {
  if (center_) { return; }

  // allocate and clear
  center_ = (double*)mju_malloc(3 * nface() * sizeof(double));
  memset(center_, 0, 3 * nface() * sizeof(double));

  for (int i = 0; i < nface(); i++) {
    // get vertex ids
    int* vertid = face_.data() + 3 * i;

    // get triangle edges
    double a[3], b[3];
    for (int j = 0; j < 3; j++) {
      a[j] = dvert[3 * vertid[0] + j] - dvert[3 * vertid[2] + j];
      b[j] = dvert[3 * vertid[1] + j] - dvert[3 * vertid[2] + j];
    }

    // compute face normal
    double nrm[3];
    mjuu_crossvec(nrm, a, b);

    // compute circumradius
    double norm_a_2 = mjuu_dot3(a, a);
    double norm_b_2 = mjuu_dot3(b, b);
    double area     = sqrt(mjuu_dot3(nrm, nrm));

    // compute circumcenter
    double res[3], vec[3] = {norm_a_2 * b[0] - norm_b_2 * a[0],
                             norm_a_2 * b[1] - norm_b_2 * a[1],
                             norm_a_2 * b[2] - norm_b_2 * a[2]};
    mjuu_crossvec(res, vec, nrm);
    center_[3 * i + 0] = res[0] / (2 * area * area) + dvert[3 * vertid[2] + 0];
    center_[3 * i + 1] = res[1] / (2 * area * area) + dvert[3 * vertid[2] + 1];
    center_[3 * i + 2] = res[2] / (2 * area * area) + dvert[3 * vertid[2] + 2];
  }
}


// compute the normals of the polygons
void mjCMesh::MakePolygonNormals(const double* dvert) {
  for (int i = 0; i < polygons_.size(); ++i) {
    double n[3]  = {0, 0, 0};
    int    nvert = polygons_[i].size();

    // compute normal using Newell's method
    for (int j = 0; j < nvert; ++j) {
      const double* v0  = &dvert[3 * polygons_[i][j]];
      const double* v1  = &dvert[3 * polygons_[i][(j + 1) % nvert]];
      n[0]             += (v0[1] - v1[1]) * (v0[2] + v1[2]);
      n[1]             += (v0[2] - v1[2]) * (v0[0] + v1[0]);
      n[2]             += (v0[0] - v1[0]) * (v0[1] + v1[1]);
    }
    double nrm = std::sqrt(mjuu_dot3(n, n));
    if (nrm < mjEPS) {
      polygon_normals_[3 * i + 0] = 1;
      polygon_normals_[3 * i + 1] = 0;
      polygon_normals_[3 * i + 2] = 0;
    } else {
      polygon_normals_[3 * i + 0] = n[0] / nrm;
      polygon_normals_[3 * i + 1] = n[1] / nrm;
      polygon_normals_[3 * i + 2] = n[2] / nrm;
    }
  }
}


// helper class to compute the polygons of a mesh
class MeshPolygon {
 public:
  // constructors (need starting face)
  MeshPolygon(const double v1[3],
              const double v2[3],
              const double v3[3],
              int          v1i,
              int          v2i,
              int          v3i,
              double       theta,
              double       phi);
  MeshPolygon()                              = delete;
  MeshPolygon(const MeshPolygon&)            = delete;
  MeshPolygon& operator=(const MeshPolygon&) = delete;
  MeshPolygon(MeshPolygon&&)                 = default;
  MeshPolygon& operator=(MeshPolygon&&)      = default;

  void InsertFace(int v1, int v2, int v3);           // insert a face into the polygon
  std::vector<std::vector<int>> Paths() const;       // return trace of the polygons
  const double* Normal() const { return normal_; }   // return the normal of the polygon
  double Normal(int i) const { return normal_[i]; }  // return the i-th component of the normal

 private:
  std::vector<std::pair<int, int>> edges_;

  // inserted faces do not necessarily share edges with the current polygon, so they're grouped as
  // islands until they can be combined with later face insertions
  std::vector<int> islands_;
  int              nisland_   = 0;
  double           normal_[3] = {0.0, 0.0, 0.0};
  void             CombineIslands(int& island1, int& island2);
};

bool MeshPolygonKey(std::pair<double, double>& angles,
                    const double               v1[3],
                    const double               v2[3],
                    const double               v3[3],
                    double                     angle_tol) {
  double diff12[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  double diff13[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
  double normal[3], norm;

  mjuu_crossvec(normal, diff12, diff13);
  if ((norm = std::sqrt(mjuu_dot3(normal, normal))) < mjMINVAL) { return false; }

  // atan2 is sensitive to sign of 0.0, adding 0.0 to enforcing only positive 0.0
  normal[0]     = (normal[0] / norm) + 0.0;
  normal[1]     = (normal[1] / norm) + 0.0;
  normal[2]     = (normal[2] / norm) + 0.0;
  double rtheta = 0.0, rphi = 0.0;

  // clamp normal to be in valid range for acos
  if (std::abs(normal[2]) > 1.0 - 1e-7) {
    if (normal[2] < 0) rphi = std::round(mjPI / angle_tol);
    angles = std::make_pair(rtheta, rphi);
    return true;
  }
  // rounded azimuthal and polar angles
  rtheta = std::round(std::atan2(normal[1], normal[0]) / angle_tol);
  rphi   = std::round(std::acos(normal[2]) / angle_tol);
  angles = std::make_pair(rtheta, rphi);
  return true;
}


MeshPolygon::MeshPolygon(const double v1[3],
                         const double v2[3],
                         const double v3[3],
                         int          v1i,
                         int          v2i,
                         int          v3i,
                         double       theta,
                         double       phi) {
  normal_[0] = std::cos(theta) * std::sin(phi);
  normal_[1] = std::sin(theta) * std::sin(phi);
  normal_[2] = std::cos(phi);

  edges_ = {
      {v1i, v2i},
      {v2i, v3i},
      {v3i, v1i}
  };
  nisland_ = 1;
  islands_ = {0, 0, 0};
}


// combine two islands when a newly inserted face connects them
void MeshPolygon::CombineIslands(int& island1, int& island2) {
  // pick the smaller island
  if (island2 < island1) {
    int tmp = island1;
    island1 = island2;
    island2 = tmp;
  }

  // renumber the islands
  for (int k = 0; k < islands_.size(); ++k) {
    if (islands_[k] == island2) {
      islands_[k] = island1;
    } else if (islands_[k] > island2) {
      islands_[k]--;
    }
  }
}


// insert a triangular face into the polygon
void MeshPolygon::InsertFace(int v1, int v2, int v3) {
  int add1 = 1, add2 = 1, add3 = 1;
  int island = -1;

  // check if face can be attached via edge v1v2
  for (int i = 0; i < edges_.size(); ++i) {
    if (edges_[i].first == v2 && edges_[i].second == v1) {
      add1   = 0;
      island = islands_[i];
      edges_.erase(edges_.begin() + i);
      islands_.erase(islands_.begin() + i);
      break;
    }
  }

  // check if face can be attached via edge v2v3
  for (int i = 0; i < edges_.size(); ++i) {
    if (edges_[i].first == v3 && edges_[i].second == v2) {
      int island2 = islands_[i];
      if (island == -1) {
        island = island2;
      } else if (island2 != island) {
        nisland_--;
        CombineIslands(island, island2);
      }
      add2 = 0;
      edges_.erase(edges_.begin() + i);
      islands_.erase(islands_.begin() + i);
      break;
    }
  }

  // check if face can be attached via edge v3v1
  for (int i = 0; i < edges_.size(); ++i) {
    if (edges_[i].first == v1 && edges_[i].second == v3) {
      int island3 = islands_[i];
      if (island == -1) {
        island = island3;
      } else if (island3 != island) {
        nisland_--;
        CombineIslands(island, island3);
      }
      add3 = 0;
      edges_.erase(edges_.begin() + i);
      islands_.erase(islands_.begin() + i);
      break;
    }
  }

  if (island == -1) { island = nisland_++; }

  // add only new edges to the polygon

  if (add1) {
    edges_.push_back({v1, v2});
    islands_.push_back(island);
  }
  if (add2) {
    edges_.push_back({v2, v3});
    islands_.push_back(island);
  }
  if (add3) {
    edges_.push_back({v3, v1});
    islands_.push_back(island);
  }
}


// return the transverse vertices of the polygon, multiple paths possible if not connected
std::vector<std::vector<int>> MeshPolygon::Paths() const {
  std::vector<std::vector<int>> paths;

  // go through each connected component of the polygon
  for (int i = 0; i < nisland_; ++i) {
    std::vector<int> path;

    // find starting vertex
    for (int j = 0; j < edges_.size(); ++j) {
      if (islands_[j] == i) {
        path.push_back(edges_[j].first);
        path.push_back(edges_[j].second);
        break;
      }
    }

    // SHOULD NOT OCCUR (See logic in MeshPolygon::CombineIslands)
    if (path.empty()) { continue; }

    // visit the next vertex given the current edge
    int next = path.back();
    for (int l = 0; l < edges_.size(); ++l) {
      int finished = 0;
      for (int k = 1; k < edges_.size(); ++k) {
        if (islands_[k] == i && edges_[k].first == next) {
          next = edges_[k].second;
          if (next == path[0]) {
            paths.push_back(path);
            finished = 1;
            break;
          }
          path.push_back(next);
          break;
        }
      }

      // back at start
      if (finished) { break; }
    }
  }
  return paths;
}


// merge coplanar mesh triangular faces into polygonal sides to represent the geometry of the mesh
void mjCMesh::MakePolygons(const double* dvert) {
  constexpr double                                                     kAngleTol = 0.01;
  std::unordered_map<std::pair<double, double>, MeshPolygon, PairHash> mesh_polygons;
  polygons_.clear();
  polygon_normals_.clear();
  polygon_map_.clear();
  polygon_map_.resize(nvert());

  // we need a convex mesh, so we use graph faces
  int* faces  = GraphFaces();
  int  nfaces = graph_[1];

  // process each face
  for (int i = 0; i < nfaces; i++) {
    int           vi1 = faces[3 * i + 0];
    int           vi2 = faces[3 * i + 1];
    int           vi3 = faces[3 * i + 2];
    const double* v1  = &dvert[3 * vi1];
    const double* v2  = &dvert[3 * vi2];
    const double* v3  = &dvert[3 * vi3];

    std::pair<double, double> key;
    if (!MeshPolygonKey(key, v1, v2, v3, kAngleTol)) { continue; }
    auto it = mesh_polygons.find(key);
    if (it == mesh_polygons.end()) {
      double theta = kAngleTol * key.first;
      double phi   = kAngleTol * key.second;
      mesh_polygons.emplace(key, MeshPolygon(v1, v2, v3, vi1, vi2, vi3, theta, phi));
    } else {
      it->second.InsertFace(vi1, vi2, vi3);
    }
  }

  for (const auto& pair : mesh_polygons) {
    const MeshPolygon&            polygon = pair.second;
    std::vector<std::vector<int>> paths   = polygon.Paths();

    // separate the polygons if they were grouped together
    for (const auto& path : paths) {
      if (path.size() < 3) continue;
      polygons_.push_back(path);
      polygon_normals_.push_back(polygon.Normal(0));
      polygon_normals_.push_back(polygon.Normal(1));
      polygon_normals_.push_back(polygon.Normal(2));
    }
  }

  // populate the polygon map
  for (int i = 0; i < polygons_.size(); i++) {
    for (int j = 0; j < polygons_[i].size(); ++j) { polygon_map_[polygons_[i][j]].push_back(i); }
  }
}


//------------------ class mjCSkin implementation --------------------------------------------------

// constructor
mjCSkin::mjCSkin(mjCModel* _model) {
  mjs_defaultSkin(&spec);
  elemtype = mjOBJ_SKIN;

  // set model pointer
  model = _model;
  if (model) compiler = &model->spec.compiler;

  // clear data
  spec_file_.clear();
  spec_material_.clear();
  spec_vert_.clear();
  spec_texcoord_.clear();
  spec_face_.clear();
  spec_bodyname_.clear();
  spec_bindpos_.clear();
  spec_bindquat_.clear();
  spec_vertid_.clear();
  spec_vertweight_.clear();

  bodyid.clear();
  matid = -1;

  // point to local
  PointToLocal();

  // in case this camera is not compiled
  CopyFromSpec();
}


mjCSkin::mjCSkin(const mjCSkin& other) {
  *this = other;
}


mjCSkin& mjCSkin::operator=(const mjCSkin& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCSkin_*>(this) = static_cast<const mjCSkin_&>(other);
    *static_cast<mjsSkin*>(this)  = static_cast<const mjsSkin&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCSkin::PointToLocal() {
  spec.element    = static_cast<mjsElement*>(this);
  spec.file       = &spec_file_;
  spec.material   = &spec_material_;
  spec.vert       = &spec_vert_;
  spec.texcoord   = &spec_texcoord_;
  spec.face       = &spec_face_;
  spec.bodyname   = &spec_bodyname_;
  spec.bindpos    = &spec_bindpos_;
  spec.bindquat   = &spec_bindquat_;
  spec.vertid     = &spec_vertid_;
  spec.vertweight = &spec_vertweight_;
  spec.info       = &info;
  file            = nullptr;
  material        = nullptr;
  vert            = nullptr;
  texcoord        = nullptr;
  face            = nullptr;
  bodyname        = nullptr;
  bindpos         = nullptr;
  bindquat        = nullptr;
  vertid          = nullptr;
  vertweight      = nullptr;
}


void mjCSkin::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  for (auto& name : spec_bodyname_) { name = m->prefix + name + m->suffix; }
  bone_prefix_ = m->prefix + bone_prefix_;
  bone_suffix_ = bone_suffix_ + m->suffix;
  if (!spec_material_.empty() && model != m) {
    spec_material_ = m->prefix + spec_material_ + m->suffix;
  }
}


void mjCSkin::CopyFromSpec() {
  *static_cast<mjsSkin*>(this) = spec;
  file_                        = spec_file_;
  material_                    = spec_material_;
  vert_                        = spec_vert_;
  texcoord_                    = spec_texcoord_;
  face_                        = spec_face_;
  bodyname_                    = spec_bodyname_;
  bindpos_                     = spec_bindpos_;
  bindquat_                    = spec_bindquat_;
  vertid_                      = spec_vertid_;
  vertweight_                  = spec_vertweight_;
}


// destructor
mjCSkin::~mjCSkin() {
  spec_file_.clear();
  spec_material_.clear();
  spec_vert_.clear();
  spec_texcoord_.clear();
  spec_face_.clear();
  spec_bodyname_.clear();
  spec_bindpos_.clear();
  spec_bindquat_.clear();
  spec_vertid_.clear();
  spec_vertweight_.clear();
  bodyid.clear();
}


// file as opened by Compile, which has already applied strippath to file_
std::string mjCSkin::ResolvedFilePath() const {
  return ResolvePath(mjs_getString(compiler->meshdir), file_);
}


void mjCSkin::ResolveReferences(const mjCModel* m) {
  size_t nbone = bodyname_.size();
  bodyid.resize(nbone);
  for (int i = 0; i < nbone; i++) {
    mjCBase* pbody = m->FindObject(mjOBJ_BODY, bodyname_[i]);
    if (!pbody) { throw mjCError(this, "unknown body '%s' in skin", bodyname_[i].c_str()); }
    bodyid[i] = pbody->id;
  }
}


// scale the compiled copy by the gauge of the bones: the vertices and the bind poses are given in
// the same coordinates, in which the offsets of the vertices from the bones scale with the bones,
// whose maps, expressed there through their bind orientations, must agree
void mjCSkin::ApplyGauge() {
  scaled_ = false;
  if (bodyid.empty()) { return; }
  std::vector<mjCGauge> gauges;
  for (int i = 0; i < bodyid.size(); i++) {
    const float* q       = bindquat_.data() + 4 * i;
    double       conj[4] = {q[0], -q[1], -q[2], -q[3]};
    gauges.push_back(model->Bodies()[bodyid[i]]->ContentGauge().Rotated(conj));
  }
  const mjCGauge& g = gauges[0];
  for (const mjCGauge& b : gauges) {
    bool same = b.uniform == g.uniform && b.shape == g.shape;
    for (int k = 0; k < 9 && same && !g.uniform; k++) {
      same = std::abs(b.map[k] - g.map[k]) <= 1e-6 * g.shape;
    }
    if (!same) { throw mjCError(this, "the bones of a skin must have the same scale"); }
  }
  if (g.uniform && g.shape == 1) { return; }
  auto points = [&g](std::vector<float>& pos) {
    for (int i = 0; i < pos.size() / 3; i++) {
      double x[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
      g.Point(x);
      for (int k = 0; k < 3; k++) { pos[3 * i + k] = (float)x[k]; }
    }
  };
  points(vert_);
  points(bindpos_);
  inflate *= g.Shape(1);
  scaled_  = true;
}


// compiler
void mjCSkin::Compile(const mjVFS* vfs) {
  CopyFromSpec();

  // load file
  if (!file_.empty()) {
    // make sure data is not present
    if (!spec_vert_.empty() ||
        !spec_texcoord_.empty() ||
        !spec_face_.empty() ||
        !spec_bodyname_.empty() ||
        !spec_bindpos_.empty() ||
        !spec_bindquat_.empty() ||
        !spec_vertid_.empty() ||
        !spec_vertweight_.empty()) {
      throw mjCError(this, "Both skin data and file were specified: %s", file_.c_str());
    }

    // remove path from file if necessary
    if (model && model->spec.strippath) { file_ = mjuu_strippath(file_); }

    // load SKN
    std::string ext = mjuu_getext(file_);
    if (strcasecmp(ext.c_str(), ".skn")) {
      throw mjCError(this, "Unknown skin file type: %s", file_.c_str());
    }

    mujoco::user::FilePath meshdir_;
    meshdir_ = FilePath(mjs_getString(compiler->meshdir));

    FilePath    filename    = meshdir_ + FilePath(file_);
    mjSpec*     owning_spec = model->FindSpec(compiler);
    mjResource* resource    = LoadResource(owning_spec->modelfiledir->c_str(), filename.Str(), vfs);

    try {
      LoadSKN(resource);
      mju_closeResource(resource);
    } catch (mjCError err) {
      mju_closeResource(resource);
      throw err;
    }

    // the bones loaded from file take the namespace of the attachments
    for (auto& name : bodyname_) { name = bone_prefix_ + name + bone_suffix_; }
  }

  // make sure all data is present
  if (vert_.empty() ||
      face_.empty() ||
      bodyname_.empty() ||
      bindpos_.empty() ||
      bindquat_.empty() ||
      vertid_.empty() ||
      vertweight_.empty()) {
    throw mjCError(this, "Missing data in skin");
  }

  // check mesh sizes
  if (vert_.size() % 3) { throw mjCError(this, "Vertex data must be multiple of 3"); }
  if (!texcoord_.empty() && texcoord_.size() != 2 * vert_.size() / 3) {
    throw mjCError(this, "Vertex and texcoord data incompatible size");
  }
  if (face_.size() % 3) { throw mjCError(this, "Face data must be multiple of 3"); }
  size_t nvert = vert_.size() / 3;
  for (size_t i = 0; i < face_.size(); i++) {
    if (face_[i] < 0 || static_cast<size_t>(face_[i]) >= nvert) {
      throw mjCError(this, "face vertex index %d out of range in skin", nullptr, face_[i]);
    }
  }

  // check bone sizes
  size_t nbone = bodyname_.size();
  if (bindpos_.size() != 3 * nbone) { throw mjCError(this, "Unexpected bindpos size in skin"); }
  if (bindquat_.size() != 4 * nbone) { throw mjCError(this, "Unexpected bindquat size in skin"); }
  if (vertid_.size() != nbone) { throw mjCError(this, "Unexpected vertid size in skin"); }
  if (vertweight_.size() != nbone) { throw mjCError(this, "Unexpected vertweight size in skin"); }

  // resolve body names
  ResolveReferences(model);
  ApplyGauge();

  // resolve material name
  mjCBase* pmat = model->FindObject(mjOBJ_MATERIAL, material_);
  if (!pmat && !material_.empty()) {
    throw mjCError(this, "unknown material '%s' in skin", material_.c_str());
  }
  matid = pmat ? pmat->id : -1;

  // set total vertex weights to 0
  std::vector<float> vw;
  vw.resize(nvert);
  fill(vw.begin(), vw.end(), 0.0f);

  // accumulate vertex weights from all bones
  for (int i = 0; i < nbone; i++) {
    // make sure bone has vertices and sizes match
    size_t nbv = vertid_[i].size();
    if (vertweight_[i].size() != nbv || nbv == 0) {
      throw mjCError(this, "vertid and vertweight must have same non-zero size in skin");
    }

    // accumulate weights in global array
    for (int j = 0; j < nbv; j++) {
      // get index and check range
      int jj = vertid_[i][j];
      if (jj < 0 || jj >= nvert) {
        throw mjCError(this, "vertid %d out of range in skin", nullptr, jj);
      }

      // accumulate
      vw[jj] += vertweight_[i][j];
    }
  }

  // check coverage
  for (int i = 0; i < nvert; i++) {
    if (vw[i] <= mjMINVAL) {
      throw mjCError(this, "vertex %d must have positive total weight in skin", nullptr, i);
    }
  }

  // normalize vertex weights
  for (int i = 0; i < nbone; i++) {
    for (int j = 0; j < vertid_[i].size(); j++) { vertweight_[i][j] /= vw[vertid_[i][j]]; }
  }

  // normalize bindquat
  for (int i = 0; i < nbone; i++) {
    double quat[4] = {(double)bindquat_[4 * i],
                      (double)bindquat_[4 * i + 1],
                      (double)bindquat_[4 * i + 2],
                      (double)bindquat_[4 * i + 3]};
    mjuu_normvec(quat, 4);

    bindquat_[4 * i]     = (float)quat[0];
    bindquat_[4 * i + 1] = (float)quat[1];
    bindquat_[4 * i + 2] = (float)quat[2];
    bindquat_[4 * i + 3] = (float)quat[3];
  }
}


// load skin in SKN BIN format
void mjCSkin::LoadSKN(mjResource* resource) {
  char* buffer    = 0;
  int   buffer_sz = mju_readResource(resource, (const void**)&buffer);

  if (buffer_sz < 0) {
    throw mjCError(this, "could not read SKN file '%s'", resource->name);
  } else if (!buffer_sz) {
    throw mjCError(this, "SKN file '%s' is empty", resource->name);
  }

  // make sure header is present
  if (buffer_sz < 16) { throw mjCError(this, "missing header in SKN file '%s'", resource->name); }

  // get sizes from header
  int nvert = 0, ntexcoord = 0, nface = 0, nbone = 0;
  ReadFromBuffer(&nvert, buffer);
  ReadFromBuffer(&ntexcoord, buffer + sizeof(nvert));
  ReadFromBuffer(&nface, buffer + sizeof(nvert) + sizeof(ntexcoord));
  ReadFromBuffer(&nbone, buffer + sizeof(nvert) + sizeof(ntexcoord) + sizeof(nface));

  // negative sizes not allowed
  if (nvert < 0 || ntexcoord < 0 || nface < 0 || nbone < 0) {
    throw mjCError(this, "negative size in header of SKN file '%s'", resource->name);
  }

  if (nvert >= INT_MAX / sizeof(decltype(vert_)::value_type) / 3 ||
      ntexcoord >= INT_MAX / sizeof(decltype(texcoord_)::value_type) / 2 ||
      nface >= INT_MAX / sizeof(decltype(face_)::value_type) / 3 ||
      nbone >= INT_MAX / 18) {
    throw mjCError(this, "too large sizes in SKN file '%s'", resource->name);
  }

  // make sure we have data for vert, texcoord, face, and bone headers
  if ((int64_t)buffer_sz < 16LL +
                               3LL * nvert * sizeof(decltype(vert_)::value_type) +
                               2LL * ntexcoord * sizeof(decltype(texcoord_)::value_type) +
                               3LL * nface * sizeof(decltype(face_)::value_type) +
                               18LL * nbone * sizeof(decltype(face_)::value_type)) {
    throw mjCError(this, "insufficient data in SKN file '%s'", resource->name);
  }

  // data pointer and counter
  float* pdata = (float*)(buffer + 16);
  int    cnt   = 0;

  // copy vert
  if (nvert) {
    vert_.resize(3 * nvert);
    memcpy(vert_.data(), pdata + cnt, 3 * nvert * sizeof(decltype(vert_)::value_type));
    cnt += 3 * nvert;
  }

  // copy texcoord
  if (ntexcoord) {
    texcoord_.resize(2 * ntexcoord);
    memcpy(texcoord_.data(), pdata + cnt, 2 * ntexcoord * sizeof(decltype(texcoord_)::value_type));
    cnt += 2 * ntexcoord;
  }

  // copy face
  if (nface) {
    face_.resize(3 * nface);
    memcpy(face_.data(), pdata + cnt, 3 * nface * sizeof(decltype(face_)::value_type));
    cnt += 3 * nface;
  }

  // allocate bone arrays
  bodyname_.clear();
  bodyname_.reserve(nbone);
  bindpos_.resize(3 * nbone);
  bindquat_.resize(4 * nbone);
  vertid_.resize(nbone);
  vertweight_.resize(nbone);

  // read bones
  for (int i = 0; i < nbone; i++) {
    // check size
    if (buffer_sz / 4 - 4 - cnt < 18) {
      throw mjCError(this, "insufficient data in SKN file '%s', bone %d", resource->name, i);
    }

    // read name
    char txt[40];
    strncpy(txt, (char*)(pdata + cnt), 39);
    txt[39]  = '\0';
    cnt     += 10;
    bodyname_.push_back(txt);

    // read bindpos
    memcpy(bindpos_.data() + 3 * i, pdata + cnt, 3 * sizeof(decltype(bindpos_)::value_type));
    cnt += 3;

    // read bind quat
    memcpy(bindquat_.data() + 4 * i, pdata + cnt, 4 * sizeof(decltype(bindquat_)::value_type));
    cnt += 4;

    // read vertex count
    int vcount = 0;
    ReadFromBuffer(&vcount, reinterpret_cast<const char*>(pdata + cnt));
    cnt += 1;

    // check for negative
    if (vcount < 1) {
      throw mjCError(this,
                     "vertex count must be positive in SKN file '%s', bone %d",
                     resource->name,
                     i);
    }

    // check size
    int rem = buffer_sz / 4 - 4 - cnt;
    if (rem / 2 < vcount) {
      throw mjCError(this, "insufficient vertex data in SKN file '%s', bone %d", resource->name, i);
    }

    // read vertid
    vertid_[i].resize(vcount);
    memcpy(vertid_[i].data(),
           (int*)(pdata + cnt),
           vcount * sizeof(decltype(vertid_)::value_type::value_type));
    cnt += vcount;

    // read vertweight
    vertweight_[i].resize(vcount);
    memcpy(vertweight_[i].data(),
           (float*)(pdata + cnt),
           vcount * sizeof(decltype(vertweight_)::value_type::value_type));
    cnt += vcount;
  }

  // check final size
  if (buffer_sz != 16 + 4 * cnt) {
    throw mjCError(this, "unexpected buffer size in SKN file '%s'", resource->name);
  }
}
