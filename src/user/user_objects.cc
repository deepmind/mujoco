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

#include "user/user_objects.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lodepng.h"  // NOLINT
#include <mujoco/mjspec.h>
#include <mujoco/mjspecmacro.h>
#include <mujoco/mujoco.h>
#include "cc/array_safety.h"
#include "user/user_api.h"
#include "user/user_cache.h"
#include "user/user_model.h"
#include "user/user_resource.h"
#include "user/user_util.h"
#include "user/user_vfs.h"

namespace {
namespace mju = ::mujoco::util;
using mujoco::user::FilePath;

class PNGImage {
 public:
  static PNGImage Load(const mjCBase* obj, mjResource* resource, LodePNGColorType color_type);

  int  Width() const { return width_; }
  int  Height() const { return height_; }
  bool IsSRGB() const { return is_srgb_; }

  std::byte operator[](int i) const { return data_[i]; }

  mjByteVec&& MoveData() && { return std::move(data_); }

 private:
  std::size_t Size() const { return data_.size() + (3 * sizeof(int)); }

  LodePNGColorType color_type_;

  int       width_;
  int       height_;
  bool      is_srgb_;
  mjByteVec data_;
};

PNGImage PNGImage::Load(const mjCBase* obj, mjResource* resource, LodePNGColorType color_type) {
  PNGImage image;
  image.color_type_ = color_type;

  // open PNG resource
  const unsigned char* buffer;

  int nbuffer = mju_readResource(resource, (const void**)&buffer);

  if (nbuffer < 0) { throw mjCError(obj, "could not read PNG file '%s'", resource->name); }

  if (!nbuffer) { throw mjCError(obj, "empty PNG file '%s'", resource->name); }

  // decode PNG from buffer
  unsigned int w, h;

  lodepng::State state;
  state.info_raw.colortype = image.color_type_;
  state.info_raw.bitdepth  = 8;
  unsigned char* data_ptr  = nullptr;
  unsigned       err       = lodepng_decode(&data_ptr, &w, &h, &state, buffer, nbuffer);
  struct free_delete {
    void operator()(unsigned char* ptr) const { std::free(ptr); }
  };
  std::unique_ptr<unsigned char, free_delete> data{data_ptr};

  // check for errors
  if (err) {
    std::stringstream ss;
    ss << "error decoding PNG file '" << resource->name << "': " << lodepng_error_text(err);
    throw mjCError(obj, "%s", ss.str().c_str());
  }

  if (data) {
    size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
    image.data_.insert(image.data_.end(),
                       reinterpret_cast<std::byte*>(data.get()),
                       reinterpret_cast<std::byte*>(&data.get()[buffersize]));
  }

  image.width_   = w;
  image.height_  = h;
  image.is_srgb_ = (state.info_png.srgb_defined == 1);

  if (image.width_ <= 0 || image.height_ < 0) {
    std::stringstream ss;
    ss << "error decoding PNG file '" << resource->name << "': " << "dimensions are invalid";
    throw mjCError(obj, "%s", ss.str().c_str());
  }

  return image;
}

// associate all child list elements with a frame and copy them to parent list, clear child list;
// elements that are already in a frame stay in it
template <typename T>
void MapFrame(std::vector<T*>& parent,
              std::vector<T*>& child,
              mjCFrame*        frame,
              mjCBody*         parent_body) {
  std::for_each(child.begin(), child.end(), [frame, parent_body](T* element) {
    element->SetParent(parent_body);  // needs to happen first, SetFrame checks the parent
    if (!element->frame) { element->SetFrame(frame); }
  });
  parent.insert(parent.end(), child.begin(), child.end());
  child.clear();
}

// express a pose given in a frame in the coordinates of the body holding the frame, using the
// specs of the frame and of its ancestors, which need not be compiled
void FrameToBody(const mjCFrame* frame, double pos[3], double quat[4]) {
  for (; frame; frame = frame->frame) {
    double framequat[4];
    mjuu_copyvec(framequat, frame->spec.quat, 4);
    const char* err = ResolveOrientation(framequat,
                                         frame->compiler->degree,
                                         frame->compiler->eulerseq,
                                         frame->spec.alt);
    if (err) { throw mjCError(frame, "orientation specification error '%s' in frame", err); }
    mjuu_normvec(framequat, 4);
    mjuu_frameaccumChild(frame->spec.pos, framequat, pos, quat);
  }
}

// orientation which a spec gives, its alternative resolved; errors are reported by compilation
static void SpecQuat(double                res[4],
                     const double          quat[4],
                     const mjsOrientation& alt,
                     const mjsCompiler*    compiler) {
  mjuu_copyvec(res, quat, 4);
  if (alt.type != mjORIENTATION_QUAT) {
    ResolveOrientation(res, compiler->degree, compiler->eulerseq, alt);
  }
  mjuu_normvec(res, 4);
}

// orientation of a geom or site which its spec gives, and the type whose size rules apply: fromto
// gives the axis, and a box or ellipsoid given by fromto is round across it
static int SpecShape(double                quat[4],
                     int                   type,
                     const double          fromto[6],
                     const double          specquat[4],
                     const mjsOrientation& alt,
                     const mjsCompiler*    compiler) {
  if (!mjuu_defined(fromto[0])) {
    SpecQuat(quat, specquat, alt, compiler);
    return type;
  }
  double vec[3] = {fromto[0] - fromto[3], fromto[1] - fromto[4], fromto[2] - fromto[5]};
  if (mjuu_normvec(vec, 3) < mjEPS) {
    mjuu_setvec(quat, 1, 0, 0, 0);
  } else {
    mjuu_z2quat(quat, vec);
  }
  return type == mjGEOM_BOX || type == mjGEOM_ELLIPSOID ? mjGEOM_CYLINDER : type;
}

// rotate an inertia matrix, given as (xx, yy, zz, xy, xz, yz)
void RotateInertia(double res[6], const double inert[6], const double quat[4]) {
  double mat[9], full[9], rotated[9];
  full[0] = inert[0];
  full[4] = inert[1];
  full[8] = inert[2];
  full[1] = full[3] = inert[3];
  full[2] = full[6] = inert[4];
  full[5] = full[7] = inert[5];
  mjuu_quat2mat(mat, quat);
  mjuu_mulRMRT(rotated, mat, full);
  res[0] = rotated[0];
  res[1] = rotated[4];
  res[2] = rotated[8];
  res[3] = rotated[1];
  res[4] = rotated[2];
  res[5] = rotated[5];
}

}  // namespace


// utiility function for checking size parameters
static void checksize(double* size, mjtGeom type, mjCBase* object, const char* name, int id) {
  // plane: handle infinite
  if (type == mjGEOM_PLANE) {
    if (size[2] <= 0) { throw mjCError(object, "plane size(3) must be positive"); }
  }

  // regular geom
  else {
    for (int i = 0; i < mjGEOMINFO[type]; i++) {
      if (size[i] <= 0) { throw mjCError(object, "size %d must be positive in geom", nullptr, i); }
    }
  }
}

// error message for missing "limited" attribute
static void checklimited(const mjCBase* obj,
                         bool           autolimits,
                         const char*    entity,
                         const char*    attr,
                         int            limited,
                         bool           hasrange) {
  if (!autolimits && limited == 2 && hasrange) {
    std::stringstream ss;
    ss << entity << " has `" << attr << "range` but not `" << attr << "limited`. "
       << "set the autolimits=\"true\" compiler option, specify `" << attr << "limited` "
       << "explicitly (\"true\" or \"false\"), or remove the `" << attr << "range` attribute.";
    throw mjCError(obj, "%s", ss.str().c_str());
  }
}

// returns true if limits should be active
static bool islimited(int limited, const double range[2]) {
  if (limited == mjLIMITED_TRUE || (limited == mjLIMITED_AUTO && range[0] < range[1])) {
    return true;
  }
  return false;
}

// forget the keyframe values which are stored under the given names, or all others if keep is true
template <class T>
static void forgetvalues(std::map<std::string, T>&       values,
                         const std::vector<std::string>& names,
                         bool                            keep) {
  std::map<std::string, T> named;
  for (const std::string& name : names) {
    auto value = values.find(name);
    if (value != values.end()) { named.insert(values.extract(value)); }
  }
  if (keep) { values = std::move(named); }
}

//------------------------- class mjCError implementation ------------------------------------------

// constructor
mjCError::mjCError(const mjCBase* obj, const char* msg, const char* str, int pos1, int pos2) {
  char temp[600];

  // init
  if (obj || msg) {
    mju::sprintf_arr(message, "Error");
  } else {
    message[0] = 0;
  }

  // construct error message
  if (msg) {
    if (str) {
      mju::sprintf_arr(temp, msg, str, pos1, pos2);
    } else {
      mju::sprintf_arr(temp, msg, pos1, pos2);
    }

    mju::strcat_arr(message, ": ");
    mju::strcat_arr(message, temp);
  }

  // append info from mjCBase element
  if (obj) {
    // with or without xml position
    if (!obj->info.empty()) {
      mju::sprintf_arr(temp,
                       "Element name '%s', id %d, %s",
                       obj->name.c_str(),
                       obj->id,
                       obj->info.c_str());
    } else {
      mju::sprintf_arr(temp, "Element name '%s', id %d", obj->name.c_str(), obj->id);
    }

    // append to message
    mju::strcat_arr(message, "\n");
    mju::strcat_arr(message, temp);
  }
}


//------------------ alternative orientation implementation ----------------------------------------

// compute frame orientation given alternative specifications
// used for geom, site, body and camera frames
const char* ResolveOrientation(double*               quat,
                               bool                  degree,
                               const char*           sequence,
                               const mjsOrientation& orient) {
  double axisangle[4];
  double xyaxes[6];
  double zaxis[3];
  double euler[3];

  mjuu_copyvec(axisangle, orient.axisangle, 4);
  mjuu_copyvec(xyaxes, orient.xyaxes, 6);
  mjuu_copyvec(zaxis, orient.zaxis, 3);
  mjuu_copyvec(euler, orient.euler, 3);

  // set quat using axisangle
  if (orient.type == mjORIENTATION_AXISANGLE) {
    // convert to radians if necessary, normalize axis
    if (degree) { axisangle[3] = axisangle[3] / 180.0 * mjPI; }
    if (mjuu_normvec(axisangle, 3) < mjEPS) { return "axisangle too small"; }

    // construct quaternion
    double ang2 = axisangle[3] / 2;
    quat[0]     = cos(ang2);
    quat[1]     = sin(ang2) * axisangle[0];
    quat[2]     = sin(ang2) * axisangle[1];
    quat[3]     = sin(ang2) * axisangle[2];
  }

  // set quat using xyaxes
  if (orient.type == mjORIENTATION_XYAXES) {
    // normalize x axis
    if (mjuu_normvec(xyaxes, 3) < mjEPS) { return "xaxis too small"; }

    // make y axis orthogonal to x axis, normalize
    double d   = mjuu_dot3(xyaxes, xyaxes + 3);
    xyaxes[3] -= xyaxes[0] * d;
    xyaxes[4] -= xyaxes[1] * d;
    xyaxes[5] -= xyaxes[2] * d;
    if (mjuu_normvec(xyaxes + 3, 3) < mjEPS) { return "yaxis too small"; }

    // compute and normalize z axis
    double z[3];
    mjuu_crossvec(z, xyaxes, xyaxes + 3);
    if (mjuu_normvec(z, 3) < mjEPS) { return "cross(xaxis, yaxis) too small"; }

    // convert frame into quaternion
    mjuu_frame2quat(quat, xyaxes, xyaxes + 3, z);
  }

  // set quat using zaxis
  if (orient.type == mjORIENTATION_ZAXIS) {
    if (mjuu_normvec(zaxis, 3) < mjEPS) { return "zaxis too small"; }
    mjuu_z2quat(quat, zaxis);
  }


  // handle euler
  if (orient.type == mjORIENTATION_EULER) {
    // convert to radians if necessary
    if (degree) {
      for (int i = 0; i < 3; i++) { euler[i] = euler[i] / 180.0 * mjPI; }
    }

    // init
    mjuu_setvec(quat, 1, 0, 0, 0);

    // loop over euler angles, accumulate rotations
    for (int i = 0; i < 3; i++) {
      double tmp[4], qrot[4] = {cos(euler[i] / 2), 0, 0, 0};
      double sa = sin(euler[i] / 2);

      // construct quaternion rotation
      if (sequence[i] == 'x' || sequence[i] == 'X') {
        qrot[1] = sa;
      } else if (sequence[i] == 'y' || sequence[i] == 'Y') {
        qrot[2] = sa;
      } else if (sequence[i] == 'z' || sequence[i] == 'Z') {
        qrot[3] = sa;
      } else {
        return "euler sequence can only contain x, y, z, X, Y, Z";
      }

      // accumulate rotation
      if (sequence[i] == 'x' || sequence[i] == 'y' || sequence[i] == 'z') {
        mjuu_mulquat(tmp, quat, qrot);  // moving axes: post-multiply
      } else {
        mjuu_mulquat(tmp, qrot, quat);  // fixed axes: pre-multiply
      }
      mjuu_copyvec(quat, tmp, 4);
    }

    // normalize, just in case
    mjuu_normvec(quat, 4);
  }

  return 0;
}


//------------------------- class mjCBoundingVolumeHierarchy implementation ------------------------


// assign position and orientation
void mjCBoundingVolumeHierarchy::Set(double ipos_element[3], double iquat_element[4]) {
  mjuu_copyvec(ipos_, ipos_element, 3);
  mjuu_copyvec(iquat_, iquat_element, 4);
}


void mjCBoundingVolumeHierarchy::AllocateBoundingVolumes(int nleaf) {
  nbvh_ = 0;
  bvh_.clear();
  child_.clear();
  nodeid_.clear();
  nodeidptr_.clear();
  level_.clear();
  bvleaf_.clear();
  bvleaf_.reserve(nleaf);
}


void mjCBoundingVolumeHierarchy::RemoveInactiveVolumes(int nmax) {
  bvleaf_.erase(bvleaf_.begin() + nmax, bvleaf_.end());
}


const mjCBoundingVolume* mjCBoundingVolumeHierarchy::AddBoundingVolume(int           id,
                                                                       int           contype,
                                                                       int           conaffinity,
                                                                       const double* pos,
                                                                       const double* quat,
                                                                       const double* aabb) {
  bvleaf_.emplace_back(id, contype, conaffinity, pos, quat, aabb);
  return &bvleaf_.back();
}


const mjCBoundingVolume* mjCBoundingVolumeHierarchy::AddBoundingVolume(const int*    id,
                                                                       int           contype,
                                                                       int           conaffinity,
                                                                       const double* pos,
                                                                       const double* quat,
                                                                       const double* aabb) {
  bvleaf_.emplace_back(id, contype, conaffinity, pos, quat, aabb);
  return &bvleaf_.back();
}


// create bounding volume hierarchy
void mjCBoundingVolumeHierarchy::CreateBVH(mjCModel* model, const mjCBase* owner) {
  std::vector<BVElement> elements;
  Make(elements);
  MakeBVH(elements.begin(), elements.end(), 0, model, owner);
}

void mjCBoundingVolumeHierarchy::Make(std::vector<BVElement>& elements) {
  // precompute the positions of each element in the hierarchy's axes, and drop
  // visual-only elements.
  elements.reserve(bvleaf_.size());
  double qinv[4] = {iquat_[0], -iquat_[1], -iquat_[2], -iquat_[3]};
  for (int i = 0; i < bvleaf_.size(); i++) {
    if (bvleaf_[i].Conaffinity() || bvleaf_[i].Contype()) {
      BVElement element;
      element.e = &bvleaf_[i];

      double vert[3] = {element.e->Pos(0) - ipos_[0],
                        element.e->Pos(1) - ipos_[1],
                        element.e->Pos(2) - ipos_[2]};
      mjuu_rotVecQuat(element.lpos, vert, qinv);
      elements.push_back(std::move(element));
    }
  }
}


// compute bounding volume hierarchy
int mjCBoundingVolumeHierarchy::MakeBVH(std::vector<BVElement>::iterator elements_begin,
                                        std::vector<BVElement>::iterator elements_end,
                                        int                              lev,
                                        mjCModel*                        model,
                                        const mjCBase*                   owner) {
  int nelements = elements_end - elements_begin;
  if (nelements == 0) { return -1; }
  constexpr double kMaxVal = std::numeric_limits<double>::max();
  double           AAMM[6] = {kMaxVal, kMaxVal, kMaxVal, -kMaxVal, -kMaxVal, -kMaxVal};

  // inverse transformation
  double qinv[4] = {iquat_[0], -iquat_[1], -iquat_[2], -iquat_[3]};

  // accumulate AAMM over elements
  for (auto element = elements_begin; element != elements_end; ++element) {
    // transform element aabb to aamm format
    double aamm[6] = {element->e->AABB(0) - element->e->AABB(3),
                      element->e->AABB(1) - element->e->AABB(4),
                      element->e->AABB(2) - element->e->AABB(5),
                      element->e->AABB(0) + element->e->AABB(3),
                      element->e->AABB(1) + element->e->AABB(4),
                      element->e->AABB(2) + element->e->AABB(5)};

    // update node AAMM
    for (int v = 0; v < 8; v++) {
      double vert[3], box[3];
      vert[0] = (v & 1 ? aamm[3] : aamm[0]);
      vert[1] = (v & 2 ? aamm[4] : aamm[1]);
      vert[2] = (v & 4 ? aamm[5] : aamm[2]);

      // rotate to the body inertial frame if specified
      if (element->e->Quat()) {
        mjuu_rotVecQuat(box, vert, element->e->Quat());
        box[0] += element->e->Pos(0) - ipos_[0];
        box[1] += element->e->Pos(1) - ipos_[1];
        box[2] += element->e->Pos(2) - ipos_[2];
        mjuu_rotVecQuat(vert, box, qinv);
      }

      AAMM[0] = std::min(AAMM[0], vert[0]);
      AAMM[1] = std::min(AAMM[1], vert[1]);
      AAMM[2] = std::min(AAMM[2], vert[2]);
      AAMM[3] = std::max(AAMM[3], vert[0]);
      AAMM[4] = std::max(AAMM[4], vert[1]);
      AAMM[5] = std::max(AAMM[5], vert[2]);
    }
  }

  // inflate flat AABBs
  for (int i = 0; i < 3; i++) {
    if (std::abs(AAMM[i] - AAMM[i + 3]) < mjEPS) {
      AAMM[i + 0] -= mjEPS;
      AAMM[i + 3] += mjEPS;
    }
  }

  // store current index
  int index = nbvh_++;
  child_.push_back(-1);
  child_.push_back(-1);
  nodeid_.push_back(-1);
  nodeidptr_.push_back(nullptr);
  level_.push_back(lev);

  // store bounding box of the current node
  bvh_.push_back((AAMM[3] + AAMM[0]) / 2);
  bvh_.push_back((AAMM[4] + AAMM[1]) / 2);
  bvh_.push_back((AAMM[5] + AAMM[2]) / 2);
  bvh_.push_back((AAMM[3] - AAMM[0]) / 2);
  bvh_.push_back((AAMM[4] - AAMM[1]) / 2);
  bvh_.push_back((AAMM[5] - AAMM[2]) / 2);

  // leaf node, return
  if (nelements == 1) {
    child_[2 * index + 0] = -1;
    child_[2 * index + 1] = -1;
    nodeid_[index]        = *elements_begin->e->Id();
    nodeidptr_[index]     = (int*)elements_begin->e->Id();
    return index;
  }

  // find longest axis, by a margin of at least mjEPS, default to 0
  int    axis     = 0;
  double edges[3] = {AAMM[3] - AAMM[0], AAMM[4] - AAMM[1], AAMM[5] - AAMM[2]};
  if (edges[1] >= edges[0] + mjEPS) axis = 1;
  if (edges[2] >= edges[axis] + mjEPS) axis = 2;

  // find median along the axis
  auto compare = [&](const BVElement& e1, const BVElement& e2) {
    if (std::abs(e1.lpos[axis] - e2.lpos[axis]) > mjEPS) { return e1.lpos[axis] < e2.lpos[axis]; }
    // comparing pointers gives a stable sort, because they both come from the same array
    return e1.e < e2.e;
  };

  // note: nth_element performs a partial sort of elements
  int m = nelements / 2;
  std::nth_element(elements_begin, elements_begin + m, elements_end, compare);

  // recursive calls
  if (m > 0) {
    child_[2 * index + 0] = MakeBVH(elements_begin, elements_begin + m, lev + 1, model, owner);
  }

  if (m != nelements) {
    child_[2 * index + 1] = MakeBVH(elements_begin + m, elements_end, lev + 1, model, owner);
  }

  // SHOULD NOT OCCUR
  if (child_[2 * index + 0] == -1 && child_[2 * index + 1] == -1) {
    mju_error("this should have been a leaf, body=%s nelements=%d", name_.c_str(), nelements);
  }

  if (lev > mjMAXTREEDEPTH) { model->AddWarning("max tree depth exceeded", owner); }

  return index;
}


//------------------------- class mjCOctree implementation -----------------------------------------
void mjCOctree::CopyLevel(int* level) const {
  for (int i = 0; i < node_.size(); ++i) { level[i] = node_[i].level; }
}


void mjCOctree::CopyChild(int* child) const {
  for (int i = 0; i < node_.size(); ++i) {
    for (int j = 0; j < 8; ++j) { child[i * 8 + j] = node_[i].child[j]; }
  }
}


void mjCOctree::CopyAabb(mjtNum* aabb) const {
  for (int i = 0; i < node_.size(); ++i) {
    aabb[i * 6 + 0] = (node_[i].aamm[0] + node_[i].aamm[3]) / 2;
    aabb[i * 6 + 1] = (node_[i].aamm[1] + node_[i].aamm[4]) / 2;
    aabb[i * 6 + 2] = (node_[i].aamm[2] + node_[i].aamm[5]) / 2;
    aabb[i * 6 + 3] = (node_[i].aamm[3] - node_[i].aamm[0]) / 2;
    aabb[i * 6 + 4] = (node_[i].aamm[4] - node_[i].aamm[1]) / 2;
    aabb[i * 6 + 5] = (node_[i].aamm[5] - node_[i].aamm[2]) / 2;
  }
}


void mjCOctree::CopyCoeff(mjtNum* coeff) const {
  for (int i = 0; i < node_.size(); ++i) {
    for (int j = 0; j < 8; ++j) { coeff[i * 8 + j] = node_[i].coeff[j]; }
  }
}


void mjCOctree::SetFace(const std::vector<double>& vert, const std::vector<int>& face) {
  face_.reserve(face.size() / 3);
  for (int i = 0; i < face.size(); i += 3) {
    std::array<double, 3> v0 = {vert[3 * face[i + 0] + 0],
                                vert[3 * face[i + 0] + 1],
                                vert[3 * face[i + 0] + 2]};
    std::array<double, 3> v1 = {vert[3 * face[i + 1] + 0],
                                vert[3 * face[i + 1] + 1],
                                vert[3 * face[i + 1] + 2]};
    std::array<double, 3> v2 = {vert[3 * face[i + 2] + 0],
                                vert[3 * face[i + 2] + 1],
                                vert[3 * face[i + 2] + 2]};
    face_.push_back({v0, v1, v2});
  }
}


// TODO: use the same code as mjCBoundingVolumeHierarchy::Make()
void mjCOctree::Make(std::vector<Triangle>& elements) {
  // rotate triangles to the body inertial frame
  elements.assign(face_.size(), {{{0}}});
  double qinv[4] = {iquat_[0], -iquat_[1], -iquat_[2], -iquat_[3]};
  for (int i = 0; i < face_.size(); i++) {
    for (int j = 0; j < 3; j++) {
      double vert[3] = {face_[i][j][0] - ipos_[0],
                        face_[i][j][1] - ipos_[1],
                        face_[i][j][2] - ipos_[2]};
      mjuu_rotVecQuat(elements[i][j].data(), vert, qinv);
    }
  }
}


void mjCOctree::CreateOctree(const double aamm[6]) {
  Clear();

  double aabb[6] = {(aamm[0] + aamm[3]) / 2,
                    (aamm[1] + aamm[4]) / 2,
                    (aamm[2] + aamm[5]) / 2,
                    (aamm[3] - aamm[0]) / 2,
                    (aamm[4] - aamm[1]) / 2,
                    (aamm[5] - aamm[2]) / 2};
  double box[6]  = {aabb[0] - 1.1 * aabb[3],
                    aabb[1] - 1.1 * aabb[4],
                    aabb[2] - 1.1 * aabb[5],
                    aabb[0] + 1.1 * aabb[3],
                    aabb[1] + 1.1 * aabb[4],
                    aabb[2] + 1.1 * aabb[5]};

  std::vector<Triangle> elements;
  Make(elements);
  std::vector<Triangle*> elements_ptrs(elements.size());
  std::transform(elements.begin(), elements.end(), elements_ptrs.begin(), [](Triangle& triangle) {
    return &triangle;
  });
  std::unordered_map<Point, int> vert_map;
  MakeOctree(elements_ptrs, box, vert_map);
  MarkHangingNodes();
}


namespace {

double pointBoxDistSq(const double* p, const mjtNum* aabb) {
  double dist_sq = 0;
  for (int i = 0; i < 3; ++i) {
    double lo = aabb[i] - aabb[i + 3];
    double hi = aabb[i] + aabb[i + 3];
    if (p[i] < lo) {
      dist_sq += (lo - p[i]) * (lo - p[i]);
    } else if (p[i] > hi) {
      dist_sq += (p[i] - hi) * (p[i] - hi);
    }
  }
  return dist_sq;
}


// Voronoi region of the closest point on a triangle: a corner, the side from corner k to
// corner k+1, or the interior
enum TriangleFeature { kCorner0, kCorner1, kCorner2, kSide01, kSide12, kSide20, kInterior };


// compute squared distance between point p and triangle (v0, v1, v2),
// and return barycentric coordinates (u,v) and the feature of the closest point
double pointTriDistSqWithUV(const double* p,
                            const double* v0,
                            const double* v1,
                            const double* v2,
                            double&       out_u,
                            double&       out_v,
                            int&          out_feature) {
  double ab[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  double ac[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  double ap[3] = {p[0] - v0[0], p[1] - v0[1], p[2] - v0[2]};

  // the closest point on the triangle is determined by partitioning space into Voronoi regions
  double d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
  double d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];

  // region A (vertex v0)
  if (d1 <= 0 && d2 <= 0) {
    out_u       = 0;
    out_v       = 0;
    out_feature = kCorner0;
    return ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2];
  }

  double bp[3] = {p[0] - v1[0], p[1] - v1[1], p[2] - v1[2]};
  double d3    = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
  double d4    = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];

  // region B (vertex v1)
  if (d3 >= 0 && d4 <= d3) {
    out_u       = 1;
    out_v       = 0;
    out_feature = kCorner1;
    return bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2];
  }

  // region AB (edge v0-v1)
  double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    double u          = d1 / (d1 - d3);
    out_u             = u;
    out_v             = 0;
    out_feature       = kSide01;
    double closest[3] = {v0[0] + u * ab[0], v0[1] + u * ab[1], v0[2] + u * ab[2]};
    return (p[0] - closest[0]) * (p[0] - closest[0]) +
           (p[1] - closest[1]) * (p[1] - closest[1]) +
           (p[2] - closest[2]) * (p[2] - closest[2]);
  }

  double cp[3] = {p[0] - v2[0], p[1] - v2[1], p[2] - v2[2]};
  double d5    = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
  double d6    = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];

  // region C (vertex v2)
  if (d6 >= 0 && d5 <= d6) {
    out_u       = 0;
    out_v       = 1;
    out_feature = kCorner2;
    return cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2];
  }

  // region AC (edge v0-v2)
  double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    double v          = d2 / (d2 - d6);
    out_u             = 0;
    out_v             = v;
    out_feature       = kSide20;
    double closest[3] = {v0[0] + v * ac[0], v0[1] + v * ac[1], v0[2] + v * ac[2]};
    return (p[0] - closest[0]) * (p[0] - closest[0]) +
           (p[1] - closest[1]) * (p[1] - closest[1]) +
           (p[2] - closest[2]) * (p[2] - closest[2]);
  }

  // region BC (edge v1-v2)
  double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    double w          = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    out_u             = 1 - w;
    out_v             = w;
    out_feature       = kSide12;
    double bc[3]      = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    double closest[3] = {v1[0] + w * bc[0], v1[1] + w * bc[1], v1[2] + w * bc[2]};
    return (p[0] - closest[0]) * (p[0] - closest[0]) +
           (p[1] - closest[1]) * (p[1] - closest[1]) +
           (p[2] - closest[2]) * (p[2] - closest[2]);
  }

  // region ABC (inside triangle)
  double denom      = 1.0 / (va + vb + vc);
  double u          = vb * denom;
  double v          = vc * denom;
  out_u             = u;
  out_v             = v;
  out_feature       = kInterior;
  double closest[3] = {v0[0] + u * ab[0] + v * ac[0],
                       v0[1] + u * ab[1] + v * ac[1],
                       v0[2] + u * ab[2] + v * ac[2]};
  return (p[0] - closest[0]) * (p[0] - closest[0]) +
         (p[1] - closest[1]) * (p[1] - closest[1]) +
         (p[2] - closest[2]) * (p[2] - closest[2]);
}


// query BVH for closest face to point p, return distance, face index, barycentric coordinates
// and feature of the closest point
void queryClosestBVHWithFace(const mjtNum* bvh,
                             const int*    child,
                             const int*    nodeid,
                             const double* vert,
                             const int*    face,
                             int           node_idx,
                             const double* p,
                             double&       best_dist_sq,
                             int&          best_face,
                             double&       best_u,
                             double&       best_v,
                             int&          best_feature) {
  const mjtNum* aabb = &bvh[node_idx * 6];
  if (pointBoxDistSq(p, aabb) >= best_dist_sq) return;

  int left  = child[node_idx * 2];
  int right = child[node_idx * 2 + 1];

  if (left == -1 && right == -1) {
    int fi = nodeid[node_idx];
    if (fi >= 0) {
      const double* v0 = vert + face[fi * 3 + 0] * 3;
      const double* v1 = vert + face[fi * 3 + 1] * 3;
      const double* v2 = vert + face[fi * 3 + 2] * 3;
      double        u, v;
      int           feature;
      double        dist_sq = pointTriDistSqWithUV(p, v0, v1, v2, u, v, feature);
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_face    = fi;
        best_u       = u;
        best_v       = v;
        best_feature = feature;
      }
    }
    return;
  }

  if (left >= 0) {
    queryClosestBVHWithFace(bvh,
                            child,
                            nodeid,
                            vert,
                            face,
                            left,
                            p,
                            best_dist_sq,
                            best_face,
                            best_u,
                            best_v,
                            best_feature);
  }
  if (right >= 0) {
    queryClosestBVHWithFace(bvh,
                            child,
                            nodeid,
                            vert,
                            face,
                            right,
                            p,
                            best_dist_sq,
                            best_face,
                            best_u,
                            best_v,
                            best_feature);
  }
}


double querySignedDistance(const mjtNum*            bvh,
                           const int*               child,
                           const int*               nodeid,
                           int                      nbvh,
                           const double*            point,
                           const double*            vert,
                           const int*               face,
                           const MeshPseudonormals& normals) {
  if (nbvh == 0) { return 0; }

  double best_dist_sq = 1e20;
  int    best_face    = -1;
  double best_u = 0, best_v = 0;
  int    best_feature = kInterior;
  queryClosestBVHWithFace(bvh,
                          child,
                          nodeid,
                          vert,
                          face,
                          0,
                          point,
                          best_dist_sq,
                          best_face,
                          best_u,
                          best_v,
                          best_feature);
  double dist = std::sqrt(best_dist_sq);

  double sign = 1.0;
  if (best_face >= 0) {
    const double* v0 = vert + face[best_face * 3 + 0] * 3;
    const double* v1 = vert + face[best_face * 3 + 1] * 3;
    const double* v2 = vert + face[best_face * 3 + 2] * 3;

    // a single face normal cannot determine the sign at a shared vertex or edge
    double        face_normal[3];
    const double* normal = face_normal;
    if (best_feature <= kCorner2) {
      normal = normals.vertex.data() + 3 * face[3 * best_face + best_feature];
    } else if (best_feature <= kSide20) {
      normal = normals.edge.data() + 3 * normals.side[3 * best_face + best_feature - kSide01];
    } else {
      double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
      double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
      mjuu_crossvec(face_normal, e1, e2);
    }

    double closest[3] = {v0[0] + best_u * (v1[0] - v0[0]) + best_v * (v2[0] - v0[0]),
                         v0[1] + best_u * (v1[1] - v0[1]) + best_v * (v2[1] - v0[1]),
                         v0[2] + best_u * (v1[2] - v0[2]) + best_v * (v2[2] - v0[2])};

    double u[3] = {point[0] - closest[0], point[1] - closest[1], point[2] - closest[2]};
    sign        = mjuu_dot3(u, normal) >= 0 ? 1.0 : -1.0;
  }

  return sign * dist;
}

}  // namespace


MeshPseudonormals::MeshPseudonormals(const double* vert, int nvert, const int* face, int nface)
    : vertex(3 * nvert, 0), side(3 * nface) {
  // edges are keyed by their sorted vertex indices
  std::unordered_map<std::uint64_t, int> edge_index;
  edge_index.reserve(3 * nface);
  for (int f = 0; f < nface; ++f) {
    const int*    v     = face + 3 * f;
    const double* p[3]  = {vert + 3 * v[0], vert + 3 * v[1], vert + 3 * v[2]};
    double        e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
    double        e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
    double        normal[3];
    mjuu_crossvec(normal, e1, e2);
    double length = std::sqrt(mjuu_dot3(normal, normal));

    for (int k = 0; k < 3; ++k) {
      // every side has an edge, but a degenerate face does not contribute normals
      auto [lo, hi]       = std::minmax(v[k], v[(k + 1) % 3]);
      std::uint64_t key   = (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint32_t>(hi);
      auto [it, inserted] = edge_index.emplace(key, edge_index.size());
      if (inserted) { edge.resize(edge.size() + 3, 0); }
      side[3 * f + k] = it->second;
      if (length == 0) { continue; }

      // vertex normals weight unit face normals by the corner angle; |e1 x e2| = length
      const double* next   = p[(k + 1) % 3];
      const double* prev   = p[(k + 2) % 3];
      double        a[3]   = {next[0] - p[k][0], next[1] - p[k][1], next[2] - p[k][2]};
      double        b[3]   = {prev[0] - p[k][0], prev[1] - p[k][1], prev[2] - p[k][2]};
      double        weight = std::atan2(length, mjuu_dot3(a, b));
      for (int i = 0; i < 3; ++i) { vertex[3 * v[k] + i] += weight * normal[i] / length; }

      // edge normals give equal weight to the unit normals of the incident faces
      for (int i = 0; i < 3; ++i) { edge[3 * it->second + i] += normal[i] / length; }
    }
  }
}


double mjCBoundingVolumeHierarchy::QuerySignedDistance(const double*            point,
                                                       const double*            vert,
                                                       const int*               face,
                                                       const MeshPseudonormals& normals) const {
  return querySignedDistance(bvh_.data(),
                             child_.data(),
                             nodeid_.data(),
                             nbvh_,
                             point,
                             vert,
                             face,
                             normals);
}


void mjCOctree::ComputeSdfCoeffs(const double*                     vert,
                                 int                               nvert,
                                 const int*                        face,
                                 int                               nface,
                                 const mjCBoundingVolumeHierarchy& tree) {
  std::vector<double> coeffs(nvert_, 0.0);
  std::vector<bool>   processed(nvert_, false);
  std::deque<int>     queue;
  MeshPseudonormals   normals(vert, nvert, face, nface);

  if (NumNodes() > 0) { queue.push_back(0); }

  while (!queue.empty()) {
    int node_idx = queue.front();
    queue.pop_front();

    // compute SDF coefficients at the 8 vertices of the octree node
    for (int j = 0; j < 8; ++j) {
      int vert_id = VertId(node_idx, j);
      if (processed[vert_id]) { continue; }
      if (Hang(vert_id).empty()) {
        // transform from octree frame (body inertial) back to mesh frame
        double p_mesh[3];
        mjuu_rotVecQuat(p_mesh, Vert(vert_id), iquat_);
        p_mesh[0] += ipos_[0];
        p_mesh[1] += ipos_[1];
        p_mesh[2] += ipos_[2];

        coeffs[vert_id] = tree.QuerySignedDistance(p_mesh, vert, face, normals);
      } else {
        // hanging node: interpolate from parents
        double sum_coeff = 0;
        for (int dep_id : Hang(vert_id)) { sum_coeff += coeffs[dep_id]; }
        coeffs[vert_id] = sum_coeff / Hang(vert_id).size();
      }
      processed[vert_id] = true;
    }

    // add children to the queue
    for (int child_idx : Children(node_idx)) {
      if (child_idx != -1) { queue.push_back(child_idx); }
    }
  }

  // optional Laplacian smoothing (smooths octree level transitions)
  if (smoothing_iterations_ > 0) {
    // build vertex neighbor graph from octree connectivity
    std::vector<std::set<int>> neighbors(nvert_);
    for (int i = 0; i < NumNodes(); ++i) {
      static const int edges[12][2] = {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
          {0, 2},
          {1, 3},
          {4, 6},
          {5, 7},
          {0, 4},
          {1, 5},
          {2, 6},
          {3, 7}
      };
      for (const auto& edge : edges) {
        int v0 = VertId(i, edge[0]);
        int v1 = VertId(i, edge[1]);
        neighbors[v0].insert(v1);
        neighbors[v1].insert(v0);
      }
    }

    // apply Laplacian smoothing
    const double        alpha = 0.2;
    std::vector<double> sdf_new(nvert_);
    for (int iter = 0; iter < smoothing_iterations_; ++iter) {
      for (int i = 0; i < nvert_; ++i) {
        if (neighbors[i].empty()) {
          sdf_new[i] = coeffs[i];
        } else {
          double avg = 0;
          for (int j : neighbors[i]) avg += coeffs[j];
          avg        /= neighbors[i].size();
          sdf_new[i]  = (1 - alpha) * coeffs[i] + alpha * avg;
        }
      }
      std::swap(coeffs, sdf_new);
    }
  }

  // copy coefficients to the octree nodes
  for (int i = 0; i < NumNodes(); ++i) {
    for (int j = 0; j < 8; j++) { AddCoeff(i, j, coeffs[VertId(i, j)]); }
  }
}


static double dot2(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1];
}


// from M. Schwarz and H.-P. Seidel, "Fast Parallel Surface and Solid Voxelization on GPUs".
static bool boxTriangle(const Triangle& v, const double aamm[6]) {
  // bounding box tests
  for (int i = 0; i < 3; i++) {
    if (v[0][i] < aamm[i] && v[1][i] < aamm[i] && v[2][i] < aamm[i]) { return false; }
    int j = i + 3;
    if (v[0][i] > aamm[j] && v[1][i] > aamm[j] && v[2][i] > aamm[j]) { return false; }
  }

  // test for triangle plane and box overlap
  double n[3];
  double e[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { e[i][j] = v[(i + 1) % 3][j] - v[i][j]; }
  }

  mjuu_crossvec(n, e[0], e[1]);
  double size[3] = {aamm[3] - aamm[0], aamm[4] - aamm[1], aamm[5] - aamm[2]};
  double c[3]    = {n[0] > 0 ? size[0] : 0, n[1] > 0 ? size[1] : 0, n[2] > 0 ? size[2] : 0};
  double c1[3]   = {c[0] - v[0][0], c[1] - v[0][1], c[2] - v[0][2]};
  double c2[3]   = {size[0] - c[0] - v[0][0], size[1] - c[1] - v[0][1], size[2] - c[2] - v[0][2]};

  if ((mjuu_dot3(n, aamm) + mjuu_dot3(n, c1)) * (mjuu_dot3(n, aamm) + mjuu_dot3(n, c2)) > 0) {
    return false;
  }

  // test projection overlap
  for (int a = 0; a < 3; a++) {
    int b = (a + 1) % 3;
    int c = (a + 2) % 3;
    for (int i = 0; i < 3; i++) {
      double sign  = n[a] >= 0 ? 1 : -1;
      double ne[2] = {-e[i][c] * sign, e[i][b] * sign};
      double vi[2] = {v[i][b], v[i][c]};
      double d     = -dot2(ne, vi) + mju_max(0, size[b] * ne[0]) + mju_max(0, size[c] * ne[1]);
      double p[2]  = {aamm[b], aamm[c]};
      if (dot2(ne, p) + d < 0) { return false; }
    }
  }

  return true;
}


void mjCOctree::TaskToNode(const OctreeTask&               task,
                           OctNode&                        node,
                           std::unordered_map<Point, int>& vert_map) {
  node.level        = task.lev;
  node.parent_index = task.parent_index;
  node.child_slot   = task.child_slot;

  if (task.parent_index != -1) {
    node_[task.parent_index].child[task.child_slot] = task.node_index;

    const auto parent_aamm = node_[task.parent_index].aamm;
    node.aamm[0] = task.child_slot & 1 ? (parent_aamm[3] + parent_aamm[0]) / 2 : parent_aamm[0];
    node.aamm[1] = task.child_slot & 2 ? (parent_aamm[4] + parent_aamm[1]) / 2 : parent_aamm[1];
    node.aamm[2] = task.child_slot & 4 ? (parent_aamm[5] + parent_aamm[2]) / 2 : parent_aamm[2];
    node.aamm[3] = task.child_slot & 1 ? parent_aamm[3] : (parent_aamm[0] + parent_aamm[3]) / 2;
    node.aamm[4] = task.child_slot & 2 ? parent_aamm[4] : (parent_aamm[1] + parent_aamm[4]) / 2;
    node.aamm[5] = task.child_slot & 4 ? parent_aamm[5] : (parent_aamm[2] + parent_aamm[5]) / 2;
  }

  for (int i = 0; i < 8; i++) {
    Point v = {
        {(i & 1) ? node.aamm[3] : node.aamm[0],
         (i & 2) ? node.aamm[4] : node.aamm[1],
         (i & 4) ? node.aamm[5] : node.aamm[2]}
    };
    auto it = vert_map.find(v);
    if (it != vert_map.end()) {
      node.vertid[i] = it->second;
    } else {
      node.vertid[i] = nvert_;
      vert_map[v]    = nvert_++;
      vert_.push_back(v);
    }
    node.child[i] = -1;
  }
}


void mjCOctree::Subdivide(const OctreeTask&               task,
                          std::unordered_map<Point, int>& vert_map,
                          std::deque<OctreeTask>*         queue,
                          const std::vector<Triangle*>&   colliding) {
  for (int i = 0; i < 8; i++) {
    OctreeTask new_task;
    new_task.elements     = colliding;
    new_task.lev          = task.lev + 1;
    new_task.parent_index = task.node_index;
    new_task.node_index   = nnode_++;
    new_task.child_slot   = i;

    node_.push_back(OctNode());
    TaskToNode(new_task, node_.back(), vert_map);
    if (queue) { queue->push_back(std::move(new_task)); }
  }
}


// recursively finds the adjacent ancestor neighbor region
int mjCOctree::FindCoarseNeighbor(int node_idx, int dir) {
  if (node_idx == -1) { return -1; }

  int parent_idx = node_[node_idx].parent_index;

  // if we are at the root, we have no parent and thus no siblings or external neighbors
  if (parent_idx == -1) { return -1; }

  int child_slot = node_[node_idx].child_slot;
  int dim        = dir / 2;
  int side       = dir % 2;
  int bit        = 1 << dim;

  if (side != ((child_slot & bit) != 0)) {
    // internal neighbor case: This is the successful termination of the climb
    // return the adjacent sibling node
    return node_[parent_idx].child[child_slot ^ bit];
  } else {
    // external neighbor case: Recurse up the tree
    // ask our parent to find its neighbor in the same direction
    return FindCoarseNeighbor(parent_idx, dir);
  }
}


int mjCOctree::FindNeighbor(int node_idx, int dir) {
  if (node_idx == -1) { return -1; }

  // call the helper to find the adjacent to the coarse neighbor.
  // this might be an internal node (e.g., our parent's sibling)
  int result = FindCoarseNeighbor(node_idx, dir);

  if (result == -1) {
    // no neighbor found (either at tree boundary or some other error)
    return -1;
  }

  // leaf descent
  double node_center[3] = {
      (node_[node_idx].aamm[0] + node_[node_idx].aamm[3]) / 2,
      (node_[node_idx].aamm[1] + node_[node_idx].aamm[4]) / 2,
      (node_[node_idx].aamm[2] + node_[node_idx].aamm[5]) / 2,
  };

  while (node_[result].child[0] != -1) {
    double result_center[3] = {
        (node_[result].aamm[0] + node_[result].aamm[3]) / 2,
        (node_[result].aamm[1] + node_[result].aamm[4]) / 2,
        (node_[result].aamm[2] + node_[result].aamm[5]) / 2,
    };

    // find relative octant of our node w.r.t. the neighbor's center
    int next_child_slot = 0;
    if (node_center[0] > result_center[0]) next_child_slot |= 1;
    if (node_center[1] > result_center[1]) next_child_slot |= 2;
    if (node_center[2] > result_center[2]) next_child_slot |= 4;

    int dim  = dir / 2;
    int side = dir % 2;
    int bit  = 1 << dim;

    // we need the child on the opposite side (adjacent to this node)
    int op_side     = (side != 1);
    next_child_slot = (next_child_slot & ~bit) | (op_side * bit);
    result          = node_[result].child[next_child_slot];
  }

  return result;
}


// refine the octree by subdividing nodes that are too coarse such that the
// maximum level difference between adjacent nodes is at most 1.
void mjCOctree::BalanceOctree(std::unordered_map<Point, int>& vert_map) {
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<int> leaves;
    for (int i = 0; i < nnode_; ++i) {
      if (node_[i].child[0] == -1) { leaves.push_back(i); }
    }

    // find the nodes that are too coarse, only leaves need to be checked
    std::vector<int> leaves_to_subdivide;
    for (int leaf_idx : leaves) {
      if (node_[leaf_idx].child[0] != -1) { continue; }

      for (int dir = 0; dir < 6; ++dir) {
        int neighbor_idx = FindNeighbor(leaf_idx, dir);
        if (neighbor_idx == -1) { continue; }
        int neighbor_level = node_[neighbor_idx].level;
        if (neighbor_level > node_[leaf_idx].level + 1) { leaves_to_subdivide.push_back(leaf_idx); }
        if (node_[leaf_idx].level > neighbor_level + 1) {
          leaves_to_subdivide.push_back(neighbor_idx);
        }
      }
    }

    // subdivide the nodes that are too coarse
    if (!leaves_to_subdivide.empty()) {
      changed = true;
      for (int node_idx : leaves_to_subdivide) {
        if (node_[node_idx].child[0] == -1) {  // check if not already subdivided
          OctreeTask task;
          task.node_index = node_idx;
          task.lev        = node_[node_idx].level;
          Subdivide(task, vert_map);
        }
      }
    }
  }
}


// mark all hanging vertices in the octree
void mjCOctree::MarkHangingNodes() {
  hang_.assign(nvert_, std::vector<int>());

  std::vector<int> leaves;
  for (int i = 0; i < nnode_; ++i) {
    if (node_[i].child[0] == -1) { leaves.push_back(i); }
  }

  for (int leaf_idx : leaves) {
    for (int dir = 0; dir < 6; ++dir) {
      int neighbor_idx = FindNeighbor(leaf_idx, dir);
      if (neighbor_idx == -1 || node_[neighbor_idx].level >= node_[leaf_idx].level) { continue; }

      // coarser neighbor found, this leaf's face has hanging nodes
      int dim  = dir / 2;
      int side = dir % 2;

      // iterate over the 4 vertices of the leaf's face
      for (int i = 0; i < 4; ++i) {
        // construct vertex index on the face
        int v_idx  = side << dim;
        int d1     = (dim + 1) % 3;
        int d2     = (dim + 2) % 3;
        v_idx     |= (i & 1) << d1;
        v_idx     |= ((i >> 1) & 1) << d2;

        int hv_id = node_[leaf_idx].vertid[v_idx];
        if (!hang_[hv_id].empty()) {
          continue;  // already processed
        }

        const double* hv_pos        = vert_[hv_id].p.data();
        const auto&   neighbor_aamm = node_[neighbor_idx].aamm;

        bool is_min[3], is_max[3];
        int  on_boundary_planes = 0;
        for (int d = 0; d < 3; ++d) {
          is_min[d] = std::abs(hv_pos[d] - neighbor_aamm[d]) < 1e-9;
          is_max[d] = std::abs(hv_pos[d] - neighbor_aamm[d + 3]) < 1e-9;
          if (is_min[d] || is_max[d]) { on_boundary_planes++; }
        }

        if (on_boundary_planes == 2) {  // edge hanging
          int d_mid = -1;
          for (int d = 0; d < 3; ++d) {
            if (!is_min[d] && !is_max[d]) {
              d_mid = d;
              break;
            }
          }

          int bits[3];
          bits[d_mid]           = 0;  // this will be toggled
          bits[(d_mid + 1) % 3] = is_max[(d_mid + 1) % 3];
          bits[(d_mid + 2) % 3] = is_max[(d_mid + 2) % 3];

          int nv_idx1 = (bits[2] << 2) | (bits[1] << 1) | bits[0];
          bits[d_mid] = 1;
          int nv_idx2 = (bits[2] << 2) | (bits[1] << 1) | bits[0];

          hang_[hv_id].push_back(node_[neighbor_idx].vertid[nv_idx1]);
          hang_[hv_id].push_back(node_[neighbor_idx].vertid[nv_idx2]);
        } else if (on_boundary_planes == 1) {  // face hanging
          int d_face = -1;
          for (int d = 0; d < 3; ++d) {
            if (is_min[d] || is_max[d]) {
              d_face = d;
              break;
            }
          }

          int bits[3];
          bits[d_face] = is_max[d_face];

          for (int j = 0; j < 4; ++j) {
            bits[(d_face + 1) % 3] = j & 1;
            bits[(d_face + 2) % 3] = (j >> 1) & 1;
            int nv_idx             = (bits[2] << 2) | (bits[1] << 1) | bits[0];
            hang_[hv_id].push_back(node_[neighbor_idx].vertid[nv_idx]);
          }
        }
      }
    }
  }
}


void mjCOctree::MakeOctree(const std::vector<Triangle*>&   elements,
                           const double                    aamm[6],
                           std::unordered_map<Point, int>& vert_map) {
  std::deque<OctreeTask> queue;
  OctreeTask             initial_task;
  initial_task.elements     = elements;
  initial_task.lev          = 0;
  initial_task.parent_index = -1;
  initial_task.child_slot   = -1;
  initial_task.node_index   = nnode_++;
  queue.push_back(std::move(initial_task));

  // create root node
  node_.push_back(OctNode());
  OctNode& root = node_.back();
  std::copy(aamm, aamm + 6, root.aamm.data());
  TaskToNode(queue.front(), node_.back(), vert_map);

  while (!queue.empty()) {
    OctreeTask task = std::move(queue.front());
    queue.pop_front();

    // find all triangles that intersect the current box
    std::vector<Triangle*> colliding;
    for (auto* element : task.elements) {
      if (boxTriangle(*element, node_[task.node_index].aamm.data())) {
        colliding.push_back(element);
      }
    }

    // skip if the box is empty
    if (colliding.empty() || task.lev >= max_depth_) { continue; }

    // subdivide the node
    Subdivide(task, vert_map, &queue, colliding);
  }

  // store the neighbors of each node
  BalanceOctree(vert_map);
}

//------------------------- class mjCDef implementation --------------------------------------------

// constructor
mjCDef::mjCDef() {
  name.clear();
  id     = 0;
  parent = nullptr;
  model  = 0;
  child.clear();
  elemtype = mjOBJ_DEFAULT;
  mjs_defaultJoint(&joint_.spec);
  mjs_defaultGeom(&geom_.spec);
  mjs_defaultSite(&site_.spec);
  mjs_defaultCamera(&camera_.spec);
  mjs_defaultLight(&light_.spec);
  mjs_defaultFlex(&flex_.spec);
  mjs_defaultMesh(&mesh_.spec);
  mjs_defaultTexture(&texture_.spec);
  mjs_defaultMaterial(&material_.spec);
  mjs_defaultPair(&pair_.spec);
  mjs_defaultEquality(&equality_.spec);
  mjs_defaultTendon(&tendon_.spec);
  mjs_defaultActuator(&actuator_.spec);

  // make sure all the pointers are local
  PointToLocal();
}


// constructor with model
mjCDef::mjCDef(mjCModel* _model) : mjCDef() {
  model = _model;
}


// copy constructor
mjCDef::mjCDef(const mjCDef& other) {
  *this = other;
}


// compiler
void mjCDef::Compile(const mjCModel* model) {
  CopyFromSpec();

  // enforce length of all default userdata arrays
  joint_.userdata_.resize(model->nuser_jnt);
  geom_.userdata_.resize(model->nuser_geom);
  site_.userdata_.resize(model->nuser_site);
  camera_.userdata_.resize(model->nuser_cam);
  tendon_.userdata_.resize(model->nuser_tendon);
  actuator_.userdata_.resize(model->nuser_actuator);
}


// assignment operator
mjCDef& mjCDef::operator=(const mjCDef& other) {
  if (this != &other) {
    CopyWithoutChildren(other);

    // copy the rest of the default tree
    *this += other;
  }
  return *this;
}


mjCDef& mjCDef::operator+=(const mjCDef& other) {
  for (unsigned int i = 0; i < other.child.size(); i++) {
    child.push_back(new mjCDef(*other.child[i]));  // triggers recursive call
    child.back()->parent = this;
  }
  return *this;
}


void mjCDef::NameSpace(const mjCModel* m) {
  if (!name.empty()) { name = m->prefix + name + m->suffix; }
  for (auto c : child) { c->NameSpace(m); }
}


void mjCDef::CopyWithoutChildren(const mjCDef& other) {
  name     = other.name;
  elemtype = other.elemtype;
  parent   = nullptr;
  child.clear();
  joint_    = other.joint_;
  geom_     = other.geom_;
  site_     = other.site_;
  camera_   = other.camera_;
  light_    = other.light_;
  flex_     = other.flex_;
  mesh_     = other.mesh_;
  texture_  = other.texture_;
  material_ = other.material_;
  pair_     = other.pair_;
  equality_ = other.equality_;
  tendon_   = other.tendon_;
  actuator_ = other.actuator_;
  PointToLocal();
}


void mjCDef::PointToLocal() {
  joint_.PointToLocal();
  geom_.PointToLocal();
  site_.PointToLocal();
  camera_.PointToLocal();
  light_.PointToLocal();
  flex_.PointToLocal();
  mesh_.PointToLocal();
  texture_.PointToLocal();
  material_.PointToLocal();
  pair_.PointToLocal();
  equality_.PointToLocal();
  tendon_.PointToLocal();
  actuator_.PointToLocal();
  spec.element  = static_cast<mjsElement*>(this);
  spec.joint    = &joint_.spec;
  spec.geom     = &geom_.spec;
  spec.site     = &site_.spec;
  spec.camera   = &camera_.spec;
  spec.light    = &light_.spec;
  spec.flex     = &flex_.spec;
  spec.mesh     = &mesh_.spec;
  spec.texture  = &texture_.spec;
  spec.material = &material_.spec;
  spec.pair     = &pair_.spec;
  spec.equality = &equality_.spec;
  spec.tendon   = &tendon_.spec;
  spec.actuator = &actuator_.spec;
}


void mjCDef::CopyFromSpec() {
  joint_.CopyFromSpec();
  geom_.CopyFromSpec();
  site_.CopyFromSpec();
  camera_.CopyFromSpec();
  light_.CopyFromSpec();
  flex_.CopyFromSpec();
  mesh_.CopyFromSpec();
  texture_.CopyFromSpec();
  material_.CopyFromSpec();
  pair_.CopyFromSpec();
  equality_.CopyFromSpec();
  tendon_.CopyFromSpec();
  actuator_.CopyFromSpec();
}


//------------------------- class mjCBase implementation -------------------------------------------

// constructor
mjCBase::mjCBase() {
  name.clear();
  classname.clear();
  id       = -1;
  info     = "";
  model    = 0;
  frame    = nullptr;
  compiler = nullptr;
}


mjCBase::mjCBase(const mjCBase& other) {
  *this = other;
}


mjCBase& mjCBase::operator=(const mjCBase& other) {
  if (this != &other) { *static_cast<mjCBase_*>(this) = static_cast<const mjCBase_&>(other); }
  return *this;
}


void mjCBase::NameSpace(const mjCModel* m) {
  if (!name.empty()) { name = m->prefix + name + m->suffix; }
  if (!classname.empty() && classname != "main" && m != model) {
    classname = m->prefix + classname + m->suffix;
  }
}


mjsCompiler* mjCBase::FindCompiler(const mjsCompiler* compiler) const {
  mjSpec* origin = model->FindSpec(compiler);
  return origin ? &origin->compiler : &model->spec.compiler;
}


// load resource if found (fallback to OS filesystem)
mjResource* mjCBase::LoadResource(const std::string& modelfiledir,
                                  const std::string& filename,
                                  const mjVFS*       vfs) {
  // try reading from provided VFS or fallback to OS filesystem
  std::array<char, 1024> error;
  mjResource*            resource =
      mju_openResource(modelfiledir.c_str(), filename.c_str(), vfs, error.data(), error.size());
  if (!resource) { throw mjCError(nullptr, "%s", error.data()); }
  return resource;
}


// path of an asset file as LoadResource opens it: modelfiledir + dir + file
std::string mjCBase::ResolvePath(const std::string& dir, const std::string& file) const {
  mjSpec* owning_spec = model->FindSpec(compiler);
  if (!owning_spec) { owning_spec = &model->spec; }
  FilePath filename = FilePath(dir) + FilePath(file);
  return FilePath(mjs_getString(owning_spec->modelfiledir), filename.Str()).Str();
}


// Get and sanitize content type from raw_text if not empty, otherwise parse
// content type from resource_name; throw error on failure
std::string mjCBase::GetAssetContentType(std::string_view resource_name,
                                         std::string_view raw_text) {
  if (!raw_text.empty()) {
    auto type    = mjuu_parseContentTypeAttrType(raw_text);
    auto subtype = mjuu_parseContentTypeAttrSubtype(raw_text);
    if (!type.has_value() || !subtype.has_value()) { return ""; }
    return std::string(*type) + "/" + std::string(*subtype);
  } else {
    return mjuu_extToContentType(resource_name);
  }
}


// the file of an asset of the spec as compilation resolves it: the directory of the spec which
// owns the asset, the asset directory of its compiler settings, then the file, stripped of its path
// if the compiler settings ask for it; empty if there is no file
std::string mjCBase::ResolvedSpecFile(const std::string& dir, std::string file) const {
  if (file.empty()) { return ""; }
  if (model->spec.strippath) { file = mjuu_strippath(file); }
  return ResolvePath(dir, file);
}


void mjCBase::SetFrame(mjCFrame* _frame) {
  if (!_frame) { return; }
  if (_frame->body && GetParent() != _frame->body) {
    throw mjCError(this, "Frame and body '%s' have mismatched parents", name.c_str());
  }
  frame = _frame;
}


// sine of the largest angle by which a scaled axis may turn and still count as aligned
static constexpr double kAlignTol = 0.0349;


// product of 3x3 matrices
static void MulMat3(double res[9], const double a[9], const double b[9]) {
  double tmp[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tmp[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    }
  }
  mjuu_copyvec(res, tmp, 9);
}


// rotation matrix of a quaternion, which need not be normalized
static void QuatMat(double mat[9], const double quat[4]) {
  double q[4] = {quat[0], quat[1], quat[2], quat[3]};
  mjuu_normvec(q, 4);
  mjuu_quat2mat(mat, q);
}


// R^T M R
static void Conjugate(double res[9], const double M[9], const double R[9]) {
  double RT[9] = {R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]}, tmp[9];
  MulMat3(tmp, RT, M);
  MulMat3(res, tmp, R);
}


// gauge of a frame scale: a change of length and mass units, or of shape and inertia only; lengths
// without a direction take the geometric mean of the three
mjCGauge mjCGauge::Of(const double scale[3], mjtScaling scaling, double massscale) {
  mjCGauge g;
  g.uniform = scale[0] == scale[1] && scale[1] == scale[2];
  g.shape   = g.uniform ? scale[0] : std::cbrt(scale[0] * scale[1] * scale[2]);
  g.inertia = massscale > 0 ? massscale : scale[0] * scale[1] * scale[2];
  if (scaling != mjSCALING_GEOMETRY) {
    g.length = g.shape;
    g.mass   = g.inertia;
  }
  for (int i = 0; i < 3; i++) { g.map[4 * i] = scale[i]; }
  return g;
}


// factor of a shape-class quantity of dimension L^a M^b, without a direction
double mjCGauge::Shape(double a, double b) const {
  return std::pow(shape, a) * std::pow(inertia, b);
}


// factor of a force-class quantity of dimension L^a M^b
double mjCGauge::Force(double a, double b) const {
  return std::pow(length, a) * std::pow(mass, b);
}


// scale a position or displacement
void mjCGauge::Point(double x[3]) const {
  if (uniform) {
    for (int i = 0; i < 3; i++) { x[i] *= shape; }
  } else {
    mjuu_mulvecmat(x, x, map);
  }
}


// length factor along a direction
double mjCGauge::Stretch(const double dir[3]) const {
  if (uniform) { return shape; }
  double v[3];
  mjuu_mulvecmat(v, dir, map);
  return std::sqrt(mjuu_dot3(v, v) / mjuu_dot3(dir, dir));
}


// true if the map keeps a direction, within the alignment tolerance
bool mjCGauge::Aligned(const double dir[3]) const {
  if (uniform) { return true; }
  double v[3], c[3];
  mjuu_mulvecmat(v, dir, map);
  mjuu_crossvec(c, v, dir);
  return std::sqrt(mjuu_dot3(c, c)) <= kAlignTol * std::sqrt(mjuu_dot3(v, v) * mjuu_dot3(dir, dir));
}


// scale the size of a geom or site of the given type and orientation; returns an error message if
// its shape cannot represent its scaled self; sites instead stretch along their own axes, and
// sphere sites take the mean factor
const char* mjCGauge::Size(double size[3], int type, const double quat[4], bool site) const {
  int n = type >= 0 && type < mjNGEOMTYPES ? mjGEOMINFO[type] : 3;
  if (uniform) {
    for (int i = 0; i < n; i++) { size[i] *= shape; }
    return nullptr;
  }
  if (type == mjGEOM_MESH || type == mjGEOM_SDF) { return nullptr; }
  if (type == mjGEOM_SPHERE && site) {
    size[0] *= shape;
    return nullptr;
  }

  // the map in the frame of the shape: the factor and the alignment of each of its axes
  double R[9], local[9], f[3];
  bool   aligned[3];
  QuatMat(R, quat);
  Conjugate(local, map, R);
  for (int i = 0; i < 3; i++) {
    double off = 0;
    for (int j = 0; j < 3; j++) { off += j == i ? 0 : local[3 * j + i] * local[3 * j + i]; }
    f[i]       = off ? std::sqrt(off + local[4 * i] * local[4 * i]) : local[4 * i];
    aligned[i] = std::sqrt(off) <= kAlignTol * f[i];
  }
  auto equal = [](double a, double b) { return std::abs(a - b) <= kAlignTol * std::max(a, b); };

  switch (type) {
    case mjGEOM_SPHERE:
      if (!aligned[0] || !aligned[1] || !aligned[2] || !equal(f[0], f[1]) || !equal(f[1], f[2])) {
        return aligned[0] && aligned[1] && aligned[2]
                   ? "a sphere can only be scaled uniformly; use an ellipsoid instead"
                   : "a sphere can only be scaled uniformly";
      }
      size[0] *= std::cbrt(f[0] * f[1] * f[2]);
      return nullptr;

    case mjGEOM_CAPSULE:
    case mjGEOM_CYLINDER:
      if (!site && !aligned[2]) {
        return "the axis of this shape is not aligned with the axes of the scale";
      }
      if (!site && (!aligned[0] || !aligned[1] || !equal(f[0], f[1]))) {
        return "this shape can only be scaled equally across its axis";
      }
      size[0] *= std::sqrt(f[0] * f[1]);
      size[1] *= f[2];
      return nullptr;

    case mjGEOM_PLANE:
      if (!aligned[0] || !aligned[1]) {
        return "a plane is not aligned with the axes of the scale";
      }
      size[0] *= f[0];
      size[1] *= f[1];
      size[2] *= shape;
      return nullptr;

    default:
      if (!site && (!aligned[0] || !aligned[1] || !aligned[2])) {
        return "this shape is not aligned with the axes of the scale";
      }
      for (int i = 0; i < n; i++) { size[i] *= f[i]; }
      return nullptr;
  }
}


// scale a mass and the diagonal or, if defined, the full inertia, given in the frame of iquat; a
// map which is not uniform gives the full inertia
void mjCGauge::Inertia(double&      mass_,
                       double       inertia_[3],
                       double       fullinertia[6],
                       const double iquat[4]) const {
  bool full = mjuu_defined(fullinertia[0]);
  if (uniform) {
    mass_ *= Shape(0, 1);
    for (int i = 0; i < 3; i++) { inertia_[i] *= Shape(2, 1); }
    if (full) {
      for (int i = 0; i < 6; i++) { fullinertia[i] *= Shape(2, 1); }
    }
    return;
  }

  // inertia tensor in the frame of iquat
  double I[9] = {0};
  if (full) {
    I[0] = fullinertia[0];
    I[4] = fullinertia[1];
    I[8] = fullinertia[2];
    I[1] = I[3] = fullinertia[3];
    I[2] = I[6] = fullinertia[4];
    I[5] = I[7] = fullinertia[5];
  } else {
    for (int i = 0; i < 3; i++) { I[4 * i] = inertia_[i]; }
  }
  mass_ *= Shape(0, 1);
  if (!full && !inertia_[0] && !inertia_[1] && !inertia_[2]) { return; }

  // the second moments of the mass, Sigma = tr(I)/2 - I, are mapped as A Sigma A^T
  double R[9], A[9], AT[9], sigma[9], tmp[9];
  QuatMat(R, iquat);
  Conjugate(A, map, R);
  double half = (I[0] + I[4] + I[8]) / 2;
  for (int i = 0; i < 9; i++) { sigma[i] = (i % 4 ? 0 : half) - I[i]; }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { AT[3 * i + j] = A[3 * j + i]; }
  }
  MulMat3(tmp, A, sigma);
  MulMat3(sigma, tmp, AT);
  double trace = (sigma[0] + sigma[4] + sigma[8]) * Shape(0, 1);
  for (int i = 0; i < 9; i++) { I[i] = (i % 4 ? 0 : trace) - sigma[i] * Shape(0, 1); }
  fullinertia[0] = I[0];
  fullinertia[1] = I[4];
  fullinertia[2] = I[8];
  fullinertia[3] = (I[1] + I[3]) / 2;
  fullinertia[4] = (I[2] + I[6]) / 2;
  fullinertia[5] = (I[5] + I[7]) / 2;
  inertia_[0] = inertia_[1] = inertia_[2] = 0;
}


// the gauge in the coordinates of a frame with the given orientation
mjCGauge mjCGauge::Rotated(const double quat[4]) const {
  if (uniform) { return *this; }
  mjCGauge res = *this;
  double   R[9];
  QuatMat(R, quat);
  Conjugate(res.map, map, R);
  return res;
}


// inverse gauge, which scales back
mjCGauge mjCGauge::Inverse() const {
  mjCGauge res;
  res.shape   = 1 / shape;
  res.length  = 1 / length;
  res.mass    = 1 / mass;
  res.inertia = 1 / inertia;
  res.uniform = uniform;
  if (uniform) {
    for (int i = 0; i < 3; i++) { res.map[4 * i] = res.shape; }
    return res;
  }
  const double* m   = map;
  double        det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                      m[1] * (m[3] * m[8] - m[5] * m[6]) +
                      m[2] * (m[3] * m[7] - m[4] * m[6]);
  res.map[0]        = (m[4] * m[8] - m[5] * m[7]) / det;
  res.map[1]        = (m[2] * m[7] - m[1] * m[8]) / det;
  res.map[2]        = (m[1] * m[5] - m[2] * m[4]) / det;
  res.map[3]        = (m[5] * m[6] - m[3] * m[8]) / det;
  res.map[4]        = (m[0] * m[8] - m[2] * m[6]) / det;
  res.map[5]        = (m[2] * m[3] - m[0] * m[5]) / det;
  res.map[6]        = (m[3] * m[7] - m[4] * m[6]) / det;
  res.map[7]        = (m[1] * m[6] - m[0] * m[7]) / det;
  res.map[8]        = (m[0] * m[4] - m[1] * m[3]) / det;
  return res;
}


// composition of gauges along an ancestry: this one, of the enclosing frame or body, after the
// other
mjCGauge mjCGauge::operator*(const mjCGauge& other) const {
  mjCGauge res;
  res.shape   = shape * other.shape;
  res.length  = length * other.length;
  res.mass    = mass * other.mass;
  res.inertia = inertia * other.inertia;
  res.uniform = uniform && other.uniform;
  if (res.uniform) {
    for (int i = 0; i < 3; i++) { res.map[4 * i] = res.shape; }
  } else {
    MulMat3(res.map, map, other.map);
  }
  return res;
}


// geometric mean of the gauges of two participants, for quantities without a direction
mjCGauge mjCGauge::Mean(const mjCGauge& a, const mjCGauge& b) {
  mjCGauge res;
  res.shape   = std::sqrt(a.shape * b.shape);
  res.length  = std::sqrt(a.length * b.length);
  res.mass    = std::sqrt(a.mass * b.mass);
  res.inertia = std::sqrt(a.inertia * b.inertia);
  for (int i = 0; i < 3; i++) { res.map[4 * i] = res.shape; }
  return res;
}


// the frame or body which directly contains an element or a container
static const mjCBase* Container(const mjCBase* element) {
  return element->frame ? element->frame : element->GetParent();
}


// gauge in force inside the lowest frame or body which contains all the given elements; a body
// contains itself
mjCGauge LowestContainerGauge(const std::vector<const mjCBase*>& elements) {
  if (elements.empty()) { return mjCGauge(); }
  auto start = [](const mjCBase* e) { return e->elemtype == mjOBJ_BODY ? e : Container(e); };
  for (const mjCBase* c = start(elements[0]); c; c = Container(c)) {
    bool all = true;
    for (int i = 1; i < elements.size() && all; i++) {
      const mjCBase* d = start(elements[i]);
      while (d && d != c) { d = Container(d); }
      all = d == c;
    }
    if (all) { return c->ContentGauge(); }
  }
  return mjCGauge();
}


// gauge of an element: that of the contents of its innermost frame or else its parent
mjCGauge mjCBase::Gauge() const {
  if (frame) { return frame->ContentGauge(); }
  const mjCBase* parent = GetParent();
  return parent ? parent->ContentGauge() : mjCGauge();
}


void mjCBase::SetUserValue(std::string_view key, const void* data, void (*cleanup)(const void*)) {
  user_payload_[std::string(key)] = UserValue(data, cleanup);
}

const void* mjCBase::GetUserValue(std::string_view key) {
  auto found = user_payload_.find(std::string(key));
  return found != user_payload_.end() ? found->second.value : nullptr;
}


void mjCBase::DeleteUserValue(std::string_view key) {
  user_payload_.erase(std::string(key));
}


//------------------ class mjCBody implementation --------------------------------------------------

// constructor
mjCBody::mjCBody(mjCModel* _model) {
  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  refcount = 1;
  mjs_defaultBody(&spec);
  elemtype    = mjOBJ_BODY;
  parent      = nullptr;
  iframe      = nullptr;
  weldid      = -1;
  dofnum      = 0;
  lastdof     = -1;
  subtreedofs = 0;
  mjuu_zerovec(xpos0, 3);
  mjuu_setvec(xquat0, 1, 0, 0, 0);
  last_attached = nullptr;
  mocapid       = -1;
  bodyadr_      = -1;

  // clear object lists
  bodies.clear();
  geoms.clear();
  frames.clear();
  joints.clear();
  sites.clear();
  cameras.clear();
  lights.clear();
  spec_userdata_.clear();

  // in case this body is not compiled
  CopyFromSpec();

  // point to local (needs to be after defaults)
  PointToLocal();
}


mjCBody::mjCBody(const mjCBody& other, mjCModel* _model) {
  model    = _model;
  compiler = FindCompiler(other.compiler);
  *this    = other;
  CopyPlugin();
}


mjCBody& mjCBody::operator=(const mjCBody& other) {
  if (this != &other) {
    spec                          = other.spec;
    *static_cast<mjCBody_*>(this) = static_cast<const mjCBody_&>(other);
    *static_cast<mjsBody*>(this)  = static_cast<const mjsBody&>(other);
    bodies.clear();
    frames.clear();
    geoms.clear();
    joints.clear();
    sites.clear();
    cameras.clear();
    lights.clear();
    id          = -1;
    mocapid     = -1;
    bodyadr_    = -1;
    subtreedofs = 0;

    // add elements to lists
    *this += other;

    // point to the copy of the frame enclosing the inertial element
    iframe = nullptr;
    for (int i = 0; i < other.frames.size(); i++) {
      if (other.frames[i] == other.iframe) { iframe = frames[i]; }
    }
  }
  PointToLocal();
  return *this;
}


// copy children of other body into body
mjCBody& mjCBody::operator+=(const mjCBody& other) {
  // map other frames to indices
  std::map<mjCFrame*, int> fmap;
  for (int i = 0; i < other.frames.size(); i++) { fmap[other.frames[i]] = i + frames.size(); }

  // copy frames, needs to happen first
  CopyList(frames, other.frames, fmap);

  // copy all children
  CopyList(geoms, other.geoms, fmap);
  CopyList(joints, other.joints, fmap);
  CopyList(sites, other.sites, fmap);
  CopyList(cameras, other.cameras, fmap);
  CopyList(lights, other.lights, fmap);

  for (int i = 0; i < other.bodies.size(); i++) {
    bodies.push_back(new mjCBody(*other.bodies[i], model));  // triggers recursive call
    bodies.back()->parent = this;
    bodies.back()->frame  = nullptr;
    if (other.bodies[i]->frame) {
      if (fmap.find(other.bodies[i]->frame) != fmap.end()) {
        bodies.back()->frame = frames[fmap[other.bodies[i]->frame]];
      } else {
        throw mjCError(this,
                       "Frame '%s' not found in other body",
                       other.bodies[i]->frame->name.c_str());
      }
      if (bodies.back()->frame && bodies.back()->frame->body != this) {
        throw mjCError(this, "Frame and body '%s' have mismatched parents", name.c_str());
      }
    }
  }

  return *this;
}


// attach frame to body
mjCBody& mjCBody::operator+=(const mjCFrame& other) {
  // append a copy of the attached spec
  if (other.model != model && !model->FindSpec(&other.model->spec.compiler)) {
    model->AppendSpec(&other.model->spec, &other.model->spec.compiler);
    static_cast<mjCModel*>(other.model->spec.element)->AddRef();
  }

  // the tree of this model is about to change: store its keyframes, before the other model lays
  // out its own, which changes the addresses of the elements they share
  if (other.model != model) { model->StoreKeyframes(nullptr); }

  // create a copy of the subtree that contains the frame
  mjCBody* subtree    = other.body;
  other.model->prefix = other.prefix;
  other.model->suffix = other.suffix;
  other.model->StoreKeyframes(model);
  mjCModel* other_model = other.model;

  // attach defaults
  if (other_model != model) {
    mjCDef* subdef = new mjCDef(*other_model->Default());
    subdef->NameSpace(other_model);
    *model += *subdef;
  }

  // copy input frame
  mjSpec*   origin = model->FindSpec(other.compiler);
  mjCFrame* newframe(model->deepcopy_ ? new mjCFrame(other) : (mjCFrame*)&other);
  frames.push_back(newframe);
  frames.back()->body     = this;
  frames.back()->model    = model;
  frames.back()->compiler = origin ? &origin->compiler : &model->spec.compiler;
  frames.back()->frame    = nullptr;  // posed in this body, enclosing frames stay behind
  frames.back()->NameSpace(other_model);
  if (!model->deepcopy_) { frames.back()->AddRef(); }
  int i         = frames.size();
  last_attached = &frames.back()->spec;

  // map input frames to index in this->frames
  std::map<mjCFrame*, int> fmap;
  for (auto frame : subtree->frames) {
    if (frame == static_cast<const mjCFrame*>(&other)) {
      fmap[frame] = frames.size() - 1;
    } else if (other.IsAncestor(frame)) {
      fmap[frame] = i++;
    }
  }

  // copy children that are inside the input frame
  CopyList(frames, subtree->frames, fmap, &other);  // needs to be done first
  CopyList(geoms, subtree->geoms, fmap, &other);
  CopyList(joints, subtree->joints, fmap, &other);
  CopyList(sites, subtree->sites, fmap, &other);
  CopyList(cameras, subtree->cameras, fmap, &other);
  CopyList(lights, subtree->lights, fmap, &other);

  int nbodies = (int)subtree->bodies.size();
  for (int i = 0; i < nbodies; i++) {
    if (!other.IsAncestor(subtree->bodies[i]->frame)) { continue; }
    if (model->deepcopy_) {
      mjCBody* newbody(new mjCBody(*subtree->bodies[i], model));  // triggers recursive call
      bodies.push_back(newbody);
      bodies.back()->NameSpace_(other_model, /*propagate=*/false);
    } else {
      bodies.push_back(subtree->bodies[i]);
      bodies.back()->SetModel(model);
      bodies.back()->ResetId();
      bodies.back()->AddRef();
      bodies.back()->NameSpace(other_model);
    }

    // the attached body does not take the values of the keyframes which stay in place in the
    // other model; a copy leaves them, and only them, in the original
    bodies.back()->ForgetKeyframes(other_model->inplacekeys_);
    if (model->deepcopy_) {
      subtree->bodies[i]->ForgetKeyframes(other_model->inplacekeys_, /*keep=*/true);
    }
    bodies.back()->parent = this;
    bodies.back()->frame =
        subtree->bodies[i]->frame ? frames[fmap[subtree->bodies[i]->frame]] : nullptr;
  }

  // attach referencing elements
  other_model->SetAttached(model->deepcopy_);
  *model += *other_model;

  // clear namespace and return body
  other_model->prefix.clear();
  other_model->suffix.clear();
  return *this;
}


// copy src list of elements into dst; set body, model and frame
template <typename T>
void mjCBody::CopyList(std::vector<T*>&          dst,
                       const std::vector<T*>&    src,
                       std::map<mjCFrame*, int>& fmap,
                       const mjCFrame*           pframe) {
  int nsrc = (int)src.size();
  int ndst = (int)dst.size();
  for (int i = 0; i < nsrc; i++) {
    if (pframe && !pframe->IsAncestor(src[i]->frame)) {
      continue;  // skip if the element is not inside pframe
    }
    mjSpec*   origin       = model->FindSpec(src[i]->compiler);
    mjCModel* source_model = src[i]->model;
    T*        new_obj      = model->deepcopy_ ? new T(*src[i]) : src[i];

    // an attached element does not take the values of the keyframes which stay in place
    if (pframe) { new_obj->ForgetKeyframes(src[i]->model->inplacekeys_); }
    dst.push_back(new_obj);
    dst.back()->body     = this;
    dst.back()->model    = model;
    dst.back()->compiler = origin ? &origin->compiler : &model->spec.compiler;
    dst.back()->id       = -1;
    dst.back()->CopyPlugin();
    dst.back()->classname = src[i]->classname;

    // increment refcount if shallow copy is made; the moved element forgets its addresses in the
    // model it comes from
    if (!model->deepcopy_) {
      dst.back()->AddRef();
      dst.back()->ResetId();
    }

    // set namespace; an element moved by reference was given this model above
    dst.back()->NameSpace(source_model);
  }

  // assign dst frame to src frame
  // needs to be done after the copy in case T is an mjCFrame
  int j = 0;
  for (int i = 0; i < src.size(); i++) {
    if (pframe && !pframe->IsAncestor(src[i]->frame)) {
      continue;  // skip if the element is not inside pframe
    }
    dst[ndst + j++]->frame = src[i]->frame ? frames[fmap[src[i]->frame]] : nullptr;
  }
}


// find and remove subtree
mjCBody& mjCBody::operator-=(const mjCBody& subtree) {
  for (int i = 0; i < bodies.size(); i++) {
    if (bodies[i] == &subtree) {
      bodies[i]->SetParent(nullptr);
      bodies[i]->frame = nullptr;
      bodies.erase(bodies.begin() + i);
      break;
    }
    *bodies[i] -= subtree;
  }

  return *this;
}


// set model of this body and its subtree
void mjCBody::SetModel(mjCModel* _model) {
  model          = _model;
  mjSpec* origin = _model->FindSpec(compiler);
  compiler       = origin ? &origin->compiler : &model->spec.compiler;

  for (auto& body : bodies) { body->SetModel(_model); }
  for (auto& frame : frames) {
    origin          = _model->FindSpec(frame->compiler);
    frame->model    = _model;
    frame->compiler = origin ? &origin->compiler : &model->spec.compiler;
  }
  for (auto& geom : geoms) {
    origin         = _model->FindSpec(geom->compiler);
    geom->model    = _model;
    geom->compiler = origin ? &origin->compiler : &model->spec.compiler;
  }
  for (auto& joint : joints) {
    origin          = _model->FindSpec(joint->compiler);
    joint->model    = _model;
    joint->compiler = origin ? &origin->compiler : &model->spec.compiler;
  }
  for (auto& site : sites) {
    origin         = _model->FindSpec(site->compiler);
    site->model    = _model;
    site->compiler = origin ? &origin->compiler : &model->spec.compiler;
  }
  for (auto& camera : cameras) {
    origin           = _model->FindSpec(camera->compiler);
    camera->model    = _model;
    camera->compiler = origin ? &origin->compiler : &model->spec.compiler;
  }
  for (auto& light : lights) {
    origin          = _model->FindSpec(light->compiler);
    light->model    = _model;
    light->compiler = origin ? &origin->compiler : &model->spec.compiler;
  }
}


// reset ids of all objects in this body
void mjCBody::ResetId() {
  id       = -1;
  mocapid  = -1;
  bodyadr_ = -1;
  for (auto& body : bodies) { body->ResetId(); }
  for (auto& frame : frames) { frame->id = -1; }
  for (auto& geom : geoms) { geom->id = -1; }
  for (auto& joint : joints) { joint->ResetId(); }
  for (auto& site : sites) { site->id = -1; }
  for (auto& camera : cameras) { camera->id = -1; }
  for (auto& light : lights) { light->id = -1; }
}


void mjCBody::PointToLocal() {
  spec.element            = static_cast<mjsElement*>(this);
  spec.childclass         = &classname;
  spec.userdata           = &spec_userdata_;
  spec.plugin.plugin_name = &plugin_name;
  spec.plugin.name        = (&plugin_instance_name);
  spec.info               = &info;
  userdata                = nullptr;
}


void mjCBody::CopyFromSpec() {
  *static_cast<mjsBody*>(this) = spec;

  userdata_          = spec_userdata_;
  plugin.active      = spec.plugin.active;
  plugin.element     = spec.plugin.element;
  plugin.plugin_name = spec.plugin.plugin_name;
  plugin.name        = spec.plugin.name;
}


void mjCBody::CopyPlugin() {
  model->CopyExplicitPlugin(this);
}


// destructor
mjCBody::~mjCBody() {
  auto release_children = [](auto& list) {
    for (auto* child : list) {
      child->SetParent(nullptr);
      child->frame = nullptr;
      child->Release();
    }
  };
  release_children(bodies);
  release_children(geoms);
  release_children(frames);
  release_children(joints);
  release_children(sites);
  release_children(cameras);
  release_children(lights);
}


// apply prefix and suffix, propagate to children
void mjCBody::NameSpace(const mjCModel* m) {
  NameSpace_(m, true);
}


// apply prefix and suffix, propagate to all descendants or only to child bodies
void mjCBody::NameSpace_(const mjCModel* m, bool propagate) {
  mjCBase::NameSpace(m);
  if (!plugin_instance_name.empty()) {
    plugin_instance_name = m->prefix + plugin_instance_name + m->suffix;
  }

  for (auto& body : bodies) {
    body->prefix = m->prefix;
    body->suffix = m->suffix;
    body->NameSpace_(m, propagate);
  }

  if (!propagate) { return; }

  for (auto& joint : joints) { joint->NameSpace(m); }

  for (auto& geom : geoms) { geom->NameSpace(m); }

  for (auto& site : sites) { site->NameSpace(m); }

  for (auto& camera : cameras) { camera->NameSpace(m); }

  for (auto& light : lights) { light->NameSpace(m); }

  for (auto& frame : frames) { frame->NameSpace(m); }
}


// create child body and add it to body
mjCBody* mjCBody::AddBody(mjCDef* _def) {
  // create body
  mjCBody* obj = new mjCBody(model);

  // handle def recursion (i.e. childclass)
  obj->classname = _def ? _def->name : classname;

  bodies.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  obj->parent = this;

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create new frame and add it to body
mjCFrame* mjCBody::AddFrame(mjCFrame* _frame) {
  mjCFrame* obj = new mjCFrame(model, _frame ? _frame : NULL);

  // set body pointer, add
  obj->body = this;

  frames.push_back(obj);
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create new free joint (no default inheritance) and add it to body
mjCJoint* mjCBody::AddFreeJoint() {
  // create free joint, don't inherit from defaults
  mjCJoint* obj  = new mjCJoint(model, NULL);
  obj->spec.type = mjJNT_FREE;

  // set body pointer, add
  obj->body = this;

  joints.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// find default class by name without inserting missing keys
static mjCDef* FindClassDef(const mjCModel* model, const std::string& classname) {
  auto it = model->def_map.find(classname);
  return it != model->def_map.end() ? it->second : nullptr;
}


// create new joint and add it to body
mjCJoint* mjCBody::AddJoint(mjCDef* _def) {
  // create joint
  mjCJoint* obj = new mjCJoint(model, _def ? _def : FindClassDef(model, classname));

  // set body pointer, add
  obj->body = this;

  joints.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create new geom and add it to body
mjCGeom* mjCBody::AddGeom(mjCDef* _def) {
  // create geom
  mjCGeom* obj = new mjCGeom(model, _def ? _def : FindClassDef(model, classname));

  //  set body pointer, add
  obj->body = this;

  geoms.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create new site and add it to body
mjCSite* mjCBody::AddSite(mjCDef* _def) {
  // create site
  mjCSite* obj = new mjCSite(model, _def ? _def : FindClassDef(model, classname));

  // set body pointer, add
  obj->body = this;

  sites.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create new camera and add it to body
mjCCamera* mjCBody::AddCamera(mjCDef* _def) {
  // create camera
  mjCCamera* obj = new mjCCamera(model, _def ? _def : FindClassDef(model, classname));

  // set body pointer, add
  obj->body = this;

  cameras.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create new light and add it to body
mjCLight* mjCBody::AddLight(mjCDef* _def) {
  // create light
  mjCLight* obj = new mjCLight(model, _def ? _def : FindClassDef(model, classname));

  // set body pointer, add
  obj->body = this;

  lights.push_back(obj);

  // recompute lists
  model->ResetTreeLists();
  model->MakeTreeLists();

  // invalidate signature
  model->InvalidateSignature();
  return obj;
}


// create a frame in the parent body and move all contents of this body into it
mjCFrame* mjCBody::ToFrame(bool mergeinertial) {
  if (!parent) { throw mjCError(this, "the world body cannot be converted to a frame"); }

  // merge the inertial into the parent; this can fail, do it before anything is modified
  if (mergeinertial && parent->name != "world") {
    // an inertial is either given in the spec, or inferred from geoms when compiling
    bool inferred      = InfersInertial() && !geoms.empty();
    bool given         = !inferred && mjuu_defined(spec.ipos[0]) && spec.mass >= mjMINVAL;
    bool parent_infers = parent->InfersInertial() && !parent->geoms.empty();
    bool parent_given  = !parent_infers && mjuu_defined(parent->spec.ipos[0]);
    bool scaled        = ScaledInertial() || parent->ScaledInertial();

    // a body which does not infer its inertial gives no mass to its geoms, e.g. with a massless
    // inertial: a parent which infers its inertial from geoms must not gain theirs, so the inertia
    // it has now is written to its spec, as when fusing static bodies
    bool keepparent = !InfersInertial() && !given && !geoms.empty() && parent->InfersInertial();

    // one is given and the other inferred: compile the tree to know what is inferred; if both
    // are inferred, the geoms move to the parent and take the inertia with them. Scaled
    // inertials are merged as compiled
    if ((given && parent_infers) || (inferred && parent_given) || (given && scaled) || keepparent) {
      if (!model->Resolve(nullptr)) {
        std::string error = model->GetError().message;
        throw mjCError(this,
                       "inertia inferred from geoms could not be calculated; if assets are in a "
                       "VFS, call mjs_adoptInertial first: %s",
                       error.c_str());
      }
      if (keepparent) {
        parent->AdoptInertial();
      } else if (given || mass >= mjMINVAL) {
        parent->MergeInertial(this);
      }
    } else if (given) {
      parent->MergeInertial(this);
    }
  }

  mjCFrame* newframe = parent->AddFrame(frame);
  mjuu_copyvec(newframe->spec.pos, spec.pos, 3);
  mjuu_copyvec(newframe->spec.quat, spec.quat, 4);
  newframe->spec.alt = spec.alt;
  MapFrame(parent->bodies, bodies, newframe, parent);
  MapFrame(parent->geoms, geoms, newframe, parent);
  MapFrame(parent->joints, joints, newframe, parent);
  MapFrame(parent->sites, sites, newframe, parent);
  MapFrame(parent->cameras, cameras, newframe, parent);
  MapFrame(parent->lights, lights, newframe, parent);
  MapFrame(parent->frames, frames, newframe, parent);
  parent->bodies.erase(std::remove_if(parent->bodies.begin(),
                                      parent->bodies.end(),
                                      [this](mjCBody* body) { return body == this; }),
                       parent->bodies.end());
  model->ResetTreeLists();
  model->MakeTreeLists();
  model->InvalidateSignature();
  return newframe;
}


// get number of objects of specified type
int mjCBody::NumObjects(mjtObj type) {
  switch (type) {
    case mjOBJ_BODY:
    case mjOBJ_XBODY:
      return (int)bodies.size();
    case mjOBJ_JOINT:
      return (int)joints.size();
    case mjOBJ_GEOM:
      return (int)geoms.size();
    case mjOBJ_SITE:
      return (int)sites.size();
    case mjOBJ_CAMERA:
      return (int)cameras.size();
    case mjOBJ_LIGHT:
      return (int)lights.size();
    default:
      return 0;
  }
}


// get poiner to specified object
mjCBase* mjCBody::GetObject(mjtObj type, int i) {
  if (i >= 0 && i < NumObjects(type)) {
    switch (type) {
      case mjOBJ_BODY:
      case mjOBJ_XBODY:
        return bodies[i];
      case mjOBJ_JOINT:
        return joints[i];
      case mjOBJ_GEOM:
        return geoms[i];
      case mjOBJ_SITE:
        return sites[i];
      case mjOBJ_CAMERA:
        return cameras[i];
      case mjOBJ_LIGHT:
        return lights[i];
      default:
        return 0;
    }
  }

  return 0;
}


// find object by name in given list
template <class T>
static T* findobject(const std::string& name, const std::vector<T*>& list) {
  for (unsigned int i = 0; i < list.size(); i++) {
    if (list[i]->name == name) { return list[i]; }
  }

  return 0;
}


// recursive find by name
mjCBase* mjCBody::FindObject(mjtObj type, const std::string& _name, bool recursive) const {
  mjCBase* res = 0;

  // check self: only for world body when searching for a body
  if (!parent && (type == mjOBJ_BODY || type == mjOBJ_XBODY) && name == _name) {
    return const_cast<mjCBody*>(this);
  }

  // search elements of this body
  if (type == mjOBJ_BODY || type == mjOBJ_XBODY) {
    res = findobject(_name, bodies);
  } else if (type == mjOBJ_JOINT) {
    res = findobject(_name, joints);
  } else if (type == mjOBJ_GEOM) {
    res = findobject(_name, geoms);
  } else if (type == mjOBJ_SITE) {
    res = findobject(_name, sites);
  } else if (type == mjOBJ_CAMERA) {
    res = findobject(_name, cameras);
  } else if (type == mjOBJ_LIGHT) {
    res = findobject(_name, lights);
  }

  // found
  if (res) { return res; }

  // search children
  if (recursive) {
    for (int i = 0; i < (int)bodies.size(); i++) {
      if ((res = bodies[i]->FindObject(type, _name, true))) { return res; }
    }
  }

  // not found
  return res;
}


// get list of a given type
template <>
const std::vector<mjCBody*>& mjCBody::GetList<mjCBody>() const {
  return bodies;
}

template <>
const std::vector<mjCJoint*>& mjCBody::GetList<mjCJoint>() const {
  return joints;
}

template <>
const std::vector<mjCGeom*>& mjCBody::GetList<mjCGeom>() const {
  return geoms;
}

template <>
const std::vector<mjCSite*>& mjCBody::GetList<mjCSite>() const {
  return sites;
}

template <>
const std::vector<mjCCamera*>& mjCBody::GetList<mjCCamera>() const {
  return cameras;
}

template <>
const std::vector<mjCLight*>& mjCBody::GetList<mjCLight>() const {
  return lights;
}

template <>
const std::vector<mjCFrame*>& mjCBody::GetList<mjCFrame>() const {
  return frames;
}


// gets next child of the same type, recursively depth first if requested
template <class T>
static mjsElement* GetNext(const mjCBody*    body,
                           const mjsElement* child,
                           bool*             found,
                           bool              recursive) {
  std::vector<T*> list = body->GetList<T>();

  for (unsigned int i = 0; i < list.size(); i++) {
    if (*found) { return list[i]->spec.element; }

    if (list[i]->spec.element == child) { *found = true; }
  }

  if (!recursive) { return nullptr; }

  for (auto& other : body->Bodies()) {
    mjsElement* candidate = GetNext<T>(other, child, found, true);
    if (candidate) { return candidate; }
  }

  return nullptr;
}


// get next body depth first
static mjsElement* GetNextBody(const mjCBody*    body,
                               const mjsElement* child,
                               bool*             found,
                               bool              recursive) {
  for (auto& other : body->Bodies()) {
    if (*found) { return other->spec.element; }

    if (other->spec.element == child) { *found = true; }

    if (!recursive) { continue; }

    mjsElement* candidate = GetNextBody(other, child, found, true);
    if (candidate) { return candidate; }
  }

  return nullptr;
}


// get next child of given type
mjsElement* mjCBody::NextChild(const mjsElement* child, mjtObj type, bool recursive) const {
  if (type == mjOBJ_UNKNOWN) {
    if (!child) {
      throw mjCError(this, "child type must be specified if no child element is given");
    } else {
      type = child->elemtype;
    }
  } else if (child && child->elemtype != type) {
    throw mjCError(this, "child element is not of requested type");
  }

  mjsElement* candidate = nullptr;
  bool        found     = child == nullptr;
  switch (type) {
    case mjOBJ_BODY:
    case mjOBJ_XBODY:
      candidate = GetNextBody(this, child, &found, recursive);
      break;
    case mjOBJ_JOINT:
      candidate = GetNext<mjCJoint>(this, child, &found, recursive);
      break;
    case mjOBJ_GEOM:
      candidate = GetNext<mjCGeom>(this, child, &found, recursive);
      break;
    case mjOBJ_SITE:
      candidate = GetNext<mjCSite>(this, child, &found, recursive);
      break;
    case mjOBJ_CAMERA:
      candidate = GetNext<mjCCamera>(this, child, &found, recursive);
      break;
    case mjOBJ_LIGHT:
      candidate = GetNext<mjCLight>(this, child, &found, recursive);
      break;
    case mjOBJ_FRAME:
      candidate = GetNext<mjCFrame>(this, child, &found, recursive);
      break;
    default:
      throw mjCError(this,
                     "Body.NextChild supports the types: body, frame, geom, "
                     "site, light, camera");
      break;
  }

  return candidate;
}


// compute geom inertial frame: ipos, iquat, mass, inertia
bool mjCBody::InertiaFromGeom(void) {
  int    sz;
  double com[3]  = {0, 0, 0};
  double toti[6] = {0, 0, 0, 0, 0, 0};

  std::vector<mjCGeom*> sel;

  // select geoms based on group, ignore tiny masses
  sel.clear();
  for (int i = 0; i < geoms.size(); i++) {
    if (geoms[i]->group >= compiler->inertiagrouprange[0] &&
        geoms[i]->group <= compiler->inertiagrouprange[1] &&
        geoms[i]->mass_ > mjEPS) {
      sel.push_back(geoms[i]);
    }
  }
  sz = sel.size();

  // single geom: copy
  if (sz == 1) {
    mjuu_copyvec(ipos, sel[0]->pos, 3);
    mjuu_copyvec(iquat, sel[0]->quat, 4);
    mass = sel[0]->mass_;
    mjuu_copyvec(inertia, sel[0]->inertia, 3);
  }

  // multiple geoms
  else if (sz > 1) {
    // compute total mass and center of mass
    mass = 0;
    for (int i = 0; i < sz; i++) {
      mass   += sel[i]->mass_;
      com[0] += sel[i]->mass_ * sel[i]->pos[0];
      com[1] += sel[i]->mass_ * sel[i]->pos[1];
      com[2] += sel[i]->mass_ * sel[i]->pos[2];
    }

    // check for small mass
    if (mass < mjEPS) {
      throw mjCError(this, "body mass is too small, cannot compute center of mass");
    }

    // ipos = geom com
    ipos[0] = com[0] / mass;
    ipos[1] = com[1] / mass;
    ipos[2] = com[2] / mass;

    // add geom inertias
    for (int i = 0; i < sz; i++) {
      double inert0[6], inert1[6];
      double dpos[3] = {sel[i]->pos[0] - ipos[0],
                        sel[i]->pos[1] - ipos[1],
                        sel[i]->pos[2] - ipos[2]};

      mjuu_globalinertia(inert0, sel[i]->inertia, sel[i]->quat);
      mjuu_offcenter(inert1, sel[i]->mass_, dpos);
      for (int j = 0; j < 6; j++) { toti[j] = toti[j] + inert0[j] + inert1[j]; }
    }

    // compute principal axes of inertia
    mjuu_copyvec(fullinertia, toti, 6);
    const char* errq = mjuu_fullInertia(iquat, inertia, fullinertia);
    if (errq && (compiler->boundinertia <= 0 || inertia[2] < -mjEPS * std::max(1.0, inertia[0]))) {
      throw mjCError(this, "error '%s' in alternative for principal axes", errq);
    }
  }
  return sz > 0;
}


// set explicitinertial to true
void mjCBody::MakeInertialExplicit() {
  spec.explicitinertial = true;
}


// make the inertial which compilation calculated for this body part of the spec, in its units
void mjCBody::AdoptInertial() {
  MakeInertialExplicit();
  iframe    = nullptr;  // the inertial frame is in body coordinates
  spec.mass = mass;
  mjuu_copyvec(spec.ipos, ipos_compiled_, 3);
  mjuu_copyvec(spec.iquat, iquat_compiled_, 4);
  mjuu_copyvec(spec.inertia, inertia, 3);
  spec.fullinertia[0] = mjNAN;
  mjs_defaultOrientation(&spec.ialt);
  const mjCGauge g = ContentGauge().Inverse();
  g.Point(spec.ipos);
  g.Inertia(spec.mass, spec.inertia, spec.fullinertia, spec.iquat);
}


// turn a compiled pose of this body into the pose in the spec which compiles to it
void mjCBody::PoseInSpec(double bodypos[3], double bodyquat[4]) const {
  // the alignment moved the frame of the body to its inertial frame
  if (aligned_) { mjuu_frameaccuminv(bodypos, bodyquat, ipos_compiled_, iquat_compiled_); }
  if (frame) { frame->ToLocal(bodypos, bodyquat); }
  Gauge().Inverse().Point(bodypos);
}


// turn a compiled pose of an element of this body into the pose in the spec which compiles to it
void mjCBody::ElementPoseInSpec(const mjCFrame* elementframe,
                                double          elementpos[3],
                                double          elementquat[4]) const {
  // the alignment moved the frames of the body with what they hold, and what is in no frame alone
  if (elementframe) {
    elementframe->ToLocal(elementpos, elementquat);
  } else if (aligned_) {
    mjuu_frameaccumChild(ipos_compiled_, iquat_compiled_, elementpos, elementquat);
  }
  (elementframe ? elementframe->ContentGauge() : ContentGauge()).Inverse().Point(elementpos);
}


// write to the spec the position and the orientation which compile to the compiled ones: the
// alignment with a free joint offsets the position by the orientation, which then gives both
void mjCBody::PoseToSpec(bool position, bool orientation) {
  double specpos[3], specquat[4];
  mjuu_copyvec(specpos, pos, 3);
  mjuu_copyvec(specquat, quat, 4);
  PoseInSpec(specpos, specquat);
  const bool offset = aligned_ && (ipos_compiled_[0] || ipos_compiled_[1] || ipos_compiled_[2]);
  if (position || (orientation && offset)) { mjuu_copyvec(spec.pos, specpos, 3); }
  if (orientation) {
    mjuu_copyvec(spec.quat, specquat, 4);
    mjs_defaultOrientation(&spec.alt);
  }
}


// write to the spec the inertial which compiles to the compiled one
void mjCBody::InertialToSpec(bool massonly) {
  // the inertia which the model has is given to the body from now on
  inertia_inferred_ = false;
  inertia_given_    = true;

  // an inertial which the spec gives stays as it was written if only the mass is new
  if (massonly && mjuu_defined(spec.ipos[0])) {
    spec.mass = mass * (iframe ? iframe->ContentGauge() : ContentGauge()).Inverse().Shape(0, 1);
    return;
  }

  // otherwise all of it is given, as when it is adopted: alignment with a free joint leaves the
  // inertial frame where the spec has it, and without alignment it is the compiled one
  if (!aligned_) {
    mjuu_copyvec(ipos_compiled_, ipos, 3);
    mjuu_copyvec(iquat_compiled_, iquat, 4);
  }
  AdoptInertial();
  explicitinertial = true;
}


// true if compilation infers the inertial of this body from its geoms
bool mjCBody::InfersInertial() const {
  return compiler->inertiafromgeom == mjINERTIAFROMGEOM_TRUE ||
         (compiler->inertiafromgeom == mjINERTIAFROMGEOM_AUTO && !mjuu_defined(spec.ipos[0]));
}


// get the inertial which compilation calculated: center of mass in body coordinates and inertia
// matrix about it
void mjCBody::CompiledInertial(double com[3], double inert[6]) const {
  double local[6] = {inertia[0], inertia[1], inertia[2], 0, 0, 0};
  mjuu_copyvec(com, ipos_compiled_, 3);
  RotateInertia(inert, local, iquat_compiled_);
}


// get the inertial in the spec: center of mass in body coordinates and inertia matrix about it
void mjCBody::SpecInertial(double com[3], double inert[6]) const {
  double orient[4];
  double local[6] = {spec.inertia[0], spec.inertia[1], spec.inertia[2], 0, 0, 0};
  mjuu_copyvec(com, spec.ipos, 3);
  mjuu_copyvec(orient, spec.iquat, 4);
  mjuu_normvec(orient, 4);

  // full or diagonal inertia in the inertial frame, same checks as the compiler
  if (mjuu_defined(spec.fullinertia[0])) {
    if (spec.ialt.type != mjORIENTATION_QUAT) {
      throw mjCError(this, "fullinertia and inertial orientation cannot both be specified");
    }
    if (spec.inertia[0] || spec.inertia[1] || spec.inertia[2]) {
      throw mjCError(this, "fullinertia and diagonal inertia cannot both be specified");
    }
    mjuu_copyvec(local, spec.fullinertia, 6);
  } else {
    const char* err = ResolveOrientation(orient, compiler->degree, compiler->eulerseq, spec.ialt);
    if (err) { throw mjCError(this, "error '%s' in inertia alternative", err); }
  }

  // frame enclosing the inertial element
  if (iframe) { FrameToBody(iframe, com, orient); }

  RotateInertia(inert, local, orient);
}


// true if compilation scales the inertial of this body, by its frame or by that of the inertial
bool mjCBody::ScaledInertial() const {
  return !ContentGauge().IsIdentity() || (iframe && !iframe->ContentGauge().IsIdentity());
}


// merge the inertial of a child body into the inertial in the spec of this body
void mjCBody::MergeInertial(const mjCBody* child) {
  const bool scaled = ScaledInertial() || child->ScaledInertial();

  // pose of the child in this body
  double childpos[3], childquat[4];
  if (scaled) {
    mjuu_copyvec(childpos, child->pos, 3);
    mjuu_copyvec(childquat, child->quat, 4);
  } else {
    mjuu_copyvec(childpos, child->spec.pos, 3);
    mjuu_copyvec(childquat, child->spec.quat, 4);
    mjuu_normvec(childquat, 4);
    const char* err = ResolveOrientation(childquat,
                                         child->compiler->degree,
                                         child->compiler->eulerseq,
                                         child->spec.alt);
    if (err) { throw mjCError(child, "error '%s' in frame alternative", err); }
    FrameToBody(child->frame, childpos, childquat);
  }

  // inertial of this body, if any: inferred from its geoms, or given. Without geoms nothing is
  // inferred, and what an earlier compilation calculated is not read: the caller compiles the
  // tree only when there are geoms to infer from
  double masses[2] = {0, 0};
  double coms[2][3], inerts[2][6];
  mjuu_zerovec(coms[0], 3);
  mjuu_zerovec(inerts[0], 6);
  if (InfersInertial() && !geoms.empty()) {
    masses[0] = mass;
    CompiledInertial(coms[0], inerts[0]);
  } else if (mjuu_defined(spec.ipos[0]) && scaled) {
    masses[0] = mass;
    CompiledInertial(coms[0], inerts[0]);
  } else if (mjuu_defined(spec.ipos[0])) {
    masses[0] = spec.mass;
    SpecInertial(coms[0], inerts[0]);
  }

  // inertial of the child, in the coordinates of this body
  if ((child->InfersInertial() && !child->geoms.empty()) || scaled) {
    masses[1] = child->mass;
    child->CompiledInertial(coms[1], inerts[1]);
  } else {
    masses[1] = child->spec.mass;
    child->SpecInertial(coms[1], inerts[1]);
  }
  mjuu_rotVecQuat(coms[1], coms[1], childquat);
  mjuu_addtovec(coms[1], childpos, 3);
  RotateInertia(inerts[1], inerts[1], childquat);

  // total mass, center of mass and inertia about it
  double totalmass     = masses[0] + masses[1];
  double totalcom[3]   = {0, 0, 0};
  double totalinert[6] = {0, 0, 0, 0, 0, 0};
  for (int j = 0; j < 2; j++) {
    for (int k = 0; k < 3; k++) { totalcom[k] += masses[j] * coms[j][k] / totalmass; }
  }
  for (int j = 0; j < 2; j++) {
    double offcenter[6];
    double dpos[3] = {coms[j][0] - totalcom[0], coms[j][1] - totalcom[1], coms[j][2] - totalcom[2]};
    mjuu_offcenter(offcenter, masses[j], dpos);
    for (int k = 0; k < 6; k++) { totalinert[k] += inerts[j][k] + offcenter[k]; }
  }
  if (scaled) {
    double diag[3] = {0, 0, 0}, unitq[4] = {1, 0, 0, 0};
    ContentGauge().Inverse().Point(totalcom);
    ContentGauge().Inverse().Inertia(totalmass, diag, totalinert, unitq);
  }

  // save as full inertia in body coordinates
  MakeInertialExplicit();
  iframe    = nullptr;
  spec.mass = totalmass;
  mjuu_copyvec(spec.ipos, totalcom, 3);
  mjuu_setvec(spec.iquat, 1, 0, 0, 0);
  mjuu_setvec(spec.inertia, 0, 0, 0);
  mjuu_copyvec(spec.fullinertia, totalinert, 6);
  mjs_defaultOrientation(&spec.ialt);
}


// accumulate compiled inertia of another body into this body
void mjCBody::AccumulateInertia(const mjsBody* other) {
  mjsBody* result = this;  // the private mjsBody

  // body_ipose = body_pose * body_ipose
  double other_ipos[3];
  double other_iquat[4];
  mjuu_copyvec(other_ipos, other->pos, 3);
  mjuu_copyvec(other_iquat, other->quat, 4);
  mjuu_frameaccum(other_ipos, other_iquat, other->ipos, other->iquat);

  // organize data
  double mass[2]       = {result->mass, other->mass};
  double inertia[2][3] = {
      {result->inertia[0], result->inertia[1], result->inertia[2]},
      {other->inertia[0],  other->inertia[1],  other->inertia[2] }
  };
  double ipos[2][3] = {
      {result->ipos[0], result->ipos[1], result->ipos[2]},
      {other_ipos[0],   other_ipos[1],   other_ipos[2]  }
  };
  double iquat[2][4] = {
      {result->iquat[0], result->iquat[1], result->iquat[2], result->iquat[3]},
      {other_iquat[0],   other_iquat[1],   other_iquat[2],   other_iquat[3]  }
  };

  // compute total mass
  result->mass = 0;
  mjuu_setvec(result->ipos, 0, 0, 0);
  for (int j = 0; j < 2; j++) {
    result->mass    += mass[j];
    result->ipos[0] += mass[j] * ipos[j][0];
    result->ipos[1] += mass[j] * ipos[j][1];
    result->ipos[2] += mass[j] * ipos[j][2];
  }

  // small mass: allow for now, check for errors later
  if (result->mass < mjMINVAL) {
    result->mass = 0;
    mjuu_setvec(result->inertia, 0, 0, 0);
    mjuu_setvec(result->ipos, 0, 0, 0);
    mjuu_setvec(result->iquat, 1, 0, 0, 0);
  }

  // proceed with regular computation
  else {
    // locipos = center-of-mass
    result->ipos[0] /= result->mass;
    result->ipos[1] /= result->mass;
    result->ipos[2] /= result->mass;

    // add inertias
    double toti[6] = {0, 0, 0, 0, 0, 0};
    for (int j = 0; j < 2; j++) {
      double inertA[6], inertB[6];
      double dpos[3] = {ipos[j][0] - result->ipos[0],
                        ipos[j][1] - result->ipos[1],
                        ipos[j][2] - result->ipos[2]};

      mjuu_globalinertia(inertA, inertia[j], iquat[j]);
      mjuu_offcenter(inertB, mass[j], dpos);
      for (int k = 0; k < 6; k++) { toti[k] += inertA[k] + inertB[k]; }
    }

    // compute principal axes of inertia
    mjuu_copyvec(result->fullinertia, toti, 6);
    const char* err1 = mjuu_fullInertia(result->iquat, result->inertia, result->fullinertia);
    if (err1) { throw mjCError(nullptr, "error '%s' in fusing static body inertias", err1); }
  }
}


// compute bounding volume hierarchy
void mjCBody::ComputeBVH() {
  // discard the tree of a previous compilation, also when no geoms are left
  tree.AllocateBoundingVolumes(geoms.size());
  if (geoms.empty()) { return; }

  tree.Set(ipos, iquat);
  for (const mjCGeom* geom : geoms) {
    tree.AddBoundingVolume(&geom->id,
                           geom->contype,
                           geom->conaffinity,
                           geom->pos,
                           geom->quat,
                           geom->aabb);
  }
  tree.CreateBVH(model, this);
}


// forget the keyframe values of this body and its subtree which are stored under the given names,
// or all others if keep is true
void mjCBody::ForgetKeyframes(const std::vector<std::string>& names, bool keep) {
  forgetvalues(mpos_, names, keep);
  forgetvalues(mquat_, names, keep);
  for (mjCJoint* joint : joints) { joint->ForgetKeyframes(names, keep); }
  for (mjCBody* body : bodies) { body->ForgetKeyframes(names, keep); }
}


mjtNum* mjCBody::mpos(const std::string& state_name) {
  if (mpos_.find(state_name) == mpos_.end()) { mpos_[state_name] = {mjNAN, 0, 0}; }
  return mpos_.at(state_name).data();
}


mjtNum* mjCBody::mquat(const std::string& state_name) {
  if (mquat_.find(state_name) == mquat_.end()) { mquat_[state_name] = {mjNAN, 0, 0, 0}; }
  return mquat_.at(state_name).data();
}


// scale the compiled copy by the gauge; the inertial takes that of the contents of its own frame
void mjCBody::ApplyGauge() {
  const mjCGauge g  = Gauge();
  const mjCGauge gi = iframe ? iframe->ContentGauge() : ContentGauge();
  if (g.IsIdentity() && gi.IsIdentity()) { return; }
  if (plugin.active) { throw mjCError(this, "a body with a plugin cannot be scaled"); }

  g.Point(pos);
  if (mjuu_defined(ipos[0])) { gi.Point(ipos); }
  if (!gi.uniform && ialt.type != mjORIENTATION_QUAT) {
    const char* err = ResolveOrientation(iquat, compiler->degree, compiler->eulerseq, ialt);
    if (err) { throw mjCError(this, "error '%s' in inertia alternative", err); }
    mjs_defaultOrientation(&ialt);
  }
  gi.Inertia(mass, inertia, fullinertia, iquat);
}


// gauge of what this body contains, in its coordinates
mjCGauge mjCBody::ContentGauge() const {
  mjCGauge g = Gauge();
  if (g.uniform) { return g; }
  double quat[4];
  SpecQuat(quat, spec.quat, spec.alt, compiler);
  return g.Rotated(quat);
}


// compiler
void mjCBody::Compile(void) {
  CopyFromSpec();
  ApplyGauge();

  // compile all frames
  for (int i = 0; i < frames.size(); i++) { frames[i]->Compile(); }

  // resize userdata
  if (userdata_.size() > model->nuser_body) {
    throw mjCError(this, "user has more values than nuser_body in body");
  }
  userdata_.resize(model->nuser_body);

  // normalize user-defined quaternions
  mjuu_normvec(quat, 4);
  mjuu_normvec(iquat, 4);

  // set parentid and weldid of children
  for (int i = 0; i < bodies.size(); i++) {
    bool weld_root    = !bodies[i]->joints.empty() || bodies[i]->spec.mocap;
    bodies[i]->weldid = (weld_root ? bodies[i]->id : weldid);
  }

  // check and process orientation alternatives for body
  if (alt.type != mjORIENTATION_QUAT) {
    const char* err = ResolveOrientation(quat, compiler->degree, compiler->eulerseq, alt);
    if (err) { throw mjCError(this, "error '%s' in frame alternative", err); }
  }

  // check orientation alternatives for inertia
  if (mjuu_defined(fullinertia[0]) && ialt.type != mjORIENTATION_QUAT) {
    throw mjCError(this, "fullinertia and inertial orientation cannot both be specified");
  }
  if (mjuu_defined(fullinertia[0]) && (inertia[0] || inertia[1] || inertia[2])) {
    throw mjCError(this, "fullinertia and diagonal inertia cannot both be specified");
  }

  // process orientation alternatives for inertia
  if (mjuu_defined(fullinertia[0])) {
    // rotate tensor from inertial frame to body frame
    double mat[9], full[9], full_body[9], fullinertia_body[6];
    mjuu_quat2mat(mat, iquat);
    full[0] = this->fullinertia[0];
    full[4] = this->fullinertia[1];
    full[8] = this->fullinertia[2];
    full[1] = full[3] = this->fullinertia[3];
    full[2] = full[6] = this->fullinertia[4];
    full[5] = full[7] = this->fullinertia[5];
    mjuu_mulRMRT(full_body, mat, full);
    fullinertia_body[0] = full_body[0];
    fullinertia_body[1] = full_body[4];
    fullinertia_body[2] = full_body[8];
    fullinertia_body[3] = full_body[1];
    fullinertia_body[4] = full_body[2];
    fullinertia_body[5] = full_body[5];

    // decompose rotated tensor into principal axes
    const char* err = mjuu_fullInertia(iquat, inertia, fullinertia_body);

    // small principal moments: bounded below with diagonal inertia, if boundinertia is set
    if (err && (id == 0 ||
                compiler->boundinertia <= 0 ||
                inertia[2] < -mjEPS * std::max(1.0, inertia[0]))) {
      throw mjCError(this, "error '%s' in fullinertia", err);
    }
  }

  if (ialt.type != mjORIENTATION_QUAT) {
    const char* err = ResolveOrientation(iquat, compiler->degree, compiler->eulerseq, ialt);
    if (err) { throw mjCError(this, "error '%s' in inertia alternative", err); }
  }

  // frame enclosing the inertial element
  if (iframe) { mjuu_frameaccumChild(iframe->pos, iframe->quat, ipos, iquat); }

  // compile all geoms
  for (int i = 0; i < geoms.size(); i++) {
    geoms[i]->inferinertia =
        id > 0 &&
        (!explicitinertial || compiler->inertiafromgeom == mjINERTIAFROMGEOM_TRUE) &&
        geoms[i]->spec.group >= compiler->inertiagrouprange[0] &&
        geoms[i]->spec.group <= compiler->inertiagrouprange[1];
    geoms[i]->Compile();
  }

  // set inertial frame from geoms if necessary; how the body got its inertia is kept, for saving
  // the compiled values also once the spec is edited
  bool infer_inertia =
      id > 0 && (compiler->inertiafromgeom == mjINERTIAFROMGEOM_TRUE ||
                 (!mjuu_defined(ipos[0]) && compiler->inertiafromgeom == mjINERTIAFROMGEOM_AUTO));
  bool from_geom     = infer_inertia && InertiaFromGeom();
  inertia_inferred_  = from_geom || (infer_inertia && !mjuu_defined(ipos[0]));
  inertia_given_     = !inertia_inferred_ && mjuu_defined(ipos[0]);
  inertia_groups_[0] = compiler->inertiagrouprange[0];
  inertia_groups_[1] = compiler->inertiagrouprange[1];

  // ipos undefined: copy body frame into inertial
  if (!mjuu_defined(ipos[0])) {
    mjuu_copyvec(ipos, pos, 3);
    mjuu_copyvec(iquat, quat, 4);
  }

  // check and correct mass and inertia
  const double unadjusted[4] = {mass, inertia[0], inertia[1], inertia[2]};
  if (id > 0) {
    // fix minimum
    mass       = std::max(mass, compiler->boundmass);
    inertia[0] = std::max(inertia[0], compiler->boundinertia);
    inertia[1] = std::max(inertia[1], compiler->boundinertia);
    inertia[2] = std::max(inertia[2], compiler->boundinertia);

    // check for negative values
    if (mass < 0 || inertia[0] < 0 || inertia[1] < 0 || inertia[2] < 0) {
      throw mjCError(this, "mass and inertia cannot be negative");
    }

    // check for non-physical inertia
    if (inertia[0] + inertia[1] < inertia[2] ||
        inertia[0] + inertia[2] < inertia[1] ||
        inertia[1] + inertia[2] < inertia[0]) {
      if (compiler->balanceinertia) {
        inertia[0] = inertia[1] = inertia[2] = (inertia[0] + inertia[1] + inertia[2]) / 3.0;
      } else {
        throw mjCError(this, "inertia must satisfy A + B >= C; use 'balanceinertia' to fix");
      }
    }
  }
  inertia_adjusted_ = mass != unadjusted[0] ||
                      inertia[0] != unadjusted[1] ||
                      inertia[1] != unadjusted[2] ||
                      inertia[2] != unadjusted[3];

  // the inertial as it would be authored: alignment with a free joint changes its frame below, and
  // settotalmass scales its mass and inertia once all bodies are compiled
  mjuu_copyvec(ipos_compiled_, ipos, 3);
  mjuu_copyvec(iquat_compiled_, iquat, 4);
  mass_compiled_ = mass;
  mjuu_copyvec(inertia_compiled_, inertia, 3);

  // frame
  if (frame) { mjuu_frameaccumChild(frame->pos, frame->quat, pos, quat); }

  // check conditions for free-joint alignment
  bool align_free = (joints.size() == 1 &&                  // only one joint AND
                     joints[0]->spec.type == mjJNT_FREE &&  // it is a free joint AND
                     bodies.empty() &&                      // no child bodies AND
                     (joints[0]->spec.align == 1 ||         // either joint.align="true"
                      (joints[0]->spec.align == 2 &&        // or (joint.align="auto"
                       compiler->alignfree)));              //     and compiler->align="true")
  aligned_        = align_free;

  // free-joint alignment, phase 1 (this body + child geoms)
  double ipos_inverse[3], iquat_inverse[4];
  if (align_free) {
    // accumulate iframe transformation to body frame
    mjuu_frameaccum(pos, quat, ipos, iquat);

    // compute inverse iframe transformation
    mjuu_frameinvert(ipos_inverse, iquat_inverse, ipos, iquat);

    // save iframe, set it to null
    mjuu_setvec(ipos, 0, 0, 0);
    mjuu_setvec(iquat, 1, 0, 0, 0);

    // apply inverse iframe transformation to all child geoms
    for (int i = 0; i < geoms.size(); i++) {
      mjuu_frameaccumChild(ipos_inverse, iquat_inverse, geoms[i]->pos, geoms[i]->quat);
    }
  }

  // compute bounding volume hierarchy
  ComputeBVH();

  // compile all joints, count dofs
  dofnum = 0;
  for (int i = 0; i < joints.size(); i++) { dofnum += joints[i]->Compile(); }

  // check for excessive number of dofs
  if (dofnum > 6) { throw mjCError(this, "more than 6 dofs in body '%s'", name.c_str()); }

  // check for rotation dof after ball joint
  bool hasball = false;
  for (int i = 0; i < joints.size(); i++) {
    if ((joints[i]->type == mjJNT_BALL || joints[i]->type == mjJNT_HINGE) && hasball) {
      throw mjCError(this, "ball followed by rotation in body '%s'", name.c_str());
    }
    if (joints[i]->type == mjJNT_BALL) { hasball = true; }
  }

  // make sure mocap body is fixed child of world
  if (mocap && (dofnum || (parent && parent->name != "world"))) {
    throw mjCError(this, "mocap body '%s' is not a fixed child of world", name.c_str());
  }

  // compute body global pose (no joint transformations in qpos0)
  if (id > 0) {
    mjuu_rotVecQuat(xpos0, pos, parent->xquat0);
    mjuu_addtovec(xpos0, parent->xpos0, 3);
    mjuu_mulquat(xquat0, parent->xquat0, quat);
  }

  // compile all sites
  for (int i = 0; i < sites.size(); i++) sites[i]->Compile();

  // compile all cameras
  for (int i = 0; i < cameras.size(); i++) cameras[i]->Compile();

  // compile all lights
  for (int i = 0; i < lights.size(); i++) lights[i]->Compile();

  // plugin
  if (plugin.active) {
    if (plugin_name.empty() && plugin_instance_name.empty()) {
      throw mjCError(this,
                     "neither 'plugin' nor 'instance' is specified for body '%s', (id = %d)",
                     name.c_str(),
                     id);
    }

    mjCPlugin* plugin_instance = static_cast<mjCPlugin*>(plugin.element);
    model->ResolvePlugin(this, plugin_name, plugin_instance_name, &plugin_instance);
    plugin.element           = plugin_instance;
    const mjpPlugin* pplugin = mjp_getPluginAtSlot(plugin_instance->plugin_slot);
    if (!(pplugin->capabilityflags & mjPLUGIN_PASSIVE)) {
      throw mjCError(this, "plugin '%s' does not support passive forces", pplugin->name);
    }
  }

  // free joint alignment, phase 2 (transform sites, cameras and lights)
  if (align_free) {
    // frames have already been compiled and applied to children

    // sites
    for (int i = 0; i < sites.size(); i++) {
      mjuu_frameaccumChild(ipos_inverse, iquat_inverse, sites[i]->pos, sites[i]->quat);
    }

    // cameras
    for (int i = 0; i < cameras.size(); i++) {
      mjuu_frameaccumChild(ipos_inverse, iquat_inverse, cameras[i]->pos, cameras[i]->quat);
    }

    // lights
    for (int i = 0; i < lights.size(); i++) {
      double qunit[4] = {1, 0, 0, 0};
      mjuu_frameaccumChild(ipos_inverse, iquat_inverse, lights[i]->pos, qunit);
      mjuu_rotVecQuat(lights[i]->dir, lights[i]->dir, iquat_inverse);
    }

    // frames: keep them in place relative to their contents, for the writer
    for (int i = 0; i < frames.size(); i++) {
      mjuu_frameaccumChild(ipos_inverse, iquat_inverse, frames[i]->pos, frames[i]->quat);
    }
  }
}


//------------------ class mjCFrame implementation -------------------------------------------------

// initialize frame
mjCFrame::mjCFrame(mjCModel* _model, mjCFrame* _frame) {
  mjs_defaultFrame(&spec);
  elemtype = mjOBJ_FRAME;
  compiled = false;
  model    = _model;
  if (_model) compiler = &_model->spec.compiler;
  body          = NULL;
  frame         = _frame ? _frame : NULL;
  last_attached = nullptr;
  PointToLocal();
  CopyFromSpec();
}


mjCFrame::mjCFrame(const mjCFrame& other) {
  *this = other;
}


mjCFrame& mjCFrame::operator=(const mjCFrame& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCFrame_*>(this) = static_cast<const mjCFrame_&>(other);
    *static_cast<mjsFrame*>(this)  = static_cast<const mjsFrame&>(other);
  }
  PointToLocal();
  return *this;
}


// attach body to frame
mjCFrame& mjCFrame::operator+=(const mjCBody& other) {
  // append a copy of the attached spec
  if (other.model != model && !model->FindSpec(&other.model->spec.compiler)) {
    model->AppendSpec(&other.model->spec, &other.model->spec.compiler);
    static_cast<mjCModel*>(other.model->spec.element)->AddRef();
  }

  // the tree of this model is about to change: store its keyframes, before the other model lays
  // out its own, which changes the addresses of the elements they share
  if (other.model != model) { model->StoreKeyframes(nullptr); }

  // apply namespace and store keyframes in the source model
  other.model->prefix = other.prefix;
  other.model->suffix = other.suffix;
  other.model->StoreKeyframes(model);
  other.model->prefix   = "";
  other.model->suffix   = "";
  mjCModel* other_model = other.model;

  // attach or copy the subtree
  mjCBody* subtree = model->deepcopy_ ? new mjCBody(other, model) : (mjCBody*)&other;

  // the attached body does not take the values of the keyframes which stay in place in the other
  // model; a copy leaves them, and only them, in the original
  subtree->ForgetKeyframes(other_model->inplacekeys_);
  if (model->deepcopy_) {
    ((mjCBody*)&other)->ForgetKeyframes(other_model->inplacekeys_, /*keep=*/true);
  } else {
    subtree->SetModel(model);
    subtree->ResetId();
    subtree->AddRef();
  }
  other_model->prefix = subtree->prefix;
  other_model->suffix = subtree->suffix;
  subtree->SetParent(body);
  subtree->SetFrame(this);
  subtree->NameSpace(other_model);

  // attach defaults
  if (other_model != model) {
    mjCDef* subdef = new mjCDef(*other_model->Default());
    subdef->NameSpace(other_model);
    *model += *subdef;
  }

  // add to body children
  body->bodies.push_back(subtree);
  last_attached = &body->bodies.back()->spec;

  // attach referencing elements
  other_model->SetAttached(model->deepcopy_);
  *model += *other_model;

  // clear suffixes and return
  other_model->suffix.clear();
  other_model->prefix.clear();
  return *this;
}


// return true if child is descendent of this frame
bool mjCFrame::IsAncestor(const mjCFrame* child) const {
  if (!child) { return false; }

  if (child == this) { return true; }

  return IsAncestor(child->frame);
}


void mjCFrame::PointToLocal() {
  spec.element    = static_cast<mjsElement*>(this);
  spec.childclass = &classname;
  spec.info       = &info;
}


// express in this frame, as compiled, a pose which is given in its body
void mjCFrame::ToLocal(double childpos[3], double childquat[4]) const {
  double invpos[3], invquat[4];
  mjuu_frameinvert(invpos, invquat, pos, quat);
  mjuu_frameaccumChild(invpos, invquat, childpos, childquat);
}


void mjCFrame::CopyFromSpec() {
  *static_cast<mjsFrame*>(this) = spec;
  mjuu_copyvec(pos, spec.pos, 3);
  mjuu_copyvec(quat, spec.quat, 4);
}


void mjCFrame::Compile() {
  if (compiled) { return; }

  CopyFromSpec();
  const char* err = ResolveOrientation(quat, compiler->degree, compiler->eulerseq, alt);
  if (err) { throw mjCError(this, "orientation specification error '%s' in site %d", err, id); }

  // scale must be finite and positive, and uniform
  for (int i = 0; i < 3; i++) {
    if (!std::isfinite(scale[i]) || scale[i] <= 0) {
      throw mjCError(this, "frame scale must be finite and positive");
    }
  }
  if ((scale[0] != scale[1] || scale[1] != scale[2]) && scaling != mjSCALING_GEOMETRY) {
    throw mjCError(this, "an anisotropic frame scale requires scaling 'geometry'");
  }
  if (!std::isfinite(massscale) || massscale < 0) {
    throw mjCError(this, "frame massscale must be finite and non-negative");
  }
  ApplyGauge();

  // compile parents and accumulate result
  if (frame) {
    frame->Compile();
    mjuu_frameaccumChild(frame->pos, frame->quat, pos, quat);
  }

  mjuu_normvec(quat, 4);
  compiled = true;
}


// scale the compiled copy by the gauge around this frame; its own scale applies to its contents
void mjCFrame::ApplyGauge() {
  const mjCGauge g = Gauge();
  if (g.IsIdentity()) { return; }
  g.Point(pos);
}


// gauge of what this frame contains, in its coordinates: the one around it composed with its scale
mjCGauge mjCFrame::ContentGauge() const {
  mjCGauge g = Gauge();
  if (!g.uniform) {
    double quat[4];
    SpecQuat(quat, spec.quat, spec.alt, compiler);
    g = g.Rotated(quat);
  }
  return g * mjCGauge::Of(spec.scale, spec.scaling, spec.massscale);
}


// map a point or a direction given in the body of this frame from the coordinates of the spec to
// the compiled ones, which scale the contents of the frame, or back
void mjCFrame::ScaleInBody(double x[3], bool point, bool inverse) const {
  const mjCGauge g = ContentGauge();
  if (g.IsIdentity()) { return; }

  // pose of this frame in its body, in the spec and as compiled
  double specpos[3] = {0, 0, 0}, specquat[4] = {1, 0, 0, 0};
  FrameToBody(this, specpos, specquat);
  const double* frompos  = inverse ? pos : specpos;
  const double* fromquat = inverse ? quat : specquat;
  const double* topos    = inverse ? specpos : pos;
  const double* toquat   = inverse ? specquat : quat;

  // express in the frame, scale, and express in the body again
  double local[3], res[3], neg[4] = {fromquat[0], -fromquat[1], -fromquat[2], -fromquat[3]};
  for (int i = 0; i < 3; i++) { local[i] = x[i] - (point ? frompos[i] : 0); }
  mjuu_rotVecQuat(res, local, neg);
  (inverse ? g.Inverse() : g).Point(res);
  mjuu_rotVecQuat(local, res, toquat);
  for (int i = 0; i < 3; i++) { x[i] = local[i] + (point ? topos[i] : 0); }
}


//------------------ class mjCJoint implementation -------------------------------------------------

// initialize default joint
mjCJoint::mjCJoint(mjCModel* _model, mjCDef* _def) {
  mjs_defaultJoint(&spec);
  elemtype = mjOBJ_JOINT;

  // clear internal variables
  spec_userdata_.clear();
  body = 0;

  // reset to default if given
  if (_def) { *this = _def->Joint(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // point to local
  PointToLocal();

  // in case this joint is not compiled
  CopyFromSpec();

  // no previous state when a joint is created
  qposadr_    = -1;
  dofadr_     = -1;
  plasticadr_ = -1;
}


mjCJoint::mjCJoint(const mjCJoint& other) {
  *this = other;
}


mjCJoint& mjCJoint::operator=(const mjCJoint& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCJoint_*>(this) = static_cast<const mjCJoint_&>(other);
    *static_cast<mjsJoint*>(this)  = static_cast<const mjsJoint&>(other);

    qposadr_    = -1;
    dofadr_     = -1;
    plasticadr_ = -1;
  }
  PointToLocal();
  return *this;
}


void mjCJoint::ResetId() {
  id          = -1;
  qposadr_    = -1;
  dofadr_     = -1;
  plasticadr_ = -1;
}


bool mjCJoint::is_limited() const {
  return islimited(limited, range);
}
bool mjCJoint::is_actfrclimited() const {
  return islimited(actfrclimited, actfrcrange);
}


int mjCJoint::nq(mjtJoint joint_type) {
  switch (joint_type) {
    case mjJNT_FREE:
      return 7;
    case mjJNT_BALL:
      return 4;
    case mjJNT_SLIDE:
    case mjJNT_HINGE:
      return 1;
  }
  return 1;
}


int mjCJoint::nv(mjtJoint joint_type) {
  switch (joint_type) {
    case mjJNT_FREE:
      return 6;
    case mjJNT_BALL:
      return 3;
    case mjJNT_SLIDE:
    case mjJNT_HINGE:
      return 1;
  }
  return 1;
}


mjtNum* mjCJoint::qpos(const std::string& state_name) {
  if (qpos_.find(state_name) == qpos_.end()) { qpos_[state_name] = {mjNAN, 0, 0, 0, 0, 0, 0}; }
  return qpos_.at(state_name).data();
}


mjtNum* mjCJoint::qvel(const std::string& state_name) {
  if (qvel_.find(state_name) == qvel_.end()) { qvel_[state_name] = {mjNAN, 0, 0, 0, 0, 0}; }
  return qvel_.at(state_name).data();
}


mjtNum* mjCJoint::plastic(const std::string& state_name) {
  if (plastic_.find(state_name) == plastic_.end()) {
    plastic_[state_name] = {mjNAN, 0, 0, 0, 0, 0, 0};
  }
  return plastic_.at(state_name).data();
}


void mjCJoint::ForgetKeyframes(const std::vector<std::string>& names, bool keep) {
  forgetvalues(qpos_, names, keep);
  forgetvalues(qvel_, names, keep);
  forgetvalues(plastic_, names, keep);
}


void mjCJoint::PointToLocal() {
  spec.element  = static_cast<mjsElement*>(this);
  spec.userdata = &spec_userdata_;
  spec.info     = &info;
  userdata      = nullptr;
}


void mjCJoint::CopyFromSpec() {
  *static_cast<mjsJoint*>(this) = spec;

  userdata_ = spec_userdata_;
}


// scale the compiled copy by the gauge, or back; Q is L^q with q = 1 for a slide, F is M L^(2-q)
// T^-2, and the shared parameters of a free joint, which mix both, are left as they are
void mjCJoint::ApplyGauge(bool inverse) {
  const mjCGauge g = inverse ? Gauge().Inverse() : Gauge();
  if (g.IsIdentity()) { return; }
  g.Point(pos);
  if (type == mjJNT_FREE) { return; }
  if (type != mjJNT_BALL && !g.Aligned(axis)) {
    throw mjCError(this, "the axis of this joint is not aligned with the axes of the scale");
  }

  // coordinate: shape class, the stretch along the axis of a slide
  const int    q      = type == mjJNT_SLIDE ? 1 : 0;
  const double fq     = q ? g.Stretch(axis) : 1;
  range[0]           *= fq;
  range[1]           *= fq;
  ref                *= fq;
  springref          *= fq;
  springyield        *= fq;
  margin             *= fq;
  solimp_limit[2]    *= fq;
  solimp_friction[2] *= fq;

  // springs, dampers, armature and forces: force class
  const int a = 2 - q;
  for (int n = 0; n <= mjNPOLY; n++) {
    stiffness[n] *= g.Force(a - (n + 1) * q, 1);
    damping[n]   *= g.Force(a - (n + 1) * q, 1);
  }
  armature       *= g.Force(a - q, 1);
  frictionloss   *= g.Force(a, 1);
  actfrcrange[0] *= g.Force(a, 1);
  actfrcrange[1] *= g.Force(a, 1);
}


// compiler
int mjCJoint::Compile(void) {
  CopyFromSpec();
  ApplyGauge();

  // resize userdata
  if (userdata_.size() > model->nuser_jnt) {
    throw mjCError(this, "user has more values than nuser_jnt in joint");
  }
  userdata_.resize(model->nuser_jnt);

  // check springdamper and springyield
  if (springdamper[0] || springdamper[1]) {
    if (springdamper[0] <= 0 || springdamper[1] <= 0) {
      throw mjCError(this, "when defined, springdamper values must be positive in joint");
    }
  }
  if (springyield < 0) { throw mjCError(this, "springyield cannot be negative in joint"); }

  // free joints cannot be limited
  if (type == mjJNT_FREE) {
    limited = mjLIMITED_FALSE;
  }

  // otherwise if limited is auto, check consistency wrt auto-limits
  else if (limited == mjLIMITED_AUTO) {
    bool hasrange = !(range[0] == 0 && range[1] == 0);
    checklimited(this, compiler->autolimits, "joint", "", limited, hasrange);
  }

  // resolve limits
  if (is_limited()) {
    // check data
    if (range[0] >= range[1] && type != mjJNT_BALL) {
      throw mjCError(this, "range[0] should be smaller than range[1] in joint");
    }
    if (range[0] && type == mjJNT_BALL) {
      throw mjCError(this, "range[0] should be 0 in ball joint");
    }

    // convert limits to radians
    if (compiler->degree && (type == mjJNT_HINGE || type == mjJNT_BALL)) {
      if (range[0]) { range[0] *= mjPI / 180.0; }
      if (range[1]) { range[1] *= mjPI / 180.0; }
    }
  }

  // actuator force range: none for free or ball joints
  if (type == mjJNT_FREE || type == mjJNT_BALL) {
    actfrclimited = mjLIMITED_FALSE;
  }

  // otherwise if actfrclimited is auto, check consistency wrt auto-limits
  else if (actfrclimited == mjLIMITED_AUTO) {
    bool hasrange = !(actfrcrange[0] == 0 && actfrcrange[1] == 0);
    checklimited(this, compiler->autolimits, "joint", "actuatorfrc", actfrclimited, hasrange);
  }

  // resolve actuator force range limits
  if (is_actfrclimited()) {
    // check data
    if (actfrcrange[0] >= actfrcrange[1]) {
      throw mjCError(this, "actfrcrange[0] should be smaller than actfrcrange[1] in joint");
    }
  }

  // axis: FREE or BALL are fixed to (0,0,1)
  if (type == mjJNT_FREE || type == mjJNT_BALL) {
    axis[0] = axis[1] = 0;
    axis[2]           = 1;
  }

  // otherwise accumulate frame rotation
  else if (frame) {
    mjuu_rotVecQuat(axis, axis, frame->quat);
  }

  // normalize axis, check norm
  if (mjuu_normvec(axis, 3) < mjEPS) { throw mjCError(this, "axis too small in joint"); }

  // check data
  if (type == mjJNT_FREE && limited == mjLIMITED_TRUE) {
    throw mjCError(this, "limits should not be defined in free joint");
  }
  if (type == mjJNT_FREE && springyield > 0) {
    throw mjCError(this, "springyield should not be defined in free joint");
  }

  // pos: FREE is fixed to (0,0,0)
  if (type == mjJNT_FREE) {
    mjuu_zerovec(pos, 3);
  }

  // otherwise accumulate frame translation
  else if (frame) {
    double qunit[4] = {1, 0, 0, 0};
    mjuu_frameaccumChild(frame->pos, frame->quat, pos, qunit);
  }

  // convert reference angles to radians for hinge joints, and springyield for hinge and ball joints
  if (compiler->degree) {
    if (type == mjJNT_HINGE) {
      ref       *= mjPI / 180.0;
      springref *= mjPI / 180.0;
    }
    if (type == mjJNT_HINGE || type == mjJNT_BALL) { springyield *= mjPI / 180.0; }
  }

  // return dofnum
  if (type == mjJNT_FREE) {
    return 6;
  } else if (type == mjJNT_BALL) {
    return 3;
  } else {
    return 1;
  }
}


// write to the spec the anchor and the axis which compile to the compiled ones
void mjCJoint::AnchorToSpec(bool anchor, bool direction) {
  double specpos[3], rotation[4] = {1, 0, 0, 0};
  mjuu_copyvec(specpos, pos, 3);
  if (frame) { frame->ToLocal(specpos, rotation); }
  Gauge().Inverse().Point(specpos);
  if (anchor) { mjuu_copyvec(spec.pos, specpos, 3); }
  if (direction) { mjuu_rotVecQuat(spec.axis, axis, rotation); }
}


//------------------ class mjCGeom implementation --------------------------------------------------

// initialize default geom
mjCGeom::mjCGeom(mjCModel* _model, mjCDef* _def) {
  mjs_defaultGeom(&spec);
  elemtype = mjOBJ_GEOM;

  mass_  = 0;
  body   = 0;
  matid  = -1;
  mesh   = nullptr;
  hfield = nullptr;
  mjuu_setvec(inertia, 0, 0, 0);
  inferinertia = true;
  spec_material_.clear();
  spec_userdata_.clear();
  spec_meshname_.clear();
  spec_hfieldname_.clear();
  spec_userdata_.clear();

  for (int i = 0; i < mjNFLUID; i++) { fluid[i] = 0; }

  // reset to default if given
  if (_def) { *this = _def->Geom(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // point to local
  PointToLocal();

  // in case this geom is not compiled
  CopyFromSpec();
}


mjCGeom::mjCGeom(const mjCGeom& other) {
  *this = other;
}


mjCGeom& mjCGeom::operator=(const mjCGeom& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCGeom_*>(this) = static_cast<const mjCGeom_&>(other);
    *static_cast<mjsGeom*>(this)  = static_cast<const mjsGeom&>(other);
  }
  PointToLocal();
  return *this;
}


// to be called after any default copy constructor
void mjCGeom::PointToLocal(void) {
  spec.element            = static_cast<mjsElement*>(this);
  spec.info               = &info;
  spec.userdata           = &spec_userdata_;
  spec.material           = &spec_material_;
  spec.meshname           = &spec_meshname_;
  spec.hfieldname         = &spec_hfieldname_;
  spec.plugin.plugin_name = &plugin_name;
  spec.plugin.name        = &plugin_instance_name;
  userdata                = nullptr;
  hfieldname              = nullptr;
  meshname                = nullptr;
  material                = nullptr;
}


void mjCGeom::CopyFromSpec() {
  *static_cast<mjsGeom*>(this) = spec;

  userdata_          = spec_userdata_;
  hfieldname_        = spec_hfieldname_;
  meshname_          = spec_meshname_;
  material_          = spec_material_;
  plugin.active      = spec.plugin.active;
  plugin.element     = spec.plugin.element;
  plugin.plugin_name = spec.plugin.plugin_name;
  plugin.name        = spec.plugin.name;
}


void mjCGeom::CopyPlugin() {
  model->CopyExplicitPlugin(this);
}


void mjCGeom::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!spec_material_.empty() && model != m) {
    spec_material_ = m->prefix + spec_material_ + m->suffix;
  }
  if (!spec_hfieldname_.empty() && model != m) {
    spec_hfieldname_ = m->prefix + spec_hfieldname_ + m->suffix;
  }
  if (!spec_meshname_.empty() && model != m) {
    spec_meshname_ = m->prefix + spec_meshname_ + m->suffix;
  }
  if (!plugin_instance_name.empty()) {
    plugin_instance_name = m->prefix + plugin_instance_name + m->suffix;
  }
}


// compute geom volume / surface area
double mjCGeom::GetVolume() const {
  // get from mesh
  if (type == mjGEOM_MESH || type == mjGEOM_SDF) {
    if (mesh->id < 0 || mesh->id > model->NumCompiledMeshes()) {
      throw mjCError(this, "invalid mesh id in mesh geom");
    }

    return mesh->GetVolumeRef();
  }

  // compute from geom shape (type) and inertia type (typeinertia)
  switch (type) {
    case mjGEOM_SPHERE: {
      double radius = size[0];
      switch (typeinertia) {
        case mjINERTIA_VOLUME:
          return 4 * mjPI * radius * radius * radius / 3;
        case mjINERTIA_SHELL:
          return 4 * mjPI * radius * radius;
      }
      break;
    }
    case mjGEOM_CAPSULE: {
      double height = 2 * size[1];
      double radius = size[0];
      switch (typeinertia) {
        case mjINERTIA_VOLUME:
          return mjPI * (radius * radius * height + 4 * radius * radius * radius / 3);
        case mjINERTIA_SHELL:
          return 4 * mjPI * radius * radius + 2 * mjPI * radius * height;
      }
      break;
    }
    case mjGEOM_CYLINDER: {
      double height = 2 * size[1];
      double radius = size[0];
      switch (typeinertia) {
        case mjINERTIA_VOLUME:
          return mjPI * radius * radius * height;
        case mjINERTIA_SHELL:
          return 2 * mjPI * radius * radius + 2 * mjPI * radius * height;
      }
      break;
    }
    case mjGEOM_ELLIPSOID: {
      switch (typeinertia) {
        case mjINERTIA_VOLUME:
          return 4 * mjPI * size[0] * size[1] * size[2] / 3;
        case mjINERTIA_SHELL: {
          // Thomsen approximation
          // https://www.numericana.com/answer/ellipsoid.htm#thomsen
          double p   = 1.6075;
          double tmp = std::pow(size[0] * size[1], p) +
                       std::pow(size[1] * size[2], p) +
                       std::pow(size[2] * size[0], p);
          return 4 * mjPI * std::pow(tmp / 3, 1 / p);
        }
      }
      break;
    }
    case mjGEOM_HFIELD:
    case mjGEOM_BOX: {
      switch (typeinertia) {
        case mjINERTIA_VOLUME:
          return size[0] * size[1] * size[2] * 8;
        case mjINERTIA_SHELL:
          return 8 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
      }
      break;
    }
    default:
      break;
  }
  return 0;
}


// set geom diagonal inertia given density
void mjCGeom::SetInertia(void) {
  // get from mesh
  if (type == mjGEOM_MESH || type == mjGEOM_SDF) {
    if (mesh->id < 0 || mesh->id > model->NumCompiledMeshes()) {
      throw mjCError(this, "invalid mesh id in mesh geom");
    }

    double* boxsz = mesh->GetInertiaBoxPtr();
    inertia[0]    = mass_ * (boxsz[1] * boxsz[1] + boxsz[2] * boxsz[2]) / 3;
    inertia[1]    = mass_ * (boxsz[0] * boxsz[0] + boxsz[2] * boxsz[2]) / 3;
    inertia[2]    = mass_ * (boxsz[0] * boxsz[0] + boxsz[1] * boxsz[1]) / 3;

    return;
  }

  // compute from geom shape (type) and inertia type (typeinertia)
  switch (type) {
    case mjGEOM_SPHERE: {
      switch (typeinertia) {
        case mjINERTIA_VOLUME:
          inertia[0] = inertia[1] = inertia[2] = 2 * mass_ * size[0] * size[0] / 5;
          return;
        case mjINERTIA_SHELL:
          inertia[0] = inertia[1] = inertia[2] = 2 * mass_ * size[0] * size[0] / 3;
          return;
      }
      break;
    }
    case mjGEOM_CAPSULE: {
      double halfheight = size[1];
      double height     = 2 * size[1];
      double radius     = size[0];
      switch (typeinertia) {
        case mjINERTIA_VOLUME: {
          double sphere_mass =
              mass_ * 4 * radius / (4 * radius + 3 * height);  // mass*(sphere_vol/total_vol)
          double cylinder_mass = mass_ - sphere_mass;

          // cylinder part
          inertia[0] = inertia[1] = cylinder_mass * (3 * radius * radius + height * height) / 12;
          inertia[2]              = cylinder_mass * radius * radius / 2;

          // add two hemispheres, displace along third axis
          double sphere_inertia = 2 * sphere_mass * radius * radius / 5;
          inertia[0] += sphere_inertia + sphere_mass * height * (3 * radius + 2 * height) / 8;
          inertia[1] += sphere_inertia + sphere_mass * height * (3 * radius + 2 * height) / 8;
          inertia[2] += sphere_inertia;
          return;
        }
        case mjINERTIA_SHELL: {
          // surface area
          double Asphere   = 4 * mjPI * radius * radius;
          double Acylinder = 2 * mjPI * radius * height;
          double Atotal    = Asphere + Acylinder;

          // mass
          double sphere_mass   = mass_ * Asphere / Atotal;  // mass*(sphere_area/total_area)
          double cylinder_mass = mass_ - sphere_mass;

          // cylinder part
          inertia[0] = inertia[1] = cylinder_mass * (6 * radius * radius + height * height) / 12;
          inertia[2]              = cylinder_mass * radius * radius;

          // add two hemispheres, displace along third axis
          double sphere_inertia = 2 * sphere_mass * radius * radius / 3;
          double hs_com         = radius / 2;           // hemisphere center of mass
          double hs_pos         = halfheight + hs_com;  // hemisphere position
          inertia[0] += sphere_inertia + sphere_mass * (hs_pos * hs_pos - hs_com * hs_com);
          inertia[1] += sphere_inertia + sphere_mass * (hs_pos * hs_pos - hs_com * hs_com);
          inertia[2] += sphere_inertia;
          return;
        } break;
      }
      break;
    }
    case mjGEOM_CYLINDER: {
      double halfheight = size[1];
      double height     = 2 * halfheight;
      double radius     = size[0];
      switch (typeinertia) {
        case mjINERTIA_VOLUME:

          inertia[0] = inertia[1] = mass_ * (3 * radius * radius + height * height) / 12;
          inertia[2]              = mass_ * radius * radius / 2;
          return;
        case mjINERTIA_SHELL: {
          // surface area
          double Adisk     = mjPI * radius * radius;
          double Acylinder = 2 * mjPI * radius * height;
          double Atotal    = 2 * Adisk + Acylinder;

          // mass
          double mass_disk     = mass_ * Adisk / Atotal;
          double mass_cylinder = mass_ - 2 * mass_disk;

          // cylinder contribution
          inertia[0] = inertia[1] = mass_cylinder * (6 * radius * radius + height * height) / 12;
          inertia[2]              = mass_cylinder * radius * radius;

          // disk inertia
          double inertia_disk_x =
              mass_disk * radius * radius / 4 + mass_disk * halfheight * halfheight;
          double inertia_disk_z = mass_disk * radius * radius / 2;

          // top and bottom disk contributions
          inertia[0] += 2 * inertia_disk_x;
          inertia[1] += 2 * inertia_disk_x;
          inertia[2] += 2 * inertia_disk_z;
          return;
        }
      }
      break;
    }
    case mjGEOM_ELLIPSOID: {
      double s00 = size[0] * size[0];
      double s11 = size[1] * size[1];
      double s22 = size[2] * size[2];
      switch (typeinertia) {
        case mjINERTIA_VOLUME: {
          inertia[0] = mass_ * (s11 + s22) / 5;
          inertia[1] = mass_ * (s00 + s22) / 5;
          inertia[2] = mass_ * (s00 + s11) / 5;
          return;
        }
        case mjINERTIA_SHELL: {
          // approximate shell inertia by subtracting ellipsoid from expanded ellipsoid
          double eps = 1e-6;

          // solid volume (a)
          double Va = 4 * mjPI * size[0] * size[1] * size[2] / 3;

          // expanded volume (b)
          double ae = size[0] + eps;
          double be = size[1] + eps;
          double ce = size[2] + eps;
          double Vb = 4 * mjPI * ae * be * ce / 3;

          // density
          double density = mass_ / (Vb - Va);

          // inertia
          double mass_a = Va * density;
          double inertia_a[3];
          inertia_a[0] = mass_a * (s11 + s22) / 5;
          inertia_a[1] = mass_a * (s00 + s22) / 5;
          inertia_a[2] = mass_a * (s00 + s11) / 5;

          double mass_b = Vb * density;
          double inertia_b[3];
          inertia_b[0] = mass_b * (be * be + ce * ce) / 5;
          inertia_b[1] = mass_b * (ae * ae + ce * ce) / 5;
          inertia_b[2] = mass_b * (ae * ae + be * be) / 5;

          // shell inertia
          inertia[0] = inertia_b[0] - inertia_a[0];
          inertia[1] = inertia_b[1] - inertia_a[1];
          inertia[2] = inertia_b[2] - inertia_a[2];
          return;
        }
      }
      break;
    }
    case mjGEOM_HFIELD:
    case mjGEOM_BOX: {
      double s00 = size[0] * size[0];
      double s11 = size[1] * size[1];
      double s22 = size[2] * size[2];
      switch (typeinertia) {
        case mjINERTIA_VOLUME: {
          inertia[0] = mass_ * (s11 + s22) / 3;
          inertia[1] = mass_ * (s00 + s22) / 3;
          inertia[2] = mass_ * (s00 + s11) / 3;
          return;
        }
        case mjINERTIA_SHELL: {
          // length
          double lx = 2 * size[0];  // side 0
          double ly = 2 * size[1];  // side 1
          double lz = 2 * size[2];  // side 2

          // surface area
          double A0     = lx * ly;  // side 0
          double A1     = ly * lz;  // side 1
          double A2     = lz * lx;  // side 2
          double Atotal = 2 * (A0 + A1 + A2);

          // side 0
          double mass0 = mass_ * A0 / Atotal;
          double Ix0   = mass0 * ly * ly / 12;
          double Iy0   = mass0 * lx * lx / 12;
          double Iz0   = mass0 * (lx * lx + ly * ly) / 12;

          // side 1
          double mass1 = mass_ * A1 / Atotal;
          double Ix1   = mass1 * (ly * ly + lz * lz) / 12;
          double Iy1   = mass1 * lz * lz / 12;
          double Iz1   = mass1 * ly * ly / 12;

          // side 3
          double mass2 = mass_ * A2 / Atotal;
          double Ix2   = mass2 * lz * lz / 12;
          double Iy2   = mass2 * (lx * lx + lz * lz) / 12;
          double Iz2   = mass2 * lx * lx / 12;

          // total inertia
          inertia[0] = 2 * (mass0 * s22 + mass2 * s11 + Ix0 + Ix1 + Ix2);
          inertia[1] = 2 * (mass0 * s22 + mass1 * s00 + Iy0 + Iy1 + Iy2);
          inertia[2] = 2 * (mass1 * s00 + mass2 * s11 + Iz0 + Iz1 + Iz2);
          return;
        } break;
      }
      break;
    }
    default:
      inertia[0] = inertia[1] = inertia[2] = 0;
      return;
  }
}


// radius of the bounding sphere of a mesh or SDF geom, zero for other geoms, whose radius
// mj_setConst computes from their size
double mjCGeom::GetRBound(void) {
  if (type != mjGEOM_MESH && type != mjGEOM_SDF) { return 0; }
  const double* aamm     = mesh->aamm();
  double        haabb[3] = {std::max(std::abs(aamm[0]), std::abs(aamm[3])),
                            std::max(std::abs(aamm[1]), std::abs(aamm[4])),
                            std::max(std::abs(aamm[2]), std::abs(aamm[5]))};
  return sqrt(haabb[0] * haabb[0] + haabb[1] * haabb[1] + haabb[2] * haabb[2]);
}


// compute bounding box
void mjCGeom::ComputeAABB(void) {
  double aamm[6];  // axis-aligned bounding box in (min, max) format
  switch (type) {
    case mjGEOM_HFIELD:
      aamm[0] = -hfield->size[0];
      aamm[1] = -hfield->size[1];
      aamm[2] = -hfield->size[3];
      aamm[3] = hfield->size[0];
      aamm[4] = hfield->size[1];
      aamm[5] = hfield->size[2];
      break;

    case mjGEOM_SPHERE:
      aamm[3] = aamm[4] = aamm[5] = size[0];
      mjuu_setvec(aamm, -aamm[3], -aamm[4], -aamm[5]);
      break;

    case mjGEOM_CAPSULE:
      aamm[3] = aamm[4] = size[0];
      aamm[5]           = size[0] + size[1];
      mjuu_setvec(aamm, -aamm[3], -aamm[4], -aamm[5]);
      break;

    case mjGEOM_CYLINDER:
      aamm[3] = aamm[4] = size[0];
      aamm[5]           = size[1];
      mjuu_setvec(aamm, -aamm[3], -aamm[4], -aamm[5]);
      break;

    case mjGEOM_MESH:
    case mjGEOM_SDF:
      mjuu_copyvec(aamm, mesh->aamm(), 6);
      break;

    case mjGEOM_PLANE:
      aamm[0] = aamm[1] = aamm[2] = -mjMAXVAL;
      aamm[3] = aamm[4] = mjMAXVAL;
      aamm[5]           = 0;
      break;

    default:
      mjuu_copyvec(aamm + 3, size, 3);
      mjuu_setvec(aamm, -size[0], -size[1], -size[2]);
      break;
  }

  // convert aamm to aabb (center, size) format
  double pos[]  = {(aamm[3] + aamm[0]) / 2, (aamm[4] + aamm[1]) / 2, (aamm[5] + aamm[2]) / 2};
  double size[] = {(aamm[3] - aamm[0]) / 2, (aamm[4] - aamm[1]) / 2, (aamm[5] - aamm[2]) / 2};
  mjuu_copyvec(aabb, pos, 3);
  mjuu_copyvec(aabb + 3, size, 3);
}

const std::string& mjCGeom::get_material() const {
  if (mesh && spec_material_.empty()) { return mesh->Material(); }
  return spec_material_;
}


// map of the gauge in the frame of this geom, which its mesh or height field instance takes
std::array<double, 9> mjCGeom::ShapeMap() const {
  mjCGauge g = Gauge();
  if (!g.uniform) {
    double quat_[4];
    SpecShape(quat_, spec.type, spec.fromto, spec.quat, spec.alt, compiler);
    g = g.Rotated(quat_);
  }
  std::array<double, 9> map;
  std::copy(g.map, g.map + 9, map.begin());
  return map;
}


// scale the compiled copy by the gauge, or back; density is M L^-3, or M L^-2 for a shell
void mjCGeom::ApplyGauge(bool inverse) {
  const mjCGauge g = inverse ? Gauge().Inverse() : Gauge();
  if (g.IsIdentity()) { return; }
  if (plugin.active) { throw mjCError(this, "a geom with a plugin cannot be scaled"); }

  // shape class; a primitive fitted to a mesh takes the shape of the mesh instance
  const double fl = g.Shape(1);
  double       quat_[4];
  const int    shape  = SpecShape(quat_, type, spec.fromto, spec.quat, spec.alt, compiler);
  const bool   fitted = mesh && type != mjGEOM_MESH && type != mjGEOM_SDF;
  g.Point(pos);
  g.Rotated(quat_).Point(surfacevel);
  if (!fitted || g.uniform) {
    const char* err = g.Size(size, shape, quat_);
    if (err) { throw mjCError(this, "%s", err); }
  }
  if (mjuu_defined(fromto[0])) {
    g.Point(fromto);
    g.Point(fromto + 3);
  }
  friction[1] *= fl;
  friction[2] *= fl;
  margin      *= fl;
  gap         *= fl;
  solimp[2]   *= fl;
  if (mjuu_defined(mass)) { mass *= g.Shape(0, 1); }
  bool shell  = typeinertia == mjINERTIA_SHELL ||
                (type == mjGEOM_MESH && mesh && mesh->spec.inertia == mjMESH_INERTIA_SHELL);
  density    *= shell ? g.Shape(-2, 1) : g.Shape(-3, 1);

  // force class
  adhesion *= g.Force(1, 1);
}


// compiler
void mjCGeom::Compile(void) {
  CopyFromSpec();
  ApplyGauge();

  // resize userdata
  if (userdata_.size() > model->nuser_geom) {
    throw mjCError(this,
                   "user has more values than nuser_geom in geom '%s' (id = %d)",
                   name.c_str(),
                   id);
  }
  userdata_.resize(model->nuser_geom);

  // check type
  if (type < 0 || type >= mjNGEOMTYPES) { throw mjCError(this, "invalid type in geom"); }

  // check condim
  if (condim != 1 && condim != 3 && condim != 4 && condim != 6) {
    throw mjCError(this, "invalid condim in geom");
  }

  // check mesh
  if ((type == mjGEOM_MESH || type == mjGEOM_SDF) && !mesh) {
    throw mjCError(this, "mesh geom '%s' (id = %d) must have valid meshid", name.c_str(), id);
  }

  // check hfield
  if ((type == mjGEOM_HFIELD && !hfield) || (type != mjGEOM_HFIELD && hfield)) {
    throw mjCError(this, "hfield geom '%s' (id = %d) must have valid hfieldid", name.c_str(), id);
  }

  // plane only allowed in bodies with no dofs (static, including mocap)
  if (type == mjGEOM_PLANE && body->weldid != 0) {
    const mjCBody* weld = body;
    while (weld->id != weld->weldid) { weld = weld->parent; }
    if (!weld->spec.mocap) { throw mjCError(this, "plane only allowed in static bodies"); }
  }

  // normalize quaternion
  mjuu_normvec(quat, 4);

  // 'fromto': compute pos, quat, size
  if (mjuu_defined(fromto[0])) {
    // check type
    if (type != mjGEOM_CAPSULE &&
        type != mjGEOM_CYLINDER &&
        type != mjGEOM_ELLIPSOID &&
        type != mjGEOM_BOX) {
      throw mjCError(this, "fromto requires capsule, cylinder, box or ellipsoid in geom");
    }

    // make sure pos is not defined; cannot use mjuu_defined because default is (0,0,0)
    if (pos[0] || pos[1] || pos[2]) { throw mjCError(this, "both pos and fromto defined in geom"); }

    // size[1] = length (for capsule and cylinder)
    double vec[3] = {fromto[0] - fromto[3], fromto[1] - fromto[4], fromto[2] - fromto[5]};
    size[1]       = mjuu_normvec(vec, 3) / 2;
    if (size[1] < mjEPS) { throw mjCError(this, "fromto points too close in geom"); }

    // adjust size for ellipsoid and box
    if (type == mjGEOM_ELLIPSOID || type == mjGEOM_BOX) {
      size[2] = size[1];
      size[1] = size[0];
    }

    // compute position
    pos[0] = (fromto[0] + fromto[3]) / 2;
    pos[1] = (fromto[1] + fromto[4]) / 2;
    pos[2] = (fromto[2] + fromto[5]) / 2;

    // compute orientation
    mjuu_z2quat(quat, vec);
  }

  // not 'fromto': try alternative
  else {
    const char* err = ResolveOrientation(quat, compiler->degree, compiler->eulerseq, alt);
    if (err) { throw mjCError(this, "orientation specification error '%s' in geom %d", err, id); }
  }

  // mesh: accumulate frame, fit geom if needed
  if (mesh) {
    // check for inapplicable fromto
    if (mjuu_defined(fromto[0])) { throw mjCError(this, "fromto cannot be used with mesh geom"); }

    // save reference in case this is not an mjGEOM_MESH
    mjCMesh* pmesh     = mesh;
    double   center[3] = {0, 0, 0};

    // fit geom if type is not mjGEOM_MESH
    if (type != mjGEOM_MESH && type != mjGEOM_SDF) {
      mesh->FitGeom(this, center);

      // remove reference to mesh
      meshname_.clear();
      mesh = nullptr;
    } else if (typeinertia == mjINERTIA_SHELL) {
      throw mjCError(this, "for mesh geoms, inertia should be specified in the mesh asset");
    }

    // rotate center to geom frame and add it to mesh frame
    double meshpos[3];
    mjuu_rotVecQuat(meshpos, center, pmesh->GetQuatPtr());
    mjuu_addtovec(meshpos, pmesh->GetPosPtr(), 3);

    // accumulate mesh frame into geom frame
    mjuu_frameaccum(pos, quat, meshpos, pmesh->GetQuatPtr());

    // re-express surfacevel in the compiled geom frame, which absorbed the mesh frame
    if (surfacevel[0] ||
        surfacevel[1] ||
        surfacevel[2] ||
        surfacevel[3] ||
        surfacevel[4] ||
        surfacevel[5]) {
      // angular origin moves to meshpos: linear part gains omega x meshpos
      double wxp[3];
      mjuu_crossvec(wxp, surfacevel + 3, meshpos);
      mjuu_addtovec(surfacevel, wxp, 3);

      // rotate both parts by the inverse mesh orientation
      const double* mq      = pmesh->GetQuatPtr();
      double        invq[4] = {mq[0], -mq[1], -mq[2], -mq[3]};
      double        tmp[3];
      mjuu_rotVecQuat(tmp, surfacevel, invq);
      mjuu_copyvec(surfacevel, tmp, 3);
      mjuu_rotVecQuat(tmp, surfacevel + 3, invq);
      mjuu_copyvec(surfacevel + 3, tmp, 3);
    }
  }

  // check size parameters
  checksize(size, type, this, name.c_str(), id);

  // set hfield sizes in geom.size
  if (type == mjGEOM_HFIELD) {
    size[0] = hfield->size[0];
    size[1] = hfield->size[1];
    size[2] = 0.25 * hfield->size[2] + 0.5 * hfield->size[3];
  } else if (type == mjGEOM_MESH || type == mjGEOM_SDF) {
    const double* aamm = mesh->aamm();

    size[0] = std::max(std::abs(aamm[0]), std::abs(aamm[3]));
    size[1] = std::max(std::abs(aamm[1]), std::abs(aamm[4]));
    size[2] = std::max(std::abs(aamm[2]), std::abs(aamm[5]));
  }

  for (double s : size) {
    if (std::isnan(s)) { throw mjCError(this, "nan size in geom"); }
  }
  // compute aabb
  ComputeAABB();

  // compute geom mass and inertia
  if (inferinertia) {
    // mass is defined
    if (mjuu_defined(mass)) {
      if (mass == 0) {
        mass_   = 0;
        density = 0;
      } else if (GetVolume() > mjEPS) {
        mass_   = mass;
        density = mass / GetVolume();
        SetInertia();
      }
    }

    // mass is not defined
    else {
      if (density == 0) {
        mass_ = 0;
      } else {
        mass_ = density * GetVolume();
        SetInertia();
      }
    }


    // check for negative values
    if (mass_ < 0 || inertia[0] < 0 || inertia[1] < 0 || inertia[2] < 0 || density < 0)
      throw mjCError(this, "mass, inertia or density are negative in geom");
  }

  // fluid-interaction coefficients; mj_setConst computes the added mass and inertia from the size
  for (int i = 0; i < mjNFLUID; i++) { fluid[i] = 0; }
  if (fluid_ellipsoid > 0) {
    fluid[0] = fluid_ellipsoid;
    for (int i = 0; i < 5; i++) { fluid[i + 1] = fluid_coefs[i]; }
  }

  // plugin
  if (plugin.active) {
    if (plugin_name.empty() && plugin_instance_name.empty()) {
      throw mjCError(this, "neither 'plugin' nor 'instance' is specified for geom");
    }

    mjCPlugin* plugin_instance = static_cast<mjCPlugin*>(plugin.element);
    model->ResolvePlugin(this, plugin_name, plugin_instance_name, &plugin_instance);
    plugin.element           = plugin_instance;
    const mjpPlugin* pplugin = mjp_getPluginAtSlot(plugin_instance->plugin_slot);
    if (!(pplugin->capabilityflags & mjPLUGIN_SDF)) {
      throw mjCError(this, "plugin '%s' does not support sign distance fields", pplugin->name);
    }
  }

  // frame
  if (frame) { mjuu_frameaccumChild(frame->pos, frame->quat, pos, quat); }
}


// write to the spec the position and the orientation which compile to the compiled ones
void mjCGeom::PoseToSpec(bool position, bool orientation) {
  double specpos[3], specquat[4];
  mjuu_copyvec(specpos, pos, 3);
  mjuu_copyvec(specquat, quat, 4);

  // compilation adds the frame of the mesh to the pose of a mesh geom: an offset mesh offsets
  // the position by the orientation, which then gives both
  bool offset = false;
  if (mesh && (type == mjGEOM_MESH || type == mjGEOM_SDF)) {
    const double* meshpos = mesh->GetPosPtr();
    mjuu_frameaccuminv(specpos, specquat, meshpos, mesh->GetQuatPtr());
    offset = meshpos[0] || meshpos[1] || meshpos[2];
  }
  body->ElementPoseInSpec(frame, specpos, specquat);
  if (position || (orientation && offset)) { mjuu_copyvec(spec.pos, specpos, 3); }
  if (orientation) {
    mjuu_copyvec(spec.quat, specquat, 4);
    mjs_defaultOrientation(&spec.alt);
  }
}


// write to the spec the size, pose and surface velocity which compile to the compiled ones
void mjCGeom::ShapeToSpec(bool newsize, bool position, bool orientation, bool newvelocity) {
  const bool fitted  = type != mjGEOM_MESH && type != mjGEOM_SDF && !spec_meshname_.empty();
  const bool span    = mjuu_defined(spec.fromto[0]);
  const bool newpose = position || orientation;

  // the size in the units of the spec, by the rule of its type, which applies once fromto is
  // replaced
  const mjCGauge g = Gauge();
  double         specsize[3], specquat[4];
  mjuu_copyvec(specsize, size, 3);
  SpecShape(specquat, type, spec.fromto, spec.quat, spec.alt, compiler);
  g.Inverse().Size(specsize, type, specquat);

  // fromto does not give the radius of a capsule or cylinder: a new one leaves fromto as it is
  if (span && !newpose && (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER)) {
    double ends[6];
    mjuu_copyvec(ends, spec.fromto, 6);
    g.Point(ends);
    g.Point(ends + 3);
    double vec[3] = {ends[0] - ends[3], ends[1] - ends[4], ends[2] - ends[5]};
    if (size[1] == mjuu_normvec(vec, 3) / 2) {
      spec.size[0] = specsize[0];
      newsize      = false;
    }
  }

  // a size and pose which are given by fromto, or by fitting the geom to a mesh, are written in
  // its place; the surface velocity is then in the compiled frame of the geom
  if ((span && (newsize || newpose)) || (fitted && (newsize || newpose || newvelocity))) {
    spec.fromto[0] = mjNAN;
    if (fitted) {
      // a geom without a material of its own has that of its mesh, which it would lose
      if (spec_material_.empty()) {
        const mjCMesh* pmesh =
            static_cast<const mjCMesh*>(model->FindObject(mjOBJ_MESH, spec_meshname_));
        if (pmesh) { spec_material_ = pmesh->Material(); }
      }
      spec_meshname_.clear();
      newvelocity = true;
    }
    newsize = position = orientation = true;
  }

  if (newsize) { mjuu_copyvec(spec.size, specsize, 3); }
  PoseToSpec(position, orientation);

  // compilation expresses the surface velocity of a mesh geom in the frame of the mesh: rotate
  // both parts back, and move the origin of the angular part back from the mesh position
  if (newvelocity) {
    mjuu_copyvec(spec.surfacevel, surfacevel, 6);
    if (mesh && (type == mjGEOM_MESH || type == mjGEOM_SDF)) {
      double wxp[3];
      mjuu_rotVecQuat(spec.surfacevel, surfacevel, mesh->GetQuatPtr());
      mjuu_rotVecQuat(spec.surfacevel + 3, surfacevel + 3, mesh->GetQuatPtr());
      mjuu_crossvec(wxp, spec.surfacevel + 3, mesh->GetPosPtr());
      for (int i = 0; i < 3; i++) { spec.surfacevel[i] -= wxp[i]; }
    }
    Gauge().Inverse().Point(spec.surfacevel);
  }
}


//------------------ class mjCSite implementation --------------------------------------------------

// initialize default site
mjCSite::mjCSite(mjCModel* _model, mjCDef* _def) {
  mjs_defaultSite(&spec);
  elemtype = mjOBJ_SITE;

  // clear internal variables
  body  = 0;
  matid = -1;
  mesh  = 0;
  spec_material_.clear();
  spec_meshname_.clear();
  spec_userdata_.clear();

  // reset to default if given
  if (_def) { *this = _def->Site(); }

  // point to local
  PointToLocal();

  // in case this site is not compiled
  CopyFromSpec();

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";
}


mjCSite::mjCSite(const mjCSite& other) {
  *this = other;
}


mjCSite& mjCSite::operator=(const mjCSite& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCSite_*>(this) = static_cast<const mjCSite_&>(other);
    *static_cast<mjsSite*>(this)  = static_cast<const mjsSite&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCSite::PointToLocal() {
  spec.element  = static_cast<mjsElement*>(this);
  spec.info     = &info;
  spec.material = &spec_material_;
  spec.meshname = &spec_meshname_;
  spec.userdata = &spec_userdata_;
  userdata      = nullptr;
  material      = nullptr;
  meshname      = nullptr;
}


void mjCSite::CopyFromSpec() {
  *static_cast<mjsSite*>(this) = spec;

  userdata_ = spec_userdata_;
  material_ = spec_material_;
  meshname_ = spec_meshname_;
}


void mjCSite::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!spec_material_.empty() && model != m) {
    spec_material_ = m->prefix + spec_material_ + m->suffix;
  }
  if (!spec_meshname_.empty() && model != m) {
    spec_meshname_ = m->prefix + spec_meshname_ + m->suffix;
  }
}


const std::string& mjCSite::get_material() const {
  if (mesh && spec_material_.empty()) { return mesh->Material(); }
  return spec_material_;
}


// scale the compiled copy by the gauge, or back
// map of the gauge in the frame of this site, which its mesh instance takes
std::array<double, 9> mjCSite::ShapeMap() const {
  mjCGauge g = Gauge();
  if (!g.uniform) {
    double quat_[4];
    SpecShape(quat_, spec.type, spec.fromto, spec.quat, spec.alt, compiler);
    g = g.Rotated(quat_);
  }
  std::array<double, 9> map;
  std::copy(g.map, g.map + 9, map.begin());
  return map;
}


void mjCSite::ApplyGauge(bool inverse) {
  const mjCGauge g = inverse ? Gauge().Inverse() : Gauge();
  if (g.IsIdentity()) { return; }
  double      quat_[4];
  const int   shape = SpecShape(quat_, type, spec.fromto, spec.quat, spec.alt, compiler);
  const char* err   = g.Size(size, shape, quat_, /*site=*/true);
  if (err) { throw mjCError(this, "%s", err); }
  g.Point(pos);
  if (mjuu_defined(fromto[0])) {
    g.Point(fromto);
    g.Point(fromto + 3);
  }
}


// compiler
void mjCSite::Compile(void) {
  CopyFromSpec();
  ApplyGauge();

  // resize userdata
  if (userdata_.size() > model->nuser_site) {
    throw mjCError(this, "user has more values than nuser_site in site");
  }
  userdata_.resize(model->nuser_site);

  // check type
  if (type < 0 || type >= mjNGEOMTYPES) { throw mjCError(this, "invalid type in site"); }

  // do not allow hfields and planes
  if (type == mjGEOM_HFIELD || type == mjGEOM_PLANE) {
    throw mjCError(this, "hfields and planes not allowed in site");
  }

  // check mesh
  if (type == mjGEOM_MESH && !mesh) {
    throw mjCError(this, "mesh site '%s' (id = %d) must have valid meshid", name.c_str(), id);
  }
  if (type != mjGEOM_MESH && mesh) {
    throw mjCError(this, "mesh can only be specified for mesh sites");
  }

  // 'fromto': compute pos, quat, size
  if (mjuu_defined(fromto[0])) {
    // check type
    if (type != mjGEOM_CAPSULE &&
        type != mjGEOM_CYLINDER &&
        type != mjGEOM_ELLIPSOID &&
        type != mjGEOM_BOX) {
      throw mjCError(this, "fromto requires capsule, cylinder, box or ellipsoid in site");
    }

    // make sure pos is not defined; cannot use mjuu_defined because default is (0,0,0)
    if (pos[0] || pos[1] || pos[2]) { throw mjCError(this, "both pos and fromto defined in site"); }

    // size[1] = length (for capsule and cylinder)
    double vec[3] = {fromto[0] - fromto[3], fromto[1] - fromto[4], fromto[2] - fromto[5]};
    size[1]       = mjuu_normvec(vec, 3) / 2;
    if (size[1] < mjEPS) { throw mjCError(this, "fromto points too close in site"); }

    // adjust size for ellipsoid and box
    if (type == mjGEOM_ELLIPSOID || type == mjGEOM_BOX) {
      size[2] = size[1];
      size[1] = size[0];
    }

    // compute position
    pos[0] = (fromto[0] + fromto[3]) / 2;
    pos[1] = (fromto[1] + fromto[4]) / 2;
    pos[2] = (fromto[2] + fromto[5]) / 2;

    // compute orientation
    mjuu_z2quat(quat, vec);
  }

  // alternative orientation
  else {
    const char* err = ResolveOrientation(quat, compiler->degree, compiler->eulerseq, alt);
    if (err) { throw mjCError(this, "orientation specification error '%s' in site %d", err, id); }
  }

  // mesh: set size
  if (mesh) {
    if (mjuu_defined(fromto[0])) { throw mjCError(this, "fromto cannot be used with mesh site"); }

    const double* aamm = mesh->aamm();

    size[0] = std::max(std::abs(aamm[0]), std::abs(aamm[3]));
    size[1] = std::max(std::abs(aamm[1]), std::abs(aamm[4]));
    size[2] = std::max(std::abs(aamm[2]), std::abs(aamm[5]));
  }

  // frame
  if (frame) { mjuu_frameaccumChild(frame->pos, frame->quat, pos, quat); }

  // normalize quaternion
  mjuu_normvec(quat, 4);

  // check size parameters
  checksize(size, type, this, name.c_str(), id);
}


// write to the spec the position and the orientation which compile to the compiled ones
void mjCSite::PoseToSpec(bool position, bool orientation) {
  double specpos[3], specquat[4];
  mjuu_copyvec(specpos, pos, 3);
  mjuu_copyvec(specquat, quat, 4);
  body->ElementPoseInSpec(frame, specpos, specquat);
  if (position) { mjuu_copyvec(spec.pos, specpos, 3); }
  if (orientation) {
    mjuu_copyvec(spec.quat, specquat, 4);
    mjs_defaultOrientation(&spec.alt);
  }
}


// write to the spec the size and pose which compile to the compiled ones
void mjCSite::ShapeToSpec(bool newsize, bool position, bool orientation) {
  const bool span    = mjuu_defined(spec.fromto[0]);
  const bool newpose = position || orientation;

  // the size in the units of the spec, by the rule of its type, which applies once fromto is
  // replaced; dividing by the stretch inverts it when it is approximate
  const mjCGauge g = Gauge();
  double         specsize[3], specquat[4], f[3] = {1, 1, 1};
  SpecShape(specquat, type, spec.fromto, spec.quat, spec.alt, compiler);
  g.Size(f, type, specquat, /*site=*/true);
  for (int i = 0; i < 3; i++) { specsize[i] = size[i] / f[i]; }

  // fromto does not give the radius of a capsule or cylinder: a new one leaves fromto as it is
  if (span && !newpose && (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER)) {
    double ends[6];
    mjuu_copyvec(ends, spec.fromto, 6);
    g.Point(ends);
    g.Point(ends + 3);
    double vec[3] = {ends[0] - ends[3], ends[1] - ends[4], ends[2] - ends[5]};
    if (size[1] == mjuu_normvec(vec, 3) / 2) {
      spec.size[0] = specsize[0];
      newsize      = false;
    }
  }

  // a size and pose which are given by fromto are written in its place
  if (span && (newsize || newpose)) {
    spec.fromto[0] = mjNAN;
    newsize = position = orientation = true;
  }

  if (newsize) { mjuu_copyvec(spec.size, specsize, 3); }
  PoseToSpec(position, orientation);
}


//------------------ class mjCCamera implementation ------------------------------------------------

// initialize defaults
mjCCamera::mjCCamera(mjCModel* _model, mjCDef* _def) {
  mjs_defaultCamera(&spec);
  elemtype = mjOBJ_CAMERA;

  // clear private variables
  body         = 0;
  targetbodyid = -1;
  spec_targetbody_.clear();

  // reset to default if given
  if (_def) { *this = _def->Camera(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // point to local
  PointToLocal();

  // in case this camera is not compiled
  CopyFromSpec();
}


mjCCamera::mjCCamera(const mjCCamera& other) {
  *this = other;
}


mjCCamera& mjCCamera::operator=(const mjCCamera& other) {
  if (this != &other) {
    this->spec                      = other.spec;
    *static_cast<mjCCamera_*>(this) = static_cast<const mjCCamera_&>(other);
    *static_cast<mjsCamera*>(this)  = static_cast<const mjsCamera&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCCamera::PointToLocal() {
  spec.element    = static_cast<mjsElement*>(this);
  spec.userdata   = &spec_userdata_;
  spec.targetbody = &spec_targetbody_;
  spec.info       = &info;
  userdata        = nullptr;
  targetbody      = nullptr;
}


void mjCCamera::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!spec_targetbody_.empty()) { spec_targetbody_ = m->prefix + spec_targetbody_ + m->suffix; }
}


void mjCCamera::CopyFromSpec() {
  *static_cast<mjsCamera*>(this) = spec;

  userdata_   = spec_userdata_;
  targetbody_ = spec_targetbody_;
}


void mjCCamera::ResolveReferences(const mjCModel* m) {
  targetbodyid = -1;
  if (!targetbody_.empty()) {
    mjCBody* tb = (mjCBody*)m->FindObject(mjOBJ_BODY, targetbody_);
    if (tb) {
      targetbodyid = tb->id;
    } else {
      throw mjCError(this, "unknown target body in camera");
    }
  }
}


// scale the compiled copy by the gauge, or back; the image is unchanged
void mjCCamera::ApplyGauge(bool inverse) {
  const mjCGauge g = inverse ? Gauge().Inverse() : Gauge();
  if (g.IsIdentity()) { return; }
  const double fl = g.Shape(1);
  g.Point(pos);
  ipd *= fl;
  for (int i = 0; i < 2; i++) {
    sensor_size[i]      *= fl;
    focal_length[i]     *= fl;
    principal_length[i] *= fl;
  }
}


// compiler
void mjCCamera::Compile(void) {
  CopyFromSpec();
  ApplyGauge();

  // resize userdata
  if (userdata_.size() > model->nuser_cam) {
    throw mjCError(this, "user has more values than nuser_cam in camera");
  }
  userdata_.resize(model->nuser_cam);

  // process orientation specifications
  const char* err = ResolveOrientation(quat, compiler->degree, compiler->eulerseq, alt);
  if (err) { throw mjCError(this, "orientation specification error '%s' in camera %d", err, id); }

  // frame
  if (frame) { mjuu_frameaccumChild(frame->pos, frame->quat, pos, quat); }

  // normalize quaternion
  mjuu_normvec(quat, 4);

  // get targetbodyid
  ResolveReferences(model);

  // make sure the image size is finite
  if (fovy >= 180) {
    throw mjCError(this,
                   "fovy too large in camera '%s' (id = %d, value = %d)",
                   name.c_str(),
                   id,
                   fovy);
  }

  // check for advanced camera intrinsic parameters
  bool has_intrinsic  = focal_length[0] ||
                        focal_length[1] ||
                        focal_pixel[0] ||
                        focal_pixel[1] ||
                        principal_length[0] ||
                        principal_length[1] ||
                        principal_pixel[0] ||
                        principal_pixel[1];
  bool has_sensorsize = sensor_size[0] > 0 && sensor_size[1] > 0;

  // intrinsic params require sensorsize
  if (has_intrinsic && !has_sensorsize) {
    throw mjCError(this,
                   "focal/principal require sensorsize in camera '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  // sensorsize requires resolution
  if (has_sensorsize && (resolution[0] <= 0 || resolution[1] <= 0)) {
    throw mjCError(this,
                   "sensorsize requires positive resolution in camera '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  // compute number of pixels per unit length
  if (sensor_size[0] > 0 && sensor_size[1] > 0) {
    float pixel_density[2] = {
        (float)resolution[0] / sensor_size[0],
        (float)resolution[1] / sensor_size[1],
    };

    // pixel values override length values when both are specified
    intrinsic[0] = focal_pixel[0] ? focal_pixel[0] / pixel_density[0] : focal_length[0];
    intrinsic[1] = focal_pixel[1] ? focal_pixel[1] / pixel_density[1] : focal_length[1];
    intrinsic[2] = principal_pixel[0] ? principal_pixel[0] / pixel_density[0] : principal_length[0];
    intrinsic[3] = principal_pixel[1] ? principal_pixel[1] / pixel_density[1] : principal_length[1];

    // fovy with principal point at (0, 0)
    fovy = std::atan2(sensor_size[1] / 2, intrinsic[1]) * 360.0 / mjPI;
  } else {
    intrinsic[0] = model->visual.map.znear;
    intrinsic[1] = model->visual.map.znear;
  }
}


// write to the spec the position and the orientation which compile to the compiled ones
void mjCCamera::PoseToSpec(bool position, bool orientation) {
  double specpos[3], specquat[4];
  mjuu_copyvec(specpos, pos, 3);
  mjuu_copyvec(specquat, quat, 4);
  body->ElementPoseInSpec(frame, specpos, specquat);
  if (position) { mjuu_copyvec(spec.pos, specpos, 3); }
  if (orientation) {
    mjuu_copyvec(spec.quat, specquat, 4);
    mjs_defaultOrientation(&spec.alt);
  }
}


// write the compiled intrinsics as focal length and principal point, in units of length, which are
// those of the spec in the spec
void mjCCamera::IntrinsicToSpec(bool tospec) {
  for (mjsCamera* camera : {static_cast<mjsCamera*>(this), tospec ? &spec : nullptr}) {
    if (!camera) { continue; }
    const double fl = camera == &spec ? Gauge().Inverse().Shape(1) : 1;
    for (int i = 0; i < 2; i++) {
      camera->focal_length[i]     = fl * intrinsic[i];
      camera->principal_length[i] = fl * intrinsic[i + 2];
      camera->focal_pixel[i]      = 0;
      camera->principal_pixel[i]  = 0;
    }
  }
}


//------------------ class mjCLight implementation -------------------------------------------------

// initialize defaults
mjCLight::mjCLight(mjCModel* _model, mjCDef* _def) {
  mjs_defaultLight(&spec);
  elemtype = mjOBJ_LIGHT;

  // clear private variables
  body         = 0;
  targetbodyid = -1;
  texid        = -1;
  spec_targetbody_.clear();
  spec_texture_.clear();


  // reset to default if given
  if (_def) { *this = _def->Light(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  PointToLocal();
  CopyFromSpec();
}


mjCLight::mjCLight(const mjCLight& other) {
  *this = other;
}


mjCLight& mjCLight::operator=(const mjCLight& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCLight_*>(this) = static_cast<const mjCLight_&>(other);
    *static_cast<mjsLight*>(this)  = static_cast<const mjsLight&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCLight::PointToLocal() {
  spec.element    = static_cast<mjsElement*>(this);
  spec.targetbody = &spec_targetbody_;
  spec.texture    = &spec_texture_;
  spec.info       = &info;
  targetbody      = nullptr;
}


void mjCLight::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!spec_targetbody_.empty()) { spec_targetbody_ = m->prefix + spec_targetbody_ + m->suffix; }
  if (!spec_texture_.empty()) { spec_texture_ = m->prefix + spec_texture_ + m->suffix; }
}


void mjCLight::CopyFromSpec() {
  *static_cast<mjsLight*>(this) = spec;

  targetbody_ = spec_targetbody_;
  texture_    = spec_texture_;
}


void mjCLight::ResolveReferences(const mjCModel* m) {
  targetbodyid = -1;
  texid        = -1;
  if (!targetbody_.empty()) {
    mjCBody* tb = (mjCBody*)m->FindObject(mjOBJ_BODY, targetbody_);
    if (tb) {
      targetbodyid = tb->id;
    } else {
      throw mjCError(this, "unknown target body in light");
    }
  }
  if (!texture_.empty()) {
    mjCTexture* tex = (mjCTexture*)m->FindObject(mjOBJ_TEXTURE, texture_);
    if (tex) {
      texid = tex->id;
    } else {
      throw mjCError(this, "unknown texture in light");
    }
  }
}


// scale the compiled copy by the gauge, or back; attenuation is 1, L^-1, L^-2
void mjCLight::ApplyGauge(bool inverse) {
  const mjCGauge g = inverse ? Gauge().Inverse() : Gauge();
  if (g.IsIdentity()) { return; }
  const double fl = g.Shape(1);
  g.Point(pos);
  bulbradius     *= fl;
  range          *= fl;
  attenuation[1] /= fl;
  attenuation[2] /= fl * fl;
}


// compiler
void mjCLight::Compile(void) {
  CopyFromSpec();
  ApplyGauge();

  // frame
  if (frame) {
    // apply frame transform to pos, qunit is unused
    double qunit[4] = {1, 0, 0, 0};
    mjuu_frameaccumChild(frame->pos, frame->quat, pos, qunit);

    // rotate dir
    mjuu_rotVecQuat(dir, dir, frame->quat);
  }

  // normalize direction, make sure it is not zero
  if (mjuu_normvec(dir, 3) < mjEPS) { throw mjCError(this, "zero direction in light"); }

  // check softness range
  if (softness < 0 || softness > 1) { throw mjCError(this, "light softness must be in [0, 1]"); }

  // get targetbodyid and texid
  ResolveReferences(model);
}


// write to the spec the position and the direction which compile to the compiled ones
void mjCLight::PoseToSpec(bool position, bool direction) {
  double specpos[3], rotation[4] = {1, 0, 0, 0};
  mjuu_copyvec(specpos, pos, 3);
  body->ElementPoseInSpec(frame, specpos, rotation);
  if (position) { mjuu_copyvec(spec.pos, specpos, 3); }
  if (direction) { mjuu_rotVecQuat(spec.dir, dir, rotation); }
}


//------------------------- class mjCHField --------------------------------------------------------

// constructor
mjCHField::mjCHField(mjCModel* _model) {
  mjs_defaultHField(&spec);
  elemtype = mjOBJ_HFIELD;

  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear variables
  data.clear();
  spec_file_.clear();
  spec_userdata_.clear();

  // point to local
  PointToLocal();

  // copy from spec
  CopyFromSpec();
}


mjCHField::mjCHField(const mjCHField& other) {
  *this = other;
}


mjCHField& mjCHField::operator=(const mjCHField& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCHField_*>(this) = static_cast<const mjCHField_&>(other);
    *static_cast<mjsHField*>(this)  = static_cast<const mjsHField&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCHField::PointToLocal() {
  spec.element      = static_cast<mjsElement*>(this);
  spec.file         = &spec_file_;
  spec.content_type = &spec_content_type_;
  spec.userdata     = &spec_userdata_;
  spec.info         = &info;
  file              = nullptr;
  content_type      = nullptr;
  userdata          = nullptr;
}


void mjCHField::CopyFromSpec() {
  *static_cast<mjsHField*>(this) = spec;
  const auto& m                  = instancemap_;
  for (int i = 0; i < 4; i++) {
    int    j   = std::min(i, 2);
    double off = 0;
    for (int k = 0; k < 3; k++) { off += k == j ? 0 : m[3 * k + j] * m[3 * k + j]; }
    size[i] *= off ? std::sqrt(off + m[4 * j] * m[4 * j]) : m[4 * j];
  }

  file_         = spec_file_;
  content_type_ = spec_content_type_;
  userdata_     = spec_userdata_;

  // clear precompiled asset. TODO: use asset cache
  data.clear();
  if (!file_.empty()) {
    nrow = 0;
    ncol = 0;
  }
}


// destructor
mjCHField::~mjCHField() {
  data.clear();
  userdata_.clear();
  spec_userdata_.clear();
}


// file as opened by Compile, which has already applied strippath to file_
std::string mjCHField::ResolvedFilePath() const {
  return ResolvePath(mjs_getString(compiler->meshdir), file_);
}


std::string mjCHField::GetCacheId(const mjResource* resource, const std::string& asset_type) {
  std::stringstream ss;
  ss << "mjCHField:" << resource->name << ";ARGS:content_type=" << asset_type;
  return ss.str();
}


// load elevation data from custom format
void mjCHField::LoadCustom(mjResource* resource) {
  // get file data in buffer
  const void* buffer    = 0;
  int         buffer_sz = mju_readResource(resource, &buffer);

  if (buffer_sz < 1) {
    throw mjCError(this, "could not read hfield file '%s'", resource->name);
  } else if (!buffer_sz) {
    throw mjCError(this, "empty hfield file '%s'", resource->name);
  }


  if (buffer_sz < 2 * sizeof(int)) {
    throw mjCError(this, "hfield missing header '%s'", resource->name);
  }

  // read dimensions
  int* pint = (int*)buffer;
  nrow      = pint[0];
  ncol      = pint[1];

  // check dimensions
  if (nrow < 2 || ncol < 2) {
    throw mjCError(this, "hfield dimensions must be at least 2 in file '%s'", resource->name);
  }

  // check buffer size
  if (buffer_sz != nrow * ncol * sizeof(float) + 8) {
    throw mjCError(this, "unexpected file size in file '%s'", resource->name);
  }

  // allocate
  data.assign(nrow * ncol, 0);
  if (data.empty()) { throw mjCError(this, "could not allocate buffers in hfield"); }

  // copy data
  memcpy(data.data(), (void*)(pint + 2), nrow * ncol * sizeof(float));
}


// load elevation data from PNG format
void mjCHField::LoadPNG(mjResource* resource) {
  PNGImage image = PNGImage::Load(this, resource, LCT_GREY);

  ncol = image.Width();
  nrow = image.Height();
  if (nrow < 2 || ncol < 2) {
    throw mjCError(this, "hfield dimensions must be at least 2 in file '%s'", resource->name);
  }

  // copy image data over with rows reversed
  data.reserve(nrow * ncol);
  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) { data.push_back((float)image[c + (nrow - 1 - r) * ncol]); }
  }
}


// compiler
void mjCHField::Compile(const mjVFS* vfs) {
  CopyFromSpec();

  // copy userdata into data
  if (!userdata_.empty()) {
    if (nrow < 2 || ncol < 2) { throw mjCError(this, "hfield dimensions must be at least 2"); }
    if (nrow * ncol != userdata_.size()) {
      throw mjCError(this, "elevation data length must match nrow*ncol");
    }
    data.assign(nrow * ncol, 0);
    if (data.empty()) { throw mjCError(this, "could not allocate buffers in hfield"); }
    memcpy(data.data(), userdata_.data(), nrow * ncol * sizeof(float));
  }

  // check size parameters
  for (int i = 0; i < 4; i++)
    if (size[i] <= 0) throw mjCError(this, "size parameter is not positive in hfield");

  // remove path from file if necessary
  if (model->strippath) { file_ = mjuu_strippath(file_); }

  // load from file if specified
  if (!file_.empty()) {
    // make sure hfield was not already specified manually
    if (nrow || ncol || !data.empty()) {
      throw mjCError(this, "hfield specified from file and manually");
    }

    mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);

    std::string asset_type = GetAssetContentType(file_, content_type_);

    // fallback to custom
    if (asset_type.empty()) { asset_type = "image/vnd.mujoco.hfield"; }

    if (asset_type != "image/png" && asset_type != "image/vnd.mujoco.hfield") {
      throw mjCError(this, "unsupported content type: '%s'", asset_type.c_str());
    }

    mujoco::user::FilePath meshdir_;
    meshdir_ = FilePath(mjs_getString(compiler->meshdir));

    FilePath    filename    = meshdir_ + FilePath(file_);
    mjSpec*     owning_spec = model->FindSpec(compiler);
    mjResource* resource    = LoadResource(owning_spec->modelfiledir->c_str(), filename.Str(), vfs);

    struct CachedHField {
      int                nrow, ncol;
      std::vector<float> data;
    };

    // cache callback
    auto callback = [&](const void* cached_data) {
      const CachedHField* cached_hfield = static_cast<const CachedHField*>(cached_data);

      nrow       = cached_hfield->nrow;
      ncol       = cached_hfield->ncol;
      this->data = cached_hfield->data;
      return true;
    };

    // try loading from cache
    if (cache && cache->PopulateData(GetCacheId(resource, asset_type), resource, callback)) {
      mju_closeResource(resource);
    } else {
      try {
        if (asset_type == "image/png") {
          LoadPNG(resource);
        } else {
          LoadCustom(resource);
        }
      } catch (mjCError err) {
        mju_closeResource(resource);
        throw err;
      }

      if (cache) {
        CachedHField* cached_hfield = new CachedHField;

        cached_hfield->nrow = nrow;
        cached_hfield->ncol = ncol;
        cached_hfield->data = this->data;
        std::size_t size    = sizeof(CachedHField) + this->data.size() * sizeof(float);
        std::shared_ptr<CachedHField> cached_data{cached_hfield};
        cache->Insert("", GetCacheId(resource, asset_type), resource, cached_data, size);
      }
      mju_closeResource(resource);
    }
  }

  // make sure hfield was specified (from file or manually)
  if (nrow < 1 || ncol < 1 || data.empty()) { throw mjCError(this, "hfield not specified"); }
  if (nrow < 2 || ncol < 2) { throw mjCError(this, "hfield dimensions must be at least 2"); }

  // set elevation data to [0-1] range
  float emin = 1E+10, emax = -1E+10;
  for (int i = 0; i < nrow * ncol; i++) {
    emin = std::min(emin, data[i]);
    emax = std::max(emax, data[i]);
  }
  if (emin > emax) { throw mjCError(this, "invalid data range in hfield '%s'", file_.c_str()); }
  for (int i = 0; i < nrow * ncol; i++) {
    data[i] -= emin;
    if (emax - emin > mjEPS) { data[i] /= (emax - emin); }
  }
}


//------------------ class mjCTexture implementation -----------------------------------------------

// initialize defaults
mjCTexture::mjCTexture(mjCModel* _model, mjCDef* _def) {
  mjs_defaultTexture(&spec);
  elemtype = mjOBJ_TEXTURE;

  // clear user settings: single file
  spec_file_.clear();
  spec_content_type_.clear();

  // clear user settings: separate file
  spec_cubefiles_.assign(6, "");

  // clear internal variables
  data_.clear();
  spec_data_.clear();

  // reset to default if given
  if (_def) { *this = _def->Texture(); }

  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = (_def ? _def->name : (_model ? "main" : ""));

  // point to local
  PointToLocal();

  // in case this texture is not compiled
  CopyFromSpec();
}


mjCTexture::mjCTexture(const mjCTexture& other) {
  *this = other;
}


mjCTexture& mjCTexture::operator=(const mjCTexture& other) {
  if (this != &other) {
    this->spec                       = other.spec;
    *static_cast<mjCTexture_*>(this) = static_cast<const mjCTexture_&>(other);
    *static_cast<mjsTexture*>(this)  = static_cast<const mjsTexture&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCTexture::PointToLocal() {
  spec.element      = static_cast<mjsElement*>(this);
  spec.file         = &spec_file_;
  spec.data         = &spec_data_;
  spec.content_type = &spec_content_type_;
  spec.cubefiles    = &spec_cubefiles_;
  spec.info         = &info;
  file              = nullptr;
  content_type      = nullptr;
  cubefiles         = nullptr;
}


// the key under which textures which would compile to the same asset are deduplicated: every
// field of the spec which compilation reads, names excluded, with the files resolved as the
// compiler resolves them; the pixel buffer is compared separately
std::string mjCTexture::AssetKey() const {
#define X(type, name, dim) +1
#define XVEC(type, name, dim) +1
  static_assert((0 MJSTEXTURE_FIELDS) == 22,
                "a field was added to mjsTexture: decide whether AssetKey reads it");
#undef X
#undef XVEC
  const mjsTexture& s   = spec;
  std::string       dir = mjs_getString(compiler->texturedir);
  std::string       key;
  mjuu_addkey(key, ResolvedSpecFile(dir, spec_file_));
  for (const std::string& cubefile : spec_cubefiles_) {
    mjuu_addkey(key, ResolvedSpecFile(dir, cubefile));
  }
  mjuu_addkey(key, spec_content_type_);
  mjuu_addkey(key, s.type);
  mjuu_addkey(key, s.colorspace);
  mjuu_addkey(key, s.builtin);
  mjuu_addkey(key, s.mark);
  mjuu_addkey(key, s.rgb1);
  mjuu_addkey(key, s.rgb2);
  mjuu_addkey(key, s.markrgb);
  mjuu_addkey(key, s.random);
  mjuu_addkey(key, s.height);
  mjuu_addkey(key, s.width);
  mjuu_addkey(key, s.nchannel);
  mjuu_addkey(key, s.gridsize);
  mjuu_addkey(key, s.gridlayout);
  mjuu_addkey(key, s.hflip);
  mjuu_addkey(key, s.vflip);
  mjuu_addkey(key, s.embed);
  mjuu_addkey(key, spec_data_.size());
  return key;
}


void mjCTexture::CopyFromSpec() {
  *static_cast<mjsTexture*>(this) = spec;

  file_         = spec_file_;
  content_type_ = spec_content_type_;
  cubefiles_    = spec_cubefiles_;

  // the buffer given by the user, if any; otherwise Compile fills the data
  data_ = spec_data_;
}


// free data storage allocated by lodepng
mjCTexture::~mjCTexture() {
  data_.clear();
}


// file as opened by Compile, which has already applied strippath to file_
std::string mjCTexture::ResolvedFilePath() const {
  return ResolvePath(mjs_getString(compiler->texturedir), file_);
}


// insert random dots
static void randomdot(
    std::byte* rgb, const double* markrgb, int width, int height, double probability) {
  // make distribution using fixed seed
  std::mt19937_64 rng;
  rng.seed(42);
  std::uniform_real_distribution<double> dist(0, 1);

  // sample
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      if (dist(rng) < probability) {
        for (int j = 0; j < 3; j++) {
          rgb[3 * (r * width + c) + j] = (std::byte)(255 * markrgb[j]);
        }
      }
    }
  }
}


// interpolate between colors based on value in (-1, +1)
static void interp(std::byte* rgb, const double* rgb1, const double* rgb2, double pos) {
  const double correction = 1.0 / sqrt(2);
  double       alpha      = 0.5 * (1 + pos / sqrt(1 + pos * pos) / correction);
  if (alpha < 0) {
    alpha = 0;
  } else if (alpha > 1) {
    alpha = 1;
  }

  for (int j = 0; j < 3; j++) {
    rgb[j] = (std::byte)(255 * (alpha * rgb1[j] + (1 - alpha) * rgb2[j]));
  }
}


// make checker pattern for one side
static void checker(
    std::byte* rgb, const std::byte* RGB1, const std::byte* RGB2, int width, int height) {
  for (int r = 0; r < height / 2; r++) {
    for (int c = 0; c < width / 2; c++) { memcpy(rgb + 3 * (r * width + c), RGB1, 3); }
  }
  for (int r = height / 2; r < height; r++) {
    for (int c = width / 2; c < width; c++) { memcpy(rgb + 3 * (r * width + c), RGB1, 3); }
  }
  for (int r = 0; r < height / 2; r++) {
    for (int c = width / 2; c < width; c++) { memcpy(rgb + 3 * (r * width + c), RGB2, 3); }
  }
  for (int r = height / 2; r < height; r++) {
    for (int c = 0; c < width / 2; c++) { memcpy(rgb + 3 * (r * width + c), RGB2, 3); }
  }
}


// make builtin: 2D
void mjCTexture::Builtin2D(void) {
  std::byte RGB1[3], RGB2[3], RGBm[3];
  // convert fixed colors
  for (int j = 0; j < 3; j++) {
    RGB1[j] = (std::byte)(255 * rgb1[j]);
    RGB2[j] = (std::byte)(255 * rgb2[j]);
    RGBm[j] = (std::byte)(255 * markrgb[j]);
  }

  //------------------ face

  // gradient
  if (builtin == mjBUILTIN_GRADIENT) {
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        // compute normalized coordinates and radius
        double x   = 2 * c / ((double)(width - 1)) - 1;
        double y   = 1 - 2 * r / ((double)(height - 1));
        double pos = 2 * sqrt(x * x + y * y) - 1;

        // interpolate through sigmoid
        interp(data_.data() + 3 * (r * width + c), rgb2, rgb1, pos);
      }
    }
  }

  // checker
  else if (builtin == mjBUILTIN_CHECKER) {
    checker(data_.data(), RGB1, RGB2, width, height);
  }

  // flat
  else if (builtin == mjBUILTIN_FLAT) {
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) { memcpy(data_.data() + 3 * (r * width + c), RGB1, 3); }
    }
  }

  //------------------ marks

  // edge
  if (mark == mjMARK_EDGE) {
    for (int r = 0; r < height; r++) {
      memcpy(data_.data() + 3 * (r * width + 0), RGBm, 3);
      memcpy(data_.data() + 3 * (r * width + width - 1), RGBm, 3);
    }
    for (int c = 0; c < width; c++) {
      memcpy(data_.data() + 3 * (0 * width + c), RGBm, 3);
      memcpy(data_.data() + 3 * ((height - 1) * width + c), RGBm, 3);
    }
  }

  // cross
  else if (mark == mjMARK_CROSS) {
    for (int r = 0; r < height; r++) {
      memcpy(data_.data() + 3 * (r * width + width / 2), RGBm, 3);
    }
    for (int c = 0; c < width; c++) {
      memcpy(data_.data() + 3 * (height / 2 * width + c), RGBm, 3);
    }
  }

  // random dots
  else if (mark == mjMARK_RANDOM && random > 0) {
    randomdot(data_.data(), markrgb, width, height, random);
  }
}


// make builtin: Cube
void mjCTexture::BuiltinCube(void) {
  std::byte RGB1[3], RGB2[3], RGBm[3], RGBi[3];
  int       w = width;
  if (w > std::numeric_limits<int>::max() / w) {
    throw mjCError(this, "Cube texture width is too large.");
  }
  mjtSize ww = width * width;

  // convert fixed colors
  for (int j = 0; j < 3; j++) {
    RGB1[j] = (std::byte)(255 * rgb1[j]);
    RGB2[j] = (std::byte)(255 * rgb2[j]);
    RGBm[j] = (std::byte)(255 * markrgb[j]);
  }

  //------------------ faces

  // gradient
  if (builtin == mjBUILTIN_GRADIENT) {
    if (ww > std::numeric_limits<std::int64_t>::max() / 18) {
      throw mjCError(this, "Gradient texture width is too large.");
    }
    for (int r = 0; r < w; r++) {
      for (int c = 0; c < w; c++) {
        // compute normalized pixel coordinates
        double x = 2 * c / ((double)(w - 1)) - 1;
        double y = 1 - 2 * r / ((double)(w - 1));

        // compute normalized elevation for sides and up/down
        double elside = asin(y / sqrt(1 + x * x + y * y)) / (0.5 * mjPI);
        double elup   = 1 - acos(1.0 / sqrt(1 + x * x + y * y)) / (0.5 * mjPI);

        // set sides
        interp(RGBi, rgb1, rgb2, elside);
        memcpy(data_.data() + 0 * 3 * ww + 3 * (r * w + c), RGBi, 3);  // 0: right
        memcpy(data_.data() + 1 * 3 * ww + 3 * (r * w + c), RGBi, 3);  // 1: left
        memcpy(data_.data() + 4 * 3 * ww + 3 * (r * w + c), RGBi, 3);  // 4: front
        memcpy(data_.data() + 5 * 3 * ww + 3 * (r * w + c), RGBi, 3);  // 5: back

        // set up and down
        interp(data_.data() + 2 * 3 * ww + 3 * (r * w + c), rgb1, rgb2, elup);   // 2: up
        interp(data_.data() + 3 * 3 * ww + 3 * (r * w + c), rgb1, rgb2, -elup);  // 3: down
      }
    }
  }

  // checker
  else if (builtin == mjBUILTIN_CHECKER) {
    checker(data_.data() + 0 * 3 * ww, RGB1, RGB2, w, w);
    checker(data_.data() + 1 * 3 * ww, RGB1, RGB2, w, w);
    checker(data_.data() + 2 * 3 * ww, RGB1, RGB2, w, w);
    checker(data_.data() + 3 * 3 * ww, RGB1, RGB2, w, w);
    checker(data_.data() + 4 * 3 * ww, RGB2, RGB1, w, w);
    checker(data_.data() + 5 * 3 * ww, RGB2, RGB1, w, w);
  }

  // flat
  else if (builtin == mjBUILTIN_FLAT) {
    for (int r = 0; r < w; r++) {
      for (int c = 0; c < w; c++) {
        // set sides and up
        memcpy(data_.data() + 0 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        memcpy(data_.data() + 1 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        memcpy(data_.data() + 2 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        memcpy(data_.data() + 4 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        memcpy(data_.data() + 5 * 3 * ww + 3 * (r * w + c), RGB1, 3);

        // set down
        memcpy(data_.data() + 3 * 3 * ww + 3 * (r * w + c), RGB2, 3);
      }
    }
  }

  //------------------ marks

  // edge
  if (mark == mjMARK_EDGE) {
    for (int j = 0; j < 6; j++) {
      for (int r = 0; r < w; r++) {
        memcpy(data_.data() + j * 3 * ww + 3 * (r * w + 0), RGBm, 3);
        memcpy(data_.data() + j * 3 * ww + 3 * (r * w + w - 1), RGBm, 3);
      }
      for (int c = 0; c < w; c++) {
        memcpy(data_.data() + j * 3 * ww + 3 * (0 * w + c), RGBm, 3);
        memcpy(data_.data() + j * 3 * ww + 3 * ((w - 1) * w + c), RGBm, 3);
      }
    }
  }

  // cross
  else if (mark == mjMARK_CROSS) {
    for (int j = 0; j < 6; j++) {
      for (int r = 0; r < w; r++) {
        memcpy(data_.data() + j * 3 * ww + 3 * (r * w + w / 2), RGBm, 3);
      }
      for (int c = 0; c < w; c++) {
        memcpy(data_.data() + j * 3 * ww + 3 * (w / 2 * w + c), RGBm, 3);
      }
    }
  }

  // random dots
  else if (mark == mjMARK_RANDOM && random > 0) {
    randomdot(data_.data(), markrgb, w, height, random);
  }
}

// load PNG file
void mjCTexture::LoadPNG(const mjCBase*          obj,
                         mjResource*             resource,
                         int                     channels,
                         std::vector<std::byte>& image,
                         unsigned int&           w,
                         unsigned int&           h,
                         bool&                   is_srgb) {
  LodePNGColorType color_type;
  if (channels == 4) {
    color_type = LCT_RGBA;
  } else if (channels == 3) {
    color_type = LCT_RGB;
  } else if (channels == 1) {
    color_type = LCT_GREY;
  } else {
    throw mjCError(obj, "Unsupported number of channels: %s", std::to_string(channels).c_str());
  }
  PNGImage png_image = PNGImage::Load(obj, resource, color_type);
  w                  = png_image.Width();
  h                  = png_image.Height();
  is_srgb            = png_image.IsSRGB();

  // Move data into image.
  image = std::move(png_image).MoveData();
}

// load KTX file
void mjCTexture::LoadKTX(const mjCBase*          obj,
                         mjResource*             resource,
                         int                     channels,
                         std::vector<std::byte>& image,
                         unsigned int&           w,
                         unsigned int&           h,
                         bool&                   is_srgb) {
  if (channels != 1) {
    throw mjCError(obj, "KTX textures must have 1 channel, got %d", nullptr, channels);
  }
  const void* buffer    = 0;
  int         buffer_sz = mju_readResource(resource, &buffer);

  // still not found
  if (buffer_sz < 0) {
    throw mjCError(obj, "could not read texture file '%s'", resource->name);
  } else if (!buffer_sz) {
    throw mjCError(obj, "texture file is empty: '%s'", resource->name);
  }

  w       = buffer_sz;
  h       = 1;
  is_srgb = false;

  image.resize(buffer_sz);
  memcpy(image.data(), buffer, buffer_sz);
}

// decode a texture file as compilation stores it, without an element
void mjCTexture::Decode(const mjCBase*          obj,
                        mjResource*             resource,
                        const std::string&      mime_type,
                        int                     channels,
                        std::vector<std::byte>& image,
                        unsigned int&           w,
                        unsigned int&           h,
                        bool&                   is_srgb) {
  if (mime_type == "image/png") {
    LoadPNG(obj, resource, channels, image, w, h, is_srgb);
  } else if (mime_type == "image/ktx") {
    LoadKTX(obj, resource, channels, image, w, h, is_srgb);
  } else {
    std::string msg  = "unsupported content type: '" + mime_type + "' for file '";
    msg             += std::string(resource->name) + "'";
    throw mjCError(obj, "%s", msg.c_str());
  }
}

// load custom file

void mjCTexture::FlipIfNeeded(std::vector<std::byte>& image, unsigned int w, unsigned int h) {
  // horizontal flip
  if (hflip) {
    for (int r = 0; r < h; r++) {
      for (int c = 0; c < w / 2; c++) {
        int  c1   = w - 1 - c;
        auto val1 = nchannel * (r * w + c);
        auto val2 = nchannel * (r * w + c1);
        for (int ch = 0; ch < nchannel; ch++) {
          auto tmp         = image[val1 + ch];
          image[val1 + ch] = image[val2 + ch];
          image[val2 + ch] = tmp;
        }
      }
    }
  }

  // vertical flip
  if (vflip) {
    for (int r = 0; r < h / 2; r++) {
      for (int c = 0; c < w; c++) {
        int  r1   = h - 1 - r;
        auto val1 = nchannel * (r * w + c);
        auto val2 = nchannel * (r1 * w + c);
        for (int ch = 0; ch < nchannel; ch++) {
          auto tmp         = image[val1 + ch];
          image[val1 + ch] = image[val2 + ch];
          image[val2 + ch] = tmp;
        }
      }
    }
  }
}

std::string mjCTexture::GetCacheId(const mjResource* resource, const std::string& asset_type) {
  std::stringstream ss;
  ss << resource->name << ";ARGS:content_type=" << asset_type << ",nchannel=" << nchannel
     << ",hflip=" << hflip << ",vflip=" << vflip << ";";
  return ss.str();
}

// load from PNG or custom file, flip if specified
void mjCTexture::LoadFlip(std::string             filename,
                          const mjVFS*            vfs,
                          std::vector<std::byte>& image,
                          unsigned int&           w,
                          unsigned int&           h,
                          bool&                   is_srgb) {
  mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);

  struct CachedImage {
    unsigned int           w, h, n_ch;
    bool                   is_srgb;
    std::vector<std::byte> image;
  };

  // cache callback
  auto callback = [&](const void* data) {
    const CachedImage* cached_image = static_cast<const CachedImage*>(data);

    w       = cached_image->w;
    h       = cached_image->h;
    is_srgb = cached_image->is_srgb;
    image   = cached_image->image;
    return true;
  };

  std::string asset_type = GetAssetContentType(filename, content_type_);

  if (asset_type != "image/png" && asset_type != "image/ktx") {
    throw mjCError(this, "unsupported content type: '%s'", asset_type.c_str());
  }

  // a KTX is stored as a single-channel blob, whatever was authored
  if (asset_type == "image/ktx") { nchannel = 1; }

  // try loading from cache
  mjSpec*     owning_spec = model->FindSpec(compiler);
  mjResource* resource    = LoadResource(owning_spec->modelfiledir->c_str(), filename, vfs);
  if (cache && cache->PopulateData(GetCacheId(resource, asset_type), resource, callback)) {
    mju_closeResource(resource);
    return;
  }

  try {
    if (asset_type == "image/ktx" && (hflip || vflip)) {
      throw mjCError(this, "cannot flip KTX textures");
    }
    Decode(this, resource, asset_type, nchannel, image, w, h, is_srgb);
  } catch (mjCError err) {
    mju_closeResource(resource);
    throw err;
  }

  FlipIfNeeded(image, w, h);
  if (cache) {
    CachedImage* cached_texture = new CachedImage;

    cached_texture->w                 = w;
    cached_texture->h                 = h;
    cached_texture->is_srgb           = is_srgb;
    cached_texture->image             = image;
    std::size_t                  size = sizeof(CachedImage) + image.size();
    std::shared_ptr<CachedImage> cached_data{cached_texture};
    cache->Insert("", GetCacheId(resource, asset_type), resource, cached_data, size);
  }
  mju_closeResource(resource);
}

// load 2D
void mjCTexture::Load2D(std::string filename, const mjVFS* vfs) {
  // load PNG or custom
  unsigned int w, h;
  bool         is_srgb;

  LoadFlip(filename, vfs, data_, w, h, is_srgb);

  // assign size
  width  = w;
  height = h;
  if (colorspace == mjCOLORSPACE_AUTO) {
    colorspace = is_srgb ? mjCOLORSPACE_SRGB : mjCOLORSPACE_LINEAR;
  }
}

// load cube or skybox from single file (repeated or grid)
void mjCTexture::LoadCubeSingle(std::string filename, const mjVFS* vfs) {
  // check gridsize
  if (gridsize[0] < 1 || gridsize[1] < 1 || gridsize[0] * gridsize[1] > 12) {
    throw mjCError(this, "gridsize must be non-zero and no more than 12 squares in texture");
  }

  // load PNG or custom
  unsigned int           w, h;
  bool                   is_srgb;
  std::vector<std::byte> image;
  LoadFlip(filename, vfs, image, w, h, is_srgb);

  // faces are copied with 3 channels
  if (nchannel != 3) {
    throw mjCError(this, "cube and skybox textures loaded from files must have 3 channels");
  }

  if (colorspace == mjCOLORSPACE_AUTO) {
    colorspace = is_srgb ? mjCOLORSPACE_SRGB : mjCOLORSPACE_LINEAR;
  }

  // check gridsize for compatibility
  if (w / gridsize[1] != h / gridsize[0] || (w % gridsize[1]) || (h % gridsize[0])) {
    throw mjCError(this,
                   "PNG size must be integer multiple of gridsize in texture '%s' (id %d)",
                   (const char*)file_.c_str(),
                   id);
  }

  // assign size: repeated or full
  if (gridsize[0] == 1 && gridsize[1] == 1) {
    width = height = w;
  } else {
    width = w / gridsize[1];
    if (width >= std::numeric_limits<int>::max() / 6) {
      throw mjCError(this, "Invalid width of cube texture");
    }
    height = 6 * width;
  }

  // allocate data
  std::int64_t size = static_cast<std::int64_t>(width) * height;
  if (size >= std::numeric_limits<std::int64_t>::max() / 3 || size <= 0) {
    throw mjCError(this, "Cube texture too large");
  }
  try {
    data_.assign(3 * size, std::byte(0));
  } catch (const std::bad_alloc& e) {
    throw mjCError(this,
                   "Could not allocate memory for texture '%s' (id %d)",
                   (const char*)file_.c_str(),
                   id);
  }

  // copy: repeated
  if (gridsize[0] == 1 && gridsize[1] == 1) {
    memcpy(data_.data(), image.data(), 3 * width * width);
  }

  // copy: grid
  else {
    // keep track of which faces were defined
    int loaded[6] = {0, 0, 0, 0, 0, 0};

    // process grid
    for (int k = 0; k < gridsize[0] * gridsize[1]; k++) {
      // decode face symbol
      int i = -1;
      if (gridlayout[k] == 'R') {
        i = 0;
      } else if (gridlayout[k] == 'L') {
        i = 1;
      } else if (gridlayout[k] == 'U') {
        i = 2;
      } else if (gridlayout[k] == 'D') {
        i = 3;
      } else if (gridlayout[k] == 'F') {
        i = 4;
      } else if (gridlayout[k] == 'B') {
        i = 5;
      } else if (gridlayout[k] != '.')
        throw mjCError(this, "gridlayout symbol is not among '.RLUDFB' in texture");

      // load if specified
      if (i >= 0) {
        // extract sub-image
        int rstart = width * (k / gridsize[1]);
        int cstart = width * (k % gridsize[1]);
        for (int j = 0; j < width; j++) {
          memcpy(data_.data() + i * 3 * width * width + j * 3 * width,
                 image.data() + (j + rstart) * 3 * w + 3 * cstart,
                 3 * width);
        }

        // mark as defined
        loaded[i] = 1;
      }
    }

    // set undefined faces to rgb1
    for (int i = 0; i < 6; i++) {
      if (!loaded[i]) {
        for (int k = 0; k < width; k++) {
          for (int s = 0; s < width; s++) {
            for (int j = 0; j < 3; j++) {
              data_[i * 3 * width * width + 3 * (k * width + s) + j] = (std::byte)(255 * rgb1[j]);
            }
          }
        }
      }
    }
  }

  image.clear();
}


// load cube or skybox from separate file
void mjCTexture::LoadCubeSeparate(const mjVFS* vfs) {
  // keep track of which faces were defined
  int loaded[6] = {0, 0, 0, 0, 0, 0};

  // process nonempty files
  for (int i = 0; i < 6; i++) {
    if (!cubefiles_[i].empty()) {
      // remove path from file if necessary
      if (model->strippath) { cubefiles_[i] = mjuu_strippath(cubefiles_[i]); }

      // make filename
      mujoco::user::FilePath texturedir_;
      texturedir_       = FilePath(mjs_getString(compiler->texturedir));
      FilePath filename = texturedir_ + FilePath(cubefiles_[i]);

      // load PNG or custom
      unsigned int w, h;
      bool         is_srgb;

      std::vector<std::byte> image;
      LoadFlip(filename.Str(), vfs, image, w, h, is_srgb);

      // faces are copied with 3 channels
      if (nchannel != 3) {
        throw mjCError(this, "cube and skybox textures loaded from files must have 3 channels");
      }

      // assume all faces have the same colorspace
      if (colorspace == mjCOLORSPACE_AUTO) {
        colorspace = is_srgb ? mjCOLORSPACE_SRGB : mjCOLORSPACE_LINEAR;
      }

      // PNG must be square
      if (w != h) {
        throw mjCError(this,
                       "Non-square PNG file '%s' in cube or skybox id %d",
                       (const char*)cubefiles_[i].c_str(),
                       id);
      }

      // first file: set size and allocate data
      if (data_.empty()) {
        width = w;
        if (width >= std::numeric_limits<int>::max() / 6) {
          throw mjCError(this, "Invalid width of builtin texture");
        }
        height            = 6 * width;
        std::int64_t size = static_cast<std::int64_t>(width) * height;
        if (size >= std::numeric_limits<mjtSize>::max() / 3 || size <= 0) {
          throw mjCError(this, "PNG texture too large");
        }
        try {
          data_.assign(3 * size, std::byte(0));
        } catch (const std::bad_alloc& e) {
          throw mjCError(this, "Could not allocate memory for texture");
        }
      }

      // otherwise check size
      else if (width != w) {
        throw mjCError(this,
                       "PNG file '%s' has incompatible size in texture id %d",
                       (const char*)cubefiles_[i].c_str(),
                       id);
      }

      // copy data
      memcpy(data_.data() + i * 3 * width * width, image.data(), 3 * width * width);
      image.clear();

      // mark as defined
      loaded[i] = 1;
    }
  }

  // set undefined faces to rgb1
  for (int i = 0; i < 6; i++) {
    if (!loaded[i]) {
      for (int k = 0; k < width; k++) {
        for (int s = 0; s < width; s++) {
          for (int j = 0; j < 3; j++) {
            data_[i * 3 * width * width + 3 * (k * width + s) + j] = (std::byte)(255 * rgb1[j]);
          }
        }
      }
    }
  }
}


// why this texture cannot be by reference, nullptr if it can: a single file used as it is, which
// a consumer reads for itself (a VFS buffer has no lifetime beyond compilation, so it is embedded)
const char* mjCTexture::Ineligible(const mjVFS* vfs) const {
  if (!data_.empty()) { return "it is given as a buffer"; }
  if (builtin != mjBUILTIN_NONE) { return "it is builtin"; }
  if (file_.empty()) { return "it is given as cube files"; }
  if (hflip || vflip) { return "it is flipped"; }
  if (gridsize[0] != 1 || gridsize[1] != 1) { return "it is a grid"; }
  if (vfs) {
    mjSpec* owning_spec = model->FindSpec(compiler);
    if (!owning_spec) { owning_spec = &model->spec; }
    std::string        file     = model->strippath ? mjuu_strippath(file_) : file_;
    FilePath           filename = FilePath(mjs_getString(compiler->texturedir)) + FilePath(file);
    mujoco::user::VFS* impl     = mujoco::user::VFS::Upcast(const_cast<mjVFS*>(vfs));
    if (impl->InMemory(owning_spec->modelfiledir->c_str(), filename.c_str())) {
      return "its file is served from memory by the VFS";
    }
  }
  return nullptr;
}


// compiler
void mjCTexture::Compile(const mjVFS* vfs) {
  CopyFromSpec();

  mujoco::user::FilePath texturedir_;
  texturedir_ = FilePath(mjs_getString(compiler->texturedir));

  // by reference: the policy of the model being compiled, unless this texture says otherwise
  byref_      = false;
  bool policy = model->spec.compiler.passthrough & mjPASSTHROUGH_TEXTURES;
  if ((policy && embed == mjEMBED_AUTO) || embed == mjEMBED_FALSE) {
    if (const char* reason = Ineligible(vfs)) {
      if (embed == mjEMBED_FALSE) {
        throw mjCError(this, "texture cannot be by reference: %s", reason);
      }
    } else {
      // nothing is read: the model gets the resolved path and no payload
      if (model->strippath) { file_ = mjuu_strippath(file_); }
      if (GetAssetContentType(file_, content_type_) == "image/ktx") { nchannel = 1; }
      width = height = 0;
      byref_         = true;
      return;
    }
  }

  // buffer from user
  if (!data_.empty()) {
    if (data_.size() != nchannel * width * height) {
      throw mjCError(this,
                     "Texture buffer has incorrect size, given %d expected %d",
                     nullptr,
                     data_.size(),
                     nchannel * width * height);
    }

    // Flip if specified.
    FlipIfNeeded(data_, width, height);
    return;
  }

  // builtin
  else if (builtin != mjBUILTIN_NONE) {
    // builtin textures are generated with 3 channels
    if (nchannel != 3) { throw mjCError(this, "builtin textures must have 3 channels"); }

    // check width
    if (width < 1) { throw mjCError(this, "Invalid width of builtin texture"); }

    // adjust height of cube texture
    if (type != mjTEXTURE_2D) {
      if (width >= std::numeric_limits<int>::max() / 6) {
        throw mjCError(this, "Invalid width of builtin texture");
      }
      height = 6 * width;
    } else {
      if (height < 1) { throw mjCError(this, "Invalid height of builtin texture"); }
    }

    std::int64_t size = static_cast<std::int64_t>(width) * height;
    if (size >= std::numeric_limits<int64_t>::max() / nchannel || size <= 0) {
      throw mjCError(this, "Builtin texture too large");
    }
    // allocate data
    try {
      data_.assign(nchannel * size, std::byte(0));
    } catch (const std::bad_alloc& e) {
      throw mjCError(this, "Could not allocate memory for texture");
    }

    // dispatch
    if (type == mjTEXTURE_2D) {
      Builtin2D();
    } else {
      BuiltinCube();
    }
  }

  // single file
  else if (!file_.empty()) {
    // remove path from file if necessary
    if (model->strippath) { file_ = mjuu_strippath(file_); }

    // make filename
    FilePath filename = texturedir_ + FilePath(file_);

    // dispatch
    if (type == mjTEXTURE_2D) {
      Load2D(filename.Str(), vfs);
    } else {
      LoadCubeSingle(filename.Str(), vfs);
    }
  }

  // separate files
  else {
    // 2D not allowed
    if (type == mjTEXTURE_2D) {
      throw mjCError(this, "Cannot load 2D texture from separate files, texture");
    }

    // at least one cubefile must be defined
    bool defined = false;
    for (int i = 0; i < 6; i++) {
      if (!cubefiles_[i].empty()) {
        defined = true;
        break;
      }
    }
    if (!defined) { throw mjCError(this, "No cubefiles_ defined in cube or skybox texture"); }

    // only cube and skybox
    LoadCubeSeparate(vfs);
  }

  // make sure someone allocated data; SHOULD NOT OCCUR
  if (data_.empty()) {
    throw mjCError(this, "texture '%s' (id %d) was not specified", name.c_str(), id);
  }
}


//------------------ class mjCMaterial implementation ----------------------------------------------

// initialize defaults
mjCMaterial::mjCMaterial(mjCModel* _model, mjCDef* _def) {
  mjs_defaultMaterial(&spec);
  elemtype = mjOBJ_MATERIAL;
  textures_.assign(mjNTEXROLE, "");
  spec_textures_.assign(mjNTEXROLE, "");

  // clear internal
  for (int i = 0; i < mjNTEXROLE; i++) { texid[i] = -1; }

  // reset to default if given
  if (_def) { *this = _def->Material(); }

  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  PointToLocal();

  // in case this material is not compiled
  CopyFromSpec();
}


mjCMaterial::mjCMaterial(const mjCMaterial& other) {
  *this = other;
}


mjCMaterial& mjCMaterial::operator=(const mjCMaterial& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCMaterial_*>(this) = static_cast<const mjCMaterial_&>(other);
    *static_cast<mjsMaterial*>(this)  = static_cast<const mjsMaterial&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCMaterial::PointToLocal() {
  spec.element  = static_cast<mjsElement*>(this);
  spec.textures = &spec_textures_;
  spec.info     = &info;
  textures      = nullptr;
}


void mjCMaterial::CopyFromSpec() {
  *static_cast<mjsMaterial*>(this) = spec;

  textures_ = spec_textures_;
}


void mjCMaterial::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  for (int i = 0; i < mjNTEXROLE; i++) {
    if (!spec_textures_[i].empty()) {
      spec_textures_[i] = m->prefix + spec_textures_[i] + m->suffix;
    }
  }
}


// compiler
void mjCMaterial::Compile(void) {
  CopyFromSpec();
}


//------------------ class mjCPair implementation --------------------------------------------------

// constructor
mjCPair::mjCPair(mjCModel* _model, mjCDef* _def) {
  mjs_defaultPair(&spec);
  elemtype = mjOBJ_PAIR;

  // set defaults
  spec_geomname1_.clear();
  spec_geomname2_.clear();

  // clear internal variables
  geom1     = nullptr;
  geom2     = nullptr;
  signature = -1;

  // reset to default if given
  if (_def) { *this = _def->Pair(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // point to local
  PointToLocal();

  // in case this camera is not compiled
  CopyFromSpec();
}


mjCPair::mjCPair(const mjCPair& other) {
  *this = other;
}


mjCPair& mjCPair::operator=(const mjCPair& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCPair_*>(this) = static_cast<const mjCPair_&>(other);
    *static_cast<mjsPair*>(this)  = static_cast<const mjsPair&>(other);

    this->geom1 = nullptr;
    this->geom2 = nullptr;
  }
  PointToLocal();
  return *this;
}


void mjCPair::PointToLocal() {
  spec.element   = static_cast<mjsElement*>(this);
  spec.geomname1 = &spec_geomname1_;
  spec.geomname2 = &spec_geomname2_;
  geomname1      = nullptr;
  geomname2      = nullptr;
  spec.info      = &info;
}


void mjCPair::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  prefix = m->prefix;
  suffix = m->suffix;
}


void mjCPair::CopyFromSpec() {
  *static_cast<mjsPair*>(this) = spec;

  geomname1_ = spec_geomname1_;
  geomname2_ = spec_geomname2_;
}


void mjCPair::ResolveReferences(const mjCModel* m) {
  geomname1_ = prefix + geomname1_ + suffix;
  geomname2_ = prefix + geomname2_ + suffix;
  geom1      = (mjCGeom*)m->FindObject(mjOBJ_GEOM, geomname1_);
  geom2      = (mjCGeom*)m->FindObject(mjOBJ_GEOM, geomname2_);

  if (!geom1 && geom2) {
    geomname1_ = spec_geomname1_;
    geom1      = (mjCGeom*)m->FindObject(mjOBJ_GEOM, geomname1_);
  }
  if (geom1 && !geom2) {
    geomname2_ = spec_geomname2_;
    geom2      = (mjCGeom*)m->FindObject(mjOBJ_GEOM, geomname2_);
  }

  if (!geom1) {
    throw mjCError(this, "geom '%s' not found in collision %d", geomname1_.c_str(), id);
  }
  if (!geom2) {
    throw mjCError(this, "geom '%s' not found in collision %d", geomname2_.c_str(), id);
  }

  // the names which a namespace gives are those of the spec from now on; without one they are
  // left as written: those of a compiled pair are in the order of the bodies
  if (!prefix.empty() || !suffix.empty()) {
    spec_geomname1_ = geomname1_;
    spec_geomname2_ = geomname2_;
  }
  prefix.clear();
  suffix.clear();

  // swap if body1 > body2
  if (geom1->body->id > geom2->body->id) {
    std::string nametmp = geomname1_;

    geomname1_ = geomname2_;
    geomname2_ = nametmp;

    mjCGeom* geomtmp = geom1;

    geom1 = geom2;
    geom2 = geomtmp;
  }

  // get geom ids and body signature
  signature = ((unsigned int)(geom1->body->id) << 16) + geom2->body->id;
}


// scale the compiled copy by the mean gauge of the two geoms, or back; undefined values are
// inherited from the geoms, which are scaled already
void mjCPair::ApplyGauge(bool inverse) {
  const mjCGauge mean = mjCGauge::Mean(geom1->Gauge(), geom2->Gauge());
  const mjCGauge g    = inverse ? mean.Inverse() : mean;
  if (g.IsIdentity()) { return; }
  const double fl = g.Shape(1);
  if (mjuu_defined(margin)) { margin *= fl; }
  if (mjuu_defined(gap)) { gap *= fl; }
  if (mjuu_defined(solimp[0])) { solimp[2] *= fl; }
  if (mjuu_defined(friction[0])) {
    for (int i = 2; i < 5; i++) { friction[i] *= fl; }
  }
  adhesion *= g.Force(1, 1);
}


// compiler
void mjCPair::Compile(void) {
  CopyFromSpec();

  // check condim
  if (condim != 1 && condim != 3 && condim != 4 && condim != 6) {
    throw mjCError(this, "invalid condim in contact pair");
  }

  // find geoms
  ResolveReferences(model);
  ApplyGauge();

  // set undefined margin: max
  if (!mjuu_defined(margin)) { margin = std::max(geom1->margin, geom2->margin); }

  // set undefined gap: max
  if (!mjuu_defined(gap)) { gap = std::max(geom1->gap, geom2->gap); }

  // set undefined condim, friction, solref, solimp: different priority
  if (geom1->priority != geom2->priority) {
    mjCGeom* pgh = (geom1->priority > geom2->priority ? geom1 : geom2);

    // condim
    if (condim < 0) { condim = pgh->condim; }

    // friction
    if (!mjuu_defined(friction[0])) {
      friction[0] = friction[1] = pgh->friction[0];
      friction[2]               = pgh->friction[1];
      friction[3] = friction[4] = pgh->friction[2];
    }

    // reference
    if (!mjuu_defined(solref[0])) {
      for (int i = 0; i < mjNREF; i++) { solref[i] = pgh->solref[i]; }
    }

    // impedance
    if (!mjuu_defined(solimp[0])) {
      for (int i = 0; i < mjNIMP; i++) { solimp[i] = pgh->solimp[i]; }
    }
  }

  // set undefined condim, friction, solref, solimp: same priority
  else {
    // condim: max
    if (condim < 0) { condim = std::max(geom1->condim, geom2->condim); }

    // friction: max
    if (!mjuu_defined(friction[0])) {
      friction[0] = friction[1] = std::max(geom1->friction[0], geom2->friction[0]);
      friction[2]               = std::max(geom1->friction[1], geom2->friction[1]);
      friction[3] = friction[4] = std::max(geom1->friction[2], geom2->friction[2]);
    }

    // solver mix factor
    double mix;
    if (geom1->solmix >= mjEPS && geom2->solmix >= mjEPS) {
      mix = geom1->solmix / (geom1->solmix + geom2->solmix);
    } else if (geom1->solmix < mjEPS && geom2->solmix < mjEPS) {
      mix = 0.5;
    } else if (geom1->solmix < mjEPS) {
      mix = 0.0;
    } else {
      mix = 1.0;
    }

    // reference
    if (!mjuu_defined(solref[0])) {
      // standard: mix
      if (geom1->solref[0] > 0 && geom2->solref[0] > 0) {
        for (int i = 0; i < mjNREF; i++) {
          solref[i] = mix * geom1->solref[i] + (1 - mix) * geom2->solref[i];
        }
      }

      // direct: min
      else {
        for (int i = 0; i < mjNREF; i++) {
          solref[i] = std::min(geom1->solref[i], geom2->solref[i]);
        }
      }
    }

    // impedance
    if (!mjuu_defined(solimp[0])) {
      for (int i = 0; i < mjNIMP; i++) {
        solimp[i] = mix * geom1->solimp[i] + (1 - mix) * geom2->solimp[i];
      }
    }
  }
}


//------------------ class mjCBodyPair implementation ----------------------------------------------

// constructor
mjCBodyPair::mjCBodyPair(mjCModel* _model) {
  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  elemtype = mjOBJ_EXCLUDE;

  // set defaults
  spec_bodyname1_.clear();
  spec_bodyname2_.clear();

  // clear internal variables
  body1 = body2 = signature = -1;

  PointToLocal();
  CopyFromSpec();
}


mjCBodyPair::mjCBodyPair(const mjCBodyPair& other) {
  *this = other;
}


mjCBodyPair& mjCBodyPair::operator=(const mjCBodyPair& other) {
  if (this != &other) {
    this->spec                        = other.spec;
    *static_cast<mjCBodyPair_*>(this) = static_cast<const mjCBodyPair_&>(other);
    *static_cast<mjsExclude*>(this)   = static_cast<const mjsExclude&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCBodyPair::PointToLocal() {
  spec.element   = static_cast<mjsElement*>(this);
  spec.bodyname1 = &spec_bodyname1_;
  spec.bodyname2 = &spec_bodyname2_;
  spec.info      = &info;
  bodyname1      = nullptr;
  bodyname2      = nullptr;
}


void mjCBodyPair::NameSpace(const mjCModel* m) {
  if (!name.empty()) { name = m->prefix + name + m->suffix; }
  prefix = m->prefix;
  suffix = m->suffix;
}


void mjCBodyPair::CopyFromSpec() {
  *static_cast<mjsExclude*>(this) = spec;

  bodyname1_ = spec_bodyname1_;
  bodyname2_ = spec_bodyname2_;
}


void mjCBodyPair::ResolveReferences(const mjCModel* m) {
  bodyname1_   = prefix + bodyname1_ + suffix;
  bodyname2_   = prefix + bodyname2_ + suffix;
  mjCBody* pb1 = (mjCBody*)m->FindObject(mjOBJ_BODY, bodyname1_);
  mjCBody* pb2 = (mjCBody*)m->FindObject(mjOBJ_BODY, bodyname2_);

  if (!pb1 && pb2) {
    bodyname1_ = spec_bodyname1_;
    pb1        = (mjCBody*)m->FindObject(mjOBJ_BODY, bodyname1_);
  }
  if (pb1 && !pb2) {
    bodyname2_ = spec_bodyname2_;
    pb2        = (mjCBody*)m->FindObject(mjOBJ_BODY, bodyname2_);
  }

  if (!pb1) { throw mjCError(this, "body '%s' not found in bodypair %d", bodyname1_.c_str(), id); }
  if (!pb2) { throw mjCError(this, "body '%s' not found in bodypair %d", bodyname2_.c_str(), id); }

  // the names which a namespace gives are those of the spec from now on; without one they are
  // left as written: those of a compiled exclude are in the order of the bodies
  if (!prefix.empty() || !suffix.empty()) {
    spec_bodyname1_ = bodyname1_;
    spec_bodyname2_ = bodyname2_;
  }
  prefix.clear();
  suffix.clear();

  // swap if body1 > body2
  if (pb1->id > pb2->id) {
    std::string nametmp = bodyname1_;

    bodyname1_ = bodyname2_;
    bodyname2_ = nametmp;

    mjCBody* bodytmp = pb1;

    pb1 = pb2;
    pb2 = bodytmp;
  }

  // get body ids and body signature
  body1     = pb1->id;
  body2     = pb2->id;
  signature = ((unsigned int)body1 << 16) + body2;
}


// compiler
void mjCBodyPair::Compile(void) {
  CopyFromSpec();

  // find bodies
  ResolveReferences(model);
}


//------------------ class mjCEquality implementation ----------------------------------------------

// initialize default constraint
mjCEquality::mjCEquality(mjCModel* _model, mjCDef* _def) {
  mjs_defaultEquality(&spec);
  elemtype = mjOBJ_EQUALITY;

  // clear internal variables
  spec_name1_.clear();
  spec_name2_.clear();
  eqadr_ = -1;
  obj1id = obj2id = -1;

  // reset to default if given
  if (_def) { *this = _def->Equality(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // point to local
  PointToLocal();

  // in case this camera is not compiled
  CopyFromSpec();
}


mjCEquality::mjCEquality(const mjCEquality& other) {
  *this = other;
}


mjCEquality& mjCEquality::operator=(const mjCEquality& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCEquality_*>(this) = static_cast<const mjCEquality_&>(other);
    *static_cast<mjsEquality*>(this)  = static_cast<const mjsEquality&>(other);

    eqadr_ = -1;
  }
  PointToLocal();
  return *this;
}


void mjCEquality::ResetId() {
  id     = -1;
  eqadr_ = -1;
}


void mjCEquality::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.name1   = &spec_name1_;
  spec.name2   = &spec_name2_;
  spec.info    = &info;
  name1        = nullptr;
  name2        = nullptr;
}


void mjCEquality::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);

  // the world body keeps its name
  bool body = spec.objtype == mjOBJ_BODY;
  if (!spec_name1_.empty() && !(body && spec_name1_ == "world")) {
    spec_name1_ = m->prefix + spec_name1_ + m->suffix;
  }
  if (!spec_name2_.empty() && !(body && spec_name2_ == "world")) {
    spec_name2_ = m->prefix + spec_name2_ + m->suffix;
  }
}


void mjCEquality::CopyFromSpec() {
  *static_cast<mjsEquality*>(this) = spec;

  name1_ = spec_name1_;
  name2_ = spec_name2_;
}


void mjCEquality::ResolveReferences(const mjCModel* m) {
  mjtObj   object_type;
  mjCBase *px1, *px2;
  mjtJoint jt1, jt2;

  // determine object type
  if (type == mjEQ_WELD) {
    if (objtype != mjOBJ_SITE && objtype != mjOBJ_BODY) {
      throw mjCError(this, "weld constraint supports only sites and bodies");
    }
    object_type = objtype;
  } else if (type == mjEQ_CONNECT) {
    if (objtype != mjOBJ_SITE && objtype != mjOBJ_BODY) {
      throw mjCError(this, "connect constraint supports only sites and bodies");
    }
    object_type = objtype;
  } else if (type == mjEQ_JOINT) {
    object_type = mjOBJ_JOINT;
  } else if (type == mjEQ_TENDON) {
    object_type = mjOBJ_TENDON;
  } else if (type == mjEQ_FLEX || type == mjEQ_FLEXSTRAIN) {
    object_type = mjOBJ_FLEX;
  } else {
    throw mjCError(this, "invalid type in equality constraint");
  }

  // find object 1, get id
  px1 = m->FindObject(object_type, name1_);
  if (!px1) { throw mjCError(this, "unknown element '%s' in equality constraint", name1_.c_str()); }
  obj1id = px1->id;

  // find object 2, get id
  if (!name2_.empty()) {
    px2 = m->FindObject(object_type, name2_);
    if (!px2) {
      throw mjCError(this, "unknown element '%s' in equality constraint %d", name2_.c_str(), id);
    }
    obj2id = px2->id;
  } else {
    // object 2 unspecified: set to -1
    obj2id = -1;
    px2    = nullptr;
  }

  // set missing body = world
  if (object_type == mjOBJ_BODY && obj2id == -1) { obj2id = 0; }

  // make sure the two objects are different
  if (obj1id == obj2id) {
    throw mjCError(this, "element '%s' is repeated in equality constraint %d", name1_.c_str(), id);
  }

  // make sure joints are scalar
  if (type == mjEQ_JOINT) {
    jt1 = ((mjCJoint*)px1)->type;
    jt2 = (px2 ? ((mjCJoint*)px2)->type : mjJNT_HINGE);
    if ((jt1 != mjJNT_HINGE && jt1 != mjJNT_SLIDE) || (jt2 != mjJNT_HINGE && jt2 != mjJNT_SLIDE)) {
      throw mjCError(this, "only HINGE and SLIDE joint allowed in constraint");
    }
  }
}


// scale the compiled copy by the gauges of the constrained objects, or back: an anchor by that of
// the body in which it is given, a coupling by those of its coordinates, and the impedance width by
// the dimension of the residual
void mjCEquality::ApplyGauge(bool inverse) {
  double width = 1;
  if (type == mjEQ_CONNECT || type == mjEQ_WELD) {
    const bool     body = objtype == mjOBJ_BODY;
    const mjCBase* o1   = model->FindObject(objtype, name1_);
    const mjCBase* o2 =
        name2_.empty() ? (body ? model->Bodies()[0] : nullptr) : model->FindObject(objtype, name2_);
    mjCGauge g1 = body ? o1->ContentGauge() : o1->Gauge();
    mjCGauge g2 = o2 ? (body ? o2->ContentGauge() : o2->Gauge()) : mjCGauge();
    if (g1.IsIdentity() && g2.IsIdentity()) { return; }
    if (inverse) {
      g1 = g1.Inverse();
      g2 = g2.Inverse();
    }
    const mjCGauge mean = mjCGauge::Mean(g1, g2);
    width               = mean.Shape(1);
    if (type == mjEQ_CONNECT) {
      if (body) {
        g1.Point(data);
        width = g1.Shape(1);
      }
    } else {
      if (body) {
        g2.Point(data);
        g1.Point(data + 3);
      }
      data[10] *= mean.Shape(1);
    }
  } else if (type == mjEQ_JOINT || type == mjEQ_TENDON) {
    // factor of the coordinate of a joint or tendon
    auto coordinate = [&](const std::string& name) -> double {
      if (name.empty()) { return 1; }
      if (type == mjEQ_JOINT) {
        const mjCJoint* joint = static_cast<const mjCJoint*>(model->FindObject(mjOBJ_JOINT, name));
        return joint->spec.type == mjJNT_SLIDE ? joint->Gauge().Stretch(joint->spec.axis) : 1;
      }
      const mjCTendon* tendon =
          static_cast<const mjCTendon*>(model->FindObject(mjOBJ_TENDON, name));
      return tendon->Linear() ? tendon->PathGauge().Shape(1) : 1;
    };
    double q1 = coordinate(name1_);
    double q2 = coordinate(name2_);
    if (q1 == 1 && q2 == 1) { return; }
    if (inverse) {
      q1 = 1 / q1;
      q2 = 1 / q2;
    }
    for (int k = 0; k < 5; k++) { data[k] *= q1 / std::pow(q2, k); }
    width = q1;
  } else if (type == mjEQ_FLEX || type == mjEQ_FLEXSTRAIN) {
    const mjCFlex* flex = static_cast<const mjCFlex*>(model->FindObject(mjOBJ_FLEX, name1_));
    width               = flex->BodyGauge().Shape(1);
    if (inverse) { width = 1 / width; }
  }
  solimp[2] *= width;
}


// compiler
void mjCEquality::Compile(void) {
  CopyFromSpec();

  // find objects
  ResolveReferences(model);
  ApplyGauge();

  // a connect between sites uses no data
  if (type == mjEQ_CONNECT && objtype == mjOBJ_SITE) { std::fill(data, data + mjNEQDATA, 0); }

  // a weld between bodies without a relative pose, given as a zero quaternion, is marked in
  // data[11] for mj_setConst to compute the pose; a given one is normalized
  if (type == mjEQ_WELD) {
    const bool body = objtype == mjOBJ_BODY;
    data[11]        = body && !data[6] && !data[7] && !data[8] && !data[9];
    if (body && !data[11] && !mjuu_normvec(data + 6, 4)) {
      data[6] = 1;
      data[7] = data[8] = data[9] = 0;
    }
  }

  // make sure flex is not rigid, and has no conflicting stretch forces
  if (type == mjEQ_FLEX || type == mjEQ_FLEXSTRAIN) {
    mjCFlex* flex = model->Flexes()[obj1id];
    if (flex->rigid) {
      throw mjCError(this, "rigid flex '%s' in equality constraint %d", name1_.c_str(), id);
    }
    // bending-only elasticity (2D bend, or 1D rod bending) is compatible with constraints
    if (flex->dim > 1 && flex->elastic2d != 1 && flex->young > 0) {
      throw mjCError(this, "flex constraints and elasticity (young) cannot both be present");
    }
    if (flex->edgestiffness > 0) {
      throw mjCError(this, "flex constraints and edge stiffness cannot both be present");
    }
    if (type == mjEQ_FLEXSTRAIN) {
      if (!flex->interpolated) {
        throw mjCError(this,
                       "flex strain equality requires an interpolated flex '%s'",
                       name1_.c_str());
      }
      int cx = flex->spec.cellcount[0];
      int cy = flex->spec.cellcount[1];
      int cz = flex->spec.cellcount[2];
      int d0 = static_cast<int>(data[0]);
      int d1 = static_cast<int>(data[1]);
      int d2 = static_cast<int>(data[2]);
      if (flex->elastic2d) {
        if (d1 != -1 || d2 != -1) {
          throw mjCError(this,
                         "volumetric flex strain equality on shell flex '%s' "
                         "(elastic2d != 0)",
                         name1_.c_str());
        }
        int nfaces = 2 * (cy * cz + cx * cz + cx * cy);
        if (d0 < 0 || d0 >= nfaces) {
          throw mjCError(this,
                         "flex strain face index out of range in equality constraint %d",
                         nullptr,
                         id);
        }
      } else {
        if (d1 == -1 || d2 == -1) {
          throw mjCError(this,
                         "shell flex strain equality on volumetric flex '%s' "
                         "(elastic2d == 0)",
                         name1_.c_str());
        }
        if (d0 < 0 || d0 >= cx || d1 < 0 || d1 >= cy || d2 < 0 || d2 >= cz) {
          throw mjCError(this,
                         "flex strain cell index out of range in equality constraint %d",
                         nullptr,
                         id);
        }
      }
    }
  }
}


//------------------ class mjCTendon implementation ------------------------------------------------

// constructor
mjCTendon::mjCTendon(mjCModel* _model, mjCDef* _def) {
  mjs_defaultTendon(&spec);
  elemtype = mjOBJ_TENDON;

  // clear internal variables
  spec_material_.clear();
  spec_userdata_.clear();
  path.clear();
  matid       = -1;
  plasticadr_ = -1;

  // reset to default if given
  if (_def) { *this = _def->Tendon(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // point to local
  PointToLocal();

  // in case this camera is not compiled
  CopyFromSpec();
}


mjCTendon::mjCTendon(const mjCTendon& other) {
  *this = other;
}


mjCTendon& mjCTendon::operator=(const mjCTendon& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCTendon_*>(this) = static_cast<const mjCTendon_&>(other);
    *static_cast<mjsTendon*>(this)  = static_cast<const mjsTendon&>(other);
    plasticadr_                     = -1;
    for (int i = 0; i < other.path.size(); i++) {
      path.push_back(new mjCWrap(*other.path[i]));
      path.back()->tendon = this;
    }
  }
  PointToLocal();
  return *this;
}


void mjCTendon::ResetId() {
  id          = -1;
  plasticadr_ = -1;
}


mjtNum* mjCTendon::plastic(const std::string& state_name) {
  if (plastic_.find(state_name) == plastic_.end()) { plastic_[state_name] = {mjNAN, 0}; }
  return plastic_.at(state_name).data();
}


void mjCTendon::ForgetKeyframes(const std::vector<std::string>& names, bool keep) {
  forgetvalues(plastic_, names, keep);
}


bool mjCTendon::is_limited() const {
  return islimited(limited, range);
}
bool mjCTendon::is_actfrclimited() const {
  return islimited(actfrclimited, actfrcrange);
}

void mjCTendon::PointToLocal() {
  spec.element  = static_cast<mjsElement*>(this);
  spec.material = &spec_material_;
  spec.userdata = &spec_userdata_;
  spec.info     = &info;
  material      = nullptr;
  userdata      = nullptr;
}


void mjCTendon::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!spec_material_.empty() && model != m) {
    spec_material_ = m->prefix + spec_material_ + m->suffix;
  }
  prefix = m->prefix;
  suffix = m->suffix;
}


void mjCTendon::CopyFromSpec() {
  *static_cast<mjsTendon*>(this) = spec;

  material_ = spec_material_;
  userdata_ = spec_userdata_;

  // propagate model pointer to wraps and clear precompiled
  for (int i = 0; i < path.size(); i++) {
    path[i]->model = model;
    if (path[i]->Type() == mjWRAP_CYLINDER) { path[i]->spec.type = mjWRAP_SPHERE; }
  }
}


// desctructor
mjCTendon::~mjCTendon() {
  // delete objects allocated here
  for (unsigned int i = 0; i < path.size(); i++) { path[i]->Release(); }

  path.clear();
}


void mjCTendon::SetModel(mjCModel* _model) {
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  for (int i = 0; i < path.size(); i++) { path[i]->model = _model; }
}


// add site as wrap object
void mjCTendon::WrapSite(std::string wrapname, std::string_view wrapinfo) {
  // create wrap object
  mjCWrap* wrap = new mjCWrap(model, this);
  wrap->info    = wrapinfo;

  // set parameters, add to path
  wrap->spec.type = mjWRAP_SITE;
  wrap->name      = wrapname;
  wrap->id        = (int)path.size();
  path.push_back(wrap);
}


// add geom (with side site) as wrap object
void mjCTendon::WrapGeom(std::string wrapname, std::string sidesite, std::string_view wrapinfo) {
  // create wrap object
  mjCWrap* wrap = new mjCWrap(model, this);
  wrap->info    = wrapinfo;

  // set parameters, add to path
  wrap->spec.type = mjWRAP_SPHERE;  // replace with cylinder later if needed
  wrap->name      = wrapname;
  wrap->sidesite  = sidesite;
  wrap->id        = (int)path.size();
  path.push_back(wrap);
}


// add joint as wrap object
void mjCTendon::WrapJoint(std::string wrapname, double coef, std::string_view wrapinfo) {
  // create wrap object
  mjCWrap* wrap = new mjCWrap(model, this);
  wrap->info    = wrapinfo;

  // set parameters, add to path
  wrap->spec.type = mjWRAP_JOINT;
  wrap->name      = wrapname;
  wrap->prm       = coef;
  wrap->id        = (int)path.size();
  path.push_back(wrap);
}


// add pulley
void mjCTendon::WrapPulley(double divisor, std::string_view wrapinfo) {
  // create wrap object
  mjCWrap* wrap = new mjCWrap(model, this);
  wrap->info    = wrapinfo;

  // set parameters, add to path
  wrap->spec.type = mjWRAP_PULLEY;
  wrap->prm       = divisor;
  wrap->id        = (int)path.size();
  path.push_back(wrap);
}


// get number of wraps
int mjCTendon::NumWraps() const {
  return (int)path.size();
}


// get pointer to specified wrap
const mjCWrap* mjCTendon::GetWrap(int i) const {
  if (i >= 0 && i < (int)path.size()) { return path[i]; }
  return nullptr;
}


void mjCTendon::ResolveReferences(const mjCModel* m) {
  int nfailure = 0;
  int npulley  = 0;
  for (int i = 0; i < path.size(); i++) {
    std::string pname     = path[i]->name;
    std::string psidesite = path[i]->sidesite;
    if (path[i]->Type() == mjWRAP_PULLEY) { npulley++; }
    try {
      // look for wrapped element with namespace
      path[i]->name = prefix + pname + suffix;
      if (!psidesite.empty()) { path[i]->sidesite = prefix + psidesite + suffix; }
      path[i]->ResolveReferences(m);
    } catch (mjCError) {
      // remove namespace from wrap names
      path[i]->name = pname;
      if (!psidesite.empty()) { path[i]->sidesite = psidesite; }
      path[i]->ResolveReferences(m);
      nfailure++;
    }
  }
  if (nfailure == path.size() - npulley) {
    throw mjCError(this, "tendon '%s' (id = %d): no attached reference found", name.c_str(), id);
  }
  prefix.clear();
  suffix.clear();
}


// gauge of the lowest frame or body containing the path
mjCGauge mjCTendon::PathGauge() const {
  std::vector<const mjCBase*> elements;
  for (const mjCWrap* wrap : path) {
    if (wrap->Type() != mjWRAP_PULLEY && wrap->obj) { elements.push_back(wrap->obj); }
  }
  return LowestContainerGauge(elements);
}


// 1 if the length of the tendon is a length: spatial, or fixed with a slide joint
int mjCTendon::Linear() const {
  for (const mjCWrap* wrap : path) {
    if (wrap->Type() != mjWRAP_JOINT) { return 1; }
    const mjCJoint* joint = static_cast<const mjCJoint*>(wrap->obj);
    if (joint && joint->spec.type == mjJNT_SLIDE) { return 1; }
  }
  return 0;
}


// tendon type: fixed if the path is non-empty and has only joints, otherwise spatial
mjtTendon mjCTendon::Type() const {
  if (path.empty()) { return mjTENDON_SPATIAL; }
  for (const mjCWrap* wrap : path) {
    if (wrap->Type() != mjWRAP_JOINT) { return mjTENDON_SPATIAL; }
  }
  return mjTENDON_FIXED;
}


// scale the compiled copy by the gauge of the path, or back; the coefficient of a joint converts
// its coordinate to the length of the tendon, so it takes the ratio of their factors
void mjCTendon::ApplyGauge(bool inverse) {
  const mjCGauge g  = inverse ? PathGauge().Inverse() : PathGauge();
  const int      q  = Linear();
  const double   fq = q ? g.Shape(1) : 1;
  if (!inverse) {
    for (mjCWrap* wrap : path) {
      wrap->prmscale = 1;
      if (wrap->Type() == mjWRAP_JOINT && wrap->obj) {
        const mjCJoint* joint = static_cast<const mjCJoint*>(wrap->obj);
        bool            slide = joint->spec.type == mjJNT_SLIDE;
        wrap->prmscale        = fq / (slide ? joint->Gauge().Stretch(joint->spec.axis) : 1);
      }
    }
  }
  if (g.IsIdentity()) { return; }

  // length: shape class
  if (springlength[0] != -1 || springlength[1] != -1) {
    springlength[0] *= fq;
    springlength[1] *= fq;
  }
  springyield        *= fq;
  range[0]           *= fq;
  range[1]           *= fq;
  margin             *= fq;
  solimp_limit[2]    *= fq;
  solimp_friction[2] *= fq;
  width              *= g.Shape(1);

  // springs, dampers, armature and forces: force class
  const int a = 2 - q;
  for (int n = 0; n <= mjNPOLY; n++) {
    stiffness[n] *= g.Force(a - (n + 1) * q, 1);
    damping[n]   *= g.Force(a - (n + 1) * q, 1);
  }
  armature       *= g.Force(a - q, 1);
  frictionloss   *= g.Force(a, 1);
  actfrcrange[0] *= g.Force(a, 1);
  actfrcrange[1] *= g.Force(a, 1);
}


// compiler
void mjCTendon::Compile(void) {
  // compile all wraps in the path
  for (mjCWrap* wrap : path) { wrap->Compile(); }

  CopyFromSpec();

  // resize userdata
  if (userdata_.size() > model->nuser_tendon) {
    throw mjCError(this, "user has more values than nuser_tendon in tendon");
  }
  userdata_.resize(model->nuser_tendon);

  // check for empty path
  int sz = (int)path.size();
  if (!sz) {
    throw mjCError(this, "tendon '%s' (id = %d): path cannot be empty", name.c_str(), id);
  }

  // determine type
  bool spatial = (Type() == mjTENDON_SPATIAL);

  // require at least two objects in spatial path
  if (spatial && sz < 2) {
    throw mjCError(this,
                   "tendon '%s' (id = %d): spatial path must contain at least two objects",
                   name.c_str(),
                   id);
  }

  // require positive width
  if (spatial && width <= 0) {
    throw mjCError(this, "tendon '%s' (id = %d) must have positive width", name.c_str(), id);
  }

  // compile objects in path
  ResolveReferences(model);
  ApplyGauge();

  if (armature < 0) {
    throw mjCError(this,
                   "tendon '%s' (id = %d): tendon armature cannot be negative",
                   name.c_str(),
                   id);
  }

  // check path
  for (int i = 0; i < sz; i++) {
    // fixed
    if (!spatial) {
      // make sure all objects are joints
      if (path[i]->Type() != mjWRAP_JOINT) {
        throw mjCError(this,
                       "tendon '%s' (id = %d): spatial object found in fixed path at pos %d",
                       name.c_str(),
                       id,
                       i);
      }

      // make sure joint is slide or hinge
      mjtJoint jnt_type = static_cast<mjCJoint*>(path[i]->obj)->spec.type;
      if (jnt_type != mjJNT_SLIDE && jnt_type != mjJNT_HINGE) {
        throw mjCError(path[i],
                       "joint '%s' in tendon %d, wrap %d must be slide or hinge",
                       path[i]->name.c_str(),
                       id,
                       path[i]->id);
      }

      // joint cannot be repeated in a fixed tendon
      for (int j = 0; j < i; j++) {
        if (path[j]->obj == path[i]->obj) {
          throw mjCError(this, "tendon '%s' (id = %d): joint %d is repeated", name.c_str(), id, i);
        }
      }
    }

    // spatial path
    else {
      switch (path[i]->Type()) {
        case mjWRAP_PULLEY:
          // pulley should not follow other pulley
          if (i > 0 && path[i - 1]->Type() == mjWRAP_PULLEY) {
            throw mjCError(this,
                           "tendon '%s' (id = %d): consecutive pulleys (pos %d)",
                           name.c_str(),
                           id,
                           i);
          }

          // pulley should not be last
          if (i == sz - 1) {
            throw mjCError(this, "tendon '%s' (id = %d): path ends with pulley", name.c_str(), id);
          }
          break;

        case mjWRAP_SITE:
          // site needs a neighbor that is not a pulley
          if ((i == 0 || path[i - 1]->Type() == mjWRAP_PULLEY) &&
              (i == sz - 1 || path[i + 1]->Type() == mjWRAP_PULLEY)) {
            throw mjCError(this,
                           "tendon '%s' (id = %d): site %d needs a neighbor that is not a pulley",
                           name.c_str(),
                           id,
                           i);
          }

          // site cannot be repeated
          if (i < sz - 1 &&
              path[i + 1]->Type() == mjWRAP_SITE &&
              path[i]->obj->id == path[i + 1]->obj->id) {
            throw mjCError(this, "tendon '%s' (id = %d): site %d is repeated", name.c_str(), id, i);
          }

          break;

        case mjWRAP_SPHERE:
        case mjWRAP_CYLINDER:
          // geom must be bracketed by sites
          if (i == 0 ||
              i == sz - 1 ||
              path[i - 1]->Type() != mjWRAP_SITE ||
              path[i + 1]->Type() != mjWRAP_SITE) {
            throw mjCError(this,
                           "tendon '%s' (id = %d): geom at pos %d not bracketed by sites",
                           name.c_str(),
                           id,
                           i);
          }

          if (armature > 0) {
            throw mjCError(this,
                           "tendon '%s' (id = %d): geom wrapping not supported by tendon armature",
                           name.c_str(),
                           id);
          }

          break;

        case mjWRAP_JOINT:
          throw mjCError(this,
                         "tendon '%s (id = %d)': joint wrap found in spatial path at pos %d",
                         name.c_str(),
                         id,
                         i);

        default:
          throw mjCError(this,
                         "tendon '%s (id = %d)': invalid wrap object at pos %d",
                         name.c_str(),
                         id,
                         i);
      }
    }
  }

  // if limited is auto, set to 1 if range is specified, otherwise unlimited
  if (limited == mjLIMITED_AUTO) {
    bool hasrange = !(range[0] == 0 && range[1] == 0);
    checklimited(this, compiler->autolimits, "tendon", "", limited, hasrange);
  }

  // check limits
  if (range[0] >= range[1] && is_limited()) { throw mjCError(this, "invalid limits in tendon"); }

  // if limited is auto, set to 1 if range is specified, otherwise unlimited
  if (actfrclimited == mjLIMITED_AUTO) {
    bool hasactfrcrange = !(actfrcrange[0] == 0 && actfrcrange[1] == 0);
    checklimited(this,
                 compiler->autolimits,
                 "tendon",
                 "actuatorfrc",
                 actfrclimited,
                 hasactfrcrange);
  }

  // check actfrclimits
  if (actfrcrange[0] >= actfrcrange[1] && is_actfrclimited()) {
    throw mjCError(this, "invalid actuatorfrcrange in tendon");
  }
  if ((actfrcrange[0] > 0 || actfrcrange[1] < 0) && is_actfrclimited()) {
    throw mjCError(this, "invalid actuatorfrcrange in tendon");
  }

  // check springlength and springyield
  if (springlength[0] > springlength[1]) { throw mjCError(this, "invalid springlength in tendon"); }
  if (springyield < 0) { throw mjCError(this, "springyield cannot be negative in tendon"); }
}


//------------------ class mjCWrap implementation --------------------------------------------------

// constructor
mjCWrap::mjCWrap(mjCModel* _model, mjCTendon* _tendon) {
  elemtype = mjOBJ_UNKNOWN;

  // set model and tendon pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  tendon = _tendon;

  // clear variables
  spec.type = mjWRAP_NONE;
  obj       = nullptr;
  sideid    = -1;
  prm       = 0;
  sidesite.clear();

  // point to local
  PointToLocal();
  CopyFromSpec();
}


mjCWrap::mjCWrap(const mjCWrap& other) {
  *this = other;
}


mjCWrap& mjCWrap::operator=(const mjCWrap& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCWrap_*>(this) = static_cast<const mjCWrap_&>(other);
    *static_cast<mjsWrap*>(this)  = static_cast<const mjsWrap&>(other);
    obj                           = nullptr;
  }
  PointToLocal();
  return *this;
}


void mjCWrap::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.info    = &info;
}

void mjCWrap::CopyFromSpec() {
  *static_cast<mjsWrap*>(this) = spec;
}

void mjCWrap::NameSpace(const mjCModel* m) {
  name = m->prefix + name + m->suffix;
  if (!sidesite.empty()) { sidesite = m->prefix + sidesite + m->suffix; }
}

void mjCWrap::Compile(void) {
  CopyFromSpec();
}

void mjCWrap::ResolveReferences(const mjCModel* m) {
  mjCBase* pside;

  // handle wrap object types
  switch (spec.type) {
    case mjWRAP_JOINT:  // joint
      // find joint by name
      obj = m->FindObject(mjOBJ_JOINT, name);
      if (!obj) {
        throw mjCError(this,
                       "joint '%s' not found in tendon %d, wrap %d",
                       name.c_str(),
                       tendon->id,
                       id);
      }

      break;

    case mjWRAP_SPHERE:    // geom (cylinder type set here)
    case mjWRAP_CYLINDER:  // a geom whose type was set here before
      // find geom by name
      obj = m->FindObject(mjOBJ_GEOM, name);
      if (!obj) {
        throw mjCError(this,
                       "geom '%s' not found in tendon %d, wrap %d",
                       name.c_str(),
                       tendon->id,
                       id);
      }

      // set/check geom type, as authored: the geom may not have been compiled
      if (((mjCGeom*)obj)->spec.type == mjGEOM_CYLINDER) {
        spec.type = mjWRAP_CYLINDER;
      } else if (((mjCGeom*)obj)->spec.type == mjGEOM_SPHERE) {
        spec.type = mjWRAP_SPHERE;
      } else {
        throw mjCError(this,
                       "geom '%s' in tendon %d, wrap %d is not sphere or cylinder",
                       name.c_str(),
                       tendon->id,
                       id);
      }

      // process side site
      if (!sidesite.empty()) {
        // find site by name
        pside = m->FindObject(mjOBJ_SITE, sidesite);
        if (!pside) {
          throw mjCError(this,
                         "side site '%s' not found in tendon %d, wrap %d",
                         sidesite.c_str(),
                         tendon->id,
                         id);
        }

        // save side site id
        sideid = pside->id;
      }
      break;

    case mjWRAP_PULLEY:  // pulley
      // make sure divisor is positive
      if (prm <= 0) {
        throw mjCError(this,
                       "pulley divisor must be positive in tendon %d, wrap %d",
                       0,
                       tendon->id,
                       id);
      }

      break;

    case mjWRAP_SITE:  // site
      // find site by name
      obj = m->FindObject(mjOBJ_SITE, name);
      if (!obj) { throw mjCError(this, "site '%s' not found in wrap %d", name.c_str(), id); }
      break;

    default:  // SHOULD NOT OCCUR
      throw mjCError(this, "unknown wrap type in tendon %d, wrap %d", 0, tendon->id, id);
  }
}


//------------------ class mjCActuator implementation ----------------------------------------------

// initialize defaults
mjCActuator::mjCActuator(mjCModel* _model, mjCDef* _def) {
  mjs_defaultActuator(&spec);
  elemtype = mjOBJ_ACTUATOR;

  // clear private variables
  ptarget = nullptr;
  spec_target_.clear();
  spec_slidersite_.clear();
  spec_refsite_.clear();
  spec_userdata_.clear();
  trnid[0] = trnid[1] = -1;

  // reset to default if given
  if (_def) { *this = _def->Actuator(); }

  // set model, def
  model = _model;
  if (_model) compiler = &_model->spec.compiler;
  classname = _def ? _def->name : "main";

  // in case this actuator is not compiled
  CopyFromSpec();

  // point to local
  PointToLocal();

  // no previous state when an actuator is created
  actadr_     = -1;
  actdim_     = -1;
  historyadr_ = -1;
  historynum_ = 0;

  // input and output blocks, set by mjCModel; all actuator types are currently 1x1
  ctrladr_  = -1;
  ctrlnum_  = 1;
  ctrlspec_ = 0;
  outadr_   = -1;
  outnum_   = 1;
  so3_      = false;
}


mjCActuator::mjCActuator(const mjCActuator& other) {
  *this = other;
}


mjCActuator& mjCActuator::operator=(const mjCActuator& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCActuator_*>(this) = static_cast<const mjCActuator_&>(other);
    *static_cast<mjsActuator*>(this)  = static_cast<const mjsActuator&>(other);

    actadr_     = -1;
    actdim_     = -1;
    ctrladr_    = -1;
    outadr_     = -1;
    historyadr_ = -1;
    historynum_ = 0;
    ptarget     = nullptr;
  }
  PointToLocal();
  return *this;
}


void mjCActuator::ResetId() {
  id          = -1;
  actadr_     = -1;
  actdim_     = -1;
  ctrladr_    = -1;
  outadr_     = -1;
  historyadr_ = -1;
  historynum_ = 0;
}


void mjCActuator::ForgetKeyframes(const std::vector<std::string>& names, bool keep) {
  forgetvalues(act_, names, keep);
  forgetvalues(ctrl_, names, keep);
}


bool mjCActuator::is_ctrllimited() const {
  return islimited(ctrllimited, ctrlrange);
}
bool mjCActuator::is_forcelimited() const {
  return islimited(forcelimited, forcerange);
}
bool mjCActuator::is_actlimited() const {
  return islimited(actlimited, actrange);
}


std::vector<mjtNum>& mjCActuator::act(const std::string& state_name) {
  return act_[state_name];
}


std::vector<mjtNum>& mjCActuator::ctrl(const std::string& state_name) {
  return ctrl_[state_name];
}


void mjCActuator::PointToLocal() {
  spec.element            = static_cast<mjsElement*>(this);
  spec.userdata           = &spec_userdata_;
  spec.target             = &spec_target_;
  spec.refsite            = &spec_refsite_;
  spec.slidersite         = &spec_slidersite_;
  spec.plugin.plugin_name = &plugin_name;
  spec.plugin.name        = &plugin_instance_name;
  spec.info               = &info;
  userdata                = nullptr;
  target                  = nullptr;
  refsite                 = nullptr;
  slidersite              = nullptr;
}


void mjCActuator::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  if (!plugin_instance_name.empty()) {
    plugin_instance_name = m->prefix + plugin_instance_name + m->suffix;
  }
  if (!spec_target_.empty()) { spec_target_ = m->prefix + spec_target_ + m->suffix; }
  if (!spec_refsite_.empty()) { spec_refsite_ = m->prefix + spec_refsite_ + m->suffix; }
  if (!spec_slidersite_.empty()) { spec_slidersite_ = m->prefix + spec_slidersite_ + m->suffix; }
}


void mjCActuator::CopyFromSpec() {
  *static_cast<mjsActuator*>(this) = spec;

  userdata_          = spec_userdata_;
  target_            = spec_target_;
  refsite_           = spec_refsite_;
  slidersite_        = spec_slidersite_;
  plugin.active      = spec.plugin.active;
  plugin.element     = spec.plugin.element;
  plugin.plugin_name = spec.plugin.plugin_name;
  plugin.name        = spec.plugin.name;
}


void mjCActuator::CopyPlugin() {
  model->CopyExplicitPlugin(this);
}


void mjCActuator::ResolveReferences(const mjCModel* m) {
  trnid[0] = trnid[1] = -1;
  switch (trntype) {
    case mjTRN_JOINT:
    case mjTRN_JOINTINPARENT:
      // get joint
      ptarget = m->FindObject(mjOBJ_JOINT, target_);
      if (!ptarget) {
        throw mjCError(this,
                       "unknown transmission target '%s' for actuator id = %d",
                       target_.c_str(),
                       id);
      }
      break;

    case mjTRN_SLIDERCRANK:
      // get slidersite, copy in trnid[1]
      if (slidersite_.empty()) {
        throw mjCError(this, "missing base site for slider-crank '%s' (id = %d)", name.c_str(), id);
      }
      ptarget = m->FindObject(mjOBJ_SITE, slidersite_);
      if (!ptarget) {
        throw mjCError(this, "base site '%s' not found for actuator %d", slidersite_.c_str(), id);
      }
      trnid[1] = ptarget->id;

      // check cranklength
      if (cranklength <= 0) {
        throw mjCError(this,
                       "crank length must be positive in actuator '%s' (id = %d)",
                       name.c_str(),
                       id);
      }

      // proceed with regular target
      ptarget = m->FindObject(mjOBJ_SITE, target_);
      break;

    case mjTRN_TENDON:
      // get tendon
      ptarget = m->FindObject(mjOBJ_TENDON, target_);
      break;

    case mjTRN_SITE:
      // get refsite, copy into trnid[1]
      if (!refsite_.empty()) {
        ptarget = m->FindObject(mjOBJ_SITE, refsite_);
        if (!ptarget) {
          throw mjCError(this,
                         "reference site '%s' not found for actuator %d",
                         refsite_.c_str(),
                         id);
        }
        trnid[1] = ptarget->id;
      }

      // proceed with regular site target
      ptarget = m->FindObject(mjOBJ_SITE, target_);
      break;

    case mjTRN_BODY:
      // get body
      ptarget = m->FindObject(mjOBJ_BODY, target_);
      break;

    default:
      throw mjCError(this, "invalid transmission type in actuator");
  }

  // assign and check
  if (!ptarget) {
    throw mjCError(this, "transmission target '%s' not found in actuator %d", target_.c_str(), id);
  } else {
    trnid[0] = ptarget->id;
  }
}


// factor of the coordinate: the stretch along a slide, or that of a length
double mjCActuator::CoordinateFactor(const mjCGauge& g) const {
  if (!linear_) { return 1; }
  const bool      tojoint = trntype == mjTRN_JOINT || trntype == mjTRN_JOINTINPARENT;
  const mjCJoint* joint   = tojoint ? static_cast<const mjCJoint*>(ptarget) : nullptr;
  const bool      slide   = joint && joint->spec.type == mjJNT_SLIDE;
  return slide ? g.Stretch(joint->spec.axis) : g.Shape(1);
}


// scale the compiled copy by the gauge of the transmission target, or back. Q is the coordinate of
// the transmission, L^q, and G the generalized force, M L^(2-q) T^-2; gains and biases are force
// class whatever their dimension, and a control follows its declared input
void mjCActuator::ApplyGauge(bool inverse) {
  // transmission target, and whether its coordinate is a length
  const bool      linear6 = gear[0] || gear[1] || gear[2];
  const bool      tojoint = trntype == mjTRN_JOINT || trntype == mjTRN_JOINTINPARENT;
  const mjCJoint* joint   = tojoint ? static_cast<const mjCJoint*>(ptarget) : nullptr;
  if (!inverse) {
    gauge_  = mjCGauge();
    linear_ = 0;
    switch (trntype) {
      case mjTRN_JOINT:
      case mjTRN_JOINTINPARENT:
        gauge_  = joint->Gauge();
        linear_ = joint->spec.type == mjJNT_SLIDE || (joint->spec.type == mjJNT_FREE && linear6);
        break;
      case mjTRN_TENDON:
        gauge_  = static_cast<const mjCTendon*>(ptarget)->PathGauge();
        linear_ = static_cast<const mjCTendon*>(ptarget)->Linear();
        break;
      case mjTRN_SITE:
        gauge_  = ptarget->Gauge();
        linear_ = linear6;
        break;
      case mjTRN_SLIDERCRANK:
        gauge_ =
            mjCGauge::Mean(ptarget->Gauge(), model->FindObject(mjOBJ_SITE, slidersite_)->Gauge());
        linear_ = 1;
        break;
      case mjTRN_BODY:
        gauge_  = ptarget->ContentGauge();
        linear_ = 1;
        break;
      default:
        break;
    }
    actscale_ = 1;
    for (int j = 0; j < 4; j++) { ctrlscales_[j] = 1; }
  }
  const mjCGauge g = inverse ? gauge_.Inverse() : gauge_;
  if (g.IsIdentity()) { return; }

  // factor of the coordinate: the stretch along a slide, or that of a length
  const double fq = CoordinateFactor(g);

  // factors of the inputs which are declared, packed in the order pos, vel, ff, voltage, and of
  // the activation
  double ctrlf[4] = {1, 1, 1, 1}, actf = 1;
  auto   packinputs = [&](double pos, double vel, double ff, double voltage) {
    const int    bits[4] = {mjINPUT_POS, mjINPUT_VEL, mjINPUT_FF, mjINPUT_VOLTAGE};
    const double f[4]    = {pos, vel, ff, voltage};
    for (int k = 0, j = 0; k < 4; k++) {
      if (ctrlspec_ & bits[k]) { ctrlf[j++] = f[k]; }
    }
  };

  // parameters which the gauge cannot interpret
  if (!g.IsForceIdentity() && (plugin.active ||
                               gaintype == mjGAIN_USER ||
                               biastype == mjBIAS_USER ||
                               dyntype == mjDYN_USER ||
                               gaintype == mjGAIN_DCMOTOR ||
                               dyntype == mjDYN_DCMOTOR)) {
    throw mjCError(this,
                   "the parameters of this actuator cannot be scaled; use scaling 'geometry'");
  }

  // transmission: shape class
  const int q = linear_;
  if ((trntype == mjTRN_SITE || (joint && joint->spec.type == mjJNT_FREE)) && linear6) {
    for (int i = 3; i < 6; i++) { gear[i] *= g.Shape(1); }
  }
  cranklength    *= g.Shape(1);
  lengthrange[0] *= fq;
  lengthrange[1] *= fq;

  // forces, dampers and armature: force class
  const int a    = 2 - q;
  forcerange[0] *= g.Force(a, 1);
  forcerange[1] *= g.Force(a, 1);
  for (int n = 0; n <= mjNPOLY; n++) { damping[n] *= g.Force(a - (n + 1) * q, 1); }
  armature *= g.Force(a - q, 1);

  // affine bias: G, G/Q, and G T/Q if negative; positive biasprm[2] is a damping ratio
  auto affinebias = [&]() {
    biasprm[0] *= g.Force(a, 1);
    biasprm[1] *= g.Force(a - q, 1);
    if (biasprm[2] < 0) { biasprm[2] *= g.Force(a - q, 1); }
  };

  // fixed or affine gain: the gain multiplies the declared input, or a command
  if (gaintype == mjGAIN_FIXED || gaintype == mjGAIN_AFFINE) {
    int    uL = 0, uM = 0;
    double fu = 1;
    if (ctrlspec == mjINPUT_POS || ctrlspec == mjINPUT_VEL) {
      uL = q;
      fu = fq;
    } else if (ctrlspec == mjINPUT_PRESSURE) {
      uL = -1;
      uM = 1;
      fu = g.Force(-1, 1);
    }
    gainprm[0] *= g.Force(a - uL, 1 - uM);
    if (gaintype == mjGAIN_AFFINE) {
      gainprm[1] *= g.Force(a - uL - q, 1 - uM);
      gainprm[2] *= g.Force(a - uL - q, 1 - uM);
    }
    if (biastype == mjBIAS_AFFINE) { affinebias(); }
    ctrlf[0] = fu;
    if (dyntype == mjDYN_INTEGRATOR || dyntype == mjDYN_FILTER || dyntype == mjDYN_FILTEREXACT) {
      actf = fu;
    }
  }

  // muscle: peak force if given; the rest is in units of the optimal length and of time
  else if (gaintype == mjGAIN_MUSCLE) {
    if (gainprm[2] > 0) { gainprm[2] *= g.Force(a, 1); }
    if (biastype == mjBIAS_MUSCLE && biasprm[2] > 0) { biasprm[2] *= g.Force(a, 1); }
  }

  // pid: inputs pos, vel and ff; ki is G/(Q T), the integral and slew limits are Q, and so are the
  // states, a setpoint and an integral over time
  else if (gaintype == mjGAIN_PID) {
    gainprm[0] *= g.Force(a - q, 1);
    affinebias();
    dynprm[0]   *= fq;
    dynprm[1]   *= fq;
    velrange[0] *= fq;
    velrange[1] *= fq;
    ffrange[0]  *= g.Force(a, 1);
    ffrange[1]  *= g.Force(a, 1);
    packinputs(fq, fq, g.Force(a, 1), 1);
    actf = fq;
  }

  // orientation servo: an angle or quaternion controls a torque
  else if (gaintype == mjGAIN_SO3) {
    gainprm[0] *= g.Force(2, 1);
    biasprm[1] *= g.Force(2, 1);
    if (biasprm[2] < 0) { biasprm[2] *= g.Force(2, 1); }
  }

  // dc motor under geometry: only position and velocity quantities change
  else if (gaintype == mjGAIN_DCMOTOR) {
    if (ctrlspec_ & (mjINPUT_POS | mjINPUT_VEL)) { dynprm[7] *= fq; }
    dynprm[8]   *= fq;
    biasprm[1]  /= fq;
    biasprm[5]  *= fq;
    velrange[0] *= fq;
    velrange[1] *= fq;
    packinputs(fq, fq, 1, 1);
  }

  // the control range is that of the first input, or of the position input of a dc motor
  const double frange  = gaintype != mjGAIN_DCMOTOR || (ctrlspec_ & mjINPUT_POS) ? ctrlf[0] : 1;
  ctrlrange[0]        *= frange;
  ctrlrange[1]        *= frange;
  actrange[0]         *= actf;
  actrange[1]         *= actf;
  if (!inverse) {
    actscale_ = actf;
    for (int j = 0; j < 4; j++) { ctrlscales_[j] = ctrlf[j]; }
  }
}


// compiler
void mjCActuator::Compile(void) {
  CopyFromSpec();

  // reset input/output block widths, resolved below
  ctrlnum_  = 1;
  ctrlspec_ = 0;
  outnum_   = 1;
  so3_      = false;

  // resize userdata
  if (userdata_.size() > model->nuser_actuator) {
    throw mjCError(this,
                   "user has more values than nuser_actuator in actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }
  userdata_.resize(model->nuser_actuator);

  // check for missing target name
  if (target_.empty()) { throw mjCError(this, "missing transmission target for actuator"); }

  // find transmission target in object arrays
  ResolveReferences(model);

  // SO3 geodesic servo: validate and resolve the SO3 transmission
  if (gaintype == mjGAIN_SO3 || biastype == mjBIAS_SO3) {
    if (gaintype != mjGAIN_SO3 || biastype != mjBIAS_SO3) {
      throw mjCError(this,
                     "gaintype and biastype must both be 'so3' in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (dyntype != mjDYN_NONE && dyntype != mjDYN_INTEGRATOR) {
      throw mjCError(this,
                     "so3 requires dyntype 'none' or 'integrator' in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (gainprm[0] != -biasprm[1]) {
      throw mjCError(this,
                     "so3 requires gainprm[0] == -biasprm[1] in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (trntype == mjTRN_SITE) {
      if (refsite_.empty()) {
        throw mjCError(this,
                       "so3 site transmission requires refsite in actuator '%s' (id = %d)",
                       name.c_str(),
                       id);
      }
    } else if (trntype == mjTRN_JOINT) {
      if (((mjCJoint*)ptarget)->spec.type != mjJNT_BALL) {
        throw mjCError(this,
                       "so3 joint transmission requires a ball joint in actuator '%s' "
                       "(id = %d)",
                       name.c_str(),
                       id);
      }
    } else {
      throw mjCError(this,
                     "so3 requires site or ball joint transmission in actuator '%s' "
                     "(id = %d)",
                     name.c_str(),
                     id);
    }

    // integrator variant: activation is the 3D orientation setpoint
    if (dyntype == mjDYN_INTEGRATOR) {
      if (actdim > 0 && actdim != 3) {
        throw mjCError(this,
                       "so3 integrator requires actdim 3 in actuator '%s' (id = %d)",
                       name.c_str(),
                       id);
      }
      actdim = 3;

      // the act setpoint is re-anchored to a bounded representative at integration time
      if (actlimited == mjLIMITED_TRUE && actrange[0] == 0 && actrange[1] == 0) {
        actlimited = mjLIMITED_FALSE;
      }
    }

    // input chart: expmap (3 controls, default) or quat (4 controls)
    ctrlspec_ = ctrlspec ? ctrlspec : mjCHART_EXPMAP;
    if (ctrlspec_ == mjCHART_QUAT) {
      if (dyntype != mjDYN_NONE) {
        throw mjCError(this,
                       "so3 quat input requires dyntype 'none' in actuator '%s' (id = %d)",
                       name.c_str(),
                       id);
      }
    } else if (ctrlspec_ != mjCHART_EXPMAP) {
      throw mjCError(this,
                     "so3 input must be expmap or quat in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }

    // force is clamped on the norm of the output torque: lower bound must be 0
    if (is_forcelimited() && forcerange[0] != 0) {
      throw mjCError(this,
                     "so3 forcerange bounds the force norm, lower bound must be 0 in "
                     "actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }

    // input and output blocks
    ctrlnum_ = ctrlspec_ == mjCHART_QUAT ? 4 : 3;
    outnum_  = 3;
    so3_     = true;
  }

  // PID servo: validate and resolve input block
  if (gaintype == mjGAIN_PID) {
    if (biastype != mjBIAS_AFFINE) {
      throw mjCError(this,
                     "pid requires biastype 'affine' in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (dyntype != mjDYN_NONE && dyntype != mjDYN_PID) {
      throw mjCError(this,
                     "pid requires dyntype 'none' or 'pid' in actuator '%s' "
                     "(id = %d)",
                     name.c_str(),
                     id);
    }
    if (dyntype == mjDYN_NONE && gainprm[0]) {
      throw mjCError(this,
                     "ki (gainprm[0]) requires dyntype 'pid' in actuator '%s' "
                     "(id = %d)",
                     name.c_str(),
                     id);
    }
    if (trntype == mjTRN_BODY) {
      throw mjCError(this,
                     "pid cannot use body transmission, actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }

    // controller states, slot order [slew, integral]: gated on slewmax (dynprm[1]) and ki
    if (dyntype == mjDYN_PID) {
      if (dynprm[0] < 0) {
        throw mjCError(this,
                       "imax (dynprm[0]) must be non-negative in actuator '%s' (id = %d)",
                       name.c_str(),
                       id);
      }
      if (dynprm[1] < 0) {
        throw mjCError(this,
                       "slewmax (dynprm[1]) must be non-negative in actuator '%s' (id = %d)",
                       name.c_str(),
                       id);
      }
      int nslot = (dynprm[1] > 0) + (gainprm[0] > 0);
      if (actdim > 0 && actdim != nslot) {
        throw mjCError(this,
                       "pid controller states require matching actdim in actuator '%s' "
                       "(id = %d)",
                       name.c_str(),
                       id);
      }
      actdim = nslot;
    }

    // input block: any subset of [pos, vel, ff], default [pos, vel]
    ctrlspec_ = ctrlspec ? ctrlspec : (mjINPUT_POS | mjINPUT_VEL);
    if (ctrlspec_ & ~(mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF)) {
      throw mjCError(this,
                     "pid inputs are a subset of [pos, vel, ff] in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (dyntype == mjDYN_PID && !(ctrlspec_ & mjINPUT_POS)) {
      throw mjCError(this,
                     "pid controller states require the pos input in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    ctrlnum_ =
        !!(ctrlspec_ & mjINPUT_POS) + !!(ctrlspec_ & mjINPUT_VEL) + !!(ctrlspec_ & mjINPUT_FF);
  }

  // DC motor: resolve input block (default: raw voltage command)
  if (gaintype == mjGAIN_DCMOTOR) {
    ctrlspec_ = ctrlspec ? ctrlspec : mjINPUT_VOLTAGE;
    if (ctrlspec_ != mjINPUT_NONE &&
        (ctrlspec_ & ~(mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF | mjINPUT_VOLTAGE))) {
      throw mjCError(this,
                     "dcmotor inputs are 'none' or a subset of [pos, vel, ff, voltage] in "
                     "actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }

    // controller inputs engage the torque-space controller, which divides by the motor constant
    int controller =
        ctrlspec_ == mjINPUT_NONE ? 0 : ctrlspec_ & (mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF);
    if (controller && gainprm[1] <= 0) {
      throw mjCError(this,
                     "dcmotor controller inputs require a positive motor constant in "
                     "actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (!controller && (gainprm[4] || gainprm[5] || gainprm[6])) {
      throw mjCError(this,
                     "dcmotor controller gains require a controller input [pos, vel, ff] "
                     "in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    if (gainprm[5] > 0 && !(ctrlspec_ & mjINPUT_POS)) {
      throw mjCError(this,
                     "dcmotor integral gain requires the pos input in actuator '%s' "
                     "(id = %d)",
                     name.c_str(),
                     id);
    }
    ctrlnum_ = !!(ctrlspec_ & mjINPUT_POS) +
               !!(ctrlspec_ & mjINPUT_VEL) +
               !!(ctrlspec_ & mjINPUT_FF) +
               !!(ctrlspec_ & mjINPUT_VOLTAGE);
    if (!controller && dynprm[7] > 0) {
      throw mjCError(this,
                     "dcmotor slew rate limiting requires a controller input [pos, vel, "
                     "ff] in actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
  }

  // pid dynamics are pid-only
  if (dyntype == mjDYN_PID && gaintype != mjGAIN_PID) {
    throw mjCError(this,
                   "dyntype 'pid' requires gaintype 'pid', actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  // fixed and affine gains: one declared input, pos, vel or pressure; none declared: a command
  if (gaintype == mjGAIN_FIXED || gaintype == mjGAIN_AFFINE) {
    if (ctrlspec &&
        ctrlspec != mjINPUT_POS &&
        ctrlspec != mjINPUT_VEL &&
        ctrlspec != mjINPUT_PRESSURE) {
      throw mjCError(this,
                     "fixed and affine gains take one input, 'pos', 'vel' or 'pressure', in "
                     "actuator '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
    ctrlspec_ = ctrlspec;
  }

  // muscle and user gains take no input signature
  else if (ctrlspec &&
           gaintype != mjGAIN_SO3 &&
           gaintype != mjGAIN_PID &&
           gaintype != mjGAIN_DCMOTOR) {
    throw mjCError(this,
                   "input is not available for muscle and user gains, actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  ApplyGauge();

  // check damping/armature only valid for joint and tendon transmission
  bool has_damping = false;
  for (int i = 0; i < mjNPOLY + 1; i++) {
    if (damping[i] != 0) {
      has_damping = true;
      break;
    }
  }
  if (has_damping &&
      trntype != mjTRN_JOINT &&
      trntype != mjTRN_JOINTINPARENT &&
      trntype != mjTRN_TENDON) {
    throw mjCError(this,
                   "damping requires joint or tendon transmission in actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }
  if (armature != 0 &&
      trntype != mjTRN_JOINT &&
      trntype != mjTRN_JOINTINPARENT &&
      trntype != mjTRN_TENDON) {
    throw mjCError(this,
                   "armature requires joint or tendon transmission in actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  // handle inheritrange
  if (((gaintype == mjGAIN_FIXED && gainprm[0] == -biasprm[1]) || gaintype == mjGAIN_PID) &&
      biastype == mjBIAS_AFFINE &&
      inheritrange > 0) {
    // semantic of actuator is the same as transmission, inheritrange is applicable
    double* range;
    if (dyntype == mjDYN_NONE || dyntype == mjDYN_FILTEREXACT || dyntype == mjDYN_PID) {
      // position or pid actuator: range applies to the position input
      range = ctrlrange;
    } else if (dyntype == mjDYN_INTEGRATOR) {
      // intvelocity actuator
      range = actrange;
    } else {
      throw mjCError(this,
                     "inheritrange only available for position "
                     "and intvelocity actuators");
    }

    const double* target_range;
    if (trntype == mjTRN_JOINT) {
      mjCJoint* pjnt = (mjCJoint*)ptarget;
      if (pjnt->spec.type != mjJNT_HINGE && pjnt->spec.type != mjJNT_SLIDE) {
        throw mjCError(this,
                       "inheritrange can only be used with hinge and slide joints, "
                       "actuator");
      }
      target_range = pjnt->get_range();
    } else if (trntype == mjTRN_TENDON) {
      mjCTendon* pten = (mjCTendon*)ptarget;
      target_range    = pten->get_range();
    } else {
      throw mjCError(this,
                     "inheritrange can only be used with joint and tendon transmission, "
                     "actuator");
    }

    if (target_range[0] == target_range[1]) {
      throw mjCError(this,
                     "inheritrange used but target '%s' has no range defined in actuator %d",
                     target_.c_str(),
                     id);
    }

    // set range automatically
    double mean   = 0.5 * (target_range[1] + target_range[0]);
    double radius = 0.5 * (target_range[1] - target_range[0]) * inheritrange;
    range[0]      = mean - radius;
    range[1]      = mean + radius;
  }

  // if limited is auto, check for inconsistency wrt to autolimits
  if (forcelimited == mjLIMITED_AUTO) {
    bool hasrange = !(forcerange[0] == 0 && forcerange[1] == 0);
    checklimited(this, compiler->autolimits, "actuator", "force", forcelimited, hasrange);
  }
  if (ctrllimited == mjLIMITED_AUTO) {
    bool hasrange = !(ctrlrange[0] == 0 && ctrlrange[1] == 0);
    checklimited(this, compiler->autolimits, "actuator", "ctrl", ctrllimited, hasrange);
  }
  if (actlimited == mjLIMITED_AUTO) {
    bool hasrange = !(actrange[0] == 0 && actrange[1] == 0);
    checklimited(this, compiler->autolimits, "actuator", "act", actlimited, hasrange);
  }

  // check limits
  if (forcerange[0] >= forcerange[1] && is_forcelimited()) {
    throw mjCError(this, "invalid force range for actuator");
  }
  if (ctrlrange[0] >= ctrlrange[1] && is_ctrllimited()) {
    throw mjCError(this, "invalid control range for actuator");
  }
  if (actrange[0] >= actrange[1] && is_actlimited()) {
    throw mjCError(this, "invalid actrange for actuator");
  }
  if (is_actlimited() && dyntype == mjDYN_NONE) {
    throw mjCError(this, "actrange specified but dyntype is 'none' in actuator");
  }

  // check and set actdim
  if (!plugin.active) {
    if (actdim > 1 &&
        dyntype != mjDYN_USER &&
        dyntype != mjDYN_DCMOTOR &&
        dyntype != mjDYN_PID &&
        !so3_) {
      throw mjCError(this, "actdim > 1 is only allowed for dyntype 'user', 'dcmotor' and 'pid'");
    }
    if (actdim == 1 && dyntype == mjDYN_NONE) {
      throw mjCError(this, "invalid actdim 1 in stateless actuator");
    }
    if (actdim == 0 && dyntype != mjDYN_NONE && dyntype != mjDYN_DCMOTOR) {
      throw mjCError(this, "invalid actdim 0 in stateful actuator");
    }
  }

  // DC motor: one activation per enabled state, as in mj_dcmotorSlots
  if (dyntype == mjDYN_DCMOTOR && !plugin.active) {
    int nslot =
        (dynprm[7] > 0) + (gainprm[5] > 0) + (dynprm[2] > 0) + (dynprm[5] > 0) + (dynprm[0] > 0);
    if (actdim < 0) {
      actdim = nslot;
    } else if (actdim != nslot) {
      char msg[200];
      mju::sprintf_arr(msg,
                       "DC motor parameters in actuator '%s' (id = %d) require actdim %d, got %d",
                       name.c_str(),
                       id,
                       nslot,
                       actdim);
      throw mjCError(this, "%s", msg);
    }
  }

  // set actdim to 1 if it is unset and type is standard one-activation dyntype
  if (actdim < 0) { actdim = (dyntype != mjDYN_NONE); }

  // DC motor always uses actearly
  if (dyntype == mjDYN_DCMOTOR && !actearly) {
    throw mjCError(this,
                   "actearly cannot be false for DC motor actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  // check muscle parameters
  for (int i = 0; i < 2; i++) {
    // select gain or bias
    double* prm = NULL;
    if (i == 0 && gaintype == mjGAIN_MUSCLE) {
      prm = gainprm;
    } else if (i == 1 && biastype == mjBIAS_MUSCLE) {
      prm = biasprm;
    }

    // nothing to check
    if (!prm) { continue; }

    // range
    if (prm[0] >= prm[1]) { throw mjCError(this, "range[0]<range[1] required in muscle"); }

    // lmin<1<lmax
    if (prm[4] >= 1 || prm[5] <= 1) { throw mjCError(this, "lmin<1<lmax required in muscle"); }

    // scale, vmax, fpmax, fvmax>0
    if (prm[3] <= 0 || prm[6] <= 0 || prm[7] <= 0 || prm[8] <= 0) {
      throw mjCError(this,
                     "positive scale, vmax, fpmax, fvmax required in muscle '%s' (id = %d)",
                     name.c_str(),
                     id);
    }
  }

  // plugin
  if (plugin.active) {
    if (plugin_name.empty() && plugin_instance_name.empty()) {
      throw mjCError(this,
                     "neither 'plugin' nor 'instance' is specified for actuator '%s', (id = %d)",
                     name.c_str(),
                     id);
    }

    mjCPlugin* plugin_instance = static_cast<mjCPlugin*>(plugin.element);
    model->ResolvePlugin(this, plugin_name, plugin_instance_name, &plugin_instance);
    plugin.element           = plugin_instance;
    const mjpPlugin* pplugin = mjp_getPluginAtSlot(plugin_instance->plugin_slot);
    if (!(pplugin->capabilityflags & mjPLUGIN_ACTUATOR)) {
      throw mjCError(this, "plugin '%s' does not support actuators", pplugin->name);
    }
  }

  // validate delay
  if (delay > 0 && nsample <= 0) {
    throw mjCError(this, "setting delay > 0 without a history buffer");
  }
  if ((delay > 0 || nsample > 0) && ctrlnum_ == 0) {
    throw mjCError(this,
                   "history and delay require an input in actuator '%s' (id = %d)",
                   name.c_str(),
                   id);
  }

  // nsample is limited to 2^24 because the cursor is stored as an mjtNum, which may be a float
  // single-precision floats can represent all integers up to 2^24 exactly
  if (nsample > 16777216) {
    throw mjCError(this, "at most 2^24 samples in history buffer, got %d", nullptr, nsample);
  }

  // resolve per-input control ranges: broadcast ctrlrange, pid overrides vel and ff
  for (int j = 0; j < ctrlnum_ && j < 4; j++) {
    ctrllimiteds_[j]  = (mjtByte)is_ctrllimited();
    ctrlranges_[j][0] = ctrlrange[0];
    ctrlranges_[j][1] = ctrlrange[1];
  }
  if (gaintype == mjGAIN_PID) {
    // present inputs pack in canonical order [pos, vel, ff]; pos keeps the ctrlrange broadcast
    int j = ctrlspec_ & mjINPUT_POS ? 1 : 0;
    if (ctrlspec_ & mjINPUT_VEL) {
      ctrllimiteds_[j]  = velrange[0] < velrange[1];
      ctrlranges_[j][0] = velrange[0];
      ctrlranges_[j][1] = velrange[1];
      j++;
    }
    if (ctrlspec_ & mjINPUT_FF) {
      ctrllimiteds_[j]  = ffrange[0] < ffrange[1];
      ctrlranges_[j][0] = ffrange[0];
      ctrlranges_[j][1] = ffrange[1];
    }
  }
}


//------------------ class mjCSensor implementation ------------------------------------------------

// initialize defaults
mjCSensor::mjCSensor(mjCModel* _model) {
  mjs_defaultSensor(&spec);
  elemtype = mjOBJ_SENSOR;

  // set model
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear private variables
  spec_objname_.clear();
  spec_refname_.clear();
  spec_userdata_.clear();
  historyadr_ = -1;
  historynum_ = 0;
  obj         = nullptr;
  ref         = nullptr;

  // in case this sensor is not compiled
  CopyFromSpec();

  // point to local
  PointToLocal();
}


mjCSensor::mjCSensor(const mjCSensor& other) {
  *this = other;
}


mjCSensor& mjCSensor::operator=(const mjCSensor& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCSensor_*>(this) = static_cast<const mjCSensor_&>(other);
    *static_cast<mjsSensor*>(this)  = static_cast<const mjsSensor&>(other);

    historyadr_ = -1;
    historynum_ = 0;
    obj         = nullptr;
    ref         = nullptr;
  }
  PointToLocal();
  return *this;
}


void mjCSensor::ResetId() {
  id          = -1;
  historyadr_ = -1;
  historynum_ = 0;
}


void mjCSensor::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);

  spec.userdata           = &spec_userdata_;
  spec.objname            = &spec_objname_;
  spec.refname            = &spec_refname_;
  spec.plugin.plugin_name = &plugin_name;
  spec.plugin.name        = &plugin_instance_name;
  spec.info               = &info;
  userdata                = nullptr;
  objname                 = nullptr;
  refname                 = nullptr;
}


void mjCSensor::NameSpace(const mjCModel* m) {
  if (!name.empty()) { name = m->prefix + name + m->suffix; }
  if (!plugin_instance_name.empty()) {
    plugin_instance_name = m->prefix + plugin_instance_name + m->suffix;
  }
  prefix = m->prefix;
  suffix = m->suffix;
}


void mjCSensor::CopyFromSpec() {
  *static_cast<mjsSensor*>(this) = spec;

  userdata_          = spec_userdata_;
  objname_           = spec_objname_;
  refname_           = spec_refname_;
  plugin.active      = spec.plugin.active;
  plugin.element     = spec.plugin.element;
  plugin.plugin_name = spec.plugin.plugin_name;
  plugin.name        = spec.plugin.name;
}


void mjCSensor::CopyPlugin() {
  model->CopyExplicitPlugin(this);
}


void mjCSensor::ResolveReferences(const mjCModel* m) {
  obj      = nullptr;
  ref      = nullptr;
  objname_ = prefix + objname_ + suffix;
  refname_ = prefix + refname_ + suffix;

  // get references using the namespace
  if (objtype != mjOBJ_UNKNOWN) { obj = m->FindObject(objtype, objname_); }
  if (reftype != mjOBJ_UNKNOWN) { ref = m->FindObject(reftype, refname_); }

  // if failure and both were requested, use namespace only on one
  if (objtype != mjOBJ_UNKNOWN && reftype != mjOBJ_UNKNOWN && !obj && ref) {
    objname_ = spec_objname_;
    obj      = m->FindObject(objtype, objname_);
  }
  if (objtype != mjOBJ_UNKNOWN && reftype != mjOBJ_UNKNOWN && obj && !ref) {
    refname_ = spec_refname_;
    ref      = m->FindObject(reftype, refname_);
  }

  // check object
  if (objtype != mjOBJ_UNKNOWN) {
    // check for missing object name
    if (objname_.empty()) { throw mjCError(this, "missing name of sensorized object in sensor"); }

    // find name
    if (!obj) {
      throw mjCError(this, "unrecognized name '%s' of sensorized object", objname_.c_str());
    }

  } else if (type != mjSENS_E_POTENTIAL &&
             type != mjSENS_E_KINETIC &&
             type != mjSENS_CLOCK &&
             type != mjSENS_PLUGIN &&
             type != mjSENS_CONTACT &&
             type != mjSENS_USER) {
    throw mjCError(this, "invalid type in sensor");
  }

  // check reference object
  if (reftype != mjOBJ_UNKNOWN) {
    // check for missing object name
    if (refname_.empty()) {
      throw mjCError(this, "missing name of reference frame object in sensor");
    }

    // find name
    if (!ref) { throw mjCError(this, "unrecognized name '%s' of object", refname_.c_str()); }

    // must be attached to object with spatial frame
    if (reftype != mjOBJ_BODY &&
        reftype != mjOBJ_XBODY &&
        reftype != mjOBJ_GEOM &&
        reftype != mjOBJ_SITE &&
        reftype != mjOBJ_CAMERA) {
      throw mjCError(this,
                     "reference frame object must be (x)body, geom, site or camera in sensor");
    }
  }

  spec_objname_ = objname_;
  spec_refname_ = refname_;
  prefix.clear();
  suffix.clear();
}

// return sensor datatype
mjtDataType sensorDatatype(mjtSensor type) {
  switch (type) {
    case mjSENS_TOUCH:
    case mjSENS_INSIDESITE:
      return mjDATATYPE_POSITIVE;

    case mjSENS_FRAMEXAXIS:
    case mjSENS_FRAMEYAXIS:
    case mjSENS_FRAMEZAXIS:
    case mjSENS_GEOMNORMAL:
      return mjDATATYPE_AXIS;

    case mjSENS_BALLQUAT:
    case mjSENS_FRAMEQUAT:
      return mjDATATYPE_QUATERNION;

    case mjSENS_ACCELEROMETER:
    case mjSENS_VELOCIMETER:
    case mjSENS_GYRO:
    case mjSENS_FORCE:
    case mjSENS_TORQUE:
    case mjSENS_MAGNETOMETER:
    case mjSENS_CAMPROJECTION:
    case mjSENS_JOINTPOS:
    case mjSENS_JOINTVEL:
    case mjSENS_TENDONPOS:
    case mjSENS_TENDONVEL:
    case mjSENS_ACTUATORPOS:
    case mjSENS_ACTUATORVEL:
    case mjSENS_ACTUATORFRC:
    case mjSENS_JOINTACTFRC:
    case mjSENS_TENDONACTFRC:
    case mjSENS_BALLANGVEL:
    case mjSENS_JOINTLIMITPOS:
    case mjSENS_JOINTLIMITVEL:
    case mjSENS_JOINTLIMITFRC:
    case mjSENS_TENDONLIMITPOS:
    case mjSENS_TENDONLIMITVEL:
    case mjSENS_TENDONLIMITFRC:
    case mjSENS_FRAMEPOS:
    case mjSENS_FRAMELINVEL:
    case mjSENS_FRAMEANGVEL:
    case mjSENS_FRAMELINACC:
    case mjSENS_FRAMEANGACC:
    case mjSENS_SUBTREECOM:
    case mjSENS_SUBTREELINVEL:
    case mjSENS_SUBTREEANGMOM:
    case mjSENS_GEOMDIST:
    case mjSENS_GEOMFROMTO:
    case mjSENS_RANGEFINDER:
    case mjSENS_CONTACT:
    case mjSENS_TACTILE:
    case mjSENS_E_POTENTIAL:
    case mjSENS_E_KINETIC:
    case mjSENS_CLOCK:
    case mjSENS_PLUGIN:
    case mjSENS_USER:
      return mjDATATYPE_REAL;
  }

  return mjDATATYPE_REAL;  // all cases are covered but GCC is extra persnickety
}

// return sensor needstage
mjtStage sensorNeedstage(mjtSensor type) {
  switch (type) {
    case mjSENS_TOUCH:
    case mjSENS_ACCELEROMETER:
    case mjSENS_FORCE:
    case mjSENS_TORQUE:
    case mjSENS_ACTUATORFRC:
    case mjSENS_JOINTACTFRC:
    case mjSENS_TENDONACTFRC:
    case mjSENS_JOINTLIMITFRC:
    case mjSENS_TENDONLIMITFRC:
    case mjSENS_FRAMELINACC:
    case mjSENS_FRAMEANGACC:
    case mjSENS_CONTACT:
      return mjSTAGE_ACC;

    case mjSENS_VELOCIMETER:
    case mjSENS_GYRO:
    case mjSENS_JOINTVEL:
    case mjSENS_TENDONVEL:
    case mjSENS_ACTUATORVEL:
    case mjSENS_BALLANGVEL:
    case mjSENS_JOINTLIMITVEL:
    case mjSENS_TENDONLIMITVEL:
    case mjSENS_FRAMELINVEL:
    case mjSENS_FRAMEANGVEL:
    case mjSENS_SUBTREELINVEL:
    case mjSENS_SUBTREEANGMOM:
    case mjSENS_TACTILE:
      return mjSTAGE_VEL;

    case mjSENS_MAGNETOMETER:
    case mjSENS_RANGEFINDER:
    case mjSENS_CAMPROJECTION:
    case mjSENS_JOINTPOS:
    case mjSENS_TENDONPOS:
    case mjSENS_ACTUATORPOS:
    case mjSENS_BALLQUAT:
    case mjSENS_JOINTLIMITPOS:
    case mjSENS_TENDONLIMITPOS:
    case mjSENS_FRAMEPOS:
    case mjSENS_FRAMEQUAT:
    case mjSENS_FRAMEXAXIS:
    case mjSENS_FRAMEYAXIS:
    case mjSENS_FRAMEZAXIS:
    case mjSENS_SUBTREECOM:
    case mjSENS_INSIDESITE:
    case mjSENS_GEOMDIST:
    case mjSENS_GEOMNORMAL:
    case mjSENS_GEOMFROMTO:
    case mjSENS_E_POTENTIAL:
    case mjSENS_E_KINETIC:
    case mjSENS_CLOCK:
    case mjSENS_PLUGIN:
    case mjSENS_USER:
      return mjSTAGE_POS;
  }

  return mjSTAGE_POS;  // all cases are covered but GCC is extra persnickety
}

// scale the cutoff and noise by the dimension of the output, in the gauge of the sensorized object,
// or back; the reference only chooses coordinates, except for distances which average both gauges
void mjCSensor::ApplyGauge(bool inverse) {
  if (!obj) { return; }

  // gauge of the object, and the factor of its coordinate
  mjCGauge g;
  int      q  = 0;
  double   fq = 1;
  switch (objtype) {
    case mjOBJ_JOINT: {
      const mjCJoint* joint = static_cast<const mjCJoint*>(obj);
      g                     = joint->Gauge();
      q                     = joint->spec.type == mjJNT_SLIDE;
      fq                    = q ? g.Stretch(joint->spec.axis) : 1;
      break;
    }
    case mjOBJ_TENDON:
      g  = static_cast<const mjCTendon*>(obj)->PathGauge();
      q  = static_cast<const mjCTendon*>(obj)->Linear();
      fq = q ? g.Shape(1) : 1;
      break;
    case mjOBJ_ACTUATOR: {
      const mjCActuator* actuator = static_cast<const mjCActuator*>(obj);
      g                           = actuator->gauge_;
      q                           = actuator->linear_;
      fq                          = actuator->CoordinateFactor(g);
      break;
    }
    case mjOBJ_BODY:
    case mjOBJ_XBODY:
      g = obj->ContentGauge();
      break;
    default:
      g = obj->Gauge();
  }
  if ((type == mjSENS_GEOMDIST ||
       type == mjSENS_GEOMNORMAL ||
       type == mjSENS_GEOMFROMTO ||
       (type == mjSENS_INSIDESITE && intprm[0])) &&
      ref) {
    bool body = reftype == mjOBJ_BODY || reftype == mjOBJ_XBODY;
    g         = mjCGauge::Mean(g, body ? ref->ContentGauge() : ref->Gauge());
  }
  if (type == mjSENS_TACTILE && ref) { g = ref->Gauge(); }
  if (g.IsIdentity()) { return; }
  if (type == mjSENS_PLUGIN) { throw mjCError(this, "a plugin sensor cannot be scaled"); }
  if (type == mjSENS_TACTILE) { throw mjCError(this, "a tactile sensor cannot be scaled"); }

  // factor of the output
  double f = 1;
  switch (type) {
    case mjSENS_TOUCH:
    case mjSENS_FORCE:
      f = g.Force(1, 1);
      break;
    case mjSENS_TORQUE:
    case mjSENS_SUBTREEANGMOM:
      f = g.Force(2, 1);
      break;
    case mjSENS_ACCELEROMETER:
    case mjSENS_VELOCIMETER:
    case mjSENS_RANGEFINDER:
    case mjSENS_GEOMDIST:
    case mjSENS_GEOMFROMTO:
    case mjSENS_FRAMEPOS:
    case mjSENS_FRAMELINVEL:
    case mjSENS_FRAMELINACC:
    case mjSENS_SUBTREECOM:
    case mjSENS_SUBTREELINVEL:
      f = g.Shape(1);
      break;
    case mjSENS_INSIDESITE:
      if (intprm[0]) { f = g.Shape(1); }
      break;
    case mjSENS_JOINTPOS:
    case mjSENS_JOINTVEL:
    case mjSENS_TENDONPOS:
    case mjSENS_TENDONVEL:
    case mjSENS_ACTUATORPOS:
    case mjSENS_ACTUATORVEL:
    case mjSENS_JOINTLIMITPOS:
    case mjSENS_JOINTLIMITVEL:
    case mjSENS_TENDONLIMITPOS:
    case mjSENS_TENDONLIMITVEL:
      f = fq;
      break;
    case mjSENS_ACTUATORFRC:
    case mjSENS_JOINTACTFRC:
    case mjSENS_TENDONACTFRC:
    case mjSENS_JOINTLIMITFRC:
    case mjSENS_TENDONLIMITFRC:
      f = g.Force(2 - q, 1);
      break;
    default:
      break;
  }
  double f_cutoff = (type == mjSENS_GEOMNORMAL) ? g.Shape(1) : f;
  if (inverse) {
    f        = 1 / f;
    f_cutoff = 1 / f_cutoff;
  }
  cutoff *= f_cutoff;
  noise  *= f;
}


// compiler
void mjCSensor::Compile(void) {
  CopyFromSpec();

  // resize userdata
  if (userdata_.size() > model->nuser_sensor) {
    throw mjCError(this, "user has more values than nuser_sensor in sensor");
  }
  userdata_.resize(model->nuser_sensor);

  // require non-negative noise
  if (noise < 0) { throw mjCError(this, "negative noise in sensor"); }

  // require non-negative cutoff
  if (cutoff < 0) { throw mjCError(this, "negative cutoff in sensor"); }

  // require non-negative interval
  if (interval[0] < 0) { throw mjCError(this, "negative interval in sensor"); }

  // require non-positive phase
  if (interval[1] > 0) { throw mjCError(this, "positive phase in sensor"); }

  // require phase > -period (values outside this are equivalent modulo period)
  if (interval[0] > 0 && interval[1] <= -interval[0]) {
    throw mjCError(this, "phase must be greater than -period in sensor");
  }

  // require nsample for delay
  if (delay > 0 && nsample <= 0) {
    throw mjCError(this, "setting delay > 0 without a history buffer");
  }

  // validate nsample size (max 2^24)
  if (nsample > 16777216) {
    throw mjCError(this, "at most 2^24 samples in sensor history buffer, got %d", nullptr, nsample);
  }

  // Find referenced object
  ResolveReferences(model);
  ApplyGauge();

  // set datatype for non-user sensors
  if (type != mjSENS_USER) { datatype = sensorDatatype(type); }

  // set needstage for non-user and non-plugin sensors
  if (type != mjSENS_USER && type != mjSENS_PLUGIN) { needstage = sensorNeedstage(type); }

  // process according to sensor type
  switch (type) {
    case mjSENS_TOUCH:
    case mjSENS_ACCELEROMETER:
    case mjSENS_VELOCIMETER:
    case mjSENS_GYRO:
    case mjSENS_FORCE:
    case mjSENS_TORQUE:
    case mjSENS_MAGNETOMETER:
    case mjSENS_CAMPROJECTION:
      // must be attached to site
      if (objtype != mjOBJ_SITE) { throw mjCError(this, "sensor must be attached to site"); }

      // check for camera resolution for camera projection sensor
      if (type == mjSENS_CAMPROJECTION) {
        if (reftype != mjOBJ_CAMERA || !ref) {
          throw mjCError(this, "camera projection sensor requires camera reference");
        }
        mjCCamera* camref = (mjCCamera*)ref;
        if (camref->resolution[0] <= 0 || camref->resolution[1] <= 0) {
          throw mjCError(this, "camera projection sensor requires camera resolution");
        }
      }
      break;

    case mjSENS_RANGEFINDER: {
      // must be attached to site or camera
      if (objtype != mjOBJ_SITE && objtype != mjOBJ_CAMERA) {
        throw mjCError(this, "sensor must be attached to site or camera");
      }
      if (objtype == mjOBJ_CAMERA) {
        mjCCamera* cam = (mjCCamera*)obj;
        if (cam->resolution[0] <= 0 || cam->resolution[1] <= 0) {
          throw mjCError(this, "rangefinder camera sensor requires positive camera resolution");
        }
      }

      // check for dataspec correctness
      int dataspec = intprm[0];
      if (dataspec <= 0) {
        throw mjCError(this, "data spec (intprm[0]) must be positive, got %d", nullptr, dataspec);
      }
      int mask = (1 << mjNRAYDATA) - 1;
      if (!(dataspec & mask)) {
        throw mjCError(this,
                       "data spec intprm[0]=%d must have at least one bit set of the first "
                       "mjNRAYDATA bits",
                       nullptr,
                       dataspec);
      }
      if (dataspec & ~mask) {
        throw mjCError(this,
                       "data spec intprm[0]=%d has bits set beyond the first "
                       "mjNRAYDATA bits",
                       nullptr,
                       dataspec);
      }
    } break;

    case mjSENS_JOINTPOS:
    case mjSENS_JOINTVEL:
    case mjSENS_JOINTACTFRC:
      // must be attached to joint
      if (objtype != mjOBJ_JOINT) { throw mjCError(this, "sensor must be attached to joint"); }

      // make sure joint is slide or hinge
      if (((mjCJoint*)obj)->type != mjJNT_SLIDE && ((mjCJoint*)obj)->type != mjJNT_HINGE) {
        throw mjCError(this, "joint must be slide or hinge in sensor");
      }
      break;

    case mjSENS_TENDONACTFRC:
      // must be attached to tendon
      if (objtype != mjOBJ_TENDON) { throw mjCError(this, "sensor must be attached to tendon"); }
      break;

    case mjSENS_TENDONPOS:
    case mjSENS_TENDONVEL:
      // must be attached to tendon
      if (objtype != mjOBJ_TENDON) { throw mjCError(this, "sensor must be attached to tendon"); }
      break;

    case mjSENS_ACTUATORPOS:
    case mjSENS_ACTUATORVEL:
    case mjSENS_ACTUATORFRC:
      // must be attached to actuator
      if (objtype != mjOBJ_ACTUATOR) {
        throw mjCError(this, "sensor must be attached to actuator");
      }
      break;

    case mjSENS_BALLQUAT:
    case mjSENS_BALLANGVEL:
      // must be attached to joint
      if (objtype != mjOBJ_JOINT) { throw mjCError(this, "sensor must be attached to joint"); }

      // make sure joint is ball
      if (((mjCJoint*)obj)->type != mjJNT_BALL) {
        throw mjCError(this, "joint must be ball in sensor");
      }
      break;

    case mjSENS_JOINTLIMITPOS:
    case mjSENS_JOINTLIMITVEL:
    case mjSENS_JOINTLIMITFRC:
      // must be attached to joint
      if (objtype != mjOBJ_JOINT) { throw mjCError(this, "sensor must be attached to joint"); }

      // make sure joint has limit
      if (!((mjCJoint*)obj)->is_limited()) {
        throw mjCError(this, "joint must be limited in sensor");
      }
      break;

    case mjSENS_TENDONLIMITPOS:
    case mjSENS_TENDONLIMITVEL:
    case mjSENS_TENDONLIMITFRC:
      // must be attached to tendon
      if (objtype != mjOBJ_TENDON) { throw mjCError(this, "sensor must be attached to tendon"); }

      // make sure tendon has limit
      if (!((mjCTendon*)obj)->is_limited()) {
        throw mjCError(this, "tendon must be limited in sensor");
      }
      break;

    case mjSENS_FRAMEPOS:
    case mjSENS_FRAMEQUAT:
    case mjSENS_FRAMEXAXIS:
    case mjSENS_FRAMEYAXIS:
    case mjSENS_FRAMEZAXIS:
    case mjSENS_FRAMELINVEL:
    case mjSENS_FRAMEANGVEL:
    case mjSENS_FRAMELINACC:
    case mjSENS_FRAMEANGACC:
      // must be attached to object with spatial frame
      if (objtype != mjOBJ_BODY &&
          objtype != mjOBJ_XBODY &&
          objtype != mjOBJ_GEOM &&
          objtype != mjOBJ_SITE &&
          objtype != mjOBJ_CAMERA) {
        throw mjCError(this, "sensor must be attached to (x)body, geom, site or camera");
      }
      break;

    case mjSENS_SUBTREECOM:
    case mjSENS_SUBTREELINVEL:
    case mjSENS_SUBTREEANGMOM:
      // must be attached to body
      if (objtype != mjOBJ_BODY) { throw mjCError(this, "sensor must be attached to body"); }
      break;

    case mjSENS_INSIDESITE:
      if (objtype != mjOBJ_BODY &&
          objtype != mjOBJ_XBODY &&
          objtype != mjOBJ_GEOM &&
          objtype != mjOBJ_SITE &&
          objtype != mjOBJ_CAMERA) {
        throw mjCError(this, "sensor must be attached to (x)body, geom, site or camera");
      }
      if (reftype != mjOBJ_SITE) { throw mjCError(this, "sensor must be associated with a site"); }
      if (intprm[0]) {
        if (objtype == mjOBJ_CAMERA) {
          throw mjCError(this, "camera is not supported in enclosed insidesite sensor");
        }
        datatype  = mjDATATYPE_REAL;
        int stype = static_cast<mjCSite*>(ref)->spec.type;
        if (stype < mjGEOM_SPHERE || stype > mjGEOM_MESH) {
          throw mjCError(this,
                         "site '%s' in enclosed insidesite sensor must be a compact convex shape",
                         ref->name.c_str());
        }
        if (objtype == mjOBJ_GEOM) {
          int gtype = static_cast<mjCGeom*>(obj)->Type();
          if (gtype < mjGEOM_SPHERE || gtype > mjGEOM_MESH) {
            throw mjCError(this,
                           "geom '%s' in enclosed insidesite sensor must be a compact convex shape",
                           obj->name.c_str());
          }
        } else if (objtype == mjOBJ_SITE) {
          int o_stype = static_cast<mjCSite*>(obj)->spec.type;
          if (o_stype < mjGEOM_SPHERE || o_stype > mjGEOM_MESH) {
            throw mjCError(this,
                           "site '%s' in enclosed insidesite sensor must be a compact convex shape",
                           obj->name.c_str());
          }
        } else if (objtype == mjOBJ_BODY || objtype == mjOBJ_XBODY) {
          std::vector<const mjCBody*> bodies = {static_cast<const mjCBody*>(obj)};
          int                         ngeom  = 0;
          for (size_t i = 0; i < bodies.size(); ++i) {
            for (const mjCGeom* geom : bodies[i]->GetList<mjCGeom>()) {
              ngeom++;
              if (geom->Type() < mjGEOM_SPHERE || geom->Type() > mjGEOM_MESH) {
                throw mjCError(
                    this,
                    "geom '%s' in enclosed insidesite sensor must be a compact convex shape",
                    geom->name.c_str());
              }
            }
            if (objtype == mjOBJ_XBODY) {
              for (const mjCBody* child : bodies[i]->Bodies()) { bodies.push_back(child); }
            }
          }
          if (ngeom == 0) {
            throw mjCError(this,
                           "body '%s' in enclosed insidesite sensor must have at least one geom",
                           obj->name.c_str());
          }
        }
      }
      break;

    case mjSENS_GEOMDIST:
    case mjSENS_GEOMNORMAL:
    case mjSENS_GEOMFROMTO:
      // must be attached to body or geom
      if ((objtype != mjOBJ_BODY && objtype != mjOBJ_GEOM) ||
          (reftype != mjOBJ_BODY && reftype != mjOBJ_GEOM)) {
        throw mjCError(this, "sensor must be attached to body or geom");
      }

      // objects must be different
      if (objtype == reftype && obj == ref) {
        throw mjCError(this, "1st body/geom must be different from 2nd body/geom");
      }

      // height fields are not necessarily convex and are not yet supported
      if ((objtype == mjOBJ_GEOM && static_cast<mjCGeom*>(obj)->Type() == mjGEOM_HFIELD) ||
          (reftype == mjOBJ_GEOM && static_cast<mjCGeom*>(ref)->Type() == mjGEOM_HFIELD)) {
        throw mjCError(this, "height fields are not supported in geom distance sensors");
      }
      break;

    case mjSENS_CONTACT: {
      // check first matching criterion
      if (objtype != mjOBJ_SITE &&
          objtype != mjOBJ_BODY &&
          objtype != mjOBJ_XBODY &&
          objtype != mjOBJ_GEOM &&
          objtype != mjOBJ_UNKNOWN) {
        throw mjCError(this, "first matching criterion: if set, must be (x)body, geom or site");
      }

      // check second matching criterion
      if (reftype != mjOBJ_BODY &&
          reftype != mjOBJ_XBODY &&
          reftype != mjOBJ_GEOM &&
          reftype != mjOBJ_UNKNOWN) {
        throw mjCError(this, "second matching criterion: if set, must be (x)body or geom");
      }

      // check for dataspec correctness
      int dataspec = intprm[0];
      if (dataspec <= 0) {
        throw mjCError(this, "data spec (intprm[0]) must be positive, got %d", nullptr, dataspec);
      }
      int mask = (1 << mjNCONDATA) - 1;
      if (!(dataspec & mask)) {
        throw mjCError(this,
                       "data spec intprm[0]=%d must have at least one bit set of the first "
                       "mjNCONDATA bits",
                       nullptr,
                       dataspec);
      }
      if (dataspec & ~mask) {
        throw mjCError(this,
                       "data spec intprm[0]=%d has bits set beyond the first "
                       "mjNCONDATA bits",
                       nullptr,
                       dataspec);
      }

      // check for reduce correctness
      int reduce = intprm[1];
      if (reduce < 0 || reduce > 3) {
        throw mjCError(this,
                       "unknown reduction criterion. got %d, "
                       "expected one of {0, 1, 2, 3}",
                       nullptr,
                       reduce);
      }

      // check for non-positive num
      if (intprm[2] <= 0) {
        throw mjCError(this,
                       "num (intprm[2]) must be positive in sensor, got %d",
                       nullptr,
                       intprm[2]);
      }

      // set needstage from requested data and reduction criterion
      bool need_acc = (dataspec & (1 << mjCONDATA_FORCE)) ||
                      (dataspec & (1 << mjCONDATA_TORQUE)) ||
                      reduce == 2 ||
                      reduce == 3;
      bool need_vel = (dataspec & (1 << mjCONDATA_LINVEL)) || (dataspec & (1 << mjCONDATA_ANGVEL));
      if (reduce == 3 && need_vel) {
        throw mjCError(this, "netforce reduction is incompatible with linvel or angvel");
      }
      needstage = need_acc ? mjSTAGE_ACC : (need_vel ? mjSTAGE_VEL : mjSTAGE_POS);
    } break;

    case mjSENS_E_POTENTIAL:
    case mjSENS_E_KINETIC:
    case mjSENS_CLOCK:
      break;

    case mjSENS_USER:
      // check for negative dim
      if (dim < 0) { throw mjCError(this, "sensor dim must be non-negative in sensor"); }

      // make sure dim is consistent with datatype
      if (datatype == mjDATATYPE_AXIS && dim != 3) {
        throw mjCError(this, "datatype AXIS requires dim=3 in sensor");
      }
      if (datatype == mjDATATYPE_QUATERNION && dim != 4) {
        throw mjCError(this, "datatype QUATERNION requires dim=4 in sensor");
      }
      break;

    case mjSENS_TACTILE:
      if (objtype != mjOBJ_MESH) { throw mjCError(this, "sensor must be associated with a mesh"); }
      if (reftype != mjOBJ_GEOM) { throw mjCError(this, "sensor must be associated with a geom"); }
      break;

    case mjSENS_PLUGIN:
      if (plugin_name.empty() && plugin_instance_name.empty()) {
        throw mjCError(this, "neither 'plugin' nor 'instance' is specified for sensor");
      }

      // resolve plugin instance, or create one if using the "plugin" attribute shortcut
      {
        mjCPlugin* plugin_instance = static_cast<mjCPlugin*>(plugin.element);
        model->ResolvePlugin(this, plugin_name, plugin_instance_name, &plugin_instance);
        plugin.element           = plugin_instance;
        const mjpPlugin* pplugin = mjp_getPluginAtSlot(plugin_instance->plugin_slot);
        if (!(pplugin->capabilityflags & mjPLUGIN_SENSOR)) {
          throw mjCError(this, "plugin '%s' does not support sensors", pplugin->name);
        }
        needstage = static_cast<mjtStage>(pplugin->needstage);
      }

      break;

    default:
      throw mjCError(this, "invalid type in sensor '%s' (id = %d)", name.c_str(), id);
  }

  dim = mjs_sensorDim(this);

  // actuator sensors report one value per force output
  if (type == mjSENS_ACTUATORPOS || type == mjSENS_ACTUATORVEL || type == mjSENS_ACTUATORFRC) {
    dim = ((mjCActuator*)obj)->outnum_;
  }

  // check cutoff for incompatible data types
  if (cutoff > 0 && (datatype == mjDATATYPE_QUATERNION ||
                     (datatype == mjDATATYPE_AXIS && type != mjSENS_GEOMNORMAL))) {
    throw mjCError(this, "cutoff applied to axis or quaternion datatype in sensor");
  }
}


//------------------ class mjCNumeric implementation -----------------------------------------------

// constructor
mjCNumeric::mjCNumeric(mjCModel* _model) {
  mjs_defaultNumeric(&spec);
  elemtype = mjOBJ_NUMERIC;

  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear variables
  spec_data_.clear();

  // point to local
  PointToLocal();

  // in case this numeric is not compiled
  CopyFromSpec();
}


mjCNumeric::mjCNumeric(const mjCNumeric& other) {
  *this = other;
}


mjCNumeric& mjCNumeric::operator=(const mjCNumeric& other) {
  if (this != &other) {
    this->spec                       = other.spec;
    *static_cast<mjCNumeric_*>(this) = static_cast<const mjCNumeric_&>(other);
    *static_cast<mjsNumeric*>(this)  = static_cast<const mjsNumeric&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCNumeric::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.data    = &spec_data_;
  spec.info    = &info;
  data         = nullptr;
}


void mjCNumeric::CopyFromSpec() {
  *static_cast<mjsNumeric*>(this) = spec;
  data_                           = spec_data_;
}


// destructor
mjCNumeric::~mjCNumeric() {
  spec_data_.clear();
  data_.clear();
}


// compiler
void mjCNumeric::Compile(void) {
  CopyFromSpec();

  // check for size conflict
  if (size && !data_.empty() && size < (int)data_.size()) {
    throw mjCError(this,
                   "numeric '%s' (id = %d): specified size smaller than initialization array",
                   name.c_str(),
                   id);
  }

  // set size if left unspecified
  if (!size) { size = (int)data_.size(); }

  // size cannot be zero
  if (!size) {
    throw mjCError(this, "numeric '%s' (id = %d): size cannot be zero", name.c_str(), id);
  }

  // the model holds the data padded with zeros up to its size
  data_.resize(size);
}


//------------------ class mjCText implementation --------------------------------------------------

// constructor
mjCText::mjCText(mjCModel* _model) {
  mjs_defaultText(&spec);
  elemtype = mjOBJ_TEXT;

  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear variables
  spec_data_.clear();

  // point to local
  PointToLocal();

  // in case this text is not compiled
  CopyFromSpec();
}


mjCText::mjCText(const mjCText& other) {
  *this = other;
}


mjCText& mjCText::operator=(const mjCText& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCText_*>(this) = static_cast<const mjCText_&>(other);
    *static_cast<mjsText*>(this)  = static_cast<const mjsText&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCText::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.data    = &spec_data_;
  spec.info    = &info;
  data         = nullptr;
}


void mjCText::CopyFromSpec() {
  *static_cast<mjsText*>(this) = spec;
  data_                        = spec_data_;
}


// destructor
mjCText::~mjCText() {
  data_.clear();
  spec_data_.clear();
}


// compiler
void mjCText::Compile(void) {
  CopyFromSpec();

  // size cannot be zero
  if (data_.empty()) {
    throw mjCError(this, "text '%s' (id = %d): size cannot be zero", name.c_str(), id);
  }
}


//------------------ class mjCTuple implementation -------------------------------------------------

// constructor
mjCTuple::mjCTuple(mjCModel* _model) {
  mjs_defaultTuple(&spec);
  elemtype = mjOBJ_TUPLE;

  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear variables
  spec_objtype_.clear();
  spec_objname_.clear();
  spec_objprm_.clear();
  obj.clear();

  // point to local
  PointToLocal();

  // in case this tuple is not compiled
  CopyFromSpec();
}


mjCTuple::mjCTuple(const mjCTuple& other) {
  *this = other;
}


mjCTuple& mjCTuple::operator=(const mjCTuple& other) {
  if (this != &other) {
    this->spec                     = other.spec;
    *static_cast<mjCTuple_*>(this) = static_cast<const mjCTuple_&>(other);
    *static_cast<mjsTuple*>(this)  = static_cast<const mjsTuple&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCTuple::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.objtype = (mjIntVec*)&spec_objtype_;
  spec.objname = &spec_objname_;
  spec.objprm  = &spec_objprm_;
  spec.info    = &info;
  objname      = nullptr;
  objprm       = nullptr;
}


void mjCTuple::NameSpace(const mjCModel* m) {
  if (!name.empty()) { name = m->prefix + name + m->suffix; }
  for (int i = 0; i < spec_objname_.size(); i++) {
    // the world body keeps its name, and so do assets, which are not copied for self-attach
    mjtObj type  = i < (int)spec_objtype_.size() ? (mjtObj)spec_objtype_[i] : mjOBJ_UNKNOWN;
    bool   world = (type == mjOBJ_BODY || type == mjOBJ_XBODY) && spec_objname_[i] == "world";
    bool   asset = type == mjOBJ_MESH ||
                   type == mjOBJ_HFIELD ||
                   type == mjOBJ_TEXTURE ||
                   type == mjOBJ_MATERIAL ||
                   type == mjOBJ_NUMERIC ||
                   type == mjOBJ_TEXT;
    if (world || (asset && model == m)) { continue; }
    spec_objname_[i] = m->prefix + spec_objname_[i] + m->suffix;
  }
}


void mjCTuple::CopyFromSpec() {
  *static_cast<mjsTuple*>(this) = spec;

  objtype_ = spec_objtype_;
  objname_ = spec_objname_;
  objprm_  = spec_objprm_;
  objtype  = (mjIntVec*)&objtype_;
}


// destructor
mjCTuple::~mjCTuple() {
  objtype_.clear();
  objname_.clear();
  objprm_.clear();
  spec_objtype_.clear();
  spec_objname_.clear();
  spec_objprm_.clear();
  obj.clear();
}


void mjCTuple::ResolveReferences(const mjCModel* m) {
  // check for empty tuple
  if (objtype_.empty()) { throw mjCError(this, "tuple '%s' (id = %d) is empty", name.c_str(), id); }

  // check for size conflict
  if (objtype_.size() != objname_.size() || objtype_.size() != objprm_.size()) {
    throw mjCError(this,
                   "tuple '%s' (id = %d) has object arrays with different sizes",
                   name.c_str(),
                   id);
  }

  // resize objid to correct size
  obj.resize(objtype_.size());

  // find objects, fill in ids
  for (int i = 0; i < objtype_.size(); i++) {
    // find object by type and name; a tuple which is being attached is not yet in the name map of
    // the model, nor are those attached with it, and it can reference itself
    mjCBase* res = m->FindObject(objtype_[i], objname_[i]);
    if (!res && objtype_[i] == mjOBJ_TUPLE) {
      res = objname_[i] == name ? this : m->SearchObject(mjOBJ_TUPLE, objname_[i]);
    }
    if (!res) {
      throw mjCError(this, "unrecognized object '%s' in tuple %d", objname_[i].c_str(), id);
    }

    // assign id
    obj[i] = res;
  }
}


// compiler
void mjCTuple::Compile(void) {
  CopyFromSpec();
  ResolveReferences(model);
}


//------------------ class mjCKey implementation ---------------------------------------------------

// constructor
mjCKey::mjCKey(mjCModel* _model) {
  mjs_defaultKey(&spec);
  elemtype = mjOBJ_KEY;

  // set model pointer
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear variables
  spec_qpos_.clear();
  spec_qvel_.clear();
  spec_act_.clear();
  spec_mpos_.clear();
  spec_mquat_.clear();
  spec_ctrl_.clear();
  spec_plastic_.clear();

  // point to local
  PointToLocal();

  // in case this keyframe is not compiled
  CopyFromSpec();
}


mjCKey::mjCKey(const mjCKey& other) {
  *this = other;
}


mjCKey& mjCKey::operator=(const mjCKey& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCKey_*>(this) = static_cast<const mjCKey_&>(other);
    *static_cast<mjsKey*>(this)  = static_cast<const mjsKey&>(other);
  }
  PointToLocal();
  return *this;
}


void mjCKey::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.qpos    = &spec_qpos_;
  spec.qvel    = &spec_qvel_;
  spec.act     = &spec_act_;
  spec.mpos    = &spec_mpos_;
  spec.mquat   = &spec_mquat_;
  spec.ctrl    = &spec_ctrl_;
  spec.plastic = &spec_plastic_;
  spec.info    = &info;
  qpos         = nullptr;
  qvel         = nullptr;
  act          = nullptr;
  mpos         = nullptr;
  mquat        = nullptr;
  ctrl         = nullptr;
  plastic      = nullptr;
}


void mjCKey::CopyFromSpec() {
  *static_cast<mjsKey*>(this) = spec;

  qpos_    = spec_qpos_;
  qvel_    = spec_qvel_;
  act_     = spec_act_;
  mpos_    = spec_mpos_;
  mquat_   = spec_mquat_;
  ctrl_    = spec_ctrl_;
  plastic_ = spec_plastic_;
}


// destructor
mjCKey::~mjCKey() {
  qpos_.clear();
  qvel_.clear();
  act_.clear();
  mpos_.clear();
  mquat_.clear();
  ctrl_.clear();
  plastic_.clear();
  spec_qpos_.clear();
  spec_qvel_.clear();
  spec_act_.clear();
  spec_mpos_.clear();
  spec_mquat_.clear();
  spec_ctrl_.clear();
  spec_plastic_.clear();
}


// compiler
void mjCKey::Compile(const mjModel* m) {
  CopyFromSpec();

  // qpos: allocate or check size
  if (qpos_.empty()) {
    qpos_.resize(m->nq);
    for (int i = 0; i < m->nq; i++) { qpos_[i] = (double)m->qpos0[i]; }
  } else if (qpos_.size() != m->nq) {
    throw mjCError(this,
                   "keyframe '%s': invalid qpos size, expected %d, got %d",
                   name.c_str(),
                   m->nq,
                   qpos_.size());
  }

  // qvel: allocate or check size
  if (qvel_.empty()) {
    qvel_.resize(m->nv);
    for (int i = 0; i < m->nv; i++) { qvel_[i] = 0; }
  } else if (qvel_.size() != m->nv) {
    throw mjCError(this,
                   "keyframe '%s': invalid qvel size, expected %d, got %d",
                   name.c_str(),
                   m->nv,
                   qvel_.size());
  }

  // act: allocate or check size
  if (act_.empty()) {
    act_.resize(m->na);
    for (int i = 0; i < m->na; i++) { act_[i] = 0; }
  } else if (act_.size() != m->na) {
    throw mjCError(this,
                   "keyframe '%s': invalid act size, expected %d, got %d",
                   name.c_str(),
                   m->na,
                   act_.size());
  }

  // mpos: allocate or check size
  if (mpos_.empty()) {
    mpos_.resize(3 * m->nmocap);
    if (m->nmocap) {
      for (int i = 0; i < m->nbody; i++) {
        if (m->body_mocapid[i] >= 0) {
          int mocapid            = m->body_mocapid[i];
          mpos_[3 * mocapid]     = m->body_pos[3 * i];
          mpos_[3 * mocapid + 1] = m->body_pos[3 * i + 1];
          mpos_[3 * mocapid + 2] = m->body_pos[3 * i + 2];
        }
      }
    }
  } else if (mpos_.size() != 3 * m->nmocap) {
    throw mjCError(this,
                   "keyframe %d: invalid mpos size, expected length %d",
                   nullptr,
                   id,
                   3 * m->nmocap);
  }

  // mquat: allocate or check size
  if (mquat_.empty()) {
    mquat_.resize(4 * m->nmocap);
    if (m->nmocap) {
      for (int i = 0; i < m->nbody; i++) {
        if (m->body_mocapid[i] >= 0) {
          int mocapid             = m->body_mocapid[i];
          mquat_[4 * mocapid]     = m->body_quat[4 * i];
          mquat_[4 * mocapid + 1] = m->body_quat[4 * i + 1];
          mquat_[4 * mocapid + 2] = m->body_quat[4 * i + 2];
          mquat_[4 * mocapid + 3] = m->body_quat[4 * i + 3];
        }
      }
    }
  } else if (mquat_.size() != 4 * m->nmocap) {
    throw mjCError(this,
                   "keyframe %d: invalid mquat size, expected length %d",
                   nullptr,
                   id,
                   4 * m->nmocap);
  }

  // ctrl: allocate or check size
  if (ctrl_.empty()) {
    ctrl_.resize(m->nu);
    for (int i = 0; i < m->nu; i++) { ctrl_[i] = 0; }
  } else if (ctrl_.size() != m->nu) {
    throw mjCError(this, "keyframe %d: invalid ctrl size, expected length %d", nullptr, id, m->nu);
  }

  // plastic: allocate or check size (default values populated after mj_setCompiledConst)
  if (plastic_.empty()) {
    plastic_.resize(m->nplastic);
    for (int i = 0; i < m->nplastic; i++) { plastic_[i] = 0; }
  } else if (plastic_.size() != m->nplastic) {
    throw mjCError(this,
                   "keyframe %d: invalid plastic size, expected length %d",
                   nullptr,
                   id,
                   m->nplastic);
  }
}


//------------------ class mjCPlugin implementation ------------------------------------------------

// initialize defaults
mjCPlugin::mjCPlugin(mjCModel* _model) {
  name        = "";
  nstate      = -1;
  stateadr_   = -1;
  statenum_   = 0;
  plugin_slot = -1;
  parent      = this;
  model       = _model;
  if (_model) compiler = &_model->spec.compiler;
  name.clear();
  plugin_name.clear();

  // public interface
  mjs_defaultPlugin(&spec);
  elemtype         = mjOBJ_PLUGIN;
  spec.plugin_name = &plugin_name;
  spec.info        = &info;

  PointToLocal();
}


mjCPlugin::mjCPlugin(const mjCPlugin& other) {
  *this = other;
  id    = -1;
}


mjCPlugin& mjCPlugin::operator=(const mjCPlugin& other) {
  if (this != &other) {
    this->spec = other.spec;

    *static_cast<mjCPlugin_*>(this) = static_cast<const mjCPlugin_&>(other);

    stateadr_   = -1;
    statenum_   = 0;
    parent      = this;
    plugin_slot = other.plugin_slot;
  }
  PointToLocal();
  return *this;
}


void mjCPlugin::ResetId() {
  id        = -1;
  stateadr_ = -1;
  statenum_ = 0;
}


void mjCPlugin::PointToLocal() {
  spec.element = static_cast<mjsElement*>(this);
  spec.name    = &name;
  spec.info    = &info;
}


// compiler
void mjCPlugin::Compile(void) {
  mjCPlugin* plugin_instance = this;
  model->ResolvePlugin(this, plugin_name, name, &plugin_instance);
  const mjpPlugin* plugin = mjp_getPluginAtSlot(plugin_slot);

  // clear precompiled
  flattened_attributes.clear();
  std::map<std::string, std::string, std::less<>> config_attribs_copy = config_attribs;

  // concatenate all of the plugin's attribute values (as null-terminated strings) into
  // flattened_attributes, in the order declared in the mjpPlugin
  // each valid attribute found is appended to flattened_attributes and removed from xml_attributes
  for (int i = 0; i < plugin->nattribute; ++i) {
    std::string_view attr(plugin->attributes[i]);
    auto             it = config_attribs_copy.find(attr);
    if (it == config_attribs_copy.end()) {
      flattened_attributes.push_back('\0');
    } else {
      auto original_size = flattened_attributes.size();
      flattened_attributes.resize(original_size + it->second.size() + 1);
      std::memcpy(&flattened_attributes[original_size], it->second.c_str(), it->second.size() + 1);
      config_attribs_copy.erase(it);
    }
  }

  // if there are no attributes, add a null terminator
  if (plugin->nattribute == 0) { flattened_attributes.push_back('\0'); }

  // anything left in xml_attributes at this stage is not a valid attribute
  if (!config_attribs_copy.empty()) {
    std::string error = "unrecognized attribute 'plugin:" +
                        config_attribs_copy.begin()->first +
                        "' for plugin " +
                        std::string(plugin->name) +
                        "'";
    throw mjCError(parent, "%s", error.c_str());
  }
}
