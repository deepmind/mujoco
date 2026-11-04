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

#ifndef MUJOCO_SRC_USER_USER_OBJECTS_H_
#define MUJOCO_SRC_USER_USER_OBJECTS_H_

#include <stdbool.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <array>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include "user/user_cache.h"
#include "user/user_util.h"

// forward declarations of all mjC/X classes
class mjCError;
class mjCBase;
class mjCBody;
class mjCFrame;
class mjCJoint;
class mjCGeom;
class mjCSite;
class mjCCamera;
class mjCLight;
class mjCHField;
class mjCFlex;
class mjCMesh;
class mjCSkin;
class mjCTexture;
class mjCMaterial;
class mjCPair;
class mjCBodyPair;
class mjCEquality;
class mjCTendon;
class mjCWrap;
class mjCActuator;
class mjCSensor;
class mjCNumeric;
class mjCText;
class mjCTuple;
class mjCDef;
class mjCModel;   // defined in user_model.h
class mjXWriter;  // defined in xml_native.h
class mjXURDF;    // defined in xml_urdf.h


//------------------------- helper constants, classes and functions --------------------------------

// number of positive size parameters for each geom type
const int mjGEOMINFO[mjNGEOMTYPES] = {3, 0, 1, 2, 3, 2, 3, 0};

// error information
class [[nodiscard]] mjCError {
 public:
  mjCError(
      const mjCBase* obj = 0, const char* msg = 0, const char* str = 0, int pos1 = 0, int pos2 = 0);

  char message[500];  // error message
};

// alternative specifications of frame orientation
const char* ResolveOrientation(double*               quat,      // set frame quat
                               bool                  degree,    // angle format: degree/radian
                               const char*           sequence,  // euler sequence format: "xyz"
                               const mjsOrientation& orient);


//------------------------- class mjCBoundingVolumeHierarchy ---------------------------------------

// bounding volume
class mjCBoundingVolume {
 public:
  mjCBoundingVolume(int          id,
                    int          contype,
                    int          conaffinity,
                    const double pos[3],
                    const double quat[4],
                    const double aabb[6])
      : contype_(contype), conaffinity_(conaffinity), idval_(id) {
    std::copy(pos, pos + 3, pos_.begin());
    std::copy(aabb, aabb + 6, aabb_.begin());
    quat_set_ = quat != nullptr;
    if (quat_set_) { std::copy(quat, quat + 4, quat_.begin()); }
  }

  mjCBoundingVolume(const int*   id,
                    int          contype,
                    int          conaffinity,
                    const double pos[3],
                    const double quat[4],
                    const double aabb[6])
      : contype_(contype), conaffinity_(conaffinity), id_(id) {
    std::copy(pos, pos + 3, pos_.begin());
    std::copy(aabb, aabb + 6, aabb_.begin());
    quat_set_ = quat != nullptr;
    if (quat_set_) { std::copy(quat, quat + 4, quat_.begin()); }
  }

  int           Contype() const { return contype_; }
  int           Conaffinity() const { return conaffinity_; }
  const double* AABB() const { return aabb_.data(); }
  double        AABB(int i) const { return aabb_[i]; }
  const double* Pos() const { return pos_.data(); }
  double        Pos(int i) const { return pos_[i]; }
  const double* Quat() const { return quat_set_ ? quat_.data() : nullptr; }
  const int*    Id() const { return id_ ? id_ : &idval_; }

  void SetContype(int val) { contype_ = val; }
  void SetConaffinity(int val) { conaffinity_ = val; }
  void SetAABB(const double* aabb) { std::copy(aabb, aabb + 6, aabb_.begin()); }
  void SetPos(const double* pos) { std::copy(pos, pos + 3, pos_.begin()); }
  void SetQuat(const double* quat) {
    quat_set_ = true;
    std::copy(quat, quat + 4, quat_.begin());
  }
  void SetId(const int* id) { id_ = id; }
  void SetId(int val) { idval_ = val; }

 private:
  int                   contype_;      // contact type
  int                   conaffinity_;  // contact affinity
  std::array<double, 6> aabb_;         // axis-aligned bounding box (center, size)
  std::array<double, 3> pos_;          // position (set by user or Compiler)
  std::array<double, 4> quat_;         // orientation (set by user or Compiler)
  bool                  quat_set_;     // boolean flag is quat_ has been set
  int                   idval_;  // local id copy for nodes not storing their id's (e.g. faces)

  // pointer to object id
  const int* id_ = nullptr;
};


// bounding volume hierarchy
struct mjCBoundingVolumeHierarchy_ {
 protected:
  int                 nbvh_ = 0;
  std::vector<mjtNum> bvh_;        // bounding boxes                          (nbvh x 6)
  std::vector<int>    child_;      // children of each node                   (nbvh x 2)
  std::vector<int>    nodeid_;     // id of elem contained by the node        (nbvh x 1)
  std::vector<int*>   nodeidptr_;  // ptr to id of elem contained by the node (nbvh x 1)
  std::vector<int>    level_;      // levels of each node                     (nbvh x 1)

  std::vector<mjCBoundingVolume> bvleaf_;
  std::string                    name_;
  double                         ipos_[3]  = {0, 0, 0};
  double                         iquat_[4] = {1, 0, 0, 0};
};

// angle-weighted vertex and edge pseudonormals of a triangle mesh, which give the sign of the
// distance to a closest point on a shared vertex or edge (Baerentzen and Aanaes, 2005)
struct MeshPseudonormals {
  MeshPseudonormals(const double* vert, int nvert, const int* face, int nface);

  std::vector<double> vertex;  // vertex pseudonormals                 (nvert x 3)
  std::vector<double> edge;    // edge pseudonormals                   (nedge x 3)
  std::vector<int>    side;    // edge index of each side of each face (nface x 3)
};

class mjCBoundingVolumeHierarchy : public mjCBoundingVolumeHierarchy_ {
 public:
  // make bounding volume hierarchy
  void                     CreateBVH(mjCModel* model, const mjCBase* owner);
  void                     Set(double ipos_element[3], double iquat_element[4]);
  void                     AllocateBoundingVolumes(int nleaf);
  void                     RemoveInactiveVolumes(int nmax);
  const mjCBoundingVolume* AddBoundingVolume(int          id,
                                             int          contype,
                                             int          conaffinity,
                                             const double pos[3],
                                             const double quat[4],
                                             const double aabb[6]);
  const mjCBoundingVolume* AddBoundingVolume(const int*   id,
                                             int          contype,
                                             int          conaffinity,
                                             const double pos[3],
                                             const double quat[4],
                                             const double aabb[6]);

  // public accessors
  int                        Nbvh() const { return nbvh_; }
  const std::vector<mjtNum>& Bvh() const { return bvh_; }
  const std::vector<int>&    Child() const { return child_; }
  const std::vector<int>&    Nodeid() const { return nodeid_; }
  int                        Nodeid(int id) const { return nodeid_[id]; }
  const int*                 Nodeidptr(int id) const { return nodeidptr_[id]; }
  const std::vector<int>&    Level() const { return level_; }
  int                        Size() const {
    return sizeof(mjCBoundingVolume) * bvleaf_.size() +
           sizeof(mjtNum) * bvh_.size() +
           sizeof(int) * child_.size() +
           sizeof(int) * nodeid_.size() +
           sizeof(int) * level_.size();
  }

  // query signed distance from point to mesh surface
  double QuerySignedDistance(const double*            point,
                             const double*            vert,
                             const int*               face,
                             const MeshPseudonormals& normals) const;

 private:
  // internal class used during BVH construction, for partial sorting of bounding volumes
  struct BVElement {
    const mjCBoundingVolume* e;
    // position of the element in the BVH axes
    double lpos[3];
  };
  void Make(std::vector<BVElement>& elements);
  int  MakeBVH(std::vector<BVElement>::iterator elements_begin,
               std::vector<BVElement>::iterator elements_end,
               int                              lev,
               mjCModel*                        model,
               const mjCBase*                   owner);
};


//------------------------- class mjCOctree --------------------------------------------------------

struct Point {
  std::array<double, 3> p;

  double&       operator[](size_t i) { return p[i]; }
  const double& operator[](size_t i) const { return p[i]; }

  bool operator==(const Point& other) const {
    constexpr double kEpsilon = 1e-9;
    return std::abs(this->p[0] - other.p[0]) < kEpsilon &&
           std::abs(this->p[1] - other.p[1]) < kEpsilon &&
           std::abs(this->p[2] - other.p[2]) < kEpsilon;
  }
};

namespace std {
template <>
struct hash<Point> {
  size_t operator()(const Point& pt) const {
    size_t h1 = hash<double>()(pt.p[0]);
    size_t h2 = hash<double>()(pt.p[1]);
    size_t h3 = hash<double>()(pt.p[2]);
    // combine hashes
    size_t seed  = h1;
    seed        ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed        ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
}  // namespace std

typedef std::array<std::array<double, 3>, 3> Triangle;

struct OctNode {
  int level        = 0;   // level of the node
  int parent_index = -1;  // index of the parent node
  int child_slot   = -1;  // slot of the child node in the parent node

  std::array<int, 8>    child  = {-1};  // children nodes
  std::array<int, 8>    vertid = {-1};  // vertex id's
  std::array<double, 6> aamm   = {0};   // bounding box
  std::array<double, 8> coeff  = {0};   // interpolation coefficients
};

struct OctreeTask {
  std::vector<Triangle*> elements;
  int                    lev;
  int                    parent_index;
  int                    child_slot;
  int                    node_index;
};

struct mjCOctree_ {
  int nnode_     = 0;
  int nvert_     = 0;
  int max_depth_ = 6;  // max octree depth (default 6)

  std::vector<OctNode>  node_;
  std::vector<Triangle> face_;  // mesh faces                (nmeshface x 3)
  std::vector<Point>    vert_;  // octree vertices           (nvert x 3)

  std::vector<std::vector<int>> hang_;  // hanging nodes status      (nvert x 1)
  double                        ipos_[3]  = {0, 0, 0};
  double                        iquat_[4] = {1, 0, 0, 0};
  int smoothing_iterations_               = 0;  // Laplacian smoothing iterations (0 = disabled)
};

class mjCOctree : public mjCOctree_ {
 public:
  void CreateOctree(const double aamm[6]);

  int                       NumNodes() const { return nnode_; }
  int                       NumVerts() const { return nvert_; }
  void                      CopyLevel(int* level) const;
  void                      CopyChild(int* child) const;
  void                      CopyAabb(mjtNum* aabb) const;
  void                      CopyCoeff(mjtNum* coeff) const;
  const double*             Vert(int i) const { return vert_[i].p.data(); }
  const std::vector<int>&   Hang(int i) const { return hang_[i]; }
  int                       VertId(int n, int v) const { return node_[n].vertid[v]; }
  const std::array<int, 8>& Children(int i) const { return node_[i].child; }
  void                      SetFace(const std::vector<double>& vert, const std::vector<int>& face);
  int                       Size() const {
    return sizeof(OctNode) * node_.size() +
           sizeof(Triangle) * face_.size() +
           sizeof(Point) * vert_.size();
  }
  void Clear() {
    nnode_ = 0;
    nvert_ = 0;
    node_.clear();
    vert_.clear();
    hang_.clear();
  }
  void   AddCoeff(int n, int v, double coeff) { node_[n].coeff[v] = coeff; }
  double Coeff(int n, int v) const { return node_[n].coeff[v]; }

  // Set max octree depth (default 6)
  void SetMaxDepth(int depth) { max_depth_ = depth; }
  int  MaxDepth() const { return max_depth_; }

  // Set number of Laplacian smoothing iterations (0 = disabled, default)
  void SetSmoothingIterations(int iterations) { smoothing_iterations_ = iterations; }
  int  SmoothingIterations() const { return smoothing_iterations_; }

  // compute SDF coefficients via BVH queries, optionally with Laplacian smoothing
  void ComputeSdfCoeffs(const double*                     vert,
                        int                               nvert,
                        const int*                        face,
                        int                               nface,
                        const mjCBoundingVolumeHierarchy& tree);

 private:
  void Make(std::vector<Triangle>& elements);
  void MakeOctree(const std::vector<Triangle*>&   elements,
                  const double                    aamm[6],
                  std::unordered_map<Point, int>& vert_map);
  void TaskToNode(const OctreeTask& task, OctNode& node, std::unordered_map<Point, int>& vert_map);
  void Subdivide(const OctreeTask&               task,
                 std::unordered_map<Point, int>& vert_map,
                 std::deque<OctreeTask>*         queue     = nullptr,
                 const std::vector<Triangle*>&   colliding = {});
  int  FindNeighbor(int node_idx, int dir);
  int  FindCoarseNeighbor(int node_idx, int dir);
  void BalanceOctree(std::unordered_map<Point, int>& vert_map);
  void MarkHangingNodes();
};


//------------------------- class mjCBase ----------------------------------------------------------
// Generic functionality for all derived classes

// accumulated scale of an element, composed along its ancestry, in the coordinates in which the
// element is posed: the linear map of its shape and the geometric mean of its length factors, the
// length and mass factors of its forces, and the factor of its masses and inertias. A uniform map
// is the identity times the shape factor
struct mjCGauge {
  double shape   = 1;
  double length  = 1;
  double mass    = 1;
  double inertia = 1;
  bool   uniform = true;
  double map[9]  = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  // gauge of a frame scale under the given policy, with the given mass factor (0: ratio of volumes)
  static mjCGauge Of(const double scale[3], mjtScaling scaling, double massscale);

  // factor of a shape-class quantity of dimension L^a M^b, without a direction
  double Shape(double a, double b = 0) const;

  // factor of a force-class quantity of dimension L^a M^b
  double Force(double a, double b) const;

  // scale a position or displacement
  void Point(double x[3]) const;

  // length factor along a direction
  double Stretch(const double dir[3]) const;

  // true if the map keeps a direction, within the alignment tolerance
  bool Aligned(const double dir[3]) const;

  // scale the size of a geom or site of the given type and orientation; returns an error message if
  // its shape cannot represent its scaled self; sites instead stretch along their own axes, and
  // sphere sites take the mean factor
  const char* Size(double size[3], int type, const double quat[4], bool site = false) const;

  // scale a mass and the diagonal or, if defined, the full inertia, given in the frame of iquat; a
  // map which is not uniform gives the full inertia
  void Inertia(double& mass, double inertia[3], double fullinertia[6], const double iquat[4]) const;

  // the gauge in the coordinates of a frame with the given orientation
  mjCGauge Rotated(const double quat[4]) const;

  // geometric mean of the gauges of two participants, for quantities without a direction
  static mjCGauge Mean(const mjCGauge& a, const mjCGauge& b);

  bool IsIdentity() const {
    return uniform && shape == 1 && length == 1 && mass == 1 && inertia == 1;
  }
  bool     IsForceIdentity() const { return length == 1 && mass == 1; }
  mjCGauge Inverse() const;
  mjCGauge operator*(const mjCGauge& other) const;
};

// gauge in force inside the lowest frame or body which contains all the given elements; a body
// contains itself
mjCGauge LowestContainerGauge(const std::vector<const mjCBase*>& elements);

class mjCBase_ : public mjsElement {
 public:
  int         id;         // object id
  std::string name;       // object name
  std::string classname;  // defaults class name
  std::string info;       // error message info set by the user
  std::string prefix;     // prefix for model operations
  std::string suffix;     // suffix for model operations
};

class mjCBase : public mjCBase_ {
  friend class mjCDef;

 public:
  // load resource if found (fallback to OS filesystem)
  static mjResource* LoadResource(const std::string& modelfiledir,
                                  const std::string& filename,
                                  const mjVFS*       vfs);

  // path of an asset file as LoadResource opens it: modelfiledir of the spec that owns this
  // element, then dir, then file; file is returned as is if absolute or with a scheme
  std::string ResolvePath(const std::string& dir, const std::string& file) const;

  // Get and sanitize content type from raw_text if not empty, otherwise parse
  // content type from resource_name; throw on failure
  static std::string GetAssetContentType(std::string_view resource_name, std::string_view raw_text);

  // the file of an asset of the spec as compilation resolves it; empty if there is no file
  std::string ResolvedSpecFile(const std::string& dir, std::string file) const;

  // Add frame transformation
  void SetFrame(mjCFrame* _frame);

  // Copy spec into private attributes
  virtual void CopyFromSpec() {}

  // Throws an error if any of the references is missing
  virtual void ResolveReferences(const mjCModel* m) {}

  // Appends prefix and suffix to reference
  virtual void NameSpace(const mjCModel* m);

  // Copy plugins instantiated in this object
  virtual void CopyPlugin() {}

  // Returns parent of this object
  virtual mjCBase* GetParent() const { return nullptr; }

  // gauge of this element: that of its innermost frame or else its parent, from the spec
  mjCGauge Gauge() const;

  // gauge of what this element contains, in its own coordinates
  virtual mjCGauge ContentGauge() const { return Gauge(); }

  // Returns the model of this object
  mjsCompiler* FindCompiler(const mjsCompiler* compiler) const;

  // Copy assignment
  mjCBase& operator=(const mjCBase& other);

  mjCFrame*    frame;     // pointer to frame transformation
  mjCModel*    model;     // pointer to model that owns object
  mjsCompiler* compiler;  // pointer to the compiler options

  virtual ~mjCBase() = default;  // destructor

  // forget the keyframe values of this element which are stored under the given names, or all
  // others if keep is true
  virtual void ForgetKeyframes(const std::vector<std::string>& names, bool keep = false) {}

  // forget the id of this element and the addresses of its state in the compiled model
  virtual void ResetId() { id = -1; }

  // increment and decrement reference count
  // release uses the argument to delete the plugin
  // which may be still owned by the source spec during shallow attach
  virtual void AddRef() { ++refcount; }
  virtual int  GetRef() { return refcount; }
  virtual void Release() {
    if (--refcount == 0) { delete this; }
  }

  // Set and get user payload
  void        SetUserValue(std::string_view key, const void* data, void (*cleanup)(const void*));
  const void* GetUserValue(std::string_view key);
  void        DeleteUserValue(std::string_view key);

 protected:
  mjCBase();                      // constructor
  mjCBase(const mjCBase& other);  // copy constructor

  // reference count for allowing deleting an attached object
  int refcount = 1;

  // Arbitrary user value that cleans up the data when destroyed.
  struct UserValue {
    const void* value            = nullptr;
    void (*cleanup)(const void*) = nullptr;

    UserValue() {}
    UserValue(const void* value, void (*cleanup)(const void*)) : value(value), cleanup(cleanup) {}
    UserValue(const UserValue& other)            = delete;
    UserValue& operator=(const UserValue& other) = delete;

    UserValue(UserValue&& other) : value(other.value), cleanup(other.cleanup) {
      other.value   = nullptr;
      other.cleanup = nullptr;
    }

    UserValue& operator=(UserValue&& other) {
      if (this != &other) {
        if (cleanup && value) { cleanup(value); }
        value         = other.value;
        cleanup       = other.cleanup;
        other.value   = nullptr;
        other.cleanup = nullptr;
      }
      return *this;
    }

    ~UserValue() {
      if (cleanup && value) { cleanup(value); }
    }
  };

  // user payload
  std::unordered_map<std::string, UserValue> user_payload_;
};


//------------------------- class mjCBody -----------------------------------------------
// Describes a rigid body

class mjCBody_ : public mjCBase {
 protected:
  mjCBody*  parent;
  mjCFrame* iframe;  // frame enclosing the inertial element, nullptr if none

  // variables computed by 'Compile' and 'AddXXX'
  int weldid;    // top index of body we are welded to
  int dofnum;    // number of motion dofs for body
  int mocapid;   // mocap id, -1: not mocap
  int bodyadr_;  // address of body in model, -1: not compiled

  double xpos0[3];   // global position in qpos0
  double xquat0[4];  // global orientation in qpos0

  // used internally by compiler
  int lastdof;      // id of last dof
  int subtreedofs;  // number of dofs in subtree, including self

  mjCBoundingVolumeHierarchy tree;  // bounding volume hierarchy

  // variable-size data
  std::string         plugin_name;
  std::string         plugin_instance_name;
  std::vector<double> userdata_;
  std::vector<double> spec_userdata_;

  // the inertial as compiled, in the body frame before a free joint is aligned with it, and with
  // the mass and inertia before settotalmass scales those of all bodies
  double ipos_compiled_[3]    = {0, 0, 0};
  double iquat_compiled_[4]   = {1, 0, 0, 0};
  double mass_compiled_       = 0;
  double inertia_compiled_[3] = {0, 0, 0};

  // compilation raised the mass or inertia to their bounds, or balanced the inertia
  bool inertia_adjusted_ = false;

  // how compilation gave the body its inertia: inferred from its geoms in the groups which it
  // counted, as the spec gave it, or none (massless)
  bool inertia_inferred_  = false;
  bool inertia_given_     = false;
  int  inertia_groups_[2] = {0, 0};

  // compilation moved the frame of the body to its inertial frame, to align it with a free joint
  bool aligned_ = false;

  // variables used for temporarily storing the state of the mocap bodies
  std::map<std::string, std::array<mjtNum, 3>> mpos_;   // saved mocap_pos
  std::map<std::string, std::array<mjtNum, 4>> mquat_;  // saved mocap_quat
};

class mjCBody : public mjCBody_, private mjsBody {
  friend class mjCJoint;
  friend class mjCGeom;
  friend class mjCSite;
  friend class mjCCamera;
  friend class mjCFrame;
  friend class mjCLight;
  friend class mjCFlex;
  friend class mjCFlexcomp;
  friend class mjCEquality;
  friend class mjCPair;
  friend class mjCModel;
  friend class mjXReader;
  friend class mjXWriter;
  friend class mjXURDF;

 public:
  explicit mjCBody(mjCModel*);
  ~mjCBody();

  // API for adding objects to body
  mjCBody*   AddBody(mjCDef* = 0);
  mjCFrame*  AddFrame(mjCFrame* = 0);
  mjCJoint*  AddJoint(mjCDef* = 0);
  mjCJoint*  AddFreeJoint();
  mjCGeom*   AddGeom(mjCDef* = 0);
  mjCSite*   AddSite(mjCDef* = 0);
  mjCCamera* AddCamera(mjCDef* = 0);
  mjCLight*  AddLight(mjCDef* = 0);

  // API for adding/removing objects to body
  mjCBody& operator+=(const mjCBody& other);
  mjCBody& operator+=(const mjCFrame& other);
  mjCBody& operator-=(const mjCBody& subtree);

  // API for accessing objects
  int      NumObjects(mjtObj type);
  mjCBase* GetObject(mjtObj type, int id);
  mjCBase* FindObject(mjtObj type, const std::string& name, bool recursive = true) const;

  // Propagate suffix and prefix to the whole tree
  void NameSpace(const mjCModel* m);

  // set explicitinertial to true
  void MakeInertialExplicit();

  // make the inertial which compilation calculated for this body part of the spec; the kinematic
  // tree must have been compiled since the spec was last edited, see mjCModel::Resolve
  void AdoptInertial();

  // compute the bounding volume hierarchy of the body.
  void ComputeBVH();

  // variables set by user
  mjsBody spec;

  // inherited
  using mjCBase::info;

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() { return userdata_; }

  // get next child of given type recursively; if `child` is found while traversing the tree,
  // then `found` is set to true and the next element encountered is returned;
  // returns nullptr if the next child is not found or if `child` is the last element, returns
  // the next child after the input `child` otherwise
  mjsElement* NextChild(const mjsElement* child,
                        mjtObj            type      = mjOBJ_UNKNOWN,
                        bool              recursive = false) const;

  // forget the keyframe values of this body and its subtree
  void ForgetKeyframes(const std::vector<std::string>& names, bool keep = false);

  // create a frame and move all contents of this body into it
  mjCFrame* ToFrame(bool mergeinertial = true);

  // get mocap position and quaternion
  mjtNum* mpos(const std::string& state_name);
  mjtNum* mquat(const std::string& state_name);

  mjsFrame* last_attached;  // last attached frame to this body

  // set parent of this body
  void     SetParent(mjCBody* _body) { parent = _body; }
  mjCBody* GetParent() const { return parent; }

  // set model of this body
  void SetModel(mjCModel* _model);

  // reset ids of all objects in this body
  void ResetId();

  // getters
  std::vector<mjCBody*> Bodies() const { return bodies; }

  // get list of a given type
  template <class T>
  const std::vector<T*>& GetList() const;

  // accumulate compiled inertia of another body into this body
  void AccumulateInertia(const mjsBody* other);

  // gauge of what this body contains, in its coordinates
  mjCGauge ContentGauge() const;

 private:
  mjCBody(const mjCBody& other, mjCModel* _model);  // copy constructor
  mjCBody& operator=(const mjCBody& other);         // copy assignment

  void Compile(void);          // compiler
  void ApplyGauge();           // scale the compiled copy by the gauge
  bool InertiaFromGeom(void);  // get inertial info from geoms

  // true if compilation infers the inertial of this body from its geoms
  bool InfersInertial() const;

  // get the inertial in the spec: center of mass in body coordinates and inertia matrix about it
  void SpecInertial(double com[3], double inert[6]) const;

  // get the inertial which compilation calculated, in the same form; the kinematic tree must
  // have been compiled since the spec was last edited
  void CompiledInertial(double com[3], double inert[6]) const;

  // true if compilation scales the inertial of this body, by its frame or by that of the inertial
  bool ScaledInertial() const;

  // merge the inertial of a child body into the inertial in the spec of this body; an inertial
  // which is inferred from geoms is taken as compiled, one which is given as it is in the spec.
  // Scaled inertials are all taken as compiled, and the sum is written in the units of this body
  void MergeInertial(const mjCBody* child);

  // turn a compiled pose of this body into the pose in the spec which compiles to it: before the
  // alignment with a free joint, and in the frame which the body is in
  void PoseInSpec(double bodypos[3], double bodyquat[4]) const;

  // the same for a compiled pose of an element of this body, which is in the given frame or in
  // none. Given the unit quaternion, it returns the rotation which brings a direction of the
  // element from the body to the spec
  void ElementPoseInSpec(const mjCFrame* elementframe,
                         double          elementpos[3],
                         double          elementquat[4]) const;

  // write to the spec what compiles to the compiled values of this body, for mj_copyBack: the
  // position and the orientation which are new, so that the other stays as it is written; and
  // the inertial, or only its mass
  void PoseToSpec(bool position, bool orientation);
  void InertialToSpec(bool massonly);

  // objects allocated by Add functions
  std::vector<mjCBody*>   bodies;   // child bodies
  std::vector<mjCGeom*>   geoms;    // geoms attached to this body
  std::vector<mjCFrame*>  frames;   // frames attached to this body
  std::vector<mjCJoint*>  joints;   // joints allowing motion relative to parent
  std::vector<mjCSite*>   sites;    // sites attached to this body
  std::vector<mjCCamera*> cameras;  // cameras attached to this body
  std::vector<mjCLight*>  lights;   // lights attached to this body

  void CopyFromSpec();  // copy spec into attributes
  void PointToLocal(void);
  void NameSpace_(const mjCModel* m, bool propagate = true);
  void CopyPlugin();

  // copy src list of elements into dst; set body, model and frame
  template <typename T>
  void CopyList(std::vector<T*>&          dst,
                const std::vector<T*>&    src,
                std::map<mjCFrame*, int>& fmap,
                const mjCFrame*           pframe = nullptr);
};


//------------------------- class mjCFrame ---------------------------------------------------------
// Describes a coordinate transformation relative to its parent

class mjCFrame_ : public mjCBase {
 protected:
  bool compiled;  // frame already compiled
};

class mjCFrame : public mjCFrame_, private mjsFrame {
  friend class mjCBase;
  friend class mjCBody;
  friend class mjCGeom;
  friend class mjCJoint;
  friend class mjCSite;
  friend class mjCCamera;
  friend class mjCLight;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  mjCFrame(mjCModel* = 0, mjCFrame* = 0);
  mjCFrame(const mjCFrame& other);
  mjCFrame& operator=(const mjCFrame& other);

  mjsFrame spec;
  using mjCBase::info;

  void     CopyFromSpec(void);
  void     PointToLocal(void);
  void     SetParent(mjCBody* _body) { body = _body; }
  mjCBody* GetParent() const { return body; }

  mjCFrame& operator+=(const mjCBody& other);

  bool IsAncestor(const mjCFrame* child) const;  // true if child is contained in this frame

  // gauge of what this frame contains: its own composed with its scale
  mjCGauge ContentGauge() const;

  // express in this frame, as compiled, a pose which is given in its body
  void ToLocal(double childpos[3], double childquat[4]) const;

  // map a point or a direction given in the body of this frame from the coordinates of the spec to
  // the compiled ones, which scale the contents of the frame, or back
  void ScaleInBody(double x[3], bool point, bool inverse) const;

  mjsBody* last_attached;  // last attached body to this frame

 private:
  void Compile(void);  // compiler
  void ApplyGauge();   // scale the compiled copy by the gauge

  mjCBody* body;  // body that owns the frame
};


//------------------------- class mjCJoint ---------------------------------------------------------
// Describes a motion degree of freedom of a body relative to its parent

class mjCJoint_ : public mjCBase {
 protected:
  mjCBody* body;  // joint's body

  // variable used for temporarily storing the state of the joint
  std::map<std::string, std::array<mjtNum, 7>> qpos_;     // qpos at the previous step
  std::map<std::string, std::array<mjtNum, 6>> qvel_;     // qvel at the previous step
  std::map<std::string, std::array<mjtNum, 7>> plastic_;  // plastic at the previous step

  // variable-size data
  std::vector<double> userdata_;
  std::vector<double> spec_userdata_;
};

class mjCJoint : public mjCJoint_, private mjsJoint {
  friend class mjCDef;
  friend class mjCEquality;
  friend class mjCBody;
  friend class mjCFlex;
  friend class mjCModel;
  friend class mjCSensor;
  friend class mjXWriter;
  friend class mjXURDF;

 public:
  explicit mjCJoint(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCJoint(const mjCJoint& other);
  mjCJoint& operator=(const mjCJoint& other);

  mjsJoint spec;
  using mjCBase::info;

  void     CopyFromSpec(void);
  void     SetParent(mjCBody* _body) { body = _body; }
  mjCBody* GetParent() const { return body; }

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() const { return userdata_; }
  const double*              get_range() const { return range; }

  bool is_limited() const;
  bool is_actfrclimited() const;

  static int nq(mjtJoint joint_type);
  static int nv(mjtJoint joint_type);
  int        nq() const { return nq(spec.type); }
  int        nv() const { return nv(spec.type); }

  mjtNum* qpos(const std::string& state_name);
  mjtNum* qvel(const std::string& state_name);
  mjtNum* plastic(const std::string& state_name);
  void    ForgetKeyframes(const std::vector<std::string>& names, bool keep = false);
  void    ResetId();

 private:
  int  Compile(void);  // compiler; return dofnum
  void PointToLocal(void);

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  // write to the spec the anchor, the axis or both, which compile to the compiled ones: in the
  // frame which the joint is in
  void AnchorToSpec(bool anchor, bool direction);

  // variables that should not be copied during copy assignment
  int qposadr_;     // address of dof in data->qpos
  int dofadr_;      // address of dof in data->qvel
  int plasticadr_;  // address in data->plastic
};


//------------------------- class mjCGeom ----------------------------------------------------------
// Describes a geometric shape belonging to a body

class mjCGeom_ : public mjCBase {
 public:
  bool inferinertia;  // true if inertia should be computed from geom

 protected:
  int        matid;            // id of geom's material
  mjCMesh*   mesh;             // geom's mesh
  mjCHField* hfield;           // geom's hfield
  double     mass_;            // mass
  double     inertia[3];       // local diagonal inertia
  double     aabb[6];          // axis-aligned bounding box (center, size)
  mjCBody*   body;             // geom's body
  mjtNum     fluid[mjNFLUID];  // compile-time fluid-interaction parameters

  // variable-size data
  std::string         plugin_name;
  std::string         plugin_instance_name;
  std::string         hfieldname_;
  std::string         meshname_;
  std::string         material_;
  std::vector<double> userdata_;
  std::string         spec_hfieldname_;
  std::string         spec_meshname_;
  std::string         spec_material_;
  std::vector<double> spec_userdata_;
};

class mjCGeom : public mjCGeom_, private mjsGeom {
  friend class mjCDef;
  friend class mjCMesh;
  friend class mjCPair;
  friend class mjCBody;
  friend class mjCModel;
  friend class mjCWrap;
  friend class mjXWriter;
  friend class mjXURDF;

 public:
  explicit mjCGeom(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCGeom(const mjCGeom& other);
  mjCGeom& operator=(const mjCGeom& other);

  mjsGeom  spec;               // variables set by user
  double   GetVolume() const;  // compute geom volume
  void     SetInertia(void);   // compute and set geom inertia
  void     SetParent(mjCBody* _body) { body = _body; }
  mjCBody* GetParent() const { return body; }
  mjtGeom  Type() const { return type; }

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() const { return userdata_; }
  const std::string&         get_hfieldname() const { return spec_hfieldname_; }
  const std::string&         get_meshname() const { return spec_meshname_; }
  const std::string&         get_material() const;
  void                       del_material() {
    spec_material_.clear();
    material_.clear();
  }

 private:
  void   Compile(void);      // compiler
  double GetRBound(void);    // bounding sphere radius of a mesh or SDF geom
  void   ComputeAABB(void);  // compute axis-aligned bounding box
  void   CopyFromSpec(void);
  void   PointToLocal(void);
  void   NameSpace(const mjCModel* m);
  void   CopyPlugin();

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  // map of the gauge in the frame of this geom, which its mesh or height field instance takes
  std::array<double, 9> ShapeMap() const;

  // write to the spec what compiles to the compiled values of this geom, for mj_copyBack: the
  // position and the orientation which are new; and those of size, pose and surface velocity
  // which are new. A size and pose which the spec gives as fromto, or by fitting to a mesh, are
  // written in its place
  void PoseToSpec(bool position, bool orientation);
  void ShapeToSpec(bool newsize, bool position, bool orientation, bool newvelocity);

  // inherited
  using mjCBase::info;
};


//------------------------- class mjCSite ----------------------------------------------------------
// Describes a site on a body

class mjCSite_ : public mjCBase {
 protected:
  // variable-size data
  std::string         material_;
  std::string         meshname_;
  std::vector<double> userdata_;
  std::string         spec_material_;
  std::string         spec_meshname_;
  std::vector<double> spec_userdata_;

  // variables computed by 'compile' and 'mjCBody::addSite'
  mjCBody* body;   // site's body
  int      matid;  // material id for rendering
  mjCMesh* mesh;   // site's mesh
};

class mjCSite : public mjCSite_, private mjsSite {
  friend class mjCDef;
  friend class mjCBody;
  friend class mjCModel;
  friend class mjXWriter;
  friend class mjXURDF;

 public:
  explicit mjCSite(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCSite(const mjCSite& other);
  mjCSite& operator=(const mjCSite& other);

  mjsSite spec;  // variables set by user

  // site's body
  mjCBody* Body() const { return body; }
  void     SetParent(mjCBody* _body) { body = _body; }
  mjCBody* GetParent() const { return body; }

  // site's mesh
  mjCMesh* Mesh() const { return mesh; }
  void     SetMesh(mjCMesh* _mesh) { mesh = _mesh; }

  // use strings from mjCBase rather than mjStrings from mjsSite
  using mjCBase::info;

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() const { return userdata_; }
  const std::string&         get_material() const;
  void                       del_material() {
    spec_material_.clear();
    material_.clear();
  }
  const std::string& get_meshname() const { return spec_meshname_; }
  void               del_meshname() { spec_meshname_.clear(); }

 private:
  void Compile(void);   // compiler
  void CopyFromSpec();  // copy spec into attributes
  void PointToLocal(void);
  void NameSpace(const mjCModel* m);

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  // map of the gauge in the frame of this site, which its mesh instance takes
  std::array<double, 9> ShapeMap() const;

  // write to the spec what compiles to the compiled values of this site, as for a geom
  void PoseToSpec(bool position, bool orientation);
  void ShapeToSpec(bool newsize, bool position, bool orientation);
};


//------------------------- class mjCCamera --------------------------------------------------------
// Describes a camera, attached to a body

class mjCCamera_ : public mjCBase {
 protected:
  mjCBody*    body;          // camera's body
  int         targetbodyid;  // id of target body; -1: none
  std::string targetbody_;
  std::string spec_targetbody_;

  std::vector<double> userdata_;
  std::vector<double> spec_userdata_;
};

class mjCCamera : public mjCCamera_, private mjsCamera {
  friend class mjCDef;
  friend class mjCBody;
  friend class mjCModel;
  friend class mjCSensor;
  friend class mjXWriter;

 public:
  explicit mjCCamera(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCCamera(const mjCCamera& other);
  mjCCamera& operator=(const mjCCamera& other);

  mjsCamera spec;
  using mjCBase::info;

  // used by mjXWriter and mjCModel
  const std::string&         get_targetbody() const { return targetbody_; }
  const std::vector<double>& get_userdata() const { return userdata_; }

  void     SetParent(mjCBody* _body) { body = _body; }
  mjCBody* GetParent() const { return body; }

 private:
  void Compile(void);  // compiler
  void CopyFromSpec(void);
  void PointToLocal(void);
  void NameSpace(const mjCModel* m);
  void ResolveReferences(const mjCModel* m);

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  // write to the spec the position and the orientation which compile to the compiled ones, each
  // only if asked
  void PoseToSpec(bool position, bool orientation);

  // write the compiled intrinsics as focal length and principal point, in units of length: to the
  // spec, and to the compiled copy, which may have them in pixels
  void IntrinsicToSpec(bool tospec);
};


//------------------------- class mjCLight ---------------------------------------------------------
// Describes a light, attached to a body

class mjCLight_ : public mjCBase {
 protected:
  mjCBody*    body;          // light's body
  int         targetbodyid;  // id of target body; -1: none
  int         texid;         // id of texture; -1: none
  std::string texture_;
  std::string spec_texture_;
  std::string targetbody_;
  std::string spec_targetbody_;
};

class mjCLight : public mjCLight_, private mjsLight {
  friend class mjCDef;
  friend class mjCBody;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCLight(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCLight(const mjCLight& other);
  mjCLight& operator=(const mjCLight& other);

  mjsLight spec;
  using mjCBase::info;

  // used by mjXWriter and mjCModel
  const std::string& get_targetbody() const { return targetbody_; }
  const std::string& get_texture() const { return texture_; }
  void               del_texture() {
    spec_texture_.clear();
    texture_.clear();
  }

  void     SetParent(mjCBody* _body) { body = _body; }
  mjCBody* GetParent() const { return body; }

 private:
  void Compile(void);  // compiler
  void CopyFromSpec(void);
  void PointToLocal(void);
  void NameSpace(const mjCModel* m);
  void ResolveReferences(const mjCModel* m);

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  // write to the spec the position and the direction which compile to the compiled ones, each
  // only if asked
  void PoseToSpec(bool position, bool direction);
};


//------------------------- class mjCFlex ----------------------------------------------------------
// Describes a flex

struct StencilFlap {
  static constexpr int kNumVerts = 4;
  int                  vertices[kNumVerts];
};

class mjCFlex_ : public mjCBase {
 protected:
  int                              nvert;         // number of vertices
  int                              nnode;         // number of nodes
  int                              nedge;         // number of edges
  int                              nelem;         // number of elements
  int                              plasticadr_;   // plastic state address
  int                              plasticnum_;   // number of plastic state variables
  std::vector<int>                 elemmatid_;    // element material ids
  bool                             rigid;         // all vertices attached to the same body
  bool                             centered;      // all vertices coordinates (0,0,0)
  bool                             interpolated;  // vertices are interpolated from nodes
  std::vector<int>                 vertbodyid;    // vertex body ids
  std::vector<int>                 nodebodyid;    // node body ids
  std::vector<std::pair<int, int>> edge;          // edge vertex ids
  std::vector<int>                 shell;         // shell fragment vertex ids (dim per fragment)
  std::vector<int>                 elemlayer;     // element layer (distance from border)
  std::vector<StencilFlap>         flaps;         // adjacent triangles
  std::vector<double>              vertxpos;      // global vertex positions
  mjCBoundingVolumeHierarchy       tree;          // bounding volume hierarchy
  std::vector<double>              elemaabb_;     // element bounding volume
  std::vector<int>                 edgeidx_;      // element edge ids
  std::vector<double>              stiffness;     // elasticity stiffness matrix
  std::vector<double>              bending;       // bending stiffness matrix
  bool                has_strain_eq = false;      // true if strain constraints reference this flex
  std::vector<bool>   cell_empty;                 // true if cell contains no mesh geometry
  std::vector<bool>   cell_full;                  // true if cell is 100% inside mesh
  std::vector<double> cell_vol;                   // material volume fraction per cell
  std::vector<std::vector<double>> cell_alphas;   // subcell quadrature occupancy for cut cells

  // variable-size data
  std::vector<std::string> vertbody_;      // vertex body names
  std::vector<std::string> nodebody_;      // node body names
  std::vector<double>      vert_;          // vertex positions
  std::vector<double>      node_;          // node positions
  std::vector<int>         elem_;          // element vertex ids
  std::vector<float>       texcoord_;      // vertex texture coordinates
  std::vector<int>         elemtexcoord_;  // face texture coordinates (OBJ only)
  std::vector<std::string> material_;      // names of materials used for rendering

  std::vector<std::string> spec_material_;
  std::vector<std::string> spec_vertbody_;
  std::vector<std::string> spec_nodebody_;
  std::vector<double>      spec_vert_;
  std::vector<double>      spec_node_;
  std::vector<int>         spec_elem_;
  std::vector<float>       spec_texcoord_;
  std::vector<int>         spec_elemtexcoord_;

  // caching
  std::vector<double>                        cached_stiffness_;  // cached stiffness matrix
  std::map<std::string, std::vector<double>> plastic_;
};

class mjCFlex : public mjCFlex_, private mjsFlex {
  friend class mjCDef;
  friend class mjCModel;
  friend class mjCFlexcomp;
  friend class mjCEquality;
  friend class mjXWriter;

 public:
  explicit mjCFlex(mjCModel* = nullptr);
  mjCFlex(const mjCFlex& other);
  mjCFlex& operator=(const mjCFlex& other);

  mjsFlex spec;
  using mjCBase::info;

  void CopyFromSpec(void);
  void PointToLocal(void);
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);

  // used by mjXWriter and mjCModel
  const std::vector<std::string>& get_material() const { return material_; }
  const std::vector<std::string>& get_vertbody() const { return vertbody_; }
  const std::vector<double>&      get_vert() const { return vert_; }
  const std::vector<double>&      get_elemaabb() const { return elemaabb_; }
  const std::vector<int>&         get_elem() const { return elem_; }
  const std::vector<float>&       get_texcoord() const { return texcoord_; }
  const std::vector<int>&         get_elemtexcoord() const { return elemtexcoord_; }
  const std::vector<std::string>& get_nodebody() const { return nodebody_; }
  const std::vector<double>&      get_node() const { return node_; }

  bool HasTexcoord() const;  // texcoord not null
  void del_material() {
    spec_material_.clear();
    material_.clear();
  }

  std::vector<double>& plastic(const std::string& state_name) { return plastic_[state_name]; }
  int                  PlasticNum() const;
  int                  PlasticMode() const;
  int                  PlasticStretchNum() const;

  // gauge of the lowest frame or body which contains the bodies of the vertices and nodes
  mjCGauge BodyGauge() const;

  static constexpr int kNumEdges[3] = {1, 3, 6};  // number of edges per element indexed by dim


 private:
  bool   IsSimple() const;           // fixed or independent XYZ-slide attachments
  void   Compile(const mjVFS* vfs);  // compiler
  void   CreateBVH(void);            // create flex BVH
  void   CreateShell(void);          // create shells
  void   ComputeCellEmpty(const double* vpos,
                          const int*    elems,  // identify cells
                          int           ne,
                          int           fdim,        // with no mesh content
                          const double  minmax[6]);  // bounding box of the node grid
  bool   NodeInEmptyCells(int gi, int gj,
                          int gk) const;  // node only in empty cells
  double NodeMassWeight(int gi, int gj,
                        int gk) const;  // FCM volume-weighted nodal mass

  // scale the compiled copy: positions by the gauges of their bodies, the rest by that of the flex
  void ApplyGauge();
  void ForgetKeyframes(const std::vector<std::string>& names, bool keep = false);
  void ResetId();

  std::vector<double> vert0_;     // vertex positions in [0, 1]^d in the bounding box
  std::vector<int>    vertcell_;  // finite cell index for vertex; -1: none
  std::vector<double> node0_;     // node Cartesian positions

  // compute unrotated node positions for stiffness computation
  // optionally outputs the grid rotation matrix R0 (stored as rows)
  std::vector<double> ComputeUnrotatedNodePositions(const std::vector<double>& nodexpos,
                                                    double* R0_out = nullptr) const;

  // stiffness and FCM geometry caching
  std::string ComputeStiffnessCacheKey(const std::vector<double>& nodexpos_local) const;
  bool        LoadCachedStiffness(const std::vector<double>& nodexpos_local);
  void        CacheStiffness(const std::vector<double>& nodexpos_local);
  std::string ComputeFcmCacheKey(
      const double* vpos, const int* elems, int ne, int fdim, const double minmax[6]) const;
  bool LoadCachedFcmGeometry(const std::string& key);
  void CacheFcmGeometry(const std::string& key);
};


//------------------------- class mjCMesh ----------------------------------------------------------
// Describes a mesh

class mjCMesh_ : public mjCBase {
 protected:
  // variable size attributes
  std::string plugin_name;
  std::string plugin_instance_name;

  std::string         content_type_ = "";   // content type of file
  std::string         file_;                // mesh file
  mjResource*         resource_ = nullptr;  // resource for mesh file
  std::vector<float>  vert_;                // vertex data
  std::vector<float>  normal_;              // normal data
  std::vector<float>  texcoord_;            // texcoord data
  std::vector<int>    face_;                // vertex indices
  std::vector<int>    facenormal_;          // normal indices
  std::vector<int>    facetexcoord_;        // texcoord indices
  std::vector<double> node_;                // volume nodes
  std::vector<int>    tet_;                 // tetrahedral cells
  std::string         material_;            // mesh fallback material

  std::string         spec_content_type_;
  std::string         spec_file_;
  std::vector<float>  spec_vert_;
  std::vector<float>  spec_normal_;
  std::vector<float>  spec_texcoord_;
  std::vector<int>    spec_face_;
  std::vector<int>    spec_facenormal_;
  std::vector<int>    spec_facetexcoord_;
  std::vector<double> spec_node_;
  std::vector<int>    spec_tet_;
  std::string         spec_material_;

  // used by the compiler
  bool                             needreorient_;  // needs reorientation
  std::vector<std::pair<int, int>> halfedge_;      // half-edge data

  // mesh processed flags
  bool processed_;    // has the mesh been processed yet
  bool transformed_;  // has the mesh been transformed to CoM and inertial frame

  // mesh properties computed by Compile
  double pos_[3];    // CoM position
  double quat_[4];   // inertia orientation
  double boxsz_[3];  // half-sizes of equivalent inertia box
  double aamm_[6];   // axis-aligned bounding box in (min, max) format
  double volume_;    // volume of the mesh
  double surface_;   // surface of the mesh

  // size of mesh data to be copied into mjModel
  int  szgraph_ = 0;  // size of graph data in ints
  bool needhull_;     // needs convex hull for collisions
  bool needsdf_;      // needs signed distance field for an sdf geom
  int  maxhullvert_;  // max vertex count of convex hull

  // bounding volume hierarchy tree
  mjCBoundingVolumeHierarchy tree_;       // bounding volume hierarchy
  std::vector<double>        face_aabb_;  // bounding boxes of all faces

  // octree
  mjCOctree octree_;  // octree of the mesh
  double    mesh_timer_[mjNCTIMER] = {0};

 public:
  // the asset of the spec which this instance repeats at another scale, if compilation made it,
  // and the linear map, in the frame of the geoms and sites which use this instance, of that scale
  const mjCMesh*        primary_     = nullptr;
  std::array<double, 9> instancemap_ = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

class mjCMesh : public mjCMesh_, private mjsMesh {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCMesh(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCMesh(const mjCMesh& other);
  mjCMesh& operator=(const mjCMesh& other);
  ~mjCMesh();

  mjsMesh spec;
  using mjCBase::info;

  void CopyFromSpec(void);
  void PointToLocal(void);
  void NameSpace(const mjCModel* m);

  // what compilation reads from the spec, names excluded: the key of mjs_dedupAssets
  std::string AssetKey() const;

  // make a mesh of a predefined shape
  void MakeHemisphere(int res, bool make_faces, bool make_cap);
  void MakeSphere(int subdiv, bool make_faces);
  void MakeTorus(int res, double radius);
  void MakeSupertorus(int res, double radius, double s, double t);
  void MakeSupersphere(int res, double e, double n);
  void MakeWedge(int resolution[2], double fov[2], double gamma);
  void MakeRect(int resolution[2]);
  void MakeCone(int nedge, double radius);
  void MakeGrid(const int count[3], const double spacing[3], int dim, bool needtex);
  void MakeCircle(const int count[3], const double spacing[3]);
  void MakeDisc(const int count[3], const double spacing[3], bool needtex);
  void MakeBox(const int count[3], const double spacing[3], int dim, bool needtex, bool open);
  void MakeCylinder(const int count[3], const double spacing[3], int dim, bool needtex, bool open);
  void MakeEllipsoid(const int count[3], const double spacing[3], int dim, bool needtex, bool open);

  // accessors
  const mjsPlugin&           Plugin() const { return plugin; }
  const std::string&         ContentType() const { return content_type_; }
  const std::string&         File() const { return file_; }
  std::string                ResolvedFilePath() const;  // modelfiledir + meshdir + file
  const double*              Refpos() const { return refpos; }
  const double*              Refquat() const { return refquat; }
  const double*              Scale() const { return scale; }
  bool                       SmoothNormal() const { return smoothnormal; }
  const std::vector<float>&  Vert() const { return vert_; }
  float                      Vert(int i) const { return vert_[i]; }
  const std::vector<float>&  UserVert() const { return spec_vert_; }
  const std::vector<float>&  UserNormal() const { return spec_normal_; }
  const std::vector<float>&  Texcoord() const { return texcoord_; }
  const std::vector<int>&    FaceTexcoord() const { return facetexcoord_; }
  const std::vector<float>&  UserTexcoord() const { return spec_texcoord_; }
  const std::vector<int>&    Face() const { return face_; }
  const std::vector<int>&    UserFace() const { return spec_face_; }
  const std::vector<double>& Node() const { return node_; }
  const std::vector<double>& UserNode() const { return spec_node_; }
  std::vector<double>&       UserNode() { return spec_node_; }
  const std::vector<int>&    Tet() const { return tet_; }
  const std::vector<int>&    UserTet() const { return spec_tet_; }
  mjtMeshInertia             Inertia() const { return spec.inertia; }
  const std::string&         Material() const { return material_; }

  // the frame of the compiled mesh in the coordinates of its geoms
  const double* Pos() const { return pos_; }
  const double* Quat() const { return quat_; }

  // true if the map of this instance is diagonal, which then multiplies the scale of the mesh
  bool InstanceDiagonal() const;

  // setters
  void SetNeedHull(bool needhull) { needhull_ = needhull; }
  void SetNeedSDF(bool needsdf) { needsdf_ = needsdf; }

  // mesh properties computed by Compile
  const double* aamm() const { return aamm_; }

  // number of vertices, normals, texture coordinates, and faces
  int nvert() const { return vert_.size() / 3; }
  int nnormal() const { return normal_.size() / 3; }
  int ntexcoord() const { return texcoord_.size() / 2; }
  int nface() const { return face_.size() / 3; }
  int nnode() const { return node_.size() / 3; }
  int ntet() const { return tet_.size() / 4; }
  int npolygon() const { return polygons_.size(); }
  int npolygonvert() const {
    int acc = 0;
    for (const auto& polygon : polygons_) { acc += polygon.size(); }
    return acc;
  }
  int npolygonmap() const {
    int acc = 0;
    for (const auto& polygon : polygon_map_) { acc += polygon.size(); }
    return acc;
  }

  // return size of graph data in ints
  int szgraph() const { return szgraph_; }

  // bounding volume hierarchy tree
  const mjCBoundingVolumeHierarchy& tree() { return tree_; }

  // octree
  const mjCOctree& octree() { return octree_; }
  mjCOctree&       mutable_octree() { return octree_; }

  void    Compile(const mjVFS* vfs);                 // compiler
  double* GetPosPtr();                               // get position
  double* GetQuatPtr();                              // get orientation
  double* GetInertiaBoxPtr();                        // get inertia box
  double  GetVolumeRef() const;                      // get volume
  void    FitGeom(mjCGeom* geom, double center[3]);  // approximate mesh with simple geom
  bool    HasTexcoord() const;                       // texcoord not null
  void    del_material() {
    spec_material_.clear();
    material_.clear();
  }

  void CopyVert(float* arr) const;        // copy vert data into array
  void CopyNormal(float* arr) const;      // copy normal data into array
  void CopyFace(int* arr) const;          // copy face data into array
  void CopyFaceNormal(int* arr) const;    // copy face normal data into array
  void CopyFaceTexcoord(int* arr) const;  // copy face texcoord data into array
  void CopyTexcoord(float* arr) const;    // copy texcoord data into array
  void CopyGraph(int* arr) const;         // copy graph data into array

  // copy polygon data into array
  void CopyPolygons(int* verts, int* adr, int* num, int poly_adr) const;

  // copy polygon map data into array
  void CopyPolygonMap(int* faces, int* adr, int* num, int poly_adr) const;

  // copy polygon normal data into array
  void CopyPolygonNormals(mjtNum* arr);

  // sets properties of a bounding volume given a face id
  void SetBoundingVolume(int faceid, const double* dvert);

  // load from OBJ, STL, or MSH file; throws mjCError on failure
  void LoadFromResource(mjResource* resource, bool remove_repeated = false);


  static bool IsSTL(std::string_view filename, std::string_view ct = "");
  static bool IsMSH(std::string_view filename, std::string_view ct = "");


  bool IsSTL() const;
  bool IsMSH() const;

 private:
  void TryCompile(const mjVFS* vfs);

  // load mesh from cache asset, return true on success (OBJ files are only supported)
  bool LoadCachedMesh(mjCCache* cache, const mjResource* resource);

  // store mesh into asset cache (OBJ files are only supported)
  void CacheMesh(mjCCache* cache, const mjResource* resource);

  // id of the mesh in the asset cache: that of its file, and its factor for an instance at another
  // scale
  std::string CacheId(const mjResource* resource) const;


  // convert vertices to double precision and remove repeated vertices if requested
  void ProcessVertices(const std::vector<float>& vert, bool remove_repeated = false);


  void LoadFromDecoder(mjResource* resource,
                       bool        remove_repeated);         // load mesh using decoder plugin
  void LoadSTL(mjResource* resource);                        // load mesh in STL BIN format
  void LoadMSH(mjResource* resource, bool remove_repeated);  // load mesh in MSH BIN format

  void   LoadSDF();                            // generate mesh using marching cubes
  void   MakeGraph(const double* dvert);       // make graph of convex hull
  void   CopyGraph();                          // copy graph into face data
  void   MakeNormal(const double* dvert);      // compute vertex normals
  void   MakeCenter(const double* dvert);      // compute face circumcircle data
  void   Process();                            // compute inertial properties
  void   ApplyTransformations(double* dvert);  // apply user transformations
  void   TransformNodes(const double pos[3], const double quat[4]);  // transform nodes like verts
  double ComputeFaceCentroid(double[3], const double* dvert) const;
  void   CheckInitialMesh() const;  // check if initial mesh is valid
  void   CopyPlugin();
  void   Rotate(double quat[4], double* dvert);     // rotate mesh by quaternion
  void   Transform(double pos[3], double quat[4]);  // transform mesh by position and quaternion
  void   MakePolygons(const double* dvert);         // compute the polygon sides of the mesh
  void   MakePolygonNormals(const double* dvert);   // compute the normals of the polygons

  // computes the inertia matrix of the mesh given the type of inertia
  double ComputeInertia(double inert[6], const double CoM[3], const double* dvert) const;

  int* GraphFaces() const { return graph_ + 2 + 3 * (graph_[0] + graph_[1]); }

  // mesh data to be copied into mjModel
  double* center_;  // face circumcenter data (3*nface)
  int*    graph_;   // convex graph data

  // mesh data for collision detection
  std::vector<std::vector<int>> polygons_;         // polygons of the mesh
  std::vector<double>           polygon_normals_;  // normals of the polygons
  std::vector<std::vector<int>> polygon_map_;      // map from vertex to polygon

  // compute the volume and center-of-mass of the mesh given the face centroid
  double ComputeVolume(double CoM[3], const double facecen[3], const double* dvert) const;
  double ComputeSurfaceArea(double CoM[3], const double facecen[3], const double* dvert) const;
};


//------------------------- class mjCSkin ---------------------------------------------------------
// Describes a skin

class mjCSkin_ : public mjCBase {
 protected:
  // variable size attributes
  std::string                     file_;
  std::string                     material_;
  std::vector<float>              vert_;
  std::vector<float>              texcoord_;
  std::vector<int>                face_;
  std::vector<std::string>        bodyname_;
  std::vector<float>              bindpos_;
  std::vector<float>              bindquat_;
  std::vector<std::vector<int>>   vertid_;
  std::vector<std::vector<float>> vertweight_;

  std::string                     spec_file_;
  std::string                     spec_material_;
  std::vector<float>              spec_vert_;
  std::vector<float>              spec_texcoord_;
  std::vector<int>                spec_face_;
  std::vector<std::string>        spec_bodyname_;
  std::vector<float>              spec_bindpos_;
  std::vector<float>              spec_bindquat_;
  std::vector<std::vector<int>>   spec_vertid_;
  std::vector<std::vector<float>> spec_vertweight_;

  int              matid;   // material id
  std::vector<int> bodyid;  // body ids

  // compilation scaled the vertices and the bind poses with the bones
  bool scaled_ = false;

  // namespace of the attachments, applied to the bone names loaded from file
  std::string bone_prefix_;
  std::string bone_suffix_;
};

class mjCSkin : public mjCSkin_, private mjsSkin {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCSkin(mjCModel* = nullptr);
  mjCSkin(const mjCSkin& other);
  mjCSkin& operator=(const mjCSkin& other);
  ~mjCSkin();

  mjsSkin spec;
  using mjCBase::info;

  const std::string&                     File() const { return file_; }
  std::string                            ResolvedFilePath() const;  // modelfiledir + meshdir + file
  const std::string&                     get_material() const { return material_; }
  const std::vector<float>&              get_vert() const { return vert_; }
  const std::vector<float>&              get_texcoord() const { return texcoord_; }
  const std::vector<int>&                get_face() const { return face_; }
  const std::vector<std::string>&        get_bodyname() const { return bodyname_; }
  const std::vector<float>&              get_bindpos() const { return bindpos_; }
  const std::vector<float>&              get_bindquat() const { return bindquat_; }
  const std::vector<std::vector<int>>&   get_vertid() const { return vertid_; }
  const std::vector<std::vector<float>>& get_vertweight() const { return vertweight_; }
  bool HasBoneNameSpace() const { return !bone_prefix_.empty() || !bone_suffix_.empty(); }
  void del_material() {
    spec_material_.clear();
    material_.clear();
  }

  void CopyFromSpec();
  void PointToLocal();

 private:
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);
  void Compile(const mjVFS* vfs);      // compiler
  void LoadSKN(mjResource* resource);  // load skin in SKN BIN format

  // scale the compiled copy by the gauge of the bones, which must be the same for all
  void ApplyGauge();
};


//------------------------- class mjCHField --------------------------------------------------------
// Describes a height field

class mjCHField_ : public mjCBase {
 protected:
  std::vector<float> data;  // elevation data, row-major format

  std::string        file_;
  std::string        content_type_;
  std::vector<float> userdata_;
  std::string        spec_file_;
  std::string        spec_content_type_;
  std::vector<float> spec_userdata_;

 public:
  // the asset of the spec which this instance repeats at another scale, if compilation made it,
  // and the linear map, in the frame of the geoms which use this instance, of that scale
  const mjCHField*      primary_     = nullptr;
  std::array<double, 9> instancemap_ = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

class mjCHField : public mjCHField_, private mjsHField {
  friend class mjCGeom;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCHField(mjCModel* model);
  mjCHField(const mjCHField& other);
  mjCHField& operator=(const mjCHField& other);
  ~mjCHField();

  mjsHField spec;
  using mjCBase::info;

  void CopyFromSpec(void);
  void PointToLocal(void);

  std::string File() const { return file_; }
  std::string ResolvedFilePath() const;  // modelfiledir + meshdir + file

  // getter for user data
  std::vector<float>& get_userdata() { return userdata_; }

 private:
  void Compile(const mjVFS* vfs);  // compiler

  std::string GetCacheId(const mjResource* resource, const std::string& asset_type);
  void        LoadCustom(mjResource* resource);  // load from custom format
  void        LoadPNG(mjResource* resource);     // load from PNG format
};


//------------------------- class mjCTexture -------------------------------------------------------
// Describes a texture

class mjCTexture_ : public mjCBase {
 protected:
  std::vector<std::byte> data_;       // texture data (rgb, roughness, etc.)
  std::vector<std::byte> spec_data_;  // texture data given by the user

  std::string              file_;
  std::string              content_type_;
  std::vector<std::string> cubefiles_;
  std::string              spec_file_;
  std::string              spec_content_type_;
  std::vector<std::string> spec_cubefiles_;
  bool                     byref_ = false;  // compiled by reference: no payload in the model
};

class mjCTexture : public mjCTexture_, private mjsTexture {
  friend class mjCDef;
  friend class mjCModel;
  friend class mjXReader;
  friend class mjXWriter;

 public:
  explicit mjCTexture(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCTexture(const mjCTexture& other);
  mjCTexture& operator=(const mjCTexture& other);
  ~mjCTexture();

  mjsTexture spec;
  using mjCBase::info;

  void   CopyFromSpec(void);
  void   PointToLocal(void);
  void   Compile(const mjVFS* vfs);
  double texture_time_ = 0;

  // what compilation reads from the spec, names excluded: the key of mjs_dedupAssets
  std::string AssetKey() const;

  std::string              File() const { return file_; }
  std::string              ResolvedFilePath() const;  // modelfiledir + texturedir + file
  std::string              get_content_type() const { return content_type_; }
  std::vector<std::string> get_cubefiles() const { return cubefiles_; }
  bool                     ByReference() const { return byref_; }

  // decode a texture file as compilation stores it: a PNG to the given channels, a KTX as the raw
  // blob of height 1, which requires channels 1; obj is only for error messages and may be null
  static void Decode(const mjCBase*          obj,
                     mjResource*             resource,
                     const std::string&      mime_type,
                     int                     channels,
                     std::vector<std::byte>& image,
                     unsigned int&           w,
                     unsigned int&           h,
                     bool&                   is_srgb);

 private:
  // store texture into asset cache
  std::string GetCacheId(const mjResource* resource, const std::string& asset_type);
  const char* Ineligible(const mjVFS* vfs) const;              // why not by reference, or nullptr
  void        Builtin2D(void);                                 // make builtin 2D
  void        BuiltinCube(void);                               // make builtin cube
  void        Load2D(std::string filename, const mjVFS* vfs);  // load 2D from file
  void        LoadCubeSingle(std::string  filename,
                             const mjVFS* vfs);    // load cube from single file
  void        LoadCubeSeparate(const mjVFS* vfs);  // load cube from separate files

  void FlipIfNeeded(std::vector<std::byte>& image, unsigned int w, unsigned int h);

  void LoadFlip(std::string             filename,
                const mjVFS*            vfs,  // load and flip
                std::vector<std::byte>& image,
                unsigned int&           w,
                unsigned int&           h,
                bool&                   is_srgb);

  static void LoadPNG(const mjCBase*          obj,
                      mjResource*             resource,
                      int                     channels,
                      std::vector<std::byte>& image,
                      unsigned int&           w,
                      unsigned int&           h,
                      bool&                   is_srgb);
  static void LoadKTX(const mjCBase*          obj,
                      mjResource*             resource,
                      int                     channels,
                      std::vector<std::byte>& image,
                      unsigned int&           w,
                      unsigned int&           h,
                      bool&                   is_srgb);
};


//------------------------- class mjCMaterial ------------------------------------------------------
// Describes a material for rendering

class mjCMaterial_ : public mjCBase {
 protected:
  int                      texid[mjNTEXROLE];  // id of material's textures
  std::vector<std::string> textures_;
  std::vector<std::string> spec_textures_;
};

class mjCMaterial : public mjCMaterial_, private mjsMaterial {
  friend class mjCDef;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCMaterial(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCMaterial(const mjCMaterial& other);
  mjCMaterial& operator=(const mjCMaterial& other);

  mjsMaterial spec;
  using mjCBase::info;

  void CopyFromSpec();
  void PointToLocal();
  void NameSpace(const mjCModel* m);

  const std::string& get_texture(int i) const { return textures_[i]; }
  void               del_textures() {
    for (auto& t : textures_) t.clear();
    for (auto& t : spec_textures_) t.clear();
  }

 private:
  void Compile(void);  // compiler
};


//------------------------- class mjCPair ----------------------------------------------------------
// Predefined geom pair for collision detection

class mjCPair_ : public mjCBase {
 protected:
  int         signature;  // body1<<16 + body2
  std::string geomname1_;
  std::string geomname2_;
  std::string spec_geomname1_;
  std::string spec_geomname2_;
};

class mjCPair : public mjCPair_, private mjsPair {
  friend class mjCDef;
  friend class mjCBody;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCPair(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCPair(const mjCPair& other);
  mjCPair& operator=(const mjCPair& other);

  mjsPair spec;
  using mjCBase::info;

  void CopyFromSpec();
  void PointToLocal();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);

  const std::string& get_geomname1() const { return geomname1_; }
  const std::string& get_geomname2() const { return geomname2_; }

  uint32_t GetSignature(void) const { return signature; }

 private:
  void Compile(void);  // compiler

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  mjCGeom* geom1;  // geom1
  mjCGeom* geom2;  // geom2
};


//------------------------- class mjCBodyPair ------------------------------------------------------
// Body pair specification, use to exclude pairs

class mjCBodyPair_ : public mjCBase {
 protected:
  int body1;      // id of body1
  int body2;      // id of body2
  int signature;  // body1<<16 + body2

  std::string bodyname1_;  // name of geom 1
  std::string bodyname2_;  // name of geom 2
  std::string spec_bodyname1_;
  std::string spec_bodyname2_;
};

class mjCBodyPair : public mjCBodyPair_, private mjsExclude {
  friend class mjCBody;
  friend class mjCModel;

 public:
  explicit mjCBodyPair(mjCModel*);
  mjCBodyPair(const mjCBodyPair& other);
  mjCBodyPair& operator=(const mjCBodyPair& other);

  mjsExclude spec;
  using mjCBase::info;

  void CopyFromSpec();
  void PointToLocal();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);

  std::string get_bodyname1() const { return bodyname1_; }
  std::string get_bodyname2() const { return bodyname2_; }

  uint32_t GetSignature() const { return signature; }

 private:
  void Compile();  // compiler
};


//------------------------- class mjCEquality ------------------------------------------------------
// Describes an equality constraint

class mjCEquality_ : public mjCBase {
 protected:
  int         eqadr_;
  int         obj1id;
  int         obj2id;
  std::string name1_;
  std::string name2_;
  std::string spec_name1_;
  std::string spec_name2_;
};

class mjCEquality : public mjCEquality_, private mjsEquality {
  friend class mjCDef;
  friend class mjCBody;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCEquality(mjCModel* = 0, mjCDef* = 0);
  mjCEquality(const mjCEquality& other);
  mjCEquality& operator=(const mjCEquality& other);

  mjsEquality spec;
  using mjCBase::info;

  void CopyFromSpec();
  void PointToLocal();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);
  void ResetId();

 private:
  void Compile(void);  // compiler

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);
};


//------------------------- class mjCTendon --------------------------------------------------------
// Describes a tendon

class mjCTendon_ : public mjCBase {
 protected:
  int matid;        // material id for rendering
  int plasticadr_;  // address in data->plastic

  // variable used for temporarily storing the state of the tendon
  std::map<std::string, std::array<mjtNum, 2>> plastic_;  // plastic at the previous step

  // variable-size data
  std::string         material_;
  std::string         spec_material_;
  std::vector<double> userdata_;
  std::vector<double> spec_userdata_;
};

class mjCTendon : public mjCTendon_, private mjsTendon {
  friend class mjCDef;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCTendon(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCTendon(const mjCTendon& other);
  mjCTendon& operator=(const mjCTendon& other);
  ~mjCTendon();

  mjsTendon spec;
  using mjCBase::info;

  void               set_material(std::string _material) { material_ = _material; }
  const std::string& get_material() const { return material_; }
  void               del_material() {
    spec_material_.clear();
    material_.clear();
  }

  // API for adding wrapping objects
  void WrapSite(std::string wrapname, std::string_view wrapinfo = "");                    // site
  void WrapGeom(std::string wrapname, std::string side, std::string_view wrapinfo = "");  // geom
  void WrapJoint(std::string wrapname, double coef, std::string_view wrapinfo = "");      // joint
  void WrapPulley(double divisor, std::string_view wrapinfo = "");                        // pulley

  // API for access to wrapping objects
  int                   NumWraps() const;      // number of wraps
  const mjCWrap*        GetWrap(int i) const;  // pointer to wrap
  std::vector<mjCWrap*> path;                  // wrapping objects

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() const { return userdata_; }
  const double*              get_range() { return range; }

  void CopyFromSpec();
  void PointToLocal();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);
  void SetModel(mjCModel* _model);

  bool is_limited() const;
  bool is_actfrclimited() const;

  // gauge of the lowest frame or body containing the path, and 1 if the length is a length
  mjCGauge PathGauge() const;
  int      Linear() const;

  mjtNum* plastic(const std::string& state_name);
  void    ForgetKeyframes(const std::vector<std::string>& names, bool keep = false);

  // tendon type: fixed if the path is non-empty and has only joints, otherwise spatial
  mjtTendon Type() const;

 private:
  void Compile(void);  // compiler
  void ResetId();

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);
};


//------------------------- class mjCWrap ----------------------------------------------------------
// Describes a tendon wrap object

class mjCWrap_ : public mjCBase {
 public:
  int         sideid;    // side site id; -1 if not applicable
  double      prm;       // parameter: divisor, coefficient
  std::string sidesite;  // name of side site

  double prmscale = 1;  // factor of the parameter in the compiled model
};

class mjCWrap : public mjCWrap_, private mjsWrap {
  friend class mjCTendon;
  friend class mjCModel;

 public:
  mjsWrap spec;
  using mjCBase::info;

  void    CopyFromSpec();
  void    PointToLocal();
  void    ResolveReferences(const mjCModel* m);
  void    NameSpace(const mjCModel* m);
  mjtWrap Type() const { return spec.type; }

  mjCBase* obj;  // wrap object pointer

 private:
  mjCWrap(mjCModel*, mjCTendon*);            // constructor
  mjCWrap(const mjCWrap& other);             // copy constructor
  mjCWrap& operator=(const mjCWrap& other);  // copy assignment

  void Compile(void);  // compiler

  mjCTendon* tendon;  // tendon owning this wrap
};


//------------------------- class mjCPlugin --------------------------------------------------------
// Describes an instance of a plugin

class mjCPlugin_ : public mjCBase {
 public:
  int nstate;  // state size for the plugin instance
  int stateadr_;
  int statenum_;

  std::map<std::string, std::string, std::less<>> config_attribs;  // raw config attributes from XML
  std::vector<char> flattened_attributes;  // config attributes flattened in plugin-declared order;

 protected:
  std::string plugin_name;
};

class mjCPlugin : public mjCPlugin_ {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCPlugin(mjCModel*);
  mjCPlugin(const mjCPlugin& other);
  mjCPlugin& operator=(const mjCPlugin& other);

  void PointToLocal();
  void ResetId();

  mjsPlugin spec;
  mjCBase*  parent;       // parent object (only used when generating error message)
  int       plugin_slot;  // global registered slot number of the plugin

 private:
  void Compile(void);  // compiler
};


//------------------------- class mjCActuator ------------------------------------------------------
// Describes an actuator

class mjCActuator_ : public mjCBase {
 protected:
  int trnid[2];  // id of transmission target

  // variable used for temporarily storing the state of the actuator
  int     actadr_;            // address of dof in data->act
  int     actdim_;            // number of dofs in data->act
  int     ctrladr_;           // address of first control in data->ctrl
  int     ctrlnum_;           // number of controls
  int     ctrlspec_;          // resolved input signature, scoped by gaintype
  int     outadr_;            // address of first force output
  int     outnum_;            // number of force outputs, from trntype
  int     historyadr_;        // address in data->history
  int     historynum_;        // number of elements in data->history
  bool    so3_;               // compiles to an SO3 transmission
  double  ctrlranges_[4][2];  // resolved per-input control ranges
  mjtByte ctrllimiteds_[4];   // resolved per-input limited flags

  // gauge of the transmission target, and what it makes of the transmission and the control
  mjCGauge gauge_;
  int      linear_        = 0;             // 1 if the transmission length is a length
  double   ctrlscales_[4] = {1, 1, 1, 1};  // factors of the controls
  double   actscale_      = 1;             // factor of the activation

  std::map<std::string, std::vector<mjtNum>> act_;   // act at the previous step
  std::map<std::string, std::vector<mjtNum>> ctrl_;  // ctrl at the previous step

  // variable-size data
  std::string         plugin_name;
  std::string         plugin_instance_name;
  std::string         target_;
  std::string         slidersite_;
  std::string         refsite_;
  std::vector<double> userdata_;
  std::string         spec_target_;
  std::string         spec_slidersite_;
  std::string         spec_refsite_;
  std::vector<double> spec_userdata_;
};

class mjCActuator : public mjCActuator_, private mjsActuator {
  friend class mjCDef;
  friend class mjCModel;
  friend class mjCSensor;
  friend class mjXWriter;

 public:
  explicit mjCActuator(mjCModel* = nullptr, mjCDef* = nullptr);
  mjCActuator(const mjCActuator& other);
  mjCActuator& operator=(const mjCActuator& other);

  mjsActuator spec;
  using mjCBase::info;

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() const { return userdata_; }
  const std::string&         get_target() const { return spec_target_; }
  const std::string&         get_slidersite() const { return spec_slidersite_; }
  const std::string&         get_refsite() const { return spec_refsite_; }

  bool is_ctrllimited() const;
  bool is_forcelimited() const;
  bool is_actlimited() const;

  std::vector<mjtNum>& act(const std::string& state_name);
  std::vector<mjtNum>& ctrl(const std::string& state_name);

 private:
  void Compile(void);  // compiler
  void CopyFromSpec();
  void PointToLocal();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);
  void CopyPlugin();

  // scale the compiled copy by the gauge, or back
  void   ApplyGauge(bool inverse = false);
  double CoordinateFactor(const mjCGauge& g) const;

  void ForgetKeyframes(const std::vector<std::string>& names, bool keep = false);
  void ResetId();

  mjCBase* ptarget;  // transmission target
};


//------------------------- class mjCSensor --------------------------------------------------------
// Describes a sensor

class mjCSensor_ : public mjCBase {
 protected:
  int historyadr_;  // address in data->history
  int historynum_;  // number of elements in data->history

  // variable-size data
  std::string         plugin_name;
  std::string         plugin_instance_name;
  std::string         objname_;
  std::string         refname_;
  std::vector<double> userdata_;
  std::string         spec_objname_;
  std::string         spec_refname_;
  std::vector<double> spec_userdata_;
};

class mjCSensor : public mjCSensor_, private mjsSensor {
  friend class mjCDef;
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCSensor(mjCModel*);
  mjCSensor(const mjCSensor& other);
  mjCSensor& operator=(const mjCSensor& other);

  mjsSensor spec;
  using mjCBase::info;

  // used by mjXWriter and mjCModel
  const std::vector<double>& get_userdata() { return userdata_; }
  const std::string&         get_objname() { return spec_objname_; }
  const std::string&         get_refname() { return spec_refname_; }

  const mjCBase* get_obj() { return obj; }
  const mjCBase* get_ref() { return ref; }

 private:
  void Compile(void);  // compiler
  void CopyFromSpec();
  void PointToLocal();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);
  void CopyPlugin();
  void ResetId();

  // scale the compiled copy by the gauge, or back
  void ApplyGauge(bool inverse = false);

  mjCBase* obj;  // sensorized object
  mjCBase* ref;  // sensorized reference
};


//------------------------- class mjCNumeric -------------------------------------------------------
// Describes a custom data field

class mjCNumeric_ : public mjCBase {
 protected:
  std::vector<double> data_;
  std::vector<double> spec_data_;
};

class mjCNumeric : public mjCNumeric_, private mjsNumeric {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCNumeric(mjCModel*);
  mjCNumeric(const mjCNumeric& other);
  mjCNumeric& operator=(const mjCNumeric& other);
  ~mjCNumeric();

  mjsNumeric spec;
  using mjCBase::info;

  void PointToLocal();
  void CopyFromSpec();

 private:
  void Compile(void);  // compiler
};


//------------------------- class mjCText ----------------------------------------------------------
// Describes a custom text field

class mjCText_ : public mjCBase {
 protected:
  std::string data_;
  std::string spec_data_;
};

class mjCText : public mjCText_, private mjsText {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCText(mjCModel*);
  mjCText(const mjCText& other);
  mjCText& operator=(const mjCText& other);
  ~mjCText();

  mjsText spec;
  using mjCBase::info;

  void PointToLocal();
  void CopyFromSpec();

 private:
  void Compile(void);  // compiler
};


//------------------------- class mjCTuple ---------------------------------------------------------
// Describes a custom tuple field

class mjCTuple_ : public mjCBase {
 protected:
  std::vector<mjCBase*>    obj;  // object pointers
  std::vector<mjtObj>      objtype_;
  std::vector<std::string> objname_;
  std::vector<double>      objprm_;
  std::vector<mjtObj>      spec_objtype_;
  std::vector<std::string> spec_objname_;
  std::vector<double>      spec_objprm_;
};

class mjCTuple : public mjCTuple_, private mjsTuple {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCTuple(mjCModel*);
  mjCTuple(const mjCTuple& other);
  mjCTuple& operator=(const mjCTuple& other);
  ~mjCTuple();

  mjsTuple spec;
  using mjCBase::info;

  void PointToLocal();
  void CopyFromSpec();
  void ResolveReferences(const mjCModel* m);
  void NameSpace(const mjCModel* m);

 private:
  void Compile(void);  // compiler
};


//------------------------- class mjCKey -----------------------------------------------------------
// Describes a keyframe

// what a change to the tree stored of a keyframe, to be reassembled by the next compilation
typedef struct mjKeyInfo_ {
  std::string name;  // name under which the values are stored in the elements

  double time;
  bool   qpos;
  bool   qvel;
  bool   act;
  bool   ctrl;
  bool   mpos;
  bool   mquat;
  bool   plastic;
} mjKeyInfo;

class mjCKey_ : public mjCBase {
 protected:
  // a keyframe is pending from a change to the tree, when its values are stored in the elements
  // they belong to, until the next compilation reassembles its vectors
  bool      ispending_ = false;
  mjKeyInfo pending_;

  // a pending keyframe of a model to which another was attached stays in place: it keeps its
  // position among the keyframes, and it is still copied when its own model is attached; one
  // which a deletion stored, or which came with an attached model, is kept last and attached as is
  bool inplace_ = false;

  std::vector<double> qpos_;
  std::vector<double> qvel_;
  std::vector<double> act_;
  std::vector<double> mpos_;
  std::vector<double> mquat_;
  std::vector<double> ctrl_;
  std::vector<double> plastic_;
  std::vector<double> spec_qpos_;
  std::vector<double> spec_qvel_;
  std::vector<double> spec_act_;
  std::vector<double> spec_mpos_;
  std::vector<double> spec_mquat_;
  std::vector<double> spec_ctrl_;
  std::vector<double> spec_plastic_;
};

class mjCKey : public mjCKey_, private mjsKey {
  friend class mjCModel;
  friend class mjXWriter;

 public:
  explicit mjCKey(mjCModel*);
  mjCKey(const mjCKey& other);
  mjCKey& operator=(const mjCKey& other);
  ~mjCKey();

  mjsKey spec;
  using mjCBase::info;

  void PointToLocal();
  void CopyFromSpec();

 private:
  void Compile(const mjModel* m);  // compiler
};


//------------------------- class mjCDef -----------------------------------------------------------
// Describes one set of defaults

class mjCDef : public mjsElement {
  friend class mjXWriter;

 public:
  mjCDef();
  explicit mjCDef(mjCModel*);
  mjCDef(const mjCDef& other);
  mjCDef& operator=(const mjCDef& other);
  mjCDef& operator+=(const mjCDef& other);

  void CopyWithoutChildren(const mjCDef& other);
  void PointToLocal(void);
  void CopyFromSpec(void);
  void NameSpace(const mjCModel* m);

  void Compile(const mjCModel* model);

  // accessors
  mjCJoint&    Joint() { return joint_; }
  mjCGeom&     Geom() { return geom_; }
  mjCSite&     Site() { return site_; }
  mjCCamera&   Camera() { return camera_; }
  mjCLight&    Light() { return light_; }
  mjCFlex&     Flex() { return flex_; }
  mjCMesh&     Mesh() { return mesh_; }
  mjCTexture&  Texture() { return texture_; }
  mjCMaterial& Material() { return material_; }
  mjCPair&     Pair() { return pair_; }
  mjCEquality& Equality() { return equality_; }
  mjCTendon&   Tendon() { return tendon_; }
  mjCActuator& Actuator() { return actuator_; }

  // identifiers
  std::string          name;    // class name
  int                  id;      // id of this default
  mjCDef*              parent;  // id of parent class
  std::vector<mjCDef*> child;   // child classes

  mjsDefault spec;
  mjCModel*  model;  // pointer to model that owns object

  void AddRef() { ++refcount; }
  int  GetRef() { return refcount; }
  void Release() {
    if (--refcount == 0) { delete this; }
  }

 private:
  int         refcount = 1;
  mjCJoint    joint_;
  mjCGeom     geom_;
  mjCSite     site_;
  mjCCamera   camera_;
  mjCLight    light_;
  mjCFlex     flex_;
  mjCMesh     mesh_;
  mjCTexture  texture_;
  mjCMaterial material_;
  mjCPair     pair_;
  mjCEquality equality_;
  mjCTendon   tendon_;
  mjCActuator actuator_;
};

#endif  // MUJOCO_SRC_USER_USER_OBJECTS_H_
