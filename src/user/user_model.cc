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

#include "user/user_model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "cc/array_safety.h"
#include "engine/engine_core_util.h"
#include "engine/engine_forward.h"
#include "engine/engine_io.h"
#include "engine/engine_name.h"
#include "engine/engine_plugin.h"
#include "engine/engine_setconst.h"
#include "engine/engine_support.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_misc.h"
#include "user/user_api.h"
#include "user/user_objects.h"
#include "user/user_threadpool.h"
#include "user/user_util.h"

namespace {
namespace mju = ::mujoco::util;
using std::string;
using std::vector;

//---------------------------------- LOCAL UTILITY FUNCTIONS ---------------------------------------

constexpr double kFrameEps = 1e-6;  // difference below which frames are considered equal

unsigned int NumCompilerThreads(int upper_bound = -1) {
  // Use at most half the available threads to avoid
  // overloading hyperthreaded CPUs.
  // Compilation is largely compute-bound so we want to give each
  // physical core a chance without too much L1/L2 cache thrashing.
  unsigned int nthreads = std::thread::hardware_concurrency() / 2;
  if (upper_bound > 0) { nthreads = std::min(nthreads, static_cast<unsigned int>(upper_bound)); }
  return std::max(static_cast<unsigned int>(1), nthreads);
}

// detect null pose
template <typename T>
bool IsNullPose(const T pos[3], const T quat[4] = nullptr) {
  static_assert(std::is_floating_point_v<T>);
  if (std::abs(pos[0]) >= kFrameEps ||
      std::abs(pos[1]) >= kFrameEps ||
      std::abs(pos[2]) >= kFrameEps) {
    return false;
  }
  if (quat) {
    bool plus  = std::abs(quat[0] - 1) < kFrameEps &&
                 std::abs(quat[1]) < kFrameEps &&
                 std::abs(quat[2]) < kFrameEps &&
                 std::abs(quat[3]) < kFrameEps;
    bool minus = std::abs(quat[0] + 1) < kFrameEps &&
                 std::abs(quat[1]) < kFrameEps &&
                 std::abs(quat[2]) < kFrameEps &&
                 std::abs(quat[3]) < kFrameEps;
    return plus || minus;
  }
  return true;
}

// get body id from wrap object
int GetBodyIdFromWrap(const mjCWrap* wrap) {
  if (!wrap || !wrap->obj) return -1;
  switch (wrap->Type()) {
    case mjWRAP_JOINT:
      return static_cast<mjCJoint*>(wrap->obj)->GetParent()->id;
    case mjWRAP_SITE:
      return static_cast<mjCSite*>(wrap->obj)->Body()->id;
    case mjWRAP_CYLINDER:
    case mjWRAP_SPHERE:
      return static_cast<mjCGeom*>(wrap->obj)->GetParent()->id;
    default:
      return -1;
  }
}

}  // namespace

//---------------------------------- CONSTRUCTOR AND DESTRUCTOR ------------------------------------

// constructor
mjCModel::mjCModel() {
  mjs_defaultSpec(&spec);
  elemtype = mjOBJ_MODEL;
  spec_comment_.clear();
  spec_modelfiledir_.clear();
  meshdir_.clear();
  texturedir_.clear();
  spec_modelname_ = "MuJoCo Model";

  //------------------------ auto-computed statistics
#ifndef MEMORY_SANITIZER
  // initializing as best practice, but want MSAN to catch uninitialized use
  meaninertia_auto = 0;
  meanmass_auto    = 0;
  meansize_auto    = 0;
  extent_auto      = 0;
  center_auto[0] = center_auto[1] = center_auto[2] = 0;
#endif

  deepcopy_ = false;
  nplugin   = 0;
  Clear();

  //------------------------ master default set
  defaults_.push_back(new mjCDef(this));
  defaults_.back()->name = "main";

  // point to model from spec
  PointToLocal();

  // world body
  mjCBody* world = new mjCBody(this);
  mjuu_zerovec(world->pos, 3);
  mjuu_setvec(world->quat, 1, 0, 0, 0);
  world->mass = 0;
  mjuu_zerovec(world->inertia, 3);
  world->id        = 0;
  world->parent    = nullptr;
  world->weldid    = 0;
  world->name      = "world";
  world->classname = "main";
  def_map["main"]  = Default();
  bodies_.push_back(world);
  names_[mjOBJ_BODY].insert("world");

  // create mjCBase lists from children lists
  CreateObjectLists();

  // set the signature
  spec.element->signature = 0;
}


mjCModel::mjCModel(const mjCModel& other) {
  CreateObjectLists();
  *this = other;
}


mjCModel& mjCModel::operator=(const mjCModel& other) {
  deepcopy_ = true;
  if (this != &other) {
    // the copies below go through AddObject, which clears the signature
    uint64_t signature = other.spec.element->signature;

    this->spec = other.spec;

    *static_cast<mjCModel_*>(this) = static_cast<const mjCModel_&>(other);
    *static_cast<mjSpec*>(this)    = static_cast<const mjSpec&>(other);
    PointToLocal();

    // copy attached specs first so that we can resolve references to them
    for (auto* s : other.specs_) {
      specs_.push_back(s);
      static_cast<mjCModel*>(s->element)->AddRef();
      compiler2spec_[&s->compiler] = specs_.back();
    }

    // unlike an attachment, a copy leaves the original as it is
    copying_ = true;

    // its elements hold what a compilation gave the model if those of the original do, unless
    // one of them is taken again from its spec
    baseline_ = other.baseline_;

    // the world copy constructor takes care of copying the tree
    mjCBody* world = new mjCBody(*other.bodies_[0], this);
    bodies_.push_back(world);

    // update tree lists
    ResetTreeLists();
    MakeTreeLists();

    // add everything else
    *this += other;

    // add keyframes
    CopyList(keys_, other.keys_, other);
    copying_ = false;

    // create new default tree
    mjCDef* subtree = new mjCDef(*other.defaults_[0]);

    *this += *subtree;

    // copy name maps
    for (int i = 0; i < mjNOBJECT; i++) { ids[i] = other.ids[i]; }
    names_ = other.names_;

    // the copy keeps what the compilation of the original gave to it
    CopyCompiled(other);

    // the copy has the same structure as the original
    spec.element->signature = signature;
  }
  deepcopy_ = other.deepcopy_;
  return *this;
}


// return true if the references of an element, as its working copy names them, resolve in a model
static bool resolves(mjCBase* element, const mjCModel* model) {
  try {
    element->ResolveReferences(model);
  } catch (mjCError err) { return false; }
  return true;
}


// copy vector of elements from another model to this model
template <class T>
void mjCModel::CopyList(std::vector<T*>&       dest,
                        const std::vector<T*>& source,
                        const mjCModel&        other) {
  // give an element of the other model its name in this model and find the objects it references;
  // a copy of a compiled model keeps an element as the compilation left it, if the references
  // which the compilation resolved are found in the copy; otherwise the element is taken as its
  // spec has it
  auto resolve = [this](T* element, mjCModel* source_model) {
    element->model = this;
    if (copying_ && compiled && resolves(element, this)) { return; }
    if (copying_) { baseline_ = false; }
    element->NameSpace(source_model);
    element->CopyFromSpec();
    element->ResolveReferences(this);
  };

  // loop over the elements from the other model
  int nsource = (int)source.size();
  for (int i = 0; i < nsource; i++) {
    mjCModel* source_model = source[i]->model;

    // an element which an earlier attachment moved by reference is no longer the other model's
    if (!copying_ && source_model != &other) { continue; }

    // try to find the referenced objects in this model on a copy, so that an element moved by
    // reference stays as it is in the other model if they are not found
    T* candidate = new T(*source[i]);
    try {
      resolve(candidate, source_model);
    } catch (mjCError err) {
      // if not present, skip the element
      // TODO: do not skip elements that contain user errors
      candidate->model = nullptr;
      delete candidate;
      continue;
    }
    if (!deepcopy_) {
      candidate->model = nullptr;
      delete candidate;
      candidate = source[i];
      resolve(candidate, source_model);
    }

    // copy the element from the other model to this model; an attached element takes the keyframe
    // values which were stored for it, except those of the keyframes which stay in place
    if (!copying_) { candidate->ForgetKeyframes(source_model->inplacekeys_); }
    if (deepcopy_) {
      if (!copying_) { source[i]->ForgetKeyframes(source_model->inplacekeys_, /*keep=*/true); }
    } else {
      // a moved element forgets its addresses in the model it comes from
      candidate->ResetId();
      candidate->AddRef();
    }

    // a copy of the model keeps what the compilation gave to the element
    if (copying_) { CopyCompiled(candidate, source[i]); }
    mjSpec* origin = FindSpec(source[i]->compiler);
    dest.push_back(candidate);
    dest.back()->model    = this;
    dest.back()->compiler = origin ? &origin->compiler : &spec.compiler;
    dest.back()->id       = -1;
    dest.back()->CopyPlugin();
  }
  if (!dest.empty()) { ProcessList_(ids, dest, dest[0]->elemtype); }
}


template <class T>
static void resetlist(std::vector<T*>& list) {
  for (auto element : list) { element->id = -1; }
  list.clear();
}


// elements of a list in the order of their ids
template <class T>
static std::vector<T*> orderbyid(const std::vector<T*>& list) {
  std::vector<T*> ordered(list.size());
  for (T* element : list) { ordered[element->id] = element; }
  return ordered;
}


void mjCModel::ResetTreeLists() {
  mjCBody* world = bodies_[0];
  resetlist(bodies_);
  resetlist(joints_);
  resetlist(geoms_);
  resetlist(sites_);
  resetlist(cameras_);
  resetlist(lights_);
  resetlist(frames_);
  world->id = 0;
  bodies_.push_back(world);
}


// save associated state addresses in related elements
void mjCModel::SaveDofOffsets(bool computesize) {
  int qposadr    = 0;
  int dofadr     = 0;
  int actadr     = 0;
  int ctrladr    = 0;
  int outadr     = 0;
  int mocapadr   = 0;
  int plasticadr = 0;

  for (auto joint : joints_) {
    joint->qposadr_     = qposadr;
    joint->dofadr_      = dofadr;
    joint->plasticadr_  = joint->spec.springyield > 0 ? plasticadr : -1;
    qposadr            += joint->nq();
    dofadr             += joint->nv();
    if (joint->spec.springyield > 0) { plasticadr += joint->nq(); }
  }

  for (auto tendon : tendons_) {
    tendon->plasticadr_ = tendon->spec.springyield > 0 ? plasticadr : -1;
    if (tendon->spec.springyield > 0) { plasticadr += 2; }
  }

  for (auto flex : flexes_) {
    int pnum           = flex->PlasticNum();
    flex->plasticnum_  = pnum;
    flex->plasticadr_  = pnum > 0 ? plasticadr : -1;
    plasticadr        += pnum;
  }

  for (auto actuator : actuators_) {
    if (actuator->actdim > 0) {
      actuator->actdim_ = actuator->actdim;
    } else if (actuator->spec.actdim > 0) {
      actuator->actdim_ = actuator->spec.actdim;
    } else {
      actuator->actdim_ = (actuator->spec.dyntype != mjDYN_NONE);
    }
    actuator->actadr_  = actuator->actdim_ ? actadr : -1;
    actadr            += actuator->actdim_;

    // input and output blocks
    actuator->ctrladr_  = actuator->ctrlnum_ ? ctrladr : -1;
    ctrladr            += actuator->ctrlnum_;
    actuator->outadr_   = outadr;
    outadr             += actuator->outnum_;
  }

  for (int i = 0; i < (int)bodies_.size(); i++) {
    mjCBody* body  = bodies_[i];
    body->bodyadr_ = i;
    if (body->spec.mocap) {
      body->mocapid = mocapadr++;
    } else {
      body->mocapid = -1;
    }
  }

  for (int i = 0; i < (int)equalities_.size(); i++) { equalities_[i]->eqadr_ = i; }

  if (computesize) {
    nq        = qposadr;
    nv        = dofadr;
    na        = actadr;
    nu        = ctrladr;
    nactuator = (int)actuators_.size();
    nout      = outadr;
    nmocap    = mocapadr;
    nplastic  = plasticadr;
  }
}


template <class T>
void mjCModel::CopyExplicitPlugin(T* obj) {
  if (!obj->plugin.active || !obj->plugin_instance_name.empty() || !obj->spec.plugin.element) {
    return;
  }
  mjCPlugin* origin    = static_cast<mjCPlugin*>(obj->spec.plugin.element);
  mjCPlugin* candidate = deepcopy_ ? new mjCPlugin(*origin) : origin;
  if (!deepcopy_) {
    // a moved plugin forgets its state address in the model it comes from
    candidate->ResetId();
    candidate->AddRef();
  }
  candidate->id    = plugins_.size();
  candidate->model = this;
  if (copying_) { CopyCompiled(candidate, origin); }
  plugins_.push_back(candidate);
  obj->spec.plugin.element = candidate;
}

template void mjCModel::CopyExplicitPlugin<mjCBody>(mjCBody* obj);
template void mjCModel::CopyExplicitPlugin<mjCGeom>(mjCGeom* obj);
template void mjCModel::CopyExplicitPlugin<mjCMesh>(mjCMesh* obj);
template void mjCModel::CopyExplicitPlugin<mjCActuator>(mjCActuator* obj);
template void mjCModel::CopyExplicitPlugin<mjCSensor>(mjCSensor* obj);


template <class T>
void mjCModel::CopyPlugin(const std::vector<mjCPlugin*>& source, const std::vector<T*>& list) {
  // store elements that reference a plugin instance
  std::unordered_map<std::string, T*> instances;
  for (const auto& element : list) {
    if (!element->plugin_instance_name.empty()) {
      instances[element->plugin_instance_name] = element;
    }
  }

  // only copy plugins that are referenced
  for (const auto& plugin : source) {
    if (plugin->name.empty() && plugin->model == this) { continue; }
    mjCPlugin* candidate = new mjCPlugin(*plugin);
    candidate->model     = this;
    candidate->NameSpace(plugin->model);
    bool referenced = instances.find(candidate->name) != instances.end();
    auto same_name  = [candidate](const mjCPlugin* dest) { return dest->name == candidate->name; };
    bool instance_exists =
        std::find_if(plugins_.begin(), plugins_.end(), same_name) != plugins_.end();
    if (referenced && !instance_exists) {
      if (copying_) { CopyCompiled(candidate, plugin); }
      plugins_.push_back(candidate);
      instances.at(candidate->name)->spec.plugin.element = candidate;
    } else {
      delete candidate;
    }
  }

  // update other elements in the list in case of multiple references
  for (auto& element : list) {
    if (!element->plugin_instance_name.empty()) {
      element->spec.plugin.element =
          instances.at(element->plugin_instance_name)->spec.plugin.element;
    }
  }
}


// return true if the plugin is already in the list of active plugins
static bool IsPluginActive(const mjpPlugin*                                     plugin,
                           const std::vector<std::pair<const mjpPlugin*, int>>& active_plugins) {
  return std::find_if(active_plugins.begin(),
                      active_plugins.end(),
                      [&plugin](const std::pair<const mjpPlugin*, int>& element) {
                        return element.first == plugin;
                      }) != active_plugins.end();
}


mjCModel& mjCModel::operator+=(const mjCModel& other) {
  // create global lists
  ResetTreeLists();
  MakeTreeLists();
  ProcessLists(/*checkrepeat=*/false);

  // copy all elements not in the tree
  if (this != &other) {
    // do not copy assets for self-attach
    // TODO: asset should be copied only when referenced
    CopyList(meshes_, other.meshes_, other);
    CopyList(skins_, other.skins_, other);
    CopyList(hfields_, other.hfields_, other);
    CopyList(textures_, other.textures_, other);
    CopyList(materials_, other.materials_, other);
    CopyList(numerics_, other.numerics_, other);
    CopyList(texts_, other.texts_, other);
  }
  CopyList(flexes_, other.flexes_, other);
  CopyList(pairs_, other.pairs_, other);
  CopyList(excludes_, other.excludes_, other);
  CopyList(tendons_, other.tendons_, other);
  CopyList(equalities_, other.equalities_, other);
  CopyList(actuators_, other.actuators_, other);
  CopyList(sensors_, other.sensors_, other);
  CopyList(tuples_, other.tuples_, other);

  // create new plugins and map them
  CopyPlugin(other.plugins_, bodies_);
  CopyPlugin(other.plugins_, geoms_);
  CopyPlugin(other.plugins_, meshes_);
  CopyPlugin(other.plugins_, actuators_);
  CopyPlugin(other.plugins_, sensors_);
  for (const auto& [plugin, slot] : other.active_plugins_) {
    if (!IsPluginActive(plugin, active_plugins_)) {
      active_plugins_.emplace_back(std::make_pair(plugin, slot));
    }
  }

  // update pointers to local elements
  PointToLocal();

  // reprocess lists to ensure ordering matches compiled model after attach
  ProcessLists(/*checkrepeat=*/false);

  // structure changed, the signature is no longer valid
  InvalidateSignature();
  return *this;
}


template <class T>
void mjCModel::RemoveFromList(std::vector<T*>& list, const mjCModel& other) {
  int nlist   = (int)list.size();
  int removed = 0;
  for (int i = 0; i < nlist; i++) {
    T* element   = list[i];
    element->id -= removed;
    try {
      // check if the element contains an error
      element->NameSpace(&other);
      element->CopyFromSpec();
      element->ResolveReferences(&other);
    } catch (mjCError err) { continue; }
    try {
      // check if the element references something that was removed
      element->NameSpace(this);
      element->CopyFromSpec();
      element->ResolveReferences(this);
    } catch (mjCError err) {
      ids[element->elemtype].erase(element->name);
      names_[element->elemtype].erase(element->name);
      element->id = -1;
      element->Release();
      list.erase(list.begin() + i);
      nlist--;
      i--;
      removed++;
    }
  }
  if (removed > 0 && !list.empty()) {
    // if any elements were removed, update ids using processlist
    ProcessList_(ids, list, list[0]->elemtype, /*checkrepeat=*/false);
  }
}


// return the plugin instances that elements or defaults reference by name or point to, including
// the elements in the subtree of body, which can be outside the tree
std::unordered_set<const mjsElement*> mjCModel::ReferencedPlugins(const mjCBody* body) {
  std::unordered_set<std::string>       names;
  std::unordered_set<const mjsElement*> pointers;

  auto mark = [&names, &pointers](const auto& element) {
    if (!element.plugin_instance_name.empty()) { names.insert(element.plugin_instance_name); }
    if (element.spec.plugin.element) { pointers.insert(element.spec.plugin.element); }
  };

  // traverse the tree, the tree lists miss the elements added since they were built
  std::vector<const mjCBody*> bodies = {bodies_[0]};
  if (body) { bodies.push_back(body); }
  for (int i = 0; i < bodies.size(); i++) {
    mark(*bodies[i]);
    for (const mjCGeom* geom : bodies[i]->geoms) { mark(*geom); }
    bodies.insert(bodies.end(), bodies[i]->bodies.begin(), bodies[i]->bodies.end());
  }
  for (const mjCMesh* mesh : meshes_) { mark(*mesh); }
  for (const mjCActuator* actuator : actuators_) { mark(*actuator); }
  for (const mjCSensor* sensor : sensors_) { mark(*sensor); }
  for (mjCDef* def : defaults_) {
    mark(def->Geom());
    mark(def->Mesh());
    mark(def->Actuator());
  }

  std::unordered_set<const mjsElement*> referenced;
  for (const mjCPlugin* plugin : plugins_) {
    if (pointers.count(plugin) || names.count(plugin->name)) { referenced.insert(plugin); }
  }
  return referenced;
}


void mjCModel::RemovePlugins(const std::unordered_set<const mjsElement*>& referenced) {
  std::unordered_set<const mjsElement*> remaining = ReferencedPlugins();

  // remove plugins that are no longer referenced, keep plugins that were never referenced
  int nlist   = (int)plugins_.size();
  int removed = 0;
  for (int i = 0; i < nlist; i++) {
    plugins_[i]->id -= removed;
    if (referenced.count(plugins_[i]) && !remaining.count(plugins_[i])) {
      ids[plugins_[i]->elemtype].erase(plugins_[i]->name);
      names_[plugins_[i]->elemtype].erase(plugins_[i]->name);
      plugins_[i]->id = -1;
      plugins_[i]->Release();
      plugins_.erase(plugins_.begin() + i);
      nlist--;
      i--;
      removed++;
    }
  }

  // if any elements were removed, update ids using processlist
  if (removed > 0 && !plugins_.empty()) {
    ProcessList_(ids, plugins_, plugins_[0]->elemtype, /*checkrepeat=*/false);
  }
}


// return the body that owns the frame, nullptr if the frame is not in the tree
mjCBody* mjCModel::FrameOwner(const mjCFrame& frame, mjCBody* body) {
  if (body == nullptr) { body = bodies_[0]; }

  if (std::find(body->frames.begin(), body->frames.end(), &frame) != body->frames.end()) {
    return body;
  }

  // recursive call to all child bodies
  for (mjCBody* child : body->bodies) {
    mjCBody* owner = FrameOwner(frame, child);
    if (owner) { return owner; }
  }
  return nullptr;
}


// remove the elements of list that are inside frame from the list and return them
template <class T>
static std::vector<T*> RemoveFromFrame(std::vector<T*>& list, const mjCFrame& frame) {
  auto inside = std::stable_partition(list.begin(), list.end(), [&frame](const T* element) {
    return !frame.IsAncestor(element->frame);
  });
  std::vector<T*> removed(inside, list.end());
  list.erase(inside, list.end());
  return removed;
}


// remove body from tree, the body and its contents are released by the caller
std::vector<mjCBase*> mjCModel::RemoveFromTree(const mjCBody& subtree) {
  mjCBody* world  = bodies_[0];
  *world         -= subtree;
  return {};
}


// remove frame from tree together with the elements inside it, which are returned
std::vector<mjCBase*> mjCModel::RemoveFromTree(const mjCFrame& frame) {
  mjCBody* body = FrameOwner(frame);

  // the frame is released by the caller, the elements inside it once the lists are rebuilt
  auto      found = std::find(body->frames.begin(), body->frames.end(), &frame);
  mjCFrame* self  = *found;
  body->frames.erase(found);
  std::vector<mjCBody*>   bodies  = RemoveFromFrame(body->bodies, frame);
  std::vector<mjCGeom*>   geoms   = RemoveFromFrame(body->geoms, frame);
  std::vector<mjCFrame*>  frames  = RemoveFromFrame(body->frames, frame);
  std::vector<mjCJoint*>  joints  = RemoveFromFrame(body->joints, frame);
  std::vector<mjCSite*>   sites   = RemoveFromFrame(body->sites, frame);
  std::vector<mjCCamera*> cameras = RemoveFromFrame(body->cameras, frame);
  std::vector<mjCLight*>  lights  = RemoveFromFrame(body->lights, frame);

  // an inertial element inside the frame is deleted as well
  if (frame.IsAncestor(body->iframe)) {
    mjsBody defaults;
    mjs_defaultBody(&defaults);
    body->iframe                = nullptr;
    body->explicitinertial      = defaults.explicitinertial;  // for XML writer
    body->spec.explicitinertial = defaults.explicitinertial;
    body->spec.mass             = defaults.mass;
    body->spec.ialt             = defaults.ialt;
    mjuu_copyvec(body->spec.ipos, defaults.ipos, 3);
    mjuu_copyvec(body->spec.iquat, defaults.iquat, 4);
    mjuu_copyvec(body->spec.inertia, defaults.inertia, 3);
    mjuu_copyvec(body->spec.fullinertia, defaults.fullinertia, 6);
  }

  // the removed elements no longer have a parent body or a frame
  self->SetParent(nullptr);
  self->frame = nullptr;
  std::vector<mjCBase*> removed;

  auto orphan = [&removed](auto& list) {
    for (auto* element : list) {
      element->SetParent(nullptr);
      element->frame = nullptr;
      removed.push_back(element);
    }
  };
  orphan(bodies);
  orphan(geoms);
  orphan(frames);
  orphan(joints);
  orphan(sites);
  orphan(cameras);
  orphan(lights);
  return removed;
}


// remove subtree from the tree, then remove all elements that reference it
template <class T>
mjCModel& mjCModel::RemoveSubtree(const T& subtree) {
  mjCModel oldmodel(*this);

  // create global lists in the old model, the name maps copied from a compiled model can be stale
  oldmodel.ProcessLists(/*checkrepeat=*/false);

  // create global lists in this model if not compiled
  if (!IsCompiled()) { ProcessLists(/*checkrepeat=*/false); }

  // all keyframes are now pending, the next compilation reassembles them
  StoreKeyframes(this);

  // remove subtree from tree
  std::vector<mjCBase*> removed = RemoveFromTree(subtree);

  // update global lists
  ResetTreeLists();
  MakeTreeLists();
  ProcessLists(/*checkrepeat=*/false);

  // check if we have to remove anything else
  RemoveFromList(pairs_, oldmodel);
  RemoveFromList(excludes_, oldmodel);
  RemoveFromList(tendons_, oldmodel);
  RemoveFromList(equalities_, oldmodel);
  RemoveFromList(actuators_, oldmodel);
  RemoveFromList(sensors_, oldmodel);

  // structure changed, the signature is no longer valid
  InvalidateSignature();

  // the lists no longer point to the elements removed along with the subtree
  for (mjCBase* element : removed) { element->Release(); }

  return *this;
}


mjCModel& mjCModel::operator-=(const mjCBody& subtree) {
  return RemoveSubtree(subtree);
}


mjCModel& mjCModel::operator-=(const mjCFrame& frame) {
  if (!FrameOwner(frame)) { throw mjCError(nullptr, "frame is not in this model"); }
  return RemoveSubtree(frame);
}


// add default tree to this model
mjCModel& mjCModel::operator+=(mjCDef& subtree) {
  defaults_.push_back(&subtree);
  def_map[subtree.name] = &subtree;
  subtree.model         = this;

  // set parent to the main default if this is not the only default in the model
  if (!subtree.parent && &subtree != defaults_[0]) {
    subtree.parent = defaults_[0];
    defaults_[0]->child.push_back(&subtree);
  }

  for (auto def : subtree.child) {
    *this += *def;  // triggers recursive call
  }
  return *this;
}


// remove default class from array
mjCModel& mjCModel::operator-=(const mjCDef& subtree) {
  // check that the default belongs to this model before modifying anything
  if (subtree.model != this) { throw mjCError(nullptr, "default is not in this model"); }

  // check we aren't trying to remove the 'main' default
  if (subtree.id == 0) { throw mjCError(0, "cannot remove the global default ('main')"); }

  // remove this default from parent's child list
  mjCDef* parent = subtree.parent;
  if (parent) {
    for (int i = 0; i < parent->child.size(); ++i) {
      if (parent->child[i] == &subtree) {
        parent->child.erase(parent->child.begin() + i);
        break;
      }
    }
  }

  // traverse tree to find all descendants starting from subtree.id
  std::vector<int> default_ids_to_remove;
  std::vector<int> stack;
  stack.push_back(subtree.id);
  while (!stack.empty()) {
    int id = stack.back();
    stack.pop_back();
    default_ids_to_remove.push_back(id);
    for (int i = 0; i < defaults_[id]->child.size(); i++) {
      stack.push_back(defaults_[id]->child[i]->id);
    }
  }

  // remove from the tree
  std::sort(default_ids_to_remove.begin(), default_ids_to_remove.end(), std::greater<int>());

  for (int id : default_ids_to_remove) {
    def_map.erase(defaults_[id]->name);
    defaults_[id]->id = -1;
    defaults_[id]->Release();
    defaults_.erase(defaults_.begin() + id);
  }

  // reset default ids
  for (int i = 0; i < defaults_.size(); ++i) { defaults_[i]->id = i; }

  return *this;
}


template <class T>
void deletefromlist(std::vector<T*>* list, mjsElement* element) {
  if (!list) { return; }
  for (int j = 0; j < list->size(); ++j) {
    list->at(j)->id = -1;
    if (list->at(j) == element) {
      list->erase(list->begin() + j);
      j--;
    }
  }
}


// delete the compiled instances of an asset
template <class T>
static void DeleteInstances(std::vector<T*>& instances, const T* primary) {
  auto of = [primary](T* instance) {
    if (instance->primary_ != primary) { return false; }
    delete instance;
    return true;
  };
  instances.erase(std::remove_if(instances.begin(), instances.end(), of), instances.end());
}


// remove the element from the model
void mjCModel::operator-=(mjsElement* el) {
  if (el->elemtype != mjOBJ_DEFAULT) {
    if (static_cast<mjCBase*>(el)->model != this) {
      throw mjCError(nullptr, "element is not in this model");
    }
  } else {
    if (static_cast<mjCDef*>(el)->model != this) {
      throw mjCError(nullptr, "default is not in this model");
    }
  }

  // plugin instances referenced before the deletion, mjs_bodyToFrame removes the body from the tree
  mjCBody* body = el->elemtype == mjOBJ_BODY ? static_cast<mjCBody*>(el) : nullptr;
  std::unordered_set<const mjsElement*> referenced = ReferencedPlugins(body);

  if (body) { *this -= *body; }

  // throws before anything is modified if the frame is not in the tree
  if (el->elemtype == mjOBJ_FRAME) {
    mjCFrame* frame  = static_cast<mjCFrame*>(el);
    *this           -= *frame;
  }

  ResetTreeLists();

  switch (el->elemtype) {
    case mjOBJ_BODY:
      break;  // removed above

    case mjOBJ_FRAME:
      break;  // removed above, frames are meta elements and have no object list

    case mjOBJ_DEFAULT:
      MakeTreeLists();  // rebuild lists that were reset at the beginning of the function
      throw mjCError(nullptr, "defaults cannot be deleted, use detach instead");
      break;

    case mjOBJ_GEOM:
      deletefromlist(&(static_cast<mjCGeom*>(el)->body->geoms), el);
      break;

    case mjOBJ_SITE:
      deletefromlist(&(static_cast<mjCSite*>(el)->body->sites), el);
      break;

    case mjOBJ_JOINT:
      deletefromlist(&(static_cast<mjCJoint*>(el)->body->joints), el);
      break;

    case mjOBJ_LIGHT:
      deletefromlist(&(static_cast<mjCLight*>(el)->body->lights), el);
      break;

    case mjOBJ_CAMERA:
      deletefromlist(&(static_cast<mjCCamera*>(el)->body->cameras), el);
      break;

    case mjOBJ_MESH:
      DeleteInstances(meshinstances_, static_cast<mjCMesh*>(el));
      deletefromlist(object_lists_[el->elemtype], el);
      break;

    case mjOBJ_HFIELD:
      DeleteInstances(hfieldinstances_, static_cast<mjCHField*>(el));
      deletefromlist(object_lists_[el->elemtype], el);
      break;

    default:
      deletefromlist(object_lists_[el->elemtype], el);
      break;
  }

  MakeTreeLists();
  ProcessLists(/*checkrepeat=*/false);

  // delete the plugin instances that only the removed elements referenced
  RemovePlugins(referenced);

  // structure changed, the signature is no longer valid
  InvalidateSignature();

  static_cast<mjCBase*>(el)->Release();
}


// TODO: we should not use C-type casting with multiple C++ inheritance
void mjCModel::CreateObjectLists() {
  for (int i = 0; i < mjNOBJECT; ++i) { object_lists_[i] = nullptr; }

  object_lists_[mjOBJ_BODY]     = (std::vector<mjCBase*>*)&bodies_;
  object_lists_[mjOBJ_XBODY]    = (std::vector<mjCBase*>*)&bodies_;
  object_lists_[mjOBJ_JOINT]    = (std::vector<mjCBase*>*)&joints_;
  object_lists_[mjOBJ_GEOM]     = (std::vector<mjCBase*>*)&geoms_;
  object_lists_[mjOBJ_SITE]     = (std::vector<mjCBase*>*)&sites_;
  object_lists_[mjOBJ_CAMERA]   = (std::vector<mjCBase*>*)&cameras_;
  object_lists_[mjOBJ_LIGHT]    = (std::vector<mjCBase*>*)&lights_;
  object_lists_[mjOBJ_FLEX]     = (std::vector<mjCBase*>*)&flexes_;
  object_lists_[mjOBJ_MESH]     = (std::vector<mjCBase*>*)&meshes_;
  object_lists_[mjOBJ_SKIN]     = (std::vector<mjCBase*>*)&skins_;
  object_lists_[mjOBJ_HFIELD]   = (std::vector<mjCBase*>*)&hfields_;
  object_lists_[mjOBJ_TEXTURE]  = (std::vector<mjCBase*>*)&textures_;
  object_lists_[mjOBJ_MATERIAL] = (std::vector<mjCBase*>*)&materials_;
  object_lists_[mjOBJ_PAIR]     = (std::vector<mjCBase*>*)&pairs_;
  object_lists_[mjOBJ_EXCLUDE]  = (std::vector<mjCBase*>*)&excludes_;
  object_lists_[mjOBJ_EQUALITY] = (std::vector<mjCBase*>*)&equalities_;
  object_lists_[mjOBJ_TENDON]   = (std::vector<mjCBase*>*)&tendons_;
  object_lists_[mjOBJ_ACTUATOR] = (std::vector<mjCBase*>*)&actuators_;
  object_lists_[mjOBJ_SENSOR]   = (std::vector<mjCBase*>*)&sensors_;
  object_lists_[mjOBJ_NUMERIC]  = (std::vector<mjCBase*>*)&numerics_;
  object_lists_[mjOBJ_TEXT]     = (std::vector<mjCBase*>*)&texts_;
  object_lists_[mjOBJ_TUPLE]    = (std::vector<mjCBase*>*)&tuples_;
  object_lists_[mjOBJ_KEY]      = (std::vector<mjCBase*>*)&keys_;
  object_lists_[mjOBJ_PLUGIN]   = (std::vector<mjCBase*>*)&plugins_;
}


void mjCModel::PointToLocal() {
  spec.element             = static_cast<mjsElement*>(this);
  spec.comment             = &spec_comment_;
  spec.modelfiledir        = &spec_modelfiledir_;
  spec.modelname           = &spec_modelname_;
  spec.compiler.meshdir    = &meshdir_;
  spec.compiler.texturedir = &texturedir_;
  comment                  = nullptr;
  modelfiledir             = nullptr;
  modelname                = nullptr;
}


void mjCModel::CopyFromSpec() {
  *static_cast<mjSpec*>(this) = spec;

  comment_      = spec_comment_;
  modelfiledir_ = spec_modelfiledir_;
  modelname_    = spec_modelname_;
}


// compute sparse matrix sizes
void mjCModel::ComputeSparseSizes() {
  // no dofs, quick return
  if (nv == 0) {
    nM = nD = nB = nC = 0;
    nJten             = 0;
    return;
  }

  // 0. allocate local index vectors
  std::vector<int> dof_parentid_pre(nv, -1);
  std::vector<int> dof_bodyid_pre(nv);
  std::vector<int> body_simple_pre(nbody);
  std::vector<int> body_rootid_pre(nbody);
  std::vector<int> dof_simplenum_pre(nv);
  std::vector<int> body_lastdof_map(nbody);

  // 1. build dof_parentid, dof_bodyid
  if (nbody > 0) {
    body_lastdof_map[0] = -1;  // world has no parent dof
  }
  for (int i = 0; i < nbody; ++i) {
    mjCBody* pb  = bodies_[i];
    mjCBody* par = pb->parent;

    // the last dof of the current body's parent chain
    int current_parent_dof = par ? body_lastdof_map[par->id] : -1;

    for (const auto* jnt : pb->joints) {
      for (int j1 = 0; j1 < jnt->nv(); ++j1) {
        int dofadr = jnt->dofadr_ + j1;
        if (dofadr < 0 || dofadr >= nv) {
          throw mjCError(jnt, "dofadr out of bounds: dofadr=%d, nv=%d", nullptr, dofadr, nv);
        }
        dof_bodyid_pre[dofadr]   = i;
        dof_parentid_pre[dofadr] = current_parent_dof;
        // the next dof in this joint is parented to the current one
        current_parent_dof = dofadr;
      }
    }
    // store the last dof added for this body
    body_lastdof_map[i] = current_parent_dof;
  }

  // 2. compute nM
  nM = 0;
  for (int i = 0; i < nv; ++i) {
    int j = i;
    while (j != -1) {
      nM++;
      j = dof_parentid_pre[j];
    }
  }

  // 3. compute nD
  nD = 2 * nM - nv;

  // 4. compute subtreedofs and nB
  for (int i = nbody - 1; i >= 0; --i) {
    bodies_[i]->subtreedofs = bodies_[i]->dofnum;
    for (const auto* child : bodies_[i]->Bodies()) {
      bodies_[i]->subtreedofs += child->subtreedofs;
    }
  }

  nB = 0;
  for (int i = 0; i < nbody; ++i) {
    nB              += bodies_[i]->subtreedofs;
    mjCBody* parent  = bodies_[i]->parent;
    while (parent && parent->id > 0) {
      nB     += parent->dofnum;
      parent  = parent->parent;
    }
  }

  // make sure all dofs are in world "subtree", SHOULD NOT OCCUR
  if (bodies_[0]->subtreedofs != nv) { throw mjCError(0, "all DOFs should be in world subtree"); }

  // 5. compute nC
  for (int i = 0; i < nbody; ++i) {
    mjCBody* pb       = bodies_[i];
    mjCBody* par      = pb->parent;
    int      parentid = par ? par->id : 0;

    // rootid
    if (i == 0 || !par || par->id == 0) {
      body_rootid_pre[i] = i;
    } else {
      body_rootid_pre[i] = body_rootid_pre[parentid];
    }

    bool sameframe = IsNullPose(pb->ipos, pb->iquat);
    body_simple_pre[i] =
        (sameframe && (body_rootid_pre[i] == i || (parentid > 0 &&
                                                   bodies_[parentid]->parent &&
                                                   bodies_[parentid]->parent->id == 0 &&
                                                   bodies_[parentid]->dofnum == 0)));

    // user override: disable simple optimization
    if (!pb->simple) { body_simple_pre[i] = 0; }
  }

  // a parent body is never simple (unless world)
  for (int i = 1; i < nbody; ++i) {
    if (bodies_[i]->parent) { body_simple_pre[bodies_[i]->parent->id] = 0; }
  }

  // joint-based demotion for body_simple_pre
  for (int i = 1; i < nbody; ++i) {
    if (!body_simple_pre[i]) continue;

    // demote if non-aligned, non-zero pos, or multiple rotational joints
    mjCBody* pb       = bodies_[i];
    int      rotfound = 0;
    for (const auto* pj : pb->joints) {
      bool axis_aligned = ((std::abs(pj->axis[0]) > mjEPS) +
                           (std::abs(pj->axis[1]) > mjEPS) +
                           (std::abs(pj->axis[2]) > mjEPS)) == 1;
      if (rotfound ||
          !IsNullPose(pj->pos) ||
          ((pj->type == mjJNT_HINGE || pj->type == mjJNT_SLIDE) && !axis_aligned)) {
        body_simple_pre[i] = 0;
        break;
      }
      if (pj->type == mjJNT_BALL || pj->type == mjJNT_HINGE) { rotfound = 1; }
    }
    if (!body_simple_pre[i]) continue;

    // promote simple bodies with only sliders to level 2
    if (pb->dofnum > 0) {
      body_simple_pre[i] = 2;
      for (const auto* pj : pb->joints) {
        if (pj->type != mjJNT_SLIDE) {
          body_simple_pre[i] = 1;
          break;
        }
      }
    }
  }

  // tendon-armature-based demotion, including the armature of actuators on the tendon
  for (const auto* tendon : tendons_) {
    bool armature = tendon->armature > 0;
    for (const auto* actuator : actuators_) {
      if (actuator->trntype == mjTRN_TENDON &&
          actuator->trnid[0] == tendon->id &&
          actuator->armature != 0) {
        armature = true;
      }
    }
    if (armature) {
      for (const auto* wrap : tendon->path) {
        int bodyId = GetBodyIdFromWrap(wrap);
        if (bodyId != -1) { body_simple_pre[bodyId] = 0; }
      }
    }
  }

  // count dof_simplenum_pre
  int count = 0;
  for (int i = nv - 1; i >= 0; --i) {
    if (dof_bodyid_pre[i] < 0 || dof_bodyid_pre[i] >= nbody) {
      throw mjCError(0, "dof_bodyid out of bounds: dof=%d, body=%d", nullptr, i, dof_bodyid_pre[i]);
    }
    if (body_simple_pre[dof_bodyid_pre[i]]) {
      count++;
    } else {
      count = 0;
    }
    dof_simplenum_pre[i] = count;
  }

  // compute nC
  nC      = 0;
  int nOD = 0;
  for (int i = 0; i < nv; ++i) {
    if (!dof_simplenum_pre[i]) {
      int j = i;
      while (j != -1) {
        if (j != i) nOD++;
        j = dof_parentid_pre[j];
      }
    }
  }
  nC = nOD + nv;

  nJten = 0;
  if (nv > 0) {
    std::vector<bool> dof_bitmap(nv, false);
    for (const auto* tendon : tendons_) {
      if (tendon->Type() == mjTENDON_FIXED) {
        nJten += tendon->path.size();
        continue;
      }

      std::fill(dof_bitmap.begin(), dof_bitmap.end(), false);
      for (const auto* wrap : tendon->path) {
        int bodyid = GetBodyIdFromWrap(wrap);
        if (bodyid > 0) {
          mjCBody* b = bodies_[bodyid];
          while (b && b->id > 0) {
            for (const auto* jnt : b->joints) {
              for (int k = 0; k < jnt->nv(); k++) { dof_bitmap[jnt->dofadr_ + k] = true; }
            }
            b = b->GetParent();
          }
        }
      }
      for (int j = 0; j < nv; j++) { nJten += dof_bitmap[j]; }
    }
  }
}


// destructor
mjCModel::~mjCModel() {
  // do not rebuild lists if we are in the process of deleting the model
  compiled = false;

  // delete kinematic tree and all objects allocated in it
  bodies_[0]->Release();

  // delete objects allocated in mjCModel
  for (int i = 0; i < flexes_.size(); i++) flexes_[i]->Release();
  for (int i = 0; i < meshes_.size(); i++) meshes_[i]->Release();
  for (int i = 0; i < skins_.size(); i++) skins_[i]->Release();
  for (int i = 0; i < hfields_.size(); i++) hfields_[i]->Release();
  for (mjCMesh* mesh : meshinstances_) delete mesh;
  for (mjCHField* hfield : hfieldinstances_) delete hfield;
  for (int i = 0; i < textures_.size(); i++) textures_[i]->Release();
  for (int i = 0; i < materials_.size(); i++) materials_[i]->Release();
  for (int i = 0; i < pairs_.size(); i++) pairs_[i]->Release();
  for (int i = 0; i < excludes_.size(); i++) excludes_[i]->Release();
  for (int i = 0; i < equalities_.size(); i++) equalities_[i]->Release();
  for (int i = 0; i < tendons_.size(); i++) tendons_[i]->Release();  // also deletes wraps
  for (int i = 0; i < actuators_.size(); i++) actuators_[i]->Release();
  for (int i = 0; i < sensors_.size(); i++) sensors_[i]->Release();
  for (int i = 0; i < numerics_.size(); i++) numerics_[i]->Release();
  for (int i = 0; i < texts_.size(); i++) texts_[i]->Release();
  for (int i = 0; i < tuples_.size(); i++) tuples_[i]->Release();
  for (int i = 0; i < keys_.size(); i++) keys_[i]->Release();
  for (int i = 0; i < defaults_.size(); i++) defaults_[i]->Release();
  for (int i = 0; i < specs_.size(); i++) mj_deleteSpec(specs_[i]);
  for (int i = 0; i < plugins_.size(); i++) plugins_[i]->Release();

  // clear sizes and pointer lists created in Compile
  Clear();
}


// clear objects allocated by Compile
void mjCModel::Clear() {
  // sizes set from list lengths
  nbody       = 0;
  nbvh        = 0;
  nbvhstatic  = 0;
  nbvhdynamic = 0;
  noct        = 0;
  njnt        = 0;
  ngeom       = 0;
  nsite       = 0;
  ncam        = 0;
  nlight      = 0;
  nflex       = 0;
  nmesh       = 0;
  nskin       = 0;
  nhfield     = 0;
  ntex        = 0;
  nmat        = 0;
  npair       = 0;
  nexclude    = 0;
  neq         = 0;
  ntendon     = 0;
  nsensor     = 0;
  nnumeric    = 0;
  ntext       = 0;

  // sizes set by Compile
  nq             = 0;
  nv             = 0;
  nu             = 0;
  nactuator      = 0;
  nout           = 0;
  na             = 0;
  nflexnode      = 0;
  nflexvert      = 0;
  nflexedge      = 0;
  nflexelem      = 0;
  nflexelemdata  = 0;
  nflexstiffness = 0;
  nflexbending   = 0;
  nefm0dof       = 0;
  nefm0L         = 0;
  nefmCvert      = 0;
  nflexelemedge  = 0;
  nflexshelldata = 0;
  nflextexcoord  = 0;
  nJfe           = 0;
  nmeshvert      = 0;
  nmeshnormal    = 0;
  nmeshtexcoord  = 0;
  nmeshface      = 0;
  nmeshgraph     = 0;
  nmeshpoly      = 0;
  nmeshpolyvert  = 0;
  nmeshpolymap   = 0;
  nskinvert      = 0;
  nskintexvert   = 0;
  nskinface      = 0;
  nskinbone      = 0;
  nskinbonevert  = 0;
  nhfielddata    = 0;
  ntexdata       = 0;
  nwrap          = 0;
  nsensordata    = 0;
  nhistory       = 0;
  nplastic       = 0;
  nnumericdata   = 0;
  ntextdata      = 0;
  ntupledata     = 0;
  npluginattr    = 0;
  nnames         = 0;
  npaths         = 0;
  memory         = -1;
  nstack         = -1;
  nemax          = 0;
  nM             = 0;
  nD             = 0;
  nB             = 0;
  nJmom          = 0;
  njmax          = -1;
  nconmax        = -1;
  nmocap         = 0;

  // internal variables
  hasImplicitPluginElem = false;
  compiled              = false;
  errInfo               = mjCError();
  ClearCompileWarnings();
  qpos0.clear();
}


//------------------------ API FOR ADDING MODEL ELEMENTS -------------------------------------------

// add object of any type
template <class T>
T* mjCModel::AddObject(vector<T*>& list, string type) {
  T* obj  = new T(this);
  obj->id = (int)list.size();
  list.push_back(obj);
  InvalidateSignature();
  return obj;
}


// add object of any type, with default parameter
template <class T>
T* mjCModel::AddObjectDefault(vector<T*>& list, string type, mjCDef* def) {
  T* obj         = new T(this, def ? def : defaults_[0]);
  obj->id        = (int)list.size();
  obj->classname = def ? def->name : "main";
  list.push_back(obj);
  InvalidateSignature();
  return obj;
}


// add flex
mjCFlex* mjCModel::AddFlex() {
  return AddObject(flexes_, "flex");
}


// add mesh
mjCMesh* mjCModel::AddMesh(mjCDef* def) {
  return AddObjectDefault(meshes_, "mesh", def);
}


// add skin
mjCSkin* mjCModel::AddSkin() {
  return AddObject(skins_, "skin");
}


// add hfield
mjCHField* mjCModel::AddHField() {
  return AddObject(hfields_, "hfield");
}


// add texture
mjCTexture* mjCModel::AddTexture(mjCDef* def) {
  return AddObjectDefault(textures_, "texture", def);
}


// add material
mjCMaterial* mjCModel::AddMaterial(mjCDef* def) {
  return AddObjectDefault(materials_, "material", def);
}


// add geom pair to include in collisions
mjCPair* mjCModel::AddPair(mjCDef* def) {
  return AddObjectDefault(pairs_, "pair", def);
}


// add body pair to exclude from collisions
mjCBodyPair* mjCModel::AddExclude() {
  return AddObject(excludes_, "exclude");
}


// add constraint
mjCEquality* mjCModel::AddEquality(mjCDef* def) {
  return AddObjectDefault(equalities_, "equality", def);
}


// add tendon
mjCTendon* mjCModel::AddTendon(mjCDef* def) {
  return AddObjectDefault(tendons_, "tendon", def);
}


// add actuator
mjCActuator* mjCModel::AddActuator(mjCDef* def) {
  return AddObjectDefault(actuators_, "actuator", def);
}


// add sensor
mjCSensor* mjCModel::AddSensor() {
  return AddObject(sensors_, "sensor");
}


// add custom
mjCNumeric* mjCModel::AddNumeric() {
  return AddObject(numerics_, "numeric");
}


// add text
mjCText* mjCModel::AddText() {
  return AddObject(texts_, "text");
}


// add tuple
mjCTuple* mjCModel::AddTuple() {
  return AddObject(tuples_, "tuple");
}


// add keyframe
mjCKey* mjCModel::AddKey() {
  mjCKey* key = AddObject(keys_, "key");

  // pending keyframes which do not stay in place are kept last, which is where the compilation
  // that resolved them used to create them: place the new keyframe before them
  auto pending = std::find_if(keys_.begin(), keys_.end(), [](const mjCKey* k) {
    return k->ispending_ && !k->inplace_;
  });
  if (pending != keys_.end()) {
    std::rotate(pending, keys_.end() - 1, keys_.end());
    for (int i = 0; i < (int)keys_.size(); i++) { keys_[i]->id = i; }

    // the map from names to positions is out of date: names are searched for in the list
    // until the lists are processed again
    ids[mjOBJ_KEY].clear();
  }
  return key;
}


// add keyframe which is pending, after all others
mjCKey* mjCModel::AddPendingKey(const std::string& name, const mjKeyInfo& info) {
  mjCKey* key     = AddObject(keys_, "key");
  key->name       = name;
  key->spec.time  = info.time;
  key->ispending_ = true;
  key->pending_   = info;
  return key;
}


// add plugin instance
mjCPlugin* mjCModel::AddPlugin() {
  return AddObject(plugins_, "plugin");
}


// append spec to spec
void mjCModel::AppendSpec(mjSpec* spec, const mjsCompiler* compiler_) {
  // TODO: check if the spec is already in the list
  specs_.push_back(spec);

  if (compiler_) { compiler2spec_[compiler_] = spec; }
}


//------------------------ API FOR ACCESS TO MODEL ELEMENTS  ---------------------------------------

// get number of objects of specified type
int mjCModel::NumObjects(mjtObj type) {
  if (!object_lists_[type]) { return 0; }
  return (int)object_lists_[type]->size();
}


// get pointer to specified object
mjCBase* mjCModel::GetObject(mjtObj type, int id) {
  if (id < 0 || id >= NumObjects(type)) { return nullptr; }
  return (*object_lists_[type])[id];
}


template <class T>
static mjsElement* GetNext(const std::vector<T*>& list, const mjsElement* child) {
  if (!child) {
    if (list.empty()) { return nullptr; }
    return list[0]->spec.element;
  }

  // use id for direct indexing if valid
  int id = static_cast<const mjCBase*>(child)->id;
  if (id >= 0 && id < (int)list.size() && list[id]->spec.element == child) {
    if (id + 1 < (int)list.size()) { return list[id + 1]->spec.element; }
    return nullptr;
  }

  // fallback to linear search if id is stale or invalid
  for (int i = 0; i < (int)list.size() - 1; i++) {
    if (list[i]->spec.element == child) { return list[i + 1]->spec.element; }
  }
  return nullptr;
}


// next object of specified type
mjsElement* mjCModel::NextObject(const mjsElement* object, mjtObj type) const {
  if (type == mjOBJ_UNKNOWN) {
    if (!object) {
      throw mjCError(nullptr, "type must be specified if no element is given");
    } else {
      type = object->elemtype;
    }
  } else if (object && object->elemtype != type) {
    throw mjCError(nullptr, "element is not of requested type");
  }

  switch (type) {
    case mjOBJ_BODY:
      if (!object) {
        return bodies_[0]->spec.element;
      } else if (object == bodies_[0]->spec.element) {
        return bodies_[0]->NextChild(NULL, type, /*recursive=*/true);
      } else {
        return bodies_[0]->NextChild(object, type, /*recursive=*/true);
      }
    case mjOBJ_SITE:
    case mjOBJ_GEOM:
    case mjOBJ_JOINT:
    case mjOBJ_CAMERA:
    case mjOBJ_LIGHT:
    case mjOBJ_FRAME:
      return bodies_[0]->NextChild(object, type, /*recursive=*/true);
    case mjOBJ_ACTUATOR:
      return GetNext(actuators_, object);
    case mjOBJ_SENSOR:
      return GetNext(sensors_, object);
    case mjOBJ_FLEX:
      return GetNext(flexes_, object);
    case mjOBJ_PAIR:
      return GetNext(pairs_, object);
    case mjOBJ_EXCLUDE:
      return GetNext(excludes_, object);
    case mjOBJ_EQUALITY:
      return GetNext(equalities_, object);
    case mjOBJ_TENDON:
      return GetNext(tendons_, object);
    case mjOBJ_NUMERIC:
      return GetNext(numerics_, object);
    case mjOBJ_TEXT:
      return GetNext(texts_, object);
    case mjOBJ_TUPLE:
      return GetNext(tuples_, object);
    case mjOBJ_KEY:
      return GetNext(keys_, object);
    case mjOBJ_MESH:
      return GetNext(meshes_, object);
    case mjOBJ_HFIELD:
      return GetNext(hfields_, object);
    case mjOBJ_SKIN:
      return GetNext(skins_, object);
    case mjOBJ_TEXTURE:
      return GetNext(textures_, object);
    case mjOBJ_MATERIAL:
      return GetNext(materials_, object);
    case mjOBJ_PLUGIN:
      return GetNext(plugins_, object);
    default:
      return nullptr;
  }
}


//------------------------ API FOR ACCESS TO PRIVATE VARIABLES -------------------------------------

// compiled flag
bool mjCModel::IsCompiled() const {
  return compiled;
}


// get reference of error object
const mjCError& mjCModel::GetError() const {
  return errInfo;
}

// add warning to vector (immediate delivery outside compile)
void mjCModel::AddWarning(std::string msg, const mjCBase* obj) {
  if (obj) {
    msg += "\nElement name '" + obj->name + "', id " + std::to_string(obj->id);
    if (!obj->info.empty()) { msg += ", " + obj->info; }
  }

  // outside compile: deliver immediately via normal handler chain
  if (!compiling_) { mju_warning("%s", msg.c_str()); }
  warnings_.push_back(std::move(msg));
}

// add grouped warning with subject/body split (immediate delivery outside
// compile)
void mjCModel::AddGroupedWarning(const std::string& subject, const std::string& body) {
  std::string full = body.empty() ? subject : subject + "\n" + body;
  warnings_.push_back(full);

  // outside compile: deliver immediately via structured log message
  if (!compiling_) {
    mjLogMessage m = {.level = mjLOG_WARNING};
    snprintf(m.subject, sizeof(m.subject), "%s", subject.c_str());
    m.body = body.empty() ? nullptr : body.c_str();
    mju_message(&m);
  }
}

// pointer to world body
mjCBody* mjCModel::GetWorld() {
  return bodies_[0];
}


// find default class name in array
mjCDef* mjCModel::FindDefault(const string& name) const {
  for (int i = 0; i < (int)defaults_.size(); i++) {
    if (defaults_[i]->name == name) { return defaults_[i]; }
  }
  return nullptr;
}


// add default class to array
mjCDef* mjCModel::AddDefault(string name, mjCDef* parent) {
  // check for repeated name
  int thisid = (int)defaults_.size();
  for (int i = 0; i < thisid; i++) {
    if (defaults_[i]->name == name) { return 0; }
  }

  // create new object
  mjCDef* def = new mjCDef(parent->model);
  defaults_.push_back(def);
  def->id = thisid;

  // initialize contents
  if (parent && parent->id < thisid) {
    parent->CopyFromSpec();
    def->CopyWithoutChildren(*parent);
    parent->child.push_back(def);
  }
  def->parent = parent;
  def->name   = name;
  def->child.clear();
  def_map[name] = def;

  return def;
}


// find object by name in given list
template <class T>
static T* findobject(std::string_view name, const vector<T*>& list, const mjKeyMap& ids) {
  // this can occur in the URDF parser
  if (ids.empty()) {
    for (unsigned int i = 0; i < list.size(); i++) {
      if (list[i]->name == name) { return list[i]; }
    }
    return nullptr;
  }

  // during model compilation
  auto id = ids.find(name);
  if (id == ids.end()) { return nullptr; }
  if (id->second > (int)list.size() - 1) { throw mjCError(0, "object not found"); }
  return list[id->second];
}


// find object in global lists by searching for its name, without the name maps
mjCBase* mjCModel::SearchObject(mjtObj type, std::string_view name) const {
  if (type < 0 || type >= mjNOBJECT || !object_lists_[type]) { return nullptr; }
  for (mjCBase* object : *object_lists_[type]) {
    if (object->name == name) { return object; }
  }
  return nullptr;
}


// find object in global lists given string type and name
mjCBase* mjCModel::FindObject(mjtObj type, string name) const {
  if (!object_lists_[type]) { return nullptr; }
  return findobject(name, *object_lists_[type], ids[type]);
}


// find body by name
mjCBase* mjCModel::FindTree(mjCBody* body, mjtObj type, std::string name) {
  switch (type) {
    case mjOBJ_BODY:
      if (body->name == name) { return body; }
      break;
    case mjOBJ_SITE:
      for (auto site : body->sites) {
        if (site->name == name) { return site; }
      }
      break;
    case mjOBJ_GEOM:
      for (auto geom : body->geoms) {
        if (geom->name == name) { return geom; }
      }
      break;
    case mjOBJ_JOINT:
      for (auto joint : body->joints) {
        if (joint->name == name) { return joint; }
      }
      break;
    case mjOBJ_CAMERA:
      for (auto camera : body->cameras) {
        if (camera->name == name) { return camera; }
      }
      break;
    case mjOBJ_LIGHT:
      for (auto light : body->lights) {
        if (light->name == name) { return light; }
      }
      break;
    case mjOBJ_FRAME:
      for (auto frame : body->frames) {
        if (frame->name == name) { return frame; }
      }
      break;
    default:
      return nullptr;
  }

  for (auto child : body->bodies) {
    auto candidate = FindTree(child, type, name);
    if (candidate) { return candidate; }
  }

  return nullptr;
}


// find spec by name
mjSpec* mjCModel::FindSpec(std::string name) const {
  for (auto spec : specs_) {
    if (mjs_getString(spec->modelname) == name) { return spec; }
  }
  return nullptr;
}


// find spec by mjsCompiler pointer
mjSpec* mjCModel::FindSpec(const mjsCompiler* compiler_) const {
  if (compiler_ == &spec.compiler) { return &const_cast<mjCModel*>(this)->spec; }

  if (auto it = compiler2spec_.find(compiler_); it != compiler2spec_.end()) { return it->second; }

  for (auto s : specs_) {
    mjSpec* source = static_cast<mjCModel*>(s->element)->FindSpec(compiler_);
    if (source) { return source; }
  }
  return nullptr;
}


//------------------------------- COMPILER PHASES --------------------------------------------------

// make lists of objects in tree: bodies, geoms, joints, sites, cameras, lights
void mjCModel::MakeTreeLists(mjCBody* body) {
  if (body == nullptr) { body = bodies_[0]; }

  // add this body if not world
  if (body != bodies_[0]) { bodies_.push_back(body); }

  // add body's geoms, joints, sites, cameras, lights
  for (mjCGeom* geom : body->geoms) geoms_.push_back(geom);
  for (mjCJoint* joint : body->joints) joints_.push_back(joint);
  for (mjCSite* site : body->sites) sites_.push_back(site);
  for (mjCCamera* camera : body->cameras) cameras_.push_back(camera);
  for (mjCLight* light : body->lights) lights_.push_back(light);
  for (mjCFrame* frame : body->frames) frames_.push_back(frame);

  // recursive call to all child bodies
  for (mjCBody* body : body->bodies) MakeTreeLists(body);
}


// set nuser fields
void mjCModel::SetNuser() {
  if (nuser_body == -1) {
    nuser_body = 0;
    for (int i = 0; i < bodies_.size(); i++) {
      nuser_body = mjMAX(nuser_body, bodies_[i]->spec_userdata_.size());
    }
  }
  if (nuser_jnt == -1) {
    nuser_jnt = 0;
    for (int i = 0; i < joints_.size(); i++) {
      nuser_jnt = mjMAX(nuser_jnt, joints_[i]->spec_userdata_.size());
    }
  }
  if (nuser_geom == -1) {
    nuser_geom = 0;
    for (int i = 0; i < geoms_.size(); i++) {
      nuser_geom = mjMAX(nuser_geom, geoms_[i]->spec_userdata_.size());
    }
  }
  if (nuser_site == -1) {
    nuser_site = 0;
    for (int i = 0; i < sites_.size(); i++) {
      nuser_site = mjMAX(nuser_site, sites_[i]->spec_userdata_.size());
    }
  }
  if (nuser_cam == -1) {
    nuser_cam = 0;
    for (int i = 0; i < cameras_.size(); i++) {
      nuser_cam = mjMAX(nuser_cam, cameras_[i]->spec_userdata_.size());
    }
  }
  if (nuser_tendon == -1) {
    nuser_tendon = 0;
    for (int i = 0; i < tendons_.size(); i++) {
      nuser_tendon = mjMAX(nuser_tendon, tendons_[i]->spec_userdata_.size());
    }
  }
  if (nuser_actuator == -1) {
    nuser_actuator = 0;
    for (int i = 0; i < actuators_.size(); i++) {
      nuser_actuator = mjMAX(nuser_actuator, actuators_[i]->spec_userdata_.size());
    }
  }
  if (nuser_sensor == -1) {
    nuser_sensor = 0;
    for (int i = 0; i < sensors_.size(); i++) {
      nuser_sensor = mjMAX(nuser_sensor, sensors_[i]->spec_userdata_.size());
    }
  }
}

// index assets
void mjCModel::IndexAssets() {
  // assets referenced in geoms
  for (int i = 0; i < geoms_.size(); i++) {
    mjCGeom* geom = geoms_[i];

    // a reference which was removed since the last compilation resolves to nothing
    geom->mesh   = nullptr;
    geom->hfield = nullptr;
    geom->matid  = -1;

    // find mesh by name
    if (!geom->get_meshname().empty()) {
      mjCMesh* mesh = static_cast<mjCMesh*>(FindObject(mjOBJ_MESH, geom->get_meshname()));
      if (mesh) {
        geom->mesh = mesh;
        if (geom->spec.type == mjGEOM_SDF) { mesh->SetNeedSDF(true); }
      } else {
        throw mjCError(geom, "mesh '%s' not found in geom %d", geom->get_meshname().c_str(), i);
      }
    }

    // find material by name, this has to happen after mesh assignment so that if
    // the geom does not specify a material but the mesh does it can fall back.
    if (!geom->get_material().empty()) {
      mjCBase* material = FindObject(mjOBJ_MATERIAL, geom->get_material());
      if (material) {
        geom->matid = material->id;
      } else {
        throw mjCError(geom, "material '%s' not found in geom %d", geom->get_material().c_str(), i);
      }
    }

    // find hfield by name
    if (!geom->get_hfieldname().empty()) {
      mjCBase* hfield = FindObject(mjOBJ_HFIELD, geom->get_hfieldname());
      if (hfield) {
        geom->hfield = (mjCHField*)hfield;
      } else {
        throw mjCError(geom, "hfield '%s' not found in geom %d", geom->get_hfieldname().c_str(), i);
      }
    }
  }

  // assets referenced in skins
  for (int i = 0; i < skins_.size(); i++) {
    mjCSkin* skin = skins_[i];
    skin->matid   = -1;

    // find material by name
    if (!skin->material_.empty()) {
      mjCBase* material = FindObject(mjOBJ_MATERIAL, skin->material_);
      if (material) {
        skin->matid = material->id;
      } else {
        throw mjCError(skin, "material '%s' not found in skin %d", skin->material_.c_str(), i);
      }
    }
  }

  // materials and meshes referenced in sites
  for (int i = 0; i < sites_.size(); i++) {
    mjCSite* site = sites_[i];
    site->mesh    = nullptr;
    site->matid   = -1;

    // find mesh by name
    if (!site->get_meshname().empty()) {
      mjCMesh* mesh = static_cast<mjCMesh*>(FindObject(mjOBJ_MESH, site->get_meshname()));
      if (mesh) {
        site->mesh = mesh;
      } else {
        throw mjCError(site, "mesh '%s' not found in site %d", site->get_meshname().c_str(), i);
      }
    }

    // find material by name
    if (!site->get_material().empty()) {
      mjCBase* material = FindObject(mjOBJ_MATERIAL, site->get_material());
      if (material) {
        site->matid = material->id;
      } else {
        throw mjCError(site, "material '%s' not found in site %d", site->get_material().c_str(), i);
      }
    }
  }

  // materials referenced in tendons
  for (int i = 0; i < tendons_.size(); i++) {
    mjCTendon* tendon = tendons_[i];
    tendon->matid     = -1;

    // find material by name
    if (!tendon->material_.empty()) {
      mjCBase* material = FindObject(mjOBJ_MATERIAL, tendon->material_);
      if (material) {
        tendon->matid = material->id;
      } else {
        throw mjCError(tendon,
                       "material '%s' not found in tendon %d",
                       tendon->material_.c_str(),
                       i);
      }
    }
  }

  // textures referenced in materials
  for (int i = 0; i < materials_.size(); i++) {
    mjCMaterial* material = materials_[i];

    // find textures by name, as the spec has them now
    material->CopyFromSpec();
    for (int j = 0; j < mjNTEXROLE; j++) {
      material->texid[j] = -1;
      if (!material->textures_[j].empty()) {
        mjCBase* texture = FindObject(mjOBJ_TEXTURE, material->textures_[j]);
        if (texture) {
          material->texid[j] = texture->id;
        } else {
          throw mjCError(material,
                         "texture '%s' not found in material %d",
                         material->textures_[j].c_str(),
                         i);
        }
      }
    }
  }
}


// the meshes and height fields as compiled: those of the spec, then the instances which the
// compilation made of those used at several scales
std::vector<mjCMesh*> mjCModel::CompiledMeshes() const {
  std::vector<mjCMesh*> meshes = meshes_;
  meshes.insert(meshes.end(), meshinstances_.begin(), meshinstances_.end());
  return meshes;
}


std::vector<mjCHField*> mjCModel::CompiledHFields() const {
  std::vector<mjCHField*> hfields = hfields_;
  hfields.insert(hfields.end(), hfieldinstances_.begin(), hfieldinstances_.end());
  return hfields;
}


// give each use of an asset the instance of the asset at the scale of the use: the asset itself at
// the scale of its first use, and instances named after it at the others
using InstanceMap = std::array<double, 9>;

template <class T>
static void Instantiate(const mjCModel*                                 model,
                        mjtObj                                          type,
                        std::vector<T*>&                                assets,
                        std::vector<T*>&                                instances,
                        const std::vector<std::pair<T**, InstanceMap>>& uses,
                        bool                                            reuse) {
  if (!reuse) {
    for (T* instance : instances) { delete instance; }
    instances.clear();
    std::unordered_map<const T*, InstanceMap> first;
    for (const auto& [use, scale] : uses) { first.emplace(*use, scale); }
    for (T* asset : assets) {
      auto        it    = first.find(asset);
      InstanceMap scale = it == first.end() ? InstanceMap{1, 0, 0, 0, 1, 0, 0, 0, 1} : it->second;
      if (asset->instancemap_ != scale) {
        asset->instancemap_ = scale;
        asset->CopyFromSpec();
      }
    }
  }

  for (const auto& use_scale : uses) {
    T**                use   = use_scale.first;
    const InstanceMap& scale = use_scale.second;
    T*                 asset = *use;
    if (asset->instancemap_ == scale) { continue; }
    auto same = [asset, &scale](const T* instance) {
      return instance->primary_ == asset && instance->instancemap_ == scale;
    };
    auto it = std::find_if(instances.begin(), instances.end(), same);
    if (it == instances.end()) {
      if (reuse) { throw mjCError(asset, "no instance of this asset at the scale of its use"); }
      auto        of   = [asset](const T* instance) { return instance->primary_ == asset; };
      std::string name = asset->name +
                         "@" +
                         std::to_string(1 + std::count_if(instances.begin(), instances.end(), of));
      if (model->FindObject(type, name)) {
        throw mjCError(asset, "the name '%s' of an instance of this asset is taken", name.c_str());
      }
      T* instance            = new T(*asset);
      instance->model        = asset->model;
      instance->compiler     = asset->compiler;
      instance->name         = name;
      instance->primary_     = asset;
      instance->instancemap_ = scale;
      instance->id           = assets.size() + instances.size();
      instance->CopyFromSpec();
      instances.push_back(instance);
      it = instances.end() - 1;
    }
    *use = *it;
  }

  // reused instances which no use refers to anymore are dropped
  if (reuse) {
    std::unordered_set<const T*> used;
    for (const auto& [use, scale] : uses) { used.insert(*use); }
    instances.erase(std::remove_if(instances.begin(),
                                   instances.end(),
                                   [&used](T* instance) {
                                     if (used.count(instance)) { return false; }
                                     delete instance;
                                     return true;
                                   }),
                    instances.end());
  }
  for (int i = 0; i < instances.size(); i++) { instances[i]->id = assets.size() + i; }
}


// give the geoms and sites which use a mesh or a height field the instance of the asset at the
// scale of their use, in the order of their ids, made again unless the assets are reused
void mjCModel::MakeInstances(bool reuse) {
  std::vector<std::pair<mjCMesh**, InstanceMap>>   meshuses;
  std::vector<std::pair<mjCHField**, InstanceMap>> hfielduses;
  for (mjCGeom* geom : geoms_) {
    if (geom->mesh) { meshuses.emplace_back(&geom->mesh, geom->ShapeMap()); }
    if (geom->hfield) { hfielduses.emplace_back(&geom->hfield, geom->ShapeMap()); }
  }
  for (mjCSite* site : sites_) {
    if (site->mesh) { meshuses.emplace_back(&site->mesh, site->ShapeMap()); }
  }
  Instantiate(this, mjOBJ_MESH, meshes_, meshinstances_, meshuses, reuse);
  Instantiate(this, mjOBJ_HFIELD, hfields_, hfieldinstances_, hfielduses, reuse);

  // the instances which sdf geoms use
  for (mjCMesh* mesh : CompiledMeshes()) { mesh->SetNeedSDF(false); }
  for (mjCGeom* geom : geoms_) {
    if (geom->mesh && geom->spec.type == mjGEOM_SDF) { geom->mesh->SetNeedSDF(true); }
  }
}


// error for an asset without a name; only the XML parser names an asset after its file
static mjCError EmptyName(const mjCBase* asset, const char* type, const std::string& file) {
  if (file.empty()) { return mjCError(asset, "empty name in %s", type); }
  std::string msg = std::string(type) +
                    " with file '" +
                    file +
                    "' has no name: an asset is named after its file only by the XML parser, " +
                    "set its name explicitly";
  return mjCError(asset, "%s", msg.c_str());
}


// throw error if a name is missing
void mjCModel::CheckEmptyNames(void) {
  // meshes
  for (int i = 0; i < meshes_.size(); i++) {
    if (meshes_[i]->name.empty()) { throw EmptyName(meshes_[i], "mesh", meshes_[i]->spec_file_); }
  }

  // hfields
  for (int i = 0; i < hfields_.size(); i++) {
    if (hfields_[i]->name.empty()) {
      throw EmptyName(hfields_[i], "height field", hfields_[i]->spec_file_);
    }
  }

  // textures
  for (int i = 0; i < textures_.size(); i++) {
    if (textures_[i]->name.empty() && textures_[i]->type != mjTEXTURE_SKYBOX) {
      throw EmptyName(textures_[i], "texture", textures_[i]->spec_file_);
    }
  }

  // materials
  for (int i = 0; i < materials_.size(); i++) {
    if (materials_[i]->name.empty()) { throw mjCError(materials_[i], "empty name in material"); }
  }
}


// total length of the resolved paths of the assets in a list which have a file
template <typename T>
static size_t getpathslength(std::vector<T> list) {
  size_t result = 0;
  for (const auto& element : list) {
    if (!element->File().empty()) { result += element->ResolvedFilePath().length() + 1; }
  }

  return result;
}

// set array sizes
void mjCModel::SetSizes() {
  std::vector<mjCMesh*>   meshes  = CompiledMeshes();
  std::vector<mjCHField*> hfields = CompiledHFields();
  // set from object list sizes
  nbody    = (int)bodies_.size();
  njnt     = (int)joints_.size();
  ngeom    = (int)geoms_.size();
  nsite    = (int)sites_.size();
  ncam     = (int)cameras_.size();
  nlight   = (int)lights_.size();
  nflex    = (int)flexes_.size();
  nmesh    = (int)meshes.size();
  nskin    = (int)skins_.size();
  nhfield  = (int)hfields.size();
  ntex     = (int)textures_.size();
  nmat     = (int)materials_.size();
  npair    = (int)pairs_.size();
  nexclude = (int)excludes_.size();
  neq      = (int)equalities_.size();
  ntendon  = (int)tendons_.size();
  nsensor  = (int)sensors_.size();
  nnumeric = (int)numerics_.size();
  ntext    = (int)texts_.size();
  ntuple   = (int)tuples_.size();
  nkey     = (int)keys_.size();
  nplugin  = (int)plugins_.size();
  nq = nv = ntree = nu = nactuator = nout = na = nmocap = 0;

  // nq, nv, ntree
  for (int i = 0; i < njnt; i++) {
    nq += joints_[i]->nq();
    nv += joints_[i]->nv();

    // increment ntree if this is the first joint in a moving body with static ancestry
    mjCBody* parent         = joints_[i]->GetParent();
    bool     is_first_joint = joints_[i] == parent->joints[0];
    if (is_first_joint) {
      // check if all ancestors are static
      bool     static_ancestry = true;
      mjCBody* ancestor        = parent;
      while (ancestor != bodies_[0]) {
        ancestor = ancestor->GetParent();
        if (!ancestor->joints.empty()) {
          static_ancestry = false;
          break;
        }
      }

      // if all ancestors are static, this joint starts a new kinematic tree
      if (static_ancestry) { ntree++; }
    }
  }

  // nu, nactuator, nout, na; all actuator types are currently 1x1
  for (int i = 0; i < actuators_.size(); i++) {
    nactuator++;
    nu   += actuators_[i]->ctrlnum_;
    nout += actuators_[i]->outnum_;
    na   += actuators_[i]->actdim;
  }

  // nbvh, nbvhstatic, nbvhdynamic
  for (int i = 0; i < nbody; i++) { nbvhstatic += bodies_[i]->tree.Nbvh(); }
  for (int i = 0; i < nmesh; i++) {
    nbvhstatic += meshes[i]->tree().Nbvh();
    noct       += meshes[i]->octree().NumNodes();
  }
  for (int i = 0; i < nflex; i++) { nbvhdynamic += flexes_[i]->tree.Nbvh(); }
  nbvh = nbvhstatic + nbvhdynamic;

  // flex counts
  for (int i = 0; i < nflex; i++) {
    nflexnode      += flexes_[i]->nnode;
    nflexvert      += flexes_[i]->nvert;
    nflexedge      += flexes_[i]->nedge;
    nflexelem      += flexes_[i]->nelem;
    nflexelemdata  += flexes_[i]->nelem * (flexes_[i]->dim + 1);
    nflexelemedge  += flexes_[i]->nelem * mjCFlex::kNumEdges[flexes_[i]->dim - 1];
    nflexshelldata += (int)flexes_[i]->shell.size();
    nflextexcoord  += (flexes_[i]->HasTexcoord() ? flexes_[i]->get_texcoord().size() / 2 : 0);
    nflexstiffness += flexes_[i]->stiffness.size();
    nflexbending   += flexes_[i]->bending.size();
    if (flexes_[i]->interpolated || flexes_[i]->rigid) { continue; }


    // count number of non-zero elements in the edge Jacobian matrix
    for (const auto& edge : flexes_[i]->edge) {
      mjCBody* b1 = bodies_[flexes_[i]->vertbodyid[edge.first]];
      mjCBody* b2 = bodies_[flexes_[i]->vertbodyid[edge.second]];

      std::unordered_set<mjCBody*> bodies_in_jac;
      while (b1 || b2) {
        if (b1) {
          bodies_in_jac.insert(b1);
          b1 = b1->parent;
        }
        if (b2) {
          bodies_in_jac.insert(b2);
          b2 = b2->parent;
        }
      }
      for (mjCBody* b : bodies_in_jac) { nJfe += b->dofnum; }
    }
  }

  // bending factor sizes: symbolic reverse-Cholesky count on the (M + K_bend) pattern.
  // Each flap couples full 3x3 blocks: its vertex bodies can have different orientations.
  // The count must match the symbolic factorization performed in mj_setConst (asserted there).
  std::vector<int> body_slot(bodies_.size(), -1);
  for (int i = 0; i < nflex; i++) {
    if (flexes_[i]->interpolated || flexes_[i]->rigid || !flexes_[i]->IsSimple()) { continue; }
    if (flexes_[i]->dim == 2 && !flexes_[i]->bending.empty()) {
      const mjCFlex* fl = flexes_[i];
      for (int v = 0; v < fl->nvert; v++) {
        int wid = bodies_[fl->vertbodyid[v]]->weldid;
        if (bodies_[wid]->dofnum == 3) { body_slot[wid] = 1; }
      }
    }
  }
  int nfree = 0;
  for (int b = 0; b < (int)bodies_.size(); b++) {
    if (body_slot[b] > 0) { body_slot[b] = nfree++; }
  }
  if (nfree) {
    // vertex adjacency from 4-vertex flap stencils across all qualifying flexes
    std::vector<std::set<int>> adj(nfree);
    for (int i = 0; i < nflex; i++) {
      if (flexes_[i]->interpolated || flexes_[i]->rigid || !flexes_[i]->IsSimple()) { continue; }
      if (flexes_[i]->dim == 2 && !flexes_[i]->bending.empty()) {
        const mjCFlex*   fl = flexes_[i];
        std::vector<int> slot(fl->nvert, -1);
        for (int v = 0; v < fl->nvert; v++) {
          slot[v] = body_slot[bodies_[fl->vertbodyid[v]]->weldid];
        }
        for (const auto& flap : fl->flaps) {
          if (flap.vertices[3] < 0) { continue; }
          for (int a = 0; a < 4; a++) {
            int sa = slot[flap.vertices[a]];
            if (sa < 0) continue;
            for (int b = 0; b < 4; b++) {
              int sb = slot[flap.vertices[b]];
              if (sb >= 0 && sb != sa) { adj[sa].insert(sb); }
            }
          }
        }
      }
    }

    // Expand each off-diagonal vertex block to all coordinate pairs. Diagonal blocks are
    // diagonal (R_b^T * R_b = I); any off-coordinate factor fill is counted symbolically.
    int                           n = 3 * nfree;
    std::vector<std::vector<int>> upper(n);
    for (int s = 0; s < nfree; s++) {
      for (int t : adj[s]) {
        if (t > s) {
          for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) { upper[3 * s + k].push_back(3 * t + l); }
          }
        }
      }
    }
    for (auto& row : upper) { std::sort(row.begin(), row.end()); }

    // flatten the pattern to CSR and count fill with the engine's symbolic factorization
    // (d == NULL: no mjData exists yet, scratch is heap-allocated)
    std::vector<int> u_rownnz(n), u_rowadr(n), u_colind;
    int              u_nnz = 0;
    for (int r = 0; r < n; r++) { u_nnz += (int)upper[r].size(); }
    u_colind.reserve(u_nnz);
    for (int r = 0; r < n; r++) {
      u_rownnz[r] = (int)upper[r].size();
      u_rowadr[r] = (int)u_colind.size();
      u_colind.insert(u_colind.end(), upper[r].begin(), upper[r].end());
    }
    std::vector<int> L_rownnz(n), L_rowadr(n), LT_rownnz(n), LT_rowadr(n);
    mjtSize          nnz = mju_cholFactorSymbolic(NULL,
                                                  L_rownnz.data(),
                                                  L_rowadr.data(),
                                                  NULL,
                                                  LT_rownnz.data(),
                                                  LT_rowadr.data(),
                                                  NULL,
                                                  u_rownnz.data(),
                                                  u_rowadr.data(),
                                                  u_colind.data(),
                                                  n,
                                                  NULL);

    nefm0dof = n;
    nefm0L   = nnz;
  }

  // sparse metric factor ordering slots, filled by mj_setConst
  // (mj_effCholSetConst): 1D rods and 2D flexes use the factor; the covered
  // vertices are disjoint dof triples
  nefmCvert = 0;
  for (int i = 0; i < nflex; i++) {
    if (flexes_[i]->interpolated || flexes_[i]->rigid) { continue; }
    if ((flexes_[i]->dim == 1 && !flexes_[i]->bending.empty()) || flexes_[i]->dim == 2) {
      nefmCvert += flexes_[i]->nvert;
    }
  }
  nefmCvert = mjMIN(nefmCvert, nv / 3);

  // mesh counts
  for (int i = 0; i < nmesh; i++) {
    nmeshvert     += meshes[i]->nvert();
    nmeshnormal   += meshes[i]->nnormal();
    nmeshface     += meshes[i]->nface();
    nmeshtexcoord += (meshes[i]->HasTexcoord() ? meshes[i]->ntexcoord() : 0);
    nmeshgraph    += meshes[i]->szgraph();
    nmeshpoly     += meshes[i]->npolygon();
    nmeshpolyvert += meshes[i]->npolygonvert();
    nmeshpolymap  += meshes[i]->npolygonmap();
  }

  // skin counts
  for (int i = 0; i < nskin; i++) {
    nskinvert    += skins_[i]->get_vert().size() / 3;
    nskintexvert += skins_[i]->get_texcoord().size() / 2;
    nskinface    += skins_[i]->get_face().size() / 3;
    nskinbone    += skins_[i]->bodyid.size();
    for (int j = 0; j < skins_[i]->bodyid.size(); j++) {
      nskinbonevert += skins_[i]->get_vertid()[j].size();
    }
  }

  // nhfielddata
  for (int i = 0; i < nhfield; i++) {
    nhfielddata += static_cast<mjtSize>(hfields[i]->nrow) * hfields[i]->ncol;
  }

  // ntexdata
  for (int i = 0; i < ntex; i++) {
    const mjCTexture* tex  = textures_[i];
    ntexdata              += static_cast<mjtSize>(tex->nchannel) * tex->width * tex->height;
  }

  // nwrap
  for (int i = 0; i < ntendon; i++) { nwrap += static_cast<mjtSize>(tendons_[i]->path.size()); }

  // nsensordata
  for (int i = 0; i < nsensor; i++) { nsensordata += sensors_[i]->dim; }

  // nhistory: layout is [user, cursor, times(n), values(n*dim)] = 2 + n + n*dim
  nhistory = 0;
  for (int i = 0; i < actuators_.size(); i++) {
    if (actuators_[i]->nsample > 0) {
      nhistory += 2 + actuators_[i]->nsample + actuators_[i]->nsample * actuators_[i]->ctrlnum_;
    }
  }
  // sensor delay: layout is [user, cursor, times(n), values(n*dim)] = 2 + n + n*dim
  for (int i = 0; i < sensors_.size(); i++) {
    if (sensors_[i]->nsample > 0) {
      nhistory += 2 + sensors_[i]->nsample + sensors_[i]->nsample * sensors_[i]->dim;
    }
  }

  // nplastic
  nplastic = 0;
  for (int i = 0; i < njnt; i++) {
    if (joints_[i]->springyield > 0) { nplastic += joints_[i]->nq(); }
  }
  for (int i = 0; i < ntendon; i++) {
    if (tendons_[i]->springyield > 0) { nplastic += 2; }
  }
  for (int i = 0; i < nflex; i++) { nplastic += flexes_[i]->PlasticNum(); }

  // nnumericdata
  for (int i = 0; i < nnumeric; i++) { nnumericdata += numerics_[i]->size; }

  // ntextdata
  for (int i = 0; i < ntext; i++) { ntextdata += (int)texts_[i]->data_.size() + 1; }

  // ntupledata
  for (int i = 0; i < ntuple; i++) { ntupledata += (int)tuples_[i]->objtype_.size(); }

  // npluginattr
  for (int i = 0; i < nplugin; i++) {
    npluginattr += (int)plugins_[i]->flattened_attributes.size();
  }

  // nnames
  nnames = (int)modelname_.size() + 1;
  // clang-format off
  for (int i = 0; i < nbody; i++)     nnames += (int)bodies_[i]->name.length() + 1;
  for (int i = 0; i < njnt; i++)      nnames += (int)joints_[i]->name.length() + 1;
  for (int i = 0; i < ngeom; i++)     nnames += (int)geoms_[i]->name.length() + 1;
  for (int i = 0; i < nsite; i++)     nnames += (int)sites_[i]->name.length() + 1;
  for (int i = 0; i < ncam; i++)      nnames += (int)cameras_[i]->name.length() + 1;
  for (int i = 0; i < nlight; i++)    nnames += (int)lights_[i]->name.length() + 1;
  for (int i = 0; i < nflex; i++)     nnames += (int)flexes_[i]->name.length() + 1;
  for (int i = 0; i < nmesh; i++)     nnames += (int)meshes[i]->name.length() + 1;
  for (int i = 0; i < nskin; i++)     nnames += (int)skins_[i]->name.length() + 1;
  for (int i = 0; i < nhfield; i++)   nnames += (int)hfields[i]->name.length() + 1;
  for (int i = 0; i < ntex; i++)      nnames += (int)textures_[i]->name.length() + 1;
  for (int i = 0; i < nmat; i++)      nnames += (int)materials_[i]->name.length() + 1;
  for (int i = 0; i < npair; i++)     nnames += (int)pairs_[i]->name.length() + 1;
  for (int i = 0; i < nexclude; i++)  nnames += (int)excludes_[i]->name.length() + 1;
  for (int i = 0; i < neq; i++)       nnames += (int)equalities_[i]->name.length() + 1;
  for (int i = 0; i < ntendon; i++)   nnames += (int)tendons_[i]->name.length() + 1;
  for (int i = 0; i < nactuator; i++) nnames += (int)actuators_[i]->name.length() + 1;
  for (int i = 0; i < nsensor; i++)   nnames += (int)sensors_[i]->name.length() + 1;
  for (int i = 0; i < nnumeric; i++)  nnames += (int)numerics_[i]->name.length() + 1;
  for (int i = 0; i < ntext; i++)     nnames += (int)texts_[i]->name.length() + 1;
  for (int i = 0; i < ntuple; i++)    nnames += (int)tuples_[i]->name.length() + 1;
  for (int i = 0; i < nkey; i++)      nnames += (int)keys_[i]->name.length() + 1;
  for (int i = 0; i < nplugin; i++)   nnames += (int)plugins_[i]->name.length() + 1;
  // clang-format on

  // npaths
  npaths  = 0;
  npaths += getpathslength(hfields);
  npaths += getpathslength(meshes);
  npaths += getpathslength(skins_);
  npaths += getpathslength(textures_);
  for (const auto& texture : textures_) {
    std::string type = texture->get_content_type();
    if (!type.empty()) { npaths += type.length() + 1; }
  }
  if (npaths == 0) { npaths = 1; }

  // nemax
  for (int i = 0; i < neq; i++) {
    if (equalities_[i]->type == mjEQ_CONNECT) {
      nemax += 3;
    } else if (equalities_[i]->type == mjEQ_WELD) {
      nemax += 7;
    } else {
      nemax += 1;
    }
  }
}


// automatic stiffness and damping computation
void mjCModel::AutoSpringDamper(mjModel* m) {
  // process all joints
  for (int n = 0; n < m->njnt; n++) {
    // get joint dof address and number of dimensions
    int adr  = m->jnt_dofadr[n];
    int ndim = mjCJoint::nv((mjtJoint)m->jnt_type[n]);

    // get timeconst and dampratio from joint specification
    mjtNum timeconst = (mjtNum)joints_[n]->springdamper[0];
    mjtNum dampratio = (mjtNum)joints_[n]->springdamper[1];

    // skip joint if either parameter is non-positive
    if (timeconst <= 0 || dampratio <= 0) { continue; }

    // get average inertia (dof_invweight0 in free joint is different for tran and rot)
    mjtNum inertia = 0;
    for (int i = 0; i < ndim; i++) { inertia += m->dof_invweight0[adr + i]; }
    inertia = ((mjtNum)ndim) / std::max(mjMINVAL, inertia);

    // compute stiffness and damping (same as solref computation)
    mjtNum stiffness = inertia / std::max(mjMINVAL, timeconst * timeconst * dampratio * dampratio);
    mjtNum damping   = 2 * inertia / std::max(mjMINVAL, timeconst);

    // save stiffness and damping in the private mjsJoints
    joints_[n]->stiffness[0] = stiffness;
    joints_[n]->damping[0]   = damping;

    // assign
    m->jnt_stiffness[n] = stiffness;
    for (int i = 0; i < ndim; i++) { m->dof_damping[adr + i] = damping; }
  }
}


// arguments for lengthrange thread function
struct _LRThreadArg {
  mjModel*       m;
  mjData*        data;
  int            start;
  int            num;
  const mjLROpt* LRopt;
  char*          error;
  int            error_sz;
};
typedef struct _LRThreadArg LRThreadArg;


// thread function for lengthrange computation
void* LRfunc(void* arg) {
  LRThreadArg* larg = (LRThreadArg*)arg;

  for (int i = larg->start; i < larg->start + larg->num; i++) {
    if (i < larg->m->nactuator) {
      if (!mj_setLengthRange(larg->m, larg->data, i, larg->LRopt, larg->error, larg->error_sz)) {
        return nullptr;
      }
    }
  }

  return nullptr;
}


// compute actuator lengthrange
void mjCModel::LengthRange(mjModel* m, mjData* data) {
  // save options and modify
  mjOption saveopt    = m->opt;
  m->opt.disableflags = mjDSBL_FRICTIONLOSS |
                        mjDSBL_CONTACT |
                        mjDSBL_SPRING |
                        mjDSBL_DAMPER |
                        mjDSBL_GRAVITY |
                        mjDSBL_ACTUATION;
  if (compiler.LRopt.timestep > 0) { m->opt.timestep = compiler.LRopt.timestep; }

  // count actuators that need computation
  int cnt = 0;
  for (int i = 0; i < m->nactuator; i++) {
    // skip depending on mode and type
    int ismuscle =
        (m->actuator_gaintype[i] == mjGAIN_MUSCLE || m->actuator_biastype[i] == mjBIAS_MUSCLE);
    int isuser = (m->actuator_gaintype[i] == mjGAIN_USER || m->actuator_biastype[i] == mjBIAS_USER);
    if ((compiler.LRopt.mode == mjLRMODE_NONE) ||
        (compiler.LRopt.mode == mjLRMODE_MUSCLE && !ismuscle) ||
        (compiler.LRopt.mode == mjLRMODE_MUSCLEUSER && !ismuscle && !isuser)) {
      continue;
    }

    // use existing length range if available
    if (compiler.LRopt.useexisting &&
        (m->actuator_lengthrange[2 * i] < m->actuator_lengthrange[2 * i + 1])) {
      continue;
    }

    // count
    cnt++;
  }

  const auto nthread = NumCompilerThreads(cnt);

  // single thread
  if (!compiler.usethread || cnt < 2 || nthread < 2) {
    char err[200];
    for (int i = 0; i < m->nactuator; i++) {
      if (!mj_setLengthRange(m, data, i, &compiler.LRopt, err, 200)) {
        throw mjCError(0, "%s", err);
      }
    }
  }

  // multiple threads
  else {
    // allocate mjData for each thread
    // using vectors to allow arbitrary number of threads without stack overflow
    std::vector<std::vector<char>> err(nthread, std::vector<char>(200));
    std::vector<mjData*>           pdata(nthread);
    pdata[0] = data;

    for (int i = 1; i < nthread; i++) { pdata[i] = mj_makeData(m); }

    // number of actuators per thread
    int num = m->nactuator / nthread;
    while (num * nthread < m->nactuator) { num++; }

    // prepare thread function arguments
    std::vector<LRThreadArg> arg(nthread);
    for (int i = 0; i < nthread; i++) {
      LRThreadArg temp = {m, pdata[i], i * num, num, &compiler.LRopt, err[i].data(), 200};
      arg[i]           = temp;
      err[i][0]        = 0;
    }

    // launch threads and wait for them to finish
    mujoco::user::ThreadPool pool(nthread);
    for (int i = 0; i < nthread; i++) {
      pool.Schedule([&arg, i]() { LRfunc(&arg[i]); });
    }
    pool.WaitCount(nthread);

    // free mjData allocated here
    for (int i = 1; i < nthread; i++) { mj_deleteData(pdata[i]); }

    // report first error
    for (int i = 0; i < nthread; i++) {
      if (err[i][0]) { throw mjCError(0, "%s", err[i].data()); }
    }
  }

  // restore options
  m->opt = saveopt;
}


// Add items to a generic list. This enables the names and paths to be stored.
// input - string to add
// adr - current address in the list
// output_adr_field - the field where the address should be stored (name_meshadr)
// output_buffer - the field where the data should be stored (i.e. names or paths)
static int addtolist(const std::string& input,
                     int                adr,
                     int*               output_adr_field,
                     char*              output_buffer) {
  *output_adr_field = adr;

  // copy input
  memcpy(output_buffer + adr, input.c_str(), input.size());
  adr += (int)input.size();

  // append 0
  output_buffer[adr] = 0;
  adr++;

  return adr;
}

// process names from one list: concatenate, compute addresses
template <class T>
static int namelist(vector<T*>& list, int adr, int* name_adr, char* names, int* map) {
  // compute hash map addresses
  int map_size = mjLOAD_MULTIPLE * list.size();
  for (unsigned int i = 0; i < list.size(); i++) {
    // ignore empty strings
    if (list[i]->name.empty()) { continue; }

    uint64_t j = mj_hashString(list[i]->name.c_str(), map_size);

    // find first empty slot using linear probing
    for (; map[j] != -1; j = (j + 1) % map_size) {}
    map[j] = i;
  }

  for (unsigned int i = 0; i < list.size(); i++) {
    adr = addtolist(list[i]->name, adr, &name_adr[i], names);
  }

  return adr;
}


// copy names, compute name addresses
void mjCModel::CopyNames(mjModel* m) {
  std::vector<mjCMesh*>   meshes  = CompiledMeshes();
  std::vector<mjCHField*> hfields = CompiledHFields();
  // start with model name
  int  adr     = (int)modelname_.size() + 1;
  int* map_adr = m->names_map;
  mju_strncpy(m->names, modelname_.c_str(), m->nnames);
  memset(m->names_map, -1, sizeof(int) * m->nnames_map);

  // process all lists
  adr      = namelist(bodies_, adr, m->name_bodyadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * bodies_.size();

  adr      = namelist(joints_, adr, m->name_jntadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * joints_.size();

  adr      = namelist(geoms_, adr, m->name_geomadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * geoms_.size();

  adr      = namelist(sites_, adr, m->name_siteadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * sites_.size();

  adr      = namelist(cameras_, adr, m->name_camadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * cameras_.size();

  adr      = namelist(lights_, adr, m->name_lightadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * lights_.size();

  adr      = namelist(flexes_, adr, m->name_flexadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * flexes_.size();

  adr      = namelist(meshes, adr, m->name_meshadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * meshes.size();

  adr      = namelist(skins_, adr, m->name_skinadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * skins_.size();

  adr      = namelist(hfields, adr, m->name_hfieldadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * hfields.size();

  adr      = namelist(textures_, adr, m->name_texadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * textures_.size();

  adr      = namelist(materials_, adr, m->name_matadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * materials_.size();

  // the model holds pairs and excludes in the order of their ids, not of the lists
  std::vector<mjCPair*> pairs  = orderbyid(pairs_);
  adr                          = namelist(pairs, adr, m->name_pairadr, m->names, map_adr);
  map_adr                     += mjLOAD_MULTIPLE * pairs_.size();

  std::vector<mjCBodyPair*> excludes = orderbyid(excludes_);
  adr      = namelist(excludes, adr, m->name_excludeadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * excludes_.size();

  adr      = namelist(equalities_, adr, m->name_eqadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * equalities_.size();

  adr      = namelist(tendons_, adr, m->name_tendonadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * tendons_.size();

  adr      = namelist(actuators_, adr, m->name_actuatoradr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * actuators_.size();

  adr      = namelist(sensors_, adr, m->name_sensoradr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * sensors_.size();

  adr      = namelist(numerics_, adr, m->name_numericadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * numerics_.size();

  adr      = namelist(texts_, adr, m->name_textadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * texts_.size();

  adr      = namelist(tuples_, adr, m->name_tupleadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * tuples_.size();

  adr      = namelist(keys_, adr, m->name_keyadr, m->names, map_adr);
  map_adr += mjLOAD_MULTIPLE * keys_.size();

  adr = namelist(plugins_, adr, m->name_pluginadr, m->names, map_adr);

  // check size, SHOULD NOT OCCUR
  if (adr != nnames) {
    throw mjCError(0, "size mismatch in %s: expected %d, got %d", "names", nnames, adr);
  }
}

// process paths from one list: concatenate resolved paths, compute addresses
template <class T>
static int pathlist(vector<T*>& list, int adr, int* path_adr, char* paths) {
  for (unsigned int i = 0; i < list.size(); ++i) {
    path_adr[i] = -1;
    if (!list[i] || list[i]->File().empty()) { continue; }
    adr = addtolist(list[i]->ResolvedFilePath(), adr, &path_adr[i], paths);
  }

  return adr;
}

// copy the resolved path of every asset with a file: the path LoadResource opened, against the
// modelfiledir and meshdir/texturedir of the spec that owns the asset, not the file as authored;
// texture content types follow the paths, -1 if not authored
void mjCModel::CopyPaths(mjModel* m) {
  std::vector<mjCMesh*>   meshes  = CompiledMeshes();
  std::vector<mjCHField*> hfields = CompiledHFields();
  // start with 0 address, unlike m->names m->paths might be empty
  size_t adr  = 0;
  m->paths[0] = 0;

  adr = pathlist(hfields, adr, m->hfield_pathadr, m->paths);
  adr = pathlist(meshes, adr, m->mesh_pathadr, m->paths);
  adr = pathlist(skins_, adr, m->skin_pathadr, m->paths);
  adr = pathlist(textures_, adr, m->tex_pathadr, m->paths);

  for (unsigned int i = 0; i < textures_.size(); ++i) {
    m->tex_contenttypeadr[i] = -1;
    std::string type         = textures_[i]->get_content_type();
    if (!type.empty()) { adr = addtolist(type, adr, &m->tex_contenttypeadr[i], m->paths); }
  }
}


// copy objects inside kinematic tree
void mjCModel::CopyTree(mjModel* m) {
  int jntadr     = 0;  // addresses in global arrays
  int dofadr     = 0;
  int qposadr    = 0;
  int bvh_adr    = 0;
  int plasticadr = 0;

  m->body_sameframe[0] = mjSAMEFRAME_BODY;

  // main loop over bodies
  for (int i = 0; i < nbody; i++) {
    // get body and parent pointers
    mjCBody* pb  = bodies_[i];
    mjCBody* par = pb->parent;

    // set body fields
    m->body_parentid[i] = pb->parent ? pb->parent->id : 0;
    m->body_weldid[i]   = pb->weldid;
    m->body_mocapid[i]  = pb->mocapid;
    m->body_jntnum[i]   = (int)pb->joints.size();
    m->body_jntadr[i]   = (!pb->joints.empty() ? jntadr : -1);
    m->body_dofnum[i]   = pb->dofnum;
    m->body_dofadr[i]   = (pb->dofnum ? dofadr : -1);
    m->body_geomnum[i]  = (int)pb->geoms.size();
    m->body_geomadr[i]  = (!pb->geoms.empty() ? pb->geoms[0]->id : -1);
    mjuu_copyvec(m->body_pos + 3 * i, pb->pos, 3);
    mjuu_copyvec(m->body_quat + 4 * i, pb->quat, 4);
    mjuu_copyvec(m->body_ipos + 3 * i, pb->ipos, 3);
    mjuu_copyvec(m->body_iquat + 4 * i, pb->iquat, 4);
    m->body_mass[i] = (mjtNum)pb->mass;
    mjuu_copyvec(m->body_inertia + 3 * i, pb->inertia, 3);
    m->body_gravcomp[i] = pb->gravcomp;
    mjuu_copyvec(m->body_user + nuser_body * i, pb->get_userdata().data(), nuser_body);

    // bounding volume hierarchy
    m->body_bvhadr[i] = pb->tree.Nbvh() ? bvh_adr : -1;
    m->body_bvhnum[i] = pb->tree.Nbvh();
    if (pb->tree.Nbvh()) {
      memcpy(m->bvh_child + 2 * bvh_adr,
             pb->tree.Child().data(),
             2 * pb->tree.Nbvh() * sizeof(int));
      memcpy(m->bvh_depth + bvh_adr, pb->tree.Level().data(), pb->tree.Nbvh() * sizeof(int));
      for (int i = 0; i < pb->tree.Nbvh(); i++) {
        m->bvh_nodeid[i + bvh_adr] = pb->tree.Nodeidptr(i) ? *(pb->tree.Nodeidptr(i)) : -1;
      }
    }
    bvh_adr += pb->tree.Nbvh();

    // count free joints
    int cntfree = 0;
    for (int j = 0; j < (int)pb->joints.size(); j++) {
      cntfree += (pb->joints[j]->type == mjJNT_FREE);
    }

    // check validity of free joint
    if (cntfree > 1 || (cntfree == 1 && pb->joints.size() > 1)) {
      throw mjCError(pb, "free joint can only appear by itself");
    }
    if (cntfree && par && par->name != "world") {
      throw mjCError(pb, "free joint can only be used on top level");
    }

    // rootid: self if world or child of world, otherwise parent's rootid
    if (i == 0 || (par && par->name == "world")) {
      m->body_rootid[i] = i;
    } else {
      m->body_rootid[i] = m->body_rootid[par->id];
    }

    // init lastdof from parent
    pb->lastdof = par ? par->lastdof : -1;

    // init simple: sameframe, and (self-root, or parent is fixed child of world)
    int parentid      = m->body_parentid[i];
    m->body_simple[i] = (IsNullPose(m->body_ipos + 3 * i, m->body_iquat + 4 * i) &&
                         (m->body_rootid[i] == i ||
                          (m->body_parentid[parentid] == 0 && m->body_dofnum[parentid] == 0)));

    // user override: disable simple optimization
    if (!pb->simple) { m->body_simple[i] = 0; }

    // a parent body is never simple (unless world)
    if (m->body_parentid[i] > 0) { m->body_simple[m->body_parentid[i]] = 0; }

    // loop over joints for this body
    int rotfound = 0;
    for (int j = 0; j < (int)pb->joints.size(); j++) {
      // get pointer and id
      mjCJoint* pj  = pb->joints[j];
      int       jid = pj->id;

      // set joint fields
      m->jnt_type[jid]       = pj->type;
      m->jnt_group[jid]      = pj->group;
      m->jnt_plasticadr[jid] = pj->springyield > 0 ? plasticadr : -1;
      pj->plasticadr_        = m->jnt_plasticadr[jid];
      if (pj->springyield > 0) { plasticadr += pj->nq(); }
      m->jnt_limited[jid]       = (mjtBool)pj->is_limited();
      m->jnt_actfrclimited[jid] = (mjtBool)pj->is_actfrclimited();
      m->jnt_actgravcomp[jid]   = pj->actgravcomp;
      m->jnt_qposadr[jid]       = pj->qposadr_;
      m->jnt_dofadr[jid]        = pj->dofadr_;
      m->jnt_bodyid[jid]        = pj->body->id;
      mjuu_copyvec(m->jnt_pos + 3 * jid, pj->pos, 3);
      mjuu_copyvec(m->jnt_axis + 3 * jid, pj->axis, 3);
      m->jnt_stiffness[jid] = (mjtNum)pj->stiffness[0];
      mjuu_copyvec(m->jnt_stiffnesspoly + mjNPOLY * jid, pj->stiffness + 1, mjNPOLY);
      m->jnt_springyield[jid] = (mjtNum)pj->springyield;
      mjuu_copyvec(m->jnt_range + 2 * jid, pj->range, 2);
      mjuu_copyvec(m->jnt_actfrcrange + 2 * jid, pj->actfrcrange, 2);
      mjuu_copyvec(m->jnt_solref + mjNREF * jid, pj->solref_limit, mjNREF);
      mjuu_copyvec(m->jnt_solimp + mjNIMP * jid, pj->solimp_limit, mjNIMP);
      m->jnt_margin[jid] = (mjtNum)pj->margin;
      mjuu_copyvec(m->jnt_user + nuser_jnt * jid, pj->get_userdata().data(), nuser_jnt);

      // not simple if: rotation already found, or pos not zero, or mis-aligned axis
      bool axis_aligned = ((std::abs(pj->axis[0]) > mjEPS) +
                           (std::abs(pj->axis[1]) > mjEPS) +
                           (std::abs(pj->axis[2]) > mjEPS)) == 1;
      if (rotfound ||
          !IsNullPose(m->jnt_pos + 3 * jid) ||
          ((pj->type == mjJNT_HINGE || pj->type == mjJNT_SLIDE) && !axis_aligned)) {
        m->body_simple[i] = 0;
      }

      // mark rotation
      if (pj->type == mjJNT_BALL || pj->type == mjJNT_HINGE) { rotfound = 1; }

      // set qpos0 and qpos_spring, check type
      switch (pj->type) {
        case mjJNT_FREE:
          mjuu_copyvec(m->qpos0 + qposadr, pb->pos, 3);
          mjuu_copyvec(m->qpos0 + qposadr + 3, pb->quat, 4);
          mjuu_copyvec(m->qpos_spring + qposadr, m->qpos0 + qposadr, 7);
          break;

        case mjJNT_BALL:
          m->qpos0[qposadr]     = 1;
          m->qpos0[qposadr + 1] = 0;
          m->qpos0[qposadr + 2] = 0;
          m->qpos0[qposadr + 3] = 0;
          mjuu_copyvec(m->qpos_spring + qposadr, m->qpos0 + qposadr, 4);
          break;

        case mjJNT_SLIDE:
        case mjJNT_HINGE:
          m->qpos0[qposadr]       = (mjtNum)pj->ref;
          m->qpos_spring[qposadr] = (mjtNum)pj->springref;
          break;

        default:
          throw mjCError(pj, "unknown joint type");
      }

      // set dof fields for this joint
      for (int j1 = 0; j1 < pj->nv(); j1++) {
        // set attributes
        m->dof_bodyid[dofadr] = pb->id;
        m->dof_jntid[dofadr]  = jid;
        mjuu_copyvec(m->dof_solref + mjNREF * dofadr, pj->solref_friction, mjNREF);
        mjuu_copyvec(m->dof_solimp + mjNIMP * dofadr, pj->solimp_friction, mjNIMP);
        m->dof_frictionloss[dofadr] = (mjtNum)pj->frictionloss;
        m->dof_armature[dofadr]     = (mjtNum)pj->armature;
        m->dof_damping[dofadr]      = (mjtNum)pj->damping[0];
        mjuu_copyvec(m->dof_dampingpoly + mjNPOLY * dofadr, pj->damping + 1, mjNPOLY);

        // set dof_parentid, update body.lastdof
        m->dof_parentid[dofadr] = pb->lastdof;
        pb->lastdof             = dofadr;

        // advance dof counter
        dofadr++;
      }

      // advance joint and qpos counters
      jntadr++;
      qposadr += pj->nq();
    }

    // simple body with sliders and no rotational dofs: promote to simple level 2
    if (m->body_simple[i] && m->body_dofnum[i]) {
      m->body_simple[i] = 2;
      for (int j = 0; j < (int)pb->joints.size(); j++) {
        if (pb->joints[j]->type != mjJNT_SLIDE) {
          m->body_simple[i] = 1;
          break;
        }
      }
    }

    // loop over geoms for this body
    for (int j = 0; j < (int)pb->geoms.size(); j++) {
      // get pointer and id
      mjCGeom* pg  = pb->geoms[j];
      int      gid = pg->id;

      // set geom fields
      m->geom_type[gid]        = pg->type;
      m->geom_contype[gid]     = pg->contype;
      m->geom_conaffinity[gid] = pg->conaffinity;
      m->geom_condim[gid]      = pg->condim;
      m->geom_bodyid[gid]      = pg->body->id;
      if (pg->mesh && (pg->type == mjGEOM_MESH || pg->type == mjGEOM_SDF)) {
        m->geom_dataid[gid] = pg->mesh->id;
      } else if (pg->hfield) {
        m->geom_dataid[gid] = pg->hfield->id;
      } else {
        m->geom_dataid[gid] = -1;
      }
      m->geom_matid[gid]      = pg->matid;
      m->geom_group[gid]      = pg->group;
      m->geom_priority[gid]   = pg->priority;
      m->geom_castshadow[gid] = (mjtBool)pg->castshadow;
      mjuu_copyvec(m->geom_size + 3 * gid, pg->size, 3);
      mjuu_copyvec(m->geom_aabb + 6 * gid, pg->aabb, 6);
      mjuu_copyvec(m->geom_pos + 3 * gid, pg->pos, 3);
      mjuu_copyvec(m->geom_quat + 4 * gid, pg->quat, 4);
      mjuu_copyvec(m->geom_friction + 3 * gid, pg->friction, 3);
      m->geom_solmix[gid] = (mjtNum)pg->solmix;
      mjuu_copyvec(m->geom_solref + mjNREF * gid, pg->solref, mjNREF);
      mjuu_copyvec(m->geom_solimp + mjNIMP * gid, pg->solimp, mjNIMP);
      m->geom_margin[gid] = (mjtNum)pg->margin;
      m->geom_gap[gid]    = (mjtNum)pg->gap;
      mjuu_copyvec(m->geom_surfacevel + 6 * gid, pg->surfacevel, 6);
      m->geom_adhesion[gid] = (mjtNum)pg->adhesion;
      mjuu_copyvec(m->geom_fluid + mjNFLUID * gid, pg->fluid, mjNFLUID);
      mjuu_copyvec(m->geom_user + nuser_geom * gid, pg->get_userdata().data(), nuser_geom);
      mjuu_copyvec(m->geom_rgba + 4 * gid, pg->rgba, 4);

      // bounding sphere of a mesh or SDF geom, mj_setConst computes those of other geoms
      m->geom_rbound[gid] = (mjtNum)pg->GetRBound();
    }

    // loop over sites for this body
    for (int j = 0; j < (int)pb->sites.size(); j++) {
      // get pointer and id
      mjCSite* ps  = pb->sites[j];
      int      sid = ps->id;

      // set site fields
      m->site_type[sid]   = ps->type;
      m->site_bodyid[sid] = ps->body->id;
      m->site_dataid[sid] = ps->mesh ? ps->mesh->id : -1;
      m->site_matid[sid]  = ps->matid;
      m->site_group[sid]  = ps->group;
      mjuu_copyvec(m->site_size + 3 * sid, ps->size, 3);
      mjuu_copyvec(m->site_pos + 3 * sid, ps->pos, 3);
      mjuu_copyvec(m->site_quat + 4 * sid, ps->quat, 4);
      mjuu_copyvec(m->site_user + nuser_site * sid, ps->userdata_.data(), nuser_site);
      mjuu_copyvec(m->site_rgba + 4 * sid, ps->rgba, 4);
    }

    // loop over cameras for this body
    for (int j = 0; j < (int)pb->cameras.size(); j++) {
      // get pointer and id
      mjCCamera* pc  = pb->cameras[j];
      int        cid = pc->id;

      // set camera fields
      m->cam_bodyid[cid]       = pc->body->id;
      m->cam_mode[cid]         = pc->mode;
      m->cam_targetbodyid[cid] = pc->targetbodyid;
      mjuu_copyvec(m->cam_pos + 3 * cid, pc->pos, 3);
      mjuu_copyvec(m->cam_quat + 4 * cid, pc->quat, 4);
      m->cam_projection[cid] = pc->proj;
      m->cam_fovy[cid]       = (mjtNum)pc->fovy;
      m->cam_ipd[cid]        = (mjtNum)pc->ipd;
      mjuu_copyvec(m->cam_resolution + 2 * cid, pc->resolution, 2);
      m->cam_output[cid] = pc->output;
      mjuu_copyvec(m->cam_sensorsize + 2 * cid, pc->sensor_size, 2);
      mjuu_copyvec(m->cam_intrinsic + 4 * cid, pc->intrinsic, 4);
      mjuu_copyvec(m->cam_user + nuser_cam * cid, pc->get_userdata().data(), nuser_cam);
    }

    // loop over lights for this body
    for (int j = 0; j < (int)pb->lights.size(); j++) {
      // get pointer and id
      mjCLight* pl  = pb->lights[j];
      int       lid = pl->id;

      // set light fields
      m->light_bodyid[lid]       = pl->body->id;
      m->light_mode[lid]         = (int)pl->mode;
      m->light_targetbodyid[lid] = pl->targetbodyid;
      m->light_type[lid]         = pl->type;
      m->light_texid[lid]        = pl->texid;
      m->light_castshadow[lid]   = (mjtBool)pl->castshadow;
      m->light_active[lid]       = (mjtBool)pl->active;
      mjuu_copyvec(m->light_pos + 3 * lid, pl->pos, 3);
      mjuu_copyvec(m->light_dir + 3 * lid, pl->dir, 3);
      m->light_bulbradius[lid] = pl->bulbradius;
      m->light_intensity[lid]  = pl->intensity;
      m->light_range[lid]      = pl->range;
      mjuu_copyvec(m->light_attenuation + 3 * lid, pl->attenuation, 3);
      m->light_cutoff[lid]   = pl->cutoff;
      m->light_softness[lid] = pl->softness;
      m->light_exponent[lid] = pl->exponent;
      mjuu_copyvec(m->light_ambient + 3 * lid, pl->ambient, 3);
      mjuu_copyvec(m->light_diffuse + 3 * lid, pl->diffuse, 3);
      mjuu_copyvec(m->light_specular + 3 * lid, pl->specular, 3);
    }
  }

  // check number of dof's constructed, SHOULD NOT OCCUR
  if (nv != dofadr) { throw mjCError(0, "unexpected number of DOFs"); }

  // count kinematic trees under world body, compute dof_treeid
  int ntree = 0;
  for (int i = 0; i < nv; i++) {
    if (m->dof_parentid[i] == -1) { ntree++; }
    m->dof_treeid[i] = ntree - 1;
  }

  // check number of trees constructed, SHOULD NOT OCCUR
  if (ntree != m->ntree) {
    throw mjCError(0,
                   "unexpected number of TREEs. Counted %d, expected %d",
                   nullptr,
                   ntree,
                   m->ntree);
  }

  // compute body_treeid
  for (int i = 0; i < nbody; i++) {
    int weldid = m->body_weldid[i];
    if (m->body_dofnum[weldid]) {
      m->body_treeid[i] = m->dof_treeid[m->body_dofadr[weldid]];
    } else {
      m->body_treeid[i] = -1;
    }
  }

  // initialize AUTO sleep policy for all trees
  for (int i = 0; i < m->ntree; i++) { m->tree_sleep_policy[i] = mjSLEEP_AUTO; }

  // loop over bodies, check and set non-default sleep policy
  for (int i = 1; i < nbody; i++) {
    mjCBody* pb = bodies_[i];

    // validate and set non-default sleep policy
    if (pb->sleep != mjSLEEP_AUTO) {
      int treeid = m->body_treeid[i];
      // non-default sleep policy only allowed for first body in a tree
      if (treeid == -1 || treeid == m->body_treeid[i - 1]) {
        throw mjCError(pb, "sleep policy only allowed for movable root bodies");
      }
      m->tree_sleep_policy[treeid] = pb->sleep;
    }
  }

  // recompute nM and dof_Madr given m.dof_parentid, validate
  int nM_post = 0;
  for (int i = 0; i < nv; i++) {
    // set address of this dof
    m->dof_Madr[i] = nM_post;

    // count ancestor dofs including self
    int j = i;
    while (j >= 0) {
      nM_post++;
      j = m->dof_parentid[j];
    }
  }
  if (nM_post != nM) throw mjCError(0, "nM mismatch: pre %d, post %d", nullptr, nM, nM_post);

  // recompute nD, validate
  int nD_post = 2 * m->nM - nv;
  if (nD_post != nD) throw mjCError(0, "nD mismatch: pre %d, post %d", nullptr, nD, nD_post);

  // bodies_[]->subtreedofs already computed in ComputeSparseSizes

  // recompute nB given {body->subtreedofs, body->dofnum, body->parent}, validate
  int nB_post = 0;
  for (int i = 0; i < nbody; i++) {
    // add subtree dofs (including self)
    nB_post += bodies_[i]->subtreedofs;
    // add dofs in ancestor bodies
    int j = bodies_[i]->parent ? bodies_[i]->parent->id : 0;
    while (j > 0) {
      nB_post += bodies_[j]->dofnum;
      j        = bodies_[j]->parent ? bodies_[j]->parent->id : 0;
    }
  }
  if (nB_post != nB) throw mjCError(0, "nB mismatch: pre %d, post %d", nullptr, nB, nB_post);
}

// copy plugin data
void mjCModel::CopyPlugins(mjModel* m) {
  // assign plugin slots and copy plugin config attributes
  {
    int adr = 0;
    for (int i = 0; i < nplugin; ++i) {
      m->plugin[i]   = plugins_[i]->plugin_slot;
      const int size = plugins_[i]->flattened_attributes.size();
      std::memcpy(m->plugin_attr + adr, plugins_[i]->flattened_attributes.data(), size);
      m->plugin_attradr[i]  = adr;
      adr                  += size;
    }
  }

  // query and set plugin-related information
  {
    // set actuator_plugin to the plugin instance ID
    std::vector<std::vector<int>> plugin_to_actuators(nplugin);
    for (int i = 0; i < nactuator; ++i) {
      if (actuators_[i]->plugin.active) {
        int actuator_plugin   = static_cast<mjCPlugin*>(actuators_[i]->plugin.element)->id;
        m->actuator_plugin[i] = actuator_plugin;
        plugin_to_actuators[actuator_plugin].push_back(i);
      } else {
        m->actuator_plugin[i] = -1;
      }
    }

    for (int i = 0; i < nbody; ++i) {
      if (bodies_[i]->plugin.active) {
        m->body_plugin[i] = static_cast<mjCPlugin*>(bodies_[i]->plugin.element)->id;
      } else {
        m->body_plugin[i] = -1;
      }
    }

    for (int i = 0; i < ngeom; ++i) {
      if (geoms_[i]->plugin.active) {
        m->geom_plugin[i] = static_cast<mjCPlugin*>(geoms_[i]->plugin.element)->id;
      } else {
        m->geom_plugin[i] = -1;
      }
    }

    std::vector<std::vector<int>> plugin_to_sensors(nplugin);
    for (int i = 0; i < nsensor; ++i) {
      if (sensors_[i]->type == mjSENS_PLUGIN) {
        int sensor_plugin   = static_cast<mjCPlugin*>(sensors_[i]->plugin.element)->id;
        m->sensor_plugin[i] = sensor_plugin;
        plugin_to_sensors[sensor_plugin].push_back(i);
      } else {
        m->sensor_plugin[i] = -1;
      }
    }

    // query plugin->nstate, compute and set plugin_state and plugin_stateadr
    // for sensor plugins, also query plugin->nsensordata and set nsensordata
    int stateadr = 0;
    for (int i = 0; i < nplugin; ++i) {
      const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[i]);
      if (!plugin->nstate) { mju_error("`nstate` is null for plugin at slot %d", m->plugin[i]); }
      int nstate              = plugin->nstate(m, i);
      m->plugin_stateadr[i]   = stateadr;
      m->plugin_statenum[i]   = nstate;
      plugins_[i]->stateadr_  = nstate > 0 ? stateadr : -1;
      plugins_[i]->statenum_  = nstate;
      stateadr               += nstate;
      if (plugin->capabilityflags & mjPLUGIN_SENSOR) {
        for (int sensor_id : plugin_to_sensors[i]) {
          if (!plugin->nsensordata) {
            mju_error("`nsensordata` is null for plugin at slot %d", m->plugin[i]);
          }
          int nsensordata = plugin->nsensordata(m, i, sensor_id);

          sensors_[sensor_id]->dim        = nsensordata;
          sensors_[sensor_id]->needstage  = static_cast<mjtStage>(plugin->needstage);
          this->nsensordata              += nsensordata;
        }
      }
    }
    m->npluginstate = stateadr;
  }
}


// compute number of dofs for a given tendon
int mjCModel::CountTendonDofs(const mjModel* m, int id) {
  std::vector<bool> dof_used(m->nv, false);

  int nv  = m->nv;
  int adr = m->tendon_adr[id];
  int num = m->tendon_num[id];

  if (m->tendon_type[id] == mjTENDON_FIXED) { return num; }

  std::fill(dof_used.begin(), dof_used.end(), false);
  for (int j = 0; j < num; j++) {
    int type   = m->wrap_type[adr + j];
    int bodyid = -1;
    if (type == mjWRAP_SITE) {
      bodyid = m->site_bodyid[m->wrap_objid[adr + j]];
    } else if (type == mjWRAP_SPHERE || type == mjWRAP_CYLINDER) {
      bodyid = m->geom_bodyid[m->wrap_objid[adr + j]];
    }
    if (bodyid > 0) {
      int bid = bodyid;
      while (bid > 0) {
        int bdofadr = m->body_dofadr[bid];
        int bdofnum = m->body_dofnum[bid];
        for (int k = 0; k < bdofnum; k++) { dof_used[bdofadr + k] = true; }
        bid = m->body_parentid[bid];
      }
    }
  }

  int count = 0;
  for (int j = 0; j < nv; j++) { count += dof_used[j]; }
  return count;
}

int mjCModel::CountNJmom(const mjModel* m) {
  int nactuator = m->nactuator;
  int nv        = m->nv;

  int count = 0;
  for (int i = 0; i < nactuator; i++) {
    // extract info
    int id = m->actuator_trnid[2 * i];

    // process according to transmission type
    switch ((mjtTrn)m->actuator_trntype[i]) {
      case mjTRN_SO3:
        // ball joint: 3 identity rows; site+refsite: 3 dense rows
        count += m->actuator_trnid[2 * i + 1] >= 0 ? 3 * nv : 3;
        break;

      case mjTRN_JOINT:
      case mjTRN_JOINTINPARENT:
        switch ((mjtJoint)m->jnt_type[id]) {
          case mjJNT_SLIDE:
          case mjJNT_HINGE:
            count += 1;
            break;

          case mjJNT_BALL:
            count += 3;
            break;

          case mjJNT_FREE:
            count += 6;
            break;
        }
        break;
      case mjTRN_SLIDERCRANK: {
        int              id_slider = m->actuator_trnid[2 * i + 1];
        std::vector<int> chain(m->nv);
        count += mj_mergeChain(m, chain.data(), m->site_bodyid[id], m->site_bodyid[id_slider], 0);
      } break;

      case mjTRN_TENDON:
        count += CountTendonDofs(m, id);
        break;

      case mjTRN_SITE: {
        int refid    = m->actuator_trnid[2 * i + 1];
        int ref_body = refid >= 0 ? m->site_bodyid[refid] : 0;

        std::vector<int> chain(m->nv);
        count += mj_mergeChain(m, chain.data(), m->site_bodyid[id], ref_body, /*flg_skipcommon=*/1);
      } break;

      case mjTRN_BODY:
        count += nv;
        break;

      default:
        // SHOULD NOT OCCUR
        throw mjCError(0, "unknown transmission type");
        break;
    }
  }
  return count;
}

// compute non-zeros in ten_J matrix
int mjCModel::CountNJten(const mjModel* m) {
  int ntendon = m->ntendon;

  int count = 0;
  for (int i = 0; i < ntendon; i++) { count += CountTendonDofs(m, i); }

  return count;
}

// copy objects outside kinematic tree
// NOLINTBEGIN(readability/fn_size)
void mjCModel::CopyObjects(mjModel* m) {
  std::vector<mjCMesh*>   meshes  = CompiledMeshes();
  std::vector<mjCHField*> hfields = CompiledHFields();
  mjtSize adr, bone_adr, vert_adr, node_adr, normal_adr, face_adr, texcoord_adr, oct_adr;
  mjtSize stiffness_adr, bending_adr;
  mjtSize edge_adr, elem_adr, elemdata_adr, elemedge_adr, shelldata_adr;
  mjtSize bonevert_adr, graph_adr, data_adr, bvh_adr;
  mjtSize poly_adr, polymap_adr, polyvert_adr;

  // sizes outside call to mj_makeModel
  m->nemax       = nemax;
  m->njmax       = njmax;
  m->nconmax     = nconmax;
  m->nsensordata = nsensordata;
  m->nhistory    = nhistory;
  m->nuserdata   = nuserdata;
  m->na          = na;

  // find bvh_adr after bodies
  bvh_adr = 0;
  for (int i = 0; i < nbody; i++) {
    bvh_adr = mjMAX(bvh_adr, m->body_bvhadr[i] + m->body_bvhnum[i]);
  }

  // meshes
  oct_adr      = 0;
  vert_adr     = 0;
  normal_adr   = 0;
  texcoord_adr = 0;
  face_adr     = 0;
  graph_adr    = 0;
  poly_adr     = 0;
  polyvert_adr = 0;
  polymap_adr  = 0;
  for (int i = 0; i < nmesh; i++) {
    // get pointer
    mjCMesh* pme = meshes[i];

    // set fields
    m->mesh_polyadr[i]     = poly_adr;
    m->mesh_polynum[i]     = pme->npolygon();
    m->mesh_vertadr[i]     = vert_adr;
    m->mesh_vertnum[i]     = pme->nvert();
    m->mesh_normaladr[i]   = normal_adr;
    m->mesh_normalnum[i]   = pme->nnormal();
    m->mesh_texcoordadr[i] = (pme->HasTexcoord() ? texcoord_adr : -1);
    m->mesh_texcoordnum[i] = pme->ntexcoord();
    m->mesh_faceadr[i]     = face_adr;
    m->mesh_facenum[i]     = pme->nface();
    m->mesh_graphadr[i]    = (pme->szgraph() ? graph_adr : -1);
    m->mesh_bvhnum[i]      = pme->tree().Nbvh();
    m->mesh_bvhadr[i]      = pme->tree().Nbvh() ? bvh_adr : -1;
    m->mesh_octnum[i]      = pme->octree().NumNodes();
    m->mesh_octadr[i]      = pme->octree().NumNodes() ? oct_adr : -1;
    mjuu_copyvec(&m->mesh_scale[3 * i], pme->Scale(), 3);
    mjuu_copyvec(&m->mesh_pos[3 * i], pme->GetPosPtr(), 3);
    mjuu_copyvec(&m->mesh_quat[4 * i], pme->GetQuatPtr(), 4);

    // copy vertices, normals, faces, texcoords, aux data
    pme->CopyVert(m->mesh_vert + 3 * vert_adr);
    pme->CopyNormal(m->mesh_normal + 3 * normal_adr);
    pme->CopyFace(m->mesh_face + 3 * face_adr);
    pme->CopyFaceNormal(m->mesh_facenormal + 3 * face_adr);
    if (pme->HasTexcoord()) {
      pme->CopyTexcoord(m->mesh_texcoord + 2 * texcoord_adr);
      pme->CopyFaceTexcoord(m->mesh_facetexcoord + 3 * face_adr);
    } else {
      memset(m->mesh_facetexcoord + 3 * face_adr, 0, 3 * pme->nface() * sizeof(int));
    }
    memset(m->mesh_extrema + 27 * i, 0, 27 * sizeof(int));
    if (pme->szgraph()) {
      pme->CopyGraph(m->mesh_graph + graph_adr);

      // compute grid extrema (local indices in graph)
      float max_val[27];
      for (int k = 0; k < 27; k++) { max_val[k] = std::numeric_limits<float>::lowest(); }

      const int*   graph         = m->mesh_graph + graph_adr;
      int          numgraphvert  = graph[0];
      const int*   vert_globalid = graph + 2 + numgraphvert;
      const float* verts         = m->mesh_vert + 3 * vert_adr;

      // map the 27 features (8 vertices, 6 faces, 12 edges) of a unit cube to the farthest
      // vertex in the mesh
      for (int local_id = 0; local_id < numgraphvert; local_id++) {
        int global_id = vert_globalid[local_id];

        float x = verts[3 * global_id + 0];
        float y = verts[3 * global_id + 1];
        float z = verts[3 * global_id + 2];

        int k = 0;
        for (int cx = -1; cx <= 1; cx++) {
          for (int cy = -1; cy <= 1; cy++) {
            for (int cz = -1; cz <= 1; cz++) {
              float dot = x * cx + y * cy + z * cz;
              if (dot > max_val[k]) {
                max_val[k]                  = dot;
                m->mesh_extrema[27 * i + k] = local_id;
              }
              k++;
            }
          }
        }
      }
    }

    pme->CopyPolygonNormals(m->mesh_polynormal + 3 * poly_adr);
    pme->CopyPolygons(m->mesh_polyvert + polyvert_adr,
                      m->mesh_polyvertadr + poly_adr,
                      m->mesh_polyvertnum + poly_adr,
                      polyvert_adr);
    pme->CopyPolygonMap(m->mesh_polymap + polymap_adr,
                        m->mesh_polymapadr + vert_adr,
                        m->mesh_polymapnum + vert_adr,
                        polymap_adr);

    // copy bvh data
    if (pme->tree().Nbvh()) {
      memcpy(m->bvh_aabb + 6 * bvh_adr,
             pme->tree().Bvh().data(),
             6 * pme->tree().Nbvh() * sizeof(mjtNum));
      memcpy(m->bvh_child + 2 * bvh_adr,
             pme->tree().Child().data(),
             2 * pme->tree().Nbvh() * sizeof(int));
      memcpy(m->bvh_depth + bvh_adr, pme->tree().Level().data(), pme->tree().Nbvh() * sizeof(int));
      for (int j = 0; j < pme->tree().Nbvh(); j++) {
        m->bvh_nodeid[j + bvh_adr] = pme->tree().Nodeid(j) > -1 ? pme->tree().Nodeid(j) : -1;
      }
    }

    // copy octree data
    if (pme->octree().NumNodes()) {
      pme->octree().CopyAabb(m->oct_aabb + 6 * oct_adr);
      pme->octree().CopyChild(m->oct_child + 8 * oct_adr);
      pme->octree().CopyLevel(m->oct_depth + oct_adr);
      pme->octree().CopyCoeff(m->oct_coeff + 8 * oct_adr);
    }

    // advance counters
    poly_adr     += pme->npolygon();
    polyvert_adr += pme->npolygonvert();
    polymap_adr  += pme->npolygonmap();
    vert_adr     += pme->nvert();
    normal_adr   += pme->nnormal();
    texcoord_adr += (pme->HasTexcoord() ? pme->ntexcoord() : 0);
    face_adr     += pme->nface();
    graph_adr    += pme->szgraph();
    bvh_adr      += pme->tree().Nbvh();
    oct_adr      += pme->octree().NumNodes();
  }

  // flexes
  vert_adr      = 0;
  node_adr      = 0;
  edge_adr      = 0;
  elem_adr      = 0;
  elemdata_adr  = 0;
  elemedge_adr  = 0;
  shelldata_adr = 0;
  texcoord_adr  = 0;
  stiffness_adr = 0;
  bending_adr   = 0;
  int flex_padr = 0;
  for (int i = 0; i < njnt; i++) {
    if (joints_[i]->springyield > 0) { flex_padr += joints_[i]->nq(); }
  }
  for (int i = 0; i < ntendon; i++) {
    if (tendons_[i]->springyield > 0) { flex_padr += 2; }
  }
  for (int i = 0; i < nflex; i++) {
    // get pointer
    mjCFlex* pfl = flexes_[i];

    // set fields: geom-like
    m->flex_contype[i]     = pfl->contype;
    m->flex_conaffinity[i] = pfl->conaffinity;
    m->flex_condim[i]      = pfl->condim;
    memcpy(m->flex_matid + elem_adr, pfl->elemmatid_.data(), pfl->nelem * sizeof(int));
    m->flex_group[i]    = pfl->group;
    m->flex_priority[i] = pfl->priority;
    m->flex_solmix[i]   = (mjtNum)pfl->solmix;
    mjuu_copyvec(m->flex_solref + mjNREF * i, pfl->solref, mjNREF);
    mjuu_copyvec(m->flex_solimp + mjNIMP * i, pfl->solimp, mjNIMP);
    m->flex_radius[i] = (mjtNum)pfl->radius;
    mjuu_copyvec(m->flex_size + 3 * i, pfl->size, 3);
    mjuu_copyvec(m->flex_friction + 3 * i, pfl->friction, 3);
    m->flex_margin[i] = (mjtNum)pfl->margin;
    m->flex_gap[i]    = (mjtNum)pfl->gap;
    mjuu_copyvec(m->flex_rgba + 4 * i, pfl->rgba, 4);

    // elasticity
    if (pfl->stiffness.empty()) {
      m->flex_stiffnessadr[i] = -1;
    } else {
      m->flex_stiffnessadr[i] = stiffness_adr;
      mjuu_copyvec(m->flex_stiffness + m->flex_stiffnessadr[i],
                   pfl->stiffness.data(),
                   pfl->stiffness.size());
    }
    if (pfl->bending.empty()) {
      m->flex_bendingadr[i] = -1;
    } else {
      m->flex_bendingadr[i] = bending_adr;
      mjuu_copyvec(m->flex_bending + m->flex_bendingadr[i],
                   pfl->bending.data(),
                   pfl->bending.size());
    }
    m->flex_damping[i]     = (mjtNum)pfl->damping;
    m->flex_yield[i]       = (mjtNum)pfl->stressyield;
    int flex_pnum          = pfl->PlasticNum();
    m->flex_plasticadr[i]  = flex_pnum > 0 ? flex_padr : -1;
    pfl->plasticadr_       = m->flex_plasticadr[i];
    pfl->plasticnum_       = flex_pnum;
    flex_padr             += flex_pnum;

    // set fields: mesh-like
    m->flex_dim[i]          = pfl->dim;
    m->flex_vertadr[i]      = vert_adr;
    m->flex_vertnum[i]      = pfl->nvert;
    m->flex_nodeadr[i]      = node_adr;
    m->flex_nodenum[i]      = pfl->nnode;
    m->flex_edgeadr[i]      = edge_adr;
    m->flex_edgenum[i]      = pfl->nedge;
    m->flex_elemadr[i]      = elem_adr;
    m->flex_elemdataadr[i]  = elemdata_adr;
    m->flex_elemedgeadr[i]  = elemedge_adr;
    m->flex_shellnum[i]     = (int)pfl->shell.size() / pfl->dim;
    m->flex_shelldataadr[i] = m->flex_shellnum[i] ? shelldata_adr : -1;
    if (pfl->texcoord_.empty()) {
      m->flex_texcoordadr[i] = -1;
      memcpy(m->flex_elemtexcoord + elemdata_adr,
             pfl->elem_.data(),
             pfl->elem_.size() * sizeof(int));
    } else {
      m->flex_texcoordadr[i] = texcoord_adr;
      memcpy(m->flex_texcoord + 2 * texcoord_adr,
             pfl->texcoord_.data(),
             pfl->texcoord_.size() * sizeof(float));
      memcpy(m->flex_elemtexcoord + elemdata_adr,
             pfl->elemtexcoord_.data(),
             pfl->elemtexcoord_.size() * sizeof(int));
    }
    m->flex_elemnum[i] = pfl->nelem;
    memcpy(m->flex_elem + elemdata_adr, pfl->elem_.data(), pfl->elem_.size() * sizeof(int));
    memcpy(m->flex_elemedge + elemedge_adr,
           pfl->edgeidx_.data(),
           pfl->edgeidx_.size() * sizeof(int));
    memcpy(m->flex_elemlayer + elem_adr, pfl->elemlayer.data(), pfl->nelem * sizeof(int));
    if (m->flex_shellnum[i]) {
      memcpy(m->flex_shell + shelldata_adr, pfl->shell.data(), pfl->shell.size() * sizeof(int));
    }
    m->flex_edgestiffness[i] = (mjtNum)pfl->edgestiffness;
    m->flex_edgedamping[i]   = (mjtNum)pfl->edgedamping;
    m->flex_rigid[i]         = pfl->rigid;
    m->flex_centered[i]      = pfl->centered;
    m->flex_flatskin[i]      = pfl->flatskin;
    m->flex_selfcollide[i]   = pfl->selfcollide;
    m->flex_activelayers[i]  = pfl->activelayers;
    m->flex_passive[i]       = pfl->passive;
    m->flex_bvhnum[i]        = pfl->tree.Nbvh();
    m->flex_bvhadr[i]        = pfl->tree.Nbvh() ? bvh_adr : -1;

    // find equality constraint referencing this flex
    m->flex_edgeequality[i] = 0;
    for (int k = 0; k < (int)equalities_.size(); k++) {
      if (equalities_[k]->name1_ == pfl->name) {
        if (equalities_[k]->type == mjEQ_FLEX) {
          m->flex_edgeequality[i] = 1;
          break;
        }
        if (equalities_[k]->type == mjEQ_FLEXSTRAIN) {
          m->flex_edgeequality[i] = 3;
          break;
        }
      }
    }

    if (!pfl->rigid &&
        m->flex_edgeequality[i] == 0 &&
        !pfl->edgestiffness &&
        !pfl->edgedamping &&
        !pfl->damping &&
        pfl->bending.empty()) {
      AddWarning("flex '" +
                     pfl->name +
                     "' is not rigid and has no equality constraints or "
                     "passive forces",
                 pfl);
    }

    // copy bvh data (flex aabb computed dynamically in mjData)
    if (pfl->tree.Nbvh()) {
      memcpy(m->bvh_child + 2 * bvh_adr,
             pfl->tree.Child().data(),
             2 * pfl->tree.Nbvh() * sizeof(int));
      memcpy(m->bvh_depth + bvh_adr, pfl->tree.Level().data(), pfl->tree.Nbvh() * sizeof(int));
      for (int i = 0; i < pfl->tree.Nbvh(); i++) {
        m->bvh_nodeid[i + bvh_adr] = pfl->tree.Nodeidptr(i) ? *(pfl->tree.Nodeidptr(i)) : -1;
      }
    }

    // copy or set vert
    if (pfl->centered && !pfl->interpolated) {
      mjuu_zerovec(m->flex_vert + 3 * vert_adr, 3 * pfl->nvert);
    } else {
      mjuu_copyvec(m->flex_vert + 3 * vert_adr, pfl->vert_.data(), 3 * pfl->nvert);
    }

    // copy or set node
    if (pfl->centered && pfl->interpolated) {
      mjuu_zerovec(m->flex_node + 3 * node_adr, 3 * pfl->nnode);
    } else if (pfl->interpolated) {
      mjuu_copyvec(m->flex_node + 3 * node_adr, pfl->node_.data(), 3 * pfl->nnode);
    }

    // copy vert0 and vertcell
    mjuu_copyvec(m->flex_vert0 + 3 * vert_adr, pfl->vert0_.data(), 3 * pfl->nvert);
    memcpy(m->flex_vertcell + vert_adr, pfl->vertcell_.data(), pfl->nvert * sizeof(int));

    // copy node0
    mjuu_copyvec(m->flex_node0 + 3 * node_adr, pfl->node0_.data(), 3 * pfl->nnode);

    // copy or set vertbodyid
    if (pfl->rigid) {
      for (int k = 0; k < pfl->nvert; k++) {
        m->flex_vertbodyid[vert_adr + k] = pfl->vertbodyid[0];
      }
    } else {
      memcpy(m->flex_vertbodyid + vert_adr, pfl->vertbodyid.data(), pfl->nvert * sizeof(int));
    }

    // copy or set nodebodyid
    if (pfl->rigid) {
      for (int k = 0; k < pfl->nnode; k++) {
        m->flex_nodebodyid[node_adr + k] = pfl->nodebodyid[0];
      }
    } else {
      memcpy(m->flex_nodebodyid + node_adr, pfl->nodebodyid.data(), pfl->nnode * sizeof(int));
    }

    // set interpolation type: positive = volumetric, negative = shell mode
    m->flex_interp[i] = pfl->spec.elastic2d ? -pfl->spec.order : pfl->spec.order;

    if (m->flex_passive[i] && (m->flex_rigid[i] || m->flex_dim[i] < 2)) {
      AddWarning("flex '" +
                     pfl->name +
                     "' has passive contact, which is not supported for rigid "
                     "or 1D flexes: attribute ignored",
                 pfl);
    }

    // set cell count for multi-cell finite cell method
    m->flex_cellnum[3 * i + 0] = pfl->spec.cellcount[0];
    m->flex_cellnum[3 * i + 1] = pfl->spec.cellcount[1];
    m->flex_cellnum[3 * i + 2] = pfl->spec.cellcount[2];

    // convert edge pairs to int array, set edge rigid
    for (int k = 0; k < pfl->nedge; k++) {
      m->flex_edge[2 * (edge_adr + k)]     = pfl->edge[k].first;
      m->flex_edge[2 * (edge_adr + k) + 1] = pfl->edge[k].second;
      if (pfl->dim == 2 && (pfl->elastic2d == 1 || pfl->elastic2d == 3)) {
        m->flex_edgeflap[2 * (edge_adr + k) + 0] = pfl->flaps[k].vertices[2];
        m->flex_edgeflap[2 * (edge_adr + k) + 1] = pfl->flaps[k].vertices[3];
      } else {
        m->flex_edgeflap[2 * (edge_adr + k) + 0] = -1;
        m->flex_edgeflap[2 * (edge_adr + k) + 1] = -1;
      }

      if (pfl->rigid) {
        m->flexedge_rigid[edge_adr + k] = 1;
      } else if (!pfl->interpolated) {
        // check if vertex body weldids are the same
        // unsupported by trilinear interpolation
        int b1 = pfl->vertbodyid[pfl->edge[k].first];
        int b2 = pfl->vertbodyid[pfl->edge[k].second];

        m->flexedge_rigid[edge_adr + k] = (bodies_[b1]->weldid == bodies_[b2]->weldid);
      } else {
        m->flexedge_rigid[edge_adr + k] = 0;
      }
    }

    // advance counters
    vert_adr      += pfl->nvert;
    node_adr      += pfl->nnode;
    edge_adr      += pfl->nedge;
    elem_adr      += pfl->nelem;
    elemdata_adr  += (pfl->dim + 1) * pfl->nelem;
    elemedge_adr  += (pfl->kNumEdges[pfl->dim - 1]) * pfl->nelem;
    shelldata_adr += (int)pfl->shell.size();
    texcoord_adr  += (int)pfl->texcoord_.size() / 2;
    bvh_adr       += pfl->tree.Nbvh();
    stiffness_adr += pfl->stiffness.size();
    bending_adr   += pfl->bending.size();
  }

  // skins
  vert_adr     = 0;
  face_adr     = 0;
  texcoord_adr = 0;
  bone_adr     = 0;
  bonevert_adr = 0;
  for (int i = 0; i < nskin; i++) {
    // get pointer
    mjCSkin* psk = skins_[i];

    // set fields
    m->skin_matid[i] = psk->matid;
    m->skin_group[i] = psk->group;
    mjuu_copyvec(m->skin_rgba + 4 * i, psk->rgba, 4);
    m->skin_inflate[i]     = psk->inflate;
    m->skin_vertadr[i]     = vert_adr;
    m->skin_vertnum[i]     = psk->get_vert().size() / 3;
    m->skin_texcoordadr[i] = (!psk->get_texcoord().empty() ? texcoord_adr : -1);
    m->skin_faceadr[i]     = face_adr;
    m->skin_facenum[i]     = psk->get_face().size() / 3;
    m->skin_boneadr[i]     = bone_adr;
    m->skin_bonenum[i]     = psk->bodyid.size();

    // copy mesh data
    memcpy(m->skin_vert + 3 * vert_adr,
           psk->get_vert().data(),
           psk->get_vert().size() * sizeof(float));
    if (!psk->get_texcoord().empty())
      memcpy(m->skin_texcoord + 2 * texcoord_adr,
             psk->get_texcoord().data(),
             psk->get_texcoord().size() * sizeof(float));
    memcpy(m->skin_face + 3 * face_adr,
           psk->get_face().data(),
           psk->get_face().size() * sizeof(int));

    // copy bind poses and body ids
    memcpy(m->skin_bonebindpos + 3 * bone_adr,
           psk->get_bindpos().data(),
           psk->get_bindpos().size() * sizeof(float));
    memcpy(m->skin_bonebindquat + 4 * bone_adr,
           psk->get_bindquat().data(),
           psk->get_bindquat().size() * sizeof(float));
    memcpy(m->skin_bonebodyid + bone_adr, psk->bodyid.data(), psk->bodyid.size() * sizeof(int));

    // copy per-bone vertex data, advance vertex counter
    for (int j = 0; j < m->skin_bonenum[i]; j++) {
      // set fields
      m->skin_bonevertadr[bone_adr + j] = bonevert_adr;
      m->skin_bonevertnum[bone_adr + j] = (int)psk->get_vertid()[j].size();

      // copy data
      memcpy(m->skin_bonevertid + bonevert_adr,
             psk->get_vertid()[j].data(),
             psk->get_vertid()[j].size() * sizeof(int));
      memcpy(m->skin_bonevertweight + bonevert_adr,
             psk->get_vertweight()[j].data(),
             psk->get_vertid()[j].size() * sizeof(float));

      // advance counter
      bonevert_adr += m->skin_bonevertnum[bone_adr + j];
    }

    // advance mesh and bone counters
    vert_adr     += m->skin_vertnum[i];
    texcoord_adr += psk->get_texcoord().size() / 2;
    face_adr     += m->skin_facenum[i];
    bone_adr     += m->skin_bonenum[i];
  }

  // hfields
  data_adr = 0;
  for (int i = 0; i < nhfield; i++) {
    // get pointer
    mjCHField* phf = hfields[i];

    // set fields
    mjuu_copyvec(m->hfield_size + 4 * i, phf->size, 4);
    m->hfield_nrow[i] = phf->nrow;
    m->hfield_ncol[i] = phf->ncol;
    m->hfield_adr[i]  = data_adr;

    // copy elevation data
    memcpy(m->hfield_data + data_adr,
           phf->data.data(),
           static_cast<mjtSize>(phf->nrow) * phf->ncol * sizeof(float));

    // advance counter
    data_adr += phf->nrow * phf->ncol;
  }

  // textures
  data_adr = 0;
  for (int i = 0; i < ntex; i++) {
    // get pointer
    mjCTexture* ptex = textures_[i];

    // set fields
    m->tex_type[i]       = ptex->type;
    m->tex_colorspace[i] = ptex->colorspace;
    m->tex_height[i]     = ptex->height;
    m->tex_width[i]      = ptex->width;
    m->tex_nchannel[i]   = ptex->nchannel;
    m->tex_adr[i]        = data_adr;

    // by reference: no payload, the resolved path is in paths at tex_pathadr
    if (ptex->ByReference()) {
      m->tex_adr[i] = -1;
      continue;
    }

    // copy rgb data
    mjtSize nbytes = static_cast<mjtSize>(ptex->nchannel) * ptex->width * ptex->height;
    memcpy(m->tex_data + data_adr, ptex->data_.data(), nbytes);

    // the model has the pixels now; the next compile loads them again
    std::vector<std::byte>().swap(ptex->data_);

    // advance counter
    data_adr += nbytes;
  }

  // materials
  for (int i = 0; i < nmat; i++) {
    // get pointer
    mjCMaterial* pmat = materials_[i];

    // set fields
    for (int j = 0; j < mjNTEXROLE; j++) { m->mat_texid[mjNTEXROLE * i + j] = pmat->texid[j]; }
    m->mat_texuniform[i] = pmat->texuniform;
    mjuu_copyvec(m->mat_texrepeat + 2 * i, pmat->texrepeat, 2);
    m->mat_emission[i]    = pmat->emission;
    m->mat_specular[i]    = pmat->specular;
    m->mat_shininess[i]   = pmat->shininess;
    m->mat_reflectance[i] = pmat->reflectance;
    m->mat_metallic[i]    = pmat->metallic;
    m->mat_roughness[i]   = pmat->roughness;
    mjuu_copyvec(m->mat_rgba + 4 * i, pmat->rgba, 4);
  }

  // geom pairs to include, in the order of their ids
  for (const mjCPair* pair : pairs_) {
    int i                = pair->id;
    m->pair_dim[i]       = pair->condim;
    m->pair_geom1[i]     = pair->geom1->id;
    m->pair_geom2[i]     = pair->geom2->id;
    m->pair_signature[i] = pair->signature;
    mjuu_copyvec(m->pair_solref + mjNREF * i, pair->solref, mjNREF);
    mjuu_copyvec(m->pair_solreffriction + mjNREF * i, pair->solreffriction, mjNREF);
    mjuu_copyvec(m->pair_solimp + mjNIMP * i, pair->solimp, mjNIMP);
    m->pair_margin[i]   = (mjtNum)pair->margin;
    m->pair_gap[i]      = (mjtNum)pair->gap;
    m->pair_adhesion[i] = (mjtNum)pair->adhesion;
    mjuu_copyvec(m->pair_friction + 5 * i, pair->friction, 5);
  }

  // body pairs to exclude, in the order of their ids
  for (const mjCBodyPair* exclude : excludes_) {
    m->exclude_signature[exclude->id] = exclude->signature;
  }

  // equality constraints
  for (int i = 0; i < neq; i++) {
    // get pointer
    mjCEquality* peq = equalities_[i];

    // set fields
    m->eq_type[i]    = peq->type;
    m->eq_obj1id[i]  = peq->obj1id;
    m->eq_obj2id[i]  = peq->obj2id;
    m->eq_objtype[i] = peq->objtype;
    m->eq_active0[i] = peq->active;
    mjuu_copyvec(m->eq_solref + mjNREF * i, peq->solref, mjNREF);
    mjuu_copyvec(m->eq_solimp + mjNIMP * i, peq->solimp, mjNIMP);
    mjuu_copyvec(m->eq_data + mjNEQDATA * i, peq->data, mjNEQDATA);
  }

  // tendons and wraps
  adr             = 0;
  int plastic_adr = 0;
  for (int i = 0; i < njnt; i++) {
    if (joints_[i]->springyield > 0) { plastic_adr += joints_[i]->nq(); }
  }
  for (int i = 0; i < ntendon; i++) {
    // get pointer
    mjCTendon* pte = tendons_[i];

    // set fields
    m->tendon_type[i]       = pte->Type();
    m->tendon_adr[i]        = adr;
    m->tendon_num[i]        = (int)pte->path.size();
    m->tendon_matid[i]      = pte->matid;
    m->tendon_group[i]      = pte->group;
    m->tendon_plasticadr[i] = pte->springyield > 0 ? plastic_adr : -1;
    pte->plasticadr_        = m->tendon_plasticadr[i];
    if (pte->springyield > 0) { plastic_adr += 2; }
    m->tendon_limited[i]       = (mjtBool)pte->is_limited();
    m->tendon_actfrclimited[i] = (mjtBool)pte->is_actfrclimited();
    m->tendon_width[i]         = (mjtNum)pte->width;
    mjuu_copyvec(m->tendon_solref_lim + mjNREF * i, pte->solref_limit, mjNREF);
    mjuu_copyvec(m->tendon_solimp_lim + mjNIMP * i, pte->solimp_limit, mjNIMP);
    mjuu_copyvec(m->tendon_solref_fri + mjNREF * i, pte->solref_friction, mjNREF);
    mjuu_copyvec(m->tendon_solimp_fri + mjNIMP * i, pte->solimp_friction, mjNIMP);
    m->tendon_range[2 * i]           = (mjtNum)pte->range[0];
    m->tendon_range[2 * i + 1]       = (mjtNum)pte->range[1];
    m->tendon_actfrcrange[2 * i]     = (mjtNum)pte->actfrcrange[0];
    m->tendon_actfrcrange[2 * i + 1] = (mjtNum)pte->actfrcrange[1];
    m->tendon_margin[i]              = (mjtNum)pte->margin;
    m->tendon_stiffness[i]           = (mjtNum)pte->stiffness[0];
    mjuu_copyvec(m->tendon_stiffnesspoly + mjNPOLY * i, pte->stiffness + 1, mjNPOLY);
    m->tendon_damping[i] = (mjtNum)pte->damping[0];
    mjuu_copyvec(m->tendon_dampingpoly + mjNPOLY * i, pte->damping + 1, mjNPOLY);
    m->tendon_armature[i]             = (mjtNum)pte->armature;
    m->tendon_frictionloss[i]         = (mjtNum)pte->frictionloss;
    m->tendon_lengthspring[2 * i]     = (mjtNum)pte->springlength[0];
    m->tendon_lengthspring[2 * i + 1] = (mjtNum)pte->springlength[1];
    m->tendon_springyield[i]          = (mjtNum)pte->springyield;
    mjuu_copyvec(m->tendon_user + nuser_tendon * i, pte->get_userdata().data(), nuser_tendon);
    mjuu_copyvec(m->tendon_rgba + 4 * i, pte->rgba, 4);

    // set wraps
    for (int j = 0; j < (int)pte->path.size(); j++) {
      m->wrap_type[adr + j]  = pte->path[j]->Type();
      m->wrap_objid[adr + j] = pte->path[j]->obj ? pte->path[j]->obj->id : -1;
      m->wrap_prm[adr + j]   = (mjtNum)(pte->path[j]->prm * pte->path[j]->prmscale);
      if (pte->path[j]->Type() == mjWRAP_SPHERE || pte->path[j]->Type() == mjWRAP_CYLINDER) {
        m->wrap_prm[adr + j] = (mjtNum)pte->path[j]->sideid;
      }
    }

    // advance address counter
    adr += (int)pte->path.size();
  }

  // actuators
  adr           = 0;
  int delay_adr = 0;
  int ctrladr   = 0;
  int outadr    = 0;
  for (int i = 0; i < nactuator; i++) {
    // get pointer
    mjCActuator* pac = actuators_[i];

    // set fields
    m->actuator_trntype[i]        = pac->so3_ ? mjTRN_SO3 : pac->trntype;
    m->actuator_dyntype[i]        = pac->dyntype;
    m->actuator_gaintype[i]       = pac->gaintype;
    m->actuator_biastype[i]       = pac->biastype;
    m->actuator_trnid[2 * i]      = pac->trnid[0];
    m->actuator_trnid[2 * i + 1]  = pac->trnid[1];
    m->actuator_actnum[i]         = pac->actdim;
    m->actuator_actadr[i]         = m->actuator_actnum[i] ? adr : -1;
    pac->actadr_                  = m->actuator_actadr[i];
    pac->actdim_                  = m->actuator_actnum[i];
    adr                          += m->actuator_actnum[i];
    m->actuator_group[i]          = pac->group;

    // input and output blocks
    m->actuator_ctrladr[i]   = ctrladr;
    m->actuator_ctrlnum[i]   = pac->ctrlnum_;
    m->actuator_ctrlspec[i]  = pac->ctrlspec_;
    pac->ctrladr_            = pac->ctrlnum_ ? ctrladr : -1;
    ctrladr                 += pac->ctrlnum_;
    m->actuator_outadr[i]    = outadr;
    m->actuator_outnum[i]    = pac->outnum_;
    pac->outadr_             = outadr;
    outadr                  += pac->outnum_;

    // historyadr
    m->actuator_delay[i]           = (mjtNum)pac->delay;
    m->actuator_history[2 * i]     = pac->nsample;
    m->actuator_history[2 * i + 1] = pac->interp;
    if (pac->nsample > 0) {
      m->actuator_historyadr[i] = delay_adr;
      pac->historyadr_          = delay_adr;
      pac->historynum_          = 2 + pac->nsample + pac->nsample * pac->ctrlnum_;
      delay_adr +=
          2 + pac->nsample + pac->nsample * pac->ctrlnum_;  // [user, cursor, times, values]
    } else {
      m->actuator_historyadr[i] = -1;
      pac->historyadr_          = -1;
      pac->historynum_          = 0;
    }

    m->actuator_actlimited[i]  = (mjtBool)pac->is_actlimited();
    m->actuator_actearly[i]    = pac->actearly;
    m->actuator_cranklength[i] = (mjtNum)pac->cranklength;
    m->actuator_damping[i]     = (mjtNum)pac->damping[0];
    mjuu_copyvec(m->actuator_dampingpoly + mjNPOLY * i, pac->damping + 1, mjNPOLY);
    m->actuator_armature[i] = (mjtNum)pac->armature;
    mjuu_copyvec(m->actuator_dynprm + mjNDYN * i, pac->dynprm, mjNDYN);
    mjuu_copyvec(m->actuator_gainprm + mjNGAIN * i, pac->gainprm, mjNGAIN);
    mjuu_copyvec(m->actuator_biasprm + mjNBIAS * i, pac->biasprm, mjNBIAS);
    mjuu_copyvec(m->actuator_actrange + 2 * i, pac->actrange, 2);
    m->actuator_forcelimited[i] = (mjtBool)pac->is_forcelimited();
    mjuu_copyvec(m->actuator_forcerange + 2 * i, pac->forcerange, 2);
    mjuu_copyvec(m->actuator_user + nuser_actuator * i, pac->get_userdata().data(), nuser_actuator);

    // per-input arrays, at the actuator's ctrl block
    for (int k = 0; k < m->actuator_ctrlnum[i]; k++) {
      int j                      = m->actuator_ctrladr[i] + k;
      m->actuator_ctrllimited[j] = (mjtBool)pac->ctrllimiteds_[k];
      mjuu_copyvec(m->actuator_ctrlrange + 2 * j, pac->ctrlranges_[k], 2);
    }

    // per-output arrays, at the actuator's output block
    for (int j = m->actuator_outadr[i]; j < m->actuator_outadr[i] + m->actuator_outnum[i]; j++) {
      mjuu_copyvec(m->actuator_gear + 6 * j, pac->gear, 6);
      mjuu_copyvec(m->actuator_lengthrange + 2 * j, pac->lengthrange, 2);
    }
  }

  // sensors
  adr = 0;
  for (int i = 0; i < nsensor; i++) {
    // get pointer
    mjCSensor* psen = sensors_[i];

    // set fields
    m->sensor_type[i]      = psen->type;
    m->sensor_datatype[i]  = psen->datatype;
    m->sensor_needstage[i] = psen->needstage;
    m->sensor_objtype[i]   = psen->objtype;
    m->sensor_objid[i]     = psen->obj ? psen->obj->id : -1;
    m->sensor_reftype[i]   = psen->reftype;
    m->sensor_refid[i]     = psen->ref ? psen->ref->id : -1;
    mjuu_copyvec(m->sensor_intprm + i * mjNSENS, psen->intprm, mjNSENS);
    m->sensor_dim[i]    = psen->dim;
    m->sensor_cutoff[i] = (mjtNum)psen->cutoff;
    m->sensor_noise[i]  = (mjtNum)psen->noise;

    // history buffer
    m->sensor_delay[i]            = (mjtNum)psen->delay;
    m->sensor_history[2 * i]      = psen->nsample;
    m->sensor_history[2 * i + 1]  = psen->interp;
    m->sensor_interval[2 * i]     = (mjtNum)psen->interval[0];
    m->sensor_interval[2 * i + 1] = (mjtNum)psen->interval[1];
    if (psen->nsample > 0) {
      m->sensor_historyadr[i] = delay_adr;
      int dim                 = psen->dim;
      psen->historyadr_       = delay_adr;
      psen->historynum_       = 2 + psen->nsample + psen->nsample * dim;
      delay_adr +=
          2 + psen->nsample + psen->nsample * dim;  // [user, cursor, times(n), values(n*dim)]
    } else {
      m->sensor_historyadr[i] = -1;
      psen->historyadr_       = -1;
      psen->historynum_       = 0;
    }

    mjuu_copyvec(m->sensor_user + nuser_sensor * i, psen->get_userdata().data(), nuser_sensor);

    // calculate address and advance
    m->sensor_adr[i]  = adr;
    adr              += psen->dim;
  }

  // numeric fields
  adr = 0;
  for (int i = 0; i < nnumeric; i++) {
    // get pointer
    mjCNumeric* pcu = numerics_[i];

    // set fields
    m->numeric_adr[i]  = adr;
    m->numeric_size[i] = pcu->size;
    for (int j = 0; j < (int)pcu->data_.size(); j++) {
      m->numeric_data[adr + j] = (mjtNum)pcu->data_[j];
    }
    for (int j = (int)pcu->data_.size(); j < (int)pcu->size; j++) { m->numeric_data[adr + j] = 0; }

    // advance address counter
    adr += m->numeric_size[i];
  }

  // text fields
  adr = 0;
  for (int i = 0; i < ntext; i++) {
    // get pointer
    mjCText* pte = texts_[i];

    // set fields
    m->text_adr[i]  = adr;
    m->text_size[i] = (int)pte->data_.size() + 1;
    mju_strncpy(m->text_data + adr, pte->data_.c_str(), m->ntextdata - adr);

    // advance address counter
    adr += m->text_size[i];
  }

  // tuple fields
  adr = 0;
  for (int i = 0; i < ntuple; i++) {
    // get pointer
    mjCTuple* ptu = tuples_[i];

    // set fields
    m->tuple_adr[i]  = adr;
    m->tuple_size[i] = (int)ptu->objtype_.size();
    for (int j = 0; j < m->tuple_size[i]; j++) {
      m->tuple_objtype[adr + j] = (int)ptu->objtype_[j];
      m->tuple_objid[adr + j]   = ptu->obj[j]->id;
      m->tuple_objprm[adr + j]  = (mjtNum)ptu->objprm_[j];
    }

    // advance address counter
    adr += m->tuple_size[i];
  }

  // copy keyframe data
  for (int i = 0; i < nkey; i++) {
    // copy data
    m->key_time[i] = (mjtNum)keys_[i]->time;
    mjuu_copyvec(m->key_qpos + i * nq, keys_[i]->qpos_.data(), nq);
    mjuu_copyvec(m->key_qvel + i * nv, keys_[i]->qvel_.data(), nv);
    if (na) { mjuu_copyvec(m->key_act + i * na, keys_[i]->act_.data(), na); }
    if (nmocap) {
      mjuu_copyvec(m->key_mpos + i * 3 * nmocap, keys_[i]->mpos_.data(), 3 * nmocap);
      mjuu_copyvec(m->key_mquat + i * 4 * nmocap, keys_[i]->mquat_.data(), 4 * nmocap);
    }

    // normalize quaternions in m->key_qpos
    for (int j = 0; j < m->njnt; j++) {
      if (m->jnt_type[j] == mjJNT_BALL || m->jnt_type[j] == mjJNT_FREE) {
        mjuu_normvec(m->key_qpos + i * nq + m->jnt_qposadr[j] + 3 * (m->jnt_type[j] == mjJNT_FREE),
                     4);
      }
    }

    // normalize quaternions in m->key_mquat
    for (int j = 0; j < nmocap; j++) { mjuu_normvec(m->key_mquat + i * 4 * nmocap + 4 * j, 4); }

    mjuu_copyvec(m->key_ctrl + i * nu, keys_[i]->ctrl_.data(), nu);
    if (nplastic) {
      mjuu_copyvec(m->key_plastic + i * nplastic, keys_[i]->plastic_.data(), nplastic);
      for (int j = 0; j < m->njnt; j++) {
        if (m->jnt_type[j] == mjJNT_BALL && m->jnt_plasticadr[j] >= 0) {
          mjtNum* q = m->key_plastic + i * nplastic + m->jnt_plasticadr[j];
          if (mjuu_defined(q[0]) &&
              mjuu_defined(q[1]) &&
              mjuu_defined(q[2]) &&
              mjuu_defined(q[3])) {
            mjuu_normvec(q, 4);
          }
        }
      }
    }
  }

  // save qpos0 in user model (to recognize changed key_qpos in write)
  qpos0.resize(nq);
  body_pos0.resize(3 * nbody);
  body_quat0.resize(4 * nbody);
  mjuu_copyvec(qpos0.data(), m->qpos0, nq);
  mjuu_copyvec(body_pos0.data(), m->body_pos, 3 * nbody);
  mjuu_copyvec(body_quat0.data(), m->body_quat, 4 * nbody);
}
// NOLINTEND(readability/fn_size)


// finalize simple bodies/dofs including tendon information
void mjCModel::FinalizeSimple(mjModel* m) {
  // demote bodies affected by inertia-bearing tendon to non-simple, the armature of actuators on
  // the tendon included, as in ComputeSparseSizes
  for (int i = 0; i < ntendon; i++) {
    bool armature = m->tendon_armature[i] != 0;
    for (int k = 0; k < nactuator; k++) {
      if (m->actuator_trntype[k] == mjTRN_TENDON &&
          m->actuator_trnid[2 * k] == i &&
          m->actuator_armature[k] != 0) {
        armature = true;
      }
    }
    if (!armature) { continue; }
    int adr = m->tendon_adr[i];
    int num = m->tendon_num[i];
    for (int j = adr; j < adr + num; j++) {
      int objid = m->wrap_objid[j];
      if (m->wrap_type[j] == mjWRAP_JOINT) { m->body_simple[m->jnt_bodyid[objid]] = 0; }
      if (m->wrap_type[j] == mjWRAP_SITE) { m->body_simple[m->site_bodyid[objid]] = 0; }
      if (m->wrap_type[j] == mjWRAP_CYLINDER || m->wrap_type[j] == mjWRAP_SPHERE) {
        m->body_simple[m->geom_bodyid[objid]] = 0;
      }
    }
  }

  // set dof_simplenum
  int count = 0;
  for (int i = nv - 1; i >= 0; i--) {
    if (m->body_simple[m->dof_bodyid[i]]) {
      count++;  // increment counter
    } else {
      count = 0;  // reset
    }
    m->dof_simplenum[i] = count;
  }

  // recompute nC given {dof_simplenum, dof_parentid}, validate
  int nOD = 0;  // number of non-simple off-diagonal parent dofs
  for (int i = 0; i < nv; i++) {
    // count ancestor (off-diagonal) dofs
    if (!m->dof_simplenum[i]) {
      int j = i;
      while (j >= 0) {
        if (j != i) nOD++;
        j = m->dof_parentid[j];
      }
    }
  }
  int nC_post = nOD + nv;
  if (nC_post != nC) throw mjCError(0, "nC mismatch: pre %d, post %d", nullptr, nC, nC_post);
}


// save the current state
template <class T>
void mjCModel::SaveState(const std::string& state_name,
                         const T*           qpos,
                         const T*           qvel,
                         const T*           act,
                         const T*           ctrl,
                         const T*           mpos,
                         const T*           mquat,
                         const T*           plastic,
                         bool               partial) {
  // a component which is not given is forgotten, unless the state is partial: then it stays
  // as it was saved before
  if (partial && !qpos && !qvel && !act && !ctrl && !mpos && !mquat && !plastic) { return; }

  // save qpos, qvel, and joint plastic
  for (auto joint : joints_) {
    if (joint->qposadr_ < -1 || joint->dofadr_ < -1 || joint->plasticadr_ < -1) {
      throw mjCError(nullptr, "SaveState: joint %s has invalid address", joint->name.c_str());
    }
    if (qpos && joint->qposadr_ != -1) {
      mjuu_copyvec(joint->qpos(state_name), qpos + joint->qposadr_, joint->nq());
    } else if (qpos || !partial) {
      joint->qpos(state_name)[0] = mjNAN;
    }
    if (qvel && joint->dofadr_ != -1) {
      mjuu_copyvec(joint->qvel(state_name), qvel + joint->dofadr_, joint->nv());
    } else if (qvel || !partial) {
      joint->qvel(state_name)[0] = mjNAN;
    }
    if (plastic && joint->plasticadr_ != -1) {
      mjuu_copyvec(joint->plastic(state_name), plastic + joint->plasticadr_, joint->nq());
    } else if (plastic || !partial) {
      joint->plastic(state_name)[0] = mjNAN;
    }
  }

  // save tendon plastic
  for (auto tendon : tendons_) {
    if (plastic && tendon->plasticadr_ != -1) {
      mjuu_copyvec(tendon->plastic(state_name), plastic + tendon->plasticadr_, 2);
    } else if (plastic || !partial) {
      tendon->plastic(state_name)[0] = mjNAN;
    }
  }

  // save flex plastic
  for (auto flex : flexes_) {
    if (plastic && flex->plasticadr_ != -1 && flex->plasticnum_ > 0) {
      flex->plastic(state_name).assign(flex->plasticnum_, 0);
      mjuu_copyvec(flex->plastic(state_name).data(),
                   plastic + flex->plasticadr_,
                   flex->plasticnum_);
    } else if (plastic || !partial) {
      flex->plastic(state_name).clear();
    }
  }

  // save act and ctrl
  for (unsigned int i = 0; i < actuators_.size(); i++) {
    auto actuator = actuators_[i];
    if (actuator->actadr_ != -1 && actuator->actdim_ > 0 && act) {
      actuator->act(state_name).assign(actuator->actdim_, 0);
      mjuu_copyvec(actuator->act(state_name).data(), act + actuator->actadr_, actuator->actdim_);
    } else if (act || !partial) {
      actuator->act(state_name).clear();
    }
    if (actuator->ctrladr_ != -1 && actuator->ctrlnum_ > 0 && ctrl) {
      actuator->ctrl(state_name).assign(actuator->ctrlnum_, 0);
      mjuu_copyvec(actuator->ctrl(state_name).data(),
                   ctrl + actuator->ctrladr_,
                   actuator->ctrlnum_);
    } else if (ctrl || !partial) {
      actuator->ctrl(state_name).clear();
    }
  }

  // save mocap pos and quat
  for (auto body : bodies_) {
    if (!body->spec.mocap || body->mocapid == -1) {
      if (mpos || !partial) { body->mpos(state_name)[0] = mjNAN; }
      if (mquat || !partial) { body->mquat(state_name)[0] = mjNAN; }
      continue;
    }
    if (mpos) { mjuu_copyvec(body->mpos(state_name), mpos + 3 * body->mocapid, 3); }
    if (mquat) { mjuu_copyvec(body->mquat(state_name), mquat + 4 * body->mocapid, 4); }
  }
}


// save full integration state for mj_recompile
void mjCModel::SaveState(const std::string& state_name,
                         const mjModel*     m,
                         const mjData*      d,
                         mjRecompileState*  state) {
  // invalidate addresses of joints whose type changed
  for (auto joint : joints_) {
    if (joint->type != joint->spec.type) {
      joint->qposadr_    = -1;
      joint->dofadr_     = -1;
      joint->plasticadr_ = -1;
    }
  }

  // save standard state components; they are restored as they are, so if a frame scale changed
  // between the compilations, the lengths in the state of what it holds are not rescaled
  SaveState(state_name, d->qpos, d->qvel, d->act, d->ctrl, d->mocap_pos, d->mocap_quat, d->plastic);

  // save time and userdata
  state->time = d->time;
  state->userdata.clear();
  if (nuserdata > 0 && d->userdata) {
    state->userdata.assign(d->userdata, d->userdata + nuserdata);
  }

  // save qfrc_applied, qacc_warmstart, and jnt_plastic
  state->qfrc_applied.clear();
  state->qacc_warmstart.clear();
  state->jnt_plastic.clear();
  for (auto joint : joints_) {
    if (joint->dofadr_ != -1 && joint->nv() > 0) {
      if (d->qfrc_applied) {
        state->qfrc_applied[joint].assign(d->qfrc_applied + joint->dofadr_,
                                          d->qfrc_applied + joint->dofadr_ + joint->nv());
      }
      if (d->qacc_warmstart) {
        state->qacc_warmstart[joint].assign(d->qacc_warmstart + joint->dofadr_,
                                            d->qacc_warmstart + joint->dofadr_ + joint->nv());
      }
    }
    if (joint->plasticadr_ != -1 && joint->nq() > 0 && d->plastic) {
      state->jnt_plastic[joint].assign(d->plastic + joint->plasticadr_,
                                       d->plastic + joint->plasticadr_ + joint->nq());
    }
  }

  // save tendon_plastic
  state->tendon_plastic.clear();
  for (auto tendon : tendons_) {
    if (tendon->plasticadr_ != -1 && d->plastic) {
      mjuu_copyvec(state->tendon_plastic[tendon].data(), d->plastic + tendon->plasticadr_, 2);
    }
  }

  // save flex_plastic
  state->flex_plastic.clear();
  for (auto flex : flexes_) {
    if (flex->plasticadr_ != -1 && flex->plasticnum_ > 0 && d->plastic) {
      state->flex_plastic[flex].assign(d->plastic + flex->plasticadr_,
                                       d->plastic + flex->plasticadr_ + flex->plasticnum_);
    }
  }

  // save xfrc_applied
  state->xfrc_applied.clear();
  for (auto body : bodies_) {
    if (body->bodyadr_ != -1 && d->xfrc_applied) {
      mjuu_copyvec(state->xfrc_applied[body].data(), d->xfrc_applied + 6 * body->bodyadr_, 6);
    }
  }

  // save eq_active
  state->eq_active.clear();
  for (auto equality : equalities_) {
    if (equality->eqadr_ != -1 && d->eq_active) {
      state->eq_active[equality] = d->eq_active[equality->eqadr_];
    }
  }

  // save actuator history
  state->actuator_history.clear();
  for (auto actuator : actuators_) {
    if (actuator->historyadr_ != -1 && actuator->historynum_ > 0 && d->history) {
      state->actuator_history[actuator].assign(
          d->history + actuator->historyadr_,
          d->history + actuator->historyadr_ + actuator->historynum_);
    }
  }

  // save sensor history
  state->sensor_history.clear();
  for (auto sensor : sensors_) {
    if (sensor->historyadr_ != -1 && sensor->historynum_ > 0 && d->history) {
      state->sensor_history[sensor].assign(d->history + sensor->historyadr_,
                                           d->history + sensor->historyadr_ + sensor->historynum_);
    }
  }

  // save plugin state
  state->plugin_state.clear();
  for (auto plugin : plugins_) {
    if (plugin->stateadr_ != -1 && plugin->statenum_ > 0 && d->plugin_state) {
      state->plugin_state[plugin].assign(d->plugin_state + plugin->stateadr_,
                                         d->plugin_state + plugin->stateadr_ + plugin->statenum_);
    }
  }
}


// clear existing data
void mjCModel::MakeData(const mjModel* m, mjData** dest) {
  mj_makeRawData(dest, m);
  mjData* d = *dest;
  if (d) {
    // d belongs to the caller: report a failed plugin init as a compile error
    int failed = mj_initPlugin(m, d);
    if (failed >= 0) { throw mjCError(0, "plugin init failed for plugin id %d", nullptr, failed); }
    mj_resetData(m, d);
  }
}


// restore the previous state
template <class T>
void mjCModel::RestoreState(const std::string& state_name,
                            const mjtNum*      pos0,
                            const mjtNum*      mpos0,
                            const mjtNum*      mquat0,
                            T*                 qpos,
                            T*                 qvel,
                            T*                 act,
                            T*                 ctrl,
                            T*                 mpos,
                            T*                 mquat,
                            T*                 plastic) {
  // restore qpos, qvel, and joint plastic
  for (auto joint : joints_) {
    if (qpos) {
      if (mjuu_defined(joint->qpos(state_name)[0])) {
        mjuu_copyvec(qpos + joint->qposadr_, joint->qpos(state_name), joint->nq());
      } else {
        mjuu_copyvec(qpos + joint->qposadr_, pos0 + joint->qposadr_, joint->nq());
      }
    }
    if (mjuu_defined(joint->qvel(state_name)[0]) && qvel) {
      mjuu_copyvec(qvel + joint->dofadr_, joint->qvel(state_name), joint->nv());
    }
    if (plastic && joint->plasticadr_ != -1) {
      if (mjuu_defined(joint->plastic(state_name)[0])) {
        mjuu_copyvec(plastic + joint->plasticadr_, joint->plastic(state_name), joint->nq());
      } else {
        for (int j = 0; j < joint->nq(); j++) { plastic[joint->plasticadr_ + j] = (T)mjNAN; }
      }
    }
  }

  // restore tendon plastic
  for (auto tendon : tendons_) {
    if (plastic && tendon->plasticadr_ != -1) {
      if (mjuu_defined(tendon->plastic(state_name)[0])) {
        mjuu_copyvec(plastic + tendon->plasticadr_, tendon->plastic(state_name), 2);
      } else {
        plastic[tendon->plasticadr_ + 0] = (T)mjNAN;
        plastic[tendon->plasticadr_ + 1] = (T)mjNAN;
      }
    }
  }

  // restore flex plastic
  for (auto flex : flexes_) {
    if (plastic && flex->plasticadr_ != -1 && flex->plasticnum_ > 0) {
      if (!flex->plastic(state_name).empty() && mjuu_defined(flex->plastic(state_name)[0])) {
        int n = std::min((int)flex->plastic(state_name).size(), flex->plasticnum_);
        mjuu_copyvec(plastic + flex->plasticadr_, flex->plastic(state_name).data(), n);
        for (int j = n; j < flex->plasticnum_; j++) { plastic[flex->plasticadr_ + j] = (T)mjNAN; }
      } else {
        for (int j = 0; j < flex->plasticnum_; j++) { plastic[flex->plasticadr_ + j] = (T)mjNAN; }
      }
    }
  }

  // restore act and ctrl
  for (unsigned int i = 0; i < actuators_.size(); i++) {
    auto actuator = actuators_[i];

    // restore act
    if (!actuator->act(state_name).empty() &&
        mjuu_defined(actuator->act(state_name)[0]) &&
        actuator->actadr_ != -1 &&
        actuator->actdim_ > 0 &&
        act) {
      int n = std::min((int)actuator->act(state_name).size(), actuator->actdim_);
      mjuu_copyvec(act + actuator->actadr_, actuator->act(state_name).data(), n);
    }

    // restore ctrl
    if (actuator->ctrladr_ != -1 && actuator->ctrlnum_ > 0 && ctrl) {
      if (!actuator->ctrl(state_name).empty() && mjuu_defined(actuator->ctrl(state_name)[0])) {
        int n = std::min((int)actuator->ctrl(state_name).size(), actuator->ctrlnum_);
        mjuu_copyvec(ctrl + actuator->ctrladr_, actuator->ctrl(state_name).data(), n);
        for (int j = n; j < actuator->ctrlnum_; j++) { ctrl[actuator->ctrladr_ + j] = 0; }
      } else {
        for (int j = 0; j < actuator->ctrlnum_; j++) { ctrl[actuator->ctrladr_ + j] = 0; }
      }
    }
  }

  // restore mocap pos and quat
  for (unsigned int i = 0; i < bodies_.size(); i++) {
    auto body = bodies_[i];
    if (!body->spec.mocap) { continue; }
    if (mpos) {
      if (mjuu_defined(body->mpos(state_name)[0])) {
        mjuu_copyvec(mpos + 3 * body->mocapid, body->mpos(state_name), 3);
      } else {
        mjuu_copyvec(mpos + 3 * body->mocapid, mpos0 + 3 * i, 3);
      }
    }
    if (mquat) {
      if (mjuu_defined(body->mquat(state_name)[0])) {
        mjuu_copyvec(mquat + 4 * body->mocapid, body->mquat(state_name), 4);
      } else {
        mjuu_copyvec(mquat + 4 * body->mocapid, mquat0 + 4 * i, 4);
      }
    }
  }
}


// restore full integration state for mj_recompile
void mjCModel::RestoreState(const std::string&      state_name,
                            const mjModel*          m,
                            mjData*                 d,
                            const mjRecompileState* state) {
  // restore standard state components
  RestoreState(state_name,
               m->qpos0,
               m->body_pos,
               m->body_quat,
               d->qpos,
               d->qvel,
               d->act,
               d->ctrl,
               d->mocap_pos,
               d->mocap_quat,
               static_cast<mjtNum*>(nullptr));

  // restore time and userdata
  d->time = state->time;
  if (!state->userdata.empty() && m->nuserdata > 0 && d->userdata) {
    mjtSize n = std::min((mjtSize)state->userdata.size(), m->nuserdata);
    mjuu_copyvec(d->userdata, state->userdata.data(), n);
  }

  // restore qfrc_applied, qacc_warmstart, and jnt_plastic
  for (auto joint : joints_) {
    if (joint->dofadr_ != -1 && joint->nv() > 0) {
      auto it_qfrc = state->qfrc_applied.find(joint);
      if (it_qfrc != state->qfrc_applied.end() &&
          (int)it_qfrc->second.size() == joint->nv() &&
          d->qfrc_applied) {
        mjuu_copyvec(d->qfrc_applied + joint->dofadr_, it_qfrc->second.data(), joint->nv());
      }
      auto it_warm = state->qacc_warmstart.find(joint);
      if (it_warm != state->qacc_warmstart.end() &&
          (int)it_warm->second.size() == joint->nv() &&
          d->qacc_warmstart) {
        mjuu_copyvec(d->qacc_warmstart + joint->dofadr_, it_warm->second.data(), joint->nv());
      }
    }
    if (joint->plasticadr_ != -1 && joint->nq() > 0 && d->plastic) {
      auto it_plastic = state->jnt_plastic.find(joint);
      if (it_plastic != state->jnt_plastic.end() && (int)it_plastic->second.size() == joint->nq()) {
        mjuu_copyvec(d->plastic + joint->plasticadr_, it_plastic->second.data(), joint->nq());
      }
    }
  }

  // restore tendon_plastic
  for (auto tendon : tendons_) {
    if (tendon->plasticadr_ != -1 && d->plastic) {
      auto it = state->tendon_plastic.find(tendon);
      if (it != state->tendon_plastic.end()) {
        mjuu_copyvec(d->plastic + tendon->plasticadr_, it->second.data(), 2);
      }
    }
  }

  // restore flex_plastic
  for (auto flex : flexes_) {
    if (flex->plasticadr_ != -1 && flex->plasticnum_ > 0 && d->plastic) {
      auto it = state->flex_plastic.find(flex);
      if (it != state->flex_plastic.end() && (int)it->second.size() == flex->plasticnum_) {
        mjuu_copyvec(d->plastic + flex->plasticadr_, it->second.data(), flex->plasticnum_);
      }
    }
  }

  // restore xfrc_applied
  for (auto body : bodies_) {
    if (body->bodyadr_ != -1 && d->xfrc_applied) {
      auto it = state->xfrc_applied.find(body);
      if (it != state->xfrc_applied.end()) {
        mjuu_copyvec(d->xfrc_applied + 6 * body->bodyadr_, it->second.data(), 6);
      }
    }
  }

  // restore eq_active
  for (auto equality : equalities_) {
    if (equality->eqadr_ != -1 && d->eq_active) {
      auto it = state->eq_active.find(equality);
      if (it != state->eq_active.end()) { d->eq_active[equality->eqadr_] = it->second; }
    }
  }

  // restore actuator history
  for (auto actuator : actuators_) {
    if (actuator->historyadr_ != -1 && actuator->historynum_ > 0 && d->history) {
      auto it = state->actuator_history.find(actuator);
      if (it != state->actuator_history.end() && (int)it->second.size() == actuator->historynum_) {
        mjuu_copyvec(d->history + actuator->historyadr_, it->second.data(), actuator->historynum_);
      }
    }
  }

  // restore sensor history
  for (auto sensor : sensors_) {
    if (sensor->historyadr_ != -1 && sensor->historynum_ > 0 && d->history) {
      auto it = state->sensor_history.find(sensor);
      if (it != state->sensor_history.end() && (int)it->second.size() == sensor->historynum_) {
        mjuu_copyvec(d->history + sensor->historyadr_, it->second.data(), sensor->historynum_);
      }
    }
  }

  // restore plugin state
  for (auto plugin : plugins_) {
    if (plugin->stateadr_ != -1 && plugin->statenum_ > 0 && d->plugin_state) {
      auto it = state->plugin_state.find(plugin);
      if (it != state->plugin_state.end() && (int)it->second.size() == plugin->statenum_) {
        mjuu_copyvec(d->plugin_state + plugin->stateadr_, it->second.data(), plugin->statenum_);
      }
    }
  }
}

// force explicit instantiations
template void mjCModel::SaveState<mjtNum>(const std::string& name,
                                          const mjtNum*      qpos,
                                          const mjtNum*      qvel,
                                          const mjtNum*      act,
                                          const mjtNum*      ctrl,
                                          const mjtNum*      mpos,
                                          const mjtNum*      mquat,
                                          const mjtNum*      plastic,
                                          bool               partial);

template void mjCModel::RestoreState<mjtNum>(const std::string& name,
                                             const mjtNum*      qpos0,
                                             const mjtNum*      mpos0,
                                             const mjtNum*      mquat0,
                                             mjtNum*            qpos,
                                             mjtNum*            qvel,
                                             mjtNum*            act,
                                             mjtNum*            ctrl,
                                             mjtNum*            mpos,
                                             mjtNum*            mquat,
                                             mjtNum*            plastic);


// check if a keyframe awaits the next compilation
bool mjCModel::HasPendingKeys() const {
  for (const mjCKey* key : keys_) {
    if (key->ispending_) { return true; }
  }
  return false;
}


// forget the state saved under a name
void mjCModel::ForgetState(const std::string& state_name) {
  for (mjCJoint* joint : joints_) {
    joint->qpos_.erase(state_name);
    joint->qvel_.erase(state_name);
    joint->plastic_.erase(state_name);
  }
  for (mjCTendon* tendon : tendons_) { tendon->plastic_.erase(state_name); }
  for (mjCFlex* flex : flexes_) { flex->plastic_.erase(state_name); }
  for (mjCActuator* actuator : actuators_) {
    actuator->act_.erase(state_name);
    actuator->ctrl_.erase(state_name);
  }
  for (mjCBody* body : bodies_) {
    body->mpos_.erase(state_name);
    body->mquat_.erase(state_name);
  }
}


// copy the state saved under a name to another name
void mjCModel::CopyState(const std::string& state_name, const std::string& copy_name) {
  for (mjCJoint* joint : joints_) {
    mjuu_copyvec(joint->qpos(copy_name), joint->qpos(state_name), 7);
    mjuu_copyvec(joint->qvel(copy_name), joint->qvel(state_name), 6);
    mjuu_copyvec(joint->plastic(copy_name), joint->plastic(state_name), 7);
  }
  for (mjCTendon* tendon : tendons_) {
    mjuu_copyvec(tendon->plastic(copy_name), tendon->plastic(state_name), 2);
  }
  for (mjCFlex* flex : flexes_) { flex->plastic(copy_name) = flex->plastic(state_name); }
  for (mjCActuator* actuator : actuators_) {
    actuator->act(copy_name)  = actuator->act(state_name);
    actuator->ctrl(copy_name) = actuator->ctrl(state_name);
  }
  for (mjCBody* body : bodies_) {
    mjuu_copyvec(body->mpos(copy_name), body->mpos(state_name), 3);
    mjuu_copyvec(body->mquat(copy_name), body->mquat(state_name), 4);
  }
}


// name under which the values of a pending keyframe are saved in the elements; it is not the
// name of the keyframe, which can be changed, and given to another keyframe, while it is pending
static std::string PendingKeyName() {
  static std::atomic<uint64_t> count{0};
  return "pending keyframe " + std::to_string(++count);
}


// complete a keyframe vector which is shorter than the model with the given value
static void completevec(std::vector<double>& vec, int size, double value) {
  if (!vec.empty() && vec.size() < size) { vec.resize(size, value); }
}


// store the values of the keyframes in the elements they belong to, ahead of a change to the tree;
// the keyframes are pending until the next compilation, which reassembles their vectors
void mjCModel::StoreKeyframes(mjCModel* dest) {
  // the addresses in the elements, the sizes and the default configuration of a compiled model:
  // recompiling carries the state of the model over through them, so they are put back when they
  // are computed here
  struct CompiledLayout {
    mjCModel*              model;
    std::vector<int>       adr;
    std::array<mjtSize, 8> size;
    std::vector<mjtNum>    qpos0, body_pos0, body_quat0;
    explicit CompiledLayout(mjCModel* m)
        : model(m),
          size{m->nq, m->nv, m->na, m->nu, m->nactuator, m->nout, m->nmocap, m->nplastic},
          qpos0(m->qpos0),
          body_pos0(m->body_pos0),
          body_quat0(m->body_quat0) {
      for (mjCJoint* joint : m->joints_) {
        adr.insert(adr.end(), {joint->qposadr_, joint->dofadr_, joint->plasticadr_});
      }
      for (mjCTendon* tendon : m->tendons_) { adr.push_back(tendon->plasticadr_); }
      for (mjCFlex* flex : m->flexes_) {
        adr.insert(adr.end(), {flex->plasticadr_, flex->plasticnum_});
      }
      for (mjCActuator* actuator : m->actuators_) {
        adr.insert(adr.end(),
                   {actuator->actdim_, actuator->actadr_, actuator->ctrladr_, actuator->outadr_});
      }
      for (mjCBody* body : m->bodies_) { adr.insert(adr.end(), {body->bodyadr_, body->mocapid}); }
      for (mjCEquality* equality : m->equalities_) { adr.push_back(equality->eqadr_); }
    }
    ~CompiledLayout() {
      const int* p = adr.data();
      for (mjCJoint* joint : model->joints_) {
        joint->qposadr_    = *p++;
        joint->dofadr_     = *p++;
        joint->plasticadr_ = *p++;
      }
      for (mjCTendon* tendon : model->tendons_) { tendon->plasticadr_ = *p++; }
      for (mjCFlex* flex : model->flexes_) {
        flex->plasticadr_ = *p++;
        flex->plasticnum_ = *p++;
      }
      for (mjCActuator* actuator : model->actuators_) {
        actuator->actdim_  = *p++;
        actuator->actadr_  = *p++;
        actuator->ctrladr_ = *p++;
        actuator->outadr_  = *p++;
      }
      for (mjCBody* body : model->bodies_) {
        body->bodyadr_ = *p++;
        body->mocapid  = *p++;
      }
      for (mjCEquality* equality : model->equalities_) { equality->eqadr_ = *p++; }
      model->nq         = size[0];
      model->nv         = size[1];
      model->na         = size[2];
      model->nu         = size[3];
      model->nactuator  = size[4];
      model->nout       = size[5];
      model->nmocap     = size[6];
      model->nplastic   = size[7];
      model->qpos0      = std::move(qpos0);
      model->body_pos0  = std::move(body_pos0);
      model->body_quat0 = std::move(body_quat0);
    }
  };

  // a model to which another one is attached has nothing to store if it has no keyframe; one
  // which is added to it later is given its vectors for the tree as it is then
  if (!dest && keys_.empty()) {
    keysstored = true;
    return;
  }

  // a keyframe of a compiled model is laid out for the last compilation until the tree changes.
  // After that, and in a model which is not compiled, a vector which a keyframe has was given to
  // it for the tree as it is now: it is laid out by the lists
  bool laidout = compiled && !keysstored;

  // an element attached to another model by reference has its addresses in that model, but this
  // model still lists it: it takes its place in the layout of these keyframes, except in a compiled
  // layout, which lost it when the element moved, and gets its addresses back on return
  struct Restore {
    std::vector<std::pair<int*, int>> saved;
    ~Restore() {
      for (auto [adr, value] : saved) { *adr = value; }
    }
  } moved;
  auto keep = [&moved, laidout](int& adr) {
    moved.saved.emplace_back(&adr, adr);
    if (laidout) { adr = -1; }
  };
  for (mjCJoint* joint : joints_) {
    if (joint->model != this) {
      keep(joint->qposadr_);
      keep(joint->dofadr_);
      keep(joint->plasticadr_);
    }
  }
  for (mjCTendon* tendon : tendons_) {
    if (tendon->model != this) { keep(tendon->plasticadr_); }
  }
  for (mjCFlex* flex : flexes_) {
    if (flex->model != this) {
      keep(flex->plasticadr_);
      keep(flex->plasticnum_);
    }
  }
  for (mjCActuator* actuator : actuators_) {
    if (actuator->model != this) {
      keep(actuator->actadr_);
      keep(actuator->actdim_);
      keep(actuator->ctrladr_);
      keep(actuator->outadr_);
    }
  }
  for (mjCBody* body : bodies_) {
    if (body->model != this) {
      keep(body->bodyadr_);
      keep(body->mocapid);
    }
  }
  for (mjCEquality* equality : equalities_) {
    if (equality->model != this) { keep(equality->eqadr_); }
  }

  // the addresses which are computed for the lists serve this function only, in a compiled model
  std::optional<CompiledLayout> compiledlayout;
  if (!laidout) {
    if (compiled) { compiledlayout.emplace(this); }
    SaveDofOffsets(/*computesize=*/true);
    ComputeReference();
  } else {
    // a joint whose type changed since the compilation has no place in its layout, as when the
    // state is saved for a recompilation
    for (mjCJoint* joint : joints_) {
      if (joint->type != joint->spec.type) {
        joint->qposadr_    = -1;
        joint->dofadr_     = -1;
        joint->plasticadr_ = -1;
      }
    }
  }

  // the list grows while it is traversed when a model is attached to itself
  std::vector<mjCKey*> keys   = keys_;
  bool                 warned = false;
  for (mjCKey* key : keys) {
    // the vectors in the size of the model: one which is shorter gives the leading elements, as
    // when compiling; the positions it does not give are left undefined, and take the default
    // configuration when the keyframe is reassembled
    std::vector<double> vqpos    = key->spec_qpos_;
    std::vector<double> vqvel    = key->spec_qvel_;
    std::vector<double> vact     = key->spec_act_;
    std::vector<double> vctrl    = key->spec_ctrl_;
    std::vector<double> vmpos    = key->spec_mpos_;
    std::vector<double> vmquat   = key->spec_mquat_;
    std::vector<double> vplastic = key->spec_plastic_;
    if (!laidout) {
      // a joint which is given in part takes the rest of its position as authored
      int nqpos = (int)vqpos.size();
      completevec(vqpos, nq, mjNAN);
      for (const mjCJoint* joint : joints_) {
        if (joint->qposadr_ < nqpos) {
          for (int i = nqpos; i < joint->qposadr_ + joint->nq(); i++) { vqpos[i] = qpos0[i]; }
        }
      }
      completevec(vqvel, nv, 0);
      completevec(vact, na, 0);
      completevec(vctrl, nu, 0);

      // a mocap body which is given in part takes its default pose
      if (vmpos.size() < 3 * nmocap) { vmpos.resize(3 * (vmpos.size() / 3)); }
      if (vmquat.size() < 4 * nmocap) { vmquat.resize(4 * (vmquat.size() / 4)); }
      completevec(vmpos, 3 * nmocap, mjNAN);
      completevec(vmquat, 4 * nmocap, mjNAN);

      // a plastic joint or tendon which is given in part takes its default plastic configuration
      int nplas = (int)vplastic.size();
      completevec(vplastic, nplastic, mjNAN);
      for (const mjCJoint* joint : joints_) {
        if (joint->plasticadr_ >= 0 &&
            joint->plasticadr_ < nplas &&
            nplas < joint->plasticadr_ + joint->nq()) {
          vplastic[joint->plasticadr_] = mjNAN;
        }
      }
      for (const mjCTendon* tendon : tendons_) {
        if (tendon->plasticadr_ >= 0 &&
            tendon->plasticadr_ < nplas &&
            nplas < tendon->plasticadr_ + 2) {
          vplastic[tendon->plasticadr_] = mjNAN;
        }
      }
      for (const mjCFlex* flex : flexes_) {
        if (flex->plasticadr_ >= 0 &&
            flex->plasticadr_ < nplas &&
            nplas < flex->plasticadr_ + flex->plasticnum_) {
          vplastic[flex->plasticadr_] = mjNAN;
        }
      }
    }

    // a model to which another one is attached leaves a keyframe which does not fit it as it is:
    // it may be written for the model that is being assembled, and is checked when compiling
    auto fits = [](const std::vector<double>& vec, int size) {
      return vec.empty() || vec.size() == size;
    };
    bool fit = fits(vqpos, nq) &&
               fits(vqvel, nv) &&
               fits(vact, na) &&
               fits(vctrl, nu) &&
               fits(vmpos, 3 * nmocap) &&
               fits(vmquat, 4 * nmocap) &&
               fits(vplastic, nplastic);
    if (!dest && !fit) { continue; }

    if (!vqpos.empty() && vqpos.size() != nq) {
      throw mjCError(nullptr,
                     "Keyframe '%s' has invalid qpos size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_qpos_.size(),
                     nq);
    }
    if (!vqvel.empty() && vqvel.size() != nv) {
      throw mjCError(nullptr,
                     "Keyframe %s has invalid qvel size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_qvel_.size(),
                     nv);
    }
    if (!vact.empty() && vact.size() != na) {
      throw mjCError(nullptr,
                     "Keyframe %s has invalid act size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_act_.size(),
                     na);
    }
    if (!vctrl.empty() && vctrl.size() != nu) {
      throw mjCError(nullptr,
                     "Keyframe %s has invalid ctrl size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_ctrl_.size(),
                     nu);
    }
    if (!vmpos.empty() && vmpos.size() != 3 * nmocap) {
      throw mjCError(nullptr,
                     "Keyframe %s has invalid mpos size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_mpos_.size(),
                     3 * nmocap);
    }
    if (!vmquat.empty() && vmquat.size() != 4 * nmocap) {
      throw mjCError(nullptr,
                     "Keyframe %s has invalid mquat size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_mquat_.size(),
                     4 * nmocap);
    }
    if (!vplastic.empty() && vplastic.size() != nplastic) {
      throw mjCError(nullptr,
                     "Keyframe %s has invalid plastic size, got %d, should be %d",
                     key->name.c_str(),
                     key->spec_plastic_.size(),
                     nplastic);
    }

    // the vectors which the keyframe has
    const double* qpos    = vqpos.empty() ? nullptr : vqpos.data();
    const double* qvel    = vqvel.empty() ? nullptr : vqvel.data();
    const double* act     = vact.empty() ? nullptr : vact.data();
    const double* ctrl    = vctrl.empty() ? nullptr : vctrl.data();
    const double* mpos    = vmpos.empty() ? nullptr : vmpos.data();
    const double* mquat   = vmquat.empty() ? nullptr : vmquat.data();
    const double* plastic = vplastic.empty() ? nullptr : vplastic.data();

    // store the vectors of the keyframe under a new name
    auto store = [&]() {
      mjKeyInfo stored;
      stored.name    = PendingKeyName();
      stored.time    = key->spec.time;
      stored.qpos    = qpos != nullptr;
      stored.qvel    = qvel != nullptr;
      stored.act     = act != nullptr;
      stored.ctrl    = ctrl != nullptr;
      stored.mpos    = mpos != nullptr;
      stored.mquat   = mquat != nullptr;
      stored.plastic = plastic != nullptr;
      SaveState(stored.name, qpos, qvel, act, ctrl, mpos, mquat, plastic);
      return stored;
    };

    // a keyframe which is still pending from an earlier change to the tree: the vectors it was
    // given since then replace what was stored, the others stay as they were stored
    mjKeyInfo& info = key->pending_;
    if (key->ispending_) {
      SaveState(info.name, qpos, qvel, act, ctrl, mpos, mquat, plastic, /*partial=*/true);
      info.qpos    |= qpos != nullptr;
      info.qvel    |= qvel != nullptr;
      info.act     |= act != nullptr;
      info.ctrl    |= ctrl != nullptr;
      info.mpos    |= mpos != nullptr;
      info.mquat   |= mquat != nullptr;
      info.plastic |= plastic != nullptr;
      info.time     = key->spec.time;
    }

    // the model is being attached: a copy of its keyframe is added to the destination, under the
    // namespace of the attachment
    bool stays = this == dest && prefix.empty() && suffix.empty();
    bool last  = key->ispending_ && !key->inplace_;
    if (dest && !stays && !last) {
      mjKeyInfo copy = key->ispending_ ? info : store();
      if (key->ispending_) {
        copy.name = PendingKeyName();
        CopyState(info.name, copy.name);
      }
      dest->AddPendingKey(prefix + key->name + suffix, copy);
    } else if (dest && this != dest) {
      // a pending keyframe which is kept last is not copied, another model takes it as it is
      if (!warned) {
        dest->AddWarning(
            "Child model has pending keyframes. They will not be namespaced "
            "correctly. "
            "To prevent this, compile the child model before attaching it again.");
        warned = true;
      }
      dest->AddPendingKey(key->name, info);
    }

    // the tree of this model is about to change: the keyframe stays in place, without its vectors
    if (!dest || this == dest) {
      if (!key->ispending_) {
        info            = store();
        key->ispending_ = true;
        key->inplace_   = true;
      }
      key->spec_qpos_.clear();
      key->spec_qvel_.clear();
      key->spec_act_.clear();
      key->spec_ctrl_.clear();
      key->spec_mpos_.clear();
      key->spec_mquat_.clear();
      key->spec_plastic_.clear();

      // a deletion moves it to the end of the list, with the keyframes which are kept last
      if (stays && key->inplace_) {
        key->inplace_ = false;
        auto it       = std::find(keys_.begin(), keys_.end(), key);
        std::rotate(it, it + 1, keys_.end());
      }
    }
  }
  for (int i = 0; i < (int)keys_.size(); i++) { keys_[i]->id = i; }
  ids[mjOBJ_KEY].clear();

  // the values of the keyframes which stay in place are not copied along with an attachment
  inplacekeys_.clear();
  for (const mjCKey* key : keys_) {
    if (key->inplace_) { inplacekeys_.push_back(key->pending_.name); }
  }

  if (!compiled) { nq = nv = na = nu = nactuator = nout = nmocap = 0; }

  // the tree of this model changes: until it is compiled again, its keyframes are given their
  // vectors for the tree as it is then
  if (!dest || this == dest) { keysstored = true; }
}


//------------------------------- FUSE STATIC ------------------------------------------------------

// change frame to parent body
static void changeframe(double       childpos[3],
                        double       childquat[4],
                        const double bodypos[3],
                        const double bodyquat[4]) {
  double pos[3], quat[4];
  mjuu_copyvec(pos, bodypos, 3);
  mjuu_copyvec(quat, bodyquat, 4);
  mjuu_frameaccum(pos, quat, childpos, childquat);
  mjuu_copyvec(childpos, pos, 3);
  mjuu_copyvec(childquat, quat, 4);
}


template <class T>
void mjCModel::ResolveReferences(std::vector<T*>& list, mjCBody* body) {
  for (auto& item : list) {
    item->CopyFromSpec();
    item->ResolveReferences(this);
  }
}


template <>
void mjCModel::ResolveReferences(std::vector<mjCSensor*>& list, mjCBody* body) {
  for (auto& item : list) {
    item->CopyFromSpec();
    item->ResolveReferences(this);
  }
  for (mjCSensor* sensor : list) {
    if (sensor->objtype == mjOBJ_SITE &&
        (sensor->type == mjSENS_FORCE || sensor->type == mjSENS_TORQUE) &&
        static_cast<mjCSite*>(sensor->obj)->body == body) {
      throw mjCError(sensor, "cannot fuse a body used by a force/torque sensor");
    }
  }
}


// true if fusing the body would break a reference to it
bool mjCModel::IsReferenced(mjCBody* body) {
  bool referenced = false;
  int  id         = body->id;

  // try to resolve references without the name of this body, taken away from the body and from
  // the map of names so that no lookup finds it; if it fails the body is referenced. A body used
  // by a force or torque sensor is kept too
  std::string name = body->name;
  body->name.clear();
  if (!name.empty()) { ids[mjOBJ_BODY].erase(name); }
  try {
    if (!name.empty()) {
      ResolveReferences(cameras_);
      ResolveReferences(lights_);
      for (mjCSkin* skin : skins_) { skin->ResolveReferences(this); }
      ResolveReferences(flexes_);
      ResolveReferences(pairs_);
      ResolveReferences(excludes_);
      ResolveReferences(equalities_);
      ResolveReferences(tendons_);
      ResolveReferences(actuators_);
      ResolveReferences(tuples_);
    }
    ResolveReferences(sensors_, body);
  } catch (mjCError err) { referenced = true; }
  body->name = name;
  if (!name.empty()) { ids[mjOBJ_BODY].insert({name, id}); }
  return referenced;
}


// fuse static bodies with their parents: a fused body becomes a frame in its parent, holding
// what the body held, in the same coordinates
int mjCModel::FuseStatic(const mjVFS* vfs) {
  int nfused = 0;

  // a body can be fused if it has no joints and is not a mocap body, unless it is to be kept;
  // one with a plugin is kept, its passive forces are specific to the body, and so is one with a
  // sleep policy, so that it is rejected as it is without fusing
  auto fusable = [](const mjCBody* body) {
    return body->joints.empty() &&
           !body->spec.mocap &&
           body->spec.fuse &&
           !body->spec.plugin.active &&
           body->spec.sleep == mjSLEEP_AUTO;
  };
  if (std::none_of(bodies_.begin() + 1, bodies_.end(), fusable)) { return 0; }

  // compile the kinematic tree, for the inertia of the bodies and the references to them
  if (!Resolve(vfs)) { throw mjCError(errInfo); }

  // a skin which is read from a file refers to the bodies that the file names: read it
  for (mjCSkin* skin : skins_) { skin->Compile(vfs); }

  // fluid forces are enabled
  bool fluid = option.density != 0 || option.viscosity != 0;

  // bodies which can be fused, in the order of the tree; whether a body is referenced is found
  // before anything is fused, since it uses the maps from names to ids
  std::vector<mjCBody*> candidates;
  for (int i = 1; i < bodies_.size(); i++) {
    mjCBody* body = bodies_[i];
    if (fusable(body) && !IsReferenced(body)) { candidates.push_back(body); }
  }

  for (mjCBody* body : candidates) {
    mjCBody* par = body->parent;

    // mass is fused with the parent's (if parent not world)
    bool fusemass = par->name != "world" && body->mass >= mjMINVAL;

    // skip if gravcomp is different, it applies to the mass of each body separately
    if (fusemass && body->gravcomp != par->gravcomp) { continue; }

    // skip if in a fluid, forces apply to the inertia and ellipsoid geoms of each body separately
    if (fluid && par->name != "world") {
      auto ellipsoid = [](const mjCGeom* geom) { return geom->fluid_ellipsoid > 0; };
      if (fusemass || std::any_of(body->geoms.begin(), body->geoms.end(), ellipsoid)) { continue; }
    }

    // if both infer their inertia from geoms, the parent has the geoms of both once the body
    // is fused and infers the sum from them; unless compilation adjusted what either inferred,
    // or the two count different geoms. The sum is then written to the spec of the parent, as it
    // is when either inertial is given
    const int* range    = body->compiler->inertiagrouprange;
    const int* parrange = par->compiler->inertiagrouprange;
    bool       reinfers = par->InfersInertial() &&
                          body->InfersInertial() &&
                          !par->inertia_adjusted_ &&
                          !body->inertia_adjusted_ &&
                          range[0] == parrange[0] &&
                          range[1] == parrange[1];

    // the mass is not fused but the geoms of the body move to the parent: a parent which infers
    // its inertia from geoms must not gain theirs, e.g. when the body gives a massless inertial,
    // so the inertia it has now is written to its spec
    bool keepinertia = !fusemass &&
                       !reinfers &&
                       par->name != "world" &&
                       par->InfersInertial() &&
                       !body->geoms.empty();

    // skip if the inertia cannot be written: the inertia of the parent is always inferred
    if ((fusemass || keepinertia) &&
        !reinfers &&
        par->compiler->inertiafromgeom == mjINERTIAFROMGEOM_TRUE) {
      continue;
    }

    // add mass and inertia to the compiled parent, which may itself be fused later
    if (fusemass) {
      par->AccumulateInertia(body);
      mjuu_copyvec(par->ipos_compiled_, par->ipos, 3);
      mjuu_copyvec(par->iquat_compiled_, par->iquat, 4);
      par->inertia_adjusted_ |= body->inertia_adjusted_;
      if (!reinfers) { par->AdoptInertial(); }
    } else if (keepinertia) {
      par->AdoptInertial();
    }

    // the children of the body become children of its parent: update their compiled frames
    for (mjCBody* child : body->bodies) {
      changeframe(child->pos, child->quat, body->pos, body->quat);
    }

    // replace the body with a frame; its child bodies take its place among those of the parent
    auto   place = std::find(par->bodies.begin(), par->bodies.end(), body) - par->bodies.begin();
    size_t nchildren = body->bodies.size();
    body->ToFrame(/*mergeinertial=*/false);
    std::rotate(par->bodies.begin() + place, par->bodies.end() - nchildren, par->bodies.end());

    // the body is not referenced by anything: release it, like a deleted element
    names_[mjOBJ_BODY].erase(body->name);
    body->SetParent(nullptr);
    body->frame = nullptr;
    body->Release();
    nfused++;
  }
  if (!nfused) { return 0; }

  // update the lists and the maps from names to ids
  ResetTreeLists();
  MakeTreeLists();
  ProcessLists(/*checkrepeat=*/false);
  InvalidateSignature();
  return nfused;
}


//------------------------------- DISCARD VISUAL ---------------------------------------------------

// discard what is only visual: materials and textures, the geoms which do not collide, and the
// meshes which are then not used; an element which another refers to by name is kept. Inertia
// which a body infers from discarded geoms becomes its explicit inertial. Return the number of
// elements discarded
int mjCModel::DiscardVisual(const mjVFS* vfs) {
  // set ids and the maps from names to ids, for the spec as it is now
  ProcessLists(/*checkrepeat=*/false);

  std::vector<bool> keepgeom(geoms_.size(), false);
  std::vector<bool> keepmesh(meshes_.size(), false);
  std::vector<bool> keepmaterial(materials_.size(), false);
  std::vector<bool> keeptexture(textures_.size(), false);
  auto              keep = [&](mjtObj type, const std::string& name) {
    std::vector<bool>* kept = type == mjOBJ_GEOM       ? &keepgeom
                              : type == mjOBJ_MESH     ? &keepmesh
                              : type == mjOBJ_MATERIAL ? &keepmaterial
                              : type == mjOBJ_TEXTURE  ? &keeptexture
                                                       : nullptr;
    mjCBase*           obj  = kept && !name.empty() ? FindObject(type, name) : nullptr;
    if (obj) { (*kept)[obj->id] = true; }
  };

  // a geom is kept if it collides or has fluid forces, or if another element refers to it
  for (const mjCGeom* geom : geoms_) {
    keepgeom[geom->id] =
        geom->spec.contype || geom->spec.conaffinity || geom->spec.fluid_ellipsoid > 0;
  }
  for (const mjCPair* pair : pairs_) {
    keep(mjOBJ_GEOM, pair->spec_geomname1_);
    keep(mjOBJ_GEOM, pair->spec_geomname2_);
  }
  for (const mjCTendon* tendon : tendons_) {
    for (const mjCWrap* wrap : tendon->path) {
      if (wrap->Type() == mjWRAP_SPHERE || wrap->Type() == mjWRAP_CYLINDER) {
        keep(mjOBJ_GEOM, wrap->name);
      }
    }
  }
  for (const mjCSensor* sensor : sensors_) {
    keep(sensor->spec.objtype, sensor->spec_objname_);
    keep(sensor->spec.reftype, sensor->spec_refname_);
  }
  for (const mjCTuple* tuple : tuples_) {
    size_t nobj = std::min(tuple->spec_objtype_.size(), tuple->spec_objname_.size());
    for (size_t i = 0; i < nobj; i++) { keep(tuple->spec_objtype_[i], tuple->spec_objname_[i]); }
  }

  // a mesh is kept if a geom which is kept or a site uses it, or if another element refers to
  // it; a material or texture only if a sensor or tuple refers to it, see above
  for (const mjCGeom* geom : geoms_) {
    if (keepgeom[geom->id]) { keep(mjOBJ_MESH, geom->get_meshname()); }
  }
  for (const mjCSite* site : sites_) { keep(mjOBJ_MESH, site->get_meshname()); }

  // bodies which infer their inertia from geoms and lose a geom which is counted for it
  auto counted = [&](const mjCBody* body, const mjCGeom* geom) {
    const int* range = body->compiler->inertiagrouprange;
    return !keepgeom[geom->id] && geom->spec.group >= range[0] && geom->spec.group <= range[1];
  };
  std::vector<mjCBody*> adopt;
  for (mjCBody* body : bodies_) {
    if (body->id == 0 || !body->InfersInertial()) { continue; }
    if (std::any_of(body->geoms.begin(), body->geoms.end(), [&](const mjCGeom* geom) {
          return counted(body, geom);
        })) {
      adopt.push_back(body);
    }
  }

  // their inertia is to become explicit: compile the kinematic tree, which calculates it, and
  // leave the bodies whose discarded geoms have no mass as they are
  if (!adopt.empty()) {
    if (!Resolve(vfs, /*textures=*/false)) { throw mjCError(errInfo); }
    adopt.erase(
        std::remove_if(
            adopt.begin(),
            adopt.end(),
            [&](const mjCBody* body) {
              return std::none_of(body->geoms.begin(), body->geoms.end(), [&](const mjCGeom* geom) {
                return counted(body, geom) && geom->mass_ > mjEPS;
              });
            }),
        adopt.end());
    for (const mjCBody* body : adopt) {
      if (body->compiler->inertiafromgeom == mjINERTIAFROMGEOM_TRUE) {
        throw mjCError(body,
                       "discarding visual geoms would change the inertia of this body, which "
                       "inertiafromgeom 'true' infers from its geoms; with inertiafromgeom 'auto' "
                       "it is kept as an explicit inertial");
      }
    }
  }

  // everything was checked, the spec is changed from here on
  std::vector<mjCGeom*>     discardgeoms;
  std::vector<mjCMesh*>     discardmeshes;
  std::vector<mjCMaterial*> discardmaterials;
  std::vector<mjCTexture*>  discardtextures;
  for (mjCGeom* geom : geoms_) {
    if (!keepgeom[geom->id]) { discardgeoms.push_back(geom); }
  }
  for (mjCMesh* mesh : meshes_) {
    if (!keepmesh[mesh->id]) { discardmeshes.push_back(mesh); }
  }
  for (mjCMaterial* material : materials_) {
    if (!keepmaterial[material->id]) { discardmaterials.push_back(material); }
  }
  for (mjCTexture* texture : textures_) {
    if (!keeptexture[texture->id]) { discardtextures.push_back(texture); }
  }
  int ndiscard =
      discardgeoms.size() + discardmeshes.size() + discardmaterials.size() + discardtextures.size();
  if (!ndiscard) { return 0; }

  // the plugin instances which elements refer to, before any of them is discarded
  const std::unordered_set<const mjsElement*> referenced = ReferencedPlugins();

  for (mjCBody* body : adopt) { body->AdoptInertial(); }

  // materials and textures, and what uses them: one which a sensor or tuple refers to stays as
  // an element, which nothing renders with
  for (mjCGeom* geom : geoms_) { geom->del_material(); }
  for (mjCSite* site : sites_) { site->del_material(); }
  for (mjCMesh* mesh : meshes_) { mesh->del_material(); }
  for (mjCSkin* skin : skins_) { skin->del_material(); }
  for (mjCFlex* flex : flexes_) { flex->del_material(); }
  for (mjCTendon* tendon : tendons_) { tendon->del_material(); }
  for (mjCLight* light : lights_) { light->del_texture(); }
  for (mjCDef* def : defaults_) {
    def->Geom().del_material();
    def->Site().del_material();
    def->Mesh().del_material();
    def->Flex().del_material();
    def->Tendon().del_material();
    def->Light().del_texture();
    def->Material().del_textures();
  }
  for (mjCMaterial* material : materials_) { material->del_textures(); }
  for (mjCMaterial* material : discardmaterials) {
    materials_.erase(std::remove(materials_.begin(), materials_.end(), material), materials_.end());
    material->Release();
  }
  for (mjCTexture* texture : discardtextures) {
    textures_.erase(std::remove(textures_.begin(), textures_.end(), texture), textures_.end());
    texture->Release();
  }
  for (mjCMaterial* material : materials_) { material->id = -1; }
  for (mjCTexture* texture : textures_) { texture->id = -1; }

  // geoms and meshes; the lists of the tree hold the geoms, empty them before any is released
  ResetTreeLists();
  for (mjCGeom* geom : discardgeoms) {
    std::vector<mjCGeom*>& geoms = geom->body->geoms;
    geoms.erase(std::remove(geoms.begin(), geoms.end(), geom), geoms.end());
    geom->Release();
  }
  for (mjCMesh* mesh : discardmeshes) {
    meshes_.erase(std::remove(meshes_.begin(), meshes_.end(), mesh), meshes_.end());
    DeleteInstances(meshinstances_, mesh);
    mesh->id = -1;
    mesh->Release();
  }
  for (mjCMesh* mesh : meshes_) { mesh->id = -1; }

  // update the lists and the maps from names to ids
  MakeTreeLists();
  ProcessLists(/*checkrepeat=*/false);

  // delete the plugin instances which only the discarded geoms and meshes referenced
  RemovePlugins(referenced);
  InvalidateSignature();
  return ndiscard;
}


//------------------------------- MERGE ASSETS -----------------------------------------------------

// map the assets of a list which repeat an earlier one, with the same key and the same content,
// to that first one
template <class T, class Mergeable, class Key, class Content>
static std::unordered_map<T*, T*> duplicates(const std::vector<T*>& list,
                                             Mergeable              mergeable,
                                             Key                    key,
                                             Content                content) {
  std::unordered_map<T*, T*>                       merged;
  std::unordered_map<std::string, std::vector<T*>> groups;
  for (T* asset : list) {
    if (!mergeable(asset)) { continue; }
    std::vector<T*>& group = groups[key(asset)];
    auto             same  = [&](const T* other) { return content(asset, other); };
    auto             first = std::find_if(group.begin(), group.end(), same);
    if (first == group.end()) {
      group.push_back(asset);
    } else {
      merged[asset] = *first;
    }
  }
  return merged;
}


// merge the textures and the meshes which would compile to the same asset: the first of each group
// keeps its name, every reference to the others is rewritten to it, and the others are deleted as
// mjs_delete would. Nothing is read: an asset is identified by what compilation reads from the
// spec, its resolved file or its buffers and all its parameters, names excluded. Return the number
// of assets merged
int mjCModel::DedupAssets() {
  // set ids and the maps from names to ids; the names of the assets must be unique, since the
  // references to them are rewritten by name
  ProcessLists(/*checkrepeat=*/false);
  CheckRepeat(mjOBJ_TEXTURE);
  CheckRepeat(mjOBJ_MESH);

  // what identifies an asset: its key, and the buffers which are compared without being copied
  auto texturekey     = [](const mjCTexture* tex) { return tex->AssetKey(); };
  auto texturecontent = [](const mjCTexture* a, const mjCTexture* b) {
    return a->spec_data_ == b->spec_data_;
  };
  auto meshkey     = [](const mjCMesh* mesh) { return mesh->AssetKey(); };
  auto meshcontent = [](const mjCMesh* a, const mjCMesh* b) {
    return a->spec_vert_ == b->spec_vert_ &&
           a->spec_normal_ == b->spec_normal_ &&
           a->spec_texcoord_ == b->spec_texcoord_ &&
           a->spec_face_ == b->spec_face_ &&
           a->spec_facenormal_ == b->spec_facenormal_ &&
           a->spec_facetexcoord_ == b->spec_facetexcoord_ &&
           a->spec_node_ == b->spec_node_ &&
           a->spec_tet_ == b->spec_tet_;
  };
  // an unnamed asset cannot be referenced, it is left as it is; a mesh with a plugin is the
  // instance of the plugin, it is not merged
  auto named    = [](const mjCTexture* tex) { return !tex->name.empty(); };
  auto noplugin = [](const mjCMesh* mesh) {
    return !mesh->name.empty() && !mesh->spec.plugin.active;
  };
  auto textures = duplicates(textures_, named, texturekey, texturecontent);
  auto meshes   = duplicates(meshes_, noplugin, meshkey, meshcontent);
  int  nmerged  = textures.size() + meshes.size();
  if (!nmerged) { return 0; }

  // the names which replace those of the merged assets
  std::unordered_map<std::string, std::string> texnames, meshnames;
  for (const auto& [tex, first] : textures) { texnames[tex->name] = first->name; }
  for (const auto& [mesh, first] : meshes) { meshnames[mesh->name] = first->name; }
  auto rename = [](const std::unordered_map<std::string, std::string>& names, std::string& name) {
    auto it = names.find(name);
    if (it != names.end()) { name = it->second; }
  };
  auto texture = [&](std::string& name) { rename(texnames, name); };
  auto mesh    = [&](std::string& name) { rename(meshnames, name); };
  auto object  = [&](mjtObj type, std::string& name) {
    if (type == mjOBJ_TEXTURE) { texture(name); }
    if (type == mjOBJ_MESH) { mesh(name); }
  };

  // the merged assets and the instances of the meshes, which the compiled copies of the elements
  // may point to until the next compilation resolves them again from the names
  std::unordered_set<const mjCBase*> gone;
  for (const auto& [tex, first] : textures) { gone.insert(tex); }
  for (const auto& [m, first] : meshes) { gone.insert(m); }
  for (const mjCMesh* instance : meshinstances_) {
    if (gone.count(instance->primary_)) { gone.insert(instance); }
  }
  auto drop = [&](auto*& pointer) {
    if (gone.count(pointer)) { pointer = nullptr; }
  };

  // everything was checked, the spec is changed from here on: rewrite the references, in the
  // spec and in the compiled copies of the elements
  auto material = [&](mjCMaterial* mat) {
    for (std::string& name : mat->spec_textures_) { texture(name); }
    for (std::string& name : mat->textures_) { texture(name); }
  };
  auto light = [&](mjCLight* l) {
    texture(l->spec_texture_);
    texture(l->texture_);
  };
  auto geom = [&](mjCGeom* g) {
    mesh(g->spec_meshname_);
    mesh(g->meshname_);
    drop(g->mesh);
  };
  auto site = [&](mjCSite* s) {
    mesh(s->spec_meshname_);
    mesh(s->meshname_);
    drop(s->mesh);
  };
  for (mjCMaterial* mat : materials_) { material(mat); }
  for (mjCLight* l : lights_) { light(l); }
  for (mjCGeom* g : geoms_) { geom(g); }
  for (mjCSite* s : sites_) { site(s); }
  for (mjCDef* def : defaults_) {
    material(&def->Material());
    light(&def->Light());
    geom(&def->Geom());
    site(&def->Site());
  }
  for (mjCSensor* sensor : sensors_) {
    object(sensor->spec.objtype, sensor->spec_objname_);
    object(sensor->spec.objtype, sensor->objname_);
    object(sensor->spec.reftype, sensor->spec_refname_);
    object(sensor->spec.reftype, sensor->refname_);
    drop(sensor->obj);
    drop(sensor->ref);
  }
  for (mjCTuple* tuple : tuples_) {
    size_t nobj = std::min(tuple->spec_objtype_.size(), tuple->spec_objname_.size());
    for (size_t i = 0; i < nobj; i++) { object(tuple->spec_objtype_[i], tuple->spec_objname_[i]); }
    nobj = std::min(tuple->objtype_.size(), tuple->objname_.size());
    for (size_t i = 0; i < nobj; i++) { object(tuple->objtype_[i], tuple->objname_[i]); }
    for (mjCBase*& obj : tuple->obj) { drop(obj); }
  }

  // delete the merged assets, with the instances which compilation made of the meshes
  auto istexture = [&](mjCTexture* tex) { return textures.count(tex) > 0; };
  auto ismesh    = [&](mjCMesh* m) { return meshes.count(m) > 0; };
  textures_.erase(std::remove_if(textures_.begin(), textures_.end(), istexture), textures_.end());
  meshes_.erase(std::remove_if(meshes_.begin(), meshes_.end(), ismesh), meshes_.end());
  for (const auto& [tex, first] : textures) {
    tex->id = -1;
    tex->Release();
  }
  for (const auto& [m, first] : meshes) {
    DeleteInstances(meshinstances_, m);
    m->id = -1;
    m->Release();
  }

  // update the ids and the maps from names to ids
  for (mjCTexture* tex : textures_) { tex->id = -1; }
  for (mjCMesh* m : meshes_) { m->id = -1; }
  ProcessLists(/*checkrepeat=*/false);
  InvalidateSignature();
  return nmerged;
}


//------------------------------- COMPILER ---------------------------------------------------------

// signature comparisons
static int comparePair(mjCPair* el1, mjCPair* el2) {
  return el1->GetSignature() < el2->GetSignature();
}
static int compareBodyPair(mjCBodyPair* el1, mjCBodyPair* el2) {
  return el1->GetSignature() < el2->GetSignature();
}


// reassign ids
template <class T>
static void reassignid(vector<T*>& list) {
  for (int i = 0; i < (int)list.size(); i++) { list[i]->id = i; }
}


// assign ids in sorted order, leaving the order of the list as it is
template <class T, class Compare>
static void sortid(const vector<T*>& list, Compare compare) {
  vector<T*> sorted = list;
  std::stable_sort(sorted.begin(), sorted.end(), compare);
  reassignid(sorted);
}


// give a copy of the model what the compilation of the original gave to it; the elements outside
// the tree and the plugins were given theirs as they were copied
void mjCModel::CopyCompiled(const mjCModel& other) {
  // the tree is copied whole
  CopyCompiled(bodies_[0], other.bodies_[0]);

  // geoms and sites refer to the copies of the assets which the compilation resolved for them;
  // meshes and height fields are copied in order, none is left out
  std::unordered_map<const mjCBase*, mjCBase*> assets;
  for (int i = 0; i < meshes_.size(); i++) { assets[other.meshes_[i]] = meshes_[i]; }
  for (int i = 0; i < hfields_.size(); i++) { assets[other.hfields_[i]] = hfields_[i]; }
  auto copied = [&assets](const mjCBase* asset) -> mjCBase* {
    auto it = assets.find(asset);
    return it == assets.end() ? nullptr : it->second;
  };
  auto instantiate = [&](auto& instances, const auto& source) {
    for (auto* instance : instances) { delete instance; }
    instances.clear();
    for (const auto* instance : source) {
      using T          = std::remove_const_t<std::remove_pointer_t<decltype(instance)>>;
      const T* primary = static_cast<const T*>(copied(instance->primary_));
      if (!primary) { continue; }
      T* copy        = new T(*instance);
      copy->model    = this;
      copy->primary_ = primary;
      copy->compiler = copy->primary_->compiler;
      if constexpr (std::is_same_v<T, mjCMesh>) {
        copy->spec.plugin.element = copy->primary_->spec.plugin.element;
      }
      assets[instance] = copy;
      instances.push_back(copy);
    }
  };
  instantiate(meshinstances_, other.meshinstances_);
  instantiate(hfieldinstances_, other.hfieldinstances_);
  for (mjCGeom* geom : geoms_) {
    geom->mesh   = static_cast<mjCMesh*>(copied(geom->mesh));
    geom->hfield = static_cast<mjCHField*>(copied(geom->hfield));
  }
  for (mjCSite* site : sites_) { site->mesh = static_cast<mjCMesh*>(copied(site->mesh)); }

  // the working copies of the elements point to the strings and the plugins of this model, as they
  // do after compiling it
  auto pointplugin = [](auto* element) {
    element->plugin.element     = element->spec.plugin.element;
    element->plugin.plugin_name = element->spec.plugin.plugin_name;
    element->plugin.name        = element->spec.plugin.name;
  };
  for (mjCBody* body : bodies_) { pointplugin(body); }
  for (mjCGeom* geom : geoms_) { pointplugin(geom); }
  for (mjCMesh* mesh : CompiledMeshes()) { pointplugin(mesh); }
  for (mjCActuator* actuator : actuators_) { pointplugin(actuator); }
  for (mjCSensor* sensor : sensors_) { pointplugin(sensor); }
  for (mjCEquality* equality : equalities_) {
    equality->name1 = equality->spec.name1;
    equality->name2 = equality->spec.name2;
  }
  for (mjCTendon* tendon : tendons_) {
    for (mjCWrap* wrap : tendon->path) { wrap->model = this; }
  }

  // a copy of a compiled model is compiled: the working copy of its spec points to its own strings,
  // and its pairs and excludes are numbered in the order of the compiled model
  if (compiled) {
    modelname    = spec.modelname;
    comment      = spec.comment;
    modelfiledir = spec.modelfiledir;
    sortid(pairs_, comparePair);
    sortid(excludes_, compareBodyPair);
  }
}


void mjCModel::CopyCompiled(mjCBody* dest, const mjCBody* source) {
  dest->bodyadr_ = source->bodyadr_;
  dest->mocapid  = source->mocapid;
  for (int i = 0; i < dest->joints.size(); i++) {
    dest->joints[i]->qposadr_    = source->joints[i]->qposadr_;
    dest->joints[i]->dofadr_     = source->joints[i]->dofadr_;
    dest->joints[i]->plasticadr_ = source->joints[i]->plasticadr_;
  }
  for (int i = 0; i < dest->bodies.size(); i++) {
    CopyCompiled(dest->bodies[i], source->bodies[i]);
  }
}


void mjCModel::CopyCompiled(mjCTendon* dest, const mjCTendon* source) {
  dest->plasticadr_ = source->plasticadr_;
}


void mjCModel::CopyCompiled(mjCFlex* dest, const mjCFlex* source) {
  dest->plasticadr_ = source->plasticadr_;
  dest->plasticnum_ = source->plasticnum_;
}


void mjCModel::CopyCompiled(mjCEquality* dest, const mjCEquality* source) {
  dest->eqadr_ = source->eqadr_;
}


void mjCModel::CopyCompiled(mjCActuator* dest, const mjCActuator* source) {
  dest->actadr_     = source->actadr_;
  dest->actdim_     = source->actdim_;
  dest->ctrladr_    = source->ctrladr_;
  dest->outadr_     = source->outadr_;
  dest->historyadr_ = source->historyadr_;
  dest->historynum_ = source->historynum_;
}


void mjCModel::CopyCompiled(mjCSensor* dest, const mjCSensor* source) {
  dest->historyadr_ = source->historyadr_;
  dest->historynum_ = source->historynum_;
}


void mjCModel::CopyCompiled(mjCPlugin* dest, const mjCPlugin* source) {
  dest->stateadr_ = source->stateadr_;
  dest->statenum_ = source->statenum_;
}


// set object ids, check for repeated names
void mjCModel::ProcessLists(bool checkrepeat) {
  for (int i = 0; i < mjNOBJECT; i++) {
    if (i != mjOBJ_XBODY && object_lists_[i]) {
      ids[i].clear();
      ProcessList_(ids, *object_lists_[i], (mjtObj)i, checkrepeat);
    }
  }

  // check repeated names in meta elements
  ProcessList_(ids, frames_, mjOBJ_FRAME, checkrepeat);
}


// set ids, check for repeated names
template <class T>
void mjCModel::ProcessList_(mjListKeyMap& ids, vector<T*>& list, mjtObj type, bool checkrepeat) {
  int slot = (type == mjOBJ_FRAME) ? mjNOBJECT : type;
  names_[slot].clear();
  for (size_t i = 0; i < list.size(); i++) {
    if (!list[i]->name.empty()) { names_[slot].insert(list[i]->name); }
  }

  // assign ids for regular elements
  if (type < mjNOBJECT) {
    for (size_t i = 0; i < list.size(); i++) {
      // check for incompatible id setting; SHOULD NOT OCCUR
      // pairs and excludes are exempt: once compiled, their ids follow the order of the model
      bool sorted = type == mjOBJ_PAIR || type == mjOBJ_EXCLUDE;
      if (!sorted && list[i]->id != -1 && list[i]->id != i) {
        throw mjCError(list[i], "incompatible id in %s array, position %d", mju_type2Str(type), i);
      }

      // id equals position in array
      list[i]->id = i;

      // add to ids map
      ids[type][list[i]->name] = i;
    }
  }

  // check for repeated names
  if (checkrepeat) { CheckRepeat(type); }
}


// check for repeated names in list
void mjCModel::CheckRepeat(mjtObj type) {
  std::vector<mjCBase*>* list = nullptr;
  if (type < mjNOBJECT) {
    list = object_lists_[type];
  } else if (type == mjOBJ_FRAME) {
    list = (std::vector<mjCBase*>*)&frames_;
  }

  // created vectors with all names
  vector<string> allnames;
  for (size_t i = 0; i < list->size(); i++) {
    if (!(*list)[i]->name.empty()) { allnames.push_back((*list)[i]->name); }
  }

  // sort and check for duplicates
  if (allnames.size() > 1) {
    std::sort(allnames.begin(), allnames.end());
    auto adjacent = std::adjacent_find(allnames.begin(), allnames.end());
    if (adjacent != allnames.end()) {
      string msg = "repeated name '" + *adjacent + "' in " + mju_type2Str(type);
      throw mjCError(nullptr, "%s", msg.c_str());
    }
  }
}


// check that newname is not used by another element of the same type
void mjCModel::CheckNameChange(mjtObj             type,
                               const std::string& oldname,
                               const std::string& newname) {
  int slot;
  if (type == mjOBJ_FRAME) {
    slot = mjNOBJECT;
  } else if (type < mjNOBJECT && type != mjOBJ_XBODY && object_lists_[type]) {
    slot = type;
  } else {
    return;
  }
  if (!newname.empty() && newname != oldname && names_[slot].count(newname)) {
    string msg = "repeated name '" + newname + "' in " + mju_type2Str(type);
    throw mjCError(nullptr, "%s", msg.c_str());
  }
  if (!oldname.empty()) { names_[slot].erase(oldname); }
  if (!newname.empty()) { names_[slot].insert(newname); }
}

// error handler for low-level engine
constexpr int                    kErrorBufferSize = 500;
static thread_local std::jmp_buf error_jmp_buf;
static thread_local char         errortext[kErrorBufferSize] = "";


// warning handler for low-level engine
static thread_local char         warningtext[kErrorBufferSize] = "";  // top-level warning buffer
static thread_local std::string* local_warningtext_ptr = nullptr;     // sub-thread warning buffer

static void compilerLogHandler(const mjLogMessage* msg) {
  if (msg->level == mjLOG_ERROR) {
    mju::strcpy_arr(errortext, msg->subject);
    std::longjmp(error_jmp_buf, 1);
  } else if (msg->level == mjLOG_WARNING) {
    // buffer for structured capture (append, not overwrite)
    if (local_warningtext_ptr) {
      if (!local_warningtext_ptr->empty()) { *local_warningtext_ptr += '\n'; }
      *local_warningtext_ptr += msg->subject;
    } else {
      if (warningtext[0]) { mju::strcat_arr(warningtext, "\n"); }
      mju::strcat_arr(warningtext, msg->subject);
    }
  }
}


// compiler
mjModel* mjCModel::Compile(const mjVFS* vfs, mjModel** m) {
  // the options which restructure the model are operations on the spec, applied before it is
  // compiled; if one fails, compilation fails. The assets are compiled once: by the first
  // operation which compiles the kinematic tree, or else by the compilation
  int ndiscarded = 0, nfused = 0;
  reuse_assets_    = true;
  assets_compiled_ = false;
  try {
    if (spec.compiler.discardvisual) { ndiscarded = DiscardVisual(vfs); }
    if (spec.compiler.fusestatic) { nfused = FuseStatic(vfs); }
  } catch (mjCError err) {
    if (m && *m) { mj_deleteModel(*m); }
    Clear();
    errInfo       = err;
    reuse_assets_ = false;
    if (ndiscarded) {
      mju::strcat_arr(errInfo.message,
                      "\nThe visual elements of the spec were discarded before this error, and "
                      "remain so.");
    }
    return nullptr;
  }
  mjModel* model = Compile(vfs, m, /*treeonly=*/false, /*textures=*/true);
  reuse_assets_  = false;

  // an operation which was applied stays applied if compilation then fails
  if (!model && ndiscarded) {
    mju::strcat_arr(errInfo.message,
                    "\nThe visual elements of the spec were discarded before this error, and "
                    "remain so.");
  }
  if (!model && nfused) {
    mju::strcat_arr(errInfo.message,
                    "\nThe static bodies of the spec were fused before this error, and remain so.");
  }
  return model;
}


bool mjCModel::Resolve(const mjVFS* vfs, bool textures) {
  Compile(vfs, nullptr, /*treeonly=*/true, textures);
  return errInfo.message[0] == 0;
}


// compile the model, or only its assets and kinematic tree
mjModel* mjCModel::Compile(const mjVFS* vfs, mjModel** m, bool treeonly, bool textures) {
  if (compiled) { Clear(); }
  baseline_ = false;

  CopyFromSpec();

  // The volatile keyword is necessary to prevent a possible memory leak due to
  // an interaction between longjmp and compiler optimization. Specifically, at
  // the point where the setjmp takes places, these pointers have never been
  // reassigned from their nullptr initialization. Without the volatile keyword,
  // the compiler is free to assume that these pointers remain nullptr when the
  // setjmp returns, and therefore to pass nullptr directly to the
  // mj_deleteModel and mj_deleteData calls in the subsequent catch block,
  // without ever reading the actual pointer values.
  mjModel* volatile model = (m && *m) ? *m : nullptr;
  mjData* volatile data   = nullptr;

  // install compiler log handler (captures warnings silently)
  mjfLogHandler prev_tls = _mjPRIVATE_setTlsLogHandler(compilerLogHandler);

  errInfo = mjCError();

  // set flag so warnings are captured in the spec vector rather than delivered
  // immediately
  compiling_ = true;
  ClearCompileWarnings();
  warningtext[0] = 0;

  try {
    if (attached_) {
      throw mjCError(0, "cannot compile child spec if attached by reference to a parent spec");
    }
    if (setjmp(error_jmp_buf) != 0) {
      // TryCompile resulted in an mju_error which was converted to a longjmp.
      std::string error_msg = errortext;
      // also include the last warning that was issued. this is useful for
      // warnings that came out of plugin implementations.
      if (warningtext[0]) {
        error_msg += '\n';
        error_msg += warningtext;
      }
      throw mjCError(0, "engine error: %s", error_msg.c_str());
    }

    if (treeonly) {
      // an operation on the spec leaves the keyframes as they are
      CompileTree(vfs, textures, /*keyframes=*/false);
    } else {
      TryCompile(*const_cast<mjModel**>(&model), *const_cast<mjData**>(&data), vfs);
    }
  } catch (mjCError err) {
    // deallocate everything allocated in Compile
    mj_deleteModel(model);
    model = nullptr;
    mj_deleteData(data);
    data = nullptr;
    Clear();

    // save error info
    errInfo = err;
    if (warningtext[0]) {
      mju::strcat_arr(errInfo.message, "\n");
      mju::strcat_arr(errInfo.message, warningtext);
    }

    // restore handler, return 0
    _mjPRIVATE_setTlsLogHandler(prev_tls);
    compiling_ = false;
    return nullptr;
  }

  // restore log handler
  _mjPRIVATE_setTlsLogHandler(prev_tls);
  compiling_ = false;
  if (treeonly) { return nullptr; }
  compiled  = true;
  baseline_ = true;

  // play back compile warnings through the normal handler chain
  for (int i = num_attach_warnings_; i < warnings_.size(); ++i) {
    mju_warning("%s", warnings_[i].c_str());
  }

  return model;
}


// compile a mesh or texture with the compiler log handler, on any thread: an mju_error is
// converted to a longjmp back here and stored as an exception, as Compile only calls setjmp on
// the calling thread
template <typename T>
static void CompileAsset(T*                  asset,
                         const mjVFS*        vfs,
                         std::exception_ptr& exception,
                         std::mutex&         exception_mutex,
                         std::string*        warningtext) {
  local_warningtext_ptr = warningtext;
  auto previous_handler = _mjPRIVATE_setTlsLogHandler(compilerLogHandler);

  // save the enclosing longjmp target, if any
  std::jmp_buf saved;
  std::memcpy(saved, error_jmp_buf, sizeof(std::jmp_buf));

  if (setjmp(error_jmp_buf) == 0) {
    try {
      asset->Compile(vfs);
    } catch (...) {
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!exception) { exception = std::current_exception(); }
    }
  } else {
    std::lock_guard<std::mutex> lock(exception_mutex);
    if (!exception) {
      exception = std::make_exception_ptr(mjCError(asset, "engine error: %s", errortext));
    }
  }

  std::memcpy(error_jmp_buf, saved, sizeof(std::jmp_buf));
  _mjPRIVATE_setTlsLogHandler(previous_handler);
  local_warningtext_ptr = nullptr;
}

// Helper function for mesh compilation used by both serial and parallel paths
static void CompileMesh(mjCMesh*            mesh,
                        const mjVFS*        vfs,
                        std::exception_ptr& exception,
                        std::mutex&         exception_mutex,
                        std::string*        warningtext) {
  CompileAsset(mesh, vfs, exception, exception_mutex, warningtext);
}

// Helper function for texture compilation used by both serial and parallel paths
static void CompileTexture(mjCTexture*         texture,
                           const mjVFS*        vfs,
                           std::exception_ptr& exception,
                           std::mutex&         exception_mutex,
                           std::string*        warningtext) {
  using Clock          = std::chrono::steady_clock;
  using Seconds        = std::chrono::duration<double>;
  Clock::time_point t0 = Clock::now();
  CompileAsset(texture, vfs, exception, exception_mutex, warningtext);
  texture->texture_time_ = Seconds(Clock::now() - t0).count();
}

// multi-threaded mesh and texture compilation with shared threadpool
void mjCModel::CompileMeshesAndTextures(const mjVFS* vfs, bool textures) {
  const std::vector<mjCMesh*> meshes      = CompiledMeshes();
  int                         nmesh       = meshes.size();
  int                         ntexture    = textures ? textures_.size() : 0;
  int                         total_tasks = nmesh + ntexture;

  // holds exceptions thrown by worker threads
  std::exception_ptr mesh_exception;
  std::mutex         mesh_except_mutex;
  std::exception_ptr texture_exception;
  std::mutex         texture_except_mutex;

  std::vector<std::string> mesh_warningtext(nmesh);
  std::vector<std::string> texture_warningtext(ntexture);

  // If no pool provided or too few total tasks, run serially
  if (!compiler.usethread || total_tasks < 2) {
    // Compile meshes serially
    for (int i = 0; i < nmesh; i++) {
      CompileMesh(meshes[i], vfs, mesh_exception, mesh_except_mutex, &mesh_warningtext[i]);
    }
    // Compile textures serially
    for (int i = 0; i < ntexture; i++) {
      CompileTexture(textures_[i],
                     vfs,
                     texture_exception,
                     texture_except_mutex,
                     &texture_warningtext[i]);
    }
  } else {
    mujoco::user::ThreadPool pool(NumCompilerThreads(total_tasks));

    // Enqueue mesh tasks
    for (int i = 0; i < nmesh; ++i) {
      pool.Schedule([mesh = meshes[i],
                     vfs,
                     &mesh_exception,
                     &mesh_except_mutex,
                     warningtext = &mesh_warningtext[i]]() {
        CompileMesh(mesh, vfs, mesh_exception, mesh_except_mutex, warningtext);
      });
    }

    // Enqueue texture tasks
    for (int i = 0; i < ntexture; ++i) {
      pool.Schedule([texture = textures_[i],
                     vfs,
                     &texture_exception,
                     &texture_except_mutex,
                     warningtext = &texture_warningtext[i]]() {
        CompileTexture(texture, vfs, texture_exception, texture_except_mutex, warningtext);
      });
    }

    // Wait for all tasks to complete
    pool.WaitCount(total_tasks);
  }

  // concatenate all mesh warnings, copy into warningtext
  std::string concatenated_warnings;
  bool        has_warning = false;
  for (int i = 0; i < nmesh; i++) {
    if (!mesh_warningtext[i].empty()) {
      if (has_warning) { concatenated_warnings += '\n'; }
      concatenated_warnings += mesh_warningtext[i];
      has_warning            = true;
    }
  }
  mju::strcpy_arr(warningtext, concatenated_warnings.c_str());

  // aggregate texture warnings
  for (int i = 0; i < ntexture; ++i) {
    if (!texture_warningtext[i].empty()) {
      if (has_warning) mju::strcat_arr(warningtext, "\n");
      mju::strcat_arr(warningtext, texture_warningtext[i].c_str());
      has_warning = true;
    }
  }

  // if exceptions were caught, rethrow the first one
  if (mesh_exception) { std::rethrow_exception(mesh_exception); }
  if (texture_exception) { std::rethrow_exception(texture_exception); }

  for (int i = 0; i < nmesh; i++) {
    for (int t = 0; t < mjNCTIMER; t++) { timer[t] += meshes[i]->mesh_timer_[t]; }
  }
  for (int i = 0; i < ntexture; i++) { timer[mjCTIMER_TEXTURE] += textures_[i]->texture_time_; }
}

// compute qpos0
void mjCModel::ComputeReference() {
  int b = 0;
  qpos0.resize(nq);
  body_pos0.resize(3 * bodies_.size());
  body_quat0.resize(4 * bodies_.size());
  for (auto body : bodies_) {
    mjuu_copyvec(body_pos0.data() + 3 * b, body->spec.pos, 3);
    mjuu_copyvec(body_quat0.data() + 4 * b, body->spec.quat, 4);
    for (auto joint : body->joints) {
      switch (joint->spec.type) {
        case mjJNT_FREE:
          mjuu_copyvec(qpos0.data() + joint->qposadr_, body->spec.pos, 3);
          mjuu_copyvec(qpos0.data() + joint->qposadr_ + 3, body->spec.quat, 4);
          break;

        case mjJNT_BALL:
          mjuu_setvec(qpos0.data() + joint->qposadr_, 1, 0, 0, 0);
          break;

        case mjJNT_SLIDE:
        case mjJNT_HINGE:
          qpos0[joint->qposadr_] = (mjtNum)joint->spec.ref;
          break;

        default:
          throw mjCError(joint, "unknown joint type");
      }
    }
    b++;
  }
}


// resizes a keyframe, filling in missing values
void mjCModel::ExpandKeyframe(mjCKey*       key,
                              const mjtNum* qpos0_,
                              const mjtNum* bpos,
                              const mjtNum* bquat) {
  if (!key->spec_qpos_.empty() && nq > key->spec_qpos_.size()) {
    int nq0 = key->spec_qpos_.size();
    key->spec_qpos_.resize(nq);
    for (int i = nq0; i < nq; i++) { key->spec_qpos_[i] = (double)qpos0_[i]; }
  }
  if (!key->spec_qvel_.empty() && nv > key->spec_qvel_.size()) { key->spec_qvel_.resize(nv); }
  if (!key->spec_act_.empty() && na > key->spec_act_.size()) { key->spec_act_.resize(na); }
  if (!key->spec_ctrl_.empty() && nu > key->spec_ctrl_.size()) { key->spec_ctrl_.resize(nu); }
  if (!key->spec_mpos_.empty() && nmocap > key->spec_mpos_.size() / 3) {
    int nmocap0 = key->spec_mpos_.size() / 3;
    key->spec_mpos_.resize(3 * nmocap);
    for (unsigned int j = 0; j < bodies_.size(); j++) {
      if (bodies_[j]->mocapid < nmocap0) { continue; }
      int i = bodies_[j]->mocapid;

      key->spec_mpos_[3 * i + 0] = (double)bpos[3 * j + 0];
      key->spec_mpos_[3 * i + 1] = (double)bpos[3 * j + 1];
      key->spec_mpos_[3 * i + 2] = (double)bpos[3 * j + 2];
    }
  }
  if (!key->spec_mquat_.empty() && nmocap > key->spec_mquat_.size() / 4) {
    int nmocap0 = key->spec_mquat_.size() / 4;
    key->spec_mquat_.resize(4 * nmocap);
    for (unsigned int j = 0; j < bodies_.size(); j++) {
      if (bodies_[j]->mocapid < nmocap0) { continue; }
      int i = bodies_[j]->mocapid;

      key->spec_mquat_[4 * i + 0] = (double)bquat[4 * j + 0];
      key->spec_mquat_[4 * i + 1] = (double)bquat[4 * j + 1];
      key->spec_mquat_[4 * i + 2] = (double)bquat[4 * j + 2];
      key->spec_mquat_[4 * i + 3] = (double)bquat[4 * j + 3];
    }
  }
  if (!key->spec_plastic_.empty() && nplastic > key->spec_plastic_.size()) {
    key->spec_plastic_.resize(nplastic, mjNAN);
  }
}

// vector of a pending keyframe to reassemble, or null if there is none to reassemble; a vector
// that was set after the change to the tree is left as it is
static double* pendingvec(bool stored, std::vector<double>& vec, int size) {
  if (!stored || !vec.empty()) { return nullptr; }
  vec.assign(size, 0);
  return vec.data();
}


// scale a point or a direction in the world, at a top-level body, from the units of the spec to
// the compiled model, or back
template <class T>
static void ScaleInWorld(const mjCBody* body, T* x, bool point, bool inverse) {
  if (!body->frame) { return; }
  double y[3] = {(double)x[0], (double)x[1], (double)x[2]};
  body->frame->ScaleInBody(y, point, inverse);
  for (int i = 0; i < 3; i++) { x[i] = (T)y[i]; }
}


// scale the vectors of a keyframe from the units of the spec to the compiled model, or back; free
// joints and mocap bodies are posed in the world, and null vectors are left as they are
template <class T>
void mjCModel::ScaleKeyframe(
    T* qpos, T* qvel, T* act, T* ctrl, T* mpos, T* plastic, bool inverse) const {
  auto scale = [inverse](T& x, double f) {
    if (mjuu_defined(x)) { x = (T)(inverse ? x / f : x * f); }
  };
  for (const mjCJoint* joint : joints_) {
    if (joint->qposadr_ < 0) { continue; }
    if (joint->spec.type == mjJNT_SLIDE) {
      const double f = joint->Gauge().Stretch(joint->spec.axis);
      if (qpos) { scale(qpos[joint->qposadr_], f); }
      if (qvel) { scale(qvel[joint->dofadr_], f); }
      if (plastic && joint->plasticadr_ >= 0) { scale(plastic[joint->plasticadr_], f); }
    } else if (joint->spec.type == mjJNT_FREE) {
      if (qpos) { ScaleInWorld(joint->body, qpos + joint->qposadr_, true, inverse); }
      if (qvel) { ScaleInWorld(joint->body, qvel + joint->dofadr_, false, inverse); }
    }
  }
  for (const mjCTendon* tendon : tendons_) {
    if (plastic && tendon->plasticadr_ >= 0 && tendon->Linear()) {
      const double f = tendon->PathGauge().Shape(1);
      scale(plastic[tendon->plasticadr_ + 0], f);
      scale(plastic[tendon->plasticadr_ + 1], f);
    }
  }
  for (const mjCFlex* flex : flexes_) {
    if (plastic && flex->plasticadr_ >= 0 && flex->plasticnum_ > 0) {
      if (flex->interpolated || !flex->spec_nodebody_.empty()) { continue; }
      const mjCGauge g           = flex->BodyGauge();
      const double   f           = g.Shape(1);
      const double   fb          = g.Force(-1, 1);
      int            stretch_num = 0;
      if (!flex->stiffness.empty()) {
        if (flex->stiffness[0] != 0) { stretch_num = (flex->dim == 2 ? 3 : 6) * flex->nelem; }
      } else if ((flex->spec.dim == 2 && flex->spec.elastic2d >= 2 && flex->spec.thickness > 0) ||
                 flex->spec.dim == 3) {
        int num_elem = (int)flex->spec_elem_.size() / (flex->spec.dim + 1);
        stretch_num  = (flex->spec.dim == 2 ? 3 : 6) * num_elem;
      }
      for (int j = 0; j < stretch_num; j++) { scale(plastic[flex->plasticadr_ + j], f * f); }
      for (int j = stretch_num; j < flex->plasticnum_; j++) {
        scale(plastic[flex->plasticadr_ + j], fb);
      }
    }
  }
  for (const mjCActuator* actuator : actuators_) {
    for (int j = 0; act && actuator->actadr_ >= 0 && j < actuator->actdim_; j++) {
      scale(act[actuator->actadr_ + j], actuator->actscale_);
    }
    for (int j = 0; ctrl && actuator->ctrladr_ >= 0 && j < actuator->ctrlnum_ && j < 4; j++) {
      scale(ctrl[actuator->ctrladr_ + j], actuator->ctrlscales_[j]);
    }
  }
  for (const mjCBody* body : bodies_) {
    if (mpos && body->mocapid >= 0) { ScaleInWorld(body, mpos + 3 * body->mocapid, true, inverse); }
  }
}


// reassemble the vectors of the pending keyframes, fill in missing default values
void mjCModel::ResolveKeyframes(const mjModel* m) {
  // store dof offsets in joints and actuators
  SaveDofOffsets();
  keysstored = false;

  // the default configuration, in the units of the spec
  std::vector<mjtNum> specqpos0(m->qpos0, m->qpos0 + m->nq);
  std::vector<mjtNum> specbodypos(m->body_pos, m->body_pos + 3 * m->nbody);
  ScaleKeyframe<mjtNum>(specqpos0.data(),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        /*inverse=*/true);
  for (int i = 0; i < m->nbody; i++) {
    if (bodies_[i]->mocapid >= 0) {
      ScaleInWorld(bodies_[i], specbodypos.data() + 3 * i, true, true);
    }
  }

  std::vector<std::string> resolved;
  for (mjCKey* key : keys_) {
    if (!key->ispending_) { continue; }
    const mjKeyInfo& info = key->pending_;
    RestoreState(info.name,
                 specqpos0.data(),
                 specbodypos.data(),
                 m->body_quat,
                 pendingvec(info.qpos, key->spec_qpos_, nq),
                 pendingvec(info.qvel, key->spec_qvel_, nv),
                 pendingvec(info.act, key->spec_act_, na),
                 pendingvec(info.ctrl, key->spec_ctrl_, nu),
                 pendingvec(info.mpos, key->spec_mpos_, 3 * nmocap),
                 pendingvec(info.mquat, key->spec_mquat_, 4 * nmocap),
                 pendingvec(info.plastic, key->spec_plastic_, nplastic));
    key->ispending_ = false;
    key->inplace_   = false;
    resolved.push_back(info.name);
  }

  // the stored values have served
  for (const std::string& name : resolved) { ForgetState(name); }

  // complete the vectors which are shorter than the model with the default configuration
  for (mjCKey* key : keys_) {
    ExpandKeyframe(key, specqpos0.data(), specbodypos.data(), m->body_quat);
  }
}

#if defined(__EMSCRIPTEN__) && !defined(MUJOCO_WASM_THREADS)
// The MuJoCo compiler defaults to usethread=1, which causes it to try to
// create pthreads for compilation. In the single-threaded WASM build, this
// crashes because there is no threading support, so we disable threading on
// the internal compiler struct (not the spec) to avoid permanently mutating
// the spec (which would cause usethread="false" to appear in a saved XML).
struct ScopedDisableThreading {
  mjtBool& ref;
  mjtBool  saved;
  explicit ScopedDisableThreading(mjtBool& r) : ref(r), saved(r) { ref = 0; }
  ~ScopedDisableThreading() { ref = saved; }
};
#endif

// return true if flex f can generate flex-flex contacts (with itself or another flex)
static bool FlexCollidesWithFlex(const mjModel* m, int f) {
  if (m->flex_contype[f] == 0 && m->flex_conaffinity[f] == 0) { return false; }
  if ((m->flex_contype[f] & m->flex_conaffinity[f]) && m->flex_selfcollide[f] != mjFLEXSELF_NONE) {
    return true;
  }
  for (int f2 = 0; f2 < m->nflex; f2++) {
    if (f2 != f && ((m->flex_contype[f] & m->flex_conaffinity[f2]) ||
                    (m->flex_contype[f2] & m->flex_conaffinity[f]))) {
      return true;
    }
  }
  return false;
}

// count DOFs and symbolic Hessian/Cholesky nonzeros of non-rigid flexes
static void EstimateFlexArena(const mjModel* m,
                              int            njmax,
                              std::size_t*   nflexdof_out,
                              std::size_t*   nflexdof_collide_out,
                              std::size_t*   nH_out,
                              std::size_t*   nL_out) {
  *nflexdof_out         = 0;
  *nflexdof_collide_out = 0;
  *nH_out               = 0;
  *nL_out               = 0;
  if (!m->nflex || !m->nv) { return; }

  // body_slot[b]: -1 if not a flex body, 0 if isolated flex body, 1 if colliding flex body
  std::vector<int> body_slot(m->nbody, -1);
  std::vector<int> body_collide(m->nbody, 0);
  bool             has_flex     = false;
  bool             has_isolated = false;
  for (int f = 0; f < m->nflex; f++) {
    if (m->flex_rigid[f]) { continue; }
    has_flex     = true;
    bool collide = FlexCollidesWithFlex(m, f);
    if (!collide) { has_isolated = true; }
    int        num    = m->flex_interp[f] ? m->flex_nodenum[f] : m->flex_vertnum[f];
    int        adr    = m->flex_interp[f] ? m->flex_nodeadr[f] : m->flex_vertadr[f];
    const int* bodyid = m->flex_interp[f] ? m->flex_nodebodyid : m->flex_vertbodyid;
    for (int i = 0; i < num; i++) {
      int b = m->body_weldid[bodyid[adr + i]];
      if (b > 0 && m->body_dofnum[b] > 0) {
        body_slot[b] = 0;
        if (collide) { body_collide[b] = 1; }
      }
    }
  }
  if (!has_flex) { return; }

  int              nb               = 0;
  std::size_t      nflexdof         = 0;
  std::size_t      nflexdof_collide = 0;
  std::vector<int> nd;
  nd.reserve(m->nbody);
  for (int b = 1; b < m->nbody; b++) {
    if (body_slot[b] == 0) {
      body_slot[b] = nb++;
      nd.push_back(m->body_dofnum[b]);
      nflexdof += m->body_dofnum[b];
      if (body_collide[b]) { nflexdof_collide += m->body_dofnum[b]; }
    }
  }
  if (!nb || (!has_isolated && nflexdof <= static_cast<std::size_t>(njmax))) { return; }

  // build upper-triangular weld-body adjacency across all static stencils of non-rigid flexes
  std::vector<std::vector<int>> upper(nb);
  auto                          add_clique = [&](const int* slots, int k) {
    for (int a = 0; a < k; a++) {
      int sa = slots[a];
      if (sa < 0) continue;
      for (int b = a + 1; b < k; b++) {
        int sb = slots[b];
        if (sb < 0 || sa == sb) continue;
        if (sa < sb) {
          upper[sa].push_back(sb);
        } else {
          upper[sb].push_back(sa);
        }
      }
    }
  };

  for (int f = 0; f < m->nflex; f++) {
    if (m->flex_rigid[f]) { continue; }
    if (m->flex_interp[f]) {
      int  order      = m->flex_interp[f];
      bool shell_mode = order < 0;
      order           = order < 0 ? -order : order;
      int cx          = m->flex_cellnum[3 * f + 0];
      int cy          = m->flex_cellnum[3 * f + 1];
      int cz          = m->flex_cellnum[3 * f + 2];
      int npe = shell_mode ? (order + 1) * (order + 1) : (order + 1) * (order + 1) * (order + 1);
      int nelem_fe      = shell_mode ? 2 * (cy * cz + cx * cz + cx * cy) : cx * cy * cz;
      const int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];
      int        gindices[125];
      int        slots[125];
      for (int fe = 0; fe < nelem_fe; fe++) {
        if (shell_mode) {
          mju_flexGatherFaceState(order,
                                  cx,
                                  cy,
                                  cz,
                                  fe,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  gindices,
                                  nullptr);
        } else {
          int ci = fe / (cy * cz);
          int cj = (fe / cz) % cy;
          int ck = fe % cz;
          mju_flexGatherCellState(order,
                                  cy,
                                  cz,
                                  ci,
                                  cj,
                                  ck,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  gindices,
                                  nullptr);
        }
        for (int i = 0; i < npe; i++) { slots[i] = body_slot[m->body_weldid[bodyid[gindices[i]]]]; }
        add_clique(slots, npe);
      }
      continue;
    }

    int              vadr = m->flex_vertadr[f];
    int              vnum = m->flex_vertnum[f];
    std::vector<int> vslot(vnum, -1);
    for (int v = 0; v < vnum; v++) {
      vslot[v] = body_slot[m->body_weldid[m->flex_vertbodyid[vadr + v]]];
    }

    // elements (triangles, tetrahedra, capsules)
    int dim  = m->flex_dim[f];
    int eadr = m->flex_elemdataadr[f];
    for (int e = 0; e < m->flex_elemnum[f]; e++) {
      const int* elem = m->flex_elem + eadr + e * (dim + 1);
      int        slots[4];
      for (int k = 0; k <= dim; k++) { slots[k] = vslot[elem[k]]; }
      add_clique(slots, dim + 1);
    }

    // edges and 2D bending flaps
    int  edgeadr    = m->flex_edgeadr[f];
    int  edgenum    = m->flex_edgenum[f];
    bool has_bend2d = (dim == 2 && m->flex_bendingadr[f] >= 0);
    for (int e = 0; e < edgenum; e++) {
      int v0       = m->flex_edge[2 * (edgeadr + e) + 0];
      int v1       = m->flex_edge[2 * (edgeadr + e) + 1];
      int slots[4] = {vslot[v0], vslot[v1], -1, -1};
      if (has_bend2d) {
        int f0 = m->flex_edgeflap[2 * (edgeadr + e) + 0];
        int f1 = m->flex_edgeflap[2 * (edgeadr + e) + 1];
        if (f0 >= 0 && f1 >= 0) {
          slots[2] = vslot[f0];
          slots[3] = vslot[f1];
          add_clique(slots, 4);
          continue;
        }
      }
      add_clique(slots, 2);
    }

    // 1D rod bending (3-vertex single hinges and 4-vertex hinge pairs)
    if (dim == 1 && m->flex_bendingadr[f] >= 0) {
      const mjtNum* bend = m->flex_bending + m->flex_bendingadr[f];
      for (int h = 0; h < static_cast<int>(bend[0]); h++) {
        const mjtNum* hinge = bend + 1 + mjROD_HINGE_SIZE * h;
        int           nr    = mj_rodHingeVert(hinge);
        int           slots[4];
        for (int k = 0; k < nr; k++) { slots[k] = vslot[static_cast<int>(hinge[k])]; }
        add_clique(slots, nr);
      }
    }
  }

  for (int q = 0; q < nb; q++) {
    auto& row = upper[q];
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }

  // block-level symbolic reverse-Cholesky count on the static flex graph
  std::vector<int>         parent(nb, -1);
  std::vector<int>         flag(nb, -1);
  std::vector<std::size_t> offdiag_dofs(nb, 0);
  std::size_t              nH = 0;
  for (int q = nb - 1; q >= 0; q--) {
    flag[q]         = q;
    std::size_t dq  = static_cast<std::size_t>(nd[q]);
    nH             += dq * (dq + 1) / 2;
    for (int p0 : upper[q]) {
      nH += dq * static_cast<std::size_t>(nd[p0]);
      for (int p = p0; p >= 0 && flag[p] != q; p = parent[p]) {
        if (parent[p] == -1) { parent[p] = q; }
        flag[p]          = q;
        offdiag_dofs[p] += dq;
      }
    }
  }

  std::size_t nL = 0;
  for (int p = 0; p < nb; p++) {
    std::size_t dp  = static_cast<std::size_t>(nd[p]);
    nL             += dp * offdiag_dofs[p] + dp * (dp + 1) / 2;
  }

  *nflexdof_out         = nflexdof;
  *nflexdof_collide_out = nflexdof_collide;
  *nH_out               = nH;
  *nL_out               = nL;
}

// first stage of compilation: the assets and the kinematic tree; keyframes are added and
// completed only if asked, which a full compilation does and an operation on the spec does not
void mjCModel::CompileTree(const mjVFS* vfs, bool textures, bool keyframes) {
#if defined(__EMSCRIPTEN__) && !defined(MUJOCO_WASM_THREADS)
  ScopedDisableThreading disable_usethread(compiler.usethread);
#endif

  using Clock   = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;

  // check if nan test works
  double test = mjNAN;
  if (mjuu_defined(test)) {
    throw mjCError(0, "NaN test does not work for present compiler/options");
  }

  // check for joints in world body
  if (!bodies_[0]->joints.empty()) { throw mjCError(0, "joint found in world body"); }

  // check for too many body+flex
  if (bodies_.size() + flexes_.size() >= 65534) {
    throw mjCError(0, "number of bodies plus flexes must be less than 65534");
  }

  // add missing keyframes
  if (keyframes) {
    for (int i = keys_.size(); i < nkey; i++) { AddKey(); }
  }

  // clear subtreedofs
  for (int i = 0; i < bodies_.size(); i++) { bodies_[i]->subtreedofs = 0; }

  // meshes and textures are compiled here, unless an operation which this compilation applied
  // has compiled them already
  bool compileassets = !(
      reuse_assets_ && assets_compiled_ && (!textures || textures_compiled_ || textures_.empty()));

  // refresh the working copies of the assets, check that those which need a name have one
  if (compileassets) {
    for (const auto& asset : meshes_) asset->CopyFromSpec();
    for (const auto& asset : textures_) asset->CopyFromSpec();
  }
  for (const auto& asset : skins_) asset->CopyFromSpec();
  CheckEmptyNames();

  // set object ids, check for repeated names
  ProcessLists();

  // map names to asset references, and give each use the instance of its asset at its scale
  for (mjCMesh* mesh : meshes_) { mesh->SetNeedSDF(false); }
  IndexAssets();
  MakeInstances(/*reuse=*/!compileassets);
  for (mjCHField* hfield : CompiledHFields()) { hfield->CopyFromSpec(); }

  // compile pairs for convex hull check
  // TODO(quaglino): Consolidate the two calls to pair->Compile() in TryCompile.
  for (auto pair : pairs_) pair->Compile();

  // mark meshes that need convex hull
  for (int i = 0; i < geoms_.size(); i++) {
    bool is_in_pair = false;
    for (const mjCPair* pair : pairs_) {
      if ((pair->geom1 && pair->geom1->id == geoms_[i]->id) ||
          (pair->geom2 && pair->geom2->id == geoms_[i]->id)) {
        is_in_pair = true;
        break;
      }
    }

    if (geoms_[i]->mesh &&
        (geoms_[i]->spec.type == mjGEOM_MESH || geoms_[i]->spec.type == mjGEOM_SDF) &&
        (geoms_[i]->spec.contype || geoms_[i]->spec.conaffinity || is_in_pair)) {
      geoms_[i]->mesh->SetNeedHull(true);
    }
  }

  // convex inertia is computed from the hull, so it is needed whether or not
  // any geom references the mesh
  for (mjCMesh* mesh : CompiledMeshes()) {
    if (mesh->spec.inertia == mjMESH_INERTIA_CONVEX) { mesh->SetNeedHull(true); }
  }

  for (int i = 0; i < sites_.size(); i++) {
    if (sites_[i]->mesh && sites_[i]->spec.type == mjGEOM_MESH) {
      sites_[i]->mesh->SetNeedHull(true);
    }
  }

  // automatically set nuser fields
  SetNuser();

  // compile meshes and textures (needed for geom compilation)
  if (compileassets) {
    double before[mjNCTIMER];
    std::copy(timer, timer + mjNCTIMER, before);
    Clock::time_point t0 = Clock::now();
    CompileMeshesAndTextures(vfs, textures);
    timer[mjCTIMER_ASSETS]  = Seconds(Clock::now() - t0).count();
    before[mjCTIMER_ASSETS] = 0;

    // keep what the rest of this compilation will not produce again
    assets_compiled_   = true;
    textures_compiled_ = textures;
    asset_warnings_    = warningtext;
    for (int i = 0; i < mjNCTIMER; i++) { asset_timer_[i] = timer[i] - before[i]; }
  } else {
    mju::strcpy_arr(warningtext, asset_warnings_.c_str());
    for (int i = 0; i < mjNCTIMER; i++) { timer[i] += asset_timer_[i]; }
  }

  // frames cache their accumulated pose, recompute it in every compile
  for (mjCFrame* frame : frames_) { frame->compiled = false; }

  // compile objects in kinematic tree
  for (int i = 0; i < bodies_.size(); i++) {
    bodies_[i]->Compile();  // also compiles joints, geoms, sites, cameras, lights, frames
  }
}


void mjCModel::TryCompile(mjModel*& m, mjData*& d, const mjVFS* vfs) {
#if defined(__EMSCRIPTEN__) && !defined(MUJOCO_WASM_THREADS)
  ScopedDisableThreading disable_usethread(compiler.usethread);
#endif

  // clear compile-phase warnings from previous compile, keep attach warnings
  ClearCompileWarnings();

  using Clock   = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;
  for (int i = 0; i < mjNCTIMER; i++) { timer[i] = 0; }
  Clock::time_point timer_start = Clock::now();

  // compile the assets and the kinematic tree
  CompileTree(vfs, /*textures=*/true, /*keyframes=*/true);

  // compile all other objects except for keyframes
  for (auto flex : flexes_) flex->Compile(vfs);
  for (auto skin : skins_) skin->Compile(vfs);
  for (auto hfield : CompiledHFields()) hfield->Compile(vfs);

  for (auto material : materials_) material->Compile();
  for (auto pair : pairs_) pair->Compile();
  for (auto exclude : excludes_) exclude->Compile();
  for (auto tendon : tendons_) tendon->Compile();
  for (auto equality : equalities_) equality->Compile();
  for (auto actuator : actuators_) actuator->Compile();
  for (auto sensor : sensors_) sensor->Compile();
  for (auto numeric : numerics_) numeric->Compile();
  for (auto text : texts_) text->Compile();
  for (auto tuple : tuples_) tuple->Compile();
  for (auto plugin : plugins_) plugin->Compile();

  // compile def: to enforce userdata length for writer
  for (mjCDef* def : defaults_) { def->Compile(this); }

  // the model holds pairs and excludes in increasing signature order: number them in that order,
  // while the lists keep the order in which they were authored
  sortid(pairs_, comparePair);
  sortid(excludes_, compareBodyPair);

  // resolve asset references, compute sizes
  IndexAssets();
  MakeInstances(/*reuse=*/true);
  SetSizes();
  SaveDofOffsets(/*computesize=*/false);  // Populate jnt->dofadr_

  // compute sparse matrix sizes
  ComputeSparseSizes();

  // set nmocap and body.mocapid
  for (mjCBody* body : bodies_) {
    if (body->mocap) {
      body->mocapid = nmocap;
      nmocap++;
    } else {
      body->mocapid = -1;
    }
  }

  // check mass and inertia of moving bodies
  for (int i = 1; i < bodies_.size(); i++) {
    if (!bodies_[i]->joints.empty() && !CheckBodyMassInertia(bodies_[i])) {
      throw mjCError(bodies_[i], "mass and inertia of moving bodies must be larger than mjMINVAL");
    }
  }

  // create low-level model
  mj_makeModel(&m,
               nq,
               nv,
               nu,
               nactuator,
               nout,
               na,
               nbody,
               nbvh,
               nbvhstatic,
               nbvhdynamic,
               noct,
               njnt,
               ntree,
               nM,
               nB,
               nC,
               nD,
               ngeom,
               nsite,
               ncam,
               nlight,
               nflex,
               nflexnode,
               nflexvert,
               nflexedge,
               nflexelem,
               nflexelemdata,
               nflexstiffness,
               nflexbending,
               nefm0dof,
               nefm0L,
               nefmCvert,
               nflexelemedge,
               nflexshelldata,
               nflextexcoord,
               nJfe,
               nmesh,
               nmeshvert,
               nmeshnormal,
               nmeshtexcoord,
               nmeshface,
               nmeshgraph,
               nmeshpoly,
               nmeshpolyvert,
               nmeshpolymap,
               nskin,
               nskinvert,
               nskintexvert,
               nskinface,
               nskinbone,
               nskinbonevert,
               nhfield,
               nhfielddata,
               ntex,
               ntexdata,
               nmat,
               npair,
               nexclude,
               neq,
               ntendon,
               nJten,
               nwrap,
               nsensor,
               nnumeric,
               nnumericdata,
               ntext,
               ntextdata,
               ntuple,
               ntupledata,
               nkey,
               nmocap,
               nplugin,
               npluginattr,
               nuser_body,
               nuser_jnt,
               nuser_geom,
               nuser_site,
               nuser_cam,
               nuser_tendon,
               nuser_actuator,
               nuser_sensor,
               nnames,
               npaths,
               nplastic);
  if (!m) { throw mjCError(0, "could not create mjModel"); }

  // copy everything into low-level model
  m->opt = option;
  m->vis = visual;
  CopyNames(m);
  CopyPaths(m);
  CopyTree(m);
  CopyPlugins(m);

  // keyframe compilation needs access to nq, nv, na, nmocap, qpos0
  ResolveKeyframes(m);

  for (int i = 0; i < keys_.size(); i++) { keys_[i]->Compile(m); }

  // the vectors which a keyframe gives are in the units of the spec
  for (mjCKey* key : keys_) {
    auto given = [](std::vector<double>& vec, const std::vector<double>& spec) {
      return spec.empty() ? nullptr : vec.data();
    };
    ScaleKeyframe(given(key->qpos_, key->spec_qpos_),
                  given(key->qvel_, key->spec_qvel_),
                  given(key->act_, key->spec_act_),
                  given(key->ctrl_, key->spec_ctrl_),
                  given(key->mpos_, key->spec_mpos_),
                  given(key->plastic_, key->spec_plastic_),
                  /*inverse=*/false);
  }

  // copy objects outsite kinematic tree (including keyframes)
  CopyObjects(m);

  // finalize simple bodies/dofs including tendon information
  FinalizeSimple(m);

  // compute non-zeros in actuator_moment
  m->nJmom = nJmom = CountNJmom(m);

  // mean inertia (deprecated)
  if (mjuu_defined(stat.meaninertia)) {
    AddWarning(
        "statistic attribute 'meaninertia' is deprecated and will be removed in a future release: "
        "it does not affect simulation");
  }

  // scale mass (deprecated)
  if (compiler.settotalmass > 0) {
    AddWarning(
        "compiler attribute 'settotalmass' is deprecated and will be removed in a future "
        "release: scale the masses and densities in the model, or call mj_setTotalmass "
        "and mj_setConst on the compiled model");
    mj_setTotalmass(m, compiler.settotalmass);
  }

  // set arena size into m->narena
  if (memory != -1) {
    // memory size is user-specified in bytes
    m->narena = memory;
  } else {
    const int   nconmax  = m->nconmax == -1 ? 100 : m->nconmax;
    const int   njmax    = m->njmax == -1 ? 500 : m->njmax;
    std::size_t nflexdof = 0, nflexdof_collide = 0, nH_flex = 0, nL_flex = 0;
    if (nstack == -1) {
      // use a conservative heuristic if neither memory nor nstack is specified in XML; in sparse
      // mode, size non-rigid flexes symbolically via their static Hessian and Cholesky factor
      // instead of including their DOFs in the dense quadratic term (MakeHessian falls back to CG
      // if runtime contact fill-in exhausts the arena)
      if (mj_isSparse(m)) {
        EstimateFlexArena(m, njmax, &nflexdof, &nflexdof_collide, &nH_flex, &nL_flex);
      }
      const std::size_t ndense  = njmax + m->neq + m->nv - nflexdof;
      const std::size_t nlinear = m->nq +
                                  m->nv +
                                  m->nu +
                                  m->na +
                                  m->nbody +
                                  m->njnt +
                                  m->ngeom +
                                  m->nsite +
                                  m->neq +
                                  m->ntendon +
                                  m->nwrap;
      m->narena                 = sizeof(mjtNum) * mjMAX(1000, 5 * ndense * ndense + 20 * nlinear);
    } else {
      // (legacy) stack size is user-specified as multiple of sizeof(mjtNum)
      m->narena = sizeof(mjtNum) * nstack;
    }

    // add an arena space equal to memory footprint prior to the introduction of the arena
    const std::size_t arena_bytes  = (nconmax * sizeof(mjContact) +
                                      njmax * (8 * sizeof(int) + 14 * sizeof(mjtNum)) +
                                      m->nv * (3 * sizeof(int)) +
                                      njmax * m->nv * (2 * sizeof(int) + 2 * sizeof(mjtNum)) +
                                      njmax * njmax * (sizeof(int) + sizeof(mjtNum)));
    m->narena                     += arena_bytes;

    // if symbolic Hessian + Cholesky factor of flexes exceeds the linear njmax*nflexdof allocation
    // already included in arena_bytes, add the excess (e.g. 3D grids with higher fill), plus a
    // bounded allowance for two-sided contact Jacobians/fill on colliding flexes
    if (nflexdof) {
      const std::size_t njmax_flex_bytes =
          static_cast<std::size_t>(njmax) * nflexdof * (2 * sizeof(int) + 2 * sizeof(mjtNum));
      const std::size_t nL_num            = (m->opt.cone == mjCONE_ELLIPTIC) ? 2 : 1;
      const std::size_t flex_solver_bytes = nH_flex * (6 * sizeof(int) + 5 * sizeof(mjtNum)) +
                                            nL_flex * (3 * sizeof(int) + nL_num * sizeof(mjtNum));
      // 1.5x safety factor on static symbolic Hessian + Cholesky factor bytes
      if (flex_solver_bytes > njmax_flex_bytes) {
        m->narena += flex_solver_bytes - njmax_flex_bytes;
      }
      m->narena += flex_solver_bytes / 2;
      if (nflexdof_collide) {
        // cap contact fill-in allowance at 64 MB so large colliding flexes do not allocate
        // hundreds of megabytes by default (heavier fill-in falls back to CG)
        constexpr std::size_t kMaxCollideBytes  = 64 << 20;
        const std::size_t     collide_bytes     = static_cast<std::size_t>(njmax) *
                                                  nflexdof_collide *
                                                  (2 * sizeof(int) + 2 * sizeof(mjtNum));
        m->narena                              += std::min(collide_bytes, kMaxCollideBytes);
      }
    }

    // round up to the nearest megabyte
    constexpr std::size_t kMegabyte   = 1 << 20;
    std::size_t           nstack_mb   = m->narena / kMegabyte;
    std::size_t           residual_mb = m->narena % kMegabyte ? 1 : 0;
    m->narena                         = kMegabyte * (nstack_mb + residual_mb);
  }

  // sparsity structures
  {
    std::vector<int> scratch(m->nv);
    std::vector<int> count(m->nbody);
    std::vector<int> M(m->nM);

    // make D
    mj_makeDofDofSparse(m->nv,
                        m->nC,
                        m->nD,
                        m->nM,
                        m->dof_parentid,
                        m->dof_simplenum,
                        m->D_rownnz,
                        m->D_rowadr,
                        m->D_diag,
                        m->D_colind,
                        /*reduced=*/0,
                        /*upper=*/1,
                        scratch.data());

    // make B
    mj_makeBSparse(m->nv,
                   m->nbody,
                   m->nB,
                   m->body_dofnum,
                   m->body_parentid,
                   m->body_dofadr,
                   m->B_rownnz,
                   m->B_rowadr,
                   m->B_colind,
                   count.data());

    // make M
    mj_makeDofDofSparse(m->nv,
                        m->nC,
                        m->nD,
                        m->nM,
                        m->dof_parentid,
                        m->dof_simplenum,
                        m->M_rownnz,
                        m->M_rowadr,
                        NULL,
                        m->M_colind,
                        /*reduced=*/1,
                        /*upper=*/0,
                        scratch.data());

    // make index mappings: mapM2D, mapD2M, mapM2M
    mj_makeDofDofMaps(m->nv,
                      m->nM,
                      m->nC,
                      m->nD,
                      m->dof_Madr,
                      m->dof_simplenum,
                      m->dof_parentid,
                      m->D_rownnz,
                      m->D_rowadr,
                      m->D_colind,
                      m->M_rownnz,
                      m->M_rowadr,
                      m->M_colind,
                      m->mapM2D,
                      m->mapD2M,
                      m->mapM2M,
                      M.data(),
                      scratch.data());
  }

  // history buffers require a positive timestep
  if (m->nhistory && m->opt.timestep <= 0) {
    char msg[100];
    snprintf(msg,
             sizeof(msg),
             "history buffers require positive timestep, got %g",
             (double)m->opt.timestep);
    throw mjCError(0, "%s", msg);
  }

  // create data
  int disableflags     = m->opt.disableflags;
  m->opt.disableflags |= mjDSBL_CONTACT;
  int enableflags      = m->opt.enableflags;
  m->opt.enableflags  &= ~mjENBL_SLEEP;
  mj_makeRawData(&d, m);
  if (!d) {
    // m will be deleted by the catch statement in mjCModel::Compile()
    throw mjCError(0, "could not create mjData");
  }
  mj_resetData(m, d);

  // normalize keyframe quaternions
  for (int i = 0; i < m->nkey; i++) { mj_normalizeQuat(m, m->key_qpos + i * m->nq); }

  // set constant fields, including those which compilation derives once
  mj_setCompiledConst(m, d);

  // initialize default plastic state and unset key_plastic vectors
  mj_resetPlastic(m, d);
  plastic0.assign(d->plastic, d->plastic + m->nplastic);
  if (m->nplastic && m->nkey) {
    std::vector<mjtNum> specplastic0(plastic0);
    ScaleKeyframe<mjtNum>(nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          specplastic0.data(),
                          /*inverse=*/true);
    for (int i = 0; i < m->nkey; i++) {
      if (keys_[i]->spec_plastic_.empty()) {
        mju_copy(m->key_plastic + i * m->nplastic, d->plastic, m->nplastic);
        for (int j = 0; j < m->nplastic; j++) { keys_[i]->plastic_[j] = (double)d->plastic[j]; }
      } else {
        for (int j = 0; j < m->nplastic; j++) {
          if (!mjuu_defined(keys_[i]->plastic_[j])) {
            keys_[i]->plastic_[j]               = (double)d->plastic[j];
            m->key_plastic[i * m->nplastic + j] = d->plastic[j];
          }
          if (j < (int)keys_[i]->spec_plastic_.size() &&
              !mjuu_defined(keys_[i]->spec_plastic_[j])) {
            keys_[i]->spec_plastic_[j] = (double)specplastic0[j];
          }
        }
        for (int j = 0; j < m->njnt; j++) {
          if (m->jnt_type[j] == mjJNT_BALL && m->jnt_plasticadr[j] >= 0) {
            mjuu_normvec(m->key_plastic + i * m->nplastic + m->jnt_plasticadr[j], 4);
          }
        }
      }
    }
  }

  // automatic spring-damper adjustment
  AutoSpringDamper(m);

  // actuator lengthrange computation
  LengthRange(m, d);

  // save automatically-computed statistics, to disambiguate when saving
  extent_auto      = m->stat.extent;
  meaninertia_auto = m->stat.meaninertia;
  meanmass_auto    = m->stat.meanmass;
  meansize_auto    = m->stat.meansize;
  mjuu_copyvec(center_auto, m->stat.center, 3);

  // override model statistics if defined by user
  if (mjuu_defined(stat.extent)) m->stat.extent = (mjtNum)stat.extent;
  if (mjuu_defined(stat.meaninertia)) m->stat.meaninertia = (mjtNum)stat.meaninertia;
  if (mjuu_defined(stat.meanmass)) m->stat.meanmass = (mjtNum)stat.meanmass;
  if (mjuu_defined(stat.meansize)) m->stat.meansize = (mjtNum)stat.meansize;
  if (mjuu_defined(stat.center[0])) mjuu_copyvec(m->stat.center, stat.center, 3);

  // assert that model has valid references
  const char* validationerr = mj_validateReferences(m);
  if (validationerr) {  // SHOULD NOT OCCUR
    // m and d will be deleted by the catch statement in mjCModel::Compile()
    throw mjCError(0, "%s", validationerr);
  }

  // delete partial mjData (no plugins), make a complete one
  mj_deleteData(d);
  d = nullptr;

  // if sleep was enabled, check for trees initialized as sleeping
  bool asleep_init = false;
  if (enableflags & mjENBL_SLEEP) {
    for (int i = 0; i < m->ntree; i++) {
      if (m->tree_sleep_policy[i] == mjSLEEP_INIT) {
        asleep_init = true;
        break;
      }
    }
  }

  // if any trees initialized as sleeping, restore flags before mj_makeData
  if (asleep_init) {
    m->opt.disableflags = disableflags;
    m->opt.enableflags  = enableflags;
  }

  d = mj_makeData(m);
  if (!d) {
    // m will be deleted by the catch statement in mjCModel::Compile()
    throw mjCError(0, "could not create mjData");
  }

  // pass compiler warnings into structured warning vector before validation
  if (warningtext[0]) {
    std::string        warnings(warningtext);
    std::istringstream stream(warnings);
    std::string        line;
    while (std::getline(stream, line)) {
      if (!line.empty()) { AddWarning(line); }
    }
  }

  // test forward simulation unless asleep_init is true (potentially expensive)
  // reset warningtext: engine warnings from validation are not compiler
  // warnings
  warningtext[0] = 0;
  if (!asleep_init) { mj_step(m, d); }

  // delete data, restore flags
  mj_deleteData(d);
  m->opt.disableflags = disableflags;
  m->opt.enableflags  = enableflags;
  d                   = nullptr;

  // the elements hold what the model was given, also where that was computed after they were
  // copied to it: by mj_setConst, mj_setLengthRange or the scaling of masses above
  BackValues(m, /*tospec=*/false, /*write=*/true);

  // save signature; the spec may have changed structurally during compilation,
  // and compilation itself may modify topology (fusestatic, discardvisual,
  // pairs, excludes)
  m->signature            = Signature();
  spec.element->signature = m->signature;

  timer[mjCTIMER_TOTAL] = Seconds(Clock::now() - timer_start).count();
}

static void PrintIndent(std::stringstream& ss, int depth) {
  // A static string of spaces, created only once during the program's lifetime.
  static const std::string spaces(1024, ' ');

  if (depth > 0) {
    // Write 'depth * 2' spaces directly to the stringstream
    // without creating any new std::string objects.
    ss.write(spaces.c_str(), std::min((size_t)depth * 2, spaces.length()));
  }
}


void mjCModel::PrintTree(std::stringstream& tree, const mjCBody* body, int depth) {
  if (depth == 1024) { throw mjCError(body, "depth limit exceeded in signature computation"); }
  PrintIndent(tree, depth);
  tree << "<body>\n";
  for (const auto& joint : body->joints) {
    PrintIndent(tree, depth + 1);
    tree << "<joint>" << std::to_string(joint->nq()) << "</joint>\n";
  }
  for (uint64_t i = 0; i < body->geoms.size(); ++i) {
    PrintIndent(tree, depth + 1);
    tree << "<geom/>\n";
  }
  for (uint64_t i = 0; i < body->sites.size(); ++i) {
    PrintIndent(tree, depth + 1);
    tree << "<site/>\n";
  }
  for (uint64_t i = 0; i < body->cameras.size(); ++i) {
    PrintIndent(tree, depth + 1);
    tree << "<camera/>\n";
  }
  for (uint64_t i = 0; i < body->lights.size(); ++i) {
    PrintIndent(tree, depth + 1);
    tree << "<light/>\n";
  }
  for (uint64_t i = 0; i < body->bodies.size(); ++i) {
    PrintTree(tree, body->bodies[i], depth + 1);
  }
  PrintIndent(tree, depth);
  tree << "</body>\n";
}


uint64_t mjCModel::Signature() {
  std::stringstream tree;
  tree << "\n";
  PrintTree(tree, bodies_[0]);
  for (unsigned int i = 0; i < flexes_.size(); ++i) { tree << "<flex/>\n"; }
  for (unsigned int i = 0; i < meshes_.size(); ++i) { tree << "<mesh/>\n"; }
  for (unsigned int i = 0; i < skins_.size(); ++i) { tree << "<skin/>\n"; }
  for (unsigned int i = 0; i < hfields_.size(); ++i) { tree << "<heightfield/>\n"; }
  for (unsigned int i = 0; i < textures_.size(); ++i) { tree << "<texture/>\n"; }
  for (unsigned int i = 0; i < materials_.size(); ++i) { tree << "<material/>\n"; }
  for (unsigned int i = 0; i < pairs_.size(); ++i) { tree << "<pair/>\n"; }
  for (unsigned int i = 0; i < excludes_.size(); ++i) { tree << "<exclude/>\n"; }
  for (unsigned int i = 0; i < equalities_.size(); ++i) { tree << "<equality/>\n"; }
  for (unsigned int i = 0; i < tendons_.size(); ++i) { tree << "<tendon/>\n"; }
  for (unsigned int i = 0; i < actuators_.size(); ++i) { tree << "<actuator/>\n"; }
  for (unsigned int i = 0; i < sensors_.size(); ++i) {
    tree << "<sensor>" << std::to_string(sensors_[i]->spec.type) << "<sensor/>\n";
  }
  for (unsigned int i = 0; i < keys_.size(); ++i) { tree << "<key/>\n"; }
  return mj_hashString(tree.str().c_str(), UINT64_MAX);
}


bool mjCModel::CheckBodyMassInertia(mjCBody* body) {
  // check if body has valid mass and inertia
  if (body->mass >= mjMINVAL &&
      body->inertia[0] >= mjMINVAL &&
      body->inertia[1] >= mjMINVAL &&
      body->inertia[2] >= mjMINVAL) {
    return true;
  }

  // body is valid if we find a single static child with valid mass and inertia
  for (int i = 0; i < body->Bodies().size(); i++) {
    // if we find a child with a joint, time to move on to the next moving body
    if (!body->Bodies()[i]->joints.empty()) { continue; }
    if (CheckBodyMassInertia(body->Bodies()[i])) { return true; }
  }

  // we did not find a child with valid mass and inertia
  return false;
}


//------------------------------- DECOMPILER -------------------------------------------------------

namespace {

// true if the model holds another value than the element
template <typename T, typename S>
bool Differs(const T* element, const S* model, int n) {
  for (int i = 0; i < n; i++) {
    if (static_cast<S>(element[i]) != model[i]) { return true; }
  }
  return false;
}

// copy to an element the values which differ in the model. An element holds its values in double
// precision whatever the precision of the model, and the values which are the same keep it
template <typename T, typename S>
void Back(T* element, const S* model, int n = 1) {
  for (int i = 0; i < n; i++) {
    if (static_cast<S>(element[i]) != model[i]) { element[i] = static_cast<T>(model[i]); }
  }
}

}  // namespace


// get numeric data back from mjModel
bool mjCModel::CopyBack(const mjModel* m) {
  // check for null pointer
  if (!m) {
    errInfo = mjCError(0, "mjModel pointer is null in CopyBack");
    return false;
  }

  // make sure model has been compiled
  if (!compiled) {
    errInfo = mjCError(0, "mjCModel has not been compiled in CopyBack");
    return false;
  }

  // a value is copied if it is not the one which the elements hold from the compilation; an
  // element of a copy whose references were not found is taken again from its spec, so every
  // value which compilation derives for it would count as changed
  if (!baseline_) {
    errInfo = mjCError(0, "copy of mjSpec does not hold what was compiled in CopyBack");
    return false;
  }

  // make sure sizes match
  if (nq != m->nq ||
      nv != m->nv ||
      nu != m->nu ||
      nactuator != m->nactuator ||
      nout != m->nout ||
      na != m->na ||
      nbody != m->nbody ||
      njnt != m->njnt ||
      ngeom != m->ngeom ||
      nsite != m->nsite ||
      ncam != m->ncam ||
      nlight != m->nlight ||
      nmesh != m->nmesh ||
      nskin != m->nskin ||
      nhfield != m->nhfield ||
      nmat != m->nmat ||
      ntex != m->ntex ||
      npair != m->npair ||
      nexclude != m->nexclude ||
      neq != m->neq ||
      ntendon != m->ntendon ||
      nwrap != m->nwrap ||
      nsensor != m->nsensor ||
      nnumeric != m->nnumeric ||
      nnumericdata != m->nnumericdata ||
      ntext != m->ntext ||
      ntextdata != m->ntextdata ||
      nnames != m->nnames ||
      nM != m->nM ||
      nD != m->nD ||
      nC != m->nC ||
      nB != m->nB ||
      nJmom != m->nJmom ||
      nemax != m->nemax ||
      nconmax != m->nconmax ||
      njmax != m->njmax ||
      npaths != m->npaths ||
      ntuple != m->ntuple ||
      ntupledata != m->ntupledata ||
      nkey != m->nkey ||
      nmocap != m->nmocap ||
      nplastic != m->nplastic ||
      nhfielddata != m->nhfielddata ||
      nuser_body != m->nuser_body ||
      nuser_jnt != m->nuser_jnt ||
      nuser_geom != m->nuser_geom ||
      nuser_site != m->nuser_site ||
      nuser_cam != m->nuser_cam ||
      nuser_tendon != m->nuser_tendon ||
      nuser_actuator != m->nuser_actuator ||
      nuser_sensor != m->nuser_sensor) {
    errInfo = mjCError(0, "incompatible models in CopyBack");
    return false;
  }

  if (spec.element->signature != m->signature) {
    errInfo = mjCError(0, "incompatible signatures in CopyBack");
    return false;
  }

  // every value which the spec cannot express is reported before anything is written
  try {
    BackValues(m, /*tospec=*/true, /*write=*/false);
  } catch (mjCError err) {
    errInfo = err;
    return false;
  }
  BackValues(m, /*tospec=*/true, /*write=*/true);
  return true;
}


// copy to the elements the values of the model which differ from those they hold. Compilation
// ends with this, so that the elements of a compiled spec hold what the model was given; after
// that, these are the values which were changed in the model. With `tospec` they are also written
// to the spec, each as what compiles to it, and a value which the spec cannot express is an
// error. Without `write` nothing is copied and only the errors are raised
void mjCModel::BackValues(const mjModel* m, bool tospec, bool write) {
  // the compiled values of the element which is being copied, and the inverse of its gauge, which
  // maps them back to the units of the spec
  struct {
    char*                 begin = nullptr;
    size_t                size  = 0;
    std::function<void()> inverse;
  } scaled;
  auto unscaling = [&](auto* compiled, std::function<void()> inverse) {
    if (!tospec) { return; }
    scaled.begin   = reinterpret_cast<char*>(compiled);
    scaled.size    = sizeof(*compiled);
    scaled.inverse = std::move(inverse);
  };

  // write compiled values to the spec, in its units: those of the element which is being copied
  // are taken from its compiled values as the inverse gauge leaves them
  auto tospecvalues = [&](auto* value, const auto* element, int n) {
    const char* p = reinterpret_cast<const char*>(element);
    if (!scaled.inverse || p < scaled.begin || p >= scaled.begin + scaled.size) {
      std::copy_n(element, n, value);
      return;
    }
    std::vector<char> saved(scaled.begin, scaled.begin + scaled.size);
    scaled.inverse();
    std::copy_n(element, n, value);
    std::memcpy(scaled.begin, saved.data(), scaled.size);
  };

  // a value which compilation copies to the model as it is; true if the model changed it
  auto copy = [&](auto* value, auto* element, const auto* model, int n = 1) {
    if (!Differs(element, model, n)) { return false; }
    if (write) {
      Back(element, model, n);
      if (tospec) { tospecvalues(value, element, n); }
    }
    return true;
  };

  // the same for values which are held in a vector
  auto copyvector = [&](auto& value, auto& element, const auto* model, int n) {
    if (!Differs(element.data(), model, n)) { return false; }
    if (write) {
      Back(element.data(), model, n);
      if (tospec) { value = element; }
    }
    return true;
  };

  // a range, which the spec gives multiplied by `scale`. Whether it limits may be inferred from
  // the range: the flag is set if the inference would not give what the model has
  auto copyrange = [&](double*            value,
                       double*            element,
                       const mjtNum*      model,
                       mjtLimited&        valuelimited,
                       mjtLimited&        elementlimited,
                       bool               modellimited,
                       const mjsCompiler* settings,
                       double             scale = 1) {
    const bool lower = Differs(element, model, 1);
    const bool upper = Differs(element + 1, model + 1, 1);
    if (!lower && !upper) { return false; }
    if (write && tospec) {
      Back(element, model, 2);
      double range[2];
      tospecvalues(range, element, 2);
      if (lower) { value[0] = scale * range[0]; }
      if (upper) { value[1] = scale * range[1]; }
      if (valuelimited == mjLIMITED_AUTO) {
        bool hasrange = value[0] != 0 || value[1] != 0;
        if ((!settings->autolimits && hasrange) || (value[0] < value[1]) != modellimited) {
          valuelimited = elementlimited = modellimited ? mjLIMITED_TRUE : mjLIMITED_FALSE;
        }
      }
    } else if (write) {
      Back(element, model, 2);
    }
    return true;
  };

  // a value which was changed in the model and which the spec cannot express
  auto refuse = [&](const mjCBase* element, const std::string& what, const std::string& why) {
    if (tospec && !write) {
      throw mjCError(element, "%s", (what + " was changed in the model: " + why).c_str());
    }
  };

  // data of the model in its reference configuration, made when it is first needed
  std::unique_ptr<mjData, void (*)(mjData*)> reference(nullptr, mj_deleteData);

  // the anchor which a connect between bodies has in its second body, or the relative pose of a
  // weld between bodies, as mj_setConst computes it from the model as it is now
  auto computeddata = [&](int i, mjtNum* computed) {
    if (!reference) {
      reference.reset(mj_makeData(m));
      mj_kinematics(m, reference.get());
    }
    const mjData* d    = reference.get();
    const mjtNum* data = m->eq_data + mjNEQDATA * i;
    int           id1  = m->eq_obj1id[i];
    int           id2  = m->eq_obj2id[i];
    mjtNum        pos[3], quat[4];
    if (m->eq_type[i] == mjEQ_CONNECT) {
      mj_local2Global(reference.get(), pos, 0, data, 0, id1, 0);
      mju_subFrom3(pos, d->xpos + 3 * id2);
      mju_mulMatTVec3(computed, d->xmat + 9 * id2, pos);
    } else {
      mj_local2Global(reference.get(), pos, 0, data, 0, id2, 0);
      mju_subFrom3(pos, d->xpos + 3 * id1);
      mju_mulMatTVec3(computed, d->xmat + 9 * id1, pos);
      mju_negQuat(quat, d->xquat + 4 * id1);
      mju_mulQuat(computed + 3, quat, d->xquat + 4 * id2);
    }
  };

  // option and visual
#define X(type, name, n)                                                        \
  if (copy(&spec.option.name, &option.name, &m->opt.name) && tospec && write) { \
    mjs_setAuthored(&spec, &spec.option.name, 1);                               \
  }
#define XVEC(type, name, n)                                                     \
  if (copy(spec.option.name, option.name, m->opt.name, n) && tospec && write) { \
    mjs_setAuthored(&spec, spec.option.name, 1);                                \
  }
  MJOPTION_FIELDS
#undef X
#undef XVEC

#define mjBACKVISUAL(group, FIELDS)    \
  {                                    \
    auto& value   = spec.visual.group; \
    auto& element = visual.group;      \
    auto& model   = m->vis.group;      \
    FIELDS                             \
  }
#define X(type, name, n)                                                  \
  if (copy(&value.name, &element.name, &model.name) && tospec && write) { \
    mjs_setAuthored(&spec, &value.name, 1);                               \
  }
#define XVEC(type, name, n)                                               \
  if (copy(value.name, element.name, model.name, n) && tospec && write) { \
    mjs_setAuthored(&spec, value.name, 1);                                \
  }
  mjBACKVISUAL(global, MJVISUAL_GLOBAL_FIELDS);
  mjBACKVISUAL(quality, MJVISUAL_QUALITY_FIELDS);
  mjBACKVISUAL(headlight, MJVISUAL_HEADLIGHT_FIELDS);
  mjBACKVISUAL(map, MJVISUAL_MAP_FIELDS);
  mjBACKVISUAL(scale, MJVISUAL_SCALE_FIELDS);
  mjBACKVISUAL(rgba, MJVISUAL_RGBA_FIELDS);
#undef X
#undef XVEC
#undef mjBACKVISUAL

  // statistics: the model was given the value which is set, or else the one computed for it
  auto statistic =
      [&](mjtNum* value, mjtNum* element, const mjtNum* model, const double* computed, int n) {
        bool isset   = mjuu_defined(element[0]);
        bool changed = false;
        for (int i = 0; i < n; i++) {
          changed |= model[i] != (isset ? element[i] : static_cast<mjtNum>(computed[i]));
        }
        if (changed && write) {
          std::copy_n(model, n, element);
          if (tospec) { std::copy_n(model, n, value); }
        }
      };
  statistic(&spec.stat.meaninertia, &stat.meaninertia, &m->stat.meaninertia, &meaninertia_auto, 1);
  statistic(&spec.stat.meanmass, &stat.meanmass, &m->stat.meanmass, &meanmass_auto, 1);
  statistic(&spec.stat.meansize, &stat.meansize, &m->stat.meansize, &meansize_auto, 1);
  statistic(&spec.stat.extent, &stat.extent, &m->stat.extent, &extent_auto, 1);
  statistic(spec.stat.center, stat.center, m->stat.center, center_auto, 3);

  // joint and dof
  for (int i = 0; i < njnt; i++) {
    mjCJoint* pj      = joints_[i];
    int       qposadr = m->jnt_qposadr[i];
    int       dofadr  = m->jnt_dofadr[i];
    unscaling(static_cast<mjsJoint*>(pj), [pj] { pj->ApplyGauge(/*inverse=*/true); });

    // an angle is in degrees in a spec which says so
    const bool   rotates = pj->type == mjJNT_HINGE || pj->type == mjJNT_BALL;
    const double degrees = pj->compiler->degree ? 180 / mjPI : 1;

    // qpos0, qpos_spring: those of a free joint are the pose of its body, below, and those of a
    // ball joint are the unit quaternion
    if (pj->type == mjJNT_SLIDE || pj->type == mjJNT_HINGE) {
      const double scale = pj->type == mjJNT_HINGE ? degrees : 1;
      if (copy(&pj->spec.ref, &pj->ref, m->qpos0 + qposadr) && tospec && write) {
        pj->spec.ref *= scale;
      }
      if (copy(&pj->spec.springref, &pj->springref, m->qpos_spring + qposadr) && tospec && write) {
        pj->spec.springref *= scale;
      }
    }

    // anchor and axis: those of a free joint are fixed, as is the axis of a ball joint
    bool anchor    = Differs(pj->pos, m->jnt_pos + 3 * i, 3);
    bool direction = Differs(pj->axis, m->jnt_axis + 3 * i, 3);
    if (anchor || direction) {
      if (pj->type == mjJNT_FREE) {
        refuse(pj, "the anchor or axis of a free joint", "they are fixed");
      } else if (direction && pj->type == mjJNT_BALL) {
        refuse(pj, "the axis of a ball joint", "it is fixed");
      }
      if (write) {
        Back(pj->pos, m->jnt_pos + 3 * i, 3);
        Back(pj->axis, m->jnt_axis + 3 * i, 3);
        if (tospec) { pj->AnchorToSpec(anchor, direction); }
      }
    }

    // the spec gives one value to all the degrees of freedom of a ball or a free joint
    const int ndof   = m->jnt_type[i] == mjJNT_FREE ? 6 : (m->jnt_type[i] == mjJNT_BALL ? 3 : 1);
    auto      shared = [&](const char* name, const mjtNum* model, int n = 1) {
      for (int k = n; k < n * ndof; k++) {
        if (model[n * dofadr + k] != model[n * dofadr + k % n]) {
          refuse(pj,
                 name,
                 "its degrees of freedom have different values, and the spec gives them one");
          return;
        }
      }
    };
    shared("the damping of a joint", m->dof_damping);
    shared("the damping of a joint", m->dof_dampingpoly, mjNPOLY);
    shared("the armature of a joint", m->dof_armature);
    shared("the friction loss of a joint", m->dof_frictionloss);
    shared("the solver parameters of the friction loss of a joint", m->dof_solref, mjNREF);
    shared("the solver parameters of the friction loss of a joint", m->dof_solimp, mjNIMP);

    // stiffness and damping: springdamper computes both, so a change of either takes its place
    bool spring = copy(pj->spec.stiffness, pj->stiffness, m->jnt_stiffness + i);
    bool damper = copy(pj->spec.damping, pj->damping, m->dof_damping + dofadr);
    if ((spring || damper) &&
        tospec &&
        write &&
        pj->springdamper[0] > 0 &&
        pj->springdamper[1] > 0) {
      tospecvalues(pj->spec.stiffness, pj->stiffness, 1);
      tospecvalues(pj->spec.damping, pj->damping, 1);
      pj->springdamper[0] = pj->springdamper[1] = 0;
      pj->spec.springdamper[0] = pj->spec.springdamper[1] = 0;
    }
    copy(pj->spec.stiffness + 1, pj->stiffness + 1, m->jnt_stiffnesspoly + mjNPOLY * i, mjNPOLY);
    copy(pj->spec.damping + 1, pj->damping + 1, m->dof_dampingpoly + mjNPOLY * dofadr, mjNPOLY);
    if (copy(&pj->spec.springyield, &pj->springyield, m->jnt_springyield + i) && tospec && write) {
      if (rotates) { pj->spec.springyield *= degrees; }
    }

    // range: the limits of a rotation are in the unit of the spec if they limit
    copyrange(pj->spec.range,
              pj->range,
              m->jnt_range + 2 * i,
              pj->spec.limited,
              pj->limited,
              m->jnt_limited[i],
              pj->compiler,
              rotates && m->jnt_limited[i] ? degrees : 1);

    // other joint data
    copy(pj->spec.solref_limit, pj->solref_limit, m->jnt_solref + mjNREF * i, mjNREF);
    copy(pj->spec.solimp_limit, pj->solimp_limit, m->jnt_solimp + mjNIMP * i, mjNIMP);
    copy(&pj->spec.margin, &pj->margin, m->jnt_margin + i);
    copyvector(pj->spec_userdata_, pj->userdata_, m->jnt_user + nuser_jnt * i, nuser_jnt);

    // other dof data
    copy(pj->spec.solref_friction, pj->solref_friction, m->dof_solref + mjNREF * dofadr, mjNREF);
    copy(pj->spec.solimp_friction, pj->solimp_friction, m->dof_solimp + mjNIMP * dofadr, mjNIMP);
    copy(&pj->spec.armature, &pj->armature, m->dof_armature + dofadr);
    copy(&pj->spec.frictionloss, &pj->frictionloss, m->dof_frictionloss + dofadr);
  }

  // body
  for (int i = 0; i < nbody; i++) {
    mjCBody* pb = bodies_[i];

    // pose: the simulation reads that of a body with a free joint from qpos0, so a change there
    // takes precedence
    const mjtNum* pose0 = nullptr;
    for (const mjCJoint* pj : pb->joints) {
      int qposadr = m->jnt_qposadr[pj->id];
      if (pj->type == mjJNT_FREE && Differs(qpos0.data() + qposadr, m->qpos0 + qposadr, 7)) {
        pose0 = m->qpos0 + qposadr;
      }
    }
    const mjtNum* modelpos    = pose0 ? pose0 : m->body_pos + 3 * i;
    const mjtNum* modelquat   = pose0 ? pose0 + 3 : m->body_quat + 4 * i;
    bool          position    = Differs(pb->pos, modelpos, 3);
    bool          orientation = Differs(pb->quat, modelquat, 4);
    if (position || orientation) {
      if (i == 0) { refuse(pb, "the pose of the world body", "it is fixed"); }
      if (write) {
        Back(pb->pos, modelpos, 3);
        Back(pb->quat, modelquat, 4);
        if (tospec && i > 0) { pb->PoseToSpec(position, orientation); }
      }
    }

    // inertial: one which is new is given to the body, which then no longer infers it from geoms
    bool newmass = Differs(&pb->mass, m->body_mass + i, 1);
    bool newframe =
        Differs(pb->ipos, m->body_ipos + 3 * i, 3) || Differs(pb->iquat, m->body_iquat + 4 * i, 4);
    bool newinertia = Differs(pb->inertia, m->body_inertia + 3 * i, 3);
    if (newmass || newframe || newinertia) {
      const char* what = "the mass or inertia of a body";
      if (i == 0) {
        refuse(pb, what, "the world body has none");
      } else if (pb->compiler->inertiafromgeom == mjINERTIAFROMGEOM_TRUE) {
        refuse(pb, what, "inertiafromgeom is 'true', so they are inferred from geoms");
      } else if (compiler.settotalmass > 0 || spec.compiler.settotalmass > 0) {
        refuse(pb, what, "settotalmass scales those of all bodies");
      } else if (newframe && pb->aligned_) {
        refuse(pb,
               "the inertial frame of a body",
               "the body is aligned with its free joint, and its frame would move with it");
      }
      if (write) {
        Back(&pb->mass, m->body_mass + i, 1);
        Back(pb->ipos, m->body_ipos + 3 * i, 3);
        Back(pb->iquat, m->body_iquat + 4 * i, 4);
        Back(pb->inertia, m->body_inertia + 3 * i, 3);
        if (tospec && i > 0) { pb->InertialToSpec(/*massonly=*/!newframe && !newinertia); }
      }
    }

    copyvector(pb->spec_userdata_, pb->userdata_, m->body_user + nuser_body * i, nuser_body);
  }
  if (write) { mjuu_copyvec(qpos0.data(), m->qpos0, m->nq); }

  // geom
  for (int i = 0; i < ngeom; i++) {
    mjCGeom* pg = geoms_[i];
    unscaling(static_cast<mjsGeom*>(pg), [pg] { pg->ApplyGauge(/*inverse=*/true); });

    // size, pose and surface velocity: the size of a mesh or height field geom is that of the asset
    bool newsize     = Differs(pg->size, m->geom_size + 3 * i, 3);
    bool position    = Differs(pg->pos, m->geom_pos + 3 * i, 3);
    bool orientation = Differs(pg->quat, m->geom_quat + 4 * i, 4);
    bool newvelocity = Differs(pg->surfacevel, m->geom_surfacevel + 6 * i, 6);
    if (newsize || position || orientation || newvelocity) {
      if (newsize && (pg->type == mjGEOM_MESH || pg->type == mjGEOM_SDF)) {
        refuse(pg, "the size of a mesh geom", "it is computed from the mesh");
      } else if (newsize && pg->type == mjGEOM_HFIELD) {
        refuse(pg, "the size of a height field geom", "it is that of the height field");
      }
      if (write) {
        Back(pg->size, m->geom_size + 3 * i, 3);
        Back(pg->pos, m->geom_pos + 3 * i, 3);
        Back(pg->quat, m->geom_quat + 4 * i, 4);
        Back(pg->surfacevel, m->geom_surfacevel + 6 * i, 6);
        if (tospec) { pg->ShapeToSpec(newsize, position, orientation, newvelocity); }
      }
    }

    copy(pg->spec.friction, pg->friction, m->geom_friction + 3 * i, 3);
    copy(pg->spec.solref, pg->solref, m->geom_solref + mjNREF * i, mjNREF);
    copy(pg->spec.solimp, pg->solimp, m->geom_solimp + mjNIMP * i, mjNIMP);
    copy(pg->spec.rgba, pg->rgba, m->geom_rgba + 4 * i, 4);
    copy(&pg->spec.solmix, &pg->solmix, m->geom_solmix + i);
    copy(&pg->spec.margin, &pg->margin, m->geom_margin + i);
    copy(&pg->spec.gap, &pg->gap, m->geom_gap + i);
    copy(&pg->spec.adhesion, &pg->adhesion, m->geom_adhesion + i);
    copyvector(pg->spec_userdata_, pg->userdata_, m->geom_user + nuser_geom * i, nuser_geom);
  }

  // mesh: the frame which compilation gave it is not something the spec gives
  const std::vector<mjCMesh*> meshes = CompiledMeshes();
  for (int i = 0; i < nmesh; i++) {
    mjCMesh* pm = meshes[i];
    if (Differs(pm->GetPosPtr(), m->mesh_pos + 3 * i, 3) ||
        Differs(pm->GetQuatPtr(), m->mesh_quat + 4 * i, 4)) {
      refuse(pm, "the frame of a mesh", "it is computed from the mesh");
      if (write) {
        Back(pm->GetPosPtr(), m->mesh_pos + 3 * i, 3);
        Back(pm->GetQuatPtr(), m->mesh_quat + 4 * i, 4);
      }
    }
  }

  // heightfield: the model holds the elevation data scaled to [0, 1]. Data which the spec gives
  // stays as it was given unless the model changed it, and data which was read from a file is
  // then given in place of the file. A height field which is used at several scales is written
  // once, from its instances, which must then agree
  const std::vector<mjCHField*> hfields = CompiledHFields();
  for (int i = 0; i < nhfield; i++) {
    mjCHField*   ph    = hfields[i];
    const float* model = m->hfield_data + m->hfield_adr[i];
    int          n     = m->hfield_nrow[i] * m->hfield_ncol[i];
    if (!Differs(ph->data.data(), model, n)) { continue; }

    // the instances of the same height field
    const mjCHField* primary = ph->primary_ ? ph->primary_ : ph;
    for (int j = 0; j < nhfield; j++) {
      bool family = hfields[j] == primary || hfields[j]->primary_ == primary;
      if (family && Differs(m->hfield_data + m->hfield_adr[j], model, n)) {
        refuse(ph,
               "the elevation data of an instance of a height field",
               "the height field is used at several scales, and its instances differ");
      }
    }

    // compilation scales the data again: only data which it leaves as it is can be given
    float lowest = model[0], highest = model[0];
    for (int k = 1; k < n; k++) {
      lowest  = std::min(lowest, model[k]);
      highest = std::max(highest, model[k]);
    }
    if (lowest != 0 || (highest != 1 && highest - lowest > mjEPS)) {
      refuse(ph,
             "the elevation data of a height field",
             "its lowest and highest values are not 0 and 1, to which compilation scales them");
    }
    if (write) {
      Back(ph->data.data(), model, n);
      if (tospec && !ph->primary_) {
        ph->userdata_ = ph->data;
        ph->file_.clear();
        ph->content_type_.clear();
        ph->spec_userdata_ = ph->data;
        ph->spec_file_.clear();
        ph->spec_content_type_.clear();
        ph->spec.nrow = ph->nrow;
        ph->spec.ncol = ph->ncol;
      } else if (!ph->userdata_.empty()) {
        ph->userdata_ = ph->data;
      }
    }
  }

  // sites
  for (int i = 0; i < nsite; i++) {
    mjCSite* ps = sites_[i];
    unscaling(static_cast<mjsSite*>(ps), [ps] { ps->ApplyGauge(/*inverse=*/true); });

    // size and pose: the size of a mesh site is that of the mesh
    bool newsize     = Differs(ps->size, m->site_size + 3 * i, 3);
    bool position    = Differs(ps->pos, m->site_pos + 3 * i, 3);
    bool orientation = Differs(ps->quat, m->site_quat + 4 * i, 4);
    if (newsize || position || orientation) {
      if (newsize && ps->type == mjGEOM_MESH) {
        refuse(ps, "the size of a mesh site", "it is computed from the mesh");
      }
      if (write) {
        Back(ps->size, m->site_size + 3 * i, 3);
        Back(ps->pos, m->site_pos + 3 * i, 3);
        Back(ps->quat, m->site_quat + 4 * i, 4);
        if (tospec) { ps->ShapeToSpec(newsize, position, orientation); }
      }
    }

    copy(ps->spec.rgba, ps->rgba, m->site_rgba + 4 * i, 4);
    copyvector(ps->spec_userdata_, ps->userdata_, m->site_user + nuser_site * i, nuser_site);
  }

  // cameras
  for (int i = 0; i < ncam; i++) {
    mjCCamera* pc     = cameras_[i];
    const bool sensor = pc->sensor_size[0] > 0 && pc->sensor_size[1] > 0;
    unscaling(static_cast<mjsCamera*>(pc), [pc] { pc->ApplyGauge(/*inverse=*/true); });

    // pose
    bool position    = Differs(pc->pos, m->cam_pos + 3 * i, 3);
    bool orientation = Differs(pc->quat, m->cam_quat + 4 * i, 4);
    if ((position || orientation) && write) {
      Back(pc->pos, m->cam_pos + 3 * i, 3);
      Back(pc->quat, m->cam_quat + 4 * i, 4);
      if (tospec) { pc->PoseToSpec(position, orientation); }
    }

    // field of view: that of a camera with a sensor is computed from the focal length
    if (Differs(&pc->fovy, m->cam_fovy + i, 1)) {
      if (sensor) {
        refuse(pc, "the field of view of a camera", "it is computed from its sensor size");
      }
      copy(&pc->spec.fovy, &pc->fovy, m->cam_fovy + i);
    }

    // intrinsics: the focal length and principal point of a camera with a sensor, which may be
    // given in pixels of a resolution; otherwise they are computed
    bool resolution = copy(pc->spec.resolution, pc->resolution, m->cam_resolution + 2 * i, 2);
    bool intrinsic  = Differs(pc->intrinsic, m->cam_intrinsic + 4 * i, 4);
    if (intrinsic && !sensor) {
      refuse(pc, "the intrinsics of a camera", "it has no sensor size, so they are computed");
    }
    if ((intrinsic || (resolution && sensor)) && write) {
      Back(pc->intrinsic, m->cam_intrinsic + 4 * i, 4);
      if (sensor) { pc->IntrinsicToSpec(tospec); }
    }

    copy(&pc->spec.ipd, &pc->ipd, m->cam_ipd + i);
    copyvector(pc->spec_userdata_, pc->userdata_, m->cam_user + nuser_cam * i, nuser_cam);
  }

  // lights
  for (int i = 0; i < nlight; i++) {
    mjCLight* pl = lights_[i];
    unscaling(static_cast<mjsLight*>(pl), [pl] { pl->ApplyGauge(/*inverse=*/true); });

    // position and direction
    bool position  = Differs(pl->pos, m->light_pos + 3 * i, 3);
    bool direction = Differs(pl->dir, m->light_dir + 3 * i, 3);
    if ((position || direction) && write) {
      Back(pl->pos, m->light_pos + 3 * i, 3);
      Back(pl->dir, m->light_dir + 3 * i, 3);
      if (tospec) { pl->PoseToSpec(position, direction); }
    }

    copy(pl->spec.attenuation, pl->attenuation, m->light_attenuation + 3 * i, 3);
    copy(&pl->spec.cutoff, &pl->cutoff, m->light_cutoff + i);
    copy(&pl->spec.exponent, &pl->exponent, m->light_exponent + i);
    copy(pl->spec.ambient, pl->ambient, m->light_ambient + 3 * i, 3);
    copy(pl->spec.diffuse, pl->diffuse, m->light_diffuse + 3 * i, 3);
    copy(pl->spec.specular, pl->specular, m->light_specular + 3 * i, 3);
  }

  // materials
  for (int i = 0; i < nmat; i++) {
    mjCMaterial* pm = materials_[i];

    copy(&pm->spec.texuniform, &pm->texuniform, m->mat_texuniform + i);
    copy(pm->spec.texrepeat, pm->texrepeat, m->mat_texrepeat + 2 * i, 2);
    copy(&pm->spec.emission, &pm->emission, m->mat_emission + i);
    copy(&pm->spec.specular, &pm->specular, m->mat_specular + i);
    copy(&pm->spec.shininess, &pm->shininess, m->mat_shininess + i);
    copy(&pm->spec.reflectance, &pm->reflectance, m->mat_reflectance + i);
    copy(&pm->spec.metallic, &pm->metallic, m->mat_metallic + i);
    copy(&pm->spec.roughness, &pm->roughness, m->mat_roughness + i);
    copy(pm->spec.rgba, pm->rgba, m->mat_rgba + 4 * i, 4);
  }

  // pairs, which the model holds in the order of their ids
  for (mjCPair* pair : pairs_) {
    int i = pair->id;
    unscaling(static_cast<mjsPair*>(pair), [pair] { pair->ApplyGauge(/*inverse=*/true); });
    copy(pair->spec.solref, pair->solref, m->pair_solref + mjNREF * i, mjNREF);
    copy(pair->spec.solreffriction,
         pair->solreffriction,
         m->pair_solreffriction + mjNREF * i,
         mjNREF);
    copy(pair->spec.solimp, pair->solimp, m->pair_solimp + mjNIMP * i, mjNIMP);
    copy(&pair->spec.margin, &pair->margin, m->pair_margin + i);
    copy(&pair->spec.gap, &pair->gap, m->pair_gap + i);
    copy(&pair->spec.adhesion, &pair->adhesion, m->pair_adhesion + i);
    copy(pair->spec.friction, pair->friction, m->pair_friction + 5 * i, 5);
  }

  // equality constraints
  for (int i = 0; i < neq; i++) {
    mjCEquality*  pe   = equalities_[i];
    const mjtNum* data = m->eq_data + mjNEQDATA * i;
    unscaling(static_cast<mjsEquality*>(pe), [pe] { pe->ApplyGauge(/*inverse=*/true); });

    if (pe->type == mjEQ_CONNECT && pe->objtype == mjOBJ_BODY) {
      // a connect between bodies: the anchor in the first body is given, and the one in the
      // second body is computed from it in the reference configuration
      copy(pe->spec.data, pe->data, data, 3);
      if (Differs(pe->data + 3, data + 3, 3)) {
        // told apart only where it is reported: compilation ends here too, with this anchor new
        if (tospec && !write) {
          mjtNum computed[3];
          computeddata(i, computed);
          if (Differs(computed, data + 3, 3)) {
            refuse(pe,
                   "the anchor of a connect in its second body",
                   "it is computed from the anchor in the first body");
          }
        }
        if (write) { Back(pe->data + 3, data + 3, 3); }
      }
      copy(pe->spec.data + 6, pe->data + 6, data + 6, mjNEQDATA - 6);
    } else if (pe->type == mjEQ_WELD && pe->objtype == mjOBJ_BODY) {
      // a weld between bodies: the anchor and the torque scale are given, and so is the relative
      // pose unless it is marked in data[11], computed from the anchor; the spec marks it with a
      // zero quaternion
      copy(pe->spec.data, pe->data, data, 3);
      const bool marked = data[11];
      if (Differs(pe->data + 3, data + 3, 7) || Differs(pe->data + 11, data + 11, 1)) {
        if (tospec && !write && marked) {
          mjtNum computed[7];
          computeddata(i, computed);
          if (Differs(computed, data + 3, 7)) {
            refuse(pe,
                   "the relative pose of a weld which is marked as computed",
                   "it is computed from the anchor");
          }
        }
        if (write) {
          Back(pe->data + 3, data + 3, 7);
          Back(pe->data + 11, data + 11, 1);
          if (tospec && marked) {
            std::fill(pe->spec.data + 6, pe->spec.data + 10, 0);
          } else if (tospec) {
            tospecvalues(pe->spec.data + 3, pe->data + 3, 7);
          }
        }
      }
      copy(pe->spec.data + 10, pe->data + 10, data + 10);
    } else {
      copy(pe->spec.data, pe->data, data, mjNEQDATA);
    }
    copy(pe->spec.solref, pe->solref, m->eq_solref + mjNREF * i, mjNREF);
    copy(pe->spec.solimp, pe->solimp, m->eq_solimp + mjNIMP * i, mjNIMP);
  }

  // tendons
  for (int i = 0; i < ntendon; i++) {
    mjCTendon* pt = tendons_[i];
    unscaling(static_cast<mjsTendon*>(pt), [pt] { pt->ApplyGauge(/*inverse=*/true); });

    copyrange(pt->spec.range,
              pt->range,
              m->tendon_range + 2 * i,
              pt->spec.limited,
              pt->limited,
              m->tendon_limited[i],
              pt->compiler);
    copyrange(pt->spec.actfrcrange,
              pt->actfrcrange,
              m->tendon_actfrcrange + 2 * i,
              pt->spec.actfrclimited,
              pt->actfrclimited,
              m->tendon_actfrclimited[i],
              pt->compiler);
    copy(pt->spec.solref_limit, pt->solref_limit, m->tendon_solref_lim + mjNREF * i, mjNREF);
    copy(pt->spec.solimp_limit, pt->solimp_limit, m->tendon_solimp_lim + mjNIMP * i, mjNIMP);
    copy(pt->spec.solref_friction, pt->solref_friction, m->tendon_solref_fri + mjNREF * i, mjNREF);
    copy(pt->spec.solimp_friction, pt->solimp_friction, m->tendon_solimp_fri + mjNIMP * i, mjNIMP);
    copy(pt->spec.rgba, pt->rgba, m->tendon_rgba + 4 * i, 4);
    copy(&pt->spec.width, &pt->width, m->tendon_width + i);
    copy(&pt->spec.margin, &pt->margin, m->tendon_margin + i);
    copy(pt->spec.stiffness, pt->stiffness, m->tendon_stiffness + i);
    copy(pt->spec.stiffness + 1, pt->stiffness + 1, m->tendon_stiffnesspoly + mjNPOLY * i, mjNPOLY);
    copy(pt->spec.damping, pt->damping, m->tendon_damping + i);
    copy(pt->spec.damping + 1, pt->damping + 1, m->tendon_dampingpoly + mjNPOLY * i, mjNPOLY);
    copy(&pt->spec.armature, &pt->armature, m->tendon_armature + i);
    copy(&pt->spec.frictionloss, &pt->frictionloss, m->tendon_frictionloss + i);
    copy(pt->spec.springlength, pt->springlength, m->tendon_lengthspring + 2 * i, 2);
    copy(&pt->spec.springyield, &pt->springyield, m->tendon_springyield + i);
    copyvector(pt->spec_userdata_, pt->userdata_, m->tendon_user + nuser_tendon * i, nuser_tendon);
  }

  // actuators
  for (int i = 0; i < nactuator; i++) {
    mjCActuator* pa = actuators_[i];
    unscaling(static_cast<mjsActuator*>(pa), [pa] { pa->ApplyGauge(/*inverse=*/true); });

    copy(pa->spec.dynprm, pa->dynprm, m->actuator_dynprm + mjNDYN * i, mjNDYN);
    copy(pa->spec.gainprm, pa->gainprm, m->actuator_gainprm + mjNGAIN * i, mjNGAIN);
    copy(pa->spec.biasprm, pa->biasprm, m->actuator_biasprm + mjNBIAS * i, mjNBIAS);

    // control ranges, one for each input: pid has its own for the velocity and feedforward
    // inputs, and the other inputs share ctrlrange. A range which is inherited from the target
    // is given in its place
    const bool integrates = pa->dyntype == mjDYN_INTEGRATOR;
    const int  nctrl      = std::min(pa->ctrlnum_, 4);
    const int  ctrladr    = m->actuator_ctrladr[i];
    int        velocity = -1, feedforward = -1;
    if (pa->gaintype == mjGAIN_PID) {
      int k = pa->ctrlspec_ & mjINPUT_POS ? 1 : 0;
      if (pa->ctrlspec_ & mjINPUT_VEL) { velocity = k++; }
      if (pa->ctrlspec_ & mjINPUT_FF) { feedforward = k; }
    }
    for (int k = 0; k < nctrl; k++) {
      const mjtNum* model = m->actuator_ctrlrange + 2 * (ctrladr + k);
      if (!Differs(pa->ctrlranges_[k], model, 2)) { continue; }
      if (k != velocity && k != feedforward) {
        for (int j = 0; j < nctrl; j++) {
          const mjtNum* other = m->actuator_ctrlrange + 2 * (ctrladr + j);
          if (j != velocity && j != feedforward && (other[0] != model[0] || other[1] != model[1])) {
            refuse(pa,
                   "the control range of one input of an actuator",
                   "its inputs have one control range");
          }
        }
      }
      if (!write) { continue; }
      Back(pa->ctrlranges_[k], model, 2);
      if (k == velocity) {
        copy(pa->spec.velrange, pa->velrange, model, 2);
      } else if (k == feedforward) {
        copy(pa->spec.ffrange, pa->ffrange, model, 2);
      } else {
        bool changed = copyrange(pa->spec.ctrlrange,
                                 pa->ctrlrange,
                                 model,
                                 pa->spec.ctrllimited,
                                 pa->ctrllimited,
                                 m->actuator_ctrllimited[ctrladr + k],
                                 pa->compiler);
        if (changed && tospec && !integrates) { pa->inheritrange = pa->spec.inheritrange = 0; }
      }
    }

    copyrange(pa->spec.forcerange,
              pa->forcerange,
              m->actuator_forcerange + 2 * i,
              pa->spec.forcelimited,
              pa->forcelimited,
              m->actuator_forcelimited[i],
              pa->compiler);
    if (copyrange(pa->spec.actrange,
                  pa->actrange,
                  m->actuator_actrange + 2 * i,
                  pa->spec.actlimited,
                  pa->actlimited,
                  m->actuator_actlimited[i],
                  pa->compiler) &&
        tospec &&
        write &&
        integrates) {
      pa->inheritrange = pa->spec.inheritrange = 0;
    }

    // length range and gear, which the outputs of the actuator share
    const int outadr = m->actuator_outadr[i];
    copy(pa->spec.lengthrange, pa->lengthrange, m->actuator_lengthrange + 2 * outadr, 2);
    copy(pa->spec.gear, pa->gear, m->actuator_gear + 6 * outadr, 6);

    copy(pa->spec.damping, pa->damping, m->actuator_damping + i);
    copy(pa->spec.damping + 1, pa->damping + 1, m->actuator_dampingpoly + mjNPOLY * i, mjNPOLY);
    copy(&pa->spec.armature, &pa->armature, m->actuator_armature + i);
    copy(&pa->spec.cranklength, &pa->cranklength, m->actuator_cranklength + i);
    copyvector(pa->spec_userdata_,
               pa->userdata_,
               m->actuator_user + nuser_actuator * i,
               nuser_actuator);
  }

  // sensors
  for (int i = 0; i < nsensor; i++) {
    mjCSensor* ps = sensors_[i];
    unscaling(static_cast<mjsSensor*>(ps), [ps] { ps->ApplyGauge(/*inverse=*/true); });

    copy(&ps->spec.cutoff, &ps->cutoff, m->sensor_cutoff + i);
    copy(&ps->spec.noise, &ps->noise, m->sensor_noise + i);
    copyvector(ps->spec_userdata_, ps->userdata_, m->sensor_user + nuser_sensor * i, nuser_sensor);
  }

  // numeric data
  for (int i = 0; i < nnumeric; i++) {
    mjCNumeric* pn = numerics_[i];
    copyvector(pn->spec_data_, pn->data_, m->numeric_data + m->numeric_adr[i], m->numeric_size[i]);
  }

  // tuple data
  for (int i = 0; i < ntuple; i++) {
    mjCTuple* pt = tuples_[i];
    copyvector(pt->spec_objprm_, pt->objprm_, m->tuple_objprm + m->tuple_adr[i], m->tuple_size[i]);
  }

  // keyframes, in the units of the spec
  for (int i = 0; i < m->nkey; i++) {
    mjCKey* pk = keys_[i];

    copy(&pk->spec.time, &pk->time, m->key_time + i);
    bool qpos = copyvector(pk->spec_qpos_, pk->qpos_, m->key_qpos + i * nq, nq);
    bool qvel = copyvector(pk->spec_qvel_, pk->qvel_, m->key_qvel + i * nv, nv);
    bool act  = copyvector(pk->spec_act_, pk->act_, m->key_act + i * na, na);
    bool mpos = copyvector(pk->spec_mpos_, pk->mpos_, m->key_mpos + i * 3 * nmocap, 3 * nmocap);
    copyvector(pk->spec_mquat_, pk->mquat_, m->key_mquat + i * 4 * nmocap, 4 * nmocap);
    bool ctrl = copyvector(pk->spec_ctrl_, pk->ctrl_, m->key_ctrl + i * nu, nu);
    bool plastic =
        copyvector(pk->spec_plastic_, pk->plastic_, m->key_plastic + i * nplastic, nplastic);
    if (tospec && write) {
      ScaleKeyframe(qpos ? pk->spec_qpos_.data() : nullptr,
                    qvel ? pk->spec_qvel_.data() : nullptr,
                    act ? pk->spec_act_.data() : nullptr,
                    ctrl ? pk->spec_ctrl_.data() : nullptr,
                    mpos ? pk->spec_mpos_.data() : nullptr,
                    plastic ? pk->spec_plastic_.data() : nullptr,
                    /*inverse=*/true);
    }
  }
}


void mjCModel::ActivatePlugin(const mjpPlugin* plugin, int slot) {
  bool already_declared = false;
  for (const auto& [existing_plugin, existing_slot] : active_plugins_) {
    if (plugin == existing_plugin) {
      already_declared = true;
      break;
    }
  }
  if (!already_declared) { active_plugins_.emplace_back(std::make_pair(plugin, slot)); }
}


void mjCModel::ResolvePlugin(mjCBase*           obj,
                             const std::string& plugin_name,
                             const std::string& plugin_instance_name,
                             mjCPlugin**        plugin_instance) {
  std::string pname = plugin_name;

  // if the plugin name is not specified by the user, infer it from the plugin instance
  if (plugin_name.empty() && !plugin_instance_name.empty()) {
    mjCBase* plugin_obj = FindObject(mjOBJ_PLUGIN, plugin_instance_name);
    if (plugin_obj) {
      pname = static_cast<mjCPlugin*>(plugin_obj)->plugin_name;
    } else {
      throw mjCError(obj,
                     "unrecognized name '%s' for plugin instance",
                     plugin_instance_name.c_str());
    }
  }

  // if plugin_name is specified, check if it is in the list of active plugins
  // (in XML, active plugins are those declared as <required>)
  int plugin_slot = -1;
  if (!pname.empty()) {
    for (int i = 0; i < active_plugins_.size(); ++i) {
      if (active_plugins_[i].first->name == pname) {
        plugin_slot = active_plugins_[i].second;
        break;
      }
    }
    if (plugin_slot == -1) { throw mjCError(obj, "unrecognized plugin '%s'", pname.c_str()); }
  }

  // meshes sharing a plugin instance resolve it concurrently on compiler worker threads
  static std::mutex           mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // implicit plugin instance
  if (*plugin_instance && (*plugin_instance)->plugin_slot == -1) {
    (*plugin_instance)->plugin_slot = plugin_slot;
    (*plugin_instance)->parent      = obj;
  }

  // explicit plugin instance, look up existing mjCPlugin by instance name
  else if (!*plugin_instance) {
    *plugin_instance = static_cast<mjCPlugin*>(FindObject(mjOBJ_PLUGIN, plugin_instance_name));
    if (!*plugin_instance) {
      throw mjCError(obj,
                     "unrecognized name '%s' for plugin instance",
                     plugin_instance_name.c_str());
    }

    // write only if changed: worker threads read plugin_slot after this call without the lock
    if ((*plugin_instance)->plugin_slot != plugin_slot) {
      (*plugin_instance)->plugin_slot = plugin_slot;
    }

    if (plugin_slot != -1 && plugin_slot != (*plugin_instance)->plugin_slot) {
      throw mjCError(obj, "'plugin' attribute does not match that of the instance");
    }
  }
}
