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

#ifndef MUJOCO_SRC_USER_USER_MODEL_H_
#define MUJOCO_SRC_USER_USER_MODEL_H_

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include "user/user_objects.h"

typedef std::map<std::string, int, std::less<>>                    mjKeyMap;
typedef std::array<mjKeyMap, mjNOBJECT>                            mjListKeyMap;
typedef std::array<std::unordered_set<std::string>, mjNOBJECT + 1> mjNameSet;

class mjCModel_ : public mjsElement {
 public:
  // attach namespaces
  std::string prefix;
  std::string suffix;

 protected:
  bool compiled;  // already compiled flag

  // the keyframes were stored in the elements since the last compilation, because the tree
  // changed: a vector which a keyframe has now was given to it for the tree as it is now
  bool keysstored = false;

  // sizes set from object list lengths
  mjtSize nbody;     // number of bodies
  mjtSize njnt;      // number of joints
  mjtSize ngeom;     // number of geoms
  mjtSize nsite;     // number of sites
  mjtSize ncam;      // number of cameras
  mjtSize nlight;    // number of lights
  mjtSize nflex;     // number of flexes
  mjtSize nmesh;     // number of meshes
  mjtSize nskin;     // number of skins
  mjtSize nhfield;   // number of height fields
  mjtSize ntex;      // number of textures
  mjtSize nmat;      // number of materials
  mjtSize npair;     // number of geom pairs in pair array
  mjtSize nexclude;  // number of excluded body pairs
  mjtSize neq;       // number of equality constraints
  mjtSize ntendon;   // number of tendons
  mjtSize nJten;     // number of non-zeros in sparse ten_J matrix
  mjtSize nsensor;   // number of sensors
  mjtSize nnumeric;  // number of numeric fields
  mjtSize ntext;     // number of text fields
  mjtSize ntuple;    // number of tuple fields
  mjtSize nmocap;    // number of mocap bodies
  mjtSize nplugin;   // number of plugin instances

  // sizes computed by Compile
  mjtSize nq;              // number of generalized coordinates = dim(qpos)
  mjtSize nv;              // number of degrees of freedom = dim(qvel)
  mjtSize nu;              // number of scalar controls = dim(ctrl)
  mjtSize nactuator;       // number of actuators
  mjtSize nout;            // number of force outputs = dim(actuator_force)
  mjtSize na;              // number of activation variables
  mjtSize ntree;           // number of trees
  mjtSize nbvh;            // number of total boundary volume hierarchies
  mjtSize nbvhstatic;      // number of static boundary volume hierarchies
  mjtSize nbvhdynamic;     // number of dynamic boundary volume hierarchies
  mjtSize noct;            // number of total octree cells
  mjtSize nflexnode;       // number of nodes in all flexes
  mjtSize nflexvert;       // number of vertices in all flexes
  mjtSize nflexedge;       // number of edges in all flexes
  mjtSize nflexelem;       // number of elements in all flexes
  mjtSize nflexelemdata;   // number of element vertex ids in all flexes
  mjtSize nflexstiffness;  // number of stiffness parameters in all flexes
  mjtSize nflexbending;    // number of bending parameters in all flexes
  mjtSize nefm0dof;        // number of dofs covered by the bending factor
  mjtSize nefm0L;          // number of non-zeros in the bending factor
  mjtSize nefmCvert;       // number of sparse metric factor ordering slots
  mjtSize nflexelemedge;   // number of element edges in all flexes
  mjtSize nflexshelldata;  // number of shell fragment vertex ids in all flexes
  mjtSize nflextexcoord;   // number of vertex texture coordinates in all flexes
  mjtSize nJfe;            // number of non-zeros in sparse flex edge constraint Jacobian
  mjtSize nmeshvert;       // number of vertices in all meshes
  mjtSize nmeshnormal;     // number of normals in all meshes
  mjtSize nmeshtexcoord;   // number of texture coordinates in all meshes
  mjtSize nmeshface;       // number of triangular faces in all meshes
  mjtSize nmeshpoly;       // number of polygon faces in all meshes
  mjtSize nmeshgraph;      // number of ints in mesh auxiliary data
  mjtSize nmeshpolyvert;   // number of vertices in all polygon faces
  mjtSize nmeshpolymap;    // number of polygons in vertex map
  mjtSize nskinvert;       // number of vertices in all skins
  mjtSize nskintexvert;    // number of vertices with texcoord in all skins
  mjtSize nskinface;       // number of faces in all skins
  mjtSize nskinbone;       // number of bones in all skins
  mjtSize nskinbonevert;   // number of vertices in all skins
  mjtSize nhfielddata;     // number of data points in all hfields
  mjtSize ntexdata;        // number of texture bytes
  mjtSize nwrap;           // number of wrap objects in all tendon paths
  mjtSize nsensordata;     // number of mjtNums in sensor data vector
  mjtSize nhistory;        // number of mjtNums in history buffer
  mjtSize nplastic;        // number of mjtNums in plastic state vector
  mjtSize nnumericdata;    // number of mjtNums in all custom fields
  mjtSize ntextdata;       // number of chars in all text fields, including 0
  mjtSize ntupledata;      // number of objects in all tuple fields
  mjtSize npluginattr;     // number of chars in all plugin config attributes
  mjtSize nnames;          // number of chars in all names
  mjtSize npaths;          // number of chars in all paths
  mjtSize nM;              // number of non-zeros in sparse inertia matrix
  mjtSize nB;              // number of non-zeros in sparse body-dof matrix
  mjtSize nC;              // number of non-zeros in reduced sparse dof-dof matrix
  mjtSize nD;              // number of non-zeros in sparse dof-dof matrix
  mjtSize nJmom;           // number of non-zeros in sparse actuator_moment matrix

  // statistics, as computed by mj_setConst
  double meaninertia_auto;  // mean diagonal inertia, as computed by mj_setConst
  double meanmass_auto;     // mean body mass, as computed by mj_setConst
  double meansize_auto;     // mean body size, as computed by mj_setConst
  double extent_auto;       // spatial extent, as computed by mj_setConst
  double center_auto[3];    // center of model, as computed by mj_setConst

  // save qpos0 and plastic0, to recognize changed key_qpos and key_plastic in write
  std::vector<mjtNum> qpos0;
  std::vector<mjtNum> body_pos0;
  std::vector<mjtNum> body_quat0;
  std::vector<mjtNum> plastic0;

  // variable-size attributes
  std::string comment_;       // comment at top of XML
  std::string modelfiledir_;  // path to model file
  std::string modelname_;
  std::string meshdir_;
  std::string texturedir_;
  std::string spec_comment_;
  std::string spec_modelfiledir_;
  std::string spec_modelname_;
};

// mjCModel contains everything needed to generate the low-level model.
// It can be constructed manually by calling 'Add' functions and setting
// the public fields of the various objects.  Alternatively it can constructed
// by loading an XML file via mjCXML.  Once an mjCModel object is
// constructed, 'Compile' can be called to generate the corresponding mjModel object
// (which is the low-level model).  The mjCModel object can then be deleted.
class mjCModel : public mjCModel_, private mjSpec {
  friend class mjCBase;
  friend class mjCBody;
  friend class mjCCamera;
  friend class mjCGeom;
  friend class mjCFlex;
  friend class mjCHField;
  friend class mjCFrame;
  friend class mjCJoint;
  friend class mjCEquality;
  friend class mjCMesh;
  friend class mjCSkin;
  friend class mjCSite;
  friend class mjCTendon;
  friend class mjCTexture;
  friend class mjCActuator;
  friend class mjCSensor;
  friend class mjCDef;
  friend class mjXReader;
  friend class mjXWriter;

 public:
  mjCModel();
  mjCModel(const mjCModel& other);
  ~mjCModel();
  void CopyFromSpec();  // copy spec to private attributes
  void PointToLocal();

  mjCModel& operator=(const mjCModel& other);    // copy other into this, if they are not the same
  mjCModel& operator+=(const mjCModel& other);   // add other into this, even if they are the same
  mjCModel& operator-=(const mjCBody& subtree);  // remove subtree and all references from model
  mjCModel& operator-=(const mjCFrame& frame);   // remove frame, its contents and all references
  mjCModel& operator+=(mjCDef& subtree);         // add default tree to this model
  mjCModel& operator-=(const mjCDef& subtree);   // remove default tree from this model

  mjSpec spec;
  double timer[mjNCTIMER] = {0};  // compiler timers

  mjModel* Compile(const mjVFS* vfs = nullptr, mjModel** m = nullptr);  // construct mjModel

  // compile the assets and the kinematic tree of the spec as it is now, without constructing an
  // mjModel, so that what compilation derives from them can be read; if it fails, save the
  // error and return false
  bool Resolve(const mjVFS* vfs = nullptr, bool textures = true);
  bool CopyBack(const mjModel*);         // DECOMPILER: copy numeric back
  int  FuseStatic(const mjVFS* vfs);     // fuse static bodies with parent, return number fused
  int  DiscardVisual(const mjVFS* vfs);  // discard visual elements, return number discarded
  int  DedupAssets();                    // merge duplicate assets, return number merged

  // API for adding model elements
  mjCFlex*     AddFlex();
  mjCMesh*     AddMesh(mjCDef* def = nullptr);
  mjCSkin*     AddSkin();
  mjCHField*   AddHField();
  mjCTexture*  AddTexture(mjCDef* def = nullptr);
  mjCMaterial* AddMaterial(mjCDef* def = nullptr);
  mjCPair*     AddPair(mjCDef* def = nullptr);      // geom pair for inclusion
  mjCBodyPair* AddExclude();                        // body pair for exclusion
  mjCEquality* AddEquality(mjCDef* def = nullptr);  // equality constraint
  mjCTendon*   AddTendon(mjCDef* def = nullptr);
  mjCActuator* AddActuator(mjCDef* def = nullptr);
  mjCSensor*   AddSensor();
  mjCNumeric*  AddNumeric();
  mjCText*     AddText();
  mjCTuple*    AddTuple();
  mjCKey*      AddKey();
  mjCPlugin*   AddPlugin();

  // append spec to this model, optionally map compiler options to the appended spec
  void AppendSpec(mjSpec* spec, const mjsCompiler* compiler = nullptr);

  // delete object from the corresponding list
  void operator-=(mjsElement* el);

  // delete default and all descendants
  void RemoveDefault(mjCDef* def);

  // API for access to model elements (outside tree)
  int         NumObjects(mjtObj type);         // number of objects in specified list
  mjCBase*    GetObject(mjtObj type, int id);  // pointer to specified object
  mjsElement* NextObject(const mjsElement* object,
                         mjtObj type = mjOBJ_UNKNOWN) const;  // next object of specified type

  // API for access to other variables
  bool            IsCompiled() const;  // is model already compiled
  const mjCError& GetError() const;    // get reference of error object
  void            SetError(const mjCError& error) { errInfo = error; }  // set value of error object
  void            AddWarning(std::string    msg,                        // add warning to vector
                             const mjCBase* obj = nullptr);
  void            AddGroupedWarning(const std::string& subject,  // add grouped warning
                                    const std::string& body);
  const std::vector<std::string>& GetWarnings() const {  // get accumulated warnings
    return warnings_;
  }
  void ClearWarnings() {
    warnings_.clear();
    num_attach_warnings_ = 0;
  }  // clear all warnings
  void ClearCompileWarnings() { warnings_.resize(num_attach_warnings_); }  // clear compile warnings
  void SetAttachWarningBoundary() {  // snapshot attach warning count
    num_attach_warnings_ = warnings_.size();
  }

  mjCBody* GetWorld();                                              // pointer to world body
  mjCDef*  FindDefault(const std::string& name) const;              // find defaults class name
  mjCDef*  AddDefault(std::string name, mjCDef* parent = nullptr);  // add defaults class to array
  mjCBase* FindObject(mjtObj type, std::string name) const;  // find object given type and name
  mjCBase* FindTree(mjCBody* body, mjtObj type, std::string name);  // find tree object given name
  mjSpec*  FindSpec(std::string name) const;                        // find spec given name
  mjSpec*  FindSpec(const mjsCompiler* compiler_) const;            // find spec given mjsCompiler
  void     ActivatePlugin(const mjpPlugin* plugin, int slot);       // activate plugin

  // find object given type and name by searching the list, without the name maps
  mjCBase* SearchObject(mjtObj type, std::string_view name) const;

  // accessors
  std::string get_meshdir() const { return meshdir_; }
  std::string get_texturedir() const { return texturedir_; }

  mjCDef* Default() const { return defaults_[0]; }
  int     NumDefaults() const { return defaults_.size(); }

  const std::vector<std::pair<const mjpPlugin*, int>>& ActivePlugins() const {
    return active_plugins_;
  };

  const std::vector<mjCFlex*>&   Flexes() const { return flexes_; }
  const std::vector<mjCMesh*>&   Meshes() const { return meshes_; }
  const std::vector<mjCSkin*>&   Skins() const { return skins_; }
  const std::vector<mjCHField*>& HFields() const { return hfields_; }

  // the meshes and height fields as compiled: those of the spec, then the instances which the
  // compilation made of those used at several scales
  std::vector<mjCMesh*>   CompiledMeshes() const;
  std::vector<mjCHField*> CompiledHFields() const;
  int NumCompiledMeshes() const { return meshes_.size() + meshinstances_.size(); }

  const std::vector<mjCTexture*>&  Textures() const { return textures_; }
  const std::vector<mjCMaterial*>& Materials() const { return materials_; }
  const std::vector<mjCPair*>&     Pairs() const { return pairs_; }
  const std::vector<mjCBodyPair*>& Excludes() const { return excludes_; }
  const std::vector<mjCEquality*>& Equalities() const { return equalities_; }
  const std::vector<mjCTendon*>&   Tendons() const { return tendons_; }
  const std::vector<mjCActuator*>& Actuators() const { return actuators_; }
  const std::vector<mjCSensor*>&   Sensors() const { return sensors_; }
  const std::vector<mjCNumeric*>&  Numerics() const { return numerics_; }
  const std::vector<mjCText*>&     Texts() const { return texts_; }
  const std::vector<mjCTuple*>&    Tuples() const { return tuples_; }
  const std::vector<mjCKey*>&      Keys() const { return keys_; }
  const std::vector<mjCPlugin*>&   Plugins() const { return plugins_; }
  const std::vector<mjCBody*>&     Bodies() const { return bodies_; }
  const std::vector<mjCGeom*>&     Geoms() const { return geoms_; }

  // resolve plugin instance, create a new one if needed
  void ResolvePlugin(mjCBase*           obj,
                     const std::string& plugin_name,
                     const std::string& plugin_instance_name,
                     mjCPlugin**        plugin_instance);

  // clear objects allocated by Compile
  void Clear();

  // temporary state saved across mj_recompile
  struct mjRecompileState {
    mjtNum              time = 0;
    std::vector<mjtNum> userdata;

    std::unordered_map<mjCJoint*, std::vector<mjtNum>>    qfrc_applied;
    std::unordered_map<mjCJoint*, std::vector<mjtNum>>    qacc_warmstart;
    std::unordered_map<mjCJoint*, std::vector<mjtNum>>    jnt_plastic;
    std::unordered_map<mjCTendon*, std::array<mjtNum, 2>> tendon_plastic;
    std::unordered_map<mjCFlex*, std::vector<mjtNum>>     flex_plastic;
    std::unordered_map<mjCBody*, std::array<mjtNum, 6>>   xfrc_applied;
    std::unordered_map<mjCEquality*, mjtByte>             eq_active;
    std::unordered_map<mjCActuator*, std::vector<mjtNum>> actuator_history;
    std::unordered_map<mjCSensor*, std::vector<mjtNum>>   sensor_history;
    std::unordered_map<mjCPlugin*, std::vector<mjtNum>>   plugin_state;
  };

  // save the current state
  template <class T>
  void SaveState(const std::string& state_name,
                 const T*           qpos,
                 const T*           qvel,
                 const T*           act,
                 const T*           ctrl,
                 const T*           mpos,
                 const T*           mquat,
                 const T*           plastic,
                 bool               partial = false);
  void SaveState(const std::string& state_name,
                 const mjModel*     m,
                 const mjData*      d,
                 mjRecompileState*  state);

  // restore the previously saved state
  template <class T>
  void RestoreState(const std::string& state_name,
                    const mjtNum*      pos0,
                    const mjtNum*      mpos0,
                    const mjtNum*      mquat0,
                    T*                 qpos,
                    T*                 qvel,
                    T*                 act,
                    T*                 ctrl,
                    T*                 mpos,
                    T*                 mquat,
                    T*                 plastic);
  void RestoreState(const std::string&      state_name,
                    const mjModel*          m,
                    mjData*                 d,
                    const mjRecompileState* state);

  // clear existing data
  void MakeData(const mjModel* m, mjData** dest);

  // store the values of the keyframes in the elements they belong to, ahead of a change to the
  // tree: a deletion (dest is this model, which has no namespace), the attachment of this model
  // to dest, or the attachment of another model to this one (dest is null)
  void StoreKeyframes(mjCModel* dest);

  // map from default class name to default class pointer
  std::unordered_map<std::string, mjCDef*> def_map;

  // set deepcopy flag
  void SetDeepCopy(bool deepcopy) { deepcopy_ = deepcopy; }

  // get deepcopy flag
  bool GetDeepCopy() const { return deepcopy_; }

  // set attached flag
  void SetAttached(bool deepcopy) { attached_ |= !deepcopy; }

  // check if model is attached
  bool IsAttached() const { return attached_; }

  // check if a keyframe awaits the next compilation, after a change to the tree
  bool HasPendingKeys() const;

  // forget the state saved under a name
  void ForgetState(const std::string& state_name);

  // copy the state saved under a name to another name
  void CopyState(const std::string& state_name, const std::string& copy_name);

  // check for repeated names in list
  void CheckRepeat(mjtObj type);

  // check that newname is not used by another element of the same type and update names_
  void CheckNameChange(mjtObj type, const std::string& oldname, const std::string& newname);

  // clear the compilation signature after a structural change
  void InvalidateSignature() { spec.element->signature = 0; }

  // the model was compiled and its structure changed since
  bool StructureChanged() const { return compiled && spec.element->signature == 0; }

  // increment and decrement reference count
  void AddRef() { ++refcount; }
  int  GetRef() const { return refcount; }
  void Release() {
    if (--refcount == 0) { delete this; }
  }

 private:
  int refcount = 1;

  // settings for each defaults class
  std::vector<mjCDef*> defaults_;

  // list of active plugins
  std::vector<std::pair<const mjpPlugin*, int>> active_plugins_;

  // make lists of bodies and children
  void MakeTreeLists(mjCBody* body = nullptr);

  // compile phases
  mjModel* Compile(const mjVFS* vfs, mjModel** m, bool treeonly, bool textures);
  void     TryCompile(mjModel*& m, mjData*& d, const mjVFS* vfs);
  void     CompileTree(const mjVFS* vfs, bool textures, bool keyframes);  // assets, kinematic tree
  void     CompileMeshesAndTextures(const mjVFS* vfs, bool textures = true);

  void SetNuser();                      // set nuser fields
  void IndexAssets();                   // convert asset names into indices
  void CheckEmptyNames();               // check empty names
  void SetSizes();                      // compute sizes
  void ComputeSparseSizes();            // compute nM, nD, nB, nC
  void AutoSpringDamper(mjModel*);      // automatic stiffness and damping computation
  void LengthRange(mjModel*, mjData*);  // compute actuator lengthrange
  void CopyNames(mjModel*);             // copy names, compute name addresses
  void CopyPaths(mjModel*);             // copy paths, compute path addresses
  void CopyObjects(mjModel*);           // copy objects outside kinematic tree
  void CopyTree(mjModel*);              // copy objects inside kinematic tree
  void FinalizeSimple(mjModel* m);      // finalize simple bodies/dofs including tendon information
  void CopyPlugins(mjModel*);           // copy plugin data
  int  CountTendonDofs(const mjModel* m,  // compute number of dofs for a given tendon
                       int            id);
  int  CountNJmom(const mjModel* m);  // compute number of non-zeros in actuator_moment matrix
  int  CountNJten(const mjModel* m);  // compute number of non-zeros in ten_J matrix

  // return the plugin instances that elements or defaults reference by name or point to, including
  // the elements in the subtree of body, which can be outside the tree
  std::unordered_set<const mjsElement*> ReferencedPlugins(const mjCBody* body = nullptr);

  // delete the referenced plugin instances that elements and defaults no longer reference
  void RemovePlugins(const std::unordered_set<const mjsElement*>& referenced);

  // give the geoms and sites which use a mesh or a height field the instance of the asset at the
  // scale of their use, made again unless the assets are reused
  void MakeInstances(bool reuse);

  // instances which compilation makes of the meshes and height fields used at several scales
  std::vector<mjCMesh*>   meshinstances_;
  std::vector<mjCHField*> hfieldinstances_;

  // objects created here
  std::vector<mjCFlex*>     flexes_;      // list of flexes
  std::vector<mjCMesh*>     meshes_;      // list of meshes
  std::vector<mjCSkin*>     skins_;       // list of skins
  std::vector<mjCHField*>   hfields_;     // list of height fields
  std::vector<mjCTexture*>  textures_;    // list of textures
  std::vector<mjCMaterial*> materials_;   // list of materials
  std::vector<mjCPair*>     pairs_;       // list of geom pairs to include
  std::vector<mjCBodyPair*> excludes_;    // list of body pairs to exclude
  std::vector<mjCEquality*> equalities_;  // list of equality constraints
  std::vector<mjCTendon*>   tendons_;     // list of tendons
  std::vector<mjCActuator*> actuators_;   // list of actuators
  std::vector<mjCSensor*>   sensors_;     // list of sensors
  std::vector<mjCNumeric*>  numerics_;    // list of numeric fields
  std::vector<mjCText*>     texts_;       // list of text fields
  std::vector<mjCTuple*>    tuples_;      // list of tuple fields
  std::vector<mjCKey*>      keys_;        // list of keyframe fields
  std::vector<mjCPlugin*>   plugins_;     // list of plugin instances
  std::vector<mjSpec*>      specs_;       // list of attached specs

  // pointers to objects created inside kinematic tree
  std::vector<mjCBody*>   bodies_;   // list of bodies
  std::vector<mjCJoint*>  joints_;   // list of joints allowing motion relative to parent
  std::vector<mjCGeom*>   geoms_;    // list of geoms attached to this body
  std::vector<mjCSite*>   sites_;    // list of sites attached to this body
  std::vector<mjCCamera*> cameras_;  // list of cameras
  std::vector<mjCLight*>  lights_;   // list of lights
  std::vector<mjCFrame*>  frames_;   // list of frames

  // array of pointers to each object list (enumerated by type)
  std::array<std::vector<mjCBase*>*, mjNOBJECT> object_lists_;

  // add object of any type
  template <class T>
  T* AddObject(std::vector<T*>& list, std::string type);

  // add object of any type, with defaults parameter
  template <class T>
  T* AddObjectDefault(std::vector<T*>& list, std::string type, mjCDef* def);

  // copy vector of elements of another model to this model
  template <class T>
  void CopyList(std::vector<T*>& dest, const std::vector<T*>& sources, const mjCModel& other);

  // copy plugins that are explicitly instantiated by the argument object to this model
  template <class T>
  void CopyExplicitPlugin(T* obj);

  // copy vector of plugins to this model
  template <class T>
  void CopyPlugin(const std::vector<mjCPlugin*>& sources, const std::vector<T*>& list);

  // give a copy of the model what the compilation of the original gave to it
  void CopyCompiled(const mjCModel& other);

  // give the copy of an element what the compilation gave to the original, which copying an
  // element resets; for a body, also to the bodies and joints of its subtree
  void CopyCompiled(mjCBase* dest, const mjCBase* source) {}
  void CopyCompiled(mjCBody* dest, const mjCBody* source);
  void CopyCompiled(mjCFlex* dest, const mjCFlex* source);
  void CopyCompiled(mjCTendon* dest, const mjCTendon* source);
  void CopyCompiled(mjCEquality* dest, const mjCEquality* source);
  void CopyCompiled(mjCActuator* dest, const mjCActuator* source);
  void CopyCompiled(mjCSensor* dest, const mjCSensor* source);
  void CopyCompiled(mjCPlugin* dest, const mjCPlugin* source);

  // delete from list the elements that cause an error
  template <class T>
  void RemoveFromList(std::vector<T*>& list, const mjCModel& other);

  // remove subtree from the tree, then remove all elements that reference it
  template <class T>
  mjCModel& RemoveSubtree(const T& subtree);

  // remove body or frame from the tree, return the elements removed along with it that are not
  // released with it: the elements inside a frame
  std::vector<mjCBase*> RemoveFromTree(const mjCBody& subtree);
  std::vector<mjCBase*> RemoveFromTree(const mjCFrame& frame);

  // return the body that owns the frame, nullptr if the frame is not in the tree
  mjCBody* FrameOwner(const mjCFrame& frame, mjCBody* body = nullptr);

  // create mjCBase lists from children lists
  void CreateObjectLists();

  // populate objects ids
  void ProcessLists(bool checkrepeat = true);

  // process list of objects
  template <class T>
  void ProcessList_(mjListKeyMap& ids, std::vector<T*>& list, mjtObj type, bool checkrepeat = true);

  // reset lists of kinematic tree
  void ResetTreeLists();

  // save dof offsets in joints and actuators
  void SaveDofOffsets(bool computesize = false);

  // convert pending keyframes info to actual keyframes
  void ResolveKeyframes(const mjModel* m);

  // add a keyframe which awaits the next compilation, after all other keyframes
  mjCKey* AddPendingKey(const std::string& name, const mjKeyInfo& info);

  // expand a keyframe, filling in missing values
  void ExpandKeyframe(mjCKey* key, const mjtNum* qpos0_, const mjtNum* bpos, const mjtNum* bquat);

  // scale the vectors of a keyframe from the units of the spec to the compiled model, or back
  template <class T>
  void ScaleKeyframe(T* qpos, T* qvel, T* act, T* ctrl, T* mpos, T* plastic, bool inverse) const;

  // compute qpos0
  void ComputeReference();

  // return true if body has valid mass and inertia
  bool CheckBodyMassInertia(mjCBody* body);

  // copy to the elements the values of the model which differ from those they hold, and with
  // `tospec` to the spec as well; without `write`, only report those which the spec cannot express
  void BackValues(const mjModel* m, bool tospec, bool write);

  // print the tree of a body
  void PrintTree(std::stringstream& tree, const mjCBody* body, int depth = 0);

  // generate a signature for the model
  uint64_t Signature();

  // true if fusing the body would break a reference to it
  bool IsReferenced(mjCBody* body);

  // resolve references in a list of objects
  template <class T>
  void ResolveReferences(std::vector<T*>& list, mjCBody* body = nullptr);

  mjListKeyMap             ids;        // map from object names to ids
  mjNameSet                names_;     // names in use per element type
  mjCError                 errInfo;    // last error info
  std::vector<std::string> warnings_;  // chronological list of non-fatal warnings
  int  num_attach_warnings_ = 0;       // boundary: [0, n) are attach, [n, size) are compile
  bool compiling_           = false;   // true during Compile()

  // a compilation which applies operations to the spec compiles the assets once
  bool                     reuse_assets_      = false;     // true during such a compilation
  bool                     assets_compiled_   = false;     // the meshes were compiled in it
  bool                     textures_compiled_ = false;     // and the textures
  std::string              asset_warnings_;                // the warnings which this gave
  double                   asset_timer_[mjNCTIMER] = {0};  // and the time it took
  bool                     deepcopy_;                      // copy objects when attaching
  bool                     copying_  = false;  // true while this model is copied from another
  bool                     attached_ = false;  // true if model is attached to a parent model
  bool                     baseline_ = false;  // the elements hold what compilation gave the model
  std::vector<std::string> inplacekeys_;       // stored names of the keyframes which stay in place
  std::unordered_map<const mjsCompiler*, mjSpec*> compiler2spec_;  // map from compiler to spec
};
#endif  // MUJOCO_SRC_USER_USER_MODEL_H_
