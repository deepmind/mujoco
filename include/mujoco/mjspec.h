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

#ifndef MUJOCO_INCLUDE_MJSPEC_H_
#define MUJOCO_INCLUDE_MJSPEC_H_

#include <stddef.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>


// this is a C-API
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#endif

//-------------------------------- handles to strings and arrays -----------------------------------

#ifdef __cplusplus
  // C++: defined to be compatible with corresponding std types
  using mjString      = std::string;
  using mjStringVec   = std::vector<std::string>;
  using mjIntVec      = std::vector<int>;
  using mjIntVecVec   = std::vector<std::vector<int>>;
  using mjFloatVec    = std::vector<float>;
  using mjFloatVecVec = std::vector<std::vector<float>>;
  using mjDoubleVec   = std::vector<double>;
  using mjByteVec     = std::vector<std::byte>;
#else
  // C: opaque types
  typedef void mjString;
  typedef void mjStringVec;
  typedef void mjIntVec;
  typedef void mjIntVecVec;
  typedef void mjFloatVec;
  typedef void mjFloatVecVec;
  typedef void mjDoubleVec;
  typedef void mjByteVec;
#endif


//-------------------------------- enum types (mjt) ------------------------------------------------

typedef enum mjtGeomInertia {      // type of inertia inference
  mjINERTIA_VOLUME = 0,            // mass distributed in the volume
  mjINERTIA_SHELL,                 // mass distributed on the surface
} mjtGeomInertia;


typedef enum mjtMeshInertia {      // type of mesh inertia
  mjMESH_INERTIA_CONVEX = 0,       // convex mesh inertia
  mjMESH_INERTIA_EXACT,            // exact mesh inertia
  mjMESH_INERTIA_LEGACY,           // legacy mesh inertia
  mjMESH_INERTIA_SHELL             // shell mesh inertia
} mjtMeshInertia;


typedef enum mjtMeshBuiltin {      // type of built-in procedural mesh
  mjMESH_BUILTIN_NONE = 0,         // no built-in mesh
  mjMESH_BUILTIN_SPHERE,           // sphere
  mjMESH_BUILTIN_HEMISPHERE,       // hemisphere
  mjMESH_BUILTIN_CONE,             // cone
  mjMESH_BUILTIN_SUPERSPHERE,      // supersphere
  mjMESH_BUILTIN_SUPERTORUS,       // supertorus
  mjMESH_BUILTIN_WEDGE,            // wedge
  mjMESH_BUILTIN_PLATE,            // plate
  mjMESH_BUILTIN_GRID,             // grid (2D or 3D)
  mjMESH_BUILTIN_BOX,              // box
  mjMESH_BUILTIN_CYLINDER,         // cylinder
  mjMESH_BUILTIN_ELLIPSOID,        // ellipsoid
  mjMESH_BUILTIN_SQUARE,           // square
  mjMESH_BUILTIN_DISC              // disc
} mjtMeshBuiltin;


typedef enum mjtFlexSection {      // cross-section of 1D flex elasticity
  mjFLEXSECTION_CIRCLE = 0,        // circle of the flex radius
  mjFLEXSECTION_BOX                // box with half-sizes sectionsize
} mjtFlexSection;


typedef enum mjtBuiltin {          // type of built-in procedural texture
  mjBUILTIN_NONE = 0,              // no built-in texture
  mjBUILTIN_GRADIENT,              // gradient: rgb1->rgb2
  mjBUILTIN_CHECKER,               // checker pattern: rgb1, rgb2
  mjBUILTIN_FLAT                   // 2d: rgb1; cube: rgb1-up, rgb2-side, rgb3-down
} mjtBuiltin;


typedef enum mjtMark {             // mark type for procedural textures
  mjMARK_NONE = 0,                 // no mark
  mjMARK_EDGE,                     // edges
  mjMARK_CROSS,                    // cross
  mjMARK_RANDOM                    // random dots
} mjtMark;


typedef enum mjtLimited {          // type of limit specification
  mjLIMITED_FALSE = 0,             // not limited
  mjLIMITED_TRUE,                  // limited
  mjLIMITED_AUTO,                  // limited inferred from presence of range
} mjtLimited;


typedef enum mjtAlignFree {        // whether to align free joints with the inertial frame
  mjALIGNFREE_FALSE = 0,           // don't align
  mjALIGNFREE_TRUE,                // align
  mjALIGNFREE_AUTO,                // respect the global compiler flag
} mjtAlignFree;


typedef enum mjtPassthrough {      // which assets the compiler leaves by reference (bitmask)
  mjPASSTHROUGH_NONE     = 0,      // all assets embedded
  mjPASSTHROUGH_TEXTURES = 1<<0,   // textures by reference; meshes reserved for a later bit
} mjtPassthrough;


typedef enum mjtEmbed {            // whether an asset's payload is in the model
  mjEMBED_FALSE = 0,               // by reference
  mjEMBED_TRUE,                    // embedded
  mjEMBED_AUTO,                    // respect the global compiler flag
} mjtEmbed;


typedef enum mjtInertiaFromGeom {  // whether to infer body inertias from child geoms
  mjINERTIAFROMGEOM_FALSE = 0,     // do not use; inertial element required
  mjINERTIAFROMGEOM_TRUE,          // always use; overwrite inertial element
  mjINERTIAFROMGEOM_AUTO           // use only if inertial element is missing
} mjtInertiaFromGeom;


typedef enum mjtOrientation {      // type of orientation specifier
  mjORIENTATION_QUAT = 0,          // quaternion
  mjORIENTATION_AXISANGLE,         // axis and angle
  mjORIENTATION_XYAXES,            // x and y axes
  mjORIENTATION_ZAXIS,             // z axis (minimal rotation)
  mjORIENTATION_EULER,             // Euler angles
} mjtOrientation;

typedef enum mjtConflict {         // conflict resolution for attach
  mjCONFLICT_WARNING = 0,          // keep parent, warn on conflict
  mjCONFLICT_MERGE,                // merge: min/max/error per field
  mjCONFLICT_ERROR,                // error on any conflict
} mjtConflict;

typedef enum mjtScaling {          // physical policy of a frame's scale
  mjSCALING_SIMILAR = 0,           // change of length and mass units: same motion in scaled gravity
  mjSCALING_GEOMETRY,              // shape and inertia only: forces and gains unchanged
} mjtScaling;

typedef enum mjtActuator {         // actuator element, which owns its shortcut parameters
  mjACTUATOR_GENERAL = 0,          // general
  mjACTUATOR_MOTOR,                // motor
  mjACTUATOR_POSITION,             // position servo
  mjACTUATOR_VELOCITY,             // velocity servo
  mjACTUATOR_INTVELOCITY,          // integrated-velocity servo
  mjACTUATOR_DAMPER,               // damper
  mjACTUATOR_CYLINDER,             // cylinder
  mjACTUATOR_MUSCLE,               // muscle
  mjACTUATOR_ADHESION,             // adhesion
  mjACTUATOR_PID,                  // pid servo
  mjACTUATOR_ORIENTATION,          // orientation servo
  mjACTUATOR_DCMOTOR               // dc motor
} mjtActuator;

typedef enum mjtCTimer {           // compiler timing categories
  // top-level timers (wall-clock)
  mjCTIMER_TOTAL = 0,              // total compile time
  mjCTIMER_ASSETS,                 // asset compilation

  // asset sub-timers (CPU time, summed across all assets)
  mjCTIMER_TEXTURE,                // textures
  mjCTIMER_MESH_LOAD,              // mesh: file loading
  mjCTIMER_MESH_HULL,              // mesh: convex hull
  mjCTIMER_MESH_POLYGON,           // mesh: normals and polygons
  mjCTIMER_MESH_INERTIA,           // mesh: volume, CoM, inertia
  mjCTIMER_MESH_BVH,               // mesh: bounding volume hierarchy
  mjCTIMER_MESH_OCTREE,            // mesh: octree and SDF

  mjNCTIMER                        // number of compiler timers
} mjtCTimer;


//-------------------------------- attribute structs (mjs) -----------------------------------------

typedef struct mjsElement_ {       // element type, do not modify
  mjtObj elemtype;                 // element type
  uint64_t signature;              // compilation signature
} mjsElement;


typedef struct mjsCompiler_ {      // compiler options
  mjtBool autolimits;              // infer "limited" attribute based on range
  double boundmass;                // enforce minimum body mass
  double boundinertia;             // enforce minimum body diagonal inertia
  double settotalmass;             // (deprecated) rescale masses and inertias; <=0: ignore
  mjtBool balanceinertia;          // automatically impose A + B >= C rule
  mjtBool fitaabb;                 // meshfit to aabb instead of inertia box
  mjtBool degree;                  // angles in radians or degrees
  char eulerseq[3];                // sequence for euler rotations
  mjtBool discardvisual;           // discard visual geoms in parser
  mjtBool usethread;               // use multiple threads to speed up compiler
  mjtBool fusestatic;              // fuse static bodies with parent
  mjtInertiaFromGeom inertiafromgeom; // use geom inertias
  int inertiagrouprange[2];        // range of geom groups used to compute inertia
  mjtBool saveinertial;            // save explicit inertial clause for all bodies to XML
  mjtBool savecompiled;            // save values as compiled, not as written in the spec
  mjtBool savecanonical;           // save quaternions and radians, not the notation of the spec
  mjtBool alignfree;               // align free joints with inertial frame
  mjtConflict conflict;            // conflict resolution for attach
  mjtPassthrough passthrough;      // assets left by reference (bitmask)
  mjLROpt LRopt;                   // options for lengthrange computation
  mjString* meshdir;               // mesh and hfield directory
  mjString* texturedir;            // texture directory
  uint64_t authored;               // bitmask of authored compiler fields
} mjsCompiler;


typedef struct mjsAuthored_ {      // authored tracking bitmasks for mjModel structs
  uint64_t option;                 // authored mjOption fields
  int      disableflags;           // individual authored disable flags
  int      enableflags;            // individual authored enable flags
  int      disableactuator;        // individual authored actuator groups
  uint64_t visual_global;          // authored visual.global fields
  uint64_t visual_quality;         // authored visual.quality fields
  uint64_t visual_headlight;       // authored visual.headlight fields
  uint64_t visual_map;             // authored visual.map fields
  uint64_t visual_scale;           // authored visual.scale fields
  uint64_t visual_rgba;            // authored visual.rgba fields
} mjsAuthored;


typedef struct mjSpec_ {           // model specification
  mjsElement* element;             // element type
  mjString* modelname;             // model name

  // compiler data
  mjsCompiler compiler;            // compiler options
  mjtBool strippath;               // automatically strip paths from mesh files

  // engine data
  mjOption option;                 // physics options
  mjVisual visual;                 // visual options
  mjStatistic stat;                // statistics override (if defined)

  // sizes
  mjtSize memory;                  // number of bytes in arena+stack memory
  int nemax;                       // max number of equality constraints
  int nuserdata;                   // number of mjtNums in userdata
  int nuser_body;                  // number of mjtNums in body_user
  int nuser_jnt;                   // number of mjtNums in jnt_user
  int nuser_geom;                  // number of mjtNums in geom_user
  int nuser_site;                  // number of mjtNums in site_user
  int nuser_cam;                   // number of mjtNums in cam_user
  int nuser_tendon;                // number of mjtNums in tendon_user
  int nuser_actuator;              // number of mjtNums in actuator_user
  int nuser_sensor;                // number of mjtNums in sensor_user
  int nkey;                        // number of keyframes
  int njmax;                       // (deprecated) max number of constraints
  int nconmax;                     // (deprecated) max number of detected contacts
  mjtSize nstack;                  // (deprecated) number of mjtNums in mjData stack

  // global data
  mjString* comment;               // comment at top of XML
  mjString* modelfiledir;          // path to model file

  // other
  mjtBool hasImplicitPluginElem;   // already encountered an implicit plugin sensor/actuator

  // authored tracking bitmasks for mjModel structs
  mjsAuthored authored;
} mjSpec;


typedef struct mjsOrientation_ {   // alternative orientation specifiers
  mjtOrientation type;             // active orientation specifier
  double axisangle[4];             // axis and angle
  double xyaxes[6];                // x and y axes
  double zaxis[3];                 // z axis (minimal rotation)
  double euler[3];                 // Euler angles
} mjsOrientation;


typedef struct mjsPlugin_ {        // plugin specification
  mjsElement* element;             // element type
  mjString* name;                  // instance name
  mjString* plugin_name;           // plugin name
  mjtBool active;                  // is the plugin active
  mjString* info;                  // message appended to compiler errors
} mjsPlugin;


typedef struct mjsBody_ {          // body specification
  mjsElement* element;             // element type
  mjString* childclass;            // childclass name

  // body frame
  double pos[3];                   // frame position
  double quat[4];                  // frame orientation
  mjsOrientation alt;              // frame alternative orientation

  // inertial frame
  double mass;                     // mass
  double ipos[3];                  // inertial frame position
  double iquat[4];                 // inertial frame orientation
  double inertia[3];               // diagonal inertia (in i-frame)
  mjsOrientation ialt;             // inertial frame alternative orientation
  double fullinertia[6];           // non-axis-aligned inertia matrix

  // other
  mjtBool mocap;                   // is this a mocap body
  double gravcomp;                 // gravity compensation
  mjtSleepPolicy sleep;            // sleep policy
  mjtByte simple;                  // simple body optimization (0: false, 1: auto)
  mjtByte fuse;                    // fuse with parent when static (0: false, 1: auto)
  mjDoubleVec* userdata;           // user data
  mjtBool explicitinertial;        // whether to save the body with explicit inertial clause
  mjsPlugin plugin;                // passive force plugin
  mjString* info;                  // message appended to compiler errors
} mjsBody;


typedef struct mjsFrame_ {         // frame specification
  mjsElement* element;             // element type
  mjString* childclass;            // childclass name
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;              // alternative orientation
  double scale[3];                 // scale of the frame's contents
  mjtScaling scaling;              // physical policy of the scale
  double massscale;                // mass factor of the contents; 0: ratio of volumes
  mjString* info;                  // message appended to compiler errors
} mjsFrame;


typedef struct mjsJoint_ {         // joint specification
  mjsElement* element;             // element type
  mjtJoint type;                   // joint type

  // kinematics
  double pos[3];                   // anchor position
  double axis[3];                  // joint axis
  double ref;                      // value at reference configuration: qpos0
  mjtAlignFree align;              // align free joint with body com

  // stiffness
  double stiffness[mjNPOLY+1];     // stiffness coefficients
  double springref;                // spring reference value: qpos_spring
  double springyield;              // plastic yield threshold (0: elastic)
  double springdamper[2];          // timeconst, dampratio

  // limits
  mjtLimited limited;              // does joint have limits
  double range[2];                 // joint limits
  double margin;                   // margin value for joint limit detection
  mjtNum solref_limit[mjNREF];     // solver reference: joint limits
  mjtNum solimp_limit[mjNIMP];     // solver impedance: joint limits
  mjtLimited actfrclimited;        // are actuator forces on joint limited
  double actfrcrange[2];           // actuator force limits

  // dof properties
  double armature;                 // armature inertia (mass for slider)
  double damping[mjNPOLY+1];       // damping coefficients
  double frictionloss;             // friction loss
  mjtNum solref_friction[mjNREF];  // solver reference: dof friction
  mjtNum solimp_friction[mjNIMP];  // solver impedance: dof friction

  // other
  int group;                       // group
  mjtBool actgravcomp;             // is gravcomp force applied via actuators
  mjDoubleVec* userdata;           // user data
  mjString* info;                  // message appended to compiler errors
} mjsJoint;


typedef struct mjsGeom_ {          // geom specification
  mjsElement* element;             // element type
  mjtGeom type;                    // geom type

  // frame, size
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;              // alternative orientation
  double fromto[6];                // alternative for capsule, cylinder, box, ellipsoid
  double size[3];                  // type-specific size

  // contact related
  int contype;                     // contact type
  int conaffinity;                 // contact affinity
  int condim;                      // contact dimensionality
  int priority;                    // contact priority
  double friction[3];              // one-sided friction coefficients: slide, roll, spin
  double solmix;                   // solver mixing for contact pairs
  mjtNum solref[mjNREF];           // solver reference
  mjtNum solimp[mjNIMP];           // solver impedance
  double margin;                   // margin for contact detection
  double gap;                      // additional contact detection buffer
  double surfacevel[6];            // surface velocity in local frame: linear, angular
  double adhesion;                 // adhesive force of contacts

  // inertia inference
  double mass;                     // used to compute density
  double density;                  // used to compute mass and inertia from volume or surface
  mjtGeomInertia typeinertia;      // selects between surface and volume inertia

  // fluid forces
  mjtNum fluid_ellipsoid;          // whether ellipsoid-fluid model is active
  mjtNum fluid_coefs[5];           // ellipsoid-fluid interaction coefs

  // visual
  mjString* material;              // name of material
  float rgba[4];                   // rgba when material is omitted
  int group;                       // group
  mjtBool castshadow;              // does geom cast shadows

  // other
  mjString* hfieldname;            // heightfield attached to geom
  mjString* meshname;              // mesh attached to geom
  double fitscale;                 // scale mesh uniformly
  mjDoubleVec* userdata;           // user data
  mjsPlugin plugin;                // sdf plugin
  mjString* info;                  // message appended to compiler errors
} mjsGeom;


typedef struct mjsSite_ {          // site specification
  mjsElement* element;             // element type

  // frame, size
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;              // alternative orientation
  double fromto[6];                // alternative for capsule, cylinder, box, ellipsoid
  double size[3];                  // geom size

  // visual
  mjtGeom type;                    // geom type
  mjString* material;              // name of material
  int group;                       // group
  float rgba[4];                   // rgba when material is omitted

  // other
  mjString* meshname;              // mesh attached to site
  mjDoubleVec* userdata;           // user data
  mjString* info;                  // message appended to compiler errors
} mjsSite;


typedef struct mjsCamera_ {        // camera specification
  mjsElement* element;             // element type

  // extrinsics
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;              // alternative orientation
  mjtCamLight mode;                // tracking mode
  mjString* targetbody;            // target body for tracking/targeting

  // intrinsics
  mjtProjection proj;              // camera projection type
  int resolution[2];               // resolution (pixel)
  int output;                      // bit flags for output type
  double fovy;                     // y-field of view
  double ipd;                      // inter-pupillary distance
  float intrinsic[4];              // camera intrinsics (length)
  float sensor_size[2];            // sensor size (length)
  float focal_length[2];           // focal length (length)
  float focal_pixel[2];            // focal length (pixel)
  float principal_length[2];       // principal point (length)
  float principal_pixel[2];        // principal point (pixel)

  // other
  mjDoubleVec* userdata;           // user data
  mjString* info;                  // message appended to compiler errors
} mjsCamera;


typedef struct mjsLight_ {         // light specification
  mjsElement* element;             // element type

  // frame
  double pos[3];                   // position
  double dir[3];                   // direction
  mjtCamLight mode;                // tracking mode
  mjString* targetbody;            // target body for targeting

  // intrinsics
  mjtBool active;                  // is light active
  mjtLightType type;               // type of light
  mjString* texture;               // texture name for image lights
  mjtBool castshadow;              // does light cast shadows
  float bulbradius;                // bulb radius, for soft shadows
  float intensity;                 // intensity, in candelas
  float range;                     // range of effectiveness
  float attenuation[3];            // OpenGL attenuation (quadratic model)
  float cutoff;                    // OpenGL cutoff
  float softness;                  // spotlight edge softness
  float exponent;                  // OpenGL exponent
  float ambient[3];                // ambient color
  float diffuse[3];                // diffuse color
  float specular[3];               // specular color

  // other
  mjString* info;                  // message appended to compiler errors
} mjsLight;


typedef struct mjsFlex_ {          // flex specification
  mjsElement* element;             // element type

  // contact properties
  int contype;                     // contact type
  int conaffinity;                 // contact affinity
  int condim;                      // contact dimensionality
  int priority;                    // contact priority
  double friction[3];              // one-sided friction coefficients: slide, roll, spin
  double solmix;                   // solver mixing for contact pairs
  mjtNum solref[mjNREF];           // solver reference
  mjtNum solimp[mjNIMP];           // solver impedance
  double margin;                   // margin for contact detection
  double gap;                      // additional contact detection buffer

  // other properties
  int dim;                         // element dimensionality
  double radius;                   // radius around primitive element
  double size[3];                  // vertex bounding box half sizes in qpos0
  mjtBool flatskin;                // render flex skin with flat shading
  mjtFlexSelf selfcollide;         // mode for flex self collision
  int passive;                     // mode for passive collisions
  int activelayers;                // number of active element layers in 3D
  int group;                       // group for visualization
  double edgestiffness;            // edge stiffness
  double edgedamping;              // edge damping
  float rgba[4];                   // rgba when material is omitted
  mjStringVec* material;           // name of material used for rendering
  double young;                    // Young's modulus
  double poisson;                  // Poisson's ratio
  double shear;                    // 1D torsion shear modulus; 0: young / (2 (1 + poisson))
  double damping;                  // Rayleigh's damping
  double thickness;                // thickness (2D only)
  double stressyield[2];           // yield strain thresholds (stretch, bend); 0: elastic
  int elastic2d;                   // 2D passive forces; 0: none, 1: bending, 2: stretching, 3: both
  int elastic3d;  // experimental 3D material (mjSpec only); 0: Saint
                  // Venant-Kirchhoff, 1: Stable Neo-Hookean
  mjtFlexSection section;          // 1D cross-section (mjtFlexSection)
  double sectionsize[2];           // 1D box cross-section half-sizes
  int cellcount[3];                // grid cell count for finite cell method
  int order;                       // interpolation order (1: trilinear, 2: quadratic)

  // mesh properties
  mjStringVec* nodebody;           // node body names
  mjStringVec* vertbody;           // vertex body names
  mjDoubleVec* node;               // node positions
  mjDoubleVec* vert;               // vertex positions
  mjIntVec* elem;                  // element vertex ids
  mjFloatVec* texcoord;            // vertex texture coordinates
  mjIntVec* elemtexcoord;          // element texture coordinates

  // other
  mjString* info;                  // message appended to compiler errors
} mjsFlex;


typedef struct mjsMesh_ {          // mesh specification
  mjsElement* element;             // element type
  mjString* content_type;          // content type of file
  mjString* file;                  // mesh file
  double refpos[3];                // reference position
  double refquat[4];               // reference orientation
  double scale[3];                 // rescale mesh
  mjtMeshInertia inertia;          // inertia type (convex, legacy, exact, shell)
  mjtBool smoothnormal;            // do not exclude large-angle faces from normals
  mjtBool needsdf;                 // compute sdf from mesh
  int maxhullvert;                 // maximum vertex count for the convex hull
  mjFloatVec* uservert;            // user vertex data
  mjFloatVec* usernormal;          // user normal data
  mjFloatVec* usertexcoord;        // user texcoord data
  mjIntVec* userface;              // user vertex indices
  mjIntVec* userfacenormal;        // user face normal indices
  mjIntVec* userfacetexcoord;      // user texcoord indices
  mjDoubleVec* usernode;           // user volume node coordinates
  mjIntVec* usertet;               // user tetrahedral cell indices
  mjsPlugin plugin;                // sdf plugin
  mjString* material;              // name of material
  int octree_maxdepth;             // max octree depth
  mjString* info;                  // message appended to compiler errors
} mjsMesh;


typedef struct mjsHField_ {        // height field specification
  mjsElement* element;             // element type
  mjString* content_type;          // content type of file
  mjString* file;                  // file: (nrow, ncol, [elevation data])
  double size[4];                  // hfield size (ignore referencing geom size)
  int nrow;                        // number of rows
  int ncol;                        // number of columns
  mjFloatVec* userdata;            // user-provided elevation data
  mjString* info;                  // message appended to compiler errors
} mjsHField;



typedef struct mjsSkin_ {          // skin specification
  mjsElement* element;             // element type
  mjString* file;                  // skin file
  mjString* material;              // name of material used for rendering
  float rgba[4];                   // rgba when material is omitted
  float inflate;                   // inflate in normal direction
  int group;                       // group for visualization

  // mesh
  mjFloatVec* vert;                // vertex positions
  mjFloatVec* texcoord;            // texture coordinates
  mjIntVec* face;                  // faces

  // skin
  mjStringVec* bodyname;           // body names
  mjFloatVec* bindpos;             // bind pos
  mjFloatVec* bindquat;            // bind quat
  mjIntVecVec* vertid;             // vertex ids
  mjFloatVecVec* vertweight;       // vertex weights

  // other
  mjString* info;                  // message appended to compiler errors
} mjsSkin;


typedef struct mjsTexture_ {       // texture specification
  mjsElement* element;             // element type
  mjtTexture type;                 // texture type
  mjtColorSpace colorspace;        // colorspace

  // method 1: builtin
  mjtBuiltin builtin;              // builtin type
  mjtMark mark;                    // mark type
  double rgb1[3];                  // first color for builtin
  double rgb2[3];                  // second color for builtin
  double markrgb[3];               // mark color
  double random;                   // probability of random dots
  int height;                      // height in pixels (square for cube and skybox)
  int width;                       // width in pixels
  int nchannel;                    // number of channels

  // method 2: single file
  mjString* content_type;          // content type of file
  mjString* file;                  // png file to load; use for all sides of cube
  int gridsize[2];                 // size of grid for composite file; (1,1)-repeat
  char gridlayout[12];             // row-major: L,R,F,B,U,D for faces; . for unused

  // method 3: separate files
  mjStringVec* cubefiles;          // different file for each side of the cube

  // method 4: from buffer read by user
  mjByteVec* data;                  // texture data

  // flip options
  mjtBool hflip;                   // horizontal flip
  mjtBool vflip;                   // vertical flip

  // payload
  mjtEmbed embed;                  // pixels in the model, or the file by reference

  // other
  mjString* info;                  // message appended to compiler errors
} mjsTexture;


typedef struct mjsMaterial_ {      // material specification
  mjsElement* element;             // element type
  mjStringVec* textures;           // names of textures (empty: none)
  mjtBool texuniform;              // make texture cube uniform
  float texrepeat[2];              // texture repetition for 2D mapping
  float emission;                  // emission
  float specular;                  // specular
  float shininess;                 // shininess
  float reflectance;               // reflectance
  float metallic;                  // metallic
  float roughness;                 // roughness
  float rgba[4];                   // rgba
  mjString* info;                  // message appended to compiler errors
} mjsMaterial;


typedef struct mjsPair_ {          // pair specification
  mjsElement* element;             // element type
  mjString* geomname1;             // name of geom 1
  mjString* geomname2;             // name of geom 2

  // optional parameters: computed from geoms if not set by user
  int condim;                      // contact dimensionality
  mjtNum solref[mjNREF];           // solver reference, normal direction
  mjtNum solreffriction[mjNREF];   // solver reference, frictional directions
  mjtNum solimp[mjNIMP];           // solver impedance
  double margin;                   // margin for contact detection
  double gap;                      // additional contact detection buffer
  double adhesion;                 // adhesive force of contacts
  double friction[5];              // full contact friction
  mjString* info;                  // message appended to errors
} mjsPair;


typedef struct mjsExclude_ {       // exclude specification
  mjsElement* element;             // element type
  mjString* bodyname1;             // name of geom 1
  mjString* bodyname2;             // name of geom 2
  mjString* info;                  // message appended to errors
} mjsExclude;


typedef struct mjsEquality_ {      // equality specification
  mjsElement* element;             // element type
  mjtEq type;                      // constraint type
  double data[mjNEQDATA];          // type-dependent data
  mjtBool active;                  // is equality initially active
  mjString* name1;                 // name of object 1
  mjString* name2;                 // name of object 2
  mjtObj objtype;                  // type of both objects
  mjtNum solref[mjNREF];           // solver reference
  mjtNum solimp[mjNIMP];           // solver impedance
  mjString* info;                  // message appended to errors
} mjsEquality;


typedef struct mjsTendon_ {        // tendon specification
  mjsElement* element;             // element type

  // stiffness, damping, friction, armature
  double stiffness[mjNPOLY+1];     // stiffness coefficients
  double springlength[2];          // spring resting length; {-1, -1}: use qpos_spring
  double springyield;              // plastic yield threshold (0: elastic)
  double damping[mjNPOLY+1];       // damping coefficients
  double frictionloss;             // friction loss
  mjtNum solref_friction[mjNREF];  // solver reference: tendon friction
  mjtNum solimp_friction[mjNIMP];  // solver impedance: tendon friction
  double armature;                 // inertia associated with tendon velocity

  // length range
  mjtLimited limited;              // does tendon have limits
  mjtLimited actfrclimited;        // does tendon have actuator force limits
  double range[2];                 // length limits
  double actfrcrange[2];           // actuator force limits
  double margin;                   // margin value for tendon limit detection
  mjtNum solref_limit[mjNREF];     // solver reference: tendon limits
  mjtNum solimp_limit[mjNIMP];     // solver impedance: tendon limits

  // visual
  mjString* material;              // name of material for rendering
  double width;                    // width for rendering
  float rgba[4];                   // rgba when material is omitted
  int group;                       // group

  // other
  mjDoubleVec* userdata;           // user data
  mjString* info;                  // message appended to errors
} mjsTendon;


typedef struct mjsWrap_ {          // wrapping object specification
  mjsElement* element;             // element type
  mjtWrap type;                    // wrap type
  mjString* info;                  // message appended to errors
} mjsWrap;


typedef struct mjsActuator_ {      // actuator specification
  mjsElement* element;             // element type
  mjtActuator type;                // element the actuator is written with

  // gain, bias
  mjtGain gaintype;                // gain type
  double gainprm[mjNGAIN];         // gain parameters
  mjtBias biastype;                // bias type
  double biasprm[mjNBIAS];         // bias parameters

  // activation state
  mjtDyn dyntype;                  // dynamics type
  double dynprm[mjNDYN];           // dynamics parameters
  int actdim;                      // number of activation variables
  int ctrlspec;                    // input signature, scoped by gaintype; 0: type default
  double velrange[2];              // range of the velocity-setpoint input (pid)
  double ffrange[2];               // range of the feedforward input (pid)
  mjtBool actearly;                // apply next activations to qfrc

  // transmission
  mjtTrn trntype;                  // transmission type
  double gear[6];                  // length and transmitted force scaling
  mjString* target;                // name of transmission target
  mjString* refsite;               // reference site, for site transmission
  mjString* slidersite;            // site defining cylinder, for slider-crank
  double cranklength;              // crank length, for slider-crank
  double lengthrange[2];           // transmission length range
  double inheritrange;             // automatic range setting for position and intvelocity
  double damping[mjNPOLY+1];       // damping coefficients
  double armature;                 // armature inertia

  // input/output clamping
  mjtLimited ctrllimited;          // are control limits defined
  double ctrlrange[2];             // control range
  mjtLimited forcelimited;         // are force limits defined
  double forcerange[2];            // force range
  mjtLimited actlimited;           // are activation limits defined
  double actrange[2];              // activation range

  // other
  int group;                       // group
  int nsample;                     // number of samples in history buffer
  int interp;                      // interpolation order (0=ZOH, 1=linear, 2=cubic)
  double delay;                    // delay time in seconds; 0: no delay
  mjDoubleVec* userdata;           // user data
  mjsPlugin plugin;                // actuator plugin
  mjString* info;                  // message appended to compiler errors
} mjsActuator;


typedef struct mjsSensor_ {        // sensor specification
  mjsElement* element;             // element type

  // sensor definition
  mjtSensor type;                  // type of sensor
  mjtObj objtype;                  // type of sensorized object
  mjString* objname;               // name of sensorized object
  mjtObj reftype;                  // type of referenced object
  mjString* refname;               // name of referenced object
  int intprm[mjNSENS];             // integer parameters

  // user-defined sensors
  mjtDataType datatype;            // data type for sensor measurement
  mjtStage needstage;              // compute stage needed to simulate sensor
  int dim;                         // number of scalar outputs

  // output post-processing
  double cutoff;                   // cutoff for real and positive datatypes
  double noise;                    // noise stdev

  // history buffer
  int nsample;                     // number of samples in history buffer
  int interp;                      // interpolation order (0=ZOH, 1=linear, 2=cubic)
  double delay;                    // delay time in seconds
  double interval[2];              // [period, time_prev] in seconds

  // other
  mjDoubleVec* userdata;           // user data
  mjsPlugin plugin;                // sensor plugin
  mjString* info;                  // message appended to compiler errors
} mjsSensor;


typedef struct mjsNumeric_ {       // custom numeric field specification
  mjsElement* element;             // element type
  mjDoubleVec* data;               // initialization data
  int size;                        // array size, can be bigger than data size
  mjString* info;                  // message appended to compiler errors
} mjsNumeric;


typedef struct mjsText_ {          // custom text specification
  mjsElement* element;             // element type
  mjString* data;                  // text string
  mjString* info;                  // message appended to compiler errors
} mjsText;


typedef struct mjsTuple_ {         // tuple specification
  mjsElement* element;             // element type
  mjIntVec* objtype;               // object types
  mjStringVec* objname;            // object names
  mjDoubleVec* objprm;             // object parameters
  mjString* info;                  // message appended to compiler errors
} mjsTuple;


typedef struct mjsKey_ {           // keyframe specification
  mjsElement* element;             // element type
  double time;                     // time
  mjDoubleVec* qpos;               // qpos
  mjDoubleVec* qvel;               // qvel
  mjDoubleVec* act;                // act
  mjDoubleVec* mpos;               // mocap pos
  mjDoubleVec* mquat;              // mocap quat
  mjDoubleVec* ctrl;               // ctrl
  mjDoubleVec* plastic;            // plastic
  mjString* info;                  // message appended to compiler errors
} mjsKey;


typedef struct mjsDefault_ {       // default specification
  mjsElement* element;             // element type
  mjsJoint* joint;                 // joint defaults
  mjsGeom* geom;                   // geom defaults
  mjsSite* site;                   // site defaults
  mjsCamera* camera;               // camera defaults
  mjsLight* light;                 // light defaults
  mjsFlex* flex;                   // flex defaults
  mjsMesh* mesh;                   // mesh defaults
  mjsTexture* texture;             // texture defaults
  mjsMaterial* material;           // material defaults
  mjsPair* pair;                   // pair defaults
  mjsEquality* equality;           // equality defaults
  mjsTendon* tendon;               // tendon defaults
  mjsActuator* actuator;           // actuator defaults
} mjsDefault;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_INCLUDE_MJSPEC_H_
