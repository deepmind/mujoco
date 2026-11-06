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

#ifndef MUJOCO_MJMODEL_H_
#define MUJOCO_MJMODEL_H_

#include <stddef.h>
#include <stdint.h>

#include <mujoco/mjtype.h>

// global constants
#define mjMINMU         1E-5      // minimum friction coefficient
#define mjMINIMP        0.0001    // minimum constraint impedance
#define mjMAXIMP        0.9999    // maximum constraint impedance
#define mjMAXCONPAIR    50        // maximum number of contacts per geom pair
#define mjMAXTREEDEPTH  50        // maximum bounding volume hierarchy depth
#define mjMAXFLEXNODES  27        // maximum number of flex nodes
#define mjMINAWAKE      10        // minimum number of timesteps before sleeping


//---------------------------------- sizes ---------------------------------------------------------

#define mjNEQDATA       12        // number of eq_data fields
#define mjNDYN          10        // number of actuator dynamics parameters
#define mjNGAIN         10        // number of actuator gain parameters
#define mjNBIAS         10        // number of actuator bias parameters
#define mjNFLUID        12        // number of fluid interaction parameters
#define mjNREF          2         // number of solver reference parameters
#define mjNIMP          5         // number of solver impedance parameters
#define mjNPOLY         2         // number of high-order polynomial coefficients
#define mjNSENS         3         // number of sensor parameters
#define mjNSOLVER       200       // size of one mjData.solver array
#define mjNISLAND       20        // number of mjData.solver arrays



//---------------------------------- mjLROpt -------------------------------------------------------

typedef struct mjLROpt_ {         // options for mj_setLengthRange()
  // flags
  int mode;                       // which actuators to process (mjtLRMode)
  int useexisting;                // use existing length range if available
  int uselimit;                   // use joint and tendon limits if available

  // algorithm parameters
  mjtNum accel;                   // target acceleration used to compute force
  mjtNum maxforce;                // maximum force; 0: no limit
  mjtNum timeconst;               // time constant for velocity reduction; min 0.01
  mjtNum timestep;                // simulation timestep; 0: use mjOption.timestep
  mjtNum inttotal;                // total simulation time interval
  mjtNum interval;                // evaluation time interval (at the end)
  mjtNum tolrange;                // convergence tolerance (relative to range)
} mjLROpt;

//---------------------------------- mjCache -------------------------------------------------------

typedef struct mjCache_ {         // asset cache used by the compiler
  void* impl_;                    // internal pointer to cache
} mjCache;

//---------------------------------- mjVFS ---------------------------------------------------------

typedef struct mjVFS_ {           // virtual file system for loading from memory
  void* impl_;                    // internal pointer to VFS memory
} mjVFS;

//---------------------------------- mjOption ------------------------------------------------------

typedef struct mjOption_ {        // physics options
  // timing parameters
  mjtNum timestep;                // timestep

  // solver parameters
  mjtNum impratio;                // ratio of friction-to-normal contact impedance
  mjtNum tolerance;               // main solver tolerance
  mjtNum ls_tolerance;            // CG/Newton linesearch tolerance
  mjtNum noslip_tolerance;        // noslip solver tolerance
  mjtNum ccd_tolerance;           // convex collision solver tolerance

  // sleep settings
  mjtNum sleep_tolerance;         // sleep velocity tolerance

  // physical constants
  mjtNum gravity[3];              // gravitational acceleration
  mjtNum wind[3];                 // wind (for lift, drag and viscosity)
  mjtNum magnetic[3];             // global magnetic flux
  mjtNum density;                 // density of medium
  mjtNum viscosity;               // viscosity of medium

  // override contact solver parameters (if enabled)
  mjtNum o_margin;                // margin
  mjtNum o_solref[mjNREF];        // solref
  mjtNum o_solimp[mjNIMP];        // solimp
  mjtNum o_friction[5];           // friction

  // discrete settings
  int integrator;                 // integration mode (mjtIntegrator)
  int cone;                       // type of friction cone (mjtCone)
  int jacobian;                   // type of Jacobian (mjtJacobian)
  int solver;                     // solver algorithm (mjtSolver)
  int iterations;                 // maximum number of main solver iterations
  int ls_iterations;              // maximum number of CG/Newton linesearch iterations
  int noslip_iterations;          // maximum number of noslip solver iterations
  int ccd_iterations;             // maximum number of convex collision solver iterations
  int disableflags;               // bit flags for disabling standard features
  int enableflags;                // bit flags for enabling optional features
  int disableactuator;            // bit flags for disabling actuators by group id

  // sdf collision settings
  int sdf_initpoints;             // number of starting points for gradient descent
  int sdf_iterations;             // max number of iterations for gradient descent
} mjOption;


//---------------------------------- mjVisual ------------------------------------------------------

typedef struct mjVisual_ {        // visualization options
  struct {                        // global parameters
    int   cameraid;               // initial camera id (-1: free)
    int   orthographic;           // is the free camera orthographic (0: no, 1: yes)
    float fovy;                   // y field-of-view of free camera (orthographic ? length : degree)
    float ipd;                    // inter-pupilary distance for free camera
    float azimuth;                // initial azimuth of free camera (degrees)
    float elevation;              // initial elevation of free camera (degrees)
    float linewidth;              // line width for wireframe and ray rendering
    float glow;                   // glow coefficient for selected body
    float realtime;               // initial real-time factor (1: real time)
    int   offwidth;               // width of offscreen buffer
    int   offheight;              // height of offscreen buffer
    int   ellipsoidinertia;       // geom for inertia visualization (0: box, 1: ellipsoid)
    int   bvactive;               // visualize active bounding volumes (0: no, 1: yes)
  } global;

  struct {                        // rendering quality
    int   shadowsize;             // size of shadowmap texture
    int   offsamples;             // number of multisamples for offscreen rendering
    int   numslices;              // number of slices for builtin geom drawing
    int   numstacks;              // number of stacks for builtin geom drawing
    int   numquads;               // number of quads for box rendering
  } quality;

  struct {                        // head light
    float ambient[3];             // ambient rgb (alpha=1)
    float diffuse[3];             // diffuse rgb (alpha=1)
    float specular[3];            // specular rgb (alpha=1)
    int   active;                 // is headlight active (0: no, 1: yes, 2: auto)
  } headlight;

  struct {                        // mapping
    float stiffness;              // mouse perturbation stiffness (space->force)
    float stiffnessrot;           // mouse perturbation stiffness (space->torque)
    float force;                  // from force units to space units
    float torque;                 // from torque units to space units
    float alpha;                  // scale geom alphas when transparency is enabled
    float fogstart;               // OpenGL fog starts at fogstart * mjModel.stat.extent
    float fogend;                 // OpenGL fog ends at fogend * mjModel.stat.extent
    float znear;                  // near clipping plane = znear * mjModel.stat.extent
    float zfar;                   // far clipping plane = zfar * mjModel.stat.extent
    float haze;                   // haze ratio
    float shadowclip;             // directional light: shadowclip * mjModel.stat.extent
    float shadowscale;            // spot light: shadowscale * light.cutoff
    float actuatortendon;         // scale tendon width
  } map;

  struct {                        // scale of decor elements relative to mean body size
    float forcewidth;             // width of force arrow
    float contactwidth;           // contact width
    float contactheight;          // contact height
    float connect;                // autoconnect capsule width
    float com;                    // com radius
    float camera;                 // camera object
    float light;                  // light object
    float selectpoint;            // selection point
    float jointlength;            // joint length
    float jointwidth;             // joint width
    float actuatorlength;         // actuator length
    float actuatorwidth;          // actuator width
    float framelength;            // bodyframe axis length
    float framewidth;             // bodyframe axis width
    float constraint;             // constraint width
    float slidercrank;            // slidercrank width
    float frustum;                // frustum zfar plane
  } scale;

  struct {                        // color of decor elements
    float fog[4];                 // fog
    float haze[4];                // haze
    float force[4];               // external force
    float inertia[4];             // inertia box
    float joint[4];               // joint
    float actuator[4];            // actuator, neutral
    float actuatornegative[4];    // actuator, negative limit
    float actuatorpositive[4];    // actuator, positive limit
    float com[4];                 // center of mass
    float camera[4];              // camera object
    float light[4];               // light object
    float selectpoint[4];         // selection point
    float connect[4];             // auto connect
    float contactpoint[4];        // contact point
    float contactforce[4];        // contact force
    float contactfriction[4];     // contact friction force
    float contacttorque[4];       // contact torque
    float contactgap[4];          // contact point in gap
    float rangefinder[4];         // rangefinder ray
    float constraint[4];          // constraint
    float slidercrank[4];         // slidercrank
    float crankbroken[4];         // used when crank must be stretched/broken
    float frustum[4];             // camera frustum
    float bv[4];                  // bounding volume
    float bvactive[4];            // active bounding volume
  } rgba;
} mjVisual;


//---------------------------------- mjStatistic ---------------------------------------------------

typedef struct mjStatistic_ {     // model statistics (in qpos0)
  mjtNum meaninertia;             // mean diagonal inertia
  mjtNum meanmass;                // mean body mass
  mjtNum meansize;                // mean body size
  mjtNum extent;                  // spatial extent
  mjtNum center[3];               // center of model
} mjStatistic;


//---------------------------------- mjModel -------------------------------------------------------

typedef struct mjModel_ {
  // ------------------------------- sizes

  // sizes needed at mjModel construction
  mjtSize nq;                     // number of generalized coordinates = dim(qpos)
  mjtSize nv;                     // number of degrees of freedom = dim(qvel)
  mjtSize nu;                     // number of scalar controls = dim(ctrl)
  mjtSize nactuator;              // number of actuators
  mjtSize nout;                   // number of force outputs, derived from transmission type
  mjtSize na;                     // number of activation states = dim(act)
  mjtSize nbody;                  // number of bodies
  mjtSize nbvh;                   // number of total bounding volumes in all bodies
  mjtSize nbvhstatic;             // number of static bounding volumes (aabb stored in mjModel)
  mjtSize nbvhdynamic;            // number of dynamic bounding volumes (aabb stored in mjData)
  mjtSize noct;                   // number of total octree cells in all meshes
  mjtSize njnt;                   // number of joints
  mjtSize ntree;                  // number of kinematic trees under world body
  mjtSize nM;                     // number of non-zeros in sparse inertia matrix
  mjtSize nB;                     // number of non-zeros in sparse body-dof matrix
  mjtSize nC;                     // number of non-zeros in sparse reduced dof-dof matrix
  mjtSize nD;                     // number of non-zeros in sparse dof-dof matrix
  mjtSize ngeom;                  // number of geoms
  mjtSize nsite;                  // number of sites
  mjtSize ncam;                   // number of cameras
  mjtSize nlight;                 // number of lights
  mjtSize nflex;                  // number of flexes
  mjtSize nflexnode;              // number of dofs in all flexes
  mjtSize nflexvert;              // number of vertices in all flexes
  mjtSize nflexedge;              // number of edges in all flexes
  mjtSize nflexelem;              // number of elements in all flexes
  mjtSize nflexelemdata;          // number of element vertex ids in all flexes
  mjtSize nflexstiffness;         // number of stiffness parameters in all flexes
  mjtSize nflexbending;           // number of bending parameters in all flexes
  mjtSize nefm0dof;               // number of dofs covered by the constant metric factor
  mjtSize nefm0L;                 // number of non-zeros in the constant metric factor
  mjtSize nefmCvert;              // number of sparse metric factor ordering slots
  mjtSize nflexelemedge;          // number of element edge ids in all flexes
  mjtSize nflexshelldata;         // number of shell fragment vertex ids in all flexes
  mjtSize nflextexcoord;          // number of vertices with texture coordinates
  mjtSize nJfe;                   // number of non-zeros in sparse flexedge Jacobian matrix
  mjtSize nmesh;                  // number of meshes
  mjtSize nmeshvert;              // number of vertices in all meshes
  mjtSize nmeshnormal;            // number of normals in all meshes
  mjtSize nmeshtexcoord;          // number of texcoords in all meshes
  mjtSize nmeshface;              // number of triangular faces in all meshes
  mjtSize nmeshgraph;             // number of ints in mesh auxiliary data
  mjtSize nmeshpoly;              // number of polygons in all meshes
  mjtSize nmeshpolyvert;          // number of vertices in all polygons
  mjtSize nmeshpolymap;           // number of polygons in vertex map
  mjtSize nskin;                  // number of skins
  mjtSize nskinvert;              // number of vertices in all skins
  mjtSize nskintexvert;           // number of vertices with texcoords in all skins
  mjtSize nskinface;              // number of triangular faces in all skins
  mjtSize nskinbone;              // number of bones in all skins
  mjtSize nskinbonevert;          // number of vertices in all skin bones
  mjtSize nhfield;                // number of heightfields
  mjtSize nhfielddata;            // number of data points in all heightfields
  mjtSize ntex;                   // number of textures
  mjtSize ntexdata;               // number of bytes in texture rgb data
  mjtSize nmat;                   // number of materials
  mjtSize npair;                  // number of predefined geom pairs
  mjtSize nexclude;               // number of excluded geom pairs
  mjtSize neq;                    // number of equality constraints
  mjtSize ntendon;                // number of tendons
  mjtSize nJten;                  // number of non-zeros in sparse ten_J matrix
  mjtSize nwrap;                  // number of wrap objects in all tendon paths
  mjtSize nsensor;                // number of sensors
  mjtSize nnumeric;               // number of numeric custom fields
  mjtSize nnumericdata;           // number of mjtNums in all numeric fields
  mjtSize ntext;                  // number of text custom fields
  mjtSize ntextdata;              // number of mjtBytes in all text fields
  mjtSize ntuple;                 // number of tuple custom fields
  mjtSize ntupledata;             // number of objects in all tuple fields
  mjtSize nkey;                   // number of keyframes
  mjtSize nmocap;                 // number of mocap bodies
  mjtSize nplugin;                // number of plugin instances
  mjtSize npluginattr;            // number of chars in all plugin config attributes
  mjtSize nuser_body;             // number of mjtNums in body_user
  mjtSize nuser_jnt;              // number of mjtNums in jnt_user
  mjtSize nuser_geom;             // number of mjtNums in geom_user
  mjtSize nuser_site;             // number of mjtNums in site_user
  mjtSize nuser_cam;              // number of mjtNums in cam_user
  mjtSize nuser_tendon;           // number of mjtNums in tendon_user
  mjtSize nuser_actuator;         // number of mjtNums in actuator_user
  mjtSize nuser_sensor;           // number of mjtNums in sensor_user
  mjtSize nnames;                 // number of chars in all names
  mjtSize npaths;                 // number of chars in all paths
  mjtSize nplastic;               // number of mjtNums in plastic state vector

  // sizes set after mjModel construction
  mjtSize nnames_map;             // number of slots in the names hash map
  mjtSize nJmom;                  // number of non-zeros in sparse actuator_moment matrix
  mjtSize ngravcomp;              // number of bodies with nonzero gravcomp
  mjtSize nemax;                  // number of potential equality-constraint rows
  mjtSize njmax;                  // number of available rows in constraint Jacobian (legacy)
  mjtSize nconmax;                // number of potential contacts in contact list (legacy)
  mjtSize npolygonmax;            // maximum number of vertices in a mesh polygon
  mjtSize nmeshdegmax;            // maximum number of edges adjacent to a mesh vertex
  mjtSize nuserdata;              // number of mjtNums reserved for the user
  mjtSize nsensordata;            // number of mjtNums in sensor data vector
  mjtSize npluginstate;           // number of mjtNums in plugin state vector
  mjtSize nhistory;               // number of mjtNums in history buffer

  // buffer sizes
  mjtSize narena;                 // number of bytes in the mjData arena (inclusive of stack)
  mjtSize nbuffer;                // number of bytes in buffer

  // ------------------------------- flags

  mjtBool flg_gravcomp;           // whether any body has nonzero gravcomp
  mjtBool flg_surfacevel;         // whether any geom has nonzero surfacevel
  mjtBool flg_adhesion;           // whether any geom or pair has nonzero adhesion

  // ------------------------------- options and statistics

  mjOption opt;                   // physics options
  mjVisual vis;                   // visualization options
  mjStatistic stat;               // model statistics

  // ------------------------------- buffers

  // main buffer
  void*     buffer;               // main buffer; all pointers point in it    (nbuffer)

  // default generalized coordinates
  mjtNum*   qpos0;                // qpos values at default pose              (nq x 1)
  mjtNum*   qpos_spring;          // reference pose for springs               (nq x 1)

  // bodies
  int*      body_parentid;        // id of body's parent                      (nbody x 1)
  int*      body_rootid;          // ancestor that is direct child of world   (nbody x 1)
  int*      body_weldid;          // top dof-less ancestor; mocap: own root   (nbody x 1)
  int*      body_mocapid;         // id of mocap data; -1: none               (nbody x 1)
  int*      body_jntnum;          // number of joints for this body           (nbody x 1)
  int*      body_jntadr;          // start addr of joints; -1: no joints      (nbody x 1)
  int*      body_dofnum;          // number of motion degrees of freedom      (nbody x 1)
  int*      body_dofadr;          // start addr of dofs; -1: no dofs          (nbody x 1)
  int*      body_treeid;          // id of body's kinematic tree; -1: static  (nbody x 1)
  int*      body_geomnum;         // number of geoms                          (nbody x 1)
  int*      body_geomadr;         // start addr of geoms; -1: no geoms        (nbody x 1)
  mjtByte*  body_simple;          // 1: diag M; 2: diag M, sliders only       (nbody x 1)
  mjtByte*  body_sameframe;       // same frame as inertia (mjtSameframe)     (nbody x 1)
  mjtNum*   body_pos;             // position offset rel. to parent body      (nbody x 3)
  mjtNum*   body_quat;            // orientation offset rel. to parent body   (nbody x 4)
  mjtNum*   body_ipos;            // local position of center of mass         (nbody x 3)
  mjtNum*   body_iquat;           // local orientation of inertia ellipsoid   (nbody x 4)
  mjtNum*   body_mass;            // mass                                     (nbody x 1)
  mjtNum*   body_subtreemass;     // mass of subtree starting at this body    (nbody x 1)
  mjtNum*   body_inertia;         // diagonal inertia in ipos/iquat frame     (nbody x 3)
  mjtNum*   body_invweight0;      // mean inv inert in qpos0 (trn, rot)       (nbody x 2)
  mjtNum*   body_gravcomp;        // antigravity force, units of body weight  (nbody x 1)
  mjtNum*   body_margin;          // MAX over all geom margins+gaps           (nbody x 1)
  mjtNum*   body_user;            // user data                                (nbody x nuser_body)
  int*      body_plugin;          // plugin instance id; -1: not in use       (nbody x 1)
  int*      body_contype;         // OR over all geom contypes                (nbody x 1)
  int*      body_conaffinity;     // OR over all geom conaffinities           (nbody x 1)
  int*      body_bvhadr;          // address of bvh root                      (nbody x 1)
  int*      body_bvhnum;          // number of bounding volumes               (nbody x 1)

  // bounding volume hierarchy
  int*      bvh_depth;            // depth in the bounding volume hierarchy   (nbvh x 1)
  int*      bvh_child;            // left and right children in tree          (nbvh x 2)
  int*      bvh_nodeid;           // geom or elem id of node; -1: non-leaf    (nbvh x 1)
  mjtNum*   bvh_aabb;             // local bounding box (center, size)        (nbvhstatic x 6)

  // octree spatial partitioning
  int*      oct_depth;            // depth in the octree                      (noct x 1)
  int*      oct_child;            // children of octree node                  (noct x 8)
  mjtNum*   oct_aabb;             // octree node bounding box (center, size)  (noct x 6)
  mjtNum*   oct_coeff;            // octree interpolation coefficients        (noct x 8)

  // joints
  int*      jnt_type;             // type of joint (mjtJoint)                 (njnt x 1)
  int*      jnt_qposadr;          // start addr in 'qpos' for joint's data    (njnt x 1)
  int*      jnt_dofadr;           // start addr in 'qvel' for joint's data    (njnt x 1)
  int*      jnt_bodyid;           // id of joint's body                       (njnt x 1)
  int*      jnt_actuatorid;       // actuator contributing damping / armature (njnt x 1)
  int*      jnt_group;            // group for visibility                     (njnt x 1)
  int*      jnt_plasticadr;       // start addr in plastic state (-1: none)   (njnt x 1)
  mjtBool*  jnt_limited;          // does joint have limits                   (njnt x 1)
  mjtBool*  jnt_actfrclimited;    // does joint have actuator force limits    (njnt x 1)
  mjtBool*  jnt_actgravcomp;      // is gravcomp force applied via actuators  (njnt x 1)
  mjtNum*   jnt_solref;           // constraint solver reference: limit       (njnt x mjNREF)
  mjtNum*   jnt_solimp;           // constraint solver impedance: limit       (njnt x mjNIMP)
  mjtNum*   jnt_pos;              // local anchor position                    (njnt x 3)
  mjtNum*   jnt_axis;             // local joint axis                         (njnt x 3)
  mjtNum*   jnt_stiffness;        // linear stiffness coefficient             (njnt x 1)
  mjtNum*   jnt_stiffnesspoly;    // high-order stiffness coefficients        (njnt x mjNPOLY)
  mjtNum*   jnt_springyield;      // plastic yield threshold                  (njnt x 1)
  mjtNum*   jnt_range;            // joint limits                             (njnt x 2)
  mjtNum*   jnt_actfrcrange;      // range of total actuator force            (njnt x 2)
  mjtNum*   jnt_margin;           // min distance for limit detection         (njnt x 1)
  mjtNum*   jnt_user;             // user data                                (njnt x nuser_jnt)

  // dofs
  int*      dof_bodyid;           // id of dof's body                         (nv x 1)
  int*      dof_jntid;            // id of dof's joint                        (nv x 1)
  int*      dof_parentid;         // id of dof's parent; -1: none             (nv x 1)
  int*      dof_treeid;           // id of dof's kinematic tree               (nv x 1)
  int*      dof_Madr;             // dof address in M-diagonal                (nv x 1)
  int*      dof_simplenum;        // number of consecutive simple dofs        (nv x 1)
  mjtNum*   dof_solref;           // constraint solver reference:frictionloss (nv x mjNREF)
  mjtNum*   dof_solimp;           // constraint solver impedance:frictionloss (nv x mjNIMP)
  mjtNum*   dof_frictionloss;     // dof friction loss                        (nv x 1)
  mjtNum*   dof_armature;         // dof armature inertia/mass                (nv x 1)
  mjtNum*   dof_damping;          // linear damping coefficient               (nv x 1)
  mjtNum*   dof_dampingpoly;      // high-order damping coefficients          (nv x mjNPOLY)
  mjtNum*   dof_invweight0;       // diag. inverse inertia in qpos0           (nv x 1)
  mjtNum*   dof_M0;               // diag. inertia in qpos0                   (nv x 1)
  mjtNum*   dof_length;           // linear: 1; angular: approx. length scale (nv x 1)

  // trees
  int*      tree_bodyadr;         // start addr of bodies                     (ntree x 1)
  int*      tree_bodynum;         // number of bodies in tree                 (ntree x 1)
  int*      tree_dofadr;          // start addr of dofs                       (ntree x 1)
  int*      tree_dofnum;          // number of dofs in tree                   (ntree x 1)
  int*      tree_sleep_policy;    // sleep policy (mjtSleepPolicy)            (ntree x 1)

  // geoms
  int*      geom_type;            // geometric type (mjtGeom)                 (ngeom x 1)
  int*      geom_contype;         // geom contact type                        (ngeom x 1)
  int*      geom_conaffinity;     // geom contact affinity                    (ngeom x 1)
  int*      geom_condim;          // contact dimensionality (1, 3, 4, 6)      (ngeom x 1)
  int*      geom_bodyid;          // id of geom's body                        (ngeom x 1)
  int*      geom_dataid;          // id of geom's mesh/hfield; -1: none       (ngeom x 1)
  int*      geom_matid;           // material id for rendering; -1: none      (ngeom x 1)
  int*      geom_group;           // group for visibility                     (ngeom x 1)
  int*      geom_priority;        // geom contact priority                    (ngeom x 1)
  int*      geom_plugin;          // plugin instance id; -1: not in use       (ngeom x 1)
  mjtByte*  geom_sameframe;       // same frame as body (mjtSameframe)        (ngeom x 1)
  mjtBool*  geom_castshadow;      // does geom cast shadows                   (ngeom x 1)
  mjtNum*   geom_solmix;          // mixing coef for solref/imp in geom pair  (ngeom x 1)
  mjtNum*   geom_solref;          // constraint solver reference: contact     (ngeom x mjNREF)
  mjtNum*   geom_solimp;          // constraint solver impedance: contact     (ngeom x mjNIMP)
  mjtNum*   geom_size;            // geom-specific size parameters            (ngeom x 3)
  mjtNum*   geom_aabb;            // bounding box, (center, size)             (ngeom x 6)
  mjtNum*   geom_rbound;          // radius of bounding sphere                (ngeom x 1)
  mjtNum*   geom_pos;             // local position offset rel. to body       (ngeom x 3)
  mjtNum*   geom_quat;            // local orientation offset rel. to body    (ngeom x 4)
  mjtNum*   geom_friction;        // friction for (slide, spin, roll)         (ngeom x 3)
  mjtNum*   geom_margin;          // geometric inflation for contact          (ngeom x 1)
  mjtNum*   geom_gap;             // additional contact detection buffer      (ngeom x 1)
  mjtNum*   geom_surfacevel;      // surface velocity in local frame: lin,ang (ngeom x 6)
  mjtNum*   geom_adhesion;        // adhesive force of contacts               (ngeom x 1)
  mjtNum*   geom_fluid;           // fluid interaction parameters             (ngeom x mjNFLUID)
  mjtNum*   geom_user;            // user data                                (ngeom x nuser_geom)
  float*    geom_rgba;            // rgba when material is omitted            (ngeom x 4)

  // sites
  int*      site_type;            // geom type for rendering (mjtGeom)        (nsite x 1)
  int*      site_bodyid;          // id of site's body                        (nsite x 1)
  int*      site_dataid;          // id of site's mesh; -1: none              (nsite x 1)
  int*      site_matid;           // material id for rendering; -1: none      (nsite x 1)
  int*      site_group;           // group for visibility                     (nsite x 1)
  mjtByte*  site_sameframe;       // same frame as body (mjtSameframe)        (nsite x 1)
  mjtNum*   site_size;            // geom size for rendering                  (nsite x 3)
  mjtNum*   site_pos;             // local position offset rel. to body       (nsite x 3)
  mjtNum*   site_quat;            // local orientation offset rel. to body    (nsite x 4)
  mjtNum*   site_user;            // user data                                (nsite x nuser_site)
  float*    site_rgba;            // rgba when material is omitted            (nsite x 4)

  // cameras
  int*      cam_mode;             // camera tracking mode (mjtCamLight)       (ncam x 1)
  int*      cam_bodyid;           // id of camera's body                      (ncam x 1)
  int*      cam_targetbodyid;     // id of targeted body; -1: none            (ncam x 1)
  mjtNum*   cam_pos;              // position rel. to body frame              (ncam x 3)
  mjtNum*   cam_quat;             // orientation rel. to body frame           (ncam x 4)
  mjtNum*   cam_poscom0;          // global position rel. to sub-com in qpos0 (ncam x 3)
  mjtNum*   cam_pos0;             // global position rel. to body in qpos0    (ncam x 3)
  mjtNum*   cam_mat0;             // global orientation in qpos0              (ncam x 9)
  int*      cam_projection;       // projection type (mjtProjection)          (ncam x 1)
  mjtNum*   cam_fovy;             // y field-of-view (ortho ? len : deg)      (ncam x 1)
  mjtNum*   cam_ipd;              // inter-pupilary distance                  (ncam x 1)
  int*      cam_resolution;       // resolution: pixels [width, height]       (ncam x 2)
  int*      cam_output;           // output types (mjtCamOut bit flags)       (ncam x 1)
  float*    cam_sensorsize;       // sensor size: length [width, height]      (ncam x 2)
  float*    cam_intrinsic;        // [focal length; principal point]          (ncam x 4)
  mjtNum*   cam_user;             // user data                                (ncam x nuser_cam)

  // lights
  int*      light_mode;           // light tracking mode (mjtCamLight)        (nlight x 1)
  int*      light_bodyid;         // id of light's body                       (nlight x 1)
  int*      light_targetbodyid;   // id of targeted body; -1: none            (nlight x 1)
  int*      light_type;           // spot, directional, etc. (mjtLightType)   (nlight x 1)
  int*      light_texid;          // texture id for image lights              (nlight x 1)
  mjtBool*  light_castshadow;     // does light cast shadows                  (nlight x 1)
  float*    light_bulbradius;     // light radius for soft shadows            (nlight x 1)
  float*    light_intensity;      // intensity, in candela                    (nlight x 1)
  float*    light_range;          // range of effectiveness                   (nlight x 1)
  mjtBool*  light_active;         // is light on                              (nlight x 1)
  mjtNum*   light_pos;            // position rel. to body frame              (nlight x 3)
  mjtNum*   light_dir;            // direction rel. to body frame             (nlight x 3)
  mjtNum*   light_poscom0;        // global position rel. to sub-com in qpos0 (nlight x 3)
  mjtNum*   light_pos0;           // global position rel. to body in qpos0    (nlight x 3)
  mjtNum*   light_dir0;           // global direction in qpos0                (nlight x 3)
  float*    light_attenuation;    // OpenGL attenuation (quadratic model)     (nlight x 3)
  float*    light_cutoff;         // OpenGL cutoff                            (nlight x 1)
  float*    light_softness;       // spotlight edge softness                  (nlight x 1)
  float*    light_exponent;       // OpenGL exponent                          (nlight x 1)
  float*    light_ambient;        // ambient rgb (alpha=1)                    (nlight x 3)
  float*    light_diffuse;        // diffuse rgb (alpha=1)                    (nlight x 3)
  float*    light_specular;       // specular rgb (alpha=1)                   (nlight x 3)

  // flexes: contact properties
  int*      flex_contype;         // flex contact type                        (nflex x 1)
  int*      flex_conaffinity;     // flex contact affinity                    (nflex x 1)
  int*      flex_condim;          // contact dimensionality (1, 3, 4, 6)      (nflex x 1)
  int*      flex_priority;        // flex contact priority                    (nflex x 1)
  mjtNum*   flex_solmix;          // mix coef for solref/imp in contact pair  (nflex x 1)
  mjtNum*   flex_solref;          // constraint solver reference: contact     (nflex x mjNREF)
  mjtNum*   flex_solimp;          // constraint solver impedance: contact     (nflex x mjNIMP)
  mjtNum*   flex_friction;        // friction for (slide, spin, roll)         (nflex x 3)
  mjtNum*   flex_margin;          // geometric inflation for contact          (nflex x 1)
  mjtNum*   flex_gap;             // additional contact detection buffer      (nflex x 1)
  int*      flex_selfcollide;     // self collision mode (mjtFlexSelf)        (nflex x 1)
  int*      flex_activelayers;    // number of active element layers, 3D only (nflex x 1)
  int*      flex_passive;         // passive collisions enabled               (nflex x 1)

  // flexes: other properties
  int*      flex_dim;             // 1: lines, 2: triangles, 3: tetrahedra    (nflex x 1)
  int*      flex_matid;           // material id for rendering                (nflexelem x 1)
  int*      flex_group;           // group for visibility                     (nflex x 1)
  int*      flex_interp;          // interpolation (0: vertex, 1: nodes)      (nflex x 1)
  int*      flex_cellnum;         // finite cell num per dimension            (nflex x 3)
  int*      flex_nodeadr;         // first node address                       (nflex x 1)
  int*      flex_nodenum;         // number of nodes                          (nflex x 1)
  int*      flex_vertadr;         // first vertex address                     (nflex x 1)
  int*      flex_vertnum;         // number of vertices                       (nflex x 1)
  int*      flex_edgeadr;         // first edge address                       (nflex x 1)
  int*      flex_edgenum;         // number of edges                          (nflex x 1)
  int*      flex_elemadr;         // first element address                    (nflex x 1)
  int*      flex_elemnum;         // number of elements                       (nflex x 1)
  int*      flex_elemdataadr;     // first element vertex id address          (nflex x 1)
  int*      flex_stiffnessadr;    // stiffness matrix address                 (nflex x 1)
  int*      flex_elemedgeadr;     // first element edge id address            (nflex x 1)
  int*      flex_bendingadr;      // first bending data address               (nflex x 1)
  int*      flex_plasticadr;      // plastic state address; -1: elastic       (nflex x 1)
  int*      flex_plasticmode;     // plastic state bits; 1: bend, 2: stretch  (nflex x 1)
  int*      flex_shellnum;        // number of shells                         (nflex x 1)
  int*      flex_shelldataadr;    // first shell data address                 (nflex x 1)
  int*      flex_texcoordadr;     // address in flex_texcoord; -1: none       (nflex x 1)
  int*      flex_nodebodyid;      // node body ids                            (nflexnode x 1)
  int*      flex_vertbodyid;      // vertex body ids                          (nflexvert x 1)
  int*      flex_vertcell;        // finite cell index for vertex; -1: none   (nflexvert x 1)
  int*      flex_edge;            // edge vertex ids (2 per edge)             (nflexedge x 2)
  int*      flex_edgeflap;        // adjacent vertex ids (dim=2 only)         (nflexedge x 2)
  int*      flex_elem;            // element vertex ids (dim+1 per elem)      (nflexelemdata x 1)
  int*      flex_elemtexcoord;    // element texture coordinates (dim+1)      (nflexelemdata x 1)
  int*      flex_elemedge;        // element edge ids                         (nflexelemedge x 1)
  int*      flex_elemlayer;       // element distance from surface, 3D only   (nflexelem x 1)
  int*      flex_shell;           // shell fragment vertex ids (dim per frag) (nflexshelldata x 1)
  mjtNum*   flex_vert;            // vertex positions in local body frames    (nflexvert x 3)
  mjtNum*   flex_vert0;           // vertex positions in qpos0 on [0, 1]^d    (nflexvert x 3)
  mjtNum*   flex_node;            // node positions in local body frames      (nflexnode x 3)
  mjtNum*   flex_node0;           // Cartesian node positions in qpos0        (nflexnode x 3)
  mjtNum*   flexedge_length0;     // edge lengths in qpos0                    (nflexedge x 1)
  mjtNum*   flexedge_invweight0;  // edge inv. weight in qpos0                (nflexedge x 1)
  mjtNum*   flex_radius;          // radius around primitive element          (nflex x 1)
  mjtNum*   flex_size;            // vertex bounding box half sizes in qpos0  (nflex x 3)
  mjtNum*   flex_stiffness;       // finite element stiffness matrix          (nflexstiffness x 1)
  mjtNum*   flex_bending;         // bending stiffness                        (nflexbending x 1)
  int*      efm0_dofid;           // constant metric factor row->dof address  (nefm0dof x 1)
  int*      efm0_L_rownnz;        // constant metric factor row nonzeros      (nefm0dof x 1)
  int*      efm0_L_rowadr;        // constant metric factor row addresses     (nefm0dof x 1)
  int*      efm0_L_colind;        // constant metric factor column indices    (nefm0L x 1)
  mjtNum*   efm0_L;               // factor of M + (dt^2+dt*d)*K_bend         (nefm0L x 1)
  int*      efmC_perm;            // sparse factor order: vertex dof or -1    (nefmCvert x 1)
  mjtNum*   flex_damping;         // Rayleigh's damping coefficient           (nflex x 1)
  mjtNum*   flex_yield;           // yield strain (stretch, bend); 0: elastic (nflex x 2)
  mjtNum*   flex_edgestiffness;   // edge stiffness                           (nflex x 1)
  mjtNum*   flex_edgedamping;     // edge damping                             (nflex x 1)
  int*      flex_edgeequality;    // 0:none, 1:edges, 3:strain                (nflex x 1)
  mjtBool*  flex_rigid;           // are all vertices in the same body        (nflex x 1)
  mjtBool*  flexedge_rigid;       // are both edge vertices in same body      (nflexedge x 1)
  mjtBool*  flex_centered;        // are all vertex coordinates (0,0,0)       (nflex x 1)
  mjtBool*  flex_flatskin;        // render flex skin with flat shading       (nflex x 1)
  int*      flex_bvhadr;          // address of bvh root; -1: no bvh          (nflex x 1)
  int*      flex_bvhnum;          // number of bounding volumes               (nflex x 1)
  int*      flexedge_J_rownnz;    // number of non-zeros in Jacobian row      (nflexedge x 1)
  int*      flexedge_J_rowadr;    // row start address in colind array        (nflexedge x 1)
  int*      flexedge_J_colind;    // column indices in sparse Jacobian        (nJfe x 1)
  float*    flex_rgba;            // rgba when material is omitted            (nflex x 4)
  float*    flex_texcoord;        // vertex texture coordinates               (nflextexcoord x 2)

  // meshes
  int*      mesh_vertadr;         // first vertex address                     (nmesh x 1)
  int*      mesh_vertnum;         // number of vertices                       (nmesh x 1)
  int*      mesh_faceadr;         // first face address                       (nmesh x 1)
  int*      mesh_facenum;         // number of faces                          (nmesh x 1)
  int*      mesh_bvhadr;          // address of bvh root                      (nmesh x 1)
  int*      mesh_bvhnum;          // number of bvh                            (nmesh x 1)
  int*      mesh_octadr;          // address of octree root                   (nmesh x 1)
  int*      mesh_octnum;          // number of octree nodes                   (nmesh x 1)
  int*      mesh_normaladr;       // first normal address                     (nmesh x 1)
  int*      mesh_normalnum;       // number of normals                        (nmesh x 1)
  int*      mesh_texcoordadr;     // texcoord data address; -1: no texcoord   (nmesh x 1)
  int*      mesh_texcoordnum;     // number of texcoord                       (nmesh x 1)
  int*      mesh_graphadr;        // graph data address; -1: no graph         (nmesh x 1)
  int*      mesh_extrema;         // extremum vertices in 3x3x3 directions    (nmesh x 27)
  float*    mesh_vert;            // vertex positions for all meshes          (nmeshvert x 3)
  float*    mesh_normal;          // normals for all meshes                   (nmeshnormal x 3)
  float*    mesh_texcoord;        // vertex texcoords for all meshes          (nmeshtexcoord x 2)
  int*      mesh_face;            // vertex face data                         (nmeshface x 3)
  int*      mesh_facenormal;      // normal face data                         (nmeshface x 3)
  int*      mesh_facetexcoord;    // texture face data                        (nmeshface x 3)
  int*      mesh_graph;           // convex graph data                        (nmeshgraph x 1)
  mjtNum*   mesh_scale;           // scaling applied to asset vertices        (nmesh x 3)
  mjtNum*   mesh_pos;             // translation applied to asset vertices    (nmesh x 3)
  mjtNum*   mesh_quat;            // rotation applied to asset vertices       (nmesh x 4)
  int*      mesh_pathadr;         // address of asset path for mesh; -1: none (nmesh x 1)
  int*      mesh_polynum;         // number of polygons per mesh              (nmesh x 1)
  int*      mesh_polyadr;         // first polygon address per mesh           (nmesh x 1)
  mjtNum*   mesh_polynormal;      // all polygon normals                      (nmeshpoly x 3)
  int*      mesh_polyvertadr;     // polygon vertex start address             (nmeshpoly x 1)
  int*      mesh_polyvertnum;     // number of vertices per polygon           (nmeshpoly x 1)
  int*      mesh_polyvert;        // all polygon vertices                     (nmeshpolyvert x 1)
  int*      mesh_polymapadr;      // first polygon address per vertex         (nmeshvert x 1)
  int*      mesh_polymapnum;      // number of polygons per vertex            (nmeshvert x 1)
  int*      mesh_polymap;         // vertex to polygon map                    (nmeshpolymap x 1)

  // skins
  int*      skin_matid;           // skin material id; -1: none               (nskin x 1)
  int*      skin_group;           // group for visibility                     (nskin x 1)
  float*    skin_rgba;            // skin rgba                                (nskin x 4)
  float*    skin_inflate;         // inflate skin in normal direction         (nskin x 1)
  int*      skin_vertadr;         // first vertex address                     (nskin x 1)
  int*      skin_vertnum;         // number of vertices                       (nskin x 1)
  int*      skin_texcoordadr;     // texcoord data address; -1: no texcoord   (nskin x 1)
  int*      skin_faceadr;         // first face address                       (nskin x 1)
  int*      skin_facenum;         // number of faces                          (nskin x 1)
  int*      skin_boneadr;         // first bone in skin                       (nskin x 1)
  int*      skin_bonenum;         // number of bones in skin                  (nskin x 1)
  float*    skin_vert;            // vertex positions for all skin meshes     (nskinvert x 3)
  float*    skin_texcoord;        // vertex texcoords for all skin meshes     (nskintexvert x 2)
  int*      skin_face;            // triangle faces for all skin meshes       (nskinface x 3)
  int*      skin_bonevertadr;     // first vertex in each bone                (nskinbone x 1)
  int*      skin_bonevertnum;     // number of vertices in each bone          (nskinbone x 1)
  float*    skin_bonebindpos;     // bind pos of each bone                    (nskinbone x 3)
  float*    skin_bonebindquat;    // bind quat of each bone                   (nskinbone x 4)
  int*      skin_bonebodyid;      // body id of each bone                     (nskinbone x 1)
  int*      skin_bonevertid;      // mesh ids of vertices in each bone        (nskinbonevert x 1)
  float*    skin_bonevertweight;  // weights of vertices in each bone         (nskinbonevert x 1)
  int*      skin_pathadr;         // address of asset path for skin; -1: none (nskin x 1)

  // height fields
  mjtNum*   hfield_size;          // (x, y, z_top, z_bottom)                  (nhfield x 4)
  int*      hfield_nrow;          // number of rows in grid                   (nhfield x 1)
  int*      hfield_ncol;          // number of columns in grid                (nhfield x 1)
  int*      hfield_adr;           // address in hfield_data                   (nhfield x 1)
  float*    hfield_data;          // elevation data                           (nhfielddata x 1)
  int*      hfield_pathadr;       // address of hfield asset path; -1: none   (nhfield x 1)

  // textures
  int*      tex_type;             // texture type (mjtTexture)                (ntex x 1)
  int*      tex_colorspace;       // texture colorspace (mjtColorSpace)       (ntex x 1)
  int*      tex_height;           // number of rows in texture image          (ntex x 1)
  int*      tex_width;            // number of columns in texture image       (ntex x 1)
  int*      tex_nchannel;         // number of channels in texture image      (ntex x 1)
  mjtSize*  tex_adr;              // start address in tex_data                (ntex x 1)
  mjtByte*  tex_data;             // pixel values                             (ntexdata x 1)
  int*      tex_pathadr;          // address of texture asset path; -1: none  (ntex x 1)
  int*      tex_contenttypeadr;   // content type address in paths; -1: none  (ntex x 1)

  // materials
  int*      mat_texid;            // indices of textures; -1: none            (nmat x mjNTEXROLE)
  mjtBool*  mat_texuniform;       // make texture cube uniform                (nmat x 1)
  float*    mat_texrepeat;        // texture repetition for 2d mapping        (nmat x 2)
  float*    mat_emission;         // emission (x rgb)                         (nmat x 1)
  float*    mat_specular;         // specular (x white)                       (nmat x 1)
  float*    mat_shininess;        // shininess coef                           (nmat x 1)
  float*    mat_reflectance;      // reflectance (0: disable)                 (nmat x 1)
  float*    mat_metallic;         // metallic coef                            (nmat x 1)
  float*    mat_roughness;        // roughness coef                           (nmat x 1)
  float*    mat_rgba;             // rgba                                     (nmat x 4)

  // predefined geom pairs for collision detection; has precedence over exclude
  int*      pair_dim;             // contact dimensionality                   (npair x 1)
  int*      pair_geom1;           // id of geom1                              (npair x 1)
  int*      pair_geom2;           // id of geom2                              (npair x 1)
  int*      pair_signature;       // body1 << 16 + body2                      (npair x 1)
  mjtNum*   pair_solref;          // solver reference: contact normal         (npair x mjNREF)
  mjtNum*   pair_solreffriction;  // solver reference: contact friction       (npair x mjNREF)
  mjtNum*   pair_solimp;          // solver impedance: contact                (npair x mjNIMP)
  mjtNum*   pair_margin;          // geometric inflation for contact          (npair x 1)
  mjtNum*   pair_gap;             // additional contact detection buffer      (npair x 1)
  mjtNum*   pair_adhesion;        // adhesive force of contacts               (npair x 1)
  mjtNum*   pair_friction;        // tangent1, 2, spin, roll1, 2              (npair x 5)

  // excluded body pairs for collision detection
  int*      exclude_signature;    // body1 << 16 + body2                      (nexclude x 1)

  // equality constraints
  int*      eq_type;              // constraint type (mjtEq)                  (neq x 1)
  int*      eq_obj1id;            // id of object 1                           (neq x 1)
  int*      eq_obj2id;            // id of object 2                           (neq x 1)
  int*      eq_objtype;           // type of both objects (mjtObj)            (neq x 1)
  mjtBool*  eq_active0;           // initial enable/disable constraint state  (neq x 1)
  mjtNum*   eq_solref;            // constraint solver reference              (neq x mjNREF)
  mjtNum*   eq_solimp;            // constraint solver impedance              (neq x mjNIMP)
  mjtNum*   eq_data;              // numeric data for constraint              (neq x mjNEQDATA)

  // tendons
  int*      tendon_type;          // type of tendon (mjtTendon)               (ntendon x 1)
  int*      tendon_adr;           // address of first object in tendon's path (ntendon x 1)
  int*      tendon_num;           // number of objects in tendon's path       (ntendon x 1)
  int*      tendon_matid;         // material id for rendering                (ntendon x 1)
  int*      tendon_actuatorid;    // actuator contributing damping / armature (ntendon x 1)
  int*      tendon_group;         // group for visibility                     (ntendon x 1)
  int*      tendon_plasticadr;    // start addr in plastic state (-1: none)   (ntendon x 1)
  int*      tendon_treenum;       // number of trees along tendon's path      (ntendon x 1)
  int*      tendon_treeid;        // first two trees along tendon's path      (ntendon x 2)
  int*      ten_J_rownnz;         // number of non-zeros in Jacobian row      (ntendon x 1)
  int*      ten_J_rowadr;         // row start address in colind array        (ntendon x 1)
  int*      ten_J_colind;         // column indices in sparse Jacobian        (nJten x 1)
  mjtBool*  tendon_limited;       // does tendon have length limits           (ntendon x 1)
  mjtBool*  tendon_actfrclimited; // does tendon have actuator force limits   (ntendon x 1)
  mjtNum*   tendon_width;         // width for rendering                      (ntendon x 1)
  mjtNum*   tendon_solref_lim;    // constraint solver reference: limit       (ntendon x mjNREF)
  mjtNum*   tendon_solimp_lim;    // constraint solver impedance: limit       (ntendon x mjNIMP)
  mjtNum*   tendon_solref_fri;    // constraint solver reference: friction    (ntendon x mjNREF)
  mjtNum*   tendon_solimp_fri;    // constraint solver impedance: friction    (ntendon x mjNIMP)
  mjtNum*   tendon_range;         // tendon length limits                     (ntendon x 2)
  mjtNum*   tendon_actfrcrange;   // range of total actuator force            (ntendon x 2)
  mjtNum*   tendon_margin;        // min distance for limit detection         (ntendon x 1)
  mjtNum*   tendon_stiffness;     // linear stiffness coefficient             (ntendon x 1)
  mjtNum*   tendon_stiffnesspoly; // high-order stiffness coefficients        (ntendon x mjNPOLY)
  mjtNum*   tendon_damping;       // linear damping coefficient               (ntendon x 1)
  mjtNum*   tendon_dampingpoly;   // high-order damping coefficients          (ntendon x mjNPOLY)
  mjtNum*   tendon_armature;      // inertia associated with tendon velocity  (ntendon x 1)
  mjtNum*   tendon_frictionloss;  // loss due to friction                     (ntendon x 1)
  mjtNum*   tendon_lengthspring;  // spring resting length range              (ntendon x 2)
  mjtNum*   tendon_springyield;   // plastic yield threshold                  (ntendon x 1)
  mjtNum*   tendon_length0;       // tendon length in qpos0                   (ntendon x 1)
  mjtNum*   tendon_invweight0;    // inv. weight in qpos0                     (ntendon x 1)
  mjtNum*   tendon_user;          // user data                                (ntendon x nuser_tendon)
  float*    tendon_rgba;          // rgba when material is omitted            (ntendon x 4)

  // list of all wrap objects in tendon paths
  int*      wrap_type;            // wrap object type (mjtWrap)               (nwrap x 1)
  int*      wrap_objid;           // object id: geom, site, joint             (nwrap x 1)
  mjtNum*   wrap_prm;             // divisor, joint coef, or site id          (nwrap x 1)

  // actuators
  int*      actuator_trntype;     // transmission type (mjtTrn)               (nactuator x 1)
  int*      actuator_dyntype;     // dynamics type (mjtDyn)                   (nactuator x 1)
  int*      actuator_gaintype;    // gain type (mjtGain)                      (nactuator x 1)
  int*      actuator_biastype;    // bias type (mjtBias)                      (nactuator x 1)
  int*      actuator_ctrladr;     // address of first control                 (nactuator x 1)
  int*      actuator_ctrlnum;     // number of controls                       (nactuator x 1)
  int*      actuator_ctrlspec;    // input signature, scoped by gaintype      (nactuator x 1)
  int*      actuator_outadr;      // address of first force output            (nactuator x 1)
  int*      actuator_outnum;      // number of force outputs, from trntype    (nactuator x 1)
  int*      actuator_actadr;      // first activation address; -1: stateless  (nactuator x 1)
  int*      actuator_actnum;      // number of activation variables           (nactuator x 1)
  int*      actuator_trnid;       // transmission id: joint, tendon, site     (nactuator x 2)
  mjtNum*   actuator_cranklength; // crank length for slider-crank            (nactuator x 1)
  mjtNum*   actuator_dynprm;      // dynamics parameters                      (nactuator x mjNDYN)
  mjtNum*   actuator_gainprm;     // gain parameters                          (nactuator x mjNGAIN)
  mjtNum*   actuator_biasprm;     // bias parameters                          (nactuator x mjNBIAS)
  mjtBool*  actuator_actlimited;  // is activation limited                    (nactuator x 1)
  mjtNum*   actuator_actrange;    // range of activations                     (nactuator x 2)
  mjtBool*  actuator_actearly;    // step activation before force             (nactuator x 1)
  int*      actuator_history;     // history buffer: [nsample, interp]        (nactuator x 2)
  int*      actuator_historyadr;  // address in history buffer; -1: none      (nactuator x 1)
  mjtNum*   actuator_delay;       // delay time; 0: no delay                  (nactuator x 1)
  mjtNum*   actuator_damping;     // linear damping coefficient               (nactuator x 1)
  mjtNum*   actuator_dampingpoly; // high-order damping coefficients          (nactuator x mjNPOLY)
  mjtNum*   actuator_armature;    // armature added to target (joint, tendon) (nactuator x 1)
  int*      actuator_group;       // group for visibility                     (nactuator x 1)
  mjtNum*   actuator_user;        // user data                                (nactuator x nuser_actuator)
  int*      actuator_plugin;      // plugin instance id; -1: not a plugin     (nactuator x 1)
  mjtBool*  actuator_forcelimited;// is force limited                         (nactuator x 1)
  mjtNum*   actuator_forcerange;  // range of forces                          (nactuator x 2)
  mjtBool*  actuator_ctrllimited; // is control limited                       (nu x 1)
  mjtNum*   actuator_ctrlrange;   // range of controls                        (nu x 2)
  mjtNum*   actuator_gear;        // scale length and transmitted force       (nout x 6)
  mjtNum*   actuator_acc0;        // acceleration from unit force in qpos0    (nout x 1)
  mjtNum*   actuator_length0;     // actuator length in qpos0                 (nout x 1)
  mjtNum*   actuator_lengthrange; // feasible actuator length range           (nout x 2)

  // sensors
  int*      sensor_type;          // sensor type (mjtSensor)                  (nsensor x 1)
  int*      sensor_datatype;      // numeric data type (mjtDataType)          (nsensor x 1)
  int*      sensor_needstage;     // required compute stage (mjtStage)        (nsensor x 1)
  int*      sensor_objtype;       // type of sensorized object (mjtObj)       (nsensor x 1)
  int*      sensor_objid;         // id of sensorized object                  (nsensor x 1)
  int*      sensor_reftype;       // type of reference frame (mjtObj)         (nsensor x 1)
  int*      sensor_refid;         // id of reference frame; -1: global frame  (nsensor x 1)
  int*      sensor_intprm;        // sensor parameters                        (nsensor x mjNSENS)
  int*      sensor_dim;           // number of scalar outputs                 (nsensor x 1)
  int*      sensor_adr;           // address in sensor array                  (nsensor x 1)
  mjtNum*   sensor_cutoff;        // cutoff for real and positive; 0: ignore  (nsensor x 1)
  mjtNum*   sensor_noise;         // noise standard deviation                 (nsensor x 1)
  int*      sensor_history;       // history buffer: [nsample, interp]        (nsensor x 2)
  int*      sensor_historyadr;    // address in history buffer; -1: none      (nsensor x 1)
  mjtNum*   sensor_delay;         // delay time in seconds; 0: no delay       (nsensor x 1)
  mjtNum*   sensor_interval;      // interval: [period, phase] in seconds     (nsensor x 2)
  mjtNum*   sensor_user;          // user data                                (nsensor x nuser_sensor)
  int*      sensor_plugin;        // plugin instance id; -1: not a plugin     (nsensor x 1)

  // plugin instances
  int*      plugin;               // globally registered plugin slot number   (nplugin x 1)
  int*      plugin_stateadr;      // address in the plugin state array        (nplugin x 1)
  int*      plugin_statenum;      // number of states in the plugin instance  (nplugin x 1)
  char*     plugin_attr;          // config attributes of plugin instances    (npluginattr x 1)
  int*      plugin_attradr;       // address to each instance's config attrib (nplugin x 1)

  // custom numeric fields
  int*      numeric_adr;          // address of field in numeric_data         (nnumeric x 1)
  int*      numeric_size;         // size of numeric field                    (nnumeric x 1)
  mjtNum*   numeric_data;         // array of all numeric fields              (nnumericdata x 1)

  // custom text fields
  int*      text_adr;             // address of text in text_data             (ntext x 1)
  int*      text_size;            // size of text field (strlen+1)            (ntext x 1)
  char*     text_data;            // array of all text fields (0-terminated)  (ntextdata x 1)

  // custom tuple fields
  int*      tuple_adr;            // address of text in text_data             (ntuple x 1)
  int*      tuple_size;           // number of objects in tuple               (ntuple x 1)
  int*      tuple_objtype;        // array of object types in all tuples      (ntupledata x 1)
  int*      tuple_objid;          // array of object ids in all tuples        (ntupledata x 1)
  mjtNum*   tuple_objprm;         // array of object params in all tuples     (ntupledata x 1)

  // keyframes
  mjtNum*   key_time;             // key time                                 (nkey x 1)
  mjtNum*   key_qpos;             // key position                             (nkey x nq)
  mjtNum*   key_qvel;             // key velocity                             (nkey x nv)
  mjtNum*   key_act;              // key activation                           (nkey x na)
  mjtNum*   key_mpos;             // key mocap position                       (nkey x nmocap*3)
  mjtNum*   key_mquat;            // key mocap quaternion                     (nkey x nmocap*4)
  mjtNum*   key_ctrl;             // key control                              (nkey x nu)
  mjtNum*   key_plastic;          // key plastic state                        (nkey x nplastic)

  // names
  int*      name_bodyadr;         // body name pointers                       (nbody x 1)
  int*      name_jntadr;          // joint name pointers                      (njnt x 1)
  int*      name_geomadr;         // geom name pointers                       (ngeom x 1)
  int*      name_siteadr;         // site name pointers                       (nsite x 1)
  int*      name_camadr;          // camera name pointers                     (ncam x 1)
  int*      name_lightadr;        // light name pointers                      (nlight x 1)
  int*      name_flexadr;         // flex name pointers                       (nflex x 1)
  int*      name_meshadr;         // mesh name pointers                       (nmesh x 1)
  int*      name_skinadr;         // skin name pointers                       (nskin x 1)
  int*      name_hfieldadr;       // hfield name pointers                     (nhfield x 1)
  int*      name_texadr;          // texture name pointers                    (ntex x 1)
  int*      name_matadr;          // material name pointers                   (nmat x 1)
  int*      name_pairadr;         // geom pair name pointers                  (npair x 1)
  int*      name_excludeadr;      // exclude name pointers                    (nexclude x 1)
  int*      name_eqadr;           // equality constraint name pointers        (neq x 1)
  int*      name_tendonadr;       // tendon name pointers                     (ntendon x 1)
  int*      name_actuatoradr;     // actuator name pointers                   (nactuator x 1)
  int*      name_sensoradr;       // sensor name pointers                     (nsensor x 1)
  int*      name_numericadr;      // numeric name pointers                    (nnumeric x 1)
  int*      name_textadr;         // text name pointers                       (ntext x 1)
  int*      name_tupleadr;        // tuple name pointers                      (ntuple x 1)
  int*      name_keyadr;          // keyframe name pointers                   (nkey x 1)
  int*      name_pluginadr;       // plugin instance name pointers            (nplugin x 1)
  char*     names;                // names of all objects, 0-terminated       (nnames x 1)
  int*      names_map;            // internal hash map of names               (nnames_map x 1)

  // paths
  char*     paths;                // resolved paths to assets, 0-terminated   (npaths x 1)

  // sparse structures
  int*      B_rownnz;             // body-dof: non-zeros in each row          (nbody x 1)
  int*      B_rowadr;             // body-dof: row addresses                  (nbody x 1)
  int*      B_colind;             // body-dof: column indices                 (nB x 1)
  int*      M_rownnz;             // reduced inertia: non-zeros in each row   (nv x 1)
  int*      M_rowadr;             // reduced inertia: row addresses           (nv x 1)
  int*      M_colind;             // reduced inertia: column indices          (nC x 1)
  int*      mapM2M;               // index mapping from qM to M               (nC x 1)
  int*      D_rownnz;             // full inertia: non-zeros in each row      (nv x 1)
  int*      D_rowadr;             // full inertia: row addresses              (nv x 1)
  int*      D_diag;               // full inertia: index of diagonal element  (nv x 1)
  int*      D_colind;             // full inertia: column indices             (nD x 1)
  int*      mapM2D;               // index mapping from M to D                (nD x 1)
  int*      mapD2M;               // index mapping from D to M                (nC x 1)

  // compilation signature
  uint64_t  signature;            // also held by the mjSpec that compiled this model
} mjModel;

#endif  // MUJOCO_MJMODEL_H_
