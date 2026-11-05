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

#ifndef MUJOCO_INCLUDE_MJTYPE_H_
#define MUJOCO_INCLUDE_MJTYPE_H_

#include <stdbool.h>
#include <stdint.h>


//---------------------------------- floating-point definition -------------------------------------

// floating point data type and constants
#ifndef mjUSESINGLE
  typedef double mjtNum;
  #define mjMINVAL    1E-15       // minimum value in any denominator
  #define mjMAXVAL    1E+10       // maximum value in qpos, qvel, qacc
#else
  typedef float mjtNum;
  #define mjMINVAL    1E-15f
  #define mjMAXVAL    1E+10f
#endif

// ratio of circumference to diameter
#define mjPI          3.14159265358979323846


//---------------------------------- byte definition -----------------------------------------------

typedef unsigned char mjtByte;    // used for small integers and binary data

#ifndef __cplusplus
  typedef _Bool mjtBool;          // used for boolean values
#else
  typedef bool mjtBool;           // used for boolean values
#endif


//---------------------------------- size definition -----------------------------------------------

typedef int64_t mjtSize;          // used for buffer sizes


//---------------------------------- enum types (mjModel) ------------------------------------------

typedef enum mjtDisableBit {      // disable default feature bitflags
  mjDSBL_CONSTRAINT   = 1<<0,     // entire constraint solver
  mjDSBL_EQUALITY     = 1<<1,     // equality constraints
  mjDSBL_FRICTIONLOSS = 1<<2,     // joint and tendon frictionloss constraints
  mjDSBL_LIMIT        = 1<<3,     // joint and tendon limit constraints
  mjDSBL_CONTACT      = 1<<4,     // contact constraints
  mjDSBL_SPRING       = 1<<5,     // passive spring forces
  mjDSBL_DAMPER       = 1<<6,     // passive damping forces
  mjDSBL_GRAVITY      = 1<<7,     // gravitational forces
  mjDSBL_CLAMPCTRL    = 1<<8,     // clamp control to specified range
  mjDSBL_WARMSTART    = 1<<9,     // warmstart constraint solver
  mjDSBL_FILTERPARENT = 1<<10,    // remove collisions with parent body
  mjDSBL_ACTUATION    = 1<<11,    // apply actuation forces
  mjDSBL_REFSAFE      = 1<<12,    // integrator safety: make ref[0]>=2*timestep
  mjDSBL_SENSOR       = 1<<13,    // sensors
  mjDSBL_MIDPHASE     = 1<<14,    // mid-phase collision filtering
  mjDSBL_EULERDAMP    = 1<<15,    // implicit integration of joint damping in Euler integrator
  mjDSBL_AUTORESET    = 1<<16,    // automatic reset when numerical issues are detected
  mjDSBL_NATIVECCD    = 1<<17,    // native convex collision detection
  mjDSBL_ISLAND       = 1<<18,    // constraint island discovery
  mjDSBL_MULTICCD     = 1<<19,    // multiple CCD contact points

  mjNDISABLE          = 20        // number of disable flags
} mjtDisableBit;


typedef enum mjtEnableBit {       // enable optional feature bitflags
  mjENBL_OVERRIDE     = 1<<0,     // override contact parameters
  mjENBL_ENERGY       = 1<<1,     // energy computation
  mjENBL_FWDINV       = 1<<2,     // record solver statistics
  mjENBL_INVDISCRETE  = 1<<3,     // discrete-time inverse dynamics
  mjENBL_SLEEP        = 1<<4,     // sleeping
  mjENBL_DIAGEXACT    = 1<<5,     // exact diagonal of constraint inertia
  mjENBL_IPC          = 1<<6,     // IPC flex contact mode of the discrete integrator
  mjENBL_SPECULATIVE  = 1<<7,     // speculative contacts

  mjNENABLE           = 8         // number of enable flags
} mjtEnableBit;


typedef enum mjtJoint {           // type of degree of freedom
  mjJNT_FREE          = 0,        // global position and orientation (quat)       (7)
  mjJNT_BALL,                     // orientation (quat) relative to parent        (4)
  mjJNT_SLIDE,                    // sliding distance along body-fixed axis       (1)
  mjJNT_HINGE                     // rotation angle (rad) around body-fixed axis  (1)
} mjtJoint;


typedef enum mjtGeom {            // type of geometric shape
  // regular geom types
  mjGEOM_PLANE        = 0,        // plane
  mjGEOM_HFIELD,                  // height field
  mjGEOM_SPHERE,                  // sphere
  mjGEOM_CAPSULE,                 // capsule
  mjGEOM_ELLIPSOID,               // ellipsoid
  mjGEOM_CYLINDER,                // cylinder
  mjGEOM_BOX,                     // box
  mjGEOM_MESH,                    // mesh
  mjGEOM_SDF,                     // signed distance field

  mjNGEOMTYPES,                   // number of regular geom types

  // rendering-only geom types: not used in mjModel, not counted in mjNGEOMTYPES
  mjGEOM_ARROW        = 100,      // arrow
  mjGEOM_ARROW1,                  // arrow without wedges
  mjGEOM_ARROW2,                  // arrow in both directions
  mjGEOM_LINE,                    // line
  mjGEOM_LINEBOX,                 // box with line edges
  mjGEOM_FLEX,                    // flex
  mjGEOM_SKIN,                    // skin
  mjGEOM_LABEL,                   // text label
  mjGEOM_TRIANGLE,                // triangle

  mjGEOM_NONE         = 1001      // missing geom type
} mjtGeom;


typedef enum mjtProjection {      // type of camera projection
  mjPROJ_PERSPECTIVE  = 0,        // perspective
  mjPROJ_ORTHOGRAPHIC             // orthographic
} mjtProjection;


typedef enum mjtCamLight {        // tracking mode for camera and light
  mjCAMLIGHT_FIXED    = 0,        // pos and rot fixed in body
  mjCAMLIGHT_TRACK,               // pos tracks body, rot fixed in global
  mjCAMLIGHT_TRACKCOM,            // pos tracks subtree com, rot fixed in body
  mjCAMLIGHT_TARGETBODY,          // pos fixed in body, rot tracks target body
  mjCAMLIGHT_TARGETBODYCOM        // pos fixed in body, rot tracks target subtree com
} mjtCamLight;


typedef enum mjtLightType {       // type of light
  mjLIGHT_SPOT        = 0,        // spot
  mjLIGHT_DIRECTIONAL,            // directional
  mjLIGHT_POINT,                  // point
  mjLIGHT_IMAGE,                  // image-based
} mjtLightType;


typedef enum mjtTexture {         // type of texture
  mjTEXTURE_2D        = 0,        // 2d texture, suitable for planes and hfields
  mjTEXTURE_CUBE,                 // cube texture, suitable for all other geom types
  mjTEXTURE_SKYBOX                // cube texture used as skybox
} mjtTexture;


typedef enum mjtTextureRole {     // role of texture map in rendering
  mjTEXROLE_USER      = 0,        // unspecified
  mjTEXROLE_RGB,                  // base color (albedo)
  mjTEXROLE_OCCLUSION,            // ambient occlusion
  mjTEXROLE_ROUGHNESS,            // roughness
  mjTEXROLE_METALLIC,             // metallic
  mjTEXROLE_NORMAL,               // normal (bump) map
  mjTEXROLE_OPACITY,              // opacity
  mjTEXROLE_EMISSIVE,             // light emission
  mjTEXROLE_RGBA,                 // base color, opacity
  mjTEXROLE_ORM,                  // occlusion, roughness, metallic
  mjNTEXROLE
} mjtTextureRole;


typedef enum mjtColorSpace {      // type of color space encoding
  mjCOLORSPACE_AUTO   = 0,        // attempts to autodetect color space, defaults to linear
  mjCOLORSPACE_LINEAR,            // linear color space
  mjCOLORSPACE_SRGB               // standard RGB color space
} mjtColorSpace;


typedef enum mjtIntegrator {      // integrator mode
  mjINT_EULER         = 0,        // semi-implicit Euler
  mjINT_RK4,                      // 4th-order Runge Kutta
  mjINT_IMPLICIT,                 // implicit in velocity
  mjINT_IMPLICITFAST,             // implicit in velocity, no rne derivative
  mjINT_DISCRETE                  // discrete step map: constraint solve in the effective metric
} mjtIntegrator;


typedef enum mjtCone {            // type of friction cone
  mjCONE_PYRAMIDAL     = 0,       // pyramidal
  mjCONE_ELLIPTIC                 // elliptic
} mjtCone;


typedef enum mjtJacobian {        // type of constraint Jacobian
  mjJAC_DENSE          = 0,       // dense
  mjJAC_SPARSE,                   // sparse
  mjJAC_AUTO                      // dense if nv<60, sparse otherwise
} mjtJacobian;


typedef enum mjtSolver {          // constraint solver algorithm
  mjSOL_PGS            = 0,       // PGS    (dual)
  mjSOL_CG,                       // CG     (primal)
  mjSOL_NEWTON                    // Newton (primal)
} mjtSolver;


typedef enum mjtEq {              // type of equality constraint
  mjEQ_CONNECT        = 0,        // connect two bodies at a point (ball joint)
  mjEQ_WELD,                      // fix relative position and orientation of two bodies
  mjEQ_JOINT,                     // couple the values of two scalar joints with cubic
  mjEQ_TENDON,                    // couple the lengths of two tendons with cubic
  mjEQ_FLEX,                      // fix all edge lengths of a flex
  mjEQ_FLEXVERT,                  // unsupported, will cause an error if used
  mjEQ_FLEXSTRAIN,                // constrain strain of a trilinear/quadratic flex (B-bar)
  mjEQ_DISTANCE                   // unsupported, will cause an error if used
} mjtEq;


typedef enum mjtTendon {          // type of tendon
  mjTENDON_FIXED      = 0,        // linear combination of joint coordinates
  mjTENDON_SPATIAL                // path through sites, around geoms
} mjtTendon;


typedef enum mjtWrap {            // type of tendon wrap object
  mjWRAP_NONE         = 0,        // null object
  mjWRAP_JOINT,                   // constant moment arm
  mjWRAP_PULLEY,                  // pulley used to split tendon
  mjWRAP_SITE,                    // pass through site
  mjWRAP_SPHERE,                  // wrap around sphere
  mjWRAP_CYLINDER                 // wrap around (infinite) cylinder
} mjtWrap;


typedef enum mjtTrn {             // type of actuator transmission
  mjTRN_JOINT         = 0,        // force on joint
  mjTRN_JOINTINPARENT,            // force on joint, expressed in parent frame
  mjTRN_SLIDERCRANK,              // force via slider-crank linkage
  mjTRN_TENDON,                   // force on tendon
  mjTRN_SITE,                     // force on site
  mjTRN_BODY,                     // adhesion force on a body's geoms
  mjTRN_SO3,                      // torque on a relative orientation (3 force outputs)

  mjTRN_UNDEFINED     = 1000      // undefined transmission type
} mjtTrn;


typedef enum mjtDyn {             // type of actuator dynamics
  mjDYN_NONE          = 0,        // no internal dynamics; ctrl specifies force
  mjDYN_INTEGRATOR,               // integrator: da/dt = u
  mjDYN_FILTER,                   // linear filter: da/dt = (u-a) / tau
  mjDYN_FILTEREXACT,              // linear filter: da/dt = (u-a) / tau, with exact integration
  mjDYN_MUSCLE,                   // piecewise linear filter with two time constants
  mjDYN_DCMOTOR,                  // DC motor electrical dynamics
  mjDYN_PID,                      // PID controller states: slew, integral
  mjDYN_USER                      // user-defined dynamics type
} mjtDyn;


typedef enum mjtGain {            // type of actuator gain
  mjGAIN_FIXED        = 0,        // fixed gain
  mjGAIN_AFFINE,                  // const + kp*length + kv*velocity
  mjGAIN_MUSCLE,                  // muscle FLV curve computed by mju_muscleGain()
  mjGAIN_DCMOTOR,                 // DC motor gain: K or K/R
  mjGAIN_SO3,                     // geodesic servo on an SO3 transmission: force = kp * log(error)
  mjGAIN_PID,                     // PID controller: position and velocity setpoint inputs
  mjGAIN_USER                     // user-defined gain type
} mjtGain;


typedef enum mjtBias {            // type of actuator bias
  mjBIAS_NONE         = 0,        // no bias
  mjBIAS_AFFINE,                  // const + kp*length + kv*velocity
  mjBIAS_MUSCLE,                  // muscle passive force computed by mju_muscleBias()
  mjBIAS_DCMOTOR,                 // DC motor bias: back-EMF, cogging, LuGre friction
  mjBIAS_SO3,                     // damping term of the SO3 geodesic servo
  mjBIAS_USER                     // user-defined bias type
} mjtBias;


typedef enum mjtCtrlChart {       // so3 input signature (actuator_ctrlspec): orientation chart
  mjCHART_EXPMAP      = 1,        // exponential-map orientation target: 3 controls
  mjCHART_QUAT        = 2         // quaternion orientation target: 4 controls
} mjtCtrlChart;


typedef enum mjtCtrlInput {       // input signature (actuator_ctrlspec): present/declared inputs
  mjINPUT_POS         = 1,        // position setpoint input
  mjINPUT_VEL         = 2,        // velocity setpoint input
  mjINPUT_FF          = 4,        // feedforward input, in the actuator's output space
  mjINPUT_VOLTAGE     = 8,        // raw terminal voltage input (dcmotor)
  mjINPUT_NONE        = 16,       // explicitly no inputs: purely passive (dcmotor)
  mjINPUT_PRESSURE    = 32        // pressure input (cylinder)
} mjtCtrlInput;


typedef enum mjtObj {             // type of MujoCo object
  mjOBJ_UNKNOWN       = 0,        // unknown object type
  mjOBJ_BODY,                     // body
  mjOBJ_XBODY,                    // body, used to access regular frame instead of i-frame
  mjOBJ_JOINT,                    // joint
  mjOBJ_DOF,                      // dof
  mjOBJ_GEOM,                     // geom
  mjOBJ_SITE,                     // site
  mjOBJ_CAMERA,                   // camera
  mjOBJ_LIGHT,                    // light
  mjOBJ_FLEX,                     // flex
  mjOBJ_MESH,                     // mesh
  mjOBJ_SKIN,                     // skin
  mjOBJ_HFIELD,                   // heightfield
  mjOBJ_TEXTURE,                  // texture
  mjOBJ_MATERIAL,                 // material for rendering
  mjOBJ_PAIR,                     // geom pair to include
  mjOBJ_EXCLUDE,                  // body pair to exclude
  mjOBJ_EQUALITY,                 // equality constraint
  mjOBJ_TENDON,                   // tendon
  mjOBJ_ACTUATOR,                 // actuator
  mjOBJ_SENSOR,                   // sensor
  mjOBJ_NUMERIC,                  // numeric
  mjOBJ_TEXT,                     // text
  mjOBJ_TUPLE,                    // tuple
  mjOBJ_KEY,                      // keyframe
  mjOBJ_PLUGIN,                   // plugin instance

  mjNOBJECT,                      // number of object types

  // meta elements, do not appear in mjModel
  mjOBJ_FRAME         = 100,      // frame
  mjOBJ_DEFAULT,                  // default
  mjOBJ_MODEL                     // entire model
} mjtObj;


typedef enum mjtSensor {         // type of sensor
  // common robotic sensors, attached to a site
  mjSENS_TOUCH        = 0,        // scalar contact normal forces summed over sensor zone
  mjSENS_ACCELEROMETER,           // 3D linear acceleration, in local frame
  mjSENS_VELOCIMETER,             // 3D linear velocity, in local frame
  mjSENS_GYRO,                    // 3D angular velocity, in local frame
  mjSENS_FORCE,                   // 3D force between site's body and its parent body
  mjSENS_TORQUE,                  // 3D torque between site's body and its parent body
  mjSENS_MAGNETOMETER,            // 3D magnetometer
  mjSENS_RANGEFINDER,             // scalar distance to nearest geom along z-axis
  mjSENS_CAMPROJECTION,           // pixel coordinates of a site in the camera image

  // sensors related to scalar joints, tendons, actuators
  mjSENS_JOINTPOS,                // scalar joint position (hinge and slide only)
  mjSENS_JOINTVEL,                // scalar joint velocity (hinge and slide only)
  mjSENS_TENDONPOS,               // scalar tendon position
  mjSENS_TENDONVEL,               // scalar tendon velocity
  mjSENS_ACTUATORPOS,             // scalar actuator position
  mjSENS_ACTUATORVEL,             // scalar actuator velocity
  mjSENS_ACTUATORFRC,             // scalar actuator force
  mjSENS_JOINTACTFRC,             // scalar actuator force, measured at the joint
  mjSENS_TENDONACTFRC,            // scalar actuator force, measured at the tendon

  // sensors related to ball joints
  mjSENS_BALLQUAT,                // 4D ball joint quaternion
  mjSENS_BALLANGVEL,              // 3D ball joint angular velocity

  // joint and tendon limit sensors, in constraint space
  mjSENS_JOINTLIMITPOS,           // joint limit distance-margin
  mjSENS_JOINTLIMITVEL,           // joint limit velocity
  mjSENS_JOINTLIMITFRC,           // joint limit force
  mjSENS_TENDONLIMITPOS,          // tendon limit distance-margin
  mjSENS_TENDONLIMITVEL,          // tendon limit velocity
  mjSENS_TENDONLIMITFRC,          // tendon limit force

  // sensors attached to an object with spatial frame: (x)body, geom, site, camera
  mjSENS_FRAMEPOS,                // 3D position
  mjSENS_FRAMEQUAT,               // 4D unit quaternion orientation
  mjSENS_FRAMEXAXIS,              // 3D unit vector: x-axis of object's frame
  mjSENS_FRAMEYAXIS,              // 3D unit vector: y-axis of object's frame
  mjSENS_FRAMEZAXIS,              // 3D unit vector: z-axis of object's frame
  mjSENS_FRAMELINVEL,             // 3D linear velocity
  mjSENS_FRAMEANGVEL,             // 3D angular velocity
  mjSENS_FRAMELINACC,             // 3D linear acceleration
  mjSENS_FRAMEANGACC,             // 3D angular acceleration

  // sensors related to kinematic subtrees; attached to a body (which is the subtree root)
  mjSENS_SUBTREECOM,              // 3D center of mass of subtree
  mjSENS_SUBTREELINVEL,           // 3D linear velocity of subtree
  mjSENS_SUBTREEANGMOM,           // 3D angular momentum of subtree

  // sensors of geometric relationships
  mjSENS_INSIDESITE,              // 1 if object is inside a site, 0 otherwise
  mjSENS_GEOMDIST,                // signed distance between two geoms
  mjSENS_GEOMNORMAL,              // normal direction between two geoms
  mjSENS_GEOMFROMTO,              // segment between two geoms

  // sensors for reporting contacts which occurred during the simulation
  mjSENS_CONTACT,                 // contacts which occurred during the simulation

  // global sensors
  mjSENS_E_POTENTIAL,             // potential energy
  mjSENS_E_KINETIC,               // kinetic energy
  mjSENS_CLOCK,                   // simulation time

  // sensors related to SDFs
  mjSENS_TACTILE,                 // tactile sensor

  // plugin-controlled sensors
  mjSENS_PLUGIN,                  // plugin-controlled

  // user-defined sensor
  mjSENS_USER                     // sensor data provided by mjcb_sensor callback
} mjtSensor;


typedef enum mjtStage {           // computation stage
  mjSTAGE_NONE        = 0,        // no computations
  mjSTAGE_POS,                    // position-dependent computations
  mjSTAGE_VEL,                    // velocity-dependent computations
  mjSTAGE_ACC                     // acceleration/force-dependent computations
} mjtStage;


typedef enum mjtDataType {        // data type for sensors
  mjDATATYPE_REAL     = 0,        // real values, no constraints
  mjDATATYPE_POSITIVE,            // positive values; 0 or negative: inactive
  mjDATATYPE_AXIS,                // 3D unit vector
  mjDATATYPE_QUATERNION           // unit quaternion
} mjtDataType;


typedef enum mjtConDataField {    // data fields returned by contact sensors
  mjCONDATA_FOUND     = 0,        // whether a contact was found
  mjCONDATA_FORCE,                // contact force
  mjCONDATA_TORQUE,               // contact torque
  mjCONDATA_DIST,                 // contact penetration distance
  mjCONDATA_POS,                  // contact position
  mjCONDATA_NORMAL,               // contact frame normal
  mjCONDATA_TANGENT,              // contact frame first tangent
  mjCONDATA_LINVEL,               // contact linear velocity
  mjCONDATA_ANGVEL,               // contact angular velocity

  mjNCONDATA                      // number of contact sensor data fields
} mjtConDataField;


typedef enum mjtRayDataField {    // data fields returned by rangefinder sensors
  mjRAYDATA_DIST     = 0,         // distance from ray origin to nearest surface
  mjRAYDATA_DIR,                  // normalized ray direction
  mjRAYDATA_ORIGIN,               // ray origin
  mjRAYDATA_POINT,                // point at which ray intersects nearest surface
  mjRAYDATA_NORMAL,               // surface normal at intersection point
  mjRAYDATA_DEPTH,                // depth along z-axis

  mjNRAYDATA                      // number of rangefinder sensor data fields
} mjtRayDataField;


typedef enum mjtCamOutBit {       // camera output type bitflags
  mjCAMOUT_RGB        = 1<<0,     // RGB image
  mjCAMOUT_DEPTH      = 1<<1,     // depth image (distance from camera plane)
  mjCAMOUT_DIST       = 1<<2,     // distance image (distance from camera origin)
  mjCAMOUT_NORMAL     = 1<<3,     // normal image
  mjCAMOUT_SEG        = 1<<4,     // segmentation image

  mjNCAMOUT           = 5         // number of camera output types
} mjtCamOutBit;


typedef enum mjtSameFrame {       // frame alignment of bodies with their children
  mjSAMEFRAME_NONE    = 0,        // no alignment
  mjSAMEFRAME_BODY,               // frame is same as body frame
  mjSAMEFRAME_INERTIA,            // frame is same as inertial frame
  mjSAMEFRAME_BODYROT,            // frame orientation is same as body orientation
  mjSAMEFRAME_INERTIAROT          // frame orientation is same as inertia orientation
} mjtSameFrame;


typedef enum mjtSleepPolicy {     // per-tree sleep policy
  mjSLEEP_AUTO        = 0,        // compiler chooses sleep policy
  mjSLEEP_AUTO_NEVER,             // compiler sleep policy: never
  mjSLEEP_AUTO_ALLOWED,           // compiler sleep policy: allowed
  mjSLEEP_NEVER,                  // user sleep policy: never
  mjSLEEP_ALLOWED,                // user sleep policy: allowed
  mjSLEEP_INIT,                   // user sleep policy: initialized asleep
} mjtSleepPolicy;


typedef enum mjtLRMode {          // mode for actuator length range computation
  mjLRMODE_NONE       = 0,        // do not process any actuators
  mjLRMODE_MUSCLE,                // process muscle actuators
  mjLRMODE_MUSCLEUSER,            // process muscle and user actuators
  mjLRMODE_ALL                    // process all actuators
} mjtLRMode;


typedef enum mjtFlexSelf {        // mode for flex selfcollide
  mjFLEXSELF_NONE     = 0,        // no self-collisions
  mjFLEXSELF_NARROW,              // skip midphase, go directly to narrowphase
  mjFLEXSELF_BVH,                 // use BVH in midphase (if midphase enabled)
  mjFLEXSELF_SAP,                 // use SAP in midphase
  mjFLEXSELF_AUTO                 // choose between BVH and SAP automatically
} mjtFlexSelf;


typedef enum mjtSDFType {         // signed distance function (SDF) type
  mjSDFTYPE_SINGLE    = 0,        // single SDF
  mjSDFTYPE_INTERSECTION,         // max(A, B)
  mjSDFTYPE_MIDSURFACE,           // A - B
  mjSDFTYPE_COLLISION,            // A + B + abs(max(A, B))
} mjtSDFType;


//---------------------------------- enum types (mjData) -------------------------------------------

typedef enum mjtState {             // state elements
  mjSTATE_TIME           = 1<<0,    // time
  mjSTATE_QPOS           = 1<<1,    // position
  mjSTATE_QVEL           = 1<<2,    // velocity
  mjSTATE_ACT            = 1<<3,    // actuator activation
  mjSTATE_HISTORY        = 1<<4,    // history buffers (control, sensor)
  mjSTATE_WARMSTART      = 1<<5,    // acceleration used for warmstart
  mjSTATE_CTRL           = 1<<6,    // control
  mjSTATE_QFRC_APPLIED   = 1<<7,    // applied generalized force
  mjSTATE_XFRC_APPLIED   = 1<<8,    // applied Cartesian force/torque
  mjSTATE_EQ_ACTIVE      = 1<<9,    // enable/disable constraints
  mjSTATE_MOCAP_POS      = 1<<10,   // positions of mocap bodies
  mjSTATE_MOCAP_QUAT     = 1<<11,   // orientations of mocap bodies
  mjSTATE_USERDATA       = 1<<12,   // user data
  mjSTATE_PLUGIN         = 1<<13,   // plugin state
  mjSTATE_PLASTIC        = 1<<14,   // plastic state

  mjNSTATE               = 15,      // number of state elements

  // convenience values for commonly used state specifications
  mjSTATE_PHYSICS        = mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_ACT | mjSTATE_HISTORY |
                           mjSTATE_PLASTIC,
  mjSTATE_FULLPHYSICS    = mjSTATE_TIME | mjSTATE_PHYSICS | mjSTATE_PLUGIN,
  mjSTATE_USER           = mjSTATE_CTRL | mjSTATE_QFRC_APPLIED | mjSTATE_XFRC_APPLIED |
                           mjSTATE_EQ_ACTIVE | mjSTATE_MOCAP_POS | mjSTATE_MOCAP_QUAT |
                           mjSTATE_USERDATA,
  mjSTATE_INTEGRATION    = mjSTATE_FULLPHYSICS | mjSTATE_USER | mjSTATE_WARMSTART
} mjtState;


typedef enum mjtConstraint {        // type of constraint
  mjCNSTR_EQUALITY       = 0,       // equality constraint
  mjCNSTR_FRICTION_DOF,             // dof friction
  mjCNSTR_FRICTION_TENDON,          // tendon friction
  mjCNSTR_LIMIT_JOINT,              // joint limit
  mjCNSTR_LIMIT_TENDON,             // tendon limit
  mjCNSTR_CONTACT_FRICTIONLESS,     // frictionless contact
  mjCNSTR_CONTACT_PYRAMIDAL,        // frictional contact, pyramidal friction cone
  mjCNSTR_CONTACT_ELLIPTIC          // frictional contact, elliptic friction cone
} mjtConstraint;


typedef enum mjtConstraintState {   // constraint state
  mjCNSTRSTATE_SATISFIED = 0,       // constraint satisfied, zero cost (limit, contact)
  mjCNSTRSTATE_QUADRATIC,           // quadratic cost (equality, friction, limit, contact)
  mjCNSTRSTATE_LINEARNEG,           // linear cost, negative side (friction)
  mjCNSTRSTATE_LINEARPOS,           // linear cost, positive side (friction)
  mjCNSTRSTATE_CONE                 // squared distance to cone cost (elliptic contact)
} mjtConstraintState;


typedef enum mjtWarning {           // warning types
  mjWARN_INERTIA         = 0,       // (near) singular inertia matrix
  mjWARN_CONTACTFULL,               // too many contacts in contact list
  mjWARN_CNSTRFULL,                 // too many constraints
  mjWARN_BADQPOS,                   // bad number in qpos
  mjWARN_BADQVEL,                   // bad number in qvel
  mjWARN_BADQACC,                   // bad number in qacc
  mjWARN_BADCTRL,                   // bad number in ctrl

  mjNWARNING                        // number of warnings
} mjtWarning;


typedef enum mjtTimer {             // internal timers
  // main api
  mjTIMER_STEP           = 0,       // step
  mjTIMER_FORWARD,                  // forward
  mjTIMER_INVERSE,                  // inverse

  // breakdown of step/forward
  mjTIMER_POSITION,                 // fwdPosition
  mjTIMER_VELOCITY,                 // fwdVelocity
  mjTIMER_ACTUATION,                // fwdActuation
  mjTIMER_CONSTRAINT,               // fwdConstraint
  mjTIMER_ADVANCE,                  // mj_Euler, mj_implicit

  // breakdown of fwdPosition
  mjTIMER_POS_KINEMATICS,           // kinematics, com, tendon, transmission
  mjTIMER_POS_INERTIA,              // inertia computations
  mjTIMER_POS_COLLISION,            // collision detection
  mjTIMER_POS_MAKE,                 // make constraints
  mjTIMER_POS_PROJECT,              // project constraints

  // breakdown of mj_collision
  mjTIMER_COL_BROAD,                // broadphase
  mjTIMER_COL_NARROW,               // narrowphase

  mjNTIMER                          // number of timers
} mjtTimer;


typedef enum mjtSleepState {        // sleep state of an object
  mjS_STATIC = -1,                  // object is static
  mjS_ASLEEP = 0,                   // object is asleep
  mjS_AWAKE  = 1                    // object is awake
} mjtSleepState;


//---------------------------------- logging -------------------------------------------------------

typedef enum mjtLogLevel {        // log message severity
  mjLOG_DEBUG       = 0,          // internal engine debug trace (opt-in via topic filtering)
  mjLOG_INFO,                     // informational (opt-in via topic filtering)
  mjLOG_WARNING,                  // warning
  mjLOG_ERROR,                    // error
} mjtLogLevel;

typedef enum mjtLogTopic {        // log topic identifiers
  mjTOPIC_NONE     = 0,           // no topic (always passes filtering)
                                  // INFO topics:
  mjTOPIC_TIME_STP = 1,           // timing diagnostics (step)
  mjTOPIC_TIME_CMP = 2,           // timing diagnostics (compile)
                                  // DEBUG topics:
  mjTOPIC_SLEEP    = 3,           // sleep/wake events

  mjNTOPIC         = 3            // number of filterable topics
} mjtLogTopic;

typedef struct mjLogMessage_ {    // structured log message
  int level;                      // mjtLogLevel
  int topic;                      // mjtLogTopic (0 for error/warning/user)
  char subject[1024];             // message subject (one-liner, printf-formatted)
  const char* body;               // message body (multi-line detail, or NULL)
  const char* func;               // __func__ or NULL
  const char* file;               // __FILE__ or NULL
  int line;                       // __LINE__ or 0
  mjtBool timestamp;              // prepend timestamp to output
} mjLogMessage;

typedef struct mjLogConfig_ {     // log handler default configuration
  mjtBool logto_console;          // print to console (default: true)
  mjtBool logto_file;             // print to log file (default: true)
  char logfile[1024];             // log file path (default: "MUJOCO_LOG.TXT")
  int topics;                     // enabled info topic bitmask (default: 0)
} mjLogConfig;

// function type for log handler callback; must be thread-safe, must not call mju_error
typedef void (*mjfLogHandler)(const mjLogMessage*);

#endif  // MUJOCO_INCLUDE_MJTYPE_H_
