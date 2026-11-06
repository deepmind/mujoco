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

#ifndef MUJOCO_MUJOCO_H_
#define MUJOCO_MUJOCO_H_

// header version; should match the library version as returned by mj_version()
#define mjVERSION_HEADER 4000001

// needed to define size_t, fabs and log10
#include <stdlib.h>
#include <math.h>

// type definitions
#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjsan.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjui.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjassert.h>

// this is a C-API
#ifdef __cplusplus
extern "C" {
#endif

// user memory handlers
MJAPI extern void* (*mju_user_malloc)(size_t);
MJAPI extern void  (*mju_user_free)(void*);

// legacy error/warning handlers (deprecated: prefer mju_setLogHandler)
MJAPI extern void  (*mju_user_error)(const char*);
MJAPI extern void  (*mju_user_warning)(const char*);


// callbacks extending computation pipeline
MJAPI extern mjfGeneric  mjcb_passive;
MJAPI extern mjfGeneric  mjcb_control;
MJAPI extern mjfConFilt  mjcb_contactfilter;
MJAPI extern mjfSensor   mjcb_sensor;
MJAPI extern mjfTime     mjcb_time;
MJAPI extern mjfAct      mjcb_act_dyn;
MJAPI extern mjfAct      mjcb_act_gain;
MJAPI extern mjfAct      mjcb_act_bias;


// collision function table
MJAPI extern mjfCollision mjCOLLISIONFUNC[mjNGEOMTYPES][mjNGEOMTYPES];


// string names
MJAPI extern const char* mjDISABLESTRING[mjNDISABLE];
MJAPI extern const char* mjENABLESTRING[mjNENABLE];
MJAPI extern const char* mjTIMERSTRING[mjNTIMER];
MJAPI extern const char* mjLABELSTRING[mjNLABEL];
MJAPI extern const char* mjFRAMESTRING[mjNFRAME];
MJAPI extern const char* mjVISSTRING[mjNVISFLAG][3];
MJAPI extern const char* mjRNDSTRING[mjNRNDFLAG][3];
MJAPI extern const char* mjTOPICSTRING[mjNTOPIC];


//---------------------------------- Virtual file system -------------------------------------------

// Initialize an empty VFS, mj_deleteVFS must be called to deallocate the VFS.
MJAPI void mj_defaultVFS(mjVFS* vfs);

// Mount a provider to handle file operations under the given path; return 0: success,
// 2: repeated name, -1: invalid provider.
MJAPI int mj_mountVFS(mjVFS* vfs, const char* filepath, const mjpResourceProvider* provider);

// Unmount a previously mounted provider; return 0: success, -1: not found in VFS.
MJAPI int mj_unmountVFS(mjVFS* vfs, const char* filename);

// Add file to VFS; return 0: success, 2: repeated name, -1: failed to load.
MJAPI int mj_addFileVFS(mjVFS* vfs, const char* directory, const char* filename);

// Add file to VFS from buffer; return 0: success, 2: repeated name, -1: failed to load.
MJAPI int mj_addBufferVFS(mjVFS* vfs, const char* name, const void* buffer, int nbuffer);

// Delete file from VFS; return 0: success, -1: not found in VFS.
MJAPI int mj_deleteFileVFS(mjVFS* vfs, const char* filename);

// Check if buffer exists in VFS; return 1: exists, 0: not found.
MJAPI int mj_containsBufferVFS(mjVFS* vfs, const char* name);

// Check if file exists in VFS; return 1: exists, 0: not found.
MJAPI int mj_containsFileVFS(mjVFS* vfs, const char* directory, const char* filename);

// Delete all files from VFS and deallocates VFS internal memory.
MJAPI void mj_deleteVFS(mjVFS* vfs);

//------------------------------------ Asset cache -------------------------------------------------

// Get the current size of the asset cache in bytes.
MJAPI size_t mj_getCacheSize(const mjCache* cache);

// Get the capacity of the asset cache in bytes.
MJAPI size_t mj_getCacheCapacity(const mjCache* cache);

// Set the capacity of the asset cache in bytes (0 to disable); return the new capacity.
MJAPI size_t mj_setCacheCapacity(mjCache* cache, size_t size);

// Get the internal asset cache used by the compiler.
MJAPI mjCache* mj_getCache(void);

// Clear the asset cache.
MJAPI void mj_clearCache(mjCache* cache);

//---------------------------------- Parse and compile ---------------------------------------------

// Parse XML file in MJCF or URDF format, compile it; return low-level model.
// If vfs is not NULL, look up files in vfs before reading from disk.
// If error is not NULL, it must have size error_sz.
// Nullable: vfs, error
MJAPI mjModel* mj_loadXML(const char* filename, const mjVFS* vfs, char* error, int error_sz);

// Parse spec from XML file.
// Nullable: vfs, error
MJAPI mjSpec* mj_parseXML(const char* filename, const mjVFS* vfs, char* error, int error_sz);

// Parse spec from XML string.
// Nullable: vfs, error
MJAPI mjSpec* mj_parseXMLString(const char* xml, const mjVFS* vfs, char* error, int error_sz);

// Parse spec from a file.
// Nullable: vfs, error
MJAPI mjSpec* mj_parse(const char* filename, const char* content_type,
                       const mjVFS* vfs, char* error, int error_sz);

// Encode spec/model to a file using a registered encoder.
// Returns the number of bytes written on success, -1 on failure.
// Nullable: m, vfs, error
MJAPI mjtSize mj_encode(const mjSpec* s, const mjModel* m, const char* filename,
                        const char* content_type, const mjVFS* vfs, char* error,
                        int error_sz);

// Compile spec to model.
// Nullable: vfs
MJAPI mjModel* mj_compile(mjSpec* s, const mjVFS* vfs);

// Copy the values which were changed in a model to the spec it was compiled from; return 1 on success.
MJAPI int mj_copyBack(mjSpec* s, const mjModel* m);

// Recompile spec to model, preserving the state; return 0 on success.
// Nullable: vfs
MJAPI int mj_recompile(mjSpec* s, const mjVFS* vfs, mjModel* m, mjData* d);

// Decode texture id to raw bytes, exactly what compilation would store in tex_data, with
// tex_nchannel channels; embedded textures are copied from the model. The buffer is allocated with
// mju_malloc and freed by the caller; colorspace receives the resolved colorspace (mjtColorSpace).
// Return 0 on success.
// Nullable: width, height, colorspace, error
MJAPI int mj_loadTexture(const mjModel* m, int id, mjtByte** data, int* width, int* height,
                         int* colorspace, char* error, int error_sz);

// Copy of m with every by-reference asset loaded into it, as compilation would have stored it.
// Return NULL with error on a file that does not resolve. A hermetic MJB file of a model with
// by-reference assets is mj_embedAssets, then mj_saveModel, then mj_deleteModel.
// Nullable: error
MJAPI mjModel* mj_embedAssets(const mjModel* m, char* error, int error_sz);

// Update XML data structures with info from low-level model created with mj_loadXML, save as MJCF.
// If error is not NULL, it must have size error_sz.
// Nullable: error
MJAPI int mj_saveLastXML(const char* filename, const mjModel* m, char* error, int error_sz);

// Free last XML model if loaded. Called internally at each load.
MJAPI void mj_freeLastXML(void);

// Save spec to XML string; return 0 on success, -1 on failure.
// If length of the output buffer is too small; return the required size.
// Nullable: error
MJAPI int mj_saveXMLString(const mjSpec* s, char* xml, int xml_sz, char* error, int error_sz);

// Save spec to XML file; return 0 on success, -1 otherwise.
// Nullable: error
MJAPI int mj_saveXML(const mjSpec* s, const char* filename, char* error, int error_sz);

// Given MJCF filename, fills dependencies with a list of all other asset files it depends on.
// The search is recursive, and the list includes the filename itself.
MJAPI void mju_getXMLDependencies(const char* filename, mjStringVec* dependencies);

//---------------------------------- Main simulation -----------------------------------------------

// Advance simulation, use control callback to obtain external force and control.
MJAPI void mj_step(const mjModel* m, mjData* d);

// Advance simulation in two steps: before external force and control is set by user.
MJAPI void mj_step1(const mjModel* m, mjData* d);

// Advance simulation in two steps: after external force and control is set by user.
MJAPI void mj_step2(const mjModel* m, mjData* d);

// Forward dynamics: same as mj_step but do not integrate in time.
MJAPI void mj_forward(const mjModel* m, mjData* d);

// Inverse dynamics: qacc must be set before calling.
MJAPI void mj_inverse(const mjModel* m, mjData* d);

// Forward dynamics with skip; skipstage is mjtStage.
MJAPI void mj_forwardSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor);

// Inverse dynamics with skip; skipstage is mjtStage.
MJAPI void mj_inverseSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor);


//---------------------------------- Initialization ------------------------------------------------

// Set default options for length range computation.
MJAPI void mj_defaultLROpt(mjLROpt* opt);

// Set solver parameters to default values.
// Nullable: solref, solimp
MJAPI void mj_defaultSolRefImp(mjtNum* solref, mjtNum* solimp);

// Set physics options to default values.
MJAPI void mj_defaultOption(mjOption* opt);

// Set visual options to default values.
MJAPI void mj_defaultVisual(mjVisual* vis);

// Copy mjModel, allocate new if dest is NULL.
// Nullable: dest
MJAPI mjModel* mj_copyModel(mjModel* dest, const mjModel* src);

// Save model to binary MJB file or memory buffer; buffer has precedence when given.
// Nullable: filename, buffer
MJAPI void mj_saveModel(const mjModel* m, const char* filename, void* buffer, mjtSize buffer_sz);

// Load model from binary MJB file.
// If vfs is not NULL, look up file in vfs before reading from disk.
// Nullable: vfs
MJAPI mjModel* mj_loadModel(const char* filename, const mjVFS* vfs);

// Load model from memory buffer.
MJAPI mjModel* mj_loadModelBuffer(const void* buffer, mjtSize buffer_sz);

// Free memory allocation in model.
MJAPI void mj_deleteModel(mjModel* m);

// Return size of buffer needed to hold model.
MJAPI mjtSize mj_sizeModel(const mjModel* m);

// Allocate mjData corresponding to given model.
// If the model buffer is unallocated the initial configuration will not be set.
MJAPI mjData* mj_makeData(const mjModel* m);

// Copy mjData.
// m is only required to contain the size fields from MJMODEL_INTS.
MJAPI mjData* mj_copyData(mjData* dest, const mjModel* m, const mjData* src);

// Copy mjData, skip large arrays not required for visualization.
MJAPI mjData* mjv_copyData(mjData* dest, const mjModel* m, const mjData* src);

// Reset ctrl to neutral values: zero, except quaternion inputs which reset to the identity.
MJAPI void mj_resetCtrl(const mjModel* m, mjData* d);

// Reset data to defaults.
MJAPI void mj_resetData(const mjModel* m, mjData* d);

// Reset data to defaults, fill everything else with debug_value.
MJAPI void mj_resetDataDebug(const mjModel* m, mjData* d, unsigned char debug_value);

// Reset data. If 0 <= key < nkey, set fields from specified keyframe.
MJAPI void mj_resetDataKeyframe(const mjModel* m, mjData* d, int key);

#ifndef mjUSEASAN  // Stack management functions declared in mjsan.h if ASAN is active.

// Mark a new frame on the mjData stack.
MJAPI void mj_markStack(mjData* d);

// Free the current mjData stack frame. All pointers returned by mj_stackAlloc since the last call
// to mj_markStack must no longer be used afterwards.
MJAPI void mj_freeStack(mjData* d);

#endif  // ADDRESS_SANITIZER

// Allocate a number of bytes on mjData stack at a specific alignment.
// Call mju_error on stack overflow.
MJAPI void* mj_stackAllocByte(mjData* d, size_t bytes, size_t alignment);

// Allocate array of mjtNums on mjData stack. Call mju_error on stack overflow.
MJAPI mjtNum* mj_stackAllocNum(mjData* d, size_t size);

// Allocate array of ints on mjData stack. Call mju_error on stack overflow.
MJAPI int* mj_stackAllocInt(mjData* d, size_t size);

// Free memory allocation in mjData.
MJAPI void mj_deleteData(mjData* d);

// Reset all callbacks to NULL pointers (NULL is the default).
MJAPI void mj_resetCallbacks(void);

// Set constant fields of mjModel, corresponding to qpos0 configuration.
MJAPI void mj_setConst(mjModel* m, mjData* d);

// Set actuator_lengthrange for specified actuator; return 1 if ok, 0 if error.
// Nullable: error
MJAPI int mj_setLengthRange(mjModel* m, mjData* d, int index,
                            const mjLROpt* opt, char* error, int error_sz);

// Create empty spec.
MJAPI mjSpec* mj_makeSpec(void);

// Copy spec.
MJAPI mjSpec* mj_copySpec(const mjSpec* s);

// Free memory allocation in mjSpec.
MJAPI void mj_deleteSpec(mjSpec* s);

// Activate plugin; return 0 on success.
MJAPI int mjs_activatePlugin(mjSpec* s, const char* name);

// Turn deep copy on or off attach; return 0 on success.
MJAPI int mjs_setDeepCopy(mjSpec* s, int deepcopy);


//---------------------------------- Printing ------------------------------------------------------

// Print mjModel to text file, specifying format.
// float_format must be a valid printf-style format string for a single float value.
MJAPI void mj_printFormattedModel(const mjModel* m, const char* filename, const char* float_format);

// Print model to text file.
MJAPI void mj_printModel(const mjModel* m, const char* filename);

// Print mjData to text file, specifying format.
// float_format must be a valid printf-style format string for a single float value.
MJAPI void mj_printFormattedData(const mjModel* m, const mjData* d, const char* filename,
                                 const char* float_format);

// Print data to text file.
MJAPI void mj_printData(const mjModel* m, const mjData* d, const char* filename);

// Print matrix to screen.
MJAPI void mju_printMat(const mjtNum* mat, int nr, int nc);

// Print sparse matrix to screen.
MJAPI void mju_printMatSparse(const mjtNum* mat, int nr,
                              const int* rownnz, const int* rowadr, const int* colind);

// Print internal XML schema as plain text or HTML, with style-padding or &nbsp;.
MJAPI int mj_printSchema(const char* filename, char* buffer, int buffer_sz,
                         int flg_html, int flg_pad);

// Print scene to text file.
MJAPI void mj_printScene(const mjvScene* s, const char* filename);

// Print scene to text file, specifying format.
// float_format must be a valid printf-style format string for a single float value.
MJAPI void mj_printFormattedScene(const mjvScene* s, const char* filename,
                                  const char* float_format);


//---------------------------------- Components ----------------------------------------------------

// Run all kinematics-like computations (kinematics, comPos, camlight, flex, tendon).
MJAPI void mj_fwdKinematics(const mjModel* m, mjData* d);

// Run position-dependent computations.
MJAPI void mj_fwdPosition(const mjModel* m, mjData* d);

// Run velocity-dependent computations.
MJAPI void mj_fwdVelocity(const mjModel* m, mjData* d);

// Compute actuator force qfrc_actuator.
MJAPI void mj_fwdActuation(const mjModel* m, mjData* d);

// Add up all non-constraint forces, compute qacc_smooth.
MJAPI void mj_fwdAcceleration(const mjModel* m, mjData* d);

// Run selected constraint solver.
MJAPI void mj_fwdConstraint(const mjModel* m, mjData* d);

// Euler integrator, semi-implicit in velocity.
MJAPI void mj_Euler(const mjModel* m, mjData* d);

// Runge-Kutta explicit order-N integrator.
MJAPI void mj_RungeKutta(const mjModel* m, mjData* d, int N);

// Implicit-in-velocity integrators.
MJAPI void mj_implicit(const mjModel* m, mjData* d);

// Run position-dependent computations in inverse dynamics.
MJAPI void mj_invPosition(const mjModel* m, mjData* d);

// Run velocity-dependent computations in inverse dynamics.
MJAPI void mj_invVelocity(const mjModel* m, mjData* d);

// Apply the analytical formula for inverse constraint dynamics.
MJAPI void mj_invConstraint(const mjModel* m, mjData* d);

// Compare forward and inverse dynamics, save results in fwdinv.
MJAPI void mj_compareFwdInv(const mjModel* m, mjData* d);


//---------------------------------- Sub components ------------------------------------------------

// Evaluate position-dependent sensors.
MJAPI void mj_sensorPos(const mjModel* m, mjData* d);

// Evaluate velocity-dependent sensors.
MJAPI void mj_sensorVel(const mjModel* m, mjData* d);

// Evaluate acceleration and force-dependent sensors.
MJAPI void mj_sensorAcc(const mjModel* m, mjData* d);

// Evaluate position-dependent energy (potential).
MJAPI void mj_energyPos(const mjModel* m, mjData* d);

// Evaluate velocity-dependent energy (kinetic).
MJAPI void mj_energyVel(const mjModel* m, mjData* d);

// Check qpos, reset if any element is too big or nan.
MJAPI void mj_checkPos(const mjModel* m, mjData* d);

// Check qvel, reset if any element is too big or nan.
MJAPI void mj_checkVel(const mjModel* m, mjData* d);

// Check qacc, reset if any element is too big or nan.
MJAPI void mj_checkAcc(const mjModel* m, mjData* d);

// Run forward kinematics.
MJAPI void mj_kinematics(const mjModel* m, mjData* d);

// Map inertias and motion dofs to global frame centered at CoM.
MJAPI void mj_comPos(const mjModel* m, mjData* d);

// Compute camera and light positions and orientations.
MJAPI void mj_camlight(const mjModel* m, mjData* d);

// Compute flex-related quantities.
MJAPI void mj_flex(const mjModel* m, mjData* d);

// Compute tendon lengths, velocities and moment arms.
MJAPI void mj_tendon(const mjModel* m, mjData* d);

// Compute actuator transmission lengths and moments.
MJAPI void mj_transmission(const mjModel* m, mjData* d);

// Run composite rigid body inertia algorithm (CRB).
MJAPI void mj_crb(const mjModel* m, mjData* d);

// Make inertia matrix.
MJAPI void mj_makeM(const mjModel* m, mjData* d);

// Compute sparse L'*D*L factorizaton of inertia matrix.
MJAPI void mj_factorM(const mjModel* m, mjData* d);

// Solve linear system M * x = y using factorization:  x = inv(L'*D*L)*y
MJAPI void mj_solveM(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y, int n);

// Half of linear solve:  x = sqrt(inv(D))*inv(L')*y
MJAPI void mj_solveM2(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y,
                      const mjtNum* sqrtInvD, int n);

// Compute cvel, cdof_dot.
MJAPI void mj_comVel(const mjModel* m, mjData* d);

// Compute qfrc_passive from spring-dampers, gravity compensation and fluid forces.
MJAPI void mj_passive(const mjModel* m, mjData* d);

// Sub-tree linear velocity and angular momentum: compute subtree_linvel, subtree_angmom.
MJAPI void mj_subtreeVel(const mjModel* m, mjData* d);

// RNE: compute M(qpos)*qacc + C(qpos,qvel); flg_acc=0 removes inertial term.
MJAPI void mj_rne(const mjModel* m, mjData* d, int flg_acc, mjtNum* result);

// RNE with complete data: compute cacc, cfrc_ext, cfrc_int.
MJAPI void mj_rnePostConstraint(const mjModel* m, mjData* d);

// Return the maximum number of contacts that can be generated between two geoms.
// If has_margin is -1, then the margin is pulled from the model, otherwise if has_margin > 0
// indicates that the geoms have a positive margin.
MJAPI int mj_maxContact(const mjModel* m, int g1, int g2, int has_margin);

// Run collision detection.
MJAPI void mj_collision(const mjModel* m, mjData* d);

// Construct constraints.
MJAPI void mj_makeConstraint(const mjModel* m, mjData* d);

// Find constraint islands.
MJAPI void mj_island(const mjModel* m, mjData* d);

// Compute inverse constraint inertia efc_AR.
MJAPI void mj_projectConstraint(const mjModel* m, mjData* d);

// Compute efc_vel, efc_aref.
MJAPI void mj_referenceConstraint(const mjModel* m, mjData* d);

// Compute efc_state, efc_force, qfrc_constraint, and (optionally) cone Hessians.
// If cost is not NULL, set *cost = s(jar) where jar = Jac*qacc-aref.
// Nullable: cost
MJAPI void mj_constraintUpdate(const mjModel* m, mjData* d, const mjtNum* jar,
                               mjtNum cost[1], int flg_coneHessian);


//---------------------------------- Support -------------------------------------------------------

// Return size of state signature.
MJAPI int mj_stateSize(const mjModel* m, int sig);

// Get state.
MJAPI void mj_getState(const mjModel* m, const mjData* d, mjtNum* state, int sig);

// Extract a subset of components from a state previously obtained via mj_getState.
MJAPI void mj_extractState(const mjModel* m, const mjtNum* src, int srcsig,
                           mjtNum* dst, int dstsig);

// Set state.
MJAPI void mj_setState(const mjModel* m, mjData* d, const mjtNum* state, int sig);

// Copy state from src to dst.
MJAPI void mj_copyState(const mjModel* m, const mjData* src, mjData* dst, int sig);

// Read ctrl value for actuator at given time.
// Returns pointer to ctrl (no history) or history buffer (exact match),
// or NULL if interpolation performed (writes to result).
// interp: 0=zero-order-hold, 1=linear, 2=cubic spline.
MJAPI const mjtNum* mj_readCtrl(const mjModel* m, const mjData* d, int id, mjtNum time,
                                mjtNum* result, int interp);

// Read sensor value from history buffer at given time.
// Returns pointer to sensordata (no history) or history buffer (exact match),
// or NULL if interpolation performed (writes to result).
// interp: 0=zero-order-hold, 1=linear, 2=cubic spline.
MJAPI const mjtNum* mj_readSensor(const mjModel* m, const mjData* d, int id, mjtNum time,
                                  mjtNum* result, int interp);

// Initialize history buffer for actuator; if times is NULL, uses existing buffer timestamps.
// Nullable: times
MJAPI void mj_initCtrlHistory(const mjModel* m, mjData* d, int id,
                              const mjtNum* times, const mjtNum* values);

// Initialize history buffer for sensor; if times is NULL, uses existing buffer timestamps.
// phase sets the user slot (last computation time for interval sensors).
// Nullable: times
MJAPI void mj_initSensorHistory(const mjModel* m, mjData* d, int id,
                                const mjtNum* times, const mjtNum* values, mjtNum phase);

// Copy current state to the k-th model keyframe.
MJAPI void mj_setKeyframe(mjModel* m, const mjData* d, int k);

// Add contact to d->contact list; return 0 if success; 1 if buffer full.
MJAPI int mj_addContact(const mjModel* m, mjData* d, const mjContact* con);

// Determine type of friction cone.
MJAPI int mj_isPyramidal(const mjModel* m);

// Determine type of constraint Jacobian.
MJAPI int mj_isSparse(const mjModel* m);

// Determine type of solver (PGS is dual, CG and Newton are primal).
MJAPI int mj_isDual(const mjModel* m);

// Multiply dense or sparse constraint Jacobian by vector.
MJAPI void mj_mulJacVec(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

// Multiply dense or sparse constraint Jacobian transpose by vector.
MJAPI void mj_mulJacTVec(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

// Compute 3/6-by-nv end-effector Jacobian of global point attached to given body.
// Nullable: jacp, jacr
MJAPI void mj_jac(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr,
                  const mjtNum point[3], int body);

// Compute body frame end-effector Jacobian.
// Nullable: jacp, jacr
MJAPI void mj_jacBody(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int body);

// Compute body center-of-mass end-effector Jacobian.
// Nullable: jacp, jacr
MJAPI void mj_jacBodyCom(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int body);

// Compute subtree center-of-mass end-effector Jacobian.
MJAPI void mj_jacSubtreeCom(const mjModel* m, mjData* d, mjtNum* jacp, int body);

// Compute geom end-effector Jacobian.
// Nullable: jacp, jacr
MJAPI void mj_jacGeom(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int geom);

// Compute site end-effector Jacobian.
// Nullable: jacp, jacr
MJAPI void mj_jacSite(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int site);

// Compute translation end-effector Jacobian of point, and rotation Jacobian of axis.
// Nullable: jacPoint, jacAxis
MJAPI void mj_jacPointAxis(const mjModel* m, mjData* d, mjtNum* jacPoint, mjtNum* jacAxis,
                           const mjtNum point[3], const mjtNum axis[3], int body);

// Compute 3/6-by-nv Jacobian time derivative of global point attached to given body.
// Nullable: jacp, jacr
MJAPI void mj_jacDot(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr,
                     const mjtNum point[3], int body);

// Compute subtree angular momentum matrix.
MJAPI void mj_angmomMat(const mjModel* m, mjData* d, mjtNum* mat, int body);

// Get id of object with the specified mjtObj type and name; return -1 if id not found.
MJAPI int mj_name2id(const mjModel* m, int type, const char* name);

// Get name of object with the specified mjtObj type and id; return NULL if name not found.
MJAPI const char* mj_id2name(const mjModel* m, int type, int id);

// Get name of actuator input, determined by the actuator type and input signature;
// return NULL if the actuator type defines no input names.
MJAPI const char* mj_actuatorInputName(const mjModel* m, int id, int input);

// Convert sparse inertia matrix into full (i.e. dense) matrix.
MJAPI void mj_fullM(const mjModel* m, const mjData* d, mjtNum* dst);

// Multiply vector by inertia matrix.
MJAPI void mj_mulM(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

// Multiply vector by (inertia matrix)^(1/2).
MJAPI void mj_mulM2(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

// Add inertia matrix to destination matrix (lower triangle only).
// Destination can be sparse or dense when all int* are NULL.
// Nullable: rownnz, rowadr, colind
MJAPI void mj_addM(const mjModel* m, mjData* d, mjtNum* dst, int* rownnz, int* rowadr, int* colind);

// Apply Cartesian force and torque (outside xfrc_applied mechanism).
// Nullable: force, torque
MJAPI void mj_applyFT(const mjModel* m, mjData* d, const mjtNum force[3], const mjtNum torque[3],
                      const mjtNum point[3], int body, mjtNum* qfrc_target);

// Compute object 6D velocity (rot:lin) in object-centered frame, world/local orientation.
MJAPI void mj_objectVelocity(const mjModel* m, const mjData* d,
                             int objtype, int objid, mjtNum res[6], int flg_local);

// Compute object 6D acceleration (rot:lin) in object-centered frame, world/local orientation.
MJAPI void mj_objectAcceleration(const mjModel* m, const mjData* d,
                                 int objtype, int objid, mjtNum res[6], int flg_local);

// Return smallest signed distance between two geoms and optionally segment from geom1 to geom2.
// Nullable: fromto
MJAPI mjtNum mj_geomDistance(const mjModel* m, mjData* d, int geom1, int geom2, mjtNum distmax,
                             mjtNum fromto[6]);

// Return 1 if point is inside a site (convex hull for meshes), 0 otherwise.
MJAPI int mj_insideSite(const mjModel* m, const mjData* d, int siteid, const mjtNum point[3]);

// Extract 6D force:torque given contact id, in the contact frame.
MJAPI void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]);

// Compute velocity by finite-differencing two positions.
MJAPI void mj_differentiatePos(const mjModel* m, mjtNum* qvel, mjtNum dt,
                               const mjtNum* qpos1, const mjtNum* qpos2);

// Integrate position with given velocity.
MJAPI void mj_integratePos(const mjModel* m, mjtNum* qpos, const mjtNum* qvel, mjtNum dt);

// Normalize all quaternions in qpos-type vector.
MJAPI void mj_normalizeQuat(const mjModel* m, mjtNum* qpos);

// Map from body local to global Cartesian coordinates, sameframe takes values from mjtSameFrame.
MJAPI void mj_local2Global(mjData* d, mjtNum xpos[3], mjtNum xmat[9], const mjtNum pos[3],
                           const mjtNum quat[4], int body, mjtByte sameframe);

// Sum all body masses.
MJAPI mjtNum mj_getTotalmass(const mjModel* m);

// Scale body masses and inertias to achieve specified total mass.
MJAPI void mj_setTotalmass(mjModel* m, mjtNum newmass);

// Return a config attribute value of a plugin instance;
// NULL: invalid plugin instance ID or attribute name
MJAPI const char* mj_getPluginConfig(const mjModel* m, int plugin_id, const char* attrib);

// Load a dynamic library. The dynamic library is assumed to register one or more plugins.
MJAPI void mj_loadPluginLibrary(const char* path);

// Scan a directory and load all dynamic libraries. Dynamic libraries in the specified directory
// are assumed to register one or more plugins. Optionally, if a callback is specified, it is called
// for each dynamic library encountered that registers plugins.
MJAPI void mj_loadAllPluginLibraries(const char* directory, mjfPluginLibraryLoadCallback callback);

// Return version number: 1.0.2 is encoded as 102.
MJAPI int mj_version(void);

// Return the current version of MuJoCo as a null-terminated string.
MJAPI const char* mj_versionString(void);


//---------------------------------- Ray casting ---------------------------------------------------

// Intersect ray (pnt+x*vec, x>=0) with visible geoms, except geoms in bodyexclude.
// Return distance (x) to nearest surface, or -1 if no intersection.
// geomgroup, flg_static are as in mjvOption; geomgroup==NULL skips group exclusion.
// Nullable: geomgroup, geomid, normal
MJAPI mjtNum mj_ray(const mjModel* m, const mjData* d, const mjtNum pnt[3], const mjtNum vec[3],
                    const mjtByte* geomgroup, mjtBool flg_static, int bodyexclude,
                    int geomid[1], mjtNum normal[3]);

// Intersect multiple rays emanating from a single point, compute normals if given.
// Similar semantics to mj_ray, but vec, normal and dist are arrays.
// Geoms further than cutoff are ignored.
// Nullable: geomgroup, geomid, normal
MJAPI void mj_multiRay(const mjModel* m, mjData* d, const mjtNum pnt[3], const mjtNum* vec,
                       const mjtByte* geomgroup, mjtBool flg_static, int bodyexclude,
                       int* geomid, mjtNum* dist, mjtNum* normal, int nray, mjtNum cutoff);

// Intersect ray with hfield; return nearest distance or -1 if no intersection.
// Nullable: normal
MJAPI mjtNum mj_rayHfield(const mjModel* m, const mjData* d, int geomid,
                          const mjtNum pnt[3], const mjtNum vec[3], mjtNum normal[3]);

// Intersect ray with mesh; return nearest distance or -1 if no intersection.
// Nullable: normal
MJAPI mjtNum mj_rayMesh(const mjModel* m, const mjData* d, int geomid,
                        const mjtNum pnt[3], const mjtNum vec[3], mjtNum normal[3]);

// Intersect ray with pure geom; return nearest distance or -1 if no intersection.
// Nullable: normal
MJAPI mjtNum mju_rayGeom(const mjtNum pos[3], const mjtNum mat[9], const mjtNum size[3],
                         const mjtNum pnt[3], const mjtNum vec[3], int geomtype,
                         mjtNum normal[3]);

// Intersect ray with flex; return nearest distance or -1 if no intersection,
// and also output nearest vertex id and surface normal.
// Nullable: vertid, normal
MJAPI mjtNum mj_rayFlex(const mjModel* m, const mjData* d, int flex_layer,
                        mjtBool flg_vert, mjtBool flg_edge, mjtBool flg_face,
                        mjtBool flg_skin, int flexid, const mjtNum pnt[3],
                        const mjtNum vec[3], int vertid[1], mjtNum normal[3]);

// Intersect ray with skin; return nearest distance or -1 if no intersection,
// and also output nearest vertex id.
// Nullable: vertid
MJAPI mjtNum mju_raySkin(int nface, int nvert, const int* face, const float* vert,
                         const mjtNum pnt[3], const mjtNum vec[3], int vertid[1]);

//---------------------------------- Interaction ---------------------------------------------------

// Set default camera.
MJAPI void mjv_defaultCamera(mjvCamera* cam);

// Set default free camera.
MJAPI void mjv_defaultFreeCamera(const mjModel* m, mjvCamera* cam);

// Set default perturbation.
MJAPI void mjv_defaultPerturb(mjvPerturb* pert);

// Transform pose from room to model space.
MJAPI void mjv_room2model(mjtNum modelpos[3], mjtNum modelquat[4], const mjtNum roompos[3],
                          const mjtNum roomquat[4], const mjvScene* scn);

// Transform pose from model to room space.
MJAPI void mjv_model2room(mjtNum roompos[3], mjtNum roomquat[4], const mjtNum modelpos[3],
                          const mjtNum modelquat[4], const mjvScene* scn);

// Get camera info in model space; average left and right OpenGL cameras.
MJAPI void mjv_cameraInModel(mjtNum headpos[3], mjtNum forward[3], mjtNum up[3],
                             const mjvScene* scn);

// Get camera info in room space; average left and right OpenGL cameras.
MJAPI void mjv_cameraInRoom(mjtNum headpos[3], mjtNum forward[3], mjtNum up[3],
                            const mjvScene* scn);

// Get frustum height at unit distance from camera; average left and right OpenGL cameras.
MJAPI mjtNum mjv_frustumHeight(const mjvScene* scn);

// Rotate 3D vec in horizontal plane by angle between (0,1) and (forward_x,forward_y).
MJAPI void mjv_alignToCamera(mjtNum res[3], const mjtNum vec[3], const mjtNum forward[3]);

// Move camera with mouse; action is mjtMouse.
MJAPI void mjv_moveCamera(const mjModel* m, int action, mjtNum reldx, mjtNum reldy, mjvCamera* cam);

// Move perturb object with mouse; action is mjtMouse.
MJAPI void mjv_movePerturb(const mjModel* m, const mjData* d, int action, mjtNum reldx,
                           mjtNum reldy, const mjvScene* scn, mjvPerturb* pert);

// Move model with mouse; action is mjtMouse.
MJAPI void mjv_moveModel(const mjModel* m, int action, mjtNum reldx, mjtNum reldy,
                         const mjtNum roomup[3], mjvScene* scn);

// Copy perturb pos,quat from selected body; set scale for perturbation.
MJAPI void mjv_initPerturb(const mjModel* m, mjData* d, const mjvScene* scn, mjvPerturb* pert);

// Set perturb pos,quat in d->mocap when selected body is mocap, and in d->qpos otherwise.
// Write d->qpos only if flg_paused and subtree root for selected body has free joint.
MJAPI void mjv_applyPerturbPose(const mjModel* m, mjData* d, const mjvPerturb* pert,
                                int flg_paused);

// Set perturb force,torque in d->xfrc_applied, if selected body is dynamic.
MJAPI void mjv_applyPerturbForce(const mjModel* m, mjData* d, const mjvPerturb* pert);

// Return the average of two OpenGL cameras.
MJAPI mjvGLCamera mjv_averageCamera(const mjvGLCamera* cam1, const mjvGLCamera* cam2);

// Converts a mjvCamera to a mjvGLCamera.
MJAPI mjvGLCamera mjv_camera2GLCamera(const mjModel* model, const mjData* data,
                                      const mjvCamera* mjv_camera);

// Select geom, flex or skin with mouse; return bodyid; -1: none selected.
// Nullable: geomid, flexid, skinid
MJAPI int mjv_select(const mjModel* m, const mjData* d, const mjvOption* vopt,
                     mjtNum aspectratio, mjtNum relx, mjtNum rely,
                     const mjvScene* scn, mjtNum selpnt[3],
                     int geomid[1], int flexid[1], int skinid[1]);


//---------------------------------- Visualization -------------------------------------------------

// Set default visualization options.
MJAPI void mjv_defaultOption(mjvOption* opt);

// Set default figure.
MJAPI void mjv_defaultFigure(mjvFigure* fig);

// Initialize given geom fields when not NULL, set the rest to their default values.
// Nullable: size, pos, mat, rgba
MJAPI void mjv_initGeom(mjvGeom* geom, int type, const mjtNum size[3],
                        const mjtNum pos[3], const mjtNum mat[9], const float rgba[4]);

// Set (type, size, pos, mat) for connector-type geom between given points.
// Assume that mjv_initGeom was already called to set all other properties.
// Width of mjGEOM_LINE is denominated in pixels.
MJAPI void mjv_connector(mjvGeom* geom, int type, mjtNum width,
                         const mjtNum from[3], const mjtNum to[3]);

// Set default abstract scene.
MJAPI void mjv_defaultScene(mjvScene* scn);

// Allocate resources in abstract scene.
MJAPI void mjv_makeScene(const mjModel* m, mjvScene* scn, int maxgeom);

// Free abstract scene.
MJAPI void mjv_freeScene(mjvScene* scn);

// Update entire scene given model state.
MJAPI void mjv_updateScene(const mjModel* m, mjData* d, const mjvOption* opt,
                           const mjvPerturb* pert, mjvCamera* cam, int catmask, mjvScene* scn);

// Copy mjModel, skip large arrays not required for abstract visualization.
// Nullable: dest
MJAPI void mjv_copyModel(mjModel* dest, const mjModel* src);

// Add geoms from selected categories.
MJAPI void mjv_addGeoms(const mjModel* m, mjData* d, const mjvOption* opt,
                        const mjvPerturb* pert, int catmask, mjvScene* scn);

// Make list of lights.
MJAPI void mjv_makeLights(const mjModel* m, const mjData* d, mjvScene* scn);

// Update camera.
MJAPI void mjv_updateCamera(const mjModel* m, const mjData* d, mjvCamera* cam, mjvScene* scn);

// Update skins.
MJAPI void mjv_updateSkin(const mjModel* m, const mjData* d, mjvScene* scn);

// Compute camera position and forward, up, and right vectors.
// Nullable: headpos, forward, up, right
MJAPI void mjv_cameraFrame(mjtNum headpos[3], mjtNum forward[3], mjtNum up[3], mjtNum right[3],
                           const mjData* d, const mjvCamera* cam);

// Compute camera frustum: vertical, horizontal, and clip planes.
// Nullable: zver, zhor, zclip
MJAPI void mjv_cameraFrustum(float zver[2], float zhor[2], float zclip[2],  const mjModel* m,
                             const mjvCamera* cam);


//---------------------------------- OpenGL rendering ----------------------------------------------

// Set default mjrContext.
MJAPI void mjr_defaultContext(mjrContext* con);

// Set default mjrRendererInfo.
MJAPI void mjr_defaultRendererInfo(mjrRendererInfo* info);

// Get active renderer information.
MJAPI void mjr_getRendererInfo(mjrRendererInfo* info);

// Allocate resources in custom OpenGL context; fontscale is mjtFontScale.
MJAPI void mjr_makeContext(const mjModel* m, mjrContext* con, int fontscale);

// Change font of existing context.
MJAPI void mjr_changeFont(int fontscale, mjrContext* con);

// Add Aux buffer with given index to context; free previous Aux buffer.
MJAPI void mjr_addAux(int index, int width, int height, int samples, mjrContext* con);

// Free resources in custom OpenGL context, set to default.
MJAPI void mjr_freeContext(mjrContext* con);

// Resize offscreen buffers.
MJAPI void mjr_resizeOffscreen(int width, int height, mjrContext* con);

// Upload texture to GPU, overwriting previous upload if any. A by-reference texture (tex_adr == -1)
// is decoded with mj_loadTexture.
MJAPI void mjr_uploadTexture(const mjModel* m, const mjrContext* con, int texid);

// Upload mesh to GPU, overwriting previous upload if any.
MJAPI void mjr_uploadMesh(const mjModel* m, const mjrContext* con, int meshid);

// Upload height field to GPU, overwriting previous upload if any.
MJAPI void mjr_uploadHField(const mjModel* m, const mjrContext* con, int hfieldid);

// Make con->currentBuffer current again.
MJAPI void mjr_restoreBuffer(const mjrContext* con);

// Set OpenGL framebuffer for rendering: mjFB_WINDOW or mjFB_OFFSCREEN.
// If only one buffer is available, set that buffer and ignore framebuffer argument.
MJAPI void mjr_setBuffer(int framebuffer, mjrContext* con);

// Read pixels from current OpenGL framebuffer to client buffer.
// Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
MJAPI void mjr_readPixels(unsigned char* rgb, float* depth,
                          mjrRect viewport, const mjrContext* con);

// Draw pixels from client buffer to current OpenGL framebuffer.
// Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
MJAPI void mjr_drawPixels(const unsigned char* rgb, const float* depth,
                          mjrRect viewport, const mjrContext* con);

// Blit from src viewpoint in current framebuffer to dst viewport in other framebuffer.
// If src, dst have different size and flg_depth==0, color is interpolated with GL_LINEAR.
MJAPI void mjr_blitBuffer(mjrRect src, mjrRect dst,
                          int flg_color, int flg_depth, const mjrContext* con);

// Set Aux buffer for custom OpenGL rendering (call restoreBuffer when done).
MJAPI void mjr_setAux(int index, const mjrContext* con);

// Blit from Aux buffer to con->currentBuffer.
MJAPI void mjr_blitAux(int index, mjrRect src, int left, int bottom, const mjrContext* con);

// Draw text at (x,y) in relative coordinates; font is mjtFont.
MJAPI void mjr_text(int font, const char* txt, const mjrContext* con,
                    float x, float y, float r, float g, float b);

// Draw text overlay; font is mjtFont; gridpos is mjtGridPos.
MJAPI void mjr_overlay(int font, int gridpos, mjrRect viewport,
                       const char* overlay, const char* overlay2, const mjrContext* con);

// Get maximum viewport for active buffer.
MJAPI mjrRect mjr_maxViewport(const mjrContext* con);

// Draw rectangle.
MJAPI void mjr_rectangle(mjrRect viewport, float r, float g, float b, float a);

// Draw rectangle with centered text.
MJAPI void mjr_label(mjrRect viewport, int font, const char* txt,
                     float r, float g, float b, float a, float rt, float gt, float bt,
                     const mjrContext* con);

// Draw 2D figure.
MJAPI void mjr_figure(mjrRect viewport, mjvFigure* fig, const mjrContext* con);

// Render 3D scene.
MJAPI void mjr_render(mjrRect viewport, mjvScene* scn, const mjrContext* con);

// Call glFinish.
MJAPI void mjr_finish(void);

// Call glGetError and return result.
MJAPI int mjr_getError(void);

// Find first rectangle containing mouse, -1: not found.
MJAPI int mjr_findRect(int x, int y, int nrect, const mjrRect* rect);


//---------------------------------- UI framework --------------------------------------------------

// Get builtin UI theme spacing (ind: 0-1).
MJAPI mjuiThemeSpacing mjui_themeSpacing(int ind);

// Get builtin UI theme color (ind: 0-3).
MJAPI mjuiThemeColor mjui_themeColor(int ind);

// Add definitions to UI.
MJAPI void mjui_add(mjUI* ui, const mjuiDef* def);

// Add definitions to UI section.
MJAPI void mjui_addToSection(mjUI* ui, int sect, const mjuiDef* def);

// Compute UI sizes.
MJAPI void mjui_resize(mjUI* ui, const mjrContext* con);

// Update specific section/item; -1: update all.
MJAPI void mjui_update(int section, int item, const mjUI* ui,
                       const mjuiState* state, const mjrContext* con);

// Handle UI event; return pointer to changed item, NULL if no change.
MJAPI mjuiItem* mjui_event(mjUI* ui, mjuiState* state, const mjrContext* con);

// Copy UI image to current buffer.
MJAPI void mjui_render(mjUI* ui, const mjuiState* state, const mjrContext* con);


//---------------------------------- Error and memory ----------------------------------------------

// Main error function; does not return to caller.
MJAPI void mju_error(const char* msg, ...) mjPRINTFLIKE(1, 2);

// Main warning function; returns to caller.
MJAPI void mju_warning(const char* msg, ...) mjPRINTFLIKE(1, 2);

// Clear user error and memory handlers.
MJAPI void mju_clearHandlers(void);

// Set the active log handler; return the previous handler.
// If handler is NULL, restore the default handler.
MJAPI mjfLogHandler mju_setLogHandler(mjfLogHandler handler);

// Get default handler configuration.
MJAPI mjLogConfig mju_getLogConfig(void);

// Set default handler configuration.
MJAPI void mju_setLogConfig(mjLogConfig config);

// Log an info message with optional topic filtering.
MJAPI void mju_info(int topic, const char* msg, ...) mjPRINTFLIKE(2, 3);

// Dispatch a structured log message to the active handler.
MJAPI void mju_message(const mjLogMessage* msg);

// Allocate memory; byte-align on 64; pad size to multiple of 64.
MJAPI void* mju_malloc(size_t size);

// Free memory, using free() by default.
MJAPI void mju_free(void* ptr);

// High-level warning function: count warnings in mjData, print only the first.
MJAPI void mj_warning(mjData* d, int warning, int info);

// Write [datetime, type: message] to MUJOCO_LOG.TXT.
MJAPI void mju_writeLog(const char* type, const char* msg);

// Get compiler error message from spec.
MJAPI const char* mjs_getError(mjSpec* s);

// Get compiler timing diagnostics from spec, returns pointer to array of size mjNCTIMER.
MJAPI const double* mjs_getTimer(mjSpec* s);

// Return 1 if compiler error is a warning. Deprecated: use mjs_numWarnings(s) > 0.
MJAPI int mjs_isWarning(mjSpec* s);

// Get number of warnings accumulated in the spec.
MJAPI int mjs_numWarnings(const mjSpec* spec);

// Get the i-th warning message (returns nullptr if index out of bounds).
MJAPI const char* mjs_getWarning(const mjSpec* spec, int index);

//---------------------------------- Standard math -------------------------------------------------

#if !defined(mjUSESINGLE)
  #define mju_sqrt    sqrt
  #define mju_exp     exp
  #define mju_sin     sin
  #define mju_cos     cos
  #define mju_tan     tan
  #define mju_asin    asin
  #define mju_acos    acos
  #define mju_atan2   atan2
  #define mju_tanh    tanh
  #define mju_pow     pow
  #define mju_abs     fabs
  #define mju_log     log
  #define mju_log10   log10
  #define mju_floor   floor
  #define mju_ceil    ceil

#else
  #define mju_sqrt    sqrtf
  #define mju_exp     expf
  #define mju_sin     sinf
  #define mju_cos     cosf
  #define mju_tan     tanf
  #define mju_asin    asinf
  #define mju_acos    acosf
  #define mju_atan2   atan2f
  #define mju_tanh    tanhf
  #define mju_pow     powf
  #define mju_abs     fabsf
  #define mju_log     logf
  #define mju_log10   log10f
  #define mju_floor   floorf
  #define mju_ceil    ceilf
#endif


//---------------------------------- Vector math ---------------------------------------------------

// Set res = 0.
MJAPI void mju_zero3(mjtNum res[3]);

// Set res = vec.
MJAPI void mju_copy3(mjtNum res[3], const mjtNum data[3]);

// Set res = vec*scl.
MJAPI void mju_scl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl);

// Set res = vec1 + vec2.
MJAPI void mju_add3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3]);

// Set res = vec1 - vec2.
MJAPI void mju_sub3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3]);

// Set res = res + vec.
MJAPI void mju_addTo3(mjtNum res[3], const mjtNum vec[3]);

// Set res = res - vec.
MJAPI void mju_subFrom3(mjtNum res[3], const mjtNum vec[3]);

// Set res = res + vec*scl.
MJAPI void mju_addToScl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl);

// Set res = vec1 + vec2*scl.
MJAPI void mju_addScl3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3], mjtNum scl);

// Normalize vector; return length before normalization.
MJAPI mjtNum mju_normalize3(mjtNum vec[3]);

// Return vector length (without normalizing the vector).
MJAPI mjtNum mju_norm3(const mjtNum vec[3]);

// Return dot-product of vec1 and vec2.
MJAPI mjtNum mju_dot3(const mjtNum vec1[3], const mjtNum vec2[3]);

// Return Cartesian distance between 3D vectors pos1 and pos2.
MJAPI mjtNum mju_dist3(const mjtNum pos1[3], const mjtNum pos2[3]);

// Multiply 3-by-3 matrix by vector: res = mat * vec.
MJAPI void mju_mulMatVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);

// Multiply transposed 3-by-3 matrix by vector: res = mat' * vec.
MJAPI void mju_mulMatTVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);

// Compute cross-product: res = cross(a, b).
MJAPI void mju_cross(mjtNum res[3], const mjtNum a[3], const mjtNum b[3]);

// Set res = 0.
MJAPI void mju_zero4(mjtNum res[4]);

// Set res = (1,0,0,0).
MJAPI void mju_unit4(mjtNum res[4]);

// Set res = vec.
MJAPI void mju_copy4(mjtNum res[4], const mjtNum data[4]);

// Normalize vector; return length before normalization.
MJAPI mjtNum mju_normalize4(mjtNum vec[4]);

// Set res = 0.
MJAPI void mju_zero(mjtNum* res, int n);

// Set res = val.
MJAPI void mju_fill(mjtNum* res, mjtNum val, int n);

// Set res = vec.
MJAPI void mju_copy(mjtNum* res, const mjtNum* vec, int n);

// Return sum(vec).
MJAPI mjtNum mju_sum(const mjtNum* vec, int n);

// Return L1 norm: sum(abs(vec)).
MJAPI mjtNum mju_L1(const mjtNum* vec, int n);

// Set res = vec*scl.
MJAPI void mju_scl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n);

// Set res = vec1 + vec2.
MJAPI void mju_add(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n);

// Set res = vec1 - vec2.
MJAPI void mju_sub(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n);

// Set res = res + vec.
MJAPI void mju_addTo(mjtNum* res, const mjtNum* vec, int n);

// Set res = res - vec.
MJAPI void mju_subFrom(mjtNum* res, const mjtNum* vec, int n);

// Set res = res + vec*scl.
MJAPI void mju_addToScl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n);

// Set res = vec1 + vec2*scl.
MJAPI void mju_addScl(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, mjtNum scl, int n);

// Normalize vector; return length before normalization.
MJAPI mjtNum mju_normalize(mjtNum* res, int n);

// Return vector length (without normalizing vector).
MJAPI mjtNum mju_norm(const mjtNum* res, int n);

// Return dot-product of vec1 and vec2.
MJAPI mjtNum mju_dot(const mjtNum* vec1, const mjtNum* vec2, int n);

// Multiply matrix and vector: res = mat * vec.
MJAPI void mju_mulMatVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc);

// Multiply transposed matrix and vector: res = mat' * vec.
MJAPI void mju_mulMatTVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc);

// Multiply square matrix with vectors on both sides: return vec1' * mat * vec2.
MJAPI mjtNum mju_mulVecMatVec(const mjtNum* vec1, const mjtNum* mat, const mjtNum* vec2, int n);

// Transpose matrix: res = mat'.
MJAPI void mju_transpose(mjtNum* res, const mjtNum* mat, int nr, int nc);

// Symmetrize square matrix res = (mat + mat')/2.
MJAPI void mju_symmetrize(mjtNum* res, const mjtNum* mat, int n);

// Set mat to the identity matrix.
MJAPI void mju_eye(mjtNum* mat, int n);

// Multiply matrices: res = mat1 * mat2.
MJAPI void mju_mulMatMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                         int r1, int c1, int c2);

// Multiply matrices, second argument transposed: res = mat1 * mat2'.
MJAPI void mju_mulMatMatT(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                          int r1, int c1, int r2);

// Multiply matrices, first argument transposed: res = mat1' * mat2.
MJAPI void mju_mulMatTMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                          int r1, int c1, int c2);

// Set res = mat' * diag * mat if diag is not NULL, and res = mat' * mat otherwise.
MJAPI void mju_sqrMatTD(mjtNum* res, const mjtNum* mat, const mjtNum* diag, int nr, int nc);

// Coordinate transform of 6D motion or force vector in rotation:translation format.
// rotnew2old is 3-by-3, NULL means no rotation; flg_force specifies force or motion type.
// Nullable: rotnew2old
MJAPI void mju_transformSpatial(mjtNum res[6], const mjtNum vec[6], int flg_force,
                                const mjtNum newpos[3], const mjtNum oldpos[3],
                                const mjtNum rotnew2old[9]);


//---------------------------------- Sparse math ---------------------------------------------------

// Convert matrix from dense to sparse.
//  nnz is size of res and colind; return 1 if too small, 0 otherwise.
MJAPI int mju_dense2sparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                           int* rownnz, int* rowadr, int* colind, int nnz);

// Convert matrix from sparse to dense.
MJAPI void mju_sparse2dense(mjtNum* res, const mjtNum* mat, int nr, int nc,
                            const int* rownnz, const int* rowadr, const int* colind);

// Convert lower-triangular symmetric CSR matrix to full dense matrix.
MJAPI void mju_sym2dense(mjtNum* res, const mjtNum* mat, int n,
                         const int* rownnz, const int* rowadr, const int* colind);


//---------------------------------- Quaternions ---------------------------------------------------

// Rotate vector by quaternion.
MJAPI void mju_rotVecQuat(mjtNum res[3], const mjtNum vec[3], const mjtNum quat[4]);

// Conjugate quaternion, corresponding to opposite rotation.
MJAPI void mju_negQuat(mjtNum res[4], const mjtNum quat[4]);

// Multiply quaternions.
MJAPI void mju_mulQuat(mjtNum res[4], const mjtNum quat1[4], const mjtNum quat2[4]);

// Multiply quaternion and axis.
MJAPI void mju_mulQuatAxis(mjtNum res[4], const mjtNum quat[4], const mjtNum axis[3]);

// Convert axisAngle to quaternion.
MJAPI void mju_axisAngle2Quat(mjtNum res[4], const mjtNum axis[3], mjtNum angle);

// Convert quaternion (corresponding to orientation difference) to 3D velocity.
MJAPI void mju_quat2Vel(mjtNum res[3], const mjtNum quat[4], mjtNum dt);

// Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
MJAPI void mju_subQuat(mjtNum res[3], const mjtNum qa[4], const mjtNum qb[4]);

// Convert quaternion to 3D rotation matrix.
MJAPI void mju_quat2Mat(mjtNum res[9], const mjtNum quat[4]);

// Convert 3D rotation matrix to quaternion.
MJAPI void mju_mat2Quat(mjtNum quat[4], const mjtNum mat[9]);

// Compute time-derivative of quaternion, given 3D rotational velocity.
MJAPI void mju_derivQuat(mjtNum res[4], const mjtNum quat[4], const mjtNum vel[3]);

// Integrate quaternion given 3D angular velocity.
MJAPI void mju_quatIntegrate(mjtNum quat[4], const mjtNum vel[3], mjtNum scale);

// Construct quaternion performing rotation from z-axis to given vector.
MJAPI void mju_quatZ2Vec(mjtNum quat[4], const mjtNum vec[3]);

// Extract 3D rotation from an arbitrary 3x3 matrix by refining the input quaternion.
// Return the number of iterations required to converge.
MJAPI int mju_mat2Rot(mjtNum quat[4], const mjtNum mat[9]);

// Convert sequence of Euler angles (radians) to quaternion.
// seq[0,1,2] must be in 'xyzXYZ', lower/upper-case mean intrinsic/extrinsic rotations.
MJAPI void mju_euler2Quat(mjtNum quat[4], const mjtNum euler[3], const char* seq);


//---------------------------------- Poses ---------------------------------------------------------

// Multiply two poses.
MJAPI void mju_mulPose(mjtNum posres[3], mjtNum quatres[4],
                       const mjtNum pos1[3], const mjtNum quat1[4],
                       const mjtNum pos2[3], const mjtNum quat2[4]);

// Conjugate pose, corresponding to the opposite spatial transformation.
MJAPI void mju_negPose(mjtNum posres[3], mjtNum quatres[4],
                       const mjtNum pos[3], const mjtNum quat[4]);

// Transform vector by pose.
MJAPI void mju_trnVecPose(mjtNum res[3], const mjtNum pos[3], const mjtNum quat[4],
                          const mjtNum vec[3]);


//--------------------------------- Decompositions / Solvers ---------------------------------------

// Cholesky decomposition: mat = L*L'; return rank, decomposition performed in-place into mat.
MJAPI int mju_cholFactor(mjtNum* mat, int n, mjtNum mindiag);

// Solve (mat*mat') * res = vec, where mat is a Cholesky factor.
MJAPI void mju_cholSolve(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n);

// Cholesky rank-one update: L*L' +/- x*x'; return rank.
MJAPI int mju_cholUpdate(mjtNum* mat, mjtNum* x, int n, int flg_plus);

// Band-dense Cholesky decomposition.
//  Return minimum value in the factorized diagonal, or 0 if rank-deficient.
//  mat has (ntotal-ndense) x nband + ndense x ntotal elements.
//  The first (ntotal-ndense) x nband store the band part, left of diagonal, inclusive.
//  The second ndense x ntotal store the band part as entire dense rows.
//  Add diagadd+diagmul*mat_ii to diagonal before factorization.
MJAPI mjtNum mju_cholFactorBand(mjtNum* mat, int ntotal, int nband, int ndense,
                                mjtNum diagadd, mjtNum diagmul);

// Solve (mat*mat')*res = vec where mat is a band-dense Cholesky factor.
MJAPI void mju_cholSolveBand(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense);

// Convert banded matrix to dense matrix, fill upper triangle if flg_sym>0.
MJAPI void mju_band2Dense(mjtNum* res, const mjtNum* mat, int ntotal, int nband, int ndense,
                          mjtBool flg_sym);

// Convert dense matrix to banded matrix.
MJAPI void mju_dense2Band(mjtNum* res, const mjtNum* mat, int ntotal, int nband, int ndense);

// Multiply band-diagonal matrix with nvec vectors, include upper triangle if flg_sym>0.
MJAPI void mju_bandMulMatVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense, int nvec, mjtBool flg_sym);

// Address of diagonal element i in band-dense matrix representation.
MJAPI int mju_bandDiag(int i, int ntotal, int nband, int ndense);

// Eigenvalue decomposition of symmetric 3x3 matrix, mat = eigvec * diag(eigval) * eigvec'.
MJAPI int mju_eig3(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9]);

// minimize 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper; return rank or -1 if failed
//   inputs:
//     n           - problem dimension
//     H           - SPD matrix                n*n
//     g           - bias vector               n
//     lower       - lower bounds              n
//     upper       - upper bounds              n
//     res         - solution warmstart        n
//   return value:
//     nfree <= n  - rank of unconstrained subspace, -1 if failure
//   outputs (required):
//     res         - solution                  n
//     R           - subspace Cholesky factor  nfree*nfree    allocated: n*(n+7)
//   outputs (optional):
//     index       - set of free dimensions    nfree          allocated: n
//   notes:
//     the initial value of res is used to warmstart the solver
//     R must have allocatd size n*(n+7), but only nfree*nfree values are used in output
//     index (if given) must have allocated size n, but only nfree values are used in output
//     only the lower triangles of H and R and are read from and written to, respectively
//     the convenience function mju_boxQPmalloc allocates the required data structures
// Nullable: index, lower, upper
MJAPI int mju_boxQP(mjtNum* res, mjtNum* R, int* index, const mjtNum* H, const mjtNum* g, int n,
                    const mjtNum* lower, const mjtNum* upper);

// allocate heap memory for box-constrained Quadratic Program
//   as in mju_boxQP, index, lower, and upper are optional
//   free all pointers with mju_free()
MJAPI void mju_boxQPmalloc(mjtNum** res, mjtNum** R, int** index, mjtNum** H, mjtNum** g, int n,
                           mjtNum** lower, mjtNum** upper);


//---------------------------------- Miscellaneous -------------------------------------------------

// Muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax).
MJAPI mjtNum mju_muscleGain(mjtNum len, mjtNum vel, const mjtNum lengthrange[2],
                            mjtNum acc0, const mjtNum prm[9]);

// Muscle passive force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax).
MJAPI mjtNum mju_muscleBias(mjtNum len, const mjtNum lengthrange[2],
                            mjtNum acc0, const mjtNum prm[9]);

// Muscle activation dynamics, prm = (tau_act, tau_deact, smoothing_width).
MJAPI mjtNum mju_muscleDynamics(mjtNum ctrl, mjtNum act, const mjtNum prm[3]);

// Convert contact force to pyramid representation.
MJAPI void mju_encodePyramid(mjtNum* pyramid, const mjtNum* force, const mjtNum* mu, int dim);

// Convert pyramid representation to contact force.
MJAPI void mju_decodePyramid(mjtNum* force, const mjtNum* pyramid, const mjtNum* mu, int dim);

// Integrate spring-damper analytically; return pos(dt).
MJAPI mjtNum mju_springDamper(mjtNum pos0, mjtNum vel0, mjtNum Kp, mjtNum Kv, mjtNum dt);

// Return min(a,b) with single evaluation of a and b.
MJAPI mjtNum mju_min(mjtNum a, mjtNum b);

// Return max(a,b) with single evaluation of a and b.
MJAPI mjtNum mju_max(mjtNum a, mjtNum b);

// Clip x to the range [min, max].
MJAPI mjtNum mju_clip(mjtNum x, mjtNum min, mjtNum max);

// Return sign of x: +1, -1 or 0.
MJAPI mjtNum mju_sign(mjtNum x);

// Round x to nearest integer.
MJAPI int mju_round(mjtNum x);

// Convert type id (mjtObj) to type name.
MJAPI const char* mju_type2Str(int type);

// Convert type name to type id (mjtObj).
MJAPI int mju_str2Type(const char* str);

// Return human readable number of bytes using standard letter suffix.
MJAPI const char* mju_writeNumBytes(size_t nbytes);

// Construct a warning message given the warning type and info.
MJAPI const char* mju_warningText(int warning, size_t info);

// Return 1 if nan or abs(x)>mjMAXVAL, 0 otherwise. Used by check functions.
MJAPI int mju_isBad(mjtNum x);

// Return 1 if all elements are 0.
MJAPI int mju_isZero(const mjtNum* vec, int n);

// Standard normal random number generator (optional second number).
MJAPI mjtNum mju_standardNormal(mjtNum* num2);

// Convert from float to mjtNum.
MJAPI void mju_f2n(mjtNum* res, const float* vec, int n);

// Convert from mjtNum to float.
MJAPI void mju_n2f(float* res, const mjtNum* vec, int n);

// Convert from double to mjtNum.
MJAPI void mju_d2n(mjtNum* res, const double* vec, int n);

// Convert from mjtNum to double.
MJAPI void mju_n2d(double* res, const mjtNum* vec, int n);

// Insertion sort, resulting list is in increasing order.
MJAPI void mju_insertionSort(mjtNum* list, int n);

// Integer insertion sort, resulting list is in increasing order.
MJAPI void mju_insertionSortInt(int* list, int n);

// Generate Halton sequence.
MJAPI mjtNum mju_Halton(int index, int base);

// Call strncpy, then set dst[n-1] = 0.
MJAPI char* mju_strncpy(char *dst, const char *src, int n);

// Sigmoid function over 0<=x<=1 using quintic polynomial.
MJAPI mjtNum mju_sigmoid(mjtNum x);


//---------------------------------- Signed Distance Functions -------------------------------------

// get sdf from geom id
MJAPI const mjpPlugin* mjc_getSDF(const mjModel* m, int id);

// signed distance function
MJAPI mjtNum mjc_distance(const mjModel* m, const mjData* d, const mjSDF* s, const mjtNum x[3]);

// gradient of sdf
MJAPI void mjc_gradient(const mjModel* m, const mjData* d, const mjSDF* s, mjtNum gradient[3],
                        const mjtNum x[3]);


//---------------------------------- Derivatives ---------------------------------------------------

// Finite differenced transition matrices (control theory notation)
//   d(x_next) = A*dx + B*du
//   d(sensor) = C*dx + D*du
//   required output matrix dimensions:
//      A: (2*nv+na x 2*nv+na)
//      B: (2*nv+na x nu)
//      D: (nsensordata x 2*nv+na)
//      C: (nsensordata x nu)
// Nullable: A, B, C, D
MJAPI void mjd_transitionFD(const mjModel* m, mjData* d, mjtNum eps, mjtBool flg_centered,
                            mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D);

// Finite differenced Jacobians of (force, sensors) = mj_inverse(state, acceleration)
//   All outputs are optional. Output dimensions (transposed w.r.t Control Theory convention):
//     DfDq: (nv x nv)
//     DfDv: (nv x nv)
//     DfDa: (nv x nv)
//     DsDq: (nv x nsensordata)
//     DsDv: (nv x nsensordata)
//     DsDa: (nv x nsensordata)
//     DmDq: (nv x nC)
//   single-letter shortcuts:
//     inputs: q=qpos, v=qvel, a=qacc
//     outputs: f=qfrc_inverse, s=sensordata, m=M
//   notes:
//     optionally computes mass matrix Jacobian DmDq
//     flg_actuation specifies whether to subtract qfrc_actuator from qfrc_inverse
// Nullable: DfDq, DfDv, DfDa, DsDq, DsDv, DsDa, DmDq
MJAPI void mjd_inverseFD(const mjModel* m, mjData* d, mjtNum eps, mjtBool flg_actuation,
                         mjtNum *DfDq, mjtNum *DfDv, mjtNum *DfDa,
                         mjtNum *DsDq, mjtNum *DsDv, mjtNum *DsDa,
                         mjtNum *DmDq);

// Derivatives of mju_subQuat.
// Nullable: Da, Db
MJAPI void mjd_subQuat(const mjtNum qa[4], const mjtNum qb[4], mjtNum Da[9], mjtNum Db[9]);

// Derivatives of mju_quatIntegrate.
// Nullable: Dquat, Dvel, Dscale
MJAPI void mjd_quatIntegrate(const mjtNum vel[3], mjtNum scale,
                             mjtNum Dquat[9], mjtNum Dvel[9], mjtNum Dscale[3]);


//---------------------------------- Plugins -------------------------------------------------------

// Set default plugin definition.
MJAPI void mjp_defaultPlugin(mjpPlugin* plugin);

// Globally register a plugin. This function is thread-safe.
// If an identical mjpPlugin is already registered, this function does nothing.
// If a non-identical mjpPlugin with the same name is already registered, an mju_error is raised.
// Two mjpPlugins are considered identical if all member function pointers and numbers are equal,
// and the name and attribute strings are all identical, however the char pointers to the strings
// need not be the same.
MJAPI int mjp_registerPlugin(const mjpPlugin* plugin);

// Return the number of globally registered plugins.
MJAPI int mjp_pluginCount(void);

// Look up a plugin by name. If slot is not NULL, also write its registered slot number into it.
MJAPI const mjpPlugin* mjp_getPlugin(const char* name, int* slot);

// Look up a plugin by the registered slot number that was returned by mjp_registerPlugin.
MJAPI const mjpPlugin* mjp_getPluginAtSlot(int slot);

// Set default provider definition.
MJAPI void mjp_defaultProvider(mjpResourceProvider* provider);

// Globally register a provider in a thread-safe manner. The provider must have a prefix
// that is not a sub-prefix or super-prefix of any current registered providers.
// Return a slot number >= 0 on success, -1 on failure.
MJAPI int mjp_registerProvider(const mjpResourceProvider* provider);

// Return the number of globally registered providers.
MJAPI int mjp_providerCount(void);

// Return the provider with the prefix that matches against the resource name.
// If no match, return NULL.
MJAPI const mjpResourceProvider* mjp_findProvider(const char* resource_name);

// Look up a provider by slot number returned by mjp_registerProvider.
// If invalid slot number, return NULL.
MJAPI const mjpResourceProvider* mjp_getProviderAtSlot(int slot);

// Globally register a decoder. This function is thread-safe.
// If an identical mjpDecoder is already registered, this function does nothing.
// If a non-identical mjpDecoder with the same name is already registered, an mju_error is raised.
MJAPI void mjp_registerDecoder(const mjpDecoder* decoder);

// Set default resource decoder definition.
MJAPI void mjp_defaultDecoder(mjpDecoder* decoder);

// Return the decoder that matches against the resource or content type.
// If no match, return NULL.
MJAPI const mjpDecoder* mjp_findDecoder(const mjResource* resource, const char* content_type);

// Globally register an encoder. This function is thread-safe.
// If an identical mjpEncoder is already registered, this function does nothing.
// If a non-identical mjpEncoder with the same name is already registered, an mju_error is raised.
MJAPI void mjp_registerEncoder(const mjpEncoder* encoder);

// Set default resource encoder definition.
MJAPI void mjp_defaultEncoder(mjpEncoder* encoder);

// Return the encoder that matches against the content type or filename extension.
// If no match, return NULL.
MJAPI const mjpEncoder* mjp_findEncoder(const char* filename, const char* content_type);

// Globally register an archive provider. This function is thread-safe.
// provider->prefix specifies the filename extension(s) (e.g. .mjz|.zip).
MJAPI void mjp_registerArchiveProvider(const mjpResourceProvider* provider);

// Return the archive provider that matches against the resource name.
// If no match, return NULL.
MJAPI const mjpResourceProvider* mjp_findArchiveProvider(const char* resource_name);

// Return the number of globally registered archive providers.
MJAPI int mjp_archiveProviderCount(void);


//---------------------------------- Resources -----------------------------------------------------

// Open a resource; if the name doesn't have a prefix matching a registered provider,
// then the OS filesystem is used.
// Nullable: dir, vfs, error
MJAPI mjResource* mju_openResource(const char* dir, const char* name,
                                   const mjVFS* vfs, char* error, size_t nerror);

// Close a resource; no-op if resource is NULL.
MJAPI void mju_closeResource(mjResource* resource);

// Set buffer to bytes read from the resource and return number of bytes in buffer;
// return negative value if error.
MJAPI int mju_readResource(mjResource* resource, const void** buffer);

// Write resource data via its provider, return bytes written or -1 on error.
// Nullable: vfs, error
MJAPI mjtSize mju_writeResource(const char* name, const void* buffer, mjtSize nbytes,
                                const mjVFS* vfs, char* error, size_t nerror);

// For a resource with a name partitioned as {dir}{filename}, get the dir and ndir pointers.
MJAPI void mju_getResourceDir(mjResource* resource, const char** dir, int* ndir);

// Compare resource timestamp to provided timestamp.
// Return 0 if timestamps match, >0 if resource is newer, <0 if resource is older.
MJAPI int mju_isModifiedResource(const mjResource* resource, const char* timestamp);

// Find the decoder for a resource and return the decoded spec.
// The caller takes ownership of the spec and is responsible for cleaning it up.
// On failure, return NULL and write the reason to error.
// Nullable: vfs, error
MJAPI mjSpec* mju_decodeResource(mjResource* resource, const char* content_type,
                                 const mjVFS* vfs, char* error, int error_sz);


//---------------------------------- Threads -------------------------------------------------------

// Create a thread pool with nthread worker threads.
MJAPI void mju_threadpool(mjData* d, int nthread);


//---------------------------------- Attachment ----------------------------------------------------

// Attach child to a parent; return the attached element if success or NULL otherwise.
MJAPI mjsElement* mjs_attach(mjsElement* parent, const mjsElement* child,
                             const char* prefix, const char* suffix);


//---------------------------------- Tree elements -------------------------------------------------

// Add child body to body; return child.
// Nullable: def
MJAPI mjsBody* mjs_addBody(mjsBody* body, const mjsDefault* def);

// Add site to body; return site spec.
// Nullable: def
MJAPI mjsSite* mjs_addSite(mjsBody* body, const mjsDefault* def);

// Add joint to body.
// Nullable: def
MJAPI mjsJoint* mjs_addJoint(mjsBody* body, const mjsDefault* def);

// Add freejoint to body.
MJAPI mjsJoint* mjs_addFreeJoint(mjsBody* body);

// Add geom to body.
// Nullable: def
MJAPI mjsGeom* mjs_addGeom(mjsBody* body, const mjsDefault* def);

// Add camera to body.
// Nullable: def
MJAPI mjsCamera* mjs_addCamera(mjsBody* body, const mjsDefault* def);

// Add light to body.
// Nullable: def
MJAPI mjsLight* mjs_addLight(mjsBody* body, const mjsDefault* def);

// Add frame to body.
MJAPI mjsFrame* mjs_addFrame(mjsBody* body, mjsFrame* parentframe);

// Remove object corresponding to the given element; return 0 on success.
MJAPI int mjs_delete(mjSpec* spec, mjsElement* element);


//---------------------------------- Non-tree elements ---------------------------------------------

// Add actuator.
// Nullable: def
MJAPI mjsActuator* mjs_addActuator(mjSpec* s, const mjsDefault* def);

// Add sensor.
MJAPI mjsSensor* mjs_addSensor(mjSpec* s);

// Add flex.
MJAPI mjsFlex* mjs_addFlex(mjSpec* s);

// Add flexcomp: create flex with auto-generated bodies/joints, return flex spec.
// Nullable: type, dof, count, cellcount, spacing, scale, pos, quat, origin, file, vfs
MJAPI mjsFlex* mjs_makeFlex(mjsBody* body, const char* name, const char* type, int dim,
                            const char* dof, const int count[3], const int cellcount[3],
                            const double spacing[3], const double scale[3], double radius,
                            double mass, double inertiabox, int equality, int rigid, int flatskin,
                            int elastic2d, const double pos[3], const double quat[4],
                            const double origin[3], const char* file, const mjVFS* vfs);

// Add contact pair.
// Nullable: def
MJAPI mjsPair* mjs_addPair(mjSpec* s, const mjsDefault* def);

// Add excluded body pair.
MJAPI mjsExclude* mjs_addExclude(mjSpec* s);

// Add equality.
// Nullable: def
MJAPI mjsEquality* mjs_addEquality(mjSpec* s, const mjsDefault* def);

// Add tendon.
// Nullable: def
MJAPI mjsTendon* mjs_addTendon(mjSpec* s, const mjsDefault* def);

// Wrap site using tendon.
MJAPI mjsWrap* mjs_wrapSite(mjsTendon* tendon, const char* name);

// Wrap geom using tendon.
MJAPI mjsWrap* mjs_wrapGeom(mjsTendon* tendon, const char* name, const char* sidesite);

// Wrap joint using tendon.
MJAPI mjsWrap* mjs_wrapJoint(mjsTendon* tendon, const char* name, double coef);

// Wrap pulley using tendon.
MJAPI mjsWrap* mjs_wrapPulley(mjsTendon* tendon, double divisor);

// Add numeric.
MJAPI mjsNumeric* mjs_addNumeric(mjSpec* s);

// Add text.
MJAPI mjsText* mjs_addText(mjSpec* s);

// Add tuple.
MJAPI mjsTuple* mjs_addTuple(mjSpec* s);

// Add keyframe.
MJAPI mjsKey* mjs_addKey(mjSpec* s);

// Add plugin.
MJAPI mjsPlugin* mjs_addPlugin(mjSpec* s);

// Add default.
// Nullable: parent
MJAPI mjsDefault* mjs_addDefault(mjSpec* s, const char* classname, const mjsDefault* parent);


//---------------------------------- Set actuator parameters ---------------------------------------

// Set actuator to motor; return error if any.
MJAPI const char* mjs_setToMotor(mjsActuator* actuator);

// Set actuator to position; return error if any.
MJAPI const char* mjs_setToPosition(mjsActuator* actuator, double kp, double kv[1],
                                    double dampratio[1], double timeconst[1], double inheritrange);

// Set actuator to integrated velocity; return error if any.
MJAPI const char* mjs_setToIntVelocity(mjsActuator* actuator, double kp, double kv[1],
                                       double dampratio[1], double timeconst[1], double inheritrange);

// Set actuator to velocity servo; return error if any.
MJAPI const char* mjs_setToVelocity(mjsActuator* actuator, double kv);

// Set actuator to orientation servo.
MJAPI const char* mjs_setToOrientation(mjsActuator* actuator, double kp, double kv[1],
                                       double dampratio[1], int ctrlspec);

// Set actuator to PID controller.
MJAPI const char* mjs_setToPID(mjsActuator* actuator, double kp, double kv[1], double dampratio[1],
                               double ki[1], double imax[1], double slewmax[1], double inheritrange,
                               int ctrlspec);

// Set actuator to activate damper; return error if any.
MJAPI const char* mjs_setToDamper(mjsActuator* actuator, double kv);

// Set actuator to hydraulic or pneumatic cylinder; return error if any.
MJAPI const char* mjs_setToCylinder(mjsActuator* actuator, double timeconst,
                                    double bias, double area, double diameter);

// Set actuator to muscle; return error if any.a
MJAPI const char* mjs_setToMuscle(mjsActuator* actuator, double timeconst[2], double tausmooth,
                                  double range[2], double force, double scale, double lmin,
                                  double lmax, double vmax, double fpmax, double fvmax);

// Set actuator to active adhesion; return error if any.
MJAPI const char* mjs_setToAdhesion(mjsActuator* actuator, double gain);

// Set actuator to DC motor; return error if any.
// Nullable: motorconst, nominal, saturation, inductance, cogging, controller, thermal, lugre
MJAPI const char* mjs_setToDCMotor(mjsActuator* actuator, double motorconst[2], double resistance,
                                   double nominal[3], double saturation[3], double inductance[2],
                                   double cogging[3], double controller[6], double thermal[6],
                                   double lugre[5], int ctrlspec);


//---------------------------------- Assets --------------------------------------------------------

// Add mesh.
// Nullable: def
MJAPI mjsMesh* mjs_addMesh(mjSpec* s, const mjsDefault* def);

// Add height field.
MJAPI mjsHField* mjs_addHField(mjSpec* s);

// Add skin.
MJAPI mjsSkin* mjs_addSkin(mjSpec* s);

// Add texture.
// Nullable: def
MJAPI mjsTexture* mjs_addTexture(mjSpec* s, const mjsDefault* def);

// Add material.
// Nullable: def
MJAPI mjsMaterial* mjs_addMaterial(mjSpec* s, const mjsDefault* def);

// Sets the vertices and normals of a mesh.
MJAPI int mjs_makeMesh(mjsMesh* mesh, mjtMeshBuiltin builtin, double* params, int nparams);

//---------------------------------- Find and get utilities ----------------------------------------

// Get spec from body.
MJAPI mjSpec* mjs_getSpec(const mjsElement* element);

// get spec that originally defined an element
// contrary to mjs_getSpec, this does not change after attachment
MJAPI mjSpec* mjs_getOriginSpec(const mjsElement* element);

// Get compiler associated with element's origin spec.
MJAPI mjsCompiler* mjs_getCompiler(const mjsElement* element);

// Find spec (model asset) by name.
MJAPI mjSpec* mjs_findSpec(const mjSpec* spec, const char* name);

// Find body in spec by name.
MJAPI mjsBody* mjs_findBody(const mjSpec* s, const char* name);

// Find element in spec by name.
MJAPI mjsElement* mjs_findElement(const mjSpec* s, mjtObj type, const char* name);

// Find child body by name.
MJAPI mjsBody* mjs_findChild(const mjsBody* body, const char* name);

// Get parent body.
MJAPI mjsBody* mjs_getParent(const mjsElement* element);

// Get parent frame.
MJAPI mjsFrame* mjs_getFrame(const mjsElement* element);

// Find frame by name.
MJAPI mjsFrame* mjs_findFrame(const mjSpec* s, const char* name);

// Get default corresponding to an element.
MJAPI mjsDefault* mjs_getDefault(const mjsElement* element);

// Find default in model by class name.
MJAPI mjsDefault* mjs_findDefault(const mjSpec* s, const char* classname);

// Get global default from model.
MJAPI mjsDefault* mjs_getSpecDefault(const mjSpec* s);

// Get element id.
MJAPI int mjs_getId(const mjsElement* element);

// Return body's first child of given type. If recurse is nonzero, also search the body's subtree.
MJAPI mjsElement* mjs_firstChild(const mjsBody* body, mjtObj type, int recurse);

// Return body's next child of the same type; return NULL if child is last.
// If recurse is nonzero, also search the body's subtree.
MJAPI mjsElement* mjs_nextChild(const mjsBody* body, const mjsElement* child, int recurse);

// Return spec's first element of selected type.
MJAPI mjsElement* mjs_firstElement(const mjSpec* s, mjtObj type);

// Return spec's next element; return NULL if element is last.
MJAPI mjsElement* mjs_nextElement(const mjSpec* s, const mjsElement* element);

// Get wrapped element in tendon path.
MJAPI mjsElement* mjs_getWrapTarget(const mjsWrap* wrap);

// Get wrapped element side site in tendon path if it has one, nullptr otherwise.
MJAPI mjsSite* mjs_getWrapSideSite(const mjsWrap* wrap);

// Get divisor of mjsWrap wrapping a puller.
MJAPI double mjs_getWrapDivisor(const mjsWrap* wrap);

// Get coefficient of mjsWrap wrapping a joint.
MJAPI double mjs_getWrapCoef(const mjsWrap* wrap);

//---------------------------------- Attribute setters ---------------------------------------------

// Set element's name; return 0 on success.
MJAPI int mjs_setName(mjsElement* element, const char* name);

// Copy buffer.
MJAPI void mjs_setBuffer(mjByteVec* dest, const void* array, int size);

// Copy text to string.
MJAPI void mjs_setString(mjString* dest, const char* text);

// Split text to entries and copy to string vector.
MJAPI void mjs_setStringVec(mjStringVec* dest, const char* text);

// Set entry in string vector.
MJAPI mjtBool mjs_setInStringVec(mjStringVec* dest, int i, const char* text);

// Append text entry to string vector.
MJAPI void mjs_appendString(mjStringVec* dest, const char* text);

// Copy int array to vector.
MJAPI void mjs_setInt(mjIntVec* dest, const int* array, int size);

// Append int array to vector of arrays.
MJAPI void mjs_appendIntVec(mjIntVecVec* dest, const int* array, int size);

// Copy float array to vector.
MJAPI void mjs_setFloat(mjFloatVec* dest, const float* array, int size);

// Append float array to vector of arrays.
MJAPI void mjs_appendFloatVec(mjFloatVecVec* dest, const float* array, int size);

// Copy double array to vector.
MJAPI void mjs_setDouble(mjDoubleVec* dest, const double* array, int size);

// Set plugin attributes.
MJAPI void mjs_setPluginAttributes(mjsPlugin* plugin, void* attributes);


//---------------------------------- Attribute getters ---------------------------------------------

// Get element's name.
MJAPI mjString* mjs_getName(mjsElement* element);

// Get string contents.
MJAPI const char* mjs_getString(const mjString* source);

// Get double array contents and optionally its size.
// Nullable: size
MJAPI const double* mjs_getDouble(const mjDoubleVec* source, int* size);

// Get number of elements a tendon wraps.
MJAPI int mjs_getWrapNum(const mjsTendon* tendonspec);

// Get mjsWrap element at position i in the tendon path.
MJAPI mjsWrap* mjs_getWrap(const mjsTendon* tendonspec, int i);

// Get plugin attributes.
MJAPI const void* mjs_getPluginAttributes(const mjsPlugin* plugin);


//---------------------------------- Spec utilities ------------------------------------------------

// Set element's default.
MJAPI void mjs_setDefault(mjsElement* element, const mjsDefault* def);

// Set element's enclosing frame; return 0 on success.
MJAPI int mjs_setFrame(mjsElement* dest, mjsFrame* frame);

// Resolve alternative orientations to quat; return error if any.
MJAPI const char* mjs_resolveOrientation(double quat[4], mjtByte degree, const char* sequence,
                                         const mjsOrientation* orientation);

// Fuse the static bodies of the spec with their parents; return the number fused, -1 on error.
// Nullable: vfs
MJAPI int mjs_fuseStatic(mjSpec* s, const mjVFS* vfs);

// Discard the visual elements of the spec; return the number discarded, -1 on error.
// Nullable: vfs
MJAPI int mjs_discardVisual(mjSpec* s, const mjVFS* vfs);

// Merge the textures and meshes of the spec which would compile to the same asset; return the number
// merged, -1 on error.
MJAPI int mjs_dedupAssets(mjSpec* s);

// Make the inertial which compilation infers for a body part of the spec, return 0 on success.
// Nullable: vfs
MJAPI int mjs_adoptInertial(mjsBody* body, const mjVFS* vfs);

// Transform body into a frame.
MJAPI mjsFrame* mjs_bodyToFrame(mjsBody** body);

// Set user payload, overriding the existing value for the specified key if present.
MJAPI void mjs_setUserValue(mjsElement* element, const char* key, const void* data);

// Set user payload, overriding the existing value for the specified key if
// present. This version differs from mjs_setUserValue in that it takes a
// cleanup function that will be called when the user payload is deleted.
MJAPI void mjs_setUserValueWithCleanup(mjsElement* element, const char* key,
                                       const void* data,
                                       void (*cleanup)(const void*));

// Return user payload or NULL if none found.
MJAPI const void* mjs_getUserValue(mjsElement* element, const char* key);

// Delete user payload.
MJAPI void mjs_deleteUserValue(mjsElement* element, const char* key);

// Return sensor dimension.
MJAPI int mjs_sensorDim(const mjsSensor* sensor);

//---------------------------------- Element initialization  ---------------------------------------

// Default spec attributes.
MJAPI void mjs_defaultSpec(mjSpec* spec);

// Default orientation attributes.
MJAPI void mjs_defaultOrientation(mjsOrientation* orient);

// Default body attributes.
MJAPI void mjs_defaultBody(mjsBody* body);

// Default frame attributes.
MJAPI void mjs_defaultFrame(mjsFrame* frame);

// Default joint attributes.
MJAPI void mjs_defaultJoint(mjsJoint* joint);

// Default geom attributes.
MJAPI void mjs_defaultGeom(mjsGeom* geom);

// Default site attributes.
MJAPI void mjs_defaultSite(mjsSite* site);

// Default camera attributes.
MJAPI void mjs_defaultCamera(mjsCamera* camera);

// Default light attributes.
MJAPI void mjs_defaultLight(mjsLight* light);

// Default flex attributes.
MJAPI void mjs_defaultFlex(mjsFlex* flex);

// Default mesh attributes.
MJAPI void mjs_defaultMesh(mjsMesh* mesh);

// Default height field attributes.
MJAPI void mjs_defaultHField(mjsHField* hfield);

// Default skin attributes.
MJAPI void mjs_defaultSkin(mjsSkin* skin);

// Default texture attributes.
MJAPI void mjs_defaultTexture(mjsTexture* texture);

// Default material attributes.
MJAPI void mjs_defaultMaterial(mjsMaterial* material);

// Default pair attributes.
MJAPI void mjs_defaultPair(mjsPair* pair);

// Default equality attributes.
MJAPI void mjs_defaultEquality(mjsEquality* equality);

// Default tendon attributes.
MJAPI void mjs_defaultTendon(mjsTendon* tendon);

// Default actuator attributes.
MJAPI void mjs_defaultActuator(mjsActuator* actuator);

// Default sensor attributes.
MJAPI void mjs_defaultSensor(mjsSensor* sensor);

// Default numeric attributes.
MJAPI void mjs_defaultNumeric(mjsNumeric* numeric);

// Default text attributes.
MJAPI void mjs_defaultText(mjsText* text);

// Default tuple attributes.
MJAPI void mjs_defaultTuple(mjsTuple* tuple);

// Default keyframe attributes.
MJAPI void mjs_defaultKey(mjsKey* key);

// Default plugin attributes.
MJAPI void mjs_defaultPlugin(mjsPlugin* plugin);


//---------------------------------- Element casting -----------------------------------------------

// Safely cast an element as mjsBody, or return NULL if the element is not an mjsBody.
MJAPI mjsBody* mjs_asBody(mjsElement* element);

// Safely cast an element as mjsGeom, or return NULL if the element is not an mjsGeom.
MJAPI mjsGeom* mjs_asGeom(mjsElement* element);

// Safely cast an element as mjsJoint, or return NULL if the element is not an mjsJoint.
MJAPI mjsJoint* mjs_asJoint(mjsElement* element);

// Safely cast an element as mjsSite, or return NULL if the element is not an mjsSite.
MJAPI mjsSite* mjs_asSite(mjsElement* element);

// Safely cast an element as mjsCamera, or return NULL if the element is not an mjsCamera.
MJAPI mjsCamera* mjs_asCamera(mjsElement* element);

// Safely cast an element as mjsLight, or return NULL if the element is not an mjsLight.
MJAPI mjsLight* mjs_asLight(mjsElement* element);

// Safely cast an element as mjsFrame, or return NULL if the element is not an mjsFrame.
MJAPI mjsFrame* mjs_asFrame(mjsElement* element);

// Safely cast an element as mjsActuator, or return NULL if the element is not an mjsActuator.
MJAPI mjsActuator* mjs_asActuator(mjsElement* element);

// Safely cast an element as mjsSensor, or return NULL if the element is not an mjsSensor.
MJAPI mjsSensor* mjs_asSensor(mjsElement* element);

// Safely cast an element as mjsFlex, or return NULL if the element is not an mjsFlex.
MJAPI mjsFlex* mjs_asFlex(mjsElement* element);

// Safely cast an element as mjsPair, or return NULL if the element is not an mjsPair.
MJAPI mjsPair* mjs_asPair(mjsElement* element);

// Safely cast an element as mjsEquality, or return NULL if the element is not an mjsEquality.
MJAPI mjsEquality* mjs_asEquality(mjsElement* element);

// Safely cast an element as mjsExclude, or return NULL if the element is not an mjsExclude.
MJAPI mjsExclude* mjs_asExclude(mjsElement* element);

// Safely cast an element as mjsTendon, or return NULL if the element is not an mjsTendon.
MJAPI mjsTendon* mjs_asTendon(mjsElement* element);

// Safely cast an element as mjsNumeric, or return NULL if the element is not an mjsNumeric.
MJAPI mjsNumeric* mjs_asNumeric(mjsElement* element);

// Safely cast an element as mjsText, or return NULL if the element is not an mjsText.
MJAPI mjsText* mjs_asText(mjsElement* element);

// Safely cast an element as mjsTuple, or return NULL if the element is not an mjsTuple.
MJAPI mjsTuple* mjs_asTuple(mjsElement* element);

// Safely cast an element as mjsKey, or return NULL if the element is not an mjsKey.
MJAPI mjsKey* mjs_asKey(mjsElement* element);

// Safely cast an element as mjsMesh, or return NULL if the element is not an mjsMesh.
MJAPI mjsMesh* mjs_asMesh(mjsElement* element);

// Safely cast an element as mjsHField, or return NULL if the element is not an mjsHField.
MJAPI mjsHField* mjs_asHField(mjsElement* element);

// Safely cast an element as mjsSkin, or return NULL if the element is not an mjsSkin.
MJAPI mjsSkin* mjs_asSkin(mjsElement* element);

// Safely cast an element as mjsTexture, or return NULL if the element is not an mjsTexture.
MJAPI mjsTexture* mjs_asTexture(mjsElement* element);

// Safely cast an element as mjsMaterial, or return NULL if the element is not an mjsMaterial.
MJAPI mjsMaterial* mjs_asMaterial(mjsElement* element);

// Safely cast an element as mjsPlugin, or return NULL if the element is not an mjsPlugin.
MJAPI mjsPlugin* mjs_asPlugin(mjsElement* element);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_MUJOCO_H_
