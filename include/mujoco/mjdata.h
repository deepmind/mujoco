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

#ifndef MUJOCO_MJDATA_H_
#define MUJOCO_MJDATA_H_

#include <stddef.h>
#include <stdint.h>

#include <mujoco/mjtype.h>
#include <mujoco/mjmodel.h>




//------------------------------------- Contact ----------------------------------------------------

typedef struct mjPreContact_ {     // contact parameters set by narrowphase collision functions
  mjtNum dist;
  mjtNum pos[3];
  mjtNum normal[3];                // contact normal of the collision
  mjtNum tangent[3];               // first tangent direction
} mjPreContact;


typedef struct mjContact_ {        // result of collision detection functions
  // contact parameters set by narrowphase collision function
  mjtNum  dist;                    // distance between nearest points; neg: penetration
  mjtNum  pos[3];                  // position of contact point: midpoint between geoms
  mjtNum  frame[9];                // normal is in [0-2], points from geom[0] to geom[1]

  // contact parameters set by mj_collideGeoms
  mjtNum  includemargin;           // margin for force generation
  mjtNum  friction[5];             // tangent1, 2, spin, roll1, 2
  mjtNum  solref[mjNREF];          // constraint solver reference, normal direction
  mjtNum  solreffriction[mjNREF];  // constraint solver reference, friction directions
  mjtNum  solimp[mjNIMP];          // constraint solver impedance
  mjtNum  adhesion;                // adhesive force along the contact normal
  mjtNum  surfacevel[3];           // surface velocity (geom[1] - geom[0]): tangent1, 2, spin

  // internal storage used by solver
  mjtNum  mu;                      // friction of regularized cone, set by mj_makeConstraint
  mjtNum  H[36];                   // cone Hessian, set by mj_constraintUpdate

  // contact descriptors set by mj_collideXXX
  int     dim;                     // contact space dimensionality: 1, 3, 4 or 6
  int     geom1;                   // id of geom 1; deprecated, use geom[0]
  int     geom2;                   // id of geom 2; deprecated, use geom[1]
  int     geom[2];                 // geom ids; -1 for flex
  int     flex[2];                 // flex ids; -1 for geom
  int     elem[2];                 // element ids; -1 for geom or flex vertex
  int     vert[2];                 // vertex ids;  -1 for geom or flex element

  // flag set by mj_setContact or mj_instantiateContact
  int     exclude;                 // 0: include, 1: in gap, 2: fused, 3: no dofs, 4: passive

  // address computed by mj_instantiateContact
  int     efc_address;             // address in efc; -1: not included
} mjContact;


//---------------------------------- diagnostics ---------------------------------------------------

typedef struct mjWarningStat_ {  // warning statistics
  int     lastinfo;              // info from last warning
  int     number;                // how many times was warning raised
} mjWarningStat;


typedef struct mjTimerStat_ {  // timer statistics
  mjtNum  duration;            // cumulative duration
  int     number;              // how many times was timer called
} mjTimerStat;


typedef struct mjSolverStat_ {  // per-iteration solver statistics
  mjtNum  improvement;          // cost reduction, scaled by 1/trace(M)
  mjtNum  gradient;             // gradient norm (primal only, scaled)
  mjtNum  lineslope;            // slope in linesearch
  int     nactive;              // number of active constraints
  int     nchange;              // number of constraint state changes
  int     neval;                // number of cost evaluations in line search
  int     nupdate;              // number of Cholesky updates in line search
} mjSolverStat;


//---------------------------------- mjData --------------------------------------------------------

typedef struct mjData_ {
  // constant sizes
  mjtSize narena;            // size of the arena in bytes (inclusive of the stack)
  mjtSize nbuffer;           // size of main buffer in bytes
  int     nplugin;           // number of plugin instances

  // stack pointer
  size_t  pstack;            // first available byte in stack (mutable)
  size_t  pbase;             // value of pstack when mj_markStack was last called (mutable)

  // arena pointer
  size_t  parena;            // first available byte in arena

  // threading
  uintptr_t threadpool;      // thread pool pointer
  mjtBool threadlock;        // disable stack freeing during threaded execution

  // memory utilization statistics
  mjtSize maxuse_stack;                       // maximum stack allocation in bytes (mutable)
  mjtSize maxuse_arena;                       // maximum arena allocation in bytes
  int     maxuse_con;                         // maximum number of contacts
  int     maxuse_efc;                         // maximum number of scalar constraints

  // solver statistics
  mjSolverStat  solver[mjNISLAND*mjNSOLVER];  // solver statistics per island, per iteration
  int           solver_niter[mjNISLAND];      // number of solver iterations, per island
  int           solver_nnz[mjNISLAND];        // number of nonzeros in solver matrix, per island
  mjtNum        solver_fwdinv[2];             // forward-inverse comparison: qfrc, efc

  // diagnostics
  mjWarningStat warning[mjNWARNING];          // warning statistics (mutable)
  mjTimerStat   timer[mjNTIMER];              // timer statistics

  // variable sizes
  int     ncon;              // number of detected contacts
  int     ne;                // number of equality constraints
  int     nf;                // number of friction constraints
  int     nl;                // number of limit constraints
  int     nefc;              // number of constraints
  int     nJ;                // number of non-zeros in constraint Jacobian

  // effective metric: per-step activity flag and sizes, set by mj_effBuild
  int     efm_active;        // implicit effective metric M+K is active (see mj_effBuild)
  int     nefmK;             // number of non-zeros in effective-stiffness CSR
  int     nefmcon;           // packed length of the contact rank-1 rows
  int     nefmT;             // number of tendons with terms in the metric
  int     nefmA;             // number of actuators with terms in the metric
  int     nefmdof;           // number of 3x3 blocks in the effective-metric preconditioner
  int     nefmL;             // size of efm_L (9*nefmdof plus optional Cholesky or island tail)

  // variable sizes, continued
  int     nY;                // number of non-zeros in constraint inverse inertia square root
  int     nA;                // number of non-zeros in constraint inverse inertia matrix
  int     nisland;           // number of detected constraint islands
  int     nidof;             // number of dofs in all islands
  int     ntree_awake;       // number of awake trees
  int     nbody_awake;       // number of awake dynamic and static bodies
  int     nparent_awake;     // number of bodies with awake parents
  int     nv_awake;          // number of awake dofs

  // flags marking lazily evaluated stages
  mjtBool flg_energypos;     // has mj_energyPos been called
  mjtBool flg_energyvel;     // has mj_energyVel been called
  mjtBool flg_subtreevel;    // has mj_subtreeVel been called
  mjtBool flg_rnepost;       // has mj_rnePostConstraint been called

  // global properties
  mjtNum  time;              // simulation time
  mjtNum  energy[2];         // potential, kinetic energy

  //-------------------- end of info header

  // buffers
  void*   buffer;            // main buffer; all pointers point in it            (nbuffer bytes)
  void*   arena;             // arena+stack buffer                               (narena bytes)

  //-------------------- main inputs and outputs of the computation

  // state
  mjtNum* qpos;              // position                                         (nq x 1)
  mjtNum* qvel;              // velocity                                         (nv x 1)
  mjtNum* act;               // actuator activation                              (na x 1)
  mjtNum* history;           // history buffer                                   (nhistory x 1)
  mjtNum* plastic;           // plastic state                                    (nplastic x 1)
  mjtNum* qacc_warmstart;    // acceleration used for warmstart                  (nv x 1)
  mjtNum* plugin_state;      // plugin state                                     (npluginstate x 1)

  // control
  mjtNum* ctrl;              // control                                          (nu x 1)
  mjtNum* qfrc_applied;      // applied generalized force                        (nv x 1)
  mjtNum* xfrc_applied;      // applied Cartesian force/torque                   (nbody x 6)
  mjtBool* eq_active;        // enable/disable constraints                       (neq x 1)

  // mocap data
  mjtNum* mocap_pos;         // positions of mocap bodies                        (nmocap x 3)
  mjtNum* mocap_quat;        // orientations of mocap bodies                     (nmocap x 4)

  // dynamics
  mjtNum* qacc;              // acceleration                                     (nv x 1)
  mjtNum* act_dot;           // time-derivative of actuator activation           (na x 1)

  // user data
  mjtNum* userdata;          // user data, not touched by engine                 (nuserdata x 1)

  // sensors
  mjtNum* sensordata;        // sensor data array                                (nsensordata x 1)

  // sleep state
  int*    tree_asleep;       // <0: awake; >=0: index cycle of sleeping trees    (ntree x 1)

  // plugins
  int*       plugin;         // copy of m->plugin, required for deletion         (nplugin x 1)
  uintptr_t* plugin_data;    // pointer to plugin-managed data structure         (nplugin x 1)

  //-------------------- POSITION dependent

  // computed by mj_fwdPosition/mj_kinematics
  mjtNum* xpos;              // Cartesian position of body frame                 (nbody x 3)
  mjtNum* xquat;             // Cartesian orientation of body frame              (nbody x 4)
  mjtNum* xmat;              // Cartesian orientation of body frame              (nbody x 9)
  mjtNum* xipos;             // Cartesian position of body com                   (nbody x 3)
  mjtNum* xiquat;            // Cartesian orientation of body inertia            (nbody x 4)
  mjtNum* ximat;             // Cartesian orientation of body inertia            (nbody x 9)
  mjtNum* xanchor;           // Cartesian position of joint anchor               (njnt x 3)
  mjtNum* xaxis;             // Cartesian joint axis                             (njnt x 3)
  mjtNum* geom_xpos;         // Cartesian geom position                          (ngeom x 3)
  mjtNum* geom_xquat;        // Cartesian geom orientation                       (ngeom x 4)
  mjtNum* geom_xmat;         // Cartesian geom orientation                       (ngeom x 9)
  mjtNum* site_xpos;         // Cartesian site position                          (nsite x 3)
  mjtNum* site_xquat;        // Cartesian site orientation                       (nsite x 4)
  mjtNum* site_xmat;         // Cartesian site orientation                       (nsite x 9)
  mjtNum* cam_xpos;          // Cartesian camera position                        (ncam x 3)
  mjtNum* cam_xmat;          // Cartesian camera orientation                     (ncam x 9)
  mjtNum* light_xpos;        // Cartesian light position                         (nlight x 3)
  mjtNum* light_xdir;        // Cartesian light direction                        (nlight x 3)

  // computed by mj_fwdPosition/mj_comPos
  mjtNum* subtree_com;       // center of mass of each subtree                   (nbody x 3)
  mjtNum* cdof;              // com-based motion axis of each dof (rot:lin)      (nv x 6)
  mjtNum* cinert;            // com-based body inertia and mass                  (nbody x 10)

  // computed by mj_fwdPosition/mj_flex
  mjtNum*  flexvert_xpos;      // Cartesian flex vertex positions                (nflexvert x 3)
  mjtNum*  flexelem_aabb;      // flex element bounding boxes (center, size)     (nflexelem x 6)
  mjtNum*  flexelem_krot;      // corotated element stiffness (implicit only)    (nflexstiffness x 1)
  mjtBool* flex_hessian_valid; // Cartesian stretch Hessian cache is current     (nflex x 1)
  mjtNum*  flexvert_hessian;   // symmetric diagonal Hessian blocks              (nflexvert x 6)
  mjtNum*  flexedge_hessian;   // oriented off-diagonal Hessian blocks           (nflexedge x 9)
  mjtNum*  flexedge_J;         // flex edge Jacobian                             (nJfe x 1)
  mjtNum*  flexedge_length;    // flex edge lengths                              (nflexedge x 1)
  mjtNum*  bvh_aabb_dyn;       // global bounding box (center, size)             (nbvhdynamic x 6)

  // AL contact state carried across steps (flag ipc, not in mjtState)
  mjtNum*  flexvert_lambda;    // flex contact multiplier                        (nflexvert x 1)
  int*     flexvert_conage;    // flex contact age: <0 loaded, >0 steps since    (nflexvert x 1)

  // computed by mj_fwdPosition/mj_tendon
  int*    ten_wrapadr;       // start address of tendon's path                   (ntendon x 1)
  int*    ten_wrapnum;       // number of wrap points in path                    (ntendon x 1)
  mjtNum* ten_J;             // tendon Jacobian                                  (nJten x 1)
  mjtNum* ten_length;        // tendon lengths                                   (ntendon x 1)
  int*    wrap_obj;          // geom id; -1: site; -2: pulley                    (nwrap x 2)
  mjtNum* wrap_xpos;         // Cartesian 3D points in all paths                 (nwrap x 6)

  // computed by mj_fwdPosition/mj_transmission
  mjtNum* actuator_length;   // actuator lengths, one per force output           (nout x 1)
  int*    moment_rownnz;     // number of non-zeros in actuator_moment row       (nout x 1)
  int*    moment_rowadr;     // row start address in colind array                (nout x 1)
  int*    moment_colind;     // column indices in sparse Jacobian                (nJmom x 1)
  mjtNum* actuator_moment;   // actuator moments                                 (nJmom x 1)

  // computed by mj_fwdPosition/mj_makeM
  mjtNum* crb;               // com-based composite inertia and mass             (nbody x 10)
  mjtNum* M;                 // inertia (sparse)                                 (nC x 1)

  // computed by mj_fwdPosition/mj_factorM
  mjtNum* qLD;               // L'*D*L factorization of M (sparse)               (nC x 1)
  mjtNum* qLDiagInv;         // 1/diag(D)                                        (nv x 1)

  // computed by mj_collision/mj_collideTree
  mjtBool* bvh_active;       // was bounding volume checked for collision        (nbvh x 1)

  // computed by mj_updateSleep
  int*    tree_awake;        // is tree awake; 0: asleep; 1: awake               (ntree x 1)
  int*    body_awake;        // body sleep state (mjtSleepState)                 (nbody x 1)
  int*    body_awake_ind;    // indices of awake and static bodies               (nbody x 1)
  int*    parent_awake_ind;  // indices of bodies with awake or static parents   (nbody x 1)
  int*    dof_awake_ind;     // indices of awake dofs                            (nv x 1)

  //-------------------- POSITION, VELOCITY dependent

  // computed by mj_fwdVelocity
  mjtNum* flexedge_velocity; // flex edge velocities                             (nflexedge x 1)
  mjtNum* ten_velocity;      // tendon velocities                                (ntendon x 1)
  mjtNum* actuator_velocity; // actuator velocities, one per force output        (nout x 1)

  // computed by mj_fwdVelocity/mj_comVel
  mjtNum* cvel;              // com-based velocity (rot:lin)                     (nbody x 6)
  mjtNum* cdof_dot;          // time-derivative of cdof (rot:lin)                (nv x 6)

  // computed by mj_fwdVelocity/mj_rne (without acceleration)
  mjtNum* qfrc_bias;         // C(qpos,qvel)                                     (nv x 1)

  // computed by mj_fwdVelocity/mj_passive
  mjtNum* qfrc_spring;       // passive spring force                             (nv x 1)
  mjtNum* qfrc_damper;       // passive damper force                             (nv x 1)
  mjtNum* qfrc_gravcomp;     // passive gravity compensation force               (nv x 1)
  mjtNum* qfrc_fluid;        // passive fluid force                              (nv x 1)
  mjtNum* qfrc_adhesion;     // passive contact adhesion force                   (nv x 1)
  mjtNum* qfrc_passive;      // total passive force                              (nv x 1)

  // computed by mj_sensorVel/mj_subtreeVel if needed
  mjtNum* subtree_linvel;    // linear velocity of subtree com                   (nbody x 3)
  mjtNum* subtree_angmom;    // angular momentum about subtree com               (nbody x 3)

  // computed by mj_Euler or mj_implicit
  mjtNum* qH;                // L'*D*L factorization of modified M               (nC x 1)
  mjtNum* qHDiagInv;         // 1/diag(D) of modified M                          (nv x 1)

  // computed by mj_implicit/mj_derivative
  mjtNum* qDeriv;            // d (passive + actuator - bias) / d qvel           (nD x 1)

  // computed by mj_implicit/mju_factorLUSparse
  mjtNum* qLU;               // sparse LU of (M - dt*qDeriv)                     (nD x 1)

  //-------------------- POSITION, VELOCITY, CONTROL/ACCELERATION dependent

  // computed by mj_fwdActuation
  mjtNum* actuator_force;    // actuator force in actuation space                (nout x 1)
  mjtNum* qfrc_actuator;     // actuator force in joint space                    (nv x 1)

  // computed by mj_fwdAcceleration
  mjtNum* qfrc_smooth;       // net unconstrained force                          (nv x 1)
  mjtNum* qacc_smooth;       // unconstrained acceleration                       (nv x 1)

  // computed by mj_fwdConstraint/mj_inverse
  mjtNum* qfrc_constraint;   // constraint force (flag ipc: incl. flex contact)  (nv x 1)

  // computed by mj_inverse
  mjtNum* qfrc_inverse;      // net external force; should equal:
                             // qfrc_applied + J'*xfrc_applied + qfrc_actuator   (nv x 1)

  // computed by mj_sensorAcc/mj_rnePostConstraint if needed; rotation:translation format
  mjtNum* cacc;              // com-based acceleration                           (nbody x 6)
  mjtNum* cfrc_int;          // com-based interaction force with parent          (nbody x 6)
  mjtNum* cfrc_ext;          // com-based external force on body                 (nbody x 6)

  //-------------------- arena-allocated: POSITION dependent

  // computed by mj_collision
  mjContact* contact;        // array of all detected contacts                   (ncon x 1)

  // computed by mj_makeConstraint
  int*    efc_type;          // constraint type (mjtConstraint)                  (nefc x 1)
  int*    efc_id;            // id of object of specified type                   (nefc x 1)
  int*    efc_J_rownnz;      // number of non-zeros in constraint Jacobian row   (nefc x 1)
  int*    efc_J_rowadr;      // row start address in colind array                (nefc x 1)
  int*    efc_J_rowsuper;    // number of subsequent rows in supernode           (nefc x 1)
  int*    efc_J_colind;      // column indices in constraint Jacobian            (nJ x 1)
  mjtNum* efc_J;             // constraint Jacobian                              (nJ x 1)
  mjtNum* efc_pos;           // constraint position (equality, contact)          (nefc x 1)
  mjtNum* efc_margin;        // inclusion margin (contact)                       (nefc x 1)
  mjtNum* efc_frictionloss;  // frictionloss (friction)                          (nefc x 1)
  mjtNum* efc_diagA;         // diagonal of A matrix, approximate or exact       (nefc x 1)
  mjtNum* efc_KBIP;          // stiffness, damping, impedance, imp'              (nefc x 4)
  mjtNum* efc_D;             // constraint mass                                  (nefc x 1)
  mjtNum* efc_R;             // inverse constraint mass                          (nefc x 1)
  int*    tendon_efcadr;     // first efc address involving tendon; -1: none     (ntendon x 1)

  // computed by mj_island (island tree structure)
  int*    tree_island;       // island id of this tree; -1: none                 (ntree x 1)
  int*    island_ntree;      // number of trees in this island                   (nisland x 1)
  int*    island_itreeadr;   // island start address in itree vector             (nisland x 1)
  int*    map_itree2tree;    // map from itree to tree                           (ntree x 1)

  // computed by mj_island (island dof structure)
  int*    dof_island;        // island id of this dof; -1: none                  (nv x 1)
  int*    island_nv;         // number of dofs in this island                    (nisland x 1)
  int*    island_idofadr;    // island start address in idof vector              (nisland x 1)
  int*    island_dofadr;     // island start address in dof vector               (nisland x 1)
  int*    map_dof2idof;      // map from dof to idof                             (nv x 1)
  int*    map_idof2dof;      // map from idof to dof;  >= nidof: unconstrained   (nv x 1)

  // computed by mj_island (dofs sorted by island)
  mjtNum* ifrc_smooth;       // net unconstrained force                          (nidof x 1)
  mjtNum* iacc_smooth;       // unconstrained acceleration                       (nidof x 1)
  mjtNum* iacc;              // acceleration                                     (nidof x 1)

  // computed by mj_island (island constraint structure)
  int*    efc_island;        // island id of this constraint                     (nefc x 1)
  int*    island_ne;         // number of equality constraints in island         (nisland x 1)
  int*    island_nf;         // number of friction constraints in island         (nisland x 1)
  int*    island_nefc;       // number of constraints in island                  (nisland x 1)
  int*    island_iefcadr;    // start address in iefc vector                     (nisland x 1)
  int*    map_efc2iefc;      // map from efc to iefc                             (nefc x 1)
  int*    map_iefc2efc;      // map from iefc to efc                             (nefc x 1)

  // computed by mj_island (constraints sorted by island)
  int*    iefc_type;         // constraint type (mjtConstraint)                  (nefc x 1)
  int*    iefc_id;           // id of object of specified type                   (nefc x 1)
  mjtNum* iefc_frictionloss; // frictionloss (friction)                          (nefc x 1)
  mjtNum* iefc_D;            // constraint mass                                  (nefc x 1)
  mjtNum* iefc_R;            // inverse constraint mass                          (nefc x 1)

  // computed by mj_projectConstraint (PGS solver)
  int*    efc_Y_rownnz;      // number of non-zeros in Y row                     (nefc x 1)
  int*    efc_Y_rowadr;      // row start address in Y colind array              (nefc x 1)
  int*    efc_Y_colind;      // column indices in sparse Y                       (nY x 1)
  mjtNum* efc_Y;             // whitened Jacobian Y = J*M^(-1/2)                 (nY x 1)
  int*    efc_AR_rownnz;     // number of non-zeros in AR                        (nefc x 1)
  int*    efc_AR_rowadr;     // row start address in AR colind array             (nefc x 1)
  int*    efc_AR_colind;     // column indices in sparse AR                      (nA x 1)
  mjtNum* efc_AR;            // J*inv(M)*J' + R                                  (nA x 1)

  //-------------------- arena-allocated: POSITION, VELOCITY dependent

  // computed by mj_fwdVelocity/mj_referenceConstraint
  mjtNum* efc_vel;           // velocity in constraint space: J*qvel             (nefc x 1)
  mjtNum* efc_aref;          // reference pseudo-acceleration                    (nefc x 1)

  // computed when the implicit effective metric M+K is active
  mjtNum* efm_c;             // smooth-force shift h*K*qvel                      (nv x 1)
  mjtNum* efm_diag;          // effective-metric diagonal h*D + h^2*K            (nv x 1)
  mjtNum* efm_ck;            // diagonal stiffness h*k, for the smooth shift     (nv x 1)
  mjtNum* efm_sdiag;         // diagonal additions to M in the backbone          (nv x 1)
  mjtNum* efm_fluid;         // fluid drag blocks in M's sparsity pattern        (nC x 1)
  int*    efm_tid;           // ids of tendons with terms in the metric          (ntendon x 1)
  mjtNum* efm_ts;            // tendon metric scale h^2*k + h*b, tid indexed     (ntendon x 1)
  mjtNum* efm_tk;            // tendon stiffness h*k for shift, tid indexed      (ntendon x 1)
  int*    efm_aid;           // ids of actuators with terms in the metric        (nactuator x 1)
  mjtNum* efm_as;            // actuator metric scale h^2*gp + h*gv, aid indexed (nactuator x 1)
  mjtNum* efm_ak;            // actuator stiffness h*gp, aid indexed             (nactuator x 1)
  mjtNum* efm_ca;            // actuation-stage smooth-force shift               (nv x 1)
  int*    efm_K_rownnz;      // effective-stiffness CSR row nonzeros             (nv x 1)
  int*    efm_K_rowadr;      // effective-stiffness CSR row addresses            (nv x 1)
  int*    efm_K_colind;      // effective-stiffness CSR column indices           (nefmK x 1)
  mjtNum* efm_K_val;         // effective-stiffness CSR values                   (nefmK x 1)
  int*    efm_dofid;         // block k -> dof address of its vertex triple      (nefmdof x 1)
  int*    efm_con_ind;       // contact rows, packed [nnz, conid, colind...]     (nefmcon x 1)
  mjtNum* efm_con_val;       // contact rows, packed [scale, force, val...]      (nefmcon x 1)
  mjtNum* efm_L;             // factored 3x3 blocks of M+K and packed tail data  (nefmL x 1)

  //-------------------- arena-allocated: POSITION, VELOCITY, CONTROL/ACCELERATION dependent

  // computed by mj_fwdConstraint/mj_inverse
  mjtNum* efc_b;             // linear cost term: J*qacc_smooth - aref           (nefc x 1)
  mjtNum* iefc_aref;         // reference pseudo-acceleration                    (nefc x 1)
  int*    iefc_state;        // constraint state (mjtConstraintState)            (nefc x 1)
  mjtNum* iefc_force;        // constraint force in constraint space             (nefc x 1)
  int*    efc_state;         // constraint state (mjtConstraintState)            (nefc x 1)
  mjtNum* efc_force;         // constraint force in constraint space             (nefc x 1)
  mjtNum* ifrc_constraint;   // constraint force                                 (nidof x 1)

  // compilation signature
  uint64_t  signature;       // also held by the mjSpec that compiled the model
} mjData;


//---------------------------------- callback function types ---------------------------------------

// generic MuJoCo function
typedef void (*mjfGeneric)(const mjModel* m, mjData* d);

// contact filter: 1- discard, 0- collide
typedef int (*mjfConFilt)(const mjModel* m, mjData* d, int geom1, int geom2);

// sensor simulation
typedef void (*mjfSensor)(const mjModel* m, mjData* d, int stage);

// timer
typedef mjtNum (*mjfTime)(void);

// actuator dynamics, gain, bias
typedef mjtNum (*mjfAct)(const mjModel* m, const mjData* d, int id);

// collision detection
typedef int (*mjfCollision)(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                            mjtNum margin);

#endif  // MUJOCO_MJDATA_H_
