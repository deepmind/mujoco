# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import dataclasses
import enum
import operator
from typing import Callable, Optional

import mujoco
import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import util_pkg

MJ_MINVAL = mujoco.mjMINVAL
MJ_MAXVAL = mujoco.mjMAXVAL
MJ_MINIMP = mujoco.mjMINIMP  # minimum constraint impedance
MJ_MAXIMP = mujoco.mjMAXIMP  # maximum constraint impedance
MJ_MAXCONPAIR = mujoco.mjMAXCONPAIR
MJ_MINMU = mujoco.mjMINMU  # minimum friction
MJ_MINAWAKE = mujoco.mjMINAWAKE  # minimum number of timesteps before sleeping
FLEX_STIFFNESS_3D = 24 if util_pkg.check_version("mujoco>=3.14.1.dev989511280") else 21
# TODO(team): remove version check once mujoco >= 4.0.0 is required.
FLEX_INTERP_PHYS_JAC = util_pkg.check_version("mujoco>=3.17.1.dev1010658301")


# TODO(team): Remove dynamic version probe once minimum MuJoCo version
# supports curved rod references and set ROD_HINGE_SIZE = 10 directly.
def _rod_hinge_size() -> int:
  """Size of a 1D rod bending record in flex_bending of the installed MuJoCo.

  Records have 10 values since MuJoCo supports curved rod references: a single hinge
  [v0, v1, v2, -1, k, 0, kb0, 0, 0, 0] or a hinge pair [v0, v1, v2, v3, w0, w1, om0 (4)], see
  passive.rod_stencil; before, they were [v0, v1, v2, k]. The size is read from a compiled rod,
  because the first MuJoCo build with the larger record is not known when this code is written.

  Returns:
    The record size, 4 or 10.

  Raises:
    NotImplementedError: If the installed MuJoCo uses another record size.
  """
  xml = """
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="grid" count="3 1 1" spacing="1 1 1" radius="0.1" dim="1">
        <elasticity young="1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  """
  try:
    nflexbending = mujoco.MjModel.from_xml_string(xml).nflexbending
  except ValueError:  # MuJoCo without 1D flex elasticity
    return 4
  if not isinstance(nflexbending, int):
    return 4
  # one hinge: nflexbending = 1 + record size, or 0 if 1D young is ignored
  size = nflexbending - 1 if nflexbending else 4
  if size not in (4, 10):
    raise NotImplementedError(f"unsupported 1D rod record size {size} in flex_bending")
  return size


ROD_HINGE_SIZE = _rod_hinge_size()

# maximum size (by number of edges) of an horizon in EPA algorithm
MJ_MAX_EPAHORIZON = 24
# maximum average number of trianglarfaces EPA can insert at each iteration
MJ_MAX_EPAFACES = 5

TILE_SIZE_JTDAJ_SPARSE = 16
TILE_SIZE_JTDAJ_DENSE = 16

# max M block size where dense tile-Cholesky beats sparse LDL (wins to ~64, degrades past ~80)
M_BLOCK_DENSE_MAX = 64

# max M block size where scalar Cholesky beats dense tile-Cholesky
M_BLOCK_SCALAR_MAX = 6

# qLD_block_adr sentinels for factorless block layouts
Q_LD_BLOCK_COMPACT = -2
Q_LD_BLOCK_SPARSE = -1

# maximum number of plugin attributes
_NPLUGINATTR = 128

# max nodes per interpolated flex cell ((2 + 1)^3 for quadratic)
FLEX_INTERP_MAX_NODES = 27


# TODO(team): add check that all wp.launch_tiled 'block_dim' settings are configurable
@dataclasses.dataclass
class BlockDim:
  """Block dimension 'block_dim' settings for wp.launch_tiled.

  TODO(team): experimental and may be removed

  Attributes:
    segmented_sort: segmented sort block dimension (collision_driver)
    nxn_broadphase: NXN broadphase block dimension (collision_driver)
    convex_ccd: convex CCD kernel block dimension (collision_convex)
    island_dsu: island discovery DSU block dimension (island)
    ray: ray block dimension (ray)
    contact_sort: contact sort block dimension (sensor)
    energy_vel_kinetic: energy velocity kinetic block dimension (sensor)
    cholesky_factorize: block-dense Cholesky factorize block dimension (smooth)
    cholesky_factorize_solve: block-dense Cholesky factorize+solve block dimension (smooth)
    cholesky_solve: Cholesky solve block dimension (smooth)
    small_cholesky: scalar small-block Cholesky block dimension (smooth)
    solve_LD_sparse_fused: solve LD sparse fused block dimension (smooth)
    update_gradient_cholesky: update gradient Cholesky block dimension (solver)
    update_gradient_cholesky_blocked: update gradient Cholesky blocked block dimension (solver)
    update_gradient_JTDAJ_sparse: update gradient JTDAJ sparse block dimension (solver)
    update_gradient_JTDAJ_dense: update gradient JTDAJ dense block dimension (solver)
    linesearch_iterative: linesearch iterative block dimension (solver)
    update_gradient_grad: update gradient grad block dimension (solver)
    solve_beta_accumulate: solve beta accumulate block dimension (solver)
    solve_search_update_cg: solve search update CG block dimension (solver)
    solve_init_search_cg: solve init search CG block dimension (solver)
    contact_jac_tiled: contact Jacobian tiled block dimension (solver)
    eff_pcg: effective-metric PCG block dimension (derivative)
    render: render block dimension (render)
  """

  # collision_driver
  segmented_sort: int = 128
  nxn_broadphase: int = 128
  # collision_convex
  convex_ccd: int = 64
  # island
  island_dsu: int = 32
  # ray
  ray: int = 64
  # sensor
  contact_sort: int = 64
  energy_vel_kinetic: int = 32
  # smooth -- block tile-Cholesky widths
  cholesky_factorize: int = 32
  cholesky_factorize_solve: int = 32
  cholesky_solve: int = 64
  small_cholesky: int = 64
  solve_LD_sparse_fused: int = 128
  # solver
  update_gradient_cholesky: int = 64
  update_gradient_cholesky_blocked: int = 32
  update_gradient_JTDAJ_sparse: int = 128
  update_gradient_JTDAJ_dense: int = 128
  linesearch_iterative: int = 32
  update_gradient_grad: int = 256
  solve_beta_accumulate: int = 256
  solve_search_update_cg: int = 256
  solve_init_search_cg: int = 256
  contact_jac_tiled: int = 32
  # derivative
  eff_pcg: int = 128
  # render
  render: int = 64


class BroadphaseType(enum.IntEnum):
  """Type of broadphase algorithm.

  Attributes:
     NXN: Broad phase checking all pairs
     SAP_TILE: Sweep and prune broad phase using tile sort
     SAP_SEGMENTED: Sweep and prune broad phase using segment sort
  """

  NXN = 0
  SAP_TILE = 1
  SAP_SEGMENTED = 2


class BroadphaseFilter(enum.IntFlag):
  """Bitmask specifying which collision functions to run during broadphase.

  Attributes:
    PLANE: collision between bounding sphere and plane
    SPHERE: collision between bounding spheres
    AABB: collision between axis-aligned bounding boxes
    OBB: collision between oriented bounding boxes
  """

  PLANE = 1 << 0
  SPHERE = 1 << 1
  AABB = 1 << 2
  OBB = 1 << 3


class OverflowType(enum.IntFlag):
  """Bitmask for physics and collision overflows.

  Attributes:
    NONE: no overflow
    NEFC: nefc > njmax
    NJMAX_NNZ: njmax_nnz overflow
    BROADPHASE: broadphase overflow / flex broadphase overflow
    NARROWPHASE: narrowphase overflow / flex narrowphase / contact overflow
    CCD: CCD overflow / flex CCD overflow
    HFIELD: height field collision overflow
    CONTACT_MATCH: contact match sensor overflow
    NVMAX: nvmax overflow (islands)
    EPA_HORIZON: EPA horizon buffer overflow
    ITERATIONS: solver iteration limit reached
    LS_ITERATIONS: linesearch iteration limit reached
    TACTILE: tactile sensor collision pair overflow
    ALL: all overflows
  """

  NONE = 0
  NEFC = 1 << 0
  NJMAX_NNZ = 1 << 1
  BROADPHASE = 1 << 2
  NARROWPHASE = 1 << 3
  CCD = 1 << 4
  HFIELD = 1 << 5
  CONTACT_MATCH = 1 << 6
  NVMAX = 1 << 7
  EPA_HORIZON = 1 << 8
  ITERATIONS = 1 << 9
  LS_ITERATIONS = 1 << 10
  TACTILE = 1 << 11
  ALL = (
    NEFC
    | NJMAX_NNZ
    | BROADPHASE
    | NARROWPHASE
    | CCD
    | HFIELD
    | CONTACT_MATCH
    | NVMAX
    | EPA_HORIZON
    | ITERATIONS
    | LS_ITERATIONS
    | TACTILE
  )


class CamLightType(enum.IntEnum):
  """Type of camera light.

  Attributes:
    FIXED: pos and rot fixed in body
    TRACK: pos tracks body, rot fixed in global
    TRACKCOM: pos tracks subtree com, rot fixed in body
    TARGETBODY: pos fixed in body, rot tracks target body
    TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
  """

  FIXED = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
  TRACK = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
  TRACKCOM = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
  TARGETBODY = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  TARGETBODYCOM = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM


class ProjectionType(enum.IntEnum):
  """Type of camera projection.

  Attributes:
    PERSPECTIVE: perspective projection
    ORTHOGRAPHIC: orthographic projection
  """

  PERSPECTIVE = mujoco.mjtProjection.mjPROJ_PERSPECTIVE
  ORTHOGRAPHIC = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC


class Stage(enum.IntEnum):
  """Computation stage.

  Attributes:
    POS: position-dependent
    VEL: velocity-dependent
    ACC: acceleration/force-dependent
  """

  POS = mujoco.mjtStage.mjSTAGE_POS
  VEL = mujoco.mjtStage.mjSTAGE_VEL
  ACC = mujoco.mjtStage.mjSTAGE_ACC


class DataType(enum.IntFlag):
  """Sensor data types.

  Attributes:
    REAL: real values, no constraints
    POSITIVE: positive values, 0 or negative: inactive
  """

  REAL = mujoco.mjtDataType.mjDATATYPE_REAL
  POSITIVE = mujoco.mjtDataType.mjDATATYPE_POSITIVE
  # unsupported: AXIS, QUATERNION


class DeterminismType(enum.IntFlag):
  """Bitmask for deterministic execution.

  Attributes:
    NONE: non-deterministic execution
    CONTACTS: stable contact ordering via geometric radix sort
    CONSTRAINT: canonical constraint row ordering
    ATOMICS: invariant floating-point atomic reductions
    ALL: all determinism features
  """

  NONE = 0
  CONTACTS = 1 << 0
  CONSTRAINT = 1 << 1
  ATOMICS = 1 << 2
  ALL = CONTACTS | CONSTRAINT | ATOMICS


class DisableBit(enum.IntFlag):
  """Disable default feature bitflags.

  Attributes:
    CONSTRAINT:   entire constraint solver
    EQUALITY:     equality constraints
    FRICTIONLOSS: joint and tendon frictionloss constraints
    LIMIT:        joint and tendon limit constraints
    CONTACT:      contact constraints
    SPRING:       passive spring forces
    DAMPER:       passive damper forces
    GRAVITY:      gravitational forces
    CLAMPCTRL:    clamp control to specified range
    WARMSTART:    warmstart constraint solver
    FILTERPARENT: disable collisions between parent and child bodies
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
    SENSOR:       sensors
    EULERDAMP:    implicit damping for Euler integration
    NATIVECCD:    native convex collision detection (ignored in MJWarp)
    ISLAND:       constraint islands
    MULTICCD:     disable multiple contacts with CCD
  """

  CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
  EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
  FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
  LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
  CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
  SPRING = mujoco.mjtDisableBit.mjDSBL_SPRING
  DAMPER = mujoco.mjtDisableBit.mjDSBL_DAMPER
  GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
  CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
  WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART
  FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
  REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
  SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
  EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  NATIVECCD = mujoco.mjtDisableBit.mjDSBL_NATIVECCD
  ISLAND = mujoco.mjtDisableBit.mjDSBL_ISLAND
  MULTICCD = mujoco.mjtDisableBit.mjDSBL_MULTICCD
  # unsupported: MIDPHASE, AUTORESET


class EnableBit(enum.IntFlag):
  """Enable optional feature bitflags.

  Attributes:
    ENERGY: energy computation
    INVDISCRETE: discrete-time inverse dynamics
    SLEEP: sleeping
  """

  ENERGY = mujoco.mjtEnableBit.mjENBL_ENERGY
  INVDISCRETE = mujoco.mjtEnableBit.mjENBL_INVDISCRETE
  SLEEP = mujoco.mjtEnableBit.mjENBL_SLEEP
  # unsupported: OVERRIDE, FWDINV, ISLAND, DIAGEXACT


class SleepPolicy(enum.IntEnum):
  """Per-tree sleep policy.

  Attributes:
    AUTO: compiler chooses sleep policy
    AUTO_NEVER: compiler sleep policy: never
    AUTO_ALLOWED: compiler sleep policy: allowed
  """

  AUTO = mujoco.mjtSleepPolicy.mjSLEEP_AUTO
  AUTO_NEVER = mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER
  AUTO_ALLOWED = mujoco.mjtSleepPolicy.mjSLEEP_AUTO_ALLOWED
  # unsupported: NEVER, ALLOWED, INIT


class SleepState(enum.IntEnum):
  """Sleep state for bodies.

  Attributes:
    STATIC: body is static (world body or mocap)
    ASLEEP: body is asleep
    AWAKE: body is awake
  """

  STATIC = mujoco.mjtSleepState.mjS_STATIC
  ASLEEP = mujoco.mjtSleepState.mjS_ASLEEP
  AWAKE = mujoco.mjtSleepState.mjS_AWAKE


class TrnType(enum.IntEnum):
  """Type of actuator transmission.

  Attributes:
    JOINT: force on joint
    JOINTINPARENT: force on joint, expressed in parent frame
    SLIDERCRANK: force via slider-crank linkage
    TENDON: force on tendon
    BODY: adhesion force on body's geoms
    SITE: force on site
  """

  JOINT = mujoco.mjtTrn.mjTRN_JOINT
  JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT
  SLIDERCRANK = mujoco.mjtTrn.mjTRN_SLIDERCRANK
  TENDON = mujoco.mjtTrn.mjTRN_TENDON
  BODY = mujoco.mjtTrn.mjTRN_BODY
  SITE = mujoco.mjtTrn.mjTRN_SITE


class DynType(enum.IntEnum):
  """Type of actuator dynamics.

  Attributes:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
    MUSCLE: piece-wise linear filter with two time constants
    USER: user-defined dynamics via act_dyn_callback
    DCMOTOR: DC motor dynamics
  """

  NONE = mujoco.mjtDyn.mjDYN_NONE
  INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR
  FILTER = mujoco.mjtDyn.mjDYN_FILTER
  FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT
  MUSCLE = mujoco.mjtDyn.mjDYN_MUSCLE
  USER = mujoco.mjtDyn.mjDYN_USER
  DCMOTOR = mujoco.mjtDyn.mjDYN_DCMOTOR


class GainType(enum.IntEnum):
  """Type of actuator gain.

  Attributes:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle FLV curve computed by muscle_gain
    USER: user-defined gain via act_gain_callback
    DCMOTOR: DC motor gain
  """

  FIXED = mujoco.mjtGain.mjGAIN_FIXED
  AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
  MUSCLE = mujoco.mjtGain.mjGAIN_MUSCLE
  USER = mujoco.mjtGain.mjGAIN_USER
  DCMOTOR = mujoco.mjtGain.mjGAIN_DCMOTOR
  SO3 = mujoco.mjtGain.mjGAIN_SO3


class BiasType(enum.IntEnum):
  """Type of actuator bias.

  Attributes:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle passive force computed by muscle_bias
    USER: user-defined bias via act_bias_callback
    DCMOTOR: DC motor back-EMF bias
  """

  NONE = mujoco.mjtBias.mjBIAS_NONE
  AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
  MUSCLE = mujoco.mjtBias.mjBIAS_MUSCLE
  USER = mujoco.mjtBias.mjBIAS_USER
  DCMOTOR = mujoco.mjtBias.mjBIAS_DCMOTOR


class CtrlInput(enum.IntFlag):
  """Control input signature bitflags for actuators.

  Attributes:
    POS: position setpoint input
    VEL: velocity setpoint input
    FF: feedforward input
    VOLTAGE: raw terminal voltage input
    NONE: explicitly no inputs (purely passive)
  """

  POS = mujoco.mjtCtrlInput.mjINPUT_POS
  VEL = mujoco.mjtCtrlInput.mjINPUT_VEL
  FF = mujoco.mjtCtrlInput.mjINPUT_FF
  VOLTAGE = mujoco.mjtCtrlInput.mjINPUT_VOLTAGE
  NONE = mujoco.mjtCtrlInput.mjINPUT_NONE


class CtrlChart(enum.IntEnum):
  """Orientation chart type for SO(3) actuators.

  Attributes:
    EXPMAP: exponential-map orientation target (3 controls)
    QUAT: quaternion orientation target (4 controls)
  """

  EXPMAP = mujoco.mjtCtrlChart.mjCHART_EXPMAP
  QUAT = mujoco.mjtCtrlChart.mjCHART_QUAT


class JointType(enum.IntEnum):
  """Type of degree of freedom.

  Attributes:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """

  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE


class ConeType(enum.IntEnum):
  """Type of friction cone.

  Attributes:
    PYRAMIDAL: pyramidal
    ELLIPTIC: elliptic
  """

  PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
  ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC


class IntegratorType(enum.IntEnum):
  """Integrator mode.

  Attributes:
    EULER: semi-implicit Euler
    RK4: 4th-order Runge Kutta
    IMPLICITFAST: implicit in velocity, no rne derivative
    IMPLICIT: implicit in velocity, with rne derivative
    DISCRETE: discrete step map: constraint solve in the effective metric
  """

  EULER = mujoco.mjtIntegrator.mjINT_EULER
  RK4 = mujoco.mjtIntegrator.mjINT_RK4
  IMPLICITFAST = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  IMPLICIT = mujoco.mjtIntegrator.mjINT_IMPLICIT
  DISCRETE = mujoco.mjtIntegrator.mjINT_DISCRETE


class GeomType(enum.IntEnum):
  """Type of geometry.

  Attributes:
    PLANE: plane
    HFIELD: heightfield
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
    SDF: sdf
    FLEX: flex
  """

  PLANE = mujoco.mjtGeom.mjGEOM_PLANE
  HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
  SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
  CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
  ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
  CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
  BOX = mujoco.mjtGeom.mjGEOM_BOX
  MESH = mujoco.mjtGeom.mjGEOM_MESH
  SDF = mujoco.mjtGeom.mjGEOM_SDF
  FLEX = mujoco.mjtGeom.mjGEOM_FLEX
  # warp only
  TRIANGLE = 999
  # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


class CollisionType(enum.IntEnum):
  """Type of narrowphase collision.

  Attributes:
    PRIMITIVE: primitive collision
    CONVEX: convex collision (CCD)
    SDF: sdf collision
  """

  PRIMITIVE = 0
  CONVEX = 1
  SDF = 2


class SolverType(enum.IntEnum):
  """Constraint solver algorithm.

  Attributes:
    CG: Conjugate gradient (primal)
    NEWTON: Newton (primal)
  """

  CG = mujoco.mjtSolver.mjSOL_CG
  NEWTON = mujoco.mjtSolver.mjSOL_NEWTON
  # unsupported: PGS


class ConstraintState(enum.IntEnum):
  """State of constraint.

  Attributes:
    SATISFIED: constraint satisfied, zero cost (limit, contact)
    QUADRATIC: quadratic cost (equality, friction, limit, contact)
    LINEARNEG: linear cost, negative side (friction)
    LINEARPOS: linear cost, positive side (friction)
    CONE: square distance to cone cost (elliptic contact)
  """

  SATISFIED = mujoco.mjtConstraintState.mjCNSTRSTATE_SATISFIED
  QUADRATIC = mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC
  LINEARNEG = mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARNEG
  LINEARPOS = mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARPOS
  CONE = mujoco.mjtConstraintState.mjCNSTRSTATE_CONE


class ConstraintType(enum.IntEnum):
  """Type of constraint.

  Attributes:
    EQUALITY: equality constraint
    FRICTION_DOF: dof friction
    FRICTION_TENDON: tendon friction
    LIMIT_JOINT: joint limit
    LIMIT_TENDON: tendon limit
    CONTACT_FRICTIONLESS: frictionless contact
    CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
    CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
  """

  EQUALITY = mujoco.mjtConstraint.mjCNSTR_EQUALITY
  FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
  FRICTION_TENDON = mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON
  LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
  LIMIT_TENDON = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
  CONTACT_FRICTIONLESS = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
  CONTACT_PYRAMIDAL = mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
  CONTACT_ELLIPTIC = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC


class SensorType(enum.IntEnum):
  """Type of sensor.

  Attributes:
    MAGNETOMETER: magnetometer
    CAMPROJECTION: camera projection
    RANGEFINDER: scalar distance to nearest geom or site along z-axis
    JOINTPOS: joint position
    TENDONPOS: scalar tendon position
    ACTUATORPOS: actuator position
    BALLQUAT: ball joint orientation
    JOINTLIMITPOS: joint limit distance-margin
    TENDONLIMITPOS: tendon limit distance-margin
    FRAMEPOS: frame position
    FRAMEXAXIS: frame x-axis
    FRAMEYAXIS: frame y-axis
    FRAMEZAXIS: frame z-axis
    FRAMEQUAT: frame orientation, represented as quaternion
    SUBTREECOM: subtree center of mass
    GEOMDIST: signed distance between two geoms
    GEOMNORMAL: normal direction between two geoms
    GEOMFROMTO: segment between two geoms
    INSIDESITE: 1 if object is inside site, 0 otherwise
    E_POTENTIAL: potential energy
    E_KINETIC: kinetic energy
    CLOCK: simulation time
    VELOCIMETER: 3D linear velocity, in local frame
    GYRO: 3D angular velocity, in local frame
    JOINTVEL: joint velocity
    TENDONVEL: scalar tendon velocity
    ACTUATORVEL: actuator velocity
    BALLANGVEL: ball joint angular velocity
    JOINTLIMITVEL: joint limit velocity
    TENDONLIMITVEL: tendon limit velocity
    FRAMELINVEL: 3D linear velocity
    FRAMEANGVEL: 3D angular velocity
    SUBTREELINVEL: subtree linear velocity
    SUBTREEANGMOM: subtree angular momentum
    TOUCH: scalar contact normal forces summed over sensor zone
    CONTACT: contacts which occurred during the simulation
    ACCELEROMETER: accelerometer
    FORCE: force
    TORQUE: torque
    ACTUATORFRC: scalar actuator force, measured at the joint
    TENDONACTFRC: scalar actuator force, measured at the tendon
    JOINTACTFRC: scalar actuator force, measured at the joint
    JOINTLIMITFRC: joint limit force
    TENDONLIMITFRC: tendon limit force
    FRAMELINACC: 3D linear acceleration
    FRAMEANGACC: 3D angular acceleration
    TACTILE: tactile sensor
    USER: user-defined sensor via sensor_callback
  """

  MAGNETOMETER = mujoco.mjtSensor.mjSENS_MAGNETOMETER
  CAMPROJECTION = mujoco.mjtSensor.mjSENS_CAMPROJECTION
  RANGEFINDER = mujoco.mjtSensor.mjSENS_RANGEFINDER
  JOINTPOS = mujoco.mjtSensor.mjSENS_JOINTPOS
  TENDONPOS = mujoco.mjtSensor.mjSENS_TENDONPOS
  ACTUATORPOS = mujoco.mjtSensor.mjSENS_ACTUATORPOS
  BALLQUAT = mujoco.mjtSensor.mjSENS_BALLQUAT
  JOINTLIMITPOS = mujoco.mjtSensor.mjSENS_JOINTLIMITPOS
  TENDONLIMITPOS = mujoco.mjtSensor.mjSENS_TENDONLIMITPOS
  FRAMEPOS = mujoco.mjtSensor.mjSENS_FRAMEPOS
  FRAMEXAXIS = mujoco.mjtSensor.mjSENS_FRAMEXAXIS
  FRAMEYAXIS = mujoco.mjtSensor.mjSENS_FRAMEYAXIS
  FRAMEZAXIS = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
  FRAMEQUAT = mujoco.mjtSensor.mjSENS_FRAMEQUAT
  SUBTREECOM = mujoco.mjtSensor.mjSENS_SUBTREECOM
  GEOMDIST = mujoco.mjtSensor.mjSENS_GEOMDIST
  GEOMNORMAL = mujoco.mjtSensor.mjSENS_GEOMNORMAL
  GEOMFROMTO = mujoco.mjtSensor.mjSENS_GEOMFROMTO
  INSIDESITE = mujoco.mjtSensor.mjSENS_INSIDESITE
  E_POTENTIAL = mujoco.mjtSensor.mjSENS_E_POTENTIAL
  E_KINETIC = mujoco.mjtSensor.mjSENS_E_KINETIC
  CLOCK = mujoco.mjtSensor.mjSENS_CLOCK
  VELOCIMETER = mujoco.mjtSensor.mjSENS_VELOCIMETER
  GYRO = mujoco.mjtSensor.mjSENS_GYRO
  JOINTVEL = mujoco.mjtSensor.mjSENS_JOINTVEL
  TENDONVEL = mujoco.mjtSensor.mjSENS_TENDONVEL
  ACTUATORVEL = mujoco.mjtSensor.mjSENS_ACTUATORVEL
  BALLANGVEL = mujoco.mjtSensor.mjSENS_BALLANGVEL
  JOINTLIMITVEL = mujoco.mjtSensor.mjSENS_JOINTLIMITVEL
  TENDONLIMITVEL = mujoco.mjtSensor.mjSENS_TENDONLIMITVEL
  FRAMELINVEL = mujoco.mjtSensor.mjSENS_FRAMELINVEL
  FRAMEANGVEL = mujoco.mjtSensor.mjSENS_FRAMEANGVEL
  SUBTREELINVEL = mujoco.mjtSensor.mjSENS_SUBTREELINVEL
  SUBTREEANGMOM = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
  TOUCH = mujoco.mjtSensor.mjSENS_TOUCH
  CONTACT = mujoco.mjtSensor.mjSENS_CONTACT
  ACCELEROMETER = mujoco.mjtSensor.mjSENS_ACCELEROMETER
  FORCE = mujoco.mjtSensor.mjSENS_FORCE
  TORQUE = mujoco.mjtSensor.mjSENS_TORQUE
  ACTUATORFRC = mujoco.mjtSensor.mjSENS_ACTUATORFRC
  TENDONACTFRC = mujoco.mjtSensor.mjSENS_TENDONACTFRC
  JOINTACTFRC = mujoco.mjtSensor.mjSENS_JOINTACTFRC
  JOINTLIMITFRC = mujoco.mjtSensor.mjSENS_JOINTLIMITFRC
  TENDONLIMITFRC = mujoco.mjtSensor.mjSENS_TENDONLIMITFRC
  FRAMELINACC = mujoco.mjtSensor.mjSENS_FRAMELINACC
  FRAMEANGACC = mujoco.mjtSensor.mjSENS_FRAMEANGACC
  TACTILE = mujoco.mjtSensor.mjSENS_TACTILE
  USER = mujoco.mjtSensor.mjSENS_USER


class RayDataField(enum.IntEnum):
  """Ray data fields for rangefinder sensors.

  Attributes:
    DIST: distance from ray origin to nearest surface
    DIR: normalized ray direction
    ORIGIN: ray origin
    POINT: point at which ray intersects nearest surface
    NORMAL: surface normal at intersection point
    DEPTH: depth along z-axis
  """

  DIST = mujoco.mjtRayDataField.mjRAYDATA_DIST
  DIR = mujoco.mjtRayDataField.mjRAYDATA_DIR
  ORIGIN = mujoco.mjtRayDataField.mjRAYDATA_ORIGIN
  POINT = mujoco.mjtRayDataField.mjRAYDATA_POINT
  NORMAL = mujoco.mjtRayDataField.mjRAYDATA_NORMAL
  DEPTH = mujoco.mjtRayDataField.mjRAYDATA_DEPTH


class ConDataField(enum.IntEnum):
  """Contact data fields for contact sensors.

  Attributes:
    FOUND: number of matching contacts found
    FORCE: contact force in contact frame
    TORQUE: contact torque in contact frame
    DIST: penetration distance
    POS: contact position
    NORMAL: contact frame normal
    TANGENT: contact frame first tangent
    LINVEL: relative linear velocity in contact frame
    ANGVEL: relative angular velocity in contact frame
  """

  FOUND = mujoco.mjtConDataField.mjCONDATA_FOUND
  FORCE = mujoco.mjtConDataField.mjCONDATA_FORCE
  TORQUE = mujoco.mjtConDataField.mjCONDATA_TORQUE
  DIST = mujoco.mjtConDataField.mjCONDATA_DIST
  POS = mujoco.mjtConDataField.mjCONDATA_POS
  NORMAL = mujoco.mjtConDataField.mjCONDATA_NORMAL
  TANGENT = mujoco.mjtConDataField.mjCONDATA_TANGENT
  LINVEL = mujoco.mjtConDataField.mjCONDATA_LINVEL
  ANGVEL = mujoco.mjtConDataField.mjCONDATA_ANGVEL


class ObjType(enum.IntEnum):
  """Type of object.

  Attributes:
    UNKNOWN: unknown object type
    BODY: body
    XBODY: body, used to access regular frame instead of i-frame
    JOINT: joint
    GEOM: geom
    FLEX: flex
    SITE: site
    CAMERA: camera
    TENDON: tendon
  """

  UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN
  BODY = mujoco.mjtObj.mjOBJ_BODY
  XBODY = mujoco.mjtObj.mjOBJ_XBODY
  JOINT = mujoco.mjtObj.mjOBJ_JOINT
  GEOM = mujoco.mjtObj.mjOBJ_GEOM
  FLEX = mujoco.mjtObj.mjOBJ_FLEX
  SITE = mujoco.mjtObj.mjOBJ_SITE
  CAMERA = mujoco.mjtObj.mjOBJ_CAMERA
  TENDON = mujoco.mjtObj.mjOBJ_TENDON


class EqType(enum.IntEnum):
  """Type of equality constraint.

  Attributes:
    CONNECT: connect two bodies at a point (ball joint)
    JOINT: couple the values of two scalar joints with cubic
    WELD: fix relative position and orientation of two bodies
    TENDON: couple the lengths of two tendons with cubic
    FLEX: couple the edge lengths of a flex
    FLEXSTRAIN: strain constraint for interpolated flex
  """

  CONNECT = mujoco.mjtEq.mjEQ_CONNECT
  WELD = mujoco.mjtEq.mjEQ_WELD
  JOINT = mujoco.mjtEq.mjEQ_JOINT
  TENDON = mujoco.mjtEq.mjEQ_TENDON
  FLEX = mujoco.mjtEq.mjEQ_FLEX
  FLEXSTRAIN = mujoco.mjtEq.mjEQ_FLEXSTRAIN
  # unsupported: DISTANCE


class TendonType(enum.IntEnum):
  """Type of tendon.

  Attributes:
    FIXED: linear combination of joint coordinates
    SPATIAL: path through sites, around geoms
  """

  # TODO(team): use mujoco.mjtTendon when mujoco>=3.17.1 is required
  FIXED = 0
  SPATIAL = 1


class WrapType(enum.IntEnum):
  """Type of tendon wrapping object.

  Attributes:
    JOINT: constant moment arm
    PULLEY: pulley used to split tendon
    SITE: pass through site
    SPHERE: wrap around sphere
    CYLINDER: wrap around (infinite) cylinder
  """

  JOINT = mujoco.mjtWrap.mjWRAP_JOINT
  PULLEY = mujoco.mjtWrap.mjWRAP_PULLEY
  SITE = mujoco.mjtWrap.mjWRAP_SITE
  SPHERE = mujoco.mjtWrap.mjWRAP_SPHERE
  CYLINDER = mujoco.mjtWrap.mjWRAP_CYLINDER


class State(enum.IntEnum):
  """State component elements as integer bitflags.

  Includes several convenient combinations of these flags.

  Attributes:
    TIME: time
    QPOS: position
    QVEL: velocity
    ACT: actuator activation
    HISTORY: delay/interval history buffers
    PLASTIC: spring plastic references
    WARMSTART: acceleration used for warmstart
    CTRL: control
    QFRC_APPLIED: applied generalized force
    XFRC_APPLIED: applied Cartesian force/torque
    EQ_ACTIVE: enable/disable constraints
    MOCAP_POS: positions of mocap bodies
    MOCAP_QUAT: orientations of mocap bodies
    NSTATE: number of state elements
    PHYSICS: TIME | QPOS | QVEL | ACT | HISTORY | PLASTIC
    FULLPHYSICS: TIME | PHYSICS | PLUGIN
    USER: CTRL | QFRC_APPLIED | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT | USERDATA
    INTEGRATION: FULLPHYSICS | USER | WARMSTART
  """

  TIME = mujoco.mjtState.mjSTATE_TIME
  QPOS = mujoco.mjtState.mjSTATE_QPOS
  QVEL = mujoco.mjtState.mjSTATE_QVEL
  ACT = mujoco.mjtState.mjSTATE_ACT
  HISTORY = mujoco.mjtState.mjSTATE_HISTORY
  WARMSTART = mujoco.mjtState.mjSTATE_WARMSTART
  CTRL = mujoco.mjtState.mjSTATE_CTRL
  QFRC_APPLIED = mujoco.mjtState.mjSTATE_QFRC_APPLIED
  XFRC_APPLIED = mujoco.mjtState.mjSTATE_XFRC_APPLIED
  EQ_ACTIVE = mujoco.mjtState.mjSTATE_EQ_ACTIVE
  MOCAP_POS = mujoco.mjtState.mjSTATE_MOCAP_POS
  MOCAP_QUAT = mujoco.mjtState.mjSTATE_MOCAP_QUAT
  USERDATA = mujoco.mjtState.mjSTATE_USERDATA
  # unsupported: PLUGIN
  PLASTIC = getattr(mujoco.mjtState, "mjSTATE_PLASTIC", 1 << 14)
  NSTATE = mujoco.mjtState.mjNSTATE
  PHYSICS = mujoco.mjtState.mjSTATE_PHYSICS  # includes HISTORY and PLASTIC
  FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS
  USER = mujoco.mjtState.mjSTATE_USER
  INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION


class SetConstGroup(enum.IntFlag):
  """Groups of set_const computations.

  Attributes:
    NONE: no group
    FIXED: independent of the configuration
    KIN0: kinematics in qpos0
    INERT0: inertia matrix in qpos0
    ALL: all supported groups
  """

  NONE = 0
  # TODO(team): replace hardcoded values with mujoco values
  FIXED = 1 << 0
  KIN0 = 1 << 1
  INERT0 = 1 << 2
  # unsupported: SPRING
  ALL = FIXED | KIN0 | INERT0


class vec5f(wp.types.vector(length=5, dtype=float)):
  pass


class vec6f(wp.types.vector(length=6, dtype=float)):
  pass


class vec6i(wp.types.vector(length=6, dtype=int)):
  pass


class vec8f(wp.types.vector(length=8, dtype=float)):
  pass


class vec8i(wp.types.vector(length=8, dtype=int)):
  pass


class vec16f(wp.types.vector(length=16, dtype=float)):
  pass


class vec16i(wp.types.vector(length=16, dtype=int)):
  pass


class vec10f(wp.types.vector(length=10, dtype=float)):
  pass


class vec11f(wp.types.vector(length=11, dtype=float)):
  pass


class vec12f(wp.types.vector(length=12, dtype=float)):
  pass


class vec_pluginattr(wp.types.vector(length=_NPLUGINATTR, dtype=float)):
  pass


class mat23f(wp.types.matrix(shape=(2, 3), dtype=float)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=float)):
  pass


class mat53f(wp.types.matrix(shape=(5, 3), dtype=float)):
  pass


class mat63f(wp.types.matrix(shape=(6, 3), dtype=float)):
  pass


class mat55f(wp.types.matrix(shape=(5, 5), dtype=float)):
  pass


class mat66f(wp.types.matrix(shape=(6, 6), dtype=float)):
  pass


class mat113f(wp.types.matrix(shape=(11, 3), dtype=float)):
  pass


vec5 = vec5f
vec6 = vec6f
vec8 = vec8f
vec10 = vec10f
vec11 = vec11f
vec12 = vec12f
vec16 = vec16f
vec128 = vec_pluginattr
mat23 = mat23f
mat43 = mat43f
mat53 = mat53f
mat63 = mat63f
mat55 = mat55f
mat66 = mat66f
mat113 = mat113f


def array(*args) -> wp.array:
  """A wrapper around wp.array that adds extra metadata to ease type introspection.

  Format is array(dim_1, dim_2, ..., dtype).  dim may be a constant int, or reference a size from
  Model or Data (e.g. "nq" or "nworld").  dim may also be "*", which means any nonzero size.
  """
  shape, dtype = args[:-1], args[-1]

  arr = wp.array(ndim=len(shape), dtype=dtype)
  arr.shape = shape

  return arr


@dataclasses.dataclass
class Option:
  """Physics options.

  Attributes:
    timestep: simulation timestep
    tolerance: main solver tolerance
    ls_tolerance: CG/Newton linesearch tolerance
    noslip_tolerance: noslip solver tolerance
    ccd_tolerance: convex collision detection tolerance
    sleep_tolerance: sleep velocity tolerance
    gravity: gravitational acceleration
    wind: wind (for lift, drag, and viscosity)
    magnetic: global magnetic flux
    density: density of medium
    viscosity: viscosity of medium
    integrator: integration mode (IntegratorType)
    cone: type of friction cone (ConeType)
    solver: solver algorithm (SolverType)
    iterations: number of main solver iterations
    ls_iterations: maximum number of CG/Newton linesearch iterations
    noslip_iterations: maximum number of noslip solver iterations
    ccd_iterations: number of iterations in convex collision detection
    disableflags: bit flags for disabling standard features
    enableflags: bit flags for enabling optional features
    sdf_initpoints: number of starting points for gradient descent
    sdf_iterations: max number of iterations for gradient descent

  warp only fields:
    impratio_invsqrt: ratio of friction-to-normal contact impedance (stored as inverse square root)
    broadphase: broadphase type (BroadphaseType)
    broadphase_filter: broadphase filter bitflag (BroadphaseFilter)
    graph_conditional: flag to use cuda graph conditional
    run_collision_detection: if False, skips collision detection and allows user-populated
      contacts during the physics step (as opposed to DisableBit.CONTACT which explicitly
      zeros out the contacts at each step)
    run_rne_postconstraint: if True, evaluates rne_postconstraint after the solver step even if
      sensors are disabled or no sensors are present
    run_qacc_smooth: if False, the Newton solver starts every world from qacc_warmstart and the
      step skips computing qacc_smooth (left stale) and factorizing M (experimental); ignored with
      CG, disabled warmstart, noslip, or zero njmax or iterations
    contact_sensor_maxmatch: max number of contacts considered by contact sensor matching criteria
                             contacts matched after this value is exceded will be ignored
    warn_overflow: overflow warning bitmask (OverflowType)
    deterministic: determinism bitmask (DeterminismType)
  """

  timestep: array("*", float)
  tolerance: array("*", float)
  ls_tolerance: array("*", float)
  noslip_tolerance: array("*", float)
  ccd_tolerance: array("*", float)
  sleep_tolerance: array("*", float)
  gravity: array("*", wp.vec3)
  wind: array("*", wp.vec3)
  magnetic: array("*", wp.vec3)
  density: array("*", float)
  viscosity: array("*", float)
  integrator: int
  cone: int
  solver: int
  iterations: int
  ls_iterations: int
  noslip_iterations: int
  ccd_iterations: int
  disableflags: int
  enableflags: int
  sdf_initpoints: int
  sdf_iterations: int
  # warp only fields:
  impratio_invsqrt: array("*", float)
  broadphase: BroadphaseType
  broadphase_filter: BroadphaseFilter
  graph_conditional: bool
  run_collision_detection: bool
  run_rne_postconstraint: bool
  run_qacc_smooth: bool
  contact_sensor_maxmatch: int
  warn_overflow: int
  deterministic: int

  @property
  def warn_overflow(self) -> int:
    return self._warn_overflow

  @warn_overflow.setter
  def warn_overflow(self, value: bool | int):
    if isinstance(value, bool):
      value = int(OverflowType.ALL) if value else 0
    self._warn_overflow = value

  @property
  def deterministic(self) -> int:
    return self._deterministic

  @deterministic.setter
  def deterministic(self, value: bool | int | None):
    if value is None or isinstance(value, (bool, np.bool_)):
      value = DeterminismType.ALL if value else DeterminismType.NONE
    value = operator.index(value)
    if value < 0:
      # complements, e.g. ~DeterminismType.CONTACTS is negative on Python 3.10
      value &= int(DeterminismType.ALL)
    if value & ~int(DeterminismType.ALL):
      raise ValueError(f"deterministic must be a DeterminismType bitmask, got {value}")
    self._deterministic = value

  # TODO(team): remove in future version
  @property
  def ls_parallel(self) -> bool:
    raise AttributeError("ls_parallel was removed in MuJoCo Warp 3.9.1.")

  @ls_parallel.setter
  def ls_parallel(self, value: bool):
    raise AttributeError("ls_parallel was removed in MuJoCo Warp 3.9.1.")

  @property
  def ls_parallel_min_step(self) -> float:
    raise AttributeError("ls_parallel_min_step was removed in MuJoCo Warp 3.9.1.")

  @ls_parallel_min_step.setter
  def ls_parallel_min_step(self, value: float):
    raise AttributeError("ls_parallel_min_step was removed in MuJoCo Warp 3.9.1.")


@dataclasses.dataclass
class TileSet:
  """Tiling configuration for decomposable block diagonal matrix.

  For non-square, non-block-diagonal tiles, use two tilesets.

  Attributes:
    adr: address of each tile in the set
    size: size of all the tiles in this set
    elemid: flat CSR gather indices for tiled blocks; empty selects the native-layout scalar path
  """

  adr: wp.array[int]
  size: int
  elemid: wp.array[int] = dataclasses.field(default_factory=lambda: wp.array([], dtype=int))

  def __eq__(self, other) -> bool:
    if self.__class__ is not other.__class__:
      return NotImplemented
    return self.size == other.size and np.array_equal(np.asarray(self.adr.numpy()), np.asarray(other.adr.numpy()))

  def __hash__(self) -> int:
    adr = np.asarray(self.adr.numpy())
    return hash((self.size, adr.dtype.str, adr.shape, adr.tobytes()))


@dataclasses.dataclass
class Callback:
  """Callbacks for custom physics behavior.

  Attributes:
    passive: custom passive forces, writes to ``Data.qfrc_passive``
    control: custom control laws, writes to ``Data.ctrl``
    act_dyn: custom actuator dynamics, writes to ``Data.act_dot``
    act_gain: custom actuator gains, writes to ``Data.actuator_force``
    act_bias: custom actuator biases, writes to ``Data.actuator_force``
    sensor: custom sensors, writes to ``Data.sensordata``
    contactfilter: custom contact filtering, writes to ``Data.contact``
  """

  passive: Callable | None = None
  control: Callable | None = None
  act_dyn: Callable | None = None
  act_gain: Callable | None = None
  act_bias: Callable | None = None
  sensor: Callable | None = None
  contactfilter: Callable | None = None


@dataclasses.dataclass
class Model:
  """Model definition and parameters.

  Attributes:
    nq: number of generalized coordinates
    nv: number of degrees of freedom
    nu: number of controls/inputs
    nactuator: number of actuators
    na: number of activation states
    nbody: number of bodies
    noct: number of total octree cells in all meshes
    njnt: number of joints
    ntree: number of kinematic trees
    nM: number of non-zeros in sparse inertia matrix
    nC: number of non-zeros in sparse body-dof matrix
    nD: number of non-zeros in sparse derivative matrix
    ngeom: number of geoms
    nsite: number of sites
    ncam: number of cameras
    nlight: number of lights
    nflex: number of flexes
    nflexnode: number of nodes in all flexes
    nflexvert: number of vertices in all flexes
    nflexedge: number of edges in all flexes
    nflexelem: number of elements in all flexes
    nflexelemdata: number of element vertex ids in all flexes
    nflexstiffness: number of stiffness parameters in all flexes
    nflexbending: number of bending parameters in all flexes
    nefm0dof: number of zero-mass flex DOFs
    nefm0L: number of non-zeros in zero-mass flex Cholesky factor
    nflexelemedge: number of element edge ids in all flexes
    nflexshelldata: number of shell fragment vertex ids in all flexes
    nJfe: number of non-zeros in sparse flexedge Jacobian
    nmesh: number of meshes
    nmeshvert: number of vertices for all meshes
    nmeshnormal: number of normals in all meshes
    nmeshface: number of faces for all meshes
    nmeshgraph: number of ints in mesh auxiliary data
    nmeshpoly: number of polygons in all meshes
    nmeshpolyvert: number of vertices in all polygons
    nmeshpolymap: number of polygons in vertex map
    nhfield: number of heightfields
    nhfielddata: size of elevation data
    nmat: number of materials
    npair: number of predefined geom pairs
    nexclude: number of excluded geom pairs
    neq: number of equality constraints
    ntendon: number of tendons
    nJten: number of non-zeros in sparse tendon Jacobian
    nwrap: number of wrap objects in all tendon paths
    nsensor: number of sensors
    nkey: number of keyframes
    nmocap: number of mocap bodies
    nplugin: number of plugin instances
    nplastic: number of plastic reference elements
    nJmom: number of non-zeros in actuator_moment
    npolygonmax: maximum number of verts per polygon
    nmeshdegmax: maximum number of polygons per vert
    nuserdata: number of custom user parameters
    nsensordata: number of elements in sensor data vector
    nhistory: number of history buffer entries
    opt: physics options
    qpos0: qpos values at default pose                       (*, nq)
    qpos_spring: reference pose for springs                  (*, nq)
    body_parentid: id of body's parent                       (nbody,)
    body_rootid: id of root above body                       (nbody,)
    body_weldid: id of body that this body is welded to      (nbody,)
    body_mocapid: id of mocap data; -1: none                 (nbody,)
    body_jntnum: number of joints for this body              (nbody,)
    body_jntadr: start addr of joints; -1: no joints         (nbody,)
    body_dofnum: number of motion degrees of freedom         (nbody,)
    body_dofadr: start addr of dofs; -1: no dofs             (nbody,)
    body_treeid: id of body's tree; -1: static               (nbody,)
    body_geomnum: number of geoms                            (nbody,)
    body_geomadr: start addr of geoms; -1: no geoms          (nbody,)
    body_simple: body simple type                            (nbody,)
    body_pos: position offset rel. to parent body            (*, nbody, 3)
    body_quat: orientation offset rel. to parent body        (*, nbody, 4)
    body_ipos: local position of center of mass              (*, nbody, 3)
    body_iquat: local orientation of inertia ellipsoid       (*, nbody, 4)
    body_mass: mass                                          (*, nbody,)
    body_subtreemass: mass of subtree starting at this body  (*, nbody,)
    body_inertia: diagonal inertia in ipos/iquat frame       (*, nbody, 3)
    body_invweight0: mean inv inert in qpos0 (trn, rot)      (*, nbody, 2)
    body_gravcomp: antigravity force, units of body weight   (*, nbody)
    body_contype: OR over all geom contypes                  (nbody,)
    body_conaffinity: OR over all geom conaffinities         (nbody,)
    oct_child: octree children                               (noct, 8)
    oct_aabb: octree axis-aligned bounding boxes             (noct, 2, 3)
    oct_coeff: octree interpolation coefficients             (noct, 8)
    jnt_type: type of joint (JointType)                      (njnt,)
    jnt_qposadr: start addr in 'qpos' for joint's data       (njnt,)
    jnt_dofadr: start addr in 'qvel' for joint's data        (njnt,)
    jnt_bodyid: id of joint's body                           (njnt,)
    jnt_actuatorid: actuator contributing damping/armature   (njnt,)
    jnt_plasticadr: start addr in plastic state (-1: none)   (njnt,)
    jnt_limited: does joint have limits                      (njnt,)
    jnt_actfrclimited: does joint have actuator force limits (njnt,)
    jnt_actgravcomp: is gravcomp force applied via actuators (njnt,)
    jnt_solref: constraint solver reference: limit           (*, njnt, mjNREF)
    jnt_solimp: constraint solver impedance: limit           (*, njnt, mjNIMP)
    jnt_pos: local anchor position                           (*, njnt, 3)
    jnt_axis: local joint axis                               (*, njnt, 3)
    jnt_stiffness: stiffness coefficient                     (*, njnt)
    jnt_stiffnesspoly: high-order stiffness coefficients     (*, njnt, 2)
    jnt_springyield: plastic yield threshold (0: elastic)    (*, njnt)
    jnt_range: joint limits                                  (*, njnt, 2)
    jnt_actfrcrange: range of total actuator force           (*, njnt, 2)
    jnt_margin: min distance for limit detection             (*, njnt)
    dof_bodyid: id of dof's body                             (nv,)
    dof_jntid: id of dof's joint                             (nv,)
    dof_parentid: id of dof's parent; -1: none               (nv,)
    dof_treeid: id of dof's tree                             (nv,)
    dof_Madr: dof address in M-diagonal                      (nv,)
    dof_solref: constraint solver reference: frictionloss    (*, nv, NREF)
    dof_solimp: constraint solver impedance: frictionloss    (*, nv, NIMP)
    dof_frictionloss: dof friction loss                      (*, nv)
    dof_armature: dof armature inertia/mass                  (*, nv)
    dof_damping: damping coefficient                         (*, nv)
    dof_dampingpoly: high-order damping coefficients         (*, nv, 2)
    dof_invweight0: diag. inverse inertia in qpos0           (*, nv)
    dof_length: dof length for weighting velocity norm       (nv,)
    tree_bodynum: number of bodies in tree (incl. root)      (ntree,)
    tree_dofadr: start address of tree's dofs                (ntree,)
    tree_dofnum: number of dofs in tree                      (ntree,)
    tree_sleep_policy: tree sleep policy (SleepPolicy)       (ntree,)
    geom_type: geometric type (GeomType)                     (ngeom,)
    geom_contype: geom contact type                          (ngeom,)
    geom_conaffinity: geom contact affinity                  (ngeom,)
    geom_condim: contact dimensionality (1, 3, 4, 6)         (ngeom,)
    geom_bodyid: id of geom's body                           (ngeom,)
    geom_dataid: id of geom's mesh/hfield; -1: none          (*, ngeom)
    geom_matid: material id for rendering                    (*, ngeom,)
    geom_group: geom group inclusion/exclusion mask          (ngeom,)
    geom_priority: geom contact priority                     (ngeom,)
    geom_solmix: mixing coef for solref/imp in geom pair     (*, ngeom,)
    geom_solref: constraint solver reference: contact        (*, ngeom, mjNREF)
    geom_solimp: constraint solver impedance: contact        (*, ngeom, mjNIMP)
    geom_size: geom-specific size parameters                 (*, ngeom, 3)
    geom_aabb: bounding box, (center, size)                  (*, ngeom, 2, 3)
    geom_rbound: radius of bounding sphere                   (*, ngeom,)
    geom_pos: local position offset rel. to body             (*, ngeom, 3)
    geom_quat: local orientation offset rel. to body         (*, ngeom, 4)
    geom_friction: friction for (slide, spin, roll)          (*, ngeom, 3)
    geom_margin: detect contact if dist<margin               (*, ngeom,)
    geom_gap: additional contact detection buffer            (*, ngeom,)
    geom_surfacevel: surface velocity in local frame: lin,ang(*, ngeom, 6)
    geom_adhesion: adhesive force of contacts                (*, ngeom,)
    geom_fluid: fluid interaction parameters                 (ngeom, mjNFLUID)
    geom_rgba: rgba when material is omitted                 (*, ngeom, 4)
    site_type: geom type for rendering (GeomType)            (nsite,)
    site_bodyid: id of site's body                           (nsite,)
    site_size: geom size for rendering                       (nsite, 3)
    site_pos: local position offset rel. to body             (*, nsite, 3)
    site_quat: local orientation offset rel. to body         (*, nsite, 4)
    cam_mode: camera tracking mode (CamLightType)            (ncam,)
    cam_bodyid: id of camera's body                          (ncam,)
    cam_targetbodyid: id of targeted body; -1: none          (ncam,)
    cam_pos: position rel. to body frame                     (*, ncam, 3)
    cam_quat: orientation rel. to body frame                 (*, ncam, 4)
    cam_poscom0: global position rel. to sub-com in qpos0    (*, ncam, 3)
    cam_pos0: global position rel. to body in qpos0          (*, ncam, 3)
    cam_mat0: global orientation in qpos0                    (*, ncam, 3, 3)
    cam_projection: projection type (ProjectionType)         (ncam,)
    cam_fovy: y field-of-view (ortho ? len : deg)            (*, ncam)
    cam_resolution: resolution: pixels [width, height]       (ncam, 2)
    cam_sensorsize: sensor size: length [width, height]      (ncam, 2)
    cam_intrinsic: [focal length; principal point]           (*, ncam, 4)
    light_mode: light tracking mode (CamLightType)           (nlight,)
    light_bodyid: id of light's body                         (nlight,)
    light_targetbodyid: id of targeted body; -1: none        (nlight,)
    light_type: spot, directional, etc. (mjtLightType)       (*, nlight)
    light_castshadow: does light cast shadows                (*, nlight)
    light_active: is light active                            (*, nlight)
    light_pos: position rel. to body frame                   (*, nlight, 3)
    light_dir: direction rel. to body frame                  (*, nlight, 3)
    light_poscom0: global position rel. to sub-com in qpos0  (*, nlight, 3)
    light_pos0: global position rel. to body in qpos0        (*, nlight, 3)
    light_dir0: global direction in qpos0                    (*, nlight, 3)
    light_attenuation: OpenGL constant/linear/quadratic      (*, nlight, 3)
    light_cutoff: spotlight half-cone angle in degrees       (*, nlight)
    light_exponent: spotlight angular falloff exponent       (*, nlight)
    light_ambient: ambient RGB                               (*, nlight, 3)
    light_diffuse: diffuse RGB                               (*, nlight, 3)
    light_specular: specular RGB                             (*, nlight, 3)
    flex_contype: flex contact type                          (nflex,)
    flex_conaffinity: flex contact affinity                  (nflex,)
    flex_condim: contact dimensionality (1, 3, 4, 6)         (nflex,)
    flex_priority: geom contact priority                     (nflex,)
    flex_solmix: mixing coef for solref/imp in geom pair     (nflex,)
    flex_solref: constraint solver reference: contact        (nflex, mjNREF)
    flex_solimp: constraint solver impedance: contact        (nflex, mjNIMP)
    flex_friction: friction for (slide, spin, roll)          (nflex, 3)
    flex_margin: detect contact if dist<margin               (nflex,)
    flex_gap: include in solver if dist<margin-gap           (nflex,)
    flex_selfcollide: self-collision mode                    (nflex,)
    flex_activelayers: active element layers                 (nflex,)
    flex_passive: passive contact mode                       (nflex,)
    flex_dim: 1: lines, 2: triangles, 3: tetrahedra          (nflex,)
    flex_interp: interpolation order (0: vertex, 1+: nodes)  (nflex,)
    flex_cellnum: cell count per dimension                   (nflex, 3)
    flex_nodeadr: first node address                         (nflex,)
    flex_nodenum: number of nodes                            (nflex,)
    flex_vertadr: first vertex address                       (nflex,)
    flex_vertnum: number of vertices                         (nflex,)
    flex_edgeadr: first edge address                         (nflex,)
    flex_edgenum: number of edges                            (nflex,)
    flex_elemadr: first element address                      (nflex,)
    flex_elemnum: number of elements                         (nflex,)
    flex_elemdataadr: first element vertex id address        (nflex,)
    flex_stiffnessadr: stiffness matrix address              (nflex,)
    flex_elemedgeadr: first element edge id address          (nflex,)
    flex_bendingadr: first bending data address              (nflex,)
    flex_plasticadr: start addr in plastic state (-1: none)  (nflex,)
    flex_plasticmode: plastic state bits (1: bend, 2: stretch) (nflex,)
    flex_shellnum: number of shells                          (nflex,)
    flex_shelldataadr: first shell data address              (nflex,)
    flex_nodebodyid: node body ids                           (nflexnode,)
    flex_vertbodyid: vertex body ids                         (nflexvert,)
    flex_vertcell: finite cell index for vertex; -1: none    (nflexvert,)
    flex_edge: edge vertex ids (2 per edge)                  (nflexedge, 2)
    flex_edgeflap: adjacent vertex ids (dim=2 only)          (nflexedge, 2)
    flex_elem: element vertex ids (dim+1 per elem)           (nflexelemdata,)
    flex_elemedge: element edge ids                          (nflexelemedge,)
    flex_elemlayer: element distance from surface            (nflexelem,)
    flex_shell: shell fragment vertex ids (dim per frag)     (nflexshelldata,)
    flex_vert: vertex local positions                        (nflexvert, 3)
    flex_vert0: reference vertex positions in qpos0          (nflexvert, 3)
    flex_node: node local positions                          (nflexnode, 3)
    flex_node0: reference node positions in qpos0            (nflexnode, 3)
    flexedge_length0: edge lengths in qpos0                  (nflexedge,)
    flexedge_invweight0: inv. inertia for the edge           (nflexedge,)
    flex_radius: radius around primitive element             (nflex,)
    flex_size: vertex bounding box half sizes in qpos0       (nflex, 3)
    flex_stiffness: finite element stiffness matrix          (nflexstiffness,)
    flex_bending: bending stiffness                          (nflexbending,)
    efm0_dofid: zero-mass flex DOF indices                   (nefm0dof,)
    efm0_L_rownnz: row non-zeros in zero-mass flex factor    (nefm0dof,)
    efm0_L_rowadr: row addresses in zero-mass flex factor    (nefm0dof,)
    efm0_L_colind: column indices in zero-mass flex factor   (nefm0L,)
    efm0_L: zero-mass flex Cholesky factor                   (nefm0L,)
    flex_damping: Rayleigh's damping coefficient             (nflex,)
    flex_yield: plastic yield strain (stretch, bend; 0: elastic) (nflex, 2)
    flex_edgestiffness: edge stiffness                       (nflex,)
    flex_edgedamping: edge damping                           (nflex,)
    flex_edgeequality: edge equality type (0:none,1:edge,3:strain) (nflex,)
    flex_rigid: flex is rigid (all vertices on same body)    (nflex,)
    flexedge_rigid: edge is rigid (e.g. vertices welded to same body) (nflexedge,)
    flex_centered: flex vertices are centered at body origin (nflex,)
    flexedge_J_rownnz: number of nonzeros in Jacobian row    (nflexedge,)
    flexedge_J_rowadr: row start address in colind array     (nflexedge,)
    flexedge_J_colind: column indices in sparse Jacobian     (nJfe,)
    mesh_vertadr: first vertex address                       (nmesh,)
    mesh_vertnum: number of vertices                         (nmesh,)
    mesh_faceadr: first face address                         (nmesh,)
    mesh_octadr: octree address for each mesh                (nmesh,)
    mesh_normaladr: first normal address                     (nmesh,)
    mesh_normalnum: number of normals                        (nmesh,)
    mesh_graphadr: graph data address; -1: no graph          (nmesh,)
    mesh_vert: vertex positions for all meshes               (nmeshvert, 3)
    mesh_normal: normals for all meshes                      (nmeshnormal, 3)
    mesh_face: face indices for all meshes                   (nface, 3)
    mesh_graph: convex graph data                            (nmeshgraph,)
    mesh_pos: translation applied to asset vertices          (nmesh, 3)
    mesh_quat: rotation applied to asset vertices            (nmesh, 4)
    mesh_polynum: number of polygons per mesh                (nmesh,)
    mesh_polyadr: first polygon address per mesh             (nmesh,)
    mesh_polynormal: all polygon normals                     (nmeshpoly, 3)
    mesh_polyvertadr: polygon vertex start address           (nmeshpoly,)
    mesh_polyvertnum: number of vertices per polygon         (nmeshpoly,)
    mesh_polyvert: all polygon vertices                      (nmeshpolyvert,)
    mesh_polymapadr: first polygon address per vertex        (nmeshvert,)
    mesh_polymapnum: number of polygons per vertex           (nmeshvert,)
    mesh_polymap: vertex to polygon map                      (nmeshpolymap,)
    hfield_size: (x, y, z_top, z_bottom)                     (nhfield, 4)
    hfield_nrow: number of rows in grid                      (nhfield,)
    hfield_ncol: number of columns in grid                   (nhfield,)
    hfield_adr: start address in hfield_data                 (nhfield,)
    hfield_data: elevation data                              (nhfielddata,)
    mat_texid: texture id for rendering                      (*, nmat, mjNTEXROLE)
    mat_texuniform: texture uniform flag (spatial scaling)   (*, nmat)
    mat_texrepeat: texture repeat for rendering              (*, nmat, 2)
    mat_emission: emission scalar (self-illumination)        (*, nmat)
    mat_specular: specular reflection scalar                 (*, nmat)
    mat_shininess: shininess in [0, 1], mapped to GL [0, 128](*, nmat)
    mat_rgba: rgba                                           (*, nmat, 4)
    pair_dim: contact dimensionality                         (npair,)
    pair_geom1: id of geom1                                  (npair,)
    pair_geom2: id of geom2                                  (npair,)
    pair_solref: solver reference: contact normal            (*, npair, mjNREF)
    pair_solreffriction: solver reference: contact friction  (*, npair, mjNREF)
    pair_solimp: solver impedance: contact                   (*, npair, mjNIMP)
    pair_margin: detect contact if dist<margin               (*, npair,)
    pair_gap: additional contact detection buffer            (*, npair,)
    pair_adhesion: adhesive force of contacts                (*, npair,)
    pair_friction: tangent1, 2, spin, roll1, 2               (*, npair, 5)
    exclude_signature: body1 << 16 + body2                   (nexclude,)
    eq_type: constraint type (EqType)                        (neq,)
    eq_obj1id: id of object 1                                (neq,)
    eq_obj2id: id of object 2                                (neq,)
    eq_objtype: type of both objects (ObjType)               (neq,)
    eq_active0: initial enable/disable constraint state      (neq,)
    eq_solref: constraint solver reference                   (*, neq, mjNREF)
    eq_solimp: constraint solver impedance                   (*, neq, mjNIMP)
    eq_data: numeric data for constraint                     (*, neq, mjNEQDATA)
    tendon_type: type of tendon (TendonType)                 (ntendon,)
    tendon_adr: address of first object in tendon's path     (ntendon,)
    tendon_num: number of objects in tendon's path           (ntendon,)
    tendon_actuatorid: actuator adding damping/armature      (ntendon,)
    tendon_plasticadr: start addr in plastic state (-1: none) (ntendon,)
    ten_J_rownnz: number of non-zeros in each tendon row     (ntendon,)
    ten_J_rowadr: row start address for sparse ten_J         (ntendon,)
    ten_J_colind: column indices in sparse ten_J             (nJten,)
    tendon_limited: does tendon have length limits           (ntendon,)
    tendon_actfrclimited: does ten have actuator force limit (ntendon,)
    tendon_solref_lim: constraint solver reference: limit    (*, ntendon, mjNREF)
    tendon_solimp_lim: constraint solver impedance: limit    (*, ntendon, mjNIMP)
    tendon_solref_fri: constraint solver reference: friction (*, ntendon, mjNREF)
    tendon_solimp_fri: constraint solver impedance: friction (*, ntendon, mjNIMP)
    tendon_range: tendon length limits                       (*, ntendon, 2)
    tendon_actfrcrange: range of total actuator force        (*, ntendon, 2)
    tendon_margin: min distance for limit detection          (*, ntendon)
    tendon_stiffness: stiffness coefficient                  (*, ntendon)
    tendon_stiffnesspoly: high-order stiffness coefficients  (*, ntendon, 2)
    tendon_damping: damping coefficient                      (*, ntendon)
    tendon_dampingpoly: high-order damping coefficients      (*, ntendon, 2)
    tendon_armature: inertia associated with tendon velocity (*, ntendon)
    tendon_frictionloss: loss due to friction                (*, ntendon)
    tendon_lengthspring: spring resting length range         (*, ntendon, 2)
    tendon_springyield: plastic yield threshold (0: elastic) (*, ntendon)
    tendon_length0: tendon length in qpos0                   (*, ntendon)
    tendon_invweight0: inv. weight in qpos0                  (*, ntendon)
    wrap_type: wrap object type (WrapType)                   (nwrap,)
    wrap_objid: object id: geom, site, joint                 (nwrap,)
    wrap_prm: divisor, joint coef, or site id                (nwrap,)
    actuator_trntype: transmission type (TrnType)            (nactuator,)
    actuator_dyntype: dynamics type (DynType)                (nactuator,)
    actuator_gaintype: gain type (GainType)                  (nactuator,)
    actuator_biastype: bias type (BiasType)                  (nactuator,)
    actuator_ctrladr: first control address; -1: none        (nactuator,)
    actuator_ctrlnum: number of control variables            (nactuator,)
    actuator_ctrlspec: input specification bitmask           (nactuator,)
    actuator_actadr: first activation address; -1: stateless (nactuator,)
    actuator_actnum: number of activation variables          (nactuator,)
    actuator_trnid: transmission id: joint, tendon, site     (nactuator, 2)
    actuator_cranklength: crank length for slider-crank      (*, nactuator)
    actuator_dynprm: dynamics parameters                     (*, nactuator, mjNDYN)
    actuator_gainprm: gain parameters                        (*, nactuator, mjNGAIN)
    actuator_biasprm: bias parameters                        (*, nactuator, mjNBIAS)
    actuator_actlimited: is activation limited               (nactuator,)
    actuator_actrange: range of activations                  (*, nactuator, 2)
    actuator_actearly: step activation before force          (nactuator,)
    actuator_history: history buffer sizes                   (*, nactuator, 2)
    actuator_historyadr: history buffer address              (*, nactuator)
    actuator_delay: delay in seconds                         (*, nactuator)
    actuator_damping: linear damping coefficient             (*, nactuator)
    actuator_dampingpoly: high-order damping coefficients    (*, nactuator, 2)
    actuator_armature: armature added to target              (*, nactuator)
    actuator_forcelimited: is force limited                  (nactuator,)
    actuator_forcerange: range of forces                     (*, nactuator, 2)
    actuator_ctrllimited: is control limited                 (nu,)
    actuator_ctrlrange: range of controls                    (*, nu, 2)
    actuator_gear: scale length and transmitted force        (*, nactuator, 6)
    actuator_acc0: acceleration from unit force in qpos0     (*, nactuator)
    actuator_lengthrange: feasible actuator length range     (*, nu, 2)
    sensor_type: sensor type (SensorType)                    (nsensor,)
    sensor_datatype: numeric data type (DataType)            (nsensor,)
    sensor_objtype: type of sensorized object (ObjType)      (nsensor,)
    sensor_objid: id of sensorized object                    (nsensor,)
    sensor_reftype: type of reference frame (ObjType)        (nsensor,)
    sensor_refid: id of reference frame; -1: global frame    (nsensor,)
    sensor_intprm: sensor parameters                         (nsensor, mjNSENS)
    sensor_dim: number of scalar outputs                     (nsensor,)
    sensor_adr: address in sensor array                      (nsensor,)
    sensor_cutoff: cutoff for real and positive; 0: ignore   (nsensor,)
    sensor_history: history buffer sizes                     (*, nsensor, 2)
    sensor_historyadr: history buffer address                (*, nsensor)
    sensor_delay: delay in seconds                           (*, nsensor)
    sensor_interval: sensor interval and phase               (*, nsensor, 2)
    plugin: globally registered plugin slot number           (nplugin,)
    plugin_attr: config attributes of geom plugin            (nplugin, _NPLUGINATTR)
    key_time: keyframe time                                  (nkey,)
    key_qpos: keyframe qpos                                  (nkey, nq)
    key_qvel: keyframe qvel                                  (nkey, nv)
    key_act: keyframe act                                    (nkey, na)
    key_mpos: keyframe mocap pos                             (nkey, nmocap, 3)
    key_mquat: keyframe mocap quat                           (nkey, nmocap, 4)
    key_ctrl: keyframe ctrl                                  (nkey, nu)
    key_plastic: keyframe plastic                            (nkey, nplastic)
    M_rownnz: number of non-zeros in each row of M           (nv,)
    M_rowadr: index of each row in M                         (nv,)
    M_colind: column indices of non-zeros in M               (nC,)
    mapM2M: index mapping from M (legacy) to M (CSR)         (nC)
    D_rownnz: non-zeros per row in D-structure               (nv,)
    D_rowadr: row start addresses in D-structure             (nv,)
    D_diag: diagonal element index within each row           (nv,)
    D_colind: column indices in D-structure                  (nD,)
    mapM2D: index mapping from M to D                        (nD,)
    mapD2M: index mapping from D to M                        (nC,)

  warp only fields:
    nefmK: number of non-zeros in effective flex stiffness matrix
    nefmdof: number of covered flex vertex triples
    efm_K_rownnz: row non-zeros in effective flex stiffness CSR (nv,)
    efm_K_rowadr: row addresses in effective flex stiffness CSR (nv,)
    efm_K_colind: column indices in effective flex stiffness CSR (nefmK,)
    efm_dofid: starting DOF index for each covered flex triple (nefmdof,)
    callback: custom physics callbacks
    nbranch: number of branches (leaf-to-root paths)
    nv_pad: number of degrees of freedom + padding
    nacttrnbody: number of actuators with body transmission
    nsensorcollision: number of unique collisions for
                      geom distance sensors
    nsensortaxel: number of taxels in all tactile sensors
    ntactileweld: number of unique weld bodies with tactile sensors
    nrangefinder: number of rangefinder rays
    nmaxcondim: maximum condim across geoms, pairs, and flexes
    nmaxpyramid: maximum number of pyramid directions
    nflexintcell: total interp cells (non-strain) for passive forces
    is_sparse: constraint Jacobian/Hessian layout (sparse vs dense). Does not affect M, whose
      factorization is a per-block decision -- see M_tiles and m_block_layout
    qLD_block_total: packed length of the dense region per world (also the offset of the LDL region)
    qLD_block_adr: packed factor offset; Q_LD_BLOCK_* sentinel otherwise (nv,)
    flg_adhesion: flag indicating if model has passive adhesion
    has_fluid: True if wind, density, or viscosity are non-zero at put_model time
    flg_surfacevel: whether model has non-zero surfacevel
    has_sdf_geom: whether the model contains SDF geoms
    has_flex_selfcollide: whether any flex has self-collision enabled
    has_flex_passive: whether any flex has passive contact enabled
    has_flex_snh: whether any flex uses 3D SNH elasticity
    has_non_simple_flex: whether any flex has non-simple vertex bodies
    has_flex_edge_stiff_damp: whether any non-rigid flex has edge stiffness or damping
    has_tendon_stiffness: whether any tendon has positive stiffness
    has_tendon_damping: whether any tendon has positive damping
    has_efm_actuator: whether any actuator contributes effective metric coupling
    has_actuator_wrap: whether any wrap-eligible rotational actuator is present
    has_actuator_dcmotor: whether any DC motor actuator is present
    efm0_active: whether zero-mass flex Cholesky preconditioner is active
    flex_interp_assemblable: whether interpolated flex stiffness can be assembled into efm_K
    has_unsupported_flex_interp: whether model has unsupported flex interpolation orders
    has_ellipsoid_geom: whether the model contains ellipsoid geoms
    has_plane_geom: whether the model contains plane geoms
    has_1d_flex: whether the model contains 1D flexes
    has_2d_flex: whether the model contains 2D flexes
    has_3d_flex: whether the model contains 3D flexes
    max_flex_dim: maximum flex dimension in the model
    block_dim: block dim options
    body_tree: list of body ids by tree level
    body_subtreenum: number of bodies in subtree, including self (nbody,)
    body_branches: flattened body ids for all branches
    body_branch_start: start index in body_branches for each branch   (nbranch + 1,)
    mocap_bodyid: id of body for mocap                       (nmocap,)
    body_fluid_ellipsoid: does body use ellipsoid fluid      (nbody,)
    body_is_free: is body root of a free rigid subtree       (nbody,)
    body_fluid_ellipsoid_adr: body ids with ellipsoid fluid  (nbody_fluid_ellipsoid,)
    body_fluid_box_adr: body ids with box fluid              (nbody_fluid_box,)
    body_freeadr: body ids of free bodies                    (nbodyfree,)
    jnt_limited_slide_hinge_adr: limited/slide/hinge jntadr
    jnt_limited_ball_adr: limited/ball jntadr
    body_isdofancestor: precomputed mask of which DOFs affect each body
    dof_tri_row: dof upper triangle row (used in solver)
    dof_tri_col: dof upper triangle col (used in solver)
    nxn_geom_pair: collision pair geom ids [-2, ngeom-1]
    nxn_geom_pair_filtered: valid collision pair geom ids
                            [-1, ngeom - 1]
    nxn_pairid: contact pair id, -1 if not predefined,
                  -2 if skipped
                collision id, else -1
    nxn_pairid_filtered: active subset of nxn_pairid
    geom_pair_type_count: count of max number of each
                          potential collision
    geom_plugin_index: geom index in plugin array            (ngeom,)
    eq_connect_adr: eq_* addresses of type `CONNECT`
    eq_wld_adr: eq_* addresses of type `WELD`
    eq_jnt_adr: eq_* addresses of type `JOINT`
    eq_ten_adr: eq_* addresses of type `TENDON`
    eq_flex_adr: eq * addresses of type `FLEX
    eq_flexstrain_adr: eq_* addresses of type `FLEXSTRAIN`
    tendon_jnt_adr: joint tendon address
    tendon_site_pair_adr: site pair tendon address
    tendon_geom_adr: geom tendon address
    tendon_limited_adr: addresses for limited tendons
    max_ten_J_rownnz: maximum number of non-zeros in a tendon row
    ten_wrapadr_site: wrap object starting address for sites
    ten_wrapnum_site: number of site wrap objects per tendon
    wrap_jnt_adr: addresses for joint tendon wrap object
    wrap_site_adr: addresses for site tendon wrap object
    wrap_site_pair_adr: first address for site wrap pair
    wrap_geom_adr: addresses for geom tendon wrap object
    wrap_pulley_scale: pulley scaling                        (nwrap,)
    actuator_trntype_body_adr: addresses for actuators
                               with body transmission
    sensor_pos_adr: addresses for position sensors
    sensor_limitpos_adr: address for limit position sensors
    sensor_vel_adr: addresses for velocity sensors
                    (excluding limit velocity sensors)
    sensor_limitvel_adr: address for limit velocity sensors
    sensor_acc_adr: addresses for acceleration sensors
                    (excluding touch, contact, limit force, and tendon actuator force sensors)
    sensor_stage_pos_adr: addresses for all stage-pos sensors
    sensor_stage_vel_adr: addresses for all stage-vel sensors
    sensor_stage_acc_adr: addresses for all stage-acc sensors
    sensor_rangefinder_adr: map rangefinder ray id to sensor id (nrangefinder,)
    rangefinder_sensor_adr: map sensor id to starting rangefinder ray id (nsensor,)
    sensor_collision_start_adr: address for sensor's first
                                item in collision
    collision_sensor_adr: map sensor id to collision id      (nsensor,)
    sensor_touch_adr: addresses for touch sensors
    sensor_limitfrc_adr: address for limit force sensors
    sensor_e_potential: evaluate energy_pos
    sensor_e_kinetic: evaluate energy_vel
    sensor_tendonactfrc_adr: address for tendonactfrc sensor
    sensor_subtree_vel: evaluate subtree_vel
    sensor_contact_pos_adr: addresses for position contact sensors (nsensor_contact_pos,)
    sensor_contact_vel_adr: addresses for velocity contact sensors (nsensor_contact_vel,)
    sensor_contact_acc_adr: addresses for acceleration contact sensors (nsensor_contact_acc,)
    sensor_rne_postconstraint: evaluate rne_postconstraint
    sensor_rangefinder_bodyid: bodyid for rangefinder        (nrangefinder,)
    weld_tactile_id: weld body to tactile weld index         (nbody,)
    taxel_vertadr: tactile sensor vertex address             (nsensortaxel,)
    taxel_sensorid: address for tactile sensors
    M_tiles: scalar and tiled block-factorization groups
    qLD_updates: sparse factor updates grouped by tree level
    qLD_all_updates: tuple of all levels concatenated
    qLD_level_offsets: tuple of start offsets for each level
    M_fullm_i: sparse mass matrix addressing
    M_fullm_j: sparse mass matrix addressing
    M_elemid: (row, col) -> CSR madr addresses; -1 if not a chain ancestor
    M_hinit_i: row index of each CSR M entry; for densifying M into the dense Newton H (nC,)
    qD_fullm_i: D-structure row indices for RNE derivatives
    qD_fullm_j: D-structure column indices for RNE derivatives
    M_mulm_rowadr: sparse matmul row pointers
    M_mulm_col: sparse matmul column indices
    M_mulm_madr: sparse matmul matrix addresses
    flex_elemflexid: maps each element index directly to its flexid         (nflexelem,)
    flex_edgeflexid: maps each edge index directly to its flexid            (nflexedge,)
    flex_shellflexid: maps each shell index directly to its flexid          (nflexshelldata,)
    flex_vertflexid: maps each vertex index directly to its flexid          (nflexvert,)
    flex_shelladr: maps each flex to its start shell index                  (nflex,)
    flex_faceadr: maps each flex to its start face index                    (nflex,)
    flex_simple: whether all vertices of flex are on simple bodies          (nflex,)
    flex_efcnum: number of edge equality rows (non-rigid edges)             (nflex,)
    flex_efcnnz: number of Jacobian nonzeros in edge equality rows          (nflex,)
    flexedge_efcadr: row of the edge among its flex's edge equality rows,   (nflexedge,)
                     -1 if the edge has no row
    flexedge_efcnnzadr: Jacobian nonzero offset of the edge among its       (nflexedge,)
                        flex's edge equality rows
    flex_cell_map: precomputed flex cell mapping (nflexintcell,)
    flexstrain_J_rownnz: number of nonzeros in flex strain Jacobian row     (neq_flexstrain,)
    flexstrain_J_rowadr: row start address in colind array (neq_flexstrain,)
    flexstrain_J_colind: column indices in sparse flex strain Jacobian      (nJfs,)
    neq_flexstrain: number of flex strain equality constraints
    nJfs: number of non-zeros in sparse flex strain Jacobian
    nflexbend_interp: number of interpolated bending edges
    flex_bend_interp_map: mapping of interpolated bending edges to flex and (nflexbend_interp, 2)
                          local edge indices
    nflexface: number of interpolated flex shell faces
    flex_face_map: mapping of face index to flex and local element face indices
    flex_face: global node indices of each face                             (nflexface, 9)
    nflexrodhinge: number of 1D rod bending hinges (single hinges or hinge pairs)
    flex_rodhinge_map: flex id and flex_bending address of each rod hinge   (nflexrodhinge, 2)
    nflexrodedge: number of 1D DER rod chain edges across all flexes
    efm_con_max_dof: maximum DOFs affected by a passive flex contact
    flex_rod_is_chain: whether flex is an open 1D single-hinge DER chain    (nflex,)
    flex_rod_has_twist: whether flex is a 1D DER rod with twist or box      (nflex,)
    flex_rod_is_box: whether flex is a 1D DER rod with box cross-section    (nflex,)
    flex_rod_root_clamped: whether first two vertices share a body          (nflex,)
    flex_rod_tail_clamped: whether last two vertices share a body           (nflex,)
    flex_eff_possible: whether flex contributes to effective metric         (nflex,)
    flex_rod_box_size: box cross-section half-sizes (a, b)                  (nflex, 2)
    flex_rod_hingeadr: start index in flex_rodhinge_* for each flex         (nflex,)
    flex_rod_edgeadr: start index in flex_rodedge_* for each flex           (nflex,)
    flex_rod_edgenum: number of DER chain edges for each flex               (nflex,)
    flex_rodedge_vert: local endpoint vertices of each DER chain edge       (nflexrodedge, 2)
    flex_rodedge_jnt: twist hinge joint id of each DER chain edge; -1 none  (nflexrodedge,)
    flex_elem_rodedge: (chain edge index, orientation sign) per flex elem   (nflexelem, 2)
    flex_vert_rodelem: incident local element index per flex vertex         (nflexvert,)
  """

  nq: int
  nv: int
  nu: int
  nactuator: int
  na: int
  nbody: int
  noct: int
  njnt: int
  ntree: int
  nM: int
  nC: int
  nD: int
  ngeom: int
  nsite: int
  ncam: int
  nlight: int
  nflex: int
  nflexnode: int
  nflexvert: int
  nflexedge: int
  nflexelem: int
  nflexelemdata: int
  nflexstiffness: int
  nflexbending: int

  nefm0dof: int
  nefm0L: int
  nflexelemedge: int
  nflexshelldata: int
  nJfe: int
  nmesh: int
  nmeshvert: int
  nmeshnormal: int
  nmeshface: int
  nmeshgraph: int
  nmeshpoly: int
  nmeshpolyvert: int
  nmeshpolymap: int
  nhfield: int
  nhfielddata: int
  nmat: int
  npair: int
  nexclude: int
  neq: int
  ntendon: int
  nJten: int
  nwrap: int
  nsensor: int
  nkey: int
  nmocap: int
  nplugin: int
  nplastic: int
  nJmom: int
  npolygonmax: int
  nmeshdegmax: int
  nuserdata: int
  nsensordata: int
  nhistory: int
  opt: Option
  qpos0: array("*", "nq", float)
  qpos_spring: array("*", "nq", float)
  body_parentid: array("nbody", int)
  body_rootid: array("nbody", int)
  body_weldid: array("nbody", int)
  body_mocapid: array("nbody", int)
  body_jntnum: array("nbody", int)
  body_jntadr: array("nbody", int)
  body_dofnum: array("nbody", int)
  body_dofadr: array("nbody", int)
  body_treeid: array("nbody", int)
  body_geomnum: array("nbody", int)
  body_geomadr: array("nbody", int)
  body_simple: array("nbody", int)
  body_pos: array("*", "nbody", wp.vec3)
  body_quat: array("*", "nbody", wp.quat)
  body_ipos: array("*", "nbody", wp.vec3)
  body_iquat: array("*", "nbody", wp.quat)
  body_mass: array("*", "nbody", float)
  body_subtreemass: array("*", "nbody", float)
  body_inertia: array("*", "nbody", wp.vec3)
  body_invweight0: array("*", "nbody", wp.vec2)
  body_gravcomp: array("*", "nbody", float)
  body_contype: array("nbody", int)
  body_conaffinity: array("nbody", int)
  oct_child: array("noct", vec8i)
  oct_aabb: array("noct", 2, wp.vec3)
  oct_coeff: array("noct", vec8)
  jnt_type: array("njnt", int)
  jnt_qposadr: array("njnt", int)
  jnt_dofadr: array("njnt", int)
  jnt_bodyid: array("njnt", int)
  jnt_actuatorid: array("njnt", int)
  jnt_plasticadr: array("njnt", int)
  jnt_limited: array("njnt", int)
  jnt_actfrclimited: array("njnt", bool)
  jnt_actgravcomp: array("njnt", int)
  jnt_solref: array("*", "njnt", wp.vec2)
  jnt_solimp: array("*", "njnt", vec5)
  jnt_pos: array("*", "njnt", wp.vec3)
  jnt_axis: array("*", "njnt", wp.vec3)
  jnt_stiffness: array("*", "njnt", float)
  jnt_stiffnesspoly: array("*", "njnt", wp.vec2)
  jnt_springyield: array("*", "njnt", float)
  jnt_range: array("*", "njnt", wp.vec2)
  jnt_actfrcrange: array("*", "njnt", wp.vec2)
  jnt_margin: array("*", "njnt", float)
  dof_bodyid: array("nv", int)
  dof_jntid: array("nv", int)
  dof_parentid: array("nv", int)
  dof_treeid: array("nv", int)
  dof_Madr: array("nv", int)
  dof_solref: array("*", "nv", wp.vec2)
  dof_solimp: array("*", "nv", vec5)
  dof_frictionloss: array("*", "nv", float)
  dof_armature: array("*", "nv", float)
  dof_damping: array("*", "nv", float)
  dof_dampingpoly: array("*", "nv", wp.vec2)
  dof_invweight0: array("*", "nv", float)
  dof_length: array("nv", float)
  tree_bodynum: array("ntree", int)
  tree_dofadr: array("ntree", int)
  tree_dofnum: array("ntree", int)
  tree_sleep_policy: array("ntree", int)
  geom_type: array("ngeom", int)
  geom_contype: array("ngeom", int)
  geom_conaffinity: array("ngeom", int)
  geom_condim: array("ngeom", int)
  geom_bodyid: array("ngeom", int)
  geom_dataid: array("*", "ngeom", int)
  geom_matid: array("*", "ngeom", int)
  geom_group: array("ngeom", int)
  geom_priority: array("ngeom", int)
  geom_solmix: array("*", "ngeom", float)
  geom_solref: array("*", "ngeom", wp.vec2)
  geom_solimp: array("*", "ngeom", vec5)
  geom_size: array("*", "ngeom", wp.vec3)
  geom_aabb: array("*", "ngeom", 2, wp.vec3)
  geom_rbound: array("*", "ngeom", float)
  geom_pos: array("*", "ngeom", wp.vec3)
  geom_quat: array("*", "ngeom", wp.quat)
  geom_friction: array("*", "ngeom", wp.vec3)
  geom_margin: array("*", "ngeom", float)
  geom_gap: array("*", "ngeom", float)
  geom_surfacevel: array("*", "ngeom", vec6)
  geom_adhesion: array("*", "ngeom", float)
  geom_fluid: array("ngeom", 12, float)
  geom_rgba: array("*", "ngeom", wp.vec4)
  site_type: array("nsite", int)
  site_bodyid: array("nsite", int)
  site_size: array("nsite", wp.vec3)
  site_pos: array("*", "nsite", wp.vec3)
  site_quat: array("*", "nsite", wp.quat)
  cam_mode: array("ncam", int)
  cam_bodyid: array("ncam", int)
  cam_targetbodyid: array("ncam", int)
  cam_pos: array("*", "ncam", wp.vec3)
  cam_quat: array("*", "ncam", wp.quat)
  cam_poscom0: array("*", "ncam", wp.vec3)
  cam_pos0: array("*", "ncam", wp.vec3)
  cam_mat0: array("*", "ncam", wp.mat33)
  cam_projection: array("ncam", int)
  cam_fovy: array("*", "ncam", float)
  cam_resolution: array("ncam", wp.vec2i)
  cam_sensorsize: array("ncam", wp.vec2)
  cam_intrinsic: array("*", "ncam", wp.vec4)
  light_mode: array("nlight", int)
  light_bodyid: array("nlight", int)
  light_targetbodyid: array("nlight", int)
  light_type: array("*", "nlight", int)
  light_castshadow: array("*", "nlight", bool)
  light_active: array("*", "nlight", bool)
  light_pos: array("*", "nlight", wp.vec3)
  light_dir: array("*", "nlight", wp.vec3)
  light_poscom0: array("*", "nlight", wp.vec3)
  light_pos0: array("*", "nlight", wp.vec3)
  light_dir0: array("*", "nlight", wp.vec3)
  light_attenuation: array("*", "nlight", wp.vec3)
  light_cutoff: array("*", "nlight", float)
  light_exponent: array("*", "nlight", float)
  light_ambient: array("*", "nlight", wp.vec3)
  light_diffuse: array("*", "nlight", wp.vec3)
  light_specular: array("*", "nlight", wp.vec3)
  flex_contype: array("nflex", int)
  flex_conaffinity: array("nflex", int)
  flex_condim: array("nflex", int)
  flex_priority: array("nflex", int)
  flex_solmix: array("nflex", float)
  flex_solref: array("nflex", wp.vec2)
  flex_solimp: array("nflex", vec5)
  flex_friction: array("nflex", wp.vec3)
  flex_margin: array("nflex", float)
  flex_gap: array("nflex", float)
  flex_selfcollide: array("nflex", int)
  flex_activelayers: array("nflex", int)
  flex_passive: array("nflex", int)
  flex_dim: array("nflex", int)
  flex_interp: array("nflex", int)
  flex_cellnum: array("nflex", wp.vec3i)
  flex_nodeadr: array("nflex", int)
  flex_nodenum: array("nflex", int)
  flex_vertadr: array("nflex", int)
  flex_vertnum: array("nflex", int)
  flex_edgeadr: array("nflex", int)
  flex_edgenum: array("nflex", int)
  flex_elemadr: array("nflex", int)
  flex_elemnum: array("nflex", int)
  flex_elemdataadr: array("nflex", int)
  flex_stiffnessadr: array("nflex", int)
  flex_elemedgeadr: array("nflex", int)
  flex_bendingadr: array("nflex", int)
  flex_plasticadr: array("nflex", int)
  flex_plasticmode: array("nflex", int)
  flex_shellnum: array("nflex", int)
  flex_shelldataadr: array("nflex", int)
  flex_nodebodyid: array("nflexnode", int)
  flex_vertbodyid: array("nflexvert", int)
  flex_vertcell: array("nflexvert", int)
  flex_edge: array("nflexedge", wp.vec2i)
  flex_edgeflap: array("nflexedge", wp.vec2i)
  flex_elem: array("nflexelemdata", int)
  flex_elemedge: array("nflexelemedge", int)
  flex_elemlayer: array("nflexelem", int)
  flex_shell: array("nflexshelldata", int)
  flex_vert: array("nflexvert", wp.vec3)
  flex_vert0: array("nflexvert", wp.vec3)
  flex_node: array("nflexnode", wp.vec3)
  flex_node0: array("nflexnode", wp.vec3)
  flexedge_length0: array("nflexedge", float)
  flexedge_invweight0: array("nflexedge", float)
  flex_radius: array("nflex", float)
  flex_size: array("nflex", wp.vec3)
  flex_stiffness: array("nflexstiffness", float)
  flex_bending: array("nflexbending", float)
  efm0_dofid: array("nefm0dof", int)
  efm0_L_rownnz: array("nefm0dof", int)
  efm0_L_rowadr: array("nefm0dof", int)
  efm0_L_colind: array("nefm0L", int)
  efm0_L: array("nefm0L", float)
  flex_damping: array("nflex", float)
  flex_yield: array("nflex", wp.vec2)
  flex_edgestiffness: array("nflex", float)
  flex_edgedamping: array("nflex", float)
  flex_edgeequality: array("nflex", int)
  flex_rigid: array("nflex", bool)
  flexedge_rigid: array("nflexedge", bool)
  flex_centered: array("nflex", bool)
  flexedge_J_rownnz: array("nflexedge", int)
  flexedge_J_rowadr: array("nflexedge", int)
  flexedge_J_colind: array("nJfe", int)
  mesh_vertadr: array("nmesh", int)
  mesh_vertnum: array("nmesh", int)
  mesh_faceadr: array("nmesh", int)
  mesh_octadr: array("nmesh", int)
  mesh_normaladr: array("nmesh", int)
  mesh_normalnum: array("nmesh", int)
  mesh_graphadr: array("nmesh", int)
  mesh_vert: array("nmeshvert", wp.vec3)
  mesh_normal: array("nmeshnormal", wp.vec3)
  mesh_face: array("nmeshface", wp.vec3i)
  mesh_graph: array("nmeshgraph", int)
  mesh_pos: array("nmesh", wp.vec3)
  mesh_quat: array("nmesh", wp.quat)
  mesh_polynum: array("nmesh", int)
  mesh_polyadr: array("nmesh", int)
  mesh_polynormal: array("nmeshpoly", wp.vec3)
  mesh_polyvertadr: array("nmeshpoly", int)
  mesh_polyvertnum: array("nmeshpoly", int)
  mesh_polyvert: array("nmeshpolyvert", int)
  mesh_polymapadr: array("nmeshvert", int)
  mesh_polymapnum: array("nmeshvert", int)
  mesh_polymap: array("nmeshpolymap", int)
  hfield_size: array("nhfield", wp.vec4)
  hfield_nrow: array("nhfield", int)
  hfield_ncol: array("nhfield", int)
  hfield_adr: array("nhfield", int)
  hfield_data: array("nhfielddata", float)
  mat_texid: array("*", "nmat", 10, int)
  mat_texuniform: array("*", "nmat", bool)
  mat_texrepeat: array("*", "nmat", wp.vec2)
  mat_emission: array("*", "nmat", float)
  mat_specular: array("*", "nmat", float)
  mat_shininess: array("*", "nmat", float)
  mat_rgba: array("*", "nmat", wp.vec4)
  pair_dim: array("npair", int)
  pair_geom1: array("npair", int)
  pair_geom2: array("npair", int)
  pair_solref: array("*", "npair", wp.vec2)
  pair_solreffriction: array("*", "npair", wp.vec2)
  pair_solimp: array("*", "npair", vec5)
  pair_margin: array("*", "npair", float)
  pair_gap: array("*", "npair", float)
  pair_adhesion: array("*", "npair", float)
  pair_friction: array("*", "npair", vec5)
  exclude_signature: array("nexclude", int)
  eq_type: array("neq", int)
  eq_obj1id: array("neq", int)
  eq_obj2id: array("neq", int)
  eq_objtype: array("neq", int)
  eq_active0: array("neq", bool)
  eq_solref: array("*", "neq", wp.vec2)
  eq_solimp: array("*", "neq", vec5)
  eq_data: array("*", "neq", vec12)
  tendon_type: array("ntendon", int)
  tendon_adr: array("ntendon", int)
  tendon_num: array("ntendon", int)
  tendon_actuatorid: array("ntendon", int)
  tendon_plasticadr: array("ntendon", int)
  ten_J_rownnz: array("ntendon", int)
  ten_J_rowadr: array("ntendon", int)
  ten_J_colind: array("nJten", int)
  tendon_limited: array("ntendon", int)
  tendon_actfrclimited: array("ntendon", bool)
  tendon_solref_lim: array("*", "ntendon", wp.vec2)
  tendon_solimp_lim: array("*", "ntendon", vec5)
  tendon_solref_fri: array("*", "ntendon", wp.vec2)
  tendon_solimp_fri: array("*", "ntendon", vec5)
  tendon_range: array("*", "ntendon", wp.vec2)
  tendon_actfrcrange: array("*", "ntendon", wp.vec2)
  tendon_margin: array("*", "ntendon", float)
  tendon_stiffness: array("*", "ntendon", float)
  tendon_stiffnesspoly: array("*", "ntendon", wp.vec2)
  tendon_damping: array("*", "ntendon", float)
  tendon_dampingpoly: array("*", "ntendon", wp.vec2)
  tendon_armature: array("*", "ntendon", float)
  tendon_frictionloss: array("*", "ntendon", float)
  tendon_lengthspring: array("*", "ntendon", wp.vec2)
  tendon_springyield: array("*", "ntendon", float)
  tendon_length0: array("*", "ntendon", float)
  tendon_invweight0: array("*", "ntendon", float)
  wrap_type: array("nwrap", int)
  wrap_objid: array("nwrap", int)
  wrap_prm: array("nwrap", float)
  actuator_trntype: array("nactuator", int)
  actuator_dyntype: array("nactuator", int)
  actuator_gaintype: array("nactuator", int)
  actuator_biastype: array("nactuator", int)
  actuator_ctrladr: array("nactuator", int)
  actuator_ctrlnum: array("nactuator", int)
  actuator_ctrlspec: array("nactuator", int)
  actuator_actadr: array("nactuator", int)
  actuator_actnum: array("nactuator", int)
  actuator_trnid: array("nactuator", wp.vec2i)
  actuator_cranklength: array("*", "nactuator", float)
  actuator_dynprm: array("*", "nactuator", vec10)
  actuator_gainprm: array("*", "nactuator", vec10)
  actuator_biasprm: array("*", "nactuator", vec10)
  actuator_actlimited: array("nactuator", bool)
  actuator_actrange: array("*", "nactuator", wp.vec2)
  actuator_actearly: array("nactuator", bool)
  actuator_history: array("*", "nactuator", wp.vec2i)
  actuator_historyadr: array("*", "nactuator", int)
  actuator_delay: array("*", "nactuator", float)
  actuator_damping: array("*", "nactuator", float)
  actuator_dampingpoly: array("*", "nactuator", wp.vec2)
  actuator_armature: array("*", "nactuator", float)
  actuator_forcelimited: array("nactuator", bool)
  actuator_forcerange: array("*", "nactuator", wp.vec2)
  actuator_ctrllimited: array("nu", bool)
  actuator_ctrlrange: array("*", "nu", wp.vec2)
  actuator_gear: array("*", "nactuator", wp.spatial_vector)
  actuator_acc0: array("*", "nactuator", float)
  actuator_lengthrange: array("*", "nactuator", wp.vec2)
  sensor_type: array("nsensor", int)
  sensor_datatype: array("nsensor", int)
  sensor_objtype: array("nsensor", int)
  sensor_objid: array("nsensor", int)
  sensor_reftype: array("nsensor", int)
  sensor_refid: array("nsensor", int)
  sensor_intprm: array("nsensor", 3, int)
  sensor_dim: array("nsensor", int)
  sensor_adr: array("nsensor", int)
  sensor_cutoff: array("nsensor", float)
  sensor_history: array("*", "nsensor", wp.vec2i)
  sensor_historyadr: array("*", "nsensor", int)
  sensor_delay: array("*", "nsensor", float)
  sensor_interval: array("*", "nsensor", wp.vec2)
  plugin: array("nplugin", int)
  plugin_attr: array("nplugin", vec_pluginattr)
  key_time: array("nkey", float)
  key_qpos: array("nkey", "nq", float)
  key_qvel: array("nkey", "nv", float)
  key_act: array("nkey", "na", float)
  key_mpos: array("nkey", "nmocap", wp.vec3)
  key_mquat: array("nkey", "nmocap", wp.quat)
  key_ctrl: array("nkey", "nu", float)
  key_plastic: array("nkey", "nplastic", float)
  M_rownnz: array("nv", int)
  M_rowadr: array("nv", int)
  M_colind: array("nC", int)
  mapM2M: array("nC", int)
  D_rownnz: array("nv", int)
  D_rowadr: array("nv", int)
  D_diag: array("nv", int)
  D_colind: array("nD", int)
  mapM2D: array("nD", int)
  mapD2M: array("nC", int)
  # warp only fields:
  nefmK: int
  nefmdof: int
  efm_K_rownnz: array("nv", int)
  efm_K_rowadr: array("nv", int)
  efm_K_colind: array("nefmK", int)
  efm_dofid: array("nefmdof", int)
  callback: Callback
  nbranch: int
  nv_pad: int
  nacttrnbody: int
  nsensorcollision: int
  nsensortaxel: int
  ntactileweld: int
  nrangefinder: int
  nmaxcondim: int
  nmaxpyramid: int
  nflexintcell: int
  is_sparse: bool
  qLD_block_total: int
  qLD_block_adr: array("nv", int)
  flg_adhesion: bool
  has_fluid: bool
  flg_surfacevel: bool
  has_sdf_geom: bool
  has_flex_selfcollide: bool
  has_flex_passive: bool
  has_flex_snh: bool
  has_non_simple_flex: bool
  has_flex_edge_stiff_damp: bool
  has_tendon_stiffness: bool
  has_tendon_damping: bool
  has_efm_actuator: bool
  has_actuator_wrap: bool
  has_actuator_dcmotor: bool
  efm0_active: bool
  flex_interp_assemblable: bool
  has_unsupported_flex_interp: bool
  has_ellipsoid_geom: bool
  has_plane_geom: bool
  has_1d_flex: bool
  has_2d_flex: bool
  has_3d_flex: bool
  max_flex_dim: int
  block_dim: BlockDim
  body_tree: tuple[array("nbody", int), ...]
  body_subtreenum: array("nbody", int)
  body_branches: array("nbody_branches", int)
  body_branch_start: array("nbranch_start", int)
  mocap_bodyid: array("nmocap", int)
  body_fluid_ellipsoid: array("nbody", bool)
  body_is_free: array("nbody", bool)
  body_fluid_ellipsoid_adr: array("nbody_fluid_ellipsoid", int)
  body_fluid_box_adr: array("nbody_fluid_box", int)
  body_freeadr: array("nbodyfree", int)
  jnt_limited_slide_hinge_adr: array("njnt_limited_slide_hinge", int)
  jnt_limited_ball_adr: array("njnt_limited_ball", int)
  body_isdofancestor: array("nbody", "nv_pad", int)
  dof_tri_row: array("ndof_tri", int)
  dof_tri_col: array("ndof_tri", int)
  nxn_geom_pair: array("nnxn_geom_pair", wp.vec2i)
  nxn_geom_pair_filtered: array("nnxn_geom_pair_filtered", wp.vec2i)
  nxn_pairid: array("nnxn_geom_pair", wp.vec2i)
  nxn_pairid_filtered: array("nnxn_geom_pair_filtered", wp.vec2i)
  geom_pair_type_count: tuple[int, ...]
  geom_plugin_index: array("ngeom", int)
  eq_connect_adr: array("neq_connect", int)
  eq_wld_adr: array("neq_wld", int)
  eq_jnt_adr: array("neq_jnt", int)
  eq_ten_adr: array("neq_ten", int)
  eq_flex_adr: array("neq_flex", int)
  eq_flexstrain_adr: array("neq_flexstrain", int)
  tendon_jnt_adr: array("ntendon_jnt", int)
  tendon_site_pair_adr: array("ntendon_site_pair", int)
  tendon_geom_adr: array("ntendon_geom", int)
  tendon_limited_adr: array("ntendon_limited", int)
  max_ten_J_rownnz: int
  ten_wrapadr_site: array("nten_wrapadr_site", int)
  ten_wrapnum_site: array("ntendon", int)
  wrap_jnt_adr: array("nwrap_jnt", int)
  wrap_site_adr: array("nwrap_site", int)
  wrap_site_pair_adr: array("nwrap_site_pair", int)
  wrap_geom_adr: array("nwrap_geom", int)
  wrap_pulley_scale: array("nwrap", float)
  actuator_trntype_body_adr: array("nacttrnbody", int)
  sensor_pos_adr: array("nsensor_pos", int)
  sensor_limitpos_adr: array("nsensor_limitpos", int)
  sensor_vel_adr: array("nsensor_vel", int)
  sensor_limitvel_adr: array("nsensor_limitvel", int)
  sensor_acc_adr: array("nsensor_acc", int)
  sensor_stage_pos_adr: array("nsensor_stage_pos", int)
  sensor_stage_vel_adr: array("nsensor_stage_vel", int)
  sensor_stage_acc_adr: array("nsensor_stage_acc", int)
  sensor_rangefinder_adr: array("nrangefinder", int)
  rangefinder_sensor_adr: array("nsensor", int)
  sensor_collision_start_adr: array("nsensor_collision_start_adr", int)
  collision_sensor_adr: array("nsensor", int)
  sensor_touch_adr: array("nsensor_touch", int)
  sensor_limitfrc_adr: array("nsensor_limitfrc", int)
  sensor_e_potential: bool
  sensor_e_kinetic: bool
  sensor_tendonactfrc_adr: array("nsensor_tendonactfrc", int)
  sensor_subtree_vel: bool
  sensor_contact_pos_adr: array("nsensor_contact_pos", int)
  sensor_contact_vel_adr: array("nsensor_contact_vel", int)
  sensor_contact_acc_adr: array("nsensor_contact_acc", int)
  sensor_rne_postconstraint: bool
  sensor_rangefinder_bodyid: array("nrangefinder", int)
  weld_tactile_id: array("nbody", int)
  taxel_vertadr: array("nsensortaxel", int)
  taxel_sensorid: array("nsensortaxel", int)
  M_tiles: tuple[TileSet, ...]
  qLD_updates: tuple[array("nqLD_all_updates", wp.vec3i), ...]
  qLD_all_updates: array("nqLD_all_updates", wp.vec3i)
  qLD_level_offsets: array("nqLD_level_offsets", int)
  # TODO(team): Remove M_fullm_i/j and M_elemid by iterating the M CSR layout
  # directly in the solver/derivative kernels
  M_fullm_i: array("nM_fullm", int)
  M_fullm_j: array("nM_fullm", int)
  M_elemid: array("nv", "nv", int)  # (row, col) -> CSR madr address; -1 if col is not a chain ancestor of row
  M_hinit_i: array("nC", int)  # row index of each CSR M entry (for densifying M into the dense Newton H)
  qD_fullm_i: array("nqD_fullm", int)  # D-structure (full square) row indices for RNE derivatives
  qD_fullm_j: array("nqD_fullm", int)  # D-structure (full square) column indices for RNE derivatives
  # Gather-based sparse mul_m indices (thread per DOF, no atomics)
  M_mulm_rowadr: array("nv_plus_1", int)  # start address for each row [nv+1]
  M_mulm_col: array("nM_mulm", int)  # column index to gather from
  M_mulm_madr: array("nM_mulm", int)  # matrix address to read
  flex_elemflexid: array("nflexelem", int)
  flex_edgeflexid: array("nflexedge", int)
  flex_shellflexid: array("nflexshelldata", int)
  flex_vertflexid: array("nflexvert", int)
  flex_shelladr: array("nflex", int)
  flex_faceadr: array("nflex", int)
  flex_simple: array("nflex", bool)
  flex_efcnum: array("nflex", int)
  flex_efcnnz: array("nflex", int)
  flexedge_efcadr: array("nflexedge", int)
  flexedge_efcnnzadr: array("nflexedge", int)
  flex_cell_map: array("nflexintcell", wp.vec4i)
  flexstrain_J_rownnz: array("neq_flexstrain", int)
  flexstrain_J_rowadr: array("neq_flexstrain", int)
  flexstrain_J_colind: array("nJfs", int)
  neq_flexstrain: int
  nJfs: int
  nflexbend_interp: int
  flex_bend_interp_map: array("nflexbend_interp", wp.vec2i)
  nflexface: int
  flex_face_map: array("nflexface", wp.vec2i)
  flex_face: array("nflexface", 9, int)
  nflexrodhinge: int
  flex_rodhinge_map: array("nflexrodhinge", wp.vec2i)
  nflexrodedge: int
  efm_con_max_dof: int
  flex_rod_is_chain: array("nflex", bool)
  flex_rod_has_twist: array("nflex", bool)
  flex_rod_is_box: array("nflex", bool)
  flex_rod_root_clamped: array("nflex", bool)
  flex_rod_tail_clamped: array("nflex", bool)
  flex_eff_possible: array("nflex", bool)
  flex_rod_box_size: array("nflex", wp.vec2)
  flex_rod_hingeadr: array("nflex", int)
  flex_rod_edgeadr: array("nflex", int)
  flex_rod_edgenum: array("nflex", int)
  flex_rodedge_vert: array("nflexrodedge", wp.vec2i)
  flex_rodedge_jnt: array("nflexrodedge", int)
  flex_elem_rodedge: array("nflexelem", wp.vec2i)
  flex_vert_rodelem: array("nflexvert", int)


class ContactType(enum.IntFlag):
  """Type of contact.

  Attributes:
    CONSTRAINT: contact for constraint solver
    SENSOR: contact for collision sensor (GEOMDIST, GEOMNORMAL, GEOMFROMTO)
    PASSIVE: passive contact for effective metric
  """

  CONSTRAINT = 1 << 0
  SENSOR = 1 << 1
  PASSIVE = 1 << 2


@dataclasses.dataclass
class Contact:
  """Contact data.

  Attributes:
    dist: distance between nearest points; neg: penetration          (naconmax,)
    pos: position of contact point: midpoint between geoms           (naconmax, 3)
    frame: normal is in [0-2], points from geom[0] to geom[1]        (naconmax, 3, 3)
    includemargin: include if dist<includemargin=margin              (naconmax,)
    friction: tangent1, 2, spin, roll1, 2                            (naconmax, 5)
    solref: constraint solver reference, normal direction            (naconmax, 2)
    solreffriction: constraint solver reference, friction directions (naconmax, 2)
    solimp: constraint solver impedance                              (naconmax, 5)
    dim: contact space dimensionality: 1, 3, 4 or 6                  (naconmax,)
    geom: geom ids; -1 for flex                                      (naconmax, 2)
    flex: flex ids; -1 for geom                                      (naconmax, 2)
    elem: element ids; -1 for geom or flex vertex                    (naconmax, 2)
    vert: vertex ids for flex/mesh contact                           (naconmax, 2)
    efc_address: address in efc; -1: not included                    (naconmax, nmaxpyramid)
    worldid: world id                                                (naconmax,)
    type: ContactType                                                (naconmax,)
    geomcollisionid: i-th contact generated for geom                 (naconmax,)
                     helps uniquely identity contact when multiple
                     contacts are generated for geom pair
    adhesion: adhesive force of contact                              (naconmax,)
    surfacevel: surface velocity in contact frame: tangent1, 2, spin (naconmax, 3)
  """

  dist: array("naconmax", float)
  pos: array("naconmax", wp.vec3)
  frame: array("naconmax", wp.mat33)
  includemargin: array("naconmax", float)
  friction: array("naconmax", vec5)
  solref: array("naconmax", wp.vec2)
  solreffriction: array("naconmax", wp.vec2)
  solimp: array("naconmax", vec5)
  dim: array("naconmax", int)
  geom: array("naconmax", wp.vec2i)
  flex: array("naconmax", wp.vec2i)
  elem: array("naconmax", wp.vec2i)
  vert: array("naconmax", wp.vec2i)
  efc_address: array("naconmax", "nmaxpyramid", int)
  worldid: array("naconmax", int)
  type: array("naconmax", int)
  geomcollisionid: array("naconmax", int)
  adhesion: array("naconmax", float)
  surfacevel: array("naconmax", wp.vec3)


@dataclasses.dataclass
class Constraint:
  """Constraint data.

  Attributes:
    type: constraint type (ConstraintType)            (nworld, njmax)
    id: id of object of specific type                 (nworld, njmax)
    jtdaj_adr: first efc row of each JTDAJ block      (nworld, njmax)
    jtdaj_nrow: efc rows per JTDAJ block              (nworld, njmax)
    jtdaj_nblock: number of JTDAJ blocks              (nworld,)
    J_rownnz: number of non-zeros in J row            (nworld, 0) dense
                                                      (nworld, njmax) sparse
    J_rowadr: row start address in colind array       (nworld, 0) dense
                                                      (nworld, njmax) sparse
    J_colind: column indices in J                     (nworld, 0, 0) dense
                                                      (nworld, 1, njmax_nnz) sparse
    J: constraint Jacobian                            (nworld, njmax_pad, nv_pad) dense
                                                      (nworld, 1, njmax_nnz) sparse
    pos: constraint position (equality, contact)      (nworld, njmax)
    margin: inclusion margin (contact)                (nworld, njmax)
    D: constraint mass                                (nworld, njmax_pad)
    vel: velocity in constraint space: J*qvel         (nworld, njmax)
    aref: reference pseudo-acceleration               (nworld, njmax)
    frictionloss: frictionloss (friction)             (nworld, njmax)
    force: constraint force in constraint space       (nworld, njmax)
    state: constraint state                           (nworld, njmax_pad)
    island: island ID per constraint                  (nworld, njmax)
  warp only fields:
    Ma: M*qacc                                        (nworld, nv)
    Jqvel: J*qvel                                     (nworld, njmax)
  """

  type: array("nworld", "njmax", int)
  id: array("nworld", "njmax", int)
  jtdaj_adr: array("nworld", "njmax", int)
  jtdaj_nrow: array("nworld", "njmax", int)
  jtdaj_nblock: array("nworld", int)
  J_rownnz: array("nworld", "njmax", int)
  J_rowadr: array("nworld", "njmax", int)
  J_colind: array("nworld", 1, "njmax_nnz", int)
  J: array("nworld", 1, "njmax_nnz", float)
  pos: array("nworld", "njmax", float)
  margin: array("nworld", "njmax", float)
  D: array("nworld", "njmax_pad", float)
  vel: array("nworld", "njmax", float)
  aref: array("nworld", "njmax", float)
  frictionloss: array("nworld", "njmax", float)
  force: array("nworld", "njmax", float)
  state: array("nworld", "njmax_pad", int)
  island: array("nworld", "njmax", int)
  Ma: array("nworld", "nv", float)
  Jqvel: array("nworld", "njmax", float)


@dataclasses.dataclass
class Data:
  """Dynamic state that updates each step.

  Attributes:
    solver_niter: number of solver iterations                   (nworld,)
    ne: number of equality constraints                          (nworld,)
    nf: number of friction constraints                          (nworld,)
    nl: number of limit constraints                             (nworld,)
    nefc: number of constraints                                 (nworld,)
    nisland: number of constraint islands                       (nworld,)
    nidof: total DOFs in islands                                (nworld,)
    ntree_awake: number of awake trees                          (nworld,)
    nbody_awake: number of awake bodies                         (nworld,)
    nv_awake: number of awake dofs                              (nworld,)
    time: simulation time                                       (nworld,)
    energy: potential, kinetic energy                           (nworld, 2)
    qpos: position                                              (nworld, nq)
    qvel: velocity                                              (nworld, nv)
    act: actuator activation                                    (nworld, na)
    history: history buffer for delays                          (nworld, nhistory)
    plastic: plastic spring references                          (nworld, nplastic)
    qacc_warmstart: acceleration used for warmstart             (nworld, nv)
    ctrl: control                                               (nworld, nu)
    qfrc_applied: applied generalized force                     (nworld, nv)
    xfrc_applied: applied Cartesian force/torque                (nworld, nbody, 6)
    eq_active: enable/disable constraints                       (nworld, neq)
    mocap_pos: position of mocap bodies                         (nworld, nmocap, 3)
    mocap_quat: orientation of mocap bodies                     (nworld, nmocap, 4)
    qacc: acceleration                                          (nworld, nv)
    act_dot: time-derivative of actuator activation             (nworld, na)
    userdata: custom user data                                  (nworld, nuserdata)
    sensordata: sensor data array                               (nworld, nsensordata,)
    tree_asleep: tree asleep counter; >=0: asleep cycle         (nworld, ntree)
    xpos: Cartesian position of body frame                      (nworld, nbody, 3)
    xquat: Cartesian orientation of body frame                  (nworld, nbody, 4)
    xmat: Cartesian orientation of body frame                   (nworld, nbody, 3, 3)
    xipos: Cartesian position of body com                       (nworld, nbody, 3)
    ximat: Cartesian orientation of body inertia                (nworld, nbody, 3, 3)
    xanchor: Cartesian position of joint anchor                 (nworld, njnt, 3)
    xaxis: Cartesian joint axis                                 (nworld, njnt, 3)
    geom_xpos: Cartesian geom position                          (nworld, ngeom, 3)
    geom_xmat: Cartesian geom orientation                       (nworld, ngeom, 3, 3)
    site_xpos: Cartesian site position                          (nworld, nsite, 3)
    site_xmat: Cartesian site orientation                       (nworld, nsite, 3, 3)
    cam_xpos: Cartesian camera position                         (nworld, ncam, 3)
    cam_xmat: Cartesian camera orientation                      (nworld, ncam, 3, 3)
    light_xpos: Cartesian light position                        (nworld, nlight, 3)
    light_xdir: Cartesian light direction                       (nworld, nlight, 3)
    subtree_com: center of mass of each subtree                 (nworld, nbody, 3)
    cdof: com-based motion axis of each dof (rot:lin)           (nworld, nv, 6)
    cinert: com-based body inertia and mass                     (nworld, nbody, 10)
    flexvert_xpos: cartesian flex vertex positions              (nworld, nflexvert, 3)
    flex_hessian_valid: whether flex stretch Hessian is current (nworld, nflex)
    flexvert_hessian: diagonal 3x3 blocks of stretch Hessian    (nworld, nflexvert, 6)
    flexedge_hessian: off-diagonal 3x3 blocks of stretch Hessian (nworld, nflexedge, 3, 3)
    flexedge_J: edge length Jacobian                            (nworld, nJfe)
    flexedge_length: flex edge lengths                          (nworld, nflexedge)
    ten_wrapadr: start address of tendon's path                 (nworld, ntendon)
    ten_wrapnum: number of wrap points in path                  (nworld, ntendon)
    ten_J: tendon Jacobian                                      (nworld, nJten)
    ten_length: tendon lengths                                  (nworld, ntendon)
    wrap_obj: geomid; -1: site; -2: pulley                      (nworld, nwrap, 2)
    wrap_xpos: Cartesian 3D points in all paths                 (nworld, nwrap, 6)
    actuator_length: actuator lengths                           (nworld, nu)
    moment_rownnz: number of non-zeros in actuator_moment row   (nworld, nu)
    moment_rowadr: row start address in actuator_moment         (nworld, nu)
    moment_colind: column indices in sparse actuator_moment     (nworld, nJmom)
    actuator_moment: actuator moments                           (nworld, nJmom)
    crb: com-based composite inertia and mass                   (nworld, nbody, 10)
    M: total inertia, CSR                                       (nworld, nC)
    qLD: per-block factor: packed dense region, then the nC     (nworld, qLD_block_total + nC)
         L'*D*L region at offset qLD_block_total (nC=0 if no sparse block)
    qLDiagInv: reciprocal diagonal for compact and sparse blocks (nworld, nv)
    tree_awake: is tree awake; 0: asleep; 1: awake              (nworld, ntree)
    body_awake: body sleep state (SleepState)                   (nworld, nbody)
    body_awake_ind: indices of awake/static bodies              (nworld, nbody)
    dof_awake_ind: indices of awake dofs                        (nworld, nv)
    flexedge_velocity: flex edge velocities                     (nworld, nflexedge)
    ten_velocity: tendon velocities                             (nworld, ntendon)
    actuator_velocity: actuator velocities                      (nworld, nu)
    cvel: com-based velocity (rot:lin)                          (nworld, nbody, 6)
    cdof_dot: time-derivative of cdof (rot:lin)                 (nworld, nv, 6)
    qfrc_bias: C(qpos,qvel)                                     (nworld, nv)
    qfrc_spring: passive spring force                           (nworld, nv)
    qfrc_damper: passive damper force                           (nworld, nv)
    qfrc_gravcomp: passive gravity compensation force           (nworld, nv)
    qfrc_fluid: passive fluid force                             (nworld, nv)
    qfrc_adhesion: passive adhesion force                       (nworld, nv)
    qfrc_passive: total passive force                           (nworld, nv)
    subtree_linvel: linear velocity of subtree com              (nworld, nbody, 3)
    subtree_angmom: angular momentum about subtree com          (nworld, nbody, 3)
    qH: modified mass matrix M + metric diagonals               (nworld, nC)
    qHDiagInv: reciprocal diagonal of qH                        (nworld, nv)
    qLU: sparse LU factorization of (M - dt*qDeriv)             (nworld, nD)
    actuator_force: actuator force in actuation space           (nworld, nu)
    qfrc_actuator: actuator force                               (nworld, nv)
    qfrc_smooth: net unconstrained force                        (nworld, nv)
    qacc_smooth: unconstrained acceleration                     (nworld, nv)
    qfrc_constraint: constraint force                           (nworld, nv)
    qfrc_inverse: net external force; should equal:             (nworld, nv)
                  qfrc_applied + J.T @ xfrc_applied
                  + qfrc_actuator
    cacc: com-based acceleration                                (nworld, nbody, 6)
    cfrc_int: com-based interaction force with parent           (nworld, nbody, 6)
    cfrc_ext: com-based external force on body                  (nworld, nbody, 6)
    efm_c: smooth-force shift h*K*qvel                          (nworld, nv)
    efm_diag: effective-metric diagonal h*D + h^2*K             (nworld, nv)
    efm_fluid: fluid drag blocks in M's sparsity pattern        (nworld, nC)
    efm_ca: actuation-stage smooth-force shift                  (nworld, nv)
    efm_K_val: effective flex stiffness CSR values              (nworld, nefmK)
    qHLD: factor of modified mass matrix qH                     (nworld, qld_total)
    efm_ts: tendon metric scale h^2*k + h*b                     (nworld, ntendon)
    efm_as: actuator metric scale h^2*gp + h*gv                 (nworld, nactuator)
    contact: contact data
    efc: constraint data
    dof_frictionadr: friction efc address (-1: none) if nf > 0  (nworld, nv)
    tree_island: island ID per tree (-1 if unconstrained)       (nworld, ntree)
    dof_island: island ID per DOF (-1 if unconstrained)         (nworld, nv)
    island_dofadr: island start address in dof vector           (nworld, ntree)
    island_idofadr: island start address in idof vector         (nworld, ntree)
    island_nv: DOFs per island                                  (nworld, ntree)
    island_nefc: constraints per island                         (nworld, ntree)
    island_ne: equality constraints per island                  (nworld, ntree)
    island_nf: friction constraints per island                  (nworld, ntree)
    island_iefcadr: island start address in efc vector          (nworld, ntree)
    map_dof2idof: global DOF -> island-local DOF                (nworld, nv)
    map_idof2dof: island-local DOF -> global DOF                (nworld, nv)
    map_efc2iefc: global EFC -> island-local EFC                (nworld, njmax)
    map_iefc2efc: island-local EFC -> global EFC                (nworld, njmax)
    dof_islandid: island ID per island-DOF                      (nworld, nv)
    efc_islandid: island ID per island-EFC                      (nworld, njmax)
    ncdof: number of active (compacted) DOFs per world          (nworld,)
    dof_cdof: global DOF -> compacted DOF; -1 if inactive       (nworld, nv)
    cdof_dof: compacted DOF -> global DOF; -1 if unused         (nworld, nvmax_pad)
    cdof_tri_row: row index of compacted Hessian dof-pairs      (nvmax_pad_sq,)
    cdof_tri_col: col index of compacted Hessian dof-pairs      (nvmax_pad_sq,)
    cM: compacted dense inertia                                 (nworld, nvmax_pad, nvmax_pad)
    cqLD: compacted upper Cholesky factor                       (nworld, nvmax_pad, nvmax_pad)
    crhs: compacted smooth-solve right-hand side                (nworld, nvmax_pad, 1)
    cx: compacted smooth-solve solution                         (nworld, nvmax_pad, 1)
    cJ: compacted dense constraint Jacobian                     (nworld, njmax_pad, nvmax_pad)
    cMa: compacted M @ qacc workspace                           (nworld, nvmax_pad)
    cqfrc_smooth: compacted net unconstrained force             (nworld, nvmax_pad)
    cqacc_smooth: compacted unconstrained acceleration          (nworld, nvmax_pad)
    cqacc_warmstart: compacted warmstart acceleration           (nworld, nvmax_pad)
    cqacc: compacted acceleration (solve output)                (nworld, nvmax_pad)
    cqfrc_constraint: compacted constraint force                (nworld, nvmax_pad)

  warp only fields:
    nworld: number of worlds
    naconmax: maximum number of contacts (shared across all worlds)
    naccdmax: maximum number of contacts for CCD (all worlds)
    njmax: maximum number of constraints per world
    nvmax: capacity for compacted active DOFs per world
    nvmax_pad: nvmax rounded up to the nearest multiple of TILE_SIZE_JTDAJ_DENSE
    njmax_pad: njmax rounded up to the nearest multiple of TILE_SIZE_JTDAJ
    njmax_nnz: number of non-zeros in constraint Jacobian
    nacon: number of detected contacts (across all worlds)      (1,)
    ncollision: collision count from broadphase                 (1,)
    flex_aabb_min: dynamic flex object bounding box min         (nworld, nflex, 3)
    flex_aabb_max: dynamic flex object bounding box max         (nworld, nflex, 3)
    flexnode_xpos: cartesian flex node positions                (nworld, nflexnode, 3)
    overflow: overflow bitmask (OverflowType)                   (nworld,)
    face_xpos: cartesian flex face positions                    (nworld, nflexface, 9, 3)
    face_quat: cartesian flex face orientations                 (nworld, nflexface, 4)
    flex_rod_chain_ok: whether 1D rod DER chain succeeded       (nworld, nflex)
    flex_rodhinge_g: 1D rod hinge kb-space gradient             (nworld, nflexrodhinge, 3)
    flex_rodhinge_Wk: 1D rod hinge kb-space stiffness weight    (nworld, nflexrodhinge, 3, 3)
    flex_rodhinge_Wd: 1D rod hinge kb-space damping weight      (nworld, nflexrodhinge, 3, 3)
    flex_rodedge_ktheta: 1D rod edge twist bending stiffness    (nworld, nflexrodedge)
    flex_rodedge_grad: 1D rod edge dE/dtheta                    (nworld, nflexrodedge)
    flex_rodedge_frc: 1D rod edge twist/director force          (nworld, nflexrodedge, 3)
    flex_rodedge_tridiag: 1D rod twist tridiagonal scratch      (nworld, nflexrodedge, 2)
    flex_rod_end_torque: 1D rod clamped-end reaction torques    (nworld, nflex, 6)
    flex_rod_chain_valid: whether 1D rod chain cache is current (nworld, nflex)
    flex_rod_ref_ok: whether 1D rod qpos0 reference is valid    (nworld, nflex)
    flex_rod_frame0_loc: 1D rod qpos0 first-edge frame (t, u)   (nworld, nflex, 2, 3)
                         in the frame of its first body
    flex_rod_frame0_ref: 1D rod qpos0 first-edge frame (t, u)   (nworld, nflex, 2, 3)
                         in world coordinates
    flex_rod_tail_loc: 1D rod qpos0 last-edge frame (t, u)      (nworld, nflex, 2, 3)
                       in the frame of its last body
    flex_rodhinge_om0: 1D rod hinge reference curvatures        (nworld, nflexrodhinge, 2)
  """

  solver_niter: array("nworld", int)
  ne: array("nworld", int)
  nf: array("nworld", int)
  nl: array("nworld", int)
  nefc: array("nworld", int)
  nisland: array("nworld", int)
  nidof: array("nworld", int)
  ntree_awake: array("nworld", int)
  nbody_awake: array("nworld", int)
  nv_awake: array("nworld", int)
  time: array("nworld", float)
  energy: array("nworld", wp.vec2)
  qpos: array("nworld", "nq", float)
  qvel: array("nworld", "nv", float)
  act: array("nworld", "na", float)
  history: array("nworld", "nhistory", float)
  plastic: array("nworld", "nplastic", float)
  qacc_warmstart: array("nworld", "nv", float)
  ctrl: array("nworld", "nu", float)
  qfrc_applied: array("nworld", "nv", float)
  xfrc_applied: array("nworld", "nbody", wp.spatial_vector)
  eq_active: array("nworld", "neq", bool)
  mocap_pos: array("nworld", "nmocap", wp.vec3)
  mocap_quat: array("nworld", "nmocap", wp.quat)
  qacc: array("nworld", "nv", float)
  act_dot: array("nworld", "na", float)
  userdata: array("nworld", "nuserdata", float)
  sensordata: array("nworld", "nsensordata", float)
  tree_asleep: array("nworld", "ntree", int)
  xpos: array("nworld", "nbody", wp.vec3)
  xquat: array("nworld", "nbody", wp.quat)
  xmat: array("nworld", "nbody", wp.mat33)
  xipos: array("nworld", "nbody", wp.vec3)
  ximat: array("nworld", "nbody", wp.mat33)
  xanchor: array("nworld", "njnt", wp.vec3)
  xaxis: array("nworld", "njnt", wp.vec3)
  geom_xpos: array("nworld", "ngeom", wp.vec3)
  geom_xmat: array("nworld", "ngeom", wp.mat33)
  site_xpos: array("nworld", "nsite", wp.vec3)
  site_xmat: array("nworld", "nsite", wp.mat33)
  cam_xpos: array("nworld", "ncam", wp.vec3)
  cam_xmat: array("nworld", "ncam", wp.mat33)
  light_xpos: array("nworld", "nlight", wp.vec3)
  light_xdir: array("nworld", "nlight", wp.vec3)
  subtree_com: array("nworld", "nbody", wp.vec3)
  cdof: array("nworld", "nv", wp.spatial_vector)
  cinert: array("nworld", "nbody", vec10)
  flexvert_xpos: array("nworld", "nflexvert", wp.vec3)
  flex_hessian_valid: array("nworld", "nflex", bool)
  flexvert_hessian: array("nworld", "nflexvert", vec6)
  flexedge_hessian: array("nworld", "nflexedge", wp.mat33)
  flexedge_J: array("nworld", "nJfe", float)
  flexedge_length: array("nworld", "nflexedge", float)
  ten_wrapadr: array("nworld", "ntendon", int)
  ten_wrapnum: array("nworld", "ntendon", int)
  ten_J: array("nworld", "nJten", float)
  ten_length: array("nworld", "ntendon", float)
  wrap_obj: array("nworld", "nwrap", wp.vec2i)
  wrap_xpos: array("nworld", "nwrap", wp.spatial_vector)
  actuator_length: array("nworld", "nactuator", float)
  moment_rownnz: array("nworld", "nactuator", int)
  moment_rowadr: array("nworld", "nactuator", int)
  moment_colind: array("nworld", "nJmom", int)
  actuator_moment: array("nworld", "nJmom", float)
  crb: array("nworld", "nbody", vec10)
  M: array("nworld", "nC", float)
  qLD: array("nworld", "qld_total", float)
  qLDiagInv: array("nworld", "nv", float)
  tree_awake: array("nworld", "ntree", int)
  body_awake: array("nworld", "nbody", int)
  body_awake_ind: array("nworld", "nbody", int)
  dof_awake_ind: array("nworld", "nv", int)
  flexedge_velocity: array("nworld", "nflexedge", float)
  ten_velocity: array("nworld", "ntendon", float)
  actuator_velocity: array("nworld", "nactuator", float)
  cvel: array("nworld", "nbody", wp.spatial_vector)
  cdof_dot: array("nworld", "nv", wp.spatial_vector)
  qfrc_bias: array("nworld", "nv", float)
  qfrc_spring: array("nworld", "nv", float)
  qfrc_damper: array("nworld", "nv", float)
  qfrc_gravcomp: array("nworld", "nv", float)
  qfrc_fluid: array("nworld", "nv", float)
  qfrc_adhesion: array("nworld", "nv", float)
  qfrc_passive: array("nworld", "nv", float)
  subtree_linvel: array("nworld", "nbody", wp.vec3)
  subtree_angmom: array("nworld", "nbody", wp.vec3)
  qH: array("nworld", "nC", float)
  qHDiagInv: array("nworld", "nv", float)
  qLU: array("nworld", "nD", float)
  actuator_force: array("nworld", "nactuator", float)
  qfrc_actuator: array("nworld", "nv", float)
  qfrc_smooth: array("nworld", "nv", float)
  qacc_smooth: array("nworld", "nv", float)
  qfrc_constraint: array("nworld", "nv", float)
  qfrc_inverse: array("nworld", "nv", float)
  cacc: array("nworld", "nbody", wp.spatial_vector)
  cfrc_int: array("nworld", "nbody", wp.spatial_vector)
  cfrc_ext: array("nworld", "nbody", wp.spatial_vector)
  efm_c: array("nworld", "nv", float)
  efm_diag: array("nworld", "nv", float)
  efm_fluid: array("nworld", "nC", float)
  efm_ca: array("nworld", "nv", float)
  efm_K_val: array("nworld", "nefmK", float)
  qHLD: array("nworld", "qld_total", float)
  efm_ts: array("nworld", "ntendon", float)
  efm_as: array("nworld", "nactuator", float)
  contact: Contact
  efc: Constraint
  dof_frictionadr: array("nworld", "nv", int)
  tree_island: array("nworld", "ntree", int)
  dof_island: array("nworld", "nv", int)
  island_dofadr: array("nworld", "ntree", int)
  island_idofadr: array("nworld", "ntree", int)
  island_nv: array("nworld", "ntree", int)
  island_nefc: array("nworld", "ntree", int)
  island_ne: array("nworld", "ntree", int)
  island_nf: array("nworld", "ntree", int)
  island_iefcadr: array("nworld", "ntree", int)
  map_dof2idof: array("nworld", "nv", int)
  map_idof2dof: array("nworld", "nv", int)
  map_efc2iefc: array("nworld", "njmax", int)
  map_iefc2efc: array("nworld", "njmax", int)
  dof_islandid: array("nworld", "nv", int)
  efc_islandid: array("nworld", "njmax", int)
  ncdof: array("nworld", int)
  dof_cdof: array("nworld", "nv", int)
  cdof_dof: array("nworld", "nvmax_pad", int)
  cdof_tri_row: array("nvmax_pad_sq", int)
  cdof_tri_col: array("nvmax_pad_sq", int)
  cM: array("nworld", "nvmax_pad", "nvmax_pad", float)
  cqLD: array("nworld", "nvmax_pad", "nvmax_pad", float)
  crhs: array("nworld", "nvmax_pad", 1, float)
  cx: array("nworld", "nvmax_pad", 1, float)
  cJ: array("nworld", "njmax_pad", "nvmax_pad", float)
  cMa: array("nworld", "nvmax_pad", float)
  cqfrc_smooth: array("nworld", "nvmax_pad", float)
  cqacc_smooth: array("nworld", "nvmax_pad", float)
  cqacc_warmstart: array("nworld", "nvmax_pad", float)
  cqacc: array("nworld", "nvmax_pad", float)
  cqfrc_constraint: array("nworld", "nvmax_pad", float)

  # warp only fields:
  nworld: int
  naconmax: int
  naccdmax: int
  njmax: int
  nvmax: int
  nvmax_pad: int
  njmax_pad: int
  njmax_nnz: int
  nacon: array(1, int)
  ncollision: array(1, int)
  flex_aabb_min: array("nworld", "nflex", wp.vec3)
  flex_aabb_max: array("nworld", "nflex", wp.vec3)
  flexnode_xpos: array("nworld", "nflexnode", wp.vec3)
  overflow: array("nworld", int)
  face_xpos: array("nworld", "nflexface", 9, wp.vec3)
  face_quat: array("nworld", "nflexface", wp.quat)
  flex_rod_chain_ok: array("nworld", "nflex", bool)
  flex_rodhinge_g: array("nworld", "nflexrodhinge", wp.vec3)
  flex_rodhinge_Wk: array("nworld", "nflexrodhinge", wp.mat33)
  flex_rodhinge_Wd: array("nworld", "nflexrodhinge", wp.mat33)
  flex_rodedge_ktheta: array("nworld", "nflexrodedge", float)
  flex_rodedge_grad: array("nworld", "nflexrodedge", float)
  flex_rodedge_frc: array("nworld", "nflexrodedge", wp.vec3)
  flex_rodedge_tridiag: array("nworld", "nflexrodedge", wp.vec2)
  flex_rod_end_torque: array("nworld", "nflex", wp.spatial_vector)
  flex_rod_chain_valid: array("nworld", "nflex", bool)
  flex_rod_ref_ok: array("nworld", "nflex", bool)
  flex_rod_frame0_loc: array("nworld", "nflex", 2, wp.vec3)
  flex_rod_frame0_ref: array("nworld", "nflex", 2, wp.vec3)
  flex_rod_tail_loc: array("nworld", "nflex", 2, wp.vec3)
  flex_rodhinge_om0: array("nworld", "nflexrodhinge", wp.vec2)


@dataclasses.dataclass
class InverseContext:
  """Workspace arrays for inverse dynamics."""

  Jaref: wp.array2d[float]
  search_dot: wp.array[float]
  done: wp.array[bool]
  quad_changed_ids: wp.array2d[int]
  quad_changed_count: wp.array[int]
  state_changed_count: wp.array[int]
  ls_exhausted: wp.array[bool]
  # the full-coordinate Data, set by solve_compact (None natively)
  compact_m_full: Optional["Model"] = None
  compact_d_full: Optional["Data"] = None
  # discrete integrator effective-metric scratch (see SolverContext)
  efm_con: Optional[tuple] = None
  flex_scratch: Optional[tuple] = None
  tw_diag: Optional[wp.array] = None


@dataclasses.dataclass
class SolverContext:
  """Workspace arrays for constraint solver."""

  Jaref: wp.array2d[float]
  search_dot: wp.array[float]
  done: wp.array[bool]
  grad: wp.array2d[float]
  grad_dot: wp.array[float]
  newton_decrement: wp.array[float]
  Mgrad: wp.array2d[float]
  search: wp.array2d[float]
  mv: wp.array2d[float]
  jv: wp.array2d[float]
  quad: wp.array2d[wp.vec3]
  alpha: wp.array[float]
  grad_scale: wp.array[float]
  state_changed_count: wp.array[int]
  improvement: wp.array[float]
  ls_exhausted: wp.array[bool]
  search_unchanged: wp.array[bool]
  prev_grad: wp.array2d[float]
  prev_Mgrad: wp.array2d[float]
  beta: wp.array[float]
  h: wp.array3d[float]
  hfactor: wp.array3d[float]
  quad_changed_ids: wp.array2d[int]
  quad_changed_count: wp.array[int]
  inertia_scale: wp.array[float]
  # the full-coordinate Data, set by solve_compact (None natively)
  compact_m_full: Optional["Model"] = None
  compact_d_full: Optional["Data"] = None
  # discrete integrator effective-metric scratch built before the solver loop, which may run
  # inside wp.capture_while where memory cannot be allocated (None otherwise)
  efm_con: Optional[tuple] = None
  flex_scratch: Optional[tuple] = None
  solve_ld_scratch: Optional[wp.array] = None
  tw_diag: Optional[wp.array] = None
  h_tile_nz: Optional[wp.array3d[int]] = None
  h_row_mask: Optional[wp.array3d[wp.uint64]] = None
  h_col_mask: Optional[wp.array3d[wp.uint64]] = None
  h_tile_is_diag: Optional[wp.array2d[int]] = None
  h_rhs_work: Optional[wp.array3d[float]] = None


@dataclasses.dataclass
class RenderContext:
  """Context for rendering.

  Attributes:
    nrender: number of actively rendering cameras
    cam_res: camera resolution for actively rendering cameras
    cam_id_map: camera id map
    use_textures: whether to use textures
    use_fast_math: whether to enable fast math for the render kernel
    use_shadows: whether to use shadows
    use_ambient_lighting: top-level switch for ambient contributions
    background_color: color used for missed rays when no skybox is rendered
    use_precomputed_rays: whether to use precomputed rays
    bvh_ngeom: number of geometries in the BVH
    enabled_geom_ids: enabled geometry ids
    mesh_registry: mesh BVH id to warp mesh mapping
    mesh_bvh_id: mesh BVH ids
    mesh_bounds_size: mesh bounds size
    mesh_texcoord: mesh texture coordinates
    mesh_texcoord_offsets: mesh texture coordinate offsets
    mesh_facetexcoord: mesh face texture coordinates
    mesh_facenormal: per-face indices into Model.mesh_normal
    samples_per_pixel: sub-pixel samples per axis; 1 disables supersampling
    aa_accum: colour accumulator, unused when samples_per_pixel is 1
    textures: textures
    textures_registry: texture registry
    hfield_registry: hfield BVH id to warp mesh mapping
    hfield_bvh_id: hfield BVH ids
    hfield_bounds_size: hfield bounds half-extents
    flex_mesh_registry: per-flex mesh BVH registry (prevents garbage collection)
    flex_rgba: flex rgba
    flex_bvh_id: per-flex BVH ids
    flex_group_root: per-flex group roots (nworld x n_flex_bvh)
    flex_render_smooth: whether to render flex meshes smoothly
    bvh_nflexgeom: number of flex geometries in the BVH
    flex_dim_np: flex dimension per flex (1D/2D/3D)
    flex_geom_flexid: map from flex geom ID to flex ID
    flex_geom_edgeid: map from flex geom ID to flex edge ID
    bvh: scene BVH
    bvh_id: scene BVH id
    lower: lower bounds
    upper: upper bounds
    group: groups
    group_root: group roots
    ray: per-pixel rays direction
    ray_offset: per-pixel ray offset from camera center (for orthographic
      camera projections only)
    rgb_data: RGB data
    rgb_adr: RGB addresses
    depth_data: depth data
    depth_adr: depth addresses
    render_rgb: per-camera RGB render flags
    render_depth: per-camera depth render flags
    seg_data: segmentation data (per-pixel object ID/type pairs)
    seg_adr: segmentation addresses
    render_seg: per-camera segmentation render flags
    znear: near plane distance
    zfar: far plane distance
    total_rays: total number of rays
    render_skybox: whether to shade missed rays with a MuJoCo skybox texture
    skybox_tex_id: per-world indices into textures of the skybox
    skybox_face_width: per-world pixel widths of the skybox cube face
    headlight_active: whether to inject MuJoCo's vis.headlight as a synthetic
      directional light at the active camera. Read from `mjm.vis.headlight.active`
      at context creation; users disable the headlight by configuring it on the
      MuJoCo model (e.g. `<visual><headlight active="0"/></visual>` in XML).
    headlight_ambient: RGB ambient color of the headlight (from vis.headlight).
    headlight_diffuse: RGB diffuse color of the headlight.
    headlight_specular: RGB specular color of the headlight.
    shadow_light_fraction: fraction of a light's direct contribution reaching an
      occluded point; 0 is a true shadow
    enable_backface_culling: drop primitive ray hits whose normal faces away
      from the ray (i.e. the ray origin is inside the geom). Matches MuJoCo's
      mesh-ray rule. When False, the renderer reports inner-surface hits, which
      is faster but causes a camera placed inside a geom to render that geom's
      back wall.
    light_attenuation_is_default: True iff every light in the model has the
      MuJoCo default `attenuation = (1, 0, 0)`. Computed once at context
      creation; when True the kernel skips the per-light polynomial
      attenuation evaluation (a divide + 3 multiplies + an add per
      non-directional light per pixel) via `wp.static`.
    has_spot_lights: True iff any light in the model has `type == SPOT`.
      When False, the kernel skips the spot-cone branch (cos cutoff +
      pow exponent) per non-directional light per pixel via `wp.static`.
    has_orthographic_camera: True iff any actively rendering camera uses
      orthographic projection. When False, the kernel skips the ray origin
      offset since it is always zero for perspective-only scenes.
    enable_vertex_normals: when True, shade meshes from their authored vertex
      normals, matching mjr_uploadMesh; when False, use the face normal.
    enable_specular: when True, evaluate the Phong specular highlight per
      light per pixel (uses `mat_specular` / `mat_shininess`). When False,
      the entire specular branch is removed at compile time. Useful for
      depth/segmentation-only workflows or when materials are matte.
    enable_emission: when True, add `mat_emission * base_color` to each
      shaded pixel. When False the term is dropped at compile time.
    enable_per_light_ambient: when True and `use_ambient_lighting` is also
      True, sum the per-light `light_ambient` colors into each shaded pixel
      even when the surface normal is perpendicular to the light direction
      or the pixel is shadowed. When False the second per-light loop for
      ambient is removed at compile time. Headlight ambient and the no-light
      fallback are controlled by `use_ambient_lighting`.
    geom_ray_types: tuple of GeomType int values present in the scene, used to
      statically eliminate unused intersection branches in the ray-cast kernels.
    splat_position: Splat centers in world coordinates (nsplat, 3)
    splat_rotation: Splat rotation as (w, x, y, z) (nsplat, 4)
    splat_scale: Splat scale as standard deviation in each dimension (nsplat, 3)
    splat_rgba: Splat color and opacity (nsplat, 4)
    splat_bvh: Splat BVH
    splat_lower: Splat lower bounds
    splat_upper: Splat upper bounds
    splat_bvh_id: Splat BVH id
    splat_group_root: Per-world selected splat BVH root
    splat_count: Number of splats
  """

  nrender: int
  cam_res: array("ncam", wp.vec2i)
  cam_id_map: array("ncam", int)
  use_textures: bool
  use_fast_math: bool
  use_shadows: bool
  use_ambient_lighting: bool
  background_color: wp.uint32
  use_precomputed_rays: bool
  render_skybox: bool
  skybox_tex_id: array("*", int)
  skybox_face_width: array("*", int)
  headlight_active: bool
  headlight_ambient: wp.vec3
  headlight_diffuse: wp.vec3
  headlight_specular: wp.vec3
  bvh_ngeom: int
  enabled_geom_ids: array("*", int)
  mesh_registry: dict
  mesh_bvh_id: array("nmesh", wp.uint64)
  mesh_bounds_size: array("nmesh", wp.vec3)
  mesh_texcoord: array("*", wp.vec2)
  mesh_texcoord_offsets: array("nmesh", int)
  mesh_facetexcoord: array("nmeshface", wp.vec3i)
  mesh_facenormal: array("nmeshface", wp.vec3i)
  samples_per_pixel: int
  aa_accum: array("*", wp.vec3)
  textures: array("*", wp.Texture2D)
  textures_registry: list[wp.Texture2D]
  hfield_registry: dict
  hfield_bvh_id: array("nhfield", wp.uint64)
  hfield_bounds_size: array("nhfield", wp.vec3)
  flex_mesh_registry: dict
  flex_rgba: array("nflex", wp.vec4)
  flex_bvh_id: array("*", wp.uint64)
  flex_group_root: array("nworld", "*", int)
  flex_render_smooth: bool
  bvh_nflexgeom: int
  flex_dim_np: array("nflex", int)
  flex_geom_flexid: array("*", int)
  flex_geom_edgeid: array("*", int)
  bvh: wp.Bvh
  bvh_id: wp.uint64
  lower: array("*", wp.vec3)
  upper: array("*", wp.vec3)
  group: array("*", int)
  group_root: array("*", int)
  ray: array("*", wp.vec3)
  ray_offset: array("*", wp.vec3)
  rgb_data: array("*", wp.uint32)
  rgb_adr: array("ncam", int)
  depth_data: array("*", wp.float32)
  depth_adr: array("ncam", int)
  render_rgb: array("ncam", bool)
  render_depth: array("ncam", bool)
  seg_data: array("*", wp.vec2i)
  seg_adr: array("ncam", int)
  render_seg: array("ncam", bool)
  znear: float
  zfar: float
  total_rays: int
  enable_backface_culling: bool
  shadow_light_fraction: float
  enable_vertex_normals: bool
  enable_specular: bool
  enable_emission: bool
  enable_per_light_ambient: bool
  light_attenuation_is_default: bool
  has_spot_lights: bool
  has_orthographic_camera: bool
  splat_position: array("*", wp.vec3)
  splat_rotation: array("*", wp.quat)
  splat_scale: array("*", wp.vec3)
  splat_rgba: array("*", wp.vec4)
  splat_bvh: Optional[wp.Bvh]
  splat_lower: array("*", wp.vec3)
  splat_upper: array("*", wp.vec3)
  splat_bvh_id: wp.uint64
  splat_group_root: array("nworld", int)
  splat_count: int
  geom_ray_types: tuple = ()
  _megakernel: Optional[wp.Kernel] = None
