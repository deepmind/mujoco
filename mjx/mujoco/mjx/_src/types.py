# Copyright 2023 DeepMind Technologies Limited
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
"""Base types used in MJX."""

import dataclasses
import enum
from typing import Any, Tuple, Union
import warnings

import jax
import mujoco
from mujoco.mjx._src.dataclasses import PyTreeNode  # pylint: disable=g-importing-member
from mujoco.mjx.warp import types as mjxw_types
import numpy as np


class Impl(enum.Enum):
  """Implementation to use."""

  CPP = 'cpp'
  C = 'cpp'  # alias C -> CPP
  JAX = 'jax'
  WARP = 'warp'

  @classmethod
  def _missing_(cls, value):
    # This method is called only when lookup by value fails
    # (e.g., Impl('JAX') fails initially because 'JAX' != 'jax')
    if not isinstance(value, str):
      return None
    for member in cls:
      if member.value == value.lower():
        return member
    return None


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
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
    SENSOR:       sensors
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
  ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
  REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
  SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
  EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  # unsupported: MIDPHASE


class EnableBit(enum.IntFlag):
  """Enable optional feature bitflags.

  Attributes:
    INVDISCRETE: discrete-time inverse dynamics
    SLEEP: sleeping
    IPC: IPC flex contact mode of the discrete integrator
  """

  INVDISCRETE = mujoco.mjtEnableBit.mjENBL_INVDISCRETE
  # unsupported: OVERRIDE, ENERGY, FWDINV, ISLAND, DIAGEXACT, SPECULATIVE
  SLEEP = mujoco.mjtEnableBit.mjENBL_SLEEP
  IPC = mujoco.mjtEnableBit.mjENBL_IPC


class JointType(enum.IntEnum):
  """Type of degree of freedom.

  Attributes:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """

  FREE = mujoco.mjtJoint.mjJNT_FREE  # pyrefly: ignore[bad-assignment]
  BALL = mujoco.mjtJoint.mjJNT_BALL  # pyrefly: ignore[bad-assignment]
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE  # pyrefly: ignore[bad-assignment]
  HINGE = mujoco.mjtJoint.mjJNT_HINGE  # pyrefly: ignore[bad-assignment]

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class IntegratorType(enum.IntEnum):
  """Integrator mode.

  Attributes:
    EULER: semi-implicit Euler
    RK4: 4th-order Runge Kutta
    IMPLICITFAST: implicit in velocity, no rne derivative
    DISCRETE: discrete step map
  """

  EULER = mujoco.mjtIntegrator.mjINT_EULER  # pyrefly: ignore[bad-assignment]
  RK4 = mujoco.mjtIntegrator.mjINT_RK4  # pyrefly: ignore[bad-assignment]
  IMPLICITFAST = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # pyrefly: ignore[bad-assignment]
  DISCRETE = mujoco.mjtIntegrator.mjINT_DISCRETE  # pyrefly: ignore[bad-assignment]
  # unsupported: IMPLICIT


class GeomType(enum.IntEnum):
  """Type of geometry.

  Attributes:
    PLANE: plane
    HFIELD: height field
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
    SDF: signed distance field
  """

  PLANE = mujoco.mjtGeom.mjGEOM_PLANE  # pyrefly: ignore[bad-assignment]
  HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD  # pyrefly: ignore[bad-assignment]
  SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE  # pyrefly: ignore[bad-assignment]
  CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE  # pyrefly: ignore[bad-assignment]
  ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID  # pyrefly: ignore[bad-assignment]
  CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER  # pyrefly: ignore[bad-assignment]
  BOX = mujoco.mjtGeom.mjGEOM_BOX  # pyrefly: ignore[bad-assignment]
  MESH = mujoco.mjtGeom.mjGEOM_MESH  # pyrefly: ignore[bad-assignment]
  # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


class ConvexMesh(PyTreeNode):
  """Geom properties for convex meshes.

  Attributes:
    vert: vertices of the convex mesh
    face: faces of the convex mesh
    face_normal: normal vectors for the faces
    edge: edge indexes for all edges in the convex mesh
    edge_face_normal: indexes for face normals adjacent to edges in `edge`
  """

  vert: jax.Array
  face: jax.Array
  face_normal: jax.Array
  edge: jax.Array
  edge_face_normal: jax.Array


class ConeType(enum.IntEnum):
  """Type of friction cone.

  Attributes:
    PYRAMIDAL: pyramidal
    ELLIPTIC: elliptic
  """

  PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL  # pyrefly: ignore[bad-assignment]
  ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC  # pyrefly: ignore[bad-assignment]


class JacobianType(enum.IntEnum):
  """Type of constraint Jacobian.

  Attributes:
    DENSE: dense
    SPARSE: sparse
    AUTO: sparse if nv>60 and device is TPU, dense otherwise
  """

  DENSE = mujoco.mjtJacobian.mjJAC_DENSE  # pyrefly: ignore[bad-assignment]
  SPARSE = mujoco.mjtJacobian.mjJAC_SPARSE  # pyrefly: ignore[bad-assignment]
  AUTO = mujoco.mjtJacobian.mjJAC_AUTO  # pyrefly: ignore[bad-assignment]


class SolverType(enum.IntEnum):
  """Constraint solver algorithm.

  Attributes:
    CG: Conjugate gradient (primal)
    NEWTON: Newton (primal)
  """

  # unsupported: PGS
  CG = mujoco.mjtSolver.mjSOL_CG  # pyrefly: ignore[bad-assignment]
  NEWTON = mujoco.mjtSolver.mjSOL_NEWTON  # pyrefly: ignore[bad-assignment]


class EqType(enum.IntEnum):
  """Type of equality constraint.

  Attributes:
    CONNECT: connect two bodies at a point (ball joint)
    WELD: fix relative position and orientation of two bodies
    JOINT: couple the values of two scalar joints with cubic
    TENDON: couple the lengths of two tendons with cubic
  """

  CONNECT = mujoco.mjtEq.mjEQ_CONNECT  # pyrefly: ignore[bad-assignment]
  WELD = mujoco.mjtEq.mjEQ_WELD  # pyrefly: ignore[bad-assignment]
  JOINT = mujoco.mjtEq.mjEQ_JOINT  # pyrefly: ignore[bad-assignment]
  TENDON = mujoco.mjtEq.mjEQ_TENDON  # pyrefly: ignore[bad-assignment]
  # unsupported: DISTANCE


class TendonType(enum.IntEnum):
  """Type of tendon.

  Attributes:
    FIXED: linear combination of joint coordinates
    SPATIAL: path through sites, around geoms
  """

  FIXED = mujoco.mjtTendon.mjTENDON_FIXED  # pyrefly: ignore[bad-assignment]
  SPATIAL = mujoco.mjtTendon.mjTENDON_SPATIAL  # pyrefly: ignore[bad-assignment]


class WrapType(enum.IntEnum):
  """Type of tendon wrap object.

  Attributes:
    JOINT: constant moment arm
    PULLEY: pulley used to split tendon
    SITE: pass through site
    SPHERE: wrap around sphere
    CYLINDER: wrap around (infinite) cylinder
  """

  JOINT = mujoco.mjtWrap.mjWRAP_JOINT  # pyrefly: ignore[bad-assignment]
  PULLEY = mujoco.mjtWrap.mjWRAP_PULLEY  # pyrefly: ignore[bad-assignment]
  SITE = mujoco.mjtWrap.mjWRAP_SITE  # pyrefly: ignore[bad-assignment]
  SPHERE = mujoco.mjtWrap.mjWRAP_SPHERE  # pyrefly: ignore[bad-assignment]
  CYLINDER = mujoco.mjtWrap.mjWRAP_CYLINDER  # pyrefly: ignore[bad-assignment]


class TrnType(enum.IntEnum):
  """Type of actuator transmission.

  Attributes:
    JOINT: force on joint
    JOINTINPARENT: force on joint, expressed in parent frame
    TENDON: force on tendon
    SITE: force on site
  """

  JOINT = mujoco.mjtTrn.mjTRN_JOINT  # pyrefly: ignore[bad-assignment]
  JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT  # pyrefly: ignore[bad-assignment]
  SITE = mujoco.mjtTrn.mjTRN_SITE  # pyrefly: ignore[bad-assignment]
  TENDON = mujoco.mjtTrn.mjTRN_TENDON  # pyrefly: ignore[bad-assignment]
  # unsupported: SLIDERCRANK, BODY


class DynType(enum.IntEnum):
  """Type of actuator dynamics.

  Attributes:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
    MUSCLE: piece-wise linear filter with two time constants
  """

  NONE = mujoco.mjtDyn.mjDYN_NONE  # pyrefly: ignore[bad-assignment]
  INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR  # pyrefly: ignore[bad-assignment]
  FILTER = mujoco.mjtDyn.mjDYN_FILTER  # pyrefly: ignore[bad-assignment]
  FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT  # pyrefly: ignore[bad-assignment]
  MUSCLE = mujoco.mjtDyn.mjDYN_MUSCLE  # pyrefly: ignore[bad-assignment]
  # unsupported: USER


class GainType(enum.IntEnum):
  """Type of actuator gain.

  Attributes:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle FLV curve computed by muscle_gain
  """

  FIXED = mujoco.mjtGain.mjGAIN_FIXED  # pyrefly: ignore[bad-assignment]
  AFFINE = mujoco.mjtGain.mjGAIN_AFFINE  # pyrefly: ignore[bad-assignment]
  MUSCLE = mujoco.mjtGain.mjGAIN_MUSCLE  # pyrefly: ignore[bad-assignment]
  # unsupported: USER


class BiasType(enum.IntEnum):
  """Type of actuator bias.

  Attributes:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle passive force computed by muscle_bias
  """

  NONE = mujoco.mjtBias.mjBIAS_NONE  # pyrefly: ignore[bad-assignment]
  AFFINE = mujoco.mjtBias.mjBIAS_AFFINE  # pyrefly: ignore[bad-assignment]
  MUSCLE = mujoco.mjtBias.mjBIAS_MUSCLE  # pyrefly: ignore[bad-assignment]
  # unsupported: USER


class ConstraintType(enum.IntEnum):
  """Type of constraint.

  Attributes:
    EQUALITY: equality constraint
    LIMIT_JOINT: joint limit
    LIMIT_TENDON: tendon limit
    CONTACT_FRICTIONLESS: frictionless contact
    CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
  """

  EQUALITY = mujoco.mjtConstraint.mjCNSTR_EQUALITY  # pyrefly: ignore[bad-assignment]
  FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF  # pyrefly: ignore[bad-assignment]
  FRICTION_TENDON = mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON  # pyrefly: ignore[bad-assignment]
  LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT  # pyrefly: ignore[bad-assignment]
  LIMIT_TENDON = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON  # pyrefly: ignore[bad-assignment]
  CONTACT_FRICTIONLESS = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS  # pyrefly: ignore[bad-assignment]
  CONTACT_PYRAMIDAL = mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL  # pyrefly: ignore[bad-assignment]
  CONTACT_ELLIPTIC = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC  # pyrefly: ignore[bad-assignment]


class CamLightType(enum.IntEnum):
  """Type of camera light.

  Attributes:
    FIXED: pos and rot fixed in body
    TRACK: pos tracks body, rot fixed in global
    TRACKCOM: pos tracks subtree com, rot fixed in body
    TARGETBODY: pos fixed in body, rot tracks target body
    TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
  """

  FIXED = mujoco.mjtCamLight.mjCAMLIGHT_FIXED  # pyrefly: ignore[bad-assignment]
  TRACK = mujoco.mjtCamLight.mjCAMLIGHT_TRACK  # pyrefly: ignore[bad-assignment]
  TRACKCOM = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM  # pyrefly: ignore[bad-assignment]
  TARGETBODY = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY  # pyrefly: ignore[bad-assignment]
  TARGETBODYCOM = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM  # pyrefly: ignore[bad-assignment]


class SensorType(enum.IntEnum):
  """Type of sensor.

  Attributes:
    MAGNETOMETER: magnetometer
    CAMPROJECTION: camera projection
    RANGEFINDER: rangefinder
    JOINTPOS: joint position
    TENDONPOS: scalar tendon position
    ACTUATORPOS: actuator position
    BALLQUAT: ball joint orientation
    FRAMEPOS: frame position
    FRAMEXAXIS: frame x-axis
    FRAMEYAXIS: frame y-axis
    FRAMEZAXIS: frame z-axis
    FRAMEQUAT: frame orientation, represented as quaternion
    SUBTREECOM: subtree centor of mass
    CLOCK: simulation time
    VELOCIMETER: 3D linear velocity, in local frame
    GYRO: 3D angular velocity, in local frame
    JOINTVEL: joint velocity
    TENDONVEL: scalar tendon velocity
    ACTUATORVEL: actuator velocity
    BALLANGVEL: ball joint angular velocity
    FRAMELINVEL: 3D linear velocity
    FRAMEANGVEL: 3D angular velocity
    SUBTREELINVEL: subtree linear velocity
    SUBTREEANGMOM: subtree angular momentum
    TOUCH: scalar contact normal forces summed over the sensor zone
    CONTACT: contacts which occurred during the simulation
    ACCELEROMETER: accelerometer
    FORCE: force
    TORQUE: torque
    ACTUATORFRC: scalar actuator force
    JOINTACTFRC: scalar actuator force, measured at the joint
    TENDONACTFRC: scalar actuator force, measured at the tendon
    FRAMELINACC: 3D linear acceleration
    FRAMEANGACC: 3D angular acceleration
  """

  MAGNETOMETER = mujoco.mjtSensor.mjSENS_MAGNETOMETER  # pyrefly: ignore[bad-assignment]
  CAMPROJECTION = mujoco.mjtSensor.mjSENS_CAMPROJECTION  # pyrefly: ignore[bad-assignment]
  RANGEFINDER = mujoco.mjtSensor.mjSENS_RANGEFINDER  # pyrefly: ignore[bad-assignment]
  JOINTPOS = mujoco.mjtSensor.mjSENS_JOINTPOS  # pyrefly: ignore[bad-assignment]
  TENDONPOS = mujoco.mjtSensor.mjSENS_TENDONPOS  # pyrefly: ignore[bad-assignment]
  ACTUATORPOS = mujoco.mjtSensor.mjSENS_ACTUATORPOS  # pyrefly: ignore[bad-assignment]
  BALLQUAT = mujoco.mjtSensor.mjSENS_BALLQUAT  # pyrefly: ignore[bad-assignment]
  FRAMEPOS = mujoco.mjtSensor.mjSENS_FRAMEPOS  # pyrefly: ignore[bad-assignment]
  FRAMEXAXIS = mujoco.mjtSensor.mjSENS_FRAMEXAXIS  # pyrefly: ignore[bad-assignment]
  FRAMEYAXIS = mujoco.mjtSensor.mjSENS_FRAMEYAXIS  # pyrefly: ignore[bad-assignment]
  FRAMEZAXIS = mujoco.mjtSensor.mjSENS_FRAMEZAXIS  # pyrefly: ignore[bad-assignment]
  FRAMEQUAT = mujoco.mjtSensor.mjSENS_FRAMEQUAT  # pyrefly: ignore[bad-assignment]
  SUBTREECOM = mujoco.mjtSensor.mjSENS_SUBTREECOM  # pyrefly: ignore[bad-assignment]
  CLOCK = mujoco.mjtSensor.mjSENS_CLOCK  # pyrefly: ignore[bad-assignment]
  VELOCIMETER = mujoco.mjtSensor.mjSENS_VELOCIMETER  # pyrefly: ignore[bad-assignment]
  GYRO = mujoco.mjtSensor.mjSENS_GYRO  # pyrefly: ignore[bad-assignment]
  JOINTVEL = mujoco.mjtSensor.mjSENS_JOINTVEL  # pyrefly: ignore[bad-assignment]
  TENDONVEL = mujoco.mjtSensor.mjSENS_TENDONVEL  # pyrefly: ignore[bad-assignment]
  ACTUATORVEL = mujoco.mjtSensor.mjSENS_ACTUATORVEL  # pyrefly: ignore[bad-assignment]
  BALLANGVEL = mujoco.mjtSensor.mjSENS_BALLANGVEL  # pyrefly: ignore[bad-assignment]
  FRAMELINVEL = mujoco.mjtSensor.mjSENS_FRAMELINVEL  # pyrefly: ignore[bad-assignment]
  FRAMEANGVEL = mujoco.mjtSensor.mjSENS_FRAMEANGVEL  # pyrefly: ignore[bad-assignment]
  SUBTREELINVEL = mujoco.mjtSensor.mjSENS_SUBTREELINVEL  # pyrefly: ignore[bad-assignment]
  SUBTREEANGMOM = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM  # pyrefly: ignore[bad-assignment]
  TOUCH = mujoco.mjtSensor.mjSENS_TOUCH  # pyrefly: ignore[bad-assignment]
  CONTACT = mujoco.mjtSensor.mjSENS_CONTACT  # pyrefly: ignore[bad-assignment]
  ACCELEROMETER = mujoco.mjtSensor.mjSENS_ACCELEROMETER  # pyrefly: ignore[bad-assignment]
  FORCE = mujoco.mjtSensor.mjSENS_FORCE  # pyrefly: ignore[bad-assignment]
  TORQUE = mujoco.mjtSensor.mjSENS_TORQUE  # pyrefly: ignore[bad-assignment]
  ACTUATORFRC = mujoco.mjtSensor.mjSENS_ACTUATORFRC  # pyrefly: ignore[bad-assignment]
  JOINTACTFRC = mujoco.mjtSensor.mjSENS_JOINTACTFRC  # pyrefly: ignore[bad-assignment]
  TENDONACTFRC = mujoco.mjtSensor.mjSENS_TENDONACTFRC  # pyrefly: ignore[bad-assignment]
  FRAMELINACC = mujoco.mjtSensor.mjSENS_FRAMELINACC  # pyrefly: ignore[bad-assignment]
  FRAMEANGACC = mujoco.mjtSensor.mjSENS_FRAMEANGACC  # pyrefly: ignore[bad-assignment]
  TACTILE = mujoco.mjtSensor.mjSENS_TACTILE  # pyrefly: ignore[bad-assignment]


class ObjType(PyTreeNode):
  """Type of object.

  Attributes:
    UNKNOWN: unknown object type
    BODY: body
    XBODY: body, used to access regular frame instead of i-frame
    GEOM: geom
    SITE: site
    CAMERA: camera
  """

  UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN
  BODY = mujoco.mjtObj.mjOBJ_BODY
  XBODY = mujoco.mjtObj.mjOBJ_XBODY
  GEOM = mujoco.mjtObj.mjOBJ_GEOM
  SITE = mujoco.mjtObj.mjOBJ_SITE
  CAMERA = mujoco.mjtObj.mjOBJ_CAMERA


class Statistic(PyTreeNode):
  """Model statistics (in qpos0).

  Attributes:
    meaninertia: mean diagonal inertia
    meanmass: mean body mass (not used)
    meansize: mean body size (not used)
    extent: spatial extent (not used)
    center: center of model (not used)
  """

  meaninertia: jax.Array
  meanmass: jax.Array
  meansize: jax.Array
  extent: jax.Array
  center: jax.Array


class OptionJAX(PyTreeNode):
  """JAX-specific option."""

  o_margin: jax.Array
  o_solref: jax.Array
  o_solimp: jax.Array
  o_friction: jax.Array
  disableactuator: int
  sdf_initpoints: int
  has_fluid_params: bool




class Option(PyTreeNode):
  """Physics options."""

  iterations: int
  ls_iterations: int
  tolerance: jax.Array
  ls_tolerance: jax.Array
  impratio: jax.Array
  gravity: jax.Array
  density: jax.Array
  viscosity: jax.Array
  magnetic: jax.Array
  wind: jax.Array
  jacobian: JacobianType
  cone: ConeType
  disableflags: DisableBit
  enableflags: int
  integrator: IntegratorType
  solver: SolverType
  timestep: jax.Array
  _impl: Union[OptionJAX, mjxw_types.OptionWarp]


class ModelCPP(PyTreeNode):
  """Minimal Model implementation holding only the pointer."""
  # To ensure that we retain the full pointer even if jax.config.enable_x64 is
  # set to True, we store the pointer as two 32-bit values. In the FFI call,
  # we combine the two values into a single pointer value.
  pointer_lo: jax.Array
  pointer_hi: jax.Array


class DataCPP(PyTreeNode):
  """Minimal Data implementation holding only the pointer."""
  # To ensure that we retain the full pointer even if jax.config.enable_x64 is
  # set to True, we store the pointer as two 32-bit values. In the FFI call,
  # we combine the two values into a single pointer value.
  pointer_lo: jax.Array
  pointer_hi: jax.Array


class ModelJAX(PyTreeNode):
  """JAX-specific model data."""

  dof_hasfrictionloss: np.ndarray
  geom_rbound_hfield: np.ndarray
  mesh_convex: Tuple[ConvexMesh, ...]
  tendon_hasfrictionloss: np.ndarray
  wrap_inside_maxiter: int
  wrap_inside_tolerance: float
  wrap_inside_z_init: float
  is_wrap_inside: np.ndarray


class Model(PyTreeNode):
  """Static model of the scene that remains unchanged with each physics step.

  Attributes:
    nq: number of generalized coordinates
    nv: number of degrees of freedom
    nu: number of actuators/controls
    na: number of activation states
    nbody: number of bodies
    njnt: number of joints
    ngeom: number of geoms
    nsite: number of sites
    ncam: number of cameras
    nlight: number of lights
    nmesh: number of meshes
    nmeshvert: number of vertices for all meshes
    nmeshnormal: number of normals in all meshes
    nmeshtexcoord: number of texcoords in all meshes
    nmeshface: number of faces for all meshes
    nmeshgraph: number of ints in mesh auxiliary data
    nmeshpoly: number of polygons in all meshes
    nmeshpolyvert: number of vertices in all polygons
    nmeshpolymap: number of polygons in vertex map
    nhfield: number of heightfields
    nhfielddata: size of elevation data
    ntex: number of textures
    ntexdata: size of texture data
    nmat: number of materials
    npair: number of predefined geom pairs
    nexclude: number of excluded geom pairs
    neq: number of equality constraints
    ntendon: number of tendons
    nwrap: number of wrap objects in all tendon paths
    nsensor: number of sensors
    nnumeric: number of numeric custom fields
    ntuple: number of tuple custom fields
    nkey: number of keyframes
    nmocap: number of mocap bodies
    nM: number of non-zeros in sparse inertia matrix
    nB: number of non-zeros in B matrix
    nC: number of non-zeros in C matrix
    nD: number of non-zeros in D matrix
    nJmom: number of non-zeros in Jacobian momentum matrix
    nJten: number of non-zeros in sparse tendon Jacobian
    ngravcomp: number of bodies with nonzero gravcomp
    nuserdata: number of elements in userdata
    nsensordata: number of elements in sensor data vector
    npluginstate: number of plugin state values
    nhistory: number of history buffer elements
    nplastic: number of plastic reference elements
    opt: physics options
    stat: model statistics
    qpos0: qpos values at default pose
    qpos_spring: reference pose for springs
  """

  nq: int
  nv: int
  nu: int
  na: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  ncam: int
  nlight: int
  nflex: int
  nmesh: int
  nmeshvert: int
  nmeshnormal: int
  nmeshtexcoord: int
  nmeshface: int
  nmeshgraph: int
  nmeshpoly: int
  nmeshpolyvert: int
  nmeshpolymap: int
  nhfield: int
  nhfielddata: int
  ntex: int
  ntexdata: int
  nmat: int
  npair: int
  nexclude: int
  neq: int
  ntendon: int
  nwrap: int
  nsensor: int
  nnumeric: int
  ntuple: int
  nkey: int
  nmocap: int
  nM: int  # pylint:disable=invalid-name
  nB: int  # pylint:disable=invalid-name
  nC: int  # pylint:disable=invalid-name
  nD: int  # pylint:disable=invalid-name
  nJmom: int  # pylint:disable=invalid-name
  nJten: int  # pylint:disable=invalid-name
  ngravcomp: int
  flg_gravcomp: bool
  flg_surfacevel: bool
  nuserdata: int
  nsensordata: int
  npluginstate: int
  nhistory: int
  nplastic: int
  opt: Option
  stat: Statistic
  qpos0: jax.Array
  qpos_spring: jax.Array
  body_parentid: np.ndarray
  body_mocapid: np.ndarray
  body_rootid: np.ndarray
  body_weldid: np.ndarray
  body_jntnum: np.ndarray
  body_jntadr: np.ndarray
  body_sameframe: np.ndarray
  body_dofnum: np.ndarray
  body_dofadr: np.ndarray
  body_treeid: np.ndarray
  body_geomnum: np.ndarray
  body_geomadr: np.ndarray
  body_simple: np.ndarray
  body_pos: jax.Array
  body_quat: jax.Array
  body_ipos: jax.Array
  body_iquat: jax.Array
  body_mass: jax.Array
  body_subtreemass: jax.Array
  body_inertia: jax.Array
  body_gravcomp: jax.Array
  body_margin: np.ndarray
  body_contype: np.ndarray
  body_conaffinity: np.ndarray
  body_invweight0: jax.Array
  jnt_type: np.ndarray
  jnt_qposadr: np.ndarray
  jnt_dofadr: np.ndarray
  jnt_bodyid: np.ndarray
  jnt_limited: np.ndarray
  jnt_actfrclimited: np.ndarray
  jnt_actgravcomp: np.ndarray
  jnt_solref: jax.Array
  jnt_solimp: jax.Array
  jnt_pos: jax.Array
  jnt_axis: jax.Array
  jnt_stiffness: jax.Array
  jnt_stiffnesspoly: jax.Array
  jnt_range: jax.Array
  jnt_actfrcrange: jax.Array
  jnt_margin: jax.Array
  dof_bodyid: np.ndarray
  dof_jntid: np.ndarray
  dof_parentid: np.ndarray
  dof_treeid: np.ndarray
  dof_Madr: np.ndarray  # pylint:disable=invalid-name
  dof_simplenum: np.ndarray
  M_rowadr: np.ndarray  # pylint:disable=invalid-name
  M_rownnz: np.ndarray  # pylint:disable=invalid-name
  M_colind: np.ndarray  # pylint:disable=invalid-name
  dof_solref: jax.Array
  dof_solimp: jax.Array
  dof_frictionloss: jax.Array
  dof_armature: jax.Array
  dof_damping: jax.Array
  dof_dampingpoly: jax.Array
  dof_invweight0: jax.Array
  dof_M0: jax.Array  # pylint:disable=invalid-name
  geom_type: np.ndarray
  geom_contype: np.ndarray
  geom_conaffinity: np.ndarray
  geom_condim: np.ndarray
  geom_bodyid: np.ndarray
  geom_sameframe: np.ndarray
  geom_dataid: np.ndarray
  geom_group: np.ndarray
  geom_matid: jax.Array
  geom_priority: np.ndarray
  geom_solmix: jax.Array
  geom_solref: jax.Array
  geom_solimp: jax.Array
  geom_size: jax.Array
  geom_aabb: jax.Array
  geom_rbound: jax.Array
  geom_pos: jax.Array
  geom_quat: jax.Array
  geom_friction: jax.Array
  geom_margin: jax.Array
  geom_gap: jax.Array
  geom_fluid: np.ndarray
  geom_rgba: jax.Array
  site_type: np.ndarray
  site_bodyid: np.ndarray
  site_sameframe: np.ndarray
  site_size: np.ndarray
  site_pos: jax.Array
  site_quat: jax.Array
  cam_mode: np.ndarray
  cam_bodyid: np.ndarray
  cam_targetbodyid: np.ndarray
  cam_pos: jax.Array
  cam_quat: jax.Array
  cam_poscom0: jax.Array
  cam_pos0: jax.Array
  cam_mat0: jax.Array
  cam_fovy: jax.Array
  cam_resolution: np.ndarray
  cam_sensorsize: np.ndarray
  cam_intrinsic: jax.Array
  light_mode: np.ndarray
  light_type: jax.Array
  light_active: jax.Array
  light_castshadow: jax.Array
  light_pos: jax.Array
  light_dir: jax.Array
  light_poscom0: jax.Array
  light_pos0: jax.Array
  light_dir0: jax.Array
  light_cutoff: jax.Array
  light_ambient: jax.Array
  light_attenuation: jax.Array
  light_diffuse: jax.Array
  light_exponent: jax.Array
  light_specular: jax.Array
  mesh_vertadr: np.ndarray
  mesh_vertnum: np.ndarray
  mesh_faceadr: np.ndarray
  mesh_facenum: np.ndarray
  mesh_bvhadr: np.ndarray
  mesh_bvhnum: np.ndarray
  mesh_octadr: np.ndarray
  mesh_octnum: np.ndarray
  mesh_normaladr: np.ndarray
  mesh_normalnum: np.ndarray
  mesh_graphadr: np.ndarray
  mesh_vert: np.ndarray
  mesh_normal: np.ndarray
  mesh_face: np.ndarray
  mesh_graph: np.ndarray
  mesh_pos: np.ndarray
  mesh_quat: np.ndarray
  mesh_texcoordadr: np.ndarray
  mesh_texcoordnum: np.ndarray
  mesh_texcoord: np.ndarray
  flex_vertadr: np.ndarray
  flex_vertnum: np.ndarray
  flex_interp: np.ndarray
  flex_vert0: np.ndarray
  flex_nodeadr: np.ndarray
  flex_nodenum: np.ndarray
  flex_nodebodyid: np.ndarray
  flex_node0: np.ndarray
  hfield_size: np.ndarray
  hfield_nrow: np.ndarray
  hfield_ncol: np.ndarray
  hfield_adr: np.ndarray
  hfield_data: jax.Array
  tex_type: np.ndarray
  tex_height: np.ndarray
  tex_width: np.ndarray
  tex_nchannel: np.ndarray
  tex_adr: np.ndarray
  tex_data: np.ndarray
  mat_rgba: jax.Array
  mat_texid: jax.Array
  mat_emission: jax.Array
  mat_specular: jax.Array
  mat_shininess: jax.Array
  pair_dim: np.ndarray
  pair_geom1: np.ndarray
  pair_geom2: np.ndarray
  pair_signature: np.ndarray
  pair_solref: jax.Array
  pair_solreffriction: jax.Array
  pair_solimp: jax.Array
  pair_margin: jax.Array
  pair_gap: jax.Array
  pair_friction: jax.Array
  exclude_signature: np.ndarray
  eq_type: np.ndarray
  eq_obj1id: np.ndarray
  eq_obj2id: np.ndarray
  eq_objtype: np.ndarray
  eq_active0: np.ndarray
  eq_solref: jax.Array
  eq_solimp: jax.Array
  eq_data: jax.Array
  tendon_type: np.ndarray
  tendon_adr: np.ndarray
  tendon_num: np.ndarray
  tendon_limited: np.ndarray
  tendon_actfrclimited: np.ndarray
  tendon_solref_lim: jax.Array
  tendon_solimp_lim: jax.Array
  tendon_solref_fri: jax.Array
  tendon_solimp_fri: jax.Array
  tendon_range: jax.Array
  tendon_actfrcrange: jax.Array
  tendon_margin: jax.Array
  tendon_stiffness: jax.Array
  tendon_stiffnesspoly: jax.Array
  tendon_damping: jax.Array
  tendon_dampingpoly: jax.Array
  tendon_armature: jax.Array
  tendon_frictionloss: jax.Array
  tendon_lengthspring: jax.Array
  tendon_length0: jax.Array
  tendon_invweight0: jax.Array
  wrap_type: np.ndarray
  wrap_objid: np.ndarray
  wrap_prm: np.ndarray
  actuator_trntype: np.ndarray
  actuator_dyntype: np.ndarray
  actuator_gaintype: np.ndarray
  actuator_biastype: np.ndarray
  actuator_trnid: np.ndarray
  actuator_actadr: np.ndarray
  actuator_actnum: np.ndarray
  actuator_group: np.ndarray
  actuator_ctrllimited: np.ndarray
  actuator_forcelimited: np.ndarray
  actuator_actlimited: np.ndarray
  actuator_dynprm: jax.Array
  actuator_gainprm: jax.Array
  actuator_biasprm: jax.Array
  actuator_actearly: np.ndarray
  actuator_ctrlrange: jax.Array
  actuator_forcerange: jax.Array
  actuator_actrange: jax.Array
  actuator_gear: jax.Array
  actuator_cranklength: jax.Array
  actuator_acc0: jax.Array
  actuator_lengthrange: jax.Array
  sensor_type: np.ndarray
  sensor_datatype: np.ndarray
  sensor_needstage: np.ndarray
  sensor_objtype: np.ndarray
  sensor_objid: np.ndarray
  sensor_reftype: np.ndarray
  sensor_refid: np.ndarray
  sensor_intprm: np.ndarray
  sensor_dim: np.ndarray
  sensor_adr: np.ndarray
  sensor_cutoff: np.ndarray
  numeric_adr: np.ndarray
  numeric_data: np.ndarray
  tuple_adr: np.ndarray
  tuple_size: np.ndarray
  tuple_objtype: np.ndarray
  tuple_objid: np.ndarray
  tuple_objprm: np.ndarray
  key_time: np.ndarray
  key_qpos: np.ndarray
  key_qvel: np.ndarray
  key_act: np.ndarray
  key_mpos: np.ndarray
  key_mquat: np.ndarray
  key_ctrl: np.ndarray
  key_plastic: np.ndarray
  name_bodyadr: np.ndarray
  name_jntadr: np.ndarray
  name_geomadr: np.ndarray
  name_siteadr: np.ndarray
  name_camadr: np.ndarray
  name_flexadr: np.ndarray
  name_meshadr: np.ndarray
  name_hfieldadr: np.ndarray
  name_pairadr: np.ndarray
  name_eqadr: np.ndarray
  name_tendonadr: np.ndarray
  name_actuatoradr: np.ndarray
  name_sensoradr: np.ndarray
  name_numericadr: np.ndarray
  name_tupleadr: np.ndarray
  name_keyadr: np.ndarray
  names: bytes
  signature: np.uint64
  _sizes: jax.Array
  _impl: Union[ModelJAX, mjxw_types.ModelWarp]

  @property
  def impl(self) -> Impl:
    return {
        ModelCPP: Impl.CPP,
        ModelJAX: Impl.JAX,
        mjxw_types.ModelWarp: Impl.WARP,
    }[type(self._impl)]

  def __getattr__(self, name: str):
    if name == 'value':
      # Special case for NNX, the value attribute may not exist on the parent
      # PyTreeNode, before it exists on the child PyTreeNode. Thanks NNX.
      return object.__getattribute__(self, 'value')

    try:
      impl_instance = object.__getattribute__(self, '_impl')
      val = getattr(impl_instance, name)
      warnings.warn(
          f'Accessing `{name}` directly from `Model` is deprecated. '
          f'Access it via `model._impl.{name}` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
    except AttributeError:
      # raise the standard exception
      raise AttributeError(  # pylint: disable=raise-missing-from
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    return val


class Contact(PyTreeNode):
  """Result of collision detection functions.

  Attributes:
    dist: distance between nearest points; neg: penetration
    pos: position of contact point: midpoint between geoms            (3,)
    frame: normal is in [0-2]                                         (9,)
    includemargin: include if dist<includemargin=margin               (1,)
    friction: tangent1, 2, spin, roll1, 2                             (5,)
    solref: constraint solver reference, normal direction             (mjNREF,)
    solreffriction: constraint solver reference, friction directions  (mjNREF,)
    solimp: constraint solver impedance                               (mjNIMP,)
    dim: contact space dimensionality: 1, 3, 4, or 6
    geom1: id of geom 1; deprecated, use geom[0]
    geom2: id of geom 2; deprecated, use geom[1]
    geom: geom ids                                                    (2,)
    efc_address: address in efc; -1: not included
  """  # fmt: skip
  dist: jax.Array
  pos: jax.Array
  frame: jax.Array
  includemargin: jax.Array
  friction: jax.Array
  solref: jax.Array
  solreffriction: jax.Array
  solimp: jax.Array
  # unsupported: mu, H (calculated locally in solver.py)
  dim: np.ndarray
  geom1: jax.Array
  geom2: jax.Array
  geom: jax.Array
  # unsupported: flex, elem, vert, exclude
  efc_address: np.ndarray


class DataJAX(PyTreeNode):
  """JAX-specific data."""

  ne: int
  nf: int
  nl: int
  nefc: int
  ncon: int
  solver_niter: jax.Array
  cinert: jax.Array
  ten_wrapadr: jax.Array
  ten_wrapnum: jax.Array
  ten_J: jax.Array  # pylint:disable=invalid-name
  wrap_obj: jax.Array
  wrap_xpos: jax.Array
  actuator_moment: jax.Array
  crb: jax.Array
  M: jax.Array  # pylint:disable=invalid-name
  qLD: jax.Array  # pylint:disable=invalid-name
  qLDiagInv: jax.Array  # pylint:disable=invalid-name
  ten_velocity: jax.Array
  actuator_velocity: jax.Array

  cacc: jax.Array
  cfrc_int: jax.Array
  cfrc_ext: jax.Array
  subtree_linvel: jax.Array
  subtree_angmom: jax.Array
  # dynamically sized data which are made static due to JAX limitations
  contact: Contact
  efc_type: jax.Array
  efc_J: jax.Array  # pylint:disable=invalid-name
  efc_pos: jax.Array
  efc_margin: jax.Array
  efc_frictionloss: jax.Array
  efc_D: jax.Array  # pylint:disable=invalid-name
  efc_aref: jax.Array
  efc_force: jax.Array


class Data(PyTreeNode):
  """Dynamic state that updates each step.

  Attributes:
    time: simulation time
    qpos: position
    qvel: velocity
    act: actuator activation
    history: actuator history buffer
    qacc_warmstart: warm start for solver
    plugin_state: plugin state values
    plastic: plastic spring reference values
    ctrl: control input
    qfrc_applied: applied generalized force
    xfrc_applied: applied Cartesian force/torque
    eq_active: enable/disable equality constraints
    mocap_pos: positions of mocap bodies
    mocap_quat: orientations of mocap bodies
    qacc: acceleration
    act_dot: time-derivative of actuator activation
    userdata: user data
    sensordata: sensor data output
    xpos: Cartesian position of body frame
    xquat: Cartesian orientation of body frame
    xmat: rotation matrix of body frame
    xipos: Cartesian position of body com
    ximat: rotation matrix of body inertia
    xanchor: Cartesian position of joint anchor
    xaxis: Cartesian joint axis
    ten_length: tendon lengths
    geom_xpos: Cartesian position of geoms
    geom_xmat: rotation matrix of geoms
    site_xpos: Cartesian position of sites
    site_xmat: rotation matrix of sites
    cam_xpos: camera positions
    cam_xmat: camera rotation matrices
    subtree_com: com of each subtree
    cvel: center of mass based velocity
    cdof: center of mass based jacobian
    cdof_dot: time-derivative of cdof
    qfrc_bias: C(qpos,qvel)
    qfrc_gravcomp: gravity compensation term
    qfrc_fluid: fluid drag and buoyancy forces
    qfrc_passive: passive force
    qfrc_actuator: actuator force
    actuator_force: actuator force in actuation space
    actuator_length: actuator lengths
    qfrc_smooth: smooth dynamics force
    qacc_smooth: acceleration without constraints
    qfrc_constraint: constraint force
    qfrc_inverse: net external force for inverse dynamics
  """

  # global properties:
  time: jax.Array
  # state:
  qpos: jax.Array
  qvel: jax.Array
  act: jax.Array
  history: jax.Array
  qacc_warmstart: jax.Array
  plugin_state: jax.Array
  plastic: jax.Array
  # control:
  ctrl: jax.Array
  qfrc_applied: jax.Array
  xfrc_applied: jax.Array
  eq_active: jax.Array
  # mocap data:
  mocap_pos: jax.Array
  mocap_quat: jax.Array
  # dynamics:
  qacc: jax.Array
  act_dot: jax.Array
  # user data:
  userdata: jax.Array
  sensordata: jax.Array
  # position dependent:
  xpos: jax.Array
  xquat: jax.Array
  xmat: jax.Array
  xipos: jax.Array
  ximat: jax.Array
  xanchor: jax.Array
  xaxis: jax.Array
  ten_length: jax.Array
  geom_xpos: jax.Array
  geom_xmat: jax.Array
  site_xpos: jax.Array
  site_xmat: jax.Array
  cam_xpos: jax.Array
  cam_xmat: jax.Array
  subtree_com: jax.Array
  cvel: jax.Array
  cdof: jax.Array
  cdof_dot: jax.Array
  qfrc_bias: jax.Array
  qfrc_gravcomp: jax.Array
  qfrc_fluid: jax.Array
  qfrc_passive: jax.Array
  qfrc_actuator: jax.Array
  actuator_force: jax.Array
  actuator_length: jax.Array
  qfrc_smooth: jax.Array
  qacc_smooth: jax.Array
  qfrc_constraint: jax.Array
  qfrc_inverse: jax.Array
  _impl: Union[DataCPP, DataJAX, mjxw_types.DataWarp]

  @property
  def impl(self) -> Impl:
    return {
        DataCPP: Impl.CPP,
        DataJAX: Impl.JAX,
        mjxw_types.DataWarp: Impl.WARP,
    }[type(self._impl)]

  def __getattr__(self, name: str):
    try:
      impl_instance = object.__getattribute__(self, '_impl')
      val = getattr(impl_instance, name)
      warnings.warn(
          f'Accessing `{name}` directly from `Data` is deprecated. '
          f'Access it via `data._impl.{name}` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
    except AttributeError:
      # raise the standard exception
      raise AttributeError(  # pylint: disable=raise-missing-from
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    return val

  def __getitem__(self, key):
    if self.impl == Impl.WARP:
      return jax.tree.map_with_path(
          lambda path, x, k=key: x[k]
          if tree_path_to_attr_str(path) not in mjxw_types.DATA_NON_VMAP
          else x,
          self,
      )

    return jax.tree.map(lambda x: x[key], self)

  def where(self, done: jax.Array, other: 'Data') -> 'Data':
    """Selectively merge self and other based on done.

    Args:
      done: Boolean array (or scalar inside vmap) indicating reset status.
      other: Data object to select when done is True.

    Returns:
      Merged Data object.
    """
    if self.impl != Impl.JAX and self.impl != Impl.WARP:
      raise NotImplementedError(
          'where is only supported for JAX and WARP implementations.'
      )

    if self.impl == Impl.JAX:
      return jax.tree.map(
          lambda x, y: jax.numpy.where(done, x, y), other, self
      )

    # Warp impl:
    def merge_leaf(path, r_val, s_val):
      field_name = tree_path_to_attr_str(path)
      is_batched = mjxw_types._BATCH_DIM['Data'].get(field_name, True)

      if is_batched:
        return jax.numpy.where(done, r_val, s_val)
      else:
        return s_val

    return jax.tree_util.tree_map_with_path(merge_leaf, other, self)


def tree_path_to_attr_str(path: jax.tree_util.KeyPath) -> str:
  """Converts a tree path to a dataclass attribute string."""
  if not isinstance(path, tuple):
    raise NotImplementedError(
        f'Parsing for jax tree path {path} not implemented.'
    )

  if any(isinstance(p, jax.tree_util.SequenceKey) for p in path):
    # get the path up to the first sequence key, we assume variadic sequences
    is_seq_key = [isinstance(p, jax.tree_util.SequenceKey) for p in path]
    path = path[: is_seq_key.index(True)]

  assert all(isinstance(p, jax.tree_util.GetAttrKey) for p in path)
  path = [p for p in path if p.name != '_impl']  # pyrefly: ignore[bad-assignment]
  return '__'.join(p.name for p in path)

