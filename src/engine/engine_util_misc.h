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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_MISC_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_MISC_H_

#include <mujoco/mjexport.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

//------------------------------ tendons and actuators ---------------------------------------------

// wrap tendons around spheres and cylinders
MJAPI mjtNum mju_wrap(mjtNum wpnt[6], const mjtNum x0[3], const mjtNum x1[3], const mjtNum xpos[3],
                      const mjtNum xmat[9], mjtNum radius, int type, const mjtNum side[3]);

// normalized muscle length-gain curve
MJAPI mjtNum mju_muscleGainLength(mjtNum length, mjtNum lmin, mjtNum lmax);

// muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)
MJAPI mjtNum mju_muscleGain(mjtNum len, mjtNum vel, const mjtNum lengthrange[2],
                            mjtNum acc0, const mjtNum prm[9]);

// muscle passive force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)
MJAPI mjtNum mju_muscleBias(mjtNum len, const mjtNum lengthrange[2],
                            mjtNum acc0, const mjtNum prm[9]);

// muscle time constant with optional smoothing
MJAPI mjtNum mju_muscleDynamicsTimescale(mjtNum dctrl, mjtNum tau_act, mjtNum tau_deact,
                                         mjtNum smoothing_width);

// muscle activation dynamics, prm = (tau_act, tau_deact, smoothing_width)
MJAPI mjtNum mju_muscleDynamics(mjtNum ctrl, mjtNum act, const mjtNum prm[3]);

// LuGre Stribeck function: g(v) = F_C + (F_S - F_C) * exp(-(v/v_S)^2)
mjtNum mj_lugreStribeck(mjtNum velocity, mjtNum F_C, mjtNum F_S, mjtNum v_S);

// DC motor activation slot indices (-1 = slot not active)
typedef struct {
  int slew;         // slew rate state
  int integral;     // integral state
  int temperature;  // temperature state
  int bristle;      // LuGre bristle state
  int current;      // current state
  int num_slots;    // number of DC motor states
} mjDCMotorSlots;

// compute activation slot indices for a DC motor actuator
// dynprm = actuator_dynprm row, gainprm = actuator_gainprm row
mjDCMotorSlots mj_dcmotorSlots(const mjtNum* dynprm, const mjtNum* gainprm);

// all 3 semi-axes of a geom
MJAPI void mju_geomSemiAxes(mjtNum semiaxes[3], const mjtNum size[3], mjtGeom type);

// return 1 if point is inside a primitive geom, 0 otherwise
int mju_insidePrimitive(const mjtNum pos[3], const mjtNum mat[9], const mjtNum size[3],
                        mjtGeom type, const mjtNum point[3]);

// compute ray origin and direction for pixel (col, row) in camera image
// directions are normalized so ray functions return actual 3D distance
void mju_camPixelRay(mjtNum origin[3], mjtNum direction[3],
                     const mjtNum cam_xpos[3], const mjtNum cam_xmat[9],
                     int col, int row, mjtNum fx, mjtNum fy, mjtNum cx, mjtNum cy,
                     int projection, mjtNum ortho_extent);

// ----------------------------- Flex interpolation ------------------------------------------------

// evaluate the deformation gradient at p using the nodal dof values
MJAPI void mju_defGradient(mjtNum res[9], const mjtNum p[3], const mjtNum* dof, int order);

// evaluate the basis function at x for the i-th node
MJAPI mjtNum mju_evalBasis(const mjtNum x[3], int i, int order);

// evaluate the basis functions at x for all nodes in the cell
MJAPI void mju_evalBasisArray(mjtNum* basis, const mjtNum x[3], int order);

// map global parametric coord to cell-local coord and build node indices
MJAPI int mju_cellLookup(const mjtNum coord[3], const int cellnum[3], int order,
                         int cell, mjtNum local[3], int* nodeindices);

// interpolate a function at x with given interpolation coefficients and order n
MJAPI void mju_interpolate3D(mjtNum res[3], const mjtNum x[3], const mjtNum* coeff, int order,
                             const int* nodeindices);

// gather cell-local quantities and optionally compute rotation
MJAPI void mju_flexGatherCellState(int order, int cy, int cz, int ci, int cj, int ck,
                                   const mjtNum* xpos_g, const mjtNum* vel_g,
                                   const mjtNum* xpos0_g, mjtNum* xpos_c, mjtNum* vel_c,
                                   mjtNum* xpos0_c, int* nodeindices, mjtNum* quat);

// gather face-element-local quantities and optionally compute rotation (shell mode)
MJAPI void mju_flexGatherFaceState(int order, int cx, int cy, int cz,
                                   int face_elem_idx,
                                   const mjtNum* xpos_g, const mjtNum* vel_g,
                                   const mjtNum* xpos0_g,
                                   mjtNum* xpos_f, mjtNum* vel_f, mjtNum* xpos0_f,
                                   int* nodeindices, mjtNum* quat);

// compute corotational rotation from 2D deformation gradient on a flat face
MJAPI void mju_flexInterpRotation2D(int order, const mjtNum* xpos_f, int npe,
                                    int axis0, int axis1, int normal_axis,
                                    const mjtNum local[2], mjtNum* quat);

// compute unnormalized surface normal and tangent vectors at a parametric point
// on a 2D face element; normal = t1 x t2 (unnormalized)
MJAPI void mju_flexFaceNormal2D(mjtNum normal[3], mjtNum t1[3], mjtNum t2[3],
                                int order, const mjtNum* xpos_f,
                                const mjtNum local[2]);


// 1D shape function: order 1 (linear) or 2 (quadratic), node index i
static inline mjtNum mju_flexPhi(mjtNum s, int i, int order) {
  if (order == 1) return i == 0 ? 1 - s : s;
  switch (i) {
    case 0: return 2*s*s - 3*s + 1;
    case 1: return 4*(s - s*s);
    case 2: return 2*s*s - s;
    default: return 0;
  }
}

// 1D shape function gradient
static inline mjtNum mju_flexDphi(mjtNum s, int i, int order) {
  if (order == 1) return i == 0 ? -1 : 1;
  switch (i) {
    case 0: return 4*s - 3;
    case 1: return 4*(1 - 2*s);
    case 2: return 4*s - 1;
    default: return 0;
  }
}
// reconstruct interior node positions from boundary nodes via Transfinite Interpolation
MJAPI void mju_shellTrackInterior(mjtNum* nodexpos, int nx, int ny, int nz);

// compute TFI weights for an interior node (i,j,k) and distribute to boundary nodes
MJAPI void mju_shellTFIWeights(int nx, int ny, int nz, int i, int j, int k,
                               mjtNum w, int* nb, int* body, mjtNum* bweight,
                               const int* nodebodyid, int nstart);


// ----------------------------- Base64 ------------------------------------------------------------

// encode data as Base64 into buf (including padding and null char)
// returns number of chars written in buf: 4 * [(ndata + 2) / 3] + 1
MJAPI size_t mju_encodeBase64(char* buf, const uint8_t* data, size_t ndata);

// return size in decoded bytes if s is a valid Base64 encoding
// return 0 if s is empty or invalid Base64 encoding
MJAPI size_t mju_isValidBase64(const char* s);

// decode valid Base64 in string s into buf, undefined behavior if s is not valid Base64
// returns number of bytes decoded (upper limit of 3 * (strlen(s) / 4))
MJAPI size_t mju_decodeBase64(uint8_t* buf, const char* s);

//------------------------------ history buffers ---------------------------------------------------

// buffer layout: [user(1), cursor(1), times(n), values(n*dim)]
// - user: 1 mjtNum reserved for user data (ignored by these functions)
// - cursor: 1 mjtNum for circular buffer index (integer stored as mjtNum)
// - times: n timestamps, contiguous at buf[2..n+1]
// - values: n*dim values, contiguous at buf[n+2..n+2+n*dim-1]
// total buffer size: 2 + n*(1 + dim)

// initialize history buffer with given times and values; times must be strictly increasing
// values is size n x dim
MJAPI void mju_historyInit(mjtNum* buf, int n, int dim, const mjtNum* times,
                           const mjtNum* values, mjtNum user);

// find insertion slot for sample at time t, maintaining sorted order
// returns pointer to value slot (size dim) where caller should write
MJAPI mjtNum* mju_historyInsert(mjtNum* buf, int n, int dim, mjtNum t);

// read vector value at time t; interp: 0=zero-order-hold, 1=linear, 2=cubic spline
// returns pointer to sample in buffer on exact match (res untouched)
// returns NULL and writes interpolated result to res otherwise
MJAPI const mjtNum* mju_historyRead(const mjtNum* buf, int n, int dim,
                                    mjtNum* res, mjtNum t, int interp);

//------------------------------ miscellaneous -----------------------------------------------------

// convert contact force to pyramid representation
MJAPI void mju_encodePyramid(mjtNum* pyramid, const mjtNum* force,
                             const mjtNum* mu, int dim);

// convert pyramid representation to contact force
MJAPI void mju_decodePyramid(mjtNum* force, const mjtNum* pyramid,
                             const mjtNum* mu, int dim);

// integrate spring-damper analytically, return pos(dt)
MJAPI mjtNum mju_springDamper(mjtNum pos0, mjtNum vel0, mjtNum Kp, mjtNum Kv, mjtNum dt);

// return 1 if point is outside box given by pos, mat, size * inflate
// return -1 if point is inside box given by pos, mat, size / inflate
// return 0 if point is between the inflated and deflated boxes
MJAPI int mju_outsideBox(const mjtNum point[3], const mjtNum pos[3], const mjtNum mat[9],
                         const mjtNum size[3], mjtNum inflate);

// print matrix
MJAPI void mju_printMat(const mjtNum* mat, int nr, int nc);

// print sparse matrix to screen
MJAPI void mju_printMatSparse(const mjtNum* mat, int nr,
                              const int* rownnz, const int* rowadr,
                              const int* colind);

// min function, single evaluation of a and b
MJAPI mjtNum mju_min(mjtNum a, mjtNum b);

// max function, single evaluation of a and b
MJAPI mjtNum mju_max(mjtNum a, mjtNum b);

// clip x to the range [min, max]
MJAPI mjtNum mju_clip(mjtNum x, mjtNum min, mjtNum max);

// sign function
MJAPI mjtNum mju_sign(mjtNum x);

// round to nearest integer
MJAPI int mju_round(mjtNum x);

// convert type id (mjtObj) to type name
MJAPI const char* mju_type2Str(int type);

// convert type name to type id (mjtObj)
MJAPI int mju_str2Type(const char* str);

// return human readable number of bytes using standard letter suffix
MJAPI const char* mju_writeNumBytes(size_t nbytes);

// warning text
MJAPI const char* mju_warningText(int warning, size_t info);

// return 1 if nan or abs(x)>mjMAXVAL, 0 otherwise
MJAPI int mju_isBad(mjtNum x);

// return 1 if all elements are numerically 0 (-0.0 treated as zero)
MJAPI int mju_isZero(const mjtNum* vec, int n);

// return 1 if all elements are 0x00, faster than mju_isZero
MJAPI int mju_isZeroByte(const unsigned char* vec, int n);

// set integer vector to 0
MJAPI void mju_zeroInt(int* res, int n);

// copy int vector vec into res
MJAPI void mju_copyInt(int* res, const int* vec, int n);

// fill int vector with val
void mju_fillInt(int* res, int val, int n);

// standard normal random number generator (optional second number)
MJAPI mjtNum mju_standardNormal(mjtNum* num2);

// convert from float to mjtNum
MJAPI void mju_f2n(mjtNum* res, const float* vec, int n);

// convert from mjtNum to float
MJAPI void mju_n2f(float* res, const mjtNum* vec, int n);

// convert from double to mjtNum
MJAPI void mju_d2n(mjtNum* res, const double* vec, int n);

// convert from mjtNum to double
MJAPI void mju_n2d(double* res, const mjtNum* vec, int n);

// gather mjtNums: res[i] = vec[ind[i]], or copy if ind is NULL
MJAPI void mju_gather(mjtNum* res, const mjtNum* vec, const int* ind, int n);

// gather mjtNums, set to 0 at negative indices
MJAPI void mju_gatherMasked(mjtNum* res, const mjtNum* vec, const int* ind, int n);

// scatter mjtNums: res[ind[i]] = vec[i], or copy if ind is NULL
MJAPI void mju_scatter(mjtNum* res, const mjtNum* vec, const int* ind, int n);

// gather integers
MJAPI void mju_gatherInt(int* res, const int* vec, const int* ind, int n);

// scatter integers
MJAPI void mju_scatterInt(int* res, const int* vec, const int* ind, int n);

// build gather indices mapping src to res, assumes pattern(res) \subseteq pattern(src)
MJAPI void mju_sparseMap(int* map, int nr,
                         const int* res_rowadr, const int* res_rownnz, const int* res_colind,
                         const int* src_rowadr, const int* src_rownnz, const int* src_colind);

// build masked-gather map to copy a lower-triangular src into symmetric res
//  `cursor` is a preallocated buffer of size `nr`
MJAPI void mju_lower2SymMap(int* map, int nr,
                            const int* res_rowadr, const int* res_rownnz, const int* res_colind,
                            const int* src_rowadr, const int* src_rownnz, const int* src_colind,
                            int* cursor);

// insertion sort, increasing order
MJAPI void mju_insertionSort(mjtNum* list, int n);

// integer insertion sort, increasing order
MJAPI void mju_insertionSortInt(int* list, int n);

// Halton sequence
MJAPI mjtNum mju_Halton(int index, int base);

// call strncpy, then set dst[n-1] = 0
MJAPI char* mju_strncpy(char *dst, const char *src, int n);

// polynomial force coefficient: force = -mju_polyForce(...) * x
//   flg_odd=0: linear + poly[0]*x   + poly[1]*x^2 + ...
//   flg_odd=1: linear + poly[0]*|x| + poly[1]*x^2 + ...
MJAPI mjtNum mju_polyForce(mjtNum linear, const mjtNum* poly, mjtNum x, int n, int flg_odd);

// derivative of (mju_polyForce * x) w.r.t. x
MJAPI mjtNum mjd_xPolyForce(mjtNum linear, const mjtNum* poly, mjtNum x, int n, int flg_odd);

// potential energy: integral from 0 to x of mju_polyForce * t dt
MJAPI mjtNum mju_polyPotential(mjtNum linear, const mjtNum* poly, mjtNum x, int n, int flg_odd);

// sigmoid function over 0<=x<=1 using quintic polynomial
MJAPI mjtNum mju_sigmoid(mjtNum x);

#ifdef __cplusplus
}
#endif
#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_MISC_H_
