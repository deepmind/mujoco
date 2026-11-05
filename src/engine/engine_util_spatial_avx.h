// Copyright 2026 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPATIAL_AVX_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPATIAL_AVX_H_

#ifdef mjUSEPLATFORMSIMD
#if defined(__AVX__) && !defined(mjUSESINGLE)

#ifndef mjUSEAVX
#define mjUSEAVX
#endif

#include <immintrin.h>

#include <mujoco/mjtype.h>

// negate quaternion
static inline
void mju_negQuat_avx(mjtNum res[4], const mjtNum quat[4]) {
  __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, -0.0, -0.0);
  _mm256_storeu_pd(res, _mm256_xor_pd(_mm256_loadu_pd(quat), sign_mask));
}


// multiply quaternions
static inline
void mju_mulQuat_avx(mjtNum res[4], const mjtNum qa[4], const mjtNum qb[4]) {
  __m256d vb = _mm256_loadu_pd(qb);
  __m256d va0 = _mm256_set1_pd(qa[0]);
  __m256d va1 = _mm256_set1_pd(qa[1]);
  __m256d va2 = _mm256_set1_pd(qa[2]);
  __m256d va3 = _mm256_set1_pd(qa[3]);

  __m256d vb1 = _mm256_permute_pd(vb, 0x05);
  __m256d vb2 = _mm256_permute2f128_pd(vb, vb, 0x01);
  __m256d vb3 = _mm256_permute_pd(vb2, 0x05);

  __m256d va2s = _mm256_xor_pd(va2, _mm256_setr_pd(-0.0, 0.0, 0.0, -0.0));
  __m256d va3s = _mm256_xor_pd(va3, _mm256_setr_pd(-0.0, -0.0, 0.0, 0.0));

  __m256d s01 = _mm256_addsub_pd(_mm256_mul_pd(va0, vb), _mm256_mul_pd(va1, vb1));
  __m256d s23 = _mm256_add_pd(_mm256_mul_pd(va2s, vb2), _mm256_mul_pd(va3s, vb3));
  _mm256_storeu_pd(res, _mm256_add_pd(s01, s23));
}


// multiply quaternion and axis
static inline
void mju_mulQuatAxis_avx(mjtNum res[4], const mjtNum quat[4], const mjtNum axis[3]) {
  __m256d vq = _mm256_loadu_pd(quat);
  __m256d vx0 = _mm256_set1_pd(axis[0]);
  __m256d vx1 = _mm256_set1_pd(axis[1]);
  __m256d vx2 = _mm256_set1_pd(axis[2]);

  __m256d vq_swap = _mm256_permute2f128_pd(vq, vq, 0x01);
  __m256d c0 = _mm256_xor_pd(_mm256_permute_pd(vq, 0x05),
                             _mm256_setr_pd(-0.0, 0.0, 0.0, -0.0));
  __m256d c1 = _mm256_xor_pd(vq_swap, _mm256_setr_pd(-0.0, -0.0, 0.0, 0.0));
  __m256d c2 = _mm256_xor_pd(_mm256_permute_pd(vq_swap, 0x05),
                             _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0));

  __m256d s01 = _mm256_add_pd(_mm256_mul_pd(c0, vx0), _mm256_mul_pd(c1, vx1));
  _mm256_storeu_pd(res, _mm256_add_pd(s01, _mm256_mul_pd(c2, vx2)));
}


// convert quaternion to 3D rotation matrix
static inline
void mju_quat2Mat_avx(mjtNum res[9], const mjtNum quat[4]) {
  // null quat: identity
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    _mm256_storeu_pd(res + 0, _mm256_setr_pd(1.0, 0.0, 0.0, 0.0));
    _mm256_storeu_pd(res + 4, _mm256_setr_pd(1.0, 0.0, 0.0, 0.0));
    res[8] = 1.0;
    return;
  }

  __m256d vq = _mm256_loadu_pd(quat);
  __m256d vq0 = _mm256_set1_pd(quat[0]);
  __m256d vq1 = _mm256_set1_pd(quat[1]);
  __m256d vq3 = _mm256_set1_pd(quat[3]);

  __m128d vq_lo = _mm256_castpd256_pd128(vq);
  __m128d vq_hi = _mm256_extractf128_pd(vq, 1);

  // diagonals: (q00 - q22) +/- (q11 - q33), (q00 + q22) - (q11 + q33)
  __m256d qsq = _mm256_mul_pd(vq, vq);
  __m128d qsq_lo = _mm256_castpd256_pd128(qsq);
  __m128d qsq_hi = _mm256_extractf128_pd(qsq, 1);
  __m128d d_diff = _mm_sub_pd(qsq_lo, qsq_hi);
  __m128d d_sum = _mm_add_pd(qsq_lo, qsq_hi);
  __m128d d_diff_hi = _mm_unpackhi_pd(d_diff, d_diff);
  __m128d diag0 = _mm_add_sd(d_diff, d_diff_hi);
  __m128d diag8 = _mm_sub_sd(d_diff, d_diff_hi);
  __m128d diag4 = _mm_sub_sd(d_sum, _mm_unpackhi_pd(d_sum, d_sum));

  // res[1..3]: 2*(q12 - q03), 2*(q13 + q02), 2*(q12 + q03)
  __m256d v_2323 = _mm256_insertf128_pd(_mm256_castpd128_pd256(vq_hi), vq_hi, 1);
  __m256d u0 = _mm256_permute_pd(v_2323, 0x04);
  __m256d w0 = _mm256_permute_pd(v_2323, 0x0B);
  __m256d s0 = _mm256_setr_pd(0.0, -0.0, 0.0, 0.0);
  __m256d h0 = _mm256_add_pd(_mm256_mul_pd(vq1, u0),
                             _mm256_mul_pd(_mm256_xor_pd(vq0, s0), w0));
  __m256d off0 = _mm256_add_pd(h0, h0);

  // res[5..7]: 2*(q23 - q01), 2*(q13 - q02), 2*(q23 + q01)
  __m128d v_12 = _mm_shuffle_pd(vq_lo, vq_hi, 0x01);
  __m256d u1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(v_12), v_12, 1);
  __m256d w1 = _mm256_permute_pd(u1, 0x05);
  __m256d s1 = _mm256_setr_pd(0.0, -0.0, -0.0, 0.0);
  __m256d h1 = _mm256_add_pd(_mm256_mul_pd(vq3, u1),
                             _mm256_mul_pd(_mm256_xor_pd(vq0, s1), w1));
  __m256d off1 = _mm256_add_pd(h1, h1);

  _mm256_storeu_pd(res + 0, _mm256_blend_pd(off0, _mm256_castpd128_pd256(diag0), 0x01));
  _mm256_storeu_pd(res + 4, _mm256_blend_pd(off1, _mm256_castpd128_pd256(diag4), 0x01));
  _mm_store_sd(res + 8, diag8);
}

#endif  // defined(__AVX__) && !defined(mjUSESINGLE)
#endif  // mjUSEPLATFORMSIMD

#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPATIAL_AVX_H_
