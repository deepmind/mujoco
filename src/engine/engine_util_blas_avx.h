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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_BLAS_AVX_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_BLAS_AVX_H_

#ifdef mjUSEPLATFORMSIMD
#if defined(__AVX__) && !defined(mjUSESINGLE)

#ifndef mjUSEAVX
#define mjUSEAVX
#endif

#include <immintrin.h>
#include <math.h>

#include <mujoco/mjtype.h>

// res = vec*scl
static inline
void mju_scl_avx(mjtNum* res, const mjtNum* vec, mjtNum scl, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    __m256d sclpar = _mm256_set1_pd(scl);
    while (i <= n_4) {
      _mm256_storeu_pd(res+i, _mm256_mul_pd(_mm256_loadu_pd(vec+i), sclpar));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] = vec[i+0]*scl;
    res[i+1] = vec[i+1]*scl;
    res[i+2] = vec[i+2]*scl;
  } else if (n_i == 2) {
    res[i+0] = vec[i+0]*scl;
    res[i+1] = vec[i+1]*scl;
  } else if (n_i == 1) {
    res[i] = vec[i]*scl;
  }
}


// res = vec1 + vec2
static inline
void mju_add_avx(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    while (i <= n_4) {
      _mm256_storeu_pd(res+i, _mm256_add_pd(_mm256_loadu_pd(vec1+i), _mm256_loadu_pd(vec2+i)));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] = vec1[i+0] + vec2[i+0];
    res[i+1] = vec1[i+1] + vec2[i+1];
    res[i+2] = vec1[i+2] + vec2[i+2];
  } else if (n_i == 2) {
    res[i+0] = vec1[i+0] + vec2[i+0];
    res[i+1] = vec1[i+1] + vec2[i+1];
  } else if (n_i == 1) {
    res[i] = vec1[i] + vec2[i];
  }
}


// res = vec1 - vec2
static inline
void mju_sub_avx(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    while (i <= n_4) {
      _mm256_storeu_pd(res+i, _mm256_sub_pd(_mm256_loadu_pd(vec1+i), _mm256_loadu_pd(vec2+i)));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] = vec1[i+0] - vec2[i+0];
    res[i+1] = vec1[i+1] - vec2[i+1];
    res[i+2] = vec1[i+2] - vec2[i+2];
  } else if (n_i == 2) {
    res[i+0] = vec1[i+0] - vec2[i+0];
    res[i+1] = vec1[i+1] - vec2[i+1];
  } else if (n_i == 1) {
    res[i] = vec1[i] - vec2[i];
  }
}


// res += vec
static inline
void mju_addTo_avx(mjtNum* res, const mjtNum* vec, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    while (i <= n_4) {
      _mm256_storeu_pd(res+i, _mm256_add_pd(_mm256_loadu_pd(res+i), _mm256_loadu_pd(vec+i)));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] += vec[i+0];
    res[i+1] += vec[i+1];
    res[i+2] += vec[i+2];
  } else if (n_i == 2) {
    res[i+0] += vec[i+0];
    res[i+1] += vec[i+1];
  } else if (n_i == 1) {
    res[i] += vec[i];
  }
}


// res -= vec
static inline
void mju_subFrom_avx(mjtNum* res, const mjtNum* vec, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    while (i <= n_4) {
      _mm256_storeu_pd(res+i, _mm256_sub_pd(_mm256_loadu_pd(res+i), _mm256_loadu_pd(vec+i)));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] -= vec[i+0];
    res[i+1] -= vec[i+1];
    res[i+2] -= vec[i+2];
  } else if (n_i == 2) {
    res[i+0] -= vec[i+0];
    res[i+1] -= vec[i+1];
  } else if (n_i == 1) {
    res[i] -= vec[i];
  }
}


// res += vec*scl
static inline
void mju_addToScl_avx(mjtNum* res, const mjtNum* vec, mjtNum scl, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    __m256d sclpar = _mm256_set1_pd(scl);
    while (i <= n_4) {
      __m256d val1 = _mm256_loadu_pd(res+i);
      __m256d val2 = _mm256_loadu_pd(vec+i);
      _mm256_storeu_pd(res+i, _mm256_add_pd(val1, _mm256_mul_pd(val2, sclpar)));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] += vec[i+0]*scl;
    res[i+1] += vec[i+1]*scl;
    res[i+2] += vec[i+2]*scl;
  } else if (n_i == 2) {
    res[i+0] += vec[i+0]*scl;
    res[i+1] += vec[i+1]*scl;
  } else if (n_i == 1) {
    res[i] += vec[i]*scl;
  }
}


// res = vec1 + vec2*scl
static inline
void mju_addScl_avx(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, mjtNum scl, int n) {
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    __m256d sclpar = _mm256_set1_pd(scl);
    while (i <= n_4) {
      __m256d val1 = _mm256_loadu_pd(vec1+i);
      __m256d val2 = _mm256_loadu_pd(vec2+i);
      _mm256_storeu_pd(res+i, _mm256_add_pd(val1, _mm256_mul_pd(val2, sclpar)));
      i += 4;
    }
  }

  int n_i = n - i;
  if (n_i == 3) {
    res[i+0] = vec1[i+0] + vec2[i+0]*scl;
    res[i+1] = vec1[i+1] + vec2[i+1]*scl;
    res[i+2] = vec1[i+2] + vec2[i+2]*scl;
  } else if (n_i == 2) {
    res[i+0] = vec1[i+0] + vec2[i+0]*scl;
    res[i+1] = vec1[i+1] + vec2[i+1]*scl;
  } else if (n_i == 1) {
    res[i] = vec1[i] + vec2[i]*scl;
  }
}


// vector dot-product
static inline
mjtNum mju_dot_avx(const mjtNum* vec1, const mjtNum* vec2, int n) {
  mjtNum res = 0;
  int i = 0;
  int n_4 = n - 4;
  if (n_4 >= 0) {
    __m256d sum = _mm256_mul_pd(_mm256_loadu_pd(vec1), _mm256_loadu_pd(vec2));
    i = 4;

    while (i <= n_4) {
      sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(vec1+i), _mm256_loadu_pd(vec2+i)));
      i += 4;
    }

    __m128d vlow = _mm256_castpd256_pd128(sum);
    __m128d vhigh = _mm256_extractf128_pd(sum, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    res = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
  }

  int n_i = n - i;
  if (n_i == 3) {
    res += vec1[i+0]*vec2[i+0] + vec1[i+1]*vec2[i+1] + vec1[i+2]*vec2[i+2];
  } else if (n_i == 2) {
    res += vec1[i+0]*vec2[i+0] + vec1[i+1]*vec2[i+1];
  } else if (n_i == 1) {
    res += vec1[i]*vec2[i];
  }

  return res;
}


// res = data (4-vector)
static inline
void mju_copy4_avx(mjtNum res[4], const mjtNum data[4]) {
  _mm256_storeu_pd(res, _mm256_loadu_pd(data));
}


// normalize 4-vector, return length before normalization
static inline
mjtNum mju_normalize4_avx(mjtNum vec[4]) {
  mjtNum v0 = vec[0];
  mjtNum v1 = vec[1];
  mjtNum v2 = vec[2];
  mjtNum v3 = vec[3];
  mjtNum sum = (v0*v0 + v1*v1) + (v2*v2 + v3*v3);
  mjtNum norm = _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(sum)));

  if (norm < mjMINVAL) {
    _mm256_storeu_pd(vec, _mm256_setr_pd(1.0, 0.0, 0.0, 0.0));
  } else if (fabs(norm - 1.0) > mjMINVAL) {
    __m256d v = _mm256_setr_pd(v0, v1, v2, v3);
    _mm256_storeu_pd(vec, _mm256_mul_pd(v, _mm256_set1_pd(1.0 / norm)));
  }

  return norm;
}

#endif  // defined(__AVX__) && !defined(mjUSESINGLE)
#endif  // mjUSEPLATFORMSIMD

#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_BLAS_AVX_H_
