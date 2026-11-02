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

#include "engine/engine_print.h"

#include <inttypes.h>  // IWYU pragma: keep
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjtype.h>
#include <mujoco/mjxmacro.h>
#include "engine/engine_core_util.h"
#include "engine/engine_io.h"
#include "engine/engine_name.h"
#include "engine/engine_macro.h"
#include "engine/engine_support.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_vis_init.h"

#ifdef MEMORY_SANITIZER
  #include <sanitizer/msan_interface.h>
#endif

#define FLOAT_FORMAT "% -9.2g"
#define FLOAT_FORMAT_MAX_LEN 20
#define INT_FORMAT " %d"
#define SIZE_FORMAT " %" PRId64
#define NAME_FORMAT "%-21s"



//----------------------------------- static utility functions -------------------------------------

static void printInt(FILE* fp, const char* name, int value) {
  fprintf(fp, NAME_FORMAT, name);
  fprintf(fp, INT_FORMAT, value);
  fprintf(fp, "\n");
}

static void printStr(FILE* fp, const char* name, const char* value) {
  fprintf(fp, NAME_FORMAT, name);
  fprintf(fp, "%s", value ? value : "");
  fprintf(fp, "\n");
}

static void printNum(FILE* fp, const char* name, float value, const char* float_format) {
  fprintf(fp, NAME_FORMAT, name);
  fprintf(fp, float_format, value);
  fprintf(fp, "\n");
}

static void printArr(FILE* fp, const char* name, const float* data, int n, const char* float_format) {
  if (!data) {
    return;
  }
  fprintf(fp, NAME_FORMAT, name);
  for (int i = 0; i < n; ++i) {
    fprintf(fp, float_format, data[i]);
    fprintf(fp, " ");
  }
  fprintf(fp, "\n");
}

// print 2D array of mjtNum into file
static void printArray2d(const char* str, int nr, int nc, const mjtNum* data, FILE* fp,
                         const char* float_format) {
  if (!data) {
    return;
  }
  if (nr && nc) {
    fprintf(fp, "%s\n", str);
    for (int r=0; r < nr; r++) {
      fprintf(fp, " ");
      for (int c=0; c < nc; c++) {
        fprintf(fp, " ");
        fprintf(fp, float_format, data[c + r*nc]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
}


// print 2D array of int into file
static void printArray2dInt(const char* str, int nr, int nc, const int* data, FILE* fp) {
  if (!data) {
    return;
  }
  if (nr && nc) {
    fprintf(fp, "%s\n", str);
    for (int r=0; r < nr; r++) {
      fprintf(fp, " ");
      for (int c=0; c < nc; c++) {
        fprintf(fp, " ");
        fprintf(fp, "%d", data[c + r*nc]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
}


// print history buffer with semantic labels
static void printDelayBuffer(const char* name, const mjtNum* buf, int nhistory, int dim,
                             FILE* fp, const char* float_format) {
  if (!buf || nhistory <= 0) {
    return;
  }
  fprintf(fp, "  %s:\n", name);

  // user value (first slot)
  fprintf(fp, "    phase  = ");
  fprintf(fp, float_format, buf[0]);
  fprintf(fp, "\n");

  // cursor (second slot, stored as mjtNum but is an integer)
  fprintf(fp, "    cursor =  %d\n", (int)buf[1]);

  // timestamps
  const mjtNum* times = buf + 2;
  fprintf(fp, "    times  = ");
  for (int i = 0; i < nhistory; i++) {
    fprintf(fp, float_format, times[i]);
  }
  fprintf(fp, "\n");

  // values
  const mjtNum* values = times + nhistory;
  if (dim == 1) {
    fprintf(fp, "    values = ");
    for (int i = 0; i < nhistory; i++) {
      fprintf(fp, float_format, values[i]);
    }
    fprintf(fp, "\n");
  } else {
    fprintf(fp, "    values:\n");
    for (int i = 0; i < nhistory; i++) {
      fprintf(fp, "      [%d] =", i);
      for (int j = 0; j < dim; j++) {
        fprintf(fp, float_format, values[i*dim + j]);
      }
      fprintf(fp, "\n");
    }
  }
}


// print sparse matrix
static void printSparse(const char* str, const mjtNum* mat, int nr,
                        const int* rownnz, const int* rowadr,
                        const int* colind, FILE* fp, const char* float_format) {
  // if no data, or too many rows to be visually useful, return
  if (!mat || !nr || nr > 300) {
    return;
  }
  fprintf(fp, "%s\n", str);

  for (int r=0; r < nr; r++) {
    fprintf(fp, "  ");
    for (int adr=rowadr[r]; adr < rowadr[r]+rownnz[r]; adr++) {
      fprintf(fp, "  ");
      fprintf(fp, "%2d: ", colind[adr]);
      fprintf(fp, float_format, mat[adr]);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "\n");
}


// print block-diagonal dense matrix, embedded in a larger matrix
static void printBlockArray(const char* str, const mjtNum* data, int nc,
                            int nisland, const int* island_nr, const int* island_nc,
                            const int* island_r, const int* island_c,
                            const int* map_r, const int* map_c,
                            FILE* fp, const char* float_format) {
  if (!data || !nisland) {
    return;
  }

  fprintf(fp, "%s\n", str);

  // determine the width of the float format (already validated by validateFloatFormat)
  char dummy_buffer[100];
  int format_width = snprintf(dummy_buffer, sizeof(dummy_buffer), float_format, 0.0);

  for (int b = 0; b < nisland; b++) {
    int bnr = island_nr[b];
    int bnc = island_nc[b];
    int r_start = island_r[b];
    int c_start = island_c[b];

    // print rows for this block
    for (int r_block = 0; r_block < bnr; r_block++) {
      fprintf(fp, " ");
      // leading dots
      for (int c = 0; c < c_start; c++) {
        for (int i = 0; i < format_width; i++) fprintf(fp, ".");
        fprintf(fp, " ");
      }

      int row = map_r[r_start + r_block];

      // block data
      for (int c = 0; c < bnc; c++) {
        int col = map_c[c_start + c];
        fprintf(fp, " ");
        fprintf(fp, float_format, data[row * nc + col]);
      }

      // trailing dots
      for (int c = c_start + bnc; c < nc; c++) {
        for (int i = 0; i < format_width; i++) fprintf(fp, ".");
        fprintf(fp, " ");
      }
      fprintf(fp, "\n");
    }
  }

  fprintf(fp, "\n");
}



// print sparse matrix structure
void mj_printSparsity(const char* str, int nr, int nc, const int* rowadr, const int* diag,
                      const int* rownnz, const int* rowsuper, const int* colind, FILE* fp) {
  // if no rows / columns, or too many columns to be visually useful, return
  if (!nr || !nc || nc > 300) {
    return;
  }
  fprintf(fp, "%s\n", str);

  for (int c=0; c < nc+2; c++) fprintf(fp, "-");
  fprintf(fp, "\n ");

  for (int r=0; r < nr; r++) {
    int adr = rowadr[r];
    int nnz = 0;
    for (int c=0; c < nc; c++) {
      if (nnz < rownnz[r] && colind[adr + nnz] == c) {
        if (diag && diag[r] == nnz) {
          fprintf(fp, "D");
        } else {
          fprintf(fp, "x");
        }
        nnz++;
      } else {
        fprintf(fp, " ");
      }
    }
    fprintf(fp, " |");
    if (rowsuper && rowsuper[r] > 0) fprintf(fp, " %d", rowsuper[r]);
    fprintf(fp, "\n");
    if (r < nr-1) fprintf(fp, " ");
  }
  for (int c=0; c < nc+2; c++) fprintf(fp, "-");
  fprintf(fp, "\n\n");
}


// print block-diagonal sparse matrix structure
void mj_printBlockSparsity(const char* str, int nr, int nc, int nisland,
                           const int* island_block_ncols,
                           const int* island_col_offset,
                           const int* entity_island,
                           const int* map_row_to_entity,
                           const int* map_col_to_entity,
                           const int* rownnz, const int* rowadr, const int* colind,
                           const int* rowsuper, FILE* fp) {
  // if no rows / columns, or too many columns to be visually useful, return
  if (!nr || !nc || nc > 300) {
    return;
  }
  fprintf(fp, "%s\n", str);

  for (int c = 0; c < nc + 2; c++) fprintf(fp, "-");
  fprintf(fp, "\n");

  for (int r = 0; r < nr; r++) {
    fprintf(fp, " ");
    int entity_r = map_row_to_entity[r];
    int island = entity_island[entity_r];

    // SHOULD NOT OCCUR
    if (island < 0 || island >= nisland) {
      for (int c = 0; c < nc; c++) fprintf(fp, " ");
      fprintf(fp, " | Error: invalid island %d for row %d (entity %d)\n", island, r, entity_r);
      continue;
    }

    int c_start = island_col_offset[island];
    int bnc = island_block_ncols[island];
    int adr = rowadr[entity_r];
    int nnz = rownnz[entity_r];
    char nz_char = (island < 10) ? ('0' + island) : 'x';

    for (int c = 0; c < nc; c++) {  // c is the block-space column index
      bool nonzero = false;
      if (c >= c_start && c < c_start + bnc) {
        int target_col = map_col_to_entity[c];
        for (int i = 0; i < nnz; i++) {
          if (colind[adr + i] == target_col) {
            nonzero = true;
            break;
          }
        }
      }
      fprintf(fp, "%c", nonzero ? nz_char : ' ');
    }
    fprintf(fp, " |");
    if (rowsuper && rowsuper[entity_r] > 0) fprintf(fp, " %d", rowsuper[entity_r]);
    fprintf(fp, "\n");
  }
  for (int c = 0; c < nc + 2; c++) fprintf(fp, "-");
  fprintf(fp, "\n\n");
}


// print vector
static void printVector(const char* str, const mjtNum* data, int n, FILE* fp,
                        const char* float_format) {
  if (!data || !n) {
    return;
  }
  // print str
  fprintf(fp, "%s", str);

  // print data
  for (int i=0; i < n; i++) {
    fprintf(fp, " ");
    fprintf(fp, float_format, data[i]);
  }
  fprintf(fp, "\n");
}


// print human readable memory size
static const char* memorySize(size_t nbytes) {
  static mjTHREADLOCAL char message[32];
  int k = 1024;

  if (nbytes < k) {
    snprintf(message, sizeof(message), "%5zu bytes", nbytes);
  } else {
    snprintf(message, sizeof(message), "%7.0f KB", (double)nbytes / (k));
  }

  return message;
}


// return memory footprint of all significant mesh-related arrays
static size_t sizeMesh(const mjModel* m) {
  size_t nbytes = 0;
  nbytes += sizeof(float)  * 3*m->nmeshvert;      // mesh_vert
  nbytes += sizeof(float)  * 3*m->nmeshnormal;    // mesh_normal
  nbytes += sizeof(float)  * 2*m->nmeshtexcoord;  // mesh_texcoord
  nbytes += sizeof(int)    * 3*m->nmeshface;      // mesh_face
  nbytes += sizeof(int)    * 3*m->nmeshface;      // mesh_facenormal
  nbytes += sizeof(int)    * 3*m->nmeshface;      // mesh_facetexcoord
  nbytes += sizeof(int)    * m->nmeshgraph;       // mesh_graph
  nbytes += sizeof(mjtNum) * 3*m->nmeshpoly;      // mesh_polynormal
  nbytes += sizeof(int)    * m->nmeshpoly;        // mesh_polyvertadr
  nbytes += sizeof(int)    * m->nmeshpoly;        // mesh_polyvertnum
  nbytes += sizeof(int)    * m->nmeshpolyvert;    // mesh_polyvert
  nbytes += sizeof(int)    * m->nmeshvert;        // mesh_polymapadr
  nbytes += sizeof(int)    * m->nmeshvert;        // mesh_polymapnum
  nbytes += sizeof(int)    * m->nmeshpolymap;     // mesh_polymap
  return nbytes;
}


// return memory footprint of all significant skin-related arrays
static size_t sizeSkin(const mjModel* m) {
  size_t nbytes = 0;
  nbytes += sizeof(float) * 3*m->nskinvert;        // skin_vert
  nbytes += sizeof(float) * 2*m->nskintexvert;     // skin_texcoord
  nbytes += sizeof(int)   * 3*m->nskinface;        // skin_face
  nbytes += sizeof(int)   * m->nskinbone;          // skin_bonevertadr
  nbytes += sizeof(int)   * m->nskinbone;          // skin_bonevertnum
  nbytes += sizeof(float) * 3*m->nskinbone;        // skin_bonebindpos
  nbytes += sizeof(float) * 4*m->nskinbone;        // skin_bonebindquat
  nbytes += sizeof(int)   * m->nskinbone;          // skin_bonebodyid
  nbytes += sizeof(int)   * m->nskinbonevert;      // skin_bonevertid
  nbytes += sizeof(float) * m->nskinbonevert;      // skin_bonevertweight
  return nbytes;
}


// return memory footprint of all BVH-related arrays
static size_t sizeBVH(const mjModel* m) {
  size_t nbytes = 0;
  nbytes += sizeof(int)    * m->nbvh;               // bvh_depth
  nbytes += sizeof(int)    * 2*m->nbvh;             // bvh_child
  nbytes += sizeof(int)    * m->nbvh;               // bvh_nodeid
  nbytes += sizeof(mjtNum) * 6*m->nbvhstatic;       // bvh_aabb
  nbytes += sizeof(int)    * m->noct;               // oct_depth
  nbytes += sizeof(int)    * 8*m->noct;             // oct_child
  nbytes += sizeof(mjtNum) * 6*m->noct;             // oct_aabb
  nbytes += sizeof(mjtNum) * 8*m->noct;             // oct_coeff
  return nbytes;
}


// return whether float_format is a valid format string for a single float
static bool validateFloatFormat(const char* float_format) {
  // check for nullptr;
  if (!float_format) {
    return false;
  }

  // example valid format string: "% -9.2g"
  if (strnlen(float_format, FLOAT_FORMAT_MAX_LEN + 1) > FLOAT_FORMAT_MAX_LEN) {
    mju_warning("Format string longer than limit of %d.", FLOAT_FORMAT_MAX_LEN);
    return false;
  }

  int cur_idx = 0;
  if (float_format[cur_idx] != '%') {
    mju_warning("Format string must start with '%%'.");
    return false;
  }
  cur_idx++;

  // flag characters. allow at most one of each flag
  const char flag_characters[] = "-+ #0";
  int flag_character_counts[sizeof(flag_characters)] = { 0 };
  const char* c;
  while (float_format[cur_idx] != '\0' &&
         (c = strchr(flag_characters, float_format[cur_idx])) != NULL) {
    int flag_idx = (c - flag_characters)/sizeof(char);
    flag_character_counts[flag_idx]++;
    if (flag_character_counts[flag_idx] > 1) {
      mju_warning("Format string contains repeated flag.");
      return false;
    }
    cur_idx++;
  }

  // width. disallow *, which requires additional argument
  while (float_format[cur_idx] != '\0' &&
         strchr("0123456789", float_format[cur_idx]) != NULL) {
    cur_idx++;
  }

  // precision. disallow *, which requires additional argument
  if (float_format[cur_idx] == '.') {
    cur_idx++;
    while (float_format[cur_idx] != '\0' &&
           strchr("0123456789", float_format[cur_idx]) != NULL) {
      cur_idx++;
    }
  }

  // length
  if (float_format[cur_idx] == 'L') {
    cur_idx++;
  }

  // specifier must be a valid float format
  if (float_format[cur_idx] == '\0' ||
      strchr("fgGeE", float_format[cur_idx]) == NULL) {
    mju_warning("Format string specifier must be one of \"fgGeE\".");
    return false;
  }
  cur_idx++;

  if (float_format[cur_idx] == '\0') {
    return true;
  } else {
    mju_warning("Unable to match format string %s with expected pattern for a single float.",
                float_format);
    return false;
  }
}


   // Clang sometimes goes OOM when the -Wuninitialized warning is enabled for this function
 #ifdef __clang__
 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Wuninitialized"
 #endif

//------------------------------ printing functions ------------------------------------------------

// print mjModel to text file, specifying format. float_format must be a
// valid printf-style format string for a single float value
// NOLINTBEGIN(readability/fn_size)
void mj_printFormattedModel(const mjModel* m, const char* filename, const char* float_format) {
  // get file
  FILE* fp;
  if (filename) {
    fp = fopen(filename, "wt");
  } else {
    fp = stdout;
  }

  // check for nullptr
  if (!fp) {
    mju_warning("Could not open file '%s' for writing mjModel", filename);
    return;
  }

  // validate format string
  if (!validateFloatFormat(float_format)) {
    mju_warning("WARNING: Received invalid float_format. Using default instead.");
    float_format = FLOAT_FORMAT;
  }

  // compute total body mass
  mjtNum totalmass = 0;
  for (int i=1; i < m->nbody; i++) {
    totalmass += m->body_mass[i];
  }

  // software version and model name
  fprintf(fp, "MuJoCo version %s\n", mj_versionString());
  fprintf(fp, "model name     %s\n\n", m->names);

  // memory footprint
  fprintf(fp, "MEMORY\n");
  size_t sz_total = mj_sizeModel(m);
  size_t sz_mesh = m->nmesh ? sizeMesh(m) : 0;
  size_t sz_bvh = (m->nbvh || m->noct) ? sizeBVH(m) : 0;
  size_t sz_tex = m->ntex ? m->ntexdata : 0;
  size_t sz_skin = m->nskin ? sizeSkin(m) : 0;
  size_t sz_other = sz_total - sz_mesh - sz_bvh - sz_tex - sz_skin;
  fprintf(fp, "  total       %s\n", memorySize(sz_total));
  if (sz_mesh)  fprintf(fp, "  meshes      %s\n", memorySize(sz_mesh));
  if (sz_bvh)   fprintf(fp, "  bvhs        %s\n", memorySize(sz_bvh));
  if (sz_tex)   fprintf(fp, "  textures    %s\n", memorySize(sz_tex));
  if (sz_skin)  fprintf(fp, "  skins       %s\n", memorySize(sz_skin));
  if (sz_other) fprintf(fp, "  other       %s\n", memorySize(sz_other));
  fprintf(fp, "\n");


  // sizes
  fprintf(fp, "SIZES\n");
#define X( name )                                 \
  if (m->name) {                                  \
    const char* format = _Generic(                \
        m->name,                                  \
        mjtSize : SIZE_FORMAT,                    \
        default : INT_FORMAT);                    \
    fprintf(fp, NAME_FORMAT, "  " #name);         \
    fprintf(fp, format, m->name);                 \
    fprintf(fp, "\n");                            \
  }

  MJMODEL_SIZES
#undef X
  fprintf(fp, "\n");

  // flags
  fprintf(fp, "FLAG\n");
  printInt(fp, "  flg_gravcomp", m->flg_gravcomp);
  printInt(fp, "  flg_surfacevel", m->flg_surfacevel);
  fprintf(fp, "\n");

  // options
  fprintf(fp, "OPTION\n");
#define X(type, name, sz)                                             \
  fprintf(fp, NAME_FORMAT, "  " #name);                               \
  {                                                                   \
    const char* format =                                              \
        _Generic(m->opt.name, mjtNum: float_format, int: INT_FORMAT); \
    fprintf(fp, format, m->opt.name);                                 \
  }                                                                   \
  fprintf(fp, "\n");
#define XVEC(type, name, sz)                                             \
  fprintf(fp, NAME_FORMAT, "  " #name);                                  \
  {                                                                      \
    const char* format =                                                 \
        _Generic(m->opt.name[0], mjtNum: float_format, int: INT_FORMAT); \
    for (int i = 0; i < sz; i++) {                                       \
      fprintf(fp, format, m->opt.name[i]);                               \
    }                                                                    \
  }                                                                      \
  fprintf(fp, "\n");

  MJOPTION_FIELDS
#undef XVEC
#undef X
  fprintf(fp, "\n");

  // visual
  fprintf(fp, "VISUAL\n");

  fprintf(fp, "  GLOBAL\n");
#define X(type, name, sz)                                                   \
  fprintf(fp, NAME_FORMAT, "    " #name);                                   \
  {                                                                         \
    const char* format =                                                    \
        _Generic(m->vis.global.name, float: float_format, int: INT_FORMAT); \
    fprintf(fp, format, m->vis.global.name);                                \
  }                                                                         \
  fprintf(fp, "\n");

  MJVISUAL_GLOBAL_FIELDS
#undef X

  fprintf(fp, "  QUALITY\n");
#define X(type, name, sz)                        \
  fprintf(fp, NAME_FORMAT, "    " #name);        \
  fprintf(fp, INT_FORMAT, m->vis.quality.name);  \
  fprintf(fp, "\n");

  MJVISUAL_QUALITY_FIELDS
#undef X

  fprintf(fp, "  HEADLIGHT\n");
#define X(type, name, sz)                                                      \
  fprintf(fp, NAME_FORMAT, "    " #name);                                      \
  {                                                                            \
    const char* format =                                                       \
        _Generic(m->vis.headlight.name, float: float_format, int: INT_FORMAT); \
    fprintf(fp, format, m->vis.headlight.name);                                \
  }                                                                            \
  fprintf(fp, "\n");
#define XVEC(type, name, sz)                                \
  fprintf(fp, NAME_FORMAT, "    " #name);                   \
  {                                                         \
    const char* format = _Generic(                          \
        m->vis.headlight.name[0],                           \
        float: float_format,                                \
        int: INT_FORMAT);                                   \
    for (int i = 0; i < sz; i++) {                          \
      fprintf(fp, format, m->vis.headlight.name[i]);        \
    }                                                       \
  }                                                         \
  fprintf(fp, "\n");
  MJVISUAL_HEADLIGHT_FIELDS
#undef XVEC
#undef X

  fprintf(fp, "  MAP\n");
#define X(type, name, sz)                      \
  fprintf(fp, NAME_FORMAT, "    " #name);      \
  fprintf(fp, float_format, m->vis.map.name);  \
  fprintf(fp, "\n");

  MJVISUAL_MAP_FIELDS
#undef X

  fprintf(fp, "  SCALE\n");
#define X(type, name, sz)                        \
  fprintf(fp, NAME_FORMAT, "    " #name);        \
  fprintf(fp, float_format, m->vis.scale.name);  \
  fprintf(fp, "\n");

  MJVISUAL_SCALE_FIELDS
#undef X

  fprintf(fp, "  RGBA\n");
#define XVEC(type, name, sz)                          \
  fprintf(fp, NAME_FORMAT, "    " #name);             \
  {                                                   \
    for (int i = 0; i < sz; i++) {                    \
      fprintf(fp, float_format, m->vis.rgba.name[i]); \
    }                                                 \
  }                                                   \
  fprintf(fp, "\n");
  MJVISUAL_RGBA_FIELDS
#undef XVEC
  fprintf(fp, "\n");

  // total mass
  fprintf(fp, NAME_FORMAT, "totalmass");
  fprintf(fp, float_format, totalmass);
  fprintf(fp, "\n\n");

  // statistics
  fprintf(fp, "STATISTIC\n");
#define X(name, sz)                                             \
  fprintf(fp, NAME_FORMAT, "  " #name);                         \
  {                                                             \
    for (int i = 0; i < sz; i++) {                              \
      fprintf(fp, float_format, ((mjtNum*)(&m->stat.name))[i]); \
    }                                                           \
  }                                                             \
  fprintf(fp, "\n");
#define XVEC X

  MJSTATISTIC_FIELDS
#undef XVEC
#undef X
  fprintf(fp, "\n");

  // qpos0
  fprintf(fp, NAME_FORMAT, "qpos0");
  for (int i=0; i < m->nq; i++) {
    fprintf(fp, float_format, m->qpos0[i]);
    fprintf(fp, " ");
  }
  fprintf(fp, "\n\n");

  // qpos_spring
  fprintf(fp, NAME_FORMAT, "qpos_spring");
  for (int i=0; i < m->nq; i++) {
    fprintf(fp, float_format, m->qpos_spring[i]);
    fprintf(fp, " ");
  }
  fprintf(fp, "\n\n");

  // values used by MJMODEL_POINTERS macro
  MJMODEL_POINTERS_PREAMBLE(m)

  // touch unused sizes to avoid unused variable warning
  (void)nq;
  (void)nv;
  (void)na;
  (void)nu;
  (void)nmocap;
  (void)nplastic;

  const mjtSize* object_class;

#define X(type, name, num, sz)                                              \
  if (&m->num == object_class && sz > 0) {                                  \
    const char* format = _Generic(*m->name,                                 \
                                  double:  float_format,                    \
                                  float:   float_format,                    \
                                  int:     INT_FORMAT,                      \
                                  mjtByte: INT_FORMAT,                      \
                                  mjtBool: INT_FORMAT,                      \
                                  mjtSize: SIZE_FORMAT,                     \
                                  default: NULL);                           \
    if (format) {                                                           \
      fprintf(fp, "  ");                                                    \
      fprintf(fp, NAME_FORMAT, #name);                                      \
      for (int j = 0; j < sz; j++) {                                        \
          fprintf(fp, format, m->name[sz * i + j]);                         \
          fprintf(fp, " ");                                                 \
      }                                                                     \
      fprintf(fp, "\n");                                                    \
    }                                                                       \
  }
#define XNV X

  // bodies
  object_class = &m->nbody;
  for (int i=0; i < m->nbody; i++) {
    fprintf(fp, "\nBODY %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_bodyadr[i]);
    MJMODEL_POINTERS_BODY
  }
  if (m->nbody) fprintf(fp, "\n");

  // joints
  object_class = &m->njnt;
  for (int i=0; i < m->njnt; i++) {
    fprintf(fp, "\nJOINT %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_jntadr[i]);
    MJMODEL_POINTERS_JOINT
  }
  if (m->njnt) fprintf(fp, "\n");

  // dofs
  object_class = &m->nv;
  for (int i=0; i < m->nv; i++) {
    fprintf(fp, "\nDOF %d:\n", i);
    MJMODEL_POINTERS_DOF
  }
  if (m->nv) fprintf(fp, "\n");

  // trees
  object_class = &m->ntree;
  for (int i=0; i < m->ntree; i++) {
    fprintf(fp, "\nTREE %d:\n", i);
    MJMODEL_POINTERS_TREE
  }
  if (m->ntree) fprintf(fp, "\n");

  // geoms
  object_class = &m->ngeom;
  for (int i=0; i < m->ngeom; i++) {
    fprintf(fp, "\nGEOM %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_geomadr[i]);
    MJMODEL_POINTERS_GEOM
  }
  if (m->ngeom) fprintf(fp, "\n");

  // sites
  object_class = &m->nsite;
  for (int i=0; i < m->nsite; i++) {
    fprintf(fp, "\nSITE %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_siteadr[i]);
    MJMODEL_POINTERS_SITE
  }
  if (m->nsite) fprintf(fp, "\n");

  // cameras
  object_class = &m->ncam;
  for (int i=0; i < m->ncam; i++) {
    fprintf(fp, "\nCAMERA %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_camadr[i]);
    MJMODEL_POINTERS_CAMERA
  }
  if (m->ncam) fprintf(fp, "\n");

  // lights
  object_class = &m->nlight;
  for (int i=0; i < m->nlight; i++) {
    fprintf(fp, "\nLIGHT %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_lightadr[i]);
    MJMODEL_POINTERS_LIGHT
  }
  if (m->nlight) fprintf(fp, "\n");

  // flexes
  object_class = &m->nflex;
  for (int i=0; i < m->nflex; i++) {
    fprintf(fp, "\nFLEX %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_flexadr[i]);
    MJMODEL_POINTERS_FLEX
  }
  if (m->nflex) fprintf(fp, "\n");

  // meshes
  object_class = &m->nmesh;
  for (int i=0; i < m->nmesh; i++) {
    fprintf(fp, "\nMESH %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_meshadr[i]);
    MJMODEL_POINTERS_MESH
    if (m->mesh_graphadr[i] >= 0) {
      fprintf(fp, "  " NAME_FORMAT, "qhull face");
      fprintf(fp, " %d\n", m->mesh_graph[m->mesh_graphadr[i]+1]);
      fprintf(fp, "  " NAME_FORMAT, "qhull vert");
      fprintf(fp, " %d\n", m->mesh_graph[m->mesh_graphadr[i]]);
    }
  }
  if (m->nmesh) fprintf(fp, "\n");

  // skins
  object_class = &m->nskin;
  for (int i=0; i < m->nskin; i++) {
    fprintf(fp, "\nSKIN %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_skinadr[i]);
    MJMODEL_POINTERS_SKIN
  }
  if (m->nskin) fprintf(fp, "\n");

  // hfields
  object_class = &m->nhfield;
  for (int i=0; i < m->nhfield; i++) {
    fprintf(fp, "\nHEIGHTFIELD %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, "  %s\n", m->names + m->name_hfieldadr[i]);
    MJMODEL_POINTERS_HFIELD
  }
  if (m->nhfield) fprintf(fp, "\n");

  // textures
  object_class = &m->ntex;
  for (int i=0; i < m->ntex; i++) {
    fprintf(fp, "\nTEXTURE %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_texadr[i]);
    MJMODEL_POINTERS_TEXTURE
  }
  if (m->ntex) fprintf(fp, "\n");

  // materials
  object_class = &m->nmat;
  for (int i=0; i < m->nmat; i++) {
    fprintf(fp, "\nMATERIAL %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_matadr[i]);
    MJMODEL_POINTERS_MATERIAL
  }
  if (m->nmat) fprintf(fp, "\n");

  // pairs
  object_class = &m->npair;
  for (int i=0; i < m->npair; i++) {
    fprintf(fp, "\nPAIR %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_pairadr[i]);
    MJMODEL_POINTERS_PAIR
  }
  if (m->npair) fprintf(fp, "\n");

  // excludes
  object_class = &m->nexclude;
  for (int i=0; i < m->nexclude; i++) {
    fprintf(fp, "\nEXCLUDE %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_excludeadr[i]);
    MJMODEL_POINTERS_EXCLUDE
  }
  if (m->nexclude) fprintf(fp, "\n");

  // equality constraints
  object_class = &m->neq;
  for (int i=0; i < m->neq; i++) {
    fprintf(fp, "\nEQUALITY %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_eqadr[i]);
    MJMODEL_POINTERS_EQUALITY
  }
  if (m->neq) fprintf(fp, "\n");

  // tendons
  object_class = &m->ntendon;
  for (int i=0; i < m->ntendon; i++) {
    fprintf(fp, "\nTENDON %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_tendonadr[i]);
    MJMODEL_POINTERS_TENDON
    fprintf(fp, "  path\n");
    fprintf(fp, "    type  objid  prm\n");
    for (int j=0; j < m->tendon_num[i]; j++) {
      int k = m->tendon_adr[i]+j;
      fprintf(fp, "    %d     %d     ", m->wrap_type[k], m->wrap_objid[k]);
      fprintf(fp, float_format, m->wrap_prm[k]);
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
  if (m->ntendon) fprintf(fp, "\n");

  // actuators: per-actuator fields, then per-input (nu) and per-output (nout) blocks
  for (int i=0; i < m->nactuator; i++) {
    fprintf(fp, "\nACTUATOR %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_actuatoradr[i]);
    object_class = &m->nactuator;
    MJMODEL_POINTERS_ACTUATOR
    {
      int actuator = i;
      int ctrladr = m->actuator_ctrladr[actuator];
      int ctrlnum = m->actuator_ctrlnum[actuator];
      int outadr = m->actuator_outadr[actuator];
      int outnum = m->actuator_outnum[actuator];
      object_class = &m->nu;
      i = ctrladr;
      while (i < ctrladr + ctrlnum) {
        MJMODEL_POINTERS_ACTUATOR
        i++;
      }
      object_class = &m->nout;
      i = outadr;
      while (i < outadr + outnum) {
        MJMODEL_POINTERS_ACTUATOR
        i++;
      }
      i = actuator;
    }
  }
  if (m->nactuator) fprintf(fp, "\n");

  // sensors
  object_class = &m->nsensor;
  for (int i=0; i < m->nsensor; i++) {
    fprintf(fp, "\nSENSOR %d:\n", i);
    fprintf(fp, "  " NAME_FORMAT, "name");
    fprintf(fp, " %s\n", m->names + m->name_sensoradr[i]);
    MJMODEL_POINTERS_SENSOR
  }
  if (m->nsensor) fprintf(fp, "\n");

  // custom numeric parameters
  for (int i=0; i < m->nnumeric; i++) {
    fprintf(fp, "\nNUMERIC %d:\n", i);
    fprintf(fp, "  name         %s\n", m->names + m->name_numericadr[i]);
    fprintf(fp, "  size         %d\n", m->numeric_size[i]);
    fprintf(fp, "  value       ");
    for (int j=0; j < m->numeric_size[i]; j++) {
      fprintf(fp, float_format, m->numeric_data[m->numeric_adr[i]+j]);
    }
    fprintf(fp, "\n");
  }
  if (m->nnumeric) fprintf(fp, "\n");

  // custom text parameters
  for (int i=0; i < m->ntext; i++) {
    fprintf(fp, "\nTEXT %d:\n", i);
    fprintf(fp, "  name         %s\n", m->names + m->name_textadr[i]);
    fprintf(fp, "  size         %d\n", m->text_size[i]);
    fprintf(fp, "  value        %s\n", m->text_data + m->text_adr[i]);
  }
  if (m->ntext) fprintf(fp, "\n");

  // custom tuple parameters
  for (int i=0; i < m->ntuple; i++) {
    fprintf(fp, "\nTUPLE %d:\n", i);
    fprintf(fp, "  name         %s\n", m->names + m->name_tupleadr[i]);
    fprintf(fp, "  size         %d\n", m->tuple_size[i]);
    fprintf(fp, "  elements\n");
    for (int j=m->tuple_adr[i]; j < m->tuple_adr[i]+m->tuple_size[i]; j++) {
      fprintf(fp, "       %s %d, prm = ",
              mju_type2Str(m->tuple_objtype[j]), m->tuple_objid[j]);
      fprintf(fp, float_format, m->tuple_objprm[j]);
      fprintf(fp, "\n");
    }
  }
  if (m->ntuple) fprintf(fp, "\n");

  // keyframes (only if different from default)
  for (int i=0; i < m->nkey; i++) {
    // print name
    if (m->names[m->name_keyadr[i]]) {
      fprintf(fp, "key_name%d    %s\n", i, m->names + m->name_keyadr[i]);
    }

    // print time if non-0
    if (m->key_time[i] != 0) {
      fprintf(fp, "key_time%d    %.4f\n", i, m->key_time[i]);
    }

    // check qpos for difference
    int k = 0;
    for (int j=0; j < m->nq; j++)
      if (m->qpos0[j] != m->key_qpos[i*m->nq + j]) {
        k = 1;
      }

    // print if different
    if (k == 1) {
      fprintf(fp, "key_qpos%d   ", i);
      for (int j=0; j < m->nq; j++) {
        fprintf(fp, float_format, m->key_qpos[i*m->nq + j]);
      }
      fprintf(fp, "\n");
    }

    // check qvel for nonzero
    for (int j=0; j < m->nv; j++)
      if (m->key_qvel[i*m->nv + j]) {
        k = 2;
      }

    // print if nonzero
    if (k == 2) {
      fprintf(fp, "key_qvel%d   ", i);
      for (int j=0; j < m->nv; j++) {
        fprintf(fp, float_format, m->key_qvel[i*m->nv + j]);
      }
      fprintf(fp, "\n");
    }

    // check act for nonzero
    for (int j=0; j < m->na; j++)
      if (m->key_act[i*m->na + j]) {
        k = 3;
      }

    // print if nonzero
    if (k == 3) {
      fprintf(fp, "key_act%d   ", i);
      for (int j=0; j < m->na; j++) {
        fprintf(fp, float_format, m->key_act[i*m->na + j]);
      }
      fprintf(fp, "\n");
    }

    // check mpos for difference
    if (m->nmocap) {
      for (int j=0; j < m->nbody; j++) {
        if (m->body_mocapid[j] >= 0) {
          int id = m->body_mocapid[j];
          if (m->body_pos[3*j] != m->key_mpos[i*3*m->nmocap + 3*id] ||
              m->body_pos[3*j+1] != m->key_mpos[i*3*m->nmocap + 3*id+1] ||
              m->body_pos[3*j+2] != m->key_mpos[i*3*m->nmocap + 3*id+2]) {
            k = 4;
            break;
          }
        }
      }
    }

    // print if nonzero
    if (k == 4) {
      fprintf(fp, "key_mpos%d   ", i);
      for (int j=0; j < 3*m->nmocap; j++) {
        fprintf(fp, float_format, m->key_mpos[i*3*m->nmocap + j]);
      }
      fprintf(fp, "\n");
    }

    // check mquat for difference
    if (m->nmocap) {
      for (int j=0; j < m->nbody; j++) {
        if (m->body_mocapid[j] >= 0) {
          int id = m->body_mocapid[j];
          if (m->body_quat[4*j] != m->key_mquat[i*4*m->nmocap + 4*id] ||
              m->body_quat[4*j+1] != m->key_mquat[i*4*m->nmocap + 4*id+1] ||
              m->body_quat[4*j+2] != m->key_mquat[i*4*m->nmocap + 4*id+2] ||
              m->body_quat[4*j+3] != m->key_mquat[i*4*m->nmocap + 4*id+3]) {
            k = 5;
            break;
          }
        }
      }
    }

    // print if nonzero
    if (k == 5) {
      fprintf(fp, "key_mquat%d   ", i);
      for (int j=0; j < 4*m->nmocap; j++) {
        fprintf(fp, float_format, m->key_mquat[i*4*m->nmocap + j]);
      }
      fprintf(fp, "\n");
    }

    // check ctrl for nonzero
    for (int j=0; j < m->nu; j++) {
      if (m->key_ctrl[i*m->nu + j]) {
        k = 6;
        break;
      }
    }

    // print if nonzero
    if (k == 6) {
      fprintf(fp, "key_ctrl%d   ", i);
      for (int j=0; j < m->nu; j++) {
        fprintf(fp, float_format, m->key_ctrl[i*m->nu + j]);
      }
      fprintf(fp, "\n");
    }

    // check plastic for difference
    if (m->nplastic) {
      for (int j=0; j < m->njnt; j++) {
        if (m->jnt_plasticadr[j] >= 0) {
          int adr = m->jnt_plasticadr[j];
          int qposadr = m->jnt_qposadr[j];
          int jnt_nq = m->jnt_type[j] == mjJNT_FREE ? 7 : (m->jnt_type[j] == mjJNT_BALL ? 4 : 1);
          for (int d=0; d < jnt_nq; d++) {
            if (m->qpos_spring[qposadr + d] != m->key_plastic[i*m->nplastic + adr + d]) {
              k = 7;
              break;
            }
          }
          if (k == 7) break;
        }
      }
      for (int j=0; j < m->ntendon && k != 7; j++) {
        if (m->tendon_plasticadr[j] >= 0) {
          int adr = m->tendon_plasticadr[j];
          if (m->tendon_lengthspring[2*j] != m->key_plastic[i*m->nplastic + adr] ||
              m->tendon_lengthspring[2*j+1] != m->key_plastic[i*m->nplastic + adr + 1]) {
            k = 7;
            break;
          }
        }
      }
      for (int j=0; j < m->nflex && k != 7; j++) {
        if (m->flex_plasticadr[j] >= 0) {
          int adr = m->flex_plasticadr[j];
          int pnum = 0;
          if (m->flex_interp[j] > 0) {
            pnum = 6 * m->flex_cellnum[3*j+0] * m->flex_cellnum[3*j+1] * m->flex_cellnum[3*j+2];
          } else if (m->flex_interp[j] < 0) {
            int cx = m->flex_cellnum[3*j+0];
            int cy = m->flex_cellnum[3*j+1];
            int cz = m->flex_cellnum[3*j+2];
            pnum = 3 * 2 * (cy*cz + cx*cz + cx*cy);
          } else if (m->flex_dim[j] == 1) {
            pnum = mjROD_STATE_SIZE;
          } else {
            if (m->flex_stiffnessadr[j] >= 0 && m->flex_stiffness[m->flex_stiffnessadr[j]] != 0) {
              pnum += ((m->flex_dim[j] == 2) ? 3 : 6) * m->flex_elemnum[j];
            }
            if (m->flex_dim[j] == 2 && m->flex_bendingadr[j] >= 0) {
              pnum += m->flex_edgenum[j];
            }
          }
          for (int d=0; d < pnum; d++) {
            if (m->key_plastic[i*m->nplastic + adr + d] != 0) {
              k = 7;
              break;
            }
          }
        }
      }
    }

    // print if different
    if (k == 7) {
      fprintf(fp, "key_plastic%d   ", i);
      for (int j=0; j < m->nplastic; j++) {
        fprintf(fp, float_format, m->key_plastic[i*m->nplastic + j]);
      }
      fprintf(fp, "\n");
    }


    // new line if any data was written
    if (k) {
      fprintf(fp, "\n");
    }
  }

#undef X

  // B sparse structure
  mj_printSparsity("B: body-dof matrix", m->nbody, m->nv, m->B_rowadr, NULL, m->B_rownnz, NULL,
                   m->B_colind, fp);
  printArray2dInt("B_ROWNNZ", 1, m->nbody, m->B_rownnz, fp);
  printArray2dInt("B_ROWADR", 1, m->nbody, m->B_rowadr, fp);
  printArray2dInt("B_COLIND", 1, m->nB,    m->B_colind, fp);

  // M sparse structure
  mj_printSparsity("M: reduced inertia matrix", m->nv, m->nv, m->M_rowadr, NULL, m->M_rownnz,
                   NULL, m->M_colind, fp);
  printArray2dInt("M_ROWNNZ", 1, m->nv, m->M_rownnz, fp);
  printArray2dInt("M_ROWADR", 1, m->nv, m->M_rowadr, fp);
  printArray2dInt("M_COLIND", 1, m->nC, m->M_colind, fp);
  printArray2dInt("MAPM2M",   1, m->nC, m->mapM2M,   fp);

  // D sparse structure
  mj_printSparsity("D: dof-dof matrix", m->nv, m->nv,
                   m->D_rowadr, m->D_diag, m->D_rownnz, NULL, m->D_colind, fp);
  printArray2dInt("D_ROWNNZ", 1, m->nv, m->D_rownnz, fp);
  printArray2dInt("D_ROWADR", 1, m->nv, m->D_rowadr, fp);
  printArray2dInt("D_COLIND", 1, m->nD, m->D_colind, fp);
  printArray2dInt("MAPM2D",   1, m->nD, m->mapM2D,   fp);
  printArray2dInt("MAPD2M",   1, m->nC, m->mapD2M,   fp);

  // signature
  fprintf(fp, "\nSIGNATURE\n");
  fprintf(fp, "  %lu\n", (unsigned long) m->signature);
  fprintf(fp, "\n");

  // BVHs
  fprintf(fp, "BVH:\n");
  fprintf(fp, "  %-8s%-8s%-8s%-10s%-s\n", "id", "depth", "nodeid", "child[0]", "child[1]");
  for (int i=0; i < m->nbvh; i++) {
    fprintf(fp, "  %-8d%-8d% -8d% -10d% -d\n",
            i, m->bvh_depth[i], m->bvh_nodeid[i], m->bvh_child[2*i], m->bvh_child[2*i+1]);
  }
  fprintf(fp, "\n");

  if (filename) {
    fclose(fp);
  }
}
// NOLINTEND(readability/fn_size)

// print mjModel to text file
void mj_printModel(const mjModel* m, const char* filename) {
  mj_printFormattedModel(m, filename, FLOAT_FORMAT);
}


// print mjModel to text file, specifying format. float_format must be a
// valid printf-style format string for a single float value
void mj_printFormattedData(const mjModel* m, const mjData* d, const char* filename,
                           const char* float_format) {
  // stack in use, SHOULD NOT OCCUR
  if (d->pstack) {
    mjERROR("attempting to print mjData when stack is in use");
  }

  // check format string
  if (!validateFloatFormat(float_format)) {
    mju_warning("WARNING: Received invalid float_format. Using default instead.");
    float_format = FLOAT_FORMAT;
  }

  // get file
  FILE* fp;
  if (filename) {
    fp = fopen(filename, "wt");
  } else {
    fp = stdout;
  }

  // check for nullptr
  if (!fp) {
    mju_warning("Could not open file '%s' for writing mjModel", filename);
    return;
  }

#ifdef MEMORY_SANITIZER
  // If memory sanitizer is active, d->buffer will be marked as poisoned, even
  // though it's really initialized to 0. This catches unintentionally
  // using uninitialized values, but in engine_print it's OK to output zeroes.

  // save current poison status of buffer before marking unpoisoned
  void* shadow = mju_malloc(d->nbuffer);
  __msan_copy_shadow(shadow, d->buffer, d->nbuffer);
  __msan_unpoison(d->buffer, d->nbuffer);
#endif

  fprintf(fp, "MEMORY\n");
  fprintf(fp, "  total         %s\n",   memorySize(sizeof(mjData) + d->nbuffer + d->narena));
  fprintf(fp, "  struct        %s\n",   memorySize(sizeof(mjData)));
  fprintf(fp, "  buffer        %s\n",   memorySize(d->nbuffer));
  double arena_percent = 100 * d->maxuse_arena/(double)(d->narena);
  fprintf(fp, "  arena         %s, used %.1f%%\n\n", memorySize(d->narena), arena_percent);

  // ---------------------------------- print mjData fields

  fprintf(fp, "SIZES\n");
#define X(type, name)                                                         \
  if (strcmp(#name, "pstack") != 0 &&                                         \
      strcmp(#name, "pbase") != 0 &&                                          \
      strcmp(#name, "parena") != 0 &&                                         \
      strcmp(#name, "threadpool") != 0) {                                     \
    const char* format = _Generic(                                            \
        d->name,                                                              \
        int : INT_FORMAT,                                                     \
        mjtSize : SIZE_FORMAT,                                                \
        mjtByte : INT_FORMAT,                                                 \
        mjtBool : INT_FORMAT,                                                 \
        default : NULL);                                                      \
    if (format) {                                                             \
      fprintf(fp, "  ");                                                      \
      fprintf(fp, NAME_FORMAT, #name);                                        \
      fprintf(fp, format, d->name);                                           \
      fprintf(fp, "\n");                                                      \
    }                                                                         \
  }

  MJDATA_SCALAR
#undef X

  int threadpool = 0;
  if (d->threadpool) {
    threadpool = 1;
  }
  fprintf(fp, "  ");
  fprintf(fp, NAME_FORMAT, "threadpool");
  fprintf(fp, INT_FORMAT, threadpool);
  fprintf(fp, "\n");

  fprintf(fp, "\n");

  // WARNING
  int active_warnings = 0;
  for (int i=0; i < mjNWARNING; i++) {
    active_warnings += d->warning[i].number;
  }
  if (active_warnings) {
    fprintf(fp, "WARNING\n");
    for (int i=0; i < mjNWARNING; i++)
      if (d->warning[i].number)
        fprintf(fp, "    %d:  lastinfo = %d   number = %d\n",
                i, d->warning[i].lastinfo, d->warning[i].number);
    fprintf(fp, "\n");
  }

  // TIMER
  mjtNum active_timers = 0;
  for (int i=0; i < mjNTIMER; i++) {
    active_timers += d->timer[i].duration;
  }
  if (active_timers) {
    fprintf(fp, "TIMER\n");
    for (int i=0; i < mjNTIMER; i++) {
      fprintf(fp, "    %d:  duration = ", i);
      fprintf(fp, float_format, d->timer[i].duration);
      fprintf(fp, "   number = %d\n", d->timer[i].number);
    }
    fprintf(fp, "\n");
  }

  // SOLVER STAT
  if (d->nefc) {
    fprintf(fp, "SOLVER STAT\n");
    printVector("  solver_fwdinv = ", d->solver_fwdinv, 2, fp, float_format);
    int nisland_stat = mjMAX(1, mjMIN(d->nisland, mjNISLAND));
    for (int island=0; island < nisland_stat; island++) {
      int niter_stat = mjMIN(mjNSOLVER, d->solver_niter[island]);
      if (niter_stat) {
        fprintf(fp, "  ISLAND %d\n", island);
        fprintf(fp, "    solver_niter = %d\n", d->solver_niter[island]);
        fprintf(fp, "    solver_nnz = %d\n",  d->solver_nnz[island]);
        for (int i=0; i < niter_stat; i++) {
          const mjSolverStat* stat = d->solver + island*mjNSOLVER + i;
          fprintf(fp, "      %d:  improvement = ", i);
          fprintf(fp, float_format, stat->improvement);
          fprintf(fp, "    gradient = ");
          fprintf(fp, float_format, stat->gradient);
          fprintf(fp, "    lineslope = ");
          fprintf(fp, float_format, stat->lineslope);
          fprintf(fp, "\n");
          fprintf(fp, "          nactive = %d   nchange = %d   neval = %d   nupdate = %d\n",
                  stat->nactive, stat->nchange,
                  stat->neval, stat->nupdate);
        }
        fprintf(fp, "\n");
      }
    }
  }

  printVector("ENERGY = ", d->energy, 2, fp, float_format);
  fprintf(fp, "\n");

  fprintf(fp, "TIME = ");
  fprintf(fp, float_format, d->time);
  fprintf(fp, "\n\n");

  printArray2d("QPOS", m->nq, 1, d->qpos, fp, float_format);
  printArray2d("QVEL", m->nv, 1, d->qvel, fp, float_format);
  printArray2d("ACT", m->na, 1, d->act, fp, float_format);

  // print history buffers with semantic structure
  if (m->nhistory) {
    fprintf(fp, "DELAY\n");

    // actuator history buffers
    for (int i = 0; i < m->nactuator; i++) {
      int adr = m->actuator_historyadr[i];
      if (adr >= 0) {
        char name[100];
        const char* actuator_name = mj_id2name(m, mjOBJ_ACTUATOR, i);
        snprintf(name, sizeof(name), "actuator %d '%s'", i, actuator_name ? actuator_name : "");
        printDelayBuffer(name, d->history + adr, m->actuator_history[2*i], m->actuator_ctrlnum[i],
                         fp, float_format);
      }
    }

    // sensor history buffers
    for (int i = 0; i < m->nsensor; i++) {
      int adr = m->sensor_historyadr[i];
      if (adr >= 0) {
        char name[100];
        const char* sensor_name = mj_id2name(m, mjOBJ_SENSOR, i);
        snprintf(name, sizeof(name), "sensor %d  '%s'", i, sensor_name ? sensor_name : "");
        printDelayBuffer(name, d->history + adr, m->sensor_history[2*i], m->sensor_dim[i],
                         fp, float_format);
      }
    }

    fprintf(fp, "\n");
  }
  printArray2d("PLASTIC", m->nplastic, 1, d->plastic, fp, float_format);
  printArray2d("QACC_WARMSTART", m->nv, 1, d->qacc_warmstart, fp, float_format);
  printArray2d("CTRL", m->nu, 1, d->ctrl, fp, float_format);
  printArray2d("QFRC_APPLIED", m->nv, 1, d->qfrc_applied, fp, float_format);
  printArray2d("XFRC_APPLIED", m->nbody, 6, d->xfrc_applied, fp, float_format);
  if (m->neq) {
    fprintf(fp, NAME_FORMAT, "EQ_ACTIVE");
    for (int c=0; c < m->neq; c++) {
      fprintf(fp, " %d", d->eq_active[c]);
    }
    fprintf(fp, "\n\n");
  }
  printArray2d("MOCAP_POS", m->nmocap, 3, d->mocap_pos, fp, float_format);
  printArray2d("MOCAP_QUAT", m->nmocap, 4, d->mocap_quat, fp, float_format);
  printArray2d("QACC", m->nv, 1, d->qacc, fp, float_format);
  printArray2d("ACT_DOT", m->na, 1, d->act_dot, fp, float_format);
  printArray2d("USERDATA", m->nuserdata, 1, d->userdata, fp, float_format);
  printArray2d("SENSOR", m->nsensordata, 1, d->sensordata, fp, float_format);
  printArray2dInt("TREE_ASLEEP", m->ntree, 1, d->tree_asleep, fp);

  printArray2d("XPOS", m->nbody, 3, d->xpos, fp, float_format);
  printArray2d("XQUAT", m->nbody, 4, d->xquat, fp, float_format);
  printArray2d("XMAT", m->nbody, 9, d->xmat, fp, float_format);
  printArray2d("XIPOS", m->nbody, 3, d->xipos, fp, float_format);
  printArray2d("XIQUAT", m->nbody, 4, d->xiquat, fp, float_format);
  printArray2d("XIMAT", m->nbody, 9, d->ximat, fp, float_format);
  printArray2d("XANCHOR", m->njnt, 3, d->xanchor, fp, float_format);
  printArray2d("XAXIS", m->njnt, 3, d->xaxis, fp, float_format);
  printArray2d("GEOM_XPOS", m->ngeom, 3, d->geom_xpos, fp, float_format);
  printArray2d("GEOM_XQUAT", m->ngeom, 4, d->geom_xquat, fp, float_format);
  printArray2d("GEOM_XMAT", m->ngeom, 9, d->geom_xmat, fp, float_format);
  printArray2d("SITE_XPOS", m->nsite, 3, d->site_xpos, fp, float_format);
  printArray2d("SITE_XQUAT", m->nsite, 4, d->site_xquat, fp, float_format);
  printArray2d("SITE_XMAT", m->nsite, 9, d->site_xmat, fp, float_format);
  printArray2d("CAM_XPOS", m->ncam, 3, d->cam_xpos, fp, float_format);
  printArray2d("CAM_XMAT", m->ncam, 9, d->cam_xmat, fp, float_format);
  printArray2d("LIGHT_XPOS", m->nlight, 3, d->light_xpos, fp, float_format);
  printArray2d("LIGHT_XDIR", m->nlight, 3, d->light_xdir, fp, float_format);

  printArray2d("SUBTREE_COM", m->nbody, 3, d->subtree_com, fp, float_format);
  printArray2d("CDOF", m->nv, 6, d->cdof, fp, float_format);
  printArray2d("CINERT", m->nbody, 10, d->cinert, fp, float_format);

  printArray2d("FLEXVERT_XPOS", m->nflexvert, 3, d->flexvert_xpos, fp, float_format);
  printArray2d("FLEXELEM_AABB", m->nflexelem, 6, d->flexelem_aabb, fp, float_format);
  mj_printSparsity("FLEXEDGE_J: flex edge connectivity", m->nflexedge, m->nv,
                    m->flexedge_J_rowadr, NULL, m->flexedge_J_rownnz, NULL, m->flexedge_J_colind,
                    fp);
  printSparse("FLEXEDGE_J", d->flexedge_J, m->nflexedge, m->flexedge_J_rownnz,
                            m->flexedge_J_rowadr, m->flexedge_J_colind, fp, float_format);
  printArray2d("FLEXEDGE_LENGTH", m->nflexedge, 1, d->flexedge_length, fp, float_format);

  printArray2d("TEN_LENGTH", m->ntendon, 1, d->ten_length, fp, float_format);
  mj_printSparsity("TEN_J: tendon moments", m->ntendon, m->nv, m->ten_J_rowadr, NULL,
                   m->ten_J_rownnz, NULL, m->ten_J_colind, fp);
  printArray2dInt("TEN_J_ROWNNZ", m->ntendon, 1, m->ten_J_rownnz, fp);
  printArray2dInt("TEN_J_ROWADR", m->ntendon, 1, m->ten_J_rowadr, fp);
  printSparse("TEN_J", d->ten_J, m->ntendon, m->ten_J_rownnz,
              m->ten_J_rowadr, m->ten_J_colind, fp, float_format);
  for (int i=0; i < m->ntendon; i++) {
    fprintf(fp, "TENDON %d: %d wrap points\n", i, d->ten_wrapnum[i]);
    for (int j=0; j < d->ten_wrapnum[i]; j++) {
      fprintf(fp, "    %d:  ", d->wrap_obj[d->ten_wrapadr[i]+j]);
      printVector("", d->wrap_xpos+3*(d->ten_wrapadr[i]+j), 3, fp, float_format);
    }
    fprintf(fp, "\n");
  }

  printArray2d("ACTUATOR_LENGTH", m->nout, 1, d->actuator_length, fp, float_format);
  mj_printSparsity("actuator_moment", m->nout, m->nv,
                   d->moment_rowadr, NULL, d->moment_rownnz, NULL, d->moment_colind, fp);
  printSparse("ACTUATOR_MOMENT", d->actuator_moment, m->nout, d->moment_rownnz,
              d->moment_rowadr, d->moment_colind, fp, float_format);
  printArray2d("CRB", m->nbody, 10, d->crb, fp, float_format);
  printSparse("M", d->M, m->nv, m->M_rownnz,
              m->M_rowadr, m->M_colind, fp, float_format);
  printSparse("QLD", d->qLD, m->nv, m->M_rownnz,
              m->M_rowadr, m->M_colind, fp, float_format);
  printArray2d("QLDIAGINV", m->nv, 1, d->qLDiagInv, fp, float_format);
  if (d->nisland) {
    // the static full inertia structure is already printed in printModel, so we only repeat it here
    // if islands are present, for comparison
    mj_printSparsity("M: inertia structure", m->nv, m->nv, m->M_rowadr, NULL,
                     m->M_rownnz, NULL, m->M_colind, fp);
    mj_printBlockSparsity("iM: block-diagonal inertia (nnzs are island ids)",
                          d->nidof, d->nidof, d->nisland,
                          d->island_nv, d->island_idofadr,
                          d->dof_island, d->map_idof2dof, d->map_idof2dof,
                          m->M_rownnz, m->M_rowadr, m->M_colind, NULL, fp);
  }

  if (!mju_isZero(d->qHDiagInv, m->nv)) {
    printSparse("QH", d->qH, m->nv, m->M_rownnz, m->M_rowadr, m->M_colind, fp, float_format);
    printArray2d("QHDIAGINV", m->nv, 1, d->qHDiagInv, fp, float_format);
  }

  // computed sleep state
  printArray2dInt("TREE_AWAKE", 1, m->ntree, d->tree_awake, fp);
  printArray2dInt("BODY_AWAKE", 1, m->nbody, d->body_awake, fp);
  printArray2dInt("BODY_AWAKE_IND", 1, d->nbody_awake, d->body_awake_ind, fp);
  printArray2dInt("PARENT_AWAKE_IND", 1, d->nparent_awake, d->parent_awake_ind, fp);
  printArray2dInt("DOF_AWAKE_IND", 1, d->nv_awake, d->dof_awake_ind, fp);

  // print qDeriv
  if (!mju_isZero(d->qDeriv, m->nD)) {
    printSparse("QDERIV", d->qDeriv, m->nv, m->D_rownnz, m->D_rowadr, m->D_colind,
                fp, float_format);
  }

  // print qLU
  if (!mju_isZero(d->qLU, m->nD)) {
    printSparse("QLU", d->qLU, m->nv, m->D_rownnz, m->D_rowadr, m->D_colind, fp, float_format);
  }

  // contact
  if (d->ncon) fprintf(fp, "CONTACT\n");
  for (int i=0; i < d->ncon; i++) {
    fprintf(fp, "  %d:\n     dim           %d\n", i, d->contact[i].dim);
    int g1 = d->contact[i].geom[0];
    int g2 = d->contact[i].geom[1];

    // special case for geom-geom contacts
    if (g1 > -1 && g2 > -1) {
      fprintf(fp, "     geoms         ");
      const char* geom1 = mj_id2name(m, mjOBJ_GEOM, g1);
      const char* geom2 = mj_id2name(m, mjOBJ_GEOM, g2);
      if (geom1) {
        fprintf(fp, "%s : ", geom1);
      } else {
        fprintf(fp, "%d : ", g1);
      }
      if (geom2) {
        fprintf(fp, "%s\n", geom2);
      } else {
        fprintf(fp, "%d\n", g2);
      }
    } else {
      fprintf(fp, "     gfev          %d %d %d %d : %d %d %d %d\n",
              d->contact[i].geom[0], d->contact[i].flex[0],
              d->contact[i].elem[0], d->contact[i].vert[0],
              d->contact[i].geom[1], d->contact[i].flex[1],
              d->contact[i].elem[1], d->contact[i].vert[1]);
    }
    fprintf(fp, "     exclude       %d\n     efc_address   %d\n",
            d->contact[i].exclude, d->contact[i].efc_address);
    printVector("     solref       ", d->contact[i].solref, mjNREF, fp, float_format);
    printVector("     solimp       ", d->contact[i].solimp, mjNIMP, fp, float_format);
    printVector("     dist         ", &d->contact[i].dist, 1, fp, float_format);
    printVector("     includemargin", &d->contact[i].includemargin, 1, fp, float_format);
    printVector("     pos          ", d->contact[i].pos, 3, fp, float_format);
    printVector("     frame        ", d->contact[i].frame, 9, fp, float_format);
    printVector("     friction     ", d->contact[i].friction, 5, fp, float_format);
    printVector("     mu           ", &d->contact[i].mu, 1, fp, float_format);
    mjtNum force[6] = {0};
    mj_contactForce(m, d, i, force);
    printVector("     force        ", force, 6, fp, float_format);
  }
  if (d->ncon) fprintf(fp, "\n");

  printArray2dInt("EFC_TYPE", d->nefc, 1, d->efc_type, fp);
  printArray2dInt("EFC_ID", d->nefc, 1, d->efc_id, fp);

  if (!mj_isSparse(m)) {
    printArray2d("EFC_J", d->nefc, m->nv, d->efc_J, fp, float_format);
    if (d->nisland) {
      printBlockArray("IEFC_J", d->efc_J, m->nv,
                      d->nisland, d->island_nefc, d->island_nv,
                      d->island_iefcadr, d->island_idofadr,
                      d->map_iefc2efc, d->map_idof2dof,
                      fp, float_format);
    }
    printArray2d("EFC_AR", d->nefc, d->nefc, d->efc_AR, fp, float_format);
  } else {
    printArray2dInt("EFC_J_ROWNNZ", d->nefc, 1, d->efc_J_rownnz, fp);
    printArray2dInt("EFC_J_ROWADR", d->nefc, 1, d->efc_J_rowadr, fp);
    printSparse("EFC_J", d->efc_J, d->nefc, d->efc_J_rownnz,
                d->efc_J_rowadr, d->efc_J_colind, fp, float_format);
    mj_printSparsity("J: constraint Jacobian", d->nefc, m->nv, d->efc_J_rowadr, NULL,
                     d->efc_J_rownnz, d->efc_J_rowsuper, d->efc_J_colind, fp);
    if (d->nY) {
      mj_printSparsity("EFC_Y: inverse constraint inertia square root", d->nefc, m->nv,
                       d->efc_Y_rowadr, NULL, d->efc_Y_rownnz, d->efc_J_rowsuper,
                       d->efc_Y_colind, fp);
    }
    if (d->nisland) {
      mj_printBlockSparsity("IEFC_J: block-diagonalized constraint Jacobian (nnzs are island ids)",
                            d->nefc, d->nidof, d->nisland,
                            d->island_nv, d->island_idofadr,
                            d->efc_island, d->map_iefc2efc, d->map_idof2dof,
                            d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                            d->efc_J_rowsuper, fp);
    }

    if (mj_isDual(m)) {
      printArray2dInt("EFC_AR_ROWNNZ", d->nefc, 1, d->efc_AR_rownnz, fp);
      printArray2dInt("EFC_AR_ROWADR", d->nefc, 1, d->efc_AR_rowadr, fp);
      printSparse("EFC_AR", d->efc_AR, d->nefc, d->efc_AR_rownnz,
                  d->efc_AR_rowadr, d->efc_AR_colind, fp, float_format);
      mj_printSparsity("EFC_AR: inverse constraint inertia", d->nefc, d->nefc, d->efc_AR_rowadr,
                       NULL, d->efc_AR_rownnz, NULL, d->efc_AR_colind, fp);
    }
  }

  printArray2d("EFC_POS", d->nefc, 1, d->efc_pos, fp, float_format);
  printArray2d("EFC_MARGIN", d->nefc, 1, d->efc_margin, fp, float_format);
  printArray2d("EFC_FRICTIONLOSS", d->nefc, 1, d->efc_frictionloss, fp, float_format);
  printArray2d("EFC_DIAGA", d->nefc, 1, d->efc_diagA, fp, float_format);
  printArray2d("EFC_KBIP", d->nefc, 4, d->efc_KBIP, fp, float_format);
  printArray2d("EFC_D", d->nefc, 1, d->efc_D, fp, float_format);
  printArray2d("EFC_R", d->nefc, 1, d->efc_R, fp, float_format);

  printArray2d("FLEXEDGE_VELOCITY", m->nflexedge, 1, d->flexedge_velocity, fp, float_format);
  printArray2d("TEN_VELOCITY", m->ntendon, 1, d->ten_velocity, fp, float_format);
  printArray2d("ACTUATOR_VELOCITY", m->nout, 1, d->actuator_velocity, fp, float_format);

  printArray2d("CVEL", m->nbody, 6, d->cvel, fp, float_format);
  printArray2d("CDOF_DOT", m->nv, 6, d->cdof_dot, fp, float_format);

  printArray2d("QFRC_BIAS", m->nv, 1, d->qfrc_bias, fp, float_format);

  printArray2d("QFRC_SPRING",   m->nv, 1, d->qfrc_spring,   fp, float_format);
  printArray2d("QFRC_DAMPER",   m->nv, 1, d->qfrc_damper,   fp, float_format);
  printArray2d("QFRC_GRAVCOMP", m->nv, 1, d->qfrc_gravcomp, fp, float_format);
  printArray2d("QFRC_FLUID",    m->nv, 1, d->qfrc_fluid,    fp, float_format);
  printArray2d("QFRC_PASSIVE",  m->nv, 1, d->qfrc_passive,  fp, float_format);

  printArray2d("EFC_VEL", d->nefc, 1, d->efc_vel, fp, float_format);
  printArray2d("EFC_AREF", d->nefc, 1, d->efc_aref, fp, float_format);

  printArray2d("SUBTREE_LINVEL", m->nbody, 3, d->subtree_linvel, fp, float_format);
  printArray2d("SUBTREE_ANGMOM", m->nbody, 3, d->subtree_angmom, fp, float_format);

  printArray2d("ACTUATOR_FORCE", m->nout, 1, d->actuator_force, fp, float_format);
  printArray2d("QFRC_ACTUATOR", m->nv, 1, d->qfrc_actuator, fp, float_format);

  printArray2d("QFRC_SMOOTH", m->nv, 1, d->qfrc_smooth, fp, float_format);
  printArray2d("QACC_SMOOTH", m->nv, 1, d->qacc_smooth, fp, float_format);

  printArray2d("EFC_B", d->nefc, 1, d->efc_b, fp, float_format);
  printArray2d("EFC_FORCE", d->nefc, 1, d->efc_force, fp, float_format);
  printArray2dInt("EFC_STATE", d->nefc, 1, d->efc_state, fp);
  printArray2d("QFRC_CONSTRAINT", m->nv, 1, d->qfrc_constraint, fp, float_format);

  printArray2d("QFRC_INVERSE", m->nv, 1, d->qfrc_inverse, fp, float_format);

  printArray2d("CACC", m->nbody, 6, d->cacc, fp, float_format);
  printArray2d("CFRC_INT", m->nbody, 6, d->cfrc_int, fp, float_format);
  printArray2d("CFRC_EXT", m->nbody, 6, d->cfrc_ext, fp, float_format);

  if (d->nisland) {
    printArray2dInt("TREE_ISLAND", 1, m->ntree, d->tree_island, fp);
    printArray2dInt("ISLAND_NTREE", 1, d->nisland, d->island_ntree, fp);
    printArray2dInt("ISLAND_ITREEADR", 1, d->nisland, d->island_itreeadr, fp);
    printArray2dInt("MAP_ITREE2TREE", 1, m->ntree, d->map_itree2tree, fp);

    fprintf(fp, NAME_FORMAT, "DOF_ISLAND");
    for (int i = 0; i < m->nv; i++) {
      fprintf(fp, " %d", d->dof_island[i]);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "ISLAND_NV");
    for (int i = 0; i < d->nisland; i++) {
      fprintf(fp, " %d", d->island_nv[i]);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "ISLAND_IDOFADR");
    for (int i = 0; i < d->nisland; i++) {
      fprintf(fp, " %d", d->island_idofadr[i]);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "MAP_IDOF2DOF");
    for (int i = 0; i < m->nv; i++) {
      int dof = d->map_idof2dof[i];
      if (i > 0) {
        int dofprev = d->map_idof2dof[i-1];

        // print '|' at island boundaries
        if (d->dof_island[dof] != d->dof_island[dofprev]) {
          fprintf(fp, " |");
        }
      }
      fprintf(fp, " %d", dof);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "EFC_ISLAND");
    for (int i = 0; i < d->nefc; i++) {
      fprintf(fp, " %d", d->efc_island[i]);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "ISLAND_NEFC");
    for (int i = 0; i < d->nisland; i++) {
      fprintf(fp, " %d", d->island_nefc[i]);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "ISLAND_IEFCADR");
    for (int i = 0; i < d->nisland; i++) {
      fprintf(fp, " %d", d->island_iefcadr[i]);
    }
    fprintf(fp, "\n\n");

    fprintf(fp, NAME_FORMAT, "MAP_IEFC2EFC");
    for (int i = 0; i < d->nefc; i++) {
      int efc = d->map_iefc2efc[i];
      if (i > 0) {
        int efcprev = d->map_iefc2efc[i-1];

        // print '|' at island boundaries
        if (d->efc_island[efc] != d->efc_island[efcprev]) {
          fprintf(fp, " |");
        }
      }
      fprintf(fp, " %d", efc);
    }
    fprintf(fp, "\n\n");
  }

  if (m->ntendon && d->tendon_efcadr) {
    fprintf(fp, NAME_FORMAT, "TENDON_EFCADR");
    for (int i = 0; i < m->ntendon; i++) {
      fprintf(fp, " %d", d->tendon_efcadr[i]);
    }
    fprintf(fp, "\n\n");
  }

#ifdef MEMORY_SANITIZER
  // restore poisoned status
  __msan_copy_shadow(d->buffer, shadow, d->nbuffer);
  mju_free(shadow);
#endif

  if (filename) {
    fclose(fp);
  }
}


 #ifdef __clang__
 #pragma clang diagnostic pop
 #endif


// print mjData to text file
void mj_printData(const mjModel* m, const mjData* d, const char* filename) {
  mj_printFormattedData(m, d, filename, FLOAT_FORMAT);
}

void mj_printScene(const mjvScene* s, const char* filename) {
  mj_printFormattedScene(s, filename, FLOAT_FORMAT);
}

void mj_printFormattedScene(const mjvScene* s, const char* filename, const char* float_format) {
  // get file
  FILE* fp;
  if (filename) {
    fp = fopen(filename, "wt");
  } else {
    fp = stdout;
  }

  // check for nullptr
  if (!fp) {
    mju_warning("Could not open file '%s' for writing mjModel", filename);
    return;
  }

  // validate format string
  if (!validateFloatFormat(float_format)) {
    mju_warning("WARNING: Received invalid float_format. Using default instead.");
    float_format = FLOAT_FORMAT;
  }

  fprintf(fp, "GEOMS %d\n", s->ngeom);
  for (int i = 0; i < s->ngeom; ++i) {
    const mjvGeom* geom = &s->geoms[i];
    fprintf(fp,  "  GEOM %d\n", i);
    printInt(fp, "    type", geom->type);
    printInt(fp, "    category", geom->category);
    printStr(fp, "    label", geom->label);
    printInt(fp, "    objtype", geom->objtype);
    printInt(fp, "    objid", geom->objid);
    printArr(fp, "    pos", geom->pos, 3, float_format);
    printArr(fp, "    mat", geom->mat, 9, float_format);
    printArr(fp, "    size", geom->size, 3, float_format);
    printInt(fp, "    segid", geom->segid);
    printInt(fp, "    dataid", geom->dataid);
    printInt(fp, "    matid", geom->matid);
    printInt(fp, "    texcoord", geom->texcoord);
    printArr(fp, "    rgba", geom->rgba, 4, float_format);
    printNum(fp, "    emission", geom->emission, float_format);
    printNum(fp, "    specular", geom->specular, float_format);
    printNum(fp, "    shininess", geom->shininess, float_format);
    printNum(fp, "    reflectance", geom->reflectance, float_format);
    printInt(fp, "    castshadow", geom->castshadow);
    fprintf(fp,  "\n");
  }
  fprintf(fp,  "\n");

  fprintf(fp, "LIGHTS %d\n", s->nlight);
  for (int i = 0; i < s->nlight; ++i) {
    const mjvLight* light = &s->lights[i];
    fprintf(fp,  "  LIGHT %d\n", i);
    printInt(fp, "    id", light->id);
    printArr(fp, "    pos", light->pos, 3, float_format);
    printArr(fp, "    dir", light->dir, 3, float_format);
    printInt(fp, "    type", light->type);
    printInt(fp, "    castshadow", light->castshadow);
    printInt(fp, "    headlight", light->headlight);
    printNum(fp, "    intensity", light->intensity, float_format);
    printNum(fp, "    range", light->range, float_format);
    printArr(fp, "    ambient", light->ambient, 3, float_format);
    printArr(fp, "    diffuse", light->diffuse, 3, float_format);
    printArr(fp, "    specular", light->specular, 3, float_format);
    printInt(fp, "    texid", light->texid);
    printNum(fp, "    exponent", light->exponent, float_format);
    printArr(fp, "    attenuation", light->attenuation, 3, float_format);
    printNum(fp, "    cutoff", light->cutoff, float_format);
    printNum(fp, "    bulbradius", light->bulbradius, float_format);
    fprintf(fp,  "\n");
  }
  fprintf(fp,  "\n");

  fprintf(fp, "CAMERAS %d\n", 2);
  for (int i = 0; i < 2; ++i) {
    const mjvGLCamera* camera = &s->camera[i];
    fprintf(fp,  "  CAMERA %d\n", i);
    printArr(fp, "    pos", camera->pos, 3, float_format);
    printArr(fp, "    forward", camera->forward, 3, float_format);
    printArr(fp, "    up", camera->up, 3, float_format);
    printInt(fp, "    orthographic", camera->orthographic);
    printNum(fp, "    frustum_center", camera->frustum_center, float_format);
    printNum(fp, "    frustum_width", camera->frustum_width, float_format);
    printNum(fp, "    frustum_bottom", camera->frustum_bottom, float_format);
    printNum(fp, "    frustum_top", camera->frustum_top, float_format);
    printNum(fp, "    frustum_near", camera->frustum_near, float_format);
    printNum(fp, "    frustum_far", camera->frustum_far, float_format);
    fprintf(fp,  "\n");
  }
  fprintf(fp,  "\n");

  fprintf(fp, "FLEX DATA %d\n", s->nflex);
  for (int i = 0; i < s->nflex; ++i) {
    fprintf(fp,  "  FLEX DATA %d\n", i);
    printInt(fp, "    face_used", s->flexfaceused[i]);
    printInt(fp, "    edge_adr", s->flexedgeadr[i]);
    printInt(fp, "    edge_num", s->flexedgenum[i]);
    printInt(fp, "    vert_adr", s->flexvertadr[i]);
    printInt(fp, "    vert_num", s->flexvertnum[i]);
    printInt(fp, "    face_adr", s->flexfaceadr[i]);
    printInt(fp, "    face_num", s->flexfacenum[i]);
    printStr(fp, "    edges", "...");  // int* flexedge: 2*nflexedge
    printStr(fp, "    verts", "...");  // float* flexvert: 3*nflexvert
    printStr(fp, "    faces", "...");  // float* flexface: 9*sum(flexfacenum)
    printStr(fp, "    normals", "...");  // float* flexnormal: 9*sum(flexfacenum)
    printStr(fp, "    texcoords", "...");  // float* flextexcoord: 6*sum(flexfacenum)
    fprintf(fp,  "\n");
  }
  fprintf(fp,  "\n");

  fprintf(fp, "SKIN DATA %d\n", s->nskin);
  for (int i = 0; i < s->nskin; ++i) {
    fprintf(fp,  "  SKIN DATA %d\n", i);
    printInt(fp, "    face_num", s->skinfacenum[i]);
    printInt(fp, "    vert_adr", s->skinvertadr[i]);
    printInt(fp, "    vert_num", s->skinvertnum[i]);
    printStr(fp, "    verts", "...");  // float* skinvert: 3*nskinvert
    printStr(fp, "    normals", "...");  // float* skinnormal: 3*nskinvert
    fprintf(fp,  "\n");
  }
  fprintf(fp,  "\n");

  fprintf(fp, "FLAGS\n");
  for (int i = 0; i < mjNRNDFLAG; ++i) {
    fprintf(fp, "  ");
    fprintf(fp, NAME_FORMAT, mjRNDSTRING[i][0]);
    fprintf(fp, INT_FORMAT, s->flags[i]);
    fprintf(fp, "\n");
  }
  printInt(fp, "  flexvertopt", s->flexvertopt);
  printInt(fp, "  flexedgeopt", s->flexedgeopt);
  printInt(fp, "  flexfaceopt", s->flexfaceopt);
  printInt(fp, "  flexskinopt", s->flexskinopt);
  printInt(fp, "  stereo", s->stereo);
  fprintf(fp,  "\n\n");

  fprintf(fp, "TRANSFORM %d\n", s->enabletransform);
  if (s->enabletransform) {
    printArr(fp, "  translate", s->translate, 3, float_format);
    printArr(fp, "  rotate", s->rotate, 4, float_format);
    printNum(fp, "  scale", s->scale, float_format);
    fprintf(fp, "\n");
  }
  fprintf(fp, "\n");

  fflush(fp);
  if (filename) {
    fclose(fp);
  }
}
