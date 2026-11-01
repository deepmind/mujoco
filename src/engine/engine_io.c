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

#include "engine/engine_io.h"

#include <inttypes.h>  // IWYU pragma: keep
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjxmacro.h>
#include "engine/engine_core_smooth.h"
#include "engine/engine_forward.h"
#include "engine/engine_init.h"
#include "engine/engine_macro.h"
#include "engine/engine_memory.h"
#include "engine/engine_plugin.h"
#include "engine/engine_sleep.h"
#include "engine/engine_thread.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"

#ifdef mjUSEASAN
  #include <sanitizer/asan_interface.h>
  #include <sanitizer/common_interface_defs.h>
#endif

#ifdef MEMORY_SANITIZER
  #include <sanitizer/msan_interface.h>
#endif

#ifdef _MSC_VER
  #pragma warning (disable: 4305)  // disable MSVC warning: truncation from 'double' to 'float'
#endif

static const int MAX_ARRAY_SIZE = INT_MAX;

// tree_asleep value for fully awake tree
static const int kAwake = -(1+mjMINAWAKE);


//----------------------------------- static utility functions -------------------------------------

// id used to identify binary mjModel file/buffer
static const int ID = 54321;


// number of ints in the mjb header
#define NHEADER 5


// macro for referring to a mjModel member in generic expressions
#define MJMODEL_MEMBER(name) (((mjModel*) NULL)->name)


// count mjtSize members in mjModel
static int getnsize(void) {
  int cnt = 0;

#define X(name) cnt += _Generic(MJMODEL_MEMBER(name), mjtSize: 1, default: 0);
  MJMODEL_SIZES
#undef X

  return cnt;
}


// count pointers in mjModel
static int getnptr(void) {
  int cnt = 0;

#define X(type, name, nr, nc) cnt++;
  MJMODEL_POINTERS
#undef X

  return cnt;
}


// write to memory buffer
static void bufwrite(const void* src, mjtSize num, mjtSize szbuf, void* buf, mjtSize* ptrbuf) {
  // check pointers
  if (!src || !buf || !ptrbuf) {
    mjERROR("NULL pointer passed to bufwrite");
  }

  // check size
  if (num < 0 || szbuf - *ptrbuf < num) {
    mjERROR("attempting to write outside model buffer");
  }

  // write, advance pointer
  memcpy((char*)buf + *ptrbuf, src, num);
  *ptrbuf += num;
}


// read from memory buffer
static void bufread(void* dest, mjtSize num, mjtSize szbuf, const void* buf, mjtSize* ptrbuf) {
  // check pointers
  if (!dest || !buf || !ptrbuf) {
    mjERROR("NULL pointer passed to bufread");
  }

  // check size
  if (num < 0 || szbuf - *ptrbuf < num) {
    mjERROR("attempting to read outside model buffer");
  }

  // read, advance pointer
  memcpy(dest, (char*)buf + *ptrbuf, num);
  *ptrbuf += num;
}


// alignment of the arrays in the mjModel and mjData buffers
#define BUFFER_ALIGN 64


// number of bytes to be skipped to achieve BUFFER_ALIGN-byte alignment
static inline unsigned int SKIP(intptr_t offset) {
  // compute skipped bytes
  return (BUFFER_ALIGN - (offset % BUFFER_ALIGN)) % BUFFER_ALIGN;
}


//----------------------------------- mjModel construction -----------------------------------------

// set pointers in mjModel buffer
void mj_setPtrModel(mjModel* m) {
  char* ptr = (char*)m->buffer;

  // prepare symbols needed by xmacro
  MJMODEL_POINTERS_PREAMBLE(m);

  // assign pointers with padding
#define X(type, name, nr, nc)                             \
  m->name = (type*)(ptr + SKIP((intptr_t)ptr));           \
  ASAN_POISON_MEMORY_REGION(ptr, PTRDIFF(m->name, ptr));  \
  ptr += SKIP((intptr_t)ptr) + sizeof(type)*(m->nr)*(nc);

  MJMODEL_POINTERS
#undef X

  // check size, nbuffer includes the largest possible padding before the first array
  ptrdiff_t sz = ptr - (char*)m->buffer - SKIP((intptr_t)m->buffer) + BUFFER_ALIGN - 1;
  if (m->nbuffer != sz) {
    mjERROR(
        "mjModel buffer size mismatch, "
        "expected size: %" PRIu64 ",  actual size: %td",
        m->nbuffer, sz);
  }

  // poison unused bytes at the end of the buffer
  ASAN_POISON_MEMORY_REGION(ptr, PTRDIFF((char*)m->buffer + m->nbuffer, ptr));
}


// increases buffer size without causing integer overflow, returns 0 if
// operation would cause overflow
// performs the following operations:
// *nbuffer += SKIP(*offset) + type_size*nr*nc;
// *offset += SKIP(*offset) + type_size*nr*nc;
static mjtSize safeAddToBufferSize(intptr_t* offset, mjtSize* nbuffer,
                                   size_t type_size, mjtSize nr, mjtSize nc) {
  if (type_size < 0 || nr < 0 || nc < 0) {
    return 0;
  }
#if (__has_builtin(__builtin_add_overflow) && __has_builtin(__builtin_mul_overflow)) \
  || (defined(__GNUC__) && __GNUC__ >= 5)
  // supported by GCC and Clang
  size_t to_add = 0;
  if (__builtin_mul_overflow(nc, nr, &to_add)) return 0;
  if (__builtin_mul_overflow(to_add, type_size, &to_add)) return 0;
  if (__builtin_add_overflow(to_add, SKIP(*offset), &to_add)) return 0;
  if (__builtin_add_overflow(*nbuffer, to_add, nbuffer)) return 0;
  if (__builtin_add_overflow(*offset, to_add, offset)) return 0;
#else
  // safe overflow checks for MSVC and other compilers without __builtin_*_overflow
  {
    size_t product;
    size_t to_add;
    size_t skip = SKIP(*offset);

    // nc * nr
    if (nr > 0 && (size_t)nc > SIZE_MAX / (size_t)nr) return 0;
    product = (size_t)nc * (size_t)nr;

    // product * type_size
    if (type_size > 0 && product > SIZE_MAX / type_size) return 0;
    product *= type_size;

    // product + SKIP(*offset)
    if (product > SIZE_MAX - skip) return 0;
    to_add = product + skip;

    // *nbuffer + to_add
    if ((size_t)*nbuffer > SIZE_MAX - to_add) return 0;
    *nbuffer += to_add;

    // *offset + to_add
    if (*offset > 0 && to_add > (size_t)(INTPTR_MAX - *offset)) return 0;
    *offset += to_add;
  }
#endif

  return 1;
}


// free model memory without destroying the struct
static void freeModelBuffers(mjModel* m) {
  mju_free(m->buffer);
}


// allocate and initialize mjModel structure
void mj_makeModel(mjModel** dest,
    mjtSize nq, mjtSize nv, mjtSize nu, mjtSize nactuator, mjtSize nout, mjtSize na,
    mjtSize nbody, mjtSize nbvh, mjtSize nbvhstatic,
    mjtSize nbvhdynamic, mjtSize noct, mjtSize njnt, mjtSize ntree, mjtSize nM, mjtSize nB,
    mjtSize nC, mjtSize nD, mjtSize ngeom, mjtSize nsite, mjtSize ncam, mjtSize nlight,
    mjtSize nflex, mjtSize nflexnode, mjtSize nflexvert, mjtSize nflexedge, mjtSize nflexelem,
    mjtSize nflexelemdata, mjtSize nflexstiffness, mjtSize nflexbending,
    mjtSize nefm0dof, mjtSize nefm0L, mjtSize nefmCvert, mjtSize nflexelemedge,
    mjtSize nflexshelldata, mjtSize nflextexcoord, mjtSize nJfe,
    mjtSize nmesh, mjtSize nmeshvert, mjtSize nmeshnormal, mjtSize nmeshtexcoord, mjtSize nmeshface,
    mjtSize nmeshgraph, mjtSize nmeshpoly, mjtSize nmeshpolyvert, mjtSize nmeshpolymap,
    mjtSize nskin, mjtSize nskinvert, mjtSize nskintexvert, mjtSize nskinface, mjtSize nskinbone,
    mjtSize nskinbonevert, mjtSize nhfield, mjtSize nhfielddata, mjtSize ntex, mjtSize ntexdata,
    mjtSize nmat, mjtSize npair, mjtSize nexclude, mjtSize neq, mjtSize ntendon, mjtSize nJten,
    mjtSize nwrap, mjtSize nsensor, mjtSize nnumeric, mjtSize nnumericdata, mjtSize ntext,
    mjtSize ntextdata, mjtSize ntuple, mjtSize ntupledata, mjtSize nkey, mjtSize nmocap,
    mjtSize nplugin, mjtSize npluginattr, mjtSize nuser_body, mjtSize nuser_jnt, mjtSize nuser_geom,
    mjtSize nuser_site, mjtSize nuser_cam, mjtSize nuser_tendon, mjtSize nuser_actuator,
    mjtSize nuser_sensor, mjtSize nnames, mjtSize npaths, mjtSize nplastic) {
  intptr_t offset = 0;
  int allocate = *dest ? 0 : 1;
  mjModel* m = NULL;

  // CHECK SIZE PARAMETERS
  {
    // dummy variables for MJMODEL_SIZES set after mjModel construction
    int nnames_map = 0, nJmom = 0, ngravcomp = 0, nemax = 0, njmax = 0, nconmax=0;
    int npolygonmax = 0, nmeshdegmax = 0;
    int nuserdata=0, nsensordata=0, npluginstate=0, nhistory=0, narena=0, nbuffer=0;

    // sizes must be non-negative and fit in int, except for the byte arrays texdata and textdata
    #define X(name)                                                                   \
      if (name < 0) {                                                                 \
        mju_warning("Invalid model: %s is negative (%lld).", #name, (long long)name); \
        return;                                                                       \
      }                                                                               \
      if (name >= MAX_ARRAY_SIZE &&                                                   \
          strcmp(#name, "ntexdata") != 0 && strcmp(#name, "ntextdata") != 0) {        \
        mju_warning("Invalid model: %s is too large. Expected < %d. Got %lld.",       \
                    #name, MAX_ARRAY_SIZE, (long long)name);                          \
        return;                                                                       \
      }
    MJMODEL_SIZES
#undef X

    // suppress unused variable warnings
    (void)nnames_map; (void)nJmom; (void)ngravcomp; (void)nemax; (void)njmax; (void)nconmax;
    (void)npolygonmax; (void)nmeshdegmax;
    (void)nuserdata; (void)nsensordata; (void)npluginstate; (void)nhistory;
    (void)narena; (void)nbuffer;
  }

  // nbody should always be positive
  if (nbody == 0) {
    mju_warning("Invalid model: nbody == 0");
    return;
  }

  // allocate mjModel
  if (!allocate) {
    m = *dest;
    freeModelBuffers(m);
  } else {
    m = (mjModel*)mju_malloc(sizeof(mjModel));
  }

  if (!m) {
    mjERROR("could not allocate mjModel");
  }
  memset(m, 0, sizeof(mjModel));

  // set size parameters
  m->nq = nq;
  m->nv = nv;
  m->nu = nu;
  m->nactuator = nactuator;
  m->nout = nout;
  m->na = na;
  m->nbody = nbody;
  m->nbvh = nbvh;
  m->nbvhstatic = nbvhstatic;
  m->nbvhdynamic = nbvhdynamic;
  m->noct = noct;
  m->njnt = njnt;
  m->ntree = ntree;
  m->nM = nM;
  m->nB = nB;
  m->nC = nC;
  m->nD = nD;
  m->ngeom = ngeom;
  m->nsite = nsite;
  m->ncam = ncam;
  m->nlight = nlight;
  m->nflex = nflex;
  m->nflexnode = nflexnode;
  m->nflexvert = nflexvert;
  m->nflexedge = nflexedge;
  m->nflexelem = nflexelem;
  m->nflexelemdata = nflexelemdata;
  m->nflexstiffness = nflexstiffness;
  m->nflexbending = nflexbending;
  m->nefm0dof = nefm0dof;
  m->nefm0L = nefm0L;
  m->nefmCvert = nefmCvert;
  m->nflexelemedge = nflexelemedge;
  m->nflexshelldata = nflexshelldata;
  m->nflextexcoord = nflextexcoord;
  m->nJfe = nJfe;
  m->nmesh = nmesh;
  m->nmeshvert = nmeshvert;
  m->nmeshnormal = nmeshnormal;
  m->nmeshtexcoord = nmeshtexcoord;
  m->nmeshface = nmeshface;
  m->nmeshgraph = nmeshgraph;
  m->nmeshpoly = nmeshpoly;
  m->nmeshpolyvert = nmeshpolyvert;
  m->nmeshpolymap = nmeshpolymap;
  m->nskin = nskin;
  m->nskinvert = nskinvert;
  m->nskintexvert = nskintexvert;
  m->nskinface = nskinface;
  m->nskinbone = nskinbone;
  m->nskinbonevert = nskinbonevert;
  m->nhfield = nhfield;
  m->nhfielddata = nhfielddata;
  m->ntex = ntex;
  m->ntexdata = ntexdata;
  m->nmat = nmat;
  m->npair = npair;
  m->nexclude = nexclude;
  m->neq = neq;
  m->ntendon = ntendon;
  m->nJten = nJten;
  m->nwrap = nwrap;
  m->nsensor = nsensor;
  m->nnumeric = nnumeric;
  m->nnumericdata = nnumericdata;
  m->ntext = ntext;
  m->ntextdata = ntextdata;
  m->ntuple = ntuple;
  m->ntupledata = ntupledata;
  m->nkey = nkey;
  m->nmocap = nmocap;
  m->nplugin = nplugin;
  m->npluginattr = npluginattr;
  m->nuser_body = nuser_body;
  m->nuser_jnt = nuser_jnt;
  m->nuser_geom = nuser_geom;
  m->nuser_site = nuser_site;
  m->nuser_cam = nuser_cam;
  m->nuser_tendon = nuser_tendon;
  m->nuser_actuator = nuser_actuator;
  m->nuser_sensor = nuser_sensor;
  m->nnames = nnames;
  long nnames_map = (long)nbody + njnt + ngeom + nsite + ncam + nlight + nflex + nmesh + nskin +
                    nhfield + ntex + nmat + npair + nexclude + neq + ntendon + nactuator + nsensor +
                    nnumeric + ntext + ntuple + nkey + nplugin;
  if (nnames_map >= INT_MAX / mjLOAD_MULTIPLE) {
    if (allocate) mju_free(m);
    mju_warning("Invalid model: size of nnames_map is larger than INT_MAX");
    return;
  }
  m->nnames_map = mjLOAD_MULTIPLE * nnames_map;
  m->npaths = npaths;
  m->nplastic = nplastic;

  // compute buffer size, including padding to align the first array at any buffer address
  m->nbuffer = BUFFER_ALIGN - 1;
#define X(type, name, nr, nc)                                                \
  if (!safeAddToBufferSize(&offset, &m->nbuffer, sizeof(type), m->nr, nc)) { \
    if (allocate) mju_free(m);                                               \
    mju_warning("Invalid model: " #name " too large.");                      \
    return;                                                                  \
  }

  MJMODEL_POINTERS
#undef X

  // allocate buffer
  m->buffer = mju_malloc(m->nbuffer);
  if (!m->buffer) {
    if (allocate) mju_free(m);
    mjERROR("could not allocate mjModel buffer");
  }

  // clear, set pointers in buffer
  memset(m->buffer, 0, m->nbuffer);
#ifdef MEMORY_SANITIZER
  // tell msan to treat the entire buffer as uninitialized
  __msan_allocated_memory(m->buffer, m->nbuffer);
#endif
  mj_setPtrModel(m);

  // set default options
  mj_defaultOption(&m->opt);
  mj_defaultVisual(&m->vis);
  mj_defaultStatistic(&m->stat);

  // copy pointer if allocated here
  if (allocate) {
    *dest = m;
  }
}


// copy mjModel, if dest==NULL create new model
mjModel* mj_copyModel(mjModel* dest, const mjModel* src) {
  // allocate new model if needed
  if (!dest) {
    mj_makeModel(
        &dest, src->nq, src->nv, src->nu, src->nactuator, src->nout, src->na,
        src->nbody, src->nbvh, src->nbvhstatic,
        src->nbvhdynamic, src->noct, src->njnt, src->ntree, src->nM, src->nB, src->nC, src->nD,
        src->ngeom, src->nsite, src->ncam, src->nlight, src->nflex, src->nflexnode, src->nflexvert,
        src->nflexedge, src->nflexelem, src->nflexelemdata, src->nflexstiffness,
        src->nflexbending, src->nefm0dof, src->nefm0L, src->nefmCvert, src->nflexelemedge,
        src->nflexshelldata,
        src->nflextexcoord, src->nJfe, src->nmesh, src->nmeshvert, src->nmeshnormal,
        src->nmeshtexcoord, src->nmeshface, src->nmeshgraph, src->nmeshpoly, src->nmeshpolyvert,
        src->nmeshpolymap, src->nskin, src->nskinvert, src->nskintexvert, src->nskinface,
        src->nskinbone, src->nskinbonevert, src->nhfield, src->nhfielddata, src->ntex,
        src->ntexdata, src->nmat, src->npair, src->nexclude, src->neq, src->ntendon, src->nJten,
        src->nwrap, src->nsensor, src->nnumeric, src->nnumericdata, src->ntext, src->ntextdata,
        src->ntuple, src->ntupledata, src->nkey, src->nmocap, src->nplugin, src->npluginattr,
        src->nuser_body, src->nuser_jnt, src->nuser_geom, src->nuser_site, src->nuser_cam,
        src->nuser_tendon, src->nuser_actuator, src->nuser_sensor, src->nnames, src->npaths,
        src->nplastic);
  }
  if (!dest) {
    mjERROR("failed to make mjModel. Invalid sizes.");
  }

  // check sizes
  if (dest->nbuffer != src->nbuffer) {
    mj_deleteModel(dest);
    mjERROR("dest and src models have different buffer size");
  }

  // save buffer ptr, copy struct, restore buffer and other pointers
  void* save_bufptr = dest->buffer;
  *dest = *src;
  dest->buffer = save_bufptr;
  mj_setPtrModel(dest);

  // copy buffer contents
  {
    MJMODEL_POINTERS_PREAMBLE(src)
    #define X(type, name, nr, nc)  \
      memcpy((char*)dest->name, (const char*)src->name, sizeof(type)*(src->nr)*nc);
    MJMODEL_POINTERS
    #undef X
  }

  return dest;
}


// copy mjModel, skip large arrays not required for abstract visualization
void mjv_copyModel(mjModel* dest, const mjModel* src) {
  // check sizes
  if (dest->nbuffer != src->nbuffer) {
    mjERROR("dest and src models have different buffer size");
  }

  // save buffer ptr, copy struct, restore buffer and other pointers
  void* save_bufptr = dest->buffer;
  *dest = *src;
  dest->buffer = save_bufptr;
  mj_setPtrModel(dest);

  // redefine XNV to do nothing
  #undef XNV
  #define XNV(type, name, nr, nc)

  // copy buffer contents, skipping arrays marked XNV
  {
    MJMODEL_POINTERS_PREAMBLE(src)
    #define X(type, name, nr, nc)    \
      memcpy((char*)dest->name, (const char*)src->name, sizeof(type)*(src->nr)*nc);
    MJMODEL_POINTERS
    #undef X
  }
  // redefine XNV to be the same as X
  #undef XNV
  #define XNV X
}


// save model to binary file, or memory buffer of szbuf>0
void mj_saveModel(const mjModel* m, const char* filename, void* buffer, mjtSize buffer_sz) {
  mjtSize ptrbuf = 0;

  // standard header
  int header[NHEADER] = {ID, sizeof(mjtNum), getnsize(), mj_version(), getnptr()};

  // no buffer: serialize to temporary buffer, then write via resource provider
  if (!buffer) {
    mjtSize sz = mj_sizeModel(m);
    void* tmpbuf = mju_malloc(sz);
    if (!tmpbuf) {
      mju_warning("Could not allocate buffer for saving model");
      return;
    }
    mj_saveModel(m, NULL, tmpbuf, sz);

    mjtSize written = mju_writeResource(filename, tmpbuf, sz, NULL, NULL, 0);
    if (written != sz) {
      mju_warning("Could not save model to '%s'", filename);
    }
    mju_free(tmpbuf);
    return;
  }

  // write standard header, info, options, buffer (omit pointers)
  bufwrite(header, sizeof(header), buffer_sz, buffer, &ptrbuf);
  #define X(name) bufwrite(&m->name, sizeof(m->name), buffer_sz, buffer, &ptrbuf);
  MJMODEL_SIZES
  #undef X
  bufwrite((void*)&m->opt, sizeof(mjOption), buffer_sz, buffer, &ptrbuf);
  bufwrite((void*)&m->vis, sizeof(mjVisual), buffer_sz, buffer, &ptrbuf);
  bufwrite((void*)&m->stat, sizeof(mjStatistic), buffer_sz, buffer, &ptrbuf);
  bufwrite(&m->flg_gravcomp, sizeof(mjtBool), buffer_sz, buffer, &ptrbuf);
  bufwrite(&m->flg_surfacevel, sizeof(mjtBool), buffer_sz, buffer, &ptrbuf);
  bufwrite(&m->flg_adhesion, sizeof(mjtBool), buffer_sz, buffer, &ptrbuf);
  {
    MJMODEL_POINTERS_PREAMBLE(m)
    #define X(type, name, nr, nc)  \
      bufwrite((void*)m->name, sizeof(type)*(m->nr)*(nc), buffer_sz, buffer, &ptrbuf);
    MJMODEL_POINTERS
    #undef X
  }
}


// load binary MJB model
mjModel* mj_loadModelBuffer(const void* buffer, mjtSize buffer_sz) {
  mjtSize ptrbuf = 0;
  mjModel *m = 0;

  if (buffer_sz < NHEADER*sizeof(int)) {
    mju_warning("Model file has an incomplete header");
    return NULL;
  }

  int header[NHEADER] = {0};
  bufread(header, NHEADER*sizeof(int), buffer_sz, buffer, &ptrbuf);

  // check header
  int expected_header[NHEADER] = {ID, sizeof(mjtNum), getnsize(), mj_version(), getnptr()};
  for (int i=0; i < NHEADER; i++) {
    if (header[i] != expected_header[i]) {
      switch (i) {
      case 0:
        mju_warning("Model missing header ID");
        return NULL;

      case 1:
        mju_warning("Model and executable have different floating point precision");
        return NULL;

      case 2:
        mju_warning("Model and executable have different number of sizes in mjModel");
        return NULL;

      case 3:
        mju_warning("Model and executable use different MuJoCo version");
        return NULL;

      default:
        mju_warning("Model and executable have different number of pointers in mjModel");
        return NULL;
      }
    }
  }

  int nsize = getnsize();  // number of mjtSize fields in mjModel

  if (ptrbuf + sizeof(mjtSize)*nsize > buffer_sz) {
    mju_warning("Truncated model file - ran out of data while reading sizes");
    return NULL;
  }

  // read mjModel construction fields
  mjtSize sizes[256];
  bufread(sizes, sizeof(mjtSize)*nsize, buffer_sz, buffer, &ptrbuf);

  // allocate new mjModel
  mj_makeModel(&m,
               sizes[0],  sizes[1],  sizes[2],  sizes[3],  sizes[4],  sizes[5],  sizes[6],
               sizes[7],  sizes[8],  sizes[9],  sizes[10], sizes[11], sizes[12], sizes[13],
               sizes[14], sizes[15], sizes[16], sizes[17], sizes[18], sizes[19], sizes[20],
               sizes[21], sizes[22], sizes[23], sizes[24], sizes[25], sizes[26], sizes[27],
               sizes[28], sizes[29], sizes[30], sizes[31], sizes[32], sizes[33], sizes[34],
               sizes[35], sizes[36], sizes[37], sizes[38], sizes[39], sizes[40], sizes[41],
               sizes[42], sizes[43], sizes[44], sizes[45], sizes[46], sizes[47], sizes[48],
               sizes[49], sizes[50], sizes[51], sizes[52], sizes[53], sizes[54], sizes[55],
               sizes[56], sizes[57], sizes[58], sizes[59], sizes[60], sizes[61], sizes[62],
               sizes[63], sizes[64], sizes[65], sizes[66], sizes[67], sizes[68], sizes[69],
               sizes[70], sizes[71], sizes[72], sizes[73], sizes[74], sizes[75], sizes[76],
               sizes[77], sizes[78], sizes[79], sizes[80], sizes[81], sizes[82], sizes[83]);

  // mj_makeModel may fail if the input buffer has invalid sizes
  if (!m) {
    mju_warning("Invalid sizes, unable to load model");
    return NULL;
  }

  // check buffer size (last mjtSize field is nbuffer)
  if (m->nbuffer != sizes[nsize-1]) {
    mju_warning("Corrupted model, wrong nbuffer field");
    mj_deleteModel(m);
    return NULL;
  }

  // set integer fields
  {
    int int_idx = 0;
    #define X(name) \
        m->name = sizes[int_idx++];
    MJMODEL_SIZES
    #undef X
  }

  // read options and buffer
  if (ptrbuf + sizeof(mjOption) + sizeof(mjVisual) + sizeof(mjStatistic) +
      sizeof(mjtBool) * 3 > buffer_sz) {
    mju_warning("Truncated model file - ran out of data while reading structs");
    mj_deleteModel(m);
    return NULL;
  }
  bufread((void*)&m->opt, sizeof(mjOption), buffer_sz, buffer, &ptrbuf);
  bufread((void*)&m->vis, sizeof(mjVisual), buffer_sz, buffer, &ptrbuf);
  bufread((void*)&m->stat, sizeof(mjStatistic), buffer_sz, buffer, &ptrbuf);
  bufread(&m->flg_gravcomp, sizeof(mjtBool), buffer_sz, buffer, &ptrbuf);
  bufread(&m->flg_surfacevel, sizeof(mjtBool), buffer_sz, buffer, &ptrbuf);
  bufread(&m->flg_adhesion, sizeof(mjtBool), buffer_sz, buffer, &ptrbuf);
  {
    MJMODEL_POINTERS_PREAMBLE(m)
    #define X(type, name, nr, nc)                                           \
      if (buffer_sz - ptrbuf < sizeof(type) * (m->nr) * (nc)) {             \
        mju_warning(                                                        \
            "Truncated model file - ran out of data while reading " #name); \
        mj_deleteModel(m);                                                  \
        return NULL;                                                        \
      }                                                                     \
      bufread(m->name, sizeof(type)*(m->nr)*(nc), buffer_sz, buffer, &ptrbuf);

    MJMODEL_POINTERS
    #undef X
  }

  // make sure buffer is the correct size
  if (ptrbuf != buffer_sz) {
    mju_warning("Model file is too large");
    mj_deleteModel(m);
    return NULL;
  }

  const char* validationError = mj_validateReferences(m);
  if (validationError) {
    mju_warning("%s", validationError);
    mj_deleteModel(m);
    return NULL;
  }

  return m;
}


// de-allocate mjModel
void mj_deleteModel(mjModel* m) {
  if (m) {
    freeModelBuffers(m);
    mju_free(m);
  }
}


// size of buffer needed to hold model
mjtSize mj_sizeModel(const mjModel* m) {
  mjtSize size = (
    sizeof(int)*NHEADER
    + sizeof(mjtSize)*getnsize()
    + sizeof(mjOption)
    + sizeof(mjVisual)
    + sizeof(mjStatistic)
    + sizeof(mjtBool)*3);

  MJMODEL_POINTERS_PREAMBLE(m)
#define X(type, name, nr, nc)         \
  size += sizeof(type)*(m->nr)*(nc);
  MJMODEL_POINTERS
#undef X

  return size;
}


//-------------------------- sparse system matrix construction -------------------------------------

// construct sparse representation of dof-dof matrix
void mj_makeDofDofSparse(int nv, int nC, int nD, int nM,
                         const int* dof_parentid, const int* dof_simplenum,
                         int* rownnz, int* rowadr, int* diag, int* colind,
                         int reduced, int upper,
                         int* remaining) {
  // no dofs, nothing to do
  if (!nv) {
    return;
  }

  // compute rownnz
  mju_zeroInt(rownnz, nv);
  for (int i = nv - 1; i >= 0; i--) {
    // init at diagonal
    int j = i;
    rownnz[i]++;

    // process below diagonal unless reduced and dof is simple
    if (!(reduced && dof_simplenum[i])) {
      while ((j = dof_parentid[j]) >= 0) {
        // both reduced and non-reduced have lower triangle
        rownnz[i]++;

        // add upper triangle if requested
        if (upper) rownnz[j]++;
      }
    }
  }

  // accumulate rowadr
  rowadr[0] = 0;
  for (int i = 1; i < nv; i++) {
    rowadr[i] = rowadr[i - 1] + rownnz[i - 1];
  }

  // populate colind
  mju_copyInt(remaining, rownnz, nv);
  for (int i = nv - 1; i >= 0; i--) {
    // init at diagonal
    remaining[i]--;
    colind[rowadr[i] + remaining[i]] = i;

    // process below diagonal unless reduced and dof is simple
    if (!(reduced && dof_simplenum[i])) {
      int j = i;
      while ((j = dof_parentid[j]) >= 0) {
        remaining[i]--;
        colind[rowadr[i] + remaining[i]] = j;

        // add upper triangle if requested
        if (upper) {
          remaining[j]--;
          colind[rowadr[j] + remaining[j]] = i;
        }
      }
    }
  }

  // check for remaining; SHOULD NOT OCCUR
  for (int i = 0; i < nv; i++) {
    if (remaining[i] != 0) {
      mjERROR("unexpected remaining");
    }
  }

  // check total nnz; SHOULD NOT OCCUR
  int expected_nnz = upper ? nD : (reduced ? nC : nM);
  if (rowadr[nv - 1] + rownnz[nv - 1] != expected_nnz) {
    mjERROR("sum of rownnz different from expected");
  }

  // find diagonal indices
  if (diag) {
    for (int i = 0; i < nv; i++) {
      int adr = rowadr[i];
      int j = 0;
      while (colind[adr + j] < i && j < rownnz[i]) {
        j++;
      }
      if (colind[adr + j] != i) {
        mjERROR("diagonal index not found");
      }
      diag[i] = j;
    }
  }
}

// construct sparse representation of body-dof matrix
void mj_makeBSparse(int nv, int nbody, int nB,
                    const int* body_dofnum, const int* body_parentid, const int* body_dofadr,
                    int* B_rownnz, int* B_rowadr, int* B_colind,
                    int* count) {
  // set rownnz to subtree dofs counts, including self
  mju_zeroInt(B_rownnz, nbody);
  for (int i = nbody - 1; i > 0; i--) {
    B_rownnz[i] += body_dofnum[i];
    B_rownnz[body_parentid[i]] += B_rownnz[i];
  }

  // check if rownnz[0] != nv; SHOULD NOT OCCUR
  if (B_rownnz[0] != nv) {
    mjERROR("rownnz[0] different from nv");
  }

  // add dofs in ancestors bodies
  for (int i = 0; i < nbody; i++) {
    int j = body_parentid[i];
    while (j > 0) {
      B_rownnz[i] += body_dofnum[j];
      j = body_parentid[j];
    }
  }

  // compute rowadr
  B_rowadr[0] = 0;
  for (int i = 1; i < nbody; i++) {
    B_rowadr[i] = B_rowadr[i - 1] + B_rownnz[i - 1];
  }

  // check if total nnz != nB; SHOULD NOT OCCUR
  if (nB != B_rowadr[nbody - 1] + B_rownnz[nbody - 1]) {
    mjERROR("sum of rownnz different from nB");
  }

  // clear incremental row counts
  mju_zeroInt(count, nbody);

  // add subtree dofs to colind
  for (int i = nbody - 1; i > 0; i--) {
    // add this body's dofs to subtree
    for (int n = 0; n < body_dofnum[i]; n++) {
      B_colind[B_rowadr[i] + count[i]] = body_dofadr[i] + n;
      count[i]++;
    }

    // add body subtree to parent
    int par = body_parentid[i];
    for (int n = 0; n < count[i]; n++) {
      B_colind[B_rowadr[par] + count[par]] = B_colind[B_rowadr[i] + n];
      count[par]++;
    }
  }

  // add all ancestor dofs
  for (int i = 0; i < nbody; i++) {
    int par = body_parentid[i];
    while (par > 0) {
      // add ancestor body dofs
      for (int n = 0; n < body_dofnum[par]; n++) {
        B_colind[B_rowadr[i] + count[i]] = body_dofadr[par] + n;
        count[i]++;
      }

      // advance to parent
      par = body_parentid[par];
    }
  }

  // process all bodies
  for (int i = 0; i < nbody; i++) {
    // make sure cnt = rownnz; SHOULD NOT OCCUR
    if (B_rownnz[i] != count[i]) {
      mjERROR("cnt different from rownnz");
    }

    // sort colind in each row
    if (count[i] > 1) {
      mju_insertionSortInt(B_colind + B_rowadr[i], count[i]);
    }
  }
}


// check D and B sparsity for consistency
static void checkDBSparse(const mjModel* m) {
  // process all dofs
  for (int j = 0; j < m->nv; j++) {
    // get body for this dof
    int i = m->dof_bodyid[j];

    // D[row j] and B[row i] should be identical
    if (m->D_rownnz[j] != m->B_rownnz[i]) {
      mjERROR("rows have different nnz");
    }
    for (int k = 0; k < m->D_rownnz[j]; k++) {
      if (m->D_colind[m->D_rowadr[j] + k] != m->B_colind[m->B_rowadr[i] + k]) {
        mjERROR("rows have different colind");
      }
    }
  }
}


// integer valued dst[D or C or M] = src[M (legacy)], handle different sparsity representations
static void copyM2Sparse(int nv,
                         const int* dof_Madr, const int* dof_simplenum, const int* dof_parentid,
                         const int* rownnz, const int* rowadr, const int* src,
                         int* dst,
                         int reduced, int upper, int* remaining) {
  // init remaining
  mju_copyInt(remaining, rownnz, nv);

  // copy data
  for (int i = nv - 1; i >= 0; i--) {
    // init at diagonal
    int adr = dof_Madr[i];
    remaining[i]--;
    dst[rowadr[i] + remaining[i]] = src[adr];
    adr++;

    // process below diagonal unless reduced and dof is simple
    if (!(reduced && dof_simplenum[i])) {
      int j = i;
      while ((j = dof_parentid[j]) >= 0) {
        remaining[i]--;
        dst[rowadr[i] + remaining[i]] = src[adr];

        // add upper triangle if requested
        if (upper) {
          remaining[j]--;
          dst[rowadr[j] + remaining[j]] = src[adr];
        }

        adr++;
      }
    }
  }

  // check that none remaining
  for (int i=0; i < nv; i++) {
    if (remaining[i]) {
      mjERROR("unassigned index");
    }
  }
}


// construct index mappings between M <-> D, M -> C, M (legacy) -> M (CSR)
void mj_makeDofDofMaps(int nv, int nM, int nC, int nD,
                       const int* dof_Madr, const int* dof_simplenum, const int* dof_parentid,
                       const int* D_rownnz, const int* D_rowadr, const int* D_colind,
                       const int* M_rownnz, const int* M_rowadr, const int* M_colind,
                       int* mapM2D, int* mapD2M, int* mapM2M,
                       int* M, int* scratch) {
  // make mapM2D: M -> D (lower to symmetric)
  mju_lower2SymMap(mapM2D, nv, D_rowadr, D_rownnz, D_colind, M_rowadr, M_rownnz, M_colind, scratch);

  // make mapD2M: D -> M (symmetric to lower)
  mju_sparseMap(mapD2M, nv, M_rowadr, M_rownnz, M_colind, D_rowadr, D_rownnz, D_colind);

  // make mapM2M
  for (int i=0; i < nM; i++) M[i] = i;
  mju_fillInt(mapM2M, -1, nC);
  copyM2Sparse(nv, dof_Madr, dof_simplenum, dof_parentid, M_rownnz,
               M_rowadr, M, mapM2M, /*reduced=*/1, /*upper=*/0, scratch);

  // check that all indices are filled in
  for (int i=0; i < nC; i++) {
    if (mapM2M[i] < 0) {
      mjERROR("unassigned index in mapM2M");
    }
  }
}


//----------------------------------- mjData construction ------------------------------------------

// set pointers into mjData buffer
static void mj_setPtrData(const mjModel* m, mjData* d) {
  char* ptr = (char*)d->buffer;

  // assign pointers with padding
#define X(type, name, nr, nc)                             \
  d->name = (type*)(ptr + SKIP((intptr_t)ptr));           \
  ASAN_POISON_MEMORY_REGION(ptr, PTRDIFF(d->name, ptr));  \
  ptr += SKIP((intptr_t)ptr) + sizeof(type)*(m->nr)*(nc);

  MJDATA_POINTERS
#undef X

  // check size, nbuffer includes the largest possible padding before the first array
  ptrdiff_t sz = ptr - (char*)d->buffer - SKIP((intptr_t)d->buffer) + BUFFER_ALIGN - 1;
  if (d->nbuffer != sz) {
    mjERROR(
        "mjData buffer size mismatch, "
        "expected size: %" PRIu64 ",  actual size: %td",
        d->nbuffer, sz);
  }

  // poison unused bytes at the end of the buffer
  ASAN_POISON_MEMORY_REGION(ptr, PTRDIFF((char*)d->buffer + d->nbuffer, ptr));

  // zero-initialize arena pointers
#define X(type, name, nr, nc) d->name = NULL;
  MJDATA_ARENA_POINTERS
#undef X

  d->contact = d->arena;
}


// initialize plugins, copy into d (required for deletion); if an init callback fails, destroy
// the instances already initialized and return the id of the failed instance, otherwise -1
int mj_initPlugin(const mjModel* m, mjData* d) {
  d->nplugin = m->nplugin;
  for (int i = 0; i < m->nplugin; ++i) {
    d->plugin[i] = m->plugin[i];
    d->plugin_data[i] = 0;
    const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[i]);
    if (plugin->init && plugin->init(m, d, i) < 0) {
      // destroy previously initialized instances
      for (int j=0; j < i; j++) {
        const mjpPlugin* prev = mjp_getPluginAtSlot(d->plugin[j]);
        if (prev->destroy) {
          prev->destroy(d, j);
        }
      }
      d->nplugin = 0;
      return i;
    }
  }
  return -1;
}


// initialize plugins of mjData allocated by the caller, free it and error on failure
static void initPluginOrFree(const mjModel* m, mjData* d) {
  int failed = mj_initPlugin(m, d);
  if (failed >= 0) {
    mju_free(d->buffer);
    mju_free(d->arena);
    mju_free(d);
    mjERROR("plugin->init failed for plugin id %d", failed);
  }
}


// free mjData memory without destroying the struct
static void freeDataBuffers(mjData* d) {
#ifdef mjUSEASAN
    // raise an error if there's a dangling stack frame
    mj_freeStack(d);
#endif

    // destroy plugin instances
    for (int i = 0; i < d->nplugin; ++i) {
      const mjpPlugin* plugin = mjp_getPluginAtSlot(d->plugin[i]);
      if (plugin->destroy) {
        plugin->destroy(d, i);
      }
    }
    mju_free(d->buffer);
    mju_free(d->arena);
}


// allocate and initialize raw mjData structure
void mj_makeRawData(mjData** dest, const mjModel* m) {
  intptr_t offset = 0;
  int allocate = *dest ? 0 : 1;
  mjData* d = NULL;
  uintptr_t save_threadpool = 0;

  // allocate mjData
  if (!allocate) {
    d = *dest;
    save_threadpool = d->threadpool;
    freeDataBuffers(d);
  } else {
    d = (mjData*) mju_malloc(sizeof(mjData));
  }

  if (!d) {
    mjERROR("could not allocate mjData");
  }

  // prevent spurious timing print from mj_resetData before _resetData zeroes the struct
  d->timer[mjTIMER_STEP].number = 0;

  // compute buffer size, including padding to align the first array at any buffer address
  d->nbuffer = BUFFER_ALIGN - 1;
  d->buffer = d->arena = NULL;
#define X(type, name, nr, nc)                                                \
  if (!safeAddToBufferSize(&offset, &d->nbuffer, sizeof(type), m->nr, nc)) { \
    if (allocate) mju_free(d);                                               \
    mju_warning("Invalid data: " #name " too large.");                       \
    return;                                                                  \
  }

  MJDATA_POINTERS
#undef X

  // copy stack size from model
  d->narena = m->narena;

  // allocate buffer
  d->buffer = mju_malloc(d->nbuffer);
  if (!d->buffer) {
    if (allocate) mju_free(d);
    mjERROR("could not allocate mjData buffer");
  }

  // allocate arena
  d->arena = mju_malloc(d->narena);
  if (!d->arena) {
    mju_free(d->buffer);
    if (allocate) mju_free(d);
    mjERROR("could not allocate mjData arena");
  }

  // set pointers into buffer
  mj_setPtrData(m, d);

  // preserve or clear threadpool
  d->threadpool = save_threadpool;
  d->threadlock = 0;

  // clear nplugin (overwritten by _initPlugin)
  d->nplugin = 0;

  // set all trees awake, treat static bodies as awake: plugin init can call mj_kinematics
  mju_fillInt(d->tree_asleep, kAwake, m->ntree);
  mj_updateSleepInit(m, d, /*flg_staticawake*/ 1);

  // copy pointer if allocated here
  if (allocate) {
    *dest = d;
  }
}


// allocate and initialize mjData structure
mjData* mj_makeData(const mjModel* m) {
  mjData* d = NULL;
  mj_makeRawData(&d, m);
  if (d) {
    initPluginOrFree(m, d);
    mj_resetData(m, d);
  }
  return d;
}


// copy mjData, if dest==NULL create new data;
// flg_all  1: copy all fields,  0: skip fields not required for visualization
mjData* mj_copyDataVisual(mjData* dest, const mjModel* m, const mjData* src, int flg_all) {
  void* save_buffer;
  void* save_arena;

  // allocate new data if needed
  if (!dest) {
    mj_makeRawData(&dest, m);
    initPluginOrFree(m, dest);
  }

  // check sizes
  if (dest->nbuffer != src->nbuffer) {
    mjERROR("dest and src data buffers have different size");
  }
  if (dest->narena != src->narena) {
    mjERROR("dest and src stacks have different size");
  }

  // stack is in use
  if (src->pstack) {
    mjERROR("attempting to copy mjData while stack is in use");
  }

  // save pointers, copy everything, restore pointers
  save_buffer = dest->buffer;
  save_arena = dest->arena;
  uintptr_t save_threadpool = dest->threadpool;
  *dest = *src;
  dest->buffer = save_buffer;
  dest->arena = save_arena;
  dest->threadpool = save_threadpool;
  dest->threadlock = 0;
  mj_setPtrData(m, dest);

  // save plugin_data, since the X macro copying block below will override it
  const size_t plugin_data_size = sizeof(*dest->plugin_data) * dest->nplugin;
  uintptr_t* save_plugin_data = NULL;
  if (plugin_data_size) {
    save_plugin_data = (uintptr_t*)mju_malloc(plugin_data_size);
    if (!save_plugin_data) {
      mjERROR("failed to allocate temporary memory for plugin_data");
    }
    memcpy(save_plugin_data, dest->plugin_data, plugin_data_size);
  }

  // copy buffer
  {
    if (flg_all) {
      #define X(type, name, nr, nc)  \
        memcpy((char*)dest->name, (const char*)src->name, sizeof(type)*(m->nr)*nc);
      MJDATA_POINTERS
      #undef X
    } else {
      // redefine XNV to nothing
      #undef XNV
      #define XNV(type, name, nr, nc)

      #define X(type, name, nr, nc)  \
        memcpy((char*)dest->name, (const char*)src->name, sizeof(type)*(m->nr)*nc);
      MJDATA_POINTERS
      #undef X

      // redefine XNV to be the same as X
      #undef XNV
      #define XNV X
    }
  }


  // copy arena memory
  #undef MJ_D
  #define MJ_D(n) (src->n)
  #undef MJ_M
  #define MJ_M(n) (m->n)

  if (flg_all) {
    #define X(type, name, nr, nc)                                                \
    if (src->name) {                                                             \
      dest->name = (type*)((char*)dest->arena + PTRDIFF(src->name, src->arena)); \
      ASAN_UNPOISON_MEMORY_REGION(dest->name, sizeof(type) * nr * nc);           \
      memcpy((char*)dest->name, (const char*)src->name, sizeof(type) * nr * nc); \
    } else {                                                                     \
      dest->name = NULL;                                                         \
    }
    MJDATA_ARENA_POINTERS
    #undef X
  } else {
    // redefine XNV to nothing
    #undef XNV
    #define XNV(type, name, nr, nc)

    #define X(type, name, nr, nc)                                                \
    if (src->name) {                                                             \
      dest->name = (type*)((char*)dest->arena + PTRDIFF(src->name, src->arena)); \
      ASAN_UNPOISON_MEMORY_REGION(dest->name, sizeof(type) * nr * nc);           \
      memcpy((char*)dest->name, (const char*)src->name, sizeof(type) * nr * nc); \
    } else {                                                                     \
      dest->name = NULL;                                                         \
    }
    MJDATA_ARENA_POINTERS
    #undef X

    // redefine XNV to be the same as X
    #undef XNV
    #define XNV X
  }

  #undef MJ_M
  #define MJ_M(n) n
  #undef MJ_D
  #define MJ_D(n) n

  // restore contact pointer
  dest->contact = dest->arena;

  // restore plugin_data
  if (plugin_data_size) {
    memcpy(dest->plugin_data, save_plugin_data, plugin_data_size);
    mju_free(save_plugin_data);
    save_plugin_data = NULL;
  }

  // copy plugin instances
  dest->nplugin = m->nplugin;
  for (int i = 0; i < m->nplugin; ++i) {
    const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[i]);
    if (plugin->copy) {
      plugin->copy(dest, m, src, i);
    }
  }

  return dest;
}


mjData* mj_copyData(mjData* dest, const mjModel* m, const mjData* src) {
  return mj_copyDataVisual(dest, m, src, /*flg_all=*/1);
}


mjData* mjv_copyData(mjData* dest, const mjModel* m, const mjData* src) {
  return mj_copyDataVisual(dest, m, src, /*flg_all=*/0);
}


// number of position and velocity coordinates for each joint type
const int nPOS[4] = {7, 4, 1, 1};
const int nVEL[4] = {6, 3, 1, 1};


// initialize plastic state from model spring references
void mj_resetPlastic(const mjModel* m, mjData* d) {
  if (!m->nplastic) {
    return;
  }
  if (m->qpos_spring && m->jnt_plasticadr) {
    for (int j=0; j < m->njnt; j++) {
      int adr = m->jnt_plasticadr[j];
      if (adr >= 0) {
        mju_copy(d->plastic + adr, m->qpos_spring + m->jnt_qposadr[j], nPOS[m->jnt_type[j]]);
      }
    }
  }
  if (m->tendon_lengthspring && m->tendon_plasticadr) {
    for (int i=0; i < m->ntendon; i++) {
      int adr = m->tendon_plasticadr[i];
      if (adr >= 0) {
        mju_copy(d->plastic + adr, m->tendon_lengthspring + 2*i, 2);
      }
    }
  }
  if (m->flex_plasticadr) {
    for (int f=0; f < m->nflex; f++) {
      int adr = m->flex_plasticadr[f];
      if (adr >= 0) {
        int dim = m->flex_dim[f];
        int n = 0;
        if (m->flex_interp[f] > 0) {
          n = 6 * m->flex_cellnum[3*f+0] * m->flex_cellnum[3*f+1] * m->flex_cellnum[3*f+2];
        } else if (m->flex_interp[f] < 0) {
          int cx = m->flex_cellnum[3*f+0];
          int cy = m->flex_cellnum[3*f+1];
          int cz = m->flex_cellnum[3*f+2];
          n = 3 * 2 * (cy*cz + cx*cz + cx*cy);
        } else {
          if (m->flex_stiffnessadr[f] >= 0 && m->flex_stiffness[m->flex_stiffnessadr[f]] != 0) {
            n += (dim == 2 ? 3 : 6) * m->flex_elemnum[f];
          }
          if (dim == 2 && m->flex_bendingadr[f] >= 0) {
            n += m->flex_edgenum[f];
          }
        }
        mju_zero(d->plastic + adr, n);
        if (d->flex_hessian_valid) {
          d->flex_hessian_valid[f] = 0;
        }
      }
    }
  }
}


// clear data, set defaults; must not call mju_error (warn instead), as callers rely on
// mj_resetData as an infallible recovery path to restore a valid mjData after a trapped error
static void _resetData(const mjModel* m, mjData* d, unsigned char debug_value) {
  // warn if history buffers cannot be initialized
  mjtNum dt = m->opt.timestep;
  if (m->nhistory && dt <= 0) {
    mju_warning("history buffers require positive timestep, got %g", dt);
  }

  //------------------------------ save plugin state and data
  mjtNum* plugin_state = NULL;
  uintptr_t* plugindata = NULL;
  if (d->nplugin) {
    if (m->npluginstate) {
      plugin_state = mju_malloc(sizeof(mjtNum) * m->npluginstate);
      memcpy(plugin_state, d->plugin_state, sizeof(mjtNum) * m->npluginstate);
    }
    plugindata = mju_malloc(sizeof(uintptr_t) * m->nplugin);
    memcpy(plugindata, d->plugin_data, sizeof(uintptr_t) * m->nplugin);
  }

  //------------------------------ clear header

  // clear stack pointer
  d->pstack = 0;
  d->pbase = 0;
  d->threadlock = 0;

  // clear arena pointers
  d->parena = 0;

  // poison the entire arena+stack memory region when built with asan
#ifdef mjUSEASAN
  ASAN_POISON_MEMORY_REGION(d->arena, d->narena);
#endif

#ifdef MEMORY_SANITIZER
  __msan_allocated_memory(d->arena, d->narena);
#endif

#define X(type, name, nr, nc) d->name = NULL;
  MJDATA_ARENA_POINTERS
#undef X
  d->contact = d->arena;

  // clear memory utilization stats
  d->maxuse_stack = 0;
  d->maxuse_arena = 0;
  d->maxuse_con = 0;
  d->maxuse_efc = 0;

  // clear solver diagnostics
  memset(d->warning, 0, mjNWARNING*sizeof(mjWarningStat));
  memset(d->timer, 0, mjNTIMER*sizeof(mjTimerStat));
  memset(d->solver, 0, mjNSOLVER*mjNISLAND*sizeof(mjSolverStat));
  mju_zeroInt(d->solver_niter, mjNISLAND);
  mju_zeroInt(d->solver_nnz, mjNISLAND);
  mju_zero(d->solver_fwdinv, 2);

  // clear variable sizes
  d->ncon = 0;
  d->ne = 0;
  d->nf = 0;
  d->nl = 0;
  d->nefc = 0;
  d->nJ = 0;
  d->nY = 0;
  d->nA = 0;
  d->nisland = 0;
  d->nidof = 0;
  d->efm_active = 0;
  d->nefmK = 0;
  d->nefmcon = 0;
  d->nefmT = 0;
  d->nefmA = 0;
  d->nefmdof = 0;
  d->nefmL = 0;

  // clear global properties
  d->time = 0;
  mju_zero(d->energy, 2);

  // clear lazy evaluation flags
  d->flg_energypos = 0;
  d->flg_energyvel = 0;
  d->flg_subtreevel = 0;
  d->flg_rnepost = 0;

  //------------------------------ clear buffer, set defaults

  // fill buffer with debug_value (normally 0)
#ifdef mjUSEASAN
  {
    #define X(type, name, nr, nc) memset(d->name, (int)debug_value, sizeof(type)*(m->nr)*(nc));
    MJDATA_POINTERS
    #undef X
  }
#else
  memset(d->buffer, (int)debug_value, d->nbuffer);
#endif

#ifdef MEMORY_SANITIZER
  // under MSAN, mark the entire buffer as uninitialized
  __msan_allocated_memory(d->buffer, d->nbuffer);
#endif

  // zero out user-settable state and input arrays (MSAN: mark as initialized)
  mju_zero(d->qpos, m->nq);
  mju_zero(d->qvel, m->nv);
  mju_zero(d->act, m->na);
  mju_zero(d->plastic, m->nplastic);
  mj_resetCtrl(m, d);
  for (int i=0; i < m->neq; i++) d->eq_active[i] = m->eq_active0[i];
  mju_zero(d->qfrc_applied, m->nv);
  mju_zero(d->xfrc_applied, 6*m->nbody);
  mju_zero(d->qacc, m->nv);  // input to inverse dynamics
  mju_zero(d->qacc_warmstart, m->nv);
  mju_zero(d->act_dot, m->na);
  mju_zero(d->userdata, m->nuserdata);
  mju_zero(d->mocap_pos, 3*m->nmocap);
  mju_zero(d->mocap_quat, 4*m->nmocap);
  mju_zero(d->flexvert_lambda, m->nflexvert);
  mju_zeroInt(d->flexvert_conage, m->nflexvert);
  memset(d->flex_hessian_valid, 0, m->nflex * sizeof(mjtBool));

  // initialize ctrl history buffers: timestamps at [-n*dt, ..., -dt]
  for (int i = 0; i < m->nactuator; i++) {
    int n = m->actuator_history[2*i];
    if (n > 0) {
      mjtNum* buf = d->history + m->actuator_historyadr[i];
      buf[0] = 0;          // user slot
      buf[1] = n - 1;      // cursor: newest at logical index n-1
      mjtNum* times = buf + 2;
      for (int j = 0; j < n; j++) {
        times[j] = -(n-j)*dt;
      }

      // clear values (or set identity quaternion for quat inputs)
      mjtNum* values = buf + 2 + n;
      int dim = m->actuator_ctrlnum[i];
      mju_zero(values, n*dim);
      if (m->actuator_gaintype[i] == mjGAIN_SO3 && m->actuator_ctrlspec[i] == mjCHART_QUAT) {
        for (int j = 0; j < n; j++) {
          values[4*j] = 1;
        }
      }
    }
  }

  // initialize sensor history buffers
  for (int i = 0; i < m->nsensor; i++) {
    int n = m->sensor_history[2*i];
    if (n > 0) {
      int dim = m->sensor_dim[i];
      mjtNum period = m->sensor_interval[2*i];
      mjtNum phase = m->sensor_interval[2*i+1];
      mjtNum* buf = d->history + m->sensor_historyadr[i];

      // user slot: last compute time (phase=0 means -period, i.e. first compute at t=0)
      buf[0] = (period > 0) ? (phase == 0 ? -period : phase) : -dt;
      buf[1] = n - 1;  // cursor: newest at logical index n-1

      mjtNum* times = buf + 2;
      if (period > 0) {
        // samples spaced at period intervals, rounded up to dt grid
        mjtNum t0 = (phase == 0) ? -period : phase;
        for (int j = 0; j < n; j++) {
          mjtNum continuous_t = t0 - (n-1-j)*period;
          times[j] = dt > 0 ? mju_ceil(continuous_t / dt) * dt : continuous_t;
        }
      } else {
        // no period: timestamps at [-n*dt, ..., -dt]
        for (int j = 0; j < n; j++) {
          times[j] = -(n-j)*dt;
        }
      }

      // clear values
      mjtNum* values = buf + 2 + n;
      mju_zero(values, n*dim);
    }
  }

  // copy qpos0 from model
  if (m->qpos0) {
    mju_copy(d->qpos, m->qpos0, m->nq);
  }

  // initialize plastic state from model spring references
  mj_resetPlastic(m, d);

  // restore pluginstate and plugindata
  if (d->nplugin) {
    if (m->npluginstate) {
      memcpy(d->plugin_state, plugin_state, sizeof(mjtNum) * m->npluginstate);
      mju_free(plugin_state);
    }
    memcpy(d->plugin_data, plugindata, sizeof(uintptr_t) * m->nplugin);
    mju_free(plugindata);

    // restore the plugin array back into d and reset the instances
    for (int i = 0; i < m->nplugin; ++i) {
      d->plugin[i] = m->plugin[i];
      const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[i]);
      if (plugin->reset) {
        plugin->reset(m, &d->plugin_state[m->plugin_stateadr[i]],
                      (void*)(d->plugin_data[i]), i);
      }
    }
  }

  // set all trees to awake
  mju_fillInt(d->tree_asleep, kAwake, m->ntree);

  // sleep enabled: handle static bodies and trees marked as mjSLEEP_INIT
  if (mjENABLED(mjENBL_SLEEP)) {
    // count trees initialized as asleep
    int num_asleep_init = 0;
    for (int i=0; i < m->ntree; i++) {
      num_asleep_init += (m->tree_sleep_policy[i] == mjSLEEP_INIT);
    }

    // update sleep arrays, treat static bodies as awake
    mj_updateSleepInit(m, d, /*flg_staticawake*/ 1);

    // partial mj_fwdPosition, functions that update STATIC values
    if (!num_asleep_init) {
      mj_kinematics(m, d);
      mj_comPos(m, d);
      mj_camlight(m, d);
      mj_tendon(m, d);
    }

    // if any trees initialized as sleeping call entire mj_forward, put them to sleep
    else {
      mj_forward(m, d);

      // mark asleep-init trees as ready to sleep
      for (int i=0; i < m->ntree; i++) {
        int init = m->tree_sleep_policy[i] == mjSLEEP_INIT;
        d->tree_asleep[i] = init ? -1 : kAwake;
      }

      int nslept = mj_sleep(m, d);

      // warn if any failed to sleep, leave them awake
      if (nslept != num_asleep_init) {
        // find root body of the first tree that could not be slept, wake the others
        int root = -1;
        for (int i=0; i < m->ntree; i++) {
          if (m->tree_sleep_policy[i] == mjSLEEP_INIT && d->tree_asleep[i] < 0) {
            if (root < 0) root = m->tree_bodyadr[i];
            d->tree_asleep[i] = kAwake;
          }
        }

        // warn and continue
        const char* hasname = mj_id2name(m, mjOBJ_BODY, root);
        const char* name = hasname ? hasname : "";
        mju_warning("%d trees were marked as sleep='init' but only %d could be slept. "
                    "Body '%s' (id=%d) is the root of the first tree that could not be slept.",
                    num_asleep_init, nslept, name, root);
      }

      // clear arrays to avoid MSAN errors upon mid-step wake
      mju_zero(d->qacc_smooth, m->nv);
      mju_zero(d->qfrc_smooth, m->nv);

      // clear arena
      mj_clearEfc(d);
    }
  }

  // update sleep arrays and counters
  mj_updateSleep(m, d);

  // set mocap_pos/quat = body_pos/quat for mocap bodies
  if (m->body_mocapid) {
    for (int i=0; i < m->nbody; i++) {
      int id = m->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos+3*id, m->body_pos+3*i);
        mju_copy4(d->mocap_quat+4*id, m->body_quat+4*i);
      }
    }
  } else {
    // set the mocap_quats to {1, 0, 0, 0}
    for (int i=0; i < m->nmocap; i++) {
      d->mocap_quat[4*i] = 1.0;
    }
  }

  // check consistency of sparse matrix representations
  checkDBSparse(m);

  // copy signature from model
  d->signature = m->signature;
}


// emit step timing diagnostics
static void mj_logTimingDiagnostics(const mjData* d) {
  int nstep = d->timer[mjTIMER_STEP].number;
  if (nstep <= 0) {
    return;
  }

  mjtNum tstep = d->timer[mjTIMER_STEP].duration / nstep;
  if (tstep <= 0) {
    return;
  }

  char buf[2048];
  int pos = 0;
  mjtNum components = 0;

#define MJ_APPEND_TIMING(...) \
  do { \
    if (pos < (int)sizeof(buf) - 1) { \
      int n = snprintf(buf + pos, sizeof(buf) - pos, __VA_ARGS__); \
      if (n > 0) { \
        pos = mjMIN((int)sizeof(buf) - 1, pos + n); \
      } \
    } \
  } while (0)

  for (int i = mjTIMER_POSITION; i <= mjTIMER_ADVANCE; i++) {
    if (d->timer[i].number > 0) {
      mjtNum istep = d->timer[i].duration / d->timer[i].number;
      components += istep;
      MJ_APPEND_TIMING("%s  %-15s %8.1f  (%5.1f%%)",
                       pos > 0 ? "\n" : "",
                       mjTIMERSTRING[i], istep * 1000, 100 * istep / tstep);

      // position sub-breakdown
      if (i == mjTIMER_POSITION) {
        for (int p = mjTIMER_POS_KINEMATICS; p <= mjTIMER_POS_PROJECT; p++) {
          if (d->timer[p].number > 0) {
            mjtNum pstep = d->timer[p].duration / d->timer[p].number;
            MJ_APPEND_TIMING("\n    %-13s %8.1f  (%5.1f%%)",
                             mjTIMERSTRING[p] + 4, pstep * 1000, 100 * pstep / tstep);

            // collision sub-breakdown
            if (p == mjTIMER_POS_COLLISION) {
              for (int c = mjTIMER_COL_BROAD; c <= mjTIMER_COL_NARROW; c++) {
                if (d->timer[c].number > 0) {
                  mjtNum cstep = d->timer[c].duration / d->timer[c].number;
                  MJ_APPEND_TIMING("\n      %-11s %8.1f  (%5.1f%%)",
                                   mjTIMERSTRING[c] + 4, cstep * 1000, 100 * cstep / tstep);
                }
              }
            }
          }
        }
      }
    }
  }

  mjtNum other = tstep - components;
  MJ_APPEND_TIMING("%s  %-15s %8.1f  (%5.1f%%)",
                   pos > 0 ? "\n" : "",
                   "other", other * 1000, 100 * other / tstep);

  MJ_APPEND_TIMING("%s  %-15s %8.1f",
                   pos > 0 ? "\n" : "",
                   "total", tstep * 1000);

#undef MJ_APPEND_TIMING

  mjLogMessage msg = {.level = mjLOG_INFO, .topic = mjTOPIC_TIME_STP, .body = buf};
  snprintf(msg.subject, sizeof(msg.subject),
           "average time per step (%d steps, units: \u00B5s)", nstep);
  mju_message(&msg);
}


// set ctrl to neutral values: zero, except quaternion inputs which reset to the identity
void mj_resetCtrl(const mjModel* m, mjData* d) {
  mju_zero(d->ctrl, m->nu);
  for (int i=0; i < m->nactuator; i++) {
    if (m->actuator_gaintype[i] == mjGAIN_SO3 && m->actuator_ctrlspec[i] == mjCHART_QUAT) {
      d->ctrl[m->actuator_ctrladr[i]] = 1;
    }
  }
}


// clear data, set data->qpos = model->qpos0
void mj_resetData(const mjModel* m, mjData* d) {
  // emit step timing diagnostics before timers are cleared
  mj_logTimingDiagnostics(d);

  _resetData(m, d, 0);
}



// clear data, set data->qpos = model->qpos0, fill with debug_value
void mj_resetDataDebug(const mjModel* m, mjData* d, unsigned char debug_value) {
  _resetData(m, d, debug_value);
}


// Reset data. If 0 <= key < nkey, set fields from specified keyframe.
void mj_resetDataKeyframe(const mjModel* m, mjData* d, int key) {
  // emit step timing diagnostics before timers are cleared
  mj_logTimingDiagnostics(d);

  _resetData(m, d, 0);

  // copy keyframe data if key is valid
  if (key >= 0 && key < m->nkey) {
    d->time = m->key_time[key];
    mju_copy(d->qpos, m->key_qpos+key*m->nq, m->nq);
    mju_copy(d->qvel, m->key_qvel+key*m->nv, m->nv);
    mju_copy(d->act,  m->key_act+ key*m->na, m->na);
    mju_copy(d->mocap_pos,  m->key_mpos+key*3*m->nmocap, 3*m->nmocap);
    mju_copy(d->mocap_quat, m->key_mquat+key*4*m->nmocap, 4*m->nmocap);
    mju_copy(d->ctrl, m->key_ctrl+key*m->nu, m->nu);
    mju_copy(d->plastic, m->key_plastic+key*m->nplastic, m->nplastic);
  }
}


// de-allocate mjData
void mj_deleteData(mjData* d) {
  if (d) {
    mju_threadpool(d, 0);
    freeDataBuffers(d);
    mju_free(d);
  }
}


static int sensorSize(mjtSensor sensor_type, int sensor_dim) {
  switch (sensor_type) {
  case mjSENS_TOUCH:
  case mjSENS_RANGEFINDER:
  case mjSENS_JOINTPOS:
  case mjSENS_JOINTVEL:
  case mjSENS_TENDONPOS:
  case mjSENS_TENDONVEL:
  case mjSENS_ACTUATORPOS:
  case mjSENS_ACTUATORVEL:
  case mjSENS_ACTUATORFRC:
  case mjSENS_JOINTACTFRC:
  case mjSENS_TENDONACTFRC:
  case mjSENS_JOINTLIMITPOS:
  case mjSENS_JOINTLIMITVEL:
  case mjSENS_JOINTLIMITFRC:
  case mjSENS_TENDONLIMITPOS:
  case mjSENS_TENDONLIMITVEL:
  case mjSENS_TENDONLIMITFRC:
  case mjSENS_GEOMDIST:
  case mjSENS_INSIDESITE:
  case mjSENS_E_POTENTIAL:
  case mjSENS_E_KINETIC:
  case mjSENS_CLOCK:
    return 1;

  case mjSENS_CAMPROJECTION:
    return 2;

  case mjSENS_ACCELEROMETER:
  case mjSENS_VELOCIMETER:
  case mjSENS_GYRO:
  case mjSENS_FORCE:
  case mjSENS_TORQUE:
  case mjSENS_MAGNETOMETER:
  case mjSENS_BALLANGVEL:
  case mjSENS_FRAMEPOS:
  case mjSENS_FRAMEXAXIS:
  case mjSENS_FRAMEYAXIS:
  case mjSENS_FRAMEZAXIS:
  case mjSENS_FRAMELINVEL:
  case mjSENS_FRAMEANGVEL:
  case mjSENS_FRAMELINACC:
  case mjSENS_FRAMEANGACC:
  case mjSENS_SUBTREECOM:
  case mjSENS_SUBTREELINVEL:
  case mjSENS_SUBTREEANGMOM:
  case mjSENS_GEOMNORMAL:
    return 3;

  case mjSENS_GEOMFROMTO:
    return 6;

  case mjSENS_BALLQUAT:
  case mjSENS_FRAMEQUAT:
    return 4;

  case mjSENS_CONTACT:
  case mjSENS_TACTILE:
  case mjSENS_USER:
    return sensor_dim;

  case mjSENS_PLUGIN:
    return -1;

    // don't use a 'default' case, so compiler warns about missing values
  }
  return -1;
}


// returns the number of objects of the given type
//   -1: mjOBJ_UNKNOWN
//   -2: invalid objtype
static int numObjects(const mjModel* m, mjtObj objtype) {
  switch (objtype) {
  case mjOBJ_DEFAULT:
  case mjOBJ_FRAME:
  case mjOBJ_UNKNOWN:
  case mjOBJ_MODEL:
    return -1;
  case mjOBJ_BODY:
  case mjOBJ_XBODY:
    return m->nbody;
  case mjOBJ_JOINT:
    return m->njnt;
  case mjOBJ_DOF:
    return m->nv;
  case mjOBJ_GEOM:
    return m->ngeom;
  case mjOBJ_SITE:
    return m->nsite;
  case mjOBJ_CAMERA:
    return m->ncam;
  case mjOBJ_LIGHT:
    return m->nlight;
  case mjOBJ_FLEX:
    return m->nflex;
  case mjOBJ_MESH:
    return m->nmesh;
  case mjOBJ_SKIN:
    return m->nskin;
  case mjOBJ_HFIELD:
    return m->nhfield;
  case mjOBJ_TEXTURE:
    return m->ntex;
  case mjOBJ_MATERIAL:
    return m->nmat;
  case mjOBJ_PAIR:
    return m->npair;
  case mjOBJ_EXCLUDE:
    return m->nexclude;
  case mjOBJ_EQUALITY:
    return m->neq;
  case mjOBJ_TENDON:
    return m->ntendon;
  case mjOBJ_ACTUATOR:
    return m->nactuator;
  case mjOBJ_SENSOR:
    return m->nsensor;
  case mjOBJ_NUMERIC:
    return m->nnumeric;
  case mjOBJ_TEXT:
    return m->ntext;
  case mjOBJ_TUPLE:
    return m->ntuple;
  case mjOBJ_KEY:
    return m->nkey;
  case mjOBJ_PLUGIN:
    return m->nplugin;
  case mjNOBJECT:
    return -2;
  }
  return -2;
}


// validate CSR sparse matrix structure; return 0 if valid, error code otherwise
//   1: rownnz out of bounds
//   2: rowadr out of bounds
//   3: colind out of bounds or unsorted
static int validateSparse(int nrows, int ncols, mjtSize nnz,
                          const int* rownnz, const int* rowadr, const int* colind,
                          int strict) {
  for (int r=0; r < nrows; r++) {
    int nnz_r = rownnz[r];
    if (nnz_r < 0 || nnz_r > ncols) {
      return 1;
    }
    int adr_r = rowadr[r];
    if (adr_r < 0 || (mjtSize)adr_r + nnz_r > nnz) {
      return 2;
    }
    for (int k=0; k < nnz_r; k++) {
      int col = colind[adr_r+k];
      if (col < 0 || col >= ncols) {
        return 3;
      }
      if (k > 0) {
        int prev = colind[adr_r+k-1];
        if (strict ? (col <= prev) : (col < prev)) {
          return 3;
        }
      }
    }
  }
  return 0;
}


// validate reference fields in a model; return null if valid, error message otherwise
const char* mj_validateReferences(const mjModel* m) {
  // for each field in mjModel that refers to another field, call X with:
  //   adrarray: array containing the references
  //   nadrs:    number of elements in refarray
  //   ntarget:  number of elements in array where references are pointing
  //   numarray: if refarray is an adr array, numarray is the corresponding num array, otherwise 0
#define MJMODEL_REFERENCES                                                       \
  X(body_parentid,      nbody,          nbody         , 0                      ) \
  X(body_rootid,        nbody,          nbody         , 0                      ) \
  X(body_weldid,        nbody,          nbody         , 0                      ) \
  X(body_mocapid,       nbody,          nmocap        , 0                      ) \
  X(body_jntadr,        nbody,          njnt          , m->body_jntnum         ) \
  X(body_dofadr,        nbody,          nv            , m->body_dofnum         ) \
  X(body_geomadr,       nbody,          ngeom         , m->body_geomnum        ) \
  X(body_bvhadr,        nbody,          nbvh          , m->body_bvhnum         ) \
  X(body_plugin,        nbody,          nplugin       , 0                      ) \
  X(jnt_qposadr,        njnt,           nq            , 0                      ) \
  X(jnt_dofadr,         njnt,           nv            , 0                      ) \
  X(jnt_bodyid,         njnt,           nbody         , 0                      ) \
  X(dof_bodyid,         nv,             nbody         , 0                      ) \
  X(dof_jntid,          nv,             njnt          , 0                      ) \
  X(dof_parentid,       nv,             nv            , 0                      ) \
  X(dof_Madr,           nv,             nM            , 0                      ) \
  X(tree_bodyadr,       ntree,          nbody         , m->tree_bodynum        ) \
  X(tree_dofadr,        ntree,          nv            , m->tree_dofnum         ) \
  X(geom_bodyid,        ngeom,          nbody         , 0                      ) \
  X(geom_matid,         ngeom,          nmat          , 0                      ) \
  X(site_bodyid,        nsite,          nbody         , 0                      ) \
  X(site_matid,         nsite,          nmat          , 0                      ) \
  X(cam_bodyid,         ncam,           nbody         , 0                      ) \
  X(cam_targetbodyid,   ncam,           nbody         , 0                      ) \
  X(light_bodyid,       nlight,         nbody         , 0                      ) \
  X(light_targetbodyid, nlight,         nbody         , 0                      ) \
  X(mesh_vertadr,       nmesh,          nmeshvert     , m->mesh_vertnum        ) \
  X(mesh_normaladr,     nmesh,          nmeshnormal   , m->mesh_normalnum      ) \
  X(mesh_texcoordadr,   nmesh,          nmeshtexcoord , m->mesh_texcoordnum    ) \
  X(mesh_faceadr,       nmesh,          nmeshface     , m->mesh_facenum        ) \
  X(mesh_bvhadr,        nmesh,          nbvh          , m->mesh_bvhnum         ) \
  X(mesh_graphadr,      nmesh,          nmeshgraph    , 0                      ) \
  X(mesh_polyadr,       nmesh,          nmeshpoly     , m->mesh_polynum        ) \
  X(mesh_polyvertadr,   nmeshpoly,      nmeshpolyvert , m->mesh_polyvertnum    ) \
  X(mesh_polymapadr,    nmeshvert,      nmeshpolymap  , m->mesh_polymapnum     ) \
  X(flex_vertadr,       nflex,          nflexvert     , m->flex_vertnum        ) \
  X(flex_edgeadr,       nflex,          nflexedge     , m->flex_edgenum        ) \
  X(flex_elemadr,       nflex,          nflexelem     , m->flex_elemnum        ) \
  X(flex_matid,         nflexelem,      nmat          , 0                      ) \
  X(flex_texcoordadr,   nflex,          nflextexcoord , 0                      ) \
  X(flex_elemdataadr,   nflex,          nflexelemdata , 0                      ) \
  X(flex_elemedgeadr,   nflex,          nflexelemedge , 0                      ) \
  X(flex_shelldataadr,  nflex,          nflexshelldata, 0                      ) \
  X(flex_edge,          nflexedge*2,    nflexvert     , 0                      ) \
  X(flex_elem,          nflexelemdata,  nflexvert     , 0                      ) \
  X(flex_elemedge,      nflexelemedge,  nflexedge     , 0                      ) \
  X(flex_shell,         nflexshelldata, nflexvert     , 0                      ) \
  X(flex_bvhadr,        nflex,          nbvh          , m->flex_bvhnum         ) \
  X(skin_matid,         nskin,          nmat          , 0                      ) \
  X(skin_vertadr,       nskin,          nskinvert     , m->skin_vertnum        ) \
  X(skin_texcoordadr,   nskin,          nskintexvert  , 0                      ) \
  X(skin_faceadr,       nskin,          nskinface     , m->skin_facenum        ) \
  X(skin_boneadr,       nskin,          nskinbone     , m->skin_bonenum        ) \
  X(skin_bonevertadr,   nskinbone,      nskinbonevert , m->skin_bonevertnum    ) \
  X(skin_bonebodyid,    nskinbone,      nbody         , 0                      ) \
  X(skin_bonevertid,    nskinbonevert,  nskinvert     , 0                      ) \
  X(pair_geom1,         npair,          ngeom         , 0                      ) \
  X(pair_geom2,         npair,          ngeom         , 0                      ) \
  X(actuator_plugin,    nactuator,      nplugin       , 0                      ) \
  X(actuator_actadr,    nactuator,      na            , m->actuator_actnum     ) \
  X(actuator_ctrladr,   nactuator,      nu            , m->actuator_ctrlnum    ) \
  X(actuator_outadr,    nactuator,      nout          , m->actuator_outnum     ) \
  X(sensor_plugin,      nsensor,        nplugin       , 0                      ) \
  X(plugin_stateadr,    nplugin,        npluginstate  , m->plugin_statenum     ) \
  X(plugin_attradr,     nplugin,        npluginattr   , 0                      ) \
  X(tendon_adr,         ntendon,        nwrap         , m->tendon_num          ) \
  X(tendon_matid,       ntendon,        nmat          , 0                      ) \
  X(tendon_treeid,      ntendon*2,      ntree         , 0                      ) \
  X(numeric_adr,        nnumeric,       nnumericdata  , m->numeric_size        ) \
  X(text_adr,           ntext,          ntextdata     , m->text_size           ) \
  X(tuple_adr,          ntuple,         ntupledata    , m->tuple_size          ) \
  X(name_bodyadr,       nbody,          nnames        , 0                      ) \
  X(name_jntadr,        njnt,           nnames        , 0                      ) \
  X(name_geomadr,       ngeom,          nnames        , 0                      ) \
  X(name_siteadr,       nsite,          nnames        , 0                      ) \
  X(name_camadr,        ncam,           nnames        , 0                      ) \
  X(name_lightadr,      nlight,         nnames        , 0                      ) \
  X(name_meshadr,       nmesh,          nnames        , 0                      ) \
  X(name_skinadr,       nskin,          nnames        , 0                      ) \
  X(name_hfieldadr,     nhfield,        nnames        , 0                      ) \
  X(name_texadr,        ntex,           nnames        , 0                      ) \
  X(name_matadr,        nmat,           nnames        , 0                      ) \
  X(name_pairadr,       npair,          nnames        , 0                      ) \
  X(name_excludeadr,    nexclude,       nnames        , 0                      ) \
  X(name_eqadr,         neq,            nnames        , 0                      ) \
  X(name_tendonadr,     ntendon,        nnames        , 0                      ) \
  X(name_actuatoradr,   nactuator,      nnames        , 0                      ) \
  X(name_sensoradr,     nsensor,        nnames        , 0                      ) \
  X(name_numericadr,    nnumeric,       nnames        , 0                      ) \
  X(name_textadr,       ntext,          nnames        , 0                      ) \
  X(name_tupleadr,      ntuple,         nnames        , 0                      ) \
  X(name_keyadr,        nkey,           nnames        , 0                      ) \
  X(hfield_pathadr,     nhfield,        npaths        , 0                      ) \
  X(mesh_pathadr,       nmesh,          npaths        , 0                      ) \
  X(skin_pathadr,       nskin,          npaths        , 0                      ) \
  X(tex_pathadr,        ntex,           npaths        , 0                      ) \
  X(tex_contenttypeadr, ntex,           npaths        , 0                      )

  #define X(adrarray, nadrs, ntarget, numarray) {             \
    int *nums = (numarray);                                   \
    for (int i=0; i<m->nadrs; i++) {                          \
      int adrsmin = m->adrarray[i];                           \
      int num = (nums ? nums[i] : 1);                         \
      if (num < 0) {                                          \
        return "Invalid model: " #numarray " is negative.";   \
      }                                                       \
      if (num > MAX_ARRAY_SIZE) {                             \
        return "Invalid model: " #numarray " is too large.";  \
      }                                                       \
      int adrsmax = m->adrarray[i] + num;                     \
      if (adrsmax > m->ntarget || adrsmin < -1) {             \
        return "Invalid model: " #adrarray " out of bounds."; \
      }                                                       \
      if (nums && adrsmin == -1 && num > 0) {                 \
        return "Invalid model: " #adrarray " out of bounds."; \
      }                                                       \
    }                                                         \
  }

  MJMODEL_REFERENCES;
  #undef X
#undef MJMODEL_REFERENCES

  #define VALIDATE_SPARSE(prefix, nrows, ncols, nnz, strict)           \
    switch (validateSparse(nrows, ncols, nnz,                          \
                           m->prefix##_rownnz, m->prefix##_rowadr,     \
                           m->prefix##_colind, strict)) {              \
    case 1: return "Invalid model: " #prefix "_rownnz out of bounds."; \
    case 2: return "Invalid model: " #prefix "_rowadr out of bounds."; \
    case 3: return "Invalid model: " #prefix "_colind out of bounds."; \
    }

  VALIDATE_SPARSE(M,          m->nv,          m->nv,       m->nC,     1);
  VALIDATE_SPARSE(D,          m->nv,          m->nv,       m->nD,     1);
  VALIDATE_SPARSE(B,          m->nbody,       m->nv,       m->nB,     1);
  VALIDATE_SPARSE(ten_J,      m->ntendon,     m->nv,       m->nJten,  0);
  VALIDATE_SPARSE(flexedge_J, m->nflexedge,   m->nv,       m->nJfe,   1);
  VALIDATE_SPARSE(efm0_L,     m->nefm0dof,    m->nefm0dof, m->nefm0L, 1);
  #undef VALIDATE_SPARSE

  // special logic that doesn't fit in the macro:
  for (int i=0; i < m->nbody; i++) {
    if ((i > 0 && m->body_parentid[i] >= i) || m->body_parentid[i] < 0) {
      return "Invalid model: bad body_parentid.";
    }
    if (m->body_rootid[i] > i || m->body_rootid[i] < 0) {
      return "Invalid model: bad body_rootid.";
    }
    if (m->body_weldid[i] > i || m->body_weldid[i] < 0) {
      return "Invalid model: bad body_weldid.";
    }
    if (i == 0) {
      if (m->B_rownnz[0] != m->nv || (m->nv > 0 && m->B_rowadr[0] != 0)) {
        return "Invalid model: B_rownnz out of bounds.";
      }
    } else {
      int np = m->body_parentid[i];
      int c = 0, cp = 0;
      while (c < m->B_rownnz[i] && cp < m->B_rownnz[np]) {
        int col = m->B_colind[m->B_rowadr[i]+c];
        int colp = m->B_colind[m->B_rowadr[np]+cp];
        if (col == colp) {
          c++;
          cp++;
        } else if (col > colp) {
          cp++;
        } else {
          break;
        }
      }
      if (c < m->B_rownnz[i]) {
        return "Invalid model: B_colind out of bounds.";
      }

      mjtSize ndof = 0;
      int np_weld = m->body_weldid[np];
      while (np_weld > 0) {
        ndof += m->body_dofnum[np_weld];
        np_weld = m->body_weldid[m->body_parentid[np_weld]];
      }
      if (ndof > m->B_rownnz[i] || ndof > m->B_rownnz[np]) {
        return "Invalid model: B_rownnz out of bounds.";
      }
    }
  }
  for (int i=0; i < m->njnt; i++) {
    if (m->jnt_type[i] >= 4 || m->jnt_type[i] < 0) {
      return "Invalid model: jnt_type out of bounds.";
    }
    int jnt_qposadr = m->jnt_qposadr[i] + nPOS[m->jnt_type[i]];
    if (jnt_qposadr > m->nq || m->jnt_qposadr[i] < 0) {
      return "Invalid model: jnt_qposadr out of bounds.";
    }
    int jnt_dofadr = m->jnt_dofadr[i] + nVEL[m->jnt_type[i]];
    if (jnt_dofadr > m->nv || m->jnt_dofadr[i] < 0) {
      return "Invalid model: jnt_dofadr out of bounds.";
    }
    if (m->jnt_plasticadr[i] < -1 ||
        (m->jnt_plasticadr[i] >= 0 &&
         m->jnt_plasticadr[i] + nPOS[m->jnt_type[i]] > m->nplastic)) {
      return "Invalid model: jnt_plasticadr out of bounds.";
    }
  }
  for (int i=0; i < m->ntendon; i++) {
    if (m->tendon_plasticadr[i] < -1 ||
        (m->tendon_plasticadr[i] >= 0 && m->tendon_plasticadr[i] + 2 > m->nplastic)) {
      return "Invalid model: tendon_plasticadr out of bounds.";
    }
  }
  mjtSize nM_post = 0;
  for (int i=0; i < m->nv; i++) {
    if (m->dof_parentid[i] >= i) {
      return "Invalid model: bad dof_parentid.";
    }
    int bodyid = m->dof_bodyid[i];
    if (bodyid < 0 || bodyid >= m->nbody) {
      return "Invalid model: dof_bodyid out of bounds.";
    }
    if (m->dof_Madr[i] != nM_post) {
      return "Invalid model: dof_Madr out of bounds.";
    }
    int nanc = 0;
    for (int j=i; j >= 0; j = m->dof_parentid[j]) {
      nanc++;
    }
    nM_post += nanc;

    int expected_nnz = m->dof_simplenum[i] ? 1 : nanc;
    if (m->M_rownnz[i] != expected_nnz) {
      return "Invalid model: M_rownnz out of bounds.";
    }
    int adr = m->M_rowadr[i];
    int k = expected_nnz - 1;
    for (int j=i; k >= 0; j = m->dof_parentid[j], k--) {
      if (m->M_colind[adr+k] != j) {
        return "Invalid model: M_colind out of bounds.";
      }
    }

    if (m->D_diag[i] != nanc - 1 || m->D_diag[i] >= m->D_rownnz[i] ||
        m->D_colind[m->D_rowadr[i]+m->D_diag[i]] != i) {
      return "Invalid model: D_diag out of bounds.";
    }

    if (m->D_rownnz[i] != m->B_rownnz[bodyid]) {
      return "Invalid model: D_rownnz out of bounds.";
    }
    for (int c=0; c < m->D_rownnz[i]; c++) {
      if (m->D_colind[m->D_rowadr[i]+c] != m->B_colind[m->B_rowadr[bodyid]+c]) {
        return "Invalid model: D_colind out of bounds.";
      }
    }
  }
  if (nM_post != m->nM) {
    return "Invalid model: dof_Madr out of bounds.";
  }
  for (int i=0; i < m->nC; i++) {
    if (m->mapM2M[i] < 0 || m->mapM2M[i] >= m->nM) {
      return "Invalid model: mapM2M out of bounds.";
    }
    if (m->mapD2M[i] < 0 || m->mapD2M[i] >= m->nD) {
      return "Invalid model: mapD2M out of bounds.";
    }
  }
  for (int i=0; i < m->nD; i++) {
    if (m->mapM2D[i] < -1 || m->mapM2D[i] >= m->nC) {
      return "Invalid model: mapM2D out of bounds.";
    }
  }
  for (int i=0; i < m->nefm0dof; i++) {
    if (m->efm0_dofid[i] < 0 || m->efm0_dofid[i] >= m->nv) {
      return "Invalid model: efm0_dofid out of bounds.";
    }
  }
  for (int i=0; i < m->nefmCvert; i++) {
    if (m->efmC_perm[i] < -1 || m->efmC_perm[i] >= m->nv) {
      return "Invalid model: efmC_perm out of bounds.";
    }
  }
  for (int f=0; f < m->nflex; f++) {
    int ncell = m->flex_interp[f] ?
                m->flex_cellnum[3*f+0] * m->flex_cellnum[3*f+1] * m->flex_cellnum[3*f+2] : 0;
    for (int v=0; v < m->flex_vertnum[f]; v++) {
      int c = m->flex_vertcell[m->flex_vertadr[f] + v];
      if (m->flex_interp[f] ? (c < 0 || c >= ncell) : (c != -1)) {
        return "Invalid model: flex_vertcell out of bounds.";
      }
    }
    int dim = m->flex_dim[f];
    int nplastic = 0;
    if (m->flex_interp[f] > 0) {
      nplastic = 6 * ncell;
    } else if (m->flex_interp[f] < 0) {
      int cx = m->flex_cellnum[3*f+0];
      int cy = m->flex_cellnum[3*f+1];
      int cz = m->flex_cellnum[3*f+2];
      nplastic = 3 * 2 * (cy*cz + cx*cz + cx*cy);
    } else {
      nplastic = (m->flex_stiffnessadr[f] >= 0 && m->flex_stiffness[m->flex_stiffnessadr[f]] != 0)
                 ? (dim == 2 ? 3 : 6) * m->flex_elemnum[f] : 0;
      if (dim == 2 && m->flex_bendingadr[f] >= 0) {
        nplastic += m->flex_edgenum[f];
      }
    }
    if (m->flex_plasticadr[f] < -1 ||
        (m->flex_plasticadr[f] >= 0 && m->flex_plasticadr[f] + nplastic > m->nplastic)) {
      return "Invalid model: flex_plasticadr out of bounds.";
    }
  }
  for (int i=0; i < m->ngeom; i++) {
    if (m->geom_condim[i] > 6 || m->geom_condim[i] < 0) {
      return "Invalid model: geom_condim out of bounds.";
    }
    if (m->geom_type[i] == mjGEOM_HFIELD) {
      if (m->geom_dataid[i] >= m->nhfield || m->geom_dataid[i] < -1) {
        return "Invalid model: geom_dataid out of bounds.";
      }
    } else if ((m->geom_type[i] == mjGEOM_MESH) || (m->geom_type[i] == mjGEOM_SDF)) {
      if (m->geom_dataid[i] >= m->nmesh || m->geom_dataid[i] < -1) {
        return "Invalid model: geom_dataid out of bounds.";
      }
    }
  }
  for (int i=0; i < m->nsite; i++) {
    if (m->site_type[i] == mjGEOM_MESH) {
      if (m->site_dataid[i] >= m->nmesh || m->site_dataid[i] < -1) {
        return "Invalid model: site_dataid out of bounds.";
      }
    }
  }
  for (int i=0; i < m->nhfield; i++) {
    mjtSize hfield_adr = m->hfield_adr[i] + ((mjtSize) m->hfield_nrow[i]) * m->hfield_ncol[i];
    if (hfield_adr > m->nhfielddata || m->hfield_adr[i] < 0) {
      return "Invalid model: hfield_adr out of bounds.";
    }
  }
  for (int i=0; i < m->ntex; i++) {
    // by-reference texture: no payload, the file is at tex_pathadr
    if (m->tex_adr[i] == -1) {
      if (m->tex_pathadr[i] < 0) {
        return "Invalid model: by-reference texture has no path.";
      }
      continue;
    }
    mjtSize nbytes = ((mjtSize) m->tex_nchannel[i]) * m->tex_height[i] * m->tex_width[i];
    if (m->tex_adr[i] + nbytes > m->ntexdata || m->tex_adr[i] < 0) {
      return "Invalid model: tex_adr out of bounds.";
    }
  }

  // mesh index arrays: mesh-local indices, read verbatim from MJB files
  for (int i=0; i < m->nmesh; i++) {
    int vertnum = m->mesh_vertnum[i];
    int polynum = m->mesh_polynum[i];
    int normalnum = m->mesh_normalnum[i];
    int texcoordnum = m->mesh_texcoordnum[i];
    int has_texcoord = (m->mesh_texcoordadr[i] >= 0);

    // face vertex, normal and texcoord indices (texcoords only used if present)
    int faceadr = m->mesh_faceadr[i];
    for (int f=0; f < m->mesh_facenum[i]; f++) {
      for (int k=0; k < 3; k++) {
        int j = 3*(faceadr + f) + k;
        int v = m->mesh_face[j];
        if (v < 0 || v >= vertnum) {
          return "Invalid model: mesh_face vertex index out of bounds.";
        }
        int n = m->mesh_facenormal[j];
        if (n < 0 || n >= normalnum) {
          return "Invalid model: mesh_facenormal normal index out of bounds.";
        }
        if (has_texcoord) {
          int t = m->mesh_facetexcoord[j];
          if (t < 0 || t >= texcoordnum) {
            return "Invalid model: mesh_facetexcoord texcoord index out of bounds.";
          }
        }
      }
    }

    // polygon vertex indices
    int polyadr = m->mesh_polyadr[i];
    for (int p=0; p < polynum; p++) {
      int pvadr = m->mesh_polyvertadr[polyadr+p];
      for (int j=0; j < m->mesh_polyvertnum[polyadr+p]; j++) {
        int v = m->mesh_polyvert[pvadr+j];
        if (v < 0 || v >= vertnum) {
          return "Invalid model: mesh_polyvert vertex index out of bounds.";
        }
      }
    }

    // vertex-to-polygon map indices
    int vertadr = m->mesh_vertadr[i];
    for (int v=0; v < vertnum; v++) {
      int pmadr = m->mesh_polymapadr[vertadr+v];
      for (int j=0; j < m->mesh_polymapnum[vertadr+v]; j++) {
        int p = m->mesh_polymap[pmadr+j];
        if (p < 0 || p >= polynum) {
          return "Invalid model: mesh_polymap polygon index out of bounds.";
        }
      }
    }
  }

  for (int i=0; i < m->npair; i++) {
    int pair_body1 = (m->pair_signature[i] & 0xFFFF);
    if (pair_body1 >= m->nbody || pair_body1 < 0) {
      return "Invalid model: pair_body1 out of bounds.";
    }

    // unsigned shift: a signed >> sign-extends signatures whose high id is >= 0x8000
    int pair_body2 = (int)((unsigned int)m->pair_signature[i] >> 16);
    if (pair_body2 >= m->nbody || pair_body2 < 0) {
      return "Invalid model: pair_body2 out of bounds.";
    }
  }
  for (int i=0; i < m->neq; i++) {
    int obj1id = m->eq_obj1id[i];
    int obj2id = m->eq_obj2id[i];
    int objtype = m->eq_objtype[i];
    switch ((mjtEq) m->eq_type[i]) {
    case mjEQ_JOINT:
      if (obj1id >= m->njnt || obj1id < 0) {
        return "Invalid model: eq_obj1id out of bounds.";
      }
      // -1 is the value used if second object is omitted.
      if (obj2id >= m->njnt || obj2id < -1) {
        return "Invalid model: eq_obj2id out of bounds.";
      }
      break;

    case mjEQ_TENDON:
      if (obj1id >= m->ntendon || obj1id < 0) {
        return "Invalid model: eq_obj1id out of bounds.";
      }
      // -1 is the value used if second object is omitted.
      if (obj2id >= m->ntendon || obj2id < -1) {
        return "Invalid model: eq_obj2id out of bounds.";
      }
      break;

    case mjEQ_WELD:
    case mjEQ_CONNECT:
      if (objtype == mjOBJ_BODY) {
        if (obj1id >= m->nbody || obj1id < 0) {
          return "Invalid model: eq_obj1id out of bounds.";
        }
        if (obj2id >= m->nbody || obj2id < 0) {
          return "Invalid model: eq_obj2id out of bounds.";
        }
      } else if (objtype == mjOBJ_SITE) {
        if (obj1id >= m->nsite || obj1id < 0) {
          return "Invalid model: eq_obj1id out of bounds.";
        }
        if (obj2id >= m->nsite || obj2id < 0) {
          return "Invalid model: eq_obj2id out of bounds.";
        }
      } else {
        return "Invalid model: eq_objtype is not body or site.";
      }
      break;

    case mjEQ_FLEX:
    case mjEQ_FLEXSTRAIN:
      if (obj1id >= m->nflex || obj1id < 0) {
        return "Invalid model: eq_obj1id out of bounds.";
      }

      // -1 is the value used if second object is omitted
      if (obj2id != -1) {
        return "Invalid model: eq_obj2id must be -1.";
      }
      break;

    default:
      // might occur in case of the now-removed distance equality constraint
      mjERROR("unknown equality constraint type.");
    }
  }
  for (int i=0; i < m->nwrap; i++) {
    int wrap_objid = m->wrap_objid[i];
    switch ((mjtWrap) m->wrap_type[i]) {
    case mjWRAP_NONE:
    case mjWRAP_PULLEY:
      // wrap_objid not used.
      break;
    case mjWRAP_JOINT:
      if (wrap_objid >= m->njnt || wrap_objid < 0) {
        return "Invalid model: wrap_objid out of bounds.";
      }
      break;
    case mjWRAP_SITE:
      if (wrap_objid >= m->nsite || wrap_objid < 0) {
        return "Invalid model: wrap_objid out of bounds.";
      }
      break;
    case mjWRAP_SPHERE:
    case mjWRAP_CYLINDER:
      if (wrap_objid >= m->ngeom || wrap_objid < 0) {
        return "Invalid model: wrap_objid out of bounds.";
      }
      break;
    }
  }
  for (int i=0; i < m->nactuator; i++) {
    int actuator_trntype = m->actuator_trntype[i];
    int id = m->actuator_trnid[2*i];
    int idslider = m->actuator_trnid[2*i+1];
    switch ((mjtTrn) actuator_trntype) {
    case mjTRN_JOINT:
    case mjTRN_JOINTINPARENT:
      if (id < 0 || id >= m->njnt) {
        return "Invalid model: actuator_trnid out of bounds.";
      }
      break;
    case mjTRN_TENDON:
      if (id < 0 || id >= m->ntendon) {
        return "Invalid model: actuator_trnid out of bounds.";
      }
      break;
    case mjTRN_SITE:
      if (id < 0 || id >= m->nsite) {
        return "Invalid model: actuator_trnid out of bounds.";
      }
      break;
    case mjTRN_SLIDERCRANK:
      if (id < 0 || id >= m->nsite) {
        return "Invalid model: actuator_trnid out of bounds.";
      }
      if (idslider < 0 || idslider >= m->nsite) {
        return "Invalid model: actuator_trnid out of bounds.";
      }
      break;
    case mjTRN_BODY:
      if (id < 0 || id >= m->nbody) {
        return "Invalid model: actuator_trnid out of bounds.";
      }
      break;
    case mjTRN_SO3:
      // ball joint target (idslider == -1) or site + refsite target
      if (idslider == -1) {
        if (id < 0 || id >= m->njnt) {
          return "Invalid model: actuator_trnid out of bounds.";
        }
      } else {
        if (id < 0 || id >= m->nsite || idslider < 0 || idslider >= m->nsite) {
          return "Invalid model: actuator_trnid out of bounds.";
        }
      }
      break;
    case mjTRN_UNDEFINED:
      // actuator_trnid not used.
      break;
    }
  }
  for (int i=0; i < m->nsensor; i++) {
    mjtSensor sensor_type = m->sensor_type[i];
    int sensor_size;
    if (sensor_type == mjSENS_PLUGIN) {
      const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[m->sensor_plugin[i]]);
      if (!plugin->nsensordata) {
        mjERROR("`nsensordata` is a null function pointer for plugin at slot %d",
                m->plugin[m->sensor_plugin[i]]);
      }
      sensor_size = plugin->nsensordata(m, m->sensor_plugin[i], i);
    } else {
      sensor_size = sensorSize(sensor_type, m->sensor_dim[i]);
    }
    if (sensor_size < 0) {
      return "Invalid model: Bad sensor_type.";
    }
    int sensor_adr = m->sensor_adr[i];
    if (sensor_adr < 0 || sensor_adr + sensor_size > m->nsensordata) {
      return "Invalid model: sensor_adr out of bounds.";
    }
    int nobj = numObjects(m, m->sensor_objtype[i]);
    if (nobj == -2) {
      return "Invalid model: invalid sensor_objtype";
    }
    if (nobj != -1 && (m->sensor_objid[i] < 0 || m->sensor_objid[i] >= nobj)) {
      return "Invalid model: invalid sensor_objid";
    }
    nobj = numObjects(m, m->sensor_reftype[i]);
    if (nobj == -2) {
      return "Invalid model: invalid sensor_reftype";
    }
    if (nobj != -1 && (m->sensor_refid[i] < -1 || m->sensor_refid[i] >= nobj)) {
      return "Invalid model: invalid sensor_refid";
    }
    if (sensor_type == mjSENS_TACTILE) {
      int parent_body = m->geom_bodyid[m->sensor_refid[i]];
      int collision_geoms = 0;
      for (int b = 0; b < m->body_geomnum[parent_body]; ++b) {
        int geom_id = m->body_geomadr[parent_body]+b;
        if (m->geom_contype[geom_id] || m->geom_conaffinity[geom_id]) {
          collision_geoms++;
        }
      }
      if (collision_geoms == 0) {
        return "Touch sensor requires a body with at least one collision geom";
      }
    }
  }
  for (int i=0; i < m->nexclude; i++) {
    int exclude_body1 = (m->exclude_signature[i] & 0xFFFF);
    if (exclude_body1 >= m->nbody || exclude_body1 < 0) {
      return "Invalid model: exclude_body1 out of bounds.";
    }
    // unsigned shift: a signed >> sign-extends signatures whose high id is >= 0x8000
    int exclude_body2 = (int)((unsigned int)m->exclude_signature[i] >> 16);
    if (exclude_body2 >= m->nbody || exclude_body2 < 0) {
      return "Invalid model: exclude_body2 out of bounds.";
    }
  }
  for (int i=0; i < m->ntuple; i++) {
    for (int j=0; j < m->tuple_size[i]; j++) {
      int adr = m->tuple_adr[i] + j;
      int nobj = numObjects(m, m->tuple_objtype[adr]);
      if (nobj == -2) {
        return "Invalid model: invalid tuple_objtype";
      }
      if (nobj != -1 && (m->tuple_objid[adr] < 0 || m->tuple_objid[adr] >= nobj)) {
        return "Invalid model: invalid tuple_objid";
      }
    }
  }

  return NULL;
}
