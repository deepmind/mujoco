// Copyright 2024 DeepMind Technologies Limited
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

#include "user/user_api.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mujoco/mjspecmacro.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "engine/engine_io.h"
#include "engine/engine_support.h"
#include "engine/engine_util_errmem.h"
#include "user/user_cache.h"
#include "user/user_flexcomp.h"
#include "user/user_model.h"
#include "user/user_objects.h"
#include "user/user_resolver.h"
#include "user/user_resource.h"
#include "user/user_util.h"

namespace {

using mujoco::user::StringToVector;

}  // namespace

// global cache size in bytes (default 500MB)
static constexpr std::size_t kGlobalCacheSize = 500 * (1 << 20);


// create model
mjSpec* mj_makeSpec() {
  mjCModel* modelC = new mjCModel;
  return &modelC->spec;
}


// copy model
mjSpec* mj_copySpec(const mjSpec* s) {
  mjCModel* modelC = nullptr;
  try {
    modelC = new mjCModel(*static_cast<mjCModel*>(s->element));
  } catch (mjCError& e) {
    static_cast<mjCModel*>(s->element)->SetError(e);
    return nullptr;
  }
  return &modelC->spec;
}

// parse file into spec
mjSpec* mj_parse(
    const char* filename, const char* content_type, const mjVFS* vfs, char* error, int error_sz) {
  mjVFS                 local_vfs;
  mujoco::user::Cleanup cleanup;

  // early exit for existing XML workflow
  auto filepath = mujoco::user::FilePath(filename);
  if (filepath.Ext() == ".xml" ||
      filepath.Ext() == ".urdf" ||
      (content_type && std::strcmp(content_type, "text/xml") == 0)) {
    return mj_parseXML(filename, vfs, error, error_sz);
  }

  // If no VFS is provided, we'll create our own temporary one for the duration
  // of this function.
  if (vfs == nullptr) {
    mj_defaultVFS(&local_vfs);
    cleanup += [&local_vfs]() { mj_deleteVFS(&local_vfs); };

    vfs = &local_vfs;
  }

  mjResource* resource = mju_openResource("", filename, vfs, error, error_sz);
  // If we are unable to open the resource, we will create our own resource with
  // just the filename. This allows decoders that rely on other systems to fetch
  // their content to function without a custom resource provider.
  // For example, USD may use identifiers to assets that are strictly in memory
  // or that are fetched on a need-be basis via URI.
  if (resource) {
    cleanup += [resource]() { mju_closeResource(resource); };

    // A resource provider may only determine the content's format when it
    // opens the resource, and record it as the extension of the resource's
    // name (e.g. mjdb://kind/origin/uid becomes mjdb://kind/origin/uid.xml).
    // XML has no decoder, so hand such resources to the XML parser.
    auto resolved = mujoco::user::FilePath(resource->name);
    if (resolved.Ext() == ".xml" || resolved.Ext() == ".urdf") {
      return mj_parseXML(filename, vfs, error, error_sz);
    }
  } else {
    resource  = (mjResource*)mju_malloc(sizeof(mjResource));
    cleanup  += [resource]() {
      if (resource) mju_free(resource);
    };

    if (resource == nullptr) {
      if (error) {
        strncpy(error, "could not allocate memory", error_sz);
        error[error_sz - 1] = '\0';
      }
      return nullptr;
    }

    // clear out resource
    memset(resource, 0, sizeof(mjResource));

    // make space for filename
    std::string fullname  = filename;
    std::size_t n         = fullname.size();
    resource->name        = (char*)mju_malloc(sizeof(char) * (n + 1));
    cleanup              += [resource]() {
      if (resource) mju_free(resource->name);
    };

    if (resource->name == nullptr) {
      if (error) {
        strncpy(error, "could not allocate memory", error_sz);
        error[error_sz - 1] = '\0';
      }
      return nullptr;
    }
    memcpy(resource->name, fullname.c_str(), sizeof(char) * (n + 1));
  }

  mjSpec* spec = mju_decodeResource(resource, content_type, vfs, error, error_sz);
  if (spec == nullptr) {
    if (error && error_sz > 0 && !error[0]) {
      strncpy(error, "could not decode content", error_sz);
      error[error_sz - 1] = '\0';
    }
  }
  return spec;
}

// Encode spec or model as MJCF XML.
//
// If a spec is provided, it is saved directly to XML. This preserves the original
// structure and any user modifications in the spec.
// If spec is null, the model must be provided, and it is saved using mj_saveLastXML.
// mj_saveLastXML reconstructs the XML from the compiled model state, which may
// differ from the original XML (e.g., losing comments, reordering elements) but
// reflects the actual compiled model.
//
// Returns file size in bytes on success, -1 on failure.
static mjtSize encode_xml(
    const mjSpec* s, const mjModel* m, const char* filename, char* error, int error_sz) {
  if (s) {
    // Save directly from the spec
    if (mj_saveXML(s, filename, error, error_sz) < 0) { return -1; }
  } else {
    if (!m) {
      if (error) {
        strncpy(error, "model is required for XML encoding when spec is null", error_sz);
        error[error_sz - 1] = '\0';
      }
      return -1;
    }
    // Reconstruct XML from the compiled model, this will copy values back
    // from the mjModel into the last compiled spec and write that out
    // to disk. If there was no last compiled spec, such as when loading from
    // MJB, this will return fail and we return -1.
    if (!mj_saveLastXML(filename, m, error, error_sz)) { return -1; }
  }
  return static_cast<mjtSize>(std::filesystem::file_size(filename));
}

// Encode model as MJB (MuJoCo binary format).
// Requires a compiled model; spec-only encoding is not supported.
// Returns file size in bytes on success, -1 on failure.
static mjtSize encode_mjb(const mjModel* m, const char* filename, char* error, int error_sz) {
  if (!m) {
    if (error) {
      strncpy(error, "model is required for MJB encoding", error_sz);
      error[error_sz - 1] = '\0';
    }
    return -1;
  }
  mj_saveModel(m, filename, nullptr, 0);
  return static_cast<mjtSize>(std::filesystem::file_size(filename));
}

// Encode model as human-readable TXT (via mj_printModel).
// Requires a compiled model; spec-only encoding is not supported.
// Returns file size in bytes on success, -1 on failure.
static mjtSize encode_txt(const mjModel* m, const char* filename, char* error, int error_sz) {
  if (!m) {
    if (error) {
      strncpy(error, "model is required for TXT encoding", error_sz);
      error[error_sz - 1] = '\0';
    }
    return -1;
  }
  mj_printModel(m, filename);
  return static_cast<mjtSize>(std::filesystem::file_size(filename));
}

// encode spec/model to file
mjtSize mj_encode(const mjSpec*  s,
                  const mjModel* m,
                  const char*    filename,
                  const char*    content_type,
                  const mjVFS*   vfs,
                  char*          error,
                  int            error_sz) {
  // special case handling
  // TODO(shaves) write encoder/decoder paths for MJCF, TXT, MJB
  auto        filepath = mujoco::user::FilePath(filename);
  std::string ext      = filepath.Ext();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if (ext == ".xml" || (content_type && std::strcmp(content_type, "text/xml") == 0)) {
    return encode_xml(s, m, filename, error, error_sz);
  }

  if (ext == ".mjb" || (content_type && std::strcmp(content_type, "model/vnd.mujoco.mjb") == 0)) {
    return encode_mjb(m, filename, error, error_sz);
  }

  if (ext == ".txt" || (content_type && std::strcmp(content_type, "text/plain") == 0)) {
    return encode_txt(m, filename, error, error_sz);
  }

  const mjpEncoder* encoder = mjp_findEncoder(filename, content_type);
  if (!encoder) {
    if (error) {
      strncpy(error, "no encoder found", error_sz);
      error[error_sz - 1] = '\0';
    }
    return -1;
  }

  mjResource resource;
  memset(&resource, 0, sizeof(resource));
  resource.name = const_cast<char*>(filename);

  const mjtSize nbytes = encoder->encode(s, m, vfs, &resource);
  if (nbytes < 0 || !resource.data) {
    if (error) {
      strncpy(error, "encoder failed", error_sz);
      error[error_sz - 1] = '\0';
    }
    return -1;
  }

  mjtSize written = mju_writeResource(filename, resource.data, nbytes, vfs, error, error_sz);
  encoder->close_resource(&resource);

  if (written != nbytes) {
    if (error && error[0] == '\0') {
      strncpy(error, "write failed", error_sz);
      error[error_sz - 1] = '\0';
    }
    return -1;
  }

  return written;
}

// helper function to log compile time diagnostics
static void LogCompileTime(const double* t) {
  std::string body(1024, '\0');

  int n = std::snprintf(body.data(),
                        body.size(),
                        "  total:     %8.1f   (wall clock)\n"
                        "  assets:    %8.1f   -\n"
                        "    load:    %8.1f     (CPU time)\n"
                        "    hull:    %8.1f     -\n"
                        "    polygon: %8.1f     -\n"
                        "    inertia: %8.1f     -\n"
                        "    bvh:     %8.1f     -\n"
                        "    octree:  %8.1f     -\n"
                        "    texture: %8.1f     -\n"
                        "  other:     %8.1f   (wall clock)",
                        1e3 * t[mjCTIMER_TOTAL],
                        1e3 * t[mjCTIMER_ASSETS],
                        1e3 * t[mjCTIMER_MESH_LOAD],
                        1e3 * t[mjCTIMER_MESH_HULL],
                        1e3 * t[mjCTIMER_MESH_POLYGON],
                        1e3 * t[mjCTIMER_MESH_INERTIA],
                        1e3 * t[mjCTIMER_MESH_BVH],
                        1e3 * t[mjCTIMER_MESH_OCTREE],
                        1e3 * t[mjCTIMER_TEXTURE],
                        1e3 * (t[mjCTIMER_TOTAL] - t[mjCTIMER_ASSETS]));
  if (n > 0 && n < body.size()) { body.resize(n); }

  // send log message
  mjLogMessage msg = {.level   = mjLOG_INFO,
                      .topic   = mjTOPIC_TIME_CMP,
                      .subject = "compile time (ms)",
                      .body    = body.c_str()};
  mju_message(&msg);
}

// compile model
mjModel* mj_compile(mjSpec* s, const mjVFS* vfs) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjModel*  m      = modelC->Compile(vfs);

  // log compile time if model was compiled successfully
  if (m) { LogCompileTime(modelC->timer); }

  return m;
}


// recompile spec to model, preserving the state, return 0 on success
[[nodiscard]] int mj_recompile(mjSpec* s, const mjVFS* vfs, mjModel* m, mjData* d) {
  mjCModel*   modelC     = static_cast<mjCModel*>(s->element);
  std::string state_name = "state";

  mjCModel::mjRecompileState state;
  try {
    if (d) { modelC->SaveState(state_name, m, d, &state); }
    if (!modelC->Compile(vfs, &m)) {
      if (d) { mj_deleteData(d); }
      return -1;
    };
    if (d) {
      modelC->MakeData(m, &d);
      modelC->RestoreState(state_name, m, d, &state);
    }
  } catch (mjCError& e) {
    modelC->SetError(e);

    // m and d are deleted on failure, as when compilation fails
    if (d) { mj_deleteData(d); }
    mj_deleteModel(m);
    return -1;
  }
  return 0;
}


// copy a message into the user's error buffer, if any
static void CopyError(char* error, int error_sz, const std::string& msg) {
  if (error && error_sz > 0) { mju_strncpy(error, msg.c_str(), error_sz); }
}


// decode texture id into pixels, exactly the bytes compilation would store in tex_data for it
int mj_loadTexture(const mjModel* m,
                   int            id,
                   mjtByte**      data,
                   int*           width,
                   int*           height,
                   int*           colorspace,
                   char*          error,
                   int            error_sz) {
  if (id < 0 || id >= m->ntex) {
    CopyError(error, error_sz, "texture id " + std::to_string(id) + " out of range");
    return 1;
  }

  // embedded: a copy of the slice
  if (m->tex_adr[id] >= 0) {
    mjtSize nbytes =
        static_cast<mjtSize>(m->tex_nchannel[id]) * m->tex_width[id] * m->tex_height[id];
    *data = static_cast<mjtByte*>(mju_malloc(nbytes));
    memcpy(*data, m->tex_data + m->tex_adr[id], nbytes);
    if (width) { *width = m->tex_width[id]; }
    if (height) { *height = m->tex_height[id]; }
    if (colorspace) { *colorspace = m->tex_colorspace[id]; }
    return 0;
  }

  // by reference: the path and content type are in the model, the file is read with no VFS
  if (m->tex_pathadr[id] < 0) {
    CopyError(error, error_sz, "texture " + std::to_string(id) + " has no path");
    return 1;
  }
  const char* path = m->paths + m->tex_pathadr[id];
  const char* type = m->tex_contenttypeadr[id] >= 0 ? m->paths + m->tex_contenttypeadr[id] : "";
  std::string content_type = mjCBase::GetAssetContentType(path, type);
  std::array<char, 1024> err;
  mjResource*            resource = mju_openResource("", path, nullptr, err.data(), err.size());
  if (!resource) {
    CopyError(error, error_sz, err.data());
    return 1;
  }

  // decode as the compiler does, with the authored type, channels and colorspace as inputs
  std::vector<std::byte> image;
  unsigned int           w, h;
  bool                   is_srgb;
  int                    nch = m->tex_nchannel[id];
  try {
    mjCTexture::Decode(nullptr, resource, content_type, nch, image, w, h, is_srgb);
  } catch (mjCError& e) {
    mju_closeResource(resource);
    CopyError(error, error_sz, e.message);
    return 1;
  }
  mju_closeResource(resource);

  // cube or skybox from a single file: a square 3-channel image, repeated on the six faces
  if (m->tex_type[id] != mjTEXTURE_2D) {
    if (nch != 3) {
      CopyError(error, error_sz, "cube and skybox textures loaded from files must have 3 channels");
      return 1;
    }
    if (w != h) {
      CopyError(error, error_sz, "cube and skybox textures loaded from a file must be square");
      return 1;
    }
  }

  *data = static_cast<mjtByte*>(mju_malloc(image.size()));
  memcpy(*data, image.data(), image.size());
  if (width) { *width = w; }
  if (height) { *height = h; }
  if (colorspace) {
    *colorspace = m->tex_colorspace[id];
    if (*colorspace == mjCOLORSPACE_AUTO) {
      *colorspace = is_srgb ? mjCOLORSPACE_SRGB : mjCOLORSPACE_LINEAR;
    }
  }
  return 0;
}


// copy of m with every by-reference asset loaded into it; NULL with error on a file that does not
// resolve
mjModel* mj_embedAssets(const mjModel* m, char* error, int error_sz) {
  // the pixels of a by-reference texture, as mj_loadTexture returns them
  struct Pixels {
    mjtByte* data  = nullptr;
    int      width = 0, height = 0, colorspace = 0;
    mjtSize  nbytes(int nchannel) const { return static_cast<mjtSize>(nchannel) * width * height; }
  };
  std::vector<Pixels> pixels(m->ntex);
  auto                free_pixels = [&pixels]() {
    for (Pixels& p : pixels) { mju_free(p.data); }
  };

  // load every by-reference texture before allocating anything, so that a failure leaves nothing
  mjtSize ntexdata = m->ntexdata;
  for (int i = 0; i < m->ntex; i++) {
    if (m->tex_adr[i] != -1) { continue; }
    Pixels&                p = pixels[i];
    std::array<char, 1024> err;
    if (mj_loadTexture(m, i, &p.data, &p.width, &p.height, &p.colorspace, err.data(), err.size())) {
      free_pixels();
      CopyError(error, error_sz, "texture " + std::to_string(i) + ": " + err.data());
      return nullptr;
    }
    ntexdata += p.nbytes(m->tex_nchannel[i]);
  }

  // a new model, with ntexdata and nbuffer for the embedded textures
  mjModel* dest = nullptr;
  mj_makeModel(&dest,
               m->nq,
               m->nv,
               m->nu,
               m->nactuator,
               m->nout,
               m->na,
               m->nbody,
               m->nbvh,
               m->nbvhstatic,
               m->nbvhdynamic,
               m->noct,
               m->njnt,
               m->ntree,
               m->nM,
               m->nB,
               m->nC,
               m->nD,
               m->ngeom,
               m->nsite,
               m->ncam,
               m->nlight,
               m->nflex,
               m->nflexnode,
               m->nflexvert,
               m->nflexedge,
               m->nflexelem,
               m->nflexelemdata,
               m->nflexstiffness,
               m->nflexbending,
               m->nefm0dof,
               m->nefm0L,
               m->nefmCvert,
               m->nflexelemedge,
               m->nflexshelldata,
               m->nflextexcoord,
               m->nJfe,
               m->nmesh,
               m->nmeshvert,
               m->nmeshnormal,
               m->nmeshtexcoord,
               m->nmeshface,
               m->nmeshgraph,
               m->nmeshpoly,
               m->nmeshpolyvert,
               m->nmeshpolymap,
               m->nskin,
               m->nskinvert,
               m->nskintexvert,
               m->nskinface,
               m->nskinbone,
               m->nskinbonevert,
               m->nhfield,
               m->nhfielddata,
               m->ntex,
               ntexdata,
               m->nmat,
               m->npair,
               m->nexclude,
               m->neq,
               m->ntendon,
               m->nJten,
               m->nwrap,
               m->nsensor,
               m->nnumeric,
               m->nnumericdata,
               m->ntext,
               m->ntextdata,
               m->ntuple,
               m->ntupledata,
               m->nkey,
               m->nmocap,
               m->nplugin,
               m->npluginattr,
               m->nuser_body,
               m->nuser_jnt,
               m->nuser_geom,
               m->nuser_site,
               m->nuser_cam,
               m->nuser_tendon,
               m->nuser_actuator,
               m->nuser_sensor,
               m->nnames,
               m->npaths,
               m->nplastic);
  if (!dest) {
    free_pixels();
    CopyError(error, error_sz, "could not allocate model");
    return nullptr;
  }

  // every scalar field, keeping the new buffer and sizes
  void*   buffer  = dest->buffer;
  mjtSize nbuffer = dest->nbuffer;
  *dest           = *m;
  dest->buffer    = buffer;
  dest->nbuffer   = nbuffer;
  dest->ntexdata  = ntexdata;
  mj_setPtrModel(dest);

  // every array but tex_data
  {
    MJMODEL_POINTERS_PREAMBLE(m)
#define X(type, name, nr, nc)                                                         \
  if (static_cast<const void*>(&m->name) != static_cast<const void*>(&m->tex_data)) { \
    memcpy(dest->name, m->name, sizeof(type) * (m->nr) * nc);                         \
  }
    MJMODEL_POINTERS
#undef X
  }

  // tex_data in texture order, as compilation lays it out
  mjtSize adr = 0;
  for (int i = 0; i < m->ntex; i++) {
    const Pixels& p = pixels[i];
    mjtSize       nbytes;
    if (m->tex_adr[i] >= 0) {
      nbytes = static_cast<mjtSize>(m->tex_nchannel[i]) * m->tex_width[i] * m->tex_height[i];
      memcpy(dest->tex_data + adr, m->tex_data + m->tex_adr[i], nbytes);
    } else {
      nbytes = p.nbytes(m->tex_nchannel[i]);
      memcpy(dest->tex_data + adr, p.data, nbytes);
      dest->tex_width[i]      = p.width;
      dest->tex_height[i]     = p.height;
      dest->tex_colorspace[i] = p.colorspace;
    }
    dest->tex_adr[i]  = adr;
    adr              += nbytes;
  }

  free_pixels();
  return dest;
}


// set frame for all elements of a body
static void SetFrame(mjsBody* body, mjtObj objtype, mjsFrame* frame) {
  mjsElement* el = mjs_firstChild(body, objtype, 0);
  while (el) {
    if (frame->element != el && mjs_getFrame(el) == nullptr) { mjs_setFrame(el, frame); }
    el = mjs_nextChild(body, el, 0);
  }
}


// check if an element of the list, inside the frame if one is given, is not in the model
template <class T>
static bool InOtherModel(const std::vector<T*>& list,
                         const mjCModel*        model,
                         const mjCFrame*        frame) {
  for (const T* element : list) {
    if ((!frame || frame->IsAncestor(element->frame)) && element->model != model) { return true; }
  }
  return false;
}


// check if attaching a body, or the part of it inside a frame if one is given, takes an element
// which is not in the model of the body: an earlier attachment by reference moved it
static bool TakesMoved(const mjCBody* body, const mjCFrame* frame = nullptr) {
  const mjCModel* model = body->model;
  if (InOtherModel(body->GetList<mjCBody>(), model, frame) ||
      InOtherModel(body->GetList<mjCFrame>(), model, frame) ||
      InOtherModel(body->GetList<mjCGeom>(), model, frame) ||
      InOtherModel(body->GetList<mjCJoint>(), model, frame) ||
      InOtherModel(body->GetList<mjCSite>(), model, frame) ||
      InOtherModel(body->GetList<mjCCamera>(), model, frame) ||
      InOtherModel(body->GetList<mjCLight>(), model, frame)) {
    return true;
  }
  for (const mjCBody* child : body->GetList<mjCBody>()) {
    if ((!frame || frame->IsAncestor(child->frame)) && TakesMoved(child)) { return true; }
  }
  return false;
}


// attach body to a frame of the parent
static mjsElement* attachBody(mjCFrame*      parent,
                              const mjCBody* child,
                              const char*    prefix,
                              const char*    suffix) {
  mjCBody* mutable_child = const_cast<mjCBody*>(child);
  mutable_child->prefix  = prefix ? prefix : "";
  mutable_child->suffix  = suffix ? suffix : "";
  try {
    *parent += *mutable_child;
  } catch (mjCError& e) {
    parent->model->SetError(e);
    return nullptr;
  }
  mjsBody* attached_body = parent->last_attached;
  parent->last_attached  = nullptr;
  return attached_body->element;
}


// attach frame to a parent body
static mjsElement* attachFrame(mjCBody*        parent,
                               const mjCFrame* child,
                               const char*     prefix,
                               const char*     suffix) {
  mjCFrame* mutable_child = const_cast<mjCFrame*>(child);
  mutable_child->prefix   = prefix ? prefix : "";
  mutable_child->suffix   = suffix ? suffix : "";
  try {
    *parent += *mutable_child;
  } catch (mjCError& e) {
    parent->model->SetError(e);
    return nullptr;
  }
  mjsFrame* attached_frame = parent->last_attached;
  parent->last_attached    = nullptr;
  return attached_frame->element;
}


// attach child body to a parent site
static mjsElement* attachToSite(mjCSite*       parent,
                                const mjCBody* child,
                                const char*    prefix,
                                const char*    suffix) {
  mjSpec*   spec  = mjs_getSpec(parent->spec.element);
  mjCBody*  body  = parent->Body();
  mjCFrame* frame = body->AddFrame(parent->frame);
  frame->SetParent(body);
  frame->spec.pos[0]  = parent->spec.pos[0];
  frame->spec.pos[1]  = parent->spec.pos[1];
  frame->spec.pos[2]  = parent->spec.pos[2];
  frame->spec.quat[0] = parent->spec.quat[0];
  frame->spec.quat[1] = parent->spec.quat[1];
  frame->spec.quat[2] = parent->spec.quat[2];
  frame->spec.quat[3] = parent->spec.quat[3];
  mjs_resolveOrientation(frame->spec.quat,
                         spec->compiler.degree,
                         spec->compiler.eulerseq,
                         &parent->spec.alt);
  return attachBody(frame, child, prefix, suffix);
}


// attach child frame to a parent site
static mjsElement* attachFrameToSite(mjCSite*        parent,
                                     const mjCFrame* child,
                                     const char*     prefix,
                                     const char*     suffix) {
  mjSpec*   spec  = mjs_getSpec(parent->spec.element);
  mjCBody*  body  = parent->Body();
  mjCFrame* frame = body->AddFrame(parent->frame);
  frame->SetParent(body);
  frame->spec.pos[0]  = parent->spec.pos[0];
  frame->spec.pos[1]  = parent->spec.pos[1];
  frame->spec.pos[2]  = parent->spec.pos[2];
  frame->spec.quat[0] = parent->spec.quat[0];
  frame->spec.quat[1] = parent->spec.quat[1];
  frame->spec.quat[2] = parent->spec.quat[2];
  frame->spec.quat[3] = parent->spec.quat[3];
  mjs_resolveOrientation(frame->spec.quat,
                         spec->compiler.degree,
                         spec->compiler.eulerseq,
                         &parent->spec.alt);

  mjsElement* attached_frame = attachFrame(body, child, prefix, suffix);
  mjs_setFrame(attached_frame, &frame->spec);
  return attached_frame;
}

mjsElement* mjs_attach(mjsElement*       parent,
                       const mjsElement* child,
                       const char*       prefix,
                       const char*       suffix) {
  if (!parent) {
    mju_error("parent element is null");
    return nullptr;
  }
  if (!child) {
    mju_error("child element is null");
    return nullptr;
  }
  mjCModel*     model      = static_cast<mjCModel*>(mjs_getSpec(parent)->element);
  const mjSpec* child_spec = nullptr;
  if (child->elemtype == mjOBJ_MODEL) {
    child_spec = &(static_cast<const mjCModel*>(child)->spec);
  } else {
    child_spec = &(static_cast<const mjCBase*>(child)->model->spec);
  }

  // the elements of a spec attached by reference are now in the spec it was attached to
  if (child->elemtype == mjOBJ_MODEL && static_cast<const mjCModel*>(child)->IsAttached()) {
    model->SetError(mjCError(0,
                             "Child spec is already attached by reference; enable deep copy to "
                             "attach a spec more than once."));
    return nullptr;
  }

  // attaching by reference moves elements of the child to the parent: an element which is in the
  // parent, its own or moved there by an earlier attachment, cannot be moved there, and an element
  // which an earlier attachment moved cannot be moved again with a body or frame which holds it
  if (!model->GetDeepCopy()) {
    if (child_spec == &model->spec) {
      model->SetError(mjCError(0,
                               "Child element is in the parent spec; enable deep copy to attach "
                               "a copy of it."));
      return nullptr;
    }
    bool moved = false;
    if (child->elemtype == mjOBJ_BODY) {
      moved = TakesMoved(static_cast<const mjCBody*>(child));
    } else if (child->elemtype == mjOBJ_FRAME) {
      const mjCFrame* frame = static_cast<const mjCFrame*>(child);
      moved                 = frame->GetParent() && TakesMoved(frame->GetParent(), frame);
    }
    if (moved) {
      model->SetError(mjCError(0,
                               "Child element contains elements which are already attached by "
                               "reference; enable deep copy to attach a copy of it."));
      return nullptr;
    }
  }

  // handle global attribute conflicts
  if (child_spec && child_spec != &model->spec) {
    std::string error_msg, warning_subject, warning_body;
    bool success = mujoco::ResolveConflicts(&model->spec,
                                            child_spec,
                                            static_cast<mjtConflict>(model->spec.compiler.conflict),
                                            &error_msg,
                                            &warning_subject,
                                            &warning_body);

    if (!success) {
      model->SetError(mjCError(0, "%s", error_msg.c_str()));
      return nullptr;
    }

    if (!warning_body.empty()) { model->AddGroupedWarning(warning_subject, warning_body); }
  }

  if (child->elemtype == mjOBJ_MODEL) {
    mjCModel* child_model = static_cast<mjCModel*>((mjsElement*)child);
    mjsBody*  worldbody   = mjs_findBody(&child_model->spec, "world");
    if (!worldbody) {
      model->SetError(mjCError(0, "Child does not have a world body."));
      return nullptr;
    }
    mjsFrame* worldframe = mjs_addFrame(worldbody, nullptr);
    SetFrame(worldbody, mjOBJ_BODY, worldframe);
    SetFrame(worldbody, mjOBJ_SITE, worldframe);
    SetFrame(worldbody, mjOBJ_FRAME, worldframe);
    SetFrame(worldbody, mjOBJ_JOINT, worldframe);
    SetFrame(worldbody, mjOBJ_GEOM, worldframe);
    SetFrame(worldbody, mjOBJ_LIGHT, worldframe);
    SetFrame(worldbody, mjOBJ_CAMERA, worldframe);
    child = worldframe->element;
  }
  mjsElement* result = nullptr;
  switch (parent->elemtype) {
    case mjOBJ_FRAME:
      if (child->elemtype == mjOBJ_BODY) {
        result = attachBody(static_cast<mjCFrame*>(parent),
                            static_cast<const mjCBody*>(child),
                            prefix,
                            suffix);
      } else if (child->elemtype == mjOBJ_FRAME) {
        mjsBody* parent_body = mjs_getParent(parent);
        if (!parent_body) {
          model->SetError(mjCError(0, "Frame does not have a parent body."));
          return nullptr;
        }
        mjCFrame*   frame          = static_cast<mjCFrame*>(parent);
        mjsElement* attached_frame = attachFrame(static_cast<mjCBody*>(parent_body->element),
                                                 static_cast<const mjCFrame*>(child),
                                                 prefix,
                                                 suffix);
        if (mjs_setFrame(attached_frame, &frame->spec)) { return nullptr; }
        result = attached_frame;
      } else {
        model->SetError(mjCError(0, "child element is not a body or frame"));
        return nullptr;
      }
      break;
    case mjOBJ_BODY:
      if (child->elemtype == mjOBJ_FRAME) {
        result = attachFrame(static_cast<mjCBody*>(parent),
                             static_cast<const mjCFrame*>(child),
                             prefix,
                             suffix);
      } else {
        model->SetError(mjCError(0, "child element is not a frame"));
        return nullptr;
      }
      break;
    case mjOBJ_SITE:
      if (child->elemtype == mjOBJ_BODY) {
        result = attachToSite(static_cast<mjCSite*>(parent),
                              static_cast<const mjCBody*>(child),
                              prefix,
                              suffix);
      } else if (child->elemtype == mjOBJ_FRAME) {
        result = attachFrameToSite(static_cast<mjCSite*>(parent),
                                   static_cast<const mjCFrame*>(child),
                                   prefix,
                                   suffix);
      } else {
        model->SetError(mjCError(0, "child element is not a body or frame"));
        return nullptr;
      }
      break;
    default:
      model->SetError(mjCError(0, "parent element is not a frame, body or site"));
      return nullptr;
  }

  if (result) {
    if (child_spec != &model->spec && (!prefix || !prefix[0])) {
      model->AddWarning(
          "Attaching a spec with an empty prefix may prevent the model from being saved "
          "correctly.");
    }

    // mark all warnings accumulated so far as attach-phase
    model->SetAttachWarningBoundary();
  }
  return result;
}


// get error message from model
const char* mjs_getError(mjSpec* s) {
  if (!s) {
    mju_error("spec is null");
    return nullptr;
  }
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  return modelC->GetError().message;
}


// get compiler timers from model
const double* mjs_getTimer(mjSpec* s) {
  if (!s) { return nullptr; }
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  return modelC->timer;
}

// check if model has warnings (but no error)
// TODO(tassa): delete this function
int mjs_isWarning(mjSpec* s) {
  if (!s) { return 0; }
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  return modelC->GetError().message[0] == '\0' && !modelC->GetWarnings().empty();
}

// get number of warnings
int mjs_numWarnings(const mjSpec* spec) {
  if (!spec) { return 0; }
  const mjCModel* modelC = static_cast<const mjCModel*>(spec->element);
  return static_cast<int>(modelC->GetWarnings().size());
}

// get the i-th warning message
const char* mjs_getWarning(const mjSpec* spec, int index) {
  if (!spec) { return nullptr; }
  const mjCModel* modelC = static_cast<const mjCModel*>(spec->element);
  if (index < 0 || index >= static_cast<int>(modelC->GetWarnings().size())) { return nullptr; }
  return modelC->GetWarnings()[index].c_str();
}

// delete model
void mj_deleteSpec(mjSpec* s) {
  if (s) {
    mjCModel* model = static_cast<mjCModel*>(s->element);
    model->Release();
  }
}


// add spec (model asset) to spec
void mjs_addSpec(mjSpec* s, mjSpec* child) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  model->AppendSpec(child);
}


// activate plugin
int mjs_activatePlugin(mjSpec* s, const char* name) {
  int              plugin_slot = -1;
  const mjpPlugin* plugin      = mjp_getPlugin(name, &plugin_slot);
  if (!plugin) { return -1; }
  mjCModel* model = static_cast<mjCModel*>(s->element);
  model->ActivatePlugin(plugin, plugin_slot);
  return 0;
}


// set deep copy flag
int mjs_setDeepCopy(mjSpec* s, int deepcopy) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  model->SetDeepCopy(deepcopy);
  return 0;
}


// copy real-valued arrays from model to spec, returns 1 on success
int mj_copyBack(mjSpec* s, const mjModel* m) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  return model->CopyBack(m);
}


// remove body from mjSpec, return 0 on success
int mjs_delete(mjSpec* s, mjsElement* element) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  if (model->IsAttached()) {
    model->SetError(mjCError(nullptr, "Cannot delete element from an attached mjSpec."));
    return -1;
  }
  if (!element) {
    model->SetError(mjCError(nullptr, "Element is null."));
    return -1;
  }
  try {
    if (element->elemtype == mjOBJ_DEFAULT) {
      mjCDef* def  = static_cast<mjCDef*>(element);
      *model      -= *def;
    } else {
      *model -= element;
    }
    return 0;
  } catch (mjCError& e) {
    model->SetError(e);
    return -1;
  }
}


// add child body to body, return child spec
mjsBody* mjs_addBody(mjsBody* bodyspec, const mjsDefault* defspec) {
  mjCDef*  def  = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCBody* body = static_cast<mjCBody*>(bodyspec->element)->AddBody(def);
  return &body->spec;
}


// add site to body, return site spec
mjsSite* mjs_addSite(mjsBody* bodyspec, const mjsDefault* defspec) {
  mjCDef*  def  = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCBody* body = static_cast<mjCBody*>(bodyspec->element);
  mjCSite* site = body->AddSite(def);
  return &site->spec;
}


// add joint to body
mjsJoint* mjs_addJoint(mjsBody* bodyspec, const mjsDefault* defspec) {
  mjCDef*   def   = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCBody*  body  = static_cast<mjCBody*>(bodyspec->element);
  mjCJoint* joint = body->AddJoint(def);
  return &joint->spec;
}


// add free joint to body
mjsJoint* mjs_addFreeJoint(mjsBody* bodyspec) {
  mjCBody*  body  = static_cast<mjCBody*>(bodyspec->element);
  mjCJoint* joint = body->AddFreeJoint();
  return &joint->spec;
}


// add geom to body
mjsGeom* mjs_addGeom(mjsBody* bodyspec, const mjsDefault* defspec) {
  mjCDef*  def  = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCBody* body = static_cast<mjCBody*>(bodyspec->element);
  mjCGeom* geom = body->AddGeom(def);
  return &geom->spec;
}


// add camera to body
mjsCamera* mjs_addCamera(mjsBody* bodyspec, const mjsDefault* defspec) {
  mjCDef*    def    = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCBody*   body   = static_cast<mjCBody*>(bodyspec->element);
  mjCCamera* camera = body->AddCamera(def);
  return &camera->spec;
}


// add light to body
mjsLight* mjs_addLight(mjsBody* bodyspec, const mjsDefault* defspec) {
  mjCDef*   def   = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCBody*  body  = static_cast<mjCBody*>(bodyspec->element);
  mjCLight* light = body->AddLight(def);
  return &light->spec;
}


// add flex to model
mjsFlex* mjs_addFlex(mjSpec* s) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCFlex*  flex   = modelC->AddFlex();
  return &flex->spec;
}


// helper: convert type string to mjtFcompType
static mjtFcompType FlexcompTypeFromStr(const char* type) {
  // clang-format off
  if (!type || !strcmp(type, "grid"))    return mjFCOMPTYPE_GRID;
  if (!strcmp(type, "box"))              return mjFCOMPTYPE_BOX;
  if (!strcmp(type, "cylinder"))         return mjFCOMPTYPE_CYLINDER;
  if (!strcmp(type, "ellipsoid"))        return mjFCOMPTYPE_ELLIPSOID;
  if (!strcmp(type, "square"))           return mjFCOMPTYPE_SQUARE;
  if (!strcmp(type, "disc"))             return mjFCOMPTYPE_DISC;
  if (!strcmp(type, "circle"))           return mjFCOMPTYPE_CIRCLE;
  if (!strcmp(type, "mesh"))             return mjFCOMPTYPE_MESH;
  if (!strcmp(type, "gmsh"))             return mjFCOMPTYPE_GMSH;
  if (!strcmp(type, "direct"))           return mjFCOMPTYPE_DIRECT;
  return mjFCOMPTYPE_GRID;  // default
  // clang-format on
}

// helper: convert dof string to mjtDof
static mjtDof FlexcompDofFromStr(const char* dof) {
  // clang-format off
  if (!dof || !strcmp(dof, "full"))      return mjFCOMPDOF_FULL;
  if (!strcmp(dof, "radial"))            return mjFCOMPDOF_RADIAL;
  if (!strcmp(dof, "trilinear"))         return mjFCOMPDOF_TRILINEAR;
  if (!strcmp(dof, "quadratic"))         return mjFCOMPDOF_QUADRATIC;
  if (!strcmp(dof, "2d"))                return mjFCOMPDOF_2D;
  return mjFCOMPDOF_FULL;  // default
  // clang-format on
}


// add flexcomp: create flex with auto-generated bodies/joints
mjsFlex* mjs_makeFlex(mjsBody*     body,
                      const char*  name,
                      const char*  type,
                      int          dim,
                      const char*  dof,
                      const int    count[3],
                      const int    cellcount[3],
                      const double spacing[3],
                      const double scale[3],
                      double       radius,
                      double       mass,
                      double       inertiabox,
                      int          equality,
                      int          rigid,
                      int          flatskin,
                      int          elastic2d,
                      const double pos[3],
                      const double quat[4],
                      const double origin[3],
                      const char*  file,
                      const mjVFS* vfs) {
  if (!body || !name) {
    mju_error("mjs_makeFlex: body and name must not be null");
    return nullptr;
  }

  mjCModel* model = static_cast<mjCBody*>(body->element)->model;

  // create temporary flexcomp with defaults
  mjCFlexcomp fcomp;
  fcomp.name    = name;
  fcomp.type    = FlexcompTypeFromStr(type);
  fcomp.doftype = FlexcompDofFromStr(dof);

  // topology
  if (count) {
    fcomp.count[0] = count[0];
    fcomp.count[1] = count[1];
    fcomp.count[2] = count[2];
  }
  if (cellcount) {
    fcomp.cellcount[0] = cellcount[0];
    fcomp.cellcount[1] = cellcount[1];
    fcomp.cellcount[2] = cellcount[2];
  }
  if (spacing) {
    fcomp.spacing[0] = spacing[0];
    fcomp.spacing[1] = spacing[1];
    fcomp.spacing[2] = spacing[2];
  }
  if (scale) {
    fcomp.scale[0] = scale[0];
    fcomp.scale[1] = scale[1];
    fcomp.scale[2] = scale[2];
  }
  if (origin) {
    fcomp.has_origin = true;
    fcomp.origin[0]  = origin[0];
    fcomp.origin[1]  = origin[1];
    fcomp.origin[2]  = origin[2];
  }

  // physics
  fcomp.has_dim               = (dim > 0);
  fcomp.def.spec.flex->dim    = dim;
  fcomp.def.spec.flex->radius = radius;
  if (mass > 0) fcomp.mass = mass;
  if (inertiabox > 0) fcomp.inertiabox = inertiabox;
  fcomp.equality = equality;
  fcomp.rigid    = rigid;

  fcomp.def.spec.flex->flatskin  = flatskin;
  fcomp.def.spec.flex->elastic2d = elastic2d;

  // pose
  if (pos) {
    fcomp.pos[0] = pos[0];
    fcomp.pos[1] = pos[1];
    fcomp.pos[2] = pos[2];
  }
  if (quat) {
    fcomp.quat[0] = quat[0];
    fcomp.quat[1] = quat[1];
    fcomp.quat[2] = quat[2];
    fcomp.quat[3] = quat[3];
  }

  // file is the name of a mesh asset if one exists, otherwise a file name
  if (file) {
    if (model->FindObject(mjOBJ_MESH, file)) {
      fcomp.mesh = file;
      if (!type) { fcomp.type = mjFCOMPTYPE_MESH; }
    } else {
      fcomp.file         = file;
      fcomp.file_or_mesh = true;
    }
  }

  // call Make
  char error[500] = "";
  if (!fcomp.Make(body, error, sizeof(error), vfs)) {
    model->SetError(mjCError(nullptr, "%s", error));
    return nullptr;
  }

  // return the flex that was created (last flex in model)
  mjCFlex* flex = model->Flexes().back();
  return &flex->spec;
}


// add frame to body
mjsFrame* mjs_addFrame(mjsBody* bodyspec, mjsFrame* parentframe) {
  mjCFrame* parentframeC = 0;
  if (parentframe) { parentframeC = static_cast<mjCFrame*>(parentframe->element); }
  mjCBody*  body   = static_cast<mjCBody*>(bodyspec->element);
  mjCFrame* frameC = body->AddFrame(parentframeC);
  frameC->SetParent(body);
  return &frameC->spec;
}


// add mesh to model
mjsMesh* mjs_addMesh(mjSpec* s, const mjsDefault* defspec) {
  mjCDef*   def    = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCMesh*  mesh   = modelC->AddMesh(def);
  return &mesh->spec;
}


// add height field to model
mjsHField* mjs_addHField(mjSpec* s) {
  mjCModel*  modelC      = static_cast<mjCModel*>(s->element);
  mjCHField* heightField = modelC->AddHField();
  return &heightField->spec;
}


// add skin to model
mjsSkin* mjs_addSkin(mjSpec* s) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCSkin*  skin   = modelC->AddSkin();
  return &skin->spec;
}


// add texture to model
mjsTexture* mjs_addTexture(mjSpec* s, const mjsDefault* defspec) {
  mjCModel*   modelC  = static_cast<mjCModel*>(s->element);
  mjCDef*     def     = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCTexture* texture = modelC->AddTexture(def);
  return &texture->spec;
}


// add material to model
mjsMaterial* mjs_addMaterial(mjSpec* s, const mjsDefault* defspec) {
  mjCModel*    modelC   = static_cast<mjCModel*>(s->element);
  mjCDef*      def      = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCMaterial* material = modelC->AddMaterial(def);
  return &material->spec;
}

// Sets the vertices and normals of a mesh.
int mjs_makeMesh(mjsMesh* mesh, mjtMeshBuiltin builtin, double* params, int nparams) {
  mjCMesh*  meshC = static_cast<mjCMesh*>(mesh->element);
  mjCModel* m     = meshC->model;
  switch (builtin) {
    case mjMESH_BUILTIN_HEMISPHERE: {
      if (nparams != 1) {
        m->SetError(mjCError(0, "Hemisphere mesh type requires 1 parameter"));
        return -1;
      }
      int subdiv = static_cast<int>(params[0]);
      if (subdiv < 0) {
        m->SetError(mjCError(0, "Hemisphere resolution cannot be negative"));
        return -1;
      }
      if (subdiv > 10) {
        m->SetError(mjCError(0, "Hemisphere resolution cannot be greater than 10"));
        return -1;
      }
      meshC->MakeHemisphere(subdiv, /*make_faces*/ true, /*make_cap*/ true);
      return 0;
    }

    case mjMESH_BUILTIN_SPHERE: {
      if (nparams != 1) {
        m->SetError(mjCError(0, "Sphere mesh type requires 1 parameter"));
        return -1;
      }
      int subdiv = static_cast<int>(params[0]);
      if (subdiv < 0) {
        m->SetError(mjCError(0, "Sphere subdivision cannot be negative"));
        return -1;
      }
      if (subdiv > 4) {
        m->SetError(mjCError(0, "Sphere subdivision cannot be greater than 4"));
        return -1;
      }
      meshC->MakeSphere(subdiv, /*make_faces*/ true);
      return 0;
    }

    case mjMESH_BUILTIN_SUPERSPHERE: {
      if (nparams != 3) {
        m->SetError(mjCError(0, "Supersphere mesh type requires 3 parameters"));
        return -1;
      }
      int res = static_cast<int>(params[0]);
      if (res < 3) {
        m->SetError(mjCError(0, "Supersphere resolution must be greater than 2"));
        return -1;
      }
      double e = params[1];
      if (e < 0) {
        m->SetError(mjCError(0, "Supersphere 'e' cannot be negative"));
        return -1;
      }
      double n = params[2];
      if (n < 0) {
        m->SetError(mjCError(0, "Supersphere 'n' cannot be negative"));
        return -1;
      }
      meshC->MakeSupersphere(res, e, n);
      return 0;
    }

    case mjMESH_BUILTIN_SUPERTORUS: {
      if (nparams != 4) {
        m->SetError(mjCError(0, "Supertorus mesh type requires 4 parameters"));
        return -1;
      }
      int res = static_cast<int>(params[0]);
      if (res < 3) {
        m->SetError(mjCError(0, "Supertorus resolution must be greater than 3"));
        return -1;
      }
      double radius = params[1];
      if (radius <= 0 || radius > 1) {
        m->SetError(mjCError(0, "Supertorus radius must be in (0, 1]"));
        return -1;
      }
      double s = params[2];
      if (s <= 0) {
        m->SetError(mjCError(0, "Supertorus 's' must be greater than 0"));
        return -1;
      }
      double t = params[3];
      if (t <= 0) {
        m->SetError(mjCError(0, "Supertorus 't' must be greater than 0"));
        return -1;
      }
      meshC->MakeSupertorus(res, radius, s, t);
      return 0;
    }

    case mjMESH_BUILTIN_WEDGE: {
      if (nparams != 5) {
        m->SetError(mjCError(0, "Wedge builtin mesh types require 5 parameters"));
        return -1;
      }
      int resolution[2] = {static_cast<int>(params[0]), static_cast<int>(params[1])};

      double fov[2] = {params[2], params[3]};
      double gamma  = params[4];
      if (fov[0] <= 0 || fov[0] > 180) {
        m->SetError(mjCError(0, "fov[0] must be a float between (0, 180] degrees"));
        return -1;
      }
      if (fov[1] <= 0 || fov[1] > 90) {
        m->SetError(mjCError(0, "`fov[1]` must be a float between (0, 90] degrees"));
        return -1;
      }
      if (resolution[0] <= 0 || resolution[1] <= 0) {
        m->SetError(mjCError(0, "Horizontal and vertical resolutions must be positive"));
        return -1;
      }
      if (gamma < 0 || gamma > 1) {
        m->SetError(mjCError(0, "`gamma` must be a nonnegative float between [0, 1]"));
        return -1;
      }
      meshC->MakeWedge(resolution, fov, gamma);
      return 0;
    }

    case mjMESH_BUILTIN_PLATE: {
      if (nparams != 2) {
        m->SetError(mjCError(0, "Plate builtin mesh type requires 2 parameters"));
        return -1;
      }
      int resolution[2] = {static_cast<int>(params[0]), static_cast<int>(params[1])};
      if (resolution[0] <= 0 || resolution[1] <= 0) {
        m->SetError(mjCError(0, "Horizontal and vertical resolutions must be positive"));
        return -1;
      }
      meshC->MakeRect(resolution);
      return 0;
    }

    case mjMESH_BUILTIN_CONE: {
      if (nparams != 2) {
        m->SetError(mjCError(0, "Cone mesh type requires 2 parameters"));
        return -1;
      }
      int nedge = static_cast<int>(params[0]);
      if (nedge < 3) {
        m->SetError(mjCError(0, "Cone nvert must be at least 3"));
        return -1;
      }
      if (!(params[1] >= 0 && params[1] <= 1)) {
        m->SetError(mjCError(0, "Cone radius must be in [0, 1]"));
        return -1;
      }
      meshC->MakeCone(nedge, params[1]);
      return 0;
    }

    case mjMESH_BUILTIN_GRID: {
      if (nparams != 6 && nparams != 7) {
        m->SetError(
            mjCError(0, "Grid mesh type requires 6 or 7 parameters (count[3], spacing[3], [dim])"));
        return -1;
      }
      int    count[3]   = {static_cast<int>(params[0]),
                           static_cast<int>(params[1]),
                           static_cast<int>(params[2])};
      double spacing[3] = {params[3], params[4], params[5]};
      int    dim        = (nparams == 7) ? static_cast<int>(params[6]) : 3;
      if (dim < 2 || dim > 3) {
        m->SetError(mjCError(0, "Grid dim must be 2 or 3"));
        return -1;
      }
      for (int i = 0; i < dim; i++) {
        if (count[i] < 1) {
          m->SetError(mjCError(0, "Grid count must be positive"));
          return -1;
        }
      }
      for (int i = dim; i < 3; i++) {
        if (count[i] != 1) {
          m->SetError(mjCError(0, "Grid count must be 1 along dimensions beyond dim"));
          return -1;
        }
      }
      meshC->MakeGrid(count, spacing, dim, false);
      if ((dim == 3 ? mesh->usertet : mesh->userface)->empty()) {
        m->SetError(mjCError(0, "Failed to create grid mesh"));
        return -1;
      }
      if (dim < 3 && mesh->inertia == mjMESH_INERTIA_LEGACY) {
        mesh->inertia = mjMESH_INERTIA_SHELL;
      }
      return 0;
    }

    case mjMESH_BUILTIN_BOX:
    case mjMESH_BUILTIN_CYLINDER:
    case mjMESH_BUILTIN_ELLIPSOID: {
      if (nparams != 6 && nparams != 7) {
        m->SetError(mjCError(0,
                             "Box/cylinder/ellipsoid mesh type requires 6 or 7 parameters "
                             "(count[3], spacing[3], [dim])"));
        return -1;
      }
      int    count[3]   = {static_cast<int>(params[0]),
                           static_cast<int>(params[1]),
                           static_cast<int>(params[2])};
      double spacing[3] = {params[3], params[4], params[5]};
      int    dim        = (nparams == 7) ? static_cast<int>(params[6]) : 3;
      if (dim < 2 || dim > 3) {
        m->SetError(mjCError(0, "Mesh dim must be 2 or 3 for box/cylinder/ellipsoid"));
        return -1;
      }
      for (int i = 0; i < 3; i++) {
        if (count[i] < 2) {
          m->SetError(mjCError(0, "Box/cylinder/ellipsoid count must be at least 2"));
          return -1;
        }
      }
      if (builtin == mjMESH_BUILTIN_BOX) {
        meshC->MakeBox(count, spacing, dim, false, true);
      } else if (builtin == mjMESH_BUILTIN_CYLINDER) {
        meshC->MakeCylinder(count, spacing, dim, false, true);
      } else {
        meshC->MakeEllipsoid(count, spacing, dim, false, true);
      }
      if ((dim == 3 ? mesh->usertet : mesh->userface)->empty()) {
        m->SetError(mjCError(0, "Failed to create box/cylinder/ellipsoid mesh"));
        return -1;
      }
      if (dim < 3 && mesh->inertia == mjMESH_INERTIA_LEGACY) {
        mesh->inertia = mjMESH_INERTIA_SHELL;
      }
      return 0;
    }

    case mjMESH_BUILTIN_SQUARE:
    case mjMESH_BUILTIN_DISC: {
      if (nparams != 6) {
        m->SetError(
            mjCError(0, "Square/disc mesh type requires 6 parameters (count[3], spacing[3])"));
        return -1;
      }
      int    count[3]   = {static_cast<int>(params[0]),
                           static_cast<int>(params[1]),
                           static_cast<int>(params[2])};
      double spacing[3] = {params[3], params[4], params[5]};
      for (int i = 0; i < 2; i++) {
        if (count[i] < 1) {
          m->SetError(mjCError(0, "Count must be positive"));
          return -1;
        }
      }
      if (count[2] != 1) {
        m->SetError(mjCError(0, "Square/disc count[2] must be 1"));
        return -1;
      }
      if (builtin == mjMESH_BUILTIN_SQUARE) {
        meshC->MakeGrid(count, spacing, 2, false);
      } else {
        meshC->MakeDisc(count, spacing, false);
      }
      if (mesh->userface->empty()) {
        m->SetError(mjCError(0, "Failed to create square/disc mesh"));
        return -1;
      }
      if (mesh->inertia == mjMESH_INERTIA_LEGACY) { mesh->inertia = mjMESH_INERTIA_SHELL; }
      return 0;
    }

    default:
      m->SetError(mjCError(0, "Unsupported mesh type"));
      return 1;
  }
}

// add pair to model
mjsPair* mjs_addPair(mjSpec* s, const mjsDefault* defspec) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCDef*   def    = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCPair*  pair   = modelC->AddPair(def);
  return &pair->spec;
}


// add pair exclusion to model
mjsExclude* mjs_addExclude(mjSpec* s) {
  mjCModel*    modelC   = static_cast<mjCModel*>(s->element);
  mjCBodyPair* bodypair = modelC->AddExclude();
  return &bodypair->spec;
}


// add equality to model
mjsEquality* mjs_addEquality(mjSpec* s, const mjsDefault* defspec) {
  mjCModel*    modelC   = static_cast<mjCModel*>(s->element);
  mjCDef*      def      = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCEquality* equality = modelC->AddEquality(def);
  return &equality->spec;
}


// add tendon to model
mjsTendon* mjs_addTendon(mjSpec* s, const mjsDefault* defspec) {
  mjCModel*  modelC = static_cast<mjCModel*>(s->element);
  mjCDef*    def    = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCTendon* tendon = modelC->AddTendon(def);
  return &tendon->spec;
}


// wrap site using tendon
mjsWrap* mjs_wrapSite(mjsTendon* tendonspec, const char* name) {
  mjCTendon* tendon = static_cast<mjCTendon*>(tendonspec->element);
  tendon->WrapSite(name);
  return &tendon->path.back()->spec;
}


// wrap geom using tendon
mjsWrap* mjs_wrapGeom(mjsTendon* tendonspec, const char* name, const char* sidesite) {
  mjCTendon* tendon = static_cast<mjCTendon*>(tendonspec->element);
  tendon->WrapGeom(name, sidesite);
  return &tendon->path.back()->spec;
}


// wrap joint using tendon
mjsWrap* mjs_wrapJoint(mjsTendon* tendonspec, const char* name, double coef) {
  mjCTendon* tendon = static_cast<mjCTendon*>(tendonspec->element);
  tendon->WrapJoint(name, coef);
  return &tendon->path.back()->spec;
}


// wrap pulley using tendon
mjsWrap* mjs_wrapPulley(mjsTendon* tendonspec, double divisor) {
  mjCTendon* tendon = static_cast<mjCTendon*>(tendonspec->element);
  tendon->WrapPulley(divisor);
  return &tendon->path.back()->spec;
}


// add actuator to model
mjsActuator* mjs_addActuator(mjSpec* s, const mjsDefault* defspec) {
  mjCModel*    modelC   = static_cast<mjCModel*>(s->element);
  mjCDef*      def      = defspec ? static_cast<mjCDef*>(defspec->element) : 0;
  mjCActuator* actuator = modelC->AddActuator(def);
  return &actuator->spec;
}


// add sensor to model
mjsSensor* mjs_addSensor(mjSpec* s) {
  mjCModel*  modelC = static_cast<mjCModel*>(s->element);
  mjCSensor* sensor = modelC->AddSensor();
  return &sensor->spec;
}


// add numeric to model
mjsNumeric* mjs_addNumeric(mjSpec* s) {
  mjCModel*   modelC  = static_cast<mjCModel*>(s->element);
  mjCNumeric* numeric = modelC->AddNumeric();
  return &numeric->spec;
}


// add text to model
mjsText* mjs_addText(mjSpec* s) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCText*  text   = modelC->AddText();
  return &text->spec;
}


// add tuple to model
mjsTuple* mjs_addTuple(mjSpec* s) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCTuple* tuple  = modelC->AddTuple();
  return &tuple->spec;
}


// add keyframe to model
mjsKey* mjs_addKey(mjSpec* s) {
  mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCKey*   key    = modelC->AddKey();
  return &key->spec;
}


// add plugin to model
mjsPlugin* mjs_addPlugin(mjSpec* s) {
  mjCModel*  modelC    = static_cast<mjCModel*>(s->element);
  mjCPlugin* plugin    = modelC->AddPlugin();
  plugin->spec.element = static_cast<mjsElement*>(plugin);
  return &plugin->spec;
}


// add default to model
mjsDefault* mjs_addDefault(mjSpec* s, const char* classname, const mjsDefault* parent) {
  mjCModel* modelC  = static_cast<mjCModel*>(s->element);
  mjCDef*   parentC = parent ? static_cast<mjCDef*>(parent->element)
                             : static_cast<mjCModel*>(s->element)->Default();
  mjCDef*   def     = modelC->AddDefault(classname, parentC);
  if (def) {
    return &def->spec;
  } else {
    return nullptr;
  }
}


// a shortcut owns the control model and its activation layout: start them from the general defaults
static void SetControl(mjsActuator* actuator, mjtActuator type) {
  mjuu_zerovec(actuator->gainprm, mjNGAIN);
  mjuu_zerovec(actuator->biasprm, mjNBIAS);
  mjuu_zerovec(actuator->dynprm, mjNDYN);
  actuator->gainprm[0] = 1;
  actuator->dynprm[0]  = 1;
  actuator->gaintype   = mjGAIN_FIXED;
  actuator->biastype   = mjBIAS_NONE;
  actuator->dyntype    = mjDYN_NONE;
  actuator->ctrlspec   = 0;
  actuator->actdim     = -1;
  actuator->actearly   = 0;
  actuator->actlimited = mjLIMITED_AUTO;
  actuator->type       = type;
}


// set actuator to motor
const char* mjs_setToMotor(mjsActuator* actuator) {
  SetControl(actuator, mjACTUATOR_MOTOR);
  return "";
}


// set to position actuator
const char* mjs_setToPosition(mjsActuator* actuator,
                              double       kp,
                              double       kv[1],
                              double       dampratio[1],
                              double       timeconst[1],
                              double       inheritrange) {
  SetControl(actuator, mjACTUATOR_POSITION);
  actuator->gainprm[0] = kp;
  actuator->biasprm[1] = -kp;

  // set biasprm[2]; negative: regular damping, positive: dampratio
  if (dampratio && kv) { return "kv and dampratio cannot both be defined"; }

  if (kv) {
    if (*kv < 0) return "kv cannot be negative";
    actuator->biasprm[2] = -(*kv);
  }
  if (dampratio) {
    if (*dampratio < 0) return "dampratio cannot be negative";
    actuator->biasprm[2] = *dampratio;
  }
  if (timeconst) {
    if (*timeconst < 0) return "timeconst cannot be negative";
    actuator->dynprm[0] = *timeconst;
    actuator->dyntype   = *timeconst == 0 ? mjDYN_NONE : mjDYN_FILTEREXACT;
  }
  actuator->inheritrange = inheritrange;

  if (inheritrange > 0) {
    if (actuator->ctrlrange[0] || actuator->ctrlrange[1]) {
      return "ctrlrange and inheritrange cannot both be defined";
    }
  }

  actuator->ctrlspec = mjINPUT_POS;
  actuator->gaintype = mjGAIN_FIXED;
  actuator->biastype = mjBIAS_AFFINE;
  return "";
}


// Set to integrated velocity actuator.
const char* mjs_setToIntVelocity(mjsActuator* actuator,
                                 double       kp,
                                 double       kv[1],
                                 double       dampratio[1],
                                 double       timeconst[1],
                                 double       inheritrange) {
  // inheritrange sets actrange, not ctrlrange: skip the position range check
  const char* err = mjs_setToPosition(actuator, kp, kv, dampratio, timeconst, 0);
  if (err[0]) return err;
  actuator->ctrlspec     = mjINPUT_VEL;
  actuator->dyntype      = mjDYN_INTEGRATOR;
  actuator->type         = mjACTUATOR_INTVELOCITY;
  actuator->inheritrange = inheritrange;

  if (inheritrange > 0) {
    if (actuator->actrange[0] || actuator->actrange[1]) {
      return "actrange and inheritrange cannot both be defined";
    }
  }
  return "";
}


// Set to orientation actuator.
const char* mjs_setToOrientation(
    mjsActuator* actuator, double kp, double kv[1], double dampratio[1], int ctrlspec) {
  if (kv && dampratio) { return "kv and dampratio cannot both be defined"; }
  SetControl(actuator, mjACTUATOR_ORIENTATION);
  actuator->gainprm[0] = kp;
  actuator->biasprm[1] = -kp;
  if (kv) {
    if (*kv < 0) return "kv cannot be negative";
    actuator->biasprm[2] = -(*kv);
  }
  if (dampratio) {
    if (*dampratio < 0) return "dampratio cannot be negative";
    actuator->biasprm[2] = *dampratio;
  }
  actuator->ctrlspec = ctrlspec;
  actuator->gaintype = mjGAIN_SO3;
  actuator->biastype = mjBIAS_SO3;
  actuator->dyntype  = mjDYN_NONE;
  return "";
}


// Set to PID actuator.
const char* mjs_setToPID(mjsActuator* actuator,
                         double       kp,
                         double       kv[1],
                         double       dampratio[1],
                         double       ki[1],
                         double       imax[1],
                         double       slewmax[1],
                         double       inheritrange,
                         int          ctrlspec) {
  if (kv && dampratio) { return "kv and dampratio cannot both be defined"; }
  SetControl(actuator, mjACTUATOR_PID);
  actuator->biasprm[1] = -kp;
  if (kv) {
    if (*kv < 0) return "kv cannot be negative";
    actuator->biasprm[2] = -(*kv);
  }
  if (dampratio) {
    if (*dampratio < 0) return "dampratio cannot be negative";
    actuator->biasprm[2] = *dampratio;
  }

  // controller states: ki in gainprm[0], imax in dynprm[0], slewmax in dynprm[1]
  double ki_value   = ki ? *ki : 0;
  double slew_value = slewmax ? *slewmax : 0;
  if (slew_value < 0) return "slewmax cannot be negative";
  actuator->gainprm[0] = ki_value;
  actuator->dynprm[1]  = slew_value;
  actuator->dyntype    = (ki_value || slew_value) ? mjDYN_PID : mjDYN_NONE;
  actuator->dynprm[0]  = (ki_value && imax) ? *imax : 0;

  actuator->inheritrange = inheritrange;
  if (inheritrange > 0) {
    if (actuator->ctrlrange[0] || actuator->ctrlrange[1]) {
      return "posrange and inheritrange cannot both be defined";
    }
  }

  actuator->ctrlspec = ctrlspec;
  actuator->gaintype = mjGAIN_PID;
  actuator->biastype = mjBIAS_AFFINE;
  return "";
}


// Set to velocity actuator.
const char* mjs_setToVelocity(mjsActuator* actuator, double kv) {
  SetControl(actuator, mjACTUATOR_VELOCITY);
  actuator->gainprm[0] = kv;
  actuator->biasprm[2] = -kv;
  actuator->ctrlspec   = mjINPUT_VEL;
  actuator->biastype   = mjBIAS_AFFINE;
  return "";
}


// Set to damper actuator.
const char* mjs_setToDamper(mjsActuator* actuator, double kv) {
  SetControl(actuator, mjACTUATOR_DAMPER);
  actuator->gainprm[0]  = 0;
  actuator->gainprm[2]  = -kv;
  actuator->ctrllimited = mjLIMITED_TRUE;
  actuator->gaintype    = mjGAIN_AFFINE;

  if (kv < 0) { return "damping coefficient cannot be negative"; }
  if (actuator->ctrlrange[0] < 0 || actuator->ctrlrange[1] < 0) {
    return "damper control range cannot be negative";
  }
  return "";
}


// Set to cylinder actuator.
const char* mjs_setToCylinder(
    mjsActuator* actuator, double timeconst, double bias, double area, double diameter) {
  SetControl(actuator, mjACTUATOR_CYLINDER);
  actuator->dynprm[0]  = timeconst;
  actuator->biasprm[0] = bias;
  actuator->gainprm[0] = area;
  if (diameter >= 0) { actuator->gainprm[0] = mjPI / 4 * diameter * diameter; }
  actuator->ctrlspec = mjINPUT_PRESSURE;
  actuator->dyntype  = mjDYN_FILTER;
  actuator->biastype = mjBIAS_AFFINE;
  return "";
}


// Set to muscle actuator.
const char* mjs_setToMuscle(mjsActuator* actuator,
                            double       timeconst[2],
                            double       tausmooth,
                            double       range[2],
                            double       force,
                            double       scale,
                            double       lmin,
                            double       lmax,
                            double       vmax,
                            double       fpmax,
                            double       fvmax) {
  SetControl(actuator, mjACTUATOR_MUSCLE);

  // muscle defaults
  actuator->dynprm[0]  = 0.01;  // tau act
  actuator->dynprm[1]  = 0.04;  // tau deact
  actuator->gainprm[0] = 0.75;  // range[0]
  actuator->gainprm[1] = 1.05;  // range[1]
  actuator->gainprm[2] = -1;    // force
  actuator->gainprm[3] = 200;   // scale
  actuator->gainprm[4] = 0.5;   // lmin
  actuator->gainprm[5] = 1.6;   // lmax
  actuator->gainprm[6] = 1.5;   // vmax
  actuator->gainprm[7] = 1.3;   // fpmax
  actuator->gainprm[8] = 1.2;   // fvmax

  if (tausmooth < 0) { return "muscle tausmooth cannot be negative"; }

  actuator->dynprm[2] = tausmooth;
  if (timeconst) {
    if (timeconst[0] >= 0) actuator->dynprm[0] = timeconst[0];
    if (timeconst[1] >= 0) actuator->dynprm[1] = timeconst[1];
  }
  if (range) {
    if (range[0] >= 0) actuator->gainprm[0] = range[0];
    if (range[1] >= 0) actuator->gainprm[1] = range[1];
  }
  if (force >= 0) actuator->gainprm[2] = force;
  if (scale >= 0) actuator->gainprm[3] = scale;
  if (lmin >= 0) actuator->gainprm[4] = lmin;
  if (lmax >= 0) actuator->gainprm[5] = lmax;
  if (vmax >= 0) actuator->gainprm[6] = vmax;
  if (fpmax >= 0) actuator->gainprm[7] = fpmax;
  if (fvmax >= 0) actuator->gainprm[8] = fvmax;

  // biasprm = gainprm
  for (int n = 0; n < 9; n++) { actuator->biasprm[n] = actuator->gainprm[n]; }

  actuator->dyntype  = mjDYN_MUSCLE;
  actuator->gaintype = mjGAIN_MUSCLE;
  actuator->biastype = mjBIAS_MUSCLE;
  return "";
}


// Set to adhesion actuator.
const char* mjs_setToAdhesion(mjsActuator* actuator, double gain) {
  SetControl(actuator, mjACTUATOR_ADHESION);
  actuator->gainprm[0]  = gain;
  actuator->ctrllimited = mjLIMITED_TRUE;

  if (gain < 0) return "adhesion gain cannot be negative";
  if (actuator->ctrlrange[0] < 0 || actuator->ctrlrange[1] < 0)
    return "adhesion control range cannot be negative";
  return "";
}


const char* mjs_setToDCMotor(mjsActuator* actuator,
                             double       motorconst[2],
                             double       resistance,
                             double       nominal[3],
                             double       saturation[3],
                             double       inductance[2],
                             double       cogging[3],
                             double       controller[6],
                             double       thermal[6],
                             double       lugre[5],
                             int          ctrlspec) {
  double R      = resistance;                      // electrical resistance
  double Kt     = motorconst ? motorconst[0] : 0;  // torque constant
  double Ke     = motorconst ? motorconst[1] : 0;  // back-EMF constant
  double vn     = nominal ? nominal[0] : 0;        // nominal voltage
  double tau0   = nominal ? nominal[1] : 0;        // stall torque
  double omega0 = nominal ? nominal[2] : 0;        // no-load speed

  // derive Ke from nominal: omega0 = vn*Ke / (Ke^2 + R*B)
  if (vn > 0 && Ke <= 0 && omega0 > 0) {
    // viscous damping (linear)
    double B = actuator->damping[0];

    if (B > 0 && R > 0) {
      // R known: solve quadratic Ke^2*omega0 - Ke*vn + R*B*omega0 = 0
      double disc = vn * vn - 4 * R * B * omega0 * omega0;
      Ke          = disc > 0 ? (vn + sqrt(disc)) / (2 * omega0) : vn / omega0;
    } else if (B > 0 && tau0 > 0) {
      // R from nominal (tau0 = Ke*vn/R, so R = Ke*vn/tau0)
      // substituting into omega0 = vn*Ke/(Ke^2 + R*B):
      //   omega0 = vn/(Ke + vn*B/tau0)  =>  Ke = vn/omega0 - vn*B/tau0
      double Ke_exact = vn / omega0 - vn * B / tau0;
      Ke              = Ke_exact > 0 ? Ke_exact : vn / omega0;
    } else {
      // B = 0 or insufficient data for B-correction: omega0 = vn*Ke/Ke^2 = vn/Ke
      Ke = vn / omega0;
    }
  }

  // resolve effective motor constant K from [Kt, Ke]
  double K = (Kt > 0 && Ke > 0) ? sqrt(Kt * Ke) : (Kt > 0) ? Kt : Ke;

  // derive R from nominal: tau0 = K*vn/R
  if (R == 0 && vn > 0 && tau0 > 0 && K > 0) { R = K * vn / tau0; }

  if (K <= 0) return "DC motor: motor constant K must be positive";
  if (R <= 0) return "DC motor: resistance R must be positive";

  // set types
  SetControl(actuator, mjACTUATOR_DCMOTOR);
  actuator->dyntype  = mjDYN_DCMOTOR;
  actuator->gaintype = mjGAIN_DCMOTOR;
  actuator->biastype = mjBIAS_DCMOTOR;

  // gainprm: [R, K, alpha, T0]
  actuator->gainprm[0] = R;
  actuator->gainprm[1] = K;

  // controller parameters: gainprm[4:6] for kp, ki, kd
  actuator->gainprm[4] = controller ? controller[0] : 0;  // kp
  actuator->gainprm[5] = controller ? controller[1] : 0;  // ki
  actuator->gainprm[6] = controller ? controller[2] : 0;  // kd

  // controller parameters: dynprm[7,8] for slewmax, Imax
  actuator->dynprm[7] = controller ? controller[3] : 0;  // slewmax
  actuator->dynprm[8] = controller ? controller[4] : 0;  // Imax

  // controller parameters: gainprm[7] for v_max
  if (controller && controller[5] > 0) {
    actuator->gainprm[7] = controller[5];  // v_max
  }

  // saturation -> forcerange
  if (saturation && (saturation[0] > 0 || saturation[1] > 0)) {
    double tau_max = saturation[0];
    if (tau_max == 0 && saturation[1] > 0) {
      tau_max = K * saturation[1];  // tau_max = K * i_max
    }
    actuator->forcerange[0] = -tau_max;
    actuator->forcerange[1] = tau_max;
    actuator->forcelimited  = mjLIMITED_TRUE;
  }

  // saturation: [tau_max, i_max, (di/dt)_max]
  if (saturation && saturation[2] > 0) {
    actuator->dynprm[1] = saturation[2];  // (di/dt)_max
  }

  // cogging: [amplitude, periodicity, phase] -> biasprm[0:3]
  actuator->biasprm[0] = cogging ? cogging[0] : 0;  // amplitude
  actuator->biasprm[1] = cogging ? cogging[1] : 0;  // periodicity
  actuator->biasprm[2] = cogging ? cogging[2] : 0;  // phase

  // count activation variables: slot order is slew, integral, temperature, bristle, current
  int actdim = 0;

  // inductance: [L, te]
  if (inductance && inductance[0] < 0) return "DC motor: inductance must be non-negative";
  if (inductance && inductance[1] < 0)
    return "DC motor: electrical time constant must be non-negative";
  double te =
      (inductance && inductance[0] > 0) ? inductance[0] / R : (inductance ? inductance[1] : 0);
  actuator->dynprm[0] = te;
  if (te > 0) { actdim++; }

  // controller states: slew rate limiting
  if (controller && controller[3] > 0) {  // slewmax
    actdim++;
  }

  // controller states: integral
  if (controller && controller[1] > 0) {  // ki
    actdim++;
  }

  // thermal -> temperature activation
  if (thermal && (thermal[0] > 0 || thermal[1] > 0 || thermal[2] > 0)) {
    double RT    = thermal[0];  // thermal resistance
    double C     = thermal[1];  // thermal capacitance
    double tth   = thermal[2];  // thermal time constant
    double alpha = thermal[3];  // temperature coefficient
    double T0    = thermal[4];  // reference temperature
    double Ta    = thermal[5];  // ambient temperature

    if (tth > 0 && RT > 0 && C == 0) {
      C = tth / RT;
    } else if (tth > 0 && C > 0 && RT == 0) {
      RT = tth / C;
    } else if (tth == 0 && RT > 0 && C > 0) {
      tth = RT * C;
    }

    if (RT <= 0) return "DC motor: thermal resistance must be positive";
    if (C <= 0) return "DC motor: thermal capacitance must be positive";

    actuator->dynprm[2]  = RT;
    actuator->dynprm[3]  = C;
    actuator->dynprm[4]  = Ta;
    actuator->gainprm[2] = alpha;
    actuator->gainprm[3] = T0;
    actdim++;
  }

  // lugre: {stiffness, damping, coulomb, static, stribeck}
  if (lugre && lugre[0] > 0) {
    actuator->dynprm[5]  = lugre[0];  // stiffness -> sigma0
    actuator->dynprm[6]  = lugre[1];  // damping   -> sigma1
    actuator->biasprm[3] = lugre[2];  // coulomb   -> tau_c
    actuator->biasprm[4] = lugre[3];  // static    -> tau_s
    actuator->biasprm[5] = lugre[4];  // stribeck  -> omega_s
    actdim++;
  }

  // set input mode and activation dimension
  actuator->gainprm[8] = 0;  // reserved (was input_mode)
  actuator->ctrlspec   = ctrlspec;
  actuator->actdim     = actdim;

  // enforce actlimited = 0; homogeneous bounds are invalid across DC motor states
  actuator->actlimited = mjLIMITED_FALSE;

  // DC motor always uses actearly
  actuator->actearly = 1;

  return "";
}


// get spec from body
mjSpec* mjs_getSpec(const mjsElement* element) {
  return &(static_cast<const mjCBase*>(element)->model->spec);
}


// get spec that originally defined an element
// contrary to mjs_getSpec, this does not change after attachment
mjSpec* mjs_getOriginSpec(const mjsElement* element) {
  const mjCModel*    model    = static_cast<const mjCBase*>(element)->model;
  const mjsCompiler* compiler = static_cast<const mjCBase*>(element)->compiler;
  return model->FindSpec(compiler);
}


mjsCompiler* mjs_getCompiler(const mjsElement* element) {
  return static_cast<const mjCBase*>(element)->compiler;
}


// find spec (model asset) by name
mjSpec* mjs_findSpec(const mjSpec* s, const char* name) {
  const mjCModel* model = static_cast<mjCModel*>(s->element);
  return model->FindSpec(name);
}


// get default
mjsDefault* mjs_getDefault(const mjsElement* element) {
  const mjCModel* model     = static_cast<const mjCBase*>(element)->model;
  std::string     classname = static_cast<const mjCBase*>(element)->classname;
  auto            it        = model->def_map.find(classname);
  return (it != model->def_map.end()) ? &it->second->spec : nullptr;
}


// Find default with given name in model.
mjsDefault* mjs_findDefault(const mjSpec* s, const char* classname) {
  const mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCDef*         cdef   = modelC->FindDefault(classname);
  return cdef ? &cdef->spec : nullptr;
}


// get default[0] from model
mjsDefault* mjs_getSpecDefault(const mjSpec* s) {
  const mjCModel* modelC = static_cast<mjCModel*>(s->element);
  mjCDef*         def    = modelC->Default();
  return def ? &def->spec : nullptr;
}


// find body in model by name
mjsBody* mjs_findBody(const mjSpec* s, const char* name) {
  mjsElement* body = mjs_findElement(s, mjOBJ_BODY, name);
  return body ? &(static_cast<mjCBody*>(body)->spec) : nullptr;
}


// find element in spec by name
mjsElement* mjs_findElement(const mjSpec* s, mjtObj type, const char* name) {
  mjCModel* model = static_cast<mjCModel*>(s->element);

  // fast lookup in the maps from names to positions; they describe the lists as they were last
  // processed, so an element added, removed or moved since then is searched for instead
  if (model->IsCompiled() && type != mjOBJ_FRAME) {
    try {
      mjCBase* found = model->FindObject(type, std::string(name));
      if (found && found->name == name) { return found; }
    } catch (mjCError&) {
      // a position which is out of range
    }
  }

  switch (type) {
    case mjOBJ_BODY:
    case mjOBJ_SITE:
    case mjOBJ_GEOM:
    case mjOBJ_JOINT:
    case mjOBJ_CAMERA:
    case mjOBJ_LIGHT:
    case mjOBJ_FRAME:
      return model->FindTree(model->GetWorld(), type, std::string(name));  // recursive search
    default:
      return model->SearchObject(type, name);
  }
}


// find child of a body by name
mjsBody* mjs_findChild(const mjsBody* bodyspec, const char* name) {
  const mjCBody* body  = static_cast<mjCBody*>(bodyspec->element);
  mjCBase*       child = body->FindObject(mjOBJ_BODY, std::string(name));
  return child ? &(static_cast<mjCBody*>(child)->spec) : nullptr;
}


// get parent body
mjsBody* mjs_getParent(const mjsElement* element) {
  switch (element->elemtype) {
    case mjOBJ_BODY:
    case mjOBJ_FRAME:
    case mjOBJ_JOINT:
    case mjOBJ_GEOM:
    case mjOBJ_SITE:
    case mjOBJ_CAMERA:
    case mjOBJ_LIGHT: {
      mjCBody* parent = static_cast<mjCBody*>(static_cast<const mjCBase*>(element)->GetParent());
      return parent ? &parent->spec : nullptr;
    }
    default:
      return nullptr;
  }
}


// get parent frame
mjsFrame* mjs_getFrame(const mjsElement* element) {
  const mjCBase* base = static_cast<const mjCBase*>(element);
  switch (element->elemtype) {
    case mjOBJ_BODY:
    case mjOBJ_FRAME:
    case mjOBJ_JOINT:
    case mjOBJ_GEOM:
    case mjOBJ_SITE:
    case mjOBJ_CAMERA:
    case mjOBJ_LIGHT:
      return base->frame ? &(base->frame->spec) : nullptr;
    default:
      return nullptr;
  }
}


// find frame by name
mjsFrame* mjs_findFrame(const mjSpec* s, const char* name) {
  mjsElement* frame = mjs_findElement(s, mjOBJ_FRAME, name);
  return frame ? &(static_cast<mjCFrame*>(frame)->spec) : nullptr;
}


// set frame
int mjs_setFrame(mjsElement* dest, mjsFrame* frame) {
  if (!frame || !dest) { return -1; }
  mjCFrame* frameC = static_cast<mjCFrame*>(frame->element);
  mjCBase*  baseC  = static_cast<mjCBase*>(dest);
  try {
    baseC->SetFrame(frameC);
    return 0;
  } catch (mjCError& e) {
    baseC->model->SetError(e);
    return -1;
  }
}


// Resolve alternative orientations.
const char* mjs_resolveOrientation(double                quat[4],
                                   mjtByte               degree,
                                   const char*           sequence,
                                   const mjsOrientation* orientation) {
  return ResolveOrientation(quat, degree, sequence, *orientation);
}


// Transform body into a frame.
// fuse the static bodies of the spec with their parents
int mjs_fuseStatic(mjSpec* s, const mjVFS* vfs) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  if (model->IsAttached()) {
    model->SetError(mjCError(nullptr, "Cannot fuse the static bodies of an attached mjSpec."));
    return -1;
  }
  try {
    return model->FuseStatic(vfs);
  } catch (mjCError& e) {
    model->SetError(e);
    return -1;
  }
}


// discard the visual elements of the spec
int mjs_discardVisual(mjSpec* s, const mjVFS* vfs) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  if (model->IsAttached()) {
    model->SetError(mjCError(nullptr, "Cannot discard the visual elements of an attached mjSpec."));
    return -1;
  }
  try {
    return model->DiscardVisual(vfs);
  } catch (mjCError& e) {
    model->SetError(e);
    return -1;
  }
}


// merge the duplicate textures and meshes of the spec
int mjs_dedupAssets(mjSpec* s) {
  mjCModel* model = static_cast<mjCModel*>(s->element);
  if (model->IsAttached()) {
    model->SetError(mjCError(nullptr, "Cannot merge the assets of an attached mjSpec."));
    return -1;
  }
  try {
    return model->DedupAssets();
  } catch (mjCError& e) {
    model->SetError(e);
    return -1;
  }
}


// make the inertial which compilation infers for a body part of the spec
int mjs_adoptInertial(mjsBody* bodyspec, const mjVFS* vfs) {
  mjCBody*  body  = static_cast<mjCBody*>(bodyspec->element);
  mjCModel* model = body->model;
  try {
    if (!body->GetParent()) { throw mjCError(body, "the world body has no inertial"); }
    int inertiafromgeom = mjs_getCompiler(bodyspec->element)->inertiafromgeom;
    if (inertiafromgeom == mjINERTIAFROMGEOM_TRUE) {
      throw mjCError(body,
                     "an inertial cannot be adopted when inertiafromgeom is 'true', which infers "
                     "the inertia of every body from its geoms");
    }

    // nothing is inferred for this body
    if (inertiafromgeom == mjINERTIAFROMGEOM_FALSE || mjuu_defined(bodyspec->ipos[0])) { return 0; }

    if (!model->Resolve(vfs)) { return -1; }
    body->AdoptInertial();
    return 0;
  } catch (mjCError& e) {
    model->SetError(e);
    return -1;
  }
}


mjsFrame* mjs_bodyToFrame(mjsBody** body) {
  mjCBody* bodyC = static_cast<mjCBody*>((*body)->element);
  try {
    mjCFrame* frameC  = bodyC->ToFrame();
    *bodyC->model    -= (*body)->element;
    *body             = nullptr;
    return &frameC->spec;
  } catch (mjCError& e) {
    bodyC->model->SetError(e);
    return nullptr;
  }
}

void mjs_setUserValue(mjsElement* element, const char* key, const void* data) {
  mjs_setUserValueWithCleanup(element, key, data, nullptr);
}

// set user payload
void mjs_setUserValueWithCleanup(mjsElement* element,
                                 const char* key,
                                 const void* data,
                                 void (*cleanup)(const void*)) {
  mjCBase* baseC = static_cast<mjCBase*>(element);
  baseC->SetUserValue(key, data, cleanup);
}

// return user payload or NULL if none found
const void* mjs_getUserValue(mjsElement* element, const char* key) {
  mjCBase* baseC = static_cast<mjCBase*>(element);
  return baseC->GetUserValue(key);
}


// delete user payload
void mjs_deleteUserValue(mjsElement* element, const char* key) {
  mjCBase* baseC = static_cast<mjCBase*>(element);
  baseC->DeleteUserValue(key);
}


// return sensor dimension
int mjs_sensorDim(const mjsSensor* sensor) {
  switch (sensor->type) {
    case mjSENS_TOUCH:
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
      return sensor->intprm[2] * mju_condataSize(sensor->intprm[0]);

    case mjSENS_TACTILE:
      return 3 * static_cast<const mjCMesh*>(static_cast<mjCSensor*>(sensor->element)->get_obj())
                     ->nvert();

    case mjSENS_RANGEFINDER: {
      int size     = mju_raydataSize(sensor->intprm[0]);
      int num_rays = 1;
      if (sensor->objtype == mjOBJ_CAMERA) {
        const mjCCamera* camera =
            static_cast<const mjCCamera*>(static_cast<mjCSensor*>(sensor->element)->get_obj());
        num_rays = camera->spec.resolution[0] * camera->spec.resolution[1];
      }
      return size * num_rays;
    }

    case mjSENS_USER:
      return sensor->dim;

    case mjSENS_PLUGIN:
      return 0;  // to be filled in by plugin
  }
  return -1;
}


// get id
int mjs_getId(const mjsElement* element) {
  if (!element) { return -1; }
  return static_cast<const mjCBase*>(element)->id;
}


// set default
void mjs_setDefault(mjsElement* element, const mjsDefault* defspec) {
  mjCBase* baseC   = static_cast<mjCBase*>(element);
  baseC->classname = static_cast<mjCDef*>(defspec->element)->name;
}


// return first child of selected type
mjsElement* mjs_firstChild(const mjsBody* body, mjtObj type, int recurse) {
  const mjCBody* bodyC = static_cast<const mjCBody*>(body->element);
  try {
    return bodyC->NextChild(NULL, type, recurse);
  } catch (mjCError& e) {
    bodyC->model->SetError(e);
    return nullptr;
  }
}


// return body's next child; return NULL if child is last
mjsElement* mjs_nextChild(const mjsBody* body, const mjsElement* child, int recurse) {
  const mjCBody* bodyC = static_cast<const mjCBody*>(body->element);
  try {
    return bodyC->NextChild(child, child->elemtype, recurse);
  } catch (mjCError& e) {
    bodyC->model->SetError(e);
    return nullptr;
  }
}


// return spec's first element of selected type
mjsElement* mjs_firstElement(const mjSpec* s, mjtObj type) {
  const mjCModel* modelC = static_cast<mjCModel*>(s->element);
  return modelC->NextObject(NULL, type);
}


// return spec's next element; return NULL if element is last
mjsElement* mjs_nextElement(const mjSpec* s, const mjsElement* element) {
  const mjCModel* modelC = static_cast<mjCModel*>(s->element);
  return modelC->NextObject(element);
}


mjsElement* mjs_getWrapTarget(const mjsWrap* wrap) {
  const mjCWrap* cwrap = static_cast<const mjCWrap*>(wrap->element);
  mjtObj         type  = mjOBJ_UNKNOWN;
  switch (cwrap->Type()) {
    case mjWRAP_SPHERE:
    case mjWRAP_CYLINDER:
      type = mjOBJ_GEOM;
      break;
    case mjWRAP_SITE:
      type = mjOBJ_SITE;
      break;
    case mjWRAP_JOINT:
      type = mjOBJ_JOINT;
      break;
    case mjWRAP_PULLEY:
      // Pulleys have no target.
      return nullptr;
    default:
      return nullptr;
  }
  const mjSpec* spec = mjs_getSpec(wrap->element);
  return mjs_findElement(spec, type, cwrap->name.c_str());
}


mjsSite* mjs_getWrapSideSite(const mjsWrap* wrap) {
  const mjCWrap* cwrap = static_cast<const mjCWrap*>(wrap->element);
  // only sphere and cylinder (geoms) have side sites
  if ((cwrap->Type() != mjWRAP_SPHERE && cwrap->Type() != mjWRAP_CYLINDER) ||
      cwrap->sidesite.empty()) {
    return nullptr;
  }

  const mjSpec* spec = mjs_getSpec(wrap->element);
  mjsElement*   site = mjs_findElement(spec, mjOBJ_SITE, cwrap->sidesite.c_str());
  if (site == nullptr) {
    mju_warning("Could not find side site %s for wrap %s in spec",
                cwrap->sidesite.c_str(),
                cwrap->name.c_str());
    return nullptr;
  }
  return mjs_asSite(site);
}


double mjs_getWrapDivisor(const mjsWrap* wrap) {
  const mjCWrap* cwrap = static_cast<const mjCWrap*>(wrap->element);
  if (cwrap->Type() != mjWRAP_PULLEY) {
    mju_warning("Querying divisor attribute of non-pulley wrap: %s", cwrap->name.c_str());
    return 1.0;
  }
  return cwrap->prm;
}


double mjs_getWrapCoef(const mjsWrap* wrap) {
  const mjCWrap* cwrap = static_cast<const mjCWrap*>(wrap->element);
  if (cwrap->Type() != mjWRAP_JOINT) {
    mju_warning("Querying coef attribute of non-joint wrap: %s", cwrap->name.c_str());
    return 1.0;
  }
  return cwrap->prm;
}


// return body given mjsElement
mjsBody* mjs_asBody(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_BODY) {
    return &(static_cast<mjCBody*>(element)->spec);
  }
  return nullptr;
}


// return geom given mjsElement
mjsGeom* mjs_asGeom(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_GEOM) {
    return &(static_cast<mjCGeom*>(element)->spec);
  }
  return nullptr;
}


// return joint given mjsElement
mjsJoint* mjs_asJoint(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_JOINT) {
    return &(static_cast<mjCJoint*>(element)->spec);
  }
  return nullptr;
}


// Return site given mjsElement
mjsSite* mjs_asSite(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_SITE) {
    return &(static_cast<mjCSite*>(element)->spec);
  }
  return nullptr;
}


// return camera given mjsElement
mjsCamera* mjs_asCamera(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_CAMERA) {
    return &(static_cast<mjCCamera*>(element)->spec);
  }
  return nullptr;
}


// return light given mjsElement
mjsLight* mjs_asLight(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_LIGHT) {
    return &(static_cast<mjCLight*>(element)->spec);
  }
  return nullptr;
}


// return frame given mjsElement
mjsFrame* mjs_asFrame(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_FRAME) {
    return &(static_cast<mjCFrame*>(element)->spec);
  }
  return nullptr;
}


// return actuator given mjsElement
mjsActuator* mjs_asActuator(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_ACTUATOR) {
    return &(static_cast<mjCActuator*>(element)->spec);
  }
  return nullptr;
}


// return sensor given mjsElement
mjsSensor* mjs_asSensor(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_SENSOR) {
    return &(static_cast<mjCSensor*>(element)->spec);
  }
  return nullptr;
}


// return flex given mjsElement
mjsFlex* mjs_asFlex(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_FLEX) {
    return &(static_cast<mjCFlex*>(element)->spec);
  }
  return nullptr;
}


// return pair given mjsElement
mjsPair* mjs_asPair(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_PAIR) {
    return &(static_cast<mjCPair*>(element)->spec);
  }
  return nullptr;
}


// return equality given mjsElement
mjsEquality* mjs_asEquality(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_EQUALITY) {
    return &(static_cast<mjCEquality*>(element)->spec);
  }
  return nullptr;
}


// return exclude given mjsElement
mjsExclude* mjs_asExclude(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_EXCLUDE) {
    return &(static_cast<mjCBodyPair*>(element)->spec);
  }
  return nullptr;
}


// return tendon given mjsElement
mjsTendon* mjs_asTendon(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_TENDON) {
    return &(static_cast<mjCTendon*>(element)->spec);
  }
  return nullptr;
}


// return numeric given mjsElement
mjsNumeric* mjs_asNumeric(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_NUMERIC) {
    return &(static_cast<mjCNumeric*>(element)->spec);
  }
  return nullptr;
}


// return text given mjsElement
mjsText* mjs_asText(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_TEXT) {
    return &(static_cast<mjCText*>(element)->spec);
  }
  return nullptr;
}


// return tuple given mjsElement
mjsTuple* mjs_asTuple(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_TUPLE) {
    return &(static_cast<mjCTuple*>(element)->spec);
  }
  return nullptr;
}


// return key given mjsElement
mjsKey* mjs_asKey(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_KEY) { return &(static_cast<mjCKey*>(element)->spec); }
  return nullptr;
}


// return mesh given mjsElement
mjsMesh* mjs_asMesh(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_MESH) {
    return &(static_cast<mjCMesh*>(element)->spec);
  }
  return nullptr;
}


// return hfield given mjsElement
mjsHField* mjs_asHField(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_HFIELD) {
    return &(static_cast<mjCHField*>(element)->spec);
  }
  return nullptr;
}


// return skin given mjsElement
mjsSkin* mjs_asSkin(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_SKIN) {
    return &(static_cast<mjCSkin*>(element)->spec);
  }
  return nullptr;
}


// return texture given mjsElement
mjsTexture* mjs_asTexture(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_TEXTURE) {
    return &(static_cast<mjCTexture*>(element)->spec);
  }
  return nullptr;
}


// return material given mjsElement
mjsMaterial* mjs_asMaterial(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_MATERIAL) {
    return &(static_cast<mjCMaterial*>(element)->spec);
  }
  return nullptr;
}


// return plugin given mjsElement
mjsPlugin* mjs_asPlugin(mjsElement* element) {
  if (element && element->elemtype == mjOBJ_PLUGIN) {
    return &(static_cast<mjCPlugin*>(element)->spec);
  }
  return nullptr;
}


// set element name
int mjs_setName(mjsElement* element, const char* name) {
  if (element->elemtype == mjOBJ_DEFAULT) {
    mjCDef* def = static_cast<mjCDef*>(element);
    def->name   = std::string(name);
    return 0;
  }
  mjCBase* baseC = static_cast<mjCBase*>(element);
  try {
    baseC->model->CheckNameChange(element->elemtype, baseC->name, name);
  } catch (mjCError& e) {
    baseC->model->SetError(e);
    return -1;
  }
  baseC->name = std::string(name);
  return 0;
}


// copy buffer to destination buffer
void mjs_setBuffer(mjByteVec* dest, const void* array, int size) {
  const std::byte* buffer = static_cast<const std::byte*>(array);
  dest->clear();
  dest->reserve(size);
  std::copy_n(buffer, size, std::back_inserter(*dest));
}


// set string
void mjs_setString(mjString* dest, const char* text) {
  std::string* str = static_cast<std::string*>(dest);
  *str             = std::string(text);
}


// Set specific entry in destination string vector.
mjtBool mjs_setInStringVec(mjStringVec* dest, int i, const char* text) {
  if (dest->size() <= i) {
    mju_error("Requested index in mjs_setInStringVec is out of bounds");
    return false;
  }
  dest->at(i) = std::string(text);
  return true;
}


// split text and copy into string array
void mjs_setStringVec(mjStringVec* dest, const char* text) {
  std::vector<std::string>* v = static_cast<std::vector<std::string>*>(dest);
  *v                          = StringToVector<std::string>(text);
}


// add text entry to destination string vector
void mjs_appendString(mjStringVec* dest, const char* text) {
  dest->push_back(std::string(text));
}


// copy int array to vector
void mjs_setInt(mjIntVec* dest, const int* array, int size) {
  dest->assign(size, 0.0);
  for (int i = 0; i < size; ++i) { (*dest)[i] = array[i]; }
}


// append int array to vector of arrays
void mjs_appendIntVec(mjIntVecVec* dest, const int* array, int size) {
  dest->push_back(std::vector<int>(array, array + size));
}


// copy float array to vector
void mjs_setFloat(mjFloatVec* dest, const float* array, int size) {
  dest->assign(size, 0.0);
  for (int i = 0; i < size; ++i) { (*dest)[i] = array[i]; }
}


// append float array to vector of arrays
void mjs_appendFloatVec(mjFloatVecVec* dest, const float* array, int size) {
  dest->push_back(std::vector<float>(array, array + size));
}


// copy double array to vector
void mjs_setDouble(mjDoubleVec* dest, const double* array, int size) {
  dest->assign(size, 0.0);
  for (int i = 0; i < size; ++i) { (*dest)[i] = array[i]; }
}


// get name
mjString* mjs_getName(mjsElement* element) {
  if (element->elemtype == mjOBJ_DEFAULT) { return &(static_cast<mjCDef*>(element)->name); }
  return &(static_cast<mjCBase*>(element)->name);
}


// get string
const char* mjs_getString(const mjString* source) {
  return source->c_str();
}


// get double array
const double* mjs_getDouble(const mjDoubleVec* source, int* size) {
  if (size) { *size = source->size(); }
  return source->data();
}

int mjs_getWrapNum(const mjsTendon* tendonspec) {
  mjCTendon* tendon = static_cast<mjCTendon*>(tendonspec->element);
  return tendon->NumWraps();
}

mjsWrap* mjs_getWrap(const mjsTendon* tendonspec, int i) {
  mjCTendon* tendon = static_cast<mjCTendon*>(tendonspec->element);
  if (i < 0 || i >= tendon->NumWraps()) {
    mju_error("Wrap index out of range (0, %d)", tendon->NumWraps());
  }
  return &const_cast<mjCWrap*>(tendon->GetWrap(i))->spec;
}

// set plugin attributes
void mjs_setPluginAttributes(mjsPlugin* plugin, void* attributes) {
  mjCPlugin* pluginC = static_cast<mjCPlugin*>(plugin->element);
  std::map<std::string, std::string, std::less<> >* config_attribs =
      reinterpret_cast<std::map<std::string, std::string, std::less<> >*>(attributes);
  pluginC->config_attribs = std::move(*config_attribs);
}


// get plugin attributes
const void* mjs_getPluginAttributes(const mjsPlugin* plugin) {
  mjCPlugin* pluginC = static_cast<mjCPlugin*>(plugin->element);
  return &pluginC->config_attribs;
}


// -------------------------- GLOBAL ASSET CACHE -------------------------------

// get the capacity of the asset cache in bytes
size_t mj_getCacheCapacity(const mjCache* cache) {
  if (cache) {
    const mjCCache* ccache = reinterpret_cast<const mjCCache*>(cache->impl_);
    if (ccache) { return ccache->Capacity(); }
  }
  return 0;
}


// set the capacity of the asset cache in bytes (0 to disable)
size_t mj_setCacheCapacity(mjCache* cache, size_t size) {
  if (cache) {
    mjCCache* ccache = reinterpret_cast<mjCCache*>(cache->impl_);
    if (ccache) {
      ccache->SetCapacity(size);
      return ccache->Capacity();
    }
  }
  return 0;
}


// get the current size of the asset cache in bytes
size_t mj_getCacheSize(const mjCache* cache) {
  if (cache) {
    const mjCCache* ccache = reinterpret_cast<const mjCCache*>(cache->impl_);
    if (ccache) { return ccache->Size(); }
  }
  return 0;
}


// clear the asset cache
void mj_clearCache(mjCache* cache) {
  if (cache) {
    mjCCache* ccache = reinterpret_cast<mjCCache*>(cache->impl_);
    if (ccache) { ccache->Reset(); }
  }
}

// get the internal asset cache used by the compiler
mjCache* mj_getCache() {
  static mjCache cache_cwrapper = []() {
    mjCache c = {0};
    // mjCCache is not trivially destructible and so the global cache needs to
    // allocated on the heap
    if constexpr (kGlobalCacheSize != 0) {
      static mjCCache* cache = new (std::nothrow) mjCCache(kGlobalCacheSize);
      c.impl_                = cache->Capacity() > 0 ? cache : nullptr;
    }
    return c;
  }();
  return &cache_cwrapper;
}


// return 1 if a field was authored, 0 otherwise
int mjs_isAuthored(const void* elem_ptr, const void* field_ptr) {
  if (!elem_ptr || !field_ptr) return 0;
  const mjsElement* el = *reinterpret_cast<const mjsElement* const*>(elem_ptr);
  if (!el) return 0;

  // model-level sub-structs (compiler, option, visual)
  if (el->elemtype == mjOBJ_MODEL) {
    const mjCModel* cel = static_cast<const mjCModel*>(el);

    int idx = 0;

#define CHECK_FIELD(FIELD_PATH, AUTHORED_MASK)                               \
  if (field_ptr == &FIELD_PATH) return (AUTHORED_MASK & (1ULL << idx)) != 0; \
  idx++;
#define CHECK_FIELD_VEC(FIELD_PATH, AUTHORED_MASK)         \
  if (field_ptr == FIELD_PATH || field_ptr == &FIELD_PATH) \
    return (AUTHORED_MASK & (1ULL << idx)) != 0;           \
  idx++;

#define X(type, name, dim) CHECK_FIELD(cel->spec.compiler.name, cel->spec.compiler.authored)
#define XVEC(type, name, dim) CHECK_FIELD_VEC(cel->spec.compiler.name, cel->spec.compiler.authored)
    idx = 0;
    MJSCOMPILER_FIELDS
#undef X
#undef XVEC

#define X(type, name, dim) CHECK_FIELD(cel->spec.option.name, cel->spec.authored.option)
#define XVEC(type, name, dim) CHECK_FIELD_VEC(cel->spec.option.name, cel->spec.authored.option)
    idx = 0;
    MJOPTION_FIELDS
#undef X
#undef XVEC

#define X(type, name, dim) \
  CHECK_FIELD(cel->spec.visual.global.name, cel->spec.authored.visual_global)
    idx = 0;
    MJVISUAL_GLOBAL_FIELDS
#undef X

#define X(type, name, dim) \
  CHECK_FIELD(cel->spec.visual.quality.name, cel->spec.authored.visual_quality)
    idx = 0;
    MJVISUAL_QUALITY_FIELDS
#undef X

#define X(type, name, dim) \
  CHECK_FIELD(cel->spec.visual.headlight.name, cel->spec.authored.visual_headlight)
#define XVEC(type, name, dim) \
  CHECK_FIELD_VEC(cel->spec.visual.headlight.name, cel->spec.authored.visual_headlight)
    idx = 0;
    MJVISUAL_HEADLIGHT_FIELDS
#undef X
#undef XVEC

#define X(type, name, dim) CHECK_FIELD(cel->spec.visual.map.name, cel->spec.authored.visual_map)
    idx = 0;
    MJVISUAL_MAP_FIELDS
#undef X

#define X(type, name, dim) CHECK_FIELD(cel->spec.visual.scale.name, cel->spec.authored.visual_scale)
    idx = 0;
    MJVISUAL_SCALE_FIELDS
#undef X

#define XVEC(type, name, dim) \
  CHECK_FIELD_VEC(cel->spec.visual.rgba.name, cel->spec.authored.visual_rgba)
    idx = 0;
    MJVISUAL_RGBA_FIELDS
#undef XVEC

#undef CHECK_FIELD
#undef CHECK_FIELD_VEC
  }

  return 0;
}


// record explicit authoring of an element's field
void mjs_setAuthored(const void* elem_ptr, const void* field_ptr, int authored) {
  if (!elem_ptr || !field_ptr) return;
  mjsElement* el = const_cast<mjsElement*>(*reinterpret_cast<const mjsElement* const*>(elem_ptr));
  if (!el) return;

#define SET_FIELD(FIELD_PATH, AUTHORED_MASK) \
  if (field_ptr == &FIELD_PATH) {            \
    if (authored)                            \
      AUTHORED_MASK |= (1ULL << idx);        \
    else                                     \
      AUTHORED_MASK &= ~(1ULL << idx);       \
    return;                                  \
  }                                          \
  idx++;

#define SET_FIELD_VEC(FIELD_PATH, AUTHORED_MASK)             \
  if (field_ptr == FIELD_PATH || field_ptr == &FIELD_PATH) { \
    if (authored)                                            \
      AUTHORED_MASK |= (1ULL << idx);                        \
    else                                                     \
      AUTHORED_MASK &= ~(1ULL << idx);                       \
    return;                                                  \
  }                                                          \
  idx++;

  // model-level sub-structs (compiler, option, visual)
  if (el->elemtype == mjOBJ_MODEL) {
    mjCModel* cel = static_cast<mjCModel*>(el);
    int       idx = 0;

#define X(type, name, dim) SET_FIELD(cel->spec.compiler.name, cel->spec.compiler.authored)
#define XVEC(type, name, dim) SET_FIELD_VEC(cel->spec.compiler.name, cel->spec.compiler.authored)
    idx = 0;
    MJSCOMPILER_FIELDS
#undef X
#undef XVEC

#define X(type, name, dim) SET_FIELD(cel->spec.option.name, cel->spec.authored.option)
#define XVEC(type, name, dim) SET_FIELD_VEC(cel->spec.option.name, cel->spec.authored.option)
    idx = 0;
    MJOPTION_FIELDS
#undef X
#undef XVEC

#define X(type, name, dim) SET_FIELD(cel->spec.visual.global.name, cel->spec.authored.visual_global)
    idx = 0;
    MJVISUAL_GLOBAL_FIELDS
#undef X

#define X(type, name, dim) \
  SET_FIELD(cel->spec.visual.quality.name, cel->spec.authored.visual_quality)
    idx = 0;
    MJVISUAL_QUALITY_FIELDS
#undef X

#define X(type, name, dim) \
  SET_FIELD(cel->spec.visual.headlight.name, cel->spec.authored.visual_headlight)
#define XVEC(type, name, dim) \
  SET_FIELD_VEC(cel->spec.visual.headlight.name, cel->spec.authored.visual_headlight)
    idx = 0;
    MJVISUAL_HEADLIGHT_FIELDS
#undef X
#undef XVEC

#define X(type, name, dim) SET_FIELD(cel->spec.visual.map.name, cel->spec.authored.visual_map)
    idx = 0;
    MJVISUAL_MAP_FIELDS
#undef X

#define X(type, name, dim) SET_FIELD(cel->spec.visual.scale.name, cel->spec.authored.visual_scale)
    idx = 0;
    MJVISUAL_SCALE_FIELDS
#undef X

#define XVEC(type, name, dim) \
  SET_FIELD_VEC(cel->spec.visual.rgba.name, cel->spec.authored.visual_rgba)
    idx = 0;
    MJVISUAL_RGBA_FIELDS
#undef XVEC
  }

#undef SET_FIELD
#undef SET_FIELD_VEC
}


// increment reference count of element (for internal bindings use only)
void _mjPRIVATE_addRefElement(mjsElement* element) {
  if (!element) { return; }
  if (element->elemtype == mjOBJ_DEFAULT) {
    static_cast<mjCDef*>(element)->AddRef();
    return;
  }
  // embedded sub-elements inside mjCDef are by-value members with model == nullptr
  mjCBase* base = static_cast<mjCBase*>(element);
  if (base->model) { base->AddRef(); }
}


// decrement reference count of element (for internal bindings use only)
void _mjPRIVATE_releaseElement(mjsElement* element) {
  if (!element) { return; }
  if (element->elemtype == mjOBJ_DEFAULT) {
    static_cast<mjCDef*>(element)->Release();
    return;
  }
  // embedded sub-elements inside mjCDef are by-value members with model == nullptr
  mjCBase* base = static_cast<mjCBase*>(element);
  if (base->model) { base->Release(); }
}
