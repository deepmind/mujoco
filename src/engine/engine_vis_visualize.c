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

#include "engine/engine_vis_visualize.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjvisualize.h>
#include "engine/engine_array_safety.h"
#include "engine/engine_core_util.h"
#include "engine/engine_memory.h"
#include "engine/engine_name.h"
#include "engine/engine_plugin.h"
#include "engine/engine_sleep.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"
#include "engine/engine_vis_init.h"
#include "engine/engine_vis_interact.h"


//----------------------------- utility functions and macros ---------------------------------------

static const mjtNum IDENTITY[9] = {1, 0, 0,
                                   0, 1, 0,
                                   0, 0, 1};


// copy float array
static void f2f(float* dest, const float* src, int n) {
  memcpy(dest, src, n*sizeof(float));
}


// make text label
static void makeLabel(const mjModel* m, mjtObj type, int id, char* label) {
  const char* typestr = mju_type2Str(type);
  const char* namestr = mj_id2name(m, type, id);
  char txt[100];

  // copy existing name or make numeric name
  if (namestr) {
    mjSNPRINTF(txt, "%s", namestr);
  } else if (typestr) {
    mjSNPRINTF(txt, "%s %d", typestr, id);
  } else {
    mjSNPRINTF(txt, "%d", id);
  }

  // copy result into label
  strncpy(label, txt, 100);
  label[99] = '\0';
}


// convert HSV to RGB
void mjv_hsv2rgb(float *RGB, float H, float S, float V) {
  float R, G, B;

  if (S <= 0) {
    R = G = B = V;
  } else {
    float hh = H * 6;
    int i = (int)hh;
    float ff = hh - i;
    float p = V * (1 - S);
    float q = V * (1 - (S * ff));
    float t = V * (1 - (S * (1 - ff)));
    if (i == 0) {
      R=V;  G=t;  B=p;
    } else if (i == 1) {
      R=q;  G=V;  B=p;
    } else if (i == 2) {
      R=p;  G=V;  B=t;
    } else if (i == 3) {
      R=p;  G=q;  B=V;
    } else if (i == 4) {
      R=t;  G=p;  B=V;
    } else {
      R=V;  G=p;  B=q;
    }
  }

  RGB[0] = R;
  RGB[1] = G;
  RGB[2] = B;
}


// assign pseudo-random rgba to constraint island using Halton sequence
static void islandColor(float rgba[4], int h, int awake) {
  // default to gray R = G = B = 0.7;
  float hue        = 1.0f;
  float saturation = 0.0f;
  float value      = 0.7f;

  // island index given, use Halton sequence to generate pseudo-random color
  if (h >= 0) {
    // hue in [0, 1]
    hue = mju_Halton(h + 1, 7);

    // saturation in [0.5, 1.0]
    saturation = .5 + .5*mju_Halton(h + 1, 3);

    // value in [0.6, 1.0]
    value = .6 + .4*mju_Halton(h + 1, 5);
  }

  // if asleep, decrease saturation and value
  if (!awake) {
    value      *= 0.6;
    saturation *= 0.7;
  }

  mjv_hsv2rgb(rgba, hue, saturation, value);
  rgba[3] = 1;
}


// mix colors for perturbation object
static void mixcolor(float rgba[4], const float ref[4], int flg1, int flg2) {
  rgba[0] = flg1 ? ref[0] : 0;
  if (flg2) {
    rgba[0] = mjMAX(rgba[0], ref[1]);
  }

  rgba[1] = flg1 ? ref[1] : 0;
  if (flg2) {
    rgba[1] = mjMAX(rgba[1], ref[0]);
  }

  rgba[2] = ref[2];
  rgba[3] = ref[3];
}


// a body is static if it is welded to the world (mocap subtrees are their own weld)
static int bodycategory(const mjModel* m, int bodyid) {
  if (m->body_weldid[bodyid] == 0) {
    return mjCAT_STATIC;
  } else {
    return mjCAT_DYNAMIC;
  }
}


//----------------------------- geom functions -----------------------------------------------------

// acquires and initializes the next available geom in the scene
mjvGeom* acquireGeom(mjvScene* scn, int objid, int category, int objtype) {
  // check for overflow, SHOULD NOT OCCUR
  if (scn->ngeom >= scn->maxgeom) {
    if (!scn->status) {
      mju_warning("Pre-allocated visual geom buffer is full. "
                  "Increase maxgeom above %d.", scn->maxgeom);
      scn->status = 1;
    }
    return NULL;
  }

  mjvGeom* thisgeom = scn->geoms + scn->ngeom;
  memset(thisgeom, 0, sizeof(mjvGeom));
  mjv_initGeom(thisgeom, mjGEOM_NONE, NULL, NULL, NULL, NULL);
  thisgeom->objtype = objtype;
  thisgeom->objid = objid;
  thisgeom->category = category;
  thisgeom->segid = scn->ngeom;
  return thisgeom;
}


// mark geom as used, set its pointer to NULL, increment scn->ngeom
void releaseGeom(mjvGeom** geom, mjvScene* scn) {
  // check geom being released was most recently acquired, SHOULD NOT OCCUR
  if (*geom != scn->geoms + scn->ngeom) {
    mju_error("Unexpected geom pointer; did you call acquireGeom?");
  }

  scn->ngeom++;
  *geom = NULL;
}


// add a triangle to the scene
static void addTriangle(mjvScene* scn, const mjtNum v0[3], const mjtNum v1[3],
                        const mjtNum v2[3], const float rgba[4],
                        int objid, int category, int objtype) {
  mjvGeom* thisgeom = acquireGeom(scn, objid, category, objtype);
  if (!thisgeom) {
    return;
  }
  mjtNum e1[3] =  {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  mjtNum e2[3] =  {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  mjtNum normal[3];
  mju_cross(normal, e1, e2);
  mjtNum lengths[3] = {mju_normalize3(e1), mju_normalize3(e2), mju_normalize3(normal)};
  mjtNum xmat[9] = {e1[0], e2[0], normal[0],
                    e1[1], e2[1], normal[1],
                    e1[2], e2[2], normal[2]};
  mjv_initGeom(thisgeom, mjGEOM_TRIANGLE, lengths, v0, xmat, rgba);
  releaseGeom(&thisgeom, scn);
}


// copy material fields from model to visual geom
static void setMaterial(const mjModel* m, mjvGeom* geom, int matid, const float* rgba,
                        const mjtByte* flags) {
  // set material properties if given
  if (matid >= 0) {
    f2f(geom->rgba, m->mat_rgba + 4*matid, 4);
    geom->emission = m->mat_emission[matid];
    geom->specular = m->mat_specular[matid];
    geom->shininess = m->mat_shininess[matid];
    geom->reflectance = m->mat_reflectance[matid];
  }

  // use rgba if different from default, or no material given
  if (rgba[0] != 0.5f || rgba[1] != 0.5f || rgba[2] != 0.5f || rgba[3] != 1.0f || matid < 0) {
    f2f(geom->rgba, rgba, 4);
  }

  // set texture
  if (flags[mjVIS_TEXTURE] && matid >= 0) {
    geom->matid = matid;
    geom->texid = m->mat_texid[matid*mjNTEXROLE + mjTEXROLE_RGB];
    geom->texuniform = m->mat_texuniform[matid];
    geom->texrepeat[0] = m->mat_texrepeat[2*matid];
    geom->texrepeat[1] = m->mat_texrepeat[2*matid+1];
  }

  // scale alpha for dynamic geoms only
  if (flags[mjVIS_TRANSPARENT] && (geom->category == mjCAT_DYNAMIC)) {
    geom->rgba[3] *= m->vis.map.alpha;
  }
}


// set (type, size, pos, mat) connector-type geom between given points
//  assume that mjv_initGeom was already called to set all other properties
void mjv_connector(mjvGeom* geom, int type, mjtNum width,
                   const mjtNum from[3], const mjtNum to[3]) {
  mjtNum quat[4], mat[9], dif[3] = {to[0]-from[0], to[1]-from[1], to[2]-from[2]};

  // require connector-compatible type
  if (type != mjGEOM_CAPSULE && type != mjGEOM_CYLINDER &&
      type != mjGEOM_ARROW && type != mjGEOM_ARROW1 && type != mjGEOM_ARROW2
      && type != mjGEOM_LINE) {
    mjERROR("invalid geom type %d for connector", type);
  }

  // assign type
  geom->type = type;

  // compute size for XYZ scaling
  geom->size[0] = geom->size[1] = (float)width;
  geom->size[2] = (float)mju_norm3(dif);

  // cylinder and capsule are centered, and size[0] is "radius"
  if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
    geom->pos[0] = 0.5*(from[0] + to[0]);
    geom->pos[1] = 0.5*(from[1] + to[1]);
    geom->pos[2] = 0.5*(from[2] + to[2]);
    geom->size[2] *= 0.5;
  }

  // arrow is not centered
  else {
    geom->pos[0] = from[0];
    geom->pos[1] = from[1];
    geom->pos[2] = from[2];
  }

  // set mat to minimal rotation aligning b-a with z axis
  mju_quatZ2Vec(quat, dif);
  mju_quat2Mat(mat, quat);
  mju_n2f(geom->mat, mat, 9);
}


// add a connector to the scene
static void addConnector(mjvScene* scn, int type, mjtNum width,
                         const mjtNum from[3], const mjtNum to[3],
                         const float rgba[4], int objid, int category, int objtype) {
  mjvGeom* thisgeom = acquireGeom(scn, objid, category, objtype);
  if (!thisgeom) {
    return;
  }
  mjv_connector(thisgeom, type, width, from, to);
  if (rgba) f2f(thisgeom->rgba, rgba, 4);
  releaseGeom(&thisgeom, scn);
}


// initialize given fields when not NULL, set the rest to their default values
void mjv_initGeom(mjvGeom* geom, int type, const mjtNum* size,
                  const mjtNum* pos, const mjtNum* mat, const float* rgba) {
  // assign type
  geom->type = type;

  // set size (for XYZ scaling)
  if (size) {
    switch ((mjtGeom) type) {
    case mjGEOM_SPHERE:
      geom->size[0] = (float)size[0];
      geom->size[1] = (float)size[0];
      geom->size[2] = (float)size[0];
      break;

    case mjGEOM_CAPSULE:
      geom->size[0] = (float)size[0];
      geom->size[1] = (float)size[0];
      geom->size[2] = (float)size[1];
      break;

    case mjGEOM_CYLINDER:
      geom->size[0] = (float)size[0];
      geom->size[1] = (float)size[0];
      geom->size[2] = (float)size[1];
      break;

    default:
      mju_n2f(geom->size, size, 3);
    }
  } else {
    geom->size[0] = 0.1f;
    geom->size[1] = 0.1f;
    geom->size[2] = 0.1f;
  }

  // set pos
  if (pos) {
    mju_n2f(geom->pos, pos, 3);
  } else {
    geom->pos[0] = 0;
    geom->pos[1] = 0;
    geom->pos[2] = 0;
  }

  // set mat
  if (mat) {
    mju_n2f(geom->mat, mat, 9);
  } else {
    geom->mat[0] = 1;
    geom->mat[1] = 0;
    geom->mat[2] = 0;
    geom->mat[3] = 0;
    geom->mat[4] = 1;
    geom->mat[5] = 0;
    geom->mat[6] = 0;
    geom->mat[7] = 0;
    geom->mat[8] = 1;
  }

  // set rgba
  if (rgba) {
    f2f(geom->rgba, rgba, 4);
  } else {
    geom->rgba[0] = 0.5;
    geom->rgba[1] = 0.5;
    geom->rgba[2] = 0.5;
    geom->rgba[3] = 1;
  }

  // set defaults that cannot be assigned via this function
  geom->dataid       = -1;
  geom->matid        = -1;
  geom->texid        = -1;
  geom->texuniform   = 0;
  geom->texrepeat[0] = 0;
  geom->texrepeat[1] = 0;
  geom->texcoord     = 0;
  geom->emission     = 0;
  geom->specular     = 0.5;
  geom->shininess    = 0.5;
  geom->reflectance  = 0;
  geom->castshadow   = 1;
  geom->label[0]     = 0;
  geom->modelrbound  = 0;
}


// mark geom as selected
static void markselected(const mjVisual* vis, mjvGeom* geom) {
  // add emission
  geom->emission += vis->global.glow;
}


// draw 3 cylinders representing a "frame" decor element
void addFrame(mjvScene* scn, int objid, const mjtNum pos[3], const mjtNum rot[9], float length,
              float width) {
  // draw separate geoms for each axis
  for (int j=0; j < 3; j++) {
    mjtNum axis[3];
    for (int k=0; k < 3; k++) {
      axis[k] = (j == k ? length : 0);
    }

    mjtNum vec[3];
    mju_mulMatVec3(vec, rot, axis);

    // create a cylinder
    mjtNum to[3];
    mju_add3(to, pos, vec);

    mjvGeom* thisgeom = acquireGeom(scn, objid, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    mjv_connector(thisgeom, mjGEOM_CYLINDER, width, pos, to);
    for (int k=0; k < 3; k++) {
      thisgeom->rgba[k] = (j == k ? 0.9 : 0);
    }
    thisgeom->rgba[3] = 1;
    releaseGeom(&thisgeom, scn);
  }
}


//----------------------------- camera functions --------------------------------------------------

// computes the camera frustum
static void getFrustum(float zver[2], float zhor[2], float znear,
                       const float intrinsic[4], const float sensorsize[2]) {
  if (zhor) {
    zhor[0] = znear / intrinsic[0] * (sensorsize[0]/2.f - intrinsic[2]);
    zhor[1] = znear / intrinsic[0] * (sensorsize[0]/2.f + intrinsic[2]);
  }
  if (zver) {
    zver[0] = znear / intrinsic[1] * (sensorsize[1]/2.f - intrinsic[3]);
    zver[1] = znear / intrinsic[1] * (sensorsize[1]/2.f + intrinsic[3]);
  }
}


void mjv_cameraFrame(mjtNum headpos[3], mjtNum forward[3], mjtNum up[3], mjtNum right[3],
                     const mjData* d, const mjvCamera* cam) {
  switch (cam->type) {
    case mjCAMERA_FREE:
    case mjCAMERA_TRACKING: {
      const mjtNum ca = mju_cos(cam->azimuth/180.0*mjPI);
      const mjtNum sa = mju_sin(cam->azimuth/180.0*mjPI);
      const mjtNum ce = mju_cos(cam->elevation/180.0*mjPI);
      const mjtNum se = mju_sin(cam->elevation/180.0*mjPI);
      if (forward) {
        forward[0] = ce*ca;
        forward[1] = ce*sa;
        forward[2] = se;
      }
      if (up) {
        up[0] = -se*ca;
        up[1] = -se*sa;
        up[2] = ce;
      }
      if (right) {
        right[0] = sa;
        right[1] = -ca;
        right[2] = 0;
      }
      if (headpos) {
        if (cam->type == mjCAMERA_TRACKING && cam->trackbodyid >= 0) {
          if (!d) {
            mjERROR("data pointer is NULL");
          }
          mju_addScl3(headpos, d->subtree_com + 3*cam->trackbodyid, forward, -cam->distance);
        } else {
          mju_addScl3(headpos, cam->lookat, forward, -cam->distance);
        }
      }
      break;
    }

    case mjCAMERA_FIXED: {
      const int cid = cam->fixedcamid;
      if (cid < 0) {
        mjERROR("fixed camera id is outside valid range");
      }
      if (!d) {
        mjERROR("data pointer is NULL");
      }
      const mjtNum* mat = d->cam_xmat + 9*cid;
      if (forward) {
        forward[0] = -mat[2];
        forward[1] = -mat[5];
        forward[2] = -mat[8];
      }
      if (up) {
        up[0] = mat[1];
        up[1] = mat[4];
        up[2] = mat[7];
      }
      if (right) {
        right[0] = mat[0];
        right[1] = mat[3];
        right[2] = mat[6];
      }
      if (headpos) {
        mju_copy3(headpos, d->cam_xpos + 3*cid);
      }
      break;
    }

    default: {
      mjERROR("unknown camera type");
    }
  }
}


void mjv_cameraFrustum(float zver[2], float zhor[2], float zclip[2], const mjModel* m,
                       const mjvCamera* cam) {
  mjtNum fovy;
  int orthographic = 0, cid = 0;
  float* intrinsic = NULL;
  float* sensorsize = NULL;

  // get ipd, fovy, orthographic, intrinsic
  switch (cam->type) {
  case mjCAMERA_FREE:
  case mjCAMERA_TRACKING:
    orthographic = m->vis.global.orthographic;
    fovy = m->vis.global.fovy;
    break;

  case mjCAMERA_FIXED:
    // get id, check range
    cid = cam->fixedcamid;
    if (cid < 0 || cid >= m->ncam) {
      mjERROR("fixed camera id is outside valid range");
    }
    orthographic = m->cam_projection[cid] == mjPROJ_ORTHOGRAPHIC;
    fovy = m->cam_fovy[cid];

    // if positive sensorsize, get sensorsize and intrinsic
    if (m->cam_sensorsize[2*cid+1]) {
      sensorsize = m->cam_sensorsize + 2*cid;
      intrinsic = m->cam_intrinsic + 4*cid;
    }
    break;

  default:
    mjERROR("unknown camera type");
  }

  const float znear = m->vis.map.znear * m->stat.extent;

  if (orthographic) {
    if (zver) zver[0] = zver[1] = fovy / 2;
    if (zhor) zhor[0] = zhor[1] = 0.0f;
  } else if (intrinsic) {
    getFrustum(zver, zhor, znear, intrinsic, sensorsize);
  } else {
    if (zver) zver[0] = zver[1] = znear * mju_tan(fovy * mjPI/360.0);
    if (zhor) zhor[0] = zhor[1] = 0.0f;
  }

  if (zclip) {
    zclip[0] = znear;
    zclip[1] = m->vis.map.zfar * m->stat.extent;
  }
}


//----------------------------- main API functions -------------------------------------------------

// add contact-related geoms in mjvObject
static void addContactGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn,
                           int catmask) {
  if (!vopt->flags[mjVIS_CONTACTPOINT] && !vopt->flags[mjVIS_CONTACTFORCE] &&
      vopt->frame != mjFRAME_CONTACT) {
    return;
  }

  int objtype = mjOBJ_UNKNOWN, category = mjCAT_DECOR;
  mjtNum mat[9], tmp[9], vec[3], frc[3], confrc[6];
  mjtNum framewidth, framelength, scl = m->stat.meansize;
  mjContact* con;
  mjvGeom* thisgeom;
  mjtByte split;

  // loop over contacts
  for (int i=0; i < d->ncon; i++) {
    // get pointer
    con = d->contact + i;

    // mat = contact rotation matrix (normal along z)
    mju_copy(tmp, con->frame+3, 6);
    mju_copy(tmp+6, con->frame, 3);
    mju_transpose(mat, tmp, 3, 3);

    // contact point
    if (vopt->flags[mjVIS_CONTACTPOINT]) {
      thisgeom = acquireGeom(scn, i, category, objtype);
      if (!thisgeom) {
        return;
      }

      thisgeom->type = mjGEOM_CYLINDER;
      thisgeom->size[0] = thisgeom->size[1] = m->vis.scale.contactwidth * scl;
      float halfheight = m->vis.scale.contactheight * scl;
      float halfdepth = -con->dist / 2;
      thisgeom->size[2] = mjMAX(halfheight, halfdepth);
      mju_n2f(thisgeom->pos, con->pos, 3);
      mju_n2f(thisgeom->mat, mat, 9);

      int efc_adr = d->contact[i].efc_address;

      // override standard colors if visualizing islands
      if (vopt->flags[mjVIS_ISLAND] && efc_adr >= 0) {
        // set hue using island's first dof
        int h = d->nisland > 0 ? d->island_dofadr[d->efc_island[efc_adr]] : -1;
        islandColor(thisgeom->rgba, h, /*awake*/1);
      }

      // otherwise regular colors (different for included and excluded contacts)
      else {
        if (efc_adr >= 0) {
          f2f(thisgeom->rgba, m->vis.rgba.contactpoint, 4);
        } else {
          f2f(thisgeom->rgba, m->vis.rgba.contactgap, 4);
        }
      }

      // label contacting geom names or ids
      if (vopt->label == mjLABEL_CONTACTPOINT) {
        char contactlabel[2][48];
        for (int k=0; k < 2; k++) {
          // make geom label
          if (con->geom[k] >= 0) {
            const char* geomname = mj_id2name(m, mjOBJ_GEOM, con->geom[k]);
            if (geomname) {
              mjSNPRINTF(contactlabel[k], "%s", geomname);
            } else {
              mjSNPRINTF(contactlabel[k], "g%d", con->geom[k]);
            }
          }

          // make flex elem or vert label
          else {
            const char* flexname = mj_id2name(m, mjOBJ_FLEX, con->flex[k]);
            if (flexname) {
              if (con->elem[k] >= 0) {
                mjSNPRINTF(contactlabel[k], "%s.e%d", flexname, con->elem[k]);
              } else {
                mjSNPRINTF(contactlabel[k], "%s.v%d", flexname, con->vert[k]);
              }
            }
            else {
              if (con->elem[k] >= 0) {
                mjSNPRINTF(contactlabel[k], "f%d.e%d", con->flex[k], con->elem[k]);
              } else {
                mjSNPRINTF(contactlabel[k], "f%d.v%d", con->flex[k], con->vert[k]);
              }
            }
          }
        }

        mjSNPRINTF(thisgeom->label, "%s | %s", contactlabel[0], contactlabel[1]);
      }

      float contactrgba[4];
      f2f(contactrgba, thisgeom->rgba, 4);
      releaseGeom(&thisgeom, scn);

      // surface velocity: relative to static side, or symmetric if both move
      if (efc_adr >= 0 && con->dim > 1 && (con->surfacevel[0] || con->surfacevel[1])) {
        const mjtNum kVelocityMap = 0.5;  // units of time: arrow length = velocity * kVelocityMap
        mjtNum vw[3];
        mju_scl3(vw, con->frame + 3, -con->surfacevel[0]);
        mju_addToScl3(vw, con->frame + 6, -con->surfacevel[1]);
        if (mju_norm3(vw) >= mjMINVAL) {
          int movable[2] = {0, 0};
          for (int k=0; k < 2; k++) {
            if (con->geom[k] >= 0) {
              movable[k] = (m->body_treeid[m->geom_bodyid[con->geom[k]]] >= 0);
            } else {
              int body[729];
              int nb = mj_contactBodyWeight(m, d, con, k, body, NULL);
              for (int j=0; j < nb; j++) {
                if (m->body_treeid[body[j]] >= 0) {
                  movable[k] = 1;
                  break;
                }
              }
            }
          }
          if (movable[0] && !movable[1]) {
            mju_scl3(vw, vw, -1);
          }

          mjtNum to[3];
          mju_addScl3(to, con->pos, vw, kVelocityMap);
          thisgeom = acquireGeom(scn, i, category, objtype);
          if (!thisgeom) {
            return;
          }
          int arrow = (movable[0] == movable[1]) ? mjGEOM_ARROW2 : mjGEOM_ARROW;
          mjv_connector(thisgeom, arrow, m->vis.scale.forcewidth * scl, con->pos, to);
          f2f(thisgeom->rgba, contactrgba, 4);
          releaseGeom(&thisgeom, scn);
        }
      }
    }

    // mat = contact frame rotation matrix (normal along x)
    mju_transpose(mat, con->frame, 3, 3);

    // contact frame
    if (vopt->frame == mjFRAME_CONTACT) {
      // set length and width of axis cylinders using half regular frame scaling
      framelength = m->vis.scale.framelength * scl / 2;
      framewidth = m->vis.scale.framewidth * scl / 2;
      addFrame(scn, i, con->pos, mat, framelength, framewidth);
    }

    // nothing else to do for excluded contacts
    if (d->contact[i].efc_address < 0) {
      continue;
    }

    // get contact force:torque in contact frame
    mj_contactForce(m, d, i, confrc);

    // contact force
    if (vopt->flags[mjVIS_CONTACTFORCE]) {
      // get force, fill zeros if only normal
      mju_zero3(frc);
      mju_copy(frc, confrc, mjMIN(3, con->dim));
      if (mju_norm3(frc) < mjMINVAL) {
        continue;
      }

      // render combined or split
      split = (vopt->flags[mjVIS_CONTACTSPLIT] && con->dim > 1);
      for (int j = (split ? 1 : 0); j < (split ? 3 : 1); j++) {
        // set vec to combined, normal or friction force, in world frame
        switch (j) {
        case 0:             // combined
          mju_mulMatVec(vec, mat, frc, 3, 3);
          break;
        case 1:             // normal
          vec[0] = mat[0]*frc[0];
          vec[1] = mat[3]*frc[0];
          vec[2] = mat[6]*frc[0];
          break;
        case 2:             // friction
          vec[0] = mat[1]*frc[1] + mat[2]*frc[2];
          vec[1] = mat[4]*frc[1] + mat[5]*frc[2];
          vec[2] = mat[7]*frc[1] + mat[8]*frc[2];
          break;
        }

        // scale vector
        mju_scl3(vec, vec, m->vis.map.force/m->stat.meanmass);

        // get bodyflex ids
        int bf[2];
        for (int k=0; k < 2; k++) {
          bf[k] = (con->geom[k] >= 0) ? m->geom_bodyid[con->geom[k]] :
                                        m->nbody + con->flex[k];
        }

        // make sure arrow points towards bodyflex with higher id
        if (bf[0] > bf[1]) {
          mju_scl3(vec, vec, -1);
        }

        // one-directional arrow for friction and world, symmetric otherwise
        thisgeom = acquireGeom(scn, i, category, objtype);
        if (!thisgeom) {
          return;
        }

        mjtNum* from = con->pos;
        mjtNum to[3];
        mju_add3(to, from, vec);
        mjv_connector(thisgeom,
                      bf[0] > 0 && bf[1] > 0 && !split ? mjGEOM_ARROW2 : mjGEOM_ARROW,
                      m->vis.scale.forcewidth * scl, from, to);
        f2f(thisgeom->rgba, j == 2 ? m->vis.rgba.contactfriction : m->vis.rgba.contactforce, 4);
        if (vopt->label == mjLABEL_CONTACTFORCE && j == (split ? 1 : 0)) {
          mjSNPRINTF(thisgeom->label, "%-.3g", mju_norm3(frc));
        }
        releaseGeom(&thisgeom, scn);
      }
    }
  }
}


// compute material id of the element that owns each shell fragment of flex f
static void flexShellMatid(const mjModel* m, mjData* d, int f, int* shell_matid) {
  int dim = m->flex_dim[f];
  int vertnum = m->flex_vertnum[f];
  int elemadr = m->flex_elemadr[f];
  int elemnum = m->flex_elemnum[f];
  int shellnum = m->flex_shellnum[f];
  const int* elemdata = m->flex_elem + m->flex_elemdataadr[f];
  const int* shelldata = m->flex_shell + m->flex_shelldataadr[f];

  // build vertex-to-element adjacency (CSR) for O(nelem + nshell) lookup
  mj_markStack(d);
  int* v_cnt = mjSTACKALLOC(d, vertnum, int);
  int* v_adr = mjSTACKALLOC(d, vertnum, int);
  int* v_elem = mjSTACKALLOC(d, (dim+1)*elemnum, int);
  mju_zeroInt(v_cnt, vertnum);
  for (int i=0; i < (dim+1)*elemnum; i++) {
    v_cnt[elemdata[i]]++;
  }
  v_adr[0] = 0;
  for (int v=0; v < vertnum-1; v++) {
    v_adr[v+1] = v_adr[v] + v_cnt[v];
  }
  mju_zeroInt(v_cnt, vertnum);
  for (int e=0; e < elemnum; e++) {
    const int* edata = elemdata + e*(dim+1);
    for (int j=0; j < dim+1; j++) {
      int v = edata[j];
      v_elem[v_adr[v] + v_cnt[v]++] = e;
    }
  }

  // find owning element of each shell fragment among elements incident to sdata[0]
  int first = m->flex_matid[elemadr];
  for (int s=0; s < shellnum; s++) {
    const int* sdata = shelldata + s*dim;
    int v0 = sdata[0];
    int matid = first;
    for (int idx=0; idx < v_cnt[v0]; idx++) {
      int e = v_elem[v_adr[v0] + idx];
      const int* edata = elemdata + e*(dim+1);
      int match = 1;
      for (int k=1; k < dim; k++) {
        int found = 0;
        for (int j=0; j < dim+1; j++) {
          if (edata[j] == sdata[k]) {
            found = 1;
            break;
          }
        }
        if (!found) {
          match = 0;
          break;
        }
      }
      if (match) {
        matid = m->flex_matid[elemadr+e];
        break;
      }
    }
    shell_matid[s] = matid;
  }
  mj_freeStack(d);
}


// collect unique material ids used by flex f in order of first appearance
static int flexUniqueMatids(const mjModel* m, int f, const mjvOption* opt, int* matids, int maxmat) {
  int elemadr = m->flex_elemadr[f];
  int elemnum = m->flex_elemnum[f];
  if (elemnum <= 0 || opt->flags[mjVIS_ISLAND]) {
    matids[0] = elemnum > 0 ? m->flex_matid[elemadr] : -1;
    return 1;
  }

  int nmat = 0;
  for (int e=0; e < elemnum; e++) {
    int matid = m->flex_matid[elemadr+e];
    int found = 0;
    for (int k=0; k < nmat; k++) {
      if (matids[k] == matid) {
        found = 1;
        break;
      }
    }
    if (!found && nmat < maxmat) {
      matids[nmat++] = matid;
    }
  }
  return nmat ? nmat : 1;
}


static void addFlexGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                         const mjvPerturb* pert, int catmask, mjvScene* scn) {
  const int category = mjCAT_DYNAMIC;
  if (!(category & catmask)) {
    return;
  }
  if (!vopt->flags[mjVIS_FLEXVERT] && !vopt->flags[mjVIS_FLEXEDGE] &&
      !vopt->flags[mjVIS_FLEXFACE] && !vopt->flags[mjVIS_FLEXSKIN]) {
    return;
  }

  mj_markStack(d);
  int maxmat = m->nmat + 1;
  int* matids = mjSTACKALLOC(d, maxmat, int);

  for (int i=0; i < m->nflex; i++) {
    if (!vopt->flexgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->flex_group[i]))]) {
      continue;
    }

    int nmat = flexUniqueMatids(m, i, vopt, matids, maxmat);
    for (int k=0; k < nmat; k++) {
      mjvGeom* thisgeom = acquireGeom(scn, i, category, mjOBJ_FLEX);
      if (!thisgeom) {
        mj_freeStack(d);
        return;
      }

      // construct geom, pos = first vertex
      // for multi-material flexes, dataid is the slice index and size[1..2] hold
      // the face offset and face count populated in mjv_updateActiveFlex
      mjv_initGeom(thisgeom, mjGEOM_FLEX, NULL,
                    d->flexvert_xpos + 3*m->flex_vertadr[i], NULL, NULL);
      thisgeom->dataid = (nmat > 1) ? k : -1;
      thisgeom->size[0] = m->flex_radius[i];
      if (nmat > 1) {
        thisgeom->size[1] = 0;
        thisgeom->size[2] = 0;
      }
      setMaterial(m, thisgeom, matids[k], m->flex_rgba+4*i, vopt->flags);

      // override if visualizing islands
      if (vopt->flags[mjVIS_ISLAND]) {
        // find first dynamic body in flex
        int bodyid = -1;
        if (m->flex_interp[i]) {
          int nodeadr = m->flex_nodeadr[i];
          for (int j=0; j < m->flex_nodenum[i] && bodyid < 0; j++) {
            int b = m->flex_nodebodyid[nodeadr+j];
            if (m->body_treeid[b] >= 0) bodyid = b;
          }
        } else {
          int vertadr = m->flex_vertadr[i];
          for (int j=0; j < m->flex_vertnum[i] && bodyid < 0; j++) {
            int b = m->flex_vertbodyid[vertadr+j];
            if (m->body_treeid[b] >= 0) bodyid = b;
          }
        }

        if (bodyid >= 0) {
          // strip material
          thisgeom->matid = -1;

          int weld_id = m->body_weldid[bodyid];
          int dof = m->body_dofadr[weld_id];
          int island = d->nisland ? d->dof_island[dof] : -1;
          int h = island >= 0 ? d->island_dofadr[island] : -1;
          int awake = d->body_awake[bodyid];

          // if sleep is enabled, color by first tree dof
          if (h == -1 && mjENABLED(mjENBL_SLEEP)) {
            int tree = m->dof_treeid[dof];
            if (!awake) tree = mj_sleepCycle(d->tree_asleep, m->ntree, tree);
            h = m->tree_dofadr[tree];
          }

          islandColor(thisgeom->rgba, h, awake);
        }
      }

      // set texcoord
      if (m->flex_texcoordadr[i] >= 0) {
        thisgeom->texcoord = 1;
      } else {
        thisgeom->matid = -1;
      }

      // glow flex if selected
      if (pert->flexselect == i) {
        markselected(&m->vis, thisgeom);
      }

      // skip if alpha is 0
      if (thisgeom->rgba[3] == 0) {
        continue;
      }

      // vopt->label (only on first part of multi-material flex)
      if (vopt->label == mjLABEL_FLEX && k == 0) {
        makeLabel(m, mjOBJ_FLEX, i, thisgeom->label);
      }

      releaseGeom(&thisgeom, scn);
    }
  }
  mj_freeStack(d);
}


static void addSkinGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                         const mjvPerturb* pert, int catmask, mjvScene* scn) {
  const int category = mjCAT_DYNAMIC;
  if (!(category & catmask)) {
    return;
  }
  if (!vopt->flags[mjVIS_SKIN]) {
    return;
  }

  for (int i=0; i < m->nskin; i++) {
    if (!vopt->skingroup[mjMAX(0, mjMIN(mjNGROUP-1, m->skin_group[i]))]) {
      continue;
    }
    mjvGeom* thisgeom = acquireGeom(scn, i, category, mjOBJ_SKIN);
    if (!thisgeom) {
      return;
    }

    // construct geom, pos = first bone
    mjv_initGeom(thisgeom, mjGEOM_SKIN, NULL,
                  d->xpos + 3*m->skin_bonebodyid[m->skin_boneadr[i]], NULL, NULL);

    // set material properties
    setMaterial(m, thisgeom, m->skin_matid[i], m->skin_rgba+4*i, vopt->flags);

    // glow skin if selected
    if (pert->skinselect == i) {
      markselected(&m->vis, thisgeom);
    }

    // set texcoord
    if (m->skin_texcoordadr[i] >= 0) {
      thisgeom->texcoord = 1;
    }

    // skip if alpha is 0
    if (thisgeom->rgba[3] == 0) {
      continue;
    }

    // vopt->label
    if (vopt->label == mjLABEL_SKIN) {
      makeLabel(m, mjOBJ_SKIN, i, thisgeom->label);
    }

    releaseGeom(&thisgeom, scn);
  }
}


static void addGeomGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                         const mjvPerturb* pert, int catmask, mjvScene* scn) {
  const int objtype = mjOBJ_GEOM;
  int planeid = -1;
  for (int i=0; i < m->ngeom; i++) {
    // count planes, put current plane number in geom->dataid
    if (m->geom_type[i] == mjGEOM_PLANE) {
      planeid++;
    }

    // skip if category is masked
    int category = bodycategory(m, m->geom_bodyid[i]);
    if (!(category & catmask)) {
      continue;
    }

    // skip if group is disabled
    if (!vopt->geomgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->geom_group[i]))]) {
      continue;
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, category, objtype);
    if (!thisgeom) {
      return;
    }

    // construct geom
    mjv_initGeom(thisgeom, m->geom_type[i], m->geom_size+3*i,
                  d->geom_xpos+3*i, d->geom_xmat+9*i, NULL);
    thisgeom->dataid = m->geom_dataid[i];

    // copy rbound and castshadow from model
    thisgeom->modelrbound = (float)m->geom_rbound[i];
    thisgeom->castshadow = m->geom_castshadow[i];

    // set material properties
    float* rgba = m->geom_rgba+4*i;
    int geom_matid = m->geom_matid[i];
    setMaterial(m, thisgeom, geom_matid, rgba, vopt->flags);

    // override if visualizing islands
    if (vopt->flags[mjVIS_ISLAND]) {
      int weld_id = m->body_weldid[m->geom_bodyid[i]];
      if (m->body_dofnum[weld_id]) {
        // strip materials off moving geom
        thisgeom->matid = -1;

        // set hue using first island dof, -1 if no island
        int dof = m->body_dofadr[weld_id];
        int island = d->nisland ? d->dof_island[dof] : -1;
        int h = island >= 0 ? d->island_dofadr[island] : -1;
        int awake = d->body_awake[m->geom_bodyid[i]];

        // if sleep is enabled, color by first tree dof
        if (h == -1 && mjENABLED(mjENBL_SLEEP)) {
          int tree = m->dof_treeid[dof];
          if (!awake) tree = mj_sleepCycle(d->tree_asleep, m->ntree, tree);
          h = m->tree_dofadr[tree];
        }

        islandColor(thisgeom->rgba, h, awake);
      }
    }

    // set texcoord
    if ((m->geom_type[i] == mjGEOM_MESH || m->geom_type[i] == mjGEOM_SDF) &&
        m->geom_dataid[i] >= 0 &&
        m->mesh_texcoordadr[m->geom_dataid[i]] >= 0) {
      thisgeom->texcoord = 1;
    }

    // skip if alpha is 0
    if (thisgeom->rgba[3] == 0) {
      continue;
    }

    // glow geoms of selected body
    if (pert->select > 0 && pert->select == m->geom_bodyid[i]) {
      markselected(&m->vis, thisgeom);
    }

    // vopt->label
    if (vopt->label == mjLABEL_GEOM) {
      makeLabel(m, mjOBJ_GEOM, i, thisgeom->label);
    }

    // mesh: 2*i is original, 2*i+1 is convex hull
    if (m->geom_type[i] == mjGEOM_MESH || m->geom_type[i] == mjGEOM_SDF) {
      thisgeom->dataid *= 2;
      if (m->mesh_graphadr[m->geom_dataid[i]] >= 0 && vopt->flags[mjVIS_CONVEXHULL] &&
          (m->geom_contype[i] || m->geom_conaffinity[i])) {
        thisgeom->dataid += 1;
      }
    }

    // plane
    else if (m->geom_type[i] == mjGEOM_PLANE) {
      // use current planeid
      thisgeom->dataid = planeid;

      // save initial pos
      mjtNum tmp[9];
      mju_copy3(tmp, d->geom_xpos+3*i);

      // re-center infinite plane
      if (m->geom_size[3*i] <= 0 || m->geom_size[3*i+1] <= 0) {
        // vec = headpos - geompos
        mjtNum vec[3];
        for (int j=0; j < 3; j++) {
          vec[j] = 0.5*(scn->camera[0].pos[j] + scn->camera[1].pos[j]) - d->geom_xpos[3*i+j];
        }

        // construct axes
        mjtNum ax[9];
        mju_transpose(ax, d->geom_xmat+9*i, 3, 3);

        // loop over (x,y)
        for (int k=0; k < 2; k++) {
          if (m->geom_size[3*i+k] <= 0) {
            // compute zfar
            mjtNum zfar = m->vis.map.zfar * m->stat.extent;

            // get size increment
            mjtNum sX;
            int matid = m->geom_matid[i];
            if (matid >= 0 && m->mat_texrepeat[2*matid+k] > 0) {
              sX = 2/m->mat_texrepeat[2*matid+k];
            } else {
              sX = 2.1*zfar/(mjMAXPLANEGRID-2);
            }

            // project on frame, round to integer increment of size
            mjtNum dX = mju_dot3(vec, ax+3*k);
            dX = 2*sX*mju_round(0.5*dX/sX);

            // translate
            mju_addToScl3(tmp, ax+3*k, dX);
          }
        }
      }

      // set final pos
      mju_n2f(thisgeom->pos, tmp, 3);
    }

    releaseGeom(&thisgeom, scn);
  }
}


static void addSiteGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                         const mjvPerturb* pert, int catmask, mjvScene* scn) {
  for (int i=0; i < m->nsite; i++) {
    // skip if category is masked
    int category = bodycategory(m, m->site_bodyid[i]);
    if (!(category & catmask)) {
      continue;
    }

    // skip if group disabled
    if (!vopt->sitegroup[mjMAX(0, mjMIN(mjNGROUP-1, m->site_group[i]))]) {
      continue;
    }
    mjvGeom* thisgeom = acquireGeom(scn, i, category, mjOBJ_SITE);
    if (!thisgeom) {
      return;
    }

    // construct geom
    const mjtNum* pos = d->site_xpos+3*i;
    const mjtNum* mat = d->site_xmat+9*i;
    mjtNum meshpos[3], meshmat[9];
    if (m->site_type[i] == mjGEOM_MESH && m->site_dataid[i] >= 0) {
      int meshid = m->site_dataid[i];
      mjtNum xquat[4];
      mju_mulMatVec3(meshpos, mat, m->mesh_pos+3*meshid);
      mju_addTo3(meshpos, pos);
      mju_mulQuat(xquat, d->site_xquat+4*i, m->mesh_quat+4*meshid);
      mju_quat2Mat(meshmat, xquat);
      pos = meshpos;
      mat = meshmat;
    }

    mjv_initGeom(thisgeom, m->site_type[i], m->site_size+3*i, pos, mat, NULL);
    thisgeom->dataid = m->site_dataid[i];

    // set texcoord
    if (m->site_type[i] == mjGEOM_MESH &&
        m->site_dataid[i] >= 0 &&
        m->mesh_texcoordadr[m->site_dataid[i]] >= 0) {
      thisgeom->texcoord = 1;
    }

    // set material if given
    setMaterial(m, thisgeom, m->site_matid[i], m->site_rgba+4*i, vopt->flags);

    // skip if alpha is 0
    if (thisgeom->rgba[3] == 0) {
      continue;
    }

    // glow
    if (pert->select > 0 && pert->select == m->site_bodyid[i]) {
      markselected(&m->vis, thisgeom);
    }

    // vopt->label
    if (vopt->label == mjLABEL_SITE) {
      makeLabel(m, mjOBJ_SITE, i, thisgeom->label);
    }

    // mesh: 2*i is original, 2*i+1 is convex hull
    if (m->site_type[i] == mjGEOM_MESH) {
      thisgeom->dataid *= 2;
      if (m->mesh_graphadr[m->site_dataid[i]] >= 0 && vopt->flags[mjVIS_CONVEXHULL]) {
        thisgeom->dataid += 1;
      }
    }

    releaseGeom(&thisgeom, scn);
  }
}


int mjv_isCatenary(const mjModel* m, const mjData* d, int i, mjtNum* length) {
  int has_stiffness = m->tendon_stiffness[i] ||
                      !mju_isZero(m->tendon_stiffnesspoly+mjNPOLY*i, mjNPOLY);
  const mjtNum* lengthspring = m->tendon_plasticadr[i] >= 0 ?
                               d->plastic + m->tendon_plasticadr[i] :
                               m->tendon_lengthspring + 2*i;

  // tendon has a deadband spring
  int limitedspring =
    has_stiffness                         &&    // positive stiffness
    lengthspring[0] == 0                  &&    // range lower-bound is 0
    lengthspring[1] > 0;                        // range upper-bound is positive

  // tendon has a simple length constraint, but is currently not limited
  mjtNum ten_length = d->ten_length[i];
  mjtNum lower = m->tendon_range[2*i];
  mjtNum upper = m->tendon_range[2*i + 1];
  int limitedconstraint =
    !has_stiffness                        &&    // zero stiffness
    m->tendon_limited[i] == 1             &&    // limited length range
    lower == 0                            &&    // range lower-bound is 0
    ten_length < upper;                         // current length is smaller than upper bound

  int has_damping = m->tendon_damping[i] || !mju_isZero(m->tendon_dampingpoly+mjNPOLY*i, mjNPOLY);

  // conditions for drawing a catenary
  int draw_catenary =
    !mjDISABLED(mjDSBL_GRAVITY)           &&    // gravity enabled
    mju_norm3(m->opt.gravity) > mjMINVAL  &&    // gravity strictly nonzero
    m->tendon_num[i] == 2                 &&    // only two sites on the tendon
    (limitedspring != limitedconstraint)  &&    // either spring or constraint length limits
    !has_damping                          &&    // no damping
    m->tendon_frictionloss[i] == 0;             // no frictionloss

  // no actuator
  if (draw_catenary) {
    for (int j=0; j < m->nactuator; j++) {
      if (m->actuator_trntype[j] == mjTRN_TENDON && m->actuator_trnid[2*j] == i) {
        draw_catenary = 0;
        break;
      }
    }
  }

  if (draw_catenary) {
    // length of the tendon
    if (limitedconstraint) {
      *length = m->tendon_range[2*i+1];
    } else {
      *length = lengthspring[1];
    }
  }

  return draw_catenary;
}


// points along geodesic on sphere or cylinder, returns number of points
int mjv_geodesic(mjtNum* pts, int npts, const mjtNum xpos[3], const mjtNum xmat[9],
                 mjtNum radius, int type, const mjtNum wpnt[12]) {
  const mjtNum* x1 = wpnt + 0;  // incoming site before wrapping geom
  const mjtNum* p1 = wpnt + 3;  // entry tangent point on geom surface
  const mjtNum* p2 = wpnt + 6;  // exit tangent point on geom surface
  const mjtNum* x2 = wpnt + 9;  // outgoing site after wrapping geom

  // coincident endpoints: skip segment
  if (mju_dist3(p1, p2) < mjMINVAL) {
    return 0;
  }
  if (npts <= 2 || radius < mjMINVAL) {
    mju_copy3(pts+0, p1);
    mju_copy3(pts+3, p2);
    return 2;
  }

  // radial vectors from geom center
  mjtNum r1[3], r2[3], w[3] = {0, 0, 0};
  mju_sub3(r1, p1, xpos);
  mju_sub3(r2, p2, xpos);

  // cylinder: project radial vectors perpendicular to cylinder axis
  mjtNum z1 = 0, z2 = 0;
  if (type == mjGEOM_CYLINDER) {
    w[0] = xmat[2];
    w[1] = xmat[5];
    w[2] = xmat[8];
    z1 = mju_dot3(w, r1);
    z2 = mju_dot3(w, r2);
    mju_addToScl3(r1, w, -z1);
    mju_addToScl3(r2, w, -z2);
  }

  // first basis vector u = normalize(r1)
  mjtNum u[3];
  mju_copy3(u, r1);
  if (mju_normalize3(u) < mjMINVAL) {
    mju_copy3(pts+0, p1);
    mju_copy3(pts+3, p2);
    return 2;
  }

  // incoming and outgoing tangent vectors: t1 = p1 - x1, t2 = x2 - p2
  mjtNum t1[3], t2[3], n1[3], n2[3], normal[3];
  mju_sub3(t1, p1, x1);
  mju_sub3(t2, x2, p2);
  mju_cross(n1, r1, t1);
  mju_cross(n2, r2, t2);
  mju_add3(normal, n1, n2);

  // determine orbit normal
  if (type == mjGEOM_CYLINDER) {
    mjtNum dot = mju_dot3(w, normal);
    if (mju_abs(dot) < mjMINVAL) {
      mju_cross(normal, r1, r2);
      dot = mju_dot3(w, normal);
    }
    mju_scl3(normal, w, dot >= 0 ? 1 : -1);
  } else {
    if (mju_normalize3(normal) < mjMINVAL) {
      mju_cross(normal, r1, r2);
      if (mju_normalize3(normal) < mjMINVAL) {
        mju_copy3(pts+0, p1);
        mju_copy3(pts+3, p2);
        return 2;
      }
    }
  }

  // second basis vector v = normalize(normal x u)
  mjtNum v[3];
  mju_cross(v, normal, u);
  if (mju_normalize3(v) < mjMINVAL) {
    mju_copy3(pts+0, p1);
    mju_copy3(pts+3, p2);
    return 2;
  }

  // wrap angle in [0, 2*pi)
  mjtNum angle = mju_atan2(mju_dot3(v, r2), mju_dot3(u, r2));
  if (angle < -1e-6) {
    angle += 2*mjPI;
  } else if (angle < 0) {
    angle = 0;
  }

  // number of segments proportional to wrap angle
  int nsegments = mjMIN(npts-1, mjMAX(1, mju_round((npts-1) * angle / mjPI)));

  // start point
  mju_copy3(pts+0, p1);

  // interior points along geodesic arc
  for (int k=1; k < nsegments; k++) {
    mjtNum t = (mjtNum)k / nsegments;
    mjtNum theta = t * angle;
    mju_copy3(pts+3*k, xpos);
    mju_addToScl3(pts+3*k, u, radius*mju_cos(theta));
    mju_addToScl3(pts+3*k, v, radius*mju_sin(theta));
    mju_addToScl3(pts+3*k, w, z1 + t*(z2 - z1));
  }

  // end point
  mju_copy3(pts+3*nsegments, p2);

  return nsegments + 1;
}



static void addSpatialTendonGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, int catmask,
                                  mjvScene* scn) {
  const int category = mjCAT_DYNAMIC;
  if (!(category & catmask)) {
    return;
  }
  if (!vopt->flags[mjVIS_TENDON]) {
    return;
  }

  for (int i=0; i < m->ntendon; i++) {
    if (!vopt->tendongroup[mjMAX(0, mjMIN(mjNGROUP-1, m->tendon_group[i]))]) {
      continue;
    }

    mjtNum length;
    int draw_catenary = mjv_isCatenary(m, d, i, &length);

    // conditions not met: draw straight lines or wrapped geodesics
    if (!draw_catenary) {
      for (int j=d->ten_wrapadr[i]; j < d->ten_wrapadr[i]+d->ten_wrapnum[i]-1; j++) {
        if (d->wrap_obj[j] != -2 && d->wrap_obj[j+1] != -2) {
          mjtNum pts[3*mjMAXCURVE];
          int npoints;
          if (d->wrap_obj[j] >= 0 && d->wrap_obj[j+1] >= 0) {
            int ngeodesic = mjMIN(m->vis.quality.numslices + 1, mjMAXCURVE);
            int gid = d->wrap_obj[j];
            npoints = mjv_geodesic(pts, ngeodesic, d->geom_xpos+3*gid, d->geom_xmat+9*gid,
                                   m->geom_size[3*gid], m->geom_type[gid], d->wrap_xpos+3*j-3);
          } else {
            mju_copy3(pts+0, d->wrap_xpos+3*j);
            mju_copy3(pts+3, d->wrap_xpos+3*j+3);
            npoints = 2;
          }

          for (int s=0; s < npoints-1; s++) {
            mjvGeom* thisgeom = acquireGeom(scn, i, category, mjOBJ_TENDON);
            if (!thisgeom) {
              return;
            }

            // construct geom
            mjv_connector(thisgeom, mjGEOM_CAPSULE, m->tendon_width[i], pts+3*s, pts+3*s+3);

            // set material properties
            int tendon_matid = m->tendon_matid[i];
            float rgba[4];
            f2f(rgba, m->tendon_rgba+4*i, 4);

            // if tendon has no material and the color is the default gray, re-color it using limit impedance
            if (tendon_matid == -1 && rgba[0] == 0.5 && rgba[1] == 0.5 && rgba[2] == 0.5 && rgba[3] == 1) {
              // loop over limit constraints, get impedance if this tendon is limited
              mjtNum imp = 0;
              int efc_start = d->ne + d->nf;
              int efc_end = efc_start + d->nl;
              for (int k=efc_start; k < efc_end; k++) {
                if (d->efc_type[k] == mjCNSTR_LIMIT_TENDON && d->efc_id[k] == i) {
                  imp = d->efc_KBIP[4*k + 2];
                }
              }

              // use impedance to mix tendon and constraint colors
              rgba[0] = (1-imp) * rgba[0] + imp * m->vis.rgba.constraint[0];
              rgba[1] = (1-imp) * rgba[1] + imp * m->vis.rgba.constraint[1];
              rgba[2] = (1-imp) * rgba[2] + imp * m->vis.rgba.constraint[2];
            }

            setMaterial(m, thisgeom, tendon_matid, rgba, vopt->flags);

            // override if visualizing islands
            if (vopt->flags[mjVIS_ISLAND]) {
              // strip material
              thisgeom->matid = -1;

              // set hue with first island dof, if constrained
              int h = -1;
              if (d->nisland && d->tendon_efcadr[i] >= 0) {
                h = d->island_dofadr[d->efc_island[d->tendon_efcadr[i]]];
              }
              islandColor(thisgeom->rgba, h, 1);
            }

            // vopt->label: only the first segment
            if (vopt->label == mjLABEL_TENDON && j == d->ten_wrapadr[i] && s == 0) {
              makeLabel(m, mjOBJ_TENDON, i, thisgeom->label);
            }

            releaseGeom(&thisgeom, scn);
          }
        }
      }
    }

    // special case handling of string-like tendons under gravity
    else {
      // two hanging points: x0, x1
      mjtNum x0[3], x1[3];
      mju_copy3(x0, d->wrap_xpos + 3*d->ten_wrapadr[i]);
      mju_copy3(x1, d->wrap_xpos + 3*d->ten_wrapadr[i] + 3);

      // get number of points along catenary path (capped at mjMAXCURVE)
      int ncatenary = mjMIN(m->vis.quality.numslices + 1, mjMAXCURVE);
      mjtNum catenary[3*mjMAXCURVE];

      // points along catenary path
      int npoints = mjv_catenary(x0, x1, m->opt.gravity, length, catenary, ncatenary);

      // draw npoints-1 segments
      for (int j=0; j < npoints-1; j++) {
        mjvGeom* thisgeom = acquireGeom(scn, i, category, mjOBJ_TENDON);
        if (!thisgeom) {
          return;
        }

        // construct geom
        mjv_connector(thisgeom, mjGEOM_CAPSULE, m->tendon_width[i], catenary+3*j, catenary+3*j+3);

        // set material if given
        setMaterial(m, thisgeom, m->tendon_matid[i], m->tendon_rgba+4*i, vopt->flags);

        // vopt->label: only the first segment
        if (vopt->label == mjLABEL_TENDON && npoints/2) {
          makeLabel(m, mjOBJ_TENDON, i, thisgeom->label);
        }

        releaseGeom(&thisgeom, scn);
      }
    }
  }
}


static void addSliderCrankGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, int catmask,
                                mjvScene* scn) {
  const int category = mjCAT_DYNAMIC;
  if (!(category & catmask)) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=0; i < m->nactuator; i++) {
    if (m->actuator_trntype[i] == mjTRN_SLIDERCRANK) {
      // get data
      int j = m->actuator_trnid[2*i];                 // crank
      int k = m->actuator_trnid[2*i+1];               // slider
      mjtNum rod = m->actuator_cranklength[i];
      mjtNum axis[3];
      axis[0] = d->site_xmat[9*k+2];
      axis[1] = d->site_xmat[9*k+5];
      axis[2] = d->site_xmat[9*k+8];

      // compute crank length
      mjtNum vec[3];
      mju_sub(vec, d->site_xpos+3*j, d->site_xpos+3*k, 3);
      mjtNum len = mju_dot3(vec, axis);
      mjtNum det = len*len + rod*rod - mju_dot3(vec, vec);
      mjtByte broken = 0;
      if (det < 0) {
        det = 0;
        broken = 1;
      }
      len = len - mju_sqrt(det);

      // compute slider endpoint
      mjtNum end[3];
      mju_scl3(end, axis, len);
      mju_addTo3(end, d->site_xpos+3*k);

      // render slider
      mjvGeom* thisgeom = acquireGeom(scn, i, category, mjOBJ_ACTUATOR);
      if (!thisgeom) {
        return;
      }

      mjv_connector(thisgeom, mjGEOM_CYLINDER, scl * m->vis.scale.slidercrank,
                    d->site_xpos+3*k, end);
      f2f(thisgeom->rgba, m->vis.rgba.slidercrank, 4);
      if (vopt->label == mjLABEL_ACTUATOR) {
        makeLabel(m, mjOBJ_ACTUATOR, i, thisgeom->label);
      }
      releaseGeom(&thisgeom, scn);

      thisgeom = acquireGeom(scn, i, category, mjOBJ_ACTUATOR);
      if (!thisgeom) {
        return;
      }

      mjv_connector(thisgeom, mjGEOM_CAPSULE, scl * m->vis.scale.slidercrank/2.0,
                    end, d->site_xpos+3*j);
      if (broken) {
        f2f(thisgeom->rgba, m->vis.rgba.crankbroken, 4);
      } else {
        f2f(thisgeom->rgba, m->vis.rgba.slidercrank, 4);
      }
      releaseGeom(&thisgeom, scn);
    }
  }
}


static void addGeomFrameGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, int catmask,
                              mjvScene* scn) {
  if (vopt->frame != mjFRAME_GEOM) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=0; i < m->ngeom; i++) {
    if (!(bodycategory(m, m->geom_bodyid[i]) & catmask)) {
      continue;
    }
    if (!vopt->geomgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->geom_group[i]))]) {
      continue;
    }

    // base element is invisible; don't show decors
    int matid = m->geom_matid[i];
    float* rgba = (matid >= 0) ? (m->mat_rgba + 4*matid) : (m->geom_rgba + 4*i);
    if (rgba[3] == 0) {
      continue;
    }

    // construct geom frame
    mjtNum width = m->vis.scale.framewidth * scl;
    mjtNum length = m->vis.scale.framelength * scl;
    addFrame(scn, i, d->geom_xpos+3*i, d->geom_xmat+9*i, length, width);
  }
}


static void addSiteFrameGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, int catmask,
                              mjvScene* scn) {
  if (vopt->frame != mjFRAME_SITE) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=0; i < m->nsite; i++) {
    if (!(bodycategory(m, m->site_bodyid[i]) & catmask)) {
      continue;
    }
    if (!vopt->sitegroup[mjMAX(0, mjMIN(mjNGROUP-1, m->site_group[i]))]) {
      continue;
    }

    // base element is invisible; don't show decors
    int matid = m->site_matid[i];
    float* rgba = (matid >= 0) ? (m->mat_rgba + 4*matid) : (m->site_rgba + 4*i);
    if (rgba[3] == 0) {
      continue;
    }

    // construct site frame
    mjtNum width = m->vis.scale.framewidth * scl;
    mjtNum length = m->vis.scale.framelength * scl;
    addFrame(scn, i, d->site_xpos+3*i, d->site_xmat+9*i, length, width);
  }
}


static void addBodyBvhGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_BODYBVH]) {
    return;
  }

  for (int i = 0; i < m->nbvhstatic; i++) {
    int isleaf = m->bvh_child[2*i] == -1 && m->bvh_child[2*i+1] == -1;
    if (m->bvh_depth[i] != vopt->bvh_depth) {
      if (!isleaf || m->bvh_depth[i] > vopt->bvh_depth) {
        continue;
      }
    }

    // find geom number
    int bodyid = 0;
    int geomid = m->bvh_nodeid[i];
    while (i >= m->body_bvhadr[bodyid] + m->body_bvhnum[bodyid]) {
      if (++bodyid >= m->nbody) {
        break;
      }
    }

    // stop after body bvh are finished
    if (bodyid >= m->nbody) {
      break;
    }

    // get xpos, xmat, size
    const mjtNum* xpos = isleaf ? d->geom_xpos + 3 * geomid : d->xipos + 3 * bodyid;
    const mjtNum* xmat = isleaf ? d->geom_xmat + 9 * geomid : d->ximat + 9 * bodyid;
    const mjtNum* size = isleaf ? m->geom_aabb + 6*geomid + 3 : m->bvh_aabb + 6*i + 3;

    // offset xpos with aabb center (not always at frame origin)
    const mjtNum* center = isleaf ? m->geom_aabb + 6*geomid : m->bvh_aabb + 6*i;
    mjtNum pos[3];
    mju_mulMatVec3(pos, xmat, center);
    mju_addTo3(pos, xpos);

    // set box color
    const float* rgba = m->vis.rgba.bv;
    if (m->vis.global.bvactive && d->bvh_active[i]) {
      rgba = m->vis.rgba.bvactive;
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    mjv_initGeom(thisgeom, mjGEOM_LINEBOX, size, pos, xmat, rgba);
    releaseGeom(&thisgeom, scn);
  }
}


static void addFlexBvhGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_MESHBVH]) {
    return;
  }
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_bvhnum[f] && vopt->flexgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->flex_group[f]))]) {
      for (int i=m->flex_bvhadr[f]; i < m->flex_bvhadr[f]+m->flex_bvhnum[f]; i++) {
        int isleaf = m->bvh_child[2*i] == -1 && m->bvh_child[2*i+1] == -1;
        if (m->bvh_depth[i] != vopt->bvh_depth) {
          if (!isleaf || m->bvh_depth[i] > vopt->bvh_depth) {
            continue;
          }
        }

        // get box data
        mjtNum* aabb = d->bvh_aabb_dyn + 6*(i - m->nbvhstatic);

        // set box color
        const float* rgba = m->vis.rgba.bv;
        if (m->vis.global.bvactive && d->bvh_active[i]) {
          rgba = m->vis.rgba.bvactive;
        }

        mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
        if (!thisgeom) {
          return;
        }
        mjv_initGeom(thisgeom, mjGEOM_LINEBOX, aabb+3, aabb, NULL, rgba);
        releaseGeom(&thisgeom, scn);
      }
    }

    if (!m->flex_interp[f]) {
      continue;
    }

    // control points box
    mjtNum* xpos = mjSTACKALLOC(d, 3*m->flex_nodenum[f], mjtNum);
    int nstart = m->flex_nodeadr[f];
    int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];
    if (m->flex_centered[f]) {
      for (int i=0; i < m->flex_nodenum[f]; i++) {
        mju_copy3(xpos + 3*i, d->xpos + 3*bodyid[i]);
      }
    } else {
      for (int i=0; i < m->flex_nodenum[f]; i++) {
        mju_mulMatVec3(xpos + 3*i, d->xmat + 9*bodyid[i], m->flex_node + 3*(i+nstart));
        mju_addTo3(xpos + 3*i, d->xpos + 3*bodyid[i]);
      }
    }

    int cx = m->flex_cellnum[3*f+0];
    int cy = m->flex_cellnum[3*f+1];
    int cz = m->flex_cellnum[3*f+2];
    int order = m->flex_interp[f];
    order = order < 0 ? -order : order;
    int NX = cx * order + 1;
    int NY = cy * order + 1;
    int NZ = cz * order + 1;

    int shell_mode = m->flex_interp[f] < 0;

    for (int i=0; i < NX; i++) {
      for (int j=0; j < NY; j++) {
        for (int k=0; k < NZ; k++) {
          int n0 = i*NY*NZ + j*NZ + k;

          // skip if this node is pinned (no joints on its body)
          if (m->body_jntnum[bodyid[n0]] == 0) {
            continue;
          }

          // shell mode: skip interior nodes entirely
          int is_boundary = (i == 0 || i == NX-1 ||
                             j == 0 || j == NY-1 ||
                             k == 0 || k == NZ-1);
          if (shell_mode && !is_boundary) {
            continue;
          }

          int offset  = 3*n0;
          int offset1 = 3*((i+1)*NY*NZ + j*NZ + k);
          int offset2 = 3*(i*NY*NZ + (j+1)*NZ + k);
          int offset3 = 3*(i*NY*NZ + j*NZ + (k+1));

          // edge along i: draw if neighbor is also on boundary (shell) or has joints
          if (i < NX-1 && m->body_jntnum[bodyid[(i+1)*NY*NZ + j*NZ + k]] > 0) {
            int nb_boundary = ((i+1) == 0 || (i+1) == NX-1 ||
                               j == 0 || j == NY-1 ||
                               k == 0 || k == NZ-1);
            if (!shell_mode || nb_boundary) {
              mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
              if (!thisgeom) {
                return;
              }
              mjv_connector(thisgeom, mjGEOM_LINE, 3, xpos+offset, xpos+offset1);
              releaseGeom(&thisgeom, scn);
            }
          }
          if (j < NY-1 && m->body_jntnum[bodyid[i*NY*NZ + (j+1)*NZ + k]] > 0) {
            int nb_boundary = (i == 0 || i == NX-1 ||
                               (j+1) == 0 || (j+1) == NY-1 ||
                               k == 0 || k == NZ-1);
            if (!shell_mode || nb_boundary) {
              mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
              if (!thisgeom) {
                return;
              }
              mjv_connector(thisgeom, mjGEOM_LINE, 3, xpos+offset, xpos+offset2);
              releaseGeom(&thisgeom, scn);
            }
          }
          if (k < NZ-1 && m->body_jntnum[bodyid[i*NY*NZ + j*NZ + (k+1)]] > 0) {
            int nb_boundary = (i == 0 || i == NX-1 ||
                               j == 0 || j == NY-1 ||
                               (k+1) == 0 || (k+1) == NZ-1);
            if (!shell_mode || nb_boundary) {
              mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
              if (!thisgeom) {
                return;
              }
              mjv_connector(thisgeom, mjGEOM_LINE, 3, xpos+offset, xpos+offset3);
              releaseGeom(&thisgeom, scn);
            }
          }
        }
      }
    }
  }
}


static void addMeshBvhGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_MESHBVH]) {
    return;
  }
  for (int geomid = 0; geomid < m->ngeom; geomid++) {
    int meshid = m->geom_dataid[geomid];
    // skip if not a mesh or if there is an octree
    if (meshid == -1 || m->geom_type[geomid] == mjGEOM_SDF || m->mesh_octadr[meshid] >= 0) {
      continue;
    }

    for (int b = 0; b < m->mesh_bvhnum[meshid]; b++) {
      int i = b + m->mesh_bvhadr[meshid];
      int isleaf = m->bvh_child[2*i] == -1 && m->bvh_child[2*i+1] == -1;
      if (m->bvh_depth[i] != vopt->bvh_depth) {
        if (!isleaf || m->bvh_depth[i] > vopt->bvh_depth) {
          continue;
        }
      }

      // box color
      const float* rgba = m->vis.rgba.bv;
      if (m->vis.global.bvactive) {
        if (d->bvh_active[i]) {
          rgba = m->vis.rgba.bvactive;
        } else {
          // when marking active bvs, skip inactive volumes
          continue;
        }
      }

      // get xpos, xmat, size
      const mjtNum* xpos = d->geom_xpos + 3 * geomid;
      const mjtNum* xmat = d->geom_xmat + 9 * geomid;
      const mjtNum* size = m->bvh_aabb + 6*i + 3;

      // offset xpos with aabb center (not always at geom origin)
      const mjtNum* center = m->bvh_aabb + 6*i;
      mjtNum pos[3];
      mju_mulMatVec3(pos, xmat, center);
      mju_addTo3(pos, xpos);

      mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
      if (!thisgeom) {
        return;
      }
      mjv_initGeom(thisgeom, mjGEOM_LINEBOX, size, pos, xmat, rgba);
      releaseGeom(&thisgeom, scn);
    }
  }
}


static void addMeshOctreeGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_MESHBVH]) {
    return;
  }
  for (int geomid = 0; geomid < m->ngeom; geomid++) {
    int meshid = m->geom_dataid[geomid];
    if (meshid == -1 || m->geom_type[geomid] == mjGEOM_HFIELD || m->mesh_octadr[meshid] == -1) {
      continue;
    }

    for (int b = 0; b < m->mesh_octnum[meshid]; b++) {
      int i = b + m->mesh_octadr[meshid];
      if (m->oct_depth[i] != vopt->bvh_depth) {
        continue;
      }

      mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
      if (!thisgeom) {
        return;
      }

      const float* rgba = m->vis.rgba.bv;
      const mjtNum* xpos = d->geom_xpos + 3 * geomid;
      const mjtNum* xmat = d->geom_xmat + 9 * geomid;
      const mjtNum* size = m->oct_aabb + 6*i + 3;

      // offset xpos with aabb center (not always at geom origin)
      const mjtNum* center = m->oct_aabb + 6*i;
      mjtNum pos[3];
      mju_mulMatVec3(pos, xmat, center);
      mju_addTo3(pos, xpos);

      mjv_initGeom(thisgeom, mjGEOM_LINEBOX, size, pos, xmat, rgba);
      releaseGeom(&thisgeom, scn);
    }
  }
}


static void addTactileSensorGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                  mjvScene* scn) {
  if (!vopt->flags[mjVIS_CONTACTPOINT]) {
    return;
  }
  for (int id = 0; id < m->nsensor; id++) {
    if (m->sensor_type[id] == mjSENS_TACTILE) {
      // get site id and frame
      int mesh_id = m->sensor_objid[id];
      int geom_id = m->sensor_refid[id];
      mjtNum* geom_pos = d->geom_xpos + 3*geom_id;
      mjtNum* geom_mat = d->geom_xmat + 9*geom_id;

      // get sensor data
      mjtNum* sensordata = d->sensordata + m->sensor_adr[id];
      int nchannel = m->sensor_dim[id] / m->mesh_vertnum[mesh_id];

      // get maximum absolute normal force
      mjtNum maxval = 0;
      for (int j=0; j < m->mesh_vertnum[mesh_id]; j++) {
        maxval = mju_max(maxval, mju_abs(sensordata[j]));
      }

      // if no normal force readings, quick return
      if (!maxval || m->geom_rbound[geom_id] < mjMINVAL) {
        continue;
      }

      // draw geoms
      float* mesh_vert = m->mesh_vert + 3*m->mesh_vertadr[mesh_id];
      int* face = m->mesh_face + 3*m->mesh_faceadr[mesh_id];
      for (int i=0; i < m->mesh_facenum[mesh_id]; i++) {
        // triangle in global frame
        mjtNum pos[3][3];
        for (int j = 0; j < 3; j++) {
          mjtNum v[3] = {mesh_vert[3 * face[3 * i + j] + 0],
                          mesh_vert[3 * face[3 * i + j] + 1],
                          mesh_vert[3 * face[3 * i + j] + 2]};
          mju_mulMatVec3(pos[j], geom_mat, v);
          mju_addTo3(pos[j], geom_pos);
        }

        // color
        float rgba[4] = {0, 0, 0, 1.0};
        mjtNum nval[3] = {0, 0, 0};
        for (int r = 0; r < mjMIN(nchannel, 3); r++) {
          for (int j = 0; j < 3; j++) {
            mjtNum val = sensordata[r*m->mesh_vertnum[mesh_id] + face[3*i+j]];
            rgba[r] += mju_abs(val) / maxval;
            if (val) {
              nval[r] += 1;
            }
          }
          if (nval[r]) {
            rgba[r] /= nval[r];
          }
        }

        if (rgba[0]==0 && rgba[1]==0 && rgba[2]==0) {
          rgba[3] = .1;
        }

        // draw triangles, one per side
        addTriangle(scn, pos[0], pos[1], pos[2], rgba, id, mjCAT_DECOR, mjOBJ_SENSOR);
      }
    }
  }
}


static void addInertiaGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                            const mjvPerturb* pert, int catmask, mjvScene* scn) {
  if (!vopt->flags[mjVIS_INERTIA]) {
    return;
  }
  int ellipsoid = m->vis.global.ellipsoidinertia == 1;
  for (int i=1; i < m->nbody; i++) {
    if (m->body_mass[i] <= mjMINVAL) {
      continue;
    }
    if (!(bodycategory(m, i) & catmask)) {
      continue;
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_BODY);
    if (!thisgeom) {
      return;
    }

    mjtNum Ixx = m->body_inertia[3*i+0];
    mjtNum Iyy = m->body_inertia[3*i+1];
    mjtNum Izz = m->body_inertia[3*i+2];
    mjtNum mass = m->body_mass[i];
    mjtNum scale_inertia = ellipsoid ? mju_sqrt(5) : mju_sqrt(3);

    mjtNum sz[3];
    sz[0] = mju_sqrt((Iyy + Izz - Ixx) / (2 * mass)) * scale_inertia;
    sz[1] = mju_sqrt((Ixx + Izz - Iyy) / (2 * mass)) * scale_inertia;
    sz[2] = mju_sqrt((Ixx + Iyy - Izz) / (2 * mass)) * scale_inertia;

    // scale with mass if enabled
    if (vopt->flags[mjVIS_SCLINERTIA]) {
      // density = mass / volume
      mjtNum scale_volume = ellipsoid ? 4.0/3.0*mjPI : 8.0;
      mjtNum volume = scale_volume * sz[0]*sz[1]*sz[2];
      mjtNum density = mass / mju_max(mjMINVAL, volume);

      // scale = root3(density)
      mjtNum scale = mju_pow(density*0.001, 1.0/3.0);

      // scale sizes, so that box/ellipsoid with density of 1000 has same mass
      sz[0] *= scale;
      sz[1] *= scale;
      sz[2] *= scale;
    }

    // construct geom
    mjtGeom type = ellipsoid ? mjGEOM_ELLIPSOID : mjGEOM_BOX;
    mjv_initGeom(thisgeom, type, sz, d->xipos+3*i, d->ximat+9*i, m->vis.rgba.inertia);

    // glow
    if (pert->select == i) {
      markselected(&m->vis, thisgeom);
    }

    // vopt->label
    if (vopt->label == mjLABEL_BODY ||
        (vopt->label == mjLABEL_SELECTION && pert->select == i)) {
      makeLabel(m, mjOBJ_BODY, i, thisgeom->label);
    }

    releaseGeom(&thisgeom, scn);
  }
}


static void addPerturbGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                            const mjvPerturb* pert, mjvScene* scn) {
  if (!vopt->flags[mjVIS_PERTOBJ]) {
    return;
  }
  if (pert->select <= 0) {
    return;
  }

  const float scl = m->stat.meansize;
  if ((pert->active | pert->active2) & mjPERT_TRANSLATE) {
    mjvGeom* thisgeom = acquireGeom(scn, pert->select, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    // compute selection point in world coordinates
    mjtNum selpos[3];
    mju_mulMatVec3(selpos, d->xmat+9*pert->select, pert->localpos);
    mju_addTo3(selpos, d->xpos+3*pert->select);

    // construct geom
    mjtNum sz[3];
    sz[0] = scl * m->vis.scale.constraint;
    mjv_connector(thisgeom, mjGEOM_CAPSULE, sz[0], selpos, pert->refselpos);

    // prepare color
    float rgba[4];
    mixcolor(rgba, m->vis.rgba.constraint,
              (pert->active & mjPERT_TRANSLATE) > 0,
              (pert->active2 & mjPERT_TRANSLATE) > 0);

    f2f(thisgeom->rgba, rgba, 4);

    releaseGeom(&thisgeom, scn);

    // add small sphere at end-effector
    thisgeom = acquireGeom(scn, pert->select, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    // construct geom
    sz[0] = 2*sz[0];
    sz[1] = sz[2] = sz[0];

    mjtNum mat[9];
    mju_quat2Mat(mat, pert->refquat);
    mjv_initGeom(thisgeom, mjGEOM_SPHERE, sz, pert->refselpos, mat, rgba);

    releaseGeom(&thisgeom, scn);
  }

  if ((pert->active | pert->active2) & mjPERT_ROTATE) {
    mjvGeom* thisgeom = acquireGeom(scn, pert->select, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    // prepare color, use inertia color
    float rgba[4];
    mixcolor(rgba, m->vis.rgba.inertia,
              (pert->active & mjPERT_ROTATE) > 0,
              (pert->active2 & mjPERT_ROTATE) > 0);

    // construct geom: if body i has a collision aabb, use that
    mjtNum pos[3] = {0};
    mjtNum sz[3];
    if (m->body_bvhnum[pert->select]) {
      mjtNum* aabb = m->bvh_aabb+6*m->body_bvhadr[pert->select];
      mju_copy3(sz, aabb+3);
      mju_mulMatVec3(pos, d->ximat+9*pert->select, aabb);
    }

    // otherwise box of size meansize
    else {
      sz[0] = sz[1] = sz[2] = scl;
    }

    mjtNum mat[9];
    mju_quat2Mat(mat, pert->refquat);
    mju_addTo3(pos, d->xipos+3*pert->select);
    mjv_initGeom(thisgeom, mjGEOM_LINEBOX, sz, pos, mat, rgba);

    releaseGeom(&thisgeom, scn);
  }
}


static void addWorldBodyFrameGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, int catmask,
                                   mjvScene* scn) {
  const float scl = m->stat.meansize;
  for (int i = (vopt->frame == mjFRAME_WORLD ? 0 : 1);
        i < (vopt->frame == mjFRAME_BODY ? m->nbody : 1);
        i++) {
    // skip if body is static and static bodies are masked
    if (i > 0 && bodycategory(m, i) & ~catmask) {
      continue;
    }

    // set length(1) and width(0) of the axis cylinders
    mjtNum sz[2];
    if (i == 0) {
      sz[1] = m->vis.scale.framelength * scl * 2;
      sz[0] = m->vis.scale.framewidth * scl * 2;
    } else {
      sz[1] = m->vis.scale.framelength * scl;
      sz[0] = m->vis.scale.framewidth * scl;
    }

    mjtNum* xmat = vopt->flags[mjVIS_INERTIA] ? d->ximat+9*i : d->xmat+9*i;
    mjtNum* xpos = vopt->flags[mjVIS_INERTIA] ? d->xipos+3*i : d->xpos+3*i;
    addFrame(scn, i, xpos, xmat, sz[1], sz[0]);
  }
}


static void addSelectionPointGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                   const mjvPerturb* pert, mjvScene* scn) {
  if (pert->select <= 0) {
    return;
  }
  if (!vopt->flags[mjVIS_SELECT]) {
    return;
  }

  const float scl = m->stat.meansize;

  // compute selection point in world coordinates
  mjtNum selpos[3];
  mju_mulMatVec3(selpos, d->xmat+9*pert->select, pert->localpos);
  mju_addTo3(selpos, d->xpos+3*pert->select);

  mjvGeom* thisgeom = acquireGeom(scn, pert->select, mjCAT_DECOR, mjOBJ_UNKNOWN);
  if (!thisgeom) {
    return;
  }

  thisgeom->type = mjGEOM_SPHERE;
  thisgeom->size[0] = thisgeom->size[1] = thisgeom->size[2] = scl * m->vis.scale.selectpoint;
  mju_n2f(thisgeom->pos, selpos, 3);
  mju_n2f(thisgeom->mat, IDENTITY, 9);
  f2f(thisgeom->rgba, m->vis.rgba.selectpoint, 4);
  if (vopt->label == mjLABEL_SELPNT) {
    mjSNPRINTF(
      thisgeom->label, "%.3f %.3f %.3f (local %.3f %.3f %.3f)",
      selpos[0], selpos[1], selpos[2],
      pert->localpos[0], pert->localpos[1], pert->localpos[2]);
  }
  releaseGeom(&thisgeom, scn);
}


static void addBodyLabelGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                              const mjvPerturb* pert, int catmask, mjvScene* scn) {
  if (vopt->flags[mjVIS_INERTIA]) {
    return;
  }
  if (vopt->label != mjLABEL_SELECTION && vopt->label != mjLABEL_BODY) {
    return;
  }
  for (int i=1; i < m->nbody; i++) {
    if (vopt->label == mjLABEL_SELECTION && pert->select != i) {
      continue;
    }
    if (bodycategory(m, i) & ~catmask) {
      continue;
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    thisgeom->type = mjGEOM_LABEL;
    mju_n2f(thisgeom->pos, d->xpos+3*i, 3);
    mju_n2f(thisgeom->mat, d->xmat+9*i, 9);
    makeLabel(m, mjOBJ_BODY, i, thisgeom->label);
    releaseGeom(&thisgeom, scn);
  }
}


static void addJointGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_JOINT]) {
    return;
  }
  const float scl = m->stat.meansize;
  for (int i=0; i < m->njnt; i++) {
    if (!vopt->jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->jnt_group[i]))]) {
      continue;
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_JOINT);
    if (!thisgeom) {
      return;
    }

    // set sz = {width, length} of the connectors
    mjtNum sz[2];
    sz[1] = m->vis.scale.jointlength * scl;
    sz[0] = m->vis.scale.jointwidth * scl;

    // set type, size, pos, mat depending on joint type
    int j = m->jnt_bodyid[i];
    mjtNum* from;
    mjtNum to[3];
    switch ((mjtJoint) m->jnt_type[i]) {
    case mjJNT_FREE:
      thisgeom->type = mjGEOM_BOX;
      thisgeom->size[0] = thisgeom->size[1] = thisgeom->size[2] = 0.3*sz[1];
      mju_n2f(thisgeom->pos, d->xanchor+3*i, 3);
      mju_n2f(thisgeom->mat, d->xmat+9*j, 9);
      break;

    case mjJNT_BALL:
      thisgeom->type = mjGEOM_SPHERE;
      thisgeom->size[0] = thisgeom->size[1] = thisgeom->size[2] = 0.3*sz[1];
      mju_n2f(thisgeom->pos, d->xanchor+3*i, 3);
      mju_n2f(thisgeom->mat, d->xmat+9*j, 9);
      break;

    case mjJNT_SLIDE:
    case mjJNT_HINGE:
      from = d->xanchor+3*i;
      mju_addScl3(to, from, d->xaxis+3*i, sz[1]);
      mjv_connector(thisgeom, m->jnt_type[i] == mjJNT_SLIDE ? mjGEOM_ARROW : mjGEOM_ARROW1,
                    sz[0], from, to);
      break;

    default:
      mjERROR("unknown joint type %d", m->jnt_type[i]);
    }

    // loop over limit constraints, get impedance if this joint is limited
    mjtNum imp = 0;
    int efc_start = d->ne + d->nf;
    int efc_end = efc_start + d->nl;
    for (int k=efc_start; k < efc_end; k++) {
      if (d->efc_type[k] == mjCNSTR_LIMIT_JOINT && d->efc_id[k] == i) {
        imp = d->efc_KBIP[4*k + 2];
      }
    }

    // use impedance to mix joint and constraint colors
    float rgba[4];
    rgba[0] = (1-imp) * m->vis.rgba.joint[0] + imp * m->vis.rgba.constraint[0];
    rgba[1] = (1-imp) * m->vis.rgba.joint[1] + imp * m->vis.rgba.constraint[1];
    rgba[2] = (1-imp) * m->vis.rgba.joint[2] + imp * m->vis.rgba.constraint[2];
    rgba[3] = 1;

    f2f(thisgeom->rgba, rgba, 4);

    // vopt->label
    if (vopt->label == mjLABEL_JOINT) {
      makeLabel(m, mjOBJ_JOINT, i, thisgeom->label);
    }

    releaseGeom(&thisgeom, scn);
  }
}


static void addActuatorGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_ACTUATOR]) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=0; i < m->nactuator; i++) {
    if (!vopt->actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->actuator_group[i]))]) {
      continue;
    }
    if (mj_actuatorDisabled(m, i)) {
      continue;
    }

    // determine extended range
    mjtNum rng[3] = {-1, 0, +1};
    mjtNum rmin = -1, rmax = 1, act = 0;
    if (m->actuator_ctrllimited[m->actuator_ctrladr[i]]) {
      rmin = m->actuator_ctrlrange[2*m->actuator_ctrladr[i]];
      rmax = m->actuator_ctrlrange[2*m->actuator_ctrladr[i]+1];
    } else if (vopt->flags[mjVIS_ACTIVATION] && m->actuator_actlimited[i]) {
      rmin = m->actuator_actrange[2*i];
      rmax = m->actuator_actrange[2*i+1];
    }
    if (rmin >= 0) {
      rng[0] = -1;
      rng[1] = rmin;
      rng[2] = rmax;
    } else if (rmax <= 0) {
      rng[0] = rmin;
      rng[1] = rmax;
      rng[2] = +1;
    } else {
      rng[0] = rmin;
      rng[1] = 0;
      rng[2] = rmax;
    }

    // adjust small ranges
    if (rng[1]-rng[0] < mjMINVAL) {
      rng[0] = rng[1] - mjMINVAL;
    }
    if (rng[2]-rng[1] < mjMINVAL) {
      rng[2] = rng[1] + mjMINVAL;
    }

    // clamp act to extended range
    if (vopt->flags[mjVIS_ACTIVATION] && m->actuator_dyntype[i]) {
      act = mju_clip(d->act[m->actuator_actadr[i] + m->actuator_actnum[i] - 1], rng[0], rng[2]);
    } else {
      act = mju_clip(d->ctrl[m->actuator_ctrladr[i]], rng[0], rng[2]);
    }

    // compute interpolants
    float amin, amean, amax;
    if (act <= rng[1]) {
      amin = (rng[1]-act) / mjMAX(mjMINVAL, rng[1]-rng[0]);
      amean = 1 - amin;
      amax = 0;
    } else {
      amax = (act-rng[1]) / mjMAX(mjMINVAL, rng[2]-rng[1]);
      amean = 1 - amax;
      amin = 0;
    }

    // interpolated color
    float rgba[4];
    for (int j=0; j < 4; j++) {
      rgba[j] = amin*m->vis.rgba.actuatornegative[j] +
                amean*m->vis.rgba.actuator[j] +
                amax*m->vis.rgba.actuatorpositive[j];
    }

    // get transmission object id
    int j = m->actuator_trnid[2*i];

    // slide and hinge joint actuators
    if (m->actuator_trntype[i] == mjTRN_JOINT ||
        m->actuator_trntype[i] == mjTRN_JOINTINPARENT ||
        m->actuator_trntype[i] == mjTRN_SITE) {
      mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_ACTUATOR);
      if (!thisgeom) {
        return;
      }

      // site actuators
      mjtNum sz[3];
      if (m->actuator_trntype[i] == mjTRN_SITE) {
        // inflate sizes by 5%
        mju_scl3(sz, m->site_size+3*j, 1.05);

        const mjtNum* pos = d->site_xpos+3*j;
        const mjtNum* mat = d->site_xmat+9*j;
        mjtNum meshpos[3], meshmat[9];
        if (m->site_type[j] == mjGEOM_MESH && m->site_dataid[j] >= 0) {
          int meshid = m->site_dataid[j];
          mjtNum xquat[4];
          mju_mulMatVec3(meshpos, mat, m->mesh_pos+3*meshid);
          mju_addTo3(meshpos, pos);
          mju_mulQuat(xquat, d->site_xquat+4*j, m->mesh_quat+4*meshid);
          mju_quat2Mat(meshmat, xquat);
          pos = meshpos;
          mat = meshmat;
        }

        // make geom
        mjv_initGeom(thisgeom, m->site_type[j], sz, pos, mat, thisgeom->rgba);
        if (m->site_type[j] == mjGEOM_MESH && m->site_dataid[j] >= 0) {
          thisgeom->dataid = 2 * m->site_dataid[j];
          if (m->mesh_graphadr[m->site_dataid[j]] >= 0 && vopt->flags[mjVIS_CONVEXHULL]) {
            thisgeom->dataid += 1;
          }
        }
      } else if (m->jnt_type[j] == mjJNT_HINGE || m->jnt_type[j] == mjJNT_SLIDE) {
        // set length(1) and width(0) of the connectors
        sz[1] = m->vis.scale.actuatorlength * scl;
        sz[0] = m->vis.scale.actuatorwidth * scl;

        // make geom
        mjtNum* from = d->xanchor + 3*j;
        mjtNum to[3];
        mju_addScl3(to, from, d->xaxis+3*j, sz[1]);
        mjv_connector(thisgeom, m->jnt_type[j] == mjJNT_SLIDE ? mjGEOM_ARROW : mjGEOM_ARROW1,
                      sz[0], from, to);
      }

      // ball or free joint
      else if (m->jnt_type[j] == mjJNT_BALL || m->jnt_type[j] == mjJNT_FREE) {
        sz[0] = sz[1] = sz[2] = m->vis.scale.jointlength * scl * 0.33;

        // make geom
        mjv_initGeom(thisgeom,
                      m->jnt_type[j] == mjJNT_BALL ? mjGEOM_SPHERE : mjGEOM_BOX, sz,
                      d->xanchor + 3*j,
                      d->xmat + 9*m->jnt_bodyid[j],
                      thisgeom->rgba);
      }

      // set interpolated color
      f2f(thisgeom->rgba, rgba, 4);

      // vopt->label
      if (vopt->label == mjLABEL_ACTUATOR) {
        makeLabel(m, mjOBJ_ACTUATOR, i, thisgeom->label);
      }

      releaseGeom(&thisgeom, scn);
    }

    // body actuators
    else if (m->actuator_trntype[i] == mjTRN_BODY) {
      // iterate over body's geoms
      int geomnum = m->body_geomnum[j];
      int geomadr = m->body_geomadr[j];
      for (int k=geomadr; k < geomadr+geomnum; k++) {
        int geomtype = m->geom_type[k];
        // add inflated geom if it is a regular primitive
        if (geomtype != mjGEOM_PLANE && geomtype != mjGEOM_HFIELD &&
            geomtype != mjGEOM_MESH  && geomtype != mjGEOM_SDF) {
          mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_ACTUATOR);
          if (!thisgeom) {
            return;
          }

          // inflate sizes by 5%
          mjtNum sz[3];
          mju_scl3(sz, m->geom_size+3*k, 1.05);

          // make geom
          mjv_initGeom(thisgeom,
                        m->geom_type[k], sz,
                        d->geom_xpos + 3*k,
                        d->geom_xmat + 9*k,
                        thisgeom->rgba);

          // set interpolated color
          f2f(thisgeom->rgba, rgba, 4);

          releaseGeom(&thisgeom, scn);
        }
      }
    }

    // spatial tendon actuators
    else if (m->actuator_trntype[i] == mjTRN_TENDON && d->ten_wrapnum[j]) {
      for (int k=d->ten_wrapadr[j]; k < d->ten_wrapadr[j]+d->ten_wrapnum[j]-1; k++) {
        if (d->wrap_obj[k] != -2 && d->wrap_obj[k+1] != -2) {
          mjtNum pts[3*mjMAXCURVE];
          int npoints;
          if (d->wrap_obj[k] >= 0 && d->wrap_obj[k+1] >= 0) {
            int ngeodesic = mjMIN(m->vis.quality.numslices + 1, mjMAXCURVE);
            int gid = d->wrap_obj[k];
            npoints = mjv_geodesic(pts, ngeodesic, d->geom_xpos+3*gid, d->geom_xmat+9*gid,
                                   m->geom_size[3*gid], m->geom_type[gid], d->wrap_xpos+3*k-3);
          } else {
            mju_copy3(pts+0, d->wrap_xpos+3*k);
            mju_copy3(pts+3, d->wrap_xpos+3*k+3);
            npoints = 2;
          }

          mjtNum width = m->tendon_width[j] * m->vis.map.actuatortendon;
          for (int s=0; s < npoints-1; s++) {
            mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_ACTUATOR);
            if (!thisgeom) {
              return;
            }

            // construct geom
            mjv_connector(thisgeom, mjGEOM_CAPSULE, width, pts+3*s, pts+3*s+3);

            // set material if given
            setMaterial(m, thisgeom, m->tendon_matid[j], m->tendon_rgba+4*j, vopt->flags);

            // set interpolated color
            f2f(thisgeom->rgba, rgba, 4);

            // vopt->label: only the first segment
            if (vopt->label == mjLABEL_ACTUATOR && k == d->ten_wrapadr[j] && s == 0) {
              makeLabel(m, mjOBJ_ACTUATOR, i, thisgeom->label);
            }

            releaseGeom(&thisgeom, scn);
          }
        }
      }
    }
  }
}


static void addIslandLabelGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                mjvScene* scn) {
  if (vopt->label != mjLABEL_ISLAND || !d->nisland) {
    return;
  }
  for (int i=1; i < m->nbody; i++) {
    int weld_id = m->body_weldid[i];
    if (!m->body_dofnum[weld_id]) {
      continue;
    }
    int islandid = d->dof_island[m->body_dofadr[weld_id]];
    if (islandid <= -1) {
      continue;
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
    if (!thisgeom) {
      return;
    }

    thisgeom->type = mjGEOM_LABEL;
    mju_n2f(thisgeom->pos, d->xipos+3*i, 3);
    mju_n2f(thisgeom->mat, d->ximat+9*i, 9);
    mjSNPRINTF(thisgeom->label, "%d", islandid);

    releaseGeom(&thisgeom, scn);
  }
}


static void addCameraGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_CAMERA]) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=0; i < m->ncam; i++) {
    // copy camera rgba
    float cam_rgba[4];
    f2f(cam_rgba, m->vis.rgba.camera, 4);

    // draw frustum if resolution larger than (1, 1)
    if (m->cam_resolution[2*i] > 1 || m->cam_resolution[2*i+1] > 1) {
      // when drawing frustum, make camera translucent
      cam_rgba[3] = 0.3;

      // locals
      const float* rgba = m->vis.rgba.frustum;
      mjtNum vnear[4][3], vfar[4][3];
      mjtNum center[3];
      mjtNum znear = m->vis.map.znear * m->stat.extent;
      mjtNum zfar = m->vis.scale.frustum * scl;
      float zver[2], zhor[2];
      int orthographic = m->cam_projection[i] == mjPROJ_ORTHOGRAPHIC;

      // get frustum
      if (orthographic) {
        float aspect = (float)m->cam_resolution[2*i] / m->cam_resolution[2*i+1];
        zver[0] = zver[1] = m->cam_fovy[i] / 2;
        zhor[0] = zhor[1] = m->cam_fovy[i] * aspect / 2;
      } else if (m->cam_sensorsize[2*i] && m->cam_sensorsize[2*i+1]) {
        // intrinsic-based perspective camera
        getFrustum(zver, zhor, znear, m->cam_intrinsic+4*i, m->cam_sensorsize+2*i);
      } else {
        // fovy-based perspective camera
        float aspect = (float)m->cam_resolution[2*i] / m->cam_resolution[2*i+1];
        zver[0] = zver[1] = znear * mju_tan(m->cam_fovy[i] * mjPI / 360.0);
        zhor[0] = zhor[1] = zver[0] * aspect;
      }

      // frustum frame to convert from planes to vertex representation
      mjtNum* cam_xpos = d->cam_xpos+3*i;
      mjtNum* cam_xmat = d->cam_xmat+9*i;
      mjtNum x[] = {cam_xmat[0], cam_xmat[3], cam_xmat[6]};
      mjtNum y[] = {cam_xmat[1], cam_xmat[4], cam_xmat[7]};
      mjtNum z[] = {cam_xmat[2], cam_xmat[5], cam_xmat[8]};

      // vertices of the near plane
      mju_addScl3(center, cam_xpos, z, -znear);
      mju_addScl3(vnear[0], center, x, -zhor[0]);
      mju_addScl3(vnear[1], center, x,  zhor[1]);
      mju_addScl3(vnear[2], center, x,  zhor[1]);
      mju_addScl3(vnear[3], center, x, -zhor[0]);
      mju_addToScl3(vnear[0], y, -zver[0]);
      mju_addToScl3(vnear[1], y, -zver[0]);
      mju_addToScl3(vnear[2], y,  zver[1]);
      mju_addToScl3(vnear[3], y,  zver[1]);

      // vertices of the far plane: scale for perspective, average(width, height) for orthographic
      if (!orthographic) {
        zhor[0] *= zfar / znear;
        zhor[1] *= zfar / znear;
        zver[0] *= zfar / znear;
        zver[1] *= zfar / znear;
      } else {
        zfar = (zhor[0] + zver[0]) / 2;
      }
      mju_addScl3(center, cam_xpos, z, -zfar);
      mju_addScl3(vfar[0], center, x, -zhor[0]);
      mju_addScl3(vfar[1], center, x,  zhor[1]);
      mju_addScl3(vfar[2], center, x,  zhor[1]);
      mju_addScl3(vfar[3], center, x, -zhor[0]);
      mju_addToScl3(vfar[0], y, -zver[0]);
      mju_addToScl3(vfar[1], y, -zver[0]);
      mju_addToScl3(vfar[2], y,  zver[1]);
      mju_addToScl3(vfar[3], y,  zver[1]);

      // triangulation and wireframe of the frustum
      for (int e=0; e < 4; e++) {
        addTriangle(scn, vnear[e], vfar[e], vnear[(e+1)%4], rgba,
                    i, mjCAT_DECOR, mjOBJ_CAMERA);
        addTriangle(scn, vfar[e], vfar[(e+1)%4], vnear[(e+1)%4], rgba,
                    i, mjCAT_DECOR, mjOBJ_CAMERA);
        addConnector(scn, mjGEOM_LINE, 3, vnear[e], vnear[(e+1)%4], rgba,
                     i, mjCAT_DECOR, mjOBJ_CAMERA);
        addConnector(scn, mjGEOM_LINE, 3, vfar[e], vfar[(e+1)%4], rgba,
                     i, mjCAT_DECOR, mjOBJ_CAMERA);
        addConnector(scn, mjGEOM_LINE, 3, vnear[e], vfar[e], rgba,
                     i, mjCAT_DECOR, mjOBJ_CAMERA);
      }
    }

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_CAMERA);
    if (!thisgeom) {
      return;
    }

    // construct geom: camera body
    thisgeom->type = mjGEOM_BOX;
    thisgeom->size[0] = scl * m->vis.scale.camera * 1.0;
    thisgeom->size[1] = scl * m->vis.scale.camera * 0.8;
    thisgeom->size[2] = scl * m->vis.scale.camera * 0.4;
    mju_n2f(thisgeom->pos, d->cam_xpos+3*i, 3);
    mju_n2f(thisgeom->mat, d->cam_xmat+9*i, 9);
    f2f(thisgeom->rgba, cam_rgba, 4);

    // vopt->label
    if (vopt->label == mjLABEL_CAMERA) {
      makeLabel(m, mjOBJ_CAMERA, i, thisgeom->label);
    }

    releaseGeom(&thisgeom, scn);

    thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_CAMERA);
    if (!thisgeom) {
      return;
    }

    // construct geom: lens
    thisgeom->pos[0] = (float)(d->cam_xpos[3*i] -
                                scl*m->vis.scale.camera*0.6 * d->cam_xmat[9*i+2]);
    thisgeom->pos[1] = (float)(d->cam_xpos[3*i+1] -
                                scl*m->vis.scale.camera*0.6 * d->cam_xmat[9*i+5]);
    thisgeom->pos[2] = (float)(d->cam_xpos[3*i+2] -
                                scl*m->vis.scale.camera*0.6 * d->cam_xmat[9*i+8]);
    thisgeom->type = mjGEOM_CYLINDER;
    thisgeom->size[0] = scl * m->vis.scale.camera * 0.4;
    thisgeom->size[1] = scl * m->vis.scale.camera * 0.4;
    thisgeom->size[2] = scl * m->vis.scale.camera * 0.3;
    mju_n2f(thisgeom->mat, d->cam_xmat+9*i, 9);
    f2f(thisgeom->rgba, cam_rgba, 4);
    for (int k=0; k < 3; k++) {
      thisgeom->rgba[k] *= 0.5;  // make lens body darker
    }

    releaseGeom(&thisgeom, scn);

    if (vopt->frame != mjFRAME_CAMERA) {
      continue;
    }
    mjtNum width = m->vis.scale.framewidth * scl;
    mjtNum length = m->vis.scale.framelength * scl;
    addFrame(scn, i, d->cam_xpos+3*i, d->cam_xmat+9*i, length, width);
  }
}


static void addLightGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_LIGHT]) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=0; i < m->nlight; i++) {
    // make light frame
    mjtNum quat[4];
    mju_quatZ2Vec(quat, d->light_xdir+3*i);

    mjtNum mat[9];
    mju_quat2Mat(mat, quat);

    // make light position: offset backward, to avoid casting shadow
    mjtNum vec[3];
    mju_addScl3(vec, d->light_xpos+3*i, d->light_xdir+3*i, -scl * m->vis.scale.light -0.0001);

    mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_LIGHT);
    if (!thisgeom) {
      return;
    }

    // construct geom
    thisgeom->type = mjGEOM_CYLINDER;
    thisgeom->size[0] = scl * m->vis.scale.light * 0.8;
    thisgeom->size[1] = scl * m->vis.scale.light * 0.8;
    thisgeom->size[2] = scl * m->vis.scale.light * 1.0;
    mju_n2f(thisgeom->pos, vec, 3);
    mju_n2f(thisgeom->mat, mat, 9);
    f2f(thisgeom->rgba, m->vis.rgba.light, 4);

    // vopt->label
    if (vopt->label == mjLABEL_LIGHT) {
      makeLabel(m, mjOBJ_LIGHT, i, thisgeom->label);
    }

    releaseGeom(&thisgeom, scn);

    if (vopt->frame != mjFRAME_LIGHT) {
      continue;
    }
    mjtNum width = m->vis.scale.framewidth * scl;
    mjtNum length = m->vis.scale.framelength * scl;
    addFrame(scn, i, d->light_xpos+3*i, mat, length, width);
  }
}


static void addCenterOfMassGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                 mjvScene* scn) {
  // center of mass for root bodies
  if (!vopt->flags[mjVIS_COM]) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=1; i < m->nbody; i++) {
    if (m->body_rootid[i] == i) {
      mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
      if (!thisgeom) {
        return;
      }

      thisgeom->type = mjGEOM_SPHERE;
      thisgeom->size[0] = thisgeom->size[1] = thisgeom->size[2] = scl * m->vis.scale.com;
      mju_n2f(thisgeom->pos, d->subtree_com+3*i, 3);
      mju_n2f(thisgeom->mat, IDENTITY, 9);
      f2f(thisgeom->rgba, m->vis.rgba.com, 4);
      releaseGeom(&thisgeom, scn);
    }
  }
}


static void addAutoConnectGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                mjvScene* scn) {
  if (!vopt->flags[mjVIS_AUTOCONNECT]) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=1; i < m->nbody; i++) {
    // do not connect to world
    if (m->body_parentid[i] == 0) {
      continue;
    }

    // start at body com, connect joint centers in reverse order
    mjtNum* cur = d->xipos+3*i;
    if (m->body_jntnum[i]) {
      for (int j=m->body_jntadr[i]+m->body_jntnum[i]-1; j >= m->body_jntadr[i]; j--) {
        mjtNum* nxt = d->xanchor+3*j;

        // construct geom
        addConnector(scn, mjGEOM_CAPSULE, scl * m->vis.scale.connect, cur, nxt,
                     m->vis.rgba.connect, i, mjCAT_DECOR, mjOBJ_UNKNOWN);

        cur = nxt;
      }
    }

    // connect first joint (or com) to parent com
    mjtNum* first = d->xipos+3*m->body_parentid[i];
    addConnector(scn, mjGEOM_CAPSULE, scl * m->vis.scale.connect, cur, first,
                 m->vis.rgba.connect, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
  }
}


static void addRangefinderGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                mjvScene* scn) {
  if (!vopt->flags[mjVIS_RANGEFINDER]) {
    return;
  }

  const float scl = m->stat.meansize;
  mjtNum framewidth = m->vis.scale.framewidth * scl;
  mjtNum framelength = m->vis.scale.framelength * scl;

  for (int i=0; i < m->nsensor; i++) {
    if (m->sensor_type[i] == mjSENS_RANGEFINDER) {
      int objid = m->sensor_objid[i];
      int adr = m->sensor_adr[i];

      // get dataspec and compute field offsets
      int dataspec = m->sensor_intprm[i*mjNSENS];
      int size = mju_raydataSize(dataspec);
      int offset[mjNRAYDATA] = {0};
      int increment = 0;
      for (int j=0; j < mjNRAYDATA; j++) {
        offset[j] = increment;
        if (dataspec & (1 << j)) {
          increment += mjRAYDATA_SIZE[j];
        }
      }

      // site-attached rangefinder
      if (m->sensor_objtype[i] == mjOBJ_SITE) {
        const mjtNum* ptr = d->sensordata + adr;

        // get distance (if present)
        mjtNum dist = -1;
        if (dataspec & (1 << mjRAYDATA_DIST)) {
          dist = ptr[offset[mjRAYDATA_DIST]];
        }

        // get point and draw line if dist is valid
        mjtNum point[3] = {0};
        if (dist >= 0) {
          mjtNum* origin = d->site_xpos + 3*objid;
          point[0] = origin[0] + d->site_xmat[9*objid+2]*dist;
          point[1] = origin[1] + d->site_xmat[9*objid+5]*dist;
          point[2] = origin[2] + d->site_xmat[9*objid+8]*dist;
          addConnector(scn, mjGEOM_LINE, 3, origin, point, m->vis.rgba.rangefinder,
                       i, mjCAT_DECOR, mjOBJ_SENSOR);
        }

        // draw point if present and non-zero
        if (dataspec & (1 << mjRAYDATA_POINT)) {
          const mjtNum* point_data = ptr + offset[mjRAYDATA_POINT];
          if (point_data[0] || point_data[1] || point_data[2]) {
            mju_copy3(point, point_data);
            mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_SENSOR);
            if (thisgeom) {
              thisgeom->type = mjGEOM_SPHERE;
              thisgeom->size[0] = thisgeom->size[1] = thisgeom->size[2] = 1.5 * framewidth;
              mju_n2f(thisgeom->pos, point, 3);
              mju_n2f(thisgeom->mat, IDENTITY, 9);
              f2f(thisgeom->rgba, m->vis.rgba.rangefinder, 4);
              releaseGeom(&thisgeom, scn);
            }
          }
        }

        // draw normal if present and point is valid
        int valid_point = dist >= 0 || point[0] || point[1] || point[2];
        if (valid_point && (dataspec & (1 << mjRAYDATA_NORMAL))) {
          const mjtNum* normal = ptr + offset[mjRAYDATA_NORMAL];
          mjtNum to[3];
          mju_addScl3(to, point, normal, 2*framelength);
          addConnector(scn, mjGEOM_ARROW1, framewidth, point, to,
                       m->vis.rgba.rangefinder, i, mjCAT_DECOR, mjOBJ_SENSOR);
        }
      }

      // camera-attached rangefinder
      else if (m->sensor_objtype[i] == mjOBJ_CAMERA) {
        const int width = m->cam_resolution[2*objid];
        const int height = m->cam_resolution[2*objid+1];
        const mjtNum* cam_xpos = d->cam_xpos + 3*objid;
        const mjtNum* cam_xmat = d->cam_xmat + 9*objid;
        const int projection = m->cam_projection[objid];

        // compute focal length in pixels using helper
        mjtNum fx, fy, cx, cy, ortho_extent;
        mju_camIntrinsics(m, objid, &fx, &fy, &cx, &cy, &ortho_extent);

        // draw for each pixel
        for (int row = 0; row < height; row++) {
          for (int col = 0; col < width; col++) {
            int idx = row*width + col;
            const mjtNum* ptr = d->sensordata + adr + idx*size;

            // get distance (if present)
            mjtNum dist = -1;
            if (dataspec & (1 << mjRAYDATA_DIST)) {
              dist = ptr[offset[mjRAYDATA_DIST]];
            }

            // compute ray origin and direction
            mjtNum origin[3], direction[3];
            mju_camPixelRay(origin, direction, cam_xpos, cam_xmat,
                            col, row, fx, fy, cx, cy, projection, ortho_extent);

            // get point and draw line if dist is valid
            mjtNum point[3] = {0};
            if (dist >= 0) {
              mju_addScl3(point, origin, direction, dist);
              addConnector(scn, mjGEOM_LINE, 3, origin, point, m->vis.rgba.rangefinder,
                           i, mjCAT_DECOR, mjOBJ_SENSOR);
            }

            // draw point if present and non-zero
            if (dataspec & (1 << mjRAYDATA_POINT)) {
              const mjtNum* point_data = ptr + offset[mjRAYDATA_POINT];
              if (point_data[0] || point_data[1] || point_data[2]) {
                mju_copy3(point, point_data);
                mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_SENSOR);
                if (thisgeom) {
                  thisgeom->type = mjGEOM_SPHERE;
                  thisgeom->size[0] = thisgeom->size[1] = thisgeom->size[2] = 1.3 * framewidth;
                  mju_n2f(thisgeom->pos, point, 3);
                  mju_n2f(thisgeom->mat, IDENTITY, 9);
                  f2f(thisgeom->rgba, m->vis.rgba.rangefinder, 4);
                  releaseGeom(&thisgeom, scn);
                }
              }
            }

            // draw normal if present and point is valid
            int valid_point = dist >= 0 || point[0] || point[1] || point[2];
            if (valid_point && (dataspec & (1 << mjRAYDATA_NORMAL))) {
              const mjtNum* normal = ptr + offset[mjRAYDATA_NORMAL];
              mjtNum to[3];
              mju_addScl3(to, point, normal, 2*framelength);
              addConnector(scn, mjGEOM_ARROW1, framewidth, point, to,
                           m->vis.rgba.rangefinder, i, mjCAT_DECOR, mjOBJ_SENSOR);
            }
          }
        }
      }
    } else if (m->sensor_type[i] == mjSENS_GEOMFROMTO) {
      // sensor data
      mjtNum* fromto = d->sensordata + m->sensor_adr[i];

      // null output: nothing to render
      if (mju_isZero(fromto, 6)) {
        continue;
      }

      // make ray
      addConnector(scn, mjGEOM_LINE, 3, fromto, fromto+3, m->vis.rgba.rangefinder,
                   i, mjCAT_DECOR, mjOBJ_SENSOR);
    }
  }
}


static void addExternalPerturbGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                                    mjvScene* scn) {
  if (!vopt->flags[mjVIS_PERTFORCE]) {
    return;
  }

  const float scl = m->stat.meansize;
  for (int i=1; i < m->nbody; i++) {
    if (mju_isZero(d->xfrc_applied+6*i, 6)) {
      continue;
    }
    // force perturbation
    mjtNum* xfrc = d->xfrc_applied+6*i;
    if (mju_norm3(xfrc) <= mjMINVAL) {
      continue;
    }

    mjtNum* from = d->xipos+3*i;

    // map force to spatial vector in world frame
    mjtNum vec[3];
    mju_scl3(vec, xfrc, m->vis.map.force/m->stat.meanmass);
    mjtNum to[3];
    mju_add3(to, from, vec);

    addConnector(scn, mjGEOM_ARROW, m->vis.scale.forcewidth * scl, from, to,
                 m->vis.rgba.force, i, mjCAT_DECOR, mjOBJ_UNKNOWN);
  }
}


static void addConstraintGeoms(const mjModel* m, mjData* d, const mjvOption* vopt, mjvScene* scn) {
  if (!vopt->flags[mjVIS_CONSTRAINT]) {
    return;
  }

  // connect or weld
  const float scl = m->stat.meansize;
  for (int i=0; i < m->neq; i++) {
    int is_weld = m->eq_type[i] == mjEQ_WELD;
    int is_connect = m->eq_type[i] == mjEQ_CONNECT;
    if (d->eq_active[i] && (is_connect || is_weld)) {
      // compute endpoints in global coordinates
      mjtNum vec[3], end[3];
      mjtNum *xmat_j, *xmat_k;
      int j = m->eq_obj1id[i], k = m->eq_obj2id[i];
      if (m->eq_objtype[i] == mjOBJ_SITE) {
        mju_copy3(vec, d->site_xpos+3*j);
        mju_copy3(end, d->site_xpos+3*k);
        xmat_j = d->site_xmat+9*j;
        xmat_k = d->site_xmat+9*k;
      } else {
        mju_mulMatVec3(vec, d->xmat+9*j, m->eq_data+mjNEQDATA*i+3*is_weld);
        mju_addTo3(vec, d->xpos+3*j);
        mju_mulMatVec3(end, d->xmat+9*k, m->eq_data+mjNEQDATA*i+3*is_connect);
        mju_addTo3(end, d->xpos+3*k);
        xmat_j = d->xmat+9*j;
        xmat_k = d->xmat+9*k;
      }

      // construct geom
      mjtNum sz[3];
      sz[0] = scl * m->vis.scale.constraint;

      mjvGeom* thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_EQUALITY);
      if (!thisgeom) {
        return;
      }

      mjv_initGeom(thisgeom, mjGEOM_SPHERE, sz, vec, xmat_j, m->vis.rgba.connect);
      if (vopt->label == mjLABEL_CONSTRAINT) {
        makeLabel(m, mjOBJ_EQUALITY, i, thisgeom->label);
      }
      releaseGeom(&thisgeom, scn);

      thisgeom = acquireGeom(scn, i, mjCAT_DECOR, mjOBJ_EQUALITY);
      if (!thisgeom) {
        return;
      }

      mjv_initGeom(thisgeom, mjGEOM_SPHERE, sz, end, xmat_k, m->vis.rgba.constraint);
      if (vopt->label == mjLABEL_CONSTRAINT) {
        makeLabel(m, mjOBJ_EQUALITY, i, thisgeom->label);
      }
      releaseGeom(&thisgeom, scn);
    }
  }
}


// add abstract geoms
void mjv_addGeoms(const mjModel* m, mjData* d, const mjvOption* vopt,
                  const mjvPerturb* pert, int catmask, mjvScene* scn) {
  // make default pert if missing
  mjvPerturb localpert;
  if (!pert) {
    mjv_defaultPerturb(&localpert);
    pert = &localpert;
  }

  // clear mjCAT_STATIC bit if mjVIS_STATIC is not set
  if (!vopt->flags[mjVIS_STATIC]) {
    catmask &= (~mjCAT_STATIC);
  }

  addFlexGeoms(m, d, vopt, pert, catmask, scn);
  addSkinGeoms(m, d, vopt, pert, catmask, scn);
  addGeomGeoms(m, d, vopt, pert, catmask, scn);
  addSiteGeoms(m, d, vopt, pert, catmask, scn);
  addSpatialTendonGeoms(m, d, vopt, catmask, scn);
  addSliderCrankGeoms(m, d, vopt, catmask, scn);

  // remaining functions only add decor elements
  if (!(catmask & mjCAT_DECOR)) {
    return;
  }

  addGeomFrameGeoms(m, d, vopt, catmask, scn);
  addSiteFrameGeoms(m, d, vopt, catmask, scn);
  addBodyBvhGeoms(m, d, vopt, scn);
  addFlexBvhGeoms(m, d, vopt, scn);
  addMeshBvhGeoms(m, d, vopt, scn);
  addMeshOctreeGeoms(m, d, vopt, scn);
  addTactileSensorGeoms(m, d, vopt, scn);
  addInertiaGeoms(m, d, vopt, pert, catmask, scn);
  addPerturbGeoms(m, d, vopt, pert, scn);
  addWorldBodyFrameGeoms(m, d, vopt, catmask, scn);
  addSelectionPointGeoms(m, d, vopt, pert, scn);
  addBodyLabelGeoms(m, d, vopt, pert, catmask, scn);
  addJointGeoms(m, d, vopt, scn);
  addActuatorGeoms(m, d, vopt, scn);
  addIslandLabelGeoms(m, d, vopt, scn);
  addCameraGeoms(m, d, vopt, scn);
  addLightGeoms(m, d, vopt, scn);
  addCenterOfMassGeoms(m, d, vopt, scn);
  addAutoConnectGeoms(m, d, vopt, scn);
  addRangefinderGeoms(m, d, vopt, scn);
  addExternalPerturbGeoms(m, d, vopt, scn);
  addConstraintGeoms(m, d, vopt, scn);
  addContactGeoms(m, d, vopt, scn, catmask);
}


// make list of lights only
void mjv_makeLights(const mjModel* m, const mjData* d, mjvScene* scn) {
  mjvLight* thislight;

  // clear counter
  scn->nlight = 0;

  // headlight
  if (m->vis.headlight.active) {
    // get pointer
    thislight = scn->lights;

    // set default properties
    memset(thislight, 0, sizeof(mjvLight));
    thislight->id = -1;
    thislight->headlight = 1;
    thislight->texid = -1;
    thislight->type = mjLIGHT_DIRECTIONAL;
    thislight->castshadow = 0;
    thislight->bulbradius = 0.02;
    thislight->intensity = 0;
    thislight->range = 10;

    // compute head position and gaze direction in model space
    mjtNum hpos[3], hfwd[3];
    mjv_cameraInModel(hpos, hfwd, NULL, scn);
    mju_n2f(thislight->pos, hpos, 3);
    mju_n2f(thislight->dir, hfwd, 3);

    // copy colors
    f2f(thislight->ambient, m->vis.headlight.ambient, 3);
    f2f(thislight->diffuse, m->vis.headlight.diffuse, 3);
    f2f(thislight->specular, m->vis.headlight.specular, 3);

    // advance counter
    scn->nlight++;
  }

  // remaining lights
  for (int i=0; i < m->nlight && scn->nlight < mjMAXLIGHT; i++) {
    if (m->light_active[i]) {
      // get pointer
      thislight = scn->lights + scn->nlight;

      // copy properties
      memset(thislight, 0, sizeof(mjvLight));
      thislight->id = i;
      thislight->type = m->light_type[i];
      thislight->texid = m->light_texid[i];
      thislight->castshadow = m->light_castshadow[i];
      thislight->bulbradius = m->light_bulbradius[i];
      thislight->intensity = m->light_intensity[i];
      thislight->range = m->light_range[i];
      if (thislight->type == mjLIGHT_SPOT) {
        f2f(thislight->attenuation, m->light_attenuation+3*i, 3);
        thislight->exponent = m->light_exponent[i];
        thislight->cutoff = m->light_cutoff[i];
        thislight->softness = m->light_softness[i];
      }

      // copy colors
      f2f(thislight->ambient, m->light_ambient+3*i, 3);
      f2f(thislight->diffuse, m->light_diffuse+3*i, 3);
      f2f(thislight->specular, m->light_specular+3*i, 3);

      // copy position and direction
      mju_n2f(thislight->pos, d->light_xpos+3*i, 3);
      mju_n2f(thislight->dir, d->light_xdir+3*i, 3);

      // advance counter
      scn->nlight++;
    }
  }
}


// update camera only
void mjv_updateCamera(const mjModel* m, const mjData* d, mjvCamera* cam, mjvScene* scn) {
  // return if nothing to do
  if (!m || !cam || cam->type == mjCAMERA_USER) {
    return;
  }

  // move lookat for tracking
  if (cam->type == mjCAMERA_TRACKING) {
    // get id and check
    int bid = cam->trackbodyid;
    if (bid < 0 || bid >= m->nbody) {
      mjERROR("track body id is outside valid range");
    }

    mju_copy3(cam->lookat, d->subtree_com + 3*bid);
  }

  // get camera frustum; for a fixed camera this validates fixedcamid
  float zver[2], zhor[2], zclip[2] = {0, 0};
  mjv_cameraFrustum(zver, zhor, zclip, m, cam);

  // get camera frame (fixedcamid already validated by mjv_cameraFrustum above)
  mjtNum headpos[3], forward[3], up[3], right[3];
  mjv_cameraFrame(headpos, forward, up, right, d, cam);

  // get ipd, orthographic
  int orthographic = 0;
  mjtNum ipd;

  switch (cam->type) {
  case mjCAMERA_FREE:
  case mjCAMERA_TRACKING:
    ipd = m->vis.global.ipd;
    orthographic = m->vis.global.orthographic;
    break;
  case mjCAMERA_FIXED: {
    // fixedcamid range already validated by mjv_cameraFrustum
    int cid = cam->fixedcamid;
    ipd = m->cam_ipd[cid];
    orthographic = m->cam_projection[cid] == mjPROJ_ORTHOGRAPHIC;
    break;
  }

  default:
    mjERROR("unknown camera type");
  }

  // compute GL cameras
  for (int view=0; view < 2; view++) {
    // set frame
    for (int i=0; i < 3; i++) {
      scn->camera[view].pos[i] = (float)(headpos[i] + (view ? ipd : -ipd)*0.5*right[i]);
      scn->camera[view].forward[i] = (float)forward[i];
      scn->camera[view].up[i] = (float)up[i];
    }

    // set orthographic
    scn->camera[view].orthographic = orthographic;

    // set symmetric frustum using intrinsic camera matrix
    scn->camera[view].frustum_top = zver[0];
    scn->camera[view].frustum_bottom = -zver[1];
    scn->camera[view].frustum_center = (zhor[1] - zhor[0]) / 2;
    scn->camera[view].frustum_width = (zhor[1] + zhor[0]) / 2;
    scn->camera[view].frustum_near = zclip[0];
    scn->camera[view].frustum_far = zclip[1];
  }

  // disable model transformation (do not clear float data; user may need it later)
  scn->enabletransform = 0;
}


// construct face, flat normals
static void makeFace(float* _face, float* _normal,  mjtNum radius, const mjtNum* vertxpos,
                     int nface, int i0, int i1, int i2) {
  float* face = _face + 9*nface;
  float* normal = _normal + 9*nface;
  const mjtNum* v0 = vertxpos + 3*i0;
  const mjtNum* v1 = vertxpos + 3*i1;
  const mjtNum* v2 = vertxpos + 3*i2;

  // compute normal
  mjtNum v01[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
  mjtNum v02[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
  mjtNum nrm[3];
  mju_cross(nrm, v01, v02);
  mju_normalize3(nrm);

  // set vertices: offset by radius*normal
  mjtNum temp[3];
  mju_addScl3(temp, v0, nrm, radius);
  mju_n2f(face, temp, 3);
  mju_addScl3(temp, v1, nrm, radius);
  mju_n2f(face+3, temp, 3);
  mju_addScl3(temp, v2, nrm, radius);
  mju_n2f(face+6, temp, 3);

  // set normals
  mju_n2f(normal, nrm, 3);
  mju_n2f(normal+3, nrm, 3);
  mju_n2f(normal+6, nrm, 3);
}


// add face normal to vertices
static void addNormal(mjtNum* vertnorm, const mjtNum* vertxpos,
                      int i0, int i1, int i2) {
  // compute normal*area
  const mjtNum* v0 = vertxpos + 3*i0;
  const mjtNum* v1 = vertxpos + 3*i1;
  const mjtNum* v2 = vertxpos + 3*i2;
  mjtNum v01[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
  mjtNum v02[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
  mjtNum nrm[3];
  mju_cross(nrm, v01, v02);
  mju_normalize3(nrm);

  // accumulate at each vertex
  mju_addTo3(vertnorm + 3*i0, nrm);
  mju_addTo3(vertnorm + 3*i1, nrm);
  mju_addTo3(vertnorm + 3*i2, nrm);
}


// construct face, smooth normals
static void makeSmooth(float* _face, float* _normal, mjtNum radius, mjtByte flg_flat,
                       const mjtNum* vertnorm, const mjtNum* vertxpos,
                       int nface, int i0, int i1, int i2) {
  float* face = _face + 9*nface;
  float* normal = _normal + 9*nface;
  int ind[3] = {i0, i1, i2};
  int sign = radius > 0 ? 1 : -1;

  // flat shading
  if (flg_flat) {
    // compute face normal
    const mjtNum* v0 = vertxpos + 3*i0;
    const mjtNum* v1 = vertxpos + 3*i1;
    const mjtNum* v2 = vertxpos + 3*i2;
    mjtNum v01[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    mjtNum v02[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    mjtNum nrm[3];
    mju_cross(nrm, v01, v02);
    mju_normalize3(nrm);

    // set all vertex normals equal to face normal
    for (int k=0; k < 3; k++){
      normal[3*k+0] = (float) nrm[0];
      normal[3*k+1] = (float) nrm[1];
      normal[3*k+2] = (float) nrm[2];
    }
  }

  // smooth shading
  else {
    for (int k=0; k < 3; k++){
      normal[3*k+0] = (float) (sign*vertnorm[3*ind[k]+0]);
      normal[3*k+1] = (float) (sign*vertnorm[3*ind[k]+1]);
      normal[3*k+2] = (float) (sign*vertnorm[3*ind[k]+2]);
    }
  }

  // set positions: vertices offset by radius*normal
  for (int k=0; k < 3; k++){
    face[3*k+0] = (float) (vertxpos[3*ind[k]+0] + radius*vertnorm[3*ind[k]+0]);
    face[3*k+1] = (float) (vertxpos[3*ind[k]+1] + radius*vertnorm[3*ind[k]+1]);
    face[3*k+2] = (float) (vertxpos[3*ind[k]+2] + radius*vertnorm[3*ind[k]+2]);
  }
}


// construct side in 2D face
static void makeSide(float* _face, float* _normal, mjtNum radius,
                     const mjtNum* vertnorm, const mjtNum* vertxpos,
                     int nface, int i0, int i1) {
  float* face = _face + 9*nface;
  float* normal = _normal + 9*nface;

  // compute normal
  const mjtNum* v0 = vertxpos + 3*i0;
  const mjtNum* v1 = vertxpos + 3*i1;
  mjtNum v01[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
  mjtNum nrm[3];
  mju_cross(nrm, v01, vertnorm+3*i1);
  if (radius < 0) {
    mju_scl3(nrm, nrm, -1);
  }
  mju_normalize3(nrm);

  // set normals
  for (int k=0; k < 3; k++){
    normal[3*k+0] = (float) nrm[0];
    normal[3*k+1] = (float) nrm[1];
    normal[3*k+2] = (float) nrm[2];
  }

  // set positions
  int ind[3] = {i0, i1, i1};
  for (int k=0; k < 3; k++){
    mjtNum sign = (k == 1 ? -1 : +1);
    face[3*k+0] = (float) (vertxpos[3*ind[k]+0] + sign*radius*vertnorm[3*ind[k]+0]);
    face[3*k+1] = (float) (vertxpos[3*ind[k]+1] + sign*radius*vertnorm[3*ind[k]+1]);
    face[3*k+2] = (float) (vertxpos[3*ind[k]+2] + sign*radius*vertnorm[3*ind[k]+2]);
  }
}


// copy texcoord for face
static void copyTex(float* dst, const float* src, int nface, int i0, int i1, int i2) {
  if (!dst || !src) {
    return;
  }

  dst[6*nface+0] = src[2*i0];
  dst[6*nface+1] = src[2*i0+1];
  dst[6*nface+2] = src[2*i1];
  dst[6*nface+3] = src[2*i1+1];
  dst[6*nface+4] = src[2*i2];
  dst[6*nface+5] = src[2*i2+1];
}


// update visible flexes only
void mjv_updateActiveFlex(const mjModel* m, mjData* d, mjvScene* scn, const mjvOption* opt) {
  // save flex visualization flags in scene (needed by renderer)
  scn->flexvertopt = opt->flags[mjVIS_FLEXVERT];
  scn->flexedgeopt = opt->flags[mjVIS_FLEXEDGE];
  scn->flexfaceopt = opt->flags[mjVIS_FLEXFACE];
  scn->flexskinopt = opt->flags[mjVIS_FLEXSKIN];

  // convert vertex positions from mjtNum to float
  for (int v=0; v < 3*m->nflexvert; v++) {
    scn->flexvert[v] = (float) d->flexvert_xpos[v];
  }

  // construct faces
  mj_markStack(d);
  int maxmat = m->nmat + 1;
  int* matids = mjSTACKALLOC(d, maxmat, int);
  int* group_offset = mjSTACKALLOC(d, maxmat, int);
  int* group_count = mjSTACKALLOC(d, maxmat, int);

  for (int f=0; f < m->nflex; f++) {
    int dim = m->flex_dim[f];
    mjtNum radius = m->flex_radius[f];
    mjtByte flg_flat = m->flex_flatskin[f];
    const mjtNum* vertxpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
    float* face = scn->flexface + 9*scn->flexfaceadr[f];
    float* normal = scn->flexnormal + 9*scn->flexfaceadr[f];
    float* texdst = m->flex_texcoordadr[f] >= 0 ?
                    scn->flextexcoord + 6*scn->flexfaceadr[f] : NULL;
    const float* texsrc = m->flex_texcoordadr[f] >= 0 ?
                          m->flex_texcoord + 2*m->flex_texcoordadr[f] : NULL;
    int nmat = flexUniqueMatids(m, f, opt, matids, maxmat);

    // 1D, or face and skin disabled: no faces
    if (dim == 1 || (!opt->flags[mjVIS_FLEXFACE] && !opt->flags[mjVIS_FLEXSKIN])) {
      scn->flexfaceused[f] = 0;
      for (int k=0; k < nmat; k++) {
        group_offset[k] = 0;
        group_count[k] = 0;
      }
    }

    // 2D or 3D face: faces from elements, flat normals, texture
    else if (!opt->flags[mjVIS_FLEXSKIN]) {
      int nface = 0;
      for (int k=0; k < nmat; k++) {
        group_offset[k] = nface;
        for (int e=0; e < m->flex_elemnum[f]; e++) {
          if (nmat > 1 && m->flex_matid[m->flex_elemadr[f]+e] != matids[k]) {
            continue;
          }
          // in 3D, show only elements in selected layer
          if (dim == 2 || m->flex_elemlayer[m->flex_elemadr[f]+e] == opt->flex_layer) {
            // get element data
            const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim+1);
            const int* tdata = m->flex_elemtexcoord + m->flex_elemdataadr[f] + e*(dim+1);

            // triangles: two faces per element
            if (dim == 2) {
              makeFace(face, normal, radius, vertxpos, nface, edata[0], edata[1], edata[2]);
              copyTex(texdst, texsrc, nface, tdata[0], tdata[1], tdata[2]);
              nface++;

              makeFace(face, normal, radius, vertxpos, nface, edata[0], edata[2], edata[1]);
              copyTex(texdst, texsrc, nface, tdata[0], tdata[2], tdata[1]);
              nface++;
            }

            // tetrahedra: four faces per element
            else {
              makeFace(face, normal, radius, vertxpos,
                       nface, edata[0], edata[1], edata[2]);
              copyTex(texdst, texsrc, nface, tdata[0], tdata[1], tdata[2]);
              nface++;

              makeFace(face, normal, radius, vertxpos,
                       nface, edata[0], edata[2], edata[3]);
              copyTex(texdst, texsrc, nface, tdata[0], tdata[2], tdata[3]);
              nface++;

              makeFace(face, normal, radius, vertxpos,
                       nface, edata[0], edata[3], edata[1]);
              copyTex(texdst, texsrc, nface, tdata[0], tdata[3], tdata[1]);
              nface++;

              makeFace(face, normal, radius, vertxpos,
                       nface, edata[1], edata[3], edata[2]);
              copyTex(texdst, texsrc, nface, tdata[1], tdata[3], tdata[2]);
              nface++;
            }
          }
        }
        group_count[k] = nface - group_offset[k];
      }

      // save face count
      scn->flexfaceused[f] = nface;
    }

    // 2D or 3D skin: faces from elements (2D) or shells (3D), smooth normals, texture
    else {
      // allocate and clear vertex normals for smoothing
      mj_markStack(d);
      mjtNum* vertnorm = mjSTACKALLOC(d, 3*m->flex_vertnum[f], mjtNum);
      mju_zero(vertnorm, 3*m->flex_vertnum[f]);

      // add vertex normals: top element sides in 2D, shell fragments in 3D
      if (dim == 2) {
        for (int e=0; e < m->flex_elemnum[f]; e++) {
          const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim+1);
          addNormal(vertnorm, vertxpos, edata[0], edata[1], edata[2]);
        }
      } else {
        for (int s=0; s < m->flex_shellnum[f]; s++) {
          const int* sdata = m->flex_shell + m->flex_shelldataadr[f] + s*dim;
          addNormal(vertnorm, vertxpos, sdata[0], sdata[1], sdata[2]);
        }
      }

      // normalize vertex normals
      for (int i=0; i < m->flex_vertnum[f]; i++) {
        mju_normalize3(vertnorm+3*i);
      }

      // precompute shell fragment material ids once per multi-material flex
      int* shell_matid = NULL;
      if (nmat > 1 && m->flex_shellnum[f] > 0) {
        shell_matid = mjSTACKALLOC(d, m->flex_shellnum[f], int);
        flexShellMatid(m, d, f, shell_matid);
      }

      // create faces, offset along smoothed vertex normals, and texcoord
      int nface = 0;
      for (int k=0; k < nmat; k++) {
        group_offset[k] = nface;
        if (dim == 2) {
          for (int e=0; e < m->flex_elemnum[f]; e++) {
            if (nmat > 1 && m->flex_matid[m->flex_elemadr[f]+e] != matids[k]) {
              continue;
            }
            const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim+1);
            const int* tdata = m->flex_elemtexcoord + m->flex_elemdataadr[f] + e*(dim+1);
            makeSmooth(face, normal, radius, flg_flat, vertnorm, vertxpos,
                       nface, edata[0], edata[1], edata[2]);
            copyTex(texdst, texsrc, nface, tdata[0], tdata[1], tdata[2]);
            nface++;
            makeSmooth(face, normal, -radius, flg_flat, vertnorm, vertxpos,
                       nface, edata[0], edata[2], edata[1]);
            copyTex(texdst, texsrc, nface, tdata[0], tdata[2], tdata[1]);
            nface++;
          }
          // 2D: close sides using shell fragments
          for (int s=0; s < m->flex_shellnum[f]; s++) {
            if (nmat > 1 && shell_matid[s] != matids[k]) {
              continue;
            }
            const int* sdata = m->flex_shell + m->flex_shelldataadr[f] + s*dim;
            makeSide(face, normal, radius, vertnorm, vertxpos,
                     nface, sdata[0], sdata[1]);
            copyTex(texdst, texsrc, nface, sdata[0], sdata[1], sdata[1]);
            nface++;
            makeSide(face, normal, -radius, vertnorm, vertxpos,
                     nface, sdata[1], sdata[0]);
            copyTex(texdst, texsrc, nface, sdata[1], sdata[0], sdata[0]);
            nface++;
          }
        } else {
          for (int s=0; s < m->flex_shellnum[f]; s++) {
            if (nmat > 1 && shell_matid[s] != matids[k]) {
              continue;
            }
            const int* sdata = m->flex_shell + m->flex_shelldataadr[f] + s*dim;
            makeSmooth(face, normal, radius, flg_flat, vertnorm, vertxpos,
                       nface, sdata[0], sdata[1], sdata[2]);
            copyTex(texdst, texsrc, nface, sdata[0], sdata[1], sdata[2]);
            nface++;
          }
        }
        group_count[k] = nface - group_offset[k];
      }

      // save face count
      scn->flexfaceused[f] = nface;
      mj_freeStack(d);
    }

    // check face count, SHOULD NOT OCCUR
    if (scn->flexfaceused[f] > scn->flexfacenum[f]) {
      mju_error("too many flex faces in mjv_updateActiveFlex");
    }

    // record per-material face offset and count on multi-material flex geoms
    if (nmat > 1) {
      for (int g=0; g < scn->ngeom; g++) {
        mjvGeom* geom = scn->geoms + g;
        if (geom->type == mjGEOM_FLEX && geom->objid == f &&
            geom->dataid >= 0 && geom->dataid < nmat) {
          geom->size[1] = (float)group_offset[geom->dataid];
          geom->size[2] = (float)group_count[geom->dataid];
        }
      }
    }
  }
  mj_freeStack(d);
}


// update all skins, here for backward API compatibility
void mjv_updateSkin(const mjModel* m, const mjData* d, mjvScene* scn) {
  mjvOption opt;
  mjv_defaultOption(&opt);
  mjv_updateActiveSkin(m, d, scn, &opt);
  mju_warning("mjv_updateSkin is deprecated, please use mjv_updateActiveSkin.");
}


// update visible skins only
void mjv_updateActiveSkin(const mjModel* m, const mjData* d, mjvScene* scn, const mjvOption* opt) {
  // process skins
  for (int i=0; i < m->nskin; i++) {
    // get info
    int vertadr = m->skin_vertadr[i];
    int vertnum = m->skin_vertnum[i];
    int faceadr = m->skin_faceadr[i];
    int facenum = m->skin_facenum[i];

    // clear positions and normals
    memset(scn->skinvert + 3*vertadr, 0, 3*vertnum*sizeof(float));
    memset(scn->skinnormal + 3*vertadr, 0, 3*vertnum*sizeof(float));

    // update only if visible
    if (opt->skingroup[mjMAX(0, mjMIN(mjNGROUP-1, m->skin_group[i]))]) {
      // accumulate positions from all bones
      for (int j=m->skin_boneadr[i];
           j < m->skin_boneadr[i]+m->skin_bonenum[i];
           j++) {
        // get bind pose
        mjtNum bindpos[3] = {
          (mjtNum) m->skin_bonebindpos[3*j],
          (mjtNum) m->skin_bonebindpos[3*j+1],
          (mjtNum) m->skin_bonebindpos[3*j+2]
        };
        mjtNum bindquat[4] = {
          (mjtNum) m->skin_bonebindquat[4*j],
          (mjtNum) m->skin_bonebindquat[4*j+1],
          (mjtNum) m->skin_bonebindquat[4*j+2],
          (mjtNum) m->skin_bonebindquat[4*j+3]
        };

        // compute rotation
        int bodyid = m->skin_bonebodyid[j];
        mjtNum quat[4], quatneg[4], rotate[9];
        mju_negQuat(quatneg, bindquat);
        mju_mulQuat(quat, d->xquat+4*bodyid, quatneg);
        mju_quat2Mat(rotate, quat);

        // compute translation
        mjtNum translate[3];
        mju_mulMatVec3(translate, rotate, bindpos);
        mju_sub3(translate, d->xpos+3*bodyid, translate);

        // process all bone vertices
        for (int k=m->skin_bonevertadr[j];
             k < m->skin_bonevertadr[j]+m->skin_bonevertnum[j];
             k++) {
          // vertex id and weight
          int vid = m->skin_bonevertid[k];
          float vweight = m->skin_bonevertweight[k];

          // get original position
          mjtNum pos[3] = {
            (mjtNum) m->skin_vert[3*(vertadr+vid)],
            (mjtNum) m->skin_vert[3*(vertadr+vid)+1],
            (mjtNum) m->skin_vert[3*(vertadr+vid)+2],
          };

          // transform
          mjtNum pos1[3];
          mju_mulMatVec3(pos1, rotate, pos);
          mju_addTo3(pos1, translate);

          // accumulate position
          scn->skinvert[3*(vertadr+vid)] += vweight*(float)pos1[0];
          scn->skinvert[3*(vertadr+vid)+1] += vweight*(float)pos1[1];
          scn->skinvert[3*(vertadr+vid)+2] += vweight*(float)pos1[2];
        }
      }

      // compute vertex normals from face normals
      for (int k=faceadr; k < faceadr+facenum; k++) {
        // get face vertex indices
        int vid[3] = {
          m->skin_face[3*k],
          m->skin_face[3*k+1],
          m->skin_face[3*k+2]
        };

        // get triangle edges
        mjtNum vec01[3], vec02[3];
        for (int r=0; r < 3; r++) {
          vec01[r] = scn->skinvert[3*(vertadr+vid[1])+r] - scn->skinvert[3*(vertadr+vid[0])+r];
          vec02[r] = scn->skinvert[3*(vertadr+vid[2])+r] - scn->skinvert[3*(vertadr+vid[0])+r];
        }

        // compute face normal
        mjtNum nrm[3];
        mju_cross(nrm, vec01, vec02);

        // add normal to each vertex with weight = area
        for (int r=0; r < 3; r++) {
          for (int t=0; t < 3; t++) {
            scn->skinnormal[3*(vertadr+vid[r])+t] += nrm[t];
          }
        }
      }

      // normalize normals
      for (int k=vertadr; k < vertadr+vertnum; k++) {
        float s = sqrtf(
          scn->skinnormal[3*k+0]*scn->skinnormal[3*k+0] +
          scn->skinnormal[3*k+1]*scn->skinnormal[3*k+1] +
          scn->skinnormal[3*k+2]*scn->skinnormal[3*k+2]
          );
        float scl = 1/mjMAX(mjMINVAL, s);

        scn->skinnormal[3*k] *= scl;
        scn->skinnormal[3*k+1] *= scl;
        scn->skinnormal[3*k+2] *= scl;
      }

      // inflate
      if (m->skin_inflate[i]) {
        float inflate = m->skin_inflate[i];
        for (int k=vertadr; k < vertadr+vertnum; k++) {
          scn->skinvert[3*k]   += inflate*scn->skinnormal[3*k];
          scn->skinvert[3*k+1] += inflate*scn->skinnormal[3*k+1];
          scn->skinvert[3*k+2] += inflate*scn->skinnormal[3*k+2];
        }
      }
    }
  }
}


// update entire scene
void mjv_updateScene(const mjModel* m, mjData* d, const mjvOption* opt,
                     const mjvPerturb* pert, mjvCamera* cam, int catmask, mjvScene* scn) {
  // clear geoms
  scn->ngeom = 0;

  // trigger plugin visualization hooks
  if (m->nplugin) {
    const int nslot = mjp_pluginCount();
    // iterate over plugins, call visualize if defined
    for (int i=0; i < m->nplugin; i++) {
      const int slot = m->plugin[i];
      const mjpPlugin* plugin = mjp_getPluginAtSlotUnsafe(slot, nslot);
      if (!plugin) {
        mjERROR("invalid plugin slot: %d", slot);
      }
      if (plugin->visualize) {
        plugin->visualize(m, d, opt, scn, i);
      }
    }
  }

  // add all categories
  mjv_addGeoms(m, d, opt, pert, catmask, scn);

  // update camera
  mjv_updateCamera(m, d, cam, scn);

  // add lights
  mjv_makeLights(m, d, scn);

  // update flexes
  if (opt->flags[mjVIS_FLEXVERT] || opt->flags[mjVIS_FLEXEDGE] ||
      opt->flags[mjVIS_FLEXFACE] || opt->flags[mjVIS_FLEXSKIN]) {
    mjv_updateActiveFlex(m, d, scn, opt);
  }

  // update skins
  if (opt->flags[mjVIS_SKIN]) {
    mjv_updateActiveSkin(m, d, scn, opt);
  }
}


//----------------------------------- catenary functions -------------------------------------------

// returns hyperbolic cosine and optionally computes hyperbolic sine
static inline mjtNum cosh_sinh(mjtNum x, mjtNum* sinh) {
  mjtNum expx = mju_exp(x);
  if (sinh) {
    *sinh = 0.5 * (expx - 1/expx);
  }
  return 0.5 * (expx + 1/expx);
}


// returns intercept of the catenary equation
static inline mjtNum catenary_intercept(mjtNum v, mjtNum h, mjtNum length) {
  return 1/mju_sqrt(mju_sqrt(length*length - v*v)/h - 1);
}


// returns residual of catenary equation and optionally computes its gradient w.r.t b
static inline mjtNum catenary_residual(mjtNum b, mjtNum intercept, mjtNum* grad) {
  mjtNum a = 0.5 / b;
  mjtNum sinh, cosh = cosh_sinh(a, &sinh);
  if (grad) {
    *grad = (a*cosh - sinh) * mju_pow(2*b*sinh - 1, -1.5);
  }
  return 1/mju_sqrt(2*b*sinh - 1) - intercept;
}


// convergence tolerance for catenary solver
static const mjtNum tolerance = 1e-9;



// solve transcendental catenary equation using change of variables proposed in
//   https://math.stackexchange.com/a/1002996
static inline mjtNum solve_catenary(mjtNum v, mjtNum h, mjtNum length) {
  mjtNum intercept = catenary_intercept(v, h, length);

  // initial guess using linear approximation to catenary_residual
  mjtNum b = intercept / mju_sqrt(24);

  // Newton steps to convergence (usually ~ 5 steps)
  for (int i=0; i < 50; i++) {
    // get value and gradient
    mjtNum grad;
    mjtNum res = catenary_residual(b, intercept, &grad);

    if (mju_abs(res) < tolerance) {
      break;
    }

    // Newton step
    mjtNum step = -res / grad;

    // backtracking line-search is not essential but can reduce number of iterations
    for (int j=0; j < 10; j++) {
      mjtNum new_res = catenary_residual(b + step, intercept, NULL);
      if (mju_abs(new_res) < mju_abs(res)) {
        break;
      } else {
        step *= 0.5;
      }
    }

    // take step
    b += step;
  }

  return b;
}


// points along catenary of given length between x0 and x1, returns number of points
int mjv_catenary(const mjtNum x0[3], const mjtNum x1[3], const mjtNum gravity[3], mjtNum length,
                 mjtNum* catenary, int ncatenary) {
  mjtNum dist = mju_dist3(x0, x1);

  // tendon is stretched longer than length: draw straight line
  if (dist > length) {
    // copy start and end points
    mju_copy3(catenary+0, x0);
    mju_copy3(catenary+3, x1);

    return 2;
  }

  // tendon is shorter than length
  else {
    // normalized up vector
    mjtNum up[3];
    mju_scl3(up, gravity, -1);
    mju_normalize3(up);

    // x0 to x1
    mjtNum x01[3];
    mju_sub3(x01, x1, x0);

    // make across orthonormal to up, points from x0 to x1
    mjtNum across[3];
    mju_copy3(across, x01);
    mjtNum tmp[3];
    mju_scl3(tmp, up, mju_dot3(up, across));
    mju_subFrom3(across, tmp);
    mjtNum norm = mju_normalize3(across);

    // if across is numerically tiny, just set to 0
    if (norm < mjMINVAL) {
      mju_zero3(across);
    }

    // extents in the suspension plane
    mjtNum h = mju_dot3(x01, across);  // horizontal suspension extent
    mjtNum v = mju_dot3(x01, up);      // vertical height difference of x1 and x0

    // near vertical tendon, use hanging bead approximation: 3 points
    if (length > 100*h) {
      // solve for location of bead hanging on tendon
      mjtNum d_up = -0.5*(mju_sqrt(length*length - h*h) - v);  // down from x0
      mjtNum d_across = h*d_up / (2*d_up - v);                 // across from x0

      // start point
      mju_copy3(catenary+0, x0);

      // midpoint: bead location
      mju_copy3(catenary+3, x0);
      mju_addToScl3(catenary+3, up, d_up);
      mju_addToScl3(catenary+3, across, d_across);

      // end point
      mju_copy3(catenary+6, x1);

      return 3;
    }

    // compute full catenary: ncatenary points
    else {
      // b*h: scaled catenary flatness
      mjtNum bh = solve_catenary(v, h, length) * h;

      // horizontal and vertical offsets
      mjtNum h_offset = -0.5 * (mju_log((length+v) / (length-v)) * bh - h);
      mjtNum v_offset = -cosh_sinh(h_offset / bh, NULL) * bh;

      // start point
      mju_copy3(catenary+0, x0);

      // hanging points
      for (int i=1; i < ncatenary-1; i++) {
        // linearly spaced horizontal offset
        mjtNum horizontal = i*h/ncatenary;
        mju_addScl3(catenary+3*i, x0, across, horizontal);

        // vertical offset, evaluate catenary values
        mjtNum vertical = bh * cosh_sinh((horizontal - h_offset) / bh, NULL) + v_offset;
        mju_addToScl3(catenary+3*i, up, vertical);
      }

      // end point
      mju_copy3(catenary+3*(ncatenary-1), x1);

      return ncatenary;
    }
  }

  return 0;  // SHOULD NOT OCCUR
}
