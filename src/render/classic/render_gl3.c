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

#include "render/classic/render_gl3.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <mujoco/mjmacro.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "engine/engine_sort.h"
#include "engine/engine_vis_init.h"
#include "render/classic/render_context.h"
#include "render/classic/render_gl2.h"
#include "render/classic/render_util.h"
#include "render/classic/glad/glad.h"


//----------------------------- low-level 3D rendering ---------------------------------------------

// check if head is behind plane
static int isBehind(const float* headpos, const float* pos, const float* mat) {
  return ((headpos[0]-pos[0])*mat[2] +
          (headpos[1]-pos[1])*mat[5] +
          (headpos[2]-pos[2])*mat[8] < 0.0f);
}



// check if geom is reflective
static int isReflective(const mjvGeom* geom) {
  return ((geom->type == mjGEOM_PLANE || geom->type == mjGEOM_BOX) &&
          !geom->transparent &&
          (geom->reflectance > 0));
}



// texture types for settexture
enum {
  mjtexSHADOW = 0,
  mjtexSKYBOX,
  mjtexREGULAR
};


// check if geom type is a builtin shape with explicit UVs
static int isBuiltinWithUV(int type) {
  return (type == mjGEOM_PLANE ||
          type == mjGEOM_SPHERE ||
          type == mjGEOM_ELLIPSOID ||
          type == mjGEOM_BOX ||
          type == mjGEOM_CYLINDER ||
          type == mjGEOM_CAPSULE ||
          type == mjGEOM_ARROW ||
          type == mjGEOM_ARROW1 ||
          type == mjGEOM_ARROW2 ||
          type == mjGEOM_TRIANGLE);
}


// enable/disable texture mapping
static void settexture(int type, int state, const mjrContext* con, const mjvGeom* geom) {
  float plane[4], scl[2];
  int texid = -1, texuniform = 0;
  float texrepeat[2] = {0, 0};
  if (geom) {
    if (geom->texid >= 0) {
      texid = geom->texid;
      texuniform = geom->texuniform;
      texrepeat[0] = geom->texrepeat[0];
      texrepeat[1] = geom->texrepeat[1];
    } else if (geom->matid >= 0) {
      texid = con->mat_texid[mjNTEXROLE * geom->matid + mjTEXROLE_RGB];
      texuniform = con->mat_texuniform[geom->matid];
      texrepeat[0] = con->mat_texrepeat[geom->matid*2];
      texrepeat[1] = con->mat_texrepeat[geom->matid*2+1];
    }
  }

  // shadow
  if (type == mjtexSHADOW) {
    // enable
    if (state) {
      glActiveTexture(GL_TEXTURE1);
      glEnable(GL_TEXTURE_2D);
      glEnable(GL_TEXTURE_GEN_S);
      glEnable(GL_TEXTURE_GEN_T);
      glEnable(GL_TEXTURE_GEN_R);
      glEnable(GL_TEXTURE_GEN_Q);
      glBindTexture(GL_TEXTURE_2D, con->shadowTex);
    }

    // disable
    else {
      glActiveTexture(GL_TEXTURE1);
      glDisable(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_GEN_S);
      glDisable(GL_TEXTURE_GEN_T);
      glDisable(GL_TEXTURE_GEN_R);
      glDisable(GL_TEXTURE_GEN_Q);
      glActiveTexture(GL_TEXTURE0);
    }
  }

  // explicit texture coordinates
  else if (type == mjtexREGULAR && geom &&
           (geom->texcoord || geom->type == mjGEOM_FLEX || geom->type == mjGEOM_SKIN ||
            (texid >= 0 && con->textureType[texid] == mjTEXTURE_2D && isBuiltinWithUV(geom->type)))) {
    // enable
    if (state && texid >= 0) {
      glActiveTexture(GL_TEXTURE0);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, con->texture[texid]);

      // determine scaling
      scl[0] = texrepeat[0] > 0 ? texrepeat[0] : 1.0f;
      scl[1] = texrepeat[1] > 0 ? texrepeat[1] : 1.0f;

      // uniform: repeat relative to spatial units rather than object
      if (texuniform && geom->type != mjGEOM_FLEX && geom->type != mjGEOM_SKIN) {
        if (geom->size[0] > 0) {
          scl[0] = scl[0] * geom->size[0];
        }
        if (geom->size[1] > 0) {
          scl[1] = scl[1] * geom->size[1];
        }
      }

      glMatrixMode(GL_TEXTURE);
      glLoadIdentity();
      if (geom->type == mjGEOM_PLANE && (geom->size[0] <= 0 || geom->size[1] <= 0)) {
        glTranslatef(-0.5f, -0.5f, 0.0f);
      }
      glScalef(scl[0], scl[1], 1.0f);
      glMatrixMode(GL_MODELVIEW);
    }

    // disable
    else {
      glActiveTexture(GL_TEXTURE0);
      glDisable(GL_TEXTURE_2D);
      glMatrixMode(GL_TEXTURE);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
    }
  }

  // 2D
  else if (type == mjtexREGULAR && texid >= 0 && con->textureType[texid] == mjTEXTURE_2D) {
    // enable
    if (state) {
      glActiveTexture(GL_TEXTURE0);
      glEnable(GL_TEXTURE_2D);
      glEnable(GL_TEXTURE_GEN_S);
      glEnable(GL_TEXTURE_GEN_T);
      glBindTexture(GL_TEXTURE_2D, con->texture[texid]);

      // determine scaling, adjust for pre-scaled geoms
      scl[0] = texrepeat[0];
      scl[1] = texrepeat[1];
      if (geom->dataid >= 0) {
        if (geom->size[0] > 0) {
          scl[0] = scl[0] / mju_max(mjMINVAL, geom->size[0]);
        }

        if (geom->size[1] > 0) {
          scl[1] = scl[1] / mju_max(mjMINVAL, geom->size[1]);
        }
      }

      // uniform: repeat relative to spatial units rather than object
      if (texuniform) {
        if (geom->size[0] > 0) {
          scl[0] = scl[0] * geom->size[0];
        }

        if (geom->size[1] > 0) {
          scl[1] = scl[1] * geom->size[1];
        }
      }

      // set mapping
      mjr_setf4(plane, 0.5*scl[0], 0, 0, -0.5);
      glTexGenfv(GL_S, GL_OBJECT_PLANE, plane);
      mjr_setf4(plane, 0, -0.5*scl[1], 0, -0.5);
      glTexGenfv(GL_T, GL_OBJECT_PLANE, plane);
    }

    // disable
    else {
      glActiveTexture(GL_TEXTURE0);
      glDisable(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_GEN_S);
      glDisable(GL_TEXTURE_GEN_T);
    }
  }

  // cube or skybox
  else {
    // enable
    if (state && texid >= 0) {
      glActiveTexture(GL_TEXTURE0);
      glEnable(GL_TEXTURE_CUBE_MAP);
      glEnable(GL_TEXTURE_GEN_S);
      glEnable(GL_TEXTURE_GEN_T);
      glEnable(GL_TEXTURE_GEN_R);
      glBindTexture(GL_TEXTURE_CUBE_MAP, con->texture[texid]);

      // set mapping : cube
      if (type == mjtexREGULAR) {
        mjr_setf4(plane, texuniform ? geom->size[0] : 1, 0, 0, 0);
        glTexGenfv(GL_S, GL_OBJECT_PLANE, plane);
        mjr_setf4(plane, 0, texuniform ? geom->size[1] : 1, 0, 0);
        glTexGenfv(GL_T, GL_OBJECT_PLANE, plane);
        mjr_setf4(plane, 0, 0, texuniform ? geom->size[2] : 1, 0);
        glTexGenfv(GL_R, GL_OBJECT_PLANE, plane);
      }

      // set mapping: skybox (rotate 90 deg around X)
      else {
        mjr_setf4(plane, 1, 0, 0, 0);
        glTexGenfv(GL_S, GL_OBJECT_PLANE, plane);
        mjr_setf4(plane, 0, 0, 1, 0);
        glTexGenfv(GL_T, GL_OBJECT_PLANE, plane);
        mjr_setf4(plane, 0, -1, 0, 0);
        glTexGenfv(GL_R, GL_OBJECT_PLANE, plane);
      }
    }

    // disable
    else {
      glActiveTexture(GL_TEXTURE0);
      glDisable(GL_TEXTURE_CUBE_MAP);
      glDisable(GL_TEXTURE_GEN_S);
      glDisable(GL_TEXTURE_GEN_T);
      glDisable(GL_TEXTURE_GEN_R);
    }
  }
}



// rendering modes for renderGeom
enum {
  mjrRND_NORMAL       = 0,
  mjrRND_SHADOWMAP    = 1,
  mjrRND_SHADOWCAST   = 2,
  mjrRND_SEGMENT      = 3,
  mjrRND_IDCOLOR      = 4
};


// render one geom
static void renderGeom(const mjvGeom* geom, int mode, const float* headpos,
                       const mjvScene* scn, const mjrContext* con) {
  const float* size = geom->size;
  float temp[4] = {0, 0, 0, 1}, av;
  float rgba[4] = {geom->rgba[0], geom->rgba[1], geom->rgba[2], geom->rgba[3]};
  int behind, whichface, lighting;

  // lines, triangles, and geoms with castshadow disabled do not cast shadows
  if (mode == mjrRND_SHADOWCAST &&
      (!geom->castshadow || geom->type == mjGEOM_LINE || geom->category == mjCAT_DECOR)) {
    return;
  }

  // make transformation matrix
  float mat[16] = {
    geom->mat[0], geom->mat[3], geom->mat[6], 0.0f,
    geom->mat[1], geom->mat[4], geom->mat[7], 0.0f,
    geom->mat[2], geom->mat[5], geom->mat[8], 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  };

  // precompute isbehind
  behind = isBehind(headpos, geom->pos, geom->mat);

  // enable texture in normal and shadowmap mode
  if (geom->matid >= 0 &&
      (mode == mjrRND_NORMAL || mode == mjrRND_SHADOWMAP)) {
    settexture(mjtexREGULAR, 1, con, geom);
  }

  // make plane more transparent from the back
  if (geom->type == mjGEOM_PLANE && behind && mode == mjrRND_NORMAL) {
    rgba[3] *= 0.3;
  }

  // set material emission: none in shadow mode
  if (mode == mjrRND_NORMAL) {
    mjr_setf4(temp,
              geom->emission*rgba[0],
              geom->emission*rgba[1],
              geom->emission*rgba[2], 1);
  } else {
    mjr_setf4(temp, 0, 0, 0, 1);
  }
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, temp);

  // set color and material
  glColor4fv(rgba);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, geom->shininess * 128.0f);
  mjr_setf4(temp, geom->specular, geom->specular, geom->specular, 1);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, temp);

  // set color for segmentation mode
  if (mode >= mjrRND_SEGMENT) {
    // segid color
    if (mode == mjrRND_IDCOLOR) {
      unsigned char seg[4] = {
        (geom->segid+1) & 0xFF,
        ((geom->segid+1)>>8) & 0xFF,
        ((geom->segid+1)>>16) & 0xFF,
        0xFF
      };
      glColor4ubv(seg);
    }

    // random color
    else {
      float seg[4] = {
        0.1f + 0.8f*mju_Halton(geom->segid+10, 2),
        0.1f + 0.8f*mju_Halton(geom->segid+10, 3),
        0.1f + 0.8f*mju_Halton(geom->segid+10, 5),
        1
      };
      glColor4fv(seg);
    }
  }

  // apply coordinate transformation, except for flex and skin which are global
  glPushMatrix();
  if (geom->type != mjGEOM_FLEX && geom->type != mjGEOM_SKIN) {
    glTranslatef(geom->pos[0], geom->pos[1], geom->pos[2]);
    glMultMatrixf(mat);
  }

  // render geom
  switch (geom->type) {
  case mjGEOM_PLANE:                              // plane
    if (behind) {
      glGetIntegerv(GL_CULL_FACE_MODE, &whichface);
      if (whichface == GL_BACK) {
        glCullFace(GL_FRONT);
      }
    }

    // use plane-specific displaylist if available
    if (geom->dataid >= 0) {
      glCallList(con->basePlane + geom->dataid+1);
    }

    // otherwise default list
    else {
      glScalef(size[0], size[1], 1.0f);
      glCallList(con->basePlane);
    }

    if (behind && whichface == GL_BACK) {
      glCullFace(GL_BACK);
    }
    break;

  case mjGEOM_HFIELD:                         // height field
    if (geom->dataid >= 0) {
      glCallList(con->baseHField + geom->dataid);
    }
    break;

  case mjGEOM_SPHERE:                         // sphere
    glScalef(size[0], size[1], size[2]);
    glCallList(con->baseBuiltin + mjrSPHERE);
    break;

  case mjGEOM_CAPSULE:                        // capsule
    av = 0.5*(size[0]+size[1]);
    glScalef(size[0], size[1], size[2]);
    glCallList(con->baseBuiltin + mjrCYLINDEROPEN);
    glScalef(1, 1, av/size[2]);
    glTranslatef(0, 0, (size[2]-av)/av);
    glCallList(con->baseBuiltin + mjrSPHERETOP);
    glTranslatef(0, 0, 2*(av-size[2])/av);
    glCallList(con->baseBuiltin + mjrSPHEREBOTTOM);
    break;

  case mjGEOM_ELLIPSOID:                      // ellipsoid
    glScalef(size[0], size[1], size[2]);
    glCallList(con->baseBuiltin + mjrSPHERE);
    break;

  case mjGEOM_CYLINDER:                       // cylinder
    glScalef(size[0], size[1], size[2]);
    glCallList(con->baseBuiltin + mjrCYLINDER);
    break;

  case mjGEOM_BOX:                            // box
    glScalef(size[0], size[1], size[2]);
    glCallList(con->baseBuiltin + mjrBOX);
    break;

  case mjGEOM_MESH:                           // mesh
  case mjGEOM_SDF:
    if (geom->dataid >= 0) {
      glCallList(con->baseMesh + geom->dataid);
    }
    break;

  case mjGEOM_ARROW:                          // arrow
    glScalef(size[0], size[1], size[2]/6.0f);
    glTranslatef(0, 0, 1);
    glCallList(con->baseBuiltin + mjrCYLINDER);
    glTranslatef(0, 0, 1);
    glScalef(1.75f, 1.75f, 1);
    glCallList(con->baseBuiltin + mjrCONE);
    break;

  case mjGEOM_ARROW1:                         // arrow without wedges
    glScalef(size[0], size[1], size[2]/6.0f);
    glTranslatef(0, 0, 1);
    glCallList(con->baseBuiltin + mjrCYLINDER);
    glTranslatef(0, 0, 1);
    glCallList(con->baseBuiltin + mjrCONE);
    break;

  case mjGEOM_ARROW2:                         // arrow in both directions
    glScalef(size[0], size[1], size[2]/3.0f);
    glCallList(con->baseBuiltin + mjrCYLINDEROPEN);
    glScalef(1.75f, 1.75f, 0.5f);
    glTranslatef(0, 0, -2);
    glRotatef(180.0f, 1.0f, 0.0f, 0.0f);
    glCallList(con->baseBuiltin + mjrCONE);
    glRotatef(180.0f, 1.0f, 0.0f, 0.0f);
    glTranslatef(0, 0, 4);
    glCallList(con->baseBuiltin + mjrCONE);
    break;

  case mjGEOM_LINE:                           // line
    glLineWidth(size[0]*con->lineWidth);
    lighting = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, size[2]);
    glEnd();
    glLineWidth(con->lineWidth);
    if (lighting) {
      glEnable(GL_LIGHTING);
    }
    break;

  case mjGEOM_LINEBOX:                        // box with line edges
    glLineWidth(1.5*con->lineWidth);
    lighting = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    // bottom face
    glBegin(GL_LINE_LOOP);
    glVertex3f(-size[0], -size[1], -size[2]);
    glVertex3f( size[0], -size[1], -size[2]);
    glVertex3f( size[0],  size[1], -size[2]);
    glVertex3f(-size[0],  size[1], -size[2]);
    glEnd();
    // top face
    glBegin(GL_LINE_LOOP);
    glVertex3f(-size[0], -size[1], size[2]);
    glVertex3f( size[0], -size[1], size[2]);
    glVertex3f( size[0],  size[1], size[2]);
    glVertex3f(-size[0],  size[1], size[2]);
    glEnd();
    // vertical edges
    glBegin(GL_LINES);
    glVertex3f(-size[0], -size[1], -size[2]);
    glVertex3f(-size[0], -size[1],  size[2]);
    glVertex3f( size[0], -size[1], -size[2]);
    glVertex3f( size[0], -size[1],  size[2]);
    glVertex3f( size[0],  size[1], -size[2]);
    glVertex3f( size[0],  size[1],  size[2]);
    glVertex3f(-size[0],  size[1], -size[2]);
    glVertex3f(-size[0],  size[1],  size[2]);
    glEnd();
    glLineWidth(con->lineWidth);
    if (lighting) {
      glEnable(GL_LIGHTING);
    }
    break;

  case mjGEOM_TRIANGLE:                       // triangle
    glDisable(GL_CULL_FACE);
    glBegin(GL_TRIANGLES);
    glNormal3f(0, 0, 1);
    glTexCoord2f(0, 1);
    glVertex3f(0, 0, 0);
    glTexCoord2f(1, 1);
    glVertex3f(size[0], 0, 0);
    glTexCoord2f(0, 0);
    glVertex3f(0, size[1], 0);
    glEnd();
    if (scn->flags[mjRND_CULL_FACE]) {
      glEnable(GL_CULL_FACE);
    }
    break;

  case mjGEOM_FLEX:                           // flex
  {
    // no texture for vertices and edges
    GLboolean texture_is_enabled = glIsEnabled(GL_TEXTURE_2D);
    if (texture_is_enabled == GL_TRUE) {
      glDisable(GL_TEXTURE_2D);
    }

    // vertex spheres
    if (size[0]>0 && geom->dataid<=0 && scn->flexvertopt &&
        !(scn->flexskinopt && scn->flexfaceused[geom->objid])) {
      for (int v=scn->flexvertadr[geom->objid];
               v<scn->flexvertadr[geom->objid]+scn->flexvertnum[geom->objid]; v++) {
        glPushMatrix();
        glTranslatef(scn->flexvert[3*v], scn->flexvert[3*v+1], scn->flexvert[3*v+2]);
        glScalef(size[0], size[0], size[0]);
        glCallList(con->baseBuiltin + mjrSPHERE);
        glPopMatrix();
      }
    }

    // edge cylinders
    if (size[0]>0 && geom->dataid<=0 && scn->flexedgeopt &&
        !(scn->flexskinopt && scn->flexfaceused[geom->objid])) {
      for (int e=scn->flexedgeadr[geom->objid];
               e<scn->flexedgeadr[geom->objid]+scn->flexedgenum[geom->objid]; e++) {
        // get vertices for this edge
        float* v1 = scn->flexvert + 3*(scn->flexvertadr[geom->objid]+scn->flexedge[2*e]);
        float* v2 = scn->flexvert + 3*(scn->flexvertadr[geom->objid]+scn->flexedge[2*e+1]);

        // compute legth and rotation matrix
        mjtNum vec[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
        mjtNum len = mju_normalize3(vec);
        mjtNum edgequat[4], edgemat[9];
        mju_quatZ2Vec(edgequat, vec);
        mju_negQuat(edgequat, edgequat);
        mju_quat2Mat(edgemat, edgequat);
        mat[0] = (float)edgemat[0];
        mat[1] = (float)edgemat[1];
        mat[2] = (float)edgemat[2];
        mat[4] = (float)edgemat[3];
        mat[5] = (float)edgemat[4];
        mat[6] = (float)edgemat[5];
        mat[8] = (float)edgemat[6];
        mat[9] = (float)edgemat[7];
        mat[10] = (float)edgemat[8];

        // transform and render
        glPushMatrix();
        glTranslatef((v1[0]+v2[0])*0.5f, (v1[1]+v2[1])*0.5f, (v1[2]+v2[2])*0.5f);
        glMultMatrixf(mat);
        glScalef(size[0], size[0], (float)(len*0.5));
        glCallList(con->baseBuiltin + mjrCYLINDEROPEN);
        glPopMatrix();
      }
    }

    // restore texture for faces
    if (texture_is_enabled == GL_TRUE) {
      glEnable(GL_TEXTURE_2D);
    }

    // face triangles
    int faceadr = scn->flexfaceadr[geom->objid] + (geom->dataid >= 0 ? (int)geom->size[1] : 0);
    int facenum = (geom->dataid >= 0) ? (int)geom->size[2] : scn->flexfaceused[geom->objid];
    if (facenum) {
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);
      glVertexPointer(3, GL_FLOAT, 0, scn->flexface + 9*faceadr);
      glNormalPointer(GL_FLOAT, 0, scn->flexnormal + 9*faceadr);
      if (geom->texcoord && geom->matid>=0) {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glTexCoordPointer(2, GL_FLOAT, 0, scn->flextexcoord + 6*faceadr);
      }
      glDrawArrays(GL_TRIANGLES, 0, 3*facenum);
      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_NORMAL_ARRAY);
      if (geom->texcoord && geom->matid>=0) {
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
      }
    }
    break;
  }

  case mjGEOM_SKIN:                           // skin
    // vertex positions
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, con->skinvertVBO[geom->objid]);
    glVertexPointer(3, GL_FLOAT, 0, NULL);

    // vertex normals
    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, con->skinnormalVBO[geom->objid]);
    glNormalPointer(GL_FLOAT, 0, NULL);

    // vertex texture coordinates
    if (con->skintexcoordVBO[geom->objid]) {
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      glBindBuffer(GL_ARRAY_BUFFER, con->skintexcoordVBO[geom->objid]);
      glTexCoordPointer(2, GL_FLOAT, 0, NULL);
    } else {
      glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    // triangle face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, con->skinfaceVBO[geom->objid]);

    // draw
    glDrawElements(GL_TRIANGLES, 3*scn->skinfacenum[geom->objid], GL_UNSIGNED_INT, NULL);

    // restore
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    break;
  }

  // undo coordinate transformation
  glPopMatrix();

  // disable texture if enabled
  if (geom->matid >= 0 &&
      (mode == mjrRND_NORMAL || mode == mjrRND_SHADOWMAP)) {
    settexture(mjtexREGULAR, 0, con, geom);
  }
}



void renderGeomReflection(int id, float reflectance, float headpos[3],
                          mjvScene* scn, const mjrContext* con) {
  float old_rgb[3];

  // save rgb, modulate rgb by this->reflectance
  for (int k=0; k < 3; k++) {
    old_rgb[k] = scn->geoms[id].rgba[k];
    scn->geoms[id].rgba[k] *= reflectance;
  }

  // render
  renderGeom(scn->geoms+id, mjrRND_NORMAL, headpos, scn, con);

  // restore rgb
  for (int k=0; k < 3; k++) {
    scn->geoms[id].rgba[k] = old_rgb[k];
  }
}



//----------------------------- high-level 3D rendering --------------------------------------------

// init, with or without special effects
static void initGL3(const mjvScene* scn, const mjrContext* con) {
  float rgbaWhite[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  // special effects
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_LIGHTING);
  if (scn->flags[mjRND_FOG]) {
    glEnable(GL_FOG);
  } else {
    glDisable(GL_FOG);
  }

  // common options
  glDisable(GL_BLEND);
  glEnable(GL_NORMALIZE);
  if (mjGLAD_GL_ARB_clip_control) {
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
  }
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  if (scn->flags[mjRND_CULL_FACE]) {
    glEnable(GL_CULL_FACE);
  } else {
    glDisable(GL_CULL_FACE);
  }
  glShadeModel(GL_SMOOTH);
  glDepthFunc(GL_GEQUAL);
  glDepthRange(0, 1);
  glAlphaFunc(GL_GEQUAL, 0.99f);
  glClearColor(0, 0, 0, 0);
  glClearDepth(0);
  glClearStencil(0);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

  // polygon mode
  glPolygonMode(GL_FRONT_AND_BACK, scn->flags[mjRND_WIREFRAME] ? GL_LINE : GL_FILL);
  glLineWidth(con->lineWidth);
  glFrontFace(GL_CCW);

  // fixed material properties; the rest are set by glColor (track material)
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, rgbaWhite);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
}



// light type is supported by the fixed-function pipeline
static int isSupportedLight(const mjvLight* light) {
  return light->type == mjLIGHT_DIRECTIONAL || light->type == mjLIGHT_SPOT;
}



// init lights
static int initLights(mjvScene* scn, int* light_id) {
  GLint max_lights = 0;
  glGetIntegerv(GL_MAX_LIGHTS, &max_lights);
  max_lights = mjMAX(0, mjMIN(8, max_lights));

  // count supported lights, compact into hardware slots
  int n = mjMIN(mjMAXLIGHT, scn->nlight);
  int nsupported = 0;
  for (int i=0; i < n && nsupported < max_lights; i++) {
    // ignore unsupported light types: mjLIGHT_POINT, mjLIGHT_IMAGE
    if (isSupportedLight(scn->lights+i)) {
      light_id[nsupported++] = i;
    }
  }

  // create some ambient light if no supported lights are present
  float global = nsupported ? 0 : 0.3f;
  float rgba_global[4] = {global, global, global, 1};

  // init light model
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, rgba_global);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);

  // set light properties
  for (int i=0; i < nsupported; i++) {
    const mjvLight* light = scn->lights+light_id[i];

    // colors
    glLightfv(GL_LIGHT0+i, GL_AMBIENT, light->ambient);
    glLightfv(GL_LIGHT0+i, GL_DIFFUSE, light->diffuse);
    glLightfv(GL_LIGHT0+i, GL_SPECULAR, light->specular);

    // parameters for directional light
    if (light->type == mjLIGHT_DIRECTIONAL) {
      glLightf(GL_LIGHT0+i, GL_SPOT_EXPONENT,         0);
      glLightf(GL_LIGHT0+i, GL_SPOT_CUTOFF,           180);
      glLightf(GL_LIGHT0+i, GL_CONSTANT_ATTENUATION,  1);
      glLightf(GL_LIGHT0+i, GL_LINEAR_ATTENUATION,    0);
      glLightf(GL_LIGHT0+i, GL_QUADRATIC_ATTENUATION, 0);
    }

    // parameters for spot light
    else if (light->type == mjLIGHT_SPOT) {
      glLightf(GL_LIGHT0+i, GL_SPOT_EXPONENT,         light->exponent);
      glLightf(GL_LIGHT0+i, GL_SPOT_CUTOFF,           light->cutoff);
      glLightf(GL_LIGHT0+i, GL_CONSTANT_ATTENUATION,  light->attenuation[0]);
      glLightf(GL_LIGHT0+i, GL_LINEAR_ATTENUATION,    light->attenuation[1]);
      glLightf(GL_LIGHT0+i, GL_QUADRATIC_ATTENUATION, light->attenuation[2]);
    }
  }

  // disable all lights (enable selectively in render)
  for (int i=0; i < max_lights; i++) {
    glDisable(GL_LIGHT0+i);
  }

  return nsupported;
}



// set projection and modelview
static void setView(int view, mjrRect viewport, const mjvScene* scn, const mjrContext* con,
                    float camProject[16], float camView[16]) {
  mjvGLCamera cam;

  // copy specified camera for stereo, average for mono (view = -1)
  if (view >= 0) {
    cam = scn->camera[view];
  } else {
    cam = mjv_averageCamera(scn->camera, scn->camera+1);
  }

  // compute frustum halfwidth so as to match viewport aspect ratio
  float halfwidth = cam.frustum_width ? cam.frustum_width
                    : 0.5f * (float)viewport.width / (float)viewport.height *
                             (cam.frustum_top - cam.frustum_bottom);

  // prepare projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  if (mjGLAD_GL_ARB_clip_control) {
    // reverse Z rendering mapping [znear, zfar] -> [1, 0] (ndc)
    glTranslatef(0.0f, 0.0f, 0.5f);
    glScalef(1.0f, 1.0f, -0.5f);
  } else {
    // reverse Z rendering mapping without shift [znear, zfar] -> [1, -1] (ndc)
    glScalef(1.0f, 1.0f, -1.0f);
  }

  // set projection, orthographic or perspective
  if (cam.orthographic) {
    glOrtho(cam.frustum_center - halfwidth,
            cam.frustum_center + halfwidth,
            cam.frustum_bottom,
            cam.frustum_top,
            cam.frustum_near,
            cam.frustum_far);
  } else {
    glFrustum(cam.frustum_center - halfwidth,
              cam.frustum_center + halfwidth,
              cam.frustum_bottom,
              cam.frustum_top,
              cam.frustum_near,
              cam.frustum_far);
  }

  // save projection matrix if requested
  if (camProject) {
    glGetFloatv(GL_PROJECTION_MATRIX, camProject);
  }

  // set modelview
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  mjr_lookAt(cam.pos, cam.forward, cam.up);
  if (scn->enabletransform) {
    mjr_transform(scn->translate, scn->rotate, scn->scale);
  }

  // save modelview matrix if requested
  if (camView) {
    glGetFloatv(GL_MODELVIEW_MATRIX, camView);
  }
}



// comparison function for geom sorting
static inline int geomcmp(int* i, int* j, void* context) {
  mjvGeom* geom = (mjvGeom*) context;
  float d1 = geom[*i].camdist;
  float d2 = geom[*j].camdist;

  if (d1 < d2) {
    return -1;
  } else if (d1 == d2) {
    return 0;
  } else {
    return 1;
  }
}

// define geomSort function for sorting geoms
mjSORT(geomSort, int, geomcmp);



// adjust light n position and direction
static void adjustLight(const mjvLight* thislight, int n) {
  float temp[4];

  // set position and direction according to type
  if (thislight->type == mjLIGHT_DIRECTIONAL) {
    mjr_setf4(temp, -thislight->dir[0], -thislight->dir[1], -thislight->dir[2], 0);
    glLightfv(GL_LIGHT0+n, GL_POSITION, temp);
  } else if (thislight->type == mjLIGHT_SPOT) {
    mjr_setf4(temp, thislight->dir[0], thislight->dir[1], thislight->dir[2], 0);
    glLightfv(GL_LIGHT0+n, GL_SPOT_DIRECTION, temp);
    mjr_setf4(temp, thislight->pos[0], thislight->pos[1], thislight->pos[2], 1);
    glLightfv(GL_LIGHT0+n, GL_POSITION, temp);
  } else {
    // ignore unsupported light types: mjLIGHT_POINT, mjLIGHT_IMAGE
  }
}



// render
void mjr_render(mjrRect viewport, mjvScene* scn, const mjrContext* con) {
  int stereo, nt, ngeom = scn->ngeom, nlight;
  int light_id[8];
  unsigned int drawbuffer;
  mjvGLCamera cam;
  mjtNum hpos[3];
  float temp[4], headpos[3], skyboxdst;
  float camProject[16], camView[16], lightProject[16], lightView[16];
  double clipplane[4];
  float biasMatrix[16] = {
    0.5f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.5f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.5f, 0.5f, 0.0f, 1.0f
  };
  if (!mjGLAD_GL_ARB_clip_control) {
    // account for conversion from ndc to window coordinates
    biasMatrix[2*4+2] = 0.5;
    biasMatrix[3*4+2] = 0.5;
  }

  float tempMatrix[16], textureMatrix[16];
  mjvGeom *thisgeom, tempgeom;
  mjvLight *thislight;

  // empty viewport: nothing to do
  if (viewport.width <= 0 || viewport.height <= 0) {
    return;
  }

  // average cameras
  cam = mjv_averageCamera(scn->camera, scn->camera+1);

  // check znear
  if (cam.frustum_near < mjMINVAL) {
    // geoms: error
    if (scn->ngeom) {
      mju_error("mjvScene frustum_near too small in mjr_render");
    }

    // no geoms: return silently
    else {
      return;
    }
  }

  // upload dynamic skin data to GPU
  for (int i=0; i < scn->nskin; i++) {
    // upload positions to VBO
    glBindBuffer(GL_ARRAY_BUFFER, con->skinvertVBO[i]);
    glBufferData(GL_ARRAY_BUFFER,
                 3*scn->skinvertnum[i]*sizeof(float),
                 scn->skinvert + 3*scn->skinvertadr[i],
                 GL_STREAM_DRAW);

    // upload normals to VBO
    glBindBuffer(GL_ARRAY_BUFFER, con->skinnormalVBO[i]);
    glBufferData(GL_ARRAY_BUFFER,
                 3*scn->skinvertnum[i]*sizeof(float),
                 scn->skinnormal + 3*scn->skinvertadr[i],
                 GL_STREAM_DRAW);
  }
  // avoid leaking VBO bindings to subsequent client-array draws (e.g. flex geoms)
  if (scn->nskin > 0) {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  // determine drawbuffer; may be changed by stereo later
  if (con->currentBuffer == mjFB_WINDOW) {
    drawbuffer = (con->windowDoublebuffer ? GL_BACK : GL_FRONT);
  } else {
    drawbuffer = GL_COLOR_ATTACHMENT0;
  }

  // init lights
  nlight = initLights(scn, light_id);

  // compute head position in model space
  mjv_cameraInModel(hpos, NULL, NULL, scn);
  mju_n2f(headpos, hpos, 3);

  // make list of transparent geoms
  nt = 0;
  for (int i=0; i < ngeom; i++) {
    // get geom pointer
    thisgeom = scn->geoms + i;

    if (thisgeom->rgba[3] < 0.995 || (thisgeom->type == mjGEOM_PLANE &&
                                      isBehind(headpos, thisgeom->pos, thisgeom->mat))) {
      // include index in list
      scn->geomorder[nt++] = i;
      thisgeom->transparent = 1;

      // compute distance to camera
      thisgeom->camdist = sqrtf((thisgeom->pos[0]-headpos[0])*(thisgeom->pos[0]-headpos[0]) +
                                (thisgeom->pos[1]-headpos[1])*(thisgeom->pos[1]-headpos[1]) +
                                (thisgeom->pos[2]-headpos[2])*(thisgeom->pos[2]-headpos[2]));

      // correct for rbound
      thisgeom->camdist -= mjv_rbound(thisgeom);

      // plane always far away
      if (thisgeom->type == mjGEOM_PLANE) {
        thisgeom->camdist = 1E+10;
      }
    } else {
      thisgeom->transparent = 0;
    }
  }

  // sort transparent geoms according to distance to camera
  if (nt > 1) {
    int *buf = (int*) mju_malloc(nt * sizeof(int));
    geomSort(scn->geomorder, buf, nt, scn->geoms);
    mju_free(buf);
  }

  // allow only one reflective geom
  int reflective_id = -1;
  for (int i=0; i < ngeom; i++) {
    if (isReflective(scn->geoms+i)) {
      reflective_id = i;
      break;
    }
  }

  // init OpenGL
  initGL3(scn, con);

  // set full viewport
  glViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);

  // clear, with scissor
  glScissor(viewport.left, viewport.bottom, viewport.width, viewport.height);
  glEnable(GL_SCISSOR_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_SCISSOR_TEST);

  // determine stereo; quadbuffered reverts to sidebyside if hardware not available
  stereo = scn->stereo;
  if (stereo == mjSTEREO_QUADBUFFERED &&
      (con->currentBuffer != mjFB_WINDOW || !con->windowStereo)) {
    stereo = mjSTEREO_SIDEBYSIDE;
  }

  // SIDEBYSIDE: reduce viewport
  mjrRect full_viewport = viewport;
  if (stereo == mjSTEREO_SIDEBYSIDE) {
    viewport.width /= 2;
  }

  // render with stereo
  for (int view = (stereo ? 0 : -1); view < (stereo ? 2 : 0); view++) {
    // change drawbuffer for QUADBUFFERED stereo
    if (stereo == mjSTEREO_QUADBUFFERED) {
      if (con->windowDoublebuffer) {
        drawbuffer = (view ? GL_BACK_RIGHT : GL_BACK_LEFT);
      } else {
        drawbuffer = (view ? GL_FRONT_RIGHT : GL_FRONT_LEFT);
      }
      glDrawBuffer(drawbuffer);

      // clear depth buffer for second view (since it is shared)
      if (view) {
        glClear(GL_DEPTH_BUFFER_BIT);
      }
    }

    // change viewport for SIDEBYSIDE stereo
    else if (stereo == mjSTEREO_SIDEBYSIDE) {
      // move viewport to the right for view 1
      if (view) {
        viewport.left += viewport.width;
      }

      // set reduced/moved viewport
      glViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);
    }

    // set projection and modelview
    setView(view, viewport, scn, con, camProject, camView);

    //---------------------------------- segmentation rendering

    if (scn->flags[mjRND_SEGMENT]) {
      // constant color rendering
      glShadeModel(GL_FLAT);
      glDisable(GL_LIGHTING);
      glDisable(GL_COLOR_MATERIAL);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glDisable(GL_FOG);
      glDisable(GL_MULTISAMPLE);

      // render all geoms
      for (int i=0; i < ngeom; i++) {
        if (scn->geoms[i].segid >= 0) {
          renderGeom(scn->geoms+i,
                     scn->flags[mjRND_IDCOLOR] ? mjrRND_IDCOLOR : mjrRND_SEGMENT,
                     headpos, scn, con);
        }
      }

      // skip the remaining passes
      continue;
    }

    //---------------------------------- reflection rendering

    // plane and box reflection rendering
    if (scn->flags[mjRND_REFLECTION] && !scn->flags[mjRND_DEPTH]) {
      for (int i=0; i < ngeom; i++) {
        // get geom pointer
        thisgeom = scn->geoms + i;

        if (i == reflective_id) {
          // if box, replace with temporary plane matching Z+ box side
          if (thisgeom->type == mjGEOM_BOX) {
            // copy and convert to plane
            tempgeom = *thisgeom;
            tempgeom.type = mjGEOM_PLANE;

            // offset position to Z+ face side
            tempgeom.pos[0] += tempgeom.size[2]*tempgeom.mat[2];
            tempgeom.pos[1] += tempgeom.size[2]*tempgeom.mat[5];
            tempgeom.pos[2] += tempgeom.size[2]*tempgeom.mat[8];

            // redirect pointer
            thisgeom = &tempgeom;
          }

          // camera behind plane or box side: nothing to show
          if (isBehind(headpos, thisgeom->pos, thisgeom->mat)) {
            continue;
          }

          // prepare to render plane in stencil buffer
          glDisable(GL_DEPTH_TEST);
          glColorMask(0, 0, 0, 0);
          glEnable(GL_STENCIL_TEST);
          glClear(GL_STENCIL_BUFFER_BIT);
          glStencilFunc(GL_ALWAYS, 1, 0xFF);
          glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);

          // render thisgeom in stencil buffer, always fill
          glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
          renderGeom(thisgeom, mjrRND_NORMAL, headpos, scn, con);
          glPolygonMode(GL_FRONT_AND_BACK, scn->flags[mjRND_WIREFRAME] ? GL_LINE : GL_FILL);

          // prepare to render scene where stencil==1
          glEnable(GL_DEPTH_TEST);
          glColorMask(1, 1, 1, 1);
          glStencilFunc(GL_EQUAL, 1, 0xFF);
          glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

          // set clipplane
          glEnable(GL_CLIP_PLANE0);
          clipplane[0] = -thisgeom->mat[2];
          clipplane[1] = -thisgeom->mat[5];
          clipplane[2] = -thisgeom->mat[8];
          clipplane[3] =
            thisgeom->pos[0]*thisgeom->mat[2] +
            thisgeom->pos[1]*thisgeom->mat[5] +
            thisgeom->pos[2]*thisgeom->mat[8];
          glClipPlane(GL_CLIP_PLANE0, clipplane);

          // set reflection matrix
          glFrontFace(GL_CW);
          glPushMatrix();
          mjr_reflect(thisgeom->pos, thisgeom->mat);

          // set light position and direction, enable supported lights
          for (int j=0; j < nlight; j++) {
            adjustLight(scn->lights+light_id[j], j);
            glEnable(GL_LIGHT0+j);
          }

          // render reflected non-transparent geoms, except for thisgeom
          for (int j=0; j < ngeom; j++) {
            if (!scn->geoms[j].transparent && i != j) {
              renderGeomReflection(j, thisgeom->reflectance, headpos, scn, con);
            }
          }

          // render reflected transparent geoms, except for thisgeom
          glDepthMask(GL_FALSE);
          glEnable(GL_BLEND);
          if (scn->flags[mjRND_ADDITIVE]) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
          } else {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
          }
          for (int j=0; j < nt; j++) {
            if (i != scn->geomorder[j]) {
              renderGeomReflection(scn->geomorder[j], thisgeom->reflectance, headpos, scn, con);
            }
          }
          if (!scn->flags[mjRND_ADDITIVE]) {
            for (int j=nt-1; j >= 0; j--) {
              if (i != scn->geomorder[j]) {
                renderGeomReflection(scn->geomorder[j], thisgeom->reflectance, headpos, scn, con);
              }
            }
          }
          glDepthMask(GL_TRUE);
          glDisable(GL_BLEND);

          // disable lights
          for (int j=0; j < nlight; j++) {
            glDisable(GL_LIGHT0+j);
          }

          // end reflection rendering
          glDisable(GL_STENCIL_TEST);
          glDisable(GL_CLIP_PLANE0);
          glPopMatrix();
          glFrontFace(GL_CCW);
        }
      }
    }

    //---------------------------------- regular rendering

    // set light position and direction, enable non-shadow lights
    for (int i=0; i < nlight; i++) {
      thislight = scn->lights+light_id[i];

      // set light
      adjustLight(thislight, i);

      // enable lights without shadows
      if (!thislight->castshadow || !(scn->flags[mjRND_SHADOW] && con->shadowFBO)) {
        glEnable(GL_LIGHT0+i);
      }
    }

    // render reflecting geoms
    //  (only one allowed, more would result in weird transparency)
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    if (reflective_id >= 0) {
      renderGeom(scn->geoms+reflective_id, mjrRND_NORMAL, headpos, scn, con);
    }
    glDisable(GL_BLEND);

    // render remaining opaque geoms
    for (int i=0; i < ngeom; i++) {
      if (!scn->geoms[i].transparent && i != reflective_id) {
        renderGeom(scn->geoms+i, mjrRND_NORMAL, headpos, scn, con);
      }
    }

    // disable lights
    for (int i=0; i < nlight; i++) {
      glDisable(GL_LIGHT0+i);
    }

    //------------------------------------ shadow rendering

    // black fog, to avoid glow
    float black[4] = {0, 0, 0, 0};
    glFogfv(GL_FOG_COLOR, black);

    // shadow map rendering
    if (scn->flags[mjRND_SHADOW] && con->shadowFBO) {
      for (int i=0; i < nlight; i++) {
        thislight = scn->lights+light_id[i];

        if (thislight->castshadow) {
          // prepare up-direction
          mjr_orthoVec(temp, thislight->dir);

          // set projection: from light viewpoint
          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          if (mjGLAD_GL_ARB_clip_control) {
            // reverse Z rendering mapping [znear, zfar] -> [1, 0] (ndc)
            glTranslatef(0.0f, 0.0f, 0.5f);
            glScalef(1.0f, 1.0f, -0.5f);
          }
          else {
            // reverse Z rendering mapping without shift [znear, zfar] -> [1, -1] (ndc)
            glScalef(1.0f, 1.0f, -1.0f);
          }
          if (thislight->type == mjLIGHT_DIRECTIONAL) {
            glOrtho(-con->shadowClip, con->shadowClip,
                    -con->shadowClip, con->shadowClip,
                    cam.frustum_near, cam.frustum_far);
          } else if (thislight->type == mjLIGHT_SPOT) {
            mjr_perspective(mju_min(2*thislight->cutoff*con->shadowScale, 160), 1,
                            cam.frustum_near, cam.frustum_far);
          }
          glGetFloatv(GL_PROJECTION_MATRIX, lightProject);

          // set modelview: from light viewpoint
          glMatrixMode(GL_MODELVIEW);
          glLoadIdentity();
          mjr_lookAt(thislight->pos, thislight->dir, temp);
          glGetFloatv(GL_MODELVIEW_MATRIX, lightView);

          // adjust OpenGL settings for shadow rendering
          glBindFramebuffer(GL_FRAMEBUFFER, con->shadowFBO);
          glDrawBuffer(GL_NONE);
          glClear(GL_DEPTH_BUFFER_BIT);
          glViewport(
              1, 1, con->shadowSize-2, con->shadowSize-2);  // avoid infinite shadows from edges
          glShadeModel(GL_FLAT);
          glDisable(GL_LIGHTING);
          glColorMask(0, 0, 0, 0);
          int cull_face = glIsEnabled(GL_CULL_FACE);
          glDisable(GL_CULL_FACE);  // all faces cast shadows
          glEnable(GL_POLYGON_OFFSET_FILL);

          // The limited resolution of the shadow maps means multiple fragments
          // sample the same texel. When light and camera directions differ on
          // surfaces that should be lit this causes "shadow acne"  because some
          // fragments will be lit while adjacent fragments are not. To mitigate
          // this artifact, an offset is applied to the depth values in the
          // shadow map. The offset must be large enough to ensure consistent
          // depth comparison occurs within the limited precision of the depth
          // buffer. The offset is computed by glPolygonOffset using parameters
          // that are chosen empirically. We need different values when clip
          // control is on/off because this setting changes the depth precision.
          float kOffsetFactor = -16.0f;
          float kOffsetUnits = -512.0f;
          if (mjGLAD_GL_ARB_clip_control) {
            kOffsetFactor = -1.5f;
            kOffsetUnits = -4.0f;
          }
          glPolygonOffset(kOffsetFactor, kOffsetUnits);

          // render all geoms to depth texture
          for (int j=0; j < ngeom; j++) {
            renderGeom(scn->geoms+j, mjrRND_SHADOWCAST, headpos, scn, con);
          }

          // restore OpenGL settings
          glBindFramebuffer(GL_FRAMEBUFFER,
                            con->currentBuffer == mjFB_WINDOW ? 0 : con->offFBO);
          glDrawBuffer(drawbuffer);
          glViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);
          if (cull_face) {
            glEnable(GL_CULL_FACE);
          }
          glDisable(GL_POLYGON_OFFSET_FILL);
          glShadeModel(GL_SMOOTH);
          glEnable(GL_LIGHTING);
          glColorMask(1, 1, 1, 1);
          glMatrixMode(GL_PROJECTION);
          glLoadMatrixf(camProject);
          glMatrixMode(GL_MODELVIEW);
          glLoadMatrixf(camView);

          // compute camera-light mapping
          mjr_mulMat44(tempMatrix, lightProject, lightView);
          mjr_mulMat44(textureMatrix, biasMatrix, tempMatrix);

          // enable texture machinery
          settexture(mjtexSHADOW, 1, con, 0);
          mjr_getrow4(temp, textureMatrix, 0);
          glTexGenfv(GL_S, GL_EYE_PLANE, temp);
          mjr_getrow4(temp, textureMatrix, 1);
          glTexGenfv(GL_T, GL_EYE_PLANE, temp);
          mjr_getrow4(temp, textureMatrix, 2);
          glTexGenfv(GL_R, GL_EYE_PLANE, temp);
          mjr_getrow4(temp, textureMatrix, 3);
          glTexGenfv(GL_Q, GL_EYE_PLANE, temp);

          // render with shadow light i
          glEnable(GL_ALPHA_TEST);
          glEnable(GL_BLEND);
          glBlendFunc(GL_ONE, GL_ONE);
          glEnable(GL_LIGHT0+i);

          // only opaque geoms accept shadows
          for (int j=0; j < ngeom; j++) {
            if (!scn->geoms[j].transparent) {
              renderGeom(scn->geoms+j, mjrRND_SHADOWMAP, headpos, scn, con);
            }
          }

          glDisable(GL_LIGHT0+i);
          glDisable(GL_BLEND);
          glDisable(GL_ALPHA_TEST);

          // disable texture machinery
          settexture(mjtexSHADOW, 0, con, 0);
        }
      }
    }

    // restore fog color
    glFogfv(GL_FOG_COLOR, con->fogRGBA);

    //------------------------------------ skybox rendering

    if (scn->flags[mjRND_SKYBOX]) {
      // skybox always filled
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      // find skybox texture
      for (int i=0; i < con->ntexture; i++) {
        if (con->textureType[i] == mjTEXTURE_SKYBOX) {
          // save first skybox texture id in tempgeom
          memset(&tempgeom, 0, sizeof(mjvGeom));
          tempgeom.texid = i;

          // modify settings
          glDisable(GL_LIGHTING);
          glDisable(GL_COLOR_MATERIAL);
          glCullFace(GL_FRONT);

          // center at headpos, scale to avoid far clipping: just below 1/sqrt(2)
          glPushMatrix();
          glTranslatef(headpos[0], headpos[1], headpos[2]);
          skyboxdst = cam.frustum_far*0.70f;
          if (scn->enabletransform) {
            skyboxdst /= scn->scale;
          }
          glScalef(skyboxdst, skyboxdst, skyboxdst);

          // render cylinder, with skybox texture
          settexture(mjtexSKYBOX, 1, con, &tempgeom);
          glColor4f(1, 1, 1, 1);
          glCallList(con->baseBuiltin + mjrCYLINDER);
          settexture(mjtexSKYBOX, 0, con, &tempgeom);

          // haze
          if (scn->flags[mjRND_HAZE]) {
            for (int j=0; j < ngeom; j++) {
              if (scn->geoms[j].type == mjGEOM_PLANE &&
                  scn->geoms[j].size[0] == 0 &&
                  scn->geoms[j].size[1] == 0) {
                // compute headpos elevation above plane
                float* mat3 = scn->geoms[j].mat;
                float elevation = ((headpos[0]-scn->geoms[j].pos[0])*mat3[2] +
                                   (headpos[1]-scn->geoms[j].pos[1])*mat3[5] +
                                   (headpos[2]-scn->geoms[j].pos[2])*mat3[8]) /
                                  skyboxdst;

                // below plane: no rendering
                if (elevation < 0) {
                  break;
                }

                // rotate to plane
                float mat4[16] = {
                  mat3[0], mat3[3], mat3[6], 0.0f,
                  mat3[1], mat3[4], mat3[7], 0.0f,
                  mat3[2], mat3[5], mat3[8], 0.0f,
                  0.0f,    0.0f,    0.0f,    1.0f
                };
                glMultMatrixf(mat4);

                // translate and scale
                glTranslatef(0, 0, -elevation);
                glScalef(1, 1, elevation);

                // render
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                glCallList(con->baseBuiltin + mjrHAZE);
                glDisable(GL_BLEND);

                // only first infinite plane
                break;
              }
            }
          }

          // recover settings
          glPopMatrix();
          glEnable(GL_LIGHTING);
          glEnable(GL_COLOR_MATERIAL);
          glCullFace(GL_BACK);

          // render only first skybox
          break;
        }
      }

      // wireframe or filled depending on flag
      glPolygonMode(GL_FRONT_AND_BACK, scn->flags[mjRND_WIREFRAME] ? GL_LINE : GL_FILL);
    }

    //------------------------------------ transparent regular rendering

    // enable supported lights
    for (int i=0; i < nlight; i++) {
      glEnable(GL_LIGHT0+i);
    }

    // blend mode
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    if (scn->flags[mjRND_ADDITIVE]) {
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    } else {
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // render transparent geoms: front to back
    for (int i=0; i < nt; i++) {
      renderGeom(scn->geoms+scn->geomorder[i], mjrRND_NORMAL, headpos, scn, con);
    }

    // render transparent geoms: back to front, if not additive
    if (!scn->flags[mjRND_ADDITIVE]) {
      for (int i=nt-1; i >= 0; i--) {
        renderGeom(scn->geoms+scn->geomorder[i], mjrRND_NORMAL, headpos, scn, con);
      }
    }

    // normal mode
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // disable lights
    for (int i=0; i < nlight; i++) {
      glDisable(GL_LIGHT0+i);
    }

    //------------------------------------ label rendering

    // render text labels if present
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    for (int i=0; i < ngeom; i++) {
      thisgeom = scn->geoms + i;

      if (thisgeom->label[0]) {
        mjr_textActual(mjFONT_SHADOW, thisgeom->label, con,
                       thisgeom->pos[0], thisgeom->pos[1], thisgeom->pos[2], 1, 1, 1);
      }
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
  }

  // restore full viewport after stereo rendering
  if (stereo == mjSTEREO_SIDEBYSIDE) {
    viewport = full_viewport;
    glViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);
  }

  // frame
  if (scn->framewidth) {
    // prepare
    glClearColor(scn->framergb[0], scn->framergb[1], scn->framergb[2], 0);
    glEnable(GL_SCISSOR_TEST);

    // bottom edge
    glScissor(viewport.left, viewport.bottom,
              viewport.width, scn->framewidth);
    glClear(GL_COLOR_BUFFER_BIT);

    // top edge
    glScissor(viewport.left, viewport.bottom+viewport.height-scn->framewidth,
              viewport.width, scn->framewidth);
    glClear(GL_COLOR_BUFFER_BIT);

    // left edge
    glScissor(viewport.left, viewport.bottom,
              scn->framewidth, viewport.height);
    glClear(GL_COLOR_BUFFER_BIT);

    // right edge
    glScissor(viewport.left+viewport.width-scn->framewidth, viewport.bottom,
              scn->framewidth, viewport.height);
    glClear(GL_COLOR_BUFFER_BIT);

    // disable scissor
    glDisable(GL_SCISSOR_TEST);
  }

  if (scn->flags[mjRND_DEPTH]) {
    glPushAttrib(GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_TEXTURE_BIT |
                 GL_LIGHTING_BIT | GL_POLYGON_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    void* depth_buf = mju_malloc(sizeof(float) * viewport.width * viewport.height);
    glReadPixels(viewport.left, viewport.bottom, viewport.width, viewport.height,
                 GL_DEPTH_COMPONENT, GL_FLOAT, depth_buf);

    GLuint depth_tex;
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &depth_tex);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, viewport.width, viewport.height,
                 0, GL_LUMINANCE, GL_FLOAT, depth_buf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();

    glDeleteTextures(1, &depth_tex);
    mju_free(depth_buf);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
  }

  // restore currentBuffer
  mjr_restoreBuffer(con);
}



// call glFinish
void mjr_finish(void) {
  glFinish();
}



// call glGetError and return result
int mjr_getError(void) {
  return (int)glGetError();
}
