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

#ifndef MUJOCO_MJVISUALIZE_H_
#define MUJOCO_MJVISUALIZE_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>


#define mjNGROUP        6         // number of geom, site, joint, skin groups with visflags
#define mjMAXLIGHT      100       // maximum number of lights in a scene
#define mjMAXOVERLAY    500       // maximum number of characters in overlay text
#define mjMAXLINE       100       // maximum number of lines per plot
#define mjMAXLINEPNT    1001      // maximum number points per line
#define mjMAXPLANEGRID  200       // maximum number of grid divisions for plane


//---------------------------------- primitive types (mjt) -----------------------------------------

typedef enum mjtCatBit_ {         // bitflags for mjvGeom category
  mjCAT_STATIC        = 1,        // model elements in body 0
  mjCAT_DYNAMIC       = 2,        // model elements in all other bodies
  mjCAT_DECOR         = 4,        // decorative geoms
  mjCAT_ALL           = 7         // select all categories
} mjtCatBit;


typedef enum mjtMouse_ {          // mouse interaction mode
  mjMOUSE_NONE        = 0,        // no action
  mjMOUSE_ROTATE_V,               // rotate (orbit) vertically
  mjMOUSE_ROTATE_H,               // rotate (orbit) horizontally
  mjMOUSE_MOVE_V,                 // move along vertical plane
  mjMOUSE_MOVE_H,                 // move along horizontal plane
  mjMOUSE_ZOOM,                   // zoom (towards/away from lookat point)
  mjMOUSE_MOVE_V_REL,             // move (truck, pedestal), vertical plane rel. to target
  mjMOUSE_MOVE_H_REL,             // move (truck, dolly), horizontal plane rel. to target
  mjMOUSE_TURN_V,                 // turn (tilt) vertically
  mjMOUSE_TURN_H,                 // turn (pan) horizontally
} mjtMouse;


typedef enum mjtPertBit_ {        // mouse perturbations
  mjPERT_TRANSLATE    = 1,        // translation
  mjPERT_ROTATE       = 2         // rotation
} mjtPertBit;


typedef enum mjtCamera_ {         // abstract camera type
  mjCAMERA_FREE       = 0,        // free camera
  mjCAMERA_TRACKING,              // tracking camera; uses trackbodyid
  mjCAMERA_FIXED,                 // fixed camera; uses fixedcamid
  mjCAMERA_USER                   // user is responsible for setting OpenGL camera
} mjtCamera;


typedef enum mjtLabel_ {          // object labeling
  mjLABEL_NONE        = 0,        // nothing
  mjLABEL_BODY,                   // body labels
  mjLABEL_JOINT,                  // joint labels
  mjLABEL_GEOM,                   // geom labels
  mjLABEL_SITE,                   // site labels
  mjLABEL_CAMERA,                 // camera labels
  mjLABEL_LIGHT,                  // light labels
  mjLABEL_TENDON,                 // tendon labels
  mjLABEL_ACTUATOR,               // actuator labels
  mjLABEL_CONSTRAINT,             // constraint labels
  mjLABEL_FLEX,                   // flex labels
  mjLABEL_SKIN,                   // skin labels
  mjLABEL_SELECTION,              // selected object
  mjLABEL_SELPNT,                 // coordinates of selection point
  mjLABEL_CONTACTPOINT,           // contact information
  mjLABEL_CONTACTFORCE,           // magnitude of contact force
  mjLABEL_ISLAND,                 // id of island

  mjNLABEL                        // number of label types
} mjtLabel;


typedef enum mjtFrame_ {          // frame visualization
  mjFRAME_NONE        = 0,        // no frames
  mjFRAME_BODY,                   // body frames
  mjFRAME_GEOM,                   // geom frames
  mjFRAME_SITE,                   // site frames
  mjFRAME_CAMERA,                 // camera frames
  mjFRAME_LIGHT,                  // light frames
  mjFRAME_CONTACT,                // contact frames
  mjFRAME_WORLD,                  // world frame

  mjNFRAME                        // number of visualization frames
} mjtFrame;


typedef enum mjtVisFlag_ {        // flags enabling model element visualization
  mjVIS_CONVEXHULL    = 0,        // mesh convex hull
  mjVIS_TEXTURE,                  // textures
  mjVIS_JOINT,                    // joints
  mjVIS_CAMERA,                   // cameras
  mjVIS_ACTUATOR,                 // actuators
  mjVIS_ACTIVATION,               // activations
  mjVIS_LIGHT,                    // lights
  mjVIS_TENDON,                   // tendons
  mjVIS_RANGEFINDER,              // rangefinder sensors
  mjVIS_CONSTRAINT,               // point constraints
  mjVIS_INERTIA,                  // equivalent inertia boxes
  mjVIS_SCLINERTIA,               // scale equivalent inertia boxes with mass
  mjVIS_PERTFORCE,                // perturbation force
  mjVIS_PERTOBJ,                  // perturbation object
  mjVIS_CONTACTPOINT,             // contact points
  mjVIS_ISLAND,                   // constraint islands
  mjVIS_CONTACTFORCE,             // contact force
  mjVIS_CONTACTSPLIT,             // split contact force into normal and tangent
  mjVIS_TRANSPARENT,              // make dynamic geoms more transparent
  mjVIS_AUTOCONNECT,              // auto connect joints and body coms
  mjVIS_COM,                      // center of mass
  mjVIS_SELECT,                   // selection point
  mjVIS_STATIC,                   // static bodies
  mjVIS_SKIN,                     // skin
  mjVIS_FLEXVERT,                 // flex vertices
  mjVIS_FLEXEDGE,                 // flex edges
  mjVIS_FLEXFACE,                 // flex element faces
  mjVIS_FLEXSKIN,                 // flex smooth skin (disables the rest)
  mjVIS_BODYBVH,                  // body bounding volume hierarchy
  mjVIS_MESHBVH,                  // mesh bounding volume hierarchy
  mjVIS_SDFITER,                  // iterations of SDF gradient descent

  mjNVISFLAG                      // number of visualization flags
} mjtVisFlag;


typedef enum mjtRndFlag_ {        // flags enabling rendering effects
  mjRND_SHADOW        = 0,        // shadows
  mjRND_WIREFRAME,                // wireframe
  mjRND_REFLECTION,               // reflections
  mjRND_ADDITIVE,                 // additive transparency
  mjRND_SKYBOX,                   // skybox
  mjRND_FOG,                      // fog
  mjRND_HAZE,                     // haze
  mjRND_DEPTH,                    // depth
  mjRND_SEGMENT,                  // segmentation with random color
  mjRND_IDCOLOR,                  // segmentation with segid+1 color
  mjRND_CULL_FACE,                // cull backward faces
  mjRND_POST_PROCESSING,          // post-processing (e.g. color grading, bloom, etc.)

  mjNRNDFLAG                      // number of rendering flags
} mjtRndFlag;


typedef enum mjtStereo_ {         // type of stereo rendering
  mjSTEREO_NONE       = 0,        // no stereo; use left eye only
  mjSTEREO_QUADBUFFERED,          // quad buffered; revert to side-by-side if no hardware support
  mjSTEREO_SIDEBYSIDE             // side-by-side
} mjtStereo;


//---------------------------------- mjvPerturb ----------------------------------------------------

typedef struct mjvPerturb_ {      // object selection and perturbation
  int      select;                // selected body id; non-positive: none
  int      flexselect;            // selected flex id; negative: none
  int      skinselect;            // selected skin id; negative: none
  int      active;                // perturbation bitmask (mjtPertBit)
  int      active2;               // secondary perturbation bitmask (mjtPertBit)
  mjtNum   refpos[3];             // reference position for selected object
  mjtNum   refquat[4];            // reference orientation for selected object
  mjtNum   refselpos[3];          // reference position for selection point
  mjtNum   localpos[3];           // selection point in object coordinates
  mjtNum   localmass;             // spatial inertia at selection point
  mjtNum   scale;                 // relative mouse motion-to-space scaling (set by initPerturb)
} mjvPerturb;


//---------------------------------- mjvCamera -----------------------------------------------------

typedef struct mjvCamera_ {       // abstract camera
  // type and ids
  int      type;                  // camera type (mjtCamera)
  int      fixedcamid;            // fixed camera id
  int      trackbodyid;           // body id to track

  // abstract camera pose specification
  mjtNum   lookat[3];             // lookat point
  mjtNum   distance;              // distance to lookat point or tracked body
  mjtNum   azimuth;               // camera azimuth (deg)
  mjtNum   elevation;             // camera elevation (deg)

  // orthographic / perspective
  int      orthographic;          // 0: perspective; 1: orthographic
} mjvCamera;


//---------------------------------- mjvGLCamera ---------------------------------------------------

typedef struct mjvGLCamera_ {     // OpenGL camera
  // camera frame
  float    pos[3];                // position
  float    forward[3];            // forward direction
  float    up[3];                 // up direction

  // camera projection
  float    frustum_center;        // hor. center (left,right set to match aspect)
  float    frustum_width;         // width (not used for rendering)
  float    frustum_bottom;        // bottom
  float    frustum_top;           // top
  float    frustum_near;          // near
  float    frustum_far;           // far

  // orthographic / perspective
  int      orthographic;          // 0: perspective; 1: orthographic
} mjvGLCamera;


//---------------------------------- mjvGeom -------------------------------------------------------

typedef struct mjvGeom_ {         // abstract geom
  // type info
  int      type;                  // geom type (mjtGeom)
  int      dataid;                // mesh, hfield or plane id; -1: none; mesh: 2*id or 2*id+1 (hull);
                                  // multi-material flex: material slice index
  int      objtype;               // mujoco object type; mjOBJ_UNKNOWN for decor
  int      objid;                 // mujoco object id; -1 for decor
  int      category;              // visual category
  int      matid;                 // material id; -1: no textured material
  int      texid;                 // texture id; -1: none
  int      texuniform;            // uniform cube mapping
  int      texcoord;              // mesh or flex geom has texture coordinates
  int      segid;                 // segmentation id; -1: not shown

  // spatial transform
  float    size[3];               // size parameters; multi-material flex: (radius, face offset, face count)
  float    pos[3];                // Cartesian position
  float    mat[9];                // Cartesian orientation

  // material properties
  float    rgba[4];               // color and transparency
  float    emission;              // emission coef
  float    specular;              // specular coef
  float    shininess;             // shininess coef
  float    reflectance;           // reflectance coef
  float    texrepeat[2];          // texture repetition for 2d mapping
  mjtByte  castshadow;            // does geom cast shadows

  char     label[100];            // text label

  // transparency rendering (set internally)
  float    camdist;               // distance to camera (used by sorter)
  float    modelrbound;           // geom rbound from model, 0 if not model geom
  mjtByte  transparent;           // treat geom as transparent
} mjvGeom;


//---------------------------------- mjvLight ------------------------------------------------------

typedef struct mjvLight_ {        // OpenGL light
  int      id;                    // light id, -1 for headlight
  float    pos[3];                // position rel. to body frame
  float    dir[3];                // direction rel. to body frame
  int      type;                  // type (mjtLightType)
  int      texid;                 // texture id for image lights
  float    attenuation[3];        // OpenGL attenuation (quadratic model)
  float    cutoff;                // OpenGL cutoff
  float    exponent;              // OpenGL exponent
  float    ambient[3];            // ambient rgb (alpha=1)
  float    diffuse[3];            // diffuse rgb (alpha=1)
  float    specular[3];           // specular rgb (alpha=1)
  mjtByte  headlight;             // headlight
  mjtByte  castshadow;            // does light cast shadows
  float    bulbradius;            // bulb radius for soft shadows
  float    intensity;             // intensity, in candelas
  float    range;                 // range of effectiveness
  float    softness;              // spotlight edge softness
} mjvLight;


//---------------------------------- mjvOption -----------------------------------------------------

typedef struct mjvOption_ {          // abstract visualization options
  int      label;                    // what objects to label (mjtLabel)
  int      frame;                    // which frame to show (mjtFrame)
  mjtByte  geomgroup[mjNGROUP];      // geom visualization by group
  mjtByte  sitegroup[mjNGROUP];      // site visualization by group
  mjtByte  jointgroup[mjNGROUP];     // joint visualization by group
  mjtByte  tendongroup[mjNGROUP];    // tendon visualization by group
  mjtByte  actuatorgroup[mjNGROUP];  // actuator visualization by group
  mjtByte  flexgroup[mjNGROUP];      // flex visualization by group
  mjtByte  skingroup[mjNGROUP];      // skin visualization by group
  mjtByte  flags[mjNVISFLAG];        // visualization flags (indexed by mjtVisFlag)
  int      bvh_depth;                // depth of the bounding volume hierarchy to be visualized
  int      flex_layer;               // element layer to be visualized for 3D flex
} mjvOption;


//---------------------------------- mjvScene ------------------------------------------------------

typedef struct mjvScene_ {        // abstract scene passed to OpenGL renderer
  // abstract geoms
  int      maxgeom;               // size of allocated geom buffer
  int      ngeom;                 // number of geoms currently in buffer
  mjvGeom* geoms;                 // buffer for geoms (ngeom)
  int*     geomorder;             // buffer for ordering geoms by distance to camera (ngeom)

  // flex data
  int      nflex;                 // number of flexes
  int*     flexedgeadr;           // address of flex edges (nflex)
  int*     flexedgenum;           // number of edges in flex (nflex)
  int*     flexvertadr;           // address of flex vertices (nflex)
  int*     flexvertnum;           // number of vertices in flex (nflex)
  int*     flexfaceadr;           // address of flex faces (nflex)
  int*     flexfacenum;           // number of flex faces allocated (nflex)
  int*     flexfaceused;          // number of flex faces currently in use (nflex)
  int*     flexedge;              // flex edge data (2*nflexedge)
  float*   flexvert;              // flex vertices (3*nflexvert)
  float*   flexface;              // flex faces vertices (9*sum(flexfacenum))
  float*   flexnormal;            // flex face normals (9*sum(flexfacenum))
  float*   flextexcoord;          // flex face texture coordinates (6*sum(flexfacenum))
  mjtByte  flexvertopt;           // copy of mjVIS_FLEXVERT mjvOption flag
  mjtByte  flexedgeopt;           // copy of mjVIS_FLEXEDGE mjvOption flag
  mjtByte  flexfaceopt;           // copy of mjVIS_FLEXFACE mjvOption flag
  mjtByte  flexskinopt;           // copy of mjVIS_FLEXSKIN mjvOption flag

  // skin data
  int      nskin;                 // number of skins
  int*     skinfacenum;           // number of faces in skin (nskin)
  int*     skinvertadr;           // address of skin vertices (nskin)
  int*     skinvertnum;           // number of vertices in skin (nskin)
  float*   skinvert;              // skin vertex data (3*nskinvert)
  float*   skinnormal;            // skin normal data (3*nskinvert)

  // OpenGL lights
  int      nlight;                // number of lights currently in buffer
  mjvLight lights[mjMAXLIGHT];    // buffer for lights (nlight)

  // OpenGL cameras
  mjvGLCamera camera[2];          // left and right camera

  // OpenGL model transformation
  mjtByte  enabletransform;       // enable model transformation
  float    translate[3];          // model translation
  float    rotate[4];             // model quaternion rotation
  float    scale;                 // model scaling

  // OpenGL rendering effects
  int      stereo;                // stereoscopic rendering (mjtStereo)
  mjtByte  flags[mjNRNDFLAG];     // rendering flags (indexed by mjtRndFlag)

  // framing
  int      framewidth;            // frame pixel width; 0: disable framing
  float    framergb[3];           // frame color

  // geom buffer status
  int      status;                // 0: ok, 1: geoms exhausted, warning issued
} mjvScene;


//---------------------------------- mjvFigure -----------------------------------------------------

typedef struct mjvFigure_ {       // abstract 2D figure passed to OpenGL renderer
  // enable flags
  int     flg_legend;             // show legend
  int     flg_ticklabel[2];       // show grid tick labels (x,y)
  int     flg_extend;             // automatically extend axis ranges to fit data
  int     flg_barplot;            // isolated line segments (i.e. GL_LINES)
  int     flg_selection;          // vertical selection line
  int     flg_symmetric;          // symmetric y-axis

  // style settings
  float   linewidth;              // line width
  float   gridwidth;              // grid line width
  int     gridsize[2];            // number of grid points in (x,y)
  float   gridrgb[3];             // grid line rgb
  float   figurergba[4];          // figure color and alpha
  float   panergba[4];            // pane color and alpha
  float   legendrgba[4];          // legend color and alpha
  float   textrgb[3];             // text color
  float   linergb[mjMAXLINE][3];  // line colors
  float   range[2][2];            // axis ranges; (min>=max) automatic
  char    xformat[20];            // x-tick label format for sprintf
  char    yformat[20];            // y-tick label format for sprintf
  char    minwidth[20];           // string used to determine min y-tick width

  // text labels
  char    title[1000];            // figure title; subplots separated with 2+ spaces
  char    xlabel[100];            // x-axis label
  char    linename[mjMAXLINE][100];  // line names for legend

  // dynamic settings
  int     legendoffset;           // number of lines to offset legend
  int     subplot;                // selected subplot (for title rendering)
  int     highlight[2];           // if point is in legend rect, highlight line
  int     highlightid;            // if id>=0 and no point, highlight id
  float   selection;              // selection line x-value

  // line data
  int     linepnt[mjMAXLINE];     // number of points in line; (0) disable
  float   linedata[mjMAXLINE][2*mjMAXLINEPNT];  // line data (x,y)

  // output from renderer
  int     xaxispixel[2];          // range of x-axis in pixels
  int     yaxispixel[2];          // range of y-axis in pixels
  float   xaxisdata[2];           // range of x-axis in data units
  float   yaxisdata[2];           // range of y-axis in data units
} mjvFigure;

#endif  // MUJOCO_MJVISUALIZE_H_
