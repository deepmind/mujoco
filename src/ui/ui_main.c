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

#include "ui/ui_main.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mujoco/mjmacro.h>
#include <mujoco/mujoco.h>
#include "engine/engine_array_safety.h"
#include "render/classic/glad/glad.h"

//------------------------------------ Default themes ----------------------------------------------

// theme spacing 0 : tight
static const mjuiThemeSpacing themeSpacing0 = {
  270,   // total
  15,    // scroll
  120,   // label
  8,     // section
  6,     // cornersect
  6,     // cornersep
  4,     // itemside
  4,     // itemmid
  4,     // itemver
  8,     // texthor
  4,     // textver
  30,    // linescroll
  4      // samples
};


// theme spacing 1 : wide
static const mjuiThemeSpacing themeSpacing1 = {
  310,   // total
  15,    // scroll
  120,   // label
  10,    // section
  10,    // cornersect
  10,    // cornersep
  7,     // itemside
  7,     // itemmid
  7,     // itemver
  10,    // texthor
  5,     // textver
  30,    // linescroll
  4      // samples
};


// theme color 0 : default
static const mjuiThemeColor themeColor0 = {
  {0.25, 0.25, 0.25},   // master
  {0.12, 0.12, 0.12},   // thumb
  {0.6,  0.2,  0.2},    // secttitle
  {0.1,  0.1,  0.1},    // secttitle2
  {0.45, 0.17, 0.17},   // secttitleuncheck
  {0.45, 0.17, 0.17},   // secttitleuncheck2
  {0.45, 0.17, 0.17},   // secttitlecheck
  {0.45, 0.17, 0.17},   // secttitlecheck2
  {1.0,  1.0,  1.0},    // sectfont
  {0.7,  0.7,  0.7},    // sectsymbol
  {0.1,  0.1,  0.1},    // sectpane
  {0.25, 0.25, 0.25},   // separator
  {0.1,  0.1,  0.1},    // separator2
  {0.0,  0.0,  1.0},    // shortcut
  {1.0,  1.0,  1.0},    // fontactive
  {0.5,  0.5,  0.5},    // fontinactive
  {0.3,  0.3,  0.3},    // decorinactive
  {0.4,  0.4,  0.4},    // decorinactive2
  {0.6,  0.4,  0.4},    // button
  {0.4,  0.4,  0.7},    // check
  {0.4,  0.6,  0.4},    // radio
  {0.4,  0.6,  0.6},    // select
  {0.2,  0.3,  0.3},    // select2
  {0.3,  0.2,  0.3},    // slider
  {0.6,  0.4,  0.6},    // slider2
  {0.6,  0.6,  0.4},    // edit
  {0.7,  0.0,  0.0},    // edit2
  {0.9,  0.9,  0.9}     // cursor
};


// theme color 1 : orange
static const mjuiThemeColor themeColor1 = {
  {0.2,  0.2,  0.2},    // master
  {0.12, 0.12, 0.12},   // thumb
  {0.3,  0.3,  0.3},    // secttitle
  {0.15, 0.15, 0.15},   // secttitle2
  {0.25, 0.25, 0.25},   // secttitleuncheck
  {0.25, 0.25, 0.25},   // secttitleuncheck2
  {0.25, 0.25, 0.25},   // secttitlecheck
  {0.25, 0.25, 0.25},   // secttitlecheck2
  {0.8,  0.8,  0.8},    // sectfont
  {0.7,  0.7,  0.7},    // sectsymbol
  {0.15, 0.15, 0.15},   // sectpane
  {0.2,  0.2,  0.2},    // separator
  {0.15, 0.15, 0.15},   // separator2
  {0.0,  0.0,  1.0},    // shortcut
  {0.9,  0.9,  0.9},    // fontactive
  {0.5,  0.5,  0.5},    // fontinactive
  {0.2,  0.2,  0.2},    // decorinactive
  {0.25, 0.25, 0.25},   // decorinactive2
  {0.6,  0.4,  0.2},    // button
  {0.6,  0.4,  0.2},    // check
  {0.6,  0.4,  0.2},    // radio
  {0.6,  0.4,  0.2},    // select
  {0.3,  0.2,  0.1},    // select2
  {0.2,  0.2,  0.2},    // slider
  {0.6,  0.4,  0.2},    // slider2
  {0.6,  0.4,  0.2},    // edit
  {0.7,  0.0,  0.0},    // edit2
  {0.9,  0.9,  0.9}     // cursor
};


// theme color 2 : white
static const mjuiThemeColor themeColor2 = {
  {0.9,  0.9,  0.9},    // master
  {0.7,  0.7,  0.7},    // thumb
  {0.8,  0.8,  0.8},    // secttitle
  {1.0,  1.0,  1.0},    // secttitle2
  {0.95, 0.95, 0.95},   // secttitleuncheck
  {0.95, 0.95, 0.95},   // secttitleuncheck2
  {0.95, 0.95, 0.95},   // secttitlecheck
  {0.95, 0.95, 0.95},   // secttitlecheck2
  {0.0,  0.0,  0.8},    // sectfont
  {0.0,  0.0,  0.8},    // sectsymbol
  {1.0,  1.0,  1.0},    // sectpane
  {0.9,  0.9,  0.9},    // separator
  {1.0,  1.0,  1.0},    // separator2
  {0.0,  1.0,  1.0},    // shortcut
  {0.0,  0.0,  0.0},    // fontactive
  {0.7,  0.7,  0.7},    // fontinactive
  {0.95, 0.95, 0.95},   // decorinactive
  {0.9,  0.9,  0.9},    // decorinactive2
  {0.8,  0.8,  0.8},    // button
  {0.8,  0.8,  0.8},    // check
  {0.8,  0.8,  0.8},    // radio
  {0.8,  0.8,  0.8},    // select
  {0.9,  0.9,  0.9},    // select2
  {0.95, 0.95, 0.95},   // slider
  {0.8,  0.8,  0.8},    // slider2
  {0.8,  0.8,  0.8},    // edit
  {1.0,  0.3,  0.3},    // edit2
  {0.2,  0.2,  0.2}     // cursor
};


// theme color 3 : black
static const mjuiThemeColor themeColor3 = {
  {0.15, 0.15, 0.15},   // master
  {0.3,  0.3,  0.3},    // thumb
  {0.25, 0.25, 0.25},   // secttitle
  {0.0,  0.0,  0.0},    // secttitle2
  {0.2,  0.2,  0.2},    // secttitleuncheck
  {0.2,  0.2,  0.2},    // secttitleuncheck2
  {0.2,  0.2,  0.2},    // secttitlecheck
  {0.2,  0.2,  0.2},    // secttitlecheck2
  {1.0,  0.3,  0.3},    // sectfont
  {1.0,  0.3,  0.3},    // sectsymbol
  {0.0,  0.0,  0.0},    // sectpane
  {0.15, 0.15, 0.15},   // separator
  {0.0,  0.0,  0.0},    // separator2
  {0.0,  0.0,  1.0},    // shortcut
  {1.0,  1.0,  1.0},    // fontactive
  {0.4,  0.4,  0.4},    // fontinactive
  {0.1,  0.1,  0.1},    // decorinactive
  {0.15, 0.15, 0.15},   // decorinactive2
  {0.3,  0.3,  0.3},    // button
  {0.3,  0.3,  0.3},    // check
  {0.3,  0.3,  0.3},    // radio
  {0.3,  0.3,  0.3},    // select
  {0.15, 0.15, 0.15},   // select2
  {0.15, 0.15, 0.15},   // slider
  {0.3,  0.3,  0.3},    // slider2
  {0.3,  0.3,  0.3},    // edit
  {0.8,  0.2,  0.2},    // edit2
  {0.8,  0.8,  0.8}     // cursor
};



//------------------------------------ Utility functions -------------------------------------------

// scale from abstract pixels to framebuffer units
static int SCL(int sz, const mjrContext* con) {
  return mjMAX(0, mju_round(sz * 0.01 * con->fontScale));
}



// init OpenGL
static void initOpenGL(const mjrRect* r, const mjrContext* con) {
  // set OpenGL options
  glDisable(GL_NORMALIZE);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_COLOR_MATERIAL);
  glDisable(GL_BLEND);
  glShadeModel(GL_SMOOTH);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // standard 2D projection, in framebuffer units
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, r->width, 0, r->height, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // set viewport
  glViewport(r->left, r->bottom, r->width, r->height);
}



// get text width up to specified limit (0: 0, -1: entire string)
static int textwidth(const char* text, const mjrContext* con, int limit) {
  int i = 0, width = 0;

  // zero limit
  if (limit == 0) {
    return 0;
  }

  // add character widths
  while (text[i] && (limit < 1 || i < limit)) {
    width += con->charWidth[(unsigned char)text[i++]];
  }

  return width;
}



// draw text, limit to maxwidth
static void drawtext(const char* txt, int x, int y, int maxwidth,
                     const float* rgb, const mjrContext* con) {
  // determine string length that fits in maxwidth
  int len = 0, width = 0;
  while (txt[len]) {
    width += con->charWidth[(unsigned char)txt[len]];
    if (width >= maxwidth) {
      break;
    } else {
      len++;
    }
  }

  // draw normal text
  glListBase(con->baseFontNormal);
  glColor3fv(rgb);
  glRasterPos3i(x, y, 0);
  glCallLists(len, GL_UNSIGNED_BYTE, txt);
}



// draw text centered in rectangle
static void drawtextrect(mjrRect rect, const char* txt,
                         const float* rgb, const mjrContext* con) {
  int dy = (rect.height - con->charHeight)/2;
  int dx = (rect.width - textwidth(txt, con, -1))/2;
  dx = mjMAX(0, dx);

  drawtext(txt, rect.left+dx, rect.bottom+dy, rect.width-dx, rgb, con);
}



// draw rectangle
static void drawrectangle(mjrRect rect, const float* rgb, const float* rgbback,
                          const mjrContext* con) {
  // outside
  glColor3fv(rgb);
  glBegin(GL_QUADS);
  glVertex2i(rect.left, rect.bottom);
  glVertex2i(rect.left+rect.width, rect.bottom);
  glVertex2i(rect.left+rect.width, rect.bottom+rect.height);
  glVertex2i(rect.left, rect.bottom+rect.height);
  glEnd();

  // inside
  if (rgbback) {
    int margin = SCL(2, con);
    glColor3fv(rgbback);
    glBegin(GL_QUADS);
    glVertex2i(rect.left+margin, rect.bottom+margin);
    glVertex2i(rect.left+rect.width-margin, rect.bottom+margin);
    glVertex2i(rect.left+rect.width-margin, rect.bottom+rect.height-margin);
    glVertex2i(rect.left+margin, rect.bottom+rect.height-margin);
    glEnd();
  }
}



// round corners of rectangle
static void roundcorner(mjrRect rect, int flg_skipbottom, int flg_separator,
                        const mjUI* ui, const mjrContext* con) {
  // get rounding from theme, exit if disabled
  int cornerspec = flg_separator ? ui->spacing.cornersep : ui->spacing.cornersect;
  if (cornerspec == 0) {
    return;
  }

  // quarter-circle divisions and radius
  int ndivide = 10;
  double radius = cornerspec * 0.01 * con->fontScale;

  // draw fans in the four corners, optionally skip bottom corners
  for (int ic = (flg_skipbottom ? 2 : 0); ic < 4; ++ic) {
    // set corner
    double corner[2];
    switch (ic) {
    case 0:   // bottom-left
      corner[0] = rect.left;
      corner[1] = rect.bottom;
      break;

    case 1:   // bottom-right
      corner[0] = rect.left + rect.width;
      corner[1] = rect.bottom;
      break;

    case 2:   // top-right
      corner[0] = rect.left + rect.width;
      corner[1] = rect.bottom + rect.height;
      break;

    default:  // top-left
      corner[0] = rect.left;
      corner[1] = rect.bottom + rect.height;
    }

    // orient fan to point inside
    double angle = ic * 0.5 * mjPI;

    // compute circle center: opposite to corner
    double center[2];
    center[0] = corner[0] + mju_sqrt(2.0) * radius * cos(angle + 0.25 * mjPI);
    center[1] = corner[1] + mju_sqrt(2.0) * radius * sin(angle + 0.25 * mjPI);

    // fill with erase color, start trinagle_fan from corner
    glColor3fv(flg_separator ? ui->color.sectpane : ui->color.master);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2d(corner[0], corner[1]);

    // compute vertices of quarter-circle
    for (int i = 0; i <= ndivide; i++) {
      double a = angle + mjPI + 0.5 * mjPI * (double)i / (double)ndivide;
      glVertex2d(center[0] + radius * cos(a), center[1] + radius * sin(a));
    }
    glEnd();
  }
}



// draw oval
static void drawoval(mjrRect rect, const float* rgb, const float* rgbback,
                     const mjrContext* con) {
  const int ndivide = 15;

  // require horizontal
  if (rect.height > rect.width) {
    return;
  }

  // circle info
  double radius = 0.5*rect.height;
  double lcenter[2] = {rect.left + radius, rect.bottom + radius};
  double rcenter[2] = {rect.left + rect.width - radius, rect.bottom + radius};

  // filled
  glColor3fv(rgb);
  glBegin(GL_POLYGON);

  // draw left half-circle
  for (int i=0; i <= ndivide; i++) {
    double angle = mjPI * (0.5 + (double)i / (double)ndivide);
    glVertex2d(lcenter[0] + radius*cos(angle), lcenter[1] + radius*sin(angle));
  }

  // draw right half-circle
  for (int i=0; i <= ndivide; i++) {
    double angle = mjPI * (1.5 + (double)i / (double)ndivide);
    glVertex2d(rcenter[0] + radius*cos(angle), rcenter[1] + radius*sin(angle));
  }
  glEnd();

  // inside
  if (rgbback) {
    int margin = SCL(2, con);
    radius -= margin;

    glColor3fv(rgbback);
    glBegin(GL_POLYGON);

    // draw left half-circle
    for (int i=0; i <= ndivide; i++) {
      double angle = mjPI * (0.5 + (double)i / (double)ndivide);
      glVertex2d(lcenter[0] + radius*cos(angle), lcenter[1] + radius*sin(angle));
    }

    // draw right half-circle
    for (int i=0; i <= ndivide; i++) {
      double angle = mjPI * (1.5 + (double)i / (double)ndivide);
      glVertex2d(rcenter[0] + radius*cos(angle), rcenter[1] + radius*sin(angle));
    }
    glEnd();
  }
}



// draw open/closed symbol in title
//  type: 0- section, 1- section with unchecked box,
//        2- section with checked box, 3- separator
static void drawsymbol(mjrRect rect, int flg_open, int type,
                       const mjUI* ui, const mjrContext* con) {
  // size and center
  int texthor =  SCL(ui->spacing.texthor, con);
  int cx = rect.left + rect.width - texthor;
  int cy = rect.bottom + rect.height/2;
  int d = mju_round(con->charHeight*0.33);

  // separator size
  if (type == 3) {
    d = mju_round(con->charHeight*0.28);
  }

  // open
  if (flg_open) {
    glColor3fv(ui->color.sectsymbol);
    glBegin(GL_TRIANGLES);
    glVertex2i(cx, cy+d);
    glVertex2i(cx-2*d, cy+d);
    glVertex2i(cx-d, cy-d);
    glEnd();
  }

  // closed
  else {
    // solid outside
    glColor3fv(ui->color.sectsymbol);
    glBegin(GL_TRIANGLES);
    glVertex2i(cx, cy-d);
    glVertex2i(cx, cy+d);
    glVertex2i(cx-2*d, cy);
    glEnd();

    // set color for inside
    switch (type) {
    case 0:   // section
      glColor3f(
        (ui->color.secttitle[0] + ui->color.secttitle2[0]) * 0.5,
        (ui->color.secttitle[1] + ui->color.secttitle2[1]) * 0.5,
        (ui->color.secttitle[2] + ui->color.secttitle2[2]) * 0.5
      );
      break;

    case 1:   // section with unchecked box
      glColor3f(
        (ui->color.secttitleuncheck[0] + ui->color.secttitleuncheck2[0]) * 0.5,
        (ui->color.secttitleuncheck[1] + ui->color.secttitleuncheck2[1]) * 0.5,
        (ui->color.secttitleuncheck[2] + ui->color.secttitleuncheck2[2]) * 0.5
      );
      break;

    case 2:   // section with checked box
      glColor3f(
        (ui->color.secttitlecheck[0] + ui->color.secttitlecheck2[0]) * 0.5,
        (ui->color.secttitlecheck[1] + ui->color.secttitlecheck2[1]) * 0.5,
        (ui->color.secttitlecheck[2] + ui->color.secttitlecheck2[2]) * 0.5
      );
      break;

    case 3:   // separator
      glColor3f(
        (ui->color.separator[0] + ui->color.separator2[0]) * 0.5,
        (ui->color.separator[1] + ui->color.separator2[1]) * 0.5,
        (ui->color.separator[2] + ui->color.separator2[2]) * 0.5
      );
    }

    // draw inside
    double margin = con->fontScale * 0.015;
    double u = 0.5 * sqrt(5.0) * margin;
    double y = d - u - 0.5 * margin;
    glBegin(GL_TRIANGLES);
    glVertex2d(cx-margin, cy-y);
    glVertex2d(cx-margin, cy+y);
    glVertex2d(cx-2*d+2*u, cy);
    glEnd();
  }
}



// radio element rectangle
static mjrRect radioelement(const mjuiItem* it, int n,
                            const mjUI* ui, const mjrContext* con) {
  // scale sizes from theme
  int g_itemmid =  SCL(ui->spacing.itemmid, con);
  int g_textver =  SCL(ui->spacing.textver, con);
  int ncol = ui->radiocol ? ui->radiocol : 2;
  int nrow = (it->multi.nelem-1)/ncol + 1;

  // compute elements
  int cellwidth = (it->rect.width-(ncol-1)*g_itemmid)/ncol;
  int cellheight = con->charHeight+2*g_textver;
  int row = n/ncol;
  int col = n%ncol;

  // construct rectangle
  mjrRect r = {
    it->rect.left + col*(cellwidth+g_itemmid),
    it->rect.bottom+(nrow-1-row)*cellheight,
    cellwidth,
    cellheight
  };

  // adjust width of last column
  if (col == ncol-1) {
    r.width = it->rect.width - r.left + it->rect.left;
  }

  return r;
}



// compute mouse position relative to ui
static void mouseinui(const mjUI* ui, const mjuiState* ins, int* x, int* y) {
  // extract data
  *x = (int)ins->x;
  *y = (int)ins->y;
  mjrRect rect = ins->rect[ui->rectid];

  // correct for scrollbar
  if (ui->height > rect.height) {
    *y -= ui->scroll;
  }

  // correct for offset
  *x -= rect.left;
  *y -= (rect.height - ui->height) + rect.bottom;
}



// compute mouse position relative to rectangle
static void mouseinrect(mjrRect rect, const mjUI* ui, const mjuiState* ins,
                        double* rx, double* ry) {
  // mouse relative to ui
  int x, y;
  mouseinui(ui, ins, &x, &y);

  // mouse relative to rectangle
  *rx = (x-rect.left) / (double)rect.width;
  *ry = (y-rect.bottom) / (double)rect.height;
}



// find radio element under mouse
static int findradio(const mjuiItem* it, const mjUI* ui,
                     const mjuiState* ins, const mjrContext* con) {
  // number of rows and columns
  int ncol = ui->radiocol ? ui->radiocol : 2;
  int nrow = (it->multi.nelem-1)/ncol + 1;

  // get mouse relative to rect
  double rx, ry;
  mouseinrect(it->rect, ui, ins, &rx, &ry);

  // count rows from top
  ry = 1-ry;

  // round and clamp
  int row = (int)floor(ry*nrow);
  int col = (int)floor(rx*ncol);
  row = mjMAX(0, mjMIN(row, nrow-1));
  col = mjMAX(0, mjMIN(col, ncol-1));

  // result
  int ind = row*ncol + col;
  if (ind < it->multi.nelem) {
    return ind;
  } else {
    return -1;
  }
}



// make list of separators for radioline elements
static void makeradioline(const mjuiItem* it, const mjrContext* con, int* sep) {
  int nelem = it->multi.nelem, totwid = 0;
  int elwid[mjMAXUIMULTI];

  // no elements
  if (!nelem) {
    return;
  }

  // compute element widths
  for (int i=0; i < nelem; i++) {
    elwid[i] = textwidth(it->multi.name[i], con, -1);
    totwid += elwid[i];
  }

  // compute per-element extra space
  double extra = ((double)(it->rect.width - totwid)) / ((double)nelem);

  // compute separators
  sep[0] = 0;
  for (int i=0; i < nelem; i++) {
    sep[i+1] = sep[i] + elwid[i] + mju_round((i+1)*extra) -mju_round(i*extra);
  }
  sep[nelem] = it->rect.width;
}



// find radioline element under mouse
static int findradioline(const mjuiItem* it, const mjUI* ui,
                         const mjuiState* ins, const mjrContext* con) {
  // make separators
  int sep[mjMAXUIMULTI+1];
  makeradioline(it, con, sep);

  // get mouse relative to rect
  double rx, ry;
  mouseinrect(it->rect, ui, ins, &rx, &ry);

  // find interval
  int x = mju_round(rx*it->rect.width);
  for (int i=0; i < it->multi.nelem; i++) {
    if (x >= sep[i] && x < sep[i+1]) {
      return i;
    }
  }

  // not found
  return -1;
}



// find select element under mouse
static int findselect(const mjuiItem* it, const mjUI* ui,
                      const mjuiState* ins, const mjrContext* con) {
  // not tracking: nothing to do
  if (!(ui->mousesect > 0 && ui->mouseitem >= 0 && it && it->type == mjITEM_SELECT)) {
    return -1;
  }

  // make box
  int g_textver = SCL(ui->spacing.textver, con);
  int cellheight = con->charHeight + 2*g_textver;
  mjrRect r = it->rect;
  r.height = it->multi.nelem * cellheight;
  r.bottom -= r.height;

  // find in box
  double rx, ry;
  mouseinrect(r, ui, ins, &rx, &ry);
  if (ry > 0 && ry < 1 && rx > 0 && rx < 1) {
    int k = (int)floor(ry*it->multi.nelem);
    k = mjMAX(0, mjMIN(it->multi.nelem-1, k));
    return it->multi.nelem-1-k;
  } else {
    return -1;
  }
}



// make scrollbar rectangles
void scrollrect(mjrRect rect, const mjUI* ui, const mjrContext* con,
                mjrRect* bar, mjrRect* thumb) {
  int w_scroll = SCL(ui->spacing.scroll, con);

  // bar
  *bar = rect;
  bar->left = rect.left+rect.width-w_scroll;
  bar->width = w_scroll;

  // thumb
  double tstart = (double)ui->scroll / (double)ui->height;
  double tend = (double)(ui->scroll+rect.height) / (double)ui->height;
  *thumb = *bar;
  thumb->bottom = rect.bottom + mju_round(rect.height*(1-tend));
  thumb->height = mju_round(rect.height*(tend-tstart));
}



// is point in rectangle
static int inside(int x, int y, mjrRect r) {
  return (x >= r.left && x <= r.left+r.width && y >= r.bottom && y <= r.bottom+r.height);
}



// is point in oval
static int insideoval(int x, int y, mjrRect r) {
  // exclude if not in rectangle
  if (!inside(x, y, r)) {
    return 0;
  }

  // check center
  int radius = r.height/2;
  if (x >= r.left+radius && x <= r.left+r.width-radius) {
    return 1;
  }

  // left circle
  int dx = x - (r.left+radius);
  int dy = y - (r.bottom+radius);
  if (dx < 0 && (dx*dx + dy*dy < radius*radius)) {
    return 1;
  }

  // right circle
  dx = x - (r.left+r.width-radius);
  if (dx > 0 && (dx*dx + dy*dy < radius*radius)) {
    return 1;
  }

  return 0;
}



// find mouse location in UI; y already inverted
// sect: -1: thumb, -2: slider down, -3: slider up, positive: 1+section
// item: -1: section title or scroll, non-negative: item number
// item: -2: in checkbox on section title
static void findmouse(const mjUI* ui, const mjuiState* ins, const mjrContext* con,
                      int* sect, int* item) {
  // clear
  *sect = 0;
  *item = 0;

  // extract data
  int x = (int)ins->x;
  int y = (int)ins->y;
  mjrRect rect = ins->rect[ui->rectid];

  // scrollbar
  if (ui->height > rect.height) {
    // construct rectangles
    mjrRect bar;
    mjrRect thumb;
    scrollrect(rect, ui, con, &bar, &thumb);

    // inside bar
    if (inside(x, y, bar)) {
      // inside thumb
      if (inside(x, y, thumb)) {
        *sect = -1;
      }

      // below thumb
      else if (y < thumb.bottom) {
        *sect = -2;
      }

      // above thumb
      else {
        *sect = -3;
      }

      // done
      *item = -1;
      return;
    }
  }

  // adjust x,y
  mouseinui(ui, ins, &x, &y);

  // check sections and items
  for (int n=0; n < ui->nsect; n++) {
    // get section pointer
    const mjuiSection* s = ui->sect + n;

    // in title
    if (s->state < 2 && inside(x, y, s->rtitle)) {
      *sect = n+1;
      *item = -1;

      // in checkbox
      if (s->checkbox > 0) {
        mjrRect rcheck = s->rtitle;
        rcheck.width = mjMIN(rcheck.height, rcheck.width);
        if (inside(x, y, rcheck)) {
          *item = -2;
        }
      }

      return;
    }

    // in content and open: check items
    if (s->state && inside(x, y, s->rcontent)) {
      for (int i=0; i < s->nitem; i++) {
        if (s->item[i].type == mjITEM_BUTTON ?
            insideoval(x, y, s->item[i].rect) :
            inside(x, y, s->item[i].rect)) {
          *sect = n+1;
          *item = i;
          return;
        }
      }
    }
  }
}



// set slider position given mouse
static void setslider(mjuiItem* it, mjUI* ui,
                      const mjuiState* ins, const mjrContext* con) {
  // get mouse relative position
  double rx, ry;
  mouseinrect(it->rect, ui, ins, &rx, &ry);

  // clamp rx, enforce divisions
  rx = mjMAX(0, mjMIN(1, rx));
  rx = mju_round(rx * it->slider.divisions) / mjMAX(1, it->slider.divisions);
  rx = mjMAX(0, mjMIN(1, rx));

  // compute value, clamp to range
  mjtNum val = mju_clip(it->slider.range[0]*(1-rx) + it->slider.range[1]*rx,
                        it->slider.range[0], it->slider.range[1]);

  // set slider position
  if (it->type == mjITEM_SLIDERINT) {
    *(int*)it->pdata = mju_round(val);
  } else {
    *(mjtNum*)it->pdata = val;
  }
}



// check edit text before conversion to array
// return 0 if ok, error code otherwise
static int checkedit(const char* text, const mjuiItem* it) {
  // text always passes
  if (it->type == mjITEM_EDITTXT) {
    return 0;
  }

  // check type
  if (it->type != mjITEM_EDITINT &&
      it->type != mjITEM_EDITNUM &&
      it->type != mjITEM_EDITFLOAT) {
    mju_error("Internal error: expected edit control");
  }

  // convert
  double val[mjMAXUIEDIT];
  int n = sscanf(text, "%lf %lf %lf %lf %lf %lf %lf",
                 val, val+1, val+2, val+3, val+4, val+5, val+6);

  // check length
  if (n != it->edit.nelem) {
    return 1;
  }

  // check range when defined
  for (int i=0; i < n; i++)
    if (it->edit.range[i][0] < it->edit.range[i][1] &&
        (val[i] < it->edit.range[i][0] || val[i] > it->edit.range[i][1])) {
      return 2;
    }

  // require int values for int type
  if (it->type == mjITEM_EDITINT) {
    for (int i=0; i < n; i++) {
      if (val[i] != (double)((int)val[i])) {
        return 3;
      }
    }
  }

  return 0;
}



// convert edit text to numeric array
// return 0 if ok, error code otherwise
static int text2array(const char* text, const mjuiItem* it) {
  // text: copy
  if (it->type == mjITEM_EDITTXT) {
    // copy string, assume mjMAXUINAME allocation
    char* pdata = (char*)it->pdata;
    strncpy(pdata, text, mjMAXUINAME);
    pdata[mjMAXUINAME-1] = 0;
    return 0;
  }

  // check type
  if (it->type != mjITEM_EDITINT && it->type != mjITEM_EDITNUM && it->type != mjITEM_EDITFLOAT) {
    mju_error("Internal error: expected edit control");
  }

  // convert
  double val[mjMAXUIEDIT];
  int n = sscanf(text, "%lf %lf %lf %lf %lf %lf %lf",
                 val, val+1, val+2, val+3, val+4, val+5, val+6);

  // check length
  if (n != it->edit.nelem) {
    return 1;
  }

  // check range when defined
  for (int i=0; i < n; i++) {
    if (it->edit.range[i][0] < it->edit.range[i][1] &&
        (val[i] < it->edit.range[i][0] || val[i] > it->edit.range[i][1])) {
      return 2;
    }
  }

  // require int values for int type
  if (it->type == mjITEM_EDITINT) {
    for (int i=0; i < n; i++) {
      if (val[i] != (double)((int)val[i])) {
        return 3;
      }
    }
  }

  // copy values
  if (it->type == mjITEM_EDITINT) {
    int* pdata = (int*)it->pdata;
    for (int i=0; i < n; i++) {
      pdata[i] = (int)val[i];
    }
  } else {
    if (it->type == mjITEM_EDITNUM) {
      mjtNum* pdata = (mjtNum*)it->pdata;
      for (int i=0; i < n; i++) {
        pdata[i] = (mjtNum)val[i];
      }
    } else {
      float* pdata = (float*)it->pdata;
      for (int i=0; i < n; i++) {
        pdata[i] = (float)val[i];
      }
    }
  }

  return 0;
}



// convert numeric array to edit text
static void array2text(char* text, const mjuiItem* it) {
  // text: copy
  if (it->type == mjITEM_EDITTXT) {
    // copy string, assume mjMAXUINAME allocation
    strncpy(text, (const char*)it->pdata, mjMAXUINAME);
    text[mjMAXUINAME-1] = 0;
    return;
  }

  // check type
  if (it->type != mjITEM_EDITINT && it->type != mjITEM_EDITNUM && it->type != mjITEM_EDITFLOAT) {
    mju_error("Internal error: expected edit control");
  }

  // print
  int n = it->edit.nelem;
  char buf[50];
  text[0] = 0;
  for (int i=0; i < n; i++) {
    if (it->type == mjITEM_EDITINT) {
      mjSNPRINTF(buf, "%d", ((int*)it->pdata)[i]);
    } else if (it->type == mjITEM_EDITNUM) {
      mjSNPRINTF(buf, "%.4g", ((mjtNum*)it->pdata)[i]);
    } else {
      mjSNPRINTF(buf, "%.4g", ((float*)it->pdata)[i]);
    }
    strncat(text, buf, mjMAXUITEXT - strlen(text) - 1);
    if (i < n-1) {
      strncat(text, "  ", mjMAXUITEXT - strlen(text) - 1);
    }
  }
}


// return (remapped) key if valid, 0 if not valid
static int validkey(int key, int sz, int type, const mjuiState* state) {
  // text
  if (type == mjITEM_EDITTXT) {
    if (sz < mjMAXUINAME-1 && key >= 32 && key <= 127) {
      // lower case if Shift (Caps Lock cannot be detected in GLWF)
      if (key >= 'A' && key <= 'Z' && !state->shift) {
        key = key + 'a' - 'A';
      }

      // Shift: remap remaining keys
      else if (state->shift) {
        if (key == '`') {
          key = '~';
        } else if (key == '-') {
          key = '_';
        } else if (key == '=') {
          key = '+';
        } else if (key == '[') {
          key = '{';
        } else if (key == ']') {
          key = '}';
        } else if (key == '\\') {
          key = '|';
        } else if (key == ';') {
          key = ':';
        } else if (key == '\'') {
          key = '"';
        } else if (key == ',') {
          key = '<';
        } else if (key == '.') {
          key = '>';
        } else if (key == '/') {
          key = '?';
        }
      }

      return key;
    } else {
      return 0;
    }
  }

  // numeric
  else if (type == mjITEM_EDITINT || type == mjITEM_EDITNUM || type == mjITEM_EDITFLOAT) {
    if (sz < (mjMAXUITEXT-1) &&
        (key == ' ' || key == '+' || key == '=' || key == '-' || (key >= '0' && key <= '9') ||
         (key >= mjKEY_NUMPAD_0 && key <= mjKEY_NUMPAD_9) ||
         ((key == 'e' || key == 'E' || key == '.') &&
          (type == mjITEM_EDITNUM || type == mjITEM_EDITFLOAT)))) {

      // remap '=' to '+'
      if (key == '=') {
        key = '+';
      }

      // remap 'E' to 'e'
      if (key == 'E') {
        key = 'e';
      }

      // remap numberpad to top row
      if (key >= mjKEY_NUMPAD_0 && key <= mjKEY_NUMPAD_9) {
        key = key - mjKEY_NUMPAD_0 + '0';
      }

      return key;

    } else {
      return 0;
    }
  }

  // other
  else {
    return 0;
  }
}



// adjust editscroll so cursor is visible
static void revealcursor(mjrRect r, mjUI* ui, const mjrContext* con) {
  // scroll left
  if (ui->editcursor <= ui->editscroll) {
    ui->editscroll = ui->editcursor;
    return;
  }

  // width of available text area
  int width = r.width - 2*SCL(ui->spacing.texthor, con);

  // scan backwards
  int i = ui->editcursor;
  while (width >= 0 && i >= ui->editscroll && i > 0) {
    i--;
    width -= con->charWidth[(unsigned char)ui->edittext[i]];
  }

  // adjust scroll if out of width
  if (width < 0) {
    ui->editscroll = i+1;
  }
}



// use mouse position to set editcursor, adjust editscroll
static void setcursor(mjrRect r, mjUI* ui, const mjuiState* ins, const mjrContext* con) {
  // correct rectangle for texthor
  int g_texthor = SCL(ui->spacing.texthor, con);
  mjrRect r1 = r;
  r1.left += g_texthor;
  r1.width -= 2*g_texthor;

  // get mouse relative position
  double rx, ry;
  mouseinrect(r1, ui, ins, &rx, &ry);

  // outside rectangle
  if (rx < 0) {
    ui->editcursor = 0;
    ui->editscroll = 0;
  } else if (rx > 1) {
    ui->editcursor = strlen(ui->edittext);
    revealcursor(r, ui, con);
  }

  // inside rectangle
  else {
    int besti = ui->editscroll, cumsum = 0;
    int R = mju_round(r1.width*rx), bestdif = R;

    // find closest between-char position
    for (int i=ui->editscroll; i < strlen(ui->edittext); i++) {
      // add next width
      cumsum += con->charWidth[(unsigned char)ui->edittext[i]];

      // update best
      if (bestdif > abs(cumsum-R)) {
        bestdif = abs(cumsum-R);
        besti = i+1;
      }
    }

    // set
    ui->editcursor = besti;
  }
}



// parse modifier and shortcut
static void parseshortcut(const char* text, int* mod, int* key) {
  // clear
  *mod = 0;
  *key = 0;

  // empty: return
  if (!text[0]) {
    return;
  }

  // require between 2 and 5 characters
  if (strlen(text) < 2 || strlen(text) > 5) {
    mju_error("mjui_add: invalid shortcut specification");
  }

  // modifier
  switch (text[0]) {
  case ' ':
    *mod = 0;
    break;

  case 'C':
    *mod = 1;
    break;

  case 'S':
    *mod = 2;
    break;

  case 'A':
    *mod = 4;
    break;

  default:
    mju_error("mjui_add: invalid shortcut modifier");
  }

  // key
  if (text[1] == '#') {
    if (sscanf(text+2, "%d", key) != 1) {
      mju_error("mjui_add: invalid shortcut numeric code");
    }
  } else {
    if (text[2] != 0) {
      mju_error("mjui_add: invalid shortcut");
    }
    *key = (int)text[1];
  }
}



// check fpr matching modifier and shortcut
static int matchshortcut(const mjuiState* ins, int mod, int key) {
  // match key
  if (!key || ins->key != key) {
    return 0;
  }

  // match modifier
  if ((ins->control != 0) + 2*(ins->shift != 0) + 4*(ins->alt != 0) != mod) {
    return 0;
  }

  return 1;
}



//---------------------------------- Public API ----------------------------------------------------

// Get builtin UI theme spacing (0-1).
mjuiThemeSpacing mjui_themeSpacing(int ind) {
  if (ind == 0) {
    return themeSpacing0;
  } else {
    return themeSpacing1;
  }
}



// Get builtin UI theme color (0-3).
mjuiThemeColor mjui_themeColor(int ind) {
  if (ind == 0) {
    return themeColor0;
  } else if (ind == 1) {
    return themeColor1;
  } else if (ind == 2) {
    return themeColor2;
  } else {
    return themeColor3;
  }
}



// Add definitions to UI.
void mjui_add(mjUI* ui, const mjuiDef* def) {
  int n = 0, i, start, num;
  double x[1+2*mjMAXUIEDIT];

  // process entries until end marker
  while (def[n].type != mjITEM_END) {
    // section
    if (def[n].type == mjITEM_SECTION) {
      // check limit
      if (ui->nsect >= mjMAXUISECT) {
        mju_error("mjui_add: too many sections");
      }

      // check data
      if (strlen(def[n].name) >= mjMAXUINAME-1) {
        mju_error("mjui_add: section name too long");
      }
      if (def[n].state != mjSECT_CLOSED && def[n].state != mjSECT_OPEN &&
          def[n].state != mjSECT_FIXED && def[n].state != mjPRESERVE) {
        mju_error("mjui_add: invalid section state");
      }

      // add section, save state
      ui->nsect++;
      mjuiSection* se = ui->sect + (ui->nsect-1);
      int oldstate = se->state;

      // clear, but preserve item states
      int itemstate[mjMAXUIITEM];
      for (int i = 0; i < mjMAXUIITEM; ++i) {
        itemstate[i] = se->item[i].state;
      }
      memset(se, 0, sizeof(mjuiSection));
      for (int i = 0; i < mjMAXUIITEM; ++i) {
        se->item[i].state = itemstate[i];
      }

      // set or restore section state
      se->state = (def[n].state == mjPRESERVE ? oldstate : def[n].state);

      // copy remaining data
      mjSTRNCPY(se->name, def[n].name);
      se->checkbox = def[n].otherint;
      parseshortcut(def[n].other, &(se->modifier), &(se->shortcut));
    }

    // item
    else if (def[n].type >= 0 && def[n].type < mjNITEM) {
      // first section must be defined
      if (ui->nsect <= 0) {
        mju_error("mjui_add: item defined outside section");
      }
      mjuiSection* se = ui->sect + (ui->nsect-1);

      // check limit
      if (se->nitem >= mjMAXUIITEM) {
        mju_error("mjui_add: too many items in section");
      }

      // check item data
      if (def[n].type < 0 || def[n].type >= mjNITEM) {
        mju_error("mjui_add: invalid item type");
      }
      if (strlen(def[n].name) >= mjMAXUINAME) {
        mju_error("mjui_add: item name too long");
      }
      if (def[n].state < 0) {
        mju_error("mjui_add: invalid item state");
      }

      // add item, save state, clear
      se->nitem++;
      mjuiItem* it = se->item + (se->nitem-1);
      int oldstate = it->state;
      memset(it, 0, sizeof(mjuiItem));

      // set or restore state for collapsible separator, copy state for others
      if (def[n].type == mjITEM_SEPARATOR && def[n].state == mjPRESERVE) {
        // mjSEPCLOSED makes separator collapsible
        it->state = (oldstate < mjSEPCLOSED ? mjSEPCLOSED : oldstate);
      }
      else {
        it->state = def[n].state;
      }

      // copy common data
      it->type = def[n].type;
      it->pdata = def[n].pdata;
      mjSTRNCPY(it->name, def[n].name);
      it->sectionid = ui->nsect - 1;
      it->itemid = se->nitem - 1;
      it->userid = def[n].otherint;

      // data pointer check
      if (it->type > mjITEM_BUTTON && it->pdata == 0) {
        mju_error("mjui_add: no data pointer for item with data");
      }
      if (it->type <= mjITEM_BUTTON && it->pdata) {
        mju_error("mjui_add: data pointer for item without data");
      }

      // parse button and check
      if (it->type == mjITEM_BUTTON ||
          it->type == mjITEM_CHECKINT ||
          it->type == mjITEM_CHECKBYTE) {
        parseshortcut(def[n].other, &(it->single.modifier), &(it->single.shortcut));
      }

      // parse static, radio, radioline, select
      else if (it->type == mjITEM_STATIC    ||
               it->type == mjITEM_RADIO     ||
               it->type == mjITEM_RADIOLINE ||
               it->type == mjITEM_SELECT) {
        // init
        it->multi.nelem = 0;
        num = strlen(def[n].other);
        i = 0;

        // find names in '/n'-separated string
        while (i < num) {
          // check limit
          if (it->multi.nelem >= mjMAXUIMULTI) {
            mju_error("mjui_add: too many multi elements");
          }

          // find next '\n' or 0
          start = i;
          while (def[n].other[i] != '\n' && def[n].other[i] != 0) {
            i++;
          }

          // check string
          if (i == start || i-start >= mjMAXUINAME-1) {
            mju_error("mjui_add: invalid multi element name");
          }

          // add
          int ind = it->multi.nelem;
          (it->multi.nelem)++;
          strncpy(it->multi.name[ind], def[n].other+start, i-start);
          it->multi.name[ind][i-start] = 0;

          // advance
          i++;
        }
      }

      // parse slider
      else if (it->type == mjITEM_SLIDERINT || it->type == mjITEM_SLIDERNUM) {
        // read and check number
        num = sscanf(def[n].other, "%lf %lf %lf", x, x+1, x+2);
        if (num != 2 && num != 3) {
          mju_error("mjui_add: slider expects 'min max [div]'");
        }

        // assign range
        it->slider.range[0] = x[0];
        it->slider.range[1] = x[1];

        // assign divisions if provided, otherwise default
        if (num == 3) {
          it->slider.divisions = mjMAX(1, x[2]);
        } else {
          if (it->type == mjITEM_SLIDERINT) {
            it->slider.divisions = mjMAX(1, x[1]-x[0]);
          } else {
            it->slider.divisions = 200;
          }
        }
      }

      // parse edit numeric
      else if (it->type == mjITEM_EDITINT || it->type == mjITEM_EDITNUM || it->type == mjITEM_EDITFLOAT) {
        // check mjMAXUIEDIT
        if (mjMAXUIEDIT > 7) {
          mju_error("internal error: mjMAXUIEDIT bigger than 7");
        }

        // parse nitem and up to 5 ranges
        num = sscanf(def[n].other, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                     x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9, x+10, x+11, x+12, x+13, x+14);

        // check length, assign
        if (num < 1) {
          mju_error("mjui_add: edit length missing");
        }
        it->edit.nelem = mju_round((mjtNum)x[0]);
        if (it->edit.nelem < 1 || it->edit.nelem > mjMAXUIEDIT) {
          mju_error("mjui_add: invalid edit length");
        }
        if (it->edit.nelem*2 != (num-1) && num > 1) {
          mju_error("mjui_add: incorrent number of edit ranges");
        }

        // copy ranges
        if (num > 1) {
          for (int i=0; i < it->edit.nelem; i++) {
            it->edit.range[i][0] = x[1+2*i];
            it->edit.range[i][1] = x[2+2*i];
          }
        }
      }

      // edit text: nothing to parse
      else if (it->type == mjITEM_EDITTXT) {
        // clear nelem, for style
        it->edit.nelem = 0;
      }
    }

    // invalid item type
    else {
      mju_error("mjui_add: invalid item type");
    }

    // advance to next definition
    n++;
  }
}



// Add definitions to UI section.
void mjui_addToSection(mjUI* ui, int sect, const mjuiDef* def) {
  int nsect = ui->nsect;
  ui->nsect = sect + 1;
  mjui_add(ui, def);
  ui->nsect = nsect;
}



// set item skip flags within section, but not in pass 0
static void setitemskip(mjuiSection* s, int pass) {
  int skip = 0;

  // process section items
  for (int i = 0; i < s->nitem; ++i) {
    mjuiItem* it = s->item + i;

    // pass 0: nothing is skipped
    if (pass == 0) {
      it->skip = 0;
      continue;
    }

    // item is a separator: update skip state for subsequent items
    if (it->type == mjITEM_SEPARATOR) {
      skip = (it->state == mjSEPCLOSED);
    }

    // item is not a separator: set skip state
    else {
      it->skip = skip;
    }
  }
}



// Compute UI sizes: internal function, may be called twice per resize
static void tryresize(mjUI* ui, const mjrContext* con) {
  // scale theme sizes
  int w_master =   SCL(ui->spacing.total,    con);
  int w_scroll =   SCL(ui->spacing.scroll,   con);
  int g_section =  SCL(ui->spacing.section,  con);
  int g_itemside = SCL(ui->spacing.itemside, con);
  int g_itemmid =  SCL(ui->spacing.itemmid,  con);
  int g_itemver =  SCL(ui->spacing.itemver,  con);
  int g_textver =  SCL(ui->spacing.textver,  con);
  int g_label =    SCL(ui->spacing.label,    con);

  // text element height, with gap above and below
  int textheight = con->charHeight + 2*g_textver;

  // column width
  int colwidth = (w_master - w_scroll - 2*g_section - 2*g_itemside - g_itemmid)/2;

  // pass 0 includes skipped items, pass 1 does not
  int Height, MaxHeight;
  for (int pass = 0; pass < 2; ++pass) {
    // init UI heights
    int height = 0;
    int maxheight = 0;

    // process sections
    for (int n = 0; n < ui->nsect; n++) {
      // vertical padding before section
      height += g_section;
      maxheight += g_section;

      // get section pointer
      mjuiSection* s = ui->sect + n;

      // set item skip flags for section, depending on pass
      setitemskip(s, pass);

      // title rectangle
      s->rtitle.left = g_section;
      s->rtitle.width = w_master - w_scroll - 2 * g_section;
      if (s->state == mjSECT_FIXED) {  // fixed section: no title
        s->rtitle.bottom = height;
        s->rtitle.height = 0;
      }
      else {                             // regular section with title
        s->rtitle.bottom = height + textheight;
        s->rtitle.height = textheight;
      }

      // count title height
      height += s->rtitle.height;
      maxheight += s->rtitle.height;

      // init content rectangle
      s->rcontent.left = s->rtitle.left;
      s->rcontent.width = s->rtitle.width;
      s->rcontent.height = 0;
      s->rcontent.bottom = 0;

      // process items within section
      for (int i = 0; i < s->nitem; i++) {
        // get item pointer, clear rectangle
        mjuiItem* it = s->item + i;
        memset(&it->rect, 0, sizeof(mjrRect));

        // item is skipped: nothing to do
        if (it->skip) {
          continue;
        }

        // vertical padding before item
        s->rcontent.height += it->type == mjITEM_SEPARATOR ? g_section : g_itemver;

        // packed pair of items
        if (i < s->nitem - 1 && s->item[i + 1].type == it->type &&
          (it->type == mjITEM_BUTTON ||
            it->type == mjITEM_CHECKINT ||
            it->type == mjITEM_CHECKBYTE)) {
          // get next item pointer
          mjuiItem* it1 = s->item + (i + 1);

          // this item rectangle
          it->rect.left = s->rcontent.left + g_itemside;
          it->rect.width = colwidth;
          it->rect.height = textheight;

          // next item rectangle (set bottom here)
          it1->rect.left = s->rcontent.left + g_itemside + colwidth + g_itemmid;
          it1->rect.width = colwidth;
          it1->rect.height = textheight;
          it1->rect.bottom = height + s->rcontent.height + it->rect.height;

          // advance
          i++;
        }

        // single-line item
        else {
          // common left border (except for labeled controls at the end)
          it->rect.left = s->rcontent.left + g_itemside;

          // static
          if (it->type == mjITEM_STATIC) {
            it->rect.width = s->rcontent.width - 2 * g_itemside;
            it->rect.height = (con->charHeight + g_textver) * it->multi.nelem;
          }

          // single column
          else if (it->type == mjITEM_BUTTON ||
            it->type == mjITEM_CHECKINT ||
            it->type == mjITEM_CHECKBYTE) {
            it->rect.width = colwidth;
            it->rect.height = textheight;
          }

          // radio
          else if (it->type == mjITEM_RADIO) {
            int ncol = ui->radiocol ? ui->radiocol : 2;
            int nrow = (it->multi.nelem - 1) / ncol + 1;
            it->rect.width = s->rcontent.width - 2 * g_itemside;
            it->rect.height = textheight * nrow;
          }

          // separator, select, slider, edit, radioline
          else {
            it->rect.width = s->rcontent.width - 2 * g_itemside;
            it->rect.height = textheight;
          }

          // add room for label
          if (it->name[0] &&
            (it->type >= mjITEM_RADIO ||
              it->type >= mjITEM_RADIOLINE ||
              it->type == mjITEM_STATIC)) {
            it->rect.left = s->rcontent.left + g_itemside + g_label;
            it->rect.width = s->rcontent.width - (2 * g_itemside + g_label);
          }
        }

        // set bottom, count height
        it->rect.bottom = height + s->rcontent.height + it->rect.height;
        s->rcontent.height += it->rect.height;
      }

      // vertical padding after last item, compute bottom
      s->rcontent.height += g_itemver;
      s->rcontent.bottom = height + s->rcontent.height;

      // count content height
      if (s->state != mjSECT_CLOSED) {
        height += s->rcontent.height;
      }
      maxheight += s->rcontent.height;
    }

    // vertical padding after last section
    height += g_section;
    maxheight += g_section;

    // save data: maxheight from pass 0, height from pass 1
    if (pass == 0) {
      MaxHeight = maxheight;
    }
    else {
      Height = height;
    }
  }

  // invert bottom for all sections and items
  for (int n=0; n < ui->nsect; n++) {
    // section
    mjuiSection* s = ui->sect + n;
    s->rtitle.bottom = Height - s->rtitle.bottom;
    s->rcontent.bottom = Height - s->rcontent.bottom;

    // items
    for (int i=0; i < s->nitem; i++) {
      s->item[i].rect.bottom = Height - s->item[i].rect.bottom;
    }
  }

  // assign UI sizes
  ui->width = w_master;
  ui->height = Height;
  ui->maxheight = MaxHeight;
}



// insertion sort of groups of ints: increasing order of leading int
static void insertionsortgroup(int* list, int num, int stride) {
  // allocate buffer of 10 ints, cannot handle more
  if (stride > 10) {
    mju_error("insertionsortgroup cannot handle stride greater than 10");
  }
  int x[10];

  for (int i = 1; i < num; i++) {
    memcpy(x, list + i * stride, sizeof(int) * stride);

    int j = i - 1;
    while (j >= 0 && list[j * stride] > x[0]) {
      memcpy(list + (j + 1) * stride, list + j * stride, sizeof(int) * stride);
      j--;
    }

    memcpy(list + (j + 1) * stride, x, sizeof(int) * stride);
  }
}



// Compute UI sizes.
void mjui_resize(mjUI* ui, const mjrContext* con) {
  // get maximum buffer size allowed by OpenGL driver
  int maxBufferSize = 0;
  glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxBufferSize);

  // USED FOR TESTING OF SMALL BUFFER SIZES
  // maxBufferSize = 3000;

  // resize with current section states, clamp maxheight
  tryresize(ui, con);
  ui->maxheight = mjMIN(ui->maxheight, maxBufferSize);

  // if height is too large, close some sections
  if (ui->height > ui->maxheight) {
    // init new height with section gaps
    int hnew = (ui->nsect + 1) * SCL(ui->spacing.section, con);

    // add titles of regular sections and contents of fixed sections
    for (int n = 0; n < ui->nsect; ++n) {
      if (ui->sect[n].state == mjSECT_FIXED) {
        hnew += ui->sect[n].rcontent.height;
      }
      else {
        hnew += ui->sect[n].rtitle.height;
      }
    }

    // if fixed height is too big, nothing we can do
    if (hnew > ui->maxheight) {
      mju_error("fixed section height already too big, closing sections cannot help");
    }

    // sort open sections by lastclick
    int nopen = 0;
    int sortbuf[2 * mjMAXUISECT] = { 0 };
    for (int n = 0; n < ui->nsect; ++n) {
      if (ui->sect[n].state == mjSECT_OPEN) {
        sortbuf[2 * nopen] = ui->sect[n].lastclick;
        sortbuf[2 * nopen + 1] = n;
        ++nopen;
      }
    }
    insertionsortgroup(sortbuf, nopen, 2);

    // nothing is open; SHOULD NOT OCCUR
    if (nopen == 0) {
      mju_error("internal error: expected some sections to be open");
    }

    // keep most recent sections: as many as can fit in maxheight
    for (int i = nopen - 1; i >= 0; --i) {
      // section fits: add height
      if (hnew + ui->sect[sortbuf[2 * i + 1]].rcontent.height <= ui->maxheight) {
        hnew += ui->sect[sortbuf[2 * i + 1]].rcontent.height;
      }

      // section does not fit: mark for closing
      else {
        sortbuf[2 * i] = -1;
      }
    }

    // close sections that were marked
    for (int i = 0; i < nopen; ++i) {
      if (sortbuf[2 * i] == -1) {
        ui->sect[sortbuf[2 * i + 1]].state = mjSECT_CLOSED;
      }
    }

    // resize with new section states, clamp maxheight again
    tryresize(ui, con);
    ui->maxheight = mjMIN(ui->maxheight, maxBufferSize);

    // make sure tryresize did what we expected; SHOULD NOT OCCUR
    if (ui->height != hnew) {
      mju_error("internal error: tryresize produced unexpeced ui height");
    }
  }
}



// predicate handler
static int evalpredicate(int state, mjfItemEnable predicate, void* userdata) {
  if (state <= 0) {
    return 0;
  } else if (state == 1 || predicate == NULL) {
    return 1;
  } else {
    return predicate(state, userdata);
  }
}



// draw shortcut text
static void shortcuthelp(mjrRect r, int modifier, int shortcut,
                         const mjUI* ui, const mjrContext* con) {
  // map of key codes and corresponding names
#define NMAP 27
  const struct {
    int key;
    const char* value;
  } keymap[NMAP] = {
    {32              , "Space"},
    {mjKEY_ESCAPE    , "Esc"},
    {mjKEY_ENTER     , "Enter"},
    {mjKEY_TAB       , "Tab"},
    {mjKEY_BACKSPACE , "BackSpace"},
    {mjKEY_INSERT    , "Ins"},
    {mjKEY_DELETE    , "Del"},
    {mjKEY_RIGHT     , "Right"},
    {mjKEY_LEFT      , "Left"},
    {mjKEY_DOWN      , "Down"},
    {mjKEY_UP        , "Up"},
    {mjKEY_PAGE_UP   , "PgUp"},
    {mjKEY_PAGE_DOWN , "PgDn"},
    {mjKEY_HOME      , "Home"},
    {mjKEY_END       , "End"},
    {mjKEY_F1        , "F1"},
    {mjKEY_F2        , "F2"},
    {mjKEY_F3        , "F3"},
    {mjKEY_F4        , "F4"},
    {mjKEY_F5        , "F5"},
    {mjKEY_F6        , "F6"},
    {mjKEY_F7        , "F7"},
    {mjKEY_F8        , "F8"},
    {mjKEY_F9        , "F9"},
    {mjKEY_F10       , "F10"},
    {mjKEY_F11       , "F11"},
    {mjKEY_F12       , "F12"}
  };

  // key: ascii or decode map
  char key[10] = "";
  if (shortcut > 32 && shortcut <= 126) {
    key[0] = (char)shortcut;
    key[1] = 0;
  } else {
    for (int i=0; i < NMAP; i++) {
      if (keymap[i].key == shortcut) {
        mjSTRNCPY(key, keymap[i].value);
        break;
      }
    }
  }

  // modifier
  char text[50] = "";
  if (modifier == 1) {
    mjSTRNCPY(text, "Ctrl ");
  } else if (modifier == 2) {
    mjSTRNCPY(text, "Shift ");
  } else if (modifier == 4) {
    mjSTRNCPY(text, "Alt ");
  }

  // combine
  strcat(text, key);

  // make rectangle for shortcut
  int g_textver = SCL(ui->spacing.textver, con);
  int width = textwidth(text, con, -1) + 2*g_textver;
  r.left += (r.width-width);
  r.width = width;
  r.bottom += g_textver;
  r.height -= 2*g_textver;

  // render
  drawrectangle(r, ui->color.shortcut, NULL, con);
  drawtext(text, r.left+g_textver, r.bottom,
           r.width, ui->color.fontactive, con);
}



// Update specific section/item; -1: update all.
void mjui_update(int section, int item, const mjUI* ui,
                 const mjuiState* state, const mjrContext* con) {
  int g_texthor =  SCL(ui->spacing.texthor,  con);
  int g_textver =  SCL(ui->spacing.textver,  con);
  int g_label =    SCL(ui->spacing.label,    con);
  int g_itemside = SCL(ui->spacing.itemside, con);
  int g_itemmid =  SCL(ui->spacing.itemmid,  con);
  int cellheight = con->charHeight + 2*g_textver;
  const float* rgbpane = ui->color.sectpane;

  // locate mouse in ui
  int msect, mitem;
  findmouse(ui, state, con, &msect, &mitem);

  // start rendering
  mjr_setAux(ui->auxid, con);
  mjrRect rgl = { 0, 0, ui->width, ui->height };
  initOpenGL(&rgl, con);

  // all sections: clear background
  if (section < 0) {
    glClearColor(ui->color.master[0], ui->color.master[1],
                 ui->color.master[2], 1);
    glClear(GL_COLOR_BUFFER_BIT);
  }

  // select sections
  int start_section = 0;
  int num_section = ui->nsect;
  if (section >= 0) {
    start_section = section;
    num_section = 1;
  }

  // draw section(s)
  mjtNum sel;
  mjrRect r;
  char text[mjMAXUITEXT];
  int sep[mjMAXUIMULTI+1];
  for (int n=start_section; n < start_section+num_section; n++) {
    // get section pointer
    const mjuiSection* s = ui->sect + n;

    // standard maxwidth for text
    int maxwidth = (s->rtitle.width - 2*g_itemside - g_itemmid)/2 - 2*g_texthor;

    // redraw section title and pane
    if (section < 0 || item < 0) {
      r = s->rtitle;

      // title shown
      if (s->state != mjSECT_FIXED) {
        // section without checkbox
        if (s->checkbox == 0) {
          // interpolated rectangle
          glBegin(GL_QUADS);
          glColor3fv(ui->color.secttitle2);
          glVertex2i(r.left, r.bottom);
          glVertex2i(r.left + r.width, r.bottom);
          glColor3fv(ui->color.secttitle);
          glVertex2i(r.left + r.width, r.bottom + r.height);
          glVertex2i(r.left, r.bottom + r.height);
          glEnd();

          // symbol and text
          drawsymbol(r, s->state, 0, ui, con);
          drawtext(s->name, r.left + g_texthor,
            r.bottom + g_textver, 2 * maxwidth,
            ui->color.sectfont, con);
        }

        // section with checkbox
        else {
          // select colors depending on check state
          const float* rgb = (s->checkbox == 1 ? ui->color.secttitleuncheck
                                               : ui->color.secttitlecheck);
          const float* rgb2 = (s->checkbox == 1 ? ui->color.secttitleuncheck2
                                                : ui->color.secttitlecheck2);

          // draw rectangle with gradient
          glBegin(GL_QUADS);
          glColor3fv(rgb2);
          glVertex2i(r.left, r.bottom);
          glVertex2i(r.left + r.width, r.bottom);
          glColor3fv(rgb);
          glVertex2i(r.left + r.width, r.bottom + r.height);
          glVertex2i(r.left, r.bottom + r.height);
          glEnd();

          // symbol and text with offset
          drawsymbol(r, s->state, s->checkbox, ui, con);
          drawtext(s->name, r.left + r.height,
            r.bottom + g_textver, 2 * maxwidth - r.height,
            ui->color.sectfont, con);

          // draw checkmark as specified
          int cgap = r.height / 4;
          mjrRect cr = {r.left + cgap,
                        r.bottom + cgap,
                        r.height - 2 * cgap,
                        r.height - 2 * cgap};
          float rgbmean[3] = {
            0.5f * (rgb[0] + rgb2[0]),
            0.5f * (rgb[1] + rgb2[1]),
            0.5f * (rgb[2] + rgb2[2]),
          };
          drawrectangle(cr, ui->color.sectsymbol,
            s->checkbox == 1 ? rgbmean : NULL, con);
        }

        // shortcut
        if (ui->mousehelp && s->shortcut) {
          shortcuthelp(r, s->modifier, s->shortcut, ui, con);
        }
      }

      // content pane, active only
      if (s->state != mjSECT_CLOSED) {
        drawrectangle(s->rcontent, ui->color.sectpane, NULL, con);
      }

      // round corners
      mjrRect rround = s->rtitle;
      if (s->state == mjSECT_FIXED) {
        rround = s->rcontent;
      }
      else if (s->state == mjSECT_OPEN) {
        rround.bottom = s->rcontent.bottom;
        rround.height = s->rtitle.height + s->rcontent.height;
      }
      roundcorner(rround, 0, 0, ui, con);
    }

    // closed: skip items
    if (!s->state) {
      continue;
    }

    // select items
    int start_item = 0;
    int num_item = s->nitem;
    if (section >= 0 && item >= 0) {
      start_item = item;
      num_item = 1;
    }

    // draw item(s)
    for (int i=start_item; i < start_item+num_item; i++) {
      // get item pointer
      const mjuiItem* it = s->item + i;

      // zero size: skip
      if (it->rect.height == 0) {
        continue;
      }

      // set rgb pointers to inactive state
      const float* rgbfont = ui->color.fontinactive;
      const float* rgbdecor = ui->color.decorinactive;
      const float* rgbdecor2 = ui->color.decorinactive2;

      // get actual state, adjust font rgb
      int state = evalpredicate(it->state, ui->predicate, ui->userdata);
      if (state) {
        rgbfont = ui->color.fontactive;
      }

      // type-specific draw
      switch (it->type) {
      case mjITEM_SEPARATOR:
        // background
        r = it->rect;
        glBegin(GL_QUADS);
        glColor3fv(ui->color.separator2);
        glVertex2i(r.left, r.bottom);
        glVertex2i(r.left+r.width, r.bottom);
        glColor3fv(ui->color.separator);
        glVertex2i(r.left+r.width, r.bottom+r.height);
        glVertex2i(r.left, r.bottom+r.height);
        glEnd();

        // name
        drawtext(it->name,
                 it->rect.left+g_texthor,
                 it->rect.bottom+g_textver,
                 it->rect.width-2*g_texthor, ui->color.sectfont, con);

        // symbol and round corners for collapsible
        if (it->state >= mjSEPCLOSED) {
          int flg_open = (it->state == mjSEPCLOSED + 1);
          drawsymbol(it->rect, flg_open, 3, ui, con);
          roundcorner(it->rect, flg_open, 1, ui, con);
        }
        break;

      case mjITEM_STATIC:
        r = it->rect;

        // name: at top
        drawtext(it->name,
                 s->rcontent.left+g_itemside+g_texthor,
                 r.bottom+g_textver+(it->multi.nelem-1)*(con->charHeight+g_textver),
                 g_label-2*g_texthor, rgbfont, con);

        // background
        drawrectangle(r, rgbpane, rgbpane, con);

        // text lines
        for (int k=0; k < it->multi.nelem; k++) {
          drawtext(it->multi.name[k],
                   r.left+g_texthor,
                   r.bottom+g_textver+(it->multi.nelem-k-1)*(con->charHeight+g_textver),
                   r.width-2*g_texthor, rgbfont, con);
        }
        break;

      case mjITEM_BUTTON:
        if (state) {
          rgbdecor = ui->color.button;
        }

        // outline or filled, depending on mouse
        if (ui->mousesect == n+1 && ui->mouseitem == i && msect == n+1 && mitem == i) {
          drawoval(it->rect, rgbdecor, NULL, con);
        } else {
          drawoval(it->rect, rgbdecor, rgbpane, con);
        }

        // name
        drawtext(it->name,
                 it->rect.left+g_texthor,
                 it->rect.bottom+g_textver,
                 maxwidth, rgbfont, con);

        // shortcut
        if (ui->mousehelp && it->single.shortcut) {
          shortcuthelp(it->rect, it->single.modifier, it->single.shortcut, ui, con);
        }
        break;

      case mjITEM_CHECKINT:
      case mjITEM_CHECKBYTE:
        {
          if (state) {
            rgbdecor = ui->color.check;
          }

          // get value according to type
          int k;
          if (it->type == mjITEM_CHECKINT) {
            k = *(int*)it->pdata;
          } else {
            k = *(mjtByte*)it->pdata;
          }

          // filled or outline
          if (k) {
            drawrectangle(it->rect, rgbdecor, NULL, con);
          } else {
            drawrectangle(it->rect, rgbdecor, rgbpane, con);
          }

          // name
          drawtext(it->name,
                  it->rect.left+g_texthor,
                  it->rect.bottom+g_textver,
                  maxwidth, rgbfont, con);

          // shortcut
          if (ui->mousehelp && it->single.shortcut)
            shortcuthelp(it->rect, it->single.modifier, it->single.shortcut,
                        ui, con);
        }
        break;

      case mjITEM_RADIO:
        if (state) {
          rgbdecor = ui->color.radio;
        }

        // name: at top
        drawtext(it->name,
                 s->rcontent.left+g_itemside+g_texthor,
                 it->rect.bottom+it->rect.height-cellheight+g_textver,
                 g_label-2*g_texthor, rgbfont, con);

        // outline
        drawrectangle(it->rect, rgbdecor, rgbpane, con);

        // fill selection
        r = radioelement(it, mjMAX(0, mjMIN(*(int*)it->pdata, it->multi.nelem-1)), ui, con);
        drawrectangle(r, rgbdecor, NULL, con);

        // element names
        for (int k=0; k < it->multi.nelem; k++) {
          r = radioelement(it, k, ui, con);
          drawtext(it->multi.name[k],
                   r.left+g_texthor,
                   r.bottom+g_textver,
                   r.width-2*g_texthor, rgbfont, con);
        }
        break;

      case mjITEM_RADIOLINE:
        {
          if (state) {
            rgbdecor = ui->color.radio;
          }

          // name
          drawtext(it->name,
                  s->rcontent.left+g_itemside+g_texthor,
                  it->rect.bottom+g_textver,
                  g_label-2*g_texthor, rgbfont, con);

          // outline
          drawrectangle(it->rect, rgbdecor, rgbpane, con);

          // make separators
          makeradioline(it, con, sep);

          // fill selected
          int k = *(int*)it->pdata;
          r = it->rect;
          r.left += sep[k];
          r.width = sep[k+1] - sep[k];
          drawrectangle(r, rgbdecor, NULL, con);

          // element names
          for (int k=0; k < it->multi.nelem; k++) {
            // compute rectangle for element
            r = it->rect;
            r.left += sep[k];
            r.width = sep[k+1] - sep[k];

            // draw centered
            drawtextrect(r, it->multi.name[k], rgbfont, con);
          }
        }
        break;

      case mjITEM_SELECT:
        if (state) {
          rgbdecor = ui->color.select;
        }

        // name
        drawtext(it->name,
                 s->rcontent.left+g_itemside+g_texthor,
                 it->rect.bottom+g_textver,
                 g_label-2*g_texthor, rgbfont, con);

        // box
        drawrectangle(it->rect, rgbdecor, NULL, con);

        // value: only if non-empty
        if (it->multi.nelem > 0) {
          drawtext(it->multi.name[*(int*)it->pdata],
                   it->rect.left+g_texthor,
                   it->rect.bottom+g_textver,
                   it->rect.width-2*g_texthor, rgbfont, con);
        }

        // draw tracking in mjui_render()
        break;

      case mjITEM_SLIDERINT:
      case mjITEM_SLIDERNUM:
        if (state) {
          rgbdecor = ui->color.slider;
          rgbdecor2 = ui->color.slider2;
        }

        // compute relative slider position
        if (it->type == mjITEM_SLIDERINT) {
          sel = (mjtNum)(*(int*)it->pdata);
        } else {
          sel = *(mjtNum*)it->pdata;
        }
        sel = (sel - it->slider.range[0]) /
              mjMAX(1e-10, (it->slider.range[1] - it->slider.range[0]));
        sel = mjMIN(1, mjMAX(0, sel));

        // fill slider with two colors
        r = it->rect;
        drawrectangle(r, rgbdecor, NULL, con);
        r.width = mju_round(r.width*sel);
        drawrectangle(r, rgbdecor2, NULL, con);

        // show divisions if not too many
        if (it->slider.divisions <= 20) {
          // repare rectangle
          r.width = SCL(2, con);
          r.height = g_textver;

          // draw ticks
          for (int k=1; k < (int)it->slider.divisions; k++) {
            r.left = it->rect.left - r.width/2 +
                     it->rect.width*k/it->slider.divisions;
            drawrectangle(r, rgbpane, NULL, con);
          }
        }

        // name
        drawtext(it->name,
                 s->rcontent.left+g_itemside+g_texthor,
                 it->rect.bottom+g_textver,
                 g_label-2*g_texthor, rgbfont, con);

        // value
        if (it->type == mjITEM_SLIDERINT) {
          mjSNPRINTF(text, "%d", *(int*)it->pdata);
        } else {
          mjSNPRINTF(text, "%.3g", *(mjtNum*)it->pdata);
        }
        drawtext(text,
                 it->rect.left+g_texthor,
                 it->rect.bottom+g_textver,
                 it->rect.width-2*g_texthor, rgbfont, con);
        break;

      case mjITEM_EDITINT:
      case mjITEM_EDITNUM:
      case mjITEM_EDITFLOAT:
      case mjITEM_EDITTXT:
        if (state) {
          rgbdecor = ui->color.edit;
        }

        // name
        drawtext(it->name,
                 s->rcontent.left+g_itemside+g_texthor,
                 it->rect.bottom+g_textver,
                 g_label-2*g_texthor, rgbfont, con);

        // activated
        if (ui->editsect > 0 && ui->editsect == n+1 && ui->edititem == i) {
          // fill box, indicate valid state with color
          if (checkedit(ui->edittext, it) == 0) {
            drawrectangle(it->rect, rgbdecor, NULL, con);
          } else {
            drawrectangle(it->rect, ui->color.edit2, NULL, con);
          }

          // show cursor
          int k = textwidth(ui->edittext + ui->editscroll, con,
                            ui->editcursor - ui->editscroll);
          r.left = it->rect.left + g_texthor + k - SCL(1, con);
          r.width = 2*SCL(1, con);
          r.bottom = it->rect.bottom + g_textver/2;
          r.height = it->rect.height - 2*(g_textver/2);
          drawrectangle(r, ui->color.cursor, NULL, con);

          // value from edittext, with scroll
          drawtext(ui->edittext + ui->editscroll,
                   it->rect.left+g_texthor,
                   it->rect.bottom+g_textver,
                   it->rect.width-2*g_texthor, rgbfont, con);
        }

        // not activated
        else {
          // outline box
          drawrectangle(it->rect, rgbdecor, rgbpane, con);

          // value from pdata, no scroll
          array2text(text, it);
          drawtext(text,
                   it->rect.left+g_texthor,
                   it->rect.bottom+g_textver,
                   it->rect.width-2*g_texthor, rgbfont, con);
        }
        break;

      default:
        mju_error("mjui_update: internal error: unexpected item type");
      }
    }
  }

  // stop rendering
  mjr_restoreBuffer(con);
}



// Handle UI event, return pointer to changed item, NULL if no change.
mjuiItem* mjui_event(mjUI* ui, mjuiState* state, const mjrContext* con) {
  int i, key, change = 0;
  mjuiItem* it;
  ui->editchanged = NULL;

  // count mouse clicks over UI
  if (state->type == mjEVENT_PRESS) {
    ++ui->mouseclicks;
  }

  // non-left mouse events: handle shortcut help
  if ((state->type == mjEVENT_PRESS || state->type == mjEVENT_MOVE ||
       state->type == mjEVENT_RELEASE) && state->button != mjBUTTON_LEFT) {
    if (state->button == mjBUTTON_RIGHT) {
      if (state->type == mjEVENT_PRESS) {
        ui->mousehelp = 1;
        mjui_update(-1, -1, ui, state, con);
      } else if (state->type == mjEVENT_RELEASE) {
        ui->mousehelp = 0;
        mjui_update(-1, -1, ui, state, con);
      }
    }

    return NULL;
  }

  // get current mouse section and item
  int sect_cur = -1;
  int item_cur = -1;
  mjuiItem* it_cur = NULL;
  findmouse(ui, state, con, &sect_cur, &item_cur);
  if (sect_cur > 0 && item_cur >= 0) {
    it_cur = ui->sect[sect_cur-1].item + item_cur;
  }

  // update section lastclick
  if (sect_cur > 0 && state->type == mjEVENT_PRESS) {
    ui->sect[sect_cur - 1].lastclick = ui->mouseclicks;
  }

  // get recorded mouse section and item
  int sect_rec = ui->mousesect;
  int item_rec = -1;
  mjuiItem* it_rec = NULL;
  if (sect_rec > 0) {
    item_rec = ui->mouseitem;
    it_rec = ui->sect[sect_rec-1].item + item_rec;
  }

  // get edit section and item
  int sect_edit = ui->editsect;
  int item_edit = -1;
  mjuiItem* it_edit = NULL;
  if (sect_edit > 0) {
    item_edit = ui->edititem;
    it_edit = ui->sect[sect_edit-1].item + item_edit;
  }

  // process according to type
  switch (state->type) {
  case mjEVENT_MOVE:
    // move scrollbar
    if (sect_rec == -1) {
      ui->scroll -= mju_round((state->dy * ui->height) /
                              state->rect[ui->rectid].height);
      ui->scroll = mjMAX(0, mjMIN(ui->scroll,
                                  ui->height-state->rect[ui->rectid].height));
    }

    // item
    else if (it_rec) {
      // move slider
      if (it_rec->type == mjITEM_SLIDERINT || it_rec->type == mjITEM_SLIDERNUM) {
        setslider(it_rec, ui, state, con);

        // redraw, return change
        mjui_update(sect_rec-1, item_rec, ui, state, con);
        return it_rec;
      }

      // move edit
      else if (it_rec->type == mjITEM_EDITINT ||
               it_rec->type == mjITEM_EDITNUM ||
               it_rec->type == mjITEM_EDITFLOAT ||
               it_rec->type == mjITEM_EDITTXT) {
        setcursor(it_rec->rect, ui, state, con);
      }

      // redraw (any item type)
      mjui_update(sect_rec-1, item_rec, ui, state, con);
    }
    return NULL;

  case mjEVENT_PRESS:
    // edit in progress
    if (sect_edit > 0) {
      // same: adjust cursor position
      if (sect_edit == sect_cur && item_edit == item_cur) {
        // start mouse tracking
        ui->mousesect = sect_cur;
        ui->mouseitem = item_cur;

        // adjust cursor and scroll
        setcursor(it_edit->rect, ui, state, con);

        // redraw, return no change
        mjui_update(sect_edit-1, item_edit, ui, state, con);
        return NULL;
      }

      // different: instantiate, record editchange
      if (text2array(ui->edittext, it_edit) == 0) {
        ui->editchanged = it_edit;
      }

      // stop editing, even if invalid
      ui->editsect = 0;

      // redraw
      mjui_update(sect_edit-1, item_edit, ui, state, con);
    }

    // free mouse focus, just in case
    ui->mousesect = 0;

    // start scrollbar drag
    if (sect_cur == -1) {
      ui->mousesect = -1;
      ui->mouseitem = 0;
    }

    // scroll down
    else if (sect_cur == -2) {
      ui->scroll = ui->height - state->rect[ui->rectid].height;
    }

    // scroll up
    else if (sect_cur == -3) {
      ui->scroll = 0;
    }

    // section title
    else if (sect_cur > 0 && item_cur < 0) {
      mjuiSection* se = ui->sect + sect_cur - 1;

      // handle section checkbox
      if (item_cur == -2 && se->checkbox > 0) {
        ui->mousesectcheck = sect_cur;
        return NULL; // leave it to user, because sections may interact
      }
      else {
        ui->mousesectcheck = 0;
      }

      // double-click: make all sections like this (exclude fixed)
      if (state->doubleclick) {
        for (int i=0; i < ui->nsect; i++) {
          if (ui->sect[i].state != mjSECT_FIXED && se->state != mjSECT_FIXED) {
            ui->sect[i].state = se->state;
          }
        }
      }

      // single click: toggle section state (exclude fixed)
      else {
        if (se->state == mjSECT_OPEN) {
          se->state = mjSECT_CLOSED;
        }
        else if (se->state == mjSECT_CLOSED) {
          se->state = mjSECT_OPEN;
        }
      }

      // resize and redraw all
      mjui_resize(ui, con);
      mjui_update(-1, -1, ui, state, con);
    }

    // item
    else if (sect_cur > 0 && item_cur >= 0) {
      // nothing to do for inactive item
      if (!evalpredicate(it_cur->state, ui->predicate, ui->userdata)) {
        return ui->editchanged;
      }

      // process according to type, only if active
      switch (it_cur->type) {
      case mjITEM_SEPARATOR:
        // expanded: collapse
        if (it_cur->state == mjSEPCLOSED+1) {
          it_cur->state = mjSEPCLOSED;
          mjui_resize(ui, con);
          mjui_update(-1, -1, ui, state, con);
        }

        // collapsed: expand
        else if (it_cur->state == mjSEPCLOSED) {
          it_cur->state = mjSEPCLOSED+1;
          mjui_resize(ui, con);
          mjui_update(-1, -1, ui, state, con);
        }

        // nothing else to do for separator
        return ui->editchanged;

      case mjITEM_BUTTON:
        // start tracking
        ui->mousesect = sect_cur;
        ui->mouseitem = item_cur;
        change = 1;
        break;

      case mjITEM_CHECKINT:
        // toggle check
        *(int*)it_cur->pdata = 1 - *(int*)it_cur->pdata;
        change = 1;
        break;

      case mjITEM_CHECKBYTE:
        // toggle check
        *(mjtByte*)it_cur->pdata = 1 - *(mjtByte*)it_cur->pdata;
        change = 1;
        break;

      case mjITEM_RADIO:
        // set selected element
        i = findradio(it_cur, ui, state, con);
        if (i >= 0) {
          *(int*)it_cur->pdata = i;
          change = 1;
        }
        break;

      case mjITEM_RADIOLINE:
        // set selected element
        i = findradioline(it_cur, ui, state, con);
        if (i >= 0) {
          *(int*)it_cur->pdata = i;
          change = 1;
        }
        break;

      case mjITEM_SELECT:
        // start tracking
        ui->mousesect = sect_cur;
        ui->mouseitem = item_cur;
        break;

      case mjITEM_SLIDERINT:
      case mjITEM_SLIDERNUM:
        // set value, start tracking
        setslider(it_cur, ui, state, con);
        ui->mousesect = sect_cur;
        ui->mouseitem = item_cur;
        change = 1;
        break;

      case mjITEM_EDITINT:
      case mjITEM_EDITNUM:
      case mjITEM_EDITFLOAT:
      case mjITEM_EDITTXT:
        // set edit text, clear scroll
        array2text(ui->edittext, it_cur);
        ui->editscroll = 0;

        // set editcursor, adjust editscroll
        setcursor(it_cur->rect, ui, state, con);

        // start tracking: both edit and mouse
        ui->editsect = sect_cur;
        ui->edititem = item_cur;
        ui->mousesect = sect_cur;
        ui->mouseitem = item_cur;
        break;
      }

      // redraw
      mjui_update(sect_cur-1, item_cur, ui, state, con);

      // return if change (otherwise return below)
      if (change) {
        return it_cur;
      }
    }

    // return editchanged, since there was no other change
    return ui->editchanged;

  case mjEVENT_RELEASE:
    // selection box change
    if (it_rec && it_rec->type == mjITEM_SELECT) {
      // find and set value
      i = findselect(it_rec, ui, state, con);
      if (i >= 0) {
        *(int*)it_rec->pdata = i;
      }

      // free mouse and redraw ALL (selection box spills over)
      ui->mousesect = 0;
      mjui_update(-1, -1, ui, state, con);

      // return if value set change
      if (i >= 0) {
        return it_rec;
      }
    }

    // button release
    else if (it_rec && it_rec->type == mjITEM_BUTTON) {
      // free mouse and redraw
      ui->mousesect = 0;
      mjui_update(sect_rec-1, item_rec, ui, state, con);
    }

    // free mouse, return no change
    ui->mousesect = 0;
    return NULL;

  case mjEVENT_SCROLL:
    // adjust scroll
    ui->scroll -= mju_round(state->sy * SCL(ui->spacing.linescroll, con));

    // clamp scroll
    ui->scroll = mjMAX(0, mjMIN(ui->scroll,
                                ui->height-state->rect[ui->rectid].height));
    return NULL;

  case mjEVENT_KEY:
    // editing
    if (sect_edit > 0) {
      // copy key and clear
      key = state->key;
      state->key = 0;

      // process key
      switch (key) {
      case mjKEY_ESCAPE:          // abandon
        ui->editsect = 0;
        break;

      case mjKEY_ENTER:           // instantiate
        if (text2array(ui->edittext, it_edit) == 0) {
          ui->editsect = 0;

          // redraw
          mjui_update(sect_edit-1, item_edit, ui, state, con);

          // record editchanged and return
          ui->editchanged = it_edit;
          return it_edit;
        }
        break;

      case mjKEY_LEFT:            // move cursor left
        // control: emulate Home
        if (state->control) {
          ui->editcursor = 0;
        } else {
          ui->editcursor = mjMAX(0, ui->editcursor-1);
        }
        break;

      case mjKEY_RIGHT:           // move cursor left
        // control: emulate End
        if (state->control) {
          ui->editcursor = strlen(ui->edittext);
        } else {
          ui->editcursor = mjMIN(strlen(ui->edittext), ui->editcursor+1);
        }
        break;

      case mjKEY_HOME:            // move cursor to first position
        ui->editcursor = 0;
        break;

      case mjKEY_END:             // move cursor to last position
        ui->editcursor = strlen(ui->edittext);
        break;

      case mjKEY_BACKSPACE:       // delete before cursor
        if (ui->editcursor > 0) {
          // shift chars after cursor to the left
          for (int i=ui->editcursor; i <= strlen(ui->edittext); i++) {
            ui->edittext[i-1] = ui->edittext[i];
          }

          // move cursor left
          ui->editcursor = ui->editcursor - 1;
        }
        break;

      case mjKEY_DELETE:          // delete after cursor
        if (ui->editcursor < strlen(ui->edittext)) {
          // shift chars after cursor to the left
          for (int i=ui->editcursor; i <= strlen(ui->edittext); i++) {
            ui->edittext[i] = ui->edittext[i+1];
          }
        }
        break;

      default:                    // insert valid key at cursor
        key = validkey(key, strlen(ui->edittext), it_edit->type, state);
        if (key) {
          // shift chars after cursor to the right
          for (int i=strlen(ui->edittext); i >= ui->editcursor; i--) {
            ui->edittext[i+1] = ui->edittext[i];
          }

          // set at cursor
          ui->edittext[ui->editcursor] = key;

          // move cursor right
          ui->editcursor = ui->editcursor + 1;
        }
      }

      // reveal cursor if still editing
      if (ui->editsect > 0) {
        revealcursor(it_edit->rect, ui, con);
      }

      // redraw
      mjui_update(sect_edit-1, item_edit, ui, state, con);
    }

    // shortcut search
    else {
      // search section shortcuts
      for (int n=0; n < ui->nsect; n++)
        if (matchshortcut(state, ui->sect[n].modifier, ui->sect[n].shortcut)) {
          // collapse all
          for (int i=0; i < ui->nsect; i++) {
            if (ui->sect[i].state < 2) {
              ui->sect[i].state = 0;
            }
          }

          // expand this
          if (ui->sect[n].state < 2) {
            ui->sect[n].state = 1;
          }

          // size and update all
          mjui_resize(ui, con);
          mjui_update(-1, -1, ui, state, con);

          // clear key
          state->key = 0;
          return NULL;
        }

      // search item shortcuts
      for (int n=0; n < ui->nsect; n++) {
        for (int i=0; i < ui->sect[n].nitem; i++) {
          // get pointer to item
          it = ui->sect[n].item + i;;

          // check
          if ((it->type == mjITEM_BUTTON || it->type == mjITEM_CHECKINT ||
               it->type == mjITEM_CHECKBYTE) &&
              matchshortcut(state, it->single.modifier, it->single.shortcut)) {
            // clear key
            state->key = 0;

            // active: process shortcut
            if (evalpredicate(it->state, ui->predicate, ui->userdata)) {
              // toggle if check
              if (it->type == mjITEM_CHECKINT) {
                *(int*)it->pdata = 1 - *(int*)it->pdata;
                mjui_update(n, i, ui, state, con);
              } else if (it->type == mjITEM_CHECKBYTE) {
                *(mjtByte*)it->pdata = 1 - *(mjtByte*)it->pdata;
                mjui_update(n, i, ui, state, con);
              }

              // return item
              return it;
            }

            // inactive: nothing to do
            else {
              return NULL;
            }
          }
        }
      }
    }

    return NULL;

  case mjEVENT_RESIZE:
    // clamp scroll
    ui->scroll = mjMAX(0, mjMIN(ui->scroll, ui->height - state->rect[ui->rectid].height));
    return NULL;
  }

  // should not get here, but just in case
  return NULL;
}



// Copy UI image to current buffer.
void mjui_render(mjUI* ui, const mjuiState* state, const mjrContext* con) {
  // get ui rectangle
  mjrRect rect = state->rect[ui->rectid];

  // clear entire rectangle
  mjr_rectangle(rect, ui->color.master[0], ui->color.master[1],
                ui->color.master[2], 1);

  // adjust scroll
  if (ui->scroll > 0 && ui->height-ui->scroll < rect.height) {
    ui->scroll = mjMAX(0, ui->height - rect.height);
  }

  // blit to current buffer
  mjrRect raux = {0, mjMAX(0, ui->height - ui->scroll - rect.height),
                  ui->width, mjMIN(rect.height, ui->height - ui->scroll)};
  mjr_blitAux(ui->auxid, raux, rect.left,
              rect.bottom + mjMAX(0, rect.height - ui->height + ui->scroll), con);

  // draw scrollbar over blit if needed
  if (ui->height > rect.height) {
    // construct rectangles
    mjrRect bar;
    mjrRect thumb;
    scrollrect(rect, ui, con, &bar, &thumb);

    // draw
    mjr_rectangle(thumb, ui->color.thumb[0], ui->color.thumb[1],
                  ui->color.thumb[2], 1);
  }

  // draw selection box tracking over blit if needed
  if (ui->mousesect > 0 && ui->mouseitem >= 0) {
    // get item pointer
    const mjuiItem* it = ui->sect[ui->mousesect-1].item + ui->mouseitem;

    // proceed if select type
    if (it->type == mjITEM_SELECT) {
      // get relevant sizes
      int g_texthor = SCL(ui->spacing.texthor, con);
      int g_textver = SCL(ui->spacing.textver, con);
      int g_itemside = SCL(ui->spacing.itemside, con);
      int cellheight = con->charHeight + 2 * g_textver;
      int offset = mjMAX(0, rect.height - ui->height + ui->scroll) -
                   mjMAX(0, ui->height - ui->scroll - rect.height);

      // margin
      mjrRect r = it->rect;
      r.left -= g_itemside;
      r.width += 2*g_itemside;
      r.height = it->multi.nelem * cellheight + g_itemside;
      r.bottom -= r.height;
      r.bottom += offset;
      r.left += rect.left;
      mjr_rectangle(r, ui->color.sectpane[0],
        ui->color.sectpane[1], ui->color.sectpane[2], 1);

      // box
      r = it->rect;
      r.height = it->multi.nelem * cellheight;
      r.bottom -= r.height;
      r.bottom += offset;
      r.left += rect.left;
      mjr_rectangle(r, ui->color.select2[0],
        ui->color.select2[1], ui->color.select2[2], 1);

      // highlight row under mouse
      int k = findselect(it, ui, state, con);
      if (k >= 0) {
        mjrRect r1 = r;
        r1.bottom = r.bottom + (it->multi.nelem-1-k)*cellheight;
        r1.height = cellheight;
        mjr_rectangle(r1, ui->color.select[0],
          ui->color.select[1], ui->color.select[2], 1);
      }

      // text values
      initOpenGL(&rect, con);
      for (int k=0; k < it->multi.nelem; k++) {
        drawtext(it->multi.name[k],
                 r.left+g_texthor - rect.left,
                 r.bottom+g_textver+(it->multi.nelem-1-k)*cellheight,
                 r.width-2*g_texthor, ui->color.fontactive, con);
      }
    }
  }
}
