// Copyright 2016 Svetoslav Kolev
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

#include <math.h>

#include "engine/engine_collision_primitive.h"
#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_misc.h"


// hard-clamp vector to range [-limit(i), +limit(i)]
static void mju_clampVec(mjtNum* vec, const mjtNum* limit, int n) {
  for (int i = 0; i < n; i++) {
    // loop over active limits
    if (limit[i] > 0) {
      vec[i] = mju_clip(vec[i], -limit[i], limit[i]);
    }
  }
}


// raw sphere : box
int mjraw_SphereBox(mjPreContact* con, mjtNum margin,
                    const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                    const mjtNum* pos2, const mjtNum* mat2, const mjtNum* size2) {
  int i, k;
  mjtNum tmp[3], center[3], clamped[3], deepest[3];
  mjtNum pos[3];
  mjtNum dist, closest;

  mji_sub3(tmp, pos1, pos2);
  mji_mulMatTVec3(center, mat2, tmp);

  mji_copy3(clamped, center);
  mju_clampVec(clamped, size2, 3);

  mji_copy3(deepest, center);
  mji_sub3(tmp, clamped, center);
  dist = mju_normalize3(tmp);

  if (dist - size1[0] > margin)
    return 0;

  // sphere center inside box
  if (dist <= mjMINVAL) {
    closest = (size2[0] + size2[1] + size2[2]) * 2;

    for (i = 0; i < 6; i++) {
      if (closest > mju_abs((i % 2 ? 1 : -1)*size2[i / 2] - center[i / 2])) {
        closest = mju_abs((i % 2 ? 1 : -1) * size2[i / 2] - center[i / 2]);
        k = i;
      }
    }

    mjtNum nearest[3] = {0};
    nearest[k / 2] = (k % 2 ? -1 : 1);

    mji_copy3(pos, center);
    mji_addToScl3(pos, nearest, (size1[0] - closest) / 2);
    mji_mulMatVec3(con[0].normal, mat2, nearest);
    dist = -closest;
  } else {
    mji_addToScl3(deepest, tmp, size1[0]);
    mju_zero3(pos);
    mji_addToScl3(pos, clamped, 0.5);
    mji_addToScl3(pos, deepest, 0.5);
    mji_mulMatVec3(con[0].normal, mat2, tmp);
  }

  mji_mulMatVec3(tmp, mat2, pos);
  mji_add3(con[0].pos, tmp, pos2);
  con[0].dist = dist - size1[0];
  mji_zero3(con[0].tangent);
  return 1;
}


// sphere : box
int mjc_SphereBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  const mjtNum* pos1  = d->geom_xpos + 3*g1;
  const mjtNum* mat1  = d->geom_xmat + 9*g1;
  const mjtNum* size1 = m->geom_size + 3*g1;
  const mjtNum* pos2  = d->geom_xpos + 3*g2;
  const mjtNum* mat2  = d->geom_xmat + 9*g2;
  const mjtNum* size2 = m->geom_size + 3*g2;
  return mjraw_SphereBox(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}


/* GENERAL THEORY OF OPERATION
   the following code is mostly for finding (line segment)/(box) collision
   after which box-sphere is called

   First the closest point to the box is found.
   Then a "sensible" second point is found if the angle
   between the segment and the box is low enough < 45

   In the comments that follow, capsule just means the capsule's line segment
   It might be hard to understand all comments but you would need
   a picture to see what is happening at each line of the code
*/

// raw capsule : box
int mjraw_CapsuleBox(mjPreContact* con, mjtNum margin,
                     const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                     const mjtNum* pos2, const mjtNum* mat2,
                     const mjtNum* size2) {
  mjtNum tmp1[3], tmp2[3], tmp3[3], halfaxis[3], axis[3], dif[3];
  mjtNum pos[3];          // position of capsule in box-local frame

  mjtNum halflength;      // half of capsule's length
  mjtNum bestdist;        // closest contact point distance
  mjtNum bestdistmax;     // init value for bestdist
  mjtNum bestsegmentpos;  // between -1 and 1 :  which point on the segment is closest to the box
  mjtNum secondpos;       // distance of 2nd contact position on capsule segment from the first
  mjtNum dist;
  mjtNum bestboxpos;      // closest contact point, position on the box's edge
  mjtNum mul, e1, e2, dp, de;
  // mjtNum penetration;

  mjtNum ma, mb, mc, u, v, det, x1, x2, idet;  // linelinedist temps

  int s1, s2;        // hold linelinedist info
  int i, j, c1, c2;  // temporary variables
  int cltype = -4;   // closest type
  int clface;        // closest face
  int clcorner = 0;  // closest corner (0..7 in binary)
  int cledge;        // closest edge axis
  int axisdir;       // direction of capsule axis in relation to the box
  int n;             // number of contacts
  int ax1, ax2, ax;  // axis temporaries


  halflength = size1[1];
  secondpos = -4;  // initialize to no 2nd contact (valid values are between -1 and 1)

  mji_sub3(tmp1, pos1, pos2);       // bring capsule to box-local frame (center's box is at (0,0,0))
  mji_mulMatTVec3(pos, mat2, tmp1);  // and axis parallel to world

  tmp1[0] = mat1[2];  // capsule's axis
  tmp1[1] = mat1[5];
  tmp1[2] = mat1[8];

  mji_mulMatTVec3(axis, mat2, tmp1);      // do the same for the capsule axis
  mji_scl3(halfaxis, axis, halflength);  // scale to get actual capsule half-axis

  axisdir = 0;
  if (halfaxis[0] > 0)
    axisdir += 1;
  if (halfaxis[1] > 0)
    axisdir += 2;
  if (halfaxis[2] > 0)
    axisdir += 4;

  // under this notion    "axisdir" and "7-axisdir" point in opposite directions,
  // essentially the same for a capsule

  bestdistmax = margin + 2 * (size1[0] + halflength + size2[0] + size2[1] +
                              size2[2]);  // initialize bestdist
  bestdistmax *= bestdistmax;  // compared against squared distances
  bestdist = bestdistmax;
  bestsegmentpos = 0;

  mju_zero3(tmp2);

  // test to see if maybe the a face of the box is closest to the capsule
  for (i = -1; i <= 1; i += 2) {
    mji_copy3(tmp1, pos);
    mji_addToScl3(tmp1, halfaxis, i);
    mji_copy3(tmp2, tmp1);

    for (c1 = 0, j = 0, c2 = -1; j < 3; j++) {
      if (tmp1[j] < -size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = -size2[j];
      } else if (tmp1[j] > size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = size2[j];
      }
    }

    if (c1 > 1)
      continue;

    mji_subFrom3(tmp1, tmp2);
    dist = mju_dot3(tmp1, tmp1);

    if (dist < bestdist) {
      bestdist = dist;
      bestsegmentpos = i;
      cltype = -2 + i;
      clface = c2;
    }
  }

  mju_zero3(tmp2);

  for (j = 0; j < 3; j++) {
    for (i = 0; i < 8; i++) {
      if ((i & (1 << j)) == 0) {
        // trick to get a corner
        tmp3[0] = ((i & 1) ? 1 : -1) * size2[0];
        tmp3[1] = ((i & 2) ? 1 : -1) * size2[1];
        tmp3[2] = ((i & 4) ? 1 : -1) * size2[2];
        tmp3[j] = 0;

        // tmp3 is the starting point on the box
        // tmp2 is the direction along the "j"-th axis
        // pos is the capsule's center
        // halfaxis is the capsule direction

        // find closest point between capsule and the edge

        mji_sub3(dif, tmp3, pos);

        ma = size2[j] * size2[j];
        mb = -size2[j] * halfaxis[j];
        mc = size1[1] * size1[1];

        u = -size2[j] * dif[j];
        v = mju_dot3(halfaxis, dif);

        det = ma * mc - mb * mb;
        if (mju_abs(det) < mjMINVAL)
          continue;
        idet = 1 / det;


        // sX : X=1 means middle of segment. X=0 or 2 one or the other end

        x1 = (mc * u - mb * v) * idet;
        x2 = (ma * v - mb * u) * idet;

        s1 = s2 = 1;

        if (x1 > 1) {
          x1 = 1;
          s1 = 2;
          x2 = (v - mb) * (1 / mc);
        } else if (x1 < -1) {
          x1 = -1;
          s1 = 0;
          x2 = (v + mb) * (1 / mc);
        }

        if (x2 > 1) {
          x2 = 1;
          s2 = 2;
          x1 = (u - mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        } else if (x2 < -1) {
          x2 = -1;
          s2 = 0;
          x1 = (u + mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        }

        mji_sub3(dif, tmp3, pos);

        mji_addToScl3(dif, halfaxis, -x2);
        dif[j] += size2[j] * x1;

        tmp1[2] = mju_dot3(dif, dif);

        c1 = s1 * 3 + s2;


        // the -MINVAL might not be necessary. Fixes numerical problem when axis is numerically
        // parallel to the box
        if (tmp1[2] < bestdist - mjMINVAL) {
          bestdist = tmp1[2];
          bestsegmentpos = x2;
          bestboxpos = x1;

          // c1<6 means that closest point on the box is at the lower end
          // or in the middle of the edge
          c2 = c1 / 6;

          clcorner = i + (1 << j) * c2;  // which corner is the closest
          cledge = j;   // which axis
          cltype = c1;  // save clamped info
        }
      }
    }
  }


  // penetration = -bestdist;


  for (j = 0; j < 3; j++) {
    if (j == 2) {
      typedef union {
        struct {
          mjtNum x, y;
        };
        mjtNum c[2];
      } d2;
      d2 p, s, dd /*, c, tmp1*/;
      mjtNum uu, vv, w, ee1, best /* ,e2 */, l /* , e3, e4 */;

      bestdist = bestdistmax;

      p.x = pos[0];
      p.y = pos[1];
      dd.x = halfaxis[0];
      dd.y = halfaxis[1];
      s.x = size2[0];
      s.y = size2[1];

      l = sqrt(dd.x * dd.x + dd.y * dd.y);

      uu = dd.x * s.y;
      vv = dd.y * s.x;
      w = dd.x * p.y - dd.y * p.x;


      best = -1;

      ee1 = +uu - vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 0;
        }
      }
      ee1 = -uu - vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 1;
        }
      }
      ee1 = +uu + vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 2;
        }
      }
      ee1 = -uu + vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 3;
        }
      }

      // c.x = s.x * ((c1 / 2) ? -1 : 1);
      // c.y = s.y * ((c1 % 2) ? -1 : 1);

      ee1 = mju_abs(w) / l;
      // e2 = best / l;

      // printf("%g %g      %g %g     %g %g\n",c.x,c.y,d.x,d.y,e1,e2);

      // tmp1.x = c.x - p.x;
      // tmp1.y = c.y - p.y;
      ee1 = dd.x * dd.x + dd.y * dd.y;
      // e2 = tmp1.x * d.x + tmp1.y * d.y;
      // e3 = e2 / e1;


      // printf("%g %g      %g %g     %g %g %g \n",c.x,c.y,d.x,d.y,e1,e2,e3);

      ee1 = p.x + (+s.y - p.y) / dd.y * dd.x;
      // e2 = p.x + (-s.y - p.y) / d.y * d.x;
      // e3 = p.y + (+s.x - p.x) / d.x * d.y;
      // e4 = p.y + (-s.x - p.x) / d.x * d.y;


      // printf("%g %g     %g %g\n",e1,e2,e3,e4);
    }
  }

  // goto skip;   // allow only the closest contact

  // cltype: -3 -1 : face is closest to the capsule
  // cltype: 0..8 : edge is closest to the capsule
  // cltype/3==0 means the lower corner is closest to the capsule (note that edges include corners)
  // cltype/3==2 means the upper corner is closest to the capsule (note that edges include corners)
  // cltype/3==1 means the middle of the edge is closest to the capsule
  // cltype%3==0 means the lower corner is closest to the box (note that edges include corners)
  // cltype%3==2 means the upper corner is closest to the box (note that edges include corners)
  // cltype%3==1 means the middle of the capsule is closest to the box


  // invalid type
  if (cltype == -4)
    return 0;

  if (cltype >= 0 && cltype / 3 != 1) {  // closest to a corner of the box
    c1 = axisdir ^ clcorner;

    // hack to find the relative orientation of capsule and corner
    // there are 2 cases:
    //    1: pointing to or away from the corner
    //    2: oriented along a face or an edge


    if (c1 == 0 || c1 == 7)
      goto skip;  // case 1: no chance of additional contact

    if (c1 == 1 || c1 == 2 || c1 == 4) {
      mul = 1;
      de = 1 - bestsegmentpos;
      dp = 1 + bestsegmentpos;
    }

    if (c1 == 3 || c1 == 5 || c1 == 6) {
      mul = -1;
      c1 = 7 - c1;
      dp = 1 - bestsegmentpos;
      de = 1 + bestsegmentpos;
    }

    // "de" and "dp" distance from first closest point on the capsule to both ends of it
    // mul is a direction along the capsule's axis

    if (c1 == 1)
      ax = 0, ax1 = 1, ax2 = 2;
    if (c1 == 2)
      ax = 1, ax1 = 2, ax2 = 0;
    if (c1 == 4)
      ax = 2, ax1 = 0, ax2 = 1;



    if (axis[ax]*axis[ax] > 0.5) {  // second point along the edge of the box
      secondpos = de;  // initial position from the
      e1 = 2 * size2[ax] / mju_abs(halfaxis[ax]);

      if (e1 < secondpos) {
        secondpos = e1;  // we overshoot, move back to the  other corner of the edge
      }
      secondpos *= mul;
    } else {  // second point along a face of the box
      secondpos = dp;

      // check for overshoot again

      e1 = 2 * size2[ax1] / mju_abs(halfaxis[ax1]);
      if (e1 < secondpos)
        secondpos = e1;

      e1 = 2 * size2[ax2] / mju_abs(halfaxis[ax2]);
      if (e1 < secondpos)
        secondpos = e1;

      secondpos *= -mul;
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {  // we are on box's edge
    // hacks to find the relative orientation of capsule and edge
    // there are 2 cases:
    //    c1= 2^n: edge and capsule are oriented in a T configuration (no more contacts
    //    c1!=2^n: oriented in a cross X configuration

    c1 = axisdir ^ clcorner;  // same trick

    c1 &= 7 - (1 << cledge);  // even more hacks

    // printf("%d %d %d %d    %lf %lf %lf\n",
    //        axisdir,clcorner,c1,cledge,halfaxis[0],halfaxis[1],halfaxis[2]);

    if (c1 != 1 && c1 != 2 && c1 != 4)
      goto skip;


    if (cledge == 0)
      ax1 = 1, ax2 = 2;
    if (cledge == 1)
      ax1 = 2, ax2 = 0;
    if (cledge == 2)
      ax1 = 0, ax2 = 1;
    ax = cledge;


    // Then it finds with which face the capsule has a lower angle and switches the axis names

    if (mju_abs(axis[ax1]) > mju_abs(axis[ax2]))
      ax1 = ax2;
    ax2 = 3 - ax - ax1;

    // keep track of the axis orientation (mul will tell us which direction along the capsule to
    // find the second point) you can notice all other references to the axis "halfaxis" are with
    // absolute value

    if (c1 & (1 << ax2)) {
      mul = 1;
      secondpos = 1 - bestsegmentpos;
    } else {
      mul = -1;
      secondpos = 1 + bestsegmentpos;
    }


    // now we have to find out whether we point towards the opposite side or towards one of the
    // sides and also find the farthest point along the capsule that is above the box

    e1 = 2 * size2[ax2] / mju_abs(halfaxis[ax2]);
    if (e1 < secondpos)
      secondpos = e1;

    if (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0))  // that is insane
      e2 = 1 - bestboxpos;
    else
      e2 = 1 + bestboxpos;

    e1 = size2[ax] * e2 / mju_abs(halfaxis[ax]);

    if (e1 < secondpos)
      secondpos = e1;

    secondpos *= mul;
  } else if (cltype < 0) {
    // similarly we handle the case when one capsule's end is closest to a face of the box
    // and find where is the other end pointing to and clamping to the farthest point
    // of the capsule that's above the box

    if (clface == -1)
      goto skip;  // here the closest point is inside the box, no need for a second point
    if (cltype == -3)
      mul = 1;
    else
      mul = -1;

    secondpos = 2;

    mji_copy3(tmp1, pos);
    mji_addToScl3(tmp1, halfaxis, -mul);

    for (i = 0; i < 3; i++) {
      if (i != clface) {
        e1 = (size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;

        e1 = (-size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;
      }
    }
    secondpos *= mul;
  }


skip:

  // create sphere in original orientation at first contact point
  mju_copy3(tmp1, pos);
  mji_addToScl3(tmp1, halfaxis, bestsegmentpos);
  mju_mulMatVec3(tmp2, mat2, tmp1);
  mju_addTo3(tmp2, pos2);

  // collide with
  n = mjraw_SphereBox(con, margin, tmp2, mat1, size1, pos2, mat2, size2);


  if (secondpos > -3) {  // secondpos was modified
    mju_copy3(tmp1, pos);
    mji_addToScl3(tmp1, halfaxis, secondpos + bestsegmentpos);  // note the summation
    mju_mulMatVec3(tmp2, mat2, tmp1);
    mju_addTo3(tmp2, pos2);
    n += mjraw_SphereBox(con + n, margin, tmp2, mat1, size1, pos2, mat2, size2);
  }

  return n;
}


// capsule : box
int mjc_CapsuleBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  const mjtNum* pos1  = d->geom_xpos + 3*g1;
  const mjtNum* pos2  = d->geom_xpos + 3*g2;
  const mjtNum* mat1  = d->geom_xmat + 9*g1;
  const mjtNum* mat2  = d->geom_xmat + 9*g2;
  const mjtNum* size1 = m->geom_size + 3*g1;
  const mjtNum* size2 = m->geom_size + 3*g2;
  return mjraw_CapsuleBox(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}


// A box-box contact manifold is computed in two stages.
//
// Stage 1, separating-axis test: find the axis of maximum separation among the 15 candidate
// directions (3 face normals per box, 9 cross products of edge directions). If the boxes are
// separated by more than margin along any candidate axis there is no contact. Face axes are
// preferred over edge axes on near-ties: a face axis yields a multi-point manifold, which the
// solver strongly prefers over a single edge contact of nearly identical depth.
//
// Stage 2, manifold generation, depends on the kind of winning axis:
//  - face axis: the owner of the face is the reference box. The face of the other (incident)
//    box least aligned with the reference normal is clipped against the four side planes of
//    the reference face (Sutherland-Hodgman). Every clipped vertex within the margin band
//    becomes a contact. Depth is the distance between the surfaces along the reference
//    normal; contact position is midway between the surfaces along the normal, so it lies
//    inside the intersection of the margin-inflated boxes.
//  - edge axis: the contact is at the midpoint of the closest-point pair between the two
//    supporting edge segments, with depth measured along the separating axis.
//
// Every surviving clipped vertex becomes a contact, so a face manifold carries at most
// mjBOXBOX_MAXVERT points. Reducing the patch below the clipped polygon is not worth it:
// on stacks of plates, whose contact patch is wide relative to their thickness, dropping
// the polygon to a four-point subset costs two to three orders of magnitude in residual
// motion at rest, because the support polygon shrinks and its vertex subset changes from
// step to step as the plates shift.

// Rounding scales, in units of mjtNum epsilon. Supports are sums of products of box
// extents with rotation entries, so their absolute error is proportional to the extents:
// mjBOXBOX_SEPEPS multiplies the summed half-sizes. The rest are dimensionless.
#ifdef mjUSESINGLE
  #define mjBOXBOX_SEPEPS 1e-6f    // slack on the separation tests, times the box scale
  #define mjBOXBOX_PAREPS 1e-7f    // sin^2 below which an edge-cross axis is noise
  #define mjBOXBOX_SGNEPS 1e-5f    // axis component below which a support corner is ambiguous
  #define mjBOXBOX_DUPEPS 1e-10f   // squared relative radius for clip-vertex deduplication
#else
  #define mjBOXBOX_SEPEPS 1e-13
  #define mjBOXBOX_PAREPS 1e-16
  #define mjBOXBOX_SGNEPS 1e-9
  #define mjBOXBOX_DUPEPS 1e-14
#endif

// relative penalty applied to edge-axis separation on near-ties with the best face axis
#define mjBOXBOX_EDGEBIAS 1e-6

// vertex capacity for face clipping: a 4-gon clipped by 4 half-planes has at most 8
// vertices, each of which may become a contact (mjMAXCONPAIR is far above that)
#define mjBOXBOX_MAXVERT 12

// clip polygon *cur (nin vertices) against the half-plane sign*v[coord] <= limit; when
// every vertex is already inside, *cur is left untouched (no copies, the common resting
// case); otherwise the result is written to spare and the buffers are swapped; vertices
// are (x, y, z) with z interpolated as an attribute; returns the vertex count
static int clipHalfPlane(int nin, mjtNum (**cur)[3], mjtNum (**spare)[3],
                         int coord, mjtNum sign, mjtNum limit) {
  mjtNum (*in)[3] = *cur;
  mjtNum d[mjBOXBOX_MAXVERT];
  int all_inside = 1;
  for (int k = 0; k < nin; k++) {
    d[k] = sign*in[k][coord] - limit;
    all_inside &= d[k] <= 0;
  }
  if (all_inside) {
    return nin;
  }

  mjtNum (*out)[3] = *spare;
  int nout = 0;
  for (int k = 0; k < nin; k++) {
    const mjtNum* p = in[k];
    int k1 = k + 1 == nin ? 0 : k + 1;
    mjtNum dp = d[k], dq = d[k1];

    // emit p if inside
    if (dp <= 0 && nout < mjBOXBOX_MAXVERT) {
      mji_copy3(out[nout++], p);
    }

    // emit intersection if the edge strictly crosses the plane
    if (((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) && nout < mjBOXBOX_MAXVERT) {
      const mjtNum* q = in[k1];
      mjtNum t = dp / (dp - dq);
      out[nout][0] = p[0] + t*(q[0] - p[0]);
      out[nout][1] = p[1] + t*(q[1] - p[1]);
      out[nout][2] = p[2] + t*(q[2] - p[2]);
      nout++;
    }
  }
  *cur = out;
  *spare = in;
  return nout;
}


// raw box : box
int mjraw_BoxBox(mjPreContact* con, mjtNum margin,
                 const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                 const mjtNum* pos2, const mjtNum* mat2, const mjtNum* size2) {

  // rot: box2 axes in box1 frame (columns); pos21: box2 center in box1 frame;
  // pos12: box1 center in box2 frame
  mjtNum rot[9], rotabs[9], pos21[3], pos12[3], tmp[3];
  mji_sub3(tmp, pos2, pos1);
  mji_mulMatTVec3(pos21, mat1, tmp);
  mji_sub3(tmp, pos1, pos2);
  mji_mulMatTVec3(pos12, mat2, tmp);
  mju_mulMatTMat3(rot, mat1, mat2);
  for (int i = 0; i < 9; i++) {
    rotabs[i] = mju_abs(rot[i]);
  }

  //------------------------------ stage 1: separating-axis test

  // the separation tests decide contact against no contact, so they carry rounding slack:
  // without it a box pair that genuinely overlaps by less than the rounding error of its
  // own support evaluation is reported as separated, and the boxes pass through each other
  mjtNum septol = margin + mjBOXBOX_SEPEPS*(size1[0] + size1[1] + size1[2] +
                                            size2[0] + size2[1] + size2[2]);

  // best separation so far (most positive; negative = penetration), and the winning axis:
  // code 0..2 face of box1, 3..5 face of box2, >= 6 edge pair (i, j) as 6 + 3*i + j
  mjtNum sep_best = -mjMAXVAL;
  mjtNum sep_face = -mjMAXVAL;
  int code = -1;

  // face axes of box1: candidate normal is axis i of box1
  for (int i = 0; i < 3; i++) {
    mjtNum radius2 = rotabs[3*i+0]*size2[0] + rotabs[3*i+1]*size2[1] + rotabs[3*i+2]*size2[2];
    mjtNum sep = mju_abs(pos21[i]) - size1[i] - radius2;
    if (sep > septol) {
      return 0;
    }
    if (sep > sep_best) {
      sep_best = sep;
      code = i;
    }
  }

  // face axes of box2: candidate normal is axis j of box2
  for (int j = 0; j < 3; j++) {
    mjtNum radius1 = rotabs[0+j]*size1[0] + rotabs[3+j]*size1[1] + rotabs[6+j]*size1[2];
    mjtNum sep = mju_abs(pos12[j]) - size2[j] - radius1;
    if (sep > septol) {
      return 0;
    }
    if (sep > sep_best) {
      sep_best = sep;
      code = 3 + j;
    }
  }
  sep_face = sep_best;
  int code_face = code;

  // edge-cross axes: candidate direction is axis i of box1 crossed with axis j of box2
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // cross product of e_i with column j of rot, in box1 frame; component i is zero
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      mjtNum ax1 = -rot[3*i2+j];
      mjtNum ax2 = rot[3*i1+j];

      // the cross product of two unit vectors has norm sin(angle); for nearly parallel
      // edges the components above are pure cancellation noise and the direction is
      // meaningless, so require sin(angle) well above rounding; the skipped axes are
      // covered by the face normals, which the cross product converges to as the angle
      // vanishes
      mjtNum norm2 = ax1*ax1 + ax2*ax2;
      if (norm2 < mjBOXBOX_PAREPS) {
        continue;
      }
      mjtNum inv = 1/mju_sqrt(norm2);
      ax1 *= inv;
      ax2 *= inv;

      // support radius of box1: component i of axis is zero by construction
      mjtNum radius1 = size1[i1]*mju_abs(ax1) + size1[i2]*mju_abs(ax2);

      // support radius of box2: transform axis to box2 frame; component j is zero there,
      // and only components i1, i2 of the axis are nonzero here
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      mjtNum a2_1 = ax1*rot[3*i1+j1] + ax2*rot[3*i2+j1];
      mjtNum a2_2 = ax1*rot[3*i1+j2] + ax2*rot[3*i2+j2];
      mjtNum radius2 = size2[j1]*mju_abs(a2_1) + size2[j2]*mju_abs(a2_2);

      mjtNum sep = mju_abs(ax1*pos21[i1] + ax2*pos21[i2]) - radius1 - radius2;
      if (sep > septol) {
        return 0;
      }

      // an edge axis must beat the best face axis by a bias-scaled amount: on exact ties
      // the face manifold (multiple points) is strictly better for the solver
      if (sep - mjBOXBOX_EDGEBIAS*mju_abs(sep) > sep_best && sep > sep_face) {
        sep_best = sep;
        code = 6 + 3*i + j;
      }
    }
  }

  if (code < 0) {
    return 0;  // cannot happen: some face axis always sets code
  }

  // a winning edge axis nearly parallel to the best face axis (within ~8 degrees)
  // duplicates it: the face manifold covers the same contact with multiple points, and
  // resting stacks flip between the two codes by rounding noise if the near-tie is
  // allowed to alternate. The face is substituted unless the edge is better by five
  // percent of the face depth (ODE's classic fudge): resting-stack energy degrades
  // continuously as this margin shrinks, while the depth cost of the substitution is
  // bounded by the same five percent. Substituting after the search, rather than
  // filtering during it, prevents a worse non-aliasing edge from stealing the contact
  // that the substitution meant to give to the face.
  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    mjtNum axis[3];
    axis[i] = 0;
    axis[i1] = -rot[3*i2+j];
    axis[i2] = rot[3*i1+j];
    mju_normalize3(axis);
    mjtNum face_dot;
    if (code_face < 3) {
      face_dot = mju_abs(axis[code_face]);
    } else {
      int f = code_face - 3;
      face_dot = mju_abs(axis[0]*rot[0+f] + axis[1]*rot[3+f] + axis[2]*rot[6+f]);
    }
    if (face_dot > 0.99 && sep_best < sep_face + 0.05*mju_abs(sep_face) + mjMINVAL) {
      code = code_face;
      sep_best = sep_face;
    }
  }

  //------------------------------ stage 2a: edge-edge contact

  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    int j1 = (j + 1) % 3, j2 = (j + 2) % 3;

    // unit separating axis in box1 frame, oriented from box1 toward box2
    mjtNum axis[3];
    axis[i] = 0;
    axis[i1] = -rot[3*i2+j];
    axis[i2] = rot[3*i1+j];
    mju_normalize3(axis);
    if (mju_dot3(axis, pos21) < 0) {
      axis[0] = -axis[0];
      axis[1] = -axis[1];
      axis[2] = -axis[2];
    }

    // supporting edges: the box1 edge runs along e_i at a corner selected by the axis
    // signs in (i1, i2); the box2 edge runs along column j at a corner selected by the
    // signs of the axis in box2 coordinates. A near-zero component makes the sign choice
    // meaningless -- both edges support the axis -- and rounding can pick the wrong one,
    // producing witness points on the wrong side of the box. Enumerate both signs for any
    // ambiguous component (at most one per box) and keep the closest witness pair.
    mjtNum a2[3] = {
      axis[0]*rot[0+0] + axis[1]*rot[3+0] + axis[2]*rot[6+0],
      axis[0]*rot[0+1] + axis[1]*rot[3+1] + axis[2]*rot[6+1],
      axis[0]*rot[0+2] + axis[1]*rot[3+2] + axis[2]*rot[6+2],
    };
    const mjtNum ambig = mjBOXBOX_SGNEPS;
    int amb1 = -1, amb2 = -1;
    if (mju_abs(axis[i1]) < ambig) amb1 = i1;
    else if (mju_abs(axis[i2]) < ambig) amb1 = i2;
    if (mju_abs(a2[j1]) < ambig) amb2 = j1;
    else if (mju_abs(a2[j2]) < ambig) amb2 = j2;

    mjtNum d2[3] = {rot[0+j], rot[3+j], rot[6+j]};
    mjtNum b = d2[i];  // d1 . d2, with d1 = e_i
    mjtNum denom = 1 - b*b;

    mjtNum w1[3], w2[3];
    mjtNum best_d2 = mjMAXVAL;
    for (int v1 = 0; v1 < (amb1 >= 0 ? 2 : 1); v1++) {
      for (int v2 = 0; v2 < (amb2 >= 0 ? 2 : 1); v2++) {
        // corner of the box1 edge: support along +axis, ambiguous component flipped by v1
        mjtNum c1[3];
        c1[i] = 0;
        c1[i1] = axis[i1] >= 0 ? size1[i1] : -size1[i1];
        c1[i2] = axis[i2] >= 0 ? size1[i2] : -size1[i2];
        if (amb1 >= 0 && v1) c1[amb1] = -c1[amb1];

        // corner of the box2 edge: support along -axis in box2 coordinates
        mjtNum cc[3];
        cc[j] = 0;
        cc[j1] = a2[j1] >= 0 ? -size2[j1] : size2[j1];
        cc[j2] = a2[j2] >= 0 ? -size2[j2] : size2[j2];
        if (amb2 >= 0 && v2) cc[amb2] = -cc[amb2];
        mjtNum c2[3];
        mji_mulMatVec3(c2, rot, cc);
        mji_addTo3(c2, pos21);

        // closest points between the two edge segments (directions are unit vectors)
        mjtNum e[3];
        mji_sub3(e, c2, c1);
        mjtNum d1e = e[i];  // d1 . e
        mjtNum d2e = mju_dot3(d2, e);
        mjtNum s = denom < mjMINVAL ? 0 : (d1e - b*d2e) / denom;

        // clamp into the segments, letting each clamp re-solve the other parameter
        s = mju_clip(s, -size1[i], size1[i]);
        mjtNum t = mju_clip(b*s - d2e, -size2[j], size2[j]);
        s = mju_clip(d1e + b*t, -size1[i], size1[i]);

        mjtNum p1[3], p2[3], gap[3];
        mji_copy3(p1, c1);
        p1[i] += s;
        mji_copy3(p2, c2);
        mji_addToScl3(p2, d2, t);
        mji_sub3(gap, p2, p1);
        mjtNum gap2 = mju_dot3(gap, gap);
        if (gap2 < best_d2) {
          best_d2 = gap2;
          mji_copy3(w1, p1);
          mji_copy3(w2, p2);
        }
      }
    }

    // signed distance along the axis
    mjtNum gap[3];
    mji_sub3(gap, w2, w1);
    mjtNum dist = mju_dot3(gap, axis);
    if (dist > septol) {
      return 0;
    }

    // contact at the midpoint of the witness pair: for penetrating edges this is inside both
    // boxes; in the margin band it is midway between the two surfaces
    mjtNum mid[3] = {0.5*(w1[0] + w2[0]), 0.5*(w1[1] + w2[1]), 0.5*(w1[2] + w2[2])};

    con[0].dist = dist;
    mji_mulMatVec3(tmp, mat1, mid);
    mji_add3(con[0].pos, tmp, pos1);
    mji_mulMatVec3(con[0].normal, mat1, axis);
    mji_zero3(con[0].tangent);
    return 1;
  }

  //------------------------------ stage 2b: face contact

  // reference box: owner of the winning face; incident box: the other one
  int ref1 = code < 3;              // is box1 the reference?
  int a = ref1 ? code : code - 3;   // face axis of the reference box
  const mjtNum* sizeref = ref1 ? size1 : size2;
  const mjtNum* sizeinc = ref1 ? size2 : size1;
  const mjtNum* posref = ref1 ? pos1 : pos2;
  const mjtNum* matref = ref1 ? mat1 : mat2;
  const mjtNum* posoi = ref1 ? pos21 : pos12;  // incident center in reference frame

  // incident box axes in reference frame: rot maps box2 to box1, transpose maps box1 to box2;
  // rinc(r, c) = component r of incident axis c, in reference frame
  mjtNum rinc[9];
  if (ref1) {
    mju_copy(rinc, rot, 9);
  } else {
    mju_transpose(rinc, rot, 3, 3);
  }

  // face direction: +1 if the incident box lies along +a, else -1
  mjtNum sgn = posoi[a] >= 0 ? 1 : -1;

  // incident face: the face of the incident box most opposed to the reference face normal
  int binc = 0;
  for (int k = 1; k < 3; k++) {
    if (mju_abs(rinc[3*a+k]) > mju_abs(rinc[3*a+binc])) {
      binc = k;
    }
  }
  mjtNum tinc = sgn*rinc[3*a+binc] > 0 ? -1 : 1;  // sign making the incident normal oppose

  // corners of the incident face in reference frame, cyclic winding; the in-plane
  // coordinates are (x, y) = the two non-a reference axes, z is the signed distance
  // above the reference face plane (negative = inside the reference box)
  int ax = (a + 1) % 3, ay = (a + 2) % 3;
  int bu = (binc + 1) % 3, bv = (binc + 2) % 3;
  mjtNum poly[2][mjBOXBOX_MAXVERT][3];

  // face center and in-face half-edge offsets, in the projected (x, y, z) coordinates
  mjtNum cx[3], du[3], dv[3];
  for (int r = 0; r < 3; r++) {
    int c = r == 0 ? ax : (r == 1 ? ay : a);
    cx[r] = posoi[c] + tinc*sizeinc[binc]*rinc[3*c+binc];
    du[r] = sizeinc[bu]*rinc[3*c+bu];
    dv[r] = sizeinc[bv]*rinc[3*c+bv];
  }
  cx[2] = sgn*cx[2] - sizeref[a];
  du[2] *= sgn;
  dv[2] *= sgn;
  static const mjtNum corner_sign[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
  for (int k = 0; k < 4; k++) {
    mjtNum su = corner_sign[k][0], sv = corner_sign[k][1];
    poly[0][k][0] = cx[0] + su*du[0] + sv*dv[0];
    poly[0][k][1] = cx[1] + su*du[1] + sv*dv[1];
    poly[0][k][2] = cx[2] + su*du[2] + sv*dv[2];
  }

  // clip against the four side planes of the reference face; the buffers swap only on
  // passes that actually clip
  int nvert = 4;
  mjtNum (*cur)[3] = poly[0];
  mjtNum (*spare)[3] = poly[1];
  nvert = clipHalfPlane(nvert, &cur, &spare, 0, 1, sizeref[ax]);
  nvert = clipHalfPlane(nvert, &cur, &spare, 0, -1, sizeref[ax]);
  nvert = clipHalfPlane(nvert, &cur, &spare, 1, 1, sizeref[ay]);
  nvert = clipHalfPlane(nvert, &cur, &spare, 1, -1, sizeref[ay]);

  // accept vertices within the margin band, dropping near-duplicates produced by clipping
  // at polygon corners; duplicate radius is relative to the reference face scale
  mjtNum accepted[mjBOXBOX_MAXVERT][3];
  int naccept = 0;
  mjtNum dupe2 = mjBOXBOX_DUPEPS*(sizeref[ax]*sizeref[ax] + sizeref[ay]*sizeref[ay]);
  for (int k = 0; k < nvert; k++) {
    if (cur[k][2] > margin) {
      continue;
    }
    int dupe = 0;
    for (int q = 0; q < naccept; q++) {
      mjtNum dx = accepted[q][0] - cur[k][0];
      mjtNum dy = accepted[q][1] - cur[k][1];
      if (dx*dx + dy*dy < dupe2) {
        dupe = 1;
        break;
      }
    }
    if (!dupe) {
      mji_copy3(accepted[naccept++], cur[k]);
    }
  }
  if (naccept == 0) {
    return 0;
  }

  // world normal points from geom1 to geom2: along +sgn*a of the reference frame when box1
  // is the reference, opposite when box2 is
  mjtNum normal[3];
  mjtNum nsign = ref1 ? sgn : -sgn;
  normal[0] = nsign*matref[3*0+a];
  normal[1] = nsign*matref[3*1+a];
  normal[2] = nsign*matref[3*2+a];

  for (int k = 0; k < naccept; k++) {
    const mjtNum* v = accepted[k];

    // contact position: on the clipped incident polygon in (x, y), midway between the
    // reference face plane and the incident surface along the face axis
    mjtNum posc[3];
    posc[ax] = v[0];
    posc[ay] = v[1];
    posc[a] = sgn*(sizeref[a] + 0.5*v[2]);

    con[k].dist = v[2];
    mji_mulMatVec3(tmp, matref, posc);
    mji_add3(con[k].pos, tmp, posref);
    mji_copy3(con[k].normal, normal);
    mji_zero3(con[k].tangent);
  }
  return naccept;
}


// box : box
int mjc_BoxBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  const mjtNum* pos1 = d->geom_xpos + 3*g1;
  const mjtNum* pos2 = d->geom_xpos + 3*g2;
  const mjtNum* mat1 = d->geom_xmat + 9*g1;
  const mjtNum* mat2 = d->geom_xmat + 9*g2;
  const mjtNum* size1 = m->geom_size + 3*g1;
  const mjtNum* size2 = m->geom_size + 3*g2;
  return mjraw_BoxBox(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}
