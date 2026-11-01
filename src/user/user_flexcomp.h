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

#ifndef MUJOCO_SRC_USER_USER_FLEXCOMP_H_
#define MUJOCO_SRC_USER_USER_FLEXCOMP_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <mujoco/mjspec.h>
#include "user/user_model.h"
#include "user/user_objects.h"


typedef enum _mjtFcompType {
  mjFCOMPTYPE_GRID = 0,
  mjFCOMPTYPE_BOX,
  mjFCOMPTYPE_CYLINDER,
  mjFCOMPTYPE_ELLIPSOID,
  mjFCOMPTYPE_SQUARE,
  mjFCOMPTYPE_DISC,
  mjFCOMPTYPE_CIRCLE,
  mjFCOMPTYPE_MESH,
  mjFCOMPTYPE_GMSH,
  mjFCOMPTYPE_DIRECT,

  mjNFCOMPTYPES
} mjtFcompType;


typedef enum _mjtDof {
  mjFCOMPDOF_FULL = 0,
  mjFCOMPDOF_RADIAL,
  mjFCOMPDOF_TRILINEAR,
  mjFCOMPDOF_QUADRATIC,
  mjFCOMPDOF_2D,

  mjNFCOMPDOFS
} mjtDof;


class mjCFlexcomp {
 public:
  mjCFlexcomp(void);
  bool Make(mjsBody* body, char* error, int error_sz, const mjVFS* vfs = nullptr);

  bool MakeGrid(char* error, int error_sz);
  bool MakeBox(char* error, int error_sz, int dim, bool open = true);
  bool MakeSquare(char* error, int error_sz);
  bool MakeMesh(mjCModel*    model,
                mjsCompiler* compiler,
                char*        error,
                int          error_sz,
                const mjVFS* vfs = nullptr);
  bool MakeGMSH(mjCModel*    model,
                mjsCompiler* compiler,
                char*        error,
                int          error_sz,
                const mjVFS* vfs = nullptr);


  int GridID(int ix, int iy);
  int GridID(int ix, int iy, int iz);

  // common properties set by user
  std::string  name;          // flex name
  mjtFcompType type;          // flexcomp type
  int          count[3];      // grid count in each dimension
  int          cellcount[3];  // number of cells for interpolation
  double       spacing[3];    // spacing between grid elements
  double       scale[3];      // scaling for mesh and direct
  double       origin[3];     // origin for generating a 3D mesh from a convex 2D mesh
  double       mass;          // total mass of auto-generated bodies
  double       inertiabox;    // size of inertia box for each body
  int          equality;      // create equality constraint, 0:none, 1:edge, 2:vert, 3:strain
  std::string  file;          // mesh/gmsh file name
  std::string  mesh;          // referenced mesh asset name
  mjtDof       doftype;       // dof type, all vertices or trilinear interpolation
  bool         has_dim;       // whether dim was explicitly specified
  bool         has_origin;    // whether origin was explicitly specified
  bool         has_flatskin;  // whether flatskin was explicitly specified
  bool         file_or_mesh;  // file could also have been a mesh name (mjs_makeFlex)

  // pin specifications
  std::vector<int> pinid;         // ids of points to pin
  std::vector<int> pinrange;      // range of ids to pin
  std::vector<int> pingrid;       // grid coordinates to pin
  std::vector<int> pingridrange;  // range of grid coordinates to pin

  // all other properties
  mjCDef def;  // local copy, parsed parameters stored here

  // pose transform relative to parent body
  double         pos[3];   // position
  double         quat[4];  // orientation
  mjsOrientation alt;      // alternative orientation
  mjsFrame*      frame;    // enclosing frame

  // set by user or computed internally
  bool                rigid;         // all vertices are in parent body (all pinned)
  bool                centered;      // all vertex coordinates are (0,0,0) (nothing pinned)
  std::vector<double> point;         // flex bodies/vertices
  std::vector<bool>   pinned;        // is point pinned (true: no new body)
  std::vector<bool>   used;          // is point used by any element (false: skip)
  std::vector<int>    element;       // flex elements
  std::vector<float>  texcoord;      // vertex texture coordinates
  std::vector<int>    elemtexcoord;  // face texture coordinates (OBJ only)

  // plugin support
  std::string plugin_name;
  std::string plugin_instance_name;
  mjsPlugin   plugin;

 private:
  // identify empty cells and pin nodes exclusively in empty cells
  void MarkEmptyCells(
      mjCFlex* flex, const double* points, const double minmax[6], int nx, int ny, int nz);
};


struct TriCellData {
  double a[3];
  double b[3];
  double c[3];
};


// evaluate solid angle of a single triangle t relative to x, divided by 4*pi
inline double TriangleWinding(const double x[3], const TriCellData& t) {
  double ra[3] = {t.a[0] - x[0], t.a[1] - x[1], t.a[2] - x[2]};
  double rb[3] = {t.b[0] - x[0], t.b[1] - x[1], t.b[2] - x[2]};
  double rc[3] = {t.c[0] - x[0], t.c[1] - x[1], t.c[2] - x[2]};
  double la    = std::sqrt(ra[0] * ra[0] + ra[1] * ra[1] + ra[2] * ra[2]);
  double lb    = std::sqrt(rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2]);
  double lc    = std::sqrt(rc[0] * rc[0] + rc[1] * rc[1] + rc[2] * rc[2]);
  double labc  = la * lb * lc;
  if (labc < 1e-30) { return 1.0; }
  double bxc[3] = {
      rb[1] * rc[2] - rb[2] * rc[1],
      rb[2] * rc[0] - rb[0] * rc[2],
      rb[0] * rc[1] - rb[1] * rc[0],
  };
  double num = ra[0] * bxc[0] + ra[1] * bxc[1] + ra[2] * bxc[2];
  double ab  = ra[0] * rb[0] + ra[1] * rb[1] + ra[2] * rb[2];
  double bc  = rb[0] * rc[0] + rb[1] * rc[1] + rb[2] * rc[2];
  double ca  = rc[0] * ra[0] + rc[1] * ra[1] + rc[2] * ra[2];
  double den = labc + ab * lc + bc * la + ca * lb;
  if (std::abs(num) <= 1e-12 * labc) {
    if (den <= 1e-12 * labc) { return 1.0; }
    return 0.0;
  }
  return 2.0 * std::atan2(num, den) / (4.0 * mjPI);
}


// Fast Winding Number (Barill et al. 2018) hierarchical BVH node for surface triangles
struct WindingBVHNode {
  double p[3]    = {0, 0, 0};  // area-weighted centroid
  double r_max   = 0;          // max distance from p to any vertex in subtree
  double A[3]    = {0, 0, 0};  // dipole moment: sum_t 0.5 * (b - a) x (c - a)
  double Q[3][3] = {};         // quadrupole moment: sum_t (c_t - p) (x) A_t
  int    left    = -1;
  int    right   = -1;
  int    begin   = 0;
  int    end     = 0;
};


struct FastWindingBVH {
  const std::vector<TriCellData>*    tris = nullptr;
  std::vector<int>                   tri_indices;
  std::vector<double>                tri_areas;
  std::vector<std::array<double, 3>> tri_centroids;
  std::vector<std::array<double, 3>> tri_normals;
  std::vector<WindingBVHNode>        nodes;

  void Build(const std::vector<TriCellData>& in_tris) {
    tris   = &in_tris;
    int ne = static_cast<int>(in_tris.size());
    tri_indices.resize(ne);
    tri_areas.resize(ne);
    tri_centroids.resize(ne);
    tri_normals.resize(ne);
    for (int i = 0; i < ne; i++) {
      tri_indices[i]       = i;
      const TriCellData& t = in_tris[i];
      for (int d = 0; d < 3; d++) { tri_centroids[i][d] = (t.a[d] + t.b[d] + t.c[d]) / 3.0; }
      double ab[3] = {t.b[0] - t.a[0], t.b[1] - t.a[1], t.b[2] - t.a[2]};
      double ac[3] = {t.c[0] - t.a[0], t.c[1] - t.a[1], t.c[2] - t.a[2]};
      double n[3]  = {
          ab[1] * ac[2] - ab[2] * ac[1],
          ab[2] * ac[0] - ab[0] * ac[2],
          ab[0] * ac[1] - ab[1] * ac[0],
      };
      for (int d = 0; d < 3; d++) { tri_normals[i][d] = 0.5 * n[d]; }
      tri_areas[i] = 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    }
    nodes.clear();
    nodes.reserve(2 * std::max(1, ne));
    if (ne > 0) { BuildNode(0, ne); }
  }

  int BuildNode(int begin, int end) {
    int node_idx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    double w_sum     = 0.0;
    double p_sum[3]  = {0, 0, 0};
    double c_mean[3] = {0, 0, 0};
    double A_sum[3]  = {0, 0, 0};
    double c_min[3]  = {1e30, 1e30, 1e30};
    double c_max[3]  = {-1e30, -1e30, -1e30};

    for (int i = begin; i < end; i++) {
      int    tid  = tri_indices[i];
      double w    = tri_areas[tid];
      w_sum      += w;
      for (int d = 0; d < 3; d++) {
        double c   = tri_centroids[tid][d];
        p_sum[d]  += w * c;
        c_mean[d] += c;
        A_sum[d]  += tri_normals[tid][d];
        c_min[d]   = std::min(c_min[d], c);
        c_max[d]   = std::max(c_max[d], c);
      }
    }

    double p[3];
    int    count = end - begin;
    if (w_sum > 1e-30) {
      for (int d = 0; d < 3; d++) { p[d] = p_sum[d] / w_sum; }
    } else {
      for (int d = 0; d < 3; d++) { p[d] = c_mean[d] / count; }
    }

    double r2_max  = 0.0;
    double Q[3][3] = {};
    for (int i = begin; i < end; i++) {
      int                tid = tri_indices[i];
      const TriCellData& t   = (*tris)[tid];
      for (const double* v : {t.a, t.b, t.c}) {
        double dx = v[0] - p[0], dy = v[1] - p[1], dz = v[2] - p[2];
        r2_max = std::max(r2_max, dx * dx + dy * dy + dz * dz);
      }
      double dc[3] = {
          tri_centroids[tid][0] - p[0],
          tri_centroids[tid][1] - p[1],
          tri_centroids[tid][2] - p[2],
      };
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) { Q[j][k] += dc[j] * tri_normals[tid][k]; }
      }
    }

    WindingBVHNode& node = nodes[node_idx];
    for (int d = 0; d < 3; d++) {
      node.p[d] = p[d];
      node.A[d] = A_sum[d];
    }
    node.r_max = std::sqrt(r2_max);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) { node.Q[j][k] = Q[j][k]; }
    }
    node.begin = begin;
    node.end   = end;

    if (count > 16) {
      int    axis = 0;
      double ext0 = c_max[0] - c_min[0];
      double ext1 = c_max[1] - c_min[1];
      double ext2 = c_max[2] - c_min[2];
      if (ext1 > ext0 && ext1 >= ext2) {
        axis = 1;
      } else if (ext2 > ext0 && ext2 > ext1) {
        axis = 2;
      }
      if (c_max[axis] - c_min[axis] > 1e-20) {
        int mid = begin + count / 2;
        std::nth_element(
            tri_indices.begin() + begin,
            tri_indices.begin() + mid,
            tri_indices.begin() + end,
            [this, axis](int a, int b) { return tri_centroids[a][axis] < tri_centroids[b][axis]; });
        int left_idx          = BuildNode(begin, mid);
        int right_idx         = BuildNode(mid, end);
        nodes[node_idx].left  = left_idx;
        nodes[node_idx].right = right_idx;
      }
    }
    return node_idx;
  }

  double Evaluate(const double x[3]) const {
    if (nodes.empty()) { return 0.0; }
    bool on_surface = false;
    return EvaluateNode(0, x, on_surface);
  }

  double EvaluateNode(int node_idx, const double x[3], bool& on_surface) const {
    const WindingBVHNode& node = nodes[node_idx];
    double                d[3] = {node.p[0] - x[0], node.p[1] - x[1], node.p[2] - x[2]};
    double                r2   = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];

    // Multipole acceptance criterion: distance r > 2.0 * r_max
    constexpr double kBeta2 = 4.0;
    if (r2 > kBeta2 * node.r_max * node.r_max && r2 > 1e-30) {
      double inv_r  = 1.0 / std::sqrt(r2);
      double inv_r2 = inv_r * inv_r;
      double inv_r3 = inv_r2 * inv_r;

      double dipole = (d[0] * node.A[0] + d[1] * node.A[1] + d[2] * node.A[2]) * inv_r3;
      double trQ    = node.Q[0][0] + node.Q[1][1] + node.Q[2][2];
      double dQd    = 0.0;
      for (int j = 0; j < 3; j++) {
        double Qd_j  = node.Q[j][0] * d[0] + node.Q[j][1] * d[1] + node.Q[j][2] * d[2];
        dQd         += d[j] * Qd_j;
      }
      double quad = (trQ - 3.0 * dQd * inv_r2) * inv_r3;
      return (dipole + quad) / (4.0 * mjPI);
    }

    if (node.left < 0) {
      double sum = 0.0;
      for (int i = node.begin; i < node.end; i++) {
        double w = TriangleWinding(x, (*tris)[tri_indices[i]]);
        if (w >= 1.0) {
          on_surface = true;
          return 1.0;
        }
        sum += w;
      }
      return sum;
    }

    double wl = EvaluateNode(node.left, x, on_surface);
    if (on_surface) { return 1.0; }
    double wr = EvaluateNode(node.right, x, on_surface);
    if (on_surface) { return 1.0; }
    return wl + wr;
  }
};

#endif  // MUJOCO_SRC_USER_USER_FLEXCOMP_H_
