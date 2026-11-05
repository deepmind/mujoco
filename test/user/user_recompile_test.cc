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

// Tests for recompiling multiple files.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>  // NOLINT
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include <mujoco/mjmodel.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "src/xml/xml_api.h"
#include "src/xml/xml_numeric_format.h"
#include "test/compare_model.h"
#include "test/compare_spec.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::IsEmpty;
using ::testing::NotNull;

std::vector<std::string> GetRecompileTestModels() {
  std::vector<std::string> models;
  std::string ext(".xml");
  for (const auto& path : {GetTestDataFilePath("."), GetModelPath(".")}) {
    for (const auto& p : std::filesystem::recursive_directory_iterator(path)) {
      if (p.path().extension() == ext) {
        // generic format, so patterns containing '/' also match on Windows
        std::string xml = p.path().generic_string();
        if (  // intentional parse or compile failure tests
            absl::StrContains(xml, "malformed_") ||
            absl::StrContains(xml, "_fail") ||
            absl::StrContains(xml, "xml/testdata/parent_") ||
            // large SDF or benchmark models too slow under sanitizers
            absl::StrContains(xml, "perf") || absl::StrContains(xml, "cow") ||
            absl::StrContains(xml, "100_humanoids")
#ifndef MJ_WITH_USD
            // requires optional USD build support
            || absl::StrContains(xml, "usd.xml")
#endif
        ) {
          continue;
        }
        models.push_back(xml);
      }
    }
  }
  return models;
}

// Differences between two specs, apart from those of a compilation completing
// keyframes: it sizes their vectors for the model, and adds keyframes up to
// the number set by nkey.
std::string WithoutKeyframeCompletion(const std::string& differences) {
  std::istringstream lines(differences);
  std::string line;
  std::string other;
  while (std::getline(lines, line)) {
    bool vector =
        absl::StartsWith(line, "key[") && absl::StrContains(line, " size: ");
    bool number = absl::StartsWith(line, "key: count: ");
    if (!vector && !number) other += line + '\n';
  }
  return other;
}

// The spec as it is saved, empty if it cannot be saved.
std::string SaveToString(const mjSpec* s) {
  std::array<char, 1000> err;
  int size = mj_saveXMLString(s, nullptr, 0, err.data(), err.size());
  if (size <= 0) return "";
  std::string xml(size + 1, '\0');
  mj_saveXMLString(s, xml.data(), size + 1, err.data(), err.size());
  xml.resize(size);
  return xml;
}

// The difference of two values, relative to their magnitude where it exceeds
// one; zero for two NaNs, infinite for unequal integers.
template <typename T>
double Difference(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    if (a == b || (std::isnan(a) && std::isnan(b))) return 0;
    double scale = std::max({1.0, std::abs(static_cast<double>(a)),
                             std::abs(static_cast<double>(b))});
    double dif = std::abs(static_cast<double>(a) - b) / scale;
    return std::isnan(dif) ? std::numeric_limits<double>::infinity() : dif;
  } else {
    return a == b ? 0 : std::numeric_limits<double>::infinity();
  }
}

// values of an mjModel array, as (array, index) pairs
using ModelValues = std::set<std::pair<std::string, int>>;

// The fields in which two models differ by more than tol, one line per field
// with its largest difference, apart from fields whose names start with one of
// the ignored prefixes, and the ignored values.
std::string ModelDifferences(const mjModel* m1, const mjModel* m2, double tol,
                             const std::vector<std::string>& ignored = {},
                             const ModelValues& ignored_values = {}) {
  std::ostringstream out;
  out.precision(17);
  auto compared = [&](const char* name) {
    return std::none_of(ignored.begin(), ignored.end(),
                        [&](const std::string& prefix) {
                          return absl::StartsWith(name, prefix);
                        });
  };

  // sizes: the arrays of models of different sizes are not compared
#define X(name)                                                     \
  if (m1->name != m2->name) {                                       \
    out << #name << ": " << m1->name << " vs " << m2->name << '\n'; \
  }
  MJMODEL_SIZES
#undef X
  if (!out.str().empty()) return out.str();

  // arrays
  MJMODEL_POINTERS_PREAMBLE(m1);
#define X(type, name, nr, nc)                                             \
  if (compared(#name)) {                                                  \
    double maxdif = 0;                                                    \
    int adr = 0;                                                          \
    for (int i = 0; i < m1->nr * (nc); i++) {                             \
      if (ignored_values.count({#name, i})) continue;                     \
      double dif = Difference(m1->name[i], m2->name[i]);                  \
      if (dif > maxdif) {                                                 \
        maxdif = dif;                                                     \
        adr = i;                                                          \
      }                                                                   \
    }                                                                     \
    if (maxdif > tol) {                                                   \
      out << #name << "[" << adr / (nc) << "][" << adr % (nc)             \
          << "]: " << +m1->name[adr] << " vs " << +m2->name[adr] << '\n'; \
    }                                                                     \
  }
  MJMODEL_POINTERS
#undef X

  // scalars which are not sizes
  auto scalar = [&](const std::string& name, double a, double b) {
    if (compared(name.c_str()) && Difference(a, b) > tol) {
      out << name << ": " << a << " vs " << b << '\n';
    }
  };
  scalar("flg_gravcomp", m1->flg_gravcomp, m2->flg_gravcomp);
  scalar("flg_surfacevel", m1->flg_surfacevel, m2->flg_surfacevel);
  scalar("flg_adhesion", m1->flg_adhesion, m2->flg_adhesion);
  auto statistic = [&](const std::string& name, const mjStatistic& s1,
                       const mjStatistic& s2) {
#define X(field, n)                                                          \
  for (int i = 0; i < n; i++) {                                              \
    scalar(name + "." #field, reinterpret_cast<const mjtNum*>(&s1.field)[i], \
           reinterpret_cast<const mjtNum*>(&s2.field)[i]);                   \
  }
#define XVEC X
    MJSTATISTIC_FIELDS
#undef XVEC
#undef X
  };
  statistic("stat", m1->stat, m2->stat);
#define X(type, name, n) scalar("opt." #name, m1->opt.name, m2->opt.name);
#define XVEC(type, name, n)                                 \
  for (int i = 0; i < n; i++) {                             \
    scalar("opt." #name, m1->opt.name[i], m2->opt.name[i]); \
  }
  MJOPTION_FIELDS
#undef X
#undef XVEC
  return out.str();
}

// the values which compilation derives once from the conventions of the spec,
// which mj_setConst leaves as compiled: the damping of actuators given a
// damping ratio, the stiffness and damping of joints given a spring-damper, and
// the spring length of tendons which have none
ModelValues CompiledValues(const mjSpec* s, const mjModel* m) {
  ModelValues values;
  mjSpec* spec = const_cast<mjSpec*>(s);
  for (mjsElement* e = mjs_firstElement(spec, mjOBJ_ACTUATOR); e;
       e = mjs_nextElement(spec, e)) {
    if (mjs_asActuator(e)->biasprm[2] > 0) {
      values.insert({"actuator_biasprm", mjNBIAS * mjs_getId(e) + 2});
    }
  }
  for (mjsElement* e = mjs_firstElement(spec, mjOBJ_JOINT); e;
       e = mjs_nextElement(spec, e)) {
    const double* springdamper = mjs_asJoint(e)->springdamper;
    if (springdamper[0] > 0 && springdamper[1] > 0) {
      int id = mjs_getId(e);
      int type = m->jnt_type[id];
      int ndof = type == mjJNT_FREE ? 6 : (type == mjJNT_BALL ? 3 : 1);
      values.insert({"jnt_stiffness", id});
      for (int k = 0; k < ndof; k++) {
        values.insert({"dof_damping", m->jnt_dofadr[id] + k});
      }
    }
  }
  for (mjsElement* e = mjs_firstElement(spec, mjOBJ_TENDON); e;
       e = mjs_nextElement(spec, e)) {
    const double* springlength = mjs_asTendon(e)->springlength;
    if (springlength[0] == -1 && springlength[1] == -1) {
      values.insert({"tendon_lengthspring", 2 * mjs_getId(e)});
      values.insert({"tendon_lengthspring", 2 * mjs_getId(e) + 1});
    }
  }
  return values;
}

// An edit of a compiled model, of the kind made to randomize its parameters.
struct ModelEdit {
  const char* name;
  void (*apply)(mjModel* m);
  const char* recompiled = "";  // prefix of fields which only compilation sets
};

// geom types whose size is given in the spec
bool IsPrimitive(int type) {
  return type == mjGEOM_PLANE || type == mjGEOM_SPHERE ||
         type == mjGEOM_CAPSULE || type == mjGEOM_ELLIPSOID ||
         type == mjGEOM_CYLINDER || type == mjGEOM_BOX;
}

const ModelEdit kModelEdits[] = {
    {"mass",
     [](mjModel* m) {
       for (int i = 1; i < m->nbody; i++) {
         m->body_mass[i] *= 1.5;
         for (int k = 0; k < 3; k++) m->body_inertia[3 * i + k] *= 1.5;
       }
     }},
    {"com",
     [](mjModel* m) {
       for (int i = 1; i < m->nbody; i++) {
         if (m->body_simple[i]) continue;
         m->body_ipos[3 * i + 0] += 0.01;
         m->body_ipos[3 * i + 1] -= 0.02;
         m->body_ipos[3 * i + 2] += 0.015;
       }
     }},
    {"armature",
     [](mjModel* m) {
       for (int i = 0; i < m->nv; i++) m->dof_armature[i] += 0.05;
     }},
    {"qpos0",
     [](mjModel* m) {
       for (int i = 0; i < m->njnt; i++) {
         if (m->jnt_type[i] == mjJNT_HINGE || m->jnt_type[i] == mjJNT_SLIDE) {
           m->qpos0[m->jnt_qposadr[i]] += 0.05;
         }
       }
     },
     // compilation completes keyframes with qpos0
     "key_qpos"},
    {"qpos_spring",
     [](mjModel* m) {
       for (int i = 0; i < m->njnt; i++) {
         if (m->jnt_type[i] == mjJNT_HINGE || m->jnt_type[i] == mjJNT_SLIDE) {
           m->qpos_spring[m->jnt_qposadr[i]] += 0.05;
         }
       }
     }},
    {"kp",
     [](mjModel* m) {
       for (int i = 0; i < m->nactuator; i++) {
         mjtNum* gainprm = m->actuator_gainprm + mjNGAIN * i;
         mjtNum* biasprm = m->actuator_biasprm + mjNBIAS * i;
         if (gainprm[0] != 0 && gainprm[0] == -biasprm[1]) {
           gainprm[0] *= 2;
           biasprm[1] *= 2;
         }
       }
     }},
    {"tendon_spring",
     [](mjModel* m) {
       for (int i = 0; i < m->ntendon; i++) {
         m->tendon_stiffness[i] = 0;
         m->tendon_damping[i] = 0;
       }
     }},
    {"margin",
     [](mjModel* m) {
       for (int i = 0; i < m->ngeom; i++) m->geom_margin[i] += 0.01;
     }},
    {"camlight",
     [](mjModel* m) {
       for (int i = 0; i < m->ncam; i++) m->cam_pos[3 * i + 2] += 0.1;
       for (int i = 0; i < m->nlight; i++) m->light_pos[3 * i + 2] += 0.1;
     }},
    {"geom_size",
     [](mjModel* m) {
       for (int i = 0; i < m->ngeom; i++) {
         if (!IsPrimitive(m->geom_type[i])) continue;
         for (int k = 0; k < 3; k++) m->geom_size[3 * i + k] *= 1.25;
       }
     }},
    {"geom_pos",
     [](mjModel* m) {
       for (int i = 0; i < m->ngeom; i++) {
         m->geom_pos[3 * i + 0] += 0.01;
         m->geom_pos[3 * i + 1] += 0.02;
         m->geom_pos[3 * i + 2] -= 0.01;
       }
     }},
};

class RecompileCompareTest : public MujocoTest,
                             public ::testing::WithParamInterface<std::string> {
 public:
};
TEST_P(RecompileCompareTest, RecompileCompare) {
  std::string xml = GetParam();
  std::string field = "";

  FullFloatPrecision increase_precision;

  // load spec
  std::array<char, 1000> err;
  mjSpec* s = mj_parseXML(xml.c_str(), 0, err.data(), err.size());

  if (!s) {
    GTEST_SKIP() << "Failed to load " << xml << ": " << err.data();
  }

  // copy spec, the copy has what was authored
  mjSpec* s_copy = mj_copySpec(s);
  const int kAllDifferences = 100000;
  EXPECT_THAT(CompareSpec(s, s_copy, kAllDifferences), IsEmpty()) << xml;

  // an uncompiled spec has no signature
  EXPECT_EQ(s->element->signature, 0) << xml;
  EXPECT_EQ(s_copy->element->signature, 0) << xml;

  // compile twice and compare
  mjModel* m_old = mj_compile(s, nullptr);

  if (!m_old) {
    std::string error_message = mjs_getError(s);
    mj_deleteSpec(s_copy);
    mj_deleteSpec(s);
    GTEST_SKIP() << "Failed to compile " << xml << ": " << error_message;
  }

  // compiling leaves what was authored as it was, unless it restructures
  if (!s->compiler.fusestatic && !s->compiler.discardvisual) {
    EXPECT_THAT(
        WithoutKeyframeCompletion(CompareSpec(s_copy, s, kAllDifferences)),
        IsEmpty())
        << xml;
  }

  // the elements of a compiled spec hold what the model was given, so copying
  // the model back changes nothing in the spec, nor in what is saved
  std::string unchanged = SaveToString(s);
  mjSpec* s_before = mj_copySpec(s);
  EXPECT_EQ(mj_copyBack(s, m_old), 1) << xml << ": " << mjs_getError(s);
  EXPECT_THAT(CompareSpec(s_before, s, kAllDifferences), IsEmpty()) << xml;
  EXPECT_EQ(SaveToString(s), unchanged) << xml;

  // and so do the elements of a copy of the spec
  mjSpec* s_copied = mj_copySpec(s);
  EXPECT_EQ(mj_copyBack(s_copied, m_old), 1)
      << xml << ": " << mjs_getError(s_copied);
  EXPECT_THAT(CompareSpec(s_before, s_copied, kAllDifferences), IsEmpty())
      << xml;
  mj_deleteSpec(s_copied);
  mj_deleteSpec(s_before);

  // a spec which was compiled, and restructured if it asks for it, is left as
  // it is by the next compilation
  mjSpec* s_compiled = mj_copySpec(s);
  mjModel* m_new = mj_compile(s, nullptr);
  EXPECT_THAT(
      WithoutKeyframeCompletion(CompareSpec(s_compiled, s, kAllDifferences)),
      IsEmpty())
      << xml;
  mj_deleteSpec(s_compiled);
  mjModel* m_copy = mj_compile(s_copy, nullptr);

  // compare signature
  EXPECT_EQ(m_old->signature, m_new->signature) << xml;
  EXPECT_EQ(m_old->signature, m_copy->signature) << xml;

  // compiling refreshes the signature of the spec
  EXPECT_EQ(s->element->signature, m_new->signature) << xml;
  EXPECT_EQ(s_copy->element->signature, m_copy->signature) << xml;

  ASSERT_THAT(m_new, NotNull())
      << "Failed to recompile " << xml << ": " << mjs_getError(s);
  ASSERT_THAT(m_copy, NotNull())
      << "Failed to compile " << xml << ": " << mjs_getError(s_copy);

  mjtNum tol = 0;

  EXPECT_LE(CompareModel(m_old, m_new, field), tol)
      << "Compiled and recompiled models are different!\n"
      << "Affected file " << xml << '\n'
      << "Different field: " << field << '\n';

  EXPECT_LE(CompareModel(m_old, m_copy, field), tol)
      << "Original and copied models are different!\n"
      << "Affected file " << xml << '\n'
      << "Different field: " << field << '\n';

  // copy to a new spec, compile and compare
  mjSpec* s_copy2 = mj_copySpec(s);
  mjModel* m_copy2 = mj_compile(s_copy2, nullptr);

  ASSERT_THAT(m_copy2, NotNull())
      << "Failed to compile " << xml << ": " << mjs_getError(s_copy2);

  EXPECT_LE(CompareModel(m_old, m_copy2, field), tol)
      << "Original and re-copied models are different!\n"
      << "Affected file " << xml << '\n'
      << "Different field: " << field << '\n';

  // mj_setConst leaves a compiled model as it is
  mjData* d = mj_makeData(m_copy2);
  mj_setConst(m_copy2, d);
  EXPECT_THAT(ModelDifferences(m_old, m_copy2, 0), IsEmpty()) << xml;
  mj_deleteData(d);

  // a copy of a compiled spec holds what was authored in the original, and is
  // saved as the original is, also once the original is deleted
  std::string saved = SaveAndReadXml(s);
  mjSpec* s_copy3 = mj_copySpec(s);
  EXPECT_THAT(CompareSpec(s, s_copy3, kAllDifferences), IsEmpty()) << xml;

  mj_deleteModel(m_new);
  mj_deleteModel(m_copy);
  mj_deleteModel(m_copy2);
  mj_deleteSpec(s_copy);
  mj_deleteSpec(s_copy2);
  mj_deleteSpec(s);
  mj_deleteModel(m_old);

  EXPECT_EQ(SaveAndReadXml(s_copy3), saved)
      << "Original and copied specs are saved differently!\n"
      << "Affected file " << xml << '\n';
  mj_deleteSpec(s_copy3);
}

// A spec which is saved as it was written reads back as a spec which is saved
// the same and compiles to the same model, in both notations.
TEST_P(RecompileCompareTest, SavedAsWritten) {
  std::string xml = GetParam();
  std::array<char, 1000> err;
  mjSpec* s = mj_parseXML(xml.c_str(), 0, err.data(), err.size());
  if (!s) {
    GTEST_SKIP() << "Failed to load " << xml << ": " << err.data();
  }

  // a spec is saved as it was written before it is compiled, unless its
  // keyframes wait for a compilation to be laid out for the model
  s->compiler.savecompiled = 0;
  s->compiler.savecanonical = 0;
  std::string uncompiled = SaveToString(s);

  mjModel* m = mj_compile(s, nullptr);
  if (!m) {
    std::string error_message = mjs_getError(s);
    mj_deleteSpec(s);
    GTEST_SKIP() << "Failed to compile " << xml << ": " << error_message;
  }

  for (bool canonical : {false, true}) {
    s->compiler.savecanonical = canonical;
    std::string saved = SaveToString(s);
    ASSERT_FALSE(saved.empty()) << xml;

    // compiling changes what is saved only by completing keyframes and by
    // restructuring
    if (!canonical && !uncompiled.empty() && !m->nkey &&
        !s->compiler.fusestatic && !s->compiler.discardvisual) {
      EXPECT_EQ(uncompiled, saved) << xml;
    }

    // the saved file is read from where the model is, to find its assets
    mjSpec* r = mj_parseXMLString(saved.c_str(), 0, err.data(), err.size());
    ASSERT_THAT(r, NotNull()) << xml << ": " << err.data() << '\n' << saved;
    mjs_setString(r->modelfiledir, mjs_getString(s->modelfiledir));
    mjModel* m_saved = mj_compile(r, nullptr);
    ASSERT_THAT(m_saved, NotNull()) << xml << ": " << mjs_getError(r);

    std::string field = "";
    EXPECT_EQ(CompareModel(m, m_saved, field), 0)
        << "Model and model saved as written are different!\n"
        << "Affected file " << xml << '\n'
        << "Canonical notation: " << canonical << '\n'
        << "Different field: " << field << '\n';

    r->compiler.savecompiled = 0;
    r->compiler.savecanonical = canonical;
    EXPECT_EQ(SaveToString(r), saved) << xml;

    mj_deleteModel(m_saved);
    mj_deleteSpec(r);
  }

  mj_deleteModel(m);
  mj_deleteSpec(s);
}

// The differences between two models outside their assets: the sizes and
// arrays of meshes, textures, bounding volumes, octrees, names and paths are
// not compared, nor are the asset ids which elements hold. Returns the first
// differing field, or an empty string.
std::string CompareWithoutAssets(const mjModel* m1, const mjModel* m2) {
  auto asset = [](std::string_view name) {
    for (std::string_view prefix :
         {"nmesh",         "ntex",        "nbvh",        "noct",
          "nnames",        "npaths",      "nbuffer",     "mesh_",
          "tex_",          "bvh_",        "oct_",        "name_",
          "names",         "paths",       "geom_dataid", "site_dataid",
          "mat_texid",     "light_texid", "flex_bvhadr", "skin_pathadr",
          "hfield_pathadr"}) {
      if (absl::StartsWith(name, prefix)) return true;
    }
    return false;
  };
  MJMODEL_POINTERS_PREAMBLE(m1);
#define X(name) \
  if (!asset(#name) && m1->name != m2->name) return #name;
  MJMODEL_SIZES
#undef X
#define X(type, name, nr, nc)                                      \
  if (!asset(#name)) {                                             \
    for (int i = 0; i < m1->nr * nc; i++) {                        \
      if (m1->name[i] != m2->name[i]) {                            \
        return std::string(#name) + "[" + std::to_string(i) + "]"; \
      }                                                            \
    }                                                              \
  }
  MJMODEL_POINTERS
#undef X
  return "";
}

// True if the meshes of two models hold the same data.
bool SameMesh(const mjModel* m1, int id1, const mjModel* m2, int id2) {
  if (id1 < 0 || id2 < 0) return id1 == id2;
  if (m1->mesh_vertnum[id1] != m2->mesh_vertnum[id2] ||
      m1->mesh_facenum[id1] != m2->mesh_facenum[id2] ||
      m1->mesh_normalnum[id1] != m2->mesh_normalnum[id2] ||
      m1->mesh_texcoordnum[id1] != m2->mesh_texcoordnum[id2] ||
      (m1->mesh_graphadr[id1] < 0) != (m2->mesh_graphadr[id2] < 0) ||
      m1->mesh_octnum[id1] != m2->mesh_octnum[id2]) {
    return false;
  }
  for (int i = 0; i < 3; i++) {
    if (m1->mesh_pos[3 * id1 + i] != m2->mesh_pos[3 * id2 + i]) return false;
    if (m1->mesh_scale[3 * id1 + i] != m2->mesh_scale[3 * id2 + i])
      return false;
  }
  for (int i = 0; i < 4; i++) {
    if (m1->mesh_quat[4 * id1 + i] != m2->mesh_quat[4 * id2 + i]) return false;
  }
  int nvert = 3 * m1->mesh_vertnum[id1];
  int nface = 3 * m1->mesh_facenum[id1];
  const float* vert1 = m1->mesh_vert + 3 * m1->mesh_vertadr[id1];
  const float* vert2 = m2->mesh_vert + 3 * m2->mesh_vertadr[id2];
  const int* face1 = m1->mesh_face + 3 * m1->mesh_faceadr[id1];
  const int* face2 = m2->mesh_face + 3 * m2->mesh_faceadr[id2];
  return std::equal(vert1, vert1 + nvert, vert2) &&
         std::equal(face1, face1 + nface, face2);
}

// True if the textures of two models hold the same data.
bool SameTexture(const mjModel* m1, int id1, const mjModel* m2, int id2) {
  if (id1 < 0 || id2 < 0) return id1 == id2;
  if (m1->tex_type[id1] != m2->tex_type[id2] ||
      m1->tex_colorspace[id1] != m2->tex_colorspace[id2] ||
      m1->tex_width[id1] != m2->tex_width[id2] ||
      m1->tex_height[id1] != m2->tex_height[id2] ||
      m1->tex_nchannel[id1] != m2->tex_nchannel[id2]) {
    return false;
  }
  int size = m1->tex_width[id1] * m1->tex_height[id1] * m1->tex_nchannel[id1];
  const mjtByte* data1 = m1->tex_data + m1->tex_adr[id1];
  const mjtByte* data2 = m2->tex_data + m2->tex_adr[id2];
  return std::equal(data1, data1 + size, data2);
}

// Merging the duplicate assets of a compiled spec and compiling again gives a
// model with the same physics: every field outside the assets is the same,
// and every geom, site, material and light refers to an asset with the same
// data. Merging again changes nothing.
TEST_P(RecompileCompareTest, DedupAssets) {
  std::string xml = GetParam();
  std::array<char, 1000> err;
  mjSpec* s = mj_parseXML(xml.c_str(), 0, err.data(), err.size());
  if (!s) {
    GTEST_SKIP() << "Failed to load " << xml << ": " << err.data();
  }
  mjModel* m = mj_compile(s, nullptr);
  if (!m) {
    std::string error_message = mjs_getError(s);
    mj_deleteSpec(s);
    GTEST_SKIP() << "Failed to compile " << xml << ": " << error_message;
  }

  int nmerged = mjs_dedupAssets(s);
  ASSERT_GE(nmerged, 0) << xml << ": " << mjs_getError(s);
  mjModel* m_merged = mj_compile(s, nullptr);
  ASSERT_THAT(m_merged, NotNull()) << xml << ": " << mjs_getError(s);
  EXPECT_EQ(m->ntex - m_merged->ntex + m->nmesh - m_merged->nmesh, nmerged)
      << xml;

  EXPECT_THAT(CompareWithoutAssets(m, m_merged), IsEmpty()) << xml;
  for (int i = 0; i < m->ngeom; i++) {
    if (m->geom_type[i] == mjGEOM_MESH || m->geom_type[i] == mjGEOM_SDF) {
      EXPECT_TRUE(
          SameMesh(m, m->geom_dataid[i], m_merged, m_merged->geom_dataid[i]))
          << xml << ": geom " << i;
    } else {
      EXPECT_EQ(m->geom_dataid[i], m_merged->geom_dataid[i])
          << xml << ": geom " << i;
    }
  }
  for (int i = 0; i < m->nsite; i++) {
    EXPECT_TRUE(
        SameMesh(m, m->site_dataid[i], m_merged, m_merged->site_dataid[i]))
        << xml << ": site " << i;
  }
  for (int i = 0; i < m->nmat * mjNTEXROLE; i++) {
    EXPECT_TRUE(
        SameTexture(m, m->mat_texid[i], m_merged, m_merged->mat_texid[i]))
        << xml << ": material " << i / mjNTEXROLE << " role " << i % mjNTEXROLE;
  }
  for (int i = 0; i < m->nlight; i++) {
    EXPECT_TRUE(
        SameTexture(m, m->light_texid[i], m_merged, m_merged->light_texid[i]))
        << xml << ": light " << i;
  }

  // applying it again changes nothing
  mjSpec* once = mj_copySpec(s);
  EXPECT_EQ(mjs_dedupAssets(s), 0) << xml << ": " << mjs_getError(s);
  EXPECT_THAT(CompareSpec(once, s), IsEmpty()) << xml;

  mj_deleteSpec(once);
  mj_deleteModel(m_merged);
  mj_deleteModel(m);
  mj_deleteSpec(s);
}

// A model which is edited and passed to mj_setConst is the model which the
// spec compiles to once the edit is copied back to it, apart from what
// compilation derives once: the values which follow from the conventions of
// the spec, the statistics, and the actuator length ranges, which
// mj_setLengthRange computes.
TEST_F(MujocoTest, SetConstMatchesRecompile) {
  static constexpr char kModelXml[] = R"(
  <mujoco>
    <option>
      <flag gravity="enable"/>
    </option>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1" mode="trackcom"/>
      <camera pos="0 -2 1" mode="targetbody" target="b1"/>
      <body name="b1" pos="0 0 1" gravcomp="0.5">
        <joint name="j1" type="hinge" axis="0 1 0" springref="0.1"
               springdamper="0.2 0.8"/>
        <joint name="j2" type="slide" axis="1 0 0" stiffness="10" damping="1"/>
        <geom name="g1" type="capsule" size="0.05 0.2" pos="0.1 0 0"
              mass="1.5" margin="0.01" fluidshape="ellipsoid"/>
        <site name="s1" pos="0.2 0 0"/>
        <body name="b2" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 0 1"/>
          <geom name="g2" type="box" size="0.06 0.05 0.04" pos="0.05 0 0"
                mass="0.8"/>
          <site name="s2" pos="0.1 0 0"/>
          <camera pos="0 0 0.5" mode="track"/>
        </body>
      </body>
      <body name="b3" pos="0.5 0 1">
        <freejoint/>
        <geom name="g3" type="sphere" size="0.08" mass="1.0"/>
        <site name="s3" pos="0 0 0.08"/>
      </body>
    </worldbody>
    <equality>
      <connect body1="b2" body2="b3" anchor="0.1 0 0"/>
      <weld body1="b1" body2="b3"/>
      <weld body1="b1" body2="b3" relpose="0.5 0 0 1 0 0 0"/>
    </equality>
    <tendon>
      <spatial name="t_spatial" stiffness="20" damping="2">
        <site site="s1"/>
        <site site="s2"/>
        <site site="s3"/>
      </spatial>
      <fixed name="t_fixed" stiffness="15" damping="1" springlength="0.1 0.2">
        <joint joint="j1" coef="1"/>
        <joint joint="j3" coef="-0.5"/>
      </fixed>
    </tendon>
    <actuator>
      <position joint="j2" kp="50" dampratio="1"/>
      <position joint="j3" kp="30" kv="3"/>
      <motor tendon="t_spatial" gear="2"/>
    </actuator>
    <keyframe>
      <key name="k0"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> err;
  mjSpec* s = mj_parseXMLString(kModelXml, nullptr, err.data(), err.size());
  ASSERT_THAT(s, NotNull()) << err.data();

  // inertias inferred from geoms are adopted, so that geometric edits leave
  // them as they are
  for (mjsElement* e = mjs_firstElement(s, mjOBJ_BODY); e;
       e = mjs_nextElement(s, e)) {
    mjsBody* body = mjs_asBody(e);
    if (mjs_getId(e) != 0) {
      ASSERT_EQ(mjs_adoptInertial(body, nullptr), 0);
    }
  }

  mjModel* m = mj_compile(s, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(s);

  ModelValues compiled = CompiledValues(s, m);
  for (const ModelEdit& edit : kModelEdits) {
    mjModel* m_edit = mj_copyModel(nullptr, m);
    edit.apply(m_edit);
    mjSpec* s_edit = mj_copySpec(s);
    ASSERT_EQ(mj_copyBack(s_edit, m_edit), 1)
        << edit.name << ": " << mjs_getError(s_edit);
    mjModel* m_recompiled = mj_compile(s_edit, nullptr);
    ASSERT_THAT(m_recompiled, NotNull())
        << edit.name << ": " << mjs_getError(s_edit);

    // the same edit, passed to mj_setConst; bounding volume hierarchies whose
    // structure compilation changes are refit by mj_setConst
    mjData* d = mj_makeData(m_edit);
    mj_setConst(m_edit, d);
    std::vector<std::string> ignored = {"actuator_lengthrange", "stat."};
    if (*edit.recompiled) ignored.push_back(edit.recompiled);
    if (std::memcmp(m_edit->bvh_nodeid, m_recompiled->bvh_nodeid,
                    sizeof(int) * m_edit->nbvh) ||
        std::memcmp(m_edit->bvh_child, m_recompiled->bvh_child,
                    sizeof(int) * 2 * m_edit->nbvh)) {
      ignored.push_back("bvh_");
    }
    EXPECT_THAT(ModelDifferences(m_edit, m_recompiled, MjTol(1e-12, 1e-6),
                                 ignored, compiled),
                IsEmpty())
        << "edit: " << edit.name;
    mj_deleteData(d);
    mj_deleteModel(m_recompiled);
    mj_deleteSpec(s_edit);
    mj_deleteModel(m_edit);
  }

  mj_deleteModel(m);
  mj_deleteSpec(s);
}

INSTANTIATE_TEST_SUITE_P(
    AllModels, RecompileCompareTest,
    ::testing::ValuesIn(GetRecompileTestModels()),
    [](const ::testing::TestParamInfo<std::string>& info) {
      std::string name = std::filesystem::path(info.param).filename().string();
      std::replace_if(
          name.begin(), name.end(), [](char c) { return !std::isalnum(c); },
          '_');
      return name + "_" + std::to_string(info.index);
    });

}  // namespace
}  // namespace mujoco
