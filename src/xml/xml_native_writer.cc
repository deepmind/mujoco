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

#include "xml/xml_native_writer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "engine/engine_io.h"
#include "engine/engine_plugin.h"
#include "engine/engine_support.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "user/user_api.h"
#include "user/user_model.h"
#include "user/user_objects.h"
#include "user/user_util.h"
#include "xml/xml_base.h"
#include "xml/xml_numeric_format.h"
#include "xml/xml_util.h"
#include "tinyxml2.h"

// typed attribute rows, generated from mjcf.schema; shared with the reader
#include "xml/generated/mjcf_read_table.inc"

namespace {

using mujoco::user::VectorToString;
using std::string;
using std::string_view;
using tinyxml2::XMLComment;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLText;

}  // namespace


// custom XML indentation: 2 spaces rather than the default 4
class mj_XMLPrinter : public tinyxml2::XMLPrinter {
  using tinyxml2::XMLPrinter::XMLPrinter;

 public:
  void PrintSpace(int depth) {
    for (int i = 0; i < depth; ++i) { Write("  "); }
  }
};


// save XML file using custom 2-space indentation
static string WriteDoc(XMLDocument& doc, char* error, size_t error_sz) {
  doc.ClearError();
  mj_XMLPrinter stream(nullptr, /*compact=*/false);
  doc.Print(&stream);
  if (doc.ErrorID()) {
    mjCopyError(error, doc.ErrorStr(), error_sz);
    return "";
  }
  string str = string(stream.CStr());

  // top level sections
  std::array<string, 17> sections = {"<actuator",
                                     "<asset",
                                     "<compiler",
                                     "<contact>",
                                     "<custom",
                                     "<default>",
                                     "<deformable",
                                     "<equality",
                                     "<extension",
                                     "<keyframe",
                                     "<option",
                                     "<sensor",
                                     "<size",
                                     "<statistic",
                                     "<tendon",
                                     "<visual",
                                     "<worldbody"};

  // position of newline before first section
  size_t first_pos = string::npos;

  // insert newlines before section headers
  for (const string& section : sections) {
    std::size_t pos = 0;
    while ((pos = str.find(section, pos)) != string::npos) {
      // find newline before this section
      std::size_t line_pos = str.rfind('\n', pos);

      // save position of first section
      if (line_pos < first_pos) first_pos = line_pos;

      // insert another newline
      if (line_pos != string::npos) {
        str.insert(line_pos + 1, 1, '\n');
        pos++;  // account for inserted newline
      }

      // advance
      pos += section.length();
    }
  }

  // remove added newline before the first section
  if (first_pos != string::npos) { str.erase(first_pos, 1); }

  return str;
}


// insert end child with given name, return child
XMLElement* mjXWriter::InsertEnd(XMLElement* parent, const char* name) {
  XMLElement* result = parent->GetDocument()->NewElement(name);
  parent->InsertEndChild(result);

  return result;
}


//---------------------------------- class mjXWriter: what is saved -------------------------------

namespace {

// a string or vector of an element: the one which the spec gives, or its compiled copy
template <typename T>
const T& Pick(bool authored, const T* given, const T& compiled) {
  return authored ? *given : compiled;
}

// a vector as text; its numbers as attributes write them, see Vector2String
template <typename T>
string Numbers(const std::vector<T>& vec) {
  if constexpr (std::is_floating_point_v<T>) {
    string text;
    mjXUtil::Vector2String(text, vec);
    return text;
  } else {
    return VectorToString(vec);
  }
}

// the byte size of the field a numeric row binds; 0 for the other kinds
size_t RowSize(const mjXAttr& row) {
  switch (row.kind) {
    case mjXAttr::kInt:
    case mjXAttr::kEnum:
    case mjXAttr::kFlags:
      return row.len * sizeof(int);
    case mjXAttr::kDouble:
      return row.len * sizeof(double);
    case mjXAttr::kNum:
      return row.len * sizeof(mjtNum);
    case mjXAttr::kFloat:
      return row.len * sizeof(float);
    case mjXAttr::kEnumByte:
    case mjXAttr::kBool:
      return sizeof(mjtByte);
    default:
      return 0;
  }
}

// copy the fields which rows bind from one actuator to another
void CopyRows(
    mjsActuator* dst, const mjsActuator* src, const mjXAttr* rows, int nrow, bool skip_nodefault) {
  for (int i = 0; i < nrow; i++) {
    if (skip_nodefault && rows[i].nodefault) { continue; }
    size_t size = RowSize(rows[i]);
    if (size) { std::memcpy((char*)dst + rows[i].offset, (const char*)src + rows[i].offset, size); }
  }
}

// true if two actuators agree on the fields the general rows bind, actdim aside (compilation
// resolves it from the control model), and on the input signature
bool SameActuator(const mjsActuator& a, const mjsActuator& b) {
  const char* pa = (const char*)&a;
  const char* pb = (const char*)&b;
  for (int i = 0; i < kGeneralAttrsN; i++) {
    const mjXAttr& row = kGeneralAttrs[i];
    if (row.offset == (int)offsetof(mjsActuator, actdim)) { continue; }
    const char* x = pa + row.offset;
    const char* y = pb + row.offset;
    switch (row.kind) {
      case mjXAttr::kDouble:
        if (!std::equal((const double*)x, (const double*)x + row.len, (const double*)y)) {
          return false;
        }
        break;
      case mjXAttr::kInt:
      case mjXAttr::kEnum:
      case mjXAttr::kFlags:
        if (!std::equal((const int*)x, (const int*)x + row.len, (const int*)y)) { return false; }
        break;
      case mjXAttr::kEnumByte:
      case mjXAttr::kBool:
        if (*(const mjtByte*)x != *(const mjtByte*)y) { return false; }
        break;
      default:
        break;
    }
  }
  return a.ctrlspec == b.ctrlspec;
}

// the input attribute of a signature: so3 chart keyword, the none keyword, or input tokens
string InputString(mjtGain gaintype, int ctrlspec) {
  if (gaintype == mjGAIN_SO3) {
    return mjXUtil::FindValue(inputchart_map, inputchart_sz, ctrlspec);
  }
  if (ctrlspec == mjINPUT_NONE) { return "none"; }
  string input;
  for (int k = 0; k < inputbit_sz; k++) {
    if (ctrlspec & inputbit_map[k].value) {
      input += string(input.empty() ? "" : " ") + inputbit_map[k].key;
    }
  }
  return input;
}

}  // namespace


// an angle of an element in the unit of the saved file: the notation may be canonical, or the
// element may be from an attached spec which has another unit. Degrees become radians as
// compilation converts them, which is not the same for a joint and for an orientation
double mjXWriter::Angle(const mjCBase* element, double angle, bool orientation) const {
  const mjsCompiler* compiler = element->compiler ? element->compiler : &model->spec.compiler;
  const bool         degree   = compiler->degree;
  if (degree == degree_) { return angle; }
  if (degree) { return orientation ? angle / 180.0 * mjPI : angle * (mjPI / 180.0); }
  return orientation ? angle / mjPI * 180.0 : angle / (mjPI / 180.0);
}


// write the position, if given, and the orientation which the spec gives an element, where they
// differ from those of its class
void mjXWriter::WriteSpecPose(XMLElement*           elem,
                              const mjCBase*        element,
                              const double          pos[3],
                              const double          quat[4],
                              const mjsOrientation& alt,
                              const double*         defpos,
                              const double*         defquat,
                              const mjsOrientation* defalt) {
  const double       unitq[4] = {1, 0, 0, 0};
  const mjsCompiler* compiler = element->compiler ? element->compiler : &model->spec.compiler;
  if (pos) { WriteAttr(elem, "pos", 3, pos, defpos ? defpos : unitq + 1); }

  // a quaternion: as written, or in place of an alternative if the notation is canonical or if
  // Euler angles have another sequence than that of the saved file
  bool sequence = !std::strncmp(compiler->eulerseq, model->spec.compiler.eulerseq, 3);
  if (alt.type == mjORIENTATION_QUAT ||
      canonical_ ||
      (alt.type == mjORIENTATION_EULER && !sequence)) {
    double resolved[4];
    SpecQuat(resolved, element, quat, alt);

    // the quaternion of the class; if the class is written with an alternative, the quaternion
    // replaces it and so is always written
    double defresolved[4] = {1, 0, 0, 0};
    bool   defquaternion  = !defalt || defalt->type == mjORIENTATION_QUAT || canonical_;
    if (defquat && defalt && defquaternion) { SpecQuat(defresolved, nullptr, defquat, *defalt); }
    WriteAttr(elem, "quat", 4, resolved, defquaternion ? defresolved : nullptr);
    return;
  }

  // an alternative, as written, unless the class has the same; angles in the unit of the saved
  // file
  const bool same = defalt && defalt->type == alt.type;
  switch (alt.type) {
    case mjORIENTATION_AXISANGLE: {
      double axisangle[4] = {alt.axisangle[0],
                             alt.axisangle[1],
                             alt.axisangle[2],
                             Angle(element, alt.axisangle[3], /*orientation=*/true)};
      WriteAttr(elem, "axisangle", 4, axisangle, same ? defalt->axisangle : nullptr);
    } break;

    case mjORIENTATION_XYAXES:
      WriteAttr(elem, "xyaxes", 6, alt.xyaxes, same ? defalt->xyaxes : nullptr);
      break;

    case mjORIENTATION_ZAXIS:
      WriteAttr(elem, "zaxis", 3, alt.zaxis, same ? defalt->zaxis : nullptr);
      break;

    case mjORIENTATION_EULER: {
      double euler[3];
      for (int i = 0; i < 3; i++) { euler[i] = Angle(element, alt.euler[i], /*orientation=*/true); }
      WriteAttr(elem, "euler", 3, euler, same ? defalt->euler : nullptr);
    } break;

    default:
      break;
  }
}


// the quaternion which the spec gives an element: as it is, or in the canonical notation the one
// which compilation makes of it and of its alternative. A null element is one of a default class,
// whose values are read as those of the saved spec
void mjXWriter::SpecQuat(double                result[4],
                         const mjCBase*        element,
                         const double          quat[4],
                         const mjsOrientation& alt) {
  const mjsCompiler* compiler = &model->spec.compiler;
  if (element && element->compiler) { compiler = element->compiler; }
  mjuu_copyvec(result, quat, 4);
  if (alt.type == mjORIENTATION_QUAT && !canonical_) { return; }
  mjuu_normvec(result, 4);
  const char* err = ResolveOrientation(result, compiler->degree, compiler->eulerseq, alt);
  if (err) {
    string message = "orientation of '" +
                     (element ? element->name : string("default")) +
                     "' cannot be saved: " +
                     err;
    throw mjXError(0, "%s", message.c_str());
  }
}


// write the size and the pose which the spec gives a geom or a site, where they differ from those
// of its class; in the canonical notation, fromto as the size and pose which it stands for
template <typename S>
void mjXWriter::WriteSpecShape(XMLElement*    elem,
                               const mjCBase* element,
                               const S&       given,
                               const S&       defgiven) {
  if (!canonical_) {
    WriteAttr(elem, "size", 3, given.size, defgiven.size, /*trim=*/true);
    if (mjuu_defined(given.fromto[0])) {
      const bool defspan = mjuu_defined(defgiven.fromto[0]);
      WriteAttr(elem, "fromto", 6, given.fromto, defspan ? defgiven.fromto : nullptr);
    } else {
      WriteSpecPose(elem,
                    element,
                    given.pos,
                    given.quat,
                    given.alt,
                    defgiven.pos,
                    defgiven.quat,
                    &defgiven.alt);
    }
    return;
  }

  // the size and pose which a spec struct stands for, as compilation finds them
  auto canonical =
      [this](const mjCBase* owner, const S& shape, double size[3], double pos[3], double quat[4]) {
        mjuu_copyvec(size, shape.size, 3);
        if (!mjuu_defined(shape.fromto[0])) {
          mjuu_copyvec(pos, shape.pos, 3);
          SpecQuat(quat, owner, shape.quat, shape.alt);
          return;
        }
        double vec[3] = {shape.fromto[0] - shape.fromto[3],
                         shape.fromto[1] - shape.fromto[4],
                         shape.fromto[2] - shape.fromto[5]};
        size[1]       = mjuu_normvec(vec, 3) / 2;
        if (shape.type == mjGEOM_ELLIPSOID || shape.type == mjGEOM_BOX) {
          size[2] = size[1];
          size[1] = size[0];
        }
        for (int i = 0; i < 3; i++) { pos[i] = (shape.fromto[i] + shape.fromto[i + 3]) / 2; }
        mjuu_z2quat(quat, vec);
      };
  double size[3], pos[3], quat[4], defsize[3], defpos[3], defquat[4];
  canonical(element, given, size, pos, quat);
  canonical(nullptr, defgiven, defsize, defpos, defquat);
  WriteAttr(elem, "size", 3, size, defsize, /*trim=*/true);
  WriteAttr(elem, "pos", 3, pos, defpos);
  WriteAttr(elem, "quat", 4, quat, defquat);
}


// write the user data which the spec gives an element, unless it is that of its default class
void mjXWriter::WriteSpecUser(XMLElement*                elem,
                              const std::vector<double>& user,
                              const std::vector<double>& defuser) {
  if (!user.empty() && user != defuser) { WriteAttr(elem, "user", user.size(), user.data()); }
}


// write the inertial which the spec gives a body, or with saveinertial the one calculated for it
void mjXWriter::WriteSpecInertial(XMLElement* elem, const mjCBody* body) {
  const mjsBody& given    = body->spec;
  const double   unitq[4] = {1, 0, 0, 0};

  if (given.explicitinertial || mjuu_defined(given.ipos[0])) {
    XMLElement* inertial = InsertEnd(elem, "inertial");
    WriteAttr(inertial, "pos", 3, given.ipos);
    if (!mjuu_defined(given.fullinertia[0])) {
      WriteSpecPose(inertial, body, nullptr, given.iquat, given.ialt);
      WriteAttr(inertial, "mass", 1, &given.mass);
      WriteAttr(inertial, "diaginertia", 3, given.inertia, unitq + 1);
    } else {
      // A full inertia, rotated out of the frame in which the spec gives it as compilation does:
      // MJCF has no orientation beside a full inertia, while a spec which is read from URDF
      // does. In the canonical notation, its principal axes and moments.
      const double* in        = given.fullinertia;
      const double  given9[9] = {in[0], in[3], in[4], in[3], in[1], in[5], in[4], in[5], in[2]};
      double        quat[4], diag[3], mat[9], out[9];
      mjuu_copyvec(quat, given.iquat, 4);
      mjuu_normvec(quat, 4);
      mjuu_quat2mat(mat, quat);
      mjuu_mulRMRT(out, mat, given9);
      const double rotated[6] = {out[0], out[4], out[8], out[1], out[2], out[5]};
      if (!canonical_) {
        WriteAttr(inertial, "mass", 1, &given.mass);
        WriteAttr(inertial, "fullinertia", 6, rotated);
      } else {
        const char*        err      = mjuu_fullInertia(quat, diag, rotated);
        const mjsCompiler* compiler = body->compiler ? body->compiler : &model->spec.compiler;
        if (err) {
          if (compiler->boundinertia > 0 && diag[2] >= -mjEPS * std::max(1.0, diag[0])) {
            diag[0] = std::max(diag[0], compiler->boundinertia);
            diag[1] = std::max(diag[1], compiler->boundinertia);
            diag[2] = std::max(diag[2], compiler->boundinertia);
          } else {
            string message = "fullinertia of '" + body->name + "' cannot be saved: " + err;
            throw mjXError(0, "%s", message.c_str());
          }
        }
        WriteAttr(inertial, "quat", 4, quat, unitq);
        WriteAttr(inertial, "mass", 1, &given.mass);
        WriteAttr(inertial, "diaginertia", 3, diag);
      }
    }
  }

  // the calculated inertial is in the frame of the body as the spec gives it, which is where it
  // was before compilation aligned the body with a free joint, and with the mass before
  // settotalmass scaled it, as the inertials which the spec gives: the saved file scales them all.
  // It is in the units of the spec, which compilation scales by the frame of the body
  else if (model->spec.compiler.saveinertial && !body->iframe) {
    const mjCGauge g = body->ContentGauge().Inverse();
    double         pos[3], inertia[3], mass = body->mass_compiled_, full[6] = {mjNAN};
    mjuu_copyvec(pos, body->ipos_compiled_, 3);
    mjuu_copyvec(inertia, body->inertia_compiled_, 3);
    g.Point(pos);
    g.Inertia(mass, inertia, full, body->iquat_compiled_);
    XMLElement* inertial = InsertEnd(elem, "inertial");
    WriteAttr(inertial, "pos", 3, pos);
    WriteAttr(inertial, "quat", 4, body->iquat_compiled_, unitq);
    WriteAttr(inertial, "mass", 1, &mass);
    if (mjuu_defined(full[0])) {
      WriteAttr(inertial, "fullinertia", 6, full);
    } else {
      WriteAttr(inertial, "diaginertia", 3, inertia);
    }
  }
}


// The saved file has the compiler settings of the model: a body without an inertial element infers
// its inertia from its geoms, or is massless if inertiafromgeom is "false". The last compilation
// gave the body its inertia by the spec as it was then, and with the settings of its own spec,
// which differ for a body of an attached spec which has others
bool mjXWriter::InertialReproduced(const mjCBody* body) const {
  const mjsCompiler& saved = model->compiler;
  if (body->inertia_inferred_) {
    return saved.inertiafromgeom != mjINERTIAFROMGEOM_FALSE &&
           body->inertia_groups_[0] == saved.inertiagrouprange[0] &&
           body->inertia_groups_[1] == saved.inertiagrouprange[1];
  }

  // the inertial which the spec gave, or none
  return !body->inertia_given_ && saved.inertiafromgeom == mjINERTIAFROMGEOM_FALSE;
}


// The elements of an attached spec are compiled with the compiler settings of that spec, and the
// saved file has one compiler element: the angle unit and the Euler sequence are converted, the
// settings which change what compilation infers cannot be. Whether a range limits is inferred
// alike under either autolimits, which only makes "auto" with a range an error if it is false
string mjXWriter::AttachedSettings() const {
  const mjsCompiler& saved = model->spec.compiler;
  auto               own   = [&](const mjCBase* element) {
    return element->compiler && element->compiler != &saved ? element->compiler : nullptr;
  };
  auto refuse = [](const mjCBase* element, const char* setting) {
    return "XML Write error: '" +
           element->name +
           "' is compiled with the setting " +
           setting +
           " of an attached model, which differs from that of this model: the saved file has one "
           "compiler element, and cannot give it both";
  };

  // bodies: what compilation infers of their inertia, and whether a free joint aligns them
  std::vector<const mjCBody*> bodies(model->GetWorld()->bodies.begin(),
                                     model->GetWorld()->bodies.end());
  while (!bodies.empty()) {
    const mjCBody* body = bodies.back();
    bodies.pop_back();
    bodies.insert(bodies.end(), body->bodies.begin(), body->bodies.end());
    if (const mjsCompiler* settings = own(body)) {
      if (settings->inertiafromgeom != saved.inertiafromgeom) {
        return refuse(body, "inertiafromgeom");
      }
      if (settings->inertiagrouprange[0] != saved.inertiagrouprange[0] ||
          settings->inertiagrouprange[1] != saved.inertiagrouprange[1]) {
        return refuse(body, "inertiagrouprange");
      }
      if (settings->boundmass != saved.boundmass) { return refuse(body, "boundmass"); }
      if (settings->boundinertia != saved.boundinertia) { return refuse(body, "boundinertia"); }
      if (settings->balanceinertia != saved.balanceinertia) {
        return refuse(body, "balanceinertia");
      }
      if (settings->alignfree != saved.alignfree) { return refuse(body, "alignfree"); }
    }
    for (const mjCJoint* joint : body->joints) {
      const mjsCompiler* settings = own(joint);
      if (settings && settings->autolimits && !saved.autolimits) {
        return refuse(joint, "autolimits");
      }
    }
  }

  // tendons and actuators: whether their ranges limit
  for (const mjCTendon* tendon : model->Tendons()) {
    const mjsCompiler* settings = own(tendon);
    if (settings && settings->autolimits && !saved.autolimits) {
      return refuse(tendon, "autolimits");
    }
  }
  for (const mjCActuator* actuator : model->Actuators()) {
    const mjsCompiler* settings = own(actuator);
    if (settings && settings->autolimits && !saved.autolimits) {
      return refuse(actuator, "autolimits");
    }
  }
  return "";
}


//---------------------------------- class mjXWriter: one-element writers --------------------------

// write flex
void mjXWriter::OneFlex(XMLElement* elem, const mjCFlex* flex) {
  string         text;
  mjCFlex        defflex;
  const mjsFlex* values = Values<mjsFlex>(flex);

  if (values->elastic3d) {
    throw mjXError(0,
                   "Stable Neo-Hookean elasticity is mjSpec-only and cannot be "
                   "written to MJCF");
  }

  // common attributes
  WriteAttrTxt(elem, "name", flex->name);
  WriteAttrTable(elem, values, static_cast<const mjsFlex*>(&defflex), kFlexAttrs, kFlexAttrsN);
  const auto& material = Pick(authored_, flex->spec.material, flex->get_material());
  if (material != defflex.get_material() && !material.empty()) {
    WriteAttrTxt(elem, "material", Numbers(material));
  }
  WriteAttr(elem, "cellcount", 3, flex->spec.cellcount, defflex.spec.cellcount);
  if (flex->spec.order != defflex.spec.order) {
    string dof_str = "full";
    if (flex->spec.order == 1)
      dof_str = "trilinear";
    else if (flex->spec.order == 2)
      dof_str = "quadratic";
    WriteAttrTxt(elem, "dof", dof_str);
  }

  // data vectors
  const auto& vertbody     = Pick(authored_, flex->spec.vertbody, flex->get_vertbody());
  const auto& vert         = Pick(authored_, flex->spec.vert, flex->get_vert());
  const auto& element      = Pick(authored_, flex->spec.elem, flex->get_elem());
  const auto& texcoord     = Pick(authored_, flex->spec.texcoord, flex->get_texcoord());
  const auto& elemtexcoord = Pick(authored_, flex->spec.elemtexcoord, flex->get_elemtexcoord());
  const auto& nodebody     = Pick(authored_, flex->spec.nodebody, flex->get_nodebody());
  const auto& node         = Pick(authored_, flex->spec.node, flex->get_node());
  if (!vertbody.empty()) {
    text = Numbers(vertbody);
    WriteAttrTxt(elem, "body", text);
  }
  if (!vert.empty()) {
    text = Numbers(vert);
    WriteAttrTxt(elem, "vertex", text);
  }
  if (!element.empty()) {
    text = Numbers(element);
    WriteAttrTxt(elem, "element", text);
  }
  if (!texcoord.empty()) {
    text = Numbers(texcoord);
    WriteAttrTxt(elem, "texcoord", text);
  }
  if (!elemtexcoord.empty()) {
    text = Numbers(elemtexcoord);
    WriteAttrTxt(elem, "elemtexcoord", text);
  }
  if (!nodebody.empty()) {
    text = Numbers(nodebody);
    WriteAttrTxt(elem, "node", text);
  }
  if (!node.empty()) { WriteVector(elem, "nodecoord", node); }

  // contact subelement
  XMLElement* cont = InsertEnd(elem, "contact");
  WriteAttrTable(cont,
                 values,
                 static_cast<const mjsFlex*>(&defflex),
                 kFlexcomp_contactAttrs,
                 kFlexcomp_contactAttrsN);

  // remove contact is no attributes
  if (!cont->FirstAttribute()) { elem->DeleteChild(cont); }

  // elasticity subelement
  XMLElement* elastic = InsertEnd(elem, "elasticity");
  WriteAttrTable(elastic,
                 values,
                 static_cast<const mjsFlex*>(&defflex),
                 kElasticityAttrs,
                 kElasticityAttrsN);

  // yield strains: a single value if it gives the bending yield strain, which is the stretching
  // yield strain for non-interpolated 2D flexes and zero otherwise
  const double* yield  = values->stressyield;
  bool          shell  = values->dim == 2 && nodebody.empty();
  bool          single = yield[1] == (shell ? yield[0] : 0);
  WriteAttr(elastic, "yield", single ? 1 : 2, yield, defflex.spec.stressyield);

  // edge subelement
  XMLElement* edge = InsertEnd(elem, "edge");
  WriteAttrTable(edge,
                 values,
                 static_cast<const mjsFlex*>(&defflex),
                 kFlex_edgeAttrs,
                 kFlex_edgeAttrsN);

  // remove edge if no attributes
  if (!edge->FirstAttribute()) { elem->DeleteChild(edge); }
}


// write mesh
void mjXWriter::OneMesh(XMLElement* elem, const mjCMesh* mesh, mjCDef* def) {
  string         text;
  const mjsMesh* values = Values<mjsMesh>(mesh);

  // an instance whose map does not multiply the scale is saved compiled with its vertices in the
  // coordinates of its geoms, which the frame of the compiled mesh gives
  mjsMesh baked;
  if (!writingdefaults && !authored_ && !mesh->InstanceDiagonal() && !mesh->Plugin().active) {
    baked = *values;
    mjuu_zerovec(baked.refpos, 3);
    mjuu_setvec(baked.refquat, 1, 0, 0, 0);
    mjuu_setvec(baked.scale, 1, 1, 1);
    values = &baked;
    std::vector<float> vert(mesh->Vert().size());
    for (int i = 0; i < mesh->nvert(); i++) {
      double v[3] = {mesh->Vert(3 * i), mesh->Vert(3 * i + 1), mesh->Vert(3 * i + 2)};
      mjuu_rotVecQuat(v, v, mesh->Quat());
      for (int k = 0; k < 3; k++) { vert[3 * i + k] = (float)(v[k] + mesh->Pos()[k]); }
    }
    WriteAttrTxt(elem, "name", mesh->name);
    if (mesh->classname != "main") { WriteAttrTxt(elem, "class", mesh->classname); }
    WriteAttrTxt(elem, "vertex", Numbers(vert));
    if (mesh->ntexcoord() == mesh->nvert() && mesh->FaceTexcoord() == mesh->Face()) {
      WriteAttrTxt(elem, "texcoord", Numbers(mesh->Texcoord()));
    }
    WriteAttrTxt(elem, "face", Numbers(mesh->Face()));
  }

  // regular
  else if (!writingdefaults) {
    WriteAttrTxt(elem, "name", mesh->name);
    if (mesh->classname != "main") { WriteAttrTxt(elem, "class", mesh->classname); }
    WriteAttrTxt(elem,
                 "content_type",
                 Pick(authored_, mesh->spec.content_type, mesh->ContentType()));
    WriteAttrTxt(elem, "file", Pick(authored_, mesh->spec.file, mesh->File()));

    // write vertex data
    if (!mesh->UserVert().empty()) {
      text = Numbers(mesh->UserVert());
      WriteAttrTxt(elem, "vertex", text);
    }

    // write normal data
    if (!mesh->UserNormal().empty()) {
      text = Numbers(mesh->UserNormal());
      WriteAttrTxt(elem, "normal", text);
    }

    // write texcoord data
    if (!mesh->UserTexcoord().empty()) {
      text = Numbers(mesh->UserTexcoord());
      WriteAttrTxt(elem, "texcoord", text);
    }

    // write face data
    if (!mesh->UserFace().empty()) {
      text = Numbers(mesh->UserFace());
      WriteAttrTxt(elem, "face", text);
    }
  }

  // defaults and regular
  WriteAttrTable(elem, values, &def->Mesh().spec, kMeshAttrs, kMeshAttrsN);
  WriteAttrInt(elem, "maxhullvert", values->maxhullvert, def->Mesh().spec.maxhullvert);
  const string& material = Pick(authored_, mesh->spec.material, mesh->Material());
  if (material != Pick(authored_, def->Mesh().spec.material, def->Mesh().Material())) {
    WriteAttrTxt(elem, "material", material);
  }
}


// write skin
void mjXWriter::OneSkin(XMLElement* elem, const mjCSkin* skin) {
  string         text;
  mjCSkin        defskin;
  const mjsSkin* values = Values<mjsSkin>(skin);
  const bool     scaled = !authored_ && skin->scaled_;
  const string   file   = scaled ? "" : Pick(authored_, skin->spec.file, skin->File());

  // bones loaded from file and namespaced by an attachment can only be saved as data
  const bool namespaced = !file.empty() && skin->HasBoneNameSpace();
  const bool authored   = authored_ && !namespaced;
  if (namespaced && skin->get_bodyname().empty()) {
    throw mjXError(0,
                   "attached skin '%s' with bones from file must be compiled to be saved",
                   skin->name.c_str());
  }

  // write attributes
  WriteAttrTxt(elem, "name", skin->name);
  if (!namespaced) { WriteAttrTxt(elem, "file", file); }
  WriteAttrTxt(elem, "material", Pick(authored_, skin->spec.material, skin->get_material()));
  WriteAttrInt(elem, "group", values->group, 0);
  WriteAttrTable(elem, values, static_cast<const mjsSkin*>(&defskin), kSkinAttrs, kSkinAttrsN);

  // write data if no file
  if (file.empty() || namespaced) {
    const auto& vert       = Pick(authored, skin->spec.vert, skin->get_vert());
    const auto& texcoord   = Pick(authored, skin->spec.texcoord, skin->get_texcoord());
    const auto& face       = Pick(authored, skin->spec.face, skin->get_face());
    const auto& bodyname   = Pick(authored, skin->spec.bodyname, skin->get_bodyname());
    const auto& bindpos    = Pick(authored, skin->spec.bindpos, skin->get_bindpos());
    const auto& bindquat   = Pick(authored, skin->spec.bindquat, skin->get_bindquat());
    const auto& vertid     = Pick(authored, skin->spec.vertid, skin->get_vertid());
    const auto& vertweight = Pick(authored, skin->spec.vertweight, skin->get_vertweight());

    // mesh vert
    text = Numbers(vert);
    WriteAttrTxt(elem, "vertex", text);

    // mesh texcoord
    if (!texcoord.empty()) {
      text = Numbers(texcoord);
      WriteAttrTxt(elem, "texcoord", text);
    }

    // mesh face
    text = Numbers(face);
    WriteAttrTxt(elem, "face", text);

    // bones
    for (size_t i = 0; i < bodyname.size(); i++) {
      // make bone
      XMLElement* bone = InsertEnd(elem, "bone");

      // write attributes
      WriteAttrTxt(bone, "body", bodyname[i]);
      WriteAttr(bone, "bindpos", 3, bindpos.data() + 3 * i);
      WriteAttr(bone, "bindquat", 4, bindquat.data() + 4 * i);

      // write vertid
      text = Numbers(vertid[i]);
      WriteAttrTxt(bone, "vertid", text);

      // write vertweight
      text = Numbers(vertweight[i]);
      WriteAttrTxt(bone, "vertweight", text);
    }
  }
}


// write texture
void mjXWriter::OneTexture(XMLElement* elem, const mjCTexture* texture, mjCDef* def) {
  const mjsTexture* values = Values<mjsTexture>(texture);
  const mjsTexture* deftex = &def->Texture().spec;
  int               ncells = std::clamp(values->gridsize[0] * values->gridsize[1], 0, 12);
  bool              gridsize_differs =
      values->gridsize[0] != deftex->gridsize[0] || values->gridsize[1] != deftex->gridsize[1];
  const string gridlayout(values->gridlayout,
                          std::min<size_t>(ncells, strnlen(values->gridlayout, 12)));
  const string defgridlayout =
      gridsize_differs
          ? string(ncells, '.')
          : string(deftex->gridlayout, std::min<size_t>(ncells, strnlen(deftex->gridlayout, 12)));

  // defaults: what differs from the parent class
  if (writingdefaults) {
    WriteAttrTable(elem, values, deftex, kTextureAttrs, kTextureAttrsN);
    if (gridlayout != defgridlayout) { WriteAttrTxt(elem, "gridlayout", gridlayout); }
    return;
  }

  const string file      = Pick(authored_, texture->spec.file, texture->File());
  const auto   cubefiles = Pick(authored_, texture->spec.cubefiles, texture->get_cubefiles());

  // write common attributes
  WriteAttrKey(elem, "type", texture_map, texture_sz, values->type);
  if (!authored_ || values->colorspace != deftex->colorspace) {
    WriteAttrKey(elem, "colorspace", colorspace_map, colorspace_sz, values->colorspace);
  }
  WriteAttrKey(elem, "embed", FalseTrueAuto_map, 3, values->embed, deftex->embed);
  WriteAttrTxt(elem, "name", texture->name);
  if (texture->classname != "main") { WriteAttrTxt(elem, "class", texture->classname); }

  // write builtin
  WriteAttrKey(elem, "builtin", builtin_map, builtin_sz, values->builtin, deftex->builtin);
  if (values->builtin != mjBUILTIN_NONE) {
    WriteAttrKey(elem, "mark", mark_map, mark_sz, values->mark, deftex->mark);
    WriteAttr(elem, "rgb1", 3, values->rgb1, deftex->rgb1);
    WriteAttr(elem, "rgb2", 3, values->rgb2, deftex->rgb2);
    WriteAttr(elem, "markrgb", 3, values->markrgb, deftex->markrgb);
    WriteAttr(elem, "random", 1, &values->random, &deftex->random);
    WriteAttrInt(elem, "width", values->width, deftex->width);
    WriteAttrInt(elem, "height", values->height, deftex->height);
  }

  // write buffer
  else if (cubefiles[0].empty() &&
           cubefiles[1].empty() &&
           cubefiles[2].empty() &&
           cubefiles[3].empty() &&
           cubefiles[4].empty() &&
           cubefiles[5].empty() &&
           file.empty() &&
           values->gridsize[0] == 1 &&
           values->gridsize[1] == 1) {
    throw mjXError(0, "no support for buffer textures.");
  }

  // write textures loaded from files
  else {
    // write single file
    WriteAttrTxt(elem,
                 "content_type",
                 Pick(authored_, texture->spec.content_type, texture->get_content_type()));
    WriteAttrTxt(elem, "file", file);

    // write separate files
    WriteAttrTxt(elem, "fileright", cubefiles[0]);
    WriteAttrTxt(elem, "fileleft", cubefiles[1]);
    WriteAttrTxt(elem, "fileup", cubefiles[2]);
    WriteAttrTxt(elem, "filedown", cubefiles[3]);
    WriteAttrTxt(elem, "filefront", cubefiles[4]);
    WriteAttrTxt(elem, "fileback", cubefiles[5]);
    WriteAttrKey(elem, "hflip", bool_map, 2, values->hflip, deftex->hflip);
    WriteAttrKey(elem, "vflip", bool_map, 2, values->vflip, deftex->vflip);
    WriteAttrInt(elem, "nchannel", values->nchannel, deftex->nchannel);
    WriteAttr(elem, "rgb1", 3, values->rgb1, deftex->rgb1);
    WriteAttr(elem, "rgb2", 3, values->rgb2, deftex->rgb2);

    // write grid
    if (gridsize_differs) {
      double gsize[2] = {(double)values->gridsize[0], (double)values->gridsize[1]};
      WriteAttr(elem, "gridsize", 2, gsize);
    }
    if (gridlayout != defgridlayout) { WriteAttrTxt(elem, "gridlayout", gridlayout); }
  }
}


// write material
void mjXWriter::OneMaterial(XMLElement* elem, const mjCMaterial* material, mjCDef* def) {
  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", material->name);
    if (material->classname != "main") { WriteAttrTxt(elem, "class", material->classname); }
  }

  // defaults and regular
  // check if we have non-rgb textures
  const auto& textures = Pick(authored_, material->spec.textures, material->textures_);
  const auto& deftextures =
      Pick(authored_, def->Material().spec.textures, def->Material().textures_);
  bool has_non_rgb = false;
  for (int i = 1; i < mjNTEXROLE; i++) {
    if (!textures[i].empty() && textures[i] != deftextures[i]) {
      if (i != mjTEXROLE_RGB) { has_non_rgb = true; }
    }
  }

  // if we have non-rgb textures, write them as layers
  if (has_non_rgb) {
    for (int i = 1; i < mjNTEXROLE; i++) {
      if (!textures[i].empty() && textures[i] != deftextures[i]) {
        XMLElement* child_elem = InsertEnd(elem, "layer");
        WriteAttrTxt(child_elem, "texture", textures[i]);
        WriteAttrTxt(child_elem, "role", FindValue(texrole_map, 9, i));
      }
    }
  } else {
    if (textures[mjTEXROLE_RGB] != deftextures[mjTEXROLE_RGB]) {
      WriteAttrTxt(elem, "texture", textures[mjTEXROLE_RGB]);
    }
  }

  WriteAttrTable(elem,
                 Values<mjsMaterial>(material),
                 &def->Material().spec,
                 kMaterialAttrs,
                 kMaterialAttrsN);
}


// write joint
void mjXWriter::OneJoint(XMLElement*     elem,
                         const mjCJoint* joint,
                         mjCDef*         def,
                         string_view     classname) {
  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", joint->name);
    if (classname != joint->classname) { WriteAttrTxt(elem, "class", joint->classname); }
  }

  // what the spec gives: angles in the unit of the saved file, pos and axis as they are
  if (authored_) {
    // compilation takes the range of a rotation as angles only if the range limits it
    auto values = [this](const mjCJoint* owner) {
      mjsJoint           result   = owner->spec;
      const mjsCompiler* settings = owner->compiler ? owner->compiler : &model->spec.compiler;
      const bool         hasrange = result.range[0] != 0 || result.range[1] != 0;
      const bool limits = result.limited == mjLIMITED_TRUE ||
                          (result.limited == mjLIMITED_AUTO && settings->autolimits && hasrange);
      if ((result.type == mjJNT_HINGE || result.type == mjJNT_BALL) && limits) {
        result.range[0] = Angle(owner, result.range[0]);
        result.range[1] = Angle(owner, result.range[1]);
      }
      if (result.type == mjJNT_HINGE || result.type == mjJNT_BALL) {
        result.springyield = Angle(owner, result.springyield);
      }
      if (result.type == mjJNT_HINGE) {
        result.ref       = Angle(owner, result.ref);
        result.springref = Angle(owner, result.springref);
      }
      return result;
    };
    const mjsJoint given    = values(joint);
    const mjsJoint defgiven = values(&def->Joint());
    WriteAttrTable(elem, &given, &defgiven, kJointAttrs, kJointAttrsN);
    WriteAttr(elem, "pos", 3, given.pos, defgiven.pos);
    WriteAttr(elem, "axis", 3, given.axis, defgiven.axis);
    WriteAttr(elem, "springdamper", 2, given.springdamper, defgiven.springdamper);
    WriteAttrKey(elem, "limited", FalseTrueAuto_map, 3, given.limited, defgiven.limited);
    WriteAttrKey(elem,
                 "actuatorfrclimited",
                 FalseTrueAuto_map,
                 3,
                 given.actfrclimited,
                 defgiven.actfrclimited);
    WriteSpecUser(elem, *given.userdata, *defgiven.userdata);
    return;
  }

  // defaults and regular
  WriteAttrTable(elem,
                 static_cast<const mjsJoint*>(joint),
                 &def->Joint().spec,
                 kJointAttrs,
                 kJointAttrsN);
  // pos and axis relative to the joint's frame
  double pos[3], axis[3], iquat[4] = {1, 0, 0, 0};
  mjuu_copyvec(pos, joint->pos, 3);
  mjuu_copyvec(axis, joint->axis, 3);
  FrameLocal(joint->frame, pos, iquat);
  mjuu_rotVecQuat(axis, axis, iquat);
  if (joint->type != mjJNT_FREE) { WriteAttr(elem, "pos", 3, pos, def->Joint().spec.pos); }
  if (joint->type != mjJNT_FREE && joint->type != mjJNT_BALL) {
    WriteAttr(elem, "axis", 3, axis, def->Joint().spec.axis);
  }
  if (joint->type != mjJNT_FREE) {
    WriteAttrKey(elem, "limited", FalseTrueAuto_map, 3, joint->limited, def->Joint().limited);
  }
  if (joint->type != mjJNT_FREE && joint->type != mjJNT_BALL) {
    WriteAttrKey(elem,
                 "actuatorfrclimited",
                 FalseTrueAuto_map,
                 3,
                 joint->actfrclimited,
                 def->Joint().actfrclimited);
  }

  // userdata
  if (writingdefaults) {
    WriteVector(elem, "user", joint->get_userdata());
  } else {
    WriteVector(elem, "user", joint->get_userdata(), def->Joint().get_userdata());
  }
}

// write geom
void mjXWriter::OneGeom(XMLElement* elem, const mjCGeom* geom, mjCDef* def, string_view classname) {
  double unitq[4] = {1, 0, 0, 0};
  double mass     = 0;

  // what the spec gives: the size and pose as fromto if they were written so, and the mesh
  // which a geom is fitted to in place of the size which that gives it
  if (authored_) {
    const mjsGeom& given    = geom->spec;
    const mjsGeom& defgiven = def->Geom().spec;
    if (!writingdefaults) {
      WriteAttrTxt(elem, "name", geom->name);
      if (classname != geom->classname) { WriteAttrTxt(elem, "class", geom->classname); }
    }
    WriteSpecShape(elem, geom, given, defgiven);

    WriteAttrTable(elem, &given, &defgiven, kGeomAttrs, kGeomAttrsN);
    WriteAttrKey(elem,
                 "fluidshape",
                 fluidshape_map,
                 2,
                 given.fluid_ellipsoid,
                 defgiven.fluid_ellipsoid);
    if (given.type != mjGEOM_MESH) {
      WriteAttrKey(elem, "shellinertia", bool_map, 2, given.typeinertia, defgiven.typeinertia);
    }
    if (mjuu_defined(given.mass)) {
      WriteAttr(elem,
                "mass",
                1,
                &given.mass,
                mjuu_defined(defgiven.mass) ? &defgiven.mass : nullptr);
    }
    WriteAttr(elem, "density", 1, &given.density, &defgiven.density);
    WriteAttr(elem, "fitscale", 1, &given.fitscale, &defgiven.fitscale);
    if (*given.material != *defgiven.material) { WriteAttrTxt(elem, "material", *given.material); }
    if (*given.hfieldname != *defgiven.hfieldname) {
      WriteAttrTxt(elem, "hfield", *given.hfieldname);
    }
    if (*given.meshname != *defgiven.meshname) { WriteAttrTxt(elem, "mesh", *given.meshname); }
    WriteSpecUser(elem, *given.userdata, *defgiven.userdata);
    if (given.plugin.active) { OnePlugin(InsertEnd(elem, "plugin"), &given.plugin); }
    return;
  }

  // regular
  mjsGeom local = *static_cast<const mjsGeom*>(geom);
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", geom->name);
    if (classname != geom->classname) { WriteAttrTxt(elem, "class", geom->classname); }
    if (mjGEOMINFO[geom->type]) {
      WriteAttr(elem, "size", mjGEOMINFO[geom->type], geom->size, def->Geom().size);
    }
    if (mjuu_defined(geom->mass)) { mass = geom->GetVolume() * def->Geom().density; }

    // pose: undo the mesh transformation, then the frame
    double pos[3], quat[4];
    mjuu_copyvec(pos, geom->pos, 3);
    mjuu_copyvec(quat, geom->quat, 4);
    if ((geom->type == mjGEOM_MESH || geom->type == mjGEOM_SDF) && geom->mesh) {
      const double* meshpos  = geom->mesh->GetPosPtr();
      const double* meshquat = geom->mesh->GetQuatPtr();
      mjuu_frameaccuminv(pos, quat, meshpos, meshquat);

      // surfacevel: undo the mesh transformation, angular origin back to the geom frame
      double pxw[3];
      mjuu_rotVecQuat(local.surfacevel, local.surfacevel, meshquat);
      mjuu_rotVecQuat(local.surfacevel + 3, local.surfacevel + 3, meshquat);
      mjuu_crossvec(pxw, meshpos, local.surfacevel + 3);
      mjuu_addtovec(local.surfacevel, pxw, 3);
    }
    FrameLocal(geom->frame, pos, quat);
    WriteAttr(elem, "pos", 3, pos, unitq + 1);
    WriteAttr(elem, "quat", 4, quat, unitq);
  } else {
    WriteAttr(elem, "size", 3, geom->size, def->Geom().size);
  }

  // defaults and regular
  WriteAttrTable(elem, &local, &def->Geom().spec, kGeomAttrs, kGeomAttrsN);
  WriteAttrKey(elem,
               "fluidshape",
               fluidshape_map,
               2,
               geom->fluid_ellipsoid,
               def->Geom().fluid_ellipsoid);
  if (geom->type != mjGEOM_MESH) {
    WriteAttrKey(elem, "shellinertia", bool_map, 2, geom->typeinertia, def->Geom().typeinertia);
  }
  if (mjuu_defined(geom->mass)) {
    WriteAttr(elem, "mass", 1, &geom->mass_, &mass);
  } else {
    WriteAttr(elem, "density", 1, &geom->density, &def->Geom().density);
  }
  if (geom->get_material() != def->Geom().get_material()) {
    WriteAttrTxt(elem, "material", geom->get_material());
  }

  // hfield and mesh attributes
  if (geom->type == mjGEOM_HFIELD) {
    WriteAttrTxt(
        elem,
        "hfield",
        geom->hfield && geom->hfield->primary_ ? geom->hfield->name : geom->get_hfieldname());
  }
  if (geom->type == mjGEOM_MESH || geom->type == mjGEOM_SDF) {
    WriteAttrTxt(elem,
                 "mesh",
                 geom->mesh && geom->mesh->primary_ ? geom->mesh->name : geom->get_meshname());
  }

  // userdata
  if (writingdefaults) {
    WriteVector(elem, "user", geom->get_userdata());
  } else {
    WriteVector(elem, "user", geom->get_userdata(), def->Geom().get_userdata());
  }

  // write plugin
  if (geom->plugin.active) { OnePlugin(InsertEnd(elem, "plugin"), &geom->plugin); }
}

// write site
void mjXWriter::OneSite(XMLElement* elem, const mjCSite* site, mjCDef* def, string_view classname) {
  double unitq[4] = {1, 0, 0, 0};

  // what the spec gives: the size and pose as fromto if they were written so
  if (authored_) {
    const mjsSite& given    = site->spec;
    const mjsSite& defgiven = def->Site().spec;
    if (!writingdefaults) {
      WriteAttrTxt(elem, "name", site->name);
      if (classname != site->classname) { WriteAttrTxt(elem, "class", site->classname); }
    }
    WriteSpecShape(elem, site, given, defgiven);

    WriteAttrTable(elem, &given, &defgiven, kSiteAttrs, kSiteAttrsN);
    if (*given.material != *defgiven.material) { WriteAttrTxt(elem, "material", *given.material); }
    if (*given.meshname != *defgiven.meshname) { WriteAttrTxt(elem, "mesh", *given.meshname); }
    WriteAttr(elem, "rgba", 4, given.rgba, defgiven.rgba);
    WriteSpecUser(elem, *given.userdata, *defgiven.userdata);
    return;
  }

  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", site->name);
    if (classname != site->classname) { WriteAttrTxt(elem, "class", site->classname); }
    if (mjGEOMINFO[site->type]) {
      WriteAttr(elem, "size", mjGEOMINFO[site->type], site->size, def->Site().size);
    }

    // pose relative to the site's frame
    double pos[3], quat[4];
    mjuu_copyvec(pos, site->pos, 3);
    mjuu_copyvec(quat, site->quat, 4);
    FrameLocal(site->frame, pos, quat);
    WriteAttr(elem, "pos", 3, pos, unitq + 1);
    WriteAttr(elem, "quat", 4, quat, unitq);
  } else {
    WriteAttr(elem, "size", 3, site->size, def->Site().size);
  }

  // defaults and regular
  WriteAttrTable(elem,
                 static_cast<const mjsSite*>(site),
                 &def->Site().spec,
                 kSiteAttrs,
                 kSiteAttrsN);
  if (site->get_material() != def->Site().get_material()) {
    WriteAttrTxt(elem, "material", site->get_material());
  }
  if (site->type == mjGEOM_MESH && site->get_meshname() != def->Site().get_meshname()) {
    WriteAttrTxt(elem,
                 "mesh",
                 site->mesh && site->mesh->primary_ ? site->mesh->name : site->get_meshname());
  }
  WriteAttr(elem, "rgba", 4, site->rgba, def->Site().rgba);

  // userdata
  if (writingdefaults) {
    WriteVector(elem, "user", site->get_userdata());
  } else {
    WriteVector(elem, "user", site->get_userdata(), def->Site().get_userdata());
  }
}

// write camera
void mjXWriter::OneCamera(XMLElement*      elem,
                          const mjCCamera* camera,
                          mjCDef*          def,
                          string_view      classname) {
  double unitq[4] = {1, 0, 0, 0};

  // what the spec gives, or the pose relative to the camera's frame
  mjsCamera local = *Values<mjsCamera>(camera);
  if (!authored_) { FrameLocal(camera->frame, local.pos, local.quat); }
  const mjsCamera& deflocal = def->Camera().spec;

  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", camera->name);
    if (classname != camera->classname) { WriteAttrTxt(elem, "class", camera->classname); }
    WriteAttrTxt(elem,
                 "target",
                 Pick(authored_, camera->spec.targetbody, camera->get_targetbody()));
    if (!authored_) { WriteAttr(elem, "quat", 4, local.quat, unitq); }
  }
  if (authored_) {
    WriteSpecPose(elem,
                  camera,
                  nullptr,
                  local.quat,
                  local.alt,
                  nullptr,
                  deflocal.quat,
                  &deflocal.alt);
  }

  // defaults and regular
  WriteAttrTable(elem, &local, &deflocal, kCameraAttrs, kCameraAttrsN);

  // camera intrinsics if specified
  if (local.sensor_size[0] > 0 && local.sensor_size[1] > 0) {
    WriteAttr(elem, "sensorsize", 2, local.sensor_size, deflocal.sensor_size);
    WriteAttr(elem, "focal", 2, local.focal_length, deflocal.focal_length);
    WriteAttr(elem, "focalpixel", 2, local.focal_pixel, deflocal.focal_pixel);
    WriteAttr(elem, "principal", 2, local.principal_length, deflocal.principal_length);
    WriteAttr(elem, "principalpixel", 2, local.principal_pixel, deflocal.principal_pixel);
  } else {
    WriteAttr(elem, "fovy", 1, &local.fovy, &deflocal.fovy);
  }

  // userdata
  const auto& user    = Pick(authored_, camera->spec.userdata, camera->get_userdata());
  const auto& defuser = Pick(authored_, def->Camera().spec.userdata, def->Camera().get_userdata());
  if (authored_) {
    WriteSpecUser(elem, user, defuser);
  } else if (writingdefaults) {
    WriteVector(elem, "user", user);
  } else {
    WriteVector(elem, "user", user, defuser);
  }
}

// write light
void mjXWriter::OneLight(XMLElement*     elem,
                         const mjCLight* light,
                         mjCDef*         def,
                         string_view     classname) {
  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", light->name);
    if (classname != light->classname) { WriteAttrTxt(elem, "class", light->classname); }
    WriteAttrTxt(elem, "target", Pick(authored_, light->spec.targetbody, light->get_targetbody()));
    WriteAttrTxt(elem, "texture", Pick(authored_, light->spec.texture, light->get_texture()));
  }

  // what the spec gives, or pos and dir relative to the light's frame
  mjsLight local = *Values<mjsLight>(light);
  if (!authored_) {
    double iquat[4] = {1, 0, 0, 0};
    FrameLocal(light->frame, local.pos, iquat);
    mjuu_rotVecQuat(local.dir, local.dir, iquat);
  }

  // defaults and regular
  WriteAttrTable(elem, &local, &def->Light().spec, kLightAttrs, kLightAttrsN);
  WriteAttrKey(elem, "type", lighttype_map, lighttype_sz, local.type, def->Light().spec.type);
}

// write pair
// write the mechanical attributes of an element, driven by the same
// generated rows the reader uses; see the declaration for the contract
template <typename T>
void mjXWriter::WriteAttrTable(XMLElement*    elem,
                               const T*       obj,
                               const T*       def,
                               const mjXAttr* rows,
                               int            nrow,
                               bool           given,
                               const T*       field) {
  const char* live = reinterpret_cast<const char*>(obj);
  const char* dflt = reinterpret_cast<const char*>(def);
  const char* spec = reinterpret_cast<const char*>(field ? field : obj);
  for (int i = 0; i < nrow; i++) {
    const mjXAttr& row = rows[i];
    if (row.handwrite || (writingdefaults && row.nodefault)) { continue; }
    const char* base = live + row.offset;
    // a null default object means the element has no defaults to compare
    // against: every defined value is written. So is a value of the model
    // which the spec says was written, whatever its default
    bool        written = given && mjs_isAuthored(&model->spec, spec + row.offset);
    const char* dbase   = dflt && !written ? dflt + row.offset : nullptr;
    const int   dkey    = dflt && !written ? 0 : -12345;  // WriteAttrKey's write-always default
    switch (row.kind) {
      case mjXAttr::kInt:
        WriteAttr(elem,
                  row.attr,
                  row.len,
                  (const int*)base,
                  (const int*)dbase,
                  /*trim=*/!row.exact);
        break;
      case mjXAttr::kDouble:
        WriteAttr(elem,
                  row.attr,
                  row.len,
                  (const double*)base,
                  (const double*)dbase,
                  /*trim=*/!row.exact);
        break;
      case mjXAttr::kNum:
        WriteAttr(elem,
                  row.attr,
                  row.len,
                  (const mjtNum*)base,
                  (const mjtNum*)dbase,
                  /*trim=*/!row.exact);
        break;
      case mjXAttr::kFloat:
        WriteAttr(elem,
                  row.attr,
                  row.len,
                  (const float*)base,
                  (const float*)dbase,
                  /*trim=*/!row.exact);
        break;
      case mjXAttr::kEnum:
        WriteAttrKey(elem,
                     row.attr,
                     row.map,
                     row.mapsz,
                     *(const int*)base,
                     dbase ? *(const int*)dbase : dkey);
        break;
      case mjXAttr::kEnumByte:
        WriteAttrKey(elem,
                     row.attr,
                     row.map,
                     row.mapsz,
                     *(const mjtByte*)base,
                     dbase ? *(const mjtByte*)dbase : dkey);
        break;
      case mjXAttr::kBool:
        WriteAttrKey(elem,
                     row.attr,
                     bool_map,
                     2,
                     *(const mjtByte*)base,
                     dbase ? *(const mjtByte*)dbase : dkey);
        break;
      case mjXAttr::kFlags:
        if (!dbase || *(const int*)base != *(const int*)dbase) {
          int              value = *(const int*)base;
          std::vector<int> data;
          for (int j = 0; j < row.mapsz; j++) {
            if (value & row.map[j].value) { data.push_back(row.map[j].value); }
          }
          if (!data.empty()) {
            WriteAttrKeys(elem, row.attr, row.map, row.mapsz, data.data(), data.size(), 0);
          }
        }
        break;
      default:
        // names, strings, files and custom-read attributes: OneX() remnant
        break;
    }
  }
}


void mjXWriter::OnePair(XMLElement* elem, const mjCPair* pair, mjCDef* def) {
  // regular
  if (!writingdefaults) {
    if (pair->classname != "main") { WriteAttrTxt(elem, "class", pair->classname); }
    WriteAttrTxt(elem, "geom1", Pick(authored_, pair->spec.geomname1, pair->get_geomname1()));
    WriteAttrTxt(elem, "geom2", Pick(authored_, pair->spec.geomname2, pair->get_geomname2()));
  }

  // defaults and regular
  WriteAttrTxt(elem, "name", pair->name);
  WriteAttrTable(elem, Values<mjsPair>(pair), &def->Pair().spec, kPairAttrs, kPairAttrsN);
}


// write equality
void mjXWriter::OneEquality(XMLElement* elem, const mjCEquality* pequality, mjCDef* def) {
  const mjCBase*     base     = pequality;
  const mjsEquality* equality = Values<mjsEquality>(pequality);

  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", base->name);
    if (base->classname != "main") { WriteAttrTxt(elem, "class", base->classname); }

    switch (equality->type) {
      case mjEQ_CONNECT:
        if (equality->objtype == mjOBJ_BODY) {
          WriteAttrTxt(elem, "body1", mjs_getString(equality->name1));
          WriteAttrTxt(elem, "body2", mjs_getString(equality->name2));
          WriteAttr(elem, "anchor", 3, equality->data);
        } else {
          WriteAttrTxt(elem, "site1", mjs_getString(equality->name1));
          WriteAttrTxt(elem, "site2", mjs_getString(equality->name2));
        }
        break;

      case mjEQ_WELD:
        if (equality->objtype == mjOBJ_BODY) {
          WriteAttrTxt(elem, "body1", mjs_getString(equality->name1));
          WriteAttrTxt(elem, "body2", mjs_getString(equality->name2));
          // unlike connect, weld's body semantic does not require anchor,
          // and the reader zeroes it when absent: zeros is the default,
          // not the constructor's union payload
          double zero3[3] = {0, 0, 0};
          WriteAttr(elem, "anchor", 3, equality->data, zero3);

          // a relative pose which mj_setConst computes, marked in data[11], is written as none:
          // a zero quaternion
          const double* defpose = def->Equality().spec.data + 3;
          double        relpose[7];
          std::copy_n(equality->data + 3, 7, relpose);
          if (!authored_ && equality->data[11]) {
            std::copy_n(defpose, 3, relpose);
            std::fill_n(relpose + 3, 4, 0);
          }
          WriteAttr(elem, "relpose", 7, relpose, defpose);
        } else {
          WriteAttrTxt(elem, "site1", mjs_getString(equality->name1));
          WriteAttrTxt(elem, "site2", mjs_getString(equality->name2));
        }
        WriteAttr(elem, "torquescale", 1, equality->data + 10, def->Equality().spec.data + 10);
        break;

      case mjEQ_JOINT:
        WriteAttrTxt(elem, "joint1", mjs_getString(equality->name1));
        WriteAttrTxt(elem, "joint2", mjs_getString(equality->name2));
        WriteAttr(elem, "polycoef", 5, equality->data, def->Equality().spec.data);
        break;

      case mjEQ_TENDON:
        WriteAttrTxt(elem, "tendon1", mjs_getString(equality->name1));
        WriteAttrTxt(elem, "tendon2", mjs_getString(equality->name2));
        WriteAttr(elem, "polycoef", 5, equality->data, def->Equality().spec.data);
        break;

      case mjEQ_FLEX:
        WriteAttrTxt(elem, "flex", mjs_getString(equality->name1));
        break;

      case mjEQ_FLEXSTRAIN:
        WriteAttrTxt(elem, "flex", mjs_getString(equality->name1));
        WriteAttr(elem, "cell", 3, equality->data, def->Equality().spec.data);
        break;

      default:
        throw mjXError(0, "unknown equality type");
    }
  }

  // defaults and regular
  WriteAttrTable(elem, equality, &def->Equality().spec, kEqualityBaseAttrs, kEqualityBaseAttrsN);
}


// write tendon
void mjXWriter::OneTendon(XMLElement* elem, const mjCTendon* ptendon, mjCDef* def) {
  const mjCTendon* base   = ptendon;
  const mjsTendon* tendon = Values<mjsTendon>(ptendon);
  bool             fixed  = (base->Type() == mjTENDON_FIXED);

  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", base->name);
    if (base->classname != "main") { WriteAttrTxt(elem, "class", base->classname); }
  }

  // defaults and regular; the fixed rows are the spatial rows without the
  // appearance attributes, which is exactly the tag difference
  const mjsTendon& deftendon = def->Tendon().spec;
  if (fixed) {
    WriteAttrTable(elem, tendon, &deftendon, kFixedAttrs, kFixedAttrsN);
  } else {
    WriteAttrTable(elem, tendon, &deftendon, kSpatialAttrs, kSpatialAttrsN);
  }
  if (tendon->springlength[0] != tendon->springlength[1] ||
      deftendon.springlength[0] != deftendon.springlength[1]) {
    WriteAttr(elem, "springlength", 2, tendon->springlength, deftendon.springlength);
  } else {
    WriteAttr(elem, "springlength", 1, tendon->springlength, deftendon.springlength);
  }
  const string& material = Pick(authored_, base->spec.material, base->get_material());
  if (!fixed && material != Pick(authored_, deftendon.material, def->Tendon().get_material())) {
    WriteAttrTxt(elem, "material", material);
  }

  // userdata
  const auto& user    = Pick(authored_, base->spec.userdata, base->get_userdata());
  const auto& defuser = Pick(authored_, deftendon.userdata, def->Tendon().get_userdata());
  if (authored_) {
    WriteSpecUser(elem, user, defuser);
  } else if (writingdefaults) {
    WriteVector(elem, "user", user);
  } else {
    WriteVector(elem, "user", user, defuser);
  }
}


// the shortcut which gives an actuator back when its tag is reloaded over the class default, as
// an entry of the actuator dispatch; the general entry if there is none, and always in the
// canonical notation. s and d get the shortcut parameters of the actuator and of the class slot
// the shortcut pre-reads
const mjXActuatorEntry* mjXWriter::ShortcutEntry(const mjsActuator* actuator,
                                                 const mjCDef*      def,
                                                 mjXShortcut*       s,
                                                 mjXShortcut*       d) const {
  const mjXActuatorEntry* general = nullptr;
  const mjXActuatorEntry* entry   = nullptr;
  for (int i = 0; i < kActuatorDispatchN; i++) {
    if (kActuatorDispatch[i].type == mjACTUATOR_GENERAL) { general = kActuatorDispatch + i; }
    if (kActuatorDispatch[i].type == actuator->type) { entry = kActuatorDispatch + i; }
  }
  if (actuator->plugin.active || canonical_ || !entry || entry == general) { return general; }
  mjtActuator        type   = actuator->type;
  const mjsActuator& defact = *def->spec.actuator;
  const mjsActuator* src    = mjXShortcutSource(&defact, def, type);
  mjXShortcutFromActuator(s, actuator, type);
  if (src) {
    mjXShortcutFromActuator(d, src, type);
  } else {
    mjXShortcutDefaults(d, type);
  }

  // what is not given is inherited: a zero kv or timeconst overrides the class, an input
  // signature cannot be unset
  if (!s->has_kv && !s->has_dampratio && (d->has_kv || d->has_dampratio)) {
    s->has_kv = true;
    s->kv     = 0;
  }
  if (d->has_timeconst && !s->has_timeconst) {
    s->has_timeconst = true;
    s->timeconst[0]  = 0;
  }
  if (!s->ctrlspec) { s->ctrlspec = d->ctrlspec; }

  // reload: the class default, the tag's mechanical attributes, then the shortcut
  mjsActuator copy = defact;
  CopyRows(&copy, actuator, entry->rows, entry->n, writingdefaults);

  // a written saturation replaces the class torque limit, as in the reader
  if (type == mjACTUATOR_DCMOTOR && !std::equal(s->saturation, s->saturation + 3, d->saturation)) {
    copy.forcelimited  = mjLIMITED_AUTO;
    copy.forcerange[0] = copy.forcerange[1] = 0;
  }
  if (mjXSetToShortcut(&copy, type, *s)[0]) { return general; }
  return SameActuator(copy, *actuator) ? entry : general;
}


// write actuator
XMLElement* mjXWriter::OneActuator(XMLElement* section, const mjCActuator* pactuator, mjCDef* def) {
  const mjCActuator* base     = pactuator;
  const mjsActuator* actuator = Values<mjsActuator>(pactuator);
  const mjsActuator& defact   = def->Actuator().spec;

  // the tag: a plugin, the shortcut which gives the actuator back, or general
  mjXShortcut             s = {}, d = {};
  const mjXActuatorEntry* entry = ShortcutEntry(actuator, def, &s, &d);
  mjtActuator             type  = (mjtActuator)entry->type;
  XMLElement* elem = InsertEnd(section, actuator->plugin.active ? "plugin" : entry->tag);

  // regular
  if (!writingdefaults) {
    WriteAttrTxt(elem, "name", base->name);
    if (base->classname != "main") { WriteAttrTxt(elem, "class", base->classname); }

    // transmission target
    switch (actuator->trntype) {
      case mjTRN_JOINT:
        WriteAttrTxt(elem, "joint", base->get_target());
        break;

      case mjTRN_JOINTINPARENT:
        WriteAttrTxt(elem, "jointinparent", base->get_target());
        break;

      case mjTRN_TENDON:
        WriteAttrTxt(elem, "tendon", base->get_target());
        break;

      case mjTRN_SLIDERCRANK:
        WriteAttrTxt(elem, "cranksite", base->get_target());
        WriteAttrTxt(elem, "slidersite", base->get_slidersite());
        break;

      case mjTRN_SITE:
        WriteAttrTxt(elem, "site", base->get_target());
        WriteAttrTxt(elem, "refsite", base->get_refsite());
        break;

      case mjTRN_BODY:
        WriteAttrTxt(elem, "body", base->get_target());
        break;

      default:  // SHOULD NOT OCCUR
        break;
    }
  }

  // defaults and regular: the mechanical attributes of the tag
  WriteAttrTable(elem, actuator, &defact, entry->rows, entry->n);
  WriteAttr(elem, "cranklength", 1, &actuator->cranklength, &defact.cranklength);

  // a range which is inherited stays so in what the spec gives; compilation resolves it
  bool has_inheritrange = type == mjACTUATOR_GENERAL ||
                          type == mjACTUATOR_POSITION ||
                          type == mjACTUATOR_INTVELOCITY ||
                          type == mjACTUATOR_PID;
  if (authored_ && has_inheritrange) {
    const double* definherit = type == mjACTUATOR_GENERAL ? &defact.inheritrange : &d.inheritrange;
    WriteAttr(elem, "inheritrange", 1, &actuator->inheritrange, definherit);
  }

  // shortcut parameters, where they differ from those of the class
  auto Damping = [&]() {
    if (s.has_kv) {
      WriteAttr(elem, "kv", 1, &s.kv, d.has_kv ? &d.kv : nullptr);
    } else if (s.has_dampratio) {
      WriteAttr(elem, "dampratio", 1, &s.dampratio, d.has_dampratio ? &d.dampratio : nullptr);
    }
  };
  auto Input = [&]() {
    if (s.ctrlspec != d.ctrlspec) {
      elem->SetAttribute("input", InputString(actuator->gaintype, s.ctrlspec).c_str());
    }
  };
  switch (type) {
    case mjACTUATOR_POSITION:
    case mjACTUATOR_INTVELOCITY:
      WriteAttr(elem, "kp", 1, &s.kp, &d.kp);
      Damping();
      if (s.has_timeconst) {
        WriteAttr(elem, "timeconst", 1, s.timeconst, d.has_timeconst ? d.timeconst : nullptr);
      }
      break;
    case mjACTUATOR_ORIENTATION:
      WriteAttr(elem, "kp", 1, &s.kp, &d.kp);
      Damping();
      Input();
      break;
    case mjACTUATOR_PID:
      WriteAttr(elem, "kp", 1, &s.kp, &d.kp);
      Damping();
      WriteAttr(elem, "ki", 1, &s.ki, &d.ki);
      WriteAttr(elem, "imax", 1, &s.imax, &d.imax);
      WriteAttr(elem, "slewmax", 1, &s.slewmax, &d.slewmax);
      Input();
      break;
    case mjACTUATOR_VELOCITY:
    case mjACTUATOR_DAMPER:
      WriteAttr(elem, "kv", 1, &s.kv, &d.kv);
      break;
    case mjACTUATOR_CYLINDER:
      WriteAttr(elem, "timeconst", 1, s.timeconst, d.timeconst);
      WriteAttr(elem, "area", 1, &s.area, &d.area);
      WriteAttr(elem, "bias", 3, s.bias, d.bias);
      break;
    case mjACTUATOR_MUSCLE:
      WriteAttr(elem, "timeconst", 2, s.timeconst, d.timeconst);
      WriteAttr(elem, "tausmooth", 1, &s.tausmooth, &d.tausmooth);
      WriteAttr(elem, "range", 2, s.range, d.range);
      WriteAttr(elem, "force", 1, &s.force, &d.force);
      WriteAttr(elem, "scale", 1, &s.scale, &d.scale);
      WriteAttr(elem, "lmin", 1, &s.lmin, &d.lmin);
      WriteAttr(elem, "lmax", 1, &s.lmax, &d.lmax);
      WriteAttr(elem, "vmax", 1, &s.vmax, &d.vmax);
      WriteAttr(elem, "fpmax", 1, &s.fpmax, &d.fpmax);
      WriteAttr(elem, "fvmax", 1, &s.fvmax, &d.fvmax);
      break;
    case mjACTUATOR_ADHESION:
      WriteAttr(elem, "gain", 1, &s.gain, &d.gain);
      break;
    case mjACTUATOR_DCMOTOR:
      WriteAttr(elem, "motorconst", 2, s.motorconst, d.motorconst, true);
      WriteAttr(elem, "resistance", 1, &s.resistance, &d.resistance);
      WriteAttr(elem, "saturation", 3, s.saturation, d.saturation, true);
      WriteAttr(elem, "inductance", 2, s.inductance, d.inductance, true);
      WriteAttr(elem, "cogging", 3, s.cogging, d.cogging, true);
      WriteAttr(elem, "controller", 6, s.controller, d.controller, true);
      WriteAttr(elem, "thermal", 6, s.thermal, d.thermal, true);
      WriteAttr(elem, "lugre", 5, s.lugre, d.lugre, true);
      Input();
      break;
    default:
      break;
  }

  // general and plugins
  if (type == mjACTUATOR_GENERAL) {
    // special handling of actdim which has default value of -1
    if (writingdefaults) {
      WriteAttrInt(elem, "actdim", actuator->actdim, defact.actdim);
    } else {
      // compilation resolves an actdim which was left out; what the spec gives is written as it is
      int default_actdim = (actuator->dyntype != mjDYN_NONE && actuator->dyntype != mjDYN_DCMOTOR);
      WriteAttrInt(elem, "actdim", actuator->actdim, authored_ ? -1 : default_actdim);
    }

    // plugins: write config attributes
    if (actuator->plugin.active) {
      OnePlugin(elem, &actuator->plugin);
    }

    // non-plugins: write actuator parameters
    else {
      WriteAttrKey(elem, "gaintype", gain_map, gain_sz, actuator->gaintype, defact.gaintype);

      // input signature, inherited only from a default with the same gaintype; written even when
      // empty, which restores the gaintype's default signature
      int defspec = actuator->gaintype == defact.gaintype ? defact.ctrlspec : 0;
      if (actuator->ctrlspec != defspec) {
        elem->SetAttribute("input", InputString(actuator->gaintype, actuator->ctrlspec).c_str());
      }
      WriteAttrKey(elem, "biastype", bias_map, bias_sz, actuator->biastype, defact.biastype);
      WriteAttr(elem, "gainprm", mjNGAIN, actuator->gainprm, defact.gainprm, true);
      WriteAttr(elem, "biasprm", mjNBIAS, actuator->biasprm, defact.biasprm, true);
    }
  }

  // userdata
  const auto& user    = Pick(authored_, base->spec.userdata, base->get_userdata());
  const auto& defuser = Pick(authored_, defact.userdata, def->Actuator().get_userdata());
  if (authored_) {
    WriteSpecUser(elem, user, defuser);
  } else if (writingdefaults) {
    WriteVector(elem, "user", user);
  } else {
    WriteVector(elem, "user", user, defuser);
  }
  return elem;
}


// write plugin
void mjXWriter::OnePlugin(XMLElement* elem, const mjsPlugin* plugin) {
  const string instance_name = string(mjs_getString(plugin->name));
  const string plugin_name   = string(mjs_getString(plugin->plugin_name));
  if (!instance_name.empty()) {
    WriteAttrTxt(elem, "instance", instance_name);
  } else {
    WriteAttrTxt(elem, "plugin", plugin_name);
    PluginConfig(elem, static_cast<mjCPlugin*>(plugin->element), plugin_name);
  }
}


// write the configuration of a plugin instance: the attributes which the spec gives it, or those
// which compilation laid out for the plugin
void mjXWriter::PluginConfig(XMLElement* elem, const mjCPlugin* instance, const string& plugin) {
  if (authored_) {
    auto write = [&](const string& key, const string& value) {
      XMLElement* config_elem = InsertEnd(elem, "config");
      WriteAttrTxt(config_elem, "key", key);
      WriteAttrTxt(config_elem, "value", value);
    };

    // in the order in which the plugin declares its attributes, if the plugin is registered,
    // followed by any others
    const auto&         given = instance->config_attribs;
    std::vector<string> declared;
    if (const mjpPlugin* pplugin = mjp_getPlugin(plugin.c_str(), nullptr)) {
      for (int i = 0; i < pplugin->nattribute; i++) { declared.push_back(pplugin->attributes[i]); }
    }
    for (const string& key : declared) {
      if (auto found = given.find(key); found != given.end()) { write(key, found->second); }
    }
    for (const auto& [key, value] : given) {
      if (std::find(declared.begin(), declared.end(), key) == declared.end()) { write(key, value); }
    }
    return;
  }

  const mjpPlugin* pplugin = mjp_getPluginAtSlot(instance->plugin_slot);
  const char*      c       = &instance->flattened_attributes[0];
  for (int i = 0; i < pplugin->nattribute; ++i) {
    string value(c);
    if (!value.empty()) {
      XMLElement* config_elem = InsertEnd(elem, "config");
      WriteAttrTxt(config_elem, "key", pplugin->attributes[i]);
      WriteAttrTxt(config_elem, "value", value);
      c += value.size();
    }
    ++c;
  }
}


//---------------------------------- class mjXWriter: top-level API --------------------------------

// constructor
mjXWriter::mjXWriter(void) {
  writingdefaults = false;
}


// cast model; copy back what was changed in the model, which fails if the model was not compiled
// from the spec or if the spec cannot express a change
void mjXWriter::SetModel(mjSpec* _spec, const mjModel* m) {
  if (_spec) { model = static_cast<mjCModel*>(_spec->element); }
  if (m && !mj_copyBack(&model->spec, m)) { throw mjXError(0, "%s", mjs_getError(&model->spec)); }
}


// save the model as MJCF: what compilation made of the spec, which must be compiled, or what the
// spec gives
string mjXWriter::Write(char* error, size_t error_sz) {
  if (!model) {
    mjCopyError(error, "XML Write error: no model to write", error_sz);
    return "";
  }

  // what is saved, and in which notation
  const mjsCompiler& settings = model->spec.compiler;
  authored_                   = !settings.savecompiled;
  canonical_                  = !authored_ || settings.savecanonical;
  degree_                     = !canonical_ && settings.degree;

  // what the spec gives is saved exactly: each number as the shortest text which reads back as it
  std::optional<mujoco::ExactFloatPrecision> exact;
  if (authored_) { exact.emplace(); }

  // compiled values, and inertials which are calculated, need a compiled model
  if (!model->IsCompiled() && (!authored_ || settings.saveinertial)) {
    mjCopyError(error, "XML Write error: Only compiled model can be written", error_sz);
    return "";
  }

  // what the compilation left in the elements no longer describes the model
  if (model->StructureChanged() && (!authored_ || settings.saveinertial)) {
    mjCopyError(error,
                "XML Write error: Model structure changed after compilation. It must be "
                "recompiled before writing XML.",
                error_sz);
    return "";
  }

  // what the spec gives is compiled with the settings of the one compiler element of the file
  if (authored_) {
    string attached = AttachedSettings();
    if (!attached.empty()) {
      mjCopyError(error, attached.c_str(), error_sz);
      return "";
    }
  }

  // create document and root
  XMLDocument doc;
  XMLElement* root = doc.NewElement("mujoco");
  root->SetAttribute("model", mjs_getString(authored_ ? model->spec.modelname : model->modelname));

  // insert root
  doc.InsertFirstChild(root);

  // write comment if present
  string text = mjs_getString(authored_ ? model->spec.comment : model->comment);
  if (!text.empty()) {
    XMLComment* comment = doc.NewComment(text.c_str());
    root->LinkEndChild(comment);
  }

  // create DOM
  Compiler(root);
  Option(root);
  Size(root);
  Statistic(root);
  Visual(root);
  writingdefaults = true;
  Default(root, model->Default());
  writingdefaults = false;
  Extension(root);
  Asset(root);
  Body(InsertEnd(root, "worldbody"), model->GetWorld(), nullptr);
  Deformable(root);
  Contact(root);
  Tendon(root);
  Equality(root);
  Actuator(root);
  Sensor(root);
  Custom(root);
  Keyframe(root);

  return WriteDoc(doc, error, error_sz);
}


// compiler section
void mjXWriter::Compiler(XMLElement* root) {
  mjSpec defspec;
  mjs_defaultSpec(&defspec);

  XMLElement* section = InsertEnd(root, "compiler");

  // the settings which the spec gives, apart from those which say how to save: a model which is
  // saved as written is compiled with them again
  if (authored_) {
    mjsCompiler given   = model->spec.compiler;
    given.saveinertial  = defspec.compiler.saveinertial;
    given.savecompiled  = defspec.compiler.savecompiled;
    given.savecanonical = defspec.compiler.savecanonical;
    given.degree        = degree_;
    if (canonical_) { mjuu_copyvec(given.eulerseq, defspec.compiler.eulerseq, 3); }
    WriteAttrTable(section,
                   &given,
                   &defspec.compiler,
                   kCompilerAttrs,
                   kCompilerAttrsN,
                   /*given=*/true,
                   &model->spec.compiler);
    for (const char* name : {"saveinertial", "savecompiled", "savecanonical"}) {
      section->DeleteAttribute(name);
    }
    if (std::strncmp(given.eulerseq, defspec.compiler.eulerseq, 3) ||
        (!canonical_ && mjs_isAuthored(&model->spec, model->spec.compiler.eulerseq))) {
      WriteAttrTxt(section, "eulerseq", string(given.eulerseq, 3));
    }
    WriteAttrTxt(section, "meshdir", *given.meshdir);
    WriteAttrTxt(section, "texturedir", *given.texturedir);
    if (model->spec.strippath) { WriteAttrTxt(section, "strippath", "true"); }

    XMLElement* lengthrange = InsertEnd(section, "lengthrange");
    WriteAttrTable(lengthrange,
                   &given.LRopt,
                   &defspec.compiler.LRopt,
                   kLengthrangeAttrs,
                   kLengthrangeAttrsN);
    if (!lengthrange->FirstAttribute()) { section->DeleteChild(lengthrange); }
    if (!section->FirstAttribute() && !section->FirstChildElement()) { root->DeleteChild(section); }
    return;
  }

  // settings
  WriteAttrTxt(section, "angle", "radian");
  if (!model->get_meshdir().empty()) { WriteAttrTxt(section, "meshdir", model->get_meshdir()); }
  if (!model->get_texturedir().empty()) {
    WriteAttrTxt(section, "texturedir", model->get_texturedir());
  }
  if (!model->compiler.usethread) { WriteAttrTxt(section, "usethread", "false"); }

  if (model->compiler.boundmass) { WriteAttr(section, "boundmass", 1, &model->compiler.boundmass); }
  if (model->compiler.boundinertia) {
    WriteAttr(section, "boundinertia", 1, &model->compiler.boundinertia);
  }
  if (model->compiler.inertiafromgeom == mjINERTIAFROMGEOM_FALSE) {
    WriteAttrTxt(section, "inertiafromgeom", "false");
  }
  WriteAttr(section,
            "inertiagrouprange",
            2,
            model->compiler.inertiagrouprange,
            defspec.compiler.inertiagrouprange);
  if (model->compiler.alignfree) { WriteAttrTxt(section, "alignfree", "true"); }
  if (!model->compiler.autolimits) { WriteAttrTxt(section, "autolimits", "false"); }
  WriteAttrKey(section,
               "conflict",
               conflict_map,
               conflict_sz,
               model->compiler.conflict,
               mjCONFLICT_WARNING);
  WriteAttrKey(section,
               "passthrough",
               passthrough_map,
               passthrough_sz,
               model->compiler.passthrough,
               mjPASSTHROUGH_NONE);
}


// option section
void mjXWriter::Option(XMLElement* root) {
  mjOption opt;
  mj_defaultOption(&opt);

  XMLElement* section = InsertEnd(root, "option");

  // option; the freshly-defaulted struct is the comparison object
  const mjOption& option = authored_ ? model->spec.option : model->option;
  WriteAttrTable(section, &option, &opt, kOptionAttrs, kOptionAttrsN, authored_);

  // actuator group disable
  int disabled_groups[31];
  int ndisabled = 0;
  for (int i = 0; i < 31; ++i) {
    if (option.disableactuator & (1 << i)) { disabled_groups[ndisabled++] = i; }
  }
  WriteAttr(section, "actuatorgroupdisable", ndisabled, disabled_groups);

  // write disable/enable flags if any of them are set; invert while writing. A flag which was
  // written in the spec is saved also if it has its default value
  const int givendisable = authored_ ? model->spec.authored.disableflags : 0;
  const int givenenable  = authored_ ? model->spec.authored.enableflags : 0;
  if (option.disableflags || option.enableflags || givendisable || givenenable) {
    XMLElement* sub = InsertEnd(section, "flag");

#define WRITEDSBL(NAME, MASK)                  \
  if (option.disableflags & MASK)              \
    WriteAttrKey(sub, NAME, enable_map, 2, 0); \
  else if (givendisable & MASK)                \
    WriteAttrKey(sub, NAME, enable_map, 2, 1, -1);
    // clang-format off
    WRITEDSBL("constraint",     mjDSBL_CONSTRAINT)
    WRITEDSBL("equality",       mjDSBL_EQUALITY)
    WRITEDSBL("frictionloss",   mjDSBL_FRICTIONLOSS)
    WRITEDSBL("limit",          mjDSBL_LIMIT)
    WRITEDSBL("contact",        mjDSBL_CONTACT)
    WRITEDSBL("spring",         mjDSBL_SPRING)
    WRITEDSBL("damper",         mjDSBL_DAMPER)
    WRITEDSBL("gravity",        mjDSBL_GRAVITY)
    WRITEDSBL("clampctrl",      mjDSBL_CLAMPCTRL)
    WRITEDSBL("warmstart",      mjDSBL_WARMSTART)
    WRITEDSBL("filterparent",   mjDSBL_FILTERPARENT)
    WRITEDSBL("actuation",      mjDSBL_ACTUATION)
    WRITEDSBL("refsafe",        mjDSBL_REFSAFE)
    WRITEDSBL("sensor",         mjDSBL_SENSOR)
    WRITEDSBL("midphase",       mjDSBL_MIDPHASE)
    WRITEDSBL("eulerdamp",      mjDSBL_EULERDAMP)
    WRITEDSBL("autoreset",      mjDSBL_AUTORESET)
    WRITEDSBL("nativeccd",      mjDSBL_NATIVECCD)
    WRITEDSBL("island",         mjDSBL_ISLAND)
    WRITEDSBL("multiccd",       mjDSBL_MULTICCD)
    // clang-format on
#undef WRITEDSBL

#define WRITEENBL(NAME, MASK)                  \
  if (option.enableflags & MASK)               \
    WriteAttrKey(sub, NAME, enable_map, 2, 1); \
  else if (givenenable & MASK)                 \
    WriteAttrKey(sub, NAME, enable_map, 2, 0, -1);
    // clang-format off
    WRITEENBL("override",       mjENBL_OVERRIDE)
    WRITEENBL("energy",         mjENBL_ENERGY)
    WRITEENBL("fwdinv",         mjENBL_FWDINV)
    WRITEENBL("invdiscrete",    mjENBL_INVDISCRETE)
    WRITEENBL("sleep",          mjENBL_SLEEP)
    WRITEENBL("diagexact",      mjENBL_DIAGEXACT)
    WRITEENBL("ipc",            mjENBL_IPC)
    WRITEENBL("speculative",    mjENBL_SPECULATIVE)
    // clang-format on
#undef WRITEENBL
  }

  // remove entire section if no attributes or elements
  if (!section->FirstAttribute() && !section->FirstChildElement()) { root->DeleteChild(section); }
}


// size section
void mjXWriter::Size(XMLElement* root) {
  XMLElement*   section = InsertEnd(root, "size");
  const mjSpec* sizes   = authored_ ? &model->spec : static_cast<const mjSpec*>(model);

  // write memory
  if (sizes->memory != -1) { WriteAttrTxt(section, "memory", mju_writeNumBytes(sizes->memory)); }

  // deprecated sizes, hand-read into locals with range checks
  WriteAttrInt(section, "njmax", sizes->njmax, -1);
  WriteAttrInt(section, "nconmax", sizes->nconmax, -1);
  WriteAttrInt(section, "nstack", sizes->nstack, -1);

  // write sizes; the spec defaults are -1 (auto), but compilation resolves
  // them to the actual counts, so the comparison object is zero-initialized;
  // what the spec gives is compared with its defaults
  mjSpec zerospec = {};
  if (authored_) { mjs_defaultSpec(&zerospec); }
  WriteAttrTable(section, sizes, &zerospec, kSizeAttrs, kSizeAttrsN);

  // remove entire section if no attributes
  if (!section->FirstAttribute()) root->DeleteChild(section);
}


// statistic section
void mjXWriter::Statistic(XMLElement* root) {
  XMLElement* section = InsertEnd(root, "statistic");

  // statistics are unset (mjNAN) rather than defaulted: there is nothing to
  // compare against, and WriteAttr skips the undefined values by itself
  WriteAttrTable(section,
                 authored_ ? &model->spec.stat : &model->stat,
                 (const mjStatistic*)nullptr,
                 kStatisticAttrs,
                 kStatisticAttrsN);

  // remove entire section if no attributes
  if (!section->FirstAttribute()) root->DeleteChild(section);
}


// visual section
void mjXWriter::Visual(XMLElement* root) {
  mjVisual visdef, *vis = authored_ ? &model->spec.visual : &model->visual;
  mj_defaultVisual(&visdef);

  XMLElement* section = InsertEnd(root, "visual");

  // the sub-sections are projections into mjVisual: their rows carry
  // member-path offsets, so one struct pair drives them all
  struct {
    const char*    tag;
    const mjXAttr* rows;
    int            n;
  } subs[] = {
      {"global",    kGlobalAttrs,    kGlobalAttrsN   },
      {"quality",   kQualityAttrs,   kQualityAttrsN  },
      {"headlight", kHeadlightAttrs, kHeadlightAttrsN},
      {"map",       kMapAttrs,       kMapAttrsN      },
      {"scale",     kScaleAttrs,     kScaleAttrsN    },
      {"rgba",      kRgbaAttrs,      kRgbaAttrsN     },
  };
  for (const auto& sub : subs) {
    XMLElement* elem = InsertEnd(section, sub.tag);
    WriteAttrTable(elem, vis, &visdef, sub.rows, sub.n, authored_);
    if (sub.rows == kHeadlightAttrs) {
      int  value         = std::clamp(vis->headlight.active, 0, 2);
      bool elem_authored = authored_ && mjs_isAuthored(&model->spec, &vis->headlight.active);
      int  def           = elem_authored ? kDefaultUnauthoredValue : visdef.headlight.active;
      WriteAttrKey(elem, "active", FalseTrueAuto_map, 3, value, def);
    }
    if (!elem->FirstAttribute()) { section->DeleteChild(elem); }
  }

  // remove entire section if no elements
  if (!section->FirstChildElement()) { root->DeleteChild(section); }
}


// default section
void mjXWriter::Default(XMLElement* root, mjCDef* def) {
  XMLElement* elem;
  XMLElement* section;

  // pointer to parent defaults
  mjCDef* parent;
  if (def->parent) {
    parent = def->parent;
  } else {
    parent = new mjCDef;
  }

  // create section, write class name
  section = InsertEnd(root, "default");
  if (def->name != "main") { WriteAttrTxt(section, "class", def->name); }

  // mesh
  elem = InsertEnd(section, "mesh");
  OneMesh(elem, &def->Mesh(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // texture
  elem = InsertEnd(section, "texture");
  OneTexture(elem, &def->Texture(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // material
  elem = InsertEnd(section, "material");
  OneMaterial(elem, &def->Material(), parent);
  if (!elem->FirstAttribute() && elem->NoChildren()) section->DeleteChild(elem);

  // joint
  elem = InsertEnd(section, "joint");
  OneJoint(elem, &def->Joint(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // geom
  elem = InsertEnd(section, "geom");
  OneGeom(elem, &def->Geom(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // site
  elem = InsertEnd(section, "site");
  OneSite(elem, &def->Site(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // camera
  elem = InsertEnd(section, "camera");
  OneCamera(elem, &def->Camera(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // light
  elem = InsertEnd(section, "light");
  OneLight(elem, &def->Light(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // pair
  elem = InsertEnd(section, "pair");
  OnePair(elem, &def->Pair(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // equality
  elem = InsertEnd(section, "equality");
  OneEquality(elem, &def->Equality(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // tendon
  elem = InsertEnd(section, "tendon");
  OneTendon(elem, &def->Tendon(), parent);
  if (!elem->FirstAttribute()) section->DeleteChild(elem);

  // actuator: an empty element whose tag is the parent's type is a no-op
  elem      = OneActuator(section, &def->Actuator(), parent);
  int ptype = canonical_ ? mjACTUATOR_GENERAL : parent->Actuator().spec.type;
  if (!elem->FirstAttribute() &&
      FindKey(actuatortype_map, actuatortype_sz, elem->Value()) == ptype) {
    section->DeleteChild(elem);
  }

  // if top-level class has no members or children, delete it and return
  if (!def->parent && section->NoChildren() && def->child.empty()) {
    root->DeleteChild(section);
    delete parent;
    return;
  }

  // add children recursively
  for (int i = 0; i < (int)def->child.size(); i++) { Default(section, def->child[i]); }

  // delete parent defaults if allocated here
  if (!def->parent) { delete parent; }
}


// extension section
void mjXWriter::Extension(XMLElement* root) {
  // skip section if there is no required plugin
  if (model->ActivePlugins().empty()) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "extension");

  // keep track of plugins whose <plugin> section have been created
  std::unordered_set<string> seen_plugins;

  // write all plugins
  string      last_plugin;
  XMLElement* plugin_elem = nullptr;
  for (int i = 0; i < model->Plugins().size(); ++i) {
    mjCPlugin* pp = static_cast<mjCPlugin*>(model->GetObject(mjOBJ_PLUGIN, i));

    if (pp->name.empty()) {
      // reached the first unnamed plugin instance, meaning that it was created through an
      // "implicit" plugin element, e.g. sensor or actuator
      break;
    }

    // the plugin which the instance is of: the one which the spec names, or the one which
    // compilation found for it
    string plugin = authored_ ? pp->plugin_name : "";
    if (plugin.empty() && pp->plugin_slot != -1) {
      plugin = mjp_getPluginAtSlot(pp->plugin_slot)->name;
    }
    if (plugin.empty()) {
      string message = "plugin instance '" + pp->name + "' does not name its plugin";
      throw mjXError(0, "%s", message.c_str());
    }

    // check if we need to open a new <plugin> section
    if (plugin != last_plugin) {
      plugin_elem = InsertEnd(section, "plugin");
      WriteAttrTxt(plugin_elem, "plugin", plugin);
      seen_plugins.insert(plugin);
      last_plugin = plugin;
    }

    // write instance element
    XMLElement* elem = InsertEnd(plugin_elem, "instance");
    WriteAttrTxt(elem, "name", pp->name);

    // write plugin config attributes
    PluginConfig(elem, pp, plugin);
  }

  // write <plugin> elements for plugins without explicit instances
  for (const auto& [plugin, slot] : model->ActivePlugins()) {
    if (seen_plugins.find(plugin->name) == seen_plugins.end()) {
      plugin_elem = InsertEnd(section, "plugin");
      WriteAttrTxt(plugin_elem, "plugin", plugin->name);
    }
  }
}


// custom section
void mjXWriter::Custom(XMLElement* root) {
  XMLElement* elem;

  // get sizes, skip section if empty
  int nnum = model->NumObjects(mjOBJ_NUMERIC);
  int ntxt = model->NumObjects(mjOBJ_TEXT);
  int ntup = model->NumObjects(mjOBJ_TUPLE);

  // skip section if empty
  if (nnum == 0 && ntxt == 0 && ntup == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "custom");

  // write all numerics
  for (int i = 0; i < nnum; i++) {
    mjCNumeric* numeric = (mjCNumeric*)model->GetObject(mjOBJ_NUMERIC, i);

    elem = InsertEnd(section, "numeric");
    WriteAttrTxt(elem, "name", numeric->name);
    if (authored_) {
      // the data as it was given: the size counts the zeros which follow it
      const std::vector<double>& data = numeric->spec_data_;
      if (numeric->spec.size) { WriteAttrInt(elem, "size", numeric->spec.size); }
      WriteAttr(elem, "data", data.size(), data.data());
    } else {
      WriteAttrInt(elem, "size", numeric->size);
      WriteAttr(elem, "data", numeric->size, numeric->data_.data());
    }
  }

  // write all texts
  for (int i = 0; i < ntxt; i++) {
    mjCText* text = (mjCText*)model->GetObject(mjOBJ_TEXT, i);

    elem = InsertEnd(section, "text");
    WriteAttrTxt(elem, "name", text->name);
    const string& data = authored_ ? text->spec_data_ : text->data_;
    if (data.find_first_of("\n\r<>&\"'") != std::string::npos) {
      XMLText* text_node = elem->GetDocument()->NewText(data.c_str());
      text_node->SetCData(true);
      elem->InsertEndChild(text_node);
    } else {
      WriteAttrTxt(elem, "data", data.c_str());
    }
  }

  // write all tuples
  for (int i = 0; i < ntup; i++) {
    mjCTuple* tuple = (mjCTuple*)model->GetObject(mjOBJ_TUPLE, i);

    elem = InsertEnd(section, "tuple");
    WriteAttrTxt(elem, "name", tuple->name);

    // write objects in tuple
    const auto& objtype = authored_ ? tuple->spec_objtype_ : tuple->objtype_;
    const auto& objname = authored_ ? tuple->spec_objname_ : tuple->objname_;
    const auto& objprm  = authored_ ? tuple->spec_objprm_ : tuple->objprm_;
    for (int j = 0; j < (int)objtype.size(); j++) {
      XMLElement* obj = InsertEnd(elem, "element");
      WriteAttrTxt(obj, "objtype", mju_type2Str((int)objtype[j]));
      WriteAttrTxt(obj, "objname", objname[j].c_str());
      double oprm = j < (int)objprm.size() ? objprm[j] : 0;
      if (oprm != 0) { WriteAttr(obj, "prm", 1, &oprm); }
    }
  }
}


// asset section
void mjXWriter::Asset(XMLElement* root) {
  XMLElement* elem;

  // get sizes
  int                           ntex    = model->NumObjects(mjOBJ_TEXTURE);
  int                           nmat    = model->NumObjects(mjOBJ_MATERIAL);
  const std::vector<mjCMesh*>   meshes  = authored_ ? model->Meshes() : model->CompiledMeshes();
  const std::vector<mjCHField*> hfields = authored_ ? model->HFields() : model->CompiledHFields();
  int                           nmesh   = meshes.size();
  int                           nhfield = hfields.size();

  // return if empty
  if (ntex == 0 && nmat == 0 && nmesh == 0 && nhfield == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "asset");

  // write textures
  for (int i = 0; i < ntex; i++) {
    // create element and write
    mjCTexture* texture = (mjCTexture*)model->GetObject(mjOBJ_TEXTURE, i);

    elem = InsertEnd(section, "texture");
    OneTexture(elem, texture, model->def_map[texture->classname]);
  }

  // write materials
  for (int i = 0; i < nmat; i++) {
    // create element and write
    mjCMaterial* material = (mjCMaterial*)model->GetObject(mjOBJ_MATERIAL, i);

    elem = InsertEnd(section, "material");
    OneMaterial(elem, material, model->def_map[material->classname]);
  }

  // write meshes
  for (int i = 0; i < nmesh; i++) {
    // create element and write
    mjCMesh* mesh = meshes[i];
    elem          = InsertEnd(section, "mesh");
    OneMesh(elem, mesh, model->def_map[mesh->classname]);
    const mjsPlugin& plugin = authored_ ? mesh->spec.plugin : mesh->Plugin();
    if (plugin.active) { OnePlugin(InsertEnd(elem, "plugin"), &plugin); }
  }

  // write hfields
  for (int i = 0; i < nhfield; i++) {
    // create element
    mjCHField*       phfield = hfields[i];
    const mjsHField* hfield  = Values<mjsHField>(phfield);
    const string&    file    = authored_ ? phfield->spec_file_ : phfield->file_;

    elem = InsertEnd(section, "hfield");

    // write attributes
    WriteAttrTxt(elem, "name", phfield->name);
    WriteAttr(elem, "size", 4, hfield->size);
    if (!file.empty()) {
      WriteAttrTxt(elem,
                   "content_type",
                   authored_ ? phfield->spec_content_type_ : phfield->content_type_);
      WriteAttrTxt(elem, "file", file);
    } else {
      int nrow = hfield->nrow;
      int ncol = hfield->ncol;
      WriteAttrInt(elem, "nrow", nrow);
      WriteAttrInt(elem, "ncol", ncol);
      const std::vector<float>& userdata =
          authored_ ? phfield->spec_userdata_ : phfield->get_userdata();
      if (!userdata.empty()) {
        // copy in reverse row order, so XML string is top-to-bottom
        std::vector<float> flipped(nrow * ncol);
        for (int i = 0; i < nrow; i++) {
          int flip = nrow - 1 - i;
          for (int j = 0; j < ncol; j++) { flipped[i * ncol + j] = userdata[flip * ncol + j]; }
        }

        string text;
        Vector2String(text, flipped, ncol);
        WriteAttrTxt(elem, "elevation", text);
      }
    }
  }
}


// strip a frame from a compiled pose: pos/quat become relative to the frame
void mjXWriter::FrameLocal(const mjCFrame* frame, double pos[3], double quat[4]) {
  if (!frame) { return; }
  double ipos[3], iquat[4];
  mjuu_frameinvert(ipos, iquat, frame->pos, frame->quat);
  mjuu_frameaccumChild(ipos, iquat, pos, quat);
}


XMLElement* mjXWriter::OneFrame(XMLElement* elem, mjCFrame* frame, string_view childclass) {
  if (!frame) { return elem; }
  double unitq[4] = {1, 0, 0, 0};

  // what the spec gives: the pose in the frame around it, which is how it was written
  if (authored_) {
    XMLElement* frame_elem = InsertEnd(elem, "frame");
    WriteAttrTxt(frame_elem, "name", frame->name);
    if (!frame->classname.empty() && frame->classname != childclass) {
      WriteAttrTxt(frame_elem, "childclass", frame->classname);
    }
    WriteSpecPose(frame_elem, frame, frame->spec.pos, frame->spec.quat, frame->spec.alt);
    const double* scale   = frame->spec.scale;
    double        unit[3] = {1, 1, 1};
    bool          uniform = scale[0] == scale[1] && scale[1] == scale[2];
    WriteAttr(frame_elem, "scale", uniform ? 1 : 3, scale, unit);
    WriteAttrKey(frame_elem,
                 "scaling",
                 scaling_map,
                 scaling_sz,
                 frame->spec.scaling,
                 mjSCALING_SIMILAR);
    double zero = 0;
    WriteAttr(frame_elem, "massscale", 1, &frame->spec.massscale, &zero);
    return frame_elem;
  }

  // pose relative to the parent frame
  double pos[3], quat[4];
  mjuu_copyvec(pos, frame->pos, 3);
  mjuu_copyvec(quat, frame->quat, 4);
  FrameLocal(frame->frame, pos, quat);

  // omit unnamed identity frame with no childclass
  bool has_class = !frame->classname.empty() && frame->classname != childclass;
  if (frame->name.empty() &&
      !has_class &&
      SameVector(pos, unitq + 1, 3) &&
      SameVector(quat, unitq, 4)) {
    return elem;
  }

  XMLElement* frame_elem = InsertEnd(elem, "frame");
  WriteAttrTxt(frame_elem, "name", frame->name);

  // childclass, unless inherited from the enclosing frame or body
  if (has_class) { WriteAttrTxt(frame_elem, "childclass", frame->classname); }

  WriteAttr(frame_elem, "pos", 3, pos, unitq + 1);
  WriteAttr(frame_elem, "quat", 4, quat, unitq);
  return frame_elem;
}


// turn the element of a free joint into a freejoint, if that says all that the spec gives it: a
// freejoint takes no values from default classes, and only it can say how the joint is aligned
void mjXWriter::FreeJoint(XMLElement* elem, const mjCJoint* joint) {
  if (joint->spec.type != mjJNT_FREE) { return; }

  // what the joint has beyond the built-in defaults
  mjCDef       builtin;
  XMLDocument* doc   = elem->GetDocument();
  XMLElement*  probe = doc->NewElement("joint");
  OneJoint(probe, joint, &builtin, joint->classname);
  bool plain = true;
  for (const tinyxml2::XMLAttribute* attr = probe->FirstAttribute(); attr; attr = attr->Next()) {
    string name = attr->Name();
    plain       = plain && (name == "name" || name == "type" || name == "group");
  }
  doc->DeleteNode(probe);

  if (plain) {
    // only the name and the group stay: the rest are built-in defaults which were written
    // because the class of the joint has other values
    std::vector<string> names;
    for (const tinyxml2::XMLAttribute* attr = elem->FirstAttribute(); attr; attr = attr->Next()) {
      names.push_back(attr->Name());
    }
    for (const string& name : names) {
      if (name != "name" && name != "group") { elem->DeleteAttribute(name.c_str()); }
    }
    elem->SetName("freejoint");
    WriteAttrKey(elem, "align", FalseTrueAuto_map, 3, joint->spec.align, mjALIGNFREE_AUTO);
  } else if (joint->spec.align != mjALIGNFREE_AUTO) {
    string message = "free joint '" +
                     joint->name +
                     "' has an alignment and other attributes, which MJCF cannot express together";
    throw mjXError(0, "%s", message.c_str());
  }
}


// recursive body and frame writer
void mjXWriter::Body(XMLElement* elem, mjCBody* body, mjCFrame* frame, string_view childclass) {
  double unitq[4] = {1, 0, 0, 0};

  // the class which is active around the body or frame
  const string enclosing = childclass.empty() ? "main" : string(childclass);

  if (!body) {
    throw mjXError(0, "missing body in XML write");  // SHOULD NOT OCCUR
  }

  // write body attributes and inertial
  else if (!frame && body != model->GetWorld()) {
    WriteAttrTxt(elem, "name", body->name);
    if (!body->classname.empty() && body->classname != enclosing) {
      WriteAttrTxt(elem, "childclass", body->classname);
    }

    // the pose which the spec gives, or the compiled pose relative to the body's frame
    const mjsBody* values = Values<mjsBody>(body);
    if (authored_) {
      WriteSpecPose(elem, body, values->pos, values->quat, values->alt);
    } else {
      double pos[3], quat[4];
      mjuu_copyvec(pos, body->pos, 3);
      mjuu_copyvec(quat, body->quat, 4);
      FrameLocal(body->frame, pos, quat);
      WriteAttr(elem, "pos", 3, pos, unitq + 1);
      WriteAttr(elem, "quat", 4, quat, unitq);
    }
    if (values->mocap) { WriteAttrKey(elem, "mocap", bool_map, 2, 1); }

    // gravity compensation
    if (values->gravcomp) { WriteAttr(elem, "gravcomp", 1, &values->gravcomp); }

    // sleep policy
    if (values->sleep != mjSLEEP_AUTO &&
        values->sleep != mjSLEEP_AUTO_NEVER &&
        values->sleep != mjSLEEP_AUTO_ALLOWED) {
      WriteAttrKey(elem, "sleep", bodysleep_map, bodysleep_sz, values->sleep);
    }

    // simple optimization
    WriteAttrKey(elem, "simple", FalseAuto_map, 2, values->simple, 1);

    // fuse with parent when static
    WriteAttrKey(elem, "fuse", FalseAuto_map, 2, values->fuse, 1);

    // userdata
    if (authored_) {
      WriteSpecUser(elem, *body->spec.userdata, {});
    } else {
      WriteVector(elem, "user", body->get_userdata());
    }

    // write inertial: the one which the spec gives is written in the frame which it is in;
    // the compiled one where the saved file would not give it to the body otherwise, and when the
    // total mass is set, which scales the masses of all bodies
    if (authored_) {
      if (!body->iframe) { WriteSpecInertial(elem, body); }
    } else if (model->compiler.saveinertial ||
               model->compiler.settotalmass > 0 ||
               !InertialReproduced(body)) {
      XMLElement* inertial = InsertEnd(elem, "inertial");
      WriteAttr(inertial, "pos", 3, body->ipos);
      WriteAttr(inertial, "quat", 4, body->iquat, unitq);
      WriteAttr(inertial, "mass", 1, &body->mass);
      WriteAttr(inertial, "diaginertia", 3, body->inertia);
    }
  }

  // the inertial which the spec gives in this frame
  if (authored_ && frame && body->iframe == frame) { WriteSpecInertial(elem, body); }

  // The elements of the body which are in this frame, and the frames which are in it. Elements
  // of one kind are numbered in the model in the order of their list in the body, which is the
  // order in which they are read: so the elements of each kind are written in that order, those
  // which follow elements inside a frame after that frame.

  // the class which is active for the elements in this frame: its own, or the one around it
  const string& own    = frame ? frame->classname : body->classname;
  const string  active = own.empty() ? enclosing : own;

  // write the elements of a list which are in this frame, from where the last call stopped up to
  // the given index
  size_t njoint = 0, ngeom = 0, nsite = 0, ncamera = 0, nlight = 0, nbody = 0;
  auto   joints = [&](size_t stop) {
    for (; njoint < stop; njoint++) {
      mjCJoint* joint = body->joints[njoint];
      if (joint->frame != frame) { continue; }
      XMLElement* joint_elem = InsertEnd(elem, "joint");
      OneJoint(joint_elem, joint, model->def_map[joint->classname], active);
      if (authored_) { FreeJoint(joint_elem, joint); }
    }
  };
  auto geoms = [&](size_t stop) {
    for (; ngeom < stop; ngeom++) {
      mjCGeom* geom = body->geoms[ngeom];
      if (geom->frame != frame) { continue; }
      OneGeom(InsertEnd(elem, "geom"), geom, model->def_map[geom->classname], active);
    }
  };
  auto sites = [&](size_t stop) {
    for (; nsite < stop; nsite++) {
      mjCSite* site = body->sites[nsite];
      if (site->frame != frame) { continue; }
      OneSite(InsertEnd(elem, "site"), site, model->def_map[site->classname], active);
    }
  };
  auto cameras = [&](size_t stop) {
    for (; ncamera < stop; ncamera++) {
      mjCCamera* camera = body->cameras[ncamera];
      if (camera->frame != frame) { continue; }
      OneCamera(InsertEnd(elem, "camera"), camera, model->def_map[camera->classname], active);
    }
  };
  auto lights = [&](size_t stop) {
    for (; nlight < stop; nlight++) {
      mjCLight* light = body->lights[nlight];
      if (light->frame != frame) { continue; }
      OneLight(InsertEnd(elem, "light"), light, model->def_map[light->classname], active);
    }
  };
  auto bodies = [&](size_t stop) {
    for (; nbody < stop; nbody++) {
      mjCBody* child = body->bodies[nbody];
      if (child->frame != frame) { continue; }
      Body(InsertEnd(elem, "body"), child, nullptr, active);
    }
  };

  // the frames in this frame, and the position among them of the one which a frame is or is
  // inside; -1 if it is none of them
  std::vector<mjCFrame*> inner;
  for (mjCFrame* child : body->frames) {
    if (child->frame == frame) { inner.push_back(child); }
  }
  auto position = [&](const mjCFrame* at) -> int {
    while (at && at->frame != frame) { at = at->frame; }
    auto found = std::find(inner.begin(), inner.end(), at);
    return found == inner.end() ? -1 : static_cast<int>(found - inner.begin());
  };

  // index of the first element of a list, from the given one, which is inside one of the frames
  // from the given position on; the size of the list if there is none
  auto first = [&](const auto& list, size_t from, int from_position) -> size_t {
    for (size_t k = from; k < list.size(); k++) {
      if (list[k]->frame != frame && position(list[k]->frame) >= from_position) { return k; }
    }
    return list.size();
  };

  // each frame after the elements which precede those inside it and inside the frames after it
  for (int i = 0; i < inner.size(); i++) {
    joints(first(body->joints, njoint, i));
    geoms(first(body->geoms, ngeom, i));
    sites(first(body->sites, nsite, i));
    cameras(first(body->cameras, ncamera, i));
    lights(first(body->lights, nlight, i));
    bodies(first(body->bodies, nbody, i));
    Body(OneFrame(elem, inner[i], active), body, inner[i], active);
  }

  // the elements which follow the last frame
  joints(body->joints.size());
  geoms(body->geoms.size());
  sites(body->sites.size());
  cameras(body->cameras.size());
  lights(body->lights.size());

  // write plugin, once: not again inside the body's frames
  const mjsPlugin& plugin = authored_ ? body->spec.plugin : body->plugin;
  if (!frame && plugin.active) { OnePlugin(InsertEnd(elem, "plugin"), &plugin); }

  bodies(body->bodies.size());
}


// collision section
void mjXWriter::Contact(XMLElement* root) {
  XMLElement* elem;

  // get number of pairs of each type
  int npair    = model->NumObjects(mjOBJ_PAIR);
  int nexclude = model->NumObjects(mjOBJ_EXCLUDE);

  // skip if section is empty
  if (npair == 0 && nexclude == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "contact");

  // write all geom pairs
  for (int i = 0; i < npair; i++) {
    // create element and write
    mjCPair* pair = (mjCPair*)model->GetObject(mjOBJ_PAIR, i);

    elem = InsertEnd(section, "pair");
    OnePair(elem, pair, model->def_map[pair->classname]);
  }

  // write all exclude pairs
  for (int i = 0; i < nexclude; i++) {
    // create element
    mjCBodyPair* exclude = (mjCBodyPair*)model->GetObject(mjOBJ_EXCLUDE, i);

    elem = InsertEnd(section, "exclude");

    // write attributes
    WriteAttrTxt(elem, "name", exclude->name);
    WriteAttrTxt(elem, "body1", Pick(authored_, exclude->spec.bodyname1, exclude->get_bodyname1()));
    WriteAttrTxt(elem, "body2", Pick(authored_, exclude->spec.bodyname2, exclude->get_bodyname2()));
  }
}


// constraint section
void mjXWriter::Equality(XMLElement* root) {
  // skip section if empty
  int num;
  if ((num = model->NumObjects(mjOBJ_EQUALITY)) == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "equality");

  // write all constraints
  for (int i = 0; i < num; i++) {
    mjCEquality* equality = (mjCEquality*)model->GetObject(mjOBJ_EQUALITY, i);
    XMLElement*  elem     = InsertEnd(
        section,
        FindValue(equality_map, equality_sz, Values<mjsEquality>(equality)->type).c_str());
    OneEquality(elem, equality, model->def_map[equality->classname]);
  }
}


// deformable section
void mjXWriter::Deformable(XMLElement* root) {
  XMLElement* elem;

  // get sizes
  int nflex = model->NumObjects(mjOBJ_FLEX);
  int nskin = model->NumObjects(mjOBJ_SKIN);

  // return if empty
  if (nflex == 0 && nskin == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "deformable");

  // write flexes
  for (int i = 0; i < nflex; i++) {
    // create element and write
    mjCFlex* flex = (mjCFlex*)model->GetObject(mjOBJ_FLEX, i);
    elem          = InsertEnd(section, "flex");
    OneFlex(elem, flex);
  }

  // write skins
  for (int i = 0; i < nskin; i++) {
    // create element and write
    mjCSkin* skin = (mjCSkin*)model->GetObject(mjOBJ_SKIN, i);
    elem          = InsertEnd(section, "skin");
    OneSkin(elem, skin);
  }
}


// tendon section
void mjXWriter::Tendon(XMLElement* root) {
  // skip section if empty
  int num;
  if ((num = model->NumObjects(mjOBJ_TENDON)) == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "tendon");

  // write all tendons
  for (int i = 0; i < num; i++) {
    // write tendon element and attributes
    mjCTendon* tendon = (mjCTendon*)model->GetObject(mjOBJ_TENDON, i);
    if (!tendon->NumWraps()) {  // SHOULD NOT OCCUR
      continue;
    }
    XMLElement* elem = InsertEnd(section, tendon->Type() == mjTENDON_FIXED ? "fixed" : "spatial");
    OneTendon(elem, tendon, model->def_map[tendon->classname]);

    // write wraps
    XMLElement* wrapelem;
    for (int j = 0; j < tendon->NumWraps(); j++) {
      const mjCWrap* wrap = tendon->GetWrap(j);
      switch (wrap->Type()) {
        case mjWRAP_JOINT: {
          wrapelem = InsertEnd(elem, "joint");
          WriteAttrTxt(wrapelem, "joint", authored_ ? wrap->name : wrap->obj->name);
          double coef = authored_ ? wrap->prm : wrap->prm * wrap->prmscale;
          WriteAttr(wrapelem, "coef", 1, &coef);
          break;
        }

        case mjWRAP_SITE:
          wrapelem = InsertEnd(elem, "site");
          WriteAttrTxt(wrapelem, "site", authored_ ? wrap->name : wrap->obj->name);
          break;

        case mjWRAP_SPHERE:
        case mjWRAP_CYLINDER:
          wrapelem = InsertEnd(elem, "geom");
          WriteAttrTxt(wrapelem, "geom", authored_ ? wrap->name : wrap->obj->name);
          if (!wrap->sidesite.empty()) { WriteAttrTxt(wrapelem, "sidesite", wrap->sidesite); }
          break;

        case mjWRAP_PULLEY:
          wrapelem = InsertEnd(elem, "pulley");
          WriteAttr(wrapelem, "divisor", 1, &wrap->prm);
          break;

        default:
          break;
      }
    }
  }
}


// actuator section
void mjXWriter::Actuator(XMLElement* root) {
  // skip section if empty
  int num;
  if ((num = model->NumObjects(mjOBJ_ACTUATOR)) == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "actuator");

  // write all actuators
  for (int i = 0; i < num; i++) {
    mjCActuator* actuator = (mjCActuator*)model->GetObject(mjOBJ_ACTUATOR, i);
    OneActuator(section, actuator, model->def_map[actuator->classname]);
  }
}


// sensor section
void mjXWriter::Sensor(XMLElement* root) {
  double zero = 0;

  // skip section if empty
  int num;
  if ((num = model->NumObjects(mjOBJ_SENSOR)) == 0) { return; }

  // create section
  XMLElement* section = InsertEnd(root, "sensor");

  // write all sensors
  for (int i = 0; i < num; i++) {
    XMLElement*      elem          = 0;
    mjCSensor*       base          = model->Sensors()[i];
    const mjsSensor* sensor        = Values<mjsSensor>(base);
    string           instance_name = "";
    string           plugin_name   = "";

    // write sensor type and type-specific attributes
    switch (sensor->type) {
      // common robotic sensors, attached to a site
      case mjSENS_TOUCH:
        elem = InsertEnd(section, "touch");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_ACCELEROMETER:
        elem = InsertEnd(section, "accelerometer");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_VELOCIMETER:
        elem = InsertEnd(section, "velocimeter");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_GYRO:
        elem = InsertEnd(section, "gyro");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_FORCE:
        elem = InsertEnd(section, "force");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_TORQUE:
        elem = InsertEnd(section, "torque");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_MAGNETOMETER:
        elem = InsertEnd(section, "magnetometer");
        WriteAttrTxt(elem, "site", base->get_objname());
        break;
      case mjSENS_RANGEFINDER: {
        elem = InsertEnd(section, "rangefinder");
        if (sensor->objtype == mjOBJ_SITE) {
          WriteAttrTxt(elem, "site", base->get_objname());
        } else {
          WriteAttrTxt(elem, "camera", base->get_objname());
        }
        int dataspec = sensor->intprm[0];
        int data[mjNRAYDATA];
        int ndata = 0;
        for (int i = 0; i < mjNRAYDATA; i++) {
          if (dataspec & (1 << i)) { data[ndata++] = i; }
        }
        WriteAttrKeys(elem, "data", raydata_map, mjNRAYDATA, data, ndata, 0);
      } break;
      case mjSENS_CAMPROJECTION:
        elem = InsertEnd(section, "camprojection");
        WriteAttrTxt(elem, "site", base->get_objname());
        WriteAttrTxt(elem, "camera", base->get_refname());
        break;

      // sensors related to scalar joints, tendons, actuators
      case mjSENS_JOINTPOS:
        elem = InsertEnd(section, "jointpos");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_JOINTVEL:
        elem = InsertEnd(section, "jointvel");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_TENDONPOS:
        elem = InsertEnd(section, "tendonpos");
        WriteAttrTxt(elem, "tendon", base->get_objname());
        break;
      case mjSENS_TENDONVEL:
        elem = InsertEnd(section, "tendonvel");
        WriteAttrTxt(elem, "tendon", base->get_objname());
        break;
      case mjSENS_ACTUATORPOS:
        elem = InsertEnd(section, "actuatorpos");
        WriteAttrTxt(elem, "actuator", base->get_objname());
        break;
      case mjSENS_ACTUATORVEL:
        elem = InsertEnd(section, "actuatorvel");
        WriteAttrTxt(elem, "actuator", base->get_objname());
        break;
      case mjSENS_ACTUATORFRC:
        elem = InsertEnd(section, "actuatorfrc");
        WriteAttrTxt(elem, "actuator", base->get_objname());
        break;
      case mjSENS_JOINTACTFRC:
        elem = InsertEnd(section, "jointactuatorfrc");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_TENDONACTFRC:
        elem = InsertEnd(section, "tendonactuatorfrc");
        WriteAttrTxt(elem, "tendon", base->get_objname());
        break;

      // sensors related to ball joints
      case mjSENS_BALLQUAT:
        elem = InsertEnd(section, "ballquat");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_BALLANGVEL:
        elem = InsertEnd(section, "ballangvel");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;

      // joint and tendon limit sensors
      case mjSENS_JOINTLIMITPOS:
        elem = InsertEnd(section, "jointlimitpos");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_JOINTLIMITVEL:
        elem = InsertEnd(section, "jointlimitvel");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_JOINTLIMITFRC:
        elem = InsertEnd(section, "jointlimitfrc");
        WriteAttrTxt(elem, "joint", base->get_objname());
        break;
      case mjSENS_TENDONLIMITPOS:
        elem = InsertEnd(section, "tendonlimitpos");
        WriteAttrTxt(elem, "tendon", base->get_objname());
        break;
      case mjSENS_TENDONLIMITVEL:
        elem = InsertEnd(section, "tendonlimitvel");
        WriteAttrTxt(elem, "tendon", base->get_objname());
        break;
      case mjSENS_TENDONLIMITFRC:
        elem = InsertEnd(section, "tendonlimitfrc");
        WriteAttrTxt(elem, "tendon", base->get_objname());
        break;

      // sensors attached to an object with spatial frame: (x)body, geom, site, camera
      case mjSENS_FRAMEPOS:
        elem = InsertEnd(section, "framepos");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMEQUAT:
        elem = InsertEnd(section, "framequat");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMEXAXIS:
        elem = InsertEnd(section, "framexaxis");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMEYAXIS:
        elem = InsertEnd(section, "frameyaxis");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMEZAXIS:
        elem = InsertEnd(section, "framezaxis");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMELINVEL:
        elem = InsertEnd(section, "framelinvel");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMEANGVEL:
        elem = InsertEnd(section, "frameangvel");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMELINACC:
        elem = InsertEnd(section, "framelinacc");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;
      case mjSENS_FRAMEANGACC:
        elem = InsertEnd(section, "frameangacc");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        break;

      // sensors related to kinematic subtrees; attached to a body (which is the subtree root)
      case mjSENS_SUBTREECOM:
        elem = InsertEnd(section, "subtreecom");
        WriteAttrTxt(elem, "body", base->get_objname());
        break;
      case mjSENS_SUBTREELINVEL:
        elem = InsertEnd(section, "subtreelinvel");
        WriteAttrTxt(elem, "body", base->get_objname());
        break;
      case mjSENS_SUBTREEANGMOM:
        elem = InsertEnd(section, "subtreeangmom");
        WriteAttrTxt(elem, "body", base->get_objname());
        break;
      case mjSENS_INSIDESITE:
        elem = InsertEnd(section, "insidesite");
        WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        WriteAttrTxt(elem, "objname", base->get_objname());
        WriteAttrTxt(elem, "site", base->get_refname());
        if (sensor->intprm[0]) { WriteAttrTxt(elem, "enclosed", "true"); }
        break;
      case mjSENS_GEOMDIST:
        elem = InsertEnd(section, "distance");
        WriteAttrTxt(elem, sensor->objtype == mjOBJ_BODY ? "body1" : "geom1", base->get_objname());
        WriteAttrTxt(elem, sensor->reftype == mjOBJ_BODY ? "body2" : "geom2", base->get_refname());
        break;
      case mjSENS_GEOMNORMAL:
        elem = InsertEnd(section, "normal");
        WriteAttrTxt(elem, sensor->objtype == mjOBJ_BODY ? "body1" : "geom1", base->get_objname());
        WriteAttrTxt(elem, sensor->reftype == mjOBJ_BODY ? "body2" : "geom2", base->get_refname());
        break;
      case mjSENS_GEOMFROMTO:
        elem = InsertEnd(section, "fromto");
        WriteAttrTxt(elem, sensor->objtype == mjOBJ_BODY ? "body1" : "geom1", base->get_objname());
        WriteAttrTxt(elem, sensor->reftype == mjOBJ_BODY ? "body2" : "geom2", base->get_refname());
        break;
      case mjSENS_CONTACT: {
        elem = InsertEnd(section, "contact");
        if (sensor->objtype == mjOBJ_BODY) {
          WriteAttrTxt(elem, "body1", base->get_objname());
        } else if (sensor->objtype == mjOBJ_XBODY) {
          WriteAttrTxt(elem, "subtree1", base->get_objname());
        } else if (sensor->objtype == mjOBJ_GEOM) {
          WriteAttrTxt(elem, "geom1", base->get_objname());
        } else if (sensor->objtype == mjOBJ_SITE) {
          WriteAttrTxt(elem, "site", base->get_objname());
        }
        if (sensor->reftype == mjOBJ_BODY) {
          WriteAttrTxt(elem, "body2", base->get_refname());
        } else if (sensor->reftype == mjOBJ_XBODY) {
          WriteAttrTxt(elem, "subtree2", base->get_refname());
        } else if (sensor->reftype == mjOBJ_GEOM) {
          WriteAttrTxt(elem, "geom2", base->get_refname());
        }
        int dataspec = sensor->intprm[0];
        WriteAttrInt(elem,
                     "num",
                     authored_ ? sensor->intprm[2] : sensor->dim / mju_condataSize(dataspec),
                     1);
        int data[mjNCONDATA];
        int ndata = 0;
        for (int i = 0; i < mjNCONDATA; i++) {
          if (dataspec & (1 << i)) { data[ndata++] = i; }
        }
        WriteAttrKeys(elem, "data", condata_map, mjNCONDATA, data, ndata, 0);
        WriteAttrKey(elem, "reduce", reduce_map, reduce_sz, sensor->intprm[1], 0);
      } break;
      case mjSENS_TACTILE:
        elem = InsertEnd(section, "tactile");
        WriteAttrTxt(elem, "geom", base->get_refname());
        WriteAttrTxt(elem, "mesh", base->get_objname());
        break;
      // global sensors
      case mjSENS_E_POTENTIAL:
        elem = InsertEnd(section, "e_potential");
        break;
      case mjSENS_E_KINETIC:
        elem = InsertEnd(section, "e_kinetic");
        break;
      case mjSENS_CLOCK:
        elem = InsertEnd(section, "clock");
        break;


      // plugin-controlled sensor
      case mjSENS_PLUGIN:
        elem = InsertEnd(section, "plugin");
        if (sensor->objtype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
          WriteAttrTxt(elem, "objname", base->get_objname());
        }
        if (sensor->reftype != mjOBJ_UNKNOWN) {
          WriteAttrTxt(elem, "reftype", mju_type2Str(sensor->reftype));
          WriteAttrTxt(elem, "refname", base->get_refname());
        }
        OnePlugin(elem, &sensor->plugin);
        break;

      // user-defined sensor
      case mjSENS_USER:
        elem = InsertEnd(section, "user");
        if (mju_type2Str(sensor->objtype)) {
          WriteAttrTxt(elem, "objtype", mju_type2Str(sensor->objtype));
        }
        WriteAttrTxt(elem, "objname", base->get_objname());
        WriteAttrInt(elem, "dim", sensor->dim);
        WriteAttrKey(elem, "needstage", stage_map, stage_sz, (int)sensor->needstage);
        WriteAttrKey(elem, "datatype", datatype_map, datatype_sz, (int)sensor->datatype);
        break;

      default:
        throw mjXError(0, "unknown sensor type");
    }

    // write name, noise, userdata
    WriteAttrTxt(elem, "name", base->name);
    WriteAttr(elem, "cutoff", 1, &sensor->cutoff, &zero);
    if (sensor->type != mjSENS_PLUGIN && sensor->type != mjSENS_TACTILE) {
      WriteAttr(elem, "noise", 1, &sensor->noise, &zero);
    }
    WriteAttrInt(elem, "nsample", sensor->nsample, 0);
    WriteAttrKey(elem, "interp", interp_map, interp_sz, sensor->interp, 0);
    WriteAttr(elem, "delay", 1, &sensor->delay, &zero);
    double zeros[2] = {0, 0};
    WriteAttr(elem, "interval", 2, sensor->interval, zeros);
    if (authored_) {
      WriteSpecUser(elem, *base->spec.userdata, {});
    } else {
      WriteVector(elem, "user", base->get_userdata());
    }
  }

  // remove section if empty
  if (!section->FirstChildElement()) { root->DeleteChild(section); }
}


// keyframe section
void mjXWriter::Keyframe(XMLElement* root) {
  // create section
  XMLElement* section = InsertEnd(root, "keyframe");

  if (model->HasPendingKeys()) {
    throw mjXError(0, "Model has pending keyframes. It must be (re)compiled before writing XML.");
  }

  // what the spec gives: the vectors which a keyframe has, each as it is. None is left out for
  // being what the model has without it: the spec may give the model another reference
  // configuration than the last compilation saw, and the vector still says what it said. A
  // keyframe which sets nothing is kept: it counts
  if (authored_) {
    for (mjCKey* key : model->Keys()) {
      XMLElement* elem = InsertEnd(section, "key");
      WriteAttrTxt(elem, "name", key->name);
      if (key->spec.time != 0) { WriteAttr(elem, "time", 1, &key->spec.time); }
      const std::vector<double>* vectors[7] = {&key->spec_qpos_,
                                               &key->spec_qvel_,
                                               &key->spec_act_,
                                               &key->spec_mpos_,
                                               &key->spec_mquat_,
                                               &key->spec_ctrl_,
                                               &key->spec_plastic_};
      const char* names[7] = {"qpos", "qvel", "act", "mpos", "mquat", "ctrl", "plastic"};
      for (int k = 0; k < 7; k++) {
        if (!vectors[k]->empty()) {
          WriteAttr(elem, names[k], vectors[k]->size(), vectors[k]->data());
        }
      }
    }

    // remove section if empty
    if (!section->FirstChildElement()) { root->DeleteChild(section); }
    return;
  }

  // write all keyframes
  for (int i = 0; i < model->nkey; i++) {
    XMLElement* elem   = InsertEnd(section, "key");
    bool        change = false;

    mjCKey* key = model->Keys()[i];

    // check name and write
    if (!key->name.empty()) {
      WriteAttrTxt(elem, "name", key->name);
      change = true;
    }

    // check time and write
    if (key->time != 0) {
      WriteAttr(elem, "time", 1, &key->time);
      change = true;
    }

    // check qpos and write
    for (int j = 0; j < model->nq; j++) {
      if (key->qpos_[j] != model->qpos0[j]) {
        WriteAttr(elem, "qpos", model->nq, key->qpos_.data());
        change = true;
        break;
      }
    }

    // check qvel and write
    for (int j = 0; j < model->nv; j++) {
      if (key->qvel_[j] != 0) {
        WriteAttr(elem, "qvel", model->nv, key->qvel_.data());
        change = true;
        break;
      }
    }

    // check act and write
    for (int j = 0; j < model->na; j++) {
      if (key->act_[j] != 0) {
        WriteAttr(elem, "act", model->na, key->act_.data());
        change = true;
        break;
      }
    }

    // check mpos and write
    if (model->nmocap) {
      for (int j = 0; j < model->nbody; j++) {
        if (model->Bodies()[j]->mocap) {
          mjCBody* body = model->Bodies()[j];
          int      id   = body->mocapid;
          if (body->pos[0] != key->mpos_[3 * id] ||
              body->pos[1] != key->mpos_[3 * id + 1] ||
              body->pos[2] != key->mpos_[3 * id + 2]) {
            WriteAttr(elem, "mpos", 3 * model->nmocap, key->mpos_.data());
            change = true;
            break;
          }
        }
      }
    }

    // check mquat and write
    if (model->nmocap) {
      for (int j = 0; j < model->nbody; j++) {
        if (model->Bodies()[j]->mocap) {
          mjCBody* body = model->Bodies()[j];
          int      id   = body->mocapid;
          if (body->quat[0] != key->mquat_[4 * id] ||
              body->quat[1] != key->mquat_[4 * id + 1] ||
              body->quat[2] != key->mquat_[4 * id + 2] ||
              body->quat[3] != key->mquat_[4 * id + 3]) {
            WriteAttr(elem, "mquat", 4 * model->nmocap, key->mquat_.data());
            change = true;
            break;
          }
        }
      }
    }

    // check ctrl and write
    for (int j = 0; j < model->nu; j++) {
      if (key->ctrl_[j] != 0) {
        WriteAttr(elem, "ctrl", model->nu, key->ctrl_.data());
        change = true;
        break;
      }
    }

    // check plastic and write
    for (int j = 0; j < model->nplastic; j++) {
      if (key->plastic_[j] != model->plastic0[j]) {
        WriteAttr(elem, "plastic", model->nplastic, key->plastic_.data());
        change = true;
        break;
      }
    }

    // remove elem if empty
    if (!change) { section->DeleteChild(elem); }
  }

  // remove section if empty
  if (!section->FirstChildElement()) { root->DeleteChild(section); }
}
