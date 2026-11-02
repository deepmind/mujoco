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

#include "xml/xml_native_reader.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <mujoco/mjmodel.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "engine/engine_util_misc.h"
#include "user/user_api.h"
#include "user/user_flexcomp.h"
#include "user/user_objects.h"
#include "user/user_util.h"
#include "xml/xml_base.h"
#include "xml/xml_util.h"
#include "tinyxml2.h"

namespace {
using mujoco::user::FilePath;
using std::string;
using std::string_view;
using std::vector;
using tinyxml2::XMLElement;
using tinyxml2::XMLText;

//---------------------------------- helper utilities ----------------------------------------------

// GetAttrPtr: overload for scalar and pointer fields
template <typename T>
inline auto GetAttrPtr(T& val) -> std::enable_if_t<!std::is_array_v<T>, decltype(&val)> {
  if constexpr (std::is_pointer_v<T>) {
    return val;
  } else {
    return &val;
  }
}

// GetAttrPtr: overload for array fields
template <typename T, size_t N>
inline T* GetAttrPtr(T (&arr)[N]) {
  return arr;
}


// helper class for reading attributes while recording authored bits
struct Reader {
  Reader(XMLElement* xml_node, const void* elem) : xml_node_(xml_node), elem_(elem) {}

  template <typename T>
  int operator()(const char* attr, int len, T& data, bool required = false, bool exact = true) {
    int res = mjXReader::ReadAttr(xml_node_, attr, len, GetAttrPtr(data), text_, required, exact);
    if (res) mjs_setAuthored(elem_, &data, 1);
    return res;
  }

  template <typename T>
  bool operator()(const char* attr, T& data, const mjMap* map, int mapsz, bool required = false) {
    int  map_val_temp;
    bool res = mjXReader::MapValue(xml_node_, attr, &map_val_temp, map, mapsz, required);
    if (res) {
      data = static_cast<T>(map_val_temp);
      mjs_setAuthored(elem_, &data, 1);
    }
    return res;
  }

  bool operator()(const char* attr, int& data, bool required = false) {
    bool res = mjXUtil::ReadAttrInt(xml_node_, attr, &data, required);
    if (res) mjs_setAuthored(elem_, &data, 1);
    return res;
  }

  bool operator()(const char* attr, mjString* target) {
    std::string txt_temp;
    bool        res = mjXUtil::ReadAttrTxt(xml_node_, attr, txt_temp);
    if (res) {
      mjs_setString(target, txt_temp.c_str());
      mjs_setAuthored(elem_, target, 1);
    }
    return res;
  }

  template <typename T>
  bool txt(const char* attr, T& target, void (&set_func)(T&, const char*)) {
    std::string txt_temp;
    bool        res = mjXUtil::ReadAttrTxt(xml_node_, attr, txt_temp);
    if (res) {
      set_func(target, txt_temp.c_str());
      mjs_setAuthored(elem_, &target, 1);
    }
    return res;
  }

  void set_node(XMLElement* node) { xml_node_ = node; }

  XMLElement* xml_node_;
  const void* elem_;
  std::string text_;
};

void ReadPluginConfigs(tinyxml2::XMLElement* elem, mjsPlugin* p) {
  std::map<string, string, std::less<> > config_attribs;

  XMLElement* child = FirstChildElement(elem);
  while (child) {
    string_view name = child->Value();
    if (name == "config") {
      string key, value;
      mjXUtil::ReadAttrTxt(child, "key", key, /* required = */ true);
      if (config_attribs.find(key) != config_attribs.end()) {
        string err = "duplicate config key: " + key;
        throw mjXError(child, "%s", err.c_str());
      }
      mjXUtil::ReadAttrTxt(child, "value", value, /* required = */ true);
      config_attribs[key] = value;
    }
    child = NextSiblingElement(child);
  }

  if (!p && !config_attribs.empty()) {
    throw mjXError(elem,
                   "plugin configuration attributes cannot be used in an "
                   "element that references a predefined plugin instance");
  } else if (p) {
    mjs_setPluginAttributes(p, &config_attribs);
  }
}

static void UpdateString(string& psuffix, int count, int i) {
  int    ndigits  = std::to_string(count).length();
  string i_string = std::to_string(i);
  string prefix   = "";
  while (ndigits-- > i_string.length()) { prefix += '0'; }
  psuffix += prefix + i_string;
}
}  // namespace


//---------------------------------- MJCF schema ---------------------------------------------------
#include "xml/generated/mjcf_table.inc"

//---------------------------------- class mjXReader implementation --------------------------------

// typed attribute rows, generated from mjcf.schema; the keyword maps the
// rows reference are generated into mjcf_map.h
#include "xml/generated/mjcf_read_table.inc"


mjXReader::mjXReader() : schema(MJCF, nMJCF, MJCF_constraints, nMJCF_constraints) {
  readingdefaults = false;
}


// print schema
void mjXReader::PrintSchema(std::stringstream& str, bool html, bool pad) {
  if (html) {
    schema.PrintHTML(str, 0, pad);
  } else {
    schema.Print(str, 0);
  }
}


// main entry point for XML parser
//  mjCModel is allocated here; caller is responsible for deallocation
void mjXReader::Parse(XMLElement* root, const mjVFS* vfs) {
  // check schema
  if (!schema.GetError().empty()) {
    throw mjXError(0, "XML Schema Construction Error: %s", schema.GetError().c_str());
  }

  // validate
  XMLElement* bad = 0;
  if ((bad = schema.Check(root, 0))) {
    throw mjXError(bad, "Schema violation: %s", schema.GetError().c_str());
  }

  // get model name
  string modelname;
  if (ReadAttrTxt(root, "model", modelname)) { mjs_setString(spec->modelname, modelname.c_str()); }

  // get comment
  if (root->FirstChild() && root->FirstChild()->ToComment()) {
    mjs_setString(spec->comment, root->FirstChild()->Value());
  } else {
    mjs_setString(spec->comment, "");
  }

  //------------------- parse MuJoCo sections embedded in all XML formats

  for (XMLElement* section = FirstChildElement(root, "compiler"); section;
       section             = NextSiblingElement(section, "compiler")) {
    Compiler(section, spec);
  }

  for (XMLElement* section = FirstChildElement(root, "option"); section;
       section             = NextSiblingElement(section, "option")) {
    Option(section, spec, &spec->option);
  }

  for (XMLElement* section = FirstChildElement(root, "size"); section;
       section             = NextSiblingElement(section, "size")) {
    Size(section, spec);
  }

  //------------------ parse MJCF-specific sections

  for (XMLElement* section = FirstChildElement(root, "visual"); section;
       section             = NextSiblingElement(section, "visual")) {
    Visual(section);
  }

  for (XMLElement* section = FirstChildElement(root, "statistic"); section;
       section             = NextSiblingElement(section, "statistic")) {
    Statistic(section);
  }

  readingdefaults = true;
  for (XMLElement* section = FirstChildElement(root, "default"); section;
       section             = NextSiblingElement(section, "default")) {
    Default(section, nullptr, vfs);
  }
  readingdefaults = false;

  for (XMLElement* section = FirstChildElement(root, "extension"); section;
       section             = NextSiblingElement(section, "extension")) {
    Extension(section);
  }

  for (XMLElement* section = FirstChildElement(root, "custom"); section;
       section             = NextSiblingElement(section, "custom")) {
    Custom(section);
  }

  for (XMLElement* section = FirstChildElement(root, "asset"); section;
       section             = NextSiblingElement(section, "asset")) {
    Asset(section, vfs);
  }

  for (XMLElement* section = FirstChildElement(root, "contact"); section;
       section             = NextSiblingElement(section, "contact")) {
    Contact(section);
  }

  for (XMLElement* section = FirstChildElement(root, "deformable"); section;
       section             = NextSiblingElement(section, "deformable")) {
    Deformable(section, vfs);
  }

  for (XMLElement* section = FirstChildElement(root, "equality"); section;
       section             = NextSiblingElement(section, "equality")) {
    Equality(section);
  }

  for (XMLElement* section = FirstChildElement(root, "tendon"); section;
       section             = NextSiblingElement(section, "tendon")) {
    Tendon(section);
  }

  for (XMLElement* section = FirstChildElement(root, "actuator"); section;
       section             = NextSiblingElement(section, "actuator")) {
    Actuator(section);
  }

  for (XMLElement* section = FirstChildElement(root, "sensor"); section;
       section             = NextSiblingElement(section, "sensor")) {
    Sensor(section);
  }

  // set deepcopy flag to true to copy child specs during attach calls
  mjs_setDeepCopy(spec, true);

  for (XMLElement* section = FirstChildElement(root, "worldbody"); section;
       section             = NextSiblingElement(section, "worldbody")) {
    Body(section, mjs_findBody(spec, "world"), nullptr, vfs);
  }

  // set deepcopy flag to false to disable copying during attach in all future calls
  mjs_setDeepCopy(spec, false);

  // keyframes describe the complete model, so they are parsed after the kinematic tree: the
  // attachments and deletions which expand a replicate would store and copy them as they go
  for (XMLElement* section = FirstChildElement(root, "keyframe"); section;
       section             = NextSiblingElement(section, "keyframe")) {
    Keyframe(section);
  }
}


// compiler section parser
void mjXReader::Compiler(XMLElement* section, mjSpec* s) {
  Reader read(section, s);

  // strippath is stored on the spec, not the compiler options
  read("strippath", s->strippath, bool_map, 2);

  // global coordinates are no longer supported
  if (int n = 0; MapValue(section, "coordinate", &n, coordinate_map, 2)) {
    if (n == 1) {
      throw mjXError(section,
                     "global coordinates no longer supported. To convert existing models, "
                     "load and save them in MuJoCo 2.3.3 or older");
    }
  }
  if (ReadAttrTxt(section, "assetdir", read.text_)) {
    mjs_setString(s->compiler.meshdir, read.text_.c_str());
    mjs_setString(s->compiler.texturedir, read.text_.c_str());
  }
  // mechanical attributes; meshdir/texturedir override assetdir above
  ReadAttrTableCore(section,
                    &s->compiler,
                    kCompilerAttrs,
                    kCompilerAttrsN,
                    /*skipnodefault=*/false,
                    /*authored=*/s);

  // lengthrange subelement
  XMLElement* elem = FindSubElem(section, "lengthrange");
  if (elem) {
    ReadAttrTableCore(elem,
                      &(s->compiler.LRopt),
                      kLengthrangeAttrs,
                      kLengthrangeAttrsN,
                      /*skipnodefault=*/false);
  }
}


// option section parser
void mjXReader::Option(XMLElement* section, mjSpec* s, mjOption* opt) {
  Reader read(section, s);

  // mechanical attributes
  ReadAttrTableCore(section,
                    opt,
                    kOptionAttrs,
                    kOptionAttrsN,
                    /*skipnodefault=*/false,
                    /*authored=*/s);

  // actuatorgroupdisable
  constexpr int num_bitflags = 31;
  int           disabled_act_groups[num_bitflags];
  int num_found = read("actuatorgroupdisable", num_bitflags, disabled_act_groups, false, false);
  for (int i = 0; i < num_found; i++) {
    int group = disabled_act_groups[i];
    if (group < 0) {
      throw mjXError(section, "disabled actuator group value must be non-negative");
    }
    if (group > num_bitflags - 1) {
      throw mjXError(section, "disabled actuator group value cannot exceed 30");
    }
    opt->disableactuator        |= (1 << group);
    s->authored.disableactuator |= (1 << group);
  }

  // read disable sub-element
  XMLElement* elem = FindSubElem(section, "flag");
  if (elem) {
    int n = 0;
#define READDSBL(NAME, MASK)                                \
  if (MapValue(elem, NAME, &n, enable_map, 2)) {            \
    opt->disableflags        ^= (opt->disableflags & MASK); \
    opt->disableflags        |= (n ? 0 : MASK);             \
    s->authored.disableflags |= MASK;                       \
  }
    // clang-format off
    READDSBL("constraint",   mjDSBL_CONSTRAINT)
    READDSBL("equality",     mjDSBL_EQUALITY)
    READDSBL("frictionloss", mjDSBL_FRICTIONLOSS)
    READDSBL("limit",        mjDSBL_LIMIT)
    READDSBL("contact",      mjDSBL_CONTACT)
    READDSBL("spring",       mjDSBL_SPRING)
    READDSBL("damper",       mjDSBL_DAMPER)
    READDSBL("gravity",      mjDSBL_GRAVITY)
    READDSBL("clampctrl",    mjDSBL_CLAMPCTRL)
    READDSBL("warmstart",    mjDSBL_WARMSTART)
    READDSBL("filterparent", mjDSBL_FILTERPARENT)
    READDSBL("actuation",    mjDSBL_ACTUATION)
    READDSBL("refsafe",      mjDSBL_REFSAFE)
    READDSBL("sensor",       mjDSBL_SENSOR)
    READDSBL("midphase",     mjDSBL_MIDPHASE)
    READDSBL("eulerdamp",    mjDSBL_EULERDAMP)
    READDSBL("autoreset",    mjDSBL_AUTORESET)
    READDSBL("nativeccd",    mjDSBL_NATIVECCD)
    READDSBL("island",       mjDSBL_ISLAND)
    READDSBL("multiccd",     mjDSBL_MULTICCD)
    // clang-format on
#undef READDSBL

#define READENBL(NAME, MASK)                              \
  if (MapValue(elem, NAME, &n, enable_map, 2)) {          \
    opt->enableflags        ^= (opt->enableflags & MASK); \
    opt->enableflags        |= (n ? MASK : 0);            \
    s->authored.enableflags |= MASK;                      \
  }
    // clang-format off
    READENBL("override",    mjENBL_OVERRIDE)
    READENBL("energy",      mjENBL_ENERGY)
    READENBL("fwdinv",      mjENBL_FWDINV)
    READENBL("invdiscrete", mjENBL_INVDISCRETE)
    READENBL("sleep",       mjENBL_SLEEP)
    READENBL("diagexact",   mjENBL_DIAGEXACT)
    READENBL("ipc",         mjENBL_IPC)
    READENBL("speculative", mjENBL_SPECULATIVE)
    // clang-format on
#undef READENBL
  }
}


// size section parser
void mjXReader::Size(XMLElement* section, mjSpec* s) {
  // read memory bytes
  {
    constexpr char err_msg[] =
        "unsigned integer with an optional suffix {K,M,G,T,P,E} is expected in "
        "attribute 'memory' (or the size specified is too big)";

    auto memory = [&]() -> std::optional<std::size_t> {
      const char* pstr = section->Attribute("memory");
      if (!pstr) { return std::nullopt; }

      // trim entire string
      string trimmed;
      {
        std::istringstream strm((string(pstr)));
        strm >> trimmed;
        string trailing;
        strm >> trailing;
        if (!trailing.empty() || !strm.eof()) { throw mjXError(section, "%s", err_msg); }

        // allow explicit specification of the default "-1" value
        if (trimmed == "-1") { return std::nullopt; }
      }

      std::istringstream strm(trimmed);

      // check that the number is not negative
      if (strm.peek() == '-') { throw mjXError(section, "%s", err_msg); }

      std::size_t base_size;
      strm >> base_size;
      if (strm.fail()) {
        // either not an integer or the number without the suffix is already bigger than size_t
        throw mjXError(section, "%s", err_msg);
      }

      // parse the multiplier suffix
      int multiplier_bit = 0;
      if (!strm.eof()) {
        char suffix = strm.get();
        if (suffix == 'K' || suffix == 'k') {
          multiplier_bit = 10;
        } else if (suffix == 'M' || suffix == 'm') {
          multiplier_bit = 20;
        } else if (suffix == 'G' || suffix == 'g') {
          multiplier_bit = 30;
        } else if (suffix == 'T' || suffix == 't') {
          multiplier_bit = 40;
        } else if (suffix == 'P' || suffix == 'p') {
          multiplier_bit = 50;
        } else if (suffix == 'E' || suffix == 'e') {
          multiplier_bit = 60;
        }

        // check for invalid suffix, or suffix longer than one character
        strm.get();
        if (!multiplier_bit || !strm.eof()) { throw mjXError(section, "%s", err_msg); }
      }

      // check that the specified suffix isn't bigger than size_t
      if (multiplier_bit + 1 > std::numeric_limits<std::size_t>::digits) {
        throw mjXError(section, "%s", err_msg);
      }

      // check that the suffix won't take the total size beyond size_t
      const std::size_t max_base_size =
          (std::numeric_limits<std::size_t>::max() << multiplier_bit) >> multiplier_bit;
      if (base_size > max_base_size) { throw mjXError(section, "%s", err_msg); }

      const std::size_t total_size = base_size << multiplier_bit;
      return total_size;
    }();

    if (memory.has_value()) {
      if (*memory / sizeof(mjtNum) > std::numeric_limits<std::size_t>::max()) {
        throw mjXError(section, "%s", err_msg);
      }
      s->memory = *memory;
    }
  }

  // mechanical attributes
  ReadAttrTableCore(section,
                    s,
                    kSizeAttrs,
                    kSizeAttrsN,
                    /*skipnodefault=*/false);

  ReadAttrInt(section, "nconmax", &s->nconmax);
  if (s->nconmax < -1) throw mjXError(section, "nconmax must be >= -1");

  // memory/nstack and memory/njmax exclusivity is enforced by the schema
  {
    int nstack = -1;
    if (ReadAttrInt(section, "nstack", &nstack)) {
      if (nstack < -1) throw mjXError(section, "nstack must be >= -1");
      s->nstack = nstack;
    }
  }
  if (ReadAttrInt(section, "njmax", &s->njmax)) {
    if (s->njmax < -1) throw mjXError(section, "njmax must be >= -1");
  }

  if (s->nuser_body < -1) throw mjXError(section, "nuser_body must be >= -1");

  if (s->nuser_jnt < -1) throw mjXError(section, "nuser_jnt must be >= -1");

  if (s->nuser_geom < -1) throw mjXError(section, "nuser_geom must be >= -1");

  if (s->nuser_site < -1) throw mjXError(section, "nuser_site must be >= -1");

  if (s->nuser_cam < -1) throw mjXError(section, "nuser_cam must be >= -1");

  if (s->nuser_tendon < -1) throw mjXError(section, "nuser_tendon must be >= -1");

  if (s->nuser_actuator < -1) throw mjXError(section, "nuser_actuator must be >= -1");

  if (s->nuser_sensor < -1) throw mjXError(section, "nuser_sensor must be >= -1");
}


// statistic section parser
void mjXReader::Statistic(XMLElement* section) {
  // mechanical attributes
  ReadAttrTableCore(section,
                    &spec->stat,
                    kStatisticAttrs,
                    kStatisticAttrsN,
                    /*skipnodefault=*/false);
  if (mjuu_defined(spec->stat.extent) && spec->stat.extent <= 0) {
    throw mjXError(section, "extent must be strictly positive");
  }
}


//---------------------------------- one-element parsers -------------------------------------------

// flex element parser
void mjXReader::OneFlex(XMLElement* elem, mjsFlex* flex) {
  string text;
  int    n;

  // mechanical attributes
  ReadAttrTable(elem, flex, flex->element, kFlexAttrs, kFlexAttrsN);

  // cellcount is seeded before reading
  flex->cellcount[0] = 1;
  flex->cellcount[1] = 1;
  flex->cellcount[2] = 1;
  ReadAttr(elem, "cellcount", 3, flex->cellcount, text);

  // dof lowers to interpolation order
  flex->order = 0;
  if (MapValue(elem, "dof", &n, fdof_map, mjNFCOMPDOFS)) {
    flex->order = (n == mjFCOMPDOF_QUADRATIC) ? 2 : (n == mjFCOMPDOF_TRILINEAR ? 1 : 0);
  }

  // sub-elements project into the flex
  XMLElement* cont = FirstChildElement(elem, "contact");
  if (cont) {
    ReadAttrTable(cont, flex, flex->element, kFlexcomp_contactAttrs, kFlexcomp_contactAttrsN);
  }
  XMLElement* edge = FirstChildElement(elem, "edge");
  if (edge) { ReadAttrTable(edge, flex, flex->element, kFlex_edgeAttrs, kFlex_edgeAttrsN); }
  XMLElement* elasticity = FirstChildElement(elem, "elasticity");
  if (elasticity) {
    ReadAttrTable(elasticity, flex, flex->element, kElasticityAttrs, kElasticityAttrsN);
  }

  // write error info
  mjs_setString(flex->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// an asset which has a file and no name is named after the file
static void NameFromFile(XMLElement* elem, mjsElement* asset, const mjString* file) {
  if (!mjs_getName(asset)->empty() || file->empty()) { return; }
  std::string name = mjuu_stripext(mjuu_strippath(*file));
  if (mjs_setName(asset, name.c_str())) {
    throw mjXError(elem, "%s", mjs_getError(mjs_getSpec(asset)));
  }
}


// mesh element parser
void mjXReader::OneMesh(XMLElement* elem, mjsMesh* mesh, const mjVFS* vfs) {
  int    n;
  string text;

  // mechanical attributes
  ReadAttrTable(elem, mesh, mesh->element, kMeshAttrs, kMeshAttrsN);

  // file, resolved against the mesh directory
  auto file = ReadAttrFile(elem, "file", vfs, MeshDir());
  if (file) { mjs_setString(mesh->file, file->c_str()); }
  NameFromFile(elem, mesh->element, mesh->file);

  // plugin sub-element
  XMLElement* eplugin = FirstChildElement(elem, "plugin");
  if (eplugin) { OnePlugin(eplugin, &mesh->plugin); }

  // maxhullvert with validation
  if (ReadAttrInt(elem, "maxhullvert", &n)) {
    if (n != -1 && n < 4) throw mjXError(elem, "maxhullvert must be larger than 3");
    mesh->maxhullvert = n;
  }

  // builtin options
  if (MapValue(elem, "builtin", &n, meshbuiltin_map, meshbuiltin_sz)) {
    std::vector<double> params;
    int                 nparams = ReadVector(elem, "params", params, text, /*required*/ true);
    if (mjs_makeMesh(mesh, (mjtMeshBuiltin)n, params.data(), nparams)) {
      throw mjXError(elem, "%s", mjs_getError(spec));
    }
  }

  // write error info
  mjs_setString(mesh->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// skin element parser
void mjXReader::OneSkin(XMLElement* elem, mjsSkin* skin, const mjVFS* vfs) {
  string text;
  float  data[4];

  // mechanical attributes
  ReadAttrTable(elem, skin, skin->element, kSkinAttrs, kSkinAttrsN);

  // file, resolved against the asset directory
  auto file = ReadAttrFile(elem, "file", vfs, AssetDir());
  if (file.has_value()) { mjs_setString(skin->file, file->c_str()); }
  NameFromFile(elem, skin->element, skin->file);

  // group with range validation
  ReadAttrInt(elem, "group", &skin->group);
  if (skin->group < 0 || skin->group >= mjNGROUP) {
    throw mjXError(elem, "skin group must be between 0 and 5");
  }

  // read bones
  XMLElement*        bone = FirstChildElement(elem, "bone");
  std::vector<float> bindpos;
  std::vector<float> bindquat;

  while (bone) {
    // read body
    ReadAttrTxt(bone, "body", text, true);
    mjs_appendString(skin->bodyname, text.c_str());

    // read bindpos
    ReadAttr(bone, "bindpos", 3, data, text, true);
    bindpos.push_back(data[0]);
    bindpos.push_back(data[1]);
    bindpos.push_back(data[2]);

    // read bindquat
    ReadAttr(bone, "bindquat", 4, data, text, true);
    bindquat.push_back(data[0]);
    bindquat.push_back(data[1]);
    bindquat.push_back(data[2]);
    bindquat.push_back(data[3]);

    // read vertid
    auto tempid = ReadAttrVec<int>(bone, "vertid", true);
    if (tempid.has_value()) { mjs_appendIntVec(skin->vertid, tempid->data(), tempid->size()); }

    // read vertweight
    auto tempweight = ReadAttrVec<float>(bone, "vertweight", true);
    if (tempweight.has_value()) {
      mjs_appendFloatVec(skin->vertweight, tempweight->data(), tempweight->size());
    }

    // advance to next bone
    bone = NextSiblingElement(bone, "bone");
  }

  // set bind vectors
  mjs_setFloat(skin->bindpos, bindpos.data(), bindpos.size());
  mjs_setFloat(skin->bindquat, bindquat.data(), bindquat.size());

  // write error info
  mjs_setString(skin->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// texture element parser
void mjXReader::OneTexture(XMLElement* elem, mjsTexture* texture, const mjVFS* vfs) {
  string text;
  int    inherited_gridsize[2] = {texture->gridsize[0], texture->gridsize[1]};

  // mechanical attributes
  ReadAttrTable(elem, texture, texture->element, kTextureAttrs, kTextureAttrsN);

  // file, resolved against the texture directory
  auto file = ReadAttrFile(elem, "file", vfs, TextureDir());
  if (file.has_value()) { mjs_setString(texture->file, file->c_str()); }
  NameFromFile(elem, texture->element, texture->file);

  // gridlayout length must equal the gridsize product (value-conditional)
  if (ReadAttrTxt(elem, "gridlayout", text)) {
    if (text.length() != texture->gridsize[0] * texture->gridsize[1]) {
      throw mjXError(elem, "gridlayout length must match gridsize");
    }
  } else if (elem->Attribute("gridsize") && (texture->gridsize[0] != inherited_gridsize[0] ||
                                             texture->gridsize[1] != inherited_gridsize[1])) {
    std::memcpy(texture->gridlayout, "............", sizeof(texture->gridlayout));
  }

  // separate files
  std::vector<string> cubefiles(6);
  std::vector<string> cubefile_names =
      {"fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"};
  for (int i = 0; i < cubefiles.size(); i++) {
    auto maybe_file = ReadAttrFile(elem, cubefile_names[i].c_str(), vfs, TextureDir());
    if (maybe_file.has_value()) {
      cubefiles[i] = maybe_file.value().Str();
    } else {
      cubefiles[i] = "";
    }
    mjs_setInStringVec(texture->cubefiles, i, cubefiles[i].c_str());
  }

  // write error info
  mjs_setString(texture->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// material element parser
void mjXReader::OneMaterial(XMLElement* elem, mjsMaterial* material) {
  string text, texture;

  // mechanical attributes
  ReadAttrTable(elem, material, material->element, kMaterialAttrs, kMaterialAttrsN);

  // the texture attribute and layer sub-elements are mutually exclusive
  bool tex_attributes_found = false;
  if (ReadAttrTxt(elem, "texture", texture)) {
    mjs_setInStringVec(material->textures, mjTEXROLE_RGB, texture.c_str());
    tex_attributes_found = true;
  }
  XMLElement* layer = FirstChildElement(elem);
  while (layer) {
    if (tex_attributes_found) {
      throw mjXError(layer, "A material with a texture attribute cannot have layer sub-elements");
    }

    // layer sub-element
    ReadAttrTxt(layer, "role", text, true);
    int role = FindKey(texrole_map, texrole_sz, text);
    if (role < 0) { throw mjXError(layer, "unknown layer role '%s'", text.c_str()); }
    ReadAttrTxt(layer, "texture", text, true);
    mjs_setInStringVec(material->textures, role, text.c_str());
    layer = NextSiblingElement(layer);
  }

  // write error info
  mjs_setString(material->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// joint element parser
void mjXReader::OneJoint(XMLElement* elem, mjsJoint* joint) {
  // mechanical attributes
  ReadAttrTable(elem, joint, joint->element, kJointAttrs, kJointAttrsN);

  // write error info
  mjs_setString(joint->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// geom element parser
void mjXReader::OneGeom(XMLElement* elem, mjsGeom* geom) {
  string text;
  int    n;

  // mechanical attributes
  ReadAttrTable(elem, geom, geom->element, kGeomAttrs, kGeomAttrsN);

  // fluid interaction model: keyword lowers to on/off
  if (MapValue(elem, "fluidshape", &n, fluidshape_map, 2)) { geom->fluid_ellipsoid = (n == 1); }

  // orientation alternatives
  ReadQuat(elem, "quat", geom->quat, text);
  ReadAlternative(elem, geom->alt);

  // plugin sub-element
  XMLElement* eplugin = FirstChildElement(elem, "plugin");
  if (eplugin) { OnePlugin(eplugin, &geom->plugin); }

  // write error info
  mjs_setString(geom->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// site element parser
void mjXReader::OneSite(XMLElement* elem, mjsSite* site) {
  string text;

  // mechanical attributes
  ReadAttrTable(elem, site, site->element, kSiteAttrs, kSiteAttrsN);

  // orientation alternatives
  ReadQuat(elem, "quat", site->quat, text);
  ReadAlternative(elem, site->alt);

  // write error info
  mjs_setString(site->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// camera element parser
void mjXReader::OneCamera(XMLElement* elem, mjsCamera* camera) {
  string text;

  // mechanical attributes
  ReadAttrTable(elem, camera, camera->element, kCameraAttrs, kCameraAttrsN);

  // orientation alternatives
  ReadQuat(elem, "quat", camera->quat, text);
  ReadAlternative(elem, camera->alt);

  // write error info
  mjs_setString(camera->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// light element parser
void mjXReader::OneLight(XMLElement* elem, mjsLight* light) {
  int n;

  // mechanical attributes
  ReadAttrTable(elem, light, light->element, kLightAttrs, kLightAttrsN);

  // directional is a legacy alias for type; the pair is mutually exclusive
  if (MapValue(elem, "directional", &n, bool_map, 2)) {
    light->type = (n == 1) ? mjLIGHT_DIRECTIONAL : mjLIGHT_SPOT;
  }
  if (MapValue(elem, "type", &n, lighttype_map, lighttype_sz)) { light->type = (mjtLightType)n; }

  // write error info
  mjs_setString(light->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// pair element parser
// read the mechanical attributes of an element, driven by its generated rows
void mjXReader::ReadAttrTable(
    XMLElement* elem, void* obj, mjsElement* el, const mjXAttr* rows, int nrow) {
  // element names need the spec for error context; the core handles the rest
  string text;
  for (int i = 0; i < nrow; i++) {
    const mjXAttr& row = rows[i];
    if (row.kind != mjXAttr::kName || (readingdefaults && row.nodefault)) { continue; }
    if (ReadAttrTxt(elem, row.attr, text, row.required) && mjs_setName(el, text.c_str())) {
      throw mjXError(elem, "%s", mjs_getError(spec));
    }
  }
  ReadAttrTableCore(elem, obj, rows, nrow, readingdefaults);
}


// static core: every row kind except element names
void mjXReader::ReadAttrTableCore(XMLElement*    elem,
                                  void*          obj,
                                  const mjXAttr* rows,
                                  int            nrow,
                                  bool           skipnodefault,
                                  const void*    authored) {
  string text;
  for (int i = 0; i < nrow; i++) {
    const mjXAttr& row = rows[i];
    if (skipnodefault && row.nodefault) { continue; }
    char* base = (char*)obj + row.offset;
    bool  got  = false;
    switch (row.kind) {
      case mjXAttr::kName:
        break;  // handled by the member wrapper
      case mjXAttr::kString:
        if (ReadAttrTxt(elem, row.attr, text, row.required)) {
          mjs_setString(*(mjString**)base, text.c_str());
          got = true;
        }
        break;
      case mjXAttr::kStringVec:
        if (ReadAttrTxt(elem, row.attr, text, row.required)) {
          mjs_setStringVec(*(mjStringVec**)base, text.c_str());
          got = true;
        }
        break;
      case mjXAttr::kInt:
        got = ReadAttr(elem, row.attr, row.len, (int*)base, text, row.required, row.exact) > 0;
        break;
      case mjXAttr::kDouble:
        got = ReadAttr(elem, row.attr, row.len, (double*)base, text, row.required, row.exact) > 0;
        break;
      case mjXAttr::kNum:
        got = ReadAttr(elem, row.attr, row.len, (mjtNum*)base, text, row.required, row.exact) > 0;
        break;
      case mjXAttr::kFloat:
        got = ReadAttr(elem, row.attr, row.len, (float*)base, text, row.required, row.exact) > 0;
        break;
      case mjXAttr::kEnum: {
        int value;
        if (MapValue(elem, row.attr, &value, row.map, row.mapsz, row.required)) {
          *(int*)base = value;  // enum fields are int-sized
          got         = true;
        }
        break;
      }
      case mjXAttr::kFlags: {
        std::vector<int> values(row.mapsz);
        int              nvalues = MapValues(elem, row.attr, values.data(), row.map, row.mapsz);
        if (nvalues) {
          int combined = 0;
          for (int j = 0; j < nvalues; j++) { combined |= values[j]; }
          *(int*)base = combined;
          got         = true;
        }
        break;
      }
      case mjXAttr::kEnumByte: {
        int value;
        if (MapValue(elem, row.attr, &value, row.map, row.mapsz, row.required)) {
          *(mjtByte*)base = (mjtByte)value;
          got             = true;
        }
        break;
      }
      case mjXAttr::kBool: {
        int value;
        if (MapValue(elem, row.attr, &value, bool_map, 2, row.required)) {
          *(mjtBool*)base = (value == 1);
          got             = true;
        }
        break;
      }
      case mjXAttr::kConst:
        *(int*)base = row.value;  // enum fields are int-sized
        break;
      case mjXAttr::kDoubleVec: {
        std::vector<double> values;
        if (ReadVector(elem, row.attr, values, text)) {
          mjs_setDouble(*(mjDoubleVec**)base, values.data(), values.size());
          got = true;
        }
        break;
      }
      case mjXAttr::kFloatVec: {
        auto values = ReadAttrVec<float>(elem, row.attr, row.required);
        if (values.has_value()) {
          mjs_setFloat(*(mjFloatVec**)base, values->data(), values->size());
          got = true;
        }
        break;
      }
      case mjXAttr::kIntVec: {
        auto values = ReadAttrVec<int>(elem, row.attr, row.required);
        if (values.has_value()) {
          mjs_setInt(*(mjIntVec**)base, values->data(), values->size());
          got = true;
        }
        break;
      }
      case mjXAttr::kChars:
        if (ReadAttrTxt(elem, row.attr, text, row.required)) {
          if (row.exact && (int)text.size() != row.len) {
            throw mjXError(elem,
                           "attribute '%s' must have exactly %d "
                           "characters",
                           row.attr,
                           row.len);
          }
          if ((int)text.size() > row.len) {
            throw mjXError(elem,
                           "attribute '%s' may have at most %d "
                           "characters",
                           row.attr,
                           row.len);
          }
          std::memset(base, 0, row.len);
          std::memcpy(base, text.data(), text.size());
          got = true;
        }
        break;
    }

    // record XML-authored fields for attach conflict resolution
    if (got && authored) { mjs_setAuthored(authored, base, 1); }
  }
}


void mjXReader::OnePair(XMLElement* elem, mjsPair* pair) {
  ReadAttrTable(elem, pair, pair->element, kPairAttrs, kPairAttrsN);

  // write error info
  mjs_setString(pair->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// equality element parser
void mjXReader::OneEquality(XMLElement* elem, mjsEquality* equality) {
  string text, name1, name2;

  // read type (bad keywords already detected by schema)
  text     = elem->Value();
  int type = FindKey(equality_map, equality_sz, text);
  if (type >= 0) equality->type = (mjtEq)type;

  // common attributes
  ReadAttrTable(elem, equality, equality->element, kEqualityBaseAttrs, kEqualityBaseAttrsN);

  // regular only
  if (!readingdefaults) {
    switch (equality->type) {
      case mjEQ_CONNECT: {
        auto maybe_site1 = ReadAttrStr(elem, "site1");
        auto maybe_site2 = ReadAttrStr(elem, "site2");
        auto maybe_body1 = ReadAttrStr(elem, "body1");
        auto maybe_body2 = ReadAttrStr(elem, "body2");
        bool has_anchor  = ReadAttr(elem, "anchor", 3, equality->data, text);

        bool body_semantic = maybe_body1.has_value() && has_anchor;

        if (body_semantic) {
          name1 = maybe_body1.value();
          if (maybe_body2.has_value()) { name2 = maybe_body2.value(); }
          equality->objtype = mjOBJ_BODY;
        } else {
          name1             = maybe_site1.value();
          name2             = maybe_site2.value();
          equality->objtype = mjOBJ_SITE;
        }
      } break;

      case mjEQ_WELD: {
        auto maybe_site1 = ReadAttrStr(elem, "site1");
        auto maybe_site2 = ReadAttrStr(elem, "site2");
        auto maybe_body1 = ReadAttrStr(elem, "body1");
        auto maybe_body2 = ReadAttrStr(elem, "body2");
        bool has_anchor  = ReadAttr(elem, "anchor", 3, equality->data, text);
        ReadAttr(elem, "relpose", 7, equality->data + 3, text);

        bool body_semantic = maybe_body1.has_value();

        if (body_semantic) {
          name1 = maybe_body1.value();
          if (maybe_body2.has_value()) { name2 = maybe_body2.value(); }
          equality->objtype = mjOBJ_BODY;
          if (!has_anchor) { mjuu_zerovec(equality->data, 3); }
        } else {
          name1             = maybe_site1.value();
          name2             = maybe_site2.value();
          equality->objtype = mjOBJ_SITE;
        }

        ReadAttr(elem, "torquescale", 1, equality->data + 10, text);
      } break;

      case mjEQ_JOINT:
        ReadAttrTxt(elem, "joint1", name1, true);
        ReadAttrTxt(elem, "joint2", name2);
        ReadAttr(elem, "polycoef", 5, equality->data, text, false, false);
        break;

      case mjEQ_TENDON:
        ReadAttrTxt(elem, "tendon1", name1, true);
        ReadAttrTxt(elem, "tendon2", name2);
        ReadAttr(elem, "polycoef", 5, equality->data, text, false, false);
        break;

      case mjEQ_FLEX:
        ReadAttrTxt(elem, "flex", name1, true);
        break;

      case mjEQ_FLEXSTRAIN:
        ReadAttrTxt(elem, "flex", name1, true);
        ReadAttr(elem, "cell", 3, equality->data, text);
        break;

      case mjEQ_DISTANCE:
        throw mjXError(elem,
                       "support for distance equality constraints was removed in MuJoCo 2.2.2");
        break;

      default:  // SHOULD NOT OCCUR
        throw mjXError(elem, "unrecognized equality constraint type");
    }

    mjs_setString(equality->name1, name1.c_str());
    if (!name2.empty()) { mjs_setString(equality->name2, name2.c_str()); }
  }

  // write error info
  mjs_setString(equality->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// tendon element parser
void mjXReader::OneTendon(XMLElement* elem, mjsTendon* tendon) {
  string text;

  // mechanical attributes; fixed tendons read a subset of the spatial rows
  ReadAttrTable(elem, tendon, tendon->element, kSpatialAttrs, kSpatialAttrsN);

  // read springlength, either one or two values; if one, copy to second value
  if (ReadAttr(elem, "springlength", 2, tendon->springlength, text, false, false) == 1) {
    tendon->springlength[1] = tendon->springlength[0];
  }

  // write error info
  mjs_setString(tendon->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// read the "input" attribute: so3 chart keyword, the "none" keyword (empty signature), or an
// input token list, required to be in canonical order [pos, vel, ff, voltage, pressure]
static bool ReadInputSpec(tinyxml2::XMLElement* elem, int* ctrlspec) {
  std::string text;
  if (!mjXUtil::ReadAttrTxt(elem, "input", text)) { return false; }

  // so3 chart keyword
  int chart = mjXUtil::FindKey(inputchart_map, inputchart_sz, text);
  if (chart >= 0) {
    *ctrlspec = chart;
    return true;
  }

  // empty-signature keyword
  int keyword = mjXUtil::FindKey(inputkeyword_map, inputkeyword_sz, text);
  if (keyword >= 0) {
    *ctrlspec = keyword;
    return true;
  }

  // input tokens; strictly ascending bits = canonical order, no duplicates
  int bits[inputbit_sz];
  int nbit = mjXUtil::MapValues(elem, "input", bits, inputbit_map, inputbit_sz);
  int spec = 0;
  for (int k = 0; k < nbit; k++) {
    if (bits[k] <= (k ? bits[k - 1] : 0)) {
      throw mjXError(elem,
                     "inputs must be listed in canonical order [pos, vel, ff, voltage, pressure]");
    }
    spec |= bits[k];
  }
  *ctrlspec = spec;
  return true;
}


// actuator element parser
void mjXReader::OneActuator(XMLElement* elem, mjsActuator* actuator, const mjsDefault* def) {
  string text, type, target, slidersite, refsite;

  // input signatures of <general> are scoped by gaintype
  mjtGain inherited_gaintype = actuator->gaintype;

  // the tag selects the shortcut and its mechanical rows; plugins read the general rows
  type                          = elem->Value();
  const mjXActuatorEntry* entry = nullptr;
  for (int i = 0; i < kActuatorDispatchN; i++) {
    if (type == kActuatorDispatch[i].tag) { entry = kActuatorDispatch + i; }
  }
  const mjXAttr* rows = entry ? entry->rows : kGeneralAttrs;
  int            nrow = entry ? entry->n : kGeneralAttrsN;
  ReadAttrTable(elem, actuator, actuator->element, rows, nrow);

  // transmission target and type
  if (ReadAttrTxt(elem, "joint", target)) {
    mjs_setString(actuator->target, target.c_str());
    actuator->trntype = mjTRN_JOINT;
  }
  if (ReadAttrTxt(elem, "jointinparent", target)) {
    mjs_setString(actuator->target, target.c_str());
    actuator->trntype = mjTRN_JOINTINPARENT;
  }
  if (ReadAttrTxt(elem, "tendon", target)) {
    mjs_setString(actuator->target, target.c_str());
    actuator->trntype = mjTRN_TENDON;
  }
  if (ReadAttrTxt(elem, "cranksite", target)) {
    mjs_setString(actuator->target, target.c_str());
    actuator->trntype = mjTRN_SLIDERCRANK;
  }
  if (ReadAttrTxt(elem, "site", target)) {
    mjs_setString(actuator->target, target.c_str());
    actuator->trntype = mjTRN_SITE;
  }
  if (ReadAttrTxt(elem, "body", target)) {
    mjs_setString(actuator->target, target.c_str());
    actuator->trntype = mjTRN_BODY;
  }
  // slidercrank-specific parameters
  int r1 = ReadAttr(elem, "cranklength", 1, &actuator->cranklength, text);
  int r2 = ReadAttrTxt(elem, "slidersite", slidersite);
  if (r2) { mjs_setString(actuator->slidersite, slidersite.c_str()); }
  if ((r1 || r2) &&
      actuator->trntype != mjTRN_SLIDERCRANK &&
      actuator->trntype != mjTRN_UNDEFINED) {
    throw mjXError(elem, "cranklength and slidersite can only be used in slidercrank transmission");
  }

  // site-specific parameters (refsite)
  int r3 = ReadAttrTxt(elem, "refsite", refsite);
  if (r3) { mjs_setString(actuator->refsite, refsite.c_str()); }
  if (r3 && actuator->trntype != mjTRN_SITE && actuator->trntype != mjTRN_UNDEFINED) {
    throw mjXError(elem, "refsite can only be used with site transmission");
  }

  // general: so3 chart keyword, or input token subset (validated per gaintype by the compiler)
  if (type == "general") {
    if (actuator->gaintype != inherited_gaintype) { actuator->ctrlspec = 0; }
    ReadInputSpec(elem, &actuator->ctrlspec);
    actuator->type = mjACTUATOR_GENERAL;
  }

  // plugin
  else if (type == "plugin") {
    OnePlugin(elem, &actuator->plugin);
    actuator->type = mjACTUATOR_GENERAL;
  }

  // shortcut: start from what the class or its nearest ancestor written with the same shortcut or
  // general gives, else the documented defaults
  else if (entry) {
    mjtActuator        acttype = (mjtActuator)entry->type;
    const mjCDef*      cdef    = def ? static_cast<const mjCDef*>(def->element) : nullptr;
    const mjsActuator* src     = mjXShortcutSource(actuator, cdef, acttype);
    mjXShortcut        s;
    if (src) {
      mjXShortcutFromActuator(&s, src, acttype);
    } else {
      mjXShortcutDefaults(&s, acttype);
    }

    // kv or dampratio, explicit; both given is an error of the setter
    bool kv = ReadAttr(elem, "kv", 1, &s.kv, text);
    bool dr = ReadAttr(elem, "dampratio", 1, &s.dampratio, text);
    if (kv || dr) {
      s.has_kv        = kv;
      s.has_dampratio = dr;
    }

    switch (acttype) {
      case mjACTUATOR_POSITION:
      case mjACTUATOR_INTVELOCITY: {
        ReadAttr(elem, "kp", 1, &s.kp, text);
        bool has_inherit = ReadAttr(elem, "inheritrange", 1, &s.inheritrange, text);
        bool has_range =
            elem->Attribute(acttype == mjACTUATOR_POSITION ? "ctrlrange" : "actrange") != nullptr;
        if (has_range && !has_inherit) {
          s.inheritrange = 0;
        } else if (has_inherit && s.inheritrange > 0 && !has_range) {
          double* range = acttype == mjACTUATOR_POSITION ? actuator->ctrlrange : actuator->actrange;
          range[0]      = 0;
          range[1]      = 0;
        }
        if (ReadAttr(elem, "timeconst", 1, s.timeconst, text)) { s.has_timeconst = true; }
        break;
      }
      case mjACTUATOR_ORIENTATION:
        ReadAttr(elem, "kp", 1, &s.kp, text);
        ReadInputSpec(elem, &s.ctrlspec);
        break;
      case mjACTUATOR_PID: {
        ReadAttr(elem, "kp", 1, &s.kp, text);
        ReadAttr(elem, "ki", 1, &s.ki, text);
        ReadAttr(elem, "imax", 1, &s.imax, text);
        ReadAttr(elem, "slewmax", 1, &s.slewmax, text);
        bool has_inherit = ReadAttr(elem, "inheritrange", 1, &s.inheritrange, text);
        ReadInputSpec(elem, &s.ctrlspec);
        // posrange is an alias of ctrlrange (the position-setpoint input)
        bool has_posrange = ReadAttr(elem, "posrange", 2, actuator->ctrlrange, text) ||
                            elem->Attribute("ctrlrange") != nullptr;
        if (has_posrange && !has_inherit) {
          s.inheritrange = 0;
        } else if (has_inherit && s.inheritrange > 0 && !has_posrange) {
          actuator->ctrlrange[0] = 0;
          actuator->ctrlrange[1] = 0;
        }
        break;
      }
      case mjACTUATOR_CYLINDER:
        ReadAttr(elem, "timeconst", 1, s.timeconst, text);
        ReadAttr(elem, "bias", 3, s.bias, text);
        ReadAttr(elem, "area", 1, &s.area, text);
        ReadAttr(elem, "diameter", 1, &s.diameter, text);
        break;
      case mjACTUATOR_MUSCLE:
        ReadAttr(elem, "timeconst", 2, s.timeconst, text);
        ReadAttr(elem, "tausmooth", 1, &s.tausmooth, text);
        ReadAttr(elem, "range", 2, s.range, text);
        ReadAttr(elem, "force", 1, &s.force, text);
        ReadAttr(elem, "scale", 1, &s.scale, text);
        ReadAttr(elem, "lmin", 1, &s.lmin, text);
        ReadAttr(elem, "lmax", 1, &s.lmax, text);
        ReadAttr(elem, "vmax", 1, &s.vmax, text);
        ReadAttr(elem, "fpmax", 1, &s.fpmax, text);
        ReadAttr(elem, "fvmax", 1, &s.fvmax, text);
        break;
      case mjACTUATOR_ADHESION:
        ReadAttr(elem, "gain", 1, &s.gain, text);
        break;
      case mjACTUATOR_DCMOTOR:
        ReadAttr(elem, "motorconst", 2, s.motorconst, text, false, false);
        ReadAttr(elem, "resistance", 1, &s.resistance, text);
        ReadAttr(elem, "nominal", 3, s.nominal, text, false, false);
        // authored saturation replaces the inherited torque limit
        if (ReadAttr(elem, "saturation", 3, s.saturation, text, false, false)) {
          actuator->forcelimited  = mjLIMITED_AUTO;
          actuator->forcerange[0] = actuator->forcerange[1] = 0;
        }
        // L and timeconst are alternatives: an authored L alone replaces the inherited timeconst
        if (ReadAttr(elem, "inductance", 2, s.inductance, text, false, false) == 1) {
          s.inductance[1] = 0;
        }
        ReadAttr(elem, "cogging", 3, s.cogging, text, false, false);
        ReadAttr(elem, "controller", 6, s.controller, text, false, false);
        ReadAttr(elem, "thermal", 6, s.thermal, text, false, false);
        ReadAttr(elem, "lugre", 5, s.lugre, text, false, false);
        ReadInputSpec(elem, &s.ctrlspec);
        break;
      default:  // motor, velocity, damper: kv is read above
        break;
    }

    // throw error if the setter failed
    const char* err = mjXSetToShortcut(actuator, acttype, s);
    if (err[0]) { throw mjXError(elem, err); }
  }

  else {  // SHOULD NOT OCCUR
    throw mjXError(elem, "unrecognized actuator type: %s", type.c_str());
  }

  // write info
  mjs_setString(actuator->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
}


// make flexcomp
void mjXReader::OneFlexcomp(XMLElement* elem, mjsBody* body, mjsFrame* frame, const mjVFS* vfs) {
  string text, material;
  int    n;

  // create out-of-DOM element
  auto         fcomp_ptr = std::make_unique<mjCFlexcomp>();
  mjCFlexcomp& fcomp     = *fcomp_ptr;
  mjsFlex&     dflex     = *fcomp.def.spec.flex;
  fcomp.frame            = frame;

  // common properties
  ReadAttrTxt(elem, "name", fcomp.name, true);
  if (MapValue(elem, "type", &n, fcomp_map, mjNFCOMPTYPES)) { fcomp.type = (mjtFcompType)n; }
  ReadAttr(elem, "count", 3, fcomp.count, text);
  ReadAttr(elem, "cellcount", 3, fcomp.cellcount, text);
  ReadAttr(elem, "spacing", 3, fcomp.spacing, text);
  ReadAttr(elem, "scale", 3, fcomp.scale, text);
  ReadAttr(elem, "mass", 1, &fcomp.mass, text);
  ReadAttr(elem, "inertiabox", 1, &fcomp.inertiabox, text);
  auto maybe_file = ReadAttrFile(elem, "file", vfs, modelfiledir_);
  if (maybe_file.has_value()) {
    fcomp.file = std::move(maybe_file.value().Str());
  } else {
    fcomp.file = "";
  }
  ReadAttrTxt(elem, "mesh", fcomp.mesh);
  if (!fcomp.mesh.empty() && !fcomp.file.empty()) {
    throw mjXError(elem, "both 'file' and 'mesh' specified");
  }
  if (!fcomp.mesh.empty() && !elem->Attribute("type")) { fcomp.type = mjFCOMPTYPE_MESH; }
  if (ReadAttrTxt(elem, "material", material)) {
    mjs_setStringVec(dflex.material, material.c_str());
  }
  ReadAttr(elem, "rgba", 4, dflex.rgba, text);
  if (MapValue(elem, "flatskin", &n, bool_map, 2)) {
    dflex.flatskin     = (n == 1);
    fcomp.has_flatskin = true;
  }
  if (ReadAttrInt(elem, "dim", &dflex.dim)) { fcomp.has_dim = true; }
  ReadAttr(elem, "radius", 1, &dflex.radius, text);
  ReadAttrInt(elem, "group", &dflex.group);
  fcomp.has_origin = ReadAttr(elem, "origin", 3, fcomp.origin, text);
  if (!fcomp.has_origin && fcomp.type == mjFCOMPTYPE_MESH && fcomp.mesh.empty() && dflex.dim == 3) {
    throw mjXError(elem, "origin must be specified for mesh flexcomps if dim=3");
  }

  // pose
  ReadAttr(elem, "pos", 3, fcomp.pos, text);
  ReadAttr(elem, "quat", 4, fcomp.quat, text);
  ReadAlternative(elem, fcomp.alt);

  // user or internal
  if (MapValue(elem, "rigid", &n, bool_map, 2)) { fcomp.rigid = (n == 1); }
  auto point = ReadAttrVec<double>(elem, "point");
  if (point.has_value()) { fcomp.point = std::move(point.value()); }
  auto element = ReadAttrVec<int>(elem, "element");
  if (element.has_value()) { fcomp.element = std::move(element.value()); }
  auto texcoord = ReadAttrVec<float>(elem, "texcoord");
  if (texcoord.has_value()) { fcomp.texcoord = std::move(texcoord.value()); }

  // dof type
  if (MapValue(elem, "dof", &n, fdof_map, mjNFCOMPDOFS)) { fcomp.doftype = (mjtDof)n; }

  // edge
  XMLElement* edge = FirstChildElement(elem, "edge");
  if (edge) {
    MapValue(edge, "equality", &fcomp.equality, flexeq_map, flexeq_sz);
    ReadAttr(edge, "solref", mjNREF, fcomp.def.spec.equality->solref, text, false, false);
    ReadAttr(edge, "solimp", mjNIMP, fcomp.def.spec.equality->solimp, text, false, false);
    ReadAttr(edge, "stiffness", 1, &dflex.edgestiffness, text);
    ReadAttr(edge, "damping", 1, &dflex.edgedamping, text);
  }

  // elasticity
  XMLElement* elasticity = FirstChildElement(elem, "elasticity");
  if (elasticity) {
    ReadAttr(elasticity, "young", 1, &dflex.young, text);
    ReadAttr(elasticity, "poisson", 1, &dflex.poisson, text);
    ReadAttr(elasticity, "shear", 1, &dflex.shear, text);
    ReadAttr(elasticity, "damping", 1, &dflex.damping, text);
    ReadAttr(elasticity, "thickness", 1, &dflex.thickness, text);
    ReadAttr(elasticity, "yield", 1, &dflex.stressyield, text);
    MapValue(elasticity, "elastic2d", &dflex.elastic2d, elastic2d_map, 4);
    MapValue(elasticity, "section", &dflex.section, flexsection_map, flexsection_sz);
    ReadAttr(elasticity, "sectionsize", 2, dflex.sectionsize, text);
  }

  // contact
  XMLElement* cont = FirstChildElement(elem, "contact");
  if (cont) {
    ReadAttrInt(cont, "contype", &dflex.contype);
    ReadAttrInt(cont, "conaffinity", &dflex.conaffinity);
    ReadAttrInt(cont, "condim", &dflex.condim);
    ReadAttrInt(cont, "priority", &dflex.priority);
    ReadAttr(cont, "friction", 3, dflex.friction, text, false, false);
    ReadAttr(cont, "solmix", 1, &dflex.solmix, text);
    ReadAttr(cont, "solref", mjNREF, dflex.solref, text, false, false);
    ReadAttr(cont, "solimp", mjNIMP, dflex.solimp, text, false, false);
    ReadAttr(cont, "margin", 1, &dflex.margin, text);
    ReadAttr(cont, "gap", 1, &dflex.gap, text);
    MapValue(cont, "selfcollide", &dflex.selfcollide, flexself_map, 5);
    if (MapValue(cont, "passive", &n, bool_map, 2)) { dflex.passive = (n == 1); }
    ReadAttrInt(cont, "activelayers", &dflex.activelayers);
  }

  // pin
  XMLElement* epin = FirstChildElement(elem, "pin");
  while (epin) {
    auto id = ReadAttrVec<int>(epin, "id");
    if (id.has_value()) { fcomp.pinid.insert(fcomp.pinid.end(), id->begin(), id->end()); }
    auto range = ReadAttrVec<int>(epin, "range");
    if (range.has_value()) {
      fcomp.pinrange.insert(fcomp.pinrange.end(), range->begin(), range->end());
    }
    auto grid = ReadAttrVec<int>(epin, "grid");
    if (grid.has_value()) { fcomp.pingrid.insert(fcomp.pingrid.end(), grid->begin(), grid->end()); }
    auto gridrange = ReadAttrVec<int>(epin, "gridrange");
    if (gridrange.has_value()) {
      fcomp.pingridrange.insert(fcomp.pingridrange.end(), gridrange->begin(), gridrange->end());
    }

    // advance
    epin = NextSiblingElement(epin, "pin");
  }

  // plugin
  XMLElement* eplugin = FirstChildElement(elem, "plugin");
  if (eplugin) { OnePlugin(eplugin, &fcomp.plugin); }

  // make flexcomp
  char error[200];
  bool res = fcomp.Make(body, error, 200, vfs);

  // throw error
  if (!res) { throw mjXError(elem, "%s", error); }
}


// add plugin
void mjXReader::OnePlugin(XMLElement* elem, mjsPlugin* plugin) {
  plugin->active       = true;
  string name          = "";
  string instance_name = "";
  ReadAttrTxt(elem, "plugin", name);
  ReadAttrTxt(elem, "instance", instance_name);
  mjs_setString(plugin->plugin_name, name.c_str());
  mjs_setString(plugin->name, instance_name.c_str());
  if (instance_name.empty()) {
    plugin->element = mjs_addPlugin(spec)->element;
    ReadPluginConfigs(elem, plugin);
  } else {
    spec->hasImplicitPluginElem = true;
  }
}


//------------------ MJCF-specific sections --------------------------------------------------------

// default section parser
void mjXReader::Default(XMLElement* section, const mjsDefault* def, const mjVFS* vfs) {
  XMLElement* elem;
  string      text, name;

  // create new default, except at top level (already added in mjCModel constructor)
  text.clear();
  ReadAttrTxt(section, "class", text);
  if (text.empty()) {
    if (def) { throw mjXError(section, "empty class name"); }
  }
  if (def) {
    def = mjs_addDefault(spec, text.c_str(), def);
    if (!def) { throw mjXError(section, "repeated default class name"); }
  } else {
    def = mjs_getSpecDefault(spec);
    if (!text.empty() && text != "main") {
      throw mjXError(section, "top-level default class 'main' cannot be renamed");
    }
  }

  // iterate over elements other than nested defaults
  elem = FirstChildElement(section);
  while (elem) {
    // get element name
    name = elem->Value();

    // read mesh
    if (name == "mesh") OneMesh(elem, def->mesh, vfs);

    // read texture
    else if (name == "texture")
      OneTexture(elem, def->texture, vfs);

    // read material
    else if (name == "material")
      OneMaterial(elem, def->material);

    // read joint
    else if (name == "joint")
      OneJoint(elem, def->joint);

    // read geom
    else if (name == "geom")
      OneGeom(elem, def->geom);

    // read site
    else if (name == "site")
      OneSite(elem, def->site);

    // read camera
    else if (name == "camera")
      OneCamera(elem, def->camera);

    // read light
    else if (name == "light")
      OneLight(elem, def->light);

    // read pair
    else if (name == "pair")
      OnePair(elem, def->pair);

    // read equality
    else if (name == "equality")
      OneEquality(elem, def->equality);

    // read tendon
    else if (name == "tendon")
      OneTendon(elem, def->tendon);

    // read actuator
    else if (FindKey(actuatortype_map, actuatortype_sz, name) >= 0) {
      OneActuator(elem, def->actuator, def);
    }

    // advance
    elem = NextSiblingElement(elem);
  }

  // iterate over nested defaults
  elem = FirstChildElement(section);
  while (elem) {
    // get element name
    name = elem->Value();

    // read default
    if (name == "default") { Default(elem, def, vfs); }

    // advance
    elem = NextSiblingElement(elem);
  }
}


// extension section parser
void mjXReader::Extension(XMLElement* section) {
  XMLElement* elem = FirstChildElement(section);

  while (elem) {
    // get sub-element name
    string_view name = elem->Value();

    if (name == "plugin") {
      string plugin_name;
      ReadAttrTxt(elem, "plugin", plugin_name, /* required = */ true);
      if (mjs_activatePlugin(spec, plugin_name.c_str())) {
        throw mjXError(elem, "plugin %s not found", plugin_name.c_str());
      }

      XMLElement* child = FirstChildElement(elem);
      while (child) {
        if (string(child->Value()) == "instance") {
          if (spec->hasImplicitPluginElem) {
            throw mjXError(child,
                           "explicit plugin instance must appear before implicit plugin elements");
          }
          string     name;
          mjsPlugin* p = mjs_addPlugin(spec);
          mjs_setString(p->plugin_name, plugin_name.c_str());
          mjs_setString(p->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
          ReadAttrTxt(child, "name", name, /* required = */ true);
          mjs_setString(p->name, name.c_str());
          if (!p->name) { throw mjXError(child, "plugin instance must have a name"); }
          ReadPluginConfigs(child, p);
        }
        child = NextSiblingElement(child);
      }
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// custom section parser
void mjXReader::Custom(XMLElement* section) {
  string      str, name;
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get sub-element name
    name = elem->Value();
    string elname;

    // numeric
    if (name == "numeric") {
      // create custom
      mjsNumeric* numeric = mjs_addNumeric(spec);

      // write error info
      mjs_setString(numeric->info, ("line " + std::to_string(elem->GetLineNum())).c_str());

      // read attributes
      ReadAttrTxt(elem, "name", elname, true);
      if (mjs_setName(numeric->element, elname.c_str())) {
        throw mjXError(elem, "%s", mjs_getError(spec));
      }
      static constexpr int kMaxNumericSize = 10000000;
      bool                 has_size        = ReadAttrInt(elem, "size", &numeric->size);
      if (has_size && numeric->size < 1) {
        throw mjXError(elem, "custom field size must be positive");
      }
      if (has_size && numeric->size > kMaxNumericSize) {
        throw mjXError(elem, "custom field size exceeds maximum");
      }
      auto maybe_data = ReadAttrVec<double>(elem, "data");
      int  len        = maybe_data.has_value() ? static_cast<int>(maybe_data->size()) : 0;
      if (!has_size) {
        if (len < 1) { throw mjXError(elem, "custom field size must be positive"); }
        if (len > kMaxNumericSize) { throw mjXError(elem, "custom field size exceeds maximum"); }
        numeric->size = len;
      } else if (len > numeric->size) {
        throw mjXError(elem, "attribute 'data' has too much data");
      }

      // copy data, zero-padded to numeric->size
      std::vector<double> data(numeric->size, 0.0);
      if (maybe_data.has_value()) {
        std::copy(maybe_data->begin(), maybe_data->end(), data.begin());
      }
      mjs_setDouble(numeric->data, data.data(), numeric->size);
    }

    // text
    else if (name == "text") {
      // create custom
      mjsText* text = mjs_addText(spec);

      // write error info
      mjs_setString(text->info, ("line " + std::to_string(elem->GetLineNum())).c_str());

      // read attributes
      ReadAttrTxt(elem, "name", elname, true);
      if (mjs_setName(text->element, elname.c_str())) {
        throw mjXError(elem, "%s", mjs_getError(spec));
      }
      string attr_val;
      bool   has_attr = ReadAttrTxt(elem, "data", attr_val, false);

      // check for CDATA block
      const XMLText* cdata_node = nullptr;
      for (const tinyxml2::XMLNode* child = elem->FirstChild(); child;
           child                          = child->NextSibling()) {
        if (const XMLText* text_node = child->ToText()) {
          if (text_node->CData()) {
            if (cdata_node) {
              throw mjXError(elem, "text field cannot have multiple CDATA sections");
            }
            cdata_node = text_node;
          }
        }
      }

      // read CDATA
      if (has_attr && cdata_node) {
        throw mjXError(elem, "text field data cannot be specified as both attribute and CDATA");
      }

      if (has_attr) {
        str = attr_val;
      } else if (cdata_node && cdata_node->Value()) {
        str = cdata_node->Value();
      } else {
        str.clear();
      }

      if (str.empty()) { throw mjXError(elem, "text field cannot be empty"); }

      // copy data
      mjs_setString(text->data, str.c_str());
    }

    // tuple
    else if (name == "tuple") {
      // create custom
      mjsTuple* tuple = mjs_addTuple(spec);

      // write error info
      mjs_setString(tuple->info, ("line " + std::to_string(elem->GetLineNum())).c_str());

      // read attributes
      ReadAttrTxt(elem, "name", elname, true);
      if (mjs_setName(tuple->element, elname.c_str())) {
        throw mjXError(elem, "%s", mjs_getError(spec));
      }

      // read objects and add
      XMLElement*         obj = FirstChildElement(elem);
      std::vector<int>    objtype;
      string              objname = "";
      std::vector<double> objprm;

      while (obj) {
        // get sub-element name
        name = obj->Value();

        // new object
        if (name == "element") {
          // read type, check and assign
          ReadAttrTxt(obj, "objtype", str, true);
          mjtObj otype = (mjtObj)mju_str2Type(str.c_str());
          if (otype == mjOBJ_UNKNOWN) { throw mjXError(obj, "unknown object type"); }
          objtype.push_back(otype);

          // read name and assign
          ReadAttrTxt(obj, "objname", str, true);
          objname += " " + str;

          // read parameter and assign
          double oprm = 0;
          ReadAttr(obj, "prm", 1, &oprm, str);
          objprm.push_back(oprm);
        }

        // advance to next object
        obj = NextSiblingElement(obj);
      }

      mjs_setInt(tuple->objtype, objtype.data(), objtype.size());
      mjs_setStringVec(tuple->objname, objname.c_str());
      mjs_setDouble(tuple->objprm, objprm.data(), objprm.size());
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// visual section parser
void mjXReader::Visual(XMLElement* section) {
  string      name;
  XMLElement* elem;
  mjVisual*   vis = &spec->visual;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get sub-element name
    name = elem->Value();

    // global sub-element
    if (name == "global") {
      ReadAttrTableCore(elem,
                        vis,
                        kGlobalAttrs,
                        kGlobalAttrsN,
                        /*skipnodefault=*/false,
                        /*authored=*/spec);
      if (vis->global.realtime <= 0) { throw mjXError(elem, "realtime must be greater than 0"); }
    }

    // quality sub-element
    else if (name == "quality") {
      ReadAttrTableCore(elem,
                        vis,
                        kQualityAttrs,
                        kQualityAttrsN,
                        false,
                        /*authored=*/spec);
    }

    // headlight sub-element
    else if (name == "headlight") {
      ReadAttrTableCore(elem,
                        vis,
                        kHeadlightAttrs,
                        kHeadlightAttrsN,
                        false,
                        /*authored=*/spec);
    }

    // map sub-element
    else if (name == "map") {
      ReadAttrTableCore(elem,
                        vis,
                        kMapAttrs,
                        kMapAttrsN,
                        false,
                        /*authored=*/spec);
      if (vis->map.znear <= 0) { throw mjXError(elem, "znear must be strictly positive"); }
    }

    // scale sub-element
    else if (name == "scale") {
      ReadAttrTableCore(elem,
                        vis,
                        kScaleAttrs,
                        kScaleAttrsN,
                        false,
                        /*authored=*/spec);
    }

    // rgba sub-element
    else if (name == "rgba") {
      ReadAttrTableCore(elem,
                        vis,
                        kRgbaAttrs,
                        kRgbaAttrsN,
                        false,
                        /*authored=*/spec);
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// asset section parser
void mjXReader::Asset(XMLElement* section, const mjVFS* vfs) {
  string      text, name;
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get sub-element name
    name = elem->Value();

    // get class if specified, otherwise use default0
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getSpecDefault(spec); }

    // texture sub-element
    if (name == "texture") {
      // create texture and parse
      mjsTexture* texture = mjs_addTexture(spec, def);
      OneTexture(elem, texture, vfs);
    }

    // material sub-element
    else if (name == "material") {
      // create material and parse
      mjsMaterial* material = mjs_addMaterial(spec, def);
      OneMaterial(elem, material);
    }

    // mesh sub-element
    else if (name == "mesh") {
      // create mesh and parse
      mjsMesh* mesh = mjs_addMesh(spec, def);
      OneMesh(elem, mesh, vfs);
    }

    // skin sub-element... deprecate ???
    else if (name == "skin") {
      // create skin and parse
      mjsSkin* skin = mjs_addSkin(spec);
      OneSkin(elem, skin, vfs);
    }

    // hfield sub-element
    else if (name == "hfield") {
      // create hfield
      mjsHField* hfield = mjs_addHField(spec);

      // write error info
      mjs_setString(hfield->info, ("line " + std::to_string(elem->GetLineNum())).c_str());

      // mechanical attributes
      ReadAttrTable(elem, hfield, hfield->element, kHfieldAttrs, kHfieldAttrsN);

      // file, resolved against the asset directory
      auto file = ReadAttrFile(elem, "file", vfs, AssetDir());
      if (file.has_value()) { mjs_setString(hfield->file, file->c_str()); }
      NameFromFile(elem, hfield->element, hfield->file);

      // allocate buffer for dynamic hfield, copy user data if given
      if (!file.has_value() && hfield->nrow > 0 && hfield->ncol > 0) {
        int nrow = hfield->nrow;
        int ncol = hfield->ncol;

        // read user data
        auto userdata = ReadAttrVec<float>(elem, "elevation");

        // user data given, copy into data
        if (userdata.has_value()) {
          if (userdata->size() != nrow * ncol) {
            throw mjXError(elem, "elevation data length must match nrow*ncol");
          }

          // copy in reverse row order, so XML string is top-to-bottom
          std::vector<float> flipped(nrow * ncol);
          for (int i = 0; i < nrow; i++) {
            int flip = nrow - 1 - i;
            for (int j = 0; j < ncol; j++) {
              flipped[flip * ncol + j] = userdata->data()[i * ncol + j];
            }
          }

          mjs_setFloat(hfield->userdata, flipped.data(), flipped.size());
        }

        // user data not given, set to 0
        else {
          std::vector<float> zero(nrow * ncol);
          mjs_setFloat(hfield->userdata, zero.data(), zero.size());
        }
      }
    }

    // model sub-element
    else if (name == "model") {
      std::string content_type;
      ReadAttrTxt(elem, "content_type", content_type);

      // parse the child
      std::array<char, 1024> error;
      auto                   filename = modelfiledir_ + ReadAttrFile(elem, "file", vfs).value();

      mjSpec* child =
          mj_parse(filename.c_str(), content_type.c_str(), vfs, error.data(), error.size());
      if (!child) {
        throw mjXError(elem, "could not parse model file with error: %s", error.data());
      }

      // overwrite model name if given
      string modelname = "";
      if (ReadAttrTxt(elem, "name", modelname)) {
        mjs_setString(child->modelname, modelname.c_str());
      }

      // store child spec in model
      mjs_addSpec(spec, child);
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}

// strip the "Error: " prefix from compiler/spec error messages
static const char* stripError(const char* err) {
  if (err && std::strncmp(err, "Error: ", 7) == 0) { return err + 7; }
  return err;
}

// body/world section parser
void mjXReader::Body(XMLElement* section, mjsBody* body, mjsFrame* frame, const mjVFS* vfs) {
  string text, name;

  // sanity check
  if (!body) { throw mjXError(section, "null body pointer"); }

  // no attributes allowed in world body
  if (mjs_getId(body->element) == 0 && section->FirstAttribute() && !frame) {
    throw mjXError(section, "World body cannot have attributes");
  }

  // when parsing the contents of a replicate, body is the temporary subtree
  // that gets attached into the parent body; anything added directly to it
  // (including via frames or includes) is replicated into the parent body
  mjsBody* replicate_body = string(section->Value()) == "replicate" ? body : nullptr;

  // stack of elements to process
  struct BodyFrame {
    XMLElement* elem;
    mjsBody*    body;
    mjsFrame*   frame;
  };
  vector<BodyFrame> stack;
  stack.push_back({FirstChildElement(section), body, frame});

  // iterate over sub-elements; attributes set while parsing parent body
  while (!stack.empty()) {
    XMLElement* elem = stack.back().elem;
    if (!elem) {
      stack.pop_back();
      continue;
    }
    body              = stack.back().body;
    frame             = stack.back().frame;
    stack.back().elem = NextSiblingElement(elem);

    // get sub-element name
    name = elem->Value();

    // get class if specified, otherwise use body
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getDefault(frame ? frame->element : body->element); }

    // inertial sub-element
    if (name == "inertial") {
      // no inertia allowed in world body
      if (mjs_getId(body->element) == 0) { throw mjXError(elem, "World body cannot have inertia"); }
      body->explicitinertial = true;
      ReadAttrTable(elem, body, body->element, kInertialAttrs, kInertialAttrsN);
      ReadQuat(elem, "quat", body->iquat, text);
      ReadAlternative(elem, body->ialt);
      ReadAttr(elem, "fullinertia", 6, body->fullinertia, text);

      // the inertial frame is relative to the enclosing frame
      mjCBody* bodyC = static_cast<mjCBody*>(body->element);
      bodyC->iframe  = frame ? static_cast<mjCFrame*>(frame->element) : nullptr;
    }

    // joint sub-element
    else if (name == "joint") {
      // no joints allowed in world body
      if (mjs_getId(body->element) == 0) { throw mjXError(elem, "World body cannot have joints"); }

      // joints would be replicated into the parent body
      if (replicate_body && body == replicate_body) {
        throw mjXError(elem, "joint cannot be a direct child of replicate");
      }

      // create joint and parse
      mjsJoint* joint = mjs_addJoint(body, def);
      OneJoint(elem, joint);
      mjs_setFrame(joint->element, frame);
    }

    // freejoint sub-element
    else if (name == "freejoint") {
      // no joints allowed in world body
      if (mjs_getId(body->element) == 0) { throw mjXError(elem, "World body cannot have joints"); }

      // joints would be replicated into the parent body
      if (replicate_body && body == replicate_body) {
        throw mjXError(elem, "joint cannot be a direct child of replicate");
      }

      // create free joint without defaults
      mjsJoint* joint = mjs_addFreeJoint(body);
      mjs_setFrame(joint->element, frame);

      // save defaults after creation, to make sure writing is ok
      mjs_setDefault(joint->element, def);

      // read attributes
      string name;
      if (ReadAttrTxt(elem, "name", name)) {
        if (mjs_setName(joint->element, name.c_str())) {
          throw mjXError(elem, "%s", mjs_getError(spec));
        }
      }
      ReadAttrInt(elem, "group", &joint->group);
      MapValue(elem, "align", &joint->align, FalseTrueAuto_map, 3);
    }

    // geom sub-element
    else if (name == "geom") {
      // create geom and parse
      mjsGeom* geom = mjs_addGeom(body, def);
      OneGeom(elem, geom);
      mjs_setFrame(geom->element, frame);
    }

    // site sub-element
    else if (name == "site") {
      // create site and parse
      mjsSite* site = mjs_addSite(body, def);
      OneSite(elem, site);
      mjs_setFrame(site->element, frame);
    }

    // camera sub-element
    else if (name == "camera") {
      // create camera and parse
      mjsCamera* camera = mjs_addCamera(body, def);
      OneCamera(elem, camera);
      mjs_setFrame(camera->element, frame);
    }

    // light sub-element
    else if (name == "light") {
      // create light and parse
      mjsLight* light = mjs_addLight(body, def);
      OneLight(elem, light);
      mjs_setFrame(light->element, frame);
    }

    // plugin sub-element
    else if (name == "plugin") {
      OnePlugin(elem, &(body->plugin));
    }

    // flexcomp sub-element
    else if (name == "flexcomp") {
      // parse flexcomp
      OneFlexcomp(elem, body, frame, vfs);
    }

    // frame sub-element
    else if (name == "frame") {
      // read childdef
      bool has_childclass = ReadAttrTxt(elem, "childclass", text);

      const mjsDefault* childdef = has_childclass ? mjs_findDefault(spec, text.c_str()) : nullptr;
      if (has_childclass && !childdef) { throw mjXError(elem, "unknown default childclass"); }

      // create frame
      mjsFrame* pframe = mjs_addFrame(body, frame);
      mjs_setString(pframe->info, ("line " + std::to_string(elem->GetLineNum())).c_str());
      mjs_setDefault(pframe->element, childdef ? childdef : def);

      // read attributes
      string name, childclass;
      if (ReadAttrTxt(elem, "name", name)) {
        if (mjs_setName(pframe->element, name.c_str())) {
          throw mjXError(elem, "%s", mjs_getError(spec));
        }
      }
      if (ReadAttrTxt(elem, "childclass", childclass)) {
        mjs_setString(pframe->childclass, childclass.c_str());
      }
      ReadAttr(elem, "pos", 3, pframe->pos, text);
      ReadQuat(elem, "quat", pframe->quat, text);
      ReadAlternative(elem, pframe->alt);
      int nscale = ReadAttr(elem, "scale", 3, pframe->scale, text, false, false);
      if (nscale == 1) {
        pframe->scale[1] = pframe->scale[2] = pframe->scale[0];
      } else if (nscale == 2) {
        throw mjXError(elem, "attribute 'scale' must have 1 or 3 values");
      }
      MapValue(elem, "scaling", &pframe->scaling, scaling_map, scaling_sz);
      ReadAttr(elem, "massscale", 1, &pframe->massscale, text);

      stack.push_back({FirstChildElement(elem), body, pframe});
    }

    // replicate sub-element
    else if (name == "replicate") {
      int    count;
      double offset[3] = {0, 0, 0};
      double euler[3]  = {0, 0, 0};
      string separator = "";
      ReadAttr(elem, "count", 1, &count, text, true);
      ReadAttr(elem, "offset", 3, offset, text);
      ReadAttr(elem, "euler", 3, euler, text);
      ReadAttrTxt(elem, "sep", separator);

      // store rotation difference
      mjsOrientation alt;
      mjs_defaultOrientation(&alt);
      alt.type = mjORIENTATION_EULER;
      mjuu_copyvec(alt.euler, euler, 3);
      double rotation[4] = {1, 0, 0, 0};
      mjs_resolveOrientation(rotation, spec->compiler.degree, spec->compiler.eulerseq, &alt);

      // read childdef
      bool has_childclass = ReadAttrTxt(elem, "childclass", text);

      const mjsDefault* childdef = has_childclass ? mjs_findDefault(spec, text.c_str()) : nullptr;
      if (has_childclass && !childdef) { throw mjXError(elem, "unknown default childclass"); }

      // create subtree
      mjsBody* subtree = mjs_addBody(body, childdef);
      double   pos[3]  = {0, 0, 0};
      double   quat[4] = {1, 0, 0, 0};

      // parent frame that will be used to attach the subtree
      mjsFrame* pframe = mjs_addFrame(subtree, nullptr);
      mjs_setDefault(pframe->element, childdef ? childdef : def);
      mjs_setString(pframe->info, ("line = " + std::to_string(elem->GetLineNum())).c_str());

      // parse subtree
      Body(elem, subtree, pframe, vfs);

      // update pframe and attach
      for (int i = 0; i < count; i++) {
        // overwrite orientation to increase precision
        alt.euler[0] = i * euler[0];
        alt.euler[1] = i * euler[1];
        alt.euler[2] = i * euler[2];
        mjs_resolveOrientation(quat, spec->compiler.degree, spec->compiler.eulerseq, &alt);

        // set position and orientation
        mjuu_setvec(pframe->pos, pos[0], pos[1], pos[2]);
        mjuu_setvec(pframe->quat, quat[0], quat[1], quat[2], quat[3]);

        // accumulate rotation
        mjuu_frameaccum(pos, quat, offset, rotation);

        // process suffix
        string suffix = separator;
        UpdateString(suffix, count, i);

        // attach to parent, inside the enclosing frame
        if (!mjs_attach(frame ? frame->element : body->element,
                        pframe->element,
                        /*prefix=*/"",
                        suffix.c_str())) {
          throw mjXError(elem, "%s", stripError(mjs_getError(spec)));
        }
      }

      // delete subtree
      if (mjs_delete(spec, subtree->element)) {
        throw mjXError(elem, "%s", stripError(mjs_getError(spec)));
      }
    }

    // body sub-element
    else if (name == "body") {
      // read childdef
      bool has_childclass = ReadAttrTxt(elem, "childclass", text);

      const mjsDefault* childdef = has_childclass ? mjs_findDefault(spec, text.c_str()) : nullptr;
      if (has_childclass && !childdef) { throw mjXError(elem, "unknown default childclass"); }

      // create child body
      mjsBody* child = mjs_addBody(body, childdef);
      if (!child) { throw mjXError(elem, "null body pointer"); }
      mjs_setString(child->info, string("line " + std::to_string(elem->GetLineNum())).c_str());

      // set default from class or childclass
      mjs_setDefault(child->element, childdef ? childdef : def);

      // mechanical attributes
      ReadAttrTable(elem, child, child->element, kBodyAttrs, kBodyAttrsN);
      string childclass;
      if (ReadAttrTxt(elem, "childclass", childclass)) {
        mjs_setString(child->childclass, childclass.c_str());
      }

      // orientation alternatives
      ReadQuat(elem, "quat", child->quat, text);
      ReadAlternative(elem, child->alt);

      // add frame
      mjs_setFrame(child->element, frame);

      // push child onto traversal stack
      stack.push_back({FirstChildElement(elem), child, nullptr});
    }

    // attachment
    else if (name == "attach") {
      string model_name, child_name, prefix;
      bool   has_model = ReadAttrTxt(elem, "model", model_name, /*required=*/false);
      bool   has_body  = ReadAttrTxt(elem, "body", child_name, /*required=*/false);
      bool   has_frame = ReadAttrTxt(elem, "frame", child_name, /*required=*/false);
      ReadAttrTxt(elem, "prefix", prefix, /*required=*/true);

      mjtObj type = mjOBJ_UNKNOWN;
      if (has_body)
        type = mjOBJ_BODY;
      else if (has_frame)
        type = mjOBJ_FRAME;

      mjsElement* source_elem = nullptr;
      if (!has_model) {  // Self-attach
        if (type == mjOBJ_UNKNOWN) {
          throw mjXError(elem,
                         "either 'body' or 'frame' attribute must be specified for self-attach");
        }

        // check for name collision in the current spec
        string full_name = prefix + child_name;
        if (mjs_findElement(spec, type, full_name.c_str())) {
          throw mjXError(elem, "cannot self-attach: element %s already exists", full_name.c_str());
        }
        source_elem = mjs_findElement(spec, type, child_name.c_str());
        if (!source_elem) {
          throw mjXError(elem,
                         "%s",
                         (string("could not find ") +
                          mju_type2Str(type) +
                          " '" +
                          child_name +
                          "' in the current model for self-attachment")
                             .c_str());
        }
      } else {  // Attach from external model asset
        // Check for name collision in the current spec
        if (!child_name.empty()) {
          string full_name = prefix + child_name;
          if (mjs_findElement(spec, type, full_name.c_str())) {
            throw mjXError(elem,
                           "%s",
                           (string("cannot attach: element ") +
                            child_name +
                            " already exists with prefix " +
                            prefix)
                               .c_str());
          }
        }

        mjSpec* asset = mjs_findSpec(spec, model_name.c_str());
        if (!asset) { throw mjXError(elem, "could not find model '%s'", model_name.c_str()); }

        if (type == mjOBJ_UNKNOWN) {  // Attach world body contents
          source_elem = asset->element;
        } else {  // Attach specific body or frame
          source_elem = mjs_findElement(asset, type, child_name.c_str());
          if (!source_elem) {
            throw mjXError(elem,
                           "%s",
                           (string("could not find ") +
                            mju_type2Str(type) +
                            " '" +
                            child_name +
                            "' in model asset '" +
                            model_name +
                            "'")
                               .c_str());
          }
        }
      }

      mjsFrame* pframe = frame ? frame : mjs_addFrame(body, nullptr);
      // Set default for the new frame from the current context
      mjs_setDefault(pframe->element, mjs_getDefault(frame ? frame->element : body->element));
      mjs_setString(pframe->info, ("line = " + std::to_string(elem->GetLineNum())).c_str());

      if (!mjs_attach(pframe->element, source_elem, prefix.c_str(), "")) {
        throw mjXError(elem, "%s", stripError(mjs_getError(spec)));
      }
    }

    // no match
    else {
      throw mjXError(elem, "unrecognized model element '%s'", name.c_str());
    }
  }
}


// contact section parser
void mjXReader::Contact(XMLElement* section) {
  string      text, name;
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get sub-element name
    name = elem->Value();

    // get class if specified, otherwise use default0
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getSpecDefault(spec); }

    // geom pair to include
    if (name == "pair") {
      // create pair and parse
      mjsPair* pair = mjs_addPair(spec, def);
      OnePair(elem, pair);
    }

    // body pair to exclude
    else if (name == "exclude") {
      mjsExclude* exclude = mjs_addExclude(spec);

      // write error info
      mjs_setString(exclude->info, ("line " + std::to_string(elem->GetLineNum())).c_str());

      // read attributes
      ReadAttrTable(elem, exclude, exclude->element, kExcludeAttrs, kExcludeAttrsN);
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// constraint section parser
void mjXReader::Equality(XMLElement* section) {
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get class if specified, otherwise use default0
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getSpecDefault(spec); }

    // create equality constraint and parse
    mjsEquality* equality = mjs_addEquality(spec, def);
    OneEquality(elem, equality);

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// deformable section parser
void mjXReader::Deformable(XMLElement* section, const mjVFS* vfs) {
  string      name;
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get sub-element name
    name = elem->Value();

    // get class if specified, otherwise use default0
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getSpecDefault(spec); }

    // flex sub-element
    if (name == "flex") {
      // create flex and parse
      mjsFlex* flex = mjs_addFlex(spec);
      OneFlex(elem, flex);
    }

    // skin sub-element
    else if (name == "skin") {
      // create skin and parse
      mjsSkin* skin = mjs_addSkin(spec);
      OneSkin(elem, skin, vfs);
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// tendon section parser
void mjXReader::Tendon(XMLElement* section) {
  string      text, text1;
  XMLElement* elem;
  double      data;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get class if specified, otherwise use default0
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getSpecDefault(spec); }

    // create tendon and parse
    mjsTendon* tendon = mjs_addTendon(spec, def);
    OneTendon(elem, tendon);

    // process wrap sub-elements
    XMLElement* sub = FirstChildElement(elem);
    while (sub) {
      // get wrap type
      string   type = sub->Value();
      mjsWrap* wrap;
      ;

      // read attributes depending on type
      if (type == "site") {
        ReadAttrTxt(sub, "site", text, true);
        wrap = mjs_wrapSite(tendon, text.c_str());
      }

      else if (type == "geom") {
        ReadAttrTxt(sub, "geom", text, true);
        if (!ReadAttrTxt(sub, "sidesite", text1)) { text1.clear(); }
        wrap = mjs_wrapGeom(tendon, text.c_str(), text1.c_str());
      }

      else if (type == "pulley") {
        ReadAttr(sub, "divisor", 1, &data, text, true);
        wrap = mjs_wrapPulley(tendon, data);
      }

      else if (type == "joint") {
        ReadAttrTxt(sub, "joint", text, true);
        ReadAttr(sub, "coef", 1, &data, text1, true);
        wrap = mjs_wrapJoint(tendon, text.c_str(), data);
      }

      else {
        throw mjXError(sub, "unknown wrap type");  // SHOULD NOT OCCUR
      }

      mjs_setString(wrap->info, ("line " + std::to_string(sub->GetLineNum())).c_str());

      // advance to next sub-element
      sub = NextSiblingElement(sub);
    }

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// actuator section parser
void mjXReader::Actuator(XMLElement* section) {
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    // get class if specified, otherwise use default0
    const mjsDefault* def = GetClass(elem);
    if (!def) { def = mjs_getSpecDefault(spec); }

    // create actuator and parse
    mjsActuator* actuator = mjs_addActuator(spec, def);
    OneActuator(elem, actuator, def);

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// sensor section parser
void mjXReader::Sensor(XMLElement* section) {
  int         n;
  XMLElement* elem = FirstChildElement(section);
  while (elem) {
    // create sensor, get string type
    mjsSensor* sensor = mjs_addSensor(spec);
    string     type   = elem->Value();
    string     text, objname, refname;

    // mechanical attributes shared by all sensor types
    ReadAttrTable(elem, sensor, sensor->element, kSensorBaseAttrs, kSensorBaseAttrsN);

    // common robotic sensors, attached to a site
    // sensors fully described by the schema: constants and references
    bool dispatched = false;
    for (int i = 0; i < kSensorDispatchN; i++) {
      if (type == kSensorDispatch[i].tag) {
        ReadAttrTable(elem, sensor, sensor->element, kSensorDispatch[i].rows, kSensorDispatch[i].n);
        dispatched = true;
        break;
      }
    }
    if (dispatched) {
    } else if (type == "rangefinder") {
      sensor->type  = mjSENS_RANGEFINDER;
      bool use_site = ReadAttrTxt(elem, "site", objname, false);
      ReadAttrTxt(elem, "camera", objname, false);
      sensor->objtype = use_site ? mjOBJ_SITE : mjOBJ_CAMERA;

      // process data specification (intprm[0])
      std::vector<int> raydata(mjNRAYDATA);

      int dataspec = 1 << mjRAYDATA_DIST;
      int nkeys    = MapValues(elem, "data", raydata.data(), raydata_map, mjNRAYDATA);
      if (nkeys) {
        dataspec = 1 << raydata[0];

        // check ordering while adding bits to dataspec
        for (int i = 1; i < nkeys; ++i) {
          if (raydata[i] <= raydata[i - 1]) {
            std::string correct_order;
            for (int j = 0; j < mjNRAYDATA; ++j) {
              correct_order += raydata_map[j].key;
              if (j < mjNRAYDATA - 1) correct_order += ", ";
            }
            throw mjXError(elem, "data attributes must be in order: %s", correct_order.c_str());
          }
          dataspec |= 1 << raydata[i];
        }
      }
      sensor->intprm[0] = dataspec;
    } else if (type == "distance" || type == "normal" || type == "fromto") {
      bool has_body1 = ReadAttrTxt(elem, "body1", objname);
      ReadAttrTxt(elem, "geom1", objname);
      sensor->objtype = has_body1 ? mjOBJ_BODY : mjOBJ_GEOM;
      bool has_body2  = ReadAttrTxt(elem, "body2", refname);
      ReadAttrTxt(elem, "geom2", refname);
      sensor->reftype = has_body2 ? mjOBJ_BODY : mjOBJ_GEOM;
      if (type == "distance") {
        sensor->type = mjSENS_GEOMDIST;
      } else if (type == "normal") {
        sensor->type = mjSENS_GEOMNORMAL;
      } else {
        sensor->type = mjSENS_GEOMFROMTO;
      }
    } else if (type == "contact") {
      // first matching criterion
      bool has_site     = ReadAttrTxt(elem, "site", objname);
      bool has_body1    = ReadAttrTxt(elem, "body1", objname);
      bool has_subtree1 = ReadAttrTxt(elem, "subtree1", objname);
      bool has_geom1    = ReadAttrTxt(elem, "geom1", objname);
      if (has_site) {
        sensor->objtype = mjOBJ_SITE;
      } else if (has_body1) {
        sensor->objtype = mjOBJ_BODY;
      } else if (has_subtree1) {
        sensor->objtype = mjOBJ_XBODY;
      } else if (has_geom1) {
        sensor->objtype = mjOBJ_GEOM;
      } else {
        sensor->objtype = mjOBJ_UNKNOWN;
      }

      // second matching criterion
      bool has_body2    = ReadAttrTxt(elem, "body2", refname);
      bool has_subtree2 = ReadAttrTxt(elem, "subtree2", refname);
      bool has_geom2    = ReadAttrTxt(elem, "geom2", refname);
      if (has_body2) {
        sensor->reftype = mjOBJ_BODY;
      } else if (has_subtree2) {
        sensor->reftype = mjOBJ_XBODY;
      } else if (has_geom2) {
        sensor->reftype = mjOBJ_GEOM;
      } else {
        sensor->reftype = mjOBJ_UNKNOWN;
      }

      // process data specification (intprm[0])
      int              dataspec = 1 << mjCONDATA_FOUND;
      std::vector<int> condata(mjNCONDATA);
      int              nkeys = MapValues(elem, "data", condata.data(), condata_map, mjNCONDATA);
      if (nkeys) {
        dataspec = 1 << condata[0];

        // check ordering while adding bits to dataspec
        for (int i = 1; i < nkeys; ++i) {
          if (condata[i] <= condata[i - 1]) {
            std::string correct_order;
            for (int j = 0; j < mjNCONDATA; ++j) {
              correct_order += condata_map[j].key;
              if (j < mjNCONDATA - 1) correct_order += ", ";
            }
            throw mjXError(elem, "data attributes must be in order: %s", correct_order.c_str());
          }
          dataspec |= 1 << condata[i];
        }
      }
      sensor->intprm[0] = dataspec;

      // reduction type (intprm[1])
      sensor->intprm[1] = 0;
      if (MapValue(elem, "reduce", &n, reduce_map, reduce_sz)) { sensor->intprm[1] = n; }

      // number of contacts (intprm[2])
      sensor->intprm[2] = 1;
      ReadAttrInt(elem, "num", &sensor->intprm[2]);
      if (sensor->intprm[2] <= 0) { throw mjXError(elem, "'num' must be positive in sensor"); }

      // sensor type
      sensor->type = mjSENS_CONTACT;
    } else if (type == "user") {
      sensor->type = mjSENS_USER;
      ReadAttrTxt(elem, "objname", objname);
      if (ReadAttrTxt(elem, "objtype", text)) {
        sensor->objtype = (mjtObj)mju_str2Type(text.c_str());
      }
      ReadAttrInt(elem, "dim", &sensor->dim, true);

      // keywords
      if (MapValue(elem, "needstage", &n, stage_map, stage_sz)) { sensor->needstage = (mjtStage)n; }
      if (MapValue(elem, "datatype", &n, datatype_map, datatype_sz)) {
        sensor->datatype = (mjtDataType)n;
      }
    }

    // tactile sensor
    if (type == "tactile") {
      sensor->type    = mjSENS_TACTILE;
      sensor->reftype = mjOBJ_GEOM;
      ReadAttrTxt(elem, "geom", refname, /*required=*/true);

      // associate the sensor with a mesh
      sensor->objtype = mjOBJ_MESH;
      ReadAttrTxt(elem, "mesh", objname, /*required=*/true);
      mjs_setString(sensor->objname, objname.c_str());
    }

    else if (type == "plugin") {
      sensor->type = mjSENS_PLUGIN;
      OnePlugin(elem, &sensor->plugin);
      ReadAttrTxt(elem, "objtype", text);
      sensor->objtype = (mjtObj)mju_str2Type(text.c_str());
      ReadAttrTxt(elem, "objname", objname);
      if (sensor->objtype != mjOBJ_UNKNOWN && objname.empty()) {
        throw mjXError(elem, "objtype is specified but objname is not");
      }
      if (sensor->objtype == mjOBJ_UNKNOWN && !objname.empty()) {
        throw mjXError(elem, "objname is specified but objtype is not");
      }
      if (ReadAttrTxt(elem, "reftype", text)) {
        sensor->reftype = (mjtObj)mju_str2Type(text.c_str());
      }
      ReadAttrTxt(elem, "refname", refname);
      if (sensor->reftype != mjOBJ_UNKNOWN && refname.empty()) {
        throw mjXError(elem, "reftype is specified but refname is not");
      }
      if (sensor->reftype == mjOBJ_UNKNOWN && !refname.empty()) {
        throw mjXError(elem, "refname is specified but reftype is not");
      }
    }

    if (!objname.empty()) { mjs_setString(sensor->objname, objname.c_str()); }

    if (!refname.empty()) { mjs_setString(sensor->refname, refname.c_str()); }

    // write info
    mjs_setString(sensor->info, ("line " + std::to_string(elem->GetLineNum())).c_str());

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// keyframe section parser
void mjXReader::Keyframe(XMLElement* section) {
  XMLElement* elem;

  // iterate over child elements
  elem = FirstChildElement(section);
  while (elem) {
    string name = "";

    // add keyframe
    mjsKey* key = mjs_addKey(spec);

    // read name: set even when the attribute is absent
    ReadAttrTxt(elem, "name", name);
    if (mjs_setName(key->element, name.c_str())) { throw mjXError(elem, "%s", mjs_getError(spec)); }

    // mechanical attributes
    ReadAttrTable(elem, key, key->element, kKeyAttrs, kKeyAttrsN);

    // advance to next element
    elem = NextSiblingElement(elem);
  }
}


// get defaults class
const mjsDefault* mjXReader::GetClass(XMLElement* section) {
  string text;

  if (!ReadAttrTxt(section, "class", text)) { return nullptr; }

  const mjsDefault* def = mjs_findDefault(spec, text.c_str());
  if (!def) {
    throw mjXError(section, string("unknown default class name '" + text + "'").c_str());
  }
  return def;
}

void mjXReader::SetModelFileDir(const string& modelfiledir) {
  modelfiledir_ = FilePath(modelfiledir);
}

void mjXReader::SetAssetDir(const string& assetdir) {
  assetdir_ = FilePath(assetdir);
}

void mjXReader::SetMeshDir(const string& meshdir) {
  meshdir_ = FilePath(meshdir);
}

void mjXReader::SetTextureDir(const string& texturedir) {
  texturedir_ = FilePath(texturedir);
}

FilePath mjXReader::AssetDir() const {
  return modelfiledir_ + assetdir_;
}

FilePath mjXReader::MeshDir() const {
  if (meshdir_.empty()) { return AssetDir(); }
  return modelfiledir_ + meshdir_;
}
FilePath mjXReader::TextureDir() const {
  if (texturedir_.empty()) { return AssetDir(); }
  return modelfiledir_ + texturedir_;
}
