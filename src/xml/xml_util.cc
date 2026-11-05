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

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "cc/array_safety.h"
#include "engine/engine_util_errmem.h"
#include "user/user_resource.h"
#include "user/user_util.h"
#include "xml/xml_util.h"
#include "xml/xml_numeric_format.h"
#include "tinyxml2.h"

namespace {

using mujoco::user::FilePath;
using std::function;
using std::optional;
using std::string;
using std::stringstream;
using std::vector;
using tinyxml2::XMLAttribute;
using tinyxml2::XMLElement;

namespace mju = ::mujoco::util;

template <typename T>
static optional<T> ParseInfOrNan(const string& s) {
  const char* str = s.c_str();
  if constexpr (std::is_floating_point_v<T>) {
    T sign = 1;
    if (s.size() == 4 && s[0] == '-') {
      sign = -1;
      ++str;
    } else if (s.size() != 3) {
      return std::nullopt;
    }
    if (std::numeric_limits<T>::has_infinity &&
        (str[0] == 'i' || str[0] == 'I') &&
        (str[1] == 'n' || str[1] == 'N') &&
        (str[2] == 'f' || str[2] == 'F')) {
      return sign * std::numeric_limits<T>::infinity();
    } else if (std::numeric_limits<T>::has_quiet_NaN &&
               (str[0] == 'n' || str[0] == 'N') &&
               (str[1] == 'a' || str[1] == 'A') &&
               (str[2] == 'n' || str[2] == 'N')) {
      return sign * std::numeric_limits<T>::quiet_NaN();
    }
  }
  return std::nullopt;
}

FilePath ResolveFilePath(XMLElement*     e,
                         const FilePath& filename,
                         const FilePath& dir,
                         const mjVFS*    vfs) {
  string path = "";
  if (filename.IsAbs()) { return filename; }

  // TODO(kylebayes): We first look in the base model directory for files to
  // remain backwards compatible.
  FilePath    fullname = dir + filename;
  mjResource* resource = mju_openResource("", fullname.c_str(), vfs, nullptr, 0);
  if (resource != nullptr) {
    mju_closeResource(resource);
    return filename;
  }

  XMLElement* parent = e->Parent()->ToElement();
  for (; parent; parent = parent->Parent()->ToElement()) {
    if (!std::strcmp(parent->Value(), "include")) {
      auto file_attr = mjXUtil::ReadAttrStr(parent, "dir", false);
      if (file_attr.has_value()) { path = file_attr.value(); }
      break;
    }
  }
  return FilePath(path) + filename;
}

// Thrown inside mju_getXMLDependencies so C++ stack objects are destroyed
// before mju_error (which may longjmp in language bindings) is called.
struct XmlDepsError {
  char message[1024];
};

template <typename... Args>
[[noreturn]] void ThrowXmlDepsError(const char* format, Args... args) {
  XmlDepsError err;
  mju::sprintf_arr(err.message, format, args...);
  throw err;
}

void GetXMLDependenciesImpl(const char* filename, mjStringVec* dependencies);

// Opens and parses the XML resource `filename` into `doc`. Returns the
// directory of the resource, or nullopt if it cannot be opened. Throws
// XmlDepsError if it cannot be read or parsed.
optional<FilePath> ParseXMLResource(const string& filename, tinyxml2::XMLDocument& doc) {
  // Open through the resource provider so this works against any registered
  // backend (OS file system, VFS, HTTP, "github:", ...).
  mjResource* resource = mju_openResource("", filename.c_str(), nullptr, nullptr, 0);
  if (resource == nullptr) { return std::nullopt; }

  // Read the XML bytes from the resource.
  const void* buffer = nullptr;
  int         size   = mju_readResource(resource, &buffer);
  if (size < 0 || !size) {
    mju_closeResource(resource);
    ThrowXmlDepsError("Could not read '%s'", filename.c_str());
  }

  // Capture the directory while the resource is still open.
  const char* dir  = nullptr;
  int         ndir = 0;
  mju_getResourceDir(resource, &dir, &ndir);
  FilePath resource_dir(string(dir, ndir));

  // Parse from buffer and close (the parsed DOM is independent of the
  // resource buffer once Parse returns).
  tinyxml2::XMLError err = doc.Parse(static_cast<const char*>(buffer), static_cast<size_t>(size));
  mju_closeResource(resource);

  if (err != tinyxml2::XML_SUCCESS) {
    ThrowXmlDepsError("Problem reading XML file '%s': %s",
                      filename.c_str(),
                      doc.ErrorStr() ? doc.ErrorStr() : "");
  }
  if (!doc.RootElement()) {
    ThrowXmlDepsError("XML root element not found in '%s'", filename.c_str());
  }
  return resource_dir;
}

void AccumulateFiles(std::unordered_set<string>& files,
                     XMLElement*                 root,
                     const FilePath&             model_dir) {
  optional<FilePath> asset_dir;
  optional<FilePath> mesh_dir;
  optional<FilePath> texture_dir;
  std::set<string>   model_files;
  std::set<string>   texture_files;
  std::set<string>   mesh_files;
  std::set<string>   hfield_files;

  auto accumulate_files = [&](const std::set<string>& candidate_files, optional<FilePath> prefix) {
    for (const auto& file : candidate_files) {
      FilePath file_with_prefix =
          !prefix.has_value() ? FilePath(file) : prefix.value() + FilePath(file);
      if (file_with_prefix.IsAbs()) {
        files.insert(file_with_prefix.Str());
      } else {
        // Else insert dir_path / prefix / file.
        auto full_path = model_dir + file_with_prefix;
        files.insert(full_path.Str());
      }
    }
  };

  // Included files are part of the model: ExpandIncludeXML in xml.cc copies
  // the included file's elements into the including document, where they are
  // read as if written in place of the <include>. They are walked here the
  // same way, so the model's <compiler> directories apply to them and their
  // assets are relative to the model directory. Each element is paired with
  // the directory of its file, for looking up nested includes.
  vector<std::unique_ptr<tinyxml2::XMLDocument>> included;
  std::stack<std::pair<XMLElement*, FilePath>>   elements;
  elements.push({root, model_dir});
  while (!elements.empty()) {
    auto [elem, dir] = elements.top();
    elements.pop();

    if (!std::strcmp(elem->Value(), "include")) {
      // include elements don't have children.
      auto file_attr = mjXUtil::ReadAttrFile(elem, "file", nullptr);
      if (!file_attr.has_value()) { continue; }

      // A relative <include> path can be relative to the model directory
      // (legacy) or to the including file's directory; try them in the same
      // order as ExpandIncludeXML in xml.cc.
      vector<FilePath> candidates = {file_attr.value()};
      if (!file_attr->IsAbs()) { candidates = {model_dir + *file_attr, dir + *file_attr}; }

      // Skip if this file was already included (unlike ExpandIncludeXML, which
      // errors on duplicate includes, the dependency scanner simply records
      // each file once and breaks A -> B -> A include cycles).
      if (std::any_of(candidates.begin(), candidates.end(), [&](const FilePath& path) {
            return files.count(path.Str());
          })) {
        continue;
      }

      // Try each candidate path until one opens, parse it into `doc`, and
      // record the resolved path in `files`.
      auto               doc = std::make_unique<tinyxml2::XMLDocument>();
      optional<FilePath> include_dir;
      for (const FilePath& path : candidates) {
        include_dir = ParseXMLResource(path.Str(), *doc);
        if (include_dir.has_value()) {
          files.insert(path.Str());
          break;
        }
      }
      if (!include_dir.has_value()) {
        if (candidates.size() > 1 && candidates[0].Str() != candidates[1].Str()) {
          ThrowXmlDepsError("Could not open '%s' or '%s'",
                            candidates[0].c_str(),
                            candidates[1].c_str());
        } else {
          ThrowXmlDepsError("Could not open '%s'", candidates[0].c_str());
        }
      }

      // Walk the included elements as part of this model. Nested includes are
      // looked up relative to the included file's directory.
      XMLElement* child = doc->RootElement()->FirstChildElement();
      while (child) {
        elements.push({child, include_dir.value()});
        child = child->NextSiblingElement();
      }

      // Keep the document alive while its elements are on the stack.
      included.push_back(std::move(doc));
      continue;
    } else if (!std::strcmp(elem->Value(), "model")) {
      // model elements don't have children.
      auto file_attr = mjXUtil::ReadAttrFile(elem, "file", nullptr);
      if (file_attr.has_value()) { model_files.insert(file_attr->Str()); }
      continue;
    } else if (!std::strcmp(elem->Value(), "compiler")) {
      auto assetdir_str = mjXUtil::ReadAttrStr(elem, "assetdir", false);
      if (assetdir_str.has_value()) asset_dir = FilePath(assetdir_str.value());
      auto meshdir_str = mjXUtil::ReadAttrStr(elem, "meshdir", false);
      if (meshdir_str.has_value()) mesh_dir = FilePath(meshdir_str.value());
      auto texturedir_str = mjXUtil::ReadAttrStr(elem, "texturedir", false);
      if (texturedir_str.has_value()) texture_dir = FilePath(texturedir_str.value());

      // compiler elements don't have children.
      continue;
    } else if (!std::strcmp(elem->Value(), "mesh") ||
               !std::strcmp(elem->Value(), "flexcomp") ||
               !std::strcmp(elem->Value(), "skin")) {
      // mesh elements don't have children.
      auto file_attr = mjXUtil::ReadAttrFile(elem, "file", nullptr);
      if (file_attr.has_value()) { mesh_files.insert(file_attr->Str()); }
      continue;
    } else if (!std::strcmp(elem->Value(), "hfield")) {
      // hfield elements don't have children.
      auto file_attr = mjXUtil::ReadAttrFile(elem, "file", nullptr);
      if (file_attr.has_value()) { hfield_files.insert(file_attr->Str()); }
      continue;
    } else if (!std::strcmp(elem->Value(), "texture")) {
      static const char* attributes[] =
          {"file", "fileright", "fileup", "fileleft", "filedown", "filefront", "fileback"};
      for (const auto& attribute : attributes) {
        auto file_attr = mjXUtil::ReadAttrFile(elem, attribute, nullptr);
        if (file_attr.has_value()) { texture_files.insert(file_attr->Str()); }
      }
    }

    XMLElement* child = elem->FirstChildElement();
    while (child) {
      elements.push({child, dir});
      child = child->NextSiblingElement();
    }
  }

  // TODO(shaves): When we have resource decoders implemented they should have a
  // "get dependencies" function to call here. For non XML types we assume they
  // have no dependencies here.

  // First resolve all dependent models; each is a separate model with its own
  // <compiler> directories.
  for (const auto& file : model_files) {
    mjStringVec subdeps;
    FilePath    full_path = model_dir + FilePath(file);
    GetXMLDependenciesImpl(full_path.Str().c_str(), &subdeps);
    for (const auto& subdep : subdeps) { files.insert(subdep); }
  }
  // Then for each non MJCF resource file, add them to the set of files using their respective
  // compiler prefixes (if they exist).
  accumulate_files(texture_files, texture_dir.has_value() ? texture_dir : asset_dir);
  accumulate_files(mesh_files, mesh_dir.has_value() ? mesh_dir : asset_dir);
  accumulate_files(hfield_files, asset_dir);
}

void GetXMLDependenciesImpl(const char* filename, mjStringVec* dependencies) {
  tinyxml2::XMLDocument doc;
  optional<FilePath>    model_dir = ParseXMLResource(filename, doc);
  if (!model_dir.has_value()) { ThrowXmlDepsError("Could not open '%s'", filename); }

  std::unordered_set<string> files = {filename};
  AccumulateFiles(files, doc.RootElement(), model_dir.value());

  *dependencies = {files.begin(), files.end()};
}
}  // namespace

//---------------------------------- utility functions ---------------------------------------------

// error string copy
void mjCopyError(char* dst, const char* src, int maxlen) {
  if (dst && maxlen > 0) {
    strncpy(dst, src, maxlen);
    dst[maxlen - 1] = 0;
  }
}

void mju_getXMLDependencies(const char* filename, mjStringVec* dependencies) {
  char error[1024] = "";
  try {
    GetXMLDependenciesImpl(filename, dependencies);
    return;
  } catch (const XmlDepsError& err) { mjCopyError(error, err.message, sizeof(error)); }
  // Called outside the catch block: a longjmp out of a handler would skip the
  // destruction of the exception object itself.
  mju_error("%s", error);
}

// error constructor
mjXError::mjXError(const XMLElement* elem, const char* msg, const char* str, int pos) {
  char temp[500];

  // construct error message
  mju::sprintf_arr(message, "XML Error");
  if (msg) {
    mju::sprintf_arr(temp, msg, str, pos);
    mju::strcat_arr(message, ": ");
    mju::strcat_arr(message, temp);
  }

  // append element, line numbers
  if (elem) {
    mju::sprintf_arr(temp, "\nElement '%s', line %d\n", elem->Value(), elem->GetLineNum());

    mju::strcat_arr(message, temp);
  }
}


//---------------------------------- class mjXSchema implementation --------------------------------

XMLElement* FirstChildElement(XMLElement* e, const char* name) {
  XMLElement* child = e->FirstChildElement();
  for (; child; child = child->NextSiblingElement()) {
    if (!std::strcmp(child->Name(), "include")) {
      XMLElement* temp = FirstChildElement(child, name);
      if (temp) { return temp; }
      continue;
    }

    if (!name || !std::strcmp(child->Name(), name)) { return child; }
  }
  return nullptr;
}

XMLElement* NextSiblingElement(XMLElement* e, const char* name) {
  XMLElement* elem = e->NextSiblingElement();
  for (; elem; elem = elem->NextSiblingElement()) {
    if (!std::strcmp(elem->Name(), "include")) {
      XMLElement* temp = FirstChildElement(elem, name);
      if (temp) { return temp; }
      continue;
    }

    if (!name || !std::strcmp(elem->Name(), name)) { return elem; }
  }

  XMLElement* parent = e->Parent()->ToElement();
  if (parent && !std::strcmp(parent->Name(), "include")) {
    return NextSiblingElement(parent, name);
  }

  return nullptr;
}

// constructor
mjXSchema::mjXSchema(vector<const char*>     schema[],
                     unsigned                nrow,
                     const mjXConstraintDef* constraints,
                     int                     nconstraint,
                     int                     first_row) {
  // set name and type
  name_ = schema[0][0];
  type_ = schema[0][1][0];

  // adopt the presence constraints declared for this row
  for (int i = 0; i < nconstraint; i++) {
    if (constraints[i].row == first_row) { constraints_.push_back(&constraints[i]); }
  }

  // set attributes
  int nattr = schema[0].size() - 2;
  for (int i = 0; i < nattr; i++) { attr_.emplace(schema[0][2 + i]); }

  // process sub-elements of complex element
  if (nrow > 1) {
    // parse block into simple and complex elements, create children
    int start = 2;
    while (start < nrow - 1) {
      int end = start;

      // look for bracketed block at start+1
      if (schema[start + 1][0][0] == '<') {
        // look for corresponding closing bracket
        int cnt = 0;
        while (end <= nrow - 1) {
          if (schema[end][0][0] == '<') {
            cnt++;
          } else if (schema[end][0][0] == '>') {
            cnt--;
            if (cnt == 0) { break; }
          }

          end++;
        }
      }

      // add child element
      subschema_.emplace_back(schema + start,
                              end - start + 1,
                              constraints,
                              nconstraint,
                              first_row + start);

      // proceed with next subelement
      start = end + 1;
    }
  }
}


// quoted list of constraint bundles: 'a' or ('a', 'b'), comma-joined
static string BundleList(const vector<vector<string>>& bundles) {
  string out;
  for (size_t i = 0; i < bundles.size(); i++) {
    if (i) { out += ", "; }
    if (bundles[i].size() == 1) {
      out += "'" + bundles[i][0] + "'";
    } else {
      out += '(';
      for (size_t j = 0; j < bundles[i].size(); j++) {
        out += (j ? ", '" : "'") + bundles[i][j] + "'";
      }
      out += ')';
    }
  }
  return out;
}


// enforce the presence constraints declared for this element
XMLElement* mjXSchema::CheckConstraints(XMLElement* elem) {
  for (const mjXConstraintDef* con : constraints_) {
    // split the spec into bundles of attribute names
    vector<vector<string>> bundles(1);

    string token;
    for (const char* c = con->spec;; c++) {
      if (*c == ' ' || *c == '|' || *c == '\0') {
        if (!token.empty()) {
          bundles.back().push_back(token);
          token.clear();
        }
        if (*c == '|') { bundles.emplace_back(); }
        if (*c == '\0') { break; }
      } else {
        token += *c;
      }
    }

    // per-bundle presence: any member present / all members present
    int n_any = 0, n_all = 0, n_attr = 0, n_present = 0;
    for (const auto& bundle : bundles) {
      bool any = false, all = true;
      for (const string& attr : bundle) {
        bool present  = elem->Attribute(attr.c_str()) != nullptr;
        any          |= present;
        all          &= present;
        n_attr++;
        n_present += present;
      }
      n_any += any;
      n_all += all;
    }

    switch (con->kind) {
      case 'e':  // at most one bundle may be present
        if (n_any > 1) {
          error = "at most one of " + BundleList(bundles) + " can be specified";
          return elem;
        }
        break;
      case 't':  // all listed attributes appear together or not at all
        if (n_present != 0 && n_present != n_attr) {
          error = "attributes " + BundleList(bundles) + " must be specified together";
          return elem;
        }
        break;
      case 'r':  // first attribute requires the second
        if (n_any &&
            elem->Attribute(bundles[0][0].c_str()) &&
            !elem->Attribute(bundles[1][0].c_str())) {
          error = "attribute '" + bundles[0][0] + "' requires attribute '" + bundles[1][0] + "'";
          return elem;
        }
        break;
      case 'o':  // at least one bundle must be complete
        if (n_all == 0) {
          error = "one of " + BundleList(bundles) + " must be specified";
          return elem;
        }
        break;
    }
  }
  return nullptr;
}


// get pointer to error message
string mjXSchema::GetError() {
  return error;
}


// print spaces
static void printspace(stringstream& str, int n, const char* space) {
  for (int i = 0; i < n; i++) { str << space; }
}


// print schema as text
void mjXSchema::Print(stringstream& str, int level) const {
  // replace body with (world)body
  string name1 = (name_ == "body") ? "(world)body" : name_;

  // space, name, type
  printspace(str, 3 * level, " ");
  str << name1 << " (" << type_ << ")";
  int baselen = 3 * level + (int)name1.size() + 4;
  if (baselen < 30) { printspace(str, 30 - baselen, " "); }

  // attributes
  int cnt = std::max(baselen, 30);
  for (const string& attr : attr_) {
    if (cnt > 60) {
      str << "\n";
      printspace(str, (cnt = std::max(30, baselen)), " ");
    }

    str << attr << " ";
    cnt += (int)attr.size() + 1;
  }
  str << "\n";

  // children
  for (const mjXSchema& subschema : subschema_) { subschema.Print(str, level + 1); }
}


// print schema as HTML table
void mjXSchema::PrintHTML(stringstream& str, int level, bool pad) const {
  // replace body with (world)body
  string name1 = (name_ == "body" ? "(world)body" : name_);

  // open table
  if (level == 0) { str << "<table border=\"1\">\n"; }

  // name: with HTML padding
  if (pad) {
    str << "<tr>\n\t<td style=\"padding-left:" << 5 + 15 * level;
    str << "\" bgcolor=\"#EEEEEE\" class=\"el\">" << name1 << "</td>\n";
  }

  // name: with &nbsp; for browsers that ignore padding
  else {
    str << "<tr>\n\t<td bgcolor=\"#EEEEEE\" class=\"el\">";
    if (level) { printspace(str, 4 * level, "&nbsp;"); }
    str << name1 << "</td>\n";
  }

  // type
  str << "\t<td class=\"ty\">" << type_ << "</td>\n";

  // attributes
  str << "\t<td class=\"at\">";
  if (!attr_.empty()) {
    for (const string& attr : attr_) { str << attr << " "; }
  } else {
    str << "<span style=\"color:black\"><i>no attributes</i></span>";
  }
  str << "</td>\n</tr>\n";

  // children
  for (const mjXSchema& subschema : subschema_) { subschema.PrintHTML(str, level + 1, pad); }

  // close table
  if (!level) { str << "</table>\n"; }
}


// check for name match
bool mjXSchema::NameMatch(XMLElement* elem, int level) {
  // special handling of body, worldbody, and frame
  if (name_ == "body" && ((level == 1 && !strcmp(elem->Value(), "worldbody")) ||
                          (level != 1 && !strcmp(elem->Value(), "body")) ||
                          (level >= 1 && !strcmp(elem->Value(), "frame")) ||
                          (level >= 1 && !strcmp(elem->Value(), "replicate")))) {
    return true;
  }

  // regular check
  return name_ == elem->Value();
}


// validator
XMLElement* mjXSchema::Check(XMLElement* elem, int level) {
  bool        missing;
  char        msg[100];
  XMLElement *bad, *sub;

  error.clear();
  if (!elem) {
    return 0;  // SHOULD NOT OCCUR
  }

  // check name (already done by parent, but hard to avoid)
  if (!NameMatch(elem, level)) {
    error = "unrecognized element";
    return elem;
  }

  // check attributes
  const XMLAttribute* attribute = elem->FirstAttribute();
  for (; attribute != nullptr; attribute = attribute->Next()) {
    // tolerate XML namespace machinery on the root, so models can carry
    // xsi:noNamespaceSchemaLocation for editor completion against mjcf.xsd
    if (level == 0 && (strchr(attribute->Name(), ':') || !strcmp(attribute->Name(), "xmlns"))) {
      continue;
    }
    if (attr_.find(attribute->Name()) == attr_.end()) {
      error = "unrecognized attribute: '" + string(attribute->Name()) + "'";
      return elem;
    }
  }

  // check presence constraints
  if ((bad = CheckConstraints(elem))) { return bad; }

  // handle recursion
  if (type_ == 'R') {
    // check child elements with same name
    sub = FirstChildElement(elem, name_.c_str());
    for (; sub != nullptr; sub = NextSiblingElement(sub, name_.c_str())) {
      if ((bad = Check(sub, level + 1))) { return bad; }
    }
  }

  // clear reference counts
  for (mjXSchema& subschema : subschema_) { subschema.refcnt_ = 0; }

  // check sub-elements, update refcnt
  sub = FirstChildElement(elem);
  for (; sub != nullptr; sub = NextSiblingElement(sub)) {
    missing = true;

    for (mjXSchema& subschema : subschema_) {
      if (subschema.NameMatch(sub, level + 1)) {
        // check sub-tree
        if ((bad = subschema.Check(sub, level + 1))) {
          error = subschema.error;
          return bad;
        }

        // mark found
        missing = false;
        subschema.refcnt_++;
        break;
      }
    }

    // missing, unless recursive
    if (missing && !(type_ == 'R' && NameMatch(sub, level + 1))) {
      error = "unrecognized element";
      return sub;
    }
  }

  // enforce sub-element types
  msg[0] = '\0';
  for (mjXSchema& subschema : subschema_) {
    switch (subschema.type_) {
      case '!':
        if (subschema.refcnt_ > 1)
          mju::sprintf_arr(msg,
                           "unique element '%s' found %d times",
                           subschema.name_.c_str(),
                           subschema.refcnt_);
        else if (subschema.refcnt_ < 1)
          mju::sprintf_arr(msg, "element '%s' is required", subschema.name_.c_str());
        break;

      case '?':
        if (subschema.refcnt_ > 1)
          mju::sprintf_arr(msg,
                           "unique element '%s' found %d times",
                           subschema.name_.c_str(),
                           subschema.refcnt_);
        break;

      default:
        break;
    }
  }

  // handle error
  if (msg[0]) {
    error = msg;
    return elem;
  }
  return nullptr;
}


//---------------------------------- class mjXUtil implementation ----------------------------------

// helper function to read multiple numerical values from an attribute
// return false if the entire attribute wasn't read (max was reached)
// throw error if syntax error while trying to read numerical data
template <typename T>
bool mjXUtil::ReadAttrValues(XMLElement*            elem,
                             const char*            attr,
                             function<void(int, T)> push,
                             int                    max) {
  const char* pstr = elem->Attribute(attr);
  T           item;

  if (pstr == nullptr) { return true; }

  // get input stream
  string             str = string(pstr);
  std::istringstream strm(str);
  string             token;

  // read numbers
  for (int i = 0; (max < 0 || i < max) && !strm.eof(); ++i) {
    strm >> token;
    std::istringstream token_strm(token);
    token_strm >> item;
    if (token_strm.fail() || !token_strm.eof()) {
      // C++ standard libraries do not always parse inf and nan as valid floating point values.
      optional<T> maybe_result = ParseInfOrNan<T>(token);
      if (maybe_result.has_value()) {
        item = maybe_result.value();
      } else {
        throw mjXError(elem, "problem reading attribute '%s'", attr);
      }
    }

    push(i, item);
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(item)) { mju_warning("XML contains a 'NaN'. Please check it carefully."); }
    }
    // clear any trailing whitespace
    strm >> std::ws;
  }

  return strm.eof();
}

template bool mjXUtil::ReadAttrValues(XMLElement*                 elem,
                                      const char*                 attr,
                                      function<void(int, double)> push,
                                      int                         max);
template bool mjXUtil::ReadAttrValues(XMLElement*                elem,
                                      const char*                attr,
                                      function<void(int, float)> push,
                                      int                        max);
template bool mjXUtil::ReadAttrValues(XMLElement*              elem,
                                      const char*              attr,
                                      function<void(int, int)> push,
                                      int                      max);
template bool mjXUtil::ReadAttrValues(XMLElement*                        elem,
                                      const char*                        attr,
                                      function<void(int, unsigned char)> push,
                                      int                                max);


// compare two vectors
template <typename T>
bool mjXUtil::SameVector(const T* vec1, const T* vec2, int n) {
  if (!vec1 || !vec2) { return false; }

  for (int i = 0; i < n; i++) {
    if (std::abs(vec1[i] - vec2[i]) > std::numeric_limits<T>::epsilon()) { return false; }
  }

  return true;
}

template bool mjXUtil::SameVector(const double* vec1, const double* vec2, int n);
template bool mjXUtil::SameVector(const float* vec1, const float* vec2, int n);
template bool mjXUtil::SameVector(const int* vec1, const int* vec2, int n);
template bool mjXUtil::SameVector(const unsigned char* vec1, const unsigned char* vec2, int n);


// find string in map, return corresponding integer (-1: not found)
int mjXUtil::FindKey(const mjMap* map, int mapsz, string key) {
  for (int i = 0; i < mapsz; i++) {
    if (map[i].key == key) { return map[i].value; }
  }

  return -1;
}


// find integer in map, return corresponding string ("": not found)
string mjXUtil::FindValue(const mjMap* map, int mapsz, int value) {
  for (int i = 0; i < mapsz; i++) {
    if (map[i].value == value) { return map[i].key; }
  }

  return "";
}


// if attribute is present, return vector of numerical data
template <typename T>
optional<vector<T>> mjXUtil::ReadAttrVec(XMLElement* elem, const char* attr, bool required) {
  vector<T>   v;
  const char* raw_cstr = elem->Attribute(attr);
  if (raw_cstr) {
    v = mujoco::user::StringToVector<T>(raw_cstr);
    if (errno == EDOM) {
      mju_warning("XML contains a 'NaN'. Please check it carefully.");
    } else if (errno == ERANGE) {
      throw mjXError(elem, "number is too large in attribute '%s'", attr);
    } else if (errno == EINVAL) {
      throw mjXError(elem, "bad format in attribute '%s'", attr);
    } else if (errno != 0) {
      throw mjXError(elem, "unknown error in attribute '%s'", attr);
    }
  }

  if (!v.size()) {
    if (required) {
      throw mjXError(elem, "required attribute missing: '%s'", attr);
    } else {
      return std::nullopt;
    }
  }

  return v;
}

template optional<vector<double>>        mjXUtil::ReadAttrVec(XMLElement* elem,
                                                              const char* attr,
                                                              bool        required);
template optional<vector<float>>         mjXUtil::ReadAttrVec(XMLElement* elem,
                                                              const char* attr,
                                                              bool        required);
template optional<vector<int>>           mjXUtil::ReadAttrVec(XMLElement* elem,
                                                              const char* attr,
                                                              bool        required);
template optional<vector<unsigned char>> mjXUtil::ReadAttrVec(XMLElement* elem,
                                                              const char* attr,
                                                              bool        required);


// if attribute is present, return attribute as a string
optional<string> mjXUtil::ReadAttrStr(XMLElement* elem, const char* attr, bool required) {
  const char* pstr = elem->Attribute(attr);

  // check if attribute exists
  if (pstr == nullptr) {
    if (required) {
      throw mjXError(elem, "required attribute missing: '%s'", attr);
    } else {
      return std::nullopt;
    }
  }

  return string(pstr);
}

// if attribute is present, return attribute as a filename
optional<FilePath> mjXUtil::ReadAttrFile(
    XMLElement* elem, const char* attr, const mjVFS* vfs, const FilePath& dir, bool required) {
  auto maybe_str = ReadAttrStr(elem, attr, required);
  if (!maybe_str.has_value()) { return std::nullopt; }
  FilePath filename(maybe_str.value());
  return ResolveFilePath(elem, filename, dir, vfs);
}

// if attribute is present, return numerical value of attribute
template <typename T>
optional<T> mjXUtil::ReadAttrNum(XMLElement* elem, const char* attr, bool required) {
  auto maybe_arr = ReadAttrArr<T, 1>(elem, attr, required);
  if (!maybe_arr.has_value()) { return std::nullopt; }

  return maybe_arr.value()[0];
}

template optional<double> mjXUtil::ReadAttrNum(XMLElement* elem, const char* attr, bool required);
template optional<float>  mjXUtil::ReadAttrNum(XMLElement* elem, const char* attr, bool required);
template optional<int>    mjXUtil::ReadAttrNum(XMLElement* elem, const char* attr, bool required);
template optional<unsigned char> mjXUtil::ReadAttrNum(XMLElement* elem,
                                                      const char* attr,
                                                      bool        required);


// read attribute "attr" of element "elem"
//  "len" is the number of floats or doubles to be read
//  the content is returned in "text", the numeric data in "data"
//  return number of elements found
template <typename T>
int mjXUtil::ReadAttr(XMLElement* elem,
                      const char* attr,
                      const int   len,
                      T*          data,
                      string&     text,
                      bool        required,
                      bool        exact) {
  auto maybe_vec = ReadAttrVec<T>(elem, attr, required);
  if (!maybe_vec.has_value()) { return 0; }

  // check if there is not enough data
  if (exact && maybe_vec->size() < len) {
    throw mjXError(elem, "attribute '%s' does not have enough data", attr);
  }

  // check if there is too much data
  if (maybe_vec->size() > len) { throw mjXError(elem, "attribute '%s' has too much data", attr); }

  std::copy(maybe_vec->begin(), maybe_vec->end(), data);
  return maybe_vec->size();
}

template int mjXUtil::ReadAttr(XMLElement* elem,
                               const char* attr,
                               int         len,
                               double*     data,
                               string&     text,
                               bool        required,
                               bool        exact);

template int mjXUtil::ReadAttr(XMLElement* elem,
                               const char* attr,
                               int         len,
                               float*      data,
                               string&     text,
                               bool        required,
                               bool        exact);

template int mjXUtil::ReadAttr(XMLElement* elem,
                               const char* attr,
                               int         len,
                               int*        data,
                               string&     text,
                               bool        required,
                               bool        exact);

template int mjXUtil::ReadAttr(XMLElement*    elem,
                               const char*    attr,
                               int            len,
                               unsigned char* data,
                               string&        text,
                               bool           required,
                               bool           exact);

// read quaternion attribute
//  throw error if identically zero
int mjXUtil::ReadQuat(
    XMLElement* elem, const char* attr, double* data, string& text, bool required) {
  int n = ReadAttr(elem, attr, /*len=*/4, data, text, required, /*exact=*/true);
  if (n == 0) return 0;

  // check for 0 quaternion
  if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 0) {
    throw mjXError(elem, "zero quaternion is not allowed");
  }

  return n;
}

// read DOUBLE array into C++ vector, return number read
int mjXUtil::ReadVector(
    XMLElement* elem, const char* attr, vector<double>& vec, string& text, bool required) {
  auto maybe_vec = ReadAttrVec<double>(elem, attr, required);
  if (!maybe_vec.has_value()) { return 0; }

  vec = std::move(maybe_vec.value());
  return vec.size();
}


// read text field
bool mjXUtil::ReadAttrTxt(XMLElement* elem, const char* attr, string& text, bool required) {
  auto maybe_str = ReadAttrStr(elem, attr, required);
  if (!maybe_str.has_value()) { return false; }

  text = maybe_str.value();
  return true;
}

// read single int
bool mjXUtil::ReadAttrInt(XMLElement* elem, const char* attr, int* data, bool required) {
  auto maybe_int = ReadAttrNum<int>(elem, attr, required);
  if (!maybe_int.has_value()) { return false; }

  *data = maybe_int.value();
  return true;
}


// find subelement with given name, make sure it is unique
XMLElement* mjXUtil::FindSubElem(XMLElement* elem, string name, bool required) {
  XMLElement* subelem = 0;

  XMLElement* iter = elem->FirstChildElement();
  while (iter) {
    // identify elements with given name
    if (name == iter->Value()) {
      // make sure name is not repeated
      if (subelem) { throw mjXError(subelem, "repeated element: '%s'", name.c_str()); }

      // save found element
      subelem = iter;
    }

    // advance to next element
    iter = iter->NextSiblingElement();
  }

  if (required && !subelem) { throw mjXError(elem, "missing element: '%s'", name.c_str()); }

  return subelem;
}


// find attribute, translate key into data, return true if found
bool mjXUtil::MapValue(
    XMLElement* elem, const char* attr, int* data, const mjMap* map, int mapSz, bool required) {
  // get attribute text
  auto maybe_text = ReadAttrStr(elem, attr, required);
  if (!maybe_text.has_value()) { return false; }

  // find keyword in map
  int value = FindKey(map, mapSz, maybe_text.value());
  if (value < 0) { throw mjXError(elem, "invalid keyword: '%s'", maybe_text->c_str()); }

  // copy
  *data = value;
  return true;
}


// find attribute, translate unique space-separated keys to data, return number of keys found
int mjXUtil::MapValues(
    XMLElement* elem, const char* attr, int* data, const mjMap* map, int mapSz, bool required) {
  // get attribute text
  auto maybe_text = ReadAttrStr(elem, attr, required);
  if (!maybe_text.has_value()) { return 0; }

  string             text = maybe_text.value();
  std::istringstream strm(text);
  string             key;
  std::set<string>   found_keys;
  int                count = 0;

  while (strm >> key) {
    if (found_keys.count(key)) {
      throw mjXError(elem, "duplicate keyword: '%s'");
      return 0;
    }

    int value = FindKey(map, mapSz, key);
    if (value == -1) {
      throw mjXError(elem, "invalid keyword: '%s'");
      return 0;
    }

    found_keys.insert(key);
    data[count++] = value;
  }

  return count;
}


//---------------------------------- write functions -----------------------------------------------

// check if double is int
static bool isint(double x) {
  return ((std::abs(x - floor(x)) < 1E-12) || (std::abs(x - ceil(x)) < 1E-12));
}


// round to nearest int
static int Round(double x) {
  if (std::abs(x - floor(x)) < std::abs(x - ceil(x))) {
    return (int)floor(x);
  } else {
    return (int)ceil(x);
  }
}


// write number: integer without decimal point, otherwise in the precision of the stream;
// precision 0 writes numbers exactly
template <typename T>
static void WriteNumber(stringstream& stream, T number) {
  const int precision = mujoco::_mjPRIVATE__get_xml_precision();
  if constexpr (std::is_floating_point_v<T>) {
    if (precision == 0) {
      stream << mujoco::ShortestNumber(number);
      return;
    }
  }
  double doubledata = static_cast<double>(number);
  if (precision <= 12 &&
      doubledata < INT_MAX &&
      doubledata > -INT_MAX &&
      isint(number) &&
      (Round(doubledata) != 0 || doubledata == 0)) {
    stream << Round(number);
  } else {
    stream << number;
  }
}


// write attribute
template <typename T>
void mjXUtil::WriteAttr(
    XMLElement* elem, string name, int n, const T* data, const T* def, bool trim) {
  // make sure all are defined
  if constexpr (std::is_floating_point_v<T>) {
    for (int i = 0; i < n; i++) {
      if (std::isnan(data[i])) { return; }
    }
  }

  // increase precision for testing; precision 0 writes numbers exactly
  stringstream stream;
  const int    precision = mujoco::_mjPRIVATE__get_xml_precision();
  if (precision > 0) { stream.precision(precision); }

  // skip default attributes: values which are the defaults to roundoff, or which are the defaults
  // if numbers are written exactly
  if (precision == 0 ? def && std::equal(data, data + n, def) : SameVector(data, def, n)) {
    return;
  }

  // trim identical trailing default values
  if (trim && def) {
    while (n > 0 && data[n - 1] == def[n - 1]) { n--; }
  }

  // process all numbers
  T max_abs = 0;
  if constexpr (std::is_floating_point_v<T>) {
    if (precision > 0 && precision <= 12) {
      for (int i = 0; i < n; i++) { max_abs = std::max(max_abs, std::abs(data[i])); }
    }
  }

  for (int i = 0; i < n; i++) {
    // add space between numbers
    if (i > 0) { stream << " "; }

    // append number
    T val = data[i];
    if constexpr (std::is_floating_point_v<T>) {
      if (precision > 0 && precision <= 12 && max_abs > 0 && std::abs(val) <= 1e-12 * max_abs) {
        val = 0;
      }
    }
    WriteNumber(stream, val);
  }

  // set attribute as string
  WriteAttrTxt(elem, name, stream.str());
}


template void mjXUtil::WriteAttr(
    XMLElement* elem, string name, int n, const double* data, const double* def, bool trim);

template void mjXUtil::WriteAttr(
    XMLElement* elem, string name, int n, const float* data, const float* def, bool trim);

template void mjXUtil::WriteAttr(
    XMLElement* elem, string name, int n, const int* data, const int* def, bool trim);

template void mjXUtil::WriteAttr(XMLElement*          elem,
                                 string               name,
                                 int                  n,
                                 const unsigned char* data,
                                 const unsigned char* def,
                                 bool                 trim);


// write vector of numbers to string, as WriteAttr writes them
template <typename T>
void mjXUtil::Vector2String(string& txt, const vector<T>& vec, int ncol) {
  stringstream strm;
  const int    precision = mujoco::_mjPRIVATE__get_xml_precision();
  if (precision > 0) { strm.precision(precision); }

  T max_abs = 0;
  if constexpr (std::is_floating_point_v<T>) {
    if (precision > 0 && precision <= 12) {
      for (size_t i = 0; i < vec.size(); i++) { max_abs = std::max(max_abs, std::abs(vec[i])); }
    }
  }

  for (size_t i = 0; i < vec.size(); i++) {
    if (ncol && (i % ncol) == 0) {
      strm << "\n            ";
    } else if (i > 0) {
      strm << " ";
    }
    T val = vec[i];
    if constexpr (std::is_floating_point_v<T>) {
      if (precision > 0 && precision <= 12 && max_abs > 0 && std::abs(val) <= 1e-12 * max_abs) {
        val = 0;
      }
    }
    WriteNumber(strm, val);
  }

  txt = strm.str();
}


template void mjXUtil::Vector2String(string& txt, const vector<float>& vec, int ncol);

template void mjXUtil::Vector2String(string& txt, const vector<double>& vec, int ncol);


// write vector<double> attribute, default = zero array
void mjXUtil::WriteVector(XMLElement* elem, string name, const vector<double>& vec) {
  // proceed only if non-zero found
  bool ok = false;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i]) {
      ok = true;
      break;
    }
  }
  if (!ok) { return; }

  // write
  WriteAttr(elem, name, vec.size(), vec.data());
}


// write vector<double> attribute, default with same size
void mjXUtil::WriteVector(XMLElement*           elem,
                          string                name,
                          const vector<double>& vec,
                          const vector<double>& def) {
  // proceed only if non-zero found
  bool ok = false;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != def[i]) {
      ok = true;
      break;
    }
  }
  if (!ok) { return; }

  // write
  WriteAttr(elem, name, vec.size(), vec.data());
}


// write attribute- string
void mjXUtil::WriteAttrTxt(XMLElement* elem, string name, string value) {
  // skip if value is empty
  if (value.empty()) { return; }

  // set attribute
  elem->SetAttribute(name.c_str(), value.c_str());
}


// write attribute- single int
void mjXUtil::WriteAttrInt(XMLElement* elem, string name, int data, int def) {
  // skip default
  if (data == def) { return; }

  elem->SetAttribute(name.c_str(), data);
}


// write attribute- keyword
void mjXUtil::WriteAttrKey(
    XMLElement* elem, string name, const mjMap* map, int mapsz, int data, int def) {
  // skip default
  if (data == def) { return; }

  WriteAttrTxt(elem, name, FindValue(map, mapsz, data));
}


// write attribute- space-separated keywords
void mjXUtil::WriteAttrKeys(
    XMLElement* elem, string name, const mjMap* map, int mapsz, int* data, int ndata, int def) {
  if (ndata <= 0) { return; }

  // skip default
  if (ndata == 1 && data[0] == def) { return; }

  string text = FindValue(map, mapsz, data[0]);
  for (int i = 1; i < ndata; ++i) {
    text += " ";
    text += FindValue(map, mapsz, data[i]);
  }

  WriteAttrTxt(elem, name, text);
}
