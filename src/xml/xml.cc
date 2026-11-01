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

#include "xml/xml.h"

#include <locale.h>
#include <cstring>

#if defined(__APPLE__) || defined(__FreeBSD__)
  #include <xlocale.h>
#endif

#include <array>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_set>

#include <mujoco/mujoco.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include "cc/array_safety.h"
#include "engine/engine_crossplatform.h"
#include <mujoco/mjspec.h>
#include "user/user_resource.h"
#include "user/user_util.h"
#include "user/user_vfs.h"
#include "xml/xml_native_reader.h"
#include "xml/xml_native_writer.h"
#include "xml/xml_urdf.h"
#include "xml/xml_util.h"
#include "tinyxml2.h"

namespace {

using mujoco::user::FilePath;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLNode;

namespace mju = ::mujoco::util;


// We are using "locale-sensitive" sprintf to read and write XML.
// When MuJoCo is being used as a plug-in for an application that respects the system locale
// (e.g. Unity), the user's locale setting can affect the formatting of numbers into strings.
// Specifically, a number of European locales (e.g. de_DE) uses commas to as decimal separators.
// In order to ensure that XMLs are locale-inpendent, we temporarily switch to the "C" locale
// when handling. Since the standard C `setlocale` is not thread-safe, we instead use
// platform-specific extensions to override the locale only in the calling thread.
// See also https://github.com/google-deepmind/mujoco/issues/131.
#ifdef _WIN32
class LocaleOverride {
 public:
  LocaleOverride()
      : old_per_thread_locale_type_(_configthreadlocale(0)),
        old_locale_(setlocale(LC_ALL, nullptr)) {
    _configthreadlocale(_ENABLE_PER_THREAD_LOCALE);
    setlocale(LC_ALL, "C");
  }

  ~LocaleOverride() {
    setlocale(LC_ALL, old_locale_.c_str());
    _configthreadlocale(old_per_thread_locale_type_);
  }

 private:
  int         old_per_thread_locale_type_;
  std::string old_locale_;
};
#else
class LocaleOverride {
 public:
  static locale_t PosixLocale() {
    static locale_t posix_locale = newlocale(LC_ALL_MASK, "C", 0);
    return posix_locale;
  }

  LocaleOverride() : old_locale_(uselocale(PosixLocale())) {}

  ~LocaleOverride() { uselocale(old_locale_); }

 private:
  locale_t old_locale_;
};
#endif

// find include elements recursively, replace them with subtree from xml file
void IncludeXML(mjXReader&                       reader,
                XMLElement*                      elem,
                const FilePath&                  dir,
                const mjVFS*                     vfs,
                std::unordered_set<std::string>& included);

// load an include element and replace it with the subtree from the xml file
void ExpandIncludeXML(mjXReader&                       reader,
                      XMLElement*                      elem,
                      const FilePath&                  dir,
                      const mjVFS*                     vfs,
                      std::unordered_set<std::string>& included) {
  // make sure include has no children
  if (!elem->NoChildren()) { throw mjXError(elem, "Include element cannot have children"); }

  // get filename
  auto file_attr = mjXUtil::ReadAttrStr(elem, "file", true);
  if (!file_attr.has_value()) { throw mjXError(elem, "Include element missing file attribute"); }
  FilePath filename = FilePath(file_attr.value());

  // block repeated include files
  if (included.find(filename.Str()) != included.end()) {
    throw mjXError(elem, "File '%s' already included", filename.c_str());
  }

  // TODO: b/325905702 - We have a messy wrapper here to remain backwards
  // compatible, which will be removed in the near future.
  // legacy behavior: try to load in top level directory
  std::array<char, 1024> error;
  mjResource*            resource = mju_openResource(reader.ModelFileDir().c_str(),
                                                     filename.c_str(),
                                                     vfs,
                                                     error.data(),
                                                     error.size());
  if (resource == nullptr) {
    // new behavior: try to load in relative directory
    if (!filename.IsAbs() && !dir.empty()) {
      FilePath fullname = dir + filename;
      if (included.find(fullname.Str()) != included.end()) {
        throw mjXError(elem, "File '%s' already included", fullname.c_str());
      }
      resource = mju_openResource("", fullname.c_str(), vfs, error.data(), error.size());
      if (resource != nullptr) { filename = fullname; }
    }
  }

  if (resource == nullptr) { throw mjXError(elem, "%s", error.data()); }

  FilePath canonical(resource->name ? resource->name : "");
  if (!canonical.Str().empty() && included.find(canonical.Str()) != included.end()) {
    mju_closeResource(resource);
    throw mjXError(elem, "File '%s' already included", filename.c_str());
  }

  const char* include_dir  = nullptr;
  int         ninclude_dir = 0;
  mju_getResourceDir(resource, &include_dir, &ninclude_dir);
  FilePath next_dir = FilePath(std::string(include_dir, ninclude_dir));
  elem->SetAttribute("dir", next_dir.c_str());

  const char* xmlstring   = nullptr;
  int         buffer_size = mju_readResource(resource, (const void**)&xmlstring);
  if (buffer_size < 0) {
    mju_closeResource(resource);
    throw mjXError(elem, "Error reading file '%s'", filename.c_str());
  } else if (!buffer_size) {
    mju_closeResource(resource);
    throw mjXError(elem, "Empty file '%s'", filename.c_str());
  }

  // load XML file or parse string
  XMLDocument doc;
  doc.Parse(xmlstring, buffer_size);

  // close resource
  mju_closeResource(resource);

  // check error
  if (doc.Error()) {
    char err[1000];
    mju::sprintf_arr(err, "XML parse error %d:\n%s\n", doc.ErrorID(), doc.ErrorStr());
    throw mjXError(elem, "Include error: '%s'", err);
  }

  // remember that file was included
  included.insert(filename.Str());
  if (!canonical.Str().empty()) { included.insert(canonical.Str()); }

  // get and check root element
  XMLElement* docroot = doc.RootElement();
  if (!docroot) { throw mjXError(elem, "Root element missing in file '%s'", filename.c_str()); }

  // get and check first child
  XMLElement* eleminc = docroot->FirstChildElement();
  if (!eleminc) { throw mjXError(elem, "Empty include file '%s'", filename.c_str()); }

  // get <include> element
  XMLElement*  include     = elem->ToElement();
  XMLDocument* include_doc = include->GetDocument();

  // clone first child of included document
  XMLNode* first = include->InsertFirstChild(eleminc->DeepClone(include_doc));

  // point to first
  XMLElement* child = first->ToElement();

  // insert remaining elements from included document as siblings
  eleminc = eleminc->NextSiblingElement();
  while (eleminc) {
    child   = include->InsertAfterChild(child, eleminc->DeepClone(include_doc))->ToElement();
    eleminc = eleminc->NextSiblingElement();
  }

  // recursively run include
  child = include->FirstChildElement();
  for (; child; child = child->NextSiblingElement()) {
    IncludeXML(reader, child, next_dir, vfs, included);
  }
}

// find include elements recursively, replace them with subtree from xml file
void IncludeXML(mjXReader&                       reader,
                XMLElement*                      elem,
                const FilePath&                  dir,
                const mjVFS*                     vfs,
                std::unordered_set<std::string>& included) {
  // capture directory defaults on first pass of XML tree
  if (!strcasecmp(elem->Value(), "compiler")) {
    auto assetdir_attr = mjXUtil::ReadAttrStr(elem, "assetdir");
    if (assetdir_attr.has_value()) { reader.SetAssetDir(assetdir_attr.value()); }

    auto texturedir_attr = mjXUtil::ReadAttrStr(elem, "texturedir");
    if (texturedir_attr.has_value()) { reader.SetTextureDir(texturedir_attr.value()); }

    auto meshdir_attr = mjXUtil::ReadAttrStr(elem, "meshdir");
    if (meshdir_attr.has_value()) { reader.SetMeshDir(meshdir_attr.value()); }
  }

  // not an include, recursively go through all children
  if (strcasecmp(elem->Value(), "include")) {
    XMLElement* child = elem->FirstChildElement();
    for (; child; child = child->NextSiblingElement()) {
      IncludeXML(reader, child, dir, vfs, included);
    }
    return;
  }

  ExpandIncludeXML(reader, elem, dir, vfs, included);
}

// Main parser function
mjSpec* SpecFromXML(std::string_view xml,
                    std::string_view dir,
                    std::string_view filename,
                    const mjVFS*     vfs,
                    char*            error,
                    int              nerror) {
  LocaleOverride locale_override;

  // clear
  mjSpec* spec = nullptr;
  if (error) { error[0] = '\0'; }

  // load XML file or parse string
  XMLDocument doc;
  doc.Parse(xml.data(), xml.size());

  // error checking
  if (doc.Error()) {
    if (error) {
      snprintf(error, nerror, "XML parse error %d:\n%s\n", doc.ErrorID(), doc.ErrorStr());
    }
    return nullptr;
  }

  // get top-level element
  XMLElement* root = doc.RootElement();
  if (!root) {
    mjCopyError(error, "XML root element not found", nerror);
    return nullptr;
  }

  // create model, set filedir
  spec = mj_makeSpec();
  mjs_setString(spec->modelfiledir, std::string(dir).c_str());


  // parse with exceptions
  try {
    if (!strcasecmp(root->Value(), "mujoco")) {
      // find include elements, replace them with subtree from xml file
      std::unordered_set<std::string> included = {std::string(filename)};
      mjXReader                       parser;
      parser.SetModelFileDir(mjs_getString(spec->modelfiledir));
      IncludeXML(parser, root, FilePath(), vfs, included);

      // parse MuJoCo model
      parser.SetModel(spec);
      parser.Parse(root, vfs);
    }

    else if (!strcasecmp(root->Value(), "robot")) {
      // parse URDF model
      mjXURDF parser;

      // set reasonable default for parsing a URDF
      // this is separate from the Parser to allow multiple URDFs to be loaded.
      spec->compiler.fusestatic    = true;
      spec->compiler.discardvisual = true;

      parser.SetModel(spec);
      parser.Parse(root);
    }

    else {
      throw mjXError(0, "Unrecognized XML model type: '%s'", root->Value());
    }
  }

  // catch known errors
  catch (mjXError err) {
    mjCopyError(error, err.message, nerror);
    mj_deleteSpec(spec);
    return nullptr;
  }

  return spec;
}
}  // namespace

mjSpec* ParseXML(const char* filename, const mjVFS* vfs, char* error, int nerror) {
  // check arguments
  if (!filename) {
    if (error) { std::snprintf(error, nerror, "ParseXML: filename argument required\n"); }
    return nullptr;
  }

  // get data source
  const char*            xml = nullptr;
  std::array<char, 1024> rerror;
  mjResource* resource = mju_openResource("", filename, vfs, rerror.data(), rerror.size());
  if (resource == nullptr) {
    std::snprintf(error, nerror, "ParseXML: %s", rerror.data());
    return nullptr;
  }

  int buffer_size = mju_readResource(resource, (const void**)&xml);
  if (buffer_size < 0) {
    if (error) { std::snprintf(error, nerror, "ParseXML: error reading file '%s'", filename); }
    mju_closeResource(resource);
    return nullptr;
  } else if (!buffer_size) {
    if (error) { std::snprintf(error, nerror, "ParseXML: empty file '%s'", filename); }
    mju_closeResource(resource);
    return nullptr;
  }

  const char* dir;
  int         ndir = 0;
  mju_getResourceDir(resource, &dir, &ndir);
  std::string_view directory(dir, ndir);

  mjSpec* spec = SpecFromXML({xml, xml + buffer_size}, directory, filename, vfs, error, nerror);

  mju_closeResource(resource);
  return spec;
}

mjSpec* ParseSpecFromString(std::string_view xml, const mjVFS* vfs, char* error, int nerror) {
  return SpecFromXML(xml, "", "", vfs, error, nerror);
}

// Main writer function - calls mjXWrite
std::string WriteXML(const mjModel* m, mjSpec* spec, char* error, int nerror) {
  LocaleOverride locale_override;

  // check for empty model
  if (!spec) {
    mjCopyError(error, "Cannot write empty model", nerror);
    return "";
  }

  mjXWriter writer;

  try {
    writer.SetModel(spec, m);
    return writer.Write(error, nerror);
  } catch (mjXError err) {
    mjCopyError(error, err.message, nerror);
    return "";
  }
}
