// Copyright 2025 DeepMind Technologies Limited
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

#include "mjcf/mjcf_file_format.h"

#include <array>
#include <cstring>
#include <memory>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

#include <mujoco/mujoco.h>
#include "mjcf/mujoco_to_usd.h"
#include "tinyxml2.h"
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/enum.h>
#include <pxr/base/tf/pathUtils.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/work/loops.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/declareHandles.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PUBLIC_TOKENS(UsdMjcfFileFormatTokens, USD_MJCF_FILE_FORMAT_TOKENS);

TF_REGISTRY_FUNCTION(TfType) {
  SDF_DEFINE_FILE_FORMAT(UsdMjcfFileFormat, SdfFileFormat);
}

enum ErrorCodes { XmlParsingError };
TF_REGISTRY_FUNCTION(TfEnum) {
  TF_ADD_ENUM_NAME(XmlParsingError, "Error when parsing XML.");
};

namespace {

bool IsAbsolutePath(const std::string &path) {
  // empty: not absolute
  if (path.empty()) {
    return false;
  }

  // path is scheme:filename which we consider an absolute path
  // e.g. file URI's are always absolute paths
  if (mjp_findProvider(path.c_str()) != nullptr) {
    return true;
  }

  // check first char
  const char *str = path.c_str();
  if (str[0] == '\\' || str[0] == '/') {
    return true;
  }

  // find ":/" or ":\"
  if (path.find(":/") != std::string::npos ||
      path.find(":\\") != std::string::npos) {
    return true;
  }

  return false;
}

void ResolveMjcfDependencies(const std::string &xml_string,
                             const std::string &resolved_path);

void AccumulateFiles(std::unordered_set<std::string> &files,
                     tinyxml2::XMLElement *root,
                     const std::string &resolved_path) {
  const char *asset_dir = nullptr;
  const char *mesh_dir = nullptr;
  const char *texture_dir = nullptr;
  std::vector<std::string> texture_files;
  std::vector<std::string> mesh_files;
  std::vector<std::string> hfield_files;

  const std::string parent_dir = TfNormPath(TfGetPathName(resolved_path));

  auto accumulate_files = [&](const std::vector<std::string> &candidate_files,
                              const char *prefix_path) {
    for (const auto &file : candidate_files) {
      std::string file_with_prefix = prefix_path == nullptr ? file : TfStringCatPaths(prefix_path, file);
      if (IsAbsolutePath(file)) {
        auto identifier = pxr::ArGetResolver().CreateIdentifier(file);
        // If this path is an absolute path, insert that path.
        files.insert(identifier);
      } else if (IsAbsolutePath(file_with_prefix)) {
        auto identifier =
            pxr::ArGetResolver().CreateIdentifier(file_with_prefix);
        // Else if prefix is an absolute path, insert prefix / file.
        files.insert(identifier);
      } else {
        // Else insert resolved_path / prefix / file.
        auto identifier = pxr::ArGetResolver().CreateIdentifier(
            file_with_prefix, pxr::ArResolvedPath(resolved_path));
        files.insert(identifier);
      }
    }
  };

  std::stack<tinyxml2::XMLElement *> elements;
  elements.push(root);
  while (!elements.empty()) {
    tinyxml2::XMLElement *elem = elements.top();
    elements.pop();

    if (!strcasecmp(elem->Value(), "include") ||
        !strcasecmp(elem->Value(), "model")) {
      const char *file = elem->Attribute("file");
      if (file != nullptr) {
        auto identifier = pxr::ArGetResolver().CreateIdentifier(
            std::string(file), pxr::ArResolvedPath(resolved_path));
        auto include_resolved_path = pxr::ArGetResolver().Resolve(identifier);
        auto asset = pxr::ArGetResolver().OpenAsset(include_resolved_path);
        ResolveMjcfDependencies(asset->GetBuffer().get(),
                                include_resolved_path);

        // Neither of these elements should have children.
        continue;
      }
    } else if (!strcasecmp(elem->Value(), "compiler")) {
      asset_dir = elem->Attribute("assetdir");
      mesh_dir = elem->Attribute("meshdir");
      texture_dir = elem->Attribute("texturedir");

      // compiler elements don't have children.
      continue;
    } else if (!strcasecmp(elem->Value(), "mesh")) {
      // mesh elements don't have children.
      const char *file = elem->Attribute("file");
      if (file != nullptr) {
        mesh_files.emplace_back(file);
      }
      continue;
    } else if (!strcasecmp(elem->Value(), "hfield")) {
      // hfield elements don't have children.
      const char *file = elem->Attribute("file");
      if (file != nullptr) {
        hfield_files.emplace_back(file);
      }
      continue;
    } else if (!strcasecmp(elem->Value(), "texture")) {
      static const char *attributes[] = {"file",     "fileright", "fileup",
                                         "fileleft", "filedown",  "filefront",
                                         "fileback"};
      for (const auto &attribute : attributes) {
        const char *file = elem->Attribute(attribute);
        if (file != nullptr) {
          texture_files.emplace_back(file);
        }
      }
    }

    tinyxml2::XMLElement *child = elem->FirstChildElement();
    while (child) {
      elements.push(child);
      child = child->NextSiblingElement();
    }
  }

  accumulate_files(texture_files,
                   texture_dir == nullptr ? asset_dir : texture_dir);
  accumulate_files(mesh_files, mesh_dir == nullptr ? asset_dir : mesh_dir);
  accumulate_files(hfield_files, asset_dir);
}

void ResolveMjcfDependencies(const std::string &xml_string,
                             const std::string &resolved_path) {
  // load XML file or parse string
  tinyxml2::XMLDocument doc;
  doc.Parse(xml_string.c_str());

  // error checking
  if (doc.Error()) {
    TF_ERROR(XmlParsingError, "%d:\n%s\n", doc.ErrorID(), doc.ErrorStr());
    return;
  }

  // get top-level element
  tinyxml2::XMLElement *root = doc.RootElement();
  if (!root) {
    TF_ERROR(XmlParsingError, "XML root element not found");
    return;
  }

  // Accumulate file dependencies.
  std::unordered_set<std::string> files = {};
  AccumulateFiles(files, root, resolved_path);

  auto open_asset = [](const std::string &identifier) {
    pxr::ArGetResolver().OpenAsset(pxr::ArGetResolver().Resolve(identifier));
  };
  // Open all assets in parallel.
  pxr::WorkParallelForEach(files.begin(), files.end(), open_asset);
}
}  // namespace

UsdMjcfFileFormat::UsdMjcfFileFormat()
    : SdfFileFormat(
          UsdMjcfFileFormatTokens->Id, UsdMjcfFileFormatTokens->Version,
          UsdMjcfFileFormatTokens->Target, UsdMjcfFileFormatTokens->Id) {}

UsdMjcfFileFormat::~UsdMjcfFileFormat() {}

bool UsdMjcfFileFormat::CanRead(const std::string &filePath) const {
  auto extension = pxr::TfGetExtension(filePath);
  if (extension.empty()) {
    return false;
  }

  return extension == this->GetFormatId();
}

bool UsdMjcfFileFormat::ReadImpl(pxr::SdfLayer *layer, mjSpec *spec) const {
  auto args = layer->GetFileFormatArguments();
  auto data = InitData(args);

  pxr::SdfChangeBlock block;
  pxr::SdfLayerRefPtr spec_layer = pxr::SdfLayer::CreateAnonymous();
  auto success = mujoco::usd::WriteSpecToData(spec, spec_layer);
  mj_deleteSpec(spec);
  if (!success) {
    return false;
  }

  layer->TransferContent(spec_layer);

  return true;
}

bool UsdMjcfFileFormat::ReadFromString(pxr::SdfLayer *layer,
                                       const std::string &str) const {
  std::array<char, 1024> error;
  mjSpec *spec =
      mj_parseXMLString(str.c_str(), nullptr, error.data(), error.size());
  if (spec == nullptr) {
    TF_WARN(XmlParsingError, "%s", error.data());
    return false;
  }

  return ReadImpl(layer, spec);
}

bool UsdMjcfFileFormat::Read(pxr::SdfLayer *layer,
                             const std::string &resolved_path,
                             bool metadata_only) const {
  // Resolved all dependencies so that they are accessible when parsing
  // the XML.
  std::shared_ptr<pxr::ArAsset> asset =
      pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(resolved_path));
  auto buffer = asset->GetBuffer();
  ResolveMjcfDependencies(buffer.get(), resolved_path);

  // Parse to USD.
  std::array<char, 1024> error;
  mjSpec *spec =
      mj_parseXML(resolved_path.c_str(), nullptr, error.data(), error.size());

  if (spec == nullptr) {
    TF_WARN(XmlParsingError, "%s", error.data());
    return false;
  }
  return ReadImpl(layer, spec);
}

bool UsdMjcfFileFormat::WriteToString(const SdfLayer &layer, std::string *str,
                                      const std::string &comment) const {
  return SdfFileFormat::FindById(pxr::TfToken("usda"))
      ->WriteToString(layer, str, comment);
}

PXR_NAMESPACE_CLOSE_SCOPE
