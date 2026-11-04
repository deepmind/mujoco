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

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <stack>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Disable unused function warnings for miniz.
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "miniz.h"
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic pop
#endif

#include <mujoco/mujoco.h>
#include "user/user_util.h"
#include "tinyxml2.h"

namespace {

namespace fs = std::filesystem;


// asset file to be packed into the archive
struct AssetEntry {
  fs::path archive_path;
  // modelfiledir of the spec that originally owned the asset
  fs::path source_dir;
  fs::path disk_path;
};


// Key for identifying an asset element that needs its file path rewritten.
// Using (tag, file, name) allows disambiguation of unnamed elements and
// individual cubemap face files on the same texture element.
struct RewriteKey {
  std::string tag;
  std::string file;
  std::string name;

  RewriteKey(std::string_view t, std::string_view f, std::string_view n)
      : tag(t), file(f), name(n) {}

  bool operator==(const RewriteKey&) const = default;
};


struct RewriteKeyHash {
  std::size_t operator()(const RewriteKey& k) const {
    const std::size_t h1    = std::hash<std::string_view>{}(k.tag);
    const std::size_t h2    = std::hash<std::string_view>{}(k.file);
    const std::size_t h3    = std::hash<std::string_view>{}(k.name);
    std::size_t       seed  = h1;
    seed                   ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed                   ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};


using RewriteMap = std::unordered_map<RewriteKey, std::string, RewriteKeyHash>;


fs::path SanitizePath(const fs::path& path) {
  // replace characters that are not alphanumeric, '/', '.', '_', or '-'
  // with '_'
  std::string result = path.generic_string();
  for (char& c : result) {
    if (!std::isalnum(static_cast<unsigned char>(c)) &&
        c != '/' &&
        c != '.' &&
        c != '_' &&
        c != '-') {
      c = '_';
    }
  }
  return result;
}


// Remove leading ".." path components from a normalized path.
std::string RemoveLeadingDotDot(const fs::path& p) {
  fs::path result;
  bool     skipping = true;
  for (const auto& component : p) {
    if (skipping && component == "..") { continue; }
    skipping  = false;
    result   /= component;
  }
  return result.generic_string();
}


// Map a path to a relative path inside the archive.
fs::path LocalizePath(const fs::path& path) {
  // sanitize the path to remove any URI schemes or other non-path characters if the file path
  // has a URI scheme (e.g. "http://foo/mesh.stl"), strip it so the archive entry uses a
  // concrete path and decoding the MJZ won't try to invoke a resource provider. First remove
  // any leading ".." path components. my_provider:a/../b/c_$.obj ->
  // my_provider_a/../b/c__.obj
  const fs::path sanitized = SanitizePath(path);
  // my_provider_a/../b/c__.obj -> my_provider_a/b/c__.obj
  const fs::path normalized = sanitized.lexically_normal();
  // ../../b/c__.obj -> b/c__.obj
  // Also remove any leading '/'.
  return RemoveLeadingDotDot(normalized.relative_path());
}


// Apply path-attribute rewrites to the serialized XML.
// For each element, all attributes are checked against the rewrite map keyed
// by (tag_name, attribute_value, element_name). This handles regular "file"
// attributes, cubemap face attributes (fileright, fileleft, etc.), and the
// compiler's meshdir and texturedir.
void ApplyRewrites(std::string& xml, const RewriteMap& rewrites) {
  if (rewrites.empty()) return;

  tinyxml2::XMLDocument doc;
  if (doc.Parse(xml.c_str()) != tinyxml2::XML_SUCCESS) return;

  // Walk all elements in the document.
  std::stack<tinyxml2::XMLElement*> stack;
  if (auto* root = doc.RootElement()) stack.push(root);
  while (!stack.empty()) {
    auto* elem = stack.top();
    stack.pop();

    const char* tag_name  = elem->Value();
    const char* name_attr = elem->Attribute("name");

    // Check path attributes for values that need rewriting.
    // This covers "file", cubemap face attributes (fileright, fileleft, etc.), meshdir and
    // texturedir.
    for (const tinyxml2::XMLAttribute* attr = elem->FirstAttribute(); attr; attr = attr->Next()) {
      std::string_view attr_name{attr->Name()};
      if (!attr_name.starts_with("file") && attr_name != "meshdir" && attr_name != "texturedir") {
        continue;
      }

      const char* attr_value = attr->Value();
      if (!attr_value) continue;

      RewriteKey key{tag_name, attr_value, name_attr ? name_attr : ""};
      auto       it = rewrites.find(key);
      if (it != rewrites.end()) { elem->SetAttribute(attr->Name(), it->second.c_str()); }
    }

    for (auto* child = elem->FirstChildElement(); child; child = child->NextSiblingElement()) {
      stack.push(child);
    }
  }

  tinyxml2::XMLPrinter printer;
  doc.Print(&printer);
  xml = printer.CStr();
}


// Rewrite the attribute value in key to path if the texts differ; fs::path equality would ignore
// repeated separators ("sub//box.obj" == "sub/box.obj"), which normalizes the XML inside the
// archive.
void AddRewrite(RewriteMap& rewrites, const RewriteKey& key, const fs::path& path) {
  std::string value = path.generic_string();
  if (value != key.file) { rewrites[key] = std::move(value); }
}


// collect all referenced asset files from the spec
//
// returns a map from unique archive entry names to AssetEntry structs
// and populates xml_rewrites with XML path attributes that need to be updated.
std::unordered_map<std::string, AssetEntry> CollectAssets(const mjSpec* spec,
                                                          RewriteMap&   xml_rewrites) {
  const mjString* root_meshdir    = spec->compiler.meshdir;
  const mjString* root_texturedir = spec->compiler.texturedir;

  // directories of meshdir and texturedir assets in the archive; the compiler attributes are
  // rewritten to these, and the decoder resolves file attributes against them
  const fs::path mesh_dir    = LocalizePath(*root_meshdir);
  const fs::path texture_dir = LocalizePath(*root_texturedir);

  // maps (root dir, full disk path) to the archived file's path relative to the root dir
  std::map<std::pair<fs::path, fs::path>, fs::path> archived_paths;
  // all asset entries, keyed by the final archive path
  std::unordered_map<std::string, AssetEntry> archive_entries;

  auto process =
      [&](mjsElement* elem, const mjString* raw_file, bool use_meshdir, std::string_view tag_name) {
        // skip empty files
        if (!raw_file || raw_file->empty()) return;

        const mjString* elem_name   = mjs_getName(elem);
        const mjSpec*   owning_spec = mjs_getOriginSpec(elem);

        mjsCompiler* comp = mjs_getCompiler(elem);
        if (!comp) {
          mju_error(
              "MJZ encoder: no compiler for element '%s', this should never "
              "happen.",
              elem_name->c_str());
        }

        const std::string file_str     = spec->strippath ? mjuu_strippath(*raw_file) : *raw_file;
        auto              add_rewrites = [&](const fs::path& target) {
          AddRewrite(xml_rewrites, RewriteKey{tag_name, *raw_file, *elem_name}, target);
          if (file_str != *raw_file) {
            AddRewrite(xml_rewrites, RewriteKey{tag_name, file_str, *elem_name}, target);
          }
        };

        const fs::path owning_spec_dir{owning_spec->modelfiledir ? *owning_spec->modelfiledir : ""};
        const fs::path raw_path{file_str};
        const fs::path prefix_dir{use_meshdir ? *comp->meshdir : *comp->texturedir};
        const fs::path& root_dir = use_meshdir ? mesh_dir : texture_dir;

        const char* ch     = std::strchr(file_str.c_str(), ':');
        const bool  is_uri = ch != nullptr;

        // full path of the asset relative to its spec, or absolute/URI if the raw path is not
        // relative.
        fs::path full_spec_path = raw_path;
        // If the raw path is relative AND it does not have a valid resource provider URI, then we
        // need to prefix it with the specs meshdir/texturedir.
        if (!is_uri && raw_path.is_relative()) { full_spec_path = prefix_dir / raw_path; }

        // full path of the asset regardless of spec, URI and absolute paths remain unchanged but
        // relative paths are prefixed with modelfiledir
        fs::path full_path = full_spec_path;
        if (!is_uri && full_spec_path.is_relative()) {
          full_path = owning_spec_dir / full_spec_path;
        }

        const fs::path localized = LocalizePath(raw_path);

        // If this file was already archived, we may still need to add a rewrite if raw_file was
        // sanitized or collision-renamed for the first occurrence, or is spelled differently here.
        if (auto it = archived_paths.find({root_dir, full_path}); it != archived_paths.end()) {
          add_rewrites(it->second);
          return;
        }

        // Collision renaming: if this archive path is already in use, try again with an incremented
        // suffix.
        fs::path file      = localized;
        fs::path parent    = file.parent_path();
        fs::path stem      = file.stem();
        fs::path extension = file.extension();
        for (int i = 0; archive_entries.contains((root_dir / file).generic_string()); ++i) {
          std::string new_name = stem.string() + "_" + std::to_string(i) + extension.string();
          file                 = parent / new_name;
        }

        add_rewrites(file);

        // path relative to the root XML in the archive, named with '/' separators on every platform
        const fs::path archive_path           = root_dir / file;
        archived_paths[{root_dir, full_path}] = file;
        archive_entries[archive_path.generic_string()] =
            AssetEntry{archive_path, owning_spec_dir, full_spec_path};
      };

  // Meshes, heightfields, and skins use meshdir.
  {
    const mjsMesh* mesh = mjs_asMesh(mjs_firstElement(spec, mjOBJ_MESH));
    while (mesh != nullptr) {
      process(mesh->element, mesh->file, /*use_meshdir=*/true, "mesh");
      mesh = mjs_asMesh(mjs_nextElement(spec, mesh->element));
    }
  }

  {
    const mjsHField* hf = mjs_asHField(mjs_firstElement(spec, mjOBJ_HFIELD));
    while (hf != nullptr) {
      process(hf->element, hf->file, /*use_meshdir=*/true, "hfield");
      hf = mjs_asHField(mjs_nextElement(spec, hf->element));
    }
  }

  {
    const mjsSkin* skin = mjs_asSkin(mjs_firstElement(spec, mjOBJ_SKIN));
    while (skin != nullptr) {
      process(skin->element, skin->file, /*use_meshdir=*/true, "skin");
      skin = mjs_asSkin(mjs_nextElement(spec, skin->element));
    }
  }

  // Textures use texturedir.
  {
    const mjsTexture* tex = mjs_asTexture(mjs_firstElement(spec, mjOBJ_TEXTURE));
    while (tex != nullptr) {
      process(tex->element, tex->file, /*use_meshdir=*/false, "texture");
      if (tex->cubefiles) {
        for (const mjString& file : *tex->cubefiles) {
          process(tex->element, &file, /*use_meshdir=*/false, "texture");
        }
      }
      tex = mjs_asTexture(mjs_nextElement(spec, tex->element));
    }
  }

  AddRewrite(xml_rewrites, {"compiler", *root_meshdir, ""}, mesh_dir);
  AddRewrite(xml_rewrites, {"compiler", *root_texturedir, ""}, texture_dir);

  return archive_entries;
}


mjtSize MjzEncode(const mjSpec*  spec,
                  const mjModel* model,
                  const mjVFS*   vfs,
                  mjResource*    resource) {
  if (!spec || !resource) { return -1; }

  // Collect assets and compute archive paths + XML rewrite list.
  RewriteMap                                  xml_rewrites;
  std::unordered_map<std::string, AssetEntry> assets = CollectAssets(spec, xml_rewrites);

  // Serialize spec to XML.
  char              error[1024] = {0};
  int               xml_sz      = 1024 * 1024;
  std::vector<char> xml_buf(xml_sz);
  int               result = mj_saveXMLString(spec, xml_buf.data(), xml_sz, error, sizeof(error));
  // mj_saveXMLString returns 0 on success, -1 on failure, or a positive value
  // indicating the required buffer size when the buffer is too small.
  if (result > 0) {
    xml_sz = result + 1;
    xml_buf.resize(xml_sz);
    result = mj_saveXMLString(spec, xml_buf.data(), xml_sz, error, sizeof(error));
  }
  if (result != 0) {
    mju_warning("MJZ encoder: failed to serialize spec to XML: %s", error);
    return -1;
  }

  // Post-process the XML: apply file attribute rewrites for collision-renamed
  // assets. Only colliding entries get rewritten; non-colliding models are
  // unchanged.
  std::string xml_str(xml_buf.data());
  ApplyRewrites(xml_str, xml_rewrites);

  // NOLINTBEGIN(misc-include-cleaner) - Disable clang-tidy for miniz.
  // Initialize zip writer.
  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  if (!mz_zip_writer_init_heap(&zip, 0, 0)) {
    mju_warning("MJZ encoder: failed to init zip writer");
    return -1;
  }

  // Add XML to archive.
  const std::string xml_name = "model.xml";
  if (!mz_zip_writer_add_mem(&zip,
                             xml_name.c_str(),
                             xml_str.data(),
                             xml_str.size(),
                             MZ_DEFAULT_COMPRESSION)) {
    mju_warning("MJZ encoder: failed to add XML to archive");
    mz_zip_writer_end(&zip);
    return -1;
  }

  // Pack each asset file into the archive using the unique archive paths.
  for (const auto& [archive_entry, entry] : assets) {
    mjResource* res = mju_openResource(entry.source_dir.string().c_str(),
                                       entry.disk_path.string().c_str(),
                                       vfs,
                                       error,
                                       sizeof(error));
    if (!res) {
      mju_warning("MJZ encoder: failed to open resource '%s' (dir='%s'): %s",
                  entry.disk_path.string().c_str(),
                  entry.source_dir.string().c_str(),
                  error);
      continue;
    }

    const void* buf    = nullptr;
    int         nbytes = mju_readResource(res, &buf);
    if (nbytes > 0 && buf) {
      if (!mz_zip_writer_add_mem(&zip,
                                 archive_entry.c_str(),
                                 buf,
                                 nbytes,
                                 MZ_DEFAULT_COMPRESSION)) {
        mju_warning("MJZ encoder: failed to write '%s' to archive", archive_entry.c_str());
      }
    }
    mju_closeResource(res);
  }

  // Finalize archive.
  void*  archive_buf  = nullptr;
  size_t archive_size = 0;
  if (!mz_zip_writer_finalize_heap_archive(&zip, &archive_buf, &archive_size)) {
    mju_warning("MJZ encoder: failed to finalize archive");
    mz_zip_writer_end(&zip);
    return -1;
  }
  mz_zip_writer_end(&zip);
  // NOLINTEND(misc-include-cleaner) - End clang-tidy disable for miniz.

  resource->data = archive_buf;

  return static_cast<mjtSize>(archive_size);
}


void MjzCloseResource(mjResource* resource) {
  if (resource && resource->data) {
    std::free(resource->data);
    resource->data = nullptr;
  }
}
}  // namespace


mjPLUGIN_LIB_INIT(mjz_encoder) {
  mjpEncoder encoder;
  mjp_defaultEncoder(&encoder);
  encoder.content_type   = "application/zip";
  encoder.extension      = ".mjz|.zip";
  encoder.encode         = MjzEncode;
  encoder.close_resource = MjzCloseResource;
  mjp_registerEncoder(&encoder);
}
