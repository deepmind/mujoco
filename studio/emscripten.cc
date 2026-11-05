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

// Main entry point for the Filament-based MuJoCo web app.

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "studio/hal/graphics_mode.h"
#include "studio/app.h"

// Global app instance. Lifetime is controlled by Init/Deinit calls which are
// triggered by Javascript.
mujoco::studio::App* g_app = nullptr;

// Static registry of assets that are loaded in JSON before the main App is
// initialized.
class AssetRegistry {
 public:
  // Returns the singleton instance of the registry.
  static AssetRegistry& Instance() {
    static AssetRegistry instance;
    return instance;
  }

  // Registers asset contents with the given filename.
  void RegisterAsset(std::string filename, std::string contents) {
    filename = std::filesystem::path(filename).filename().string();
    assets_[filename] = std::move(contents);
  }

  const std::string& Get(std::string_view filename) const {
    filename = filename.substr(filename.find_first_of(':') + 1);
    static std::string empty;
    auto it = assets_.find(std::string(filename));
    return it != assets_.end() ? it->second : empty;
  }

 private:
  std::unordered_map<std::string, std::string> assets_;
};

// ---------------------------------------------------------------------------
// HTTP/HTTPS resource fetching via the JS fetch API (uses ASYNCIFY to yield).
// ---------------------------------------------------------------------------

// Fetches a URL using the JS fetch API. Returns a malloc'd buffer and its size.
// Also extracts the filename from the Content-Disposition header if present.
// The caller is responsible for freeing out_data and out_filename.
EM_ASYNC_JS(int, FetchUrl,
            (const char* url, char** out_data, std::int32_t* out_size, char** out_filename), {
              const urlStr = UTF8ToString(url);
              const dlEl = document.getElementById("loadingDownload");
              const dlFileEl = document.getElementById("loadingDownloadFile");
              if (dlEl) {
                dlEl.style.display = "block";
              }
              try {
                const response = await fetch(urlStr);
                if (!response.ok) {
                  console.error("Fetch failed: " + response.status + " " + urlStr);
                  return 0;
                }
                const buffer = await response.arrayBuffer();
                const bytes = new Uint8Array(buffer);
                const ptr = _malloc(bytes.length);
                HEAPU8.set(bytes, ptr);
                setValue(out_data, ptr, "*");
                setValue(out_size, bytes.length, "i32");

                // Extract filename from Content-Disposition header if present.
                const cd = response.headers.get("content-disposition") || "";
                let filename = "";
                const cdParts = cd.split(";");
                for (let i = 0; i < cdParts.length; ++i) {
                  const part = cdParts[i].trim();
                  if (part.toLowerCase().startsWith("filename=")) {
                    let fn = part.substring(9).trim();
                    if (fn.startsWith("\"") && fn.endsWith("\"")) {
                      fn = fn.substring(1, fn.length - 1);
                    }
                    filename = fn;
                    break;
                  }
                }
                if (out_filename && filename) {
                  const fnLen = lengthBytesUTF8(filename) + 1;
                  const fnPtr = _malloc(fnLen);
                  stringToUTF8(filename, fnPtr, fnLen);
                  setValue(out_filename, fnPtr, "*");
                } else if (out_filename) {
                  setValue(out_filename, 0, "*");
                }

                if (dlFileEl) {
                  dlFileEl.textContent = (filename || urlStr.split("/").pop() || urlStr) + "\u2026";
                }
                return 1;
              } catch (e) {
                console.error("Fetch error:", e);
                return 0;
              } finally {
                if (dlEl) {
                  dlEl.style.display = "none";
                }
                if (dlFileEl) {
                  dlFileEl.textContent = "";
                }
              }
            });

// Cache entry for data fetched via HTTP/HTTPS.
struct FetchEntry {
  std::string url;
  std::string data;
  std::string filename;
  int refcount = 0;
};

// Cache for data fetched via HTTP/HTTPS. Stores downloaded (or pre-primed)
// bytes keyed by the resource URL so that the resource provider's read()
// can return a pointer into stable storage.
class FetchCache {
 public:
  static FetchCache& Instance() {
    static FetchCache instance;
    return instance;
  }

  // Returns the size of the entry for `url`, fetching it over the network
  // if necessary. Entries that have been pre-populated via Prime() are
  // returned without touching the network. Returns 0 on fetch failure.
  int Fetch(const char* url) {
    if (auto it = entries_.find(url); it != entries_.end()) {
      int size = static_cast<int>(it->second.data.size());
      if (size > 0) {
        ++it->second.refcount;
      }
      return size;
    }
    char* buf = nullptr;
    std::int32_t size = 0;
    char* filename_str = nullptr;
    if (!FetchUrl(url, &buf, &size, &filename_str)) {
      return 0;
    }
    std::string data(buf, size);
    std::free(buf);

    std::string filename = filename_str ? filename_str : "";
    if (filename_str) std::free(filename_str);

    FetchEntry entry{url, std::move(data), std::move(filename)};
    auto& stored = entries_.emplace(url, std::move(entry)).first->second;
    int stored_size = static_cast<int>(stored.data.size());
    if (stored_size > 0) {
      ++stored.refcount;
    }
    return stored_size;
  }

  const FetchEntry* GetEntry(const char* url) const {
    auto it = entries_.find(url);
    if (it == entries_.end()) return nullptr;
    return &it->second;
  }

  // Pre-populates the cache so a subsequent Fetch() of the same URL hits
  // memory instead of the network. First writer wins.
  void Prime(std::string url, std::string bytes) { Prime(std::move(url), std::move(bytes), ""); }

  void Prime(std::string url, std::string bytes, std::string filename) {
    std::string url_copy = url;
    entries_.try_emplace(std::move(url),
                         FetchEntry{std::move(url_copy), std::move(bytes), std::move(filename)});
  }

  // Hands back a pointer into the cached bytes for the given URL.
  // Returns -1 if the URL has not been fetched or primed.
  int Read(const char* url, const void** buffer) const {
    auto it = entries_.find(url);
    if (it == entries_.end()) return -1;
    *buffer = it->second.data.data();
    return static_cast<int>(it->second.data.size());
  }

  void Close(const char* url) {
    auto it = entries_.find(url);
    if (it == entries_.end()) {
      return;
    }
    if (--it->second.refcount <= 0) {
      entries_.erase(it);
    }
  }

 private:
  std::unordered_map<std::string, FetchEntry> entries_;
};

// ---------------------------------------------------------------------------

// Javascript-facing function to register a build-time asset (font, IBL,
// .filamat material) that is fetched via the static AssetRegistry rather
// than over HTTP.
void RegisterAsset(std::string filename, std::string contents) {
  AssetRegistry::Instance().RegisterAsset(std::move(filename), std::move(contents));
}

// Javascript-facing function to pre-populate the HTTP/HTTPS fetch cache.
// Used by the JS-side parallel prefetcher to feed already-downloaded asset
// bytes to MuJoCo's compiler before LoadUrl is invoked, sidestepping the
// otherwise strictly-serial EM_ASYNC_JS fetch chain.
void PrimeFetchCache(std::string url, std::string bytes) {
  FetchCache::Instance().Prime(std::move(url), std::move(bytes));
}

// Javascript-facing wrapper around mju_getXMLDependencies. Returns the
// list of resource URLs that the given root XML transitively references
// (meshes, textures, includes, ...), per MuJoCo's compiler rules.
std::vector<std::string> GetXMLDependencies(const std::string& root_url) {
  mjStringVec deps;
  mju_getXMLDependencies(root_url.c_str(), &deps);
  return deps;
}

// Javascript-facing function to initialize the app.
void Init(const std::string& title, bool dark_theme) {
  // Note: dimensions do not matter as window will be resized to fit canvas.
  const int width = 100;
  const int height = 100;
  const std::string ini_path = "";

  mjpResourceProvider resource_provider;
  mjp_defaultProvider(&resource_provider);

  resource_provider.open = [](mjResource* resource) {
    AssetRegistry& r = AssetRegistry::Instance();
    return static_cast<int>(r.Get(resource->name).size());
  };
  resource_provider.read = [](mjResource* resource, const void** buffer) {
    AssetRegistry& r = AssetRegistry::Instance();
    const std::string& contents = r.Get(resource->name);
    *buffer = contents.data();
    return static_cast<int>(contents.size());
  };
  resource_provider.close = [](mjResource* resource) {};

  resource_provider.prefix = "font";
  mjp_registerProvider(&resource_provider);
  resource_provider.prefix = "filament";
  mjp_registerProvider(&resource_provider);

  // Register HTTP/HTTPS resource providers so that models loaded from URLs
  // can automatically fetch referenced assets (meshes, textures, etc.) over
  // the network.
  mjpResourceProvider http_provider;
  mjp_defaultProvider(&http_provider);

  http_provider.open = [](mjResource* resource) {
    int size = FetchCache::Instance().Fetch(resource->name);
    if (size <= 0) return 0;
    const FetchEntry* entry = FetchCache::Instance().GetEntry(resource->name);
    if (entry) {
      // Store the FetchEntry pointer in resource->data for fast access in
      // read() and to recover the original URL in close().
      resource->data = const_cast<void*>(static_cast<const void*>(entry));

      // If the HTTP response provided a filename (e.g. from the
      // Content-Disposition header set during asset publication), update
      // resource->name while qualifying it with the URL path so downstream
      // decoders (such as mjz_decoder) can identify the file extension and
      // mount each resource under a unique VFS path without colliding even if
      // multiple distinct HTTP resources share identical filenames.
      if (!entry->filename.empty()) {
        std::string base_url = entry->url;
        auto qpos = base_url.find('?');
        if (qpos != std::string::npos) {
          base_url = base_url.substr(0, qpos);
        }
        if (!base_url.ends_with(entry->filename) && !base_url.ends_with("/" + entry->filename)) {
          if (!base_url.empty() && base_url.back() != '/') {
            base_url += '/';
          }
          base_url += entry->filename;
        }
        delete[] resource->name;
        resource->name = new char[base_url.size() + 1];
        std::memcpy(resource->name, base_url.c_str(), base_url.size() + 1);
      }
    }
    return size;
  };
  http_provider.read = [](mjResource* resource, const void** buffer) {
    if (resource->data) {
      const FetchEntry* entry = static_cast<const FetchEntry*>(resource->data);
      *buffer = entry->data.data();
      return static_cast<int>(entry->data.size());
    }
    return FetchCache::Instance().Read(resource->name, buffer);
  };
  http_provider.close = [](mjResource* resource) {
    if (resource->data) {
      const FetchEntry* entry = static_cast<const FetchEntry*>(resource->data);
      std::string url = entry->url;
      // Reset resource->data so VFS cleanup does not report an un-cleared data
      // pointer.
      resource->data = nullptr;
      // Release cached download memory once the resource is closed by MuJoCo.
      FetchCache::Instance().Close(url.c_str());
    }
  };

  http_provider.prefix = "http";
  mjp_registerProvider(&http_provider);
  http_provider.prefix = "https";
  mjp_registerProvider(&http_provider);

  // Register a "github:" resource provider that resolves
  // github:org/repo/branch/path/to/file.xml to
  // https://raw.githubusercontent.com/org/repo/branch/path/to/file.xml
  mjpResourceProvider github_provider;
  mjp_defaultProvider(&github_provider);

  github_provider.open = [](mjResource* resource) {
    std::string name(resource->name);
    // Strip the "github:" prefix and prepend the raw.githubusercontent URL.
    std::string url = "https://raw.githubusercontent.com/" + name.substr(strlen("github:"));
    int size = FetchCache::Instance().Fetch(url.c_str());
    if (size <= 0) return 0;
    const FetchEntry* entry = FetchCache::Instance().GetEntry(url.c_str());
    if (entry) {
      resource->data = const_cast<void*>(static_cast<const void*>(entry));
      if (!entry->filename.empty()) {
        std::string base_url = url;
        if (!base_url.ends_with(entry->filename) && !base_url.ends_with("/" + entry->filename)) {
          if (!base_url.empty() && base_url.back() != '/') {
            base_url += '/';
          }
          base_url += entry->filename;
        }
        delete[] resource->name;
        resource->name = new char[base_url.size() + 1];
        std::memcpy(resource->name, base_url.c_str(), base_url.size() + 1);
      }
    }
    return size;
  };
  github_provider.read = [](mjResource* resource, const void** buffer) {
    if (resource->data) {
      const FetchEntry* entry = static_cast<const FetchEntry*>(resource->data);
      *buffer = entry->data.data();
      return static_cast<int>(entry->data.size());
    }
    std::string name(resource->name);
    std::string url = "https://raw.githubusercontent.com/" + name.substr(strlen("github:"));
    return FetchCache::Instance().Read(url.c_str(), buffer);
  };
  github_provider.close = [](mjResource* resource) {
    if (resource->data) {
      const FetchEntry* entry = static_cast<const FetchEntry*>(resource->data);
      std::string url = entry->url;
      resource->data = nullptr;
      FetchCache::Instance().Close(url.c_str());
    }
  };

  github_provider.prefix = "github";
  mjp_registerProvider(&github_provider);

  mujoco::studio::App::Config config;
  config.width = width;
  config.height = height;
  config.ini_path = ini_path;
  config.gfx_mode = mujoco::studio::GraphicsMode::FilamentWebGl;
  config.initial_theme =
      dark_theme ? mujoco::studio::GuiTheme::kDark : mujoco::studio::GuiTheme::kLight;
  if (!title.empty()) {
    config.title = title;
  }
  g_app = new mujoco::studio::App(std::move(config));
  g_app->InitEmptyModel();
}

// Javascript-facing function to load a model from an uploaded file.
void LoadFile(const std::string& filename, const std::string& data) {
  if (!g_app) {
    return;
  }

  std::string content_type;
  if (filename.ends_with(".mjb")) {
    content_type = "application/mjb";
  } else if (filename.ends_with(".mjz")) {
    content_type = "application/zip";
  } else if (filename.ends_with(".zip")) {
    content_type = "application/zip";
  } else if (filename.ends_with(".xml")) {
    content_type = "text/xml";
  } else {
    return;
  }

  const auto ptr = reinterpret_cast<const std::byte*>(data.data());
  g_app->LoadModelFromBuffer({ptr, ptr + data.size()}, content_type, filename);
}

// Javascript-facing function to load a model from a URL.
// The URL is passed directly to LoadModelFromFile, which will use the
// registered HTTP/HTTPS resource providers to fetch the model and any
// referenced assets.
void LoadUrl(const std::string& url) {
  if (!g_app) {
    return;
  }
  g_app->LoadModelFromFile(url);
}

// Javascript-facing function to select and load a keyframe by name or index.
void LoadKeyframe(const std::string& keyframe) {
  if (!g_app) {
    return;
  }
  g_app->LoadKeyframe(keyframe);
}

// Javascript-facing function to render a single frame.
void RenderFrame() {
  if (g_app) {
    if (g_app->Update()) {
      g_app->BuildGui();
      g_app->Render();
    }
  }
}

// Javascript-facing function to deinitialize the app.
void Deinit() {
  delete g_app;
  g_app = nullptr;
}

EMSCRIPTEN_BINDINGS(studio_bindings) {
  // Matches the C-level `mjStringVec` typedef in mjspec.h.
  emscripten::register_vector<std::string>("mjStringVec");
  emscripten::function("registerAsset", &RegisterAsset);
  emscripten::function("primeFetchCache", &PrimeFetchCache);
  emscripten::function("getXMLDependencies", &GetXMLDependencies);
  emscripten::function("init", &Init);
  emscripten::function("loadFile", &LoadFile);
  emscripten::function("loadUrl", &LoadUrl);
  emscripten::function("loadKeyframe", &LoadKeyframe);
  emscripten::function("renderFrame", &RenderFrame);
  emscripten::function("deinit", &Deinit);
}
