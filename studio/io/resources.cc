// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "studio/io/resources.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32) || defined(__CYGWIN__)
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

#include <mujoco/mujoco.h>

namespace mujoco::studio {

namespace {

std::string Resolve(std::string_view path) {
  std::string_view subpath = path.substr(path.find(':') + 1);
  std::filesystem::path module_dir = GetModuleDir((void*)&Resolve);
  if (module_dir.empty()) {
    return std::string("assets/") + std::string(subpath);
  }
  std::filesystem::path resources_dir = module_dir.parent_path() / "Resources";
  if (std::filesystem::exists(resources_dir / "assets")) {
    return (resources_dir / "assets" / subpath).string();
  }
  if (std::filesystem::exists(module_dir / "assets")) {
    return (module_dir / "assets" / subpath).string();
  }
  // For a module at the Python package root such as mujoco/_render_filament,
  // assets are staged in the studio package directory.
  std::filesystem::path studio_assets = module_dir / "studio" / "assets";
  if (std::filesystem::exists(studio_assets)) {
    return (studio_assets / subpath).string();
  }
  return (module_dir / "assets" / subpath).string();
}

class FileResource {
 public:
  explicit FileResource(const std::string& path) : file_(path, std::ios::binary | std::ios::ate) {
    if (!file_.is_open()) {
      mju_warning("Cannot open file %s", path.c_str());
      return;
    }

    size_ = file_.tellg();
    file_.seekg(0, std::ios::beg);
  }

  int Read(const void** buffer) {
    buffer_.resize(size_);
    if (!file_.read(reinterpret_cast<char*>(buffer_.data()), size_)) {
      return 0;
    }
    *buffer = buffer_.data();
    return size_;
  }

  int Size() const { return size_; }

  FileResource(const FileResource&) = delete;
  FileResource& operator=(const FileResource&) = delete;

 private:
  std::ifstream file_;
  std::vector<char> buffer_;
  int size_ = 0;
};

}  // namespace

std::string GetModuleDir(void* addr) {
#if defined(_WIN32) || defined(__CYGWIN__)
  HMODULE hModule = NULL;
  if (GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)addr, &hModule)) {
    char path[MAX_PATH];
    DWORD written = GetModuleFileNameA(hModule, path, sizeof(path));
    if (written > 0 && written < sizeof(path)) {
      return std::filesystem::path(path).parent_path().string();
    }
  }
  return "";
#else
  Dl_info info;
  if (dladdr(addr, &info) != 0 && info.dli_fname != nullptr) {
    std::filesystem::path p(info.dli_fname);
    if (p.is_absolute()) {
      return p.parent_path().string();
    }
    std::error_code ec;
    std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
      return (cwd / p).parent_path().string();
    }
    return p.parent_path().string();
  }
  return "";
#endif
}

void RegisterProviders() {
  // Every executable and extension module links its own copy of this
  // library, but the engine's provider table is per process and
  // registering a prefix twice is an error: the first caller's providers
  // serve everyone.
  if (mjp_findProvider("font:") != nullptr) {
    return;
  }

  mjpResourceProvider resource_provider;
  mjp_defaultProvider(&resource_provider);

  resource_provider.open = [](mjResource* resource) {
    const std::string resolved_path = Resolve(resource->name);
    FileResource* f = new FileResource(resolved_path);
    if (f->Size() == 0) {
      delete f;
      return 0;
    }
    resource->data = f;
    return f->Size();
  };
  resource_provider.read = [](mjResource* resource, const void** buffer) {
    FileResource* f = static_cast<FileResource*>(resource->data);
    return f->Read(buffer);
  };
  resource_provider.close = [](mjResource* resource) {
    delete static_cast<FileResource*>(resource->data);
    resource->data = nullptr;
  };

  resource_provider.prefix = "font";
  mjp_registerProvider(&resource_provider);
  resource_provider.prefix = "filament";
  mjp_registerProvider(&resource_provider);
}

}  // namespace mujoco::studio
