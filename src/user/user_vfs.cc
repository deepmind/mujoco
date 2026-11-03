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

#include "user/user_vfs.h"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "user/user_util.h"

namespace {

// struct for holding the contents of a file
struct ResourceFileData {
  std::vector<uint8_t> contents;
  time_t               modified_time = 0;
  bool                 is_read       = false;
};

int OpenFile(const char* filename, mjResource* resource) {
  struct stat file_stat;
  if (mju_stat(filename, &file_stat) == 0) {
    ResourceFileData* data = new ResourceFileData();
    resource->data         = data;

    data->modified_time = file_stat.st_mtime;
    mju_encodeBase64(resource->timestamp, (uint8_t*)&file_stat.st_mtime, sizeof(time_t));
    return 1;
  }
  return 0;
}

int ReadFile(const char* filename, mjResource* resource, const void** buffer) {
  ResourceFileData* data = (ResourceFileData*)resource->data;
  if (!data->is_read) {
    data->contents = mujoco::user::FileToMemory(filename);
    data->is_read  = true;
  }
  *buffer = data->contents.data();
  return static_cast<int>(data->contents.size());
}

void CloseFile(mjResource* resource) {
  delete (ResourceFileData*)resource->data;
  resource->data = nullptr;
}

int FileModified(const mjResource* resource, const char* timestamp) {
  if (mju_isValidBase64(timestamp) != sizeof(time_t)) { return 1; }
  time_t time;
  mju_decodeBase64((uint8_t*)&time, timestamp);

  ResourceFileData* data = (ResourceFileData*)resource->data;
  const double      diff = difftime(data->modified_time, time);

  if (diff < 0) return -1;
  if (diff > 0) return 1;
  return 0;
}

std::string StripPathAndLower(std::string path) {
  std::size_t n = path.find_last_of("/\\");
  if (n != std::string::npos) { path = path.substr(n + 1); }
  std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return path;
}

}  // namespace

namespace mujoco::user {

VFS::VFS(mjVFS* vfs) {
  wrapped_vfs_.impl_ = this;
  mjp_defaultProvider(&default_provider_);
  default_provider_.open = [](mjResource* res) { return OpenFile(res->name, res); };
  default_provider_.read = [](mjResource* res, const void** buffer) {
    return ReadFile(res->name, res, buffer);
  };
  default_provider_.close    = [](mjResource* res) { CloseFile(res); };
  default_provider_.modified = [](const mjResource* res, const char* time) {
    return FileModified(res, time);
  };
  default_provider_.write = [](mjResource* res, const void* buffer, mjtSize nbytes) -> mjtSize {
    FILE* fp = mju_fopen(res->name, "wb");
    if (!fp) { return -1; }
    mjtSize written = static_cast<mjtSize>(fwrite(buffer, 1, nbytes, fp));
    fclose(fp);
    return written == nbytes ? written : -1;
  };

  default_provider_.prefix = nullptr;
  default_mount_.vfs       = &wrapped_vfs_;
  default_mount_.provider  = &default_provider_;
  default_mount_.data      = nullptr;
  default_mount_.name      = nullptr;
}

VFS::~VFS() {
  if (!open_resources_.empty()) {
    mju_warning("VFS destroyed with %zu open resources. Resources will be invalidated.",
                open_resources_.size());
  }
  for (auto& [ptr, res] : open_resources_) {
    if (res->provider->close) { res->provider->close(res.get()); }
  }
  open_resources_.clear();

  for (auto& [path, res] : archive_mounts_) {
    if (res->provider->unmount) { res->provider->unmount(res.get()); }
  }
  archive_mounts_.clear();

  for (auto& [path, res] : mounts_) {
    if (res->provider->unmount) { res->provider->unmount(res.get()); }
  }
  mounts_.clear();
}

mjResource* VFS::Open(const char* dir, const char* name, char* error, size_t nerror) {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ++in_flight_open_;
  }

  const std::string path = FilePath(dir, name).Str();

  const mjResource* mount = FindMount(path);
  if (!mount || !mount->provider) {
    if (error) { std::snprintf(error, nerror, "No provider found for '%s'", name); }

    {
      std::lock_guard<std::recursive_mutex> lock(mutex_);
      --in_flight_open_;
    }
    MaybeSelfDestruct();
    return nullptr;
  }

  const mjpResourceProvider* provider = mount->provider;

  ResourcePtr res = CreateResource(path.c_str(), provider);

  // Smuggle the mounted resource provider's mount-specific data pointer in the
  // requested resource's data pointer. This allows the provider to access its
  // own per-mount data without any intrusive changes to the provider interface.
  res->data        = mount->data;
  const int result = provider->open ? provider->open(res.get()) : 0;
  // If the data pointer was not modified, then that means the resource did not
  // set its own data pointer. So, we need to set it back to nullptr.
  if (res->data == mount->data) { res->data = nullptr; }

  if (result == 0) {
    res.reset();
    {
      std::lock_guard<std::recursive_mutex> lock(mutex_);
      --in_flight_open_;
    }
    MaybeSelfDestruct();
    return nullptr;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  --in_flight_open_;
  mjResource* res_ptr = res.get();
  open_resources_.emplace(res_ptr, std::move(res));
  return res_ptr;
}

VFS::Status VFS::Mount(const FilePath& path, const mjpResourceProvider* provider) {
  if (!provider) { return kInvalidProvider; }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mounts_.contains(path.Str())) { return kRepeatedName; }

  ResourcePtr res     = CreateResource(path.c_str(), provider);
  mjResource* res_ptr = res.get();
  mounts_.emplace(path.Str(), std::move(res));
  if (provider->mount) { provider->mount(res_ptr); }
  return kSuccess;
}

VFS::Status VFS::Close(mjResource* res) {
  VFS::Status status        = kInvalidResource;
  bool        last_resource = false;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (auto it = open_resources_.find(res); it != open_resources_.end()) {
      if (res->provider->close) { res->provider->close(res); }
      open_resources_.erase(it);
      last_resource = open_resources_.empty() && in_flight_open_ == 0;
      status        = kSuccess;
    }
  }

  if (status == kSuccess && last_resource) { MaybeSelfDestruct(); }
  return status;
}

VFS::Status VFS::Unmount(const FilePath& path) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const std::string                     path_str = path.Str();
  const std::string                     stripped = StripPathAndLower(path_str);
  for (auto it = archive_mounts_.begin(); it != archive_mounts_.end();) {
    const std::string& arch_path = it->first;
    const bool is_child = arch_path.starts_with(path_str) &&
                          arch_path.size() > path_str.size() &&
                          (arch_path[path_str.size()] == '/' || arch_path[path_str.size()] == '\\');
    if (arch_path == path_str || is_child || StripPathAndLower(arch_path) == stripped) {
      if (it->second->provider->unmount) { it->second->provider->unmount(it->second.get()); }
      it = archive_mounts_.erase(it);
    } else {
      ++it;
    }
  }
  if (auto it = mounts_.find(path_str); it != mounts_.end()) {
    if (it->second->provider->unmount) { it->second->provider->unmount(it->second.get()); }
    mounts_.erase(it);
    return kSuccess;
  }
  return kInvalidProvider;
}

bool VFS::ContainsBuffer(const char* name) {
  if (name == nullptr) { return false; }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return mounts_.contains(name);
}

bool VFS::ContainsFile(const char* directory, const char* filename) {
  if (filename == nullptr) { return false; }
  mujoco::user::FilePath                path(directory ? directory : "", filename);
  std::string                           key = path.StripPath().Lower().Str();
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return mounts_.contains(key);
}

bool VFS::InMemory(const char* dir, const char* name) {
  if (name == nullptr) { return false; }
  const std::string path = FilePath(dir ? dir : "", name).Str();
  if (mjp_findProvider(path.c_str())) { return false; }
  const mjResource* mount = FindMount(path);
  return mount && mount != &default_mount_;
}

int VFS::Read(mjResource* resource, const void** buffer) {
  if (resource && resource->provider && resource->provider->read) {
    return resource->provider->read(resource, buffer);
  }
  return kFailedToRead;
}

mjtSize VFS::Write(mjResource* resource, const void* buffer, mjtSize nbytes) {
  if (resource) {
    if (!resource->provider) {
      const mjResource* mount = FindMount(resource->name);
      if (mount) {
        resource->provider = mount->provider;
        // Smuggle mount data if not already set.
        if (!resource->data) { resource->data = mount->data; }
      }
    }
    if (resource->provider && resource->provider->write) {
      return resource->provider->write(resource, buffer, nbytes);
    }
  }
  return -1;
}

VFS::ResourcePtr VFS::CreateResource(std::string_view name, const mjpResourceProvider* provider) {
  mjResource* res = new mjResource();
  res->vfs        = &wrapped_vfs_;
  res->provider   = provider;
  res->data       = nullptr;
  res->name       = new char[name.size() + 1];
  std::strncpy(res->name, name.data(), name.size());
  res->name[name.size()] = 0;
  res->timestamp[0]      = 0;

  return ResourcePtr(res, [](mjResource* ptr) {
    if (ptr->data) {
      // TODO: Make this an error eventually. For now, we continue to allow
      // users to free their data pointers without resetting them to nullptr.
      mju_warning("mjResource::data is not null; did you forget to close/unmount it?");
    }
    delete[] ptr->name;
    delete ptr;
  });
}

mjResource* VFS::FindMount(const std::string& fullpath) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::string str = fullpath;
  while (!str.empty()) {
    if (str.size() < fullpath.size() &&
        (fullpath[str.size()] == '/' || fullpath[str.size()] == '\\')) {
      if (auto it = archive_mounts_.find(str); it != archive_mounts_.end()) {
        return it->second.get();
      }
      const mjpResourceProvider* archive_prov = mjp_findArchiveProvider(str.c_str());
      if (archive_prov) {
        ResourcePtr res = CreateResource(str, archive_prov);
        if (archive_prov->mount(res.get())) {
          mjResource* res_ptr = res.get();
          archive_mounts_.emplace(str, std::move(res));
          return res_ptr;
        }
        return nullptr;
      }
    }

    auto it = mounts_.find(str);
    if (it != mounts_.end()) { return it->second.get(); }

    std::size_t n = str.find_last_of("/\\");
    if (n == std::string::npos) {
      str = "";
    } else {
      str = str.substr(0, n);
    }
  }

  const mjpResourceProvider* provider = mjp_findProvider(fullpath.c_str());
  if (provider) {
    if (auto it = mounts_.find(provider->prefix); it != mounts_.end()) { return it->second.get(); }

    ResourcePtr res     = CreateResource(provider->prefix, provider);
    mjResource* res_ptr = res.get();
    mounts_.emplace(provider->prefix, std::move(res));
    if (provider->mount) { provider->mount(res_ptr); }
    return res_ptr;
  }

  // Legacy use-case: match on just the case-insensitive filename.
  const std::string filename = StripPathAndLower(fullpath);
  for (auto& [path, res] : mounts_) {
    if (StripPathAndLower(path) == filename) { return res.get(); }
  }

  return &default_mount_;
}

void VFS::MaybeSelfDestruct() {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (in_flight_open_ > 0 || !open_resources_.empty()) { return; }
  }
  if (destructor_) {
    // Copy the destructor to a local variable so that we can destroy `this`
    // object within the destructor.
    auto fn = std::move(destructor_);
    fn();
  }
}

void VFS::SetToSelfDestruct(std::function<void()> destructor) {
  destructor_ = std::move(destructor);
}

VFS* VFS::Upcast(mjVFS* vfs) {
  return vfs ? static_cast<VFS*>(vfs->impl_) : nullptr;
}

const VFS* VFS::Upcast(const mjVFS* vfs) {
  return vfs ? static_cast<const VFS*>(vfs->impl_) : nullptr;
}

}  // namespace mujoco::user

void mj_defaultVFS(mjVFS* vfs) {
  if (vfs == nullptr) {
    mju_error("mjVFS is null.");
  } else {
    vfs->impl_ = new mujoco::user::VFS(vfs);
  }
}

void mj_deleteVFS(mjVFS* vfs) {
  if (vfs) {
    delete mujoco::user::VFS::Upcast(vfs);
    vfs->impl_ = nullptr;
  }
}

int mj_mountVFS(mjVFS* vfs, const char* filepath, const mjpResourceProvider* provider) {
  mujoco::user::VFS* impl = mujoco::user::VFS::Upcast(vfs);
  if (impl == nullptr) {
    mju_error("mjVFS is null.");
    return mujoco::user::VFS::kInvalidVfs;
  }

  if (filepath == nullptr) { return mujoco::user::VFS::kNotFound; }
  const mujoco::user::FilePath    path(filepath);
  const mujoco::user::VFS::Status status = impl->Mount(path, provider);
  return static_cast<int>(status);
}

int mj_unmountVFS(mjVFS* vfs, const char* filename) {
  mujoco::user::VFS* impl = mujoco::user::VFS::Upcast(vfs);
  if (impl == nullptr) {
    mju_error("mjVFS is null.");
    return mujoco::user::VFS::kInvalidVfs;
  }

  if (filename == nullptr) { return mujoco::user::VFS::kNotFound; }
  const mujoco::user::FilePath    path(filename);
  const mujoco::user::VFS::Status status = impl->Unmount(path);
  return static_cast<int>(status);
}

namespace {

// Custom provider for mj_addFileVFS and mj_addBufferVFS.
class BufferProvider : public mjpResourceProvider {
 public:
  template <typename... Args>
  static int Mount(mjVFS* vfs, Args&&... args) {
    mujoco::user::VFS* impl = mujoco::user::VFS::Upcast(vfs);
    if (impl == nullptr) {
      mju_error("mjVFS is null.");
      return -1;
    }

    BufferProvider* provider = new BufferProvider(std::forward<Args>(args)...);
    provider->mount = [](mjResource* res) { return static_cast<int>(mujoco::user::VFS::kSuccess); };
    provider->unmount = [](mjResource* res) {
      delete (BufferProvider*)res->provider;
      return static_cast<int>(mujoco::user::VFS::kSuccess);
    };
    provider->open = [](mjResource* res) {
      BufferProvider* self = (BufferProvider*)res->provider;
      mju_encodeBase64(res->timestamp, (std::uint8_t*)&self->hash_, sizeof(self->hash_));
      return 1;
    };
    provider->read = [](mjResource* res, const void** out) {
      BufferProvider* self = (BufferProvider*)res->provider;

      *out = reinterpret_cast<void*>(self->contents_.data());
      return static_cast<int>(self->contents_.size());
    };
    provider->close = [](mjResource* res) {
      // no-op
    };
    provider->modified = [](const mjResource* res, const char* timestamp) {
      const BufferProvider* self = (const BufferProvider*)res->provider;
      if (mju_isValidBase64(timestamp) != sizeof(std::uint64_t)) { return 1; }
      std::uint64_t test = 0;
      mju_decodeBase64((std::uint8_t*)&test, timestamp);
      if (self->hash_ != test) { return 1; }
      return 0;
    };

    const mujoco::user::VFS::Status status = impl->Mount(provider->path_, provider);
    if (status != mujoco::user::VFS::kSuccess) { delete provider; }
    return static_cast<int>(status);
  }

 private:
  BufferProvider(const char* dir, const char* filename) {
    mjp_defaultProvider(this);

    mujoco::user::FilePath file_path(dir ? dir : "", filename);
    path_     = file_path.StripPath().Lower();
    contents_ = mujoco::user::FileToMemory(file_path.c_str());

    static constexpr std::uint64_t prime = 0x100000001b3;

    hash_ = contents_.empty() ? 0 : 0xcbf29ce484222325;
    for (const std::uint8_t& byte : contents_) {
      hash_ ^= byte;
      hash_ *= prime;
    }
  }

  BufferProvider(const char* name, const void* src, size_t n) {
    mjp_defaultProvider(this);
    path_ = mujoco::user::FilePath(name);

    static constexpr std::uint64_t prime = 0x100000001b3;

    hash_ = n ? 0xcbf29ce484222325 : 0;

    const std::uint8_t* src_bytes = static_cast<const std::uint8_t*>(src);
    contents_.reserve(n);
    for (size_t i = 0; i < n; i++) {
      contents_.push_back(src_bytes[i]);
      hash_ ^= src_bytes[i];
      hash_ *= prime;
    }
  }

  mujoco::user::FilePath    path_;
  std::vector<std::uint8_t> contents_;
  std::uint64_t             hash_ = 0;
};

}  // namespace

int mj_addFileVFS(mjVFS* vfs, const char* directory, const char* filename) {
  // Opens the files and copies its contents into the BufferProvider, then
  // mounts the provider at the given path.
  return BufferProvider::Mount(vfs, directory, filename);
}

int mj_addBufferVFS(mjVFS* vfs, const char* name, const void* buffer, int nbuffer) {
  // Copies the buffer into the BufferProvider and mounts it at the given path.
  return BufferProvider::Mount(vfs, name, buffer, nbuffer);
}

int mj_deleteFileVFS(mjVFS* vfs, const char* filename) {
  if (filename == nullptr) { return mujoco::user::VFS::kNotFound; }

  if (mj_unmountVFS(vfs, filename) != 0) {
    mujoco::user::FilePath path(filename);
    return mj_unmountVFS(vfs, path.StripPath().Lower().c_str());
  }
  return mujoco::user::VFS::kSuccess;
}

int mj_containsBufferVFS(mjVFS* vfs, const char* name) {
  mujoco::user::VFS* impl = mujoco::user::VFS::Upcast(vfs);
  if (impl == nullptr) {
    mju_error("mjVFS is null.");
    return -1;
  }
  return impl->ContainsBuffer(name);
}

int mj_containsFileVFS(mjVFS* vfs, const char* directory, const char* filename) {
  mujoco::user::VFS* impl = mujoco::user::VFS::Upcast(vfs);
  if (impl == nullptr) {
    mju_error("mjVFS is null.");
    return -1;
  }
  return impl->ContainsFile(directory, filename);
}
