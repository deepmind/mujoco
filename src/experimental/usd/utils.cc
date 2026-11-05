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

// clang-format off
#include <cstddef>
#include <string>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <pxr/base/plug/registry.h>
// clang-format on

#include <mujoco/experimental/usd/utils.h>

#include <mujoco/mujoco.h>
#include <pxr/usd/sdf/path.h>

namespace mujoco {
namespace usd {

constexpr const char* kUsdPrimPathKey = "usd_primpath";

void SetUsdPrimPathUserValue(mjsElement* element,
                             const pxr::SdfPath& prim_path) {
  // The value is a pointer to a newly allocated SdfPath, which will be deleted
  // when the mjsElement is deleted.
  const pxr::SdfPath* usd_primpath = new pxr::SdfPath(prim_path);
  mjs_setUserValueWithCleanup(
      element, kUsdPrimPathKey, usd_primpath,
      [](const void* data) { delete static_cast<const pxr::SdfPath*>(data); });
}

pxr::SdfPath GetUsdPrimPathUserValue(mjsElement* element) {
  if (!element) {
    return pxr::SdfPath();
  }
  const void* user_data = mjs_getUserValue(element, kUsdPrimPathKey);
  if (user_data) {
    return *static_cast<const pxr::SdfPath*>(user_data);
  }
  return pxr::SdfPath();
}

}  // namespace usd
}  // namespace mujoco

namespace {
//Automatically register USD plugins when the library is loaded.
//This avoids the need for users to manually set PXR_PLUGINPATH_NAME.
[[maybe_unused]]
bool RegisterMujocoUsdPlugins() {
  std::string plugin_path;

#if defined(_WIN32) || defined(__CYGWIN__)
  HMODULE hModule = NULL;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                        (LPCTSTR)&mujoco::usd::SetUsdPrimPathUserValue,
                        &hModule)) {
    char path[MAX_PATH];
    if (GetModuleFileName(hModule, path, MAX_PATH)) {
      std::string lib_path(path);
      // Remove filename to get directory
      size_t last_slash = lib_path.find_last_of("\\/");
      if (last_slash != std::string::npos) {
        plugin_path = lib_path.substr(0, last_slash);
      }
    }
  }
#else
  Dl_info info;
  if (dladdr((void*)&mujoco::usd::SetUsdPrimPathUserValue, &info)) {
    std::string lib_path(info.dli_fname);
    // Remove filename to get directory
    size_t last_slash = lib_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      plugin_path = lib_path.substr(0, last_slash);
    }
  }
#endif

  if (!plugin_path.empty()) {
#if defined(_WIN32) || defined(__CYGWIN__)
    pxr::PlugRegistry::GetInstance().RegisterPlugins(
        plugin_path + "\\mujoco-usd-resources\\**\\plugInfo.json");
    pxr::PlugRegistry::GetInstance().RegisterPlugins(
        plugin_path + "\\..\\lib\\mujoco-usd-resources\\**\\plugInfo.json");
#else
    pxr::PlugRegistry::GetInstance().RegisterPlugins(
        plugin_path + "/mujoco-usd-resources/**/plugInfo.json");
    pxr::PlugRegistry::GetInstance().RegisterPlugins(
        plugin_path + "/../lib/mujoco-usd-resources/**/plugInfo.json");
#endif
  }
  return true;
}

// Force registration at library load time.
bool registered = RegisterMujocoUsdPlugins();

}  // namespace
