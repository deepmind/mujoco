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

#include "studio/ux/fonts.h"

#include <cstddef>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>

namespace mujoco::studio {

namespace {

// Copies `data` into ImGui-owned memory and adds it to the current context's
// atlas; the atlas frees the copy with the context.
ImFont* AddFontCopy(const std::vector<std::byte>& data, float size_pixels,
                    const ImFontConfig* config = nullptr, const ImWchar* glyph_ranges = nullptr) {
  void* copy = ImGui::MemAlloc(data.size());
  memcpy(copy, data.data(), data.size());
  return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(copy, static_cast<int>(data.size()),
                                                    size_pixels, config, glyph_ranges);
}

}  // namespace

std::vector<std::byte> LoadFontAsset(const std::string& assets_dir, std::string_view filename) {
  if (assets_dir.empty()) return {};
  const std::string file_path = (std::filesystem::path(assets_dir) / filename).string();

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return {};
  }
  const std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<std::byte> buffer(file_size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    return {};
  }
  return buffer;
}

std::vector<std::byte> LoadFontResource(std::string_view filename) {
  const std::string name = std::string("font:") + std::string(filename);
  mjResource* resource = mju_openResource("", name.c_str(), nullptr, nullptr, 0);
  if (resource == nullptr) {
    return {};
  }
  const void* data = nullptr;
  const int size = mju_readResource(resource, &data);
  std::vector<std::byte> buffer;
  if (size > 0) {
    const std::byte* bytes = static_cast<const std::byte*>(data);
    buffer.assign(bytes, bytes + size);
  }
  // Resource can be closed because the font atlas owns a copy of the data.
  mju_closeResource(resource);
  return buffer;
}

void AddStudioFonts(const FontLoaderFn& load) {
  const std::vector<std::byte> main_data = load(kMainFontFile);
  if (!main_data.empty()) {
    AddFontCopy(main_data, 16.0f);
  }

  // The icon font merges into the font added above; without a base font the
  // merge has nothing to attach to, so skip it.
  const std::vector<std::byte> icon_data = load(kIconFontFile);
  if (!icon_data.empty() && !main_data.empty()) {
    ImFontConfig icon_config;
    icon_config.MergeMode = true;
    static constexpr ImWchar kIconRanges[] = {0xf000, 0xf3ff, 0};
    AddFontCopy(icon_data, 13.0f, &icon_config, kIconRanges);
  }

  const std::vector<std::byte> mono_data = load(kMonoFontFile);
  if (!mono_data.empty()) {
    AddFontCopy(mono_data, 14.0f);
  }
}

}  // namespace mujoco::studio
