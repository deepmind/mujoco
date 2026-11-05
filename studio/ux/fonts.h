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

#ifndef MUJOCO_STUDIO_UX_FONTS_H_
#define MUJOCO_STUDIO_UX_FONTS_H_

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mujoco::studio {

// The Studio UI font files (every viewer loads this same files).
inline constexpr char kMainFontFile[] = "AtkinsonHyperlegibleNext[wght].ttf";
inline constexpr char kIconFontFile[] = "fontawesome-webfont.ttf";
inline constexpr char kMonoFontFile[] = "AtkinsonHyperlegibleMono-Regular.ttf";

// Maps a Studio font filename to its TTF bytes; empty means unavailable.
using FontLoaderFn = std::function<std::vector<std::byte>(std::string_view)>;

// A FontLoaderFn that reads font `filename` from an `assets_dir` folder.
std::vector<std::byte> LoadFontAsset(const std::string& assets_dir, std::string_view filename);

// A FontLoaderFn that reads font `font:<filename>` using a resource provider.
std::vector<std::byte> LoadFontResource(std::string_view filename);

// Adds the Studio fonts to the current ImGui context's atlas, fetching font
// data via `load`. The ImGui font atlas will own a copy of the data so it can
// rebuild at new DPI scales without buffers used by `load` staying alive.
void AddStudioFonts(const FontLoaderFn& load);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_FONTS_H_
