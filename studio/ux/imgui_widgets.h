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

#ifndef MUJOCO_STUDIO_UX_IMGUI_WIDGETS_H_
#define MUJOCO_STUDIO_UX_IMGUI_WIDGETS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include "studio/ux/enum_utils.h"
#include "studio/ux/well_tempered.h"

namespace mujoco::studio {

// Helper to get available content region width that does not change/reflow
// when the vertical scrollbar appears or disappears.
// Walks up the parent window chain to find the scroll-owning ancestor (the
// actual panel window, not a child window), and reserves scrollbar width
// when that panel's scrollbar is not currently visible.
inline float GetStableAvailWidth() {
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  float avail_x = ImGui::GetContentRegionAvail().x;

  // Walk up past child windows to the actual panel that owns the scrollbar.
  ImGuiWindow* scroll_owner = window;
  while (scroll_owner && (scroll_owner->Flags & ImGuiWindowFlags_ChildWindow) &&
         scroll_owner->ParentWindow) {
    scroll_owner = scroll_owner->ParentWindow;
  }

  if (scroll_owner && !(scroll_owner->Flags & ImGuiWindowFlags_NoScrollbar) &&
      !scroll_owner->ScrollbarY) {
    avail_x -= ImGui::GetStyle().ScrollbarSize;
  }
  return avail_x;
}

// Draws one line of text horizontally centered in the current window.
inline void CenteredLine(const char* text, const ImVec4* color = nullptr) {
  ImGui::SetCursorPosX(ImMax(0.0f, (ImGui::GetWindowWidth() - ImGui::CalcTextSize(text).x) * 0.5f));
  if (color != nullptr) {
    ImGui::TextColored(*color, "%s", text);
  } else {
    ImGui::TextUnformatted(text);
  }
}

// Large centered banner text.
inline void CenteredBanner(const char* text, const ImVec4& color) {
  ImGui::SetWindowFontScale(1.6f);
  ImGui::SetCursorPosX(ImMax(0.0f, (ImGui::GetWindowWidth() - ImGui::CalcTextSize(text).x) * 0.5f));
  ImGui::TextColored(color, "%s", text);
  ImGui::SetWindowFontScale(1.0f);
}

// FontAwesome icon codes.
static constexpr const char ICON_FA_ADJUST[] = "\xEF\x81\x82";
static constexpr const char ICON_FA_ARROWS[] = "\xEF\x81\x87";
static constexpr const char ICON_FA_CAMERA[] = "\xEF\x80\xBD";
static constexpr const char ICON_FA_CARET_LEFT[] = "\xEF\x83\x99";
static constexpr const char ICON_FA_CARET_RIGHT[] = "\xEF\x83\x9A";
static constexpr const char ICON_FA_CHECK_SQUARE_O[] = "\xEF\x81\x9D";
static constexpr const char ICON_FA_CIRCLE_O[] = "\xEF\x84\x8C";
static constexpr const char ICON_FA_CIRCLE[] = "\xEF\x84\x91";
static constexpr const char ICON_FA_COMMENT[] = "\xEF\x83\xA5";
static constexpr const char ICON_FA_COPY[] = "\xEF\x83\x85";
static constexpr const char ICON_FA_CROSSHAIRS[] = "\xEF\x81\x9B";
static constexpr const char ICON_FA_DIAMOND[] = "\xEF\x88\x99";
static constexpr const char ICON_FA_DOWNLOAD[] = "\xEF\x80\x99";
static constexpr const char ICON_FA_EJECT[] = "\xEF\x81\x92";
static constexpr const char ICON_FA_EXCLAMATION_TRIANGLE[] = "\xEF\x81\xB1";
static constexpr const char ICON_FA_FAST_FORWARD[] = "\xEF\x81\x90";
static constexpr const char ICON_FA_FLASK[] = "\xEF\x83\x83";
static constexpr const char ICON_FA_MAGIC[] = "\xEF\x83\x90";
static constexpr const char ICON_FA_MOON[] = "\xEF\x86\x86";
static constexpr const char ICON_FA_PAUSE[] = "\xEF\x81\x8C";
static constexpr const char ICON_FA_PLAY[] = "\xEF\x81\x8B";
static constexpr const char ICON_FA_PLUS[] = "\xEF\x81\xA7";
static constexpr const char ICON_FA_REFRESH[] = "\xEF\x80\xA1";
static constexpr const char ICON_FA_REPEAT[] = "\xEF\x80\x9E";
static constexpr const char ICON_FA_SQUARE_O[] = "\xEF\x87\x9B";
static constexpr const char ICON_FA_SUN[] = "\xEF\x86\x85";
static constexpr const char ICON_FA_TACHOMETER[] = "\xEF\x83\xA4";
static constexpr const char ICON_FA_TRASH_CAN[] = "\xEF\x87\xB8";
static constexpr const char ICON_FA_UNDO[] = "\xEF\x83\xA2";
static constexpr const char ICON_FA_UPLOAD[] = "\xEF\x82\x93";

using KeyValues = std::unordered_map<std::string, std::string>;

// This is a workaround to fix compilation on gcc <= 12 and clang <= 16
template <typename T>
struct dependent_false : std::false_type {};

template <typename T>
bool ImGui_Checkbox(const char* name, T& value);

// Appends key/value pairs to an Ini file.
void AppendIniSection(std::string& ini, const std::string& section, const KeyValues& key_values);

// Reads key/value pairs from an Ini file section.
KeyValues ReadIniSection(const std::string& contents, const std::string& section);

template <typename T>
T ReadIniValue(const KeyValues& key_values, const std::string& key, T def) {
  auto iter = key_values.find(key);
  if (iter == key_values.end()) {
    return def;
  }
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(iter->second);
  } else if constexpr (std::is_same_v<T, float>) {
    return std::stof(iter->second);
  } else if constexpr (std::is_same_v<T, double>) {
    return std::stod(iter->second);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return iter->second;
  } else if constexpr (std::is_enum_v<T>) {
    return static_cast<T>(std::stoi(iter->second));
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type");
  }
}

enum class ScopedFont {
  kDefault = 0,
  kMono = 1,
};

// Helper class for setting ImGui style options; automatically resets the
// styles when going out of scope.
struct ScopedStyle {
  ScopedStyle() = default;
  ~ScopedStyle() { Reset(); }

  ScopedStyle(const ScopedStyle&) = delete;
  ScopedStyle& operator=(const ScopedStyle&) = delete;
  ScopedStyle(ScopedStyle&& other) { Swap(other); }
  ScopedStyle& operator=(ScopedStyle&& other) {
    Swap(other);
    return *this;
  }

  void Swap(ScopedStyle& other) {
    std::swap(num_colors, other.num_colors);
    std::swap(num_vars, other.num_vars);
    std::swap(num_fonts, other.num_fonts);
  }

  ScopedStyle& Font(int index) {
    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    if (atlas && index >= 0 && index < atlas->Fonts.Size) {
      ImGui::PushFont(atlas->Fonts[index]);
      ++num_fonts;
    }
    return *this;
  }

  ScopedStyle& Font(ScopedFont font) { return Font(static_cast<int>(font)); }

  ScopedStyle& Color(ImGuiCol col, ImColor color) {
    ImGui::PushStyleColor(col, (ImU32)color);
    ++num_colors;
    return *this;
  }

  ScopedStyle& Color(ImGuiCol col, ImGuiCol col2) {
    ImGui::PushStyleColor(col, CurrentColor(col2));
    ++num_colors;
    return *this;
  }

  ScopedStyle& Var(ImGuiStyleVar var, float value) {
    ImGui::PushStyleVar(var, value);
    ++num_vars;
    return *this;
  }

  ScopedStyle& Var(ImGuiStyleVar var, const ImVec2& value) {
    ImGui::PushStyleVar(var, value);
    ++num_vars;
    return *this;
  }

  ImVec4 CurrentColor(ImGuiCol col) { return ImGui::GetStyle().Colors[col]; }

  void Reset() {
    for (int i = 0; i < num_fonts; ++i) {
      ImGui::PopFont();
    }
    ImGui::PopStyleVar(num_vars);
    ImGui::PopStyleColor(num_colors);
    num_colors = 0;
    num_vars = 0;
    num_fonts = 0;
  }

  int num_colors = 0;
  int num_vars = 0;
  int num_fonts = 0;
};

// Helper for displaying rows of key/value pairs in an ImGui table.
class ImGui_DataPtrTable {
 public:
  // Starts the table (i.e. ImGui::BeginTable()) with two columns of the
  // specified widths.
  ImGui_DataPtrTable(float w1 = 0.25f, float w2 = 0.75f);

  // Ends the table (e.g. ImGui::EndTable().
  ~ImGui_DataPtrTable();

  ImGui_DataPtrTable(const ImGui_DataPtrTable& other) = delete;
  ImGui_DataPtrTable& operator=(const ImGui_DataPtrTable& other) = delete;

  // Displays a labelled value in the table. These functions are intended
  // specifically for displaying data from mjModel and mjData which store data
  // in contiguous arrays. Each value to be displayed is at a given
  // index into the array (based on the object's ID) and then has a
  // dimensionality of n.
  void DataPtr(const char* label, const char* ptr, int index, int n);
  void DataPtr(const char* label, const mjtByte* ptr, int index, int n);
  void DataPtr(const char* label, const mjtBool* ptr, int index, int n);
  void DataPtr(const char* label, const mjtSize* ptr, int index, int n);
  void DataPtr(const char* label, const int* ptr, int index, int n);
  void DataPtr(const char* label, const float* ptr, int index, int n);
  void DataPtr(const char* label, const double* ptr, int index, int n);
  void DataPtr(const char* label, const uintptr_t* ptr, int index, int n);

  // Sets the prefix that will be removed from all labels. Note: that we simply
  // remove the first N characters of the label without actually comparing
  // against this prefix. This works well with mjModel and mjData because
  // data belonging to a given object type has a common prefix (e.g. all joints
  // properties are prefixed with "jnt_").
  void SetPrefix(const char* prefix);

 protected:
  template <typename T>
  void Numeric(const char* label, const T* ptr, int index, int n);
  void MakeLabel(const char* label, int index = 0, int total = 1);

  int prefix_ = 0;
};

// Helper for displaying mjSpec elements in an ImGui table.
class ImGui_SpecElementTable : public ImGui_DataPtrTable {
 public:
  explicit ImGui_SpecElementTable(bool read_only = true) : read_only_(read_only) {}

  // Scalar values used by mjSpec elements.
  void operator()(const char* label, mjtByte& val, const mjtByte& ref, const char* tooltip);
  void operator()(const char* label, bool& val, const bool& ref, const char* tooltip);
  void operator()(const char* label, mjtSize& val, const mjtSize& ref, const char* tooltip);
  void operator()(const char* label, int& val, const int& ref, const char* tooltip);
  void operator()(const char* label, float& val, const float& ref, const char* tooltip);
  void operator()(const char* label, double& val, const double& ref, const char* tooltip);

  // C++ container values used by mjSpec elements.
  void operator()(const char* label, std::string* ptr, const std::string* ref, const char* tooltip);
  void operator()(const char* label, std::vector<int>* ptr, const std::vector<int>* ref,
                  const char* tooltip);
  void operator()(const char* label, std::vector<double>* ptr, const std::vector<double>* ref,
                  const char* tooltip);
  void operator()(const char* label, std::vector<std::string>* ptr,
                  const std::vector<std::string>* ref, const char* tooltip);

  // C-style array values used by mjSpec elements.
  template <std::size_t N>
  void operator()(const char* label, int (&val)[N], const int (&ref)[N], const char* tooltip) {
    for (int i = 0; i < N; ++i) {
      Label(label, tooltip, i, N);
      Input(val[i], ref[i]);
    }
  }
  template <std::size_t N>
  void operator()(const char* label, float (&val)[N], const float (&ref)[N], const char* tooltip) {
    for (int i = 0; i < N; ++i) {
      Label(label, tooltip, i, N);
      Input(val[i], ref[i]);
    }
  }
  template <std::size_t N>
  void operator()(const char* label, double (&val)[N], const double (&ref)[N],
                  const char* tooltip) {
    for (int i = 0; i < N; ++i) {
      Label(label, tooltip, i, N);
      Input(val[i], ref[i]);
    }
  }

  // Special handling for fixed-length character arrays.
  template <std::size_t N>
  void operator()(const char* label, char (&val)[N], const char (&ref)[N], const char* tooltip) {
    Label(label, tooltip);

    char buf[N + 1];
    strncpy(buf, val, N);
    buf[N] = 0;
    if (read_only_) {
      ImGui::Text("%s", buf);
    } else {
      ImGui::PushID(&val);
      if (ImGui::InputText("##", buf, N + 1)) {
        strncpy(val, buf, N);
        modified_ = true;
      }
      ImGui::PopID();
    }
  }

  // Special handling for treating enum values as integers.
  template <typename T, typename U = std::enable_if_t<std::is_enum_v<T>, T>>
  void operator()(const char* label, T& val, const T& ref, const char* tooltip) {
    Label(label, tooltip);
    Input(val, ref);
  }

  // Special handling for quaternion/orientation pairs.
  void operator()(const char* name, const char* alt_name, double (&quat)[4],
                  const double (&ref_quat)[4], mjsOrientation& alt, const mjsOrientation& ref_alt,
                  const char* tooltip);

  // Returns true if any value in the table was modified.
  bool WasModified() const { return modified_; }

 private:
  void Label(const char* label, const char* tooltip, int i = 0, int n = 1);

  template <typename T>
  void Input(T& val, const T& ref = T()) {
    ScopedStyle style;
    if (!read_only_ && val != ref) {
      constexpr ImVec4 kModifiedColor = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
      style.Color(ImGuiCol_Text, kModifiedColor);
    }

    ImGui::PushID(&val);
    ImGui::SetNextItemWidth(-1.0f);
    if constexpr (std::is_same_v<T, mjtByte> || std::is_same_v<T, mjtBool>) {
      if (read_only_) {
        ImGui::Text("%s", val ? "true" : "false");
      } else {
        const int flags = ImGuiComboFlags_NoArrowButton;
        if (ImGui::BeginCombo("##", val ? "true" : "false", flags)) {
          if (ImGui::Selectable("true", val != 0)) {
            val = 1;
            modified_ = true;
          }
          if (ImGui::Selectable("false", val == 0)) {
            val = 0;
            modified_ = true;
          }
          ImGui::EndCombo();
        }
      }
    } else if constexpr (std::is_enum_v<T>) {
      const std::string preview(enum_utils::enum_to_string(val));
      if (read_only_) {
        ImGui::Text("%s", preview.c_str());
      } else {
        const int flags = ImGuiComboFlags_NoArrowButton;
        if (ImGui::BeginCombo("##", preview.c_str(), flags)) {
          for (const auto& [v, n] : enum_utils::entries_v<T>) {
            if (ImGui::Selectable(std::string(n).c_str(), val == v)) {
              val = v;
              modified_ = true;
            }
          }
          ImGui::EndCombo();
        }
      }
    } else if constexpr (std::is_same_v<T, int>) {
      if (read_only_) {
        ImGui::Text("%d", val);
      } else {
        if (ImGui::InputInt("##", &val)) {
          modified_ = true;
        }
      }
    } else if constexpr (std::is_same_v<T, float>) {
      if (read_only_) {
        ImGui::Text("%f", val);
      } else {
        if (ImGui::InputFloat("##", &val)) {
          modified_ = true;
        }
      }
    } else if constexpr (std::is_same_v<T, double>) {
      if (read_only_) {
        ImGui::Text("%f", val);
      } else {
        if (ImGui::InputDouble("##", &val)) {
          modified_ = true;
        }
      }
    } else if constexpr (std::is_same_v<T, int64_t>) {
      if (read_only_) {
        ImGui::Text("%lld", val);
      } else {
        if (ImGui::InputScalarN("##", ImGuiDataType_S64, &val, 1)) {
          modified_ = true;
        }
      }
    } else if constexpr (std::is_same_v<T, std::string>) {
      if (read_only_) {
        ImGui::Text("%s", val.c_str());
      } else {
        if (ImGui::InputText("##", &val)) {
          modified_ = true;
        }
      }
    }

    if (!read_only_) {
      if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        val = ref;
        modified_ = true;
      }
    }

    ImGui::PopID();
  }

  bool modified_ = false;
  bool read_only_ = false;
};

// ImGui horizontal splitter. `height` is the height of the upper pane.
// `open` is whether the bottom pane is open.  Usage is:
//
//   if (ImGui_BeginHSplit("top", &height, &open)) {
//     // Contents of the upper pane.
//     if (ImGui_HSplit("bottom", &height, &open)) {
//       // Contents of the bottom pane.
//     }
//     ImGui_EndHSplit(open);
//   }
bool ImGui_BeginHSplit(const char* id, float* height, bool* open);
bool ImGui_HSplit(const char* id, float* height, bool* open);
void ImGui_EndHSplit(bool open);

// ImGui Slider that supports both float and double types.
bool ImGui_Slider(const char* name, mjtNum* value, mjtNum min, mjtNum max);

// Logarithmic slider; the geometric mean of min and max sits mid-slider.
// Ctrl+Click values typed by the user may exceed [min, max].
bool ImGui_SliderLog(const char* name, mjtNum* value, mjtNum min, mjtNum max);

// Small right-aligned reset button on the current line; returns true when
// clicked. Use after a widget to give it a visible reset affordance.
bool ImGui_ResetButton(const char* id, const char* icon = ICON_FA_UNDO,
                       const char* tooltip = "Reset");

template <typename T>
bool ImGui_Checkbox(const char* name, T& value) {
  static_assert(std::is_integral<T>());
  bool b = (value != 0);
  const bool res = ImGui::Checkbox(name, &b);
  if (res) {
    value = b ? 1 : 0;
  }
  return res;
}

template <typename T>
bool ImGui_ButtonToggle(const char* label, T* boolean, const ImVec2& size = ImVec2(0, 0)) {
  static_assert(std::is_integral_v<T>, "Toggle only supports integral types.");

  ScopedStyle style;
  const int color = *boolean ? ImGuiCol_TabSelected : ImGuiCol_WindowBg;
  style.Color(ImGuiCol_Button, ImGui::GetStyle().Colors[color]);
  style.Var(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

  const std::string txt = std::string(" ") +
                          std::string(*boolean ? ICON_FA_CHECK_SQUARE_O : ICON_FA_SQUARE_O) + "  " +
                          label;
  if (ImGui::Button(txt.c_str(), size)) {
    *boolean = !(*boolean);
    return true;
  }
  return false;
}

template <typename T>
bool ImGui_SwitchToggle(const char* label, T* boolean, const ImVec2& size = ImVec2(0, 0)) {
  static_assert(std::is_integral_v<T>, "Toggle only supports integral types.");

  int i = static_cast<int>(*boolean);
  const ImGuiSliderFlags flags = ImGuiSliderFlags_NoInput;
  if (size.x > 0) {
    ImGui::SetNextItemWidth(size.x);
  }
  const bool changed = ImGui::SliderInt(label, &i, 0, 1, label, flags);
  *boolean = (i != 0);
  return changed;
}

inline bool ImGui_BitToggle(const char* label, int* flags, int flags_value,
                            const ImVec2& size = ImVec2(0, 0)) {
  bool boolean = (*flags) & flags_value;
  const bool changed = ImGui_ButtonToggle(label, &boolean, size);
  if (changed) {
    *flags = boolean ? ((*flags) | flags_value) : ((*flags) & ~flags_value);
  }
  return changed;
}

// Options for ImGui_InputN (see below).
template <typename T>
struct ImGuiOpts {
  std::optional<T> min;
  std::optional<T> max;
  std::optional<T> step;
  std::optional<T> step_fast;
  std::optional<float> width;
  const char* format = std::is_floating_point_v<T> ? "%.3g" : "%d";
  // ImGui_LogStepper only: the ladder's bottom rung is 0 below this value.
  std::optional<T> zero_below;
};

// A compile-time wrapper around ImGui::InputScalarN. This is useful because
// MuJoCo uses an `mjtNum` type which is an alias for float or double.
//
// Options can be used to specify step sizes, clamp ranges, and formatting.
template <typename T>
bool ImGui_InputN(const char* name, T* value, int num, ImGuiOpts<T> opts = {}) {
  bool res = false;
  if (opts.width) {
    ImGui::SetNextItemWidth(opts.width.value());
  }
  if constexpr (std::is_same_v<T, int>) {
    const int step = opts.step.value_or(1);
    const int step_fast = opts.step_fast.value_or(100);
    const int* pstep = step > 0 ? &step : nullptr;
    const int* pstep_fast = step_fast > 0 ? &step_fast : nullptr;
    const char* format = opts.format;
    res = ImGui::InputScalarN(name, ImGuiDataType_S32, value, num, pstep, pstep_fast, format);

  } else if constexpr (std::is_same_v<T, float>) {
    const float step = opts.step.value_or(0.f);
    const float step_fast = opts.step_fast.value_or(0.f);
    const float* pstep = opts.step.has_value() ? &step : nullptr;
    const float* pstep_fast = opts.step_fast.has_value() ? &step_fast : nullptr;
    const char* format = opts.format ? opts.format : "%.3f";
    res = ImGui::InputScalarN(name, ImGuiDataType_Float, value, num, pstep, pstep_fast, format);

  } else if constexpr (std::is_same_v<T, double>) {
    const double step = opts.step.value_or(0.0);
    const double step_fast = opts.step_fast.value_or(0.0);
    const double* pstep = opts.step.has_value() ? &step : nullptr;
    const double* pstep_fast = opts.step_fast.has_value() ? &step_fast : nullptr;
    const char* format = opts.format ? opts.format : "%.3f";
    res = ImGui::InputScalarN(name, ImGuiDataType_Double, value, num, pstep, pstep_fast, format);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type");
  }

  if (opts.min.has_value()) {
    if (*value < *opts.min) *value = *opts.min;
  }
  if (opts.max.has_value()) {
    if (*value > *opts.max) *value = *opts.max;
  }
  return res;
}

template <typename T>
bool ImGui_Input(const char* name, T* value, ImGuiOpts<T> opts = {}) {
  return ImGui_InputN(name, value, 1, opts);
}

// A numeric input whose - / + buttons step through a "well-tempered"
// logarithmic ladder (see WellTemperedStep): round numbers spaced evenly in log
// space that close each decade on a power of ten. Typed edits are applied on
// enter or focus change, like ImGui_Input. Suited to quantities spanning
// several orders of magnitude, such as the simulation timestep. The buttons
// auto-repeat when held.
template <typename T>
bool ImGui_LogStepper(const char* name, T* value, ImGuiOpts<T> opts = {}) {
  const float button = ImGui::GetFrameHeight();
  const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
  float input_width = ImGui::CalcItemWidth() - 2 * (button + spacing);
  if (input_width < 1) input_width = 1;

  const double zero_below = opts.zero_below ? static_cast<double>(*opts.zero_below) : 0.0;
  auto clamp = [&](T v) {
    if (opts.min && v < *opts.min) v = *opts.min;
    if (opts.max && v > *opts.max) v = *opts.max;
    return v;
  };
  auto step_fn = [&](T v, int dir) -> T {
    if constexpr (std::is_integral_v<T>) {
      double cur = static_cast<double>(v);
      for (int i = 0; i < 4; ++i) {
        cur = WellTemperedStep(cur, dir, zero_below);
        T next = static_cast<T>(std::lround(cur));
        if (next != v || cur == 0.0) {
          return next;
        }
      }
      return v;
    } else {
      return static_cast<T>(WellTemperedStep(v, dir, zero_below));
    }
  };

  bool changed = false;
  ImGui::PushID(name);

  // Lay out like ImGui's own InputScalar stepper: the field, then the - / +
  // buttons to its right, then the label.
  ImGuiOpts<T> field = opts;
  field.step = std::is_integral_v<T> ? std::optional<T>(0) : std::nullopt;
  field.step_fast = std::is_integral_v<T> ? std::optional<T>(0) : std::nullopt;
  field.width = input_width;
  if (ImGui_InputN("##value", value, 1, field)) changed = true;

  ImGui::PushButtonRepeat(true);
  ImGui::SameLine(0, spacing);
  if (ImGui::Button("-", ImVec2(button, button))) {
    *value = clamp(step_fn(*value, -1));
    changed = true;
  }
  ImGui::SameLine(0, spacing);
  if (ImGui::Button("+", ImVec2(button, button))) {
    *value = clamp(step_fn(*value, +1));
    changed = true;
  }
  ImGui::PopButtonRepeat();

  ImGui::SameLine(0, spacing);
  ImGui::TextUnformatted(name);
  ImGui::PopID();
  return changed;
}

// Returns true if the given chord is has _just_ been pressed in this frame.
// (This is opposed to "Pressed" which means the chord is active, i.e. the user
// is holding down the keys.)
inline bool ImGui_IsChordJustPressed(ImGuiKeyChord chord) {
  return ImGui::IsKeyChordPressed(chord, 0);
}

// Stateful button that displays the given color when active, and shows a
// semi-transparent hover color (controlled by hover_alpha) when inactive.
inline bool ImGui_ColorButton(const char* label, bool active, ImColor color,
                              const ImVec2& size = ImVec2(0, 0), float hover_alpha = 0.5f) {
  ScopedStyle style;
  const ImColor hover(color.Value.x, color.Value.y, color.Value.z, color.Value.w * hover_alpha);
  if (active) {
    style.Color(ImGuiCol_Button, color);
    style.Color(ImGuiCol_ButtonHovered, color);
  } else {
    style.Color(ImGuiCol_ButtonHovered, hover);
  }
  style.Color(ImGuiCol_ButtonActive, color);
  return ImGui::Button(label, size);
}

// Like ImGui_ColorButton, but with per-corner rounding control via ImDrawFlags.
// Use ImDrawFlags_RoundCornersLeft, ImDrawFlags_RoundCornersRight,
// ImDrawFlags_RoundCornersNone, ImDrawFlags_RoundCornersAll, etc.
inline bool ImGui_ColorButtonEx(const char* label, bool active, ImColor color, ImDrawFlags corners,
                                const ImVec2& size = ImVec2(0, 0), float hover_alpha = 0.5f) {
  const ImGuiStyle& s = ImGui::GetStyle();
  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
  const ImVec2 btn_size(size.x > 0 ? size.x : label_size.x + s.FramePadding.x * 2,
                        size.y > 0 ? size.y : label_size.y + s.FramePadding.y * 2);

  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton(label, btn_size);
  const bool clicked = ImGui::IsItemClicked();
  const bool hovered = ImGui::IsItemHovered();

  // Determine background color.
  const ImColor hover_color(color.Value.x, color.Value.y, color.Value.z,
                            color.Value.w * hover_alpha);
  ImColor bg;
  if (active) {
    bg = color;
  } else if (hovered) {
    bg = hover_color;
  } else {
    bg = ImGui::GetColorU32(ImGuiCol_Button);
  }

  // Draw background with per-corner rounding.
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 max(pos.x + btn_size.x, pos.y + btn_size.y);
  dl->AddRectFilled(pos, max, bg, s.FrameRounding, corners);

  // Draw border.
  if (s.FrameBorderSize > 0) {
    dl->AddRect(pos, max, ImGui::GetColorU32(ImGuiCol_Border), s.FrameRounding, corners,
                s.FrameBorderSize);
  }

  // Draw label centered.
  const ImVec2 text_pos(pos.x + (btn_size.x - label_size.x) * 0.5f,
                        pos.y + (btn_size.y - label_size.y) * 0.5f);
  dl->AddText(text_pos, ImGui::GetColorU32(ImGuiCol_Text), label);

  return clicked;
}

// Begin a boxed section with outer borders - use EndBoxSection to close.
inline bool BeginBoxSection(const char* id, ImGuiTableFlags extra_flags = 0) {
  ImGuiTableFlags flags = ImGuiTableFlags_BordersOuter | extra_flags;
  if (ImGui::BeginTable(id, 1, flags)) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    return true;
  }
  return false;
}

inline void EndBoxSection() { ImGui::EndTable(); }

// Saves the given contents to the clipboard if the clipboard is available.
void MaybeSaveToClipboard(const std::string& contents);

// Returns the expected width of a UI label based on MuJoCo visualization flags.
float GetExpectedLabelWidth();

// Returns plot flags with title/legend conditionally hidden when the plot
// area is too small.  `plot_size` is the final rendered size of the plot.
ImPlotFlags ImPlot_SetupPlotFlags(ImVec2 plot_size);

// Sets up the X axis as a "time/frame" axis.
// Hides tick labels when the plot is narrow.
// Uses `label` as the axis label (empty string to hide) and auto-fit limits.
void ImPlot_SetupTimeAxis(ImVec2 plot_size, const char* label = "",
                          ImPlotAxisFlags extra_flags = ImPlotAxisFlags_AutoFit);

// Sets up a Y axis with auto-fit limits.
// Hides tick labels when the plot is short.
void ImPlot_SetupValueAxis(ImVec2 plot_size, const char* label = "", const char* format = nullptr,
                           ImPlotAxisFlags extra_flags = ImPlotAxisFlags_AutoFit);

// Sets up a Y axis with fixed limits and optional explicit ticks.
// Hides tick labels when the plot is short.
void ImPlot_SetupFixedAxis(ImVec2 plot_size, double y_min, double y_max, const char* label = "",
                           const char* format = nullptr, const double* tick_values = nullptr,
                           const char* const* tick_labels = nullptr, int n_ticks = 0);

enum class ImPlotLayoutDirection {
  kHorizontal,
  kVertical,
};

struct ImPlotPairLayout {
  ImVec2 plot_size;  // Size for each individual plot.
  ImPlotLayoutDirection direction;
};

// Computes a responsive layout for two plots that share the available
// content region.  When the region is wider than tall, the plots are placed
// side-by-side; otherwise they are stacked vertically.
ImPlotPairLayout ImPlot_ComputePairLayout();

// Draws text at the given screen coordinates in clip space (i.e. [-1,-1,-1] to
// [1,1,1]), mapped onto the whole display.
void DrawTextAt(const char* text, float x, float y, float z);

// Like DrawTextAt, but the clip-space coordinates map onto `rect` (x, y, width,
// height in logical pixels) instead of the whole display: for a scene rendered
// into part of the window.
void DrawTextAtInRect(const char* text, float x, float y, float z, const ImVec4& rect);

// Height of Studio's status bar (and of the web viewer's role bar, which
// stands in for it) in logical pixels, before DPI and font scaling.
constexpr float kStatusBarHeight = 32.0f;

// Grid position for overlays within a viewport rectangle.
enum class OverlayPos {
  kTopLeft,
  kTop,
  kTopRight,
  kBottomLeft,
  kBottom,
  kBottomRight,
};

// Begin an overlay window at the given grid position within a workspace rect.
// workspace_rect: (x, y, width, height) of the viewport area.
// min_width:      minimum window width in scaled pixels (0 = auto).
// alpha:          window transparency (0.0-1.0).
// Returns true if the overlay is visible (like ImGui::Begin).
// Must be paired with EndOverlay(), even if this returns false.
bool BeginOverlay(const char* id, OverlayPos pos, ImVec4 workspace_rect, float min_width = 0.0f,
                  float alpha = 0.8f);
void EndOverlay();

// One-liner: display a text overlay with automatic width sizing.
// Measures the longest line in `text`, clamps to workspace width, wraps text
// if needed.
// color: text color. If all components are 0, uses the theme default.
// font_scale: scale factor for the default font (1.0 = normal).
void TextOverlay(const char* id, OverlayPos pos, ImVec4 workspace_rect, const char* text,
                 ImVec4 color = ImVec4(0, 0, 0, 0), float font_scale = 1.0f, float alpha = 0.8f,
                 float min_width = 0.0f);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_IMGUI_WIDGETS_H_
