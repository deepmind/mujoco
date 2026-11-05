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

#include "studio/ux/imgui_widgets.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>

#if defined(__EMSCRIPTEN__)
  #include <emscripten/em_js.h>

// Writes text to the browser's clipboard, which only JavaScript can reach.
// clang-format off
EM_JS(void, WriteBrowserClipboard, (const char* text), {
  const str = UTF8ToString(text);
  // Selects the text in an off-screen textarea and copies the selection, giving
  // keyboard focus back to whatever had it, normally the canvas.
  const copyWithExecCommand = () => {
    if (typeof document === "undefined" || !document.body) {
      return false;
    }
    const focused = document.activeElement;
    const textarea = document.createElement("textarea");
    textarea.value = str;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }
    document.body.removeChild(textarea);
    if (focused && focused.focus) {
      focused.focus();
    }
    return copied;
  };
  // Prefer the asynchronous Clipboard API, which browsers only expose in secure
  // contexts (https or localhost) and with a recent user gesture.
  if (typeof navigator !== "undefined" && navigator.clipboard &&
      navigator.clipboard.writeText) {
    navigator.clipboard.writeText(str).catch((error) => {
      // Fall back to execCommand("copy") if writeText refuses, e.g. because the
      // document is not focused or the user gesture expired.
      if (!copyWithExecCommand()) {
        console.warn("Copy to clipboard failed:", error);
      }
    });
    return;
  }
  // Fall back to the deprecated execCommand("copy") when the Clipboard API is
  // missing, e.g. the WebViewer served over plain http (still needs a recent
  // user gesture).
  if (!copyWithExecCommand()) {
    console.warn("Copy to clipboard failed: the Clipboard API needs https or " +
                 "localhost, and execCommand('copy') was refused.");
  }
});
// clang-format on
#endif  // defined(__EMSCRIPTEN__)

namespace mujoco::studio {

void AppendIniSection(std::string& ini, const std::string& section, const KeyValues& key_values) {
  if (section.front() != '[' || section.back() != ']') {
    mju_error("Section must be enclosed in square brackets.");
  }
  ini += "\n" + std::string(section) + "\n";
  for (auto& [key, value] : key_values) {
    ini += key + "=" + value + "\n";
  }
}

KeyValues ReadIniSection(const std::string& contents, const std::string& section) {
  if (section.front() != '[' || section.back() != ']') {
    mju_error("Section must be enclosed in square brackets.");
  }
  bool in_section = false;

  KeyValues key_values;
  std::istringstream f(contents);
  std::string line;
  while (std::getline(f, line)) {
    if (line[0] == '[') {
      in_section = (line == section);
    } else if (in_section) {
      std::string::size_type pos = line.find('=');
      if (pos != std::string::npos) {
        key_values[line.substr(0, pos)] = line.substr(pos + 1);
      }
    }
  }
  return key_values;
}

ImGui_DataPtrTable::ImGui_DataPtrTable(float w1, float w2) {
  ImGui::BeginTable("##PropertiesTable", 2, ImGuiTableFlags_RowBg);
  const float width = GetStableAvailWidth();
  ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, width * w1);
  ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, width * w2);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.0f, 0.0f));
}
ImGui_DataPtrTable::~ImGui_DataPtrTable() {
  ImGui::PopStyleVar();
  ImGui::EndTable();
}

void ImGui_DataPtrTable::SetPrefix(const char* prefix) { prefix_ = strlen(prefix); }

void ImGui_DataPtrTable::DataPtr(const char* label, const uintptr_t* ptr, int index, int n) {
  for (int i = 0; i < n; ++i) {
    MakeLabel(label, i, n);
    ImGui::Text("(%s)", &ptr[index + i] ? "[ptr]" : "null");
  }
}

void ImGui_DataPtrTable::DataPtr(const char* label, const char* ptr, int index, int n) {
  MakeLabel(label);
  ScopedStyle style;
  style.Color(ImGuiCol_Text, ImColor(255, 0, 0, 255));
  ImGui::Text("%s", "(char not implemented, please report bug)");
}

void ImGui_DataPtrTable::DataPtr(const char* label, const mjtByte* ptr, int index, int n) {
  for (int i = 0; i < n; ++i) {
    MakeLabel(label, i, n);
    ImGui::Text("%s", ptr[index + i] ? "true" : "false");
  }
}

void ImGui_DataPtrTable::DataPtr(const char* label, const mjtBool* ptr, int index, int n) {
  for (int i = 0; i < n; ++i) {
    MakeLabel(label, i, n);
    ImGui::Text("%s", ptr[index + i] ? "true" : "false");
  }
}

void ImGui_DataPtrTable::DataPtr(const char* label, const mjtSize* ptr, int index, int n) {
  Numeric(label, ptr, index, n);
}

void ImGui_DataPtrTable::DataPtr(const char* label, const int* ptr, int index, int n) {
  Numeric(label, ptr, index, n);
}

void ImGui_DataPtrTable::DataPtr(const char* label, const float* ptr, int index, int n) {
  Numeric(label, ptr, index, n);
}

void ImGui_DataPtrTable::DataPtr(const char* label, const double* ptr, int index, int n) {
  Numeric(label, ptr, index, n);
}

template <typename T>
void ImGui_DataPtrTable::Numeric(const char* label, const T* ptr, int index, int n) {
  const T* addr = ptr + index * n;

  using U = std::conditional_t<std::is_floating_point_v<T>, float, int>;

  // special treatment for NaNs.
  if constexpr (std::is_same_v<U, float>) {
    if (*addr != *addr) {
      MakeLabel(label);
      ImGui::Text("nan");
      return;
    }
  }

  constexpr const char* fmt1 = std::is_floating_point_v<U> ? "%f" : "%d";
  constexpr const char* fmt2 = std::is_floating_point_v<U> ? "%f  %f" : "%d  %d";
  constexpr const char* fmt3 = std::is_floating_point_v<U> ? "%f  %f  %f" : "%d  %d  %d";
  constexpr const char* fmt4 = std::is_floating_point_v<U> ? "%f  %f  %f  %f" : "%d  %d  %d  %d";

  auto text1 = [&](int offset) { ImGui::Text(fmt1, (U)(addr[offset])); };
  auto text2 = [&](int offset) { ImGui::Text(fmt2, (U)(addr[offset + 0]), (U)(addr[offset + 1])); };
  auto text3 = [&](int offset) {
    ImGui::Text(fmt3, (U)(addr[offset + 0]), (U)(addr[offset + 1]), (U)(addr[offset + 2]));
  };
  auto text4 = [&](int offset) {
    ImGui::Text(fmt4, (U)(addr[offset + 0]), (U)(addr[offset + 1]), (U)(addr[offset + 2]),
                (U)(addr[offset + 3]));
  };

  if (n == 1) {
    MakeLabel(label);
    text1(0);
  } else if (n == 2) {
    MakeLabel(label);
    text2(0);
  } else if (n == 3) {
    MakeLabel(label);
    text3(0);
  } else if (n == 4) {
    MakeLabel(label);
    text4(0);
  } else if (n == 6) {
    MakeLabel(label);
    text3(0);
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    text3(3);
  } else if (n == 9) {
    MakeLabel(label);
    text3(0);
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    text3(3);
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    text3(6);
  } else {
    for (int i = 0; i < n; ++i) {
      MakeLabel(label, i, n);
      text1(i);
    }
  }
}

void ImGui_DataPtrTable::MakeLabel(const char* label, int index, int total) {
  if (total == 1) {
    ImGui::TableNextColumn();
    ImGui::Text("%s", &label[prefix_]);
    ImGui::TableNextColumn();
  } else {
    const std::string tmp = std::string(&label[prefix_]) + "[" + std::to_string(index) + "]";
    ImGui::TableNextColumn();
    ImGui::Text("%s", tmp.c_str());
    ImGui::TableNextColumn();
  }
}

void ImGui_SpecElementTable::operator()(const char* label, mjtByte& val, const mjtByte& ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(val, ref);
}

void ImGui_SpecElementTable::operator()(const char* label, bool& val, const bool& ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(val, ref);
}

void ImGui_SpecElementTable::operator()(const char* label, mjtSize& val, const mjtSize& ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(val, ref);
}

void ImGui_SpecElementTable::operator()(const char* label, int& val, const int& ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(val, ref);
}

void ImGui_SpecElementTable::operator()(const char* label, float& val, const float& ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(val, ref);
}

void ImGui_SpecElementTable::operator()(const char* label, double& val, const double& ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(val, ref);
}

void ImGui_SpecElementTable::operator()(const char* label, std::string* ptr, const std::string* ref,
                                        const char* tooltip) {
  Label(label, tooltip);
  Input(*ptr, *ref);
}

void ImGui_SpecElementTable::operator()(const char* label, std::vector<int>* ptr,
                                        const std::vector<int>* ref, const char* tooltip) {
  Label(label, tooltip);
  if (ptr == nullptr || ptr->empty()) {
    ImGui::Text("[empty]");
  } else {
    ImGui::Text("[%zu values]", ptr->size());
  }
}

void ImGui_SpecElementTable::operator()(const char* label, std::vector<double>* ptr,
                                        const std::vector<double>* ref, const char* tooltip) {
  Label(label, tooltip);
  if (ptr == nullptr || ptr->empty()) {
    ImGui::Text("[empty]");
  } else {
    ImGui::Text("[%zu values]", ptr->size());
  }
}

void ImGui_SpecElementTable::operator()(const char* label, std::vector<std::string>* ptr,
                                        const std::vector<std::string>* ref, const char* tooltip) {
  for (int i = 0; i < ptr->size(); ++i) {
    Label(label, tooltip, i, ptr->size());
    Input(ptr->at(i));
  }
}

void ImGui_SpecElementTable::operator()(const char* name, const char* alt_name, double (&quat)[4],
                                        const double (&ref_quat)[4], mjsOrientation& alt,
                                        const mjsOrientation& ref_alt, const char* tooltip) {
  auto aname = [&](const char* label) { return std::string(alt_name) + "." + label; };

  switch (alt.type) {
    case mjORIENTATION_QUAT:
      (*this)(name, quat, ref_quat, tooltip);
      break;
    case mjORIENTATION_AXISANGLE:
      (*this)(aname("axisangle").c_str(), alt.axisangle, ref_alt.axisangle, tooltip);
      break;
    case mjORIENTATION_XYAXES:
      (*this)(aname("xyaxes").c_str(), alt.xyaxes, ref_alt.xyaxes, tooltip);
      break;
    case mjORIENTATION_ZAXIS:
      (*this)(aname("zaxis").c_str(), alt.zaxis, ref_alt.zaxis, tooltip);
      break;
    case mjORIENTATION_EULER:
      (*this)(aname("euler").c_str(), alt.euler, ref_alt.euler, tooltip);
      break;
  }
}

void ImGui_SpecElementTable::Label(const char* label, const char* tooltip, int i, int n) {
  MakeLabel(label, i, n);
  ImGui::SetItemTooltip("%s", tooltip);
}

bool ImGui_Slider(const char* name, mjtNum* value, mjtNum min, mjtNum max) {
  float f = *value;
  const bool res = ImGui::SliderFloat(name, &f, min, max);
  if (res) {
    *value = mju_clip(f, min, max);
  }
  return res;
}

bool ImGui_SliderLog(const char* name, mjtNum* value, mjtNum min, mjtNum max) {
  constexpr ImGuiDataType type =
      sizeof(mjtNum) == sizeof(double) ? ImGuiDataType_Double : ImGuiDataType_Float;
  return ImGui::SliderScalar(name, type, value, &min, &max, "%.3g", ImGuiSliderFlags_Logarithmic);
}

bool ImGui_ResetButton(const char* id, const char* icon, const char* tooltip) {
  const float size = ImGui::GetFrameHeight();
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - size);
  ImGui::PushID(id);
  const bool clicked = ImGui::SmallButton(icon);
  ImGui::PopID();
  ImGui::SetItemTooltip("%s", tooltip);
  return clicked;
}

bool ImGui_BeginHSplit(const char* id, float* height, bool* open) {
  const ImVec2 region = ImGui::GetContentRegionAvail();
  if (*height < 0 && region.y > 0) {
    *height = region.y / 2;
  }
  // If the split is closed, use the full region height.
  const float h = *open ? std::max(0.0f, *height) : region.y;
  return ImGui::BeginChild(id, ImVec2(0, h), 0);
}

bool ImGui_HSplit(const char* id, float* height, bool* open) {
  // End the parent child.
  ImGui::EndChild();

  const ImVec2 region = ImGui::GetContentRegionAvail();
  if (*open) {
    // Just to make the splitter bar centered.
    ImGui::InvisibleButton("##empty", ImVec2(10.0f, 8.0f));

    // The splitter bar; drag to change the height.
    ImGui::SameLine();
    ImGui::Button(id, ImVec2(region.x - 30.0f, 8.0f));
    if (ImGui::IsItemActive()) {
      *height += ImGui::GetIO().MouseDelta.y;
      if (*height < 0) *height = 0;
    }

    // The collapse button.
    ImGui::SameLine();
    if (ImGui::Button("x", ImVec2(10.0f, 8.0f))) {
      *open = false;
    }
  }
  if (*open) {
    return ImGui::BeginChild(id, ImVec2(region.x, 0), 1);
  } else {
    return false;
  }
}

void ImGui_EndHSplit(bool open) {
  if (open) {
    ImGui::EndChild();
  }
}

void MaybeSaveToClipboard(const std::string& contents) {
  // ImGui's platform clipboard, which in-app paste reads. In the browser builds
  // this is only SDL's internal buffer, since SDL's Emscripten port has no
  // clipboard, so also write the browser's clipboard there.
  ImGui::SetClipboardText(contents.c_str());
#if defined(__EMSCRIPTEN__)
  WriteBrowserClipboard(contents.c_str());
#endif
}

float GetExpectedLabelWidth() {
  static float expected_label_width = 0;
  if (expected_label_width == 0) {
    int longest = 0;
    const char* longest_label = "";
    for (int i = 0; i < mjNVISFLAG; ++i) {
      int length = static_cast<int>(std::strlen(mjVISSTRING[i][0]));
      if (length > longest) {
        longest_label = mjVISSTRING[i][0];
        longest = length;
      }
    }
    expected_label_width = ImGui::CalcTextSize(longest_label).x + 16;
  }
  return expected_label_width;
}

ImPlotFlags ImPlot_SetupPlotFlags(ImVec2 plot_size) {
  ImPlotFlags flags = ImPlotFlags_None;
  if (plot_size.x > 0 && plot_size.y > 0) {
    const float min_dim = std::min(plot_size.x, plot_size.y);
    if (min_dim < 150) {
      flags |= ImPlotFlags_NoTitle;
    }
    if (min_dim < 140) {
      flags |= ImPlotFlags_NoLegend;
    }
  }
  return flags;
}

void ImPlot_SetupTimeAxis(ImVec2 plot_size, const char* label, ImPlotAxisFlags extra_flags) {
  ImPlotAxisFlags flags = extra_flags;
  if (plot_size.x > 0 && plot_size.x < 180) {
    flags |= ImPlotAxisFlags_NoTickLabels;
  }
  ImPlot::SetupAxis(ImAxis_X1, label, flags);
}

void ImPlot_SetupValueAxis(ImVec2 plot_size, const char* label, const char* format,
                           ImPlotAxisFlags extra_flags) {
  ImPlotAxisFlags flags = extra_flags;
  if (plot_size.y > 0 && plot_size.y < 90) {
    flags |= ImPlotAxisFlags_NoTickLabels;
  }
  ImPlot::SetupAxis(ImAxis_Y1, label, flags);
  if (format) {
    ImPlot::SetupAxisFormat(ImAxis_Y1, format);
  }
}

void ImPlot_SetupFixedAxis(ImVec2 plot_size, double y_min, double y_max, const char* label,
                           const char* format, const double* tick_values,
                           const char* const* tick_labels, int n_ticks) {
  ImPlotAxisFlags flags = ImPlotAxisFlags_None;
  if (plot_size.y > 0 && plot_size.y < 90) {
    flags |= ImPlotAxisFlags_NoTickLabels;
  }
  ImPlot::SetupAxis(ImAxis_Y1, label, flags);
  ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);
  if (format) {
    ImPlot::SetupAxisFormat(ImAxis_Y1, format);
  }
  if (tick_values && n_ticks > 0) {
    ImPlot::SetupAxisTicks(ImAxis_Y1, tick_values, n_ticks, tick_labels);
  }
}

ImPlotPairLayout ImPlot_ComputePairLayout() {
  ImVec2 avail = ImGui::GetContentRegionAvail();
  bool is_wide = avail.x > avail.y;

  const float item_spacing = ImGui::GetStyle().ItemSpacing.y;
  ImVec2 plot_size(is_wide ? (avail.x - item_spacing) * 0.5f : avail.x,
                   is_wide ? avail.y : (avail.y - item_spacing) * 0.5f);

  return {
      plot_size,
      is_wide ? ImPlotLayoutDirection::kHorizontal : ImPlotLayoutDirection::kVertical,
  };
}

static ImVec2 ClipSpaceToRectCoordinates(float x, float y, const ImVec4& rect) {
  const float pos_x = rect.x + rect.z * ((x + 1) * 0.5f);
  const float pos_y = rect.y + rect.w * (1.0f - ((y + 1) * 0.5f));
  return ImVec2(pos_x, pos_y);
}

void DrawTextAt(const char* text, float x, float y, float z) {
  const ImVec2& display_size = ImGui::GetIO().DisplaySize;
  DrawTextAtInRect(text, x, y, z, ImVec4(0.0f, 0.0f, display_size.x, display_size.y));
}

void DrawTextAtInRect(const char* text, float x, float y, float z, const ImVec4& rect) {
  if (x < -1 || y < -1 || x > 1 || y > 1 || z < -1 || z > 1) {
    return;
  }

  const ImVec2 center_pos = ClipSpaceToRectCoordinates(x, y, rect);
  const ImVec2 size = ImGui::CalcTextSize(text);

  const ImVec2 pos = ImVec2(center_pos.x - size.x / 2, center_pos.y);
  const ImVec2 shadow_pos = ImVec2(pos.x + 2, pos.y + 2);
  const int flags = ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoFocusOnAppearing |
                    ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDecoration |
                    ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav;

  ImGui::Begin("labels", nullptr, flags);
  ImGui::BeginChild("labels", ImGui::GetIO().DisplaySize, 0, flags);
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  draw_list->AddText(shadow_pos, IM_COL32_BLACK, text);
  draw_list->AddText(pos, IM_COL32_WHITE, text);
  ImGui::EndChild();
  ImGui::End();
}
namespace {

void SetNextWindowPosInside(OverlayPos pos, ImVec4 rect) {
  // compute anchor point and pivot from position enum
  ImVec2 anchor, pivot;
  float x = rect.x, y = rect.y, w = rect.z, h = rect.w;
  float scale = ImGui::GetWindowDpiScale();
  float margin = 8.0f * scale;
  switch (pos) {
    case OverlayPos::kTopLeft:
      anchor = {x + margin, y + margin};
      pivot = {0.0f, 0.0f};
      break;
    case OverlayPos::kTop:
      anchor = {x + w * 0.5f, y + margin};
      pivot = {0.5f, 0.0f};
      break;
    case OverlayPos::kTopRight:
      anchor = {x + w - margin, y + margin};
      pivot = {1.0f, 0.0f};
      break;
    case OverlayPos::kBottomLeft:
      anchor = {x + margin, y + h - margin};
      pivot = {0.0f, 1.0f};
      break;
    case OverlayPos::kBottom:
      anchor = {x + w * 0.5f, y + h - margin};
      pivot = {0.5f, 1.0f};
      break;
    case OverlayPos::kBottomRight:
      anchor = {x + w - margin, y + h - margin};
      pivot = {1.0f, 1.0f};
      break;
  }

  ImGui::SetNextWindowPos(anchor, ImGuiCond_Always, pivot);
}

}  // namespace

bool BeginOverlay(const char* id, OverlayPos pos, ImVec4 rect, float min_width, float alpha) {
  SetNextWindowPosInside(pos, rect);

  if (min_width > 0.0f) {
    ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, -1.0f), ImVec2(FLT_MAX, -1.0f));
  }

  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                      ImGuiWindowFlags_AlwaysAutoResize |
                                      ImGuiWindowFlags_NoSavedSettings;

  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
  return ImGui::Begin(id, nullptr, kFlags);
}

void EndOverlay() {
  ImGui::End();
  ImGui::PopStyleVar();
}

void TextOverlay(const char* id, OverlayPos pos, ImVec4 workspace_rect, const char* text,
                 ImVec4 color, float font_scale, float alpha, float min_width) {
  float scale = ImGui::GetWindowDpiScale();
  float padding = 30.0f * scale;
  float max_width = std::max(0.0f, workspace_rect.z - 20.0f);
  float text_width = ImGui::CalcTextSize(text).x * font_scale;
  float min_target = std::min(min_width * scale, max_width);
  float target = 0.0f;
  if (text_width + padding > max_width || min_width > 0.0f) {
    target = std::clamp(text_width + padding, min_target, max_width);
  }

  if (BeginOverlay(id, pos, workspace_rect, target, alpha)) {
    if (font_scale != 1.0f) {
      ImGui::SetWindowFontScale(font_scale);
    }
    bool has_color = color.x != 0 || color.y != 0 || color.z != 0 || color.w != 0;
    if (has_color) {
      ImGui::PushStyleColor(ImGuiCol_Text, color);
    }
    if (text_width + padding > max_width) {
      ImGui::PushTextWrapPos(max_width - padding);
      ImGui::TextUnformatted(text);
      ImGui::PopTextWrapPos();
    } else {
      ImGui::TextUnformatted(text);
    }
    if (has_color) {
      ImGui::PopStyleColor();
    }
    if (font_scale != 1.0f) {
      ImGui::SetWindowFontScale(1.0f);
    }
  }
  EndOverlay();
}

}  // namespace mujoco::studio
