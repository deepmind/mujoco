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

#include "studio/ux/theme.h"

#include <imgui.h>
#include <imgui_internal.h>  // ImTrunc

namespace mujoco::studio {

void SetupTheme(GuiTheme theme) {
  ImGuiStyle& s = ImGui::GetStyle();
  ImVec4* c = s.Colors;
  if (theme == GuiTheme::kDark) {
    ImGui::StyleColorsDark(&s);

    // Cool slate backgrounds with subtle blue undertone.
    c[ImGuiCol_Text] = ImVec4(0.92, 0.93, 0.95, 1.00);
    c[ImGuiCol_TextDisabled] = ImVec4(0.45, 0.47, 0.50, 1.00);
    c[ImGuiCol_WindowBg] = ImVec4(0.16, 0.17, 0.19, 1.00);
    c[ImGuiCol_ChildBg] = ImVec4(0.16, 0.17, 0.19, 1.00);
    c[ImGuiCol_PopupBg] = ImVec4(0.24, 0.25, 0.28, 1.00);
    c[ImGuiCol_Border] = ImVec4(0.10, 0.10, 0.12, 0.40);
    c[ImGuiCol_BorderShadow] = ImVec4(0.00, 0.00, 0.00, 0.00);

    // Input fields: subtle inset, no transparency.
    c[ImGuiCol_FrameBg] = ImVec4(0.22, 0.23, 0.26, 1.00);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.26, 0.27, 0.30, 1.00);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.30, 0.31, 0.35, 1.00);

    // Title bars: darkest layer.
    c[ImGuiCol_TitleBg] = ImVec4(0.12, 0.13, 0.15, 1.00);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.14, 0.15, 0.17, 1.00);
    c[ImGuiCol_TitleBgCollapsed] = ImVec4(0.12, 0.13, 0.15, 0.90);
    c[ImGuiCol_MenuBarBg] = ImVec4(0.12, 0.13, 0.15, 1.00);

    // Scrollbars: transparent background, visible thumb only.
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.00, 0.00, 0.00, 0.00);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.34, 0.35, 0.38, 1.00);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.42, 0.43, 0.46, 1.00);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.55, 0.56, 0.60, 1.00);

    // Interactive elements.
    c[ImGuiCol_CheckMark] = ImVec4(0.35, 0.50, 0.72, 1.00);
    c[ImGuiCol_SliderGrab] = ImVec4(0.28, 0.40, 0.58, 1.00);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.35, 0.50, 0.72, 1.00);

    // Buttons: slightly lighter than frame background (0.22) to provide subtle
    // but visible contrast.
    c[ImGuiCol_Button] = ImVec4(0.28, 0.29, 0.33, 1.00);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.34, 0.36, 0.40, 1.00);
    c[ImGuiCol_ButtonActive] = ImVec4(0.32, 0.45, 0.66, 1.00);

    // Headers and tree nodes: muted warm-slate accent.
    c[ImGuiCol_Header] = ImVec4(0.22, 0.23, 0.26, 1.00);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.32, 0.36, 0.42, 0.60);
    c[ImGuiCol_HeaderActive] = ImVec4(0.36, 0.41, 0.48, 0.85);

    // Separators: crisp dividers.
    c[ImGuiCol_Separator] = ImVec4(0.08, 0.08, 0.10, 0.50);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.36, 0.41, 0.48, 0.50);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.36, 0.41, 0.48, 0.80);

    c[ImGuiCol_ResizeGrip] = ImVec4(0.36, 0.41, 0.48, 0.25);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0.36, 0.41, 0.48, 0.67);
    c[ImGuiCol_ResizeGripActive] = ImVec4(0.36, 0.41, 0.48, 0.95);

    c[ImGuiCol_PlotLines] = ImVec4(0.35, 0.50, 0.72, 1.00);
    c[ImGuiCol_PlotLinesHovered] = ImVec4(1.00, 0.43, 0.35, 1.00);
    c[ImGuiCol_PlotHistogram] = ImVec4(0.35, 0.50, 0.72, 1.00);
    c[ImGuiCol_PlotHistogramHovered] = ImVec4(0.36, 0.41, 0.48, 1.00);
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.32, 0.36, 0.42, 0.50);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00, 0.00, 0.00, 0.50);
    c[ImGuiCol_DragDropTarget] = ImVec4(0.35, 0.50, 0.72, 0.90);
    c[ImGuiCol_NavCursor] = ImVec4(0.36, 0.41, 0.48, 1.00);
    c[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00, 1.00, 1.00, 0.70);
    c[ImGuiCol_NavWindowingDimBg] = ImVec4(0.00, 0.00, 0.00, 0.30);

    // Tabs.
    c[ImGuiCol_DockingEmptyBg] = ImVec4(0.14, 0.15, 0.17, 1.00);
    c[ImGuiCol_Tab] = ImVec4(0.16, 0.17, 0.19, 1.00);
    c[ImGuiCol_TabHovered] = ImVec4(0.32, 0.36, 0.42, 0.60);
    c[ImGuiCol_TabSelected] = ImVec4(0.22, 0.23, 0.26, 1.00);
    c[ImGuiCol_TabDimmed] = ImVec4(0.14, 0.15, 0.17, 1.00);
    c[ImGuiCol_TabDimmedSelected] = ImVec4(0.20, 0.21, 0.24, 1.00);
    c[ImGuiCol_DockingPreview] = ImVec4(0.36, 0.41, 0.48, 0.40);
    c[ImGuiCol_TableRowBg] = ImVec4(0.16, 0.17, 0.19, 1.00);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(0.20, 0.21, 0.24, 1.00);
  } else if (theme == GuiTheme::kLight) {
    ImGui::StyleColorsLight(&s);

    // Clean, warm-white palette with blue accent.
    ImVec4 text = ImVec4(0.14, 0.15, 0.18, 1.00);
    ImVec4 text_dim = ImVec4(0.45, 0.47, 0.50, 1.00);
    ImVec4 bg = ImVec4(0.96, 0.96, 0.97, 1.00);
    ImVec4 surface = ImVec4(1.00, 1.00, 1.00, 1.00);
    ImVec4 border = ImVec4(0.82, 0.83, 0.85, 1.00);
    ImVec4 input = ImVec4(0.93, 0.93, 0.95, 1.00);
    ImVec4 accent = ImVec4(0.38, 0.60, 0.90, 1.00);
    ImVec4 accent_hover = ImVec4(0.38, 0.60, 0.90, 0.18);
    ImVec4 accent_dim = ImVec4(0.38, 0.60, 0.90, 0.10);
    ImVec4 header = ImVec4(0.90, 0.91, 0.92, 1.00);
    ImVec4 grab = ImVec4(0.70, 0.71, 0.73, 1.00);

    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = text_dim;
    c[ImGuiCol_WindowBg] = bg;
    c[ImGuiCol_ChildBg] = bg;
    c[ImGuiCol_PopupBg] = ImVec4(0.94, 0.94, 0.95, 1.00);
    c[ImGuiCol_Border] = border;
    c[ImGuiCol_BorderShadow] = ImVec4(0.00, 0.00, 0.00, 0.00);

    c[ImGuiCol_FrameBg] = input;
    c[ImGuiCol_FrameBgHovered] = accent_hover;
    c[ImGuiCol_FrameBgActive] = ImVec4(0.38, 0.60, 0.90, 0.30);

    c[ImGuiCol_TitleBg] = header;
    c[ImGuiCol_TitleBgActive] = ImVec4(0.88, 0.89, 0.90, 1.00);
    c[ImGuiCol_TitleBgCollapsed] = ImVec4(0.92, 0.93, 0.94, 0.90);
    c[ImGuiCol_MenuBarBg] = header;

    c[ImGuiCol_ScrollbarBg] = ImVec4(0.00, 0.00, 0.00, 0.00);
    c[ImGuiCol_ScrollbarGrab] = grab;
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.56, 0.57, 0.60, 1.00);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.42, 0.43, 0.46, 1.00);

    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.30, 0.52, 0.82, 1.00);

    // Buttons: slightly darker than frame background (0.93) to provide subtle
    // but visible contrast.
    c[ImGuiCol_Button] = ImVec4(0.86, 0.87, 0.89, 1.00);
    c[ImGuiCol_ButtonHovered] = accent_hover;
    c[ImGuiCol_ButtonActive] = accent;
    c[ImGuiCol_Header] = header;
    c[ImGuiCol_HeaderHovered] = accent_hover;
    c[ImGuiCol_HeaderActive] = ImVec4(0.38, 0.60, 0.90, 0.30);

    c[ImGuiCol_Separator] = border;
    c[ImGuiCol_SeparatorHovered] = accent;
    c[ImGuiCol_SeparatorActive] = accent;

    c[ImGuiCol_ResizeGrip] = accent_dim;
    c[ImGuiCol_ResizeGripHovered] = accent_hover;
    c[ImGuiCol_ResizeGripActive] = accent;

    c[ImGuiCol_Tab] = bg;
    c[ImGuiCol_TabHovered] = accent_hover;
    c[ImGuiCol_TabSelected] = surface;
    c[ImGuiCol_TabDimmed] = bg;
    c[ImGuiCol_TabDimmedSelected] = ImVec4(0.97, 0.97, 0.98, 1.00);
    c[ImGuiCol_DockingEmptyBg] = header;
    c[ImGuiCol_DockingPreview] = ImVec4(0.22, 0.47, 0.82, 0.30);

    c[ImGuiCol_TextSelectedBg] = accent_hover;
    c[ImGuiCol_DragDropTarget] = accent;
    c[ImGuiCol_NavCursor] = accent;
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00, 0.00, 0.00, 0.30);
    c[ImGuiCol_TableRowBg] = surface;
    c[ImGuiCol_TableRowBgAlt] = ImVec4(0.92, 0.92, 0.94, 1.00);
  } else {
    ImGui::StyleColorsDark(&s);
    ImVec4 black = ImVec4(0.00, 0.00, 0.00, 1.0);
    ImVec4 window = ImVec4(0.25, 0.25, 0.25, 1.0);
    ImVec4 font_active = ImVec4(1.00, 1.00, 1.00, 1.0);
    ImVec4 font_inactive = ImVec4(0.50, 0.50, 0.50, 1.0);
    ImVec4 thumb = ImVec4(0.12, 0.12, 0.12, 1.0);
    ImVec4 section = ImVec4(0.40, 0.15, 0.15, 1.0);
    ImVec4 button = ImVec4(0.60, 0.40, 0.40, 1.0);
    ImVec4 check = ImVec4(0.40, 0.40, 0.70, 1.0);
    ImVec4 frame = ImVec4(0.40, 0.30, 0.40, 1.0);
    ImVec4 slider = ImVec4(0.60, 0.40, 0.60, 1.0);

    c[ImGuiCol_WindowBg] = window;
    c[ImGuiCol_ChildBg] = black;
    c[ImGuiCol_PopupBg] = window;
    c[ImGuiCol_Text] = font_active;
    c[ImGuiCol_TextDisabled] = font_inactive;
    c[ImGuiCol_CheckMark] = font_active;
    c[ImGuiCol_Header] = section;
    c[ImGuiCol_HeaderHovered] = section;
    c[ImGuiCol_HeaderActive] = section;
    c[ImGuiCol_TitleBgActive] = window;
    c[ImGuiCol_ScrollbarBg] = window;
    c[ImGuiCol_ScrollbarGrab] = thumb;
    c[ImGuiCol_ScrollbarGrabHovered] = thumb;
    c[ImGuiCol_ScrollbarGrabActive] = thumb;
    c[ImGuiCol_FrameBg] = frame;
    c[ImGuiCol_FrameBgHovered] = frame;
    c[ImGuiCol_FrameBgActive] = frame;
    c[ImGuiCol_SliderGrab] = slider;
    c[ImGuiCol_SliderGrabActive] = slider;
    c[ImGuiCol_Button] = window;
    c[ImGuiCol_ButtonHovered] = button;
    c[ImGuiCol_ButtonActive] = button;
    c[ImGuiCol_Tab] = window;
    c[ImGuiCol_TabHovered] = check;
    c[ImGuiCol_TabSelected] = check;
    c[ImGuiCol_TabDimmed] = window;
    c[ImGuiCol_TabDimmedSelected] = check;
  }

  float scale = s.FontScaleDpi;

  int hspacing = 3;
  int vspacing = 2;
  float rounding = 3.0f;
  s.DisplaySafeAreaPadding = ImVec2(0, 0);
  s.WindowPadding = ImTrunc(ImVec2(6.0f * scale, 6.0f * scale));
  s.FramePadding = ImTrunc(ImVec2(hspacing * scale, 4.0f * scale));
  s.ItemSpacing = ImTrunc(ImVec2(hspacing * scale, vspacing * scale));
  s.ItemInnerSpacing = ImTrunc(ImVec2(hspacing * scale, vspacing * scale));
  s.WindowRounding = ImTrunc(rounding * scale);
  s.FrameRounding = ImTrunc(rounding * scale);
  s.TabRounding = ImTrunc(rounding * scale);
  s.ScrollbarRounding = ImTrunc(rounding * scale);
  s.ChildRounding = ImTrunc(rounding * scale);
  s.GrabRounding = ImTrunc(rounding * scale);
  s.PopupRounding = ImTrunc(rounding * scale);
  s.WindowBorderSize = 0.0f;
  s.FrameBorderSize = 1.0f;
  s.PopupBorderSize = 1.0f;
  s.IndentSpacing = ImTrunc(10.0f * scale);
  s.ScrollbarSize = ImTrunc(10.0f * scale);
  s.GrabMinSize = ImTrunc(5.0f * scale);
  s.WindowMenuButtonPosition = ImGuiDir_None;
  s.TabCloseButtonMinWidthSelected = 0.0f;
  s.DockingNodeHasCloseButton = false;
}

const char* ThemeName(GuiTheme theme) {
  switch (theme) {
    case GuiTheme::kLight:
      return "Light";
    case GuiTheme::kDark:
      return "Dark";
    case GuiTheme::kClassic:
      return "Classic";
  }
  return "Dark";
}

}  // namespace mujoco::studio
