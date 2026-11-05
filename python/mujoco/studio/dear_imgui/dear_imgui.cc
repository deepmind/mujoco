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

#include <cfloat>
#include <cstdint>
#include <tuple>
#define NAMESPACE ImGui
#include "dear_imgui_macros.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// NOLINTBEGIN(whitespace/line_length)

namespace py = pybind11;
using ImString = const char*;
static constexpr const ImVec2 ImVec2_Zero = ImVec2(0.0f, 0.0f);
static constexpr const ImVec2 ImVec2_One = ImVec2(1.0f, 1.0f);
static constexpr const ImVec2 ImVec2_Min_Zero = ImVec2(-FLT_MIN, 0.0f);
static constexpr const ImVec4 ImVec4_Zero = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
static constexpr const ImVec4 ImVec4_One = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);


PYBIND11_MODULE(dear_imgui, m) {
  // Types.

  py::class_<ImVec2>(m, "Vec2")
      .def(py::init<>())
      .def(py::init<float, float>(), py::arg("_x"), py::arg("_y"))
      .def_readwrite("x", &ImVec2::x)
      .def_readwrite("y", &ImVec2::y);

  py::class_<ImVec4>(m, "Vec4")
      .def(py::init<>())
      .def(py::init<float, float, float, float>(), py::arg("_x"), py::arg("_y"), py::arg("_z"), py::arg("_w"))
      .def_readwrite("x", &ImVec4::x)
      .def_readwrite("y", &ImVec4::y)
      .def_readwrite("z", &ImVec4::z)
      .def_readwrite("w", &ImVec4::w);

  py::class_<ImGuiStyle>(m, "Style")
      .def_readwrite("FontScaleMain", &ImGuiStyle::FontScaleMain)
      .def_readwrite("FramePadding", &ImGuiStyle::FramePadding)
      .def_readwrite("ItemSpacing", &ImGuiStyle::ItemSpacing)
      .def_readwrite("WindowPadding", &ImGuiStyle::WindowPadding);

  m.def("GetStyle", &ImGui::GetStyle, py::return_value_policy::reference);

  py::class_<ImGuiIO>(m, "IO")
      .def_readwrite("ConfigFlags", &ImGuiIO::ConfigFlags)
      .def_readwrite("BackendFlags", &ImGuiIO::BackendFlags)
      .def_readwrite("ConfigErrorRecoveryEnableAssert",
                     &ImGuiIO::ConfigErrorRecoveryEnableAssert)
      .def_readwrite("ConfigErrorRecoveryEnableDebugLog",
                     &ImGuiIO::ConfigErrorRecoveryEnableDebugLog)
      .def_readwrite("ConfigErrorRecoveryEnableTooltip",
                     &ImGuiIO::ConfigErrorRecoveryEnableTooltip)
      .def_readwrite("DisplaySize", &ImGuiIO::DisplaySize)
      .def_readonly("FontGlobalScale", &ImGuiIO::FontGlobalScale)
      .def_readonly("DeltaTime", &ImGuiIO::DeltaTime)
      .def_property_readonly(
          "Fonts", [](const ImGuiIO& io) -> ImFontAtlas* { return io.Fonts; },
          py::return_value_policy::reference)
      .def_readonly("Framerate", &ImGuiIO::Framerate)
      .def_readonly("WantCaptureMouse", &ImGuiIO::WantCaptureMouse)
      .def_readonly("WantCaptureKeyboard", &ImGuiIO::WantCaptureKeyboard)
      .def_readwrite("WantSaveIniSettings", &ImGuiIO::WantSaveIniSettings)
      .def_readonly("KeyShift", &ImGuiIO::KeyShift)
      .def_readonly("KeyCtrl", &ImGuiIO::KeyCtrl)
      .def_readonly("KeyAlt", &ImGuiIO::KeyAlt)
      .def_readonly("KeySuper", &ImGuiIO::KeySuper)
      .def_readonly("MousePos", &ImGuiIO::MousePos)
      .def_readonly("MouseWheel", &ImGuiIO::MouseWheel)
      .def_readonly("MouseDelta", &ImGuiIO::MouseDelta);

  m.def("GetIO", []() -> ImGuiIO& { return ImGui::GetIO(); }, py::return_value_policy::reference);

  m.def("GetCurrentContext", []() {
    return reinterpret_cast<uintptr_t>(ImGui::GetCurrentContext());
  });
  m.def("SetCurrentContext", [](uintptr_t ptr) {
    ImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(ptr));
  });

  // Context and frame lifecycle, for hosts that drive ImGui from Python (e.g.
  // headless tests). Rendering backends stay in C++; after Render() the draw
  // data is simply dropped unless a C++ backend consumes it.
  m.def(
      "CreateContext",
      []() { return reinterpret_cast<uintptr_t>(ImGui::CreateContext()); },
      "Creates a context and makes it current; returns its handle.");
  m.def(
      "DestroyContext",
      [](uintptr_t ptr) {
        ImGui::DestroyContext(reinterpret_cast<ImGuiContext*>(ptr));
      },
      py::arg("ptr") = 0, "Destroys the given context (0: the current one).");
  DEF0(NewFrame);
  DEF0(EndFrame);
  DEF0(Render);

  py::class_<ImGuiListClipper>(m, "ListClipper")
      .def(py::init<>())
      .def("Begin", &ImGuiListClipper::Begin, py::arg("items_count"), py::arg("items_height") = -1.0f)
      .def("End", &ImGuiListClipper::End)
      .def("Step", &ImGuiListClipper::Step)
      .def("IncludeItemsByIndex", &ImGuiListClipper::IncludeItemsByIndex, py::arg("item_begin"), py::arg("item_end"))
      .def("IncludeItemByIndex", &ImGuiListClipper::IncludeItemByIndex, py::arg("item_index"))
      .def("SeekCursorForItem", &ImGuiListClipper::SeekCursorForItem, py::arg("item_index"))
      .def_readonly("DisplayStart", &ImGuiListClipper::DisplayStart)
      .def_readonly("DisplayEnd", &ImGuiListClipper::DisplayEnd);

  py::class_<ImFontAtlas>(m, "FontAtlas")
      .def("__len__", [](const ImFontAtlas& atlas) { return atlas.Fonts.Size; })
      .def(
          "__getitem__",
          [](const ImFontAtlas& atlas, int i) -> ImFont* {
            if (i < 0) i += atlas.Fonts.Size;
            if (i < 0 || i >= atlas.Fonts.Size) throw py::index_error();
            return atlas.Fonts[i];
          },
          py::return_value_policy::reference);

  py::class_<ImFont>(m, "Font");

  // Low-level drawing for custom widgets, e.g.
  //   dl = GetWindowDrawList(); dl.AddCircleFilled(center, 8.0, GetColorU32(col))
  // Colors are packed ImU32 values (see GetColorU32, which applies the style
  // alpha). Angles are radians, clockwise from +x since y points down.
  py::class_<ImDrawList>(m, "DrawList")
      .def("AddLine", &ImDrawList::AddLine, py::arg("p1"), py::arg("p2"), py::arg("col"), py::arg("thickness") = 1.0f)
      .def("AddRect", [](ImDrawList& self, const ImVec2& p_min, const ImVec2& p_max, ImU32 col, float rounding, float thickness, ImDrawFlags flags) {
        self.AddRect(p_min, p_max, col, rounding, thickness, flags);
      }, py::arg("p_min"), py::arg("p_max"), py::arg("col"), py::arg("rounding") = 0.0f, py::arg("thickness") = 1.0f, py::arg("flags") = 0)
      .def("AddRectFilled", &ImDrawList::AddRectFilled, py::arg("p_min"), py::arg("p_max"), py::arg("col"), py::arg("rounding") = 0.0f, py::arg("flags") = 0)
      .def("AddTriangle", &ImDrawList::AddTriangle, py::arg("p1"), py::arg("p2"), py::arg("p3"), py::arg("col"), py::arg("thickness") = 1.0f)
      .def("AddTriangleFilled", &ImDrawList::AddTriangleFilled, py::arg("p1"), py::arg("p2"), py::arg("p3"), py::arg("col"))
      .def("AddCircle", &ImDrawList::AddCircle, py::arg("center"), py::arg("radius"), py::arg("col"), py::arg("num_segments") = 0, py::arg("thickness") = 1.0f)
      .def("AddCircleFilled", &ImDrawList::AddCircleFilled, py::arg("center"), py::arg("radius"), py::arg("col"), py::arg("num_segments") = 0)
      .def("AddText", [](ImDrawList& self, const ImVec2& pos, ImU32 col, const char* text) {
        self.AddText(pos, col, text);
      }, py::arg("pos"), py::arg("col"), py::arg("text"))
      .def("AddPolyline", [](ImDrawList& self, const std::vector<ImVec2>& points, ImU32 col, float thickness, ImDrawFlags flags) {
        self.AddPolyline(points.data(), static_cast<int>(points.size()), col, thickness, flags);
      }, py::arg("points"), py::arg("col"), py::arg("thickness") = 1.0f, py::arg("flags") = 0)
      .def("AddConvexPolyFilled", [](ImDrawList& self, const std::vector<ImVec2>& points, ImU32 col) {
        self.AddConvexPolyFilled(points.data(), static_cast<int>(points.size()), col);
      }, py::arg("points"), py::arg("col"))
      .def("PathClear", &ImDrawList::PathClear)
      .def("PathLineTo", &ImDrawList::PathLineTo, py::arg("pos"))
      .def("PathArcTo", &ImDrawList::PathArcTo, py::arg("center"), py::arg("radius"), py::arg("a_min"), py::arg("a_max"), py::arg("num_segments") = 0)
      .def("PathStroke", [](ImDrawList& self, ImU32 col, float thickness, ImDrawFlags flags) {
        self.PathStroke(col, thickness, flags);
      }, py::arg("col"), py::arg("thickness") = 1.0f, py::arg("flags") = 0)
      .def("PathFillConvex", &ImDrawList::PathFillConvex, py::arg("col"));

  m.def("GetWindowDrawList", &ImGui::GetWindowDrawList, py::return_value_policy::reference);
  m.def("GetBackgroundDrawList", []() { return ImGui::GetBackgroundDrawList(); }, py::return_value_policy::reference);
  m.def("GetForegroundDrawList", []() { return ImGui::GetForegroundDrawList(); }, py::return_value_policy::reference);

  // Enumerations.

  py::enum_<ImGuiWindowFlags_>(m, "WindowFlags")
      .value("None", ImGuiWindowFlags_None)
      .value("NoTitleBar", ImGuiWindowFlags_NoTitleBar)
      .value("NoResize", ImGuiWindowFlags_NoResize)
      .value("NoMove", ImGuiWindowFlags_NoMove)
      .value("NoScrollbar", ImGuiWindowFlags_NoScrollbar)
      .value("NoScrollWithMouse", ImGuiWindowFlags_NoScrollWithMouse)
      .value("NoCollapse", ImGuiWindowFlags_NoCollapse)
      .value("AlwaysAutoResize", ImGuiWindowFlags_AlwaysAutoResize)
      .value("NoBackground", ImGuiWindowFlags_NoBackground)
      .value("NoSavedSettings", ImGuiWindowFlags_NoSavedSettings)
      .value("NoMouseInputs", ImGuiWindowFlags_NoMouseInputs)
      .value("MenuBar", ImGuiWindowFlags_MenuBar)
      .value("HorizontalScrollbar", ImGuiWindowFlags_HorizontalScrollbar)
      .value("NoFocusOnAppearing", ImGuiWindowFlags_NoFocusOnAppearing)
      .value("NoBringToFrontOnFocus", ImGuiWindowFlags_NoBringToFrontOnFocus)
      .value("AlwaysVerticalScrollbar", ImGuiWindowFlags_AlwaysVerticalScrollbar)
      .value("AlwaysHorizontalScrollbar", ImGuiWindowFlags_AlwaysHorizontalScrollbar)
      .value("NoNavInputs", ImGuiWindowFlags_NoNavInputs)
      .value("NoNavFocus", ImGuiWindowFlags_NoNavFocus)
      .value("UnsavedDocument", ImGuiWindowFlags_UnsavedDocument)
      .value("NoDocking", ImGuiWindowFlags_NoDocking)
      .value("NoNav", ImGuiWindowFlags_NoNav)
      .value("NoDecoration", ImGuiWindowFlags_NoDecoration)
      .value("NoInputs", ImGuiWindowFlags_NoInputs)
      .value("ChildWindow", ImGuiWindowFlags_ChildWindow)
      .value("Tooltip", ImGuiWindowFlags_Tooltip)
      .value("Popup", ImGuiWindowFlags_Popup)
      .value("Modal", ImGuiWindowFlags_Modal)
      .value("ChildMenu", ImGuiWindowFlags_ChildMenu)
      .value("DockNodeHost", ImGuiWindowFlags_DockNodeHost);

  py::enum_<ImGuiChildFlags_>(m, "ChildFlags")
      .value("None", ImGuiChildFlags_None)
      .value("Borders", ImGuiChildFlags_Borders)
      .value("AlwaysUseWindowPadding", ImGuiChildFlags_AlwaysUseWindowPadding)
      .value("ResizeX", ImGuiChildFlags_ResizeX)
      .value("ResizeY", ImGuiChildFlags_ResizeY)
      .value("AutoResizeX", ImGuiChildFlags_AutoResizeX)
      .value("AutoResizeY", ImGuiChildFlags_AutoResizeY)
      .value("AlwaysAutoResize", ImGuiChildFlags_AlwaysAutoResize)
      .value("FrameStyle", ImGuiChildFlags_FrameStyle)
      .value("NavFlattened", ImGuiChildFlags_NavFlattened);

  py::enum_<ImGuiInputTextFlags_>(m, "InputTextFlags")
      .value("None", ImGuiInputTextFlags_None)
      .value("CharsDecimal", ImGuiInputTextFlags_CharsDecimal)
      .value("CharsHexadecimal", ImGuiInputTextFlags_CharsHexadecimal)
      .value("CharsScientific", ImGuiInputTextFlags_CharsScientific)
      .value("CharsUppercase", ImGuiInputTextFlags_CharsUppercase)
      .value("CharsNoBlank", ImGuiInputTextFlags_CharsNoBlank)
      .value("AllowTabInput", ImGuiInputTextFlags_AllowTabInput)
      .value("EnterReturnsTrue", ImGuiInputTextFlags_EnterReturnsTrue)
      .value("EscapeClearsAll", ImGuiInputTextFlags_EscapeClearsAll)
      .value("CtrlEnterForNewLine", ImGuiInputTextFlags_CtrlEnterForNewLine)
      .value("ReadOnly", ImGuiInputTextFlags_ReadOnly)
      .value("Password", ImGuiInputTextFlags_Password)
      .value("AlwaysOverwrite", ImGuiInputTextFlags_AlwaysOverwrite)
      .value("AutoSelectAll", ImGuiInputTextFlags_AutoSelectAll)
      .value("ParseEmptyRefVal", ImGuiInputTextFlags_ParseEmptyRefVal)
      .value("DisplayEmptyRefVal", ImGuiInputTextFlags_DisplayEmptyRefVal)
      .value("NoHorizontalScroll", ImGuiInputTextFlags_NoHorizontalScroll)
      .value("NoUndoRedo", ImGuiInputTextFlags_NoUndoRedo)
      .value("CallbackCompletion", ImGuiInputTextFlags_CallbackCompletion)
      .value("CallbackHistory", ImGuiInputTextFlags_CallbackHistory)
      .value("CallbackAlways", ImGuiInputTextFlags_CallbackAlways)
      .value("CallbackCharFilter", ImGuiInputTextFlags_CallbackCharFilter)
      .value("CallbackResize", ImGuiInputTextFlags_CallbackResize)
      .value("CallbackEdit", ImGuiInputTextFlags_CallbackEdit)
      .value("WordWrap", ImGuiInputTextFlags_WordWrap);

  py::enum_<ImGuiTreeNodeFlags_>(m, "TreeNodeFlags")
      .value("None", ImGuiTreeNodeFlags_None)
      .value("Selected", ImGuiTreeNodeFlags_Selected)
      .value("Framed", ImGuiTreeNodeFlags_Framed)
      .value("AllowOverlap", ImGuiTreeNodeFlags_AllowOverlap)
      .value("NoTreePushOnOpen", ImGuiTreeNodeFlags_NoTreePushOnOpen)
      .value("NoAutoOpenOnLog", ImGuiTreeNodeFlags_NoAutoOpenOnLog)
      .value("DefaultOpen", ImGuiTreeNodeFlags_DefaultOpen)
      .value("OpenOnDoubleClick", ImGuiTreeNodeFlags_OpenOnDoubleClick)
      .value("OpenOnArrow", ImGuiTreeNodeFlags_OpenOnArrow)
      .value("Leaf", ImGuiTreeNodeFlags_Leaf)
      .value("Bullet", ImGuiTreeNodeFlags_Bullet)
      .value("FramePadding", ImGuiTreeNodeFlags_FramePadding)
      .value("SpanAvailWidth", ImGuiTreeNodeFlags_SpanAvailWidth)
      .value("SpanFullWidth", ImGuiTreeNodeFlags_SpanFullWidth)
      .value("SpanLabelWidth", ImGuiTreeNodeFlags_SpanLabelWidth)
      .value("SpanAllColumns", ImGuiTreeNodeFlags_SpanAllColumns)
      .value("NavLeftJumpsBackHere", ImGuiTreeNodeFlags_NavLeftJumpsBackHere)
      .value("CollapsingHeader", ImGuiTreeNodeFlags_CollapsingHeader);

  py::enum_<ImGuiPopupFlags_>(m, "PopupFlags")
      .value("None", ImGuiPopupFlags_None)
      .value("MouseButtonLeft", ImGuiPopupFlags_MouseButtonLeft)
      .value("MouseButtonRight", ImGuiPopupFlags_MouseButtonRight)
      .value("MouseButtonMiddle", ImGuiPopupFlags_MouseButtonMiddle)
      .value("MouseButtonMask", ImGuiPopupFlags_MouseButtonMask_)
      .value("NoReopen", ImGuiPopupFlags_NoReopen)
      .value("NoOpenOverExistingPopup", ImGuiPopupFlags_NoOpenOverExistingPopup)
      .value("NoOpenOverItems", ImGuiPopupFlags_NoOpenOverItems)
      .value("AnyPopupId", ImGuiPopupFlags_AnyPopupId)
      .value("AnyPopupLevel", ImGuiPopupFlags_AnyPopupLevel)
      .value("AnyPopup", ImGuiPopupFlags_AnyPopup);

  py::enum_<ImGuiSelectableFlags_>(m, "SelectableFlags")
      .value("None", ImGuiSelectableFlags_None)
      .value("DontClosePopups", ImGuiSelectableFlags_DontClosePopups)
      .value("SpanAllColumns", ImGuiSelectableFlags_SpanAllColumns)
      .value("AllowDoubleClick", ImGuiSelectableFlags_AllowDoubleClick)
      .value("Disabled", ImGuiSelectableFlags_Disabled)
      .value("AllowOverlap", ImGuiSelectableFlags_AllowOverlap);

  py::enum_<ImGuiComboFlags_>(m, "ComboFlags")
      .value("None", ImGuiComboFlags_None)
      .value("PopupAlignLeft", ImGuiComboFlags_PopupAlignLeft)
      .value("HeightSmall", ImGuiComboFlags_HeightSmall)
      .value("HeightRegular", ImGuiComboFlags_HeightRegular)
      .value("HeightLarge", ImGuiComboFlags_HeightLarge)
      .value("HeightLargest", ImGuiComboFlags_HeightLargest)
      .value("NoArrowButton", ImGuiComboFlags_NoArrowButton)
      .value("NoPreview", ImGuiComboFlags_NoPreview)
      .value("WidthFitPreview", ImGuiComboFlags_WidthFitPreview)
      .value("HeightMask", ImGuiComboFlags_HeightMask_);

  py::enum_<ImGuiTabBarFlags_>(m, "TabBarFlags")
      .value("None", ImGuiTabBarFlags_None)
      .value("Reorderable", ImGuiTabBarFlags_Reorderable)
      .value("AutoSelectNewTabs", ImGuiTabBarFlags_AutoSelectNewTabs)
      .value("TabListPopupButton", ImGuiTabBarFlags_TabListPopupButton)
      .value("NoCloseWithMiddleMouseButton", ImGuiTabBarFlags_NoCloseWithMiddleMouseButton)
      .value("NoTabListScrollingButtons", ImGuiTabBarFlags_NoTabListScrollingButtons)
      .value("NoTooltip", ImGuiTabBarFlags_NoTooltip)
      .value("DrawSelectedOverline", ImGuiTabBarFlags_DrawSelectedOverline)
      .value("FittingPolicyResizeDown", ImGuiTabBarFlags_FittingPolicyResizeDown)
      .value("FittingPolicyScroll", ImGuiTabBarFlags_FittingPolicyScroll)
      .value("FittingPolicyMask", ImGuiTabBarFlags_FittingPolicyMask_)
      .value("FittingPolicyDefault", ImGuiTabBarFlags_FittingPolicyDefault_);

  py::enum_<ImGuiTabItemFlags_>(m, "TabItemFlags")
      .value("None", ImGuiTabItemFlags_None)
      .value("UnsavedDocument", ImGuiTabItemFlags_UnsavedDocument)
      .value("SetSelected", ImGuiTabItemFlags_SetSelected)
      .value("NoCloseWithMiddleMouseButton", ImGuiTabItemFlags_NoCloseWithMiddleMouseButton)
      .value("NoPushId", ImGuiTabItemFlags_NoPushId)
      .value("NoTooltip", ImGuiTabItemFlags_NoTooltip)
      .value("NoReorder", ImGuiTabItemFlags_NoReorder)
      .value("Leading", ImGuiTabItemFlags_Leading)
      .value("Trailing", ImGuiTabItemFlags_Trailing)
      .value("NoAssumedClosure", ImGuiTabItemFlags_NoAssumedClosure);

  py::enum_<ImGuiFocusedFlags_>(m, "FocusedFlags")
      .value("None", ImGuiFocusedFlags_None)
      .value("ChildWindows", ImGuiFocusedFlags_ChildWindows)
      .value("RootWindow", ImGuiFocusedFlags_RootWindow)
      .value("AnyWindow", ImGuiFocusedFlags_AnyWindow)
      .value("NoPopupHierarchy", ImGuiFocusedFlags_NoPopupHierarchy)
      .value("DockHierarchy", ImGuiFocusedFlags_DockHierarchy)
      .value("RootAndChildWindows", ImGuiFocusedFlags_RootAndChildWindows);

  py::enum_<ImGuiHoveredFlags_>(m, "HoveredFlags")
      .value("None", ImGuiHoveredFlags_None)
      .value("ChildWindows", ImGuiHoveredFlags_ChildWindows)
      .value("RootWindow", ImGuiHoveredFlags_RootWindow)
      .value("AnyWindow", ImGuiHoveredFlags_AnyWindow)
      .value("NoPopupHierarchy", ImGuiHoveredFlags_NoPopupHierarchy)
      .value("DockHierarchy", ImGuiHoveredFlags_DockHierarchy)
      .value("AllowWhenBlockedByPopup", ImGuiHoveredFlags_AllowWhenBlockedByPopup)
      .value("AllowWhenBlockedByActiveItem", ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)
      .value("AllowWhenOverlappedByItem", ImGuiHoveredFlags_AllowWhenOverlappedByItem)
      .value("AllowWhenOverlappedByWindow", ImGuiHoveredFlags_AllowWhenOverlappedByWindow)
      .value("AllowWhenDisabled", ImGuiHoveredFlags_AllowWhenDisabled)
      .value("NoNavOverride", ImGuiHoveredFlags_NoNavOverride)
      .value("AllowWhenOverlapped", ImGuiHoveredFlags_AllowWhenOverlapped)
      .value("RectOnly", ImGuiHoveredFlags_RectOnly)
      .value("RootAndChildWindows", ImGuiHoveredFlags_RootAndChildWindows)
      .value("ForTooltip", ImGuiHoveredFlags_ForTooltip)
      .value("Stationary", ImGuiHoveredFlags_Stationary)
      .value("DelayNone", ImGuiHoveredFlags_DelayNone)
      .value("DelayShort", ImGuiHoveredFlags_DelayShort)
      .value("DelayNormal", ImGuiHoveredFlags_DelayNormal)
      .value("NoSharedDelay", ImGuiHoveredFlags_NoSharedDelay);

  py::enum_<ImGuiDockNodeFlags_>(m, "DockNodeFlags")
      .value("None", ImGuiDockNodeFlags_None)
      .value("KeepAliveOnly", ImGuiDockNodeFlags_KeepAliveOnly)
      .value("NoDockingOverCentralNode", ImGuiDockNodeFlags_NoDockingOverCentralNode)
      .value("PassthruCentralNode", ImGuiDockNodeFlags_PassthruCentralNode)
      .value("NoDockingSplit", ImGuiDockNodeFlags_NoDockingSplit)
      .value("NoResize", ImGuiDockNodeFlags_NoResize)
      .value("AutoHideTabBar", ImGuiDockNodeFlags_AutoHideTabBar)
      .value("NoUndocking", ImGuiDockNodeFlags_NoUndocking);

  py::enum_<ImGuiDragDropFlags_>(m, "DragDropFlags")
      .value("None", ImGuiDragDropFlags_None)
      .value("SourceNoPreviewTooltip", ImGuiDragDropFlags_SourceNoPreviewTooltip)
      .value("SourceNoDisableHover", ImGuiDragDropFlags_SourceNoDisableHover)
      .value("SourceNoHoldToOpenOthers", ImGuiDragDropFlags_SourceNoHoldToOpenOthers)
      .value("SourceAllowNullID", ImGuiDragDropFlags_SourceAllowNullID)
      .value("SourceExtern", ImGuiDragDropFlags_SourceExtern)
      .value("PayloadAutoExpire", ImGuiDragDropFlags_PayloadAutoExpire)
      .value("PayloadNoCrossContext", ImGuiDragDropFlags_PayloadNoCrossContext)
      .value("PayloadNoCrossProcess", ImGuiDragDropFlags_PayloadNoCrossProcess)
      .value("AcceptBeforeDelivery", ImGuiDragDropFlags_AcceptBeforeDelivery)
      .value("AcceptNoDrawDefaultRect", ImGuiDragDropFlags_AcceptNoDrawDefaultRect)
      .value("AcceptNoPreviewTooltip", ImGuiDragDropFlags_AcceptNoPreviewTooltip)
      .value("AcceptPeekOnly", ImGuiDragDropFlags_AcceptPeekOnly);

  py::enum_<ImGuiDataType_>(m, "DataType")
      .value("S8", ImGuiDataType_S8)
      .value("U8", ImGuiDataType_U8)
      .value("S16", ImGuiDataType_S16)
      .value("U16", ImGuiDataType_U16)
      .value("S32", ImGuiDataType_S32)
      .value("U32", ImGuiDataType_U32)
      .value("S64", ImGuiDataType_S64)
      .value("U64", ImGuiDataType_U64)
      .value("Float", ImGuiDataType_Float)
      .value("Double", ImGuiDataType_Double);

  py::enum_<ImGuiDir>(m, "Dir")
      .value("None", ImGuiDir_None)
      .value("Left", ImGuiDir_Left)
      .value("Right", ImGuiDir_Right)
      .value("Up", ImGuiDir_Up)
      .value("Down", ImGuiDir_Down);

  py::enum_<ImGuiSortDirection>(m, "SortDirection")
      .value("None", ImGuiSortDirection_None)
      .value("Ascending", ImGuiSortDirection_Ascending)
      .value("Descending", ImGuiSortDirection_Descending);

  py::enum_<ImGuiInputFlags_>(m, "InputFlags")
      .value("None", ImGuiInputFlags_None)
      .value("Repeat", ImGuiInputFlags_Repeat)
      .value("RouteActive", ImGuiInputFlags_RouteActive)
      .value("RouteFocused", ImGuiInputFlags_RouteFocused)
      .value("RouteGlobal", ImGuiInputFlags_RouteGlobal)
      .value("RouteAlways", ImGuiInputFlags_RouteAlways)
      .value("RouteOverFocused", ImGuiInputFlags_RouteOverFocused)
      .value("RouteOverActive", ImGuiInputFlags_RouteOverActive)
      .value("RouteUnlessBgFocused", ImGuiInputFlags_RouteUnlessBgFocused)
      .value("RouteFromRootWindow", ImGuiInputFlags_RouteFromRootWindow)
      .value("Tooltip", ImGuiInputFlags_Tooltip);

  py::enum_<ImGuiConfigFlags_>(m, "ConfigFlags")
      .value("None", ImGuiConfigFlags_None)
      .value("NavEnableKeyboard", ImGuiConfigFlags_NavEnableKeyboard)
      .value("NavEnableGamepad", ImGuiConfigFlags_NavEnableGamepad)
      .value("NavEnableSetMousePos", ImGuiConfigFlags_NavEnableSetMousePos)
      .value("NavNoCaptureKeyboard", ImGuiConfigFlags_NavNoCaptureKeyboard)
      .value("NoMouse", ImGuiConfigFlags_NoMouse)
      .value("NoMouseCursorChange", ImGuiConfigFlags_NoMouseCursorChange)
      .value("NoKeyboard", ImGuiConfigFlags_NoKeyboard)
      .value("DockingEnable", ImGuiConfigFlags_DockingEnable)
      .value("ViewportsEnable", ImGuiConfigFlags_ViewportsEnable)
      .value("DpiEnableScaleViewports", ImGuiConfigFlags_DpiEnableScaleViewports)
      .value("DpiEnableScaleFonts", ImGuiConfigFlags_DpiEnableScaleFonts)
      .value("IsSRGB", ImGuiConfigFlags_IsSRGB)
      .value("IsTouchScreen", ImGuiConfigFlags_IsTouchScreen);

  py::enum_<ImGuiBackendFlags_>(m, "BackendFlags")
      .value("None", ImGuiBackendFlags_None)
      .value("HasGamepad", ImGuiBackendFlags_HasGamepad)
      .value("HasMouseCursors", ImGuiBackendFlags_HasMouseCursors)
      .value("HasSetMousePos", ImGuiBackendFlags_HasSetMousePos)
      .value("RendererHasVtxOffset", ImGuiBackendFlags_RendererHasVtxOffset)
      .value("RendererHasTextures", ImGuiBackendFlags_RendererHasTextures)
      .value("PlatformHasViewports", ImGuiBackendFlags_PlatformHasViewports)
      .value("HasMouseHoveredViewport",
             ImGuiBackendFlags_HasMouseHoveredViewport)
      .value("RendererHasViewports", ImGuiBackendFlags_RendererHasViewports);

  py::enum_<ImGuiCol_>(m, "Col")
      .value("Text", ImGuiCol_Text)
      .value("TextDisabled", ImGuiCol_TextDisabled)
      .value("WindowBg", ImGuiCol_WindowBg)
      .value("ChildBg", ImGuiCol_ChildBg)
      .value("PopupBg", ImGuiCol_PopupBg)
      .value("Border", ImGuiCol_Border)
      .value("BorderShadow", ImGuiCol_BorderShadow)
      .value("FrameBg", ImGuiCol_FrameBg)
      .value("FrameBgHovered", ImGuiCol_FrameBgHovered)
      .value("FrameBgActive", ImGuiCol_FrameBgActive)
      .value("TitleBg", ImGuiCol_TitleBg)
      .value("TitleBgActive", ImGuiCol_TitleBgActive)
      .value("TitleBgCollapsed", ImGuiCol_TitleBgCollapsed)
      .value("MenuBarBg", ImGuiCol_MenuBarBg)
      .value("ScrollbarBg", ImGuiCol_ScrollbarBg)
      .value("ScrollbarGrab", ImGuiCol_ScrollbarGrab)
      .value("ScrollbarGrabHovered", ImGuiCol_ScrollbarGrabHovered)
      .value("ScrollbarGrabActive", ImGuiCol_ScrollbarGrabActive)
      .value("CheckMark", ImGuiCol_CheckMark)
      .value("SliderGrab", ImGuiCol_SliderGrab)
      .value("SliderGrabActive", ImGuiCol_SliderGrabActive)
      .value("Button", ImGuiCol_Button)
      .value("ButtonHovered", ImGuiCol_ButtonHovered)
      .value("ButtonActive", ImGuiCol_ButtonActive)
      .value("Header", ImGuiCol_Header)
      .value("HeaderHovered", ImGuiCol_HeaderHovered)
      .value("HeaderActive", ImGuiCol_HeaderActive)
      .value("Separator", ImGuiCol_Separator)
      .value("SeparatorHovered", ImGuiCol_SeparatorHovered)
      .value("SeparatorActive", ImGuiCol_SeparatorActive)
      .value("ResizeGrip", ImGuiCol_ResizeGrip)
      .value("ResizeGripHovered", ImGuiCol_ResizeGripHovered)
      .value("ResizeGripActive", ImGuiCol_ResizeGripActive)
      .value("TabHovered", ImGuiCol_TabHovered)
      .value("Tab", ImGuiCol_Tab)
      .value("TabSelected", ImGuiCol_TabSelected)
      .value("TabSelectedOverline", ImGuiCol_TabSelectedOverline)
      .value("TabDimmed", ImGuiCol_TabDimmed)
      .value("TabDimmedSelected", ImGuiCol_TabDimmedSelected)
      .value("TabDimmedSelectedOverline", ImGuiCol_TabDimmedSelectedOverline)
      .value("DockingPreview", ImGuiCol_DockingPreview)
      .value("DockingEmptyBg", ImGuiCol_DockingEmptyBg)
      .value("PlotLines", ImGuiCol_PlotLines)
      .value("PlotLinesHovered", ImGuiCol_PlotLinesHovered)
      .value("PlotHistogram", ImGuiCol_PlotHistogram)
      .value("PlotHistogramHovered", ImGuiCol_PlotHistogramHovered)
      .value("TableHeaderBg", ImGuiCol_TableHeaderBg)
      .value("TableBorderStrong", ImGuiCol_TableBorderStrong)
      .value("TableBorderLight", ImGuiCol_TableBorderLight)
      .value("TableRowBg", ImGuiCol_TableRowBg)
      .value("TableRowBgAlt", ImGuiCol_TableRowBgAlt)
      .value("TextLink", ImGuiCol_TextLink)
      .value("TextSelectedBg", ImGuiCol_TextSelectedBg)
      .value("DragDropTarget", ImGuiCol_DragDropTarget)
      .value("NavHighlight", ImGuiCol_NavHighlight)
      .value("NavWindowingHighlight", ImGuiCol_NavWindowingHighlight)
      .value("NavWindowingDimBg", ImGuiCol_NavWindowingDimBg)
      .value("ModalWindowDimBg", ImGuiCol_ModalWindowDimBg);

  py::enum_<ImGuiStyleVar_>(m, "StyleVar")
      .value("Alpha", ImGuiStyleVar_Alpha)
      .value("DisabledAlpha", ImGuiStyleVar_DisabledAlpha)
      .value("WindowPadding", ImGuiStyleVar_WindowPadding)
      .value("WindowRounding", ImGuiStyleVar_WindowRounding)
      .value("WindowBorderSize", ImGuiStyleVar_WindowBorderSize)
      .value("WindowMinSize", ImGuiStyleVar_WindowMinSize)
      .value("WindowTitleAlign", ImGuiStyleVar_WindowTitleAlign)
      .value("ChildRounding", ImGuiStyleVar_ChildRounding)
      .value("ChildBorderSize", ImGuiStyleVar_ChildBorderSize)
      .value("PopupRounding", ImGuiStyleVar_PopupRounding)
      .value("PopupBorderSize", ImGuiStyleVar_PopupBorderSize)
      .value("FramePadding", ImGuiStyleVar_FramePadding)
      .value("FrameRounding", ImGuiStyleVar_FrameRounding)
      .value("FrameBorderSize", ImGuiStyleVar_FrameBorderSize)
      .value("ItemSpacing", ImGuiStyleVar_ItemSpacing)
      .value("ItemInnerSpacing", ImGuiStyleVar_ItemInnerSpacing)
      .value("IndentSpacing", ImGuiStyleVar_IndentSpacing)
      .value("CellPadding", ImGuiStyleVar_CellPadding)
      .value("ScrollbarSize", ImGuiStyleVar_ScrollbarSize)
      .value("ScrollbarRounding", ImGuiStyleVar_ScrollbarRounding)
      .value("GrabMinSize", ImGuiStyleVar_GrabMinSize)
      .value("GrabRounding", ImGuiStyleVar_GrabRounding)
      .value("TabRounding", ImGuiStyleVar_TabRounding)
      .value("TabBorderSize", ImGuiStyleVar_TabBorderSize)
      .value("TabBarBorderSize", ImGuiStyleVar_TabBarBorderSize)
      .value("TableAngledHeadersAngle", ImGuiStyleVar_TableAngledHeadersAngle)
      .value("TableAngledHeadersTextAlign", ImGuiStyleVar_TableAngledHeadersTextAlign)
      .value("ButtonTextAlign", ImGuiStyleVar_ButtonTextAlign)
      .value("SelectableTextAlign", ImGuiStyleVar_SelectableTextAlign)
      .value("SeparatorTextBorderSize", ImGuiStyleVar_SeparatorTextBorderSize)
      .value("SeparatorTextAlign", ImGuiStyleVar_SeparatorTextAlign)
      .value("SeparatorTextPadding", ImGuiStyleVar_SeparatorTextPadding)
      .value("DockingSeparatorSize", ImGuiStyleVar_DockingSeparatorSize);

  py::enum_<ImGuiButtonFlags_>(m, "ButtonFlags")
      .value("None", ImGuiButtonFlags_None)
      .value("MouseButtonLeft", ImGuiButtonFlags_MouseButtonLeft)
      .value("MouseButtonRight", ImGuiButtonFlags_MouseButtonRight)
      .value("MouseButtonMiddle", ImGuiButtonFlags_MouseButtonMiddle)
      .value("MouseButtonMask", ImGuiButtonFlags_MouseButtonMask_);

  py::enum_<ImDrawFlags_>(m, "DrawFlags")
      .value("None", ImDrawFlags_None)
      .value("Closed", ImDrawFlags_Closed)
      .value("RoundCornersTopLeft", ImDrawFlags_RoundCornersTopLeft)
      .value("RoundCornersTopRight", ImDrawFlags_RoundCornersTopRight)
      .value("RoundCornersBottomLeft", ImDrawFlags_RoundCornersBottomLeft)
      .value("RoundCornersBottomRight", ImDrawFlags_RoundCornersBottomRight)
      .value("RoundCornersNone", ImDrawFlags_RoundCornersNone)
      .value("RoundCornersTop", ImDrawFlags_RoundCornersTop)
      .value("RoundCornersBottom", ImDrawFlags_RoundCornersBottom)
      .value("RoundCornersLeft", ImDrawFlags_RoundCornersLeft)
      .value("RoundCornersRight", ImDrawFlags_RoundCornersRight)
      .value("RoundCornersAll", ImDrawFlags_RoundCornersAll);

  py::enum_<ImGuiColorEditFlags_>(m, "ColorEditFlags")
      .value("None", ImGuiColorEditFlags_None)
      .value("NoAlpha", ImGuiColorEditFlags_NoAlpha)
      .value("NoPicker", ImGuiColorEditFlags_NoPicker)
      .value("NoOptions", ImGuiColorEditFlags_NoOptions)
      .value("NoSmallPreview", ImGuiColorEditFlags_NoSmallPreview)
      .value("NoInputs", ImGuiColorEditFlags_NoInputs)
      .value("NoTooltip", ImGuiColorEditFlags_NoTooltip)
      .value("NoLabel", ImGuiColorEditFlags_NoLabel)
      .value("NoSidePreview", ImGuiColorEditFlags_NoSidePreview)
      .value("NoDragDrop", ImGuiColorEditFlags_NoDragDrop)
      .value("NoBorder", ImGuiColorEditFlags_NoBorder)
      .value("AlphaBar", ImGuiColorEditFlags_AlphaBar)
      .value("AlphaPreview", ImGuiColorEditFlags_AlphaPreview)
      .value("AlphaPreviewHalf", ImGuiColorEditFlags_AlphaPreviewHalf)
      .value("HDR", ImGuiColorEditFlags_HDR)
      .value("DisplayRGB", ImGuiColorEditFlags_DisplayRGB)
      .value("DisplayHSV", ImGuiColorEditFlags_DisplayHSV)
      .value("DisplayHex", ImGuiColorEditFlags_DisplayHex)
      .value("Uint8", ImGuiColorEditFlags_Uint8)
      .value("Float", ImGuiColorEditFlags_Float)
      .value("PickerHueBar", ImGuiColorEditFlags_PickerHueBar)
      .value("PickerHueWheel", ImGuiColorEditFlags_PickerHueWheel)
      .value("InputRGB", ImGuiColorEditFlags_InputRGB)
      .value("InputHSV", ImGuiColorEditFlags_InputHSV)
      .value("DefaultOptions", ImGuiColorEditFlags_DefaultOptions_)
      .value("DisplayMask", ImGuiColorEditFlags_DisplayMask_)
      .value("DataTypeMask", ImGuiColorEditFlags_DataTypeMask_)
      .value("PickerMask", ImGuiColorEditFlags_PickerMask_)
      .value("InputMask", ImGuiColorEditFlags_InputMask_);

  py::enum_<ImGuiSliderFlags_>(m, "SliderFlags")
      .value("None", ImGuiSliderFlags_None)
      .value("AlwaysClamp", ImGuiSliderFlags_AlwaysClamp)
      .value("Logarithmic", ImGuiSliderFlags_Logarithmic)
      .value("NoRoundToFormat", ImGuiSliderFlags_NoRoundToFormat)
      .value("NoInput", ImGuiSliderFlags_NoInput)
      .value("WrapAround", ImGuiSliderFlags_WrapAround)
      .value("InvalidMask", ImGuiSliderFlags_InvalidMask_);

  py::enum_<ImGuiMouseButton_>(m, "MouseButton")
      .value("Left", ImGuiMouseButton_Left)
      .value("Right", ImGuiMouseButton_Right)
      .value("Middle", ImGuiMouseButton_Middle);

  py::enum_<ImGuiMouseCursor_>(m, "MouseCursor")
      .value("None", ImGuiMouseCursor_None)
      .value("Arrow", ImGuiMouseCursor_Arrow)
      .value("TextInput", ImGuiMouseCursor_TextInput)
      .value("ResizeAll", ImGuiMouseCursor_ResizeAll)
      .value("ResizeNS", ImGuiMouseCursor_ResizeNS)
      .value("ResizeEW", ImGuiMouseCursor_ResizeEW)
      .value("ResizeNESW", ImGuiMouseCursor_ResizeNESW)
      .value("ResizeNWSE", ImGuiMouseCursor_ResizeNWSE)
      .value("Hand", ImGuiMouseCursor_Hand)
      .value("NotAllowed", ImGuiMouseCursor_NotAllowed);

  py::enum_<ImGuiMouseSource>(m, "MouseSource")
      .value("Mouse", ImGuiMouseSource_Mouse)
      .value("TouchScreen", ImGuiMouseSource_TouchScreen)
      .value("Pen", ImGuiMouseSource_Pen);

  py::enum_<ImGuiCond_>(m, "Cond")
      .value("None", ImGuiCond_None)
      .value("Always", ImGuiCond_Always)
      .value("Once", ImGuiCond_Once)
      .value("FirstUseEver", ImGuiCond_FirstUseEver)
      .value("Appearing", ImGuiCond_Appearing);

  py::enum_<ImGuiTableFlags_>(m, "TableFlags")
      .value("None", ImGuiTableFlags_None)
      .value("Resizable", ImGuiTableFlags_Resizable)
      .value("Reorderable", ImGuiTableFlags_Reorderable)
      .value("Hideable", ImGuiTableFlags_Hideable)
      .value("Sortable", ImGuiTableFlags_Sortable)
      .value("NoSavedSettings", ImGuiTableFlags_NoSavedSettings)
      .value("ContextMenuInBody", ImGuiTableFlags_ContextMenuInBody)
      .value("RowBg", ImGuiTableFlags_RowBg)
      .value("BordersInnerH", ImGuiTableFlags_BordersInnerH)
      .value("BordersOuterH", ImGuiTableFlags_BordersOuterH)
      .value("BordersInnerV", ImGuiTableFlags_BordersInnerV)
      .value("BordersOuterV", ImGuiTableFlags_BordersOuterV)
      .value("BordersH", ImGuiTableFlags_BordersH)
      .value("BordersV", ImGuiTableFlags_BordersV)
      .value("BordersInner", ImGuiTableFlags_BordersInner)
      .value("BordersOuter", ImGuiTableFlags_BordersOuter)
      .value("Borders", ImGuiTableFlags_Borders)
      .value("NoBordersInBody", ImGuiTableFlags_NoBordersInBody)
      .value("NoBordersInBodyUntilResize", ImGuiTableFlags_NoBordersInBodyUntilResize)
      .value("SizingFixedFit", ImGuiTableFlags_SizingFixedFit)
      .value("SizingFixedSame", ImGuiTableFlags_SizingFixedSame)
      .value("SizingStretchProp", ImGuiTableFlags_SizingStretchProp)
      .value("SizingStretchSame", ImGuiTableFlags_SizingStretchSame)
      .value("NoHostExtendX", ImGuiTableFlags_NoHostExtendX)
      .value("NoHostExtendY", ImGuiTableFlags_NoHostExtendY)
      .value("NoKeepColumnsVisible", ImGuiTableFlags_NoKeepColumnsVisible)
      .value("PreciseWidths", ImGuiTableFlags_PreciseWidths)
      .value("NoClip", ImGuiTableFlags_NoClip)
      .value("PadOuterX", ImGuiTableFlags_PadOuterX)
      .value("NoPadOuterX", ImGuiTableFlags_NoPadOuterX)
      .value("NoPadInnerX", ImGuiTableFlags_NoPadInnerX)
      .value("ScrollX", ImGuiTableFlags_ScrollX)
      .value("ScrollY", ImGuiTableFlags_ScrollY)
      .value("SortMulti", ImGuiTableFlags_SortMulti)
      .value("SortTristate", ImGuiTableFlags_SortTristate)
      .value("HighlightHoveredColumn", ImGuiTableFlags_HighlightHoveredColumn)
      .value("SizingMask", ImGuiTableFlags_SizingMask_);

  py::enum_<ImGuiTableColumnFlags_>(m, "TableColumnFlags")
      .value("None", ImGuiTableColumnFlags_None)
      .value("Disabled", ImGuiTableColumnFlags_Disabled)
      .value("DefaultHide", ImGuiTableColumnFlags_DefaultHide)
      .value("DefaultSort", ImGuiTableColumnFlags_DefaultSort)
      .value("WidthStretch", ImGuiTableColumnFlags_WidthStretch)
      .value("WidthFixed", ImGuiTableColumnFlags_WidthFixed)
      .value("NoResize", ImGuiTableColumnFlags_NoResize)
      .value("NoReorder", ImGuiTableColumnFlags_NoReorder)
      .value("NoHide", ImGuiTableColumnFlags_NoHide)
      .value("NoClip", ImGuiTableColumnFlags_NoClip)
      .value("NoSort", ImGuiTableColumnFlags_NoSort)
      .value("NoSortAscending", ImGuiTableColumnFlags_NoSortAscending)
      .value("NoSortDescending", ImGuiTableColumnFlags_NoSortDescending)
      .value("NoHeaderLabel", ImGuiTableColumnFlags_NoHeaderLabel)
      .value("NoHeaderWidth", ImGuiTableColumnFlags_NoHeaderWidth)
      .value("PreferSortAscending", ImGuiTableColumnFlags_PreferSortAscending)
      .value("PreferSortDescending", ImGuiTableColumnFlags_PreferSortDescending)
      .value("IndentEnable", ImGuiTableColumnFlags_IndentEnable)
      .value("IndentDisable", ImGuiTableColumnFlags_IndentDisable)
      .value("AngledHeader", ImGuiTableColumnFlags_AngledHeader)
      .value("IsEnabled", ImGuiTableColumnFlags_IsEnabled)
      .value("IsVisible", ImGuiTableColumnFlags_IsVisible)
      .value("IsSorted", ImGuiTableColumnFlags_IsSorted)
      .value("IsHovered", ImGuiTableColumnFlags_IsHovered)
      .value("WidthMask", ImGuiTableColumnFlags_WidthMask_)
      .value("IndentMask", ImGuiTableColumnFlags_IndentMask_)
      .value("StatusMask", ImGuiTableColumnFlags_StatusMask_)
      .value("NoDirectResize", ImGuiTableColumnFlags_NoDirectResize_);

  py::enum_<ImGuiTableRowFlags_>(m, "TableRowFlags")
      .value("None", ImGuiTableRowFlags_None)
      .value("Headers", ImGuiTableRowFlags_Headers);

  py::enum_<ImGuiTableBgTarget_>(m, "TableBgTarget")
      .value("None", ImGuiTableBgTarget_None)
      .value("RowBg0", ImGuiTableBgTarget_RowBg0)
      .value("RowBg1", ImGuiTableBgTarget_RowBg1)
      .value("CellBg", ImGuiTableBgTarget_CellBg);

  py::enum_<ImGuiKey>(m, "Key")
      .value("None", ImGuiKey_None)
      .value("Tab", ImGuiKey_Tab)
      .value("LeftArrow", ImGuiKey_LeftArrow)
      .value("RightArrow", ImGuiKey_RightArrow)
      .value("UpArrow", ImGuiKey_UpArrow)
      .value("DownArrow", ImGuiKey_DownArrow)
      .value("PageUp", ImGuiKey_PageUp)
      .value("PageDown", ImGuiKey_PageDown)
      .value("Home", ImGuiKey_Home)
      .value("End", ImGuiKey_End)
      .value("Insert", ImGuiKey_Insert)
      .value("Delete", ImGuiKey_Delete)
      .value("Backspace", ImGuiKey_Backspace)
      .value("Space", ImGuiKey_Space)
      .value("Enter", ImGuiKey_Enter)
      .value("Escape", ImGuiKey_Escape)
      .value("LeftCtrl", ImGuiKey_LeftCtrl)
      .value("LeftShift", ImGuiKey_LeftShift)
      .value("LeftAlt", ImGuiKey_LeftAlt)
      .value("LeftSuper", ImGuiKey_LeftSuper)
      .value("RightCtrl", ImGuiKey_RightCtrl)
      .value("RightShift", ImGuiKey_RightShift)
      .value("RightAlt", ImGuiKey_RightAlt)
      .value("RightSuper", ImGuiKey_RightSuper)
      .value("Menu", ImGuiKey_Menu)
      // "N" prefix ensures "imgui.Key.N0" parses correctly in Python
      // (numbers are not valid identifier prefixes).
      .value("N0", ImGuiKey_0)
      .value("N1", ImGuiKey_1)
      .value("N2", ImGuiKey_2)
      .value("N3", ImGuiKey_3)
      .value("N4", ImGuiKey_4)
      .value("N5", ImGuiKey_5)
      .value("N6", ImGuiKey_6)
      .value("N7", ImGuiKey_7)
      .value("N8", ImGuiKey_8)
      .value("N9", ImGuiKey_9)
      .value("A", ImGuiKey_A)
      .value("B", ImGuiKey_B)
      .value("C", ImGuiKey_C)
      .value("D", ImGuiKey_D)
      .value("E", ImGuiKey_E)
      .value("F", ImGuiKey_F)
      .value("G", ImGuiKey_G)
      .value("H", ImGuiKey_H)
      .value("I", ImGuiKey_I)
      .value("J", ImGuiKey_J)
      .value("K", ImGuiKey_K)
      .value("L", ImGuiKey_L)
      .value("M", ImGuiKey_M)
      .value("N", ImGuiKey_N)
      .value("O", ImGuiKey_O)
      .value("P", ImGuiKey_P)
      .value("Q", ImGuiKey_Q)
      .value("R", ImGuiKey_R)
      .value("S", ImGuiKey_S)
      .value("T", ImGuiKey_T)
      .value("U", ImGuiKey_U)
      .value("V", ImGuiKey_V)
      .value("W", ImGuiKey_W)
      .value("X", ImGuiKey_X)
      .value("Y", ImGuiKey_Y)
      .value("Z", ImGuiKey_Z)
      .value("F1", ImGuiKey_F1)
      .value("F2", ImGuiKey_F2)
      .value("F3", ImGuiKey_F3)
      .value("F4", ImGuiKey_F4)
      .value("F5", ImGuiKey_F5)
      .value("F6", ImGuiKey_F6)
      .value("F7", ImGuiKey_F7)
      .value("F8", ImGuiKey_F8)
      .value("F9", ImGuiKey_F9)
      .value("F10", ImGuiKey_F10)
      .value("F11", ImGuiKey_F11)
      .value("F12", ImGuiKey_F12)
      .value("F13", ImGuiKey_F13)
      .value("F14", ImGuiKey_F14)
      .value("F15", ImGuiKey_F15)
      .value("F16", ImGuiKey_F16)
      .value("F17", ImGuiKey_F17)
      .value("F18", ImGuiKey_F18)
      .value("F19", ImGuiKey_F19)
      .value("F20", ImGuiKey_F20)
      .value("F21", ImGuiKey_F21)
      .value("F22", ImGuiKey_F22)
      .value("F23", ImGuiKey_F23)
      .value("F24", ImGuiKey_F24)
      .value("Apostrophe", ImGuiKey_Apostrophe)
      .value("Comma", ImGuiKey_Comma)
      .value("Minus", ImGuiKey_Minus)
      .value("Period", ImGuiKey_Period)
      .value("Slash", ImGuiKey_Slash)
      .value("Semicolon", ImGuiKey_Semicolon)
      .value("Equal", ImGuiKey_Equal)
      .value("LeftBracket", ImGuiKey_LeftBracket)
      .value("Backslash", ImGuiKey_Backslash)
      .value("RightBracket", ImGuiKey_RightBracket)
      .value("GraveAccent", ImGuiKey_GraveAccent)
      .value("CapsLock", ImGuiKey_CapsLock)
      .value("ScrollLock", ImGuiKey_ScrollLock)
      .value("NumLock", ImGuiKey_NumLock)
      .value("PrintScreen", ImGuiKey_PrintScreen)
      .value("Pause", ImGuiKey_Pause)
      .value("Keypad0", ImGuiKey_Keypad0)
      .value("Keypad1", ImGuiKey_Keypad1)
      .value("Keypad2", ImGuiKey_Keypad2)
      .value("Keypad3", ImGuiKey_Keypad3)
      .value("Keypad4", ImGuiKey_Keypad4)
      .value("Keypad5", ImGuiKey_Keypad5)
      .value("Keypad6", ImGuiKey_Keypad6)
      .value("Keypad7", ImGuiKey_Keypad7)
      .value("Keypad8", ImGuiKey_Keypad8)
      .value("Keypad9", ImGuiKey_Keypad9)
      .value("KeypadDecimal", ImGuiKey_KeypadDecimal)
      .value("KeypadDivide", ImGuiKey_KeypadDivide)
      .value("KeypadMultiply", ImGuiKey_KeypadMultiply)
      .value("KeypadSubtract", ImGuiKey_KeypadSubtract)
      .value("KeypadAdd", ImGuiKey_KeypadAdd)
      .value("KeypadEnter", ImGuiKey_KeypadEnter)
      .value("KeypadEqual", ImGuiKey_KeypadEqual)
      .value("AppBack", ImGuiKey_AppBack)
      .value("AppForward", ImGuiKey_AppForward)
      .value("GamepadStart", ImGuiKey_GamepadStart)
      .value("GamepadBack", ImGuiKey_GamepadBack)
      .value("GamepadFaceLeft", ImGuiKey_GamepadFaceLeft)
      .value("GamepadFaceRight", ImGuiKey_GamepadFaceRight)
      .value("GamepadFaceUp", ImGuiKey_GamepadFaceUp)
      .value("GamepadFaceDown", ImGuiKey_GamepadFaceDown)
      .value("GamepadDpadLeft", ImGuiKey_GamepadDpadLeft)
      .value("GamepadDpadRight", ImGuiKey_GamepadDpadRight)
      .value("GamepadDpadUp", ImGuiKey_GamepadDpadUp)
      .value("GamepadDpadDown", ImGuiKey_GamepadDpadDown)
      .value("GamepadL1", ImGuiKey_GamepadL1)
      .value("GamepadR1", ImGuiKey_GamepadR1)
      .value("GamepadL2", ImGuiKey_GamepadL2)
      .value("GamepadR2", ImGuiKey_GamepadR2)
      .value("GamepadL3", ImGuiKey_GamepadL3)
      .value("GamepadR3", ImGuiKey_GamepadR3)
      .value("GamepadLStickLeft", ImGuiKey_GamepadLStickLeft)
      .value("GamepadLStickRight", ImGuiKey_GamepadLStickRight)
      .value("GamepadLStickUp", ImGuiKey_GamepadLStickUp)
      .value("GamepadLStickDown", ImGuiKey_GamepadLStickDown)
      .value("GamepadRStickLeft", ImGuiKey_GamepadRStickLeft)
      .value("GamepadRStickRight", ImGuiKey_GamepadRStickRight)
      .value("GamepadRStickUp", ImGuiKey_GamepadRStickUp)
      .value("GamepadRStickDown", ImGuiKey_GamepadRStickDown)
      .value("MouseLeft", ImGuiKey_MouseLeft)
      .value("MouseRight", ImGuiKey_MouseRight)
      .value("MouseMiddle", ImGuiKey_MouseMiddle)
      .value("MouseX1", ImGuiKey_MouseX1)
      .value("MouseX2", ImGuiKey_MouseX2)
      .value("MouseWheelX", ImGuiKey_MouseWheelX)
      .value("MouseWheelY", ImGuiKey_MouseWheelY)
      .value("ReservedForModCtrl", ImGuiKey_ReservedForModCtrl)
      .value("ReservedForModShift", ImGuiKey_ReservedForModShift)
      .value("ReservedForModAlt", ImGuiKey_ReservedForModAlt)
      .value("ReservedForModSuper", ImGuiKey_ReservedForModSuper)
      .value("Ctrl", ImGuiMod_Ctrl)
      .value("Shift", ImGuiMod_Shift)
      .value("Alt", ImGuiMod_Alt)
      .value("Super", ImGuiMod_Super)
      .value("Mask", ImGuiMod_Mask_);

  // Functions.
  //
  // Most functions can be bound directly with no custom implementation. A few
  // overloaded functions are bound with unique names to prevent ambiguities
  // on the python side. Which function is given an alternative name is more
  // or less arbitrary.
  //
  // Functions that take pointers require custom implementations. Pointers are
  // usually used as either in/out parameters or lists. For in/out parameters,
  // we return a tuple containing the return value of the function and the
  // updated value stored in the pointer. For lists, we accept a std::vector
  // of the corresponding type and pass the raw array into the ImGui functions.
  //
  // Any printf-like function (that took a format + va_list) instead simply
  // takes a string argument, on the assumption that the formatting will be done
  // in python.
  //
  // Note: this list of functions isn't guaranteed to be complete. We ignore all
  // rendering specific functions as well as debug functions and anything else
  // that we feel isn't worth the effort to support.

  DEF0_F(ShowDemoWindow, {
    bool open = true;
    ImGui::ShowDemoWindow(&open);
    return open;
  });

  DEF1(StyleColorsDark, (ImGuiStyle*, dst, = nullptr));
  DEF1(StyleColorsLight, (ImGuiStyle*, dst, = nullptr));
  DEF1(StyleColorsClassic, (ImGuiStyle*, dst, = nullptr));

  DEF3_F(Begin, (ImString, name, ), (bool*, p_open, = nullptr), (ImGuiWindowFlags, flags, = 0), {
    const auto result = ImGui::Begin(name, p_open, flags);
    return std::make_tuple(result, *p_open);
  });
  DEF0(End);
  DEF4(BeginChild, (ImString, str_id, ), (const ImVec2&, size, = ImVec2_Zero), (ImGuiChildFlags, child_flags, = 0), (ImGuiWindowFlags, window_flags, = 0));
  DEF4_AS(BeginChild, BeginChildId, (ImGuiID, id, ), (const ImVec2&, size, = ImVec2_Zero), (ImGuiChildFlags, child_flags, = 0), (ImGuiWindowFlags, window_flags, = 0));
  DEF0(EndChild);
  DEF0(IsWindowAppearing);
  DEF0(IsWindowCollapsed);
  DEF1(IsWindowFocused, (ImGuiFocusedFlags, flags, = 0));
  DEF1(IsWindowHovered, (ImGuiHoveredFlags, flags, = 0));
  DEF0(GetWindowDpiScale);
  DEF0(GetWindowPos);
  DEF0(GetWindowSize);
  DEF0(GetWindowWidth);
  DEF0(GetWindowHeight);
  DEF3(SetNextWindowPos, (const ImVec2&, pos, ), (ImGuiCond, cond, = 0), (const ImVec2&, pivot, = ImVec2_Zero));
  DEF2(SetNextWindowSize, (const ImVec2&, size, ), (ImGuiCond, cond, = 0));
  DEF2_F(SetNextWindowSizeConstraints, (const ImVec2&, size_min, ), (const ImVec2&, size_max, ), {
    ImGui::SetNextWindowSizeConstraints(size_min, size_max);
  });
  DEF1(SetNextWindowContentSize, (const ImVec2&, size, ));
  DEF2(SetNextWindowCollapsed, (bool, collapsed, ), (ImGuiCond, cond, = 0));
  DEF0(SetNextWindowFocus);
  DEF1(SetNextWindowScroll, (const ImVec2&, scroll, ));
  DEF1(SetNextWindowBgAlpha, (float, alpha, ));
  DEF1(SetNextWindowViewport, (ImGuiID, viewport_id, ));
  DEF2(SetWindowPos, (const ImVec2&, pos, ), (ImGuiCond, cond, = 0));
  DEF2(SetWindowSize, (const ImVec2&, size, ), (ImGuiCond, cond, = 0));
  DEF2(SetWindowCollapsed, (bool, collapsed, ), (ImGuiCond, cond, = 0));
  DEF0(SetWindowFocus);
  DEF1(SetWindowFontScale, (float, scale, ));
  DEF3(SetWindowPos, (ImString, name, ), (const ImVec2&, pos, ), (ImGuiCond, cond, = 0));
  DEF3(SetWindowSize, (ImString, name, ), (const ImVec2&, size, ), (ImGuiCond, cond, = 0));
  DEF3(SetWindowCollapsed, (ImString, name, ), (bool, collapsed, ), (ImGuiCond, cond, = 0));
  DEF1(SetWindowFocus, (ImString, name, ));
  DEF0(GetContentRegionAvail);
  DEF0(GetContentRegionMax);
  DEF0(GetWindowContentRegionMin);
  DEF0(GetWindowContentRegionMax);
  DEF0(GetScrollX);
  DEF0(GetScrollY);
  DEF1(SetScrollX, (float, scroll_x, ));
  DEF1(SetScrollY, (float, scroll_y, ));
  DEF0(GetScrollMaxX);
  DEF0(GetScrollMaxY);
  DEF1(SetScrollHereX, (float, center_x_ratio, = 0.5f));
  DEF1(SetScrollHereY, (float, center_y_ratio, = 0.5f));
  DEF2(SetScrollFromPosX, (float, local_x, ), (float, center_x_ratio, = 0.5f));
  DEF2(SetScrollFromPosY, (float, local_y, ), (float, center_y_ratio, = 0.5f));
  DEF2(PushFont, (ImFont*, font, ), (float, font_size_base_unscaled, = 0.0f));
  DEF0(PopFont);
  DEF2(PushStyleColor, (ImGuiCol, idx, ), (ImU32, col, ));
  DEF2(PushStyleColor, (ImGuiCol, idx, ), (const ImVec4&, col, ));
  DEF1(PopStyleColor, (int, count, = 1));
  DEF2(PushStyleVar, (ImGuiStyleVar, idx, ), (float, val, ));
  DEF2(PushStyleVar, (ImGuiStyleVar, idx, ), (const ImVec2&, val, ));
  DEF1(PopStyleVar, (int, count, = 1));
  DEF1(PushTabStop, (bool, tab_stop, ));
  DEF0(PopTabStop);
  DEF1(PushButtonRepeat, (bool, repeat, ));
  DEF0(PopButtonRepeat);
  DEF1(PushItemWidth, (float, item_width, ));
  DEF0(PopItemWidth);
  DEF1(SetNextItemWidth, (float, item_width, ));
  DEF0(CalcItemWidth);
  DEF1(PushTextWrapPos, (float, wrap_local_pos_x, = 0.0f));
  DEF0(PopTextWrapPos);
  DEF0(GetFontSize);
  DEF0(GetFontTexUvWhitePixel);
  DEF2(GetColorU32, (ImGuiCol, idx, ), (float, alpha_mul, = 1.0f));
  DEF1(GetColorU32, (const ImVec4&, col, ));
  DEF2(GetColorU32, (ImU32, col, ), (float, alpha_mul, = 1.0f));
  DEF1(GetStyleColorVec4, (ImGuiCol, idx, ));
  DEF0(GetCursorScreenPos);
  DEF1(SetCursorScreenPos, (const ImVec2&, pos, ));
  DEF0(GetCursorPos);
  DEF0(GetCursorPosX);
  DEF0(GetCursorPosY);
  DEF1(SetCursorPos, (const ImVec2&, local_pos, ));
  DEF1(SetCursorPosX, (float, local_x, ));
  DEF1(SetCursorPosY, (float, local_y, ));
  DEF0(GetCursorStartPos);
  DEF0(Separator);
  DEF2(SameLine, (float, offset_from_start_x, = 0.0f), (float, spacing, = -1.0f));
  DEF0(NewLine);
  DEF0(Spacing);
  DEF1(Dummy, (const ImVec2&, size, ));
  DEF1(Indent, (float, indent_w, = 0.0f));
  DEF1(Unindent, (float, indent_w, = 0.0f));
  DEF0(BeginGroup);
  DEF0(EndGroup);
  DEF0(AlignTextToFramePadding);
  DEF0(GetTextLineHeight);
  DEF0(GetTextLineHeightWithSpacing);
  DEF0(GetFrameHeight);
  DEF0(GetFrameHeightWithSpacing);
  DEF1(PushID, (ImString, str_id, ));
  DEF2(PushID, (ImString, str_id_begin, ), (ImString, str_id_end, ));
  DEF1(PushID, (int, int_id, ));
  DEF0(PopID);
  DEF1(GetID, (ImString, str_id, ));
  DEF2(GetID, (ImString, str_id_begin, ), (ImString, str_id_end, ));
  DEF2_F(DockSpaceOverMainViewport, (ImGuiID, id, = 0), (ImGuiDockNodeFlags, flags, = 0), {
    return ImGui::DockSpaceOverViewport(id, nullptr, flags);
  });
  DEF3_F(DockSpace, (ImGuiID, id, ), (const ImVec2&, size, = ImVec2_Zero), (ImGuiDockNodeFlags, flags, = 0), {
    return ImGui::DockSpace(id, size, flags);
  });
  DEF1_F(DockBuilderRemoveNode, (ImGuiID, node_id, ), {
    ImGui::DockBuilderRemoveNode(node_id);
  });
  DEF2_F(DockBuilderAddNode, (ImGuiID, node_id, = 0), (ImGuiDockNodeFlags, flags, = 0), {
    return ImGui::DockBuilderAddNode(node_id, flags);
  });
  DEF2_F(DockBuilderSetNodeSize, (ImGuiID, node_id, ), (const ImVec2&, size, ), {
    ImGui::DockBuilderSetNodeSize(node_id, size);
  });
  DEF3_F(DockBuilderSplitNode, (ImGuiID, id, ), (ImGuiDir, split_dir, ), (float, size_ratio_for_node_at_dir, ), {
    ImGuiID id_at_dir = 0;
    ImGuiID id_at_opp = 0;
    ImGui::DockBuilderSplitNode(id, split_dir, size_ratio_for_node_at_dir, &id_at_dir, &id_at_opp);
    return std::make_tuple(id_at_dir, id_at_opp);
  });
  DEF2_F(DockBuilderDockWindow, (ImString, window_name, ), (ImGuiID, node_id, ), {
    ImGui::DockBuilderDockWindow(window_name, node_id);
  });
  DEF1_F(DockBuilderFinish, (ImGuiID, node_id, ), {
    ImGui::DockBuilderFinish(node_id);
  });
  DEF1_F(DockBuilderGetNode, (ImGuiID, node_id, ), {
    return ImGui::DockBuilderGetNode(node_id) != nullptr;
  });
  DEF1_F(TextUnformatted, (ImString, txt, ), {
    ImGui::TextUnformatted(txt);
  });
  DEF1_F(Text, (ImString, txt, ), {
    return ImGui::Text("%s", txt);
  });
  DEF2_F(TextColored, (const ImVec4&, col, ), (ImString, txt, ), {
    return ImGui::TextColored(col, "%s", txt);
  });
  DEF1_F(TextDisabled, (ImString, txt, ), {
    return ImGui::TextDisabled("%s", txt);
  });
  DEF1_F(TextWrapped, (ImString, txt, ), {
    return ImGui::TextWrapped("%s", txt);
  });
  DEF2_F(LabelText, (ImString, label, ), (ImString, txt, ), {
    return ImGui::LabelText(label, "%s", txt);
  });
  DEF1_F(BulletText, (ImString, txt, ), {
    return ImGui::BulletText("%s", txt);
  });
  DEF1(SeparatorText, (ImString, label, ));
  DEF2(Button, (ImString, label, ), (const ImVec2&, size, = ImVec2_Zero));
  DEF1(SmallButton, (ImString, label, ));
  DEF3(InvisibleButton, (ImString, str_id, ), (const ImVec2&, size, ), (ImGuiButtonFlags, flags, = 0));
  DEF2(ArrowButton, (ImString, str_id, ), (ImGuiDir, dir, ));
  DEF2_F(Checkbox, (ImString, label, ), (bool*, v, ), {
    auto result = ImGui::Checkbox(label, v);
    return std::make_tuple(result, *v);
  });
  DEF2(RadioButton, (ImString, label, ), (bool, active, ));
  DEF3(ProgressBar, (float, fraction, ), (const ImVec2&, size_arg, = ImVec2_Min_Zero), (ImString, overlay, = nullptr));
  DEF0(Bullet);
  DEF1(TextLink, (ImString, label, ));
  DEF2(TextLinkOpenURL, (ImString, label, ), (ImString, url, = nullptr));
  DEF6_F(Image, (long, user_texture_id, ), (const ImVec2&, image_size, ), (const ImVec2&, uv0, = ImVec2_Zero), (const ImVec2&, uv1, = ImVec2_One), (const ImVec4&, tint_col, = ImVec4_One), (const ImVec4&, border_col, = ImVec4_Zero), {
    return ImGui::Image(reinterpret_cast<void*>(user_texture_id), image_size, uv0, uv1, tint_col, border_col);
  });

  DEF7_F(ImageButton, (ImString, str_id, ), (long, user_texture_id, ), (const ImVec2&, image_size, ), (const ImVec2&, uv0, = ImVec2_Zero), (const ImVec2&, uv1, = ImVec2_One), (const ImVec4&, bg_col, = ImVec4_Zero), (const ImVec4&, tint_col, = ImVec4_One), {
    return ImGui::ImageButton(str_id, reinterpret_cast<void*>(user_texture_id), image_size, uv0, uv1, bg_col, tint_col);
  });
  DEF3(BeginCombo, (ImString, label, ), (ImString, preview_value, ), (ImGuiComboFlags, flags, = 0));
  DEF0(EndCombo);
  m.def(
      "Combo",
      [](const char* label, int current_item, std::vector<std::string> items,
         int popup_max_height_in_items) {
        std::vector<const char*> items_ptr;
        items_ptr.reserve(items.size());
        for (const auto& item : items) {
          items_ptr.push_back(item.c_str());
        }
        const auto result =
            ImGui::Combo(label, &current_item, items_ptr.data(),
                         items_ptr.size(), popup_max_height_in_items);
        return std::make_tuple(result, current_item);
      },
      py::arg("label"), py::arg("current_item"), py::arg("items"),
      py::arg("popup_max_height_in_items") = -1);
  m.def(
      "ComboStr",
      [](const char* label, int current_item,
         const char* items_separated_by_zeros, int popup_max_height_in_items) {
        const auto result =
            ImGui::Combo(label, &current_item, items_separated_by_zeros,
                         popup_max_height_in_items);
        return std::make_tuple(result, current_item);
      },
      py::arg("label"), py::arg("current_item"),
      py::arg("items_separated_by_zeros"),
      py::arg("popup_max_height_in_items") = -1);
  DEF7_F(DragFloat, (ImString, label, ), (float*, v,   ), (float, v_speed, = 1.0f), (float, v_min, = 0.0f), (float, v_max, = 0.0f), (ImString, format, = "%.3f"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::DragFloat(label, v, v_speed, v_min, v_max, format, flags);
    return std::make_tuple(result, *v);
  });
  DEF7_F(DragFloatN, (ImString, label, ), (std::vector<float>, v, ), (float, v_speed, = 1.0f), (float, v_min, = 0.0f), (float, v_max, = 0.0f), (ImString, format, = "%.3f"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::DragScalarN(label, ImGuiDataType_Float, v.data(), v.size(), v_speed, &v_min, &v_max, format, flags);
    return std::make_tuple(result, v);
  });
  DEF7_F(DragInt, (ImString, label, ), (int*, v, ), (float, v_speed, = 1.0f), (int, v_min, = 0), (int, v_max, = 0), (ImString, format, = "%d"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::DragInt(label, v, v_speed, v_min, v_max, format, flags);
    return std::make_tuple(result, *v);
  });
  DEF7_F(DragIntN, (ImString, label, ), (std::vector<int>, v, ), (float, v_speed, = 1.0f), (int, v_min, = 0), (int, v_max, = 0), (ImString, format, = "%d"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::DragScalarN(label, ImGuiDataType_S32, v.data(), v.size(), v_speed, &v_min, &v_max, format, flags);
    return std::make_tuple(result, v);
  });
  DEF6_F(SliderFloat, (ImString, label, ), (float*, v, ), (float, v_min, ), (float, v_max, ), (ImString, format, = "%.3f"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
    return std::make_tuple(result, *v);
  });
  DEF6_F(SliderFloatN, (ImString, label, ), (std::vector<float>, v, ), (float, v_min, ), (float, v_max, ), (ImString, format, = "%.3f"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::SliderScalarN(label, ImGuiDataType_Float, v.data(), v.size(), &v_min, &v_max, format, flags);
    return std::make_tuple(result, v);
  });
  DEF6_F(SliderAngle, (ImString, label, ), (float*, v_rad, ), (float, v_degrees_min, = -360.0f), (float, v_degrees_max, = +360.0f), (ImString, format, = "%.0f deg"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::SliderAngle(label, v_rad, v_degrees_min, v_degrees_max, format, flags);
    return std::make_tuple(result, *v_rad);
  });
  DEF6_F(SliderInt, (ImString, label, ), (int*, v, ), (int, v_min, ), (int, v_max, ), (ImString, format, = "%d"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::SliderInt(label, v, v_min, v_max, format, flags);
    return std::make_tuple(result, *v);
  });
  DEF6_F(SliderIntN, (ImString, label, ), (std::vector<int>, v, ), (int, v_min, ), (int, v_max, ), (ImString, format, = "%d"), (ImGuiSliderFlags, flags, = 0), {
    const auto result = ImGui::SliderScalarN(label, ImGuiDataType_S32, v.data(), v.size(), &v_min, &v_max, format, flags);
    return std::make_tuple(result, v);
  });
  DEF6_F(InputFloat, (ImString, label, ), (float*, v, ), (float, step, = 0.0f), (float, step_fast, = 0.0f), (ImString, format, = "%.3f"), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputFloat(label, v, step, step_fast, format, flags);
    return std::make_tuple(result, *v);
  });
  DEF4_F(InputFloatN, (ImString, label, ), (std::vector<float>, v, ), (ImString, format, = "%.3f"), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputScalarN(label, ImGuiDataType_Float, v.data(), v.size(), NULL, NULL, format, flags);
    return std::make_tuple(result, v);
  });
  DEF5_F(InputInt, (ImString, label, ), (int*, v, ), (int, step, = 1), (int, step_fast, = 100), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputInt(label, v, step, step_fast, flags);
    return std::make_tuple(result, *v);
  });
  DEF3_F(InputIntN, (ImString, label, ), (std::vector<int>, v, ), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputScalarN(label, ImGuiDataType_S32, v.data(), v.size(), NULL, NULL, "%d", flags);
    return std::make_tuple(result, v);
  });
  DEF6_F(InputDouble, (ImString, label, ), (double*, v, ), (double, step, = 0.0), (double, step_fast, = 0.0), (ImString, format, = "%.6f"), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputDouble(label, v, step, step_fast, format, flags);
    return std::make_tuple(result, *v);
  });
  DEF3_F(InputText, (ImString, label, ), (std::string, text, ), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputText(label, &text, flags);
    return std::make_tuple(result, text);
  });
  DEF4_F(InputTextMultiline, (ImString, label, ), (std::string, text, ), (const ImVec2&, size, = ImVec2_Zero), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputTextMultiline(label, &text, size, flags);
    return std::make_tuple(result, text);
  });
  DEF4_F(InputTextWithHint, (ImString, label, ), (ImString, hint, ), (std::string, text, ), (ImGuiInputTextFlags, flags, = 0), {
    const auto result = ImGui::InputTextWithHint(label, hint, &text, flags);
    return std::make_tuple(result, text);
  });
  DEF3_F(ColorEdit3, (ImString, label, ), (std::vector<float>, col, ), (ImGuiColorEditFlags, flags, = 0), {
    const auto result = ImGui::ColorEdit3(label, col.data(), flags);
    return std::make_tuple(result, col);
  });
  DEF3_F(ColorEdit4, (ImString, label, ), (std::vector<float>, col, ), (ImGuiColorEditFlags, flags, = 0), {
    const auto result = ImGui::ColorEdit4(label, col.data(), flags);
    return std::make_tuple(result, col);
  });
  DEF3_F(ColorPicker3, (ImString, label, ), (std::vector<float>, col, ), (ImGuiColorEditFlags, flags, = 0), {
    const auto result = ImGui::ColorPicker3(label, col.data(), flags);
    return std::make_tuple(result, col);
  });
  DEF4_F(ColorPicker4, (ImString, label, ), (std::vector<float>, col, ), (ImGuiColorEditFlags, flags, = 0), (const float*, ref_col, = nullptr), {
    const auto result = ImGui::ColorPicker4(label, col.data(), flags, ref_col);
    return std::make_tuple(result, col);
  });
  DEF4(ColorButton, (ImString, desc_id, ), (const ImVec4&, col, ), (ImGuiColorEditFlags, flags, = 0), (const ImVec2&, size, = ImVec2_Zero));
  DEF1(SetColorEditOptions, (ImGuiColorEditFlags, flags, ));
  DEF1(TreeNode, (ImString, label, ));
  DEF2_F(TreeNode, (ImString, str_id, ), (ImString, txt, ), {
    return ImGui::TreeNode(str_id, "%s", txt);
  });
  DEF2(TreeNodeEx, (ImString, label, ), (ImGuiTreeNodeFlags, flags, = 0));
  DEF3_F(TreeNodeEx, (ImString, str_id, ), (ImGuiTreeNodeFlags, flags, ), (ImString, txt, ), {
    return ImGui::TreeNodeEx(str_id, flags, "%s", txt);
  });
  DEF1(TreePush, (ImString, str_id, ));
  DEF0(TreePop);
  DEF0(GetTreeNodeToLabelSpacing);
  DEF2(CollapsingHeader, (ImString, label, ), (ImGuiTreeNodeFlags, flags, = 0));
  DEF3_F(CollapsingHeader2, (ImString, label, ), (bool*, p_visible, ), (ImGuiTreeNodeFlags, flags, = 0), {
    const auto result = ImGui::CollapsingHeader(label, p_visible, flags);
    return std::make_tuple(result, *p_visible);
  });
  DEF2(SetNextItemOpen, (bool, is_open, ), (ImGuiCond, cond, = 0));
  DEF4(Selectable, (ImString, label, ), (bool, selected, = false), (ImGuiSelectableFlags, flags, = 0), (const ImVec2&, size, = ImVec2_Zero));
  DEF4_F(Selectable2, (ImString, label, ), (bool*, p_selected, ), (ImGuiSelectableFlags, flags, = 0), (const ImVec2&, size, = ImVec2_Zero), {
    const auto result = ImGui::Selectable(label, p_selected, flags, size);
    return std::make_tuple(result, *p_selected);
  });
  DEF2(BeginListBox, (ImString, label, ), (const ImVec2&, size, = ImVec2_Zero));
  DEF0(EndListBox);
  DEF2(Value, (ImString, prefix, ), (bool, b, ));
  DEF2(Value, (ImString, prefix, ), (int, v, ));
  DEF2(Value, (ImString, prefix, ), (unsigned int, v, ));
  DEF3(Value, (ImString, prefix, ), (float, v, ), (ImString, float_format, = nullptr));
  DEF0(BeginMenuBar);
  DEF0(EndMenuBar);
  DEF0(BeginMainMenuBar);
  DEF0(EndMainMenuBar);
  DEF2(BeginMenu, (ImString, label, ), (bool, enabled, = true));
  DEF0(EndMenu);
  DEF4(MenuItem, (ImString, label, ), (ImString, shortcut, = nullptr), (bool, selected, = false), (bool, enabled, = true));
  DEF4_F(MenuItem, (ImString, label, ), (ImString, shortcut, ), (bool*, p_selected, ), (bool, enabled, = true), {
    const auto result = ImGui::MenuItem(label, shortcut, p_selected, enabled);
    return std::make_tuple(result, *p_selected);
  });
  DEF0(BeginTooltip);
  DEF0(EndTooltip);
  DEF1_F(SetTooltip, (ImString, txt, ), {
    return ImGui::SetTooltip("%s", txt);
  });
  DEF0(BeginItemTooltip);
  DEF1_F(SetItemTooltip, (ImString, txt, ), {
    return ImGui::SetItemTooltip("%s", txt);
  });
  DEF2(BeginPopup, (ImString, str_id, ), (ImGuiWindowFlags, flags, = 0));
  DEF3_F(BeginPopupModal, (ImString, name, ), (bool*, p_open, = nullptr), (ImGuiWindowFlags, flags, = 0), {
    const auto result = ImGui::BeginPopupModal(name, p_open, flags);
    return std::make_tuple(result, *p_open);
  });
  DEF0(EndPopup);
  DEF2(OpenPopup, (ImString, str_id, ), (ImGuiPopupFlags, popup_flags, = 0));
  DEF2(OpenPopup, (ImGuiID, id, ), (ImGuiPopupFlags, popup_flags, = 0));
  DEF2(OpenPopupOnItemClick, (ImString, str_id, = nullptr), (ImGuiPopupFlags, popup_flags, = 0));
  DEF0(CloseCurrentPopup);
  DEF2(BeginPopupContextItem, (ImString, str_id, = nullptr), (ImGuiPopupFlags, popup_flags, = 0));
  DEF2(BeginPopupContextWindow, (ImString, str_id, = nullptr), (ImGuiPopupFlags, popup_flags, = 0));
  DEF2(BeginPopupContextVoid, (ImString, str_id, = nullptr), (ImGuiPopupFlags, popup_flags, = 0));
  DEF2(IsPopupOpen, (ImString, str_id, ), (ImGuiPopupFlags, flags, = 0));
  DEF5(BeginTable, (ImString, str_id, ), (int, columns, ), (ImGuiTableFlags, flags, = 0), (const ImVec2&, outer_size, = ImVec2_Zero), (float, inner_width, = 0.0f));
  DEF0(EndTable);
  DEF2(TableNextRow, (ImGuiTableRowFlags, row_flags, = 0), (float, min_row_height, = 0.0f));
  DEF0(TableNextColumn);
  DEF1(TableSetColumnIndex, (int, column_n, ));
  DEF4(TableSetupColumn, (ImString, label, ), (ImGuiTableColumnFlags, flags, = 0), (float, init_width_or_weight, = 0.0f), (ImGuiID, user_id, = 0));
  DEF2(TableSetupScrollFreeze, (int, cols, ), (int, rows, ));
  DEF1(TableHeader, (ImString, label, ));
  DEF0(TableHeadersRow);
  DEF0(TableAngledHeadersRow);
  DEF0(TableGetColumnCount);
  DEF0(TableGetColumnIndex);
  DEF0(TableGetRowIndex);
  DEF1(TableGetColumnName, (int, column_n, = -1));
  DEF1(TableGetColumnFlags, (int, column_n, = -1));
  DEF2(TableSetColumnEnabled, (int, column_n, ), (bool, v, ));
  DEF0(TableGetHoveredColumn);
  DEF3(TableSetBgColor, (ImGuiTableBgTarget, target, ), (ImU32, color, ), (int, column_n, = -1));
  DEF3(Columns, (int, count, = 1), (ImString, id, = nullptr), (bool, border, = true));
  DEF0(NextColumn);
  DEF0(GetColumnIndex);
  DEF1(GetColumnWidth, (int, column_index, = -1));
  DEF2(SetColumnWidth, (int, column_index, ), (float, width, ));
  DEF1(GetColumnOffset, (int, column_index, = -1));
  DEF2(SetColumnOffset, (int, column_index, ), (float, offset_x, ));
  DEF0(GetColumnsCount);
  DEF2(BeginTabBar, (ImString, str_id, ), (ImGuiTabBarFlags, flags, = 0));
  DEF0(EndTabBar);
  DEF3_F(BeginTabItem, (ImString, label, ), (bool*, p_open, ), (ImGuiTabItemFlags, flags, = 0), {
    const auto result = ImGui::BeginTabItem(label, p_open, flags);
    return std::make_tuple(result, p_open ? *p_open : true);
  });
  // Convenience overload for non-closable tab items.
  // This function returns a boolean so you can call it directly in an if condition.
  DEF2_F(BeginTabItem, (ImString, label, ), (ImGuiTabItemFlags, flags, = 0), {
    return ImGui::BeginTabItem(label, nullptr, flags);
  });
  DEF0(EndTabItem);
  DEF2(TabItemButton, (ImString, label, ), (ImGuiTabItemFlags, flags, = 0));
  DEF1(SetTabItemClosed, (ImString, tab_or_docked_window_label, ));
  DEF2(SetNextWindowDockID, (ImGuiID, dock_id, ), (ImGuiCond, cond, = 0));
  DEF0(GetWindowDockID);
  DEF0(IsWindowDocked);
  DEF1(BeginDisabled, (bool, disabled, = true));
  DEF0(EndDisabled);
  DEF3(PushClipRect, (const ImVec2&, clip_rect_min, ), (const ImVec2&, clip_rect_max, ), (bool, intersect_with_current_clip_rect, ));
  DEF0(PopClipRect);
  DEF0(SetItemDefaultFocus);
  DEF1(SetKeyboardFocusHere, (int, offset, = 0));
  DEF0(SetNextItemAllowOverlap);
  DEF1(IsItemHovered, (ImGuiHoveredFlags, flags, = 0));
  DEF0(IsItemActive);
  DEF0(IsItemFocused);
  DEF1(IsItemClicked, (ImGuiMouseButton, mouse_button, = 0));
  DEF0(IsItemVisible);
  DEF0(IsItemEdited);
  DEF0(IsItemActivated);
  DEF0(IsItemDeactivated);
  DEF0(IsItemDeactivatedAfterEdit);
  DEF0(IsItemToggledOpen);
  DEF0(IsAnyItemHovered);
  DEF0(IsAnyItemActive);
  DEF0(IsAnyItemFocused);
  DEF0(GetItemID);
  DEF0(GetItemRectMin);
  DEF0(GetItemRectMax);
  DEF0(GetItemRectSize);
  DEF1(IsRectVisible, (const ImVec2&, size, ));
  DEF2(IsRectVisible, (const ImVec2&, rect_min, ), (const ImVec2&, rect_max, ));
  DEF0(GetTime);
  DEF0(GetFrameCount);
  DEF4(CalcTextSize, (ImString, text, ), (ImString, text_end, = nullptr), (bool, hide_text_after_double_hash, = false), (float, wrap_width, = -1.0f));
  DEF1(ColorConvertU32ToFloat4, (ImU32, in, ));
  DEF1(ColorConvertFloat4ToU32, (const ImVec4&, in, ));
  DEF6(ColorConvertRGBtoHSV, (float, r, ), (float, g, ), (float, b, ), (float&, out_h, ), (float&, out_s, ), (float&, out_v, ));
  DEF6(ColorConvertHSVtoRGB, (float, h, ), (float, s, ), (float, v, ), (float&, out_r, ), (float&, out_g, ), (float&, out_b, ));
  DEF1(IsKeyDown, (ImGuiKey, key, ));
  DEF2(IsKeyPressed, (ImGuiKey, key, ), (bool, repeat, = true));
  DEF1(IsKeyReleased, (ImGuiKey, key, ));
  DEF1(IsKeyChordPressed, (ImGuiKeyChord, key_chord, ));
  DEF3(GetKeyPressedAmount, (ImGuiKey, key, ), (float, repeat_delay, ), (float, rate, ));
  DEF1(GetKeyName, (ImGuiKey, key, ));
  DEF1(SetNextFrameWantCaptureKeyboard, (bool, want_capture_keyboard, ));
  DEF2(Shortcut, (ImGuiKeyChord, key_chord, ), (ImGuiInputFlags, flags, = 0));
  DEF2(SetNextItemShortcut, (ImGuiKeyChord, key_chord, ), (ImGuiInputFlags, flags, = 0));
  DEF1(IsMouseDown, (ImGuiMouseButton, button, ));
  DEF2(IsMouseClicked, (ImGuiMouseButton, button, ), (bool, repeat, = false));
  DEF1(IsMouseReleased, (ImGuiMouseButton, button, ));
  DEF1(IsMouseDoubleClicked, (ImGuiMouseButton, button, ));
  DEF1(GetMouseClickedCount, (ImGuiMouseButton, button, ));
  DEF3(IsMouseHoveringRect, (const ImVec2&, r_min, ), (const ImVec2&, r_max, ), (bool, clip, = true));
  DEF1(IsMousePosValid, (const ImVec2*, mouse_pos, = nullptr));
  DEF0(IsAnyMouseDown);
  DEF0(GetMousePos);
  DEF0(GetMousePosOnOpeningCurrentPopup);
  DEF2(IsMouseDragging, (ImGuiMouseButton, button, ), (float, lock_threshold, = -1.0f));
  DEF2(GetMouseDragDelta, (ImGuiMouseButton, button, = 0), (float, lock_threshold, = -1.0f));
  DEF1(ResetMouseDragDelta, (ImGuiMouseButton, button, = 0));
  DEF0(GetMouseCursor);
  DEF1(SetMouseCursor, (ImGuiMouseCursor, cursor_type, ));
  DEF1(SetNextFrameWantCaptureMouse, (bool, want_capture_mouse, ));
  DEF0(GetClipboardText);
  DEF1(SetClipboardText, (ImString, text, ));
}

// NOLINTEND(whitespace/line_length)
